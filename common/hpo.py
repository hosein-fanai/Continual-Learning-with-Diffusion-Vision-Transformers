"""Optuna studies that reuse the project's config-driven training pipeline.

Searches deliberately use a few validated architecture templates instead of
sampling arbitrary routing dictionaries. Every trial writes a YAML config,
reloads it with :func:`common.config.load_config`, and runs
:func:`common.train.main`.
"""

from __future__ import annotations

import tensorflow as tf

import pandas as pd

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import gc

import re

from collections.abc import Mapping, Sequence
from typing import Any

from common.config import Config, load_config, save_config
from common.dataloader import get_dataset_spec
from common.train import main


_DIFFUSION_MODELS = {
    "diffusion_transformer", "dit_classifier", "dit_decoder", 
    "dit_encoder_decoder", "dit_encoder_decoder_classifier", 
    "unet", "unet_classifier"
}
_DIFFUSION_CLASSIFIER_MODELS = {
    "dit_classifier", "dit_encoder_decoder_classifier", 
    "unet_classifier"
}

_OPTIMIZATION = {
    "batch_size": "categorical; architecture-appropriate powers of two", 
    "learning_rate": "log-uniform; model-family-specific bounds", 
    "optimizer": "Adam or AdamW", 
    "weight_decay": "log-uniform when AdamW is selected"
}
_DIT = {
    **_OPTIMIZATION, 
    "patch_size": "valid image-size divisor", 
    "capacity": "coupled embedding width and attention-head count", 
    "depth": "2, 4, or 6 transformer blocks", 
    "mlp_ratio": "2 or 4", 
    "drop_path": "0, 0.05, or 0.1", 
    "cnn_patchify": "boolean", 
    "timesteps": "250, 500, or 1000", 
    "noise_schedule": "linear, scaled-linear, cosine, or clipped-cosine", 
    "p_uncond": "0.05, 0.1, or 0.2", 
    "ema_decay": "0.99, 0.995, or 0.999", 
    "image_loss_coefficient": "0, 0.05, or 0.1"
}
_UNET = {
    **_OPTIMIZATION,
    "width_template": "(32,64), (32,64,96), or (64,96,128)", 
    "block_depth": "1 or 2 residual blocks per scale", 
    "bottleneck_multiplier": "1.5 or 2 times the widest stage", 
    "bottleneck_depth": "1, 2, or 3", 
    "embedding_budget": "64, 96, or 128 channels", 
    "normalization": "batch normalization on/off", 
    "dropout": "0, 0.05, or 0.1", 
    "resampling": "average/interpolation or learned convolutional pair", 
    "timesteps/schedule/CFG/EMA": "same compact diffusion space as DiT", 
    "image_loss_coefficient": "0, 0.05, or 0.1"
}
_VAE = {
    **_OPTIMIZATION, 
    "latent_dim": "8, 16, 32, 64, or 128", 
    "hidden_template": "validated descending dense-width template", 
    "beta": "log-uniform 0.01 to 2.0", 
    "activation": "ReLU or SELU", 
    "batch_normalization": "boolean (disabled for SELU)"
}
_CNN = {
    **_OPTIMIZATION, 
    "width_template": "three validated convolutional stage templates", 
    "stage_depth": "1 or 2 blocks per stage", 
    "kernel_size": "3 or 5; first kernel 3, 5, or 7", 
    "batch_normalization": "boolean", 
    "pooling": "max or average; global max or average", 
    "dropout": "0 to 0.5"
}
_DNN = {
    **_OPTIMIZATION, 
    "hidden_template": "one to three descending dense layers", 
    "activation/initializer": "coupled ReLU-He, ELU-He, or SELU-LeCun", 
    "batch_normalization": "boolean except with SELU", 
    "dropout": "0 to 0.5"
}
_PRETRAINED = {
    **_OPTIMIZATION, 
    "unfrozen_tail": "1, 5, 20, or all Xception layers", 
    "dropout": "0 to 0.5"
}
_JOINT_NOTE = {
    "classifier_loss_coefficient": "log-uniform 1e-3 to 1e-1", 
    "masking_recipe": "CFG-null, timestep, or both", 
    "objective": "Pareto minimize generative loss / maximize selected accuracy"
}
_DIT_CLASSIFIER_NOTE = {
    "classifier_depth": "1, 2, or 3", 
    "classifier_dropout": "0, 0.1, or 0.2"
}
_CONTINUAL_NOTE = {
    "replay_samples_per_class": "100, 500, or 1000", 
    "generator_train_samples": "current data, 1000, or 5000", 
    "objective": "maximize selected mean class-incremental accuracy"
}


SEARCH_SPACES = {
    "generation": {
        "diffusion_transformer": _DIT, 
        "dit_decoder": {
            **{key: value for key, value in _DIT.items() if key != "depth"}, 
            "decoder_depth": "1, 2, or 4"
        },
        "dit_encoder_decoder": {
            **{key: value for key, value in _DIT.items() if key != "depth"}, 
            "encoder_depth": "2, 4, or 6", 
            "decoder_depth": "1, 2, or 4"
        }, 
        "unet": _UNET, 
        "vae": _VAE
    },
    "joint": {
        "dit_classifier": {**_DIT, **_DIT_CLASSIFIER_NOTE, **_JOINT_NOTE}, 
        "dit_encoder_decoder_classifier": {
            **{key: value for key, value in _DIT.items() if key != "depth"}, 
            **_DIT_CLASSIFIER_NOTE, 
            **_JOINT_NOTE, 
            "encoder_depth": "2, 4, or 6", 
            "decoder_depth": "1, 2, or 4"
        }, 
        "unet_classifier": {
            **_UNET, 
            "classifier_depth": "1, 2, or 3", 
            **_JOINT_NOTE
        }, 
        "vae_classifier": {
            **_VAE, 
            "alpha": "log-uniform 1e-5 to 1e-2 (mean CE)",
            "objective": "Pareto minimize reconstruction / maximize accuracy"
        }
    },
    "classification": {
        "cnn": _CNN, 
        "dnn": _DNN, 
        "pretrained": _PRETRAINED
    }, 
    "continual": {
        "diffusion_transformer": {**_DIT, **_CONTINUAL_NOTE}, 
        "dit_classifier": {
            **_DIT, **_DIT_CLASSIFIER_NOTE, **_JOINT_NOTE, **_CONTINUAL_NOTE
        }, 
        "dit_decoder": {
            **{key: value for key, value in _DIT.items() if key != "depth"}, 
            "decoder_depth": "1, 2, or 4", 
            **_CONTINUAL_NOTE
        }, 
        "dit_encoder_decoder": {
            **{key: value for key, value in _DIT.items() if key != "depth"}, 
            "encoder_depth": "2, 4, or 6", 
            "decoder_depth": "1, 2, or 4", 
            **_CONTINUAL_NOTE
        }, 
        "dit_encoder_decoder_classifier": {
            **{key: value for key, value in _DIT.items() if key != "depth"}, 
            **_DIT_CLASSIFIER_NOTE, 
            **_JOINT_NOTE, 
            "encoder_depth": "2, 4, or 6", 
            "decoder_depth": "1, 2, or 4", 
            **_CONTINUAL_NOTE
        }, 
        "unet": {**_UNET, **_CONTINUAL_NOTE}, 
        "unet_classifier": {
            **_UNET, 
            "classifier_depth": "1, 2, or 3", 
            **_JOINT_NOTE, 
            **_CONTINUAL_NOTE
        }, 
        "vae": {**_VAE, **_CONTINUAL_NOTE}, 
        "vae_classifier": {
            **_VAE, 
            "alpha": "log-uniform 1e-5 to 1e-2 (mean CE)",
            **_CONTINUAL_NOTE
        }
    }
}


_MODEL_TAGS = {
    "diffusion_transformer": "dt", 
    "dit_classifier": "dc", 
    "dit_decoder": "dd", 
    "dit_encoder_decoder": "de", 
    "dit_encoder_decoder_classifier": "dec", 
    "unet": "u", 
    "unet_classifier": "uc", 
    "vae": "v", 
    "vae_classifier": "cv", 
    "cnn": "c", 
    "dnn": "d", 
    "pretrained": "p"
}


def _value_tag(value: object) -> str:
    """Convert one sampled value to a compact path-safe tag.

    Args:
        value (object): Optuna parameter value.

    Returns:
        str: Compact alphanumeric representation used in event filenames.
    """

    short_values = {
        "adam": "a", "adamw": "aw", "linear": "l", 
        "scaled_linear": "sl", "squaredcos_cap_v2": "co", 
        "clipped_cosine": "cc", "convolution": "c", "pool": "p", 
        "timestep": "t", "both": "b", "null": "n", 
        "relu": "r", "elu": "e", "selu": "s", "max": "x", 
        "avg": "a"
    }

    # Use stable abbreviations for common categorical values.
    if value in short_values:
        return short_values[value]

    # Encode booleans compactly for filesystem-safe trial names.
    if isinstance(value, bool):
        return str(int(value))

    # Format floats compactly while retaining meaningful precision.
    if isinstance(value, float):
        return f"{value:.2g}".replace("+", "")

    return re.sub(r"[^A-Za-z0-9.-]", "", str(value)).replace("_", "")


def _tensorboard_name(trial: Any) -> str:
    """Build a compact deterministic TensorBoard suffix for one trial.

    Args:
        trial (optuna.trial.Trial): Trial with ``number`` and ``params`` state.

    Returns:
        str: Trial-number prefix followed by value tags in key-sorted order.
    """

    # Values follow alphabetical parameter-name order. The full mapping is in
    # config.yaml and the TensorBoard text summary; omitting repeated long keys
    # keeps Windows event-file paths below MAX_PATH.
    parts = [f"t{trial.number:04d}"]
    for _, value in sorted(trial.params.items()):
        parts.append(_value_tag(value))

    return "-".join(parts)


def _suggest_optimizer(trial: Any, family: str) -> dict[str, object]:
    """Suggest optimizer and batch settings for a model family.

    Args:
        trial (optuna.trial.Trial): Active Optuna trial.
        family (str): Normalized model-family name.

    Returns:
        dict[str, object]: Batch size and nested optimizer configuration.
    """

    # Search conservative rates for pretrained feature extractors.
    if family == "pretrained":
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True)
        batch_choices = [32, 64, 128]
    # Search the VAE-specific learning-rate range.
    elif family in ("vae", "vae_classifier"):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_choices = [128, 256, 512]
    # Search the classifier-specific learning-rate range.
    elif family in ("cnn", "dnn"):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
        batch_choices = [64, 128, 256]
    # Search the U-Net-specific learning-rate range.
    elif family in ("unet", "unet_classifier"):
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True)
        batch_choices = [32, 64, 128]
    # Use the transformer learning-rate range for remaining families.
    else:
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        batch_choices = [32, 64, 128]

    optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    weight_decay = trial.suggest_float(
        "weight_decay", 1e-6, 1e-3, log=True
    ) if optimizer == "adamw" else 0.

    return {
        "batch_size": trial.suggest_categorical("batch_size", batch_choices), 
        "optimizer": {
            "name": optimizer, 
            "initial_learning_rate": learning_rate, 
            "weight_decay": weight_decay, 
            "schedule": "cosine"
        }
    }


def _suggest_diffusion_wrapper(
    trial: Any, 
) -> tuple[int, dict[str, object]]:
    """Suggest diffusion process and evaluation settings.

    Args:
        trial (optuna.trial.Trial): Active Optuna trial.

    Returns:
        tuple[int, dict[str, object]]: Training timestep count and wrapper
        keyword mapping.
    """

    timesteps = trial.suggest_categorical("timesteps", [250, 500, 1000])

    return timesteps, {
        "use_ema": True, 
        "ema_decay": trial.suggest_categorical(
            "ema_decay", [0.99, 0.995, 0.999]
        ), 
        "scheduler_name": trial.suggest_categorical(
            "schedule", 
            ["linear", "scaled_linear", "squaredcos_cap_v2", "clipped_cosine"], 
        ), 
        "p_uncond": trial.suggest_categorical("p_uncond", [0.05, 0.1, 0.2]), 
        "image_loss_coef": trial.suggest_categorical(
            "image_loss_coef", [0., 0.05, 0.1]
        ), 
        "test_steps": min(50, timesteps), 
        "test_cfg_scale": 3.
    }


def _suggest_dit(
    trial: Any, 
    image_size: int, 
    model_name: str
) -> dict[str, object]:
    """Suggest a shape-compatible transformer architecture.

    Args:
        trial (optuna.trial.Trial): Active Optuna trial.
        image_size (int): Square input resolution.
        model_name (str): Selected DiT-family name.

    Returns:
        dict[str, object]: Raw-network constructor options.
    """

    capacity = trial.suggest_categorical(
        "capacity", ["32x2", "64x4", "96x4", "128x8"]
    )
    dim, heads = (int(value) for value in capacity.split("x"))
    patch_choices = [value for value in (2, 4, 7, 8) if image_size % value == 0]
    kwargs = {
        "patch_size": trial.suggest_categorical("patch_size", patch_choices), 
        "dim": dim, 
        "mha_num_heads": heads, 
        "vit_block_mlp_ratio": trial.suggest_categorical("mlp_ratio", [2., 4.]), 
        "drop_prob": trial.suggest_categorical("drop_prob", [0., 0.05, 0.1]), 
        "patchify_with_cnn": trial.suggest_categorical(
            "cnn_patchify", 
            [False, True]
        )
    }

    # Tune encoder and decoder depths independently for joint DiT models.
    if model_name in ("dit_encoder_decoder", "dit_encoder_decoder_classifier"):
        kwargs["depth"] = trial.suggest_categorical("encoder_depth", [2, 4, 6])
        kwargs["decoder_kwargs"] = {
            "depth": trial.suggest_categorical("decoder_depth", [1, 2, 4]), 
            "mha_num_heads": heads, 
            "vit_block_mlp_ratio": kwargs["vit_block_mlp_ratio"], 
            "shift_inputs": False
        }
    # Tune decoder depth for a standalone DiT decoder.
    elif model_name == "dit_decoder":
        kwargs["depth"] = trial.suggest_categorical("decoder_depth", [1, 2, 4])
        kwargs.update({
            "decoder_separate_cond": True, 
            "shift_inputs": False, 
            "use_causal_mask": False
        })
    # Tune a single shared depth for the remaining transformer families.
    else:
        kwargs["depth"] = trial.suggest_categorical("depth", [2, 4, 6])

    return kwargs


def _suggest_unet(
    trial: Any, 
    classifier: bool = False
) -> dict[str, object]:
    """Suggest a convolutional U-Net architecture.

    Args:
        trial (optuna.trial.Trial): Active Optuna trial.
        classifier (bool): Include classifier-depth settings when true.

    Returns:
        dict[str, object]: Raw U-Net constructor options.
    """

    widths_name = trial.suggest_categorical(
        "widths", ["32-64", "32-64-96", "64-96-128"]
    )
    widths = tuple(int(value) for value in widths_name.split("-"))
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 96, 128])
    quotient, remainder = divmod(embedding_dim, 3)
    resampling = trial.suggest_categorical(
        "resampling", ["pool", "convolution"]
    )

    kwargs = {
        "widths": widths, 
        "block_depth": trial.suggest_categorical("block_depth", [1, 2]), 
        "bottleneck_width": int(max(widths) * trial.suggest_categorical(
            "bottleneck_mult", [1.5, 2.]
        )), 
        "bottleneck_depth": trial.suggest_categorical(
            "bottleneck_depth", [1, 2, 3]
        ), 
        "image_embedding_dim": quotient, 
        "time_embedding_dim": quotient + remainder, 
        "label_embedding_dim": quotient, 
        "use_batch_norm": trial.suggest_categorical("batch_norm", [False, True]), 
        "dropout_rate": trial.suggest_categorical("dropout", [0., 0.05, 0.1]), 
        "downsampling_method": "avg_pooling" if resampling == "pool" else "cnn_stride", 
        "upsampling_method": "interpolate" if resampling == "pool" else "cnn_transpose"
    }

    # Add classification-head settings for classifier-capable U-Nets.
    if classifier:
        kwargs.update({
            "clf_depth": trial.suggest_categorical("clf_depth", [1, 2, 3])
        })

    return kwargs


def _suggest_vae(
    trial: Any, 
    classifier: bool = False
) -> dict[str, object]:
    """Suggest a dense VAE architecture and loss coefficients.

    Args:
        trial (optuna.trial.Trial): Active Optuna trial.
        classifier (bool): Include the classifier coefficient when true.

    Returns:
        dict[str, object]: VAE constructor options.
    """

    latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 32, 64, 128])
    template = trial.suggest_categorical(
        "hidden_template", ["256-64", "512-128", "512-256-64"]
    )
    activation = trial.suggest_categorical("activation", ["relu", "selu"])
    batch_norm = False if activation == "selu" else trial.suggest_categorical(
        "batch_norm", [False, True]
    )

    kwargs = {
        "conditioned": True, 
        "latent_dim": latent_dim, 
        "hiddens_dims": tuple(int(value) for value in template.split("-")), 
        "hiddens_kwargs": {
            "actv": activation, 
            "use_batch_norm": batch_norm, 
            "kernel_init": "lecun_normal" if activation == "selu" else "he_normal"
        }, 
        "beta": trial.suggest_float("beta", 0.01, 2., log=True)
    }

    # Tune joint classification weight only when a classifier branch exists.
    if classifier:
        kwargs["alpha"] = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)

    return kwargs


def _suggest_joint(
    trial: Any, 
    model_name: str, 
    kwargs: dict[str, object], 
    wrapper_kwargs: dict[str, object]
) -> None:
    """Add joint-classification suggestions to mutable model settings.

    Args:
        trial (optuna.trial.Trial): Active Optuna trial.
        model_name (str): Selected joint model family.
        kwargs (dict[str, object]): Raw-model options updated in place.
        wrapper_kwargs (dict[str, object]): Wrapper options updated in place.

    Returns:
        None.
    """

    # Add DiT-specific decoder and conditioning choices.
    if model_name.startswith("dit"):
        kwargs.update({
            "clf_depth": trial.suggest_categorical("clf_depth", [1, 2, 3]), 
            "clf_drop_prob": trial.suggest_categorical(
                "clf_dropout", [0., 0.1, 0.2]
            )
        })

    masking = trial.suggest_categorical(
        "masking", ["null", "timestep", "both"]
    )
    wrapper_kwargs.update({
        "clf_loss_coef": trial.suggest_float(
            "clf_loss_coef", 
            1e-3, 1e-1, 
            log=True
        ), 
        "mask_by_nulls": masking in ("null", "both"), 
        "mask_by_t_threshold": masking in ("timestep", "both")
    })

    # Tune timestep masking only for modes that use it.
    if masking in ("timestep", "both"):
        wrapper_kwargs["mask_t_percentage"] = trial.suggest_categorical(
            "mask_t", [50, 70, 90]
        )


def _suggest_classifier(
    trial: Any, 
    model_name: str
) -> dict[str, object]:
    """Suggest a standalone classifier architecture.

    Args:
        trial (optuna.trial.Trial): Active Optuna trial.
        model_name (str): ``cnn``, ``dnn``, or ``pretrained``.

    Returns:
        dict[str, object]: Classifier-family constructor options.
    """

    # Build tunable convolutional stage widths and depths.
    if model_name == "cnn":
        widths = trial.suggest_categorical(
            "widths", ["32-64-128", "64-128-256", "64-128-128-256"]
        )
        filters = tuple(int(value) for value in widths.split("-"))
        depth = trial.suggest_categorical("stage_depth", [1, 2])

        return {
            "dropout_rate": trial.suggest_float("dropout", 0., 0.5, step=0.1), 
            "architecture_kwargs": {
                "conv_filters": filters, 
                "conv_depths": (depth,) * len(filters), 
                "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]), 
                "first_kernel_size": trial.suggest_categorical(
                    "first_kernel", [3, 5, 7]
                ), 
                "use_batch_norm": trial.suggest_categorical(
                    "batch_norm", [False, True]
                ), 
                "pooling": trial.suggest_categorical("pooling", ["max", "avg"]), 
                "global_pooling": trial.suggest_categorical(
                    "global_pooling", ["avg", "max"]
                )
            }
        }

    # Build a tunable dense hidden-layer template.
    if model_name == "dnn":
        template = trial.suggest_categorical(
            "hidden_template", ["256", "512-128", "512-256-64"]
        )
        activation = trial.suggest_categorical(
            "activation", ["relu", "elu", "selu"]
        )

        return {
            "dropout_rate": trial.suggest_float("dropout", 0., 0.5, step=0.1), 
            "architecture_kwargs": {
                "hidden_dims": tuple(int(value) for value in template.split("-")), 
                "activation": activation, 
                "use_batch_norm": False if activation == "selu" else
                                trial.suggest_categorical("batch_norm", [False, True]), 
                "kernel_initializer": "lecun_normal" if activation == "selu"
                                    else "he_normal", 
            },
        }

    return {
        "dropout_rate": trial.suggest_float("dropout", 0., 0.5, step=0.1), 
        "num_last_not_frozen": trial.suggest_categorical(
            "unfrozen", [1, 5, 20, 0]
        )
    }


def _build_trial_config(
    trial: Any, 
    task: str, 
    model_name: str, 
    dataset_name: str, 
    epochs: int, 
    seed: int, 
    results_path: str | Path,
    use_ensemble_accuracy: bool = False,
    ensemble_accuracy_kwargs: Mapping[str, object] | None = None
) -> Config:
    """Build one complete, shape-compatible trial configuration.

    Args:
        trial (optuna.trial.Trial): Active Optuna trial.
        task (str): Generation, joint, classification, or continual task.
        model_name (str): Selected model family.
        dataset_name (str): Supported dataset name.
        epochs (int): Maximum epochs per phase.
        seed (int): Trial-specific random seed.
        results_path (str | pathlib.Path): HPO artifact root.
        use_ensemble_accuracy (bool): Use post-training ensemble accuracy as
            the classification objective for diffusion-classifier studies.
        ensemble_accuracy_kwargs (Mapping[str, object] | None): Options passed
            to ``DiffusionClassifier.evaluate_ensemble_accuracy``.

    Returns:
        Config: Fully typed trial configuration.

    Raises:
        ValueError: If the dataset/model combination is unsupported.
    """

    dataset_name = dataset_name.lower()

    # Reject datasets outside the four supported HPO families.
    if dataset_name not in ("fmnist", "mnist", "cifar10", "cifar100"):
        raise ValueError("dataset_name must be FMNIST, MNIST, CIFAR10, or CIFAR100.")
    # Restrict pretrained Xception search to three-channel CIFAR inputs.
    if model_name == "pretrained" and dataset_name not in ("cifar10", "cifar100"):
        raise ValueError("The Xception classifier requires three-channel CIFAR data.")
    # Restrict ensemble feedback to studies backed by classifier diffusion wrappers.
    if use_ensemble_accuracy and not (
        task in ("joint", "continual")
        and model_name in _DIFFUSION_CLASSIFIER_MODELS
    ):
        raise ValueError(
            "use_ensemble_accuracy requires a joint or continual diffusion "
            "classifier study."
        )

    ensemble_accuracy_kwargs = dict(ensemble_accuracy_kwargs or {})

    _, image_shape, _ = get_dataset_spec(dataset_name)
    image_size = image_shape[0]
    optimization = _suggest_optimizer(trial, model_name)

    model_kwargs = {}
    wrapper_kwargs = {}
    classifier_name = None
    classifier_kwargs = {}
    wrapper_name = None
    preprocess = "standardize" if model_name in (
        "diffusion_transformer", "dit_classifier", "dit_decoder", 
        "dit_encoder_decoder", "dit_encoder_decoder_classifier", 
        "unet", "unet_classifier"
    ) else "min-max"
    return_features = False
    features_path = None
    onehot_labels = model_name in ("vae", "vae_classifier")

    # Tune transformer diffusion schedules and wrapper behavior.
    if model_name.startswith("dit") or model_name == "diffusion_transformer":
        timesteps, wrapper_kwargs = _suggest_diffusion_wrapper(trial)
        model_kwargs = _suggest_dit(trial, image_size, model_name)
        model_kwargs.update({"timesteps": timesteps, "use_cfg": True})
    # Tune U-Net diffusion schedules and wrapper behavior.
    elif model_name in ("unet", "unet_classifier"):
        timesteps, wrapper_kwargs = _suggest_diffusion_wrapper(trial)
        model_kwargs = _suggest_unet(
            trial, 
            classifier=model_name == "unet_classifier"
        )
        model_kwargs.update({"timesteps": timesteps, "use_cfg": True})
    # Tune VAE architecture and reconstruction settings.
    elif model_name in ("vae", "vae_classifier"):
        model_kwargs = _suggest_vae(
            trial, 
            classifier=model_name == "vae_classifier"
        )
        model_kwargs["last_activation"] = "sigmoid"

        classifier_name = "dnn" if model_name == "vae_classifier" else None
        classifier_kwargs = {
            "dropout_rate": 0.2, 
            "architecture_kwargs": {"hidden_dims": (256,), "activation": "relu"}
        }
    # Tune standalone classifier architecture for classification tasks.
    elif task == "classification":
        model_kwargs = _suggest_classifier(trial, model_name)

        # Use saved features for dense-classifier trials.
        if model_name == "dnn":
            preprocess = "normalize"
        # Preserve raw image scale for pretrained preprocessing layers.
        elif model_name == "pretrained":
            preprocess = None

    # Add joint loss/search options for supported generative classifiers.
    if task in ("joint", "continual") and model_name in (
        "dit_classifier", "dit_encoder_decoder_classifier", 
        "unet_classifier"
    ):
        _suggest_joint(trial, model_name, model_kwargs, wrapper_kwargs)
        wrapper_name = "diffusion_classifier"

    continual_kwargs = {}
    # Tune replay policy only for continual-learning studies.
    if task == "continual":
        replay_samples = trial.suggest_categorical(
            "replay_samples", [100, 500, 1000]
        )
        train_num = trial.suggest_categorical("train_num", [-1, 1000, 5000])
        continual_kwargs = {
            "remove_prev_classes": True, 
            "keep_same_model": True, 
            "plot_results": False, 
            "generative_model_kwargs": {
                "samples_per_class": replay_samples, 
                "train_num": train_num
            }
        }

        # Enable per-task diffusion ensemble reports when they are the HPO signal.
        if use_ensemble_accuracy:
            continual_kwargs.update({
                "evaluate_ensemble_accuracy": True,
                "ensemble_accuracy_kwargs": ensemble_accuracy_kwargs
            })

        # Select and configure a dense classifier for VAE replay.
        if model_name in ("vae", "vae_classifier"):
            classifier_name = "dnn"
            return_features = dataset_name in ("cifar10", "cifar100")
            preprocess, onehot_labels = "normalize", True
            # Point dense VAE replay at the dataset's saved feature archive.
            if return_features:
                features_path = str(
                    Path("data")
                    / f"{dataset_name}_xception_gavgpooled_features_train_val_test"
                )
            model_kwargs["last_activation"] = "linear"

            classifier_kwargs = {
                "dropout_rate": 0.2, 
                "architecture_kwargs": {
                    "hidden_dims": (256,), 
                    "activation": "relu", 
                }
            }
        # Select a convolutional classifier for image-space replay.
        else:
            classifier_name = "cnn"
            preprocess = "min-max"
            classifier_kwargs = {
                "dropout_rate": 0.2, 
                "architecture_kwargs": {
                    "conv_filters": (32, 64, 128), 
                    "conv_depths": (1, 1, 1), 
                }
            }

    # Optimize validation accuracy for standalone classification.
    if task == "classification":
        monitor, monitor_mode = "val_accuracy", "max"
    # Optimize both generation loss and classification accuracy jointly.
    elif task == "joint":
        monitor = "val_clf_accuracy" if model_name == "vae_classifier" \
                else "val_classifier_accuracy"
        monitor_mode = "max"
    # Optimize the final continual-learning accuracy.
    elif task == "continual":
        monitor, monitor_mode = "val_accuracy", "max"
    # Minimize validation reconstruction loss for VAE generation.
    elif model_name in ("vae", "vae_classifier"):
        monitor, monitor_mode = "val_recon_loss", "min"
    # Minimize validation diffusion noise loss for other generators.
    else:
        monitor, monitor_mode = "val_noise_loss", "min"

    tensorboard_name = _tensorboard_name(trial)
    task_tag = {"generation": "g", "joint": "j", "classification": "c", "continual": "l"}[task]
    dataset_tag = {
        "mnist": "m", "fmnist": "fm",
        "cifar10": "c10", "cifar100": "c100"
    }[
        dataset_name
    ]
    project_tag = f"t{trial.number:04d}"
    trial_root = Path(results_path) / task / model_name / dataset_name
    # Match the separate study storage used for ensemble-feedback trials.
    if use_ensemble_accuracy:
        trial_root /= "ensemble_accuracy"
    tensorboard_root = Path(results_path) / "_tb" / (
        task_tag + _MODEL_TAGS[model_name] + dataset_tag
    )
    # Avoid TensorBoard event-name collisions with ordinary-accuracy trials.
    if use_ensemble_accuracy:
        tensorboard_root /= "ensemble_accuracy"
    config = Config(
        dataset={
            "name": dataset_name, 
            "batch_size": optimization["batch_size"], 
            "preprocess": preprocess, 
            "features_path": features_path,
            "return_features": return_features, 
            "onehot_labels": onehot_labels
        }, 
        model={
            "name": model_name, 
            "wrapper_name": wrapper_name, 
            "kwargs": model_kwargs, 
            "wrapper_kwargs": wrapper_kwargs, 
            "classifier_name": classifier_name, 
            "classifier_kwargs": classifier_kwargs, 
            "show_network_summary": False
        }, 
        optimizer=optimization["optimizer"], 
        training={
            "task": task, 
            "epochs": epochs, 
            "seed": seed, 
            "verbose": 0, 
            "patience": 5 if task in ("generation", "classification") else 0, 
            "monitor": monitor, 
            "monitor_mode": monitor_mode, 
            "tensorboard": True, 
            "tensorboard_path": str(tensorboard_root), 
            "tensorboard_run_name": tensorboard_name, 
            "report_every_epoch": False, 
            "show_images": False, 
            "save_gifs": False, 
            "results_path": str(trial_root / "runs"), 
            "project_tag": project_tag
        }, 
        continually_learn=continual_kwargs, 
        reporting={
            "show_history_plot": False, 
            "save_history_plot": True, 
            "show_final_images": False, 
            "save_final_images": task in ("generation", "joint"), 
            "save_final_gifs": model_name in _DIFFUSION_MODELS and task != "continual", 
            "final_images_steps": min(50, model_kwargs.get("timesteps", 50)), 
            "final_images_cfg_scale": 3., 
            "plot_without_20percent": False, 
            "run_trainset_eval": False, 
            "run_valset_eval": task != "continual", 
            "evaluate_ensemble_accuracy": (
                use_ensemble_accuracy and task == "joint"
            ),
            "ensemble_accuracy_kwargs": ensemble_accuracy_kwargs,
            "save_csv": True
        }, 
        hpo={
            "study_task": task, 
            "study_model": model_name, 
            "trial_number": trial.number, 
            "params": dict(trial.params), 
            "tensorboard_name": tensorboard_name,
            "use_ensemble_accuracy": use_ensemble_accuracy,
            "ensemble_accuracy_kwargs": ensemble_accuracy_kwargs
        }
    )

    return config


def _history_value(
    history: Mapping[str, Sequence[float]], 
    names: Sequence[str], 
    best: str | None = None
) -> float:
    """Read a final, minimum, or maximum available history metric.

    Args:
        history (Mapping[str, Sequence[float]]): Logged metric sequences.
        names (Sequence[str]): Candidate names in preference order.
        best (str | None): ``min``, ``max``, or ``None`` for the final value.

    Returns:
        float: Selected metric value.

    Raises:
        KeyError: If no candidate has a nonempty sequence.
    """

    for name in names:
        # Read the requested nonempty metric series from training history.
        if name in history and history[name]:
            values = history[name]

            # Select the minimum observed value for minimization objectives.
            if best == "min":
                return float(min(values))

            # Select the maximum observed value for maximization objectives.
            if best == "max":
                return float(max(values))

            return float(values[-1])

    raise KeyError(
        "None of the objective metrics were logged: " + ", ".join(names)
    )


def _ensemble_accuracy_value(
    evaluations: Mapping[str, object]
) -> float:
    """Read validation ensemble accuracy, preferring EMA over raw weights.

    Args:
        evaluations (Mapping[str, object]): Final report evaluations containing
            raw and optional EMA validation dictionaries.

    Returns:
        float: Selected validation ensemble accuracy.

    Raises:
        KeyError: If no validation ensemble result was reported.
    """

    for name in ("valset_ema_eval", "valset_network_eval"):
        network_results = evaluations.get(name)
        # Return the first exact ensemble result in network preference order.
        if isinstance(network_results, Mapping) \
        and "ensemble_accuracy" in network_results:
            return float(network_results["ensemble_accuracy"])

    raise KeyError("No validation ensemble_accuracy was reported.")


def _objective_values(
    task: str,  
    model_name: str,  
    history: Mapping[str, Sequence[float]], 
    evaluations: Mapping[str, object] | None = None, 
    use_ensemble_accuracy: bool = False
) -> float | tuple[float, float]:
    """Convert training history to the study's objective value or pair.

    Args:
        task (str): Study task.
        model_name (str): Selected model family.
        history (Mapping[str, Sequence[float]]): Training metric history.
        evaluations (Mapping[str, object] | None): Final report evaluations,
            required for joint ensemble-accuracy feedback.
        use_ensemble_accuracy (bool): Select ensemble instead of ordinary
            classification accuracy.

    Returns:
        float | tuple[float, float]: Scalar objective, or generation-loss and
        classification-accuracy pair for joint studies.
    """

    # Reject ensemble feedback for models without a diffusion classifier wrapper.
    if use_ensemble_accuracy and not (
        task in ("joint", "continual") and 
        model_name in _DIFFUSION_CLASSIFIER_MODELS
    ):
        raise ValueError(
            "use_ensemble_accuracy requires a joint "
            "or continual diffusion classifier study."
        )

    # Return the single generation objective for generation studies.
    if task == "generation":
        names = [
            "val_recon_loss", "recon_loss", 
            "val_total_loss", "total_loss"
        ] if model_name in ("vae", "vae_classifier") else [
            "val_noise_loss", "noise_loss", 
            "val_loss", "loss"
        ]

        return _history_value(history, names, best="min")

    # Combine generation and classification objectives for joint studies.
    if task == "joint":
        generation = _history_value(history, [
            "val_noise_loss", "val_recon_loss", 
            "noise_loss", "recon_loss"
        ])
        # Read the post-training report when ensemble feedback is requested.
        if use_ensemble_accuracy:
            classification = _ensemble_accuracy_value(evaluations or {})
        # Preserve the legacy training-history objective otherwise.
        else:
            classification = _history_value(history, [
                "val_classifier_accuracy", "val_clf_accuracy", 
                "classifier_accuracy", "clf_accuracy"
            ])

        return generation, classification

    # Return only classifier accuracy for classification studies.
    if task == "classification":
        return _history_value(
            history, 
            ["val_accuracy", "accuracy"], 
            best="max"
        )

    continual_metric = "continual_ensemble_accuracy" if use_ensemble_accuracy \
                    else "continual_accuracy"

    return float(pd.Series(history[continual_metric]).mean())


def run_hpo(
    *, 
    task: str, 
    model_name: str, 
    dataset_name: str = "CIFAR10", 
    n_trials: int = 30, 
    epochs: int = 30, 
    seed: int = 42, 
    results_path: str = "results/hpo", 
    timeout: float | None = None,
    use_ensemble_accuracy: bool = False,
    ensemble_accuracy_kwargs: Mapping[str, object] | None = None
) -> Any:
    """Run a persistent Optuna study and return its ``Study`` object.

    ``joint`` studies are Pareto searches with generative loss minimized and
    classification accuracy maximized. Other tasks have one objective. Each
    trial saves all normal training artifacts plus its input/resolved config;
    the dataset-specific study directory stores SQLite state and an
    incrementally updated CSV.

    Args:
        task (str): ``generation``, ``joint``, ``classification``, or
            ``continual``.
        model_name (str): A key in ``SEARCH_SPACES[task]``.
        dataset_name (str): ``FMNIST``, ``MNIST``, ``CIFAR10``, or
            ``CIFAR100``. Xception requires a three-channel CIFAR dataset.
        n_trials (int): Positive number of additional trials to run.
        epochs (int): Positive maximum epochs per fit/task.
        seed (int): Reproducible TPE and first-trial seed.
        results_path (str): HPO root. Study state is written below
            ``<task>/<model>/<dataset>`` and TensorBoard events below ``_tb``.
        timeout (float | None): Optional study wall-time limit in seconds.
        use_ensemble_accuracy (bool): Use post-training ensemble accuracy as
            HPO feedback for joint or continual diffusion-classifier studies.
        ensemble_accuracy_kwargs (Mapping[str, object] | None): Options passed
            to ``DiffusionClassifier.evaluate_ensemble_accuracy``.

    Returns:
        optuna.study.Study: Resumable completed/partial study. Joint studies
        expose ``best_trials``; other studies expose ``best_trial``.

    Raises:
        ImportError: If Optuna is unavailable.
        ValueError: If the task/model pair or positive budgets are invalid.
    """

    try:
        import optuna
    except ImportError as error:
        raise ImportError(
            "Optuna is required for HPO. "
            "Install the project requirements."
        ) from error


    task = task.lower()
    model_name = model_name.lower()

    # Require a supported task/model search-space pairing.
    if task not in SEARCH_SPACES or model_name not in SEARCH_SPACES[task]:
        raise ValueError(f"Unsupported task/model pair: {task}/{model_name}")
    # Restrict ensemble feedback to supported diffusion-classifier studies.
    if use_ensemble_accuracy and not (
        task in ("joint", "continual")
        and model_name in _DIFFUSION_CLASSIFIER_MODELS
    ):
        raise ValueError(
            "use_ensemble_accuracy requires a joint "
            "or continual diffusion classifier study."
        )
    # Require positive trial and epoch budgets.
    if n_trials <= 0 or epochs <= 0:
        raise ValueError("n_trials and epochs must be positive.")

    root = Path(results_path)
    study_root = root / task / model_name / dataset_name.lower()
    # Keep ensemble-feedback trials separate from legacy accuracy studies.
    if use_ensemble_accuracy:
        study_root /= "ensemble_accuracy"

    configs_path = study_root / "configs"
    configs_path.mkdir(parents=True, exist_ok=True)
    storage_path = (study_root / "study.db").resolve().as_posix()
    study_name = f"{task}-{model_name}-{dataset_name.lower()}" + (
        "-ensemble-accuracy" if use_ensemble_accuracy else ""
    )
    create_kwargs = {
        "study_name": study_name, 
        "storage": "sqlite:///" + storage_path, 
        "sampler": optuna.samplers.TPESampler(seed=seed), 
        "load_if_exists": True
    }

    # Configure two optimization directions for joint studies.
    if task == "joint":
        create_kwargs["directions"] = ["minimize", "maximize"]
    # Configure the single direction used by other study types.
    else:
        create_kwargs["direction"] = (
            "minimize" if task == "generation" else "maximize"
        )

    study = optuna.create_study(**create_kwargs)


    def objective(trial: Any) -> float | tuple[float, float]:
        """Execute and score one Optuna trial.

        Args:
            trial (optuna.trial.Trial): Active trial.

        Returns:
            float | tuple[float, float]: Study objective value or joint pair.
        """
        tf.keras.backend.clear_session()
        gc.collect()

        config = _build_trial_config(
            trial, 
            task, 
            model_name, 
            dataset_name, 
            epochs, 
            seed + trial.number, 
            results_path=root, 
            use_ensemble_accuracy=use_ensemble_accuracy, 
            ensemble_accuracy_kwargs=ensemble_accuracy_kwargs
        )

        input_config_path = configs_path / f"trial-{trial.number:04d}.yaml"
        save_config(config, input_config_path)
        config = load_config(input_config_path)

        result = main(config)
        values = _objective_values(
            task, 
            model_name, 
            result["history"], 
            evaluations=result["evaluations"], 
            use_ensemble_accuracy=config.hpo["use_ensemble_accuracy"]
        )
        values_list = list(values) if isinstance(values, tuple) else [values]
        config.hpo["objectives"] = values_list

        resolved_path = Path(result["results_path"]) / "config.yaml"
        save_config(config, resolved_path)
        pd.DataFrame([{
            "name": "generation_loss" if task == "joint" else (
                    "ensemble_accuracy" if use_ensemble_accuracy else "objective"
            ),
            "value": values_list[0], 
        }, *([{
            "name": "ensemble_accuracy" if use_ensemble_accuracy \
                    else "classification_accuracy",
            "value": values_list[1], 
        }] if task == "joint" else [])]).to_csv(
            Path(result["results_path"]) / "objectives.csv", 
            index=False
        )
        trial.set_user_attr("results_path", str(result["results_path"]))
        trial.set_user_attr("config_path", str(resolved_path))

        return values


    def save_trials(study_: Any, trial_: Any) -> None:
        """Persist the study table after a completed trial.

        Args:
            study_ (optuna.study.Study): Updated study.
            trial_ (optuna.trial.FrozenTrial): Just-completed trial; unused.

        Returns:
            None.
        """

        del trial_
        study_.trials_dataframe().to_csv(
            study_root / "trials.csv", 
            index=False
        )


    study.optimize(
        objective, 
        n_trials=n_trials, 
        timeout=timeout, 
        callbacks=[save_trials], 
        catch=(tf.errors.ResourceExhaustedError,), 
        gc_after_trial=True
    )

    return study


__all__ = ["SEARCH_SPACES", "run_hpo"]
