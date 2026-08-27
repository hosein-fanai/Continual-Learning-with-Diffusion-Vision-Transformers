"""Optuna studies that reuse the project's config-driven training pipeline.

Searches deliberately use a few validated architecture templates instead of
sampling arbitrary routing dictionaries. Every trial writes a YAML config,
reloads it with :func:`common.config.load_config`, and runs
:func:`common.train.main`.
"""

from __future__ import annotations

import tensorflow as tf

import pandas as pd
import numpy as np

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
    "learning_rate_schedule": "cosine decay or constant", 
    "optimizer": "SGD, RMSprop, Adam, AdamW, or Nadam", 
    "weight_decay": "AdamW-only log-uniform value from 1e-6 to 1e-3"
}
_DIT = {
    **_OPTIMIZATION, 
    "patch_size": "valid image-size divisor", 
    "capacity": (
        "32x2, 64x4, 96x4, 128x8, 32x4, 64x2, 96x2, 128x4, "
        "32x6, 64x6, 96x6, or 128x6"
    ), 
    "depth": "2, 3, 4, 5, or 6 transformer blocks", 
    "mlp_ratio": "1, 2, 4, or 6", 
    "drop_prob": "0, 0.05, 0.1, or 0.2", 
    "patchify_with_cnn": "boolean", 
    "use_refiner_cnn": "boolean (applied to the decoder when present)", 
    "timesteps": "250, 500, or 1000", 
    "noise_schedule": "linear, scaled-linear, cosine, or clipped-cosine", 
    "modify_first_t": "boolean", 
    "p_uncond": "0.05, 0.1, 0.2, or 0.25", 
    "ema_decay": "0.99, 0.995, or 0.999", 
    "loss_function": "mse or mae", 
    "image_loss_coefficient": "0, 0.01, 0.05, or 0.1", 
    "test_steps": "10, 20, 50, 100, 250, 500, or 1000 up to timesteps", 
    "test_cfg_scale": "uniform 1.1 to 7", 
    "test_eta": "uniform 0 to 1"
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
    "timesteps": "250, 500, or 1000", 
    "noise_schedule": "linear, scaled-linear, cosine, or clipped-cosine", 
    "modify_first_t": "boolean", 
    "p_uncond": "0.05, 0.1, 0.2, or 0.25", 
    "ema_decay": "0.99, 0.995, or 0.999", 
    "loss_function": "mse or mae", 
    "image_loss_coefficient": "0, 0.01, 0.05, or 0.1", 
    "test_steps": "10, 20, 50, 100, 250, 500, or 1000 up to timesteps", 
    "test_cfg_scale": "uniform 1.1 to 7", 
    "test_eta": "uniform 0 to 1"
}
_VAE = {
    **_OPTIMIZATION, 
    "latent_dim": "8, 16, 32, 64, or 128", 
    "hidden_template": "validated descending dense-width template", 
    "beta": "log-uniform 0.01 to 2.0", 
    "loss_function": "mse or mae", 
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
    "classifier_loss_coefficient": "log-uniform 1e-4 to 1e-1", 
    "ctr_loss_coef": (
        "0, 1e-4, 1e-3, 1e-2, or 5e-2; positive values enable a final "
        "classifier token regularizer"
    ), 
    "masking_recipe": "V1 only: CFG-null, timestep, both, or neither", 
    "mask_t_percentage": (
        "V1 only: 35, 50, 70, or 90 when timestep masking is used"
    ), 
    "distillation": (
        "with a runtime or previous-task teacher: hard or soft targets and "
        "log-uniform distil_loss_coef"
    ), 
    "objective": "Pareto minimize generative loss / maximize selected accuracy"
}
_DIT_CLASSIFIER_NOTE = {
    "classifier_architecture": (
        "linear, local mixer, connection, cross-attention, decoder "
        "cross-attention, cross-attention aggregation, U-shaped, or U-VAE"
    ), 
    "feature_aggregation": "last denoiser feature or all features", 
    "classifier_only_cls_token": "boolean", 
    "classifier_cls_token_type": (
        "new weight, time-label, or label when a separate token is used"
    ), 
    "classifier_depth": "1, 2, 3, 4, or 5", 
    "classifier_layer_norm_adaptation": "enabled or disabled", 
    "classifier_block_dropout": "0, 0.1, 0.2, or 0.25", 
    "classifier_mlp_ratio": "None, 1, or 2", 
    "classifier_head_dropout": "0, 0.1, 0.2, 0.25, or 0.5"
}
_DIT_CLASSIFIER_WRAPPER_NOTE = {
    "wrapper_name": "diffusion_classifier (V1) or diffusion_classifier_v2 (V2)", 
    "v2_classifier_variables": (
        "V2 only: classifier-only, shared embedding, first/final, or final-two "
        "main-transformer variable recipes"
    ), 
    "clf_train_noisified_max_timesteps": (
        "V2 only: None (clean input at timestep 0), 64, 128, 256, 512, "
        "or timesteps (full [0, timesteps) range); numeric caps are bounded "
        "by timesteps"
    )
}
_CONTINUAL_NOTE = {
    "replay_samples_per_class": "100, 500, 1000, 2500, or 5000", 
    "generator_train_samples": (
        "current data, 1000, 2500, 5000, 7500, or 10000"
    ), 
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
        "dit_classifier": {
            **_DIT, 
            **_DIT_CLASSIFIER_NOTE, 
            **_DIT_CLASSIFIER_WRAPPER_NOTE, 
            **_JOINT_NOTE
        }, 
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
            **_DIT, 
            **_DIT_CLASSIFIER_NOTE, 
            **_DIT_CLASSIFIER_WRAPPER_NOTE, 
            **_JOINT_NOTE, 
            **_CONTINUAL_NOTE
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
_FIT_METHODS = {"fit", "fit_progressively"}


def _validate_fit_request(
    model_name: str,
    fit_method: str,
    fit_kwargs: Mapping[str, object],
) -> None:
    """Validate the training-method part of an HPO request.

    Args:
        model_name (str): Normalized model family name.
        fit_method (str): Ordinary or progressive fit selector.
        fit_kwargs (Mapping[str, object]): Selected-method arguments.

    Returns:
        None: The request is compatible with the selected model family.

    Raises:
        ValueError: If the method is unsupported or a progressive request is
            incompatible with the model or lacks ``stage_tasks``.
    """

    # Restrict HPO orchestration to its two public training methods.
    if fit_method not in _FIT_METHODS:
        raise ValueError("fit_method must be 'fit' or 'fit_progressively'.")
    # Progressive curricula are implemented only by diffusion wrappers.
    if fit_method == "fit_progressively" and model_name not in _DIFFUSION_MODELS:
        raise ValueError("fit_progressively requires a diffusion model family.")
    # Require the curriculum property changed by every progressive stage.
    if fit_method == "fit_progressively" and fit_kwargs.get("stage_tasks") is None:
        raise ValueError("fit_kwargs must include stage_tasks for fit_progressively.")


def _value_tag(value: object) -> str:
    """Convert one sampled value to a compact path-safe tag.

    Args:
        value (object): Optuna parameter value.

    Returns:
        str: Compact alphanumeric representation used in event filenames.
    """

    short_values = {
        "sgd": "s", "rmsprop": "r", "adam": "a", "adamw": "aw", 
        "nadam": "n", "cosine": "c", "constant": "k", 
        "linear": "l", "scaled_linear": "sl", "squaredcos_cap_v2": "co", 
        "clipped_cosine": "cc", "convolution": "c", "pool": "p", 
        "timestep": "t", "both": "b", "null": "n", "neither": "x", 
        "last": "l", "all": "a", "new_weight": "nw", 
        "time_label": "tl", "label": "y", 
        "diffusion_classifier": "v1", 
        "diffusion_classifier_v2": "v2", 
        "local_mixer": "lm", "connection": "cn", 
        "cross_attention": "ca", "cross_attention_decoder": "cad", 
        "cross_attention_aggregation": "caa", 
        "u_shape": "us", "u_vae": "uv", 
        "none": "x", "conditions": "ce", "core": "co", 
        "notebook": "nb", "first": "f", "last": "z", 
        "last_two": "zz", "timesteps": "ts", 
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


def _suggest_optimizer(
    trial: Any, 
    family: str
) -> dict[str, object]:
    """Suggest optimizer and batch settings for a model family.

    Args:
        trial (optuna.trial.Trial): Active Optuna trial.
        family (str): Normalized model-family name.

    Returns:
        dict[str, object]: Batch size and nested optimizer configuration.
    """

    # Search conservative rates for pretrained feature extractors.
    if family == "pretrained":
        learning_rate = trial.suggest_float(
            "learning_rate", 1e-6, 5e-4, log=True
        )
        batch_choices = [32, 64, 128]
    # Search the VAE-specific learning-rate range.
    elif family in ("vae", "vae_classifier"):
        learning_rate = trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True
        )
        batch_choices = [128, 256, 512]
    # Search the classifier-specific learning-rate range.
    elif family in ("cnn", "dnn"):
        learning_rate = trial.suggest_float(
            "learning_rate", 1e-5, 3e-3, log=True
        )
        batch_choices = [64, 128, 256]
    # Search the U-Net-specific learning-rate range.
    elif family in ("unet", "unet_classifier"):
        learning_rate = trial.suggest_float(
            "learning_rate", 1e-4, 2e-3, log=True
        )
        batch_choices = [32, 64, 128]
    # Use the transformer learning-rate range for remaining families.
    else:
        learning_rate = trial.suggest_float(
            "learning_rate", 1e-5, 5e-3, log=True
        )
        batch_choices = [32, 64, 128]

    batch_size = trial.suggest_categorical("batch_size", batch_choices)
    optimizer = trial.suggest_categorical("optimizer", [
        "sgd", "rmsprop", "adam", 
        "adamw", "nadam"
    ])
    # TensorFlow 2.10 exposes weight decay only through AdamW here.
    weight_decay = trial.suggest_float(
        "weight_decay", 1e-6, 1e-3, log=True
    ) if optimizer == "adamw" else None
    schedule = trial.suggest_categorical(
        "learning_rate_schedule", ["cosine", "constant"]
    )

    return {
        "batch_size": batch_size, 
        "optimizer": {
            "name": optimizer, 
            "initial_learning_rate": learning_rate, 
            "weight_decay": weight_decay, 
            "schedule": schedule
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

    timesteps = trial.suggest_categorical(
        "timesteps", [250, 500, 1000]
    )
    test_step_choices = [
        value for value in (10, 20, 50, 100, 250, 500, 1_000)
        if value <= timesteps
    ]
    test_steps_index = trial.suggest_int(
        "test_steps_index", 0, len(test_step_choices) - 1
    )

    return timesteps, {
        "use_ema": True, 
        "ema_decay": trial.suggest_categorical(
            "ema_decay", [0.99, 0.995, 0.999]
        ), 
        "scheduler_name": trial.suggest_categorical(
            "schedule", [
                "linear", "scaled_linear", 
                "squaredcos_cap_v2", "clipped_cosine"
            ]
        ), 
        "modify_first_t": trial.suggest_categorical(
            "modify_first_t", [True, False]
        ), 
        "p_uncond": trial.suggest_categorical(
            "p_uncond", [0.05, 0.1, 0.2, 0.25]
        ), 
        "image_loss_coef": trial.suggest_categorical(
            "image_loss_coef", [0., 0.01, 0.05, 0.1]
        ), 
        "test_steps": test_step_choices[test_steps_index], 
        "test_cfg_scale": trial.suggest_float(
            "test_cfg_scale", 1.1, 7.
        ), 
        "test_eta": trial.suggest_float(
            "test_eta", 0., 1.
        )
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
        "capacity", [
            "32x2", "64x4", "96x4", "128x8", 
            "32x4", "64x2", "96x2", "128x4", 
            "32x6", "64x6", "96x6", "128x6", 
        ]
    )
    dim, heads = (int(value) for value in capacity.split("x"))
    patch_choices = [value for value in (2, 4, 7, 8) if image_size % value == 0]
    kwargs = {
        "patchify_with_cnn": trial.suggest_categorical(
            "patchify_with_cnn", [False, True]
        ), 
        "patch_size": trial.suggest_categorical("patch_size", patch_choices), 
        "dim": dim, 
        "mha_num_heads": heads, 
        "vit_block_mlp_ratio": trial.suggest_categorical(
            "mlp_ratio", [1., 2., 4., 6.]
        ), 
        "drop_prob": trial.suggest_categorical(
            "drop_prob", [0., 0.05, 0.1, 0.2]
        ), 
        "use_refiner_cnn": trial.suggest_categorical(
            "use_refiner_cnn", [False, True]
        )
    }

    # Tune encoder and decoder depths independently for joint DiT models.
    if model_name in ("dit_encoder_decoder", "dit_encoder_decoder_classifier"):
        kwargs["depth"] = trial.suggest_categorical("encoder_depth", [2, 4, 6])
        use_refiner_cnn = kwargs.pop("use_refiner_cnn")
        kwargs["decoder_kwargs"] = {
            "depth": trial.suggest_categorical("decoder_depth", [1, 2, 4]), 
            "mha_num_heads": heads, 
            "vit_block_mlp_ratio": kwargs["vit_block_mlp_ratio"], 
            "shift_inputs": False, 
            "use_refiner_cnn": use_refiner_cnn
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
        kwargs["depth"] = trial.suggest_categorical(
            "depth", [2, 3, 4, 5, 6]
        )

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
    embedding_dim = trial.suggest_categorical(
        "embedding_dim", [64, 96, 128]
    )
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

    latent_dim = trial.suggest_categorical(
        "latent_dim", [8, 16, 32, 64, 128]
    )
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
    wrapper_kwargs: dict[str, object], 
    tune_masking: bool = True, 
    use_distillation: bool = False
) -> None:
    """Add joint-classification suggestions to mutable model settings.

    Args:
        trial (optuna.trial.Trial): Active Optuna trial.
        model_name (str): Selected joint model family.
        kwargs (dict[str, object]): Raw-model options updated in place.
        wrapper_kwargs (dict[str, object]): Wrapper options updated in place.
        tune_masking (bool): Suggest V1 classifier masking options when true.
        use_distillation (bool): Add a student distillation head and suggest
            teacher-loss settings for a runtime or continual snapshot teacher.

    Returns:
        None.
    """

    classifier_architecture = "linear"
    # Add DiT-specific architecture and conditioning choices.
    if model_name.startswith("dit"):
        classifier_architecture = trial.suggest_categorical(
            "classifier_architecture", [
                "linear", "local_mixer", "connection", "cross_attention", 
                "cross_attention_decoder", "cross_attention_aggregation", 
                "u_shape", "u_vae"
            ]
        )
        classifier_only_cls_token = trial.suggest_categorical(
            "classifier_only_cls_token", [True, False]
        )
        feature_aggregation = trial.suggest_categorical(
            "feature_aggregation", ["last", "all"]
        )
        # Preserve the existing classifier-depth search for the linear baseline.
        if classifier_architecture == "linear":
            clf_depth = trial.suggest_categorical(
                "clf_depth", [1, 2, 3, 4, 5]
            )
        # Use two stages for the compact mixer and routing templates.
        elif classifier_architecture in (
            "local_mixer", "connection", "cross_attention", 
            "cross_attention_decoder"
        ):
            clf_depth = 2
        # Cross-attention aggregation is applied at the first classifier stage.
        elif classifier_architecture == "cross_attention_aggregation":
            clf_depth = 1
        # Use a down/bottleneck/up classifier for the U-shaped template.
        elif classifier_architecture == "u_shape":
            clf_depth = 3
        # Reserve two middle stages for the variational bottleneck.
        else:
            clf_depth = 5

        kwargs.update({
            "feature_aggregation_ids_dict": {
                1: [-1] if feature_aggregation == "last" else [None]
            }, 
            "classifier_only_cls_token": classifier_only_cls_token, 
            "clf_depth": clf_depth,
            "clf_ln_no_adaptation": trial.suggest_categorical(
                "clf_ln_no_adaptation", [True, False]
            ), 
            "clf_drop_prob": trial.suggest_categorical(
                "clf_drop_prob", [0., 0.1, 0.2, 0.25]
            ), 
            "classifier_mlp_ratio": trial.suggest_categorical(
                "classifier_mlp_ratio", [None, 1, 2]
            ), 
            "dropout_rate": trial.suggest_categorical(
                "dropout_rate", [0., 0.1, 0.2, 0.25, 0.5]
            )
        })

        # Add a spatially local residual mixer between transformer blocks.
        if classifier_architecture == "local_mixer":
            kwargs.update({
                "clf_local_mixer_ids": [1], 
                "clf_local_mixer_kwargs": {"pos_embed_type": None}
            })
        # Merge the classifier input and first-stage output before stage two.
        elif classifier_architecture == "connection":
            kwargs.update({
                "clf_connection_ids_dict": {2: [0, 1], -1: [-1]},
                "clf_connection_kwargs": {
                    "connect_type": trial.suggest_categorical(
                        "clf_connection_type", ["add", "concat"]
                    )
                }
            })
        # Route the classifier input through ordinary cross-attention.
        elif classifier_architecture in (
            "cross_attention", 
            "cross_attention_decoder"
        ):
            kwargs.update({
                "clf_cross_attention_ids_dict": {2: [0]}, 
                "clf_cross_attention_plug_type": trial.suggest_categorical(
                    "clf_cross_attention_plug_type", ["values", "queries"]
                )
            })
            # Select the existing decoder block for the decoder variant.
            if classifier_architecture == "cross_attention_decoder":
                kwargs["clf_use_decoder_ids"] = [2]
        # Cross-attend directly to the main transformer's input feature.
        elif classifier_architecture == "cross_attention_aggregation":
            kwargs.update({
                "cross_attention_aggregation_ids_dict": {1: [0]}, 
                "clf_cross_attention_plug_type": trial.suggest_categorical(
                    "clf_cross_attention_plug_type", ["values", "queries"]
                )
            })
        # Build a compact spatial downsample/bottleneck/upsample classifier.
        elif classifier_architecture == "u_shape":
            kwargs.update({
                "clf_vit_block_ids": [1, 2, 3], 
                "clf_downsample_ids": [1], 
                "clf_downsample_kwargs": {"scaling_method": "avg_pooling"}, 
                "clf_upsample_ids": [3], 
                "clf_upsample_kwargs": {
                    "scaling_method": "interpolate", 
                    "scaling_interpolation_method": "bilinear"
                }
            })
        # Insert a KL-enabled flatten/unflatten bottleneck into the U-shape.
        elif classifier_architecture == "u_vae":
            kwargs.update({
                "clf_vit_block_ids": [1, 4, 5], 
                "clf_downsample_ids": [1], 
                "clf_downsample_kwargs": {"scaling_method": "avg_pooling"}, 
                "clf_reshaper_ids_dict": {2: "flatten", 3: "unflatten"}, 
                "clf_reshaper_kwargs": {
                    "add_kl": True, 
                    "latent_dim_ratio": trial.suggest_categorical(
                        "clf_latent_dim_ratio", [0.125, 0.25, 0.5]
                    )
                }, 
                "clf_upsample_ids": [4], 
                "clf_upsample_kwargs": {
                    "scaling_method": "interpolate", 
                    "scaling_interpolation_method": "bilinear"
                }
            })

        # Project concatenated all-depth features back to the classifier width.
        if feature_aggregation == "all":
            kwargs.update({
                "clf_dim": kwargs["dim"], 
                "clf_dim_forced": True
            })

        # Tune a dedicated classifier token only when the branch uses one.
        if classifier_only_cls_token:
            kwargs["clf_cls_token_type"] = trial.suggest_categorical(
                "clf_cls_token_type", ["new_weight", "time_label", "label"]
            )

        # Weight the classifier's variational bottleneck only when it exists.
        if classifier_architecture == "u_vae":
            wrapper_kwargs["kl_loss_coef"] = trial.suggest_float(
                "kl_loss_coef", 1e-4, 1e-1, log=True
            )

    # Enable the family-specific student distillation head when requested.
    if use_distillation:
        # Give transformer classifiers a dedicated learned distillation token.
        if model_name.startswith("dit"):
            kwargs.update({
                "classifier_only_distil_token": True, 
                "clf_distil_token_type": "new_weight"
            })
        # Give the convolutional classifier its parallel distillation head.
        elif model_name == "unet_classifier":
            kwargs["classifier_only_distil_token"] = True

        wrapper_kwargs.update({
            "distil_type": trial.suggest_categorical(
                "distil_type", ["hard", "soft"]
            ), 
            "distil_loss_coef": trial.suggest_float(
                "distil_loss_coef", 1e-4, 1e-1, log=True
            )
        })

    ctr_loss_coef = trial.suggest_categorical(
        "ctr_loss_coef", [0., 1e-4, 1e-3, 1e-2, 5e-2]
    )
    wrapper_kwargs["ctr_loss_coef"] = ctr_loss_coef
    # Materialize one final classifier regularizer when its loss is active.
    if ctr_loss_coef > 0.:
        regularizer_train_type = "normal"
        # Let a teacher supervise the auxiliary regularizer when available.
        if use_distillation:
            regularizer_train_type = trial.suggest_categorical(
                "regularizer_train_type", ["normal", "distil", "both"]
            )
        regularizer_distil_type = "hard"
        # Tune hard/soft regularizer targets only for teacher-backed modes.
        if regularizer_train_type in ("distil", "both"):
            regularizer_distil_type = trial.suggest_categorical(
                "regularizer_distil_type", ["hard", "soft"]
            )

        kwargs["clf_cls_token_regularizer_ids"] = [kwargs["clf_depth"]]
        regularizer_kwargs = {
            "start": (
                0 if kwargs.get("classifier_only_cls_token", False)
                else int(use_distillation)
            ),
            "end": (
                1 if kwargs.get("classifier_only_cls_token", False)
                else int(use_distillation) + 1
            ),
            "train_type": regularizer_train_type,
            "distil_type": regularizer_distil_type,
        }
        # DiT classifiers expose a classifier-specific regularizer mapping.
        if model_name.startswith("dit"):
            kwargs["clf_cls_token_regularizer_kwargs"] = regularizer_kwargs
        # U-Net classifiers reuse the inherited regularizer metadata mapping.
        else:
            kwargs["cls_token_regularizer_kwargs"] = regularizer_kwargs

    active_accuracy_heads = 1 + int(use_distillation) + int(
        ctr_loss_coef > 0.
    )
    accuracy_coef = 1. / active_accuracy_heads
    wrapper_kwargs.update({
        "clf_acc_coef": accuracy_coef, 
        "distil_acc_coef": accuracy_coef if use_distillation else 0., 
        "ctr_acc_coef": accuracy_coef if ctr_loss_coef > 0. else 0.
    })

    wrapper_kwargs["clf_loss_coef"] = trial.suggest_float(
        "clf_loss_coef", 1e-4, 1e-1, log=True
    )

    # Tune classifier masking only for the jointly trained V1 wrapper.
    if tune_masking:
        masking = trial.suggest_categorical(
            "masking", ["null", "timestep", "both", "neither"]
        )
        wrapper_kwargs.update({
            "mask_by_nulls": masking in ("null", "both"), 
            "mask_by_t_threshold": masking in ("timestep", "both")
        })

        # Tune timestep masking only for modes that use it.
        if masking in ("timestep", "both"):
            wrapper_kwargs["mask_t_percentage"] = trial.suggest_categorical(
                "mask_t", [35, 50, 70, 90]
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
        "dropout_rate": trial.suggest_float(
            "dropout", 0., 0.5, step=0.1
        ), 
        "num_last_not_frozen": trial.suggest_categorical(
            "unfrozen", [1, 5, 20, None]
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
    ensemble_accuracy_kwargs: Mapping[str, object] | None = None, 
    use_distillation: bool = False, 
    fit_method: str = "fit", 
    fit_kwargs: Mapping[str, object] | None = None
) -> Config:
    """Build one complete, shape-compatible trial configuration.

    Args:
        trial (optuna.trial.Trial): Active Optuna trial.
        task (str): Generation, joint, classification, or continual task.
        model_name (str): Selected model family.
        dataset_name (str): Supported dataset name.
        epochs (int): Ordinary-fit budget and cosine-schedule sizing value.
            Progressive diffusion phases use their stage/final budgets.
        seed (int): Trial-specific random seed.
        results_path (str | pathlib.Path): HPO artifact root.
        use_ensemble_accuracy (bool): Use post-training ensemble accuracy as
            the classification objective for diffusion-classifier studies.
        ensemble_accuracy_kwargs (Mapping[str, object] | None): Options passed
            to ``DiffusionClassifier.evaluate_ensemble_accuracy``.
        use_distillation (bool): Configure a student distillation head and
            teacher objective. Continual trials create teachers from completed
            task snapshots; joint trials receive a runtime teacher separately.
        fit_method (str): ``"fit"`` for ordinary Keras training or
            ``"fit_progressively"`` for a diffusion curriculum.
        fit_kwargs (Mapping[str, object] | None): Selected-method arguments.
            Progressive named arguments are stored in their explicit
            :class:`TrainingConfig` fields; remaining Keras fit arguments are
            stored in ``TrainingConfig.fit_kwargs``.

    Returns:
        Config: Fully typed trial configuration.

    Raises:
        ValueError: If the dataset/model combination or fit selection is
            unsupported, or progressive training omits ``stage_tasks``.
    """

    dataset_name = dataset_name.lower()
    fit_kwargs = dict(fit_kwargs or {})

    # Reject datasets outside the four supported HPO families.
    if dataset_name not in ("fmnist", "mnist", "cifar10", "cifar100"):
        raise ValueError("dataset_name must be FMNIST, MNIST, CIFAR10, or CIFAR100.")
    _validate_fit_request(model_name, fit_method, fit_kwargs)
    # Restrict pretrained Xception search to three-channel CIFAR inputs.
    if model_name == "pretrained" and dataset_name not in ("cifar10", "cifar100"):
        raise ValueError("The Xception classifier requires three-channel CIFAR data.")
    # Restrict ensemble feedback to studies backed by classifier diffusion wrappers.
    if use_ensemble_accuracy and not (
        task in ("joint", "continual")
        and model_name in _DIFFUSION_CLASSIFIER_MODELS
    ):
        raise ValueError(
            "use_ensemble_accuracy requires a joint or "
            "continual diffusion classifier study."
        )

    ensemble_accuracy_kwargs = dict(ensemble_accuracy_kwargs or {})

    progressive_fields = {}
    # Move named curriculum controls into their typed training-config fields.
    if fit_method == "fit_progressively":
        for name in (
            "stage_tasks", 
            "stages_num", 
            "stages_verbose", 
            "stage_epochs", 
            "final_epochs", 
            "timestep_boundaries", 
            "timestep_clustering_type", 
            "resolutions", 
            "depths", 
            "pacing_type", 
            "earlystopping_type", 
            "min_delta", 
            "stopper_mode"
        ):
            # Preserve TrainingConfig defaults for omitted progressive values.
            if name in fit_kwargs:
                progressive_fields[name] = fit_kwargs.pop(name)

        # Avoid colliding with ordinary orchestration early-stopping settings.
        if "monitor" in fit_kwargs:
            progressive_fields["progressive_monitor"] = fit_kwargs.pop("monitor")
        # Keep stage pacing patience independent from ordinary fit patience.
        if "patience" in fit_kwargs:
            progressive_fields["progressive_patience"] = fit_kwargs.pop("patience")

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
    loss_function = "mse"

    # Tune the shared generative loss only for diffusion and VAE families.
    if model_name in _DIFFUSION_MODELS \
    or model_name in ("vae", "vae_classifier"):
        loss_function = trial.suggest_categorical(
            "loss_function", ["mse", "mae"]
        )

    # Restrict teacher objectives to classifier-capable diffusion studies.
    if use_distillation and not (
        task in ("joint", "continual")
        and model_name in _DIFFUSION_CLASSIFIER_MODELS
    ):
        raise ValueError(
            "Distillation requires a joint or "
            "continual diffusion classifier study."
        )

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
        wrapper_name = "diffusion_classifier"
        # Compare the two existing wrappers for the DiT classifier family.
        if model_name == "dit_classifier":
            wrapper_name = trial.suggest_categorical("wrapper_name", [
                "diffusion_classifier", "diffusion_classifier_v2"
            ])

        _suggest_joint(
            trial, 
            model_name, 
            model_kwargs, 
            wrapper_kwargs, 
            tune_masking=wrapper_name == "diffusion_classifier",
            use_distillation=use_distillation,
        )
        # Tune V2-only classifier-input noising for the DiT classifier family.
        if model_name == "dit_classifier" \
        and wrapper_name == "diffusion_classifier_v2":
            clf_max_timesteps = trial.suggest_categorical(
                "clf_train_noisified_max_timesteps", 
                [None, 64, 128, 256, 512, "timesteps"]
            )
            wrapper_kwargs["clf_train_noisified_max_timesteps"] = (
                None if clf_max_timesteps is None 
                else timesteps if clf_max_timesteps == "timesteps" 
                else min(clf_max_timesteps, timesteps)
            )
            embedding_recipe = trial.suggest_categorical(
                "clf_vars_embedding_recipe", [
                    "none", "label", "conditions", "core", "notebook"
                ]
            )
            noise_recipe = trial.suggest_categorical(
                "clf_vars_noise_recipe", [
                    "none", "first", "last", "last_two"
                ]
            )
            wrapper_kwargs["clf_vars_embedding_ids"] = {
                "none": [],
                "label": [2],
                "conditions": [1, 2],
                "core": [0, 1, 2],
                "notebook": [0, 1, 2, 3],
            }[embedding_recipe]
            wrapper_kwargs["clf_vars_noise_part_ids"] = {
                "none": [],
                "first": [1],
                "last": [-1],
                "last_two": [-2, -1],
            }[noise_recipe]

    continual_kwargs = {}
    # Tune replay policy only for continual-learning studies.
    if task == "continual":
        replay_samples = trial.suggest_categorical(
            "replay_samples", [100, 500, 1_000, 2_500, 5_000]
        )
        train_num = trial.suggest_categorical(
            "train_num", [-1, 1_000, 2_500, 5_000, 7_500, 10_000]
        )
        continual_kwargs = {
            "remove_prev_classes": True, 
            "keep_same_model": True, 
            "use_distillation": use_distillation,
            "plot_results": False, 
            "generative_model_kwargs": {
                "samples_per_class": replay_samples, 
                "train_num": train_num
            }
        }

        # Train and evaluate every diffusion classifier's attached head.
        if model_name in _DIFFUSION_CLASSIFIER_MODELS:
            continual_kwargs.update({
                "use_generative_model_classifier": True, 
                "train_classifier_separately": (
                    wrapper_name == "diffusion_classifier_v2"
                )
            })

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
                    "activation": "relu"
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
                    "conv_depths": (1, 1, 1)
                }
            }

    # Optimize validation accuracy for standalone classification.
    if task == "classification":
        monitor, monitor_mode = "val_accuracy", "max"
    # Optimize both generation loss and classification accuracy jointly.
    elif task == "joint":
        monitor = "val_clf_accuracy" if model_name == "vae_classifier" \
                else "val_total_accuracy" if use_distillation or \
                    wrapper_kwargs.get("ctr_loss_coef", 0.) > 0. \
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
    task_tag = {
        "generation": "g", 
        "joint": "j", 
        "classification": "c", 
        "continual": "l"
    }[task]
    dataset_tag = {
        "mnist": "m", "fmnist": "fm",
        "cifar10": "c10", "cifar100": "c100"
    }[dataset_name]
    project_tag = f"t{trial.number:04d}"
    trial_root = Path(results_path) / task / model_name / dataset_name

    # Match the separate study storage used for ensemble-feedback trials.
    if use_ensemble_accuracy:
        trial_root /= "ensemble_accuracy"
    # Keep teacher-distillation trials separate from ordinary studies.
    if use_distillation:
        trial_root /= "distillation"
    # Keep progressive trials out of resumable ordinary-fit studies.
    if fit_method == "fit_progressively":
        trial_root /= "fit_progressively"

    tensorboard_root = Path(results_path) / "_tb" / (
        task_tag + _MODEL_TAGS[model_name] + dataset_tag
    )
    # Avoid TensorBoard event-name collisions with ordinary-accuracy trials.
    if use_ensemble_accuracy:
        tensorboard_root /= "ensemble_accuracy"
    # Keep distillation TensorBoard events in their matching study family.
    if use_distillation:
        tensorboard_root /= "distillation"
    # Keep progressive TensorBoard runs beside their separate study.
    if fit_method == "fit_progressively":
        tensorboard_root /= "fit_progressively"

    config = Config(
        dataset={
            "name": dataset_name, 
            "batch_size": optimization["batch_size"], 
            "preprocess": preprocess, 
            "features_path": features_path, 
            "return_features": return_features, 
            "onehot_labels": onehot_labels,
            "validation_ratio": 0.2,
        }, 
        model={
            "name": model_name, 
            "wrapper_name": wrapper_name, 
            "kwargs": model_kwargs, 
            "wrapper_kwargs": wrapper_kwargs, 
            "classifier_name": classifier_name, 
            "classifier_kwargs": classifier_kwargs, 
            "loss_function": loss_function, 
            "show_network_summary": False
        }, 
        optimizer=optimization["optimizer"], 
        training={
            "task": task, 
            "epochs": epochs, 
            "fit_method": fit_method,
            "fit_kwargs": fit_kwargs,
            **progressive_fields,
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
            "ensemble_accuracy_kwargs": ensemble_accuracy_kwargs, 
            "use_distillation": use_distillation
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
                "val_total_accuracy", 
                "val_classifier_accuracy", 
                "val_cls_token_accuracy", 
                "val_avg_pooling_accuracy", 
                "val_clf_accuracy", 
                "total_accuracy", 
                "classifier_accuracy", 
                "cls_token_accuracy", 
                "avg_pooling_accuracy", 
                "clf_accuracy"
            ])

        return generation, classification

    # Return only classifier accuracy for classification studies.
    if task == "classification":
        return _history_value(
            history, 
            ["val_accuracy", "accuracy"], 
            best="max"
        )

    # Continual ensemble scores are currently test-only and cannot tune HPO.
    if use_ensemble_accuracy:
        raise ValueError(
            "Continual ensemble HPO requires a validation-ensemble metric; "
            "test ensemble accuracy is reserved for final evaluation."
        )

    validation_values = np.asarray(
        history.get("task_val_accuracy", []),
        dtype="float64",
    )
    # Never substitute cumulative test accuracy for missing validation feedback.
    if validation_values.size == 0 or not np.any(np.isfinite(validation_values)):
        raise ValueError(
            "Continual HPO requires an explicit validation split with "
            "task_val_accuracy."
        )

    return float(np.nanmean(validation_values))


def run_hpo(
    task: str, 
    model_name: str, 
    dataset_name: str = "CIFAR10", 
    n_trials: int = 30, 
    epochs: int = 30, 
    seed: int = 42, 
    results_path: str = "results/hpo", 
    timeout: float | None = None,
    use_ensemble_accuracy: bool = False,
    ensemble_accuracy_kwargs: Mapping[str, object] | None = None, 
    fit_method: str = "fit", 
    fit_kwargs: Mapping[str, object] | None = None, 
    teacher_network: tf.keras.Model | None = None,
    use_distillation: bool = False
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
        epochs (int): Positive ordinary-fit budget. Progressive diffusion
            phases use ``stage_epochs`` and ``final_epochs`` instead; this
            value still sizes existing cosine schedules and any ordinary
            continual classifier phase.
        seed (int): Reproducible TPE and first-trial seed.
        results_path (str): HPO root. Study state is written below
            ``<task>/<model>/<dataset>`` and TensorBoard events below ``_tb``.
        timeout (float | None): Optional study wall-time limit in seconds.
        use_ensemble_accuracy (bool): Use post-training ensemble accuracy for
            joint diffusion-classifier studies. Continual ensemble HPO is
            rejected until a validation-only ensemble metric is available.
        ensemble_accuracy_kwargs (Mapping[str, object] | None): Options passed
            to ``DiffusionClassifier.evaluate_ensemble_accuracy``.
        teacher_network (tf.keras.Model | None): Runtime-only frozen teacher.
            Supplying it enables hard/soft distillation suggestions and gives
            continual task one an optional initial teacher. The object is never
            written to trial YAML.
        use_distillation (bool): Enable the distillation search space. Without
            ``teacher_network`` this is supported for continual diffusion
            classifiers, whose later tasks use the immediately preceding raw
            student snapshot as their teacher.
        fit_method (str): ``"fit"`` for the ordinary epoch loop or
            ``"fit_progressively"`` for diffusion curriculum training.
        fit_kwargs (Mapping[str, object] | None): YAML-safe arguments for the
            selected method. Progressive use requires ``stage_tasks`` and
            accepts the named curriculum controls plus ordinary variadic
            Keras fit keys.

    Returns:
        optuna.study.Study: Resumable completed/partial study. Joint studies
        expose ``best_trials``; other studies expose ``best_trial``.

    Raises:
        ImportError: If Optuna is unavailable.
        ValueError: If the task/model pair, fit selection, or positive budgets
            are invalid, or progressive training omits ``stage_tasks``.
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
    fit_kwargs = dict(fit_kwargs or {})
    effective_distillation = bool(
        use_distillation or teacher_network is not None
    )

    # Require a supported task/model search-space pairing.
    if task not in SEARCH_SPACES or model_name not in SEARCH_SPACES[task]:
        raise ValueError(f"Unsupported task/model pair: {task}/{model_name}")
    _validate_fit_request(model_name, fit_method, fit_kwargs)
    # Restrict ensemble feedback to supported diffusion-classifier studies.
    if use_ensemble_accuracy and not (
        task in ("joint", "continual")
        and model_name in _DIFFUSION_CLASSIFIER_MODELS
    ):
        raise ValueError(
            "use_ensemble_accuracy requires a joint "
            "or continual diffusion classifier study."
        )
    # Continual ensemble reports currently evaluate test data, never validation.
    if use_ensemble_accuracy and task == "continual":
        raise ValueError(
            "Continual ensemble HPO requires a validation-ensemble metric; "
            "test ensemble accuracy is reserved for final evaluation."
        )
    # Restrict runtime teachers to supported distillation study families.
    if teacher_network is not None and not (
        task in ("joint", "continual")
        and model_name in _DIFFUSION_CLASSIFIER_MODELS
    ):
        raise ValueError(
            "teacher_network requires a joint or "
            "continual diffusion classifier study."
        )
    # Teacher-free distillation relies on the continual previous-task snapshot.
    if use_distillation and teacher_network is None and not (
        task == "continual"
        and model_name in _DIFFUSION_CLASSIFIER_MODELS
    ):
        raise ValueError(
            "Teacher-free use_distillation requires a continual diffusion "
            "classifier study."
        )
    # Require positive trial and epoch budgets.
    if n_trials <= 0 or epochs <= 0:
        raise ValueError("n_trials and epochs must be positive.")

    root = Path(results_path)
    study_root = root / task / model_name / dataset_name.lower()
    # Keep ensemble-feedback trials separate from legacy accuracy studies.
    if use_ensemble_accuracy:
        study_root /= "ensemble_accuracy"
    # Isolate the conditional distillation search space from ordinary trials.
    if effective_distillation:
        study_root /= "distillation"
    # Keep progressive trials separate from resumable ordinary-fit studies.
    if fit_method == "fit_progressively":
        study_root /= "fit_progressively"

    configs_path = study_root / "configs"
    configs_path.mkdir(parents=True, exist_ok=True)
    storage_path = (study_root / "study.db").resolve().as_posix()
    study_name = f"{task}-{model_name}-{dataset_name.lower()}" + (
        "-ensemble-accuracy" if use_ensemble_accuracy else ""
    )
    # Give distillation trials an independent persistent study identity.
    if effective_distillation:
        study_name += "-distillation"
    # Give progressive studies an independent SQLite study identity.
    if fit_method == "fit_progressively":
        study_name += "-fit-progressively"
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
            ensemble_accuracy_kwargs=ensemble_accuracy_kwargs, 
            use_distillation=effective_distillation,
            fit_method=fit_method, 
            fit_kwargs=fit_kwargs
        )

        input_config_path = configs_path / f"trial-{trial.number:04d}.yaml"
        save_config(config, input_config_path)
        config = load_config(input_config_path)

        result = main(config, teacher_network=teacher_network)
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
