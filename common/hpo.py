"""Optuna studies that reuse the project's config-driven training pipeline.

Searches deliberately use a few validated architecture templates instead of
sampling arbitrary routing dictionaries. Every trial writes a YAML config,
reloads it with :func:`common.config.load_config`, and runs
:func:`common.train.main`.
"""

from __future__ import annotations

import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import gc
import hashlib
import json
import math
import os
import uuid

import re

from collections.abc import Mapping, Sequence
from typing import Any

from common.config import (
    Config, 
    load_config, 
    normalize_training_task, 
    resolve_continual_schedule, 
    save_config
)
from common.dataloader import get_dataset_spec
from common.recovery import find_latest_task_checkpoint, fingerprint_state
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
_RECOVERY_ENQUEUED_ATTR = "recovery_enqueued_trial_numbers"
"""Study user-attribute key recording source trials queued for recovery."""

_STUDY_SPEC_ATTR = "study_spec"
_STUDY_SPEC_FINGERPRINT_ATTR = "study_spec_fingerprint"
_SAMPLER_RNG_STATE_ATTR = "sampler_rng_state"
_STUDY_SPEC_FILE = "study_spec.json"
_FIT_METHODS = {"fit", "fit_progressively"}


def _validate_fit_request(
    model_name: str, 
    fit_method: str, 
    fit_kwargs: Mapping[str, object]
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
        "all": "a", "new_weight": "nw",
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
    trial: Any
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


def _default_objective_metrics(
    task: str, 
    use_ensemble_accuracy: bool = False
) -> tuple[str, ...]:
    """Return stable, human-readable metric names for a study task.

    Args:
        task (str): Normalized HPO task name.
        use_ensemble_accuracy (bool): Whether ensemble accuracy names the joint
            classification objective.

    Returns:
        tuple[str, ...]: Default metric names in optimization order.
    """

    # Name the legacy scalar generation target.
    if task == "generation":
        return ("generation_loss",)
    # Name both legacy joint Pareto targets.
    if task == "joint":
        return (
            "generation_loss",
            "ensemble_accuracy" if use_ensemble_accuracy
            else "classification_accuracy",
        )
    # Name the legacy standalone validation target.
    if task == "classification":
        return ("validation_accuracy",)
    # Name the validation-matrix aggregate used by continual studies.
    if task == "continual":
        return ("final_average_accuracy",)
    raise ValueError(f"Unsupported HPO task: {task}")


def _inferred_objective_direction(metric_name: str) -> str:
    """Infer a conservative Optuna direction from a metric's name.

    Args:
        metric_name (str): Configured objective name.

    Returns:
        str: ``"minimize"`` for cost-like metrics, otherwise ``"maximize"``.
    """

    normalized = metric_name.lower()
    # Minimize conventional cost, error, resource, and forgetting names.
    if any(token in normalized for token in (
        "loss", "error", "forgetting", "latency", "memory",
    )):
        return "minimize"
    return "maximize"


def _normalize_objective_spec(
    task: str, 
    objective_metrics: str | Sequence[str] | None = None, 
    objective_directions: str | Sequence[str] | None = None, 
    use_ensemble_accuracy: bool = False
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Normalize objective names/directions and validate their cardinality.

    Args:
        task (str): Normalized HPO task name.
        objective_metrics (str | Sequence[str] | None): Optional metric name or
            ordered names.
        objective_directions (str | Sequence[str] | None): Optional matching
            Optuna directions.
        use_ensemble_accuracy (bool): Select the ensemble joint default name.

    Returns:
        tuple[tuple[str, ...], tuple[str, ...]]: Normalized metric names and
        one matching direction per name.

    Raises:
        TypeError: If either specification has an unsupported container type.
        ValueError: If names are empty/duplicated or directions are invalid.
    """

    # Supply task defaults when the caller did not name objectives.
    if objective_metrics is None:
        metrics = _default_objective_metrics(task, use_ensemble_accuracy)
    # Treat one string as one objective rather than a character sequence.
    elif isinstance(objective_metrics, str):
        metrics = (objective_metrics,)
    # Materialize an ordered caller-provided metric sequence.
    elif isinstance(objective_metrics, Sequence):
        metrics = tuple(objective_metrics)
    # Reject mappings, scalars, and other ambiguous containers.
    else:
        raise TypeError("objective_metrics must be a string or sequence of strings.")

    # Reject empty specifications and blank/non-string metric names.
    if not metrics or any(not isinstance(name, str) or not name.strip()
                          for name in metrics):
        raise ValueError("objective_metrics must contain nonempty strings.")
    metrics = tuple(name.strip() for name in metrics)
    # Prevent ambiguous duplicate objective columns and Optuna dimensions.
    if len(set(metrics)) != len(metrics):
        raise ValueError("objective_metrics must not contain duplicate names.")

    # Infer directions only when the caller omitted them.
    if objective_directions is None:
        # Preserve the established joint minimize/maximize ordering.
        if objective_metrics is None and task == "joint":
            directions = ("minimize", "maximize")
        # Preserve the established scalar generation direction.
        elif objective_metrics is None and task == "generation":
            directions = ("minimize",)
        # Infer custom and accuracy-family directions from each name.
        else:
            directions = tuple(
                _inferred_objective_direction(name) for name in metrics
            )
    # Treat one direction string as one dimension.
    elif isinstance(objective_directions, str):
        directions = (objective_directions,)
    # Materialize an ordered caller-provided direction sequence.
    elif isinstance(objective_directions, Sequence):
        directions = tuple(objective_directions)
    # Reject mappings, scalars, and other ambiguous containers.
    else:
        raise TypeError(
            "objective_directions must be a string or sequence of strings."
        )

    directions = tuple(
        str(direction).strip().lower() for direction in directions
    )
    # Require Optuna directions to align exactly with objective dimensions.
    if len(directions) != len(metrics):
        raise ValueError(
            "objective_directions must have one entry per objective metric."
        )
    invalid = sorted(set(directions) - {"minimize", "maximize"})
    # Reject spellings that Optuna cannot interpret.
    if invalid:
        raise ValueError(
            "objective_directions entries must be 'minimize' or 'maximize': "
            + ", ".join(invalid)
        )
    return metrics, directions


def _study_json_sort_key(value: object) -> str:
    """Return a canonical sort key for an already JSON-safe study value.

    Args:
        value (object): JSON-safe metadata value.

    Returns:
        str: Compact key-sorted JSON representation.
    """

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _study_json_value(value: object) -> object:
    """Convert HPO inputs to deterministic JSON-safe study metadata.

    Args:
        value (object): Scientific study option to normalize.

    Returns:
        object: Deterministic JSON-safe representation.

    Raises:
        TypeError: If no stable serialization exists for ``value``.
        ValueError: If a non-finite numeric value is supplied.
    """

    # Preserve ordinary JSON scalar values directly.
    if value is None or isinstance(value, (bool, str, int)):
        return value
    # Require finite floats because strict study metadata uses standard JSON.
    if isinstance(value, float):
        # Reject NaN and infinities instead of producing nonportable JSON.
        if not math.isfinite(value):
            raise ValueError("HPO study metadata cannot contain non-finite values.")
        return value
    # Convert NumPy scalar wrappers to their corresponding Python values.
    if isinstance(value, np.generic):
        return _study_json_value(value.item())
    # Represent array content compactly without embedding large payloads.
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        # Reject Python-object pointers that have no stable byte representation.
        if array.dtype.hasobject:
            raise TypeError("Object arrays cannot enter HPO study metadata.")
        return {
            "array_shape": list(array.shape),
            "array_dtype": array.dtype.str,
            "array_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    # Normalize explicit pathlib inputs to ordinary path strings.
    if isinstance(value, Path):
        return str(value)
    # Preserve mappings in canonical key order.
    if isinstance(value, Mapping):
        # Use a normal JSON object when keys already satisfy JSON semantics.
        if all(isinstance(key, str) for key in value):
            return {
                key: _study_json_value(value[key]) for key in sorted(value)
            }
        entries = [
            {
                "key": _study_json_value(key),
                "value": _study_json_value(item),
            }
            for key, item in value.items()
        ]
        entries.sort(key=_study_json_sort_key)
        return {"mapping": entries}
    # Preserve ordered sequences while normalizing each member.
    if isinstance(value, (list, tuple)):
        return [_study_json_value(item) for item in value]
    # Sort unordered collections by their canonical JSON representation.
    if isinstance(value, (set, frozenset)):
        items = [_study_json_value(item) for item in value]
        items.sort(key=_study_json_sort_key)
        return {"set": items}
    # Serialize TensorFlow numeric dtypes by their registered names.
    if isinstance(value, tf.dtypes.DType):
        return {"dtype": value.name}
    get_config = getattr(value, "get_config", None)
    # Capture the semantic config of Keras-compatible objects.
    if callable(get_config):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "config": _study_json_value(get_config()),
        }
    # Identify plain callables by definition instead of process-local repr.
    if callable(value):
        return {
            "callable": (
                f"{getattr(value, '__module__', type(value).__module__)}."
                f"{getattr(value, '__qualname__', type(value).__qualname__)}"
            )
        }
    raise TypeError(
        "Unsupported HPO study metadata type: " + type(value).__qualname__
    )


def _teacher_study_signature(model: object | None) -> object:
    """Fingerprint a runtime teacher that intentionally stays out of YAML.

    Args:
        model (object | None): Optional frozen Keras teacher.

    Returns:
        object: JSON-safe teacher config/weight signature, or ``None``.

    Raises:
        TypeError: If the teacher exposes no serializable configuration.
    """

    # Distinguish teacher-free self-distillation from an external teacher.
    if model is None:
        return None
    get_config = getattr(model, "get_config", None)
    # Require a reconstructable topology for immutable study identity.
    if not callable(get_config):
        raise TypeError("teacher_network must expose a serializable get_config().")
    weight_descriptors = []
    for weight in list(getattr(model, "weights", []) or []):
        array = np.ascontiguousarray(weight.numpy())
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        weight_descriptors.append({
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "sha256": digest,
        })
    return {
        "type": f"{type(model).__module__}.{type(model).__qualname__}",
        "config": _study_json_value(get_config()),
        "weights": weight_descriptors,
    }


def _make_study_spec(
    *, 
    study_name: str, 
    task: str, 
    model_name: str, 
    dataset_name: str, 
    epochs: int, 
    seed: int, 
    use_ensemble_accuracy: bool, 
    ensemble_accuracy_kwargs: Mapping[str, object] | None, 
    fit_method: str, 
    fit_kwargs: Mapping[str, object], 
    teacher_network: object | None, 
    effective_distillation: bool, 
    objective_metrics: Sequence[str], 
    objective_directions: Sequence[str], 
    dtype_policy: str, 
    deterministic_ops: bool, 
    snapshot_network_name: str, 
    class_num: int | None, 
    class_order: Sequence[int] | None, 
    task_groups: Sequence[Sequence[int]] | None, 
    task_size: int, 
    class_order_mode: str, 
    task_order_mode: str
) -> dict[str, object]:
    """Build the immutable scientific identity of a persistent HPO study.

    Args:
        study_name (str): Exact Optuna storage identity.
        task (str): Training task family.
        model_name (str): Searched model family.
        dataset_name (str): Dataset selector.
        epochs (int): Ordinary training epoch budget.
        seed (int): Study, trial, and sampler seed.
        use_ensemble_accuracy (bool): Whether ensemble scores are authoritative.
        ensemble_accuracy_kwargs (Mapping[str, object] | None): Ensemble options.
        fit_method (str): Ordinary or progressive fit selector.
        fit_kwargs (Mapping[str, object]): Selected fit-method arguments.
        teacher_network (object | None): Optional external distillation teacher.
        effective_distillation (bool): Whether the distillation space is active.
        objective_metrics (Sequence[str]): Ordered objective metric names.
        objective_directions (Sequence[str]): Matching Optuna directions.
        dtype_policy (str): Keras numerical policy.
        deterministic_ops (bool): Whether deterministic kernels are requested.
        snapshot_network_name (str): Raw/EMA continual teacher branch.
        class_num (int | None): Selected continual class count.
        class_order (Sequence[int] | None): Requested continual class order.
        task_groups (Sequence[Sequence[int]] | None): Requested task groups.
        task_size (int): Automatic continual task width.
        class_order_mode (str): Fixed or seeded-random class ordering mode.
        task_order_mode (str): Fixed or seeded-random whole-task ordering mode.

    Returns:
        dict[str, object]: Strict JSON-safe immutable study specification.
    """

    return _study_json_value({
        "schema_version": 1, 
        "study_name": study_name, 
        "task": task, 
        "model_name": model_name, 
        "dataset_name": dataset_name.lower(), 
        "epochs": int(epochs), 
        "seed": int(seed), 
        "use_ensemble_accuracy": bool(use_ensemble_accuracy), 
        "ensemble_accuracy_kwargs": dict(ensemble_accuracy_kwargs or {}), 
        "fit_method": fit_method, 
        "fit_kwargs": dict(fit_kwargs), 
        "effective_distillation": bool(effective_distillation), 
        "teacher": _teacher_study_signature(teacher_network), 
        "objective_metrics": list(objective_metrics), 
        "objective_directions": list(objective_directions), 
        "dtype_policy": dtype_policy, 
        "deterministic_ops": bool(deterministic_ops), 
        "snapshot_network_name": snapshot_network_name, 
        "continual_schedule": {
            "class_num": class_num, 
            "class_order": None if class_order is None else list(class_order), 
            "task_groups": None if task_groups is None else [
                list(group) for group in task_groups
            ], 
            "task_size": task_size, 
            "class_order_mode": class_order_mode, 
            "task_order_mode": task_order_mode
        }
    })


def _read_study_spec(study_root: Path) -> dict[str, object]:
    """Read and self-validate the pre-study identity file.

    Args:
        study_root (pathlib.Path): Persistent HPO study directory.

    Returns:
        dict[str, object]: Authenticated immutable study specification.

    Raises:
        FileNotFoundError: If no identity sidecar exists.
        ValueError: If its structure or checksum is invalid.
    """

    path = study_root / _STUDY_SPEC_FILE
    # Require metadata that can block identity mismatches before Optuna loads.
    if not path.is_file():
        raise FileNotFoundError(
            "HPO resume study root has no study_spec.json: " + str(study_root)
        )

    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    # Require the documented envelope and mapping-shaped specification.
    if not isinstance(payload, dict) or not isinstance(payload.get("spec"), dict):
        raise ValueError("HPO study specification file is invalid.")
    # Authenticate the sidecar against its stable specification hash.
    if payload.get("fingerprint") != fingerprint_state(payload["spec"]):
        raise ValueError("HPO study specification checksum is invalid.")

    return payload["spec"]


def _write_study_spec(study_root: Path, spec: Mapping[str, object]) -> None:
    """Atomically persist identity before creating or loading Optuna.

    Args:
        study_root (pathlib.Path): Persistent HPO study directory.
        spec (Mapping[str, object]): JSON-safe immutable study specification.

    Returns:
        None: The checksummed sidecar is atomically replaced on success.
    """

    study_root.mkdir(parents=True, exist_ok=True)
    path = study_root / _STUDY_SPEC_FILE
    payload = {
        "spec": dict(spec), 
        "fingerprint": fingerprint_state(spec)
    }
    temporary = study_root / f".{_STUDY_SPEC_FILE}.tmp-{uuid.uuid4().hex}"

    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload, 
                stream, 
                ensure_ascii=False, 
                allow_nan=False, 
                sort_keys=True, 
                separators=(",", ":")
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        # Remove only this call's unique temporary file after a failed replace.
        if temporary.exists():
            temporary.unlink()


def _sampler_random_states(sampler: object) -> tuple[object, object]:
    """Return Optuna 3.6 TPE's two NumPy RandomState instances.

    Args:
        sampler (object): Optuna TPESampler instance.

    Returns:
        tuple[object, object]: TPE and fallback-random ``RandomState`` objects.

    Raises:
        RuntimeError: If sampler internals differ from the supported contract.
    """

    try:
        tpe_rng = sampler._rng.rng
        random_rng = sampler._random_sampler._rng.rng
    except AttributeError as error:
        raise RuntimeError(
            "This Optuna TPESampler version does not expose the expected "
            "recoverable RNG streams."
        ) from error
    # Fail explicitly when a future Optuna version changes RNG implementation.
    if not isinstance(tpe_rng, np.random.RandomState) \
    or not isinstance(random_rng, np.random.RandomState):
        raise RuntimeError(
            "Optuna TPESampler RNG internals are incompatible with recovery."
        )

    return tpe_rng, random_rng


def _random_state_payload(rng: np.random.RandomState) -> dict[str, object]:
    """Encode a NumPy RandomState tuple for Optuna user attributes.

    Args:
        rng (numpy.random.RandomState): Sampler stream to snapshot.

    Returns:
        dict[str, object]: JSON-safe complete MT19937 cursor/state.
    """

    name, keys, position, has_gauss, cached_gaussian = rng.get_state()

    return {
        "bit_generator": name, 
        "keys": keys.tolist(), 
        "position": int(position), 
        "has_gauss": int(has_gauss), 
        "cached_gaussian": float(cached_gaussian)
    }


def _capture_sampler_rng_state(sampler: object) -> dict[str, object]:
    """Capture both stochastic streams used by Optuna 3.6 TPESampler.

    Args:
        sampler (object): Optuna TPESampler instance.

    Returns:
        dict[str, object]: Versioned JSON-safe state for both RNG streams.
    """

    tpe_rng, random_rng = _sampler_random_states(sampler)

    return {
        "schema_version": 1, 
        "tpe": _random_state_payload(tpe_rng), 
        "random_sampler": _random_state_payload(random_rng)
    }


def _restore_sampler_rng_state(
    sampler: object, 
    state: Mapping[str, object]
) -> None:
    """Restore both TPESampler RandomState streams in place.

    Args:
        sampler (object): Optuna TPESampler instance to restore.
        state (Mapping[str, object]): State from
            :func:`_capture_sampler_rng_state`.

    Returns:
        None: Both internal streams are updated in place.

    Raises:
        ValueError: If the state schema or either stream is absent.
    """

    # Reject state created by an unknown future representation.
    if int(state.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported Optuna sampler RNG-state schema.")

    tpe_rng, random_rng = _sampler_random_states(sampler)
    for rng, name in ((tpe_rng, "tpe"), (random_rng, "random_sampler")):
        payload = state.get(name)
        # Require both independent streams for next-suggestion equivalence.
        if not isinstance(payload, Mapping):
            raise ValueError(f"Optuna sampler RNG state is missing {name!r}.")

        rng.set_state((
            str(payload["bit_generator"]),
            np.asarray(payload["keys"], dtype=np.uint32),
            int(payload["position"]),
            int(payload["has_gauss"]),
            float(payload["cached_gaussian"]),
        ))


def _has_committed_task_checkpoint(checkpoint_dir: Path) -> bool:
    """Report whether a trial checkpoint root has a committed task boundary.

    Args:
        checkpoint_dir (pathlib.Path): Per-trial continual checkpoint root.

    Returns:
        bool: Whether the root contains a fully validated committed task.
    """

    try:
        find_latest_task_checkpoint(checkpoint_dir)
    except (FileNotFoundError, OSError, TypeError, ValueError):
        # A marker alone is not durable evidence: validation checks the sealed
        # manifest, schedule, payload set, and every recorded checksum.
        return False

    return True


def _trial_checkpoint_dir(study_root: Path, trial: Any) -> Path:
    """Resolve and constrain the checkpoint root assigned to an Optuna trial.

    Args:
        study_root (pathlib.Path): Active persistent HPO study root.
        trial (optuna.trial.BaseTrial): Live or frozen trial carrying optional
            recovery user attributes.

    Returns:
        pathlib.Path: Resolved per-trial checkpoint directory below the study.

    Raises:
        ValueError: If persisted trial metadata points outside the study root.
    """

    user_attrs = dict(getattr(trial, "user_attrs", {}) or {})
    configured = user_attrs.get("resume_checkpoint_dir") \
                or user_attrs.get("checkpoint_dir")
    # Reuse the original checkpoint root for an automatically queued retry.
    if configured is not None:
        candidate = Path(str(configured)).resolve()
    # Assign a fresh stable directory to an ordinary trial.
    else:
        candidate = (
            study_root / "checkpoints" / f"trial-{trial.number:04d}"
        ).resolve()

    checkpoint_root = (study_root / "checkpoints").resolve()

    # Reject tampered SQLite metadata that escapes the selected study root.
    if candidate != checkpoint_root and checkpoint_root not in candidate.parents:
        raise ValueError(
            "Trial checkpoint directory must remain below the HPO study root."
        )

    return candidate


def _enqueue_recovery_trials(study: Any, study_root: Path) -> tuple[int, ...]:
    """Queue one parameter-identical retry for each recoverable trial.

    The source trial number is recorded both on the queued retry and in study
    metadata. A killed retry may itself be queued once on a later invocation,
    while the same dead source can never be enqueued repeatedly.

    Args:
        study (optuna.study.Study): Loaded persistent Optuna study.
        study_root (pathlib.Path): Root containing the study's checkpoints.

    Returns:
        tuple[int, ...]: Source trial numbers newly enqueued by this call.
    """

    trials = tuple(study.get_trials(deepcopy=False))
    tracked = {
        int(number)
        for number in study.user_attrs.get(_RECOVERY_ENQUEUED_ATTR, [])
    }
    # Treat already-persisted queued retries as authoritative deduplication data.
    for existing in trials:
        source = existing.user_attrs.get("resume_source_trial_number")
        # Record a retry even if a crash preceded the study-attribute update.
        if source is not None:
            tracked.add(int(source))

    enqueued: list[int] = []
    # Inspect persisted trials for killed runs or failed runs with a committed
    # task boundary. Failed trials without recovery state stay failed.
    for frozen in trials:
        state_name = str(getattr(frozen.state, "name", frozen.state)).upper()
        checkpoint_dir = _trial_checkpoint_dir(study_root, frozen)
        has_task_checkpoint = _has_committed_task_checkpoint(checkpoint_dir)
        recoverable = state_name == "RUNNING" or (
            state_name == "FAIL" and has_task_checkpoint
        )
        # Leave completed, pruned, waiting, and unrecoverable failed trials.
        if not recoverable:
            continue
        source_number = int(frozen.number)
        # Never queue the same abandoned source trial more than once.
        if source_number in tracked:
            continue
        canonical_original = int(frozen.user_attrs.get(
            "resume_original_trial_number",
            source_number,
        ))
        study.enqueue_trial(
            dict(frozen.params),
            user_attrs={
                "resume_checkpoint_dir": str(checkpoint_dir), 
                "resume_has_task_checkpoint": has_task_checkpoint, 
                "resume_original_trial_number": canonical_original, 
                "resume_source_trial_number": source_number
            }
        )
        tracked.add(source_number)
        enqueued.append(source_number)

    # Persist deduplication state after queued trials are durable in storage.
    if enqueued:
        study.set_user_attr(_RECOVERY_ENQUEUED_ATTR, sorted(tracked))

    return tuple(enqueued)


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
    fit_kwargs: Mapping[str, object] | None = None,
    objective_metrics: str | Sequence[str] | None = None,
    objective_directions: str | Sequence[str] | None = None,
    dtype_policy: str = "float32",
    deterministic_ops: bool = False,
    snapshot_network_name: str = "raw",
    class_num: int | None = None,
    class_order: Sequence[int] | None = None,
    task_groups: Sequence[Sequence[int]] | None = None,
    task_size: int = 1,
    class_order_mode: str = "fixed",
    task_order_mode: str = "fixed",
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
        objective_metrics (str | Sequence[str] | None): Objective names stored
            with the resolved trial configuration.
        objective_directions (str | Sequence[str] | None): Matching Optuna
            directions. Missing directions are inferred from metric names.
        dtype_policy (str): Keras numeric policy installed before construction.
        deterministic_ops (bool): Request deterministic TensorFlow kernels.
        snapshot_network_name (str): ``"raw"`` or ``"ema"`` teacher snapshot
            used by continual distillation.
        class_num (int | None): Number of classes in a continual study.
        class_order (Sequence[int] | None): Optional original-label order.
        task_groups (Sequence[Sequence[int]] | None): Optional explicit tasks.
        task_size (int): Classes per automatically constructed task.
        class_order_mode (str): ``"fixed"`` or seeded ``"random"`` order.
        task_order_mode (str): ``"fixed"`` or seeded ``"random"`` task order.

    Returns:
        Config: Fully typed trial configuration.

    Raises:
        ValueError: If the dataset/model combination or fit selection is
            unsupported, or progressive training omits ``stage_tasks``.
    """

    dataset_name = dataset_name.lower()
    fit_kwargs = dict(fit_kwargs or {})
    snapshot_network_name = str(snapshot_network_name).lower()
    class_order_mode = str(class_order_mode).lower()
    task_order_mode = str(task_order_mode).lower()
    class_order = None if class_order is None else list(class_order)
    task_groups = None if task_groups is None else [
        list(group) for group in task_groups
    ]
    # Reject teacher snapshot selectors unsupported by diffusion wrappers.
    if snapshot_network_name not in ("raw", "ema"):
        raise ValueError("snapshot_network_name must be 'raw' or 'ema'.")
    normalized_metrics, normalized_directions = _normalize_objective_spec(
        task,
        objective_metrics,
        objective_directions,
        use_ensemble_accuracy,
    )

    # Reject datasets outside the four supported HPO families.
    if dataset_name not in ("fmnist", "mnist", "cifar10", "cifar100"):
        raise ValueError("dataset_name must be FMNIST, MNIST, CIFAR10, or CIFAR100.")
    available_class_num, _, _ = get_dataset_spec(dataset_name)
    schedule_requested = class_num is not None or class_order is not None \
        or task_groups is not None or task_size != 1 \
        or class_order_mode != "fixed" or task_order_mode != "fixed"
    # Keep continual-only schedule controls from being silently ignored.
    if task != "continual" and schedule_requested:
        raise ValueError("Continual schedule options require task='continual'.")
    # Validate direct builder calls with the same canonical schedule resolver.
    if task == "continual":
        resolve_continual_schedule(
            class_num,
            class_order,
            task_groups,
            available_class_num=available_class_num,
            task_size=task_size,
            class_order_mode=class_order_mode,
            task_order_mode=task_order_mode,
            seed=seed,
        )
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

        # Normalize the flattened raw pixels used by dense-classifier trials.
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
            "class_num": class_num,
            "class_order": class_order,
            "task_groups": task_groups,
            "task_size": task_size,
            "class_order_mode": class_order_mode,
            "task_order_mode": task_order_mode,
            "seed": seed,
            "remove_prev_classes": True, 
            "keep_same_model": True, 
            "use_distillation": use_distillation,
            "snapshot_network_name": snapshot_network_name,
            # HPO is model development: locked test rows cannot enter a trial.
            "experiment_phase": "development",
            # HPO trials opt into task-boundary checkpoints for study recovery.
            "save_task_checkpoints": True,
            "use_ensemble_accuracy": bool(use_ensemble_accuracy),
            "evaluate_ensemble_accuracy": bool(use_ensemble_accuracy),
            "ensemble_accuracy_kwargs": ensemble_accuracy_kwargs,
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
            "dtype_policy": dtype_policy,
            "deterministic_ops": bool(deterministic_ops),
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
            "use_distillation": use_distillation,
            "objective_metrics": list(normalized_metrics),
            "objective_directions": list(normalized_directions),
            "seed": seed,
            "dtype_policy": dtype_policy,
            "deterministic_ops": bool(deterministic_ops),
            "snapshot_network_name": snapshot_network_name,
            "continual_schedule": {
                "class_num": class_num,
                "class_order": class_order,
                "task_groups": task_groups,
                "task_size": task_size,
                "class_order_mode": class_order_mode,
                "task_order_mode": task_order_mode,
            },
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


def _continual_validation_value(
    evaluations: Mapping[str, object], 
    metric_name: str
) -> float:
    """Read one scalar exclusively from validation continual metrics.

    Args:
        evaluations (Mapping[str, object]): Training report evaluation mapping.
        metric_name (str): Exact validation continual metric to read.

    Returns:
        float: Requested validation-only scalar.

    Raises:
        KeyError: If the validation mapping or requested metric is absent.
        TypeError: If the requested value is not scalar.
    """

    metrics = evaluations.get("validation_continual_metrics")
    # Refuse training/test fallbacks when validation aggregates are unavailable.
    if not isinstance(metrics, Mapping):
        raise KeyError(
            "Continual HPO requires "
            "evaluations['validation_continual_metrics']; test and training "
            "metrics are never substituted."
        )
    # Require the exact configured validation metric.
    if metric_name not in metrics:
        raise KeyError(
            "Requested continual validation objective was not reported: "
            + metric_name
        )

    value = np.asarray(metrics[metric_name])
    # Keep every Optuna objective dimension scalar.
    if value.ndim != 0:
        raise TypeError(
            "Continual validation objective must be scalar: " + metric_name
        )

    return float(value)


def _configured_objective_value(
    task: str, 
    model_name: str, 
    history: Mapping[str, Sequence[float]], 
    evaluations: Mapping[str, object], 
    metric_name: str, 
    direction: str
) -> float:
    """Resolve one explicit objective without changing legacy defaults.

    Args:
        task (str): Normalized HPO task name.
        model_name (str): Normalized model-family name.
        history (Mapping[str, Sequence[float]]): Epoch metric sequences.
        evaluations (Mapping[str, object]): Final report evaluation mapping.
        metric_name (str): Exact or supported semantic objective name.
        direction (str): Matching normalized Optuna direction.

    Returns:
        float: Selected scalar objective value.
    """

    # Enforce the dedicated validation-only continual namespace.
    if task == "continual":
        return _continual_validation_value(evaluations, metric_name)

    best = "min" if direction == "minimize" else "max"
    # Resolve diffusion ensemble accuracy from validation report output.
    if metric_name == "ensemble_accuracy":
        return _ensemble_accuracy_value(evaluations)
    # Resolve a model-family-aware semantic generation loss alias.
    if metric_name == "generation_loss":
        names = [
            "val_recon_loss", "recon_loss", "val_total_loss", "total_loss",
        ] if model_name in ("vae", "vae_classifier") else [
            "val_noise_loss", "noise_loss", "val_loss", "loss",
        ]
        return _history_value(history, names, best=best)
    # Resolve the semantic joint classifier accuracy alias.
    if metric_name == "classification_accuracy":
        return _history_value(history, [
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
        ], best=best)
    # Resolve the semantic standalone validation accuracy alias.
    if metric_name == "validation_accuracy":
        return _history_value(history, ["val_accuracy", "accuracy"], best=best)

    history_names = [metric_name]
    # Accept a readable validation prefix alongside Keras's ``val_`` prefix.
    if metric_name.startswith("validation_"):
        history_names.append("val_" + metric_name[len("validation_"):])

    return _history_value(history, history_names, best=best)


def _objective_values(
    task: str, 
    model_name: str,  
    history: Mapping[str, Sequence[float]], 
    evaluations: Mapping[str, object] | None = None, 
    use_ensemble_accuracy: bool = False,
    objective_metrics: str | Sequence[str] | None = None,
    objective_directions: str | Sequence[str] | None = None,
) -> float | tuple[float, ...]:
    """Convert training outputs to the study's objective value or tuple.

    Args:
        task (str): Study task.
        model_name (str): Selected model family.
        history (Mapping[str, Sequence[float]]): Training metric history.
        evaluations (Mapping[str, object] | None): Final report evaluations,
            required for joint ensemble-accuracy feedback.
        use_ensemble_accuracy (bool): Select ensemble instead of ordinary
            classification accuracy.
        objective_metrics (str | Sequence[str] | None): Explicit metric names.
            Continual names are resolved only under
            ``evaluations['validation_continual_metrics']``.
        objective_directions (str | Sequence[str] | None): Matching Optuna
            directions, used when selecting from explicit history metrics.

    Returns:
        float | tuple[float, ...]: Scalar objective or configured metric tuple.
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

    metrics, directions = _normalize_objective_spec(
        task,
        objective_metrics,
        objective_directions,
        use_ensemble_accuracy,
    )
    evaluations = evaluations or {}

    # Explicit metric lists use exact, independently directed resolution.
    if objective_metrics is not None:
        values = tuple(
            _configured_objective_value(
                task,
                model_name,
                history,
                evaluations,
                metric_name,
                direction,
            )
            for metric_name, direction in zip(metrics, directions)
        )
        return values[0] if len(values) == 1 else values

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
            classification = _ensemble_accuracy_value(evaluations)
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

    # Continual studies use the validation accuracy matrix's aggregate metric.
    return _continual_validation_value(evaluations, metrics[0])


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
    use_distillation: bool = False,
    objective_metrics: str | Sequence[str] | None = None,
    objective_directions: str | Sequence[str] | None = None,
    dtype_policy: str = "float32",
    deterministic_ops: bool = False,
    resume_from: str | Path | None = None,
    snapshot_network_name: str = "raw",
    class_num: int | None = None,
    class_order: Sequence[int] | None = None,
    task_groups: Sequence[Sequence[int]] | None = None,
    task_size: int = 1,
    class_order_mode: str = "fixed",
    task_order_mode: str = "fixed",
) -> Any:
    """Run a persistent Optuna study and return its ``Study`` object.

    By default, ``joint`` studies are Pareto searches with generative loss
    minimized and classification accuracy maximized, while continual studies
    maximize validation ``final_average_accuracy``. Explicit metric sequences
    create matching multi-objective studies. Each trial saves all normal
    training artifacts plus its input/resolved config; the dataset-specific
    study directory stores SQLite state and an incrementally updated CSV.

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
            joint or continual diffusion-classifier studies. Continual scoring
            remains validation-only.
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
        objective_metrics (str | Sequence[str] | None): Metric name or names.
            Continual objectives are read only from the validation continual
            metric mapping.
        objective_directions (str | Sequence[str] | None): Matching
            ``"minimize"``/``"maximize"`` directions. Names infer directions
            when omitted.
        dtype_policy (str): Keras numeric policy recorded for every trial.
        deterministic_ops (bool): Request deterministic TensorFlow kernels.
        resume_from (str | pathlib.Path | None): Existing HPO study root whose
            ``study.db`` should be loaded. This does not denote a model file.
        snapshot_network_name (str): ``"raw"`` or ``"ema"`` branch used for
            previous-task continual distillation teachers.
        class_num (int | None): Selected class count for continual studies.
        class_order (Sequence[int] | None): Optional original-label order.
        task_groups (Sequence[Sequence[int]] | None): Optional explicit tasks.
        task_size (int): Classes per automatically constructed task.
        class_order_mode (str): ``"fixed"`` or seeded ``"random"`` order.
        task_order_mode (str): ``"fixed"`` or seeded ``"random"`` task order.

    Returns:
        optuna.study.Study: Resumable completed/partial study. Multi-objective
        studies expose ``best_trials``; scalar studies expose ``best_trial``.

    Raises:
        ImportError: If Optuna is unavailable.
        TypeError: If ``task`` is not a string.
        ValueError: If the task/model pair, fit selection, or positive budgets
            are invalid, or progressive training omits ``stage_tasks``.
    """

    task = normalize_training_task(task)
    try:
        import optuna
    except ImportError as error:
        raise ImportError(
            "Optuna is required for HPO. "
            "Install the project requirements."
        ) from error

    model_name = model_name.lower()
    fit_kwargs = dict(fit_kwargs or {})
    snapshot_network_name = str(snapshot_network_name).lower()
    class_order_mode = str(class_order_mode).lower()
    task_order_mode = str(task_order_mode).lower()
    class_order = None if class_order is None else list(class_order)
    task_groups = None if task_groups is None else [
        list(group) for group in task_groups
    ]
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
    # Reject teacher snapshot selectors unsupported by diffusion wrappers.
    if snapshot_network_name not in ("raw", "ema"):
        raise ValueError("snapshot_network_name must be 'raw' or 'ema'.")
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

    available_class_num, _, _ = get_dataset_spec(dataset_name)
    schedule_requested = class_num is not None or class_order is not None \
        or task_groups is not None or task_size != 1 \
        or class_order_mode != "fixed" or task_order_mode != "fixed"
    # Reject schedule switches that a non-continual objective would ignore.
    if task != "continual" and schedule_requested:
        raise ValueError("Continual schedule options require task='continual'.")
    # Validate every continual schedule before creating study metadata/storage.
    if task == "continual":
        resolve_continual_schedule(
            class_num,
            class_order,
            task_groups,
            available_class_num=available_class_num,
            task_size=task_size,
            class_order_mode=class_order_mode,
            task_order_mode=task_order_mode,
            seed=seed,
        )

    normalized_metrics, normalized_directions = _normalize_objective_spec(
        task,
        objective_metrics,
        objective_directions,
        use_ensemble_accuracy,
    )

    root = Path(results_path)
    # Reuse the explicitly selected persistent study directory when resuming.
    if resume_from is not None:
        study_root = Path(resume_from)
        # Require the supplied study root to exist before any writes.
        if not study_root.is_dir():
            raise FileNotFoundError(
                "HPO resume study root does not exist: " + str(study_root)
            )
        # Require existing SQLite state rather than starting a misleading study.
        if not (study_root / "study.db").is_file():
            raise FileNotFoundError(
                "HPO resume study root has no study.db: " + str(study_root)
            )
    # Construct the established study hierarchy for a new study.
    else:
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

    study_name = f"{task}-{model_name}-{dataset_name.lower()}" + (
        "-ensemble-accuracy" if use_ensemble_accuracy else ""
    )
    # Give distillation trials an independent persistent study identity.
    if effective_distillation:
        study_name += "-distillation"
    # Give progressive studies an independent SQLite study identity.
    if fit_method == "fit_progressively":
        study_name += "-fit-progressively"

    study_spec = _make_study_spec(
        study_name=study_name,
        task=task,
        model_name=model_name,
        dataset_name=dataset_name,
        epochs=epochs,
        seed=seed,
        use_ensemble_accuracy=use_ensemble_accuracy,
        ensemble_accuracy_kwargs=ensemble_accuracy_kwargs,
        fit_method=fit_method,
        fit_kwargs=fit_kwargs,
        teacher_network=teacher_network,
        effective_distillation=effective_distillation,
        objective_metrics=normalized_metrics,
        objective_directions=normalized_directions,
        dtype_policy=dtype_policy,
        deterministic_ops=deterministic_ops,
        snapshot_network_name=snapshot_network_name,
        class_num=class_num,
        class_order=class_order,
        task_groups=task_groups,
        task_size=task_size,
        class_order_mode=class_order_mode,
        task_order_mode=task_order_mode,
    )
    # Validate identity from a sidecar before touching Optuna storage. This
    # prevents a mismatched resume request from creating a second study name in
    # the supplied SQLite database.
    spec_path = study_root / _STUDY_SPEC_FILE
    # Require a matching sidecar before loading an explicitly resumed study.
    if resume_from is not None:
        persisted_file_spec = _read_study_spec(study_root)
        # Reject any changed scientific option before accessing SQLite.
        if fingerprint_state(persisted_file_spec) != fingerprint_state(study_spec):
            raise ValueError(
                "Requested HPO study specification differs from resume_from."
            )
    # Validate an existing sidecar when reopening by the ordinary output path.
    elif spec_path.is_file():
        persisted_file_spec = _read_study_spec(study_root)
        # Keep accidental path reuse from mixing incompatible experiments.
        if fingerprint_state(persisted_file_spec) != fingerprint_state(study_spec):
            raise ValueError(
                "Existing HPO study specification differs from this request."
            )
    # Seal a new study identity before creating its SQLite entry.
    else:
        _write_study_spec(study_root, study_spec)

    configs_path = study_root / "configs"
    configs_path.mkdir(parents=True, exist_ok=True)
    storage_path = (study_root / "study.db").resolve().as_posix()
    sampler = optuna.samplers.TPESampler(seed=seed)
    create_kwargs = {
        "study_name": study_name, 
        "storage": "sqlite:///" + storage_path, 
        "sampler": sampler,
        "load_if_exists": True
    }

    # Optuna uses a distinct argument for scalar and multi-objective studies.
    if len(normalized_directions) == 1:
        create_kwargs["direction"] = normalized_directions[0]
    # Configure one direction per dimension for a Pareto study.
    else:
        create_kwargs["directions"] = list(normalized_directions)

    # Load-only semantics prevent resume from creating a missing study name.
    if resume_from is not None:
        # load_study cannot create a missing identity, unlike
        # create_study(load_if_exists=True).
        study = optuna.load_study(
            study_name=study_name,
            storage=create_kwargs["storage"],
            sampler=sampler,
        )
    # Create or intentionally reopen the normal non-resume study hierarchy.
    else:
        study = optuna.create_study(**create_kwargs)

    persisted_attr_spec = study.user_attrs.get(_STUDY_SPEC_ATTR)
    existing_trials = tuple(study.get_trials(deepcopy=False))
    # Initialize metadata only for a truly empty newly created study.
    if persisted_attr_spec is None:
        # Refuse to bless pre-existing trials whose identity was never sealed.
        if resume_from is not None or existing_trials:
            raise ValueError(
                "Existing HPO study has no validated study_spec user attribute."
            )
        study.set_user_attr(_STUDY_SPEC_ATTR, study_spec)
        study.set_user_attr(
            _STUDY_SPEC_FINGERPRINT_ATTR,
            fingerprint_state(study_spec),
        )
    # Authenticate both persisted user-attribute representations.
    elif fingerprint_state(persisted_attr_spec) != fingerprint_state(study_spec) \
    or study.user_attrs.get(_STUDY_SPEC_FINGERPRINT_ATTR) \
    != fingerprint_state(study_spec):
        raise ValueError(
            "Persisted Optuna study specification differs from this request."
        )

    # Restore the post-suggestion sampler cursor before queuing/optimizing.
    if resume_from is not None:
        sampler_state = study.user_attrs.get(_SAMPLER_RNG_STATE_ATTR)
        # A nonempty study must have persisted its exact sampler position.
        if sampler_state is None:
            # Fail instead of silently replaying TPE draws from the initial seed.
            if existing_trials:
                raise ValueError(
                    "Existing HPO study has trials but no recoverable TPE RNG state."
                )
        # Apply a complete two-stream state when the study contains one.
        else:
            _restore_sampler_rng_state(sampler, sampler_state)
    # Convert recoverable abandoned trials into one-time queued retries.
    if resume_from is not None:
        _enqueue_recovery_trials(study, study_root)


    def objective(trial: Any) -> float | tuple[float, ...]:
        """Execute and score one Optuna trial.

        Args:
            trial (optuna.trial.Trial): Active trial.

        Returns:
            float | tuple[float, ...]: Scalar or multi-objective value.
        """

        tf.keras.backend.clear_session()
        gc.collect()

        # Keep the data split and initialization seed fixed across candidates;
        # Optuna's independently seeded sampler supplies the search variation.
        trial_seed = seed
        config = _build_trial_config(
            trial, 
            task, 
            model_name, 
            dataset_name, 
            epochs, 
            trial_seed,
            results_path=root, 
            use_ensemble_accuracy=use_ensemble_accuracy, 
            ensemble_accuracy_kwargs=ensemble_accuracy_kwargs, 
            use_distillation=effective_distillation,
            fit_method=fit_method, 
            fit_kwargs=fit_kwargs,
            objective_metrics=objective_metrics,
            objective_directions=objective_directions,
            dtype_policy=dtype_policy,
            deterministic_ops=deterministic_ops,
            snapshot_network_name=snapshot_network_name,
            class_num=class_num,
            class_order=class_order,
            task_groups=task_groups,
            task_size=task_size,
            class_order_mode=class_order_mode,
            task_order_mode=task_order_mode,
        )
        # All trial suggestions have now consumed their sampler draws. Persist
        # both TPE streams before any training/file work so a killed trial can
        # be retried without rewinding subsequent suggestions.
        study.set_user_attr(
            _SAMPLER_RNG_STATE_ATTR,
            _capture_sampler_rng_state(sampler),
        )

        input_config_path = configs_path / f"trial-{trial.number:04d}.yaml"
        checkpoint_dir = _trial_checkpoint_dir(study_root, trial)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Keep every trial's artifacts beneath the selected (possibly resumed)
        # study root rather than recomputing a second hierarchy from results_path.
        config.training.results_path = str(study_root / "runs")
        config.hpo.update({
            "study_root": str(study_root),
            "checkpoint_dir": str(checkpoint_dir),
            "input_config_path": str(input_config_path),
        })
        recovery_original = trial.user_attrs.get("resume_original_trial_number")
        # Retain the canonical source identity in retried trial configurations.
        if recovery_original is not None:
            config.hpo["resume_original_trial_number"] = int(recovery_original)
        # Keep resumed-study TensorBoard events under the explicit study root.
        if resume_from is not None:
            config.training.tensorboard_path = str(study_root / "tensorboard")
        # Install task-boundary recovery only for continual training.
        if task == "continual":
            config.continually_learn.checkpoint_dir = str(checkpoint_dir)
            # Resume only when a committed task boundary already exists. Passing
            # a newly created empty directory would turn a fresh trial into an
            # invalid recovery request.
            if _has_committed_task_checkpoint(checkpoint_dir):
                config.continually_learn.resume_from = str(checkpoint_dir)

        # Publish recovery metadata before training so interrupted trials remain
        # discoverable from the persistent Optuna database.
        trial.set_user_attr("seed", trial_seed)
        trial.set_user_attr("checkpoint_dir", str(checkpoint_dir))
        trial.set_user_attr("config_path", str(input_config_path))
        save_config(config, input_config_path)
        config = load_config(input_config_path)

        result = main(config, teacher_network=teacher_network)
        values = _objective_values(
            task, 
            model_name, 
            result["history"], 
            evaluations=result["evaluations"], 
            use_ensemble_accuracy=config.hpo["use_ensemble_accuracy"],
            objective_metrics=objective_metrics,
            objective_directions=objective_directions,
        )
        values_list = list(values) if isinstance(values, tuple) else [values]
        config.hpo["objectives"] = values_list

        resolved_path = Path(result["results_path"]) / "config.yaml"
        save_config(config, resolved_path)
        pd.DataFrame([
            {"name": name, "direction": direction, "value": value}
            for name, direction, value in zip(
                normalized_metrics,
                normalized_directions,
                values_list,
            )
        ]).to_csv(
            Path(result["results_path"]) / "objectives.csv", 
            index=False
        )
        trial.set_user_attr("results_path", str(result["results_path"]))
        trial.set_user_attr("config_path", str(resolved_path))
        trial.set_user_attr("resolved_config_path", str(resolved_path))

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
