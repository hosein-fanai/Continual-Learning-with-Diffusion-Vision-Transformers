"""Persistent Optuna optimization through the shared Config training pipeline.

SEARCH_SPACES describes validated architecture templates; trial suggestion
helpers instantiate those templates into common.config.Config. Every trial
saves and reloads YAML before common.train.main runs, preserving the same
configuration API used by ordinary experiments. The public run_hpo function
returns an Optuna Study and saves study identity, sampler state, trial tables,
resolved configs, and scalar or Pareto objective values under the study root.

Objectives come exclusively from post-training validation reports. Continual
studies use validation_continual_metrics; enabling EnsembleAccuracy makes its
task matrix feed every selected continual aggregate. Distilled V2 classifiers
use the learner's automatic separate optimization phases and raw/EMA teacher
snapshots. Immutable study specifications prevent incompatible resume requests,
and committed task checkpoints allow interrupted continual trials to recover.
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
from common.utils import load_feature_split_metadata


_DIFFUSION_MODELS = {
    "diffusion_transformer", "dit_classifier", "dit_decoder", 
    "dit_encoder_decoder", "dit_encoder_decoder_classifier", 
    "unet", "unet_classifier"
}
_DIFFUSION_CLASSIFIER_MODELS = {
    "dit_classifier", "dit_encoder_decoder_classifier", 
    "unet_classifier"
}
_DIFFUSION_CLASSIFIER_STUDY = "diffusion_classifier"
_DIFFUSION_HPO_MODELS = _DIFFUSION_MODELS | {
    _DIFFUSION_CLASSIFIER_STUDY
}
_DIFFUSION_HPO_CLASSIFIER_MODELS = _DIFFUSION_CLASSIFIER_MODELS | {
    _DIFFUSION_CLASSIFIER_STUDY
}
# Version 10 expands scheduled classes before resampling and derives public
# accuracies from the same predictions as the continual accuracy matrices.
SEARCH_SPACE_VERSION = 10

_OPTIMIZATION = {
    "batch_size": "categorical; architecture-appropriate powers of two", 
    "learning_rate": "log-uniform; model-family-specific bounds", 
    "learning_rate_schedule": "cosine decay or constant", 
    "optimizer": "family-specific subset of SGD, RMSprop, Adam, AdamW, Nadam",
    "weight_decay": "AdamW-only log-uniform value from 1e-6 to 1e-3",
    "momentum": "SGD/RMSprop only: 0, 0.3, 0.9, or 0.95",
    "clipnorm": "None, 0.5, 1, or 5"
}
_DIT = {
    **_OPTIMIZATION,
    "learning_rate_schedule": (
        "cosine for ordinary fit; constant for continual/progressive fit"
    ),
    "patch_size": "2 or 4 when it divides the image size",
    "capacity": "32x4, 64x4, 96x4, or 128x4",
    "architecture": (
        "standalone generator: plain DiT or compact two-level feature-skip U-DiT"
    ),
    "depth": (
        "2 through 6 plain blocks; standalone compact U-DiT uses 9 stages"
    ),
    "mlp_ratio": "2 or 4",
    "drop_prob": "0, 0.05, or 0.1",
    "patchify_with_cnn": "boolean", 
    "patch_position_embedding": "fixed 2D sinusoidal",
    "normalization_adaptation": "fixed enabled",
    "resampling_position_embedding": (
        "compact U-DiT only: 2D sinusoidal or learned new weights"
    ),
    "use_refiner_cnn": "boolean (applied to the decoder when present)", 
    "timesteps": "500 or 1000",
    "noise_schedule": "linear, cosine, or clipped-cosine",
    "modify_first_t": "fixed false",
    "p_uncond": "0.05, 0.1, or 0.2",
    "ema_decay": "0.995 or 0.999",
    "evaluation_network": "fixed EMA",
    "loss_function": "mse",
    "image_loss_coefficient": (
        "0, 0.01, 0.05, or 0.1; fixed to 0 for input reconstruction"
    ),
    "variational_kl_coefficient": (
        "log-uniform 1e-4 to 1e-2 for input reconstruction"
    ),
    "noise_distillation_coefficient": (
        "log-uniform 1e-4 to 1 when teacher distillation is enabled"
    ),
}
_UNET = {
    **_OPTIMIZATION,
    "learning_rate_schedule": (
        "cosine for ordinary fit; constant for continual/progressive fit"
    ),
    "width_template": "(32,64), (32,64,96), or (64,96,128)", 
    "block_depth": "1 or 2 residual blocks per scale", 
    "bottleneck_multiplier": "1.5 or 2 times the widest stage", 
    "bottleneck_depth": "1, 2, or 3", 
    "embedding_budget": "64, 96, or 128 channels", 
    "normalization": "batch normalization on/off", 
    "dropout": "0, 0.05, or 0.1", 
    "resampling": "average/interpolation or learned convolutional pair", 
    "timesteps": "500 or 1000",
    "noise_schedule": "linear, cosine, or clipped-cosine",
    "modify_first_t": "fixed false",
    "p_uncond": "0.05, 0.1, or 0.2",
    "ema_decay": "0.995 or 0.999",
    "evaluation_network": "fixed EMA",
    "loss_function": "mse",
    "image_loss_coefficient": (
        "0, 0.01, 0.05, or 0.1; fixed to 0 for input reconstruction"
    ),
    "variational_kl_coefficient": (
        "log-uniform 1e-4 to 1e-2 for input reconstruction"
    ),
    "noise_distillation_coefficient": (
        "log-uniform 1e-4 to 1 when teacher distillation is enabled"
    ),
}
_VAE = {
    **_OPTIMIZATION, 
    "latent_dim": "8, 16, 32, 64, or 128", 
    "hidden_template": "16, 64-16, 256-64, 512-128, or 512-256-64",
    "beta": "log-uniform 0.01 to 2.0", 
    "loss_function": "mse or mae", 
    "activation": "ReLU or SELU", 
    "batch_normalization": "boolean (disabled for SELU)"
}
_CNN = {
    **_OPTIMIZATION, 
    "width/depth_template": "three coupled convolutional stage templates",
    "kernel_size": "3 or 5; first kernel 3, 5, or 7", 
    "batch_normalization": "boolean", 
    "pooling": "max between stages and global average",
    "dropout": "0 to 0.3, including 0.15"
}
_DNN = {
    **_OPTIMIZATION, 
    "hidden_template": "linear, 256-128, 512-256, or 1024-512",
    "activation/initializer": "coupled ReLU-He, ELU-He, or SELU-LeCun", 
    "batch_normalization": "boolean except with SELU", 
    "dropout": "0 to 0.5"
}
_PRETRAINED = {
    **_OPTIMIZATION, 
    "unfrozen_tail": "1, 5, 12, 20, or all Xception layers",
    "dropout": "0 to 0.5"
}
_JOINT_NOTE = {
    "classifier_loss_coefficient": "log-uniform 1e-3 to 3e-2",
    "ctr_loss_coef": (
        "0, 1e-4, 1e-3, 1e-2, or 5e-2; positive values enable a final "
        "classifier token regularizer"
    ), 
    "masking_recipe": "V1 only: CFG-null, timestep, both, or neither",
    "mask_t_percentage": (
        "V1 only: 50, 70, or 90 when timestep masking is used"
    ),
    "classifier_train_type": (
        "V1 only: conditional prediction or unconditional prediction at CFG 1"
    ),
    "distillation": (
        "with a runtime or previous-task teacher: hard or soft targets and "
        "log-uniform clf_distil_loss_coef; soft temperature and example scope"
    ), 
    "objective": "Pareto minimize generative loss / maximize selected accuracy"
}
_DIT_CLASSIFIER_NOTE = {
    "classifier_architecture": (
        "linear, feature connection, U-shaped, central U-VAE, or a central "
        "multilevel variational U-shaped bottleneck"
    ), 
    "feature_aggregation": "last, early depth-1, or all denoiser features",
    "classifier_only_cls_token": "boolean", 
    "classifier_cls_token_type": (
        "new weight or time-label when a separate token is used"
    ), 
    "classifier_depth": "1 through 15, depending on the selected template",
    "classifier_layer_norm_adaptation": "fixed enabled",
    "classifier_block_dropout": "0, 0.05, or 0.1",
    "classifier_mlp_ratio": "None or 1",
    "classifier_head_dropout": "0, 0.05, or 0.1",
    "variational_latent_width": (
        "independent 16, 32, or 64 feature budget for each central pair"
    )
}
_DIT_CLASSIFIER_WRAPPER_NOTE = {
    "wrapper_name": "diffusion_classifier (V1) or diffusion_classifier_v2 (V2)", 
    "v2_classifier_variables": (
        "V2 only: coupled separate, conditions-only, or notebook "
        "classification-heavy variable recipes"
    ), 
    "clf_train_noisified_max_timesteps": (
        "V2 only: None (clean input at timestep 0), 64, or 256; numeric caps "
        "are bounded by timesteps"
    )
}
_CONTINUAL_NOTE = {
    "replay_samples_per_class": "100, 500, 1000, 2500, or 5000", 
    "generator_train_samples": (
        "current data, 1000, 2500, 5000, 7500, or 10000"
    ), 
    "objective": "maximize selected mean class-incremental accuracy"
}
_CONTINUAL_DIFFUSION_NOTE = {
    **_CONTINUAL_NOTE,
    "test_steps": "20, 50, or 100 up to timesteps",
    "test_cfg_scale": "uniform 2.5 to 5",
    "test_eta": "0 or 1",
}
_CONTINUAL_CLASSIFIER_NOTE = {
    "protocol": (
        "multiclass first task: sequential, cumulative, or reservoir replay; "
        "singleton first task: cumulative or reservoir replay"
    ),
    "task_size": "one or more classes per task; at least two tasks required",
    "reservoir_capacity": "2500, 5000, or 10000 rows",
    "reservoir_sample_count": "500, 1000, or 2500 rows",
    "reservoir_insert_count": "500 or 1000 rows",
    "objective": "maximize validation class-incremental accuracy",
}
_CONTINUAL_DIFFUSION_CLASSIFIER_NOTE = {
    **_DIT,
    **_UNET,
    **_DIT_CLASSIFIER_NOTE,
    **_DIT_CLASSIFIER_WRAPPER_NOTE,
    **_JOINT_NOTE,
    **_CONTINUAL_DIFFUSION_NOTE,
    "model_family": (
        "DiT classifier, encoder-decoder DiT classifier, or U-Net classifier"
    ),
    "continual_strategy": (
        "new-only, cumulative, or generated replay when mathematically valid"
    ),
    "replay_budget_mode": "per-class legacy or fixed-total exposure",
    "replay_selection": (
        "all, uniform, confidence, surprise, or confidence-surprise gating"
    ),
    "teacher_snapshot": "raw or EMA completed-task student",
}

SEARCH_SPACES = {
    "generation": {
        "diffusion_transformer": _DIT, 
        "dit_decoder": {
            # Replace the shared depth dimension with decoder-specific depth choices.
            **{key: value for key, value in _DIT.items() if key != "depth"}, 
            "decoder_depth": "1, 2, or 4"
        }, 
        "dit_encoder_decoder": {
            # Replace shared depth with independently tuned encoder and decoder depths.
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
            # Use separate encoder/decoder depths for this joint architecture.
            **{key: value for key, value in _DIT.items() if key != "depth"}, 
            **_DIT_CLASSIFIER_NOTE, 
            **_DIT_CLASSIFIER_WRAPPER_NOTE,
            **_JOINT_NOTE, 
            "encoder_depth": "2, 4, or 6", 
            "decoder_depth": "1, 2, or 4"
        }, 
        "unet_classifier": {
            **_UNET, 
            **_DIT_CLASSIFIER_WRAPPER_NOTE,
            "classifier_depth": "1, 2, or 3", 
            **_JOINT_NOTE
        }, 
        "vae_classifier": {
            **_VAE, 
            "alpha": "log-uniform 1e-5 to 1e-2 (mean CE)", 
            "objective": "Pareto minimize beta-VAE loss / maximize accuracy"
        }
    },
    "classification": {
        "cnn": _CNN, 
        "dnn": _DNN, 
        "pretrained": _PRETRAINED
    }, 
    "continual": {
        "cnn": {**_CNN, **_CONTINUAL_CLASSIFIER_NOTE},
        "dnn": {**_DNN, **_CONTINUAL_CLASSIFIER_NOTE},
        "pretrained": {**_PRETRAINED, **_CONTINUAL_CLASSIFIER_NOTE},
        "diffusion_transformer": {**_DIT, **_CONTINUAL_DIFFUSION_NOTE},
        "dit_classifier": {
            **_DIT, 
            **_DIT_CLASSIFIER_NOTE, 
            **_DIT_CLASSIFIER_WRAPPER_NOTE, 
            **_JOINT_NOTE, 
            **_CONTINUAL_DIFFUSION_NOTE
        }, 
        "dit_decoder": {
            # Replace shared depth with decoder depth for continual decoder studies.
            **{key: value for key, value in _DIT.items() if key != "depth"}, 
            "decoder_depth": "1, 2, or 4", 
            **_CONTINUAL_DIFFUSION_NOTE
        }, 
        "dit_encoder_decoder": {
            # Use encoder/decoder depth dimensions for this continual architecture.
            **{key: value for key, value in _DIT.items() if key != "depth"}, 
            "encoder_depth": "2, 4, or 6", 
            "decoder_depth": "1, 2, or 4", 
            **_CONTINUAL_DIFFUSION_NOTE
        }, 
        "dit_encoder_decoder_classifier": {
            # Use separate encoder/decoder depths for the continual classifier.
            **{key: value for key, value in _DIT.items() if key != "depth"}, 
            **_DIT_CLASSIFIER_NOTE, 
            **_DIT_CLASSIFIER_WRAPPER_NOTE,
            **_JOINT_NOTE, 
            "encoder_depth": "2, 4, or 6", 
            "decoder_depth": "1, 2, or 4", 
            **_CONTINUAL_DIFFUSION_NOTE
        }, 
        "unet": {**_UNET, **_CONTINUAL_DIFFUSION_NOTE},
        "unet_classifier": {
            **_UNET, 
            **_DIT_CLASSIFIER_WRAPPER_NOTE,
            "classifier_depth": "1, 2, or 3", 
            **_JOINT_NOTE, 
            **_CONTINUAL_DIFFUSION_NOTE
        }, 
        _DIFFUSION_CLASSIFIER_STUDY: (
            _CONTINUAL_DIFFUSION_CLASSIFIER_NOTE
        ),
        "vae": {**_VAE, **_CONTINUAL_NOTE},
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
    _DIFFUSION_CLASSIFIER_STUDY: "dcf",
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


class _TrialView:
    """Adapt an Optuna trial to conditional namespaces and sealed search overrides.

    Suggestions retain the underlying trial's storage and random sampler while
    names receive an optional raw-family prefix. Overrides are checked first by
    prefixed exact name, prefixed suffix-stripped name, exact local name, then
    suffix-stripped local name. Numeric mappings replace sampling bounds;
    categorical values restrict validated templates, with explicit exceptions
    for batch/replay counts and timestep-bounded sampling steps.

    Attributes:
        number (int): Underlying trial's storage-assigned number.
        params (Mapping[str, object]): Underlying sampled parameters, including
            any namespaced names; returned without copying.
        user_attrs (Mapping[str, object]): Underlying metadata, or an empty
            mapping for a trial double without user attributes.
    """

    def __init__(
        self,
        trial: Any,
        prefix: str = "",
        overrides: Mapping[str, object] | None = None,
    ) -> None:
        """Retain a trial and copy the overrides used by subsequent suggestions.

        Args:
            trial (optuna.trial.Trial | object): Active Optuna trial or a compatible
                test double providing the requested suggestion methods.
            prefix (str): Literal prefix appended before every suggested parameter name,
                usually a raw family followed by a period. Defaults to '' for unprefixed
                suggestions.
            overrides (Mapping[str, object] | None): Categorical scalar/list or {'choices':
                ...} replacements, or numeric mappings with low, high, step, and log keys.
                Defaults to None for no overrides. The outer mapping is copied; its nested
                values are read without mutation.

        Returns:
            None: The adapter references the trial and stores its namespace/options.
        """

        self._trial = trial
        self._prefix = prefix
        self._overrides = dict(overrides or {})

    @property
    def number(self) -> int:
        """Read the storage-assigned number of the wrapped trial.

        Returns:
            int: The underlying trial's number, unchanged by parameter prefixes.
        """
        return self._trial.number

    @property
    def params(self) -> Mapping[str, object]:
        """Expose the sampled parameter mapping of the wrapped trial.

        Returns:
            Mapping[str, object]: The underlying mapping, not a defensive copy;
            parameter names retain any prefixes used when suggesting them.
        """
        return self._trial.params

    @property
    def user_attrs(self) -> Mapping[str, object]:
        """Expose wrapped metadata while supporting minimal trial doubles.

        Returns:
            Mapping[str, object]: Underlying user attributes without copying, or a
            new empty dictionary if the wrapped object has no user_attrs attribute.
        """
        return getattr(self._trial, "user_attrs", {})

    def _name(self, name: str) -> str:
        """Resolve a local suggestion name to its stored Optuna name.

        Args:
            name (str): Local parameter name, including any topology suffix.

        Returns:
            str: The configured prefix followed immediately by name; neither part
            is normalized or escaped.
        """

        return self._prefix + name

    def _override(self, name: str) -> object | None:
        """Find the most specific configured override for a local parameter.

        Terminal topology suffixes t<number>, grid<number>, pair<number>, flat,
        singleton, or multiclass (preceded by an underscore) are stripped to form
        a second lookup name. Prefixed exact/base names take precedence over both
        unprefixed names, allowing one umbrella study to specialize each family.

        Args:
            name (str): Local parameter name before the adapter prefix is added.

        Returns:
            object | None: First matched override, returned without copying; None
            means no effective override, including an explicitly stored None.
        """

        base_name = re.sub(
            r"_(?:t\d+|grid\d+|pair\d+|flat|singleton|multiclass)$",
            "",
            name,
        )
        local_names = (name, base_name)
        for candidate in (
            *(self._prefix + item for item in local_names),
            *local_names,
        ):
            # Use the first matching prefixed or unprefixed parameter override.
            if candidate in self._overrides:
                return self._overrides[candidate]
        return None

    def suggest_categorical(
        self,
        name: str,
        choices: Sequence[object],
    ) -> object:
        """Sample one permitted categorical choice after resolving overrides.

        Args:
            name (str): Local parameter name used for override lookup and prefixing.
            choices (Sequence[object]): Nonempty preset choices. An override can
                supply a sequence, scalar, or mapping with a choices key. Strings
                and bytes are single choices rather than sequences of characters.
                Most overrides must remain within the preset. Batch/replay counts
                may extend it within their permitted integer range; test_steps_tN
                may extend it within 1..N. train_num also permits -1 for all rows.

        Returns:
            object: The value selected by the underlying trial; count overrides are
            converted to integers before they are offered to the sampler.

        Raises:
            ValueError: If replacement choices are empty, contain forbidden template
                values, or violate a parameter's allowed count/timestep range.
            TypeError: If a count override cannot be converted to an integer.
        """
        override = self._override(name)
        default_choices = list(choices)
        # Read categorical choices from a structured override mapping.
        if isinstance(override, Mapping):
            choices = override.get("choices", choices)
        # Treat a direct override value as the replacement choice set.
        elif override is not None:
            choices = override
        # Wrap a scalar or text choice so it is sampled as one value.
        if isinstance(choices, (str, bytes)) or not isinstance(
            choices, Sequence
        ):
            choices = [choices]
        choices = list(choices)
        # Reject an empty choice set before asking Optuna to sample.
        if not choices:
            raise ValueError(f"Search override {name!r} has no choices.")

        extensible_counts = {
            "batch_size",
            "replay_samples",
            "replay_old_examples",
            "replay_current_examples",
            "replay_candidate_multiplier",
            "replay_buffer_capacity",
            "replay_buffer_insert_count",
            "replay_buffer_sample_count",
            "train_num",
        }
        # Allow caller-defined replay and batch counts beyond the preset choices.
        if override is not None and name in extensible_counts:
            # Allow -1 for all training rows; require positive batches/multipliers and nonnegative replay counts.
            minimum = -1 if name == "train_num" else (
                1 if name in ("batch_size", "replay_candidate_multiplier")
                else 0
            )
            choices = [int(value) for value in choices]
            # Reject counts outside their allowed range, including an empty training pool.
            if any(
                value < minimum
                or (name == "train_num" and value == 0)
                for value in choices
            ):
                raise ValueError(
                    f"Search override {name!r} has an invalid count."
                )
        # Allow custom sampling step counts within the selected diffusion horizon.
        elif override is not None and re.fullmatch(r"test_steps_t\d+", name):
            timesteps = int(name.rsplit("t", 1)[1])
            choices = [int(value) for value in choices]
            # Reject reverse-process step counts outside [1, timesteps].
            if any(
                not 1 <= value <= timesteps
                for value in choices
            ):
                raise ValueError(
                    "test_steps overrides must be within [1, timesteps]."
                )
        # Keep other categorical overrides within the validated architecture choices.
        elif override is not None and any(
            value not in default_choices for value in choices
        ):
            raise ValueError(
                f"Search override {name!r} must use choices from "
                f"{default_choices!r}."
            )
        return self._trial.suggest_categorical(
            self._name(name), choices
        )

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        """Sample a float with optional namespacing and numeric-bound overrides.

        Args:
            name (str): Local parameter name used for override lookup and prefixing.
            low (float): Inclusive lower sampling bound before overrides.
            high (float): Inclusive upper bound before overrides; Optuna may adjust
                it to the step grid when the interval is not exactly divisible.
            step (float | None): Quantization spacing. Defaults to None for a continuous
                interval. A mapping override may replace it.
            log (bool): Sample in log space when True, requiring positive bounds and no
                step. Defaults to False. A mapping override may replace it.

        Returns:
            float: The underlying trial's sampled float. Mapping overrides replace
            low, high, step, and log independently; nonmapping overrides are ignored.

        Raises:
            ValueError: If numeric conversion or Optuna's range/grid validation fails.
            TypeError: If an overridden bound cannot be converted to a float.
        """
        override = self._override(name)
        # Apply explicit numeric bounds and sampling options from a float override.
        if isinstance(override, Mapping):
            low = float(override.get("low", low))
            high = float(override.get("high", high))
            step = override.get("step", step)
            log = bool(override.get("log", log))
        return self._trial.suggest_float(
            self._name(name), low, high, step=step, log=log
        )

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int:
        """Sample an integer with optional namespacing and bound overrides.

        Args:
            name (str): Local parameter name used for override lookup and prefixing.
            low (int): Inclusive lower sampling bound before overrides.
            high (int): Inclusive upper bound before overrides; the sampler may
                reduce it to the last point on the step grid.
            step (int): Positive integer grid spacing. Defaults to 1. A mapping override may
                replace it; log sampling requires a unit step.
            log (bool): Sample a positive integer range in log space when True. Defaults to
                False. A mapping override may replace it.

        Returns:
            int: The underlying trial's sampled integer. Mapping overrides replace
            low, high, step, and log independently; nonmapping overrides are ignored.

        Raises:
            ValueError: If conversion or Optuna's range/grid validation fails.
            TypeError: If an overridden numeric setting cannot be converted to int.
        """
        override = self._override(name)
        # Apply explicit numeric bounds and sampling options from an integer override.
        if isinstance(override, Mapping):
            low = int(override.get("low", low))
            high = int(override.get("high", high))
            step = int(override.get("step", step))
            log = bool(override.get("log", log))
        return self._trial.suggest_int(
            self._name(name), low, high, step=step, log=log
        )

    def set_user_attr(self, name: str, value: object) -> None:
        """Record unprefixed metadata if the wrapped trial supports it.

        Args:
            name (str): Metadata key. Parameter prefixes do not modify this key.
            value (object): JSON-serializable Optuna metadata value; forwarded as-is.

        Returns:
            None: The attribute is stored when set_user_attr is callable on the
            wrapped object; otherwise the request has no effect.
        """
        setter = getattr(self._trial, "set_user_attr", None)
        # Record metadata only when the underlying trial supports attributes.
        if callable(setter):
            setter(name, value)

    def __getattr__(self, name: str) -> object:
        """Delegate trial capabilities not explicitly adapted by this class.

        Args:
            name (str): Missing attribute or method requested on the adapter.

        Returns:
            object: Underlying attribute, including its original bound-method state.

        Raises:
            AttributeError: If the wrapped trial also lacks the requested attribute.
        """

        return getattr(self._trial, name)


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
    if fit_method == "fit_progressively" and model_name not in _DIFFUSION_HPO_MODELS:
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
        "u_multilevel_vae": "umv",
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
        str: Readable value tags when short, otherwise a stable parameter hash.
    """

    # Values follow alphabetical parameter-name order. The full mapping is in
    # config.yaml and the TensorBoard text summary; omitting repeated long keys
    # keeps ordinary event names readable. Hash unusually wide conditional
    # spaces so the complete event path also stays below Windows MAX_PATH.
    parts = [f"t{trial.number:04d}"]
    for _, value in sorted(trial.params.items()):
        parts.append(_value_tag(value))

    name = "-".join(parts)
    # Use readable parameter tags while the event suffix remains short.
    if len(name) <= 80:
        return name

    payload = json.dumps(
        dict(sorted(trial.params.items())),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"t{trial.number:04d}-h{hashlib.sha256(payload).hexdigest()[:16]}"


def _suggest_optimizer(
    trial: Any, 
    family: str,
    allow_cosine: bool = True,
) -> dict[str, object]:
    """Suggest optimizer and batch settings for a model family.

    Args:
        trial (optuna.trial.Trial): Active Optuna trial.
        family (str): Normalized model-family name.
        allow_cosine (bool): Search cosine decay only when the update budget is known
            before training. Defaults to ``True``.

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
            "learning_rate", 1e-4, 3e-3, log=True
        )
        batch_choices = [128, 256, 512]
    # Search the classifier-specific learning-rate range.
    elif family in ("cnn", "dnn"):
        learning_rate = trial.suggest_float(
            "learning_rate", 1e-4, 3e-3, log=True
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
            "learning_rate", 3e-4, 5e-3, log=True
        )
        batch_choices = [32, 64, 128]

    batch_size = trial.suggest_categorical("batch_size", batch_choices)
    # Restrict diffusion optimization to Adam-family optimizers.
    if family in _DIFFUSION_MODELS:
        optimizer_choices = ["adam", "adamw"]
    # Include Nadam alongside Adam and AdamW for VAE models.
    elif family in ("vae", "vae_classifier"):
        optimizer_choices = ["adam", "adamw", "nadam"]
    # Use Adam or AdamW when fine-tuning the pretrained extractor.
    elif family == "pretrained":
        optimizer_choices = ["adam", "adamw"]
    # Offer gradient descent and RMSprop as well for ordinary classifiers.
    else:
        optimizer_choices = ["sgd", "rmsprop", "adam", "adamw", "nadam"]
    optimizer = trial.suggest_categorical("optimizer", optimizer_choices)
    # TensorFlow 2.10 exposes weight decay only through AdamW here.
    # Sample weight decay only for AdamW; leave it unset otherwise.
    weight_decay = trial.suggest_float(
        "weight_decay", 1e-6, 1e-3, log=True
    ) if optimizer == "adamw" else None
    # Sample momentum for SGD/RMSprop; other optimizers do not use it.
    momentum = trial.suggest_categorical(
        "momentum", [0., 0.3, 0.9, 0.95]
    ) if optimizer in ("sgd", "rmsprop") else 0.
    clipnorm = trial.suggest_categorical(
        "clipnorm", [None, 0.5, 1., 5.]
    )
    # Use constant learning rates when the update budget is not known in advance.
    if not allow_cosine:
        schedule_choices = ["constant"]
    # Use cosine decay for diffusion runs with a known update budget.
    elif family in _DIFFUSION_MODELS:
        schedule_choices = ["cosine"]
    # Compare constant and cosine schedules for the remaining model families.
    else:
        schedule_choices = ["cosine", "constant"]
    schedule = trial.suggest_categorical(
        "learning_rate_schedule", schedule_choices,
    )

    return {
        "batch_size": batch_size, 
        "optimizer": {
            "name": optimizer, 
            "initial_learning_rate": learning_rate, 
            "weight_decay": weight_decay, 
            "momentum": momentum,
            "clipnorm": clipnorm,
            "schedule": schedule
        }
    }


def _suggest_diffusion_wrapper(
    trial: Any,
    tune_sampling: bool = False,
    swap_noise_image: bool = False,
    fixed_kl_loss_coef: float | None = None,
) -> tuple[int, dict[str, object]]:
    """Suggest diffusion process and evaluation settings.

    Args:
        trial (optuna.trial.Trial): Active Optuna trial.
        tune_sampling (bool): Tune reverse-process settings used by continual generative
            replay. Loss-based generation/joint studies use fixed evaluation settings
            because these values do not affect their objectives. Defaults to ``False``.
        swap_noise_image (bool): Configure reconstruction of the noisy input ``x_t``. No
            auxiliary image-loss weight is sampled in this mode. Defaults to ``False``.
        fixed_kl_loss_coef (float | None): Immutable positive KL weight for input
            reconstruction. None searches the variational weight. Defaults to ``None``.

    Returns:
        tuple[int, dict[str, object]]: Training timestep count and wrapper
        keyword mapping.
    """

    timesteps = trial.suggest_categorical(
        "timesteps", [500, 1000]
    )
    wrapper_kwargs = {
        "use_ema": True, 
        "ema_decay": trial.suggest_categorical(
            "ema_decay", [0.995, 0.999]
        ), 
        "scheduler_name": trial.suggest_categorical(
            "schedule", [
                "clipped_cosine", "squaredcos_cap_v2", "linear",
            ]
        ), 
        "modify_first_t": False,
        "p_uncond": trial.suggest_categorical(
            "p_uncond", [0.05, 0.1, 0.2]
        ), 
        "test_network_name": "ema",
        # Disable auxiliary image loss for input reconstruction; otherwise tune its weight.
        "image_loss_coef": 0. if swap_noise_image else trial.suggest_categorical(
            "image_loss_coef", [0., 0.01, 0.05, 0.1]
        ),
    }
    # Include a variational KL term when reconstructing the noisy input.
    if swap_noise_image:
        # Keep a fixed KL weight when supplied; otherwise sample it.
        wrapper_kwargs["kl_loss_coef"] = fixed_kl_loss_coef \
            if fixed_kl_loss_coef is not None else trial.suggest_float(
                "kl_loss_coef", 1e-4, 1e-2, log=True
            )
    # Reverse-process settings affect continual replay samples, but not the
    # loss-based generation and joint objectives used by this module.
    if tune_sampling:
        test_step_choices = [
            value for value in (20, 50, 100)
            # Keep reverse-process step choices within the diffusion horizon.
            if value <= timesteps
        ]
        test_steps = trial.suggest_categorical(
            f"test_steps_t{timesteps}", test_step_choices
        )
        wrapper_kwargs.update({
            "test_steps": test_steps,
            "test_cfg_scale": trial.suggest_float(
                "test_cfg_scale", 2.5, 5.
            ),
            "test_eta": trial.suggest_categorical("test_eta", [0., 1.]),
        })
        set_user_attr = getattr(trial, "set_user_attr", None)
        # Record resolved sampling steps when trial metadata is supported.
        if callable(set_user_attr):
            set_user_attr("test_steps", test_steps)
    # Use fixed sampling settings when reverse sampling is not an objective dimension.
    else:
        wrapper_kwargs.update({
            "test_steps": min(50, timesteps),
            "test_cfg_scale": 4.,
            "test_eta": 0.,
        })

    return timesteps, wrapper_kwargs


def _fixed_dit_hpo_depth(
    model_overrides: Mapping[str, object] | None,
) -> int:
    """Resolve the immutable depth required by an x0 DiT topology.

    Positive stage IDs determine the smallest constructible depth.  A caller
    may provide a larger explicit ``depth``, but x0 HPO never samples depth
    independently from its fixed routing and reshaper graph.

    Args:
        model_overrides (Mapping[str, object] | None): Fixed DiT topology.

    Returns:
        int: Explicit or topology-derived transformer depth.

    Raises:
        ValueError: If no absolute stage can determine depth, or an explicit
            depth is smaller than a referenced stage.
    """

    overrides = dict(model_overrides or {})
    topology_ids: list[object] = []
    for name in (
        "vit_block_ids", "use_decoder_ids", "local_mixer_ids",
        "downsample_ids", "upsample_ids", "cls_token_regularizer_ids",
    ):
        topology_ids.extend(overrides.get(name, ()))

    for name in (
        "connection_ids_dict", "cross_attention_ids_dict",
        "vit_block_mlp_output_dims", "reshaper_ids_dict",
    ):
        values = overrides.get(name, {})
        topology_ids.extend(values)
        # Count source stages as well as destinations for connection routes.
        if name in ("connection_ids_dict", "cross_attention_ids_dict"):
            for sources in values.values():
                topology_ids.extend(sources)

    for name in (
        "feature_aggregation_ids_dict",
        "cross_attention_aggregation_ids_dict",
    ):
        for sources in overrides.get(name, {}).values():
            topology_ids.extend(sources)

    required_depth = max((
        depth for depth in topology_ids
        # Use positive absolute stage IDs to infer the minimum depth.
        if depth is not None and depth > 0
    ), default=0)
    configured_depth = overrides.get("depth")
    # Infer depth from topology when the caller did not fix it.
    if configured_depth is None:
        # Require explicit depth when relative IDs cannot determine an absolute bound.
        if required_depth == 0:
            raise ValueError(
                "DiT x0 HPO needs an explicit depth when its fixed topology "
                "contains no positive stage IDs."
            )
        return required_depth

    # Reject fixed depths that omit a referenced topology stage.
    if configured_depth < required_depth:
        raise ValueError(
            "DiT x0 HPO depth must cover every fixed topology stage; "
            f"expected at least {required_depth}, got {configured_depth}."
        )
    return configured_depth


def _validate_swap_noise_hpo(
    model_name: str,
    model_overrides: Mapping[str, object] | None,
    wrapper_overrides: Mapping[str, object] | None,
) -> tuple[bool, float | None]:
    """Validate the fixed bottleneck topology of an input-reconstruction study.

    Ordinary noise-prediction requests bypass these topology checks. Swap mode
    requires a KL-enabled bottleneck with no feature route bypassing the central
    variational boundary. DiT routes must form consecutive flatten/unflatten
    pairs with computation on both sides; U-Net templates generate their own
    reshaper IDs from their sampled widths.

    Args:
        model_name (str): Exact supported DiT or U-Net raw family selector.
        model_overrides (Mapping[str, object] | None): Immutable architecture
            settings, including reshaper_kwargs and, for DiT, explicit stage
            selections/routes. None supplies an empty mapping.
        wrapper_overrides (Mapping[str, object] | None): Fixed wrapper values.
            swap_noise_image selects reconstruction mode; optional kl_loss_coef
            fixes the KL weight. None supplies an ordinary noise-prediction
            request. An explicitly supplied noise_loss_coef must remain 1.

    Returns:
        tuple[bool, float | None]: Whether swap mode is enabled and its fixed KL
        coefficient, or None for a KL coefficient to be sampled later. Ordinary
        noise prediction returns (False, None).

    Raises:
        ValueError: If the family, reshaper pairs, stage ordering, connections,
            latent-ratio count, or fixed loss coefficients violate this mode.
        TypeError: If override containers or stage IDs have incompatible types.
    """

    wrapper = dict(wrapper_overrides or {})
    swap_noise_image = bool(wrapper.get("swap_noise_image", False))
    # Skip variational-topology checks for ordinary noise prediction.
    if not swap_noise_image:
        return False, None

    overrides = dict(model_overrides or {})
    reshaper_kwargs = overrides.get("reshaper_kwargs")
    # Require a KL-enabled latent bottleneck for input-reconstruction studies.
    if not isinstance(reshaper_kwargs, Mapping) \
    or reshaper_kwargs.get("add_kl") is not True:
        raise ValueError(
            "swap_noise_image HPO requires model_overrides with "
            "reshaper_kwargs={'add_kl': True}."
        )

    latent_dim_ratios = reshaper_kwargs.get("latent_dim_ratio")
    # Validate per-bottleneck latent ratios only when they were supplied.
    if latent_dim_ratios is not None:
        # Require an ordered list of per-bottleneck latent ratios.
        if not isinstance(latent_dim_ratios, list):
            raise ValueError(
                "reshaper_kwargs latent_dim_ratio must be a list."
            )

    # Let each U-Net width template define its own variational bottleneck.
    if model_name in ("unet", "unet_classifier"):
        # Reject fixed U-Net reshape routes that conflict with sampled widths.
        if overrides.get("reshaper_ids_dict"):
            raise ValueError(
                "U-Net x0 HPO must leave reshaper_ids_dict empty so each "
                "sampled width template creates its own bottleneck pair."
            )
    # Validate explicit central bottleneck routes for transformer families.
    elif model_name in (
        "diffusion_transformer", "dit_classifier", "dit_decoder",
        "dit_encoder_decoder",
        "dit_encoder_decoder_classifier",
    ):
        reshapers = dict(overrides.get("reshaper_ids_dict") or {})
        flatten_ids = sorted(
            depth for depth, reshape_type in reshapers.items()
            # Collect flatten stages, each of which must start a bottleneck pair.
            if reshape_type == "flatten"
        )
        # Reject missing, unpaired, or extra reshape stages.
        if not flatten_ids or any(
            reshapers.get(depth + 1) != "unflatten"
            for depth in flatten_ids
        ) or len(reshapers) != 2 * len(flatten_ids):
            raise ValueError(
                "DiT x0 HPO requires at least one consecutive "
                "flatten/unflatten pair."
            )
        # Require one supplied latent ratio for each bottleneck pair.
        if latent_dim_ratios is not None \
        and len(latent_dim_ratios) != len(flatten_ids):
            raise ValueError(
                "reshaper_kwargs latent_dim_ratio must contain exactly one "
                "entry per ascending flatten/unflatten pair; expected "
                f"{len(flatten_ids)}, got {len(latent_dim_ratios)}."
            )
        fixed_depth = _fixed_dit_hpo_depth(overrides)
        # Keep multiple bottleneck pairs adjacent in one central bridge.
        if len(flatten_ids) > 1 and sorted(reshapers) != list(range(
            flatten_ids[0], flatten_ids[-1] + 2
        )):
            raise ValueError(
                "Multilevel DiT x0 HPO reshaper pairs must form one "
                "contiguous central bridge."
            )

        first_bridge_depth = flatten_ids[0]
        last_bridge_depth = flatten_ids[-1] + 1
        # Require both an encoder side and a decoder side around the bridge.
        if first_bridge_depth <= 1 or fixed_depth <= last_bridge_depth:
            raise ValueError(
                "DiT x0 HPO requires at least one stage before and after its "
                "central bridge."
            )

        def resolved_stage_ids(name: str, required: bool = False) -> list[int]:
            """Resolve an explicit component selector against the fixed DiT depth.

            Args:
                name (str): Stage-list key in the enclosing architecture overrides.
                required (bool): Require an explicit mapping entry when True. Defaults to False,
                    which treats an omitted entry as an empty selection.

            Returns:
                list[int]: IDs in supplied order, with negative IDs shifted by
                fixed_depth + 1. Every returned ID is between 1 and fixed_depth.

            Raises:
                ValueError: If a required entry is absent, a None all-depth marker is
                    present, or a resolved ID is outside the fixed stage range.
                TypeError: If the supplied selector is not iterable or its items cannot
                    be compared and offset as integer stage IDs.
            """

            # Require explicit stage lists when constructor defaults would cross the bridge.
            if required and name not in overrides:
                raise ValueError(
                    "DiT x0 HPO requires explicit "
                    f"{name}; the constructor default crosses the central "
                    "bridge."
                )
            values = overrides.get(name, [])
            # Reject all-depth markers where absolute stage membership is required.
            if any(value is None for value in values):
                raise ValueError(
                    f"DiT x0 HPO {name} must contain explicit stage IDs."
                )
            resolved = [
                # Resolve negative stage IDs relative to fixed depth; preserve positive IDs.
                value + fixed_depth + 1 if value < 0 else value
                for value in values
            ]
            # Reject stage selections outside the fixed network depth.
            if any(not 1 <= value <= fixed_depth for value in resolved):
                raise ValueError(
                    f"DiT x0 HPO {name} contains an out-of-range stage ID."
                )
            return resolved

        vit_block_ids = resolved_stage_ids("vit_block_ids", required=True)
        local_mixer_ids = resolved_stage_ids("local_mixer_ids")
        downsample_ids = resolved_stage_ids("downsample_ids")
        upsample_ids = resolved_stage_ids("upsample_ids")
        use_decoder_ids = resolved_stage_ids("use_decoder_ids")
        # Keep transformer and local-mixer computation outside the reshape bridge.
        if any(
            first_bridge_depth <= depth <= last_bridge_depth
            for depth in (*vit_block_ids, *local_mixer_ids)
        ):
            raise ValueError(
                "DiT x0 HPO transformer/local-mixer stages cannot occur "
                "inside the central bridge."
            )
        # Require all spatial reductions to occur before the bridge.
        if any(depth >= first_bridge_depth for depth in downsample_ids):
            raise ValueError(
                "DiT x0 HPO downsample stages must precede the central bridge."
            )
        # Require all spatial expansions to occur after the bridge.
        if any(depth <= last_bridge_depth for depth in upsample_ids):
            raise ValueError(
                "DiT x0 HPO upsample stages must follow the central bridge."
            )
        # Require actual computation on both sides of the variational bridge.
        if not any(
            depth < first_bridge_depth
            for depth in (*vit_block_ids, *local_mixer_ids, *downsample_ids)
        ) or not any(
            depth > last_bridge_depth
            for depth in (*vit_block_ids, *local_mixer_ids, *upsample_ids)
        ):
            raise ValueError(
                "DiT x0 HPO requires actual encoder and decoder computation "
                "before and after its central bridge."
            )
        # Require decoder blocks after the bridge and encoder blocks before it.
        if any(depth <= last_bridge_depth for depth in use_decoder_ids) \
        or not set(use_decoder_ids).issubset(vit_block_ids) \
        or any(
            depth > last_bridge_depth and depth not in use_decoder_ids
            for depth in vit_block_ids
        ):
            raise ValueError(
                "DiT x0 HPO encoder blocks must precede the central bridge "
                "and decoder blocks must follow it."
            )
        first_flatten = flatten_ids[0]
        for route_name in (
            "connection_ids_dict",
            "cross_attention_ids_dict",
        ):
            routes = overrides.get(route_name, {})
            # Reject a connection that bypasses the variational bottleneck.
            if isinstance(routes, Mapping) and any(
                depth > first_flatten
                and reshapers.get(depth) != "flatten"
                and any(
                    source < first_flatten for source in sources
                )
                for depth, sources in routes.items()
            ):
                raise ValueError(
                    f"{route_name} cannot bypass the x0 variational boundary."
                )
    # Reject input reconstruction for unsupported raw-model families.
    else:
        raise ValueError(
            "swap_noise_image requires a supported DiT or U-Net family."
        )

    fixed_kl_loss_coef = wrapper.get("kl_loss_coef")
    # Validate the KL coefficient when the caller fixes it.
    if fixed_kl_loss_coef is not None:
        fixed_kl_loss_coef = float(fixed_kl_loss_coef)
        # Require a finite positive KL weight for a meaningful variational objective.
        if not math.isfinite(fixed_kl_loss_coef) \
        or fixed_kl_loss_coef <= 0.:
            raise ValueError("x0 kl_loss_coef must be finite and positive.")

    # Keep reconstruction at unit weight in the input-reconstruction objective.
    if "noise_loss_coef" in wrapper \
    and wrapper["noise_loss_coef"] != 1.:
        raise ValueError("x0 noise_loss_coef must remain 1.0.")

    return True, fixed_kl_loss_coef


def _suggest_dit(
    trial: Any, 
    image_size: int, 
    model_name: str,
    allow_u_shape: bool = True,
    fixed_depth: int | None = None,
) -> dict[str, object]:
    """Suggest a shape-compatible transformer architecture.

    Args:
        trial (optuna.trial.Trial): Active Optuna trial.
        image_size (int): Square input resolution.
        model_name (str): Selected DiT-family name.
        allow_u_shape (bool): Include the compact U-DiT template when its spatial grid
            is compatible. Defaults to ``True``.
        fixed_depth (int | None): Immutable topology-derived depth. ``None`` retains the
            ordinary depth search. Defaults to ``None``.

    Returns:
        dict[str, object]: Raw-network constructor options.
    """

    capacity = trial.suggest_categorical(
        "capacity", ["32x4", "64x4", "96x4", "128x4"]
    )
    dim, heads = (int(value) for value in capacity.split("x"))
    # Only sample patch sizes that divide the image width exactly.
    patch_choices = [value for value in (2, 4) if image_size % value == 0]
    kwargs = {
        "patchify_with_cnn": trial.suggest_categorical(
            "patchify_with_cnn", [False, True]
        ), 
        "patch_size": trial.suggest_categorical("patch_size", patch_choices), 
        "dim": dim, 
        "mha_num_heads": heads, 
        "vit_block_mlp_ratio": trial.suggest_categorical(
            "mlp_ratio", [2., 4.]
        ), 
        "drop_prob": trial.suggest_categorical(
            "drop_prob", [0., 0.05, 0.1]
        ), 
        "patches_pos_embed_type": "2d_sincos",
        "ln_no_adaptation": False,
        "use_refiner_cnn": trial.suggest_categorical(
            "use_refiner_cnn", [False, True]
        )
    }

    # Tune encoder and decoder depths independently for joint DiT models.
    if model_name in ("dit_encoder_decoder", "dit_encoder_decoder_classifier"):
        # Use immutable topology depth when supplied; otherwise tune encoder depth.
        kwargs["depth"] = fixed_depth if fixed_depth is not None \
            else trial.suggest_categorical("encoder_depth", [2, 4, 6])
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
        # Use immutable topology depth when supplied; otherwise tune decoder depth.
        kwargs["depth"] = fixed_depth if fixed_depth is not None \
            else trial.suggest_categorical("decoder_depth", [1, 2, 4])
        kwargs.update({
            "decoder_separate_cond": True, 
            "shift_inputs": False, 
            "use_causal_mask": False
        })
    # Compare the plain backbone with the notebook's compact, symmetric
    # feature-skip U-DiT when two spatial reductions are shape-compatible.
    elif model_name == "diffusion_transformer":
        patch_grid = image_size // kwargs["patch_size"]
        architecture_choices = ["plain"]
        # Offer a U-shaped backbone only when two reductions fit the patch grid.
        if fixed_depth is None and allow_u_shape and patch_grid % 4 == 0:
            architecture_choices.append("u_skip")
        architecture_parameter = (
            # Separate grid-compatible and plain-only categorical distributions in Optuna.
            "dit_architecture_grid4" if patch_grid % 4 == 0
            else "dit_architecture_plain"
        )
        architecture = trial.suggest_categorical(
            architecture_parameter, architecture_choices
        )
        set_user_attr = getattr(trial, "set_user_attr", None)
        # Record the chosen backbone architecture when trial metadata is supported.
        if callable(set_user_attr):
            set_user_attr("dit_architecture", architecture)
        # Use a stack of transformer blocks for the plain backbone.
        if architecture == "plain":
            # Respect fixed topology depth; otherwise sample the plain stack depth.
            kwargs["depth"] = fixed_depth if fixed_depth is not None else \
                trial.suggest_categorical("depth", [2, 3, 4, 5, 6])
        # Build the symmetric multiscale backbone for the U-shaped choice.
        else:
            resampling_pos = trial.suggest_categorical(
                "resampling_pos_embed_type", ["2d_sincos", "new_weight"]
            )
            kwargs.update({
                "depth": 9,
                "dim_forced": False,
                "connection_ids_dict": {7: [3, 6], 9: [1, 8]},
                "vit_block_ids": [1, 3, 5, 7, 9],
                "vit_block_mlp_output_dims": {
                    1: dim, 3: 2 * dim, 5: 2 * dim,
                    7: dim, 9: dim,
                },
                "downsample_ids": [2, 4],
                "downsample_kwargs": {
                    "use_layer_norm": True,
                    "ln_no_adaptation": False,
                    "scaling_method": "avg_pooling",
                    "pos_embed_type": resampling_pos,
                },
                "upsample_ids": [6, 8],
                "upsample_kwargs": {
                    "use_layer_norm": True,
                    "ln_no_adaptation": False,
                    "scaling_method": "interpolate",
                    "scaling_interpolation_method": "bilinear",
                    "pos_embed_type": resampling_pos,
                },
            })
    # Tune a single shared depth for the remaining transformer families.
    else:
        # Use fixed topology depth when supplied; otherwise sample shared depth.
        kwargs["depth"] = fixed_depth if fixed_depth is not None else \
            trial.suggest_categorical("depth", [2, 3, 4, 5, 6])

    return kwargs


def _suggest_unet(
    trial: Any, 
    classifier: bool = False
) -> dict[str, object]:
    """Suggest a convolutional U-Net architecture.

    Args:
        trial (optuna.trial.Trial): Active Optuna trial.
        classifier (bool): Include classifier-depth settings when true. Defaults to
            ``False``.

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
        # Choose average pooling or learned strided convolution for downsampling.
        "downsampling_method": "avg_pooling" if resampling == "pool" else "cnn_stride", 
        # Pair pooling with interpolation and strided convolution with transposed convolution.
        "upsampling_method": "interpolate" if resampling == "pool" else "cnn_transpose"
    }

    # Add classification-head settings for classifier-capable U-Nets.
    if classifier:
        aggregation = trial.suggest_categorical(
            "unet_feature_aggregation", ["last", "all"]
        )
        kwargs.update({
            "aggregate_from_noises": trial.suggest_categorical(
                "aggregate_from_noises", [False, True]
            ),
            "feature_aggregation_ids_dict": {
                # Aggregate only the final feature map or every denoiser feature map.
                1: [-1] if aggregation == "last" else [None]
            },
            "clf_depth": trial.suggest_categorical(
                "unet_clf_depth", [1, 2, 3]
            ),
            "clf_block_depth": trial.suggest_categorical(
                "clf_block_depth", [1, 2]
            ),
            "force_global_avg_pooling": trial.suggest_categorical(
                "force_global_avg_pooling", [False, True]
            ),
            "classifier_mlp_ratio": trial.suggest_categorical(
                "unet_classifier_mlp_ratio", [None, 1, 2]
            ),
            "classifier_mlp_activation_func": trial.suggest_categorical(
                "classifier_mlp_activation", ["tanh", "relu", "gelu"]
            ),
        })

    return kwargs


def _suggest_vae(
    trial: Any, 
    classifier: bool = False
) -> dict[str, object]:
    """Suggest a dense VAE architecture and loss coefficients.

    Args:
        trial (optuna.trial.Trial): Active Optuna trial.
        classifier (bool): Include the classifier coefficient when true. Defaults to
            ``False``.

    Returns:
        dict[str, object]: VAE constructor options.
    """

    latent_dim = trial.suggest_categorical(
        "latent_dim", [8, 16, 32, 64, 128]
    )
    template = trial.suggest_categorical(
        "hidden_template", [
            "16", "64-16", "256-64", "512-128", "512-256-64"
        ]
    )
    activation = trial.suggest_categorical("activation", ["relu", "selu"])
    # Disable batch normalization for SELU; otherwise let the trial choose.
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
            # Match SELU with LeCun initialization and ReLU with He initialization.
            "kernel_init": "lecun_normal" if activation == "selu" else "he_normal"
        }, 
        "beta": trial.suggest_float("beta", 0.01, 2., log=True)
    }

    # Tune joint classification weight only when a classifier branch exists.
    if classifier:
        kwargs["alpha"] = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)

    return kwargs


def _suggest_latent_dim_ratios(
    trial: Any,
    flattened_dims: Sequence[int],
) -> list[float]:
    """Suggest one absolute latent width per flatten/unflatten pair.

    The reshaper consumes ratios, but sampling absolute widths prevents the
    later, larger feature maps in a multilevel VAE from receiving
    disproportionately large latent projections.

    Args:
        trial (optuna.trial.Trial): Active Optuna trial.
        flattened_dims (Sequence[int]): Flattened input width for each pair in
            occurrence order.

    Returns:
        list[float]: Per-pair latent-width ratios in occurrence order.
    """

    return [
        trial.suggest_categorical(
            f"clf_latent_dim_pair{index}", [16, 32, 64]
        ) / flattened_dim
        for index, flattened_dim in enumerate(flattened_dims, start=1)
    ]


def _suggest_joint(
    trial: Any, 
    model_name: str, 
    kwargs: dict[str, object], 
    wrapper_kwargs: dict[str, object], 
    tune_masking: bool = True, 
    use_distillation: bool = False,
    image_size: int | None = None,
    clf_distil_scope: str = "current_and_replay",
) -> None:
    """Add joint-classification suggestions to mutable model settings.

    Args:
        trial (optuna.trial.Trial): Active Optuna trial.
        model_name (str): Selected joint model family.
        kwargs (dict[str, object]): Raw-model options updated in place.
        wrapper_kwargs (dict[str, object]): Wrapper options updated in place.
        tune_masking (bool): Suggest V1 classifier masking options when true. Defaults
            to ``True``.
        use_distillation (bool): Add a student distillation head and suggest
            teacher-loss settings for a runtime or continual snapshot teacher. Defaults
            to ``False``.
        image_size (int | None): Input width used to filter multiscale architectures
            whose down/up grids would not align. Defaults to ``None``.
        clf_distil_scope (str): Trial-selected continual teacher example scope. Defaults
            to ``'current_and_replay'``.

    Returns:
        None.
    """

    classifier_architecture = "linear"
    # Add DiT-specific architecture and conditioning choices.
    if model_name.startswith("dit"):
        architecture_choices = ["linear", "connection"]
        # Derive the patch grid when image size is known; otherwise leave it unspecified.
        patch_grid = None if image_size is None else (
            image_size // kwargs["patch_size"]
        )
        # Offer one spatial reduction when the known grid is divisible by two.
        if patch_grid is None or patch_grid % 2 == 0:
            architecture_choices.append("u_shape")
        # Offer deeper variational templates when two reductions fit the known grid.
        if patch_grid is None or patch_grid % 4 == 0:
            architecture_choices.extend(["u_vae", "u_multilevel_vae"])
        # Name the categorical distribution by grid divisibility, or use the unspecified-grid name.
        architecture_parameter = "classifier_architecture" if patch_grid is None \
            else "classifier_architecture_grid4" if patch_grid % 4 == 0 \
            else "classifier_architecture_grid2" if patch_grid % 2 == 0 \
            else "classifier_architecture_flat"
        classifier_architecture = trial.suggest_categorical(
            architecture_parameter, architecture_choices,
        )
        set_user_attr = getattr(trial, "set_user_attr", None)
        # Record the classifier architecture when the trial supports attributes.
        if callable(set_user_attr):
            set_user_attr("classifier_architecture", classifier_architecture)
        classifier_only_cls_token = trial.suggest_categorical(
            "classifier_only_cls_token", [True, False]
        )
        feature_aggregation = trial.suggest_categorical(
            "feature_aggregation", ["last", "early", "all"]
        )
        # Preserve the existing classifier-depth search for the linear baseline.
        if classifier_architecture == "linear":
            clf_depth = trial.suggest_categorical(
                "clf_depth", [1, 2, 3]
            )
        # Use two stages for the compact feature-connection template.
        elif classifier_architecture == "connection":
            clf_depth = 2
        # Use a down/bottleneck/up classifier for the U-shaped template.
        elif classifier_architecture == "u_shape":
            clf_depth = 4
        # Use one central stochastic bottleneck in the compact U-VAE.
        elif classifier_architecture == "u_vae":
            clf_depth = 11
        # Use three stochastic latent levels in the nested U-VAE.
        else:
            clf_depth = 15

        kwargs.update({
            "feature_aggregation_ids_dict": {
                1: (
                    # Route final, first, or all denoiser features to classifier stage one.
                    [-1] if feature_aggregation == "last"
                    else [1] if feature_aggregation == "early"
                    else [None]
                )
            }, 
            "classifier_only_cls_token": classifier_only_cls_token, 
            "clf_depth": clf_depth,
            "clf_ln_no_adaptation": False,
            "clf_drop_prob": trial.suggest_categorical(
                "clf_drop_prob", [0., 0.05, 0.1]
            ), 
            "classifier_mlp_ratio": trial.suggest_categorical(
                "classifier_mlp_ratio", [None, 1]
            ), 
            "dropout_rate": trial.suggest_categorical(
                "dropout_rate", [0., 0.05, 0.1]
            )
        })

        # Merge the classifier input and first-stage output before stage two.
        if classifier_architecture == "connection":
            kwargs.update({
                "clf_connection_ids_dict": {2: [0, 1], -1: [-1]},
                "clf_connection_kwargs": {
                    "connect_type": trial.suggest_categorical(
                        "clf_connection_type", ["add", "concat"]
                    )
                }
            })
        # Build a compact spatial downsample/bottleneck/upsample classifier.
        elif classifier_architecture == "u_shape":
            kwargs.update({
                "clf_vit_block_ids": [1, 2, 4],
                "clf_downsample_ids": [1], 
                "clf_downsample_kwargs": {"scaling_method": "avg_pooling"}, 
                "clf_upsample_ids": [3], 
                "clf_connection_ids_dict": {4: [0, 3], -1: [-1]},
                "clf_connection_kwargs": {"connect_type": "add"},
                "clf_upsample_kwargs": {
                    "scaling_method": "interpolate", 
                    "scaling_interpolation_method": "bilinear"
                }
            })
        # Insert a KL-enabled flatten/unflatten bottleneck into the U-shape.
        elif classifier_architecture == "u_vae":
            prefix_tokens = int(classifier_only_cls_token) + int(
                use_distillation
            )
            deepest_width = 2 * kwargs["dim"] * (
                (patch_grid // 4) ** 2 + prefix_tokens
            )
            kwargs.update({
                "clf_vit_block_ids": [1, 3, 5, 9, 11],
                "clf_use_decoder_ids": [9, 11],
                "clf_vit_block_mlp_output_dims": {
                    1: kwargs["dim"], 3: 2 * kwargs["dim"],
                    5: 2 * kwargs["dim"], 9: kwargs["dim"],
                    11: kwargs["dim"],
                },
                "clf_downsample_ids": [2, 4],
                "clf_downsample_kwargs": {
                    "scaling_method": "avg_pooling",
                    "pos_embed_type": "2d_sincos",
                },
                "clf_reshaper_ids_dict": {
                    6: "flatten", 7: "unflatten",
                },
                "clf_reshaper_kwargs": {
                    "add_kl": True, 
                    "latent_dim_ratio": _suggest_latent_dim_ratios(
                        trial, [deepest_width]
                    ),
                }, 
                "clf_upsample_ids": [8, 10],
                "clf_upsample_kwargs": {
                    "scaling_method": "interpolate", 
                    "scaling_interpolation_method": "bilinear",
                    "pos_embed_type": "2d_sincos",
                },
                "clf_cross_attention_ids_dict": {9: [3], 11: [1]},
                "clf_cross_attention_kwargs": {
                    "use_layer_norm": True,
                    "ln_no_adaptation": False,
                },
            })
        # Stack three KL bottlenecks and restore both same-grid U skips.
        elif classifier_architecture == "u_multilevel_vae":
            prefix_tokens = int(classifier_only_cls_token) + int(
                use_distillation
            )
            flattened_dims = [
                2 * kwargs["dim"] * (
                    (patch_grid // 4) ** 2 + prefix_tokens
                ),
                2 * kwargs["dim"] * (
                    (patch_grid // 2) ** 2 + prefix_tokens
                ),
                kwargs["dim"] * (patch_grid ** 2 + prefix_tokens),
            ]
            kwargs.update({
                "clf_vit_block_ids": [1, 3, 5, 13, 15],
                "clf_use_decoder_ids": [13, 15],
                "clf_vit_block_mlp_output_dims": {
                    1: kwargs["dim"], 3: 2 * kwargs["dim"],
                    5: 2 * kwargs["dim"], 13: kwargs["dim"],
                    15: kwargs["dim"],
                },
                "clf_downsample_ids": [2, 4],
                "clf_downsample_kwargs": {
                    "scaling_method": "avg_pooling",
                    "pos_embed_type": "2d_sincos",
                },
                "clf_reshaper_ids_dict": {
                    6: "flatten", 7: "unflatten",
                    8: "flatten", 9: "unflatten",
                    10: "flatten", 11: "unflatten",
                },
                "clf_reshaper_kwargs": {
                    "add_kl": True,
                    "latent_dim_ratio": _suggest_latent_dim_ratios(
                        trial, flattened_dims
                    ),
                },
                "clf_connection_ids_dict": {
                    8: [3], 10: [1], 12: [7], -1: [-1]
                },
                "clf_upsample_ids": [12, 14],
                "clf_upsample_kwargs": {
                    "scaling_method": "interpolate",
                    "scaling_interpolation_method": "bilinear",
                    "pos_embed_type": "2d_sincos",
                },
                "clf_cross_attention_ids_dict": {13: [9], 15: [11]},
                "clf_cross_attention_kwargs": {
                    "use_layer_norm": True,
                    "ln_no_adaptation": False,
                },
            })

        # Project concatenated all-depth features back to the classifier width.
        if feature_aggregation == "all":
            kwargs.update({
                "clf_dim": kwargs["dim"], 
                "clf_dim_forced": True,
            })

        # Tune a dedicated classifier token only when the branch uses one.
        if classifier_only_cls_token:
            kwargs["clf_cls_token_type"] = trial.suggest_categorical(
                "clf_cls_token_type", ["new_weight", "time_label"]
            )

        # Weight the classifier's variational bottleneck only when it exists.
        if classifier_architecture in ("u_vae", "u_multilevel_vae") \
        and "kl_loss_coef" not in wrapper_kwargs:
            wrapper_kwargs["kl_loss_coef"] = trial.suggest_float(
                "kl_loss_coef", 1e-3, 3e-2, log=True
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

        clf_distil_type = trial.suggest_categorical(
            "clf_distil_type", ["hard", "soft"]
        )
        wrapper_kwargs.update({
            "clf_distil_type": clf_distil_type,
            # Tune soft-target temperature; hard distillation keeps temperature one.
            "clf_distil_temperature": trial.suggest_float(
                "clf_distil_temperature", 0.5, 8., log=True
            ) if clf_distil_type == "soft" else 1.,
            "clf_distil_scope": clf_distil_scope,
            "clf_distil_loss_coef": trial.suggest_float(
                "clf_distil_loss_coef", 1e-4, 1e-1, log=True
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

        # Regularize the variational bridge when present; otherwise use the last classifier stage.
        regularizer_depth = 6 if classifier_architecture in (
            "u_vae", "u_multilevel_vae"
        ) else kwargs["clf_depth"]
        kwargs["clf_cls_token_regularizer_ids"] = [regularizer_depth]
        regularizer_kwargs = {
            "start": (
                # Start at the class token when present; otherwise skip any distillation token.
                0 if kwargs.get("classifier_only_cls_token", False)
                else int(use_distillation)
            ),
            "end": (
                # Select exactly one regularized token after the corresponding start position.
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
        # Give the distillation head a vote only when distillation is enabled.
        "clf_distil_acc_coef": accuracy_coef if use_distillation else 0., 
        # Give the regularizer head a vote only when its loss is active.
        "ctr_acc_coef": accuracy_coef if ctr_loss_coef > 0. else 0.
    })

    wrapper_kwargs["clf_loss_coef"] = trial.suggest_float(
        "clf_loss_coef", 1e-3, 3e-2, log=True
    )

    # Tune classifier masking only for the jointly trained V1 wrapper.
    if tune_masking:
        clf_train_type = trial.suggest_categorical(
            "clf_train_type", ["cond", "uncond"]
        )
        wrapper_kwargs["clf_train_type"] = clf_train_type
        # Use unit classifier-free guidance for unconditional classifier training.
        if clf_train_type == "uncond":
            wrapper_kwargs["train_cfg_scale"] = 1.

        masking = trial.suggest_categorical(
            "masking", ["neither", "null", "timestep", "both"]
        )
        wrapper_kwargs.update({
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
        template = trial.suggest_categorical(
            "cnn_template", ["cifar", "compact", "wide"]
        )
        filters, depths = {
            "cifar": ((64, 128, 128, 256), (1, 2, 2, 1)),
            "compact": ((32, 64, 128), (1, 1, 1)),
            "wide": ((64, 128, 256), (1, 2, 2)),
        }[template]

        return {
            "dropout_rate": trial.suggest_categorical(
                "dropout", [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
            ),
            "architecture_kwargs": {
                "conv_filters": filters, 
                "conv_depths": depths,
                "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]), 
                "first_kernel_size": trial.suggest_categorical(
                    "first_kernel", [3, 5, 7]
                ), 
                "use_batch_norm": trial.suggest_categorical(
                    "batch_norm", [False, True]
                ), 
                "pooling": "max",
                "global_pooling": "avg",
            }
        }

    # Build a tunable dense hidden-layer template.
    if model_name == "dnn":
        template = trial.suggest_categorical(
            "hidden_template", [
                "linear", "256-128", "512-256", "1024-512"
            ]
        )
        # Use a direct linear classifier when the template has no hidden layers.
        if template == "linear":
            return {
                "dropout_rate": trial.suggest_categorical(
                    "dropout", [0., 0.1, 0.25, 0.5]
                ),
                "architecture_kwargs": {"hidden_dims": ()},
            }
        activation = trial.suggest_categorical(
            "activation", ["relu", "elu", "selu"]
        )

        return {
            "dropout_rate": trial.suggest_categorical(
                "dropout", [0., 0.1, 0.25, 0.5]
            ),
            "architecture_kwargs": {
                "hidden_dims": tuple(int(value) for value in template.split("-")), 
                "activation": activation, 
                # Disable batch normalization for SELU; tune it for other activations.
                "use_batch_norm": False if activation == "selu" else
                                trial.suggest_categorical("batch_norm", [False, True]), 
                # Use LeCun initialization with SELU and He initialization otherwise.
                "kernel_initializer": "lecun_normal" if activation == "selu"
                                    else "he_normal", 
            },
        }

    return {
        "dropout_rate": trial.suggest_float(
            "dropout", 0., 0.5, step=0.1
        ), 
        "num_last_not_frozen": trial.suggest_categorical(
            "unfrozen", [1, 5, 12, 20, None]
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
            classification objective. Defaults to ``False``.

    Returns:
        tuple[str, ...]: generation gives ('generation_loss',); joint gives
        generation_loss followed by classification_accuracy or ensemble_accuracy;
        classification gives ('validation_accuracy',); continual gives
        ('final_average_accuracy',).

    Raises:
        ValueError: If task is not one of these four HPO families.
    """

    # Name the legacy scalar generation target.
    if task == "generation":
        return ("generation_loss",)
    # Name both legacy joint Pareto targets.
    if task == "joint":
        return (
            "generation_loss",
            # Use ensemble accuracy when requested; otherwise use ordinary classifier accuracy.
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
        str: Inferred ``"minimize"`` or ``"maximize"`` direction.

    Raises:
        ValueError: If the custom metric name has no reliable convention.
    """

    normalized = metric_name.lower()
    # Minimize conventional cost, error, resource, and forgetting names.
    if any(token in normalized for token in (
        "loss", "error", "forgetting", "latency", "memory", "rmse",
        "mae", "mse", "nll", "cost", "runtime", "parameter_count",
        "params", "flops",
    )):
        return "minimize"
    # Maximize predictive-quality and transfer metrics.
    if any(token in normalized for token in (
        "accuracy", "auc", "precision", "recall", "f1", "f-score",
        "backward_transfer", "forward_transfer",
    )):
        return "maximize"
    raise ValueError(
        "Cannot infer the objective direction for metric "
        f"{metric_name!r}; provide objective_directions explicitly."
    )


def _normalize_objective_spec(
    task: str, 
    objective_metrics: str | Sequence[str] | None = None, 
    objective_directions: str | Sequence[str] | None = None, 
    use_ensemble_accuracy: bool = False
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Normalize objective names/directions and validate their cardinality.

    Args:
        task (str): Normalized HPO task name.
        objective_metrics (str | Sequence[str] | None): Optional metric name or ordered
            names. Defaults to ``None``.
        objective_directions (str | Sequence[str] | None): Optional matching Optuna
            directions. Defaults to ``None``.
        use_ensemble_accuracy (bool): Select the ensemble joint default name. Defaults
            to ``False``.

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
        directions = tuple(_inferred_objective_direction(name) for name in metrics)
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


def _feature_archive_signature(
    base_path: str | Path | None,
    dataset_name: str | None = None,
) -> dict[str, object] | None:
    """Validate a safe continual-VAE bundle and fingerprint its exact contents.

    The archive must be a non-pickled ZIP/NumPy bundle containing ordered train,
    validation, and test arrays named arr_0, arr_1, and arr_2. Each is a nonempty
    rank-two feature matrix with width 2048. Companion split metadata is checked
    by the shared provenance loader and included in the immutable study record.

    Args:
        base_path (str | pathlib.Path | None): Archive base without its '.npy'
            and '.metadata.json' suffixes. None disables archive inspection.
        dataset_name (str | None): Optional dataset identity marker required in the
            archive filename. Defaults to None to omit the filename check.

    Returns:
        dict[str, object] | None: File size, SHA-256 hex digest, split shapes,
        NumPy dtype strings, and parsed provenance metadata; None when no base
        path was supplied. The matrices themselves are not returned.

    Raises:
        FileNotFoundError: If the data bundle or companion metadata is absent.
        ValueError: If the dataset marker, archive format, split layout, feature
            shapes, or split provenance is invalid; includes malformed JSON.
        OSError: If archive or metadata files cannot be read.
    """

    # Omit feature-archive metadata when no archive is configured.
    if base_path is None:
        return None
    path = Path(str(base_path) + ".npy")
    # Reject a feature archive path that has no data file.
    if not path.is_file():
        raise FileNotFoundError("Feature archive does not exist: " + str(path))
    # Check dataset identity when a dataset name was supplied.
    if dataset_name is not None:
        marker = f"{dataset_name.lower()}_"
        # Reject archive names that identify a different dataset.
        if marker not in path.stem.lower():
            raise ValueError(
                f"Feature archive name must contain the dataset marker "
                f"{marker!r}."
            )

    with path.open("rb") as stream:
        # Require the safe ZIP bundle format instead of pickled NumPy data.
        if stream.read(4) != b"PK\x03\x04":
            raise ValueError(
                "Continual VAE HPO requires a non-pickled safe feature bundle."
            )
    with np.load(path, allow_pickle=False) as bundle:
        expected_keys = ["arr_0", "arr_1", "arr_2"]
        # Require train, validation, and test members in their documented order.
        if bundle.files != expected_keys:
            raise ValueError(
                "Feature archive must contain train, validation, and test "
                "members in arr_0..arr_2 order."
            )
        arrays = [bundle[key] for key in expected_keys]
        for split_name, array in zip(
            ("train", "validation", "test"),
            arrays,
        ):
            # Reject empty or incorrectly shaped feature matrices.
            if array.ndim != 2 or array.shape[0] == 0 \
            or array.shape[1] != 2048:
                raise ValueError(
                    f"{split_name} features must be a nonempty "
                    "[samples, 2048] array."
                )
        split_shapes = [list(array.shape) for array in arrays]
        split_dtypes = [array.dtype.str for array in arrays]

    metadata_path = Path(str(base_path) + ".metadata.json")
    # Require the companion metadata that records safe split provenance.
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Feature archive split metadata does not exist: "
            + str(metadata_path)
        )
    load_feature_split_metadata(base_path)
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "split_shapes": split_shapes,
        "split_dtypes": split_dtypes,
        "metadata": metadata,
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
    task_order_mode: str,
    feature_archive_path: str | Path | None = None,
    model_overrides: Mapping[str, object] | None = None,
    wrapper_overrides: Mapping[str, object] | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    n_startup_trials: int = 10,
    search_space_overrides: Mapping[str, object] | None = None,
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
        feature_archive_path (str | pathlib.Path | None): Safe feature bundle used by
            continual VAE families. Defaults to ``None``.
        model_overrides (Mapping[str, object] | None): Fixed raw architecture controls
            outside the sampled template. Defaults to ``None``.
        wrapper_overrides (Mapping[str, object] | None): Fixed wrapper controls outside
            the sampled template. Defaults to ``None``.
        max_train_samples (int | None): Fixed development training-row cap. Defaults to
            ``None``.
        max_val_samples (int | None): Fixed development validation-row cap. Defaults to
            ``None``.
        n_startup_trials (int): Random observations before TPE model fitting. Defaults
            to ``10``.
        search_space_overrides (Mapping[str, object] | None): Study-level categorical
            choices or numeric low/high bounds. Defaults to ``None``.

    Returns:
        dict[str, object]: Strict JSON-safe immutable study specification.
    """

    return _study_json_value({
        "schema_version": 1, 
        "search_space_version": SEARCH_SPACE_VERSION,
        "search_space_fingerprint": fingerprint_state(
            SEARCH_SPACES[task][model_name]
        ),
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
        "feature_archive": _feature_archive_signature(
            feature_archive_path,
            dataset_name,
        ),
        "model_overrides": dict(model_overrides or {}),
        "wrapper_overrides": dict(wrapper_overrides or {}),
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "n_startup_trials": int(n_startup_trials),
        "search_space_overrides": dict(search_space_overrides or {}),
        "objective_metrics": list(objective_metrics), 
        "objective_directions": list(objective_directions), 
        "dtype_policy": dtype_policy, 
        "deterministic_ops": bool(deterministic_ops), 
        "snapshot_network_name": snapshot_network_name,
        "continual_schedule": {
            "class_num": class_num, 
            # Keep an omitted class order distinct from an explicitly supplied order.
            "class_order": None if class_order is None else list(class_order), 
            # Serialize explicit task groups as lists; preserve their absence otherwise.
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
    snapshot_network_name: str = "search",
    class_num: int | None = None,
    class_order: Sequence[int] | None = None,
    task_groups: Sequence[Sequence[int]] | None = None,
    task_size: int = 1,
    class_order_mode: str = "fixed",
    task_order_mode: str = "fixed",
    feature_archive_path: str | Path | None = None,
    model_overrides: Mapping[str, object] | None = None,
    wrapper_overrides: Mapping[str, object] | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    search_space_overrides: Mapping[str, object] | None = None,
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
        use_ensemble_accuracy (bool): Use timestep-ensemble accuracy for joint or continual
            diffusion classifiers. For continual runs it supplies the authoritative task
            accuracy matrix, so every selected derived continual metric uses that same
            matrix. False retains ordinary classifier predictions. Defaults to ``False``.
        ensemble_accuracy_kwargs (Mapping[str, object] | None): Options for
            DiffusionClassifier.evaluate_ensemble_accuracy/EnsembleAccuracy: max_t,
            'chunked'/'batched' compute_type, t_chunk_size, SNR weighted flag, seed,
            separate_probas, network_name, and head coefficients. None uses wrapper
            defaults. Continual runs honor an explicit network_name and use a task-derived
            seed when seed is omitted; ordinary reporting selects its raw/EMA branch.
            Defaults to ``None``.
        use_distillation (bool): Enable teacher-dependent objectives: noise distillation for
            supported diffusion generators and optional hard/soft classifier-token
            distillation for classifier families. Continual runs may start teacher-free and
            snapshot each completed raw/EMA student; non-continual runs need a runtime
            teacher. The caller's teacher_network also activates this space in run_hpo.
            Defaults to ``False``.
        fit_method (str): ``"fit"`` for ordinary Keras training or
            ``"fit_progressively"`` for a diffusion curriculum. Defaults to ``'fit'``.
        fit_kwargs (Mapping[str, object] | None): YAML-safe fit arguments; None supplies no
            extras. Progressive mode requires stage_tasks and accepts named curriculum
            controls, which are stored in TrainingConfig's explicit fields. Remaining
            ordinary Keras fit keys are stored in TrainingConfig.fit_kwargs. Defaults to
            ``None``.
        objective_metrics (str | Sequence[str] | None): One metric name or an ordered
            nonempty sequence. None chooses task defaults: generation_loss; joint
            generation_loss plus classification_accuracy (ensemble_accuracy when enabled);
            classification validation_accuracy; or continual final_average_accuracy.
            Continual names must exist exactly in validation_continual_metrics, for example
            final_average_accuracy, average_incremental_accuracy, forgetting, or
            backward_transfer. Defaults to ``None``.
        objective_directions (str | Sequence[str] | None): One 'minimize'/'maximize' string
            or an ordered sequence matching the metric count. None infers directions by
            metric-name conventions: losses/errors/forgetting minimize; accuracies and
            forward/backward transfer maximize. Unrecognized custom names require explicit
            directions. Defaults to ``None``.
        dtype_policy (str): Keras numeric policy installed before construction. Defaults
            to ``'float32'``.
        deterministic_ops (bool): Request deterministic TensorFlow kernels. Defaults to
            ``False``.
        snapshot_network_name (str): ``"raw"``, ``"ema"``, or ``"search"`` for a
            trial-selected continual teacher snapshot. Defaults to ``'search'``.
        class_num (int | None): Selected continual class count. None infers it from explicit
            class_order/task_groups or the dataset class count when no schedule is supplied;
            the resolved schedule needs at least two tasks. Defaults to ``None``.
        class_order (Sequence[int] | None): Ordered unique original dataset labels. None
            uses flattened explicit task_groups or the natural order; an explicit
            class_order must agree with explicit groups. Defaults to ``None``.
        task_groups (Sequence[Sequence[int]] | None): Nonempty groups of original labels
            introduced per task. None groups the resolved class order by task_size; the
            final group may be shorter. Defaults to ``None``.
        task_size (int): Classes per automatically constructed task. Defaults to ``1``.
        class_order_mode (str): 'fixed' keeps original order; 'random' shuffles classes with
            the study seed before automatic grouping. Explicit task_groups require fixed
            class ordering. Defaults to ``'fixed'``.
        task_order_mode (str): 'fixed' preserves group order; 'random' shuffles whole groups
            with the study seed while preserving each group's internal order. An
            automatically generated short remainder is not placed first. Defaults to
            ``'fixed'``.
        feature_archive_path (str | pathlib.Path | None): Base path of the safe
            continual-CIFAR VAE feature archive. None uses the family's conventional
            data/<dataset>_xception_gavgpooled_features_train_val_test_safe base when
            needed; otherwise no feature archive is used. Defaults to ``None``.
        model_overrides (Mapping[str, object] | None): Immutable raw-model controls outside
            sampled template keys, such as bottleneck routes. None adds no overrides.
            Mapping-valued settings merge only when nested keys are disjoint; conflicts with
            sampled values are rejected. The umbrella diffusion_classifier family requires
            choosing an exact raw family before using these controls. Defaults to ``None``.
        wrapper_overrides (Mapping[str, object] | None): Immutable diffusion-wrapper
            constructor controls outside sampled keys, for an exact raw family. None adds no
            overrides; conflicting sampled keys and non-diffusion families are rejected. Fix
            the separately sampled wrapper family through
            search_space_overrides={'wrapper_name': ['diffusion_classifier_v2']}, not
            through this mapping. Defaults to ``None``.
        max_train_samples (int | None): Fixed positive development training-row cap shared
            by all trials; None keeps the full split. Continual caps preserve every selected
            class. This is an experiment budget, not a sampled hyperparameter. Defaults to
            ``None``.
        max_val_samples (int | None): Fixed positive development validation-row cap shared
            by all trials; None keeps the full split. Classes are preserved, and test rows
            are never substituted. Defaults to ``None``.
        search_space_overrides (Mapping[str, object] | None): Immutable parameter-name
            mapping of categorical scalars/lists or {'choices': ...}, and numeric {'low':
            ..., 'high': ..., 'step': ..., 'log': ...} overrides. None leaves template
            distributions unchanged. Names may be prefixed by a raw family and may omit
            topology suffixes; _TrialView resolves their precedence and validates
            categorical choices. Defaults to ``None``.

    Returns:
        Config: Fully typed development-run configuration with a validation split,
        resolved model/wrapper/schedule settings, sealed HPO metadata, and
        ensemble/distillation controls. Config construction does not train a model.

    Raises:
        ValueError: If the dataset/model combination or fit selection is
            unsupported, or progressive training omits ``stage_tasks``.
    """

    dataset_name = dataset_name.lower()
    study_model_name = model_name
    base_trial = trial
    # Choose a raw classifier family for an umbrella diffusion-classifier study.
    if model_name == _DIFFUSION_CLASSIFIER_STUDY:
        # Require an exact family when fixed architecture or wrapper overrides are supplied.
        if model_overrides or wrapper_overrides:
            raise ValueError(
                "model_overrides and wrapper_overrides require an exact "
                "diffusion classifier family, not diffusion_classifier."
            )
        selector = _TrialView(
            base_trial,
            overrides=search_space_overrides,
        )
        model_name = selector.suggest_categorical("model_family", [
            "dit_classifier",
            "dit_encoder_decoder_classifier",
            "unet_classifier",
        ])
        selector.set_user_attr("model_family", model_name)
        trial = _TrialView(
            base_trial,
            prefix=model_name + ".",
            overrides=search_space_overrides,
        )
    # Apply unprefixed overrides when the study already names an exact family.
    elif search_space_overrides:
        trial = _TrialView(
            base_trial,
            overrides=search_space_overrides,
        )
    fit_kwargs = dict(fit_kwargs or {})
    fixed_model_overrides = dict(model_overrides or {})
    fixed_wrapper_overrides = dict(wrapper_overrides or {})
    # Reject wrapper overrides for model families without a diffusion wrapper.
    if fixed_wrapper_overrides and model_name not in _DIFFUSION_MODELS:
        raise ValueError("wrapper_overrides requires a diffusion model family.")
    swap_noise_image = fixed_wrapper_overrides.get("swap_noise_image", False)
    # Reject noise distillation together with noisy-input reconstruction.
    if swap_noise_image and use_distillation:
        raise ValueError(
            "noise distillation is incompatible with "
            "swap_noise_image=True input reconstruction."
        )
    swap_noise_image, fixed_kl_loss_coef = _validate_swap_noise_hpo(
        model_name,
        fixed_model_overrides,
        fixed_wrapper_overrides,
    )
    # Fix depth from immutable topology only for transformer input reconstruction.
    fixed_dit_depth = _fixed_dit_hpo_depth(fixed_model_overrides) if (
        swap_noise_image and (
            model_name.startswith("dit")
            or model_name == "diffusion_transformer"
        )
    ) else None
    fixed_wrapper_overrides.pop("swap_noise_image", None)
    # Remove the fixed KL override after the reconstruction helper has consumed it.
    if swap_noise_image:
        fixed_wrapper_overrides.pop("kl_loss_coef", None)
    snapshot_network_name = str(snapshot_network_name).lower()
    class_order_mode = str(class_order_mode).lower()
    task_order_mode = str(task_order_mode).lower()
    # Materialize an explicit class order while preserving the default unset value.
    class_order = None if class_order is None else list(class_order)
    # Materialize explicit task groups while preserving the default unset value.
    task_groups = None if task_groups is None else [
        list(group) for group in task_groups
    ]
    # Reject teacher snapshot selectors unsupported by diffusion wrappers.
    if snapshot_network_name not in ("raw", "ema", "search"):
        raise ValueError(
            "snapshot_network_name must be 'raw', 'ema', or 'search'."
        )
    # Resolve the requested teacher-branch search into one trial setting.
    if snapshot_network_name == "search":
        # Search raw versus EMA teachers only for distillation; otherwise retain raw.
        snapshot_network_name = trial.suggest_categorical(
            "snapshot_network_name", ["raw", "ema"]
        ) if use_distillation else "raw"
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
    resolved_task_groups = None
    # Validate direct builder calls with the same canonical schedule resolver.
    if task == "continual":
        _, resolved_task_groups = resolve_continual_schedule(
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
    optimization = _suggest_optimizer(
        trial,
        model_name,
        allow_cosine=task != "continual" and fit_method != "fit_progressively",
    )

    model_kwargs = {}
    wrapper_kwargs = {}
    classifier_name = None
    classifier_kwargs = {}
    wrapper_name = None
    # Standardize diffusion inputs; use min-max scaling for the remaining families.
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
        # Compare MSE and MAE for VAEs; diffusion objectives retain MSE.
        loss_choices = ["mse", "mae"] if model_name in (
            "vae", "vae_classifier"
        ) else ["mse"]
        loss_function = trial.suggest_categorical(
            "loss_function", loss_choices
        )

    # Teacher objectives operate on every diffusion wrapper.
    if use_distillation and model_name not in _DIFFUSION_MODELS:
        raise ValueError(
            "Distillation requires a diffusion model family."
        )

    # Tune transformer diffusion schedules and wrapper behavior.
    if model_name.startswith("dit") or model_name == "diffusion_transformer":
        timesteps, wrapper_kwargs = _suggest_diffusion_wrapper(
            trial,
            tune_sampling=task == "continual" and not swap_noise_image,
            swap_noise_image=swap_noise_image,
            fixed_kl_loss_coef=fixed_kl_loss_coef,
        )
        # Carry input-reconstruction mode into the transformer wrapper configuration.
        if swap_noise_image:
            wrapper_kwargs["swap_noise_image"] = swap_noise_image
        model_kwargs = _suggest_dit(
            trial,
            image_size,
            model_name,
            allow_u_shape=not swap_noise_image,
            fixed_depth=fixed_dit_depth,
        )
        model_kwargs.update({"timesteps": timesteps, "use_cfg": True})
    # Tune U-Net diffusion schedules and wrapper behavior.
    elif model_name in ("unet", "unet_classifier"):
        timesteps, wrapper_kwargs = _suggest_diffusion_wrapper(
            trial,
            tune_sampling=task == "continual" and not swap_noise_image,
            swap_noise_image=swap_noise_image,
            fixed_kl_loss_coef=fixed_kl_loss_coef,
        )
        # Carry input-reconstruction mode into the U-Net wrapper configuration.
        if swap_noise_image:
            wrapper_kwargs["swap_noise_image"] = swap_noise_image
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

        # Attach a dense classifier only for the joint VAE-classifier family.
        classifier_name = "dnn" if model_name == "vae_classifier" else None
        classifier_kwargs = {
            "dropout_rate": 0.2, 
            "architecture_kwargs": {"hidden_dims": (256,), "activation": "relu"}
        }
    # Tune standalone classifier architecture for classification tasks.
    elif task in ("classification", "continual") \
    and model_name in ("cnn", "dnn", "pretrained"):
        model_kwargs = _suggest_classifier(trial, model_name)

        # Normalize the flattened raw pixels used by dense-classifier trials.
        if model_name == "dnn":
            preprocess = "normalize"
        # Preserve raw image scale for pretrained preprocessing layers.
        elif model_name == "pretrained":
            preprocess = None

    continual_strategy = "generative_replay"
    use_generative_replay = True
    remove_prev_classes = True
    clf_distil_scope = "current_and_replay"
    # Search a continual replay strategy for diffusion classifiers.
    if task == "continual" and model_name in _DIFFUSION_CLASSIFIER_MODELS:
        singleton_first = len(resolved_task_groups[0]) == 1
        # Give singleton-first and multiclass-first strategies separate Optuna distributions.
        strategy_parameter = "continual_strategy_singleton" \
            if singleton_first else "continual_strategy_multiclass"
        continual_strategy = trial.suggest_categorical(
            strategy_parameter,
            # Exclude new-only training for singleton starts; allow it for multiclass starts.
            ["generative_replay", "cumulative"] if singleton_first else [
                "generative_replay", "new_only", "cumulative"
            ],
        )
        use_generative_replay = continual_strategy == "generative_replay"
        remove_prev_classes = continual_strategy != "cumulative"
        # Search teacher example scope only when distillation is enabled.
        if use_distillation:
            scope_choices = {
                "generative_replay": [
                    "old_classes", "replay_only", "current_and_replay"
                ],
                "cumulative": ["old_classes", "current_and_replay"],
                "new_only": ["current_and_replay"],
            }[continual_strategy]
            clf_distil_scope = trial.suggest_categorical(
                "clf_distil_scope_" + continual_strategy,
                scope_choices,
            )
        set_user_attr = getattr(trial, "set_user_attr", None)
        # Record the resolved replay strategy when trial attributes are supported.
        if callable(set_user_attr):
            set_user_attr("continual_strategy", continual_strategy)
            # Record the teacher scope only for a distillation trial.
            if use_distillation:
                set_user_attr("clf_distil_scope", clf_distil_scope)

    # Configure optional noise distillation for a diffusion student.
    if use_distillation and model_name in _DIFFUSION_MODELS:
        use_noise_distillation = model_name not in _DIFFUSION_CLASSIFIER_MODELS \
            or trial.suggest_categorical(
                "use_noise_distillation", [False, True]
            )
        wrapper_kwargs["noise_distil_loss_coef"] = (
            # Sample the noise-teacher loss weight when active; otherwise set it to zero.
            trial.suggest_float("noise_distil_loss_coef", 1e-4, 1., log=True)
            if use_noise_distillation else 0.
        )

    # Add joint loss/search options for supported generative classifiers.
    if task in ("joint", "continual") and model_name in (
        "dit_classifier", "dit_encoder_decoder_classifier", 
        "unet_classifier"
    ):
        # Compare both wrappers for every classifier-capable diffusion family.
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
            image_size=image_size,
            clf_distil_scope=clf_distil_scope,
        )
        # Tune classifier-input noising and shared-variable recipes for V2.
        if wrapper_name == "diffusion_classifier_v2":
            wrapper_kwargs["mask_by_nulls"] = False
            clf_timestep_choices = [None]
            clf_timestep_choices.extend(
                value for value in (64, 256)
                # Keep classifier-noising caps below the selected diffusion horizon.
                if value < timesteps
            )
            timestep_parameter = (
                f"clf_train_noisified_max_timesteps_t{timesteps}"
            )
            clf_max_timesteps = trial.suggest_categorical(
                timestep_parameter,
                clf_timestep_choices,
            )
            wrapper_kwargs["clf_train_noisified_max_timesteps"] = (
                clf_max_timesteps
            )
            set_user_attr = getattr(trial, "set_user_attr", None)
            # Record the resolved V2 classifier-noising cap when attributes are supported.
            if callable(set_user_attr):
                set_user_attr(
                    "clf_train_noisified_max_timesteps",
                    wrapper_kwargs["clf_train_noisified_max_timesteps"],
                )
            variable_recipe = trial.suggest_categorical(
                "clf_vars_recipe", [
                    "separate", "conditions", "notebook"
                ]
            )
            wrapper_kwargs["clf_vars_embedding_ids"] = {
                "separate": [],
                "conditions": [1, 2],
                "notebook": [0, 1, 2, 3],
            }[variable_recipe]
            wrapper_kwargs["clf_vars_noise_part_ids"] = {
                "separate": [],
                "conditions": [],
                "notebook": [1],
            }[variable_recipe]

    continual_kwargs = {}
    # Tune replay policy only for continual-learning studies.
    if task == "continual":
        classifier_only = model_name in ("cnn", "dnn", "pretrained")
        # Disable generative replay for standalone classifier studies.
        if classifier_only:
            use_generative_replay = False
        train_num = -1
        replay_samples = 0
        replay_budget_mode = "legacy"
        replay_old_examples = None
        replay_current_examples = None
        replay_selection = "all"
        replay_candidate_multiplier = 1
        replay_surprise_weight = 0.5
        # Search replay budgets and selection only when a generator supplies replay.
        if not classifier_only and use_generative_replay:
            replay_budget_mode = trial.suggest_categorical(
                "replay_budget_mode", ["legacy", "fixed_total"]
            )
            # Sample per-class replay and generator-training counts in legacy budget mode.
            if replay_budget_mode == "legacy":
                replay_samples = trial.suggest_categorical(
                    "replay_samples", [100, 500, 1_000, 2_500, 5_000]
                )
                train_num = trial.suggest_categorical(
                    "train_num", [-1, 1_000, 2_500, 5_000, 7_500, 10_000]
                )
            # Sample exact old/current exposure counts in fixed-total budget mode.
            else:
                replay_old_examples = trial.suggest_categorical(
                    "replay_old_examples", [100, 500, 1_000, 2_500, 5_000]
                )
                replay_current_examples = trial.suggest_categorical(
                    "replay_current_examples",
                    [100, 500, 1_000, 2_500, 5_000],
                )
            replay_selection = trial.suggest_categorical(
                "replay_selection", [
                    "all", "uniform", "confidence", "surprise",
                    "confidence_surprise",
                ]
            )
            # Search an enlarged candidate pool when a selection rule filters replay.
            if replay_selection != "all":
                replay_candidate_multiplier = trial.suggest_categorical(
                    "replay_candidate_multiplier", [1, 2, 4]
                )
            # Tune the relative surprise weight only for the combined selection rule.
            if replay_selection == "confidence_surprise":
                replay_surprise_weight = trial.suggest_float(
                    "replay_surprise_weight", 0., 1.
                )
        # Tune generator-training exposure even when the generative model supplies no replay.
        elif not classifier_only:
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
            "remove_prev_classes": remove_prev_classes,
            "keep_same_model": True, 
            "use_generative_replay": use_generative_replay,
            "replay_budget_mode": replay_budget_mode,
            "replay_old_examples": replay_old_examples,
            "replay_current_examples": replay_current_examples,
            "replay_candidate_multiplier": replay_candidate_multiplier,
            "replay_selection": replay_selection,
            "replay_surprise_weight": replay_surprise_weight,
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

        # Choose a non-generative continual protocol for standalone classifiers.
        if classifier_only:
            # Exclude sequential-only training for singleton starts; allow it for multiclass starts.
            protocol_choices = ["cumulative", "reservoir_er"] \
                if len(resolved_task_groups[0]) == 1 \
                else ["sequential", "cumulative", "reservoir_er"]
            # Separate singleton and multiclass protocol choices in Optuna storage.
            protocol_name = "continual_protocol_singleton" \
                if len(resolved_task_groups[0]) == 1 \
                else "continual_protocol_multiclass"
            protocol = trial.suggest_categorical(
                protocol_name,
                protocol_choices,
            )
            set_user_attr = getattr(trial, "set_user_attr", None)
            # Record the standalone continual protocol when attributes are supported.
            if callable(set_user_attr):
                set_user_attr("continual_protocol", protocol)
            continual_kwargs["baseline"] = protocol
            # Tune buffer capacity, sampling, and insertion for reservoir replay.
            if protocol == "reservoir_er":
                replay_capacity = trial.suggest_categorical(
                    "replay_buffer_capacity", [2_500, 5_000, 10_000]
                )
                replay_sample_count = trial.suggest_categorical(
                    "replay_buffer_sample_count", [500, 1_000, 2_500]
                )
                replay_insert_count = trial.suggest_categorical(
                    "replay_buffer_insert_count", [500, 1_000]
                )
                continual_kwargs["buffer_kwargs"] = {
                    "maxlen": replay_capacity,
                    "sample_num": replay_sample_count,
                    "insert_num": replay_insert_count,
                    "strategy": "reservoir",
                }

        # Train and evaluate every diffusion classifier's attached head.
        if model_name in _DIFFUSION_CLASSIFIER_MODELS:
            continual_kwargs.update({
                "use_generative_model_classifier": True, 
                "train_classifier_separately": (
                    wrapper_name == "diffusion_classifier_v2"
                )
            })
        # Use the attached classifier for the joint VAE family.
        elif model_name == "vae_classifier":
            continual_kwargs["use_generative_model_classifier"] = True

        # Select and configure a dense classifier for VAE replay.
        if model_name in ("vae", "vae_classifier"):
            classifier_name = "dnn"
            return_features = dataset_name in ("cifar10", "cifar100")
            preprocess, onehot_labels = "normalize", True

            # Point dense VAE replay at the dataset's saved feature archive.
            if return_features:
                features_path = str(feature_archive_path or (
                    Path("data")
                    / (
                        f"{dataset_name}_xception_gavgpooled_features_"
                        "train_val_test_safe"
                    )
                ))

            model_kwargs["last_activation"] = "linear"

            classifier_kwargs = {
                "dropout_rate": 0.2, 
                "architecture_kwargs": {
                    "hidden_dims": (256,), 
                    "activation": "relu"
                }
            }
        # Select a convolutional classifier for image-space replay.
        elif not classifier_only:
            classifier_name = "cnn"
            preprocess = "min-max"
            classifier_kwargs = {
                "dropout_rate": 0.2, 
                "architecture_kwargs": {
                    "conv_filters": (32, 64, 128), 
                    "conv_depths": (1, 1, 1)
                }
            }

    for name, value in fixed_model_overrides.items():
        # The x0 topology resolver already installed this compatible fixed depth.
        if name == "depth" and fixed_dit_depth is not None:
            continue
        # Check fixed overrides that target an already sampled raw-model setting.
        if name in model_kwargs:
            # Merge mapping-valued overrides only when their nested keys are disjoint.
            if isinstance(model_kwargs[name], Mapping) \
            and isinstance(value, Mapping):
                duplicate = set(model_kwargs[name]) & set(value)
                # Reject fixed nested settings that overwrite a sampled dimension.
                if duplicate:
                    raise ValueError(
                        f"model_overrides[{name!r}] replaces sampled keys: "
                        f"{sorted(duplicate)}"
                    )
                model_kwargs[name] = {
                    **model_kwargs[name], **dict(value)
                }
                continue
            raise ValueError(
                f"model_overrides replaces sampled setting {name!r}."
            )
        model_kwargs[name] = value

    for name, value in fixed_wrapper_overrides.items():
        # Reject fixed wrapper settings that overwrite a sampled dimension.
        if name in wrapper_kwargs:
            raise ValueError(
                f"wrapper_overrides replaces sampled setting {name!r}."
            )
        wrapper_kwargs[name] = value

    # Optimize validation accuracy for standalone classification.
    if task == "classification":
        monitor, monitor_mode = "val_accuracy", "max"
    # Optimize both generation loss and classification accuracy jointly.
    elif task == "joint":
        # Monitor VAE classifier accuracy, combined active heads, or the primary diffusion head.
        monitor = "val_clf_accuracy" if model_name == "vae_classifier" \
                else "val_total_accuracy" if use_distillation or \
                    wrapper_kwargs.get("ctr_loss_coef", 0.) > 0. \
                else "val_classifier_accuracy"
        monitor_mode = "max"
    # Optimize the final continual-learning accuracy.
    elif task == "continual":
        monitor, monitor_mode = "val_accuracy", "max"
    # Restore generator-only VAE/x0 weights by their full variational loss.
    elif model_name in ("vae", "vae_classifier") or swap_noise_image:
        monitor, monitor_mode = "val_loss", "min"
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
    trial_root = Path(results_path) / task / study_model_name / dataset_name

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
        task_tag + _MODEL_TAGS[study_model_name] + dataset_tag
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
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
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
            # Enable early stopping for scalar generation/classification studies only.
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
            "save_final_gifs": (
                model_name in _DIFFUSION_MODELS
                and task != "continual"
                and not swap_noise_image
            ),
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
            "study_model": study_model_name,
            "model_family": model_name,
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
            # Record replay strategy only for continual diffusion-classifier studies.
            "continual_strategy": continual_strategy if task == "continual"
            and model_name in _DIFFUSION_CLASSIFIER_MODELS else None,
            # Record teacher example scope only when distillation is enabled.
            "clf_distil_scope": clf_distil_scope \
                if use_distillation else None,
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


def _validation_evaluation_value(
    evaluations: Mapping[str, object],
    model_name: str,
    names: Sequence[str],
    diffusion_network_name: str = "ema",
) -> float:
    """Read one metric from the post-training validation evaluation.

    Args:
        evaluations (Mapping[str, object]): Final report evaluation mapping.
        model_name (str): Model family used to select standard or diffusion
            report keys.
        names (Sequence[str]): Candidate names in preference order.
        diffusion_network_name (str): ``"ema"`` or ``"raw"`` validation branch for
            diffusion wrappers. Defaults to ``'ema'``.

    Returns:
        float: Selected metric value.

    Raises:
        KeyError: If the selected validation report or metric is unavailable.
        TypeError: If the selected metric is not scalar.
        ValueError: If the network selector is invalid.
    """

    # Read a raw or EMA report for diffusion model families.
    if model_name in _DIFFUSION_MODELS:
        # Reject an unsupported diffusion evaluation branch.
        if diffusion_network_name not in ("ema", "raw"):
            raise ValueError("diffusion_network_name must be 'ema' or 'raw'.")
        # Select the EMA validation report or the raw-network validation report.
        report_key = "valset_ema_eval" if diffusion_network_name == "ema" \
            else "valset_network_eval"
    # Read the ordinary validation report for non-diffusion models.
    else:
        report_key = "valset_eval"

    results = evaluations.get(report_key)
    # Fail when the required post-training validation report is absent.
    if not isinstance(results, Mapping):
        raise KeyError(f"No post-training validation report named {report_key}.")
    for name in names:
        # Try the next candidate metric when this name was not reported.
        if name not in results:
            continue
        value = np.asarray(results[name])
        # Reject vector-valued metrics as individual Optuna objectives.
        if value.ndim != 0:
            raise TypeError("Validation objective must be scalar: " + name)
        return float(value)

    raise KeyError(
        f"None of the objective metrics were reported in {report_key}: "
        + ", ".join(names)
    )


def _x0_generation_value(
    evaluations: Mapping[str, object],
    model_name: str,
    kl_loss_coef: float,
    diffusion_network_name: str,
) -> float:
    """Combine noisy-input reconstruction and the weighted main-latent KL.

    In swap mode the metric named noise_loss measures reconstruction of x_t.
    The returned scalar is noise_loss + kl_loss_coef * kl_loss. Classifier
    losses and auxiliary classifier latents do not enter this generative score.

    Args:
        evaluations (Mapping[str, object]): Post-training validation reports.
        model_name (str): Diffusion family whose report namespace is selected.
        kl_loss_coef (float): Fixed or sampled multiplier for the main latent KL.
        diffusion_network_name (str): 'ema' or 'raw' validation network branch.

    Returns:
        float: Weighted generative score from one selected validation report.
        Undefined numeric values are preserved for Optuna's failed-trial handling.

    Raises:
        KeyError: If the validation report or either required metric is absent.
        TypeError: If a reported objective is not scalar or cannot be converted.
        ValueError: If the diffusion branch selector or numeric conversion fails.
    """

    reconstruction = _validation_evaluation_value(
        evaluations,
        model_name,
        ("noise_loss",),
        diffusion_network_name,
    )
    kl_loss = _validation_evaluation_value(
        evaluations,
        model_name,
        ("kl_loss",),
        diffusion_network_name,
    )
    return reconstruction + float(kl_loss_coef) * kl_loss


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
    evaluations: Mapping[str, object], 
    metric_name: str,
    diffusion_network_name: str = "ema",
    swap_noise_image: bool = False,
    kl_loss_coef: float = 0.,
) -> float:
    """Resolve one objective from the appropriate validation report.

    Args:
        task (str): Normalized HPO task name.
        model_name (str): Normalized model-family name.
        evaluations (Mapping[str, object]): Final report evaluation mapping.
        metric_name (str): Exact metric name or semantic alias. generation_loss chooses
            family-specific reconstruction/generative loss (plus weighted KL in swap mode);
            classification_accuracy chooses the first available classifier accuracy;
            validation_accuracy reads accuracy. Ordinary explicit names accept
            val_/validation_ prefixes; continual names are exact keys in
            validation_continual_metrics.
        diffusion_network_name (str): Selected diffusion validation branch. Defaults to
            ``'ema'``.
        swap_noise_image (bool): Whether diffusion reconstructs the noisy input ``x_t``.
            Defaults to ``False``.
        kl_loss_coef (float): Main variational KL coefficient in x0 mode. Defaults to
            ``0.0``.

    Returns:
        float: Selected validation scalar, preserving undefined numeric values.

    Raises:
        KeyError: If the required validation report/metric is missing.
        TypeError: If a selected metric is not scalar or cannot be converted.
        ValueError: If the diffusion branch selector or numeric conversion fails.
    """

    # Enforce the dedicated validation-only continual namespace.
    if task == "continual":
        return _continual_validation_value(evaluations, metric_name)

    # Resolve a model-family-aware semantic generation loss alias.
    if metric_name == "generation_loss":
        # Combine reconstruction and weighted KL for the input-reconstruction objective.
        if swap_noise_image:
            return _x0_generation_value(
                evaluations,
                model_name,
                kl_loss_coef,
                diffusion_network_name,
            )
        # Use joint VAE generative loss, standalone VAE loss, or diffusion noise loss.
        names = [
            "generative_loss",
        ] if model_name == "vae_classifier" else [
            "loss", "total_loss", "recon_loss",
        ] if model_name == "vae" else ["noise_loss", "loss"]
        return _validation_evaluation_value(
            evaluations,
            model_name,
            names,
            diffusion_network_name,
        )
    # Resolve the semantic joint classifier accuracy alias.
    if metric_name == "classification_accuracy":
        return _validation_evaluation_value(
            evaluations,
            model_name,
            (
                "total_accuracy",
                "classifier_accuracy",
                "cls_token_accuracy",
                "avg_pooling_accuracy",
                "clf_accuracy",
            ),
            diffusion_network_name,
        )
    # Resolve the semantic standalone validation accuracy alias.
    if metric_name == "validation_accuracy":
        return _validation_evaluation_value(
            evaluations,
            model_name,
            ("accuracy",),
            diffusion_network_name,
        )

    # Explicit non-continual objectives are always resolved from validation.
    evaluation_names = [metric_name]
    # Resolve a short validation-prefixed name to its report metric first.
    if metric_name.startswith("val_"):
        evaluation_names.insert(0, metric_name[len("val_"):])
    # Resolve a full validation-prefixed name to its report metric first.
    if metric_name.startswith("validation_"):
        evaluation_names.insert(0, metric_name[len("validation_"):])

    return _validation_evaluation_value(
        evaluations,
        model_name,
        evaluation_names,
        diffusion_network_name,
    )


def _objective_values(
    task: str, 
    model_name: str,  
    history: Mapping[str, Sequence[float]], 
    evaluations: Mapping[str, object] | None = None, 
    use_ensemble_accuracy: bool = False,
    objective_metrics: str | Sequence[str] | None = None,
    objective_directions: str | Sequence[str] | None = None,
    diffusion_network_name: str = "ema",
    swap_noise_image: bool = False,
    kl_loss_coef: float = 0.,
) -> float | tuple[float, ...]:
    """Convert training outputs to the study's objective value or tuple.

    Args:
        task (str): Study task.
        model_name (str): Selected model family.
        history (Mapping[str, Sequence[float]]): Training metric history.
            Retained for call compatibility; objective values come from the
            post-training evaluation mapping.
        evaluations (Mapping[str, object] | None): Post-training report mapping. Although
            the compatibility default is None, successful objective extraction requires the
            relevant validation report; an absent mapping becomes empty and raises KeyError
            for the requested metric. Defaults to ``None``.
        use_ensemble_accuracy (bool): Select ensemble instead of ordinary classification
            accuracy. Defaults to ``False``.
        objective_metrics (str | Sequence[str] | None): Explicit metric names. Continual
            names are resolved only under
            ``evaluations['validation_continual_metrics']``. Defaults to ``None``.
        objective_directions (str | Sequence[str] | None): Matching Optuna directions
            for explicit post-training validation metrics. Defaults to ``None``.
        diffusion_network_name (str): Raw or EMA post-training validation branch used
            for diffusion objectives. Defaults to ``'ema'``.
        swap_noise_image (bool): Whether diffusion reconstructs the noisy input ``x_t``.
            Defaults to ``False``.
        kl_loss_coef (float): Main variational KL coefficient in x0 mode. Defaults to
            ``0.0``.

    Returns:
        float | tuple[float, ...]: One float for one metric, otherwise a tuple in
        configured metric order. NaNs remain undefined for Optuna's failed-trial
        handling; no training/test fallback or invented finite score is used.

    Raises:
        ValueError: If ensemble feedback is incompatible with the family/task,
            metric directions are invalid, or a diffusion branch is unsupported.
        KeyError: If a required validation report or objective metric is absent.
        TypeError: If objective specifications are malformed or a metric is not scalar.
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

    del history

    metrics, _ = _normalize_objective_spec(
        task,
        objective_metrics,
        objective_directions,
        use_ensemble_accuracy,
    )
    evaluations = evaluations or {}

    # Defaults and explicit names share one validation-only resolution path.
    # Optuna handles NaN results as failed trials; do not replace undefined
    # scientific metrics with invented scores or abort the whole study.
    values = tuple(
        _configured_objective_value(
            task,
            model_name,
            evaluations,
            metric_name,
            diffusion_network_name,
            swap_noise_image,
            kl_loss_coef,
        )
        for metric_name in metrics
    )
    # Return one scalar for a single objective and a tuple for multiple objectives.
    return values[0] if len(values) == 1 else values


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
    snapshot_network_name: str = "search",
    class_num: int | None = None,
    class_order: Sequence[int] | None = None,
    task_groups: Sequence[Sequence[int]] | None = None,
    task_size: int = 1,
    class_order_mode: str = "fixed",
    task_order_mode: str = "fixed",
    feature_archive_path: str | Path | None = None,
    model_overrides: Mapping[str, object] | None = None,
    wrapper_overrides: Mapping[str, object] | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    n_startup_trials: int = 10,
    search_space_overrides: Mapping[str, object] | None = None,
) -> Any:
    """Run a persistent Optuna study and return its ``Study`` object.

    By default, ``joint`` studies are Pareto searches with generative loss
    minimized and classification accuracy maximized, while continual studies
    maximize validation ``final_average_accuracy``. Explicit metric sequences
    create matching multi-objective studies. Each trial saves all normal
    training artifacts plus its input/resolved config; the dataset-specific
    study directory stores SQLite state and an incrementally updated CSV.

    Training resource-exhaustion errors are recorded as failed trials and the
    search continues; undefined NaN objectives are left for Optuna to reject.
    Other training/configuration errors propagate so an invalid experiment is
    not silently treated as a successful objective.

    Args:
        task (str): ``generation``, ``joint``, ``classification``, or
            ``continual``.
        model_name (str): A key in ``SEARCH_SPACES[task]``. Continual
            ``"diffusion_classifier"`` searches all three raw classifier
            families in one conditionally prefixed study.
        dataset_name (str): ``FMNIST``, ``MNIST``, ``CIFAR10``, or ``CIFAR100``.
            Xception requires a three-channel CIFAR dataset. Defaults to ``'CIFAR10'``.
        n_trials (int): Positive number of additional trials to run. Defaults to ``30``.
        epochs (int): Positive ordinary-fit budget. Progressive diffusion phases use
            ``stage_epochs`` and ``final_epochs`` instead; this value still sizes
            existing cosine schedules and any ordinary continual classifier phase.
            Defaults to ``30``.
        seed (int): Fixed split, model-initialization, and training seed across all trials;
            Optuna's independently seeded sampler supplies hyperparameter variation.
            Defaults to ``42``.
        results_path (str): HPO root. Study state is written below
            ``<task>/<model>/<dataset>`` and TensorBoard events below ``_tb``. Defaults
            to ``'results/hpo'``.
        timeout (float | None): Study optimization wall-time limit in seconds. None imposes
            no time limit. Optuna checks the limit between trials, so a running fit is not
            interrupted immediately. Defaults to ``None``.
        use_ensemble_accuracy (bool): Use timestep-ensemble accuracy for joint or continual
            diffusion classifiers. For continual runs it supplies the authoritative task
            accuracy matrix, so every selected derived continual metric uses that same
            matrix. False retains ordinary classifier predictions. Defaults to ``False``.
        ensemble_accuracy_kwargs (Mapping[str, object] | None): Options for
            DiffusionClassifier.evaluate_ensemble_accuracy/EnsembleAccuracy: max_t,
            'chunked'/'batched' compute_type, t_chunk_size, SNR weighted flag, seed,
            separate_probas, network_name, and head coefficients. None uses wrapper
            defaults. Continual runs honor an explicit network_name and use a task-derived
            seed when seed is omitted; ordinary reporting selects its raw/EMA branch.
            Defaults to ``None``.
        teacher_network (tf.keras.Model | None): Runtime-only frozen teacher. Supplying
            it enables hard/soft distillation suggestions and gives continual task one
            an optional initial teacher. The object is never written to trial YAML.
            Defaults to ``None``.
        use_distillation (bool): Enable teacher-dependent objectives: noise distillation for
            supported diffusion generators and optional hard/soft classifier-token
            distillation for classifier families. Continual runs may start teacher-free and
            snapshot each completed raw/EMA student; non-continual runs need a runtime
            teacher. The caller's teacher_network also activates this space in run_hpo.
            Defaults to ``False``.
        fit_method (str): ``"fit"`` for the ordinary epoch loop or
            ``"fit_progressively"`` for diffusion curriculum training. Defaults to
            ``'fit'``.
        fit_kwargs (Mapping[str, object] | None): YAML-safe fit arguments; None supplies no
            extras. Progressive mode requires stage_tasks and accepts named curriculum
            controls, which are stored in TrainingConfig's explicit fields. Remaining
            ordinary Keras fit keys are stored in TrainingConfig.fit_kwargs. Defaults to
            ``None``.
        objective_metrics (str | Sequence[str] | None): One metric name or an ordered
            nonempty sequence. None chooses task defaults: generation_loss; joint
            generation_loss plus classification_accuracy (ensemble_accuracy when enabled);
            classification validation_accuracy; or continual final_average_accuracy.
            Continual names must exist exactly in validation_continual_metrics, for example
            final_average_accuracy, average_incremental_accuracy, forgetting, or
            backward_transfer. Defaults to ``None``.
        objective_directions (str | Sequence[str] | None): One 'minimize'/'maximize' string
            or an ordered sequence matching the metric count. None infers directions by
            metric-name conventions: losses/errors/forgetting minimize; accuracies and
            forward/backward transfer maximize. Unrecognized custom names require explicit
            directions. Defaults to ``None``.
        dtype_policy (str): Keras numeric policy recorded for every trial. Defaults to
            ``'float32'``.
        deterministic_ops (bool): Request deterministic TensorFlow kernels. Defaults to
            ``False``.
        resume_from (str | pathlib.Path | None): Existing HPO study root containing study.db
            and its immutable specification. None starts/loads the conventional
            task/model/dataset study directory. A supplied path refers to a study, not an
            individual model file; scientific settings must match the stored study. Defaults
            to ``None``.
        snapshot_network_name (str): ``"raw"``, ``"ema"``, or ``"search"`` for a
            trial-selected previous-task teacher branch. Defaults to ``'search'``.
        class_num (int | None): Selected continual class count. None infers it from explicit
            class_order/task_groups or the dataset class count when no schedule is supplied;
            the resolved schedule needs at least two tasks. Defaults to ``None``.
        class_order (Sequence[int] | None): Ordered unique original dataset labels. None
            uses flattened explicit task_groups or the natural order; an explicit
            class_order must agree with explicit groups. Defaults to ``None``.
        task_groups (Sequence[Sequence[int]] | None): Nonempty groups of original labels
            introduced per task. None groups the resolved class order by task_size; the
            final group may be shorter. Defaults to ``None``.
        task_size (int): Classes per automatically constructed task. Defaults to ``1``.
        class_order_mode (str): 'fixed' keeps original order; 'random' shuffles classes with
            the study seed before automatic grouping. Explicit task_groups require fixed
            class ordering. Defaults to ``'fixed'``.
        task_order_mode (str): 'fixed' preserves group order; 'random' shuffles whole groups
            with the study seed while preserving each group's internal order. An
            automatically generated short remainder is not placed first. Defaults to
            ``'fixed'``.
        feature_archive_path (str | pathlib.Path | None): Base path without '.npy' of a safe
            continual CIFAR VAE feature bundle. None resolves to
            data/<dataset>_xception_gavgpooled_features_train_val_test_safe for that
            task/family and is otherwise unused. Supplying an archive for another family is
            rejected. Defaults to ``None``.
        model_overrides (Mapping[str, object] | None): Immutable raw-model controls outside
            sampled template keys, such as bottleneck routes. None adds no overrides.
            Mapping-valued settings merge only when nested keys are disjoint; conflicts with
            sampled values are rejected. The umbrella diffusion_classifier family requires
            choosing an exact raw family before using these controls. Defaults to ``None``.
        wrapper_overrides (Mapping[str, object] | None): Immutable diffusion-wrapper
            constructor controls outside sampled keys, for an exact raw family. None adds no
            overrides; conflicting sampled keys and non-diffusion families are rejected. Fix
            the separately sampled wrapper family through
            search_space_overrides={'wrapper_name': ['diffusion_classifier_v2']}, not
            through this mapping. Defaults to ``None``.
        max_train_samples (int | None): Fixed positive development training-row cap shared
            by all trials; None keeps the full split. Continual caps preserve every selected
            class. This is an experiment budget, not a sampled hyperparameter. Defaults to
            ``None``.
        max_val_samples (int | None): Fixed positive development validation-row cap shared
            by all trials; None keeps the full split. Classes are preserved, and test rows
            are never substituted. Defaults to ``None``.
        n_startup_trials (int): Random observations before TPE begins using its fitted
            density model. Defaults to ``10``.
        search_space_overrides (Mapping[str, object] | None): Immutable parameter-name
            mapping of categorical scalars/lists or {'choices': ...}, and numeric {'low':
            ..., 'high': ..., 'step': ..., 'log': ...} overrides. None leaves template
            distributions unchanged. Names may be prefixed by a raw family and may omit
            topology suffixes; _TrialView resolves their precedence and validates
            categorical choices. Defaults to ``None``.

    Example:
        To keep a distilled continual study on V2 while optimizing two metrics
        from one ensemble matrix, use task='continual', model_name='dit_classifier',
        use_distillation=True, use_ensemble_accuracy=True,
        objective_metrics=['final_average_accuracy', 'forgetting'], and
        search_space_overrides={'wrapper_name': ['diffusion_classifier_v2']}.
        Directions infer to maximize accuracy and minimize forgetting. The saved
        trial YAML includes both HPO objective metadata and continual ensemble/KD
        settings, and can be loaded through common.config.load_config.

    Returns:
        optuna.study.Study: Resumable completed/partial study. Multi-objective
        studies expose ``best_trials``; scalar studies expose ``best_trial``.

    Raises:
        ImportError: If Optuna is unavailable.
        TypeError: If objective specifications or serialized override values have
            incompatible types.
        AttributeError: If task/model/dataset selectors do not support string
            normalization.
        ValueError: If the task/model pair, fit selection, or positive budgets
            are invalid, progressive training omits ``stage_tasks``, or a resume
            specification differs from the stored study.
        FileNotFoundError: If a requested resume study or feature archive is absent.
        OSError: If study/config/artifact files cannot be read or written.
        KeyError: If a requested post-training validation objective is missing.
    """

    task = normalize_training_task(task)
    try:
        import optuna
    except ImportError as error:
        raise ImportError(
            "Optuna is required for HPO. "
            "Install the project requirements."
        ) from error
    # Reject a study budget that would perform no training epochs.
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    model_name = model_name.lower()
    fit_kwargs = dict(fit_kwargs or {})
    snapshot_network_name = str(snapshot_network_name).lower()
    search_space_overrides = dict(search_space_overrides or {})
    n_startup_trials = int(n_startup_trials)
    class_order_mode = str(class_order_mode).lower()
    task_order_mode = str(task_order_mode).lower()
    # Materialize an explicit class order while preserving an omitted schedule.
    class_order = None if class_order is None else list(class_order)
    # Materialize explicit task groups while preserving automatic grouping.
    task_groups = None if task_groups is None else [
        list(group) for group in task_groups
    ]
    effective_distillation = bool(
        use_distillation or teacher_network is not None
    )
    fixed_wrapper_overrides = dict(wrapper_overrides or {})
    # Require an exact raw family for fixed architecture or wrapper overrides.
    if model_name == _DIFFUSION_CLASSIFIER_STUDY and (
        model_overrides or fixed_wrapper_overrides
    ):
        raise ValueError(
            "model_overrides and wrapper_overrides require an exact "
            "diffusion classifier family, not diffusion_classifier."
        )
    # Reject wrapper overrides for non-diffusion studies.
    if fixed_wrapper_overrides and model_name not in _DIFFUSION_HPO_MODELS:
        raise ValueError("wrapper_overrides requires a diffusion model family.")
    swap_noise_image = fixed_wrapper_overrides.get(
        "swap_noise_image",
        False,
    )
    # Reject teacher noise targets when the student reconstructs noisy inputs.
    if swap_noise_image and effective_distillation:
        raise ValueError(
            "noise distillation is incompatible with "
            "swap_noise_image=True input reconstruction."
        )
    _validate_swap_noise_hpo(
        model_name,
        model_overrides,
        fixed_wrapper_overrides,
    )

    uses_feature_archive = (
        task == "continual"
        and model_name == "vae"
        and dataset_name.lower() in ("cifar10", "cifar100")
    )
    # Reject a feature archive for studies that do not consume saved features.
    if feature_archive_path is not None and not uses_feature_archive:
        raise ValueError(
            "feature_archive_path is only used by continual CIFAR VAE studies."
        )
    # Use the standard safe feature archive for continual CIFAR VAE studies.
    if uses_feature_archive and feature_archive_path is None:
        feature_archive_path = Path("data") / (
            f"{dataset_name.lower()}_xception_gavgpooled_features_"
            "train_val_test_safe"
        )
    # Validate a configured feature archive before creating study storage.
    if feature_archive_path is not None:
        _feature_archive_signature(feature_archive_path, dataset_name)

    # Require a supported task/model search-space pairing.
    if task not in SEARCH_SPACES or model_name not in SEARCH_SPACES[task]:
        raise ValueError(f"Unsupported task/model pair: {task}/{model_name}")
    _validate_fit_request(model_name, fit_method, fit_kwargs)
    # Restrict ensemble feedback to supported diffusion-classifier studies.
    if use_ensemble_accuracy and not (
        task in ("joint", "continual")
        and model_name in _DIFFUSION_HPO_CLASSIFIER_MODELS
    ):
        raise ValueError(
            "use_ensemble_accuracy requires a joint "
            "or continual diffusion classifier study."
        )
    # Reject teacher snapshot selectors unsupported by diffusion wrappers.
    if snapshot_network_name not in ("raw", "ema", "search"):
        raise ValueError(
            "snapshot_network_name must be 'raw', 'ema', or 'search'."
        )
    # Use the raw snapshot setting when no distillation teacher is needed.
    if not effective_distillation:
        snapshot_network_name = "raw"
    # Runtime teachers can supervise any diffusion wrapper in its valid task.
    if teacher_network is not None and model_name not in _DIFFUSION_HPO_MODELS:
        raise ValueError(
            "teacher_network requires a diffusion model family."
        )
    # Require one exact raw family for compatibility with a supplied runtime teacher.
    if teacher_network is not None and model_name == _DIFFUSION_CLASSIFIER_STUDY:
        raise ValueError(
            "A runtime teacher requires an exact compatible diffusion "
            "classifier family; use teacher-free previous-task snapshots "
            "for the diffusion_classifier umbrella study."
        )
    # Teacher-free distillation relies on the continual previous-task snapshot.
    if use_distillation and teacher_network is None and not (
        task == "continual"
        and model_name in _DIFFUSION_HPO_MODELS
    ):
        raise ValueError(
            "Teacher-free use_distillation requires a continual diffusion "
            "study."
        )
    available_class_num, _, _ = get_dataset_spec(dataset_name)
    schedule_requested = class_num is not None or class_order is not None \
        or task_groups is not None or task_size != 1 \
        or class_order_mode != "fixed" or task_order_mode != "fixed"
    # Reject schedule switches that a non-continual objective would ignore.
    if task != "continual" and schedule_requested:
        raise ValueError("Continual schedule options require task='continual'.")
    resolved_task_groups = None
    # Validate every continual schedule before creating study metadata/storage.
    if task == "continual":
        _, resolved_task_groups = resolve_continual_schedule(
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

    # Append the ensemble study suffix only when ensemble feedback is enabled.
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
        feature_archive_path=feature_archive_path,
        model_overrides=model_overrides,
        wrapper_overrides=wrapper_overrides,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        n_startup_trials=n_startup_trials,
        search_space_overrides=search_space_overrides,
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
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=n_startup_trials,
    )
    pruner = optuna.pruners.NopPruner()
    create_kwargs = {
        "study_name": study_name, 
        "storage": "sqlite:///" + storage_path, 
        "sampler": sampler,
        "pruner": pruner,
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
            pruner=pruner,
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
    if existing_trials:
        sampler_state = study.user_attrs.get(_SAMPLER_RNG_STATE_ATTR)
        # A nonempty study must have persisted its exact sampler position.
        if sampler_state is None:
            # Fail instead of silently replaying TPE draws from the initial seed.
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
        """Construct, persist, reload, train, and score one Optuna trial.

        The closure uses run_hpo's fixed scientific specification and study paths.
        Sampler state is saved even if config suggestion fails. A committed task
        checkpoint selects continuation for a recovered continual trial; otherwise
        training starts fresh. Input/resolved YAML, objective CSV, and trial metadata
        record the exact Config consumed and produced by the shared training API.

        Args:
            trial (optuna.trial.Trial): Active trial receiving suggestions, recovery
                metadata, and result/config artifact paths.

        Returns:
            float | tuple[float, ...]: Validation scalar or ordered metric tuple,
            with ensemble-derived continual values selected from the saved Config.

        Raises:
            ValueError: If sampled/fixed settings violate the configured experiment.
            KeyError: If a required final validation report or metric is absent.
            TypeError: If serialization or scalar objective contracts are violated.
            OSError: If trial artifacts or checkpoints cannot be written/read.
            tf.errors.ResourceExhaustedError: If training exceeds device resources;
                the outer study records this trial as failed and continues.
        """

        tf.keras.backend.clear_session()
        gc.collect()

        # Keep the data split and initialization seed fixed across candidates;
        # Optuna's independently seeded sampler supplies the search variation.
        trial_seed = seed
        try:
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
                feature_archive_path=feature_archive_path,
                model_overrides=model_overrides,
                wrapper_overrides=wrapper_overrides,
                max_train_samples=max_train_samples,
                max_val_samples=max_val_samples,
                search_space_overrides=search_space_overrides,
            )
        finally:
            # Persist every draw even when conditional config construction
            # fails, so a resumed study cannot rewind the TPE sampler.
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
            config.model.name,
            result["history"], 
            evaluations=result["evaluations"], 
            use_ensemble_accuracy=config.hpo["use_ensemble_accuracy"],
            objective_metrics=objective_metrics,
            objective_directions=objective_directions,
            diffusion_network_name=config.model.wrapper_kwargs.get(
                "test_network_name",
                "ema",
            ),
            swap_noise_image=bool(
                config.model.wrapper_kwargs.get("swap_noise_image", False)
            ),
            kl_loss_coef=float(
                config.model.wrapper_kwargs.get("kl_loss_coef", 0.)
            ),
        )
        # Serialize a scalar objective as one list entry or preserve all tuple dimensions.
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
        """Persist the study table after a trial finishes, including failed trials.

        Args:
            study_ (optuna.study.Study): Updated study.
            trial_ (optuna.trial.FrozenTrial): Just-completed trial; unused.

        Returns:
            None: study_root/trials.csv is replaced with the current Optuna table.

        Raises:
            OSError: If the study CSV cannot be opened or written.
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
