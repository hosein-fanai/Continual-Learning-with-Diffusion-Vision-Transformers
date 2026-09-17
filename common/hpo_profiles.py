"""Explicit, bounded recipes layered on the common HPO configuration API.

The joint DiT profile trains every CIFAR class together with the V1 wrapper,
raw weights, and a classifier projection supported by semantic consolidation.
It creates no teacher, distillation token, or continual task stream.
"""

from __future__ import annotations

from copy import deepcopy

from math import isfinite

from pathlib import Path

from typing import Any, Mapping

from common.config import Config


JOINT_CLASSIFIER_PROFILE = "joint_dit_classifier"
JOINT_CLASSIFIER_PROFILE_VERSION = 10

JOINT_CLASSIFIER_SEARCH_SPACE = {
    "learning_rate": {"low": 1e-5, "high": 5e-3, "log": True}, 
    "optimizer": ["adam", "adamw"], 
    "weight_decay": {"low": 0., "high": 1e-3, "log": True}, 
    "patch_size": [2, 4], 
    "mha_num_heads": [4, 6, 8], 
    "dim": [32, 64, 128, 256], 
    "depth": [2, 3, 4, 5, 6, 7], 
    "clf_depth": [1, 2, 3, 4, 5], 
    "clf_cond_type": ["time_label", "time", "label", None], 
    "dropout_rate": [0.0, 0.15, 0.25], 
    "clf_drop_prob": [0.0, 0.15, 0.25], 
    "feature_aggregation": ["last", "all"], 
    "classifier_mlp_ratio": [1, 2, 4]
}


def _add_fixed_overrides(
    settings: dict[str, object], 
    overrides: Mapping[str, object] | None, 
    name: str
) -> None:
    """Add explicit constructor options without replacing the profile contract."""

    for key, value in (overrides or {}).items():
        # Scientific recipe settings cannot be replaced by additive options.
        if key in settings:
            raise ValueError(
                f"{name} replaces profile setting {key!r}."
            )

        settings[key] = deepcopy(value)


def build_joint_classifier_config(
    trial: Any,
    *,
    dataset_name: str,
    epochs: int,
    seed: int,
    results_path: str | Path,
    dtype_policy: str = "float32",
    deterministic_ops: bool = False,
    ensemble_accuracy_kwargs: Mapping[str, object] | None = None,
    search_space_overrides: Mapping[str, object] | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    validation_source: str = "split",
    validation_ratio: float = 0.2,
    model_overrides: Mapping[str, object] | None = None,
    wrapper_overrides: Mapping[str, object] | None = None
) -> Config:
    """Build a raw V1, two-objective ordinary joint-classifier HPO trial.

    Categorical overrides restrict the documented choices; numeric overrides
    use the common HPO low/high/step/log schema. V1, cosine decay, batch size 128
    and CNN patchification are fixed settings, not Optuna parameters.
    Classifier CE and denoising MSE each
    have coefficient 1.0; classification uses internal backbone features.

    Denoising retains conditional CFG dropout. A separate clean, unconditional
    classification pass supervises every image without exposing its target
    label as input. Both passes update the shared backbone in one joint phase
    with one optimizer.

    Every finite trial trains for the full epoch budget and uses final weights.
    Final HPO feedback uses raw classifier accuracy, with no timestep ensemble.
    Gradient clipping and plateau learning-rate adjustment are disabled.
    ``ensemble_accuracy_kwargs`` must be empty for this profile.

    Fixed controls deliberately keep unrequested dimensions out of the search:
    1000 diffusion timesteps, clipped-cosine noise schedule, p_uncond=0.1,
    epsilon MSE, no EMA or auxiliary losses, FFN ratio 4, four classifier heads,
    learned classifier-only token, and all-depth projection back to ``dim``.
    Data selection is explicit: ``validation_source='split'`` uses the selected
    training/validation ratio, while ``'test'`` trains on all official training
    rows and uses all official test rows for fit validation and HPO. The latter
    bypasses internal splitting and yields tuning rather than independent test
    scores. Both modes retain the final partial training batch. Float32, fixed
    pixel scaling and a positive classifier projection follow the local
    TMCL-inspired route's architecture/runtime contract. Offline HPO retains
    its full-range denoising objective; it is not a continual
    semantic-consolidation run.
    """

    # Import lazily: common.hpo dispatches into profiles during trial creation.
    from common.hpo import _TrialView, _tensorboard_name


    dataset_name = dataset_name.lower()
    # Match the runtime precision required by the local semantic adapter.
    if dtype_policy != "float32":
        raise ValueError("The TMCL-compatible joint profile requires dtype_policy='float32'.")
    # Ordinary accuracy is the common classifier objective for all candidates.
    if ensemble_accuracy_kwargs:
        raise ValueError(
            "joint_dit_classifier reports ordinary accuracy; "
            "ensemble_accuracy_kwargs must be empty."
        )
    # This recipe fixes image geometry and class counts for the CIFAR datasets.
    if dataset_name not in ("cifar10", "cifar100"):
        raise ValueError("joint_dit_classifier supports cifar10 and cifar100 only.")
    # Reject truncation and boolean values in the finite epoch budget.
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    # Every candidate must use the same explicitly selected validation source.
    if validation_source not in ("split", "test"):
        raise ValueError("validation_source must be 'split' or 'test'.")
    # Booleans are not meaningful fractions despite their numeric Python type.
    if isinstance(validation_ratio, bool):
        raise ValueError("validation_ratio must be a finite number in [0, 1).")
    validation_ratio = float(validation_ratio)
    # Require a finite fraction that leaves at least some training examples.
    if not isfinite(validation_ratio) or not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be a finite number in [0, 1).")
    # Internal validation must reserve examples when it supplies HPO scores.
    if validation_source == "split" and validation_ratio == 0:
        raise ValueError("HPO with validation_source='split' requires a positive validation_ratio.")
    for name, cap in (("max_train_samples", max_train_samples), 
                      ("max_val_samples", max_val_samples)):
        # Optional sample caps are exact positive row counts.
        if cap is not None and (
            isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0
        ):
            raise ValueError(f"{name} must be a positive integer or None.")

    overrides = dict(search_space_overrides or {})
    unknown = set(overrides) - set(JOINT_CLASSIFIER_SEARCH_SPACE)
    # Removed dimensions must not silently reappear in a saved study.
    if unknown:
        raise ValueError(
            f"Unknown joint classifier search overrides: {sorted(unknown)}"
        )

    # These native defaults are part of the scientific recipe even though they
    # are intentionally omitted from wrapper_kwargs below. Do not let the
    # additive override API silently change the target or guidance convention.
    for name, expected in {
        "train_cfg_scale": None, 
        "test_cfg_scale": 4.0, 
        "swap_noise_image": False,
        "modify_first_t": False,
        "test_noisified_min_timesteps": 0, 
        "test_noisified_max_timesteps": -1
    }.items():
        # Defaults omitted from the constructor mapping remain fixed controls.
        if name in (wrapper_overrides or {}) and wrapper_overrides[name] != expected:
            raise ValueError(f"wrapper_overrides replaces profile default {name!r}.")

    for name, extra in (("model_overrides", model_overrides or {}),
                        ("wrapper_overrides", wrapper_overrides or {})):
        # Constructor overrides cannot bypass the top-level precision contract.
        if "dtype" in extra and extra["dtype"] != "float32":
            raise ValueError(f"{name} requires dtype='float32' for TMCL compatibility.")
        compile_args = extra.get("compile_args") or {}
        # Compilation options must be inspectable without invoking user objects.
        if not isinstance(compile_args, Mapping):
            raise ValueError(f"{name}.compile_args must be a mapping.")
        # The profile's optimizer and denoising target define the experiment.
        if {"optimizer", "loss"}.intersection(compile_args):
            raise ValueError(f"{name}.compile_args replaces the profile optimizer or loss.")

    removed_v2_options = {
        "clf_train_noisified_max_timesteps", 
        "clf_test_noisified_max_timesteps", 
        "clf_vars_embedding_ids", 
        "clf_vars_noise_part_ids"
    }.intersection(wrapper_overrides or {})
    # Phase ownership and classifier corruption caps belong only to V2.
    if removed_v2_options:
        raise ValueError(
            "V2-only wrapper overrides are incompatible with the V1 profile: "
            f"{sorted(removed_v2_options)}"
        )

    suggestions = _TrialView(trial, overrides=overrides)


    def categorical(name: str) -> object:
        """Suggest one value from this recipe's bounded categorical dimension."""

        return suggestions.suggest_categorical(name, JOINT_CLASSIFIER_SEARCH_SPACE[name])


    optimizer_name = categorical("optimizer")
    learning_rate = suggestions.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = suggestions.suggest_float(
        "weight_decay", 1e-6, 1e-2, log=True
    ) if optimizer_name == "adamw" else None

    # Custom numeric bounds still need to yield a usable learning rate.
    if not isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    # AdamW weight decay must remain finite and positive when selected.
    if weight_decay is not None and (not isfinite(weight_decay) or weight_decay <= 0):
        raise ValueError("AdamW weight_decay must be finite and positive.")

    feature_aggregation = categorical("feature_aggregation")
    dim = categorical("dim")
    clf_cond_type = categorical("clf_cond_type")

    model_kwargs = {
        "num_classes": 10 if dataset_name == "cifar10" else 100, 
        "image_size": 32, 
        "channels": 3, 
        "timesteps": 1000, 
        "use_cfg": True, 
        "patch_size": categorical("patch_size"), 
        "mha_num_heads": categorical("mha_num_heads"), 
        "dim": dim, 
        "depth": categorical("depth"), 
        "clf_depth": categorical("clf_depth"), 
        "clf_cond_type": clf_cond_type, 
        "clf_ln_no_adaptation": clf_cond_type is None, 
        "dropout_rate": categorical("dropout_rate"), 
        "clf_drop_prob": categorical("clf_drop_prob"), 
        "aggregate_from_noises": False,
        "feature_aggregation_ids_dict": {1: [None] if feature_aggregation == "all" else [-1]}, 
        "clf_dim": dim if feature_aggregation == "all" else None, 
        "clf_dim_forced": feature_aggregation == "all", 
        "classifier_mlp_ratio": categorical("classifier_mlp_ratio"), 
        "patchify_with_cnn": True,
        "cls_token_type": None, 
        "classifier_only_cls_token": True, 
        "clf_cls_token_type": "new_weight", 
        "force_global_avg_pooling": False, 
        "distil_token_type": None, 
        "clf_distil_token_type": None, 
        "classifier_only_distil_token": True, 
        "cls_token_regularizer_ids": [], 
        "clf_cls_token_regularizer_ids": [], 
        "vit_block_mlp_ratio": 4.0, 
        "clf_vit_block_mlp_ratio": 4.0, 
        "clf_mha_num_heads": 4, 
        "drop_prob": 0.0, 
        "cond_type": "time_label", 
        "ln_no_adaptation": False, 
        "patches_pos_embed_type": "2d_sincos", 
        "use_unpatchify": True, 
        "use_refiner_cnn": False
    }
    wrapper_kwargs = {
        "use_ema": False, 
        "test_network_name": "raw", 
        "scheduler_name": "clipped_cosine", 
        "p_uncond": 0.1, 
        "noise_loss_coef": 1.0, 
        "image_loss_coef": 0.0, 
        "kl_loss_coef": 0.0, 
        "ctr_loss_coef": 0.0, 
        "noise_distil_loss_coef": 0.0, 
        "clf_distil_loss_coef": 0.0, 
        "clf_loss_coef": 1.0, 
        "clf_acc_coef": 1.0, 
        "clf_distil_acc_coef": 0.0, 
        "ctr_acc_coef": 0.0, 
        "clf_train_noisy_input_type": "clean",
        "clf_train_class_input_type": "null_class_only",
        "clf_train_type": "uncond",
        "mask_by_nulls": False,
        "mask_by_t_threshold": False, 
        "use_ensemble_loss_instead": False, 
        "test_steps": 50, 
        "test_eta": 0.0
    }
    _add_fixed_overrides(model_kwargs, model_overrides, "model_overrides")
    _add_fixed_overrides(wrapper_kwargs, wrapper_overrides, "wrapper_overrides")
    # Frozen semantic targets require deterministic feature extraction. Match
    # the route validator while still permitting nonvariational reshaping.
    for prefix in ("", "clf_"):
        reshapers = model_kwargs.get(f"{prefix}reshaper_ids_dict") or {}
        options = model_kwargs.get(f"{prefix}reshaper_kwargs")
        # Classifier reshapers inherit the backbone defaults when not specified.
        if options is None:
            options = model_kwargs.get("reshaper_kwargs") or {}
        # Stochastic latent samples cannot serve as deterministic semantic targets.
        if "flatten" in reshapers.values() and options.get("add_kl", False):
            raise ValueError("Stochastic variational flattening is incompatible with TMCL.")

    tensorboard_name = _tensorboard_name(trial)
    profile_root = (
        Path(results_path) / "joint" 
        / "dit_classifier" / dataset_name
        / JOINT_CLASSIFIER_PROFILE
    )
    config = Config(
        dataset={
            "name": dataset_name, 
            "batch_size": 128,
            "preprocess": "fixed-standardize", 
            "onehot_labels": False, 
            "validation_ratio": validation_ratio,
            "validation_source": validation_source,
            "drop_remainder": False,
            "max_train_samples": max_train_samples, 
            "max_val_samples": max_val_samples
        }, 
        model={
            "name": "dit_classifier", 
            "wrapper_name": "diffusion_classifier",
            "kwargs": model_kwargs, 
            "wrapper_kwargs": wrapper_kwargs, 
            "loss_function": "mse"
        }, 
        optimizer={
            "name": optimizer_name, 
            "initial_learning_rate": learning_rate, 
            "weight_decay": weight_decay, 
            "clipnorm": None,
            "global_clipnorm": None, 
            "schedule": "cosine",
            "plateau_jump": False
        }, 
        training={
            "task": "joint", 
            "epochs": epochs, 
            "fit_method": "fit", 
            "seed": seed, 
            "dtype_policy": dtype_policy, 
            "deterministic_ops": bool(deterministic_ops), 
            "verbose": 1, 
            "patience": 0, 
            "monitor": "val_classifier_accuracy", 
            "monitor_mode": "max", 
            "reduce_lr_patience": 0, 
            "reduce_lr_factor": 0.5, 
            "min_learning_rate": 1e-6, 
            "ensemble_monitor": False, 
            "tensorboard": True, 
            "tensorboard_path": str(profile_root / "tensorboard"), 
            "tensorboard_run_name": tensorboard_name, 
            "report_every_epoch": False, 
            "show_images": False, 
            "save_gifs": False, 
            "results_path": str(profile_root / "runs"), 
            "project_tag": f"t{trial.number:04d}"
        }, 
        reporting={
            "show_history_plot": False, 
            "save_history_plot": True, 
            "show_final_images": False, 
            "save_final_images": True, 
            "save_final_gifs": False, 
            "final_images_steps": 50, 
            "final_generation_network_name": "raw", 
            "final_generation_add_null_label": True, 
            "final_generation_modes": [
                {"name": "quick_scale3", "steps": 50, "scale": 3.0, "eta": 0.0}, 
            ], 
            "plot_without_20percent": False, 
            "run_trainset_eval": False, 
            "run_valset_eval": True, 
            "evaluate_ensemble_accuracy": False,
            "ensemble_accuracy_kwargs": {},
            "save_csv": True
        }, 
        hpo={
            "search_profile": JOINT_CLASSIFIER_PROFILE,
            "profile_version": JOINT_CLASSIFIER_PROFILE_VERSION,
            "study_task": "joint",
            "study_model": "dit_classifier",
            "model_family": "dit_classifier",
            "trial_number": trial.number,
            "params": dict(trial.params),
            "tensorboard_name": tensorboard_name,
            "use_ensemble_accuracy": False,
            "ensemble_accuracy_kwargs": {},
            "accuracy_metric": "classification_accuracy",
            "use_distillation": False,
            "objective_metrics": ["classification_accuracy", "noise_loss"],
            "objective_directions": ["maximize", "minimize"],
            "prune_nonfinite_losses": True,
            "objective_network": "raw", 
            "noise_evaluation_protocol": {
                "timestep_min_inclusive": 0, 
                "timestep_max_exclusive": 1000, 
                "cfg_scale": 4.0, 
                "corruption": "fixed_seed_per_split", 
                "reduction": "mean_over_all_examples_and_pixels"
            }, 
            "seed": seed,
            "dtype_policy": dtype_policy,
            "deterministic_ops": bool(deterministic_ops),
            "checkpoint_selection_metric": None,
            "checkpoint_selection_policy": "final_epoch",
            "epoch_budget": {
                "generator": None,
                "classifier": None,
                "joint": epochs,
                "maximum_total_epochs": epochs
            },
            "fixed_recipe": {
                "wrapper_name": "diffusion_classifier",
                "learning_rate_schedule": "cosine",
                "batch_size": 128,
                "patchify_with_cnn": True,
                "modify_first_t": False,
                "classification_labels": "all_examples_unconditional",
                "clf_train_noisy_input_type": "clean",
                "clf_train_class_input_type": "null_class_only",
                "classifier_gradients": "shared_backbone_and_head",
                "classifier_representation": "internal_features",
                "mask_by_nulls_requested": False,
                "mask_by_nulls_effective": False,
                "diffusion_prediction": "conditional_with_cfg_dropout",
                "validation_source": validation_source,
                "validation_ratio": validation_ratio if validation_source == "split" else 0.0,
                "drop_remainder": False,
                "test_set_used_for_hpo": validation_source == "test",
                "test_set_used_for_fit_validation": validation_source == "test",
                "independent_test_estimate": False,
                "classifier_loss_coefficient": 1.0,
                "classifier_heads": 4,
                "classifier_width": "project_all_features_to_dim",
                "timesteps": 1000,
                "diffusion_schedule": "clipped_cosine",
                "p_uncond": 0.1,
                "ema": False,
                "auxiliary_losses": False,
                "ensemble_each_epoch": False
            }
        }
    )
    return config
