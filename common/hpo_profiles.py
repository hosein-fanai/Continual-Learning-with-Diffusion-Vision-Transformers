"""Explicit, bounded recipes layered on the common HPO configuration API.

The joint DiT profile trains every CIFAR class together. V1 optimizes diffusion
and classification together; V2 fits a generator followed by its classifier.
Neither recipe creates a teacher, distillation token, or continual task stream.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from common.config import Config


JOINT_CLASSIFIER_PROFILE = "joint_dit_classifier"
JOINT_CLASSIFIER_PROFILE_VERSION = 5

JOINT_CLASSIFIER_SEARCH_SPACE = {
    "learning_rate_schedule": ["constant", "cosine"], 
    "learning_rate": {"low": 1e-5, "high": 1e-3, "log": True}, 
    "wrapper_name": ["diffusion_classifier", "diffusion_classifier_v2"], 
    "optimizer": ["adam", "adamw"], 
    "weight_decay": {"low": 1e-6, "high": 1e-2, "log": True}, 
    "patch_size": [2, 4], 
    "mha_num_heads": [4, 6, 8], 
    "dim": [32, 64, 128, 256], 
    "depth": [3, 4, 5, 6, 7], 
    "clf_depth": [1, 2, 3, 4, 5], 
    "clf_cond_type": ["time_label", "time", "label", None], 
    "dropout_rate": [0.0, 0.25], 
    "clf_drop_prob": [0.0, 0.25], 
    "aggregate_from_noises": [False, True], 
    "feature_aggregation": ["last", "all"], 
    "classifier_mlp_ratio": [None, 1, 2, 4], 
    "patchify_with_cnn": [False, True], 
    "clf_train_noisified_max_timesteps": [None, 32, 128, 256, 512], 
    "clipnorm": [0.0, 1.0], 
    "clf_loss_coef": [0.001, 0.01, 0.1, 0.25, 0.5, 1.0],
    "modify_first_t": [True, False],
    # This is a fixed budget by default, recorded in trial parameters.
    "batch_size": [128]
}


def _add_fixed_overrides(
    settings: dict[str, object], 
    overrides: Mapping[str, object] | None, 
    name: str
) -> None:
    """Add explicit constructor options without replacing the profile contract."""

    for key, value in (overrides or {}).items():
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
    """Build an EMA, two-objective ordinary joint-classifier HPO trial.

    Categorical overrides restrict the documented choices; numeric overrides
    use the common HPO low/high/step/log schema. Batch size defaults to a fixed
    128 and can be explicitly changed. V1 searches classifier loss balance;
    V2 fixes its coefficient to 1 and does not sample that dimension.

    Native classifier training and CFG defaults are retained. V1's null mask
    restricts classification loss to randomly label-dropped examples.
    V2 uses its native disjoint variable groups, generator first and classifier
    second, with up to ``epochs`` in each phase and independent optimizers.

    Early stopping selects weights using ordinary validation accuracy (generator
    validation loss during V2's first phase). Final HPO feedback uses a timestep
    ensemble for V1 and for V2 with a positive noising cap. Clean V2 uses ordinary
    accuracy. V1 retains the wrapper's 128-timestep ensemble default; V2's
    horizon always equals its classifier cap. Chunking changes memory use, not
    the number of predictions. An explicit ``max_t`` override affects V1 only.

    Fixed controls deliberately keep unrequested dimensions out of the search:
    1000 diffusion timesteps, clipped-cosine noise schedule, p_uncond=0.1,
    epsilon MSE, EMA, no auxiliary losses, FFN ratio 4, four classifier heads,
    learned classifier-only token, and all-depth projection back to ``dim``.
    Data selection is explicit: ``validation_source='split'`` uses the selected
    training/validation ratio, while ``'test'`` trains on all official training
    rows and uses all official test rows for fit validation and HPO. The latter
    bypasses internal splitting and yields tuning rather than independent test
    scores. Both modes retain the final partial training batch.
    """

    # Import lazily: common.hpo dispatches into profiles during trial creation.
    from common.hpo import _TrialView, _tensorboard_name


    dataset_name = dataset_name.lower()
    if dataset_name not in ("cifar10", "cifar100"):
        raise ValueError("joint_dit_classifier supports cifar10 and cifar100 only.")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if validation_source not in ("split", "test"):
        raise ValueError("validation_source must be 'split' or 'test'.")
    if isinstance(validation_ratio, bool):
        raise ValueError("validation_ratio must be a finite number in [0, 1).")
    validation_ratio = float(validation_ratio)
    if not isfinite(validation_ratio) or not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be a finite number in [0, 1).")
    if validation_source == "split" and validation_ratio == 0:
        raise ValueError("HPO with validation_source='split' requires a positive validation_ratio.")
    for name, cap in (("max_train_samples", max_train_samples), 
                      ("max_val_samples", max_val_samples)):
        if cap is not None and (
            isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0
        ):
            raise ValueError(f"{name} must be a positive integer or None.")

    overrides = dict(search_space_overrides or {})
    unknown = set(overrides) - set(JOINT_CLASSIFIER_SEARCH_SPACE)
    if unknown:
        raise ValueError(
            f"Unknown joint classifier search overrides: {sorted(unknown)}"
        )

    # These native defaults are part of the scientific recipe even though they
    # are intentionally omitted from wrapper_kwargs below. Do not let the
    # additive override API silently change the target or guidance convention.
    for name, expected in {
        "clf_train_type": "cond", "train_cfg_scale": None,
        "test_cfg_scale": 4.0, "swap_noise_image": False,
    }.items():
        if name in (wrapper_overrides or {}) and wrapper_overrides[name] != expected:
            raise ValueError(f"wrapper_overrides replaces profile default {name!r}.")

    suggestions = _TrialView(trial, overrides=overrides)


    def categorical(name: str) -> object:
        return suggestions.suggest_categorical(name, JOINT_CLASSIFIER_SEARCH_SPACE[name])


    wrapper_name = categorical("wrapper_name")
    is_v2 = wrapper_name == "diffusion_classifier_v2"
    optimizer_name = categorical("optimizer")
    learning_rate = suggestions.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = suggestions.suggest_float(
        "weight_decay", 1e-6, 1e-2, log=True
    ) if optimizer_name == "adamw" else None

    if not isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if weight_decay is not None and (not isfinite(weight_decay) or weight_decay <= 0):
        raise ValueError("AdamW weight_decay must be finite and positive.")

    schedule = categorical("learning_rate_schedule")
    clipnorm = categorical("clipnorm")
    batch_size = categorical("batch_size")
    aggregate_from_noises = categorical("aggregate_from_noises")
    # Predicted-noise input bypasses the main feature routes. Do not sample an
    # inactive dimension or force a different classifier width through it.
    feature_aggregation = None if aggregate_from_noises else categorical("feature_aggregation")
    dim = categorical("dim")
    clf_cond_type = categorical("clf_cond_type")
    clf_loss_coef = 1.0 if is_v2 else categorical("clf_loss_coef")

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
        "aggregate_from_noises": aggregate_from_noises, 
        "feature_aggregation_ids_dict": {1: [None] if feature_aggregation == "all" else [-1]}, 
        "clf_dim": dim if feature_aggregation == "all" else None, 
        "clf_dim_forced": feature_aggregation == "all", 
        "classifier_mlp_ratio": categorical("classifier_mlp_ratio"), 
        "patchify_with_cnn": categorical("patchify_with_cnn"), 
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
        "use_ema": True, 
        "test_network_name": "ema", 
        "scheduler_name": "clipped_cosine", 
        "modify_first_t": categorical("modify_first_t"), 
        "p_uncond": 0.1, 
        "noise_loss_coef": 1.0, 
        "image_loss_coef": 0.0, 
        "kl_loss_coef": 0.0, 
        "ctr_loss_coef": 0.0, 
        "noise_distil_loss_coef": 0.0, 
        "clf_distil_loss_coef": 0.0, 
        "clf_loss_coef": clf_loss_coef, 
        "clf_acc_coef": 1.0, 
        "clf_distil_acc_coef": 0.0, 
        "ctr_acc_coef": 0.0, 
        "mask_by_nulls": True, 
        "mask_by_t_threshold": False, 
        "use_ensemble_loss_instead": False, 
        "test_steps": 50, 
        "test_eta": 0.0
    }
    cap = categorical("clf_train_noisified_max_timesteps") if is_v2 else None
    if is_v2:
        wrapper_kwargs.update({
            "clf_train_noisified_max_timesteps": cap, 
            "clf_test_noisified_max_timesteps": cap, 
            "clf_vars_embedding_ids": [], 
            "clf_vars_noise_part_ids": []
        })
    _add_fixed_overrides(model_kwargs, model_overrides, "model_overrides")
    _add_fixed_overrides(wrapper_kwargs, wrapper_overrides, "wrapper_overrides")

    use_ensemble = not is_v2 or (cap is not None and cap > 0)
    ensemble_options = {
        "compute_type": "chunked", 
        "max_t": 128, 
        "t_chunk_size": 8, 
        "weighted": False, 
        "separate_probas": False, 
        "network_name": "ema", 
        "seed": seed, 
        **dict(ensemble_accuracy_kwargs or {})
    }
    if is_v2 and use_ensemble:
        ensemble_options["max_t"] = cap
    if ensemble_options["network_name"] != "ema":
        raise ValueError("This profile evaluates EMA weights; network_name must be 'ema'.")
    for name in ("max_t", "t_chunk_size"):
        value = ensemble_options[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Ensemble {name} must be a positive integer.")
    if ensemble_options["max_t"] > 1000:
        raise ValueError("Ensemble max_t cannot exceed 1000 diffusion timesteps.")

    tensorboard_name = _tensorboard_name(trial)
    profile_root = (
        Path(results_path) / "joint" / "dit_classifier" / dataset_name
        / JOINT_CLASSIFIER_PROFILE
    )
    accuracy_metric = "ensemble_accuracy" if use_ensemble else "classification_accuracy"
    config = Config(
        dataset={
            "name": dataset_name, 
            "batch_size": batch_size, 
            "preprocess": "standardize", 
            "onehot_labels": False, 
            "validation_ratio": validation_ratio,
            "validation_source": validation_source,
            "drop_remainder": False,
            "max_train_samples": max_train_samples, 
            "max_val_samples": max_val_samples
        }, 
        model={
            "name": "dit_classifier", 
            "wrapper_name": wrapper_name, 
            "kwargs": model_kwargs, 
            "wrapper_kwargs": wrapper_kwargs, 
            "loss_function": "mse"
        }, 
        optimizer={
            "name": optimizer_name, 
            "initial_learning_rate": learning_rate, 
            "weight_decay": weight_decay, 
            "clipnorm": None if clipnorm == 0 else clipnorm, 
            "global_clipnorm": None, 
            "schedule": schedule, 
            "plateau_jump": True
        }, 
        training={
            "task": "joint", 
            "epochs": epochs, 
            "fit_method": "fit", 
            "seed": seed, 
            "dtype_policy": dtype_policy, 
            "deterministic_ops": bool(deterministic_ops), 
            "verbose": 1, 
            "patience": 10, 
            "monitor": "val_classifier_accuracy", 
            "monitor_mode": "max", 
            "reduce_lr_patience": 5, 
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
            "save_final_gifs": True,
            "final_generation_network_name": "ema",
            "final_generation_add_null_label": True,
            "final_generation_modes": [
                {"name": "full_stochastic_scale3", "steps": 1000, "scale": 3.0, "eta": 1.0},
                {"name": "full_default_eta_scale3", "steps": 1000, "scale": 3.0, "eta": None},
                {"name": "default_scale3", "steps": None, "scale": 3.0, "eta": None},
                {"name": "default_scale4", "steps": None, "scale": 4.0, "eta": None},
            ],
            "plot_without_20percent": False, 
            "run_trainset_eval": False, 
            "run_valset_eval": True, 
            "evaluate_ensemble_accuracy": use_ensemble, 
            "ensemble_accuracy_kwargs": ensemble_options, 
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
            "use_ensemble_accuracy": use_ensemble,
            "ensemble_accuracy_kwargs": ensemble_options,
            "accuracy_metric": accuracy_metric,
            "use_distillation": False,
            "objective_metrics": ["classification_accuracy", "noise_loss"],
            "objective_directions": ["maximize", "minimize"],
            "prune_nonfinite_losses": True,
            "objective_network": "ema",
            "noise_evaluation_protocol": {
                "timestep_min_inclusive": 0,
                "timestep_max_exclusive": 1000,
                "cfg_scale": 4.0,
                "corruption": "fixed_seed_per_split",
                "reduction": "mean_over_all_examples_and_pixels",
            },
            "seed": seed,
            "dtype_policy": dtype_policy,
            "deterministic_ops": bool(deterministic_ops),
            "checkpoint_selection_metric": "val_classifier_accuracy",
            "epoch_budget": {
                "generator": epochs if is_v2 else None,
                "classifier": epochs if is_v2 else None,
                "joint": None if is_v2 else epochs,
                "maximum_total_epochs": epochs * (2 if is_v2 else 1)
            },
            "fixed_recipe": {
                "classification_labels": (
                    "unconditional_for_all_examples" if is_v2 else "native_cfg_dropout_null_rows"
                ),
                "mask_by_nulls_requested": True,
                "mask_by_nulls_effective": not is_v2,
                "diffusion_prediction": "conditional_with_cfg_dropout",
                "v2_variable_recipe": "separate" if is_v2 else None,
                "validation_source": validation_source,
                "validation_ratio": validation_ratio if validation_source == "split" else 0.0,
                "drop_remainder": False,
                "test_set_used_for_hpo": validation_source == "test",
                "test_set_used_for_fit_validation": validation_source == "test",
                "independent_test_estimate": False,
                "classifier_loss_coefficient": clf_loss_coef,
                "classifier_heads": 4,
                "classifier_width": "project_all_features_to_dim",
                "timesteps": 1000,
                "diffusion_schedule": "clipped_cosine",
                "p_uncond": 0.1,
                "ema": True,
                "auxiliary_losses": False,
                "ensemble_each_epoch": False
            }
        }
    )
    return config
