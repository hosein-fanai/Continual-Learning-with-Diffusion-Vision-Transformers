"""Archive-informed, teacher-free DiT classifier HPO for the maintained runner.

Raw classifier accuracy is maximized and denoising loss is minimized jointly.
Final raw weights supply both objectives after the same epoch budget. Profile
version 2 supersedes the initial scalar-accuracy recipe and requires a new study.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from common.config import Config
from common.hpo_profiles import _add_fixed_overrides


PROFILE = "dit_classifier_runner"
VERSION = 2
SEARCH_SPACE = {
    "dim": [64, 128, 256], 
    "depth": [4, 6, 8], 
    "clf_depth": [2, 4, 6, 8], 
    "mha_num_heads": [4, 6, 8], 
    "patch_size": [2, 4], 
    "batch_size": [32, 64, 128], 
    "optimizer": ["adam", "adamw"], 
    "learning_rate": {"low": 1e-4, "high": 1e-3, "log": True}, 
    "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True}, 
    "clf_loss_coef": [0.001075, 0.00215, 0.0043, 0.0086], 
    "scheduler_name": ["clipped_cosine", "linear"], 
    "p_uncond": [0.1, 0.2], 
    "clf_cond_type": ["time_label", "time", None], 
    "classifier_noise": ["clean", "noisy32", "noisy128"], 
    "classifier_readout": ["cls", "gap_with_cls", "gap_without_cls"], 
    "classifier_dropout_rate": [0.0, 0.1, 0.25, 0.5], 
    "clf_droppath_rate": [0.0, 0.05, 0.1], 
    "clf_vit_block_dropout_rate": [0.0, 0.1], 
    "clf_vit_block_attention_dropout_rate": [0.0, 0.1], 
    "clf_vit_block_mlp_ratio": [2.0, 4.0], 
    "classifier_mlp_ratio": [None, 1, 2], 
    "feature_aggregation": ["last", "middle_last", "all"], 
    "classifier_route": [
        "plain", "local_first", "local_alternating", 
        "generator_cross_late", "classifier_cross_late"
    ]
}


def baseline_hints() -> list[dict[str, object]]:
    """Return a fresh archive-baseline suggestion without scores or weights.

    Returns:
        list[dict[str, object]]: A detached 128-wide eight-plus-eight-stage
            teacher-free suggestion. The new study owns its data and budget.
    """

    return [{
        "dim": 128, "depth": 8, "clf_depth": 8, "mha_num_heads": 4, 
        "patch_size": 2, "batch_size": 128, "optimizer": "adam", 
        "learning_rate": 0.001, "clf_loss_coef": 0.0043, 
        "scheduler_name": "clipped_cosine", "p_uncond": 0.1, 
        "clf_cond_type": "time_label", "classifier_noise": "clean", 
        "classifier_readout": "cls", "classifier_dropout_rate": 0.5, 
        "clf_droppath_rate": 0.0, "clf_vit_block_dropout_rate": 0.0, 
        "clf_vit_block_attention_dropout_rate": 0.0, 
        "clf_vit_block_mlp_ratio": 4.0, "classifier_mlp_ratio": None, 
        "feature_aggregation": "last", "classifier_route": "plain"
    }]


def build_dit_classifier_config(
    trial: Any, 
    dataset_name: str, 
    epochs: int, 
    results_path: str | Path, 
    seed: int, 
    dtype_policy: str = "float32", 
    deterministic_ops: bool = False, 
    ensemble_accuracy_kwargs: Mapping[str, object] | None = None, 
    search_space_overrides: Mapping[str, object] | None = None, 
    max_train_samples: int | None = None, 
    max_val_samples: int | None = None, 
    validation_source: str = "test", 
    validation_ratio: float = 0.0, 
    model_overrides: Mapping[str, object] | None = None, 
    wrapper_overrides: Mapping[str, object] | None = None
) -> Config:
    """Resolve the bounded classifier recipe through native project APIs.

    Full-horizon denoising retains coefficient one. Classification uses null
    labels and all examples in a separate forward pass. Its clean or capped
    noising never truncates denoising. Clean raw accuracy and full-horizon noise
    loss form the two HPO objectives. Finite trials train for the full common
    epoch budget without performance pruning or accuracy-based early stopping.
    Per-epoch validation remains available for TensorBoard. Final raw weights
    supply both objectives; no accuracy-selected checkpoint replaces them.
    Official-test feedback is tuning evidence, not independent test accuracy.

    Args:
        trial (Any): Optuna-compatible provider with number and params.
        dataset_name (str): CIFAR10 or CIFAR100, case-insensitive.
        epochs (int): Shared full epoch budget; cosine spans this horizon.
        results_path (str | Path): Artifact root; this builder writes no files.
        seed (int): Trial model, shuffle and evaluation seed.
        dtype_policy (str): Float32 runtime contract. Defaults to float32.
        deterministic_ops (bool): Request deterministic kernels at execution.
        ensemble_accuracy_kwargs (Mapping[str, object] | None): Must be empty;
            the accuracy objective uses ordinary raw classification.
        search_space_overrides (Mapping[str, object] | None): Categorical
            restrictions or common HPO numeric distribution replacements.
        max_train_samples (int | None): Optional development training-row cap.
        max_val_samples (int | None): Optional development feedback-row cap.
        validation_source (str): Official test feedback or internal split.
        validation_ratio (float): Internal holdout fraction in split mode;
            official-test mode resolves its effective fraction to zero.
        model_overrides (Mapping[str, object] | None): Additive raw options;
            existing fixed or sampled recipe options cannot be replaced.
        wrapper_overrides (Mapping[str, object] | None): Additive wrapper
            options preserving the teacher-free joint objective recipe.

    Returns:
        Config: Native V1 joint classifier with TensorBoard, per-epoch
            validation, full-budget training and final-raw Pareto feedback.

    Raises:
        ValueError: Unsupported data/runtime/objective or conflicting options.
    """

    from common.hpo import _TrialView, _tensorboard_name


    dataset_name = dataset_name.lower()
    # Data/runtime domains identify the experiment and its comparable objective.
    if dataset_name not in ("cifar10", "cifar100"):
        raise ValueError(f"{PROFILE} supports cifar10 and cifar100 only.")
    # Keep trials on the declared framework precision.
    if dtype_policy != "float32":
        raise ValueError(f"{PROFILE} requires dtype_policy='float32'.")
    # Timestep ensembling would replace the ordinary classifier objective.
    if ensemble_accuracy_kwargs:
        raise ValueError(f"{PROFILE} requires ordinary accuracy without ensembling.")
    # The selected feedback source must have an unambiguous dataset meaning.
    if validation_source not in ("test", "split"):
        raise ValueError("validation_source must be 'test' or 'split'.")
    # Internal feedback requires held-out examples.
    if validation_source == "split" and validation_ratio <= 0:
        raise ValueError("Split feedback requires a positive validation_ratio.")
    overrides = dict(search_space_overrides or {})
    unknown = set(overrides) - set(SEARCH_SPACE)
    # Misspelled dimensions must not silently leave the intended search.
    if unknown:
        raise ValueError(f"Unknown {PROFILE} search overrides: {sorted(unknown)}")
    suggestions = _TrialView(trial, overrides=overrides)


    def categorical(name: str) -> object:
        """Sample a declared categorical domain through the common adapter.

        Args:
            name (str): Categorical SEARCH_SPACE key.

        Returns:
            object: Selected value also recorded by the trial.
        """

        return suggestions.suggest_categorical(name, SEARCH_SPACE[name])


    optimizer_name = categorical("optimizer")
    learning_rate = suggestions.suggest_float("learning_rate", **SEARCH_SPACE["learning_rate"])
    weight_decay = suggestions.suggest_float(
        "weight_decay", **SEARCH_SPACE["weight_decay"]
    ) if optimizer_name == "adamw" else None
    dim = categorical("dim")
    depth = categorical("depth")
    clf_depth = categorical("clf_depth")
    aggregation = categorical("feature_aggregation")
    route = categorical("classifier_route")
    readout = categorical("classifier_readout")
    conditioning = categorical("clf_cond_type")
    corruption = categorical("classifier_noise")
    cap = {"clean": 0, "noisy32": 32, "noisy128": 128}[corruption]
    sources = {
        "last": [depth], "middle_last": [depth // 2, depth], "all": [None]
    }[aggregation]
    local_ids = (
        [1] if route == "local_first" else list(range(1, clf_depth + 1, 2))
        if route == "local_alternating" else []
    )
    model_kwargs = {
        "num_classes": 10 if dataset_name == "cifar10" else 100, 
        "image_size": 32, "channels": 3, "timesteps": 1000, "use_cfg": True, 
        "dim": dim, "depth": depth, "clf_depth": clf_depth, 
        "mha_num_heads": categorical("mha_num_heads"), "mha_key_dim": None, 
        "clf_mha_num_heads": 4, "clf_mha_key_dim": None, 
        "patch_size": categorical("patch_size"), "patchify_with_cnn": True, 
        "patches_pos_embed_type": "2d_sincos", "cond_type": "time_label", 
        "ln_no_adaptation": False, "clf_cond_type": conditioning, 
        "clf_ln_no_adaptation": conditioning is None, 
        "vit_block_mlp_ratio": 4.0, "droppath_rate": 0.0, 
        "vit_block_dropout_rate": 0.0, "vit_block_attention_dropout_rate": 0.0, 
        "clf_vit_block_mlp_ratio": categorical("clf_vit_block_mlp_ratio"), 
        "clf_droppath_rate": categorical("clf_droppath_rate"), 
        "clf_vit_block_dropout_rate": categorical("clf_vit_block_dropout_rate"), 
        "clf_vit_block_attention_dropout_rate": categorical("clf_vit_block_attention_dropout_rate"), 
        "classifier_mlp_ratio": categorical("classifier_mlp_ratio"), 
        "classifier_mlp_activation_func": "tanh", 
        "classifier_dropout_rate": categorical("classifier_dropout_rate"), 
        "aggregate_from_noises": False, "feature_aggregation_ids_dict": {1: sources}, 
        "feature_aggregation_kwargs": {"connect_type": "concat"}, 
        "clf_dim": dim, "clf_dim_forced": True, 
        "cls_token_type": None, "classifier_only_cls_token": True, 
        "clf_cls_token_type": None if readout == "gap_without_cls" else "new_weight", 
        "force_global_avg_pooling": readout != "cls", 
        "distil_token_type": None, "clf_distil_token_type": None, 
        "classifier_only_distil_token": True, "cls_token_regularizer_ids": [], 
        "clf_cls_token_regularizer_ids": [], "local_mixer_ids": [], 
        "clf_local_mixer_ids": local_ids, "clf_local_mixer_kwargs": {}, 
        "cross_attention_aggregation_ids_dict": (
            {clf_depth: [depth // 2]} if route == "generator_cross_late" else {}
        ), 
        "clf_cross_attention_ids_dict": (
            {clf_depth: [clf_depth // 2]} if route == "classifier_cross_late" else {}
        ), 
        "clf_cross_attention_plug_type": "values", "clf_use_decoder_ids": [], 
        "use_unpatchify": True, "use_refiner_cnn": False
    }
    wrapper_kwargs = {
        "preprocess_type": "standardize", "use_ema": False, "test_network_name": "raw", 
        "scheduler_name": categorical("scheduler_name"), "p_uncond": categorical("p_uncond"), 
        "modify_first_t": True, "noise_loss_coef": 1.0, "image_loss_coef": 0.0, 
        "train_noisified_min_timesteps": 0, "train_noisified_max_timesteps": -1, 
        "test_noisified_min_timesteps": 0, "test_noisified_max_timesteps": -1, 
        "swap_noise_image": False, 
        "clf_loss_coef": categorical("clf_loss_coef"), "kl_loss_coef": 0.0, 
        "ctr_loss_coef": 0.0, "noise_distil_loss_coef": 0.0, "clf_distil_loss_coef": 0.0, 
        "clf_acc_coef": 1.0, "clf_distil_acc_coef": 0.0, "ctr_acc_coef": 0.0, 
        "clf_train_batch_fraction": 0.0, "clf_train_noisy_input_type": "noisy", 
        "clf_train_noisified_max_timesteps": cap, "clf_test_noisified_max_timesteps": 0, 
        "clf_train_class_input_type": "null_class_only", "clf_train_type": "cond", 
        "mask_by_nulls": False, "mask_by_t_threshold": False, 
        "use_ensemble_loss_instead": False, "train_cfg_scale": None, 
        "test_cfg_scale": 4.0, "test_steps": 50, "test_eta": 0.0
    }
    # Teacher loading or custom compilation changes the fixed experiment.
    if any("teacher" in key for key in (wrapper_overrides or {})):
        raise ValueError("Wrapper overrides cannot introduce teacher settings.")
    for extra in (model_overrides or {}, wrapper_overrides or {}):
        # Custom compilation or precision would replace the declared objective/runtime.
        if "compile_args" in extra:
            raise ValueError("Overrides cannot replace profile compilation.")
        # Raw or wrapper dtype options must preserve the declared precision.
        if "dtype" in extra and extra["dtype"] != "float32":
            raise ValueError("Overrides must preserve dtype='float32'.")
    _add_fixed_overrides(model_kwargs, model_overrides, "model_overrides")
    _add_fixed_overrides(wrapper_kwargs, wrapper_overrides, "wrapper_overrides")
    batch_size = categorical("batch_size")
    tensorboard_name = _tensorboard_name(trial)
    profile_root = Path(results_path) / "joint" / "dit_classifier" / dataset_name / PROFILE
    ratio = validation_ratio if validation_source == "split" else 0.0
    return Config(
        dataset={
            "name": dataset_name, "batch_size": batch_size, "preprocess": None, 
            "onehot_labels": False, "validation_ratio": ratio, 
            "validation_source": validation_source, "drop_remainder": False, 
            "max_train_samples": max_train_samples, "max_val_samples": max_val_samples
        }, 
        model={
            "name": "dit_classifier", "wrapper_name": "diffusion_classifier", 
            "kwargs": model_kwargs, "wrapper_kwargs": wrapper_kwargs, "loss_function": "mse"
        }, 
        optimizer={
            "name": optimizer_name, "initial_learning_rate": learning_rate, 
            "weight_decay": weight_decay, "clipnorm": None, "global_clipnorm": None, 
            "schedule": "cosine", "plateau_jump": False
        }, 
        reporting={
            "show_history_plot": False, "save_history_plot": True, 
            "show_final_images": False, "save_final_images": True, "save_final_gifs": False, 
            "final_images_steps": 50, "final_generation_network_name": "raw", 
            "final_generation_add_null_label": True, 
            "final_generation_modes": [{"name": "quick_scale3", "steps": 50, "scale": 3.0, "eta": 0.0}], 
            "plot_without_20percent": False, "run_trainset_eval": False, 
            "run_valset_eval": True, "evaluate_ensemble_accuracy": False, 
            "ensemble_accuracy_kwargs": {}, "save_csv": True
        }, 
        hpo={
            "search_profile": PROFILE, "profile_version": VERSION, "study_task": "joint", 
            "study_model": "dit_classifier", "model_family": "dit_classifier", 
            "trial_number": trial.number, "params": deepcopy(dict(trial.params)), 
            "tensorboard_name": tensorboard_name, "use_ensemble_accuracy": False, 
            "ensemble_accuracy_kwargs": {}, "accuracy_metric": "classification_accuracy", 
            "use_distillation": False, "objective_metrics": ["classification_accuracy", "noise_loss"], 
            "objective_directions": ["maximize", "minimize"], 
            "prune_nonfinite_losses": True, "objective_network": "raw", 
            "seed": seed, "dtype_policy": dtype_policy, "deterministic_ops": bool(deterministic_ops), 
            "checkpoint_selection_metric": None, 
            "checkpoint_selection_policy": "final_epoch", 
            "epoch_budget": {"joint": epochs, "maximum_total_epochs": epochs}, 
            "classifier_training": {
                "recipe": corruption, "exclusive_noising_cap": cap, 
                "evaluation_noising_cap": 0, "class_input": "null_class_only", 
                "classifier_rows": "all_examples", "diffusion_rows": "all_examples", 
                "student_forward_passes": 2, "generator_timestep_range": [0, 1000]
            }, 
            "fixed_recipe": {
                "wrapper_name": "diffusion_classifier", "learning_rate_schedule": "cosine", 
                "patchify_with_cnn": True, "modify_first_t": True, "noise_loss_coefficient": 1.0, 
                "classifier_gradients": "shared_backbone_and_head", 
                "classifier_representation": "internal_features", "classifier_heads": 4, 
                "classifier_width": "project_features_to_dim", "timesteps": 1000, 
                "ema": False, "auxiliary_losses": False, "teachers": False, 
                "validation_source": validation_source, "validation_ratio": ratio, 
                "fit_validation": True, "test_set_used_for_hpo": validation_source == "test", 
                "test_set_used_for_fit_validation": validation_source == "test", 
                "independent_test_estimate": False, "ensemble_each_epoch": False
            }
        }, 
        training={
            "task": "joint", "epochs": epochs, "fit_method": "fit", 
            "fit_kwargs": {"validation_freq": 1}, "use_valset": True, 
            "seed": seed, "dtype_policy": dtype_policy, "deterministic_ops": bool(deterministic_ops), 
            "verbose": 1, "patience": 0, "monitor": "val_loss", 
            "monitor_mode": "min", "reduce_lr_patience": 0, "reduce_lr_factor": 0.5, 
            "min_learning_rate": 1e-6, "ensemble_monitor": False, "tensorboard": True, 
            "tensorboard_path": str(profile_root / "tensorboard"), "tensorboard_run_name": tensorboard_name, 
            "report_every_epoch": False, "show_images": False, "save_gifs": False, 
            "results_path": str(profile_root / "runs"), "project_tag": f"t{trial.number:04d}"
        }
    )
