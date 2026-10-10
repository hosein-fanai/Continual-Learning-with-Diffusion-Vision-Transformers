"""Fixed-architecture V1 DiT continual search through the existing learner.

The notebook coordinator can inspect and seal this recipe without importing
TensorFlow. Models and specialist teachers are constructed only in workers.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from common.config import Config, load_config, resolve_continual_schedule


PROFILE = "dit_continual_runner"
VERSION = 1
SEARCH_SPACE = {
    "classifier_teacher_source": ["none", "previous", "current", "both"], 
    "noise_teacher_source": ["none", "previous", "current", "both"], 
    "continual_strategy": ["generative_replay", "new_only", "cumulative"], 
    "replay_budget_mode": ["legacy", "fixed_total"], 
    "replay_samples": [100, 500, 1000, 2500, 5000], 
    "train_num": [-1, 1000, 2500, 5000, 7500, 10000], 
    "replay_old_examples": [100, 500, 1000, 2500, 5000], 
    "replay_current_examples": [100, 500, 1000, 2500, 5000], 
    "replay_selection": ["all", "uniform", "confidence", "surprise", "confidence_surprise"], 
    "replay_candidate_multiplier": [1, 2, 4], 
    "replay_surprise_weight": {"low": 0.0, "high": 1.0}, 
    "test_steps": [20, 50, 100], 
    "test_cfg_scale": {"low": 2.5, "high": 5.0}, 
    "test_eta": [0.0, 1.0], 
    "batch_size": [32, 64, 128], 
    "optimizer": ["adam", "adamw"], 
    "learning_rate": {"low": 3e-4, "high": 5e-3, "log": True}, 
    "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True}, 
    "clipnorm": [None, 0.5, 1.0, 5.0], 
    "global_clipnorm": [None, 0.5, 1.0, 5.0], 
    "classifier_noise": ["clean", "noisy32", "noisy128", "full"], 
    "p_uncond": [0.05, 0.1, 0.2]
}


def _digest(path: Path) -> str:
    """Hash an input artifact without loading a model into the coordinator."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_continual_profile(
    continual_profile: Mapping[str, object], 
    dataset_name: str = "cifar10", 
    seed: int = 42
) -> dict:
    """Seal native model inputs and one random five-by-two CIFAR10 schedule.

    ``student_config`` is a native Config mapping or YAML path. Specialist
    descriptors are normalized by the existing-model artifact bridge. The
    architecture, losses, teacher coefficients, and diffusion process remain
    fixed inputs. Repeated normalization verifies rather than replaces hashes.
    ``task_seed`` owns the task partition and validation cohort independently
    of model seeds used in search or confirmation.
    """

    from common.specialist_teacher_artifacts import normalize_specialist_teacher_descriptors


    # This protocol fixes both the dataset and the number of classes per task.
    if dataset_name.lower() != "cifar10":
        raise ValueError("dit_continual_runner requires CIFAR10 and five two-class tasks.")
    # Architecture input must be explicit before any artifact or study is created.
    if not isinstance(continual_profile, Mapping) or not continual_profile.get("student_config"):
        raise ValueError("Supply continual_profile.student_config as a native DiT CLF Config or YAML path.")
    allowed = {
        "student_config", "specialist_teacher_descriptors", "task_seed", 
        "task_groups", "class_order", "artifact_sha256", "profile_version"
    }
    # Reject unused controls instead of silently omitting them from study identity.
    if set(continual_profile) - allowed:
        raise ValueError(f"Unknown continual profile options: {sorted(set(continual_profile) - allowed)}")
    profile = deepcopy(dict(continual_profile))
    # A changed profile version cannot reuse a sealed scientific recipe.
    if profile.get("profile_version", VERSION) != VERSION:
        raise ValueError("Continual profile version changed; create a new study.")
    artifacts = dict(profile.get("artifact_sha256", {}))
    supplied = profile["student_config"]
    source_directory = Path.cwd()
    # YAML inputs bind both their original bytes and any relative checkpoint path.
    if isinstance(supplied, (str, Path)):
        path = Path(supplied).expanduser().resolve(strict=True)
        source_directory = path.parent
        digest = _digest(path)
        # Verify a prior seal before recording the YAML's current digest.
        if str(path) in artifacts and artifacts[str(path)] != digest:
            raise ValueError(f"Continual input artifact changed: {path}")
        artifacts[str(path)] = digest
        config = load_config(path)
    # Detached mappings retain the caller's native Config fields without file loading.
    elif isinstance(supplied, Mapping):
        config = Config(**deepcopy(dict(supplied)))
    # Runtime models and other objects cannot cross the serialized worker boundary.
    else:
        raise TypeError("student_config must be a native Config mapping or YAML path.")
    for path, expected in artifacts.items():
        # Re-normalization authenticates every artifact retained by the original seal.
        if _digest(Path(path)) != expected:
            raise ValueError(f"Continual input artifact changed: {path}")
    # This fixed recipe implements the attached DiT classifier with the V1 wrapper.
    if config.model.name not in (None, "dit_classifier") or (
        config.model.name is None and not config.model.with_classifier
    ) or config.model.wrapper_name not in (None, "diffusion_classifier"):
        raise ValueError("Supply a DiT classifier architecture using the V1 diffusion_classifier wrapper.")
    raw = deepcopy(config.model.kwargs or config.model.dit_classifier.kwargs())
    wrapper = deepcopy(config.model.wrapper_kwargs or config.model.diffusion_classifier.kwargs())
    # Workers construct specialist teachers from descriptors after GPU admission.
    if any(key.endswith("teacher_network") for key in wrapper):
        raise ValueError("Supply specialist artifact descriptors, never live teachers in model configuration.")
    # The selected teacher objectives require epsilon targets from the student.
    if wrapper.get("swap_noise_image", False):
        raise ValueError("This noise-distillation study requires epsilon denoising, not input reconstruction.")
    # Null-label classification and guided replay need the CFG conditioning branch.
    if not raw.get("use_cfg", True):
        raise ValueError("The continual classifier recipe requires CFG for null-label classification and replay.")
    # Fixed network inputs must match the CIFAR-10 image tensor shape.
    if raw.get("image_size", 32) != 32 or raw.get("channels", 3) != 3:
        raise ValueError("The supplied architecture must accept 32x32 RGB CIFAR10 images.")
    # Classification uses the supplied DiT head rather than an additional student model.
    if config.model.classifier_name is not None:
        raise ValueError("The attached DiT head is the classifier; do not supply a separate student classifier.")
    # Bind optional initial weights to an absolute path and immutable file digest.
    if config.model.weights_path is not None:
        weights = Path(config.model.weights_path).expanduser()
        weights = (source_directory / weights).resolve() if not weights.is_absolute() else weights.resolve()
        digest = _digest(weights)
        # Reusing a checkpoint path cannot replace the already sealed starting weights.
        if str(weights) in artifacts and artifacts[str(weights)] != digest:
            raise ValueError("The supplied student checkpoint changed.")
        artifacts[str(weights)] = digest
        config.model.weights_path = str(weights)
    config.model.name = "dit_classifier"
    config.model.wrapper_name = "diffusion_classifier"
    config.model.kwargs = raw
    config.model.wrapper_kwargs = wrapper
    descriptors = normalize_specialist_teacher_descriptors(
        profile.get("specialist_teacher_descriptors", {})
    )
    task_seed = profile.get("task_seed", seed)
    class_order, groups = resolve_continual_schedule(
        10, available_class_num=10, task_size=2, class_order_mode="random", seed=task_seed
    )
    # Every candidate and confirmation must retain the task-seed-derived partition.
    if profile.get("task_groups", groups) != groups or profile.get("class_order", class_order) != class_order:
        raise ValueError("The sealed five-task schedule differs from task_seed.")
    normalized = {
        "student_config": asdict(config), 
        "specialist_teacher_descriptors": descriptors, 
        "task_seed": task_seed, "task_groups": groups, "class_order": class_order, 
        "artifact_sha256": artifacts, "profile_version": VERSION
    }
    # A worker request must survive JSON/YAML without hidden live objects.
    json.dumps(normalized, allow_nan=False)
    return normalized


def _choice_options(overrides: Mapping[str, object], name: str) -> list:
    """Resolve categorical restrictions for coordinator hints and conditional branches."""

    value = overrides.get(name, SEARCH_SPACE[name])
    # Mapping overrides use the same explicit choices envelope as the HPO adapter.
    if isinstance(value, Mapping):
        # Numeric or unknown categorical fields cannot define a valid source grid.
        if set(value) != {"choices"}:
            raise ValueError(f"Categorical override {name} accepts only choices.")
        value = value["choices"]
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    # Restrict warm-start and conditional options to supported nonempty choices.
    if not values or any(item not in SEARCH_SPACE[name] for item in values):
        raise ValueError(f"Unsupported {name} choices: {values}")
    return values


def baseline_hints(search_space_overrides: Mapping[str, object] | None = None) -> list[dict]:
    """Queue each requested classifier/noise source pair once as a fresh trial."""

    overrides = dict(search_space_overrides or {})
    classifier = _choice_options(overrides, "classifier_teacher_source")
    noise = _choice_options(overrides, "noise_teacher_source")
    return [
        {"classifier_teacher_source": clf_source, "noise_teacher_source": noise_source}
        for clf_source in classifier for noise_source in noise
    ]


def build_dit_continual_config(
    trial: Any, 
    dataset_name: str, 
    epochs: int, 
    results_path: str | Path, 
    continual_profile: Mapping[str, object], 
    search_space_overrides: Mapping[str, object] | None = None, 
    objective_metrics: object = None, 
    objective_directions: object = None, 
    use_ensemble_accuracy: bool = False, 
    ensemble_accuracy_kwargs: Mapping[str, object] | None = None, 
    max_train_samples: int | None = None, 
    max_val_samples: int | None = None, 
    dtype_policy: str = "float32", 
    deterministic_ops: bool = False, 
    seed: int = 42
) -> Config:
    """Build one existing-API V1 continual trial without searching architecture.

    Classifier/noise teacher sources form a Cartesian sixteen-treatment search.
    Selected terms retain the input's distillation type, temperature, loss and
    accuracy coefficients and per-role weights. Disabled terms are gated to
    zero, never reweighted or optimized. Current specialists are trained on
    each task's real current examples by the existing continual learner.
    """

    from common.hpo import _TrialView, _normalize_objective_spec, _tensorboard_name


    profile = normalize_continual_profile(continual_profile, dataset_name, seed=seed)
    # Keep the numerical execution policy common across all candidate treatments.
    if dtype_policy != "float32":
        raise ValueError("The fixed continual profile requires float32.")
    overrides = dict(search_space_overrides or {})
    # Direct builder calls reject unavailable inputs before consuming any trial draws.
    validate_continual_search(profile, overrides)
    # Unknown dimensions cannot alter or enlarge the fixed architecture recipe.
    if set(overrides) - set(SEARCH_SPACE):
        raise ValueError(f"Unknown continual search dimensions: {sorted(set(overrides) - set(SEARCH_SPACE))}")
    suggestions = _TrialView(trial, overrides=overrides)


    def categorical(name: str) -> object:
        """Sample one named categorical choice through the common HPO adapter."""

        return suggestions.suggest_categorical(name, SEARCH_SPACE[name])


    def numeric(name: str) -> float:
        """Sample one named numeric distribution through the common adapter."""

        return suggestions.suggest_float(name, **SEARCH_SPACE[name])


    config = Config(**deepcopy(profile["student_config"]))
    raw = config.model.kwargs
    wrapper = config.model.wrapper_kwargs
    declared = deepcopy(wrapper)
    clf_source = categorical("classifier_teacher_source")
    noise_source = categorical("noise_teacher_source")
    previous = clf_source in ("previous", "both") or noise_source in ("previous", "both")
    current = clf_source in ("current", "both") or noise_source in ("current", "both")
    descriptors = {}
    for head, source, loss_key in (
        ("classifier", clf_source, "clf_distil_loss_coef"), 
        ("noise", noise_source, "noise_distil_loss_coef")
    ):
        active = source != "none"
        # An enabled teacher treatment must retain a nonzero declared loss term.
        if active and not declared.get(loss_key, 0):
            raise ValueError(f"The input architecture must declare an active {loss_key} for {head} distillation.")
        wrapper[loss_key] = declared.get(loss_key, 0.0) if active else 0.0
        role_head = "clf" if head == "classifier" else "noise"
        for role in ("previous", "current"):
            key = f"{role}_teacher_{role_head}_loss_weight"
            selected = source in (role, "both")
            # A sampled source must contribute through its fixed role-specific weight.
            if selected and not declared.get(key, 1.0):
                raise ValueError(f"Selected teacher route has zero input weight: {key}")
            wrapper[key] = declared.get(key, 1.0) if selected else 0.0
        # Only active current-task routes load independent specialist artifacts.
        if source in ("current", "both"):
            # A requested specialist cannot silently fall back to the student snapshot.
            if head not in profile["specialist_teacher_descriptors"]:
                raise ValueError(f"Supply the {head} specialist descriptor before searching current-task distillation.")
            descriptors[head] = deepcopy(profile["specialist_teacher_descriptors"][head])
    wrapper["clf_distil_acc_coef"] = declared.get("clf_distil_acc_coef", 0.0) if clf_source != "none" else 0.0
    wrapper.update({
        "use_ema": False, "test_network_name": "raw", 
        "defer_teacher": previous, "trainable_teacher": current, 
        "teacher_training": "each_task", "teacher_dynamic_classes": current, 
        "dual_teacher_scope": "task", "clf_distil_scope": "old_classes", 
        "clf_train_batch_fraction": 0.0, "clf_train_class_input_type": "null_class_only", 
        "clf_train_noisy_input_type": "noisy", "clf_test_noisified_max_timesteps": 0, 
        "mask_by_nulls": False, "mask_by_t_threshold": False, 
        "use_ensemble_loss_instead": False, "train_cfg_scale": None
    })
    corruption = categorical("classifier_noise")
    cap = {"clean": 0, "noisy32": 32, "noisy128": 128, "full": -1}[corruption]
    # Classifier corruption must stay inside the unchanged diffusion horizon.
    if cap > raw.get("timesteps", 1000):
        raise ValueError("Classifier corruption cap exceeds the fixed input diffusion horizon; restrict classifier_noise.")
    wrapper["clf_train_noisified_max_timesteps"] = cap
    wrapper["p_uncond"] = categorical("p_uncond")
    strategy_key = "continual_strategy_with_previous" if previous else "continual_strategy_without_previous"
    strategies = ["generative_replay", "cumulative"] if previous else SEARCH_SPACE["continual_strategy"]
    requested_strategies = _choice_options(overrides, "continual_strategy")
    permitted_strategies = [item for item in requested_strategies if item in strategies]
    # Previous teachers need old examples in the training protocol to remain active.
    if not permitted_strategies:
        raise ValueError("Previous-task distillation requires old examples through cumulative training or generative replay.")
    strategy = _TrialView(trial, overrides={strategy_key: permitted_strategies}).suggest_categorical(
        strategy_key, strategies
    )
    replay = strategy == "generative_replay"
    budget_mode = "legacy"
    replay_samples = 0
    train_num = -1
    old_examples = None
    current_examples = None
    selection = "all"
    candidate_multiplier = 1
    surprise_weight = 0.5
    # Reverse sampling and replay-selection controls apply only to generated replay.
    if replay:
        budget_mode = categorical("replay_budget_mode")
        # Legacy budgets specify examples per old class and a current-training cap.
        if budget_mode == "legacy":
            replay_samples = categorical("replay_samples")
            train_num = categorical("train_num")
        # Fixed-total budgets specify independent old and current example counts.
        else:
            old_examples = categorical("replay_old_examples")
            current_examples = categorical("replay_current_examples")
        selection = categorical("replay_selection")
        # Candidate overgeneration is useful only when a selector filters candidates.
        if selection != "all":
            candidate_multiplier = categorical("replay_candidate_multiplier")
        # Only the combined score needs a confidence-versus-surprise mixing weight.
        if selection == "confidence_surprise":
            surprise_weight = numeric("replay_surprise_weight")
        horizon = raw.get("timesteps", 1000)
        steps = suggestions.suggest_categorical(
            f"test_steps_t{horizon}", [value for value in SEARCH_SPACE["test_steps"] if value <= horizon]
        )
        wrapper.update({"test_steps": steps, "test_cfg_scale": numeric("test_cfg_scale"), "test_eta": categorical("test_eta")})
    # Data-only protocols still search the shared cap on current training examples.
    else:
        train_num = categorical("train_num")
    optimizer = categorical("optimizer")
    clipnorm = categorical("clipnorm")
    optimization = {
        "name": optimizer, "initial_learning_rate": numeric("learning_rate"), 
        "weight_decay": numeric("weight_decay") if optimizer == "adamw" else None, 
        "clipnorm": clipnorm, "global_clipnorm": categorical("global_clipnorm") if clipnorm is None else None, 
        "schedule": "constant", "plateau_jump": False
    }
    config.optimizer = type(config.optimizer)(**optimization)
    config.dataset = type(config.dataset)(
        batch_size=categorical("batch_size"), preprocess=None, 
        validation_ratio=0.2, validation_source="split", onehot_labels=False, 
        max_train_samples=max_train_samples, max_val_samples=max_val_samples, drop_remainder=False, name="cifar10"
    )
    config.continually_learn = type(config.continually_learn)(
        class_num=10, class_order=profile["class_order"], task_groups=profile["task_groups"], 
        task_size=2, class_order_mode="fixed", task_order_mode="fixed", 
        remove_prev_classes=strategy != "cumulative", keep_same_model=True, 
        use_generative_replay=replay, replay_budget_mode=budget_mode, 
        replay_old_examples=old_examples, replay_current_examples=current_examples, 
        replay_selection=selection, replay_candidate_multiplier=candidate_multiplier, 
        replay_surprise_weight=surprise_weight, use_generative_model_classifier=True, 
        train_classifier_separately=False, use_distillation=previous, 
        dual_teacher_distillation=False, snapshot_network_name="raw", 
        specialist_teacher_descriptors=descriptors, 
        use_ensemble_accuracy=use_ensemble_accuracy, evaluate_ensemble_accuracy=use_ensemble_accuracy, 
        ensemble_accuracy_kwargs=dict(ensemble_accuracy_kwargs or {}), 
        generative_model_kwargs={"samples_per_class": replay_samples, "train_num": train_num}, 
        plot_results=False, show_generated_images=False, save_task_checkpoints=True, 
        experiment_phase="development", seed=seed
    )
    metrics, directions = _normalize_objective_spec(
        "continual", objective_metrics, objective_directions, use_ensemble_accuracy
    )
    root = Path(results_path) / "continual" / "dit_classifier" / "cifar10" / PROFILE
    tensorboard_name = _tensorboard_name(trial)
    config.hpo = {
        "search_profile": PROFILE, "profile_version": VERSION, "study_task": "continual", 
        "study_model": "dit_classifier", "model_family": "dit_classifier", 
        "trial_number": trial.number, "params": deepcopy(dict(trial.params)), 
        "tensorboard_name": tensorboard_name, "objective_metrics": list(metrics), 
        "objective_directions": list(directions), "objective_network": "raw", 
        "use_ensemble_accuracy": use_ensemble_accuracy, "ensemble_accuracy_kwargs": dict(ensemble_accuracy_kwargs or {}), 
        "use_distillation": previous or current, "snapshot_network_name": "raw", 
        "continual_strategy": strategy, "classifier_teacher_source": clf_source, "noise_teacher_source": noise_source, 
        "continual_dataset_seed": profile["task_seed"], "continual_profile": profile, 
        "continual_schedule": {"class_num": 10, "class_order": profile["class_order"], "task_groups": profile["task_groups"], "task_size": 2}, 
        "declared_distillation": {key: value for key, value in declared.items() if "distil" in key or "acc_coef" in key or "teacher" in key}, 
        "effective_distillation": {key: value for key, value in wrapper.items() if "distil" in key or "acc_coef" in key or "teacher" in key}, 
        "classifier_training": {"recipe": corruption, "class_input": "null_class_only", "rows": "all_examples"}, 
        "fixed_recipe": {"architecture_search": False, "ema": False, "wrapper": "diffusion_classifier", "teacher_training": "each_task", "test_set_used_for_hpo": False}, 
        "seed": seed, "dtype_policy": dtype_policy, "deterministic_ops": bool(deterministic_ops)
    }
    config.training = type(config.training)(
        task="continual", epochs=epochs, fit_method="fit", use_valset=True, 
        patience=0, reduce_lr_patience=0, tensorboard=True, 
        tensorboard_path=str(root / "tensorboard"), tensorboard_run_name=tensorboard_name, 
        results_path=str(root / "runs"), project_tag=f"t{trial.number:04d}", 
        show_images=False, save_gifs=False, report_every_epoch=False, 
        dtype_policy=dtype_policy, deterministic_ops=deterministic_ops, verbose=1, seed=seed
    )
    config.reporting = type(config.reporting)(
        show_history_plot=False, save_history_plot=True, show_final_images=False, 
        save_final_images=False, save_final_gifs=False, run_trainset_eval=False, 
        run_valset_eval=False, save_csv=True
    )
    config.model.show_network_summary = False
    return config


def validate_continual_search(
    continual_profile: Mapping[str, object], 
    search_space_overrides: Mapping[str, object] | None = None
) -> None:
    """Require usable fixed inputs for every requested teacher treatment before study allocation.

    The profile must already be normalized. This check reads its serialized
    wrapper and descriptors without importing TensorFlow or loading a model.
    Active teacher terms need nonzero declared coefficients and role weights;
    current-task sources need specialist artifacts. Previous-task sources need
    cumulative data or generative replay so their old-example route is active.
    Numeric loss domains remain owned by the existing model APIs.
    """

    overrides = dict(search_space_overrides or {})
    # Validate the complete requested space before any trials or workers are allocated.
    if set(overrides) - set(SEARCH_SPACE):
        raise ValueError(f"Unknown continual search dimensions: {sorted(set(overrides) - set(SEARCH_SPACE))}")
    wrapper = continual_profile["student_config"]["model"]["wrapper_kwargs"]
    descriptors = continual_profile["specialist_teacher_descriptors"]
    previous_requested = False
    for head, loss_key in (
        ("classifier", "clf_distil_loss_coef"), ("noise", "noise_distil_loss_coef")
    ):
        sources = _choice_options(overrides, f"{head}_teacher_source")
        # Any active requested source needs its head's declared distillation loss.
        if any(source != "none" for source in sources) and not wrapper.get(loss_key, 0.0):
            raise ValueError(f"The input architecture must declare an active {loss_key} for {head} distillation.")
        role_head = "clf" if head == "classifier" else "noise"
        for role in ("previous", "current"):
            selected = any(source in (role, "both") for source in sources)
            key = f"{role}_teacher_{role_head}_loss_weight"
            # A declared zero role weight would make this treatment a silent no-op.
            if selected and not wrapper.get(key, 1.0):
                raise ValueError(f"Selected teacher route has zero input weight: {key}")
            # Current-task sources require the corresponding independent specialist.
            if role == "current" and selected and head not in descriptors:
                raise ValueError(f"Supply the {head} specialist descriptor before searching current-task distillation.")
            # Native denoiser specialists obey this experiment's no-EMA policy too.
            if role == "current" and selected and head == "noise":
                noise_config = load_config(descriptors[head]["path"])
                noise_wrapper = noise_config.model.wrapper_kwargs or noise_config.model.diffusion_model.kwargs()
                if noise_wrapper.get("use_ema", True) or noise_wrapper.get("test_network_name", "ema") != "raw":
                    # Reject an incompatible input rather than mutate the supplied teacher recipe.
                    raise ValueError("The UNet specialist Config must use use_ema=False and test_network_name='raw'.")
            # Remember previous-source requirements before checking the strategy grid.
            if role == "previous" and selected:
                previous_requested = True
    strategies = _choice_options(overrides, "continual_strategy")
    # Reject a grid whose previous teachers would never receive an old example.
    if previous_requested and not any(strategy in ("generative_replay", "cumulative") for strategy in strategies):
        raise ValueError("Previous-task distillation requires old examples through cumulative training or generative replay.")
