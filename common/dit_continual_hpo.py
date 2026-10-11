"""Evidence-informed V1 DiT continual search through the existing learner.

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
VERSION = 2
SEARCH_SPACE = {
    "student_config": [], 
    "classifier_teacher_descriptor": [], 
    "noise_teacher_descriptor": [], 
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

    A catalog exposes sealed student and specialist IDs to Optuna; a legacy
    ``student_config`` supplies one native Config mapping or YAML path. The
    artifact bridge normalizes specialist descriptors without loading models.
    Each recipe retains its architecture, losses and diffusion convention;
    repeated normalization verifies rather than replaces artifact identities.
    ``task_seed`` owns the task partition independently of model seeds used in
    search or confirmation. Evaluation uses the official test set.
    """

    from common.specialist_teacher_artifacts import normalize_specialist_teacher_descriptors


    # Catalogs are sealed before Optuna sees candidate IDs or starts any workers.
    if isinstance(continual_profile, Mapping) and "student_candidates" in continual_profile:
        return _normalize_candidate_profile(continual_profile, dataset_name, seed=seed)
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
    # Every student recipe implements the attached DiT classifier with the V1 wrapper.
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


def _normalize_candidate_profile(
    profile: Mapping[str, object], dataset_name: str, seed: int = 42
) -> dict:
    """Bind every candidate recipe and a common randomized five-task schedule.

    Candidate IDs are categorical hyperparameters. Configurations, allowed batch
    sizes and provenance are immutable inputs; trained historical scores are not
    imported as Optuna observations. All candidate epochs use the same horizon.
    """

    from common.specialist_teacher_artifacts import normalize_specialist_teacher_descriptors


    allowed = {
        "student_candidates", "classifier_teacher_candidates", "noise_teacher_candidates", 
        "task_seed", "task_groups", "class_order", "artifact_sha256", "profile_version"
    }
    # Unknown fields or an obsolete seal cannot silently change study identity.
    if set(profile) - allowed or profile.get("profile_version", VERSION) != VERSION:
        raise ValueError("Unknown or obsolete continual candidate profile; create a fresh study.")
    students = profile["student_candidates"]
    # Optuna needs at least one explicit student recipe to construct its choices.
    if not isinstance(students, Mapping) or not students:
        raise ValueError("Supply a nonempty student_candidates mapping.")
    normalized = {"student_candidates": {}, "artifact_sha256": {}, "profile_version": VERSION}
    for candidate_id, item in students.items():
        # Stable categorical IDs must identify serialized recipe mappings.
        if not isinstance(candidate_id, str) or not candidate_id or not isinstance(item, Mapping):
            raise ValueError("Candidate IDs must be nonempty strings with explicit recipe mappings.")
        # Reject unused candidate controls before they can appear in a sealed recipe.
        if set(item) - {"config", "provenance", "batch_sizes", "epoch_multiplier", "artifact_sha256"}:
            raise ValueError(f"Unknown student candidate fields: {candidate_id}")
        # Unequal horizons would confound architecture selection with training exposure.
        if item.get("epoch_multiplier", 1.0) != 1.0:
            raise ValueError("Student candidates must share the per-task epoch horizon.")
        fixed = normalize_continual_profile({
            "student_config": item["config"], "task_seed": profile.get("task_seed", seed), 
            "artifact_sha256": item.get("artifact_sha256", {})
        }, dataset_name, seed=seed)
        batches = list(item.get("batch_sizes", [64, 128]))
        # A candidate declares an exact categorical batch set, not an implicit range.
        if not batches or len(set(batches)) != len(batches) or any(type(value) is not int for value in batches):
            raise ValueError("Candidate batch_sizes must contain distinct integer choices.")
        normalized["student_candidates"][candidate_id] = {
            "config": fixed["student_config"], "batch_sizes": batches, "epoch_multiplier": 1.0, 
            "artifact_sha256": fixed["artifact_sha256"], "provenance": deepcopy(item.get("provenance", {}))
        }
        normalized["artifact_sha256"].update(fixed["artifact_sha256"])
        for field in ("task_seed", "task_groups", "class_order"):
            normalized[field] = fixed[field]
    for role in ("classifier", "noise"):
        key = f"{role}_teacher_candidates"
        candidates = profile.get(key, {})
        # Specialist roles use the same stable ID-to-recipe convention as students.
        if not isinstance(candidates, Mapping):
            raise ValueError(f"{key} must map candidate IDs to descriptors.")
        normalized[key] = {}
        for candidate_id, item in candidates.items():
            # Each specialist entry contains only its descriptor and evidence record.
            if not isinstance(candidate_id, str) or not candidate_id or not isinstance(item, Mapping) \
            or set(item) - {"descriptor", "provenance"}:
                raise ValueError(f"Invalid {role} candidate declaration.")
            descriptor = normalize_specialist_teacher_descriptors({role: item["descriptor"]})[role]
            normalized[key][candidate_id] = {
                "descriptor": descriptor, "provenance": deepcopy(item.get("provenance", {}))
            }
    for field in ("task_groups", "class_order", "artifact_sha256"):
        # Repeated normalization must preserve the frozen schedule and artifact seal.
        if field in profile and profile[field] != normalized[field]:
            raise ValueError(f"Sealed continual candidate {field} changed.")
    json.dumps(normalized, allow_nan=False)
    return normalized


def _candidate_options(profile: Mapping[str, object], overrides: Mapping[str, object], role: str) -> list:
    """Resolve a stable categorical distribution against the sealed catalog."""

    key = "student_config" if role == "student" else f"{role}_teacher_descriptor"
    pool = "student_candidates" if role == "student" else f"{role}_teacher_candidates"
    available = list(profile[pool])
    requested = overrides.get(key, available)
    # The adapter's mapping syntax permits only an explicit categorical envelope.
    if isinstance(requested, Mapping):
        # Numeric distributions cannot identify a subset of named recipes.
        if set(requested) != {"choices"}:
            raise ValueError(f"{key} accepts only explicit categorical choices.")
        requested = requested["choices"]
    requested = list(requested) if isinstance(requested, (list, tuple)) else [requested]
    # Restrict candidate draws to distinct IDs present in the sealed catalog.
    if not requested or len(set(requested)) != len(requested) or any(value not in available for value in requested):
        raise ValueError(f"Unknown or empty {key} candidates: {requested}")
    return requested


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


def baseline_hints(
    search_space_overrides: Mapping[str, object] | None = None, 
    continual_profile: Mapping[str, object] | None = None
) -> list[dict]:
    """Queue source treatments and balance catalog IDs without importing old scores."""

    overrides = dict(search_space_overrides or {})
    classifier = _choice_options(overrides, "classifier_teacher_source")
    noise = _choice_options(overrides, "noise_teacher_source")
    hints = [
        {"classifier_teacher_source": clf_source, "noise_teacher_source": noise_source}
        for clf_source in classifier for noise_source in noise
    ]
    # Catalog hints distribute student and active specialist IDs across source routes.
    if continual_profile is not None and "student_candidates" in continual_profile:
        students = _candidate_options(continual_profile, overrides, "student")
        counters = {"classifier": 0, "noise": 0}
        for index, hint in enumerate(hints):
            hint["student_config"] = students[index % len(students)]
            for role in ("classifier", "noise"):
                # Only current-task routes consume an independent specialist choice.
                if hint[f"{role}_teacher_source"] in ("current", "both"):
                    candidates = _candidate_options(continual_profile, overrides, role)
                    hint[f"{role}_teacher_descriptor"] = candidates[counters[role] % len(candidates)]
                    counters[role] += 1
        # Matched no-distillation anchors give every student an interpretable baseline.
        if "none" in classifier and "none" in noise:
            for student_id in students:
                anchor = {
                    "student_config": student_id, "classifier_teacher_source": "none", 
                    "noise_teacher_source": "none"
                }
                # Existing source hints already contain some of these exact anchors.
                if anchor not in hints:
                    hints.append(anchor)
    return hints


def _classifier_teacher_input_policy(
    student_wrapper: Mapping[str, object], 
    student_horizon: int, 
    source: str, 
    descriptor: Mapping[str, object] | None
) -> dict:
    """Choose a supported shared classifier view without changing epsilon inputs.

    Native classifier roles share one teacher corruption draw in the existing
    wrapper. Mixed clean/noisy roles therefore use their common clean view;
    this restriction changes neither each teacher's training recipe nor the
    independent noise-distillation route.
    """

    from common.specialist_teacher_artifacts import specialist_teacher_config


    def training_cap(wrapper: Mapping[str, object], horizon: int) -> int:
        """Read the classifier's actual training support using native cap meanings."""

        # Explicit clean-input recipes support only the clean classifier view.
        if wrapper.get("clf_train_noisy_input_type", "noisy") == "clean":
            return 0
        cap = wrapper.get("clf_train_noisified_max_timesteps")
        return horizon if cap == -1 else 0 if cap is None else cap


    caps = {}
    # A previous snapshot inherits the selected student's classifier training support.
    if source in ("previous", "both"):
        caps["previous"] = training_cap(student_wrapper, student_horizon)
    # Current specialists supply their own training support to the shared input policy.
    if source in ("current", "both"):
        # Ordinary image classifiers have no diffusion-noise training convention.
        if descriptor["format"] == "keras":
            caps["current"] = 0
        # Native joint specialists retain the noise cap from their sealed recipe.
        else:
            teacher = specialist_teacher_config(descriptor)
            teacher_wrapper = teacher.model.wrapper_kwargs or teacher.model.diffusion_classifier.kwargs()
            teacher_network = teacher.model.kwargs or teacher.model.dit_classifier.kwargs()
            caps["current"] = training_cap(teacher_wrapper, teacher_network.get("timesteps", 1000))
    shared_cap = min(caps.values()) if caps else None
    return {
        "enabled": bool(caps), "policy": "common_supported_classifier_view", 
        "role_training_support_caps": caps, "shared_native_role_input": True, 
        "input_type": "clean" if shared_cap == 0 else "noisy", "cap": shared_cap, 
        "noise_distillation_inputs_changed": False, 
        "limitation": "Native classifier roles share one draw; mixed clean/noisy roles use clean targets."
    }


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
    """Build one test-guided V1 continual trial from sealed native recipes.

    Classifier/noise teacher sources form a Cartesian sixteen-treatment search.
    Catalog mode also selects student architecture and current specialist IDs.
    Selected terms retain the recipe's distillation type, temperature, loss and
    accuracy coefficients and per-role weights. Disabled terms are gated to
    zero, never reweighted or optimized. Current specialists are trained on
    each task's real current examples by the existing continual learner. The
    classifier-teacher input policy uses common supported clean/noisy views;
    it leaves epsilon-distillation inputs unchanged. All candidates share the
    sealed five-task schedule and official-test evaluation protocol.
    """

    from common.hpo import _TrialView, _normalize_objective_spec, _tensorboard_name


    profile = normalize_continual_profile(continual_profile, dataset_name, seed=seed)
    # Keep the numerical execution policy common across all candidate treatments.
    if dtype_policy != "float32":
        raise ValueError("The fixed continual profile requires float32.")
    overrides = dict(search_space_overrides or {})
    # Direct builder calls reject unavailable inputs before consuming any trial draws.
    validate_continual_search(profile, overrides)
    # Unknown dimensions cannot silently enlarge the declared search protocol.
    if set(overrides) - set(SEARCH_SPACE):
        raise ValueError(f"Unknown continual search dimensions: {sorted(set(overrides) - set(SEARCH_SPACE))}")
    suggestions = _TrialView(trial, overrides=overrides)


    def categorical(name: str) -> object:
        """Sample one named categorical choice through the common HPO adapter."""

        return suggestions.suggest_categorical(name, SEARCH_SPACE[name])


    def numeric(name: str) -> float:
        """Sample one named numeric distribution through the common adapter."""

        return suggestions.suggest_float(name, **SEARCH_SPACE[name])


    candidate_mode = "student_candidates" in profile
    selected_candidates = {}
    student_entry = None
    # Catalog mode samples the sealed student ID before its dependent parameters.
    if candidate_mode:
        student_id = suggestions.suggest_categorical("student_config", _candidate_options(profile, overrides, "student"))
        student_entry = profile["student_candidates"][student_id]
        selected_candidates["student_config"] = student_id
        config = Config(**deepcopy(student_entry["config"]))
    # Single-recipe callers retain their explicitly supplied native student config.
    else:
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
            # A catalog route samples a descriptor from its sealed role-specific pool.
            if candidate_mode:
                key = f"{head}_teacher_descriptor"
                candidate_id = suggestions.suggest_categorical(key, _candidate_options(profile, overrides, head))
                selected_candidates[key] = candidate_id
                descriptors[head] = deepcopy(profile[f"{head}_teacher_candidates"][candidate_id]["descriptor"])
            # A requested specialist cannot silently fall back to the student snapshot.
            elif head not in profile["specialist_teacher_descriptors"]:
                raise ValueError(f"Supply the {head} specialist descriptor before searching current-task distillation.")
            # Legacy profiles retain the one descriptor explicitly supplied for this role.
            else:
                descriptors[head] = deepcopy(profile["specialist_teacher_descriptors"][head])
    wrapper["clf_distil_acc_coef"] = declared.get("clf_distil_acc_coef", 0.0) if clf_source != "none" else 0.0
    wrapper.update({
        "use_ema": False, "test_network_name": "raw", 
        "defer_teacher": previous, "trainable_teacher": current, 
        "teacher_training": "each_task", "teacher_dynamic_classes": current, 
        "dual_teacher_scope": "task", "clf_distil_scope": "old_classes", 
        "clf_train_batch_fraction": 0.0, "clf_train_class_input_type": "null_class_only", 
        "clf_train_noisy_input_type": "noisy", 
        "mask_by_nulls": False, "mask_by_t_threshold": False, 
        "use_ensemble_loss_instead": False, "train_cfg_scale": None
    })
    if candidate_mode:
        # Clean/noisy candidates retain their paired train AND test corruption recipe.
        cap = declared.get("clf_train_noisified_max_timesteps", 0)
        corruption = "clean" if cap == 0 else f"candidate_noisy{cap}"
        wrapper["clf_test_noisified_max_timesteps"] = declared.get("clf_test_noisified_max_timesteps", 0)
    # Legacy single-recipe search retains its independent training-noise dimension.
    else:
        corruption = categorical("classifier_noise")
        cap = {"clean": 0, "noisy32": 32, "noisy128": 128, "full": -1}[corruption]
        wrapper["clf_test_noisified_max_timesteps"] = 0
    # Classifier corruption must stay inside the unchanged diffusion horizon.
    if cap > raw.get("timesteps", 1000):
        raise ValueError("Classifier corruption cap exceeds the fixed input diffusion horizon; restrict classifier_noise.")
    wrapper["clf_train_noisified_max_timesteps"] = cap
    classifier_teacher_inputs = {"policy": "declared_wrapper_inputs"}
    # Catalog teachers can differ in training support, so select a shared valid view.
    if candidate_mode:
        classifier_teacher_inputs = _classifier_teacher_input_policy(
            wrapper, raw.get("timesteps", 1000), clf_source, descriptors.get("classifier")
        )
        # Disabled classifier distillation does not need a replacement input policy.
        if classifier_teacher_inputs["enabled"]:
            wrapper.update({
                "clf_distil_noisy_input_type": classifier_teacher_inputs["input_type"], 
                "clf_distil_train_noisified_max_timesteps": classifier_teacher_inputs["cap"], 
                "clf_distil_test_noisified_max_timesteps": classifier_teacher_inputs["cap"]
            })
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
        step_options = [value for value in _choice_options(overrides, "test_steps") if value <= horizon]
        # A replay route must expose at least one sampling count inside its horizon.
        if not step_options:
            raise ValueError("No replay sampling step count fits the selected diffusion horizon.")
        steps = suggestions.suggest_categorical(f"test_steps_t{horizon}", step_options)
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
    # Catalog batch choices belong to the selected student's declared runtime recipe.
    if candidate_mode:
        allowed_batches = student_entry["batch_sizes"]
        # Global batch restrictions narrow the candidate set without inventing new sizes.
        if "batch_size" in overrides:
            requested_batches = overrides["batch_size"]
            requested_batches = requested_batches.get("choices") if isinstance(requested_batches, Mapping) else requested_batches
            requested_batches = list(requested_batches) if isinstance(requested_batches, (list, tuple)) else [requested_batches]
            allowed_batches = [value for value in allowed_batches if value in requested_batches]
        # An empty intersection cannot produce a valid batch-size draw.
        if not allowed_batches:
            raise ValueError("No batch size fits the selected student's declared candidate choices.")
        # Candidate-specific names avoid Optuna's changing-categorical-distribution error.
        batch_size = suggestions.suggest_categorical(f"batch_size_{student_id}", allowed_batches)
    # Single-recipe profiles use the shared batch-size search distribution.
    else:
        batch_size = categorical("batch_size")
    config.dataset = type(config.dataset)(
        batch_size=batch_size, preprocess=None, 
        validation_ratio=0.0, validation_source="test", onehot_labels=False, 
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
        "selected_candidates": selected_candidates, "evaluation_source": "official_test", 
        "continual_schedule": {"class_num": 10, "class_order": profile["class_order"], "task_groups": profile["task_groups"], "task_size": 2}, 
        "declared_distillation": {key: value for key, value in declared.items() if "distil" in key or "acc_coef" in key or "teacher" in key}, 
        "effective_distillation": {key: value for key, value in wrapper.items() if "distil" in key or "acc_coef" in key or "teacher" in key}, 
        "classifier_teacher_inputs": classifier_teacher_inputs, 
        "classifier_training": {"recipe": corruption, "class_input": "null_class_only", "rows": "all_examples"}, 
        "fixed_recipe": {"architecture_search": candidate_mode, "ema": False, "wrapper": "diffusion_classifier", "teacher_training": "each_task", "test_set_used_for_hpo": True}, 
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
    """Require usable sealed inputs for every requested treatment before study allocation.

    The profile must already be normalized. This check reads its serialized
    wrapper and descriptors without importing TensorFlow or loading a model.
    Active teacher terms need nonzero declared coefficients and role weights;
    current-task sources need specialist descriptors. Catalogs validate each
    requested student/specialist combination. Previous-task sources need
    cumulative data or generative replay so their old-example route is active.
    Numeric loss domains remain owned by the existing model APIs.
    """

    overrides = dict(search_space_overrides or {})
    # Validate the complete requested space before any trials or workers are allocated.
    if set(overrides) - set(SEARCH_SPACE):
        raise ValueError(f"Unknown continual search dimensions: {sorted(set(overrides) - set(SEARCH_SPACE))}")
    # Validate every selectable catalog combination before any Optuna allocation.
    if "student_candidates" in continual_profile:
        # Candidate IDs own paired train/test corruption and must remain interpretable.
        if "classifier_noise" in overrides:
            raise ValueError("Candidate IDs own paired classifier train/test noise; do not override classifier_noise.")
        student_ids = _candidate_options(continual_profile, overrides, "student")
        descriptor_sets = [{}]
        for role in ("classifier", "noise"):
            sources = _choice_options(overrides, f"{role}_teacher_source")
            # Independent specialist combinations exist only for active current routes.
            if any(source in ("current", "both") for source in sources):
                ids = _candidate_options(continual_profile, overrides, role)
                descriptor_sets = [
                    {**entries, role: continual_profile[f"{role}_teacher_candidates"][candidate_id]["descriptor"]}
                    for entries in descriptor_sets for candidate_id in ids
                ]
        fixed_overrides = {
            key: value for key, value in overrides.items()
            if key not in {"student_config", "classifier_teacher_descriptor", "noise_teacher_descriptor"}
        }
        for student_id in student_ids:
            for descriptors in descriptor_sets:
                validate_continual_search({
                    "student_config": continual_profile["student_candidates"][student_id]["config"], 
                    "specialist_teacher_descriptors": descriptors
                }, fixed_overrides)
        return
    # Single-recipe profiles cannot resolve catalog-only categorical dimensions.
    if any(key in overrides for key in ("student_config", "classifier_teacher_descriptor", "noise_teacher_descriptor")):
        raise ValueError("Candidate selection dimensions require a candidate catalog.")
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
                from common.specialist_teacher_artifacts import specialist_teacher_config


                noise_config = specialist_teacher_config(descriptors[head])
                defaults = noise_config.model.diffusion_classifier if noise_config.model.name == "dit_classifier" \
                    else noise_config.model.diffusion_model
                noise_wrapper = noise_config.model.wrapper_kwargs or defaults.kwargs()
                if noise_wrapper.get("use_ema", True) or noise_wrapper.get("test_network_name", "ema") != "raw":
                    # Reject an incompatible input rather than mutate the supplied teacher recipe.
                    raise ValueError("The noise specialist Config must use use_ema=False and test_network_name='raw'.")
            # Remember previous-source requirements before checking the strategy grid.
            if role == "previous" and selected:
                previous_requested = True
    strategies = _choice_options(overrides, "continual_strategy")
    # Reject a grid whose previous teachers would never receive an old example.
    if previous_requested and not any(strategy in ("generative_replay", "cumulative") for strategy in strategies):
        raise ValueError("Previous-task distillation requires old examples through cumulative training or generative replay.")
