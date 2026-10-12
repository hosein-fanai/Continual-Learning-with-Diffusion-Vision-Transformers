"""Semantic phase optimization around sealed native continual recipes.

The profile reads and seals configuration without importing TensorFlow. Search
ranges are development domains, not validated optima. Every trial runs the
existing acquisition/consolidation implementation after every joint task.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from common.config import Config, _safe_load_unique_yaml, load_config, resolve_continual_schedule
from semantic_consolidation.config import RouteConfig, RouteSettings, validate_route_config


PROFILE = "semantic_consolidation_runner"
VERSION = 1
RECIPE_CATALOG_VERSION = 1
PHASE_BUDGET_FIELDS = (
    "acquisition_steps", "consolidation_steps", "batch_size", "augmentation_views", "noise_levels"
)
SEARCH_SPACE = {
    "acquisition_steps": [50, 100, 200, 400, 1000], 
    "consolidation_steps": [100, 200, 400, 800, 2000], 
    "batch_size": [16, 32, 64, 128], 
    "learning_rate": {"low": 1e-5, "high": 3e-3, "log": True}, 
    "temperature": {"low": 0.03, "high": 0.5, "log": True}, 
    "alignment_weight": {"low": 0.1, "high": 10.0, "log": True}, 
    "ce_weight": {"low": 0.1, "high": 10.0, "log": True}, 
    "orthogonality_weight": {"low": 0.1, "high": 10.0, "log": True}, 
    "gain_limit": {"low": 0.1, "high": 2.0, "log": True}, 
    "bias_limit": {"low": 0.1, "high": 2.0, "log": True}, 
    "modulation_init_std": {"low": 0.001, "high": 0.1, "log": True}, 
    "acquisition_noise_level": [0, 10, 50, 100, 250, 500], 
    "ce_noise_level": [0, 10, 50, 100, 250, 500], 
    "noise_levels": ["0", "10", "50", "100", "250", "500", "0,10,50", "0,50,100", "0,100,250", "0,250,500"], 
    "image_augmentation": ["none", "tmcl"], 
    "augmentation_views": [2, 4, 8], 
    "reliability": ["uniform", "alpha_bar"], 
    "reliability_floor": [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
}
FIELD_CATALOG = {
    "acquisition_steps": {"role": "searched", "condition": "learned", "reason": "Positive optimizer-update budget per task; zero would remove acquisition."}, 
    "consolidation_steps": {"role": "searched", "condition": "learned", "reason": "Positive optimizer-update budget per task; zero would remove consolidation."}, 
    "batch_size": {"role": "searched", "condition": "both phases", "reason": "Independent semantic batch; the native balanced sampler may shrink it to available positive pairs."}, 
    "learning_rate": {"role": "searched", "condition": "both phases", "reason": "Shared semantic Adam rate; the ordinary joint optimizer remains fixed."}, 
    "temperature": {"role": "searched", "condition": "learned InfoNCE", "reason": "Consolidation contrastive temperature."}, 
    "alignment_weight": {"role": "searched", "condition": "learned", "reason": "Positive alignment contribution preserves the consolidation method."}, 
    "ce_weight": {"role": "searched", "condition": "learned", "reason": "Positive supervised contribution preserves the core objective."}, 
    "orthogonality_weight": {"role": "conditional", "condition": "acquisition_objective == contrastive", "reason": "Absent from suggestions when true-class CE is the fixed acquisition ablation."}, 
    "gain_limit": {"role": "searched", "condition": "learned modulation", "reason": "Positive affine gain bound."}, 
    "bias_limit": {"role": "searched", "condition": "learned modulation", "reason": "Positive affine bias bound."}, 
    "modulation_init_std": {"role": "searched", "condition": "learned modulation", "reason": "Gate initialization scale, independent of the fixed model architecture."}, 
    "acquisition_noise_level": {"role": "searched", "condition": "level < fixed diffusion horizon", "reason": "Zero is genuinely clean; positive entries are actual schedule indices."}, 
    "ce_noise_level": {"role": "searched", "condition": "level < fixed diffusion horizon", "reason": "Independent supervised consolidation noise, without changing joint training noise."}, 
    "noise_levels": {"role": "searched", "condition": "all levels < fixed diffusion horizon", "reason": "Encoded categorical level sets resolve to native integer sequences; no timestep rescaling."}, 
    "image_augmentation": {"role": "searched", "condition": "CIFAR RGB32 geometry", "reason": "Native same-input or independent TMCL image views."}, 
    "augmentation_views": {"role": "conditional", "condition": "image_augmentation == tmcl", "reason": "No dummy suggestion when image augmentation is disabled."}, 
    "reliability": {"role": "conditional", "condition": "any positive consolidation noise level", "reason": "Clean views have alpha_bar=1, making both policies equivalent."}, 
    "reliability_floor": {"role": "conditional", "condition": "positive consolidation noise and reliability == alpha_bar", "reason": "A sampled floor only changes weights when it exceeds the selected schedule signal; record effective noise/weights when interpreting importance."}, 
    "condition": {"role": "fixed_method", "condition": "learned", "reason": "Baseline, random, CE-only and feature-distillation controls belong to separate paired ablations."}, 
    "extra_joint_seconds": {"role": "fixed_ablation", "condition": "not used by learned", "reason": "Measured time-matched baseline budgets cannot be invented or searched as semantic parameters."}, 
    "consolidation_scope": {"role": "fixed_ablation", "condition": "native input", "reason": "Default semantic projection/head scope; backbone scope changes the intervention and must be declared separately."}, 
    "acquisition_objective": {"role": "fixed_ablation", "condition": "native input", "reason": "Default contrastive acquisition; true-class CE is a declared shortcut-prone control."}, 
    "retain_modulators": {"role": "fixed_ablation", "condition": "native input", "reason": "Retention changes the continual memory protocol and remains fixed across the search."}, 
    "probe_batches": {"role": "fixed_diagnostic", "condition": "native input", "reason": "Observation coverage, not a training hyperparameter."}, 
    "probe_max_gates": {"role": "fixed_diagnostic", "condition": "native input", "reason": "Observation coverage, not a training bank cap."}, 
    "seed": {"role": "fixed_provenance", "condition": "search seed; paired confirmation seed", "reason": "Never optimized; trials share a seed, confirmations retain the sealed task/data split."}, 
    "extensions": {"role": "fixed_extension", "condition": "native input", "reason": "Scheduling, replay, evaluation and KD-allocation treatments remain declared fixed mappings."}, 
    "experimental": {"role": "fixed_diagnostic", "condition": "native input", "reason": "Held-out diagnostics do not become optimization dimensions or training data."}, 
    "checkpoint_interval": {"role": "fixed_operational", "condition": "native input", "reason": "Recovery frequency is not an accuracy dimension; interrupted RUNNING trials require reconciliation and explicit reattempts start fresh."}
}


def _digest(path: Path) -> str:
    """Hash one supplied artifact without constructing a model."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    """Bind the complete JSON-compatible scientific recipe."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _seal_path(path: str | Path, artifacts: dict) -> Path:
    """Resolve and authenticate a declared file against any existing input seal."""

    selected = Path(path).expanduser().resolve(strict=True)
    digest = _digest(selected)
    # Reusing an input path cannot silently replace the scientific artifact.
    if str(selected) in artifacts and artifacts[str(selected)] != digest:
        raise ValueError(f"Semantic input artifact changed: {selected}")
    artifacts[str(selected)] = digest
    return selected


def _comparison_protocol(profile: Mapping[str, object]) -> dict:
    """Extract the common data, task, budget and evaluation comparison protocol."""

    native = profile["student_config"]
    dataset = deepcopy(native["dataset"])
    for key in ("batch_size", "trainset_len"):
        dataset.pop(key, None)
    model = native["model"]
    wrapper = deepcopy(model["diffusion_classifier"])
    # Generic wrapper mappings use the same defaults as the typed configuration.
    if model["name"] is not None and model["wrapper_kwargs"]:
        wrapper = asdict(Config().model.diffusion_classifier)
        wrapper.update(model["wrapper_kwargs"])
    return {
        "dataset": dataset, 
        "training": {
            key: native["training"][key] for key in (
                "epochs", "dtype_policy", "deterministic_ops", "seed", "task", "use_valset", 
                "fit_method", "stage_tasks", "stages_num", "stage_epochs", "final_epochs", 
                "timestep_boundaries", "timestep_clustering_type", "resolutions", "depths", 
                "pacing_type", "earlystopping_type", "progressive_monitor", "progressive_patience", 
                "min_delta", "stopper_mode", "fit_kwargs", "patience", "monitor", "monitor_mode", 
                "ensemble_monitor"
            )
        }, 
        "continual": {
            key: native["continually_learn"][key] for key in (
                "class_num", "task_size", "remove_prev_classes", "experiment_phase", 
                "use_ensemble_accuracy", "evaluate_ensemble_accuracy", "ensemble_accuracy_kwargs"
            )
        }, 
        "prediction": {
            key: wrapper[key] for key in (
                "test_network_name", "test_noisified_min_timesteps", "test_noisified_max_timesteps", 
                "clf_test_noisified_max_timesteps", "clf_acc_coef", "clf_distil_acc_coef", "ctr_acc_coef", 
                "preprocess_type", "resize_method", "resize_antialias"
            )
        }, 
        "reporting": deepcopy(native["reporting"]), 
        "task_seed": profile["task_seed"], "dataset_seed": profile["dataset_seed"], 
        "class_order": profile["class_order"], "task_groups": profile["task_groups"]
    }


def _recipe_profile(profile: Mapping[str, object], candidate_id: str) -> dict:
    """Expose one catalog recipe to native semantic range resolution."""

    candidates = profile.get("recipe_candidates", {})
    # Trial parameter evidence must identify an existing sealed complete recipe.
    if not isinstance(candidate_id, str) or candidate_id not in candidates:
        raise ValueError(f"Unknown semantic base_recipe: {candidate_id!r}.")
    selected = dict(profile)
    selected["student_config"] = candidates[candidate_id]
    return selected


def _normalize_phase_budgets(profile: Mapping[str, object]) -> dict:
    """Seal coupled native phase budgets and validate each against every recipe."""

    supplied = profile["phase_budget_candidates"]
    # Stable nonempty IDs identify complete coupled treatments.
    if not isinstance(supplied, Mapping) or not supplied or any(
        not isinstance(key, str) or not key for key in supplied
    ):
        raise ValueError("phase_budget_candidates must map nonempty string identifiers to native budgets.")
    budgets = {}
    recipes = profile.get("recipe_candidates", {"reference": profile["student_config"]})
    for budget_id in sorted(supplied):
        values = supplied[budget_id]
        # Partial or additional fields would change the declared coupling contract.
        if not isinstance(values, Mapping) or set(values) != set(PHASE_BUDGET_FIELDS):
            raise ValueError(f"Semantic budget {budget_id!r} must declare exactly {PHASE_BUDGET_FIELDS}.")
        candidate = deepcopy(dict(values))
        levels = candidate["noise_levels"]
        # Accept Optuna's categorical encoding as well as native integer sequences.
        if isinstance(levels, str):
            levels = [int(value) for value in levels.split(",")]
        candidate["noise_levels"] = levels
        settings = RouteSettings(**{**deepcopy(profile["route_settings"]), **candidate})
        normalized = asdict(settings)
        for field in PHASE_BUDGET_FIELDS:
            value = normalized[field]
            encoded = ",".join(str(level) for level in value) if field == "noise_levels" else value
            # Coupled tiers stay inside the supported native categorical domains.
            if encoded not in SEARCH_SPACE[field]:
                raise ValueError(f"Unsupported semantic budget {field}: {value!r}.")
        settings.extensions = {}
        settings.experimental = {}
        for recipe in recipes.values():
            validate_route_config(RouteConfig(common=Config(**deepcopy(recipe)), route=settings))
        budgets[budget_id] = {key: normalized[key] for key in PHASE_BUDGET_FIELDS}
    return budgets


def normalize_semantic_profile(
    semantic_profile: Mapping[str, object], 
    dataset_name: str = "cifar10", 
    seed: int = 42
) -> dict:
    """Seal native continual configuration and semantic settings without TensorFlow.

    ``student_config`` accepts Config, its mapping, or a native YAML path.
    ``route_settings`` accepts RouteSettings, its mapping, or standalone settings
    YAML. Omission uses the learned method with TMCL image augmentation. Native
    training, teacher, replay, optimizer, data and architecture values are fixed
    within each recipe. Optional ``recipe_candidates`` maps stable identifiers to
    complete native recipes. All recipes share the data, task, evaluation and
    epoch-budget protocol; their catalog and JSON provenance enter the seal.
    Optional extensions are validated by their owning runtime after admission.
    """

    from common.specialist_teacher_artifacts import normalize_specialist_teacher_descriptors


    # A complete native continual recipe is required before study allocation.
    if not isinstance(semantic_profile, Mapping) or "student_config" not in semantic_profile:
        raise ValueError("Supply semantic_profile.student_config as a native continual Config or YAML path.")
    allowed = {
        "student_config", "route_settings", "task_seed", "dataset_seed", "class_order", "task_groups", 
        "artifact_sha256", "profile_version", "fixed_recipe_sha256", 
        "recipe_candidates", "candidate_provenance", "recipe_catalog_version", 
        "selection_metric", "phase_budget_candidates"
    }
    # Unknown controls cannot masquerade as sealed scientific inputs.
    if set(semantic_profile) - allowed:
        raise ValueError(f"Unknown semantic profile options: {sorted(set(semantic_profile) - allowed)}")
    profile = deepcopy(dict(semantic_profile))
    # A changed profile version requires a distinct study identity.
    if profile.get("profile_version", VERSION) != VERSION:
        raise ValueError("Semantic profile version changed; create a new study.")
    artifacts = dict(profile.get("artifact_sha256", {}))
    supplied = profile["student_config"]
    source_directory = Path.cwd()
    # YAML paths bind exact bytes and provide the base for relative teacher paths.
    if isinstance(supplied, (str, Path)):
        path = _seal_path(supplied, artifacts)
        source_directory = path.parent
        config = load_config(path)
    # Native dataclasses retain all supplied values through a detached mapping.
    elif isinstance(supplied, Config):
        config = Config(**deepcopy(asdict(supplied)))
    # Serialized profiles reenter through the same typed Config boundary.
    elif isinstance(supplied, Mapping):
        config = Config(**deepcopy(dict(supplied)))
    # Live models cannot cross a coordinator-to-worker recipe boundary.
    else:
        raise TypeError("student_config must be Config, its mapping, or a native YAML path.")
    settings_input = profile.get("route_settings", {"image_augmentation": "tmcl"})
    # Standalone route settings files can be supplied without changing common fields.
    if isinstance(settings_input, (str, Path)):
        path = _seal_path(settings_input, artifacts)
        with path.open("r", encoding="utf-8") as stream:
            settings_input = _safe_load_unique_yaml(stream)
        # A one-key route envelope is accepted alongside native saved settings.
        if isinstance(settings_input, Mapping) and set(settings_input) == {"route"}:
            settings_input = settings_input["route"]
    # RouteSettings owns normalization and mathematical validity of native values.
    if isinstance(settings_input, RouteSettings):
        settings = RouteSettings(**deepcopy(asdict(settings_input)))
    # Mapping inputs never bypass RouteSettings validation.
    elif isinstance(settings_input, Mapping):
        settings = RouteSettings(**deepcopy(dict(settings_input)))
    # A complete route YAML cannot silently replace frozen common settings here.
    else:
        raise TypeError("route_settings must be RouteSettings, its mapping, or standalone settings YAML.")
    for path, expected in artifacts.items():
        # Renormalization authenticates every file preserved by the initial seal.
        if _digest(Path(path)) != expected:
            raise ValueError(f"Semantic input artifact changed: {path}")
    # Dataset selection must agree with the complete frozen native recipe.
    if dataset_name.lower() not in ("cifar10", "cifar100") or config.dataset.name != dataset_name.lower():
        raise ValueError("Semantic HPO requires matching native CIFAR10 or CIFAR100 configuration.")
    # Comparison of semantic strengths must retain the learned method in every trial.
    if settings.condition != "learned":
        raise ValueError("Semantic HPO searches condition='learned'; run other controls as separate paired ablations.")
    # Test-informed benchmarks are not a training-validation optimization protocol.
    if config.continually_learn.experiment_phase != "development" or config.dataset.validation_source != "split" \
    or not config.training.use_valset or config.dataset.validation_ratio <= 0:
        raise ValueError("Semantic HPO requires development phase and a positive held-out training validation split.")
    # A manifest belongs to its registered recipe, which a search deliberately varies.
    if any(getattr(config.continually_learn, key) is not None for key in (
        "experiment_manifest_path", "experiment_manifest_hash", "experiment_run_id"
    )):
        raise ValueError("Do not optimize a registered experiment manifest; supply a separate development recipe.")
    # Trial restoration is not allowed to bypass semantic gate/controller identity.
    if config.continually_learn.resume_from is not None:
        raise ValueError("Semantic HPO starts complete trials; native resume_from must be null.")
    wrapper = config.model.wrapper_kwargs if config.model.name is not None and config.model.wrapper_kwargs else asdict(config.model.diffusion_classifier)
    # The semantic adapter and task endpoint must share the raw no-EMA network.
    if wrapper.get("test_network_name", "ema") != "raw":
        raise ValueError("Semantic HPO requires the native wrapper test_network_name='raw'.")
    # Search feedback uses the same ordinary predictor before and after consolidation.
    if config.continually_learn.use_ensemble_accuracy:
        raise ValueError("Semantic HPO selects ordinary final_average_accuracy; native use_ensemble_accuracy must be false.")
    # Runtime teacher objects cannot be hashed or reconstructed as native inputs.
    if any(key.endswith("teacher_network") for key in wrapper):
        raise ValueError("Use native specialist descriptors, never live teacher objects in wrapper configuration.")
    descriptors = deepcopy(config.continually_learn.specialist_teacher_descriptors)
    for descriptor in descriptors.values():
        for key in ("path", "weights_path"):
            # Relative specialist paths belong to the native student YAML directory.
            if descriptor.get(key) is not None and not Path(descriptor[key]).is_absolute():
                descriptor[key] = str((source_directory / descriptor[key]).resolve())
    config.continually_learn.specialist_teacher_descriptors = normalize_specialist_teacher_descriptors(descriptors)
    task_seed = config.continually_learn.seed if config.continually_learn.seed is not None else config.training.seed
    dataset_seed = config.hpo.get("continual_dataset_seed", task_seed)
    # An explicitly unseeded native partition cannot be held fixed across candidates.
    if dataset_seed is None:
        raise ValueError("Semantic HPO requires a fixed native dataset seed; continual_dataset_seed cannot be null.")
    # A supplied task seed may only repeat, not replace, the native stream seed.
    if "task_seed" in profile and profile["task_seed"] != task_seed:
        raise ValueError("task_seed must match the frozen native continual/training seed.")
    # Native HPO input may already bind a data seed independently of model initialization.
    if "dataset_seed" in profile and profile["dataset_seed"] != dataset_seed:
        raise ValueError("dataset_seed must match the frozen native data partition seed.")
    core_settings = deepcopy(settings)
    core_settings.extensions = {}
    core_settings.experimental = {}
    validate_route_config(RouteConfig(common=config, route=core_settings))
    continual = config.continually_learn
    order, groups = resolve_continual_schedule(
        continual.class_num, continual.class_order, continual.task_groups, 
        available_class_num=10 if config.dataset.name == "cifar10" else 100, 
        task_size=continual.task_size, class_order_mode=continual.class_order_mode, 
        task_order_mode=continual.task_order_mode, seed=task_seed
    )
    # A continual search needs more than one task to measure retention.
    if len(groups) < 2:
        raise ValueError("Semantic continual HPO requires at least two class-incremental tasks.")
    # Repeated normalization must recover the exact original dense label mapping.
    if profile.get("class_order", order) != order or profile.get("task_groups", groups) != groups:
        raise ValueError("The sealed native task schedule changed.")
    normalized = {
        "student_config": asdict(config), "route_settings": asdict(settings), 
        "task_seed": task_seed, "dataset_seed": dataset_seed, "class_order": order, "task_groups": groups, 
        "artifact_sha256": artifacts, "profile_version": VERSION
    }
    # The catalog is optional so existing version-one single-recipe seals stay valid.
    if "recipe_candidates" in profile:
        candidates = profile["recipe_candidates"]
        # Empty catalogs or unstable identifiers cannot define an Optuna dimension.
        if not isinstance(candidates, Mapping) or not candidates or any(
            not isinstance(key, str) or not key for key in candidates
        ):
            raise ValueError("recipe_candidates must map nonempty string identifiers to native recipes.")
        # Catalog protocol changes require a new sealed study.
        if profile.get("recipe_catalog_version", RECIPE_CATALOG_VERSION) != RECIPE_CATALOG_VERSION:
            raise ValueError("Semantic recipe catalog version changed; create a new study.")
        provenance = profile.get("candidate_provenance", {})
        # Evidence must survive coordinator/worker JSON serialization.
        if not isinstance(provenance, Mapping):
            raise TypeError("candidate_provenance must be a JSON-compatible mapping.")
        # Serialization both authenticates evidence and rejects live objects or NaNs.
        provenance = json.loads(json.dumps(dict(provenance), sort_keys=True, allow_nan=False))
        reference_protocol = _comparison_protocol(normalized)
        sealed_candidates = {}
        for candidate_id in sorted(candidates):
            candidate = normalize_semantic_profile({
                "student_config": candidates[candidate_id], "route_settings": normalized["route_settings"]
            }, dataset_name, seed=seed)
            # Architecture/teacher changes cannot obtain different comparison cohorts.
            if _comparison_protocol(candidate) != reference_protocol:
                raise ValueError(
                    f"Semantic base_recipe {candidate_id!r} changed the shared data, task, training-budget or evaluation protocol."
                )
            for path, expected in candidate["artifact_sha256"].items():
                # Shared input paths retain one immutable artifact identity.
                if path in artifacts and artifacts[path] != expected:
                    raise ValueError(f"Semantic input artifact changed: {path}")
                artifacts[path] = expected
            sealed_candidates[candidate_id] = candidate["student_config"]
        normalized.update({
            "recipe_candidates": sealed_candidates, "candidate_provenance": provenance, 
            "recipe_catalog_version": RECIPE_CATALOG_VERSION
        })
    # Catalog metadata without a catalog cannot silently alter a legacy profile.
    elif "candidate_provenance" in profile or "recipe_catalog_version" in profile:
        raise ValueError("Candidate provenance and catalog version require recipe_candidates.")
    # An explicitly different scalar endpoint belongs in the scientific seal.
    if "selection_metric" in profile:
        selection_metric = profile["selection_metric"]
        # Preserve the named native average and the explicitly defined retention metric.
        if selection_metric not in ("final_average_accuracy", "retention_balanced_accuracy"):
            raise ValueError(f"Unsupported semantic selection_metric: {selection_metric!r}.")
        normalized["selection_metric"] = selection_metric
    # Optional coupled budgets preserve legacy independent semantic dimensions.
    if "phase_budget_candidates" in profile:
        normalized["phase_budget_candidates"] = profile["phase_budget_candidates"]
        normalized["phase_budget_candidates"] = _normalize_phase_budgets(normalized)
    digest = _json_digest(normalized)
    # Mapping mutation is detected independently of external file hashes.
    if profile.get("fixed_recipe_sha256", digest) != digest:
        raise ValueError("The sealed native semantic recipe changed.")
    normalized["fixed_recipe_sha256"] = digest
    return normalized


def _choices(field_name: str, overrides: Mapping[str, object], profile: Mapping[str, object]) -> list:
    """Resolve categorical restrictions inside a sealed diffusion horizon."""

    catalog_field = {"base_recipe": "recipe_candidates", "semantic_budget": "phase_budget_candidates"}.get(field_name)
    choices = list(profile.get(catalog_field, {})) if catalog_field is not None else list(SEARCH_SPACE[field_name])
    raw = profile["student_config"]["model"]
    effective = raw["kwargs"] if raw["name"] is not None and raw["kwargs"] else raw["dit_classifier"]
    horizon = int(effective.get("timesteps", 1000))
    # Native noise indices cannot equal or exceed the fixed diffusion horizon.
    if field_name in ("acquisition_noise_level", "ce_noise_level"):
        choices = [value for value in choices if value < horizon]
    # The string encoding keeps Optuna categoricals JSON-compatible and stable.
    elif field_name == "noise_levels":
        choices = [value for value in choices if max(int(item) for item in value.split(",")) < horizon]
    # Unrestricted categorical dimensions use the supported native choices.
    if field_name not in overrides:
        return choices
    selected = overrides[field_name]
    # Common search overrides allow only an explicit categorical choices envelope.
    if isinstance(selected, Mapping):
        # Numeric distributions cannot be attached to categorical native controls.
        if set(selected) != {"choices"}:
            raise ValueError(f"Categorical override {field_name} accepts only choices.")
        selected = selected["choices"]
    selected = list(selected) if isinstance(selected, (list, tuple)) else [selected]
    # Reject silently dropped interventions, including out-of-horizon noise levels.
    if not selected or len(set(selected)) != len(selected) or any(value not in choices for value in selected):
        raise ValueError(f"Unsupported semantic {field_name} choices: {selected}")
    return selected


def validate_semantic_search(
    semantic_profile: Mapping[str, object], 
    search_space_overrides: Mapping[str, object] | None = None
) -> None:
    """Reject undeclared recipes and unreachable or inconsistent semantic domains."""

    overrides = dict(search_space_overrides or {})
    allowed = set(SEARCH_SPACE)
    # Whole native recipes are a categorical choice only when a catalog was sealed.
    if "recipe_candidates" in semantic_profile:
        allowed.add("base_recipe")
    # A declared tier replaces the five independently sampled cost controls.
    if "phase_budget_candidates" in semantic_profile:
        allowed.add("semantic_budget")
        coupled_overrides = set(PHASE_BUDGET_FIELDS).intersection(overrides)
        # Independent changes would invalidate a coupled tier's identity.
        if coupled_overrides:
            raise ValueError(f"Coupled semantic budget fields cannot be overridden independently: {sorted(coupled_overrides)}")
        _choices("semantic_budget", overrides, semantic_profile)
    # Undeclared dimensions cannot be ignored or applied to a different recipe.
    if set(overrides) - allowed:
        raise ValueError(f"Unknown semantic search dimensions: {sorted(set(overrides) - allowed)}")
    # All available native recipes must support the declared semantic domain.
    if "recipe_candidates" in semantic_profile:
        domains = None
        for candidate_id in _choices("base_recipe", overrides, semantic_profile):
            candidate_profile = _recipe_profile(semantic_profile, candidate_id)
            candidate_profile.pop("recipe_candidates")
            candidate_overrides = {key: value for key, value in overrides.items() if key != "base_recipe"}
            validate_semantic_search(candidate_profile, candidate_overrides)
            horizon_fields = ("acquisition_noise_level", "ce_noise_level") if "phase_budget_candidates" in semantic_profile \
                else ("acquisition_noise_level", "ce_noise_level", "noise_levels")
            candidate_domains = {
                name: _choices(name, candidate_overrides, candidate_profile)
                for name in horizon_fields
            }
            # Optuna requires the same categorical domain for every draw of one name.
            if domains is not None and candidate_domains != domains:
                raise ValueError("Semantic base recipes need identical noise search choices; restrict them to the shared horizon.")
            domains = candidate_domains
        return
    choices = {
        name: _choices(name, overrides, semantic_profile)
        for name, distribution in SEARCH_SPACE.items() if isinstance(distribution, list)
    }
    # Reliability activity depends on actual tier noise sets, not unused defaults.
    if "phase_budget_candidates" in semantic_profile:
        budgets = [
            semantic_profile["phase_budget_candidates"][name]
            for name in _choices("semantic_budget", overrides, semantic_profile)
        ]
        choices["noise_levels"] = [",".join(str(level) for level in budget["noise_levels"]) for budget in budgets]
    settings = semantic_profile["route_settings"]
    # A fixed CE acquisition ablation has no orthogonality term to tune.
    if settings["acquisition_objective"] != "contrastive" and "orthogonality_weight" in overrides:
        raise ValueError("orthogonality_weight is inactive for true_class_ce acquisition.")
    # Disabled augmentation cannot consume an augmentation-view search restriction.
    if "tmcl" not in choices["image_augmentation"] and "augmentation_views" in overrides:
        raise ValueError("augmentation_views is inactive when image_augmentation is always none.")
    noisy = any(any(int(level) > 0 for level in item.split(",")) for item in choices["noise_levels"])
    # Clean-only alignment has unit reliability regardless of either policy.
    if not noisy and {"reliability", "reliability_floor"}.intersection(overrides):
        raise ValueError("Reliability controls are inactive for clean-only consolidation.")
    # A uniform weighting policy never reads the clipping floor.
    if "alpha_bar" not in choices["reliability"] and "reliability_floor" in overrides:
        raise ValueError("reliability_floor is inactive for uniform reliability.")
    dataset = semantic_profile["student_config"]["dataset"]
    # Every requested augmentation choice must fit the fixed native image geometry.
    if "tmcl" in choices["image_augmentation"] and dataset["pad"] != 0:
        raise ValueError("TMCL augmentation search requires the fixed native unpadded CIFAR geometry.")
    model = semantic_profile["student_config"]["model"]
    raw = model["kwargs"] if model["name"] is not None and model["kwargs"] else model["dit_classifier"]
    # Requested augmentation cannot change the fixed model's spatial/channel geometry.
    if "tmcl" in choices["image_augmentation"] and (raw.get("image_size", 32) != 32 or raw.get("channels", 3) != 3):
        raise ValueError("TMCL augmentation search requires the fixed native 32x32 RGB geometry.")
    for name in ("alignment_weight", "ce_weight"):
        specification = overrides.get(name, {})
        # A zero contribution removes a core objective and belongs to a separate ablation.
        if isinstance(specification, Mapping) and specification.get("low", SEARCH_SPACE[name]["low"]) == 0:
            raise ValueError(f"Semantic HPO requires an active {name}; zero-loss controls are separate ablations.")


def baseline_hints(search_space_overrides: Mapping[str, object] | None = None) -> list[dict]:
    """Leave baseline controls to separately paired runs, outside optimization."""

    return []


def _sample_settings(trial: Any, profile: Mapping[str, object], overrides: Mapping[str, object]) -> RouteSettings:
    """Sample only active native semantic fields, keeping fixed ablations intact."""

    from common.hpo import _TrialView


    suggestions = _TrialView(trial, overrides=overrides)
    values = deepcopy(profile["route_settings"])
    budget_fields = set()
    # One Optuna draw selects all coupled update/view/batch controls.
    if "phase_budget_candidates" in profile:
        budget_id = suggestions.suggest_categorical("semantic_budget", _choices("semantic_budget", overrides, profile))
        values.update(deepcopy(profile["phase_budget_candidates"][budget_id]))
        budget_fields = set(PHASE_BUDGET_FIELDS)
    for name in (
        "acquisition_steps", "consolidation_steps", "batch_size", "acquisition_noise_level", 
        "ce_noise_level", "noise_levels", "image_augmentation"
    ):
        # A tier-owned field cannot also become an independent suggestion.
        if name not in budget_fields:
            values[name] = suggestions.suggest_categorical(name, _choices(name, overrides, profile))
    # Legacy categorical strings resolve to the same native sequence as tier values.
    if isinstance(values["noise_levels"], str):
        values["noise_levels"] = [int(value) for value in values["noise_levels"].split(",")]
    for name in (
        "learning_rate", "temperature", "alignment_weight", "ce_weight", 
        "gain_limit", "bias_limit", "modulation_init_std"
    ):
        values[name] = suggestions.suggest_float(name, **SEARCH_SPACE[name])
    # Orthogonality contributes only to the native contrastive gate objective.
    if values["acquisition_objective"] == "contrastive":
        values["orthogonality_weight"] = suggestions.suggest_float("orthogonality_weight", **SEARCH_SPACE["orthogonality_weight"])
    # Independent image-view count is active only in the TMCL policy.
    if values["image_augmentation"] == "tmcl" and "augmentation_views" not in budget_fields:
        values["augmentation_views"] = suggestions.suggest_categorical("augmentation_views", _choices("augmentation_views", overrides, profile))
    # Clean consolidation has alpha_bar=1, so neither reliability control is sampled.
    if any(value > 0 for value in values["noise_levels"]):
        values["reliability"] = suggestions.suggest_categorical("reliability", _choices("reliability", overrides, profile))
        # Uniform weighting does not read reliability_floor.
        if values["reliability"] == "alpha_bar":
            values["reliability_floor"] = suggestions.suggest_categorical("reliability_floor", _choices("reliability_floor", overrides, profile))
    settings = RouteSettings(**values)
    # Both consolidation terms must remain active even for direct non-Optuna callers.
    if settings.alignment_weight == 0 or settings.ce_weight == 0:
        raise ValueError("Semantic HPO requires positive alignment and CE contributions.")
    return settings


def _fixed_projection(config: Config) -> dict:
    """Remove only operational destinations, seed controls and resolved schedule form."""

    result = deepcopy(asdict(config))
    result.pop("hpo", None)
    for key in ("seed", "results_path", "project_tag", "tensorboard", "tensorboard_path", "tensorboard_run_name"):
        result["training"].pop(key, None)
    for key in ("seed", "class_order", "task_groups", "class_order_mode", "task_order_mode", "checkpoint_dir"):
        result["continually_learn"].pop(key, None)
    result["model"].pop("show_network_summary", None)
    for key in ("kwargs", "wrapper_kwargs"):
        result["model"][key].pop("seed", None)
    result["model"]["diffusion_classifier"].pop("seed", None)
    return result


def reseed_semantic_config(config: Config, seed: int) -> None:
    """Change training RNGs without creating a sparse override of typed architecture."""

    config.training.seed = seed
    config.continually_learn.seed = seed
    for values in (config.model.kwargs, config.model.wrapper_kwargs):
        # Explicit generic seed fields follow the trial's initialization seed.
        if "seed" in values:
            values["seed"] = seed
    # A typed wrapper may supply an explicit seed without a generic kwargs mapping.
    if config.model.diffusion_classifier.seed is not None:
        config.model.diffusion_classifier.seed = seed
    # Prepared trial configs also retain the semantic phase RNG in their native settings.
    if "semantic_consolidation" in config.hpo:
        config.hpo["semantic_consolidation"]["seed"] = seed
    config.hpo["seed"] = seed


def _trial_digest(config: Config) -> str:
    """Bind sampled settings and parameters while allowing paired seed repeats."""

    settings = deepcopy(config.hpo["semantic_consolidation"])
    settings.pop("seed", None)
    return _json_digest({
        "settings": settings, "params": config.hpo["params"], 
        "search_space_overrides": config.hpo["semantic_search_space_overrides"], 
        "fixed_recipe_sha256": config.hpo["semantic_profile"]["fixed_recipe_sha256"]
    })


def build_semantic_config(
    trial: Any, 
    dataset_name: str, 
    epochs: int, 
    results_path: str | Path, 
    semantic_profile: Mapping[str, object], 
    search_space_overrides: Mapping[str, object] | None = None, 
    objective_metrics: object = None, 
    objective_directions: object = None, 
    use_ensemble_accuracy: bool | None = None, 
    ensemble_accuracy_kwargs: Mapping[str, object] | None = None, 
    max_train_samples: int | None = None, 
    max_val_samples: int | None = None, 
    dtype_policy: str | None = None, 
    deterministic_ops: bool | None = None, 
    seed: int = 42
) -> Config:
    """Build one semantic candidate from its complete sealed native CL recipe.

    Explicit global controls must agree with the supplied configuration. Result
    destinations, TensorBoard and random seeds are operational trial metadata.
    An optional catalog searches whole named architecture/teacher/replay/joint
    recipes; fields within a selected recipe remain immutable. Task budgets,
    validation protocol and reporting endpoint stay shared across the catalog.
    """

    from common.hpo import _TrialView, _normalize_objective_spec, _tensorboard_name


    profile = normalize_semantic_profile(semantic_profile, dataset_name, seed=seed)
    overrides = dict(search_space_overrides or {})
    validate_semantic_search(profile, overrides)
    selected_profile = profile
    # A recipe is sampled before semantic noise domains or native fields are resolved.
    if "recipe_candidates" in profile:
        candidate_id = _TrialView(trial, overrides=overrides).suggest_categorical(
            "base_recipe", _choices("base_recipe", overrides, profile)
        )
        selected_profile = _recipe_profile(profile, candidate_id)
    config = Config(**deepcopy(selected_profile["student_config"]))
    continual = config.continually_learn
    explicit = {
        "epochs": (epochs, config.training.epochs), 
        "use_ensemble_accuracy": (use_ensemble_accuracy, continual.use_ensemble_accuracy), 
        "ensemble_accuracy_kwargs": (ensemble_accuracy_kwargs, continual.ensemble_accuracy_kwargs), 
        "max_train_samples": (max_train_samples, config.dataset.max_train_samples), 
        "max_val_samples": (max_val_samples, config.dataset.max_val_samples), 
        "dtype_policy": (dtype_policy, config.training.dtype_policy), 
        "deterministic_ops": (deterministic_ops, config.training.deterministic_ops)
    }
    for key, pair in explicit.items():
        # Explicit caller controls cannot override a frozen ordinary-training value.
        if pair[0] is not None and pair[0] != pair[1]:
            raise ValueError(f"Semantic HPO freezes native {key}; requested {pair[0]!r}, native {pair[1]!r}.")
    settings = _sample_settings(trial, selected_profile, overrides)
    settings.seed = seed
    selection_metric = profile.get("selection_metric", "final_average_accuracy")
    metrics, directions = _normalize_objective_spec(
        "continual", [selection_metric] if objective_metrics is None else objective_metrics, 
        objective_directions, continual.use_ensemble_accuracy
    )
    # This method comparison retains one prespecified ordinary validation endpoint.
    if list(metrics) != [selection_metric] or list(directions) != ["maximize"]:
        raise ValueError(f"Semantic HPO requires {selection_metric} with direction maximize.")
    continual.class_order = deepcopy(profile["class_order"])
    continual.task_groups = deepcopy(profile["task_groups"])
    continual.class_order_mode = "fixed"
    continual.task_order_mode = "fixed"
    reseed_semantic_config(config, seed=seed)
    root = Path(results_path) / "continual" / "dit_classifier" / dataset_name / PROFILE
    tensorboard_name = _tensorboard_name(trial)
    config.training.results_path = str(root / "runs" / f"trial-{trial.number:04d}")
    config.training.project_tag = f"t{trial.number:04d}"
    config.training.tensorboard = True
    config.training.tensorboard_path = str(root / "tensorboard")
    config.training.tensorboard_run_name = tensorboard_name
    config.model.show_network_summary = False
    config.hpo.update({
        "search_profile": PROFILE, "profile_version": VERSION, "study_task": "continual", 
        "study_model": "dit_classifier", "model_family": "dit_classifier", 
        "trial_number": trial.number, "params": deepcopy(dict(trial.params)), 
        "tensorboard_name": tensorboard_name, "objective_metrics": list(metrics), 
        "objective_directions": list(directions), "objective_network": "raw", 
        "use_ensemble_accuracy": continual.use_ensemble_accuracy, 
        "ensemble_accuracy_kwargs": deepcopy(continual.ensemble_accuracy_kwargs), 
        "use_distillation": continual.use_distillation, "snapshot_network_name": continual.snapshot_network_name, 
        "continual_dataset_seed": profile["dataset_seed"], "semantic_profile": profile, 
        "semantic_consolidation": asdict(settings), "semantic_search_space_overrides": overrides, 
        "continual_schedule": {"class_num": continual.class_num, "class_order": profile["class_order"], "task_groups": profile["task_groups"], "task_size": continual.task_size}, 
        "fixed_recipe": {
            "architecture_search": "recipe_candidates" in profile, 
            "continual_search": "recipe_candidates" in profile, "semantic_search": True, 
            "test_set_used_for_hpo": False, "native_recipe_sha256": profile["fixed_recipe_sha256"]
        }, 
        "seed": seed, "dtype_policy": config.training.dtype_policy, 
        "deterministic_ops": config.training.deterministic_ops, "semantic_trial_recovery": "restart_whole_trial"
    })
    config.hpo["semantic_trial_sha256"] = _trial_digest(config)
    validate_semantic_config(config, profile)
    return config


def validate_semantic_config(config: Config, semantic_profile: Mapping[str, object] | None = None) -> None:
    """Authenticate immutable native settings and active semantic trial provenance."""

    profile = normalize_semantic_profile(config.hpo.get("semantic_profile"), config.dataset.name)
    # An external reference cannot conceal mutation of the catalog embedded in a trial.
    if semantic_profile is not None:
        supplied = normalize_semantic_profile(semantic_profile, config.dataset.name)
        # Caller references supplement rather than replace the embedded identity check.
        if supplied["fixed_recipe_sha256"] != profile["fixed_recipe_sha256"]:
            raise ValueError("Semantic trial uses a different sealed native recipe catalog.")
    selected_profile = profile
    # Catalog trials must name their exact native recipe before field comparison.
    if "recipe_candidates" in profile:
        selected_profile = _recipe_profile(profile, config.hpo.get("params", {}).get("base_recipe"))
        allowed_recipes = _choices("base_recipe", config.hpo["semantic_search_space_overrides"], profile)
        # Restricted studies cannot dispatch a recipe excluded from their search.
        if config.hpo["params"]["base_recipe"] not in allowed_recipes:
            raise ValueError("Semantic base_recipe is outside the sealed search restriction.")
    # Legacy profiles have no catalog selection parameter.
    elif "base_recipe" in config.hpo.get("params", {}):
        raise ValueError("Semantic base_recipe requires a sealed recipe catalog.")
    reference = Config(**deepcopy(selected_profile["student_config"]))
    # Every scientific common field stays fixed; only documented operational deltas are removed.
    if _fixed_projection(config) != _fixed_projection(reference):
        raise ValueError("Semantic trial changed the frozen native model, dataset, continual, training or optimizer recipe.")
    continual = config.continually_learn
    order, groups = resolve_continual_schedule(
        continual.class_num, continual.class_order, continual.task_groups, 
        available_class_num=10 if config.dataset.name == "cifar10" else 100, 
        task_size=continual.task_size, class_order_mode=continual.class_order_mode, 
        task_order_mode=continual.task_order_mode, seed=continual.seed
    )
    # Confirmations may change model seeds but never the task partition or data cohort.
    if order != profile["class_order"] or groups != profile["task_groups"] \
    or config.hpo.get("continual_dataset_seed") != profile["dataset_seed"]:
        raise ValueError("Semantic trial changed the sealed task schedule or dataset seed.")
    # The runtime dispatch must match the profile whose recipe was authenticated.
    if config.hpo.get("search_profile") != PROFILE or config.hpo.get("objective_network") != "raw" \
    or config.hpo.get("objective_metrics") != [profile.get("selection_metric", "final_average_accuracy")] \
    or config.hpo.get("objective_directions") != ["maximize"]:
        raise ValueError("Semantic trial requires the semantic profile and raw objective network.")
    # Saved trial settings must still agree with the exact sampled parameter evidence.
    if config.hpo.get("semantic_trial_sha256") != _trial_digest(config):
        raise ValueError("Semantic trial settings or parameter evidence changed.")
    settings = RouteSettings(**deepcopy(config.hpo["semantic_consolidation"]))
    # A stored tier identity binds its complete coupled intervention.
    if "phase_budget_candidates" in profile:
        budget_id = config.hpo.get("params", {}).get("semantic_budget")
        choices = _choices("semantic_budget", config.hpo["semantic_search_space_overrides"], profile)
        # Unknown or excluded tier IDs cannot masquerade as valid phase settings.
        if not isinstance(budget_id, str) or budget_id not in choices:
            raise ValueError(f"Unknown or excluded semantic_budget: {budget_id!r}.")
        for key, value in profile["phase_budget_candidates"][budget_id].items():
            # A tier is compared as one coherent treatment rather than independent knobs.
            if getattr(settings, key) != value:
                raise ValueError(f"Semantic trial changed the selected phase budget field {key}.")
    # Legacy independent searches cannot invent an unsealed tier dimension.
    elif "semantic_budget" in config.hpo.get("params", {}):
        raise ValueError("semantic_budget requires sealed phase_budget_candidates.")
    reference_settings = RouteSettings(**profile["route_settings"])
    for key, entry in FIELD_CATALOG.items():
        # Fixed intervention, diagnostic and extension settings cannot vary secretly by trial.
        if entry["role"].startswith("fixed") and key != "seed" and getattr(settings, key) != getattr(reference_settings, key):
            raise ValueError(f"Semantic trial changed the fixed route field {key}.")
    # Training and phase RNGs follow one trial seed; the validation split stays separate.
    if settings.seed != config.training.seed or continual.seed != config.training.seed:
        raise ValueError("Semantic trial training, continual and phase seeds must agree.")
    core_settings = deepcopy(settings)
    core_settings.extensions = {}
    core_settings.experimental = {}
    validate_route_config(RouteConfig(common=config, route=core_settings))


def _append_retention_metrics(config: Config, result: dict[str, object]) -> None:
    """Audit a complete validation matrix and add a distinct retention objective."""

    profile = config.hpo["semantic_profile"]
    task_count = len(profile["task_groups"])
    details = result.get("model", {}).get("continual_details", {})
    matrix = details.get("validation_accuracy_matrix")
    # The retention objective requires measured task evidence, not aggregate guesses.
    if matrix is None:
        raise ValueError("Retention selection requires the complete validation_accuracy_matrix.")
    # Convert array-like runtime output without importing or initializing a framework.
    rows = matrix.tolist() if hasattr(matrix, "tolist") else matrix
    # Missing tasks or ragged rows cannot define a complete continual stream.
    if not isinstance(rows, (list, tuple)) or len(rows) != task_count or any(
        not isinstance(row, (list, tuple)) or len(row) != task_count for row in rows
    ):
        raise ValueError("Retention selection requires a square matrix covering every sealed task.")
    audited = []
    for row_id, row in enumerate(rows):
        audited_row = []
        for column_id, value in enumerate(row):
            # Future task cells may be undefined; observed cells must never be imputed.
            if value is None and column_id > row_id:
                audited_row.append(None)
                continue
            # Truth values are not observed numeric accuracies.
            if isinstance(value, bool):
                raise ValueError("Validation accuracy entries must be numeric observed fractions.")
            try:
                observed = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError("Validation accuracy entries must be numeric observed fractions.") from error
            # Undefined future cells are serialized as null without imputing observations.
            if column_id > row_id and not math.isfinite(observed):
                audited_row.append(None)
                continue
            # Every observed task accuracy must remain a finite fraction.
            if not math.isfinite(observed) or not 0 <= observed <= 1:
                raise ValueError("Retention selection requires finite observed validation accuracy fractions.")
            audited_row.append(observed)
        audited.append(audited_row)
    final_row = audited[-1]
    metrics = result.get("evaluations", {}).get("validation_continual_metrics")
    # The new endpoint supplements the verified native mean instead of replacing it.
    if not isinstance(metrics, dict) or "final_average_accuracy" not in metrics:
        raise ValueError("Retention selection requires the native validation final_average_accuracy.")
    final_average = sum(final_row) / task_count
    native_average = float(metrics["final_average_accuracy"])
    # Floating-point representation permits only small numerical reduction differences.
    if not math.isfinite(native_average) or not math.isclose(native_average, final_average, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("Validation final_average_accuracy disagrees with the complete final matrix row.")
    old_accuracy = sum(final_row[:-1]) / (task_count - 1)
    new_accuracy = final_row[-1]
    balanced = 2 * old_accuracy * new_accuracy / (old_accuracy + new_accuracy) if old_accuracy + new_accuracy else 0.0
    additions = {
        "final_old_task_accuracy": old_accuracy, "final_new_task_accuracy": new_accuracy, 
        "retention_balanced_accuracy": balanced
    }
    for name, value in additions.items():
        # Existing differently defined metrics cannot be silently overwritten.
        if name in metrics and metrics[name] != value:
            raise ValueError(f"Existing validation metric {name} disagrees with its matrix definition.")
    metrics.update(additions)
    audit = {
        "accuracy_matrix_source": "validation_accuracy_matrix", "validation_accuracy_matrix": audited, 
        "definition": "Harmonic mean of final old-task mean accuracy and final newest-task accuracy; zero when both are zero.", 
        "validation_continual_metrics": {"final_average_accuracy": native_average, **additions}, 
        "task_groups": profile["task_groups"], "fixed_recipe_sha256": profile["fixed_recipe_sha256"], 
        "semantic_trial_sha256": config.hpo["semantic_trial_sha256"]
    }
    audit["audit_sha256"] = _json_digest(audit)
    destination = Path(result.get("results_path", config.training.results_path))
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "semantic_hpo_metrics.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_semantic_trial(config: Config) -> dict[str, object]:
    """Execute the authenticated semantic adapter after remote worker admission.

    Native run() owns full extension validation, semantic checkpoint/controller
    creation, route diagnostics and the standard validation report. A partial
    HPO attempt remains evidence; an explicitly reattempted trial starts fresh.
    """

    validate_semantic_config(config)
    from semantic_consolidation.runner import run


    result = run(RouteConfig(common=config, route=RouteSettings(**config.hpo["semantic_consolidation"])))
    # Legacy profiles retain their existing output contract unless the new endpoint is selected.
    if config.hpo["semantic_profile"].get("selection_metric") == "retention_balanced_accuracy":
        _append_retention_metrics(config, result)
    return result
