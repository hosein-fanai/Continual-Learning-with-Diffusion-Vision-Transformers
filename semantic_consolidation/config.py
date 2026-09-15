"""Validated route-one controls layered on the project's ordinary Config API."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
import math

import yaml

from common.config import Config, _safe_load_unique_yaml, load_config, resolve_continual_schedule


@dataclass
class RouteSettings:
    """Explicit phase budgets, loss coefficients, and mechanistic controls.

    ``noise_levels`` uses zero for a genuinely clean semantic view and positive
    values for diffusion schedule indices. Losses average examples and noise
    levels. The reliability multiplier applies only to semantic alignment.
    Modulation state is retained across tasks unless its removal is an explicit
    ablation. Steps count optimizer updates, independently of joint-fit epochs.
    ``consolidation_scope='semantic'`` permits only classifier projection/head
    and temporary predictor updates; the shared diffusion backbone stays fixed.
    Multiple noise levels pair the same corrupted input within each level, not
    features from different levels. ``probe_max_gates`` caps measured coverage,
    never the training bank.
    """

    acquisition_steps: int = 100
    consolidation_steps: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    temperature: float = 0.1
    alignment_weight: float = 1.0
    ce_weight: float = 1.0
    orthogonality_weight: float = 1.0
    gain_limit: float = 1.0
    bias_limit: float = 1.0
    modulation_init_std: float = 0.02
    acquisition_noise_level: int = 0
    ce_noise_level: int = 0
    noise_levels: tuple[int, ...] = (0,)
    reliability: str = "alpha_bar"
    reliability_floor: float = 0.05
    condition: str = "learned"
    extra_joint_seconds: tuple[float, ...] | None = None
    consolidation_scope: str = "semantic"
    acquisition_objective: str = "contrastive"
    retain_modulators: bool = True
    probe_batches: int = 4
    probe_max_gates: int = 16
    seed: int | None = None
    extensions: dict = field(default_factory=dict)
    experimental: dict = field(default_factory=dict)
    checkpoint_interval: int = 0

    def __post_init__(self) -> None:
        """Normalize sequence controls and reject invalid mathematical inputs.

        Returns:
            validated (None): None; numeric coefficients and optional sequences are normalized
                in place.

        Raises:
            TypeError: If extensions or experimental are not mappings.
            ValueError: If budgets, coefficients, seeds, views or treatment combinations are
                invalid.
        """

        # route.extensions must be a mapping.
        if not isinstance(self.extensions, Mapping):
            raise TypeError("route.extensions must be a mapping.")
        # route.experimental must be a mapping.
        if not isinstance(self.experimental, Mapping):
            raise TypeError("route.experimental must be a mapping.")
        for name in ("acquisition_steps", "consolidation_steps", "batch_size", "probe_batches", "probe_max_gates"):
            value = getattr(self, name)
            minimum = 0 if name.endswith("steps") else 1
            # Exact phase and probe budgets cannot be boolean or fractional counts.
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"route.{name} must be an integer >= {minimum}.")
        # Zero preserves completed-task recovery; positive intervals also commit fit progress.
        if type(self.checkpoint_interval) is not int or self.checkpoint_interval < 0:
            raise ValueError("route.checkpoint_interval must be a nonnegative integer.")
        # route.batch_size must be at least four for positives and negatives.
        if self.batch_size < 4:
            raise ValueError("route.batch_size must be at least four for positives and negatives.")
        # route.probe_max_gates must be at least two to cover old and new gates.
        if self.probe_max_gates < 2:
            raise ValueError("route.probe_max_gates must be at least two to cover old and new gates.")
        for name in (
            "learning_rate", "temperature", "gain_limit", "bias_limit",
            "alignment_weight", "ce_weight", "orthogonality_weight",
            "modulation_init_std", "reliability_floor",
        ):
            value = float(getattr(self, name))
            # Nonfinite or negative coefficients do not define the stated objectives.
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"route.{name} must be finite and nonnegative.")
            setattr(self, name, value)
        # route.learning_rate and route.temperature must be positive.
        if self.learning_rate == 0 or self.temperature == 0:
            raise ValueError("route.learning_rate and route.temperature must be positive.")
        # route.orthogonality_weight must be positive for contrastive separation.
        if self.orthogonality_weight == 0:
            raise ValueError("route.orthogonality_weight must be positive for contrastive separation.")
        # At least one of route.gain_limit and route.bias_limit must be positive.
        if self.gain_limit == 0 and self.bias_limit == 0:
            raise ValueError("At least one of route.gain_limit and route.bias_limit must be positive.")
        # route.reliability_floor must be in [0, 1].
        if self.reliability_floor > 1:
            raise ValueError("route.reliability_floor must be in [0, 1].")
        # route.noise_levels must be a nonempty sequence of integer indices.
        if isinstance(self.noise_levels, (str, bytes)) or not self.noise_levels:
            raise ValueError("route.noise_levels must be a nonempty sequence of integer indices.")
        self.noise_levels = tuple(self.noise_levels)
        # route.noise_levels must contain nonnegative integers.
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in self.noise_levels):
            raise ValueError("route.noise_levels must contain nonnegative integers.")
        # route.noise_levels must not contain duplicate views.
        if len(set(self.noise_levels)) != len(self.noise_levels):
            raise ValueError("route.noise_levels must not contain duplicate views.")
        for name in ("acquisition_noise_level", "ce_noise_level"):
            value = getattr(self, name)
            # Acquisition and CE views must identify discrete nonnegative schedule indices.
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"route.{name} must be a nonnegative integer.")
        choices = {
            "condition": {"baseline", "learned", "random", "no_consolidation", "feature_distillation", "unmodulated_feature_distillation", "extra_joint", "time_matched_joint"},
            "reliability": {"alpha_bar", "uniform"},
            "consolidation_scope": {"semantic", "backbone"},
            "acquisition_objective": {"contrastive", "true_class_ce"},
        }
        for name, values in choices.items():
            # Reject an unknown treatment name rather than silently choosing a default.
            if getattr(self, name) not in values:
                raise ValueError(f"route.{name} must be one of {sorted(values)}.")
        # Normalize explicitly supplied measured time budgets once at configuration time.
        if self.extra_joint_seconds is not None:
            # route.extra_joint_seconds must contain positive measured per-task durations.
            if isinstance(self.extra_joint_seconds, (str, bytes)) or not self.extra_joint_seconds:
                raise ValueError("route.extra_joint_seconds must contain positive measured per-task durations.")
            self.extra_joint_seconds = tuple(float(value) for value in self.extra_joint_seconds)
            # route.extra_joint_seconds values must be finite and positive.
            if any(not math.isfinite(value) or value <= 0 for value in self.extra_joint_seconds):
                raise ValueError("route.extra_joint_seconds values must be finite and positive.")
        # time_matched_joint requires measured per-task extra_joint_seconds.
        if self.condition == "time_matched_joint" and self.extra_joint_seconds is None:
            raise ValueError("time_matched_joint requires measured per-task extra_joint_seconds.")
        # route.retain_modulators must be boolean.
        if not isinstance(self.retain_modulators, bool):
            raise ValueError("route.retain_modulators must be boolean.")
        # route.seed must be an integer or null.
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError("route.seed must be an integer or null.")
        # route.seed must be in [0, 2**32).
        if self.seed is not None and not 0 <= self.seed < 2 ** 32:
            raise ValueError("route.seed must be in [0, 2**32).")
        # true_class_ce requires retaining the modulator for every seen class.
        if self.acquisition_objective == "true_class_ce" and not self.retain_modulators:
            raise ValueError("true_class_ce requires retaining the modulator for every seen class.")


@dataclass
class RouteConfig:
    """An ordinary common configuration and isolated semantic phase settings."""

    common: Config = field(default_factory=Config)
    route: RouteSettings = field(default_factory=RouteSettings)

    def __post_init__(self) -> None:
        """Resolve mapping inputs and require the two validated configuration components.

        Returns:
            validated (None): None; mapping components become Config and RouteSettings
                instances.

        Raises:
            TypeError: If components are incompatible or contain unknown fields.
            ValueError: If a constructed settings component fails validation.
        """
        # Resolve ordinary project mappings through the existing typed configuration.
        if isinstance(self.common, Mapping):
            self.common = Config(**self.common)
        # Resolve mechanism mappings without leaving unvalidated keyword dictionaries.
        if isinstance(self.route, Mapping):
            self.route = RouteSettings(**self.route)
        # RouteConfig requires common.Config and RouteSettings.
        if not isinstance(self.common, Config) or not isinstance(self.route, RouteSettings):
            raise TypeError("RouteConfig requires common.Config and RouteSettings.")


def _merge(base: dict, overrides: Mapping) -> dict:
    """Merge nested configuration mappings without changing either input.

    Args:
        base (dict): Original nested configuration dict, copied before applying overrides.
        overrides (Mapping): Mapping of nested overrides; scalars and sequences replace the
            corresponding base value.

    Returns:
        merged (dict): Independent deep copy with recursive mapping overrides.

    Raises:
        TypeError: If a value cannot be copied by deepcopy.
    """

    merged = deepcopy(base)
    for key, value in overrides.items():
        # Nested mappings retain unspecified base settings when overrides are applied.
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(dict(merged[key]), value)
        # Scalar and sequence overrides replace the corresponding base value.
        else:
            merged[key] = deepcopy(value)
    return merged


def load_route_config(path: str | Path) -> RouteConfig:
    """Load ``base_config`` plus optional ``common`` overrides and ``route``.

    A base file uses common.load_config; its relative path is resolved against
    this route file. Without a base, ``common`` is the ordinary Config mapping.
    Unknown keys and repeated explicit YAML keys fail before data or models
    are loaded. YAML merge keys retain their standard override precedence.

    Args:
        path (str | Path): Route YAML input path; a relative base_config resolves from this
            file.

    Returns:
        config (RouteConfig): Validated common and semantic configuration tree.

    Raises:
        OSError: If route/base YAML cannot be read.
        yaml.YAMLError: If YAML is malformed or repeats an explicit key.
        TypeError: If sections are not mappings or dataclass fields are unknown.
        ValueError: If top-level keys or the resolved experiment are unsupported.
    """

    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        data = _safe_load_unique_yaml(stream)
    # The route configuration root must be a mapping.
    if not isinstance(data, Mapping):
        raise TypeError("The route configuration root must be a mapping.")
    unknown = set(data) - {"base_config", "common", "route"}
    # Unknown top-level settings must not disappear during route loading.
    if unknown:
        raise ValueError(f"Unknown route configuration keys: {sorted(unknown)}.")
    common_data = data.get("common", {})
    # common must be a mapping of project configuration sections.
    if not isinstance(common_data, Mapping):
        raise TypeError("common must be a mapping of project configuration sections.")
    # Load an explicitly selected base before applying route-local common overrides.
    if data.get("base_config") is not None:
        base_path = Path(data["base_config"])
        # A relative base path belongs to its referring configuration directory.
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        common_data = _merge(asdict(load_config(base_path)), common_data)
    # Supply base_config or an explicit common configuration.
    elif "common" not in data:
        raise ValueError("Supply base_config or an explicit common configuration.")
    config = RouteConfig(common=Config(**common_data), route=data.get("route", {}))
    validate_route_config(config)
    return config


def validate_route_config(config: RouteConfig) -> None:
    """Validate the bounded V1 implementation before performing expensive work.

    Args:
        config (RouteConfig): Validated configuration tree, with the data, model and
            experiment settings consumed by this operation.

    Returns:
        validation (None): None; valid settings are normalized in place without constructing
            models or loading data.

    Raises:
        TypeError: If nested extension settings have invalid mapping types.
        ValueError: If model, precision, schedule, data, checkpoint or replay controls
            violate the supported protocol.
    """

    settings, project = config.route, config.common
    settings.__post_init__()
    # Validate requested held-out diagnostics and contextual-reference semantics.
    if settings.experimental:
        from semantic_consolidation.experimental import validate_experimental
        validate_experimental(project, settings.experimental, settings.condition)
    continual, training, dataset = project.continually_learn, project.training, project.dataset
    # Validate optional scheduling and replay-selection treatments before training.
    if settings.extensions:
        from semantic_consolidation.extensions import validate_extensions
        validate_extensions(project, settings.extensions)
        replay = settings.extensions.get("replay")
        # Contrastive semantic phases require two distinct candidate rows per class.
        if replay and settings.condition not in ("baseline", "extra_joint", "time_matched_joint") and (
            not replay.get("class_coverage", True) or replay.get("min_per_class", 1) < 2
        ):
            raise ValueError("Semantic phases require extension replay class_coverage=true and min_per_class>=2.")
    # Provide a common training or continual seed for reproducible route runs.
    if continual.seed is None and training.seed is None:
        raise ValueError("Provide a common training or continual seed for reproducible route runs.")
    # Route one requires training.task='continual' and fit_method='fit'.
    if training.task != "continual" or training.fit_method != "fit":
        raise ValueError("Route one requires training.task='continual' and fit_method='fit'.")
    # The audited route implementation currently requires float32.
    if training.dtype_policy != "float32":
        raise ValueError("The audited route implementation currently requires float32.")
    # Use patience=0 and optimizer_steps_per_epoch=null to retain complete task pools.
    if training.patience != 0 or continual.optimizer_steps_per_epoch is not None:
        raise ValueError("Use patience=0 and optimizer_steps_per_epoch=null to retain complete task pools.")
    # Route one requires a fresh complete joint phase; fit_kwargs steps_per_epoch/initial_epoch are
    # unsupported.
    if {"steps_per_epoch", "initial_epoch"}.intersection(training.fit_kwargs):
        raise ValueError("Route one requires a fresh complete joint phase; fit_kwargs steps_per_epoch/initial_epoch are unsupported.")
    # At least one joint-learning epoch is required per task.
    if training.epochs < 1:
        raise ValueError("At least one joint-learning epoch is required per task.")
    # Route one requires the attached joint classifier and no real replay buffer.
    if not continual.use_generative_model_classifier or continual.use_buffer:
        raise ValueError("Route one requires the attached joint classifier and no real replay buffer.")
    # Route one requires remove_prev_classes=true; cumulative historical real-data access is a separate
    # reference protocol.
    if not continual.remove_prev_classes:
        raise ValueError("Route one requires remove_prev_classes=true; cumulative historical real-data access is a separate reference protocol.")
    # The route's primary endpoint requires use_ensemble_accuracy=false; optional ensemble diagnostics may
    # still be evaluated.
    if continual.use_ensemble_accuracy:
        raise ValueError("The route's primary endpoint requires use_ensemble_accuracy=false; optional ensemble diagnostics may still be evaluated.")
    # Use route.condition controls and explicit common replay/KD switches, not named common baselines.
    if continual.baseline is not None:
        raise ValueError("Use route.condition controls and explicit common replay/KD switches, not named common baselines.")
    # Use replay_budget_mode='fixed_total' for explicit current/replay exposure budgets.
    if continual.replay_budget_mode != "fixed_total":
        raise ValueError("Use replay_budget_mode='fixed_total' for explicit current/replay exposure budgets.")
    # fixed_total requires a nonnegative replay_old_examples count.
    if continual.replay_old_examples is None or continual.replay_old_examples < 0:
        raise ValueError("fixed_total requires a nonnegative replay_old_examples count.")
    # Use raw images, sparse labels, and dataset.preprocess='fixed-standardize'.
    if dataset.return_features or dataset.onehot_labels or dataset.preprocess != "fixed-standardize":
        raise ValueError("Use raw images, sparse labels, and dataset.preprocess='fixed-standardize'.")
    # The audited route currently supports model.name='dit_classifier' only.
    if project.model.name not in (None, "dit_classifier"):
        raise ValueError("The audited route currently supports model.name='dit_classifier' only.")
    # The audited route requires the V1 diffusion_classifier wrapper.
    if project.model.wrapper_name not in (None, "diffusion_classifier"):
        raise ValueError("The audited route requires the V1 diffusion_classifier wrapper.")
    # Route one requires model.with_classifier=true.
    if not project.model.with_classifier:
        raise ValueError("Route one requires model.with_classifier=true.")
    # Training route runs start fresh; use load_inference_model for a completed checkpoint.
    if project.model.weights_path is not None:
        raise ValueError("Training route runs start fresh; use load_inference_model for a completed checkpoint.")
    raw = project.model.kwargs if project.model.name is not None and project.model.kwargs else asdict(project.model.dit_classifier)
    wrapper = project.model.wrapper_kwargs if project.model.name is not None and project.model.wrapper_kwargs else asdict(project.model.diffusion_classifier)
    # Route one currently requires diffusion_classifier.use_ema=false.
    if wrapper.get("use_ema", True):
        raise ValueError("Route one currently requires diffusion_classifier.use_ema=false.")
    # Require CFG and positive classifier_mlp_ratio for an unmodulated semantic projection.
    if not raw.get("use_cfg", True) or not raw.get("classifier_mlp_ratio"):
        raise ValueError("Require CFG and positive classifier_mlp_ratio for an unmodulated semantic projection.")
    # classifier_mlp_ratio must be positive.
    if float(raw["classifier_mlp_ratio"]) <= 0:
        raise ValueError("classifier_mlp_ratio must be positive.")
    levels = (*settings.noise_levels, settings.acquisition_noise_level, settings.ce_noise_level)
    # Every semantic noise level must be below the diffusion timestep count.
    if max(levels) >= int(raw.get("timesteps", 1000)):
        raise ValueError("Every semantic noise level must be below the diffusion timestep count.")
    for prefix in ("", "clf_"):
        reshapers = raw.get(f"{prefix}reshaper_ids_dict") or {}
        options = raw.get(f"{prefix}reshaper_kwargs")
        # Classifier reshaping inherits the shared options when no separate mapping exists.
        if options is None:
            options = raw.get("reshaper_kwargs") or {}
        # Stochastic variational flattening is unsupported: frozen semantic targets must be deterministic.
        if "flatten" in reshapers.values() and options.get("add_kl", False):
            raise ValueError("Stochastic variational flattening is unsupported: frozen semantic targets must be deterministic.")
    # The primary endpoint requires clean test bounds: min=0 and max=0.
    if wrapper.get("test_noisified_min_timesteps", 0) != 0 or wrapper.get("test_noisified_max_timesteps", -1) not in (None, 0):
        raise ValueError("The primary endpoint requires clean test bounds: min=0 and max=0.")
    available = {"mnist": 10, "fmnist": 10, "cifar10": 10, "cifar100": 100}.get(dataset.name)
    _, groups = resolve_continual_schedule(
        continual.class_num, continual.class_order, continual.task_groups,
        available_class_num=available, task_size=continual.task_size,
        class_order_mode=continual.class_order_mode,
        task_order_mode=continual.task_order_mode,
        seed=continual.seed if continual.seed is not None else training.seed,
    )
    # Every acquisition task must contain at least two classes.
    if any(len(group) < 2 for group in groups):
        raise ValueError("Every acquisition task must contain at least two classes.")
    # Retained old gates need usable positive pairs throughout later semantic phases.
    if len(groups) > 1 and settings.retain_modulators and settings.condition not in (
        "baseline", "extra_joint", "time_matched_joint"
    ):
        # Retained old modulators require generated old-class replay in later semantic phases.
        if not continual.use_generative_replay:
            raise ValueError("Retained old modulators require generated old-class replay in later semantic phases.")
        required = 2 * (sum(map(len, groups)) - len(groups[-1]))
        # Semantic positive pairs require replay_old_examples >= two rows per old class in the largest
        # task.
        if continual.replay_old_examples < required:
            raise ValueError("Semantic positive pairs require replay_old_examples >= two rows per old class in the largest task.")
    # route.extra_joint_seconds must contain exactly one measured budget per task.
    if settings.extra_joint_seconds is not None and len(settings.extra_joint_seconds) != len(groups):
        raise ValueError("route.extra_joint_seconds must contain exactly one measured budget per task.")


def save_route_settings(settings: RouteSettings, path: str | Path) -> None:
    """Persist every route coefficient and control using ordinary safe YAML.

    Args:
        settings (RouteSettings): Validated settings instance for this component; its fields
            select the behavior described above.
        path (str | Path): YAML output path; its parent directory must already exist.

    Returns:
        saved (None): None; all settings are written as ordinary safe YAML values.

    Raises:
        OSError: If the destination cannot be written.
    """

    data = asdict(settings)
    data["noise_levels"] = list(settings.noise_levels)
    # Serialize optional time budgets as ordinary YAML sequences.
    if settings.extra_joint_seconds is not None:
        data["extra_joint_seconds"] = list(settings.extra_joint_seconds)
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=True)
