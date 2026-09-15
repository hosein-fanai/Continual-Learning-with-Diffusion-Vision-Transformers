"""Integrate review section 10 at the two existing routes' sample/fit boundaries.

Common owns data loading, candidate generation, class growth, the optimizer,
losses and the previous-task teacher. This module retains only the transient
candidate pool that common already generates, then selects after wake updates.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

from common.keras_compat import optimizer_iterations
from common.runtime import derive_seed
from semantic_consolidation.evaluation import EnsembleEvaluationSettings, evaluate_checkpoint
from semantic_consolidation.scheduling import ScheduleSettings, execute_schedule, split_task_pool


def validate_extensions(project: object, values: Mapping) -> None:
    """Validate explicit extension settings without changing the common protocol.

    Args:
        project (object): Common Config instance containing the dataset, training, model and
            continual sections.
        values (Mapping): Mapping of schedule, replay, evaluation and optional allocation
            controls validated against the common protocol.

    Returns:
        validated (None): None; validates extension/common interactions without creating
            models or changing the protocol.

    Raises:
        TypeError: If a nested component is not a mapping.
        ValueError: If schedule, replay, evaluation, teacher, budget or callback
            combinations are unsupported.
    """

    allowed = {"schedule", "replay", "evaluation", "reselect_each_wake", "quality_fit_split", "kd_allocation"}
    # Unknown keys usually mean an intended intervention would silently be lost.
    if not isinstance(values, Mapping) or set(values) - allowed:
        raise ValueError(f"route.extensions accepts only {sorted(allowed)}.")
    for name in ("schedule", "replay", "evaluation", "kd_allocation"):
        # Every optional component is an ordinary configuration mapping.
        if not isinstance(values.get(name, {}), Mapping):
            raise TypeError(f"extensions.{name} must be a mapping.")
    schedule = ScheduleSettings(**values.get("schedule", {}))
    evaluation = EnsembleEvaluationSettings(**values.get("evaluation", {}))
    # Schedule/selection interactions must have an explicit interpretation.
    if not isinstance(values.get("reselect_each_wake", False), bool):
        raise ValueError("extensions.reselect_each_wake must be boolean.")
    # Quality calibration may use permitted training or held-out validation only.
    if values.get("quality_fit_split", "training") not in ("training", "validation"):
        raise ValueError("extensions.quality_fit_split must be training or validation.")
    continual, training = project.continually_learn, project.training
    raw = project.model.kwargs or asdict(project.model.dit_classifier)
    horizon = int(raw.get("timesteps", 1000))
    # Optional KD allocation is a separate experiment over scheduled replay fits.
    if values.get("kd_allocation"):
        from allocation_study.weighting import AllocationSettings

        allocation = AllocationSettings(**values["kd_allocation"])
        # Explicitly disabled settings retain ordinary validation and execution.
        if allocation.condition != "disabled":
            wrapper = project.model.wrapper_kwargs or asdict(project.model.diffusion_classifier)
            token = raw.get("clf_distil_token_type") if raw.get("classifier_only_distil_token", True) else raw.get("distil_token_type")
            # An inactive or missing independent KD objective cannot test allocation.
            if not token or float(wrapper.get("clf_distil_loss_coef", 0.)) <= 0 or not continual.use_distillation:
                raise ValueError("KD allocation requires active independent classifier KD and a previous-task teacher.")
            # The initial intervention varies weights alone under fixed replay timing.
            if schedule.mode != "fixed" or not schedule.replay_updates:
                raise ValueError("KD allocation requires a fixed schedule with positive replay updates.")
    # Both routes use the same raw network for the clean primary endpoint.
    if evaluation.enabled and (evaluation.network_name != "raw" or max(evaluation.horizons) > horizon):
        raise ValueError("Route ensemble evaluation needs raw weights and horizons <= diffusion timesteps.")
    # Evaluation must have a held-out split available before model construction.
    if evaluation.enabled and (not training.use_valset or project.dataset.validation_ratio <= 0):
        raise ValueError("Extension checkpoint evaluation requires held-out validation data.")
    # A cached pool bypasses sample(), so it cannot authenticate captured candidates.
    if values.get("replay") and (continual.replay_cache_mode != "off" or continual.replay_cache_dir):
        raise ValueError("Section-10 replay capture requires common replay caching off.")
    if schedule.mode != "joint":
        # Old real images would invalidate fallback provenance and the memory protocol.
        if not continual.remove_prev_classes or not continual.keep_same_model:
            raise ValueError("Explicit scheduling requires remove_prev_classes and keep_same_model.")
        # Callback early stopping or custom fit overrides would break exact budgets.
        if training.fit_kwargs or training.patience != 0:
            raise ValueError("Explicit scheduling owns fit budgets; use empty fit_kwargs and patience=0.")
        # Resampling by the outer learner would discard the intended finite pool.
        if continual.generative_model_kwargs.get("train_num", -1) != -1:
            raise ValueError("Explicit scheduling requires generative_model_kwargs.train_num=-1.")
        # A positive historical phase needs both replay data and its frozen teacher.
        if schedule.replay_updates and (not continual.use_generative_replay or not continual.use_distillation
                                        or not continual.replay_old_examples):
            raise ValueError("Positive replay phases require generative replay, prior-task distillation and an old-row budget.")
    # Replay ranking is an explicit optional component.
    if values.get("replay"):
        from semantic_consolidation.replay_selection import ReplaySelectionSettings

        replay = ReplaySelectionSettings(**values["replay"])
        # Validation-derived thresholds need a real held-out source before training.
        if values.get("quality_fit_split", "training") == "validation" and replay.quality_threshold is None and (
            not training.use_valset or not 0 < project.dataset.validation_ratio < 1
        ):
            raise ValueError("Validation quality fitting requires held-out validation data.")
        # The retained-count contract uses the shared fixed total budget.
        if continual.replay_budget_mode != "fixed_total":
            raise ValueError("Section-10 candidate accounting requires common fixed_total replay.")
        # Selection must follow actual current-data updates against the old teacher.
        if schedule.mode == "joint" or not schedule.replay_updates:
            raise ValueError("Drift replay selection requires an explicit wake/replay schedule.")
        # Generation and post-wake selection must describe the same candidate count.
        if continual.replay_candidate_multiplier != replay.candidate_multiplier:
            raise ValueError("common replay_candidate_multiplier must equal extensions.replay.candidate_multiplier.")
        # Teacher-only common ranking is superseded by the later extension selection.
        if continual.replay_selection not in ("all", "uniform", "random"):
            raise ValueError("Use common all/uniform/random selection; extension ranking runs after wake.")
        # Every requested noisy view must exist in the diffusion schedule.
        if max(replay.noise_levels) >= horizon:
            raise ValueError("Replay scoring noise levels must be below the diffusion horizon.")
        # Reversible virtual updates require deterministic training and prepared tensors.
        if replay.strategy == "mir":
            wrapper = project.model.wrapper_kwargs or asdict(project.model.diffusion_classifier)
            # Dropout random draws cannot be restored by copying model variables.
            if any(float(raw.get(name, 0.) or 0.) > 0 for name in ("dropout_rate", "drop_prob", "clf_drop_prob")):
                raise ValueError("Reversible MIR requires zero dropout_rate, drop_prob and clf_drop_prob.")
            # The inherited train step must consume the already prepared virtual batch.
            if not wrapper.get("map_preprocess", False) and not any(
                float(wrapper.get(name, 0.) or 0.) > 0 for name in ("noise_distil_loss_coef", "clf_distil_loss_coef")
            ):
                raise ValueError("MIR requires mapped preprocessing: enable map_preprocess or an existing teacher-KD loss.")
        # A fixed retained pool must represent all old classes. Smaller training
        # batches visit that pool cyclically; the lower-level selector also exposes
        # rotating selection windows for other explicitly managed protocols.
        if replay.class_coverage:
            from common.config import resolve_continual_schedule
            order, groups = resolve_continual_schedule(
                continual.class_num, continual.class_order, continual.task_groups,
                task_size=continual.task_size, class_order_mode=continual.class_order_mode,
                task_order_mode=continual.task_order_mode,
                seed=continual.seed if continual.seed is not None else training.seed,
            )
            # A frozen retained pool cannot cover more classes than it contains rows.
            if continual.replay_old_examples < replay.min_per_class * (len(order) - len(groups[-1])):
                raise ValueError("Fixed-pool class coverage requires replay_old_examples >= min_per_class * the largest old-class count.")
        # Sampling callbacks would mix unrelated images into the captured task pool.
        if training.show_images or training.save_gifs or training.report_every_epoch:
            raise ValueError("Disable automatic image/report callbacks with replay candidate capture.")
    # Drift-driven timing needs the same explicit replay scorer as its probe.
    elif schedule.mode == "drift":
        raise ValueError("The drift schedule requires extensions.replay for its matched-view probe.")


def _validation_arrays(wrapper: object, dataset: object) -> tuple[np.ndarray, np.ndarray]:
    """Materialize the supplied held-out split and map labels via the existing API.

    Args:
        wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
            schedules and existing training or inference APIs.
        dataset (object): Finite batched tf.data.Dataset containing raw images, sparse
            original labels and optional binary replay provenance.

    Returns:
        arrays (tuple[np.ndarray, np.ndarray]): (images, labels): float32 image ndarray and
            dense int64 labels from the supplied validation rows.

    Raises:
        ValueError: If validation is absent, empty or unbounded.
        KeyError: If a validation label is unmapped.
    """

    # Materializing unbounded or absent validation is outside this evaluator.
    if dataset is None or int(tf.data.experimental.cardinality(dataset).numpy()) < 0:
        raise ValueError("A known finite validation dataset is required.")
    images, labels = [], []
    for batch in dataset.as_numpy_iterator():
        images.append(np.asarray(batch[0], dtype=np.float32))
        labels.append(np.asarray([wrapper.seen_classes[int(label)]
                                  for label in np.asarray(batch[1]).reshape(-1)], dtype=np.int64))
    # Calibration and inference metrics are undefined on an empty split.
    if not images:
        raise ValueError("Validation data must not be empty.")
    return np.concatenate(images), np.concatenate(labels)


def _digest(network: object) -> str:
    """Reuse the route's weight invariant implementation without a copied model.

    Args:
        network (object): Raw classifier network exposing predict_class and the existing
            primary classifier layers.

    Returns:
        digest (str): SHA-256 string from the existing ordered weight-digest implementation.

    Raises:
        AttributeError: If network weights do not expose eager values.
    """

    from semantic_consolidation.controller import weight_digest

    return weight_digest(network.weights)


class ExtensionController:
    """Task-local candidate capture, post-wake selection and checkpoint evaluation."""

    def __init__(self, project: object, values: Mapping, seed: int) -> None:
        """Retain validated settings; no persistent image memory is introduced.

        Args:
            project (object): Common Config instance containing the dataset, training, model and
                continual sections.
            values (Mapping): Validated extension mappings for schedule, replay, evaluation and
                optional allocation.
            seed (int): Explicit integer random seed; local or derived streams preserve
                reproducibility without reseeding caller-owned generators.

        Returns:
            initialized (None): None; stores validated settings and initializes empty
                task/candidate records.

        Raises:
            TypeError: If nested settings are not mappings.
            ValueError: If extension/common protocol settings are incompatible.
        """

        validate_extensions(project, values)
        self.settings = dict(values)
        self.schedule = ScheduleSettings(**values.get("schedule", {}))
        self.evaluation = EnsembleEvaluationSettings(**values.get("evaluation", {}))
        self.seed = seed
        self.budget = int(project.continually_learn.replay_old_examples or 0)
        self.records: list[dict] = []
        self.candidates: list[tuple[np.ndarray, np.ndarray]] = []
        self.accepting_candidates = True
        self.pending: dict | None = None

    def capture(self, wrapper: object, result: object, labels: object) -> None:
        """Copy only the old-label candidates generated between complete task fits.

        Args:
            wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
                schedules and existing training or inference APIs.
            result (object): Generated sample output from the existing wrapper API, normally
                float32 NHWC pixels in [0, 1].
            labels (object): Integer CFG class-condition IDs [N] generated between completed
                task fits.

        Returns:
            captured (None): None; appends exact old candidate occurrences rescaled from [0, 1]
                to [-1, 1], or ignores inactive/auxiliary calls.

        Raises:
            ValueError: If condition IDs are not old classes or sample arrays are
                malformed/nonfinite.
        """

        # Ignore ordinary auxiliary generation and active-fit callbacks.
        if not self.accepting_candidates or not self.settings.get("replay") or labels is None:
            return
        ids = np.asarray(labels).reshape(-1) - int(wrapper.use_cfg)
        teacher = wrapper.teacher_network
        # The first experience has no historical candidate population.
        if teacher is None or not len(ids):
            return
        # Capture accepts only class conditions the previous teacher can represent.
        if np.any(ids < 0) or np.any(ids >= int(teacher.num_classes)):
            raise ValueError("Captured replay conditions must be in the prior teacher's vocabulary.")
        images = np.asarray(result, dtype=np.float32)
        # Reject trajectories or malformed samples before copying candidate pixels.
        if images.ndim != 4 or len(images) != len(ids) or not np.isfinite(images).all():
            raise ValueError("Candidate capture expects finite NHWC samples from the existing [0,1] sample API.")
        self.candidates.append((images * 2. - 1., ids.astype(np.int64)))

    def fit(self, wrapper: object, dataset: object, kwargs: dict, fit_function: object) -> tuple:
        """Execute common joint fitting or the explicit section-10 update schedule.

        Args:
            wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
                schedules and existing training or inference APIs.
            dataset (object): Finite batched tf.data.Dataset containing raw images, sparse
                original labels and optional binary replay provenance.
            kwargs (dict): Keyword arguments forwarded to the existing fit or constructor API,
                subject to the documented phase controls.
            fit_function (object): Existing bound fit callable; when None the scheduler calls
                the ordinary DiffusionClassifier fit implementation.

        Returns:
            fit_result (tuple): (history, effective_dataset): ordinary Keras History and the
                final finite pool passed to semantic phases.

        Raises:
            RuntimeError: If required candidates were not captured, update budgets fail or the
                prior teacher changes.
            ValueError: If supplied pools, quality screening or replay replacement violate the
                declared protocol.
        """

        self.accepting_candidates = False
        task = len(self.records) + 1
        teacher = wrapper.teacher_network
        teacher_digest = _digest(teacher) if teacher is not None else None
        before_iterations = int(optimizer_iterations(wrapper.optimizer).numpy())
        self.pending = {"task": task, "candidate_pool_bytes": 0, "candidate_count": 0}
        selector_callback, drift_probe, finish_probe = None, None, None
        # A first task has no historical teacher or replay candidates.
        if self.settings.get("replay") and self.candidates:
            selector_callback, drift_probe, finish_probe = self._selection_callbacks(wrapper, dataset, kwargs, task)
        # Replay configured after class growth must have passed through capture.
        elif self.settings.get("replay") and teacher is not None and int(teacher.num_classes) < int(wrapper.network.num_classes):
            raise RuntimeError("No captured old candidates: common sampling was bypassed or the task lifecycle is inconsistent.")
        try:
            allocation_context = nullcontext(None)
            # Omitted/disabled allocation does not install runtime state or score.
            allocation_values = self.settings.get("kd_allocation", {})
            # An enabled study runtime exists only around the scheduled fits.
            if allocation_values and allocation_values.get("condition", "disabled") != "disabled":
                from allocation_study.weighting import AllocationSettings, allocation_run

                allocation_context = allocation_run(
                    wrapper, AllocationSettings(**allocation_values),
                    seed=derive_seed(self.seed, "allocation_task", task),
                )
            with allocation_context as allocation:
                history, schedule_audit, effective = execute_schedule(
                    wrapper, dataset, kwargs, self.schedule, fit_function=fit_function,
                    replay_selector=selector_callback, drift_probe=drift_probe,
                    seed=derive_seed(self.seed, "section10_schedule", task),
                )
                # Snapshot graph-side diagnostics before releasing temporary state.
                if allocation is not None:
                    self.pending["kd_allocation"] = allocation.audit()
            self.pending["schedule"] = schedule_audit
            # Measure recoverability after scheduled replay without updating the learner.
            if finish_probe is not None:
                self.pending["recovery_probe"] = finish_probe(wrapper)
            self.pending["joint_optimizer_updates"] = int(optimizer_iterations(wrapper.optimizer).numpy()) - before_iterations
            self.pending["prior_teacher_unchanged"] = teacher is None or _digest(teacher) == teacher_digest
            # The previous-task target must remain physically unchanged during fitting.
            if not self.pending["prior_teacher_unchanged"]:
                raise RuntimeError("Wake/replay scheduling changed the frozen prior-task teacher.")
            return history, effective
        finally:
            # Full candidate pixels live for one task only, outside the gist byte cap.
            self.candidates.clear()

    def _selection_callbacks(self, wrapper: object, dataset: object, kwargs: dict, task: int) -> tuple:
        """Fit the quality rule before wake and build stateful post-wake selectors.

        Args:
            wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
                schedules and existing training or inference APIs.
            dataset (object): Finite batched tf.data.Dataset containing raw images, sparse
                original labels and optional binary replay provenance.
            kwargs (dict): Keyword arguments forwarded to the existing fit or constructor API,
                subject to the documented phase controls.
            task (int): One-based integer task index used for derived streams and exported
                records.

        Returns:
            callbacks (tuple): (select, probe, finish): task-local callables sharing candidate
                and fixed quality-threshold state.

        Raises:
            ValueError: If captured count, held-out quality inputs or prior-teacher scoring are
                invalid.
        """

        from semantic_consolidation.replay_selection import (
            DriftReplaySelector, ReplaySelectionSettings, virtual_update_interference,
            prepare_virtual_current_batch,
        )

        settings = ReplaySelectionSettings(**self.settings["replay"])
        selector = DriftReplaySelector(settings, seed=derive_seed(self.seed, "section10_replay", task))
        images = np.concatenate([item[0] for item in self.candidates])
        labels = np.concatenate([item[1] for item in self.candidates])
        # Concatenation briefly coexists with captured chunks, then releases them.
        self.candidates.clear()
        expected = self.budget * settings.candidate_multiplier
        # Candidate generation must match the fixed common count exactly.
        if len(images) != expected:
            raise ValueError(f"Captured {len(images)} candidates; the declared common replay budget requires {expected}.")
        self.pending.update(candidate_count=len(images), candidate_pool_bytes=int(images.nbytes + labels.nbytes),
                            candidate_concatenation_array_peak_bytes=int(2 * (images.nbytes + labels.nbytes)),
                            candidate_sha256=hashlib.sha256(images.tobytes() + labels.tobytes()).hexdigest())
        fit_split = self.settings.get("quality_fit_split", "training")
        quality_x, quality_y = images, labels
        # Fit validation thresholds on old classes from the supplied held-out split.
        if fit_split == "validation":
            quality_x, quality_y = _validation_arrays(wrapper, kwargs.get("validation_data"))
            old = quality_y < int(wrapper.teacher_network.num_classes)
            quality_x, quality_y = quality_x[old], quality_y[old]
        quality_scores = selector.score(wrapper, quality_x, quality_y) if settings.quality_threshold is None else None
        self.pending["quality_fit"] = selector.fit_quality_threshold(
            wrapper, quality_x, quality_y, split=fit_split, scored=quality_scores,
        )
        # Count threshold-fitting forwards separately from post-wake ranking.
        if quality_scores is not None:
            self.pending["quality_fit"]["scoring"] = quality_scores["diagnostics"]
        # Expanded heads can already invade old predictions before any wake update.
        initial_scores = quality_scores if fit_split == "training" and quality_scores is not None else selector.score(wrapper, images, labels)
        self.pending["pre_wake_candidate_drift"] = {
            "mean_js": float(np.mean(initial_scores["drift"])),
            "mean_new_class_invasion": float(np.mean(initial_scores["new_class_invasion"])),
            "scoring_reused_from_quality_fit": initial_scores is quality_scores,
            "scoring": initial_scores["diagnostics"],
        }
        state = {"selected": None, "scored": None, "wake": None, "selection_scores": None, "ever_selected": set()}
        old_classes = list(range(int(wrapper.teacher_network.num_classes)))
        inverse = {int(value): int(key) for key, value in wrapper.seen_classes.items()}
        pool = split_task_pool(wrapper, dataset)

        def select(current: object, replay_x: np.ndarray, replay_y: np.ndarray, wake: int) -> tuple:
            """Rank once after initial wake, or refresh at each declared wake boundary.

            Args:
                current (object): Live post-wake wrapper to score against its unchanged previous-
                    task teacher.
                replay_x (np.ndarray): Current retained replay float32 image array [N, H, W, C].
                replay_y (np.ndarray): Aligned replay labels in the wrapper-facing original-label
                    convention.
                wake (int): Integer number of completed current-only optimizer updates at this
                    selection boundary.

            Returns:
                selected (tuple): (images, labels, audit): float32 retained pixels, original-label
                    integer vector and selection/cost dict.

            Raises:
                ValueError: If strict quality/class quotas cannot meet the budget or MIR
                    preconditions fail.
                RuntimeError: If virtual optimizer state cannot be restored.
            """

            started = time.perf_counter()
            state["scored"] = selector.score(current, images, labels)
            state["wake"] = wake
            # Retaining the first selected pool isolates timing from pool refreshes.
            if state["selected"] is not None and not self.settings.get("reselect_each_wake", False):
                selected_x, selected_y, audit = state["selected"]
                return selected_x, selected_y, {"reused_selection": True,
                                               "scoring": state["scored"]["diagnostics"],
                                               "scoring_seconds": time.perf_counter() - started}
            interference = None
            interference_audit = None
            if settings.strategy == "mir":
                # The scorer owns a reversible virtual update, never a lasting fit.
                count = min(self.schedule.batch_size, len(pool["current_labels"]))
                batch = (tf.convert_to_tensor(pool["current_images"][:count]),
                         tf.convert_to_tensor(pool["current_labels"][:count]),
                         tf.zeros((count,), tf.bool))
                preparation_started = time.perf_counter()
                prepared = prepare_virtual_current_batch(current, batch)
                preparation_seconds = time.perf_counter() - preparation_started
                interference, interference_audit = virtual_update_interference(
                    current, prepared, selector, images, labels, before_scores=state["scored"],
                )
                interference_audit["virtual_batch_policy"] = "first bounded current-pool rows; prospective current update, not a claim to preview the scheduler's next batch"
                interference_audit["preparation_seconds"] = preparation_seconds
                interference_audit["preparation_scope"] = "Inherited noising/CFG and teacher-target preprocessing; its teacher forwards are additional to candidate-scoring counts."
            selected_x, selected_y, audit = selector.select(
                images, labels, self.budget, old_classes, scored=state["scored"], interference=interference,
            )
            selected_y = np.asarray([inverse[int(label)] for label in selected_y], dtype=replay_y.dtype)
            audit["after_wake_updates"] = wake
            # Virtual work is charged even though its optimizer update is restored.
            if interference_audit is not None:
                audit["virtual_interference"] = interference_audit
            state["selected"] = selected_x, selected_y, audit
            state["selection_scores"] = state["scored"]
            state["ever_selected"].update(audit["selected_indices"])
            return selected_x, selected_y, audit

        def probe(current: object, replay_x: np.ndarray, replay_y: np.ndarray) -> dict:
            """Use every fixed candidate so selected-pool changes cannot move the probe.

            Args:
                current (object): Post-wake wrapper; cached selection scores already refer to this
                    checkpoint.
                replay_x (np.ndarray): Current retained replay float32 image array [N, H, W, C].
                replay_y (np.ndarray): Aligned replay labels in the wrapper-facing original-label
                    convention.

            Returns:
                drift (dict): Dict of Python float full-candidate JS/invasion summaries with
                    scoring-reuse metadata.

            Raises:
                TypeError: If called before select has established the post-wake score mapping.
            """

            scored = state["scored"]
            return {"mean_js": float(np.mean(scored["drift"])),
                    "mean_js_change_from_pre_wake": float(np.mean(scored["drift"]) - np.mean(initial_scores["drift"])),
                    "mean_new_class_invasion": float(np.mean(scored["new_class_invasion"])),
                    "candidate_count": len(labels), "score_reused_from_selection": True}

        def finish(current: object) -> dict:
            """Measure internal recoverability on the same candidates after replay.

            Args:
                current (object): Live wrapper after all scheduled replay updates and before
                    semantic phases.

            Returns:
                recovery (dict): Dict of selected/unselected candidate loss and teacher-agreement
                    changes with independent scoring cost.

            Raises:
                ValueError: If the final checkpoint cannot produce valid matched-view candidate
                    scores.
            """

            final_scores = selector.score(current, images, labels)
            selected = np.zeros(len(labels), dtype=bool)
            selected[list(state["ever_selected"])] = True
            result = {"scope": "after_schedule_before_route_specific_phases",
                      "interpretation": "Conditioning-label loss and teacher agreement are internal proxies, not independent image quality or causal efficacy.",
                      "scoring": final_scores["diagnostics"], "groups": {}}
            for name, mask in (("ever_selected", selected), ("never_selected", ~selected)):
                # Empty selected/rejected groups have no defined mean recovery statistic.
                if np.any(mask):
                    reference = state["selection_scores"]
                    result["groups"][name] = {
                        "examples": int(np.sum(mask)),
                        "conditioning_nll_before_last_selection": float(np.mean(reference["student_label_loss"][mask])),
                        "conditioning_nll_after_schedule": float(np.mean(final_scores["student_label_loss"][mask])),
                        "conditioning_nll_reduction": float(np.mean(reference["student_label_loss"][mask] - final_scores["student_label_loss"][mask])),
                        "mean_js_after_schedule": float(np.mean(final_scores["drift"][mask])),
                        "mean_new_class_invasion_after_schedule": float(np.mean(final_scores["new_class_invasion"][mask])),
                    }
            return result

        return select, probe, finish

    def after_task(self, wrapper: object, validation_data: object) -> None:
        """Compare inference treatments after all route-specific learning has ended.

        Args:
            wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
                schedules and existing training or inference APIs.
            validation_data (object): Finite held-out validation dataset with sparse original
                labels; never a source of gradient updates.

        Returns:
            recorded (None): None; evaluates enabled held-out variants, commits the pending task
                record and reopens candidate capture.

        Raises:
            ValueError: If requested held-out data or inference settings are invalid.
            RuntimeError: If inference changes the completed model weights.
        """

        # Inference reporting is only valid after the corresponding task fit.
        if self.pending is None:
            raise RuntimeError("A completed extension fit must precede checkpoint evaluation.")
        # Optional ensembles evaluate the same completed raw checkpoint.
        if self.evaluation.enabled:
            images, labels = _validation_arrays(wrapper, validation_data)
            before = _digest(wrapper.network)
            observer = getattr(wrapper, "experimental_controller", None)
            self.pending["evaluation"] = evaluate_checkpoint(wrapper, images, labels, self.evaluation, split="validation",
                old_class_count=observer.old_count if observer is not None else None)
            self.pending["evaluation_weights_unchanged"] = _digest(wrapper.network) == before
            # The previous-task target must remain physically unchanged during fitting.
            if not self.pending["evaluation_weights_unchanged"]:
                raise RuntimeError("Inference evaluation changed the completed checkpoint.")
        self.records.append(self.pending)
        self.pending = None
        self.accepting_candidates = True

    def save(self, directory: str | Path) -> None:
        """Export independent work ledgers; pixels and teachers are never serialized.

        Args:
            directory (str | Path): Output directory for this operation, resolved using ordinary
                pathlib path semantics.

        Returns:
            saved (None): None; writes extension JSON and optional ensemble-evaluation CSV, then
                releases candidate capture.

        Raises:
            OSError: If output artifacts cannot be written.
        """

        self.accepting_candidates = False
        self.candidates.clear()
        path = Path(directory)
        from semantic_consolidation.controller import _json_value
        with (path / "extensions.json").open("w", encoding="utf-8") as stream:
            json.dump(_json_value({"settings": self.settings, "tasks": self.records,
                                  "resource_scope": "Candidate pixels are transient and separate from persistent replay storage; schedule times are already included in common training time."}),
                      stream, indent=2, sort_keys=True, allow_nan=False)
        rows = []
        for record in self.records:
            for variant in record.get("evaluation", {}).get("variants", []):
                row = {"task": record["task"], "split": record["evaluation"]["split"], "variant": variant["name"],
                       **variant["metrics"], **variant["evaluation_cost"]}
                # Scalar fitting and its classifier work remain distinct from evaluation.
                if variant["temperature_fit"] is not None:
                    row["temperature"] = variant["temperature_fit"]["temperature"]
                    row.update({f"calibrated_{key}": value for key, value in variant["calibrated_metrics"].items()})
                    row.update({f"calibration_{key}": value for key, value in variant["calibration_cost"].items()})
                rows.append(row)
        # Write a comparison table only when inference variants were requested.
        if rows:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            with (path / "ensemble_evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
