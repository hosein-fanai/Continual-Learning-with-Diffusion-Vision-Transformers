"""Section 11 observation at existing route boundaries; no training objective.

All online diagnostics use held-out validation, never test examples. The common
learner owns locked-test evaluation and accuracy matrices. Observations are
reported with their own costs and do not drive optimization or model selection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import csv
import json
from pathlib import Path
import threading
import time

import numpy as np
import tensorflow as tf

from common.keras_compat import optimizer_iterations

def validate_experimental(project: object, values: Mapping[str, object], condition: str | None=None) -> None:
    """Fail before training on ambiguous diagnostics or reference labels.

    Args:
        project (object): Common Config instance containing the dataset, training, model and
            continual sections.
        values (Mapping[str, object]): Mapping of optional
            enabled/probe/generation/learning-curve/reference controls.
        condition (str | None): Optional route condition name; contextual references require
            baseline.

    Returns:
        validated (None): None; accepted observer settings leave all training parameters
            unchanged.

    Raises:
        ValueError: If fields, sample budgets, held-out data or contextual-reference
            settings are invalid.
    """
    allowed = {"enabled", "probe_per_class", "generation_per_class", "batch_size",
               "ece_bins", "feature_extractor", "learning_curves", "reference"}
    # Unknown observer settings cannot silently change or disable the intended measurement.
    if not isinstance(values, Mapping) or set(values) - allowed:
        raise ValueError(f"route.experimental accepts only {sorted(allowed)}.")
    for name in ("enabled", "learning_curves"):
        # Require explicit booleans for optional observation switches.
        if not isinstance(values.get(name, True), bool):
            raise ValueError(f"experimental.{name} must be boolean.")
    for name, default in (("probe_per_class", 8), ("generation_per_class", 8),
                          ("batch_size", 32), ("ece_bins", 15)):
        value = values.get(name, default)
        # Exact diagnostic sample budgets must be positive integer counts.
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"experimental.{name} must be a positive integer.")
        # experimental.generation_per_class must be at least two for the unbiased estimator.
        if name == "generation_per_class" and value < 2:
            raise ValueError("experimental.generation_per_class must be at least two for the unbiased estimator.")
    # Runtime feature comparison uses fixed_pixels; custom frozen extractors use the explicit diagnostic
    # API.
    if values.get("feature_extractor", "fixed_pixels") != "fixed_pixels":
        raise ValueError("Runtime feature comparison uses fixed_pixels; custom frozen extractors use the explicit diagnostic API.")
    # Section 11 online diagnostics require held-out validation.
    if values.get("enabled", False) and (not project.training.use_valset or project.dataset.validation_ratio <= 0):
        raise ValueError("Section 11 online diagnostics require held-out validation.")
    reference = values.get("reference")
    # experimental.reference must be clean_finetune, offline_joint or null.
    if reference not in (None, "clean_finetune", "offline_joint"):
        raise ValueError("experimental.reference must be clean_finetune, offline_joint or null.")
    # Ordinary observations do not impose the separate contextual-reference restrictions.
    if reference is None:
        return
    # Contextual references require route.condition=baseline.
    if condition != "baseline":
        raise ValueError("Contextual references require route.condition=baseline.")
    continual = project.continually_learn
    # Contextual references do not use replay or KD.
    if continual.use_distillation or continual.use_generative_replay:
        raise ValueError("Contextual references do not use replay or KD.")
    # Offline joint training belongs to its separate reference runner.
    if reference == "offline_joint":
        raise ValueError("Offline joint training uses a separate reference runner, outside the continual route runner.")
    # The remaining contextual reference is clean current-only supervised fine-tuning.
    else:
        wrapper = project.model.wrapper_kwargs or asdict(project.model.diffusion_classifier)
        # clean_finetune requires positive primary CE and no ensemble training loss.
        if wrapper.get("use_ensemble_loss_instead", False) or not np.isfinite(float(wrapper.get("clf_loss_coef", 1.))) or float(wrapper.get("clf_loss_coef", 1.)) <= 0:
            raise ValueError("clean_finetune requires positive primary CE and no ensemble training loss.")
        for name in ("noise_loss_coef", "noise_distil_loss_coef", "clf_distil_loss_coef",
                     "image_loss_coef", "kl_loss_coef", "ctr_loss_coef"):
            # clean_finetune requires only the supervised primary classifier objective.
            if float(wrapper.get(name, 0.)) != 0.:
                raise ValueError("clean_finetune requires only the supervised primary classifier objective.")
        # clean_finetune requires genuinely clean training inputs.
        if wrapper.get("train_noisified_min_timesteps", 0) != 0 or wrapper.get("train_noisified_max_timesteps") != 0:
            raise ValueError("clean_finetune requires genuinely clean training inputs.")
        # clean_finetune requires unconditional current-only training.
        if wrapper.get("p_uncond") != 1. or not continual.remove_prev_classes:
            raise ValueError("clean_finetune requires unconditional current-only training.")


class MemoryMonitor:
    """Sample process RSS, and query the TensorFlow allocator's actual high water.

    RSS sampling can miss short peaks. TensorFlow peaks cover its allocator,
    not total device occupancy. Neither is inferred from tensor payload sizes.
    """

    def __init__(self) -> None:
        """Initialize optional process sampling and supported GPU allocator peak counters.

        Returns:
            initialized (None): None; optional RSS sampling and supported GPU allocator counters
                are prepared.

        Raises:
            None: Missing psutil and unsupported GPU allocator counters are treated as
                unavailable measurements.
        """
        self.peak_rss = None
        self.samples = 0
        self.stop_event = threading.Event()
        self.thread = None
        self.process = None
        self.devices = []
        try:
            import psutil
            self.process = psutil.Process()
        except ImportError:
            pass
        for device in tf.config.list_logical_devices("GPU"):
            name = device.name.split("device:")[-1]
            try:
                tf.config.experimental.reset_memory_stats(name)
                self.devices.append(name)
            except (ValueError, RuntimeError):
                pass

    def start(self) -> None:
        """Start asynchronous RSS sampling when process measurements are available.

        Returns:
            started (None): None; starts a daemon RSS sampler when psutil is available.

        Raises:
            RuntimeError: If the operating system cannot start the sampler thread.
        """
        # RSS sampling is optional when the process-measurement dependency is unavailable.
        if self.process is not None:
            self.thread = threading.Thread(target=self._poll, daemon=True)
            self.thread.start()

    def _poll(self) -> None:
        """Sample process RSS until cancellation without blocking the training thread.

        Returns:
            stopped (None): None when cancelled or process RSS becomes unavailable; updates peak
                RSS and sample count while active.

        Raises:
            None: Process-read OSError ends sampling without disrupting training.
        """
        while not self.stop_event.is_set():
            try:
                rss = int(self.process.memory_info().rss)
                self.peak_rss = max(self.peak_rss or 0, rss)
                self.samples += 1
            except OSError:
                return
            self.stop_event.wait(0.05)

    def snapshot(self, include_previous: bool = True) -> dict[str, object]:
        """Return observed process and device peaks with their measurement scope.

        Args:
            include_previous (bool): Include separately retained measurements from
                prior process lifetimes after recovery; False returns only this
                process's sample and allocator observations.

        Returns:
            memory (dict[str, object]): Dict of integer sampled RSS/allocator bytes, counts and
                scope; unavailable values are None.

        Raises:
            None: Unsupported device-memory queries are recorded as unavailable.
        """
        devices = {}
        for name in self.devices:
            try:
                devices[name] = dict(tf.config.experimental.get_memory_info(name))
            except (ValueError, RuntimeError):
                devices[name] = None
        result = {"sampled_process_peak_rss_bytes": self.peak_rss, "rss_samples": self.samples,
                "rss_sampling_seconds": 0.05, "tf_allocator_devices": devices,
                "scope": "Cumulative since observer creation; RSS sampled, TF allocator high water only; not total device occupancy."}
        # Previous process peaks are retained as observations, not current allocator state.
        if include_previous and getattr(self, "previous_segments", None):
            result["previous_process_segments"] = self.previous_segments
        return result

    def close(self) -> None:
        """Signal and join the optional sampler so its thread does not outlive the run.

        Returns:
            closed (None): None; requests sampler cancellation and waits at most one second for
                its thread.

        Raises:
            RuntimeError: If called from the sampler thread itself, which cannot join itself.
        """
        self.stop_event.set()
        # Only a running sampler thread needs to be joined during cleanup.
        if self.thread is not None:
            self.thread.join(timeout=1.)


def classification_outcomes(probabilities: object, labels: object, old_count: int, bins: int) -> dict[str, object]:
    """Use common calibrated-probability metrics and expose old/new confusions.

    Args:
        probabilities (object): Finite nonnegative prediction matrix [N, C] with unit row
            mass, aligned with sparse labels when supplied.
        labels (object): Sparse integer label vector aligned with the image rows; the label
            convention for this operation is described above.
        old_count (int): Integer boundary K in [0, number of seen classes] for old/new
            outcome decomposition.
        bins (int): Positive integer number of equal-width confidence bins used for ECE.

    Returns:
        outcomes (dict[str, object]): JSON-compatible dict of float calibration/accuracy,
            integer confusion counts and old/new decomposition.

    Raises:
        ValueError: If probabilities, labels, old-class boundary or bin count are invalid.
    """
    from common.mechanistic import calibration_metrics
    from semantic_consolidation.evaluation import _targets
    probabilities = np.asarray(probabilities)
    # Probabilities require a sample and class axis.
    if probabilities.ndim != 2:
        raise ValueError("Probabilities require a sample and class axis.")
    labels = _targets(labels, len(probabilities), probabilities.shape[1])
    # old_count must be an integer in the seen-class range.
    if isinstance(old_count, bool) or not isinstance(old_count, (int, np.integer)) or not 0 <= old_count <= probabilities.shape[1]:
        raise ValueError("old_count must be an integer in the seen-class range.")
    # bins must be a positive integer.
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer.")
    result = calibration_metrics(probabilities, labels, bins=bins)
    predicted = probabilities.argmax(axis=1)
    classes = probabilities.shape[1]
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (labels, predicted), 1)
    counts = matrix.sum(axis=1)
    result.update({"per_class_recall": [float(matrix[k, k] / counts[k]) if counts[k] else None
                                         for k in range(classes)],
                   "class_counts": counts.tolist(), "confusion_matrix": matrix.tolist(),
                   "confusion_convention": "row=true dense class, column=predicted dense class",
                   "ece_protocol": {"bins": bins, "kind": "equal-width maximum-confidence", "temperature": 1.},
                   "old_class_count": old_count})
    for name, mask in (("old", labels < old_count), ("new", labels >= old_count)):
        result[f"{name}_accuracy"] = float(np.mean(predicted[mask] == labels[mask])) if mask.any() else None
    result["old_to_new_errors"] = int(matrix[:old_count, old_count:].sum())
    result["new_to_old_errors"] = int(matrix[old_count:, :old_count].sum())
    return result


class _LearningCurve(tf.keras.callbacks.Callback):
    """Observe clean held-out outcomes at existing epoch boundaries without optimization."""
    def __init__(self, observer: ExperimentalController, wrapper: object) -> None:
        """Bind the route observer and live wrapper for validation callbacks.

        Args:
            observer (ExperimentalController): ExperimentalController owning the fixed held-out
                cohort and observation records.
            wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
                schedules and existing training or inference APIs.

        Returns:
            initialized (None): None; stores the observer and live wrapper for epoch callbacks.

        Raises:
            None: Construction only attaches the already-created observer and wrapper.
        """
        super().__init__()
        self.observer, self.wrapper = observer, wrapper

    def on_epoch_end(self, epoch: int, logs: dict | None=None) -> None:
        """Record held-out accuracy and calibration against the completed optimizer count.

        Args:
            epoch (int): Zero-based integer index of the completed fit epoch.
            logs (dict | None): Optional Keras batch or epoch log mapping; this callback does
                not modify it.

        Returns:
            recorded (None): None; appends validation outcomes, optimizer count and cost to the
                observer curves.

        Raises:
            ValueError: If the observer lacks a valid held-out cohort or model probabilities are
                invalid.
        """
        observer = self.observer
        started = time.perf_counter()
        outcome, cost = observer._classify(self.wrapper, observer.validation[0], observer.validation[1])
        observer.curves.append({"task": len(observer.records) + 1, "observation": len(observer.curves),
                                "fit_epoch": int(epoch), "optimizer_iterations": int(optimizer_iterations(self.wrapper.optimizer).numpy()),
                                "outcomes": outcome, "cost": cost,
                                "scope": "held-out validation after joint/scheduled fit epoch; route phase traces are separate"})
        observer.curve_seconds += time.perf_counter() - started


class ExperimentalController:
    """Bounded held-out diagnostics and measured resources for both route APIs."""

    def __init__(self, project: object, values: Mapping[str, object], seed: int, bundle: dict | None=None) -> None:
        """Configure bounded held-out cohorts, candidate reservoirs, and measured resources.

        Args:
            project (object): Common Config instance containing the dataset, training, model and
                continual sections.
            values (Mapping[str, object]): Validated mapping defining held-out sample caps,
                learning curves and optional reference metadata.
            seed (int): Explicit integer random seed; local or derived streams preserve
                reproducibility without reseeding caller-owned generators.
            bundle (dict | None): Optional common model bundle; retained classifier-template
                state is included in resource inventory.

        Returns:
            initialized (None): None; allocates fixed-probe metadata and starts optional
                resource monitoring.

        Raises:
            ValueError: If fixed-probe budgets or seed are invalid.
        """
        from semantic_consolidation.experimental_diagnostics import FixedHiddenProbe
        self.project, self.settings, self.seed = project, dict(values), seed
        self.verbose = getattr(getattr(project, "training", None), "verbose", False)
        self.bundle = bundle
        self.probe = FixedHiddenProbe(per_class=values.get("probe_per_class", 8),
                                     batch_size=values.get("batch_size", 32), seed=seed)
        self.records, self.curves, self.candidates, self.representatives = [], [], [], []
        self.generated_counts = {}
        self.candidate_rng = {}
        self.accepting_candidates = False
        self.old_count = 0
        self.validation = None
        self.curve_seconds = 0.
        self.sampling_seconds = 0.
        self.started = time.perf_counter()
        self.monitor = MemoryMonitor()
        self.monitor.start()

    def before_task(self, wrapper: object, validation_data: tf.data.Dataset, kwargs: dict) -> dict:
        """Map the supplied validation labels and append the optional observation callback.

        Args:
            wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
                schedules and existing training or inference APIs.
            validation_data (tf.data.Dataset): Finite held-out validation dataset with sparse
                original labels; never a source of gradient updates.
            kwargs (dict): Keyword arguments forwarded to the existing fit or constructor API,
                subject to the documented phase controls.

        Returns:
            fit_kwargs (dict): New dict preserving caller fit settings and appending the
                optional learning-curve callback.

        Raises:
            ValueError: If held-out validation is absent, empty or unbounded.
            KeyError: If a supplied label is absent from the wrapper mapping.
        """
        from semantic_consolidation.extensions import _validation_arrays
        self.verbose = kwargs.get(
            "verbose", getattr(getattr(self.project, "training", None), "verbose", False)
        )
        if self.verbose:
            print("Boundary diagnostics: preparing held-out validation data...", flush=True)
        self.validation = _validation_arrays(wrapper, validation_data)
        self.accepting_candidates = False
        self.fit_started = time.perf_counter()
        self.curve_seconds = 0.
        result = dict(kwargs)
        # Append learning-curve observation without removing existing fit callbacks.
        if self.settings.get("learning_curves", True):
            result["callbacks"] = list(kwargs.get("callbacks") or []) + [_LearningCurve(self, wrapper)]
        return result

    def _classify(self, wrapper: object, images: np.ndarray, labels: object, *,
                  verbose: bool | int | str = False) -> tuple[dict, dict]:
        """Evaluate the clean primary classifier and report old/new outcomes and work.

        Args:
            wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
                schedules and existing training or inference APIs.
            images (np.ndarray): Numeric sample-major images in the configured model-input
                scale, normally float32 NHWC values in [-1, 1].
            labels (object): Sparse integer label vector aligned with the image rows; the label
                convention for this operation is described above.
            verbose (bool | int | str): Keras-style progress verbosity for classifier batches;
                defaults to quiet for per-epoch learning-curve observations.

        Returns:
            observation (tuple[dict, dict]): (outcomes, cost): JSON-compatible dictionaries of
                clean old/new metrics and classifier work.

        Raises:
            ValueError: If images, probabilities or label support are invalid.
        """
        from semantic_consolidation.evaluation import EnsembleEvaluationSettings, _predict
        settings = EnsembleEvaluationSettings(batch_size=self.settings.get("batch_size", 32), seed=self.seed)
        probabilities, cost = _predict(wrapper, images, settings, None, "section11", verbose=verbose)
        return classification_outcomes(probabilities, labels, self.old_count,
                                       self.settings.get("ece_bins", 15)), cost

    def capture(self, wrapper: object, result: object, labels: object, seconds: float) -> None:
        """Keep a seeded uniform per-class reservoir of actual candidate occurrences.

        Args:
            wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
                schedules and existing training or inference APIs.
            result (object): Generated sample output from the existing wrapper API, normally
                float32 NHWC pixels in [0, 1].
            labels (object): Integer CFG class-condition IDs [N]; subtracting the null-label
                offset yields old dense class IDs.
            seconds (float): Actual elapsed duration of this sampling call, in seconds.

        Returns:
            captured (None): None; updates bounded per-class candidate reservoirs and elapsed
                sampling cost, or ignores calls outside capture windows.

        Raises:
            ValueError: If sample geometry/conditions are misaligned or a condition lies outside
                the old vocabulary.
        """
        from common.runtime import derive_seed
        # Ignore generation outside the replay-capture window or without class conditions.
        if labels is None or not self.accepting_candidates:
            return
        images = np.asarray(result, dtype=np.float32) * 2. - 1.
        dense = np.asarray(labels, dtype=np.int64).reshape(-1) - int(wrapper.use_cfg)
        # Section 11 replay capture requires aligned NHWC images and conditions.
        if len(images) != len(dense) or images.ndim != 4:
            raise ValueError("Section 11 replay capture requires aligned NHWC images and conditions.")
        cap = self.settings.get("generation_per_class", 8)
        for image, label in zip(images, dense):
            key = int(label)
            # Captured replay contains a class not available at the previous checkpoint.
            if key < 0 or key >= self.old_count:
                raise ValueError("Captured replay contains a class not available at the previous checkpoint.")
            self.generated_counts[key] = self.generated_counts.get(key, 0) + 1
            positions = [i for i, row in enumerate(self.candidates) if row[0] == key]
            # Each class gets its own reproducible reservoir stream independent of chunking.
            if key not in self.candidate_rng:
                self.candidate_rng[key] = np.random.default_rng(derive_seed(self.seed, "section11_reservoir", len(self.records), key))
            # Fill the reservoir before using probabilistic replacement.
            if len(positions) < cap:
                self.candidates.append((key, self.generated_counts[key], image.copy()))
            # Algorithm R gives later candidate occurrences the same inclusion probability.
            else:
                replacement = int(self.candidate_rng[key].integers(self.generated_counts[key]))
                # Replace a stored occurrence only when the uniform reservoir draw selects its slot.
                if replacement < cap:
                    self.candidates[positions[replacement]] = (key, self.generated_counts[key], image.copy())
        self.sampling_seconds += seconds

    def after_task(self, wrapper: object) -> None:
        """Capture completed-task predictions, hidden drift, replay diagnostics, and memory.

        Args:
            wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
                schedules and existing training or inference APIs.

        Returns:
            recorded (None): None; appends completed-task measurements, retains sampled audit
                arrays and resets task-local candidate state.

        Raises:
            ValueError: If held-out inputs, hidden features or generated-candidate diagnostics
                violate their contracts.
        """
        from semantic_consolidation.experimental_diagnostics import generated_memory_diagnostics
        from semantic_consolidation.evaluation import EnsembleEvaluationSettings, _predict
        from gist_memory.controller import tensor_inventory
        images, labels = self.validation
        fit_seconds = time.perf_counter() - self.fit_started
        started = time.perf_counter()
        if self.verbose:
            print(f"Boundary diagnostics (task {len(self.records) + 1}): "
                  f"validation evaluation on {len(images)} images", flush=True)
        outcome, cost = self._classify(wrapper, images, labels, verbose=self.verbose)
        if self.verbose:
            metrics = ", ".join(
                f"{name}={outcome[name]:.4f}" if outcome.get(name) is not None else f"{name}=unavailable"
                for name in ("accuracy", "old_accuracy", "new_accuracy", "ece")
            )
            print(f"Validation: {metrics}", flush=True)
            print("Boundary diagnostics: fixed hidden-feature probes", flush=True)
        inverse_labels = {dense: original for original, dense in wrapper.seen_classes.items()}
        hidden_labels = np.asarray([inverse_labels[int(label)] for label in labels], dtype=np.int64)
        hidden = self.probe.observe(wrapper, images, hidden_labels, len(self.records) + 1)
        generation = {"available": False, "reason": "No actual generated replay before this task."}
        # Generated-memory diagnostics require actual captured replay candidates.
        if self.candidates:
            if self.verbose:
                print(f"Boundary diagnostics: generated replay audit on {len(self.candidates)} images", flush=True)
            gx = np.stack([row[2] for row in self.candidates])
            gy = np.asarray([row[0] for row in self.candidates], dtype=np.int64)
            settings = EnsembleEvaluationSettings(batch_size=self.settings.get("batch_size", 32), seed=self.seed)
            probabilities, generation_cost = _predict(
                wrapper, gx, settings, None, "section11-generated", verbose=self.verbose
            )
            generation = generated_memory_diagnostics(gx, gy, list(range(self.old_count)),
                real_images=images, real_labels=labels, probabilities=probabilities,
                seed=self.seed, max_per_class=self.settings.get("generation_per_class", 8))
            generation.update({"available": True, "actual_candidate_counts": dict(self.generated_counts),
                               "selection": "uniform reservoir over candidate occurrences per class; sample metrics describe the audited subset",
                               "classifier": "current learner primary head; internal label consistency, not independent semantic labels",
                               "prediction_cost": generation_cost})
            self.representatives.append((len(self.records) + 1, gx, gy))
        elif self.verbose:
            print("Boundary diagnostics: generated replay audit unavailable (no captured replay)", flush=True)
        if self.verbose:
            print("Boundary diagnostics: resource accounting", flush=True)
        route = getattr(wrapper, "route_controller", None)
        memory = getattr(wrapper, "memory_controller", None)
        bank = getattr(route, "bank", None)
        optimizer = wrapper.optimizer.variables
        optimizer = optimizer() if callable(optimizer) else optimizer
        # Common retains its unused classifier template until the stream returns.
        # The shared attached head replaces it afterward; identities are deduplicated.
        companion = self.bundle.get("classifier") if self.bundle is not None else None
        companion_optimizer = getattr(getattr(companion, "optimizer", None), "variables", [])
        companion_optimizer = companion_optimizer() if callable(companion_optimizer) else companion_optimizer
        inventory = tensor_inventory({"raw": wrapper.network.weights,
            "ema": getattr(getattr(wrapper, "ema_network", None), "weights", []),
            "teacher": getattr(wrapper.teacher_network, "weights", []), "optimizer": optimizer,
            "encoder": getattr(getattr(memory, "scorer", None), "weights", []),
            "factory_classifier_template": getattr(companion, "weights", []),
            "factory_classifier_optimizer": companion_optimizer,
            "modulation": [v for pair in bank.vectors.values() for v in pair] if bank is not None else []})
        encoded = getattr(getattr(memory, "memory", None), "byte_size", 0)
        record = {"task": len(self.records) + 1, "split": "validation", "class_order": self.project.continually_learn.class_order,
                  "outcomes": outcome, "classification_cost": cost, "hidden": hidden, "generated_memory": generation,
                  "tensor_inventory": inventory, "encoded_replay_bytes": encoded,
                  "tensor_inventory_scope": "End of fit before common attaches the completed-task teacher; see post_boundary_teacher for its replacement.",
                  "retained_generated_audit_array_bytes": sum(x.nbytes + y.nbytes for _, x, y in self.representatives),
                  "validation_rows": len(images), "validation_array_bytes": images.nbytes + labels.nbytes,
                  "historical_validation_rows": int(np.sum(labels < self.old_count)),
                  "validation_policy": "retained seen-class validation from training split; never gradient data; fixed probe history additionally retained",
                  "resource_measurement": self.monitor.snapshot(),
                  "seconds": {"fit_including_route_and_online_evaluation": fit_seconds,
                              "learning_curve_evaluation": self.curve_seconds,
                              "generated_replay_sampling": self.sampling_seconds,
                              "boundary_diagnostics": time.perf_counter() - started}}
        self.records.append(record)
        if self.verbose:
            print(f"Boundary diagnostics complete in {record['seconds']['boundary_diagnostics']:.2f}s", flush=True)
        self.old_count = wrapper.network.num_classes
        self.candidates.clear()
        self.generated_counts.clear()
        self.candidate_rng.clear()
        self.sampling_seconds = 0.
        self.validation = None
        self.accepting_candidates = True

    def teacher_boundary(self, wrapper: object) -> None:
        """Record the actual full-width teacher after common's final snapshot.

        The old training teacher inventory and its replacement are distinct time
        points, not tensors to add together. Constructor and mid-fit setter calls
        do not create task observations.

        Args:
            wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
                schedules and existing training or inference APIs.

        Returns:
            recorded (None): None; appends the actual completed teacher inventory after common
                advances it; constructor/in-fit calls do nothing.

        Raises:
            TypeError: If a present teacher variable has an unsupported dtype or incomplete
                shape.
        """
        from gist_memory.controller import tensor_inventory
        # Constructor and in-fit teacher setter calls are not completed-task boundaries.
        if not self.records or not self.accepting_candidates:
            return
        teacher = wrapper.teacher_network
        self.records[-1]["post_boundary_teacher"] = {
            "class_count": getattr(teacher, "num_classes", 0),
            "inventory": tensor_inventory({"completed_teacher": getattr(teacher, "weights", [])}),
            "resource_measurement": self.monitor.snapshot(),
            "scope": "After the common completed-task teacher setter; replaces the end-of-fit prior teacher."}

    def save(self, directory: str | Path) -> None:
        """Export strict diagnostic JSON, sampled generations, and scalar learning curves.

        Args:
            directory (str | Path): Output directory for this operation, resolved using ordinary
                pathlib path semantics.

        Returns:
            saved (None): None; writes strict JSON, sampled-generation NPZ files and optional
                scalar learning-curve CSV.

        Raises:
            OSError: If the output directory or artifacts cannot be read/written.
        """
        from semantic_consolidation.controller import _json_value
        path = Path(directory)
        payload = {"settings": self.settings, "tasks": self.records,
                   "seconds_since_model_created": time.perf_counter() - self.started,
                   "memory": self.monitor.snapshot(),
                   "cost_scope": "Common task seconds already include fit, route and online diagnostics; do not add nested ledgers twice.",
                   "learning_curves": self.curves}
        payload["artifact_storage"] = {
            "files": {item.relative_to(path).as_posix(): item.stat().st_size
                      for item in sorted(path.rglob("*")) if item.is_file()},
            "scope": "Existing run artifacts at observer export; disk bytes are separate from live tensor payload and episodic byte cap."}
        (path / "section11.json").write_text(json.dumps(_json_value(payload), indent=2, allow_nan=False), encoding="utf-8")
        for task, images, labels in self.representatives:
            np.savez_compressed(path / f"generated_examples_task_{task:03d}.npz", images=images, labels=labels)
        rows = [{"task": row["task"], "observation": row["observation"], "fit_epoch": row["fit_epoch"],
                 "optimizer_iterations": row["optimizer_iterations"],
                 **{key: row["outcomes"][key] for key in ("accuracy", "old_accuracy", "new_accuracy", "nll", "ece")}}
                for row in self.curves]
        # Write scalar learning curves only when epoch observations were actually requested.
        if rows:
            with (path / "learning_curves.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def close(self) -> None:
        """Close candidate capture and release the background resource monitor.

        Returns:
            closed (None): None; closes capture and stops the monitor while retaining records
                for export.

        Raises:
            RuntimeError: If monitor cleanup attempts to join the current sampler thread.
        """
        self.accepting_candidates = False
        self.monitor.close()
