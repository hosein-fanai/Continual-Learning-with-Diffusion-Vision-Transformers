"""Budgeted current acquisition and replay phases using the existing fit API.

The drift policy changes *when* a fixed replay budget is spent. It never buys
extra updates after observing a large drift. This is a scheduling experiment,
not a reproduction of Wake-Sleep Consolidated Learning or a biological model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import math
import time

import numpy as np
import tensorflow as tf

from common.dataloader import get_dataset
from common.keras_compat import optimizer_iterations
from common.runtime import derive_seed


@dataclass
class ScheduleSettings:
    """Exact update and presentation budgets for one task's joint model.

    ``joint`` retains ordinary common fitting. Other modes replace its epoch
    budget with ``wake_updates`` current-only and ``replay_updates`` replay-only
    full batches. Task one has no historical replay and uses only wake updates.
    ``fixed`` inserts replay blocks after wake blocks; ``drift`` defers a block
    below the threshold and spends every remaining replay update after wake.
    ``interleaved`` alternates individual current/replay updates as a control.
    All three modes retain common's CE, diffusion, and KD losses unchanged.
    """

    mode: str = "joint"
    wake_updates: int = 10
    replay_updates: int = 10
    batch_size: int = 32
    wake_block_updates: int = 5
    replay_block_updates: int = 5
    drift_threshold: float = 0.05

    def __post_init__(self) -> None:
        """Reject undefined policies, fractional updates, and invalid JS thresholds.

        Returns:
            validated (None): None; normalizes the finite float JS threshold and checks integer
                budgets.

        Raises:
            ValueError: If the mode, update counts, batch/block sizes or threshold are invalid.
        """

        # A mode names one implemented data/update ordering only.
        if self.mode not in ("joint", "fixed", "drift", "interleaved"):
            raise ValueError("schedule.mode must be joint, fixed, drift, or interleaved.")
        for name in ("wake_updates", "replay_updates", "batch_size", "wake_block_updates", "replay_block_updates"):
            value = getattr(self, name)
            minimum = 0 if name == "replay_updates" else 1
            # Boolean and fractional budgets cannot identify an exact update count.
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"schedule.{name} must be an integer >= {minimum}.")
        # Boolean thresholds usually indicate a configuration type error.
        if isinstance(self.drift_threshold, bool):
            raise ValueError("schedule.drift_threshold must be a finite JS divergence in nats.")
        self.drift_threshold = float(self.drift_threshold)
        # Natural-log Jensen-Shannon divergence cannot exceed log(2).
        if not math.isfinite(self.drift_threshold) or not 0 <= self.drift_threshold <= math.log(2):
            raise ValueError("schedule.drift_threshold must lie in [0, log(2)] nats.")


class PhaseSchedule:
    """Small state machine that preserves budgets independently of drift.

    Call ``next_wake`` then ``after_wake`` until wake is exhausted. An observed
    divergence can affect replay timing only; the final boundary flushes the
    remainder. Values supplied to fixed controls are validated but not used.
    """

    def __init__(self, settings: ScheduleSettings, has_replay: bool) -> None:
        """Initialize one task's counters, with no replay allocation on task one.

        Args:
            settings (ScheduleSettings): Validated settings instance for this component; its
                fields select the behavior described above.
            has_replay (bool): Whether this task has historical replay; False assigns zero
                effective replay updates.

        Returns:
            initialized (None): None; starts zero completed counts and an effective replay
                budget of zero when has_replay is false.

        Raises:
            ValueError: If settings are invalid or mode is joint, which bypasses this state
                machine.
        """

        settings.__post_init__()
        # Ordinary joint fitting remains outside the explicit state machine.
        if settings.mode == "joint":
            raise ValueError("The ordinary joint mode does not use a phase schedule.")
        self.settings = settings
        self.wake_completed = 0
        self.replay_completed = 0
        self.replay_budget = settings.replay_updates if has_replay else 0
        self.awaiting_observation = False

    def next_wake(self) -> int:
        """Reserve the next current-only block; zero indicates completion.

        Returns:
            updates (int): Integer reserved current-only block size; zero means all wake updates
                have been allocated.

        Raises:
            RuntimeError: If the preceding wake block has not received its scheduling
                observation.
        """

        # A missing observation must not silently consume another wake block.
        if self.awaiting_observation:
            raise RuntimeError("Observe the preceding wake block before reserving another.")
        size = 1 if self.settings.mode == "interleaved" else self.settings.wake_block_updates
        updates = min(size, self.settings.wake_updates - self.wake_completed)
        self.wake_completed += updates
        self.awaiting_observation = bool(updates)
        return updates

    def after_wake(self, mean_js: float | None = None) -> tuple[int, str]:
        """Allocate replay after completed wake, flushing the final remainder.

        Args:
            mean_js (float | None): Optional finite mean Jensen-Shannon divergence in nats,
                bounded by log(2); required by adaptive decisions with replay remaining.

        Returns:
            allocation (tuple[int, str]): (updates, reason): integer replay allocation and
                explanatory string; the final wake flushes all remaining replay.

        Raises:
            RuntimeError: If no wake block is awaiting observation.
            ValueError: If JS is nonfinite/outside [0, log(2)] or missing for adaptive replay
                decisions.
        """

        # Each wake block has exactly one policy decision.
        if not self.awaiting_observation:
            raise RuntimeError("A wake block must precede its scheduling observation.")
        # Invalid measurements cannot select an adaptive learning treatment.
        if mean_js is not None and (
            not math.isfinite(float(mean_js)) or not 0 <= float(mean_js) <= math.log(2) + 1e-7
        ):
            raise ValueError("Scheduling requires a finite mean JS in [0, log(2)] nats.")
        remaining = self.replay_budget - self.replay_completed
        # Adaptive decisions require actual evidence while replay remains.
        if self.settings.mode == "drift" and remaining and mean_js is None:
            raise ValueError("The drift schedule requires a measured mean JS after each wake block.")
        final = self.wake_completed == self.settings.wake_updates
        # The first task and exhausted budgets need no additional replay update.
        if not remaining:
            updates, reason = 0, "no_replay_budget"
        # Spend deferred updates at the final boundary to preserve matched budgets.
        elif final:
            updates, reason = remaining, "final_budget_flush"
        # A small drift delays integration without deleting its reserved budget.
        elif self.settings.mode == "drift" and mean_js < self.settings.drift_threshold:
            updates, reason = 0, "below_drift_threshold"
        # Fixed boundaries and exceeded thresholds release the next replay block.
        else:
            size = 1 if self.settings.mode == "interleaved" else self.settings.replay_block_updates
            updates = min(size, remaining)
            reason = "drift_threshold" if self.settings.mode == "drift" else "fixed_boundary"
        self.replay_completed += updates
        self.awaiting_observation = False
        return updates, reason


def split_task_pool(wrapper: object, dataset: tf.data.Dataset) -> dict:
    """Read only the finite supplied pool and preserve its wrapper-facing labels.

    Explicit replay provenance is authoritative. Without metadata, the existing
    class-incremental protocol identifies replay by the previous teacher's dense
    vocabulary. This fallback assumes old real training rows are not retained;
    route configuration must enforce that assumption.

    Args:
        wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
            schedules and existing training or inference APIs.
        dataset (tf.data.Dataset): Finite batched tf.data.Dataset containing raw images,
            sparse original labels and optional binary replay provenance.

    Returns:
        pool (dict): Dict of image/label arrays for current and replay rows, preserving
            dtypes and recording provenance/old support.

    Raises:
        ValueError: If cardinality, batches, integer labels, binary provenance, old-class
            support or current-row availability are invalid.
    """

    # Unknown or infinite cardinality cannot define a bounded materialized pool.
    if not isinstance(dataset, tf.data.Dataset) or int(tf.data.experimental.cardinality(dataset).numpy()) < 0:
        raise ValueError("Phase scheduling requires a known finite tf.data.Dataset.")
    images, labels, flags = [], [], []
    has_metadata = None
    teacher = getattr(wrapper, "teacher_network", None)
    old_width = int(teacher.num_classes) if teacher is not None else 0
    mapping = getattr(wrapper, "seen_classes", {})
    for batch in dataset.as_numpy_iterator():
        # Prepared/noised tuples are not raw task datasets and must not be split.
        if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
            raise ValueError("Expected (images, sparse labels[, replay provenance]) batches.")
        x, y = np.asarray(batch[0]), np.asarray(batch[1]).reshape(-1)
        # Keep row counts and sparse integer labels aligned before indexing.
        if y.dtype.kind not in "iu" or len(x) != len(y) or not np.isfinite(x).all():
            raise ValueError("Phase pools require finite images and aligned integer labels.")
        mapped = np.asarray([mapping.get(int(label), int(label)) for label in y], dtype=np.int64)
        # Dense class vocabulary positions cannot be negative.
        if np.any(mapped < 0):
            raise ValueError("Phase pool labels must be nonnegative.")
        provided = len(batch) == 3
        # Mixing explicit and inferred provenance would make the data rule ambiguous.
        if has_metadata is not None and has_metadata != provided:
            raise ValueError("Replay provenance must be consistent across all batches.")
        has_metadata = provided
        mask = np.asarray(batch[2]).reshape(-1) if provided else mapped < old_width
        # Sample weights are not replay flags unless all values are binary.
        if len(mask) != len(y) or not np.isin(mask, (0, 1)).all():
            raise ValueError("Replay provenance must be aligned binary values.")
        mask = mask.astype(bool)
        # Current/new-class rows cannot be labeled as historical teacher replay.
        if np.any(mask & (mapped >= old_width)):
            raise ValueError("Replay rows must belong to the previous teacher's class vocabulary.")
        images.append(x)
        labels.append(y)
        flags.append(mask)
    # Empty pools cannot provide the mandated current acquisition phase.
    if not images:
        raise ValueError("Phase scheduling requires a nonempty current pool.")
    x, y, replay = np.concatenate(images), np.concatenate(labels), np.concatenate(flags)
    # At least one current example is needed for positive wake updates.
    if np.all(replay):
        raise ValueError("Phase scheduling requires current acquisition examples.")
    return {
        "current_images": x[~replay], "current_labels": y[~replay],
        "replay_images": x[replay], "replay_labels": y[replay],
        "provenance": "explicit_replay_mask" if has_metadata else "disjoint_previous_teacher_vocabulary",
        "old_class_count": old_width,
    }


def _pool_hash(images: np.ndarray, labels: np.ndarray) -> str:
    """Fingerprint ordered image/label values, including their shape and dtype.

    Args:
        images (np.ndarray): Numeric sample-major images in the configured model-input
            scale, normally float32 NHWC values in [-1, 1].
        labels (np.ndarray): Sparse integer label vector aligned with the image rows; the
            label convention for this operation is described above.

    Returns:
        digest (str): SHA-256 of ordered image/label shapes, dtypes and C-order values.

    Raises:
        AttributeError: If arguments are not ndarrays exposing shape/dtype and values.
    """

    digest = hashlib.sha256()
    for value in (images, labels):
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _cycle_indices(size: int, start: int, count: int, seed: int) -> np.ndarray:
    """Draw deterministic shuffled passes independent of fit-block boundaries.

    Args:
        size (int): Integer number of available rows in the finite pool.
        start (int): Nonnegative integer offset in the deterministic cyclic presentation
            stream.
        count (int): Exact integer number of expected or selected rows.
        seed (int): Explicit integer random seed; local or derived streams preserve
            reproducibility without reseeding caller-owned generators.

    Returns:
        indices (np.ndarray): Int64 vector of count shuffled-pass ordinals; concatenated
            blocks match one uninterrupted exposure stream.

    Raises:
        ValueError: If the source pool is empty or seed cannot define a random stream.
    """

    # Positive exposure cannot be sampled from an absent source pool.
    if size < 1:
        raise ValueError("A nonempty pool is required for positive phase exposures.")
    chunks = []
    while count:
        cycle, offset = divmod(start, size)
        order = np.random.default_rng(derive_seed(seed, cycle, "pool_cycle")).permutation(size)
        take = min(count, size - offset)
        chunks.append(order[offset:offset + take])
        start += take
        count -= take
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


class _ExposureAudit(tf.keras.callbacks.Callback):
    """Count completed full batches; optimizer deltas independently verify work."""

    def __init__(self) -> None:
        """Start an independent completed-batch counter for one explicit fit.

        Returns:
            initialized (None): None; starts with zero completed batches.

        Raises:
            None: Construction only initializes the Keras callback and an integer counter.
        """

        super().__init__()
        self.batches = 0

    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        """Count completed updates when a partial execution group ends.

        Args:
            batch (int): Zero-based completed Keras batch index, possibly the last index in a
                grouped execution callback.
            logs (dict | None): Optional Keras batch or epoch log mapping; this callback does
                not modify it.

        Returns:
            counted (None): None; stores the completed index capped by the finite
                block's step count, separately from its optimizer delta.

        Raises:
            TypeError: If the batch or finite step count is invalid.
        """

        # Keras reports the whole execution group's end, even for a shorter tail.
        self.batches = min(int(batch) + 1, int(self.params["steps"]))


def execute_schedule(
    wrapper: object, dataset: tf.data.Dataset, fit_kwargs: dict,
    settings: ScheduleSettings, *, fit_function: Callable | None = None,
    drift_probe: Callable | None = None, replay_selector: Callable | None = None,
    seed: int = 0,
) -> tuple[tf.keras.callbacks.History, dict, tf.data.Dataset]:
    """Fit existing losses on separated data and return histories and an audit.

    ``fit_function`` is a bound existing fit method (e.g. ``super().fit``).
    ``replay_selector(wrapper, x, y, wake_updates_completed)`` may replace the
    retained replay pool after every wake block and returns ``(x, y, audit)``.
    It must preserve retained row count. ``drift_probe(wrapper, x, y)`` returns
    mean JS in nats or a mapping containing ``mean_js`` plus scoring costs. To
    isolate scheduling from selection, use the same fixed pool and supply the
    same probe to every mode; refreshed selections change the data treatment.

    Each phase calls the existing fit with a finite exact-length dataset. No
    optimizer, loss, train_step, or teacher is replaced. Validation runs once,
    on the last block. Epoch-budget overrides and repeated callback lifecycles
    are disclosed, and incomplete/skipped updates fail instead of being called
    matched computation. Equal updates are not an assertion of equal FLOPs or
    wall time; scoring, fit overhead, and denoising draws remain separate costs.

    Args:
        wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
            schedules and existing training or inference APIs.
        dataset (tf.data.Dataset): Finite batched tf.data.Dataset containing raw images,
            sparse original labels and optional binary replay provenance.
        fit_kwargs (dict): Existing fit arguments, including optional validation_data and
            callbacks; explicit schedules own update budgets.
        settings (ScheduleSettings): Validated settings instance for this component; its
            fields select the behavior described above.
        fit_function (Callable | None): Existing bound fit callable; when None the scheduler
            calls the ordinary DiffusionClassifier fit implementation.
        drift_probe (Callable | None): Optional callable returning mean JS or an audit
            mapping after wake; required for adaptive replay timing.
        replay_selector (Callable | None): Optional callable returning replacement replay
            images, labels and audit after wake; retained row count must stay fixed.
        seed (int): Explicit integer random seed; local or derived streams preserve
            reproducibility without reseeding caller-owned generators.

    Returns:
        execution (tuple[tf.keras.callbacks.History, dict, tf.data.Dataset]): (history,
            audit, effective_pool): Keras History, JSON-compatible work/selection record and
            finite raw tf.data.Dataset.

    Raises:
        ValueError: If competing fit controls, early stopping, pool/selector/probe contracts
            or replay budgets are invalid.
        RuntimeError: If optimizer or observed batch counts fail to equal allocated updates.
    """

    settings.__post_init__()
    # Default directly to the existing method to avoid recursively invoking adapters.
    if fit_function is None:
        from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier
        fit_function = lambda **kwargs: DiffusionClassifier.fit(wrapper, **kwargs)
    # Opting out preserves the original dataset, callbacks, and fit budget exactly.
    if settings.mode == "joint":
        before = int(optimizer_iterations(wrapper.optimizer).numpy())
        history = fit_function(x=dataset, **fit_kwargs)
        return history, {
            "mode": "joint", "updates": int(optimizer_iterations(wrapper.optimizer).numpy()) - before,
            "fixed_budget": False,
        }, dataset
    forbidden = {"steps_per_epoch", "initial_epoch", "batch_size", "validation_split", "class_weight", "sample_weight"}.intersection(fit_kwargs)
    # Competing fit controls would change the declared update or exposure treatment.
    if forbidden:
        raise ValueError(f"Explicit scheduling owns fit controls: {sorted(forbidden)}.")
    # Early stopping can shorten training and restore a different checkpoint.
    if any(isinstance(callback, tf.keras.callbacks.EarlyStopping) for callback in fit_kwargs.get("callbacks", []) or []):
        raise ValueError("Early stopping is incompatible with the exact scheduling update budget.")
    pool = split_task_pool(wrapper, dataset)
    # Historical classes require actual supplied replay to spend a positive budget.
    if pool["old_class_count"] and settings.replay_updates and not len(pool["replay_labels"]):
        raise ValueError("A positive replay update budget requires supplied historical replay.")
    # The adaptive policy cannot operate on a missing measurement callback.
    if settings.mode == "drift" and len(pool["replay_labels"]) and settings.replay_updates and drift_probe is None:
        raise ValueError("Provide a matched-view drift probe for drift scheduling.")
    schedule = PhaseSchedule(settings, bool(len(pool["replay_labels"])))
    options = dict(fit_kwargs)
    validation_data = options.pop("validation_data", None)
    input_epochs = options.pop("epochs", 1)
    base_callbacks = list(options.pop("callbacks", []) or [])
    options.update(epochs=1, shuffle=False)
    merged = tf.keras.callbacks.History()
    merged.set_model(wrapper)
    merged.history, merged.epoch = {}, []
    counts = {"wake": 0, "replay": 0}
    exposure_digests = {phase: hashlib.sha256() for phase in counts}
    actual_class_counts = {"wake": {}, "replay": {}}
    audit = {
        "mode": settings.mode, "fixed_budget": True,
        "requested_wake_updates": settings.wake_updates,
        "requested_replay_updates": settings.replay_updates,
        "effective_replay_updates": schedule.replay_budget,
        "batch_size": settings.batch_size, "replaced_common_epochs": input_epochs,
        "pool_provenance": pool["provenance"],
        "current_pool_rows": len(pool["current_labels"]),
        "replay_pool_rows": len(pool["replay_labels"]),
        "transient_pool_bytes": sum(pool[key].nbytes for key in (
            "current_images", "current_labels", "replay_images", "replay_labels")),
        "memory_scope": "Materialized NumPy pool arrays only; excludes original tf.data storage, sampled block copies, and allocator overhead.",
        "blocks": [], "decisions": [], "selections": [],
        "computation_scope": "Exact full-batch updates and presentations; scoring and fit overhead reported separately, not equal FLOPs.",
        "callback_lifecycle": "Existing callbacks restart for each explicit fit block; validation runs on final block only.",
    }
    started = time.perf_counter()

    def fit_block(phase: str, updates: int, final: bool = False) -> None:
        """Fit one existing-objective block and verify every allocated update.

        Args:
            phase (str): wake for current-only data or replay for historical generated data.
            updates (int): Positive integer number of completed full-batch optimizer updates
                allocated to this block.
            final (bool): Whether this is the final fit block; only that block receives
                validation_data.

        Returns:
            fitted (None): None; fits one exact full-batch block and updates merged history,
                presentation hashes and cost audit.

        Raises:
            ValueError: If the selected phase pool cannot supply the allocated presentations.
            RuntimeError: If completed optimizer or batch counts differ from updates.
        """

        key = "current" if phase == "wake" else "replay"
        images, labels = pool[f"{key}_images"], pool[f"{key}_labels"]
        indices = _cycle_indices(len(labels), counts[phase], updates * settings.batch_size,
                                 derive_seed(seed, phase, "schedule_pool"))
        selected_labels = labels[indices]
        finite = get_dataset(
            images[indices], selected_labels, batch_size=settings.batch_size,
            shuffle_buffer=0, drop_remainder=False,
            metadata=np.full(len(indices), phase == "replay", dtype=bool),
        )
        observer = _ExposureAudit()
        block_options = dict(options)
        block_options["callbacks"] = [*base_callbacks, observer]
        # Hold validation work to one call independently of the number of fit blocks.
        if final and validation_data is not None:
            block_options["validation_data"] = validation_data
        before = int(optimizer_iterations(wrapper.optimizer).numpy())
        block_started = time.perf_counter()
        allocation = getattr(wrapper, "_scheduled_kd_allocation", None)
        # An opt-in runtime context affects replay KD only and restores after errors.
        with allocation.phase(wrapper, phase) if allocation is not None else nullcontext():
            history = fit_function(x=finite, **block_options)
        elapsed = getattr(wrapper, "_checkpoint_elapsed_seconds", time.perf_counter() - block_started)
        actual = int(optimizer_iterations(wrapper.optimizer).numpy()) - before
        # Shortened fits or skipped optimizer steps cannot count as matched budgets.
        if actual != updates or observer.batches != updates:
            raise RuntimeError(f"{phase} budget incomplete: expected {updates}, optimizer applied {actual}, observed {observer.batches} batches.")
        counts[phase] += len(indices)
        # Row-wise digests permit identical exposure streams across different block sizes.
        for index in indices:
            exposure_digests[phase].update(_pool_hash(images[index:index + 1], labels[index:index + 1]).encode())
        labels_unique, frequencies = np.unique(selected_labels, return_counts=True)
        for label, count in zip(labels_unique, frequencies):
            label = str(int(label))
            actual_class_counts[phase][label] = actual_class_counts[phase].get(label, 0) + int(count)
        block = {
            "phase": phase, "updates": actual, "presentations": len(indices),
            "seconds": elapsed, "pool_sha256": _pool_hash(images, labels),
            "presentations_sha256": _pool_hash(images[indices], selected_labels),
            "unique_pool_indices": len(np.unique(indices)),
            "class_presentations": {str(int(label)): int(count) for label, count in zip(labels_unique, frequencies)},
            "history": {name: list(map(float, values)) for name, values in history.history.items()},
        }
        audit["blocks"].append(block)
        position = len(merged.epoch)
        # Missing validation entries remain NaN in the aggregate epoch series.
        for name in set(merged.history) | set(history.history):
            # Newly encountered validation metrics need explicit earlier missing cells.
            if name not in merged.history:
                merged.history[name] = [float("nan")] * position
            merged.history[name].append(float(history.history[name][-1]) if name in history.history else float("nan"))
        merged.epoch.append(position)

    while True:
        updates = schedule.next_wake()
        # Zero remaining acquisition means the final replay flush already completed.
        if not updates:
            break
        final_wake = schedule.wake_completed == settings.wake_updates
        remaining_replay = schedule.replay_budget - schedule.replay_completed
        fit_block("wake", updates, final=final_wake and remaining_replay == 0)
        # Select on the live post-wake model, never before its acquisition updates.
        if replay_selector is not None and len(pool["replay_labels"]):
            before_selection = time.perf_counter()
            replacement = replay_selector(wrapper, pool["replay_images"], pool["replay_labels"], schedule.wake_completed)
            # A replacement must carry its own resource and selection evidence.
            if not isinstance(replacement, (tuple, list)) or len(replacement) != 3:
                raise ValueError("Replay selector must return images, labels, and an audit mapping.")
            images, labels, selected_audit = replacement
            images, labels = np.asarray(images), np.asarray(labels).reshape(-1)
            # Enlarging the retained pool would change the declared data budget.
            if images.shape != pool["replay_images"].shape or len(labels) != len(pool["replay_labels"]):
                raise ValueError("Selection must preserve retained replay rows and image shape.")
            # Validate the replacement before it can become training input.
            if labels.dtype.kind not in "iu" or not np.isfinite(images).all() or not isinstance(selected_audit, Mapping):
                raise ValueError("Selection must return finite images, integer labels, and an audit mapping.")
            mapped = [wrapper.seen_classes.get(int(label), int(label)) for label in labels]
            # Replay selection cannot introduce labels unknown to the prior teacher.
            if any(label < 0 or label >= pool["old_class_count"] for label in mapped):
                raise ValueError("Selected replay labels must remain in the previous vocabulary.")
            changed = _pool_hash(images, labels) != _pool_hash(pool["replay_images"], pool["replay_labels"])
            pool["replay_images"], pool["replay_labels"] = images, labels
            audit["selections"].append({
                "after_wake_updates": schedule.wake_completed, "pool_changed": changed,
                "seconds": time.perf_counter() - before_selection, "audit": dict(selected_audit),
            })
        measurement = None
        probe_seconds = 0.
        # Fixed controls can run the identical probe to match scoring work.
        if drift_probe is not None and len(pool["replay_labels"]):
            probe_started = time.perf_counter()
            measurement = drift_probe(wrapper, pool["replay_images"], pool["replay_labels"])
            probe_seconds = time.perf_counter() - probe_started
            measurement = dict(measurement) if isinstance(measurement, Mapping) else {"mean_js": float(measurement)}
        mean_js = measurement["mean_js"] if measurement is not None else None
        replay_updates, reason = schedule.after_wake(mean_js)
        audit["decisions"].append({
            "after_wake_updates": schedule.wake_completed, "replay_updates": replay_updates,
            "reason": reason, "measurement": measurement, "probe_seconds": probe_seconds,
        })
        # Below-threshold boundaries reserve replay for a later wake boundary.
        if replay_updates:
            fit_block("replay", replay_updates, final=final_wake)
    audit.update({
        "updates": schedule.wake_completed + schedule.replay_completed,
        "wake_updates": schedule.wake_completed, "replay_updates": schedule.replay_completed,
        "presentations": counts, "class_presentations": actual_class_counts,
        "exposure_sequence_sha256": {phase: digest.hexdigest() for phase, digest in exposure_digests.items()},
        "seconds": time.perf_counter() - started,
        "selection_changed_during_schedule": any(item["pool_changed"] for item in audit["selections"]),
    })
    effective = get_dataset(
        np.concatenate((pool["current_images"], pool["replay_images"])),
        np.concatenate((pool["current_labels"], pool["replay_labels"])),
        batch_size=settings.batch_size, shuffle_buffer=0, drop_remainder=False,
        metadata=np.concatenate((np.zeros(len(pool["current_labels"]), dtype=bool),
                                 np.ones(len(pool["replay_labels"]), dtype=bool))),
    )
    return merged, audit, effective
