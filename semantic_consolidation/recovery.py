"""Plain semantic checkpoint state encoded by the existing common serializer."""

from __future__ import annotations

from copy import deepcopy
import time

import numpy as np


def completed_observer_state(observer: object | None) -> dict[str, object] | None:
    """Copy the diagnostic history and fixed cohorts at a complete task boundary.

    Args:
        observer (object | None): Attached ExperimentalController, or None when
            observation is disabled. Its task-local candidate pools must be empty.

    Returns:
        state (dict[str, object] | None): Detached records, learning curves,
            retained float32 images/features, integer class/task metadata and
            measured resource segments; None preserves an absent observer.

    Raises:
        ValueError: If an observer still owns an unfinished task or candidate pool.
    """
    # Disabled observation contributes no payload or runtime work.
    if observer is None:
        return None
    # Task-local image reservoirs are committed through the task diagnostic record.
    if observer.validation is not None or observer.candidates or observer.candidate_rng:
        raise ValueError("Observer task state is not at a completed boundary.")
    from semantic_consolidation.controller import _json_value
    return deepcopy({
        "records": _json_value(observer.records), "curves": _json_value(observer.curves),
        "representatives": observer.representatives,
        "cohorts": {str(key): value for key, value in observer.probe.cohorts.items()},
        "last_task": observer.probe.last_task, "old_count": observer.old_count,
        "elapsed_seconds": time.perf_counter() - observer.started,
        "resource_segments": [*getattr(observer.monitor, "previous_segments", []),
                              observer.monitor.snapshot(include_previous=False)],
    })


def validate_observer_state(observer: object | None, state: dict[str, object] | None,
                            completed_tasks: int, class_count: int) -> None:
    """Check persisted cohorts and task alignment without mutating the observer.

    Args:
        observer (object | None): Configured ExperimentalController or None.
        state (dict[str, object] | None): Decoded state from completed_observer_state;
            image/feature arrays retain their original float32 dtype.
        completed_tasks (int): Nonnegative authoritative completed-task count;
            zero identifies the initial empty observer boundary.
        class_count (int): Nonnegative authoritative dense class count, zero
            before any task has completed.

    Returns:
        validated (None): None; no model, controller, random stream or monitor is
            changed, including when malformed state is rejected.

    Raises:
        ValueError: If ownership, task ordering, cohort array contracts, image
            hashes, class counts or nonnegative resource durations are invalid.
    """
    # Configuration determines whether observer state is required.
    if (observer is None) != (state is None):
        raise ValueError("Semantic checkpoint observer ownership differs from the configured route.")
    # There is nothing further to validate when observation is disabled.
    if state is None:
        return
    keys = {"records", "curves", "representatives", "cohorts", "last_task", "old_count",
            "elapsed_seconds", "resource_segments"}
    # Require a complete schema rather than silently defaulting missing measurements.
    if (not isinstance(state, dict) or set(state) != keys
            or state["last_task"] != (completed_tasks or None)
            or type(state["old_count"]) is not int or state["old_count"] != class_count
            or not isinstance(state["records"], list) or len(state["records"]) != completed_tasks
            or any(not isinstance(row, dict) or row.get("task") != index
                   for index, row in enumerate(state["records"], 1))
            or not isinstance(state["curves"], list) or not isinstance(state["representatives"], list)
            or not isinstance(state["cohorts"], dict) or not isinstance(state["resource_segments"], list)
            or not isinstance(state["elapsed_seconds"], (int, float))
            or not np.isfinite(state["elapsed_seconds"]) or state["elapsed_seconds"] < 0):
        raise ValueError("Semantic checkpoint observer schema or task cursor is invalid.")
    from semantic_consolidation.experimental_diagnostics import _image_hash
    for class_id, cohort in state["cohorts"].items():
        required = {"images", "hashes", "acquisition_features", "acquisition_task",
                    "previous_features", "previous_task"}
        # Cohorts retain original integer label identities and ordered observations.
        if (not isinstance(class_id, str) or not class_id.isdecimal()
                or not isinstance(cohort, dict) or set(cohort) != required
                or type(cohort["acquisition_task"]) is not int
                or type(cohort["previous_task"]) is not int
                or not 1 <= cohort["acquisition_task"] <= cohort["previous_task"] <= completed_tasks
                or not isinstance(cohort["hashes"], list)
                or not 1 <= len(cohort["hashes"]) <= observer.probe.per_class):
            raise ValueError("Semantic checkpoint fixed-cohort metadata is invalid.")
        rows = len(cohort["hashes"])
        for name in ("acquisition_features", "previous_features"):
            values = cohort[name]
            # Fixed-cohort comparisons need aligned, finite feature matrices.
            if (not isinstance(values, np.ndarray) or values.dtype != np.dtype("float32")
                    or values.ndim != 2 or values.shape[0] != rows or not np.isfinite(values).all()):
                raise ValueError("Semantic checkpoint fixed features must be finite float32 matrices.")
        # Both reference observations must use the same hidden projection.
        if cohort["acquisition_features"].shape != cohort["previous_features"].shape:
            raise ValueError("Semantic checkpoint fixed feature widths differ.")
        images = cohort["images"]
        # Runtime observers retain their exact validation rows, never replacement images.
        if (not isinstance(images, np.ndarray) or images.dtype != np.dtype("float32")
                or images.ndim != 4 or len(images) != rows or not np.isfinite(images).all()
                or [_image_hash(image) for image in images] != cohort["hashes"]):
            raise ValueError("Semantic checkpoint fixed validation images or hashes are invalid.")
    for row in state["curves"]:
        # Every learning-curve observation belongs to an already completed task.
        if not isinstance(row, dict) or type(row.get("task")) is not int or not 1 <= row["task"] <= completed_tasks:
            raise ValueError("Semantic checkpoint learning-curve task is invalid.")
    for row in state["representatives"]:
        # Generated representatives remain typed audit arrays, outside training replay.
        if (not isinstance(row, (list, tuple)) or len(row) != 3 or type(row[0]) is not int
                or not 1 <= row[0] <= completed_tasks or not isinstance(row[1], np.ndarray)
                or row[1].dtype != np.dtype("float32") or row[1].ndim != 4
                or not np.isfinite(row[1]).all() or not isinstance(row[2], np.ndarray)
                or row[2].dtype.kind not in "iu" or row[2].shape != (len(row[1]),)):
            raise ValueError("Semantic checkpoint generated representative arrays are invalid.")
    # Resource histories are observations from separate process lifetimes.
    if any(not isinstance(segment, dict) for segment in state["resource_segments"]):
        raise ValueError("Semantic checkpoint resource segments must be mappings.")


def restore_observer_state(observer: object | None, state: dict[str, object] | None) -> None:
    """Install prevalidated history while retaining the new process's live monitor.

    Args:
        observer (object | None): Configured live observer, or None when disabled.
        state (dict[str, object] | None): Payload already accepted by
            validate_observer_state; arrays are copied before installation.

    Returns:
        restored (None): None; restores independent diagnostic cohorts and records.
            Active elapsed time excludes restart downtime; old allocator peaks
            remain separate process segments, never current allocator values.

    Raises:
        KeyError: If called directly with an incomplete, unvalidated payload.
    """
    # Disabled observation has no live or saved state.
    if observer is None:
        return
    saved = deepcopy(state)
    for name in ("records", "curves", "representatives", "old_count"):
        setattr(observer, name, saved[name])
    observer.probe.cohorts = {int(key): value for key, value in saved["cohorts"].items()}
    observer.probe.last_task = saved["last_task"]
    observer.started = time.perf_counter() - saved["elapsed_seconds"]
    observer.monitor.previous_segments = saved["resource_segments"]
    observer.candidates, observer.generated_counts, observer.candidate_rng = [], {}, {}
    observer.validation = None
    observer.sampling_seconds = 0.
    observer.accepting_candidates = True
