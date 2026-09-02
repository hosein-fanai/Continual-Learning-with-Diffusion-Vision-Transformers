"""Machine-readable CSV and TensorBoard reporting for continual experiments.

The helpers in this module consume the mapping returned by
``common.learner._run_continual_tasks(return_details=True)`` without importing
the learner or training orchestration. Missing optional histories, evaluations,
schedules, and matrices produce no fabricated metric rows; every CSV remains
parseable with a stable schema, and the summary still records zero represented
tasks and an empty matrix inventory.
"""

from __future__ import annotations

import numpy as np

import csv

import json

import re

from pathlib import Path

from collections.abc import Mapping, Sequence
from numbers import Integral, Real


_MISSING = object()
"""Sentinel separating unsupported values from valid falsey scalars."""

_HISTORY_SOURCES = (
    ("classifier", "histories"), 
    ("generator", "generative_histories")
)
"""Phase names and their corresponding learner-detail history keys."""

_CLASSIFIER_HISTORY_NAMES = frozenset((
    "classifier_loss", 
    "clf_loss", 
    "clf_kl_loss", 
    "clf_ctr_loss", 
    "clf_distil_loss", 
    "classifier_accuracy", 
    "clf_accuracy", 
    "cls_token_accuracy", 
    "avg_pooling_accuracy", 
    "clf_ctr_accuracy", 
    "clf_distil_acc", 
    "total_accuracy"
))
"""Classifier/KD metrics embedded in a joint generator history."""

_EVALUATION_SOURCES = (
    ("classifier", "classifier_evaluations"), 
    ("generator", "generative_evaluations")
)
"""Phase names and their corresponding learner-detail evaluation keys."""

_TASK_DIAGNOSTIC_SOURCES = (
    ("resource", "task_resource_metrics", "resource_metrics"), 
    ("mechanistic", "task_mechanistic_metrics", "mechanistic_metrics")
)
"""Task diagnostics with canonical learner keys and legacy-friendly aliases."""


def _observed(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return non-NaN values as a float64 array."""

    array = np.asarray(values, dtype="float64")
    return array[~np.isnan(array)]


def observed_mean(values: Sequence[float] | np.ndarray) -> float:
    """Return the observed mean, or NaN when every value is unavailable."""

    values = _observed(values)
    return float(np.mean(values)) if values.size else float("nan")


def observed_max(values: Sequence[float] | np.ndarray) -> float:
    """Return the observed maximum, or NaN when every value is unavailable."""

    values = _observed(values)
    return float(np.max(values)) if values.size else float("nan")


def continual_metrics(
    accuracy_matrix: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Compute task-balanced metrics from a continual accuracy matrix."""

    matrix = np.asarray(accuracy_matrix, dtype="float64")
    task_num = len(matrix)
    if task_num == 0:
        return {
            "final_average_accuracy": np.nan,
            "average_incremental_accuracy": np.nan,
            "average_forgetting": np.nan,
            "backward_transfer": np.nan,
        }

    row_averages = [
        observed_mean(matrix[index, :index + 1])
        for index in range(task_num)
    ]
    if task_num == 1:
        average_forgetting = 0.
        backward_transfer = 0.
    else:
        final_old = matrix[-1, :task_num - 1]
        maxima = np.asarray([
            observed_max(matrix[index:task_num - 1, index])
            for index in range(task_num - 1)
        ])
        diagonal = np.asarray([
            matrix[index, index] for index in range(task_num - 1)
        ])
        average_forgetting = observed_mean(maxima - final_old)
        backward_transfer = observed_mean(final_old - diagonal)

    return {
        "final_average_accuracy": observed_mean(matrix[-1, :task_num]),
        "average_incremental_accuracy": observed_mean(row_averages),
        "average_forgetting": average_forgetting,
        "backward_transfer": backward_transfer,
    }


def task_accuracy_summaries(
    accuracy_matrix: Sequence[Sequence[float]],
) -> tuple[list[float], list[float]]:
    """Return current-task and prior-task macro accuracy after each task."""

    new_task_accuracy = [
        float(accuracy_matrix[index][index])
        for index in range(len(accuracy_matrix))
    ]
    old_task_accuracy = [
        np.nan if index == 0 else observed_mean(accuracy_matrix[index][:index])
        for index in range(len(accuracy_matrix))
    ]
    return new_task_accuracy, old_task_accuracy


def _python_value(value: object) -> object:
    """Convert tensor and NumPy containers to ordinary Python values.

    Args:
        value (object): Scalar or container that may expose ``numpy`` or
            ``tolist`` conversion methods.

    Returns:
        object: A Python scalar/container when conversion is supported, or the
        original value otherwise.
    """

    # Materialize eager TensorFlow values without importing TensorFlow here.
    if callable(getattr(value, "numpy", None)):
        value = value.numpy()

    # Convert NumPy arrays and scalar objects to JSON/CSV-friendly values.
    if isinstance(value, np.ndarray):
        return value.tolist()

    # Unbox one NumPy scalar while preserving its exact Python value.
    if isinstance(value, np.generic):
        return value.item()

    return value


def _numeric_scalar(value: object) -> object:
    """Return one real scalar or the internal missing-value sentinel.

    Args:
        value (object): Candidate metric value.

    Returns:
        object: A Python bool, integer, or float for a scalar real value;
        otherwise ``_MISSING``.
    """

    value = _python_value(value)

    # Preserve Boolean metadata and ordinary real-valued metrics.
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    # Preserve integral metrics without converting them to floating point.
    if isinstance(value, Integral):
        return int(value)

    # Normalize remaining real metrics to ordinary Python floats.
    if isinstance(value, Real):
        return float(value)

    return _MISSING


def _scalar_series(value: object) -> list[tuple[int, object]]:
    """Extract indexed scalar values from a scalar or one-dimensional series.

    Args:
        value (object): History value, normally a scalar sequence with one
            value per epoch.

    Returns:
        list[tuple[int, object]]: ``(index, scalar)`` pairs. Unsupported or
        multidimensional values contribute no pairs.
    """

    scalar = _numeric_scalar(value)
    # Treat a direct scalar as the first and only observation.
    if scalar is not _MISSING:
        return [(0, scalar)]

    value = _python_value(value)
    # Mappings and text are metadata, not scalar metric trajectories.
    if isinstance(value, Mapping) or isinstance(
        value, (str, bytes, bytearray)
    ):
        return []

    # Materialize only finite one-dimensional iterable structures.
    if isinstance(value, Sequence):
        values = list(value)
    # Fall back to NumPy conversion for tensor-like non-Sequence values.
    else:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return []
        # Reject matrices and higher-rank tensors as scalar histories.
        if array.ndim != 1:
            return []

        values = array.tolist()

    extracted = []
    for index, item in enumerate(values):
        item_scalar = _numeric_scalar(item)
        # Preserve every scalar observation while skipping structured entries.
        if item_scalar is not _MISSING:
            extracted.append((index, item_scalar))

    return extracted


def _task_mappings(value: object) -> list[Mapping[str, object]]:
    """Normalize optional per-task mappings into an owned list.

    Args:
        value (object): One mapping, a sequence of optional mappings, or a
            missing/unsupported value.

    Returns:
        list[Mapping[str, object]]: One mapping per represented task. Missing
        sequence items become empty mappings.
    """

    # A direct history/evaluation mapping represents one task.
    if isinstance(value, Mapping):
        return [value]

    # Reject text and non-sequence values before task normalization.
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        return []

    normalized = []
    for item in value:
        # Retain task alignment even when an optional phase did not run.
        normalized.append(item if isinstance(item, Mapping) else {})

    return normalized


def _is_classifier_history_metric(name: object) -> bool:
    """Return whether one joint-history key belongs to classifier/KD output.

    Args:
        name (object): Keras history metric name, optionally ``val_`` prefixed.

    Returns:
        bool: True for the stable classifier and distillation metric family.
    """

    normalized = str(name).lower()
    # Validation metrics retain the phase of their unprefixed counterpart.
    if normalized.startswith("val_"):
        normalized = normalized[4:]

    return normalized in _CLASSIFIER_HISTORY_NAMES


def _same_history(
    first: Mapping[str, object], 
    second: Mapping[str, object]
) -> bool:
    """Compare two serialized Keras histories without array truth ambiguity.

    Args:
        first (Mapping[str, object]): Candidate classifier-side mapping.
        second (Mapping[str, object]): Candidate generator-side mapping.

    Returns:
        bool: True when both contain the same JSON-representable trajectories.
    """

    # Live joint learner details intentionally reference one shared mapping.
    if first is second:
        return True

    # Distinct metric keys prove that separate optimization phases ran.
    if set(first) != set(second):
        return False

    return _json_cell(first) == _json_cell(second)


def _histories_by_phase(
    details: Mapping[str, object]
) -> dict[str, list[Mapping[str, object]]]:
    """Return task histories with duplicate joint mappings phase-partitioned.

    The learner retains a joint history under both legacy detail keys so old
    consumers still see the complete mapping. Reporting partitions only that
    exact duplicate: classifier/KD metrics appear under ``classifier`` and all
    noise, reconstruction, and other generator metrics under ``generator``.

    Args:
        details (Mapping[str, object]): Continual learner detail mapping.

    Returns:
        dict[str, list[Mapping[str, object]]]: Task-aligned phase histories.
    """

    classifier = _task_mappings(details.get("histories"))
    generator = _task_mappings(details.get("generative_histories"))
    task_count = max(len(classifier), len(generator))
    resolved: dict[str, list[Mapping[str, object]]] = {
        "classifier": [], 
        "generator": []
    }
    for task_index in range(task_count):
        classifier_history = classifier[task_index] if task_index < len(classifier) \
                            else {}
        generator_history = generator[task_index] if task_index < len(generator) \
                            else {}

        # Split only the known duplicate produced by one joint Keras fit.
        if classifier_history and generator_history and _same_history(
            classifier_history, generator_history
        ):
            resolved["classifier"].append({
                name: values
                for name, values in classifier_history.items()
                if _is_classifier_history_metric(name)
            })
            resolved["generator"].append({
                name: values
                for name, values in generator_history.items()
                if not _is_classifier_history_metric(name)
            })
        # Preserve independently fitted phase histories without filtering.
        else:
            resolved["classifier"].append(classifier_history)
            resolved["generator"].append(generator_history)

    return resolved


def _task_diagnostic_mappings(
    details: Mapping[str, object]
) -> list[tuple[str, list[Mapping[str, object]]]]:
    """Resolve canonical per-task diagnostics and optional short aliases.

    Canonical ``task_*`` fields take precedence whenever present, including an
    explicitly empty list. This prevents an alias from accidentally duplicating
    or reviving diagnostics that a caller intentionally disabled.

    Args:
        details (Mapping[str, object]): Continual learner detail mapping.

    Returns:
        list[tuple[str, list[Mapping[str, object]]]]: Reporting phase and its
        task-aligned nested metric mappings.
    """

    resolved = []
    for phase, canonical_name, alias_name in _TASK_DIAGNOSTIC_SOURCES:
        source = details.get(
            canonical_name, 
            details.get(alias_name, _MISSING)
        )
        mappings = [] if source is _MISSING else _task_mappings(source)
        resolved.append((phase, mappings))

    return resolved


def _task_groups(details: Mapping[str, object]) -> list[list[object]]:
    """Return normalized original-label task groups from learner details.

    Args:
        details (Mapping[str, object]): Continual learner detail mapping.

    Returns:
        list[list[object]]: Independent class lists in task order. Invalid or
        missing group metadata produces an empty list.
    """

    value = details.get("task_classes", details.get("task_groups"))
    # Require a non-text sequence before interpreting task groups.
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        return []

    groups = []
    for group in value:
        group = _python_value(group)
        # Preserve ordinary task sequences while rejecting scalar labels here.
        if isinstance(group, Sequence) and not isinstance(
            group, (str, bytes, bytearray)
        ):
            groups.append(list(group))
        # Retain task alignment when one group is malformed or absent.
        else:
            groups.append([])

    return groups


def _seen_classes(groups: Sequence[Sequence[object]], index: int) -> list[object]:
    """Flatten task groups through one zero-based task index.

    Args:
        groups (Sequence[Sequence[object]]): Original-label groups in schedule
            order.
        index (int): Inclusive task index.

    Returns:
        list[object]: Classes introduced through ``index`` in schedule order.
    """

    return [
        label
        for group in groups[:index + 1]
        for label in group
    ]


def _jsonable(value: object) -> object:
    """Recursively convert common scientific values to JSON-safe structures.

    Args:
        value (object): Candidate metadata value.

    Returns:
        object: JSON-compatible scalar/container, or ``_MISSING`` when the
        value cannot be represented without an unstable object repr.
    """

    value = _python_value(value)
    scalar = _numeric_scalar(value)

    # Preserve numeric scalars in their ordinary Python representation.
    if scalar is not _MISSING:
        return scalar

    # Preserve ordinary JSON null and text values.
    if value is None or isinstance(value, str):
        return value

    # Recursively normalize mappings while requiring string-like keys.
    if isinstance(value, Mapping):
        normalized_mapping = {}
        for key, item in value.items():
            normalized_item = _jsonable(item)
            # Skip live models, callbacks, and other unsupported metadata.
            if normalized_item is not _MISSING:
                normalized_mapping[str(key)] = normalized_item

        return normalized_mapping

    # Recursively normalize sequences while excluding binary data.
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        normalized_sequence = []
        for item in value:
            normalized_item = _jsonable(item)

            # Omit unsupported objects without losing supported neighbors.
            if normalized_item is not _MISSING:
                normalized_sequence.append(normalized_item)
    
        return normalized_sequence

    return _MISSING


def _json_cell(value: object) -> str:
    """Serialize one supported metadata value into a compact JSON cell.

    Args:
        value (object): Metadata scalar or container.

    Returns:
        str: Compact JSON text, or an empty string for unsupported values.
    """

    normalized = _jsonable(value)
    # Keep unsupported live objects out of persistent metadata.
    if normalized is _MISSING:
        return ""

    return json.dumps(
        normalized, 
        ensure_ascii=False, 
        separators=(",", ":"), 
        allow_nan=True
    )


def _flatten_scalar_mapping(
    value: object, 
    prefix: str = ""
) -> list[tuple[str, object]]:
    """Flatten nested evaluation mappings and scalar sequences.

    Args:
        value (object): Nested mapping, scalar, or scalar sequence.
        prefix (str): Slash-delimited source path accumulated by recursion.

    Returns:
        list[tuple[str, object]]: Metric-path and scalar-value pairs.
    """

    scalar = _numeric_scalar(value)
    # Emit a scalar leaf under its complete source path.
    if scalar is not _MISSING:
        return [(prefix or "value", scalar)]

    value = _python_value(value)
    # Recurse through named evaluation mappings.
    if isinstance(value, Mapping):
        flattened = []
        for name, item in value.items():
            child_prefix = f"{prefix}/{name}" if prefix else str(name)
            flattened.extend(_flatten_scalar_mapping(item, child_prefix))

        return flattened

    series = _scalar_series(value)

    # Preserve scalar list results such as ordinary Keras ``evaluate`` output.
    if series:
        return [
            (f"{prefix}/{index}" if prefix else str(index), scalar_value)
            for index, scalar_value in series
        ]

    return []


def _task_count(details: Mapping[str, object]) -> int:
    """Infer the represented continual task count without requiring a schedule.

    Args:
        details (Mapping[str, object]): Continual learner detail mapping.

    Returns:
        int: Maximum task count represented by schedules, histories,
        evaluations, task-level metric series, or accuracy matrices.
    """

    counts = [len(_task_groups(details))]

    for _, source_name in (*_HISTORY_SOURCES, *_EVALUATION_SOURCES):
        counts.append(len(_task_mappings(details.get(source_name))))

    for _, diagnostics in _task_diagnostic_mappings(details):
        counts.append(len(diagnostics))

    for name, value in details.items():
        normalized_name = str(name).lower()
        # Count top-level per-task accuracy trajectories, but not matrices.
        if "accur" in normalized_name and "matrix" not in normalized_name:
            # Direct scalar summaries do not establish a continual task count.
            if _numeric_scalar(value) is _MISSING:
                counts.append(len(_scalar_series(value)))

    for _, matrix in _accuracy_matrices(details):
        matrix = _python_value(matrix)
        # Matrix row count also identifies the number of completed tasks.
        if isinstance(matrix, Sequence) and not isinstance(
            matrix, (str, bytes, bytearray)
        ):
            counts.append(len(matrix))

    return max(counts, default=0)


def _accuracy_matrices(
    details: Mapping[str, object]
) -> list[tuple[str, object]]:
    """Discover every top-level or grouped accuracy matrix.

    Args:
        details (Mapping[str, object]): Continual learner detail mapping.

    Returns:
        list[tuple[str, object]]: Stable matrix-name and matrix-value pairs.
        Recognized top-level names contain ``"accuracy_matrix"``; a mapping
        stored as ``accuracy_matrices`` contributes each child matrix.
    """

    matrices = []
    seen_names = set()
    for name, value in details.items():
        normalized_name = str(name).lower()
        # Expand a future grouped matrix mapping without prescribing names.
        if normalized_name == "accuracy_matrices" and isinstance(value, Mapping):
            for child_name, child_matrix in value.items():
                child_name = str(child_name)

                # Keep the first occurrence when a matrix is exposed twice.
                if child_name not in seen_names:
                    matrices.append((child_name, child_matrix))
                    seen_names.add(child_name)
        # Capture current and future explicitly named accuracy matrices.
        elif "accuracy_matrix" in normalized_name:
            matrix_name = str(name)

            # Preserve the first matrix under each stable source key.
            if matrix_name not in seen_names:
                matrices.append((matrix_name, value))
                seen_names.add(matrix_name)

    return matrices


def _write_csv(
    path: Path, 
    fieldnames: Sequence[str], 
    rows: Sequence[Mapping[str, object]]
) -> None:
    """Atomically write one UTF-8 CSV file with a stable header.

    Args:
        path (pathlib.Path): Final artifact path.
        fieldnames (Sequence[str]): Ordered CSV columns.
        rows (Sequence[Mapping[str, object]]): Row mappings to serialize.

    Returns:
        None.
    """

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(path)


def _epoch_rows(
    details: Mapping[str, object], 
    groups: Sequence[Sequence[object]]
) -> list[dict[str, object]]:
    """Build long-form epoch metric rows for classifier and generator phases.

    Args:
        details (Mapping[str, object]): Continual learner detail mapping.
        groups (Sequence[Sequence[object]]): Original-label task groups.

    Returns:
        list[dict[str, object]]: One row for every scalar history observation.
    """

    rows = []
    for phase, histories in _histories_by_phase(details).items():
        for task_index, history in enumerate(histories):
            task_classes = list(groups[task_index]) \
                if task_index < len(groups) else []
            seen_classes = _seen_classes(groups, task_index)
            for metric, values in history.items():
                for epoch, value in _scalar_series(values):
                    rows.append({
                        "task_index": task_index, 
                        "phase": phase, 
                        "epoch": epoch, 
                        "task_classes": _json_cell(task_classes), 
                        "seen_classes": _json_cell(seen_classes), 
                        "metric": str(metric), 
                        "value": value
                    })

    return rows


def _task_metric_rows(
    details: Mapping[str, object], 
    groups: Sequence[Sequence[object]]
) -> list[dict[str, object]]:
    """Build long-form task-level evaluation and accuracy rows.

    Args:
        details (Mapping[str, object]): Continual learner detail mapping.
        groups (Sequence[Sequence[object]]): Original-label task groups.

    Returns:
        list[dict[str, object]]: Scalar evaluation leaves and top-level
        per-task accuracy values.
    """

    rows = []
    for phase, source_name in _EVALUATION_SOURCES:
        evaluations = _task_mappings(details.get(source_name))
        for task_index, evaluation in enumerate(evaluations):
            task_classes = list(groups[task_index]) if task_index < len(groups) \
                        else []
            seen_classes = _seen_classes(groups, task_index)
            for metric, value in _flatten_scalar_mapping(evaluation):
                rows.append({
                    "task_index": task_index, 
                    "phase": phase, 
                    "task_classes": _json_cell(task_classes), 
                    "seen_classes": _json_cell(seen_classes), 
                    "metric": metric, 
                    "value": value
                })

    # Store optional nested task diagnostics beside ordinary evaluations.
    for phase, task_diagnostics in _task_diagnostic_mappings(details):
        for task_index, diagnostics in enumerate(task_diagnostics):
            task_classes = list(groups[task_index]) if task_index < len(groups) \
                        else []
            seen_classes = _seen_classes(groups, task_index)
            for metric, value in _flatten_scalar_mapping(diagnostics):
                rows.append({
                    "task_index": task_index, 
                    "phase": phase, 
                    "task_classes": _json_cell(task_classes), 
                    "seen_classes": _json_cell(seen_classes), 
                    "metric": metric, 
                    "value": value
                })

    for name, values in details.items():
        normalized_name = str(name).lower()
        # Include every top-level per-task accuracy trajectory exactly once.
        if "accur" not in normalized_name or "matrix" in normalized_name:
            continue

        scalar_values = _scalar_series(values)
        # Direct scalar summaries belong in summary.csv, not a task trajectory.
        if len(scalar_values) == 1 and _numeric_scalar(values) is not _MISSING:
            continue

        for task_index, value in scalar_values:
            task_classes = list(groups[task_index]) if task_index < len(groups) \
                        else []
            rows.append({
                "task_index": task_index, 
                "phase": "continual", 
                "task_classes": _json_cell(task_classes), 
                "seen_classes": _json_cell(_seen_classes(groups, task_index)), 
                "metric": str(name), 
                "value": value
            })

    return rows


def _matrix_rows(
    details: Mapping[str, object], 
    groups: Sequence[Sequence[object]]
) -> list[dict[str, object]]:
    """Build long-form cells for every available accuracy matrix.

    Args:
        details (Mapping[str, object]): Continual learner detail mapping.
        groups (Sequence[Sequence[object]]): Original-label task groups.

    Returns:
        list[dict[str, object]]: One row per scalar matrix cell, including NaN
        placeholders for unavailable future-task evaluations.
    """

    rows = []
    for matrix_name, matrix in _accuracy_matrices(details):
        matrix = _python_value(matrix)
        # Ignore malformed scalar/text matrix values without failing reporting.
        if not isinstance(matrix, Sequence) or isinstance(
            matrix, (str, bytes, bytearray)
        ):
            continue

        for after_task_index, matrix_row in enumerate(matrix):
            matrix_row = _python_value(matrix_row)
            # Skip malformed rows while retaining every other valid matrix row.
            if not isinstance(matrix_row, Sequence) or isinstance(
                matrix_row, (str, bytes, bytearray)
            ):
                continue

            for evaluated_task_index, value in enumerate(matrix_row):
                scalar = _numeric_scalar(value)
                # Accuracy matrix cells must be scalar to enter long-form data.
                if scalar is _MISSING:
                    continue

                after_classes = list(groups[after_task_index]) if after_task_index < len(groups) \
                                else []
                evaluated_classes = list(groups[evaluated_task_index]) if evaluated_task_index < len(groups) \
                                else []
                rows.append({
                    "matrix": matrix_name, 
                    "after_task_index": after_task_index, 
                    "evaluated_task_index": evaluated_task_index, 
                    "after_task_classes": _json_cell(after_classes), 
                    "evaluated_task_classes": _json_cell(evaluated_classes), 
                    "seen_classes": _json_cell(
                        _seen_classes(groups, after_task_index)
                    ), 
                    "value": scalar
                })

    return rows


def _schedule_rows(
    details: Mapping[str, object], 
    groups: Sequence[Sequence[object]]
) -> list[dict[str, object]]:
    """Build one schedule row per represented task group.

    Args:
        details (Mapping[str, object]): Continual learner detail mapping.
        groups (Sequence[Sequence[object]]): Original-label task groups.

    Returns:
        list[dict[str, object]]: Task introduction and cumulative schedule rows.
    """

    class_order = details.get("class_order", [])
    seed = _numeric_scalar(details.get("seed"))
    # Use an empty cell when no scalar seed was recorded.
    if seed is _MISSING:
        seed = ""

    rows = []
    for task_index, task_classes in enumerate(groups):
        seen_classes = _seen_classes(groups, task_index)
        rows.append({
            "task_index": task_index, 
            "task_classes": _json_cell(task_classes), 
            "seen_classes": _json_cell(seen_classes), 
            "introduced_class_count": len(task_classes), 
            "seen_class_count": len(seen_classes), 
            "class_order": _json_cell(class_order), 
            "seed": seed
        })

    return rows


def _summary_rows(
    details: Mapping[str, object], 
    metadata: Mapping[str, object] | None
) -> list[dict[str, object]]:
    """Build key/value summary rows from scalar and schedule metadata.

    Args:
        details (Mapping[str, object]): Continual learner detail mapping.
        metadata (Mapping[str, object] | None): Optional run-level metadata to
            include ahead of learner-derived fields.

    Returns:
        list[dict[str, object]]: Stable source/key/value/type summary rows.
    """

    rows = []

    def append_value(source: str, key: str, value: object) -> None:
        """Append one supported scalar or structured summary value.

        Args:
            source (str): Metadata source namespace.
            key (str): Stable metadata key.
            value (object): Candidate value.

        Returns:
            None.
        """

        scalar = _numeric_scalar(value)

        # Store real scalars directly for straightforward numeric parsing.
        if scalar is not _MISSING:
            rows.append({
                "source": source, 
                "key": key, 
                "value": scalar, 
                "value_type": type(scalar).__name__
            })
            return

        normalized = _jsonable(value)

        # Skip live model/callback objects and other unstable representations.
        if normalized is _MISSING:
            return

        rows.append({
            "source": source, 
            "key": key, 
            "value": _json_cell(normalized), 
            "value_type": "json"
        })

    # Put explicit caller metadata first so run identifiers are easy to find.
    if metadata is not None:
        for key, value in metadata.items():
            append_value("metadata", str(key), value)

    for key in ("class_order", "task_classes", "seed"):
        # Retain only schedule metadata that the learner actually returned.
        if key in details:
            append_value("schedule", key, details[key])

    for source_name in (
        "continual_metrics", 
        "validation_continual_metrics"
    ):
        continual_metrics = details.get(source_name, {})

        # Flatten all scalar summaries without prescribing metric names.
        if isinstance(continual_metrics, Mapping):
            for key, value in _flatten_scalar_mapping(continual_metrics):
                append_value(source_name, key, value)

    reserved = {
        "class_order", "task_classes", "task_groups", "seed",
        "continual_metrics", "validation_continual_metrics", "histories",
        "generative_histories", "classifier_evaluations", "generative_evaluations", 
        "model", "generative_model", "task_resource_metrics",
        "task_mechanistic_metrics", "resource_metrics", "mechanistic_metrics",
    }

    for key, value in details.items():
        normalized_key = str(key)

        # Preserve extra scalar run metadata while avoiding task trajectories.
        if normalized_key in reserved or "accuracy_matrix" in normalized_key:
            continue

        scalar = _numeric_scalar(value)
        # Add only scalar extra details to the compact summary table.
        if scalar is not _MISSING:
            append_value("details", normalized_key, scalar)

    append_value("artifacts", "accuracy_matrices", [
        name for name, _ in _accuracy_matrices(details)
    ])
    append_value("schedule", "task_count", _task_count(details))

    return rows


def write_continual_csv_artifacts(
    details: Mapping[str, object], 
    output_dir: str | Path, 
    *, 
    metadata: Mapping[str, object] | None = None
) -> dict[str, Path]:
    """Write stable long-form CSV artifacts from continual learner details.

    Args:
        details (Mapping[str, object]): Mapping returned by the continual
            learner. Every documented field is optional for reporting.
        output_dir (str | pathlib.Path): Destination directory, created with
            parents when needed.
        metadata (Mapping[str, object] | None): Optional run identifiers and
            other JSON-safe values added to ``summary.csv``.

    Returns:
        dict[str, pathlib.Path]: Artifact names ``epoch_metrics``,
        ``task_metrics``, ``accuracy_matrices``, ``schedule``, and ``summary``
        mapped to their written CSV paths.

    Raises:
        TypeError: If ``details`` or non-None ``metadata`` is not a mapping.
        OSError: If the destination cannot be created or written.
    """

    # Require mapping semantics before any directories or files are created.
    if not isinstance(details, Mapping):
        raise TypeError("details must be a mapping.")
    # Apply the same stable-mapping requirement to optional run metadata.
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    groups = _task_groups(details)
    paths = {
        "epoch_metrics": output_path / "epoch_metrics.csv", 
        "task_metrics": output_path / "task_metrics.csv", 
        "accuracy_matrices": output_path / "accuracy_matrices.csv", 
        "schedule": output_path / "schedule.csv", 
        "summary": output_path / "summary.csv"
    }

    _write_csv(
        paths["epoch_metrics"], 
        (
            "task_index", "phase", "epoch", "task_classes", 
            "seen_classes", "metric", "value", 
        ), 
        _epoch_rows(details, groups)
    )
    _write_csv(
        paths["task_metrics"], 
        (
            "task_index", "phase", "task_classes", 
            "seen_classes", "metric", "value"
        ),
        _task_metric_rows(details, groups)
    )
    _write_csv(
        paths["accuracy_matrices"], 
        (
            "matrix", "after_task_index", "evaluated_task_index", 
            "after_task_classes", "evaluated_task_classes", 
            "seen_classes", "value"
        ), 
        _matrix_rows(details, groups)
    )
    _write_csv(
        paths["schedule"], 
        (
            "task_index", "task_classes", "seen_classes", 
            "introduced_class_count", "seen_class_count", 
            "class_order", "seed"
        ), 
        _schedule_rows(details, groups)
    )
    _write_csv(
        paths["summary"], 
        ("source", "key", "value", "value_type"), 
        _summary_rows(details, metadata)
    )

    return paths


def _tag_segment(value: object, fallback: str = "unknown") -> str:
    """Return one TensorBoard-safe namespace segment.

    Args:
        value (object): Text-like namespace value.
        fallback (str): Segment used when sanitization removes every character.

    Returns:
        str: Nonempty segment containing alphanumerics, dots, underscores, or
        hyphens.
    """

    segment = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")

    return segment or fallback


def _class_segment(classes: Sequence[object]) -> str:
    """Encode one task's original classes for a TensorBoard namespace.

    Args:
        classes (Sequence[object]): Original labels introduced by a task.

    Returns:
        str: Hyphen-separated safe class labels, or ``"unknown"``.
    """

    # Preserve schedule order so tag namespaces identify the exact task group.
    if classes:
        return "-".join(_tag_segment(label) for label in classes)

    return "unknown"


def _metric_tag(metric: object) -> str:
    """Sanitize a possibly nested metric path without losing hierarchy.

    Args:
        metric (object): Metric name or slash-delimited evaluation path.

    Returns:
        str: Slash-delimited safe TensorBoard metric path.
    """

    return "/".join(
        _tag_segment(part)
        for part in str(metric).split("/")
    )


def write_continual_tensorboard_summaries(
    details: Mapping[str, object], 
    log_dir: str | Path, 
    *, 
    start_step: int = 0
) -> dict[str, int]:
    """Write task/phase-separated TensorBoard scalar summaries.

    Each writer lives under ``task-NNN_classes-A-B/<phase>``. Scalar tags also
    contain ``task_NNN/classes_A-B/<phase>`` so exported event data retains the
    original task labels even when files are moved or merged. Every scalar
    history observation is written; nested evaluation scalars are written below
    an ``evaluation`` tag at the phase's next epoch step. Optional resource and
    mechanistic mappings use their own phase writers and are recursively
    flattened below the same class-identifying namespace.

    Args:
        details (Mapping[str, object]): Continual learner details with optional
            histories, evaluations, task classes, and per-task accuracies.
        log_dir (str | pathlib.Path): TensorBoard root directory.
        start_step (int): Nonnegative offset applied to each phase's epoch
            indices. Writers are already task/phase separated, so the default
            zero is unambiguous.

    Returns:
        dict[str, int]: Relative task/phase writer directories mapped to the
        number of scalar summaries written there.

    Raises:
        TypeError: If ``details`` is not a mapping or ``start_step`` is not a
            non-boolean integer.
        ValueError: If ``start_step`` is negative.
        ImportError: If TensorFlow is unavailable.
    """

    # Validate inputs before importing TensorFlow or creating event files.
    if not isinstance(details, Mapping):
        raise TypeError("details must be a mapping.")
    # Reject booleans and non-integral TensorBoard step offsets.
    if isinstance(start_step, bool) or not isinstance(start_step, Integral):
        raise TypeError("start_step must be a non-boolean integer.")
    # Keep every emitted TensorBoard step nonnegative.
    if start_step < 0:
        raise ValueError("start_step must be nonnegative.")

    import tensorflow as tf


    root = Path(log_dir)
    root.mkdir(parents=True, exist_ok=True)
    groups = _task_groups(details)
    writer_counts: dict[str, int] = {}

    histories_by_phase = _histories_by_phase(details)
    evaluations_by_phase = {
        phase: _task_mappings(details.get(source_name))
        for phase, source_name in _EVALUATION_SOURCES
    }
    diagnostics_by_phase = dict(_task_diagnostic_mappings(details))

    task_count = _task_count(details)
    for task_index in range(task_count):
        task_classes = list(groups[task_index]) if task_index < len(groups) else []
        class_namespace = _class_segment(task_classes)
        seen_classes = _seen_classes(groups, task_index)

        for phase in ("classifier", "generator"):
            histories = histories_by_phase[phase]
            evaluations = evaluations_by_phase[phase]
            history = histories[task_index] if task_index < len(histories) else {}
            evaluation = evaluations[task_index] if task_index < len(evaluations) else {}

            # Skip phases absent from both training history and evaluation data.
            if not history and not evaluation:
                continue

            relative_path = Path(
                f"task-{task_index:03d}_classes-{class_namespace}"
            ) / phase
            writer = tf.summary.create_file_writer(str(root / relative_path))
            namespace = (
                f"task_{task_index:03d}/classes_{class_namespace}/{phase}"
            )
            scalar_count = 0
            final_epoch_step = start_step

            try:
                with writer.as_default():
                    tf.summary.text(
                        f"{namespace}/task_classes", 
                        _json_cell(task_classes), 
                        step=start_step
                    )
                    tf.summary.text(
                        f"{namespace}/seen_classes", 
                        _json_cell(seen_classes), 
                        step=start_step
                    )

                    for metric, values in history.items():
                        for epoch, value in _scalar_series(values):
                            step = start_step + epoch
                            tf.summary.scalar(
                                f"{namespace}/{_metric_tag(metric)}", 
                                value, 
                                step=step
                            )
                            scalar_count += 1
                            final_epoch_step = max(final_epoch_step, step + 1)

                    for metric, value in _flatten_scalar_mapping(evaluation):
                        tf.summary.scalar(
                            f"{namespace}/evaluation/{_metric_tag(metric)}", 
                            value, 
                            step=final_epoch_step
                        )
                        scalar_count += 1

                writer.flush()
            finally:
                # Bound writer/file-descriptor use for long task streams and HPO.
                writer.close()

            writer_counts[relative_path.as_posix()] = scalar_count

        # Write non-epoch resource and mechanistic diagnostics per task.
        for phase, task_diagnostics in diagnostics_by_phase.items():
            diagnostics = task_diagnostics[task_index] if task_index < len(task_diagnostics) \
                        else {}
            flattened_diagnostics = _flatten_scalar_mapping(diagnostics)

            # Avoid event files for absent, empty, or wholly structured data.
            if not flattened_diagnostics:
                continue

            relative_path = Path(
                f"task-{task_index:03d}_classes-{class_namespace}"
            ) / phase
            writer = tf.summary.create_file_writer(str(root / relative_path))
            namespace = (
                f"task_{task_index:03d}/classes_{class_namespace}/{phase}"
            )

            try:
                with writer.as_default():
                    tf.summary.text(
                        f"{namespace}/task_classes", 
                        _json_cell(task_classes), 
                        step=start_step
                    )
                    tf.summary.text(
                        f"{namespace}/seen_classes", 
                        _json_cell(seen_classes), 
                        step=start_step
                    )
                    for metric, value in flattened_diagnostics:
                        tf.summary.scalar(
                            f"{namespace}/{_metric_tag(metric)}", 
                            value, 
                            step=start_step
                        )

                writer.flush()
            finally:
                # Close each diagnostic writer before advancing to the next task.
                writer.close()

            writer_counts[relative_path.as_posix()] = len(
                flattened_diagnostics
            )

        task_series = []
        for name, values in details.items():
            normalized_name = str(name).lower()

            # Collect top-level per-task accuracy values for a separate phase.
            if "accur" not in normalized_name or "matrix" in normalized_name:
                continue

            # Keep global scalar summaries out of per-task TensorBoard series.
            if _numeric_scalar(values) is not _MISSING:
                continue

            scalar_values = dict(_scalar_series(values))

            # Retain this task's scalar when the trajectory covers the task.
            if task_index in scalar_values:
                task_series.append((str(name), scalar_values[task_index]))

        # Avoid empty continual writers when no task-level values were returned.
        if not task_series:
            continue

        relative_path = Path(
            f"task-{task_index:03d}_classes-{class_namespace}"
        ) / "continual"
        writer = tf.summary.create_file_writer(str(root / relative_path))
        namespace = (
            f"task_{task_index:03d}/classes_{class_namespace}/continual"
        )

        try:
            with writer.as_default():
                tf.summary.text(
                    f"{namespace}/task_classes", 
                    _json_cell(task_classes), 
                    step=start_step
                )
                tf.summary.text(
                    f"{namespace}/seen_classes", 
                    _json_cell(seen_classes), 
                    step=start_step
                )
                for metric, value in task_series:
                    tf.summary.scalar(
                        f"{namespace}/{_metric_tag(metric)}", 
                        value, 
                        step=start_step
                    )

            writer.flush()
        finally:
            # Close the per-task continual writer after its scalar batch.
            writer.close()

        writer_counts[relative_path.as_posix()] = len(task_series)

    # Final CL summaries are mappings rather than per-task series, so write
    # them once in a dedicated namespace after every task has completed.
    final_metrics = []
    for source_name in (
        "continual_metrics", 
        "validation_continual_metrics"
    ):
        source = details.get(source_name, {})

        # Include only structured final metric collections in the summary.
        if isinstance(source, Mapping):
            final_metrics.extend(
                (source_name, metric, value)
                for metric, value in _flatten_scalar_mapping(source)
            )

    # Create the final writer only when at least one scalar metric exists.
    if final_metrics:
        relative_path = Path("final") / "continual"
        writer = tf.summary.create_file_writer(str(root / relative_path))
        try:
            with writer.as_default():
                tf.summary.text(
                    "continual/final/class_order", 
                    _json_cell(details.get("class_order", [])), 
                    step=start_step + task_count
                )
                for source_name, metric, value in final_metrics:
                    tf.summary.scalar(
                        "continual/final/"
                        f"{_tag_segment(source_name)}/{_metric_tag(metric)}", 
                        value, 
                        step=start_step + task_count
                    )

            writer.flush()
        finally:
            # Finalize the summary event file before returning to the caller.
            writer.close()

        writer_counts[relative_path.as_posix()] = len(final_metrics)

    return writer_counts


__all__ = [
    "write_continual_csv_artifacts", 
    "write_continual_tensorboard_summaries"
]
