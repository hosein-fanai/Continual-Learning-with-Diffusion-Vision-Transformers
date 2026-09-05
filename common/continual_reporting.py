"""Compute continual metrics and export learner details as CSV and TensorBoard.

The input contract is the mapping returned by the continual learner with
``return_details=True``. Accuracy matrices use rows for completed training
tasks and columns for evaluated tasks. Missing observations remain NaN;
task-balanced summaries never substitute training-history values for missing
held-out scores. Reporting can consume ordinary or ensemble accuracy matrices
without knowing how their predictions were computed.

CSV output separates epoch histories, task evaluations/diagnostics, individual
matrix cells, the original-label schedule, and final summaries. TensorBoard
uses task/class/phase namespaces and a separate final-summary namespace.
Optional detail fields may be absent: schemas remain stable, unsupported live
objects are omitted, and missing phases produce no fabricated measurements.
The metric/CSV helpers require NumPy but do not import training orchestration;
TensorFlow is imported only when TensorBoard events are requested."""

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
    """Extract available numeric observations without changing missing-data semantics.

    Args:
        values (Sequence[float] | numpy.ndarray): Numeric scalar sequence or
            array. NaN denotes an unavailable observation; infinities are retained.

    Returns:
        numpy.ndarray: One-dimensional float64 array containing every non-NaN
        element in NumPy traversal order. Empty/all-NaN input returns an empty array.

    Raises:
        TypeError: If NumPy cannot convert an input element to float64.
        ValueError: If the input cannot form a numeric array."""

    array = np.asarray(values, dtype="float64")
    return array[~np.isnan(array)]


def observed_mean(values: Sequence[float] | np.ndarray) -> float:
    """Average available observations while allowing an entirely unavailable series.

    Args:
        values (Sequence[float] | numpy.ndarray): Numeric observations. NaNs are
            excluded, but zeros, negative values, and infinities are retained.

    Returns:
        float: Arithmetic mean of non-NaN elements, or NaN for empty/all-NaN input.
        The empty case avoids NumPy's empty-slice warning. No sample weighting or
        clipping is applied.

    Raises:
        TypeError: If an element cannot be converted to a real numeric value.
        ValueError: If NumPy cannot construct the numeric input array."""

    values = _observed(values)
    # Average observed values when present; an entirely unavailable series stays NaN.
    return float(np.mean(values)) if values.size else float("nan")


def observed_max(values: Sequence[float] | np.ndarray) -> float:
    """Find the largest available observation without inventing an empty-series value.

    Args:
        values (Sequence[float] | numpy.ndarray): Numeric observations with NaN
            marking missing values. Infinities and signed values remain eligible.

    Returns:
        float: Largest non-NaN element, or NaN when no observation is available.
        This permits signed forgetting when final accuracy exceeds every prior peak.

    Raises:
        TypeError: If an element cannot be converted to a real numeric value.
        ValueError: If NumPy cannot construct the numeric input array."""

    values = _observed(values)
    # Take the maximum observed value when present; an unavailable series stays NaN.
    return float(np.max(values)) if values.size else float("nan")


def continual_metrics(
    accuracy_matrix: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Compute task-balanced accuracy, signed forgetting, and backward transfer.

    For a T-task matrix A, A[i, j] is task j's accuracy after training task i.
    The final average uses A[T-1, :T]; incremental accuracy averages the mean of
    each learned prefix A[i, :i+1]. For old task j, forgetting is its maximum
    observed score from rows j through T-2 minus its final score. Backward transfer
    is its final score minus A[j, j]. Average these differences across old tasks.
    Forgetting is signed: improvement beyond the previous peak gives a negative
    value, and positive backward transfer indicates improvement after acquisition.

    Args:
        accuracy_matrix (Sequence[Sequence[float]]): Square T-by-T ordinary or
            ensemble accuracy matrix in a consistent scale, normally [0, 1].
            Future-task and otherwise unavailable cells should be NaN. Empty input
            is supported. Values are converted to float64; shape is caller-owned.

    Returns:
        dict[str, float]: ``final_average_accuracy``,
        ``average_incremental_accuracy``, ``average_forgetting``, and
        ``backward_transfer``. Means and prior maxima ignore NaN observations;
        differences missing either operand remain unavailable. Empty input returns
        four NaNs. A single task has zero forgetting and backward transfer because
        there are no old tasks; its accuracy summaries can still be NaN.

    Raises:
        ValueError: If the input cannot be converted to a rectangular numeric array.
        IndexError: If nonempty input lacks the expected task rows or columns."""

    matrix = np.asarray(accuracy_matrix, dtype="float64")
    task_num = len(matrix)
    # An empty task stream has no defined continual accuracy or transfer metrics.
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
    # A single task has no preceding tasks to forget or transfer knowledge to.
    if task_num == 1:
        average_forgetting = 0.
        backward_transfer = 0.
    # Multiple tasks compare final old-task accuracy with prior peaks and acquisition scores.
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
    """Extract newly learned and previously learned task accuracy trajectories.

    Args:
        accuracy_matrix (Sequence[Sequence[float]]): T-by-T continual matrix whose
            diagonal records acquisition accuracy and whose row prefixes record
            previously introduced tasks. Missing scores should be NaN.

    Returns:
        tuple[list[float], list[float]]: ``(new_task_accuracy, old_task_accuracy)``.
        The first list contains each diagonal cell. The second contains NaN at
        task zero and the non-NaN mean of earlier task columns for later rows.
        Each task receives equal weight regardless of its evaluation sample count.
        Empty input returns two empty lists.

    Raises:
        IndexError: If a row lacks its expected diagonal cell.
        TypeError: If an observed cell cannot be converted to a real number."""

    new_task_accuracy = [
        float(accuracy_matrix[index][index])
        for index in range(len(accuracy_matrix))
    ]
    # The first task has no old-task accuracy; later tasks average their prior-task columns.
    old_task_accuracy = [
        np.nan if index == 0 else observed_mean(accuracy_matrix[index][:index])
        for index in range(len(accuracy_matrix))
    ]
    return new_task_accuracy, old_task_accuracy


def _python_value(value: object) -> object:
    """Materialize one tensor/NumPy value into a Python scalar or container.

    Args:
        value (object): Candidate scientific value. Objects with a callable
            ``numpy`` method are materialized first; NumPy arrays use ``tolist``
            and NumPy scalar objects use ``item``. Other objects pass through.

    Returns:
        object: Converted scalar/list or the original unsupported object. This
        helper does not recursively normalize arbitrary mappings or nested Python
        containers; recursive callers perform that work themselves."""

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
    """Distinguish real scalar measurements from structured or unsupported values.

    Args:
        value (object): Candidate metric or metadata value. Eager tensors and
            NumPy values are first converted by :func:`_python_value`.

    Returns:
        bool | int | float | object: A Python scalar preserving Boolean/integer
        categories, or float for other real numbers. Non-scalar and non-real values
        return the private ``_MISSING`` sentinel. False, zero, NaN, and infinity
        remain valid scalar results and are never confused with missing structure."""

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
    """Extract original indices and real scalars from a history-like value.

    Args:
        value (object): One real scalar, a Python sequence, or a tensor/array-like
            one-dimensional series. Mapping/text inputs are metadata and contribute
            no observations. Non-Sequence array-like inputs must have rank one.

    Returns:
        list[tuple[int, bool | int | float]]: ``(index, scalar)`` observations.
        A direct scalar becomes ``[(0, value)]``. Unsupported sequence entries are
        skipped without renumbering later entries; matrices and wholly structured
        sequences contribute no scalar observations. Empty input returns ``[]``."""

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
    """Normalize optional per-task dictionaries while preserving task alignment.

    Args:
        value (object): A direct history/evaluation mapping, a sequence of optional
            mappings, or an absent/unsupported source. Text is not a task sequence.

    Returns:
        list[Mapping[str, object]]: A new outer list. One direct mapping becomes
        one task; sequence entries that are not mappings become empty dictionaries.
        Missing/unsupported sources produce ``[]``. Valid mappings are referenced
        rather than deep-copied, and this helper does not mutate them."""

    # A direct history/evaluation mapping represents one task.
    if isinstance(value, Mapping):
        return [value]

    # Reject text and non-sequence values before task normalization.
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        return []

    # Keep task mappings and replace absent or malformed task entries with empty mappings.
    return [item if isinstance(item, Mapping) else {} for item in value]


def _is_classifier_history_metric(name: object) -> bool:
    """Identify classifier/distillation keys inside a duplicated joint history.

    Args:
        name (object): History key converted to lowercase text. A leading ``val_``
            is removed so validation and training versions belong to the same phase.

    Returns:
        bool: Whether the normalized name occurs in the module's explicit
        classifier/KD metric inventory. Unknown metric names remain on the generator
        side when a joint history is partitioned."""

    normalized = str(name).lower()
    # Validation metrics retain the phase of their unprefixed counterpart.
    if normalized.startswith("val_"):
        normalized = normalized[4:]

    return normalized in _CLASSIFIER_HISTORY_NAMES


def _same_history(
    first: Mapping[str, object],
    second: Mapping[str, object]
) -> bool:
    """Compare two scientific history mappings without ambiguous array truth tests.

    Args:
        first (Mapping[str, object]): Classifier-side candidate history, normally
            metric names mapped to epoch trajectories.
        second (Mapping[str, object]): Generator-side candidate history to compare.

    Returns:
        bool: True for the same mapping object, or for equal key sets whose compact
        supported JSON representations match. Different keys return False early.
        Equality follows serialization order and supported-value conversion; it is
        not a general numeric tolerance comparison or equality of live objects."""

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
    """Partition duplicate joint histories while retaining independent phase fits.

    The learner may expose one joint fit under both legacy history keys. When
    both nonempty mappings match, classifier/KD metric names go to ``classifier``
    and remaining metrics go to ``generator``. Independent histories remain intact.

    Args:
        details (Mapping[str, object]): Learner details with optional ``histories``
            and ``generative_histories`` mappings or per-task sequences. Missing
            tasks are represented by empty mappings up to the longer source length.

    Returns:
        dict[str, list[Mapping[str, object]]]: Task-aligned ``classifier`` and
        ``generator`` histories. The original detail mappings are not mutated;
        partitioned histories are new dictionaries."""

    classifier = _task_mappings(details.get("histories"))
    generator = _task_mappings(details.get("generative_histories"))
    task_count = max(len(classifier), len(generator))
    resolved: dict[str, list[Mapping[str, object]]] = {
        "classifier": [],
        "generator": []
    }
    for task_index in range(task_count):
        # Use the classifier history at this task index, or an empty mapping when absent.
        classifier_history = classifier[task_index] if task_index < len(classifier) \
                            else {}
        # Use the generator history at this task index, or an empty mapping when absent.
        generator_history = generator[task_index] if task_index < len(generator) \
                            else {}

        # Split only the known duplicate produced by one joint Keras fit.
        if classifier_history and generator_history and _same_history(
            classifier_history, generator_history
        ):
            # Keep only classifier and distillation metrics in the classifier phase.
            resolved["classifier"].append({
                name: values
                for name, values in classifier_history.items()
                if _is_classifier_history_metric(name)
            })
            # Keep the remaining joint-history metrics in the generator phase.
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
    """Resolve task resource/mechanistic diagnostics and their legacy aliases.

    Args:
        details (Mapping[str, object]): Learner details containing optional
            ``task_resource_metrics`` and ``task_mechanistic_metrics``. The shorter
            aliases ``resource_metrics`` and ``mechanistic_metrics`` are used only
            when the corresponding canonical key is absent, not when it is empty.

    Returns:
        list[tuple[str, list[Mapping[str, object]]]]: Ordered ``resource`` and
        ``mechanistic`` phase entries, each with normalized per-task mappings.
        An absent source contributes an empty list; malformed task entries retain
        their position as empty mappings."""

    resolved = []
    for phase, canonical_name, alias_name in _TASK_DIAGNOSTIC_SOURCES:
        source = details.get(
            canonical_name,
            details.get(alias_name, _MISSING)
        )
        # Missing diagnostic sources contribute no tasks; supplied sources retain task
        # alignment.
        mappings = [] if source is _MISSING else _task_mappings(source)
        resolved.append((phase, mappings))

    return resolved


def _task_groups(details: Mapping[str, object]) -> list[list[object]]:
    """Read original-label task groups for report rows and event namespaces.

    Args:
        details (Mapping[str, object]): Learner details. ``task_classes`` takes
            precedence over ``task_groups`` whenever present. The outer value must
            be a non-text Sequence; individual NumPy/tensor groups are converted.

    Returns:
        list[list[object]]: New outer and group lists in schedule order. Unsupported
        or missing outer metadata returns ``[]``; malformed individual groups become
        empty lists so later task indices retain their alignment."""

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
    """Flatten the original-label schedule through an inclusive task index.

    Args:
        groups (Sequence[Sequence[object]]): Ordered groups of classes introduced
            by each task. No sorting or deduplication is performed.
        index (int): Inclusive zero-based task index used in ``groups[:index + 1]``.
            An index beyond the final task includes all available groups.

    Returns:
        list[object]: Classes introduced through the selected task in group order.
        Empty groups produce an empty list, and input lists are not modified."""

    return [
        label
        for group in groups[:index + 1]
        for label in group
    ]


def _jsonable(value: object) -> object:
    """Recursively retain scientific metadata that can be serialized predictably.

    Args:
        value (object): Real scalar, text, None, eager tensor, NumPy value, mapping,
            or non-binary sequence. Mapping keys are converted to strings. Live
            models, callbacks, binary data, and other unsupported objects are omitted.

    Returns:
        object: Python scalar/container or the private ``_MISSING`` sentinel for
        unsupported top-level values. Unsupported mapping items/sequence elements
        are dropped recursively. NaN and infinity remain real values; callers
        decide how to encode them. Input containers are not mutated."""

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
    """Encode supported metadata as a compact CSV-compatible JSON cell.

    Args:
        value (object): Metadata normalized recursively by :func:`_jsonable`.
            None is valid JSON null; unsupported live objects are not serialized.

    Returns:
        str: Compact Unicode JSON with insertion order preserved and NaN/infinity
        tokens allowed. An unsupported top-level value returns an empty string.
        This is the permissive report-cell format, not strict interoperable JSON
        for immutable experiment manifests."""

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
    """Convert nested evaluation values into slash-delimited scalar metric paths.

    Args:
        value (object): Nested mapping, real scalar, or scalar sequence. Scientific
            tensor/NumPy values are normalized; unsupported leaves are skipped.
        prefix (str): Parent metric path. Default ``""`` starts at the root. Mapping
            children append their names, and scalar sequence entries append indices.

    Returns:
        list[tuple[str, bool | int | float]]: Metric path/value pairs in input order.
        A root scalar uses the name ``value``. Root scalar sequences use numeric
        index names; unsupported or empty structures return an empty list."""

    scalar = _numeric_scalar(value)
    # Emit a scalar leaf under its complete source path.
    if scalar is not _MISSING:
        return [(prefix or "value", scalar)]

    value = _python_value(value)
    # Recurse through named evaluation mappings.
    if isinstance(value, Mapping):
        flattened = []
        for name, item in value.items():
            # Append child names to an existing metric path; root children use their name
            # alone.
            child_prefix = f"{prefix}/{name}" if prefix else str(name)
            flattened.extend(_flatten_scalar_mapping(item, child_prefix))

        return flattened

    series = _scalar_series(value)

    # Preserve scalar list results such as ordinary Keras ``evaluate`` output.
    if series:
        # Prefix indexed scalar values when a parent path exists; root sequences use bare
        # indices.
        return [
            (f"{prefix}/{index}" if prefix else str(index), scalar_value)
            for index, scalar_value in series
        ]

    return []


def _task_count(details: Mapping[str, object]) -> int:
    """Infer how many task observations the supplied reporting sources represent.

    Args:
        details (Mapping[str, object]): Optional schedule, phase histories and
            evaluations, diagnostics, top-level accuracy trajectories, and accuracy
            matrices. Direct scalar accuracy summaries do not establish task count.

    Returns:
        int: Maximum source length, counting normalized task mappings, available
        scalar-series observations, and matrix rows. Missing sources yield zero.
        This reflects represented report data and does not prove that every task
        completed training or that malformed sparse sources contain every index."""

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
    """Discover named accuracy matrices without prescribing a fixed inventory.

    Args:
        details (Mapping[str, object]): Learner detail mapping. Top-level names
            containing ``accuracy_matrix`` are recognized case-insensitively.
            An ``accuracy_matrices`` mapping additionally exposes its child names.

    Returns:
        list[tuple[str, object]]: Matrix name/value pairs in encounter order, keeping
        the first occurrence of each exact string name. Values are not validated or
        copied here; row builders decide which scalar cells can be reported."""

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
    """Replace one CSV artifact through a sibling temporary file.

    Args:
        path (pathlib.Path): Destination CSV path whose parent directory already
            exists. The temporary file appends ``.tmp`` to its existing suffix.
        fieldnames (Sequence[str]): Ordered stable header columns.
        rows (Sequence[Mapping[str, object]]): Row mappings accepted by DictWriter.
            Empty input still writes the header; missing fields become empty cells.

    Returns:
        None: Writes UTF-8 CSV and replaces the destination after writing finishes.
        Each file is replaced independently; this is not a transaction spanning
        every artifact, nor a concurrent-writer lock.

    Raises:
        OSError: If temporary-file creation, writing, or replacement fails.
        ValueError: If a row contains keys outside ``fieldnames``."""

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
    """Build one long-form row for each observed phase/epoch metric value.

    Args:
        details (Mapping[str, object]): Learner details whose classifier/generator
            histories are normalized and duplicate joint histories are partitioned.
        groups (Sequence[Sequence[object]]): Original-label task groups. Missing
            schedule entries use empty class lists without dropping metric rows.

    Returns:
        list[dict[str, object]]: Rows with zero-based ``task_index`` and ``epoch``,
        ``phase``, JSON ``task_classes``/``seen_classes``, ``metric``, and scalar
        ``value``. Structured observations are skipped; no files are written."""

    rows = []
    for phase, histories in _histories_by_phase(details).items():
        for task_index, history in enumerate(histories):
            # Attach known task classes; missing schedule metadata uses an empty class list.
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
    """Build task evaluation, diagnostic, and accuracy-trajectory CSV rows.

    Args:
        details (Mapping[str, object]): Learner details with optional phase
            evaluations, nested resource/mechanistic diagnostics, and top-level
            accuracy trajectories. Scalar summaries and accuracy matrices are excluded.
        groups (Sequence[Sequence[object]]): Original-label groups used for each
            task's class metadata; missing entries produce empty class lists.

    Returns:
        list[dict[str, object]]: Rows containing ``task_index``, ``phase``, JSON
        ``task_classes``/``seen_classes``, slash-delimited ``metric``, and scalar
        ``value``. Evaluations and diagnostics retain their phase; top-level
        trajectories use ``continual``. Empty sources contribute no rows."""

    rows = []
    sources = [
        (phase, _task_mappings(details.get(source_name)))
        for phase, source_name in _EVALUATION_SOURCES
    ] + _task_diagnostic_mappings(details)
    for phase, task_metrics in sources:
        for task_index, metrics in enumerate(task_metrics):
            # Use available class-group metadata; leave classes empty for unscheduled metric
            # rows.
            task_classes = list(groups[task_index]) if task_index < len(groups) \
                        else []
            seen_classes = _seen_classes(groups, task_index)
            for metric, value in _flatten_scalar_mapping(metrics):
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
            # Use known classes for each accuracy observation, or an empty list without
            # schedule metadata.
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
    """Expand every supported accuracy-matrix cell into a long-form row.

    Args:
        details (Mapping[str, object]): Learner details containing named top-level
            or grouped accuracy matrices. Tensor/NumPy containers are normalized;
            malformed matrix containers, rows, and non-scalar cells are skipped.
        groups (Sequence[Sequence[object]]): Original-label task schedule for row,
            column, and cumulative class metadata. Missing groups remain empty.

    Returns:
        list[dict[str, object]]: Rows with ``matrix``, zero-based
        ``after_task_index``/``evaluated_task_index``, JSON class metadata, and
        scalar ``value``. NaN future-task cells are preserved rather than omitted,
        allowing consumers to reconstruct each represented matrix shape."""

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

                # Resolve classes learned at this matrix row, or leave missing schedule
                # metadata empty.
                after_classes = list(groups[after_task_index]) if after_task_index < len(groups) \
                                else []
                # Resolve classes evaluated by this matrix column, or leave missing metadata
                # empty.
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
    """Describe task introductions and cumulative class counts as schedule rows.

    Args:
        details (Mapping[str, object]): Learner details supplying optional
            ``class_order`` and scalar ``seed``. Missing class order becomes ``[]``;
            missing/non-scalar seed becomes an empty CSV cell.
        groups (Sequence[Sequence[object]]): Ordered original-label task groups.

    Returns:
        list[dict[str, object]]: One row per group, containing ``task_index``, JSON
        ``task_classes``, ``seen_classes``, and ``class_order``, numeric
        ``introduced_class_count``/``seen_class_count``, and ``seed``. Empty input
        groups produce no rows. This function does not validate schedule uniqueness."""

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
    """Collect final metrics, schedule information, and scalar run metadata.

    Args:
        details (Mapping[str, object]): Learner details. The ordinary and validation
            ``continual_metrics`` mappings are flattened separately; live models,
            histories, diagnostics, and matrix payloads are excluded from extra
            scalar metadata. Matrix names and inferred task count are always recorded.
        metadata (Mapping[str, object] | None): Additional caller metadata placed
            first under the ``metadata`` source. None adds no caller rows; keys may
            also occur under other source namespaces without overriding each other.

    Returns:
        list[dict[str, object]]: Rows with ``source``, ``key``, ``value``, and
        ``value_type``. Real scalars keep bool/int/float categories; other supported
        values use compact JSON and type ``json``. Unsupported objects are skipped."""

    rows = []

    def append_value(source: str, key: str, value: object) -> None:
        """Append one supported value to the enclosing summary row collection.

        Args:
            source (str): Namespace identifying metadata, schedule, final metrics, or
                artifact inventory. It is stored unchanged in the row.
            key (str): Stable field name within that namespace.
            value (object): Candidate scalar or structured metadata. Real scalars are
                stored directly; other supported values use compact JSON.

        Returns:
            None: Appends one row to the enclosing ``rows`` list, or appends nothing
            for an unsupported live object. The input value is not modified."""

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
    """Write the five stable long-form CSV artifacts for a continual run.

    Missing optional sources still produce parseable header-only files. Each file
    is replaced independently through a temporary sibling; existing report files
    may be refreshed. No learner state, model, or history is mutated.

    Args:
        details (Mapping[str, object]): Mapping returned by the continual learner.
            Histories, evaluations, task groups, matrices, diagnostics, and final
            metrics are optional and normalized by the corresponding row builders.
        output_dir (str | pathlib.Path): Destination directory, created recursively
            when necessary. Filenames are fixed independently of run metadata.
        metadata (Mapping[str, object] | None): Additional run identifiers/settings
            for ``summary.csv``. Default None adds no explicit metadata namespace.

    Returns:
        dict[str, pathlib.Path]: ``epoch_metrics``, ``task_metrics``,
        ``accuracy_matrices``, ``schedule``, and ``summary`` mapped to the matching
        ``<name>.csv`` paths. Epoch/task tables hold scalar observations, the matrix
        table holds individual cells, schedule rows describe original labels, and
        summary rows hold final metrics plus the matrix inventory and task count.

    Raises:
        OSError: If directories or CSV files cannot be created, written, or replaced."""

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
    """Sanitize one TensorBoard namespace component without retaining separators.

    Args:
        value (object): Value converted to text. Runs of characters outside ASCII
            letters, digits, dot, underscore, and hyphen become one hyphen.
        fallback (str): Text used if sanitization leaves no characters after
            trimming edge hyphens. Default ``"unknown"``. The fallback itself is
            returned unchanged rather than sanitized again.

    Returns:
        str: Sanitized segment, or ``fallback`` when the segment is empty."""

    segment = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")

    return segment or fallback


def _class_segment(classes: Sequence[object]) -> str:
    """Encode original task labels into a stable TensorBoard class namespace.

    Args:
        classes (Sequence[object]): Ordered class labels. Each label is independently
            sanitized by :func:`_tag_segment`; schedule order is preserved.

    Returns:
        str: Hyphen-joined label segments, or ``"unknown"`` for an empty task group.
        This is a readable display identifier, not a collision-proof encoding."""

    # Preserve schedule order so tag namespaces identify the exact task group.
    if classes:
        return "-".join(_tag_segment(label) for label in classes)

    return "unknown"


def _metric_tag(metric: object) -> str:
    """Sanitize each component of a hierarchical TensorBoard metric name.

    Args:
        metric (object): Name converted to text and split on ``/``. Slash-separated
            mapping paths retain their hierarchy; empty components use ``unknown``.

    Returns:
        str: Slash-joined sanitized components suitable for scalar event tags."""

    return "/".join(
        _tag_segment(part)
        for part in str(metric).split("/")
    )


def write_continual_tensorboard_summaries(
    details: Mapping[str, object],
    log_dir: str | Path,
    *,
    start_step: int = 0,
) -> dict[str, int]:
    """Write task/class/phase events and final continual metric summaries.

    Each task phase has its own ``task-NNN_classes-A-B/<phase>`` writer and matching
    ``task_NNN/classes_A-B/<phase>`` tag namespace. Joint histories are partitioned
    without duplicating classifier/KD metrics. Resource/mechanistic diagnostics
    and top-level accuracy trajectories have separate phases. Final ordinary and
    validation summaries are written under ``final/continual``.

    Args:
        details (Mapping[str, object]): Learner details with optional histories,
            evaluations, task groups, diagnostic mappings, accuracy trajectories,
            and ordinary/validation final metric dictionaries. Missing sources
            produce no fabricated scalar events.
        log_dir (str | pathlib.Path): Root event directory, created recursively.
            Repeated calls create additional event files rather than deleting old ones.
        start_step (int): Offset for each task/phase epoch series. Default 0.
            Histories use offset plus epoch index; evaluations use the step after
            the last scalar history observation, or the offset for an empty history.
            Diagnostics/trajectories use the offset and final summaries use offset
            plus represented task count. Nonnegative values are intended.

    Returns:
        dict[str, int]: Relative writer directories mapped to their scalar-event
        count. Task/class text events are excluded from counts. Writers are flushed
        and closed before returning, including when a write raises an exception.

    Raises:
        ImportError: If TensorFlow is unavailable.
        OSError: If the root directory cannot be created.
        tensorflow.errors.OpError: If TensorFlow cannot create or write event files."""
    import tensorflow as tf

    root = Path(log_dir)
    root.mkdir(parents=True, exist_ok=True)
    groups = _task_groups(details)
    counts = {}

    def write_events(relative_path, namespace, scalars, metadata, metadata_step):
        """Write and close one event stream, then record its scalar count.

        Args:
            relative_path (pathlib.Path): Writer directory relative to the enclosing
                TensorBoard root. Its POSIX spelling becomes the count-map key.
            namespace (str): Shared prefix for every text/scalar tag in this stream.
            scalars (Sequence[tuple[str, object, int]]): Metric name, real scalar value,
                and step tuples. Metric names are sanitized while preserving slash paths.
            metadata (Mapping[str, object]): Named task/class or final-order metadata
                encoded as compact JSON text events before scalar observations.
            metadata_step (int): Event step shared by this stream's text metadata.

        Returns:
            None: Creates event files below the enclosing ``root``, flushes/closes the
            writer, and updates enclosing ``counts`` after successful writing. Closing
            also occurs on failure; a failed stream does not receive a completed count.

        Raises:
            tensorflow.errors.OpError: If event writer creation or writing fails."""
        writer = tf.summary.create_file_writer(str(root / relative_path))
        try:
            with writer.as_default():
                for name, value in metadata.items():
                    tf.summary.text(
                        f"{namespace}/{name}", _json_cell(value), step=metadata_step,
                    )
                for name, value, step in scalars:
                    tf.summary.scalar(
                        f"{namespace}/{_metric_tag(name)}", value, step=step,
                    )
            writer.flush()
        finally:
            writer.close()
        counts[relative_path.as_posix()] = len(scalars)

    histories = _histories_by_phase(details)
    evaluations = {
        phase: _task_mappings(details.get(name))
        for phase, name in _EVALUATION_SOURCES
    }
    diagnostics = _task_diagnostic_mappings(details)
    # Collect accuracy trajectories only, excluding matrices and global scalar summaries.
    task_series = {
        str(name): dict(_scalar_series(values))
        for name, values in details.items()
        if "accur" in str(name).lower() and "matrix" not in str(name).lower()
        and _numeric_scalar(values) is _MISSING
    }
    task_count = _task_count(details)
    for index in range(task_count):
        # Use known task classes for event namespaces; missing task groups stay empty.
        classes = list(groups[index]) if index < len(groups) else []
        class_segment = _class_segment(classes)
        metadata = {"task_classes": classes, "seen_classes": _seen_classes(groups, index)}
        phase_scalars = {}
        for phase in histories:
            # Use this phase's task history when available; absent histories are empty
            # mappings.
            history = histories[phase][index] if index < len(histories[phase]) else {}
            # Use this phase's task evaluation when available; absent evaluations are empty
            # mappings.
            evaluation = evaluations[phase][index] if index < len(evaluations[phase]) else {}
            # Skip phases with neither training history nor evaluation output.
            if not history and not evaluation:
                continue
            scalars = [
                (str(metric), value, start_step + epoch)
                for metric, values in history.items()
                for epoch, value in _scalar_series(values)
            ]
            evaluation_step = max((step + 1 for _, _, step in scalars), default=start_step)
            scalars.extend(
                (f"evaluation/{metric}", value, evaluation_step)
                for metric, value in _flatten_scalar_mapping(evaluation)
            )
            phase_scalars[phase] = scalars

        for phase, tasks in diagnostics:
            # Use diagnostics at this task index, or an empty mapping when the source ends
            # earlier.
            task = tasks[index] if index < len(tasks) else {}
            scalars = [
                (metric, value, start_step)
                for metric, value in _flatten_scalar_mapping(task)
            ]
            # Create a diagnostic phase only when it contains scalar observations.
            if scalars:
                phase_scalars[phase] = scalars
        # Include only trajectories that actually contain an observation for this task.
        scalars = [
            (metric, values[index], start_step)
            for metric, values in task_series.items() if index in values
        ]
        # Create continual task events only when accuracy observations exist.
        if scalars:
            phase_scalars["continual"] = scalars

        for phase, scalars in phase_scalars.items():
            write_events(
                Path(f"task-{index:03d}_classes-{class_segment}") / phase,
                f"task_{index:03d}/classes_{class_segment}/{phase}",
                scalars, metadata, start_step,
            )

    final_scalars = []
    for name in ("continual_metrics", "validation_continual_metrics"):
        source = details.get(name, {})
        # Flatten final metric collections only when they are structured mappings.
        if isinstance(source, Mapping):
            final_scalars.extend(
                (f"{name}/{metric}", value, start_step + task_count)
                for metric, value in _flatten_scalar_mapping(source)
            )
    # Create the final summary writer only when at least one scalar summary exists.
    if final_scalars:
        write_events(
            Path("final") / "continual", "continual/final", final_scalars,
            {"class_order": details.get("class_order", [])}, start_step + task_count,
        )
    return counts


__all__ = [
    "write_continual_csv_artifacts",
    "write_continual_tensorboard_summaries"
]
