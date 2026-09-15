"""Saved-only, stream-first evidence package for the minimum Route One chapter.

The public entry authenticates the frozen campaign through workflow.py. Nothing
in this module constructs a model, loads a dataset or predicts new outcomes.
"""

from __future__ import annotations

import hashlib
import json
from numbers import Real
from pathlib import Path
from typing import Callable, TYPE_CHECKING

# Import annotation-only types without changing the runtime backend.
if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd

from common.continual_reporting import continual_metrics, task_accuracy_summaries
from common.experiment import materialize_run_plan


SEEDS = [1103, 2207, 3301]
METHODS = {"baseline": "Platform", "extra_joint": "Extra joint", "learned": "Learned",
           "random": "Random", "ce_only": "CE only"}
COLORS = {"baseline": "#555555", "extra_joint": "#0072B2", "learned": "#D55E00",
          "random": "#009E73", "ce_only": "#AA4499"}
MARKERS = {"baseline": "o", "extra_joint": "s", "learned": "D", "random": "^", "ce_only": "v"}
METRICS = {"final_average_accuracy": ("Final accuracy", "%"),
           "average_incremental_accuracy": ("Average incremental accuracy", "%"),
           "average_forgetting": ("Signed forgetting", "percentage points"),
           "backward_transfer": ("Backward transfer", "percentage points")}
PHASE_METRICS = {"clean_accuracy": ("Deployed all-seen accuracy", "%", 100),
                 "old_accuracy": ("Deployed old-class accuracy", "%", 100),
                 "new_accuracy": ("Deployed new-class accuracy", "%", 100),
                 "hidden_target_cosine": ("Predictor-free hidden-target cosine", "cosine", 1),
                 "hidden_infonce": ("Predictor-free hidden InfoNCE", "loss", 1),
                 "representation.centered_effective_rank": ("Hidden effective rank", "rank", 1),
                 "representation.mean_off_diagonal_cosine": ("Hidden off-diagonal cosine", "cosine", 1)}
KEYS = ["dataset", "condition", "method", "run_id", "block_id", "seed"]
LIMITATIONS = ("Three full training streams are the independent replicates; tasks, images, gates and "
    "minibatches are not additional replicates. SD describes stream variation, whereas the native "
    "paired 95% t interval describes primary-effect uncertainty under its assumptions. Small n gives "
    "weak precision; three pairs cannot establish the shape of the difference distribution. "
    "The two dataset-specific primary intervals are unadjusted for multiple comparisons. "
    "Equal observed paired differences produce a zero-width conventional t interval; that does "
    "not establish an exact population effect. Local backward transfer is final minus acquisition "
    "accuracy, not TMCL's separately trained single-task-reference BT. TMCL FT and CDNV are not "
    "reported here. This reduced local DiTClassifier/V1 experiment is neither an exact JDCL/TMCL "
    "reproduction nor evidence of convergence, biological fidelity or state-of-the-art performance. "
    "Semantic consolidation updates the classifier projection/head and temporary predictor; the "
    "denoising backbone is frozen during this phase. Eight-example per-class CKA is descriptive, "
    "not strong representation-preservation evidence. Cosine geometry and CKA are not TMCL's CDNV. "
    "Replay label agreement is learner self-consistency, not independent perceptual quality. "
    "Update-matched extra joint is not matched wall time, FLOPs or example presentations. "
    "Absence and nonfinite measurements remain unavailable; no favorable-run selection is allowed.")


def _read(path: str | Path) -> object:
    """Read the saved artifact.

    Args:
        path (str | Path): File to read or write; relative paths use the current working
            directory.

    Returns:
        saved (object): Decoded JSON evidence, leaving the source file unchanged.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If JSON decoding fails.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite(value: object) -> float:
    """Finite real scalar as a Python float; booleans, missing, unsupported and nonfinite values
    become float NaN.

    Args:
        value (object): JSON-like object or scalar. Missing and nonfinite values are handled as
            described in the return contract.

    Returns:
        scalar (float): Finite real scalar as a Python float; booleans, missing, unsupported and
            nonfinite values become float NaN.

    Raises:
        None: Arbitrary scalar inputs are treated as unavailable when unsupported.
    """
    # Normalize this supported value representation explicitly.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return np.nan
    return float(value) if np.isfinite(value) else np.nan


def _get(value: object, key: str) -> object:
    """Nested saved value, or None if a component is absent or not a dictionary.

    Args:
        value (object): JSON-like object or scalar. Missing and nonfinite values are handled as
            described in the return contract.
        key (str): Dot-separated nested dictionary key; a missing component produces None.

    Returns:
        selected (object): Nested saved value, or None if a component is absent or not a
            dictionary.

    Raises:
        None: Missing dictionary components are represented by None.
    """
    for part in key.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _clean(value: object) -> object:
    """Recursive JSON-compatible object; NumPy scalars become native scalars and missing/nonfinite
    values become None.

    Args:
        value (object): JSON-like object or scalar. Missing and nonfinite values are handled as
            described in the return contract.

    Returns:
        serializable (object): Recursive JSON-compatible object; NumPy scalars become native
            scalars and missing/nonfinite values become None.

    Raises:
        None: Unsupported object leaves are returned unchanged for the eventual serializer to
            validate.
    """
    # Normalize this supported value representation explicitly.
    if value is pd.NA or value is pd.NaT:
        return None
    # Normalize this supported value representation explicitly.
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    # Normalize this supported value representation explicitly.
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    # Normalize this supported value representation explicitly.
    if isinstance(value, np.generic):
        return _clean(value.item())
    # Keep finite measurements separate from unavailable values.
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json(path: str | Path, value: object) -> None:
    """Read the saved artifact.

    Args:
        path (str | Path): File to read or write; relative paths use the current working
            directory.
        value (object): JSON-like object or scalar. Missing and nonfinite values are handled as
            described in the return contract.

    Returns:
        written (None): None; serializes saved evidence with explicit null values for missing or
            nonfinite observations.

    Raises:
        TypeError: If a value is not JSON-compatible.
        OSError: If the output cannot be written.
    """
    Path(path).write_text(json.dumps(_clean(value), indent=2, allow_nan=False), encoding="utf-8")


def _hash(path: str | Path) -> str:
    """Hexadecimal SHA-256 calculated incrementally over exact file bytes.

    Args:
        path (str | Path): File to read or write; relative paths use the current working
            directory.

    Returns:
        sha256 (str): Hexadecimal SHA-256 calculated incrementally over exact file bytes.

    Raises:
        OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_cka(value: object) -> object:
    """Copy saved observations; never let legacy tiny/unknown cohorts imply CKA evidence.

    Args:
        value (object): JSON-like object or scalar. Missing and nonfinite values are handled as
            described in the return contract.

    Returns:
        sanitized (object): Copied saved observations. CKA without more than two actual aligned
            integer samples, a finite score or availability becomes None with a reason;
            independent drift fields remain unchanged.

    Raises:
        None: Non-container leaves are returned unchanged.
    """
    # Normalize this supported value representation explicitly.
    if isinstance(value, list):
        return [sanitize_cka(item) for item in value]
    # Normalize this supported value representation explicitly.
    if not isinstance(value, dict):
        return value
    result = {key: sanitize_cka(item) for key, item in value.items()}
    for field, count_field, reason_field in (
        ("linear_cka", "sample_count", "linear_cka_unavailable_reason"),
        ("hidden_feature_cka", "hidden_feature_cka_sample_count", "hidden_feature_cka_unavailable_reason"),
    ):
        # Leave absent CKA fields absent.
        if field not in result:
            continue
        count = result.get(count_field)
        reason = result.get(reason_field)
        # Normalize this supported value representation explicitly.
        if not isinstance(count, int) or isinstance(count, bool):
            reason = "actual_aligned_sample_count_unavailable"
        # Two or fewer aligned rows cannot support the reported CKA interpretation.
        elif count <= 2:
            reason = "fewer_than_three_aligned_observations"
        # Keep finite measurements separate from unavailable values.
        elif not np.isfinite(_finite(result[field])):
            reason = reason or "constant_missing_or_nonfinite_representation"
        # Keep unavailable CKA numeric values empty while retaining their reason.
        if reason:
            result[field] = None
        result[reason_field] = reason
    return result


def summarize_streams(rows: list[dict] | pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Mean, sample SD (ddof=1), n; rows must already be one value per stream/group.

    Args:
        rows (list[dict] | pd.DataFrame): Saved observation records or equivalent DataFrame;
            repeated measurements must be reduced before between-stream aggregation.
        groups (list[str]): Grouping columns for independent-stream summaries; run_id is the
            replicate identifier.

    Returns:
        summary (pd.DataFrame): Per-group float mean/sample SD (ddof=1), integer
            available-stream n and measured run IDs. Empty inputs return named columns; n<2
            gives unavailable SD.

    Raises:
        ValueError: If repeated stream/group rows would be counted as independent replicates.
        KeyError: If nonempty inputs omit required group, run_id or value columns.
    """
    frame = pd.DataFrame(rows)
    columns = [*groups, "mean", "sample_sd", "n", "run_ids"]
    # Keep absent observation tables distinct from numerical zero.
    if frame.empty:
        return pd.DataFrame(columns=columns)
    # Repeated within-stream observations must be reduced before stream
    # aggregation.
    if frame.duplicated([*groups, "run_id"]).any():
        raise ValueError("Repeated within-stream observations must be reduced before stream aggregation.")
    # Boolean status flags are not numerical research observations.
    values = frame["value"].map(lambda value: np.nan if isinstance(value, (bool, np.bool_)) else value)
    frame["value"] = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    records = []
    for labels, group in frame.groupby(groups[0] if len(groups) == 1 else groups, dropna=False, sort=False):
        labels = labels if isinstance(labels, tuple) else (labels,)
        measured = group.loc[group["value"].notna()]
        records.append({**dict(zip(groups, labels)), "mean": measured["value"].mean(),
                        "sample_sd": measured["value"].std(ddof=1), "n": len(measured),
                        "run_ids": "|".join(measured["run_id"].astype(str))})
    return pd.DataFrame(records, columns=columns)


def _within_stream(rows: list[dict] | pd.DataFrame, keys: list[str], rule: str="mean") -> pd.DataFrame:
    """One value and available-observation count per within-stream group; empty input preserves the
    expected columns.

    Args:
        rows (list[dict] | pd.DataFrame): Saved observation records or equivalent DataFrame;
            repeated measurements must be reduced before between-stream aggregation.
        keys (list[str]): Within-stream grouping columns, including run identity and measurement
            identity.
        rule (str): Pandas aggregation name; mean is the default for repeated within-stream
            measurements.

    Returns:
        summary (pd.DataFrame): One value and available-observation count per within-stream
            group; empty input preserves the expected columns.

    Raises:
        KeyError: If nonempty rows omit grouping/value columns.
        ValueError: If the requested pandas aggregation is invalid.
    """
    frame = pd.DataFrame(rows)
    # Keep absent observation tables distinct from numerical zero.
    if frame.empty:
        return pd.DataFrame(columns=[*keys, "value", "n_observations"])
    return frame.groupby(keys, dropna=False, sort=False)["value"].agg(
        value=rule, n_observations="count").reset_index()


def saved_task_runtime(costs: pd.DataFrame, task_count: int) -> dict:
    """Sum a complete disjoint native task_total ledger; retain missing counts.

    Duplicate or foreign task indices are conflicting evidence. Missing or nonfinite timers are
    unavailable, never a partial complete-stream sum.

    Args:
        costs (pd.DataFrame): Saved native resource ledger with metric, task_index and value
            columns; an empty frame means no timers.
        task_count (int): Number of declared tasks; observed indices must lie within that
            schedule.

    Returns:
        timing (dict): Complete sum in float seconds, integer available task count and
            missingness reason. Missing/nonfinite tasks yield NaN instead of a partial total.

    Raises:
        ValueError: If task indices conflict or elapsed time is negative.
        KeyError: If a nonempty ledger omits required columns.
    """
    # Keep absent observation tables distinct from numerical zero.
    # A timer ledger must describe a positive integer task schedule.
    if isinstance(task_count, bool) or not isinstance(task_count, (int, np.integer)) or task_count < 1:
        raise ValueError("task_count must be a positive integer.")
    # An absent ledger cannot establish any complete task runtime.
    if costs.empty:
        return {"seconds": np.nan, "n_tasks": 0, "reason": "task_timers_unavailable"}
    timers = costs.loc[costs["metric"].eq("seconds/task_total")]
    indices = pd.to_numeric(timers["task_index"].map(
        lambda value: np.nan if isinstance(value, (bool, np.bool_)) else value), errors="coerce")
    # Task runtime ledger has duplicate, noninteger or out-of-schedule task
    # indices.
    if indices.isna().any() or not indices.isin(range(task_count)).all() or indices.duplicated().any():
        raise ValueError("Task runtime ledger has duplicate, noninteger or out-of-schedule task indices.")
    values = pd.to_numeric(timers["value"].map(
        lambda value: np.nan if isinstance(value, (bool, np.bool_)) else value), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    # Task runtime ledger contains a negative elapsed time.
    if np.any(values < 0):
        raise ValueError("Task runtime ledger contains a negative elapsed time.")
    measured = np.isfinite(values)
    complete = len(values) == task_count and measured.all()
    return {"seconds": float(values.sum()) if complete else np.nan, "n_tasks": int(measured.sum()),
            "reason": None if complete else "incomplete_or_nonfinite_task_timers"}


def _unique_task_records(rows: list[dict] | pd.DataFrame, task_count: int, description: str) -> None:
    """Preserve missing diagnostic tasks but reject accidental replicate duplication.

    Args:
        rows (list[dict] | pd.DataFrame): Saved observation records or equivalent DataFrame;
            repeated measurements must be reduced before between-stream aggregation.
        task_count (int): Number of declared tasks; observed indices must lie within that
            schedule.
        description (str): Human-readable origin used to identify invalid task records.

    Returns:
        validated (None): None; missing tasks are allowed while every observed task must have
            one valid 1-based identity.

    Raises:
        ValueError: If tasks are duplicated, noninteger or outside the declared schedule.
    """
    tasks = [row.get("task") for row in rows]
    # This value has duplicate or out-of-schedule task observations.
    if any(isinstance(task, bool) or not isinstance(task, int) or not 1 <= task <= task_count for task in tasks) \
            or len(tasks) != len(set(tasks)):
        raise ValueError(f"{description} has duplicate or out-of-schedule task observations.")


def aligned_phase_endpoints(before: dict, after: dict) -> bool:
    """Authenticate same saved validation rows; requested probe budgets are insufficient.

    Args:
        before (dict): Saved pre-consolidation endpoint, including split, input hash and actual
            example count.
        after (dict): Saved post-consolidation endpoint; it must identify the same validation
            rows as before.

    Returns:
        aligned (bool): True for identical positive-size hashed validation cohorts; False for
            absent or explicitly unavailable endpoints.

    Raises:
        ValueError: If present endpoints cannot prove identical validation examples.
    """
    # Apply this case only when not before or not after or before.get('split') == 'unavailable' or
    # (after.get('split') == 'unavailable').
    if not before or not after or before.get("split") == "unavailable" or after.get("split") == "unavailable":
        return False
    count = before.get("examples")
    aligned = isinstance(count, int) and not isinstance(count, bool) and count > 0 \
        and isinstance(after.get("examples"), int) and not isinstance(after.get("examples"), bool) \
        and count == after.get("examples") and before.get("split") == after.get("split") == "validation" \
        and bool(before.get("input_sha256")) and before.get("input_sha256") == after.get("input_sha256")
    # Phase endpoints do not prove identical fixed examples on the validation
    # split.
    if not aligned:
        raise ValueError("Phase endpoints do not prove identical fixed examples on the validation split.")
    return True


def _check_final_design(record: dict, manifests: dict[str, dict]) -> None:
    """None; verifies all 24 declared three-seed full-class streams and exact within-seed pairing.

    Args:
        record (dict): Frozen design or completion record, as required by this operation;
            scientific fields are retained.
        manifests (dict[str, dict]): Dataset-name mapping to native paired study manifests.

    Returns:
        validated (None): None; verifies all 24 declared three-seed full-class streams and exact
            within-seed pairing.

    Raises:
        ValueError: If seeds, methods, class schedules or stream counts differ.
    """
    # Final chapter export requires the new three-seed campaign [1103, 2207,
    # 3301].
    if record.get("seeds") != SEEDS or set(manifests) != {"cifar10", "cifar100"}:
        raise ValueError("Final chapter export requires the new three-seed campaign [1103, 2207, 3301]. Preserve older campaigns separately.")
    for dataset, tasks, width, methods in (("cifar10", 5, 2, {"baseline", "extra_joint", "learned"}),
                                           ("cifar100", 10, 10, set(METHODS))):
        entries = materialize_run_plan(manifests[dataset])
        # Final chapter export requires the selected artifact the selected artifact streams.
        if len(entries) != len(methods) * 3:
            raise ValueError(f"Final chapter export requires {len(methods) * 3} {dataset} streams.")
        for seed in SEEDS:
            paired = [entry for entry in entries if entry["stream"]["stream_seed"] == seed]
            # Incomplete declared paired conditions for the selected artifact, seed this
            # value.
            if {entry["condition"] for entry in paired} != methods or len(paired) != len(methods):
                raise ValueError(f"Incomplete declared paired conditions for {dataset}, seed {seed}.")
            schedules = [entry["stream"]["task_groups"] for entry in paired]
            # Paired methods must share the exact class schedule.
            if any(schedule != schedules[0] for schedule in schedules):
                raise ValueError("Paired methods must share the exact class schedule.")
            schedule = schedules[0]
            # Final chapter export requires the full the selected artifact class schedule.
            if len(schedule) != tasks or any(len(group) != width for group in schedule) \
                    or sorted(c for group in schedule for c in group) != list(range(tasks * width)):
                raise ValueError(f"Final chapter export requires the full {dataset} class schedule.")


def extract_saved_evidence(manifests: dict[str, dict], outputs: dict[str, dict]) -> dict:
    """Extract authenticated records into narrow metric tables, without model access.

    Completion validation is owned by workflow. This separately checks fixed phase endpoints and
    suppresses unsafe legacy CKA in exported observations.

    Args:
        manifests (dict[str, dict]): Dataset-name mapping to native paired study manifests.
        outputs (dict[str, dict]): Dataset-name mapping to completed records keyed by declared
            run ID.

    Returns:
        evidence (dict): Stream-first numeric tables, source-file hashes and run inventory. Test
            efficacy remains separate from validation diagnostics; absent measurements stay NaN.

    Raises:
        ValueError: If task observations, pairing, endpoint identities or resource ledgers
            conflict.
        OSError: If required saved files cannot be read.
    """
    main, trajectories, phases, temporal, resources, task_resources, replay = [], [], [], [], [], [], []
    sources, run_inventory = {}, []
    for dataset, manifest in manifests.items():
        plan = {entry["run_id"]: entry for entry in materialize_run_plan(manifest)}
        for run_id, record in sorted(outputs[dataset].items()):
            entry, directory = plan[run_id], Path(record["results_path"])
            identity = dict(dataset=dataset, condition=record["condition"], method=METHODS[record["condition"]],
                            run_id=run_id, block_id=entry["block_id"], seed=entry["stream"]["stream_seed"])
            files = [directory / name for name in ("route_metrics.json", "route_resources.csv", "task_metrics.csv",
                     "section11.json", "accuracy_matrices.csv", "schedule.csv", "summary.csv", "config.yaml", "input_config.yaml", "route.settings.yaml", "source_provenance.json")]
            # Bind the prespecified qualitative run before reading its saved replay.
            if identity["condition"] == "learned" and identity["seed"] == SEEDS[0]:
                files.append(directory / f"generated_examples_task_{len(entry['stream']['task_groups']):03d}.npz")
            sources[f"{dataset}/{run_id}"] = [{"path": str(path), "sha256": _hash(path)} for path in files if path.is_file()]
            run_inventory.append({**identity, "results_path": str(directory), "manifest_hash": manifest["manifest_hash"],
                                  "class_order": entry["stream"]["class_order"], "task_groups": entry["stream"]["task_groups"]})
            matrix = np.asarray(record["accuracy_matrix"], dtype=float)
            for metric, score in continual_metrics(matrix).items():
                main.append({**identity, "metric": metric, "unit": METRICS[metric][1], "value": score * 100})
            new, old = task_accuracy_summaries(matrix)
            for task in range(len(matrix)):
                for cohort, score in (("new", new[task]), ("old", old[task]), ("all_seen", matrix[task, :task + 1].mean())):
                    trajectories.append({**identity, "task": task + 1, "cohort": cohort, "unit": "%", "value": score * 100})
            route_path, observer_path = directory / "route_metrics.json", directory / "section11.json"
            route = sanitize_cka(_read(route_path)) if route_path.is_file() else []
            observer = sanitize_cka(_read(observer_path)) if observer_path.is_file() else {}
            _unique_task_records(route, len(matrix), f"{dataset}/{run_id} route diagnostics")
            _unique_task_records(observer.get("tasks", []), len(matrix), f"{dataset}/{run_id} observer diagnostics")
            route_by_task = {row["task"]: row for row in route}
            for task in range(1, len(matrix) + 1):
                row = route_by_task.get(task, {})
                before, after = row.get("before_consolidation") or {}, row.get("after_consolidation") or {}
                aligned = aligned_phase_endpoints(before, after)
                for metric, (_, unit, scale) in PHASE_METRICS.items():
                    source_metric = f"frozen_target_alignment.aggregates.selected_gates.{metric}" if metric in ("hidden_infonce", "hidden_target_cosine") else metric
                    a = _finite(_get(before, source_metric)) if aligned else np.nan
                    b = _finite(_get(after, source_metric)) if aligned else np.nan
                    for phase, value in (("before", a), ("after", b), ("after_minus_before", b - a)):
                        phases.append({**identity, "task": task, "metric": metric, "phase": phase,
                            "unit": "percentage points" if phase == "after_minus_before" and unit == "%" else unit,
                            "value": value * scale, "probe_examples": before.get("examples"),
                            "reason": None if aligned else "consolidation_boundary_unavailable"})
            for task in observer.get("tasks", []):
                for class_id, cohort in task.get("hidden", {}).get("per_class", {}).items():
                    for reference in ("since_acquisition", "since_previous_observation"):
                        drift = cohort.get(reference) or {}
                        for metric in ("linear_cka", "mean_sample_l2_drift", "relative_frobenius_drift", "centroid_drift.mean_centroid_drift"):
                            temporal.append({**identity, "task": task["task"], "class_id": class_id,
                                "reference": reference, "metric": metric, "unit": "ratio" if metric in ("linear_cka", "relative_frobenius_drift") else "feature units",
                                "value": _finite(_get(drift, metric)), "aligned_examples": drift.get("sample_count"),
                                "reason": drift.get("linear_cka_unavailable_reason") if metric == "linear_cka" else None})
                generation = task.get("generated_memory", {})
                for metric in ("label_consistency", "normalized_label_entropy", "class_coverage"):
                    replay.append({**identity, "task": task["task"], "metric": metric, "unit": "fraction",
                                   "value": _finite(generation.get("summary", {}).get(metric)),
                                   "reason": generation.get("reason")})
            cost_path = directory / "task_metrics.csv"
            costs = pd.read_csv(cost_path) if cost_path.is_file() else pd.DataFrame()
            # Keep absent observation tables distinct from numerical zero.
            if not costs.empty:
                costs = costs.loc[costs["phase"].eq("resource")].copy()
                costs["value"] = pd.to_numeric(costs["value"], errors="coerce")
                task_resources.extend({**identity, **row} for row in costs.to_dict("records"))
            timing = saved_task_runtime(costs, len(matrix))
            checkpoint_costs = costs.loc[costs["metric"].eq("checkpointing/io_seconds")].copy() if not costs.empty else costs.copy()
            checkpoint_costs["metric"] = "seconds/task_total"
            checkpoint_timing = saved_task_runtime(checkpoint_costs, len(matrix))
            ledger = {"measured_task_runtime": (timing["seconds"], "seconds", "sum of nonoverlapping active task_total; resumed committed segments included; measured progress writes and downtime excluded"),
                      "measured_checkpoint_io": (checkpoint_timing["seconds"], "seconds", "sum of recorded progress-checkpoint writes; interrupted unfinished write timers and uncommitted lost work are unavailable"),
                      "elapsed_notebook_time": (_finite(record.get("seconds")), "seconds", "current attempt only; includes setup and manual pauses; excludes earlier interrupted attempts"),
                      "total_optimizer_updates": (_finite(record.get("total_updates")), "updates", "joint + acquisition + consolidation + extra joint"),
                      "sampled_process_peak_rss": (_finite(_get(observer, "memory.sampled_process_peak_rss_bytes")), "bytes", "sampled process RSS, not exact peak or GPU occupancy")}
            for phase, key in (("joint", "joint_updates"), ("extra_joint", "extra_joint_updates"),
                               ("acquisition", "acquisition.updates"), ("consolidation", "consolidation.updates")):
                values = [_finite(_get(row, key)) for row in route]
                ledger[f"{phase}_updates"] = (sum(values) if len(values) == len(matrix) else np.nan, "updates", "sum of saved per-task phase optimizer updates")
            for device, measurements in (observer.get("memory", {}).get("tf_allocator_devices", {}) or {}).items():
                ledger[f"allocator_peak/{device}"] = (_finite((measurements or {}).get("peak")), "bytes", "TensorFlow allocator high water; not total device memory")
            inventory = [_finite(_get(row, "tensor_inventory.unique_tensor_bytes")) for row in observer.get("tasks", [])]
            measured_inventory = [value for value in inventory if np.isfinite(value)]
            ledger["tensor_inventory_max"] = (max(measured_inventory) if measured_inventory else np.nan, "bytes", "maximum end-of-fit deduplicated tensor payload; excludes Python and allocator overhead")
            memory_names = sorted({key for row in route for key in row.get("memory_bytes", {})})
            for name in memory_names:
                values = [_finite(row.get("memory_bytes", {}).get(name)) for row in route]
                measured = [value for value in values if np.isfinite(value)]
                ledger[f"tensor_storage_max/{name}"] = (max(measured) if measured else np.nan, "bytes", "maximum recorded component tensor/array storage; do not sum overlapping components")
            for metric, (value, unit, scope) in ledger.items():
                measurement = timing if metric == "measured_task_runtime" else checkpoint_timing if metric == "measured_checkpoint_io" else {}
                resources.append({**identity, "metric": metric, "unit": unit, "value": value, "measurement_scope": scope,
                    "observed_tasks": measurement.get("n_tasks"), "unavailable_reason": measurement.get("reason")})
    frames = {name: pd.DataFrame(rows) for name, rows in (("individual_runs", main), ("trajectories_individual", trajectories),
              ("phase_observations", phases), ("temporal_observations", temporal), ("resources_individual", resources),
              ("task_resources", task_resources), ("replay_observations", replay))}
    basic = ["dataset", "condition", "method", "metric", "unit"]
    frames["main_results"] = summarize_streams(main, basic)
    frames["trajectories"] = summarize_streams(trajectories, ["dataset", "condition", "method", "task", "cohort", "unit"])
    phase_keys = [*KEYS, "metric", "phase", "unit"]
    frames["phase_individual"] = _within_stream(phases, phase_keys)
    frames["phase_changes"] = summarize_streams(frames["phase_individual"], [*basic, "phase"])
    # Equal class weight within task, then equal observed task weight within a
    # stream: later tasks/classes are never counted as training replicates.
    temporal_keys = [*KEYS, "reference", "metric", "unit"]
    temporal_task = _within_stream(temporal, [*temporal_keys, "task"])
    frames["temporal_individual"] = _within_stream(temporal_task, temporal_keys)
    frames["temporal_drift"] = summarize_streams(frames["temporal_individual"], [*basic, "reference"])
    frames["resources"] = summarize_streams(resources, [*basic, "measurement_scope"])
    frames["replay_individual"] = _within_stream(replay, [*KEYS, "metric", "unit"])
    frames["replay"] = summarize_streams(frames["replay_individual"], basic)
    effects = []
    scores = frames["individual_runs"]
    # Keep absent observation tables distinct from numerical zero.
    if not scores.empty:
        scores = scores.loc[scores["metric"].eq("final_average_accuracy")]
        for (dataset, block), paired in scores.groupby(["dataset", "block_id"]):
            learned = paired.loc[paired["condition"].eq("learned")]
            # Keep absent observation tables distinct from numerical zero.
            if learned.empty:
                continue
            learned = learned.iloc[0]
            for _, comparator in paired.loc[paired["condition"].ne("learned")].iterrows():
                effects.append({"dataset": dataset, "block_id": block, "seed": learned["seed"],
                    "condition": comparator["condition"], "method": comparator["method"],
                    "comparison": f"Learned minus {comparator['method']}", "role": "primary" if comparator["condition"] == "extra_joint" else "secondary descriptive",
                    "run_id": learned["run_id"], "comparator_run_id": comparator["run_id"],
                    "metric": "final_average_accuracy_difference", "unit": "percentage points", "value": learned["value"] - comparator["value"]})
    frames["paired_individual"] = pd.DataFrame(effects)
    frames["paired_effects"] = summarize_streams(effects, ["dataset", "condition", "method", "comparison", "role", "metric", "unit"])
    frames["cifar100_mechanism_comparison"] = frames["main_results"].loc[
        frames["main_results"]["dataset"].eq("cifar100") & frames["main_results"]["condition"].isin(["learned", "random", "ce_only"])].copy()
    frames["thesis_summary"] = compact_summary(frames)
    return {"tables": frames, "sources": sources, "runs": run_inventory}


def compact_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Format one small row per treatment using existing stream-first summaries.

    Args:
        tables (dict[str, pd.DataFrame]): Saved-evidence tables. main_results and resources
            contain mean, sample_sd and independent-stream n columns.

    Returns:
        summary (pd.DataFrame): Dataset/method rows with final accuracy (%), incremental
            accuracy (%), signed forgetting and backward transfer (percentage points),
            optimizer updates and measured task seconds. Each cell states mean, sample SD
            and actual n; missing values remain unavailable. Numeric source tables retain
            full precision.

    Raises:
        KeyError: If an expected summary column or table is missing.
        ValueError: If a treatment has multiple summaries for the same metric.
    """
    selected = {"final_average_accuracy": "Final accuracy (%)",
                "average_incremental_accuracy": "Incremental accuracy (%)",
                "average_forgetting": "Signed forgetting (pp)",
                "backward_transfer": "Backward transfer (pp)",
                "total_optimizer_updates": "Optimizer updates",
                "measured_task_runtime": "Active task seconds"}
    rows = pd.concat([tables["main_results"], tables["resources"]], ignore_index=True)
    rows = rows.loc[rows["metric"].isin(selected)].copy()
    rows["measurement"] = rows["metric"].map(selected)
    values = []
    for row in rows.itertuples():
        # A missing timer or outcome is not a numerical zero.
        if row.n == 0 or not np.isfinite(row.mean):
            values.append("unavailable (n=0)")
        # Handle the complementary supported case without inventing observations.
        else:
            sd = f"{row.sample_sd:.2f}" if np.isfinite(row.sample_sd) else "unavailable"
            values.append(f"{row.mean:.2f} ± {sd} (n={int(row.n)})")
    rows["mean ± sample SD (n)"] = values
    return rows.pivot(index=["dataset", "method"], columns="measurement",
                      values="mean ± sample SD (n)").reindex(columns=list(selected.values())).reset_index()


def _markdown(frame: pd.DataFrame) -> str:
    """Readable Markdown table with escaped delimiters, five-significant-digit floats and explicit
    unavailable cells; empty input produces a missing-data message.

    Args:
        frame (pd.DataFrame): Saved tabular measurements; missing values stay unavailable.

    Returns:
        markdown (str): Readable Markdown table with escaped delimiters, five-significant-digit
            floats and explicit unavailable cells; empty input produces a missing-data message.

    Raises:
        None: This formatter adds no validation beyond accessing the DataFrame.
    """
    # Keep absent observation tables distinct from numerical zero.
    if frame.empty:
        return "Unavailable: no saved observations.\n"
    def cell(value: object) -> str:
        """Escaped cell text; None, pandas missing values and NaN display as unavailable.

        Args:
            value (object): JSON-like object or scalar. Missing and nonfinite values are handled
                as described in the return contract.

        Returns:
            formatted (str): Escaped cell text; None, pandas missing values and NaN display as
                unavailable.

        Raises:
            None: Values otherwise use ordinary string conversion.
        """
        # Keep finite measurements separate from unavailable values.
        if value is None or value is pd.NA or value is pd.NaT or isinstance(value, float) and np.isnan(value):
            return "unavailable"
        return (f"{value:.5g}" if isinstance(value, float) else str(value)).replace("|", "; ").replace("\n", " ")
    return "| " + " | ".join(frame.columns) + " |\n| " + " | ".join("---" for _ in frame.columns) + " |\n" + "\n".join(
        "| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)) + "\n"


def _context(record: dict, manifests: dict[str, dict], evidence: dict, status: str) -> str:
    """Portable explanatory Markdown distinguishing fixed recipe, outcome definitions, repeated
    observations and interpretation limits.

    Args:
        record (dict): Frozen design or completion record, as required by this operation;
            scientific fields are retained.
        manifests (dict[str, dict]): Dataset-name mapping to native paired study manifests.
        evidence (dict): Saved-only evidence dictionary containing per-stream tables, run
            identities and source hashes.
        status (str): Explicit output label; synthetic fixtures must state that they are not
            research results.

    Returns:
        context (str): Portable explanatory Markdown distinguishing fixed recipe, outcome
            definitions, repeated observations and interpretation limits.

    Raises:
        KeyError: If required frozen design or evidence fields are missing.
    """
    text = [f"# Study context — {status}",
        "Research question: does learned temporary class modulation followed by semantic consolidation improve clean class-incremental retention and new-class learning beyond extra ordinary joint updates?",
        "Methods: Platform (native baseline) uses joint diffusion/classification, generated replay and classification/denoising distillation. Extra joint receives the acquisition-plus-consolidation update allowance. Learned adds trained gates and CE + InfoNCE semantic consolidation. Random uses random gates. CE only (native no_consolidation) retains acquisition and runs the replacement CE phase without the alignment gradient.",
        f"Declared confirmation seeds: {record['seeds']}. Development seed 17 is separate. Revised final design: CIFAR-10, five two-class tasks, three methods, nine streams; CIFAR-100, ten ten-class tasks, five methods, fifteen streams. This export contains {len(evidence['runs'])} completed streams; a progress export is not the final design. Actual class schedules and source/config identities are in provenance/study_design.json and tables/T00_run_inventory.csv.",
        "Inference: clean, null-conditioned raw-network primary classifier over every seen class, without task identity, gates or predictor. Main efficacy uses the ordinary held-out test matrix; phase and temporal diagnostics use validation data and never replace the efficacy endpoint.",
        "Metrics: for A[i,j], accuracy on task j after training i, final accuracy averages the final row's learned tasks; incremental accuracy averages each learned-prefix row mean. Signed forgetting averages max(A[j:T-1,j]) - A[T-1,j] over old tasks, excluding the final row from the maximum. BWT averages A[T-1,j] - A[j,j] over old tasks. These native formulas are computed separately for each full stream before mean and sample SD (ddof=1). Accuracy is displayed as percent; forgetting, BWT and accuracy differences as percentage points. Negative forgetting and positive BWT indicate improvement. First-task old accuracy is unavailable.",
        "Repeated measurements: trajectory means/SD use one observation per stream at each task/cohort. Phase summaries first average the matched within-task before, after or after-minus-before observations within each stream. Temporal drift averages observed classes within task, then tasks within stream. n_observations and n preserve available cohort/task and independent-stream counts. Missing phases and n<2 SD remain unavailable.",
        "Resources: active stream time sums only complete seconds/task_total entries, including resumed committed segments. Measured checkpoint writes are reported separately; downtime, uncommitted lost work and interrupted unfinished write timers are not complete observations. Notebook elapsed time covers only the current attempt, including setup and pauses. Route/fit/sampling timers overlap task totals and are not added to them. Tensor storage, process RSS sampling and TF allocator high-water values are separate measurements; per-component maxima are not simultaneous total memory.", LIMITATIONS]
    for dataset, manifest in manifests.items():
        config = manifest["spec"]["base_config"]
        common, route = config["common"], config["route"]
        exposure = common["continually_learn"]
        text.extend([f"\n## {dataset.upper()} frozen settings",
            f"Current examples per task: {exposure.get('replay_current_examples')} (null means all permitted current training data, after validation split/caps); fixed old replay budget: {exposure.get('replay_old_examples')}. Replay is a fixed generated pool within each task. Extra epochs reuse that pool. Old raw arrays remain in simulator host memory; historical validation supports diagnostics. Actual exposure and runtime are in task_resources.",
            "The complete resolved frozen settings follow; identifiers and categorical options are not averaged.",
            "```json\n" + json.dumps({"dataset": common.get("dataset"), "model": common.get("model"),
                "optimizer": common.get("optimizer"), "training": common.get("training"),
                "continually_learn": exposure, "route": route}, indent=2) + "\n```"])
    return "\n\n".join(text) + "\n"


TABLE_CAPTIONS = {
    "cifar100_mechanism_comparison": "CIFAR-100 learned/random/CE-only mechanism comparison on the ordinary held-out test matrix. Mean, sample SD and n independent full streams. Random and CE-only differences support interpretation of this adaptation; they do not prove a unique cognitive mechanism.",
    "individual_runs": "Native continual metrics recomputed independently for every saved full-stream ordinary test matrix, displayed in % or percentage points; no averaged-matrix forgetting.",
    "main_results": "Main continual outcomes: mean, sample SD (ddof=1) and actual n full streams. Signed forgetting and BWT retain their sign. These compare efficacy in the frozen reduced protocol, not convergence or general superiority.",
    "paired_individual": "Each paired learned-minus-comparator final test-accuracy effect in percentage points. Extra joint is primary; other comparisons are descriptive supporting evidence.",
    "paired_effects": "Mean and sample SD across available independent paired effects. SD is not a confidence interval; native primary intervals are supplied separately.",
    "trajectories_individual": "Per-stream ordinary test new-task, old-task and all-seen task-balanced accuracy (%). No old classes exist at task one; its old value remains unavailable.",
    "trajectories": "Combined clean test accuracy trajectories by dataset, method, task and cohort. Each mean/SD uses at most three independent streams at that task, with explicit n. Repeated tasks are not independent replicates.",
    "phase_observations": "Fixed identical validation examples before and after consolidation: deployed accuracy and predictor-free hidden observations. Absent platform/extra-joint phases stay unavailable. Accuracy changes use percentage points.",
    "phase_individual": "Before, after and paired within-task change, averaged over available tasks within each stream before uncertainty is summarized. n_observations counts tasks, not independent runs.",
    "phase_changes": "Stream-first validation consolidation summaries with mean, sample SD and n streams. Paired boundary changes establish local effects on the measured classifier/cohort, not independently causal evidence or whole-backbone preservation.",
    "temporal_observations": "Fixed per-class validation temporal drift. CKA requires actual aligned n>2, valid finite nonconstant features; legacy unknown/tiny-cohort CKA is unavailable with reasons. L2/Frobenius observations remain when valid.",
    "temporal_individual": "Temporal drift: equal observed class weighting within task, then equal observed task weighting within each independent stream. Small-cohort CKA is descriptive and is not CDNV.",
    "temporal_drift": "Stream-first predictor-free temporal representation drift mean, sample SD and actual n; no tasks/classes are treated as replicates. Missing acquisition references stay unavailable.",
    "resources_individual": "Per-stream active task time, separately measured progress-checkpoint writes, current-attempt notebook elapsed time, optimizer work and separately named memory measurements. Nested timers and component memory maxima must not be added together.",
    "resources": "Runtime/work/memory mean, sample SD and n full streams, grouped only for like quantities and units. Missing measurements do not become zero. These do not establish equal time/FLOPs.",
    "task_resources": "Native per-task resource ledger; individual observation detail only. Timers overlap and identifiers/class labels are not numerical outcomes to average.",
    "replay_observations": "Saved generated replay self-consistency observations against the current learner; first-task replay is unavailable. Not independent perceptual or semantic validation.",
    "replay_individual": "Replay diagnostics averaged over available tasks within each stream. n_observations is task count, not independent training replicates.",
    "replay": "Saved replay self-consistency mean, sample SD and n streams after within-stream reduction. No independently validated image-quality claim follows.",
}


def _plots(evidence: dict, directory: Path, register: Callable[[str, Path, str, str], None], status: str, native: dict) -> None:
    """None; writes optional saved-data figures, registering provenance without new predictions or
    generation.

    Args:
        evidence (dict): Saved-only evidence dictionary containing per-stream tables, run
            identities and source hashes.
        directory (Path): Destination used by the caller; publication requires a fresh staging
            directory.
        register (Callable[[str, Path, str, str], None]): Callback recording each saved
            artifact, its caption and its source-table identity.
        status (str): Explicit output label; synthetic fixtures must state that they are not
            research results.
        native (dict): Native paired-statistics mapping by dataset; empty for a progress or
            synthetic view.

    Returns:
        saved (None): None; writes optional saved-data figures, registering provenance without
            new predictions or generation. Missing measurements remain labeled unavailable.

    Raises:
        KeyError: If evidence tables omit required plotting columns.
        OSError: If figures cannot be saved.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tables = evidence["tables"]
    def save(fig: Figure, name: str, caption: str, data: str) -> None:
        """None; applies the supplied status label, saves PNG/SVG, registers both and closes the
        Figure.

        Args:
            fig (Figure): Matplotlib Figure to label, save and close.
            name (str): Output artifact basename without the file extension.
            caption (str): Description of measurement, aggregation and interpretation limits.
            data (str): Name of the source evidence table or qualitative selection used for
                provenance.

        Returns:
            saved (None): None; applies the supplied status label, saves PNG/SVG, registers both
                and closes the Figure.

        Raises:
            OSError: If image publication fails.
        """
        prefix = "SYNTHETIC VALIDATION — " if "SYNTHETIC" in status else "PROGRESS — " if status != "FINAL CHAPTER EVIDENCE" else ""
        fig.suptitle(prefix + caption.split(".")[0], fontsize=12)
        for suffix in ("png", "svg"):
            path = directory / f"{name}.{suffix}"
            fig.savefig(path, dpi=320, facecolor="white")
            register(name, path, caption, data)
        plt.close(fig)
    def errorpoints(ax: Axes, frame: pd.DataFrame, metric: str, title: str) -> None:
        """None; draws per-treatment saved means and sample SD with actual n, explicitly showing
        unavailable categories.

        Args:
            ax (Axes): Matplotlib Axes receiving the plotted saved summaries.
            frame (pd.DataFrame): Saved tabular measurements; missing values stay unavailable.
            metric (str): Exact measurement name to select from the saved summary table.
            title (str): Readable panel title; no inferred result is added.

        Returns:
            drawn (None): None; draws per-treatment saved means and sample SD with actual n,
                explicitly showing unavailable categories.

        Raises:
            KeyError: If required summary columns are absent.
        """
        selected = frame.loc[frame["metric"].eq(metric)] if not frame.empty else frame
        conditions = [condition for condition in METHODS if not selected.empty and selected["condition"].eq(condition).any()]
        for index, condition in enumerate(conditions):
            row = selected.loc[selected["condition"].eq(condition)] if not selected.empty else selected
            # Keep absent observation tables distinct from numerical zero.
            if row.empty:
                continue
            row = row.iloc[0]
            # Keep finite measurements separate from unavailable values.
            if np.isfinite(row["mean"]):
                ax.errorbar(index, row["mean"], yerr=row["sample_sd"] if np.isfinite(row["sample_sd"]) else None,
                            marker=MARKERS[condition], color=COLORS[condition], capsize=4, linestyle="none")
                ax.annotate(f"n={row['n']}", (index, row["mean"]), xytext=(3, 6), textcoords="offset points", fontsize=7)
            # Handle the complementary supported case without inventing observations.
            else:
                ax.text(index, .08, "unavailable\n(n=0)", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color=".4")
        # Text for unavailable methods does not affect Matplotlib's data limits.
        # Fix the complete category extent after plotting so n=0 slots remain
        # inside the axes instead of being clipped by observed-point autoscaling.
        ax.set(xticks=range(len(conditions)), xticklabels=[METHODS[condition] for condition in conditions],
               xlim=(-.5, max(.5, len(conditions) - .5)), title=title)
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(axis="y", alpha=.2)
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
        for row, dataset in enumerate(("cifar10", "cifar100")):
            frame = tables["main_results"]
            frame = frame.loc[frame["dataset"].eq(dataset)]
            for column, (metric, (label, unit)) in enumerate(METRICS.items()):
                errorpoints(axes[row, column], frame, metric, f"{dataset.upper()}: {label}")
                axes[row, column].set_ylabel(unit)
                # Use percentage-point labels for accuracy differences.
                if unit == "%":
                    axes[row, column].set_ylim(0, 100)
                # Handle the complementary supported case without inventing observations.
                else:
                    axes[row, column].axhline(0, color="black", linewidth=.6)
        save(fig, "F01_main_results", "Main continual outcomes. Points are stream means; whiskers are sample SD, not confidence intervals. Source: main_results table; units and n appear on each panel.", "main_results")
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        for row, dataset in enumerate(("cifar10", "cifar100")):
            for column, cohort in enumerate(("old", "new", "all_seen")):
                ax = axes[row, column]
                for condition in METHODS:
                    frame = tables["trajectories"]
                    selected = frame.loc[frame["dataset"].eq(dataset) & frame["condition"].eq(condition) & frame["cohort"].eq(cohort)].sort_values("task")
                    # Keep absent observation tables distinct from numerical zero.
                    if selected.empty:
                        continue
                    x, y, sd = (selected[key].to_numpy(dtype=float) for key in ("task", "mean", "sample_sd"))
                    ax.plot(x, y, label=METHODS[condition], color=COLORS[condition], marker=MARKERS[condition], markersize=4)
                    ax.fill_between(x, y - sd, y + sd, color=COLORS[condition], alpha=.12)
                ax.set(title=f"{dataset.upper()} — {cohort.replace('_', ' ')}", xlabel="Completed task (count)", ylabel="Clean test accuracy (%)", ylim=(0, 100))
                task_count = 5 if dataset == "cifar10" else 10
                ax.set(xticks=range(1, task_count + 1), xlim=(.8, task_count + .2))
                ax.grid(alpha=.2)
        axes[0, 2].legend(fontsize=8)
        axes[1, 2].legend(fontsize=8)
        save(fig, "F02_accuracy_trajectories", "Old, new and all-seen accuracy trajectories. Lines show full-stream means and bands show sample SD (not CI); n per point is in CSV. Missing first-task old accuracy stays a gap. Test split, task-balanced means.", "trajectories")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, dataset in zip(axes, ("cifar10", "cifar100")):
            frame = tables["paired_individual"]
            frame = frame.loc[frame["dataset"].eq(dataset)] if not frame.empty else frame
            comparators = [condition for condition in METHODS if condition != "learned" and not frame.empty and frame["condition"].eq(condition).any()]
            for index, condition in enumerate(comparators):
                points = frame.loc[frame["condition"].eq(condition)] if not frame.empty else frame
                # Keep absent observation tables distinct from numerical zero.
                if points.empty:
                    continue
                values = points["value"].to_numpy(dtype=float)
                ax.scatter(index + np.linspace(-.1, .1, len(values)), values, color=COLORS[condition], marker=MARKERS[condition])
                ax.plot(index, values.mean(), marker="_", color="black", markersize=18)
                # Overlay the native interval only for the declared primary comparison.
                if condition == "extra_joint" and dataset in native:
                    n = native[dataset]
                    ax.vlines(index, n["ci_95_lower"] * 100, n["ci_95_upper"] * 100, color="black", linewidth=2, label="Native primary 95% paired t CI")
            ax.axhline(0, color="black", linewidth=.7)
            ax.set(xticks=range(len(comparators)), xticklabels=[METHODS[c] for c in comparators], title=dataset.upper(), ylabel="Learned minus comparator (percentage points)", xlabel="Comparator")
            ax.tick_params(axis="x", labelrotation=20)
            # Display the native primary paired interval when final analysis is available.
            if dataset in native:
                ax.legend(fontsize=8)
        save(fig, "F03_paired_effects", "Individual paired effects. Each point is one independent paired seed; black ticks are means. Only Extra joint has the native primary 95% paired t confidence interval; other comparisons are secondary descriptive evidence. Test final accuracy, percentage points.", "paired_individual")
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        frame = tables["phase_changes"]
        frame = frame.loc[frame["dataset"].eq("cifar100") & frame["phase"].eq("after_minus_before")]
        for ax, metric in zip(axes, ("old_accuracy", "hidden_target_cosine")):
            errorpoints(ax, frame, metric, PHASE_METRICS[metric][0])
            ax.axhline(0, color="black", linewidth=.7)
            ax.set_ylabel("After minus before (pp)" if metric == "old_accuracy" else "After minus before (cosine)")
        save(fig, "F04_consolidation_changes", "CIFAR-100 consolidation boundary effects. Fixed validation examples; task changes are averaged within each stream before means and sample SD across streams. Left: deployed old accuracy; right: predictor-free hidden-target cosine. Missing baseline phases stay unavailable; this does not establish whole-backbone preservation.", "phase_changes")
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        for ax, dataset in zip(axes, ("cifar10", "cifar100")):
            frame = tables["resources"]
            frame = frame.loc[frame["dataset"].eq(dataset)]
            errorpoints(ax, frame, "measured_task_runtime", dataset.upper())
            ax.set(ylabel="Measured active task time (seconds)", ylim=(0, None))
        save(fig, "F05_runtime", "Recorded active task time. Sum of complete nonoverlapping task_total measurements within each stream; points and whiskers show means and sample SD across streams. Resumed committed segments are included; measured checkpoint writes and downtime are separate. Lost uncommitted work is not measured. Overlapping route timers are not added. Memory and optimizer work remain separate tables.", "resources")


def _qualitative(evidence: dict, manifests: dict[str, dict], directory: Path, register: Callable[[str, Path, str, str], None], status: str) -> None:
    """Fixed first declared seed/learned/final task, then first six saved label/row indices.

    Args:
        evidence (dict): Saved-only evidence dictionary containing per-stream tables, run
            identities and source hashes.
        manifests (dict[str, dict]): Dataset-name mapping to native paired study manifests.
        directory (Path): Destination used by the caller; publication requires a fresh staging
            directory.
        register (Callable[[str, Path, str, str], None]): Callback recording each saved
            artifact, its caption and its source-table identity.
        status (str): Explicit output label; synthetic fixtures must state that they are not
            research results.

    Returns:
        saved (None): None; saves the prespecified first-seed learned final-task replay view and
            source-row selection. Missing candidates stay unavailable; no alternate seed is
            substituted.

    Raises:
        ValueError: If saved arrays have incompatible counts or nonfinite images.
        OSError: If an archive or figure cannot be read or written.
    """
    import matplotlib.pyplot as plt
    chosen, selection = [], []
    for dataset in ("cifar10", "cifar100"):
        candidates = [run for run in evidence["runs"] if run["dataset"] == dataset and run["condition"] == "learned" and run["seed"] == SEEDS[0]]
        # Do not replace a missing prespecified qualitative run with another seed.
        if not candidates:
            continue
        run = candidates[0]
        path = Path(run["results_path"]) / f"generated_examples_task_{len(run['task_groups']):03d}.npz"
        # Use existing evidence only when the corresponding artifact is present.
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as archive:
            images, labels = archive["images"], archive["labels"]
            # Invalid saved qualitative array: the selected artifact.
            if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) \
                    or len(images) != len(labels) or not np.isfinite(images).all() \
                    or np.any(labels < 0) or np.any(labels >= len(run["class_order"])):
                raise ValueError(f"Invalid saved qualitative array: {path}")
            # Use one row per label first to avoid the first six images all coming from one class.
            indices = [next(index for index, label in enumerate(labels) if label == target) for target in sorted(set(labels.tolist()))[:6]]
            # An empty saved replay selection has no images to display.
            if not indices:
                continue
            # The native ExperimentalController.capture artifact contract is
            # always 2 * generated_[0,1]_pixels - 1, independent of dataset
            # preprocessing. Never normalize each displayed image separately.
            display_images = (images[indices] + 1) / 2
            chosen.append((run, np.clip(display_images, 0, 1), labels[indices]))
            source_hash = next((entry["sha256"] for entry in evidence["sources"][f"{dataset}/{run['run_id']}"]
                                if entry["path"] == str(path)), None)
            # Hash the selected archive if it was not already in the source catalog.
            if source_hash is None:
                source_hash = _hash(path)
            for index in indices:
                selection.append({**{key: run[key] for key in KEYS}, "source": str(path), "sha256": source_hash,
                    "row_index": index, "conditioning_dense_class": int(labels[index]),
                    "conditioning_original_class": run["class_order"][int(labels[index])],
                    "transform": "inverse native capture (stored_pixels+1)/2; fixed [0,1] clipping; nearest pixels; no per-image contrast"})
    frame = pd.DataFrame(selection)
    evidence["qualitative_selection"] = selection
    path = directory / "F06_qualitative_selection.csv"
    frame.to_csv(path, index=False)
    caption = "Saved replay examples, qualitative only. Select seed 1103's learned run in each dataset and its final task archive, then the first saved row for each of the first six sorted dense conditioning IDs; displayed labels map back to the saved original class order. No quality ranking, inference or new generation. Invert the native capture transform using (stored_pixels+1)/2, with fixed [0,1] display clipping; this is independent of dataset preprocessing. Labels are conditioning classes, not verified semantic labels. Missing prespecified files are unavailable, never replaced by another seed."
    register("F06_qualitative", path, caption, "qualitative_selection")
    # Write explicit missingness when no prespecified replay archive is available.
    if not chosen:
        (directory / "F06_qualitative_unavailable.md").write_text(caption + "\n\nNo eligible prespecified saved examples were available.\n", encoding="utf-8")
        register("F06_qualitative", directory / "F06_qualitative_unavailable.md", caption, "qualitative_selection")
        return
    columns = max(len(images) for _, images, _ in chosen)
    fig, axes = plt.subplots(len(chosen), columns, figsize=(10, 2.4 * len(chosen)), squeeze=False, constrained_layout=True)
    for row, (run, images, labels) in enumerate(chosen):
        for column, ax in enumerate(axes[row]):
            ax.axis("off")
            # Leave surplus grid cells blank.
            if column < len(images):
                ax.imshow(images[column].squeeze(), interpolation="nearest", cmap="gray")
                original_class = run["class_order"][int(labels[column])]
                ax.set_title(f"{run['dataset']} | seed {run['seed']}\nconditioning class {original_class}", fontsize=8)
    prefix = "SYNTHETIC VALIDATION — " if "SYNTHETIC" in status else "PROGRESS — " if status != "FINAL CHAPTER EVIDENCE" else ""
    fig.suptitle(prefix + "Saved generated replay — qualitative context", fontsize=11)
    for extension in ("png", "svg"):
        path = directory / f"F06_qualitative.{extension}"
        fig.savefig(path, dpi=320, facecolor="white")
        register("F06_qualitative", path, caption, "qualitative_selection")
    plt.close(fig)


def _write_package(directory: Path, record: dict, manifests: dict[str, dict], evidence: dict, native: dict, *, status: str, details: bool=True) -> Path:
    """Write into a fresh staging directory; synthetic fixture callers must label status.

    Args:
        directory (Path): Destination used by the caller; publication requires a fresh staging
            directory.
        record (dict): Frozen design or completion record, as required by this operation;
            scientific fields are retained.
        manifests (dict[str, dict]): Dataset-name mapping to native paired study manifests.
        evidence (dict): Saved-only evidence dictionary containing per-stream tables, run
            identities and source hashes.
        native (dict): Native paired-statistics mapping by dataset; empty for a progress or
            synthetic view.
        status (str): Explicit output label; synthetic fixtures must state that they are not
            research results.
        details (bool): True includes saved diagnostic figures and extended views; False keeps
            the compact scalar presentation.

    Returns:
        package (Path): Fresh staged package directory. details=False limits visible tables and
            omits plots; numeric JSON and exact source provenance are retained.

    Raises:
        FileExistsError: If the staging directory exists.
        ValueError: If evidence cannot be serialized or plotted consistently.
        OSError: If package publication fails.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tables_dir, figures_dir, provenance = (directory / name for name in ("tables", "figures", "provenance"))
    for path in (tables_dir, figures_dir, provenance):
        path.mkdir()
    artifacts = []
    def register(identifier: str, path: str | Path, caption: str, data: str) -> None:
        """None; appends the artifact byte hash, measurement split, source-run identities and
        interpretation to the enclosing manifest.

        Args:
            identifier (str): Stable table or figure identifier used in the artifact manifest.
            path (str | Path): File to read or write; relative paths use the current working
                directory.
            caption (str): Description of measurement, aggregation and interpretation limits.
            data (str): Name of the source evidence table or qualitative selection used for
                provenance.

        Returns:
            registered (None): None; appends the artifact byte hash, measurement split,
                source-run identities and interpretation to the enclosing manifest.

        Raises:
            OSError: If the newly written artifact cannot be read for hashing.
        """
        datasets = ["cifar100"] if identifier == "F04_consolidation_changes" or data == "cifar100_mechanism_comparison" else sorted(manifests)
        source_runs = {key: value for key, value in evidence["sources"].items() if key.split("/")[0] in datasets}
        selected_runs = [run for run in evidence["runs"] if run["dataset"] in datasets]
        # Restrict qualitative provenance to the actually selected runs.
        if data == "qualitative_selection":
            selected_ids = {f"{row['dataset']}/{row['run_id']}" for row in evidence.get("qualitative_selection", [])}
            source_runs = {key: value for key, value in source_runs.items() if key in selected_ids}
            selected_runs = [run for run in selected_runs if f"{run['dataset']}/{run['run_id']}" in selected_ids]
        counts = {f"{dataset}/{condition}": sum(run["dataset"] == dataset and run["condition"] == condition for run in selected_runs)
                  for dataset in datasets for condition in METHODS}
        split = "test outcomes and saved training resources" if data == "thesis_summary" else "test" if data in ("individual_runs", "main_results", "paired_individual", "paired_effects", "trajectories_individual", "trajectories", "cifar100_mechanism_comparison", "native_primary_statistics.json") else "validation" if data.startswith(("phase_", "temporal_")) else "saved training resource/replay observations"
        artifacts.append({"id": identifier, "file": path.relative_to(directory).as_posix(), "sha256": _hash(path),
            "caption": caption, "dataset": datasets, "split": split,
            "methods": METHODS, "aggregation": "independent complete stream first; mean and sample SD ddof=1 unless explicitly native paired CI",
            "sample_count": "actual n and within-stream counts in linked CSV; maximum three streams per condition",
            "independent_stream_counts_available": counts,
            "source_run_ids": list(source_runs),
            "source_files": {"catalog": "source_runs", "run_ids": list(source_runs)}, "data_table": data,
            "generating_procedure": "results_package.export_results_package -> extract_saved_evidence -> _write_package",
            "interpretation_limit": caption + " " + LIMITATIONS})
    inventory = pd.DataFrame(evidence["runs"])
    inventory_path = tables_dir / "T00_run_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    register("T00", inventory_path, "Run identities and exact paired class schedules; these identifiers are never averaged.", "run_inventory")
    table_index = {}
    for index, (name, frame) in enumerate(evidence["tables"].items(), 1):
        # The default chapter package keeps scalar outcomes and their source rows.
        if not details and name not in ("thesis_summary", "individual_runs", "paired_individual", "resources_individual"):
            continue
        identifier = f"T{index:02d}"
        table_index[name] = identifier
        caption = TABLE_CAPTIONS.get(name, "Mean ± sample SD and actual complete-stream n. Signed forgetting is not clipped. Active task seconds sum disjoint completed-task timers, with checkpoint writes reported separately; optimizer updates do not imply matched compute.")
        for suffix in ("csv", "md"):
            path = tables_dir / f"{identifier}_{name}.{suffix}"
            # Preserve full-precision numeric values in CSV.
            if suffix == "csv":
                frame.to_csv(path, index=False, na_rep="")
            # Handle the complementary supported case without inventing observations.
            else:
                path.write_text(f"# {identifier} — {name.replace('_', ' ')}\n\n{status}\n\n{caption}\n\n" + _markdown(frame), encoding="utf-8")
            register(identifier, path, caption, name)
    paired = []
    for dataset, values in native.items():
        paired.append({"dataset": dataset, "comparison": "Learned minus Extra joint", "role": "primary",
                       "unit": "percentage points", "n": values["pair_count"],
                       **{field: values[field] * 100 for field in ("mean_paired_difference", "sample_sd_paired_difference", "ci_95_lower", "ci_95_upper")}})
    native_frame = pd.DataFrame(paired)
    for suffix in ("csv", "md"):
        path = tables_dir / f"T90_primary_native_interval.{suffix}"
        # Preserve full-precision numeric values in CSV.
        if suffix == "csv":
            native_frame.to_csv(path, index=False)
        # Handle the complementary supported case without inventing observations.
        else:
            path.write_text("# T90 — Native primary paired 95% t interval\n\n" + status + "\n\n" + _markdown(native_frame), encoding="utf-8")
        register("T90", path, "Authenticated native learned-minus-extra-joint paired final test-accuracy effect and 95% t interval, rescaled from fractions to percentage points. Preserves the native analysis; n=3 pairs gives weak precision. No secondary interval is invented.", "native_primary_statistics.json")
    _json(provenance / "native_primary_statistics.json", native)
    _json(provenance / "study_design.json", {"record": record, "manifests": manifests})
    _json(provenance / "source_files.json", evidence["sources"])
    # Compact scalar diagnostic evidence, with legacy CKA sanitation. Large
    # histories, confusion arrays and NPZ pixel arrays stay in original runs.
    _json(provenance / "cka_interpretation.json", {"policy": "Only actual aligned integer sample_count >2 and finite CKA with no unavailable reason can be shown; absent legacy counts remain unavailable.",
        "observations_csv": next((item["file"] for item in artifacts if item["data_table"] == "temporal_observations" and item["file"].endswith(".csv")), None)})
    shutil.copyfile(Path(__file__), provenance / "results_package.py")
    for name in ("HYPERPARAMETER_RATIONALE.md", "recipe_sources.json"):
        source = Path(__file__).parent / name
        # Use existing evidence only when the corresponding artifact is present.
        if source.is_file():
            shutil.copyfile(source, provenance / name)
    # Apply the requested compact, detailed or progress presentation policy.
    if details:
        _plots(evidence, figures_dir, register, status, native)
        _qualitative(evidence, manifests, figures_dir, register, status)
    # Every numerical plot has a named CSV copy next to the image.
    for figure, name in (("F01_main_results", "main_results"), ("F02_accuracy_trajectories", "trajectories"),
                         ("F03_paired_effects", "paired_individual"), ("F04_consolidation_changes", "phase_changes"), ("F05_runtime", "resources")):
        # Apply the requested compact, detailed or progress presentation policy.
        if not details:
            continue
        path = figures_dir / f"{figure}_data.csv"
        plot_data = evidence["tables"][name].copy()
        # Keep absent observation tables distinct from numerical zero.
        if figure == "F03_paired_effects" and not plot_data.empty:
            for field in ("ci_95_lower", "ci_95_upper"):
                plot_data[f"native_primary_{field}_pp"] = [native[row.dataset][field] * 100
                    if row.condition == "extra_joint" and row.dataset in native else np.nan for row in plot_data.itertuples()]
        plot_data.to_csv(path, index=False, na_rep="")
        register(figure, path, TABLE_CAPTIONS[name], name)
    (directory / "STUDY_CONTEXT.md").write_text(_context(record, manifests, evidence, status), encoding="utf-8")
    (directory / "CAPTIONS.md").write_text("# Captions and interpretation limits\n\n" + status + "\n\n" + "\n\n".join(
        f"**{item['id']} — {item['file']}**\n\n{item['caption']}\n\nDatasets: {', '.join(item['dataset'])}. {item['sample_count']}. Exact run/file hashes and procedure: ARTIFACT_MANIFEST.json."
        for item in artifacts if item["file"].endswith((".md", ".png"))), encoding="utf-8")
    (directory / "READ_ME_FIRST.md").write_text(f"# Route One writing package\n\n**{status}**\n\n"
        "Read the thesis_summary table and T90_primary_native_interval first. Numeric per-stream source rows, study context and exact provenance are retained. Optional detailed exports add diagnostics and figures with numerical CSV data. RESULT_SUMMARY.json and ARTIFACT_MANIFEST.json supply machine-readable values, source hashes, run identities, sample counts and generating procedure.\n\n"
        "Draft the results chapter from these saved observations. Preserve negative/uncertain findings and missing values, distinguish test outcomes from validation diagnostics, and cite table/figure IDs. Mean ± SD is not a replacement for the native primary paired interval.\n\n"
        f"{LIMITATIONS}\n\n"
        "No models were trained, no test predictions or generated images were newly computed, and no checkpoint/dataset or large intermediate array is included. Missing entries are blank in CSV, unavailable in Markdown and null in JSON. Original run artifacts are read only.\n", encoding="utf-8")
    _json(directory / "RESULT_SUMMARY.json", {"status": status, "seeds": record["seeds"], "completed_streams": len(evidence["runs"]),
        "tables": {name: frame.to_dict("records") for name, frame in evidence["tables"].items()}, "native_primary_statistics": native,
        "limits": LIMITATIONS, "table_ids": table_index})
    _json(directory / "ARTIFACT_MANIFEST.json", {"status": status, "schema_version": 2, "artifacts": artifacts,
        "source_resolution": "Each artifact.source_files.run_ids indexes source_runs below, where exact file paths and SHA-256 values are stored once.",
        "exporter_sha256": _hash(Path(__file__)), "source_runs": evidence["sources"],
        "software": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": __import__("matplotlib").__version__}})
    return directory


def export_results_package(record_path: str | Path, *, progress: bool=False, output_dir: str | Path | None=None, details: bool=False) -> dict:
    """Authenticate saved results and publish the final 24-stream package or progress view.

    Repeated exports reuse an existing identical input/exporter package. A new identity refuses
    to overwrite a package; select a new output_dir explicitly. Original completion artifacts
    are never changed by extraction/publication.

    Args:
        record_path (str | Path): Externally retained frozen_design.json; its identities and
            source hashes must still match.
        progress (bool): True labels an incomplete saved-results view and omits final paired
            inference; False requires all 24 streams.
        output_dir (str | Path | None): Separate package directory. None uses the default
            final or progress package directory beside the frozen campaign record. A sibling
            ZIP archive is also written; original run evidence is unchanged.
        details (bool): True includes saved diagnostic figures and extended views; False keeps
            the compact scalar presentation.

    Returns:
        package (dict): Directory and ZIP Paths plus bool reused. Identical authenticated
            exports are reused; changed inputs require a new destination. Default details=False
            exports compact scalar evidence.

    Raises:
        FileExistsError: If a conflicting package or ZIP already exists.
        ValueError: If frozen/native evidence, target location or an existing package is
            invalid.
        OSError: If source authentication or atomic publication fails.
    """
    from notebooks.thesis.workflow import _campaign, _outputs, analyze_campaign
    record_path = Path(record_path).resolve()
    record, manifests = _campaign(record_path)
    # Apply the requested compact, detailed or progress presentation policy.
    if not progress:
        _check_final_design(record, manifests)
    outputs = {dataset: _outputs(Path(record["studies"][dataset]["manifest_path"]), manifest, complete=not progress)
               for dataset, manifest in manifests.items()}
    # No authenticated completed streams yet; there is no saved progress to
    # export.
    if not any(outputs.values()):
        raise ValueError("No authenticated completed streams yet; there is no saved progress to export.")
    native = {} if progress else analyze_campaign(record_path)
    evidence = extract_saved_evidence(manifests, outputs)
    shared_hashes = {}
    for dataset, runs in outputs.items():
        study_dir = Path(record["studies"][dataset]["manifest_path"]).parent
        for run_id, run in runs.items():
            paths = [study_dir / "manifest.json", study_dir / f"{run_id}.yaml", study_dir / f"{run_id}.completed.json"]
            for path in paths:
                # Use existing evidence only when the corresponding artifact is present.
                if path.is_file():
                    # Hash each shared manifest/configuration artifact once per export.
                    if path not in shared_hashes:
                        shared_hashes[path] = _hash(path)
                    evidence["sources"][f"{dataset}/{run_id}"].append({"path": str(path), "sha256": shared_hashes[path]})
    destination = Path(output_dir).resolve() if output_dir else record_path.parent / ("progress_package" if progress else "writing_package")
    # Package output must be a separate directory from source campaign/run
    # artifacts.
    if destination == record_path.parent or any(destination == Path(run["results_path"]).resolve()
            or destination in Path(run["results_path"]).resolve().parents
            or Path(run["results_path"]).resolve() in destination.parents for run in evidence["runs"]):
        raise ValueError("Package output must be a separate directory from source campaign/run artifacts.")
    rationale_files = [Path(__file__).parent / name for name in ("HYPERPARAMETER_RATIONALE.md", "recipe_sources.json")]
    identity = {"frozen_record_sha256": _hash(record_path), "exporter_sha256": _hash(Path(__file__)),
                "rationale_sha256": {path.name: _hash(path) for path in rationale_files if path.is_file()},
                "source_files": evidence["sources"], "completed": outputs, "progress": progress, "details": details}
    fingerprint = hashlib.sha256(json.dumps(_clean(identity), sort_keys=True).encode()).hexdigest()
    # Use existing evidence only when the corresponding artifact is present.
    if destination.exists():
        saved = destination / "PACKAGE_IDENTITY.json"
        saved_identity = _read(saved) if saved.is_file() else {}
        # Existing package has different inputs.
        if saved_identity.get("sha256") != fingerprint:
            raise FileExistsError(f"Existing package has different inputs. Preserve it and choose a new output_dir: {destination}")
        checked_hashes = {}
        for name, digest in saved_identity.get("package_files", {}).items():
            path = destination / name
            # Existing generated package was changed or is incomplete: the selected artifact.
            if not path.is_file() or _hash(path) != digest:
                raise ValueError(f"Existing generated package was changed or is incomplete: {path}. Preserve it and export to a new directory.")
            checked_hashes[name] = digest
        # Package ZIP is missing: the selected artifact.
        if not destination.with_suffix(".zip").is_file():
            raise FileNotFoundError(f"Package ZIP is missing: {destination.with_suffix('.zip')}. Preserve the directory and export to a new output_dir.")
        with zipfile.ZipFile(destination.with_suffix(".zip")) as archived:
            expected_files = {f"{destination.name}/{path.relative_to(destination).as_posix()}": path
                              for path in destination.rglob("*") if path.is_file()}
            # Existing package ZIP contents differ from the generated directory;
            # preserve it and export to a new output_dir.
            if set(archived.namelist()) != set(expected_files) or len(archived.namelist()) != len(expected_files):
                raise ValueError("Existing package ZIP contents differ from the generated directory; preserve it and export to a new output_dir.")
            checked_hashes["PACKAGE_IDENTITY.json"] = _hash(saved)
            # Existing package ZIP content was changed; preserve it and export to
            # a new output_dir.
            if any(hashlib.sha256(archived.read(name)).hexdigest() != checked_hashes.get(path.relative_to(destination).as_posix())
                   for name, path in expected_files.items()):
                raise ValueError("Existing package ZIP content was changed; preserve it and export to a new output_dir.")
        return {"directory": destination, "zip": destination.with_suffix(".zip"), "reused": True}
    # An archive already exists; choose a new output_dir: the selected artifact.
    if destination.with_suffix(".zip").exists():
        raise FileExistsError(f"An archive already exists; choose a new output_dir: {destination.with_suffix('.zip')}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(tempfile.mkdtemp(prefix=".results-package-", dir=destination.parent))
    try:
        staged = _write_package(staging_parent / destination.name, record, manifests, evidence, native,
                                status="PROGRESS ONLY — NOT FINAL CHAPTER RESULTS" if progress else "FINAL CHAPTER EVIDENCE", details=details)
        _json(staged / "PACKAGE_IDENTITY.json", {"sha256": fingerprint, "identity": identity,
              "package_files": {path.relative_to(staged).as_posix(): _hash(path) for path in staged.rglob("*") if path.is_file()}})
        archive = staging_parent / "package.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            for path in sorted(staged.rglob("*")):
                # Use existing evidence only when the corresponding artifact is present.
                if path.is_file():
                    zipped.write(path, arcname=f"{destination.name}/{path.relative_to(staged).as_posix()}")
        staged.rename(destination)
        archive.replace(destination.with_suffix(".zip"))
    finally:
        shutil.rmtree(staging_parent)
    return {"directory": destination, "zip": destination.with_suffix(".zip"), "reused": False}
