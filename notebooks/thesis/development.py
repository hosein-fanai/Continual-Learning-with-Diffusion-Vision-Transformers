"""Small saved-only validation review of one stream; no training or prediction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from notebooks.thesis.results_package import _unique_task_records, aligned_phase_endpoints, saved_task_runtime


def _read(path: Path) -> object:
    """Read a saved JSON observation without changing its source.

    Args:
        path (Path): JSON file to read; relative paths use the current working directory.

    Returns:
        saved (object): Decoded JSON evidence, leaving the source file unchanged.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If JSON decoding fails.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def review_development_run(run_dir: str | Path, *, output_dir: str | Path | None=None) -> dict[str, pd.DataFrame]:
    """Summarize actual phase coverage, transfer, replay, runtime and memory.

    This deliberately does not issue an automatic adequacy/convergence verdict. Every displayed
    task is a repeated observation of ONE training stream.

    Args:
        run_dir (str | Path): Original saved run directory; input artifacts are read only.
        output_dir (str | Path | None): Separate directory for CSV tables. None returns the
            tables without writing; a path also saves each table outside the native run.

    Returns:
        tables (dict[str, pd.DataFrame]): Per-task validation learning, matched phase
            differences, gate coverage and separate resource measurements. output_dir=None
            returns tables only; a separate path also saves CSV. No convergence verdict is
            inferred.

    Raises:
        ValueError: If observations are not validation, task identities conflict, phase cohorts
            mismatch or output overlaps native evidence.
        OSError: If required saved files cannot be read or CSV cannot be written.
    """
    run = Path(run_dir).resolve()
    route = _read(run / "route_metrics.json")
    observer = _read(run / "section11.json")
    _unique_task_records(route, len(route), "Saved route diagnostics")
    _unique_task_records(observer.get("tasks", []), len(route), "Saved observer diagnostics")
    tables = {}
    learning = []
    for row in observer.get("tasks", []):
        # This view must never label held-out test outcomes as development evidence.
        if row.get("split", "validation") != "validation":
            raise ValueError("Development diagnostics require saved validation observations.")
        outcome = row.get("outcomes", {})
        learning.append({"task": row["task"], "split": row.get("split", "validation"),
            **{name + "_percent": 100 * value if not isinstance(value, bool) and isinstance(value, (int, float)) and np.isfinite(value) else None
               for name in ("accuracy", "old_accuracy", "new_accuracy") for value in [outcome.get(name)]}})
    tables["learning_and_retention"] = pd.DataFrame(learning)
    coverage, effects, memory = [], [], []
    for task in route:
        for phase in ("acquisition", "consolidation"):
            row = task.get(phase, {})
            counts = list(row.get("focus_class_updates", {}).values())
            coverage.append({"task": task["task"], "phase": phase,
                             "updates": row.get("updates"), "gates_observed": len(counts),
                             "minimum_visits_per_gate": min(counts) if counts else None,
                             "maximum_visits_per_gate": max(counts) if counts else None,
                             "untrained_gates": len(row.get("untrained_focus_classes", [])) if counts else None,
                             "example_presentations": row.get("example_draws")})
        before, after = task.get("before_consolidation") or {}, task.get("after_consolidation") or {}
        aligned = aligned_phase_endpoints(before, after)
        for key, scale, unit in (("clean_accuracy", 100., "percentage points"),
                                 ("old_accuracy", 100., "percentage points"),
                                 ("new_accuracy", 100., "percentage points"),
                                 ("representation.centered_effective_rank", 1., "rank"),
                                 ("frozen_target_alignment.aggregates.selected_gates.hidden_target_cosine", 1., "cosine")):
            def lookup(values: object) -> float | int | None:
                """Finite saved numeric endpoint for the enclosing dot-separated key, or None for
                absent, boolean or nonfinite values.

                Args:
                    values (object): Nested saved endpoint mapping; the enclosing measurement
                        key selects the scalar.

                Returns:
                    scalar (float | int | None): Finite saved numeric endpoint for the enclosing
                        dot-separated key, or None for absent, boolean or nonfinite values.

                Raises:
                    None: Missing or unsupported endpoint values become None.
                """
                for part in key.split("."):
                    values = values.get(part) if isinstance(values, dict) else None
                return values if not isinstance(values, bool) and isinstance(values, (int, float)) and np.isfinite(values) else None
            start, end = (lookup(before), lookup(after)) if aligned else (None, None)
            effects.append({"task": task["task"], "measurement": key,
                            "before": start * scale if start is not None else None,
                            "after": end * scale if end is not None else None,
                            "before_after_unit": "%" if scale == 100. else unit,
                            "change_after_minus_before": (end-start)*scale if aligned and start is not None and end is not None else None,
                            "change_unit": unit, "split": "validation",
                            "unavailable_reason": None if aligned else "consolidation_boundary_unavailable"})
    tables["gate_coverage"] = pd.DataFrame(coverage)
    tables["deployed_classifier_and_hidden_phase_changes"] = pd.DataFrame(effects)
    tables["optimizer_work"] = pd.read_csv(run / "route_resources.csv")
    costs = pd.read_csv(run / "task_metrics.csv")
    costs = costs.loc[costs.phase.eq("resource")].copy()
    keys = ["current_examples_available", "current_examples_exposed", "training_examples_total",
            "replay/candidate_count", "replay/selected_count", "seconds/generator_sampling",
            "seconds/generator_fit", "seconds/task_total"]
    tables["task_exposure_and_cost"] = costs.loc[costs.metric.isin(keys),
        ["task_index", "metric", "value"]].rename(columns={"task_index": "task_zero_based"})
    tables["task_exposure_and_cost"]["unit"] = np.where(
        tables["task_exposure_and_cost"]["metric"].str.startswith("seconds/"), "seconds", "examples")
    timing = saved_task_runtime(costs, len(route))
    tables["measured_task_runtime"] = pd.DataFrame([{
        "completed_tasks_with_timer": timing["n_tasks"],
        "sum_measured_task_seconds": timing["seconds"], "unavailable_reason": timing["reason"],
        "scope": "Sum of disjoint active task totals including committed earlier segments and route work; measured progress writes and downtime are separate; uncommitted lost work is unavailable."}])
    checkpoint_costs = costs.loc[costs.metric.eq("checkpointing/io_seconds")].copy()
    checkpoint_costs["metric"] = "seconds/task_total"
    checkpoint_timing = saved_task_runtime(checkpoint_costs, len(route))
    tables["measured_checkpoint_io"] = pd.DataFrame([{
        "completed_tasks_with_timer": checkpoint_timing["n_tasks"],
        "recorded_checkpoint_seconds": checkpoint_timing["seconds"],
        "unavailable_reason": checkpoint_timing["reason"],
        "scope": "Measured progress-checkpoint writes, separate from active task time; interrupted unfinished writes and lost uncommitted work are unavailable."}])
    for row in observer.get("tasks", []):
        for boundary, measure in (("end_of_fit", row.get("resource_measurement", {})),
                                  ("completed_teacher", row.get("post_boundary_teacher", {}).get("resource_measurement", {}))):
            # Skip absent resource observations without inventing zero usage.
            if not measure:
                continue
            memory.append({"task": row["task"], "boundary": boundary, "kind": "sampled process RSS",
                           "bytes": measure.get("sampled_process_peak_rss_bytes"),
                           "scope": "Cumulative sampled maximum since observer creation, not an instantaneous task peak."})
            for device, allocator in measure.get("tf_allocator_devices", {}).items():
                memory.append({"task": row["task"], "boundary": boundary, "kind": f"TensorFlow allocator {device}",
                               "bytes": (allocator or {}).get("peak"),
                               "scope": "Cumulative allocator high water; not total GPU memory occupancy."})
    tables["sampled_and_allocator_memory"] = pd.DataFrame(memory)
    tables["replay_self_consistency"] = pd.json_normalize(observer.get("tasks", [])).reindex(columns=[
        "task", "generated_memory.available", "generated_memory.summary.class_coverage",
        "generated_memory.summary.label_consistency", "generated_memory.summary.pixel_diversity"])
    # Keep actual inventory names instead of pretending their sum is process memory.
    flat = pd.json_normalize(observer.get("tasks", []))
    inventory = [name for name in flat if "inventory" in name and ("bytes" in name or "scope" in name)]
    tables["recorded_tensor_storage"] = flat.reindex(columns=["task", *inventory])
    # Save separate CSV views only when a destination is requested.
    if output_dir is not None:
        destination = Path(output_dir).resolve()
        # Write the development review outside the original run artifacts.
        if destination == run or run in destination.parents:
            raise ValueError("Write the development review outside the original run artifacts.")
        destination.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(destination / f"{name}.csv", index=False)
        (destination / "README.md").write_text(
            "# Saved single-stream validation diagnostics\n\nSource: " + str(run) +
            "\n\nOne completed training stream; no across-stream uncertainty or confirmation-test efficacy claim. "
            "Read all tasks, especially the last CIFAR-100 task. Gate visits count updates, not distinct images. "
            "Accuracy is percent; accuracy differences are percentage points. Phase changes require the same fixed validation rows. "
            "Replay self-consistency is not independent image-quality validation. Missing measurements stay unavailable.\n",
            encoding="utf-8")
    return tables
