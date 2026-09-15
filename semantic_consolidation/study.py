"""Prepare, run, and analyze paired route-one streams with common.experiment.

A block is a complete seed/class-order stream. The test split is only evaluated
for a frozen confirmation design. Development uses validation endpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import yaml

from allocation_study.artifacts import (
    native_study_metadata, read_completed_runs, replace_completed_index,
    validate_completed_artifact, validate_study_source, write_completed_artifact,
)
from common.config import Config, resolve_continual_schedule
from common.continual_reporting import continual_metrics
from common.experiment import (
    collect_final_stream_metrics, create_paired_block_manifest,
    materialize_run_plan, paired_run_statistics, read_experiment_manifest,
    write_experiment_manifest, write_long_results,
)
from semantic_consolidation.config import RouteConfig, _merge, load_route_config, validate_route_config


def planned_config(entry: dict, manifest_path: str | Path) -> RouteConfig:
    """Resolve an exact condition/stream specification into the ordinary APIs.

    Args:
        entry (dict): One materialized manifest run containing its stream, condition
            overrides, base settings and authenticated identities.
        manifest_path (str | Path): Path to the paired experiment manifest; executable run
            files and completion artifacts reside alongside it.

    Returns:
        config (RouteConfig): RouteConfig with the manifest stream/order/seed and run
            identities applied and validated.

    Raises:
        KeyError: If a required materialized-run field is absent.
        ValueError: If the resolved condition cannot execute under the route protocol.
    """

    values = _merge(entry["base_config"], entry["condition_settings"])
    config = RouteConfig(**values)
    stream, project = entry["stream"], config.common
    continual = project.continually_learn
    continual.class_order = stream["class_order"]
    continual.task_groups = stream["task_groups"]
    continual.class_num = len(stream["class_order"])
    continual.class_order_mode = "fixed"
    continual.task_order_mode = "fixed"
    continual.seed = stream["stream_seed"]
    project.training.seed = stream["stream_seed"]
    config.route.seed = stream["stream_seed"]
    continual.experiment_phase = entry["phase"]
    continual.experiment_manifest_path = str(Path(manifest_path).resolve())
    continual.experiment_manifest_hash = entry["manifest_hash"]
    continual.experiment_run_id = entry["run_id"]
    project.training.project_tag = entry["run_id"]
    validate_route_config(config)
    return config


def validate_planned_config(config: RouteConfig) -> None:
    """Bind every scientific setting, including route coefficients, to the manifest.

    Args:
        config (RouteConfig): Validated configuration tree, with the data, model and
            experiment settings consumed by this operation.

    Returns:
        validated (None): None; checks scientific settings/source against the frozen
            manifest, allowing artifact destination relocation.

    Raises:
        ValueError: If confirmation lacks a manifest or run identity, source or scientific
            settings differ from the design.
        OSError: If the required manifest/source cannot be read.
    """

    continual = config.common.continually_learn
    # Standalone development is permitted; confirmation always needs a frozen manifest.
    if not continual.experiment_manifest_path:
        # Confirmation requires a frozen paired experiment manifest.
        if continual.experiment_phase == "confirmation":
            raise ValueError("Confirmation requires a frozen paired experiment manifest.")
        return
    manifest = read_experiment_manifest(
        continual.experiment_manifest_path, expected_hash=continual.experiment_manifest_hash
    )
    matching = [entry for entry in materialize_run_plan(manifest)
                if entry["run_id"] == continual.experiment_run_id]
    # The configured run does not occur exactly once in its manifest.
    if len(matching) != 1:
        raise ValueError("The configured run does not occur exactly once in its manifest.")
    expected = asdict(planned_config(matching[0], continual.experiment_manifest_path))
    actual = asdict(config)
    # Artifact destinations may be relocated. Scientific knobs may not change.
    for data in (expected, actual):
        data["common"]["training"].pop("results_path", None)
        data["common"]["training"].pop("project_tag", None)
        data["common"]["continually_learn"].pop("checkpoint_dir", None)
        data["common"]["continually_learn"].pop("resume_from", None)
    # Run settings differ from the manifest; prepare a new study for a changed design.
    if actual != expected:
        raise ValueError("Run settings differ from the manifest; prepare a new study for a changed design.")
    validate_study_source(manifest, "semantic_consolidation")


def prepare_study(
    template: RouteConfig, directory: str | Path, seeds: list[int],
    conditions: dict[str, dict] | None = None, phase: str = "development",
) -> Path:
    """Write immutable paired manifest and one executable route YAML per run.

    Each independent seed generates one saved class permutation. All treatments
    within that block share it and the seed. Defaults compare learned control
    with random modulation and extra joint updates. No training is launched.

    Args:
        template (RouteConfig): Validated RouteConfig used as the common scientific basis
            for paired conditions.
        directory (str | Path): Output directory for this operation, resolved using ordinary
            pathlib path semantics.
        seeds (list[int]): At least two distinct integer full-stream seeds in [0, 2**32).
        conditions (dict[str, dict] | None): Optional mapping of condition names to nested
            common/route overrides; None selects the documented default controls.
        phase (str): development for validation-only exploration or confirmation for an
            authenticated frozen test design.

    Returns:
        manifest_path (Path): Path to the saved paired manifest and executable per-run YAML
            files; no training starts.

    Raises:
        FileExistsError: If the study directory exists.
        ValueError: If seeds, conditions, schedule or planned configurations are invalid.
        OSError: If the design cannot be written.
    """

    validate_route_config(template)
    # Stream seeds must be integers in [0, 2**32).
    if any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
           or not 0 <= int(seed) < 2 ** 32 for seed in seeds):
        raise ValueError("Stream seeds must be integers in [0, 2**32).")
    seeds = [int(seed) for seed in seeds]
    # Use at least two distinct full-stream seeds.
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Use at least two distinct full-stream seeds.")
    directory = Path(directory).resolve()
    # Preparation never overwrites an existing design or its outcomes.
    if directory.exists():
        raise FileExistsError(f"Study directory already exists: {directory}")
    conditions = conditions if conditions is not None else {
        "learned": {"route": {"condition": "learned"}},
        "random": {"route": {"condition": "random"}},
        "extra_joint": {"route": {"condition": "extra_joint"}},
    }
    # A paired study requires at least two conditions.
    if len(conditions) < 2:
        raise ValueError("A paired study requires at least two conditions.")
    continual = template.common.continually_learn
    order, groups = resolve_continual_schedule(
        continual.class_num, continual.class_order, continual.task_groups,
        task_size=continual.task_size, seed=seeds[0],
    )
    streams = []
    for index, seed in enumerate(seeds):
        shuffled = np.random.default_rng(seed).permutation(order).tolist()
        cursor, scheduled = 0, []
        for group in groups:
            scheduled.append(shuffled[cursor:cursor + len(group)])
            cursor += len(group)
        streams.append({"block_id": f"stream-{index + 1:02d}", "stream_seed": seed,
                        "class_order": shuffled, "task_groups": scheduled})
    names = list(conditions)
    contrast_a = "learned" if "learned" in names else names[0]
    contrast_b = "extra_joint" if "extra_joint" in names else next(n for n in names if n != contrast_a)
    manifest = create_paired_block_manifest(
        conditions, streams, seed=seeds[0], phase=phase,
        base_config={"common": asdict(template.common), "route": asdict(template.route)},
        analysis_spec={"condition_a": contrast_a, "condition_b": contrast_b,
                       "primary_metric": "final_average_accuracy",
                       "native_route_study": native_study_metadata("semantic_consolidation")},
    )
    manifest_path = directory / "manifest.json"
    # Validate every planned configuration before making the design persistent.
    planned = [(entry, planned_config(entry, manifest_path)) for entry in materialize_run_plan(manifest)]
    write_experiment_manifest(manifest_path, manifest)
    for entry, config in planned:
        config.common.training.results_path = str(directory / "runs")
        values = {"common": asdict(config.common), "route": asdict(config.route)}
        values["route"]["noise_levels"] = list(config.route.noise_levels)
        with (directory / f"{entry['run_id']}.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(values, stream, sort_keys=True)
    return manifest_path


def _read_study_manifest(manifest_path: Path, expected_hash: str | None) -> dict:
    """Require a separately retained design digest before confirmation access.

    A manifest's own digest detects corruption, but a modified design can be
    resealed. It cannot authenticate itself against the preregistered design.
    Development keeps the convenient optional external digest.

    Args:
        manifest_path (Path): Path to the paired experiment manifest; executable run files
            and completion artifacts reside alongside it.
        expected_hash (str | None): Optional independently retained manifest SHA-256;
            required for confirmation access.

    Returns:
        manifest (dict): Authenticated experiment dict with current production-source
            verification.

    Raises:
        ValueError: If manifest/source identity is invalid or confirmation lacks an
            independently retained expected hash.
        OSError: If manifest/source files cannot be read.
    """

    manifest = read_experiment_manifest(manifest_path, expected_hash=expected_hash)
    # Confirmation requires an externally retained expected_hash (--expected-hash).
    if manifest["phase"] == "confirmation" and expected_hash is None:
        raise ValueError("Confirmation requires an externally retained expected_hash (--expected-hash).")
    validate_study_source(manifest, "semantic_consolidation")
    return manifest


def _completed_metrics(matrix: object, tasks: int) -> dict[str, float]:
    """Validate complete class-incremental observations before summarizing a run.

    General reporting supports missing cells. An accepted completed study
    instead requires every learned-task cell, no future-task observation, and
    fractional accuracy values. Metric formulas remain owned by common.

    Args:
        matrix (object): Numeric task-by-task accuracy matrix: finite fractions in learned-
            task cells and NaN/None in future-task cells.
        tasks (int): Positive integer number of tasks in the complete planned stream.

    Returns:
        metrics (dict[str, float]): Dict of Python float continual metrics computed by
            common.continual_reporting.

    Raises:
        ValueError: If shape, complete lower-triangular fractional observations or
            unavailable future-task cells are invalid.
    """

    values = np.asarray(matrix, dtype="float64")
    # A run is complete only at the exact scheduled final task boundary.
    if values.shape != (tasks, tasks) or tasks < 1:
        raise ValueError("Completed outcome matrix must match the full scheduled task count.")
    observed = values[np.tril_indices(tasks)]
    # Missing or out-of-range observations cannot be averaged away as a valid run.
    if not np.isfinite(observed).all() or np.any((observed < 0.) | (observed > 1.)):
        raise ValueError("Completed outcome matrix needs finite fractional accuracies in every learned-task cell.")
    # Future-task observations violate this lower-triangular evaluation protocol.
    if not np.isnan(values[np.triu_indices(tasks, 1)]).all():
        raise ValueError("Future-task accuracy cells must be unavailable.")
    return continual_metrics(values)


def run_study(manifest_path: str | Path, *, expected_hash: str | None = None) -> dict:
    """Execute complete streams sequentially and save outcomes after each run.

    Partial study recovery is deliberately not automatic: reusing an outcome
    requires checking its artifacts and design. A new output index is exclusive.

    Args:
        manifest_path (str | Path): Path to the paired experiment manifest; executable run
            files and completion artifacts reside alongside it.
        expected_hash (str | None): Optional independently retained manifest SHA-256;
            required for confirmation access.

    Returns:
        completed (dict): Mapping from run IDs to completed full-stream artifact/metric
            records; progress is saved after each accepted run.

    Raises:
        FileExistsError: If a completion index already exists.
        ValueError: If manifest, run files, source identity or completed outcome contracts
            fail.
        OSError: If a design, run or output artifact cannot be read/written.
    """

    import tensorflow as tf
    from semantic_consolidation.runner import run

    manifest_path = Path(manifest_path).resolve()
    manifest = _read_study_manifest(manifest_path, expected_hash)
    plan = materialize_run_plan(manifest, expected_hash=expected_hash)
    configs = []
    # Validate the entire executable plan before creating outcomes or training
    # any cell. A later invalid file must not consume a partial study budget.
    for entry in plan:
        config = load_route_config(manifest_path.parent / f"{entry['run_id']}.yaml")
        identity = config.common.continually_learn
        # Planned run file has a different run or manifest identity.
        if identity.experiment_run_id != entry["run_id"] or (
            identity.experiment_manifest_hash != manifest["manifest_hash"]
        ) or Path(identity.experiment_manifest_path).resolve() != manifest_path:
            raise ValueError("Planned run file has a different run or manifest identity.")
        validate_planned_config(config)
        configs.append(config)
    index_path = manifest_path.parent / "completed_runs.json"
    with index_path.open("x", encoding="utf-8") as stream:
        json.dump({}, stream)
    outputs = {}
    for entry, config in zip(plan, configs):
        # A source edit between paired streams changes the declared implementation.
        validate_study_source(manifest, "semantic_consolidation")
        validate_planned_config(config)
        started = time.perf_counter()
        result = run(config)
        validate_study_source(manifest, "semantic_consolidation")
        details = result["model"]["continual_details"]
        matrix_name = "validation_accuracy_matrix" if manifest["phase"] == "development" else "ordinary_accuracy_matrix"
        matrix = np.asarray(details[matrix_name], dtype="float64")
        metrics = _completed_metrics(matrix, len(entry["stream"]["task_groups"]))
        from semantic_consolidation.controller import _json_value
        outputs[entry["run_id"]] = {
            "manifest_hash": manifest["manifest_hash"],
            "run_id": entry["run_id"], "condition": entry["condition"],
            "results_path": result["results_path"], "seconds": time.perf_counter() - started,
            "total_updates": sum(record["total_updates"] for record in result["route_records"]),
            "accuracy_matrix": _json_value(matrix), "accuracy_matrix_source": matrix_name,
            "metrics": metrics,
        }
        outputs[entry["run_id"]]["completed_artifact"] = write_completed_artifact(
            manifest_path.parent, outputs[entry["run_id"]],
        )
        replace_completed_index(index_path, outputs)
        del result
        tf.keras.backend.clear_session()
    return outputs


def analyze_study(manifest_path: str | Path, *, expected_hash: str | None = None) -> dict:
    """Use common's complete-stream paired statistics for the primary contrast.

    Args:
        manifest_path (str | Path): Path to the paired experiment manifest; executable run
            files and completion artifacts reside alongside it.
        expected_hash (str | None): Optional independently retained manifest SHA-256;
            required for confirmation access.

    Returns:
        analysis (dict): Dict of paired stream statistics and evidence scope, also saved
            with long-form complete outcomes.

    Raises:
        ValueError: If source, manifest, completed artifacts, stream coverage or summary
            metrics are inconsistent.
        OSError: If study inputs or analysis outputs cannot be read/written.
    """

    manifest_path = Path(manifest_path).resolve()
    manifest = _read_study_manifest(manifest_path, expected_hash)
    outputs = read_completed_runs(manifest_path.parent / "completed_runs.json")
    planned = {entry["run_id"]: entry for entry in materialize_run_plan(manifest)}
    for run_id, result in outputs.items():
        # Completed outcomes do not belong to this manifest and its planned runs.
        if run_id not in planned or result.get("manifest_hash") != manifest["manifest_hash"] or (
            result.get("run_id") != run_id or result.get("condition") != planned[run_id]["condition"]
        ):
            raise ValueError("Completed outcomes do not belong to this manifest and its planned runs.")
    spec = manifest["spec"]["analysis_spec"]
    metric = spec["primary_metric"]
    verified_artifacts, scalar_only_runs = [], []
    for run_id, result in outputs.items():
        verified = validate_completed_artifact(
            manifest_path.parent, result, required=manifest["phase"] == "confirmation",
        )
        # Record exactly which run contents were checked beyond the summary index.
        if verified:
            verified_artifacts.append(run_id)
        value = result["metrics"][metric]
        # Every supported route study uses a fractional final accuracy endpoint.
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0. <= value <= 1.:
            raise ValueError("The primary final-average-accuracy outcome must be a finite fraction in [0, 1].")
        # New exports retain the exact source matrix; older indexes remain usable.
        if "accuracy_matrix" in result:
            expected_source = "validation_accuracy_matrix" if manifest["phase"] == "development" else "ordinary_accuracy_matrix"
            # Validation outcomes cannot be mislabeled as locked-test confirmation outcomes.
            if result.get("accuracy_matrix_source") != expected_source:
                raise ValueError("Completed accuracy matrix source differs from the study phase.")
            checked = _completed_metrics(result["accuracy_matrix"], len(planned[run_id]["stream"]["task_groups"]))
            # Saved summary metrics disagree with the completed accuracy matrix.
            if any(name not in result["metrics"] or not np.isclose(checked[name], result["metrics"][name], rtol=0., atol=1e-12)
                   for name in checked):
                raise ValueError("Saved summary metrics disagree with the completed accuracy matrix.")
        # Matrix-free outcomes retain exploratory compatibility only.
        else:
            # Confirmation must establish the complete scheduled evaluation trajectory.
            if manifest["phase"] == "confirmation":
                raise ValueError("Confirmation requires a complete saved accuracy matrix for every run.")
            scalar_only_runs.append(run_id)
    rows = collect_final_stream_metrics(
        manifest, {run_id: result["metrics"][metric] for run_id, result in outputs.items()},
        expected_hash=manifest["manifest_hash"],
    )
    statistics = paired_run_statistics(
        rows, condition_a=spec["condition_a"], condition_b=spec["condition_b"],
        metric=metric, manifest=manifest, expected_hash=manifest["manifest_hash"],
    )
    statistics["artifact_validation"] = {
        "source": validate_study_source(manifest, "semantic_consolidation"),
        "hashed_completed_artifact_runs": verified_artifacts,
        "legacy_scalar_only_runs": scalar_only_runs,
        "scope": "Recomputed saved matrices and authenticated per-run contents where present; legacy scalar-only development cannot establish matrix completeness.",
    }
    write_long_results(manifest_path.parent / "paired_results.csv", rows)
    # Degenerate t statistics may be infinite; label unavailable JSON values.
    from semantic_consolidation.controller import _json_value
    with (manifest_path.parent / "paired_statistics.json").open("x", encoding="utf-8") as stream:
        json.dump(_json_value(statistics), stream, indent=2, allow_nan=False)
    return statistics


def main(argv: list[str] | None = None) -> None:
    """Dispatch study preparation, authenticated execution, or paired analysis from the CLI.

    Args:
        argv (list[str] | None): Command-line strings; None reads the current process
            arguments.

    Returns:
        completed (None): None; parses command-line arguments, executes the requested
            operation and prints its artifact location.

    Raises:
        SystemExit: If arguments are invalid or help is requested.
        OSError: If requested input/output artifacts cannot be accessed.
        ValueError: If the selected configuration or experiment contract is invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    prepare.add_argument("--phase", choices=["development", "confirmation"], default="development")
    for name in ("run", "analyze"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--expected-hash", help="Separately retained manifest SHA-256; required for confirmation")
    args = parser.parse_args(argv)
    # Preparation freezes executable inputs without launching any model training.
    if args.command == "prepare":
        print(prepare_study(load_route_config(args.config), args.output, args.seeds, phase=args.phase))
    # Execution authenticates the supplied design before consuming the study budget.
    elif args.command == "run":
        run_study(args.manifest, expected_hash=args.expected_hash)
    # Analysis accepts only complete outcomes belonging to the authenticated plan.
    else:
        print(analyze_study(args.manifest, expected_hash=args.expected_hash))


# Run this module directly while keeping imports free of execution side effects.
if __name__ == "__main__":
    main()
