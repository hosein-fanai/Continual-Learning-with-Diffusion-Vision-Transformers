"""Materialize the existing mechanistic controls as executable paired runs.

Preparation never loads data or starts training. Optional time allowances come
from an explicitly supplied learned pilot, not invented example durations.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from common.config import resolve_continual_schedule
from semantic_consolidation.config import RouteConfig, load_route_config
from semantic_consolidation.study import prepare_study


def control_conditions() -> dict[str, dict]:
    """Name treatments while preserving the template's other scientific knobs.

    Returns:
        conditions (dict[str, dict]): Fresh dict of learned, random, identity, replacement-
            CE and extra-joint route overrides.

    Raises:
        None: This function only constructs ordinary dictionaries.
    """

    return {
        "learned": {"route": {"condition": "learned"}},
        "random": {"route": {"condition": "random"}},
        "identity_infonce": {"route": {
            "condition": "random", "modulation_init_std": 0.0,
        }},
        "ce_only": {"route": {"condition": "no_consolidation"}},
        "extra_joint": {"route": {"condition": "extra_joint"}},
    }


def measured_allowances(template: RouteConfig, path: str | Path) -> tuple[list[float], dict]:
    """Read acquisition + snapshot + consolidation fit seconds for every task.

    The pilot's hardware, model, data, and noise protocol must match the planned
    comparison. Records permit checking phase counts, not proving that match.
    Diagnostics and general route setup are excluded from the training allowance.

    Args:
        template (RouteConfig): Validated RouteConfig used as the common scientific basis
            for paired conditions.
        path (str | Path): Input or output filesystem path used by the operation described
            above.

    Returns:
        timing (tuple[list[float], dict]): (allowances, provenance): list of Python float
            seconds and a dict recording the source hash and scope.

    Raises:
        OSError: If timing records cannot be read.
        ValueError: If JSON, task order, phase budgets or measured durations are invalid.
    """

    path = Path(path).resolve()
    payload = path.read_bytes()
    records = json.loads(payload)
    continual = template.common.continually_learn
    _, groups = resolve_continual_schedule(
        continual.class_num, continual.class_order, continual.task_groups,
        task_size=continual.task_size, seed=continual.seed,
    )
    # Each planned task needs its own observed allowance; never extrapolate it.
    if not isinstance(records, list) or len(records) != len(groups):
        raise ValueError("Timing records must contain one learned record per planned task.")
    allowances = []
    for task, record in enumerate(records, 1):
        # Only ordered learned-pilot records define this comparison's allowance.
        if not isinstance(record, dict) or record.get("task") != task or record.get("condition") != "learned":
            raise ValueError("Timing records must be ordered learned tasks starting at one.")
        try:
            acquisition, consolidation = record["acquisition"], record["consolidation"]
            # Different phase budgets would calibrate a different treatment.
            if acquisition["updates"] != template.route.acquisition_steps or (
                consolidation["updates"] != template.route.consolidation_steps
            ):
                raise ValueError("Pilot phase budgets differ from the planned template.")
            components = [acquisition["seconds"], record["target_snapshot_seconds"],
                          consolidation["seconds"]]
        except (KeyError, TypeError) as error:
            raise ValueError("Timing records lack measured phase/snapshot durations.") from error
        # Reject unavailable or invalid timings rather than manufacturing a budget.
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or (
            not math.isfinite(value) or value < 0
        ) for value in components) or sum(components) <= 0:
            raise ValueError("Measured durations must be finite, nonnegative, and positive in total.")
        allowances.append(float(sum(components)))
    return allowances, {
        "timing_records_path": str(path), "timing_records_sha256": sha256(payload).hexdigest(),
        "allowance_seconds_per_task": allowances,
        "allowance_components": ["acquisition.seconds", "target_snapshot_seconds", "consolidation.seconds"],
        "scope": "One calibration stream's per-task allowances, shared by planned seed blocks.",
        "excluded": ["validation diagnostics", "general route setup"],
        "hardware_and_protocol_match": "Must be established from the pilot configuration and hardware record.",
    }


def prepare_controls(
    template: RouteConfig, directory: str | Path, seeds: list[int],
    phase: str = "development", timing_records: str | Path | None = None,
) -> Path:
    """Create frozen paired seed/order runs, optionally adding measured-time joint training.

    Args:
        template (RouteConfig): Validated RouteConfig used as the common scientific basis
            for paired conditions.
        directory (str | Path): Output directory for this operation, resolved using ordinary
            pathlib path semantics.
        seeds (list[int]): At least two distinct integer full-stream seeds in [0, 2**32).
        phase (str): development or confirmation, forwarded unchanged to the paired-study
            preparation API.
        timing_records (str | Path | None): Optional path to actual learned-pilot
            route_metrics.json with matching per-task phase budgets.

    Returns:
        manifest_path (Path): Path to the prepared immutable paired manifest; no model is
            trained.

    Raises:
        FileExistsError: If the study directory already exists.
        ValueError: If seeds, controls or optional measured timing records are invalid.
        OSError: If study files cannot be written.
    """

    conditions = control_conditions()
    timing_basis = None
    # A supplied measured pilot enables the optional time-budget condition.
    if timing_records is not None:
        allowances, timing_basis = measured_allowances(template, timing_records)
        conditions["time_matched_joint"] = {"route": {
            "condition": "time_matched_joint", "extra_joint_seconds": allowances,
        }}
    path = prepare_study(template, directory, seeds, conditions=conditions, phase=phase)
    # Preserve timing provenance only when measured allowances entered the design.
    if timing_basis is not None:
        with (path.parent / "timing_basis.json").open("x", encoding="utf-8") as stream:
            json.dump(timing_basis, stream, indent=2, allow_nan=False)
    return path


def main(argv: list[str] | None = None) -> None:
    """Parse preparation options and print the created paired manifest path.

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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--phase", choices=["development", "confirmation"], default="development")
    parser.add_argument("--timing-records", type=Path,
                        help="An actual learned pilot route_metrics.json with matching phase budgets.")
    args = parser.parse_args(argv)
    print(prepare_controls(load_route_config(args.config), args.output, args.seeds,
                           phase=args.phase, timing_records=args.timing_records))


# Direct module execution prepares configurations without launching training.
if __name__ == "__main__":
    main()
