"""Reproducible paired-block manifests and run-level paired inference.

Each block is one independent, complete continual-learning stream.  Every
condition is executed once against that same stream, with condition order
randomized inside the block.  Tasks and epochs are repeated observations inside
a run and are therefore never treated as independent replicates here.

Confirmation manifests are immutable specifications whose trusted hash should
be preregistered separately.  Internal hashes detect accidental corruption;
``validate_frozen_confirmation`` additionally checks the externally retained
hash, and ``validate_confirmation_rerun`` refuses changed rerun requests.

Runner workflow: create and write a manifest, call ``materialize_run_plan`` to
obtain ordered immutable run inputs, execute each full stream, then pass the
single final scalar from every run to ``collect_final_stream_metrics`` and
``write_long_results``.  This boundary deliberately does not accept per-task or
per-epoch measurements as inferential replicates.
"""

from __future__ import annotations

from scipy import stats

import csv

import hashlib

import json

import math

import os

import random

import re

import statistics

from pathlib import Path

from collections.abc import Mapping, Sequence

from common.validation import require


EXPERIMENT_SCHEMA_VERSION = 2
"""Version of the paired-block experiment manifest schema."""

LONG_RESULT_FIELDS = (
    "manifest_hash", 
    "phase", 
    "block_id", 
    "run_id", 
    "condition", 
    "metric", 
    "value", 
    "analysis_unit"
)
"""Canonical columns for run-level long-form result files."""

_PHASES = frozenset(("development", "confirmation"))
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_INDEPENDENT_UNIT = "continual_stream_block"


def _canonical_json(value: object) -> str:
    """Serialize JSON data into a deterministic strict representation.

    Args:
        value (object): JSON-compatible value to serialize.

    Returns:
        str: Canonical JSON text.
    """

    try:
        return json.dumps(
            value, 
            allow_nan=False, 
            ensure_ascii=False, 
            separators=(",", ":"), 
            sort_keys=True
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Experiment specifications must contain finite JSON-compatible "
            "values with string mapping keys."
        ) from error


def _canonical_copy(value: object) -> object:
    """Return a detached canonical JSON-compatible copy.

    Args:
        value (object): JSON-compatible value to copy.

    Returns:
        object: Detached value containing only JSON data types.
    """

    return json.loads(_canonical_json(value))


def _fingerprint(value: object) -> str:
    """Compute the stable SHA-256 digest of a JSON-compatible value.

    Args:
        value (object): Canonicalizable value to fingerprint.

    Returns:
        str: Lowercase hexadecimal SHA-256 digest.
    """

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _derive_stream_seed(seed: int, block_id: str) -> int:
    """Derive a stable child seed for one complete continual stream.

    Args:
        seed (int): Experiment randomization seed.
        block_id (str): Stable independent-block identifier.

    Returns:
        int: Deterministic child seed in TensorFlow's signed range.
    """

    digest = hashlib.sha256(f"{seed}:{block_id}:stream".encode("utf-8")).digest()

    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _normalize_conditions(
    conditions: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    """Validate named experimental conditions and detach their settings.

    Args:
        conditions (Mapping[str, Mapping[str, object]]): At least two named
            condition configuration mappings.

    Returns:
        dict[str, object]: Conditions in canonical name order.
    """

    # A paired contrast requires at least two conditions.
    if not isinstance(conditions, Mapping) or len(conditions) < 2:
        raise ValueError("conditions must map at least two names to settings.")

    normalized: dict[str, object] = {}
    for name in sorted(conditions):
        # Keep names portable across JSON, CSV, and filesystem tooling.
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every condition name must be a nonempty string.")

        settings = conditions[name]

        # Freeze complete settings rather than a display name alone.
        if not isinstance(settings, Mapping):
            raise TypeError(f"Condition {name!r} settings must be a mapping.")

        normalized[name] = _canonical_copy(dict(settings))

    return normalized


def _normalize_base_config(
    base_config: Mapping[str, object] | None
) -> dict[str, object]:
    """Validate and detach the configuration shared by every run.

    Args:
        base_config (Mapping[str, object] | None): Optional complete shared
            configuration.  Condition settings remain separate overrides.

    Returns:
        dict[str, object]: Canonical JSON-compatible shared configuration.
    """

    # An omitted shared configuration preserves the original manifest API.
    if base_config is None:
        return {}
    # Runner inputs must be named configuration mappings, not sequences.
    if not isinstance(base_config, Mapping):
        raise TypeError("base_config must be a mapping or None.")

    copied = _canonical_copy(dict(base_config))
    require(isinstance(copied, dict))

    return copied


def _normalize_analysis_spec(
    analysis_spec: Mapping[str, object] | None, 
    condition_names: Sequence[str]
) -> dict[str, object]:
    """Normalize the preregistered primary paired-analysis specification.

    Args:
        analysis_spec (Mapping[str, object] | None): Optional primary metric and
            contrast declaration.
        condition_names (Sequence[str]): Valid condition names in canonical
            order.

    Returns:
        dict[str, object]: Complete two-sided 95% paired-analysis declaration.
    """

    defaults: dict[str, object] = {
        "primary_metric": "final_average_accuracy", 
        "condition_a": condition_names[0], 
        "condition_b": condition_names[1], 
        "confidence_level": 0.95, 
        "alternative": "two-sided"
    }
    supplied = dict(analysis_spec or {})
    defaults.update(supplied)
    normalized = _canonical_copy(defaults)
    require(isinstance(normalized, dict))

    metric = normalized.get("primary_metric")
    # A confirmatory analysis needs an outcome fixed before testing.
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("analysis_spec.primary_metric must be nonempty.")

    condition_a = normalized.get("condition_a")
    condition_b = normalized.get("condition_b")

    # The contrast must identify two distinct declared conditions.
    if condition_a == condition_b \
    or condition_a not in condition_names \
    or condition_b not in condition_names:
        raise ValueError(
            "analysis_spec must contrast two distinct declared conditions."
        )

    # This module currently implements the declared two-sided 95% interval.
    if normalized.get("confidence_level") != 0.95 \
    or normalized.get("alternative") != "two-sided":
        raise ValueError(
            "analysis_spec must use confidence_level=0.95 and a two-sided test."
        )

    return normalized


def _normalize_stream(
    stream: Mapping[str, object], 
    index: int, 
    experiment_seed: int
) -> tuple[str, dict[str, object]]:
    """Validate one complete continual-stream block.

    Args:
        stream (Mapping[str, object]): Stream with resolved ``class_order`` and
            ``task_groups``; ``block_id`` and ``stream_seed`` are optional.
        index (int): One-based block position used for the default identifier.
        experiment_seed (int): Parent seed used to derive a missing stream seed.

    Returns:
        tuple[str, dict[str, object]]: Block identifier and normalized stream.
    """

    # Require an explicit resolved stream mapping.
    if not isinstance(stream, Mapping):
        raise TypeError("Every continual stream must be a mapping.")

    copied = _canonical_copy(dict(stream))
    require(isinstance(copied, dict))
    block_id = copied.pop("block_id", f"block-{index:04d}")

    # Stable block identifiers define the independent pairing units.
    if not isinstance(block_id, str) or not block_id.strip():
        raise ValueError("Every block_id must be a nonempty string.")

    class_order = copied.get("class_order")
    task_groups = copied.get("task_groups")

    # A complete stream fixes both class order and task boundaries.
    if not isinstance(class_order, list) or not class_order:
        raise ValueError("Every stream needs a nonempty resolved class_order.")

    # Empty or missing tasks do not define a complete continual stream.
    if not isinstance(task_groups, list) \
    or not task_groups \
    or any(not isinstance(group, list) or not group for group in task_groups):
        raise ValueError("Every stream needs nonempty resolved task_groups.")

    flattened = [label for group in task_groups for label in group]
    encoded_order = [_canonical_json(label) for label in class_order]
    encoded_flattened = [_canonical_json(label) for label in flattened]

    # Task groups must be an exact partition of the declared class order.
    if encoded_flattened != encoded_order:
        raise ValueError("Flattening task_groups must equal class_order exactly.")

    # Repeated labels would make class introduction ambiguous.
    if len(set(encoded_order)) != len(encoded_order):
        raise ValueError("class_order must contain unique labels.")

    # Fill a missing per-stream seed from a stable block-specific child stream.
    if "stream_seed" not in copied:
        copied["stream_seed"] = _derive_stream_seed(experiment_seed, block_id)

    copied["stream_seed"] = int(copied["stream_seed"])

    return block_id, copied


def _manifest_payload(manifest: Mapping[str, object]) -> dict[str, object]:
    """Return the portion of a manifest covered by its digest.

    Args:
        manifest (Mapping[str, object]): Full experiment manifest.

    Returns:
        dict[str, object]: Manifest without its self-referential hash field.
    """

    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    copied = _canonical_copy(payload)
    require(isinstance(copied, dict))

    return copied


def create_paired_block_manifest(
    conditions: Mapping[str, Mapping[str, object]], 
    continual_streams: Sequence[Mapping[str, object]], 
    *, 
    seed: int, 
    phase: str = "development", 
    analysis_spec: Mapping[str, object] | None = None, 
    base_config: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Create a seeded conditions-by-complete-stream experiment manifest.

    Args:
        conditions (Mapping[str, Mapping[str, object]]): Named condition settings
            to cross with every stream block.
        continual_streams (Sequence[Mapping[str, object]]): Independent complete
            continual schedules used as paired blocks.
        seed (int): Seed controlling within-block condition execution order.
        phase (str): ``"development"`` or frozen ``"confirmation"`` label.
        analysis_spec (Mapping[str, object] | None): Optional primary paired
            contrast specification.
        base_config (Mapping[str, object] | None): Optional configuration shared
            by every run.  It is sealed by the manifest hash and materialized
            separately from condition-specific settings.

    Returns:
        dict[str, object]: Validated, hash-sealed experiment manifest.
    """

    seed = int(seed)

    # Keep exploratory development distinct from confirmatory testing.
    if phase not in _PHASES:
        raise ValueError("phase must be 'development' or 'confirmation'.")

    # At least two independent blocks are needed for run-level sample variance.
    if isinstance(continual_streams, (str, bytes)) \
    or not isinstance(continual_streams, Sequence) \
    or len(continual_streams) < 2:
        raise ValueError("continual_streams must contain at least two blocks.")

    normalized_conditions = _normalize_conditions(conditions)
    normalized_base_config = _normalize_base_config(base_config)
    condition_names = tuple(normalized_conditions)
    normalized_analysis = _normalize_analysis_spec(
        analysis_spec, 
        condition_names
    )
    rng = random.Random(seed)
    blocks: list[dict[str, object]] = []
    seen_block_ids: set[str] = set()
    seen_stream_specs: set[str] = set()
    for index, stream in enumerate(continual_streams, start=1):
        block_id, normalized_stream = _normalize_stream(stream, index, seed)
        # Pairing identifiers must refer to exactly one independent run block.
        if block_id in seen_block_ids:
            raise ValueError(f"Duplicate block_id: {block_id!r}.")

        seen_block_ids.add(block_id)
        stream_spec = _canonical_json(normalized_stream)

        # Renaming one identical stream does not create an independent block.
        if stream_spec in seen_stream_specs:
            raise ValueError(
                "Duplicate canonical continual-stream specification; distinct "
                "block_id values do not create independent replicates."
            )

        seen_stream_specs.add(stream_spec)
        execution_order = list(condition_names)
        rng.shuffle(execution_order)
        runs = [{
                "run_id": f"{block_id}-run-{order_index + 1:02d}", 
                "condition": condition, 
                "execution_index": order_index
        } for order_index, condition in enumerate(execution_order)]
        blocks.append({
            "block_id": block_id, 
            "stream": normalized_stream, 
            "execution_order": execution_order, 
            "runs": runs
        })

    manifest: dict[str, object] = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION, 
        "phase": phase, 
        "frozen": phase == "confirmation", 
        "spec": {
            "randomization_seed": seed, 
            "independent_unit": _INDEPENDENT_UNIT, 
            "tasks_are_replicates": False, 
            "base_config": normalized_base_config, 
            "conditions": normalized_conditions, 
            "analysis_spec": normalized_analysis, 
            "blocks": blocks
        }
    }
    manifest["manifest_hash"] = _fingerprint(_manifest_payload(manifest))

    return validate_experiment_manifest(manifest)


def validate_experiment_manifest(
    manifest: Mapping[str, object], 
    *, 
    expected_hash: str | None = None
) -> dict[str, object]:
    """Validate schema, crossing, randomization, and optional external hash.

    Args:
        manifest (Mapping[str, object]): Candidate experiment manifest.
        expected_hash (str | None): Optional trusted digest retained outside the
            manifest.

    Returns:
        dict[str, object]: Detached canonical validated manifest.
    """

    # Reject non-mapping documents before canonicalization.
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping.")

    canonical = _canonical_copy(dict(manifest))
    require(isinstance(canonical, dict))

    # A strict top-level schema prevents unhashed operational fields.
    if set(canonical) != {
        "schema_version", "phase", "frozen", "spec", "manifest_hash"
    }:
        raise ValueError("Experiment manifest has unexpected top-level fields.")

    # Reject manifests produced by an incompatible schema.
    if canonical["schema_version"] != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("Unsupported experiment manifest schema version.")

    phase = canonical["phase"]

    # Preserve the declared development/confirmation distinction.
    if phase not in _PHASES:
        raise ValueError("Manifest phase is invalid.")

    # Confirmation is frozen while development remains explicitly mutable.
    if canonical["frozen"] is not (phase == "confirmation"):
        raise ValueError("Manifest frozen flag disagrees with its phase.")

    supplied_hash = canonical["manifest_hash"]

    # Hashes use one canonical lowercase representation.
    if not isinstance(supplied_hash, str) \
    or _HASH_PATTERN.fullmatch(supplied_hash) is None:
        raise ValueError("Manifest hash is invalid.")

    computed_hash = _fingerprint(_manifest_payload(canonical))

    # Detect accidental edits to the sealed document.
    if supplied_hash != computed_hash:
        raise ValueError("Manifest content does not match manifest_hash.")

    # Compare against a separately retained preregistration digest when given.
    if expected_hash is not None and supplied_hash != expected_hash:
        raise ValueError("Manifest does not match the expected external hash.")

    spec = canonical["spec"]
    # Require the complete paired-block design declaration.
    if not isinstance(spec, dict) or set(spec) != {
        "randomization_seed", 
        "independent_unit", 
        "tasks_are_replicates", 
        "base_config", 
        "conditions", 
        "analysis_spec", 
        "blocks"
    }:
        raise ValueError("Manifest spec has unexpected fields.")

    seed = int(spec["randomization_seed"])

    # Make the independent replication level machine-readable.
    if spec["independent_unit"] != _INDEPENDENT_UNIT \
    or spec["tasks_are_replicates"] is not False:
        raise ValueError("Tasks cannot be declared as independent replicates.")

    base_config = _normalize_base_config(spec["base_config"])
    conditions = _normalize_conditions(spec["conditions"])
    condition_names = tuple(conditions)
    analysis_spec = _normalize_analysis_spec(
        spec["analysis_spec"], 
        condition_names
    )

    # Reject a semantically noncanonical condition or analysis declaration.
    if base_config != spec["base_config"] \
    or conditions != spec["conditions"] \
    or analysis_spec != spec["analysis_spec"]:
        raise ValueError("Manifest condition or analysis spec is not canonical.")

    blocks = spec["blocks"]
    # The inferential procedure needs at least two independent pairs.
    if not isinstance(blocks, list) or len(blocks) < 2:
        raise ValueError("Manifest must contain at least two stream blocks.")

    rng = random.Random(seed)
    block_ids: set[str] = set()
    run_ids: set[str] = set()
    stream_specs: set[str] = set()
    for index, block in enumerate(blocks, start=1):
        # Every block has one stream and a full randomized condition crossing.
        if not isinstance(block, dict) or set(block) != {
            "block_id", "stream", "execution_order", "runs"
        }:
            raise ValueError("Manifest block has unexpected fields.")

        block_id = block["block_id"]
        normalized_id, normalized_stream = _normalize_stream(
            {**block["stream"], "block_id": block_id}, 
            index, 
            seed
        )

        # A stream must remain in its canonical representation.
        if normalized_id != block_id or normalized_stream != block["stream"]:
            raise ValueError("Manifest stream is not canonical.")

        # Block identifiers cannot collapse independent experimental units.
        if block_id in block_ids:
            raise ValueError("Manifest contains duplicate block identifiers.")

        block_ids.add(block_id)
        stream_spec = _canonical_json(normalized_stream)

        # A cosmetic block identifier cannot turn a duplicate run into a pair.
        if stream_spec in stream_specs:
            raise ValueError(
                "Manifest contains duplicate canonical stream specifications."
            )

        stream_specs.add(stream_spec)
        expected_order = list(condition_names)
        rng.shuffle(expected_order)

        # Verify both complete crossing and the seeded execution order.
        if block["execution_order"] != expected_order:
            raise ValueError("Block condition order is incomplete or not seeded.")

        expected_runs = [{
            "run_id": f"{block_id}-run-{order_index + 1:02d}", 
            "condition": condition, 
            "execution_index": order_index
        } for order_index, condition in enumerate(expected_order)]

        # Run rows are derived exactly from the randomized block order.
        if block["runs"] != expected_runs:
            raise ValueError("Block run declarations do not match execution order.")

        for run in expected_runs:
            # Run identifiers must remain globally unique in the manifest.
            if run["run_id"] in run_ids:
                raise ValueError("Manifest contains duplicate run identifiers.")

            run_ids.add(run["run_id"])

    return canonical


def validate_frozen_confirmation(
    manifest: Mapping[str, object], 
    *, 
    expected_hash: str
) -> dict[str, object]:
    """Validate a confirmation manifest against its trusted frozen digest.

    Args:
        manifest (Mapping[str, object]): Candidate frozen confirmation manifest.
        expected_hash (str): Preregistered digest retained outside the manifest.

    Returns:
        dict[str, object]: Detached validated confirmation manifest.
    """

    # An external digest must itself use the canonical representation.
    if not isinstance(expected_hash, str) \
    or _HASH_PATTERN.fullmatch(expected_hash) is None:
        raise ValueError("expected_hash must be a lowercase SHA-256 digest.")

    validated = validate_experiment_manifest(
        manifest, 
        expected_hash=expected_hash
    )

    # Development manifests cannot be promoted after outcomes are examined.
    if validated["phase"] != "confirmation" or validated["frozen"] is not True:
        raise ValueError("A frozen confirmation manifest is required.")

    return validated


def validate_confirmation_rerun(
    frozen_manifest: Mapping[str, object], 
    requested_manifest: Mapping[str, object], 
    *, 
    frozen_hash: str, 
    test_results_accessed: bool
) -> str:
    """Authorize only an exact rerun of a preregistered confirmation design.

    Args:
        frozen_manifest (Mapping[str, object]): Original confirmation manifest.
        requested_manifest (Mapping[str, object]): Proposed rerun manifest.
        frozen_hash (str): Externally retained digest of the original manifest.
        test_results_accessed (bool): Whether confirmation outcomes have been
            viewed before requesting the rerun.

    Returns:
        str: Original manifest hash when the rerun is an exact replication.
    """

    original = validate_frozen_confirmation(
        frozen_manifest, 
        expected_hash=frozen_hash
    )
    requested = validate_experiment_manifest(requested_manifest)

    # A rerun must remain a frozen confirmation experiment.
    if requested["phase"] != "confirmation" or requested["frozen"] is not True:
        raise ValueError("A confirmation rerun must remain frozen confirmation.")

    # Outcome-informed changes are an explicit confirmation-protocol violation.
    if requested["manifest_hash"] != original["manifest_hash"] \
    and test_results_accessed:
        raise ValueError(
            "Rejected test-informed rerun change: use a new development study."
        )

    # Even before outcome access, a frozen specification cannot be overwritten.
    if requested["manifest_hash"] != original["manifest_hash"]:
        raise ValueError("A frozen confirmation specification cannot change.")

    return original["manifest_hash"]


def write_experiment_manifest(
    path: str | os.PathLike[str], 
    manifest: Mapping[str, object]
) -> Path:
    """Validate and exclusively create one immutable manifest JSON file.

    Args:
        path (str | os.PathLike[str]): Destination JSON path.
        manifest (Mapping[str, object]): Valid experiment manifest.

    Returns:
        Path: Created manifest path.
    """

    validated = validate_experiment_manifest(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical_json(validated))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

    return target


def read_experiment_manifest(
    path: str | os.PathLike[str], 
    *, 
    expected_hash: str | None = None
) -> dict[str, object]:
    """Read and validate one paired-block experiment manifest.

    Args:
        path (str | os.PathLike[str]): Manifest JSON path.
        expected_hash (str | None): Optional trusted external digest.

    Returns:
        dict[str, object]: Detached validated manifest.
    """

    with Path(path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    return validate_experiment_manifest(manifest, expected_hash=expected_hash)


def materialize_run_plan(
    manifest: Mapping[str, object], 
    *, 
    expected_hash: str | None = None
) -> list[dict[str, object]]:
    """Expand a manifest into ordered, self-contained runner inputs.

    Each returned entry represents one condition trained and evaluated over one
    complete continual stream.  Entries follow block order and the seeded
    within-block execution order recorded in the manifest.

    Args:
        manifest (Mapping[str, object]): Paired-block experiment manifest.
        expected_hash (str | None): Optional externally retained manifest hash.

    Returns:
        list[dict[str, object]]: Ordered run identifiers, condition settings,
        and complete stream specifications for a runner or CLI adapter.
    """

    validated = validate_experiment_manifest(
        manifest, 
        expected_hash=expected_hash
    )
    spec = validated["spec"]
    conditions = spec["conditions"]
    base_config = spec["base_config"]
    plan: list[dict[str, object]] = []
    for block in spec["blocks"]:
        for run in block["runs"]:
            condition = run["condition"]
            plan.append({
                "manifest_hash": validated["manifest_hash"], 
                "phase": validated["phase"], 
                "analysis_unit": _INDEPENDENT_UNIT, 
                "block_id": block["block_id"], 
                "run_id": run["run_id"], 
                "execution_index": run["execution_index"], 
                "condition": condition, 
                "base_config": _canonical_copy(base_config), 
                "condition_settings": _canonical_copy(conditions[condition]), 
                "stream": _canonical_copy(block["stream"])
            })

    return plan


def collect_final_stream_metrics(
    manifest: Mapping[str, object], 
    run_metrics: Mapping[str, object], 
    *, 
    metric: str | None = None, 
    expected_hash: str | None = None
) -> list[dict[str, object]]:
    """Create one canonical final-metric row per planned full-stream run.

    Args:
        manifest (Mapping[str, object]): Manifest that defines every required
            condition-by-stream run.
        run_metrics (Mapping[str, object]): Mapping from each planned ``run_id``
            to its single final scalar value.
        metric (str | None): Metric name, defaulting to the frozen primary metric.
        expected_hash (str | None): Optional externally retained manifest hash.

    Returns:
        list[dict[str, object]]: Long-form rows suitable for
        :func:`write_long_results` and :func:`paired_run_statistics`.
    """

    # Require named final values rather than an order-dependent sequence.
    if not isinstance(run_metrics, Mapping):
        raise TypeError("run_metrics must map planned run_id values to scalars.")

    # Keep identifier comparisons and error reporting deterministic.
    if any(not isinstance(run_id, str) for run_id in run_metrics):
        raise TypeError("Every run_metrics key must be a string run_id.")

    validated = validate_experiment_manifest(
        manifest, 
        expected_hash=expected_hash
    )
    plan = materialize_run_plan(validated)
    planned_ids = {str(run["run_id"]) for run in plan}
    supplied_ids = set(run_metrics)

    # Missing or foreign run identifiers would break the complete crossing.
    if supplied_ids != planned_ids:
        missing = sorted(planned_ids - supplied_ids)
        extra = sorted(supplied_ids - planned_ids)
        raise ValueError(
            f"run_metrics must cover every planned run exactly; "
            f"missing={missing}, extra={extra}."
        )

    primary_metric = validated["spec"]["analysis_spec"]["primary_metric"]
    metric_name = primary_metric if metric is None else metric

    # Preserve a portable nonempty name for the final outcome column.
    if not isinstance(metric_name, str) or not metric_name.strip():
        raise ValueError("metric must be a nonempty string.")

    # Confirmation outcomes must retain the preregistered primary metric.
    if validated["phase"] == "confirmation" and metric_name != primary_metric:
        raise ValueError(
            "A confirmation manifest cannot override its primary metric."
        )

    rows: list[dict[str, object]] = []
    for run in plan:
        rows.append(_normalize_result_row({
            "manifest_hash": run["manifest_hash"], 
            "phase": run["phase"], 
            "block_id": run["block_id"], 
            "run_id": run["run_id"], 
            "condition": run["condition"], 
            "metric": metric_name, 
            "value": run_metrics[run["run_id"]], 
            "analysis_unit": _INDEPENDENT_UNIT
        }))

    return rows


def _normalize_result_row(row: Mapping[str, object]) -> dict[str, object]:
    """Validate one run-level long-form result row.

    Args:
        row (Mapping[str, object]): Candidate result row.

    Returns:
        dict[str, object]: Canonical typed result row.
    """

    # A fixed schema prevents task-level columns from being mistaken for runs.
    if not isinstance(row, Mapping) or set(row) != set(LONG_RESULT_FIELDS):
        raise ValueError(
            "Result rows must contain exactly the canonical long-form fields."
        )

    normalized = dict(row)
    manifest_hash = normalized["manifest_hash"]

    # Link every result to one immutable experiment specification.
    if not isinstance(manifest_hash, str) \
    or _HASH_PATTERN.fullmatch(manifest_hash) is None:
        raise ValueError("Result manifest_hash is invalid.")

    # Keep development and confirmation outcomes distinguishable.
    if normalized["phase"] not in _PHASES:
        raise ValueError("Result phase is invalid.")

    for field in ("block_id", "run_id", "condition", "metric"):
        value = normalized[field]
        # Identifiers must survive lossless CSV serialization.
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Result {field} must be a nonempty string.")

    # Refuse task- or epoch-level pseudoreplicates at the schema boundary.
    if normalized["analysis_unit"] != _INDEPENDENT_UNIT:
        raise ValueError(
            "Paired inference requires analysis_unit='continual_stream_block'."
        )
    try:
        numeric_value = float(normalized["value"])
    except (TypeError, ValueError) as error:
        raise ValueError("Result value must be numeric.") from error

    # Nonfinite outcomes cannot support the declared t interval.
    if not math.isfinite(numeric_value):
        raise ValueError("Result value must be finite.")

    normalized["value"] = numeric_value

    return normalized


def write_long_results(
    path: str | os.PathLike[str], 
    rows: Sequence[Mapping[str, object]]
) -> Path:
    """Exclusively create a canonical run-level long-form CSV file.

    Args:
        path (str | os.PathLike[str]): Destination CSV path.
        rows (Sequence[Mapping[str, object]]): Run-level result rows.

    Returns:
        Path: Created CSV path.
    """

    # Strings are sequences but never valid collections of result rows.
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of result mappings.")

    normalized = [_normalize_result_row(row) for row in rows]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LONG_RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(normalized)
        stream.flush()
        os.fsync(stream.fileno())

    return target


def read_long_results(
    path: str | os.PathLike[str]
) -> list[dict[str, object]]:
    """Read and validate a canonical run-level long-form CSV file.

    Args:
        path (str | os.PathLike[str]): Result CSV path.

    Returns:
        list[dict[str, object]]: Validated rows with numeric values restored.
    """

    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        # Column identity and order are part of the portable result schema.
        if tuple(reader.fieldnames or ()) != LONG_RESULT_FIELDS:
            raise ValueError("Result CSV header does not match the canonical schema.")

        return [_normalize_result_row(row) for row in reader]


def paired_run_statistics(
    rows: Sequence[Mapping[str, object]], 
    *, 
    condition_a: str, 
    condition_b: str, 
    metric: str, 
    manifest: Mapping[str, object] | None = None, 
    expected_hash: str | None = None
) -> dict[str, object]:
    """Summarize one condition contrast across independent stream blocks.

    Differences are ``condition_a - condition_b``.  Each block contributes
    exactly one pair; tasks and epochs cannot enter as replicates.

    Args:
        rows (Sequence[Mapping[str, object]]): Canonical long-form run results.
        condition_a (str): First condition in the signed contrast.
        condition_b (str): Second condition in the signed contrast.
        metric (str): Run-level metric to analyze.
        manifest (Mapping[str, object] | None): Optional experiment manifest.
            It is mandatory for confirmation results and verifies the planned
            runs when supplied for development results.
        expected_hash (str | None): Optional trusted external manifest digest.
            It is mandatory for confirmation results.

    Returns:
        dict[str, object]: Pair count, means, paired differences, sample SD,
        standard error, t statistic, 95% t interval, and optional SciPy p-value.
    """

    # The signed contrast requires two distinct condition names.
    if condition_a == condition_b:
        raise ValueError("condition_a and condition_b must differ.")
    # A trusted hash has no verification target without its manifest.
    if manifest is None and expected_hash is not None:
        raise ValueError("expected_hash requires manifest.")

    validated_manifest = None

    # Development callers may opt into the same manifest consistency checks.
    if manifest is not None:
        validated_manifest = validate_experiment_manifest(
            manifest, 
            expected_hash=expected_hash
        )

        # A supplied confirmation design must be externally frozen up front.
        if validated_manifest["phase"] == "confirmation":
            # Confirmation inference requires an independently supplied hash.
            if expected_hash is None:
                raise ValueError(
                    "Confirmation statistics require a frozen "
                    "manifest and its expected_hash."
                )

            validated_manifest = validate_frozen_confirmation(
                validated_manifest, 
                expected_hash=expected_hash
            )
            preregistered = validated_manifest["spec"]["analysis_spec"]
            requested = (condition_a, condition_b, metric)
            frozen = (
                preregistered["condition_a"], 
                preregistered["condition_b"], 
                preregistered["primary_metric"]
            )

            # Reversing the contrast or changing the outcome changes the hypothesis.
            if requested != frozen:
                raise ValueError(
                    "Confirmation statistics must use the preregistered "
                    "condition_a, condition_b, and primary_metric."
                )

    normalized = [_normalize_result_row(row) for row in rows]
    selected = [
        row for row in normalized
        if row["metric"] == metric and 
        row["condition"] in (condition_a, condition_b)
    ]

    # Do not infer an effect from absent run-level measurements.
    if not selected:
        raise ValueError("No rows match the requested paired contrast.")

    hashes = {row["manifest_hash"] for row in selected}
    phases = {row["phase"] for row in selected}

    # Never pair runs across different experiment specifications or phases.
    if len(hashes) != 1 or len(phases) != 1:
        raise ValueError("Paired rows must share one manifest and one phase.")

    selected_hash = next(iter(hashes))
    selected_phase = next(iter(phases))

    # Confirmation inference must be anchored to an externally frozen design.
    if selected_phase == "confirmation":
        # Reject confirmation rows without their validated manifest credentials.
        if validated_manifest is None or expected_hash is None:
            raise ValueError(
                "Confirmation statistics require a frozen "
                "manifest and its expected_hash."
            )

        # A development manifest cannot authorize confirmation inference.
        if validated_manifest["phase"] != "confirmation" \
        or validated_manifest["frozen"] is not True:
            raise ValueError("Confirmation statistics require a frozen manifest.")

    # Supplied manifests must describe the exact rows being analyzed.
    if validated_manifest is not None:
        # Bind every selected row to the supplied manifest hash and phase.
        if selected_hash != validated_manifest["manifest_hash"] \
        or selected_phase != validated_manifest["phase"]:
            raise ValueError("Paired rows do not match the supplied manifest.")

        planned = materialize_run_plan(validated_manifest)
        planned_cells = {
            str(run["run_id"]): (
                str(run["block_id"]), 
                str(run["condition"])
            )
            for run in planned
            if run["condition"] in (condition_a, condition_b)
        }
        selected_cells = {
            str(row["run_id"]): (
                str(row["block_id"]), 
                str(row["condition"])
            )
            for row in selected
        }

        # Complete manifest verification prevents cherry-picked stream blocks.
        if selected_cells != planned_cells:
            raise ValueError(
                "Paired rows do not cover the manifest's planned contrast exactly."
            )

    by_condition: dict[str, dict[str, float]] = {
        condition_a: {}, 
        condition_b: {}
    }
    run_ids: set[str] = set()
    for row in selected:
        condition = str(row["condition"])
        block_id = str(row["block_id"])
        run_id = str(row["run_id"])

        # Each run identifier refers to one result row for this metric.
        if run_id in run_ids:
            raise ValueError("Duplicate run_id in the selected result rows.")

        run_ids.add(run_id)

        # A block-condition cell must contain exactly one run-level outcome.
        if block_id in by_condition[condition]:
            raise ValueError("Duplicate condition result within a stream block.")

        by_condition[condition][block_id] = float(row["value"])

    a_blocks = set(by_condition[condition_a])
    b_blocks = set(by_condition[condition_b])

    # Fail rather than silently discard an incomplete experimental pair.
    if a_blocks != b_blocks:
        raise ValueError("Every stream block must contain both condition results.")
    # Sample SD and a t interval require at least two independent pairs.
    if len(a_blocks) < 2:
        raise ValueError("At least two complete stream blocks are required.")

    block_ids = sorted(a_blocks)
    a_values = [by_condition[condition_a][block_id] for block_id in block_ids]
    b_values = [by_condition[condition_b][block_id] for block_id in block_ids]
    differences = [a - b for a, b in zip(a_values, b_values)]
    pair_count = len(differences)
    mean_difference = statistics.fmean(differences)
    sample_sd = statistics.stdev(differences)
    standard_error = sample_sd / math.sqrt(pair_count)
    critical = float(stats.t.ppf(0.975, pair_count - 1))
    margin = critical * standard_error

    # Define the degenerate zero-variance statistic without dividing by zero.
    if standard_error == 0.0 and mean_difference == 0.0:
        t_statistic = 0.0
        p_value = None
    # A constant nonzero difference has an unbounded t statistic.
    elif standard_error == 0.0:
        t_statistic = math.copysign(math.inf, mean_difference)
        p_value = 0.0
    # Use the paired-difference standard error for the ordinary case.
    else:
        t_statistic = mean_difference / standard_error
        p_value = float(stats.ttest_rel(a_values, b_values).pvalue)
        if not math.isfinite(p_value):
            p_value = None

    return {
        "manifest_hash": selected_hash, 
        "phase": selected_phase, 
        "analysis_unit": _INDEPENDENT_UNIT, 
        "condition_a": condition_a, 
        "condition_b": condition_b, 
        "metric": metric, 
        "pair_count": pair_count, 
        "block_ids": block_ids, 
        "condition_a_mean": statistics.fmean(a_values), 
        "condition_b_mean": statistics.fmean(b_values), 
        "paired_differences": differences, 
        "mean_paired_difference": mean_difference, 
        "sample_sd_paired_difference": sample_sd, 
        "standard_error": standard_error, 
        "degrees_of_freedom": pair_count - 1, 
        "t_statistic": t_statistic, 
        "t_critical_95": critical, 
        "ci_95_lower": mean_difference - margin, 
        "ci_95_upper": mean_difference + margin, 
        "paired_t_p_value": p_value,
        "tasks_used_as_replicates": False
    }


__all__ = [
    "EXPERIMENT_SCHEMA_VERSION", 
    "LONG_RESULT_FIELDS", 
    "create_paired_block_manifest", 
    "collect_final_stream_metrics", 
    "materialize_run_plan", 
    "paired_run_statistics", 
    "read_experiment_manifest", 
    "read_long_results", 
    "validate_confirmation_rerun", 
    "validate_experiment_manifest", 
    "validate_frozen_confirmation", 
    "write_experiment_manifest", 
    "write_long_results"
]
