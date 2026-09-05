"""Define reproducible paired-stream experiments and analyze final run outcomes.

A block is one complete continual-learning stream, and every condition runs
once within each block. Condition execution order is randomized reproducibly
inside blocks. Tasks and epochs are repeated observations within a run, not
independent replicates. Callers remain responsible for creating scientifically
independent stream blocks; the module rejects duplicate canonical stream
specifications and incomplete or duplicated condition-by-block result cells.

Manifests record the shared base configuration, condition overrides, resolved
class/task schedules, seeds, analysis contrast, and execution plan. Canonical
JSON and SHA-256 bind those values. Confirmation manifests require unchanged
specifications; externally retained hashes authenticate frozen designs where
the confirmation APIs require them. Internal hashes detect content changes
but do not independently establish preregistration.

Use ``create_paired_block_manifest`` and ``write_experiment_manifest``, expand
the plan with ``materialize_run_plan``, execute each full stream, and collect
one final scalar per run with ``collect_final_stream_metrics``. Long-form CSV
helpers preserve the fixed run-level schema. ``paired_run_statistics`` reports
the signed A-minus-B paired difference, sample uncertainty, and a two-sided
95% Student-t interval/test over complete stream blocks."""

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
    """Serialize manifest-compatible data with deterministic JSON formatting.

    Args:
        value (object): JSON-compatible scalar/container. Mappings should use
            string keys, sequences should contain supported values, and numeric
            values must be finite. JSON's own scalar/key conversions apply.

    Returns:
        str: Unicode JSON with sorted mapping keys, compact separators, and
        ``allow_nan=False``. List order is preserved, so schedule and execution
        order affect the resulting fingerprint. No caller objects are modified.

    Raises:
        TypeError: If JSON serialization rejects an unsupported object, non-finite
            number, incompatible mapping keys, or another invalid container value.
            Serialization TypeError/ValueError exceptions are wrapped with a
            manifest-compatibility explanation."""

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
    """Detach a JSON-compatible value by canonical serialization and parsing.

    Args:
        value (object): Scientific experiment settings supported by
            ``_canonical_json``. Tuples become lists and mapping keys follow JSON
            conversion; arbitrary live models, arrays, and callbacks are unsupported.

    Returns:
        object: Newly parsed Python JSON data with canonical mapping order. Scalar
        values retain JSON semantics and nested containers no longer alias input
        containers. None remains None; empty containers remain empty.

    Raises:
        TypeError: If the value cannot be represented by strict canonical JSON."""

    return json.loads(_canonical_json(value))


def _fingerprint(value: object) -> str:
    """Hash the complete canonical JSON representation of an experiment value.

    Args:
        value (object): JSON-compatible settings or manifest payload. The caller
            must exclude self-referential hash fields before hashing a manifest.

    Returns:
        str: Lowercase 64-character SHA-256 hex digest of canonical UTF-8 JSON.
        Mapping insertion order does not change the digest; sequence order and
        distinct serialized values do. This is content identity, not authentication
        unless compared with a separately trusted digest.

    Raises:
        TypeError: If canonical JSON cannot serialize the value."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _derive_stream_seed(seed: int, block_id: str) -> int:
    """Derive a stable stream seed from the experiment seed and block identity.

    Args:
        seed (int): Parent experiment randomization seed. It is formatted into the
            hash input together with the block ID and a fixed stream namespace.
        block_id (str): Stable identifier for the independent stream block. Changing
            this ID changes the derived seed when the stream has no explicit seed.

    Returns:
        int: Integer in ``[0, 2**31 - 2]`` derived from the first eight SHA-256
        digest bytes. This does not consume or mutate Python/NumPy global RNG state."""

    digest = hashlib.sha256(f"{seed}:{block_id}:stream".encode("utf-8")).digest()

    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _normalize_conditions(
    conditions: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    """Validate named treatments and detach their settings in canonical order.

    Args:
        conditions (Mapping[str, Mapping[str, object]]): At least two condition
            names mapped to JSON-compatible override/settings mappings. Names must
            be nonempty strings. Empty settings mappings are permitted.

    Returns:
        dict[str, object]: New mapping sorted by condition name, with each settings
        mapping converted to detached canonical JSON data. These sorted names define
        the default analysis contrast and input to seeded execution-order shuffling.

    Raises:
        ValueError: If fewer than two conditions are supplied or a name is empty.
        TypeError: If a condition's settings are not a mapping, names cannot be
            sorted, or settings cannot be serialized canonically."""

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
    """Detach the configuration shared by every condition and stream block.

    Args:
        base_config (Mapping[str, object] | None): Shared complete project settings.
            None is accepted as an empty shared configuration. Condition-specific
            settings are stored separately and are not merged by this helper.

    Returns:
        dict[str, object]: Detached canonical JSON mapping, or ``{}`` for None.
        Nested containers are copied, so later edits cannot mutate the input settings.

    Raises:
        TypeError: If the supplied value is neither a mapping nor None, or contains
            values unsupported by strict canonical JSON."""

    # An omitted shared configuration preserves the original manifest API.
    if base_config is None:
        return {}
    # Runner inputs must be named configuration mappings, not sequences.
    if not isinstance(base_config, Mapping):
        raise TypeError("base_config must be a mapping or None.")

    return _canonical_copy(dict(base_config))


def _normalize_analysis_spec(
    analysis_spec: Mapping[str, object] | None,
    condition_names: Sequence[str]
) -> dict[str, object]:
    """Resolve the fixed two-sided primary paired-analysis contract.

    Args:
        analysis_spec (Mapping[str, object] | None): Optional analysis overrides.
            None/empty input uses ``primary_metric="final_average_accuracy"``, the
            first two condition names as A/B, ``confidence_level=0.95``, and
            ``alternative="two-sided"``. Additional JSON-compatible metadata fields
            are retained. Other confidence levels/alternatives are unsupported.
        condition_names (Sequence[str]): At least two declared condition names in
            canonical order, used for defaults and contrast-membership validation.

    Returns:
        dict[str, object]: Detached primary metric, distinct A/B contrast,
        confidence level, alternative, and any additional supplied analysis metadata.

    Raises:
        ValueError: If the metric is empty, the contrast repeats/uses undeclared
            conditions, or the requested confidence level/alternative is unsupported.
        TypeError: If analysis metadata cannot be converted to canonical JSON."""

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
    """Validate one resolved continual schedule and complete its stream identity.

    Args:
        stream (Mapping[str, object]): JSON-compatible stream settings containing
            nonempty ``class_order`` and ``task_groups`` sequences that canonicalize
            to lists. Groups must be nonempty and flatten exactly to the unique class
            order. Optional ``block_id`` and ``stream_seed`` override derived values;
            other stream metadata is retained.
        index (int): One-based stream position used to default the block ID to
            ``block-XXXX`` when no ID is supplied.
        experiment_seed (int): Parent seed used with the resolved block ID to derive
            a missing stream seed. An explicit stream_seed is normalized with int.

    Returns:
        tuple[str, dict[str, object]]: ``(block_id, normalized_stream)``. The stream
        is a detached canonical copy with integer ``stream_seed`` and without its
        separate block_id field. Label equality/uniqueness uses canonical JSON,
        allowing supported JSON labels without relying on their Python hashability.

    Raises:
        TypeError: If stream is not a mapping or contains unsupported JSON values.
        ValueError: If identifiers/groups/order are missing or inconsistent, labels
            repeat, or an explicit stream seed cannot be converted to int."""

    # Require an explicit resolved stream mapping.
    if not isinstance(stream, Mapping):
        raise TypeError("Every continual stream must be a mapping.")

    copied = _canonical_copy(dict(stream))
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
    """Extract the manifest fields covered by its content digest.

    Args:
        manifest (Mapping[str, object]): Full manifest or a candidate payload.
            A top-level ``manifest_hash`` is removed when present; its absence is
            accepted so the same helper can prepare an initial hash input.

    Returns:
        dict[str, object]: Shallow mapping copy without ``manifest_hash``. Nested
        values remain shared references until a caller serializes/canonicalizes them.
        The original mapping is unchanged."""

    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    return payload


def create_paired_block_manifest(
    conditions: Mapping[str, Mapping[str, object]],
    continual_streams: Sequence[Mapping[str, object]],
    *,
    seed: int,
    phase: str = "development",
    analysis_spec: Mapping[str, object] | None = None,
    base_config: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Create a reproducible conditions-by-complete-stream experiment plan.

    Conditions are sorted canonically, then independently shuffled within each
    block using one local Random(seed) stream. Every condition appears once per
    block. Caller-supplied class/task schedules are validated rather than invented;
    missing stream IDs/seeds are derived deterministically. The resulting payload
    is SHA-256 sealed and validated before return.

    Args:
        conditions (Mapping[str, Mapping[str, object]]): At least two nonempty
            condition names mapped to JSON-compatible treatment settings/overrides.
        continual_streams (Sequence[Mapping[str, object]]): At least two complete
            resolved stream specifications containing class_order and task_groups.
            Block IDs and canonical stream specifications must be unique. Tasks
            within a stream do not count as independent blocks.
        seed (int): Experiment randomization seed, normalized with int. It controls
            condition execution order and supplies missing per-stream seeds.
        phase (str): Experiment phase. Default ``"development"`` allows exploratory
            analysis; ``"confirmation"`` sets the frozen flag. Other values fail.
        analysis_spec (Mapping[str, object] | None): Primary metric/contrast settings.
            Default None selects final_average_accuracy, the first two sorted
            conditions as A/B, and a two-sided 95% paired analysis.
        base_config (Mapping[str, object] | None): Shared project settings sealed
            into the manifest. Default None produces an empty shared mapping.
            Condition overrides remain separate for the downstream runner to apply.

    Returns:
        dict[str, object]: Detached manifest with ``schema_version``, ``phase``,
        ``frozen``, ``manifest_hash``, and ``spec``. The spec contains the seed,
        independent-unit declaration, base_config, conditions, analysis_spec, and
        ordered blocks. Each block contains its stream, shuffled execution_order,
        and run declarations with run_id, condition, and zero-based execution_index.
        No files are created and no runs are executed.

    Raises:
        ValueError: If the phase, pairing design, identifiers, schedule partition,
            contrast, or stream uniqueness is invalid.
        TypeError: If inputs have unsupported mapping/sequence/JSON types."""

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
    """Check a manifest's schema, identity, schedule, and complete seeded crossing.

    The top-level/spec/block fields must match the supported schema. Validation
    recomputes the content digest, checks phase/frozen agreement, normalizes shared
    settings and schedules, reproduces each shuffled condition order, and verifies
    every declared run. Duplicate blocks, stream specifications, or run IDs fail.

    Args:
        manifest (Mapping[str, object]): Candidate paired-block manifest, including
            its supplied ``manifest_hash`` and schema version.
        expected_hash (str | None): Trusted digest retained separately from the
            manifest. Default None checks internal content identity only; a supplied
            digest must match the validated manifest hash exactly.

    Returns:
        dict[str, object]: Detached canonical validated manifest. The caller's
        mapping is unchanged. Validation checks declared design consistency, not
        whether the actual training runs were executed independently or as declared.

    Raises:
        TypeError: If manifest is not a mapping or contains unsupported JSON values.
        ValueError: If fields, schema version, hashes, canonical settings, schedule,
            condition crossing, seeded order, or identity uniqueness are inconsistent."""

    # Reject non-mapping documents before canonicalization.
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping.")

    canonical = _canonical_copy(dict(manifest))

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
    """Authenticate a frozen confirmation manifest against a trusted digest.

    Args:
        manifest (Mapping[str, object]): Candidate confirmation manifest whose
            schema/content will be fully validated.
        expected_hash (str): Required externally retained lowercase 64-character
            SHA-256 digest. This value should come from the separately frozen design,
            not merely be copied from an untrusted manifest under review.

    Returns:
        dict[str, object]: Detached validated manifest with phase ``confirmation``
        and frozen=True, whose content matches the external digest. No file or
        configuration mutation occurs.

    Raises:
        ValueError: If the digest is malformed, content differs, or the manifest is
            not a frozen confirmation design.
        TypeError: If manifest content is not supported canonical JSON data."""

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
    """Require an exact rerun of an externally frozen confirmation specification.

    Args:
        frozen_manifest (Mapping[str, object]): Original frozen confirmation design.
        requested_manifest (Mapping[str, object]): Proposed rerun design, validated
            independently before its hash is compared with the original.
        frozen_hash (str): Trusted externally retained hash of the original design.
        test_results_accessed (bool): Whether confirmation outcomes were already
            viewed. True gives a specific test-informed-change error for changed
            designs; False still does not permit changing the frozen specification.

    Returns:
        str: Original manifest hash when both manifests represent exactly the same
        frozen design. This helper validates intent; it neither launches a run nor
        writes an authorization artifact.

    Raises:
        ValueError: If either design/hash is invalid, the rerun is not confirmation,
            or any hash-covered setting changes, regardless of outcome access.
        TypeError: If a manifest cannot be represented as canonical JSON."""

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
    """Validate and exclusively create an immutable canonical JSON manifest file.

    Args:
        path (str | os.PathLike[str]): Destination JSON path. Parent directories are
            created when necessary; the destination itself must not already exist.
        manifest (Mapping[str, object]): Complete manifest validated before writing.
            Its canonical compact representation is followed by one newline.

    Returns:
        pathlib.Path: Created path in the caller's relative/absolute form. The file
        is flushed and fsynced before return. Writing uses exclusive creation and
        does not replace an existing frozen artifact; a write failure can leave a
        partial newly created file for the caller to inspect.

    Raises:
        FileExistsError: If the destination already exists.
        OSError: If directory creation, writing, flushing, or syncing fails.
        ValueError: If manifest consistency validation fails.
        TypeError: If the manifest contains unsupported JSON values."""

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
    """Read canonical experiment data and validate its declared design and hash.

    Args:
        path (str | os.PathLike[str]): Existing UTF-8 JSON manifest file.
        expected_hash (str | None): Optional trusted external digest. Default None
            validates only the internal content hash and schema/design consistency.

    Returns:
        dict[str, object]: Detached canonical validated manifest. Reading performs
        no training, artifact rewriting, or global random-state changes.

    Raises:
        OSError: If the manifest cannot be opened or read.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the decoded manifest fails schema/design/hash validation.
        TypeError: If decoded content is not a supported manifest mapping."""

    with Path(path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    return validate_experiment_manifest(manifest, expected_hash=expected_hash)


def materialize_run_plan(
    manifest: Mapping[str, object],
    *,
    expected_hash: str | None = None
) -> list[dict[str, object]]:
    """Expand a validated manifest into ordered inputs for complete-stream runs.

    Entries follow block order and the seeded condition execution order within
    each block. Each entry represents one condition trained over one complete
    continual stream; this function does not expand tasks into independent runs or
    merge condition overrides into the shared base configuration.

    Args:
        manifest (Mapping[str, object]): Paired-block experiment manifest to validate
            and expand.
        expected_hash (str | None): Optional externally retained digest. Default
            None performs internal consistency validation without external identity.

    Returns:
        list[dict[str, object]]: Entries with ``manifest_hash``, ``phase``,
        ``analysis_unit``, ``block_id``, ``run_id``, ``execution_index``, ``condition``,
        ``base_config``, ``condition_settings``, and ``stream``. Each settings/stream
        value is copied independently so edits to one run plan do not mutate another
        entry or the manifest. No runs are executed or files written.

    Raises:
        ValueError: If the manifest or trusted hash fails validation.
        TypeError: If manifest data cannot be canonicalized."""

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
    """Convert one final scalar per planned stream run into canonical result rows.

    Args:
        manifest (Mapping[str, object]): Manifest defining every required
            condition-by-stream run. It is validated before outcomes are collected.
        run_metrics (Mapping[str, object]): Exact mapping from planned string run_id
            values to final numeric outcomes. No runs may be missing or extra, and
            each outcome must convert to a finite float. Per-task trajectories are
            not aggregated automatically into a final outcome.
        metric (str | None): Outcome name. Default None uses the manifest's primary
            metric. Development may explicitly name another nonempty metric;
            confirmation must retain its frozen primary metric.
        expected_hash (str | None): Trusted external manifest digest. Default None
            checks internal identity only; supply it to bind collection externally.

    Returns:
        list[dict[str, object]]: One row per planned run in execution-plan order,
        with the exact ``LONG_RESULT_FIELDS`` schema: manifest_hash, phase, block_id,
        run_id, condition, metric, float value, and continual_stream_block analysis
        unit. No CSV is written until ``write_long_results`` is called.

    Raises:
        TypeError: If outcomes are not a mapping, run IDs are not strings, or
            manifest content cannot be canonicalized.
        ValueError: If run coverage is incomplete/foreign, the selected metric
            violates confirmation, values are nonnumeric/non-finite, or design
            validation fails."""

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
    # Use the manifest's primary metric unless a comparison metric was supplied.
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
    """Validate and normalize one finite complete-stream result record.

    Args:
        row (Mapping[str, object]): Mapping with exactly ``LONG_RESULT_FIELDS``.
            It must contain a lowercase SHA-256 manifest_hash, supported phase,
            nonempty string block_id/run_id/condition/metric, the analysis unit
            ``continual_stream_block``, and a finite float-convertible value.

    Returns:
        dict[str, object]: Shallow row copy with ``value`` normalized to float.
        Other fields remain unchanged. It validates a row's declared schema, not
        its membership in a particular manifest or independence from other runs.

    Raises:
        ValueError: If fields differ from the schema, identifiers/phase/unit are
            invalid, or value is nonnumeric/non-finite."""

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
    """Exclusively create a UTF-8 CSV containing validated run-level result rows.

    Args:
        path (str | os.PathLike[str]): Destination CSV path. Missing parents are
            created; an existing destination is never silently replaced.
        rows (Sequence[Mapping[str, object]]): Complete-stream result rows normalized
            individually against ``LONG_RESULT_FIELDS``. Input order is preserved.
            An empty sequence is allowed and writes the stable header only.

    Returns:
        pathlib.Path: Created CSV path. The header has canonical column order,
        numeric values are normalized to float, and the file is flushed/fsynced
        before return. Completeness of a paired contrast is checked by collection
        or statistical-analysis functions, not this writer alone.

    Raises:
        TypeError: If rows is not a non-text Sequence.
        ValueError: If any result row violates the canonical schema.
        FileExistsError: If the destination already exists.
        OSError: If creation, writing, or syncing fails."""

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
    """Read fixed-schema CSV results and restore numeric outcome values.

    Args:
        path (str | os.PathLike[str]): Existing UTF-8 result CSV with header exactly
            matching ``LONG_RESULT_FIELDS`` in its declared order.

    Returns:
        list[dict[str, object]]: Validated rows in file order, with finite float
        ``value`` fields. A header-only file returns ``[]``. The function does not
        group runs, infer missing pairs, or compare results with a manifest.

    Raises:
        OSError: If the file cannot be opened or read.
        ValueError: If the header order/fields or any result row is invalid."""

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
    """Estimate an A-minus-B condition effect across complete paired stream blocks.

    For each block, difference d is condition_a minus condition_b. With n complete
    blocks, this function reports mean(d), sample SD(d), SE=SD/sqrt(n), and the
    two-sided 95% interval mean(d) +/- t[0.975, n-1]*SE. Ordinary p-values come from
    SciPy's paired t-test. Tasks/epochs are not replicates; callers must provide
    scientifically independent stream blocks and appropriate paired-test assumptions.

    Args:
        rows (Sequence[Mapping[str, object]]): Canonical result records. All rows
            are individually validated before selecting the requested metric and
            two conditions. Selected rows must share one manifest/phase and contain
            exactly one result per condition per block, with at least two blocks.
        condition_a (str): First condition in the signed contrast. Must differ from
            condition_b and match the preregistered A condition in confirmation.
        condition_b (str): Second condition subtracted from A. In confirmation its
            name and orientation must match the frozen contrast.
        metric (str): Exact metric name used to select rows. Confirmation must use
            the preregistered primary metric.
        manifest (Mapping[str, object] | None): Design against which selected run
            IDs, block IDs, conditions, phase, and hash are checked. Default None
            allows development inference from internally consistent rows alone.
            Confirmation results require the frozen manifest and external hash.
        expected_hash (str | None): Trusted digest for a supplied manifest. Default
            None is allowed in development but not confirmation. A hash without a
            manifest is invalid.

    Returns:
        dict[str, object]: Identity fields, ordered ``block_ids`` and ``pair_count``;
        ``condition_a_mean``/``condition_b_mean``; aligned ``paired_differences``;
        ``mean_paired_difference``, ``sample_sd_paired_difference``, ``standard_error``,
        ``degrees_of_freedom``, ``t_statistic``, ``t_critical_95``, ``ci_95_lower``,
        ``ci_95_upper``, and ``paired_t_p_value``; plus ``tasks_used_as_replicates=False``.
        A constant zero difference gives t=0 and unavailable p=None. A constant
        nonzero difference gives signed infinite t and p=0. Both cases have zero
        interval width. A non-finite SciPy p-value is represented as None. The
        function does not correct for multiple contrasts or mutate input rows.

    Raises:
        ValueError: If rows/design/contrast are invalid, no rows match, hashes or
            phases differ, pair/run coverage is duplicated/incomplete, fewer than
            two pairs remain, or confirmation requirements are not met.
        TypeError: If supplied manifest data cannot be canonically serialized."""

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
    # Retain rows for the requested metric and the two compared conditions.
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
        # Count planned runs only for the conditions in this paired comparison.
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
        # Treat an undefined paired t-test p-value as unavailable.
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
