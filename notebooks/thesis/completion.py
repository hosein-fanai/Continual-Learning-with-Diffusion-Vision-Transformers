"""Validate and recover completion bookkeeping using saved native evidence only.

No model or training API is imported. Recovery never edits a completion artifact,
native result, planned YAML or frozen hash. A started marker alone is not recovery.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator, TYPE_CHECKING

# Import annotation-only types without changing the runtime backend.
if TYPE_CHECKING:
    from semantic_consolidation.config import RouteConfig
import tempfile

import numpy as np
import yaml

from common.study_artifacts import (
    _unique_keys, read_completed_runs, replace_completed_index,
    validate_completed_artifact,
)
from common.config import _safe_load_unique_yaml, load_config
from common.experiment import materialize_run_plan
from semantic_consolidation.config import load_route_config, primary_accuracy_matrix_name
from semantic_consolidation.study import (
    _completed_metrics, _read_study_manifest, planned_config, validate_planned_config,
)


def _reject_nonfinite_json(value: object) -> None:
    """Never returns; strict completion JSON cannot contain NaN or Infinity.

    Args:
        value (object): JSON-like object or scalar. Missing and nonfinite values are handled as
            described in the return contract.

    Returns:
        rejected (None): Never returns; strict completion JSON cannot contain NaN or Infinity.

    Raises:
        ValueError: Always, identifying the rejected nonfinite JSON token.
    """
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _json(path: str | Path) -> object:
    """Read the saved artifact.

    Args:
        path (str | Path): File to read or write; relative paths use the current working
            directory.

    Returns:
        saved (object): Strict decoded JSON; duplicate keys and nonfinite constants are
            rejected.

    Raises:
        ValueError: If JSON contains duplicate keys, malformed content or nonfinite constants.
        OSError: If the source cannot be read.
    """
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_unique_keys,
                         parse_constant=_reject_nonfinite_json)


def _plain(value: object) -> object:
    """Normalize safe YAML sequences and integer map keys without dropping fields.

    Args:
        value (object): JSON-like object or scalar. Missing and nonfinite values are handled as
            described in the return contract.

    Returns:
        normalized (object): JSON-compatible copy with tuple sequences converted to lists and
            integer mapping keys represented as strings; no scientific fields are removed.

    Raises:
        TypeError: If a value cannot be serialized.
        ValueError: If serialization encounters nonfinite values.
    """
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _csv(path: str | Path) -> list[dict[str, str]]:
    """Read the saved artifact.

    Args:
        path (str | Path): File to read or write; relative paths use the current working
            directory.

    Returns:
        rows (list[dict[str, str]]): CSV rows with string field values in saved order.

    Raises:
        OSError: If the CSV cannot be read.
    """
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _publish_new_json(path: str | Path, value: object) -> None:
    """Flush a complete file, then publish atomically without replacing evidence.

    Args:
        path (str | Path): File to read or write; relative paths use the current working
            directory.
        value (object): JSON-like object or scalar. Missing and nonfinite values are handled as
            described in the return contract.

    Returns:
        published (None): None; fsyncs a complete temporary JSON file and atomically links it to
            a previously absent destination.

    Raises:
        FileExistsError: If the destination exists.
        ValueError: If JSON contains nonfinite data.
        OSError: If writing, linking or cleanup fails.
    """
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".pending", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        # Unlike replace(), link() fails if the destination already exists.
        os.link(temporary, path)
    finally:
        # Use existing evidence only when the corresponding artifact is present.
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _artifact_descriptor(path: str | Path) -> dict:
    """Relative artifact basename and SHA-256 of its exact bytes.

    Args:
        path (str | Path): File to read or write; relative paths use the current working
            directory.

    Returns:
        descriptor (dict): Relative artifact basename and SHA-256 of its exact bytes.

    Raises:
        OSError: If the artifact cannot be read.
    """
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@contextmanager
def _completion_lock(manifest_path: Path) -> Iterator[None]:
    """Keep completion read/validate/publish atomic across local notebook kernels.

    Args:
        manifest_path (Path): Native paired manifest identifying the study and adjacent run
            artifacts.

    Returns:
        lock (Iterator[None]): Context manager yielding None while an OS file lock serializes
            local completion validation and publication.

    Raises:
        OSError: If the lock file cannot be opened, acquired or released.
    """
    with (Path(manifest_path).parent / ".completed_runs.lock").open("a+b", buffering=0) as stream:
        # Use the native lock behavior for this operating system.
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            # Windows permits locking a byte beyond EOF. Acquire it before
            # initializing an empty file: another handle may have opened that
            # same empty file before the first publisher obtained its lock.
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        # Handle the complementary supported case without inventing observations.
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            # Use the native lock behavior for this operating system.
            if os.name == "nt" and os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"\0")
            yield
        finally:
            # Use the native lock behavior for this operating system.
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            # Handle the complementary supported case without inventing observations.
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _validate_config(result_path: Path, expected: RouteConfig, entry: dict, manifest_path: Path) -> None:
    """Compare scientific settings, permitting only documented runtime rewrites.

    common.train writes a pre-training common config, then adds inference weights, the final
    dynamic head width/label map and schedule-request metadata. Neither common snapshot is a
    route YAML. Dataset sizing and inferred cosine duration are runtime values; route
    coefficients are retained separately and in common.hpo.

    Args:
        result_path (Path): Saved native run directory containing immutable configurations,
            metrics and model artifacts.
        expected (RouteConfig): Exact planned RouteConfig before documented runtime artifact and
            inference rewrites.
        entry (dict): One materialized condition/seed entry with its exact schedule, run ID and
            manifest hash.
        manifest_path (Path): Native paired manifest identifying the study and adjacent run
            artifacts.

    Returns:
        validated (None): None; checks route settings and both native common snapshots, allowing
            only declared runtime locator, dataset-size and final-head/weight rewrites.

    Raises:
        ValueError: If scientific settings, stream locators or inference artifacts disagree.
        OSError: If required configurations or weights cannot be read.
    """
    route = _plain(asdict(expected.route))
    with (result_path / "route.settings.yaml").open(encoding="utf-8") as stream:
        saved_route = _safe_load_unique_yaml(stream)
    # Saved route.
    if _plain(saved_route) != route:
        raise ValueError("Saved route.settings.yaml differs from the planned intervention.")
    base = _plain(asdict(expected.common))
    base["hpo"]["semantic_consolidation"] = route
    snapshots = {}
    for filename in ("input_config.yaml", "config.yaml"):
        actual = _plain(asdict(load_config(result_path / filename)))
        snapshots[filename] = deepcopy(actual)
        wanted = deepcopy(base)
        identity = actual["continually_learn"]
        # This value belongs to a different manifest path.
        if Path(identity["experiment_manifest_path"]).resolve() != manifest_path:
            raise ValueError(f"{filename} belongs to a different manifest path.")
        identity["experiment_manifest_path"] = wanted["continually_learn"]["experiment_manifest_path"]
        # Only this stream's checkpoint root is relocatable; other streams cannot be adopted.
        checkpoint_root = manifest_path.parent / "checkpoints" / entry["run_id"]
        # Authenticate this stream's runtime checkpoint and resume locators.
        if identity.get("save_task_checkpoints"):
            # This value checkpoint root differs from this stream.
            if Path(identity["checkpoint_dir"]).resolve() != checkpoint_root.resolve():
                raise ValueError(f"{filename} checkpoint root differs from this stream.")
            resume = identity.get("resume_from")
            # This value resume locator belongs to another stream.
            if resume is not None and not Path(resume).resolve().is_relative_to(checkpoint_root.resolve()):
                raise ValueError(f"{filename} resume locator belongs to another stream.")
            for field in ("checkpoint_dir", "resume_from"):
                identity[field] = wanted["continually_learn"][field]
        # This value results_path does not identify these native results.
        if Path(actual["training"]["results_path"]).resolve() != result_path:
            raise ValueError(f"{filename} results_path does not identify these native results.")
        actual["training"]["results_path"] = wanted["training"]["results_path"]
        # Artifact tags are explicitly relocatable in the native plan validator.
        actual["training"]["project_tag"] = wanted["training"]["project_tag"]
        recorded_length = actual["dataset"]["trainset_len"]
        # This value has invalid runtime dataset sizing.
        if recorded_length is not None and (isinstance(recorded_length, bool)
                                            or not isinstance(recorded_length, int) or recorded_length <= 0):
            raise ValueError(f"{filename} has invalid runtime dataset sizing.")
        # Accept native dataset sizing only when the recipe left it unresolved.
        if wanted["dataset"]["trainset_len"] is None:
            wanted["dataset"]["trainset_len"] = recorded_length
        # Infer only an unspecified native cosine duration from the saved training size.
        if wanted["optimizer"]["schedule"] == "cosine" and wanted["optimizer"]["decay_steps"] is None:
            # Resolved cosine schedule requires saved dataset sizing.
            if recorded_length is None:
                raise ValueError("Resolved cosine schedule requires saved dataset sizing.")
            wanted["optimizer"]["decay_steps"] = wanted["training"]["epochs"] * recorded_length
        hpo = actual["hpo"]
        input_path = hpo.pop("input_config_path", None)
        # This value has a foreign/missing immutable input_config_path.
        if input_path is None or Path(input_path).resolve() != result_path / "input_config.yaml":
            raise ValueError(f"{filename} has a foreign/missing immutable input_config_path.")
        # Validate the documented inference-only rewrites in the final configuration.
        if filename == "config.yaml":
            schedule_request = hpo.pop("schedule_request", None)
            continual = wanted["continually_learn"]
            # Saved inference schedule request differs from the planned stream.
            if schedule_request != {"task_size": continual["task_size"],
                                    "class_order_mode": "fixed", "task_order_mode": "fixed",
                                    "seed": entry["stream"]["stream_seed"]}:
                raise ValueError("Saved inference schedule request differs from the planned stream.")
            # Require the requested inference artifacts in this saved run.
            if wanted["training"]["save_weights"]:
                for weights in (actual["model"].get("weights_path"), hpo.pop("classifier_weights_path", None)):
                    # Saved inference weights must refer to this result directory.
                    if not weights or Path(weights).resolve().parent != result_path:
                        raise ValueError("Saved inference weights must refer to this result directory.")
                    weight_path = Path(weights)
                    # Saved inference weights are missing: the selected artifact.
                    if not weight_path.is_file() and not (Path(str(weight_path) + ".index").is_file()
                                                          and list(weight_path.parent.glob(weight_path.name + ".data-*"))):
                        raise ValueError(f"Saved inference weights are missing: {weights}.")
                actual["model"]["weights_path"] = wanted["model"]["weights_path"]
            # The learner remaps original labels to schedule positions before the
            # dynamic DiT sees them. The wrapper saves that dense identity map.
            model = actual["model"]
            model_values = model["kwargs"] if model["name"] is not None and model["kwargs"] else model["dit_classifier"]
            desired_values = wanted["model"]["kwargs"] if model["name"] is not None and model["kwargs"] else wanted["model"]["dit_classifier"]
            # Validate and normalize the final grown classifier width.
            if model_values.get("num_classes") != desired_values.get("num_classes"):
                # Saved dynamic classifier width differs from the full schedule.
                if model_values.get("num_classes") != len(entry["stream"]["class_order"]):
                    raise ValueError("Saved dynamic classifier width differs from the full schedule.")
                # Restore the explicitly declared initial width for comparison.
                if "num_classes" in desired_values:
                    model_values["num_classes"] = desired_values["num_classes"]
                # Handle the complementary supported case without inventing observations.
                else:
                    model_values.pop("num_classes")
            wrapper = model["wrapper_kwargs"] if model["name"] is not None and model["wrapper_kwargs"] else model["diffusion_classifier"]
            desired_wrapper = wanted["model"]["wrapper_kwargs"] if model["name"] is not None and model["wrapper_kwargs"] else wanted["model"]["diffusion_classifier"]
            # Validate and normalize the final dense label map.
            if wrapper.get("seen_classes") != desired_wrapper.get("seen_classes"):
                label_map = {str(index): index for index in range(len(entry["stream"]["class_order"]))}
                # Saved inference label map differs from the planned class order.
                if wrapper.get("seen_classes") != label_map:
                    raise ValueError("Saved inference label map differs from the planned class order.")
                # Restore the explicitly declared initial label map for comparison.
                if "seen_classes" in desired_wrapper:
                    wrapper["seen_classes"] = desired_wrapper["seen_classes"]
                # Handle the complementary supported case without inventing observations.
                else:
                    wrapper.pop("seen_classes")
        # This value scientific settings differ from the frozen planned
        # configuration.
        if actual != wanted:
            raise ValueError(f"{filename} scientific settings differ from the frozen planned configuration.")
    for section, field in (("dataset", "trainset_len"), ("optimizer", "decay_steps")):
        # Input and inference configurations disagree about the selected artifact.
        if snapshots["input_config.yaml"][section][field] != snapshots["config.yaml"][section][field]:
            raise ValueError(f"Input and inference configurations disagree about {section}.{field}.")


def _validate_native_result(record: dict, entry: dict, manifest_path: Path, manifest: dict) -> None:
    """Validate the saved native result before accepting completion.

    Args:
        record (dict): Frozen design or completion record, as required by this operation;
            scientific fields are retained.
        entry (dict): One materialized condition/seed entry with its exact schedule, run ID and
            manifest hash.
        manifest_path (Path): Native paired manifest identifying the study and adjacent run
            artifacts.
        manifest (dict): Authenticated native study manifest with materialized condition and
            stream definitions.

    Returns:
        validated (None): None; verifies the complete selected test matrix, recomputed metrics,
            schedule, source and update counts against saved native evidence.

    Raises:
        ValueError: If completion scalars or any required scientific artifact disagree.
        OSError: If required native files cannot be read.
    """
    # Completion elapsed seconds must be a finite nonnegative measurement.
    if isinstance(record.get("seconds"), bool) or not isinstance(record.get("seconds"), (int, float)) \
            or not np.isfinite(record["seconds"]) or record["seconds"] < 0:
        raise ValueError("Completion elapsed seconds must be a finite nonnegative measurement.")
    # Completion optimizer work must be a nonnegative integer.
    if isinstance(record.get("total_updates"), bool) or not isinstance(record.get("total_updates"), int) \
            or record["total_updates"] < 0:
        raise ValueError("Completion optimizer work must be a nonnegative integer.")
    expected = planned_config(entry, manifest_path)
    config = load_route_config(manifest_path.parent / f"{entry['run_id']}.yaml")
    identity = config.common.continually_learn
    # Planned YAML identifies a different run or manifest.
    if identity.experiment_run_id != entry["run_id"] or identity.experiment_manifest_hash != manifest["manifest_hash"]:
        raise ValueError("Planned YAML identifies a different run or manifest.")
    # Copied frozen plans retain their original YAML bytes. Only the locator is
    # resolved locally; the declared identity and every scientific setting stay checked.
    identity.experiment_manifest_path = str(manifest_path)
    validate_planned_config(config)
    result_path = Path(record["results_path"]).resolve()
    _validate_config(result_path, expected, entry, manifest_path)
    provenance = _json(result_path / "source_provenance.json")
    declared = manifest["spec"]["analysis_spec"]["native_route_study"]["source"]["files"]
    expected_files = {name: digest for name, digest in declared.items() if name.endswith(".py")}
    # Native source provenance differs from the frozen executable source.
    if provenance.get("files") != expected_files or provenance.get("source_sha256") != hashlib.sha256(
            json.dumps(expected_files, sort_keys=True).encode()).hexdigest():
        raise ValueError("Native source provenance differs from the frozen executable source.")
    groups = entry["stream"]["task_groups"]
    size = len(groups)
    matrix = np.asarray(record["accuracy_matrix"], dtype="float64")
    metrics = _completed_metrics(matrix, size)
    matrix_name = primary_accuracy_matrix_name(expected)
    # Confirmation authenticates the predictor selected in the frozen configuration.
    if record.get("accuracy_matrix_source") != matrix_name:
        raise ValueError(f"Confirmation requires the configured {matrix_name} endpoint.")
    # Completed metrics differ from their full saved matrix.
    if not isinstance(record.get("metrics"), dict) or any(
            key not in record["metrics"] or isinstance(record["metrics"][key], bool)
            or not np.isclose(value, record["metrics"][key], rtol=0., atol=1e-12)
            for key, value in metrics.items()):
        raise ValueError("Completed metrics differ from their full saved matrix.")
    rows = [row for row in _csv(result_path / "accuracy_matrices.csv")
            if row["matrix"] == matrix_name]
    coordinates = set()
    for row in rows:
        i, j = int(row["after_task_index"]), int(row["evaluated_task_index"])
        # Native selected matrix has duplicate/out-of-schedule cells.
        if not 0 <= i < size or not 0 <= j < size or (i, j) in coordinates:
            raise ValueError("Native selected matrix has duplicate/out-of-schedule cells.")
        coordinates.add((i, j))
        # Native matrix class metadata differs from the frozen schedule.
        if json.loads(row["after_task_classes"]) != groups[i] or json.loads(row["evaluated_task_classes"]) != groups[j] \
                or json.loads(row["seen_classes"]) != sum(groups[:i + 1], []):
            raise ValueError("Native matrix class metadata differs from the frozen schedule.")
        value = float(row["value"]) if row["value"] else np.nan
        # Native selected matrix differs from the completion artifact.
        if not np.isclose(value, matrix[i, j], rtol=0., atol=1e-12, equal_nan=True):
            raise ValueError("Native selected matrix differs from the completion artifact.")
    # Native selected matrix is incomplete; expected every scheduled cell.
    if len(coordinates) != size * size:
        raise ValueError("Native selected matrix is incomplete; expected every scheduled cell.")
    schedule = _csv(result_path / "schedule.csv")
    # Native schedule does not cover the full stream.
    if len(schedule) != size:
        raise ValueError("Native schedule does not cover the full stream.")
    for i, row in enumerate(schedule):
        # Native schedule/seed differs from the frozen stream.
        if int(row["task_index"]) != i or json.loads(row["task_classes"]) != groups[i] \
                or json.loads(row["class_order"]) != entry["stream"]["class_order"] \
                or json.loads(row["seen_classes"]) != sum(groups[:i + 1], []) \
                or int(row["seed"]) != entry["stream"]["stream_seed"]:
            raise ValueError("Native schedule/seed differs from the frozen stream.")
    summary_rows = [row for row in _csv(result_path / "summary.csv") if row["source"] == "continual_metrics"]
    summary = {row["key"]: row["value"] for row in summary_rows}
    # Native summary metrics differ from the recomputed matrix metrics.
    if len(summary) != len(summary_rows) or any(key not in summary or not np.isclose(
            value, float(summary[key]), rtol=0., atol=1e-12) for key, value in metrics.items()):
        raise ValueError("Native summary metrics differ from the recomputed matrix metrics.")
    route_rows = _json(result_path / "route_metrics.json")
    # Native route records are incomplete or identify a different condition.
    if not isinstance(route_rows, list) or len(route_rows) != size or any(
            row.get("task") != i + 1 or row.get("condition") != expected.route.condition
            or isinstance(row.get("total_updates"), bool) or not isinstance(row.get("total_updates"), int)
            or row["total_updates"] < 0 for i, row in enumerate(route_rows)):
        raise ValueError("Native route records are incomplete or identify a different condition.")
    # Completion optimizer work differs from native route records.
    if sum(row["total_updates"] for row in route_rows) != record.get("total_updates"):
        raise ValueError("Completion optimizer work differs from native route records.")


def _validated_record(record: dict, entry: dict, manifest_path: Path, manifest: dict) -> None:
    """Completion record after artifact and native-result authentication, retaining its exact run
    identity.

    Args:
        record (dict): Frozen design or completion record, as required by this operation;
            scientific fields are retained.
        entry (dict): One materialized condition/seed entry with its exact schedule, run ID and
            manifest hash.
        manifest_path (Path): Native paired manifest identifying the study and adjacent run
            artifacts.
        manifest (dict): Authenticated native study manifest with materialized condition and
            stream definitions.

    Returns:
        validated (None): None; checks artifact and native-result authenticity without modifying
            the completion record.

    Raises:
        ValueError: If the artifact hash or native evidence is invalid.
        OSError: If an artifact cannot be read.
    """
    # Completion has a foreign run, condition, or manifest identity.
    if not isinstance(record, dict) or record.get("run_id") != entry["run_id"] \
            or record.get("condition") != entry["condition"] or record.get("manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("Completion has a foreign run, condition, or manifest identity.")
    validate_completed_artifact(manifest_path.parent, record, required=True)
    _validate_native_result(record, entry, manifest_path, manifest)


def reconcile_completions(manifest_path: Path, *, expected_hash: str) -> dict:
    """Validate and reconcile completed artifacts under one process-safe lock.

    Args:
        manifest_path (Path): Native paired manifest identifying the study and adjacent run
            artifacts.
        expected_hash (str): Externally retained SHA-256 identity of the study manifest.

    Returns:
        completed (dict): Authenticated completed run mapping; only missing index entries with
            complete valid native evidence are repaired under a lock.

    Raises:
        ValueError: If indexed or pending evidence conflicts with the frozen plan.
        OSError: If locking, reading or atomic index publication fails.
    """
    manifest_path = Path(manifest_path).resolve()
    with _completion_lock(manifest_path):
        return _reconcile_completions(manifest_path, expected_hash=expected_hash)


def _reconcile_completions(manifest_path: Path, *, expected_hash: str) -> dict:
    """Verify saved completions and repair only validated missing index entries.

    All candidates are checked before any index publication. A separate recovery receipt states
    the attempted bookkeeping repair; the index establishes success. Repeated calls on matching
    indexed artifacts do not write any files.

    Args:
        manifest_path (Path): Native paired manifest identifying the study and adjacent run
            artifacts.
        expected_hash (str): Externally retained SHA-256 identity of the study manifest.

    Returns:
        completed (dict): Validated run mapping after idempotent reconciliation; caller must
            hold the completion lock.

    Raises:
        ValueError: If evidence conflicts or cannot prove completion.
        OSError: If an artifact or index cannot be read or published.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_study_manifest(manifest_path, expected_hash)
    planned = {entry["run_id"]: entry for entry in materialize_run_plan(manifest)}
    index_path = manifest_path.parent / "completed_runs.json"
    outputs = read_completed_runs(index_path) if index_path.exists() else {}
    # Completion index contains unplanned runs; preserve it and inspect the
    # conflict.
    if not set(outputs) <= set(planned):
        raise ValueError("Completion index contains unplanned runs; preserve it and inspect the conflict.")
    recovered = []
    for path in sorted(manifest_path.parent.glob("*.completed.json")):
        run_id = path.name[:-len(".completed.json")]
        # Unplanned completion artifact: the selected artifact; preserve and inspect it.
        if run_id not in planned:
            raise ValueError(f"Unplanned completion artifact: {path}; preserve and inspect it.")
        # Respect the selected stream identity and its saved completion state.
        if run_id not in outputs:
            try:
                record = _json(path)
            except (OSError, ValueError, TypeError) as error:
                raise ValueError(f"Malformed completion artifact {path}: {error} Preserve the file and inspect the interrupted publication; no training was restarted.") from error
            # Malformed standalone completion artifact: the selected artifact.
            if not isinstance(record, dict) or "completed_artifact" in record:
                raise ValueError(f"Malformed standalone completion artifact: {path}.")
            outputs[run_id] = {**record, "completed_artifact": _artifact_descriptor(path)}
            recovered.append(run_id)
    for run_id, record in outputs.items():
        try:
            _validated_record(record, planned[run_id], manifest_path, manifest)
        except (OSError, ValueError, TypeError, KeyError, IndexError, yaml.YAMLError) as error:
            raise ValueError(f"Cannot accept completion {run_id}: {error} Preserve artifacts; inspect this run's native results and frozen configuration. No training was restarted.") from error
    for run_id in recovered:
        receipt_path = manifest_path.parent / f"{run_id}.recovered.json"
        receipt = {"schema_version": 1, "run_id": run_id, "manifest_hash": manifest["manifest_hash"],
                   "completed_artifact": outputs[run_id]["completed_artifact"],
                   "action": "validated missing-index completion; atomic index reconstruction requested",
                   "training_called": False}
        # Use existing evidence only when the corresponding artifact is present.
        if receipt_path.exists():
            previous = _json(receipt_path)
            # Conflicting recovery receipt: the selected artifact; preserve and inspect it.
            if {key: previous.get(key) for key in receipt} != receipt:
                raise ValueError(f"Conflicting recovery receipt: {receipt_path}; preserve and inspect it.")
        # Handle the complementary supported case without inventing observations.
        else:
            _publish_new_json(receipt_path, {**receipt, "recorded_utc": datetime.now(timezone.utc).isoformat()})
    # Publish the index only when valid missing entries were recovered.
    if recovered:
        replace_completed_index(index_path, outputs)
    return outputs


def publish_completion(manifest_path: Path, record: dict, *, expected_hash: str) -> dict:
    """Publish native completion evidence and its index in one locked transaction.

    Args:
        manifest_path (Path): Native paired manifest identifying the study and adjacent run
            artifacts.
        record (dict): Frozen design or completion record, as required by this operation;
            scientific fields are retained.
        expected_hash (str): Externally retained SHA-256 identity of the study manifest.

    Returns:
        completed (dict): Published completion record/index result using an atomic lock;
            matching retries preserve original evidence.

    Raises:
        ValueError: If the candidate conflicts with existing completion or native evidence.
        OSError: If locking or publication fails.
    """
    manifest_path = Path(manifest_path).resolve()
    with _completion_lock(manifest_path):
        return _publish_completion(manifest_path, record, expected_hash=expected_hash)


def _publish_completion(manifest_path: Path, record: dict, *, expected_hash: str) -> dict:
    """Publish a validated completion artifact, then atomically update the index.

    An interruption between the two publications is repaired by reconciliation. Submitting the
    same saved completion again is a no-op, never a training resume.

    Args:
        manifest_path (Path): Native paired manifest identifying the study and adjacent run
            artifacts.
        record (dict): Frozen design or completion record, as required by this operation;
            scientific fields are retained.
        expected_hash (str): Externally retained SHA-256 identity of the study manifest.

    Returns:
        completed (dict): Published completion record/index result; caller must hold the
            completion lock.

    Raises:
        ValueError: If the candidate does not authenticate or conflicts with completed evidence.
        OSError: If a record or index cannot be published.
    """
    manifest_path = Path(manifest_path).resolve()
    outputs = _reconcile_completions(manifest_path, expected_hash=expected_hash)
    run_id = record.get("run_id")
    # Respect the selected stream identity and its saved completion state.
    if run_id in outputs:
        # Conflicting completion submission for the selected artifact; existing evidence was
        # preserved.
        if {key: value for key, value in outputs[run_id].items() if key != "completed_artifact"} != record:
            raise ValueError(f"Conflicting completion submission for {run_id}; existing evidence was preserved.")
        return outputs[run_id]
    manifest = _read_study_manifest(manifest_path, expected_hash)
    planned = {entry["run_id"]: entry for entry in materialize_run_plan(manifest)}
    # Completion publication does not match the frozen planned identity.
    if run_id not in planned or record.get("manifest_hash") != manifest["manifest_hash"] \
            or record.get("condition") != planned[run_id]["condition"] or "completed_artifact" in record:
        raise ValueError("Completion publication does not match the frozen planned identity.")
    _validate_native_result(record, planned[run_id], manifest_path, manifest)
    path = manifest_path.parent / f"{run_id}.completed.json"
    _publish_new_json(path, record)
    completed = {**record, "completed_artifact": _artifact_descriptor(path)}
    outputs[run_id] = completed
    replace_completed_index(manifest_path.parent / "completed_runs.json", outputs)
    return completed
