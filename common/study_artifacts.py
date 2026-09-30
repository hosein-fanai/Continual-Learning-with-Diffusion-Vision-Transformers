"""Source and completed-run integrity shared by the ordinary research studies.

This module has no dependency on a route runner or study adapter. It can be
imported by either route and by the allocation orchestrators without cycles.
Content hashes detect changed artifacts; independently retaining the design
digest and source snapshot remains necessary for confirmation provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SOURCE_PACKAGES = ("common", "autoencoder", "diffusion", "semantic_consolidation")
OPTIONAL_SOURCE_PACKAGES = ("allocation_study", "gist_memory")
SOURCE_ROOT = Path(__file__).resolve().parents[1]


def source_files(root: str | Path = SOURCE_ROOT, additional_packages: tuple[str, ...] = ()) -> dict[str, str]:
    """Hash maintained source, with explicitly requested optional local routes.

    The default scope exists in a fresh GitHub checkout. Optional research
    packages are included only when requested; their generated snapshots never
    become executable source. Missing requested packages fail explicitly.

    Args:
        root (str | Path): Checkout root containing all required production packages.
            Defaults to ``SOURCE_ROOT``.
        additional_packages (tuple[str, ...]): Registered optional package names
            explicitly included in the identity; duplicate names are counted once.
            Defaults to ``()``.

    Returns:
        dict[str, str]: Relative POSIX Python paths mapped to SHA-256 byte digests.
        Files under tests, prepared/results, cache, or audit directories are excluded.

    Raises:
        ValueError: A requested package is unknown or a required directory is absent.
        OSError: Source enumeration or reading fails.
    """

    root = Path(root).resolve()
    # Only registered optional routes may extend the shared source identity.
    if set(additional_packages) - set(OPTIONAL_SOURCE_PACKAGES):
        raise ValueError("Unknown optional source package.")
    files = {}
    excluded = {"tests", "__pycache__", "prepared", "results", ".tmp", ".audit"}
    for package in (*SOURCE_PACKAGES, *dict.fromkeys(additional_packages)):
        directory = root / package
        # Missing source cannot be silently excluded from a frozen identity.
        if not directory.is_dir():
            raise ValueError(f"Missing required source package: {package}.")
        for parent, directories, names in os.walk(directory):
            directories[:] = sorted(name for name in directories if name not in excluded)
            for name in sorted(names):
                path = Path(parent) / name
                # Executable Python defines the package's source contribution.
                if path.suffix == ".py":
                    files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def source_fingerprint(root: str | Path = SOURCE_ROOT, additional_packages: tuple[str, ...] = ()) -> dict:
    """Bind repository-relative production Python and declared dependencies.

    Tests, results, documentation and bytecode do not define executable training
    identity. Untracked production files are included, so a Git commit alone
    cannot silently omit the actual implementation used by a study.

    Args:
        root (str | Path): Checkout root; production files are read without importing.
            Defaults to ``SOURCE_ROOT``.
        additional_packages (tuple[str, ...]): Optional packages passed to source_files.
            Defaults to ``()``.

    Returns:
        dict: sha256 for the canonical sorted file-digest JSON and its files mapping.
        requirements.txt is included when present; no dependency versions are inferred.

    Raises:
        ValueError: Requested production packages are invalid or missing.
        OSError: A source/dependency file cannot be read.
    """

    root = Path(root).resolve()
    files = source_files(root, additional_packages=additional_packages)
    # Bind the shared dependency declaration used by local and hosted notebooks.
    requirements = root / "requirements.txt"
    # Bind the dependency declaration when the checkout provides one.
    if requirements.exists():
        files[requirements.name] = hashlib.sha256(requirements.read_bytes()).hexdigest()
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


def native_study_metadata(route: str) -> dict:
    """Declare the implemented native protocol before any data are inspected.

    Args:
        route (str): semantic_consolidation or gist_memory; the latter requires its
            optional source package in this checkout.

    Returns:
        dict: Schema version 1, selected route, executable source identity and the
        declared completed-matrix/artifact evidence requirement.

    Raises:
        ValueError: The route is unsupported or its required source is missing.
    """

    # Native studies name exactly one of the two supported cognitive adapters.
    if route not in ("semantic_consolidation", "gist_memory"):
        raise ValueError("A native study must identify its semantic or gist route.")
    additional = tuple(["gist_memory"]) if route == "gist_memory" else ()
    return {"schema_version": 1, "route": route, "source": source_fingerprint(additional_packages=additional), 
            "confirmation_evidence": "complete matrices and hashed per-run artifacts"}


def validate_study_source(manifest: Mapping, route: str) -> dict:
    """Check the executable source bound by native, allocation, or Section 11 designs.

    Legacy development manifests remain readable, with an explicit unverified
    status. Frozen confirmation and benchmark designs require the declared
    implementation contract; attaching a current hash to old outcomes is invalid.

    Args:
        manifest (Mapping): Study declaration with phase and spec.analysis_spec;
            the stored source identity must match the current production tree.
        route (str): Expected native route used to check a native declaration.

    Returns:
        dict: verified=True and current digest, or verified=False with an explicit
        reason only for a legacy development manifest lacking source provenance.

    Raises:
        ValueError: Native schema/route, frozen source requirement or content differs.
        OSError: A declared production file cannot be read.
    """

    analysis = manifest["spec"]["analysis_spec"]
    native = analysis.get("native_route_study")
    # A malformed native declaration cannot fall through to legacy compatibility.
    if native is not None and (not isinstance(native, Mapping) or native.get("schema_version") != 1 or native.get("route") != route):
        raise ValueError("Native study schema or route differs from this adapter.")
    declaration = native or analysis.get("allocation_study") or analysis.get("section11")
    # Historical exploratory outcomes cannot acquire retrospective source provenance.
    if not isinstance(declaration, Mapping) or "source" not in declaration:
        # Both frozen study modes require source identity bound before execution.
        if manifest["phase"] in ("confirmation", "benchmark"):
            raise ValueError("Frozen study has no bound executable source; prepare a new source-bound design.")
        return {"verified": False, "reason": "legacy development manifest without executable source identity"}
    declared_files = declaration["source"].get("files", {})
    additional = tuple(package for package in OPTIONAL_SOURCE_PACKAGES
                       if any(name.startswith(package + "/") for name in declared_files))
    current = source_fingerprint(additional_packages=additional)
    # Scientific execution and analysis must use the exact prepared implementation.
    if current != declaration["source"]:
        raise ValueError("Source fingerprint differs from the frozen study; use its retained source snapshot or prepare a new design.")
    return {"verified": True, "sha256": current["sha256"]}


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict:
    """Reject duplicate result identities and nested fields instead of overwriting them.

    Args:
        pairs (list[tuple[str, Any]]): Ordered key/value pairs from one JSON object,
            including nested objects when installed as object_pairs_hook.

    Returns:
        dict: Decoded object retaining each distinct key once.

    Raises:
        ValueError: A duplicate key could hide a conflicting endpoint or run identity.
    """

    result = {}
    for key, value in pairs:
        # Repeated JSON keys can hide a conflicting run or endpoint value.
        if key in result:
            raise ValueError(f"Duplicate JSON key/run identity: {key}")
        result[key] = value
    return result


def read_completed_runs(path: str | Path) -> dict:
    """Read a unique-key result mapping before accepting any stream outcome.

    Args:
        path (str | Path): UTF-8 JSON index mapping run IDs to result dictionaries.

    Returns:
        dict: Parsed records in file order; nested duplicate keys are rejected.

    Raises:
        ValueError: JSON is malformed, keys repeat, or the index/records have wrong types.
        OSError: The index cannot be read.
    """

    with Path(path).open(encoding="utf-8") as stream:
        result = json.load(stream, object_pairs_hook=_unique_keys)
    # A result index is a run-ID mapping, never a sequence or scalar.
    if not isinstance(result, dict) or any(not isinstance(value, dict) for value in result.values()):
        raise ValueError("Completed runs must be a mapping from run identities to records.")
    return result


def write_completed_artifact(directory: str | Path, record: dict) -> dict:
    """Save one exclusive per-run outcome and return its relative path and digest.

    Args:
        directory (str | Path): Existing study directory for an exclusive outcome file.
        record (dict): JSON-safe finite outcome with run_id; this initial payload
            should not yet contain the completed_artifact descriptor.

    Returns:
        dict[str, str]: Relative path and SHA-256 content digest. The input is
        unchanged. Serialization is validated before reserving the file so a failed
        encoding leaves no partial completion and a corrected run can retry.

    Raises:
        ValueError: The run ID escapes its directory or JSON contains nonfinite values.
        TypeError: The outcome contains a non-JSON object.
        FileExistsError: This run already has an outcome; existing evidence is preserved.
        OSError: The exclusive outcome file cannot be written/read.
    """

    directory = Path(directory).resolve()
    path = (directory / f"{record['run_id']}.completed.json").resolve()
    # A run identity must not redirect its completion artifact outside the study.
    if path.parent != directory:
        raise ValueError("Completed artifacts must remain inside their study directory.")
    # Validate the complete payload before reserving the exclusive outcome path.
    # Serialization failures must not leave a truncated "completed" file or
    # prevent retrying this run with corrected, finite endpoint measurements.
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def replace_completed_index(path: str | Path, records: Mapping) -> None:
    """Atomically publish progress without truncating previously completed streams.

    Args:
        path (str | Path): Destination index in an existing parent directory.
        records (Mapping): JSON-compatible finite run records to publish together.

    Returns:
        None: Serializes and fsyncs an adjacent temporary file, then atomically
        replaces the index. Failure removes the temporary file and preserves an
        existing destination until the successful replace.

    Raises:
        ValueError: A nonfinite JSON number cannot be serialized.
        TypeError: A record is not JSON compatible.
        OSError: Writing, syncing, replacement or cleanup fails.
    """

    path = Path(path).resolve()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, 
                                         prefix=f".{path.name}.", suffix=".pending", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(records, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        # Failed serialization or replacement must leave the previous index intact.
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_completed_artifact(directory: str | Path, record: dict, required: bool) -> bool:
    """Recheck saved run contents separately from duplicated index summaries.

    Args:
        directory (str | Path): Study directory containing the exclusive outcome file.
        record (dict): Indexed run outcome, optionally with a completed_artifact
            mapping containing exactly path and sha256.
        required (bool): True rejects an absent descriptor; False identifies such
            legacy evidence as unverified without inventing an artifact.

    Returns:
        bool: True after path, digest and decoded outcome match the index excluding
        its descriptor; False only for an allowed missing legacy descriptor.

    Raises:
        ValueError: Descriptor identity, file contents, JSON keys or endpoints disagree.
        OSError: A referenced artifact cannot be read.
    """

    descriptor = record.get("completed_artifact")
    # Legacy exploratory indexes remain identifiable without invented audit files.
    if descriptor is None:
        # Confirmation requires independent artifact contents for every completed run.
        if required:
            raise ValueError("Confirmation requires a hashed completed artifact for every run.")
        return False
    directory = Path(directory).resolve()
    # An artifact reference consists of a relative filename and its content digest.
    if not isinstance(descriptor, Mapping) or set(descriptor) != {"path", "sha256"}:
        raise ValueError("Malformed completed artifact reference.")
    path = (directory / descriptor["path"]).resolve()
    # Foreign result locations cannot supply an in-study completion record.
    if path.parent != directory or path.name != f"{record['run_id']}.completed.json":
        raise ValueError("Completed artifact path differs from its planned run identity.")
    # Check the independently saved bytes before trusting the duplicated index values.
    if hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]:
        raise ValueError("Completed artifact hash differs from its recorded digest.")
    with path.open(encoding="utf-8") as stream:
        saved = json.load(stream, object_pairs_hook=_unique_keys)
    indexed = {key: value for key, value in record.items() if key != "completed_artifact"}
    # Changing only an index endpoint must not silently alter paired inference.
    if saved != indexed:
        raise ValueError("Completed index differs from its independently saved run artifact.")
    return True
