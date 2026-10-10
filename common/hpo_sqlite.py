"""Opt-in node-local SQLite with immutable snapshots on shared study storage.

Call enable_local_sqlite and snapshot while holding the study's coordinator
lock. Ordinary studies keep their established study.db path. Local studies
commit snapshots at caller-selected lifecycle boundaries; a lost container can
lose writes after the last committed snapshot. A process restart on the same
container retains those writes in its local database.
"""

from __future__ import annotations

from contextlib import closing, contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import tempfile
import time
from typing import Any, Iterator
import uuid


_MARKER = "sqlite_local.json"
_IDENTITY = "study_identity.json"
_SNAPSHOTS = "sqlite_snapshots"


def _read_json(path: Path) -> dict[str, Any]:
    """Bound retries for a briefly incomplete shared-filesystem JSON read."""

    for attempt in range(4):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("SQLite storage metadata must be a JSON object.")
            return value
        except (OSError, json.JSONDecodeError):
            if attempt == 3:
                raise
            time.sleep(0.05)
    raise RuntimeError("SQLite storage metadata could not be read.")


def _sha256(path: Path) -> str:
    """Hash a file without materializing the database in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_copy(source: Path, target: Path) -> None:
    """Publish a closed file using a unique temporary sibling and replacement."""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        if _sha256(temporary) != _sha256(source):
            raise OSError("SQLite snapshot copy checksum mismatch.")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(target: Path, value: dict[str, Any]) -> None:
    """Commit metadata only after its referenced snapshot exists."""

    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_database(path: Path) -> None:
    """Require a standalone SQLite image with a successful integrity check."""

    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as database:
        result = database.execute("PRAGMA integrity_check").fetchall()
        foreign_keys = database.execute("PRAGMA foreign_key_check").fetchall()
    if result != [tuple(["ok"])]:
        raise ValueError("SQLite integrity check failed for " + str(path))
    if foreign_keys:
        raise ValueError("SQLite foreign key check failed for " + str(path))


def _cache_path(root: Path, local_root: Path) -> Path:
    """Give each absolute study root a distinct node-local directory."""

    identity = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
    return local_root / identity / "study.db"


def _marker(root: Path) -> dict[str, Any] | None:
    """Validate the durable path and snapshot identities without opening SQLite."""

    path = root / _MARKER
    if not path.exists():
        return None
    value = _read_json(path)
    if value.get("version") != 1 or value.get("study_root") != str(root):
        raise ValueError("Local SQLite marker belongs to another study or version.")
    local_root = Path(value.get("local_root", "")).resolve()
    database = _cache_path(root, local_root)
    if value.get("database_path") != str(database):
        raise ValueError("Local SQLite database path does not match its study identity.")
    digest = value.get("snapshot_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Local SQLite snapshot checksum is invalid.")
    if value.get("snapshot_path") != (_SNAPSHOTS + "/" + digest + ".db"):
        raise ValueError("Local SQLite snapshot path does not match its checksum.")
    if isinstance(value.get("generation"), bool) or not isinstance(value.get("generation"), int) or value["generation"] < 1:
        raise ValueError("Local SQLite snapshot generation is invalid.")
    if not isinstance(value.get("hostname"), str) or not value["hostname"] or not value.get("cache_id"):
        raise ValueError("Local SQLite cache ownership is missing.")
    return value


def _identity(marker: dict[str, Any]) -> dict[str, Any]:
    """Select the immutable node-cache ownership fields."""

    return {key: marker[key] for key in ("version", "study_root", "hostname", "database_path", "cache_id")}


def _require_cache(marker: dict[str, Any]) -> Path:
    """Reject an absent, foreign or replaced local cache without restoring it."""

    if marker["hostname"] != socket.gethostname():
        raise RuntimeError("Local SQLite belongs to another container; explicitly enable it here to restore its snapshot.")
    database = Path(marker["database_path"])
    if not database.is_file():
        raise RuntimeError("Local SQLite cache is missing; explicitly enable it to restore the committed snapshot.")
    identity_path = database.parent / _IDENTITY
    if not identity_path.is_file() or _read_json(identity_path) != _identity(marker):
        raise RuntimeError("Local SQLite cache identity differs from its study marker.")
    return database


def database_path(study_root: str | Path) -> Path:
    """Resolve the authoritative database; never silently restore missing state."""

    root = Path(study_root).resolve()
    marker = _marker(root)
    return root / "study.db" if marker is None else _require_cache(marker)


def _local_snapshot(database: Path) -> Path:
    """Use SQLite backup only between local files, retaining transactional state."""

    descriptor, temporary_name = tempfile.mkstemp(prefix="snapshot-", suffix=".db", dir=database.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    started = time.monotonic()

    def progress(status: int, remaining: int, total: int) -> None:
        """Bound SQLite busy retries instead of hanging a coordinator forever."""

        if status in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED} and time.monotonic() - started >= 30.0:
            raise TimeoutError("Node-local SQLite snapshot remained busy for 30 seconds.")

    try:
        # A crashed local writer may leave a hot rollback journal. Opening the
        # existing file read-write lets SQLite recover that committed state.
        with closing(sqlite3.connect(database.as_uri() + "?mode=rw", uri=True)) as source:
            with closing(sqlite3.connect(temporary)) as destination:
                source.backup(destination, pages=-1, progress=progress, sleep=0.1)
        _validate_database(temporary)
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _publish(root: Path, marker: dict[str, Any]) -> dict[str, Any]:
    """Commit an immutable snapshot, then refresh the compatibility database."""

    database = _require_cache(marker)
    temporary = _local_snapshot(database)
    try:
        digest = _sha256(temporary)
        relative = _SNAPSHOTS + "/" + digest + ".db"
        durable = root / relative
        if durable.exists():
            if _sha256(durable) != digest:
                raise ValueError("An immutable SQLite snapshot has changed.")
        else:
            _atomic_copy(temporary, durable)
        published = {
            **marker, "snapshot_sha256": digest, "snapshot_path": relative, 
            "generation": marker.get("generation", 0) + 1, "published_at_unix": time.time()
        }
        # The marker is the commit point. A failed compatibility copy cannot
        # invalidate the snapshot it already references.
        _atomic_json(root / _MARKER, published)
        for suffix in ("-journal", "-wal", "-shm"):
            sidecar = root / ("study.db" + suffix)
            if sidecar.exists() and sidecar.stat().st_size:
                raise RuntimeError("Preserve shared SQLite sidecar before refreshing the compatibility database: " + str(sidecar))
        _atomic_copy(durable, root / "study.db")
        return published
    finally:
        temporary.unlink(missing_ok=True)


def snapshot(study_root: str | Path) -> dict[str, Any] | None:
    """Persist node-local state under the caller's existing coordinator lock.

    Ordinary studies are unchanged. Publication errors propagate while leaving
    the authoritative local database intact. Callers must stop rather than
    continue claiming durable trial progress after an error.
    """

    root = Path(study_root).resolve()
    marker = _marker(root)
    return None if marker is None else _publish(root, marker)


@contextmanager
def sqlite_snapshots(study_root: str | Path) -> Iterator[None]:
    """Snapshot on exit while retaining an original operation failure."""

    try:
        yield
    except BaseException as error:
        try:
            snapshot(study_root)
        except BaseException as snapshot_error:
            error.add_note("SQLite snapshot also failed; local state was retained: " + str(snapshot_error))
        raise
    else:
        snapshot(study_root)


def enable_local_sqlite(
    study_root: str | Path, 
    local_root: str | Path, 
    source_database: str | Path | None = None
) -> dict[str, Any]:
    """Enable local transactions or explicitly restore a committed snapshot.

    Hold the study and notebook coordinator locks, verify no writer is alive,
    and choose genuinely node-local storage before calling. On first enable,
    source_database may name an independently recovered, clean SQLite image;
    otherwise the existing study.db is used. Back up damaged source files before
    enabling. Archive any original journal/WAL before replacing the persistent
    compatibility copy; this function refuses those sidecars. On a new
    container, the committed immutable snapshot supplies
    recovery, even when an interrupted compatibility copy left study.db stale.

    Existing same-container caches retain every write. This API does not
    overwrite them, silently import a replacement source, or alter trial states.
    """

    root = Path(study_root).resolve()
    local = Path(local_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for suffix in ("-journal", "-wal"):
        sidecar = root / ("study.db" + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise RuntimeError("Preserve and recover SQLite sidecar before enabling: " + str(sidecar))
    previous = _marker(root)
    if previous is not None:
        if source_database is not None:
            raise ValueError("An enabled study must restore its committed snapshot, not a replacement source.")
        existing = Path(previous["database_path"])
        if previous["hostname"] == socket.gethostname() and existing.exists():
            _require_cache(previous)
            # Republish rather than roll back newer same-container transactions.
            return _publish(root, previous)
        source = root / previous["snapshot_path"]
        if not source.is_file() or _sha256(source) != previous["snapshot_sha256"]:
            raise ValueError("The committed SQLite snapshot is absent or its checksum differs.")
    else:
        source = root / "study.db" if source_database is None else Path(source_database).resolve()
    if not source.is_file():
        raise FileNotFoundError("No SQLite study exists to enable: " + str(source))
    for suffix in ("-journal", "-wal"):
        sidecar = Path(str(source) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise RuntimeError("Local SQLite initialization requires a clean standalone source: " + str(sidecar))
    database = _cache_path(root, local)
    if database.exists():
        raise RuntimeError("Unregistered local SQLite state exists; preserve and inspect it before enabling.")
    for suffix in ("-journal", "-wal", "-shm"):
        sidecar = Path(str(database) + suffix)
        if sidecar.exists():
            raise RuntimeError("Preserve and inspect residual local SQLite sidecar before restoring: " + str(sidecar))
    database.parent.mkdir(parents=True, exist_ok=True)
    # Copy only an explicitly supplied clean image or a committed snapshot.
    # Never execute SQLite transactions against the shared source filesystem.
    _atomic_copy(source, database)
    _validate_database(database)
    marker = {
        "version": 1, "study_root": str(root), "hostname": socket.gethostname(), 
        "local_root": str(local), "database_path": str(database), "cache_id": uuid.uuid4().hex, 
        "generation": 0 if previous is None else previous["generation"]
    }
    _atomic_json(database.parent / _IDENTITY, _identity(marker))
    return _publish(root, marker)
