"""Reserve independent new-run result directories with atomic collision handling.

The returned directory belongs to one new execution. Its owner shares that path
with all artifact writers. Task-checkpoint resume has a separate authenticated
destination contract and must not use this allocator to reinterpret old output.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import uuid


def reserve_result_directory(
    base_path: str | os.PathLike[str],
    project_tag: str | None = None,
    *,
    timestamp: datetime | None = None,
) -> Path:
    """Atomically claim a new timestamp/tag directory beneath an explicit root.

    The first candidate retains the historical readable name. Occupied files,
    directories and symlinks cause a unique-suffix retry, never reuse. Exclusive
    mkdir provides the ownership guarantee across threads and processes; UUIDs
    reduce contention but are not the guarantee. Existing run contents are untouched.

    Args:
        base_path: Parent directory for new outputs; created if needed.
        project_tag: Optional portable filename fragment.
        timestamp: Optional fixed clock value for deterministic allocation tests.

    Returns:
        Resolved, exclusively created output directory.

    Raises:
        ValueError: The tag is not a portable filename fragment.
        FileExistsError: Every bounded candidate collided.
        OSError: The filesystem cannot create the requested directories.
    """
    # Reject ambiguous nontext tags before creating any output directory.
    if project_tag is not None and not isinstance(project_tag, str):
        raise ValueError("project_tag must be a portable filename fragment.")
    tag = "" if project_tag is None else project_tag.strip()
    # A tag is a suffix, never a path or a platform-specific filename escape.
    if any(ord(char) < 32 or char in '/\\<>:"|?*' for char in tag) or tag.endswith('.'):
        raise ValueError("project_tag must be a portable filename fragment.")
    root = Path(base_path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    name = (timestamp or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    name += " " + tag if tag else ""
    for attempt in range(128):
        suffix = "" if attempt == 0 else "-" + uuid.uuid4().hex
        candidate = root / (name + suffix)
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError("Unable to reserve a unique result directory after 128 attempts.")
