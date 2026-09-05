"""Manage continual-learning replay storage, candidate sampling, and cache reuse.

``ReplayBuffer`` implements FIFO, global reservoir, and balanced per-class
reservoir insertion with a private RNG and recoverable state. Array helpers
prepare exact exposure counts and conditioning labels. Optional replay caches
store numeric candidate pools in authenticated NPZ files and publish them
atomically so repeated or concurrent runs cannot overwrite a different pool.
"""

from __future__ import annotations

import tensorflow as tf

import numpy as np

import json

import os

import random

import uuid

from pathlib import Path

from operator import index

from collections.abc import Iterable, Mapping, Sequence
from collections import deque
from typing import Literal

from common.recovery import _array_recovery_descriptor, fingerprint_state


ReplayStrategy = Literal[
    "fifo", 
    "reservoir", 
    "class_balanced"
]

_STRATEGY_ALIASES = {
    "fifo": "fifo", 
    "reservoir": "reservoir", 
    "class_balanced": "class_balanced", 
    "class-balanced": "class_balanced"
}


def _sample_exact_rows(
    x: np.ndarray, 
    y: np.ndarray, 
    count: int | None, 
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Select an exact number of aligned exposure rows with a caller-owned RNG.

    Requests no larger than the pool sample without replacement; larger requests
    sample with replacement. A None count returns all normalized input rows without
    consuming the RNG, and zero returns shape-preserving empty array slices.

    Args:
        x (np.ndarray): Candidate samples shaped ``[N, ...]``.
        y (np.ndarray): Corresponding sparse or one-hot labels with N leading rows.
        count (int | None): Required nonnegative exposure count, or None for all rows.
        rng (np.random.Generator): Local generator consumed only by actual sampling.

    Returns:
        tuple[np.ndarray, np.ndarray]: Aligned sample/label arrays preserving their
        non-sample shapes and dtypes, with ``count`` rows or all N rows for None.

    Raises:
        ValueError: If row counts differ, positive exposure is requested from an
            empty task, or NumPy rejects an invalid sampling count.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    # Reject misaligned sample and label arrays before selecting exposure rows.
    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of rows.")

    # Keep the complete exposure pool when no count was requested.
    if count is None:
        return x, y

    # Reject positive exposure requests when the source task is empty.
    if count > 0 and len(x) == 0:
        raise ValueError("cannot sample positive exposure from an empty task.")

    # Return shape-preserving empty arrays for zero exposure.
    if count == 0:
        return x[:0], y[:0]

    indices = rng.choice(len(x), size=count, replace=count > len(x))

    return x[indices], y[indices]


def _restore_replay_label_shape(
    label_ids: np.ndarray,
    reference_labels: np.ndarray
) -> np.ndarray:
    """Represent selected integer labels like the loader's original label arrays.

    Multi-column reference labels select one-hot encoding; a single column selects
    sparse column IDs; vectors select sparse vectors. Labels are cast back to the
    reference dtype without changing their selected order.

    Args:
        label_ids (np.ndarray): Selected class IDs, flattened to an int64 vector.
        reference_labels (np.ndarray): Sparse vector/column or one-hot matrix used
            only to determine output rank, categorical width, and dtype.

    Returns:
        np.ndarray: Labels shaped ``[N]``, ``[N, 1]``, or ``[N, C]`` according to
        the reference representation, with the reference label dtype.

    Raises:
        IndexError: If an ID cannot index the reference one-hot class width.
    """

    ids = np.asarray(label_ids, dtype="int64").reshape(-1)
    reference = np.asarray(reference_labels)
    # Restore one-hot labels when the reference has multiple class columns.
    if reference.ndim == 2 and reference.shape[1] > 1:
        return np.eye(reference.shape[1], dtype=reference.dtype)[ids]

    # Restore a sparse column when the reference labels have a column axis.
    if reference.ndim > 1:
        return ids[:, None].astype(reference.dtype, copy=False)

    return ids.astype(reference.dtype, copy=False)


def _balanced_generation_labels(
    classes: Sequence[int],
    count: int,
    rng: np.random.Generator
) -> np.ndarray:
    """Allocate an exact generated-replay budget nearly equally across old classes.

    Each class receives ``count // len(classes)`` labels. A seeded random class
    ordering allocates remainder examples, then the complete label sequence is
    shuffled. Class allocations differ by at most one when the class IDs are unique.

    Args:
        classes (Sequence[int]): Old class IDs to condition on; must be nonempty
            when count is positive.
        count (int): Nonnegative number of conditioning labels to return.
        rng (np.random.Generator): Local generator used for remainder allocation
            and final ordering. A zero count does not consume it.

    Returns:
        np.ndarray: Int64 class IDs shaped ``[count]``; an empty vector for zero.
    """

    classes = [int(class_id) for class_id in classes]

    # Return no conditioning labels when the generation budget is zero.
    if count == 0:
        return np.empty((0,), dtype="int64")

    base, remainder = divmod(count, len(classes))
    shuffled = list(np.asarray(classes)[rng.permutation(len(classes))])
    labels = np.concatenate([
        np.repeat(class_id, base + int(index < remainder))
        for index, class_id in enumerate(shuffled)
    ]).astype("int64", copy=False)
    rng.shuffle(labels)

    return labels


def _cache_digest(value: np.ndarray) -> str:
    """Hash a replay array using its dtype, shape, and contiguous numeric content.

    The digest shares the recovery-array encoding, so cache validation agrees with
    checkpoint fingerprinting. It reads the array without changing or persisting it.

    Args:
        value (np.ndarray): Numeric or other non-object replay array of any shape.

    Returns:
        str: Lowercase SHA-256 hexadecimal digest including array metadata.

    Raises:
        TypeError: If the array contains an object dtype.
    """

    return _array_recovery_descriptor(value)["sha256"]


def _cached_replay_candidates(
    x: np.ndarray, 
    y: np.ndarray, 
    cache_dir: str | None, 
    cache_mode: str, 
    task_index: int, 
    old_classes: Sequence[int], 
    seed: int | None, 
    context_fingerprint: str | None = None
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Read or atomically publish one experiment-specific replay candidate pool.

    ``off`` returns inputs without I/O; ``read`` requires a matching cache; ``write``
    publishes a new pool or accepts an identical retry; ``read_write`` reads a
    matching existing pool or publishes the supplied candidates when none exists.
    Metadata and checksums authenticate reuse. Atomic no-replace publication lets a
    concurrent winner remain authoritative instead of overwriting its samples.

    Args:
        x (np.ndarray): Candidate samples shaped ``[N, ...]`` with a non-object dtype.
        y (np.ndarray): Aligned conditioning labels with N leading rows.
        cache_dir (str | None): Cache root; None/empty is allowed only for ``off``.
        cache_mode (str): Case-insensitive ``off``, ``read``, ``write``, or
            ``read_write`` operation.
        task_index (int): Zero-based task index used in metadata and the filename.
        old_classes (Sequence[int]): Ordered original class IDs defining this pool.
        seed (int | None): Recorded replay seed; None represents unseeded generation.
            This helper performs no random sampling itself.
        context_fingerprint (str | None): Experiment identity checked on reuse.
            Defaults to None for the legacy namespace.

    Returns:
        tuple[np.ndarray, np.ndarray, str | None]: Selected candidate arrays and the
        resolved archive path. ``off`` returns a None path; read modes return stored
        arrays rather than the supplied candidates. Array dtypes/shapes are retained.

    Raises:
        ValueError: If the mode/path options, saved metadata, or checksums are invalid.
        FileNotFoundError: If read mode requires an unavailable cache file.
        FileExistsError: If a write would replace incompatible candidates or a
            non-file path already occupies the archive destination.
        OSError: If cache directories/files cannot be created, read, or published.

    Side Effects:
        Enabled modes create the cache root if needed. Publication writes a private
        NPZ, creates its final hard link, and removes the temporary file. Existing
        committed candidate files are not overwritten.
    """

    cache_mode = str(cache_mode).lower()
    # Reject cache modes outside the supported read/write vocabulary.
    if cache_mode not in ("off", "write", "read", "read_write"):
        raise ValueError("replay_cache_mode must be off, write, read, or read_write.")
    # Preserve the legacy no-I/O path exactly.
    if cache_mode == "off":
        return np.asarray(x), np.asarray(y), None
    # Require a cache directory whenever cache I/O is enabled.
    if cache_dir is None or not str(cache_dir).strip():
        raise ValueError("replay_cache_dir is required when cache mode is enabled.")

    path = _replay_cache_path(
        cache_dir,
        task_index,
        old_classes,
        len(y),
        context_fingerprint=context_fingerprint,
        create_root=True
    )

    expected = {
        "schema_version": 1,
        "task_index": int(task_index),
        "old_classes": [int(class_id) for class_id in old_classes],
        "candidate_count": int(len(y)),
        "seed": seed,
        "context_fingerprint": context_fingerprint,
    }
    x, y = np.asarray(x), np.asarray(y)
    # Read an existing pool, or attempt the required read even if its path is missing.
    if cache_mode == "read" or path.exists():
        # Reject cache paths that cannot be opened as files.
        if not path.is_file():
            # Report a conflicting write target as an existing-path error.
            if cache_mode == "write":
                raise FileExistsError(f"Replay cache path is not a file: {path}")
            raise FileNotFoundError(f"Replay cache does not exist: {path}")

        with np.load(path, allow_pickle=False) as archive:
            cached_x, cached_y = archive["x"], archive["y"]
            metadata = json.loads(str(archive["metadata"].item()))
        cached_hashes = (_cache_digest(cached_x), _cache_digest(cached_y))
        valid_metadata = all(metadata.get(key) == value for key, value in expected.items())
        valid_archive = cached_hashes == (metadata.get("x_sha256"), metadata.get("y_sha256"))

        # A retried write may reuse the same pool but never replace its bytes.
        if cache_mode == "write":
            # Reject retried writes whose metadata, stored bytes, or regenerated pool differ.
            if not (valid_metadata and valid_archive and cached_hashes == (
                _cache_digest(x), _cache_digest(y)
            )):
                raise FileExistsError(
                    "Replay cache already exists with incompatible candidates: "
                    f"{path}"
                )
        # Read modes authenticate the saved pool without comparing newly supplied samples.
        else:
            # Reject a cached pool from a different experiment or stochastic stream.
            if not valid_metadata:
                raise ValueError("Replay cache metadata differs from this experiment.")
            # Reject cached arrays whose checksums no longer match their metadata.
            if not valid_archive:
                raise ValueError("Replay cache checksum validation failed.")
        return cached_x, cached_y, str(path)

    metadata = {
        **expected,
        "x_sha256": _cache_digest(x),
        "y_sha256": _cache_digest(y),
    }
    temporary = path.with_name(
        "." + path.name + ".tmp-" + uuid.uuid4().hex
    )

    try:
        with open(temporary, "xb") as stream:
            np.savez_compressed(
                stream,
                x=x,
                y=y,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True))
            )
        try:
            # Publish by atomic no-replace hard link so concurrent treatments
            # cannot overwrite the first complete authenticated candidate pool.
            os.link(temporary, path)
        except FileExistsError:
            # Reject a colliding publisher whose destination is not a regular file.
            if not path.is_file():
                raise FileExistsError(
                    f"Replay cache path is not a file: {path}"
                )
            return _cached_replay_candidates(
                x,
                y,
                cache_dir,
                cache_mode,
                task_index,
                old_classes,
                seed,
                context_fingerprint,
            )
    finally:
        # Remove the private temporary archive only if it still exists.
        if temporary.exists():
            temporary.unlink()

    return x, y, str(path)


def _replay_cache_path(
    cache_dir: str, 
    task_index: int, 
    old_classes: Sequence[int], 
    candidate_count: int, 
    context_fingerprint: str | None = None, 
    create_root: bool = False
) -> Path:
    """Resolve a compact cache filename for one task, class set, and candidate count.

    The current filename hashes the ordered old-class list and uses a short context
    fingerprint. Existing legacy filenames with literal class IDs are reused when
    no canonical filename exists, preserving compatibility with earlier caches.

    Args:
        cache_dir (str): Root directory for replay candidate archives.
        task_index (int): Zero-based task index, formatted one-based in the filename.
        old_classes (Sequence[int]): Ordered old-class IDs defining the candidate pool.
        candidate_count (int): Number of candidate rows recorded in the filename.
        context_fingerprint (str | None): Run identity prefix. Defaults to None
            for a ``legacy`` namespace; other strings use their first 16 characters.
        create_root (bool): Defaults to False for lookup only. True creates the
            cache directory and any missing parents.

    Returns:
        Path: Existing compatible legacy path or canonical NPZ destination. Except
        for optional root creation, no archive file is written.

    Raises:
        OSError: If requested cache-root creation fails.
    """

    root = Path(cache_dir)
    # Create the cache directory only when publication requests it.
    if create_root:
        root.mkdir(parents=True, exist_ok=True)
    class_text = fingerprint_state([
        int(class_id) for class_id in old_classes
    ])[:16]
    # Use a legacy namespace without a run fingerprint; otherwise use its stable prefix.
    context_text = "legacy" if context_fingerprint is None else str(
        context_fingerprint
    )[:16]
    path = root / (
        f"context-{context_text}_task-{task_index + 1:04d}_"
        f"classes-{class_text}_"
        f"candidates-{int(candidate_count)}.npz"
    )
    legacy_classes = "-".join(str(int(value)) for value in old_classes)
    legacy_path = root / (
        f"context-{context_text}_task-{task_index + 1:04d}_"
        f"classes-{legacy_classes}_"
        f"candidates-{int(candidate_count)}.npz"
    )
    # Reuse an existing legacy path only when the current canonical path is absent.
    return legacy_path if legacy_path.exists() and not path.exists() else path


class ReplayBuffer(object):
    """Store and randomly replay ``(sample, label)`` items.

    FIFO storage (the default) uses a bounded :class:`collections.deque` and is
    exactly backward compatible with the original implementation. ``reservoir``
    applies Algorithm R, giving every item observed in the stream equal
    inclusion probability. ``class_balanced`` assigns near-equal per-class
    quotas and applies reservoir sampling independently inside each class.
    Sampling is non-destructive and uses an instance-local seeded generator.

    Attributes:
        maxlen (int | None): Maximum retained item count.  ``None`` creates an
            unbounded deque and ``0`` creates a deque that retains no items.
        buffer (collections.deque): Current replay items, initialized empty.
        strategy (ReplayStrategy): Selected insertion/eviction policy.
        sample_dtype (np.dtype): Floating NumPy dtype captured from the Keras
            variable policy at construction. It controls sampled input casting
            even if the global policy subsequently changes; label dtypes remain
            those originally stored.
    """

    def __init__(
        self: ReplayBuffer, 
        maxlen: int | None, 
        strategy: ReplayStrategy | str = "fifo", 
        seed: int | float | str | bytes | bytearray | None = None
    ) -> None:
        """Create an empty replay buffer with a local random generator.

        Args:
            maxlen (int | None): Capacity passed to ``deque``. Once full,
                FIFO removes the oldest element while reservoir policies
                choose replacements probabilistically. ``None`` is unbounded.
            strategy (ReplayStrategy | str): ``"fifo"`` (the exact historical
                behavior), ``"reservoir"`` (uniform Algorithm R), or
                ``"class_balanced"`` (balanced per-class reservoirs). The
                alias ``"class-balanced"`` is accepted.
                Defaults to ``'fifo'``.
            seed (int | float | str | bytes | bytearray | None): Seed for this
                buffer's private generator. ``None`` uses system entropy and
                no module-level random state is changed.
                Defaults to ``None``.

        Returns:
            None.

        Raises:
            ValueError: If ``strategy`` is unsupported. ``deque`` validates
                the annotated capacity.
        """

        # Preserve None for unbounded storage; normalize bounded capacities as integers.
        self.maxlen = None if maxlen is None else index(maxlen)

        # Restrict insertion behavior to the documented strategy vocabulary.
        if not isinstance(strategy, str) or strategy.lower() not in _STRATEGY_ALIASES:
            raise ValueError(
                "strategy must be 'fifo', 'reservoir', or 'class_balanced'."
            )

        self.strategy: ReplayStrategy = _STRATEGY_ALIASES[strategy.lower()]
        variable_dtype = tf.keras.mixed_precision.global_policy().variable_dtype
        self.sample_dtype = np.dtype(tf.as_dtype(variable_dtype).as_numpy_dtype)
        self._rng = random.Random(seed)

        self.clear()

    def __len__(self: ReplayBuffer) -> int:
        """Return the number of currently retained items.

        Returns:
            int: A value from ``0`` through ``maxlen`` for bounded buffers.
        """

        return len(self.buffer)

    def clear(self: ReplayBuffer) -> None:
        """Empty retained replay and reset insertion counters and class-allocation state.

        The backing deque is replaced at the existing capacity. Historical stream/class
        counts and class priorities are cleared, while the private RNG keeps its current
        state; clearing therefore does not restart seeded sampling sequences.

        Returns:
            None: This buffer is empty and ready to observe a new stream.
        """

        self.buffer = deque(maxlen=self.maxlen)
        self._items_seen = 0
        self._class_order: list[object] = []
        self._class_seen: dict[object, int] = {}
        self._class_priorities: dict[object, float] = {}

    @staticmethod
    def _label_key(item: object) -> object:
        """Return a hashable class identifier from one ``(sample, label)`` item.

        Scalar and one-element sparse-column labels are used directly. A vector
        with multiple entries is treated as one-hot/probability encoded and
        mapped with ``argmax``; normalization is not checked here.

        Args:
            item (object): Replay item whose second element is its label.

        Returns:
            object: Hashable Python scalar identifying the item's class.

        Raises:
            TypeError: If the item is not a pair or its label is unsupported.
        """

        # Require the paired item representation used by continual replay.
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError(
                "class_balanced replay items must be (sample, label) pairs."
            )

        label = np.asarray(item[1])

        # Accept one sample's scalar, sparse-column, or one-hot label only.
        if label.size == 0 or label.ndim > 1:
            raise TypeError(
                "class_balanced labels must be scalars or one-dimensional vectors."
            )

        # Read scalar/sparse-column labels directly; decode multi-entry one-hot vectors.
        value = label.reshape(-1)[0].item() if label.size == 1 \
                else int(np.argmax(label))

        try:
            hash(value)
        except TypeError as error:
            raise TypeError("class_balanced labels must be hashable.") from error

        return value

    def _class_quotas(self: ReplayBuffer) -> dict[object, int]:
        """Return the current capacity allocation for every observed class.

        Random priorities allocate remainder slots. Consequently, when there
        are more classes than slots, the stored classes form a reservoir-like
        uniform subset rather than permanently favoring the earliest labels.

        Returns:
            dict[object, int]: Class-label to allowed retained count. A bounded,
            nonempty class registry allocates all ``maxlen`` slots. An empty
            registry returns ``{}``; an unbounded buffer returns observed counts
            instead. This calculation does not advance the RNG or mutate state.
        """

        # Unbounded buffers can retain every observation from each class.
        if self.maxlen is None:
            return {
                label: self._class_seen[label]
                for label in self._class_order
            }

        # An empty class registry has no capacity allocation to compute.
        if not self._class_order:
            return {}

        base, remainder = divmod(self.maxlen, len(self._class_order))
        ranked_classes = [
            (-self._class_priorities[label], index, label)
            for index, label in enumerate(self._class_order)
        ]
        ranked_classes.sort()
        extra_classes = {
            label for _, _, label in ranked_classes[:remainder]
        }

        return {
            label: base + int(label in extra_classes)
            for label in self._class_order
        }

    def _rebalance_classes(self: ReplayBuffer) -> None:
        """Downsample stored classes uniformly to their current quotas.

        For each over-capacity class, sample retained positions without
        replacement using the private RNG. Rebuild the deque in original order
        for the chosen positions. Zero-quota classes lose all retained items;
        underfilled classes remain underfilled, and unbounded buffers are
        unchanged. Observation counters and allocation priorities are retained.

        Returns:
            None: The bounded buffer is rebuilt with quota-compliant contents.
        """

        # An unbounded buffer needs no eviction or quota enforcement.
        if self.maxlen is None:
            return

        items = list(self.buffer)
        quotas = self._class_quotas()
        indices_by_class = {label: [] for label in self._class_order}
        for index, item in enumerate(items):
            indices_by_class[self._label_key(item)].append(index)

        kept_indices = set()
        for label in self._class_order:
            indices = indices_by_class[label]
            quota = quotas[label]
            # Preserve a class already at or below its feasible allocation.
            if len(indices) <= quota:
                kept_indices.update(indices)
            # Uniformly thin a class whose prior allocation was larger.
            elif quota > 0:
                kept_indices.update(self._rng.sample(indices, quota))

        # Retain only the indices selected by per-class quota thinning.
        self.buffer = deque(
            (item for index, item in enumerate(items) if index in kept_indices),
            maxlen=self.maxlen,
        )

    def _append_reservoir(self: ReplayBuffer, item: object) -> None:
        """Insert one stream item using standard uniform Algorithm R.

        Increment the stream cursor for every observation, including rejected
        items. Append until storage fills; thereafter draw uniformly from all
        observed positions and replace only if the draw addresses a retained
        slot. A bounded capacity ``k`` retains each of ``n >= k`` observations
        with probability ``k / n``. Unbounded buffers append every item;
        zero-capacity buffers retain none. Random draws use the private RNG.

        Args:
            item (object): Next item in the observed replay stream.

        Returns:
            None: The item is appended, replaces one slot, or is discarded.
        """

        self._items_seen += 1
        # Fill available storage before probabilistic replacement begins.
        if self.maxlen is None or len(self.buffer) < self.maxlen:
            self.buffer.append(item)
            return

        # A zero-capacity reservoir records the cursor but retains no item.
        if self.maxlen == 0:
            return

        replacement = self._rng.randrange(self._items_seen)

        # Algorithm R accepts exactly the first ``capacity`` draw positions.
        if replacement < self.maxlen:
            self.buffer[replacement] = item

    def _append_class_balanced(self: ReplayBuffer, item: object) -> None:
        """Insert one item into a balanced per-class reservoir.

        Register unseen labels with a random allocation priority, rebalance
        existing storage if class quotas shrink, and advance both global and
        per-class observation counters. Fill the incoming class's quota or
        apply Algorithm R using its historical count. Zero-quota classes keep
        counters only; unbounded storage retains every item. All draws use the
        buffer's private RNG and retained sample objects are stored by reference.

        Args:
            item (object): Next ``(sample, label)`` stream item.

        Returns:
            None: Class state and bounded storage are updated in place.
        """

        label = self._label_key(item)
        is_new_class = label not in self._class_seen
        # Give each newly observed class an exchangeable allocation priority.
        if is_new_class:
            self._class_order.append(label)
            self._class_seen[label] = 0
            self._class_priorities[label] = self._rng.random()

        self._items_seen += 1
        self._class_seen[label] += 1
        # A new class can reduce earlier quotas, so thin them before insertion.
        if is_new_class:
            self._rebalance_classes()

        # Unbounded class-balanced storage retains the complete stream.
        if self.maxlen is None:
            self.buffer.append(item)
            return

        quota = self._class_quotas()[label]

        # Classes outside a remainder allocation retain no item at capacity.
        if quota == 0:
            return

        # Consider replacement slots only among stored items of the incoming class.
        class_indices = [
            index for index, stored in enumerate(self.buffer)
            if self._label_key(stored) == label
        ]

        # Fill this class's available quota before reservoir replacement.
        if len(class_indices) < quota:
            self.buffer.append(item)
            return

        replacement = self._rng.randrange(self._class_seen[label])

        # Apply Algorithm R within the class's current quota.
        if replacement < quota:
            self.buffer[class_indices[replacement]] = item

    def state_dict(self: ReplayBuffer) -> dict[str, object]:
        """Return all state required for an exact deterministic continuation.

        Creates fresh outer mappings/lists without consuming randomness. The
        retained ``items`` are shallow references, so copy/serialize them before
        mutating their sample arrays. Numeric sample dtype remains a constructor
        setting and is not included in this state schema.

        Returns:
            dict[str, object]: Schema version 1 with ``maxlen``, ``strategy``,
            ordered ``items``, private Python ``rng_state``, ``items_seen``, and
            ordered ``classes`` records containing ``label``, ``seen``, and
            ``priority``. FIFO records current retained length as its cursor;
            reservoir modes record all observations. Non-balanced modes have
            no class records.
        """

        # FIFO records retained length; reservoir strategies preserve the full stream cursor.
        return {
            "schema_version": 1,
            "maxlen": self.maxlen,
            "strategy": self.strategy,
            "items": list(self.buffer),
            "rng_state": self._rng.getstate(),
            # FIFO does not need a historical stream cursor; report its current
            # retained count so the common serialized invariant remains valid.
            "items_seen": len(self.buffer) if self.strategy == "fifo" \
                        else self._items_seen,
            "classes": [{
                "label": label,
                "seen": self._class_seen[label],
                "priority": self._class_priorities[label]
            } for label in self._class_order]
        }

    def load_state_dict(
        self: ReplayBuffer, 
        state: Mapping[str, object]
    ) -> None:
        """Restore a state produced by :meth:`state_dict` without reinsertion.

        Validate version, capacity, strategy, counters, class quotas, retained
        contents, and private RNG state before replacing live storage. Restore
        sample objects by reference and preserve their order; no new insertion
        decisions or random draws occur. Capacity, strategy, and constructor
        ``sample_dtype`` remain unchanged.

        Args:
            state (Mapping[str, object]): Version-1 mapping emitted by
                ``state_dict`` or reconstructed by the recovery archive reader.
                Its capacity and strategy must match this instance.

        Returns:
            None: Storage, counters, class allocation, and private RNG state
            have been replaced in place.

        Raises:
            ValueError: If capacity, strategy, counters, or retained contents
                are incompatible with this buffer, or RNG state is invalid.
            TypeError: If malformed schema values cannot be normalized.
        """

        schema_version = int(state.get("schema_version"))
        # Accept only the state schema emitted by the current serializer.
        if schema_version != 1:
            raise ValueError("Unsupported replay-checkpoint schema version.")

        # Refuse cross-capacity restoration because it changes inclusion odds.
        if state.get("maxlen") != self.maxlen:
            raise ValueError("Replay-buffer capacity differs from the checkpoint.")

        saved_strategy = str(state.get("strategy", "fifo"))

        # Refuse restoration under a different insertion distribution.
        if saved_strategy != self.strategy:
            raise ValueError("Replay-buffer strategy differs from the checkpoint.")

        items = list(state.get("items", []))

        # Guard against silently truncated or malformed retained contents.
        if self.maxlen is not None and len(items) > self.maxlen:
            raise ValueError("Replay checkpoint exceeds the configured capacity.")

        items_seen = int(state.get("items_seen", len(items)))

        # FIFO has no algorithmic stream cursor beyond its retained contents.
        if self.strategy == "fifo":
            items_seen = len(items)

        # Reservoir cursors must cover every currently retained observation.
        if items_seen < len(items):
            raise ValueError("Replay checkpoint has an invalid stream count.")

        saved_classes = state.get("classes", [])
        # Require the serialized ordered record collection used by state_dict().
        if isinstance(saved_classes, (str, bytes)) \
        or not isinstance(saved_classes, Sequence):
            raise ValueError("Replay checkpoint class state must be a sequence.")
        classes = list(saved_classes)
        # Require complete mapping records before reading their fields.
        if any(
            not isinstance(record, Mapping)
            or not {"label", "seen", "priority"}.issubset(record)
            for record in classes
        ):
            raise ValueError("Replay checkpoint contains a malformed class record.")
        class_order = [record["label"] for record in classes]

        try:
            unique_class_count = len(set(class_order))
        except TypeError as error:
            raise ValueError(
                "Replay checkpoint contains an unhashable class label."
            ) from error
        # Each class must have exactly one allocation/counter record.
        if len(class_order) != unique_class_count:
            raise ValueError("Replay checkpoint contains duplicate classes.")

        # Allocation priorities originate from random.random() in [0, 1).
        if any(
            not 0. <= float(record["priority"]) < 1.
            for record in classes
        ):
            raise ValueError("Replay checkpoint contains an invalid class priority.")

        class_seen = {
            record["label"]: int(record["seen"])
            for record in classes
        }
        class_priorities = {
            record["label"]: float(record["priority"])
            for record in classes
        }

        # Registered balanced classes have each consumed at least one item.
        if any(count <= 0 for count in class_seen.values()):
            raise ValueError("Replay checkpoint contains a nonpositive class count.")
        # Balanced state accounts for the full stream class by class.
        if self.strategy == "class_balanced":
            # Per-class counters must sum to the global insertion cursor.
            if sum(class_seen.values()) != items_seen:
                raise ValueError("Replay class counts do not match items_seen.")
            item_labels = [self._label_key(item) for item in items]
            # Every retained item must belong to a registered class.
            if any(label not in class_seen for label in item_labels):
                raise ValueError("Replay items contain an unknown class.")
            retained_counts = {
                label: item_labels.count(label) for label in class_order
            }
            # Retention cannot exceed the number historically observed.
            if any(
                retained_counts[label] > class_seen[label]
                for label in class_order
            ):
                raise ValueError("Replay items exceed a class observation count.")

            # Reconstruct the allocation without mutating live buffer state.
            if self.maxlen is None:
                quotas = dict(class_seen)
            # An empty bounded checkpoint has no class allocation to divide.
            elif not class_order:
                quotas = {}
            # Split bounded capacity by the serialized random priorities.
            else:
                base, remainder = divmod(self.maxlen, len(class_order))
                ranked_classes = sorted(
                    (
                        -class_priorities[label], index, label
                    )
                    for index, label in enumerate(class_order)
                )
                extra_classes = {
                    label for _, _, label in ranked_classes[:remainder]
                }
                quotas = {
                    label: base + int(label in extra_classes)
                    for label in class_order
                }
            # Reject retained contents impossible under the saved allocation.
            if any(
                retained_counts[label] > quotas[label]
                for label in class_order
            ):
                raise ValueError("Replay items exceed a class allocation quota.")
        # Other strategies never carry class allocation metadata.
        elif classes:
            raise ValueError("Only class_balanced replay may contain class state.")

        validated_rng = random.Random()
        try:
            validated_rng.setstate(state["rng_state"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Replay checkpoint contains invalid RNG state.") from error

        self.buffer = deque(items, maxlen=self.maxlen)
        self._items_seen = items_seen
        self._class_order = class_order
        self._class_seen = class_seen
        self._class_priorities = class_priorities
        self._rng.setstate(validated_rng.getstate())

    def sample(
        self: ReplayBuffer, 
        list_: Sequence[object] | deque, 
        num: int
    ) -> list[object]:
        """Randomly select up to ``num`` elements without replacement.

        Args:
            list_ (Sized iterable): Population accepted by ``random.sample``;
                normally a list or the buffer's deque.
            num (int): Requested sample count. ``0`` returns an empty list;
                positive values select that many items when available.

        Returns:
            list[object]: ``[]`` for an empty population, a random sample when
            ``len(list_) >= num``, or every supplied item in its current order
            when the requested count exceeds the population.

        Python's ``random.sample`` validates the annotated count.
        """

        num = index(num)

        # Preserve the empty population without invoking random sampling.
        if len(list_) == 0:
            return []

        # Return every item when the request exceeds the population.
        if len(list_) < num:
            return list(list_)

        return self._rng.sample(list(list_), k=num)

    def sample_buffer(self: ReplayBuffer, num: int) -> list[object]:
        """Sample retained replay items without removing them.

        Args:
            num (int): Requested item count.  If it exceeds the buffer length,
                all buffered items are returned in deque order.

        Returns:
            list[object]: Sampled items, or an empty list for an empty buffer.
        """

        return self.sample(self.buffer, num)

    def append(self: ReplayBuffer, item: object) -> None:
        """Offer one stream item to the configured replay insertion policy.

        FIFO retains the newest items and evicts the oldest on overflow. Reservoir
        sampling may retain, replace, or discard the incoming item using Algorithm R;
        class-balanced replay applies that rule within the item's class quota. Zero
        capacity retains no items, while unbounded storage retains every accepted item.

        Args:
            item (object): Value to offer. Continual callers use ``(sample, label)``
                pairs; class-balanced insertion requires such a pair and a supported
                scalar, sparse-column, or one-hot label.

        Returns:
            None: Retained storage, policy counters, and possibly private RNG state
            change in place. A discarded reservoir item still advances stream counters.

        Raises:
            TypeError: If class-balanced insertion receives a malformed pair/label.
        """

        # Preserve the exact historical bounded-deque behavior by default.
        if self.strategy == "fifo":
            self.buffer.append(item)
        # Route opt-in global-uniform insertion through Algorithm R.
        elif self.strategy == "reservoir":
            self._append_reservoir(item)
        # The remaining validated strategy is the class-balanced reservoir.
        else:
            self._append_class_balanced(item)

    def extend(self: ReplayBuffer, items: Iterable[object]) -> None:
        """Offer every item from an iterable to replay in its existing stream order.

        FIFO uses deque.extend and retains the most recent capacity-limited suffix.
        Reservoir and class-balanced strategies process each item through their insertion
        policy, preserving the same counters and RNG progression as repeated append.

        Args:
            items (Iterable[object]): Stream entries to consume once. Class-balanced
                entries must be valid sample-label pairs.

        Returns:
            None: Storage, insertion counters, and any consumed private RNG state are
            updated in place. The iterable itself is not copied or reordered.

        Raises:
            TypeError: If a class-balanced stream contains a malformed item/label.
        """

        # Preserve deque.extend, including its historical ordering semantics.
        if self.strategy == "fifo":
            self.buffer.extend(items)
            return

        for item in items:
            self.append(item)

    def sample_buffer_and_prepare_dataset(
        self: ReplayBuffer, 
        num: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert sampled ``(x, y)`` pairs to NumPy training arrays.

        Args:
            num (int): Maximum number of replay pairs to sample.  Requests
                larger than the buffer return all items.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: ``(x_buffer, y_buffer)``.
            Samples are cast to ``sample_dtype`` captured at construction and
            labels retain their stored dtype. Their leading dimension is the
            number sampled; an empty buffer produces two arrays with shape
            ``(0,)``.
        """

        x_buffer, y_buffer = [], []
        for x, y in self.sample_buffer(num):
            x_buffer.append(x)
            y_buffer.append(y)

        x_buffer = np.array(x_buffer, dtype=self.sample_dtype)
        y_buffer = np.array(y_buffer)

        return x_buffer, y_buffer

    def sample_dataset_and_extend_buffer(
        self: ReplayBuffer, 
        dataset: tuple[Iterable[object], ...], 
        num: int
    ) -> None:
        """Sample aligned arrays and append their paired elements.

        Args:
            dataset (tuple[Iterable[object], ...]): Parallel iterables, normally
                ``(x_array, y_array)``.  ``zip(*dataset)`` truncates to the
                shortest iterable and creates tuple items.
            num (int): Number of pairs requested. When the zipped dataset has
                fewer than ``num`` pairs, all available dataset pairs are used.

        Returns:
            None.
        """

        items = self.sample(list(zip(*dataset)), num)
        self.extend(items)
