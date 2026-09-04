"""Fixed-capacity replay storage used by the continual-learning workflow."""

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
    """Select an exact seeded exposure count from aligned arrays."""

    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of rows.")

    if count is None:
        return x, y

    if count > 0 and len(x) == 0:
        raise ValueError("cannot sample positive exposure from an empty task.")

    if count == 0:
        return x[:0], y[:0]

    indices = rng.choice(len(x), size=count, replace=count > len(x))

    return x[indices], y[indices]


def _restore_replay_label_shape(
    label_ids: np.ndarray,
    reference_labels: np.ndarray
) -> np.ndarray:
    """Represent selected integer IDs like the loader's original labels."""

    ids = np.asarray(label_ids, dtype="int64").reshape(-1)
    reference = np.asarray(reference_labels)
    if reference.ndim == 2 and reference.shape[1] > 1:
        return np.eye(reference.shape[1], dtype=reference.dtype)[ids]

    if reference.ndim > 1:
        return ids[:, None].astype(reference.dtype, copy=False)

    return ids.astype(reference.dtype, copy=False)


def _balanced_generation_labels(
    classes: Sequence[int],
    count: int,
    rng: np.random.Generator
) -> np.ndarray:
    """Allocate an exact generated-replay count nearly equally by class."""

    classes = [int(class_id) for class_id in classes]

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
    """Hash one non-object array with dtype and shape metadata."""

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
    """Read or atomically write one matched replay candidate pool."""

    cache_mode = str(cache_mode).lower()
    if cache_mode not in ("off", "write", "read", "read_write"):
        raise ValueError("replay_cache_mode must be off, write, read, or read_write.")
    # Preserve the legacy no-I/O path exactly.
    if cache_mode == "off":
        return np.asarray(x), np.asarray(y), None
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

    should_read = cache_mode == "read" or (
        cache_mode == "read_write" and path.exists()
    )
    if should_read:
        if not path.is_file():
            raise FileNotFoundError(f"Replay cache does not exist: {path}")

        with np.load(path, allow_pickle=False) as archive:
            cached_x = archive["x"]
            cached_y = archive["y"]
            metadata = json.loads(str(archive["metadata"].item()))
        expected = {
            "schema_version": 1,
            "task_index": int(task_index),
            "old_classes": [int(class_id) for class_id in old_classes],
            "candidate_count": int(len(y)),
            "seed": seed,
            "context_fingerprint": context_fingerprint,
        }

        # Refuse a cache created for another stochastic stream or replay budget.
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Replay cache metadata differs from this experiment.")
        if metadata.get("x_sha256") != _cache_digest(cached_x) \
        or metadata.get("y_sha256") != _cache_digest(cached_y):
            raise ValueError("Replay cache checksum validation failed.")

        return cached_x, cached_y, str(path)

    x = np.asarray(x)
    y = np.asarray(y)
    # A retried write may reuse the exact authenticated pool left by an
    # interrupted task, but it must never replace different candidate bytes.
    if cache_mode == "write" and path.exists():
        if not path.is_file():
            raise FileExistsError(f"Replay cache path is not a file: {path}")

        with np.load(path, allow_pickle=False) as archive:
            cached_x = archive["x"]
            cached_y = archive["y"]
            metadata = json.loads(str(archive["metadata"].item()))

        expected = {
            "schema_version": 1,
            "task_index": int(task_index),
            "old_classes": [int(class_id) for class_id in old_classes],
            "candidate_count": int(len(y)),
            "seed": seed,
            "context_fingerprint": context_fingerprint
        }
        # Authenticate metadata, stored checksums, and the regenerated pool.
        valid_metadata = all(
            metadata.get(key) == value
            for key, value in expected.items()
        )
        valid_archive = (
            metadata.get("x_sha256") == _cache_digest(cached_x)
            and metadata.get("y_sha256") == _cache_digest(cached_y)
        )
        same_candidates = (
            _cache_digest(cached_x) == _cache_digest(x)
            and _cache_digest(cached_y) == _cache_digest(y)
        )

        if not (valid_metadata and valid_archive and same_candidates):
            raise FileExistsError(
                "Replay cache already exists with incompatible candidates: "
                f"{path}"
            )

        return cached_x, cached_y, str(path)

    metadata = {
        "schema_version": 1,
        "task_index": int(task_index),
        "old_classes": [int(class_id) for class_id in old_classes],
        "candidate_count": int(len(y)),
        "seed": seed,
        "context_fingerprint": context_fingerprint,
        "x_sha256": _cache_digest(x),
        "y_sha256": _cache_digest(y)
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
    """Return the condition-independent path for one replay candidate pool."""

    root = Path(cache_dir)
    if create_root:
        root.mkdir(parents=True, exist_ok=True)
    class_text = fingerprint_state([
        int(class_id) for class_id in old_classes
    ])[:16]
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
                appending removes the oldest element; ``None`` is unbounded.
            strategy (ReplayStrategy | str): ``"fifo"`` (the exact historical
                behavior), ``"reservoir"`` (uniform Algorithm R), or
                ``"class_balanced"`` (balanced per-class reservoirs). The
                alias ``"class-balanced"`` is accepted.
            seed (int | float | str | bytes | bytearray | None): Seed for this
                buffer's private generator. ``None`` uses system entropy and
                no module-level random state is changed.

        Returns:
            None.

        Raises:
            ValueError: If ``strategy`` is unsupported. ``deque`` validates
                the annotated capacity.
        """

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
        """Replace the backing deque with a new empty deque.

        Returns:
            None.
        """

        self.buffer = deque(maxlen=self.maxlen)
        self._items_seen = 0
        self._class_order: list[object] = []
        self._class_seen: dict[object, int] = {}
        self._class_priorities: dict[object, float] = {}

    @staticmethod
    def _label_key(item: object) -> object:
        """Return a hashable class identifier from one ``(sample, label)`` item.

        Scalar labels are used directly. A one-dimensional vector is treated as
        one-hot/probability encoded and mapped with ``argmax``.

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
            dict[object, int]: Per-class quotas summing to ``maxlen``.
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

        self.buffer = deque(
            (item for index, item in enumerate(items) if index in kept_indices),
            maxlen=self.maxlen,
        )

    def _append_reservoir(self: ReplayBuffer, item: object) -> None:
        """Insert one stream item using standard uniform Algorithm R.

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

        Returns:
            dict[str, object]: Capacity, strategy, retained items, private RNG,
            stream count, and class-reservoir allocation state.
        """

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

        Args:
            state (Mapping[str, object]): Serialized replay state.

        Returns:
            None.

        Raises:
            ValueError: If capacity, strategy, counters, or retained contents
                are incompatible with this buffer.
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
        if items_seen < len(items) or items_seen < 0:
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
            not np.isfinite(float(record["priority"]))
            or not 0. <= float(record["priority"]) < 1.
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
        """Append one item, evicting the oldest item if capacity is full.

        Args:
            item (object): Value to retain; continual-learning callers use an
                ``(x, y)`` pair.

        Returns:
            None.
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
        """Append every item from an iterable in order.

        Args:
            items (Iterable[object]): Values to add.  With a bounded deque, only
                the newest ``maxlen`` values remain after overflow.

        Returns:
            None.
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
            Samples are cast to the active policy's stable variable dtype and
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
