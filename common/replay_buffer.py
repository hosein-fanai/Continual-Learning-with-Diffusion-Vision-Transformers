"""Fixed-capacity replay storage used by the continual-learning workflow."""

from __future__ import annotations

import tensorflow as tf

import numpy as np

from collections import deque

import random

from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from typing import Literal


ReplayStrategy = Literal["fifo", "reservoir", "class_balanced"]
_STRATEGY_ALIASES = {
    "fifo": "fifo", 
    "reservoir": "reservoir", 
    "class_balanced": "class_balanced", 
    "class-balanced": "class_balanced", 
    "class_balanced_reservoir": "class_balanced"
}


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
        seed: int | float | str | bytes | bytearray | None = None, 
        strategy: ReplayStrategy | str = "fifo"
    ) -> None:
        """Create an empty replay buffer with a local random generator.

        Args:
            maxlen (int | None): Non-boolean, nonnegative capacity passed to
                ``deque``. Once full, appending removes the oldest element;
                ``None`` is unbounded.
            seed (int | float | str | bytes | bytearray | None): Seed for this
                buffer's private generator. ``None`` uses system entropy and
                no module-level random state is changed.
            strategy (ReplayStrategy | str): ``"fifo"`` (the exact historical
                behavior), ``"reservoir"`` (uniform Algorithm R), or
                ``"class_balanced"`` (balanced per-class reservoirs). The
                aliases ``"class-balanced"`` and
                ``"class_balanced_reservoir"`` are accepted.

        Returns:
            None.

        Raises:
            ValueError: If ``maxlen`` is neither ``None`` nor a non-boolean,
                nonnegative integer, or if ``strategy`` is unsupported.
        """

        # Require an optional non-boolean integral buffer capacity.
        if maxlen is not None and (
            isinstance(maxlen, bool)
            or not isinstance(maxlen, Integral)
            or maxlen < 0
        ):
            raise ValueError("maxlen must be None or a nonnegative integer.")

        self.maxlen = int(maxlen) if maxlen is not None else None

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

        classes = list(state.get("classes", []))
        class_order = [record["label"] for record in classes]

        # Each class must have exactly one allocation/counter record.
        if len(class_order) != len(set(class_order)):
            raise ValueError("Replay checkpoint contains duplicate classes.")

        class_seen = {
            record["label"]: int(record["seen"]) 
            for record in classes
        }
        class_priorities = {
            record["label"]: float(record["priority"]) 
            for record in classes
        }

        # Historical class observation counts cannot be negative.
        if any(count < 0 for count in class_seen.values()):
            raise ValueError("Replay checkpoint contains a negative class count.")
        # Balanced state accounts for the full stream class by class.
        if self.strategy == "class_balanced":
            # Per-class counters must sum to the global insertion cursor.
            if sum(class_seen.values()) != items_seen:
                raise ValueError("Replay class counts do not match items_seen.")
            # Every retained item must belong to a registered class.
            if any(self._label_key(item) not in class_seen for item in items):
                raise ValueError("Replay items contain an unknown class.")
        # Other strategies never carry class allocation metadata.
        elif classes:
            raise ValueError("Only class_balanced replay may contain class state.")

        self.buffer = deque(items, maxlen=self.maxlen)
        self._items_seen = items_seen
        self._class_order = class_order
        self._class_seen = class_seen
        self._class_priorities = class_priorities
        self._rng.setstate(state["rng_state"])

    def sample(
        self: ReplayBuffer, 
        list_: Sequence[object] | deque, 
        num: int
    ) -> list[object]:
        """Randomly select up to ``num`` elements without replacement.

        Args:
            list_ (Sized iterable): Population accepted by ``random.sample``;
                normally a list or the buffer's deque.
            num (int): Non-boolean requested sample count. ``0`` returns an
                empty list; positive values select that many items when
                available.

        Returns:
            list[object]: ``[]`` for an empty population, a random sample when
            ``len(list_) >= num``, or every supplied item in its current order
            when the requested count exceeds the population.

        Raises:
            TypeError: If ``num`` is not a non-boolean integer.
            ValueError: If ``num`` is negative.
        """

        # Reject booleans and non-integral replay sample counts.
        if isinstance(num, bool) or not isinstance(num, Integral):
            raise TypeError("num must be a non-boolean integer.")

        # Keep replay sample counts nonnegative.
        if num < 0:
            raise ValueError("num must be nonnegative.")

        num = int(num)

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

    def pop(self: ReplayBuffer) -> object:
        """Remove and return the newest (rightmost) replay item.

        Returns:
            object: The removed item.

        Raises:
            IndexError: If the buffer is empty.
        """

        return self.buffer.pop()

    def pop_and_append(self: ReplayBuffer) -> object:
        """Return the newest item after removing and immediately re-adding it.

        The final deque contents and order are unchanged; this method therefore
        acts as a non-random peek at the rightmost item.

        Returns:
            object: The newest buffered item.

        Raises:
            IndexError: If the buffer is empty.
        """

        item = self.pop()
        # Do not treat a non-destructive peek as another stream observation.
        self.buffer.append(item)
        
        return item

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
            labels to ``uint8``. Their leading dimension is the number sampled;
            an empty buffer produces two arrays with shape ``(0,)``.
        """

        x_buffer, y_buffer = [], []
        for x, y in self.sample_buffer(num):
            x_buffer.append(x)
            y_buffer.append(y)

        x_buffer = np.array(x_buffer, dtype=self.sample_dtype)
        y_buffer = np.array(y_buffer, dtype="uint8")

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


def run_self_tests() -> dict[str, str]:
    """Run capacity, sampling, mutation, and dataset-conversion tests.

    The suite covers empty, zero-capacity, bounded, and unbounded buffers;
    deterministic seeded sampling; all sampling count branches; overflow,
    clearing, popping, peeking, dtype conversion, aligned dataset ingestion,
    zip truncation, and the documented undersized-external-population behavior.

    Args:
        None.

    Returns:
        dict[str, str]: ``{"ReplayBuffer": "passed"}`` after every assertion
        succeeds.
    """

    try:
        ReplayBuffer(maxlen=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("deque must reject a negative replay capacity.")
    for invalid_capacity in (True, 1.5):
        try:
            ReplayBuffer(maxlen=invalid_capacity)
        except ValueError:
            pass
        else:
            raise AssertionError("Replay capacity must be an integer or None.")

    empty = ReplayBuffer(maxlen=3, seed=1)
    assert len(empty) == 0
    assert empty.sample_buffer(10) == []
    try:
        empty.sample([], -1)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative sample sizes must fail for every population.")
    for invalid_count in (True, 1.5):
        try:
            empty.sample([], invalid_count)
        except TypeError:
            pass
        else:
            raise AssertionError("Replay sample counts must be integers.")
    x_empty, y_empty = empty.sample_buffer_and_prepare_dataset(2)
    assert x_empty.shape == (0,) and x_empty.dtype == np.float32
    assert y_empty.shape == (0,) and y_empty.dtype == np.uint8
    try:
        empty.pop()
    except IndexError:
        pass
    else:
        raise AssertionError("Popping an empty replay buffer must fail.")
    try:
        empty.pop_and_append()
    except IndexError:
        pass
    else:
        raise AssertionError("Peeking an empty replay buffer must fail.")

    zero_capacity = ReplayBuffer(maxlen=0, seed=2)
    zero_capacity.append("discarded")
    zero_capacity.extend(["also", "discarded"])
    assert zero_capacity.maxlen == 0 and len(zero_capacity) == 0

    bounded = ReplayBuffer(maxlen=3, seed=7)
    bounded.append(1)
    bounded.extend([2, 3, 4])
    assert list(bounded.buffer) == [2, 3, 4]
    assert len(bounded) == 3
    assert bounded.sample_buffer(99) == [2, 3, 4]
    assert bounded.sample([10, 20], 0) == []
    one_sample = bounded.sample([10, 20, 30], 1)
    assert len(one_sample) == 1 and one_sample[0] in {10, 20, 30}
    assert bounded.sample(["external"], 2) == ["external"]
    try:
        bounded.sample([1], -1)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative nonempty sample sizes must fail.")

    before_peek = list(bounded.buffer)
    assert bounded.pop_and_append() == 4
    assert list(bounded.buffer) == before_peek
    assert bounded.pop() == 4
    assert list(bounded.buffer) == [2, 3]
    bounded.clear()
    assert len(bounded) == 0 and bounded.buffer.maxlen == 3

    unbounded = ReplayBuffer(maxlen=None, seed=8)
    unbounded.extend(range(100))
    assert len(unbounded) == 100 and unbounded.buffer.maxlen is None
    unbounded.clear()
    assert len(unbounded) == 0 and unbounded.buffer.maxlen is None

    seeded_a = ReplayBuffer(maxlen=10, seed=1234)
    seeded_a.extend(range(6))
    sample_a = seeded_a.sample_buffer(4)
    seeded_b = ReplayBuffer(maxlen=10, seed=1234)
    seeded_b.extend(range(6))
    sample_b = seeded_b.sample_buffer(4)
    assert sample_a == sample_b
    assert len(set(sample_a)) == 4
    assert len(seeded_a) == 6, "Sampling must be non-destructive."

    pairs = ReplayBuffer(maxlen=4, seed=5)
    pairs.extend([
        (np.array([1, 2]), 3), 
        (np.array([4, 5]), 6), 
    ])
    x_pairs, y_pairs = pairs.sample_buffer_and_prepare_dataset(99)
    np.testing.assert_array_equal(x_pairs, np.array([[1, 2], [4, 5]], np.float32))
    np.testing.assert_array_equal(y_pairs, np.array([3, 6], np.uint8))
    assert x_pairs.dtype == np.float32 and y_pairs.dtype == np.uint8

    dataset_buffer = ReplayBuffer(maxlen=10, seed=9)
    dataset_x = np.arange(12, dtype=np.float32).reshape(4, 3)
    dataset_y = np.array([0, 1, 2, 3], dtype=np.uint8)
    assert dataset_buffer.sample_dataset_and_extend_buffer(
        (dataset_x, dataset_y), 2
    ) is None
    assert len(dataset_buffer) == 2
    for x_item, y_item in dataset_buffer.buffer:
        matching_index = int(y_item)
        np.testing.assert_array_equal(x_item, dataset_x[matching_index])

    truncated_buffer = ReplayBuffer(maxlen=10, seed=10)
    truncated_buffer.sample_dataset_and_extend_buffer(
        (np.array([[1], [2], [3]]), np.array([7, 8])), 2
    )
    assert len(truncated_buffer) == 2
    assert {int(item[1]) for item in truncated_buffer.buffer} == {7, 8}

    undersized_buffer = ReplayBuffer(maxlen=5, seed=11)
    undersized_buffer.append(("existing", 1))
    undersized_buffer.sample_dataset_and_extend_buffer(
        (["new"], [2]), 2
    )
    assert list(undersized_buffer.buffer) == [
        ("existing", 1), 
        ("new", 2),
    ]

    return {"ReplayBuffer": "passed"}


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
