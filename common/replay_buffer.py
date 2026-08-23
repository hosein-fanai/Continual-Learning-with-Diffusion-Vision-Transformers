"""Fixed-capacity replay storage used by the continual-learning workflow."""

from __future__ import annotations

import numpy as np

from collections import deque

from collections.abc import Iterable, Sequence
from numbers import Integral

import random


class ReplayBuffer(object):
    """Store and randomly replay ``(sample, label)`` items.

    The backing :class:`collections.deque` discards the oldest item from its
    left side when a bounded buffer is full. Sampling is non-destructive and
    uses an instance-local seeded random generator.

    Attributes:
        maxlen (int | None): Maximum retained item count.  ``None`` creates an
            unbounded deque and ``0`` creates a deque that retains no items.
        buffer (collections.deque): Current replay items, initialized empty.
    """

    def __init__(
        self: ReplayBuffer, 
        maxlen: int | None, 
        seed: int | float | str | bytes | bytearray | None = None
    ) -> None:
        """Create an empty replay buffer with a local random generator.

        Args:
            maxlen (int | None): Non-boolean, nonnegative capacity passed to
                ``deque``. Once full, appending removes the oldest element;
                ``None`` is unbounded.
            seed (int | float | str | bytes | bytearray | None): Seed for this
                buffer's private generator. ``None`` uses system entropy and
                no module-level random state is changed.

        Returns:
            None.

        Raises:
            ValueError: If ``maxlen`` is neither ``None`` nor a non-boolean,
                nonnegative integer.
        """

        # Require an optional non-boolean integral buffer capacity.
        if maxlen is not None and (
            isinstance(maxlen, bool)
            or not isinstance(maxlen, Integral)
            or maxlen < 0
        ):
            raise ValueError("maxlen must be None or a nonnegative integer.")

        self.maxlen = int(maxlen) if maxlen is not None else None

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

        self.buffer.append(item)

    def extend(self: ReplayBuffer, items: Iterable[object]) -> None:
        """Append every item from an iterable in order.

        Args:
            items (Iterable[object]): Values to add.  With a bounded deque, only
                the newest ``maxlen`` values remain after overflow.

        Returns:
            None.
        """

        self.buffer.extend(items)

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
        self.append(item)
        
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
            Samples are cast to ``float32`` and labels to ``uint8``.  Their
            leading dimension is the number sampled; an empty buffer produces
            two arrays with shape ``(0,)``.
        """

        x_buffer, y_buffer = [], []
        for x, y in self.sample_buffer(num):
            x_buffer.append(x)
            y_buffer.append(y)

        x_buffer = np.array(x_buffer, dtype="float32")
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
