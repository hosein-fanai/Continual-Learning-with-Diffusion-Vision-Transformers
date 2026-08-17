"""Fixed-capacity replay storage used by the continual-learning workflow."""

import numpy as np

from collections import deque

import random


class ReplayBuffer(object):
    """Store and randomly replay ``(sample, label)`` items.

    The backing :class:`collections.deque` discards the oldest item from its
    left side when a bounded buffer is full.  Sampling is non-destructive and
    uses Python's process-wide :mod:`random` generator.

    Attributes:
        maxlen (int | None): Maximum retained item count.  ``None`` creates an
            unbounded deque and ``0`` creates a deque that retains no items.
        buffer (collections.deque): Current replay items, initialized empty.
    """

    def __init__(self, maxlen, seed=None):
        """Create an empty replay buffer and seed Python's random generator.

        Args:
            maxlen (int | None): Capacity passed to ``deque``.  Once full,
                appending removes the oldest element; ``None`` is unbounded.
            seed (int | float | str | bytes | bytearray | None): Seed passed to
                ``random.seed``.  ``None`` uses system entropy.  Because this
                seeds the module-level generator, it also affects other code
                that uses :mod:`random`.

        Returns:
            None.
        """
        self.maxlen = maxlen

        random.seed(seed)

        self.clear()

    def __len__(self):
        """Return the number of currently retained items.

        Returns:
            int: A value from ``0`` through ``maxlen`` for bounded buffers.
        """
        return len(self.buffer)

    def clear(self):
        """Replace the backing deque with a new empty deque.

        Returns:
            None.
        """
        self.buffer = deque(maxlen=self.maxlen)

    def sample(self, list_, num):
        """Randomly select up to ``num`` elements without replacement.

        Args:
            list_ (Sized iterable): Population accepted by ``random.sample``;
                normally a list or the buffer's deque.
            num (int): Requested sample count.  ``0`` returns an empty list when
                the population is nonempty; positive values select that many
                items when available.

        Returns:
            list[object]: ``[]`` for an empty population, a random sample when
            ``len(list_) >= num``, or ``list(self.buffer)`` when the requested
            count exceeds the population.  The last behavior is significant
            for populations other than ``self.buffer``: it returns the replay
            buffer, not the supplied population.

        Raises:
            ValueError: If ``num`` is negative and sampling reaches
                ``random.sample``.
        """
        if len(list_) == 0:
            return []
        
        if len(list_) < num:
            return list(self.buffer)

        return random.sample(list_, k=num)

    def sample_buffer(self, num):
        """Sample retained replay items without removing them.

        Args:
            num (int): Requested item count.  If it exceeds the buffer length,
                all buffered items are returned in deque order.

        Returns:
            list[object]: Sampled items, or an empty list for an empty buffer.
        """
        return self.sample(self.buffer, num)

    def append(self, item):
        """Append one item, evicting the oldest item if capacity is full.

        Args:
            item (object): Value to retain; continual-learning callers use an
                ``(x, y)`` pair.

        Returns:
            None.
        """
        self.buffer.append(item)

    def extend(self, items):
        """Append every item from an iterable in order.

        Args:
            items (Iterable[object]): Values to add.  With a bounded deque, only
                the newest ``maxlen`` values remain after overflow.

        Returns:
            None.
        """
        self.buffer.extend(items)

    def pop(self):
        """Remove and return the newest (rightmost) replay item.

        Returns:
            object: The removed item.

        Raises:
            IndexError: If the buffer is empty.
        """
        return self.buffer.pop()

    def pop_and_append(self):
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

    def sample_buffer_and_prepare_dataset(self, num):
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

    def sample_dataset_and_extend_buffer(self, dataset, num):
        """Sample aligned arrays and append their paired elements.

        Args:
            dataset (tuple[Iterable[object], ...]): Parallel iterables, normally
                ``(x_array, y_array)``.  ``zip(*dataset)`` truncates to the
                shortest iterable and creates tuple items.
            num (int): Number of pairs requested.  When the zipped dataset has
                fewer than ``num`` pairs, :meth:`sample` currently returns the
                existing buffer rather than all dataset pairs, so the buffer is
                duplicated instead of ingesting that undersized dataset.

        Returns:
            None.
        """
        items = self.sample(list(zip(*dataset)), num)
        self.extend(items)
