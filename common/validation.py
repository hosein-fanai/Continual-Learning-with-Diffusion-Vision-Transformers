"""Optimization-invariant validation helpers shared across project modules."""

from __future__ import annotations


def require(condition: object, message: object = None) -> None:
    """Raise :class:`AssertionError` when a required condition is false.

    Unlike Python's ``assert`` statement, this check remains active when the
    interpreter runs with ``-O``. It intentionally retains assertion-style
    truth testing and exception behavior for internal invariants.

    Args:
        condition (object): Value tested with normal Python truth semantics.
        message (object): Optional object passed to :class:`AssertionError`.
            ``None`` preserves the zero-argument assertion form.

    Returns:
        None: The condition was truthy.

    Raises:
        AssertionError: If ``condition`` is falsey.
    """

    # Return without allocating an exception when the invariant holds.
    if condition:
        return

    # Preserve the argument-free form produced by ``assert condition``.
    if message is None:
        raise AssertionError

    raise AssertionError(message)
