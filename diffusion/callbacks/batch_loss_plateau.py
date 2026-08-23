"""Batch-granularity early stopping for progressive training stages."""

from tensorflow.keras import callbacks

import numpy as np
from typing import Any


class BatchLossPlateau(callbacks.Callback):
    """Stop one progressive curriculum stage after batch-level loss stagnation.

    This is deliberately stage-local: ``fit_progressively`` creates a fresh
    instance for every curriculum stage, so ``best`` and ``wait`` are reset
    whenever a harder timestep interval is introduced.

    Args:
        monitor: Key read from the Keras batch ``logs`` mapping, commonly
            ``"noise_loss"`` or another scalar training metric.
        patience: Positive number of consecutive non-improving batches
            tolerated before training is stopped, matching Keras callback
            patience semantics.
        min_delta: Non-negative improvement margin. A value is improving only
            when ``current < best - min_delta``.

    Inputs:
        Keras supplies a zero-based integer batch index and an optional mapping
        of string metric names to scalar numeric values at batch end.

    Outputs:
        Callback hooks return ``None``. On a plateau, the callback sets
        ``model.stop_training`` to ``True``.
    """

    def __init__(
        self, 
        monitor: str = "noise_loss", 
        patience: int = 200, 
        min_delta: float = 0.
    ) -> None:
        """Initialize an empty stage-local best value and wait counter.

        Args:
            monitor (str): Metric key read from the Keras batch-log mapping.
            patience (int): Positive number of non-improving batches tolerated.
            min_delta (float): Non-negative minimum decrease that counts as an
                improvement.

        Returns:
            ``None``.
        """

        super().__init__()

        # Require a usable metric key for batch-log lookup.
        if not isinstance(monitor, str) or not monitor:
            raise ValueError("monitor must be a non-empty string.")
        # Require a positive integer number of tolerated plateau batches.
        if not isinstance(patience, int) or isinstance(patience, bool) \
        or patience < 1:
            raise ValueError("patience must be a positive integer.")
        # Require a finite, non-negative improvement threshold.
        if not np.isscalar(min_delta) or not np.isfinite(min_delta) \
        or min_delta < 0:
            raise ValueError("min_delta must be a finite non-negative scalar.")

        self.monitor = monitor
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = np.inf
        self.wait = 0

    def on_train_batch_end(
        self, 
        batch: int, 
        logs: dict[str, Any] | None = None
    ) -> None:
        """Inspect a completed batch and stop after sustained non-improvement.

        Args:
            batch (int): Zero-based batch index supplied by Keras. It is not
                used in the plateau calculation.
            logs (dict[str, Any] | None): Optional mapping from metric name to
                a scalar numeric value. Missing ``monitor`` keys leave all
                state unchanged; values must convert to Python ``float``.

        Returns:
            ``None``. ``best`` and ``wait`` are updated in place, and the bound
            model may have ``stop_training=True`` assigned.
        """

        logs = {} if logs is None else logs

        current = logs.get(self.monitor, None)
        # Ignore batches that do not report the monitored metric.
        if current is None:
            return

        current = float(current)
        # Record a sufficiently lower loss as a new best value.
        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
            return

        self.wait += 1
        # Stop training once the configured plateau patience is exhausted.
        if self.wait >= self.patience:
            self.model.stop_training = True


def run_self_tests() -> dict[str, str]:
    """Test initialization and every update branch of :class:`BatchLossPlateau`.

    Args:
        None.

    Returns:
        dict[str, str]: A one-entry mapping after patience boundaries, missing logs,
        improvements, minimum deltas, plateau stopping, and scalar conversion.
    """

    from types import SimpleNamespace


    for invalid_patience in (-3, 0, 1.5, True):
        try:
            BatchLossPlateau(patience=invalid_patience)
        except ValueError:
            pass
        else:
            raise AssertionError("patience values below one must fail.")

    callback = BatchLossPlateau(
        monitor="custom_loss", 
        patience=2, 
        min_delta=0.1,
    )
    model = SimpleNamespace(stop_training=False)
    callback.set_model(model)
    assert np.isinf(callback.best) and callback.wait == 0

    callback.on_train_batch_end(0, None)
    callback.on_train_batch_end(1, {})
    callback.on_train_batch_end(2, {"other": 1.0})
    assert np.isinf(callback.best) and callback.wait == 0

    callback.on_train_batch_end(3, {"custom_loss": np.float32(2.0)})
    assert callback.best == 2.0 and callback.wait == 0
    callback.on_train_batch_end(4, {"custom_loss": 1.95})
    assert callback.best == 2.0 and callback.wait == 1
    callback.on_train_batch_end(5, {"custom_loss": 1.9})
    assert callback.wait == 2 and model.stop_training
    model.stop_training = False
    callback.on_train_batch_end(6, {"custom_loss": 1.89})
    assert callback.best == 1.89 and callback.wait == 0
    callback.on_train_batch_end(7, {"custom_loss": 1.9})
    callback.on_train_batch_end(8, {"custom_loss": 2.0})
    assert model.stop_training and callback.wait == 2

    default_callback = BatchLossPlateau(patience=1)
    default_model = SimpleNamespace(stop_training=False)
    default_callback.set_model(default_model)
    default_callback.on_train_batch_end(0, {"noise_loss": 1})
    default_callback.on_train_batch_end(1, {"noise_loss": 1})
    assert default_model.stop_training

    try:
        BatchLossPlateau(patience=1, min_delta=-0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("negative min_delta values must fail.")
    for invalid_delta in (np.inf, np.nan, [0.1]):
        try:
            BatchLossPlateau(patience=1, min_delta=invalid_delta)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("min_delta must be a finite scalar.")

    return {"BatchLossPlateau": "passed"}


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
