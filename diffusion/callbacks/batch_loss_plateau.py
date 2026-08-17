"""Batch-granularity early stopping for progressive training stages."""

from tensorflow.keras import callbacks

import numpy as np


class BatchLossPlateau(callbacks.Callback):
    """Stop one progressive curriculum stage after batch-level loss stagnation.

    This is deliberately stage-local: ``fit_progressively`` creates a fresh
    instance for every curriculum stage, so ``best`` and ``wait`` are reset
    whenever a harder timestep interval is introduced.

    Args:
        monitor: Key read from the Keras batch ``logs`` mapping, commonly
            ``"noise_loss"`` or another scalar training metric.
        patience: Positive number of non-improving batches tolerated. Because
            stopping occurs when ``wait > patience``, training is stopped on
            the ``patience + 1``-th consecutive non-improving batch.
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
        min_delta: float = 0., 
    ):
        """Initialize an empty stage-local best value and wait counter.

        Arguments and accepted types are documented on the class.

        Returns:
            ``None``.
        """

        super().__init__()
        if patience < 1:
            raise ValueError("patience must be >= 1.")

        self.monitor = monitor
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = np.inf
        self.wait = 0

    def on_train_batch_end(self, batch, logs=None):
        """Inspect a completed batch and stop after sustained non-improvement.

        Args:
            batch: Zero-based integer batch index supplied by Keras. It is not
                used in the plateau calculation.
            logs: Optional mapping from metric name to scalar numeric value.
                Missing ``monitor`` keys leave all state unchanged; values must
                be convertible to Python ``float``.

        Returns:
            ``None``. ``best`` and ``wait`` are updated in place, and the bound
            model may have ``stop_training=True`` assigned.
        """

        logs = {} if logs is None else logs

        current = logs.get(self.monitor, None)
        if current is None:
            return

        current = float(current)
        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
            return

        self.wait += 1
        if self.wait > self.patience:
            self.model.stop_training = True
