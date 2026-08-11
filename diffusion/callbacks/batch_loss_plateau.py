from tensorflow.keras import callbacks

import numpy as np


class BatchLossPlateau(callbacks.Callback):
    """Stop one progressive curriculum stage after batch-level loss stagnation.

    This is deliberately stage-local: ``fit_progressively`` creates a fresh
    instance for every curriculum stage, so ``best`` and ``wait`` are reset
    whenever a harder timestep interval is introduced.
    """

    def __init__(
        self, 
        monitor: str = "noise_loss", 
        patience: int = 200, 
        min_delta: float = 0., 
    ):
        super().__init__()
        if patience < 1:
            raise ValueError("patience must be >= 1.")

        self.monitor = monitor
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = np.inf
        self.wait = 0

    def on_train_batch_end(self, batch, logs=None):
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
