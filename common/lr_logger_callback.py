"""Keras callback for adding the effective learning rate to epoch logs."""

from __future__ import annotations

from tensorflow.keras import callbacks
from tensorflow.keras import backend as K


class LrLoggerCallback(callbacks.Callback):
    """Record an optimizer's current learning rate after every epoch.

    Keras assigns ``model`` when the callback is attached to ``fit``.  The
    callback supports both scalar/variable learning rates and callable
    schedules, and writes a Python ``float`` under ``"learning_rate"`` in the
    epoch log mapping.
    """

    def on_epoch_end(
        self: LrLoggerCallback, 
        epoch: int, 
        logs: dict[str, object] | None = None
    ) -> None:
        """Insert the effective learning rate into Keras epoch logs.

        Args:
            epoch (int): Zero-based completed epoch index.  It is accepted for
                the callback protocol and does not alter the calculation.
            logs (dict[str, object] | None): Mutable Keras log mapping.  When a
                dictionary is supplied it receives ``learning_rate``.  When
                ``None`` is supplied a temporary dictionary is created, so no
                value is returned to the caller.

        Returns:
            None.

        Raises:
            AttributeError: If no model/optimizer has been attached.
        """

        lr = self.model.optimizer.learning_rate
        # Evaluate scheduled learning rates at the current optimizer step.
        if callable(lr):
            lr = lr(self.model.optimizer.iterations)

        # Create a log mapping when Keras supplies no mapping.
        if logs is None:
            logs = {}
        logs["learning_rate"] = float(K.get_value(lr))
