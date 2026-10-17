"""Record an optimizer's effective learning rate in Keras epoch logs.

``LrLogger`` reads scalar learning rates or evaluates callable schedules
at the optimizer's current iteration. It adds a Python float to the supplied
log mapping, allowing ordinary history and TensorBoard callbacks to consume it.
The callback changes neither optimizer configuration nor training weights.
"""

from __future__ import annotations

from tensorflow.keras import callbacks
from tensorflow.keras import backend as K

from common.keras_compat import optimizer_iterations


class LrLogger(callbacks.Callback):
    """Record an optimizer's current learning rate after every epoch.

    Keras assigns ``model`` when the callback is attached to ``fit``.  The
    callback supports both scalar/variable learning rates and callable
    schedules, and writes a Python ``float`` under ``"learning_rate"`` in the
    epoch log mapping.
    """

    recovery_state_scope = "stateless"

    def get_recovery_config(self) -> dict[str, object]:
        """Declare the observer's empty configuration for strict task recovery.

        Learning-rate and iteration state belong to the optimizer. The callback
        reads that restored state and retains no state between task fits.

        Returns:
            config (dict[str, object]): Empty configuration for this fixed observer.
        """

        return {}

    def on_epoch_end(
        self: LrLogger, 
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
                Defaults to ``None``.

        Returns:
            None.

        Raises:
            AttributeError: If no model/optimizer has been attached.
        """

        lr = self.model.optimizer.learning_rate
        lr = lr(
            optimizer_iterations(
                self.model.optimizer
            )
        ) if callable(lr) else lr

        # Create a log mapping when Keras supplies no mapping.
        if logs is None:
            logs = {}
        logs["learning_rate"] = float(K.get_value(lr))
