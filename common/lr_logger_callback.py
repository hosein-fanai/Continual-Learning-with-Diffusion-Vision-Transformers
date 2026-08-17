"""Keras callback for adding the effective learning rate to epoch logs."""

from tensorflow.keras import callbacks
from tensorflow.keras import backend as K


class LrLoggerCallback(callbacks.Callback):
    """Record an optimizer's current learning rate after every epoch.

    Keras assigns ``model`` when the callback is attached to ``fit``.  The
    callback supports both scalar/variable learning rates and callable
    schedules, and writes a Python ``float`` under ``"learning_rate"`` in the
    epoch log mapping.
    """

    def on_epoch_end(self, epoch, logs=None):
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
        if callable(lr):
            lr = lr(self.model.optimizer.iterations)

        logs = logs or {}
        logs["learning_rate"] = float(K.get_value(lr))


def run_self_tests() -> dict[str, str]:
    """Run callback tests for scalar and scheduled learning rates.

    The checks exercise attached and unattached callbacks, tensor/scalar and
    callable rates, preservation of existing log entries, and the documented
    behavior in which an empty dictionary is replaced rather than mutated.

    Args:
        None.

    Returns:
        dict[str, str]: ``{"LrLoggerCallback": "passed"}`` after all
        assertions succeed.
    """

    from types import SimpleNamespace

    import tensorflow as tf


    callback = LrLoggerCallback()
    fixed_optimizer = SimpleNamespace(
        learning_rate=tf.Variable(0.125, dtype=tf.float32), 
        iterations=tf.Variable(2, dtype=tf.int64), 
    )
    callback.set_model(SimpleNamespace(optimizer=fixed_optimizer))
    logs = {"loss": 1.0}
    assert callback.on_epoch_end(0, logs) is None
    assert logs["loss"] == 1.0
    assert abs(logs["learning_rate"] - 0.125) < 1e-7

    fixed_optimizer.learning_rate.assign(0.0625)
    callback.on_epoch_end(1, logs)
    assert abs(logs["learning_rate"] - 0.0625) < 1e-7


    def scheduled_rate(step: tf.Tensor | tf.Variable) -> tf.Tensor:
        """Return a deterministic rate for the supplied optimizer step.

        Args:
            step (tf.Tensor | tf.Variable): Current optimizer iteration.

        Returns:
            tf.Tensor: ``0.4 / (1 + step)`` as ``float32``.
        """

        return tf.constant(0.4, tf.float32) / (
            1.0 + tf.cast(step, tf.float32)
        )


    scheduled_optimizer = SimpleNamespace(
        learning_rate=scheduled_rate, 
        iterations=tf.Variable(3, dtype=tf.int64), 
    )
    callback.set_model(SimpleNamespace(optimizer=scheduled_optimizer))
    scheduled_logs = {"metric": 2.0}
    callback.on_epoch_end(2, scheduled_logs)
    assert abs(scheduled_logs["learning_rate"] - 0.1) < 1e-7
    assert scheduled_logs["metric"] == 2.0

    empty_logs = {}
    callback.on_epoch_end(3, empty_logs)
    assert empty_logs == {}, (
        "The current `logs or {}` implementation intentionally does not "
        "mutate an empty caller-supplied dictionary."
    )
    assert callback.on_epoch_end(4, None) is None

    unattached = LrLoggerCallback()
    try:
        unattached.on_epoch_end(0, {})
    except AttributeError:
        pass
    else:
        raise AssertionError("An unattached callback must not invent an optimizer.")

    return {"LrLoggerCallback": "passed"}


if __name__ == "__main__":
    print(run_self_tests())
