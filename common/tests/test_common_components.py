"""Focused tests for small reusable common components."""

from __future__ import annotations

import unittest

from types import SimpleNamespace

import tensorflow as tf

from common.argument_saver import (
    ArgumentSaver,
    ArgumentSaverLayer,
    ArgumentSaverModel,
)
from common.lr_logger_callback import LrLoggerCallback
from common.masked_loss import MaskedLoss


class _LayerProbe(ArgumentSaverLayer):
    """Minimal serializable layer used by argument-saver tests."""

    def __init__(self, value: int = 1, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._save_init_args(locals())


class _ModelProbe(ArgumentSaverModel):
    """Minimal serializable model used by argument-saver tests."""

    def __init__(self, width: int = 2, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._save_init_args(locals())


class CommonComponentTests(unittest.TestCase):
    """Verify serialization, learning-rate logging, and masked losses."""

    def test_argument_saver_copies_mutable_values(self) -> None:
        """Caller mutation must not change saved constructor state."""

        saver = ArgumentSaver()
        source = {"items": [1, 2]}
        saved = saver._save_init_args({
            "self": saver,
            "payload": source,
            "build": False,
        })
        source["items"].append(3)

        self.assertEqual(saved, {"payload": {"items": [1, 2]}, "build": False})
        self.assertEqual(saver.payload, {"items": [1, 2]})
        self.assertFalse(saver.build_)

    def test_argument_saver_keras_round_trip(self) -> None:
        """Layer and model constructor values must survive Keras config."""

        layer = _LayerProbe(value=4, name="probe_layer", trainable=False)
        layer_clone = _LayerProbe.from_config(layer.get_config())
        self.assertEqual(layer_clone.value, 4)
        self.assertFalse(layer_clone.trainable)

        model = _ModelProbe(width=8, name="probe_model")
        model_clone = _ModelProbe.from_config(model.get_config())
        self.assertEqual(model_clone.width, 8)
        self.assertEqual(model_clone.name, "probe_model")

    def test_learning_rate_logger_handles_values_and_schedules(self) -> None:
        """The callback must log scalar and scheduled effective rates."""

        callback = LrLoggerCallback()
        optimizer = SimpleNamespace(
            learning_rate=tf.Variable(0.125, dtype=tf.float32),
            iterations=tf.Variable(2, dtype=tf.int64),
        )
        callback.set_model(SimpleNamespace(optimizer=optimizer))
        logs = {"loss": 1.0}
        callback.on_epoch_end(0, logs)
        self.assertAlmostEqual(logs["learning_rate"], 0.125)

        optimizer.learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=0.4,
            decay_steps=1,
            decay_rate=1.0,
        )
        callback.on_epoch_end(1, logs)
        self.assertEqual(logs["loss"], 1.0)
        self.assertAlmostEqual(logs["learning_rate"], 0.4 / 3.0)

    def test_masked_loss_modes_and_serialization(self) -> None:
        """MAE/MSE must use the target prefix and serialize exactly."""

        y_true = tf.constant([
            [1.0, 3.0, 99.0],
            [2.0, 4.0, -99.0],
        ])
        y_pred = tf.constant([
            [0.0, 1.0],
            [2.0, 2.0],
        ])

        mae = MaskedLoss()
        mse = MaskedLoss(loss_type="mse")
        self.assertAlmostEqual(float(mae(y_true, y_pred)), 1.25)
        self.assertAlmostEqual(float(mse(y_true, y_pred)), 2.25)

        clone = tf.keras.losses.deserialize(tf.keras.losses.serialize(mae))
        self.assertIsInstance(clone, MaskedLoss)
        self.assertEqual(clone.loss_type, "mae")


if __name__ == "__main__":
    unittest.main()
