"""Regression checks for serialization, masked losses, and learning-rate logging.

Small Keras probes isolate constructor metadata and callback behavior. Loss tests compare
explicit reductions and sample weighting, and reject shapes that would accidentally
broadcast. Inputs are local tensors and simple state objects; no dataset download or
training study is required.

Inputs are fixtures constructed by the test methods and their helpers. Tests return no
application result: unittest records assertion outcomes and errors. Run this module directly
or through ``python -m unittest`` discovery. Importing it defines fixtures and cases; it
does not itself start a test run.
"""

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
    """Minimal serializable layer used by argument-saver tests.

    This fixture implements only the interface required by its surrounding regression tests.
    Construction and mutable state are described by ``__init__``; it does not provide a
    general production replacement.

    Args:
        value (int): Scalar metadata used to test saved constructor arguments. Defaults to
            ``1``.
        **kwargs (object): Optional Keras Layer constructor options such as name, dtype, and
            trainable; empty by default.

    Returns:
        _LayerProbe: A new local test fixture with independent instance state.
    """

    def __init__(self, value: int = 1, **kwargs: object) -> None:
        """Create a layer that records a scalar constructor value.

        The probe has no custom forward computation. It delegates Keras setup and stores
        value through ArgumentSaver for serialization tests.

        Args:
            value (int): Scalar metadata used to test saved constructor arguments. Defaults
                to ``1``.
            **kwargs (object): Optional Keras Layer constructor options such as name, dtype,
                and trainable; empty by default.

        Returns:
            None: The fixture state is initialized in place.
        """

        super().__init__(**kwargs)
        self._save_init_args(locals())


class _ModelProbe(ArgumentSaverModel):
    """Minimal serializable model used by argument-saver tests.

    This fixture implements only the interface required by its surrounding regression tests.
    Construction and mutable state are described by ``__init__``; it does not provide a
    general production replacement.

    Args:
        width (int): Width metadata checked by the serialization test; no actual layer is
            built from it. Defaults to ``2``.
        **kwargs (object): Optional Keras Model constructor options; empty by default.

    Returns:
        _ModelProbe: A new local test fixture with independent instance state.
    """

    def __init__(self, width: int = 2, **kwargs: object) -> None:
        """Create a model that records its requested width as constructor metadata.

        Args:
            width (int): Width metadata checked by the serialization test; no actual layer
                is built from it. Defaults to ``2``.
            **kwargs (object): Optional Keras Model constructor options; empty by default.

        Returns:
            None: The fixture state is initialized in place.
        """

        super().__init__(**kwargs)
        self._save_init_args(locals())


class CommonComponentTests(unittest.TestCase):
    """Verify serialization, learning-rate logging, and masked losses.

    The unittest runner executes the selected test method with its local fixtures;
    individual methods describe the configurations and failure cases they exercise. There is
    no application model or experiment result returned by constructing this test case.

    Args:
        methodName (str): Test method selected by unittest. Defaults to ``"runTest"``;
            discovery supplies each named ``test_*`` method.

    Attributes:
        _testMethodName (str): Selected method name maintained by unittest.
    """

    def test_argument_saver_copies_mutable_values(self) -> None:
        """Caller mutation must not change saved constructor state.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

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
        """Layer and model constructor values must survive Keras config.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        layer = _LayerProbe(value=4, name="probe_layer", trainable=False)
        layer_clone = _LayerProbe.from_config(layer.get_config())
        self.assertEqual(layer_clone.value, 4)
        self.assertFalse(layer_clone.trainable)

        model = _ModelProbe(width=8, name="probe_model")
        model_clone = _ModelProbe.from_config(model.get_config())
        self.assertEqual(model_clone.width, 8)
        self.assertEqual(model_clone.name, "probe_model")

    def test_learning_rate_logger_handles_values_and_schedules(self) -> None:
        """The callback must log scalar and scheduled effective rates.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

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
        """MAE/MSE must use the target prefix and serialize exactly.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

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

    def test_masked_loss_preserves_sample_weighting_for_structured_outputs(self) -> None:
        """Average features per example before Keras applies sample weights.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        targets = tf.constant([[[1., 9.], [3., 9.]], [[5., 9.], [7., 9.]]])
        predictions = tf.zeros((2, 2, 1))
        loss = MaskedLoss()
        self.assertAlmostEqual(float(loss(targets, predictions, sample_weight=[1., 0.])), 1.)

    def test_masked_loss_rejects_broadcasting_in_a_dynamic_graph(self) -> None:
        """Singleton targets must not silently broadcast across examples/classes.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        loss = MaskedLoss()

        @tf.function(input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float32),
            tf.TensorSpec(shape=None, dtype=tf.float32),
        ])
        def evaluate(targets: tf.Tensor, predictions: tf.Tensor) -> tf.Tensor:
            """Evaluate masked loss with graph input ranks left dynamic.

            The surrounding tf.function signature prevents static shape checks from
            replacing the runtime assertion path.

            Args:
                targets (tf.Tensor): Float32 target tensor passed to the loss under test.
                predictions (tf.Tensor): Float32 prediction tensor whose shape must match
                    the target prefix.

            Returns:
                tf.Tensor: Masked loss result when shape assertions pass.

            Raises:
                tf.errors.InvalidArgumentError: Target/prediction ranks or dimensions are
                incompatible.
            """
            return loss.call(targets, predictions)

        self.assertEqual(evaluate(tf.ones((2, 3)), tf.zeros((2, 2))).shape, (2,))
        for target_shape, prediction_shape in (
            ((1, 3), (2, 2)),
            ((2, 1), (2, 2)),
            ((2,), (2, 2)),
        ):
            with self.subTest(target_shape=target_shape), self.assertRaises(tf.errors.InvalidArgumentError):
                evaluate(tf.ones(target_shape), tf.zeros(prediction_shape))


# Run this module's tests when executed directly.
if __name__ == "__main__":
    unittest.main()
