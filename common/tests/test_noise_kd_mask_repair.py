"""Independent float64 value/gradient oracles for per-teacher noise-KD masks."""

import types
import unittest

import numpy as np
import tensorflow as tf
from keras.src import backend

from common.keras_compat import compute_compiled_loss
from common.masked_loss import MaskedLoss
from diffusion.models.wrapper.diffusion_model import DiffusionModel


class _RowHuber(tf.keras.losses.Loss):
    """Expose Huber losses after all non-batch dimensions have been averaged."""

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return one delta-one Huber value for each example."""

        residual = tf.abs(y_pred - y_true)
        pixels = tf.where(residual <= 1., .5 * residual ** 2, residual - .5)
        return tf.reduce_mean(pixels, axis=tf.range(1, tf.rank(pixels)))


class _CountingRowMSE(tf.keras.losses.Loss):
    """Record execution count to rule out probing a custom loss twice."""

    def __init__(self) -> None:
        """Create a stable-dtype loss and an independent execution counter."""

        super().__init__(dtype="float64", name="counting_row_mse")
        self.calls = tf.Variable(0, trainable=False)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Increment the counter once and compute row MSE."""

        self.calls.assign_add(1)
        return tf.reduce_mean(tf.square(y_pred - y_true), axis=[1, 2, 3])


class _MaskedPixelMSE(tf.keras.losses.Loss):
    """Supply an output mask distinct from the incoming prediction mask."""

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return pixel MSE with only the first row eligible in its output mask."""

        values = tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)
        backend.set_keras_mask(values, tf.constant([
            [[True, True], [True, True]], [[False, False], [False, False]]
        ]))
        return values


class _ScalarMSE(tf.keras.losses.Loss):
    """Represent a custom objective that has already aggregated the batch."""

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Reduce all squared residuals to a single scalar."""

        return tf.reduce_mean(tf.square(y_pred - y_true))


class _UnregisteredSlottedMSE(tf.keras.losses.Loss):
    """Keep custom slot/callable state without relying on serialization or __copy__."""

    __slots__ = ("factor", "optional_state")

    def __init__(self) -> None:
        """Retain float32-default configuration and independently observable call state."""

        super().__init__(name="unregistered_slotted_mse")
        self.factor = 3.
        self.calls = tf.Variable(0, trainable=False)

    def __copy__(self) -> object:
        """Reject application-level copying so the dtype bridge must preserve raw state."""

        raise RuntimeError("A custom clone would not preserve the live loss state.")

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute a scaled per-row loss while retaining the shared execution counter."""

        self.calls.assign_add(1)
        return self.factor * tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)


class NoiseKDMaskRepairTests(unittest.TestCase):
    """Test the actual diffusion loss helper with compiled loss containers."""

    def tearDown(self) -> None:
        """Release isolated Keras model state after each regression."""

        tf.keras.backend.clear_session()

    def _model(self, loss: object, loss_weights: float | None = None) -> tf.keras.Model:
        """Use the production helpers without allocating a diffusion network."""

        model = tf.keras.Model(dtype="float64")
        model.compile(optimizer="sgd", loss=loss, loss_weights=loss_weights)
        model._compute_base_loss = types.MethodType(DiffusionModel._compute_base_loss, model)
        return model

    def test_native_and_custom_losses_match_selected_row_values_and_gradients(self) -> None:
        """Pixel and row MSE, MAE and Huber agree with hand-computed weighted means."""

        cases = [
            ("mse", [1., 100.], [2., 20.]), 
            (MaskedLoss("mse"), [1., 100.], [2., 20.]), 
            ("mae", [1., 10.], [1., 1.]), 
            (MaskedLoss("mae"), [1., 10.], [1., 1.]), 
            (tf.keras.losses.Huber(), [.5, 9.5], [1., 1.]), 
            (_RowHuber(dtype="float64", name="row_huber"), [.5, 9.5], [1., 1.])
        ]
        for configured_loss, row_values, row_derivatives in cases:
            model = self._model(configured_loss)

            def evaluate(mask: tf.Tensor) -> tuple:
                """Differentiate one real masked objective against both source tensors."""

                student = tf.reshape(tf.repeat(tf.constant([1., 10.], tf.float64), 4), [2, 2, 2, 1])
                teacher = tf.zeros_like(student)
                with tf.GradientTape() as tape:
                    tape.watch([student, teacher])
                    result = DiffusionModel._compute_single_teacher_noise_loss(model, teacher, student, mask)
                student_gradient, teacher_gradient = tape.gradient(result, [student, teacher])
                return result, student_gradient, teacher_gradient

            for graph in (False, True):
                evaluator = tf.function(evaluate) if graph else evaluate
                for mask in ([1., 0.], [0., 1.], [1., 1.], [0., 0.], [.25, .75]):
                    with self.subTest(loss=str(configured_loss), graph=graph, mask=mask):
                        weights = np.asarray(mask, dtype=np.float64)
                        total = weights.sum()
                        weights = weights / total if total else weights
                        expected = np.dot(weights, row_values)
                        derivatives = weights * np.asarray(row_derivatives) / 4.
                        expected_gradient = np.repeat(derivatives, 4).reshape(2, 2, 2, 1)
                        value, gradient, teacher_gradient = evaluator(tf.constant(mask, tf.float64))
                        self.assertEqual(value.dtype, tf.float64)
                        np.testing.assert_allclose(value.numpy(), expected, rtol=1e-12, atol=1e-12)
                        np.testing.assert_allclose(gradient.numpy(), expected_gradient, rtol=1e-12, atol=1e-12)
                        self.assertIsNone(teacher_gradient)

    def test_configured_reductions_and_compile_coefficients_remain_effective(self) -> None:
        """Broadcast weights fully for weighted-mean denominators and retain sum/none semantics."""

        student = tf.reshape(tf.repeat(tf.constant([1., 10.], tf.float64), 4), [2, 2, 2, 1])
        for reduction, expected in (
            ("sum_over_batch_size", 1.), ("mean", 1.), 
            ("mean_with_sample_weight", 1.), ("sum", 8.), 
            ("none", np.asarray([[[2., 2.], [2., 2.]], [[0., 0.], [0., 0.]]]))
        ):
            with self.subTest(reduction=reduction):
                model = self._model(tf.keras.losses.MeanSquaredError(reduction=reduction), loss_weights=.3)
                value = DiffusionModel._compute_single_teacher_noise_loss(
                    model, tf.zeros_like(student), student, tf.constant([1., 0.], tf.float64)
                )
                np.testing.assert_allclose(value.numpy(), np.asarray(expected) * .3, rtol=1e-12, atol=1e-12)

    def test_each_custom_loss_executes_once_in_eager_and_graph_mode(self) -> None:
        """Determine weight rank from the one actual loss result without an extra probe."""

        loss = _CountingRowMSE()
        model = self._model(loss)
        student = tf.ones([2, 2, 2, 1], tf.float64)

        def evaluate() -> tf.Tensor:
            """Run the selected-row compiled-loss bridge exactly once."""

            return DiffusionModel._compute_single_teacher_noise_loss(
                model, tf.zeros_like(student), student, tf.constant([1., 0.], tf.float64)
            )

        self.assertEqual(float(evaluate()), 1.)
        self.assertEqual(int(loss.calls), 1)
        self.assertEqual(float(tf.function(evaluate)()), 1.)
        self.assertEqual(int(loss.calls), 2)

    def test_input_and_output_masks_are_intersected_before_keras_reduction(self) -> None:
        """The sole pixel accepted by both Keras masks supplies all loss and gradient."""

        model = self._model(_MaskedPixelMSE(dtype="float64", name="masked_pixel_mse"))
        student = tf.reshape(tf.repeat(tf.constant([1., 10.], tf.float64), 4), [2, 2, 2, 1])
        for graph in (False, True):
            with self.subTest(graph=graph):

                def evaluate(values: tf.Tensor) -> tuple:
                    """Attach a prediction mask and differentiate the intersected masked loss."""

                    backend.set_keras_mask(values, tf.constant([
                        [[True, False], [False, False]], [[True, True], [True, True]]
                    ]))
                    with tf.GradientTape() as tape:
                        tape.watch(values)
                        result = compute_compiled_loss(
                            model, tf.zeros_like(values), values, 
                            sample_weight=tf.ones(2, tf.float64), sample_weight_by_batch=True
                        )
                    return result, tape.gradient(result, values)

                evaluator = tf.function(evaluate) if graph else evaluate
                value, gradient = evaluator(student)
                expected_gradient = np.zeros((2, 2, 2, 1))
                expected_gradient[0, 0, 0, 0] = 2.
                np.testing.assert_allclose(value.numpy(), 1., rtol=1e-6)
                np.testing.assert_allclose(gradient.numpy(), expected_gradient, rtol=1e-6)

    def test_scalar_custom_loss_is_rejected_only_when_row_weights_are_requested(self) -> None:
        """An aggregate cannot silently fabricate per-example masking semantics."""

        model = self._model(_ScalarMSE(dtype="float64", name="scalar_mse"))
        student = tf.ones([2, 2, 2, 1], tf.float64)
        self.assertEqual(float(compute_compiled_loss(model, tf.zeros_like(student), student)), 1.)
        for graph in (False, True):
            with self.subTest(graph=graph):

                def evaluate() -> tf.Tensor:
                    """Request a row mask for an incompatible scalar custom loss."""

                    return DiffusionModel._compute_single_teacher_noise_loss(
                        model, tf.zeros_like(student), student, tf.constant([1., 0.], tf.float64)
                    )

                evaluator = tf.function(evaluate) if graph else evaluate
                with self.assertRaisesRegex(ValueError, "leading batch dimension"):
                    evaluator()

    def test_default_compiled_loss_and_regularizers_are_unchanged(self) -> None:
        """Ordinary broadcasting retains the compiled Keras value and adds penalties once."""

        configured = MaskedLoss("mse")
        model = self._model(configured)
        student = tf.reshape(tf.range(8, dtype=tf.float64), [2, 2, 2, 1])
        expected = float(np.mean(np.square(np.arange(4))))
        value = compute_compiled_loss(
            model, tf.zeros_like(student), student, sample_weight=tf.constant([2., 0.], tf.float64), 
            regularization_losses=[tf.constant(.125, tf.float64)]
        )
        self.assertEqual(float(value), expected + .125)
        self.assertEqual(configured.dtype, "float32")
        mixed_inputs = compute_compiled_loss(
            model, tf.zeros_like(student, dtype=tf.float32), tf.cast(student, tf.float32), 
            sample_weight=tf.constant([2, 0], tf.int32), sample_weight_by_batch=True
        )
        self.assertEqual(mixed_inputs.dtype, tf.float64)
        self.assertEqual(float(mixed_inputs), expected)


    def test_unregistered_custom_losses_align_precision_without_serialization(self) -> None:
        """Default float32 loss objects preserve live state and float64 residual precision."""

        def unregistered_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            """Retain a local unregistered function object in the resolved loss wrapper."""

            return tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)

        delta = 2. ** -30
        true = tf.constant([[1.], [1.]], tf.float64)
        prediction = true + tf.constant([[delta], [2. * delta]], tf.float64)
        custom = _UnregisteredSlottedMSE()
        for configured, factor in ((unregistered_mse, 1.), (custom, 3.)):
            for row_weights in (False, True):
                with self.subTest(custom=isinstance(configured, tf.keras.losses.Loss), row_weights=row_weights):
                    model = self._model(configured)
                    for graph in (False, True):

                        def evaluate() -> tf.Tensor:
                            """Use a residual lost by any intermediate float32 conversion."""

                            return compute_compiled_loss(
                                model, true, prediction, sample_weight=tf.constant([2., 0.], tf.float64), 
                                sample_weight_by_batch=row_weights
                            )

                        evaluator = tf.function(evaluate) if graph else evaluate
                        value = evaluator()
                        self.assertEqual(value.dtype, tf.float64)
                        self.assertEqual(float(value), factor * delta ** 2)
                    resolved = model._compile_loss._flat_losses[0].loss
                    # Custom object storage is independent while live state remains shared.
                    if configured is custom:
                        self.assertIsNot(resolved, custom)
                        self.assertIs(resolved.calls, custom.calls)
                        self.assertEqual(resolved.factor, custom.factor)
                        self.assertFalse(hasattr(resolved, "optional_state"))
                        self.assertEqual(custom.dtype, "float32")
                    # Function wrappers retain the exact local callable without registration.
                    else:
                        self.assertIs(resolved.fn, unregistered_mse)
        self.assertEqual(int(custom.calls), 4)


# Direct execution runs these focused mathematical regressions.
if __name__ == "__main__":
    unittest.main()
