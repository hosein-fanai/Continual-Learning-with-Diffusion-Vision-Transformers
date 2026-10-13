"""Independent recurrence, sparse-gradient and recovery checks for TMCL AdamW."""

from __future__ import annotations

import unittest

import numpy as np
import tensorflow as tf

from semantic_consolidation.tmcl_optimizers import TMCLAdamW, TMCLSGD


def _reference(initial: np.ndarray, gradients: list[np.ndarray], rates: list[float], 
               weight_decay: float, beta_1: float, beta_2: float, epsilon: float, 
               amsgrad: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the AdamW recurrence in NumPy float64 without optimizer internals."""

    parameter = np.array(initial, copy=True, dtype=np.float64)
    first = np.zeros_like(parameter)
    second = np.zeros_like(parameter)
    maximum = np.zeros_like(parameter)
    for step, (gradient, rate) in enumerate(zip(gradients, rates), start=1):
        values = np.asarray(gradient, dtype=np.float64)
        parameter *= 1. - rate * weight_decay
        first = beta_1 * first + (1. - beta_1) * values
        second = beta_2 * second + (1. - beta_2) * values ** 2
        maximum = np.maximum(maximum, second)
        uncorrected = maximum if amsgrad else second
        corrected_first = first / (1. - beta_1 ** step)
        corrected_second = uncorrected / (1. - beta_2 ** step)
        parameter -= rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
    return parameter, first, second, maximum


class TMCLAdamWTests(unittest.TestCase):
    """Distinguish reference Adam epsilon from Keras epsilon hat without PyTorch."""

    def test_multistep_reference_includes_decay_and_changing_learning_rate(self) -> None:
        """Signed, zero and tiny gradients exercise moments and cosine-like rate changes."""

        initial = np.array([0.3, -0.7, 0.], dtype=np.float32)
        gradients = [np.array(values, dtype=np.float32) for values in [
            [0.4, -0.1, 1e-11], [0., 0.2, -3e-12], [-0.7, 0.8, 0.], 
            [0.01, -0.003, 1e-10], [0., 0., -1e-12]
        ]]
        rates = [0.003, 0.002, 0.001, 0.0005, 0.0001]
        for amsgrad in (False, True):
            with self.subTest(amsgrad=amsgrad):
                variable = tf.Variable(initial)
                optimizer = TMCLAdamW(
                    learning_rate=rates[0], weight_decay=0.2, beta_1=0.8, beta_2=0.7, 
                    epsilon=1e-8, amsgrad=amsgrad
                )
                for gradient, rate in zip(gradients, rates):
                    optimizer.learning_rate.assign(rate)
                    optimizer.apply_gradients([(tf.constant(gradient), variable)])
                expected, first, second, maximum = _reference(
                    initial, gradients, rates, 0.2, 0.8, 0.7, 1e-8, amsgrad
                )
                np.testing.assert_allclose(variable.numpy(), expected, rtol=3e-6, atol=3e-8)
                np.testing.assert_allclose(optimizer._momentums[0].numpy(), first, rtol=3e-6, atol=1e-15)
                np.testing.assert_allclose(optimizer._velocities[0].numpy(), second, rtol=3e-6, atol=1e-24)
                self.assertEqual(int(optimizer.iterations), len(gradients))
                # The native maximum slot exists only for the AMSGrad configuration.
                if amsgrad:
                    np.testing.assert_allclose(optimizer._velocity_hats[0].numpy(), maximum, rtol=3e-6, atol=1e-24)

    def test_tiny_gradient_detects_epsilon_hat_placement(self) -> None:
        """Equal scalar epsilon values still produce different native Keras updates."""

        reference_variable = tf.Variable([0.])
        keras_variable = tf.Variable([0.])
        gradient = tf.constant([1e-12])
        reference_optimizer = TMCLAdamW(learning_rate=0.001, weight_decay=0.)
        keras_optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0., epsilon=1e-8)
        reference_optimizer.apply_gradients([(gradient, reference_variable)])
        keras_optimizer.apply_gradients([(gradient, keras_variable)])
        expected = -0.001 * 1e-12 / (1e-12 + 1e-8)
        self.assertAlmostEqual(float(reference_variable[0]), expected, delta=1e-12)
        self.assertGreater(abs(float(reference_variable[0])), 10. * abs(float(keras_variable[0])))

    def test_native_weight_decay_exclusion_is_preserved(self) -> None:
        """A zero-gradient vector excluded by identity receives no decoupled decay."""

        decayed = tf.Variable([2., -3.])
        excluded = tf.Variable([2., -3.])
        optimizer = TMCLAdamW(learning_rate=0.1, weight_decay=0.2)
        optimizer.exclude_from_weight_decay(var_list=[excluded])
        optimizer.apply_gradients([(tf.zeros_like(decayed), decayed), (tf.zeros_like(excluded), excluded)])
        np.testing.assert_allclose(decayed.numpy(), [1.96, -2.94], rtol=1e-6)
        np.testing.assert_array_equal(excluded.numpy(), [2., -3.])

    def test_repeated_sparse_indices_match_summed_dense_gradients(self) -> None:
        """Squaring sparse values before summing would fail this cancellation fixture."""

        initial = np.array([[0.5, -0.2], [0.1, 0.4], [-0.8, 0.3]], dtype=np.float32)
        sparse_variable = tf.Variable(initial)
        dense_variable = tf.Variable(initial)
        sparse_optimizer = TMCLAdamW(learning_rate=0.02, weight_decay=0.01)
        dense_optimizer = TMCLAdamW(learning_rate=0.02, weight_decay=0.01)
        sparse_gradient = tf.IndexedSlices(
            values=tf.constant([[1., 2.], [-1., 3.], [0.5, -0.5]]), 
            indices=tf.constant([1, 1, 2]), dense_shape=tf.constant([3, 2])
        )
        dense_gradient = tf.constant([[0., 0.], [0., 5.], [0.5, -0.5]])
        for _ in range(3):
            sparse_optimizer.apply_gradients([(sparse_gradient, sparse_variable)])
            dense_optimizer.apply_gradients([(dense_gradient, dense_variable)])
        np.testing.assert_array_equal(sparse_variable.numpy(), dense_variable.numpy())
        for actual, expected in zip(sparse_optimizer.variables, dense_optimizer.variables):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_serialized_configuration_and_slots_resume_identical_updates(self) -> None:
        """Native reconstruction plus restored moment state continues both AMSGrad modes."""

        for amsgrad in (False, True):
            with self.subTest(amsgrad=amsgrad):
                variable = tf.Variable([0.3, -0.6])
                optimizer = TMCLAdamW(
                    learning_rate=0.007, weight_decay=0.02, beta_1=0.85, beta_2=0.97, 
                    epsilon=3e-8, amsgrad=amsgrad
                )
                for gradient in ([0.1, -0.2], [-0.4, 0.3], [1e-10, -1e-10]):
                    optimizer.apply_gradients([(tf.constant(gradient), variable)])
                restored = tf.keras.optimizers.deserialize(tf.keras.optimizers.serialize(optimizer))
                self.assertIsInstance(restored, TMCLAdamW)
                restored_variable = tf.Variable(variable.numpy())
                restored.build([restored_variable])
                restored.set_weights([value.numpy() for value in optimizer.variables])
                self.assertEqual(restored.get_config(), optimizer.get_config())
                for gradient in ([0.6, 0.2], [0., 0.], [-0.1, 0.4]):
                    optimizer.apply_gradients([(tf.constant(gradient), variable)])
                    restored.apply_gradients([(tf.constant(gradient), restored_variable)])
                np.testing.assert_array_equal(restored_variable.numpy(), variable.numpy())
                for actual, expected in zip(restored.variables, optimizer.variables):
                    np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_native_graph_updates_match_eager_reference(self) -> None:
        """Tensor iteration clocks remain valid when native training traces the update."""

        variable = tf.Variable([0.3, -0.2])
        optimizer = TMCLAdamW(learning_rate=0.001, weight_decay=0.01)
        optimizer.build([variable])

        @tf.function
        def update(gradient: tf.Tensor) -> tf.Tensor:
            """Apply one already-built optimizer step inside a traced function."""

            optimizer.apply_gradients([(gradient, variable)])
            return tf.identity(variable)

        gradients = [np.array([0.1, -0.2], dtype=np.float32), np.array([-0.3, 1e-12], dtype=np.float32)]
        for gradient in gradients:
            update(tf.constant(gradient))
        expected = _reference(np.array([0.3, -0.2], dtype=np.float32), gradients, [0.001] * 2, 0.01, 0.9, 0.999, 1e-8, False)[0]
        np.testing.assert_allclose(variable.numpy(), expected, rtol=3e-6, atol=3e-8)

    def test_unsupported_parameter_precision_fails_before_slot_creation(self) -> None:
        """The documented float32 route cannot silently use another moment precision."""

        optimizer = TMCLAdamW()
        with self.assertRaisesRegex(ValueError, "float32"):
            optimizer.build([tf.Variable([1.], dtype=tf.float64)])
        self.assertFalse(optimizer.built)


class TMCLSGDTests(unittest.TestCase):
    """Check scheduled-rate reference SGD and persistent unscaled momentum state."""

    def test_scheduled_rates_match_independent_numpy_recurrence(self) -> None:
        """A rate decrease exposes accidental storage of rate-scaled momentum."""

        gradients = [[0.5, -0.2], [0.1, 0.4], [-0.3, 0.], [0., -0.1]]
        rates = [0.4, 0.2, 0.04, 0.]
        for momentum in (0., 0.9):
            for nesterov in (False, True):
                with self.subTest(momentum=momentum, nesterov=nesterov):
                    variable = tf.Variable([0.3, -0.6])
                    optimizer = TMCLSGD(learning_rate=rates[0], momentum=momentum, nesterov=nesterov)
                    expected = np.asarray(variable.numpy(), dtype=np.float64)
                    buffer = np.zeros_like(expected)
                    for values, rate in zip(gradients, rates):
                        gradient = np.array(values, dtype=np.float64)
                        direction = gradient
                        # Reference momentum is independent of every learning rate.
                        if momentum != 0.:
                            buffer = momentum * buffer + gradient
                            direction = gradient + momentum * buffer if nesterov else buffer
                        expected -= rate * direction
                        optimizer.learning_rate.assign(rate)
                        optimizer.apply_gradients([(tf.constant(values), variable)])
                        np.testing.assert_allclose(variable.numpy(), expected, rtol=3e-6, atol=3e-8)
                    # A momentum-free ablation intentionally allocates no buffer.
                    if momentum != 0.:
                        np.testing.assert_allclose(optimizer.momentums[0].numpy(), buffer, rtol=3e-6, atol=3e-8)

    def test_learning_rate_change_differs_from_native_scaled_velocity(self) -> None:
        """The second hand-computed update must multiply the old buffer by the new rate."""

        reference_variable = tf.Variable([0.])
        native_variable = tf.Variable([0.])
        reference = TMCLSGD(learning_rate=0.1, momentum=0.9)
        native = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
        for rate in (0.1, 0.01):
            for optimizer, variable in ((reference, reference_variable), (native, native_variable)):
                optimizer.learning_rate.assign(rate)
                optimizer.apply_gradients([(tf.constant([1.]), variable)])
        self.assertAlmostEqual(float(reference_variable[0]), -0.119, places=6)
        self.assertAlmostEqual(float(native_variable[0]), -0.2, places=6)

    def test_serialized_momentum_state_resumes_after_rate_change(self) -> None:
        """Reconstruction keeps the buffer interpretation when cosine scheduling resumes."""

        for nesterov in (False, True):
            with self.subTest(nesterov=nesterov):
                variable = tf.Variable([0.5, -0.2])
                optimizer = TMCLSGD(learning_rate=0.4, momentum=0.9, nesterov=nesterov)
                optimizer.apply_gradients([(tf.constant([0.2, 0.3]), variable)])
                restored = tf.keras.optimizers.deserialize(tf.keras.optimizers.serialize(optimizer))
                self.assertIsInstance(restored, TMCLSGD)
                restored_variable = tf.Variable(variable.numpy())
                restored.build([restored_variable])
                restored.set_weights([value.numpy() for value in optimizer.variables])
                for rate, gradient in [(0.1, [-0.1, 0.4]), (0.03, [0.3, 0.]), (0., [1., 1.])]:
                    for active, parameter in ((optimizer, variable), (restored, restored_variable)):
                        active.learning_rate.assign(rate)
                        active.apply_gradients([(tf.constant(gradient), parameter)])
                np.testing.assert_array_equal(restored_variable.numpy(), variable.numpy())
                for actual, expected in zip(restored.variables, optimizer.variables):
                    np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_repeated_sparse_indices_share_dense_momentum_semantics(self) -> None:
        """A null-label embedding can aggregate repeated sparse rows before momentum."""

        sparse_variable = tf.Variable([[0.], [1.]])
        dense_variable = tf.Variable([[0.], [1.]])
        sparse = TMCLSGD(learning_rate=0.2, momentum=0.9)
        dense = TMCLSGD(learning_rate=0.2, momentum=0.9)
        gradient = tf.IndexedSlices(tf.constant([[2.], [-0.5]]), tf.constant([1, 1]), tf.constant([2, 1]))
        for rate in (0.2, 0.03):
            sparse.learning_rate.assign(rate)
            dense.learning_rate.assign(rate)
            sparse.apply_gradients([(gradient, sparse_variable)])
            dense.apply_gradients([(tf.constant([[0.], [1.5]]), dense_variable)])
        np.testing.assert_array_equal(sparse_variable.numpy(), dense_variable.numpy())

    def test_nonzero_decay_and_nonfloat32_state_are_explicitly_unsupported(self) -> None:
        """The probe cannot silently enable native decoupled SGD or mixed precision."""

        with self.assertRaisesRegex(ValueError, "zero weight decay"):
            TMCLSGD(weight_decay=0.01)
        with self.assertRaisesRegex(ValueError, "float32"):
            TMCLSGD().build([tf.Variable([1.], dtype=tf.float64)])


# Direct invocation is opt-in; importing a test module performs no computation.
if __name__ == "__main__":
    unittest.main()
