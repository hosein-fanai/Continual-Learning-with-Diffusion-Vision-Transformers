"""Numerical checks for Keras 3 optimizer and loss-scaling bridges."""

import unittest
import numpy as np
import tensorflow as tf

from common.keras_compat import register_optimizer_variables, format_variable_name
from common.gradients import apply_policy_gradients


class Keras3CompatibilityTests(unittest.TestCase):
    """Check numerical behavior, state retention, and readable variable paths."""

    def test_adam_extension_preserves_old_slots_and_next_update(self) -> None:
        """Reordered and newly added variables retain the old Adam trajectory.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """
        original = tf.Variable([1., 2.], name="original")
        control = tf.Variable([1., 2.], name="control")
        new = tf.Variable([3.], name="new")
        optimizer = tf.keras.optimizers.Adam(.01)
        reference = tf.keras.optimizers.Adam(.01)
        gradient = tf.constant([.4, -.3])
        for _ in range(3):
            optimizer.apply_gradients([(gradient, original)])
            reference.apply_gradients([(gradient, control)])
        replacement = register_optimizer_variables(optimizer, [new, original])
        self.assertIsNot(replacement, optimizer)
        self.assertEqual(int(replacement.iterations.numpy()), 3)
        replacement.apply_gradients([(tf.ones_like(new), new), (gradient, original)])
        reference.apply_gradients([(gradient, control)])
        np.testing.assert_allclose(original.numpy(), control.numpy(), rtol=1e-7)
        self.assertLess(float(new.numpy()[0]), 3.)
        self.assertIs(register_optimizer_variables(replacement, [original]), replacement)

    def test_loss_scale_extension_preserves_inner_slots_and_scale(self) -> None:
        """Preserve dynamic scaling and inner moments across optimizer rebuilds.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """
        value = tf.Variable(2., name="scaled_value")
        extra = tf.Variable(3., name="scaled_extra")
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            tf.keras.optimizers.Adam(.01), initial_scale=128.)
        with tf.GradientTape() as tape:
            loss = value ** 2
        apply_policy_gradients(tape, optimizer, loss, [value])
        replacement = register_optimizer_variables(optimizer, [value, extra])
        self.assertEqual(int(replacement.inner_optimizer.iterations.numpy()), 1)
        self.assertEqual(float(replacement.dynamic_scale.numpy()), 128.)
        self.assertEqual(int(replacement.step_counter.numpy()), 1)
        old_moment = optimizer.inner_optimizer._momentums[0].numpy()
        np.testing.assert_array_equal(replacement.inner_optimizer._momentums[0].numpy(), old_moment)

    def test_nonfinite_scaled_update_is_skipped(self) -> None:
        """An infinite gradient lowers the scale without changing weights.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """
        variable = tf.Variable(2.)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            tf.keras.optimizers.SGD(.1), initial_scale=128.)
        with tf.GradientTape() as tape:
            loss = variable * tf.constant(float("inf"))
        apply_policy_gradients(tape, optimizer, loss, [variable])
        self.assertEqual(float(variable.numpy()), 2.)
        self.assertEqual(int(optimizer.inner_optimizer.iterations.numpy()), 0)
        self.assertEqual(float(optimizer.dynamic_scale.numpy()), 64.)

    def test_sparse_scaled_update_matches_unscaled_sgd(self) -> None:
        """Unscale a sparse embedding gradient without densification or drift.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """
        variable = tf.Variable([[2.], [3.]])
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            tf.keras.optimizers.SGD(.1), initial_scale=128.)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(tf.gather(variable, [1]) ** 2)
        pairs = apply_policy_gradients(tape, optimizer, loss, [variable])
        self.assertIsInstance(pairs[0][0], tf.IndexedSlices)
        np.testing.assert_allclose(pairs[0][0].values.numpy(), [[6.]])
        np.testing.assert_allclose(variable.numpy(), [[2.], [2.4]])

    def test_variable_display_retains_parent_path(self) -> None:
        """The displayed kernel name includes its automatically tracked parent.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """
        layer = tf.keras.layers.Dense(2, name="parent__child")
        layer(tf.ones((1, 3)))
        self.assertEqual(format_variable_name(layer.kernel), "parent__child__kernel")

    def test_task_reset_replays_keras_dropout_stream(self) -> None:
        """Task resets restore Keras-owned stochastic state after extra draws.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """
        from common.learner import _reset_task_random_streams
        model = tf.keras.Sequential([tf.keras.layers.Dropout(.5, seed=7)])
        inputs = tf.ones((8, 16))
        _reset_task_random_streams(model, 31)
        first = model(inputs, training=True).numpy()
        model(inputs, training=True)
        _reset_task_random_streams(model, 31)
        np.testing.assert_array_equal(model(inputs, training=True).numpy(), first)

    def test_vae_task_reset_updates_live_sampling_seed(self) -> None:
        """A VAE task seed controls the serializable graph's actual sampler.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """
        from common.learner import _reset_task_random_streams
        from autoencoder import VariationalAutoencoder
        model = VariationalAutoencoder(data_dim=4, latent_dim=2, hiddens_dims=(), seed=7)
        _reset_task_random_streams(model, 31)
        self.assertEqual(model.encoder.get_layer("z_sample").seed,
                         model.reparameterization_seed)


# Run the focused compatibility regressions when invoked as a script.
if __name__ == "__main__":
    unittest.main()
