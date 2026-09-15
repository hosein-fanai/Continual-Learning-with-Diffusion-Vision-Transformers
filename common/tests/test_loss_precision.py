"""Check residual precision and native weighting in custom training losses."""

from __future__ import annotations

import unittest

import numpy as np
import tensorflow as tf

from autoencoder import VAEClassifier, VariationalAutoencoder
from common.keras_compat import compute_compiled_loss
from common.masked_loss import MaskedLoss
from common.runtime import configure_runtime
from common.tests.test_dtype_models import _make_dit_network
from diffusion import DiffusionClassifier


class LossPrecisionTests(unittest.TestCase):
    """Compare tiny float64 residuals with independent NumPy calculations."""

    def setUp(self) -> None:
        """Select float64 for the small models created by each test.

        Returns:
            result (None): The test process uses float64 model policies.
        """
        tf.keras.backend.clear_session()
        configure_runtime(71, "float64")

    def tearDown(self) -> None:
        """Restore the ordinary policy after precision-sensitive checks.

        Returns:
            result (None): The test process uses float32 again.
        """
        tf.keras.backend.clear_session()
        configure_runtime(71, "float32")

    def test_diffusion_loss_preserves_weighted_residuals_and_gradients(self) -> None:
        """Compare native aliases and Loss objects in eager and traced execution.

        Returns:
            result (None): Unittest checks precision, reduction and gradients.
        """
        model = DiffusionClassifier(network=_make_dit_network(), use_ema=False,
                                    test_network_name="raw", test_steps=2, seed=71)
        values = np.array([[1.000000001, 1.000000002],
                           [1.000000003, 1.000000004]], dtype=np.float64)
        delta = values - 1.
        weights = np.array([1., 3.], dtype=np.float64)
        for specification, divisor in (("mse", 2.),
                                        (tf.keras.losses.MeanSquaredError(reduction="sum"), 1.),
                                        (MaskedLoss("mse"), 2.)):
            for traced in (False, True):
                with self.subTest(loss=str(specification), traced=traced):
                    model.compile(optimizer="adam", loss=specification, loss_weights=.25)
                    loss_fn = tf.function(model._compute_base_loss) if traced else model._compute_base_loss
                    prediction = tf.Variable(values)
                    with tf.GradientTape() as tape:
                        loss = loss_fn(tf.ones_like(prediction), prediction, tf.constant(weights))
                    gradient = tape.gradient(loss, prediction)
                    expected = .25 * np.sum(np.mean(delta ** 2, axis=1) * weights) / divisor
                    np.testing.assert_allclose(loss.numpy(), expected, rtol=1e-12)
                    np.testing.assert_allclose(gradient.numpy(),
                                               .25 * delta * weights[:, None] / divisor,
                                               rtol=1e-12)
                    self.assertEqual(loss.dtype, tf.float64)
                    self.assertGreater(float(loss), 0.)
                    # A caller may share this loss with a different model policy.
                    if isinstance(specification, tf.keras.losses.Loss):
                        self.assertEqual(specification.dtype, "float32")

    def test_vae_training_losses_preserve_reconstruction_precision(self) -> None:
        """Exercise both native VAE test steps with an exactly constant decoder.

        Returns:
            result (None): Both reconstruction trackers match weighted NumPy MSE.
        """
        values = np.array([[1.000000001, 1.000000002],
                           [1.000000003, 1.000000004]], dtype=np.float64)
        expected = np.mean(np.mean((values - 1.) ** 2, axis=1) * [.5, 1.5])
        for classifier in (False, True):
            with self.subTest(classifier=classifier):
                kwargs = dict(data_dim=2, latent_dim=1, hiddens_dims=(2,),
                              last_activation=None, beta=0., compile=False)
                # The joint variant fixes conditioning and uses a separate classifier.
                if classifier:
                    network = tf.keras.Sequential([tf.keras.layers.Input((2,)),
                                                   tf.keras.layers.Dense(2, activation="softmax")])
                    model = VAEClassifier(class_num=2, classifier=network, alpha=0., **kwargs)
                # The plain VAE consumes the same features without label conditioning.
                else:
                    model = VariationalAutoencoder(conditioned=False, **kwargs)
                model.compile(optimizer="adam", loss="mse", run_eagerly=True)
                for variable in model.decoder.trainable_variables:
                    variable.assign(tf.zeros_like(variable))
                model.decoder.layers[-1].bias.assign(tf.ones_like(model.decoder.layers[-1].bias))
                labels = tf.one_hot([0, 1], 2, dtype=tf.float64)
                result = model.test_step((tf.constant(values), labels, tf.constant([1., 3.], tf.float64)))
                np.testing.assert_allclose(result["recon_loss"].numpy(), expected, rtol=1e-12)

    def test_loss_weights_and_regularizers_remain_native(self) -> None:
        """Preserve the loss weight and sum regularizers once in stable precision.

        Returns:
            result (None): The combined loss matches its weighted analytical value.
        """
        model = tf.keras.Model(dtype="float64")
        shared = tf.keras.losses.MeanSquaredError()
        model.compile(loss=shared, loss_weights=.25)
        first = tf.constant([[1.000000001]], tf.float64)
        penalty = tf.constant([1.e-18, 2.e-18], tf.float64)
        result = compute_compiled_loss(model, tf.ones_like(first), first,
                                       regularization_losses=[penalty])
        expected = .25 * (first.numpy() - 1.) ** 2 + 3.e-18
        np.testing.assert_allclose(result.numpy(), expected.item(), rtol=1e-12)
        self.assertEqual(shared.dtype, "float32")


# Run directly as well as through unittest discovery.
if __name__ == "__main__":
    unittest.main()
