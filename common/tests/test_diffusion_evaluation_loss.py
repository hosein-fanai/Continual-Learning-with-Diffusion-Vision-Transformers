"""Fixed evaluation-loss contracts for studies comparing training losses."""

import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer
from diffusion.models.wrapper.diffusion_model import DiffusionModel


class DiffusionEvaluationLossTests(unittest.TestCase):
    """Validation can use MSE without replacing the compiled gradient objective."""

    def tearDown(self) -> None:
        """Release the small model and optimizer state created by each case."""

        tf.keras.backend.clear_session()

    def _model(self, dtype: str = "float32") -> DiffusionModel:
        """Construct a tiny teacher-free wrapper with no EMA or random data reads."""

        return DiffusionModel(
            network=DiffusionTransformer(
                image_size=4, channels=1, patch_size=2, dim=4, depth=1, 
                mha_num_heads=1, vit_block_mlp_ratio=1., num_classes=2, 
                timesteps=8, use_cfg=False, seed=17, dtype=dtype
            ), 
            use_ema=False, test_steps=4, preprocess_type=None, p_uncond=0., 
            seed=17, dtype=dtype
        )

    def test_optional_evaluation_loss_preserves_training_gradients(self) -> None:
        """A known residual has MAE gradient one while validation reports MSE four."""

        model = self._model()
        model.compile(loss="mae", evaluation_loss="mse", optimizer="sgd", jit_compile=False)
        target = tf.zeros([2, 4, 4, 1])
        prediction = tf.Variable(tf.ones_like(target) * 2.)
        with tf.GradientTape() as tape:
            training_loss = model._compute_base_loss(target, prediction)
        gradient = tape.gradient(training_loss, prediction)
        self.assertAlmostEqual(float(training_loss), 2.)
        self.assertAlmostEqual(float(tf.reduce_sum(gradient)), 1.)
        self.assertAlmostEqual(float(model._compute_base_loss(target, prediction, evaluation=True)), 4.)

    def test_forward_route_changes_only_evaluation_noise_and_image_units(self) -> None:
        """The real forward loss dispatcher carries its training flag to both metrics."""

        model = self._model()
        model.compile(loss="mae", evaluation_loss="mse", optimizer="sgd", jit_compile=False)
        zero = tf.zeros([2, 4, 4, 1])
        labels = tf.constant([0, 1])
        predictions = (tf.ones_like(zero) * 3., tf.ones_like(zero) * 2., ([], []), ([], []))
        with patch.object(model, "forward", return_value=predictions):
            for training, noise, image in ((True, 2., 3.), (False, 4., 9.)):
                with self.subTest(training=training):
                    result = model.forward_and_compute_loss(
                        "raw", zero, zero, labels, zero, labels, labels, labels, 1., 
                        use_image_loss=True, training=training
                    )
                    self.assertAlmostEqual(float(result[1]), noise)
                    self.assertAlmostEqual(float(result[5]), image)

    def test_default_and_recompile_restore_compiled_loss_behavior(self) -> None:
        """None keeps legacy validation behavior and clears a prior evaluation override."""

        model = self._model()
        target = tf.zeros([2, 4, 4, 1])
        prediction = tf.ones_like(target) * 2.
        model.compile(loss="mae", optimizer="sgd", jit_compile=False)
        self.assertAlmostEqual(float(model._compute_base_loss(target, prediction, evaluation=True)), 2.)
        self.assertNotIn("evaluation_loss", model.get_compile_config())
        model.compile(loss="mae", evaluation_loss="mse", optimizer="sgd", jit_compile=False)
        self.assertEqual(model.get_compile_config()["evaluation_loss"], "mse")
        model.compile(loss="mae", optimizer="sgd", jit_compile=False)
        self.assertAlmostEqual(float(model._compute_base_loss(target, prediction, evaluation=True)), 2.)
        self.assertNotIn("evaluation_loss", model.get_compile_config())

    def test_compile_config_round_trip_keeps_evaluation_choice(self) -> None:
        """Recovery can reconstruct both training and validation loss selections."""

        original = self._model()
        original.compile(loss="mae", evaluation_loss="mse", optimizer="sgd", jit_compile=False)
        recovered = self._model()
        recovered.compile(**tf.keras.utils.deserialize_keras_object(original.get_compile_config()))
        self.assertEqual(recovered.get_compile_config()["loss"], "mae")
        self.assertEqual(recovered.get_compile_config()["evaluation_loss"], "mse")

    def test_evaluation_loss_preserves_float64_precision(self) -> None:
        """The alternate Keras loss follows the wrapper's stable variable precision."""

        model = self._model(dtype="float64")
        model.compile(loss="mae", evaluation_loss="mse", optimizer="sgd", jit_compile=False)
        target = tf.zeros([2, 4, 4, 1], dtype=tf.float64)
        prediction = tf.ones_like(target) * tf.constant(1.0000000001, dtype=tf.float64)
        value = model._compute_base_loss(target, prediction, evaluation=True)
        self.assertEqual(value.dtype, tf.float64)
        self.assertAlmostEqual(float(value), 1.0000000002, places=13)

    def test_split_losses_share_evaluation_units(self) -> None:
        """Optional conditional and null diagnostics use the same selected loss."""

        model = self._model()
        model.compile(loss="mae", evaluation_loss="mse", optimizer="sgd", jit_compile=False)
        target = tf.zeros([2, 4, 4, 1])
        prediction = tf.ones_like(target) * 2.
        conditional, unconditional = model.compute_separate_noise_losses(
            target, prediction, tf.constant([0, 1]), evaluation=True
        )
        self.assertAlmostEqual(float(conditional), 4.)
        self.assertTrue(np.isfinite(float(unconditional)))
        self.assertEqual(float(unconditional), 0.)


# Run only when this focused regression module is invoked directly.
if __name__ == "__main__":
    unittest.main()
