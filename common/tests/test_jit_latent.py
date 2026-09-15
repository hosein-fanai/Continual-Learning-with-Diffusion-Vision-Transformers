"""Check diffusion bottleneck sampling under actual XLA compilation."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from autoencoder.variational_autoencoder import _GaussianSampling
from diffusion.layers.convolution.variational_reshaper import VariationalReshaper
from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer


class JitLatentTests(unittest.TestCase):
    """Preserve advancing, reproducible latent draws and saved RNG state."""

    def tearDown(self):
        tf.keras.backend.clear_session()

    def test_multiple_bottlenecks_and_blocks_have_unique_weight_paths(self):
        configurations = (
            dict(depth=4, vit_block_ids=[],
                 reshaper_ids_dict={1: "flatten", 2: "unflatten", 3: "flatten", 4: "unflatten"},
                 reshaper_kwargs={"add_kl": True, "latent_dim_ratio": [0.5, 0.5]}),
            dict(depth=2),
        )
        for configuration in configurations:
            with self.subTest(configuration=configuration):
                model = DiffusionTransformer(
                    image_size=4, channels=1, patch_size=2, dim=4,
                    mha_num_heads=1, num_classes=2, timesteps=4,
                    seed=43, name="multi_stream_transformer", **configuration,
                )
                paths = [weight.path for weight in model.weights]
                self.assertGreaterEqual(sum("seed_state" in path for path in paths), 2)
                self.assertEqual(len(paths), len(set(paths)))

    @staticmethod
    def _fixture(kind, dtype="float32"):
        if kind == "convolution":
            owner = VariationalReshaper(
                "flatten", (2, 2, 2), add_kl=True, latent_dim_ratio=0.5,
                seed=43, dtype=dtype, name="latent_reshaper",
            )
            reshaper = owner
            shape = (2, 2, 2, 2)
        else:
            owner = DiffusionTransformer(
                image_size=4, channels=1, patch_size=2, dim=4, depth=2,
                mha_num_heads=1, num_classes=2, timesteps=4, vit_block_ids=[],
                reshaper_ids_dict={1: "flatten", 2: "unflatten"},
                reshaper_kwargs={"add_kl": True, "latent_dim_ratio": [0.5]},
                seed=43, dtype=dtype, name="latent_transformer",
            )
            reshaper = owner.layers_dicts[0][owner.R]
            shape = (2, 4, 4)
        inputs = tf.zeros(shape, dtype=owner.compute_dtype)
        sampler = next(layer for layer in reshaper.layers
                       if isinstance(layer, _GaussianSampling))
        return owner, reshaper, sampler, inputs

    def test_xla_draws_advance_and_reseed_under_each_numeric_policy(self):
        for kind in ("convolution", "transformer"):
            for dtype in ("float32", "float64", "mixed_float16"):
                with self.subTest(kind=kind, dtype=dtype):
                    owner, reshaper, sampler, inputs = self._fixture(kind, dtype)
                    draw = tf.function(reshaper, jit_compile=True)
                    sampler.reset_seed(137)
                    first = draw(inputs)[0].numpy()
                    second = draw(inputs)[0].numpy()
                    self.assertFalse(np.array_equal(first, second))
                    self.assertTrue(np.isfinite(first).all())
                    self.assertEqual(first.dtype, np.dtype(owner.compute_dtype))
                    sampler.reset_seed(137)
                    np.testing.assert_array_equal(draw(inputs)[0].numpy(), first)

    def test_xla_sampling_restores_the_next_draw_from_model_weights(self):
        for kind in ("convolution", "transformer"):
            with self.subTest(kind=kind):
                owner, reshaper, _, inputs = self._fixture(kind)
                draw = tf.function(reshaper, jit_compile=True)
                draw(inputs)
                weights = owner.get_weights()
                expected = draw(inputs)[0].numpy()
                draw(inputs)
                owner.set_weights(weights)
                np.testing.assert_array_equal(draw(inputs)[0].numpy(), expected)

    def test_xla_sampling_preserves_gradients_to_gaussian_parameters(self):
        for kind in ("convolution", "transformer"):
            with self.subTest(kind=kind):
                _, reshaper, _, inputs = self._fixture(kind)

                @tf.function(jit_compile=True)
                def gradients(value):
                    with tf.GradientTape() as tape:
                        sample = reshaper(value, training=True)[0]
                        loss = tf.reduce_sum(tf.square(sample))
                    return tape.gradient(loss, reshaper.trainable_variables)

                values = gradients(inputs)
                self.assertTrue(all(value is not None for value in values))
                self.assertTrue(all(np.isfinite(value.numpy()).all() for value in values))
                self.assertTrue(any(np.any(value.numpy() != 0) for value in values))

    def test_keras_round_trip_preserves_sampler_and_next_xla_draw(self):
        for kind in ("convolution", "transformer"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                owner, reshaper, sampler, inputs = self._fixture(kind)
                draw = tf.function(reshaper, jit_compile=True)
                draw(inputs)
                path = Path(directory) / "latent.keras"
                owner.save(path)
                expected = draw(inputs)[0].numpy()
                # The parent transformer has no serialization decorator; the new
                # sampling layer resolves through its own canonical registration.
                restored = tf.keras.models.load_model(
                    path, compile=False,
                    custom_objects={"DiffusionTransformer": DiffusionTransformer},
                )
                restored_reshaper = (restored if kind == "convolution"
                                     else restored.layers_dicts[0][restored.R])
                restored_sampler = restored_reshaper.get_layer(sampler.name)
                self.assertIsInstance(restored_sampler, _GaussianSampling)
                self.assertEqual(restored_sampler.dtype_policy.name,
                                 sampler.dtype_policy.name)
                restored_draw = tf.function(restored_reshaper, jit_compile=True)
                np.testing.assert_array_equal(restored_draw(inputs)[0].numpy(), expected)


if __name__ == "__main__":
    unittest.main()
