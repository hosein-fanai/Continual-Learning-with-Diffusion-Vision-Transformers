"""Small XLA training and checkpointed stochastic-stream regressions."""

import os
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from common.random import SeedStream
from diffusion.layers.drop_path import DropPath
from diffusion.layers.embedding.patch_embedding import PatchEmbedding
from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer
from diffusion.models.wrapper.diffusion_model import DiffusionModel


def make_wrapper(**kwargs):
    network = DiffusionTransformer(
        num_classes=2, use_cfg=True, timesteps=4, image_size=4,
        channels=1, patch_size=2, dim=4, depth=0,
        mha_num_heads=1, vit_block_mlp_ratio=1.,
    )
    return DiffusionModel(
        network, scheduler_name="linear", test_steps=2, seed=17, **kwargs,
    )


class DiffusionJitTests(unittest.TestCase):
    def tearDown(self):
        tf.keras.backend.clear_session()

    def test_stateless_stream_advances_resets_and_restores(self):
        stream = SeedStream(17)
        draw = tf.function(
            lambda: tf.random.stateless_normal((16,), stream.next_seed()),
            jit_compile=True,
        )
        first = draw().numpy()
        saved = stream.get_weights()
        second = draw().numpy()
        self.assertFalse(np.array_equal(first, second))
        stream.set_weights(saved)
        np.testing.assert_array_equal(draw().numpy(), second)
        stream.reset_seed(17)
        np.testing.assert_array_equal(draw().numpy(), first)

    def test_parallel_dataset_uses_unique_stream_counters(self):
        stream = SeedStream(17)
        dataset = tf.data.Dataset.range(512).map(
            lambda _: stream.next_seed(), num_parallel_calls=8,
        )
        seeds = np.asarray(list(dataset.as_numpy_iterator()))
        self.assertEqual(np.unique(seeds, axis=0).shape[0], 512)
        self.assertEqual(int(stream.next_seed()[1]), 512)

    def test_native_diffusion_train_evaluate_and_saved_rng(self):
        model = make_wrapper(show_separate_noise_losses=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="mse", jit_compile=True)
        self.assertTrue(model.jit_compile)
        images = tf.reshape(tf.linspace(-1., 1., 32), (2, 4, 4, 1))
        classes = tf.constant([0, 1])
        before = [value.numpy().copy() for value in model.network.trainable_variables]
        dataset = tf.data.Dataset.from_tensor_slices((images, classes)).batch(2)
        train = model.fit(dataset, epochs=1, verbose=0).history
        test = model.evaluate(dataset, verbose=0, return_dict=True)
        batch_train = model.train_on_batch(images, classes, return_dict=True)
        batch_test = model.test_on_batch(images, classes, return_dict=True)
        self.assertTrue(all(np.isfinite(value).all() for value in train.values()))
        self.assertTrue(all(np.isfinite(value) for value in test.values()))
        self.assertTrue(all(np.isfinite(value) for value in batch_train.values()))
        self.assertTrue(all(np.isfinite(value) for value in batch_test.values()))
        self.assertTrue(any(not np.array_equal(a, b.numpy()) for a, b in zip(before, model.network.trainable_variables)))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "probe.weights.h5")
            model.save_weights(path)
            expected = model.noisify(images)[1].numpy()
            model.load_weights(path)
            np.testing.assert_array_equal(model.noisify(images)[1].numpy(), expected)
        samples = model.sample(labels=[1, 2], steps=2, verbose=0)
        self.assertEqual(samples.shape, (2, 4, 4, 1))
        self.assertTrue(np.isfinite(samples.numpy()).all())

    def test_optional_bicubic_uses_default_graph(self):
        patches = PatchEmbedding(dim=4, grid_size=2, patch_size=2,
                                 pos_embed_type="2d_learned_interpolate")
        model = tf.keras.Sequential([tf.keras.layers.Input((4, 4, 1)), patches])
        model.compile(loss="mse")
        self.assertFalse(patches.supports_jit)
        self.assertFalse(model.jit_compile)
        self.assertEqual(model(tf.ones((2, 4, 4, 1))).shape, (2, 4, 4))

    def test_droppath_compiled_masks_and_checkpoint(self):
        layer = DropPath(.5, seed=23)
        inputs = tf.ones((64, 2, 2))
        draw = tf.function(lambda x: layer(x, training=True), jit_compile=True)
        first = draw(inputs).numpy()
        second = draw(inputs).numpy()
        self.assertFalse(np.array_equal(first, second))
        layer.reset_seed(23)
        np.testing.assert_array_equal(draw(inputs).numpy(), first)

    def test_task_reset_replays_independent_diffusion_streams(self):
        from common.learner import _reset_task_random_streams

        model = make_wrapper()
        images = tf.ones((4, 4, 4, 1))
        _reset_task_random_streams(model, 31)
        first = model.noisify(images)[1].numpy()
        model.get_cfg_labels(tf.ones((4,), tf.int32))
        second = model.noisify(images)[1].numpy()
        _reset_task_random_streams(model, 31)
        np.testing.assert_array_equal(model.noisify(images)[1].numpy(), first)
        np.testing.assert_array_equal(model.noisify(images)[1].numpy(), second)

    def test_progressive_unsupported_resize_disables_only_that_stage(self):
        model = make_wrapper()
        model.compile(loss="mse", jit_compile=True)
        self.assertTrue(model.jit_compile)
        with self.assertWarns(UserWarning):
            model.set_current_resolution(8)
        self.assertFalse(model.jit_compile)
        model.set_current_resolution(None)
        self.assertTrue(model.jit_compile)


if __name__ == "__main__":
    unittest.main()
