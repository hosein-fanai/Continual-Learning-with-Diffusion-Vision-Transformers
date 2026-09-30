"""Diffusion plots use the wrapper's saved input mode and raw pixel outputs."""

import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.utils import plot_noisy_images
from diffusion import DiffusionModel, DiffusionTransformer


class DiffusionPlottingTests(unittest.TestCase):
    """Check plotting coordinates with small real wrappers and no training."""

    def tearDown(self) -> None:
        """Release the isolated Keras graph after each test."""

        tf.keras.backend.clear_session()

    def test_noisy_plot_uses_model_preprocessing_without_changing_state(self) -> None:
        """Raw images enter the configured model range and return as unit plots."""

        images = np.full((1, 2, 2, 1), 127.5, dtype=np.float32)
        for preprocess_type, expected_input in (("standardize", 0.), ("min-max", .5), (None, 127.5)):
            with self.subTest(preprocess_type=preprocess_type):
                network = DiffusionTransformer(
                    num_classes=2, image_size=2, channels=1, patch_size=1, 
                    timesteps=4, dim=4, depth=1, mha_num_heads=1, 
                    vit_block_mlp_ratio=1., seed=17
                )
                model = DiffusionModel(
                    network, use_ema=False, scheduler_name="linear", test_steps=2, 
                    preprocess_type=preprocess_type, modify_first_t=True, seed=17
                )
                before = model.get_weights()
                with patch("common.utils.plot_images") as plot, patch.object(
                    model, "q_sample", wraps=model.q_sample
                ) as q_sample:
                    plot_noisy_images(model, images, interval=2, seed=31)
                q_images, times, noise = q_sample.call_args.args
                np.testing.assert_array_equal(q_images.numpy(), np.full((2, 2, 2, 1), expected_input))
                np.testing.assert_array_equal(times.numpy(), [0, 3])
                np.testing.assert_array_equal(noise[0].numpy(), noise[1].numpy())
                displayed = plot.call_args.args[0]
                np.testing.assert_allclose(displayed[0], .5)
                self.assertTrue(np.all((displayed >= 0.) & (displayed <= 1.)))
                self.assertEqual(plot.call_args.kwargs["titles"], ["t=0", "t=3"])
                for actual, original in zip(model.get_weights(), before):
                    np.testing.assert_array_equal(actual, original)
        np.testing.assert_array_equal(images, np.full_like(images, 127.5))
