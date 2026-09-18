"""Numerical and reproducibility checks for the TMCL augmentation policy."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from semantic_consolidation import augmentation as aug


class AugmentationTests(unittest.TestCase):
    """Check phase policies, pixel-space semantics and isolated stateless RNGs."""

    @staticmethod
    def _images(count: int = 4, height: int = 32, width: int = 32) -> tf.Tensor:
        """Make deterministic colored patterns without touching a global RNG."""

        return tf.random.stateless_uniform((count, height, width, 3), (21, 19), minval=-1., maxval=1.)

    def test_acquisition_only_flips_and_preserves_pixels(self) -> None:
        """Every image is exactly itself or its horizontal reflection."""

        images = self._images(32, 5, 7)
        result = aug.acquisition_augmentation(images, 41).numpy()
        same = np.all(result == images.numpy(), axis=(1, 2, 3))
        flipped = np.all(result == images.numpy()[:, :, ::-1, :], axis=(1, 2, 3))
        self.assertTrue(np.all(same | flipped))
        self.assertTrue(np.any(same))
        self.assertTrue(np.any(flipped))
        np.testing.assert_array_equal(result, aug.acquisition_augmentation(images, 41))
        self.assertFalse(np.array_equal(result, aug.acquisition_augmentation(images, 42)))

    def test_flip_probability_is_per_image(self) -> None:
        """A large identical-image batch contains independent near-half flips."""

        image = tf.constant([[[[-1., 0., 1.], [1., 0., -1.]]]])
        images = tf.repeat(image, 4096, axis=0)
        result = aug.acquisition_augmentation(images, 99).numpy()
        rate = np.mean(result[:, 0, 0, 0] == 1.)
        self.assertGreater(rate, 0.47)
        self.assertLess(rate, 0.53)

    def test_four_independent_deterministic_default_views(self) -> None:
        """The default returns four RGB32 views with replayable sample draws."""

        images = tf.repeat(self._images(1), 4, axis=0)
        first = aug.consolidation_views(images, (123, 5))
        replay = aug.consolidation_views(images, (123, 5))
        self.assertEqual(len(first), 4)
        for view, repeated in zip(first, replay):
            self.assertEqual(view.shape, (4, 32, 32, 3))
            self.assertEqual(view.dtype, tf.float32)
            self.assertTrue(np.isfinite(view).all())
            np.testing.assert_array_equal(view, repeated)
            self.assertFalse(np.array_equal(view[0], view[1]))
        for view in first[1:]:
            self.assertFalse(np.array_equal(first[0], view))

    def test_num_views_does_not_change_earlier_streams(self) -> None:
        """An extra view does not shift any existing view's random draws."""

        images = self._images(2)
        two = aug.consolidation_views(images, 31, num_views=2)
        four = aug.consolidation_views(images, 31)
        for expected, actual in zip(two, four):
            np.testing.assert_array_equal(expected, actual)

    def test_model_scale_round_trip_on_unmodified_pixels(self) -> None:
        """Intensity transforms receive [0,1] pixels rather than model values."""

        images = tf.constant([[[[-1., 0., 1.]]]])
        with mock.patch.object(aug, "_consolidation_view", return_value=(images + 1.) / 2.) as view:
            result = aug.consolidation_views(images, 1, num_views=1)[0]
        np.testing.assert_array_equal(result, images)
        np.testing.assert_array_equal(view.call_args.args[0], [[[[0., 0.5, 1.]]]])

    def test_color_brightness_is_multiplicative(self) -> None:
        """Kornia ColorJitter brightness scales rather than adds an offset."""

        pixels = tf.constant([[[[0.2, 0.4, 0.9]]]])
        result = aug._color_operation(pixels, tf.constant(0), tf.constant([[[[1.4]]]]))
        np.testing.assert_allclose(result, [[[[0.28, 0.56, 1.]]]], atol=1e-7)

    def test_contrast_uses_scalar_mean_luminance(self) -> None:
        """RGB contrast is anchored at luminance, not per-channel means."""

        pixels = tf.constant([[[[1., 0., 0.], [0., 1., 0.]]]])
        result = aug._color_operation(pixels, tf.constant(1), tf.zeros((1, 1, 1, 1)))
        np.testing.assert_allclose(result, np.full((1, 1, 2, 3), (0.299 + 0.587) / 2), atol=1e-7)

    def test_saturation_blends_with_luminance(self) -> None:
        """Zero saturation produces Kornia's grayscale coefficients."""

        pixels = tf.constant([[[[1., 0., 0.], [0., 0., 1.]]]])
        result = aug._color_operation(pixels, tf.constant(2), tf.zeros((1, 1, 1, 1)))
        np.testing.assert_allclose(result, [[[[0.299] * 3, [0.114] * 3]]], atol=1e-7)

    def test_hue_uses_fraction_of_revolution(self) -> None:
        """A one-third turn maps red to green through HSV hue."""

        pixels = tf.constant([[[[1., 0., 0.]]]])
        result = aug._color_operation(pixels, tf.constant(3), tf.constant([[[[1. / 3.]]]]))
        np.testing.assert_allclose(result, [[[[0., 1., 0.]]]], atol=1e-6)

    def test_solarization_zero_width_means_midpoint_threshold(self) -> None:
        """The paper's thresholds=0 in Kornia does not invert dark pixels."""

        pixels = tf.constant([-0.1, 0., 0.2, 0.5, 0.8, 1., 1.1])
        np.testing.assert_allclose(aug._solarize(pixels), [0., 0., 0.2, 0.5, 0.2, 0., 0.], atol=1e-7)

    def test_pipeline_order_and_even_paper_views(self) -> None:
        """Solarization follows flip only on one-based views two and four."""

        pixels = tf.ones((2, 4, 4, 3)) * 0.75
        for view_index in range(4):
            with self.subTest(view=view_index + 1):
                calls = []

                def crop(images: tf.Tensor, seed: tf.Tensor, size: tuple[int, int]) -> tf.Tensor:
                    """Record the crop operation while preserving the test image."""

                    calls.append("crop")
                    return images

                def color(images: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
                    """Record the color operation while preserving the test image."""

                    calls.append("color")
                    return images

                def gray(images: tf.Tensor) -> tf.Tensor:
                    """Record grayscale evaluation while preserving the test image."""

                    calls.append("gray")
                    return images

                def flip(images: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
                    """Record the flip operation while preserving the test image."""

                    calls.append("flip")
                    return images

                def solarize(images: tf.Tensor) -> tf.Tensor:
                    """Record solarization evaluation while preserving the image."""

                    calls.append("solarize")
                    return images

                with (
                    mock.patch.object(aug, "_random_resized_crop", side_effect=crop),
                    mock.patch.object(aug, "_color_jitter", side_effect=color),
                    mock.patch.object(aug, "_gray", side_effect=gray),
                    mock.patch.object(aug, "_flip", side_effect=flip),
                    mock.patch.object(aug, "_solarize", side_effect=solarize),
                ):
                    aug._consolidation_view(pixels, aug._seed_pair(3), view_index, (4, 4))
                expected = ["crop", "color", "gray", "flip"]
                # The mathematical paper views are numbered from one.
                if view_index in (1, 3):
                    expected.append("solarize")
                self.assertEqual(calls, expected)

    def test_crop_fallback_is_centered_with_valid_aspect_ratio(self) -> None:
        """Ten invalid proposals use the standard aspect-constrained center."""

        with mock.patch.object(aug, "_uniform", return_value=tf.ones(10)):
            box = aug._crop_box(tf.constant([16, 64, 3]), aug._seed_pair(1))
        self.assertEqual(tuple(int(value) for value in box), (0, 21, 16, 21))

    def test_bicubic_crop_preserves_corners_and_overshoots(self) -> None:
        """Corner alignment matches Kornia; resize alone must not clamp."""

        pixels = tf.constant([[[[0., 0., 0.], [1., 1., 1.], [0., 0., 0.], [0., 0., 0.]]]])
        with mock.patch.object(aug, "_crop_box", return_value=(0, 0, 1, 4)):
            result = aug._random_resized_crop(pixels, aug._seed_pair(1), (1, 16))
        np.testing.assert_array_equal(result[0, 0, 0], pixels[0, 0, 0])
        np.testing.assert_array_equal(result[0, 0, -1], pixels[0, 0, -1])
        self.assertLess(float(tf.reduce_min(result)), 0.)

    def test_tf_function_matches_eager(self) -> None:
        """The same stateless policy works in ordinary graph execution."""

        images = self._images(2, 8, 8)

        @tf.function(input_signature=[tf.TensorSpec((None, 8, 8, 3), tf.float32)])
        def transform(batch: tf.Tensor) -> tuple[tf.Tensor, ...]:
            """Trace two small views to exercise graph control flow."""

            return aug.consolidation_views(batch, 34, num_views=2, image_size=None)

        eager = aug.consolidation_views(images, 34, num_views=2, image_size=None)
        for expected, actual in zip(eager, transform(images)):
            np.testing.assert_allclose(expected, actual, atol=2e-6)
            self.assertEqual(actual.shape, (2, 8, 8, 3))

    def test_augmentation_does_not_advance_global_rng(self) -> None:
        """Local augmentation seeds do not perturb caller-owned randomness."""

        tf.random.set_seed(198)
        expected = tf.random.uniform((8,))
        tf.random.set_seed(198)
        aug.consolidation_views(self._images(1, 4, 4), 88, num_views=1, image_size=4)
        actual = tf.random.uniform((8,))
        np.testing.assert_array_equal(expected, actual)

    def test_invalid_geometry_range_and_seed_fail(self) -> None:
        """Unsupported clean inputs fail instead of being silently normalized."""

        for images in (tf.zeros((2, 8, 8, 1)), tf.ones((2, 8, 8, 3)) * 2., tf.zeros((8, 8, 3))):
            with self.subTest(shape=images.shape), self.assertRaises((ValueError, tf.errors.InvalidArgumentError)):
                aug.acquisition_augmentation(images, 0)
        for seed in (1.5, [1], [1, 2, 3]):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                aug.acquisition_augmentation(self._images(1), seed)
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                aug.consolidation_views(self._images(1), 0, num_views=count)
        for size in (0, -1, True, 2.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                aug.consolidation_views(self._images(1), 0, image_size=size)


# Support the same direct unittest entry point as the neighboring test files.
if __name__ == "__main__":
    unittest.main()
