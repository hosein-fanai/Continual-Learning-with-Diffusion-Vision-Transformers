"""Known-outcome checks for post-generation replay variation measurements."""

from __future__ import annotations

import unittest

import numpy as np

from common.replay_diagnostics import generated_sample_variation


class GeneratedSampleVariationTests(unittest.TestCase):
    """Compare image contrast and sample diversity with independent outcomes."""

    def test_identical_textured_images_have_contrast_but_no_diversity(self) -> None:
        """Repeated texture has within-image contrast but no sample variation."""
        samples = np.tile(np.array([0.0, 1.0, 0.0, 1.0]), (3, 1))
        result = generated_sample_variation(samples, [4, 4, 4], batch_size=2)
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["class_count"], 1)
        self.assertEqual(result["eligible_class_count"], 1)
        self.assertEqual(result["mean_image_std"], 0.5)
        self.assertEqual(result["mean_pixel_std"], 0.0)
        self.assertEqual(result["within_class_pixel_std"], 0.0)

    def test_class_collapse_can_coexist_with_global_variation(self) -> None:
        """Two distinct class prototypes still have zero within-class diversity."""
        samples = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.float32)
        result = generated_sample_variation(samples, [4, 4, 8, 8], batch_size=3)
        self.assertEqual(result["mean_image_std"], 0.0)
        self.assertEqual(result["mean_pixel_std"], 0.5)
        self.assertEqual(result["within_class_pixel_std"], 0.0)
        self.assertEqual(result["eligible_class_count"], 2)

    def test_two_images_have_hand_computed_normalized_standard_deviations(self) -> None:
        """Both coordinate and image deviations are one before range scaling."""
        samples = np.array([[0.0, 2.0], [2.0, 4.0]])
        original = samples.copy()
        result = generated_sample_variation(samples, [[5], [5]], value_range=2.0)
        for metric in ("mean_image_std", "mean_pixel_std", "within_class_pixel_std"):
            self.assertEqual(result[metric], 0.5)
        np.testing.assert_array_equal(samples, original)

    def test_three_constant_images_have_hand_computed_population_variance(self) -> None:
        """Three means 0, 2, 4 give variance 8/3; the paired class has std 1."""
        result = generated_sample_variation(
            np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]]), [2, 2, 7]
        )
        self.assertEqual(result["mean_image_std"], 0.0)
        self.assertAlmostEqual(result["mean_pixel_std"], np.sqrt(8.0 / 3.0))
        self.assertEqual(result["within_class_pixel_std"], 1.0)
        self.assertEqual(result["eligible_class_count"], 1)

    def test_classes_have_equal_weight_despite_different_sample_counts(self) -> None:
        """Class deviations one and two contribute equally to the result 1.5."""
        result = generated_sample_variation(
            np.array([[0.0], [2.0], [0.0], [0.0], [4.0], [4.0]]),
            [0, 0, 1, 1, 1, 1],
            batch_size=2,
        )
        self.assertEqual(result["within_class_pixel_std"], 1.5)

    def test_chunk_boundaries_and_partial_tails_preserve_pool_statistics(self) -> None:
        """Chunked moments agree with direct full-pool NumPy statistics."""
        samples = np.square(np.arange(102, dtype=np.float64)).reshape(17, 2, 3)
        labels = np.array([0, 1, 2, 0, 0, 1, 0, 2, 2, 2, 1, 0, 2, 1, 2, 0, 3])
        flat = samples.reshape(17, -1)
        expected = {
            "mean_image_std": np.std(flat, axis=1).mean(),
            "mean_pixel_std": np.std(flat, axis=0).mean(),
            "within_class_pixel_std": np.mean([
                np.std(flat[labels == label], axis=0).mean() for label in [0, 1, 2]
            ]),
        }
        for batch_size in (1, 2, 4, 8, 17, 32):
            with self.subTest(batch_size=batch_size):
                result = generated_sample_variation(samples, labels, batch_size=batch_size)
                self.assertEqual(result["sample_count"], 17)
                self.assertEqual(result["class_count"], 4)
                self.assertEqual(result["eligible_class_count"], 3)
                for metric, value in expected.items():
                    self.assertAlmostEqual(result[metric], value, places=10)

    def test_large_common_offset_does_not_destroy_small_variation(self) -> None:
        """Centered moments retain std 1 beside a large shared pixel offset."""
        samples = np.array([[1e12, 1e12 + 2], [1e12 + 2, 1e12 + 4]])
        result = generated_sample_variation(samples, [0, 0], batch_size=1)
        self.assertEqual(result["mean_image_std"], 1.0)
        self.assertEqual(result["mean_pixel_std"], 1.0)
        self.assertEqual(result["within_class_pixel_std"], 1.0)

    def test_singletons_have_no_eligible_within_class_measurement(self) -> None:
        """A singleton pool has zero across-image std and unavailable class std."""
        result = generated_sample_variation(np.array([[0.0, 2.0]]), [0])
        self.assertEqual(result["mean_image_std"], 1.0)
        self.assertEqual(result["mean_pixel_std"], 0.0)
        self.assertIsNone(result["within_class_pixel_std"])
        self.assertEqual(result["eligible_class_count"], 0)
        result = generated_sample_variation(np.array([[0.0], [2.0]]), [0, 1])
        self.assertEqual(result["mean_pixel_std"], 1.0)
        self.assertIsNone(result["within_class_pixel_std"])
        self.assertEqual(result["eligible_class_count"], 0)

    def test_empty_pool_reports_unavailable_measurements(self) -> None:
        """An empty image pool has zero counts and no numerical measurements."""
        self.assertEqual(
            generated_sample_variation(np.empty((0, 2, 2, 3)), np.empty(0, dtype=int)),
            {
                "sample_count": 0,
                "class_count": 0,
                "mean_image_std": None,
                "mean_pixel_std": None,
                "within_class_pixel_std": None,
                "eligible_class_count": 0,
            },
        )

    def test_invalid_range_batch_size_or_sample_shapes_are_rejected(self) -> None:
        """Invalid inputs fail before producing misleading variation values."""
        samples = np.zeros((2, 2))
        for value_range in (0, -1, np.nan, np.inf):
            with self.subTest(value_range=value_range):
                with self.assertRaisesRegex(ValueError, "value_range"):
                    generated_sample_variation(samples, [0, 0], value_range=value_range)
        for batch_size in (0, -1, 1.5, True):
            with self.subTest(batch_size=batch_size):
                with self.assertRaisesRegex(ValueError, "batch_size"):
                    generated_sample_variation(samples, [0, 0], batch_size=batch_size)
        with self.assertRaisesRegex(ValueError, "labels"):
            generated_sample_variation(samples, [0])
        with self.assertRaisesRegex(ValueError, "finite"):
            generated_sample_variation(np.array([[np.nan]]), [0])
        with self.assertRaisesRegex(ValueError, "coordinate"):
            generated_sample_variation(np.empty((2, 0)), [0, 0])


if __name__ == "__main__":
    unittest.main()
