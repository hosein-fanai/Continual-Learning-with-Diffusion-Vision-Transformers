"""Analytic section-11 statistics and fixed-row probes on a real tiny DiT."""

from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.mechanistic import calibration_metrics, linear_cka
from semantic_consolidation.experimental_diagnostics import (
    FixedHiddenProbe, extract_hidden, generated_memory_diagnostics,
    hidden_feature_change, polynomial_kid,
)


class PolynomialKidTests(unittest.TestCase):
    """Compare the estimator to independently enumerated ordered pairs."""

    def test_unequal_sample_counts_match_explicit_U_statistic(self) -> None:
        """Verify unequal sample counts match explicit U statistic."""
        real = np.asarray([[1., 2.], [-1., 1.], [0., 2.]])
        generated = np.asarray([[2., 0.], [0., -1.]])

        def kernel(first: np.ndarray, second: np.ndarray) -> float:
            """Evaluate the cubic feature kernel in the explicit reference estimator."""
            return (sum(a * b for a, b in zip(first, second)) / 2. + 1.) ** 3

        real_sum = sum(kernel(real[i], real[j]) for i in range(3) for j in range(3) if i != j)
        generated_sum = sum(kernel(generated[i], generated[j]) for i in range(2) for j in range(2) if i != j)
        cross_sum = sum(kernel(first, second) for first in real for second in generated)
        expected = real_sum / 6. + generated_sum / 2. - 2. * cross_sum / 6.
        report = polynomial_kid(real, generated)
        self.assertAlmostEqual(report["value"], expected)
        self.assertEqual(report["real_sample_count"], 3)
        self.assertEqual(report["generated_sample_count"], 2)

    def test_negative_finite_sample_estimate_is_not_clipped(self) -> None:
        """Verify negative finite sample estimate is not clipped."""
        values = np.asarray([[-1.], [1.]])
        # Within off-diagonals are zero; the cross mean including diagonals is four.
        report = polynomial_kid(values, values.copy())
        self.assertEqual(report["value"], -8.)
        self.assertFalse(report["negative_estimates_clipped"])

    def test_estimator_unbiased_over_all_independent_binary_draws(self) -> None:
        """Verify estimator unbiased over all independent binary draws."""
        pairs = [np.asarray([[first], [second]], dtype="float64")
                 for first in (-1., 1.) for second in (-1., 1.)]
        estimates = [polynomial_kid(first, second)["value"] for first in pairs for second in pairs]
        # Exhaustive IID draws from the same population have population MMD exactly zero.
        self.assertAlmostEqual(float(np.mean(estimates)), 0.)

    def test_invalid_features_are_rejected(self) -> None:
        """Verify invalid features are rejected."""
        for real, generated in (([[1.]], [[1.], [2.]]),
                                ([[1.], [2.]], [[1., 2.], [3., 4.]]),
                                ([[np.nan], [2.]], [[1.], [2.]]),
                                ([[1e200], [1e200]], [[1.], [2.]])):
            with self.subTest(real=real, generated=generated), self.assertRaises(ValueError):
                polynomial_kid(real, generated)


class GeneratedMemoryTests(unittest.TestCase):
    """Coverage, agreement, minority loss and artifacts describe the actual pool."""

    def test_per_class_missingness_and_full_seen_classifier_columns(self) -> None:
        """Verify per class missingness and full seen classifier columns."""
        images = np.asarray([-1., 0., 1.], dtype="float32").reshape(3, 1, 1, 1)
        labels = np.asarray([0, 0, 1])
        probabilities = np.asarray([[.8, .1, .1, 0.], [.1, .2, .7, 0.], [.1, .7, .1, .1]])
        real = np.asarray([-.8, .1, .8, .9, -.5, .5], dtype="float32").reshape(6, 1, 1, 1)
        result = generated_memory_diagnostics(images, labels, [0, 1, 2],
                                              real_images=real, real_labels=[0, 0, 1, 1, 2, 2],
                                              probabilities=probabilities, seed=7)
        self.assertAlmostEqual(result["summary"]["class_coverage"], 2. / 3.)
        self.assertEqual(result["summary"]["class_counts"], {"0": 2, "1": 1, "2": 0})
        self.assertEqual(result["per_class"]["0"]["label_consistency"], .5)
        self.assertEqual(result["per_class"]["1"]["label_consistency"], 1.)
        self.assertIsNone(result["per_class"]["2"]["label_consistency"])
        self.assertIsNone(result["per_class"]["1"]["pixel_diversity"])
        self.assertIsNone(result["per_class"]["1"]["polynomial_kid"])
        self.assertEqual(result["kid_classes_measured"], 1)
        self.assertEqual(result["kid_classes_expected"], 3)
        self.assertEqual(result["macro_polynomial_kid"], result["per_class"]["0"]["polynomial_kid"]["value"])
        self.assertEqual(result["summary"]["calibration"], calibration_metrics(probabilities, labels))
        self.assertEqual(result["feature_extractor"]["name"], "identity_flattened_pixels")
        json.dumps(result, allow_nan=False)

    def test_representatives_are_reproducible_and_recover_source_pixels(self) -> None:
        """Verify representatives are reproducible and recover source pixels."""
        images = np.arange(32, dtype="float32").reshape(8, 2, 2, 1) / 32.
        labels = np.repeat([0, 1], 4)
        root = Path(__file__).resolve().parents[2] / ".tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / "representatives.npz"
            result = generated_memory_diagnostics(images, labels, [0, 1], artifact_path=path,
                                                  representatives_per_class=2, seed=19)
            second = generated_memory_diagnostics(images[::-1], labels[::-1], [0, 1],
                                                  representatives_per_class=2, seed=19)
            with np.load(path, allow_pickle=False) as samples:
                np.testing.assert_array_equal(samples["images"], images[samples["source_indices"]])
                np.testing.assert_array_equal(samples["labels"], labels[samples["source_indices"]])
                selected_again = sum((second["per_class"][str(c)]["representative_source_indices"] for c in [0, 1]), [])
                np.testing.assert_array_equal(samples["images"], images[::-1][selected_again])
            self.assertEqual(result["representative_count"], 4)
            self.assertGreater(result["representative_artifact"]["file_bytes"], 0)
            self.assertEqual(len(result["representative_artifact"]["sha256"]), 64)

    def test_empty_pool_is_explicit_not_an_invented_zero_score(self) -> None:
        """Verify empty pool is explicit not an invented zero score."""
        result = generated_memory_diagnostics(np.empty((0, 2, 2, 1)), [], [0, 1])
        self.assertEqual(result["summary"]["class_coverage"], 0.)
        self.assertIsNone(result["macro_polynomial_kid"])
        self.assertEqual(result["kid_classes_measured"], 0)
        self.assertEqual(result["representative_count"], 0)
        json.dumps(result, allow_nan=False)

    def test_custom_extractor_requires_identity_and_preserves_rows(self) -> None:
        """Verify custom extractor requires identity and preserves rows."""
        images = np.zeros((2, 2, 2, 1), dtype="float32")
        with self.assertRaisesRegex(ValueError, "metadata"):
            generated_memory_diagnostics(images, [0, 0], [0], feature_extractor=lambda values: values)
        metadata = {"name": "frozen-fixture", "pretraining": "none", "preprocessing": "identity", "identity_sha256": "a" * 64}
        with self.assertRaisesRegex(ValueError, "sample count"):
            generated_memory_diagnostics(images, [0, 0], [0], real_images=images, real_labels=[0, 0],
                                          feature_extractor=lambda values: np.ones((3, 2)), feature_metadata=metadata)
        result = generated_memory_diagnostics(images, [0, 0], [0], real_images=images, real_labels=[0, 0],
                                               feature_extractor=lambda values: np.ones((len(values), 2)), feature_metadata=metadata)
        self.assertEqual(result["macro_polynomial_kid"], 0.)
        self.assertEqual(result["feature_extractor"], metadata)


class HiddenDriftTests(unittest.TestCase):
    """Different geometric invariants distinguish movement from representation shape."""

    def test_translation_changes_coordinate_drift_but_not_centered_cka(self) -> None:
        """Verify translation changes coordinate drift but not centered cka."""
        values = np.asarray([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        report = hidden_feature_change(values, values + [3., 4.], [0, 0, 1, 1])
        self.assertAlmostEqual(report["linear_cka"], 1.)
        self.assertAlmostEqual(report["mean_sample_l2_drift"], 5.)
        self.assertAlmostEqual(report["centroid_drift"]["mean_centroid_drift"], 5.)

    def test_width_change_keeps_cka_and_constant_features_are_unavailable(self) -> None:
        """Verify width change keeps cka and constant features are unavailable."""
        first = np.asarray([[-1.], [0.], [1.]])
        report = hidden_feature_change(first, np.concatenate((first, first), axis=1), [0, 0, 0])
        self.assertAlmostEqual(report["linear_cka"], 1.)
        self.assertIsNone(report["mean_sample_l2_drift"])
        constant = hidden_feature_change(np.ones((8, 1)), np.ones((8, 1)), np.zeros(8))
        self.assertIsNone(constant["linear_cka"])
        self.assertEqual(constant["linear_cka_unavailable_reason"], "constant_centered_representation")
        self.assertEqual(constant["mean_sample_l2_drift"], 0.)
        zero = hidden_feature_change(np.zeros((8, 1)), np.ones((8, 1)), np.zeros(8))
        self.assertIsNone(zero["linear_cka"])
        self.assertIsNone(zero["relative_frobenius_drift"])
        self.assertEqual(zero["mean_sample_l2_drift"], 1.)
        self.assertEqual(zero["centroid_drift"]["mean_centroid_drift"], 1.)

    def test_invalid_representations_are_rejected_without_dropping_rows(self) -> None:
        """Nonfinite, missing and misaligned features cannot become valid cohorts."""
        for first, last in (([[np.nan], [2.], [3.]], [[1.], [2.], [3.]]),
                            ([[1.], [2.], [3.]], [[1.], [np.inf], [3.]]),
                            (np.empty((3, 0)), np.ones((3, 1))),
                            (np.ones((3, 1)), np.ones((4, 1)))):
            with self.subTest(first=first, last=last), self.assertRaises(ValueError):
                hidden_feature_change(first, last, np.zeros(3))

    def test_requested_eight_rows_does_not_validate_two_actual_rows(self) -> None:
        """A scarce class retains fixed examples and drift, without inflated counts."""
        images = np.arange(8, dtype="float32").reshape(2, 2, 2, 1)
        wrapper = SimpleNamespace(seen_classes={0: 0}, get_network=lambda name: None)
        probe = FixedHiddenProbe(per_class=8, seed=31)
        with patch("semantic_consolidation.experimental_diagnostics.extract_hidden",
                   side_effect=lambda network, values, batch_size: values.reshape(len(values), -1)):
            first = probe.observe(wrapper, images, [0, 0], 0)
            second = probe.observe(wrapper, images[:0], [], 1)
        self.assertEqual(first["per_class"]["0"]["sample_sha256"], second["per_class"]["0"]["sample_sha256"])
        for comparison in ("since_acquisition", "since_previous_observation"):
            report = second["per_class"]["0"][comparison]
            self.assertEqual(report["sample_count"], 2)
            self.assertIsNone(report["linear_cka"])
            self.assertEqual(report["linear_cka_unavailable_reason"], "fewer_than_three_aligned_observations")
            self.assertEqual(report["mean_sample_l2_drift"], 0.)

    def test_fixed_rows_are_unchanged_by_dataset_reordering(self) -> None:
        """Verify fixed rows are unchanged by dataset reordering."""
        images = np.arange(32, dtype="float32").reshape(8, 2, 2, 1)
        labels = np.repeat([0, 1], 4)
        wrapper = SimpleNamespace(seen_classes={0: 0, 1: 1}, get_network=lambda name: None)
        probe = FixedHiddenProbe(per_class=3, seed=31, retain_images=False)
        with patch("semantic_consolidation.experimental_diagnostics.extract_hidden",
                   side_effect=lambda network, values, batch_size: values.reshape(len(values), -1)):
            first = probe.observe(wrapper, images, labels, 0)
            second = probe.observe(wrapper, images[::-1], labels[::-1], 1)
            missing = probe.observe(wrapper, images[labels == 1], labels[labels == 1], 2)
        for class_id in ("0", "1"):
            self.assertEqual(first["per_class"][class_id]["sample_sha256"], second["per_class"][class_id]["sample_sha256"])
            self.assertEqual(second["per_class"][class_id]["since_acquisition"]["mean_sample_l2_drift"], 0.)
        self.assertEqual(missing["per_class"]["0"]["availability"], "fixed_validation_rows_unavailable")
        self.assertEqual(probe.retained_bytes["image_bytes"], 0)
        self.assertGreater(probe.retained_bytes["feature_bytes"], 0)

    def test_future_labels_and_duplicate_checkpoints_are_rejected(self) -> None:
        """Verify future labels and duplicate checkpoints are rejected."""
        images = np.ones((2, 2, 2, 1), dtype="float32")
        wrapper = SimpleNamespace(seen_classes={0: 0}, get_network=lambda name: None)
        probe = FixedHiddenProbe()
        with self.assertRaisesRegex(ValueError, "future"):
            probe.observe(wrapper, images, [0, 1], 0)
        with patch("semantic_consolidation.experimental_diagnostics.extract_hidden", return_value=np.ones((2, 2))):
            probe.observe(wrapper, images, [0, 0], 0)
        with self.assertRaisesRegex(ValueError, "increasing"):
            probe.observe(wrapper, images, [0, 0], 0)

    def test_missing_ema_cannot_be_reported_as_an_ema_probe(self) -> None:
        """The shared wrapper's legacy raw fallback cannot replace a requested branch."""

        wrapper = SimpleNamespace(seen_classes={0: 0}, use_ema=False, get_network=lambda name: object())
        probe = FixedHiddenProbe(network_name="ema")
        with self.assertRaisesRegex(ValueError, "actual EMA network"):
            probe.observe(wrapper, np.ones((2, 2, 2, 1)), [0, 0], 0)
        self.assertEqual(probe.cohorts, {})
        self.assertIsNone(probe.last_task)


class RealHiddenProbeTests(unittest.TestCase):
    """Exercise genuine feature extraction with unchanged real-network weights."""

    def test_real_tiny_dit_projection_matches_classifier_and_detects_change(self) -> None:
        """Verify real tiny dit projection matches classifier and detects change."""
        from semantic_consolidation.controller import weight_digest
        from semantic_consolidation.tests.test_phases import _make_wrapper

        prior_policy = tf.keras.mixed_precision.global_policy().name
        try:
            wrapper = _make_wrapper()
            wrapper.seen_classes = {0: 0, 1: 1}
            images = np.random.default_rng(53).normal(size=(6, 4, 4, 1)).astype("float32")
            labels = np.repeat([0, 1], 3)
            input_copy, labels_copy = images.copy(), labels.copy()
            before = weight_digest(wrapper.weights)
            features = extract_hidden(wrapper.network, images, batch_size=2)
            times = tf.zeros((6,), dtype=tf.int32)
            probabilities = wrapper.network.predict_class((images, times, times), training=False)
            reconstructed = wrapper.network.classifier.layers[-1](features, training=False)
            np.testing.assert_allclose(probabilities.numpy(), reconstructed.numpy(), rtol=2e-5, atol=1e-6)
            self.assertEqual(weight_digest(wrapper.weights), before)
            probe = FixedHiddenProbe(per_class=3, batch_size=2)
            python_random_state = random.getstate()
            numpy_random_state = np.random.get_state()
            # Deterministic TensorFlow requires explicit initialization before
            # the first global-Generator read; other tests may enable it first.
            try:
                generator = tf.random.get_global_generator()
            except RuntimeError:
                generator = tf.random.Generator.from_seed(53)
                tf.random.set_global_generator(generator)
            generator_state = generator.state.numpy().copy()
            tf.random.set_seed(991)
            expected_next_random = tf.random.uniform((4,)).numpy()
            tf.random.set_seed(991)
            acquisition = probe.observe(wrapper, images, labels, 0)
            unchanged = probe.observe(wrapper, images[::-1], labels[::-1], 1)
            self.assertEqual(weight_digest(wrapper.weights), before)
            self.assertEqual(random.getstate(), python_random_state)
            current_numpy_state = np.random.get_state()
            self.assertEqual(current_numpy_state[0], numpy_random_state[0])
            np.testing.assert_array_equal(current_numpy_state[1], numpy_random_state[1])
            self.assertEqual(current_numpy_state[2:], numpy_random_state[2:])
            np.testing.assert_array_equal(generator.state.numpy(), generator_state)
            np.testing.assert_array_equal(tf.random.uniform((4,)).numpy(), expected_next_random)
            np.testing.assert_array_equal(images, input_copy)
            np.testing.assert_array_equal(labels, labels_copy)
            for class_id in ("0", "1"):
                self.assertIsNone(acquisition["per_class"][class_id]["since_acquisition"])
                self.assertAlmostEqual(unchanged["per_class"][class_id]["since_acquisition"]["mean_sample_l2_drift"], 0.)
            # Perturb the hidden projection, then observe previously cached images.
            projection = next(layer for layer in reversed(wrapper.network.classifier.layers[:-1])
                              if isinstance(layer, tf.keras.layers.Dense))
            perturbation = np.random.default_rng(17).normal(0., .05, size=projection.kernel.shape)
            projection.kernel.assign_add(tf.convert_to_tensor(perturbation, dtype=projection.kernel.dtype))
            changed = probe.observe(wrapper, images[:0], labels[:0], 2)
            self.assertTrue(any(changed["per_class"][class_id]["since_acquisition"]["mean_sample_l2_drift"] > 1e-4
                                for class_id in ("0", "1")))
            self.assertEqual(probe.retained_bytes["image_bytes"], images.nbytes)
            json.dumps(changed, allow_nan=False)
        finally:
            tf.keras.mixed_precision.set_global_policy(prior_policy)


class SmallCohortCkaTests(unittest.TestCase):
    """Two observations make temporal centered CKA uninformative."""

    def test_two_row_degeneracy_is_unavailable_while_drift_remains_measured(self) -> None:
        """The generic statistic is one for changed two-row clouds, so hide it."""
        first = np.asarray([[1., 0.], [-1., 0.]])
        last = np.asarray([[0., 20.], [0., -20.]])
        self.assertAlmostEqual(linear_cka(first, last), 1.)
        originals = first.copy(), last.copy()
        for count in (1, 2):
            with self.subTest(count=count):
                report = hidden_feature_change(first[:count], last[:count], np.zeros(count))
                self.assertIsNone(report["linear_cka"])
                self.assertEqual(report["sample_count"], count)
                self.assertEqual(report["linear_cka_unavailable_reason"],
                                 "fewer_than_three_aligned_observations")
                self.assertGreater(report["mean_sample_l2_drift"], 0.)
                self.assertGreater(report["relative_frobenius_drift"], 0.)
                self.assertIsNotNone(report["centroid_drift"])
                json.dumps(report, allow_nan=False)
        np.testing.assert_array_equal(first, originals[0])
        np.testing.assert_array_equal(last, originals[1])

    def test_eight_row_cohort_can_detect_changed_centered_geometry(self) -> None:
        """Orthogonal centered scalar observations have zero linear CKA."""
        first = np.asarray([-3., -2., -1., 0., 0., 1., 2., 3.])[:, None]
        last = np.asarray([1., -1., 1., -1., -1., 1., -1., 1.])[:, None]
        report = hidden_feature_change(first, last, np.zeros(8))
        self.assertAlmostEqual(report["linear_cka"], 0.)
        self.assertIsNone(report["linear_cka_unavailable_reason"])
        self.assertEqual(report["sample_count"], 8)
        self.assertGreater(report["mean_sample_l2_drift"], 0.)
        self.assertAlmostEqual(hidden_feature_change(first, first.copy(), np.zeros(8))["linear_cka"], 1.)


# Run this module directly while keeping imports free of execution side effects.
if __name__ == "__main__":
    unittest.main()
