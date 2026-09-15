"""Regressions for diagnostic validity, independent of treatment efficacy."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.dataloader import get_dataset
from semantic_consolidation.config import RouteSettings
from semantic_consolidation.controller import RouteController
from semantic_consolidation.diagnostics import (
    balanced_probe, class_geometry, diagnostic_view, gate_coverage, one_vs_rest, probe_batches,
)
from semantic_consolidation.objectives import contrastive_alignment_loss


def _features(network: object, images: tf.Tensor, times: tf.Tensor) -> tuple:
    """Expose a controlled hidden map and a predictor-independent classifier."""

    hidden = tf.linalg.matmul(images, network.transform)
    return hidden, tf.nn.softmax(hidden[:, :network.num_classes])


class DiagnosticTests(unittest.TestCase):
    """Separate measurement correctness from any claimed learning efficacy."""

    def setUp(self) -> None:
        """Use the ordinary route defaults with an explicit reproducible seed."""

        self.settings = RouteSettings(seed=41)

    def test_four_class_probe_is_mixed_and_all_gates_see_each_batch(self) -> None:
        """Reject the original one-true-class-gate-per-batch diagnostic coupling."""

        labels = np.repeat(np.arange(4, dtype="int32"), 40)
        pixels = np.arange(640, dtype="float32").reshape(160, 4)
        controller = RouteController(self.settings)
        dataset = get_dataset(pixels, labels, batch_size=32, shuffle_buffer=0, drop_remainder=False)
        probe = controller._probe_data(dataset, dict(enumerate(range(4))))
        network = SimpleNamespace(num_classes=4, transform=tf.eye(4))
        wrapper = SimpleNamespace(network=network)
        bank = {c: (tf.zeros(4), tf.zeros(4)) for c in range(4)}
        with patch("semantic_consolidation.controller.semantic_features", side_effect=_features):
            report, _, _ = controller._probe(wrapper, probe, {0, 1}, network, bank, None)
        alignment = report["frozen_target_alignment"]
        self.assertEqual(alignment["gate_coverage"]["gate_ids"], [0, 1, 2, 3])
        for batch in range(4):
            rows = [r for r in alignment["comparisons"] if r["batch_id"] == batch]
            self.assertEqual({r["gate_id"] for r in rows}, {0, 1, 2, 3})
            self.assertTrue(all(set(r["class_counts"]) == {0, 1, 2, 3} for r in rows))
            self.assertEqual(len({r["input_sha256"] for r in rows}), 1)
        self.assertEqual(alignment["aggregates"]["selected_gates"]["example_gate_noise_comparisons"], 512)

    def test_hundred_class_subset_spans_old_and_new_and_all_gates_when_requested(self) -> None:
        """Verify executed gate coverage rather than trusting reported IDs alone."""

        old = set(range(90))
        coverage = gate_coverage(list(range(100)), old, self.settings.probe_max_gates, 41)
        self.assertEqual(coverage["old"]["measured"], 8)
        self.assertEqual(coverage["new"]["measured"], 8)
        self.assertEqual(coverage["scope"], "balanced_old_new_gate_subset")
        self.assertTrue(any(c >= 90 for c in coverage["gate_ids"]))
        self.assertEqual(coverage, gate_coverage(list(reversed(range(100))), old, 16, 41))
        whole = gate_coverage(list(range(100)), old, 100, 41)
        self.assertEqual(whole["gate_ids"], list(range(100)))
        self.assertEqual(whole["scope"], "all_available_gates")
        # Actual probe execution must honor selection, not merely its metadata.
        labels = np.repeat(np.arange(100, dtype="int32"), 2)
        pixels = np.eye(100, dtype="float32")[labels]
        probe = balanced_probe(pixels, labels, self.settings)
        network = SimpleNamespace(num_classes=100, transform=tf.eye(100))
        bank = {c: (tf.zeros(100), tf.zeros(100)) for c in range(100)}
        with patch("semantic_consolidation.controller.semantic_features", side_effect=_features):
            report, _, _ = RouteController(self.settings)._probe(
                SimpleNamespace(network=network), probe, old, network, bank, None,
            )
        alignment = report["frozen_target_alignment"]
        for indices in probe_batches(len(labels), 32):
            self.assertGreater(len(np.unique(probe[1][indices])), 1)
        chosen = set(alignment["gate_coverage"]["gate_ids"])
        for batch in range(7):
            rows = [r for r in alignment["comparisons"] if r["batch_id"] == batch]
            self.assertEqual({r["gate_id"] for r in rows}, chosen)
        self.assertEqual(alignment["aggregates"]["selected_gates"]["example_gate_noise_comparisons"], 200 * 16)

    def test_input_order_cannot_restore_true_class_gate_coupling(self) -> None:
        """Canonical sampling yields the same mixed rows after input permutation."""

        labels = np.repeat(np.arange(4, dtype="int32"), 41)
        images = np.arange(len(labels) * 4, dtype="float32").reshape(-1, 4)
        expected = balanced_probe(images, labels, self.settings)
        for order in (np.arange(len(labels))[::-1], np.random.default_rng(67).permutation(len(labels))):
            actual = balanced_probe(images[order], labels[order], self.settings)
            for first, second in zip(expected, actual):
                np.testing.assert_array_equal(first, second)
            for indices in probe_batches(len(actual[1]), self.settings.batch_size):
                self.assertEqual(set(actual[1][indices]), set(range(4)))

    def test_predictor_only_change_does_not_improve_deployed_measures(self) -> None:
        """A training-only predictor solves alignment without changing hidden features."""

        settings = replace(self.settings, batch_size=4)
        controller = RouteController(settings)
        hidden = np.eye(4, dtype="float32")
        targets = np.roll(hidden, 1, axis=0)
        student = SimpleNamespace(num_classes=4, transform=tf.eye(4))
        target = SimpleNamespace(num_classes=4, transform=tf.constant(targets))
        wrapper = SimpleNamespace(network=student)
        predictor = tf.keras.layers.Dense(4, use_bias=False, kernel_initializer="identity")
        predictor(hidden)
        bank = {0: (tf.zeros(4), tf.zeros(4))}
        with patch("semantic_consolidation.controller.semantic_features", side_effect=_features):
            before, h0, views = controller._probe(wrapper, (hidden, np.arange(4)), {0, 1}, target, bank, predictor)
            predictor.kernel.assign(targets)
            after, h1, cached = controller._probe(wrapper, (hidden, np.arange(4)), {0, 1}, target, bank, predictor, views)
        self.assertIs(cached, views)
        np.testing.assert_array_equal(h0, h1)
        for key in ("representation", "class_geometry", "clean_accuracy", "old_accuracy", "new_accuracy", "calibration"):
            self.assertEqual(before[key], after[key], key)
        first = before["frozen_target_alignment"]["aggregates"]["selected_gates"]
        last = after["frozen_target_alignment"]["aggregates"]["selected_gates"]
        for key in ("hidden_infonce", "hidden_target_cosine"):
            self.assertEqual(first[key], last[key])
        self.assertGreater(first["predictor_infonce"], 10.)
        self.assertLess(last["predictor_infonce"], 0.0002)

    def test_noise_is_cached_reproducible_and_does_not_consume_training_rng(self) -> None:
        """Diagnostic corruptions are repeatable and independent of training draws."""

        wrapper = SimpleNamespace(q_sample=lambda x, t, eps: x + eps)
        images = tf.ones((4, 4))
        tf.random.set_seed(101)
        expected = tf.random.normal((4,)).numpy()
        tf.random.set_seed(101)
        first, _ = diagnostic_view(wrapper, images, 2, 53)
        actual = tf.random.normal((4,)).numpy()
        second, _ = diagnostic_view(wrapper, images, 2, 53)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(expected, actual)
        clean, times = diagnostic_view(wrapper, images, 0, 53)
        np.testing.assert_array_equal(clean, images)
        np.testing.assert_array_equal(times, [0, 0, 0, 0])

    def test_geometry_matches_hand_computed_pairs_and_reports_missingness(self) -> None:
        """Count the intended pairs and distinguish missing geometry from zero loss."""

        hidden = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], dtype="float32")
        labels = np.array([0, 0, 1, 1])
        geometry = class_geometry(hidden, labels, {0})
        self.assertAlmostEqual(geometry["all"]["within_class_cosine_distance"], 0., places=6)
        self.assertEqual(geometry["all"]["between_class_cosine_distance"], 1.)
        self.assertEqual(geometry["all"]["within_pairs"], 4)
        self.assertEqual(geometry["old_new"]["between_pairs"], 8)
        self.assertIsNone(geometry["old"]["between_class_cosine_distance"])
        gate = one_vs_rest(hidden, labels, 0)
        self.assertEqual(gate["one_vs_rest_squared_cosine"], 0.)
        self.assertEqual(gate["positive_negative_pairs"], 4)
        self.assertIsNone(one_vs_rest(hidden, labels, 7)["within_class_cosine_distance"])
        self.assertIsNone(one_vs_rest(hidden[:1], labels[:1], 0)["one_vs_rest_squared_cosine"])

    def test_before_after_reuses_exact_noisy_inputs_and_frozen_target_features(self) -> None:
        """All gates and both endpoints share the cached tensors at each noise level."""

        controller = RouteController(replace(self.settings, batch_size=4, noise_levels=(0, 2)))
        network = SimpleNamespace(num_classes=4, transform=tf.eye(4))
        wrapper = SimpleNamespace(network=network, q_sample=lambda x, t, eps: x + eps)
        probe = (np.eye(4, dtype="float32"), np.arange(4))
        bank = {c: (tf.zeros(4), tf.zeros(4)) for c in range(4)}
        with patch("semantic_consolidation.controller.semantic_features", side_effect=_features), patch(
            "semantic_consolidation.controller.diagnostic_view", wraps=diagnostic_view,
        ) as draw:
            before, _, views = controller._probe(wrapper, probe, {0, 1}, network, bank, None)
            cached = [(v["images"].numpy().copy(), v["target_hidden"].numpy().copy()) for v in views]
            after, _, reused = controller._probe(wrapper, probe, {0, 1}, network, bank, None, views)
        self.assertEqual(draw.call_count, 2)
        self.assertIs(views, reused)
        for original, view in zip(cached, reused):
            np.testing.assert_array_equal(original[0], view["images"])
            np.testing.assert_array_equal(original[1], view["target_hidden"])
        self.assertEqual(before["frozen_target_alignment"], after["frozen_target_alignment"])

    def test_identical_class_targets_bound_instance_loss_by_log_p(self) -> None:
        """Identical targets impose the analytic log-count instance-loss floor."""

        for count in (2, 4, 7):
            target = tf.ones((count, 3))
            student = tf.random.stateless_normal((count, 3), seed=[41, count])
            self.assertAlmostEqual(float(contrastive_alignment_loss(student, target, 0.1)), np.log(count), places=5)

    def test_gate_limit_requires_both_strata_to_be_representable(self) -> None:
        """Reject caps that preclude old/new coverage or lack an integer meaning."""

        for value in (0, 1, True, 2.5):
            with self.assertRaises(ValueError):
                replace(self.settings, probe_max_gates=value)


# Direct invocation runs only this bounded diagnostic suite.
if __name__ == "__main__":
    unittest.main()
