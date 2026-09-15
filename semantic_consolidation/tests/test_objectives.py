"""Analytical and gradient checks for route-one acquisition/consolidation losses."""

from __future__ import annotations

import math
import unittest

import numpy as np
import tensorflow as tf

from semantic_consolidation.objectives import (
    contrastive_alignment_loss,
    modulation_separation_loss,
    normalized_feature_distillation_loss,
    normalized_features,
    reliability_weights,
)


class ObjectiveTests(unittest.TestCase):
    """Verify specified reductions, pair semantics, freezing, and domain checks."""

    def test_uniform_collapse_is_log_batch_size(self) -> None:
        """Constant targets cannot make instance-matched contrastive loss zero."""

        for batch_size in (2, 4, 9):
            for temperature in (0.2, 1e-6):
                with self.subTest(batch_size=batch_size, temperature=temperature):
                    collapsed = tf.ones((batch_size, 3))
                    loss = contrastive_alignment_loss(collapsed, collapsed, temperature=temperature)
                    self.assertAlmostEqual(float(loss), math.log(batch_size), places=5)

    def test_matches_hand_computed_two_row_loss(self) -> None:
        """Orthogonal matched rows give softplus(-1 / temperature)."""

        features = tf.eye(2)
        loss = contrastive_alignment_loss(features, features, temperature=0.5)
        self.assertAlmostEqual(float(loss), math.log1p(math.exp(-2.0)), places=6)

    def test_matched_pairs_beat_permuted_targets(self) -> None:
        """The diagonal indexes paired images, not the nearest available target."""

        features = tf.eye(4)
        matched = contrastive_alignment_loss(features, features)
        permuted = contrastive_alignment_loss(features, tf.roll(features, 1, axis=0))
        self.assertLess(float(matched), float(permuted))
        self.assertAlmostEqual(float(permuted - matched), 10.0, places=5)

    def test_same_class_rows_remain_instance_negatives(self) -> None:
        """Duplicate target rows retain the documented false-negative cost."""

        features = tf.constant([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        loss = contrastive_alignment_loss(features, features, temperature=0.01)
        self.assertAlmostEqual(float(loss), math.log(2), places=5)

    def test_weighting_preserves_batch_denominator(self) -> None:
        """Reliability scales the loss instead of being normalized away."""

        features = tf.ones((4, 2))
        loss = contrastive_alignment_loss(features, features, row_weights=[0., 0.5, 1., 0.5])
        self.assertAlmostEqual(float(loss), 0.5 * math.log(4), places=5)
        zero = contrastive_alignment_loss(features, features, row_weights=tf.zeros(4))
        self.assertEqual(float(zero), 0.0)

    def test_targets_and_weights_are_detached(self) -> None:
        """Only student features receive the consolidation objective gradient."""

        for objective in (contrastive_alignment_loss, normalized_feature_distillation_loss):
            with self.subTest(objective=objective.__name__):
                student = tf.Variable([[1., 0.2], [0.3, 1.]])
                target = tf.Variable([[1., 0.], [0., 1.]])
                weights = tf.Variable([0.7, 1.])
                with tf.GradientTape() as tape:
                    loss = objective(student, target, row_weights=weights)
                student_gradient, target_gradient, weight_gradient = tape.gradient(
                    loss, [student, target, weights]
                )
                self.assertIsNotNone(student_gradient)
                self.assertGreater(float(tf.linalg.global_norm([student_gradient])), 0.)
                self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(student_gradient))))
                self.assertIsNone(target_gradient)
                self.assertIsNone(weight_gradient)

    def test_normalized_mse_hand_computed_and_scale_invariant(self) -> None:
        """Two orthogonal unit feature pairs have per-dimension MSE one."""

        student = tf.constant([[4., 0.], [0., 2.]])
        target = tf.constant([[0., 3.], [6., 0.]])
        self.assertAlmostEqual(float(normalized_feature_distillation_loss(student, target)), 1.)
        self.assertAlmostEqual(float(normalized_feature_distillation_loss(
            student, target, row_weights=[0.25, 0.75]
        )), 0.5)
        self.assertEqual(float(normalized_feature_distillation_loss(student[:1], student[:1])), 0.)

    def test_separation_zero_for_aligned_positives_orthogonal_negatives(self) -> None:
        """The acquisition objective has a realizable zero-loss geometry."""

        features = tf.constant([[2., 0.], [4., 0.], [0., 3.], [0., -5.]])
        loss = modulation_separation_loss(features, [True, True, False, False])
        self.assertAlmostEqual(float(loss), 0., places=6)

    def test_separation_excludes_positive_self_pairs(self) -> None:
        """Opposite positive vectors give distance two, without diagonal dilution."""

        features = tf.constant([[1., 0.], [-1., 0.], [0., 1.]])
        loss = modulation_separation_loss(features, [True, True, False])
        self.assertAlmostEqual(float(loss), 2.)

    def test_negative_cosines_cannot_cancel(self) -> None:
        """Opposite signs of cross-class cosine both receive a penalty."""

        features = tf.constant([[1., 0.], [1., 0.], [1., 0.], [-1., 0.]])
        loss = modulation_separation_loss(features, [True, True, False, False], 2.)
        self.assertAlmostEqual(float(loss), 2., places=6)

    def test_constant_bias_and_zero_gain_cannot_solve_acquisition(self) -> None:
        """One common affine transform cannot encode labels solely in its bias."""

        features = tf.constant([[1., 2.], [2., 1.], [-1., 0.], [0., -1.]])
        gain = tf.zeros(2)
        bias = tf.constant([3., -4.])
        loss = modulation_separation_loss(features * gain + bias, [True, True, False, False])
        self.assertAlmostEqual(float(loss), 1., places=6)
        zeros = modulation_separation_loss(tf.zeros_like(features), [True, True, False, False])
        self.assertEqual(float(zeros), 1.)

    def test_separation_gradients_reach_shared_affine_parameters(self) -> None:
        """Gain and bias can learn from all rows under a selected-class mask."""

        features = tf.constant([[1., 0.3], [0.8, 0.1], [0.2, 1.], [-0.1, 1.3]])
        gain = tf.Variable([1., 1.])
        bias = tf.Variable([0., 0.])
        with tf.GradientTape() as tape:
            loss = modulation_separation_loss(features * gain + bias, [True, True, False, False])
        gradients = tape.gradient(loss, [gain, bias])
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(gradient))))
            self.assertGreater(float(tf.linalg.global_norm([gradient])), 0.)

    def test_reliability_is_bounded_monotone_and_detached(self) -> None:
        """Lower signal power reduces semantic weight, with an explicit floor."""

        signal_power = tf.Variable([-0.1, 0., 0.2, 0.8, 1., 1.1])
        with tf.GradientTape() as tape:
            weights = reliability_weights(signal_power, floor=0.1)
            total = tf.reduce_sum(weights)
        np.testing.assert_allclose(weights.numpy(), [0.1, 0.1, 0.2, 0.8, 1., 1.])
        self.assertIsNone(tape.gradient(total, signal_power))

    def test_float32_stability_and_large_norms(self) -> None:
        """Half precision inputs and large finite norms yield finite losses."""

        features = tf.constant([[1., 0.], [0., 1.]], dtype=tf.float16)
        loss = contrastive_alignment_loss(features, tf.reverse(features, [0]), temperature=1e-4)
        self.assertEqual(loss.dtype, tf.float32)
        self.assertAlmostEqual(float(loss), 10000., places=2)
        large = normalized_features(tf.constant([[3e30, 4e30], [0., 0.]]))
        np.testing.assert_allclose(large.numpy(), [[0.6, 0.8], [0., 0.]], rtol=1e-6)

    def test_graph_execution_matches_eager(self) -> None:
        """Dynamic-batch graph mode preserves loss definitions and validations."""

        @tf.function(input_signature=[tf.TensorSpec([None, 2], tf.float32)])
        def graph_loss(features: tf.Tensor) -> tf.Tensor:
            """Evaluate the alignment objective under graph execution for numerical parity."""
            return contrastive_alignment_loss(features, features)

        features = tf.eye(2)
        self.assertAlmostEqual(float(graph_loss(features)), float(contrastive_alignment_loss(features, features)))
        with self.assertRaises(tf.errors.InvalidArgumentError):
            graph_loss(features[:1])

    def test_rejects_invalid_alignment_arguments(self) -> None:
        """Singleton batches, nonfinite values, shapes, and invalid weights fail."""

        features = tf.eye(2)
        invalid_calls = [
            lambda: contrastive_alignment_loss(features[:1], features[:1]),
            lambda: contrastive_alignment_loss(features, tf.eye(3)),
            lambda: contrastive_alignment_loss(features, tf.ones((2, 1))),
            lambda: contrastive_alignment_loss([1., 0.], [1., 0.]),
            lambda: contrastive_alignment_loss(tf.zeros((0, 2)), tf.zeros((0, 2))),
            lambda: contrastive_alignment_loss([[float("nan"), 0.], [0., 1.]], features),
        ]
        invalid_calls.extend(
            lambda temperature=temperature: contrastive_alignment_loss(features, features, temperature)
            for temperature in (0., -0.1, float("nan"), float("inf"), [0.1])
        )
        invalid_calls.extend(
            lambda weights=weights: contrastive_alignment_loss(features, features, row_weights=weights)
            for weights in ([1.], [[1.], [1.]], [-0.1, 1.], [1.1, 1.], [float("nan"), 1.])
        )
        for index, invalid_call in enumerate(invalid_calls):
            with self.subTest(case=index), self.assertRaises((ValueError, tf.errors.InvalidArgumentError)):
                invalid_call()

    def test_rejects_invalid_acquisition_arguments(self) -> None:
        """No valid class separation objective exists without both pair families."""

        features = tf.ones((3, 2))
        for mask in ([True, False, False], [True, True, True], [False, False, False], [True, False]):
            with self.subTest(mask=mask), self.assertRaises((ValueError, tf.errors.InvalidArgumentError)):
                modulation_separation_loss(features, mask)
        with self.assertRaises(TypeError):
            modulation_separation_loss(features, [1, 1, 0])
        for weight in (0., -1., float("inf"), float("nan")):
            with self.subTest(weight=weight), self.assertRaises((ValueError, tf.errors.InvalidArgumentError)):
                modulation_separation_loss(features, [True, True, False], weight)

    def test_rejects_invalid_reliability_arguments(self) -> None:
        """An invalid floor or nonfinite schedule fails before silently clipping."""

        for floor in (-0.1, 1.1, float("nan"), [0.1]):
            with self.subTest(floor=floor), self.assertRaises((ValueError, tf.errors.InvalidArgumentError)):
                reliability_weights([0.5], floor)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            reliability_weights([float("inf")])


# Run this module directly while keeping imports free of execution side effects.
if __name__ == "__main__":
    unittest.main()
