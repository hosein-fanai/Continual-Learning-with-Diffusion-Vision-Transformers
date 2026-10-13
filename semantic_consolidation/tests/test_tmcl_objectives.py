"""Independent numeric and gradient checks for the pinned TMCL objectives."""

from __future__ import annotations

import math
import unittest

import numpy as np
import tensorflow as tf

from semantic_consolidation.objectives import (
    modulation_separation_loss, 
    multiview_barlow_loss, 
    orthogonal_projection_loss
)


def _reference_multiview(
    views: list[np.ndarray], off_diagonal_weight: float, scale: float, epsilon: float
) -> float:
    """Evaluate individual correlation entries using independent NumPy loops."""

    matrices = [np.asarray(view, dtype=np.float64) for view in views]
    joined = np.concatenate(matrices, axis=0)
    mean = np.mean(joined, axis=0)
    variance = np.mean((joined - mean) ** 2, axis=0)
    normalized = [(matrix - mean) / np.sqrt(variance + epsilon) for matrix in matrices]
    expected = sum(normalized) / len(normalized)
    batch, width = matrices[0].shape
    total = 0.
    for view in normalized:
        for row in range(width):
            for column in range(width):
                correlation = sum(expected[index, row] * view[index, column]
                                  for index in range(batch)) / batch
                # The diagonal enforces feature-wise invariance to the view mean.
                if row == column:
                    total += scale * (correlation - 1.) ** 2
                # Every ordered off-diagonal entry receives the redundancy penalty.
                else:
                    total += scale * off_diagonal_weight * correlation ** 2
    return float(total)


class TMCLObjectiveTests(unittest.TestCase):
    """Distinguish official multiview/OPL semantics from nearby loss variants."""

    def test_joint_normalization_preserves_between_view_offsets(self) -> None:
        """Joint normalization yields one half where per-branch BT would yield zero."""

        first = tf.constant([[-1.], [1.]])
        shifted = tf.constant([[1.], [3.]])
        loss = multiview_barlow_loss([first, shifted], scale=1., epsilon=0.)
        self.assertAlmostEqual(float(loss), 0.5, places=6)

    def test_view_mean_correlation_and_sum_are_not_pairwise_barlow(self) -> None:
        """Opposite views cancel their mean; duplicating both doubles the summed loss."""

        first = tf.constant([[-1.], [1.]])
        second = -first
        loss = multiview_barlow_loss([first, second], scale=1., epsilon=0.)
        duplicated = multiview_barlow_loss([first, second, first, second], scale=1., epsilon=0.)
        self.assertAlmostEqual(float(loss), 2., places=6)
        self.assertAlmostEqual(float(duplicated), 4., places=6)

    def test_four_view_loss_matches_independent_entrywise_reference(self) -> None:
        """Asymmetric views expose centering, view averaging and off-diagonal errors."""

        views = [
            np.array([[0., 1.], [2., -1.], [4., 2.]], dtype=np.float32), 
            np.array([[1., 3.], [-2., 0.], [2., 1.]], dtype=np.float32), 
            np.array([[2., -1.], [1., 4.], [0., -2.]], dtype=np.float32), 
            np.array([[-3., 2.], [0., 1.], [5., 0.]], dtype=np.float32)
        ]
        expected = _reference_multiview(views, 0.17, 0.024, 1e-5)
        observed = multiview_barlow_loss(views, off_diagonal_weight=0.17)
        self.assertAlmostEqual(float(observed), expected, places=6)

    def test_collapsed_channels_keep_the_full_diagonal_penalty(self) -> None:
        """Constant features stay finite and cannot give zero BT loss."""

        views = [tf.ones((4, 3), dtype=tf.float16) for _ in range(4)]
        loss = multiview_barlow_loss(views)
        self.assertEqual(loss.dtype, tf.float32)
        self.assertAlmostEqual(float(loss), 4 * 3 * 0.024, places=6)

    def test_all_view_gradients_match_finite_differences(self) -> None:
        """Joint-moment gradients reach both views and agree with numeric derivatives."""

        first = tf.constant([[0.2, 0.7], [-1.3, 2.2], [0.5, -0.4]])
        second = tf.constant([[1.1, -0.2], [0.8, 0.6], [-0.7, 1.3]])

        def objective(left: tf.Tensor, right: tf.Tensor) -> tf.Tensor:
            """Retain both arguments in the jointly normalized view objective."""

            return multiview_barlow_loss([left, right], off_diagonal_weight=0.13, scale=1.)

        analytical, numerical = tf.test.compute_gradient(objective, [first, second], delta=1e-3)
        for actual, expected in zip(analytical, numerical):
            self.assertGreater(float(np.linalg.norm(actual)), 0.)
            np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)

    def test_stopped_target_is_owned_by_the_caller(self) -> None:
        """MI-style explicit stops block targets while VI leaves them differentiable."""

        first = tf.Variable([[0.2, 0.7], [-1.3, 2.2], [0.5, -0.4]])
        second = tf.Variable([[1.1, -0.2], [0.8, 0.6], [-0.7, 1.3]])
        with tf.GradientTape() as tape:
            stopped_loss = multiview_barlow_loss([first, tf.stop_gradient(second)])
        first_gradient, second_gradient = tape.gradient(stopped_loss, [first, second])
        self.assertIsNone(second_gradient)
        self.assertIsNotNone(first_gradient)
        self.assertGreater(float(tf.linalg.global_norm([first_gradient])), 0.)
        with tf.GradientTape() as tape:
            joint_loss = multiview_barlow_loss([first, second])
        for gradient in tape.gradient(joint_loss, [first, second]):
            self.assertIsNotNone(gradient)
            self.assertGreater(float(tf.linalg.global_norm([gradient])), 0.)

    def test_graph_loss_and_gradients_match_eager(self) -> None:
        """Tracing preserves the view collection and all moment dependencies."""

        first = tf.constant([[0.2, 0.7], [-1.3, 2.2], [0.5, -0.4]])
        second = tf.constant([[1.1, -0.2], [0.8, 0.6], [-0.7, 1.3]])

        def objective_and_gradients(left: tf.Tensor, right: tf.Tensor) -> tuple:
            """Return one loss and both view gradients in a common traceable path."""

            with tf.GradientTape() as tape:
                tape.watch([left, right])
                loss = multiview_barlow_loss([left, right])
            return loss, tape.gradient(loss, [left, right])

        eager_loss, eager_gradients = objective_and_gradients(first, second)
        graph_loss, graph_gradients = tf.function(objective_and_gradients)(first, second)
        self.assertAlmostEqual(float(eager_loss), float(graph_loss), places=6)
        for eager, graph in zip(eager_gradients, graph_gradients):
            np.testing.assert_allclose(eager.numpy(), graph.numpy(), rtol=1e-5, atol=1e-6)

    def test_view_shapes_and_nonfinite_inputs_fail(self) -> None:
        """Malformed pairing never becomes an apparently valid representation score."""

        for views in [[], [tf.ones((2, 2))], tf.ones((2, 2, 2))]:
            with self.assertRaises(ValueError):
                multiview_barlow_loss(views)
        for views in [
            [tf.ones((2, 2)), tf.ones((3, 2))], 
            [tf.ones((2, 2)), tf.ones((2, 3))], 
            [tf.ones((0, 2)), tf.ones((0, 2))], 
            [tf.ones((2, 2)), tf.constant([[float("nan"), 0.], [1., 2.]])]
        ]:
            with self.assertRaises((ValueError, tf.errors.InvalidArgumentError)):
                multiview_barlow_loss(views)

    def test_opl_uses_mean_absolute_cosine_before_reduction(self) -> None:
        """Opposite negative directions cannot cancel as an absolute-mean loss would."""

        positive = tf.constant([[1., 0.], [1., 0.]])
        negative = tf.constant([[1., 0.], [-1., 0.]])
        loss = orthogonal_projection_loss(positive, negative)
        self.assertAlmostEqual(float(loss), 1., places=6)
        weighted = orthogonal_projection_loss(positive, negative, negative_weight=2.)
        self.assertAlmostEqual(float(weighted), 2., places=6)

    def test_opl_excludes_self_pairs_and_supports_exact_square_variant(self) -> None:
        """Opposite positives have default cost two and squared cost four."""

        positive = tf.constant([[1., 0.], [-1., 0.]])
        negative = tf.constant([[0., 1.]])
        self.assertAlmostEqual(float(orthogonal_projection_loss(positive, negative)), 2., places=6)
        self.assertAlmostEqual(float(orthogonal_projection_loss(
            positive, negative, square_loss=True
        )), 4., places=6)

    def test_official_opl_remains_distinct_from_legacy_squared_separation(self) -> None:
        """Adding the official loss does not silently change the historic objective."""

        positive = tf.constant([[1., 0.], [1., 0.]])
        negative = tf.constant([[0.5, math.sqrt(0.75)]])
        official = orthogonal_projection_loss(positive, negative)
        legacy = modulation_separation_loss(tf.concat([positive, negative], axis=0), [True, True, False])
        self.assertAlmostEqual(float(official), 0.5, places=6)
        self.assertAlmostEqual(float(legacy), 0.25, places=6)

    def test_opl_gradients_reach_both_feature_groups(self) -> None:
        """Class separation supplies finite nonzero gradients through every group."""

        positive = tf.Variable([[1., 0.2], [0.5, 1.]])
        negative = tf.Variable([[0.4, -0.7], [-0.8, 0.3]])
        with tf.GradientTape() as tape:
            loss = orthogonal_projection_loss(positive, negative)
        for gradient in tape.gradient(loss, [positive, negative]):
            self.assertIsNotNone(gradient)
            self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(gradient))))
            self.assertGreater(float(tf.linalg.global_norm([gradient])), 0.)

    def test_opl_rejects_undefined_cosines_and_missing_pair_families(self) -> None:
        """A zero vector or absent pair family cannot silently define the OPL loss."""

        for positive, negative in [
            (tf.ones((1, 2)), tf.ones((1, 2))), 
            (tf.ones((2, 2)), tf.ones((0, 2))), 
            (tf.zeros((2, 2)), tf.ones((1, 2))), 
            (tf.ones((2, 2)), tf.zeros((1, 2))), 
            (tf.ones((2, 2)), tf.ones((1, 3)))
        ]:
            with self.assertRaises((ValueError, tf.errors.InvalidArgumentError)):
                orthogonal_projection_loss(positive, negative)


# Direct invocation remains explicit; imports do not launch tests.
if __name__ == "__main__":
    unittest.main()
