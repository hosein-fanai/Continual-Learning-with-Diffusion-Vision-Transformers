"""Analytic and gradient checks for optional source-audited TMCL state invariance."""

from __future__ import annotations

import unittest

import numpy as np
import tensorflow as tf

from semantic_consolidation.objectives import (
    barlow_twins_loss, ghosh_barlow_loss, multiview_barlow_loss, state_invariance_loss
)


def _reference_pair(first: np.ndarray, second: np.ndarray, weight: float, scale: float) -> float:
    """Compute independent branch-normalized correlations entry by entry."""

    normalized = []
    for value in (first, second):
        values = np.asarray(value, dtype=np.float64)
        normalized.append((values - values.mean(axis=0)) / np.sqrt(values.var(axis=0) + 1e-5))
    left, right = normalized
    total = 0.
    for row in range(left.shape[1]):
        for column in range(left.shape[1]):
            correlation = sum(left[index, row] * right[index, column]
                              for index in range(len(left))) / len(left)
            # The same feature has the unit-correlation target.
            if row == column:
                total += (correlation - 1.) ** 2
            # Cross-feature correlations receive the independent redundancy coefficient.
            else:
                total += weight * correlation ** 2
    return float(scale * total)


class TMCLStateInvarianceObjectiveTests(unittest.TestCase):
    """Detect source-option conflation and the formerly omitted PNR repulsion."""

    def test_pair_normalization_is_per_branch_unless_explicitly_joint(self) -> None:
        """A constant view offset is removed per branch but not by joint normalization."""

        first = tf.constant([[-1.], [1.]])
        second = tf.constant([[1.], [3.]])
        per_branch = barlow_twins_loss(first, second, scale=1., epsilon=0.)
        joint = barlow_twins_loss(first, second, scale=1., normalization="joint", epsilon=0.)
        self.assertAlmostEqual(float(per_branch), 0., places=6)
        self.assertAlmostEqual(float(joint), 1., places=6)

    def test_pair_matches_independent_float64_entrywise_reference(self) -> None:
        """An asymmetric fixture checks both diagonal and ordered off-diagonal terms."""

        first = np.array([[1., -2.], [3., 0.], [-1., 5.]], dtype=np.float32)
        second = np.array([[2., 1.], [0., 3.], [4., -1.]], dtype=np.float32)
        expected = _reference_pair(first, second, 0.13, 0.1)
        actual = barlow_twins_loss(first, second, off_diagonal_weight=0.13, scale=0.1)
        self.assertAlmostEqual(float(actual), expected, places=6)

    def test_ghosh_reduction_is_not_the_main_multiview_reduction(self) -> None:
        """Opposite views give one averaged Ghosh penalty versus two main penalties."""

        first = tf.constant([[-1.], [1.]])
        self.assertAlmostEqual(float(ghosh_barlow_loss([first, -first], scale=1., epsilon=0.)), 1., places=6)
        self.assertAlmostEqual(float(multiview_barlow_loss([first, -first], scale=1., epsilon=0.)), 2., places=6)
        shifted = first + 4.
        self.assertAlmostEqual(float(ghosh_barlow_loss([first, shifted], scale=1., epsilon=0.)), 0., places=6)

    def test_ghosh_covariance_of_view_mean_is_penalized_once(self) -> None:
        """Identical correlated channels have two off-diagonal entries, regardless of K."""

        view = tf.constant([[-1., -1.], [1., 1.]])
        for count in (2, 4):
            loss = ghosh_barlow_loss([view] * count, off_diagonal_weight=0.25, scale=1., epsilon=0.)
            self.assertAlmostEqual(float(loss), 0.5, places=6)

    def test_barlow_si_uses_only_two_views_while_ghosh_uses_every_pair(self) -> None:
        """Additional source Barlow views are intentionally not part of its SI loss."""

        first = tf.constant([[-1.], [1.]])
        predictions = [first, first, -first, -first]
        targets = [first] * 4
        barlow = state_invariance_loss(predictions, targets, scale=1., epsilon=0.)
        ghosh = state_invariance_loss(predictions, targets, objective="ghosh", scale=1., epsilon=0.)
        self.assertAlmostEqual(float(barlow), 0., places=6)
        self.assertAlmostEqual(float(ghosh), 0.5, places=6)

    def test_pnr_barlow_actually_adds_the_cross_view_repulsion(self) -> None:
        """This directly rejects the pinned branch's computed-but-discarded PNR term."""

        view = tf.eye(2)
        cassle = state_invariance_loss([view, view], [view, view])
        pnr = state_invariance_loss([view, view], [view, view], method="pnr", negative_weight=0.7)
        self.assertAlmostEqual(float(pnr - cassle), 0.7, places=6)
        opposite = state_invariance_loss([view, view], [-view, -view], method="pnr", negative_weight=0.7)
        opposite_base = state_invariance_loss([view, view], [-view, -view])
        self.assertAlmostEqual(float(opposite - opposite_base), -0.7, places=6)

    def test_pnr_ghosh_uses_both_declared_crossed_pairs(self) -> None:
        """The second crossed pair cancels the first, rather than being ignored."""

        view = tf.eye(2)
        predictions = [view] * 4
        targets = [view, view, -view, -view]
        baseline = state_invariance_loss(predictions, targets, objective="ghosh")
        pnr = state_invariance_loss(predictions, targets, method="pnr", objective="ghosh")
        self.assertAlmostEqual(float(pnr), float(baseline), places=6)

    def test_targets_are_frozen_and_prediction_gradients_are_finite(self) -> None:
        """Every optional SI mode differentiates only through current predictions."""

        for method in ("cassle", "pnr"):
            for objective in ("barlow", "ghosh"):
                predictions = [tf.Variable([[0.2, 0.7], [-1.3, 2.2], [0.5, -0.4]]) for _ in range(4)]
                targets = [tf.Variable([[1.1, -0.2], [0.8, 0.6], [-0.7, 1.3]]) for _ in range(4)]
                with tf.GradientTape() as tape:
                    loss = state_invariance_loss(predictions, targets, method=method, objective=objective)
                gradients = tape.gradient(loss, predictions + targets)
                self.assertTrue(all(value is None for value in gradients[4:]))
                active_count = 2 if objective == "barlow" else 4
                for gradient in gradients[:active_count]:
                    self.assertIsNotNone(gradient)
                    self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(gradient))))
                    self.assertGreater(float(tf.linalg.global_norm([gradient])), 0.)
                self.assertTrue(all(value is None for value in gradients[active_count:4]))

    def test_pnr_gradient_matches_finite_difference(self) -> None:
        """The correction has the declared repulsive sign in the actual gradient."""

        prediction = tf.constant([[0.2, 0.7], [-1.3, 2.2], [0.5, -0.4]])
        target = tf.constant([[1.1, -0.2], [0.8, 0.6], [-0.7, 1.3]])

        def objective(value: tf.Tensor) -> tf.Tensor:
            """Use asymmetric paired and crossed frozen targets for a numeric gradient."""

            return state_invariance_loss([value, value + 0.3], [target, -target], method="pnr")

        analytical, numerical = tf.test.compute_gradient(objective, [prediction], delta=1e-3)
        np.testing.assert_allclose(analytical[0], numerical[0], rtol=3e-3, atol=3e-3)

    def test_pnr_zero_vectors_use_source_cosine_epsilon_convention(self) -> None:
        """Unlike OPL's dependency, SimSiam cosine explicitly defines zero-vector scores."""

        zeros = tf.zeros((2, 3))
        loss = state_invariance_loss([zeros, zeros], [zeros, zeros], method="pnr", scale=1.)
        self.assertAlmostEqual(float(loss), 3., places=6)

    def test_invalid_modes_pairing_or_branch_batch_fail(self) -> None:
        """Scientific mode identity and view correspondence cannot silently change."""

        view = tf.eye(2)
        for options in [{"method": "unknown"}, {"objective": "unknown"}]:
            with self.assertRaises(ValueError):
                state_invariance_loss([view, view], [view, view], **options)
        with self.assertRaises(ValueError):
            state_invariance_loss([view, view], [view])
        with self.assertRaises(ValueError):
            state_invariance_loss([view, view], [view, view], method="pnr", objective="ghosh")
        with self.assertRaises(ValueError):
            barlow_twins_loss(view, view, normalization="unknown")
        with self.assertRaises(tf.errors.InvalidArgumentError):
            barlow_twins_loss(view[:1], view[:1])


# Test execution is explicit and remains confined to the selected remote process.
if __name__ == "__main__":
    unittest.main()
