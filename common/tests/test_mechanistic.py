"""Regression checks for sparse-column and one-hot mechanistic metric labels.

Explicit probability and label arrays compare column-vector labels with equivalent sparse
IDs, retain one-class one-hot behavior, and check replay diagnostic consistency. These local
numerical tests neither train a model nor create artifacts.

Inputs are fixtures constructed by the test methods and their helpers. Tests return no
application result: unittest records assertion outcomes and errors. Run this module directly
or through ``python -m unittest`` discovery. Importing it defines fixtures and cases; it
does not itself start a test run.
"""

from __future__ import annotations

import unittest

import numpy as np

from common.mechanistic import (
    calibration_metrics, linear_cka, replay_quality_metrics, select_replay_candidates,
)


class MechanisticLabelTests(unittest.TestCase):
    """Sparse columns must retain the same class IDs as sparse vectors.

    The unittest runner executes the selected test method with its local fixtures;
    individual methods describe the configurations and failure cases they exercise. There is
    no application model or experiment result returned by constructing this test case.

    Args:
        methodName (str): Test method selected by unittest. Defaults to ``"runTest"``;
            discovery supplies each named ``test_*`` method.

    Attributes:
        _testMethodName (str): Selected method name maintained by unittest.
    """

    def test_sparse_columns_and_onehot_labels_have_identical_calibration(self) -> None:
        """Perfect predictions have zero calibration error under either encoding.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        probabilities = np.asarray([[1., 0.], [0., 1.], [0., 1.]])
        labels = np.asarray([0, 1, 1])
        expected = {"accuracy": 1., "entropy": 0., "nll": 0., "brier": 0., "ece": 0.}
        for encoding in (labels, labels[:, None], probabilities):
            with self.subTest(shape=encoding.shape):
                self.assertEqual(calibration_metrics(probabilities, encoding), expected)

    def test_sparse_column_replay_keeps_its_conditioning_class(self) -> None:
        """A pool for class one remains class one during coverage reporting.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        result = replay_quality_metrics(
            np.asarray([[0., 1.], [1., 0.]]),
            np.asarray([[1], [1]]),
            expected_classes=[1],
        )
        self.assertEqual(result["class_coverage"], 1.)
        self.assertEqual(result["normalized_label_entropy"], 1.)
        self.assertEqual(result["class_counts"], {"1": 2})

    def test_uniform_replay_randomizes_remainder_classes_reproducibly(self) -> None:
        """Give every class access to an undersized or remainder replay budget.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Seeded selections repeat and remainder recipients span all classes.
        """

        labels = np.repeat(np.arange(4), 3)
        samples = np.arange(len(labels))[:, None]
        for budget in (1, 3, 5):
            recipients = set()
            for seed in range(32):
                first = select_replay_candidates(
                    samples, labels, budget, strategy="uniform", seed=seed,
                )
                repeated = select_replay_candidates(
                    samples, labels, budget, strategy="uniform", seed=seed,
                )
                np.testing.assert_array_equal(first[0], repeated[0])
                counts = np.bincount(first[1], minlength=4)
                self.assertEqual(len(first[0]), budget)
                self.assertLessEqual(int(counts.max() - counts.min()), 1)
                recipients.update(np.flatnonzero(counts > budget // 4).tolist())
            self.assertEqual(recipients, {0, 1, 2, 3})

    def test_first_task_one_class_onehot_calibration(self) -> None:
        """A one-class probability/target matrix still represents class zero.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        result = calibration_metrics(np.ones((2, 1)), np.ones((2, 1)))
        self.assertEqual(result["accuracy"], 1.)
        self.assertEqual(result["nll"], 0.)

    def test_invalid_metric_labels_and_calibration_settings_fail(self) -> None:
        """Reject ambiguous identities and undefined calibration partitions.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """

        for labels in ([.75], [True], [2 ** 63], [-1], [[1., 0., 0.]]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                calibration_metrics([[.9, .1]], labels)
        for settings in ({"bins": 2.5}, {"bins": True}, {"bins": 0},
                         {"epsilon": float("nan")}, {"epsilon": float("inf")},
                         {"epsilon": 1.1}, {"epsilon": 0.}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                calibration_metrics([[.9, .1]], [0], **settings)
        largest_id = np.asarray([np.iinfo(np.int64).max], dtype="int64")
        _, retained, _ = select_replay_candidates([[1.]], largest_id, 1)
        np.testing.assert_array_equal(retained, largest_id)

    def test_cka_preserves_scale_and_orthogonal_invariance(self) -> None:
        """Both CKA algorithms agree with centered Gram alignment at extreme scales.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """

        rng = np.random.default_rng(7)
        for width in (3, 12):
            x, y = rng.normal(size=(8, width)), rng.normal(size=(8, width))
            rotation, _ = np.linalg.qr(rng.normal(size=(width, width)))
            centered_x, centered_y = x - x.mean(axis=0), y - y.mean(axis=0)
            gram_x, gram_y = centered_x @ centered_x.T, centered_y @ centered_y.T
            expected = np.sum(gram_x * gram_y) / (np.linalg.norm(gram_x) * np.linalg.norm(gram_y))
            for scale_x, scale_y in ((1., 1.), (1e100, 1e-100), (1e-100, 1e100)):
                with self.subTest(width=width, scales=(scale_x, scale_y)):
                    self.assertAlmostEqual(linear_cka(x * scale_x, (y @ rotation) * scale_y), expected, places=12)
            self.assertAlmostEqual(linear_cka(x, x @ rotation + 7.), 1., places=12)
        self.assertTrue(np.isnan(linear_cka(np.ones((3, 2)), np.ones((3, 2)))))
        for invalid in (np.empty((3, 0)), np.full((3, 2), np.nan), np.full((3, 2), np.inf)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                linear_cka(invalid, np.ones((3, 2)))


# Run this module's tests when executed directly.
if __name__ == "__main__":
    unittest.main()
