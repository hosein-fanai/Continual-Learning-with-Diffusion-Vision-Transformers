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
    calibration_metrics, replay_quality_metrics, select_replay_candidates,
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


# Run this module's tests when executed directly.
if __name__ == "__main__":
    unittest.main()
