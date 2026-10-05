"""Verify plotting from observed epochs or the caller's actual Keras fit schedule."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib


matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from common.train import _RecordedHistory
from common.utils import _history_metric_epochs, plot_history


class HistoryPlotScheduleTests(unittest.TestCase):
    """Keep plots and exports aligned without manual metric-coordinate preprocessing."""

    def tearDown(self) -> None:
        """Release figures created while exercising the public plotting interface."""

        plt.close("all")

    def test_plain_history_accepts_integer_and_list_validation_schedules(self) -> None:
        """A supplied Keras schedule locates sparse observations within the completed fit."""

        history = {"loss": [5., 4., 3., 2., 1.], "val_loss": [4.5, 2.5]}
        cases = ((0, 2, [2, 4]), (2, 2, [4, 6]), (0, [1, 4], [1, 4]))
        for initial_epoch, validation_freq, expected in cases:
            with self.subTest(initial_epoch=initial_epoch, validation_freq=validation_freq):
                epochs = _history_metric_epochs(
                    history, None, validation_freq=validation_freq, initial_epoch=initial_epoch
                )
                self.assertEqual(epochs["loss"].tolist(), list(range(initial_epoch + 1, initial_epoch + 6)))
                self.assertEqual(epochs["val_loss"].tolist(), expected)

    def test_observed_epochs_take_precedence_over_fit_schedule(self) -> None:
        """An irregular recorded diagnostic retains its actual dates despite ordinary fit options."""

        history = _RecordedHistory(
            {"loss": [4., 3., 2., 1.], "val_auxiliary": [0.4, 0.1]}, 
            {"loss": [3, 4, 5, 6], "val_auxiliary": [3, 6]}
        )
        epochs = _history_metric_epochs(history, None, validation_freq=2, initial_epoch=0)
        self.assertEqual(epochs["loss"].tolist(), [3, 4, 5, 6])
        self.assertEqual(epochs["val_auxiliary"].tolist(), [3, 6])

    def test_recorded_history_requires_no_additional_plot_arguments(self) -> None:
        """The recorded mapping supplies sparse and resumed coordinates by itself."""

        history = _RecordedHistory(
            {"loss": [3., 2., 1.], "val_loss": [0.5]}, 
            {"loss": [6, 7, 8], "val_loss": [7]}
        )
        epochs = _history_metric_epochs(history, None)
        self.assertEqual(epochs["loss"].tolist(), [6, 7, 8])
        self.assertEqual(epochs["val_loss"].tolist(), [7])

    def test_explicit_override_preserves_other_observed_axes_and_inputs(self) -> None:
        """Per-metric overrides retain unrelated metadata without mutating either input mapping."""

        history = _RecordedHistory(
            {"loss": [4., 3., 2., 1.], "val_loss": [0.4, 0.1]}, 
            {"loss": [3, 4, 5, 6], "val_loss": [3, 6]}
        )
        explicit = {"val_loss": [4, 5]}
        original_values = deepcopy(dict(history))
        original_observed = deepcopy(history.metric_epochs)
        original_explicit = deepcopy(explicit)
        epochs = _history_metric_epochs(history, explicit, validation_freq=2, initial_epoch=0)
        self.assertEqual(epochs["loss"].tolist(), [3, 4, 5, 6])
        self.assertEqual(epochs["val_loss"].tolist(), [4, 5])
        self.assertEqual(history, original_values)
        self.assertEqual(history.metric_epochs, original_observed)
        self.assertEqual(explicit, original_explicit)

    def test_schedule_fills_only_missing_recorded_coordinates(self) -> None:
        """An incomplete metadata mapping supplies the training axis for scheduled validation."""

        history = _RecordedHistory(
            {"loss": [4., 3., 2., 1.], "val_loss": [0.4, 0.1]}, 
            {"loss": [3, 4, 5, 6]}
        )
        epochs = _history_metric_epochs(history, None, validation_freq=2)
        self.assertEqual(epochs["loss"].tolist(), [3, 4, 5, 6])
        self.assertEqual(epochs["val_loss"].tolist(), [4, 6])

    def test_dense_defaults_and_resumed_axes_remain_aligned(self) -> None:
        """Dense partners keep the historical default and share a supplied resumed cursor."""

        history = {"loss": [3., 2., 1.], "val_loss": [3.5, 2.5, 1.5]}
        for initial_epoch in (0, 5):
            with self.subTest(initial_epoch=initial_epoch):
                epochs = _history_metric_epochs(history, None, initial_epoch=initial_epoch)
                expected = list(range(initial_epoch + 1, initial_epoch + 4))
                self.assertEqual(epochs["loss"].tolist(), expected)
                self.assertEqual(epochs["val_loss"].tolist(), expected)

    def test_auxiliary_validation_uses_only_an_unambiguous_training_axis(self) -> None:
        """Validation-only diagnostics need a common fit interval before a schedule can date them."""

        history = {"loss": [4., 3., 2., 1.], "noise_loss": [3., 2., 1., 0.], 
                   "val_auxiliary": [0.4, 0.1]}
        epochs = _history_metric_epochs(history, None, validation_freq=2, initial_epoch=2)
        self.assertEqual(epochs["val_auxiliary"].tolist(), [4, 6])
        history["noise_loss"] = [1.]
        with self.assertRaisesRegex(ValueError, "metric_epochs"):
            _history_metric_epochs(history, None, validation_freq=2)

    def test_supplied_schedule_must_match_observation_count(self) -> None:
        """A mismatching declared cadence cannot be replaced by a ratio of metric lengths."""

        history = {"loss": [5., 4., 3., 2., 1.], "val_loss": [4.5, 2.5]}
        with self.assertRaisesRegex(ValueError, "metric_epochs|cadence|schedule"):
            _history_metric_epochs(history, None, validation_freq=3)

    def test_sparse_plain_history_without_schedule_remains_ambiguous(self) -> None:
        """Omitting metadata and cadence cannot authenticate sparse or standalone validation dates."""

        histories = ({"loss": [3., 2., 1.], "val_loss": [0.5]}, {"val_loss": [0.5]})
        for history in histories:
            with self.subTest(history=history), self.assertRaisesRegex(ValueError, "metric_epochs"):
                _history_metric_epochs(history, None)

    def test_invalid_fit_schedule_coordinates_are_rejected(self) -> None:
        """Only integer Keras epoch identities can date recorded observations."""

        history = {"loss": [3., 2., 1.], "val_loss": [0.5]}
        frequencies = (0, -1, True, 1.5, "2", [0], [-1], [True], [1.5], ["2"])
        for validation_freq in frequencies:
            with self.subTest(validation_freq=validation_freq), self.assertRaises(ValueError):
                _history_metric_epochs(history, None, validation_freq=validation_freq)
        for initial_epoch in (-1, True, 0.5):
            with self.subTest(initial_epoch=initial_epoch), self.assertRaises(ValueError):
                _history_metric_epochs(history, None, validation_freq=3, initial_epoch=initial_epoch)

    def test_numpy_integer_schedule_preserves_exact_epochs(self) -> None:
        """NumPy integer fit options have the same epoch identity as Python integers."""

        epochs = _history_metric_epochs(
            {"loss": [3., 2., 1.], "val_loss": [0.5]}, None, 
            validation_freq=np.int64(3), initial_epoch=np.int64(3)
        )
        self.assertEqual(epochs["loss"].tolist(), [4, 5, 6])
        self.assertEqual(epochs["val_loss"].tolist(), [6])

    def test_public_plot_and_csv_share_supplied_schedule(self) -> None:
        """One plotting call dates resumed sparse validation while CSV retains the full run."""

        history = {"loss": [5., 4., 3., 2., 1.], "val_loss": [4.5, 2.5]}
        original = deepcopy(history)
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "history.csv"
            with patch("matplotlib.pyplot.show"):
                plot_history(
                    history, range_=(1, None), col=1, show_plots=True, csv_path=csv_path, 
                    validation_freq=2, initial_epoch=2
                )
            coordinates = [line.get_xdata().tolist() for line in plt.gcf().axes[0].lines]
            table = pd.read_csv(csv_path)
        self.assertEqual(coordinates, [[4, 5, 6, 7], [4, 6]])
        self.assertEqual(table.epoch.tolist(), [3, 4, 5, 6, 7])
        self.assertEqual(table.loc[table.val_loss.notna(), "epoch"].tolist(), [4, 6])
        self.assertTrue(table.loc[table.epoch.isin([3, 5, 7]), "val_loss"].isna().all())
        self.assertEqual(history, original)


# Execute only these focused plotting regressions when the file is run directly.
if __name__ == "__main__":
    unittest.main()
