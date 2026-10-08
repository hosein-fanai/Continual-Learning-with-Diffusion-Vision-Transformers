"""Verify automatic and recorded epoch alignment in generic history plots."""

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
    """Keep figures and exports aligned without metric-specific caller workarounds."""

    def tearDown(self) -> None:
        """Release figures created while exercising the public plotting interface."""

        plt.close("all")

    def plot_lines(self, history: dict, **options: object) -> list[list[tuple[list, list]]]:
        """Return every visible panel's plotted x/y values through the public interface."""

        with patch("matplotlib.pyplot.show"):
            plot_history(history, col=1, show_plots=True, **options)
        return [
            [(line.get_xdata().tolist(), line.get_ydata().tolist()) for line in axis.lines]
            for axis in plt.gcf().axes
            if axis.get_visible()
        ]

    def test_plain_history_accepts_integer_and_list_validation_schedules(self) -> None:
        """A supplied Keras schedule locates sparse observations within the completed fit."""

        history = {"arbitrary_score": [5., 4., 3., 2., 1.], "val_arbitrary_score": [4.5, 2.5]}
        cases = ((0, 2, [2, 4]), (2, 2, [4, 6]), (0, [1, 4], [1, 4]))
        for initial_epoch, validation_freq, expected in cases:
            with self.subTest(initial_epoch=initial_epoch, validation_freq=validation_freq):
                epochs = _history_metric_epochs(
                    history, None, validation_freq=validation_freq, initial_epoch=initial_epoch
                )
                self.assertEqual(epochs["arbitrary_score"].tolist(), 
                                 list(range(initial_epoch + 1, initial_epoch + 6)))
                self.assertEqual(epochs["val_arbitrary_score"].tolist(), expected)

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

    def test_dense_pairs_ignore_sparse_schedule_and_preserve_slice_steps(self) -> None:
        """Equal-length metrics remain dense even when a cadence option is supplied."""

        history = {"custom_measure": [5., 4., 3., 2., 1.], 
                   "val_custom_measure": [5.5, 4.5, 3.5, 2.5, 1.5]}
        for validation_freq in (None, 5, [5]):
            for initial_epoch in (0, 5):
                with self.subTest(validation_freq=validation_freq, initial_epoch=initial_epoch):
                    epochs = _history_metric_epochs(
                        history, None, validation_freq=validation_freq, initial_epoch=initial_epoch
                    )
                    expected = list(range(initial_epoch + 1, initial_epoch + 6))
                    self.assertEqual(epochs["custom_measure"].tolist(), expected)
                    self.assertEqual(epochs["val_custom_measure"].tolist(), expected)
                    panels = self.plot_lines(
                        history, range_=(None, None, 2), 
                        validation_freq=validation_freq, initial_epoch=initial_epoch
                    )
                    self.assertEqual(panels, [[
                        (expected[::2], history["custom_measure"][::2]), 
                        (expected[::2], history["val_custom_measure"][::2])
                    ]])

    def test_arbitrary_metric_pairs_infer_independent_validation_cadences(self) -> None:
        """Each ordinary or merged pair derives its own axis from complete series lengths."""

        history = {
            "custom_measure": list(range(50)), 
            "val_custom_measure": list(range(100, 110)), 
            "unseen_phase_signal": list(range(12)), 
            "unseen_phase_val_signal": [0.4, 0.3, 0.2]
        }
        original = deepcopy(history)
        epochs = _history_metric_epochs(history, None)
        self.assertEqual(epochs["val_custom_measure"].tolist(), list(range(5, 51, 5)))
        self.assertEqual(epochs["unseen_phase_val_signal"].tolist(), [4, 8, 12])
        panels = self.plot_lines(history)
        self.assertEqual(panels, [
            [(list(range(1, 51)), history["custom_measure"]), 
             (list(range(5, 51, 5)), history["val_custom_measure"])], 
            [(list(range(1, 13)), history["unseen_phase_signal"]), 
             ([4, 8, 12], history["unseen_phase_val_signal"])]
        ])
        self.assertEqual(history, original)

    def test_partial_range_preserves_all_nine_later_validation_points(self) -> None:
        """A start at epoch index nine selects epochs ten through fifty for both series."""

        history = {"custom_measure": list(range(50)), "val_custom_measure": list(range(100, 110))}
        original = deepcopy(history)
        for validation_freq in (None, 5):
            with self.subTest(validation_freq=validation_freq), tempfile.TemporaryDirectory() as directory:
                csv_path = Path(directory) / "history.csv"
                panels = self.plot_lines(
                    history, range_=(9, None), csv_path=csv_path, validation_freq=validation_freq
                )
                table = pd.read_csv(csv_path)
                self.assertEqual(panels, [[
                    (list(range(10, 51)), history["custom_measure"][9:]), 
                    (list(range(10, 51, 5)), history["val_custom_measure"][1:])
                ]])
                self.assertEqual(table.epoch.tolist(), list(range(1, 51)))
                self.assertEqual(table.loc[table.val_custom_measure.notna(), "epoch"].tolist(), 
                                 list(range(5, 51, 5)))
                self.assertEqual(table.val_custom_measure.dropna().tolist(), history["val_custom_measure"])
        self.assertEqual(history, original)

    def test_validation_only_view_uses_the_paired_epoch_interval(self) -> None:
        """Requesting only validation still applies the range to training epoch positions."""

        history = {"custom_measure": list(range(50)), "val_custom_measure": list(range(100, 110))}
        panels = self.plot_lines(history, range_=(9, None), metrics=["val_custom_measure"])
        self.assertEqual(panels, [[
            (list(range(10, 51, 5)), history["val_custom_measure"][1:])
        ]])

    def test_empty_validation_series_remain_empty_without_hiding_training(self) -> None:
        """An unobserved validation metric creates no invented dates or plotted observations."""

        history = {"custom_measure": [3., 2., 1.], "val_custom_measure": [], "val_unmatched": []}
        epochs = _history_metric_epochs(history, None)
        self.assertEqual(epochs["val_custom_measure"].tolist(), [])
        self.assertEqual(epochs["val_unmatched"].tolist(), [])
        panels = self.plot_lines(history, metrics=["custom_measure"])
        self.assertEqual(panels, [[([1, 2, 3], [3., 2., 1.])]])

    def test_unmatched_validation_uses_longest_training_axis_before_range_filtering(self) -> None:
        """A validation-only diagnostic uses the run timeline even beside a shorter phase."""

        history = {
            "short_phase": [3., 2.], 
            "reference_measure": list(range(50)), 
            "val_unpaired_probe": list(range(100, 110))
        }
        for validation_freq in (None, 5):
            with self.subTest(validation_freq=validation_freq):
                epochs = _history_metric_epochs(history, None, validation_freq=validation_freq)
                self.assertEqual(epochs["val_unpaired_probe"].tolist(), list(range(5, 51, 5)))
                panels = self.plot_lines(
                    history, range_=(9, None), metrics=["val_unpaired_probe"], 
                    validation_freq=validation_freq
                )
                self.assertEqual(panels, [[
                    (list(range(10, 51, 5)), history["val_unpaired_probe"][1:])
                ]])

    def test_standalone_validation_uses_dense_or_supplied_cadence(self) -> None:
        """Validation without a training series still plots every observed value."""

        history = {"val_custom_probe": [0.7, 0.5, 0.2]}
        cases = ((None, 0, [1, 2, 3]), (None, 3, [4, 5, 6]), 
                 (5, 0, [5, 10, 15]), (5, 3, [5, 10, 15]))
        for validation_freq, initial_epoch, expected in cases:
            with self.subTest(validation_freq=validation_freq, initial_epoch=initial_epoch):
                epochs = _history_metric_epochs(
                    history, None, validation_freq=validation_freq, initial_epoch=initial_epoch
                )
                self.assertEqual(epochs["val_custom_probe"].tolist(), expected)
                panels = self.plot_lines(
                    history, validation_freq=validation_freq, initial_epoch=initial_epoch
                )
                self.assertEqual(panels, [[(expected, history["val_custom_probe"])]])

    def test_nondivisible_lengths_use_regular_integer_cadence_estimate(self) -> None:
        """Five training rows and two validation rows infer a two-epoch interval."""

        history = {"custom_probe": [5., 4., 3., 2., 1.], "val_custom_probe": [4.5, 2.5]}
        epochs = _history_metric_epochs(history, None)
        self.assertEqual(epochs["val_custom_probe"].tolist(), [2, 4])
        self.assertEqual(self.plot_lines(history), [[
            ([1, 2, 3, 4, 5], history["custom_probe"]), 
            ([2, 4], history["val_custom_probe"])
        ]])

    def test_supplied_integer_cadence_keeps_all_observation_coordinates(self) -> None:
        """A supplied interval dates every value without rejecting a count mismatch."""

        history = {"custom_probe": [5., 4., 3., 2., 1.], "val_custom_probe": [4.5, 2.5]}
        epochs = _history_metric_epochs(history, None, validation_freq=3)
        self.assertEqual(epochs["val_custom_probe"].tolist(), [3, 6])
        self.assertEqual(self.plot_lines(history, validation_freq=3), [[
            ([1, 2, 3, 4, 5], history["custom_probe"]), 
            ([3, 6], history["val_custom_probe"])
        ]])
        self.assertEqual(self.plot_lines(history, range_=(0, 5), validation_freq=3), [[
            ([1, 2, 3, 4, 5], history["custom_probe"]), 
            ([3], [4.5])
        ]])

    def test_standalone_sparse_validation_range_uses_epoch_span(self) -> None:
        """A standalone sparse series uses epoch positions instead of observation indices."""

        history = {"val_custom_probe": list(range(100, 110))}
        panels = self.plot_lines(history, range_=(9, None), validation_freq=5)
        self.assertEqual(panels, [[
            (list(range(10, 51, 5)), history["val_custom_probe"][1:])
        ]])

    def test_explicit_range_stop_limits_sparse_observations_by_epoch(self) -> None:
        """A bounded epoch slice trims paired x/y observations at the same boundaries."""

        history = {"custom_measure": list(range(50)), "val_custom_measure": list(range(100, 110))}
        self.assertEqual(self.plot_lines(history, range_=(9, 21)), [[
            (list(range(10, 22)), history["custom_measure"][9:21]), 
            ([10, 15, 20], history["val_custom_measure"][1:4])
        ]])

    def test_sparse_validation_ignores_training_slice_step_within_epoch_bounds(self) -> None:
        """A stepped training view keeps every sparse validation observation inside its bounds."""

        history = {"custom_measure": list(range(50)), "val_custom_measure": list(range(100, 110))}
        cases = (
            ((8, None, 2), list(range(9, 50, 2)), list(range(8, 50, 2)), 
             list(range(10, 51, 5)), list(range(101, 110))), 
            ((8, 20, 2), [9, 11, 13, 15, 17, 19], [8, 10, 12, 14, 16, 18], 
             [10, 15, 20], [101, 102, 103])
        )
        for validation_freq in (None, 5):
            for selected_range, training_x, training_y, validation_x, validation_y in cases:
                with self.subTest(validation_freq=validation_freq, selected_range=selected_range):
                    panels = self.plot_lines(
                        history, range_=selected_range, validation_freq=validation_freq
                    )
                    self.assertEqual(panels, [[
                        (training_x, training_y), (validation_x, validation_y)
                    ]])

    def test_sparse_validation_follows_reverse_epoch_ranges(self) -> None:
        """Reverse views retain sparse observations in the same direction as training."""

        history = {"custom_measure": list(range(10)), "val_custom_measure": [100, 101]}
        cases = (
            ((9, 1, -1), list(range(10, 2, -1)), list(range(9, 1, -1))), 
            ((9, None, -2), [10, 8, 6, 4, 2], [9, 7, 5, 3, 1])
        )
        for validation_freq in (None, 5):
            for selected_range, training_x, training_y in cases:
                with self.subTest(validation_freq=validation_freq, selected_range=selected_range):
                    panels = self.plot_lines(
                        history, range_=selected_range, validation_freq=validation_freq
                    )
                    self.assertEqual(panels, [[
                        (training_x, training_y), ([10, 5], [101, 100])
                    ]])

    def test_standalone_explicit_epochs_override_later_start_cursor(self) -> None:
        """Recorded validation dates remain visible even when an ordinary fit cursor is later."""

        history = {"val_custom_probe": [0.7, 0.2]}
        panels = self.plot_lines(
            history, metric_epochs={"val_custom_probe": [5, 10]}, initial_epoch=10
        )
        self.assertEqual(panels, [[([5, 10], [0.7, 0.2])]])

    def test_empty_selected_ranges_do_not_restore_earlier_points(self) -> None:
        """A range beyond a short series produces empty lines rather than resetting the slice."""

        history = {"custom_measure": [3., 2., 1.], "val_custom_measure": [3.5, 2.5, 1.5]}
        for selected_range in ((9, None), (2, 2)):
            with self.subTest(selected_range=selected_range):
                self.assertEqual(self.plot_lines(history, range_=selected_range), [[([], []), ([], [])]])
        self.assertEqual(self.plot_lines({"custom_measure": [], "val_custom_measure": []}), [[([], [])]])

    def test_explicit_epoch_list_requires_one_coordinate_per_observation(self) -> None:
        """An explicit list remains a structural observation-to-coordinate mapping."""

        history = {"custom_probe": [5., 4., 3., 2., 1.], "val_custom_probe": [4.5, 2.5]}
        with self.assertRaises(ValueError):
            _history_metric_epochs(history, None, validation_freq=[3])

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
            values = [line.get_ydata().tolist() for line in plt.gcf().axes[0].lines]
            table = pd.read_csv(csv_path)
        self.assertEqual(coordinates, [[4, 5, 6, 7], [4, 6]])
        self.assertEqual(values, [[4., 3., 2., 1.], [4.5, 2.5]])
        self.assertEqual(table.epoch.tolist(), [3, 4, 5, 6, 7])
        self.assertEqual(table.loc[table.val_loss.notna(), "epoch"].tolist(), [4, 6])
        self.assertTrue(table.loc[table.epoch.isin([3, 5, 7]), "val_loss"].isna().all())
        self.assertEqual(history, original)


# Execute only these focused plotting regressions when the file is run directly.
if __name__ == "__main__":
    unittest.main()
