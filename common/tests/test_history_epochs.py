"""Verify recorded and inferred epoch coordinates in history figures and CSV reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib


matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from common.train import report
from common.utils import plot_history


class HistoryEpochTests(unittest.TestCase):
    """Exercise recorded and inferred axes across dense, sparse, resumed and merged histories."""

    def tearDown(self) -> None:
        """Release every temporary Matplotlib figure after a passing or failing case."""

        plt.close("all")

    def _render(
        self, 
        history: Mapping[str, Sequence[float]], 
        metric_epochs: Mapping[str, Sequence[int]] | None = None, 
        **options: object
    ) -> tuple[list[list[int]], pd.DataFrame]:
        """Render one subplot and capture its plotted coordinates plus complete CSV.

        Args:
            history (Mapping[str, Sequence[float]]): Synthetic ordered scalar observations.
            metric_epochs (Mapping[str, Sequence[int]] | None): Explicit epoch coordinates.
            **options (object): Additional plot_history selection/range arguments.

        Returns:
            tuple[list[list[int]], pd.DataFrame]: X coordinates for each plotted line
                and the unsliced CSV table read from a removed temporary directory.
        """

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            with patch("matplotlib.pyplot.show"):
                plot_history(
                    history, col=1, show_plots=True, csv_path=path, 
                    metric_epochs=metric_epochs, **options
                )
            coordinates = [line.get_xdata().tolist() for line in plt.gcf().axes[0].lines]
            return coordinates, pd.read_csv(path)

    def test_sparse_validation_plot_and_csv_share_actual_epochs(self) -> None:
        """Five training epochs with frequency two place validation only at epochs two/four."""

        coordinates, table = self._render(
            {"loss": [5., 4., 3., 2., 1.], "val_loss": [4.5, 2.5]}, 
            {"val_loss": [2, 4]}
        )
        self.assertEqual(coordinates, [[1, 2, 3, 4, 5], [2, 4]])
        self.assertEqual(table.loc[table.val_loss.notna(), "epoch"].tolist(), [2, 4])
        self.assertTrue(table.loc[table.epoch.isin([1, 3, 5]), "val_loss"].isna().all())

    def test_validation_only_partial_view_uses_resumed_training_interval(self) -> None:
        """A validation-only panel slices its training reference while CSV keeps all observations."""

        coordinates, table = self._render(
            {"loss": [6., 5., 4., 3., 2., 1.], "val_loss": [5.5, 2.5, 0.5]}, 
            {"loss": [5, 6, 7, 8, 9, 10], "val_loss": [5, 8, 10]}, 
            metrics=["val_loss"], range_=(2, None)
        )
        self.assertEqual(coordinates, [[8, 10]])
        self.assertEqual(table.epoch.tolist(), [5, 6, 7, 8, 9, 10])
        self.assertEqual(table.loc[table.val_loss.notna(), "epoch"].tolist(), [5, 8, 10])

    def test_dense_validation_preserves_slice_step(self) -> None:
        """Dense paired histories preserve the same positional slicing and step as before."""

        coordinates, _ = self._render(
            {"loss": [5., 4., 3., 2., 1.], "val_loss": [6., 5., 4., 3., 2.]}, 
            range_=(None, None, 2)
        )
        self.assertEqual(coordinates, [[1, 3, 5], [1, 3, 5]])

    def test_sparse_and_standalone_validation_infer_generic_coordinates(self) -> None:
        """Plain histories infer a regular cadence for arbitrary validation metric names."""

        cases = (
            ({"custom_score": [3., 2., 1.], "val_custom_score": [1.5]}, [[1, 2, 3], [3]]), 
            ({"val_custom_score": [1.5]}, [[1]]), 
            ({"custom_phase_score": [3., 2., 1.], "custom_phase_val_score": [1.5]}, 
             [[1, 2, 3], [3]])
        )
        for history, expected in cases:
            with self.subTest(history=history):
                coordinates, _ = self._render(history)
                self.assertEqual(coordinates, expected)

    def test_merged_phase_validation_uses_explicit_coordinates(self) -> None:
        """V2 merged validation names share the correct phase subplot and actual sparse epoch."""

        coordinates, table = self._render(
            {"generator_loss": [3., 2., 1.], "generator_val_loss": [1.5]}, 
            {"generator_val_loss": [2]}
        )
        self.assertEqual(coordinates, [[1, 2, 3], [2]])
        self.assertEqual(table.loc[table.generator_val_loss.notna(), "epoch"].tolist(), [2])

    def test_invalid_coordinate_identity_is_rejected(self) -> None:
        """Mismatched lengths, repeated epochs and unknown keys cannot silently mislabel rows."""

        for epochs in ({"loss": [1]}, {"loss": [2, 2]}, {"unknown": [1, 2]}):
            with self.subTest(epochs=epochs), self.assertRaisesRegex(ValueError, "metric_epochs"):
                plot_history({"loss": [2., 1.]}, metric_epochs=epochs, show_plots=False)

    def test_validation_only_auxiliary_uses_known_ordinary_fit_axis(self) -> None:
        """Keep supplied fit dates and infer missing diagnostics from the longest training axis."""

        from common.train import _report_history_epochs


        history = {"loss": [4., 3., 2., 1.], "noise_loss": [3., 2., 1., 0.], 
                   "val_image_loss": [0.4, 0.2]}
        epochs = _report_history_epochs(None, history, object(), 
                                       {"fit_kwargs": {"initial_epoch": 2, "validation_freq": 2}}, None)
        self.assertEqual(epochs["val_image_loss"], [4, 6])
        _, table = self._render(history, epochs)
        self.assertEqual(table.loc[table.val_image_loss.notna(), "epoch"].tolist(), [4, 6])
        history["noise_loss"] = [1.]
        epochs = _report_history_epochs(None, history, object(), {"fit_kwargs": {"validation_freq": 2}}, None)
        _, table = self._render(history, epochs, metrics=["val_image_loss"])
        self.assertEqual(table.loc[table.val_image_loss.notna(), "epoch"].tolist(), [2, 4])

    def test_scheduled_block_history_owns_dense_padded_axis(self) -> None:
        """Merged block indices supersede restarted local epochs and retain absent validation as NaN."""

        from types import SimpleNamespace
        import tensorflow as tf
        from common.train import _fit_with_callback_cleanup, _report_history_epochs


        def fit(callbacks: Sequence[tf.keras.callbacks.Callback]) -> tf.keras.callbacks.History:
            """Emulate the schedule merger while exercising actual observer callback events.

            Args:
                callbacks (Sequence[tf.keras.callbacks.Callback]): Observers installed
                    by the shared fitting helper; each block has local Keras epoch zero.

            Returns:
                tf.keras.callbacks.History: Three block rows with validation only
                    on the final block and explicit NaN cells for earlier blocks.
            """

            for block in range(3):
                logs = {"loss": float(3 - block)}
                # Scheduled validation runs only after the final allocated block.
                if block == 2:
                    logs["val_auxiliary"] = 0.5
                for callback in callbacks:
                    callback.on_epoch_end(0, logs)
            result = tf.keras.callbacks.History()
            result.epoch = [0, 1, 2]
            result.history = {"loss": [3., 2., 1.], "val_auxiliary": [np.nan, np.nan, 0.5]}
            return result

        trained = _fit_with_callback_cleanup(SimpleNamespace(fit=fit))
        epochs = _report_history_epochs(None, trained.history, object(), {}, None)
        self.assertEqual(epochs, {"loss": [1, 2, 3], "val_auxiliary": [1, 2, 3]})
        _, table = self._render(trained.history, epochs)
        self.assertEqual(table.loc[table.val_auxiliary.notna(), "epoch"].tolist(), [3])
        self.assertTrue(table.val_auxiliary.iloc[:2].isna().all())

    def test_epoch_observer_participates_in_strict_callback_recovery(self) -> None:
        """Authenticate and restore prior observation coordinates through the actual recovery API."""

        from common.train import _MetricEpochRecorder
        from common.recovery import (callback_recovery_descriptor, callback_recovery_state, 
                                     restore_callback_recovery_state)


        original = _MetricEpochRecorder()
        original.on_epoch_end(1, {"loss": 2., "val_auxiliary": 0.5})
        restored = _MetricEpochRecorder()
        self.assertEqual(callback_recovery_descriptor([original], strict=True), 
                         callback_recovery_descriptor([restored], strict=True))
        saved = callback_recovery_state([original])
        restore_callback_recovery_state([restored], saved)
        restored.on_epoch_end(3, {"loss": 1.})
        self.assertEqual(restored.metric_epochs, {"loss": [2, 4], "val_auxiliary": [2]})
        self.assertEqual(original.metric_epochs, {"loss": [2], "val_auxiliary": [2]})
        self.assertEqual(saved, [{"loss": [2], "val_auxiliary": [2]}])

    def test_real_fit_captures_sparse_validation_only_metric_epochs(self) -> None:
        """Actual callback logs survive resumed fitting and drive CSV dates without schedule inference."""

        import tensorflow as tf
        from common.train import _fit_with_callback_cleanup, _report_history_epochs


        class AuxiliaryMetric(tf.keras.callbacks.Callback):
            """Emit a validation-only diagnostic after each actual validation event."""

            def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
                """Append a scalar diagnostic only when this epoch actually validated.

                Args:
                    epoch (int): Zero-based Keras epoch; its value identifies the diagnostic.
                    logs (dict[str, float] | None): Mutable Keras logs, or no event metrics.

                Returns:
                    None: Add val_auxiliary without introducing a training counterpart.
                """

                # Validation presence identifies actual observations independently of requested cadence.
                if logs is not None and "val_loss" in logs:
                    logs["val_auxiliary"] = float(epoch + 1)

        model = tf.keras.Sequential([tf.keras.layers.Input(tuple([1])), tf.keras.layers.Dense(1)])
        model.compile(optimizer="sgd", loss="mse", run_eagerly=True)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        dataset = tf.data.Dataset.from_tensor_slices(([[1.], [2.]], [[2.], [4.]])).batch(2).with_options(options)
        auxiliary = AuxiliaryMetric()
        original_callbacks = [auxiliary]
        trained = _fit_with_callback_cleanup(model, x=dataset, validation_data=dataset, 
                                             epochs=6, initial_epoch=2, validation_freq=2, 
                                             callbacks=original_callbacks, verbose=0)
        self.assertEqual(original_callbacks, [auxiliary])
        self.assertEqual(trained.history["val_auxiliary"], [4., 6.])
        epochs = _report_history_epochs(None, trained.history, model, 
                                       {"fit_method": "fit_progressively"}, None)
        self.assertEqual(epochs["loss"], [3, 4, 5, 6])
        self.assertEqual(epochs["val_auxiliary"], [4, 6])
        _, table = self._render(trained.history, epochs)
        self.assertEqual(table.loc[table.val_auxiliary.notna(), "epoch"].tolist(), [4, 6])
        tf.keras.backend.clear_session()

    def test_report_derives_integer_list_and_resumed_fit_cadence(self) -> None:
        """CSV-only reports preserve known Keras frequency/list schedules and initial_epoch."""

        cases = ((0, 2, [2, 4]), (2, 2, [4, 6]), (0, [1, 4], [1, 4]))
        for initial, frequency, expected in cases:
            with self.subTest(initial=initial, frequency=frequency), tempfile.TemporaryDirectory() as directory:
                result = report(
                    history={"loss": [5., 4., 3., 2., 1.], "val_loss": [4.5, 2.5]}, 
                    model=object(), results_path=directory, save_csv=True, 
                    show_history_plot=False, save_history_plot=False, 
                    plot_without_20percent=False, run_trainset_eval=False, 
                    run_valset_eval=False, show_final_images=False, 
                    save_final_images=False, save_final_gifs=False, 
                    fit_kwargs={"initial_epoch": initial, "validation_freq": frequency}
                )
                self.assertEqual(result, {})
                table = pd.read_csv(Path(directory) / "train history.csv")
                self.assertEqual(table.loc[table.val_loss.notna(), "epoch"].tolist(), expected)
                self.assertEqual(table.epoch.tolist(), list(range(initial + 1, initial + 6)))

    def test_report_uses_same_metadata_for_full_and_partial_figures(self) -> None:
        """Both report views receive the exact cadence rather than independent length estimates."""

        with patch("common.train.plot_history") as plotted:
            report(
                history={"loss": [5., 4., 3., 2., 1.], "val_loss": [4.5, 2.5]}, 
                model=object(), show_history_plot=True, save_csv=False, 
                save_history_plot=False, plot_without_20percent=True, 
                run_trainset_eval=False, run_valset_eval=False, 
                show_final_images=False, save_final_images=False, save_final_gifs=False, 
                fit_kwargs={"validation_freq": 2}
            )
        self.assertEqual(plotted.call_count, 2)
        for call in plotted.call_args_list:
            self.assertEqual(call.kwargs["metric_epochs"]["val_loss"], [2, 4])

    def test_staged_report_uses_plot_inference_and_preserves_explicit_coordinates(self) -> None:
        """Staged reports use generic plotting estimates unless actual epoch metadata is supplied."""

        with tempfile.TemporaryDirectory() as directory:
            arguments = {
                "history": {"loss": [5., 4., 3., 2., 1.], "val_loss": [4.5, 2.5]}, 
                "model": object(), "results_path": directory, "save_csv": True, 
                "show_history_plot": False, "save_history_plot": False, 
                "plot_without_20percent": False, "run_trainset_eval": False, 
                "run_valset_eval": False, "show_final_images": False, 
                "save_final_images": False, "save_final_gifs": False, 
                "fit_method": "fit_progressively", "fit_kwargs": {"validation_freq": 2}
            }
            report(**arguments)
            table = pd.read_csv(Path(directory) / "train history.csv")
            self.assertEqual(table.loc[table.val_loss.notna(), "epoch"].tolist(), [2, 4])
            report(metric_epochs={"val_loss": [1, 4]}, **arguments)
            table = pd.read_csv(Path(directory) / "train history.csv")
            self.assertEqual(table.loc[table.val_loss.notna(), "epoch"].tolist(), [1, 4])

    def test_continual_generator_reporting_preserves_its_phase_cadence(self) -> None:
        """Real task orchestration forwards sparse generator cadence independently of dense classifiers."""

        from common.learner import _run_continual_tasks
        from common.tests.test_continual_integration import ContinualIntegrationTests
        from diffusion import DiffusionModel


        def fit_fixture(config: object, model: object, dataset: object, **options: object) -> dict[str, list[float]]:
            """Return synthetic histories matching the actual phase's forwarded fit cadence.

            Args:
                config (object): Unused direct-mode configuration placeholder.
                model (object): Generator or external classifier being fitted.
                dataset (object): Unused real task dataset constructed by the learner.
                **options (object): Actual phase options, accepted without modifying them.

            Returns:
                dict[str, list[float]]: Five training values and two generator or five
                    classifier validation values. No optimizer update is performed.
            """

            del config, dataset, options
            count = 2 if isinstance(model, DiffusionModel) else 5
            return {"loss": [5., 4., 3., 2., 1.], "val_loss": [1.] * count}

        def report_fixture(*args: object, **options: object) -> dict[str, object]:
            """Exercise the actual report resolver while omitting redundant model evaluation.

            Args:
                *args (object): Unchanged positional task reporter inputs.
                **options (object): Task report options including the actual fit cadence.

            Returns:
                dict[str, object]: The actual common report result with validation
                    evaluation disabled; history resolution and plot forwarding stay active.
            """

            return report(*args, **{**options, "run_valset_eval": False})

        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "template.h5"
            ContinualIntegrationTests._template(template)
            with patch("common.train.train_model", side_effect=fit_fixture), patch(
                "common.train.report", side_effect=report_fixture
            ), patch("common.train.plot_history") as plotted:
                _run_continual_tasks(
                    class_num=3, task_size=2, load_dataset_fn=ContinualIntegrationTests._loader, 
                    load_dataset_fn_kwargs={"preprocess": None}, 
                    tuned_model_path=str(template), 
                    generative_model=ContinualIntegrationTests._generator(), 
                    generative_model_kwargs={"train_num": -1}, 
                    use_generative_replay=False, remove_prev_classes=False, 
                    batch_size=4, epochs=5, callback_patience=0, 
                    plot_results=False, fit_kwargs={"validation_freq": 2}, show_network_summary=False, 
                    verbose=1, seed=31
                )
            observed = [call.kwargs["metric_epochs"]["val_loss"] for call in plotted.call_args_list]
            self.assertEqual(observed.count([2, 4]), 2)
            self.assertEqual(observed.count([1, 2, 3, 4, 5]), 2)

    def test_v2_only_explicit_single_phase_fits_supply_cadence(self) -> None:
        """A selected V2 phase has ordinary epochs; a merged V2 fit retains explicit metadata only."""

        from unittest.mock import MagicMock
        from common.train import _report_history_epochs
        from diffusion import DiffusionClassifierV2


        model = MagicMock(spec=DiffusionClassifierV2)
        history = {"loss": [5., 4., 3., 2., 1.], "val_loss": [4.5, 2.5]}
        for method in ("fit_generator", "fit_discriminator"):
            with self.subTest(method=method):
                epochs = _report_history_epochs(
                    None, history, model, 
                    {"fit_method": method, "fit_kwargs": {"validation_freq": 2}}, None
                )
                self.assertEqual(epochs["val_loss"], [2, 4])
        self.assertIsNone(_report_history_epochs(None, history, model, {"fit_method": "fit"}, None))



    def test_disabled_history_outputs_skip_coordinate_resolution_and_plotting(self) -> None:
        """An evaluation-only report does not inspect ambiguous history it was not asked to emit."""

        with patch("common.train._report_history_epochs") as resolve, patch("common.train.plot_history") as plotted:
            result = report(
                history={"loss": [3., 2., 1.], "val_loss": [1.5]}, 
                model=object(), show_history_plot=False, save_history_plot=False, save_csv=False, 
                run_trainset_eval=False, run_valset_eval=False, 
                show_final_images=False, save_final_images=False, save_final_gifs=False
            )
        self.assertEqual(result, {})
        resolve.assert_not_called()
        plotted.assert_not_called()


# Run these focused reporting regressions only when invoked directly.
if __name__ == "__main__":
    unittest.main()
