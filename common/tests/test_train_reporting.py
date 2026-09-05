"""Regression checks for training/reporting dispatch and visual artifact controls.

Small continual detail mappings, patched training paths, and a fake diffusion sampler
isolate report output, paths, task-specific seeds, image/GIF selection, and Config
propagation. The tests use temporary destinations and inspect calls rather than launching a
full HPO study.

Inputs are fixtures constructed by the test methods and their helpers. Tests return no
application result: unittest records assertion outcomes and errors. Run this module directly
or through ``python -m unittest`` discovery. Importing it defines fixtures and cases; it
does not itself start a test run.
"""

from __future__ import annotations

import tempfile
import unittest
import warnings

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import tensorflow as tf

from common.config import Config, load_config
from common.runtime import derive_seed
from common.train import _report_final_visuals, main, report, train_model
from autoencoder.variational_autoencoder import VariationalAutoencoder


class _FakeDiffusion:
    """Minimal diffusion sampling protocol used by reporting tests.

    This fixture implements only the interface required by its surrounding regression tests.
    Construction and mutable state are described by ``__init__``; it does not provide a
    general production replacement.

    Returns:
        _FakeDiffusion: A new local test fixture with independent instance state.
    """

    swap_noise_image = False
    test_network_name = "raw"
    timesteps = 10

    def __init__(self: "_FakeDiffusion") -> None:
        """Create an empty sampling-call log.

        Returns:
            None: The fake starts with no recorded calls.
        """

        self.calls: list[dict[str, object]] = []

    def sample(self: "_FakeDiffusion", **kwargs: object) -> object:
        """Record sampling arguments and return compatible fake outputs.

        Args:
            **kwargs (object): Diffusion sampling options supplied by reporting.

        Returns:
            object: Final images, or final images plus two GIF frame sequences.
        """

        self.calls.append(dict(kwargs))
        images = np.zeros((1, 2, 2, 1), dtype=np.float32)
        # GIF requests need intermediate frames as well as final images.
        if kwargs.get("return_x_ts"):
            return images, [images], [images]
        return images


class TrainReportingTests(unittest.TestCase):
    """Verify continual visual artifacts and isolated report RNG streams.

    The unittest runner executes the selected test method with its local fixtures;
    individual methods describe the configurations and failure cases they exercise. There is
    no application model or experiment result returned by constructing this test case.

    Args:
        methodName (str): Test method selected by unittest. Defaults to ``"runTest"``;
            discovery supplies each named ``test_*`` method.

    Attributes:
        _testMethodName (str): Selected method name maintained by unittest.
    """

    def test_singleton_validation_accuracy_is_warning_free(self) -> None:
        """Keep an undefined first-task validation score as NaN.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        inputs = tf.keras.Input((1,))
        classifier = tf.keras.Model(inputs, tf.keras.layers.Dense(1)(inputs))
        details = {
            "model": classifier,
            "generative_model": None,
            "accuracies": [np.nan, 0.70, 0.80],
            "ensemble_accuracies": [],
            "validation_accuracy_matrix": [
                [np.nan, np.nan, np.nan],
                [0.60, 0.70, np.nan],
                [0.50, 0.65, 0.80],
            ],
            "histories": [{}, {}, {}],
        }

        with tempfile.TemporaryDirectory() as temporary, patch(
            "common.learner._run_continual_tasks",
            return_value=details,
        ), warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            history = train_model(
                model={"classifier_name": "tiny", "classifier": classifier},
                trainset=object(),
                results_path=temporary,
                task="continual",
                dataset_name="mnist",
                epochs=1,
                batch_size=2,
                show_images=False,
                save_gifs=False,
                save_weights=False,
                use_tensorboard=False,
                verbose=0,
                continually_learn_kwargs={
                    "class_num": 3,
                    "task_groups": [[0], [1], [2]],
                    "baseline": "cumulative",
                },
            )

        self.assertTrue(np.isnan(history["task_val_accuracy"][0]))

    def test_direct_vae_train_materializes_dataset_rows(self) -> None:
        """Pass raw aligned rows to the VAE resampling training API.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        images = np.arange(12, dtype="float32").reshape(4, 3)
        labels = np.eye(2, dtype="float32")[[0, 1, 0, 1]]
        trainset = tf.data.Dataset.from_tensor_slices(
            (images, labels)
        ).batch(3)
        model = MagicMock(spec=VariationalAutoencoder)
        model.conditioned = True
        model.train.return_value = {"loss": [1.0]}

        history = train_model(
            model=model,
            trainset=trainset,
            fit_method="train",
            fit_kwargs={"train_num": -1},
            epochs=1,
            show_images=True,
            report_every_epoch=False,
            save_weights=False,
            verbose=0,
        )

        self.assertEqual(history, {"loss": [1.0]})
        train_args = model.train.call_args
        np.testing.assert_array_equal(train_args.args[0], images)
        np.testing.assert_array_equal(train_args.kwargs["y"], labels)
        self.assertEqual(train_args.kwargs["batch_size"], 128)

    def test_configured_continual_seed_is_forwarded_once(self) -> None:
        """Use the effective training seed without duplicate keyword routing.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        inputs = tf.keras.Input((1,))
        classifier = tf.keras.Model(
            inputs,
            tf.keras.layers.Dense(2, activation="softmax")(inputs),
        )
        details = {
            "model": classifier,
            "generative_model": None,
            "accuracies": [],
            "ensemble_accuracies": [],
            "validation_accuracy_matrix": [],
            "histories": [],
        }

        with tempfile.TemporaryDirectory() as temporary, patch(
            "common.learner._run_continual_tasks",
            return_value=details,
        ) as continual:
            train_model(
                model={
                    "classifier_name": "tiny",
                    "classifier": classifier,
                },
                trainset=object(),
                results_path=temporary,
                task="continual",
                dataset_name="mnist",
                seed=19,
                epochs=1,
                batch_size=2,
                show_images=False,
                save_gifs=False,
                save_weights=False,
                use_tensorboard=False,
                verbose=0,
                continually_learn_kwargs={
                    "class_num": 4,
                    "task_groups": [[0, 1], [2, 3]],
                    "task_size": 2,
                    "seed": 19,
                    "baseline": "cumulative",
                },
            )

        self.assertEqual(continual.call_args.kwargs["seed"], 19)

    def test_input_config_remains_pretraining_recovery_specification(
        self: "TrainReportingTests",
    ) -> None:
        """Keep immutable input settings separate from resolved output paths.

        Returns:
            None: The input YAML excludes the post-training weight path.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        inputs = tf.keras.Input((1,))
        outputs = tf.keras.layers.Dense(2, activation="softmax")(inputs)
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer="sgd",
            loss="sparse_categorical_crossentropy",
        )
        trainset = tf.data.Dataset.from_tensor_slices((
            np.asarray([[0.], [1.], [0.5], [0.25]], dtype="float32"),
            np.asarray([0, 1, 1, 0], dtype="int32"),
        )).batch(2)

        with tempfile.TemporaryDirectory() as temporary:
            config = Config(training={
                "epochs": 1,
                "results_path": temporary,
                "project_tag": "immutable-input",
                "save_weights": True,
                "save_gifs": False,
                "report_every_epoch": False,
                "verbose": 0,
                "task": "classification",
            })
            train_model(config, model, trainset)
            result_path = Path(config.training.results_path)
            input_path = result_path / "input_config.yaml"
            final_path = result_path / "config.yaml"
            input_config = load_config(input_path)
            final_config = load_config(final_path)

            self.assertTrue(input_path.is_file())
            self.assertTrue(final_path.is_file())
            self.assertIsNone(input_config.model.weights_path)
            self.assertEqual(
                final_config.model.weights_path,
                config.model.weights_path,
            )
            self.assertTrue(Path(final_config.model.weights_path).is_file())
            self.assertEqual(
                Path(input_config.hpo["input_config_path"]).resolve(),
                input_path.resolve(),
            )

    def test_continual_report_forwards_final_visual_options(
        self: "TrainReportingTests",
    ) -> None:
        """Do not let the continual summary return skip its replay model.

        Returns:
            None: The helper invocation is asserted in place.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        replay_model = object()
        bundle = {
            "generative_model": replay_model,
            "continual_details": {},
        }
        with tempfile.TemporaryDirectory() as temporary, \
                patch("common.train.plot_history"), \
                patch("common.train._report_final_visuals") as visuals:
            result = report(
                history={"continual_accuracy": [0.25, 0.5]},
                model=bundle,
                trainset=object(),
                results_path=temporary,
                show_history_plot=False,
                save_history_plot=False,
                save_csv=False,
                show_final_images=False,
                save_final_images=True,
                save_final_gifs=True,
                final_images_steps=7,
                final_images_cfg_scale=2.5,
                dataset_name="CIFAR10",
                seed=19,
            )

        self.assertEqual(result["final_accuracy"], 0.5)
        visuals.assert_called_once_with(
            replay_model,
            "CIFAR10",
            temporary,
            False,
            True,
            True,
            7,
            2.5,
            19,
        )

    def test_continual_report_all_nan_means_are_warning_free(self) -> None:
        """Undefined singleton summaries remain NaN without RuntimeWarning.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        bundle = {"generative_model": None, "continual_details": {}}
        with tempfile.TemporaryDirectory() as temporary, \
                patch("common.train.plot_history"), \
                patch("common.train._report_final_visuals"), \
                warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = report(
                history={
                    "continual_accuracy": [np.nan],
                    "continual_ensemble_accuracy": [np.nan],
                },
                model=bundle,
                trainset=object(),
                results_path=temporary,
                show_history_plot=False,
                save_history_plot=False,
                save_csv=False,
                show_final_images=False,
                save_final_images=False,
                save_final_gifs=False,
            )

        self.assertTrue(np.isnan(result["average_accuracy"]))
        self.assertTrue(np.isnan(result["average_ensemble_accuracy"]))

    def test_diffusion_final_gif_uses_derived_seed(
        self: "TrainReportingTests",
    ) -> None:
        """Keep final GIF sampling separate from training RNG consumption.

        Returns:
            None: Sampling and GIF arguments are asserted in place.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        model = _FakeDiffusion()
        with tempfile.TemporaryDirectory() as temporary, \
                patch("common.train.DiffusionModel", _FakeDiffusion), \
                patch("common.train.create_gif") as create_gif:
            _report_final_visuals(
                model,
                "MNIST",
                temporary,
                show_final_images=False,
                save_final_images=False,
                save_final_gifs=True,
                final_images_steps=5,
                final_images_cfg_scale=1.5,
                seed=23,
            )

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(
            model.calls[0]["seed"],
            derive_seed(23, "final_report", "diffusion_sampling"),
        )
        self.assertTrue(model.calls[0]["return_x_ts"])
        self.assertTrue(model.calls[0]["return_x0s"])
        self.assertEqual(model.calls[0]["steps"], 5)
        self.assertEqual(model.calls[0]["scale"], 1.5)
        gif_path = Path(create_gif.call_args.args[0])
        self.assertEqual(gif_path.parent, Path(temporary))
        self.assertIn("steps-5_scale-1.5", gif_path.name)

    def test_report_rejects_missing_required_path_before_plotting(self) -> None:
        """Expose a clear precondition instead of failing inside os.path.join.

        Returns:
            None: The result-path error precedes plotting side effects.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        with patch("common.train.plot_history") as plot_history:
            with self.assertRaisesRegex(TypeError, "history-plot saving"):
                report(
                    history={"loss": [1.0]},
                    model=object(),
                    trainset=object(),
                    results_path=None,
                    save_history_plot=True,
                    save_csv=False,
                    show_history_plot=False,
                    plot_without_20percent=False,
                    run_trainset_eval=False,
                    run_valset_eval=False,
                    show_final_images=False,
                    save_final_images=False,
                    save_final_gifs=False,
                )

        plot_history.assert_not_called()

    def test_report_allows_no_path_when_every_file_output_is_disabled(self) -> None:
        """Retain display/evaluation-only reporting without an artifact root.

        Returns:
            None: A no-output report completes without creating paths.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        result = report(
            history={},
            model=object(),
            trainset=object(),
            results_path=None,
            save_history_plot=False,
            save_csv=False,
            show_history_plot=False,
            run_trainset_eval=False,
            run_valset_eval=False,
            show_final_images=False,
            save_final_images=False,
            save_final_gifs=False,
        )
        self.assertEqual(result, {})

    def test_final_visuals_require_a_path_before_sampling(self) -> None:
        """Check file destinations before delegating controls to sampling.

        Returns:
            None: A missing output path fails before model sampling.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        model = _FakeDiffusion()
        with patch("common.train.DiffusionModel", _FakeDiffusion):
            with self.assertRaisesRegex(TypeError, "final image saving"):
                _report_final_visuals(
                    model,
                    "MNIST",
                    None,
                    False,
                    True,
                    False,
                    5,
                    1.0,
                    None,
                )
        self.assertEqual(model.calls, [])

    def test_train_model_requires_path_for_enabled_weight_saving(self) -> None:
        """Reject missing artifact roots before callback or fit construction.

        Returns:
            None: Weight persistence reports its path dependency directly.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        with self.assertRaisesRegex(TypeError, "weight saving"):
            train_model(
                model=object(),
                trainset=object(),
                results_path=None,
                show_images=True,
                save_weights=True,
            )

    def test_main_rejects_unknown_direct_task_before_loading_data(self) -> None:
        """Validate direct task routing before dataset or model side effects.

        Returns:
            None: Unknown selection fails before dataset construction.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        with patch("common.train.get_datasets") as get_datasets:
            with self.assertRaisesRegex(ValueError, "training task"):
                main(task="prediction")

        get_datasets.assert_not_called()

    def test_direct_main_reports_and_returns_concrete_results_path(self) -> None:
        """Propagate training's timestamped artifact directory in direct mode.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        concrete_path = "results/2026-09-05_12-00-00"
        history = {"loss": [1.0]}
        model = object()

        def fake_train_model(
            *args: object,
            _run_state: dict[str, object],
            **kwargs: object,
        ) -> dict[str, list[float]]:
            """Publish the simulated resolved result path and return fixed history.

            The closure uses concrete_path and history from the surrounding test. It lets
            main/report propagation be checked without constructing a training dataset or
            updating model weights.

            Args:
                *args (object): Accepted train_model positional inputs; ignored and empty by
                    default.
                _run_state (dict[str, object]): Required keyword-only mutable handoff
                    mapping whose results_path is set to the concrete destination.
                **kwargs (object): Other training options, accepted but ignored; empty by
                    default.

            Returns:
                dict[str, list[float]]: The surrounding history mapping by identity.
            """

            del args, kwargs
            _run_state["results_path"] = concrete_path
            return history

        with patch("common.train.configure_runtime"), patch(
            "common.train.get_datasets",
            return_value=(object(), None),
        ), patch(
            "common.train.get_model",
            return_value=model,
        ), patch(
            "common.train.train_model",
            side_effect=fake_train_model,
        ), patch(
            "common.train.report",
            return_value={"accuracy": 0.5},
        ) as report_mock:
            result = main(
                task="classification",
                results_path="results",
                trainset_len=1,
            )

        self.assertEqual(report_mock.call_args.kwargs["results_path"], concrete_path)
        self.assertEqual(result["results_path"], concrete_path)
        self.assertIs(result["history"], history)


# Run this module's tests when executed directly.
if __name__ == "__main__":
    unittest.main()
