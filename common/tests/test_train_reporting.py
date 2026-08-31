"""Integration tests for final visual reporting in continual bundles."""

from __future__ import annotations

import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import Config, load_config
from common.runtime import derive_seed
from common.train import _report_final_visuals, main, report, train_model


class _FakeDiffusion:
    """Minimal diffusion sampling protocol used by reporting tests."""

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
        # Select the test action required by this condition.
        if kwargs.get("return_x_ts"):
            return images, [images], [images]
        return images


class TrainReportingTests(unittest.TestCase):
    """Verify continual visual artifacts and isolated report RNG streams."""

    def test_input_config_remains_pretraining_recovery_specification(
        self: "TrainReportingTests",
    ) -> None:
        """Keep immutable input settings separate from resolved output paths.

        Returns:
            None: The input YAML excludes the post-training weight path.
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

    def test_diffusion_final_gif_uses_derived_seed(
        self: "TrainReportingTests",
    ) -> None:
        """Keep final GIF sampling separate from training RNG consumption.

        Returns:
            None: Sampling and GIF arguments are asserted in place.
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
        """

        with patch("common.train.get_datasets") as get_datasets:
            with self.assertRaisesRegex(ValueError, "training task"):
                main(task="prediction")

        get_datasets.assert_not_called()


# Select the test action required by this condition.
if __name__ == "__main__":
    unittest.main()
