"""Regression checks for task selection and continual dataset setup.

The tests check accepted task spelling, rejected tasks, sample limits, and the early
continual-loader return used to size optimizer schedules. Dataset/model mocks keep these
tests focused on orchestration rather than downloading images.

Inputs are fixtures constructed by the test methods and their helpers. Tests return no
application result: unittest records assertion outcomes and errors. Run this module directly
or through ``python -m unittest`` discovery. Importing it defines fixtures and cases; it
does not itself start a test run.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from common.config import Config
from common.dataloader import get_datasets, load_mnist, preprocess_dataset
from common.hpo import run_hpo
from common.learner import _load_continual_arrays, continually_learn
from common.model import get_model


class DatasetTaskValidationTests(unittest.TestCase):
    """Keep direct dataset construction on the shared task vocabulary.

    The unittest runner executes the selected test method with its local fixtures;
    individual methods describe the configurations and failure cases they exercise. There is
    no application model or experiment result returned by constructing this test case.

    Args:
        methodName (str): Test method selected by unittest. Defaults to ``"runTest"``;
            discovery supplies each named ``test_*`` method.

    Attributes:
        _testMethodName (str): Selected method name maintained by unittest.
    """

    def test_continual_only_sizes_the_deferred_training_pipeline(self) -> None:
        """Resolve cosine sizing without constructing discarded task datasets.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        images = np.zeros((9, 28, 28), dtype="uint8")
        labels = np.arange(9) % 2
        config = Config(
            dataset={"batch_size": 4, "pad": 2},
            model={"name": "cnn", "show_network_summary": False},
            training={"task": "continual"},
            continually_learn={"class_num": 2},
        )
        with patch("common.dataloader.load_mnist") as loader, patch(
            "common.dataloader.get_dataset"
        ) as get_dataset:
            loader.return_value = (images, labels, images, labels, images, labels)
            result = get_datasets(config)

        self.assertEqual(result, (loader, None))
        self.assertEqual(config.dataset.trainset_len, 3)
        get_dataset.assert_not_called()

    def test_unknown_direct_task_is_rejected_before_loading_data(self) -> None:
        """An unsupported direct task must not reach a dataset loader.

        Returns:
            None.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        with patch("common.dataloader.load_mnist") as load_mnist:
            with self.assertRaisesRegex(ValueError, "training task must be one of"):
                get_datasets(task="unknown")

        load_mnist.assert_not_called()

    def test_mutated_config_and_hpo_tasks_use_native_type_errors(self) -> None:
        """Every task-bearing entry point lets string operations reject mutation.

        Returns:
            None.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        config = Config()
        config.training.task = None

        with patch("common.dataloader.load_mnist") as load_mnist:
            for entry_point in (get_datasets, get_model, continually_learn):
                with self.subTest(entry_point=entry_point.__name__):
                    with self.assertRaises(AttributeError):
                        entry_point(config)

        load_mnist.assert_not_called()

        with self.assertRaises(AttributeError):
            run_hpo(None, "cnn", n_trials=1)

    def test_direct_vae_preprocessing_follows_reconstruction_activation(self) -> None:
        """Keep direct VAE targets in the configured reconstruction range.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify activation-dependent loader preprocessing.
        """

        images = np.zeros((4, 28, 28), dtype="uint8")
        labels = np.asarray([0, 1, 0, 1], dtype="uint8")
        for model_name in ("vae", "variational_autoencoder", "vae_classifier"):
            for option_name in ("model_kwargs", "kwargs"):
                for activation, expected in (
                    ("tanh", "standardize"), ("sigmoid", "min-max"),
                    ("linear", "normalize"), (None, "normalize"),
                ):
                    with self.subTest(
                        model_name=model_name,
                        option_name=option_name,
                        activation=activation,
                    ), patch("common.dataloader.load_mnist") as loader:
                        loader.return_value = (
                            images, labels, None, None, images, labels
                        )
                        get_datasets(
                            model_name=model_name,
                            preprocess=None,
                            use_valset=False,
                            **{option_name: {"last_activation": activation}},
                        )
                        self.assertEqual(
                            loader.call_args.kwargs["preprocess"], expected
                        )

    def test_fixed_pixel_scaling_is_independent_of_future_class_extrema(self) -> None:
        """Keep first-task pixels unchanged when future-task image statistics change.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Both fixed scales match public pixel bounds on every split.
        """

        labels = np.repeat(np.arange(2), 4)
        first_task = np.asarray([64, 96, 128, 160], dtype="uint8")
        test_pixels = np.asarray([0, 255], dtype="uint8")[:, None, None]
        for mode, multiplier, offset in (
            ("fixed-min-max", 1., 0.), ("fixed-standardize", 2., -1.),
        ):
            first_observations = []
            for future_pixels in ([0, 32, 224, 255], [48, 64, 176, 192]):
                pixels = np.concatenate([
                    first_task, np.asarray(future_pixels, dtype="uint8"),
                ])[:, None, None]
                prepared = preprocess_dataset(
                    pixels, labels, test_pixels, np.asarray([0, 1]),
                    class_num=2, indices=[0, 1], validation_ratio=0.5,
                    preprocess=mode, return_features=False, features_path=None,
                    onehot_labels=False, seed=19, verbose=0,
                )
                train_x, train_y, val_x, val_y, test_x, _ = prepared
                first_observations.append(np.concatenate([
                    train_x[train_y == 0], val_x[val_y == 0],
                ]))
                np.testing.assert_allclose(
                    np.sort(first_observations[-1].reshape(-1)),
                    first_task.astype("float32") / 255. * multiplier + offset,
                    rtol=1e-6,
                )
                np.testing.assert_allclose(
                    test_x, test_pixels.astype("float32") / 255. * multiplier + offset,
                    rtol=1e-6,
                )
            np.testing.assert_array_equal(*first_observations)

    def test_fixed_standardize_padding_uses_the_public_lower_bound(self) -> None:
        """Keep ordinary and continual padded borders at diffusion-space minus one.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Both input pipelines preserve the declared public pixel scale.
        """

        pixels = np.full((4, 2, 2), 128, dtype="uint8")
        labels = np.asarray([0, 1, 0, 1], dtype="uint8")
        with patch(
            "tensorflow.keras.datasets.mnist.load_data",
            return_value=((pixels, labels), (pixels, labels)),
        ):
            dataset, _ = get_datasets(
                model_name="cnn", preprocess="fixed-standardize", pad=1,
                validation_ratio=0., batch_size=4, shuffle_buffer=0,
            )
            arrays, _ = _load_continual_arrays(
                load_mnist, [0, 1], False,
                {"preprocess": "fixed-standardize", "onehot_labels": False,
                 "validation_ratio": 0.},
                None, None, 1, 19,
            )
        ordinary = next(iter(dataset))[0].numpy()[..., 0]
        for prepared in (ordinary, arrays[0], arrays[4]):
            np.testing.assert_array_equal(prepared[:, 0, :], -1.)
            np.testing.assert_array_equal(prepared[:, :, 0], -1.)
            np.testing.assert_allclose(
                prepared[:, 1:-1, 1:-1], 128. / 255. * 2. - 1., atol=1e-7,
            )

    def test_fixed_pixel_scaling_rejects_saved_feature_units(self) -> None:
        """Reject unknown feature units before loading a saved feature archive.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Both fixed pixel modes reject feature preprocessing explicitly.
        """

        pixels = np.zeros((4, 2, 2), dtype="uint8")
        labels = np.asarray([0, 1, 0, 1])
        for mode in ("fixed-min-max", "fixed-standardize"):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                ValueError, "not supported for saved features",
            ):
                preprocess_dataset(
                    pixels, labels, pixels, labels,
                    class_num=2, indices=[0, 1], validation_ratio=0.,
                    preprocess=mode, return_features=True, features_path=None,
                    onehot_labels=False, seed=19, verbose=0,
                )

    def test_vae_conditioning_selects_onehot_labels(self) -> None:
        """Keep VAE factory inputs aligned with their conditioning mode.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        images = np.zeros((4, 28, 28), dtype="uint8")
        sparse = np.asarray([0, 1, 0, 1], dtype="uint8")
        onehot = np.eye(10, dtype="float32")[sparse]

        with patch("common.dataloader.load_mnist") as load_mnist:
            load_mnist.return_value = (
                images, onehot, None, None, images[:2], onehot[:2]
            )
            config = Config(
                model={"name": "vae_classifier"},
                training={"task": "joint", "use_valset": False},
            )
            get_datasets(config)

            self.assertTrue(config.dataset.onehot_labels)
            self.assertTrue(load_mnist.call_args.kwargs["onehot_labels"])

        with patch("common.dataloader.load_mnist") as load_mnist:
            load_mnist.return_value = (
                images, onehot, None, None, images[:2], onehot[:2]
            )
            config = Config(
                model={"name": "vae"},
                training={"task": "continual", "use_valset": False},
                continually_learn={"class_num": 2},
            )
            get_datasets(config)

            self.assertTrue(config.dataset.onehot_labels)
            self.assertTrue(load_mnist.call_args.kwargs["onehot_labels"])

        with patch("common.dataloader.load_mnist") as load_mnist:
            load_mnist.return_value = (
                images, sparse, None, None, images[:2], sparse[:2]
            )
            get_datasets(model_name="vae", use_valset=False)

            self.assertFalse(load_mnist.call_args.kwargs["onehot_labels"])
            self.assertFalse(get_model(model_name="vae").conditioned)
