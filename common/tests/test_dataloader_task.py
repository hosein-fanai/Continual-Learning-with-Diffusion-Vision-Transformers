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
from common.dataloader import get_datasets
from common.hpo import run_hpo
from common.learner import continually_learn
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
