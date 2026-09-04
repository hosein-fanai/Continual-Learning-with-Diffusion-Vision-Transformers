"""Focused task-boundary regressions for :mod:`common.dataloader`."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from common.config import Config
from common.dataloader import get_datasets
from common.hpo import run_hpo
from common.learner import continually_learn
from common.model import get_model


class DatasetTaskValidationTests(unittest.TestCase):
    """Keep direct dataset construction on the shared task vocabulary."""

    def test_unknown_direct_task_is_rejected_before_loading_data(self) -> None:
        """An unsupported direct task must not reach a dataset loader.

        Returns:
            None.
        """

        with patch("common.dataloader.load_mnist") as load_mnist:
            with self.assertRaisesRegex(ValueError, "training task must be one of"):
                get_datasets(task="unknown")

        load_mnist.assert_not_called()

    def test_mutated_config_and_hpo_tasks_use_native_type_errors(self) -> None:
        """Every task-bearing entry point lets string operations reject mutation.

        Returns:
            None.
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
