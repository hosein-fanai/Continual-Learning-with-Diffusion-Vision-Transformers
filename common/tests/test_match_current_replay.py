"""Check exact generated/current class balance without a full CIFAR training run."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import tensorflow as tf

from common.config import Config, load_config, save_config
from common.learner import _matched_current_class_count, _run_continual_tasks
from common.tests import test_continual_integration as fixtures


class MatchCurrentReplayTests(unittest.TestCase):
    """Exercise the new exposure policy and preserve legacy budget behavior."""

    def tearDown(self) -> None:
        """Release this isolated test process's Keras state after each case."""
        tf.keras.backend.clear_session()

    @staticmethod
    def _loader(indices: list[int], **kwargs: object) -> tuple[np.ndarray, ...]:
        """Return permitted current counts 6, 4 and 2 with disjoint validation rows."""
        del kwargs
        labels = np.concatenate([np.full(6 - 2 * int(label), label, dtype="int32")
                                 for label in indices])
        images = np.broadcast_to((.1 + labels * .2)[:, None, None, None],
                                 (len(labels), 2, 2, 1)).astype("float32").copy()
        validation_labels = np.asarray(indices, dtype="int32")
        validation = np.broadcast_to((.15 + validation_labels * .2)[:, None, None, None],
                                     (len(indices), 2, 2, 1)).astype("float32").copy()
        return images, labels, validation, validation_labels, validation.copy(), validation_labels.copy()

    @staticmethod
    def _arguments(template: Path) -> dict[str, object]:
        """Supply a tiny dynamic generator and ordinary task-training controls."""
        return {
            "class_num": 3, "task_size": 1,
            "load_dataset_fn": MatchCurrentReplayTests._loader,
            "load_dataset_fn_kwargs": {"preprocess": "fixed-min-max", "onehot_labels": False},
            "tuned_model_path": str(template),
            "compile_args": {"optimizer": tf.keras.optimizers.Adam(.01),
                             "loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"]},
            "generative_model": fixtures.ContinualIntegrationTests._generator(),
            "generative_model_kwargs": {"samples_per_class": 99, "train_num": 1},
            "replay_budget_mode": "match_current", "remove_prev_classes": True,
            "batch_size": 5, "epochs": 1, "callback_patience": 0,
            "plot_results": False, "show_generated_images": False, "verbose": 0,
            "seed": 31, "experiment_phase": "development", "return_details": True,
        }

    def test_policy_roundtrips_without_explicit_budgets(self) -> None:
        """Preserve dynamic budgeting in both supported YAML forms."""
        config = Config(continually_learn={"replay_budget_mode": "match_current",
                                          "replay_old_examples": None,
                                          "replay_current_examples": None})
        with tempfile.TemporaryDirectory() as directory:
            for compact in (False, True):
                path = Path(directory) / "recipe.yaml"
                save_config(config, path, shorten=compact)
                self.assertEqual(load_config(path), config)

    def test_per_class_count_handles_sparse_onehot_and_invalid_pools(self) -> None:
        """Infer counts from permitted labels without duplicating scarce current rows."""
        labels = np.repeat([0, 1, 2], [3, 3, 2])
        for representation in (labels, labels[:, None], np.eye(3)[labels]):
            self.assertEqual(_matched_current_class_count(representation, [0, 1]), 3)
            with self.assertRaisesRegex(ValueError, "equal positive current-class counts"):
                _matched_current_class_count(representation, [0, 2])
            with self.assertRaisesRegex(ValueError, "equal positive current-class counts"):
                _matched_current_class_count(representation, [3])

    def test_conflicting_controls_fail_before_loading_data(self) -> None:
        """Reject budgets, replay sources or selectors that cannot preserve the policy."""
        generator = fixtures.ContinualIntegrationTests._generator()
        cases = ({"replay_old_examples": 3}, {"replay_current_examples": 3},
                 {"remove_prev_classes": False}, {"use_buffer": True},
                 {"use_generative_replay": False}, {"generative_model": None},
                 {"replay_selection": "random"}, {"replay_selection": "confidence"},
                 {"replay_selection": "all", "replay_candidate_multiplier": 2})
        for override in cases:
            loader = Mock(side_effect=AssertionError("Invalid controls loaded data"))
            arguments = {"class_num": 3, "task_size": 1, "seed": 31,
                         "load_dataset_fn": loader, "generative_model": generator,
                         "replay_budget_mode": "match_current", **override}
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "match_current"):
                _run_continual_tasks(**arguments)
            loader.assert_not_called()

    def test_dynamic_generated_counts_keep_all_current_rows_and_resume(self) -> None:
        """Train tiny pools and resume the same balanced exposure from task one."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "classifier.h5"
            fixtures.ContinualIntegrationTests._template(template)
            arguments = self._arguments(template)
            arguments.update(save_task_checkpoints=True, checkpoint_dir=str(root / "checkpoints"))
            full = _run_continual_tasks(**arguments)
            resumed_arguments = self._arguments(template)
            resumed_arguments.update(resume_from=str(root / "checkpoints" / "task-0000"))
            resumed = _run_continual_tasks(**resumed_arguments)
        expected = [{"0": 6}, {"0": 4, "1": 4}, {"0": 2, "1": 2, "2": 2}]
        for result in (full, resumed):
            resources = result["task_resource_metrics"]
            self.assertEqual([row["training_class_counts"] for row in resources], expected)
            self.assertEqual([row["current_examples_exposed"] for row in resources], [6, 4, 2])
            self.assertEqual([row["current_examples_available"] for row in resources], [6, 4, 2])
            self.assertEqual([row["replay"]["selected_count"] for row in resources], [0, 4, 4])
            self.assertEqual([row["training_examples_total"] for row in resources], [6, 8, 6])
            self.assertEqual(result["run_descriptor"]["replay"]["matched_current_examples_per_class"], [6, 4, 2])
            # train_num=1 must not reduce the six/eight/six-row generator pools.
            self.assertEqual([row["optimizer_updates"]["replay_optimizer"] for row in resources], [2, 2, 2])
        for expected_weights, actual_weights in zip(full["generative_model"].get_weights(), resumed["generative_model"].get_weights()):
            np.testing.assert_allclose(expected_weights, actual_weights, rtol=0, atol=0)

    def test_uniform_selection_preserves_quotas_from_expanded_candidates(self) -> None:
        """Retain equal class counts when a larger generated pool is subsampled."""
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "classifier.h5"
            fixtures.ContinualIntegrationTests._template(template)
            arguments = self._arguments(template)
            arguments.update(replay_selection="uniform", replay_candidate_multiplier=3)
            result = _run_continual_tasks(**arguments)
        resources = result["task_resource_metrics"]
        self.assertEqual([row["replay"]["candidate_count"] for row in resources], [0, 12, 12])
        self.assertEqual([row["replay"]["selected_count"] for row in resources], [0, 4, 4])
        self.assertEqual([row["training_class_counts"] for row in resources],
                         [{"0": 6}, {"0": 4, "1": 4}, {"0": 2, "1": 2, "2": 2}])

    def test_unbalanced_future_task_fails_before_any_fit(self) -> None:
        """Validate later current pools before starting an otherwise balanced first task."""
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "classifier.h5"
            fixtures.ContinualIntegrationTests._template(template)
            arguments = self._arguments(template)
            arguments["task_groups"] = [[0], [1, 2]]
            with patch("common.train.train_model", side_effect=AssertionError("Unbalanced stream started training")) as fit:
                with self.assertRaisesRegex(ValueError, "equal positive current-class counts"):
                    _run_continual_tasks(**arguments)
            fit.assert_not_called()


# Permit focused standalone execution without changing notebook kernels.
if __name__ == "__main__":
    unittest.main()
