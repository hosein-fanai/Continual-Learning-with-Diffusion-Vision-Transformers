"""Held-out Section 11 observers and scientifically labeled reference controls."""

from __future__ import annotations

from copy import deepcopy
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from semantic_consolidation.experimental import (
    ExperimentalController, MemoryMonitor, classification_outcomes, validate_experimental,
)


def project_fixture() -> SimpleNamespace:
    """Supply only training-stream configuration, with no test-data source."""
    return SimpleNamespace(
        training=SimpleNamespace(use_valset=True),
        dataset=SimpleNamespace(validation_ratio=0.2),
        continually_learn=SimpleNamespace(use_distillation=False, use_generative_replay=False,
            class_num=4, class_order=[0, 1, 2, 3], task_groups=[[0, 1], [2, 3]],
            task_size=2, remove_prev_classes=True),
        model=SimpleNamespace(wrapper_kwargs={
            "noise_loss_coef": 0., "noise_distil_loss_coef": 0., "clf_distil_loss_coef": 0.,
            "image_loss_coef": 0., "kl_loss_coef": 0., "ctr_loss_coef": 0.,
            "clf_loss_coef": 1., "train_noisified_min_timesteps": 0,
            "train_noisified_max_timesteps": 0, "p_uncond": 1.,
            "use_ensemble_loss_instead": False,
        }),
    )


class ClassificationOutcomeTests(unittest.TestCase):
    """Hand-check confusion directions, class support and calibrated probabilities."""

    def test_old_new_recall_confusion_and_nll(self) -> None:
        """Verify old new recall confusion and nll."""
        probabilities = np.asarray([[.7, .1, .1, .1], [.1, .1, .7, .1],
                                    [.1, .6, .2, .1], [.1, .1, .1, .7]])
        result = classification_outcomes(probabilities, [0, 0, 2, 3], old_count=2, bins=5)
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["old_accuracy"], .5)
        self.assertEqual(result["new_accuracy"], .5)
        self.assertEqual(result["class_counts"], [2, 0, 1, 1])
        self.assertEqual(result["per_class_recall"], [.5, None, 0., 1.])
        self.assertEqual(result["old_to_new_errors"], 1)
        self.assertEqual(result["new_to_old_errors"], 1)
        self.assertAlmostEqual(result["nll"], -np.log([.7, .1, .2, .7]).mean())
        self.assertEqual(result["ece_protocol"]["bins"], 5)

    def test_absent_old_group_is_unavailable_and_column_targets_are_supported(self) -> None:
        """Verify absent old group is unavailable and column targets are supported."""
        result = classification_outcomes([[.8, .2], [.1, .9]], [[0], [1]], 0, 4)
        self.assertIsNone(result["old_accuracy"])
        self.assertEqual(result["new_accuracy"], 1.)
        self.assertEqual(result["old_to_new_errors"], 0)

    def test_fractional_labels_and_ambiguous_old_support_are_rejected(self) -> None:
        """Verify fractional labels and ambiguous old support are rejected."""
        with self.assertRaises(ValueError):
            classification_outcomes([[.8, .2]], [.9], 0, 5)
        for count in (-1, 3, True, .5):
            with self.subTest(old_count=count), self.assertRaises(ValueError):
                classification_outcomes([[.8, .2]], [0], count, 5)
        for bins in (True, 1.5, 0):
            with self.subTest(bins=bins), self.assertRaises(ValueError):
                classification_outcomes([[.8, .2]], [0], 0, bins)

    def test_unnormalized_scores_are_never_interpreted_as_probabilities(self) -> None:
        """Verify unnormalized scores are never interpreted as probabilities."""
        with self.assertRaises(ValueError):
            classification_outcomes([[8., 2.]], [0], 0, 5)


class ExperimentalValidationTests(unittest.TestCase):
    """A reference label must enforce the training treatment it claims."""

    def test_clean_reference_requires_clean_supervision_without_replay(self) -> None:
        """Verify clean reference requires clean supervision without replay."""
        project = project_fixture()
        validate_experimental(project, {"reference": "clean_finetune"}, "baseline")
        for name, value in (("noise_loss_coef", 1.), ("train_noisified_max_timesteps", 4),
                            ("p_uncond", .1), ("use_ensemble_loss_instead", True),
                            ("clf_loss_coef", 0.)):
            changed = deepcopy(project)
            changed.model.wrapper_kwargs[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_experimental(changed, {"reference": "clean_finetune"}, "baseline")
        project.continually_learn.use_generative_replay = True
        with self.assertRaises(ValueError):
            validate_experimental(project, {"reference": "clean_finetune"}, "baseline")

    def test_offline_reference_uses_the_standalone_common_joint_runner(self) -> None:
        """Verify offline reference uses the standalone common joint runner."""
        project = project_fixture()
        with self.assertRaisesRegex(ValueError, "reference"):
            validate_experimental(project, {"reference": "offline_joint"}, "baseline")
        project.continually_learn.task_groups = [[0, 1, 2, 3]]
        with self.assertRaisesRegex(ValueError, "reference"):
            validate_experimental(project, {"reference": "offline_joint"}, "baseline")
        with self.assertRaises(ValueError):
            validate_experimental(project, {"reference": "offline_joint"}, "learned")

    def test_online_observation_needs_training_validation_and_two_kid_rows(self) -> None:
        """Verify online observation needs training validation and two kid rows."""
        project = project_fixture()
        for values in ({"generation_per_class": 1}, {"enabled": "false"}, {"ece_bins": 1.5},
                       {"feature_extractor": "undeclared_pretrained_model"}, {"test_data": True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_experimental(project, values)
        project.training.use_valset = False
        with self.assertRaisesRegex(ValueError, "held-out validation"):
            validate_experimental(project, {"enabled": True})


class ExperimentalObserverTests(unittest.TestCase):
    """Verify dense classifier targets and original-label fixed cohorts stay distinct."""

    def fixture(self) -> tuple:
        """Build a small isolated observer and synthetic validation cohort for runtime checks."""
        project = project_fixture()
        project.continually_learn.class_order = [8, 2]
        with mock.patch("semantic_consolidation.experimental.MemoryMonitor"):
            observer = ExperimentalController(project, {"probe_per_class": 2, "generation_per_class": 2}, 7)
        self.addCleanup(observer.close)
        wrapper = SimpleNamespace(seen_classes={8: 0, 2: 1}, network=SimpleNamespace(num_classes=2, weights=[]),
            teacher_network=None, optimizer=SimpleNamespace(variables=[]))
        images = np.zeros((2, 2, 2, 1), dtype="float32")
        dataset = tf.data.Dataset.from_tensor_slices((images, np.asarray([8, 2], dtype="int32"))).batch(2)
        return observer, wrapper, images, dataset

    def test_validation_does_not_request_new_data_and_preserves_existing_callbacks(self) -> None:
        """Verify validation does not request new data and preserves existing callbacks."""
        observer, wrapper, images, dataset = self.fixture()
        callback = tf.keras.callbacks.Callback()
        with mock.patch("common.dataloader.get_datasets", side_effect=AssertionError("data access")):
            arguments = observer.before_task(wrapper, dataset, {"callbacks": [callback], "epochs": 2})
        self.assertIs(arguments["callbacks"][0], callback)
        self.assertEqual(len(arguments["callbacks"]), 2)
        self.assertEqual(arguments["epochs"], 2)
        np.testing.assert_array_equal(observer.validation[0], images)
        np.testing.assert_array_equal(observer.validation[1], [0, 1])

    def test_shuffled_original_class_ids_reach_hidden_probe(self) -> None:
        """Verify shuffled original class ids reach hidden probe."""
        observer, wrapper, images, dataset = self.fixture()
        observer.before_task(wrapper, dataset, {})
        observer.probe.observe = mock.Mock(return_value={"retained_bytes": {}})
        observer._classify = mock.Mock(return_value=({}, {}))
        observer.after_task(wrapper)
        np.testing.assert_array_equal(observer.probe.observe.call_args.args[2], [8, 2])
        np.testing.assert_array_equal(observer._classify.call_args.args[2], [0, 1])
        self.assertEqual(observer.records[0]["split"], "validation")
        self.assertEqual(observer.old_count, 2)
        self.assertIsNone(observer.validation)

    def test_predictor_never_receives_true_labels_or_task_identity(self) -> None:
        """Verify predictor never receives true labels or task identity."""
        observer, wrapper, images, _ = self.fixture()
        with mock.patch("semantic_consolidation.evaluation._predict", return_value=(np.asarray([[.8, .2], [.3, .7]]), {})) as predictor:
            outcome, _ = observer._classify(wrapper, images, np.asarray([0, 1]))
        self.assertEqual(outcome["accuracy"], 1.)
        self.assertEqual(len(predictor.call_args.args), 5)
        self.assertIs(predictor.call_args.args[0], wrapper)
        self.assertIs(predictor.call_args.args[1], images)
        self.assertIsNone(predictor.call_args.args[3])

    def test_candidate_reservoir_is_chunk_invariant_and_preserves_duplicate_occurrences(self) -> None:
        """Verify candidate reservoir is chunk invariant and preserves duplicate occurrences."""
        first, wrapper, _, _ = self.fixture()
        second, _, _, _ = self.fixture()
        wrapper.use_cfg = True
        for observer in (first, second):
            observer.old_count = 2
            observer.accepting_candidates = True
        images = np.arange(24, dtype="float32").reshape(6, 2, 2, 1) / 24.
        labels = np.asarray([1, 2, 1, 2, 1, 2])
        first.capture(wrapper, images, labels, .3)
        second.capture(wrapper, images[:3], labels[:3], .1)
        second.capture(wrapper, images[3:], labels[3:], .2)
        self.assertEqual(first.generated_counts, {0: 3, 1: 3})
        self.assertEqual([(row[0], row[1]) for row in first.candidates], [(row[0], row[1]) for row in second.candidates])
        np.testing.assert_array_equal(np.stack([row[2] for row in first.candidates]), np.stack([row[2] for row in second.candidates]))
        duplicate, _, _, _ = self.fixture()
        duplicate.old_count = 2
        duplicate.accepting_candidates = True
        duplicate.capture(wrapper, images[:1].repeat(2, axis=0), [1, 1], .1)
        self.assertEqual([row[1] for row in duplicate.candidates], [1, 2])
        self.assertEqual(duplicate.generated_counts, {0: 2})


class MemoryMonitorTests(unittest.TestCase):
    """Unavailable hardware remains unknown and the sampling thread can stop."""

    def test_cpu_only_monitor_does_not_claim_device_peak(self) -> None:
        """Verify cpu only monitor does not claim device peak."""
        with mock.patch("tensorflow.config.list_logical_devices", return_value=[]), mock.patch.dict(sys.modules, {"psutil": None}):
            monitor = MemoryMonitor()
        monitor.start()
        report = monitor.snapshot()
        monitor.close()
        self.assertEqual(report["tf_allocator_devices"], {})
        self.assertIsNone(report["sampled_process_peak_rss_bytes"])
        self.assertEqual(report["rss_samples"], 0)

    def test_allocator_failure_remains_unavailable(self) -> None:
        """Verify allocator failure remains unavailable."""
        with mock.patch("tensorflow.config.list_logical_devices", return_value=[SimpleNamespace(name="/device:GPU:0")]), mock.patch(
            "tensorflow.config.experimental.reset_memory_stats", side_effect=ValueError("unsupported")), mock.patch.dict(sys.modules, {"psutil": None}):
            monitor = MemoryMonitor()
        self.assertEqual(monitor.snapshot()["tf_allocator_devices"], {})
        monitor.close()


# Run this module directly while keeping imports free of execution side effects.
if __name__ == "__main__":
    unittest.main()
