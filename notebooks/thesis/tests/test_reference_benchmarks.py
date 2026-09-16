"""Focused reference-protocol checks; synthetic fixtures are not thesis results."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from common.dataloader import load_cifar10
from common.learner import _load_continual_arrays
from diffusion.models.transformer.di_t_classifier import DiTClassifier
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier
from notebooks.thesis import reference_benchmarks as reference
from semantic_consolidation.config import load_route_config


NOTEBOOKS = Path(__file__).resolve().parents[1]
TEMPLATES = {name: NOTEBOOKS / "configs" / f"{name}.yaml"
             for name in ("cifar10", "cifar100")}


def _synthetic_cifar() -> tuple:
    """Give training rows unique pixel IDs and reserve white images for test."""
    labels = np.repeat(np.arange(4, dtype="uint8"), 10)
    values = np.arange(1, len(labels) + 1, dtype="uint8")
    images = np.broadcast_to(values[:, None, None, None], (40, 32, 32, 3)).copy()
    test_labels = np.repeat(np.arange(4, dtype="uint8"), 2)
    test_images = np.full((8, 32, 32, 3), 255, dtype="uint8")
    return (images, labels[:, None]), (test_images, test_labels[:, None])


def _small_template(directory: Path) -> Path:
    """Materialize a tiny inherited platform without altering the central YAML."""
    config = load_route_config(TEMPLATES["cifar10"])
    config.common.dataset.batch_size = 7
    config.common.dataset.validation_ratio = .2
    config.common.model.show_network_summary = False
    config.common.model.kwargs.update(
        dim=8, depth=1, patch_size=8, mha_num_heads=1,
        clf_mha_num_heads=1, timesteps=4, compile_args={"run_eagerly": True})
    config.common.model.wrapper_kwargs.update(test_steps=2)
    config.common.training.epochs = 1
    config.common.training.verbose = 0
    config.common.continually_learn.class_num = 4
    config.common.continually_learn.class_order = None
    config.common.continually_learn.task_groups = None
    config.common.continually_learn.task_size = 2
    config.route.experimental["enabled"] = False
    path = directory / "synthetic-platform.yaml"
    path.write_text(yaml.safe_dump(asdict(config)), encoding="utf-8")
    return path


def _row_pairs(dataset: tf.data.Dataset) -> np.ndarray:
    """Recover unique pixel IDs and associated targets without depending on shuffle."""
    rows = np.concatenate([
        np.column_stack((np.asarray(images)[:, 0, 0, 0], np.asarray(labels).reshape(-1)))
        for images, labels in dataset
    ])
    return rows[np.argsort(rows[:, 0])]


class ReferenceConfigurationTests(unittest.TestCase):
    """Pair the reference protocols and retain the declared thesis platform."""

    def test_both_datasets_preserve_recipe_and_match_seeded_schedules(self) -> None:
        """Keep paired schedules, architecture and objectives without rewriting recipes."""
        before = {path: path.read_bytes() for path in TEMPLATES.values()}
        for dataset, classes, task_size in (("cifar10", 10, 2), ("cifar100", 100, 10)):
            original = load_route_config(TEMPLATES[dataset]).common
            for seed in (17, 1103):
                configs = [reference.configure_reference(dataset, name, seed=seed)
                           for name in reference.BENCHMARKS]
                expected = np.random.default_rng(seed).permutation(classes).tolist()
                for config in configs:
                    with self.subTest(dataset=dataset, benchmark=config.training.task, seed=seed):
                        control = config.continually_learn
                        self.assertEqual(control.class_order, expected)
                        self.assertEqual(control.task_groups,
                                         [expected[i:i + task_size] for i in range(0, classes, task_size)])
                        self.assertEqual(config.dataset.indices, expected)
                        self.assertEqual(control.seed, seed)
                        self.assertEqual(config.training.seed, seed)
                        self.assertEqual(control.experiment_phase, "development")
                        self.assertFalse(control.use_buffer)
                        self.assertFalse(control.use_generative_replay)
                        self.assertFalse(control.use_distillation)
                        self.assertEqual(control.replay_old_examples, 0)
                        self.assertIsNone(control.replay_current_examples)
                        self.assertEqual(config.model.name, "dit_classifier")
                        for name in ("dim", "depth", "patch_size", "clf_depth"):
                            self.assertEqual(config.model.kwargs[name], original.model.kwargs[name])
                        for name in ("noise_loss_coef", "clf_loss_coef"):
                            self.assertEqual(config.model.wrapper_kwargs[name],
                                             original.model.wrapper_kwargs[name])
                            self.assertGreater(config.model.wrapper_kwargs[name], 0)
                        for name in ("clf_distil_loss_coef", "noise_distil_loss_coef"):
                            self.assertEqual(config.model.wrapper_kwargs[name], 0)
                        self.assertEqual(config.training.epochs, original.training.epochs)
                        self.assertEqual(config.optimizer, original.optimizer)
                        self.assertEqual(config.dataset.validation_ratio, original.dataset.validation_ratio)
                        self.assertEqual(config.hpo["reference_benchmark"]["evaluation_split"], "validation")
                        self.assertFalse(config.hpo["reference_benchmark"]["confirmation_campaign_member"])
                offline, naive = configs
                self.assertEqual(offline.training.task, "joint")
                self.assertEqual(offline.model.kwargs["num_classes"], classes)
                self.assertEqual(naive.training.task, "continual")
                self.assertEqual(naive.continually_learn.baseline, "joint_none")
                self.assertTrue(naive.continually_learn.keep_same_model)
                self.assertTrue(naive.continually_learn.remove_prev_classes)
                self.assertTrue(naive.continually_learn.use_generative_model_classifier)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_test_access_is_explicit_and_not_confirmation(self) -> None:
        """Require an explicit test split and retain its supplemental status."""
        for benchmark in reference.BENCHMARKS:
            config = reference.configure_reference("cifar10", benchmark, evaluation_split="test")
            self.assertEqual(config.continually_learn.experiment_phase, "legacy")
            self.assertEqual(config.hpo["reference_benchmark"]["evaluation_split"], "test")
            self.assertFalse(config.hpo["reference_benchmark"]["confirmation_campaign_member"])
            self.assertIsNone(config.continually_learn.experiment_manifest_hash)

    def test_invalid_protocol_and_reintroduced_retention_fail_before_preparation(self) -> None:
        """Reject malformed or retention-enabled references before accessing data."""
        for options in ({"dataset": "mnist"}, {"benchmark": "cumulative"},
                        {"evaluation_split": "train"}, {"seed": True}, {"seed": -1}):
            kwargs = {"dataset": "cifar10", "benchmark": "offline_joint", **options}
            with self.subTest(options=options), self.assertRaises(ValueError):
                reference.configure_reference(**kwargs)
        config = reference.configure_reference("cifar10", "naive_sequential")
        config.continually_learn.use_generative_replay = True
        with patch.object(reference, "get_datasets", side_effect=AssertionError("No data access")):
            with self.assertRaisesRegex(ValueError, "disabled"):
                reference.prepare_reference(config)


class ReferenceExecutionTests(unittest.TestCase):
    """Exercise native loading, fitting and saved outcomes on small synthetic pixels."""

    def tearDown(self) -> None:
        """Release this test process's Keras objects after each isolated check."""
        tf.keras.backend.clear_session()

    def test_offline_inputs_match_native_split_remapping_and_keep_partial_batches(self) -> None:
        """Match native row partitions and target identities without dropping examples."""
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_REFERENCE_SPLITS_") as temporary:
            directory = Path(temporary)
            template = _small_template(directory)
            offline = reference.configure_reference("cifar10", "offline_joint", config_path=template,
                                                    results_root=directory / "runs")
            naive = reference.configure_reference("cifar10", "naive_sequential", config_path=template,
                                                  results_root=directory / "runs")
            with patch("tensorflow.keras.datasets.cifar10.load_data", return_value=_synthetic_cifar()):
                expected, _ = _load_continual_arrays(
                    load_cifar10, naive.continually_learn.class_order, False,
                    {"preprocess": naive.dataset.preprocess, "onehot_labels": False,
                     "validation_ratio": naive.dataset.validation_ratio,
                     "features_path": None, "seed": naive.training.seed},
                    None, None, 0, naive.training.seed)
                context = reference.prepare_reference(offline)
            train_rows, validation_rows = _row_pairs(context["trainset"]), _row_pairs(context["valset"])
            for actual, x, y in ((train_rows, expected[0], expected[1]),
                                 (validation_rows, expected[2], expected[3])):
                wanted = np.column_stack((x[:, 0, 0, 0], np.asarray(y).reshape(-1)))
                wanted = wanted[np.argsort(wanted[:, 0])]
                np.testing.assert_allclose(actual, wanted, atol=1e-7)
            self.assertEqual((len(train_rows), len(validation_rows)), (32, 8))
            self.assertEqual(len(set(train_rows[:, 0]) & set(validation_rows[:, 0])), 0)
            self.assertEqual([len(x) for x, _ in context["trainset"]], [7, 7, 7, 7, 4])
            self.assertEqual([len(x) for x, _ in context["valset"]], [7, 1])
            np.testing.assert_array_equal(context["evaluation_arrays"][0], expected[2])
            np.testing.assert_array_equal(context["evaluation_arrays"][1], expected[3])
            # Pixel IDs encode original labels; targets must follow the shuffled schedule.
            mapping = {label: index for index, label in enumerate(naive.continually_learn.class_order)}
            for pixel, target in np.concatenate((train_rows, validation_rows)):
                pixel_id = int(round((pixel + 1) * 127.5))
                self.assertEqual(int(target), mapping[(pixel_id - 1) // 10])
            self.assertFalse(context["training_started"])
            self.assertFalse(context["training_finished"])
            with self.assertRaisesRegex(RuntimeError, "Complete training"):
                reference.finish_reference(offline, context, {})

    def test_both_references_fit_save_and_report_without_replay_or_test_predictions(self) -> None:
        """Run both actual training routes while rejecting replay and test leakage."""
        original_predict = DiTClassifier.predict_class
        observed_prediction_sizes = []

        def guarded_predict(model: DiTClassifier, inputs: tuple, *args: object, **kwargs: object) -> object:
            """Reject reserved test images before forwarding primary-head predictions."""
            images = np.asarray(inputs[0])
            self.assertFalse(np.any(np.all(np.isclose(images, 1.), axis=(1, 2, 3))),
                             "Validation reference predicted a locked test sentinel.")
            observed_prediction_sizes.append(len(images))
            return original_predict(model, inputs, *args, **kwargs)

        central_before = {path: path.read_bytes() for path in TEMPLATES.values()}
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_REFERENCE_SMOKE_") as temporary:
            directory = Path(temporary)
            template = _small_template(directory)
            with patch("tensorflow.keras.datasets.cifar10.load_data", return_value=_synthetic_cifar()), \
                    patch.object(DiTClassifier, "predict_class", new=guarded_predict), \
                    patch.object(DiffusionClassifier, "sample", side_effect=AssertionError("Reference sampled replay")):
                for benchmark in reference.BENCHMARKS:
                    with self.subTest(benchmark=benchmark):
                        config = reference.configure_reference("cifar10", benchmark, config_path=template,
                                                               results_root=directory / "runs")
                        context = reference.prepare_reference(config)
                        history = reference.train_reference(config, context)
                        summary, per_task = reference.finish_reference(config, context, history)
                        run = context["run_dir"]
                        self.assertTrue(context["training_finished"])
                        self.assertEqual(json.loads((run / "status.json").read_text())["state"], "completed")
                        self.assertEqual(json.loads((run / "summary.json").read_text()), summary)
                        pd.testing.assert_frame_equal(pd.read_csv(run / "final_per_task_accuracy.csv"), per_task)
                        self.assertTrue((run / "reference_config.yaml").is_file())
                        self.assertTrue((run / "reference_plan.json").is_file())
                        self.assertTrue(list(Path(config.training.results_path).glob("*.weights.h5")))
                        self.assertEqual(summary["metric_scale"], "fraction")
                        self.assertEqual(summary["evaluation_split"], "validation")
                        self.assertEqual(len(per_task), 2)
                        self.assertTrue(per_task.accuracy.between(0, 1).all())
                        np.testing.assert_allclose(per_task.accuracy_percent, 100 * per_task.accuracy)
                        self.assertAlmostEqual(summary["final_average_accuracy"], per_task.accuracy.mean())
                        # Offline learning has no temporal trajectory to score.
                        if benchmark == "offline_joint":
                            model = context["model"]
                            self.assertFalse((run / "accuracy_matrix.csv").exists())
                            for metric in ("average_incremental_accuracy", "average_forgetting", "backward_transfer"):
                                self.assertIsNone(summary[metric])
                        # Naive learning must expose the complete native task trajectory.
                        else:
                            model = context["model"]["generative_model"]
                            details = context["model"]["continual_details"]
                            matrix = np.asarray(details["accuracy_matrix"])
                            self.assertEqual(matrix.shape, (2, 2))
                            self.assertTrue(np.isnan(matrix[0, 1]))
                            self.assertTrue(np.isfinite(matrix[np.tril_indices(2)]).all())
                            self.assertFalse(details["test_evaluated"])
                            self.assertEqual(details["ordinary_accuracy_matrix"], [])
                            self.assertEqual(details["baseline"], "joint_none")
                            self.assertTrue((run / "accuracy_matrix.csv").is_file())
                            for resource in details["task_resource_metrics"]:
                                self.assertEqual(resource["replay"]["source"], "none")
                                self.assertEqual(resource["replay"]["candidate_count"], 0)
                                self.assertEqual(resource["replay"]["selected_count"], 0)
                                self.assertEqual(resource["training_examples_total"], 16)
                        self.assertIsInstance(model, DiffusionClassifier)
                        self.assertIsNone(model.teacher_network)
                        self.assertIsNone(getattr(model, "route_controller", None))
                        self.assertFalse(model.use_clf_distil_loss)
                        self.assertFalse(model.use_noise_distil_loss)
                        with self.assertRaisesRegex(RuntimeError, "fresh kernel"):
                            reference.train_reference(config, context)
                        with patch.object(reference, "report", side_effect=AssertionError("Duplicate report")):
                            again, table = reference.finish_reference(config, context, history)
                            self.assertIs(again, summary)
                            self.assertIs(table, per_task)
                        tf.keras.backend.clear_session()
        self.assertGreater(len(observed_prediction_sizes), 0)
        self.assertEqual(central_before, {path: path.read_bytes() for path in central_before})


# Support standalone execution as well as unittest discovery.
if __name__ == "__main__":
    unittest.main()
