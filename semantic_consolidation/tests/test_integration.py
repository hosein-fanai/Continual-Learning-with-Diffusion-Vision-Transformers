"""Real two-task route integration, replacing only downloaded MNIST pixels.

The common configuration, loaders, factory, joint training, generated replay,
classifier/noise KD, modulation acquisition, consolidation, reporting, and
checkpoint reload all execute normally. Tiny synthetic images test plumbing
and invariants; their accuracy is not a dataset benchmark.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tensorflow as tf

from common.config import load_config
from common.continual_reporting import continual_metrics
from common.dataloader import get_dataset
from common.model import get_model
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier
from semantic_consolidation.config import RouteSettings, load_route_config
from semantic_consolidation.controller import RouteController
from semantic_consolidation.memory import ModulationBank
from semantic_consolidation.model import SemanticConsolidationClassifier
from semantic_consolidation.runner import load_inference_model, run


_ROOT = Path(__file__).resolve().parents[2]


class RouteIntegrationTests(unittest.TestCase):
    """Exercise the complete route and its separation from inference artifacts."""

    @staticmethod
    def _pixels() -> tuple:
        """Generate four classes with distinct stripes and independent test rows."""

        rng = np.random.default_rng(23)
        labels = np.repeat(np.arange(4, dtype="uint8"), 16)
        images = rng.integers(0, 40, size=(len(labels), 28, 28), dtype="uint8")
        for image, label in zip(images, labels):
            image[2 + 6 * int(label):6 + 6 * int(label), 4:24] = 215
        test_labels = np.repeat(np.arange(4, dtype="uint8"), 4)
        test_images = np.full((len(test_labels), 28, 28), 251, dtype="uint8")
        return (images, labels), (test_images, test_labels)

    def test_joint_replay_three_phases_reporting_and_inference_reload(self) -> None:
        """Verify actual optimization, frozen boundaries, metrics and saved weights."""

        config = load_route_config(_ROOT / "semantic_consolidation/configs/smoke.yaml")
        temporary_root = _ROOT / ".tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="route-integration-", dir=temporary_root, ignore_cleanup_errors=True,
        ) as directory, patch(
            "tensorflow.keras.datasets.mnist.load_data", side_effect=self._pixels,
        ):
            config.common.training.results_path = directory
            result = run(config)
            wrapper = result["model"]["generative_model"]
            details = result["model"]["continual_details"]
            records = result["route_records"]
            self.assertIsInstance(wrapper, SemanticConsolidationClassifier)
            self.assertEqual(wrapper.network.num_classes, 4)
            self.assertEqual(wrapper.seen_classes, {0: 0, 1: 1, 2: 2, 3: 3})
            self.assertEqual(details["task_classes"], [[2, 0], [3, 1]])
            self.assertEqual(details["ordinary_accuracy_matrix"], [])
            self.assertFalse(details["test_evaluated"])
            self.assertEqual(len(records), 2)
            for record, expected_joint in zip(records, (3, 5)):
                self.assertEqual(record["joint_updates"], expected_joint)
                self.assertEqual(record["acquisition"]["updates"], 2)
                self.assertEqual(record["consolidation"]["updates"], 1)
                self.assertEqual(record["total_updates"], expected_joint + 3)
                self.assertTrue(all(record["invariants"].values()))
                self.assertGreater(record["memory_bytes"]["consolidation_target"], 0)
                self.assertGreater(record["memory_bytes"]["persistent_modulators"], 0)
                for phase in ("acquisition", "consolidation"):
                    self.assertTrue(np.isfinite(record[phase]["history"]["loss"]).all())
                    self.assertTrue(record[phase]["gradient_variable_names"])
            self.assertEqual(sum(records[1]["pool_class_counts"].values()), 20)
            self.assertEqual(set(map(int, records[1]["pool_class_counts"])), {0, 1, 2, 3})
            history = details["generative_histories"][1]
            self.assertGreater(history["clf_distil_loss"][-1], 0.)
            self.assertGreater(history["noise_distil_loss"][-1], 0.)
            self.assertFalse(wrapper.teacher_network.trainable)
            self.assertEqual(wrapper.teacher_network.num_classes, 4)
            for actual, target in zip(wrapper.network.classifier.weights, wrapper.teacher_network.classifier.weights):
                np.testing.assert_array_equal(actual.numpy(), target.numpy())

            matrix = np.asarray(details["validation_accuracy_matrix"])
            self.assertEqual(matrix.shape, (2, 2))
            self.assertTrue(np.isnan(matrix[0, 1]))
            self.assertTrue(np.isfinite(matrix[np.tril_indices(2)]).all())
            self.assertEqual(
                result["evaluations"]["validation_continual_metrics"], continual_metrics(matrix),
            )
            output = Path(result["results_path"])
            for filename in (
                "config.yaml", "input_config.yaml", "route.settings.yaml",
                "route_metrics.json", "route_resources.csv", "route_steps.csv",
                "accuracy_matrices.csv", "summary.csv", "modulators.npz",
                "source_provenance.json",
            ):
                self.assertTrue((output / filename).is_file(), filename)
            with (output / "route_metrics.json").open(encoding="utf-8") as stream:
                saved_records = json.load(stream)
            self.assertEqual(saved_records, records)
            with np.load(output / "modulators.npz", allow_pickle=False) as bank:
                self.assertEqual(len(bank.files), 8)
                self.assertTrue(all(np.isfinite(bank[name]).all() for name in bank.files))
            csv = pd.read_csv(output / "accuracy_matrices.csv")
            saved_matrix = csv[csv["matrix"] == "validation_accuracy_matrix"]["value"].to_numpy().reshape(2, 2)
            np.testing.assert_allclose(matrix, saved_matrix, equal_nan=True)
            common_config = load_config(output / "config.yaml")
            self.assertEqual(common_config.hpo["semantic_consolidation"]["noise_levels"], [0, 2])

            tracked_ids = {id(variable) for variable in wrapper.weights}
            bank_variables = [variable for pair in wrapper.route_controller.bank.vectors.values() for variable in pair]
            self.assertTrue(tracked_ids.isdisjoint({id(variable) for variable in bank_variables}))
            inputs = tf.reshape(tf.linspace(-1., 1., 2 * 28 * 28), (2, 28, 28, 1))
            times = tf.zeros((2,), dtype=tf.int32)
            before = wrapper.network.predict_class((inputs, times, tf.zeros_like(times)), training=False)
            restored = load_inference_model(output / "config.yaml")
            self.assertIs(type(restored), DiffusionClassifier)
            self.assertFalse(hasattr(restored, "route_controller"))
            after = restored.network.predict_class((inputs, times, tf.zeros_like(times)), training=False)
            np.testing.assert_allclose(before.numpy(), after.numpy(), rtol=1e-6, atol=1e-7)

    def test_mechanistic_controls_run_with_matched_phase_updates(self) -> None:
        """Exercise every mechanism control from the same acquired tiny network."""

        values = np.linspace(-1., 1., 8 * 4 * 4, dtype="float32").reshape(8, 4, 4, 1)
        labels = np.repeat(np.arange(2, dtype="int32"), 4)
        dataset = get_dataset(values, labels, batch_size=4, shuffle_buffer=0, drop_remainder=False)
        base = get_model(
            model_name="dit_classifier", task="joint", image_shape=(4, 4, 1),
            class_num=2, seed=29, dtype_policy="float32", show_network_summary=False,
            model_kwargs={
                "timesteps": 8, "patch_size": 2, "dim": 4, "depth": 1,
                "mha_num_heads": 1, "vit_block_mlp_ratio": 1.,
                "clf_mha_num_heads": 1, "clf_vit_block_mlp_ratio": 1.,
                "classifier_mlp_ratio": 1, "compile_args": {"run_eagerly": True},
            },
            wrapper_kwargs={
                "use_ema": False, "p_uncond": 1., "clf_loss_coef": 1.,
                "test_noisified_max_timesteps": 0, "test_steps": 2,
            },
        )
        base.fit(dataset, epochs=1, verbose=0)
        controls = (
            ("baseline", {}), ("extra_joint", {}), ("random", {}),
            ("time_matched_joint", {"extra_joint_seconds": (0.001,)}),
            ("no_consolidation", {}), ("feature_distillation", {}),
            ("unmodulated_feature_distillation", {}),
            ("learned", {"retain_modulators": False}),
            ("learned", {"consolidation_scope": "backbone"}),
            ("learned", {"acquisition_objective": "true_class_ce"}),
        )
        for condition, changes in controls:
            with self.subTest(condition=condition, changes=changes):
                network = base.snapshot_teacher_network("raw")
                network.trainable = True
                constructor = dict(base.get_config())
                # The standalone acquired fixture has a fixed two-class head;
                # declare its natural label map when adapting it to route tasks.
                constructor.update(network=network, teacher_network=None, seen_classes={0: 0, 1: 1})
                wrapper = DiffusionClassifier(**constructor)
                wrapper.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse", run_eagerly=True)
                settings = replace(
                    RouteSettings(acquisition_steps=2, consolidation_steps=1,
                                  batch_size=4, noise_levels=(0, 2), seed=29),
                    condition=condition, **changes,
                )
                controller = RouteController(settings)
                record = controller.run(wrapper, dataset, {"route_joint_updates": 0})
                # The measured-time control reports actual elapsed work without constructing a gate bank.
                if condition == "time_matched_joint":
                    self.assertIsNone(controller.bank)
                    self.assertGreaterEqual(record["extra_joint_updates"], 1)
                    self.assertGreaterEqual(record["extra_joint_training_seconds"], 0.001)
                    self.assertEqual(record["total_updates"], record["extra_joint_updates"])
                    continue
                self.assertEqual(record["total_updates"], 0 if condition == "baseline" else 3)
                # Platform controls have no semantic memory and use only their declared extra joint
                # updates.
                if condition in ("baseline", "extra_joint"):
                    self.assertIsNone(controller.bank)
                    self.assertEqual(record["extra_joint_updates"], 3 if condition == "extra_joint" else 0)
                    continue
                self.assertEqual(record["acquisition"]["updates"], 2)
                self.assertEqual(record["consolidation"]["updates"], 1)
                self.assertEqual(sorted(record["acquisition"]["focus_class_updates"].values()), [1, 1])
                self.assertEqual(record["acquisition"]["untrained_focus_classes"], [])
                self.assertTrue(record["invariants"]["consolidation_target_unchanged"])
                self.assertTrue(record["invariants"]["modulators_frozen_during_consolidation"])
                # Semantic-only consolidation must preserve every nonsemantic model weight.
                if settings.consolidation_scope == "semantic":
                    self.assertTrue(record["invariants"]["nonsemantic_weights_unchanged"])
                # The whole-backbone ablation intentionally permits a shared-path change.
                else:
                    self.assertFalse(record["invariants"]["nonsemantic_weights_unchanged"])
                # Discarding gates removes the persistent parameter memory after consolidation.
                if not settings.retain_modulators:
                    self.assertEqual(controller.bank.vectors, {})
                    self.assertEqual(record["memory_bytes"]["persistent_modulators"], 0)
                # Random-control gates must match their deterministic untrained initialization.
                if condition == "random":
                    initial = ModulationBank(settings, controller.bank.dimension, settings.seed)
                    initial.add([0, 1])
                    for class_id, pair in controller.bank.vectors.items():
                        for actual, expected in zip(pair, initial.vectors[class_id]):
                            np.testing.assert_array_equal(actual.numpy(), expected.numpy())
                    self.assertTrue(record["acquisition"]["zero_learning_rate_control"])


# Run this module directly while keeping imports free of execution side effects.
if __name__ == "__main__":
    unittest.main()
