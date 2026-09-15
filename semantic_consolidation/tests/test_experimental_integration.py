"""Full Section 11 observation through both real route runners on tiny pixels."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from allocation_study.experimental_analysis import audit_matrix_metrics
from semantic_consolidation.tests import test_integration as fixtures


ROOT = Path(__file__).resolve().parents[2]


class ExperimentalIntegrationTests(unittest.TestCase):
    """Exercise actual tiny route fits, replay, and observer invariants end to end."""
    def test_both_routes_observe_real_training_and_replay(self) -> None:
        """Verify both routes observe real training and replay."""
        from semantic_consolidation.config import load_route_config as semantic_config
        from semantic_consolidation.runner import run as semantic_run
        from gist_memory.config import load_route_config as gist_config
        from gist_memory.runner import run as gist_run
        for route, loader, run in (("semantic_consolidation", semantic_config, semantic_run),
                                   ("gist_memory", gist_config, gist_run)):
            with self.subTest(route=route), tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
                config = loader(ROOT / route / "configs/smoke.yaml")
                config.common.training.results_path = directory
                config.route.experimental = {"enabled": True, "probe_per_class": 2,
                    "generation_per_class": 2, "batch_size": 8, "ece_bins": 4,
                    "feature_extractor": "fixed_pixels", "learning_curves": True}
                with patch("tensorflow.keras.datasets.mnist.load_data", side_effect=fixtures.RouteIntegrationTests._pixels):
                    result = run(config)
                records = result["experimental_records"]
                self.assertEqual(len(records), 2)
                self.assertFalse(records[0]["generated_memory"]["available"])
                self.assertTrue(records[1]["generated_memory"]["available"])
                self.assertEqual(records[1]["generated_memory"]["kid_classes_expected"], 2)
                self.assertEqual(records[1]["outcomes"]["old_class_count"], 2)
                self.assertGreater(records[1]["tensor_inventory"]["unique_tensor_bytes"], 0)
                self.assertGreater(records[1]["tensor_inventory"]["groups"]["factory_classifier_template"]["tensor_bytes"], 0)
                self.assertEqual([row["post_boundary_teacher"]["class_count"] for row in records], [2, 4])
                self.assertEqual(records[1]["post_boundary_teacher"]["inventory"]["unique_tensor_bytes"],
                                 records[1]["tensor_inventory"]["groups"]["raw"]["tensor_bytes"])
                self.assertGreater(records[1]["historical_validation_rows"], 0)
                self.assertEqual(np.asarray(records[1]["outcomes"]["confusion_matrix"]).shape, (4, 4))
                self.assertIsNotNone(records[1]["hidden"]["per_class"]["0"]["since_acquisition"])
                details = result["model"]["continual_details"]
                self.assertFalse(details["test_evaluated"])
                self.assertTrue(audit_matrix_metrics(details["validation_accuracy_matrix"], expected_tasks=2)["passed"])
                output = Path(result["results_path"])
                saved = json.loads((output / "section11.json").read_text())
                self.assertEqual(len(saved["learning_curves"]), 2)
                with np.load(output / "generated_examples_task_002.npz", allow_pickle=False) as samples:
                    self.assertEqual(len(samples["images"]), 4)
                    self.assertTrue(np.isfinite(samples["images"]).all())
                self.assertTrue((output / "learning_curves.csv").is_file())
                tf.keras.backend.clear_session()

    def test_observation_preserves_baseline_weights(self) -> None:
        """Verify observation preserves baseline weights."""
        from semantic_consolidation.config import load_route_config
        from semantic_consolidation.runner import run
        weights = []
        for enabled in (False, True):
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
                config = load_route_config(ROOT / "semantic_consolidation/configs/smoke.yaml")
                config.common.training.results_path = directory
                config.route.condition = "baseline"
                config.route.experimental = {"enabled": enabled, "probe_per_class": 2,
                                             "generation_per_class": 2, "batch_size": 8}
                with patch("tensorflow.keras.datasets.mnist.load_data", side_effect=fixtures.RouteIntegrationTests._pixels):
                    result = run(config)
                weights.append([value.numpy().copy() for value in result["model"]["generative_model"].network.weights])
                tf.keras.backend.clear_session()
        for before, after in zip(*weights):
            np.testing.assert_allclose(before, after, rtol=0., atol=0.)


# Run this module directly while keeping imports free of execution side effects.
if __name__ == "__main__":
    unittest.main()
