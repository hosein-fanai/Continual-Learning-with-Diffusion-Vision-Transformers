"""End-to-end section-10 controls on both real continual route implementations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def extension_settings(strategy: str = "drift") -> dict:
    """Keep every extension active while using bounded CPU integration budgets."""

    return {
        "schedule": {"mode": "drift", "wake_updates": 2, "replay_updates": 2,
                     "wake_block_updates": 1, "replay_block_updates": 1,
                     "batch_size": 4, "drift_threshold": 0.05},
        "replay": {"strategy": strategy, "candidate_multiplier": 2, "batch_size": 4,
                   "noise_levels": [0, 2], "quality_quantile": 0., "min_per_class": 2},
        "evaluation": {"enabled": True, "horizons": [1, 2], "batch_size": 4,
                       "calibration_fraction": 0.5, "seed": 17},
    }


class ExtensionConfigTests(unittest.TestCase):
    """Fail before training when an extension would imply a different protocol."""

    def test_nested_configuration_roundtrip_and_protocol_validation(self) -> None:
        """Exercise both route schemas, independent defaults and common API rules."""

        import yaml
        from gist_memory.config import RouteSettings, load_route_config, validate_route_config

        first, second = RouteSettings(), RouteSettings()
        first.extensions["schedule"] = {}
        self.assertEqual(second.extensions, {})
        config = load_route_config(ROOT / "gist_memory/configs/smoke.yaml")
        config.common.continually_learn.replay_candidate_multiplier = 2
        config.route.extensions = extension_settings()
        validate_route_config(config)
        restored = yaml.safe_load(yaml.safe_dump(config.route.extensions))
        self.assertEqual(config.route.extensions, restored)
        for mutation in (
            {"unknown": 1}, {"schedule": {"mode": "joint"}, "replay": {"strategy": "drift"}},
            {"schedule": {"mode": "drift"}}, {"evaluation": {"enabled": True, "network_name": "ema"}},
            {"evaluation": {"enabled": True, "horizons": [999]}},
            {"quality_fit_split": "test"},
        ):
            candidate = copy.deepcopy(config)
            candidate.route.extensions = mutation
            with self.subTest(mutation=mutation), self.assertRaises((TypeError, ValueError)):
                validate_route_config(candidate)

    def test_semantic_floor_and_paired_extension_manifests(self) -> None:
        """Protect positive-pair feasibility and authenticate each selection control."""

        from common.experiment import materialize_run_plan, read_experiment_manifest
        from semantic_consolidation.config import load_route_config, validate_route_config
        from semantic_consolidation.study import prepare_study, planned_config, validate_planned_config

        config = load_route_config(ROOT / "semantic_consolidation/configs/extensions_smoke.yaml")
        invalid = copy.deepcopy(config)
        invalid.route.extensions["replay"]["min_per_class"] = 1
        with self.assertRaisesRegex(ValueError, "min_per_class"):
            validate_route_config(invalid)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            path = prepare_study(config, Path(directory) / "planned", [17, 29], conditions={
                "drift": {}, "random": {"route": {"extensions": {"replay": {"strategy": "random"}}}},
            })
            manifest = read_experiment_manifest(path)
            entries = materialize_run_plan(manifest)
            self.assertEqual(len(entries), 4)
            for entry in entries:
                planned = planned_config(entry, path)
                validate_planned_config(planned)
                self.assertEqual(planned.route.extensions["replay"]["strategy"], entry["condition"])
                planned.route.extensions["evaluation"]["seed"] += 1
                with self.assertRaisesRegex(ValueError, "manifest"):
                    validate_planned_config(planned)


class ExtensionIntegrationTests(unittest.TestCase):
    """Execute both complete routes using synthetic pixels through common loaders."""

    def test_both_routes_schedule_select_evaluate_save_and_restore(self) -> None:
        """Check enlarged capture, live drift, budgets, teacher and inference state."""

        from semantic_consolidation.tests.test_integration import RouteIntegrationTests

        for name in ("semantic_consolidation", "gist_memory"):
            # Reuse each public route runner; replace only the download boundary.
            if name == "gist_memory":
                from gist_memory.config import load_route_config
                from gist_memory.runner import run, load_inference_model
            # The semantic route keeps its own public configuration and runner.
            else:
                from semantic_consolidation.config import load_route_config
                from semantic_consolidation.runner import run, load_inference_model
            config = load_route_config(ROOT / name / "configs/smoke.yaml")
            config.common.continually_learn.replay_candidate_multiplier = 2
            config.route.extensions = extension_settings()
            with self.subTest(route=name), tempfile.TemporaryDirectory(
                prefix=f"section10-{name}-", ignore_cleanup_errors=True,
            ) as directory, patch("tensorflow.keras.datasets.mnist.load_data", side_effect=RouteIntegrationTests._pixels):
                config.common.training.results_path = directory
                result = run(config)
                wrapper = result["model"]["generative_model"]
                controller = wrapper.section10_controller
                self.assertEqual(len(controller.records), 2)
                first, second = controller.records
                self.assertEqual(first["schedule"]["updates"], 2)
                self.assertEqual(second["candidate_count"], 16)
                self.assertEqual(second["schedule"]["updates"], 4)
                self.assertEqual(second["schedule"]["presentations"], {"wake": 8, "replay": 8})
                self.assertTrue(second["prior_teacher_unchanged"])
                self.assertTrue(second["evaluation_weights_unchanged"])
                self.assertFalse(controller.candidates)
                selection = second["schedule"]["selections"][0]["audit"]
                self.assertEqual(selection["selected_count"], 8)
                self.assertEqual(selection["after_wake_updates"], 1)
                decision = second["schedule"]["decisions"][0]["measurement"]
                self.assertGreater(decision["mean_js"], 0.)
                self.assertGreater(decision["mean_new_class_invasion"], 0.)
                self.assertLessEqual(decision["mean_js"], np.log(2.))
                self.assertEqual(len(second["evaluation"]["variants"]), 3)
                self.assertEqual(second["evaluation"]["split"], "validation")
                output = Path(result["results_path"])
                saved = json.loads((output / "extensions.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["tasks"][1]["candidate_count"], 16)
                loaded = load_inference_model(output / "config.yaml")
                self.assertFalse(hasattr(loaded, "section10_controller"))
                np.testing.assert_allclose(loaded.network.get_weights()[0], wrapper.network.get_weights()[0])
                # Serialized episodic state must obey the original byte cap.
                if name == "gist_memory":
                    self.assertLessEqual((output / "memory.bin").stat().st_size, config.route.budget_bytes)



# Support direct execution and ordinary unittest discovery.
if __name__ == "__main__":
    unittest.main()
