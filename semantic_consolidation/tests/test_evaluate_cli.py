"""Saved-checkpoint CLI, original split reconstruction and confirmation guards."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from common.config import load_config, save_config
from common.study_artifacts import native_study_metadata
from common.experiment import create_paired_block_manifest, materialize_run_plan, write_experiment_manifest
from common.model import get_model
from semantic_consolidation.config import load_route_config
from semantic_consolidation.evaluate import _load_arrays, evaluate_saved_checkpoint, main
from semantic_consolidation.evaluation import EnsembleEvaluationSettings
from semantic_consolidation.study import planned_config
from semantic_consolidation.tests.test_evaluation import _Wrapper


_ROOT = Path(__file__).resolve().parents[2]


class SavedCheckpointTests(unittest.TestCase):
    """Exercise real preprocessing and checkpoint reload with bounded image fixtures."""

    @staticmethod
    def _pixels() -> tuple:
        """Give original labels seven and four distinct train/test pixel ranges."""

        labels = np.repeat(np.asarray([7, 4, 2, 0], dtype="uint8"), 24)
        train = np.arange(len(labels), dtype="uint8")[:, None, None] + np.zeros((len(labels), 28, 28), "uint8")
        test_labels = np.repeat(np.asarray([7, 4, 2, 0], dtype="uint8"), 4)
        test = np.full((len(test_labels), 28, 28), 240, dtype="uint8")
        return (train, labels), (test, test_labels)

    @staticmethod
    def _template() -> object:
        """Resolve a valid tiny two-class route with explicit evaluation settings."""

        route = load_route_config(_ROOT / "semantic_consolidation/configs/smoke.yaml")
        route.common.continually_learn.class_num = 4
        route.common.continually_learn.class_order = [7, 4, 2, 0]
        route.common.continually_learn.task_groups = [[7, 4], [2, 0]]
        route.common.dataset.max_val_samples = 16
        route.common.continually_learn.seed = 17
        route.common.training.seed = 17
        route.route.seed = 17
        route.route.extensions = {"evaluation": asdict(EnsembleEvaluationSettings(
            enabled=True, horizons=(1, 2), batch_size=4, calibration_fraction=0.5,
        ))}
        return route

    def _saved(self, directory: str, confirmation: bool = False) -> Path:
        """Write a saved common config and optional genuine frozen run manifest."""

        root = Path(directory)
        route = self._template()
        # A real common manifest binds every inference and data setting for test.
        if confirmation:
            manifest = create_paired_block_manifest(
                {"a": {}, "b": {}},
                [{"class_order": [7, 4, 2, 0], "task_groups": [[7, 4], [2, 0]], "stream_seed": 17},
                 {"class_order": [4, 7, 0, 2], "task_groups": [[4, 7], [0, 2]], "stream_seed": 29}],
                seed=5, phase="confirmation",
                base_config={"common": asdict(route.common), "route": asdict(route.route)},
                analysis_spec={"native_route_study": native_study_metadata("semantic_consolidation")},
            )
            manifest_path = root / "manifest.json"
            write_experiment_manifest(manifest_path, manifest)
            entry = next(item for item in materialize_run_plan(manifest) if item["stream"]["stream_seed"] == 17)
            route = planned_config(entry, manifest_path)
        route.common.hpo["semantic_consolidation"] = asdict(route.route)
        route.common.model.weights_path = "fixture-checkpoint.weights"
        path = root / "config.yaml"
        save_config(route.common, path)
        return path

    @staticmethod
    def _wrapper() -> _Wrapper:
        """Return a counted inference fixture using introduction IDs zero and one."""

        wrapper = _Wrapper(input_dependent=True)
        wrapper.seen_classes = {0: 0, 1: 1}
        return wrapper

    def test_validation_uses_original_mapping_and_writes_report(self) -> None:
        """Use real shared loading/splitting and prevent overwriting earlier metrics."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._saved(directory)
            wrapper = self._wrapper()
            with patch("tensorflow.keras.datasets.mnist.load_data", side_effect=self._pixels), patch(
                "common.model.get_model", return_value={"generative_model": wrapper},
            ) as factory:
                result = evaluate_saved_checkpoint(path)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(result["original_class_to_classifier_id"], {"7": 0, "4": 1})
            self.assertTrue(result["checkpoint_weights_unchanged"])
            self.assertEqual(result["evaluation_class_counts"], {"0": 3, "1": 2})
            self.assertEqual(len(result["variants"]), 3)
            self.assertEqual(json.loads(Path(result["output_path"]).read_text()), result)
            with self.assertRaises(FileExistsError):
                evaluate_saved_checkpoint(path)

    def test_confirmation_uses_validation_calibration_and_locked_settings(self) -> None:
        """Fit calibration on low-valued training-source pixels and score test pixels."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._saved(directory, confirmation=True)
            with patch("tensorflow.keras.datasets.mnist.load_data", side_effect=self._pixels), patch(
                "common.model.get_model", return_value={"generative_model": self._wrapper()},
            ):
                result = evaluate_saved_checkpoint(path, split="test")
            self.assertEqual(result["confirmation"]["phase"], "confirmation")
            self.assertEqual(result["evaluation_class_counts"], {"0": 4, "1": 4})
            self.assertEqual(result["calibration_source"], "caller-supplied held-out validation")
            self.assertEqual(result["variants"][0]["temperature_fit"]["sample_count"], 4)
            overrides = Path(directory) / "changed.yaml"
            overrides.write_text(yaml.safe_dump(asdict(EnsembleEvaluationSettings(enabled=True, horizons=(3,)))), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen manifest"):
                evaluate_saved_checkpoint(path, split="test", settings_path=overrides,
                                          output_path=Path(directory) / "changed.json")

    def test_test_access_rejects_development_and_tampered_data_before_loading(self) -> None:
        """Keep forbidden test evaluation unreachable before dataset/model loading."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._saved(directory)
            with patch("common.model.get_model") as factory, self.assertRaisesRegex(ValueError, "confirmation"):
                evaluate_saved_checkpoint(path, split="test")
            factory.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            path = self._saved(directory, confirmation=True)
            config = load_config(path)
            config.dataset.validation_ratio = 0.4
            save_config(config, path)
            with patch("common.model.get_model") as factory, self.assertRaisesRegex(ValueError, "data split"):
                evaluate_saved_checkpoint(path, split="test")
            factory.assert_not_called()

    def test_controlled_source_reuses_its_loader_without_cifar(self) -> None:
        """Regenerate the controlled source using its disclosed independent seeds."""

        from gist_memory.config import load_route_config as load_gist_config

        route = load_gist_config(_ROOT / "gist_memory/configs/controlled.yaml")
        route.route.seed = route.common.continually_learn.seed
        # Standard loader calls would incorrectly access CIFAR images for this source.
        with patch("common.dataloader.get_datasets", side_effect=AssertionError("CIFAR must not be loaded")):
            arrays = _load_arrays(route.common, "gist_memory", asdict(route.route))
        self.assertEqual(arrays[0].shape[1:], (32, 32, 3))
        self.assertTrue(set(np.asarray(arrays[3]).reshape(-1)).issubset({0, 1, 2, 3}))
        self.assertGreater(len(arrays[2]), 0)
        self.assertGreater(len(arrays[4]), 0)

    def test_real_checkpoint_reload_never_calls_fit(self) -> None:
        """Round-trip actual tiny DiffusionClassifier weights through the common factory."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._saved(directory)
            config = load_config(path)
            config.model.weights_path = None
            config.model.kwargs["num_classes"] = 4
            config.model.wrapper_kwargs["seen_classes"] = {0: 0, 1: 1, 2: 2, 3: 3}
            config.dataset.trainset_len = 4
            model = get_model(config)["generative_model"]
            checkpoint = str(Path(directory) / "model.weights.h5")
            model.save_weights(checkpoint)
            config.model.weights_path = checkpoint
            save_config(config, path)
            with patch("tensorflow.keras.datasets.mnist.load_data", side_effect=self._pixels), patch(
                "diffusion.models.wrapper.diffusion_classifier.DiffusionClassifier.fit",
                side_effect=AssertionError("Checkpoint evaluation must never train"),
            ):
                result = evaluate_saved_checkpoint(path)
            self.assertTrue(result["checkpoint_weights_unchanged"])
            self.assertEqual(result["seen_class_count"], 4)
            self.assertTrue(all(np.isfinite(record["metrics"]["nll"]) for record in result["variants"]))

    def test_main_accepts_saved_route_settings_yaml(self) -> None:
        """Resolve route.settings.yaml and expose the user-facing module entry point."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._saved(directory)
            settings_path = Path(directory) / "route.settings.yaml"
            settings_path.write_text(yaml.safe_dump(asdict(self._template().route)), encoding="utf-8")
            output = Path(directory) / "cli.json"
            with patch("tensorflow.keras.datasets.mnist.load_data", side_effect=self._pixels), patch(
                "common.model.get_model", return_value={"generative_model": self._wrapper()},
            ):
                main(["--config", str(path), "--settings", str(settings_path), "--output", str(output)])
            self.assertTrue(output.is_file())


# Execute the focused tests only when the file is invoked directly.
if __name__ == "__main__":
    unittest.main()
