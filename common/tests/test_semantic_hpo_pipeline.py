"""Real isolated semantic HPO trials with tiny synthetic CIFAR images.

Run only in an authorized remote container. CPU workers exercise the full model,
joint fit, replay, acquisition, consolidation and held-out objective pipeline.
Synthetic scores validate plumbing and do not estimate scientific performance.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import optuna
import numpy as np

from common.config import load_config
from common.dit_hpo_runner import _digest
from common.hpo import run_hpo
from common.semantic_hpo import SEARCH_SPACE
from common.semantic_hpo_runner import run_confirmation
from semantic_consolidation.config import load_route_config


class SemanticPipelineTests(TestCase):
    """Train separate semantic candidates through the real saved-YAML worker CLI."""

    def test_real_workers_run_both_semantic_phases_on_every_task(self) -> None:
        """Keep native replay/KD and collect actual route records for both tasks."""

        root = Path(__file__).resolve().parents[2]
        native = load_route_config(root / "semantic_consolidation/configs/smoke.yaml")
        native.common.dataset.name = "cifar10"
        native.common.dataset.max_train_samples = 40
        native.common.dataset.max_val_samples = 20
        native.common.model.kwargs.update({"timesteps": 4, "patch_size": 8})
        native.common.model.wrapper_kwargs["test_steps"] = 2
        native.common.model.show_network_summary = False
        native.common.continually_learn.replay_current_examples = 8
        native.common.continually_learn.replay_old_examples = 8
        native.common.continually_learn.show_generated_images = False
        native.common.continually_learn.mechanistic_metrics = False
        native.route.acquisition_steps = 1
        native.route.consolidation_steps = 1
        native.route.noise_levels = tuple([0])
        native.route.probe_batches = 1
        native.route.probe_max_gates = 2
        overrides = {
            "acquisition_steps": [1], "consolidation_steps": [1], "batch_size": [4], 
            "acquisition_noise_level": [0], "ce_noise_level": [0], 
            "noise_levels": ["0"], "image_augmentation": ["none"]
        }
        for name, distribution in SEARCH_SPACE.items():
            # Pin continuous controls only for this bounded plumbing check.
            if isinstance(distribution, dict):
                value = getattr(native.route, name)
                overrides[name] = {"low": value, "high": value, "log": distribution.get("log", False)}
        fixture = (
            "import sys,runpy,numpy as np\n"
            "from unittest.mock import patch\n"
            "from common.semantic_hpo import SEARCH_SPACE\n"
            "rng=np.random.default_rng(91)\n"
            "data=((rng.integers(0,256,(160,32,32,3),dtype=np.uint8),(np.arange(160)%4).reshape(-1,1)),"
            "(rng.integers(0,256,(40,32,32,3),dtype=np.uint8),(np.arange(40)%4).reshape(-1,1)))\n"
            "sys.argv=['common.hpo_worker',*sys.argv[1:]]\n"
            "with patch('tensorflow.keras.datasets.cifar10.load_data',return_value=data),"
            "patch.dict(SEARCH_SPACE,{'acquisition_steps':[1],'consolidation_steps':[1],'batch_size':[4]}):\n"
            " runpy.run_module('common.hpo_worker',run_name='__main__')\n"
        )
        original_popen = subprocess.Popen
        children = []


        def synthetic_worker(command: list[str], **options: object) -> subprocess.Popen:
            """Replace downloaded images while retaining the actual worker process."""

            child = original_popen([command[0], "-u", "-c", fixture, *command[4:]], **options)
            children.append(child)
            return child


        with TemporaryDirectory() as directory, patch.dict(os.environ, {
            "CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "3", 
            "OMP_NUM_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1"
        }), patch.dict(SEARCH_SPACE, {
            "acquisition_steps": [1], "consolidation_steps": [1], "batch_size": [4]
        }), patch("common.hpo_process.subprocess.Popen", side_effect=synthetic_worker):
            study = run_hpo(
                task="continual", model_name="dit_classifier", dataset_name="cifar10", 
                n_trials=2, epochs=1, results_path=directory, concurrent_trials=2, 
                n_startup_trials=1, deterministic_ops=True, search_profile="semantic_consolidation_runner", 
                semantic_profile={"student_config": native.common, "route_settings": native.route}, 
                search_space_overrides=overrides, seed=17
            )
            self.assertEqual(len(children), 2)
            self.assertTrue(all(child.poll() == 0 for child in children))
            self.assertTrue(all(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials))
            self.assertEqual(len({child.pid for child in children}), 2)
            destinations = set()
            first_source = None
            for trial in study.trials:
                config = load_config(trial.user_attrs["config_path"])
                self.assertEqual(config.continually_learn.task_groups, [[2, 0], [3, 1]])
                self.assertEqual(config.hpo["continual_dataset_seed"], 41)
                self.assertEqual(config.hpo["semantic_consolidation"]["condition"], "learned")
                study_root = Path(config.hpo["study_root"])
                # Finalists replay the immutable input, never a trained checkpoint config.
                if first_source is None:
                    first_source = study_root / "configs" / f"trial-{trial.number:04d}.yaml"
                receipt = json.loads((study_root / "workers" / f"trial-{trial.number:04d}.json").read_text())
                self.assertEqual(receipt["status"], "complete")
                output = Path(receipt["results_path"])
                destinations.add(output)
                records = json.loads((output / "route_metrics.json").read_text())
                self.assertEqual(len(records), 2)
                for record in records:
                    self.assertEqual(record["acquisition"]["updates"], 1)
                    self.assertEqual(record["consolidation"]["updates"], 1)
                    self.assertTrue(all(record["invariants"].values()))
                metrics = receipt["evaluations"]["validation_continual_metrics"]
                self.assertAlmostEqual(trial.value, metrics["final_average_accuracy"])
                self.assertTrue((output / "modulators.npz").is_file())
                self.assertTrue(Path(config.continually_learn.checkpoint_dir).is_dir())
            self.assertEqual(len(destinations), 2)
            rng = np.random.default_rng(91)
            pixels = (
                (rng.integers(0, 256, (160, 32, 32, 3), dtype=np.uint8), (np.arange(160) % 4).reshape(-1, 1)), 
                (rng.integers(0, 256, (40, 32, 32, 3), dtype=np.uint8), (np.arange(40) % 4).reshape(-1, 1))
            )
            with patch("tensorflow.keras.datasets.cifar10.load_data", return_value=pixels):
                repeated = run_confirmation(first_source, Path(directory) / "confirmation", 101, _digest(first_source))
            self.assertEqual(repeated["training_seed"], 101)
            self.assertEqual(repeated["dataset_seed"], 41)
            self.assertEqual(repeated["task_groups"], [[2, 0], [3, 1]])
            repeated_records = json.loads((Path(repeated["results_path"]) / "route_metrics.json").read_text())
            self.assertEqual(len(repeated_records), 2)
            self.assertTrue(all(record["acquisition"]["updates"] == 1 for record in repeated_records))
            self.assertTrue(all(record["consolidation"]["updates"] == 1 for record in repeated_records))
