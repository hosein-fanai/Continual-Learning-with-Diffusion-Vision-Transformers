"""Real parallel continual trials across the existing YAML/process/training boundary.

Only CIFAR10 loading is replaced by bounded synthetic pixels. Run this CPU
integration test in an authorized remote container, never on the laptop.
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

from common.config import load_config
from common.hpo import run_hpo


class DitContinualPipelineTests(TestCase):
    """Train two independent five-task V1 candidates in real isolated CPU workers."""

    def test_two_real_workers_complete_the_same_five_task_stream(self) -> None:
        """Validate dynamic class growth, per-task recovery and validation-only objectives."""

        profile = {
            "task_seed": 42, 
            "student_config": {
                "model": {
                    "name": "dit_classifier", "wrapper_name": "diffusion_classifier", 
                    "kwargs": {
                        "dim": 8, "depth": 1, "clf_depth": 1, "patch_size": 8, 
                        "mha_num_heads": 1, "clf_mha_num_heads": 1, 
                        "timesteps": 4, "use_cfg": True, "image_size": 32, "channels": 3, 
                        "classifier_only_cls_token": True, "clf_cls_token_type": "new_weight", 
                        "compile_args": {"run_eagerly": True, "jit_compile": False}
                    }, 
                    "wrapper_kwargs": {
                        "test_steps": 2, "scheduler_name": "clipped_cosine", 
                        "clf_loss_coef": 0.01, "clf_acc_coef": 1.0, 
                        "clf_distil_loss_coef": 0.0, "clf_distil_acc_coef": 0.0
                    }
                }
            }
        }
        overrides = {
            "classifier_teacher_source": ["none"], "noise_teacher_source": ["none"], 
            "continual_strategy": ["new_only"], "classifier_noise": ["clean"], 
            "batch_size": [4], "train_num": [-1], "optimizer": ["adam"], 
            "learning_rate": {"low": 0.0003, "high": 0.0003}, 
            "clipnorm": [1.0], "p_uncond": [0.1]
        }
        fixture = (
            "import sys,runpy,numpy as np\n"
            "from unittest.mock import patch\n"
            "rng=np.random.default_rng(91)\n"
            "data=((rng.integers(0,256,(100,32,32,3),dtype=np.uint8),(np.arange(100)%10).reshape(-1,1)),"
            "(rng.integers(0,256,(20,32,32,3),dtype=np.uint8),(np.arange(20)%10).reshape(-1,1)))\n"
            "sys.argv=['common.hpo_worker',*sys.argv[1:]]\n"
            "with patch('tensorflow.keras.datasets.cifar10.load_data',return_value=data):\n"
            " runpy.run_module('common.hpo_worker',run_name='__main__')\n"
        )
        original_popen = subprocess.Popen
        children = []


        def synthetic_worker(command: list[str], **options: object) -> subprocess.Popen:
            """Replace only the dataset shim while preserving the actual worker CLI."""

            child = original_popen([command[0], "-u", "-c", fixture, *command[4:]], **options)
            children.append(child)
            return child


        with TemporaryDirectory() as directory, patch.dict(os.environ, {
            "CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "3", 
            "OMP_NUM_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1"
        }), patch("common.hpo_process.subprocess.Popen", side_effect=synthetic_worker):
            study = run_hpo(
                task="continual", model_name="dit_classifier", dataset_name="cifar10", 
                n_trials=2, epochs=1, results_path=directory, concurrent_trials=2, 
                n_startup_trials=1, search_profile="dit_continual_runner", continual_profile=profile, 
                search_space_overrides=overrides, max_train_samples=20, max_val_samples=20, seed=17
            )
            self.assertEqual(len(children), 2)
            self.assertTrue(all(child.poll() == 0 for child in children))
            self.assertTrue(all(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials))
            self.assertEqual(len({child.pid for child in children}), 2)
            groups = None
            for trial in study.trials:
                config = load_config(trial.user_attrs["config_path"])
                # Every independent worker consumes the same explicit class stream.
                if groups is None:
                    groups = config.continually_learn.task_groups
                self.assertEqual(config.continually_learn.task_groups, groups)
                self.assertEqual(len(groups), 5)
                self.assertEqual(config.hpo["continual_dataset_seed"], 42)
                self.assertEqual(config.model.wrapper_kwargs["use_ema"], False)
                study_root = Path(config.hpo["study_root"])
                result = json.loads((study_root / "workers" / f"trial-{trial.number:04d}.json").read_text())
                self.assertEqual(result["status"], "complete")
                metrics = result["evaluations"]["validation_continual_metrics"]
                self.assertAlmostEqual(trial.value, metrics["final_average_accuracy"])
                self.assertTrue(Path(config.continually_learn.checkpoint_dir).is_dir())
