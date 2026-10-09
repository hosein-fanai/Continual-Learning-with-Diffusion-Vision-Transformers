"""Classifier runner dispatch, Pareto objectives, and isolated failure handling.

Training and worker execution are replaced by existing protocol doubles. Real
Optuna studies, resolved YAML, resume seals, and objective extraction remain in
scope. Run this module only in an authorized remote environment.
"""

from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

from common.callbacks.hpo_guard import NonFiniteLossGuard
from common.config import load_config
from common.hpo import run_hpo, summarize_hpo
from common.hpo_worker import run_worker
from common.tests.test_dit_hpo_api import _PruningWorkers
from common.tests.test_hpo_concurrency import _Workers, _training_result


def _nonfinite_classifier_result(handle: SimpleNamespace) -> dict:
    """Retain finite denoising and EMA decoys while invalidating raw accuracy."""

    result = _Workers.success(handle)
    result["evaluations"]["valset_network_eval"]["classifier_accuracy"] = float("nan")
    return result


def _nonfinite_noise_result(handle: SimpleNamespace) -> dict:
    """Retain finite accuracy and EMA decoys while invalidating raw noise loss."""

    result = _Workers.success(handle)
    result["evaluations"]["valset_network_eval"]["noise_loss"] = float("inf")
    return result


def _nonfinite_loss_worker(handle: SimpleNamespace) -> dict:
    """Use the real worker envelope around an ordinary joint loss-guard failure."""

    run_directory = Path(handle.config.training.results_path) / f"trial-{handle.number:04d}"
    handle.config.training.results_path = str(run_directory)
    tensorflow = Mock()
    tensorflow.errors.ResourceExhaustedError = MemoryError
    tensorflow.config.list_physical_devices.return_value = []

    def train(config: object) -> dict:
        """Feed the real numerical guard without constructing a model or dataset."""

        callback = NonFiniteLossGuard(phase="joint", evidence_dir=config.training.results_path)
        callback.on_train_begin()
        callback.on_epoch_end(0, {"loss": 0.1, "classifier_accuracy": 0.5})
        callback.on_epoch_end(1, {"loss": float("nan"), "classifier_accuracy": 0.6})
        raise AssertionError("A nonfinite loss must stop before publishing completed training.")

    modules = {
        "tensorflow": tensorflow, 
        "common.config": SimpleNamespace(load_config=Mock(return_value=handle.config), save_config=Mock()), 
        "common.train": SimpleNamespace(main=train)
    }
    with patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=True), \
            redirect_stderr(io.StringIO()):
        run_worker(run_directory / "input.yaml", handle.output_path)
    return json.loads(handle.output_path.read_text(encoding="utf-8"))


class DitClassifierHpoDispatchTests(TestCase):
    """Protect the new joint recipe without changing historical study semantics."""

    def setUp(self) -> None:
        """Allocate private disposable study storage without real model work."""

        temporary = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def options(self, **changes: object) -> dict:
        """Return the new notebook's public API protocol with a tiny trial budget."""

        options = {
            "task": "joint", "model_name": "dit_classifier", 
            "dataset_name": "cifar10", "search_profile": "dit_classifier_runner", 
            "results_path": str(self.root), "n_trials": 1, "epochs": 3, 
            "n_startup_trials": 1, "trial_budget_mode": "total", 
            "validation_source": "test", "validation_ratio": 0.0, 
            "concurrent_trials": 1, "seed": 17
        }
        options.update(changes)
        return options

    @staticmethod
    def study_root(study: object) -> Path:
        """Locate a real study directory through its persisted trial handoff."""

        config = load_config(study.trials[0].user_attrs["config_path"])
        return Path(config.hpo["study_root"])

    def test_default_dispatch_uses_final_raw_pareto_objectives_on_both_cifar_datasets(self) -> None:
        """Both final objectives ignore history, EMA, and separate-test score decoys."""

        for dataset in ("cifar10", "cifar100"):
            with self.subTest(dataset=dataset), patch("common.hpo.main", side_effect=_training_result) as train:
                study = run_hpo(**self.options(dataset_name=dataset, results_path=str(self.root / dataset)))
            self.assertEqual(train.call_count, 1)
            self.assertEqual(study.trials[0].values, [0.4, 0.2])
            self.assertEqual([direction.name for direction in study.directions], ["MAXIMIZE", "MINIMIZE"])
            spec = study.user_attrs["study_spec"]
            self.assertEqual(spec["search_profile"], "dit_classifier_runner")
            self.assertEqual(spec["profile_version"], 2)
            self.assertEqual(spec["objective_metrics"], ["classification_accuracy", "noise_loss"])
            self.assertEqual(spec["objective_directions"], ["maximize", "minimize"])
            config = load_config(study.trials[0].user_attrs["config_path"])
            self.assertEqual(config.hpo["objective_network"], "raw")
            self.assertEqual(config.hpo["objectives"], [0.4, 0.2])
            self.assertEqual(config.hpo["checkpoint_selection_policy"], "final_epoch")
            self.assertIsNone(config.hpo["checkpoint_selection_metric"])
            self.assertEqual(config.training.patience, 0)
            self.assertEqual(config.training.epochs, 3)
            self.assertEqual(config.training.fit_kwargs, {"validation_freq": 1})
            self.assertEqual(config.model.wrapper_name, "diffusion_classifier")
            self.assertFalse(config.model.wrapper_kwargs["use_ema"])
            self.assertEqual(config.model.wrapper_kwargs["test_network_name"], "raw")
            self.assertEqual(config.model.wrapper_kwargs["clf_test_noisified_max_timesteps"], 0)
            self.assertEqual(config.dataset.validation_source, "test")
            self.assertEqual(config.dataset.validation_ratio, 0.0)
            self.assertFalse(config.dataset.drop_remainder)
            self.assertAlmostEqual(study.trials[0].user_attrs["validation_metrics"]["noise_loss"], 0.2)
            self.assertTrue(config.training.tensorboard)
            self.assertEqual(config.training.tensorboard_path, str(self.study_root(study) / "tensorboard"))
            self.assertTrue((self.study_root(study) / "pareto_trials.csv").is_file())

    def test_legacy_joint_pareto_and_generation_ema_objectives_remain_unchanged(self) -> None:
        """The archive-informed recipe has a separate identity from established searches."""

        with patch("common.hpo.main", side_effect=_training_result):
            classifier = run_hpo(**self.options(results_path=str(self.root / "classifier")))
            legacy = run_hpo(**self.options(
                search_profile="joint_dit_classifier", results_path=str(self.root / "legacy")
            ))
            generation = run_hpo(**self.options(
                task="generation", model_name="diffusion_transformer", search_profile=None, 
                objective_metrics=["generation_loss"], objective_directions=["minimize"], 
                results_path=str(self.root / "generation")
            ))
        self.assertEqual(classifier.trials[0].values, [0.4, 0.2])
        self.assertEqual(legacy.trials[0].values, [0.4, 0.2])
        self.assertEqual([direction.name for direction in legacy.directions], ["MAXIMIZE", "MINIMIZE"])
        self.assertTrue((self.study_root(legacy) / "pareto_trials.csv").is_file())
        self.assertAlmostEqual(generation.trials[0].value, 0.01)
        self.assertEqual([direction.name for direction in generation.directions], ["MINIMIZE"])
        self.assertNotIn("search_profile", generation.user_attrs["study_spec"])
        self.assertNotEqual(
            classifier.user_attrs["study_spec"]["profile_specification"], 
            legacy.user_attrs["study_spec"]["profile_specification"]
        )
        self.assertEqual(len({classifier.study_name, legacy.study_name, generation.study_name}), 3)

    def test_existing_joint_study_cannot_resume_under_the_new_search_profile(self) -> None:
        """Matching objective names cannot bypass incompatible recipe and space identities."""

        with patch("common.hpo.main", side_effect=_training_result):
            legacy = run_hpo(**self.options(search_profile="joint_dit_classifier"))
        with patch("optuna.load_study") as load, patch("common.hpo.main") as train:
            with self.assertRaisesRegex(ValueError, "specification differs"):
                run_hpo(**self.options(n_trials=2, resume_from=self.study_root(legacy)))
        load.assert_not_called()
        train.assert_not_called()
        self.assertEqual(legacy.trials[0].values, [0.4, 0.2])

    def test_single_to_three_gpu_resume_preserves_identity_and_both_scores(self) -> None:
        """Routing is operational and never invokes unsupported scalar Optuna pruning."""

        workers = _Workers(delays={1: 6})
        with workers.installed(), \
                patch("optuna.trial.Trial.report", side_effect=AssertionError("No scalar report in a Pareto study.")) as report, \
                patch("optuna.trial.Trial.should_prune", side_effect=AssertionError("No scalar pruning in a Pareto study.")) as should_prune:
            first = run_hpo(**self.options(worker_gpu_ids=[0]))
            root = self.study_root(first)
            resumed = run_hpo(**self.options(
                n_trials=4, concurrent_trials=3, worker_gpu_ids=[0, 1, 2], 
                worker_gpu_memory_limit_mb=16384, resume_from=root
            ))
        report.assert_not_called()
        should_prune.assert_not_called()
        self.assertEqual(resumed.user_attrs["study_spec"], first.user_attrs["study_spec"])
        self.assertEqual(len(resumed.trials), 4)
        self.assertEqual(workers.max_active, 3)
        self.assertEqual(
            [handle.launch_options["gpu_id"] for handle in workers.handles], 
            ["0", "0", "1", "2"]
        )
        for trial in resumed.trials:
            self.assertEqual(trial.values, [0.4 + trial.number / 1000, 0.2 + trial.number / 2000])
            self.assertEqual(trial.intermediate_values, {})
            config = load_config(trial.user_attrs["config_path"])
            self.assertEqual(config.hpo["objective_network"], "raw")
            self.assertEqual(config.training.tensorboard_path, str(root / "tensorboard"))
        for handle in workers.handles:
            self.assertIsNone(handle.launch_options["pruning_monitor"])

    def test_multiobjective_performance_pruning_is_rejected_before_storage(self) -> None:
        """No monitor choice can turn the paired objective into scalar percentile pruning."""

        for index, policy in enumerate([
            {}, {"monitor": "val_classifier_accuracy"}, {"monitor": "val_noise_loss"}, 
            {"type": "percentile", "percentile": 75.0, "n_startup_trials": 1}
        ]):
            destination = self.root / str(index)
            with self.subTest(policy=policy), patch("optuna.create_study") as create:
                with self.assertRaises(ValueError):
                    run_hpo(**self.options(pruning=policy, results_path=str(destination)))
                create.assert_not_called()
                self.assertFalse(destination.exists())

    def test_incompatible_profile_protocols_fail_before_storage(self) -> None:
        """Teacher, ensemble, fitting and objective changes cannot alter the named recipe."""

        cases = [
            {"objective_metrics": ["classification_accuracy"]}, 
            {"objective_metrics": ["noise_loss"], "objective_directions": ["minimize"]}, 
            {"objective_directions": ["minimize", "maximize"]}, 
            {"objective_metrics": ["noise_loss", "classification_accuracy"], "objective_directions": ["minimize", "maximize"]}, 
            {"fit_kwargs": {"validation_freq": 2}}, {"fit_kwargs": {"callbacks": []}}, 
            {"teacher_network": object()}, {"use_distillation": True}, 
            {"use_ensemble_accuracy": True}, {"ensemble_accuracy_kwargs": {"max_t": 128}}
        ]
        for index, changes in enumerate(cases):
            options = self.options(results_path=str(self.root / str(index)))
            options.update(changes)
            with self.subTest(changes=changes), patch("optuna.create_study") as create:
                with self.assertRaises(ValueError):
                    run_hpo(**options)
                create.assert_not_called()
                self.assertFalse(Path(options["results_path"]).exists())

    def test_conflicting_raw_evaluation_overrides_fail_before_worker_launch(self) -> None:
        """The profile builder owns conflicting fixed options during trial preparation."""

        for index, overrides in enumerate([
            {"test_network_name": "ema"}, {"use_ema": True}, 
            {"clf_test_noisified_max_timesteps": 128}
        ]):
            workers = _Workers()
            with self.subTest(overrides=overrides), workers.installed():
                with self.assertRaisesRegex(ValueError, "replaces profile setting"):
                    run_hpo(**self.options(
                        concurrent_trials=2, wrapper_overrides=overrides, 
                        results_path=str(self.root / f"override-{index}")
                    ))
            self.assertEqual(workers.handles, [])

    def test_generation_pruning_retains_ema_noise_monitor(self) -> None:
        """The paired classifier profile leaves ordinary generation's scalar pruning intact."""

        workers = _PruningWorkers({0: [0.7, 0.6]})
        with workers.installed():
            study = run_hpo(**self.options(
                task="generation", model_name="diffusion_transformer", search_profile=None, 
                objective_metrics=["generation_loss"], objective_directions=["minimize"], pruning={}
            ))
        self.assertAlmostEqual(study.trials[0].value, 0.01)
        self.assertEqual(workers.handles[0].launch_options["pruning_monitor"], "val_noise_loss")
        self.assertEqual(study.user_attrs["study_spec"]["pruning"]["monitor"], "val_noise_loss")
        self.assertEqual(study.trials[0].intermediate_values, {0: 0.7, 1: 0.6})

    def test_nonfinite_joint_loss_survives_worker_envelope_as_pruned(self) -> None:
        """Numerical training failure prunes its allocation without scalar Optuna reports."""

        workers = _Workers(outcomes={1: _nonfinite_loss_worker})
        with workers.installed():
            study = run_hpo(**self.options(n_trials=3, concurrent_trials=3))
        self.assertEqual([trial.state.name for trial in study.trials], ["COMPLETE", "PRUNED", "COMPLETE"])
        evidence = study.trials[1].user_attrs["divergence"]
        self.assertEqual(evidence["reason"], "nonfinite_loss")
        self.assertEqual(evidence["metric"], "loss")
        self.assertEqual(evidence["phase"], "joint")
        self.assertEqual(evidence["hook"], "epoch_end")
        self.assertEqual(evidence["epoch"], 1)
        self.assertEqual(evidence["value"], "nan")
        self.assertEqual([row["epoch"] for row in evidence["partial_history"]], [0])
        path = Path(study.trials[1].user_attrs["divergence_path"])
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), evidence)
        self.assertEqual(study.trials[2].values, [0.402, 0.201])
        self.assertFalse(workers.active)

    def test_oom_or_either_nonfinite_objective_prunes_without_finite_decoy_fallback(self) -> None:
        """Accuracy and denoising must both be finite before a trial can enter the frontier."""

        workers = _Workers(outcomes={
            1: {"status": "oom", "error": "controlled classifier memory exhaustion"}, 
            2: _nonfinite_classifier_result, 3: _nonfinite_noise_result
        })
        with workers.installed():
            study = run_hpo(**self.options(n_trials=4, concurrent_trials=3, worker_gpu_ids=[0, 1, 2]))
        self.assertEqual([trial.state.name for trial in study.trials], ["COMPLETE", "PRUNED", "PRUNED", "PRUNED"])
        self.assertEqual(study.trials[1].user_attrs["oom"]["reason"], "out_of_memory")
        for number in (2, 3):
            self.assertEqual(study.trials[number].user_attrs["divergence"]["reason"], "nonfinite_objective")
            self.assertEqual(study.trials[number].user_attrs["divergence"]["network"], "raw")
        self.assertEqual(study.trials[2].user_attrs["divergence"]["objectives"]["classification_accuracy"], "nan")
        self.assertEqual(study.trials[3].user_attrs["divergence"]["objectives"]["noise_loss"], "inf")
        table = summarize_hpo(study, pareto_only=False).set_index("trial")
        self.assertAlmostEqual(table.loc[0, "classification_accuracy"], 0.4)
        self.assertAlmostEqual(table.loc[0, "noise_loss"], 0.2)
        self.assertTrue(table.loc[[1, 2, 3], ["classification_accuracy", "noise_loss"]].isna().all().all())
        self.assertEqual([trial.number for trial in study.best_trials], [0])
        self.assertFalse(workers.active)


# This focused entry point never starts real model training.
if __name__ == "__main__":
    main()