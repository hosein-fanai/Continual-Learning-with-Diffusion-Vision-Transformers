"""Continual profile transport, validation objectives and task-boundary recovery.

The profile builder and workers are bounded protocol doubles. Optuna storage,
YAML handoff, schedule sealing, objective extraction and recovery are real.
Run these tests only on an authorized remote container.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

from common.config import Config, load_config
import common.dit_continual_hpo as continual_recipe
from common.hpo import run_hpo
from common.tests.test_hpo_concurrency import _Workers


_GROUPS = [[8, 1], [5, 0], [7, 2], [9, 4], [3, 6]]


def _normalize_profile(profile: dict, dataset_name: str, seed: int) -> dict:
    """Seal one shared schedule without constructing students or teachers."""

    del dataset_name, seed
    return dict(deepcopy(profile), task_groups=deepcopy(_GROUPS))


def _build_profile(trial: object, **options: object) -> Config:
    """Construct a serializable continual recipe at the production profile boundary."""

    source = trial.suggest_categorical("classifier_teacher_source", ["previous", "specialist"])
    profile = options["continual_profile"]
    return Config(
        dataset={"name": options["dataset_name"]}, 
        model={"wrapper_name": "diffusion_classifier_v2", "name": "dit_classifier"}, 
        continually_learn={
            "class_num": 10, "task_size": 2, "task_groups": profile["task_groups"], 
            "class_order": [label for group in profile["task_groups"] for label in group], 
            "seed": profile["task_seed"], "save_task_checkpoints": True
        }, 
        hpo={
            "trial_number": trial.number, "params": dict(trial.params), 
            "use_ensemble_accuracy": options["use_ensemble_accuracy"], 
            "profile": deepcopy(profile), "classifier_teacher_source": source
        }, 
        training={
            "task": "continual", "epochs": options["epochs"], 
            "results_path": str(options["results_path"]), "seed": options["seed"]
        }
    )


def _continual_result(handle: SimpleNamespace) -> dict:
    """Publish authoritative continual validation and higher-valued test decoys."""

    result = _Workers.success(handle)
    result["evaluations"].update({
        "validation_continual_metrics": {
            "final_average_accuracy": 0.5 + handle.number / 1000, 
            "average_forgetting": 0.1 + handle.number / 1000
        }, 
        "continual_metrics": {"final_average_accuracy": 0.99}, 
        "test_continual_metrics": {"final_average_accuracy": 0.98}
    })
    return result


class DitContinualHpoDispatchTests(TestCase):
    """Verify isolated continual trials retain native objective and recovery semantics."""

    def setUp(self) -> None:
        """Allocate disposable storage and replace only the named recipe boundary."""

        temporary = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.profile = {
            "student_config": {"model": {"name": "dit_classifier"}}, 
            "specialist_teacher_descriptors": {}, "teacher_policy": "per_task", 
            "task_seed": 42
        }
        self.validate_search = Mock()
        context = patch.multiple(
            continual_recipe, 
            SEARCH_SPACE={"classifier_teacher_source": ["previous", "specialist"]}, 
            VERSION=1, normalize_continual_profile=_normalize_profile, 
            build_dit_continual_config=_build_profile, validate_continual_search=self.validate_search
        )
        context.start()
        self.addCleanup(context.stop)

    def options(self, **changes: object) -> dict:
        """Return a small new-profile public request with explicit isolated GPU routing."""

        options = {
            "task": "continual", "model_name": "dit_classifier", "dataset_name": "cifar10", 
            "search_profile": "dit_continual_runner", "continual_profile": deepcopy(self.profile), 
            "results_path": str(self.root), "n_trials": 3, "epochs": 2, "n_startup_trials": 1, 
            "trial_budget_mode": "total", "concurrent_trials": 2, "worker_gpu_ids": [0, 1], 
            "worker_gpu_memory_limit_mb": 8192, "seed": 17
        }
        options.update(changes)
        return options

    @staticmethod
    def study_root(study: object) -> Path:
        """Locate a study from a terminal trial's saved configuration."""

        return Path(load_config(study.trials[0].user_attrs["config_path"]).hpo["study_root"])

    def workers(self, **options: object) -> _Workers:
        """Provide normal continual results unless a test supplies an explicit outcome."""

        outcomes = {number: _continual_result for number in range(10)}
        outcomes.update(options.pop("outcomes", {}))
        return _Workers(outcomes=outcomes, **options)

    def test_parallel_trials_share_schedule_and_use_only_continual_validation(self) -> None:
        """Out-of-order completion keeps each score, seed, checkpoint and GPU paired."""

        workers = self.workers(delays={0: 5})
        admission = Mock()
        with workers.installed():
            study = run_hpo(**self.options(gpu_worker_context=admission))
        self.assertEqual([trial.value for trial in study.trials], [0.5, 0.501, 0.502])
        self.assertEqual([direction.name for direction in study.directions], ["MAXIMIZE"])
        self.assertEqual(workers.max_active, 2)
        self.assertEqual(study.user_attrs["study_spec"]["continual_schedule"]["task_groups"], _GROUPS)
        self.assertEqual(study.user_attrs["study_spec"]["continual_profile"]["task_groups"], _GROUPS)
        self.assertEqual(len({handle.config.continually_learn.checkpoint_dir for handle in workers.handles}), 3)
        for handle in workers.handles:
            self.assertEqual(handle.config.continually_learn.task_groups, _GROUPS)
            self.assertEqual(handle.config.continually_learn.seed, 42)
            self.assertEqual(handle.config.training.seed, 17)
            self.assertIs(handle.launch_options["gpu_worker_context"], admission)
            self.assertEqual(handle.launch_options["gpu_memory_limit_mb"], 8192)
            self.assertIsNone(handle.config.continually_learn.resume_from)
        self.assertTrue((self.study_root(study) / "pareto_trials.csv").is_file())

    def test_multiobjective_scores_and_direction_order_survive_worker_transport(self) -> None:
        """Custom continual aggregates retain their independent minimize/maximize axes."""

        workers = self.workers()
        with workers.installed():
            study = run_hpo(**self.options(
                n_trials=1, objective_metrics=["final_average_accuracy", "average_forgetting"]
            ))
        self.assertEqual(study.trials[0].values, [0.5, 0.1])
        self.assertEqual([direction.name for direction in study.directions], ["MAXIMIZE", "MINIMIZE"])
        self.assertEqual(study.trials[0].user_attrs["validation_metrics"]["average_forgetting"], 0.1)

    def test_missing_validation_never_falls_back_to_test_metrics(self) -> None:
        """A worker's complete envelope is insufficient without the declared validation report."""

        workers = self.workers(outcomes={0: _Workers.success})
        with workers.installed(), self.assertRaisesRegex(KeyError, "validation_continual_metrics"):
            run_hpo(**self.options(n_trials=1))

    def test_oom_and_nonfinite_objective_prune_without_abandoning_other_trials(self) -> None:
        """Resource and numerical failures preserve evidence while later candidates finish."""

        def nonfinite(handle: SimpleNamespace) -> dict:
            """Model a nonfinite aggregate produced only at final task-matrix scoring."""

            result = _continual_result(handle)
            result["evaluations"]["validation_continual_metrics"]["final_average_accuracy"] = float("nan")
            return result

        workers = self.workers(outcomes={
            0: {"status": "oom", "error": "MemoryError: test", "results_path": str(self.root / "partial")}, 
            1: nonfinite
        })
        with workers.installed():
            study = run_hpo(**self.options())
        self.assertEqual([trial.state.name for trial in study.trials], ["PRUNED", "PRUNED", "COMPLETE"])
        self.assertEqual(study.trials[1].user_attrs["divergence"]["reason"], "nonfinite_objective")
        self.assertEqual(study.trials[2].value, 0.502)

    def test_interrupted_trial_recovers_parameters_and_committed_task_checkpoint(self) -> None:
        """Resume reuses the original task-checkpoint root and sampled recipe exactly once."""

        workers = self.workers(outcomes={0: KeyboardInterrupt()})
        with workers.installed(), self.assertRaises(KeyboardInterrupt):
            run_hpo(**self.options(n_trials=1))
        original = workers.handles[0].config
        original_checkpoint = original.continually_learn.checkpoint_dir
        root = Path(original.hpo["study_root"])
        retry_workers = self.workers()
        with retry_workers.installed(), patch("common.hpo._has_committed_task_checkpoint", return_value=True):
            resumed = run_hpo(**self.options(n_trials=2, resume_from=root))
            repeated = run_hpo(**self.options(n_trials=2, resume_from=root))
        self.assertEqual(len(retry_workers.handles), 1)
        self.assertEqual(len(repeated.trials), 2)
        retry = retry_workers.handles[0].config
        self.assertEqual(retry.continually_learn.resume_from, original_checkpoint)
        self.assertEqual(retry.continually_learn.checkpoint_dir, original_checkpoint)
        self.assertEqual(resumed.trials[1].params, resumed.trials[0].params)
        self.assertEqual(resumed.trials[1].user_attrs["resume_original_trial_number"], 0)

    def test_changed_student_or_schedule_seed_cannot_resume_existing_study(self) -> None:
        """Scientific identity rejects changed inputs before loading or writing study storage."""

        workers = self.workers()
        with workers.installed():
            study = run_hpo(**self.options(n_trials=1))
        for field, replacement in (("student_config", {"model": {"name": "different"}}), ("task_seed", 9)):
            profile = dict(self.profile, **{field: replacement})
            with self.subTest(field=field), patch("optuna.load_study") as load:
                with self.assertRaisesRegex(ValueError, "specification differs"):
                    run_hpo(**self.options(continual_profile=profile, resume_from=self.study_root(study)))
                load.assert_not_called()

    def test_runtime_teachers_and_conflicting_protocol_controls_fail_before_storage(self) -> None:
        """Only the profile's serialized student, teachers and schedule can reach workers."""

        for changes in (
            {"teacher_network": object()}, {"use_distillation": True}, 
            {"model_overrides": {"dim": 32}}, {"task_groups": _GROUPS}, 
            {"validation_source": "test"}, {"pruning": {}}, 
            {"fit_kwargs": {"validation_freq": 1}}
        ):
            with self.subTest(changes=changes), patch("optuna.create_study") as create:
                with self.assertRaises(ValueError):
                    run_hpo(**self.options(**changes))
                create.assert_not_called()

    def test_missing_requested_teacher_input_fails_before_creating_study(self) -> None:
        """Preflight validates the entire source grid before any allocation or worker launch."""

        self.validate_search.side_effect = ValueError("Supply the noise specialist descriptor")
        with patch("optuna.create_study") as create, patch("common.hpo.start_worker") as start:
            with self.assertRaisesRegex(ValueError, "noise specialist"):
                run_hpo(**self.options())
        self.validate_search.assert_called_once()
        create.assert_not_called()
        start.assert_not_called()


# Permit direct execution as well as unittest discovery.
if __name__ == "__main__":
    main()
