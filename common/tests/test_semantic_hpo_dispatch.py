"""Real Optuna identity/objective handling around bounded semantic worker doubles.

Run only in an authorized remote container. These tests exercise orchestration,
not the numerical correctness of semantic optimization.
"""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from common.config import Config, load_config
from common.hpo import run_hpo
from common.tests.test_hpo_concurrency import _Workers
from common.tests.test_dit_continual_hpo_dispatch import _continual_result
import common.semantic_hpo as semantic


_GROUPS = [[0, 1], [2, 3]]


def _normalize(profile: dict, dataset_name: str, seed: int) -> dict:
    """Keep fixed recipe semantics while skipping file loading in transport tests."""

    del dataset_name, seed
    return deepcopy(profile)


def _build(trial: object, **options: object) -> Config:
    """Build a small serializable semantic trial at the real profile boundary."""

    config = Config(**deepcopy(options["semantic_profile"]["student_config"]))
    steps = trial.suggest_categorical("acquisition_steps", [10, 20])
    config.hpo = {
        "trial_number": trial.number, "params": dict(trial.params), 
        "search_profile": semantic.PROFILE, "semantic_profile": deepcopy(options["semantic_profile"]), 
        "semantic_consolidation": {"acquisition_steps": steps}, 
        "use_ensemble_accuracy": False, "objective_metrics": ["final_average_accuracy"], 
        "objective_directions": ["maximize"]
    }
    return config


class SemanticDispatchTests(TestCase):
    """Keep task/output identity and validation ranking across the public API."""

    def setUp(self) -> None:
        """Allocate separate study storage and replace only the profile constructor."""

        temporary = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        config = Config(
            dataset={"name": "cifar10"}, 
            model={"name": "dit_classifier", "wrapper_name": "diffusion_classifier"}, 
            continually_learn={"class_num": 4, "task_size": 2, "task_groups": _GROUPS, "class_order": [0, 1, 2, 3]}, 
            training={"task": "continual", "epochs": 1, "seed": 42}
        )
        self.profile = {"student_config": asdict(config), "task_groups": _GROUPS, "route_settings": {"condition": "learned"}}
        context = patch.multiple(
            semantic, SEARCH_SPACE={"acquisition_steps": [10, 20]}, VERSION=1, 
            normalize_semantic_profile=_normalize, build_semantic_config=_build, validate_semantic_search=Mock()
        )
        context.start()
        self.addCleanup(context.stop)

    def options(self, **changes: object) -> dict:
        """Make an isolated two-worker request whose GPU processes are test doubles."""

        options = {
            "task": "continual", "model_name": "dit_classifier", "dataset_name": "cifar10", 
            "search_profile": semantic.PROFILE, "semantic_profile": self.profile, 
            "results_path": str(self.root), "n_trials": 2, "epochs": 1, "n_startup_trials": 1, 
            "trial_budget_mode": "total", "concurrent_trials": 2, "worker_gpu_ids": [0, 1], "seed": 42
        }
        options.update(changes)
        return options

    def test_semantic_trials_use_validation_and_disjoint_output_paths(self) -> None:
        """Official-test decoys never rank trials and native route outputs cannot collide."""

        workers = _Workers(outcomes={0: _continual_result, 1: _continual_result})
        with workers.installed():
            study = run_hpo(**self.options())
        self.assertEqual([trial.value for trial in study.trials], [0.5, 0.501])
        self.assertEqual(study.direction.name, "MAXIMIZE")
        sealed = study.user_attrs["study_spec"]["semantic_profile"]
        self.assertEqual(sealed["route_settings"], self.profile["route_settings"])
        self.assertEqual(sealed["task_groups"], _GROUPS)
        self.assertEqual(sealed["student_config"]["training"]["epochs"], 1)
        self.assertEqual(len({handle.config.training.results_path for handle in workers.handles}), 2)
        self.assertTrue(all(handle.config.continually_learn.resume_from is None for handle in workers.handles))
        self.assertTrue(all(handle.config.hpo["search_profile"] == semantic.PROFILE for handle in workers.handles))

    def test_changed_semantic_recipe_cannot_resume(self) -> None:
        """Fixed protocol changes are rejected before opening existing study storage."""

        workers = _Workers(outcomes={0: _continual_result})
        with workers.installed():
            study = run_hpo(**self.options(n_trials=1))
        root = load_config(study.trials[0].user_attrs["config_path"]).hpo["study_root"]
        changed = deepcopy(self.profile)
        changed["route_settings"]["condition"] = "random"
        with patch("optuna.load_study") as load, self.assertRaisesRegex(ValueError, "specification differs"):
            run_hpo(**self.options(semantic_profile=changed, resume_from=root))
        load.assert_not_called()

    def test_completed_trials_resume_without_retraining(self) -> None:
        """A terminal study can be extended without replaying already completed streams."""

        workers = _Workers(outcomes={0: _continual_result, 1: _continual_result})
        with workers.installed():
            study = run_hpo(**self.options(n_trials=1))
            root = load_config(study.trials[0].user_attrs["config_path"]).hpo["study_root"]
            resumed = run_hpo(**self.options(resume_from=root))
        self.assertEqual(len(workers.handles), 2)
        self.assertEqual([trial.value for trial in resumed.trials], [0.5, 0.501])

    def test_interrupted_trials_require_explicit_reconciliation(self) -> None:
        """An abandoned semantic stream cannot silently restore native-only checkpoints."""

        workers = _Workers(outcomes={0: KeyboardInterrupt()})
        with workers.installed(), self.assertRaises(KeyboardInterrupt):
            run_hpo(**self.options(n_trials=1))
        root = workers.handles[0].config.hpo["study_root"]
        with self.assertRaisesRegex(ValueError, "unfinished RUNNING"):
            run_hpo(**self.options(resume_from=root))

    def test_conflicting_native_controls_fail_before_storage(self) -> None:
        """Semantic-only search cannot alter native budgets, task order or validation."""

        for changes in (
            {"epochs": 2}, {"dtype_policy": "mixed_float16"}, {"task_groups": _GROUPS}, 
            {"validation_source": "test"}, {"pruning": {}}, {"model_overrides": {"dim": 8}}, 
            {"teacher_network": object()}, {"use_ensemble_accuracy": True}
        ):
            with self.subTest(changes=changes), patch("optuna.create_study") as create:
                with self.assertRaises(ValueError):
                    run_hpo(**self.options(**changes))
                create.assert_not_called()
