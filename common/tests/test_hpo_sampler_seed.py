"""Check independent Optuna seeds with real storage and bounded worker doubles.

Run only on an authorized remote container. These tests execute orchestration
and persistence; numerical model training is replaced by the worker fixture.
"""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

import optuna

from common.config import Config, load_config
from common.hpo import run_hpo
import common.semantic_hpo as semantic
from common.tests.test_dit_continual_hpo_dispatch import _continual_result
from common.tests.test_hpo_concurrency import _Workers
from common.tests.test_semantic_hpo_dispatch import _build, _normalize


class HpoSamplerSeedTests(TestCase):
    """Keep sampler diversity separate from model initialization and split identity."""

    def setUp(self) -> None:
        """Prepare a native semantic transport fixture without creating model objects."""

        temporary = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        groups = [[0, 1], [2, 3]]
        native = Config(
            dataset={"name": "cifar10"}, 
            model={"name": "dit_classifier", "wrapper_name": "diffusion_classifier"}, 
            continually_learn={
                "class_num": 4, "task_size": 2, "task_groups": groups, 
                "class_order": [0, 1, 2, 3], "seed": 42
            }, 
            training={"task": "continual", "epochs": 1, "seed": 42}
        )
        self.profile = {
            "student_config": asdict(native), "task_groups": groups, 
            "task_seed": 42, "dataset_seed": 42, "route_settings": {"condition": "learned"}
        }
        self.builder = Mock(side_effect=_build)
        context = patch.multiple(
            semantic, SEARCH_SPACE={"acquisition_steps": [10, 20]}, VERSION=1, 
            normalize_semantic_profile=_normalize, build_semantic_config=self.builder, 
            validate_semantic_search=Mock()
        )
        context.start()
        self.addCleanup(context.stop)

    def options(self, directory: str = "study", **changes: object) -> dict:
        """Construct a one-trial request using isolated fake GPU workers."""

        options = {
            "task": "continual", "model_name": "dit_classifier", "dataset_name": "cifar10", 
            "search_profile": semantic.PROFILE, "semantic_profile": deepcopy(self.profile), 
            "results_path": str(self.root / directory), "n_trials": 1, "epochs": 1, 
            "n_startup_trials": 2, "trial_budget_mode": "total", 
            "concurrent_trials": 2, "worker_gpu_ids": [0, 1], "seed": 42
        }
        options.update(changes)
        return options

    def remember_study(self, study: object) -> None:
        """Dispose each real SQLite engine before deleting the temporary study tree."""

        storage = getattr(study._storage, "_backend", study._storage)
        self.addCleanup(storage.engine.dispose)
        self.addCleanup(storage.remove_session)

    def test_independent_sampler_seeds_reach_optuna_without_reseeding_models(self) -> None:
        """Separate host searches change TPE initialization while preserving seed42 models."""

        sampler_class = optuna.samplers.TPESampler
        for sampler_seed in (421, 422):
            with self.subTest(sampler_seed=sampler_seed):
                workers = _Workers(outcomes={0: _continual_result})
                with workers.installed(), patch("optuna.samplers.TPESampler", wraps=sampler_class) as constructor:
                    study = run_hpo(**self.options(str(sampler_seed), sampler_seed=sampler_seed))
                self.remember_study(study)
                constructor.assert_called_once_with(n_startup_trials=2, seed=sampler_seed)
                self.assertEqual(study.user_attrs["study_spec"]["sampler_seed"], sampler_seed)
                self.assertEqual(study.user_attrs["study_spec"]["seed"], 42)
                self.assertEqual(study.user_attrs["study_spec"]["semantic_profile"]["dataset_seed"], 42)
                self.assertEqual(workers.handles[0].config.training.seed, 42)
                self.assertEqual(workers.handles[0].config.continually_learn.seed, 42)
        self.assertTrue(all(call.kwargs["seed"] == 42 for call in self.builder.call_args_list))

    def test_changed_sampler_seed_is_rejected_before_resume_opens_optuna(self) -> None:
        """A shard cannot silently replace its sealed sampler identity on restart."""

        workers = _Workers(outcomes={0: _continual_result})
        with workers.installed():
            study = run_hpo(**self.options(sampler_seed=421))
        self.remember_study(study)
        root = load_config(study.trials[0].user_attrs["config_path"]).hpo["study_root"]
        with patch("optuna.load_study") as load, self.assertRaisesRegex(ValueError, "specification differs"):
            run_hpo(**self.options(sampler_seed=422, resume_from=root))
        load.assert_not_called()

    def test_omitted_and_none_sampler_seeds_preserve_legacy_identity(self) -> None:
        """An unused optional control adds no new key to an existing study seal."""

        sampler_class = optuna.samplers.TPESampler
        specifications = []
        for directory, changes in [("omitted", {}), ("explicit-none", {"sampler_seed": None})]:
            workers = _Workers(outcomes={0: _continual_result})
            with workers.installed(), patch("optuna.samplers.TPESampler", wraps=sampler_class) as constructor:
                study = run_hpo(**self.options(directory, **changes))
            self.remember_study(study)
            constructor.assert_called_once_with(n_startup_trials=2, seed=42)
            specifications.append(study.user_attrs["study_spec"])
            self.assertNotIn("sampler_seed", specifications[-1])
        self.assertEqual(specifications[0], specifications[1])

    def test_same_explicit_seed_resumes_and_retains_completed_models(self) -> None:
        """A valid restart preserves sampler identity without repeating trial0."""

        workers = _Workers(outcomes={0: _continual_result, 1: _continual_result})
        with workers.installed():
            study = run_hpo(**self.options(sampler_seed=421))
            self.remember_study(study)
            root = load_config(study.trials[0].user_attrs["config_path"]).hpo["study_root"]
            resumed = run_hpo(**self.options(sampler_seed=421, n_trials=2, resume_from=root))
        self.remember_study(resumed)
        self.assertEqual([handle.number for handle in workers.handles], [0, 1])
        self.assertEqual([trial.state.name for trial in resumed.trials], ["COMPLETE", "COMPLETE"])
        self.assertEqual(resumed.user_attrs["study_spec"]["sampler_seed"], 421)
        self.assertTrue(all(handle.config.training.seed == 42 for handle in workers.handles))

    def test_legacy_study_resumes_with_explicit_none_without_new_identity(self) -> None:
        """Backward compatibility applies to actual stored state as well as fresh specs."""

        workers = _Workers(outcomes={0: _continual_result, 1: _continual_result})
        with workers.installed():
            study = run_hpo(**self.options())
            self.remember_study(study)
            root = load_config(study.trials[0].user_attrs["config_path"]).hpo["study_root"]
            resumed = run_hpo(**self.options(sampler_seed=None, n_trials=2, resume_from=root))
        self.remember_study(resumed)
        self.assertNotIn("sampler_seed", resumed.user_attrs["study_spec"])
        self.assertEqual([handle.number for handle in workers.handles], [0, 1])
