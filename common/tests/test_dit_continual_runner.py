"""Continual notebook recipe, maximizing selection and confirmation contracts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from common import dit_continual_runner as runner
from common import dit_hpo_runner as shared
from common.dit_continual_hpo import baseline_hints


def _trial(number: int, value: float, classifier: str = "none", noise: str = "none") -> SimpleNamespace:
    """Provide the finite public Optuna fields read by the notebook coordinator."""

    return SimpleNamespace(
        number=number, value=value, state=SimpleNamespace(name="COMPLETE"), 
        params={"classifier_teacher_source": classifier, "noise_teacher_source": noise}
    )


class ContinualRunnerTests(unittest.TestCase):
    """Use real immutable artifacts around mocked GPU admission and training."""

    def setUp(self) -> None:
        """Prepare a fixed task stream without importing TensorFlow or Keras."""

        temporary = tempfile.TemporaryDirectory(prefix="dit-continual-runner-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.profile = {
            "student_config": {"model": {"name": "dit_classifier"}}, 
            "specialist_teacher_descriptors": {}, "task_seed": 42, 
            "task_groups": [[7, 3], [2, 8], [5, 6], [9, 4], [0, 1]], 
            "class_order": [7, 3, 2, 8, 5, 6, 9, 4, 0, 1], "artifact_sha256": {}, "profile_version": 1
        }
        identity = {
            "source_sha256": {}, "versions": {}, "python": "test-python", 
            "worker_policy": {"tf_memory_mib": 12288}
        }
        self.remote = SimpleNamespace(inspect_remote=Mock(return_value=identity))
        self.profile_module = SimpleNamespace(
            normalize_continual_profile=Mock(return_value=deepcopy(self.profile)), 
            validate_continual_search=Mock(), 
            baseline_hints=baseline_hints
        )
        modules = {"common.dit_hpo_remote": self.remote, "common.dit_continual_hpo": self.profile_module}
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.plan = runner.make_plan(
            self.root, self.root / "results", search_profile="dit_continual_runner", 
            continual_profile=self.profile
        )
        self.trials = []
        self.study = SimpleNamespace(get_trials=lambda deepcopy=False: list(self.trials))
        loader = patch.object(shared, "_load_study", return_value=self.study)
        loader.start()
        self.addCleanup(loader.stop)

    def test_recipe_selects_continual_raw_accuracy_and_sixteen_hints(self) -> None:
        """The new profile cannot accidentally inherit generation minimization."""

        options = self.plan["hpo"]
        self.assertEqual(options["task"], "continual")
        self.assertEqual(options["objective_metrics"], ["final_average_accuracy"])
        self.assertEqual(options["objective_directions"], ["maximize"])
        self.assertEqual(len(options["initial_trials"]), 16)
        self.assertEqual(options["continual_profile"]["task_groups"], self.profile["task_groups"])
        self.assertTrue(self.plan["study_root"].endswith("continual/dit_classifier/cifar10/dit_continual_runner"))
        self.assertNotIn("time_budget", self.plan)

    def test_task_recipe_change_is_rejected(self) -> None:
        """A new stream cannot be mixed into an existing notebook study."""

        changed = deepcopy(self.profile)
        changed["task_groups"] = list(reversed(changed["task_groups"]))
        self.profile_module.normalize_continual_profile.return_value = changed
        with self.assertRaisesRegex(ValueError, "recipe changed"):
            runner.make_plan(
                self.root, self.root / "results", search_profile="dit_continual_runner", 
                continual_profile=changed
            )

    def test_summary_maximizes_and_distinguishes_failed_coverage(self) -> None:
        """Pruned source routes remain incomplete and lower accuracy cannot win."""

        self.trials.extend([_trial(1, 0.35), _trial(2, 0.75, classifier="both", noise="both")])
        pruned = _trial(3, 0.99, classifier="current", noise="previous")
        pruned.state.name = "PRUNED"
        self.trials.append(pruned)
        summary = shared.search_summary(self.plan)
        self.assertEqual(summary["best_trial"], 2)
        self.assertEqual(summary["best_validation_final_average_accuracy"], 0.75)
        self.assertEqual(summary["teacher_source_coverage"]["current/previous"], {"allocated": 1, "complete": 0})
        self.assertFalse(summary["all_teacher_modes_completed"])

    def test_finalist_selection_and_dispatch_preserve_maximum_accuracy(self) -> None:
        """Shared orchestration freezes the higher-accuracy continual recipe."""

        self.trials.extend([_trial(1, 0.35), _trial(2, 0.75, classifier="both", noise="both")])
        configs = Path(self.plan["study_root"]) / "configs"
        configs.mkdir(parents=True)
        for trial in self.trials:
            (configs / f"trial-{trial.number:04d}.yaml").write_text(f"trial: {trial.number}\n", encoding="utf-8")
        manifest = shared.freeze_finalists(self.plan, [101], top_k=1)
        self.assertEqual(manifest["candidates"][0]["trial_number"], 2)
        self.assertEqual(manifest["selection_policy"]["objective_direction"], "maximize")
        self.assertEqual(manifest["task_groups"], self.profile["task_groups"])
        self.assertEqual(shared.freeze_finalists(self.plan, [101], top_k=1), manifest)
        Path(manifest["candidates"][0]["input_config_path"]).write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "has changed"):
            shared.freeze_finalists(self.plan, [101], top_k=1)

    def test_coverage_resolves_scalar_and_choices_mapping_overrides(self) -> None:
        """Coverage uses the same categorical override interpretation as queued trials."""

        self.trials.extend([_trial(1, 0.8, classifier="current", noise="none")])
        for classifier, noise in (
            ("current", "none"), 
            ({"choices": ["current"]}, {"choices": ["none"]}), 
            (["current"], {"choices": "none"})
        ):
            with self.subTest(classifier=classifier, noise=noise):
                self.plan["hpo"]["search_space_overrides"] = {
                    "classifier_teacher_source": classifier, "noise_teacher_source": noise
                }
                summary = shared.search_summary(self.plan)
                self.assertEqual(summary["teacher_source_coverage"], {"current/none": {"allocated": 1, "complete": 1}})
                self.assertTrue(summary["all_teacher_modes_completed"])

    def test_confirmation_identity_rejects_altered_task_stream(self) -> None:
        """A finite scalar alone does not authenticate a continual repeat."""

        result = {
            "objective": 0.75, "objective_metric": "final_average_accuracy", 
            "objective_direction": "maximize", "objective_network": "raw", 
            "task_groups": self.profile["task_groups"]
        }
        shared._validate_confirmation_objective(self.plan, result)
        result["task_groups"] = list(reversed(result["task_groups"]))
        with self.assertRaisesRegex(ValueError, "frozen task stream"):
            shared._validate_confirmation_objective(self.plan, result)

    def test_confirmation_replays_public_api_with_frozen_data_and_new_training_seed(self) -> None:
        """A fresh repeat retains the supplied initial weights and teacher recipes."""

        source = self.root / "input.yaml"
        source.write_text("fixed-input", encoding="utf-8")
        config = SimpleNamespace(
            model=SimpleNamespace(
                wrapper_name="diffusion_classifier", 
                kwargs={"seed": 42}, wrapper_kwargs={"use_ema": False, "test_network_name": "raw", "seed": 42}, 
                weights_path="original-student.weights.h5", name="dit_classifier"
            ), 
            dataset=SimpleNamespace(validation_source="split"), 
            continually_learn=SimpleNamespace(task_groups=deepcopy(self.profile["task_groups"])), 
            hpo={
                "search_profile": "dit_continual_runner", "objective_metrics": ["final_average_accuracy"], 
                "objective_directions": ["maximize"], "trial_number": 4, 
                "continual_profile": self.profile, 
                "specialist_teacher_descriptors": {"classifier": {"path": "input.keras"}}
            }, 
            training=SimpleNamespace(task="continual", use_valset=True, seed=42)
        )
        output = self.root / "run-output"
        output.mkdir()
        main = Mock(return_value={
            "results_path": str(output), 
            "evaluations": {"validation_continual_metrics": {"final_average_accuracy": 0.8}}
        })
        fake_config = SimpleNamespace(load_config=Mock(return_value=config), save_config=Mock())
        with patch.dict(sys.modules, {"common.config": fake_config, "common.train": SimpleNamespace(main=main)}):
            result = runner.run_confirmation(source, self.root / "confirmation", 101, shared._digest(source))
        main.assert_called_once_with(config)
        self.assertEqual(config.training.seed, 101)
        self.assertEqual(config.hpo["continual_dataset_seed"], 42)
        self.assertEqual(config.model.weights_path, "original-student.weights.h5")
        self.assertEqual(config.model.kwargs["seed"], 101)
        self.assertEqual(config.continually_learn.seed, 101)
        self.assertEqual(result["task_groups"], self.profile["task_groups"])
        self.assertEqual(result["objective"], 0.8)


# Direct execution retains the standard focused unittest entry point.
if __name__ == "__main__":
    unittest.main()
