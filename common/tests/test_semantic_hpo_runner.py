"""Semantic HPO selection, immutable recipes and actual route replay contracts."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from common import dit_hpo_runner as shared
from common import semantic_hpo_runner as runner


def _trial(number: int, value: float, hours: float = 2.0) -> SimpleNamespace:
    """Provide real finite-trial timing and metric fields for coordinator checks."""

    started = datetime(2026, 1, 1)
    return SimpleNamespace(
        number=number, value=value, state=SimpleNamespace(name="COMPLETE"), 
        params={"semantic_learning_rate": 0.001 * number}, 
        datetime_start=started, datetime_complete=started + timedelta(hours=hours)
    )


def _reseed(config: object, seed: int) -> None:
    """Stand in for the separately tested native dataclass seed projection."""

    config.training.seed = seed
    config.continually_learn.seed = seed
    config.model.kwargs["seed"] = seed
    config.model.wrapper_kwargs["seed"] = seed
    config.hpo["semantic_consolidation"]["seed"] = seed


class SemanticRunnerTests(unittest.TestCase):
    """Use real immutable artifacts with mocked remote admission and training."""

    def setUp(self) -> None:
        """Prepare a complete profile without importing a local learning framework."""

        temporary = tempfile.TemporaryDirectory(prefix="semantic-runner-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.profile = {
            "student_config": {
                "model": {"name": "dit_classifier"}, 
                "dataset": {"validation_source": "split", "validation_ratio": 0.2}, 
                "training": {"epochs": 50, "dtype_policy": "float32", "deterministic_ops": False}, 
                "continually_learn": {
                    "task_size": 2, "use_distillation": True, "use_ensemble_accuracy": False, 
                    "ensemble_accuracy_kwargs": {}
                }
            }, 
            "route_settings": {"condition": "learned"}, "task_seed": 42, "dataset_seed": 42, 
            "task_groups": [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], 
            "artifact_sha256": {}, "profile_version": 1, "fixed_recipe_sha256": "native-recipe"
        }
        identity = {
            "source_sha256": {"semantic_consolidation/runner.py": "source-identity"}, 
            "versions": {}, "python": "test-python", "worker_policy": {"tf_memory_mib": 12288}
        }
        self.remote = SimpleNamespace(inspect_remote=Mock(return_value=identity))
        self.profile_module = SimpleNamespace(
            normalize_semantic_profile=Mock(return_value=deepcopy(self.profile)), 
            validate_semantic_search=Mock(), validate_semantic_config=Mock(), run_semantic_trial=Mock(), 
            reseed_semantic_config=Mock(side_effect=_reseed)
        )
        modules = {"common.dit_hpo_remote": self.remote, "common.semantic_hpo": self.profile_module}
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.plan = self.make_plan()
        self.trials = []
        self.study = SimpleNamespace(get_trials=lambda deepcopy=False: list(self.trials))
        loader = patch.object(shared, "_load_study", return_value=self.study)
        loader.start()
        self.addCleanup(loader.stop)

    def make_plan(self, **kwargs: object) -> dict:
        """Build one native semantic profile through the public shared planner."""

        return runner.make_plan(
            self.root, self.root / "results", search_profile=runner.PROFILE, 
            semantic_profile=self.profile, **kwargs
        )

    def test_plan_seals_native_inputs_and_maximizes_full_stream(self) -> None:
        """Native task/validation settings survive without teacher-source hints."""

        options = self.plan["hpo"]
        self.assertEqual(options["task"], "continual")
        self.assertEqual(options["objective_metrics"], ["final_average_accuracy"])
        self.assertEqual(options["objective_directions"], ["maximize"])
        self.assertEqual(options["semantic_profile"], self.profile)
        self.assertNotIn("initial_trials", options)
        self.assertTrue(self.plan["study_root"].endswith("continual/dit_classifier/cifar10/semantic_consolidation_runner"))
        self.assertNotIn("time_budget", self.plan)
        recipe = shared._read(Path(self.plan["control_root"]) / "recipe.json")
        self.assertIn("semantic_consolidation/runner.py", recipe["source_sha256"])

    def test_plan_rejects_changed_native_recipe_on_restart(self) -> None:
        """An edited fixed architecture cannot be mixed into an existing study."""

        changed = deepcopy(self.profile)
        changed["fixed_recipe_sha256"] = "other-architecture"
        self.profile_module.normalize_semantic_profile.return_value = changed
        with self.assertRaisesRegex(ValueError, "recipe changed"):
            self.make_plan()

    def test_plan_rejects_different_validation_or_epoch_budget_before_admission(self) -> None:
        """Planner mismatches fail before consuming remote worker reservations."""

        self.remote.inspect_remote.reset_mock()
        with self.assertRaisesRegex(ValueError, "must match"):
            self.make_plan(epochs=1)
        with self.assertRaisesRegex(ValueError, "must match"):
            self.make_plan(validation_source="test", validation_ratio=0.0)
        with self.assertRaisesRegex(ValueError, "pruning=None"):
            self.make_plan(pruning={"enabled": True})
        self.remote.inspect_remote.assert_not_called()

    def test_summary_ignores_pruned_scores_and_maximizes(self) -> None:
        """A high partial task score cannot win the final-stream ranking."""

        self.trials.extend([_trial(1, 0.35), _trial(2, 0.75), _trial(3, 0.99)])
        self.trials[-1].state.name = "PRUNED"
        summary = shared.search_summary(self.plan)
        self.assertEqual(summary["best_trial"], 2)
        self.assertEqual(summary["completed_finite_trials"], 2)
        self.assertEqual(summary["best_validation_final_average_accuracy"], 0.75)

    def test_finalists_preserve_highest_score_fresh_seeds_and_artifact_identity(self) -> None:
        """Shared selection retains native inputs and authenticates frozen YAML."""

        self.trials.extend([_trial(1, 0.35), _trial(2, 0.75)])
        configs = Path(self.plan["study_root"]) / "configs"
        configs.mkdir(parents=True)
        for trial in self.trials:
            (configs / f"trial-{trial.number:04d}.yaml").write_text(f"trial: {trial.number}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different from the search seed"):
            shared.freeze_finalists(self.plan, [42], top_k=1)
        manifest = shared.freeze_finalists(self.plan, [101], top_k=1)
        self.assertEqual(manifest["candidates"][0]["trial_number"], 2)
        self.assertEqual(manifest["task_groups"], self.profile["task_groups"])
        self.assertEqual(shared.freeze_finalists(self.plan, [101], top_k=1), manifest)
        Path(manifest["candidates"][0]["input_config_path"]).write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "has changed"):
            shared.freeze_finalists(self.plan, [101], top_k=1)

    def test_confirmation_rejects_plain_continual_receipt(self) -> None:
        """A finite ordinary continual result cannot claim semantic completion."""

        result = {
            "objective": 0.75, "objective_metric": "final_average_accuracy", 
            "objective_direction": "maximize", "objective_network": "raw", 
            "task_groups": self.profile["task_groups"]
        }
        with self.assertRaisesRegex(ValueError, "frozen task stream"):
            shared._validate_confirmation_objective(self.plan, result)
        result.update({"search_profile": runner.PROFILE, "fixed_recipe_sha256": "native-recipe"})
        shared._validate_confirmation_objective(self.plan, result)
        result["task_groups"] = list(reversed(result["task_groups"]))
        with self.assertRaisesRegex(ValueError, "frozen task stream"):
            shared._validate_confirmation_objective(self.plan, result)

    def test_confirmation_calls_actual_semantic_adapter_with_new_seeds(self) -> None:
        """A fresh repeat routes through the semantic training helper, not main()."""

        source = self.root / "input.yaml"
        source.write_text("fixed-input", encoding="utf-8")
        config = SimpleNamespace(
            model=SimpleNamespace(
                kwargs={"seed": 42}, wrapper_kwargs={"seed": 42}, 
                weights_path="original-student.weights.h5"
            ), 
            dataset=SimpleNamespace(name="CIFAR10"), 
            continually_learn=SimpleNamespace(task_groups=deepcopy(self.profile["task_groups"])), 
            hpo={
                "search_profile": runner.PROFILE, "trial_number": 4, "semantic_profile": self.profile, 
                "continual_dataset_seed": 42, "semantic_consolidation": {"condition": "learned", "seed": 42}
            }, 
            training=SimpleNamespace(seed=42)
        )
        output = self.root / "run-output"
        output.mkdir()
        self.profile_module.run_semantic_trial.return_value = {
            "results_path": str(output), 
            "evaluations": {"validation_continual_metrics": {"final_average_accuracy": 0.8}}
        }
        fake_config = SimpleNamespace(load_config=Mock(return_value=config), save_config=Mock())
        with patch.dict(sys.modules, {"common.config": fake_config}):
            result = runner.run_confirmation(source, self.root / "confirmation", 101, shared._digest(source))
        self.profile_module.run_semantic_trial.assert_called_once_with(config)
        self.assertEqual(self.profile_module.validate_semantic_config.call_count, 2)
        self.assertEqual(config.training.seed, 101)
        self.assertEqual(config.model.kwargs["seed"], 101)
        self.assertEqual(config.model.wrapper_kwargs["seed"], 101)
        self.assertEqual(config.continually_learn.seed, 101)
        self.assertEqual(config.hpo["semantic_consolidation"]["seed"], 101)
        self.assertEqual(config.hpo["continual_dataset_seed"], 42)
        self.assertEqual(config.model.weights_path, "original-student.weights.h5")
        self.assertEqual(result["objective"], 0.8)
        shared._validate_confirmation_objective(self.plan, result)

    def test_cost_estimate_requires_measured_complete_streams(self) -> None:
        """No timing is invented; short failed attempts do not bias trial duration."""

        self.assertIsNone(runner.cost_estimate(self.plan)["median_full_trial_hours"])
        self.trials.extend([_trial(1, 0.7, hours=2.0), _trial(2, 0.8, hours=4.0), _trial(3, 0.1, hours=0.1)])
        self.trials[-1].state.name = "FAIL"
        estimate = runner.cost_estimate(self.plan, remaining_trials=6, confirmation_runs=2, workers=2)
        self.assertEqual(estimate["completed_timed_trials"], 2)
        self.assertEqual(estimate["median_full_trial_hours"], 3.0)
        self.assertEqual(estimate["estimated_remaining_wall_hours_median"], 15.0)
        self.assertEqual(estimate["failed_or_pruned_worker_hours"], 0.1)


# Direct execution retains the repository's focused unittest entry point.
if __name__ == "__main__":
    unittest.main()
