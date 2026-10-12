"""Semantic HPO selection, immutable recipes and actual route replay contracts."""

from __future__ import annotations

from contextlib import nullcontext
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


    def retention_plan(self) -> dict:
        """Select the sealed retention objective while preserving the native stream."""

        plan = deepcopy(self.plan)
        plan["hpo"]["semantic_profile"]["selection_metric"] = "retention_balanced_accuracy"
        plan["hpo"]["objective_metrics"] = ["retention_balanced_accuracy"]
        return plan

    def write_trial_configs(self) -> None:
        """Create immutable source bytes for the completed trial fixtures."""

        configs = Path(self.plan["study_root"]) / "configs"
        configs.mkdir(parents=True, exist_ok=True)
        for trial in self.trials:
            (configs / f"trial-{trial.number:04d}.yaml").write_text(f"trial: {trial.number}\n", encoding="utf-8")

    def test_retention_selection_binds_objective_and_both_schedule_seeds(self) -> None:
        """Retention scores never masquerade as average accuracy in summaries."""

        plan = self.retention_plan()
        plan["hpo"]["semantic_profile"]["task_seed"] = 77
        self.trials.extend([_trial(1, 0.2), _trial(2, 0.6), _trial(3, 0.8)])
        self.trials[2].params = dict(self.trials[1].params)
        self.write_trial_configs()
        summary = runner.search_summary(plan)
        self.assertEqual(summary["best_validation_objective"], 0.8)
        self.assertNotIn("best_validation_final_average_accuracy", summary)
        manifest = runner.freeze_finalists(plan, [101, 202, 303], top_k=2)
        self.assertEqual([candidate["trial_number"] for candidate in manifest["candidates"]], [3, 1])
        self.assertEqual(manifest["task_seed"], 77)
        self.assertEqual(manifest["dataset_seed"], 42)
        winner = manifest["candidates"][0]
        self.assertEqual(winner["search_objective"], 0.8)
        self.assertNotIn("search_final_average_accuracy", winner)
        self.assertEqual(winner["source_input_config_sha256"], winner["config_sha256"])
        changed = deepcopy(plan)
        changed["hpo"]["semantic_profile"]["dataset_seed"] = 17
        with self.assertRaisesRegex(ValueError, "different recipe"):
            runner.freeze_finalists(changed, [101, 202, 303], top_k=2)

    def test_finalists_block_live_workers_and_explicitly_close_queued_admission(self) -> None:
        """Selection excludes no live worker and preserves unstarted anchor evidence."""

        self.trials.extend([_trial(1, 0.2), _trial(2, 0.8)])
        self.write_trial_configs()
        self.trials[-1].state.name = "RUNNING"
        with self.assertRaisesRegex(ValueError, "running trials"):
            runner.freeze_finalists(self.plan, [101], top_k=1)
        self.trials[-1].state.name = "WAITING"
        manifest = runner.freeze_finalists(self.plan, [101], top_k=1)
        self.assertEqual(manifest["unstarted_trials_at_search_close"]["trial_numbers"], [2])
        self.assertEqual(self.trials[-1].state.name, "WAITING")

    def test_retention_receipt_rejects_mismatched_metric_and_seed(self) -> None:
        """A finite harmonic score needs the exact sealed task and split identities."""

        plan = self.retention_plan()
        result = {
            "objective": 0.3, "objective_metric": "retention_balanced_accuracy", 
            "objective_direction": "maximize", "objective_network": "raw", 
            "search_profile": runner.PROFILE, "fixed_recipe_sha256": "native-recipe", 
            "task_groups": self.profile["task_groups"], "dataset_seed": 42, "task_seed": 42, 
            "validation_continual_metrics": {"retention_balanced_accuracy": 0.3, "final_average_accuracy": 0.6}
        }
        runner.validate_confirmation_objective(plan, result)
        result["dataset_seed"] = 99
        with self.assertRaisesRegex(ValueError, "seed identity"):
            runner.validate_confirmation_objective(plan, result)
        result["dataset_seed"] = 42
        result["objective"] = 0.6
        with self.assertRaisesRegex(ValueError, "recorded validation metric"):
            runner.validate_confirmation_objective(plan, result)

    def test_retention_summary_uses_authenticated_objectives_and_reports_missing_seeds(self) -> None:
        """Paired harmonic scores retain their own names and incomplete coverage."""

        plan = self.retention_plan()
        self.trials.append(_trial(1, 0.4))
        self.write_trial_configs()
        manifest = runner.freeze_finalists(plan, [101, 202, 303], top_k=1)
        candidate = manifest["candidates"][0]
        manifest_path = Path(plan["control_root"]) / "finalists.json"
        record = {
            "identity": {
                "manifest_sha256": shared._digest(manifest_path), "config_sha256": candidate["config_sha256"], 
                "trial_number": 1, "training_seed": 101
            }, 
            "result": {
                "objective": 0.5, "objective_metric": "retention_balanced_accuracy", 
                "objective_direction": "maximize", "objective_network": "raw", 
                "search_profile": runner.PROFILE, "fixed_recipe_sha256": "native-recipe", 
                "task_groups": self.profile["task_groups"], "task_seed": 42, "dataset_seed": 42
            }
        }
        destination = Path(plan["control_root"]) / "confirmations" / "trial-0001" / "seed-101" / "completed.json"
        shared._write(destination, record)
        row = runner.confirmation_summary(plan)[0]
        self.assertEqual(row["mean_objective"], 0.5)
        self.assertEqual(row["completed_seeds"], 1)
        self.assertEqual(row["required_seeds"], 3)
        self.assertFalse(row["comparable"])
        self.assertNotIn("mean_final_average_accuracy", row)

    def test_retention_confirmation_selects_harmonic_metric_without_overwriting_accuracy(self) -> None:
        """The worker receipt scores retention while preserving both native metrics."""

        profile = deepcopy(self.profile)
        profile["selection_metric"] = "retention_balanced_accuracy"
        self.profile_module.normalize_semantic_profile.return_value = profile
        source = self.root / "retention.yaml"
        source.write_text("fixed-input", encoding="utf-8")
        config = SimpleNamespace(
            model=SimpleNamespace(kwargs={}, wrapper_kwargs={}, weights_path=None), 
            dataset=SimpleNamespace(name="cifar10"), 
            continually_learn=SimpleNamespace(task_groups=deepcopy(profile["task_groups"])), 
            hpo={
                "search_profile": runner.PROFILE, "trial_number": 4, "semantic_profile": profile, 
                "continual_dataset_seed": 42, "semantic_consolidation": {"condition": "learned", "seed": 42}
            }, 
            training=SimpleNamespace(seed=42)
        )
        output = self.root / "retention-output"
        output.mkdir()
        self.profile_module.run_semantic_trial.return_value = {
            "results_path": str(output), 
            "evaluations": {"validation_continual_metrics": {"final_average_accuracy": 0.6, "retention_balanced_accuracy": 0.3}}
        }
        fake_config = SimpleNamespace(load_config=Mock(return_value=config), save_config=Mock())
        with patch.dict(sys.modules, {"common.config": fake_config}):
            result = runner.run_confirmation(source, self.root / "retention-confirmation", 101, shared._digest(source))
        self.assertEqual(result["objective"], 0.3)
        self.assertEqual(result["validation_continual_metrics"]["final_average_accuracy"], 0.6)
        self.assertEqual(result["task_seed"], 42)
        self.assertEqual(result["dataset_seed"], 42)
        self.assertEqual(result["training_seed"], 101)

    def test_capacity_has_no_invented_pilot_throughput(self) -> None:
        """No completed streams leave both reserve and admission estimates unknown."""

        capacity = runner.capacity_plan(self.plan)
        self.assertTrue(capacity["pilot_required"])
        self.assertIsNone(capacity["conservative_full_trial_seconds"])
        self.assertIsNone(capacity["reserved_seconds"])
        self.assertIsNone(capacity["can_admit_full_trial"])

    def test_capacity_reserves_separate_confirmation_and_control_waves(self) -> None:
        """Completed p90 timings constrain a shared clock without extending it."""

        self.trials.extend([_trial(1, 0.4, hours=0.5), _trial(2, 0.6, hours=1.0)])
        plan = deepcopy(self.plan)
        plan["hpo"]["concurrent_trials"] = 16
        plan["identity"]["gpus"] = [{"gpu_id": index} for index in range(8)]
        plan["time_budget"] = {"confirmation_reserve_hours": 3.0}
        clock = {"deadline_unix": 43200.0, "search_deadline_unix": 32400.0, "cleanup_seconds": 60.0}
        with patch.object(shared, "_budget_state", return_value=clock), patch.object(runner.time, "time", return_value=7200.0):
            capacity = runner.capacity_plan(plan)
        self.assertEqual(capacity["repeat_waves"], 5)
        self.assertEqual(capacity["conservative_full_trial_seconds"], 4320.0)
        self.assertEqual(capacity["reserved_seconds"], 21600.0)
        self.assertEqual(capacity["search_completion_cutoff_unix"], 21480.0)
        self.assertEqual(capacity["admission_stop_unix"], 17160.0)
        self.assertEqual(capacity["sustainable_target"], 50)
        self.assertTrue(capacity["can_admit_full_trial"])

    def test_search_stops_before_a_full_trial_would_cross_the_cutoff(self) -> None:
        """The scheduler keeps the reserve instead of launching a doomed last wave."""

        capacity = {"can_admit_full_trial": False, "admission_stop_unix": 1.0, "sustainable_target": 0}
        with patch.object(runner, "capacity_plan", return_value=capacity), patch.object(shared, "_launch") as launch:
            status = runner.run_search(self.plan, target_completed=64)
        launch.assert_not_called()
        self.assertTrue(status["admission_stopped_for_full_trial"])
        self.assertFalse(status["time_budget_exhausted"])

    def test_paired_queue_reserves_one_combined_wave_at_verified_concurrency(self) -> None:
        """Three admitted slots run one learned repeat and its two controls concurrently."""

        self.trials.append(_trial(1, 0.4, hours=1.0))
        plan = deepcopy(self.plan)
        plan["hpo"]["concurrent_trials"] = 3
        plan["identity"]["gpus"] = [{"gpu_id": 0}]
        paired = runner.capacity_plan(plan, confirmation_runs=1, control_runs=2, paired_queue=True)
        legacy = runner.capacity_plan(plan, confirmation_runs=1, control_runs=2)
        self.assertEqual(paired["repeat_waves"], 1)
        self.assertEqual(paired["assumed_confirmation_workers"], 3)
        self.assertEqual(paired["estimated_repeat_reserve_seconds"], 4320.0)
        self.assertEqual((paired["confirmation_runs"], paired["control_runs"]), (1, 2))
        self.assertTrue(paired["paired_queue"])
        self.assertEqual(legacy["repeat_waves"], 3)
        self.assertFalse(legacy["paired_queue"])

    def test_admission_close_preserves_waiting_trials_and_detects_later_study_changes(self) -> None:
        """Early closure permits immutable selection without pretending hints completed."""

        self.trials.extend([_trial(1, 0.4), _trial(2, 0.8)])
        self.trials[-1].state.name = "WAITING"
        self.write_trial_configs()
        capacity = {"can_admit_full_trial": False, "admission_stop_unix": 1.0, "sustainable_target": 1}
        with patch.object(runner, "capacity_plan", return_value=capacity):
            runner.run_search(self.plan, target_completed=64)
        marker = runner.validate_search_closed(self.plan)
        self.assertEqual(marker["unstarted_trial_numbers"], [2])
        manifest = runner.freeze_finalists(self.plan, [101], top_k=1)
        self.assertEqual(manifest["unstarted_trials_at_search_close"]["trial_numbers"], [2])
        self.assertEqual(self.trials[-1].state.name, "WAITING")
        self.trials[0].value = 0.9
        with self.assertRaisesRegex(ValueError, "changed after"):
            runner.validate_search_closed(self.plan)

    def test_search_pilot_consumes_one_queued_worker_wave_without_extra_allocations(self) -> None:
        """An existing anchor queue cannot turn a bounded pilot into several waves."""

        plan = deepcopy(self.plan)
        plan["hpo"]["concurrent_trials"] = 2
        self.trials.extend([_trial(0, 0.3), _trial(1, 0.4), _trial(2, 0.5)])
        for trial in self.trials:
            trial.state.name = "WAITING"

        def launch(selected: dict, payload: dict, tag: str, **kwargs: object) -> dict:
            """Complete only the prefix authorized by the shared total allowance."""

            self.assertEqual(payload["allocated_target"], 2)
            for trial in self.trials[:2]:
                trial.state.name = "COMPLETE"
            return {}

        with patch.object(shared, "_launch", side_effect=launch) as admitted:
            status = runner.run_search(plan, target_completed=2, max_attempts=3)
        self.assertEqual(admitted.call_count, 1)
        self.assertEqual(status["completed_finite_trials"], 2)
        self.assertEqual(len(self.trials), 3)
        self.assertEqual(self.trials[-1].state.name, "WAITING")
        self.assertFalse(runner.search_closed(plan))

    def test_adaptive_pilot_target_never_expands_into_main_search(self) -> None:
        """A fast pilot remains a review boundary even when ample capacity remains."""

        self.trials.extend([_trial(1, 0.4), _trial(2, 0.5)])
        capacity = {"can_admit_full_trial": True, "admission_stop_unix": 9999999999.0, "sustainable_target": 128}
        with patch.object(runner, "capacity_plan", return_value=capacity), patch.object(shared, "_launch") as admitted:
            status = runner.run_search(self.plan, target_completed=2)
        admitted.assert_not_called()
        self.assertEqual(status["target_completed"], 2)
        self.assertFalse(runner.search_closed(self.plan))

    def test_measured_search_refills_past_one_wave_with_separate_admission_deadline(self) -> None:
        """Fast completed workers receive more trials while active streams may drain."""

        plan = deepcopy(self.plan)
        plan["hpo"]["concurrent_trials"] = 2
        self.trials.extend([_trial(1, 0.4), _trial(2, 0.5)])
        capacity = {
            "can_admit_full_trial": True, "admission_stop_unix": 2000.0, "sustainable_target": 128, 
            "conservative_full_trial_seconds": 100.0, "search_completion_cutoff_unix": 2100.0
        }

        def launch(selected: dict, payload: dict, tag: str, **kwargs: object) -> dict:
            """Complete the remaining target in one continuously refilled child."""

            self.assertEqual(payload["allocated_target"], 8)
            self.assertEqual(payload["admission_deadline"], 2000.0)
            self.assertEqual(kwargs["deadline"], 2100.0)
            self.trials.extend(_trial(number, 0.5) for number in range(3, 9))
            return {}

        with patch.object(runner, "capacity_plan", return_value=capacity), \
        patch.object(shared, "_phase_deadline", return_value=2200.0), \
        patch.object(runner.time, "time", return_value=1000.0), \
        patch.object(shared, "_launch", side_effect=launch) as admitted:
            status = runner.run_search(plan, target_completed=8)
        self.assertEqual(admitted.call_count, 1)
        self.assertEqual(status["completed_finite_trials"], 8)
        self.assertFalse(runner.search_closed(plan))

    def test_admission_expiry_during_worker_startup_is_a_normal_no_allocation_stop(self) -> None:
        """Startup overhead may close admission without a trial-state transition."""

        self.trials.append(_trial(1, 0.4))
        capacity = {
            "can_admit_full_trial": True, "admission_stop_unix": 2000.0, "sustainable_target": 128, 
            "conservative_full_trial_seconds": 100.0, "search_completion_cutoff_unix": 2100.0
        }
        with patch.object(runner, "capacity_plan", return_value=capacity), \
        patch.object(shared, "_phase_deadline", return_value=2200.0), \
        patch.object(runner.time, "time", return_value=1000.0), \
        patch.object(shared, "_launch", return_value={"admission_stopped": True}) as admitted:
            status = runner.run_search(self.plan, target_completed=8)
        self.assertEqual(admitted.call_count, 1)
        self.assertTrue(status["admission_stopped_for_full_trial"])
        self.assertTrue(runner.search_closed(self.plan))

    def test_control_dispatch_is_separate_paired_and_idempotent(self) -> None:
        """Native controls cannot enter learned summaries or repeat completed jobs."""

        self.trials.append(_trial(1, 0.4))
        self.write_trial_configs()
        runner.freeze_finalists(self.plan, [101], top_k=1)

        def launch(plan: dict, payload: dict, tag: str, **kwargs: object) -> dict:
            """Return an admitted control's authenticated full-stream result."""

            self.assertEqual(payload["kind"], "semantic_control")
            return {
                "objective": 0.3, "objective_metric": "final_average_accuracy", 
                "objective_direction": "maximize", "objective_network": "raw", 
                "search_profile": runner.PROFILE, "fixed_recipe_sha256": "native-recipe", 
                "task_groups": self.profile["task_groups"], "task_seed": 42, "dataset_seed": 42, 
                "training_seed": payload["training_seed"], "source_input_config_sha256": payload["expected_config_sha256"], 
                "control_condition": payload["control_condition"]
            }

        with patch.object(shared, "_launch", side_effect=launch) as admitted:
            first = runner.run_controls(self.plan)
            second = runner.run_controls(self.plan)
        self.assertEqual(admitted.call_count, 2)
        self.assertEqual(first, second)
        self.assertEqual({row["result"]["control_condition"] for row in first}, {"baseline", "unmodulated_feature_distillation"})
        with self.assertRaisesRegex(ValueError, "cannot count"):
            runner.validate_confirmation_objective(self.plan, first[0]["result"])
        summary = runner.controls_summary(self.plan)
        self.assertTrue(all(row["all_seeds_complete"] for row in summary))
        self.assertTrue(all(not row["comparable"] for row in summary))

    def test_shared_worker_drains_admitted_trials_before_hard_completion_deadline(self) -> None:
        """Separate admission timeout preserves active streams and the hard watchdog."""

        plan = deepcopy(self.plan)
        plan["hpo"]["concurrent_trials"] = 2
        factory = object()
        remote = SimpleNamespace(
            managed_parallel_coordinator=Mock(side_effect=lambda *args: nullcontext(factory)), 
            managed_worker=Mock(side_effect=lambda *args: nullcontext(None))
        )
        for admission, expected_timeout, stop_active in ((500.0, 400.0, False), (None, 840.0, True)):
            payload = {"kind": "search", "allocated_target": 8}
            # Measured searches use a separate admission cutoff; legacy searches do not.
            if admission is not None:
                payload["admission_deadline"] = admission
            path = self.root / f"worker-{stop_active}.json"
            receipt = path.with_suffix(".result.json")
            shared._write(path, {"plan": plan, "payload": payload, "deadline": 1000.0, "receipt_path": str(receipt)})
            with self.subTest(admission=admission), patch.object(shared.time, "time", return_value=100.0) as clock:

                def complete(**kwargs: object) -> None:
                    """Simulate an active stream draining after new admission has ended."""

                    clock.return_value = 700.0

                hpo = SimpleNamespace(run_hpo=Mock(side_effect=complete))
                with patch.dict(sys.modules, {"common.dit_hpo_remote": remote, "common.hpo": hpo}), \
                patch.object(shared, "search_summary", return_value={}):
                    shared._worker(path)
            arguments = hpo.run_hpo.call_args.kwargs
            self.assertEqual(arguments["timeout"], expected_timeout)
            self.assertEqual(arguments["stop_active_on_timeout"], stop_active)
            self.assertIs(arguments["worker_context"], factory)
            result = shared._read(receipt)["result"]
            self.assertFalse(result["time_budget_exhausted"])
            self.assertEqual(result.get("admission_stopped", False), admission is not None)
        remote.managed_worker.assert_not_called()

    def test_late_confirmation_keeps_full_stream_buffer_and_does_not_launch(self) -> None:
        """A resumed confirmation cannot start a stream with insufficient measured time."""

        self.trials.append(_trial(1, 0.4))
        self.write_trial_configs()
        runner.freeze_finalists(self.plan, [101], top_k=1)
        capacity = {"conservative_full_trial_seconds": 500.0}
        with patch.object(shared, "_phase_deadline", return_value=1000.0), \
        patch.object(shared.time, "time", return_value=600.0), \
        patch.object(runner, "capacity_plan", return_value=capacity), \
        patch.object(shared, "_launch") as admitted:
            completed = runner.run_confirmations(self.plan)
        admitted.assert_not_called()
        self.assertEqual(completed, [])
        status = shared._read(Path(self.plan["control_root"]) / "confirmation_status.json")
        self.assertTrue(status["admission_stopped_for_full_trial"])
        self.assertFalse(status["time_budget_exhausted"])
        self.assertFalse(status["all_pairs_complete"])

    def test_shared_control_worker_dispatches_only_the_native_control_entrypoint(self) -> None:
        """Control dispatch uses one admitted worker and never calls learned HPO."""

        remote = SimpleNamespace(
            managed_parallel_coordinator=Mock(), 
            managed_worker=Mock(side_effect=lambda *args: nullcontext(None))
        )
        payload = {
            "kind": "semantic_control", "input_config_path": "frozen.yaml", "output_root": "attempt", 
            "training_seed": 101, "expected_config_sha256": "sha", "control_condition": "baseline"
        }
        path = self.root / "control-worker.json"
        receipt = path.with_suffix(".result.json")
        shared._write(path, {"plan": self.plan, "payload": payload, "receipt_path": str(receipt)})
        with patch.dict(sys.modules, {"common.dit_hpo_remote": remote}), \
        patch.object(runner, "run_control", return_value={"objective": 0.3}) as control, \
        patch.object(runner, "run_confirmation") as learned:
            shared._worker(path)
        control.assert_called_once_with(**{key: value for key, value in payload.items() if key != "kind"})
        learned.assert_not_called()
        remote.managed_parallel_coordinator.assert_not_called()
        remote.managed_worker.assert_called_once()
        self.assertEqual(shared._read(receipt)["result"], {"objective": 0.3})

    def test_control_launch_routes_one_device_without_mutating_the_campaign(self) -> None:
        """A control cannot accidentally inherit the complete multi-GPU reservation."""

        plan = deepcopy(self.plan)
        plan["identity"]["gpus"] = [{"gpu_id": 0}, {"gpu_id": 1}]
        original = deepcopy(plan)
        serial = {**plan["identity"], "gpus": [{"gpu_id": 1}], "concurrent_trials": 1}

        def launch(command: list, checkout: str, identity: dict, log_path: Path, **kwargs: object) -> object:
            """Publish a child receipt using the exact selected serial identity."""

            self.assertEqual(identity, serial)
            request_path = Path(command[-1])
            request = shared._read(request_path)
            self.assertEqual(request["plan"]["identity"], serial)
            shared._write(request["receipt_path"], {"request_sha256": shared._digest(request_path), "result": {"objective": 0.3}})
            return nullcontext(SimpleNamespace(wait=lambda: 0))

        remote = SimpleNamespace(serial_worker_identity=Mock(return_value=serial), launch_worker=launch)
        with patch.dict(sys.modules, {"common.dit_hpo_remote": remote}):
            result = shared._launch(plan, {"kind": "semantic_control"}, "control-device", gpu_id=1)
        remote.serial_worker_identity.assert_called_once_with(plan["checkout_root"], original["identity"], gpu_id=1)
        self.assertEqual(result, {"objective": 0.3})
        self.assertEqual(plan, original)


# Direct execution retains the repository's focused unittest entry point.
if __name__ == "__main__":
    unittest.main()
