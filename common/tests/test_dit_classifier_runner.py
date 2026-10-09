"""Pareto classifier runner and confirmation contracts without scientific training."""

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

from common import dit_hpo_runner as runner
from common.dit_hpo_confirmation import run_confirmation
from common.tests import test_dit_hpo_confirmation as generation_confirmation_tests
from common.tests import test_dit_hpo_runner as generation_runner_tests


class _ParetoTrial:
    """Expose vector trial fields while detecting accidental scalar accesses."""

    def __init__(self, number: int, values: list | None, state: str = "COMPLETE") -> None:
        """Retain two ordered objectives and distinguishable sampled settings."""

        self.number = number
        self.values = values
        self.state = SimpleNamespace(name=state)
        self.params = {"dim": 64, "learning_rate": 0.001 + number / 10000}

    @property
    def value(self) -> None:
        """Match Optuna's refusal to provide a scalar for a multiobjective trial."""

        raise RuntimeError("A Pareto trial has no scalar value.")


def _result(accuracy: float = 0.8, noise: float = 0.02) -> dict:
    """Produce one complete raw classifier confirmation receipt payload."""

    return {
        "objectives": [accuracy, noise], "objective_metrics": ["classification_accuracy", "noise_loss"], 
        "objective_directions": ["maximize", "minimize"], "objective_network": "raw"
    }


class ClassifierRunnerTests(unittest.TestCase):
    """Keep Pareto feedback separate from the unchanged scalar generation runner."""

    def setUp(self) -> None:
        """Reuse public-API fixtures while selecting an independent classifier study."""

        self.fixture = generation_runner_tests.DitHpoRunnerTests("test_recipe_restart_is_immutable")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        with patch.dict(sys.modules, {"common.dit_hpo_remote": self.fixture.remote}):
            self.plan = runner.make_plan(
                self.fixture.checkout, self.fixture.root / "classifier", 
                validation_source="test", validation_ratio=0.0, search_profile="dit_classifier_runner"
            )
        self.fixture.plan = self.plan

    def _candidate(self, number: int, accuracy: float, noise: float) -> _ParetoTrial:
        """Persist a vector trial and its immutable input for finalist selection."""

        trial = _ParetoTrial(number, [accuracy, noise])
        self.fixture.trials.append(trial)
        path = Path(self.plan["study_root"]) / "configs" / f"trial-{number:04d}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"candidate: {number}\n", encoding="utf-8")
        return trial

    def test_named_profile_paths_and_immutable_pareto_recipe(self) -> None:
        """The shared HPO path and both directed objectives are sealed together."""

        self.assertEqual(self.plan["study_name"], "joint-dit_classifier-cifar10-dit_classifier_runner")
        self.assertEqual(
            Path(self.plan["study_root"]), 
            self.fixture.root / "classifier" / "joint" / "dit_classifier" / "cifar10" / "dit_classifier_runner"
        )
        self.assertEqual(self.plan["hpo"]["objective_metrics"], ["classification_accuracy", "noise_loss"])
        self.assertEqual(self.plan["hpo"]["objective_directions"], ["maximize", "minimize"])
        self.assertEqual(self.plan["hpo"]["validation_source"], "test")
        with patch.dict(sys.modules, {"common.dit_hpo_remote": self.fixture.remote}):
            same = runner.make_plan(
                self.fixture.checkout, self.fixture.root / "classifier", 
                validation_source="test", validation_ratio=0.0, search_profile="dit_classifier_runner"
            )
            self.assertEqual(same["hpo"], self.plan["hpo"])
            with self.assertRaisesRegex(ValueError, "recipe changed"):
                runner.make_plan(
                    self.fixture.checkout, self.fixture.root / "classifier", 
                    validation_source="split", search_profile="dit_classifier_runner"
                )

    def test_baseline_is_sealed_and_explicit_overrides_omit_it(self) -> None:
        """Fresh hints use the existing queue without forcing values outside narrowed distributions."""

        self.assertEqual(len(self.plan["hpo"]["initial_trials"]), 1)
        recipe = runner._read(Path(self.plan["control_root"]) / "recipe.json")
        self.assertEqual(recipe["hpo"]["initial_trials"], self.plan["hpo"]["initial_trials"])
        with patch.dict(sys.modules, {"common.dit_hpo_remote": self.fixture.remote}):
            narrow = runner.make_plan(
                self.fixture.checkout, self.fixture.root / "narrow", 
                search_space_overrides={"dim": [64]}, search_profile="dit_classifier_runner"
            )
            generation = runner.make_plan(self.fixture.checkout, self.fixture.root / "generation")
        self.assertNotIn("initial_trials", narrow["hpo"])
        self.assertNotIn("initial_trials", generation["hpo"])
        self.assertNotIn("search_profile", generation["hpo"])

    def test_scalar_pruning_and_generation_transfer_fail_before_inspection(self) -> None:
        """Unsupported semantics cannot create files or inspect a remote runtime."""

        self.fixture.remote.inspect_remote.reset_mock()
        with patch.dict(sys.modules, {"common.dit_hpo_remote": self.fixture.remote}):
            for options in (
                {"search_profile": "joint_dit_classifier"}, 
                {"search_profile": "dit_classifier_runner", "transfer_manifest": {}}, 
                {"search_profile": "dit_classifier_runner", "pruning": {"monitor": "val_classifier_accuracy"}}
            ):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    runner.make_plan(self.fixture.checkout, self.fixture.root / "invalid", **options)
        self.fixture.remote.inspect_remote.assert_not_called()

    def test_vector_progress_front_and_extrema_exclude_dominated_or_invalid_scores(self) -> None:
        """Progress counts both finite metrics while reporting tradeoffs without scalar ranking."""

        self.fixture.trials.extend([
            _ParetoTrial(0, [0.9, 0.05]), _ParetoTrial(1, [0.8, 0.03]), 
            _ParetoTrial(2, [0.7, 0.06]), _ParetoTrial(3, [float("nan"), 0.01]), 
            _ParetoTrial(4, [0.99, 0.001], "FAIL"), _ParetoTrial(5, [0.8, float("inf")])
        ])
        self.fixture.trials[1].params["classifier_route"] = "generator_cross_late"
        summary = runner.search_summary(self.plan)
        self.assertEqual(summary["completed_finite_trials"], 3)
        self.assertEqual(summary["pareto_trial_numbers"], [0, 1])
        self.assertEqual(summary["pareto_front_size"], 2)
        self.assertEqual(summary["max_validation_accuracy"], 0.9)
        self.assertEqual(summary["min_validation_noise_loss"], 0.03)
        self.assertEqual(summary["architecture_counts"], {"plain": 2, "generator_cross_late": 1})
        self.assertNotIn("best_trial", summary)
        self.assertNotIn("best_validation_accuracy", summary)

    def test_progress_reports_each_objective_improvement_separately(self) -> None:
        """Recent extrema do not imply scalarized Pareto convergence."""

        self.fixture.trials.extend([
            _ParetoTrial(number, [0.6, 0.1] if number < 10 else [0.8, 0.04])
            for number in range(60)
        ])
        summary = runner.search_summary(self.plan)
        self.assertAlmostEqual(summary["accuracy_improvement_last_50_valid_trials"], 0.2)
        self.assertAlmostEqual(summary["noise_loss_improvement_last_50_valid_trials"], 0.06)
        self.assertNotIn("improvement_last_50_valid_trials", summary)

    def test_scalar_recipe_cannot_be_reinterpreted_as_pareto(self) -> None:
        """Old classifier plans fail their objective contract before vector ranking."""

        changed = deepcopy(self.plan)
        changed["hpo"]["objective_metrics"] = ["classification_accuracy"]
        changed["hpo"]["objective_directions"] = ["maximize"]
        with self.assertRaisesRegex(ValueError, "noise_loss minimization"):
            runner.search_summary(changed)

    def test_worker_forwards_both_objectives_and_device_deadline_controls(self) -> None:
        """The existing isolated worker receives the full Pareto recipe unchanged."""

        plan = deepcopy(self.plan)
        plan["hpo"].update({"concurrent_trials": 3, "worker_gpu_ids": ["GPU-a", "GPU-b", "GPU-c"]})
        context = Mock()
        remote = SimpleNamespace(
            managed_worker=Mock(), managed_parallel_coordinator=Mock(return_value=nullcontext(context))
        )
        hpo = Mock()
        request = self.fixture.root / "classifier-request.json"
        receipt = self.fixture.root / "classifier-result.json"
        runner._write(request, {
            "plan": plan, "payload": {"kind": "search", "allocated_target": 50}, 
            "receipt_path": str(receipt), "deadline": 1500.0
        })
        with patch.dict(sys.modules, {
            "common.hpo": SimpleNamespace(run_hpo=hpo), "common.dit_hpo_remote": remote
        }), patch.object(runner.time, "time", return_value=1000.0):
            runner._worker(request)
        hpo.assert_called_once_with(**{
            **plan["hpo"], "n_trials": 50, "gpu_worker_context": context, 
            "timeout": 440.0, "stop_active_on_timeout": True
        })

    def test_freeze_keeps_extremes_and_middle_tradeoff_without_scalarization(self) -> None:
        """Representative selection excludes dominated trials and records coverage ordering."""

        for number in range(5):
            self._candidate(number, 0.95 - 0.05 * number, 0.05 - 0.01 * number)
        self._candidate(5, 0.5, 0.1)
        manifest = runner.freeze_finalists(self.plan, [101, 202], top_k=3)
        self.assertEqual([row["trial_number"] for row in manifest["candidates"]], [0, 2, 4])
        self.assertEqual(manifest["distinct_pareto_configurations"], 5)
        self.assertFalse(manifest["selection_policy"]["order_is_fitness_rank"])
        self.assertEqual(manifest["candidates"][0]["display_order"], 1)
        self.assertNotIn("rank", manifest["candidates"][0])
        self.assertEqual(runner.freeze_finalists(self.plan, [101, 202], top_k=3), manifest)

    def test_small_front_allows_fewer_finalists_and_single_slot_is_explicit(self) -> None:
        """TOP_K is a cap for classifier studies and one slot selects maximum accuracy."""

        first = self._candidate(0, 0.9, 0.05)
        second = self._candidate(1, 0.8, 0.03)
        self.assertEqual(runner._pareto_subset([first, second], 1), [first])
        manifest = runner.freeze_finalists(self.plan, [101], top_k=3)
        self.assertEqual(manifest["selected_candidates"], 2)
        self.assertEqual(manifest["top_k"], 3)

    def test_paired_confirmation_summaries_include_both_metrics_and_resume(self) -> None:
        """Each completed seed authenticates both scores and is skipped on reexecution."""

        self._candidate(0, 0.9, 0.05)
        self._candidate(1, 0.8, 0.03)
        runner.freeze_finalists(self.plan, [101, 202], top_k=3)
        results = [_result(0.8, 0.05), _result(0.9, 0.03), _result(0.7, 0.02), _result(0.8, 0.01)]
        with patch.object(runner, "_launch", side_effect=results) as launch:
            records = runner.run_confirmations(self.plan)
        self.assertEqual(launch.call_count, 4)
        summary = runner.confirmation_summary(self.plan)
        self.assertAlmostEqual(summary[0]["mean_accuracy"], 0.85)
        self.assertAlmostEqual(summary[0]["mean_noise_loss"], 0.04)
        self.assertGreater(summary[0]["std_accuracy"], 0)
        self.assertGreater(summary[0]["std_noise_loss"], 0)
        self.assertTrue(all(row["comparable"] for row in summary))
        with patch.object(runner, "_launch") as launch:
            self.assertEqual(runner.run_confirmations(self.plan), records)
        launch.assert_not_called()

    def test_partial_confirmations_are_not_comparable(self) -> None:
        """A successful sibling survives failure without publishing complete paired means."""

        self._candidate(0, 0.9, 0.05)
        runner.freeze_finalists(self.plan, [101, 202], top_k=3)
        with patch.object(runner, "_launch", side_effect=[_result(), RuntimeError("interrupted")]):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                runner.run_confirmations(self.plan)
        [row] = runner.confirmation_summary(self.plan)
        self.assertEqual(row["completed_seeds"], 1)
        self.assertFalse(row["comparable"])
        self.assertFalse(row["all_seeds_complete"])

    def test_receipt_rejects_missing_nonfinite_swapped_or_stale_scalar_objectives(self) -> None:
        """Neither a missing noise score nor old scalar identity can count as confirmation."""

        self._candidate(7, 0.9, 0.05)
        runner.freeze_finalists(self.plan, [101], top_k=3)
        for changes in (
            {"objectives": [0.8]}, {"objectives": [0.8, float("nan")]}, 
            {"objectives": [True, 0.02]}, {"objective_network": "ema"}, 
            {"objective_metrics": ["noise_loss", "classification_accuracy"]}, 
            {"objective_directions": ["minimize", "maximize"]}
        ):
            with self.subTest(changes=changes), patch.object(runner, "_launch", return_value={**_result(), **changes}):
                with self.assertRaises(ValueError):
                    runner.run_confirmations(self.plan)
        with patch.object(runner, "_launch", return_value={"objective": 0.8}):
            with self.assertRaises(ValueError):
                runner.run_confirmations(self.plan)
        with patch.object(runner, "_launch", return_value=_result()):
            runner.run_confirmations(self.plan)
        completed = Path(self.plan["control_root"]) / "confirmations" / "trial-0007" / "seed-101" / "completed.json"
        record = runner._read(completed)
        record["result"]["objectives"] = [0.8]
        runner._write(completed, record)
        for operation in (runner.run_confirmations, runner.confirmation_summary):
            with self.subTest(operation=operation.__name__), self.assertRaises(ValueError):
                operation(self.plan)


class ClassifierConfirmationTests(unittest.TestCase):
    """Replay fresh paired-seed classifier models and require both raw validation metrics."""

    def setUp(self) -> None:
        """Use public training doubles with the named two-objective raw V1 protocol."""

        self.fixture = generation_confirmation_tests.ConfirmationTests("test_fixed_split_fresh_weights_and_public_pipeline")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        config = self.fixture.config
        config.training.task = "joint"
        config.training.patience = 0
        config.training.fit_kwargs = {"validation_freq": 1}
        config.model.name = "dit_classifier"
        config.model.wrapper_name = "diffusion_classifier"
        config.model.wrapper_kwargs.update({"use_ema": False, "test_network_name": "raw"})
        config.dataset.validation_source = "test"
        config.dataset.validation_ratio = 0.0
        config.hpo.update({
            "search_profile": "dit_classifier_runner", "objective_network": "raw", 
            "objective_metrics": ["classification_accuracy", "noise_loss"], 
            "objective_directions": ["maximize", "minimize"]
        })
        self.fixture.modules["common.train"].report.return_value = {
            "valset_network_eval": {"classifier_accuracy": 0.875, "noise_loss": 0.025}
        }

    def test_both_raw_objectives_test_selection_and_fresh_seed_replay(self) -> None:
        """Source data is retained while fresh weights produce both final feedback metrics."""

        fixture = self.fixture
        result = run_confirmation(fixture.source, fixture.output, 101, fixture.digest)
        model_config = fixture.snapshots["model"]
        self.assertEqual(fixture.snapshots["data"].training.seed, 42)
        self.assertEqual(model_config.training.seed, 101)
        self.assertEqual(model_config.model.kwargs["seed"], 101)
        self.assertEqual(model_config.model.wrapper_kwargs["seed"], 101)
        self.assertEqual(model_config.dataset.validation_source, "test")
        self.assertIsNone(model_config.model.weights_path)
        self.assertIsNone(model_config.continually_learn.resume_from)
        self.assertEqual(result["objectives"], [0.875, 0.025])
        self.assertEqual(result["objective_metrics"], ["classification_accuracy", "noise_loss"])
        self.assertEqual(result["objective_directions"], ["maximize", "minimize"])
        self.assertEqual(result["objective_network"], "raw")
        self.assertNotIn("objective", result)
        self.assertEqual(fixture.saved_configs[-1][0].hpo["objectives"], [0.875, 0.025])
        self.assertEqual(result["early_stopping_patience"], 0)

    def test_scalar_recipe_or_ema_cannot_be_replayed_as_pareto(self) -> None:
        """Frozen identities must select both metrics on the raw V1 wrapper."""

        fixture = self.fixture
        original = deepcopy(fixture.config)
        for key, value in [
            ("objective_metrics", ["classification_accuracy"]), 
            ("objective_directions", ["maximize"]), ("objective_network", "ema")
        ]:
            fixture.config = deepcopy(original)
            fixture.config.hpo[key] = value
            fixture.modules["common.config"].load_config.return_value = fixture.config
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Classifier confirmation"):
                run_confirmation(fixture.source, fixture.output, 101, fixture.digest)
        fixture.modules["common.dataloader"].get_datasets.assert_not_called()

    def test_missing_noise_is_not_optional_diagnostic(self) -> None:
        """A finite accuracy alone cannot complete the two-objective confirmation."""

        fixture = self.fixture
        fixture.modules["common.train"].report.return_value = {
            "valset_network_eval": {"classifier_accuracy": 0.875}
        }
        with self.assertRaises(KeyError):
            run_confirmation(fixture.source, fixture.output, 101, fixture.digest)

    def test_nonfinite_noise_fails_even_with_finite_accuracy(self) -> None:
        """Noise divergence prevents the pair from becoming a completed result."""

        fixture = self.fixture
        fixture.modules["common.train"].report.return_value = {
            "valset_network_eval": {"classifier_accuracy": 0.875, "noise_loss": float("nan")}
        }
        with self.assertRaisesRegex(ValueError, "noise_loss is nonfinite"):
            run_confirmation(fixture.source, fixture.output, 101, fixture.digest)
        self.assertEqual(len(fixture.saved_configs), 1)
        self.assertNotIn("objectives", fixture.saved_configs[0][0].hpo)


# Keep direct focused execution available beside unittest discovery.
if __name__ == "__main__":
    unittest.main()
