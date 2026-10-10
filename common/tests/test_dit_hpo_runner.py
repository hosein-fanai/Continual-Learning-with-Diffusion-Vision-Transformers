"""Budget and recovery regression tests without scientific training."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Iterator
import unittest
from unittest.mock import Mock, patch

from common import dit_hpo_runner as runner


def _trial(number: int, value: float | None = None, state: str = "COMPLETE") -> SimpleNamespace:
    """Construct the public Optuna fields read by the coordinator."""

    return SimpleNamespace(
        number=number, value=value, state=SimpleNamespace(name=state), 
        params={"dit_architecture_grid4": "plain", "learning_rate": 0.001 + number / 10000}
    )


class DitHpoRunnerTests(unittest.TestCase):
    """Exercise real persisted state around a mocked scientific worker."""

    def setUp(self) -> None:
        """Create a temporary recipe and mutable Optuna study facade."""

        temporary = tempfile.TemporaryDirectory(prefix="dit-hpo-runner-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        identity = {"source_sha256": {}, "versions": {}, "python": "test-python", 
                    "worker_policy": {"tf_memory_mib": 12288}}
        self.remote = SimpleNamespace(inspect_remote=Mock(return_value=identity))
        with patch.dict(sys.modules, {"common.dit_hpo_remote": self.remote}):
            self.plan = runner.make_plan(self.checkout, self.root / "results")
        self.trials = []
        self.study = SimpleNamespace(get_trials=lambda deepcopy=False: list(self.trials))
        loader = patch.object(runner, "_load_study", return_value=self.study)
        loader.start()
        self.addCleanup(loader.stop)
        output = patch("builtins.print")
        output.start()
        self.addCleanup(output.stop)

    def _candidate(self, number: int, value: float) -> None:
        """Persist a distinguishable recipe for one completed candidate."""

        self.trials.append(_trial(number, value))
        path = Path(self.plan["study_root"]) / "configs" / f"trial-{number:04d}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"candidate: {number}\n", encoding="utf-8")

    def _freeze_one(self) -> dict:
        """Freeze one candidate and one fresh seed for recovery checks."""

        self._candidate(7, 0.4)
        return runner.freeze_finalists(self.plan, [101], top_k=1)

    def test_recipe_restart_is_immutable(self) -> None:
        """A restart retains defaults and rejects changed scientific settings."""

        settings = self.plan["hpo"]
        self.assertEqual(settings["n_startup_trials"], 40)
        self.assertEqual(settings["epochs"], 50)
        self.assertEqual(settings["seed"], 42)
        self.assertEqual(settings["task"], "generation")
        self.assertEqual(settings["model_name"], "diffusion_transformer")
        with patch.dict(sys.modules, {"common.dit_hpo_remote": self.remote}):
            self.assertEqual(runner.make_plan(self.checkout, self.root / "results")["hpo"], settings)
            with self.assertRaisesRegex(ValueError, "recipe changed"):
                runner.make_plan(self.checkout, self.root / "results", epochs=51)

    def test_validation_protocol_defaults_and_test_selection_are_sealed(self) -> None:
        """The default holdout remains compatible while official-test plans seal both controls."""

        self.assertEqual(self.plan["hpo"]["validation_source"], "split")
        self.assertEqual(self.plan["hpo"]["validation_ratio"], 0.2)
        with patch.dict(sys.modules, {"common.dit_hpo_remote": self.remote}):
            plan = runner.make_plan(
                self.checkout, self.root / "test-selection", validation_source="test", validation_ratio=0.0
            )
            self.assertEqual(plan["hpo"]["validation_source"], "test")
            self.assertEqual(plan["hpo"]["validation_ratio"], 0.0)
            for options in ({"validation_source": "split", "validation_ratio": 0.2}, {"validation_source": "test", "validation_ratio": 0.1}):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, "recipe changed"):
                    runner.make_plan(self.checkout, self.root / "test-selection", **options)

    def test_persistent_clock_starts_at_execution_and_cannot_be_reset(self) -> None:
        """Setup does not spend time; restarting retains the original phase cutoffs."""

        with patch.dict(sys.modules, {"common.dit_hpo_remote": self.remote}), patch.object(runner.time, "time", return_value=1000.0):
            plan = runner.make_plan(
                self.checkout, self.root / "results", experiment_hours=12.0, confirmation_reserve_hours=2.0
            )
            self.assertEqual(runner.budget_summary(plan), {"configured": True, "started": False})
            self.assertFalse((Path(plan["control_root"]) / "budget.json").exists())
            with runner._coordinator(plan):
                cutoff = runner._phase_deadline(plan, "search", start=True)
            self.assertEqual(cutoff, 36940.0)
        with patch.object(runner.time, "time", return_value=2000.0):
            self.assertEqual(runner._phase_deadline(plan, "search", start=True), cutoff)
            self.assertEqual(runner._phase_deadline(plan, "confirmation"), 44140.0)
            self.assertEqual(runner.budget_summary(self.plan)["started_at_unix"], 1000.0)
        changed = {**plan, "time_budget": {"experiment_hours": 24.0, "confirmation_reserve_hours": 2.0}}
        with self.assertRaisesRegex(ValueError, "budget changed"):
            runner.budget_summary(changed)
        recipe = runner._read(Path(plan["control_root"]) / "recipe.json")
        self.assertNotIn("time_budget", recipe)

    def test_explicit_start_includes_preflight_and_retains_phase_deadlines(self) -> None:
        """Time spent before search consumes the same durable ten-hour budget."""

        plan = {**self.plan, "time_budget": {"experiment_hours": 10.0, "confirmation_reserve_hours": 2.0}}
        with patch.object(runner.time, "time", return_value=1000.0), patch.object(runner, "_coordinator", wraps=runner._coordinator) as coordinator:
            started = runner.start_experiment(plan)
        coordinator.assert_called_once_with(plan)
        self.assertEqual(started["started_at_unix"], 1000.0)
        self.assertEqual(started["deadline_unix"], 37000.0)
        self.assertEqual(started["search_deadline_unix"] - started["cleanup_seconds"], 29740.0)
        self.assertFalse((Path(plan["study_root"]) / "study.db").exists())
        with patch.object(runner.time, "time", return_value=4600.0):
            resumed = runner.start_experiment(plan)
            self.assertEqual(resumed["started_at_unix"], 1000.0)
            self.assertEqual(resumed["remaining_seconds"], 32400.0)
            self.assertEqual(runner._phase_deadline(plan, "confirmation"), 36940.0)
            with patch.object(runner, "_launch", side_effect=TimeoutError("deadline")) as launch:
                runner.run_search(plan, target_completed=1)
        self.assertEqual(launch.call_args.kwargs["deadline"], 29740.0)

    def test_explicit_start_cannot_extend_expired_clock_or_launch_search(self) -> None:
        """An expired preflight budget remains expired after notebook restart."""

        plan = {**self.plan, "time_budget": {"experiment_hours": 10.0, "confirmation_reserve_hours": 2.0}}
        with patch.object(runner.time, "time", return_value=1000.0):
            runner.start_experiment(plan)
        with patch.object(runner.time, "time", return_value=37000.0), patch.object(runner, "_launch") as launch:
            resumed = runner.start_experiment(plan)
            summary = runner.run_search(plan, target_completed=1)
        self.assertTrue(resumed["time_budget_exhausted"])
        self.assertTrue(summary["time_budget_exhausted"])
        self.assertEqual(resumed["started_at_unix"], 1000.0)
        launch.assert_not_called()
        changed = {**plan, "time_budget": {"experiment_hours": 20.0, "confirmation_reserve_hours": 2.0}}
        with self.assertRaisesRegex(ValueError, "budget changed"):
            runner.start_experiment(changed)

    def test_explicit_start_preserves_untimed_behavior(self) -> None:
        """The optional entry point does not invent a limit for untimed callers."""

        self.assertEqual(runner.start_experiment(self.plan), {"configured": False, "started": False})
        self.assertFalse((Path(self.plan["control_root"]) / "budget.json").exists())

    def test_search_budget_above_300_and_timed_stop_preserve_progress(self) -> None:
        """A large count ceiling is legal and deadline expiry returns authenticated study progress."""

        plan = {**self.plan, "time_budget": {"experiment_hours": 12.0, "confirmation_reserve_hours": 2.0}}
        with patch.object(runner.time, "time", return_value=1000.0), patch.object(runner, "_launch", side_effect=TimeoutError("deadline")) as launch:
            summary = runner.run_search(plan, target_completed=1000, max_attempts=5000, batch_trials=100)
        self.assertTrue(summary["time_budget_exhausted"])
        self.assertFalse(summary["target_reached"])
        self.assertEqual(launch.call_args.kwargs["deadline"], 36940.0)
        self.assertEqual(launch.call_args.args[1]["allocated_target"], 100)
        with patch.object(runner.time, "time", return_value=37000.0), patch.object(runner, "_launch") as launch:
            repeated = runner.run_search(plan, target_completed=1050, max_attempts=5000)
        launch.assert_not_called()
        self.assertTrue(repeated["time_budget_exhausted"])
        self.assertEqual(repeated["time_budget"]["started_at_unix"], 1000.0)

    def test_untimed_search_does_not_hide_unexpected_timeout_errors(self) -> None:
        """Only configured deadlines normalize timeouts into a completed search stage."""

        with patch.object(runner, "_launch", side_effect=TimeoutError("unexpected")):
            with self.assertRaisesRegex(TimeoutError, "unexpected"):
                runner.run_search(self.plan, target_completed=1)

    def test_worker_forwards_test_protocol_and_remaining_hard_timeout(self) -> None:
        """Public HPO receives the selected data protocol and a reduced cleanup-aware timeout."""

        plan = copy.deepcopy(self.plan)
        plan["hpo"].update({"validation_source": "test", "validation_ratio": 0.0, "concurrent_trials": 2})
        factory = Mock()
        remote = SimpleNamespace(managed_worker=Mock(), managed_parallel_coordinator=Mock(return_value=nullcontext(factory)))
        hpo = Mock()
        request = self.root / "timed-request.json"
        receipt = self.root / "timed-result.json"
        runner._write(request, {
            "plan": plan, "payload": {"kind": "search", "allocated_target": 1000}, 
            "receipt_path": str(receipt), "deadline": 1500.0
        })
        with patch.dict(sys.modules, {"common.hpo": SimpleNamespace(run_hpo=hpo), "common.dit_hpo_remote": remote}), patch.object(runner.time, "time", return_value=1000.0):
            runner._worker(request)
        hpo.assert_called_once_with(**{
            **plan["hpo"], "n_trials": 1000, "worker_context": factory, 
            "timeout": 440.0, "stop_active_on_timeout": True
        })
        self.assertFalse(runner._read(receipt)["result"]["time_budget_exhausted"])

    def test_deadline_bounds_child_wait_and_releases_admission(self) -> None:
        """A hung child times out inside the admitted context and cannot produce success."""

        process = SimpleNamespace(wait=Mock(side_effect=runner.TimeoutExpired("worker", 50.0)))
        released = []

        @contextmanager
        def admitted(*args: object, **kwargs: object) -> Iterator[SimpleNamespace]:
            """Record deadline propagation and guaranteed context cleanup."""

            self.assertEqual(kwargs["deadline"], 1050.0)
            try:
                yield process
            finally:
                released.append(True)

        remote = SimpleNamespace(launch_worker=admitted)
        with patch.dict(sys.modules, {"common.dit_hpo_remote": remote}), patch.object(runner.time, "time", return_value=1000.0):
            with self.assertRaisesRegex(TimeoutError, "during worker execution"):
                runner._launch(self.plan, {"kind": "search"}, "bounded", deadline=1050.0)
        process.wait.assert_called_once_with(timeout=50.0)
        self.assertEqual(released, [True])
        request = runner._read(Path(self.plan["control_root"]) / "jobs" / "bounded-001.json")
        self.assertEqual(request["deadline"], 1050.0)


    def test_success_budget_excludes_failures_and_nonfinite_scores(self) -> None:
        """Only finite COMPLETE scores fill the target; all trials consume attempts."""

        self.trials.extend([
            _trial(0, 0.8), _trial(1, float("nan")), 
            _trial(2, state="FAIL"), _trial(3, state="PRUNED")
        ])
        targets = []
        outcomes = iter([("FAIL", None), ("COMPLETE", 0.5), ("COMPLETE", 0.3)])
        original = copy.deepcopy(self.plan)

        def launch(plan: dict, payload: dict, tag: str) -> dict:
            """Allocate a bounded batch with scripted terminal outcomes."""

            self.assertEqual(plan, original)
            self.assertTrue(tag.startswith("search-"))
            target = payload["allocated_target"]
            targets.append(target)
            while len(self.trials) < target:
                state, value = next(outcomes)
                self.trials.append(_trial(len(self.trials), value, state))
            return {}

        with patch.object(runner, "_launch", side_effect=launch):
            summary = runner.run_search(self.plan, target_completed=3, max_attempts=8, batch_trials=2)
        self.assertEqual(targets, [6, 7])
        self.assertEqual(summary["allocated_trials"], 7)
        self.assertEqual(summary["completed_finite_trials"], 3)
        self.assertEqual(summary["states"], {"COMPLETE": 4, "FAIL": 2, "PRUNED": 1})
        self.assertTrue(summary["target_reached"])
        self.assertEqual(self.plan, original)

    def test_attempt_ceiling_idempotence_and_no_progress(self) -> None:
        """Spent budgets stop, completed cells do no work and stuck workers fail."""

        self.trials.extend([_trial(0, 0.6), _trial(1, state="FAIL"), _trial(2, state="PRUNED")])
        with patch.object(runner, "_launch") as launch:
            summary = runner.run_search(self.plan, target_completed=2, max_attempts=3)
            self.assertFalse(summary["target_reached"])
            self.assertTrue(runner.run_search(self.plan, target_completed=1)["target_reached"])
        launch.assert_not_called()
        self.trials.clear()
        with patch.object(runner, "_launch", return_value={}) as launch:
            with self.assertRaisesRegex(RuntimeError, "no observable"):
                runner.run_search(self.plan, target_completed=1)
        self.assertEqual(launch.call_count, 1)

    def test_finalist_copies_are_stable_and_detect_tampering(self) -> None:
        """Finite losses rank ties by trial number and frozen copies retain identity."""

        self._candidate(5, 0.3)
        self._candidate(2, 0.1)
        self._candidate(1, 0.1)
        self.trials.extend([_trial(0, float("inf")), _trial(4, 0.0, "FAIL")])
        manifest = runner.freeze_finalists(self.plan, [101, 202, 303])
        self.assertEqual([item["trial_number"] for item in manifest["candidates"]], [1, 2, 5])
        self.assertEqual(manifest["dataset_seed"], 42)
        candidate = manifest["candidates"][0]
        frozen = Path(candidate["input_config_path"])
        expected = frozen.read_bytes()
        self.assertEqual(hashlib.sha256(expected).hexdigest(), candidate["config_sha256"])
        original = Path(self.plan["study_root"]) / "configs" / frozen.name
        original.write_text("changed source\n", encoding="utf-8")
        self.assertEqual(runner.freeze_finalists(self.plan, [101, 202, 303]), manifest)
        self.assertEqual(frozen.read_bytes(), expected)
        with self.assertRaisesRegex(ValueError, "different seeds"):
            runner.freeze_finalists(self.plan, [404, 505, 606])
        frozen.write_text("tampered frozen input\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "configuration has changed"):
            runner.freeze_finalists(self.plan, [101, 202, 303])

    def test_duplicate_configurations_do_not_fill_finalist_slots(self) -> None:
        """Repeated sampled settings contribute only their best trial to selection."""

        for number, value in enumerate([0.1, 0.11, 0.2, 0.3]):
            self._candidate(number, value)
        self.trials[1].params = dict(self.trials[0].params)
        manifest = runner.freeze_finalists(self.plan, [101])
        self.assertEqual([row["trial_number"] for row in manifest["candidates"]], [0, 2, 3])

    def test_finalists_require_fresh_seeds_and_finished_trials(self) -> None:
        """Invalid seed designs and pending trials cannot freeze a selection."""

        self._candidate(0, 0.1)
        for seeds in ([42], [101, 101], [True], []):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                runner.freeze_finalists(self.plan, seeds, top_k=1)
        for state in ("RUNNING", "WAITING"):
            self.trials[:] = [self.trials[0], _trial(1, state=state)]
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "pending trials"):
                runner.freeze_finalists(self.plan, [101], top_k=1)
        self.assertFalse((Path(self.plan["control_root"]) / "finalists.json").exists())

    def test_frozen_selection_prevents_search_extension(self) -> None:
        """Executing an extension cell cannot change a confirmation selection."""

        self._freeze_one()
        with patch.object(runner, "_launch") as launch:
            with self.assertRaisesRegex(ValueError, "Finalists are frozen"):
                runner.run_search(self.plan, target_completed=2)
            self.assertTrue(runner.run_search(self.plan, target_completed=1)["target_reached"])
        launch.assert_not_called()

    def test_waiting_finalists_require_a_started_elapsed_search_clock(self) -> None:
        """A status flag or an unstarted budget cannot discard queued suggestions."""

        self._candidate(0, 0.1)
        self.trials.append(_trial(1, state="WAITING"))
        plan = {**self.plan, "time_budget": {"experiment_hours": 12.0, "confirmation_reserve_hours": 2.0}}
        control = Path(plan["control_root"])
        runner._write(control / "search_status.json", {"time_budget_exhausted": True})
        with patch.object(runner.time, "time", return_value=1000.0):
            with self.assertRaisesRegex(ValueError, "pending trials"):
                runner.freeze_finalists(plan, [101], top_k=1)
            self.assertFalse((control / "budget.json").exists())
            with runner._coordinator(plan):
                runner._phase_deadline(plan, "search", start=True)
        original_budget = (control / "budget.json").read_bytes()
        # At one second before the effective search cutoff, hints remain runnable.
        with patch.object(runner.time, "time", return_value=36879.0):
            with self.assertRaisesRegex(ValueError, "pending trials"):
                runner.freeze_finalists(plan, [101], top_k=1)
        self.assertFalse((control / "finalists.json").exists())
        self.assertEqual((control / "budget.json").read_bytes(), original_budget)
        self.assertEqual([trial.state.name for trial in self.trials], ["COMPLETE", "WAITING"])

    def test_elapsed_search_preserves_waiting_hints_and_allows_confirmations(self) -> None:
        """Queued hints survive the cutoff while selection and confirmation use completed trials."""

        self._candidate(0, 0.1)
        self.trials.extend([_trial(1, state="WAITING"), _trial(2, state="WAITING")])
        self.trials[1].user_attrs = {"initial_trial": {"sha256": "hint-list", "index": 0}}
        original_trials = copy.deepcopy(self.trials)
        plan = {**self.plan, "time_budget": {"experiment_hours": 12.0, "confirmation_reserve_hours": 2.0}}
        control = Path(plan["control_root"])
        with patch.object(runner.time, "time", return_value=1000.0), runner._coordinator(plan):
            runner._phase_deadline(plan, "search", start=True)
        original_budget = (control / "budget.json").read_bytes()
        # The same cleanup-aware cutoff prevents search replay from launching work.
        with patch.object(runner.time, "time", return_value=36880.0):
            manifest = runner.freeze_finalists(plan, [101], top_k=1)
            self.assertEqual(manifest["unstarted_trials_at_search_deadline"], {
                "trial_numbers": [1, 2], "search_execution_deadline_unix": 36880.0
            })
            self.assertEqual([item["trial_number"] for item in manifest["candidates"]], [0])
            self.assertEqual(manifest["study_allocated_trials"], 3)
            frozen_bytes = (control / "finalists.json").read_bytes()
            with patch.object(runner, "_launch") as launch:
                status = runner.run_search(plan, target_completed=200, max_attempts=5000)
            launch.assert_not_called()
            self.assertTrue(status["time_budget_exhausted"])
            self.assertEqual(status["states"], {"COMPLETE": 1, "WAITING": 2})
            with patch.object(runner, "_launch", return_value={"objective": 0.2}) as launch:
                self.assertEqual(len(runner.run_confirmations(plan)), 1)
            self.assertEqual(launch.call_args.args[1]["kind"], "confirmation")
            self.assertEqual(runner.freeze_finalists(plan, [101], top_k=1), manifest)
        self.assertEqual((control / "finalists.json").read_bytes(), frozen_bytes)
        self.assertEqual((control / "budget.json").read_bytes(), original_budget)
        self.assertEqual(self.trials, original_trials)

    def test_elapsed_search_never_freezes_with_running_trials(self) -> None:
        """A running worker remains a blocker even when queued suggestions cannot start."""

        self._candidate(0, 0.1)
        completed = self.trials[0]
        plan = {**self.plan, "time_budget": {"experiment_hours": 12.0, "confirmation_reserve_hours": 2.0}}
        with patch.object(runner.time, "time", return_value=1000.0), runner._coordinator(plan):
            runner._phase_deadline(plan, "search", start=True)
        for include_waiting in (False, True):
            self.trials[:] = [completed, _trial(1, state="RUNNING")]
            # Mixed pending states must not let the unstarted-hint exception win.
            if include_waiting:
                self.trials.append(_trial(2, state="WAITING"))
            original_trials = copy.deepcopy(self.trials)
            with self.subTest(include_waiting=include_waiting), patch.object(runner.time, "time", return_value=37000.0):
                with self.assertRaisesRegex(ValueError, "pending trials"):
                    runner.freeze_finalists(plan, [101], top_k=1)
            self.assertEqual(self.trials, original_trials)
        self.assertFalse((Path(plan["control_root"]) / "finalists.json").exists())

    def test_expired_frozen_search_replays_status_without_extending_finalists(self) -> None:
        """Run-all after a timed stop reaches confirmation recovery without reopening selection."""

        self._freeze_one()
        path = Path(self.plan["control_root"]) / "finalists.json"
        original = path.read_bytes()
        plan = {**self.plan, "time_budget": {"experiment_hours": 12.0, "confirmation_reserve_hours": 2.0}}
        with patch.object(runner.time, "time", return_value=1000.0), runner._coordinator(plan):
            runner._phase_deadline(plan, "search", start=True)
        with patch.object(runner.time, "time", return_value=37000.0), patch.object(runner, "_launch") as launch:
            for target in (200, 1000, 1050):
                status = runner.run_search(plan, target_completed=target, max_attempts=5000)
                self.assertTrue(status["time_budget_exhausted"])
                self.assertFalse(status["target_reached"])
        launch.assert_not_called()
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(len(self.trials), 1)

    def test_paired_confirmations_resume_and_report_variability(self) -> None:
        """Three finalists cross three fresh seeds exactly once with sample SD."""

        for number in range(3):
            self._candidate(number, 0.1 + number / 10)
        runner.freeze_finalists(self.plan, [101, 202, 303])
        losses = [0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.6, 0.7, 0.8]
        with patch.object(runner, "_launch", side_effect=[{"objective": value} for value in losses]) as launch:
            records = runner.run_confirmations(self.plan)
        self.assertEqual(len(records), 9)
        self.assertEqual(launch.call_count, 9)
        self.assertEqual([call.args[1]["training_seed"] for call in launch.call_args_list], [101, 202, 303] * 3)
        with patch.object(runner, "_launch") as launch:
            self.assertEqual(runner.run_confirmations(self.plan), records)
        launch.assert_not_called()
        summary = runner.confirmation_summary(self.plan)
        self.assertTrue(all(row["all_seeds_complete"] for row in summary))
        for row, mean in zip(summary, [0.3, 0.5, 0.7]):
            self.assertEqual(row["completed_seeds"], 3)
            self.assertAlmostEqual(row["mean_noise_loss"], mean)
            self.assertAlmostEqual(row["std_noise_loss"], 0.1)

    def test_interrupted_confirmation_retains_artifacts_and_retries(self) -> None:
        """A partial run remains intact while its retry uses a new directory."""

        self._freeze_one()
        attempts = []

        def fail(plan: dict, payload: dict, tag: str) -> dict:
            """Leave the partial artifacts of an interrupted scientific worker."""

            del plan, tag
            output = Path(payload["output_root"])
            output.mkdir(parents=True)
            (output / "partial.txt").write_text("keep checkpoint", encoding="utf-8")
            attempts.append(output)
            raise RuntimeError("worker interrupted")

        with patch.object(runner, "_launch", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "worker interrupted"):
                runner.run_confirmations(self.plan)
        with patch.object(runner, "_launch", return_value={"objective": 0.2}) as launch:
            records = runner.run_confirmations(self.plan)
        self.assertEqual(len(records), 1)
        self.assertEqual(Path(launch.call_args.args[1]["output_root"]).name, "attempt-002")
        self.assertEqual(attempts[0].name, "attempt-001")
        self.assertEqual((attempts[0] / "partial.txt").read_text(), "keep checkpoint")

    def test_receipts_authenticate_identity_and_finite_scores(self) -> None:
        """A saved result cannot be reused for another seed or an invalid score."""

        self._freeze_one()
        with patch.object(runner, "_launch", return_value={"objective": 0.2}):
            records = runner.run_confirmations(self.plan)
        path = Path(self.plan["control_root"]) / "confirmations" / "trial-0007" / "seed-101" / "completed.json"
        for mutation, message in (("identity", "different inputs"), ("objective", "nonfinite")):
            tampered = copy.deepcopy(records[0])
            # Change identity and numeric validity independently.
            if mutation == "identity":
                tampered["identity"]["training_seed"] = 202
            # Nonfinite text is valid JSON but not a valid scientific score.
            else:
                tampered["result"]["objective"] = "nan"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            for operation in (runner.run_confirmations, runner.confirmation_summary):
                with self.subTest(mutation=mutation, operation=operation.__name__), patch.object(runner, "_launch") as launch:
                    with self.assertRaisesRegex(ValueError, message):
                        operation(self.plan)
                    launch.assert_not_called()

    def test_nonfinite_confirmation_never_publishes_success(self) -> None:
        """Invalid worker scores remain retryable instead of filling the seed budget."""

        self._freeze_one()
        with patch.object(runner, "_launch", return_value={"objective": float("nan")}):
            with self.assertRaisesRegex(ValueError, "not finite"):
                runner.run_confirmations(self.plan)
        path = Path(self.plan["control_root"]) / "confirmations" / "trial-0007" / "seed-101" / "completed.json"
        self.assertFalse(path.exists())

    def test_execution_controls_do_not_change_scientific_recipe(self) -> None:
        """Changing concurrency preserves one recipe and forwards the verified cap."""

        with patch.dict(sys.modules, {"common.dit_hpo_remote": self.remote}):
            parallel = runner.make_plan(self.checkout, self.root / "results", concurrent_trials=2)
            self.assertEqual(parallel["hpo"]["concurrent_trials"], 2)
            self.assertEqual(parallel["hpo"]["worker_gpu_memory_limit_mb"], 12288)
            self.remote.inspect_remote.assert_called_with(self.checkout, concurrent_trials=2)
            serial = runner.make_plan(self.checkout, self.root / "results", concurrent_trials=1)
            self.assertNotIn("worker_gpu_memory_limit_mb", serial["hpo"])
        recipe = json.loads((Path(parallel["control_root"]) / "recipe.json").read_text())
        self.assertNotIn("concurrent_trials", recipe["hpo"])
        self.assertNotIn("worker_gpu_memory_limit_mb", recipe["hpo"])

    def test_explicit_worker_memory_forwards_runtime_budget_without_changing_recipe(self) -> None:
        """An explicit larger serial-worker cap retains the existing scientific recipe."""

        path = Path(self.plan["control_root"]) / "recipe.json"
        before = path.read_bytes()
        self.remote.inspect_remote.return_value["worker_policy"]["tf_memory_mib"] = 24576
        with patch.dict(sys.modules, {"common.dit_hpo_remote": self.remote}):
            plan = runner.make_plan(
                self.checkout, self.root / "results", worker_gpu_memory_limit_mb=24576
            )
        self.remote.inspect_remote.assert_called_with(
            self.checkout, concurrent_trials=1, worker_gpu_memory_limit_mb=24576
        )
        self.assertEqual(plan["hpo"]["worker_gpu_memory_limit_mb"], 24576)
        self.assertEqual(path.read_bytes(), before)
        self.assertNotIn("worker_gpu_memory_limit_mb", runner._read(path)["hpo"])

    def test_pruning_policy_is_copied_and_sealed_for_recovery(self) -> None:
        """Changing selection rules cannot silently mix scientific study recipes."""

        policy = {"type": "percentile", "percentile": 75.0}
        with patch.dict(sys.modules, {"common.dit_hpo_remote": self.remote}):
            plan = runner.make_plan(self.checkout, self.root / "pruned", pruning=policy)
            self.assertEqual(plan["hpo"]["pruning"], policy)
            self.assertEqual(plan["hpo"]["worker_gpu_memory_limit_mb"], 12288)
            policy["percentile"] = 50.0
            self.assertEqual(plan["hpo"]["pruning"]["percentile"], 75.0)
            with self.assertRaisesRegex(ValueError, "recipe changed"):
                runner.make_plan(self.checkout, self.root / "pruned", pruning=policy)
        recipe = json.loads((Path(plan["control_root"]) / "recipe.json").read_text())
        self.assertEqual(recipe["hpo"]["pruning"]["percentile"], 75.0)

    def test_search_space_overrides_are_deep_copied_and_sealed(self) -> None:
        """Nested candidate choices cannot mutate or silently replace an existing recipe."""

        options = {"dim": [16, 32], "time_freq_dim": [None, 1, 2]}
        original = copy.deepcopy(options)
        with patch.dict(sys.modules, {"common.dit_hpo_remote": self.remote}):
            plan = runner.make_plan(
                self.checkout, self.root / "expanded", search_space_overrides=options
            )
            options["dim"].append(64)
            options["time_freq_dim"][0] = 4
            self.assertEqual(plan["hpo"]["search_space_overrides"], original)
            repeated = runner.make_plan(
                self.checkout, self.root / "expanded", search_space_overrides=original
            )
            self.assertEqual(repeated["hpo"], plan["hpo"])
            for changed in (options, None):
                with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "recipe changed"):
                    runner.make_plan(
                        self.checkout, self.root / "expanded", search_space_overrides=changed
                    )
        recipe = runner._read(Path(plan["control_root"]) / "recipe.json")
        self.assertEqual(recipe["hpo"]["search_space_overrides"], original)

    def test_omitted_search_space_preserves_existing_recipe(self) -> None:
        """Legacy plans retain their exact public HPO arguments and saved recipe."""

        path = Path(self.plan["control_root"]) / "recipe.json"
        before = path.read_bytes()
        self.assertNotIn("search_space_overrides", self.plan["hpo"])
        with patch.dict(sys.modules, {"common.dit_hpo_remote": self.remote}):
            repeated = runner.make_plan(
                self.checkout, self.root / "results", search_space_overrides=None
            )
        self.assertEqual(repeated["hpo"], self.plan["hpo"])
        self.assertEqual(path.read_bytes(), before)

    def test_search_workers_forward_public_search_space_overrides(self) -> None:
        """Serial and parallel dispatch preserve the explicit candidate distributions."""

        options = {"dim": [16, 32], "use_cfg": [True, False], "time_freq_dim": [None, 1]}
        for concurrent in (1, 2):
            with self.subTest(concurrent=concurrent):
                with patch.dict(sys.modules, {"common.dit_hpo_remote": self.remote}):
                    plan = runner.make_plan(
                        self.checkout, self.root / "expanded-worker", concurrent_trials=concurrent, 
                        search_space_overrides=options
                    )
                factory = Mock()
                hpo = Mock()
                remote = SimpleNamespace(
                    managed_worker=Mock(return_value=nullcontext()), 
                    managed_parallel_coordinator=Mock(return_value=nullcontext(factory))
                )
                request = self.root / "expanded-request.json"
                receipt = self.root / "expanded-result.json"
                runner._write(request, {
                    "plan": plan, "payload": {"kind": "search", "allocated_target": 7}, 
                    "receipt_path": str(receipt)
                })
                with patch.dict(sys.modules, {"common.hpo": SimpleNamespace(run_hpo=hpo), "common.dit_hpo_remote": remote}):
                    runner._worker(request)
                expected = {**plan["hpo"], "n_trials": 7}
                # Only parallel dispatch adds a shared worker resource context.
                if concurrent > 1:
                    expected["worker_context"] = factory
                hpo.assert_called_once_with(**expected)
                self.assertEqual(hpo.call_args.kwargs["search_space_overrides"], options)

    def test_single_worker_pruning_keeps_optuna_in_cpu_coordinator(self) -> None:
        """A pruned single-GPU search uses the existing isolated worker scheduler."""

        with patch.dict(sys.modules, {"common.dit_hpo_remote": self.remote}):
            plan = runner.make_plan(self.checkout, self.root / "pruned", pruning={"type": "percentile"})
        factory = Mock()
        hpo = Mock()
        remote = SimpleNamespace(
            managed_worker=Mock(side_effect=AssertionError("Pruning initialized the coordinator GPU.")), 
            managed_parallel_coordinator=Mock(return_value=nullcontext(factory))
        )
        receipt = self.root / "pruned-result.json"
        request = self.root / "pruned-request.json"
        request.write_text(json.dumps({
            "plan": plan, "payload": {"kind": "search", "allocated_target": 10}, 
            "receipt_path": str(receipt)
        }), encoding="utf-8")
        with patch.dict(sys.modules, {"common.hpo": SimpleNamespace(run_hpo=hpo), "common.dit_hpo_remote": remote}):
            runner._worker(request)
        hpo.assert_called_once_with(**{**plan["hpo"], "n_trials": 10, "worker_context": factory})
        remote.managed_worker.assert_not_called()

    def test_multigpu_confirmation_launch_reserves_only_first_device(self) -> None:
        """Serial confirmation authenticates a reduced runtime without mutating the plan."""

        plan = copy.deepcopy(self.plan)
        plan["identity"]["gpus"] = [{"gpu_id": 1}, {"gpu_id": 0}]
        original = copy.deepcopy(plan)
        serial_identity = {**plan["identity"], "gpus": [{"gpu_id": 1}], "concurrent_trials": 1}

        @contextmanager
        def launch(command: list[str], checkout_root: str, identity: dict, log_path: Path) -> Iterator[SimpleNamespace]:
            """Publish an authenticated result from the exact serialized worker request."""

            self.assertEqual(identity, serial_identity)
            request_path = Path(command[-1])
            request = json.loads(request_path.read_text())
            self.assertEqual(request["plan"]["identity"], serial_identity)
            self.assertEqual(request["plan"]["hpo"], plan["hpo"])
            runner._write(request["receipt_path"], {
                "request_sha256": runner._digest(request_path), "result": {"objective": 0.2}
            })
            yield SimpleNamespace(wait=lambda: 0)

        remote = SimpleNamespace(
            serial_worker_identity=Mock(return_value=serial_identity), launch_worker=launch
        )
        with patch.dict(sys.modules, {"common.dit_hpo_remote": remote}):
            result = runner._launch(plan, {"kind": "confirmation"}, "confirm-device")
        remote.serial_worker_identity.assert_called_once_with(plan["checkout_root"], original["identity"])
        self.assertEqual(result, {"objective": 0.2})
        self.assertEqual(plan, original)

    def test_gpu_selection_is_runtime_only_and_forwards_verified_uuids(self) -> None:
        """Physical selections reach admission and reuse the same scientific recipe."""

        identity = self.remote.inspect_remote.return_value
        for indices in [[0], [1], [0, 1]]:
            with self.subTest(indices=indices):
                identity["gpus"] = [{"gpu_id": index, "gpu_uuid": "GPU-" + str(index)} for index in indices]
                with patch.dict(sys.modules, {"common.dit_hpo_remote": self.remote}):
                    plan = runner.make_plan(self.checkout, self.root / "results", concurrent_trials=5, gpu_ids=indices)
                self.remote.inspect_remote.assert_called_with(self.checkout, concurrent_trials=5, gpu_ids=indices)
                self.assertEqual(plan["hpo"]["worker_gpu_ids"], ["GPU-" + str(index) for index in indices])
                self.assertEqual(plan["hpo"]["worker_gpu_memory_limit_mb"], 12288)
                recipe = json.loads((Path(plan["control_root"]) / "recipe.json").read_text())
                self.assertNotIn("worker_gpu_ids", recipe["hpo"])

    def test_gpu_routed_search_uses_existing_process_api_for_one_or_many_workers(self) -> None:
        """Explicit routing uses a GPU-aware resource factory even for one trial."""

        for concurrent, indices in [(1, [1]), (5, [0]), (10, [0, 1])]:
            with self.subTest(concurrent=concurrent, indices=indices):
                self.remote.inspect_remote.return_value["gpus"] = [
                    {"gpu_id": index, "gpu_uuid": "GPU-" + str(index)} for index in indices
                ]
                with patch.dict(sys.modules, {"common.dit_hpo_remote": self.remote}):
                    plan = runner.make_plan(self.checkout, self.root / "results", concurrent_trials=concurrent, gpu_ids=indices)
                factory = Mock()
                hpo = Mock()
                remote = SimpleNamespace(
                    managed_worker=Mock(side_effect=AssertionError("Routed search initialized the coordinator GPU.")), 
                    managed_parallel_coordinator=Mock(return_value=nullcontext(factory))
                )
                receipt = self.root / "gpu-result.json"
                request = self.root / "gpu-request.json"
                request.write_text(json.dumps({
                    "plan": plan, "payload": {"kind": "search", "allocated_target": 13}, 
                    "receipt_path": str(receipt)
                }), encoding="utf-8")
                with patch.dict(sys.modules, {"common.hpo": SimpleNamespace(run_hpo=hpo), "common.dit_hpo_remote": remote}):
                    runner._worker(request)
                hpo.assert_called_once_with(**{**plan["hpo"], "n_trials": 13, "gpu_worker_context": factory})
                remote.managed_worker.assert_not_called()
                remote.managed_parallel_coordinator.assert_called_once_with(plan["checkout_root"], plan["identity"])

    def test_parallel_search_forwards_existing_scheduler_resource_context(self) -> None:
        """The public HPO scheduler receives the verified five-slot resource factory."""

        with patch.dict(sys.modules, {"common.dit_hpo_remote": self.remote}):
            plan = runner.make_plan(self.checkout, self.root / "results", concurrent_trials=5)
        factory = Mock()
        hpo = Mock()
        remote = SimpleNamespace(
            managed_worker=Mock(side_effect=AssertionError("Parallel coordinator initialized a GPU.")), 
            managed_parallel_coordinator=Mock(return_value=nullcontext(factory))
        )
        receipt = self.root / "parallel-result.json"
        request = self.root / "parallel-request.json"
        request.write_text(json.dumps({
            "plan": plan, "payload": {"kind": "search", "allocated_target": 7}, 
            "receipt_path": str(receipt)
        }), encoding="utf-8")
        with patch.dict(sys.modules, {"common.hpo": SimpleNamespace(run_hpo=hpo), "common.dit_hpo_remote": remote}):
            runner._worker(request)
        hpo.assert_called_once_with(**{**plan["hpo"], "n_trials": 7, "worker_context": factory})
        remote.managed_worker.assert_not_called()
        remote.managed_parallel_coordinator.assert_called_once_with(plan["checkout_root"], plan["identity"])
        self.assertEqual(json.loads(receipt.read_text())["request_sha256"], runner._digest(request))

    def test_confirmation_uses_one_admitted_worker_under_parallel_search_plan(self) -> None:
        """Confirmation keeps its paired training API and does not open another study."""

        plan = copy.deepcopy(self.plan)
        plan["hpo"]["concurrent_trials"] = 5
        plan["hpo"]["worker_gpu_ids"] = ["GPU-first", "GPU-second"]
        remote = SimpleNamespace(
            managed_worker=Mock(return_value=nullcontext()), 
            managed_parallel_coordinator=Mock(side_effect=AssertionError("Confirmation opened an HPO pool."))
        )
        confirmation = Mock(return_value={"objective": 0.2})
        request = self.root / "confirmation-request.json"
        receipt = self.root / "confirmation-result.json"
        request.write_text(json.dumps({
            "plan": plan, "payload": {"kind": "confirmation", "training_seed": 101}, 
            "receipt_path": str(receipt)
        }), encoding="utf-8")
        with patch.dict(sys.modules, {
            "common.dit_hpo_remote": remote, 
            "common.dit_hpo_confirmation": SimpleNamespace(run_confirmation=confirmation)
        }):
            runner._worker(request)
        confirmation.assert_called_once_with(training_seed=101)
        remote.managed_worker.assert_called_once_with(plan["checkout_root"], plan["identity"])
        remote.managed_parallel_coordinator.assert_not_called()

    def test_worker_forwards_public_hpo_recipe_and_resume_identity(self) -> None:
        """Worker dispatch passes the fixed recipe and total target to run_hpo."""

        run_hpo = Mock()
        fake_hpo = SimpleNamespace(run_hpo=run_hpo)
        fake_remote = SimpleNamespace(
            managed_worker=Mock(side_effect=lambda *args: nullcontext()), 
            managed_parallel_coordinator=Mock(side_effect=AssertionError("Serial search used parallel admission."))
        )
        receipt = self.root / "worker-result.json"
        request = self.root / "worker-request.json"
        request.write_text(json.dumps({
            "plan": self.plan, "payload": {"kind": "search", "allocated_target": 200}, 
            "receipt_path": str(receipt)
        }), encoding="utf-8")
        database = Path(self.plan["study_root"]) / "study.db"
        database.touch()
        with patch.dict(sys.modules, {"common.hpo": fake_hpo, "common.dit_hpo_remote": fake_remote}):
            runner._worker(request)
        expected = {**self.plan["hpo"], "n_trials": 200, "resume_from": self.plan["study_root"]}
        run_hpo.assert_called_once_with(**expected)
        fake_remote.managed_worker.assert_called_once_with(self.plan["checkout_root"], self.plan["identity"])
        result = json.loads(receipt.read_text())
        self.assertEqual(result["request_sha256"], hashlib.sha256(request.read_bytes()).hexdigest())
        self.assertEqual(result["result"]["allocated_trials"], 0)


# Keep direct execution equivalent to unittest module discovery.
if __name__ == "__main__":
    unittest.main()

