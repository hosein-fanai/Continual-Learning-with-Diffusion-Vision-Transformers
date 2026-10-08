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

