"""Epoch-report identity, worker cleanup, and finite performance-pruning checks."""

from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from common.hpo_process import answer_pruning_report, finish_worker, read_pruning_report, start_worker, stop_workers
from common.hpo_pruning import (
    TrialPerformancePruned, create_exchange, decision_path, read_report, report_epoch, validate_exchange, write_atomic_json
)
from common.hpo_worker import run_worker


class HpoPruningTransportTests(unittest.TestCase):
    """Keep epoch reports authenticated and Optuna exclusively in the coordinator."""

    def setUp(self) -> None:
        """Give each test private artifacts and automatic child-process cleanup."""

        temporary = tempfile.TemporaryDirectory(prefix="pruning-protocol-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text("{}", encoding="utf-8")
        self.handles = []
        self.addCleanup(stop_workers, self.handles)

    def _start(self, script: str | None = None, enabled: bool = True) -> tuple:
        """Launch either a real protocol-only child or a controlled process double."""

        original_popen = subprocess.Popen
        child = Mock(pid=24680)
        child.poll.return_value = None
        child.wait.return_value = 1

        def launch(command: list[str], **options: object) -> object:
            """Replace only the executable to test the real launch/parent-pipe path."""

            # Real children exercise blocking epoch replies without importing TensorFlow.
            if script is not None:
                return original_popen([command[0], "-u", "-c", script, str(self.root / "output.json")], **options)
            return child

        with patch("common.hpo_process.subprocess.Popen", side_effect=launch) as popen:
            handle = start_worker(
                self.config_path, self.root / "output.json", self.root / "worker.log", 
                gpu_memory_limit_mb=None, pruning_monitor="val_noise_loss" if enabled else None, 
                pruning_trial_number=7 if enabled else None
            )
        self.handles.append(handle)
        return handle, popen

    def _report(self, handle: object, step: int = 0) -> dict:
        """Build an exact finite report for one test worker's launch identity."""

        exchange = handle.pruning_exchange
        return {
            "protocol_version": 1, "token": exchange["token"], "trial_number": 7, 
            "monitor": "val_noise_loss", "step": step, "request_token": "a" * 32, "value": 0.8
        }

    def test_transport_imports_remain_framework_and_optuna_free(self) -> None:
        """Epoch IPC cannot initialize a model or write study storage in a child."""

        result = subprocess.run(
            [sys.executable, "-c", "import sys; import common.hpo_pruning,common.hpo_process,common.hpo_worker; "
             "print(any(key in sys.modules for key in ['tensorflow','keras','optuna']))"], 
            text=True, capture_output=True, check=True, timeout=20
        )
        self.assertEqual(result.stdout.strip(), "False")

    def test_enabled_exchange_is_unique_owned_and_removed_only_after_exit(self) -> None:
        """Each launch gets its own identity, and legacy launches inherit no channel."""

        with patch.dict(os.environ, {"HPO_PRUNING_EXCHANGE": "stale"}):
            handle, popen = self._start()
            exchange = json.loads(popen.call_args.kwargs["env"]["HPO_PRUNING_EXCHANGE"])
            self.assertEqual(exchange, handle.pruning_exchange)
            self.assertEqual(exchange["trial_number"], 7)
            directory = Path(exchange["directory"])
            self.assertTrue(directory.is_dir())
            self.assertIsNone(read_pruning_report(handle))
            self.assertEqual(os.environ["HPO_PRUNING_EXCHANGE"], "stale")
            stop_workers([handle])
            self.assertFalse(directory.exists())
            legacy, legacy_popen = self._start(enabled=False)
            self.assertNotIn("HPO_PRUNING_EXCHANGE", legacy_popen.call_args.kwargs["env"])
            self.assertIsNone(read_pruning_report(legacy))

    def test_reports_reject_wrong_trial_attempt_monitor_and_nonfinite_values(self) -> None:
        """Malformed intermediate observations never reach the Optuna coordinator."""

        handle, _ = self._start()
        path = Path(handle.pruning_exchange["directory"]) / "report.json"
        for field, value in [
            ("token", "b" * 32), ("trial_number", 8), ("trial_number", True), 
            ("monitor", "loss"), ("step", True), ("step", -1), 
            ("request_token", "../bad"), ("value", float("nan")), ("value", [1])
        ]:
            with self.subTest(field=field, value=value):
                report = {**self._report(handle), field: value}
                path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_pruning_report(handle)

    def test_each_report_is_answered_once_and_replayed_epochs_fail(self) -> None:
        """Duplicate scheduler polls are harmless; changed or stale identities fail."""

        handle, _ = self._start()
        path = Path(handle.pruning_exchange["directory"]) / "report.json"
        report = self._report(handle)
        write_atomic_json(path, report)
        self.assertEqual(read_pruning_report(handle), report)
        with self.assertRaisesRegex(ValueError, "does not match"):
            answer_pruning_report(handle, {**report, "value": 0.7}, False)
        answer_pruning_report(handle, report, False)
        self.assertIsNone(read_pruning_report(handle))
        with self.assertRaisesRegex(ValueError, "does not match"):
            answer_pruning_report(handle, report, True)
        write_atomic_json(path, {**report, "request_token": "b" * 32})
        with self.assertRaisesRegex(ValueError, "stale or replayed"):
            read_pruning_report(handle)

    def test_reporter_rejects_a_reply_for_a_different_epoch(self) -> None:
        """An atomic but mismatched coordinator reply cannot stop the wrong epoch."""

        exchange = create_exchange(self.root, "val_noise_loss", 7)

        def mismatched_reply(seconds: float) -> None:
            """Publish a deliberately stale decision at the current response path."""

            report = read_report(exchange)
            write_atomic_json(decision_path(exchange, report), {"report": {**report, "step": 3}, "prune": True})

        with patch("common.hpo_pruning.time.sleep", side_effect=mismatched_reply):
            with self.assertRaisesRegex(ValueError, "does not match"):
                report_epoch(exchange, 0, 1.0)

    def test_real_child_reports_two_epochs_and_retains_pruning_evidence(self) -> None:
        """A real isolated worker blocks for each decision and returns a finite prune."""

        script = (
            "import json,os,sys,threading\n"
            "from pathlib import Path\n"
            "from common.hpo_worker import _watch_parent\n"
            "from common.hpo_pruning import report_epoch\n"
            "ready=threading.Event()\n"
            "threading.Thread(target=_watch_parent,args=(ready,),daemon=True).start()\n"
            "assert ready.wait(5)\n"
            "exchange=json.loads(os.environ['HPO_PRUNING_EXCHANGE'])\n"
            "prune,report=report_epoch(exchange,0,0.9)\n"
            "assert not prune\n"
            "prune,report=report_epoch(exchange,1,0.8)\n"
            "assert prune\n"
            "evidence={'reason':'performance_pruning','epoch':1,'metric':'val_noise_loss','value':0.8,"
            "'report':report,'partial_history':[{'epoch':0,'metrics':{'val_noise_loss':0.9}}]}\n"
            "Path(sys.argv[1]).write_text(json.dumps({'status':'pruned','config_path':'input.yaml',"
            "'error':'performance prune','pruning':evidence,'divergence':None}))\n"
            "sys.exit(1)\n"
        )
        handle, _ = self._start(script)
        seen = []
        deadline = time.monotonic() + 10
        while handle.process.poll() is None and time.monotonic() < deadline:
            report = read_pruning_report(handle)
            # Answer only complete, previously unanswered epoch publications.
            if report is not None:
                seen.append(report["step"])
                answer_pruning_report(handle, report, report["step"] == 1)
            time.sleep(0.01)
        self.assertEqual(handle.process.wait(timeout=5), 1)
        directory = Path(handle.pruning_exchange["directory"])
        result = finish_worker(handle)
        self.assertEqual(seen, [0, 1])
        self.assertEqual(result["pruning"]["value"], 0.8)
        self.assertIsNone(result["divergence"])
        self.assertFalse(directory.exists())

    def test_parent_eof_stops_a_child_waiting_for_pruning(self) -> None:
        """A pending decision cannot orphan a worker when its coordinator dies."""

        script = (
            "import json,os,threading\n"
            "from common.hpo_worker import _watch_parent\n"
            "from common.hpo_pruning import report_epoch\n"
            "ready=threading.Event()\n"
            "threading.Thread(target=_watch_parent,args=(ready,),daemon=True).start()\n"
            "assert ready.wait(5)\n"
            "report_epoch(json.loads(os.environ['HPO_PRUNING_EXCHANGE']),0,1.0)\n"
        )
        handle, _ = self._start(script)
        deadline = time.monotonic() + 10
        while read_pruning_report(handle) is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNotNone(read_pruning_report(handle))
        handle.process.stdin.close()
        self.assertEqual(handle.process.wait(timeout=5), 1)
        self.assertFalse(handle.output_path.exists())

    def test_claimed_pruning_requires_the_parent_decision(self) -> None:
        """A child cannot forge successful pruning by writing an unauthenticated result."""

        handle, _ = self._start()
        handle.process.poll.return_value = 1
        handle.output_path.write_text(json.dumps({
            "status": "pruned", "config_path": str(self.config_path), "error": "claimed prune", 
            "pruning": {"reason": "performance_pruning", "report": self._report(handle), "partial_history": []}
        }), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "matching divergence or performance evidence"):
            finish_worker(handle)

    def test_worker_binds_trial_and_never_saves_runtime_exchange(self) -> None:
        """Launch-only exchange metadata reaches training but not the resolved YAML."""

        exchange = create_exchange(self.root, "val_noise_loss", 7)
        tensorflow = Mock()
        tensorflow.errors.ResourceExhaustedError = MemoryError
        tensorflow.config.list_physical_devices.return_value = []
        config = SimpleNamespace(hpo={"trial_number": 7}, training=SimpleNamespace(results_path=str(self.root)))
        saved = []
        received = []

        def train(value: object) -> dict:
            """Observe the runtime channel while returning ordinary saved metrics."""

            received.append(dict(value.hpo["pruning_exchange"]))
            return {"results_path": str(self.root), "history": {}, "evaluations": {}}

        def save(value: object, path: object) -> None:
            """Observe the persisted scientific configuration after IPC cleanup."""

            saved.append(dict(value.hpo))

        modules = {
            "tensorflow": tensorflow, 
            "common.config": SimpleNamespace(load_config=Mock(return_value=config), save_config=save), 
            "common.train": SimpleNamespace(main=train), 
            "common.callbacks.hpo_guard": SimpleNamespace(TrainingDiverged=FloatingPointError)
        }
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"HPO_PRUNING_EXCHANGE": json.dumps(exchange)}), \
                redirect_stderr(io.StringIO()):
            self.assertEqual(run_worker(self.config_path, self.root / "result.json"), 0)
            self.assertEqual(received, [exchange])
            self.assertEqual(saved, [{"trial_number": 7}])
            config.hpo["trial_number"] = 8
            self.assertEqual(run_worker(self.config_path, self.root / "result.json"), 1)
        self.assertIn("saved trial number", json.loads((self.root / "result.json").read_text())["error"])


class HpoPruningCallbackTests(unittest.TestCase):
    """Exercise finite, missing, structured, and numerically divergent epoch logs."""

    def test_missing_and_structured_monitor_fail_before_reporting(self) -> None:
        """No fallback metric or array can become an intermediate pruning value."""

        from common.callbacks.hpo_pruning import EpochPruningCallback


        with tempfile.TemporaryDirectory() as directory:
            exchange = create_exchange(Path(directory), "val_noise_loss", 1)
            for logs in ({"loss": 0.8}, {"val_noise_loss": [0.8]}, {"val_noise_loss": "0.8"}):
                with self.subTest(logs=logs), patch("common.callbacks.hpo_pruning.report_epoch") as reporter:
                    callback = EpochPruningCallback(exchange)
                    callback.on_train_begin()
                    with self.assertRaisesRegex(ValueError, "finite scalar epoch metric"):
                        callback.on_epoch_end(0, logs)
                    reporter.assert_not_called()

    def test_nonfinite_monitor_preserves_numerical_divergence(self) -> None:
        """NaN and infinity retain the existing divergence evidence and never rank."""

        from common.callbacks.hpo_guard import TrainingDiverged
        from common.callbacks.hpo_pruning import EpochPruningCallback


        with tempfile.TemporaryDirectory() as directory:
            exchange = create_exchange(Path(directory), "val_noise_loss", 1)
            for value in (float("nan"), float("inf")):
                with self.subTest(value=value), patch("common.callbacks.hpo_pruning.report_epoch") as reporter:
                    callback = EpochPruningCallback(exchange)
                    callback.on_train_begin()
                    with self.assertRaises(TrainingDiverged) as context:
                        callback.on_epoch_end(0, {"val_noise_loss": value})
                    self.assertEqual(context.exception.evidence["reason"], "nonfinite_loss")
                    reporter.assert_not_called()

    def test_finite_performance_prune_carries_partial_history(self) -> None:
        """Performance pruning retains the pruning epoch and a distinct reason."""

        from common.callbacks.hpo_pruning import EpochPruningCallback


        with tempfile.TemporaryDirectory() as directory:
            exchange = create_exchange(Path(directory), "val_noise_loss", 1)
            callback = EpochPruningCallback(exchange, evidence_dir=directory)
            callback.on_train_begin()
            with patch("common.callbacks.hpo_pruning.report_epoch", return_value=(False, {})):
                callback.on_epoch_end(0, {"val_noise_loss": 0.8})
            with patch("common.callbacks.hpo_pruning.report_epoch", return_value=(True, {"step": 1})):
                with self.assertRaises(TrialPerformancePruned) as context:
                    callback.on_epoch_end(1, {"val_noise_loss": 0.7})
            evidence = context.exception.evidence
            self.assertEqual(evidence["reason"], "performance_pruning")
            self.assertEqual([row["epoch"] for row in evidence["partial_history"]], [0, 1])
            self.assertEqual(json.loads(context.exception.evidence_path.read_text()), evidence)
            with self.assertRaisesRegex(ValueError, "increase strictly"):
                callback.on_epoch_end(1, {"val_noise_loss": 0.7})


# Execute focused checks only when selected by the remote test process.
if __name__ == "__main__":
    unittest.main()
