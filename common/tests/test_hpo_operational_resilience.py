"""Pruning fairness and bounded publication recovery without real model training."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from common.hpo import run_hpo
from common.hpo_process import finish_worker, stop_workers
from common.hpo_pruning import create_exchange, read_atomic_json, read_report, report_epoch
from common.tests import test_dit_hpo_api as api_tests
from common.tests import test_hpo_pruning_transport as transport_tests


class PruningLaunchFairnessTests(unittest.TestCase):
    """Serve existing epoch handshakes before attempting every remaining launch."""

    def setUp(self) -> None:
        """Reuse the real Optuna fixture while replacing all training processes."""

        self.fixture = api_tests.DitHpoApiTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_pending_epochs_are_answered_before_the_next_launch(self) -> None:
        """Slow preparation of later workers cannot defer every decision until pool fill."""

        workers = api_tests._PruningWorkers({0: [1.0] * 3, 1: [1.0], 2: [1.0]})
        observed = []

        def inspect_next_launch(handle: object) -> None:
            """Require the first worker to advance once between successive launches."""

            # The first process has no preceding worker whose report needs service.
            if handle.number > 0:
                self.assertIn((0, handle.number - 1, False), workers.decisions)
                observed.append(handle.number)

        workers.after_start = inspect_next_launch
        with workers.installed():
            study = run_hpo(**self.fixture.options(n_trials=3, concurrent_trials=3, pruning={}))
        self.assertEqual(observed, [1, 2])
        self.assertEqual([trial.state.name for trial in study.trials], ["COMPLETE"] * 3)
        self.assertFalse(workers.active)

    def test_invalid_early_report_aborts_before_more_workers_launch(self) -> None:
        """Serving during launch retains error recording and reaps the existing child."""

        workers = api_tests._PruningWorkers({0: [1.0], 1: [1.0], 2: [1.0]})
        with workers.installed(), patch("common.hpo.read_pruning_report", side_effect=ValueError("foreign report")):
            with self.assertRaisesRegex(ValueError, "foreign report"):
                run_hpo(**self.fixture.options(n_trials=3, concurrent_trials=3, pruning={}))
        self.assertEqual([handle.number for handle in workers.handles], [0])
        self.assertEqual(workers.stopped, [0])
        self.assertFalse(workers.active)

    def test_disabled_pruning_does_not_poll_epoch_transport(self) -> None:
        """Existing non-pruned scheduling retains its transport-free behavior."""

        workers = api_tests._Workers()
        with workers.installed(), patch("common.hpo.read_pruning_report", side_effect=AssertionError("disabled IPC")):
            study = run_hpo(**self.fixture.options(n_trials=3, concurrent_trials=3, pruning=None))
        self.assertEqual(len(study.trials), 3)
        self.assertEqual(workers.max_active, 3)
        self.assertFalse(workers.active)


class PublicationVisibilityTests(unittest.TestCase):
    """Retry publication visibility only, preserving strict decoded-data validation."""

    def setUp(self) -> None:
        """Create one private protocol directory without workers or framework execution."""

        temporary = tempfile.TemporaryDirectory(prefix="hpo-visibility-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.exchange = create_exchange(self.root, "val_noise_loss", 7)
        self.report = {
            "protocol_version": 1, "token": self.exchange["token"], "trial_number": 7, 
            "monitor": "val_noise_loss", "step": 0, "request_token": "a" * 32, "value": 0.8
        }

    def test_partial_json_and_transient_io_can_resolve(self) -> None:
        """An incomplete publication and transient EIO get bounded read retries."""

        with patch.object(Path, "read_text", side_effect=["{", OSError(errno.EIO, "transient"), '{"ok": true}']) as read, \
                patch("common.hpo_pruning.time.sleep") as sleep:
            self.assertEqual(read_atomic_json(self.root / "value.json"), {"ok": True})
        self.assertEqual(read.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_persistent_malformed_json_is_bounded_and_propagates(self) -> None:
        """A corrupt file remains a failure after twenty visibility retries."""

        with patch.object(Path, "read_text", return_value="{") as read, \
                patch("common.hpo_pruning.time.sleep") as sleep:
            with self.assertRaises(json.JSONDecodeError):
                read_atomic_json(self.root / "value.json")
        self.assertEqual(read.call_count, 21)
        self.assertEqual(sleep.call_count, 20)
        self.assertTrue(all(call.args == tuple([0.05]) for call in sleep.call_args_list))

    def test_missing_terminal_result_retries_but_unpublished_poll_does_not(self) -> None:
        """A missing first epoch must not delay the coordinator's other workers."""

        with patch.object(Path, "read_text", side_effect=FileNotFoundError("not visible")) as read, \
                patch("common.hpo_pruning.time.sleep") as sleep:
            with self.assertRaises(FileNotFoundError):
                read_atomic_json(self.root / "terminal.json")
            self.assertEqual(read.call_count, 21)
            self.assertEqual(sleep.call_count, 20)
            read.reset_mock()
            sleep.reset_mock()
            self.assertIsNone(read_report(self.exchange))
            self.assertEqual(read.call_count, 1)
            sleep.assert_not_called()

    def test_permanent_io_errors_are_not_retried(self) -> None:
        """Permission failures cannot be hidden as eventual storage consistency."""

        with patch.object(Path, "read_text", side_effect=PermissionError(errno.EACCES, "denied")) as read, \
                patch("common.hpo_pruning.time.sleep") as sleep:
            with self.assertRaises(PermissionError):
                read_atomic_json(self.root / "value.json")
        self.assertEqual(read.call_count, 1)
        sleep.assert_not_called()

    def test_report_retries_decode_only_and_rejects_foreign_identity(self) -> None:
        """A decoded report must satisfy the unchanged trial/attempt schema immediately."""

        with patch.object(Path, "read_text", side_effect=["{", json.dumps(self.report)]), \
                patch("common.hpo_pruning.time.sleep") as sleep:
            self.assertEqual(read_report(self.exchange), self.report)
        sleep.assert_called_once_with(0.05)
        foreign = {**self.report, "trial_number": 8}
        with patch.object(Path, "read_text", return_value=json.dumps(foreign)) as read, \
                patch("common.hpo_pruning.time.sleep") as sleep:
            with self.assertRaisesRegex(ValueError, "different trial_number"):
                read_report(self.exchange)
        self.assertEqual(read.call_count, 1)
        sleep.assert_not_called()

    def test_decision_retries_partial_json_but_not_wrong_schema(self) -> None:
        """Reply visibility may recover while identity and boolean contracts stay strict."""

        reply = {"report": self.report, "prune": False}
        with patch("common.hpo_pruning.uuid4", return_value=SimpleNamespace(hex="a" * 32)), \
                patch.object(Path, "read_text", side_effect=["{", json.dumps(reply)]), \
                patch("common.hpo_pruning.time.sleep") as sleep:
            self.assertEqual(report_epoch(self.exchange, 0, 0.8), (False, self.report))
        sleep.assert_called_once_with(0.05)
        with patch("common.hpo_pruning.uuid4", return_value=SimpleNamespace(hex="a" * 32)), \
                patch.object(Path, "read_text", return_value=json.dumps({**reply, "prune": 1})) as read, \
                patch("common.hpo_pruning.time.sleep") as sleep:
            with self.assertRaisesRegex(ValueError, "does not match"):
                report_epoch(self.exchange, 0, 0.8)
        self.assertEqual(read.call_count, 1)
        sleep.assert_not_called()


class PrivatePruningExchangeTests(unittest.TestCase):
    """Keep temporary handshakes local and close them only after child termination."""

    def setUp(self) -> None:
        """Reuse fake-process transport fixtures without loading or training a model."""

        self.fixture = transport_tests.HpoPruningTransportTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_exchange_is_private_node_local_and_unique(self) -> None:
        """Study paths and inherited TMPDIR cannot move POSIX epoch IPC onto shared storage."""

        with patch.dict(os.environ, {"TMPDIR": str(self.fixture.root)}):
            first, _ = self.fixture._start()
            second, _ = self.fixture._start()
        first_root = Path(first.pruning_exchange["directory"])
        second_root = Path(second.pruning_exchange["directory"])
        self.assertNotEqual(first_root, second_root)
        self.assertNotEqual(first.pruning_exchange["token"], second.pruning_exchange["token"])
        self.assertFalse(first_root.is_relative_to(self.fixture.root))
        # POSIX explicitly chooses node-local storage and private directory permissions.
        if os.name == "posix":
            self.assertEqual(first_root.parent, Path("/tmp"))
            self.assertEqual(first_root.stat().st_mode & 0o077, 0)
        stop_workers([first, second])
        self.assertFalse(first_root.exists())
        self.assertFalse(second_root.exists())

    def test_terminal_result_visibility_recovers_before_releasing_exchange(self) -> None:
        """A reaped worker's temporarily missing result is retried before strict validation."""

        handle, _ = self.fixture._start()
        handle.process.poll.return_value = 1
        payload = {"status": "oom", "config_path": "input.yaml", "error": "controlled OOM"}
        with patch.object(Path, "read_text", side_effect=[FileNotFoundError("not visible"), json.dumps(payload)]) as read, \
                patch("common.hpo_pruning.time.sleep") as sleep:
            result = finish_worker(handle)
        self.assertEqual(result, payload)
        self.assertEqual(read.call_count, 2)
        sleep.assert_called_once_with(0.05)
        self.assertFalse(Path(handle.pruning_exchange["directory"]).exists())


# Run only this focused suite when explicitly invoked as a script.
if __name__ == "__main__":
    unittest.main()
