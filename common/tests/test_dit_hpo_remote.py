"""Check GPU admission orchestration without claiming slots or importing TF."""

from contextlib import contextmanager, ExitStack
from pathlib import Path
import signal
from types import SimpleNamespace
from typing import Any, Iterator
import unittest
from unittest.mock import Mock, patch

from common import dit_hpo_remote as remote


class AdmissionBlocked(RuntimeError):
    """Represent the deployed allocator's fail-closed admission error."""


class FakeProcess:
    """Track subprocess lifetime without creating a real worker."""

    def __init__(self, events: list[str]) -> None:
        """Retain the lifecycle event log and initial running state."""

        self.pid = 42
        self.events = events
        self.returncode = None

    def poll(self) -> int | None:
        """Return the fake worker's current exit status."""

        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        """Record that the coordinator reaped the worker."""

        self.events.append("wait")
        self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        """Record a termination request without pretending it was reaped."""

        self.events.append("terminate")

    def kill(self) -> None:
        """Record forced termination of the fake worker."""

        self.events.append("kill")


@contextmanager
def fake_launch() -> Iterator[tuple[FakeProcess, SimpleNamespace, SimpleNamespace, list[str]]]:
    """Replace admission, OS pipes, and subprocess creation with event mocks."""

    events = []
    process = FakeProcess(events)
    lease = SimpleNamespace(
        record={"slot": 0, "owner": {"pid": 1, "start_ticks": "2"}}, 
        register_worker=Mock(side_effect=lambda pid: events.append("register")), 
        close=Mock(side_effect=lambda: events.append("close"))
    )
    allocator = SimpleNamespace(AdmissionBlocked=AdmissionBlocked, acquire=Mock(return_value=lease))
    identity = {"checkout_root": "/workspace/test", "pool_host": "a100_2"}

    def start_process(command: list[str], **kwargs: Any) -> FakeProcess:
        """Retain the launch configuration and record process creation."""

        events.append("spawn")
        process.launch_options = kwargs
        return process

    patches = [
        (remote, "_verify_snapshot", Mock(return_value=identity)), 
        (remote, "_in_existing_allocation", Mock(return_value=False)), 
        (remote, "_allocator", Mock(return_value=allocator)), 
        (remote, "sys", SimpleNamespace(modules={})), 
        (remote.subprocess, "Popen", Mock(side_effect=start_process)), 
        (remote.os, "pipe", Mock(return_value=(101, 102))), 
        (remote.os, "close", Mock()), 
        (remote.os, "write", Mock(side_effect=lambda descriptor, value: events.append("gate"))), 
        (remote.time, "sleep", Mock())
    ]
    with ExitStack() as stack:
        for target, attribute, replacement in patches:
            stack.enter_context(patch.object(target, attribute, replacement))
        yield process, lease, allocator, events


class RemoteAdmissionTests(unittest.TestCase):
    """Exercise ownership, memory reservation, and worker cleanup boundaries."""

    def test_registered_before_gate_and_reaped_before_release(self) -> None:
        """Do not initialize workers before registration or release live slots."""

        with fake_launch() as (process, lease, allocator, events):
            with remote.launch_worker(["python", "worker.py"], Path("/workspace/test"), {}) as worker:
                self.assertIs(worker, process)
                worker.wait()
            allocator.acquire.assert_called_once_with(memory_mb=13312, gpu=0, max_jobs=2)
            lease.register_worker.assert_called_once_with(42)
            self.assertEqual(events, ["spawn", "register", "gate", "wait", "close"])
            self.assertEqual(process.launch_options["pass_fds"], tuple([101]))
            self.assertEqual(process.launch_options["env"]["CUDA_VISIBLE_DEVICES"], "0")

    def test_interruption_reaps_only_owned_worker_before_release(self) -> None:
        """A notebook interrupt terminates and reaps its child before release."""

        with fake_launch() as (process, lease, allocator, events):
            with self.assertRaises(KeyboardInterrupt):
                with remote.launch_worker(["python", "worker.py"], Path("/workspace/test"), {}):
                    raise KeyboardInterrupt()
            self.assertEqual(events[-3:], ["terminate", "wait", "close"])

    def test_capacity_shortage_queues_and_rechecks(self) -> None:
        """Occupied slots wait without introducing a global deadline."""

        with fake_launch() as (process, lease, allocator, events):
            allocator.acquire.side_effect = [
                AdmissionBlocked("Both remote workload slots are occupied; no third job may start."), 
                lease
            ]
            with remote.launch_worker(["python", "worker.py"], Path("/workspace/test"), {}) as worker:
                worker.wait()
            self.assertEqual(allocator.acquire.call_count, 2)
            remote.time.sleep.assert_called_once_with(15)
            self.assertEqual(remote._verify_snapshot.call_count, 3)

    def test_unknown_ownership_fails_without_launch_or_queue(self) -> None:
        """Unknown GPU/kernel workers must not be bypassed as a capacity issue."""

        with fake_launch() as (process, lease, allocator, events):
            allocator.acquire.side_effect = AdmissionBlocked("Unmapped GPU/kernel workers block admission")
            with self.assertRaisesRegex(AdmissionBlocked, "Unmapped"):
                with remote.launch_worker(["python", "worker.py"], Path("/workspace/test"), {}):
                    self.fail("A worker was launched despite unknown ownership.")
            self.assertEqual(events, [])
            remote.time.sleep.assert_not_called()
            lease.close.assert_not_called()

    def test_parent_death_signal_armed(self) -> None:
        """A worker requests Linux termination when the coordinator exits."""

        library = SimpleNamespace(prctl=Mock(return_value=0))
        with patch.object(remote.ctypes, "CDLL", return_value=library):
            remote._arm_parent_exit()
        library.prctl.assert_called_once_with(1, signal.SIGTERM, 0, 0, 0)

    def test_parent_death_signal_failure_is_fatal(self) -> None:
        """Do not continue GPU work if orphan prevention is unavailable."""

        library = SimpleNamespace(prctl=Mock(return_value=-1))
        with patch.object(remote.ctypes, "CDLL", return_value=library):
            with self.assertRaises(OSError):
                remote._arm_parent_exit()


# Allow the focused checks to run directly as a standalone test module.
if __name__ == "__main__":
    unittest.main()
