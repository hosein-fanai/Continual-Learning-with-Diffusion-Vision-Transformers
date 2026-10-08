"""Check deadline-aware admission without reserving GPUs or importing TensorFlow."""

import unittest
from unittest.mock import Mock, patch

from common import dit_hpo_remote as remote
from common.tests.test_dit_hpo_remote import AdmissionBlocked, fake_launch


class AdmissionDeadlineTests(unittest.TestCase):
    """Exercise expiry at admission and ownership-preserving cleanup."""

    def test_expired_budget_does_not_acquire_or_spawn(self) -> None:
        """An expired campaign cannot claim another GPU reservation."""

        with fake_launch() as (process, lease, allocator, events):
            with patch.object(remote.time, "time", return_value=101.0):
                with self.assertRaises(TimeoutError):
                    with remote.launch_worker(["python", "worker.py"], "/workspace/test", {}, deadline=100.0):
                        self.fail("Expired admission yielded a worker.")
            allocator.acquire.assert_not_called()
            self.assertNotIn("spawn", events)

    def test_capacity_queue_expires_without_spawning(self) -> None:
        """Known capacity shortages stop at the remaining wall-time budget."""

        with fake_launch() as (process, lease, allocator, events):
            allocator.acquire.side_effect = AdmissionBlocked("Insufficient free GPU memory")
            with patch.object(remote.time, "time", side_effect=[98.0, 99.0, 99.0, 100.0]):
                with self.assertRaises(TimeoutError):
                    with remote.launch_worker(["python", "worker.py"], "/workspace/test", {}, deadline=100.0):
                        self.fail("Queued admission yielded a worker.")
            remote.time.sleep.assert_called_once_with(1.0)
            self.assertNotIn("spawn", events)

    def test_expiry_after_reservation_releases_before_launch(self) -> None:
        """A reservation obtained at the cutoff is released without training."""

        with fake_launch() as (process, lease, allocator, events):
            with patch.object(remote.time, "time", side_effect=[99.0, 100.0]):
                with self.assertRaises(TimeoutError):
                    with remote.launch_worker(["python", "worker.py"], "/workspace/test", {}, deadline=100.0):
                        self.fail("Late admission yielded a worker.")
            self.assertEqual(events, ["close"])

    def test_child_inherits_deadline_and_timeout_reaps_before_release(self) -> None:
        """A runner timeout preserves the worker's lifetime and reservation order."""

        with fake_launch() as (process, lease, allocator, events):
            with patch.object(remote.time, "time", return_value=99.0):
                with self.assertRaises(TimeoutError):
                    with remote.launch_worker(["python", "worker.py"], "/workspace/test", {}, deadline=100.0):
                        raise TimeoutError("Runner deadline")
            self.assertEqual(process.launch_options["env"]["DIT_HPO_DEADLINE_UTC"], "100.0")
            self.assertEqual(events[-3:], ["terminate", "wait", "close"])

    def test_cancellation_wakes_queued_admission(self) -> None:
        """Thread-group cancellation releases a queued confirmation promptly."""

        event = Mock()
        event.is_set.side_effect = [False, False, True]
        with fake_launch() as (process, lease, allocator, events):
            allocator.acquire.side_effect = AdmissionBlocked("Insufficient free GPU memory")
            with self.assertRaises(InterruptedError):
                with remote.launch_worker(["python", "worker.py"], "/workspace/test", {}, cancel_event=event):
                    self.fail("Cancelled admission yielded a worker.")
            event.wait.assert_called_once_with(remote.POLL_SECONDS)
            remote.time.sleep.assert_not_called()
            self.assertNotIn("spawn", events)

    def test_confirmation_gpu_must_belong_to_verified_plan(self) -> None:
        """Parallel confirmation can select each approved GPU, never another one."""

        identity = {"gpu_ids": [0, 1, 2]}
        with patch.object(remote, "_verify_snapshot", return_value=identity), \
        patch.object(remote, "inspect_remote", return_value={}) as inspect:
            remote.serial_worker_identity("/workspace/test", identity, gpu_id=2)
            inspect.assert_called_once_with("/workspace/test", concurrent_trials=1, gpu_ids=[2])
            with self.assertRaises(ValueError):
                remote.serial_worker_identity("/workspace/test", identity, gpu_id=3)


# Direct execution runs only this focused admission suite.
if __name__ == "__main__":
    unittest.main()
