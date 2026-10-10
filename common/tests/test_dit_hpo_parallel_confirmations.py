"""CPU-only parallel confirmation scheduling, recovery and cancellation checks."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from common import dit_hpo_runner as runner


class ParallelConfirmationTests(unittest.TestCase):
    """Keep paired repeats isolated, bounded and recoverable across GPU slots."""

    def setUp(self) -> None:
        """Create real frozen inputs around a mocked admitted launcher."""

        temporary = tempfile.TemporaryDirectory(prefix="dit-parallel-confirmation-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = {
            "control_root": str(self.root / "control"), 
            "hpo": {"task": "generation", "model_name": "diffusion_transformer"}, 
            "identity": {"gpus": [{"gpu_id": index} for index in range(3)]}
        }
        self.manifest_path = Path(self.plan["control_root"]) / "finalists.json"
        self.write_manifest([101, 202, 303])
        phase = patch.object(runner, "_phase_deadline", return_value=None)
        self.phase = phase.start()
        self.addCleanup(phase.stop)

    def write_manifest(self, seeds: list[int], candidates: int = 2) -> None:
        """Save immutable inputs for every requested candidate and shared seed."""

        entries = []
        for number in range(candidates):
            source = self.root / f"trial-{number:04d}.yaml"
            source.write_text(f"trial: {number}\n", encoding="utf-8")
            entries.append({
                "trial_number": number, "input_config_path": str(source), 
                "config_sha256": runner._digest(source)
            })
        runner._write(self.manifest_path, {"candidates": entries, "seeds": seeds})

    def test_three_slots_keep_pairs_parallel_ordered_and_idempotent(self) -> None:
        """Three devices overlap while each owns at most one repeat at a time."""

        barrier = threading.Barrier(3)
        lock = threading.Lock()
        active = {index: 0 for index in range(3)}
        peaks = {index: 0 for index in range(3)}
        global_peaks = []

        def launch(plan: dict, payload: dict, tag: str, deadline: float | None = None, 
                   gpu_id: int | None = None, cancel_event: threading.Event | None = None) -> dict:
            """Hold every assigned GPU until all three paired repeats overlap."""

            with lock:
                active[gpu_id] += 1
                peaks[gpu_id] = max(peaks[gpu_id], active[gpu_id])
                global_peaks.append(sum(active.values()))
            try:
                barrier.wait(timeout=5)
                self.assertIsNotNone(cancel_event)
                return {"objective": payload["training_seed"] / 1000}
            finally:
                with lock:
                    active[gpu_id] -= 1

        with patch.object(runner, "_launch", side_effect=launch) as launcher:
            records = runner.run_confirmations(self.plan)
        self.assertEqual(launcher.call_count, 6)
        self.assertEqual(peaks, {0: 1, 1: 1, 2: 1})
        self.assertEqual(max(global_peaks), 3)
        self.assertEqual([(row["identity"]["trial_number"], row["identity"]["training_seed"]) for row in records], 
                         [(number, seed) for number in range(2) for seed in [101, 202, 303]])
        with patch.object(runner, "_launch") as launcher:
            self.assertEqual(runner.run_confirmations(self.plan), records)
        launcher.assert_not_called()
        status = runner._read(self.manifest_path.parent / "confirmation_status.json")
        self.assertTrue(status["all_pairs_complete"])
        self.assertFalse(status["time_budget_exhausted"])

    def test_expired_absolute_deadline_launches_no_work(self) -> None:
        """An exhausted experiment budget does not create any fresh attempt."""

        self.phase.return_value = 0.0
        with patch.object(runner, "_launch") as launcher:
            self.assertEqual(runner.run_confirmations(self.plan), [])
        launcher.assert_not_called()
        self.assertFalse((self.manifest_path.parent / "confirmations").exists())
        status = runner._read(self.manifest_path.parent / "confirmation_status.json")
        self.assertTrue(status["time_budget_exhausted"])
        self.assertEqual(status["required_pairs"], 6)

    def test_worker_timeout_retains_authenticated_completed_pairs(self) -> None:
        """Timeout cancels remaining work without discarding a prior valid repeat."""

        manifest = runner._read(self.manifest_path)
        candidate = manifest["candidates"][0]
        expected = {
            "manifest_sha256": runner._digest(self.manifest_path), 
            "config_sha256": candidate["config_sha256"], "trial_number": 0, "training_seed": 101
        }
        path = self.manifest_path.parent / "confirmations" / "trial-0000" / "seed-101" / "completed.json"
        record = {"identity": expected, "result": {"objective": 0.2}}
        runner._write(path, record)
        self.phase.return_value = 10_000_000_000.0
        with patch.object(runner, "_launch", side_effect=TimeoutError("deadline")) as launcher:
            self.assertEqual(runner.run_confirmations(self.plan), [record])
        self.assertEqual(launcher.call_count, 3)
        self.assertTrue(all(call.kwargs["cancel_event"].is_set() for call in launcher.call_args_list))
        self.assertEqual(runner._read(path), record)
        self.assertTrue(runner._read(self.manifest_path.parent / "confirmation_status.json")["time_budget_exhausted"])

    def test_interrupt_signals_workers_before_executor_shutdown(self) -> None:
        """Notebook interruption closes admitted workers instead of waiting hours."""

        closed = []

        def launch(plan: dict, payload: dict, tag: str, deadline: float | None = None, 
                   gpu_id: int | None = None, cancel_event: threading.Event | None = None) -> dict:
            """Wait until coordinator cancellation reaches the mocked launcher."""

            self.assertTrue(cancel_event.wait(timeout=5))
            closed.append(gpu_id)
            raise InterruptedError("cancelled")

        with patch.object(runner, "_launch", side_effect=launch), \
                patch("concurrent.futures.wait", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                runner.run_confirmations(self.plan)
        self.assertEqual(sorted(closed), [0, 1, 2])

    def test_unexpected_error_still_publishes_successful_sibling(self) -> None:
        """A completed sibling is not retrained after another repeat fails."""

        self.write_manifest([101, 202], candidates=1)
        self.plan["identity"]["gpus"] = [{"gpu_id": 0}, {"gpu_id": 1}]
        barrier = threading.Barrier(2)

        def launch(plan: dict, payload: dict, tag: str, deadline: float | None = None, 
                   gpu_id: int | None = None, cancel_event: threading.Event | None = None) -> dict:
            """Finish one repeat as its sibling reports an unrelated failure."""

            barrier.wait(timeout=5)
            # Keep programming errors distinct from ordinary deadline exhaustion.
            if payload["training_seed"] == 101:
                raise RuntimeError("unexpected failure")
            return {"objective": 0.2}

        with patch.object(runner, "_launch", side_effect=launch):
            with self.assertRaisesRegex(RuntimeError, "unexpected failure"):
                runner.run_confirmations(self.plan)
        completed = self.manifest_path.parent / "confirmations" / "trial-0000" / "seed-202" / "completed.json"
        self.assertTrue(completed.is_file())
        with patch.object(runner, "_launch", return_value={"objective": 0.3}) as launcher:
            records = runner.run_confirmations(self.plan)
        self.assertEqual(launcher.call_count, 1)
        self.assertEqual(launcher.call_args.args[1]["training_seed"], 101)
        self.assertEqual(len(records), 2)


# Focused execution never launches scientific workers or initializes a GPU.
if __name__ == "__main__":
    unittest.main(verbosity=2)
