"""Deadline cancellation keeps real Optuna outcomes without launching model work."""

from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time as wall_time
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import optuna
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from common.hpo import _enqueue_recovery_trials, run_hpo
from common.tests.test_hpo_concurrency import _Workers


class HpoDeadlineTests(TestCase):
    """Exercise runtime deadlines with persistent studies and deterministic children."""

    def setUp(self) -> None:
        """Keep every study and event file in an independent temporary directory."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def options(self, **changes: object) -> dict:
        """Build a supported isolated DiT recipe without training a model."""

        options = {
            "task": "generation", "model_name": "diffusion_transformer", 
            "dataset_name": "CIFAR10", "results_path": str(self.root), 
            "n_trials": 6, "epochs": 1, "concurrent_trials": 2, 
            "trial_budget_mode": "total", "timeout": 1.0, 
            "stop_active_on_timeout": True, "seed": 42
        }
        options.update(changes)
        return options

    def run_at_cutoff(self, workers: _Workers, **changes: object) -> optuna.study.Study:
        """Expire a scheduler clock immediately after both initial children start."""

        clock = SimpleNamespace(now=0.0)

        def expire(handle: SimpleNamespace) -> None:
            """Give both slots time to launch before advancing the fake clock."""

            # Crossing the deadline does not depend on CPU speed or test timing.
            if handle.number == 1:
                clock.now = 2.0

        workers.after_start = expire
        timer = SimpleNamespace(monotonic=lambda: clock.now, sleep=lambda seconds: None, time=wall_time)
        with workers.installed(), patch("common.hpo.time", timer):
            return run_hpo(**self.options(**changes))

    def study_root(self) -> Path:
        """Resolve this test's sole ordinary generation study directory."""

        return self.root / "generation" / "diffusion_transformer" / "cifar10"

    def test_deadline_reaps_active_workers_and_persists_failures(self) -> None:
        """Cancellation saves FAIL diagnostics and events without invented objectives."""

        workers = _Workers(delays={0: 100, 1: 100})
        study = self.run_at_cutoff(workers)
        self.assertEqual(workers.stopped, [0, 1])
        self.assertFalse(workers.active)
        self.assertEqual(len(workers.handles), 2)
        self.assertEqual([trial.state.name for trial in study.trials], ["FAIL", "FAIL"])
        self.assertTrue(study.user_attrs["execution"]["stop_active_on_timeout"])
        with (self.study_root() / "trials.csv").open(newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual([row["state"] for row in rows], ["FAIL", "FAIL"])
        self.assertEqual([row["user_attrs_stop_reason"] for row in rows], ["deadline", "deadline"])
        for trial in study.trials:
            self.assertIsNone(trial.values)
            self.assertEqual(trial.user_attrs["stop_reason"], "deadline")
            self.assertEqual(trial.user_attrs["deadline"]["timeout_seconds"], 1.0)
            self.assertGreaterEqual(trial.user_attrs["deadline"]["elapsed_seconds"], 1.0)
            self.assertNotIn("oom", trial.user_attrs)
            self.assertTrue(Path(trial.user_attrs["config_path"]).is_file())
            events = EventAccumulator(str(self.study_root() / "tensorboard" / f"trial-{trial.number:04d}" / "outcome"))
            events.Reload()
            self.assertEqual(events.Tensors("hpo/failed")[-1].tensor_proto.float_val[0], 1.0)
            self.assertEqual(events.Tensors("hpo/pruned")[-1].tensor_proto.float_val[0], 0.0)
            self.assertEqual(events.Tensors("hpo/deadline_cancelled")[-1].tensor_proto.float_val[0], 1.0)
            self.assertEqual(events.Tensors("hpo/stop_reason")[-1].tensor_proto.string_val[0], b"deadline")
            self.assertNotIn("hpo/generation_loss", events.Tags()["tensors"])

    def test_finished_result_at_deadline_keeps_its_finite_objective(self) -> None:
        """An already exited worker is completed before the slow child is cancelled."""

        workers = _Workers(delays={1: 100})
        study = self.run_at_cutoff(workers)
        self.assertEqual([trial.state.name for trial in study.trials], ["COMPLETE", "FAIL"])
        self.assertAlmostEqual(study.trials[0].value, 0.01)
        self.assertNotIn("stop_reason", study.trials[0].user_attrs)
        self.assertEqual(workers.stopped, [1])

    def test_finished_oom_at_deadline_remains_resource_pruned(self) -> None:
        """An OOM response retains its own terminal reason when a peer times out."""

        workers = _Workers(delays={1: 100}, outcomes={
            0: {"status": "oom", "error": "controlled allocation failure"}
        })
        study = self.run_at_cutoff(workers)
        self.assertEqual([trial.state.name for trial in study.trials], ["PRUNED", "FAIL"])
        self.assertEqual(study.trials[0].user_attrs["oom"]["reason"], "out_of_memory")
        self.assertNotIn("stop_reason", study.trials[0].user_attrs)
        self.assertEqual(workers.stopped, [1])

    def test_legacy_timeout_drains_and_unlimited_mode_does_not_cancel(self) -> None:
        """The default timeout contract and an omitted timeout retain completed work."""

        workers = _Workers(delays={0: 4, 1: 2})
        study = self.run_at_cutoff(workers, stop_active_on_timeout=False)
        self.assertEqual([trial.state.name for trial in study.trials], ["COMPLETE", "COMPLETE"])
        self.assertFalse(workers.stopped)
        unlimited = _Workers(delays={0: 4})
        with unlimited.installed():
            study = run_hpo(**self.options(timeout=None, n_trials=3))
        self.assertEqual([trial.state.name for trial in study.trials], ["COMPLETE"] * 3)
        self.assertFalse(unlimited.stopped)

    def test_execution_option_can_change_on_resume_without_retrying_cancellation(self) -> None:
        """Changing timeout behavior preserves the scientific seal and count allowance."""

        study = self.run_at_cutoff(_Workers(delays={0: 100, 1: 100}))
        original_spec = (self.study_root() / "study_spec.json").read_bytes()
        workers = _Workers()
        with workers.installed(), patch("common.hpo._has_committed_task_checkpoint", return_value=True):
            resumed = run_hpo(**self.options(
                resume_from=self.study_root(), n_trials=3, timeout=None, stop_active_on_timeout=False
            ))
        self.assertEqual(resumed.study_name, study.study_name)
        self.assertEqual([trial.state.name for trial in resumed.trials], ["FAIL", "FAIL", "COMPLETE"])
        self.assertEqual([handle.number for handle in workers.handles], [2])
        self.assertNotIn("resume_source_trial_number", resumed.trials[2].user_attrs)
        self.assertEqual((self.study_root() / "study_spec.json").read_bytes(), original_spec)

    def test_only_published_deadline_failures_are_excluded_from_recovery(self) -> None:
        """Ordinary failed and still-running records retain parameter-identical recovery."""

        study = optuna.create_study()
        cancelled = study.ask()
        cancelled.set_user_attr("stop_reason", "deadline")
        study.tell(cancelled, state=optuna.trial.TrialState.FAIL)
        failed = study.ask()
        study.tell(failed, state=optuna.trial.TrialState.FAIL)
        running = study.ask()
        running.set_user_attr("stop_reason", "deadline")
        with patch("common.hpo._has_committed_task_checkpoint", return_value=True):
            retried = _enqueue_recovery_trials(study, self.root)
        self.assertEqual(retried, (1, 2))
        self.assertEqual([trial.user_attrs["resume_source_trial_number"] for trial in study.trials[3:]], [1, 2])

    def test_surviving_worker_is_not_published_as_cancelled(self) -> None:
        """Unsuccessful cleanup leaves RUNNING state and raises for the owner to handle."""

        workers = _Workers(delays={0: 100, 1: 100})
        clock = SimpleNamespace(now=0.0)

        def expire(handle: SimpleNamespace) -> None:
            """Advance time only after the initial worker group is allocated."""

            # A failed stop must never be mistaken for successful reaping.
            if handle.number == 1:
                clock.now = 2.0

        workers.after_start = expire
        timer = SimpleNamespace(monotonic=lambda: clock.now, sleep=lambda seconds: None, time=wall_time)
        with workers.installed(), patch("common.hpo.time", timer), patch("common.hpo.stop_workers") as stop:
            with self.assertRaisesRegex(RuntimeError, "survived deadline cleanup"):
                run_hpo(**self.options())
        self.assertEqual(stop.call_count, 2)
        study = optuna.load_study(
            study_name="generation-diffusion_transformer-cifar10", 
            storage="sqlite:///" + (self.study_root() / "study.db").as_posix()
        )
        self.assertEqual([trial.state.name for trial in study.trials], ["RUNNING", "RUNNING"])
        self.assertTrue(all("stop_reason" not in trial.user_attrs for trial in study.trials))

    def test_in_process_execution_rejects_unenforceable_hard_stop(self) -> None:
        """The runtime flag cannot silently claim to interrupt an in-process fit."""

        with patch("optuna.create_study") as create:
            with self.assertRaisesRegex(ValueError, "requires subprocess execution"):
                run_hpo(**self.options(concurrent_trials=1))
        create.assert_not_called()


# Direct execution selects the same CPU-only boundary tests as discovery.
if __name__ == "__main__":
    main()
