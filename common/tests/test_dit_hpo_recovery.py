"""CPU-only stopped-search recovery checks using a temporary real Optuna database."""

from __future__ import annotations

from contextlib import closing
import copy
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import optuna

from common import dit_hpo_recovery as recovery
from common.hpo_process import study_lock


class StoppedSearchRecoveryTests(unittest.TestCase):
    """Exercise durable backup, guarded state transitions and idempotent replay."""

    def setUp(self) -> None:
        """Create a sealed expired study with every relevant trial state."""

        temporary = tempfile.TemporaryDirectory(prefix="dit-stopped-search-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.control = self.root / "notebook_runner"
        self.control.mkdir()
        hpo = {
            "task": "generation", "model_name": "diffusion_transformer", "dataset_name": "CIFAR100", 
            "epochs": 50, "seed": 42, "fit_method": "fit", "objective_metrics": ["generation_loss"], 
            "objective_directions": ["minimize"], "dtype_policy": "float32", "n_startup_trials": 40, 
            "validation_source": "test", "validation_ratio": 0.0
        }
        self.plan = {
            "version": 2, "checkout_root": str(self.root), "study_root": str(self.root), 
            "control_root": str(self.control), "study_name": "generation-diffusion_transformer-cifar100", 
            "hpo": hpo, "identity": {"source_sha256": {}, "versions": {}, "python": "test"}, 
            "time_budget": {"experiment_hours": 20.0, "confirmation_reserve_hours": 2.0}
        }
        recipe = {"version": 2, "hpo": hpo, **self.plan["identity"]}
        self.write(self.control / "recipe.json", recipe)
        self.write(self.control / "budget.json", {
            "policy": self.plan["time_budget"], "started_at_unix": 0.0, 
            "deadline_unix": 2000.0, "search_deadline_unix": 800.0, "cleanup_seconds": 60.0
        })
        self.spec = {
            **hpo, "study_name": self.plan["study_name"], "dataset_name": "cifar100", 
            "data_selection": {"resolved": {"validation_source": "test", "validation_ratio": 0.0}}
        }
        self.write(self.root / "study_spec.json", {"spec": self.spec, "fingerprint": recovery._fingerprint(self.spec)})
        self.study = optuna.create_study(
            storage="sqlite:///" + (self.root / "study.db").as_posix(), 
            study_name=self.plan["study_name"], direction="minimize"
        )
        self.study.set_user_attr("study_spec", self.spec)
        self.study.set_user_attr("study_spec_fingerprint", recovery._fingerprint(self.spec))
        self.study.set_user_attr("sampler_rng_state", {"preserve": "exact"})
        complete = self.study.ask()
        complete.suggest_categorical("depth", [3, 4])
        complete.set_user_attr("evidence", "untouched")
        self.study.tell(complete, 0.25)
        pruned = self.study.ask()
        pruned.report(0.5, 1)
        self.study.tell(pruned, state=optuna.trial.TrialState.PRUNED)
        self.running = self.study.ask()
        self.running.set_user_attr("worker_error", "historical interruption")
        self.study.enqueue_trial({"depth": 3})
        self.study.trials_dataframe().to_csv(self.root / "trials.csv", index=False)
        self.remote = SimpleNamespace(_remote_root=Mock(return_value=(self.root, "test-host")), _verify_snapshot=Mock(return_value={}))
        remote_patch = patch.dict(sys.modules, {"common.dit_hpo_remote": self.remote})
        remote_patch.start()
        self.addCleanup(remote_patch.stop)
        idle = patch.object(recovery, "_require_stopped", return_value={"caller_pid": 123, "gpu_processes": []})
        self.idle = idle.start()
        self.addCleanup(idle.stop)
        clock = patch.object(recovery.time, "time", return_value=1000.0)
        clock.start()
        self.addCleanup(clock.stop)

    def write(self, path: Path, value: dict) -> None:
        """Persist fixture JSON without invoking a model library."""

        path.write_text(json.dumps(value), encoding="utf-8")

    def test_recovery_preserves_completed_pruned_waiting_and_budget(self) -> None:
        """Only orphan RUNNING changes, and SQLite backup contains the original state."""

        before = self.study.get_trials()
        unchanged = recovery._trial_identity([trial for trial in before if trial.number != 2])
        budget = (self.control / "budget.json").read_bytes()
        attrs = copy.deepcopy(self.study.user_attrs)
        receipt = recovery.recover_stopped_search(self.plan)
        after = self.study.get_trials()
        self.assertEqual([trial.state.name for trial in after], ["COMPLETE", "PRUNED", "FAIL", "WAITING"])
        self.assertEqual(recovery._trial_identity([trial for trial in after if trial.number != 2]), unchanged)
        self.assertEqual(after[2].user_attrs["worker_error"], "historical interruption")
        self.assertEqual(after[2].user_attrs["stop_reason"], "interrupted")
        self.assertEqual((self.control / "budget.json").read_bytes(), budget)
        self.assertEqual(self.study.user_attrs, attrs)
        self.assertEqual(receipt["recovered_trial_numbers"], [2])
        self.assertFalse(receipt["training_started"])
        backup = Path(receipt["backup_path"])
        self.assertTrue((backup / "recipe.json").is_file())
        self.assertTrue((backup / "trials.csv").is_file())
        with closing(sqlite3.connect(backup / "study.db")) as database:
            self.assertEqual(database.execute("SELECT state FROM trials ORDER BY number").fetchall(), [tuple([state]) for state in ["COMPLETE", "PRUNED", "RUNNING", "WAITING"]])
        self.assertEqual(self.idle.call_count, 2)

    def test_idempotent_receipt_and_no_additional_backup(self) -> None:
        """A second invocation returns the same receipt without another mutation."""

        receipt = recovery.recover_stopped_search(self.plan)
        self.assertEqual(recovery.recover_stopped_search(self.plan), receipt)
        self.assertEqual(len(list((self.control / "stopped_search_recovery").iterdir())), 1)
        self.assertEqual(len(self.study.trials), 4)

    def test_unstarted_and_no_running_are_noops_before_deadline(self) -> None:
        """The API can be called in every notebook run without forcing recovery."""

        self.study.tell(self.running, state=optuna.trial.TrialState.FAIL)
        with patch.object(recovery.time, "time", return_value=0.0):
            self.assertEqual(recovery.recover_stopped_search(self.plan)["status"], "no_running_trials")
        self.idle.assert_not_called()
        plan = {**self.plan, "study_root": str(self.root / "empty"), "control_root": str(self.root / "empty/notebook_runner")}
        self.assertFalse(recovery.recover_stopped_search(plan)["study_started"])

    def test_live_workers_prevent_backup_and_mutation(self) -> None:
        """A live process cannot be relabelled merely because its clock expired."""

        self.idle.side_effect = RuntimeError("live workers")
        with self.assertRaisesRegex(RuntimeError, "live workers"):
            recovery.recover_stopped_search(self.plan)
        self.assertEqual(self.study.trials[2].state.name, "RUNNING")
        self.assertFalse((self.control / "stopped_search_recovery").exists())

    def test_second_idle_check_prevents_mutation_after_backup(self) -> None:
        """A process appearing while remote files are copied blocks recovery."""

        self.idle.side_effect = [{"caller_pid": 123}, RuntimeError("new worker")]
        with self.assertRaisesRegex(RuntimeError, "new worker"):
            recovery.recover_stopped_search(self.plan)
        self.assertEqual(self.study.trials[2].state.name, "RUNNING")

    def test_unexpired_missing_clock_and_frozen_finalists_reject(self) -> None:
        """The explicit recovery API cannot cancel unexpired or selected work."""

        with patch.object(recovery.time, "time", return_value=0.0), self.assertRaisesRegex(ValueError, "deadline"):
            recovery.recover_stopped_search(self.plan)
        self.write(self.control / "finalists.json", {})
        with self.assertRaisesRegex(ValueError, "frozen"):
            recovery.recover_stopped_search(self.plan)
        (self.control / "finalists.json").unlink()
        (self.control / "budget.json").unlink()
        with self.assertRaisesRegex(ValueError, "deadline"):
            recovery.recover_stopped_search(self.plan)

    def test_recipe_source_and_study_spec_mismatch_reject(self) -> None:
        """Scientific provenance is authenticated before state changes."""

        self.remote._verify_snapshot.side_effect = RuntimeError("source identity")
        with self.assertRaisesRegex(RuntimeError, "source identity"):
            recovery.recover_stopped_search(self.plan)
        self.remote._verify_snapshot.side_effect = None
        self.study.set_user_attr("study_spec_fingerprint", "tampered")
        with self.assertRaisesRegex(ValueError, "storage identity"):
            recovery.recover_stopped_search(self.plan)
        self.assertEqual(self.study.trials[2].state.name, "RUNNING")

    def test_backup_failure_cannot_change_a_trial(self) -> None:
        """State changes wait for a complete successful backup."""

        with patch.object(recovery, "_backup", side_effect=OSError("full disk")), self.assertRaisesRegex(OSError, "full disk"):
            recovery.recover_stopped_search(self.plan)
        self.assertEqual(self.study.trials[2].state.name, "RUNNING")

    def test_different_validation_feedback_rejects(self) -> None:
        """Matching hashes cannot hide a study using different feedback data."""

        self.spec["data_selection"]["resolved"]["validation_source"] = "split"
        self.write(self.root / "study_spec.json", {"spec": self.spec, "fingerprint": recovery._fingerprint(self.spec)})
        self.study.set_user_attr("study_spec", self.spec)
        self.study.set_user_attr("study_spec_fingerprint", recovery._fingerprint(self.spec))
        with self.assertRaisesRegex(ValueError, "data selection"):
            recovery.recover_stopped_search(self.plan)
        self.assertEqual(self.study.trials[2].state.name, "RUNNING")

    def test_both_coordinator_locks_are_required(self) -> None:
        """A kernel or HPO coordinator owning either lock blocks all mutations."""

        for directory in (self.root, self.control):
            with self.subTest(directory=directory), study_lock(directory), self.assertRaisesRegex(RuntimeError, "coordinator"):
                recovery.recover_stopped_search(self.plan)
        self.assertEqual(self.study.trials[2].state.name, "RUNNING")

    def test_deadline_reason_requires_existing_cancellation_evidence(self) -> None:
        """Existing deadline cancellation evidence retains its recorded reason."""

        self.running.set_user_attr("stop_reason", "deadline")
        self.running.set_user_attr("deadline", {"timeout_seconds": 1.0, "elapsed_seconds": 2.0})
        recovery.recover_stopped_search(self.plan)
        self.assertEqual(self.study.trials[2].user_attrs["stop_reason"], "deadline")

    def test_failed_staging_leaves_original_running(self) -> None:
        """An Optuna error in the private copy cannot partially repair the real study."""

        before = recovery._trial_identity(self.study.get_trials())
        with patch.object(recovery, "_stage_recovery", side_effect=RuntimeError("staging failed")), self.assertRaisesRegex(RuntimeError, "staging failed"):
            recovery.recover_stopped_search(self.plan)
        self.assertEqual(recovery._trial_identity(self.study.get_trials()), before)
        self.assertFalse((self.control / "stopped_search_recovery.json").exists())

    def test_writer_during_preparation_prevents_publication(self) -> None:
        """A writer bypassing the coordinator lock is detected before publication."""

        stage = recovery._stage_recovery

        def write_during_staging(*args: object, **kwargs: object) -> list:
            """Change the original through Optuna after the private copy is prepared."""

            result = stage(*args, **kwargs)
            self.running.set_user_attr("late_writer", "preserve")
            return result

        with patch.object(recovery, "_stage_recovery", side_effect=write_during_staging), self.assertRaisesRegex(RuntimeError, "original study changed"):
            recovery.recover_stopped_search(self.plan)
        self.assertEqual(self.study.trials[2].state.name, "RUNNING")
        self.assertEqual(self.study.trials[2].user_attrs["late_writer"], "preserve")

    def test_publication_preserves_inode_and_existing_sqlite_connection(self) -> None:
        """Existing connections observe committed changes through the original file."""

        database = self.root / "study.db"
        inode = database.stat().st_ino
        with closing(sqlite3.connect(database)) as existing:
            self.assertEqual(existing.execute("SELECT state FROM trials WHERE number=2").fetchone(), tuple(["RUNNING"]))
            receipt = recovery.recover_stopped_search(self.plan)
            self.assertEqual(existing.execute("SELECT state FROM trials WHERE number=2").fetchone(), tuple(["FAIL"]))
        self.assertEqual(database.stat().st_ino, inode)
        self.assertTrue(receipt["database_inode_preserved"])

    def test_locked_publication_times_out_without_changing_original(self) -> None:
        """An actual SQLite writer lock aborts the backup without exposing partial writes."""

        database = self.root / "study.db"
        staged = self.root / "locked-publication-source.db"
        recovery._copy_database(database, staged)
        staged_study = optuna.load_study(study_name=self.plan["study_name"], storage="sqlite:///" + staged.as_posix())
        staged_study.tell(2, state=optuna.trial.TrialState.FAIL)
        before = recovery._trial_identity(self.study.get_trials())
        with closing(sqlite3.connect(database)) as writer:
            writer.execute("BEGIN IMMEDIATE")
            try:
                with patch.object(recovery.time, "monotonic", side_effect=[0.0, 31.0]), self.assertRaisesRegex(TimeoutError, "locked"):
                    recovery._copy_database(staged, database, existing_destination=True)
            finally:
                writer.rollback()
        self.assertEqual(recovery._trial_identity(self.study.get_trials()), before)


class StoppedSearchProcessGuardTests(unittest.TestCase):
    """Check process evidence without running a GPU framework or command."""

    def test_only_calling_notebook_kernel_is_allowed(self) -> None:
        """Other idle kernels are still incompatible with exclusive recovery."""

        with patch.object(recovery.subprocess, "check_output", return_value=""), patch.object(recovery.os, "getpid", return_value=10), patch.object(
            recovery, "_process_snapshot", return_value={10: "python -m ipykernel_launcher", 11: "jupyter-lab"}
        ) as snapshot:
            self.assertEqual(recovery._require_stopped([])["caller_pid"], 10)
            snapshot.return_value[12] = "python -m ipykernel_launcher"
            with self.assertRaisesRegex(RuntimeError, "kernels"):
                recovery._require_stopped([])

    def test_gpu_process_or_hpo_process_blocks(self) -> None:
        """Either independent ownership signal suffices to prevent mutation."""

        with patch.object(recovery.subprocess, "check_output", return_value="321, GPU-test\n"), self.assertRaisesRegex(RuntimeError, "GPU processes"):
            recovery._require_stopped([])
        with patch.object(recovery.subprocess, "check_output", return_value=""), patch.object(recovery, "_process_snapshot", return_value={11: "python -m common.hpo_worker"}), self.assertRaisesRegex(RuntimeError, "HPO workers"):
            recovery._require_stopped([])

    def test_recycled_recorded_pid_is_ambiguous_and_blocks(self) -> None:
        """A live recorded PID cannot be declared dead from an unrelated command name."""

        trial = SimpleNamespace(user_attrs={"worker_pid": 321})
        with patch.object(recovery.subprocess, "check_output", return_value=""), patch.object(recovery, "_process_snapshot", return_value={321: "python unrelated.py"}), self.assertRaisesRegex(RuntimeError, "321"):
            recovery._require_stopped([trial])


# Direct execution runs only these CPU fixtures.
if __name__ == "__main__":
    unittest.main()
