"""Durability and explicit-resume checks for opt-in local HPO SQLite."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from common import hpo_sqlite


class LocalSQLiteTests(unittest.TestCase):
    """Exercise record preservation and publication failure boundaries."""

    def setUp(self):
        """Create an independent study with representative persistent records."""

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "shared" / "study"
        self.local = self.base / "node"
        self.root.mkdir(parents=True)
        self.original = self.root / "study.db"
        with closing(sqlite3.connect(self.original)) as database:
            database.execute("CREATE TABLE trials (number INTEGER PRIMARY KEY, state TEXT, value REAL)")
            database.execute("INSERT INTO trials VALUES (1, 'COMPLETE', 0.125)")
            database.execute("INSERT INTO trials VALUES (2, 'RUNNING', NULL)")
            database.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
            database.execute("INSERT INTO metadata VALUES ('sampler_rng', 'unchanged-state')")
            database.commit()

    def records(self, path):
        """Read complete trial and sampler evidence from a selected database."""

        with closing(sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True)) as database:
            return {
                "trials": database.execute("SELECT * FROM trials ORDER BY number").fetchall(), 
                "metadata": database.execute("SELECT * FROM metadata ORDER BY key").fetchall()
            }

    def add_trial(self, path, number=3):
        """Commit an extra trial only to the selected authoritative database."""

        with closing(sqlite3.connect(path)) as database:
            database.execute("INSERT INTO trials VALUES (?, 'COMPLETE', 0.0625)", [number])
            database.commit()

    def marker(self):
        """Read the currently committed snapshot pointer."""

        return json.loads((self.root / "sqlite_local.json").read_text(encoding="utf-8"))

    def test_legacy_mode_is_unchanged(self):
        """Opt-in storage leaves ordinary studies byte-for-byte unchanged."""

        before = self.original.read_bytes()
        self.assertEqual(hpo_sqlite.database_path(self.root), self.original)
        self.assertIsNone(hpo_sqlite.snapshot(self.root))
        with hpo_sqlite.sqlite_snapshots(self.root):
            pass
        self.assertEqual(self.original.read_bytes(), before)
        self.assertFalse((self.root / "sqlite_local.json").exists())

    def test_enable_preserves_all_records_and_separates_writes(self):
        """Local writes do not mutate the last published snapshot."""

        before = self.records(self.original)
        marker = hpo_sqlite.enable_local_sqlite(self.root, self.local)
        active = hpo_sqlite.database_path(self.root)
        self.assertNotEqual(active, self.original)
        self.assertEqual(self.records(active), before)
        self.add_trial(active)
        self.assertEqual(self.records(self.original), before)
        self.assertEqual(self.records(self.root / marker["snapshot_path"]), before)
        self.assertEqual(len(self.records(active)["trials"]), 3)

    def test_snapshot_preserves_old_generation_and_publishes_new_records(self):
        """A new durable generation includes both original and newly committed data."""

        previous = hpo_sqlite.enable_local_sqlite(self.root, self.local)
        active = hpo_sqlite.database_path(self.root)
        self.add_trial(active)
        current = hpo_sqlite.snapshot(self.root)
        self.assertEqual(current["generation"], previous["generation"] + 1)
        self.assertEqual(self.records(self.original), self.records(active))
        self.assertEqual(len(self.records(self.root / previous["snapshot_path"])["trials"]), 2)
        self.assertEqual(self.records(self.root / current["snapshot_path"]), self.records(active))

    def test_same_host_reenable_preserves_unpublished_progress(self):
        """A kernel restart never restores an older snapshot over local state."""

        hpo_sqlite.enable_local_sqlite(self.root, self.local)
        active = hpo_sqlite.database_path(self.root)
        self.add_trial(active)
        hpo_sqlite.enable_local_sqlite(self.root, self.local)
        self.assertEqual(len(self.records(hpo_sqlite.database_path(self.root))["trials"]), 3)
        self.assertEqual(self.records(self.original), self.records(active))

    def test_missing_cache_requires_explicit_restore(self):
        """Read paths fail closed; the explicit enable operation restores a snapshot."""

        hpo_sqlite.enable_local_sqlite(self.root, self.local)
        active = hpo_sqlite.database_path(self.root)
        before = self.records(active)
        active.unlink()
        with self.assertRaisesRegex(RuntimeError, "cache is missing"):
            hpo_sqlite.database_path(self.root)
        hpo_sqlite.enable_local_sqlite(self.root, self.local)
        self.assertEqual(self.records(hpo_sqlite.database_path(self.root)), before)

    def test_missing_cache_preserves_residual_local_sidecars(self):
        """A replacement database is never copied beside stale recovery evidence."""

        hpo_sqlite.enable_local_sqlite(self.root, self.local)
        active = hpo_sqlite.database_path(self.root)
        active.unlink()
        for suffix in ("-journal", "-wal", "-shm"):
            with self.subTest(suffix=suffix):
                sidecar = Path(str(active) + suffix)
                sidecar.write_bytes(b"unreconciled local evidence")
                with self.assertRaisesRegex(RuntimeError, "residual local SQLite sidecar"):
                    hpo_sqlite.enable_local_sqlite(self.root, self.local)
                self.assertFalse(active.exists())
                self.assertEqual(sidecar.read_bytes(), b"unreconciled local evidence")
                sidecar.unlink()

    def test_same_host_hot_journal_is_recovered_before_snapshot(self):
        """Real crash recovery rolls back uncommitted local transactions."""

        hpo_sqlite.enable_local_sqlite(self.root, self.local)
        active = hpo_sqlite.database_path(self.root)
        expected = self.records(active)
        with closing(sqlite3.connect(active)) as database:
            database.execute("CREATE TABLE payload_data (id INTEGER PRIMARY KEY, payload BLOB)")
            database.executemany(
                "INSERT INTO payload_data VALUES (?, zeroblob(4096))", 
                [[number] for number in range(200)]
            )
            database.commit()
        script = """import os
import sqlite3
import sys


database = sqlite3.connect(sys.argv[1])
database.execute("PRAGMA journal_mode=DELETE")
database.execute("PRAGMA synchronous=FULL")
database.execute("PRAGMA cache_size=5")
database.execute("BEGIN IMMEDIATE")
database.execute("UPDATE trials SET state='FAIL'")
database.execute("UPDATE payload_data SET payload=randomblob(4096)")
os._exit(0)
"""
        subprocess.run([sys.executable, "-c", script, str(active)], check=True, timeout=30)
        journal = Path(str(active) + "-journal")
        self.assertTrue(journal.is_file())
        with journal.open("rb") as stream:
            self.assertNotEqual(stream.read(8), b"\0" * 8)
        hpo_sqlite.enable_local_sqlite(self.root, self.local)
        self.assertEqual(self.records(active), expected)
        self.assertEqual(self.records(self.original), expected)

    def test_snapshot_rejects_foreign_key_violations(self):
        """Structural page integrity cannot bless orphaned relational records."""

        hpo_sqlite.enable_local_sqlite(self.root, self.local)
        active = hpo_sqlite.database_path(self.root)
        previous = self.marker()
        with closing(sqlite3.connect(active)) as database:
            database.execute("CREATE TABLE child (trial_number INTEGER REFERENCES trials(number))")
            database.execute("INSERT INTO child VALUES (999)")
            database.commit()
        with self.assertRaisesRegex(ValueError, "foreign key check"):
            hpo_sqlite.snapshot(self.root)
        self.assertEqual(self.marker(), previous)

    def test_new_host_requires_explicit_snapshot_restore(self):
        """Container migration uses committed data and retains the former node copy."""

        hpo_sqlite.enable_local_sqlite(self.root, self.local)
        old_active = hpo_sqlite.database_path(self.root)
        before = self.records(old_active)
        with patch.object(hpo_sqlite.socket, "gethostname", return_value="replacement-container"):
            with self.assertRaisesRegex(RuntimeError, "another container"):
                hpo_sqlite.database_path(self.root)
            hpo_sqlite.enable_local_sqlite(self.root, self.base / "replacement-node")
            self.assertEqual(self.records(hpo_sqlite.database_path(self.root)), before)
        self.assertEqual(self.records(old_active), before)

    def test_tampered_committed_snapshot_is_not_restored(self):
        """A plausible compatibility database cannot hide corrupted provenance."""

        marker = hpo_sqlite.enable_local_sqlite(self.root, self.local)
        hpo_sqlite.database_path(self.root).unlink()
        (self.root / marker["snapshot_path"]).write_bytes(b"damaged snapshot")
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            hpo_sqlite.enable_local_sqlite(self.root, self.local)

    def test_marker_cannot_redirect_to_another_study(self):
        """Storage root identity is checked before any active database is returned."""

        marker = hpo_sqlite.enable_local_sqlite(self.root, self.local)
        marker["study_root"] = str(self.base / "another-study")
        (self.root / "sqlite_local.json").write_text(json.dumps(marker), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "another study"):
            hpo_sqlite.database_path(self.root)

    def test_local_cache_identity_is_checked(self):
        """An unrelated database at the same path is not silently adopted."""

        hpo_sqlite.enable_local_sqlite(self.root, self.local)
        active = hpo_sqlite.database_path(self.root)
        (active.parent / "study_identity.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "cache identity differs"):
            hpo_sqlite.database_path(self.root)

    def test_failed_marker_commit_preserves_previous_snapshot_and_local_progress(self):
        """A failed durable commit cannot consume newer node-local records."""

        previous = hpo_sqlite.enable_local_sqlite(self.root, self.local)
        active = hpo_sqlite.database_path(self.root)
        self.add_trial(active)
        with patch.object(hpo_sqlite, "_atomic_json", side_effect=OSError("publication failed")):
            with self.assertRaisesRegex(OSError, "publication failed"):
                hpo_sqlite.snapshot(self.root)
        self.assertEqual(self.marker(), previous)
        self.assertEqual(len(self.records(active)["trials"]), 3)
        self.assertEqual(len(self.records(self.root / previous["snapshot_path"])["trials"]), 2)
        hpo_sqlite.snapshot(self.root)
        self.assertEqual(self.records(self.original), self.records(active))

    def test_failed_compatibility_copy_keeps_new_committed_snapshot_recoverable(self):
        """Host recovery follows the marker even if the compatibility file is stale."""

        hpo_sqlite.enable_local_sqlite(self.root, self.local)
        active = hpo_sqlite.database_path(self.root)
        self.add_trial(active)
        expected = self.records(active)
        original_copy = hpo_sqlite._atomic_copy

        def fail_compatibility(source, target):
            """Inject failure only after the durable snapshot and marker are committed."""

            if target == self.original:
                raise OSError("compatibility copy failed")
            return original_copy(source, target)

        with patch.object(hpo_sqlite, "_atomic_copy", side_effect=fail_compatibility):
            with self.assertRaisesRegex(OSError, "compatibility copy failed"):
                hpo_sqlite.snapshot(self.root)
        self.assertEqual(len(self.records(self.original)["trials"]), 2)
        self.assertEqual(self.records(self.root / self.marker()["snapshot_path"]), expected)
        active.unlink()
        hpo_sqlite.enable_local_sqlite(self.root, self.local)
        self.assertEqual(self.records(hpo_sqlite.database_path(self.root)), expected)

    def test_replacement_source_cannot_overwrite_enabled_study(self):
        """Explicit source input is limited to the first migration."""

        hpo_sqlite.enable_local_sqlite(self.root, self.local)
        with self.assertRaisesRegex(ValueError, "replacement source"):
            hpo_sqlite.enable_local_sqlite(self.root, self.local, source_database=self.original)

    def test_original_hot_journal_requires_preservation_first(self):
        """A pending recovery sidecar cannot be left beside a replacement database."""

        (self.root / "study.db-journal").write_bytes(b"preserve this evidence")
        with self.assertRaisesRegex(RuntimeError, "Preserve and recover"):
            hpo_sqlite.enable_local_sqlite(self.root, self.local)
        self.assertFalse((self.root / "sqlite_local.json").exists())

    def test_snapshot_preserves_shared_sidecar_and_compatibility_database(self):
        """A legacy writer's sidecar blocks compatibility replacement."""

        hpo_sqlite.enable_local_sqlite(self.root, self.local)
        active = hpo_sqlite.database_path(self.root)
        self.add_trial(active)
        original_bytes = self.original.read_bytes()
        sidecar = self.root / "study.db-journal"
        sidecar.write_bytes(b"legacy writer evidence")
        with self.assertRaisesRegex(RuntimeError, "Preserve shared SQLite sidecar"):
            hpo_sqlite.snapshot(self.root)
        self.assertEqual(self.original.read_bytes(), original_bytes)
        self.assertEqual(sidecar.read_bytes(), b"legacy writer evidence")
        self.assertEqual(self.records(self.root / self.marker()["snapshot_path"]), self.records(active))

    def test_context_preserves_primary_failure_if_snapshot_also_fails(self):
        """The scientific operation's original error remains the raised exception."""

        original = ValueError("original worker error")
        with patch.object(hpo_sqlite, "snapshot", side_effect=OSError("shared storage failed")):
            with self.assertRaises(ValueError) as raised:
                with hpo_sqlite.sqlite_snapshots(self.root):
                    raise original
        self.assertIs(raised.exception, original)
        self.assertIn("shared storage failed", " ".join(original.__notes__))

    def test_context_propagates_snapshot_failure_after_success(self):
        """Successful work does not hide inability to publish its durable state."""

        with patch.object(hpo_sqlite, "snapshot", side_effect=OSError("shared storage failed")):
            with self.assertRaisesRegex(OSError, "shared storage failed"):
                with hpo_sqlite.sqlite_snapshots(self.root):
                    pass

    def test_optuna_runner_reads_local_progress_and_migration_preserves_trial_evidence(self):
        """Real runner reads use local state and preserve full Optuna recovery records."""

        try:
            import optuna


        except ImportError:
            self.skipTest("Optuna is required for the runner storage integration check.")
        from common import dit_hpo_runner


        def evidence(study):
            """Serialize all scientifically relevant trial and study metadata."""

            return json.dumps({
                "name": study.study_name, "directions": [direction.name for direction in study.directions], 
                "user_attrs": study.user_attrs, 
                "trials": [{
                    "number": trial.number, "state": trial.state.name, "values": trial.values, 
                    "params": trial.params, "user_attrs": trial.user_attrs, "system_attrs": trial.system_attrs, 
                    "intermediate_values": trial.intermediate_values, 
                    "datetime_start": str(trial.datetime_start), "datetime_complete": str(trial.datetime_complete), 
                    "distributions": {
                        key: optuna.distributions.distribution_to_json(value)
                        for key, value in trial.distributions.items()
                    }
                } for trial in study.get_trials(deepcopy=True)]
            }, sort_keys=True)

        def close_study(study):
            """Dispose test readers before replacing persistent compatibility files."""

            backend = getattr(study._storage, "_backend", study._storage)
            backend.remove_session()
            backend.engine.dispose()

        self.root = self.base / "optuna-study"
        self.root.mkdir()
        self.original = self.root / "study.db"
        plan = {"study_root": str(self.root), "study_name": "sqlite-runner-integration"}
        original = optuna.create_study(
            study_name=plan["study_name"], storage="sqlite:///" + self.original.as_posix(), 
            sampler=optuna.samplers.TPESampler(seed=42), direction="minimize"
        )
        original.set_user_attr("study_spec", {"model": "DiT", "validation_source": "test"})
        rng = original.sampler._rng.rng.get_state()
        original.set_user_attr("sampler_rng_state", {
            "algorithm": rng[0], "keys": rng[1].tolist(), "position": rng[2], 
            "has_gauss": rng[3], "cached_gaussian": rng[4]
        })
        complete = original.ask()
        complete.suggest_categorical("batch_size", [32, 64, 128])
        complete.set_user_attr("results_path", "/preserved/completed-trial")
        complete.report(0.2, step=0)
        original.tell(complete, 0.125)
        running = original.ask()
        running.suggest_float("learning_rate", 0.0001, 0.01, log=True)
        running.set_user_attr("checkpoint_dir", "/preserved/running-trial")
        running.report(0.3, step=1)
        baseline = evidence(original)
        close_study(original)

        hpo_sqlite.enable_local_sqlite(self.root, self.local)
        loaded = dit_hpo_runner._load_study(plan)
        self.assertEqual(evidence(loaded), baseline)
        additional = loaded.ask()
        additional.suggest_categorical("batch_size", [32, 64, 128])
        additional.set_user_attr("seed", 202)
        additional.report(0.12, step=2)
        loaded.tell(additional, 0.0625)
        expected = evidence(loaded)
        close_study(loaded)
        reader = dit_hpo_runner._load_study(plan)
        self.assertEqual(evidence(reader), expected)
        self.assertEqual([trial.state.name for trial in reader.trials], ["COMPLETE", "RUNNING", "COMPLETE"])
        close_study(reader)
        stale = optuna.load_study(
            study_name=plan["study_name"], storage="sqlite:///" + self.original.as_posix()
        )
        self.assertEqual(evidence(stale), baseline)
        close_study(stale)

        hpo_sqlite.snapshot(self.root)
        with patch.object(hpo_sqlite.socket, "gethostname", return_value="replacement-optuna-container"):
            hpo_sqlite.enable_local_sqlite(self.root, self.base / "replacement-optuna-node")
            restored = dit_hpo_runner._load_study(plan)
            try:
                self.assertEqual(evidence(restored), expected)
            finally:
                close_study(restored)


if __name__ == "__main__":
    unittest.main()
