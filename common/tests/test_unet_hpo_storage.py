"""Exercise fresh local UNet storage without training or GPU initialization."""

from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import optuna

from common import dit_hpo_runner as runner
from common.hpo_sqlite import database_path
from common.unet_hpo_storage import is_fresh_storage, prepare_storage


class UNetHpoStorageTests(TestCase):
    """Keep bootstrap identity strict and existing resume semantics unchanged."""

    def setUp(self) -> None:
        """Create an isolated source identity, recipe and local cache directory."""

        temporary = TemporaryDirectory(prefix="unet-storage-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.source = self.checkout / "source.py"
        self.source.write_text("identity = 1\n", encoding="utf-8")
        identity = {
            "source_sha256": {"source.py": hashlib.sha256(self.source.read_bytes()).hexdigest()}, 
            "versions": {}, "python": "test-python", "worker_policy": {"tf_memory_mib": 24576}
        }
        remote = SimpleNamespace(inspect_remote=Mock(return_value=identity))
        with patch.dict("sys.modules", {"common.dit_hpo_remote": remote}):
            self.plan = runner.make_plan(
                self.checkout, self.root / "results", concurrent_trials=2, 
                experiment_hours=12, confirmation_reserve_hours=3, model_name="unet"
            )
        self.study_root = Path(self.plan["study_root"])
        self.control = Path(self.plan["control_root"])
        self.local = self.root / "node-local"

    def open_study(self) -> tuple:
        """Return a local reader whose engine is disposed at test cleanup."""

        storage = optuna.storages.RDBStorage("sqlite:///" + database_path(self.study_root).as_posix())
        self.addCleanup(storage.engine.dispose)
        self.addCleanup(storage.remove_session)
        return optuna.load_study(study_name=self.plan["study_name"], storage=storage), storage

    def test_prepare_is_empty_resumable_and_does_not_start_clock(self) -> None:
        """Setup records local storage and preserves its receipt on rerun."""

        recipe_before = (self.control / "recipe.json").read_bytes()
        source_before = self.source.read_bytes()
        first = prepare_storage(self.plan, self.local)
        receipt_before = (self.control / "local_sqlite_bootstrap.json").read_bytes()
        self.assertTrue(is_fresh_storage(self.plan))
        second = prepare_storage(self.plan, self.local)
        study, storage = self.open_study()
        self.assertEqual(study.trials, [])
        self.assertEqual(study.user_attrs, {})
        self.assertEqual(study.direction.name, "MINIMIZE")
        self.assertEqual(first["cache_id"], second["cache_id"])
        self.assertEqual((self.control / "local_sqlite_bootstrap.json").read_bytes(), receipt_before)
        self.assertEqual((self.control / "recipe.json").read_bytes(), recipe_before)
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertFalse((self.control / "budget.json").exists())
        self.assertFalse((self.study_root / "study_spec.json").exists())
        self.assertTrue(database_path(self.study_root).is_relative_to(self.local))

    def test_existing_unmarked_database_is_preserved(self) -> None:
        """An unrelated persistent database must never be adopted as fresh."""

        database = self.study_root / "study.db"
        database.write_bytes(b"existing evidence")
        with self.assertRaisesRegex(ValueError, "Preserve existing"):
            prepare_storage(self.plan, self.local)
        self.assertEqual(database.read_bytes(), b"existing evidence")
        self.assertFalse((self.study_root / "sqlite_local.json").exists())

    def test_plan_and_source_mismatches_fail_before_storage_creation(self) -> None:
        """Neither a changed model protocol nor source mutation can initialize."""

        invalid = copy.deepcopy(self.plan)
        invalid["hpo"]["use_distillation"] = True
        with self.assertRaisesRegex(ValueError, "teacher-free"):
            prepare_storage(invalid, self.local)
        self.source.write_text("identity = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source identity"):
            prepare_storage(self.plan, self.local)
        self.assertFalse((self.study_root / "study.db").exists())
        self.assertFalse(self.local.exists())

    def test_modified_or_missing_bootstrap_receipt_fails_closed(self) -> None:
        """Local database presence alone cannot authorize an unsealed study."""

        prepare_storage(self.plan, self.local)
        receipt = self.control / "local_sqlite_bootstrap.json"
        value = json.loads(receipt.read_text(encoding="utf-8"))
        value["recipe_sha256"] = "0" * 64
        runner._write(receipt, value)
        with self.assertRaisesRegex(ValueError, "receipt differs"):
            is_fresh_storage(self.plan)
        receipt.unlink()
        with self.assertRaisesRegex(ValueError, "no bootstrap receipt"):
            is_fresh_storage(self.plan)

    def test_unsealed_trials_or_metadata_are_not_fresh(self) -> None:
        """Unexpected scientific state requires investigation rather than recovery."""

        prepare_storage(self.plan, self.local)
        study, storage = self.open_study()
        study.set_user_attr("sampler_rng_state", {"unexpected": True})
        with self.assertRaisesRegex(ValueError, "existing trial or study state"):
            is_fresh_storage(self.plan)
        study.ask()
        with self.assertRaisesRegex(ValueError, "existing trial or study state"):
            is_fresh_storage(self.plan)

    def worker_arguments(self) -> dict:
        """Exercise the real worker dispatch while replacing execution boundaries."""

        factory = Mock()
        remote = SimpleNamespace(
            managed_worker=Mock(), 
            managed_parallel_coordinator=Mock(return_value=nullcontext(factory))
        )
        hpo = Mock()
        request = self.root / "request.json"
        runner._write(request, {
            "plan": self.plan, "payload": {"kind": "search", "allocated_target": 2}, 
            "receipt_path": str(self.root / "worker-receipt.json"), "deadline": None
        })
        with patch.dict("sys.modules", {
            "common.hpo": SimpleNamespace(run_hpo=hpo), "common.dit_hpo_remote": remote
        }), patch.object(runner, "search_summary", return_value={}):
            runner._worker(request)
        hpo.assert_called_once()
        return hpo.call_args.kwargs

    def test_unsealed_system_metadata_is_not_fresh(self) -> None:
        """Use the storage API retained by Optuna 5 to reject unexpected metadata."""

        prepare_storage(self.plan, self.local)
        study, storage = self.open_study()
        study_id = storage.get_study_id_from_name(study.study_name)
        storage.set_study_system_attr(study_id, "unexpected", True)
        with self.assertRaisesRegex(ValueError, "existing trial or study state"):
            is_fresh_storage(self.plan)

    def test_first_worker_uses_ordinary_initialization_then_sealed_resume(self) -> None:
        """Only the authenticated bootstrap omits resume_from on first dispatch."""

        prepare_storage(self.plan, self.local)
        first = self.worker_arguments()
        self.assertNotIn("resume_from", first)
        self.assertEqual(first["model_name"], "unet")
        runner._write(self.study_root / "study_spec.json", {"sealed": True})
        self.assertFalse(is_fresh_storage(self.plan))
        resumed = self.worker_arguments()
        self.assertEqual(resumed["resume_from"], self.plan["study_root"])

    def test_ordinary_unmarked_worker_keeps_strict_resume(self) -> None:
        """An existing ordinary database cannot use the fresh-store exception."""

        (self.study_root / "study.db").write_bytes(b"preserved")
        self.assertFalse(is_fresh_storage(self.plan))
        self.assertEqual(self.worker_arguments()["resume_from"], self.plan["study_root"])

    def test_native_hpo_seals_bootstrap_without_allocating_trials(self) -> None:
        """The real HPO initialization accepts the store before its scheduler runs."""

        from common import hpo


        prepare_storage(self.plan, self.local)
        with patch.object(hpo, "_optimize_concurrently") as scheduler:
            study = hpo.run_hpo(**{**self.plan["hpo"], "n_trials": 1})
        scheduler.assert_called_once()
        self.assertTrue((self.study_root / "study_spec.json").is_file())
        self.assertIn("study_spec", study.user_attrs)
        self.assertEqual(study.trials, [])
        self.assertFalse(is_fresh_storage(self.plan))
        backend = getattr(study._storage, "_backend", study._storage)
        backend.remove_session()
        backend.engine.dispose()

    def test_storage_module_does_not_import_frameworks(self) -> None:
        """A clean notebook coordinator can import storage without TensorFlow."""

        command = (
            "import sys; import common.unet_hpo_storage; "
            "assert 'tensorflow' not in sys.modules; assert 'keras' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", command], cwd=Path(__file__).resolve().parents[2], 
            capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


# Focused execution remains separate from notebook kernels.
if __name__ == "__main__":
    main()
