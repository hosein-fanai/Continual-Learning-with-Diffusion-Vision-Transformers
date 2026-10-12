"""Run semantic SQLite bootstrap checks only on an authorized remote container."""

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import optuna

from common import dit_hpo_runner as runner
from common.hpo_sqlite import database_path
from common.semantic_hpo_candidates import semantic_candidate_profile
from common.semantic_hpo_storage import is_fresh_storage, prepare_storage


class SemanticHpoStorageTests(TestCase):
    """Preserve durable provenance while keeping transactional SQLite host-local."""

    def setUp(self) -> None:
        """Prepare a real native profile with mocked remote hardware admission."""

        temporary = TemporaryDirectory(prefix="semantic-storage-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.source = self.checkout / "source.py"
        self.source.write_text("identity = 1\n", encoding="utf-8")
        identity = {
            "source_sha256": {"source.py": hashlib.sha256(self.source.read_bytes()).hexdigest()}, 
            "versions": {}, "python": "test-python", "hostname": socket.gethostname(), 
            "worker_policy": {"tf_memory_mib": 24576}
        }
        self.remote = SimpleNamespace(inspect_remote=Mock(return_value=identity))
        with patch.dict("sys.modules", {"common.dit_hpo_remote": self.remote}):
            self.plan = runner.make_plan(
                self.checkout, self.root / "results", epochs=10, concurrent_trials=2, 
                validation_source="split", validation_ratio=0.1, 
                experiment_hours=12, confirmation_reserve_hours=3, 
                search_profile="semantic_consolidation_runner", semantic_profile=semantic_candidate_profile()
            )
        self.study_root = Path(self.plan["study_root"])
        self.control = Path(self.plan["control_root"])
        self.local = self.root / "node-local"

    def open_study(self) -> tuple:
        """Open the authoritative local cache with deterministic resource cleanup."""

        storage = optuna.storages.RDBStorage("sqlite:///" + database_path(self.study_root).as_posix())
        self.addCleanup(storage.engine.dispose)
        self.addCleanup(storage.remove_session)
        return optuna.load_study(study_name=self.plan["study_name"], storage=storage), storage

    def test_prepare_is_empty_repeatable_and_does_not_start_campaign_clock(self) -> None:
        """Setup creates only an authenticated cache and durable snapshots."""

        recipe_before = (self.control / "recipe.json").read_bytes()
        first = prepare_storage(self.plan, self.local)
        receipt_before = (self.control / "local_sqlite_bootstrap.json").read_bytes()
        self.assertTrue(is_fresh_storage(self.plan))
        second = prepare_storage(self.plan, self.local)
        study, storage = self.open_study()
        self.assertEqual(study.trials, [])
        self.assertEqual(study.user_attrs, {})
        self.assertEqual(study.direction.name, "MAXIMIZE")
        self.assertEqual(first["cache_id"], second["cache_id"])
        self.assertEqual((self.control / "local_sqlite_bootstrap.json").read_bytes(), receipt_before)
        self.assertEqual((self.control / "recipe.json").read_bytes(), recipe_before)
        self.assertFalse((self.control / "budget.json").exists())
        self.assertFalse((self.study_root / "study_spec.json").exists())
        self.assertTrue(database_path(self.study_root).is_relative_to(self.local))
        self.assertTrue((self.study_root / first["snapshot_path"]).is_file())

    def test_existing_unmarked_database_and_overlapping_cache_are_preserved(self) -> None:
        """Preparation cannot adopt existing evidence or place transactions in outputs."""

        with self.assertRaisesRegex(ValueError, "separate"):
            prepare_storage(self.plan, self.study_root / "cache")
        database = self.study_root / "study.db"
        database.write_bytes(b"existing evidence")
        with self.assertRaisesRegex(ValueError, "Preserve existing"):
            prepare_storage(self.plan, self.local)
        self.assertEqual(database.read_bytes(), b"existing evidence")
        self.assertFalse((self.study_root / "sqlite_local.json").exists())

    def test_protocol_and_source_changes_fail_before_storage_creation(self) -> None:
        """Only the authenticated validation objective and source can bootstrap."""

        invalid = deepcopy(self.plan)
        invalid["hpo"]["semantic_profile"]["student_config"]["dataset"]["validation_source"] = "test"
        with self.assertRaisesRegex(ValueError, "raw full-stream validation"):
            prepare_storage(invalid, self.local)
        invalid = deepcopy(self.plan)
        invalid["hpo"]["objective_metrics"] = ["final_average_accuracy"]
        with self.assertRaisesRegex(ValueError, "raw full-stream validation"):
            prepare_storage(invalid, self.local)
        self.source.write_text("identity = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source identity"):
            prepare_storage(self.plan, self.local)
        self.assertFalse(self.local.exists())

    def test_modified_or_missing_bootstrap_receipt_fails_closed(self) -> None:
        """Database presence alone does not authorize a fresh Optuna initialization."""

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

    def test_unsealed_trial_and_user_metadata_are_not_fresh(self) -> None:
        """Partial scientific state requires strict recovery or explicit investigation."""

        prepare_storage(self.plan, self.local)
        study, storage = self.open_study()
        study.set_user_attr("unexpected", True)
        with self.assertRaisesRegex(ValueError, "existing trial or study state"):
            is_fresh_storage(self.plan)
        study.ask()
        with self.assertRaisesRegex(ValueError, "existing trial or study state"):
            is_fresh_storage(self.plan)

    def test_unsealed_system_metadata_is_not_fresh(self) -> None:
        """System attributes cannot hide partially initialized scientific state."""

        prepare_storage(self.plan, self.local)
        study, storage = self.open_study()
        storage.set_study_system_attr(storage.get_study_id_from_name(study.study_name), "unexpected", True)
        with self.assertRaisesRegex(ValueError, "existing trial or study state"):
            is_fresh_storage(self.plan)

    def worker_arguments(self) -> dict:
        """Read actual worker dispatch options with training and hardware calls mocked."""

        factory = Mock()
        remote = SimpleNamespace(
            managed_worker=Mock(), managed_parallel_coordinator=Mock(return_value=nullcontext(factory))
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

    def test_first_worker_bootstraps_then_sealed_study_uses_strict_resume(self) -> None:
        """The fresh-store exception ends when ordinary HPO publishes its specification."""

        prepare_storage(self.plan, self.local)
        self.assertNotIn("resume_from", self.worker_arguments())
        runner._write(self.study_root / "study_spec.json", {"sealed": True})
        self.assertFalse(is_fresh_storage(self.plan))
        self.assertEqual(self.worker_arguments()["resume_from"], self.plan["study_root"])

    def test_different_shards_receive_distinct_local_databases(self) -> None:
        """Sharing a cache parent cannot cause independent studies to share transactions."""

        first = prepare_storage(self.plan, self.local)
        with patch.dict("sys.modules", {"common.dit_hpo_remote": self.remote}):
            second_plan = runner.make_plan(
                self.checkout, self.root / "results-second", epochs=10, concurrent_trials=2, 
                validation_source="split", validation_ratio=0.1, 
                experiment_hours=12, confirmation_reserve_hours=3, 
                search_profile="semantic_consolidation_runner", semantic_profile=semantic_candidate_profile()
            )
        second = prepare_storage(second_plan, self.local)
        self.assertNotEqual(first["database_path"], second["database_path"])
        self.assertNotEqual(first["cache_id"], second["cache_id"])
        self.assertTrue(is_fresh_storage(self.plan))
        self.assertTrue(is_fresh_storage(second_plan))
