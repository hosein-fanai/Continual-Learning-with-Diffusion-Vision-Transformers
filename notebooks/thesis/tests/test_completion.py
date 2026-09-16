"""Synthetic bookkeeping fixtures only: no CIFAR data, training or thesis results."""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from common.study_artifacts import replace_completed_index, write_completed_artifact
from common.config import save_config
from common.continual_reporting import write_continual_csv_artifacts
from common.experiment import materialize_run_plan, read_experiment_manifest
from notebooks.thesis import completion, workflow
from semantic_consolidation.config import load_route_config, save_route_settings
from semantic_consolidation.provenance import save_provenance, source_provenance
from semantic_consolidation.study import _completed_metrics, prepare_study


class CompletionRecoveryTests(unittest.TestCase):
    """Exercise real plan/artifact validators with tiny saved-result fixtures."""

    def setUp(self) -> None:
        """Prepare isolated synthetic fixtures.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-completion-validation-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        config = load_route_config(Path(__file__).resolve().parents[1] / "configs/cifar10.yaml")
        config.common.continually_learn.class_num = 4
        config.common.continually_learn.class_order = [0, 1, 2, 3]
        config.common.continually_learn.task_groups = [[0, 1], [2, 3]]
        self.manifest_path = prepare_study(config, self.directory / "study", [1103, 2207],
                                          conditions={"learned": {"route": {"condition": "learned"}},
                                                      "extra_joint": {"route": {"condition": "extra_joint"}}},
                                          phase="confirmation")
        self.manifest = read_experiment_manifest(self.manifest_path)
        self.digest = self.manifest["manifest_hash"]
        self.entries = materialize_run_plan(self.manifest)
        self.entry = next(entry for entry in self.entries if entry["condition"] == "learned")
        self.record = self.native_fixture(self.entry)
        self.index = self.manifest_path.parent / "completed_runs.json"

    def native_fixture(self, entry: dict) -> dict:
        """Exercise the existing native artifact operation with the enclosing test fixture.

        Args:
            entry (dict): Fixture value supplied by the enclosing regression; existing native
                validators interpret its fields.

        Returns:
            result (dict): Synthetic fixture or native artifact result used only by the
                enclosing assertions.

        Raises:
            AssertionError: If the stated regression invariant fails.
            OSError: If a required temporary fixture cannot be read or written.
        """
        config = load_route_config(self.manifest_path.parent / f"{entry['run_id']}.yaml")
        result_path = self.directory / "synthetic-native-results" / entry["run_id"]
        result_path.mkdir(parents=True)
        config.common.training.results_path = str(result_path)
        config.common.continually_learn.checkpoint_dir = str(self.manifest_path.parent / "checkpoints" / entry["run_id"])
        config.common.dataset.trainset_len = 8
        # Apply this case only when config.common.optimizer.schedule == 'cosine' and
        # config.common.optimizer.decay_steps is None.
        if config.common.optimizer.schedule == "cosine" and config.common.optimizer.decay_steps is None:
            config.common.optimizer.decay_steps = config.common.training.epochs * 8
        config.common.hpo["semantic_consolidation"] = asdict(config.route)
        config.common.hpo["input_config_path"] = str(result_path / "input_config.yaml")
        save_config(config.common, result_path / "input_config.yaml")
        # Apply the actual common.train inference rewrites to the common snapshot.
        config.common.hpo["schedule_request"] = {
            "task_size": config.common.continually_learn.task_size,
            "class_order_mode": "fixed", "task_order_mode": "fixed", "seed": entry["stream"]["stream_seed"]}
        config.common.model.kwargs["num_classes"] = 4
        config.common.model.wrapper_kwargs["seen_classes"] = {index: index for index in range(4)}
        # Apply this case only when config.common.training.save_weights.
        if config.common.training.save_weights:
            replay = result_path / "replay-model.weights.h5"
            classifier = result_path / "model.weights.h5"
            replay.write_bytes(b"synthetic fixture, not model weights")
            classifier.write_bytes(b"synthetic fixture, not model weights")
            config.common.model.weights_path = str(replay)
            config.common.hpo["classifier_weights_path"] = str(classifier)
        save_config(config.common, result_path / "config.yaml")
        save_route_settings(config.route, result_path / "route.settings.yaml")
        save_provenance(source_provenance(), result_path)
        matrix = np.asarray([[0.7, np.nan], [0.6, 0.8]])
        metrics = _completed_metrics(matrix, 2)
        write_continual_csv_artifacts({
            "ordinary_accuracy_matrix": matrix, "continual_metrics": metrics,
            "class_order": entry["stream"]["class_order"], "task_classes": entry["stream"]["task_groups"],
            "seed": entry["stream"]["stream_seed"]}, result_path)
        (result_path / "route_metrics.json").write_text(json.dumps([
            {"task": index + 1, "condition": config.route.condition, "total_updates": 5}
            for index in range(2)]), encoding="utf-8")
        return {"manifest_hash": self.digest, "run_id": entry["run_id"], "condition": entry["condition"],
                "results_path": str(result_path), "seconds": 1.25, "total_updates": 10,
                "started_utc": "2026-09-14T00:00:00+00:00",
                "accuracy_matrix": [[0.7, None], [0.6, 0.8]],
                "accuracy_matrix_source": "ordinary_accuracy_matrix", "metrics": metrics}

    def artifact(self, record: dict | None=None) -> dict:
        """Exercise the existing native artifact operation with the enclosing test fixture.

        Args:
            record (dict | None): Fixture value supplied by the enclosing regression; existing
                native validators interpret its fields.

        Returns:
            result (dict): Synthetic fixture or native artifact result used only by the
                enclosing assertions.

        Raises:
            AssertionError: If the stated regression invariant fails.
            OSError: If a required temporary fixture cannot be read or written.
        """
        record = record or self.record
        return write_completed_artifact(self.manifest_path.parent, record)

    def reconcile(self) -> dict:
        """Exercise the existing native artifact operation with the enclosing test fixture.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (dict): Synthetic fixture or native artifact result used only by the
                enclosing assertions.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        return completion.reconcile_completions(self.manifest_path, expected_hash=self.digest)

    def test_failure_after_artifact_then_recovery_without_training(self) -> None:
        """Verify failure after artifact then recovery without training.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        replace_completed_index(self.index, {})
        with patch.object(completion, "replace_completed_index", side_effect=OSError("injected index failure")):
            with self.assertRaisesRegex(OSError, "injected index failure"):
                completion.publish_completion(self.manifest_path, self.record, expected_hash=self.digest)
        path = self.manifest_path.parent / f"{self.entry['run_id']}.completed.json"
        before = path.read_bytes()
        self.assertEqual(json.loads(self.index.read_text()), {})
        with patch("semantic_consolidation.runner.run", side_effect=AssertionError("training is forbidden")) as train:
            outputs = self.reconcile()
            train.assert_not_called()
        self.assertEqual(outputs[self.entry["run_id"]]["metrics"], self.record["metrics"])
        self.assertEqual(path.read_bytes(), before)
        receipt = json.loads((self.manifest_path.parent / f"{self.entry['run_id']}.recovered.json").read_text())
        self.assertFalse(receipt["training_called"])

    def test_repeated_recovery_and_matching_publication_are_noops(self) -> None:
        """Verify repeated recovery and matching publication are noops.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        self.artifact()
        self.reconcile()
        before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in self.manifest_path.parent.glob("*.json")}
        with patch.object(completion, "replace_completed_index", side_effect=AssertionError("unexpected write")):
            for _ in range(2):
                self.assertEqual(len(self.reconcile()), 1)
            completion.publish_completion(self.manifest_path, self.record, expected_hash=self.digest)
        after = {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                 for path in self.manifest_path.parent.glob("*.json")}
        self.assertEqual(before, after)

    def test_already_indexed_valid_completion_is_noop(self) -> None:
        """Verify already indexed valid completion is noop.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        descriptor = self.artifact()
        replace_completed_index(self.index, {self.entry["run_id"]: {**self.record, "completed_artifact": descriptor}})
        with patch.object(completion, "replace_completed_index", side_effect=AssertionError("unexpected write")):
            self.assertEqual(len(self.reconcile()), 1)
        self.assertFalse(list(self.manifest_path.parent.glob("*.recovered.json")))

    def test_corrupt_artifact_does_not_rewrite_previous_index(self) -> None:
        """Verify corrupt artifact does not rewrite previous index.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        descriptor = self.artifact()
        replace_completed_index(self.index, {self.entry["run_id"]: {**self.record, "completed_artifact": descriptor}})
        previous = self.index.read_bytes()
        (self.manifest_path.parent / descriptor["path"]).write_text("{bad-json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            self.reconcile()
        self.assertEqual(self.index.read_bytes(), previous)

    def test_conflicting_completion_submission_preserves_evidence(self) -> None:
        """Verify conflicting completion submission preserves evidence.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        completion.publish_completion(self.manifest_path, self.record, expected_hash=self.digest)
        previous = self.index.read_bytes()
        with self.assertRaisesRegex(ValueError, "Conflicting completion"):
            completion.publish_completion(self.manifest_path, {**self.record, "seconds": 999.}, expected_hash=self.digest)
        self.assertEqual(self.index.read_bytes(), previous)

    def test_concurrent_publications_retain_both_validated_records(self) -> None:
        """Separate publishers cannot overwrite each other's completed index entry.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        other = self.native_fixture(next(entry for entry in self.entries if entry != self.entry))
        barrier = threading.Barrier(2)

        def slow_replace(path: Path, values: dict) -> None:
            """Exercise the existing native artifact operation with the enclosing test fixture.

            Args:
                path (Path): Temporary artifact location owned by this test.
                values (dict): Fixture value supplied by the enclosing regression; existing
                    native validators interpret its fields.

            Returns:
                result (None): None; unittest records the assertions and any failure.

            Raises:
                AssertionError: If the stated regression invariant fails.
                OSError: If a required temporary fixture cannot be read or written.
            """
            time.sleep(0.05)  # Expose the old unprotected read/modify/write interval.
            replace_completed_index(path, values)

        def publish(record: dict) -> dict:
            """Exercise the existing native artifact operation with the enclosing test fixture.

            Args:
                record (dict): Fixture value supplied by the enclosing regression;
                    existing native validators interpret its fields.

            Returns:
                result (dict): Authenticated native completion record from publication.

            Raises:
                AssertionError: If the stated regression invariant fails.
                OSError: If a required temporary fixture cannot be read or written.
            """
            barrier.wait(timeout=30)
            return completion.publish_completion(self.manifest_path, record, expected_hash=self.digest)

        with patch.object(completion, "replace_completed_index", side_effect=slow_replace):
            with ThreadPoolExecutor(max_workers=2) as executor:
                records = list(executor.map(publish, (self.record, other)))
        self.assertEqual(set(json.loads(self.index.read_text())), {record["run_id"] for record in records})
        self.assertEqual(len(self.reconcile()), 2)

    def test_incomplete_or_mismatched_completion_matrices(self) -> None:
        """Verify incomplete or mismatched completion matrices.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        invalid = ([[0.7]], [[0.7, None], [None, 0.8]], [[0.7, None], [0.2, 0.8]],
                   [[0.7, 0.4], [0.6, 0.8]])
        for matrix in invalid:
            with self.subTest(matrix=matrix):
                value = {**self.record, "accuracy_matrix": matrix}
                path = self.manifest_path.parent / f"{self.entry['run_id']}.completed.json"
                self.artifact(value)
                try:
                    with self.assertRaises(ValueError):
                        self.reconcile()
                    self.assertFalse(self.index.exists())
                finally:
                    path.unlink()  # Remove only this deliberately invalid temporary test fixture.

    def test_native_matrix_mismatch_rejected_even_with_matching_summary(self) -> None:
        """Verify native matrix mismatch rejected even with matching summary.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        value = deepcopy(self.record)
        value["accuracy_matrix"][1][0] = 0.3
        value["metrics"] = _completed_metrics(value["accuracy_matrix"], 2)
        self.artifact(value)
        with self.assertRaisesRegex(ValueError, "Native ordinary matrix differs"):
            self.reconcile()

    def test_missing_native_matrix_cell_is_incomplete(self) -> None:
        """Verify missing native matrix cell is incomplete.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        self.artifact()
        matrix_path = Path(self.record["results_path"]) / "accuracy_matrices.csv"
        lines = matrix_path.read_text().splitlines()
        matrix_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Native ordinary matrix is incomplete"):
            self.reconcile()
        self.assertFalse(self.index.exists())

    def test_failed_recovery_index_replacement_can_be_retried(self) -> None:
        """Verify failed recovery index replacement can be retried.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        self.artifact()
        replace_completed_index(self.index, {})
        before = self.index.read_bytes()
        with patch.object(completion, "replace_completed_index", side_effect=OSError("recovery replace failed")):
            with self.assertRaisesRegex(OSError, "recovery replace failed"):
                self.reconcile()
        self.assertEqual(self.index.read_bytes(), before)
        receipt = self.manifest_path.parent / f"{self.entry['run_id']}.recovered.json"
        before_receipt = receipt.read_bytes()
        self.assertEqual(len(self.reconcile()), 1)
        self.assertEqual(receipt.read_bytes(), before_receipt)

    def test_native_configuration_and_provenance_mismatches_are_not_repaired(self) -> None:
        """Verify native configuration and provenance mismatches are not repaired.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        self.artifact()
        result = Path(self.record["results_path"])
        settings = result / "route.settings.yaml"
        original_settings = settings.read_bytes()
        settings.write_text("condition: random\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "planned intervention"):
            self.reconcile()
        settings.write_bytes(original_settings)
        provenance_path = result / "source_provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["source_sha256"] = "0" * 64
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source provenance"):
            self.reconcile()
        self.assertFalse(self.index.exists())

    def test_saved_inference_scientific_change_is_rejected(self) -> None:
        """Verify saved inference scientific change is rejected.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        self.artifact()
        config_path = Path(self.record["results_path"]) / "config.yaml"
        config = completion.load_config(config_path)
        config.optimizer.initial_learning_rate *= 2
        save_config(config, config_path)
        with self.assertRaisesRegex(ValueError, "scientific settings differ"):
            self.reconcile()

    def test_started_receipt_with_missing_identity_is_rejected_without_training(self) -> None:
        """Verify started receipt with missing identity is rejected without training.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        marker = self.manifest_path.parent / f"{self.entry['run_id']}.started.json"
        marker.write_text(json.dumps({"run_id": self.entry["run_id"]}), encoding="utf-8")
        self.assertEqual(self.reconcile(), {})
        record = {"schema_version": 1, "studies": {"cifar10": {"manifest_path": str(self.manifest_path)}}}
        with patch.object(workflow, "_campaign", return_value=(record, {"cifar10": self.manifest})), \
                patch.object(workflow, "_initialize", side_effect=AssertionError("training initialization forbidden")):
            with self.assertRaisesRegex(ValueError, "receipt differs"):
                workflow.load_run(self.directory / "frozen_design.json", "cifar10", "learned", repeat_index=None)
        self.assertTrue(marker.exists())
        self.assertFalse(self.index.exists())

    def test_valid_started_receipt_without_checkpoint_retries_same_stream(self) -> None:
        """Select the same failed stream without manufacturing completed evidence.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        marker = self.manifest_path.parent / f"{self.entry['run_id']}.started.json"
        marker.write_text(json.dumps({"run_id": self.entry["run_id"], "manifest_hash": self.digest}), encoding="utf-8")
        before = marker.read_bytes()
        record = {"schema_version": 1, "studies": {"cifar10": {"manifest_path": str(self.manifest_path)}}}
        with patch.object(workflow, "_campaign", return_value=(record, {"cifar10": self.manifest})), \
                patch.object(workflow, "_initialize", side_effect=lambda config, context: (config, context)):
            config, context = workflow.load_run(self.directory / "frozen_design.json", "cifar10", "learned", repeat_index=None)
        self.assertEqual(context["entry"]["run_id"], self.entry["run_id"])
        self.assertIsNone(config.common.continually_learn.resume_from)
        self.assertEqual(marker.read_bytes(), before)
        self.assertFalse(self.index.exists())

    def test_selection_recovers_completed_started_run_before_choosing_next(self) -> None:
        """Verify selection recovers completed started run before choosing next.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        self.artifact()
        (self.manifest_path.parent / f"{self.entry['run_id']}.started.json").write_text("{}", encoding="utf-8")
        record = {"schema_version": 1, "studies": {"cifar10": {"manifest_path": str(self.manifest_path)}}}
        with patch.object(workflow, "_campaign", return_value=(record, {"cifar10": self.manifest})), \
                patch.object(workflow, "_initialize", side_effect=lambda config, context: (config, context)):
            _, context = workflow.load_run(self.directory / "frozen_design.json", "cifar10", "learned", repeat_index=None)
        self.assertNotEqual(context["entry"]["run_id"], self.entry["run_id"])
        self.assertIn(self.entry["run_id"], json.loads(self.index.read_text()))

    def test_final_analysis_reconciles_before_rejecting_partial_campaign(self) -> None:
        """Verify final analysis reconciles before rejecting partial campaign.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        self.artifact()
        with self.assertRaisesRegex(ValueError, "finish every planned stream"):
            workflow._outputs(self.manifest_path, self.manifest, complete=True)
        self.assertIn(self.entry["run_id"], json.loads(self.index.read_text()))

    def test_finish_retry_recovers_without_reporting_or_rewriting_native_results(self) -> None:
        """Verify finish retry recovers without reporting or rewriting native results.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        self.artifact()
        config = load_route_config(self.manifest_path.parent / f"{self.entry['run_id']}.yaml")
        config.common.training.results_path = self.record["results_path"]
        controller = Mock()
        context = {"record_path": self.directory / "frozen_design.json", "entry": self.entry,
                   "manifest_path": self.manifest_path, "started_utc": self.record["started_utc"],
                   "finished": False, "controller": controller, "config": config}
        result_path = Path(self.record["results_path"])
        before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in result_path.iterdir()}
        report = Mock(side_effect=AssertionError("report/evaluation is forbidden on retry"))
        with patch.object(workflow, "_check_context"), \
                patch.dict("sys.modules", {"common.train": SimpleNamespace(report=report)}):
            for _ in range(2):
                result = workflow.finish_run(context, config, None, None, None, None)
                self.assertEqual(result, self.record["metrics"])
        after = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in result_path.iterdir()}
        self.assertEqual(before, after)
        self.assertTrue(context["finished"])
        controller.save.assert_not_called()
        report.assert_not_called()

    def test_frozen_hash_mismatch_never_reconstructs_an_index(self) -> None:
        """Verify frozen hash mismatch never reconstructs an index.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        self.artifact()
        with self.assertRaises(ValueError):
            completion.reconcile_completions(self.manifest_path, expected_hash="0" * 64)
        self.assertFalse(self.index.exists())

    def test_failed_atomic_index_replacement_preserves_previous_index(self) -> None:
        """Verify failed atomic index replacement preserves previous index.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        replace_completed_index(self.index, {})
        previous = self.index.read_bytes()
        with patch("common.study_artifacts.os.replace", side_effect=OSError("injected replacement failure")):
            with self.assertRaisesRegex(OSError, "injected replacement failure"):
                completion.publish_completion(self.manifest_path, self.record, expected_hash=self.digest)
        self.assertEqual(self.index.read_bytes(), previous)
        self.assertFalse(list(self.manifest_path.parent.glob("*.pending")))
        self.assertEqual(len(self.reconcile()), 1)

    def test_duplicate_keys_and_foreign_identity_rejected(self) -> None:
        """Verify duplicate keys and foreign identity rejected.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        path = self.manifest_path.parent / f"{self.entry['run_id']}.completed.json"
        path.write_text('{"run_id":"one","run_id":"two"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            self.reconcile()
        path.write_text(json.dumps({**self.record, "condition": "random"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "foreign run, condition"):
            self.reconcile()
        self.assertFalse(self.index.exists())

        # The shared native YAML loader must retain duplicate-key rejection
        # when completion bookkeeping reads the separately saved route settings.
        path.write_text(json.dumps(self.record), encoding="utf-8")
        route_path = Path(self.record["results_path"]) / "route.settings.yaml"
        original_route = route_path.read_bytes()
        route_path.write_bytes(original_route + b"\nlearning_rate: 0.001\n")
        evidence = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "duplicate key.*learning_rate"):
            self.reconcile()
        self.assertEqual(path.read_bytes(), evidence)
        self.assertFalse(self.index.exists())
        self.assertFalse(list(self.manifest_path.parent.glob("*.recovered.json")))


class PortableCampaignTests(unittest.TestCase):
    """Bounded saved-evidence regression fixtures; never research outcomes."""

    def test_frozen_plan_relocation_preserves_hashes_and_scientific_validation(self) -> None:
        """Moving a fresh plan changes locators, never frozen bytes or settings.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        notebook_dir = Path(workflow.__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="synthetic-portable-plan-") as temporary:
            directory = Path(temporary)
            original = workflow.prepare_campaign(directory / "original", {
                dataset: notebook_dir / "configs" / f"{dataset}.yaml"
                for dataset in workflow.CONDITIONS}, workflow.CONFIRMATION_SEEDS)
            frozen = original.read_bytes()
            shutil.copytree(original.parent, directory / "moved")
            moved = directory / "moved" / original.name
            with patch.object(workflow, "_initialize", side_effect=lambda config, context: (config, context)):
                config, context = workflow.load_run(moved, "cifar10", "learned")
            self.assertEqual(moved.read_bytes(), frozen)
            self.assertEqual(config.common.continually_learn.experiment_manifest_path,
                             str(moved.parent / "cifar10" / "manifest.json"))
            self.assertEqual(config.common.training.results_path, str(moved.parent / "cifar10" / "runs"))
            source_yaml = original.parent / "cifar10" / context["config_path"].name
            self.assertEqual(source_yaml.read_bytes(), context["config_path"].read_bytes())
            # A second source checkout resolves bindings relative to its own root.
            record = json.loads(frozen)
            source_root = notebook_dir.parents[1]
            relocated_root = directory / "other-checkout"
            for filename in record["bound_files"]:
                destination = relocated_root / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_root / filename, destination)
            with patch.object(workflow, "__file__", str(relocated_root / "notebooks/thesis/workflow.py")):
                workflow._campaign(moved)
            changed = load_route_config(context["config_path"])
            changed.route.alignment_weight *= 2
            import yaml
            context["config_path"].write_text(yaml.safe_dump(asdict(changed)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differ from the manifest"):
                workflow.load_run(moved, "cifar10", "learned")


# Run this isolated unittest module when invoked as a script.
if __name__ == "__main__":
    unittest.main()
