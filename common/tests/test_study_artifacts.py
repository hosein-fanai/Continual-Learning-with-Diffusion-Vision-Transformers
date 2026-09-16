"""Fresh-checkout source identity and completed-artifact integrity checks."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common.study_artifacts import (
    SOURCE_PACKAGES, read_completed_runs, replace_completed_index,
    source_files, source_fingerprint, validate_completed_artifact,
    validate_study_source, write_completed_artifact,
)


class StudyArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for package in SOURCE_PACKAGES:
            directory = self.root / package
            directory.mkdir()
            (directory / "module.py").write_text("value = 1\n", encoding="utf-8")

    def test_fresh_checkout_scope_ignores_optional_routes_and_generated_snapshots(self):
        expected = source_fingerprint(self.root)
        for relative in ("gist_memory/local.py", "allocation_study/prepared/copy.py",
                         "common/tests/fixture.py", "semantic_consolidation/prepared/copy.py"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("unrelated = 2\n", encoding="utf-8")
        self.assertEqual(expected, source_fingerprint(self.root))
        (self.root / "common/module.py").write_text("value = 2\n", encoding="utf-8")
        self.assertNotEqual(expected, source_fingerprint(self.root))

    def test_requested_optional_source_is_required_and_excludes_snapshots(self):
        with self.assertRaisesRegex(ValueError, "gist_memory"):
            source_files(self.root, additional_packages=("gist_memory",))
        (self.root / "gist_memory/prepared").mkdir(parents=True)
        (self.root / "gist_memory/module.py").write_text("value = 3\n", encoding="utf-8")
        (self.root / "gist_memory/prepared/copy.py").write_text("value = 4\n", encoding="utf-8")
        result = source_files(self.root, additional_packages=("gist_memory",))
        self.assertIn("gist_memory/module.py", result)
        self.assertNotIn("gist_memory/prepared/copy.py", result)

    def test_both_dependency_manifests_change_identity(self):
        previous = source_fingerprint(self.root)
        for name in ("requirements.txt", "requirements-colab.txt"):
            (self.root / name).write_text("numpy==2.3.2\n", encoding="utf-8")
            current = source_fingerprint(self.root)
            self.assertNotEqual(previous, current)
            self.assertIn(name, current["files"])
            previous = current

    def test_frozen_source_changes_are_still_rejected(self):
        expected = source_fingerprint(self.root)
        manifest = {"phase": "confirmation", "spec": {"analysis_spec": {
            "native_route_study": {"schema_version": 1, "route": "semantic_consolidation",
                                   "source": expected}}}}
        with patch("common.study_artifacts.source_fingerprint", return_value=expected):
            self.assertTrue(validate_study_source(manifest, "semantic_consolidation")["verified"])
        changed = deepcopy(expected)
        changed["sha256"] = "changed"
        with patch("common.study_artifacts.source_fingerprint", return_value=changed):
            with self.assertRaisesRegex(ValueError, "Source fingerprint differs"):
                validate_study_source(manifest, "semantic_consolidation")

    def test_completion_round_trip_rejects_changed_endpoint(self):
        record = {"run_id": "synthetic-only", "accuracy": .5}
        record["completed_artifact"] = write_completed_artifact(self.root, record)
        self.assertTrue(validate_completed_artifact(self.root, record, required=True))
        record["accuracy"] = .9
        with self.assertRaisesRegex(ValueError, "index differs"):
            validate_completed_artifact(self.root, record, required=True)

    def test_failed_index_replacement_preserves_previous_file(self):
        path = self.root / "completed_runs.json"
        original = {"synthetic": {"accuracy": .5}}
        path.write_text(json.dumps(original), encoding="utf-8")
        with patch("common.study_artifacts.os.replace", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                replace_completed_index(path, {"different": {"accuracy": .9}})
        self.assertEqual(original, read_completed_runs(path))
        self.assertEqual([path], list(self.root.glob("*.json")))
        self.assertFalse(list(self.root.glob("*.pending")))


if __name__ == "__main__":
    unittest.main()
