"""Synthetic benchmark study checks; no training or benchmark outcomes."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from common.experiment import materialize_run_plan, read_experiment_manifest
from common.study_artifacts import write_completed_artifact
from semantic_consolidation.config import load_route_config, primary_accuracy_matrix_name
from semantic_consolidation.study import (
    _completed_metrics, _read_study_manifest, analyze_study, prepare_study, validate_planned_config,
)


class BenchmarkStudyTests(unittest.TestCase):
    """Require benchmark source, run identity, and complete saved observations."""

    def setUp(self) -> None:
        """Create a source-bound synthetic study in an isolated directory."""
        temporary = tempfile.TemporaryDirectory(prefix="SYNTHETIC_benchmark_")
        self.addCleanup(temporary.cleanup)
        template = load_route_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")
        self.path = prepare_study(template, Path(temporary.name) / "study", [17, 29], phase="benchmark")
        self.manifest = read_experiment_manifest(self.path)
        self.plan = materialize_run_plan(self.manifest)

    def test_planned_benchmark_uses_test_endpoint_and_cannot_change_phase(self) -> None:
        """Per-run configurations retain benchmark identity and test scoring."""
        config = load_route_config(self.path.parent / f"{self.plan[0]['run_id']}.yaml")
        validate_planned_config(config)
        self.assertEqual(config.common.continually_learn.experiment_phase, "benchmark")
        self.assertEqual(primary_accuracy_matrix_name(config), "ordinary_accuracy_matrix")
        config.common.continually_learn.experiment_phase = "confirmation"
        with self.assertRaisesRegex(ValueError, "differ from the manifest"):
            validate_planned_config(config)
        config.common.continually_learn.experiment_phase = "benchmark"
        config.common.continually_learn.experiment_manifest_path = None
        with self.assertRaisesRegex(ValueError, "frozen paired experiment manifest"):
            validate_planned_config(config)

    def test_benchmark_study_requires_separately_retained_hash(self) -> None:
        """A manifest's own hash cannot authorize benchmark execution or analysis."""
        with self.assertRaisesRegex(ValueError, "externally retained"):
            _read_study_manifest(self.path, None)
        self.assertEqual(_read_study_manifest(self.path, self.manifest["manifest_hash"])["phase"], "benchmark")

    def _outcomes(self, *, include_matrix: bool, include_artifact: bool) -> dict:
        """Create invented complete-stream outcomes with optional required evidence."""
        outcomes = {}
        for entry in self.plan:
            score = .6 + .1 * (entry["condition"] == "learned")
            matrix = [[score, None], [score, score]]
            result = {"fixture_kind": "SYNTHETIC_UNIT_TEST_NOT_THESIS_RESULTS",
                      "manifest_hash": entry["manifest_hash"], "run_id": entry["run_id"],
                      "condition": entry["condition"], "results_path": "SYNTHETIC_ONLY",
                      "metrics": _completed_metrics(matrix, 2)}
            # Matrix omission exercises rejection of unauditable scalar summaries.
            if include_matrix:
                result.update(accuracy_matrix=matrix, accuracy_matrix_source="ordinary_accuracy_matrix")
            # Artifact omission exercises rejection of unhashed completion records.
            if include_artifact:
                result["completed_artifact"] = write_completed_artifact(self.path.parent, result)
            outcomes[entry["run_id"]] = result
        (self.path.parent / "completed_runs.json").write_text(json.dumps(outcomes), encoding="utf-8")
        return outcomes

    def test_benchmark_analysis_requires_hashed_artifacts(self) -> None:
        """Complete scalar values and matrices do not replace authenticated artifacts."""
        self._outcomes(include_matrix=True, include_artifact=False)
        with self.assertRaises(ValueError):
            analyze_study(self.path, expected_hash=self.manifest["manifest_hash"])

    def test_benchmark_analysis_requires_saved_matrix(self) -> None:
        """Even authenticated scalar summaries lack the required task trajectory."""
        self._outcomes(include_matrix=False, include_artifact=True)
        with self.assertRaisesRegex(ValueError, "complete saved accuracy matrix"):
            analyze_study(self.path, expected_hash=self.manifest["manifest_hash"])

    def test_complete_synthetic_analysis_preserves_benchmark_label(self) -> None:
        """Auditable invented results retain benchmark status in native statistics."""
        self._outcomes(include_matrix=True, include_artifact=True)
        result = analyze_study(self.path, expected_hash=self.manifest["manifest_hash"])
        self.assertEqual(result["phase"], "benchmark")
        self.assertEqual(result["pair_count"], 2)
        self.assertAlmostEqual(result["mean_paired_difference"], .1)
