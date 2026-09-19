"""Synthetic checks for authenticated, explicitly test-informed benchmarks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from common.experiment import (
    collect_final_stream_metrics, create_paired_block_manifest, materialize_run_plan,
    paired_run_statistics, validate_frozen_confirmation, validate_frozen_experiment,
    write_experiment_manifest,
)
from common.learner import _run_continual_tasks
from common.study_artifacts import validate_study_source


class BenchmarkExperimentTests(unittest.TestCase):
    """Keep frozen execution guarantees separate from test-set independence."""

    @staticmethod
    def _manifest(phase: str = "benchmark") -> dict:
        """Build two complete synthetic paired streams without loading data."""
        return create_paired_block_manifest(
            {"learned": {}, "extra_joint": {}},
            [{"block_id": "stream-a", "stream_seed": 43,
              "class_order": [0, 1], "task_groups": [[0], [1]]},
             {"block_id": "stream-b", "stream_seed": 47,
              "class_order": [1, 0], "task_groups": [[1], [0]]}],
            seed=17, phase=phase,
            analysis_spec={"condition_a": "learned", "condition_b": "extra_joint",
                           "primary_metric": "final_average_accuracy"},
            base_config={"selection_provenance": {"official_test_used": True}},
        )

    @staticmethod
    def _values(manifest: dict) -> dict:
        """Assign invented stream outcomes solely to exercise analysis guards."""
        return {run["run_id"]: .5 + .1 * (run["condition"] == "learned")
                for run in materialize_run_plan(manifest)}

    def test_benchmark_is_frozen_but_never_confirmation(self) -> None:
        """Authentication preserves the explicit benchmark phase and provenance."""
        manifest = self._manifest()
        checked = validate_frozen_experiment(manifest, expected_hash=manifest["manifest_hash"])
        self.assertTrue(checked["frozen"])
        self.assertEqual(checked["phase"], "benchmark")
        self.assertTrue(checked["spec"]["base_config"]["selection_provenance"]["official_test_used"])
        self.assertTrue(all(run["phase"] == "benchmark" for run in materialize_run_plan(checked)))
        with self.assertRaisesRegex(ValueError, "confirmation manifest"):
            validate_frozen_confirmation(manifest, expected_hash=manifest["manifest_hash"])

    def test_frozen_authentication_rejects_missing_hash_and_relabeling(self) -> None:
        """A different digest or post-hoc phase label cannot authenticate a design."""
        manifest = self._manifest()
        for digest in (None, "", "0" * 64):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                validate_frozen_experiment(manifest, expected_hash=digest)
        relabeled = deepcopy(manifest)
        relabeled["phase"] = "confirmation"
        with self.assertRaises(ValueError):
            validate_frozen_experiment(relabeled, expected_hash=manifest["manifest_hash"])
        development = self._manifest("development")
        with self.assertRaisesRegex(ValueError, "frozen"):
            validate_frozen_experiment(development, expected_hash=development["manifest_hash"])

    def test_frozen_benchmark_preserves_metric_contrast_and_complete_pairs(self) -> None:
        """Synthetic benchmark statistics retain the declared outcome and pairing."""
        manifest = self._manifest()
        values = self._values(manifest)
        rows = collect_final_stream_metrics(manifest, values)
        kwargs = {"condition_a": "learned", "condition_b": "extra_joint",
                  "metric": "final_average_accuracy", "manifest": manifest,
                  "expected_hash": manifest["manifest_hash"]}
        result = paired_run_statistics(rows, **kwargs)
        self.assertEqual(result["phase"], "benchmark")
        self.assertEqual(result["pair_count"], 2)
        self.assertAlmostEqual(result["mean_paired_difference"], .1)
        with self.assertRaisesRegex(ValueError, "cannot override"):
            collect_final_stream_metrics(manifest, values, metric="backward_transfer")
        with self.assertRaisesRegex(ValueError, "declared"):
            paired_run_statistics(rows, **{**kwargs, "condition_a": "extra_joint", "condition_b": "learned"})
        with self.assertRaisesRegex(ValueError, "planned contrast"):
            paired_run_statistics(rows[:-1], **kwargs)

    def test_benchmark_analysis_cannot_drop_external_authentication(self) -> None:
        """Benchmark rows require their unchanged manifest and trusted digest."""
        manifest = self._manifest()
        rows = collect_final_stream_metrics(manifest, self._values(manifest))
        kwargs = {"condition_a": "learned", "condition_b": "extra_joint",
                  "metric": "final_average_accuracy"}
        with self.assertRaisesRegex(ValueError, "frozen manifest"):
            paired_run_statistics(rows, **kwargs)
        with self.assertRaisesRegex(ValueError, "expected_hash"):
            paired_run_statistics(rows, manifest=manifest, **kwargs)
        with self.assertRaises(ValueError):
            paired_run_statistics(rows, manifest=manifest, expected_hash="0" * 64, **kwargs)

    def test_source_less_benchmark_cannot_use_development_compatibility(self) -> None:
        """Frozen benchmark evidence requires a bound executable source identity."""
        with self.assertRaisesRegex(ValueError, "source-bound"):
            validate_study_source(self._manifest(), "semantic_consolidation")

    def test_learner_requires_benchmark_credentials_before_data_access(self) -> None:
        """An unbound benchmark request fails before any dataset or model work."""
        loader = Mock(side_effect=AssertionError("No data access in this synthetic test."))
        with self.assertRaisesRegex(ValueError, "benchmark requires"):
            _run_continual_tasks(class_num=2, load_dataset_fn=loader, experiment_phase="benchmark")
        loader.assert_not_called()

    def test_learner_binds_benchmark_phase_schedule_seed_and_run(self) -> None:
        """The native learner rejects identity changes before runtime setup."""
        manifest = self._manifest()
        run = next(item for item in materialize_run_plan(manifest) if item["block_id"] == "stream-a")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.json"
            write_experiment_manifest(path, manifest)
            kwargs = {"class_num": 2, "class_order": [0, 1], "task_groups": [[0], [1]], "seed": 43,
                      "experiment_phase": "benchmark", "experiment_manifest_path": str(path),
                      "experiment_manifest_hash": manifest["manifest_hash"], "experiment_run_id": run["run_id"]}
            for name, overrides, pattern in (
                ("phase", {"experiment_phase": "confirmation"}, "phase differs"),
                ("run", {"experiment_run_id": "unplanned"}, "not unique"),
                ("schedule", {"class_order": [1, 0], "task_groups": [[1], [0]]}, "schedule"),
                ("seed", {"seed": 44}, "seed differs"),
            ):
                loader = Mock(side_effect=AssertionError("No data access in this synthetic test."))
                with self.subTest(case=name), self.assertRaisesRegex(ValueError, pattern):
                    _run_continual_tasks(load_dataset_fn=loader, **{**kwargs, **overrides})
                loader.assert_not_called()
            loader = Mock(side_effect=AssertionError("No data access in this synthetic test."))
            with patch("common.learner.configure_runtime", side_effect=RuntimeError("authenticated-only-stop")):
                with self.assertRaisesRegex(RuntimeError, "authenticated-only-stop"):
                    _run_continual_tasks(load_dataset_fn=loader, **kwargs)
            loader.assert_not_called()
