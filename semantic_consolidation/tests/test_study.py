"""Frozen-design and paired-analysis tests with explicitly SYNTHETIC outcomes.

These tests never train a benchmark model. Every numerical outcome is an
invented unit-test fixture, written only inside a temporary directory and
labeled ``SYNTHETIC_UNIT_TEST_NOT_THESIS_RESULTS`` in its record.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from common.experiment import materialize_run_plan, read_experiment_manifest, read_long_results
from common.study_artifacts import write_completed_artifact
from semantic_consolidation.config import load_route_config, validate_route_config
from semantic_consolidation.study import _completed_metrics, analyze_study, prepare_study, run_study, validate_planned_config


_SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
_SYNTHETIC_LABEL = "SYNTHETIC_UNIT_TEST_NOT_THESIS_RESULTS"


class StudyTests(unittest.TestCase):
    """Bind executable route configs to a paired design and audit result coverage."""

    def setUp(self) -> None:
        """Create isolated fixtures and preserve the caller state needed for this test."""
        self.temporary = tempfile.TemporaryDirectory(prefix="route_SYNTHETIC_test_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.template = load_route_config(_SMOKE_CONFIG)

    def _prepare(self, phase: str = "development") -> tuple[Path, dict, list[dict]]:
        """Create an isolated paired design and return its authenticated manifest and run plan."""
        manifest_path = prepare_study(
            self.template, self.directory / "synthetic_study", [17, 29, 43], phase=phase
        )
        manifest = read_experiment_manifest(manifest_path)
        return manifest_path, manifest, materialize_run_plan(manifest)

    def _synthetic_outcomes(self, plan: list[dict]) -> dict:
        """Invent known complete-stream pairs exclusively for numerical testing."""

        block_ids = sorted({entry["block_id"] for entry in plan})
        synthetic_scores = {
            "learned": [0.70, 0.65, 0.60],
            "extra_joint": [0.60, 0.50, 0.40],
            "random": [0.55, 0.55, 0.55],
        }
        return {
            entry["run_id"]: {
                "fixture_kind": _SYNTHETIC_LABEL,
                "manifest_hash": entry["manifest_hash"],
                "run_id": entry["run_id"],
                "condition": entry["condition"],
                "results_path": _SYNTHETIC_LABEL,
                "seconds": 0., "total_updates": 0,
                "metrics": {
                    "final_average_accuracy": synthetic_scores[entry["condition"]][
                        block_ids.index(entry["block_id"])
                    ]
                },
            }
            for entry in plan
        }

    def _write_synthetic_outcomes(self, manifest_path: Path, outcomes: dict) -> None:
        """Write explicitly synthetic full-stream results for analysis contract tests."""
        manifest = read_experiment_manifest(manifest_path)
        # Confirmation fixtures exercise the full matrix and per-run artifact contract.
        if manifest["phase"] == "confirmation":
            for result in outcomes.values():
                value = result["metrics"]["final_average_accuracy"]
                result["accuracy_matrix"] = [[value, None], [value, value]]
                result["accuracy_matrix_source"] = "ordinary_accuracy_matrix"
                result["metrics"] = _completed_metrics(result["accuracy_matrix"], 2)
                result["completed_artifact"] = write_completed_artifact(manifest_path.parent, result)
        with (manifest_path.parent / "completed_runs.json").open("w", encoding="utf-8") as stream:
            json.dump(outcomes, stream, indent=2, allow_nan=False)

    def test_prepare_materializes_complete_paired_streams_and_loadable_configs(self) -> None:
        """All three treatments share each independent seed and class permutation."""

        path, manifest, plan = self._prepare()
        self.assertEqual(len(plan), 9)
        self.assertFalse((path.parent / "completed_runs.json").exists())
        self.assertEqual(manifest["spec"]["analysis_spec"]["condition_a"], "learned")
        self.assertEqual(manifest["spec"]["analysis_spec"]["condition_b"], "extra_joint")
        grouped = {}
        for entry in plan:
            grouped.setdefault(entry["block_id"], []).append(entry)
            config = load_route_config(path.parent / f"{entry['run_id']}.yaml")
            validate_planned_config(config)
            self.assertEqual(config.route.condition, entry["condition"])
            self.assertEqual(config.common.training.seed, entry["stream"]["stream_seed"])
            self.assertEqual(config.route.seed, entry["stream"]["stream_seed"])
            self.assertEqual(config.common.continually_learn.class_order, entry["stream"]["class_order"])
            self.assertEqual(config.common.continually_learn.experiment_manifest_hash, manifest["manifest_hash"])
        self.assertEqual(len(grouped), 3)
        for entries in grouped.values():
            self.assertEqual({entry["condition"] for entry in entries}, {"learned", "random", "extra_joint"})
            self.assertTrue(all(entry["stream"] == entries[0]["stream"] for entry in entries))
            self.assertEqual(sorted(entries[0]["stream"]["class_order"]), [0, 1, 2, 3])
            self.assertEqual([len(group) for group in entries[0]["stream"]["task_groups"]], [2, 2])

    def test_confirmation_binds_route_model_and_replay_settings(self) -> None:
        """Changing scientific coefficients, architecture, or replay invalidates a run."""

        path, _, plan = self._prepare("confirmation")
        original = load_route_config(path.parent / f"{plan[0]['run_id']}.yaml")
        validate_planned_config(original)
        alterations = {
            "alignment_weight": lambda config: setattr(config.route, "alignment_weight", config.route.alignment_weight + 0.25),
            "temperature": lambda config: setattr(config.route, "temperature", config.route.temperature + 0.1),
            "noise_levels": lambda config: setattr(config.route, "noise_levels", (0,)),
            "model_dimension": lambda config: config.common.model.kwargs.update(dim=config.common.model.kwargs["dim"] + 8),
            "replay_budget": lambda config: setattr(config.common.continually_learn, "replay_old_examples", 100),
        }
        for name, alter in alterations.items():
            changed = deepcopy(original)
            alter(changed)
            with self.subTest(setting=name), self.assertRaisesRegex(ValueError, "differ from the manifest"):
                validate_planned_config(changed)

    def test_frozen_run_allows_artifact_destination_relocation(self) -> None:
        """Relocating results preserves the same frozen scientific configuration."""

        path, _, plan = self._prepare("confirmation")
        config = load_route_config(path.parent / f"{plan[0]['run_id']}.yaml")
        config.common.training.results_path = str(self.directory / "relocated_artifacts")
        validate_planned_config(config)

    def test_confirmation_without_a_manifest_is_rejected(self) -> None:
        """A confirmation label requires a frozen experiment binding."""

        self.template.common.continually_learn.experiment_phase = "confirmation"
        self.template.common.continually_learn.experiment_manifest_path = None
        with self.assertRaisesRegex(ValueError, "frozen paired experiment manifest"):
            validate_planned_config(self.template)

    def test_manifest_tampering_is_rejected(self) -> None:
        """A changed design cannot retain a previously assigned digest."""

        path, manifest, plan = self._prepare("confirmation")
        config = load_route_config(path.parent / f"{plan[0]['run_id']}.yaml")
        manifest["spec"]["base_config"]["route"]["temperature"] = 0.9
        with path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream)
        with self.assertRaises(ValueError):
            validate_planned_config(config)

    def test_synthetic_analysis_uses_full_stream_pairs_not_tasks(self) -> None:
        """Synthetic paired differences 0.10, 0.15, 0.20 have mean 0.15."""

        path, manifest, plan = self._prepare("confirmation")
        synthetic_outcomes = self._synthetic_outcomes(plan)
        self.assertTrue(all(row["fixture_kind"] == _SYNTHETIC_LABEL for row in synthetic_outcomes.values()))
        self._write_synthetic_outcomes(path, synthetic_outcomes)
        statistics = analyze_study(path, expected_hash=manifest["manifest_hash"])
        self.assertEqual(statistics["pair_count"], 3)
        self.assertEqual(statistics["analysis_unit"], "continual_stream_block")
        self.assertFalse(statistics["tasks_used_as_replicates"])
        np.testing.assert_allclose(statistics["paired_differences"], [0.10, 0.15, 0.20], atol=1e-12)
        self.assertAlmostEqual(statistics["mean_paired_difference"], 0.15)
        self.assertAlmostEqual(statistics["sample_sd_paired_difference"], 0.05)
        self.assertEqual(statistics["degrees_of_freedom"], 2)
        rows = read_long_results(path.parent / "paired_results.csv")
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(row["manifest_hash"] == manifest["manifest_hash"] for row in rows))
        with (path.parent / "paired_statistics.json").open(encoding="utf-8") as stream:
            saved = json.load(stream)
        self.assertAlmostEqual(saved["mean_paired_difference"], 0.15)

    def test_synthetic_analysis_rejects_missing_or_foreign_run_outcomes(self) -> None:
        """Partial or foreign fixture runs cannot silently enter paired inference."""

        path, _, plan = self._prepare()
        complete = self._synthetic_outcomes(plan)
        incomplete = deepcopy(complete)
        incomplete.pop(plan[0]["run_id"])
        foreign = deepcopy(complete)
        foreign["foreign_SYNTHETIC_run"] = deepcopy(next(iter(complete.values())))
        for label, outcomes in (("incomplete", incomplete), ("foreign", foreign)):
            with self.subTest(kind=label):
                self._write_synthetic_outcomes(path, outcomes)
                expected_message = "cover every planned run" if label == "incomplete" else "do not belong"
                with self.assertRaisesRegex(ValueError, expected_message):
                    analyze_study(path)
                self.assertFalse((path.parent / "paired_statistics.json").exists())

    def test_swapped_valid_run_files_fail_before_training(self) -> None:
        """Valid YAML from another planned run must not be relabeled by its filename."""

        path, _, plan = self._prepare()
        first_path = path.parent / f"{plan[0]['run_id']}.yaml"
        second_path = path.parent / f"{plan[1]['run_id']}.yaml"
        first_contents, second_contents = first_path.read_bytes(), second_path.read_bytes()
        first_path.write_bytes(second_contents)
        second_path.write_bytes(first_contents)
        with patch("semantic_consolidation.runner.run") as train:
            with self.assertRaisesRegex(ValueError, "different run or manifest identity"):
                run_study(path)
            train.assert_not_called()

    def test_synthetic_outcomes_from_another_valid_manifest_are_rejected(self) -> None:
        """Matching run names cannot authorize stale outcomes from a different design."""

        _, first_manifest, first_plan = self._prepare()
        altered_template = deepcopy(self.template)
        altered_template.route.temperature += 0.1
        second_path = prepare_study(
            altered_template, self.directory / "another_synthetic_study", [17, 29, 43],
        )
        second_manifest = read_experiment_manifest(second_path)
        self.assertNotEqual(first_manifest["manifest_hash"], second_manifest["manifest_hash"])
        first_ids = {entry["run_id"] for entry in first_plan}
        second_ids = {entry["run_id"] for entry in materialize_run_plan(second_manifest)}
        self.assertEqual(first_ids, second_ids)
        self._write_synthetic_outcomes(second_path, self._synthetic_outcomes(first_plan))
        with self.assertRaisesRegex(ValueError, "do not belong"):
            analyze_study(second_path)
        self.assertFalse((second_path.parent / "paired_statistics.json").exists())

    def test_prepare_rejects_duplicate_seeds_and_existing_destination(self) -> None:
        """Independent blocks and exclusive output locations are explicit requirements."""

        for seeds in ([17], [17, 17]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                prepare_study(self.template, self.directory / "invalid_study", seeds)
        self.assertFalse((self.directory / "invalid_study").exists())
        with self.assertRaises(FileExistsError):
            prepare_study(self.template, self.directory, [17, 29])

    def test_confirmation_requires_external_hash_before_any_work(self) -> None:
        """Self-consistency of a resealable manifest is not preregistration evidence."""

        path, manifest, plan = self._prepare("confirmation")
        self._write_synthetic_outcomes(path, self._synthetic_outcomes(plan))
        with patch("semantic_consolidation.runner.run") as train:
            for operation in (run_study, analyze_study):
                with self.subTest(operation=operation.__name__):
                    with self.assertRaisesRegex(ValueError, "externally retained expected_hash"):
                        operation(path)
                    with self.assertRaises(ValueError):
                        operation(path, expected_hash="0" * 64)
            train.assert_not_called()
        self.assertFalse((path.parent / "paired_statistics.json").exists())
        # The retained original digest authenticates the complete synthetic plan.
        self.assertEqual(analyze_study(path, expected_hash=manifest["manifest_hash"])["pair_count"], 3)

    def test_later_invalid_run_is_rejected_before_earlier_runs_train(self) -> None:
        """The whole plan must be reviewable before the first expensive stream."""

        path, _, plan = self._prepare()
        last = path.parent / f"{plan[-1]['run_id']}.yaml"
        last.write_bytes((path.parent / f"{plan[0]['run_id']}.yaml").read_bytes())
        with patch("semantic_consolidation.runner.run") as train:
            with self.assertRaisesRegex(ValueError, "different run or manifest identity"):
                run_study(path)
            train.assert_not_called()
        self.assertFalse((path.parent / "completed_runs.json").exists())

    def test_prepare_rejects_invalid_seed_types_and_missing_contrast(self) -> None:
        """Boolean seeds and a lone condition cannot silently identify a study."""

        for seeds in ([True, 17], [17.5, 29], [-1, 29], [2 ** 32, 29]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                prepare_study(self.template, self.directory / "bad_seed", seeds)
        for conditions in ({}, {"learned": {"route": {"condition": "learned"}}}):
            with self.subTest(conditions=conditions), self.assertRaisesRegex(ValueError, "at least two conditions"):
                prepare_study(self.template, self.directory / "bad_contrast", [17, 29], conditions=conditions)

    def test_semantic_protocol_rejects_privileged_history_and_missing_positive_pairs(self) -> None:
        """A nominal continual semantic run cannot quietly receive old real images."""

        cumulative = deepcopy(self.template)
        cumulative.common.continually_learn.remove_prev_classes = False
        with self.assertRaisesRegex(ValueError, "historical real-data"):
            validate_route_config(cumulative)
        no_replay = deepcopy(self.template)
        no_replay.common.continually_learn.use_generative_replay = False
        with self.assertRaisesRegex(ValueError, "old modulators require"):
            validate_route_config(no_replay)
        too_small = deepcopy(self.template)
        too_small.common.continually_learn.replay_old_examples = 3
        with self.assertRaisesRegex(ValueError, "two rows per old class"):
            validate_route_config(too_small)

    def test_completed_matrix_does_not_average_away_missing_or_future_cells(self) -> None:
        """Only complete fractional lower triangles define a finished stream outcome."""

        valid = [[.7, None], [.6, .8]]
        self.assertAlmostEqual(_completed_metrics(valid, 2)["final_average_accuracy"], .7)
        invalid = ([[.7, None], [None, .8]], [[.7, .1], [.6, .8]],
                   [[.7, None], [1.6, .8]], [[.7, None], [float("inf"), .8]], [[.7]])
        for matrix in invalid:
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                _completed_metrics(matrix, 2)

    def test_analysis_verifies_retained_matrix_against_summary(self) -> None:
        """A modified scalar cannot override the retained complete accuracy matrix."""

        path, _, plan = self._prepare()
        outputs = self._synthetic_outcomes(plan)
        for result in outputs.values():
            value = result["metrics"]["final_average_accuracy"]
            result["accuracy_matrix"] = [[value, None], [value, value]]
            result["accuracy_matrix_source"] = "validation_accuracy_matrix"
            result["metrics"] = _completed_metrics(result["accuracy_matrix"], 2)
        outputs[plan[0]["run_id"]]["metrics"]["final_average_accuracy"] += .01
        self._write_synthetic_outcomes(path, outputs)
        with self.assertRaisesRegex(ValueError, "disagree with the completed accuracy matrix"):
            analyze_study(path)
        self.assertFalse((path.parent / "paired_statistics.json").exists())

    def test_legacy_index_primary_accuracy_must_still_be_a_finite_fraction(self) -> None:
        """Older scalar-only exports do not permit impossible or boolean accuracies."""

        path, _, plan = self._prepare()
        for invalid in (True, 1.01, -.1, "0.7"):
            outputs = self._synthetic_outcomes(plan)
            outputs[plan[0]["run_id"]]["metrics"]["final_average_accuracy"] = invalid
            self._write_synthetic_outcomes(path, outputs)
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "finite fraction"):
                analyze_study(path)


# Run this module directly while keeping imports free of execution side effects.
if __name__ == "__main__":
    unittest.main()
