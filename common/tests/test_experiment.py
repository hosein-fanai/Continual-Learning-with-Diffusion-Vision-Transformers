"""Focused tests for paired-block research manifests and run-level inference."""

from __future__ import annotations

import copy
import math
import tempfile
import unittest

from pathlib import Path

from common.experiment import (
    LONG_RESULT_FIELDS,
    collect_final_stream_metrics,
    create_paired_block_manifest,
    materialize_run_plan,
    paired_run_statistics,
    read_experiment_manifest,
    read_long_results,
    validate_confirmation_rerun,
    validate_experiment_manifest,
    validate_frozen_confirmation,
    write_experiment_manifest,
    write_long_results,
)


class ExperimentDesignTests(unittest.TestCase):
    """Verify deterministic blocking, freezing, and valid replication units."""

    @staticmethod
    def _conditions() -> dict[str, dict[str, object]]:
        """Return three complete experimental condition declarations.

        Args:
            None.

        Returns:
            dict[str, dict[str, object]]: Named snapshot/distillation settings.
        """

        return {
            "distilled_ema": {"distillation": True, "snapshot": "ema"},
            "distilled_raw": {"distillation": True, "snapshot": "raw"},
            "no_distillation": {"distillation": False, "snapshot": "raw"},
        }

    @staticmethod
    def _streams() -> list[dict[str, object]]:
        """Return independent complete continual-stream specifications.

        Args:
            None.

        Returns:
            list[dict[str, object]]: Resolved class schedules for three blocks.
        """

        return [
            {
                "block_id": "seed-run-01",
                "class_order": [0, 1, 2, 3],
                "task_groups": [[0], [1], [2], [3]],
            },
            {
                "block_id": "seed-run-02",
                "class_order": [2, 0, 3, 1],
                "task_groups": [[2], [0], [3], [1]],
            },
            {
                "block_id": "seed-run-03",
                "class_order": [1, 3, 0, 2],
                "task_groups": [[1], [3], [0], [2]],
            },
        ]

    @staticmethod
    def _result_rows(
        manifest_hash: str,
        phase: str = "confirmation",
    ) -> list[dict[str, object]]:
        """Return four independent pairs with differences one through four.

        Args:
            manifest_hash (str): Experiment digest linked from every row.
            phase (str): Development or confirmation result label.

        Returns:
            list[dict[str, object]]: Eight run-level long-form result rows.
        """

        rows: list[dict[str, object]] = []
        a_values = (2.0, 4.0, 5.0, 7.0)
        b_values = (1.0, 2.0, 2.0, 3.0)
        for index, (a_value, b_value) in enumerate(
            zip(a_values, b_values),
            start=1,
        ):
            block_id = f"block-{index:04d}"
            rows.extend((
                {
                    "manifest_hash": manifest_hash,
                    "phase": phase,
                    "block_id": block_id,
                    "run_id": f"{block_id}-a",
                    "condition": "a",
                    "metric": "final_average_accuracy",
                    "value": a_value,
                    "analysis_unit": "continual_stream_block",
                },
                {
                    "manifest_hash": manifest_hash,
                    "phase": phase,
                    "block_id": block_id,
                    "run_id": f"{block_id}-b",
                    "condition": "b",
                    "metric": "final_average_accuracy",
                    "value": b_value,
                    "analysis_unit": "continual_stream_block",
                },
            ))
        return rows

    def test_manifest_crosses_conditions_and_seeded_stream_blocks(self) -> None:
        """Cross every condition with every stream and preserve seeded order.

        Args:
            None.

        Returns:
            None: Manifest structure and reproducibility are asserted in place.
        """

        manifest = create_paired_block_manifest(
            self._conditions(),
            self._streams(),
            seed=91,
            phase="development",
        )
        repeated = create_paired_block_manifest(
            self._conditions(),
            self._streams(),
            seed=91,
            phase="development",
        )
        changed_seed = create_paired_block_manifest(
            self._conditions(),
            self._streams(),
            seed=92,
            phase="development",
        )

        self.assertEqual(manifest, repeated)
        self.assertNotEqual(manifest["manifest_hash"], changed_seed["manifest_hash"])
        self.assertEqual(manifest["phase"], "development")
        self.assertFalse(manifest["frozen"])
        spec = manifest["spec"]
        self.assertEqual(spec["independent_unit"], "continual_stream_block")
        self.assertFalse(spec["tasks_are_replicates"])
        expected_conditions = sorted(self._conditions())
        self.assertEqual(len(spec["blocks"]), 3)
        for block in spec["blocks"]:
            self.assertEqual(sorted(block["execution_order"]), expected_conditions)
            self.assertEqual(len(block["runs"]), len(expected_conditions))
            self.assertEqual(
                [run["condition"] for run in block["runs"]],
                block["execution_order"],
            )

    def test_base_config_is_sealed_canonical_and_materialized(self) -> None:
        """Seal the shared run configuration and detach each planned copy.

        Args:
            None.

        Returns:
            None: Hash, canonicalization, and plan-copy contracts are asserted.
        """

        base_config = {
            "train": {"epochs": 5, "batch_size": 32},
            "continually_learn": {"seed": 17},
        }
        reordered = {
            "continually_learn": {"seed": 17},
            "train": {"batch_size": 32, "epochs": 5},
        }
        manifest = create_paired_block_manifest(
            self._conditions(),
            self._streams(),
            seed=91,
            base_config=base_config,
        )
        repeated = create_paired_block_manifest(
            self._conditions(),
            self._streams(),
            seed=91,
            base_config=reordered,
        )
        changed = create_paired_block_manifest(
            self._conditions(),
            self._streams(),
            seed=91,
            base_config={"train": {"epochs": 6, "batch_size": 32}},
        )

        self.assertEqual(manifest, repeated)
        self.assertNotEqual(manifest["manifest_hash"], changed["manifest_hash"])
        self.assertEqual(manifest["spec"]["base_config"], reordered)
        plan = materialize_run_plan(manifest)
        self.assertTrue(all(run["base_config"] == reordered for run in plan))
        plan[0]["base_config"]["train"]["epochs"] = 999
        self.assertEqual(manifest["spec"]["base_config"]["train"]["epochs"], 5)
        with self.assertRaisesRegex(TypeError, "base_config"):
            create_paired_block_manifest(
                self._conditions(),
                self._streams(),
                seed=91,
                base_config=[("train", {})],
            )

    def test_duplicate_canonical_streams_are_not_independent_blocks(self) -> None:
        """Reject one stream copied under two cosmetic block identifiers.

        Args:
            None.

        Returns:
            None: Duplicate canonical stream rejection is asserted in place.
        """

        streams = [
            {
                "block_id": block_id,
                "stream_seed": 29,
                "class_order": [0, 1],
                "task_groups": [[0], [1]],
            }
            for block_id in ("copy-a", "copy-b")
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate canonical"):
            create_paired_block_manifest(
                {"a": {}, "b": {}},
                streams,
                seed=5,
            )

    def test_frozen_confirmation_rejects_test_informed_changes(self) -> None:
        """Require a trusted hash and permit only exact confirmation reruns.

        Args:
            None.

        Returns:
            None: Hash and rerun rejection contracts are asserted in place.
        """

        confirmation = create_paired_block_manifest(
            self._conditions(),
            self._streams(),
            seed=117,
            phase="confirmation",
        )
        frozen_hash = confirmation["manifest_hash"]
        self.assertEqual(
            validate_frozen_confirmation(
                confirmation,
                expected_hash=frozen_hash,
            ),
            confirmation,
        )
        self.assertEqual(
            validate_confirmation_rerun(
                confirmation,
                confirmation,
                frozen_hash=frozen_hash,
                test_results_accessed=True,
            ),
            frozen_hash,
        )

        changed = create_paired_block_manifest(
            self._conditions(),
            self._streams(),
            seed=118,
            phase="confirmation",
        )
        with self.assertRaisesRegex(ValueError, "test-informed"):
            validate_confirmation_rerun(
                confirmation,
                changed,
                frozen_hash=frozen_hash,
                test_results_accessed=True,
            )
        with self.assertRaisesRegex(ValueError, "expected external hash"):
            validate_frozen_confirmation(
                confirmation,
                expected_hash="0" * 64,
            )

        tampered = copy.deepcopy(confirmation)
        tampered["spec"]["randomization_seed"] = 119
        with self.assertRaisesRegex(ValueError, "manifest_hash"):
            validate_experiment_manifest(tampered)

    def test_manifest_and_long_results_round_trip_without_overwrite(self) -> None:
        """Persist canonical manifest and CSV artifacts without silent replacement.

        Args:
            None.

        Returns:
            None: Round-trip and exclusive-create behavior are asserted in place.
        """

        manifest = create_paired_block_manifest(
            {"a": {"model": "raw"}, "b": {"model": "ema"}},
            self._streams(),
            seed=23,
            phase="confirmation",
        )
        rows = self._result_rows(manifest["manifest_hash"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "confirmation_manifest.json"
            results_path = root / "run_metrics.csv"
            write_experiment_manifest(manifest_path, manifest)
            write_long_results(results_path, rows)
            self.assertEqual(
                read_experiment_manifest(
                    manifest_path,
                    expected_hash=manifest["manifest_hash"],
                ),
                manifest,
            )
            self.assertEqual(read_long_results(results_path), rows)
            with self.assertRaises(FileExistsError):
                write_experiment_manifest(manifest_path, manifest)
            with self.assertRaises(FileExistsError):
                write_long_results(results_path, rows)

    def test_paired_statistics_use_runs_not_tasks_as_replicates(self) -> None:
        """Compute the planned paired t interval across independent run blocks.

        Args:
            None.

        Returns:
            None: Run-level estimates and pseudoreplication guards are asserted.
        """

        manifest_hash = "a" * 64
        rows = self._result_rows(manifest_hash, phase="development")
        result = paired_run_statistics(
            rows,
            condition_a="a",
            condition_b="b",
            metric="final_average_accuracy",
        )
        self.assertEqual(result["pair_count"], 4)
        self.assertEqual(result["paired_differences"], [1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(result["mean_paired_difference"], 2.5)
        self.assertAlmostEqual(
            result["sample_sd_paired_difference"],
            math.sqrt(5.0 / 3.0),
        )
        self.assertAlmostEqual(result["t_critical_95"], 3.182446305284263)
        self.assertAlmostEqual(result["ci_95_lower"], 0.445739743239121)
        self.assertAlmostEqual(result["ci_95_upper"], 4.554260256760879)
        self.assertFalse(result["tasks_used_as_replicates"])
        # SciPy is optional, so validate its result only when present.
        if result["paired_t_p_value"] is not None:
            self.assertGreaterEqual(result["paired_t_p_value"], 0.0)
            self.assertLessEqual(result["paired_t_p_value"], 1.0)

        task_rows = copy.deepcopy(rows)
        task_rows[0]["analysis_unit"] = "task"
        with self.assertRaisesRegex(ValueError, "continual_stream_block"):
            paired_run_statistics(
                task_rows,
                condition_a="a",
                condition_b="b",
                metric="final_average_accuracy",
            )
        with self.assertRaisesRegex(ValueError, "both condition"):
            paired_run_statistics(
                rows[:-1],
                condition_a="a",
                condition_b="b",
                metric="final_average_accuracy",
            )
        self.assertEqual(tuple(rows[0]), LONG_RESULT_FIELDS)

    def test_confirmation_analysis_enforces_the_frozen_primary_contrast(self) -> None:
        """Bind confirmation collection and inference to preregistered choices.

        Args:
            None.

        Returns:
            None: Trusted-hash and primary-contrast guards are asserted.
        """

        analysis_spec = {
            "condition_a": "distilled_ema",
            "condition_b": "distilled_raw",
            "primary_metric": "final_average_accuracy",
        }
        manifest = create_paired_block_manifest(
            self._conditions(),
            self._streams(),
            seed=313,
            phase="confirmation",
            analysis_spec=analysis_spec,
            base_config={"train": {"epochs": 3}},
        )
        plan = materialize_run_plan(manifest)
        values = {
            run["run_id"]: float(index)
            for index, run in enumerate(plan, start=1)
        }
        rows = collect_final_stream_metrics(manifest, values)
        result = paired_run_statistics(
            rows,
            condition_a="distilled_ema",
            condition_b="distilled_raw",
            metric="final_average_accuracy",
            manifest=manifest,
            expected_hash=manifest["manifest_hash"],
        )
        self.assertEqual(result["pair_count"], len(self._streams()))

        with self.assertRaisesRegex(ValueError, "cannot override"):
            collect_final_stream_metrics(
                manifest,
                values,
                metric="backward_transfer",
            )
        with self.assertRaisesRegex(ValueError, "frozen manifest"):
            paired_run_statistics(
                rows,
                condition_a="distilled_ema",
                condition_b="distilled_raw",
                metric="final_average_accuracy",
            )
        with self.assertRaisesRegex(ValueError, "preregistered"):
            paired_run_statistics(
                rows,
                condition_a="distilled_raw",
                condition_b="distilled_ema",
                metric="final_average_accuracy",
                manifest=manifest,
                expected_hash=manifest["manifest_hash"],
            )
        with self.assertRaisesRegex(ValueError, "expected external hash"):
            paired_run_statistics(
                rows,
                condition_a="distilled_ema",
                condition_b="distilled_raw",
                metric="final_average_accuracy",
                manifest=manifest,
                expected_hash="0" * 64,
            )

    def test_runner_plan_collects_one_final_metric_per_complete_stream(self) -> None:
        """Expose full runner inputs and require every planned final scalar.

        Args:
            None.

        Returns:
            None: Runner expansion and complete result collection are asserted.
        """

        manifest = create_paired_block_manifest(
            self._conditions(),
            self._streams(),
            seed=211,
            phase="confirmation",
        )
        plan = materialize_run_plan(
            manifest,
            expected_hash=manifest["manifest_hash"],
        )
        self.assertEqual(len(plan), 9)
        self.assertTrue(all(
            run["analysis_unit"] == "continual_stream_block"
            for run in plan
        ))
        self.assertEqual(
            [run["condition"] for run in plan[:3]],
            manifest["spec"]["blocks"][0]["execution_order"],
        )
        plan[0]["stream"]["class_order"][0] = 999
        self.assertNotEqual(
            plan[0]["stream"],
            manifest["spec"]["blocks"][0]["stream"],
        )

        values = {
            run["run_id"]: float(index)
            for index, run in enumerate(materialize_run_plan(manifest), start=1)
        }
        rows = collect_final_stream_metrics(
            manifest,
            values,
            expected_hash=manifest["manifest_hash"],
        )
        self.assertEqual(len(rows), len(plan))
        self.assertEqual(
            {row["run_id"] for row in rows},
            set(values),
        )
        self.assertEqual(
            {row["metric"] for row in rows},
            {"final_average_accuracy"},
        )
        with self.assertRaisesRegex(ValueError, "cover every planned run"):
            collect_final_stream_metrics(
                manifest,
                dict(list(values.items())[:-1]),
            )
        with self.assertRaisesRegex(TypeError, "string run_id"):
            collect_final_stream_metrics(manifest, {1: 0.5})
        invalid_values = dict(values)
        invalid_values[rows[0]["run_id"]] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            collect_final_stream_metrics(manifest, invalid_values)


# Run the focused module tests when invoked directly.
if __name__ == "__main__":
    unittest.main()
