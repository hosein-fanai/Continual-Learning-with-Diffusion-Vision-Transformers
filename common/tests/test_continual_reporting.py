"""Focused tests for long-form continual-learning report artifacts."""

from __future__ import annotations

import csv
import tempfile
import unittest

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import tensorflow as tf

from common.continual_reporting import (
    write_continual_csv_artifacts,
    write_continual_tensorboard_summaries,
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read one UTF-8 CSV artifact into dictionary rows.

    Args:
        path (pathlib.Path): CSV path written by the reporting helper.

    Returns:
        list[dict[str, str]]: Rows preserving the file's header names.
    """

    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _details_fixture() -> dict[str, object]:
    """Return representative complete continual learner details.

    Returns:
        dict[str, object]: Two-task histories, evaluations, schedules, metrics,
        and test/validation accuracy matrices.
    """

    return {
        "class_order": [3, 1, 2],
        "task_classes": [[3, 1], [2]],
        "dataset_seed": 7,
        "histories": [
            {
                "loss": [1.0, 0.5],
                "accuracy": [np.float32(0.25), np.float64(0.5)],
                "structured_metric": [[1.0, 2.0]],
            },
            {"loss": [0.4]},
        ],
        "generative_histories": [
            {"noise_loss": [2.0]},
            None,
        ],
        "classifier_evaluations": [
            {"valset_eval": [0.5, 0.75]},
            {"valset_ema_eval": {"ensemble_accuracy": 0.8}},
        ],
        "generative_evaluations": [
            {"valset_network_eval": {"noise_loss": 0.9}},
            {},
        ],
        "accuracies": [0.6, 0.7],
        "ensemble_accuracies": [0.62, 0.72],
        "new_task_accuracy": [0.8, 0.9],
        "old_task_accuracy": [np.nan, 0.65],
        "accuracy_matrix": [
            [0.8, np.nan],
            [0.65, 0.9],
        ],
        "validation_accuracy_matrix": [
            [0.75, np.nan],
            [0.6, 0.85],
        ],
        "accuracy_matrices": {
            "ensemble_accuracy_matrix": [
                [0.81, np.nan],
                [0.66, 0.91],
            ],
        },
        "continual_metrics": {
            "final_average_accuracy": 0.775,
            "average_forgetting": 0.15,
        },
    }


class ContinualReportingTests(unittest.TestCase):
    """Verify complete, sparse, and TensorBoard continual report outputs."""

    def test_csv_artifacts_preserve_scalar_metrics(self: "ContinualReportingTests") -> None:
        """Verify long-form rows, schedules, matrices, and summaries.

        Args:
            self (ContinualReportingTests): Active unit-test case.

        Returns:
            None.
        """

        details = _details_fixture()
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_continual_csv_artifacts(
                details,
                temporary_directory,
                metadata={"run_id": "run-001"},
            )

            self.assertEqual(
                set(paths),
                {
                    "epoch_metrics", "task_metrics", "accuracy_matrices",
                    "schedule", "summary",
                },
            )
            self.assertTrue(all(path.is_file() for path in paths.values()))

            epoch_rows = _read_rows(paths["epoch_metrics"])
            epoch_metrics = {row["metric"] for row in epoch_rows}
            self.assertEqual(
                epoch_metrics,
                {"loss", "accuracy", "noise_loss"},
            )
            self.assertEqual(len(epoch_rows), 6)
            self.assertNotIn("structured_metric", epoch_metrics)

            task_rows = _read_rows(paths["task_metrics"])
            task_metrics = {row["metric"] for row in task_rows}
            self.assertIn("valset_ema_eval/ensemble_accuracy", task_metrics)
            self.assertIn("valset_network_eval/noise_loss", task_metrics)
            self.assertIn("accuracies", task_metrics)
            self.assertIn("old_task_accuracy", task_metrics)

            matrix_rows = _read_rows(paths["accuracy_matrices"])
            self.assertEqual(len(matrix_rows), 12)
            self.assertEqual(
                {row["matrix"] for row in matrix_rows},
                {
                    "accuracy_matrix",
                    "validation_accuracy_matrix",
                    "ensemble_accuracy_matrix",
                },
            )
            self.assertTrue(any(row["value"].lower() == "nan" for row in matrix_rows))

            schedule_rows = _read_rows(paths["schedule"])
            self.assertEqual(len(schedule_rows), 2)
            self.assertEqual(schedule_rows[0]["task_classes"], "[3,1]")
            self.assertEqual(schedule_rows[1]["seen_classes"], "[3,1,2]")

            summary_rows = _read_rows(paths["summary"])
            summary_keys = {(row["source"], row["key"]) for row in summary_rows}
            self.assertIn(("metadata", "run_id"), summary_keys)
            self.assertIn(
                ("continual_metrics", "final_average_accuracy"),
                summary_keys,
            )
            self.assertIn(("schedule", "task_count"), summary_keys)

    def test_csv_artifacts_tolerate_missing_optional_fields(
        self: "ContinualReportingTests",
    ) -> None:
        """Verify an empty detail mapping still creates parseable schemas.

        Args:
            self (ContinualReportingTests): Active unit-test case.

        Returns:
            None.
        """

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_continual_csv_artifacts({}, temporary_directory)
            for name, path in paths.items():
                self.assertTrue(path.is_file())
                rows = _read_rows(path)
                # Metric and schedule tables remain empty without source data.
                if name != "summary":
                    self.assertEqual(rows, [])
                # Summary retains explicit zero-task/empty-matrix metadata.
                else:
                    self.assertEqual(
                        {(row["source"], row["key"]) for row in rows},
                        {
                            ("artifacts", "accuracy_matrices"),
                            ("schedule", "task_count"),
                        },
                    )

    def test_task_diagnostics_flatten_into_task_metrics_csv(
        self: "ContinualReportingTests",
    ) -> None:
        """Flatten nested resource and mechanistic mappings by task.

        Args:
            self (ContinualReportingTests): Active unit-test case.

        Returns:
            None: CSV rows retain phase, task, and original-class identity.
        """

        details = {
            "task_classes": [[3, 1], [2]],
            "task_resource_metrics": [
                {
                    "runtime": {"train_seconds": 1.25},
                    "memory": {"peak_mb": 128},
                },
                {"runtime": {"train_seconds": 2.5}},
            ],
            "task_mechanistic_metrics": [
                {
                    "teacher": {"ece": 0.1},
                    "replay": {"class_coverage": 1.0},
                },
                {"representation": {"linear_cka": 0.8}},
            ],
            # A present canonical field is authoritative over its short alias.
            "resource_metrics": [{"ignored_alias_metric": 999}],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_continual_csv_artifacts(
                details,
                temporary_directory,
            )
            task_rows = _read_rows(paths["task_metrics"])

        indexed_rows = {
            (row["phase"], row["task_index"], row["metric"]): row
            for row in task_rows
        }
        self.assertEqual(
            indexed_rows[("resource", "0", "runtime/train_seconds")][
                "task_classes"
            ],
            "[3,1]",
        )
        self.assertEqual(
            indexed_rows[("resource", "1", "runtime/train_seconds")][
                "seen_classes"
            ],
            "[3,1,2]",
        )
        self.assertEqual(
            indexed_rows[("mechanistic", "0", "teacher/ece")]["value"],
            "0.1",
        )
        self.assertIn(
            ("mechanistic", "1", "representation/linear_cka"),
            indexed_rows,
        )
        self.assertNotIn(
            ("resource", "0", "ignored_alias_metric"),
            indexed_rows,
        )

    def test_joint_history_metrics_are_reported_once_in_correct_phase(
        self: "ContinualReportingTests",
    ) -> None:
        """Partition one shared joint fit without changing learner details.

        Args:
            self (ContinualReportingTests): Active unit-test case.

        Returns:
            None: Noise and classifier/KD trajectories have unique phases.
        """

        joint_history = {
            "loss": [3.0],
            "total_noise_loss": [2.0],
            "classifier_loss": [0.8],
            "distil_loss": [0.2],
            "distil_token_accuracy": [0.75],
        }
        details = {
            "task_classes": [[4]],
            "histories": [joint_history],
            "generative_histories": [joint_history],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_continual_csv_artifacts(
                details,
                temporary_directory,
            )
            rows = _read_rows(paths["epoch_metrics"])

        observed = [(row["phase"], row["metric"]) for row in rows]
        self.assertEqual(len(observed), len(joint_history))
        self.assertIn(("generator", "total_noise_loss"), observed)
        self.assertIn(("generator", "loss"), observed)
        self.assertIn(("classifier", "classifier_loss"), observed)
        self.assertIn(("classifier", "distil_loss"), observed)
        self.assertIn(("classifier", "distil_token_accuracy"), observed)

    def test_short_diagnostic_aliases_remain_supported(
        self: "ContinualReportingTests",
    ) -> None:
        """Accept concise diagnostic keys when canonical fields are absent.

        Args:
            self (ContinualReportingTests): Active unit-test case.

        Returns:
            None: Alias values produce the same reporting phases.
        """

        details = {
            "task_classes": [[7]],
            "resource_metrics": [{"gpu_hours": 0.25}],
            "mechanistic_metrics": [{"replay": {"coverage": 0.75}}],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_continual_csv_artifacts(
                details,
                temporary_directory,
            )
            rows = _read_rows(paths["task_metrics"])

        self.assertEqual(
            {(row["phase"], row["metric"]) for row in rows},
            {
                ("resource", "gpu_hours"),
                ("mechanistic", "replay/coverage"),
            },
        )

    def test_tensorboard_namespaces_include_task_classes_and_phases(
        self: "ContinualReportingTests",
    ) -> None:
        """Verify writer paths and scalar tags encode classes and phases.

        Args:
            self (ContinualReportingTests): Active unit-test case.

        Returns:
            None.
        """

        created_paths = []
        created_writers = []

        def make_writer(path: str) -> Mock:
            """Create a context-compatible mock summary writer.

            Args:
                path (str): TensorBoard log directory requested by the helper.

            Returns:
                unittest.mock.Mock: Writer with context and flush methods.
            """

            created_paths.append(Path(path))
            writer = Mock()
            writer.as_default.return_value = nullcontext()
            created_writers.append(writer)
            return writer

        details = _details_fixture()
        with tempfile.TemporaryDirectory() as temporary_directory, patch.object(
            tf.summary,
            "create_file_writer",
            side_effect=make_writer,
        ), patch.object(tf.summary, "scalar") as scalar_mock, patch.object(
            tf.summary,
            "text",
        ) as text_mock:
            counts = write_continual_tensorboard_summaries(
                details,
                temporary_directory,
                start_step=5,
            )

        relative_paths = {
            path.relative_to(temporary_directory).as_posix()
            for path in created_paths
        }
        self.assertIn("task-000_classes-3-1/classifier", relative_paths)
        self.assertIn("task-000_classes-3-1/generator", relative_paths)
        self.assertIn("task-001_classes-2/continual", relative_paths)
        self.assertGreater(sum(counts.values()), 0)

        scalar_tags = {call.args[0] for call in scalar_mock.call_args_list}
        self.assertIn(
            "task_000/classes_3-1/classifier/loss",
            scalar_tags,
        )
        self.assertIn(
            "task_000/classes_3-1/generator/noise_loss",
            scalar_tags,
        )
        self.assertIn(
            "task_001/classes_2/classifier/evaluation/"
            "valset_ema_eval/ensemble_accuracy",
            scalar_tags,
        )
        self.assertTrue(all(call.kwargs["step"] >= 5 for call in scalar_mock.call_args_list))
        self.assertGreater(text_mock.call_count, 0)
        self.assertTrue(created_writers)
        self.assertTrue(all(
            writer.close.call_count == 1 for writer in created_writers
        ))

    def test_tensorboard_writes_task_diagnostic_namespaces(
        self: "ContinualReportingTests",
    ) -> None:
        """Write diagnostics under class-identifying per-task namespaces.

        Args:
            self (ContinualReportingTests): Active unit-test case.

        Returns:
            None: Writer paths, tags, counts, and steps are verified.
        """

        created_paths = []
        created_writers = []

        def make_writer(path: str) -> Mock:
            """Create a context-compatible mock summary writer.

            Args:
                path (str): TensorBoard log directory requested by the helper.

            Returns:
                unittest.mock.Mock: Writer with context and flush methods.
            """

            created_paths.append(Path(path))
            writer = Mock()
            writer.as_default.return_value = nullcontext()
            created_writers.append(writer)
            return writer

        details = {
            "task_classes": [[3, 1], [2]],
            "task_resource_metrics": [
                {"runtime": {"train_seconds": 1.25}},
                {},
            ],
            "task_mechanistic_metrics": [
                {"teacher": {"ece": 0.1}},
                {"representation": {"linear_cka": 0.8}},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory, patch.object(
            tf.summary,
            "create_file_writer",
            side_effect=make_writer,
        ), patch.object(tf.summary, "scalar") as scalar_mock, patch.object(
            tf.summary,
            "text",
        ) as text_mock:
            counts = write_continual_tensorboard_summaries(
                details,
                temporary_directory,
                start_step=11,
            )

        relative_paths = {
            path.relative_to(temporary_directory).as_posix()
            for path in created_paths
        }
        self.assertEqual(
            relative_paths,
            {
                "task-000_classes-3-1/resource",
                "task-000_classes-3-1/mechanistic",
                "task-001_classes-2/mechanistic",
            },
        )
        self.assertEqual(set(counts), relative_paths)
        scalar_tags = {call.args[0] for call in scalar_mock.call_args_list}
        self.assertIn(
            "task_000/classes_3-1/resource/runtime/train_seconds",
            scalar_tags,
        )
        self.assertIn(
            "task_001/classes_2/mechanistic/representation/linear_cka",
            scalar_tags,
        )
        self.assertTrue(
            all(call.kwargs["step"] == 11 for call in scalar_mock.call_args_list)
        )
        self.assertEqual(text_mock.call_count, 6)
        self.assertTrue(created_writers)
        self.assertTrue(all(
            writer.close.call_count == 1 for writer in created_writers
        ))


# Support direct execution in addition to unittest discovery.
if __name__ == "__main__":
    unittest.main()
