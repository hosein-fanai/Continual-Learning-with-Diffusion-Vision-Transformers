"""Synthetic saved-only exporter fixtures; these are never thesis results."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np
import pandas as pd

from notebooks.thesis import results_package as package


def synthetic_campaign(directory: Path, *, scope: str = "notebooks_02_09") -> tuple[dict, dict, dict]:
    """Construct tiny scalar-only full-schedule fixtures; never train/predict.

    Args:
        directory (Path): Temporary artifact location owned by this test.
        scope (str): Original 24-stream fixture or explicitly registered notebooks_03_09.

    Returns:
        result (tuple[dict, dict, dict]): Synthetic fixture or native artifact result used only
            by the enclosing assertions.

    Raises:
        AssertionError: If the stated regression invariant fails.
        OSError: If a required temporary fixture cannot be read or written.
    """
    directory = Path(directory)
    record = {"seeds": package.SEEDS, "studies": {}}
    conditions_by_dataset = {"cifar10": list(package.METHODS)[:3], "cifar100": list(package.METHODS)}
    # Reduced fixtures carry the explicit scope required for a final 21-stream export.
    if scope == "notebooks_03_09":
        conditions_by_dataset["cifar10"] = ["extra_joint", "learned"]
        record.update(campaign_scope=scope, declared_conditions=conditions_by_dataset,
                      declared_stream_count=21)
    manifests, outputs = {}, {}
    for dataset, count, width in (("cifar10", 5, 2), ("cifar100", 10, 10)):
        conditions = conditions_by_dataset[dataset]
        entries, results = [], {}
        config = {"common": {"dataset": {"preprocess": "fixed-standardize", "name": dataset},
                             "continually_learn": {"replay_current_examples": None, "replay_old_examples": 1024}}, "route": {}}
        for block, seed in enumerate(package.SEEDS):
            for method_index, condition in enumerate(conditions):
                run_id = f"stream-{block + 1:02d}-run-{method_index + 1:02d}"
                run = directory / dataset / "runs" / run_id
                run.mkdir(parents=True)
                groups = [list(range(task * width, (task + 1) * width)) for task in range(count)]
                entries.append({"run_id": run_id, "condition": condition, "block_id": f"stream-{block + 1:02d}",
                                "stream": {"stream_seed": seed, "task_groups": groups, "class_order": list(range(count * width))}})
                matrix = np.full((count, count), np.nan)
                for task in range(count):
                    matrix[task, :task + 1] = .25 + .03 * block + .005 * task + .01 * method_index
                route, tasks, costs = [], [], []
                for task in range(1, count + 1):
                    row = {"task": task, "joint_updates": 5, "extra_joint_updates": 0,
                           "acquisition": {"updates": 2}, "consolidation": {"updates": 3},
                           "memory_bytes": {"student": 100 + task}}
                    # Apply this case only when condition not in ('baseline', 'extra_joint').
                    if condition not in ("baseline", "extra_joint"):
                        probe = {"input_sha256": f"fixed-{task}", "examples": 8, "split": "validation",
                                 "clean_accuracy": .25, "old_accuracy": None if task == 1 else .2,
                                 "new_accuracy": .3, "frozen_target_alignment": {"aggregates": {"selected_gates": {"hidden_target_cosine": .4}}},
                                 "representation": {"centered_effective_rank": 3.0}}
                        row.update(before_consolidation=probe, after_consolidation={**probe, "clean_accuracy": .28,
                                  "old_accuracy": None if task == 1 else .21,
                                  "frozen_target_alignment": {"aggregates": {"selected_gates": {"hidden_target_cosine": .45}}}},
                                   hidden_feature_cka=1., hidden_feature_cka_sample_count=2)
                    route.append(row)
                    tasks.append({"task": task, "hidden": {"per_class": {"0": {"since_acquisition": {
                        "sample_count": 2 if block == 0 else 8, "linear_cka": 1. if block == 0 else .5,
                        "mean_sample_l2_drift": .2, "relative_frobenius_drift": .1,
                        "centroid_drift": {"mean_centroid_drift": .15}}}}},
                        "generated_memory": {"summary": {"label_consistency": .4, "class_coverage": 1., "normalized_label_entropy": .9}} if task > 1 else {"reason": "No replay before first task."},
                        "tensor_inventory": {"unique_tensor_bytes": 200 + task}})
                    costs.extend([{"task_index": task - 1, "phase": "resource", "metric": "seconds/task_total", "value": 10},
                                  {"task_index": task - 1, "phase": "resource", "metric": "checkpointing/io_seconds", "value": 2},
                                  {"task_index": task - 1, "phase": "resource", "metric": "seconds/generator_fit", "value": 8}])
                package._json(run / "route_metrics.json", route)
                package._json(run / "section11.json", {"tasks": tasks, "memory": {"sampled_process_peak_rss_bytes": 1000,
                    "tf_allocator_devices": {"GPU:0": {"peak": 800, "current": 300}}}})
                pd.DataFrame(costs).to_csv(run / "task_metrics.csv", index=False)
                # Apply this case only when condition == 'learned'.
                if condition == "learned":
                    np.savez_compressed(run / f"generated_examples_task_{count:03d}.npz", images=np.zeros((3, 4, 4, 3)), labels=np.arange(3))
                results[run_id] = {"run_id": run_id, "condition": condition, "results_path": str(run),
                    "accuracy_matrix": package._clean(matrix.tolist()), "seconds": 999,
                    "accuracy_matrix_source": "ordinary_accuracy_matrix",
                    "total_updates": count * 10, "metrics": package.continual_metrics(matrix)}
        manifests[dataset] = {"manifest_hash": "synthetic-only", "spec": {"base_config": config}, "_entries": entries}
        outputs[dataset] = results
        record["studies"][dataset] = {"manifest_path": str(directory / dataset / "manifest.json")}
    return record, manifests, outputs


class AggregationTests(unittest.TestCase):
    """Bounded saved-evidence regression fixtures; never research outcomes."""

    def test_native_unavailable_phase_endpoint_remains_unavailable(self) -> None:
        """Verify native unavailable phase endpoint remains unavailable.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        self.assertFalse(package.aligned_phase_endpoints({"split": "unavailable"}, {"split": "unavailable"}))
        valid = {"split": "validation", "input_sha256": "same", "examples": 8}
        self.assertTrue(package.aligned_phase_endpoints(valid, valid))
        with self.assertRaisesRegex(ValueError, "validation split"):
            package.aligned_phase_endpoints(valid, {**valid, "split": "test"})

    def test_nullable_and_nonfinite_values_are_unavailable(self) -> None:
        """Verify nullable and nonfinite values are unavailable.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        rows = pd.DataFrame({"dataset": ["x"] * 3, "run_id": ["a", "b", "c"],
                             "value": pd.Series([1., pd.NA, np.inf], dtype="Float64")})
        result = package.summarize_streams(rows, ["dataset"]).iloc[0]
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["mean"], 1.)
        self.assertTrue(pd.isna(result["sample_sd"]))
        self.assertEqual(package._clean({"missing": pd.NA, "time": pd.NaT}), {"missing": None, "time": None})
        self.assertTrue(np.isnan(package._finite(pd.NA)))
        boolean = package.summarize_streams(
            [{"dataset": "x", "run_id": "flag", "value": True}], ["dataset"])
        self.assertEqual(boolean.iloc[0]["n"], 0)

    def test_task_runtime_requires_exact_unique_finite_nonnegative_timers(self) -> None:
        """Verify task runtime requires exact unique finite nonnegative timers.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        costs = pd.DataFrame([{"task_index": 0, "metric": "seconds/task_total", "value": 1.},
                              {"task_index": 1, "metric": "seconds/task_total", "value": 2.},
                              {"task_index": 1, "metric": "seconds/generator_fit", "value": 100.}])
        self.assertEqual(package.saved_task_runtime(costs, 2)["seconds"], 3.)
        for damaged in (pd.concat([costs, costs.iloc[:1]], ignore_index=True),
                        costs.assign(task_index=[0, 99, 1]), costs.assign(task_index=[0, .5, 1])):
            with self.assertRaisesRegex(ValueError, "task indices"):
                package.saved_task_runtime(damaged, 2)
        for missing in (np.nan, np.inf):
            damaged = costs.copy()
            damaged.loc[0, "value"] = missing
            result = package.saved_task_runtime(damaged, 2)
            self.assertTrue(np.isnan(result["seconds"]))
            self.assertEqual(result["n_tasks"], 1)
        with self.assertRaisesRegex(ValueError, "negative"):
            package.saved_task_runtime(costs.assign(value=[-1., 2., 100.]), 2)
        with self.assertRaisesRegex(ValueError, "task indices"):
            package.saved_task_runtime(costs.assign(task_index=[False, True, 1]), 2)
        for invalid in (0, True, 2.5):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                package.saved_task_runtime(costs, invalid)

    def test_sample_sd_missing_n_and_no_identifier_averaging(self) -> None:
        """Verify sample sd missing n and no identifier averaging.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        rows = [{"dataset": "x", "run_id": str(index), "seed": seed, "value": value}
                for index, (seed, value) in enumerate(((1103, 1.), (2207, 3.), (3301, np.nan)))]
        result = package.summarize_streams(rows, ["dataset"]).iloc[0]
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["mean"], 2.)
        self.assertAlmostEqual(result["sample_sd"], np.sqrt(2))
        self.assertNotIn("seed", result.index)
        one = package.summarize_streams(rows[:1], ["dataset"]).iloc[0]
        self.assertTrue(np.isnan(one["sample_sd"]))
        with self.assertRaisesRegex(ValueError, "within-stream"):
            package.summarize_streams(rows + rows[:1], ["dataset"])

    def test_legacy_cka_sanitization_preserves_drift_and_original(self) -> None:
        """Verify legacy cka sanitization preserves drift and original.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        original = {"nested": [{"sample_count": 2, "linear_cka": 1., "mean_sample_l2_drift": 3.},
                               {"linear_cka": 1.}, {"sample_count": 8, "linear_cka": .4},
                               {"sample_count": 8, "linear_cka": None, "linear_cka_unavailable_reason": "constant_centered_representation"}],
                    "hidden_feature_cka": 1.}
        result = package.sanitize_cka(original)
        self.assertIsNone(result["nested"][0]["linear_cka"])
        self.assertEqual(result["nested"][0]["mean_sample_l2_drift"], 3.)
        self.assertIsNone(result["nested"][1]["linear_cka"])
        self.assertEqual(result["nested"][2]["linear_cka"], .4)
        self.assertEqual(result["nested"][3]["linear_cka_unavailable_reason"], "constant_centered_representation")
        self.assertIsNone(result["hidden_feature_cka"])
        self.assertEqual(original["nested"][0]["linear_cka"], 1.)

    def test_stream_first_forgetting_counterexample(self) -> None:
        # Different prior maxima: max(mean(matrix)) != mean(max(each matrix)).
        """Verify stream first forgetting counterexample.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        matrices = [np.array([[.9, np.nan, np.nan], [.1, .6, np.nan], [.2, .3, .5]]),
                    np.array([[.1, np.nan, np.nan], [.9, .6, np.nan], [.2, .3, .5]])]
        values = [package.continual_metrics(matrix)["average_forgetting"] for matrix in matrices]
        rows = [{"metric": "forgetting", "run_id": str(i), "value": value} for i, value in enumerate(values)]
        self.assertAlmostEqual(package.summarize_streams(rows, ["metric"]).iloc[0]["mean"], .5)
        averaged = package.continual_metrics((matrices[0] + matrices[1]) / 2)["average_forgetting"]
        self.assertAlmostEqual(averaged, .3)


class LearningViewTests(unittest.TestCase):
    """Do not turn an incomplete or invalid trajectory into a final notebook value."""

    def test_invalid_outcomes_are_rejected_before_display_or_file_creation(self) -> None:
        """Reject invalid final matrices on both supported evaluation splits.

        Args:
            None. This test owns synthetic matrices and temporary output paths.

        Returns:
            checked (None): None; verifies rejection before display or view publication.

        Raises:
            AssertionError: If incomplete or invalid outcomes are displayed or saved.
        """
        from notebooks.thesis.presentation import show_learning_results

        invalid = (
            [[.5]],
            [[.5, None], [None, .7]],
            [[.5, None], [float("inf"), .7]],
            [[.5, None], [1.1, .7]],
            [[.5, .2], [.4, .7]],
        )
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_VIEW_VALIDATION_") as temporary:
            directory = Path(temporary)
            continual = types.SimpleNamespace(class_num=4, class_order=[0, 1, 2, 3],
                task_groups=[[0, 1], [2, 3]], task_size=2)
            config = types.SimpleNamespace(common=types.SimpleNamespace(
                training=types.SimpleNamespace(results_path=str(directory / "native")),
                continually_learn=continual))
            for phase, name in (("confirmation", "ordinary_accuracy_matrix"),
                                ("development", "validation_accuracy_matrix")):
                continual.experiment_phase = phase
                for index, matrix in enumerate(invalid):
                    target = directory / f"{phase}-{index}"
                    with self.subTest(phase=phase, matrix=matrix), \
                            patch("IPython.display.display") as display:
                        with self.assertRaises(ValueError):
                            show_learning_results(config, {"continual_details": {name: matrix}},
                                                  output_dir=target, details=False)
                        display.assert_not_called()
                        self.assertFalse(target.exists())

    def test_saved_scalar_view_identifies_split_and_complete_task_count(self) -> None:
        """Retain the split, complete task count and signed values in scalar CSVs.

        Args:
            None. This test owns a complete synthetic two-task validation matrix.

        Returns:
            checked (None): None; verifies saved metadata and independently known values.

        Raises:
            AssertionError: If the saved split, task count or scalar values differ.
            OSError: If the temporary output cannot be read or written.
        """
        from notebooks.thesis.presentation import show_learning_results

        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_VIEW_VALIDATION_") as temporary:
            directory = Path(temporary)
            config = types.SimpleNamespace(common=types.SimpleNamespace(
                training=types.SimpleNamespace(results_path=str(directory / "native")),
                continually_learn=types.SimpleNamespace(experiment_phase="development",
                    class_num=4, class_order=[0, 1, 2, 3], task_groups=[[0, 1], [2, 3]], task_size=2)))
            bundle = {"continual_details": {"validation_accuracy_matrix": [[.5, None], [.4, .7]]}}
            with patch("IPython.display.display"):
                _, views = show_learning_results(config, bundle, output_dir=directory / "views", details=False)
            saved = pd.read_csv(views / "metrics.csv")
            self.assertTrue(saved["split"].eq("validation").all())
            self.assertTrue(saved["completed_tasks"].eq(2).all())
            values = saved.set_index("metric")["value"]
            self.assertAlmostEqual(values["final_average_accuracy"], 55.)
            self.assertAlmostEqual(values["backward_transfer"], -10.)

    def test_ensemble_scalar_view_never_substitutes_ordinary_accuracy(self) -> None:
        """Both splits export ensemble values and fail if only ordinary values are present."""
        from notebooks.thesis.presentation import show_learning_results

        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_ENSEMBLE_VIEW_") as temporary:
            directory = Path(temporary)
            continual = types.SimpleNamespace(use_ensemble_accuracy=True,
                class_num=4, class_order=[0, 1, 2, 3], task_groups=[[0, 1], [2, 3]], task_size=2)
            config = types.SimpleNamespace(common=types.SimpleNamespace(
                training=types.SimpleNamespace(results_path=str(directory / "native")),
                continually_learn=continual))
            for phase, selected, ordinary in (
                ("development", "validation_ensemble_accuracy_matrix", "validation_accuracy_matrix"),
                ("confirmation", "ensemble_accuracy_matrix", "ordinary_accuracy_matrix"),
            ):
                continual.experiment_phase = phase
                bundle = {"continual_details": {ordinary: [[.9, None], [.8, 1.]],
                                                selected: [[.5, None], [.4, .7]]}}
                with self.subTest(phase=phase), patch("IPython.display.display"):
                    _, views = show_learning_results(config, bundle, output_dir=directory / phase, details=False)
                    saved = pd.read_csv(views / "metrics.csv")
                    self.assertTrue(saved.accuracy_matrix_source.eq(selected).all())
                    self.assertTrue(saved.inference.eq("timestep ensemble").all())
                    self.assertAlmostEqual(saved.set_index("metric").loc["final_average_accuracy", "value"], 55.)
                    del bundle["continual_details"][selected]
                    with self.assertRaises(KeyError):
                        show_learning_results(config, bundle, output_dir=directory / f"{phase}-missing", details=False)
                    self.assertFalse((directory / f"{phase}-missing").exists())


class SavedPackageTests(unittest.TestCase):
    """Bounded saved-evidence regression fixtures; never research outcomes."""

    def setUp(self) -> None:
        """Prepare isolated synthetic fixtures.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        self.temporary = tempfile.TemporaryDirectory(prefix="SYNTHETIC_export_validation_")
        self.directory = Path(self.temporary.name)
        self.record, self.manifests, self.outputs = synthetic_campaign(self.directory)
        self.plan = patch.object(package, "materialize_run_plan", side_effect=lambda manifest: manifest["_entries"])
        self.plan.start()

    def tearDown(self) -> None:
        """Release this test case's runtime and temporary resources.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        self.plan.stop()
        self.temporary.cleanup()

    def test_full_design_guard(self) -> None:
        """Verify full design guard.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        package._check_final_design(self.record, self.manifests)
        package._check_final_design({**self.record, "campaign_scope": "notebooks_02_09",
                                     "declared_stream_count": 24}, self.manifests)
        with self.assertRaisesRegex(ValueError, "three-seed"):
            package._check_final_design({**self.record, "seeds": [1103, 2207]}, self.manifests)
        bad = deepcopy(self.manifests)
        bad["cifar100"]["_entries"][0]["stream"]["task_groups"] = [[0, 1]]
        with self.assertRaises(ValueError):
            package._check_final_design(self.record, bad)

    def test_reduced_design_requires_explicit_scope_and_complete_paired_plan(self) -> None:
        """Accept exactly 6+15 declared streams without accepting a truncated legacy design."""
        record, manifests, _ = synthetic_campaign(self.directory / "reduced_guard", scope="notebooks_03_09")
        package._check_final_design(record, manifests)
        legacy = {key: value for key, value in record.items()
                  if key not in ("campaign_scope", "declared_conditions", "declared_stream_count")}
        for declaration in (legacy, {**legacy, "campaign_scope": "notebooks_02_09"}):
            with self.subTest(declaration=declaration.get("campaign_scope")), self.assertRaisesRegex(ValueError, "9 cifar10"):
                package._check_final_design(declaration, manifests)
        for field in ("declared_conditions", "declared_stream_count"):
            incomplete = deepcopy(record)
            incomplete.pop(field)
            with self.subTest(missing=field), self.assertRaisesRegex(ValueError, "21-stream"):
                package._check_final_design(incomplete, manifests)
        wrong_methods = deepcopy(record)
        wrong_methods["declared_conditions"]["cifar100"].remove("baseline")
        for invalid in (wrong_methods, {**record, "declared_stream_count": 24},
                        {**record, "campaign_scope": "arbitrary_subset"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                package._check_final_design(invalid, manifests)
        for dataset in manifests:
            incomplete = deepcopy(manifests)
            incomplete[dataset]["_entries"].pop()
            with self.subTest(dataset=dataset), self.assertRaisesRegex(ValueError, "streams"):
                package._check_final_design(record, incomplete)

    def test_reduced_complete_public_package_retains_both_primary_comparisons(self) -> None:
        """Publish a complete synthetic 21-stream package with no invented CIFAR-10 Platform row."""
        record, manifests, outputs = synthetic_campaign(self.directory / "reduced_complete", scope="notebooks_03_09")
        record_path = self.directory / "reduced_complete" / "frozen_design.json"
        package._json(record_path, record)
        fake_workflow = types.ModuleType("notebooks.thesis.workflow")
        fake_workflow._campaign = Mock(return_value=(record, manifests))
        fake_workflow._outputs = Mock(side_effect=lambda path, manifest, complete: outputs[path.parent.name])
        fake_workflow.analyze_campaign = Mock(return_value={})
        destination = self.directory / "SYNTHETIC_REDUCED_PACKAGE"
        with patch.dict("sys.modules", {"notebooks.thesis.workflow": fake_workflow}):
            result = package.export_results_package(record_path, output_dir=destination)
        summary = package._read(destination / "RESULT_SUMMARY.json")
        self.assertEqual(summary["campaign_scope"], "notebooks_03_09")
        self.assertEqual(summary["declared_stream_count"], 21)
        self.assertEqual(summary["completed_streams"], 21)
        self.assertEqual(summary["declared_conditions"], record["declared_conditions"])
        rows = pd.DataFrame(summary["tables"]["main_results"])
        self.assertEqual(set(rows.query("dataset == 'cifar10'").condition), {"extra_joint", "learned"})
        self.assertEqual(set(rows.query("dataset == 'cifar100'").condition), set(package.METHODS))
        self.assertTrue(rows.n.eq(3).all())
        paired = pd.DataFrame(summary["tables"]["paired_individual"])
        primary = paired.loc[paired.role.eq("primary")]
        self.assertEqual(primary.groupby("dataset").size().to_dict(), {"cifar10": 3, "cifar100": 3})
        self.assertTrue(primary.condition.eq("extra_joint").all())
        self.assertFalse(paired.query("dataset == 'cifar10'").condition.eq("baseline").any())
        context = (destination / "STUDY_CONTEXT.md").read_text(encoding="utf-8")
        self.assertIn("Registered final design: 21 streams", context)
        self.assertIn("2 methods, 6 streams", context)
        self.assertIn("CIFAR-10 Platform is explicitly omitted", context)
        self.assertIn("primary comparison remains learned minus extra joint on both datasets", context)
        self.assertTrue(all(call.kwargs["complete"] for call in fake_workflow._outputs.call_args_list))
        self.assertTrue(result["zip"].is_file())

    def test_reduced_final_export_still_requires_every_declared_completion(self) -> None:
        """One unfinished run on either dataset blocks publication before final statistics."""
        record, manifests, outputs = synthetic_campaign(self.directory / "reduced_missing", scope="notebooks_03_09")
        record_path = self.directory / "reduced_missing" / "frozen_design.json"
        package._json(record_path, record)
        fake_workflow = types.ModuleType("notebooks.thesis.workflow")
        fake_workflow._campaign = Mock(return_value=(record, manifests))

        def completed(path: Path, manifest: dict, *, complete: bool) -> dict:
            """Model the authenticated workflow's requirement for all planned completion IDs."""
            outcomes = incomplete[path.parent.name]
            expected = {entry["run_id"] for entry in manifest["_entries"]}
            # Final publication cannot turn a missing run into an omitted condition.
            if complete and set(outcomes) != expected:
                raise ValueError("finish every planned stream before analysis")
            return outcomes

        fake_workflow._outputs = Mock(side_effect=completed)
        fake_workflow.analyze_campaign = Mock()
        for dataset in manifests:
            incomplete = deepcopy(outputs)
            incomplete[dataset].pop(next(iter(incomplete[dataset])))
            with self.subTest(dataset=dataset), patch.dict("sys.modules", {"notebooks.thesis.workflow": fake_workflow}), \
                    patch.object(package, "_write_package") as writer:
                with self.assertRaisesRegex(ValueError, "finish every planned"):
                    package.export_results_package(record_path)
                writer.assert_not_called()
                fake_workflow.analyze_campaign.assert_not_called()

    def test_extraction_units_phase_n_memory_and_nonoverlapping_runtime(self) -> None:
        """Verify extraction units phase n memory and nonoverlapping runtime.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        before = {str(path): package._hash(path) for path in self.directory.rglob("*") if path.is_file()}
        evidence = package.extract_saved_evidence(self.manifests, self.outputs)
        tables = evidence["tables"]
        self.assertEqual(len(tables["individual_runs"]), 96)
        self.assertTrue(tables["main_results"]["n"].eq(3).all())
        self.assertTrue(tables["trajectories"].query("task == 1 and cohort == 'old'")["n"].eq(0).all())
        self.assertTrue(tables["phase_changes"].query("condition == 'baseline'")["n"].eq(0).all())
        self.assertTrue(np.allclose(tables["phase_changes"].query("condition == 'learned' and metric == 'hidden_target_cosine' and phase == 'after_minus_before'")["mean"], .05))
        self.assertTrue(tables["temporal_drift"].query("metric == 'linear_cka' and reference == 'since_acquisition'")["n"].eq(2).all())
        resources = tables["resources_individual"]
        self.assertTrue(resources.query("dataset == 'cifar10' and metric == 'measured_task_runtime'")["value"].eq(50).all())
        self.assertTrue(resources.query("dataset == 'cifar10' and metric == 'measured_checkpoint_io'")["value"].eq(10).all())
        self.assertTrue(resources.query("metric == 'allocator_peak/GPU:0'")["value"].eq(800).all())
        self.assertTrue(resources.query("dataset == 'cifar100' and metric == 'tensor_inventory_max'")["value"].eq(210).all())
        self.assertTrue(np.allclose(tables["replay"].query("metric == 'label_consistency'")["mean"], .4))
        after = {str(path): package._hash(path) for path in self.directory.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_phase_alignment_mismatch_fails(self) -> None:
        """Verify phase alignment mismatch fails.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        run = next(record for record in self.outputs["cifar10"].values() if record["condition"] == "learned")
        path = Path(run["results_path"]) / "route_metrics.json"
        rows = package._read(path)
        rows[0]["after_consolidation"]["input_sha256"] = "changed"
        package._json(path, rows)
        with self.assertRaisesRegex(ValueError, "identical fixed examples"):
            package.extract_saved_evidence(self.manifests, self.outputs)

    def test_ensemble_exports_label_efficacy_and_preserve_ordinary_diagnostics(self) -> None:
        """Exported endpoint metadata cannot relabel ordinary saved phase measurements."""
        for manifest in self.manifests.values():
            manifest["spec"]["base_config"]["common"]["continually_learn"]["use_ensemble_accuracy"] = True
        for outcomes in self.outputs.values():
            for record in outcomes.values():
                record["accuracy_matrix_source"] = "ensemble_accuracy_matrix"
        evidence = package.extract_saved_evidence(self.manifests, self.outputs)
        for name in ("individual_runs", "main_results", "trajectories_individual", "trajectories", "paired_effects", "thesis_summary"):
            self.assertTrue(evidence["tables"][name].accuracy_matrix_source.eq("ensemble_accuracy_matrix").all(), name)
        self.assertNotIn("accuracy_matrix_source", evidence["tables"]["phase_observations"])
        self.assertIn("ordinary clean", package.TABLE_CAPTIONS["phase_observations"])
        first = next(iter(self.outputs["cifar10"].values()))
        first["accuracy_matrix_source"] = "ordinary_accuracy_matrix"
        with self.assertRaisesRegex(ValueError, "configured accuracy predictor"):
            package.extract_saved_evidence(self.manifests, self.outputs)

    def test_unavailable_phase_payload_cannot_produce_a_numeric_change(self) -> None:
        """Retain missingness even when an unavailable endpoint has stale scalars.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        run = next(record for record in self.outputs["cifar10"].values() if record["condition"] == "learned")
        path = Path(run["results_path"]) / "route_metrics.json"
        rows = package._read(path)
        for row in rows:
            row["before_consolidation"]["split"] = "unavailable"
            row["after_consolidation"]["split"] = "unavailable"
        package._json(path, rows)
        table = package.extract_saved_evidence(self.manifests, self.outputs)["tables"]["phase_individual"]
        selected = table.loc[table["run_id"].eq(run["run_id"]) & table["dataset"].eq("cifar10")]
        self.assertTrue(selected["value"].isna().all())

    def test_compact_export_uses_saved_scalar_results_without_plotting(self) -> None:
        """Compact publication retains per-stream evidence and exact primary effects.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        evidence = package.extract_saved_evidence(self.manifests, self.outputs)
        summary = evidence["tables"]["thesis_summary"]
        self.assertEqual(len(summary), 8)
        self.assertIn("n=3", summary.iloc[0]["Final accuracy (%)"])
        self.assertIn("n=3", summary.iloc[0]["Backward transfer (pp)"])
        with patch.object(package, "_plots", side_effect=AssertionError("plot requested")), \
                patch.object(package, "_qualitative", side_effect=AssertionError("replay plot requested")):
            result = package._write_package(self.directory / "compact", self.record, self.manifests,
                                            evidence, {}, status="SYNTHETIC CHECK ONLY", details=False)
        self.assertFalse(list((result / "figures").iterdir()))
        self.assertTrue(list((result / "tables").glob("*thesis_summary.csv")))
        self.assertTrue(list((result / "tables").glob("*individual_runs.csv")))

    def test_repeated_observer_tasks_cannot_become_replicates(self) -> None:
        """Verify repeated observer tasks cannot become replicates.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        run = next(iter(self.outputs["cifar10"].values()))
        path = Path(run["results_path"]) / "section11.json"
        observation = package._read(path)
        observation["tasks"].append(deepcopy(observation["tasks"][0]))
        package._json(path, observation)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            package.extract_saved_evidence(self.manifests, self.outputs)

    def test_development_review_uses_matching_validation_rows_and_percent_units(self) -> None:
        """Verify development review uses matching validation rows and percent units.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        from notebooks.thesis.development import review_development_run
        run = next(row for row in self.outputs["cifar10"].values() if row["condition"] == "learned")
        directory = Path(run["results_path"])
        pd.DataFrame([{"task": 1, "joint_updates": 5}]).to_csv(directory / "route_resources.csv", index=False)
        review = review_development_run(directory)
        phase = review["deployed_classifier_and_hidden_phase_changes"]
        clean = phase.loc[phase["measurement"].eq("clean_accuracy")].iloc[0]
        self.assertEqual(clean["before"], 25.)
        self.assertEqual(clean["before_after_unit"], "%")
        self.assertAlmostEqual(clean["change_after_minus_before"], 3.)
        self.assertEqual(review["measured_task_runtime"].iloc[0]["sum_measured_task_seconds"], 50.)
        path = directory / "route_metrics.json"
        records = package._read(path)
        records[0]["after_consolidation"]["input_sha256"] = "different-rows"
        package._json(path, records)
        with self.assertRaisesRegex(ValueError, "identical fixed examples"):
            review_development_run(directory)

    def test_missing_task_runtime_is_unavailable_not_partial_sum(self) -> None:
        """Verify missing task runtime is unavailable not partial sum.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        run = next(iter(self.outputs["cifar10"].values()))
        path = Path(run["results_path"]) / "task_metrics.csv"
        rows = pd.read_csv(path)
        rows.iloc[1:].to_csv(path, index=False)
        evidence = package.extract_saved_evidence(self.manifests, self.outputs)
        resource = evidence["tables"]["resources_individual"]
        selected = resource.loc[resource["dataset"].eq("cifar10") & resource["metric"].eq("measured_task_runtime") & resource["run_id"].eq(run["run_id"])]
        self.assertTrue(selected["value"].isna().all())

    def test_writing_package_has_data_captions_provenance_and_synthetic_banner(self) -> None:
        """Verify writing package has data captions provenance and synthetic banner.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        evidence = package.extract_saved_evidence(self.manifests, self.outputs)
        destination = Path(os.environ.get("THESIS_EXPORT_VALIDATION_OUTPUT", self.directory / "SYNTHETIC_PACKAGE"))
        native = {dataset: {"pair_count": 3, "mean_paired_difference": .01,
                  "sample_sd_paired_difference": 0., "ci_95_lower": .01, "ci_95_upper": .01} for dataset in self.manifests}
        package._write_package(destination, self.record, self.manifests, evidence, native, status="SYNTHETIC VALIDATION — NOT THESIS RESULTS")
        for filename in ("READ_ME_FIRST.md", "STUDY_CONTEXT.md", "RESULT_SUMMARY.json", "CAPTIONS.md", "ARTIFACT_MANIFEST.json"):
            self.assertTrue((destination / filename).is_file())
        self.assertIn("SYNTHETIC VALIDATION", (destination / "READ_ME_FIRST.md").read_text())
        manifest = package._read(destination / "ARTIFACT_MANIFEST.json")
        self.assertTrue(all(item["caption"] and item["source_files"] and item["generating_procedure"] for item in manifest["artifacts"]))
        for item in manifest["artifacts"]:
            self.assertEqual(item["source_files"]["catalog"], "source_runs")
            self.assertTrue(set(item["source_files"]["run_ids"]) <= set(manifest["source_runs"]))
        self.assertEqual(len(list((destination / "figures").glob("*.png"))), 6)
        self.assertEqual(len(list((destination / "figures").glob("*.svg"))), 6)
        self.assertEqual(len(list((destination / "figures").glob("*.csv"))), 6)
        self.assertFalse(list(destination.rglob("*.npz")))
        self.assertFalse(list(destination.rglob("*.h5")))
        self.assertEqual(package._read(destination / "provenance/native_primary_statistics.json"), native)
        self.assertTrue(pd.read_csv(destination / "tables/T90_primary_native_interval.csv")["ci_95_lower"].eq(1.).all())

    def test_public_export_requires_completion_and_repeats_without_rewriting(self) -> None:
        """Verify public export requires completion and repeats without rewriting.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        record_path = self.directory / "frozen_design.json"
        package._json(record_path, self.record)
        fake_workflow = types.ModuleType("notebooks.thesis.workflow")
        fake_workflow._campaign = Mock(return_value=(self.record, self.manifests))
        fake_workflow._outputs = Mock(side_effect=lambda path, manifest, complete: self.outputs[path.parent.name])
        fake_workflow.analyze_campaign = Mock(return_value={})
        def fixture_writer(directory: Path, record: dict | None, manifests: dict, evidence: dict, native: dict, *, status: str, details: bool) -> Path:
            """Exercise the existing native artifact operation with the enclosing test fixture.

            Args:
                directory (Path): Temporary artifact location owned by this test.
                record (dict | None): Fixture value supplied by the enclosing regression;
                    existing native validators interpret its fields.
                manifests (dict): Fixture value supplied by the enclosing regression; existing
                    native validators interpret its fields.
                evidence (dict): Fixture value supplied by the enclosing regression; existing
                    native validators interpret its fields.
                native (dict): Fixture value supplied by the enclosing regression; existing
                    native validators interpret its fields.
                status (str): Explicit synthetic/progress label; never a thesis result.
                details (bool): Fixture value supplied by the enclosing regression; existing
                    native validators interpret its fields.

            Returns:
                result (Path): Synthetic fixture or native artifact result used only by the
                    enclosing assertions.

            Raises:
                AssertionError: If the stated regression invariant fails.
                OSError: If a required temporary fixture cannot be read or written.
            """
            directory.mkdir()
            # Public routing is tested here; plotting is exercised separately.
            (directory / "READ_ME_FIRST.md").write_text("SYNTHETIC VALIDATION — NOT THESIS RESULTS")
            return directory
        destination = self.directory / "SYNTHETIC_PUBLIC_PACKAGE"
        with patch.dict("sys.modules", {"notebooks.thesis.workflow": fake_workflow}), \
                patch.object(package, "_write_package", side_effect=fixture_writer) as writer:
            result = package.export_results_package(record_path, output_dir=destination)
            original = package._hash(result["zip"])
            second = package.export_results_package(record_path, output_dir=destination)
            self.assertTrue(second["reused"])
            self.assertEqual(original, package._hash(second["zip"]))
            self.assertEqual(writer.call_count, 1)
            source_run = Path(next(iter(self.outputs["cifar10"].values()))["results_path"])
            for forbidden in (source_run, source_run / "package", source_run.parent):
                with self.assertRaisesRegex(ValueError, "separate directory"):
                    package.export_results_package(record_path, output_dir=forbidden)
            self.assertTrue(all(call.kwargs["complete"] for call in fake_workflow._outputs.call_args_list))
            with zipfile.ZipFile(result["zip"]) as archive:
                self.assertIn("SYNTHETIC_PUBLIC_PACKAGE/READ_ME_FIRST.md", archive.namelist())
                self.assertFalse(any(name.endswith((".h5", ".npz")) for name in archive.namelist()))
            (destination / "READ_ME_FIRST.md").write_text("changed after publication")
            with self.assertRaisesRegex(ValueError, "changed or is incomplete"):
                package.export_results_package(record_path, output_dir=destination)

    def test_public_incomplete_final_refuses_and_progress_is_marked(self) -> None:
        """Verify public incomplete final refuses and progress is marked.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        record_path = self.directory / "frozen_design.json"
        package._json(record_path, self.record)
        fake_workflow = types.ModuleType("notebooks.thesis.workflow")
        fake_workflow._campaign = Mock(return_value=(self.record, self.manifests))
        def completed(path: Path, manifest: dict, *, complete: bool) -> dict:
            """Exercise the existing native artifact operation with the enclosing test fixture.

            Args:
                path (Path): Temporary artifact location owned by this test.
                manifest (dict): Fixture value supplied by the enclosing regression; existing
                    native validators interpret its fields.
                complete (bool): Fixture value supplied by the enclosing regression; existing
                    native validators interpret its fields.

            Returns:
                result (dict): Synthetic fixture or native artifact result used only by the
                    enclosing assertions.

            Raises:
                AssertionError: If the stated regression invariant fails.
                OSError: If a required temporary fixture cannot be read or written.
            """
            # Reject this case: finish every planned stream before analysis.
            if complete:
                raise ValueError("finish every planned stream before analysis")
            return self.outputs[path.parent.name]
        fake_workflow._outputs = Mock(side_effect=completed)
        fake_workflow.analyze_campaign = Mock()
        def fixture_writer(directory: Path, *args: object, status: str, details: bool) -> Path:
            """Exercise the existing native artifact operation with the enclosing test fixture.

            Args:
                directory (Path): Temporary artifact location owned by this test.
                status (str): Explicit synthetic/progress label; never a thesis result.
                details (bool): Fixture value supplied by the enclosing regression; existing
                    native validators interpret its fields.
                args (object): Fixture value supplied by the enclosing regression; existing
                    native validators interpret its fields.

            Returns:
                result (Path): Synthetic fixture or native artifact result used only by the
                    enclosing assertions.

            Raises:
                AssertionError: If the stated regression invariant fails.
                OSError: If a required temporary fixture cannot be read or written.
            """
            self.assertIn("PROGRESS ONLY", status)
            directory.mkdir()
            (directory / "READ_ME_FIRST.md").write_text("SYNTHETIC VALIDATION — " + status)
            return directory
        with patch.dict("sys.modules", {"notebooks.thesis.workflow": fake_workflow}), \
                patch.object(package, "_write_package", side_effect=fixture_writer) as writer:
            with self.assertRaisesRegex(ValueError, "finish every planned"):
                package.export_results_package(record_path)
            self.assertFalse(writer.called)
            self.assertFalse(fake_workflow.analyze_campaign.called)
            result = package.export_results_package(record_path, progress=True)
            self.assertTrue(result["zip"].is_file())
            self.assertFalse(fake_workflow.analyze_campaign.called)


# Run this isolated unittest module when invoked as a script.
if __name__ == "__main__":
    unittest.main()
