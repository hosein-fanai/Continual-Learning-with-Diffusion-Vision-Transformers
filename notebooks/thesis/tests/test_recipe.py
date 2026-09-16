"""Bounded recipe and saved-review validation; fixtures are never thesis results."""

from __future__ import annotations

import ast
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd
import yaml

from common.experiment import materialize_run_plan
from notebooks.thesis import workflow
from notebooks.thesis.development import review_development_run
from notebooks.thesis.tests.test_bootstrap import NOTEBOOK_NAMES
from semantic_consolidation.config import load_route_config
from semantic_consolidation.study import validate_planned_config


NOTEBOOKS = Path(__file__).resolve().parents[1]
TEMPLATES = {name: NOTEBOOKS / "configs" / f"{name}.yaml" for name in ("cifar10", "cifar100")}
SEEDS = [1103, 2207, 3301]


class PreparedRecipeTests(unittest.TestCase):
    """Exercise native materialization of the complete declared reduced campaign."""

    def test_development_identity_tracks_inherited_settings_and_source(self) -> None:
        """Keep revised pilots separate without rewriting their earlier evidence.

        Args:
            None. This case owns temporary inherited YAML and checkpoint files.

        Returns:
            checked (None): Identical recipes select the same location; inherited
                changes and source changes each select a new location.

        Raises:
            AssertionError: If changed pilots collide or old evidence changes.
        """
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            base = yaml.safe_load((NOTEBOOKS.parents[1] / "semantic_consolidation/configs/common_v1.yaml").read_text("utf-8"))
            leaf = yaml.safe_load(TEMPLATES["cifar10"].read_text("utf-8"))
            leaf["base_config"] = "base.yaml"
            leaf["common"]["training"]["results_path"] = str(directory / "runs")
            (directory / "recipe.yaml").write_text(yaml.safe_dump(leaf), encoding="utf-8")
            (directory / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
            with patch.object(workflow, "_initialize", side_effect=lambda config, context: (config, context)), \
                    patch.object(workflow, "source_fingerprint", return_value={"sha256": "source-a"}):
                first, _ = workflow.load_development(directory / "recipe.yaml")
                same, _ = workflow.load_development(directory / "recipe.yaml")
                old_root = Path(first.common.continually_learn.checkpoint_dir)
                self.assertEqual(old_root, Path(same.common.continually_learn.checkpoint_dir))
                old_root.mkdir(parents=True)
                evidence = old_root / "preserved-evidence.txt"
                evidence.write_bytes(b"previous pilot evidence")
                base["dataset"]["validation_ratio"] = .25
                (directory / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
                changed, _ = workflow.load_development(directory / "recipe.yaml")
                self.assertEqual(changed.common.dataset.validation_ratio, .25)
                self.assertNotEqual(old_root, Path(changed.common.continually_learn.checkpoint_dir))
            with patch.object(workflow, "_initialize", side_effect=lambda config, context: (config, context)), \
                    patch.object(workflow, "source_fingerprint", return_value={"sha256": "source-b"}):
                revised, _ = workflow.load_development(directory / "recipe.yaml")
                self.assertNotEqual(changed.common.continually_learn.checkpoint_dir,
                                    revised.common.continually_learn.checkpoint_dir)
                before = workflow._development_identity(revised)
                revised.common.continually_learn.checkpoint_dir = "different/runtime/location"
                revised.common.continually_learn.resume_from = "different/runtime/snapshot"
                self.assertEqual(before, workflow._development_identity(revised))
            self.assertEqual(evidence.read_bytes(), b"previous pilot evidence")

    def test_all_24_native_configs_preserve_paired_full_streams_and_recipe(self) -> None:
        """Verify all 24 native configs preserve paired full streams and recipe.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        with tempfile.TemporaryDirectory(prefix="synthetic-recipe-validation-") as temporary:
            campaign = Path(temporary) / "campaign"
            frozen = workflow.prepare_campaign(campaign, TEMPLATES, SEEDS)
            record, manifests = workflow._campaign(frozen)
            self.assertEqual(record["schema_version"], 2)
            self.assertEqual(record["seeds"], SEEDS)
            self.assertEqual(record["declared_stream_count"], 24)
            self.assertTrue({"init.py", "notebooks/init.py"} <= record["bound_files"].keys())
            shared_initializer = NOTEBOOKS.parent / "init.py"
            original_digest = workflow._digest
            with patch.object(workflow, "_digest", side_effect=lambda path:
                              "changed-initializer" if Path(path) == shared_initializer
                              else original_digest(path)):
                with self.assertRaisesRegex(ValueError, "Frozen.*notebooks/init.py"):
                    workflow._campaign(frozen)
            total = 0
            for dataset, classes, tasks, task_size, epochs, acquisition, consolidation in (
                ("cifar10", 10, 5, 2, 40, 200, 400),
                ("cifar100", 100, 10, 10, 60, 1000, 2000),
            ):
                manifest = manifests[dataset]
                plan = materialize_run_plan(manifest)
                total += len(plan)
                self.assertEqual(len(plan), 9 if dataset == "cifar10" else 15)
                self.assertEqual(Counter(entry["condition"] for entry in plan),
                                 {condition: 3 for condition in workflow.CONDITIONS[dataset]})
                self.assertEqual({entry["stream"]["stream_seed"] for entry in plan}, set(SEEDS))
                references = {}
                for entry in plan:
                    with self.subTest(dataset=dataset, run=entry["run_id"]):
                        config = load_route_config(campaign / dataset / f"{entry['run_id']}.yaml")
                        validate_planned_config(config)
                        project, route = config.common, config.route
                        continual, stream = project.continually_learn, entry["stream"]
                        self.assertEqual(continual.class_num, classes)
                        self.assertEqual(len(continual.task_groups), tasks)
                        self.assertEqual([len(group) for group in continual.task_groups], [task_size] * tasks)
                        self.assertEqual(sorted(continual.class_order), list(range(classes)))
                        self.assertEqual(sum(continual.task_groups, []), continual.class_order)
                        self.assertEqual(continual.class_order, stream["class_order"])
                        self.assertEqual((project.training.seed, continual.seed, route.seed),
                                         (stream["stream_seed"],) * 3)
                        self.assertEqual(continual.experiment_phase, "confirmation")
                        self.assertEqual(continual.experiment_run_id, entry["run_id"])
                        self.assertEqual(continual.experiment_manifest_hash, manifest["manifest_hash"])
                        self.assertEqual(project.dataset.name, dataset)
                        self.assertEqual(project.dataset.validation_ratio, .2)
                        self.assertIsNone(project.dataset.max_train_samples)
                        self.assertIsNone(continual.replay_current_examples)
                        self.assertEqual(continual.replay_old_examples, 2048)
                        self.assertEqual(continual.replay_budget_mode, "fixed_total")
                        self.assertTrue(continual.remove_prev_classes)
                        self.assertEqual(project.dataset.batch_size, 64)
                        self.assertEqual(project.training.epochs, epochs)
                        self.assertEqual(project.model.name, "dit_classifier")
                        self.assertEqual(project.model.wrapper_name, "diffusion_classifier")
                        self.assertEqual(project.model.kwargs["dim"], 128)
                        self.assertEqual(project.model.kwargs["depth"], 4)
                        self.assertEqual(project.optimizer.name, "adam")
                        self.assertEqual(project.optimizer.initial_learning_rate, .0002)
                        self.assertEqual(project.optimizer.weight_decay, 0.)
                        self.assertEqual(project.optimizer.schedule, "constant")
                        self.assertFalse(continual.use_ensemble_accuracy)
                        self.assertFalse(continual.evaluate_ensemble_accuracy)
                        self.assertFalse(project.model.wrapper_kwargs["use_ema"])
                        self.assertEqual(project.model.wrapper_kwargs["test_network_name"], "raw")
                        self.assertEqual(project.model.wrapper_kwargs["test_noisified_min_timesteps"], 0)
                        self.assertEqual(project.model.wrapper_kwargs["test_noisified_max_timesteps"], 0)
                        self.assertEqual(route.experimental["probe_per_class"], 8)
                        self.assertEqual((route.acquisition_steps, route.consolidation_steps),
                                         (acquisition, consolidation))
                        self.assertEqual(route.acquisition_steps // task_size, 100)
                        self.assertEqual(route.consolidation_steps // classes, 40 if classes == 10 else 20)
                        # Extra joint consumes this same allowance in the controller.
                        self.assertEqual(route.acquisition_steps + route.consolidation_steps,
                                         600 if classes == 10 else 3000)
                        self.assertEqual(route.condition,
                                         workflow.CONDITIONS[dataset][entry["condition"]]["route"]["condition"])
                        common = asdict(project)
                        common["training"].pop("project_tag")
                        common["continually_learn"].pop("experiment_run_id")
                        reference = references.setdefault(entry["block_id"], (stream, common))
                        self.assertEqual(stream, reference[0])
                        self.assertEqual(common, reference[1], "Within-stream common platform must match across conditions.")
            self.assertEqual(total, 24)
            self.assertEqual(len(list(campaign.rglob("*.yaml"))), 24)
            self.assertFalse(list(campaign.rglob("*.started.json")))
            self.assertFalse(list(campaign.rglob("*.completed.json")))
            original = {path.relative_to(campaign): path.read_bytes() for path in campaign.rglob("*") if path.is_file()}
            with self.assertRaises(FileExistsError):
                workflow.prepare_campaign(campaign, TEMPLATES, SEEDS)
            self.assertEqual(original, {path.relative_to(campaign): path.read_bytes()
                                       for path in campaign.rglob("*") if path.is_file()})

    def test_two_seed_or_development_seed_campaign_is_rejected_without_creation(self) -> None:
        """Verify two seed or development seed campaign is rejected without creation.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        with tempfile.TemporaryDirectory(prefix="synthetic-seed-validation-") as temporary:
            for index, seeds in enumerate(([1103, 2207], [17, 2207, 3301], [1103, 2207, 2207])):
                destination = Path(temporary) / str(index)
                with self.subTest(seeds=seeds), self.assertRaisesRegex(ValueError, "requires seeds"):
                    workflow.prepare_campaign(destination, TEMPLATES, seeds)
                self.assertFalse(destination.exists())

    def test_development_is_separate_seed_17_full_validation_stream(self) -> None:
        """Verify development is separate seed 17 full validation stream.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        with patch.object(workflow, "_initialize", side_effect=lambda config, context: (config, context)):
            for dataset, tasks in (("cifar10", 5), ("cifar100", 10)):
                config, context = workflow.load_development(TEMPLATES[dataset], condition="learned")
                self.assertEqual(config.common.training.seed, 17)
                self.assertEqual(config.route.seed, 17)
                self.assertEqual(config.common.continually_learn.experiment_phase, "development")
                self.assertIsNone(config.common.continually_learn.experiment_manifest_path)
                self.assertIsNone(context["record_path"])
                self.assertEqual(len(config.common.continually_learn.task_groups), tasks)


class NotebookContractTests(unittest.TestCase):
    """Structural checks do not execute training cells or claim useful learning."""

    def test_all_thirteen_notebooks_are_valid_clean_and_syntactically_executable(self) -> None:
        """Validate canonical notebook sources and the portable hosted kernel metadata.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        paths = [NOTEBOOKS / name for name in NOTEBOOK_NAMES]
        self.assertEqual([path.name[:2] for path in paths], [f"{index:02d}" for index in range(13)])
        for path in paths:
            with self.subTest(notebook=path.name):
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                self.assertEqual(notebook.metadata.kernelspec.name, "python3")
                self.assertEqual(notebook.metadata.kernelspec.display_name, "Python 3 (ipykernel)")
                for index, cell in enumerate(notebook.cells):
                    # Apply this case only when cell.cell_type == 'code'.
                    if cell.cell_type == "code":
                        ast.parse(cell.source, filename=f"{path.name}:cell{index}")
                        self.assertIsNone(cell.execution_count)
                        self.assertEqual(cell.outputs, [])

    def test_each_training_notebook_selects_one_next_unfinished_stream(self) -> None:
        """Verify each training notebook selects one next unfinished stream.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        for path in (NOTEBOOKS / name for name in NOTEBOOK_NAMES[2:10]):
            with self.subTest(notebook=path.name):
                notebook = nbformat.read(path, as_version=4)
                tree = ast.parse("\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code"))
                selections = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                              and isinstance(node.func, ast.Name) and node.func.id == "load_run"]
                fits = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name) and node.func.id == "train_model"]
                self.assertEqual(len(selections), 1)
                self.assertEqual(len(fits), 1)
                choice = next(keyword.value for keyword in selections[0].keywords if keyword.arg == "repeat_index")
                self.assertIsInstance(choice, ast.Constant)
                self.assertIsNone(choice.value)
                for loop in (node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While, ast.AsyncFor))):
                    self.assertNotIn(selections[0], list(ast.walk(loop)))
                    self.assertNotIn(fits[0], list(ast.walk(loop)))

    def test_seed_settings_and_campaign_version_agree_and_collection_only_reads_saved_results(self) -> None:
        """Verify seed settings and campaign version agree and collection only reads saved results.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        for path in (NOTEBOOKS / name for name in NOTEBOOK_NAMES):
            with self.subTest(notebook=path.name):
                notebook = nbformat.read(path, as_version=4)
                tree = ast.parse("\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code"))
                settings = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                            for target in node.targets if isinstance(target, ast.Name)}
                # Only freeze, confirmation, and collection share the frozen campaign.
                if path.name in NOTEBOOK_NAMES[1:11]:
                    campaign = settings["CAMPAIGN"]
                    self.assertTrue(any(isinstance(node, ast.Constant) and isinstance(node.value, str)
                                        and (node.value == workflow.CAMPAIGN_VERSION or
                                             node.value.endswith("/" + workflow.CAMPAIGN_VERSION))
                                        for node in ast.walk(campaign)))
                # Apply this case only when path.name.startswith('00_').
                if path.name.startswith("00_"):
                    self.assertEqual(ast.literal_eval(settings["SEED"]), 17)
                # Apply this case only when path.name.startswith('01_').
                if path.name.startswith("01_"):
                    self.assertEqual(ast.literal_eval(settings["SEEDS"]), SEEDS)
                # Apply this case only when path.name.startswith(('01_', '10_')).
                if path.name.startswith(("01_", "10_")):
                    forbidden = {"train_model", "get_model", "get_datasets", "load_run", "load_development",
                                 "sample", "predict", "predict_class", "evaluate"}
                    calls = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                             for node in ast.walk(tree) if isinstance(node, ast.Call)
                             and isinstance(node.func, (ast.Name, ast.Attribute))}
                    self.assertFalse(calls & forbidden, "Preparation and collection must not execute model/data APIs.")


class SavedDevelopmentReviewTests(unittest.TestCase):
    """Use deliberately synthetic late-task observations without model execution."""

    def test_compact_learning_view_keeps_scalar_metrics_and_skips_plots(self) -> None:
        """Honor details=False even when the native result mapping is populated.

        Args:
            None. This case owns synthetic saved outcomes and a temporary view path.

        Returns:
            checked (None): None; only the full-precision scalar metrics CSV is written.

        Raises:
            AssertionError: If compact mode evaluates diagnostic curves, plots, or
                writes extra view artifacts instead of preserving the scalar values.
        """
        from notebooks.thesis.presentation import show_learning_results
        config = load_route_config(TEMPLATES["cifar10"])
        config.common.training.results_path = str(self.run)
        config.common.continually_learn.experiment_phase = "confirmation"
        config.common.continually_learn.class_num = 4
        config.common.continually_learn.class_order = [0, 1, 2, 3]
        config.common.continually_learn.task_groups = [[0, 1], [2, 3]]
        matrix = np.asarray([[0.5, np.nan], [0.4, 0.7]])
        bundle = {"continual_details": {"ordinary_accuracy_matrix": matrix,
                  "generative_histories": [{"loss": [1.0, 0.5]}]}}
        target = self.directory / "compact"
        with patch("IPython.display.display") as display, \
                patch("common.continual_reporting.task_accuracy_summaries", side_effect=AssertionError("diagnostic curve requested")), \
                patch("common.utils.plot_history", side_effect=AssertionError("history plot requested")):
            run, views = show_learning_results(config, bundle, output_dir=target, details=False)
        self.assertEqual((run, views), (self.run, target))
        self.assertEqual({path.name for path in target.iterdir()}, {"metrics.csv"})
        scalar = pd.read_csv(target / "metrics.csv").set_index("metric")["value"]
        self.assertAlmostEqual(scalar["final_average_accuracy"], 55.0)
        self.assertAlmostEqual(scalar["average_forgetting"], 10.0)
        self.assertEqual(display.call_count, 1)

    def setUp(self) -> None:
        """Prepare isolated synthetic fixtures.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-development-review-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.run = self.directory / "SYNTHETIC_NOT_THESIS_RESULTS"
        self.run.mkdir()
        self.route = []
        self.tasks = []
        for task in range(1, 11):
            self.route.append({"task": task,
                "acquisition": {"updates": 1000, "focus_class_updates": {str(c): 100 for c in range(10)},
                                "untrained_focus_classes": [], "example_draws": 32000},
                "consolidation": {"updates": 2000,
                                  "focus_class_updates": {str(c): 2000 // (10 * task) + int(c < 2000 % (10 * task))
                                                          for c in range(10 * task)},
                                  "untrained_focus_classes": [], "example_draws": 64000},
                "before_consolidation": {"input_sha256": f"fixed-{task}", "examples": 8, "split": "validation",
                                         "clean_accuracy": .6, "old_accuracy": None if task == 1 else .5,
                                         "new_accuracy": .7, "representation": {"centered_effective_rank": 4.}},
                "after_consolidation": {"input_sha256": f"fixed-{task}", "examples": 8, "split": "validation",
                                        "clean_accuracy": .65, "old_accuracy": None if task == 1 else .6,
                                        "new_accuracy": .6, "representation": {"centered_effective_rank": 3.}}})
            self.tasks.extend([
                {"task_index": task - 1, "phase": "resource", "metric": "seconds/task_total", "value": task},
                {"task_index": task - 1, "phase": "resource", "metric": "seconds/generator_fit", "value": .8 * task},
                {"task_index": task - 1, "phase": "resource", "metric": "current_examples_exposed", "value": 4000},
            ])
        (self.run / "route_metrics.json").write_text(json.dumps(self.route), encoding="utf-8")
        (self.run / "section11.json").write_text(json.dumps({"tasks": [
            {"task": 10, "resource_measurement": {"sampled_process_peak_rss_bytes": 1234,
                "tf_allocator_devices": {"GPU:0": {"peak": 5678}}},
             "generated_memory": {"available": True, "summary": {"label_consistency": .4}}}]}), encoding="utf-8")
        pd.DataFrame(self.tasks).to_csv(self.run / "task_metrics.csv", index=False)
        pd.DataFrame([{"task": 10, "joint_updates": 5700, "acquisition_updates": 1000,
                       "consolidation_updates": 2000}]).to_csv(self.run / "route_resources.csv", index=False)

    def test_saved_review_keeps_signed_changes_actual_coverage_and_nonoverlapping_runtime(self) -> None:
        """Verify saved review keeps signed changes actual coverage and nonoverlapping runtime.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        original = {path.name: path.read_bytes() for path in self.run.iterdir()}
        with ExitStack() as stack:
            for name in ("common.train.train_model", "common.model.get_model", "common.dataloader.get_datasets"):
                stack.enter_context(patch(name, side_effect=AssertionError("Saved review must not execute model/data APIs.")))
            tables = review_development_run(self.run, output_dir=self.directory / "SYNTHETIC_REVIEW")
        coverage = tables["gate_coverage"]
        final = coverage.loc[coverage.task.eq(10) & coverage.phase.eq("consolidation")].iloc[0]
        self.assertEqual(final["gates_observed"], 100)
        self.assertEqual(final["minimum_visits_per_gate"], 20)
        effects = tables["deployed_classifier_and_hidden_phase_changes"]
        self.assertTrue(effects.loc[effects.task.eq(1) & effects.measurement.eq("old_accuracy"),
                                    "change_after_minus_before"].isna().all())
        self.assertAlmostEqual(effects.loc[effects.task.eq(10) & effects.measurement.eq("old_accuracy"),
                                           "change_after_minus_before"].iloc[0], 10.)
        self.assertAlmostEqual(effects.loc[effects.task.eq(10) & effects.measurement.eq("new_accuracy"),
                                           "change_after_minus_before"].iloc[0], -10.)
        self.assertEqual(tables["measured_task_runtime"].iloc[0]["sum_measured_task_seconds"], 55.)
        self.assertEqual(tables["measured_task_runtime"].iloc[0]["completed_tasks_with_timer"], 10)
        self.assertEqual(set(tables["sampled_and_allocator_memory"]["bytes"]), {1234, 5678})
        self.assertEqual(original, {path.name: path.read_bytes() for path in self.run.iterdir()})

    def test_missing_timer_and_absent_baseline_phases_remain_unavailable(self) -> None:
        """Verify missing timer and absent baseline phases remain unavailable.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        self.route[-1].pop("before_consolidation")
        self.route[-1].pop("after_consolidation")
        (self.run / "route_metrics.json").write_text(json.dumps(self.route), encoding="utf-8")
        pd.DataFrame([row for row in self.tasks if not
                      (row["task_index"] == 9 and row["metric"] == "seconds/task_total")]).to_csv(
                          self.run / "task_metrics.csv", index=False)
        tables = review_development_run(self.run)
        self.assertTrue(pd.isna(tables["measured_task_runtime"].iloc[0]["sum_measured_task_seconds"]))
        self.assertEqual(tables["measured_task_runtime"].iloc[0]["completed_tasks_with_timer"], 9)
        effects = tables["deployed_classifier_and_hidden_phase_changes"]
        self.assertTrue(effects.loc[effects.task.eq(10), "change_after_minus_before"].isna().all())
        with self.assertRaisesRegex(ValueError, "outside the original"):
            review_development_run(self.run, output_dir=self.run / "review")


# Run this isolated unittest module when invoked as a script.
if __name__ == "__main__":
    unittest.main()
