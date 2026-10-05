"""Keep distilled-CIFAR recipe selection separate from the held-out test split."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, sentinel

import numpy as np


class NotebookValidationSplitTests(unittest.TestCase):
    """Execute actual notebook selection cells with inert data and model substitutes."""

    def setUp(self) -> None:
        """Read maintained notebook cells without running setup, training or downloads."""

        path = Path(__file__).resolve().parents[2] / "files/notebooks/CIFAR10 distilled DiT CLF.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.sources = ["".join(cell["source"]) for cell in notebook["cells"]
                        if cell["cell_type"] == "code"]
        self.trees = [ast.parse(source) for source in self.sources]

    def test_training_validation_and_test_arrays_keep_their_own_roles(self) -> None:
        """The real split cell reserves seeded validation and never fits on test rows."""

        split = next(tree for tree in self.trees if any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "load_cifar10" for node in ast.walk(tree)
        ))
        assignments = [node for node in split.body if isinstance(node, ast.Assign)]
        self.assertEqual(len(assignments), 4)
        loader_assignment = assignments[0]
        self.assertEqual([target.id for target in loader_assignment.targets[0].elts], [
            "x_train", "y_train", "x_val", "y_val", "x_test", "y_test"
        ])
        images = [np.full((size, 2, 2, 1), value) for size, value in ((4, 1), (2, 2), (2, 3))]
        labels = [np.arange(len(value)).reshape(-1, 1) for value in images]
        loader = Mock(return_value=tuple(value for pair in zip(images, labels) for value in pair))
        datasets = Mock(side_effect=[sentinel.trainset, sentinel.valset, sentinel.testset])
        namespace = {"load_cifar10": loader, "get_dataset": datasets}
        code = ast.Module(body=assignments, type_ignores=[])
        exec(compile(code, "notebook_split", "exec"), namespace)
        loader.assert_called_once_with(indices=None, validation_ratio=0.1, verbose=0, seed=42)
        self.assertEqual(datasets.call_count, 3)
        for index, call in enumerate(datasets.call_args_list):
            self.assertIs(call.args[0], images[index])
            np.testing.assert_array_equal(call.args[1], labels[index].reshape(-1))
            expected_options = {} if index == 0 else {"shuffle_buffer": 0, "drop_remainder": False}
            self.assertEqual(call.kwargs, expected_options)
        self.assertIs(namespace["trainset"], sentinel.trainset)
        self.assertIs(namespace["valset"], sentinel.valset)
        self.assertIs(namespace["testset"], sentinel.testset)
        fits = [node for tree in self.trees for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and ast.unparse(node.func) in ("model.fit_teacher", "model.fit")]
        self.assertEqual(len(fits), 2)
        for fit in fits:
            self.assertEqual(ast.unparse(fit.args[0]), "trainset")
            validation = next(keyword.value for keyword in fit.keywords if keyword.arg == "validation_data")
            self.assertEqual(ast.unparse(validation), "valset")

    def test_only_validation_selects_the_single_final_test_recipe(self) -> None:
        """All twelve original recipes use validation; the winner receives one test call."""

        helper_index = next(i for i, tree in enumerate(self.trees) if any(
            isinstance(node, ast.FunctionDef) and node.name == "evaluate_validation_ensemble"
            for node in tree.body
        ))
        grid_indices = [i for i, tree in enumerate(self.trees) if any(
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "evaluate_validation_ensemble" for node in tree.body
        )]
        final_index = next(i for i, tree in enumerate(self.trees) if any(
            isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "test_accuracy" for target in node.targets
            ) for node in tree.body
        ))
        expected_recipes = [
            {"max_t": depth, "t_range_drop_rate": rate}
            for depth in (256, 128, 64, 32, 16) for rate in (0.75, 0.99)
        ] + [{"max_t": 8}, {"max_t": 4}]
        self.assertEqual(len(grid_indices), len(expected_recipes))
        actual_options = []
        for index, expected_recipe in zip(grid_indices, expected_recipes):
            call = next(node.value for node in self.trees[index].body
                        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call))
            options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
            self.assertIn("t_chunk_size", options)
            self.assertEqual({key: value for key, value in options.items() if key != "t_chunk_size"}, 
                             expected_recipe)
            actual_options.append(options)
        self.assertLess(helper_index, min(grid_indices))
        self.assertGreater(final_index, max(grid_indices))
        scores = [.7 + index * .001 for index in range(12)]
        scores[7] = .95
        model = Mock()
        model.evaluate_ensemble_accuracy.side_effect = scores + [.8]
        namespace = {"model": model, "valset": sentinel.valset, "testset": sentinel.testset}
        exec(self.sources[helper_index], namespace)
        for index in grid_indices:
            exec(self.sources[index], namespace)
        self.assertEqual(model.evaluate_ensemble_accuracy.call_count, 12)
        self.assertEqual(len(namespace["ensemble_validation_results"]), 12)
        self.assertEqual([result["options"] for result in namespace["ensemble_validation_results"]], actual_options)
        for call, options in zip(model.evaluate_ensemble_accuracy.call_args_list, actual_options):
            self.assertEqual(call.args, tuple([sentinel.valset]))
            self.assertEqual(call.kwargs, {"seed": 42, **options})
        exec(self.sources[final_index], namespace)
        self.assertEqual(model.evaluate_ensemble_accuracy.call_count, 13)
        final_call = model.evaluate_ensemble_accuracy.call_args
        self.assertEqual(final_call.args, tuple([sentinel.testset]))
        self.assertEqual(final_call.kwargs, {"seed": 42, **actual_options[7]})
        self.assertEqual(namespace["selected_ensemble_options"], actual_options[7])
        self.assertEqual(namespace["best_ensemble"]["accuracy"], .95)
        self.assertEqual(namespace["test_accuracy"], .8)
        exec(self.sources[helper_index], namespace)
        self.assertEqual(namespace["ensemble_validation_results"], [])
        self.assertEqual(model.evaluate_ensemble_accuracy.call_count, 13)

    def test_other_evaluations_never_consult_the_test_split(self) -> None:
        """The actual notebook contains exactly one held-out model evaluation site."""

        evaluations = [node for tree in self.trees for node in ast.walk(tree)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and ast.unparse(node.func) in ("model.evaluate", "model.evaluate_ensemble_accuracy")]
        dataset_names = [ast.unparse(call.args[0]) for call in evaluations]
        self.assertEqual(dataset_names.count("testset"), 1)
        self.assertEqual(dataset_names.count("valset"), len(dataset_names) - 1)
        test_calls = [call for call in evaluations if ast.unparse(call.args[0]) == "testset"]
        self.assertEqual([keyword.arg for keyword in test_calls[0].keywords], ["seed", None])
        self.assertEqual(ast.literal_eval(test_calls[0].keywords[0].value), 42)
        self.assertEqual(ast.unparse(test_calls[0].keywords[1].value), "selected_ensemble_options")


# Direct invocation runs the same isolated notebook contract tests as discovery.
if __name__ == "__main__":
    unittest.main()
