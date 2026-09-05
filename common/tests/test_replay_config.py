"""Regression checks for replay settings and shared Config validation.

In-memory mappings and temporary YAML files verify defaults, independent mutable settings,
compact/full serialization, task normalization, and deferred runtime validation. Distilled
V2 ensemble configurations are checked without starting a training run.

Inputs are fixtures constructed by the test methods and their helpers. Tests return no
application result: unittest records assertion outcomes and errors. Run this module directly
or through ``python -m unittest`` discovery. Importing it defines fixtures and cases; it
does not itself start a test run.
"""

from __future__ import annotations

import tempfile
import unittest

import yaml

from pathlib import Path

from common.config import (
    Config,
    ContinuallyLearnConfig,
    OptimizerConfig,
    TrainingConfig,
    load_config,
    normalize_training_task,
    resolve_continual_schedule,
    save_config,
)
from common.replay_buffer import ReplayBuffer


class ReplayConfigTests(unittest.TestCase):
    """Verify replay policies are defaulted, normalized, and serialized.

    The unittest runner executes the selected test method with its local fixtures;
    individual methods describe the configurations and failure cases they exercise. There is
    no application model or experiment result returned by constructing this test case.

    Args:
        methodName (str): Test method selected by unittest. Defaults to ``"runTest"``;
            discovery supplies each named ``test_*`` method.

    Attributes:
        _testMethodName (str): Selected method name maintained by unittest.
    """

    def test_fifo_is_the_dataclass_and_yaml_default(self) -> None:
        """The repository example is fully loadable and explicitly uses FIFO.

        Returns:
            None: Dataclass and current-schema YAML defaults are compared.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        self.assertEqual(
            Config().continually_learn.buffer_kwargs["strategy"],
            "fifo",
        )
        default_path = Path(__file__).parents[2] / "configs" / "default.yaml"
        loaded_config = load_config(default_path)
        loaded = loaded_config.continually_learn
        self.assertEqual(
            loaded.buffer_kwargs["strategy"],
            "fifo",
        )
        self.assertEqual(loaded_config.dataset.name, "cifar10")
        self.assertEqual(loaded_config.model.name, "cnn")
        self.assertEqual(loaded_config.training.task, "classification")
        self.assertIsNone(Config().continually_learn.optimizer_steps_per_epoch)
        self.assertIsNone(loaded.optimizer_steps_per_epoch)

    def test_runtime_options_remain_passive_config_values(self) -> None:
        """Runtime policy validation belongs to the replay implementation.

        Returns:
            None: Configuration retains values without interpreting them.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        partial = ContinuallyLearnConfig(buffer_kwargs={"maxlen": 23})
        self.assertEqual(partial.buffer_kwargs, {"maxlen": 23})
        canonical = ContinuallyLearnConfig(buffer_kwargs={
            "maxlen": 17,
            "strategy": "class_balanced",
        })
        self.assertEqual(canonical.buffer_kwargs["maxlen"], 17)
        self.assertEqual(canonical.buffer_kwargs["strategy"], "class_balanced")
        explicit_none = ContinuallyLearnConfig(buffer_kwargs=None)
        self.assertIsNone(explicit_none.buffer_kwargs)

    def test_strategy_round_trips_through_yaml(self) -> None:
        """A nondefault reservoir policy survives full YAML serialization.

        Returns:
            None: The reloaded config retains reservoir and companion controls.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        config = Config(continually_learn={
            "use_buffer": True,
            "optimizer_steps_per_epoch": 9,
            "buffer_kwargs": {
                "maxlen": 101,
                "sample_num": 11,
                "insert_num": 7,
                "strategy": "reservoir",
            },
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            save_config(config, path)
            restored = load_config(path)
        self.assertEqual(restored, config)
        self.assertTrue(restored.continually_learn.use_buffer)
        self.assertEqual(
            restored.continually_learn.optimizer_steps_per_epoch,
            9,
        )
        self.assertEqual(
            restored.continually_learn.buffer_kwargs,
            config.continually_learn.buffer_kwargs,
        )

    def test_distilled_v2_ensemble_config_round_trips_in_both_formats(self) -> None:
        """Full and compact YAML preserve the complete continual HPO setup.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """
        config = Config(
            model={
                "name": "dit_classifier",
                "wrapper_name": "diffusion_classifier_v2",
                "dit_classifier": {
                    "clf_distil_token_type": "new_weight",
                },
                "diffusion_classifier_v2": {
                    "clf_distil_type": "soft",
                    "clf_distil_temperature": 2.0,
                    "clf_distil_loss_coef": 0.05,
                },
            },
            training={"task": "continual"},
            continually_learn={
                "class_order": [2, 0, 1],
                "task_groups": [[2, 0], [1]],
                "use_generative_model_classifier": True,
                "train_classifier_separately": True,
                "use_distillation": True,
                "use_ensemble_accuracy": True,
                "ensemble_accuracy_kwargs": {"max_t": 4, "weighted": True},
            },
            hpo={
                "objective_metrics": [
                    "final_average_accuracy", "average_forgetting",
                ],
                "objective_directions": ["maximize", "minimize"],
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            for shorten in (False, True):
                with self.subTest(shorten=shorten):
                    path = Path(directory) / "config.yaml"
                    save_config(config, path, shorten=shorten)
                    self.assertEqual(load_config(path), config)

    def test_behavior_owner_validates_replay_strategy(self) -> None:
        """A config stores a strategy and ReplayBuffer decides if it is valid.

        Returns:
            None: Validation occurs only when replay behavior is constructed.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        config = ContinuallyLearnConfig(
            buffer_kwargs={"strategy": "recent"},
            baseline="experimental",
            optimizer_steps_per_epoch=0,
        )
        self.assertEqual(config.buffer_kwargs["strategy"], "recent")
        self.assertEqual(config.baseline, "experimental")
        self.assertEqual(config.optimizer_steps_per_epoch, 0)
        self.assertFalse(hasattr(config, "teacher_network_name"))
        with self.assertRaisesRegex(ValueError, "strategy"):
            ReplayBuffer(
                maxlen=10,
                strategy=config.buffer_kwargs["strategy"],
            )


class CoreConfigValidationTests(unittest.TestCase):
    """Verify shared task, result-path, and clipping contracts.

    The unittest runner executes the selected test method with its local fixtures;
    individual methods describe the configurations and failure cases they exercise. There is
    no application model or experiment result returned by constructing this test case.

    Args:
        methodName (str): Test method selected by unittest. Defaults to ``"runTest"``;
            discovery supplies each named ``test_*`` method.

    Attributes:
        _testMethodName (str): Selected method name maintained by unittest.
    """

    def test_nested_model_sections_and_kwargs_are_independent(self) -> None:
        """Nested mappings must convert to configs and return copied kwargs.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        config = Config(
            dataset={"batch_size": 4},
            model={
                "diffusion_transformer": {"dim": 16},
                "diffusion_classifier": {
                    "clf_distil_temperature": 0.0,
                    "clf_distil_scope": "EXPERIMENTAL",
                },
            },
        )
        self.assertEqual(config.dataset.batch_size, 4)
        self.assertEqual(config.model.diffusion_transformer.dim, 16)
        self.assertEqual(
            config.model.diffusion_classifier.clf_distil_temperature,
            0.0
        )
        self.assertEqual(
            config.model.diffusion_classifier.clf_distil_scope,
            "EXPERIMENTAL",
        )

        kwargs_copy = config.model.diffusion_transformer.kwargs()
        kwargs_copy["dim"] = 99
        self.assertEqual(config.model.diffusion_transformer.dim, 16)

    def test_explicit_continual_schedule_is_preserved(self) -> None:
        """Supplied class order and task groups must remain authoritative.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        order, groups = resolve_continual_schedule(
            3,
            class_order=[2, 0, 1],
            task_groups=[[2], [0, 1]],
            available_class_num=3,
        )
        self.assertEqual(order, [2, 0, 1])
        self.assertEqual(groups, [[2], [0, 1]])

    def test_task_config_is_passive_and_public_normalizer_is_strict(self) -> None:
        """Task values are interpreted only at task-bearing entry points.

        Returns:
            None: Passive storage and explicit normalization are distinguished.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        self.assertEqual(normalize_training_task("Joint"), "joint")
        self.assertEqual(TrainingConfig(task="CLASSIFICATION").task, "CLASSIFICATION")
        self.assertEqual(Config(training={"task": "prediction"}).training.task, "prediction")
        with self.assertRaisesRegex(ValueError, "training task"):
            normalize_training_task("prediction")
        with self.assertRaises(AttributeError):
            normalize_training_task(None)

    def test_duplicate_yaml_keys_are_rejected(self) -> None:
        """Refuse ambiguous configs instead of silently taking the last value.

        Returns:
            None: A duplicate nested task key raises a marked YAML error.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.yaml"
            path.write_text(
                "training:\n"
                "  task: classification\n"
                "  task: continual\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                yaml.constructor.ConstructorError,
                "duplicate key 'task'",
            ):
                load_config(path)

            merge_path = Path(directory) / "merge.yaml"
            merge_path.write_text(
                "training:\n"
                "  <<: &defaults\n"
                "    task: classification\n"
                "  task: continual\n",
                encoding="utf-8",
            )
            self.assertEqual(load_config(merge_path).training.task, "continual")

    def test_training_result_path_uses_filesystem_normalization(self) -> None:
        """Normalize path-like values without imposing an extra text policy.

        Returns:
            None: Text and byte filesystem representations are preserved.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        config = TrainingConfig(results_path=Path("artifacts"))
        self.assertEqual(config.results_path, "artifacts")
        self.assertEqual(
            TrainingConfig(results_path=b"artifacts").results_path,
            b"artifacts",
        )

    def test_optimizer_config_defers_clipnorm_validation(self) -> None:
        """The optimizer factory, rather than its data container, owns bounds.

        Returns:
            None: Representative runtime values are stored unchanged.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        for value in (2.5, True, "1.0", 0.0, -1.0, float("nan")):
            with self.subTest(value=value):
                stored = OptimizerConfig(clipnorm=value).clipnorm
                # Compare NaN through its defining non-reflexive behavior.
                if isinstance(value, float) and value != value:
                    self.assertNotEqual(stored, stored)
                # All other passive values are preserved exactly.
                else:
                    self.assertEqual(stored, value)


# Run this focused suite directly when the module is executed as a script.
if __name__ == "__main__":
    unittest.main()
