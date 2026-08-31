"""Configuration contract tests for replay and cross-cutting boundaries."""

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
    save_config,
)
from common.replay_buffer import ReplayBuffer


class ReplayConfigTests(unittest.TestCase):
    """Verify replay policies are defaulted, normalized, and serialized."""

    def test_fifo_is_the_dataclass_and_yaml_default(self) -> None:
        """The repository example is fully loadable and explicitly uses FIFO.

        Returns:
            None: Dataclass and current-schema YAML defaults are compared.
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
        self.assertTrue(restored.continually_learn.use_buffer)
        self.assertEqual(
            restored.continually_learn.optimizer_steps_per_epoch,
            9,
        )
        self.assertEqual(
            restored.continually_learn.buffer_kwargs,
            config.continually_learn.buffer_kwargs,
        )

    def test_behavior_owner_validates_replay_strategy(self) -> None:
        """A config stores a strategy and ReplayBuffer decides if it is valid.

        Returns:
            None: Validation occurs only when replay behavior is constructed.
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
    """Verify shared task, result-path, and clipping contracts."""

    def test_task_config_is_passive_and_public_normalizer_is_strict(self) -> None:
        """Task values are interpreted only at task-bearing entry points.

        Returns:
            None: Passive storage and explicit normalization are distinguished.
        """

        self.assertEqual(normalize_training_task("Joint"), "joint")
        self.assertEqual(TrainingConfig(task="CLASSIFICATION").task, "CLASSIFICATION")
        self.assertEqual(Config(training={"task": "prediction"}).training.task, "prediction")
        with self.assertRaisesRegex(ValueError, "training task"):
            normalize_training_task("prediction")
        with self.assertRaisesRegex(TypeError, "training task"):
            normalize_training_task(None)

    def test_duplicate_yaml_keys_are_rejected(self) -> None:
        """Refuse ambiguous configs instead of silently taking the last value.

        Returns:
            None: A duplicate nested task key raises a marked YAML error.
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
