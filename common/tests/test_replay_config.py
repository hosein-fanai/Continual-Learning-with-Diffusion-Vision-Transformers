"""Configuration contract tests for fixed-memory replay strategies."""

from __future__ import annotations

import tempfile
import unittest

from pathlib import Path

import yaml

from common.config import Config, ContinuallyLearnConfig, load_config, save_config


class ReplayConfigTests(unittest.TestCase):
    """Verify replay policies are defaulted, normalized, and serialized."""

    def test_fifo_is_the_dataclass_and_yaml_default(self) -> None:
        """Both supported configuration entry points explicitly select FIFO.

        Returns:
            None: Dataclass and repository YAML defaults are compared.
        """

        self.assertEqual(
            Config().continually_learn.buffer_kwargs["strategy"],
            "fifo",
        )
        default_path = Path(__file__).parents[2] / "configs" / "default.yaml"
        default_data = yaml.safe_load(default_path.read_text(encoding="utf-8"))
        loaded = ContinuallyLearnConfig(**default_data["continually_learn"])
        self.assertEqual(
            loaded.buffer_kwargs["strategy"],
            "fifo",
        )
        self.assertIsNone(Config().continually_learn.optimizer_steps_per_epoch)
        self.assertIsNone(loaded.optimizer_steps_per_epoch)

    def test_partial_and_alias_mappings_are_canonicalized(self) -> None:
        """Omitted policy becomes FIFO and documented aliases become canonical.

        Returns:
            None: Partial controls retain values and receive normalized policy.
        """

        partial = ContinuallyLearnConfig(buffer_kwargs={"maxlen": 23})
        self.assertEqual(partial.buffer_kwargs, {
            "maxlen": 23,
            "strategy": "fifo",
        })
        aliased = ContinuallyLearnConfig(buffer_kwargs={
            "maxlen": 17,
            "strategy": "class-balanced",
        })
        self.assertEqual(aliased.buffer_kwargs["maxlen"], 17)
        self.assertEqual(
            aliased.buffer_kwargs["strategy"],
            "class_balanced",
        )
        explicit_none = ContinuallyLearnConfig(buffer_kwargs=None)
        self.assertEqual(explicit_none.buffer_kwargs["strategy"], "fifo")

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

    def test_invalid_strategy_and_container_fail_early(self) -> None:
        """Malformed replay controls fail during typed config construction.

        Returns:
            None: Invalid value and container errors are asserted.
        """

        with self.assertRaisesRegex(ValueError, "strategy"):
            ContinuallyLearnConfig(buffer_kwargs={"strategy": "recent"})
        with self.assertRaisesRegex(TypeError, "mapping"):
            ContinuallyLearnConfig(buffer_kwargs=["fifo"])
        for invalid_steps in (True, 0, -1, 1.5):
            with self.subTest(optimizer_steps_per_epoch=invalid_steps):
                with self.assertRaisesRegex(ValueError, "optimizer_steps"):
                    ContinuallyLearnConfig(
                        optimizer_steps_per_epoch=invalid_steps
                    )


# Run this focused suite directly when the module is executed as a script.
if __name__ == "__main__":
    unittest.main()
