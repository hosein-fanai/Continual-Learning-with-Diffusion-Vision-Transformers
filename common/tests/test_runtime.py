"""Focused unit tests for process-wide experiment runtime configuration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import random
import subprocess
import sys

import numpy as np
import tensorflow as tf

from common.runtime import (
    configure_runtime,
    derive_seed,
    effective_seed,
    validate_model_dtype_policy,
)
from common.validation import require


class RuntimeTests(TestCase):
    """Verify seed precedence, child streams, policies, and global seeding."""

    def setUp(self: RuntimeTests) -> None:
        """Remember the caller's Keras policy before each test.

        Returns:
            None.
        """

        self.previous_policy = tf.keras.mixed_precision.global_policy().name

    def tearDown(self: RuntimeTests) -> None:
        """Restore the caller's Keras policy after each test.

        Returns:
            None.
        """

        tf.keras.mixed_precision.set_global_policy(self.previous_policy)

    def test_require_preserves_assertion_semantics(self: RuntimeTests) -> None:
        """Retain truth testing, messages, and bare assertion arguments.

        Returns:
            None.
        """

        self.assertIsNone(require(object()))
        with self.assertRaises(AssertionError) as bare_error:
            require(0)
        self.assertEqual(bare_error.exception.args, ())

        marker = object()
        with self.assertRaises(AssertionError) as message_error:
            require([], marker)
        self.assertIs(message_error.exception.args[0], marker)

    def test_require_remains_active_under_optimization(
        self: RuntimeTests,
    ) -> None:
        """Keep required invariants active in a Python ``-O`` subprocess.

        Returns:
            None.
        """

        project_root = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [
                sys.executable,
                "-O",
                "-c",
                "from common.validation import require; "
                "require(False, 'active under -O')",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("AssertionError: active under -O", completed.stderr)

    def test_seed_precedence_and_normalization(self: RuntimeTests) -> None:
        """Apply continual precedence and normalize seed inputs.

        Returns:
            None.
        """

        continual_config = SimpleNamespace(
            training=SimpleNamespace(task="continual", seed=11),
            continually_learn=SimpleNamespace(seed=22),
        )
        fallback_config = SimpleNamespace(
            training=SimpleNamespace(task="continual", seed=11),
            continually_learn=SimpleNamespace(seed=None),
        )
        ordinary_config = SimpleNamespace(
            training=SimpleNamespace(task="classification", seed=11),
            continually_learn=SimpleNamespace(seed=22),
        )
        self.assertEqual(effective_seed(continual_config), 22)
        self.assertEqual(effective_seed(fallback_config), 11)
        self.assertEqual(effective_seed(ordinary_config), 11)
        self.assertEqual(effective_seed(seed=7, task="continual"), 7)
        self.assertIsNone(effective_seed())
        self.assertEqual(effective_seed(seed=True), 1)
        self.assertEqual(effective_seed(seed=1.5), 1)
        self.assertEqual(effective_seed(seed="7"), 7)
        with self.assertRaises(ValueError):
            effective_seed(seed=-1)

    def test_child_seeds_are_stable_and_isolated(self: RuntimeTests) -> None:
        """Derive repeatable, named, bounded child RNG streams.

        Returns:
            None.
        """

        first = derive_seed(1234, "dataloader", 2, "shuffle")
        self.assertEqual(first, 2_042_430_465)
        self.assertEqual(first, derive_seed(1234, "dataloader", 2, "shuffle"))
        self.assertNotEqual(first, derive_seed(1234, "dataloader", 3, "shuffle"))
        self.assertNotEqual(first, derive_seed(1234, "replay", 2, "shuffle"))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2 ** 31 - 1)
        self.assertIsNone(derive_seed(None, "dataloader"))
        with self.assertRaises(ValueError):
            derive_seed(1234)

    def test_dtype_policy_validation(self: RuntimeTests) -> None:
        """Accept floating and mixed policies while rejecting invalid ones.

        Returns:
            None.
        """

        self.assertEqual(configure_runtime(dtype_policy="float64"), "float64")
        self.assertEqual(
            configure_runtime(dtype_policy="mixed_float16"),
            "mixed_float16",
        )
        with self.assertRaises(ValueError):
            configure_runtime(dtype_policy="not_a_policy")

    def test_configuration_seeds_all_global_generators(self: RuntimeTests) -> None:
        """Reset Python, NumPy, and TensorFlow from one effective seed.

        Returns:
            None.
        """

        policy_name = configure_runtime(
            seed=17,
            dtype_policy="float64",
        )
        first = (
            random.random(),
            float(np.random.random()),
            float(tf.random.uniform(())),
        )
        configure_runtime(
            seed=17,
            dtype_policy="float64",
        )
        second = (
            random.random(),
            float(np.random.random()),
            float(tf.random.uniform(())),
        )
        self.assertEqual(first, second)
        self.assertEqual(policy_name, "float64")
        self.assertEqual(
            tf.keras.mixed_precision.global_policy().name,
            "float64",
        )

    def test_deterministic_ops_are_explicit_and_seeded(self: RuntimeTests) -> None:
        """Enable deterministic kernels only when requested with a seed.

        Returns:
            None.
        """

        with patch.object(
            tf.config.experimental,
            "enable_op_determinism",
            create=True,
        ) as enable:
            configure_runtime(seed=9, deterministic_ops=False)
            enable.assert_not_called()
            configure_runtime(seed=9, deterministic_ops=True)
            enable.assert_called_once_with()
        with self.assertRaises(ValueError):
            configure_runtime(seed=None, deterministic_ops=True)

    def test_model_dtype_policy_normalization(self: RuntimeTests) -> None:
        """Resolve policy names while reusing existing policy objects.

        Returns:
            None.
        """

        policy = tf.keras.mixed_precision.Policy("float32")
        self.assertIs(validate_model_dtype_policy(policy), policy)
        self.assertEqual(
            validate_model_dtype_policy("mixed_float16").name,
            "mixed_float16",
        )

    def test_config_adapter_uses_continual_seed(self: RuntimeTests) -> None:
        """Read runtime controls from a typed-config-compatible object.

        Returns:
            None.
        """

        config = SimpleNamespace(
            training=SimpleNamespace(
                task="continual",
                seed=13,
                dtype_policy="float32",
                deterministic_ops=False,
            ),
            continually_learn=SimpleNamespace(seed=29),
        )
        selected_seed = effective_seed(config)
        policy_name = configure_runtime(
            seed=selected_seed,
            dtype_policy=config.training.dtype_policy,
            deterministic_ops=config.training.deterministic_ops,
        )
        self.assertEqual(selected_seed, 29)
        self.assertEqual(policy_name, "float32")
        # Missing structure is reported naturally by the behavior owner.
        with self.assertRaises(AttributeError):
            effective_seed(SimpleNamespace())


# Run only this focused suite when the file is executed directly.
if __name__ == "__main__":
    main()
