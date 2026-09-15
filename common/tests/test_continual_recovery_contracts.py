"""Check exact continual budgets and behavior-aware compile recovery identities."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np
import tensorflow as tf

from common.learner import _run_continual_tasks
from common.recovery import _recovery_descriptor, compile_recovery_descriptor, fingerprint_state


def _scaled_loss(scale: float) -> object:
    """Create an immutable-capture loss without external runtime dependencies.

    Args:
        scale (float): Multiplicative loss coefficient.

    Returns:
        loss (object): Tensor-compatible callable with the captured coefficient.
    """
    def loss(y_true: object, y_pred: object) -> object:
        """Return scaled squared residuals for tensor or scalar inputs.

        Args:
            y_true (object): Reference values.
            y_pred (object): Corresponding predictions.

        Returns:
            residuals (object): Elementwise scaled squared errors.
        """
        return scale * (y_true - y_pred) ** 2

    return loss


class ContinualBudgetTests(unittest.TestCase):
    """Reject altered or empty budgets before runtime, data, or model mutation."""

    def test_invalid_budgets_fail_before_runtime(self) -> None:
        """Reject zeros, negative counts, fractions, strings and booleans.

        Returns:
            result (None): Every invalid count fails before any setup side effect.
        """
        for name in ("replay_candidate_multiplier", "optimizer_steps_per_epoch"):
            for value in (0, -1, 1.9, 1.0, True, np.bool_(True), "2"):
                with self.subTest(name=name, value=value):
                    loader = Mock()
                    with patch("common.learner.configure_runtime") as runtime:
                        with self.assertRaisesRegex(ValueError, name + " must be a positive integer"):
                            _run_continual_tasks(class_num=2, load_dataset_fn=loader, **{name: value})
                    runtime.assert_not_called()
                    loader.assert_not_called()

    def test_valid_integer_and_optional_budgets_reach_runtime(self) -> None:
        """Preserve positive Python/NumPy integers and the omitted update budget.

        Returns:
            result (None): Valid values pass validation and reach the setup boundary.
        """
        for options in (
            {"replay_candidate_multiplier": 1, "optimizer_steps_per_epoch": None},
            {"replay_candidate_multiplier": np.int64(2), "optimizer_steps_per_epoch": np.int64(3)},
        ):
            with self.subTest(options=options):
                with patch("common.learner.configure_runtime", side_effect=RuntimeError("runtime boundary")):
                    with self.assertRaisesRegex(RuntimeError, "runtime boundary"):
                        _run_continual_tasks(class_num=2, load_dataset_fn=Mock(), **options)


class CompileRecoveryTests(unittest.TestCase):
    """Authenticate supported callable behavior without restricting ordinary fitting."""

    def test_immutable_loss_capture_changes_checkpoint_identity(self) -> None:
        """Distinguish objectives whose function name and code are otherwise identical.

        Returns:
            result (None): A changed coefficient changes the strict fingerprint.
        """
        first = {"loss": _scaled_loss(1.), "metrics": [_scaled_loss(2.)]}
        same = {"loss": _scaled_loss(1.), "metrics": (_scaled_loss(2.),)}
        changed_loss = {"loss": _scaled_loss(10.), "metrics": [_scaled_loss(2.)]}
        changed_metric = {"loss": _scaled_loss(1.), "metrics": [_scaled_loss(3.)]}
        expected = fingerprint_state(compile_recovery_descriptor(first, strict=True))
        self.assertEqual(expected, fingerprint_state(compile_recovery_descriptor(same, strict=True)))
        for candidate in (changed_loss, changed_metric):
            self.assertNotEqual(expected, fingerprint_state(compile_recovery_descriptor(candidate, strict=True)))

    def test_ordinary_tensorflow_loss_callable_remains_permitted(self) -> None:
        """Keep ordinary callables and reject opaque strict dependencies explicitly.

        Returns:
            result (None): Non-checkpoint descriptors remain unchanged; strict calls
                direct the caller to a configured loss or metric.
        """
        options = {"loss": tf.keras.losses.mse, "metrics": []}
        self.assertEqual(compile_recovery_descriptor(options, strict=False), _recovery_descriptor(options))
        with self.assertRaisesRegex(ValueError, "configured Keras Loss or Metric with get_config"):
            compile_recovery_descriptor(options, strict=True)

    def test_declared_keras_loss_and_metrics_retain_existing_identity(self) -> None:
        """Preserve declarative built-in options used by current checkpoints.

        Returns:
            result (None): Strings and configured Keras objects keep their descriptors.
        """
        for options in (
            {"optimizer": "adam", "loss": "mse", "metrics": ["accuracy"]},
            {"loss": tf.keras.losses.MeanSquaredError(), "metrics": [tf.keras.metrics.MeanAbsoluteError()]},
        ):
            self.assertEqual(compile_recovery_descriptor(options, strict=True), _recovery_descriptor(options))


# Direct execution runs the same focused cases as unittest discovery.
if __name__ == "__main__":
    unittest.main()
