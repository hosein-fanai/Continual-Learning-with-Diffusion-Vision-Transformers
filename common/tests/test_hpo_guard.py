"""Numerical-stop evidence without an Optuna study or model training."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import tensorflow as tf

from common.callbacks.hpo_guard import NonFiniteLossGuard, TrainingDiverged


class HpoGuardTests(unittest.TestCase):
    def _guard(self, **kwargs) -> NonFiniteLossGuard:
        guard = NonFiniteLossGuard(**kwargs)
        guard.set_model(SimpleNamespace())
        guard.on_train_begin()
        return guard

    def test_finite_underperformance_is_never_pruned(self) -> None:
        guard = self._guard()
        guard.on_epoch_begin(0)
        guard.on_train_batch_end(0, {"loss": 1e100, "classifier_accuracy": 0.0})
        guard.on_test_batch_end(0, {"noise_loss": 1e100})
        logs = {"loss": 1e100, "val_noise_loss": 1e100, "val_classifier_accuracy": 0.0}
        guard.on_epoch_end(0, logs)
        self.assertEqual(guard.partial_history, [{"epoch": 0, "metrics": logs}])
        self.assertFalse(hasattr(guard.model, "stop_training"))

    def test_batch_hooks_are_noops_and_only_epoch_logs_are_checked(self) -> None:
        guard = self._guard()
        guard.on_train_batch_end(0, {"loss": np.nan})
        guard.on_test_batch_end(0, {"loss": np.inf})
        self.assertEqual(guard.last_finite_metrics, {})
        self.assertEqual(guard.partial_history, [])
        guard.on_epoch_end(0, {"loss": 0.5, "val_loss": 0.6})
        self.assertEqual(guard.last_finite_metrics, {"loss": 0.5, "val_loss": 0.6})

    def test_every_nonfinite_loss_is_captured_at_epoch_end(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            for metric in ("loss", "total_noise_loss", "val_clf_loss"):
                with self.subTest(value=value, metric=metric):
                    guard = self._guard(phase="generator")
                    guard.on_epoch_begin(4)
                    with self.assertRaises(TrainingDiverged) as raised:
                        guard.on_epoch_end(4, {metric: value, "classifier_accuracy": 0.2})
                    error = raised.exception
                    self.assertEqual(error.phase, "generator")
                    self.assertEqual(error.epoch, 4)
                    self.assertIsNone(error.batch)
                    self.assertEqual(error.evidence["hook"], "epoch_end")
                    self.assertEqual(error.metric, metric)
                    self.assertEqual(error.evidence["value"], str(value))
                    self.assertEqual(error.evidence["finite_metrics"], {"classifier_accuracy": 0.2})
                    self.assertIsNone(error.evidence_path)
                    json.dumps(error.evidence, allow_nan=False)

    def test_partial_history_and_evidence_survive_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            guard = self._guard(evidence_dir=directory)
            guard.on_epoch_begin(0)
            guard.on_epoch_end(0, {"loss": 0.5, "val_loss": 0.6})
            with self.assertRaises(TrainingDiverged) as raised:
                guard.on_epoch_end(1, {"loss": np.float32(np.nan), "noise_loss": np.inf})
            error = raised.exception
            self.assertEqual(error.evidence["last_finite_metrics"], {"loss": 0.5, "val_loss": 0.6})
            self.assertEqual(error.evidence["partial_history"], [
                {"epoch": 0, "metrics": {"loss": 0.5, "val_loss": 0.6}},
            ])
            self.assertEqual(error.evidence["nonfinite_losses"], {"loss": "nan", "noise_loss": "inf"})
            self.assertEqual(error.evidence_path, Path(directory) / "hpo-divergence-joint.json")
            self.assertEqual(json.loads(error.evidence_path.read_text(encoding="utf-8")),
                             error.evidence)
            self.assertFalse(list(Path(directory).glob("*.tmp")))
            guard.partial_history.clear()
            self.assertEqual(len(error.evidence["partial_history"]), 1)

    def test_new_fit_resets_state_and_infers_v2_phase(self) -> None:
        guard = self._guard()
        guard.model._train_part = "generator"
        guard.on_train_begin()
        guard.on_epoch_end(3, {"loss": 0.5})
        self.assertEqual(guard.phase, "generator")
        guard.model._train_part = "discriminator"
        guard.on_train_begin()
        self.assertEqual(guard.phase, "discriminator")
        self.assertIsNone(guard.epoch)
        self.assertEqual(guard.partial_history, [])
        self.assertEqual(guard.last_finite_metrics, {})
        with self.assertRaises(TrainingDiverged) as raised:
            guard.on_epoch_end(0, {"clf_loss": float("nan")})
        self.assertEqual(raised.exception.phase, "discriminator")
        self.assertEqual(raised.exception.epoch, 0)

    def test_explicit_phase_takes_precedence(self) -> None:
        guard = self._guard(phase="joint")
        guard.model._train_part = "generator"
        guard.on_train_begin()
        self.assertEqual(guard.phase, "joint")
        for phase in ("", " ", 1):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                NonFiniteLossGuard(phase=phase)

    def test_scalar_tensor_losses_and_nonloss_diagnostics(self) -> None:
        guard = self._guard()
        guard.on_epoch_end(0, {
            "loss": tf.constant(0.25, dtype=tf.bfloat16),
            "accuracy": np.nan,
            "loss_scale_factor": np.inf,
            "diagnostics": np.array([1.0, 2.0]),
            "metadata": "nan",
        })
        self.assertEqual(guard.last_finite_metrics, {"loss": 0.25})
        with self.assertRaises(TrainingDiverged):
            guard.on_epoch_end(1, {"loss": tf.constant(np.inf, dtype=tf.bfloat16)})

    def test_evidence_write_failure_does_not_hide_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            blocked = Path(directory) / "file"
            blocked.write_text("existing file", encoding="utf-8")
            guard = self._guard(evidence_dir=blocked)
            with self.assertRaises(TrainingDiverged) as raised:
                guard.on_epoch_end(0, {"loss": float("nan")})
            self.assertIsNone(raised.exception.evidence_path)
            self.assertIn("evidence_write_error", raised.exception.evidence)
            self.assertEqual(blocked.read_text(encoding="utf-8"), "existing file")


if __name__ == "__main__":
    unittest.main()
