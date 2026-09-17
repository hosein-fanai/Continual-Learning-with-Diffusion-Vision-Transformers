"""Pruned-trial persistence and actual Keras callback dispatch without training."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import optuna
import pandas as pd
import tensorflow as tf

from common.callbacks.hpo_guard import NonFiniteLossGuard, TrainingDiverged
from common.config import Config
from common.hpo import run_hpo, summarize_hpo
from common.train import _fit_with_callback_cleanup, train_model


class _ScriptedLossModel(tf.keras.Model):
    """Return fixed losses through real Keras fit hooks; never update weights."""

    def __init__(self, failure_phase: str):
        super().__init__()
        self.failure_phase = failure_phase
        self.train_calls = 0
        self.validation_calls = 0
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs):
        return inputs

    def train_step(self, data):
        del data
        self.train_calls += 1
        # Two batches per epoch: corrupt the first batch of the second epoch.
        bad = self.failure_phase == "train" and self.train_calls == 3
        self.loss_tracker.update_state(tf.constant(float("nan") if bad else 0.5))
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        del data
        self.validation_calls += 1
        bad = self.failure_phase == "validation" and self.validation_calls == 3
        self.loss_tracker.update_state(tf.constant(float("inf") if bad else 0.25))
        return {"loss": self.loss_tracker.result()}


class JointHpoGuardIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    @staticmethod
    def _scalars(directory: Path) -> dict[str, list[float]]:
        scalars: dict[str, list[float]] = {}
        for path in directory.rglob("events.out.tfevents.*"):
            for event in tf.compat.v1.train.summary_iterator(str(path)):
                for value in event.summary.value:
                    if value.metadata.plugin_data.plugin_name == "scalars":
                        scalars.setdefault(value.tag, []).append(
                            float(tf.make_ndarray(value.tensor).item())
                        )
        return scalars

    def test_pruned_trial_preserves_path_evidence_and_search_continues(self):
        """A real Path evidence file is JSON-safe in Optuna and has no objectives."""
        def fake_main(config, **kwargs):
            del kwargs
            output = Path(config.training.results_path) / f"trial-{config.hpo['trial_number']:04d}"
            output.mkdir(parents=True, exist_ok=True)
            config.training.results_path = str(output)
            if config.hpo["trial_number"] == 0:
                guard = NonFiniteLossGuard(phase="generator", evidence_dir=output)
                guard.set_model(SimpleNamespace())
                guard.on_train_begin()
                guard.on_epoch_end(0, {"loss": 0.8, "val_noise_loss": 0.9})
                guard.on_epoch_end(1, {"loss": float("nan")})
            return {
                "results_path": str(output),
                "history": {},
                "evaluations": {
                    "valset_ema_eval": {"ensemble_accuracy": 0.6, "noise_loss": 0.4},
                    "valset_network_eval": {"ensemble_accuracy": 0.99, "noise_loss": 0.001},
                },
            }

        options = {
            "task": "joint", "model_name": "dit_classifier", "dataset_name": "cifar10",
            "search_profile": "joint_dit_classifier", "trial_budget_mode": "total",
            "n_trials": 2, "epochs": 2, "seed": 17, "n_startup_trials": 1,
            "results_path": str(self.root),
            "search_space_overrides": {"wrapper_name": ["diffusion_classifier"]},
        }
        with patch("common.hpo.main", side_effect=fake_main) as training:
            study = run_hpo(**options)
            self.assertEqual(training.call_count, 2)
            pruned, completed = study.trials
            self.assertEqual(pruned.state, optuna.trial.TrialState.PRUNED)
            self.assertIsNone(pruned.values)
            self.assertEqual(completed.values, [0.6, 0.4])
            self.assertEqual(completed.state, optuna.trial.TrialState.COMPLETE)
            self.assertIsInstance(pruned.user_attrs["divergence_path"], str)
            evidence_path = Path(pruned.user_attrs["divergence_path"])
            self.assertEqual(json.loads(evidence_path.read_text(encoding="utf-8")),
                             pruned.user_attrs["divergence"])
            self.assertEqual(pruned.user_attrs["divergence"]["phase"], "generator")
            self.assertEqual(pruned.user_attrs["divergence"]["hook"], "epoch_end")
            self.assertIsNone(pruned.user_attrs["divergence"]["batch"])
            self.assertEqual(len(pruned.user_attrs["divergence"]["partial_history"]), 1)
            self.assertTrue(Path(pruned.user_attrs["config_path"]).is_file())
            self.assertFalse((Path(pruned.user_attrs["results_path"]) / "objectives.csv").exists())

            study_root = Path(pruned.user_attrs["config_path"]).parent.parent
            scalars = self._scalars(study_root / "tensorboard" / "trial-0000" / "outcome")
            self.assertEqual(scalars["hpo/pruned"], [1.0])
            self.assertEqual(scalars["hpo/failed"], [0.0])
            self.assertEqual(scalars["hpo/completed"], [0.0])
            self.assertNotIn("hpo/classification_accuracy", scalars)
            self.assertNotIn("hpo/noise_loss", scalars)
            self.assertEqual(pd.read_csv(study_root / "trials.csv")["state"].tolist(),
                             ["PRUNED", "COMPLETE"])
            self.assertEqual(summarize_hpo(study)["trial"].tolist(), [1])
            all_trials = summarize_hpo(study, pareto_only=False)
            self.assertEqual(all_trials["state"].tolist(), ["PRUNED", "COMPLETE"])
            self.assertTrue(pd.isna(all_trials.loc[0, "classification_accuracy"]))
            self.assertTrue(pd.isna(all_trials.loc[0, "noise_loss"]))
            self.assertEqual(pd.read_csv(study_root / "pareto_trials.csv")["trial"].tolist(), [1])
            repeated = run_hpo(**options, resume_from=study_root)
            self.assertEqual(len(repeated.trials), 2)
            self.assertEqual(training.call_count, 2)

    def test_train_model_guard_catches_training_and_validation_and_keeps_prior_logs(self):
        """Bad batches finish the full train/validation epoch before termination."""
        tensorboard_callbacks = []

        class TensorBoardSpy(tf.keras.callbacks.TensorBoard):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.close_calls = 0
                tensorboard_callbacks.append(self)

            def on_train_end(self, logs=None):
                self.close_calls += 1
                return super().on_train_end(logs)

        for failure_phase in ("train", "validation"):
            with self.subTest(failure_phase=failure_phase):
                tensorboard_callbacks.clear()
                output = self.root / failure_phase
                output.mkdir()
                config = Config(
                    training={
                        "task": "joint", "epochs": 3, "verbose": 0,
                        "results_path": str(output), "save_weights": False,
                        "tensorboard": True, "tensorboard_path": str(output / "tensorboard"),
                        "project_tag": "tiny", "patience": 0, "reduce_lr_patience": 0,
                        "report_every_epoch": False, "show_images": False, "save_gifs": False,
                    },
                    hpo={"prune_nonfinite_losses": True},
                )
                model = _ScriptedLossModel(failure_phase)
                model.compile(optimizer="sgd", run_eagerly=True)
                rows = [[1.0], [2.0], [3.0], [4.0]]
                dataset = tf.data.Dataset.from_tensor_slices((rows, rows)).batch(2)
                options = tf.data.Options()
                options.threading.private_threadpool_size = 1
                options.threading.max_intra_op_parallelism = 1
                dataset = dataset.with_options(options)
                with patch("common.train.ImageGenerator") as image, patch(
                    "common.train.callbacks.TensorBoard", TensorBoardSpy
                ):
                    image.return_value.results_path = str(output)
                    with self.assertRaises(TrainingDiverged) as raised:
                        train_model(config, model=model, trainset=dataset, valset=dataset)
                evidence = raised.exception.evidence
                self.assertEqual(evidence["hook"], "epoch_end")
                self.assertEqual(evidence["epoch"], 1)
                self.assertIsNone(evidence["batch"])
                self.assertEqual(evidence["metric"], "loss" if failure_phase == "train" else "val_loss")
                self.assertEqual(len(evidence["partial_history"]), 1)
                self.assertEqual(evidence["partial_history"][0]["metrics"]["val_loss"], 0.25)
                self.assertTrue(raised.exception.evidence_path.is_file())
                self.assertTrue((output / "input_config.yaml").is_file())
                self.assertTrue((output / "config.yaml").is_file())
                # Read before callback/model destruction: persisted logs cannot rely on GC.
                scalars = self._scalars(output / "tensorboard")
                self.assertIn(0.5, scalars.get("epoch_loss", []))
                self.assertIn(0.25, scalars.get("epoch_loss", []))
                self.assertEqual(model.train_calls, 4)
                self.assertEqual(model.validation_calls, 4)
                self.assertEqual(len(tensorboard_callbacks), 1)
                self.assertEqual(tensorboard_callbacks[0].close_calls, 1)

    def test_cleanup_deduplicates_v2_callbacks_and_preserves_original_error(self):
        """A failed close cannot mask divergence/OOM or skip another writer."""
        divergence = TrainingDiverged({
            "phase": "discriminator", "epoch": 1, "batch": None,
            "metric": "clf_loss", "value": "nan", "hook": "epoch_end",
        })
        for error in (divergence, tf.errors.ResourceExhaustedError(None, None, "synthetic OOM")):
            with self.subTest(error=type(error).__name__):
                first = tf.keras.callbacks.TensorBoard(log_dir=str(self.root / "generator"))
                second = tf.keras.callbacks.TensorBoard(log_dir=str(self.root / "discriminator"))
                first.on_train_end = Mock(side_effect=RuntimeError("synthetic close failure"))
                second.on_train_end = Mock()
                stopper = tf.keras.callbacks.EarlyStopping()
                stopper.on_train_end = Mock()
                model = SimpleNamespace(fit=Mock(side_effect=error))
                with self.assertRaises(type(error)) as raised:
                    _fit_with_callback_cleanup(
                        model, callbacks=[first, stopper],
                        gen_kwargs={"callbacks": [first, stopper]},
                        clf_kwargs={"callbacks": [first, second, stopper]},
                    )
                self.assertIs(raised.exception, error)
                first.on_train_end.assert_called_once_with()
                second.on_train_end.assert_called_once_with()
                stopper.on_train_end.assert_not_called()
                self.assertTrue(any("synthetic close failure" in note for note in error.__notes__))

    def test_successful_fit_keeps_its_return_and_callback_lifecycle(self):
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=str(self.root / "success"))
        tensorboard.on_train_end = Mock()
        expected = object()
        model = SimpleNamespace(fit=Mock(return_value=expected))
        self.assertIs(_fit_with_callback_cleanup(model, callbacks=[tensorboard], epochs=2), expected)
        model.fit.assert_called_once_with(callbacks=[tensorboard], epochs=2)
        tensorboard.on_train_end.assert_not_called()


if __name__ == "__main__":
    unittest.main()
