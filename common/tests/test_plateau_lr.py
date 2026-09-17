"""Numerical and orchestration checks for opt-in epoch plateau controls."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import tensorflow as tf

from common.callbacks.lr_logger import LrLogger
from common.callbacks.plateau_lr import (
    OffsetCosineDecay, PlateauLearningRate, ValidationEnsembleAccuracy,
)
from common.config import Config
from common.model import _make_optimizer
from common.train import _fit_control_callbacks, _resolve_training_options, train_model
from diffusion.models.wrapper.diffusion_classifier_v2 import DiffusionClassifierV2
from diffusion.models.transformer.di_t_classifier import DiTClassifier


class PlateauLearningRateTests(unittest.TestCase):
    def test_traced_optimizer_uses_jump_without_iteration_or_slot_reset(self):
        schedule = OffsetCosineDecay(0.1, 100, min_learning_rate=0.001)
        optimizer = tf.keras.optimizers.SGD(schedule, momentum=0.9)
        weight = tf.Variable(1.0)

        @tf.function
        def step():
            optimizer.apply_gradients([(tf.constant(1.0), weight)])
            return weight.read_value()

        step()
        original_iterations = int(optimizer.iterations.numpy())
        original_slots = [value.numpy().copy() for value in optimizer.variables]
        previous_rate = float(optimizer.learning_rate.numpy())
        schedule.jump(optimizer.iterations, 0.5, 0.001)
        self.assertEqual(int(optimizer.iterations.numpy()), original_iterations)
        for expected, actual in zip(original_slots, optimizer.variables):
            np.testing.assert_array_equal(expected, actual.numpy())
        self.assertAlmostEqual(float(optimizer.learning_rate.numpy()), previous_rate * 0.5, places=6)
        before_weight = float(weight.numpy())
        # Previous SGD momentum is -0.1; the next update includes the new half rate.
        step()
        self.assertAlmostEqual(float(weight.numpy()), before_weight - 0.09 - previous_rate * 0.5, places=6)
        offset = float(schedule.offset.numpy())
        schedule.jump(optimizer.iterations, 0.5, 0.001)
        self.assertGreater(float(schedule.offset.numpy()), offset)
        rates = [float(schedule(index).numpy()) for index in (2, 20, 100, 1000)]
        self.assertTrue(all(left >= right for left, right in zip(rates, rates[1:])))
        self.assertAlmostEqual(rates[-1], 0.001, places=7)

    def test_optimizer_serialization_clones_schedule_and_current_offset(self):
        optimizer = _make_optimizer(name="adam", schedule="cosine", plateau_jump=True,
                                    decay_steps=100, initial_learning_rate=0.01)
        optimizer._learning_rate.jump(0, 0.5)
        clone = tf.keras.optimizers.deserialize(tf.keras.optimizers.serialize(optimizer))
        self.assertIsNot(clone._learning_rate, optimizer._learning_rate)
        self.assertIsNot(clone._learning_rate.offset, optimizer._learning_rate.offset)
        self.assertAlmostEqual(float(clone.learning_rate.numpy()), float(optimizer.learning_rate.numpy()))
        clone._learning_rate.jump(0, 0.5)
        self.assertAlmostEqual(float(optimizer.learning_rate.numpy()), 0.005, places=7)
        self.assertAlmostEqual(float(clone.learning_rate.numpy()), 0.0025, places=7)

    def test_v2_compile_clones_independent_phase_clocks(self):
        network = DiTClassifier(
            num_classes=2, use_cfg=True, timesteps=4, image_size=4, channels=1,
            patch_size=2, dim=4, depth=1, mha_num_heads=1,
            vit_block_mlp_ratio=1.0, clf_mha_num_heads=1,
            clf_vit_block_mlp_ratio=1.0,
            feature_aggregation_ids_dict={1: (-1,)}, clf_connection_ids_dict={-1: (-1,)},
        )
        wrapper = DiffusionClassifierV2(
            network=network, use_ema=False, test_network_name="raw",
            scheduler_name="linear", test_steps=2, seed=43,
        )
        optimizer = _make_optimizer(name="adam", schedule="cosine", plateau_jump=True,
                                    decay_steps=100, initial_learning_rate=0.01)
        wrapper.compile(optimizer=optimizer, loss="mse", run_eagerly=True)
        generator = wrapper.gen_optimizer._learning_rate
        classifier = wrapper.clf_optimizer._learning_rate
        self.assertIsNot(generator, classifier)
        self.assertIsNot(generator.offset, classifier.offset)
        generator.jump(wrapper.gen_optimizer.iterations, 0.5)
        self.assertAlmostEqual(float(wrapper.gen_optimizer.learning_rate.numpy()), 0.005, places=7)
        self.assertAlmostEqual(float(wrapper.clf_optimizer.learning_rate.numpy()), 0.01, places=7)

    def test_checkpoint_restores_offset_and_optimizer_iteration(self):
        optimizer = _make_optimizer(name="sgd", schedule="cosine", plateau_jump=True,
                                    decay_steps=100, initial_learning_rate=0.01)
        optimizer.iterations.assign(7)
        optimizer._learning_rate.jump(optimizer.iterations, 0.5)
        expected = float(optimizer.learning_rate.numpy())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = tf.train.Checkpoint(optimizer=optimizer)
            path = checkpoint.save(str(Path(directory) / "state"))
            optimizer._learning_rate.offset.assign(0)
            optimizer.iterations.assign(0)
            checkpoint.restore(path).assert_consumed()
        self.assertEqual(int(optimizer.iterations.numpy()), 7)
        self.assertAlmostEqual(float(optimizer.learning_rate.numpy()), expected, places=7)

    def test_patience_improvements_fit_reset_and_scalar_floor(self):
        optimizer = tf.keras.optimizers.SGD(0.1)
        callback = PlateauLearningRate("score", patience=5, mode="max", min_learning_rate=0.04)
        callback.set_model(SimpleNamespace(optimizer=optimizer))
        callback.on_train_begin()
        callback.on_epoch_end(0, {"score": 0.5})
        for epoch in range(1, 5):
            callback.on_epoch_end(epoch, {"score": 0.5})
        self.assertAlmostEqual(float(optimizer.learning_rate.numpy()), 0.1)
        callback.on_epoch_end(5, {"score": 0.5})
        self.assertAlmostEqual(float(optimizer.learning_rate.numpy()), 0.05)
        callback.on_epoch_end(6, {"score": 0.6})
        self.assertEqual(callback.wait, 0)
        for epoch in range(7, 12):
            callback.on_epoch_end(epoch, {"score": 0.6})
        self.assertAlmostEqual(float(optimizer.learning_rate.numpy()), 0.04)
        callback.on_train_begin()
        self.assertIsNone(callback.best)
        self.assertEqual(callback.wait, 0)
        self.assertEqual(int(optimizer.iterations.numpy()), 0)

    def test_unsupported_immutable_schedule_rejected_before_training(self):
        callback = PlateauLearningRate()
        optimizer = tf.keras.optimizers.SGD(tf.keras.optimizers.schedules.CosineDecay(0.1, 100))
        callback.set_model(SimpleNamespace(optimizer=optimizer))
        with self.assertRaisesRegex(ValueError, "plateau_jump"):
            callback.on_train_begin()

    def test_default_optimizer_and_callbacks_remain_disabled(self):
        config = Config()
        config.optimizer.decay_steps = 10
        optimizer = _make_optimizer(config)
        self.assertIsInstance(optimizer._learning_rate, tf.keras.optimizers.schedules.CosineDecay)
        options = _resolve_training_options(config, None, {})
        self.assertEqual(_fit_control_callbacks(options, object()), [])

    def test_phase_controls_are_independent_and_ensemble_runs_first(self):
        config = Config()
        config.training.patience = 10
        config.training.reduce_lr_patience = 5
        config.training.monitor = "val_ensemble_accuracy"
        config.training.monitor_mode = "max"
        config.training.ensemble_monitor = True
        config.reporting.ensemble_accuracy_kwargs = {"max_t": 8, "seed": 21}
        options = _resolve_training_options(config, None, {})
        validation = object()
        generator = _fit_control_callbacks(options, validation, "generator")
        classifier = _fit_control_callbacks(options, validation, "discriminator")
        self.assertIsInstance(classifier[0], ValidationEnsembleAccuracy)
        self.assertEqual([item.monitor for item in generator], ["val_noise_loss"] * 2)
        self.assertEqual([item.mode for item in generator], ["min"] * 2)
        self.assertEqual([item.monitor for item in classifier[1:]], ["val_ensemble_accuracy"] * 2)
        self.assertIsNot(generator[0], classifier[1])
        model = SimpleNamespace(evaluate_ensemble_accuracy=MagicMock(return_value=0.75))
        classifier[0].set_model(model)
        logs = {}
        classifier[0].on_epoch_end(0, logs)
        self.assertEqual(logs["val_ensemble_accuracy"], 0.75)
        model.evaluate_ensemble_accuracy.assert_called_once_with(validation, max_t=8, seed=21, verbose=False)

    def test_early_stop_restores_best_weights_at_ten_bad_epochs(self):
        config = Config()
        config.training.patience = 10
        config.training.reduce_lr_patience = 5
        config.training.monitor = "val_classifier_accuracy"
        options = _resolve_training_options(config, None, {})
        callback = _fit_control_callbacks(options, object(), "discriminator")[0]
        model = tf.keras.Sequential([tf.keras.layers.Input((1,)), tf.keras.layers.Dense(1, use_bias=False)])
        model.set_weights([np.array([[3.0]], np.float32)])
        model.stop_training = False
        callback.set_model(model)
        callback.on_train_begin()
        callback.on_epoch_end(0, {"val_classifier_accuracy": 0.8})
        model.set_weights([np.array([[7.0]], np.float32)])
        for epoch in range(1, 10):
            callback.on_epoch_end(epoch, {"val_classifier_accuracy": 0.7})
        self.assertFalse(model.stop_training)
        callback.on_epoch_end(10, {"val_classifier_accuracy": 0.7})
        self.assertTrue(model.stop_training)
        callback.on_train_end()
        self.assertEqual(float(model.get_weights()[0][0, 0]), 3.0)

    def test_v2_training_dispatch_separates_callbacks_and_tensorboard_paths(self):
        model = MagicMock(spec=DiffusionClassifierV2)
        model.fit.return_value = {"val_classifier_accuracy": [0.5]}
        with tempfile.TemporaryDirectory() as directory, patch("common.train.ImageGenerator") as image:
            image.return_value.results_path = directory
            train_model(model=model, trainset=object(), valset=object(),
                        results_path=directory, show_images=False, save_gifs=False,
                        report_every_epoch=False, save_weights=False, verbose=0,
                        patience=10, reduce_lr_patience=5, monitor="val_classifier_accuracy",
                        monitor_mode="max", tensorboard=True)
            generator = model.fit.call_args.kwargs["gen_kwargs"]["callbacks"]
            classifier = model.fit.call_args.kwargs["clf_kwargs"]["callbacks"]
            for group, monitor, phase in ((generator, "val_noise_loss", "generator"),
                                          (classifier, "val_classifier_accuracy", "discriminator")):
                stop = next(item for item in group if isinstance(item, tf.keras.callbacks.EarlyStopping))
                plateau = next(item for item in group if isinstance(item, PlateauLearningRate))
                logger = next(item for item in group if isinstance(item, LrLogger))
                tensorboard = next(item for item in group if isinstance(item, tf.keras.callbacks.TensorBoard))
                self.assertEqual(stop.monitor, monitor)
                self.assertEqual(plateau.monitor, monitor)
                self.assertLess(group.index(plateau), group.index(logger))
                self.assertLess(group.index(logger), group.index(tensorboard))
                self.assertEqual(Path(tensorboard.log_dir).name, phase)
            self.assertTrue(all(a is not b for a in generator for b in classifier))


if __name__ == "__main__":
    unittest.main()
