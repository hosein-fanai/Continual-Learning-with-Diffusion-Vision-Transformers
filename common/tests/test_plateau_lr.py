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
from common.callbacks.hpo_guard import NonFiniteLossGuard
from common.callbacks.plateau_lr import (
    OffsetCosineDecay, PlateauLearningRate, ValidationEnsembleAccuracy
)
from common.config import Config
from common.model import _make_optimizer
from common.train import _fit_control_callbacks, _resolve_training_options, train_model
from diffusion.models.wrapper.diffusion_classifier_v2 import DiffusionClassifierV2
from diffusion.models.transformer.di_t_classifier import DiTClassifier


class PlateauLearningRateTests(unittest.TestCase):
    """Check cosine-clock math, checkpointing, mixed precision, and phase controls."""

    def test_optimizer_factory_delegates_duration_bounds_to_its_schedule(self) -> None:
        """Exercise native and offset-cosine constructor errors through the common factory."""

        for plateau_jump, message in (
            (False, "Argument `decay_steps` must be > 0"), 
            (True, "Cosine rate/duration must be positive")
        ):
            for duration in (0, -1):
                with self.subTest(plateau_jump=plateau_jump, duration=duration), self.assertRaisesRegex(
                    ValueError, message
                ):
                    _make_optimizer(name="adam", schedule="cosine", plateau_jump=plateau_jump, 
                                    decay_steps=duration, initial_learning_rate=0.01)

    def test_traced_optimizer_uses_jump_without_iteration_or_slot_reset(self) -> None:
        """Observe cosine jumps in an existing graph while preserving momentum and iterations."""

        schedule = OffsetCosineDecay(0.1, 100, min_learning_rate=0.001)
        optimizer = tf.keras.optimizers.SGD(schedule, momentum=0.9)
        weight = tf.Variable(1.0)

        @tf.function
        def step() -> tf.Tensor:
            """Apply one scalar SGD update through the traced optimizer and return its float32 weight."""

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

    def test_optimizer_serialization_clones_schedule_and_current_offset(self) -> None:
        """Clone independent schedule clocks at the same current learning rate."""

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

    def test_v2_compile_clones_independent_phase_clocks(self) -> None:
        """Give V2 generator and classifier optimizers independent cosine offsets."""

        network = DiTClassifier(
            num_classes=2, use_cfg=True, timesteps=4, image_size=4, channels=1, 
            patch_size=2, dim=4, depth=1, mha_num_heads=1, 
            vit_block_mlp_ratio=1.0, clf_mha_num_heads=1, 
            clf_vit_block_mlp_ratio=1.0, 
            feature_aggregation_ids_dict={1: tuple([-1])}, clf_connection_ids_dict={-1: tuple([-1])}
        )
        wrapper = DiffusionClassifierV2(
            network=network, use_ema=False, test_network_name="raw", 
            scheduler_name="linear", test_steps=2, seed=43
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

    def test_loss_scaled_optimizer_jumps_its_inner_cosine(self) -> None:
        """Plateau reduction changes the effective mixed-precision schedule only."""

        schedule = OffsetCosineDecay(0.1, 100)
        inner = tf.keras.optimizers.SGD(schedule)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(inner)
        optimizer.iterations.assign(7)
        callback = PlateauLearningRate("score", patience=1, mode="max")
        callback.set_model(SimpleNamespace(optimizer=optimizer))
        callback.on_train_begin()
        callback.on_epoch_end(0, {"score": 0.5})
        before = float(optimizer.learning_rate.numpy())
        callback.on_epoch_end(1, {"score": 0.4})
        self.assertAlmostEqual(float(optimizer.learning_rate.numpy()), before * 0.5)
        self.assertGreater(float(schedule.offset.numpy()), 0.0)
        self.assertEqual(int(optimizer.iterations.numpy()), 7)

    def test_loss_scaled_immutable_schedule_is_rejected_before_fit(self) -> None:
        """Validate the actual inner schedule before a mixed-precision epoch runs."""

        inner = tf.keras.optimizers.SGD(
            tf.keras.optimizers.schedules.CosineDecay(0.1, 100)
        )
        callback = PlateauLearningRate()
        callback.set_model(SimpleNamespace(
            optimizer=tf.keras.mixed_precision.LossScaleOptimizer(inner)
        ))
        with self.assertRaisesRegex(ValueError, "plateau_jump"):
            callback.on_train_begin()

    def test_v2_early_stopping_alone_uses_independent_phase_metrics(self) -> None:
        """Early-stopping-only V2 fits select generator and classifier monitors."""

        model = MagicMock(spec=DiffusionClassifierV2)
        model.fit.return_value = {"val_classifier_accuracy": [0.5]}
        with tempfile.TemporaryDirectory() as directory, patch("common.train.ImageGenerator") as image:
            image.return_value.results_path = directory
            train_model(model=model, trainset=object(), valset=object(), 
                        results_path=directory, show_images=False, save_gifs=False, 
                        report_every_epoch=False, save_weights=False, verbose=0, 
                        patience=2, reduce_lr_patience=0)
        phases = model.fit.call_args.kwargs
        generator = [c for c in phases["gen_kwargs"]["callbacks"]
                     if isinstance(c, tf.keras.callbacks.EarlyStopping)]
        classifier = [c for c in phases["clf_kwargs"]["callbacks"]
                      if isinstance(c, tf.keras.callbacks.EarlyStopping)]
        self.assertEqual(len(generator), 1)
        self.assertEqual(len(classifier), 1)
        self.assertIsNot(generator[0], classifier[0])
        self.assertEqual(generator[0].monitor, "val_noise_loss")
        self.assertEqual(generator[0].mode, "min")
        self.assertEqual(classifier[0].monitor, "val_classifier_accuracy")
        self.assertEqual(classifier[0].mode, "max")

    def test_v2_single_phase_controls_follow_the_selected_fit(self) -> None:
        """Route only the selected phase's stopping, plateau, divergence, and logging state."""

        cases = (
            ("fit_generator", "generator", False, False, "noise_loss", "min"), 
            ("fit_generator", "generator", True, True, "val_noise_loss", "min"), 
            ("fit_discriminator", "discriminator", False, False, "classifier_accuracy", "max"), 
            ("fit_discriminator", "discriminator", True, False, "val_classifier_accuracy", "max"), 
            ("fit_discriminator", "discriminator", True, True, "val_ensemble_accuracy", "max")
        )
        tensorboard_type = tf.keras.callbacks.TensorBoard
        for method, phase, validation, ensemble, monitor, mode in cases:
            for reduce_lr_patience in (0, 1):
                with self.subTest(method=method, validation=validation, ensemble=ensemble, 
                                  reduce_lr_patience=reduce_lr_patience):
                    model = MagicMock(spec=DiffusionClassifierV2)
                    fit = getattr(model, method)
                    fit.return_value = SimpleNamespace(history={monitor: [0.5]})
                    with tempfile.TemporaryDirectory() as directory, \
                         patch("common.train.ImageGenerator") as image, \
                         patch("common.train.callbacks.TensorBoard", wraps=tensorboard_type) as board_factory, \
                         patch("common.train._fit_control_callbacks", wraps=_fit_control_callbacks) as control_factory:
                        image.return_value.results_path = directory
                        train_model(
                            model=model, trainset=object(), valset=object() if validation else None, 
                            results_path=directory, show_images=False, save_gifs=False, 
                            report_every_epoch=False, save_weights=False, verbose=0, 
                            fit_method=method, patience=2, reduce_lr_patience=reduce_lr_patience, 
                            ensemble_monitor=ensemble, tensorboard=True, 
                            hpo={"prune_nonfinite_losses": True}
                        )
                        self.assertEqual(board_factory.call_count, 1)
                        self.assertEqual(control_factory.call_count, 1)
                        self.assertEqual(control_factory.call_args.args[2], phase)
                    fit.assert_called_once()
                    model.fit.assert_not_called()
                    selected = fit.call_args.kwargs["callbacks"]
                    stops = [item for item in selected if isinstance(item, tf.keras.callbacks.EarlyStopping)]
                    plateaus = [item for item in selected if isinstance(item, PlateauLearningRate)]
                    self.assertEqual([(item.monitor, item.mode) for item in stops], [(monitor, mode)])
                    self.assertEqual([(item.monitor, item.mode) for item in plateaus], 
                                     [(monitor, mode)] * reduce_lr_patience)
                    self.assertEqual([item.phase for item in selected if isinstance(item, NonFiniteLossGuard)], 
                                     [phase])
                    self.assertEqual([Path(item.log_dir).name for item in selected
                                      if isinstance(item, tensorboard_type)], [phase])
                    self.assertEqual(sum(isinstance(item, ValidationEnsembleAccuracy) for item in selected), 
                                     int(ensemble and phase == "discriminator"))

    def test_checkpoint_restores_offset_and_optimizer_iteration(self) -> None:
        """Recover both cosine offset and optimizer iteration from a TensorFlow checkpoint."""

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

    def test_zero_patience_reduces_first_nonimprovement_without_clipping_tolerance(self) -> None:
        """Reduce a real scalar optimizer at zero patience while preserving the signed tolerance."""

        optimizer = tf.keras.optimizers.SGD(0.1)
        optimizer.iterations.assign(7)
        callback = PlateauLearningRate("score", patience=0, mode="max", min_delta=-0.05)
        self.assertEqual(callback.patience, 0)
        self.assertEqual(callback.min_delta, -0.05)
        callback.set_model(SimpleNamespace(optimizer=optimizer))
        callback.on_train_begin()
        callback.on_epoch_end(0, {"score": 0.5})
        self.assertAlmostEqual(float(optimizer.learning_rate.numpy()), 0.1)
        logs = {"score": 0.4}
        callback.on_epoch_end(1, logs)
        self.assertAlmostEqual(float(optimizer.learning_rate.numpy()), 0.05)
        self.assertAlmostEqual(logs["learning_rate"], 0.05)
        self.assertEqual(callback.wait, 0)
        self.assertEqual(callback.best, 0.5)
        self.assertEqual(int(optimizer.iterations.numpy()), 7)

    def test_patience_improvements_fit_reset_and_scalar_floor(self) -> None:
        """Reset patience on improvement or a new fit and never reduce below the scalar floor."""

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

    def test_unsupported_immutable_schedule_rejected_before_training(self) -> None:
        """Reject a schedule that cannot apply plateau changes before starting epochs."""

        callback = PlateauLearningRate()
        optimizer = tf.keras.optimizers.SGD(tf.keras.optimizers.schedules.CosineDecay(0.1, 100))
        callback.set_model(SimpleNamespace(optimizer=optimizer))
        with self.assertRaisesRegex(ValueError, "plateau_jump"):
            callback.on_train_begin()

    def test_default_optimizer_and_callbacks_remain_disabled(self) -> None:
        """Preserve the ordinary cosine optimizer and absence of opt-in epoch controls."""

        config = Config()
        config.optimizer.decay_steps = 10
        optimizer = _make_optimizer(config)
        self.assertIsInstance(optimizer._learning_rate, tf.keras.optimizers.schedules.CosineDecay)
        options = _resolve_training_options(config, None, {})
        self.assertEqual(_fit_control_callbacks(options, object()), [])

    def test_phase_controls_are_independent_and_ensemble_runs_first(self) -> None:
        """Evaluate ensemble validation before independent stopping and plateau phase controls."""

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

    def test_early_stop_restores_best_weights_at_ten_bad_epochs(self) -> None:
        """Stop after ten non-improving epochs and restore the best classifier weights."""

        config = Config()
        config.training.patience = 10
        config.training.reduce_lr_patience = 5
        config.training.monitor = "val_classifier_accuracy"
        options = _resolve_training_options(config, None, {})
        callback = _fit_control_callbacks(options, object(), "discriminator")[0]
        model = tf.keras.Sequential([tf.keras.layers.Input(tuple([1])), tf.keras.layers.Dense(1, use_bias=False)])
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

    def test_v2_training_dispatch_separates_callbacks_and_tensorboard_paths(self) -> None:
        """Route independent V2 control callbacks and TensorBoard paths in dependency order."""

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


# Run these regression cases when the module is executed directly.
if __name__ == "__main__":
    unittest.main()
