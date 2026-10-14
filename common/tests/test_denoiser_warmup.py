"""Verify denoiser warmup objectives, traced epoch transitions and fit cleanup."""

import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.tests import test_clean_classifier_training as fixtures
from diffusion import DiffusionClassifier


_COEFFICIENT_NAMES = (
    "noise_loss_coef", "clf_loss_coef", "ctr_loss_coef", 
    "clf_distil_loss_coef", "noise_distil_loss_coef", "image_loss_coef"
)


class _CoefficientObserver(tf.keras.callbacks.Callback):
    """Record public epoch coefficients and optionally interrupt callback execution."""

    def __init__(self, stop_epoch: int | None = None, failure_event: str | None = None) -> None:
        """Configure a bounded early stop or a deliberate callback exception."""

        super().__init__()
        self.stop_epoch = stop_epoch
        self.failure_event = failure_event
        self.epochs = []
        self.train_end = None

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        """Capture the effective coefficients before any training batch runs."""

        self.epochs.append((epoch, {
            name: float(getattr(self.model, name)) for name in _COEFFICIENT_NAMES
        }))
        # Callback failures must leave the wrapper ready for another ordinary fit.
        if self.failure_event == "epoch_begin":
            raise RuntimeError("deliberate warmup callback failure")

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Stop before the warmup boundary when the test requests early stopping."""

        # Leave normal fitting alone unless this epoch is the requested stopping point.
        if epoch == self.stop_epoch:
            self.model.stop_training = True

    def on_train_end(self, logs: dict | None = None) -> None:
        """Observe restoration before user end callbacks, including one that raises."""

        self.train_end = {
            name: float(getattr(self.model, name)) for name in _COEFFICIENT_NAMES
        }
        # Restoration must happen even if a user callback cannot finish normally.
        if self.failure_event == "train_end":
            raise RuntimeError("deliberate warmup callback failure")


class DenoiserWarmupTests(unittest.TestCase):
    """Exercise actual graph losses and isolate callback lifecycle edge cases."""

    setUp = fixtures.CleanClassifierTrainingTests.setUp
    tearDown = fixtures.CleanClassifierTrainingTests.tearDown
    make_network = fixtures.CleanClassifierTrainingTests.make_network

    def make_wrapper(self, **overrides: object) -> DiffusionClassifier:
        """Use the existing four-pixel fixture with distinctive loss coefficients."""

        options = dict(
            clf_train_noisy_input_type="noisy", 
            clf_train_class_input_type="all_classes", clf_train_type="cond", 
            p_uncond=0., noise_loss_coef=.37, clf_loss_coef=.41, 
            ctr_loss_coef=.29, image_loss_coef=.13, map_num_parallel_calls=1
        )
        options.update(overrides)
        return fixtures.CleanClassifierTrainingTests.make_wrapper(self, **options)

    def dataset(self) -> tf.data.Dataset:
        """Keep each epoch to one real batch and one private input thread."""

        dataset = tf.data.Dataset.from_tensor_slices((self.images, self.labels)).batch(4)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        return dataset.with_options(options)

    def coefficient_state(self, wrapper: DiffusionClassifier) -> dict[str, object]:
        """Retain exact entry objects so cleanup checks include identity restoration."""

        return {name: getattr(wrapper, name) for name in _COEFFICIENT_NAMES}

    def assert_restored(self, wrapper: DiffusionClassifier, original: dict[str, object]) -> None:
        """Require the original tensor objects after both successful and failed fits."""

        for name, value in original.items():
            self.assertIs(getattr(wrapper, name), value, name)

    def assert_epoch_coefficients(
        self, observed: dict[str, float], original: dict[str, object], warmup: bool
    ) -> None:
        """Verify only the requested coefficients change during warmup epochs."""

        expected = {name: float(value) for name, value in original.items()}
        # Warmup leaves noise distillation and reconstruction at their entry values.
        if warmup:
            expected.update(
                noise_loss_coef=1., clf_loss_coef=0., ctr_loss_coef=0., 
                clf_distil_loss_coef=0.
            )
        self.assertEqual(observed, expected)

    def emulate_fit(self, wrapper: DiffusionClassifier, **options: object) -> object:
        """Drive Keras callback events while exercising the actual wrapper fit lifecycle."""

        def keras_fit(
            model: tf.keras.Model, x: object = None, y: object = None, 
            callbacks: object = None, epochs: int = 1, initial_epoch: int = 0, 
            **kwargs: object
        ) -> tf.keras.callbacks.History:
            """Emulate only epoch/end dispatch; graph objectives are tested separately."""

            callback_list = tf.keras.callbacks.CallbackList(
                callbacks, model=model, epochs=epochs, steps=1
            )
            history = tf.keras.callbacks.History()
            history.epoch = []
            history.history = {}
            model.stop_training = False
            callback_list.on_train_begin()
            for epoch in range(initial_epoch, epochs):
                callback_list.on_epoch_begin(epoch)
                callback_list.on_epoch_end(epoch, {})
                history.epoch.append(epoch)
                # Honor user early stopping without manufacturing extra epoch events.
                if model.stop_training:
                    break
            callback_list.on_train_end()
            return history

        with patch.object(tf.keras.Model, "fit", autospec=True, side_effect=keras_fit):
            return wrapper.fit(**options)

    def assert_history_objective(
        self, history: dict[str, list[float]], epoch: int, warmup: bool, prefix: str = ""
    ) -> None:
        """Compare actual traced totals with independently weighted component metrics."""

        noise = history[prefix + "noise_loss"][epoch]
        image = history[prefix + "image_loss"][epoch]
        noise_distil = history[prefix + "noise_distil_loss"][epoch]
        expected = (1. if warmup else .37) * noise + .13 * image + .23 * noise_distil
        # All classifier terms rejoin the objective after the absolute epoch boundary.
        if not warmup:
            expected += .41 * history[prefix + "classifier_loss"][epoch]
            expected += .29 * history[prefix + "clf_ctr_loss"][epoch]
            expected += .19 * history[prefix + "clf_distil_loss"][epoch]
        np.testing.assert_allclose(history[prefix + "loss"][epoch], expected, rtol=2e-5, atol=2e-5)

    def test_graph_objectives_cross_boundary_with_validation_and_noise_distillation(self) -> None:
        """Mapped and online graph fits restore classification without suppressing noise KD."""

        for mapped in (False, True):
            with self.subTest(map_preprocess=mapped):
                teacher = self.make_network()
                teacher((self.images, tf.zeros_like(self.labels), self.labels), training=False)
                # Distinct teacher weights make unchanged noise KD numerically observable.
                for variable in teacher.trainable_variables:
                    variable.assign(tf.ones_like(variable) * .05)
                network = self.make_network(
                    clf_cls_token_regularizer_ids=[1], 
                    clf_cls_token_regularizer_kwargs={
                        "train_type": "both", "distil_type": "soft", "start": 0, "end": 1
                    }
                )
                wrapper = self.make_wrapper(
                    network=network, teacher_network=teacher, 
                    clf_distil_loss_coef=.19, noise_distil_loss_coef=.23, 
                    map_preprocess=mapped
                )
                wrapper.optimizer.learning_rate.assign(.001)
                original = self.coefficient_state(wrapper)
                self.assertTrue(wrapper.use_clf_distil_ctr_loss)
                observer = _CoefficientObserver()
                callbacks = [observer]
                history = wrapper.fit(
                    self.dataset(), denoiser_warmup_epochs=1, epochs=3, 
                    callbacks=callbacks, validation_data=self.dataset(), verbose=0
                )
                self.assertEqual(callbacks, [observer])
                self.assertEqual(history.epoch, [0, 1, 2])
                self.assertGreater(history.history["noise_distil_loss"][0], 0.)
                for index, (epoch, state) in enumerate(observer.epochs):
                    self.assert_epoch_coefficients(state, original, epoch < 1)
                    for prefix in ("", "val_"):
                        self.assert_history_objective(history.history, index, epoch < 1, prefix)
                self.assert_restored(wrapper, original)
                self.assert_epoch_coefficients(observer.train_end, original, False)

                # End with a warmup trace still active, then verify cleanup on the next fit.
                warmup_only = wrapper.fit(
                    self.dataset(), denoiser_warmup_epochs=5, epochs=1, 
                    validation_data=self.dataset(), verbose=0
                )
                for prefix in ("", "val_"):
                    self.assert_history_objective(warmup_only.history, 0, True, prefix)
                self.assert_restored(wrapper, original)

                # An ordinary follow-up fit must never reuse a graph with warmup constants.
                ordinary = wrapper.fit(
                    self.dataset(), epochs=1, validation_data=self.dataset(), verbose=0
                )
                for prefix in ("", "val_"):
                    self.assert_history_objective(ordinary.history, 0, False, prefix)
                self.assert_restored(wrapper, original)

    def test_default_and_explicit_zero_leave_coefficients_and_callbacks_unchanged(self) -> None:
        """The default path and a zero warmup preserve the existing fit contract."""

        wrapper = self.make_wrapper()
        original = self.coefficient_state(wrapper)
        for options in ({}, {"denoiser_warmup_epochs": 0}):
            with self.subTest(options=options):
                observer = _CoefficientObserver()
                callbacks = [observer]
                history = self.emulate_fit(wrapper, epochs=2, callbacks=callbacks, **options)
                self.assertEqual(history.epoch, [0, 1])
                self.assertEqual(callbacks, [observer])
                for _, state in observer.epochs:
                    self.assert_epoch_coefficients(state, original, False)
                self.assert_restored(wrapper, original)

    def test_initial_epoch_uses_absolute_warmup_boundary(self) -> None:
        """Resuming inside warmup consumes its remainder; resuming after it skips warmup."""

        wrapper = self.make_wrapper()
        original = self.coefficient_state(wrapper)
        for initial_epoch in (1, 2, 4):
            with self.subTest(initial_epoch=initial_epoch):
                observer = _CoefficientObserver()
                history = self.emulate_fit(
                    wrapper, denoiser_warmup_epochs=2, initial_epoch=initial_epoch, 
                    epochs=initial_epoch + 2, callbacks=[observer]
                )
                self.assertEqual(history.epoch, [initial_epoch, initial_epoch + 1])
                for epoch, state in observer.epochs:
                    self.assert_epoch_coefficients(state, original, epoch < 2)
                self.assert_restored(wrapper, original)

    def test_early_stop_and_short_fit_restore_before_user_train_end(self) -> None:
        """A fit ending entirely inside warmup restores coefficients before user cleanup."""

        wrapper = self.make_wrapper()
        original = self.coefficient_state(wrapper)
        for stop_epoch in (None, 0):
            with self.subTest(stop_epoch=stop_epoch):
                observer = _CoefficientObserver(stop_epoch=stop_epoch)
                history = self.emulate_fit(
                    wrapper, denoiser_warmup_epochs=5, epochs=2, callbacks=[observer]
                )
                self.assertEqual(len(history.epoch), 2 if stop_epoch is None else 1)
                for _, state in observer.epochs:
                    self.assert_epoch_coefficients(state, original, True)
                self.assert_epoch_coefficients(observer.train_end, original, False)
                self.assert_restored(wrapper, original)

    def test_callback_exceptions_restore_entry_coefficients(self) -> None:
        """Failures during warmup and user end hooks cannot strand temporary coefficients."""

        wrapper = self.make_wrapper()
        original = self.coefficient_state(wrapper)
        for event in ("epoch_begin", "train_end"):
            with self.subTest(event=event):
                observer = _CoefficientObserver(failure_event=event)
                with self.assertRaisesRegex(RuntimeError, "deliberate warmup callback failure"):
                    self.emulate_fit(
                        wrapper, denoiser_warmup_epochs=5, epochs=1, callbacks=[observer]
                    )
                self.assert_restored(wrapper, original)
                # End-callback failures still observe the already restored public state.
                if event == "train_end":
                    self.assert_epoch_coefficients(observer.train_end, original, False)


# Keep this module available as a focused remote unittest entry point.
if __name__ == "__main__":
    unittest.main()
