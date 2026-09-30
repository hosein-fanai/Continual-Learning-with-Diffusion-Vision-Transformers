"""Behavioral coverage for growing ordinary Keras classifier teachers."""

import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.model import get_model
from common.tests import test_fit_keras_teacher as fixtures
from common.tests.test_pretrained_backbones import _tiny_base
from diffusion import DiffusionClassifier, DiffusionClassifierV2


class _BeginProbe(tf.keras.callbacks.Callback):
    """Inspect the live teacher before any gradient update."""

    def __init__(self, inspect: object) -> None:
        """Retain the assertion callback without capturing a stale teacher."""

        super().__init__()
        self.inspect = inspect

    def on_train_begin(self, logs: object | None = None) -> None:
        """Expose the model Keras attached to this fitting invocation."""

        self.inspect(self.model)


class DynamicKerasTeacherTests(unittest.TestCase):
    """Grow classifier outputs without losing fine-tuning or label identity."""

    setUp = fixtures.FitKerasTeacherTests.setUp
    tearDown = fixtures.FitKerasTeacherTests.tearDown
    make_network = fixtures.FitKerasTeacherTests.make_network
    make_teacher = fixtures.FitKerasTeacherTests.make_teacher
    make_wrapper = fixtures.FitKerasTeacherTests.make_wrapper
    assert_weights_equal = fixtures.FitKerasTeacherTests.assert_weights_equal
    assert_fit_mask = fixtures.FitKerasTeacherTests.assert_fit_mask

    def dataset(
        self, labels: object, images: object | None = None, 
        weights: object | None = None
    ) -> tf.data.Dataset:
        """Create one finite deterministic batch, optionally carrying sample weights."""

        values = (self.images if images is None else images, labels)
        # Preserve the ordinary Keras weighted-dataset structure.
        if weights is not None:
            values += tuple([weights])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        return tf.data.Dataset.from_tensor_slices(values).batch(4).with_options(options)

    def assert_prefixes(
        self, before: list[np.ndarray], after: list[np.ndarray], 
        zero_extensions: bool = False
    ) -> None:
        """Require every old tensor value to survive shape-preserving or growing copies."""

        self.assertEqual(len(after), len(before))
        for expected, actual in zip(before, after):
            self.assertEqual(actual.ndim, expected.ndim)
            self.assertTrue(all(new >= old for old, new in zip(expected.shape, actual.shape)))
            prefix = tuple(slice(0, width) for width in expected.shape)
            np.testing.assert_array_equal(actual[prefix], expected)
            # New optimizer slot regions must not inherit unrelated trained columns.
            if zero_extensions and actual.shape != expected.shape:
                extension = actual.copy()
                extension[prefix] = 0.
                np.testing.assert_array_equal(extension, np.zeros_like(extension))

    def test_arrays_and_datasets_map_sparse_targets_validation_and_weights(self) -> None:
        """Remap real labels while retaining pixels, weights and caller callbacks."""

        labels = np.array([9, 7, 9, 7], dtype=np.int32)
        validation_labels = labels[::-1].copy()
        weights = np.array([1., .5, 2., 1.], dtype=np.float32)
        pixels = self.images.numpy()
        for as_dataset, wrapper_cls in (
            (False, DiffusionClassifier), (True, DiffusionClassifierV2)
        ):
            with self.subTest(dataset=as_dataset, wrapper=wrapper_cls.__name__):
                teacher = self.make_teacher()
                model = self.make_wrapper(
                    teacher, wrapper_cls, teacher_dynamic_classes=True
                )
                callback_list = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy")]
                history = tf.keras.callbacks.History()

                def inspect(**kwargs: object) -> tf.keras.callbacks.History:
                    """Check both Keras input forms at the delegated fit boundary."""

                    self.assert_fit_mask(teacher)
                    self.assertIs(kwargs["callbacks"], callback_list)
                    # Datasets keep aligned sample weights in their third component.
                    if as_dataset:
                        batch = next(iter(kwargs["x"]))
                        validation_batch = next(iter(kwargs["validation_data"]))
                        np.testing.assert_array_equal(batch[0], pixels)
                        np.testing.assert_array_equal(batch[1], [1, 0, 1, 0])
                        np.testing.assert_array_equal(batch[2], weights)
                        np.testing.assert_array_equal(validation_batch[1], [0, 1, 0, 1])
                        np.testing.assert_array_equal(validation_batch[2], weights)
                    # Array inputs retain their original image and sample-weight objects.
                    else:
                        self.assertIs(kwargs["x"], pixels)
                        np.testing.assert_array_equal(kwargs["y"], [1, 0, 1, 0])
                        self.assertIs(kwargs["sample_weight"], weights)
                        self.assertIs(kwargs["validation_data"][0], pixels)
                        np.testing.assert_array_equal(
                            kwargs["validation_data"][1], [0, 1, 0, 1]
                        )
                        self.assertIs(kwargs["validation_data"][2], weights)
                    return history

                x = self.dataset(labels, pixels, weights) if as_dataset else pixels
                y = None if as_dataset else labels
                validation = self.dataset(validation_labels, pixels, weights) \
                    if as_dataset else (pixels, validation_labels, weights)
                options = {} if as_dataset else {"sample_weight": weights}
                with patch.object(teacher, "fit", side_effect=inspect):
                    actual = model.fit_teacher(
                        x, y, validation_data=validation, callbacks=callback_list, 
                        epochs=1, verbose=0, **options
                    )
                self.assertIs(actual, history)
                self.assertEqual(dict(teacher._diffusion_seen_classes), {7: 0, 9: 1})
                self.assertTrue(teacher._diffusion_dynamic_classes)
                self.assertEqual(model.seen_classes, {})
                self.assertFalse(teacher.trainable)

    def test_growth_preserves_columns_adam_state_and_repeated_fine_tuning(self) -> None:
        """Append new labels while retaining old weights, slots and the student state."""

        for wrapper_cls in (DiffusionClassifier, DiffusionClassifierV2):
            with self.subTest(wrapper=wrapper_cls.__name__):
                teacher = self.make_teacher(compile_teacher=False)
                model = self.make_wrapper(
                    teacher, wrapper_cls, teacher_dynamic_classes=True
                )
                optimizer = tf.keras.optimizers.Adam(.001)
                model.compile_teacher(
                    optimizer=optimizer, loss="sparse_categorical_crossentropy", 
                    metrics=["accuracy"], run_eagerly=False, jit_compile=False
                )
                student_weights = model.network.get_weights()
                student_optimizer = model.optimizer
                first = self.dataset(tf.constant([9, 7, 9, 7]))
                model.fit_teacher(first, epochs=1, verbose=0)
                self.assertIs(model.teacher_network, teacher)
                self.assertIs(teacher.optimizer, optimizer)
                self.assertEqual(int(optimizer.iterations), 1)
                old_weights = teacher.get_weights()
                old_slots = [variable.numpy().copy() for variable in optimizer.variables]
                frozen = [teacher.get_layer("base").get_layer(name).get_weights()
                          for name in ("early", "normalization")]
                callbacks_seen = []

                def inspect_grown(grown: tf.keras.Model) -> None:
                    """Verify growth before fitting can change any migrated parameter."""

                    callbacks_seen.append(grown)
                    self.assertEqual(grown.output_shape[-1], 3)
                    self.assertEqual(dict(grown._diffusion_seen_classes), {7: 0, 9: 1, 12: 2})
                    self.assert_fit_mask(grown)
                    self.assert_prefixes(old_weights, grown.get_weights())
                    self.assert_prefixes(
                        old_slots, [variable.numpy() for variable in grown.optimizer.variables], 
                        zero_extensions=True
                    )
                    self.assertEqual(int(grown.optimizer.iterations), 1)

                second = self.dataset(tf.constant([12, 9, 12, 9]))
                history = model.fit_teacher(
                    second, validation_data=second, epochs=1, 
                    callbacks=[_BeginProbe(inspect_grown)], verbose=0
                )
                grown = model.teacher_network
                self.assertEqual(callbacks_seen, [grown])
                self.assertIn("val_accuracy", history.history)
                self.assertTrue(all(np.isfinite(values).all()
                                    for values in history.history.values()))
                self.assertEqual(int(grown.optimizer.iterations), 2)
                grown_optimizer = grown.optimizer
                model.fit_teacher(second, epochs=1, verbose=0)
                self.assertIs(model.teacher_network, grown)
                self.assertIs(grown.optimizer, grown_optimizer)
                self.assertEqual(int(grown_optimizer.iterations), 3)
                self.assertEqual(grown.output_shape[-1], 3)
                self.assertFalse(grown.trainable)
                self.assertEqual(grown.trainable_weights, [])
                for name, before in zip(("early", "normalization"), frozen):
                    self.assert_weights_equal(grown.get_layer("base").get_layer(name), before)
                self.assert_weights_equal(model.network, student_weights)
                self.assertIs(model.optimizer, student_optimizer)
                self.assertEqual(int(student_optimizer.iterations), 0)
                self.assertEqual(model.seen_classes, {})

                def fail(live_teacher: tf.keras.Model) -> None:
                    """Fail only after the preserved fine-tuning mask has been restored."""

                    self.assert_fit_mask(live_teacher)
                    raise RuntimeError("deliberate dynamic teacher failure")

                with self.assertRaisesRegex(RuntimeError, "deliberate"):
                    model.fit_teacher(
                        second, epochs=1, callbacks=[_BeginProbe(fail)], verbose=0
                    )
                self.assertFalse(model.teacher_network.trainable)
                self.assertEqual(int(grown_optimizer.iterations), 3)

    def test_explicit_initial_column_order_is_preserved_and_validated(self) -> None:
        """Respect a pretrained head's class order even when a task covers one class."""

        teacher = self.make_teacher()
        model = self.make_wrapper(teacher, teacher_dynamic_classes=True)
        labels = np.full(4, 7, dtype=np.int32)

        def inspect(**kwargs: object) -> tf.keras.callbacks.History:
            """Require dataset class 7 to retain pretrained output column one."""

            np.testing.assert_array_equal(kwargs["y"], np.ones(4, dtype=np.int32))
            return tf.keras.callbacks.History()

        with patch.object(teacher, "fit", side_effect=inspect):
            model.fit_teacher(
                self.images, labels, teacher_class_ids=[9, 7], epochs=1, verbose=0
            )
        self.assertEqual(dict(teacher._diffusion_seen_classes), {9: 0, 7: 1})
        self.assertEqual(teacher.output_shape[-1], 2)
        for class_ids in ([7], [7, 7]):
            with self.subTest(class_ids=class_ids):
                invalid_teacher = self.make_teacher()
                invalid_model = self.make_wrapper(
                    invalid_teacher, teacher_dynamic_classes=True
                )
                with patch.object(invalid_teacher, "fit") as fit, \
                     self.assertRaisesRegex(ValueError, "class|column|unique|width"):
                    invalid_model.fit_teacher(
                        self.images, labels, teacher_class_ids=class_ids, epochs=1, verbose=0
                    )
                fit.assert_not_called()
                self.assertFalse(invalid_teacher.trainable)
                self.assertEqual(int(invalid_teacher.optimizer.iterations), 0)

    def test_validation_cannot_silently_add_an_untrained_class(self) -> None:
        """Reject validation-only labels before training updates or accidental growth."""

        teacher = self.make_teacher()
        model = self.make_wrapper(teacher, teacher_dynamic_classes=True)
        labels = tf.constant([7, 9, 7, 9])
        with patch.object(teacher, "fit", return_value=tf.keras.callbacks.History()):
            model.fit_teacher(self.images, labels, epochs=1, verbose=0)
        for validation in (
            (self.images, tf.constant([7, 12, 7, 12])), 
            self.dataset(tf.constant([7, 12, 7, 12]))
        ):
            with self.subTest(dataset=isinstance(validation, tf.data.Dataset)):
                with patch.object(teacher, "fit") as fit, \
                     self.assertRaises((ValueError, tf.errors.InvalidArgumentError)):
                    model.fit_teacher(
                        self.dataset(labels), validation_data=validation, epochs=1, verbose=0
                    )
                fit.assert_not_called()
                self.assertEqual(dict(teacher._diffusion_seen_classes), {7: 0, 9: 1})
                self.assertEqual(teacher.output_shape[-1], 2)
                self.assertEqual(int(teacher.optimizer.iterations), 0)
                self.assertFalse(teacher.trainable)

    def test_distillation_aligns_teacher_columns_to_reordered_student_classes(self) -> None:
        """Compute hard and soft KD using dataset identity instead of column position."""

        teacher = self.make_teacher()
        model = self.make_wrapper(
            teacher, teacher_dynamic_classes=True, seen_classes={9: 0, 7: 1}
        )
        with patch.object(teacher, "fit", return_value=tf.keras.callbacks.History()):
            model.fit_teacher(
                self.images, tf.constant([7, 9, 7, 9]), epochs=1, verbose=0
            )
        student = tf.constant([[.1, .9]] * 4)
        targets = tf.constant([[.8, .2]] * 4)
        for kind in ("hard", "soft"):
            with self.subTest(kind=kind):
                loss, returned = model.compute_clf_distil_loss(
                    targets, student, classes=self.labels, clf_distil_type=kind, 
                    clf_distil_temperature=1., student_logits=tf.math.log(student)
                )
                expected = -np.log(.9) if kind == "hard" else \
                    np.sum(np.array([.2, .8]) * np.log(np.array([.2, .8]) / [.1, .9]))
                np.testing.assert_allclose(loss, expected, rtol=1e-5, atol=1e-7)
                np.testing.assert_array_equal(returned, student)
        self.assertEqual(model.seen_classes, {9: 0, 7: 1})
        self.assertEqual(dict(teacher._diffusion_seen_classes), {7: 0, 9: 1})

    def test_explicit_loss_weight_keeps_reordered_hard_and_soft_alignment(self) -> None:
        """Apply a public role-weight override after resolving teacher class identity."""

        teacher = self.make_teacher()
        model = self.make_wrapper(
            teacher, teacher_dynamic_classes=True, seen_classes={9: 0, 7: 1}
        )
        with patch.object(teacher, "fit", return_value=tf.keras.callbacks.History()):
            model.fit_teacher(
                self.images, tf.constant([7, 9, 7, 9]), epochs=1, verbose=0
            )
        student = tf.constant([[.1, .9]] * 4)
        targets = tf.constant([[.8, .2]] * 4)
        for kind in ("hard", "soft"):
            with self.subTest(kind=kind):
                loss, returned = model.compute_clf_distil_loss(
                    targets, student, classes=self.labels, clf_distil_type=kind, 
                    clf_distil_temperature=1., student_logits=tf.math.log(student), 
                    teacher_loss_weight=.5
                )
                unweighted = -np.log(.9) if kind == "hard" else \
                    np.sum(np.array([.2, .8]) * np.log(np.array([.2, .8]) / [.1, .9]))
                np.testing.assert_allclose(loss, .5 * unweighted, rtol=1e-5, atol=1e-7)
                np.testing.assert_array_equal(returned, student)

    def test_initial_growth_compile_failure_keeps_source_optimizer_usable(self) -> None:
        """Discard failed replacement graphs without binding the source optimizer to them."""

        teacher = self.make_teacher(compile_teacher=False)
        model = self.make_wrapper(teacher, teacher_dynamic_classes=True)
        optimizer = tf.keras.optimizers.Adam(.001)
        model.compile_teacher(
            optimizer=optimizer, loss="sparse_categorical_crossentropy", 
            metrics=["accuracy"], run_eagerly=False, jit_compile=False
        )
        original_weights = teacher.get_weights()
        self.assertFalse(optimizer.built)
        candidates = []

        def fail(candidate: tf.keras.Model, **kwargs: object) -> None:
            """Fail replacement compilation after observing its independent optimizer."""

            candidates.append(candidate)
            self.assertIsNot(candidate, teacher)
            self.assertIsNot(kwargs["optimizer"], optimizer)
            self.assert_fit_mask(candidate)
            raise RuntimeError("deliberate expanded compile failure")

        with patch.object(tf.keras.Sequential, "compile", autospec=True, side_effect=fail), \
             self.assertRaisesRegex(RuntimeError, "deliberate"):
            model.fit_teacher(
                self.dataset(tf.constant([7, 9, 12, 7])), epochs=1, verbose=0
            )
        self.assertEqual(len(candidates), 1)
        self.assertFalse(candidates[0].trainable)
        self.assertIs(model.teacher_network, teacher)
        self.assertIs(teacher.optimizer, optimizer)
        self.assertFalse(optimizer.built)
        self.assertEqual(int(optimizer.iterations), 0)
        self.assertFalse(teacher.trainable)
        self.assertEqual(dict(teacher._diffusion_seen_classes), {})
        self.assert_weights_equal(teacher, original_weights)
        history = model.fit_teacher(
            self.dataset(tf.constant([7, 9, 7, 9])), epochs=1, verbose=0
        )
        self.assertIn("accuracy", history.history)
        self.assertIs(model.teacher_network, teacher)
        self.assertIs(teacher.optimizer, optimizer)
        self.assertEqual(int(optimizer.iterations), 1)
        self.assertFalse(teacher.trainable)

    def test_get_model_efficientnet_teacher_grows_without_recreating_backbone(self) -> None:
        """Exercise the notebook factory with local application weights and real fitting."""

        with patch.object(
            tf.keras.applications, "EfficientNetV2L", 
            side_effect=lambda **options: _tiny_base("EfficientNetV2L", **options)
        ) as constructor:
            teacher = get_model(
                2, model_type="pretrained", conv_base_name="EfficientNetV2L", 
                num_last_not_frozen=3, dropout_rate=.50, resize=(32, 32), compile_args={"optimizer": tf.keras.optimizers.Adam(.001), 
                              "run_eagerly": False, "jit_compile": False}, 
                verbose=0
            )
            model = self.make_wrapper(
                teacher, teacher_dynamic_classes=True, preprocess_type="standardize", 
                network=self.make_network(image_size=32, channels=3, patch_size=16)
            )
            base = next(layer for layer in teacher.layers if isinstance(layer, tf.keras.Model))
            frozen = {name: base.get_layer(name).get_weights()
                      for name in ("early_conv", "early_bn", "tail_bn")}
            pixels = tf.reshape(
                tf.linspace(150., 250., 4 * 32 * 32 * 3), (4, 32, 32, 3)
            )
            model.fit_teacher(
                self.dataset(tf.constant([7, 9, 7, 9]), pixels), epochs=1, verbose=0
            )
            old_weights = model.teacher_network.get_weights()

            def inspect_grown(grown: tf.keras.Model) -> None:
                """Keep every pretrained value and the factory's partial trainability."""

                self.assert_prefixes(old_weights, grown.get_weights())
                live_base = next(layer for layer in grown.layers
                                 if isinstance(layer, tf.keras.Model))
                self.assertTrue(live_base.get_layer("tail_conv").trainable)
                for name in frozen:
                    self.assertFalse(live_base.get_layer(name).trainable)

            history = model.fit_teacher(
                self.dataset(tf.constant([12, 9, 12, 9]), pixels), epochs=1, 
                callbacks=[_BeginProbe(inspect_grown)], verbose=0
            )
            constructor.assert_called_once()
        grown = model.teacher_network
        self.assertEqual(grown.output_shape[-1], 3)
        self.assertEqual(int(grown.optimizer.iterations), 2)
        self.assertIn("accuracy", history.history)
        self.assertFalse(grown.trainable)
        grown_base = next(layer for layer in grown.layers if isinstance(layer, tf.keras.Model))
        for name, before in frozen.items():
            self.assert_weights_equal(grown_base.get_layer(name), before)


# Permit focused execution while keeping imports side-effect free.
if __name__ == "__main__":
    unittest.main()
