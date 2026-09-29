"""Regression coverage for fitting ordinary compiled Keras classifier teachers."""

import json
import unittest
from unittest.mock import Mock, patch

import numpy as np
import tensorflow as tf

from common.config import DiffusionClassifierConfig, DiffusionClassifierV2Config
from common.learner import _run_continual_tasks
from common.model import get_model
from common.tests import test_clean_classifier_training as fixtures
from common.tests.test_pretrained_backbones import _tiny_base
from diffusion import DiffusionClassifier, DiffusionClassifierV2


class _ClassifierBlock(tf.keras.layers.Layer):
    """Own frozen children under an ordinary Layer rather than a nested Model."""

    def __init__(self) -> None:
        """Create a frozen projection and BN followed by a trainable projection."""

        super().__init__()
        self.early = tf.keras.layers.Conv2D(
            2, 1, use_bias=False, kernel_initializer="ones", trainable=False
        )
        self.normalization = tf.keras.layers.BatchNormalization(trainable=False)
        self.tail = tf.keras.layers.Conv2D(
            2, 1, use_bias=False, kernel_initializer="ones"
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Apply the children using their individual fine-tuning flags."""

        hidden = self.early(inputs)
        hidden = self.normalization(hidden, training=training)
        return self.tail(hidden)


class FitKerasTeacherTests(unittest.TestCase):
    """Preserve classifier fine-tuning state while isolating the student lifecycle."""

    setUp = fixtures.CleanClassifierTrainingTests.setUp
    tearDown = fixtures.CleanClassifierTrainingTests.tearDown
    make_network = fixtures.CleanClassifierTrainingTests.make_network

    def make_teacher(self) -> tf.keras.Model:
        """Build a compiled nested classifier with frozen convolution and BN layers."""

        inputs = tf.keras.Input(shape=(4, 4, 1))
        hidden = tf.keras.layers.Conv2D(
            2, 1, use_bias=False, kernel_initializer="ones", name="early"
        )(inputs)
        hidden = tf.keras.layers.BatchNormalization(name="normalization")(hidden)
        hidden = tf.keras.layers.Conv2D(
            2, 1, use_bias=False, kernel_initializer="ones", name="tail"
        )(hidden)
        base = tf.keras.Model(inputs, hidden, name="base")
        base.get_layer("early").trainable = False
        base.get_layer("normalization").trainable = False
        teacher = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(4, 4, 1)), 
            base, 
            tf.keras.layers.GlobalAveragePooling2D(), 
            tf.keras.layers.Dense(2, activation="softmax", name="head")
        ])
        teacher.compile(
            optimizer=tf.keras.optimizers.SGD(.01), 
            loss="sparse_categorical_crossentropy", metrics=["accuracy"], 
            run_eagerly=False, jit_compile=False
        )
        return teacher

    def make_wrapper(
        self, teacher: tf.keras.Model, wrapper_cls: type = DiffusionClassifier, 
        compile_model: bool = True, **overrides: object
    ) -> DiffusionClassifier:
        """Build an independently optimized student with classification distillation."""

        options = dict(
            network=self.make_network(), teacher_network=teacher, 
            trainable_teacher=True, use_ema=False, seed=811, 
            scheduler_name="clipped_cosine", test_steps=4, p_uncond=0., 
            mask_by_nulls=False, mask_by_t_threshold=False, 
            clf_loss_coef=0., noise_loss_coef=0., clf_distil_loss_coef=1., 
            clf_distil_type="soft", clf_distil_temperature=2.
        )
        options.update(overrides)
        model = wrapper_cls(**options)
        # Preserve an uncompiled wrapper for compile-precondition coverage.
        if compile_model:
            model.compile(
                optimizer=tf.keras.optimizers.SGD(.05), loss="mse", 
                run_eagerly=False, jit_compile=False
            )
        return model

    def dataset(self, images: object | None = None) -> tf.data.Dataset:
        """Create one deterministic batch using a bounded private thread pool."""

        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        return tf.data.Dataset.from_tensor_slices(
            (self.images if images is None else images, self.labels)
        ).batch(4).with_options(options)

    def assert_weights_equal(self, model: tf.keras.Model, before: list) -> None:
        """Check exact preservation of every recorded model weight."""

        self.assertEqual(len(model.get_weights()), len(before))
        for expected, actual in zip(before, model.get_weights()):
            np.testing.assert_array_equal(actual, expected)

    def assert_fit_mask(self, teacher: tf.keras.Model) -> None:
        """Check the nested parent is enabled without enabling frozen descendants."""

        self.assertTrue(teacher.trainable)
        base = teacher.get_layer("base")
        self.assertTrue(base.trainable)
        self.assertFalse(base.get_layer("early").trainable)
        self.assertFalse(base.get_layer("normalization").trainable)
        self.assertTrue(base.get_layer("tail").trainable)
        self.assertTrue(teacher.get_layer("head").trainable)

    def test_fit_preserves_compile_mask_and_student_then_distills(self) -> None:
        """Train teachers repeatedly with their own metrics before student KD fitting."""

        for wrapper_cls in (DiffusionClassifier, DiffusionClassifierV2):
            with self.subTest(wrapper=wrapper_cls.__name__):
                teacher = self.make_teacher()
                teacher_optimizer = teacher.optimizer
                teacher_compile = teacher.get_compile_config()
                base = teacher.get_layer("base")
                early_before = base.get_layer("early").get_weights()
                bn_before = base.get_layer("normalization").get_weights()
                tail_before = base.get_layer("tail").get_weights()
                head_before = teacher.get_layer("head").get_weights()
                model = self.make_wrapper(teacher, wrapper_cls)
                student_before = model.network.get_weights()
                self.assertEqual(teacher.get_compile_config(), teacher_compile)
                self.assertIs(teacher.optimizer, teacher_optimizer)
                self.assertIsNot(teacher_optimizer, model.optimizer)
                self.assertFalse(teacher.trainable)
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_accuracy", patience=10, min_delta=.01
                    ), 
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_accuracy", patience=4, min_delta=.01, factor=.8
                    )
                ]
                for expected_iterations in (1, 2):
                    history = model.fit_teacher(
                        self.dataset(), validation_data=self.dataset(), epochs=1, 
                        callbacks=callbacks, verbose=0
                    )
                    self.assertIn("accuracy", history.history)
                    self.assertIn("val_accuracy", history.history)
                    self.assertNotIn("noise_loss", history.history)
                    self.assertTrue(all(np.isfinite(values).all()
                                        for values in history.history.values()))
                    self.assertEqual(int(teacher_optimizer.iterations), expected_iterations)
                    self.assertIs(teacher.optimizer, teacher_optimizer)
                    self.assertFalse(teacher.trainable)
                    self.assertEqual(int(model.optimizer.iterations), 0)
                    self.assert_weights_equal(model.network, student_before)
                    self.assert_weights_equal(base.get_layer("early"), early_before)
                    self.assert_weights_equal(base.get_layer("normalization"), bn_before)
                    self.assertFalse({id(value) for value in teacher.weights}
                                     & {id(value) for value in model.weights})
                    # Student recompilation must retain the teacher's completed optimizer steps.
                    if expected_iterations == 1:
                        model.compile(
                            optimizer=tf.keras.optimizers.SGD(.05), loss="mse", 
                            run_eagerly=False, jit_compile=False
                        )
                        self.assertIs(teacher.optimizer, teacher_optimizer)
                        self.assertEqual(int(teacher_optimizer.iterations), 1)
                        self.assertEqual(teacher.get_compile_config(), teacher_compile)
                self.assertTrue(any(not np.array_equal(before, after) for before, after in
                                    zip(tail_before, base.get_layer("tail").get_weights())))
                self.assertTrue(any(not np.array_equal(before, after) for before, after in
                                    zip(head_before, teacher.get_layer("head").get_weights())))
                fitted_teacher = teacher.get_weights()
                student_fit = model.fit_discriminator if wrapper_cls is DiffusionClassifierV2 \
                    else model.fit
                history = student_fit(x=self.dataset(), epochs=1, verbose=0)
                self.assertGreater(history.history["clf_distil_loss"][0], 0.)
                self.assertTrue(any(not np.array_equal(before, after) for before, after in
                                    zip(student_before, model.network.get_weights())))
                self.assert_weights_equal(teacher, fitted_teacher)
                self.assertEqual(int(teacher_optimizer.iterations), 2)

    def test_arrays_pass_unchanged_and_failure_restores_inference_state(self) -> None:
        """Forward raw arrays and callbacks and refreeze after a delegated failure."""

        teacher = self.make_teacher()
        model = self.make_wrapper(teacher, teacher_classifier_input_range="pixels")
        pixels = ((self.images + 1.) * 127.5).numpy()
        labels = self.labels.numpy()
        validation = (pixels.copy(), labels.copy())
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy")]
        sentinel = tf.keras.callbacks.History()

        def fit(**kwargs: object) -> tf.keras.callbacks.History:
            """Inspect the data and fine-tuning mask visible to ordinary Keras fit."""

            self.assertIs(kwargs["x"], pixels)
            self.assertIs(kwargs["y"], labels)
            self.assertIs(kwargs["validation_data"], validation)
            self.assertIs(kwargs["callbacks"], callbacks)
            self.assertEqual(kwargs["batch_size"], 2)
            self.assert_fit_mask(teacher)
            return sentinel

        with patch.object(teacher, "fit", side_effect=fit) as delegated:
            result = model.fit_teacher(
                pixels, labels, validation_data=validation, callbacks=callbacks, 
                batch_size=2, epochs=1, verbose=0
            )
        self.assertIs(result, sentinel)
        delegated.assert_called_once()
        self.assertFalse(teacher.trainable)
        model.train_function = object()
        model.test_function = object()
        model.predict_function = object()

        def fail(**kwargs: object) -> None:
            """Raise only after fit_teacher has restored the classifier training mask."""

            self.assert_fit_mask(teacher)
            raise RuntimeError("deliberate teacher fit failure")

        with patch.object(teacher, "fit", side_effect=fail), \
             self.assertRaisesRegex(RuntimeError, "deliberate"):
            model.fit_teacher(pixels, labels, epochs=1, verbose=0)
        self.assertFalse(teacher.trainable)
        self.assertEqual(teacher.trainable_weights, [])
        self.assertIsNone(model.train_function)
        self.assertIsNone(model.test_function)
        self.assertIsNone(model.predict_function)
        self.assertEqual(int(model.optimizer.iterations), 0)

    def test_reattach_and_replacement_keep_original_fine_tuning_masks(self) -> None:
        """Retain a reattached teacher's mask and capture an independent replacement."""

        original = self.make_teacher()
        model = self.make_wrapper(original)
        model.set_teacher_network(original)
        with patch.object(original, "fit", side_effect=lambda **kwargs: self.assert_fit_mask(original)):
            model.fit_teacher(self.dataset(), epochs=1, verbose=0)
        replacement = self.make_teacher()
        replacement.get_layer("base").trainable = False
        model.set_teacher_network(replacement)

        def inspect_replacement(**kwargs: object) -> None:
            """Keep the replacement trunk frozen while fitting its classification head."""

            self.assertTrue(replacement.trainable)
            self.assertFalse(replacement.get_layer("base").trainable)
            self.assertEqual(replacement.get_layer("base").trainable_weights, [])
            self.assertTrue(replacement.get_layer("head").trainable)

        with patch.object(replacement, "fit", side_effect=inspect_replacement):
            model.fit_teacher(self.dataset(), epochs=1, verbose=0)
        self.assertFalse(original.trainable)
        self.assertFalse(replacement.trainable)
        self.assertIs(model.teacher_network, replacement)
        self.assertFalse({id(value) for value in replacement.weights}
                         & {id(value) for value in model.weights})

    def test_classifier_input_range_routes_clean_targets_and_roundtrips(self) -> None:
        """Apply optional pixel conversion only when obtaining callable class targets."""

        for wrapper_cls, config_cls in (
            (DiffusionClassifier, DiffusionClassifierConfig), 
            (DiffusionClassifierV2, DiffusionClassifierV2Config)
        ):
            for input_range in ("diffusion", "pixels"):
                with self.subTest(wrapper=wrapper_cls.__name__, input_range=input_range):
                    teacher = self.make_teacher()
                    model = self.make_wrapper(
                        teacher, wrapper_cls, teacher_classifier_input_range=input_range
                    )
                    # V2 class targets belong to the discriminator phase.
                    if wrapper_cls is DiffusionClassifierV2:
                        model._switch_train_part("discriminator")
                        model._test_part = "discriminator"
                    expected_images = (self.images + 1.) * 127.5 \
                        if input_range == "pixels" else self.images
                    expected = teacher(expected_images, training=False)
                    original_call = teacher.call

                    def inspect(
                        inputs: tf.Tensor, mask: object | None = None, 
                        training: bool = False
                    ) -> tf.Tensor:
                        """Assert the teacher receives clean images in its configured range."""

                        tf.debugging.assert_equal(inputs, expected_images)
                        self.assertFalse(training)
                        return original_call(inputs, mask=mask, training=training)

                    with patch.object(teacher, "call", side_effect=inspect):
                        mapped = model.prep_inputs_map(self.images, self.labels)
                    np.testing.assert_allclose(mapped[-1], expected, atol=1e-6)
                    serialized = json.loads(json.dumps(model.get_config()))
                    self.assertEqual(serialized["teacher_classifier_input_range"], input_range)
                    self.assertNotIn("teacher_network", serialized)
                    clone = wrapper_cls.from_config({**serialized, "defer_teacher": True})
                    self.assertEqual(clone.teacher_classifier_input_range, input_range)
                    typed = config_cls(teacher_classifier_input_range=input_range)
                    self.assertEqual(typed.kwargs()["teacher_classifier_input_range"], input_range)

    def test_get_model_efficientnet_teacher_preserves_factory_fine_tuning(self) -> None:
        """Fit a factory EfficientNet classifier with a local application replacement."""

        with patch.object(
            tf.keras.applications, "EfficientNetV2L", 
            side_effect=lambda **options: _tiny_base("EfficientNetV2L", **options)
        ):
            teacher = get_model(
                2, model_type="pretrained", conv_base_name="EfficientNetV2L", 
                num_last_not_frozen=3, dropout_rate=.50, resize=(32, 32), verbose=0, 
                compile_args={"optimizer": tf.keras.optimizers.SGD(.01), 
                              "run_eagerly": False, "jit_compile": False}
            )
        model = self.make_wrapper(
            teacher, network=self.make_network(image_size=32, channels=3, patch_size=16), 
            teacher_classifier_input_range="pixels"
        )
        base = next(layer for layer in teacher.layers if isinstance(layer, tf.keras.Model))
        frozen = [base.get_layer(name) for name in ("early_conv", "early_bn", "tail_bn")]
        frozen_weights = [layer.get_weights() for layer in frozen]
        tail_before = base.get_layer("tail_conv").get_weights()
        pixels = tf.reshape(tf.linspace(150., 250., 4 * 32 * 32 * 3), (4, 32, 32, 3))
        history = model.fit_teacher(self.dataset(pixels), epochs=1, verbose=0)
        self.assertIn("accuracy", history.history)
        for layer, before in zip(frozen, frozen_weights):
            self.assert_weights_equal(layer, before)
        self.assertTrue(any(not np.array_equal(before, after) for before, after in
                            zip(tail_before, base.get_layer("tail_conv").get_weights())))
        self.assertEqual(int(teacher.optimizer.iterations), 1)
        self.assertFalse(teacher.trainable)

    def test_unsupported_modes_and_shared_optimizer_fail_before_fitting(self) -> None:
        """Reject native-only fit methods and sharing the teacher's optimizer."""

        teacher = self.make_teacher()
        model = self.make_wrapper(teacher)
        with patch.object(teacher, "fit") as fit:
            for method in ("fit_progressively", "fit_generator", "fit_discriminator"):
                with self.subTest(method=method), self.assertRaisesRegex(ValueError, "fit"):
                    model.fit_teacher(self.dataset(), fit_method=method, epochs=1)
            fit.assert_not_called()
        self.assertFalse(teacher.trainable)
        with self.assertRaisesRegex(ValueError, "optimizer"):
            model.compile(optimizer=teacher.optimizer, loss="mse", jit_compile=False)

    def test_custom_layer_children_keep_frozen_weights_and_batchnorm_state(self) -> None:
        """Traverse descendants of custom Layers when restoring fine-tuning flags."""

        block = _ClassifierBlock()
        teacher = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(4, 4, 1)), 
            block, 
            tf.keras.layers.GlobalAveragePooling2D(), 
            tf.keras.layers.Dense(2, activation="softmax")
        ])
        teacher.compile(
            optimizer=tf.keras.optimizers.SGD(.01), 
            loss="sparse_categorical_crossentropy", metrics=["accuracy"], 
            run_eagerly=False, jit_compile=False
        )
        early_before = block.early.get_weights()
        normalization_before = block.normalization.get_weights()
        tail_before = block.tail.get_weights()
        model = self.make_wrapper(teacher)
        model.fit_teacher(self.dataset(), epochs=1, verbose=0)
        self.assert_weights_equal(block.early, early_before)
        self.assert_weights_equal(block.normalization, normalization_before)
        self.assertTrue(any(not np.array_equal(before, after) for before, after in
                            zip(tail_before, block.tail.get_weights())))
        self.assertFalse(teacher.trainable)
        self.assertEqual(teacher.trainable_weights, [])

    def test_v2_phase_optimizer_aliases_fail_before_teacher_fit(self) -> None:
        """Reject a teacher optimizer sharing any active or inactive student phase."""

        for phase, loss_scaled in (
            ("generator", False), ("discriminator", False), ("discriminator", True)
        ):
            with self.subTest(phase=phase, loss_scaled=loss_scaled):
                teacher = self.make_teacher()
                model = self.make_wrapper(teacher, DiffusionClassifierV2)
                optimizer = model.gen_optimizer if phase == "generator" else model.clf_optimizer
                model._switch_train_part("discriminator" if phase == "generator" else "generator")
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer) \
                    if loss_scaled else optimizer
                teacher.compile(
                    optimizer=optimizer, loss="sparse_categorical_crossentropy", 
                    metrics=["accuracy"], jit_compile=False
                )
                with patch.object(teacher, "fit") as fit, \
                     self.assertRaisesRegex(ValueError, "optimizer"):
                    model.fit_teacher(self.dataset(), epochs=1, verbose=0)
                fit.assert_not_called()
                self.assertEqual(int(model.gen_optimizer.iterations), 0)
                self.assertEqual(int(model.clf_optimizer.iterations), 0)
                self.assertFalse(teacher.trainable)

    def test_continual_lifecycle_rejects_classifier_before_data_loading(self) -> None:
        """Keep automatic teacher growth and recovery on the native diffusion path."""

        teacher = self.make_teacher()
        model = self.make_wrapper(teacher)
        loader = Mock(name="dataset_loader")
        with patch.object(model, "fit_teacher") as teacher_fit, \
             self.assertRaisesRegex(ValueError, "continual.*native diffusion teacher"):
            _run_continual_tasks(
                class_num=4, task_size=2, load_dataset_fn=loader, 
                load_dataset_fn_kwargs={"preprocess": "diffusion"}, 
                generative_model=model, use_generative_model_classifier=True, 
                use_distillation=True, use_generative_replay=False, 
                epochs=1, callback_patience=0, plot_results=False, verbose=0, 
                show_generated_images=False, show_network_summary=False
            )
        loader.assert_not_called()
        teacher_fit.assert_not_called()
        self.assertEqual(int(teacher.optimizer.iterations), 0)
        self.assertEqual(int(model.optimizer.iterations), 0)
        self.assertFalse(teacher.trainable)

    def test_uncompiled_and_unbuilt_classifiers_fail_before_attachment(self) -> None:
        """Report missing teacher compile or build state at the owning boundary."""

        uncompiled = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(4, 4, 1)), 
            tf.keras.layers.GlobalAveragePooling2D(), 
            tf.keras.layers.Dense(2, activation="softmax")
        ])
        with self.assertRaisesRegex(ValueError, "compil"):
            self.make_wrapper(uncompiled)
        unbuilt = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(), 
            tf.keras.layers.Dense(2, activation="softmax")
        ])
        unbuilt.compile(optimizer="sgd", loss="sparse_categorical_crossentropy")
        with self.assertRaisesRegex(ValueError, "built|build"):
            self.make_wrapper(unbuilt)


# Permit focused regression execution without running tests on import.
if __name__ == "__main__":
    unittest.main()
