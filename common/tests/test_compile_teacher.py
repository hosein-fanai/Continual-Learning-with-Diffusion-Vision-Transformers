"""Behavioral coverage for independent public teacher compilation."""

import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.tests import test_fit_keras_teacher as keras_fixtures
from common.tests import test_fit_teacher as native_fixtures
from diffusion import DiffusionClassifier, DiffusionClassifierV2


class CompileKerasTeacherTests(unittest.TestCase):
    """Compile image classifiers without changing student or fine-tuning state."""

    setUp = keras_fixtures.FitKerasTeacherTests.setUp
    tearDown = keras_fixtures.FitKerasTeacherTests.tearDown
    make_network = keras_fixtures.FitKerasTeacherTests.make_network
    make_teacher = keras_fixtures.FitKerasTeacherTests.make_teacher
    make_wrapper = keras_fixtures.FitKerasTeacherTests.make_wrapper
    dataset = keras_fixtures.FitKerasTeacherTests.dataset
    assert_weights_equal = keras_fixtures.FitKerasTeacherTests.assert_weights_equal
    assert_fit_mask = keras_fixtures.FitKerasTeacherTests.assert_fit_mask

    def test_uncompiled_teacher_compiles_before_student_and_fits_with_own_loss(self) -> None:
        """Accept built classifiers and preserve frozen layers through real fitting."""

        for wrapper_cls in (DiffusionClassifier, DiffusionClassifierV2):
            with self.subTest(wrapper=wrapper_cls.__name__):
                teacher = self.make_teacher(compile_teacher=False)
                model = self.make_wrapper(teacher, wrapper_cls, compile_model=False)
                teacher_optimizer = tf.keras.optimizers.SGD(.01)
                base = teacher.get_layer("base")
                early_before = base.get_layer("early").get_weights()
                bn_before = base.get_layer("normalization").get_weights()
                tail_before = base.get_layer("tail").get_weights()
                model.compile_teacher(
                    optimizer=teacher_optimizer, loss="sparse_categorical_crossentropy", 
                    metrics=["accuracy"], run_eagerly=False, jit_compile=False
                )
                self.assertFalse(model.compiled)
                self.assertTrue(teacher.compiled)
                self.assertFalse(teacher.trainable)
                self.assertIs(teacher.optimizer, teacher_optimizer)
                with self.assertRaisesRegex(ValueError, "compile"):
                    model.fit_teacher(self.dataset(), epochs=1, verbose=0)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(.002), loss="mse", 
                    run_eagerly=False, jit_compile=False
                )
                student_before = model.network.get_weights()
                history = model.fit_teacher(
                    self.dataset(), validation_data=self.dataset(), epochs=1, verbose=0
                )
                self.assertIn("accuracy", history.history)
                self.assertIn("val_accuracy", history.history)
                self.assertEqual(teacher.get_compile_config()["loss"], "sparse_categorical_crossentropy")
                self.assertEqual(int(teacher_optimizer.iterations), 1)
                self.assertEqual(int(model.optimizer.iterations), 0)
                self.assertFalse(teacher.trainable)
                self.assert_weights_equal(model.network, student_before)
                self.assert_weights_equal(base.get_layer("early"), early_before)
                self.assert_weights_equal(base.get_layer("normalization"), bn_before)
                self.assertTrue(any(not np.array_equal(before, after) for before, after in
                                    zip(tail_before, base.get_layer("tail").get_weights())))

    def test_recompile_preserves_student_state_and_restores_mask_during_compile(self) -> None:
        """Reconfigure teacher optimization while leaving compiled student traces intact."""

        teacher = self.make_teacher()
        model = self.make_wrapper(teacher)
        student_before = model.network.get_weights()
        student_optimizer = model.optimizer
        student_optimizer.iterations.assign(7)
        sentinels = {name: object() for name in (
            "train_function", "test_function", "predict_function"
        )}
        for name, sentinel in sentinels.items():
            setattr(model, name, sentinel)
        teacher_optimizer = tf.keras.optimizers.Adam(.0001)
        original_compile = teacher.compile

        def compile_classifier(**kwargs: object) -> None:
            """Require compilation to see the original partial fine-tuning mask."""

            self.assert_fit_mask(teacher)
            original_compile(**kwargs)

        with patch.object(teacher, "compile", side_effect=compile_classifier) as delegated:
            model.compile_teacher(
                optimizer=teacher_optimizer, loss="sparse_categorical_crossentropy", 
                metrics=["accuracy"], run_eagerly=False, jit_compile=False
            )
        delegated.assert_called_once()
        self.assertIs(teacher.optimizer, teacher_optimizer)
        self.assertFalse(teacher.trainable)
        self.assertIs(model.optimizer, student_optimizer)
        self.assertEqual(int(student_optimizer.iterations), 7)
        self.assert_weights_equal(model.network, student_before)
        for name, sentinel in sentinels.items():
            self.assertIs(getattr(model, name), sentinel)

    def test_compile_failure_refreezes_teacher_and_preserves_training_mask(self) -> None:
        """A failed Keras compile leaves the teacher frozen and able to recover."""

        teacher = self.make_teacher()
        model = self.make_wrapper(teacher)
        student_optimizer = model.optimizer

        def fail(**kwargs: object) -> None:
            """Raise after observing the restored classifier training mask."""

            self.assert_fit_mask(teacher)
            raise RuntimeError("deliberate teacher compile failure")

        with patch.object(teacher, "compile", side_effect=fail), \
             self.assertRaisesRegex(RuntimeError, "deliberate"):
            model.compile_teacher(optimizer="sgd", loss="sparse_categorical_crossentropy")
        self.assertFalse(teacher.trainable)
        self.assertEqual(teacher.trainable_weights, [])
        self.assertIs(model.optimizer, student_optimizer)
        with patch.object(teacher, "compile", side_effect=lambda **kwargs: self.assert_fit_mask(teacher)):
            model.compile_teacher(optimizer="sgd", loss="sparse_categorical_crossentropy")
        self.assertFalse(teacher.trainable)

    def test_v2_shared_optimizers_are_rejected_before_either_compile_mutates(self) -> None:
        """Reject direct and loss-scaled aliases of either side's optimizer state."""

        teacher = self.make_teacher()
        model = self.make_wrapper(teacher, DiffusionClassifierV2)
        original_teacher_optimizer = teacher.optimizer
        original_student_optimizer = model.optimizer
        original_generator_optimizer = model.gen_optimizer
        original_classifier_optimizer = model.clf_optimizer
        for optimizer in (
            model.gen_optimizer, 
            tf.keras.mixed_precision.LossScaleOptimizer(model.clf_optimizer)
        ):
            with patch.object(teacher, "compile") as delegated, \
                 self.assertRaisesRegex(ValueError, "optimizer"):
                model.compile_teacher(
                    optimizer=optimizer, loss="sparse_categorical_crossentropy"
                )
            delegated.assert_not_called()
            self.assertIs(teacher.optimizer, original_teacher_optimizer)
            self.assertFalse(teacher.trainable)
        for optimizer in (
            teacher.optimizer, 
            tf.keras.mixed_precision.LossScaleOptimizer(teacher.optimizer)
        ):
            with self.assertRaisesRegex(ValueError, "optimizer"):
                model.compile(optimizer=optimizer, loss="mse", jit_compile=False)
            self.assertIs(model.optimizer, original_student_optimizer)
            self.assertIs(model.gen_optimizer, original_generator_optimizer)
            self.assertIs(model.clf_optimizer, original_classifier_optimizer)
            self.assertIs(teacher.optimizer, original_teacher_optimizer)


class CompileNativeTeacherTests(unittest.TestCase):
    """Retain explicit teacher compile settings across student and fit lifecycles."""

    setUp = native_fixtures.FitTeacherTests.setUp
    tearDown = native_fixtures.FitTeacherTests.tearDown
    make_network = native_fixtures.FitTeacherTests.make_network
    make_wrapper = native_fixtures.FitTeacherTests.make_wrapper
    dataset = native_fixtures.FitTeacherTests.dataset
    assert_student_unchanged = native_fixtures.FitTeacherTests.assert_student_unchanged

    def test_explicit_compile_survives_student_compile_and_replacement_uses_defaults(self) -> None:
        """Keep a native teacher's optimizer until that teacher is replaced."""

        for classifier in (False, True):
            with self.subTest(classifier=classifier):
                model = self.make_wrapper(classifier, compile_model=False)
                teacher_optimizer = tf.keras.optimizers.SGD(.02)
                model.compile_teacher(
                    optimizer=teacher_optimizer, loss="mae", 
                    run_eagerly=True, jit_compile=False
                )
                teacher = model._teacher_model
                self.assertFalse(model.compiled)
                self.assertIs(teacher.optimizer, teacher_optimizer)
                self.assertFalse(model.teacher_network.trainable)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(.001), loss="mse", 
                    run_eagerly=True, jit_compile=False
                )
                student_before = model.network.get_weights()
                model.fit_teacher(self.dataset([0, 1]), epochs=1, verbose=0)
                self.assertIs(teacher.optimizer, teacher_optimizer)
                self.assertEqual(teacher.get_compile_config()["loss"], "mae")
                self.assertEqual(int(teacher_optimizer.iterations), 1)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(.002), loss="mse", 
                    run_eagerly=True, jit_compile=False
                )
                self.assertIs(model._teacher_model, teacher)
                self.assertIs(teacher.optimizer, teacher_optimizer)
                self.assertEqual(int(teacher_optimizer.iterations), 1)
                self.assert_student_unchanged(model, student_before)
                replacement = self.make_network(classifier)
                model.set_teacher_network(replacement)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(.003), loss="mse", 
                    run_eagerly=True, jit_compile=False
                )
                self.assertIsNot(model._teacher_model, teacher)
                self.assertIs(model._teacher_model.network, replacement)
                self.assertIsInstance(model._teacher_model.optimizer, tf.keras.optimizers.Adam)
                self.assertEqual(model._teacher_model.get_compile_config()["loss"], "mse")
                self.assertFalse(replacement.trainable)

    def test_v2_compile_populates_both_teacher_groups_and_trains_both_phases(self) -> None:
        """Compile while the raw teacher is enabled so V2 phases retain variables."""

        model = DiffusionClassifierV2(
            network=self.make_network(True), teacher_network=self.make_network(True), 
            trainable_teacher=True, use_ema=False, scheduler_name="linear", 
            test_steps=2, p_uncond=0., seed=541
        )
        teacher_optimizer = tf.keras.optimizers.SGD(.01)
        model.compile_teacher(
            optimizer=teacher_optimizer, loss="mae", run_eagerly=True, jit_compile=False
        )
        teacher = model._teacher_model
        self.assertTrue(teacher.gen_trainable_variables)
        self.assertTrue(teacher.clf_trainable_variables)
        self.assertIs(teacher.gen_optimizer, teacher_optimizer)
        classifier_optimizer = teacher.clf_optimizer
        self.assertIsNot(classifier_optimizer, teacher_optimizer)
        self.assertFalse(model.teacher_network.trainable)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(.001), loss="mse", 
            run_eagerly=True, jit_compile=False
        )
        student_before = model.network.get_weights()
        history = model.fit_teacher(
            gen_kwargs={"x": self.dataset([0, 1]), "epochs": 1, "verbose": 0}, 
            clf_kwargs={"x": self.dataset([0, 1]), "epochs": 1, "verbose": 0}
        )
        self.assertIn("noise_loss", history)
        self.assertIn("classifier_loss", history)
        self.assertIs(teacher.gen_optimizer, teacher_optimizer)
        self.assertIs(teacher.clf_optimizer, classifier_optimizer)
        self.assertEqual(int(teacher.gen_optimizer.iterations), 1)
        self.assertEqual(int(teacher.clf_optimizer.iterations), 1)
        self.assertEqual(int(model.gen_optimizer.iterations), 0)
        self.assertEqual(int(model.clf_optimizer.iterations), 0)
        self.assertFalse(model.teacher_network.trainable)
        self.assert_student_unchanged(model, student_before)

    def test_native_compile_failure_refreezes_teacher(self) -> None:
        """Refreeze the raw network even when the native wrapper compile fails."""

        model = self.make_wrapper()
        teacher = model._teacher_model

        def fail(**kwargs: object) -> None:
            """Verify native variables are enabled before compilation fails."""

            self.assertTrue(model.teacher_network.trainable)
            raise RuntimeError("deliberate native compile failure")

        with patch.object(teacher, "compile", side_effect=fail), \
             self.assertRaisesRegex(RuntimeError, "deliberate"):
            model.compile_teacher(optimizer="sgd", loss="mae")
        self.assertFalse(model.teacher_network.trainable)
        self.assertEqual(int(model.optimizer.iterations), 0)

    def test_compile_requires_training_opt_in_and_attached_teacher(self) -> None:
        """Report missing teacher training prerequisites before compilation."""

        frozen = self.make_wrapper(trainable_teacher=False)
        with self.assertRaisesRegex(ValueError, "trainable_teacher"):
            frozen.compile_teacher(optimizer="sgd", loss="mse")
        missing = self.make_wrapper(teacher_network=None)
        with self.assertRaisesRegex(ValueError, "teacher"):
            missing.compile_teacher(optimizer="sgd", loss="mse")


# Allow focused regression execution without running on import.
if __name__ == "__main__":
    unittest.main()
