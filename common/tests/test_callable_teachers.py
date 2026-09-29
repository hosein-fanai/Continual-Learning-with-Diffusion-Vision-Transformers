"""Behavioral coverage for ordinary Keras teachers with only a call method."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import (
    DiffusionClassifierConfig, DiffusionClassifierV2Config, DiffusionModelConfig,
)
from common.tests import test_clean_classifier_training as fixtures
from diffusion import (
    DiffusionClassifier, DiffusionClassifierV2, DiffusionModel, DiffusionTransformer,
)


class _ImageTeacher(tf.keras.Model):
    """Score clean images while rejecting extra inputs or training-mode calls."""

    def __init__(self, images: tf.Tensor, logits: bool = False, invalid: str | None = None) -> None:
        """Create a minimal image-only classifier with optional malformed outputs."""
        super().__init__(name="image_teacher")
        self.expected_images = images
        self.logits = logits
        self.invalid = invalid
        self.normalization = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(.9)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.head = tf.keras.layers.Dense(
            2, kernel_initializer="ones",
            bias_initializer=tf.keras.initializers.Constant([1., -1.]),
        )

    def call(self, inputs: tf.Tensor | tuple[tf.Tensor, ...], training: bool = False) -> tf.Tensor:
        """Run inference using only the public Keras call arguments."""
        tf.debugging.assert_rank(inputs, 4)
        tf.debugging.assert_equal(inputs, self.expected_images)
        tf.debugging.assert_equal(training, False)
        hidden = self.normalization(inputs, training=training)
        hidden = self.dropout(hidden, training=training)
        scores = self.head(self.pool(hidden))
        scores = scores if self.logits else tf.nn.softmax(scores)
        # Produce a deliberate rank mismatch for validation coverage.
        if self.invalid == "rank":
            return scores[:, 0]
        # Produce a teacher batch that cannot match the student.
        if self.invalid == "batch":
            return scores[:1]
        # Use negative values to violate the probability contract.
        if self.invalid == "negative":
            return scores - 1.
        # Keep finite positive values but violate probability normalization.
        if self.invalid == "not_normalized":
            return scores * 2.
        # Inject nonfinite values into an otherwise valid output.
        if self.invalid == "nan":
            return scores * tf.constant(float("nan"))
        return scores


class _NoiseTeacher(tf.keras.Model):
    """Predict epsilon without repository metadata or full_return arguments."""

    def __init__(self, input_type: str = "images_timesteps_labels", invalid: str | None = None) -> None:
        """Choose a callable input contract and optional malformed epsilon output."""
        super().__init__(name="noise_teacher")
        self.input_type = input_type
        self.invalid = invalid
        self.gain = self.add_weight(name="gain", shape=(), initializer="ones")

    def call(self, inputs: tf.Tensor | tuple[tf.Tensor, ...], training: bool = False) -> tf.Tensor:
        """Run inference using only the public Keras call arguments."""
        tf.debugging.assert_equal(training, False)
        # Image-only teachers receive no timestep or label container.
        if self.input_type == "images":
            images = inputs
            timestep = labels = None
        # Unpack exactly the configured timestep and optional label inputs.
        else:
            expected_count = 3 if self.input_type == "images_timesteps_labels" else 2
            # Reject any additional or missing callable inputs.
            if not isinstance(inputs, (tuple, list)) or len(inputs) != expected_count:
                raise ValueError("Wrong callable noise teacher inputs")
            images, timestep = inputs[:2]
            labels = inputs[2] if expected_count == 3 else None
        tf.debugging.assert_rank(images, 4)
        noise = tf.ones_like(images) * self.gain
        # Make timestep routing visible in the predicted epsilon.
        if timestep is not None:
            noise += tf.cast(timestep[:, None, None, None], noise.dtype)
        # Make conditional and null-label predictions numerically distinct.
        if labels is not None:
            noise += tf.cast(labels[:, None, None, None], noise.dtype) * 10.
        # Deliberately remove the channel dimension contents.
        if self.invalid == "shape":
            return noise[:, :, :, :0]
        # Require the adapter to reject multiple epsilon tensors.
        if self.invalid == "tuple":
            return noise, noise
        # Inject nonfinite values into an otherwise valid output.
        if self.invalid == "nan":
            return noise * tf.constant(float("nan"))
        return noise


class CallableTeacherTests(unittest.TestCase):
    """Keep generic teacher inference separate from the student's noisy inputs."""

    setUp = fixtures.CleanClassifierTrainingTests.setUp
    tearDown = fixtures.CleanClassifierTrainingTests.tearDown
    make_network = fixtures.CleanClassifierTrainingTests.make_network

    def classifier(self, wrapper_cls: type = DiffusionClassifier, teacher: tf.keras.Model | None = None, **overrides: object) -> DiffusionClassifier:
        """Compile a noisy student whose sole classification objective is teacher KD."""
        # Provide a strict image-only teacher when the caller omits one.
        if teacher is None:
            teacher = _ImageTeacher(self.images)
        teacher(self.images, training=False)
        options = dict(
            network=self.make_network(), teacher_network=teacher, use_ema=False,
            seed=811, scheduler_name="clipped_cosine", test_steps=4, p_uncond=0.,
            mask_by_nulls=False, mask_by_t_threshold=False,
            clf_loss_coef=0., noise_loss_coef=0., clf_distil_loss_coef=1.,
            clf_distil_type="soft", clf_distil_temperature=2.,
            clf_train_noisified_max_timesteps=-1,
            clf_test_noisified_max_timesteps=-1,
        )
        options.update(overrides)
        model = wrapper_cls(**options)
        model.compile(optimizer=tf.keras.optimizers.SGD(.05), loss="mse",
                      run_eagerly=False, jit_compile=False)
        return model

    def noise_model(self, teacher: tf.keras.Model | None = None, **overrides: object) -> DiffusionModel:
        """Compile a small student with callable epsilon distillation enabled."""
        options = dict(
            network=DiffusionTransformer(
                image_size=4, channels=1, patch_size=2, dim=4, depth=1,
                mha_num_heads=1, vit_block_mlp_ratio=1., num_classes=2,
                timesteps=8, use_cfg=True, seed=811,
            ),
            teacher_network=teacher if teacher is not None else _NoiseTeacher(),
            use_ema=False, seed=811, scheduler_name="clipped_cosine", test_steps=4,
            p_uncond=0., noise_loss_coef=0., noise_distil_loss_coef=1.,
        )
        options.update(overrides)
        model = DiffusionModel(**options)
        model.compile(optimizer=tf.keras.optimizers.SGD(.001), loss="mse",
                      run_eagerly=False, jit_compile=False)
        return model

    def dataset(self) -> tf.data.Dataset:
        """Return one deterministic batch with a bounded private thread pool."""
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        return tf.data.Dataset.from_tensor_slices(
            (self.images, self.labels)
        ).batch(4).with_options(options)

    def assert_finite(self, values: object) -> None:
        """Require every returned metric to remain finite."""
        self.assertTrue(all(np.isfinite(np.asarray(value)).all() for value in values))

    def test_classification_mapping_and_graph_steps_keep_teacher_clean(self) -> None:
        """Exercise cached graph training and validation with clean teacher targets."""
        for wrapper_cls in (DiffusionClassifier, DiffusionClassifierV2):
            for training in (True, False):
                with self.subTest(wrapper=wrapper_cls.__name__, training=training):
                    model = self.classifier(wrapper_cls)
                    # Select V2 classifier phases before preparing mapped tensors.
                    if wrapper_cls is DiffusionClassifierV2:
                        model._switch_train_part("discriminator")
                        model._test_part = "discriminator"
                    model._preprocess_training = training
                    mapped = model.prep_inputs_map(self.images, self.labels)
                    model._preprocess_training = None
                    expected_scores = model.teacher_network(self.images, training=False)
                    np.testing.assert_allclose(mapped[-1], expected_scores)
                    original_predict = model.network.predict_class

                    def predict(inputs: tuple[tf.Tensor, ...], **kwargs: object) -> tuple:
                        """Verify actual student predictions consume corrupted images."""
                        tf.debugging.assert_greater(
                            tf.reduce_max(tf.abs(inputs[0] - self.images)), 0.,
                            message="Student should receive genuinely noisy images",
                        )
                        return original_predict(inputs, **kwargs)

                    with patch.object(model.network, "predict_class", side_effect=predict):
                        step = model.train_step if training else model.test_step
                        results = tf.function(step)(mapped)
                    self.assertGreater(float(results["clf_distil_loss"]), 0.)
                    self.assert_finite(results.values())

    def test_classifier_fit_updates_student_and_freezes_teacher(self) -> None:
        """Fit V1 and V2 students while preserving teacher weights and BN state."""
        for wrapper_cls in (DiffusionClassifier, DiffusionClassifierV2):
            with self.subTest(wrapper=wrapper_cls.__name__):
                model = self.classifier(wrapper_cls)
                teacher = model.teacher_network
                teacher_before = teacher.get_weights()
                student_before = model.network.get_weights()
                fit = model.fit_discriminator if wrapper_cls is DiffusionClassifierV2 else model.fit
                history = fit(x=self.dataset(), validation_data=self.dataset(),
                              epochs=1, verbose=0)
                self.assert_finite(history.history.values())
                self.assertGreater(history.history["clf_distil_loss"][0], 0.)
                self.assertGreater(history.history["val_clf_distil_loss"][0], 0.)
                self.assertTrue(any(not np.array_equal(before, after) for before, after in
                                    zip(student_before, model.network.get_weights())))
                for before, after in zip(teacher_before, teacher.get_weights()):
                    np.testing.assert_array_equal(before, after)
                self.assertFalse(teacher.trainable)
                self.assertFalse({id(v) for v in teacher.weights}
                                 & {id(v) for v in model.weights})

    def test_image_logits_are_softmaxed_once_and_targets_stop_gradients(self) -> None:
        """Normalize logits once and detach both probability and logit targets."""
        for logits in (False, True):
            with self.subTest(logits=logits):
                teacher = _ImageTeacher(self.images, logits=logits)
                model = self.classifier(teacher=teacher, teacher_classifier_from_logits=logits)
                with tf.GradientTape() as tape:
                    tape.watch(self.images)
                    mapped = model.prep_inputs_map(self.images, self.labels)
                    target_sum = tf.reduce_sum(mapped[-1][:, 0])
                scores = teacher(self.images, training=False)
                expected = tf.nn.softmax(scores) if logits else scores
                np.testing.assert_allclose(mapped[-1], expected, rtol=1e-6)
                self.assertIsNone(tape.gradient(target_sum, self.images))

    def test_noise_input_modes_and_classifier_free_guidance(self) -> None:
        """Adapt all supported callable inputs and retain guided epsilon arithmetic."""
        for input_type in ("images", "images_timesteps", "images_timesteps_labels"):
            with self.subTest(input_type=input_type):
                options = {} if input_type == "images_timesteps_labels" else {
                    "teacher_noise_input_type": input_type,
                }
                model = self.noise_model(_NoiseTeacher(input_type), **options)
                times = tf.constant([1, 2, 3, 4], tf.int32)
                cond = tf.constant([1, 2, 1, 2], tf.int32)
                uncond = tf.zeros_like(cond)
                for scale in (None, 2.):
                    with self.subTest(scale=scale):
                        _, actual, *_ = model.forward(
                            "teacher", self.images, times, times,
                            cond_labels=cond, uncond_labels=uncond,
                            scale=scale, training=False,
                        )
                        expected = tf.ones_like(self.images)
                        # Time-aware teachers add their supplied timestep to epsilon.
                        if input_type != "images":
                            expected += tf.cast(times[:, None, None, None], tf.float32)
                        # Only conditional teachers contribute the guided label term.
                        if input_type == "images_timesteps_labels":
                            expected += tf.cast(cond[:, None, None, None], tf.float32) \
                                * (10. if scale is None else 20.)
                        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_noise_distillation_trains_and_evaluates_with_frozen_plain_teacher(self) -> None:
        """Update the student using graph-mode epsilon KD and preserve the teacher."""
        model = self.noise_model()
        teacher_before = model.teacher_network.get_weights()
        student_before = model.network.get_weights()
        with tf.GradientTape() as tape:
            tape.watch(model.teacher_network.gain.value)
            target = model.prep_inputs_map(self.images, self.labels)[-2]
            target_sum = tf.reduce_sum(target)
        self.assertIsNone(tape.gradient(target_sum, model.teacher_network.gain.value))
        for training in (True, False):
            model._preprocess_training = training
            mapped = model.prep_inputs_map(self.images, self.labels)
            model._preprocess_training = None
            step = model.train_step if training else model.test_step
            results = tf.function(step)(mapped)
            self.assertGreater(float(results["noise_distil_loss"]), 0.)
            self.assert_finite(results.values())
        self.assertTrue(any(not np.array_equal(before, after) for before, after in
                            zip(student_before, model.network.get_weights())))
        for before, after in zip(teacher_before, model.teacher_network.get_weights()):
            np.testing.assert_array_equal(before, after)
        self.assertFalse({id(v) for v in model.teacher_network.weights}
                         & {id(v) for v in model.weights})

    def test_noise_only_teacher_works_in_joint_wrappers(self) -> None:
        """Run joint V1 and V2 generator KD without any teacher class method."""
        for wrapper_cls in (DiffusionClassifier, DiffusionClassifierV2):
            with self.subTest(wrapper=wrapper_cls.__name__):
                model = wrapper_cls(
                    network=self.make_network(), teacher_network=_NoiseTeacher(),
                    use_ema=False, scheduler_name="clipped_cosine", test_steps=4,
                    p_uncond=0., mask_by_nulls=False, mask_by_t_threshold=False,
                    noise_loss_coef=0., noise_distil_loss_coef=1.,
                    clf_loss_coef=1., clf_distil_loss_coef=0., seed=811,
                )
                model.compile(optimizer=tf.keras.optimizers.SGD(.001), loss="mse",
                              run_eagerly=False, jit_compile=False)
                # V2 keeps epsilon KD in its generator phase.
                if wrapper_cls is DiffusionClassifierV2:
                    model._switch_train_part("generator")
                    model._test_part = "generator"
                for training in (True, False):
                    model._preprocess_training = training
                    mapped = model.prep_inputs_map(self.images, self.labels)
                    model._preprocess_training = None
                    self.assertFalse(model.use_classifier_distil)
                    self.assertEqual(len(mapped), 9)
                    step = model.train_step if training else model.test_step
                    results = tf.function(step)(mapped)
                    self.assertGreater(float(results["noise_distil_loss"]), 0.)
                    self.assert_finite(results.values())

    def test_raw_v2_classifier_preparation_keeps_teacher_clean(self) -> None:
        """Use clean targets even when the V2 raw-batch path noises the student."""
        model = self.classifier(DiffusionClassifierV2)
        model._switch_train_part("discriminator")
        prepared, targets, _ = model._prepare_discriminator_batch(
            (self.images, self.labels), 8,
        )
        self.assertGreater(float(tf.reduce_max(tf.abs(prepared[1] - self.images))), 0.)
        np.testing.assert_allclose(targets, model.teacher_network(self.images, training=False))

    def test_invalid_classification_outputs_fail_before_updates(self) -> None:
        """Reject wrong shape, invalid probabilities, and nonfinite scores early."""
        for invalid in ("rank", "batch", "negative", "not_normalized", "nan"):
            with self.subTest(invalid=invalid):
                model = self.classifier(teacher=_ImageTeacher(self.images, invalid=invalid))
                with self.assertRaises((ValueError, tf.errors.InvalidArgumentError)):
                    model.prep_inputs_map(self.images, self.labels)
                self.assertEqual(int(model.optimizer.iterations), 0)

    def test_invalid_noise_outputs_fail_before_updates(self) -> None:
        """Reject malformed epsilon outputs before an optimizer can update."""
        for invalid in ("shape", "tuple", "nan"):
            with self.subTest(invalid=invalid):
                model = self.noise_model(_NoiseTeacher(invalid=invalid))
                with self.assertRaises((ValueError, TypeError, tf.errors.InvalidArgumentError)):
                    model.prep_inputs_map(self.images, self.labels)
                self.assertEqual(int(model.optimizer.iterations), 0)

    def test_generic_teachers_cannot_opt_into_repository_teacher_training(self) -> None:
        """Reject fit_teacher configuration for unrelated Keras architectures."""
        for classifier in (False, True):
            with self.subTest(classifier=classifier):
                make = self.classifier if classifier else self.noise_model
                with self.assertRaisesRegex(ValueError, "trainable_teacher|fit_teacher|teacher"):
                    make(trainable_teacher=True)

    def test_adapter_options_round_trip_without_serializing_teacher_weights(self) -> None:
        """Preserve adapter options in wrapper and typed configuration round trips."""
        for wrapper_cls, config_cls in (
            (DiffusionClassifier, DiffusionClassifierConfig),
            (DiffusionClassifierV2, DiffusionClassifierV2Config),
        ):
            with self.subTest(wrapper=wrapper_cls.__name__):
                model = self.classifier(
                    wrapper_cls, teacher=_ImageTeacher(self.images, logits=True),
                    teacher_classifier_from_logits=True, teacher_noise_input_type="images",
                    defer_teacher=True,
                )
                serialized = json.loads(json.dumps(model.get_config()))
                self.assertNotIn("teacher_network", serialized)
                clone = wrapper_cls.from_config(serialized)
                self.assertTrue(clone.teacher_classifier_from_logits)
                self.assertEqual(clone.teacher_noise_input_type, "images")
                typed = config_cls(teacher_classifier_from_logits=True,
                                   teacher_noise_input_type="images")
                self.assertTrue(typed.kwargs()["teacher_classifier_from_logits"])
                self.assertEqual(typed.kwargs()["teacher_noise_input_type"], "images")
        model = self.noise_model(teacher_noise_input_type="images", defer_teacher=True)
        clone = DiffusionModel.from_config(model.get_config())
        self.assertEqual(clone.teacher_noise_input_type, "images")
        self.assertEqual(DiffusionModelConfig().teacher_noise_input_type,
                         "images_timesteps_labels")
        with self.assertRaises((ValueError, AssertionError)):
            self.noise_model(teacher_noise_input_type="unknown")


# Allow focused execution without starting tests during imports.
if __name__ == "__main__":
    unittest.main()
