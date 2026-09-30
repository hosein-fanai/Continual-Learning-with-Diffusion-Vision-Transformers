"""Continual growth and exact recovery for ordinary pretrained Keras teachers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.learner import _run_continual_tasks
from common.model import get_model
from common.runtime import configure_runtime
from diffusion import DiTClassifier, DiffusionClassifier


class ContinualKerasTeacherTests(unittest.TestCase):
    """Exercise the real get_model teacher path with a bounded local backbone."""

    def setUp(self) -> None:
        """Remember the caller's numerical policy before deterministic model creation."""

        self.policy = tf.keras.mixed_precision.global_policy()

    def tearDown(self) -> None:
        """Release independent models and restore the caller's numerical policy."""

        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy(self.policy)

    @staticmethod
    def backbone(**kwargs: object) -> tf.keras.Model:
        """Replace only the downloaded EfficientNet body with a small trainable graph."""

        inputs = tf.keras.Input(shape=kwargs["input_shape"])
        hidden = tf.keras.layers.Rescaling(1. / 255., name="input_scale")(inputs)
        hidden = tf.keras.layers.Conv2D(2, 1, name="frozen_conv")(hidden)
        hidden = tf.keras.layers.BatchNormalization(name="frozen_bn")(hidden)
        hidden = tf.keras.layers.Conv2D(2, 1, name="finetuned_conv")(hidden)
        return tf.keras.Model(inputs, hidden, name="tiny_efficientnet")

    @staticmethod
    def loader(indices: list[int], **kwargs: object) -> tuple:
        """Return class-coded raw RGB pixels while retaining original class IDs."""

        del kwargs
        labels = np.repeat(np.asarray(indices, dtype="int32"), 4)
        images = np.broadcast_to(
            (30. + 50. * labels)[:, None, None, None], (len(labels), 32, 32, 3)
        ).astype("float32").copy()
        return images, labels, images.copy(), labels.copy(), images.copy(), labels.copy()

    def make_model(self) -> DiffusionClassifier:
        """Build a dynamic student and ordinary get_model teacher with separate Adam state."""

        tf.keras.backend.clear_session()
        configure_runtime(dtype_policy="float32", deterministic_ops=True, seed=613)
        with patch.object(tf.keras.applications, "EfficientNetV2L", side_effect=self.backbone):
            teacher = get_model(
                1, dataset_name="cifar10", model_type="pretrained", conv_base_name="EfficientNetV2L", 
                num_last_not_frozen=2, dropout_rate=.2, resize=(8, 8), compile_args={
                    "optimizer": tf.keras.optimizers.Adam(.003), 
                    "loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], 
                    "run_eagerly": True, "jit_compile": False
                }, 
                show_network_summary=False, 
                verbose=0, seed=613
            )
        network = DiTClassifier(
            image_size=32, channels=3, patch_size=16, dim=4, depth=1, 
            mha_num_heads=1, vit_block_mlp_ratio=1., num_classes=None, 
            timesteps=4, use_cfg=True, clf_depth=1, clf_mha_num_heads=1, 
            clf_vit_block_mlp_ratio=1., classifier_mlp_ratio=1, seed=613
        )
        model = DiffusionClassifier(
            network=network, teacher_network=teacher, teacher_dynamic_classes=True, 
            trainable_teacher=True, teacher_training="each_task", 
            use_ema=False, preprocess_type="standardize", scheduler_name="linear", 
            test_steps=2, p_uncond=0., clf_loss_coef=1., 
            noise_loss_coef=0., clf_distil_loss_coef=.2, mask_by_nulls=False, 
            mask_by_t_threshold=False, seed=613
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(.001), loss="mse", 
            run_eagerly=True, jit_compile=False
        )
        return model

    def run_options(self, model: DiffusionClassifier) -> dict:
        """Declare a small two-task protocol with reproducible output-column order."""

        return dict(
            class_num=4, class_order=[2, 0, 3, 1], task_size=2, 
            load_dataset_fn=self.loader, load_dataset_fn_kwargs={"preprocess": None}, 
            generative_model=model, use_generative_model_classifier=True, 
            generative_model_kwargs={"train_num": -1}, use_generative_replay=False, 
            use_distillation=True, batch_size=8, epochs=1, optimizer_steps_per_epoch=1, 
            callback_patience=0, plot_results=False, deterministic_ops=True, show_generated_images=False, 
            show_network_summary=False, verbose=0, seed=613
        )

    def test_two_tasks_grow_teacher_without_mutating_student_during_teacher_fit(self) -> None:
        """Fit a two-then-four-class teacher while retaining frozen backbone statistics."""

        model = self.make_model()
        frozen_before = {
            name: model.teacher_network.get_layer("tiny_efficientnet").get_layer(name).get_weights()
            for name in ("frozen_conv", "frozen_bn")
        }
        original_fit = model.fit_teacher
        observations = []

        def observe(*args: object, **kwargs: object) -> object:
            """Check isolation across the public teacher fit invoked by the learner."""

            before = model.network.get_weights()
            iteration = int(model.optimizer.iterations.numpy())
            history = original_fit(*args, **kwargs)
            self.assertEqual(int(model.optimizer.iterations.numpy()), iteration)
            for expected, actual in zip(before, model.network.get_weights()):
                np.testing.assert_array_equal(actual, expected)
            observations.append((
                model.teacher_network.output_shape[-1], 
                int(model.teacher_network.optimizer.iterations.numpy())
            ))
            self.assertFalse(model.teacher_network.trainable)
            return history

        with patch.object(model, "fit_teacher", side_effect=observe):
            details = _run_continual_tasks(mechanistic_metrics=True, **self.run_options(model))
        self.assertEqual(observations, [(2, 1), (4, 2)])
        self.assertEqual(model.teacher_network._diffusion_seen_classes, {0: 0, 1: 1, 2: 2, 3: 3})
        self.assertIs(model._teacher_model, model.teacher_network)
        self.assertEqual(len(details["teacher_histories"]), 2)
        self.assertFalse({id(value) for value in model.weights}
                         & {id(value) for value in model.teacher_network.weights})
        for name, expected_weights in frozen_before.items():
            layer = model.teacher_network.get_layer("tiny_efficientnet").get_layer(name)
            for expected, actual in zip(expected_weights, layer.get_weights()):
                np.testing.assert_array_equal(actual, expected)

    def test_keras_teacher_task_recovery_restores_head_mask_and_optimizer(self) -> None:
        """Resume task two with the same student, teacher, optimizer and fine-tuning mask."""

        with tempfile.TemporaryDirectory() as temporary:
            models = []
            for resume in (False, True):
                model = self.make_model()
                model._check_new_teacher_labels(y=np.asarray([0]), verbose=False)
                options = self.run_options(model)
                options.update(
                    save_task_checkpoints=True, 
                    checkpoint_dir=str(Path(temporary) / ("resumed" if resume else "original"))
                )
                # Restore the first committed task into a fresh independent run directory.
                if resume:
                    options["resume_from"] = str(Path(temporary) / "original" / "task-0000")
                _run_continual_tasks(**options)
                models.append(model)
        expected, actual = models
        self.assertEqual(actual.teacher_network._diffusion_seen_classes, expected.teacher_network._diffusion_seen_classes)
        self.assertEqual(actual.teacher_network.output_shape[-1], 4)
        self.assertFalse(actual.teacher_network.trainable)
        expected_flags = [flag for _, flag in expected._keras_teacher_fit_state[1]]
        self.assertEqual([flag for _, flag in actual._keras_teacher_fit_state[1]], expected_flags)
        for left, right in (
            (expected.network, actual.network), (expected.teacher_network, actual.teacher_network)
        ):
            self.assertEqual(len(left.weights), len(right.weights))
            for expected_weight, actual_weight in zip(left.weights, right.weights):
                np.testing.assert_array_equal(actual_weight.numpy(), expected_weight.numpy())
        for left, right in (
            (expected.optimizer, actual.optimizer), 
            (expected.teacher_network.optimizer, actual.teacher_network.optimizer)
        ):
            self.assertEqual(len(left.variables), len(right.variables))
            for expected_value, actual_value in zip(left.variables, right.variables):
                np.testing.assert_array_equal(actual_value.numpy(), expected_value.numpy())


# Keep these bounded integration checks available as an ordinary unittest module.
if __name__ == "__main__":
    unittest.main()
