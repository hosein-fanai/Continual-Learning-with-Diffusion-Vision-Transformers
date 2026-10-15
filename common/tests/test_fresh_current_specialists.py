"""Fresh current experts remain independent of students and previous snapshots."""

from contextlib import redirect_stdout
import io
import unittest

import numpy as np
import tensorflow as tf

from common.learner import continually_learn
from common.tests.test_head_teachers import HeadTeacherTests
from diffusion import DiffusionModel


class FreshCurrentSpecialistTests(unittest.TestCase):
    """Exercise actual task fits with fresh classifier and denoiser specialists."""

    def setUp(self) -> None:
        """Prepare the existing tiny joint-training fixture."""

        self.fixture = HeadTeacherTests()
        self.fixture.setUp()

    def tearDown(self) -> None:
        """Release this isolated test process's model state."""

        self.fixture.tearDown()

    def test_new_experts_reset_each_task_without_resetting_previous_snapshot(self) -> None:
        """Run both experts and student warmup while preserving task-local vocabularies."""

        model = self.fixture.make_model(num_classes=None, noise_loss_coef=1., clf_loss_coef=1.)
        created = {"classifier": [], "noise": []}
        observations = {"classifier": [], "noise": [], "student": []}

        def classifier_factory(task_index: int, new_classes: list[int], seed: int | None) -> tf.keras.Model:
            """Use a fixed ten-column classifier, as in the requested EfficientNet recipe."""

            teacher = tf.keras.Sequential([
                tf.keras.Input((4, 4, 1)), tf.keras.layers.Rescaling(1. / 255.), 
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(
                    10, activation="softmax", kernel_initializer=tf.keras.initializers.GlorotUniform(seed)
                )
            ])
            teacher.compile(optimizer=tf.keras.optimizers.Adam(.003), 
                            loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            created["classifier"].append((teacher, task_index, new_classes, seed))
            return teacher

        def noise_factory(task_index: int, new_classes: list[int], seed: int | None) -> DiffusionModel:
            """Construct a separately compiled native denoiser with no prior vocabulary."""

            teacher = DiffusionModel(
                network=self.fixture.noise_network(None), use_ema=False, 
                scheduler_name="clipped_cosine", test_steps=4, preprocess_type="standardize", seed=seed
            )
            teacher.compile(optimizer=tf.keras.optimizers.Adam(.002), loss="mse", run_eagerly=True)
            created["noise"].append((teacher, task_index, new_classes, seed))
            return teacher

        class Observation(tf.keras.callbacks.Callback):
            """Record the actual phase owner and its fresh optimizer counter."""

            def __init__(self, role: str) -> None:
                """Bind the phase's independent observation list."""

                super().__init__()
                self.role = role

            def on_train_begin(self, logs: dict | None = None) -> None:
                """Observe state before the first gradient update."""

                observations[self.role].append({
                    "model": self.model, "optimizer": self.model.optimizer, 
                    "iterations": int(self.model.optimizer.iterations.numpy()), 
                    "previous": model.teacher_network
                })

        options = self.fixture.continual_options(
            model, epochs=2, reinstantiate_optimizer=True, fit_kwargs={"denoiser_warmup_epochs": 1}, 
            callbacks_list=[Observation("student")], 
            current_teacher_factories={"classifier": classifier_factory, "noise": noise_factory}, 
            current_teacher_callbacks={"classifier": [Observation("classifier")], "noise": [Observation("noise")]}
        )
        with redirect_stdout(io.StringIO()):
            result = continually_learn(**options)
        self.assertEqual(len(result["generative_histories"]), 2)
        for role in ("classifier", "noise", "student"):
            self.assertEqual(len(observations[role]), 2)
            self.assertEqual([item["iterations"] for item in observations[role]], [0, 0])
            self.assertIsNot(observations[role][0]["optimizer"], observations[role][1]["optimizer"])
        for role in ("classifier", "noise"):
            self.assertEqual([item[2] for item in created[role]], [[0, 1], [2, 3]])
            self.assertNotEqual(created[role][0][3], created[role][1][3])
            self.assertIsNot(created[role][0][0], created[role][1][0])
            self.assertIsNone(observations[role][0]["previous"])
            self.assertIsNotNone(observations[role][1]["previous"])
            for teacher, _, classes, _ in created[role]:
                mapping = getattr(teacher, "seen_classes", getattr(teacher, "_diffusion_seen_classes", {}))
                self.assertEqual(set(mapping), set(classes))
                self.assertGreater(int(teacher.optimizer.iterations.numpy()), 0)
                self.assertTrue(all(np.isfinite(value).all() for value in teacher.get_weights()))
        self.assertEqual(model.get_teacher_network("classifier").output_shape[-1], 10)
        self.assertEqual(set(model.get_teacher_model("noise").seen_classes), {2, 3})

    def test_factory_rejects_a_native_teacher_with_student_optimizer(self) -> None:
        """Reject shared native update state before any teacher or student gradient update."""

        model = self.fixture.make_model(num_classes=None)
        model.set_classifier_teacher_network(None)
        created = []

        def aliased_factory(task_index: int, new_classes: list[int], seed: int | None) -> DiffusionModel:
            """Return a compiled native owner whose optimizer incorrectly aliases the student."""

            teacher = DiffusionModel(
                network=self.fixture.noise_network(None), use_ema=False, 
                scheduler_name="clipped_cosine", test_steps=4, preprocess_type="standardize", seed=seed
            )
            teacher.compile(optimizer=model.optimizer, loss="mse", run_eagerly=True)
            created.append(teacher)
            return teacher

        options = self.fixture.continual_options(
            model, current_teacher_factories={"noise": aliased_factory}
        )
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "independent optimizers"):
            continually_learn(**options)
        self.assertEqual(len(created), 1)
        self.assertIs(created[0].optimizer, model.optimizer)
        self.assertEqual(int(model.optimizer.iterations.numpy()), 0)
        self.assertIsNone(model.teacher_network)
        self.assertFalse(created[0].network.trainable)

    def test_factory_options_reject_missing_role_and_recovery(self) -> None:
        """Reject unsupported attachment and recovery contracts before fitting."""

        for override in (
            {"current_teacher_factories": {"missing": lambda **kwargs: None}}, 
            {"current_teacher_factories": {"noise": lambda **kwargs: None}, "save_task_checkpoints": True}, 
            {"current_teacher_callbacks": {"noise": []}, "save_task_checkpoints": True}
        ):
            with self.subTest(override=override):
                model = self.fixture.make_model(num_classes=None)
                with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    continually_learn(**self.fixture.continual_options(model, **override))


# Direct invocation starts unittest after all definitions are available.
if __name__ == "__main__":
    unittest.main()
