"""Behavioral coverage for independent teacher fitting and continual growth."""

import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import DiffusionClassifierConfig, DiffusionModelConfig
from common.learner import _run_continual_tasks
from common.train import train_model
from diffusion import DiTClassifier, DiffusionClassifier, DiffusionModel, DiffusionTransformer


class FitTeacherTests(unittest.TestCase):
    """Train tiny teachers through the existing wrapper fit implementations."""

    def setUp(self) -> None:
        """Use seeded float32 models and a two-image training batch."""

        self.original_policy = tf.keras.mixed_precision.global_policy().name
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.utils.set_random_seed(541)
        self.images = tf.reshape(tf.linspace(-1., 1., 32), (2, 4, 4, 1))

    def tearDown(self) -> None:
        """Release model state and restore the caller's numerical policy."""

        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy(self.original_policy)

    def make_network(
        self, classifier: bool = False, num_classes: int | None = 2
    ) -> DiffusionTransformer:
        """Construct a small configurable network with one transformer block."""

        options = dict(
            image_size=4, channels=1, patch_size=2, dim=4, depth=1, 
            mha_num_heads=1, vit_block_mlp_ratio=1., num_classes=num_classes, 
            timesteps=4, use_cfg=True, seed=541
        )
        # A classifier adds a separate minimal classification transformer.
        if classifier:
            options.update(clf_depth=1, clf_mha_num_heads=1, 
                           clf_vit_block_mlp_ratio=1., classifier_mlp_ratio=1)
            return DiTClassifier(**options)
        return DiffusionTransformer(**options)

    def make_wrapper(
        self, classifier: bool = False, num_classes: int | None = 2, 
        compile_model: bool = True, **overrides: object
    ) -> DiffusionModel:
        """Build independent student and teacher networks with one optimizer each."""

        options = dict(
            network=self.make_network(classifier, num_classes), 
            teacher_network=self.make_network(classifier, num_classes), 
            trainable_teacher=True, use_ema=False, seed=541, 
            scheduler_name="linear", test_steps=2, p_uncond=0.
        )
        # Exercise the class objective on every image without timestep masking.
        if classifier:
            options.update(clf_loss_coef=1., clf_train_type="cond", 
                           mask_by_nulls=False, mask_by_t_threshold=False)
        options.update(overrides)
        wrapper_class = DiffusionClassifier if classifier else DiffusionModel
        model = wrapper_class(**options)
        # Uncompiled fixtures exercise the public compile precondition directly.
        if compile_model:
            model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="mse", 
                          run_eagerly=True, jit_compile=False)
        return model

    def dataset(self, labels: list[int]) -> tf.data.Dataset:
        """Return one finite batch with a bounded private TensorFlow thread pool."""

        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        return tf.data.Dataset.from_tensor_slices(
            (self.images, tf.constant(labels, tf.int32))
        ).batch(2).with_options(options)

    def assert_student_unchanged(
        self, model: DiffusionModel, before: list[np.ndarray]
    ) -> None:
        """Assert teacher fitting leaves every student-network weight unchanged."""

        self.assertEqual(len(model.network.get_weights()), len(before))
        for actual, expected in zip(model.network.get_weights(), before):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(int(model.optimizer.iterations.numpy()), 0)

    def test_fixed_classes_train_only_the_requested_teacher_objectives(self) -> None:
        """Fit noise-only and joint teachers while preserving the student state."""

        for classifier in (False, True):
            with self.subTest(classifier=classifier):
                model = self.make_wrapper(classifier, image_loss_coef=1., 
                                          noise_loss_coef=0.)
                student_before = model.network.get_weights()
                teacher_before = model.teacher_network.get_weights()
                history = model.fit_teacher(self.dataset([0, 1]), epochs=1, verbose=0)
                teacher_model = model._teacher_model
                self.assertIs(teacher_model.network, model.teacher_network)
                self.assertIsNot(teacher_model.optimizer, model.optimizer)
                self.assertEqual(int(teacher_model.optimizer.iterations.numpy()), 1)
                self.assertGreater(history.history["noise_loss"][0], 0.)
                self.assertEqual("classifier_loss" in history.history, classifier)
                self.assertNotIn("image_loss", history.history)
                self.assertNotIn("noise_distil_loss", history.history)
                self.assertTrue(all(np.isfinite(values).all()
                                    for values in history.history.values()))
                self.assertTrue(any(not np.array_equal(old, new) for old, new in
                                    zip(teacher_before, model.teacher_network.get_weights())))
                self.assertFalse({id(v) for v in model.teacher_network.weights}
                                 & {id(v) for v in model.weights})
                self.assertFalse(model.teacher_network.trainable)
                self.assert_student_unchanged(model, student_before)

    def test_dynamic_classes_keep_mapping_and_optimizer_between_fits(self) -> None:
        """Add a new real label without losing previous labels or optimizer steps."""

        for classifier in (False, True):
            with self.subTest(classifier=classifier):
                model = self.make_wrapper(classifier, num_classes=None)
                student_before = model.network.get_weights()
                model.fit_teacher(self.dataset([7, 9]), epochs=1, verbose=0)
                teacher_model = model._teacher_model
                old_teacher = model.teacher_network
                self.assertEqual(teacher_model.seen_classes, {7: 0, 9: 1})
                model.fit_teacher(self.dataset([12, 7]), epochs=1, verbose=0)
                self.assertIs(model._teacher_model, teacher_model)
                self.assertIsNot(model.teacher_network, old_teacher)
                self.assertIs(model.teacher_network, teacher_model.network)
                self.assertEqual(teacher_model.seen_classes, {7: 0, 9: 1, 12: 2})
                self.assertEqual(model.teacher_network.num_classes, 3)
                self.assertEqual(int(teacher_model.optimizer.iterations.numpy()), 2)
                self.assertEqual(model.seen_classes, {})
                self.assertFalse(model.teacher_network.trainable)
                self.assert_student_unchanged(model, student_before)

    def test_progressive_teacher_fit_adds_depth_and_trains_the_new_stage(self) -> None:
        """Forward progressive fit arguments to grow only the teacher network."""

        for classifier in (False, True):
            with self.subTest(classifier=classifier):
                model = self.make_wrapper(classifier)
                student_before = model.network.get_weights()
                history = model.fit_teacher(
                    self.dataset([0, 1]), fit_method="fit_progressively", 
                    stage_tasks=[("depth", "vision_transformer_block")], 
                    stage_epochs=1, final_epochs=1, stages_verbose=False, verbose=0
                )
                self.assertEqual(model.teacher_network.depth, 2)
                self.assertEqual(model.network.depth, 1)
                self.assertEqual(int(model._teacher_model.optimizer.iterations.numpy()), 2)
                self.assertEqual(len(history.history["noise_loss"]), 2)
                self.assertEqual(len(history.progressive_stages), 2)
                self.assertFalse(model.teacher_network.trainable)
                self.assert_student_unchanged(model, student_before)

    def test_supplied_grown_wrapper_preserves_its_real_label_mapping(self) -> None:
        """Retain a supplied teacher wrapper's vocabulary when a new class arrives."""

        for classifier in (False, True):
            with self.subTest(classifier=classifier):
                supplied = self.make_wrapper(classifier, num_classes=None, 
                                             teacher_network=None, 
                                             trainable_teacher=False)
                supplied.fit(self.dataset([7, 9]), epochs=1, verbose=0)
                model = self.make_wrapper(classifier, num_classes=None, 
                                          teacher_network=supplied)
                model.fit_teacher(self.dataset([12, 7]), epochs=1, verbose=0)
                self.assertEqual(model._teacher_model.seen_classes, 
                                 {7: 0, 9: 1, 12: 2})
                self.assertEqual(model.teacher_network.num_classes, 3)
                self.assertEqual(model.seen_classes, {})

    def test_constructor_training_settings_reach_teacher_fit(self) -> None:
        """Use the configured scheduler, preprocessing and timestep range."""

        model = self.make_wrapper(
            True, map_preprocess=True, scheduler_name="clipped_cosine", 
            train_noisified_min_timesteps=1, train_noisified_max_timesteps=3, 
            clf_train_noisy_input_type="clean", 
            clf_train_class_input_type="null_class_only", clf_loss_coef=0.
        )
        history = model.fit_teacher(self.dataset([0, 1]), epochs=1, verbose=0)
        teacher_model = model._teacher_model
        self.assertTrue(teacher_model.map_preprocess)
        self.assertEqual(teacher_model.scheduler_name, "clipped_cosine")
        self.assertEqual(teacher_model.train_noisified_min_timesteps, 1)
        self.assertEqual(teacher_model.train_noisified_max_timesteps, 3)
        self.assertEqual(teacher_model.clf_train_noisy_input_type, "clean")
        self.assertEqual(teacher_model.clf_train_class_input_type, "null_class_only")
        self.assertGreater(history.history["classifier_loss"][0], 0.)

    def test_trainable_flag_roundtrip_and_default_frozen_teacher(self) -> None:
        """Serialize the opt-in setting while preserving frozen-teacher defaults."""

        for classifier in (False, True):
            with self.subTest(classifier=classifier):
                config_class = DiffusionClassifierConfig if classifier else DiffusionModelConfig
                self.assertFalse(config_class().kwargs()["trainable_teacher"])
                self.assertEqual(config_class().kwargs()["teacher_training"], "each_task")
                self.assertEqual(config_class(teacher_training="first_task").kwargs()["teacher_training"], "first_task")
                self.assertTrue(config_class(trainable_teacher=True).kwargs()["trainable_teacher"])
                model = self.make_wrapper(classifier, teacher_training="first_task")
                clone = type(model).from_config(model.get_config())
                self.assertTrue(clone.trainable_teacher)
                self.assertEqual(clone.teacher_training, "first_task")
                frozen = self.make_wrapper(classifier, trainable_teacher=False)
                self.assertFalse(frozen.teacher_network.trainable)
                with self.assertRaisesRegex(ValueError, "trainable_teacher"):
                    frozen.fit_teacher(self.dataset([0, 1]), epochs=1, verbose=0)

    def test_replacing_teacher_rebuilds_its_compiled_fit_state(self) -> None:
        """Train a replacement teacher with the student's latest compile settings."""

        model = self.make_wrapper()
        model.fit_teacher(self.dataset([0, 1]), epochs=1, verbose=0)
        previous = model._teacher_model
        replacement = self.make_network()
        model.set_teacher_network(replacement)
        model.compile(optimizer=tf.keras.optimizers.SGD(.02), loss="mae", 
                      run_eagerly=True, jit_compile=False)
        model.fit_teacher(self.dataset([0, 1]), epochs=1, verbose=0)
        self.assertIsNot(model._teacher_model, previous)
        self.assertIs(model.teacher_network, replacement)
        self.assertIsInstance(model._teacher_model.optimizer, tf.keras.optimizers.SGD)
        self.assertIsNot(model._teacher_model.optimizer, model.optimizer)
        self.assertEqual(model._teacher_model.get_compile_config()["loss"], "mae")
        self.assertEqual(int(model._teacher_model.optimizer.iterations.numpy()), 1)
        self.assertEqual(int(model.optimizer.iterations.numpy()), 0)


    @staticmethod
    def continual_loader(indices: list[int], **kwargs: object) -> tuple:
        """Return two deterministic 4x4 images per requested original class ID."""

        del kwargs
        labels = np.repeat(np.asarray(indices, dtype="int32"), 2)
        images = np.broadcast_to(
            (labels.astype("float32") / 2. - .75)[:, None, None, None], 
            (len(labels), 4, 4, 1)
        ).copy()
        return images, labels, images.copy(), labels.copy(), images.copy(), labels.copy()

    def test_continual_teacher_policy_controls_training_and_snapshot_lifecycle(self) -> None:
        """Run both constructor-selected policies and the frozen default on two tasks."""

        for trainable, policy, expected_fits in (
            (True, "each_task", 2), (True, "first_task", 1), 
            (False, "each_task", 0)
        ):
            with self.subTest(trainable=trainable, policy=policy):
                model = self.make_wrapper(
                    True, num_classes=None, trainable_teacher=trainable, 
                    teacher_training=policy, noise_distil_loss_coef=.1
                )
                initial_teacher = model.teacher_network
                with patch.object(model, "fit_teacher", wraps=model.fit_teacher) as fit_teacher:
                    details = _run_continual_tasks(
                        class_num=4, task_size=2, load_dataset_fn=self.continual_loader, 
                        load_dataset_fn_kwargs={"preprocess": "diffusion"}, 
                        generative_model=model, use_generative_model_classifier=True, 
                        generative_model_kwargs={"train_num": -1, "samples_per_class": 1}, 
                        use_generative_replay=True, use_distillation=True, 
                        batch_size=8, epochs=1, optimizer_steps_per_epoch=1, 
                        callback_patience=0, plot_results=False, verbose=0, 
                        seed=541, show_generated_images=False, show_network_summary=False
                    )
                self.assertEqual(fit_teacher.call_count, expected_fits)
                self.assertEqual(len(details["generative_histories"]), 2)
                self.assertEqual(sum(history is not None for history in details["teacher_histories"]), expected_fits)
                self.assertEqual(int(model.optimizer.iterations.numpy()), 2)
                self.assertEqual(model.seen_classes, {0: 0, 1: 1, 2: 2, 3: 3})
                self.assertFalse(model.teacher_network.trainable)
                self.assertIsNot(model.teacher_network, initial_teacher)
                # An every-task teacher retains its optimizer and full scheduled vocabulary.
                if trainable and policy == "each_task":
                    self.assertIs(model.teacher_network, model._teacher_model.network)
                    self.assertEqual(model._teacher_model.seen_classes, model.seen_classes)
                    self.assertEqual(int(model._teacher_model.optimizer.iterations.numpy()), 2)
                    self.assertTrue(any(not np.array_equal(student, teacher) for student, teacher in
                                        zip(model.network.get_weights(), model.teacher_network.get_weights())))
                    second_fit = fit_teacher.call_args_list[1]
                    dataset = second_fit.kwargs.get("x")
                    # The public fit API accepts either positional or keyword input datasets.
                    if dataset is None:
                        dataset = second_fit.args[0]
                    trained_labels = np.concatenate([batch[1] for batch in dataset.as_numpy_iterator()])
                    self.assertEqual(set(trained_labels.tolist()), {0, 1, 2, 3})
                # A first-task-only teacher and the default finish with student snapshots.
                else:
                    for teacher, student in zip(model.teacher_network.get_weights(), model.network.get_weights()):
                        np.testing.assert_array_equal(teacher, student)

    def test_continual_teacher_initializes_without_a_supplied_teacher_or_validation(self) -> None:
        """Warm up an automatic teacher using the task's normal no-validation path."""

        model = self.make_wrapper(
            True, num_classes=None, teacher_network=None, 
            teacher_training="first_task", noise_distil_loss_coef=.1, defer_teacher=True
        )
        details = _run_continual_tasks(
            class_num=4, task_size=2, load_dataset_fn=self.continual_loader, 
            load_dataset_fn_kwargs={"preprocess": "diffusion"}, 
            generative_model=model, use_generative_model_classifier=True, 
            generative_model_kwargs={"train_num": -1}, 
            use_generative_replay=False, use_distillation=True, use_valset=False, 
            batch_size=4, epochs=1, optimizer_steps_per_epoch=1, 
            callback_patience=0, plot_results=False, verbose=0, 
            seed=541, show_generated_images=False, show_network_summary=False
        )
        self.assertEqual(len(details["teacher_histories"]), 2)
        self.assertIsNone(details["teacher_histories"][1])
        self.assertGreater(details["teacher_histories"][0]["noise_loss"][0], 0.)
        self.assertGreater(details["teacher_histories"][0]["classifier_loss"][0], 0.)
        self.assertEqual(int(model.optimizer.iterations.numpy()), 2)
        self.assertFalse(model.teacher_network.trainable)
        self.assertEqual(model.teacher_network.num_classes, 4)

    def test_shared_trainer_delegates_teacher_progressive_fit(self) -> None:
        """Use the shared training entry point without changing student depth or weights."""

        model = self.make_wrapper(True)
        student_before = model.network.get_weights()
        with self.assertRaisesRegex(ValueError, "save_weights=False"):
            train_model(None, model, self.dataset([0, 1]), 
                        fit_method="fit_teacher", save_weights=True)
        history = train_model(
            None, model, self.dataset([0, 1]), fit_method="fit_teacher", 
            fit_kwargs={
                "fit_method": "fit_progressively", 
                "stage_tasks": [("depth", "vision_transformer_block")], 
                "stage_epochs": 1, "final_epochs": 1, "stages_verbose": False
            }, 
            epochs=1, verbose=0, results_path=None, patience=0, 
            save_config_=False, show_images=True, save_gifs=False, 
            report_every_epoch=False, save_weights=False
        )
        self.assertEqual(model.teacher_network.depth, 2)
        self.assertEqual(model.network.depth, 1)
        self.assertEqual(len(history["noise_loss"]), 2)
        self.assertEqual(len(history["classifier_loss"]), 2)
        self.assertFalse(model.teacher_network.trainable)
        self.assert_student_unchanged(model, student_before)

    def test_teacher_fit_reports_missing_network_and_compile(self) -> None:
        """Reject absent or uncompiled teachers before any fit can update weights."""

        model = self.make_wrapper(teacher_network=None)
        with self.assertRaisesRegex(ValueError, "teacher"):
            model.fit_teacher(self.dataset([0, 1]), epochs=1, verbose=0)
        model = self.make_wrapper(compile_model=False)
        with self.assertRaisesRegex((ValueError, RuntimeError), "compile"):
            model.fit_teacher(self.dataset([0, 1]), epochs=1, verbose=0)


# Run this focused suite when invoked as a script.
if __name__ == "__main__":
    unittest.main()
