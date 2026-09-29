"""Public continual APIs and graph-mode dual-teacher compatibility regressions."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import Config, load_config, save_config
from common.current_task_teacher import make_current_task_teacher
from common.learner import continually_learn
from common.tests import test_dual_teachers as dual_fixtures
from common.train import train_model
from diffusion import DiffusionClassifier, DiffusionClassifierV2


class DualTeacherApiTests(unittest.TestCase):
    """Exercise real public orchestration alongside independent role controls."""

    setUp = dual_fixtures.DualTeacherTests.setUp
    tearDown = dual_fixtures.DualTeacherTests.tearDown
    make_network = dual_fixtures.DualTeacherTests.make_network
    noise_network = staticmethod(dual_fixtures.DualTeacherTests.noise_network)
    noise_model = dual_fixtures.DualTeacherTests.noise_model
    classifier = dual_fixtures.DualTeacherTests.classifier
    continual_model = dual_fixtures.DualTeacherTests.continual_model
    continual_loader = staticmethod(dual_fixtures.DualTeacherTests.continual_loader)
    assert_finite = dual_fixtures.DualTeacherTests.assert_finite

    @staticmethod
    def pixels() -> tuple:
        """Replace only downloaded MNIST arrays while retaining actual preprocessing."""
        labels = np.repeat(np.asarray([7, 1, 9, 3], dtype="uint8"), 6)
        images = np.broadcast_to(labels[:, None, None] * 20, (24, 28, 28)).copy()
        test_labels = np.repeat(np.asarray([7, 1, 9, 3], dtype="uint8"), 2)
        test_images = np.broadcast_to(test_labels[:, None, None] * 20, (8, 28, 28)).copy()
        return (images, labels), (test_images, test_labels)

    @staticmethod
    def configuration(root: Path) -> Config:
        """Declare a full public warm-start run with shuffled original class IDs."""
        return Config(
            dataset=dict(name="mnist", preprocess="fixed-standardize", batch_size=8,
                         validation_ratio=.25, shuffle_buffer=24),
            model=dict(
                name="dit_classifier", wrapper_name="diffusion_classifier",
                show_network_summary=False,
                kwargs=dict(image_size=28, channels=1, patch_size=14, dim=4, depth=1,
                            mha_num_heads=1, vit_block_mlp_ratio=1., num_classes=None,
                            timesteps=4, use_cfg=True, clf_depth=1, clf_mha_num_heads=1,
                            clf_vit_block_mlp_ratio=1., classifier_mlp_ratio=1,
                            compile_args=dict(run_eagerly=False, jit_compile=False)),
                wrapper_kwargs=dict(use_ema=False, test_steps=2, scheduler_name="linear",
                                    p_uncond=0., mask_by_nulls=False, mask_by_t_threshold=False,
                                    clf_loss_coef=1., noise_distil_loss_coef=.1,
                                    clf_distil_loss_coef=.1, clf_train_batch_fraction=.5),
                classifier_kwargs=dict(architecture_kwargs=dict(conv_filters=[2], conv_depths=[1])),
            ),
            optimizer=dict(name="sgd", schedule="constant", initial_learning_rate=.01),
            training=dict(task="continual", seed=811, epochs=1, verbose=0,
                          results_path=str(root), save_weights=False, show_images=False,
                          save_gifs=False, report_every_epoch=False, patience=0),
            continually_learn=dict(
                class_num=4, class_order=[7, 1, 9, 3], task_size=2,
                use_generative_model_classifier=True, use_generative_replay=False,
                remove_prev_classes=False, use_distillation=True,
                dual_teacher_distillation=True, current_teacher_init="student",
                generative_model_kwargs=dict(train_num=-1), optimizer_steps_per_epoch=1,
                plot_results=False, show_generated_images=False, return_details=True,
            ),
            reporting=dict(save_history_plot=False, save_final_images=False,
                           save_final_gifs=False, save_csv=False,
                           run_trainset_eval=False, run_valset_eval=False),
        )

    def public_options(self, **overrides: object) -> dict:
        """Supply only serializable controls shared by the public adapters."""
        options = dict(
            class_num=4, class_order=[2, 0, 3, 1], task_size=2,
            use_generative_model_classifier=True, use_generative_replay=False,
            remove_prev_classes=False, use_distillation=True,
            dual_teacher_distillation=True, current_teacher_init="fresh",
            generative_model_kwargs=dict(train_num=-1), optimizer_steps_per_epoch=1,
            plot_results=False, show_generated_images=False, return_details=True,
        )
        options.update(overrides)
        return options

    def test_config_public_api_fits_warm_teachers_with_nontrivial_class_order(self) -> None:
        """Use YAML, public Config entry, real factory, graph split batches, and two tasks."""
        captures = []

        def factory(student: object, class_ids: list[int], **kwargs: object) -> tuple:
            """Verify each public warm teacher begins with independent student weights."""
            teacher, output_ids = make_current_task_teacher(student, class_ids, **kwargs)
            self.assertEqual(kwargs["initialization"], "student")
            for expected, actual in zip(student.network.get_weights(), teacher.network.get_weights()):
                np.testing.assert_array_equal(actual, expected)
            self.assertIsNot(teacher.optimizer, student.optimizer)
            captures.append((teacher, list(class_ids), output_ids))
            return teacher, output_ids

        with tempfile.TemporaryDirectory() as temporary:
            config = self.configuration(Path(temporary) / "results")
            path = Path(temporary) / "input.yaml"
            save_config(config, path)
            recovered = load_config(path)
            self.assertEqual(recovered.continually_learn.current_teacher_init, "student")
            self.assertTrue(recovered.continually_learn.dual_teacher_distillation)
            with patch("tensorflow.keras.datasets.mnist.load_data", side_effect=self.pixels), \
                 patch("common.learner.make_current_task_teacher", side_effect=factory), \
                 redirect_stdout(io.StringIO()):
                details = continually_learn(asdict(recovered))
        self.assertEqual(details["class_order"], [7, 1, 9, 3])
        self.assertEqual(details["task_classes"], [[7, 1], [9, 3]])
        self.assertEqual(len(details["teacher_histories"]), 2)
        self.assertEqual([entry[1] for entry in captures], [[0, 1], [2, 3]])
        self.assertEqual([entry[2] for entry in captures], [[0, 1], [0, 1, 2, 3]])
        student = details["generative_model"]
        self.assertEqual(student.network.num_classes, 4)
        self.assertEqual(int(student.optimizer.iterations), 2)
        self.assertFalse(student.run_eagerly)
        self.assertEqual(student.clf_train_batch_fraction, .5)
        self.assertIsNone(student.current_teacher_network)
        self.assertIn("evaluations", details)
        for history in details["generative_histories"]:
            self.assert_finite(history.values())
            self.assertIn("noise_distil_loss", history)
            self.assertIn("clf_distil_loss", history)

    def test_direct_public_v2_api_fits_both_graph_phases(self) -> None:
        """Run reordered tasks through public direct keywords and both V2 optimizers."""
        model = self.continual_model(DiffusionClassifierV2)
        model.clf_distil_scope = "replay_only"
        model._init_config["clf_distil_scope"] = "replay_only"
        model.compile(optimizer=tf.keras.optimizers.SGD(.01), loss="mse",
                      run_eagerly=False, jit_compile=False)
        with redirect_stdout(io.StringIO()):
            details = continually_learn(
                generative_model=model, load_dataset_fn=self.continual_loader,
                load_dataset_fn_kwargs=dict(preprocess="diffusion"), seed=811,
                batch_size=8, epochs=1, callback_patience=0, verbose=0,
                show_network_summary=False,
                **self.public_options(use_generative_replay=True,
                                      generative_model_kwargs=dict(train_num=-1, samples_per_class=1)),
            )
        self.assertEqual(details["task_classes"], [[2, 0], [3, 1]])
        self.assertEqual(model.seen_classes, {0: 0, 1: 1, 2: 2, 3: 3})
        self.assertEqual(int(model.gen_optimizer.iterations), 2)
        self.assertEqual(int(model.clf_optimizer.iterations), 2)
        self.assertEqual(len(details["teacher_histories"]), 2)
        self.assertIsNone(model.current_teacher_network)
        for history in details["teacher_histories"]:
            self.assertIn("classifier_loss", history)
            self.assertIn("noise_loss", history)
            self.assert_finite(history.values())

    def test_public_train_bundle_supports_independent_progressive_teachers(self) -> None:
        """Grow current teachers and students through the public continual training adapter."""
        student = self.continual_model(DiffusionClassifier)
        inputs = tf.keras.Input((4, 4, 1))
        classifier = tf.keras.Model(inputs, tf.keras.layers.Dense(2, activation="softmax")(
            tf.keras.layers.Flatten()(inputs)))
        classifier.compile(optimizer="sgd", loss="sparse_categorical_crossentropy")
        bundle = dict(classifier_name="tiny", classifier=classifier, generative_model=student)
        teachers = []

        def factory(*args: object, **kwargs: object) -> tuple:
            """Retain trained expert wrappers for independent growth assertions."""
            teacher, output_ids = make_current_task_teacher(*args, **kwargs)
            teachers.append(teacher)
            return teacher, output_ids

        with tempfile.TemporaryDirectory() as temporary, \
             patch("common.learner.make_current_task_teacher", side_effect=factory), \
             redirect_stdout(io.StringIO()):
            history = train_model(
                model=bundle, trainset=self.continual_loader, task="continual",
                model_name="dit_classifier", dataset_name="mnist", preprocess="diffusion",
                results_path=temporary, save_config_=False, epochs=1, batch_size=8,
                seed=811, verbose=0, show_network_summary=False, show_images=False,
                save_gifs=False, save_weights=False, report_every_epoch=False,
                fit_method="fit_progressively",
                fit_kwargs=dict(stage_tasks=[("depth", "vision_transformer_block")],
                                stage_epochs=1, final_epochs=1, stages_verbose=False),
                continually_learn_kwargs=self.public_options(),
            )
        self.assertEqual(student.network.depth, 3)
        self.assertEqual([teacher.network.depth for teacher in teachers], [2, 3])
        self.assertEqual(int(student.optimizer.iterations), 4)
        self.assertEqual([int(teacher.optimizer.iterations) for teacher in teachers], [2, 2])
        self.assertEqual(len(history["continual_accuracy"]), 2)
        self.assertTrue(bundle["continual_details"]["dual_teacher_distillation"])
        self.assertIsNone(student.current_teacher_network)

    def test_public_existing_nonidentity_map_preserves_teacher_routes_and_accuracy(self) -> None:
        """Fit existing class columns and score perfect mapped outputs without relabeling."""
        student = self.continual_model(DiffusionClassifier)
        for label in (1, 0, 3, 2):
            student._check_new_labels(y=np.asarray([label]), verbose=0)
        expected_mapping = {1: 0, 0: 1, 3: 2, 2: 3}
        self.assertEqual(student.seen_classes, expected_mapping)
        original_attach = student.set_current_teacher_network
        attachments = []

        def attach(network: object, **kwargs: object) -> None:
            """Observe actual current-teacher maps at the public learner boundary."""
            # Only live attachments carry current-task output and eligibility mappings.
            if network is not None:
                attachments.append((list(kwargs["class_ids"]), list(kwargs["task_class_ids"])))
            original_attach(network, **kwargs)

        def perfect_predictions(model: object, inputs: np.ndarray, labels: np.ndarray,
                                *args: object, **kwargs: object) -> np.ndarray:
            """Return exact ground-truth scores in the student's existing column order."""
            del inputs, args, kwargs
            columns = model._map_classes(tf.constant(labels)).numpy()
            return np.eye(model.network.num_classes, dtype="float32")[columns]

        with patch.object(student, "set_current_teacher_network", side_effect=attach), \
             patch("common.learner._predict_diffusion_classes", side_effect=perfect_predictions), \
             redirect_stdout(io.StringIO()):
            details = continually_learn(
                generative_model=student, load_dataset_fn=self.continual_loader,
                load_dataset_fn_kwargs=dict(preprocess="diffusion"), seed=811,
                batch_size=8, epochs=1, callback_patience=0, verbose=0,
                show_network_summary=False,
                **self.public_options(class_order=[0, 1, 2, 3], use_generative_replay=True,
                                      generative_model_kwargs=dict(train_num=-1, samples_per_class=1)),
            )
        self.assertEqual(attachments, [([1, 0], [1, 0]), ([3, 2], [3, 2])])
        self.assertEqual(student.seen_classes, expected_mapping)
        self.assertEqual(int(student.optimizer.iterations), 2)
        for key in ("ordinary_accuracy_matrix", "validation_accuracy_matrix"):
            matrix = np.asarray(details[key])
            np.testing.assert_array_equal(matrix[np.isfinite(matrix)], np.ones(3))
        np.testing.assert_array_equal(details["accuracies"], [1., 1.])

    def test_zero_class_role_weights_disable_kd_and_public_learner_rejects_empty_objective(self) -> None:
        """Allow ordinary supervised training but reject requested KD with no active role."""
        model = DiffusionClassifier(
            network=self.make_network(num_classes=4, seed=811), seed=811,
            use_ema=False, test_steps=4, scheduler_name="clipped_cosine", p_uncond=0.,
            mask_by_nulls=False, mask_by_t_threshold=False,
            noise_loss_coef=0., clf_loss_coef=1., clf_distil_loss_coef=1.,
            previous_teacher_clf_loss_weight=0., current_teacher_clf_loss_weight=0.,
        )
        model.compile(optimizer=tf.keras.optimizers.SGD(.01), loss="mse",
                      run_eagerly=False, jit_compile=False)
        self.assertFalse(model.use_classifier_distil)
        self.assertIsNone(model.teacher_network)
        results = tf.function(model.train_step)((self.images, self.labels))
        self.assertGreater(float(results["classifier_loss"]), 0.)
        self.assertEqual(int(model.optimizer.iterations), 1)
        with patch("common.train.train_model", side_effect=AssertionError("invalid KD reached training")), \
             self.assertRaisesRegex(ValueError, "positive|active|objective"), \
             redirect_stdout(io.StringIO()):
            continually_learn(
                generative_model=model, load_dataset_fn=self.continual_loader,
                load_dataset_fn_kwargs=dict(preprocess="diffusion"), seed=811,
                batch_size=8, epochs=1, callback_patience=0, verbose=0,
                show_network_summary=False, **self.public_options(class_order=[0, 1, 2, 3]),
            )
        self.assertEqual(int(model.optimizer.iterations), 1)

    def test_disabled_role_objectives_only_invoke_their_active_teacher(self) -> None:
        """Train graph joint/separate phases with different noise and class teacher roles."""
        for wrapper_cls in (DiffusionClassifier, DiffusionClassifierV2):
            for noise_role in ("previous", "current"):
                with self.subTest(wrapper=wrapper_cls.__name__, noise_role=noise_role):
                    previous = self.make_network(num_classes=2, seed=811)
                    current = self.make_network(num_classes=2, seed=812)
                    weights = {
                        "previous_teacher_noise_loss_weight": float(noise_role == "previous"),
                        "current_teacher_noise_loss_weight": float(noise_role == "current"),
                        "previous_teacher_clf_loss_weight": float(noise_role == "current"),
                        "current_teacher_clf_loss_weight": float(noise_role == "previous"),
                    }
                    options = dict(network=self.make_network(num_classes=4, seed=811),
                                   teacher_network=previous, current_teacher_network=current,
                                   use_ema=False, test_steps=4, scheduler_name="clipped_cosine",
                                   p_uncond=0., mask_by_nulls=False, mask_by_t_threshold=False,
                                   noise_loss_coef=0., clf_loss_coef=0.,
                                   noise_distil_loss_coef=1., clf_distil_loss_coef=1., **weights)
                    # V1 shares a forward pass across disjoint noise/classifier row allocations.
                    if wrapper_cls is DiffusionClassifier:
                        options["clf_train_batch_fraction"] = .5
                    model = wrapper_cls(**options)
                    model.set_current_teacher_network(current, class_ids=[2, 3])
                    model.compile(optimizer=tf.keras.optimizers.SGD(.001), loss="mse",
                                  run_eagerly=False, jit_compile=False)
                    before = [teacher.get_weights() for teacher in (previous, current)]
                    original_noise = model._predict_teacher_noise
                    original_class = model._predict_single_teacher_labels
                    noise_calls, class_calls = [], []

                    def predict_noise(*args: object, **kwargs: object) -> tf.Tensor:
                        """Record only actual active noise-target inference."""
                        noise_calls.append(kwargs.get("teacher_network"))
                        return original_noise(*args, **kwargs)

                    def predict_class(*args: object, **kwargs: object) -> tf.Tensor:
                        """Record only actual active classification-target inference."""
                        class_calls.append(kwargs.get("teacher_network"))
                        return original_class(*args, **kwargs)

                    parts = ("generator", "discriminator") if wrapper_cls is DiffusionClassifierV2 else (None,)
                    for part in parts:
                        # V2 selects a different tuple contract for each optimizer phase.
                        if part is not None:
                            model._switch_train_part(part)
                            model._test_part = part
                        model._preprocess_training = True
                        with patch.object(model, "_predict_teacher_noise", side_effect=predict_noise), \
                             patch.object(model, "_predict_single_teacher_labels", side_effect=predict_class):
                            mapped = model.prep_inputs_map(self.images, self.labels)
                        model._preprocess_training = None
                        allocation = tf.constant([False, False, True, True]) \
                            if noise_role == "previous" else tf.constant([True, True, False, False])
                        with patch.object(model, "_classifier_batch_mask", return_value=allocation):
                            results = tf.function(model.train_step)(mapped)
                        self.assert_finite(results.values())
                    expected_noise = previous if noise_role == "previous" else current
                    expected_class = current if noise_role == "previous" else previous
                    self.assertTrue(noise_calls)
                    self.assertTrue(class_calls)
                    self.assertTrue(all(teacher is expected_noise for teacher in noise_calls))
                    self.assertTrue(all(teacher is expected_class for teacher in class_calls))
                    for teacher, values in zip((previous, current), before):
                        for old, new in zip(values, teacher.get_weights()):
                            np.testing.assert_array_equal(old, new)

    def test_replay_only_scope_does_not_remove_current_real_teacher_rows(self) -> None:
        """Apply historical replay filtering to the previous role while teaching new real rows."""
        model = self.classifier(clf_distil_scope="replay_only")
        student = tf.fill((4, 4), .25)
        targets = (tf.constant([[.9, .1]] * 4), tf.constant([[.1, .9]] * 4))
        loss, _ = model.compute_clf_distil_loss(
            targets, student, classes=self.labels,
            replay_mask=tf.constant([True, True, False, False]),
            clf_distil_type="hard", update_teacher_metrics=True,
        )
        np.testing.assert_allclose(loss, 2. * np.log(4.), rtol=1e-6)
        self.assertEqual(float(model.previous_teacher_clf_loss_tracker.count), 2.)
        self.assertEqual(float(model.current_teacher_clf_loss_tracker.count), 2.)

    def test_rejected_teacher_replacement_preserves_the_active_runtime(self) -> None:
        """Reject a native noise-only class teacher without freezing or replacing anything."""
        for role in ("previous", "current"):
            with self.subTest(role=role):
                model = self.classifier()
                candidate = self.noise_network()
                original_previous = model.teacher_network
                original_current = model.current_teacher_network
                original_maps = (model.current_teacher_class_ids, model.current_teacher_task_class_ids)
                original_flags = (model.map_preprocess, model.use_classifier_distil,
                                  model.use_noise_distil_loss, model.defer_teacher)
                sentinel = object()
                model.train_function = sentinel
                self.assertTrue(candidate.trainable)
                with self.assertRaises(ValueError):
                    # The current setter additionally owns local-to-student class maps.
                    if role == "current":
                        model.set_current_teacher_network(candidate, class_ids=[2, 3])
                    # The previous setter must also validate before changing base state.
                    else:
                        model.set_teacher_network(candidate)
                self.assertIs(model.teacher_network, original_previous)
                self.assertIs(model.current_teacher_network, original_current)
                self.assertEqual((model.current_teacher_class_ids, model.current_teacher_task_class_ids),
                                 original_maps)
                self.assertEqual((model.map_preprocess, model.use_classifier_distil,
                                  model.use_noise_distil_loss, model.defer_teacher), original_flags)
                self.assertIs(model.train_function, sentinel)
                self.assertTrue(candidate.trainable)


    def test_deferred_soft_teacher_rejection_preserves_student_and_candidate(self) -> None:
        """Reject missing student logits before attaching or freezing a deferred teacher."""
        for role in ("previous", "current"):
            with self.subTest(role=role):
                model = self.classifier(previous=False, defer_teacher=True, clf_distil_type="soft")
                model.set_current_teacher_network(None)
                candidate = self.make_network(num_classes=2)
                original_maps = (model.current_teacher_class_ids, model.current_teacher_task_class_ids)
                original_flags = (model.map_preprocess, model.use_classifier_distil,
                                  model.use_noise_distil_loss, model.defer_teacher)
                sentinel = object()
                model.train_function = sentinel

                def probability_only(inputs: tf.Tensor, training: bool = False,
                                     full_return: bool = False) -> tf.Tensor:
                    """Stand in for a custom student's unsupported probability-only call."""
                    raise AssertionError("Validation must not execute the student.")

                with patch.object(model.network, "call", probability_only), \
                     self.assertRaisesRegex(ValueError, "return_logits"):
                    # Each public setter must reject before reaching any attachment mutation.
                    if role == "current":
                        model.set_current_teacher_network(candidate, class_ids=[2, 3])
                    # The previous role uses the same pure student-capability check.
                    else:
                        model.set_teacher_network(candidate)
                self.assertIsNone(model.teacher_network)
                self.assertIsNone(model.current_teacher_network)
                self.assertEqual((model.current_teacher_class_ids, model.current_teacher_task_class_ids),
                                 original_maps)
                self.assertEqual((model.map_preprocess, model.use_classifier_distil,
                                  model.use_noise_distil_loss, model.defer_teacher), original_flags)
                self.assertIs(model.train_function, sentinel)
                self.assertTrue(candidate.trainable)


# Permit focused execution without running tests at module import.
if __name__ == "__main__":
    unittest.main()
