"""Independent previous/current teacher targets and continual task isolation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.learner import _run_continual_tasks
from common.runtime import configure_runtime
from common.tests import test_callable_teachers as callable_fixtures
from common.tests import test_clean_classifier_training as classifier_fixtures
from common.tests import test_fit_teacher as fit_fixtures
from common.train import train_model
from diffusion import (
    DiffusionClassifier, DiffusionClassifierV2, DiffusionModel, DiffusionTransformer
)


class DualTeacherTests(unittest.TestCase):
    """Exercise separate losses, vocabularies, teacher freezing, and task data."""

    tearDown = classifier_fixtures.CleanClassifierTrainingTests.tearDown
    make_network = classifier_fixtures.CleanClassifierTrainingTests.make_network
    dataset = callable_fixtures.CallableTeacherTests.dataset
    assert_finite = callable_fixtures.CallableTeacherTests.assert_finite

    def setUp(self) -> None:
        """Use deterministic four-class images and preserve the precision policy."""

        classifier_fixtures.CleanClassifierTrainingTests.setUp(self)
        self.labels = tf.constant([0, 1, 2, 3], tf.int32)

    def classifier(
        self, wrapper_cls: type = DiffusionClassifier, previous: bool = True, 
        **overrides: object
    ) -> DiffusionClassifier:
        """Attach two image-only teachers with disjoint two-class vocabularies."""

        old = callable_fixtures._ImageTeacher(self.images) if previous else None
        current = callable_fixtures._ImageTeacher(self.images)
        # Build previous teacher variables before checking independent freezing.
        if old is not None:
            old(self.images, training=False)
        current(self.images, training=False)
        current.head.bias.assign([-1., 1.])
        options = dict(
            network=self.make_network(num_classes=4), teacher_network=old, 
            current_teacher_network=current, use_ema=False, seed=811, 
            scheduler_name="clipped_cosine", test_steps=4, p_uncond=0., 
            mask_by_nulls=False, mask_by_t_threshold=False, 
            clf_loss_coef=0., noise_loss_coef=0., clf_distil_loss_coef=1., 
            clf_distil_type="soft", clf_distil_temperature=1., 
            clf_train_noisified_max_timesteps=-1, 
            clf_test_noisified_max_timesteps=-1
        )
        options.update(overrides)
        model = wrapper_cls(**options)
        model.set_current_teacher_network(current, class_ids=[2, 3], task_class_ids=[2, 3])
        model.compile(optimizer=tf.keras.optimizers.SGD(.01), loss="mse", 
                      run_eagerly=False, jit_compile=False)
        return model

    @staticmethod
    def noise_network(num_classes: int | None = 2) -> DiffusionTransformer:
        """Construct a minimal native epsilon network with optional dynamic classes."""

        return DiffusionTransformer(
            image_size=4, channels=1, patch_size=2, dim=4, depth=1, 
            mha_num_heads=1, vit_block_mlp_ratio=1., num_classes=num_classes, 
            timesteps=8, use_cfg=True, seed=811
        )

    def noise_model(self, **overrides: object) -> DiffusionModel:
        """Attach old and new native epsilon teachers to a four-class student."""

        options = dict(
            network=self.noise_network(4), teacher_network=self.noise_network(), 
            current_teacher_network=self.noise_network(), use_ema=False, 
            seed=811, scheduler_name="clipped_cosine", test_steps=4, p_uncond=0., 
            noise_loss_coef=0., noise_distil_loss_coef=1.
        )
        options.update(overrides)
        model = DiffusionModel(**options)
        model.set_current_teacher_network(model.current_teacher_network, class_ids=[2, 3])
        model.compile(optimizer=tf.keras.optimizers.SGD(.001), loss="mse", 
                      run_eagerly=False, jit_compile=False)
        return model

    def test_classification_losses_keep_disagreeing_teachers_independent(self) -> None:
        """Match separate weighted hard CE and soft KL on full student support."""

        student = tf.constant([
            [.1, .2, .3, .4], [.4, .3, .2, .1], 
            [.25, .25, .1, .4], [.1, .1, .7, .1]
        ])
        targets = (tf.constant([[.8, .2]] * 4), tf.constant([[.25, .75]] * 4))
        model = self.classifier(previous_teacher_clf_loss_weight=2., 
                                current_teacher_clf_loss_weight=3.)
        for scope in ("task", "all"):
            model.dual_teacher_scope = scope
            for kind in ("hard", "soft"):
                with self.subTest(scope=scope, kind=kind):
                    actual, returned = model.compute_clf_distil_loss(
                        targets, student, classes=self.labels, clf_distil_type=kind, 
                        clf_distil_temperature=2., student_logits=tf.math.log(student)
                    )
                    expected = 0.
                    for role, (target, columns, weight) in enumerate(zip(
                        targets, ([0, 1], [2, 3]), (2., 3.)
                    )):
                        rows = np.arange(4) if scope == "all" else np.array(columns)
                        # Hard KD retains each teacher's own argmax and class mapping.
                        if kind == "hard":
                            target_class = columns[int(tf.argmax(target[0]))]
                            row_losses = -np.log(student.numpy()[:, target_class])
                        # Soft KD retains separate entropy terms and the full denominator.
                        else:
                            q = tf.nn.softmax(tf.math.log(target) / 2.).numpy()
                            log_p = tf.nn.log_softmax(tf.math.log(student) / 2.).numpy()
                            row_losses = np.sum(q * (np.log(q) - log_p[:, columns]), axis=-1) * 4.
                        expected += weight * float(np.mean(row_losses[rows]))
                    np.testing.assert_allclose(actual, expected, rtol=1e-5)
                    self.assertIs(returned, student)

    def test_noise_losses_are_separately_masked_and_weighted(self) -> None:
        """Each role contributes its own masked MSE rather than an averaged target."""

        model = self.noise_model(previous_teacher_noise_loss_weight=2., 
                                 current_teacher_noise_loss_weight=3.)
        old = tf.ones_like(self.images) * 2.
        current = tf.ones_like(self.images) * 5.
        student = tf.ones_like(self.images)
        masks = (tf.constant([True, True, False, False]), 
                 tf.constant([False, False, True, True]))
        actual = model.compute_distil_noise_loss((old, current), student, 
                                                teacher_noise_mask=masks)
        np.testing.assert_allclose(actual, 50., rtol=1e-6)
        empty_current = (masks[0], tf.zeros_like(masks[1]))
        actual = model.compute_distil_noise_loss((old, current), student, 
                                                teacher_noise_mask=empty_current)
        np.testing.assert_allclose(actual, 2., rtol=1e-6)

    def test_noise_scope_uses_true_classes_and_preserves_cfg_null_labels(self) -> None:
        """Dropped conditions remain null while task eligibility follows original rows."""

        model = self.noise_model(p_uncond=1., train_cfg_scale=2.)
        model._preprocess_training = True
        calls = []
        original_call = model.current_teacher_network.call

        def record(inputs: tuple, **kwargs: object) -> object:
            """Observe local current-teacher label IDs before native inference."""

            calls.append(inputs[2].numpy().copy())
            return original_call(inputs, **kwargs)

        with patch.object(model.current_teacher_network, "call", side_effect=record):
            mapped = model.prep_inputs_map(self.images, self.labels)
        masks = mapped[-1]
        self.assertIsInstance(mapped[-2], tuple)
        np.testing.assert_array_equal(masks[0], [True, True, False, False])
        np.testing.assert_array_equal(masks[1], [False, False, True, True])
        self.assertGreaterEqual(len(calls), 2)
        for labels in calls:
            np.testing.assert_array_equal(labels, [0, 0, 0, 0])
        model.p_uncond = 0.
        calls.clear()
        with patch.object(model.current_teacher_network, "call", side_effect=record):
            model.prep_inputs_map(self.images, self.labels)
        np.testing.assert_array_equal(calls[0], [0, 0, 1, 2])

    def test_unconditional_noise_teachers_exclude_unknown_labels_in_all_scope(self) -> None:
        """Without a CFG null ID, unknown current conditions cannot teach local class zero."""

        networks = [DiffusionTransformer.from_config({
            **self.noise_network(width).get_config(), "use_cfg": False
        }) for width in (4, 2, 2)]
        model = self.noise_model(network=networks[0], teacher_network=networks[1], 
                                 current_teacher_network=networks[2], dual_teacher_scope="all")
        mapped = model.prep_inputs_map(self.images, self.labels)
        np.testing.assert_array_equal(mapped[-1][0], [True, True, False, False])
        np.testing.assert_array_equal(mapped[-1][1], [False, False, True, True])

    def test_full_width_teachers_exclude_columns_they_have_not_been_taught(self) -> None:
        """Fixed and copied full heads retain only their explicitly taught support."""

        model = self.classifier()
        previous = self.make_network(num_classes=4)
        current = self.make_network(num_classes=4)
        object.__setattr__(previous, "_diffusion_task_class_ids", (0, 1))
        model.set_teacher_network(previous)
        model.set_current_teacher_network(current, class_ids=[0, 1, 2, 3], 
                                          task_class_ids=[2, 3])
        student = tf.constant([[.1, .2, .3, .4]] * 4)
        targets = (tf.constant([[.08, .02, .8, .1]] * 4), 
                   tf.constant([[.7, .1, .05, .15]] * 4))
        actual, _ = model.compute_clf_distil_loss(
            targets, student, classes=self.labels, clf_distil_type="hard"
        )
        np.testing.assert_allclose(actual, -np.log(.1) - np.log(.4), rtol=1e-6)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            model.compute_clf_distil_loss(
                (targets[0], tf.constant([[.5, .5, 0., 0.]] * 4)), student, 
                classes=self.labels, clf_distil_type="hard"
            )
        noise_model = self.noise_model(teacher_network=self.noise_network(4), p_uncond=1.)
        object.__setattr__(noise_model.teacher_network, "_diffusion_task_class_ids", (0, 1))
        mapped = noise_model.prep_inputs_map(self.images, self.labels)
        np.testing.assert_array_equal(mapped[-1][0], [True, True, False, False])
        np.testing.assert_array_equal(mapped[-1][1], [False, False, True, True])

    def test_reported_teacher_losses_are_invariant_to_batch_partition(self) -> None:
        """Report weighted role means consistently across uneven, disjoint batches."""

        classifier = self.classifier(previous_teacher_clf_loss_weight=2., 
                                     current_teacher_clf_loss_weight=3.)
        noise_model = self.noise_model(previous_teacher_noise_loss_weight=2., 
                                       current_teacher_noise_loss_weight=3.)
        probabilities = tf.constant([
            [.1, .2, .3, .4], [.4, .3, .2, .1], 
            [.25, .25, .1, .4], [.1, .1, .7, .1]
        ])
        scores = (tf.constant([[.8, .2]] * 4), tf.constant([[.25, .75]] * 4))
        epsilon = tuple(tf.broadcast_to(tf.constant(values, tf.float32)[:, None, None, None], 
                                       self.images.shape)
                        for values in ([1., 3., 7., 8.], [9., 8., 2., 4.]))
        masks = (tf.constant([True, True, False, False]), 
                 tf.constant([False, False, True, True]))
        summaries = []
        for partitions in (tuple([[0, 1, 2, 3]]), ([0], [1, 2, 3])):
            classifier.reset_metrics()
            noise_model.reset_metrics()
            for rows in partitions:
                classes = tf.gather(self.labels, rows)
                student = tf.gather(probabilities, rows)
                clf_loss, _ = classifier.compute_clf_distil_loss(
                    tuple(tf.gather(target, rows) for target in scores), student, 
                    classes=classes, clf_distil_type="hard", update_teacher_metrics=True
                )
                clf_results = classifier.get_clf_results_dict(
                    tf.constant(0.), classes, student, clf_distil_loss=clf_loss, 
                    distil_classes=student, use_total_loss=False, 
                    use_kl_loss=False, use_ctr_loss=False
                )
                noise_loss = noise_model.compute_distil_noise_loss(
                    tuple(tf.gather(target, rows) for target in epsilon), 
                    tf.zeros_like(tf.gather(self.images, rows)), 
                    teacher_noise_mask=tuple(tf.gather(mask, rows) for mask in masks), 
                    update_teacher_metrics=True
                )
                noise_results = noise_model.get_results_dict(
                    tf.constant(0.), noise_distil_loss=noise_loss, classes=classes, 
                    use_total_loss=False, use_noise_distil_loss=True, 
                    use_image_loss=False, use_kl_loss=False, use_ctr_loss=False
                )
            summaries.append({
                **{name: float(clf_results[name]) for name in (
                    "previous_teacher_clf_loss", "current_teacher_clf_loss", "clf_distil_loss")}, 
                **{name: float(noise_results[name]) for name in (
                    "previous_teacher_noise_distil_loss", "current_teacher_noise_distil_loss", 
                    "noise_distil_loss")}
            })
        for name, expected in summaries[0].items():
            np.testing.assert_allclose(summaries[1][name], expected, rtol=1e-6)
        np.testing.assert_allclose(summaries[0]["noise_distil_loss"], 40., rtol=1e-6)
        classifier.set_current_teacher_network(classifier.current_teacher_network, class_ids=[2, 3])
        self.assertEqual(float(classifier.previous_teacher_clf_loss_tracker.count), 0.)
        self.assertEqual(float(classifier.current_teacher_clf_loss_tracker.count), 0.)

    def test_both_image_teachers_keep_clean_inputs_during_noisy_student_fit(self) -> None:
        """Train V1/V2 students without changing either image teacher's weights."""

        for wrapper_cls in (DiffusionClassifier, DiffusionClassifierV2):
            with self.subTest(wrapper=wrapper_cls.__name__):
                model = self.classifier(wrapper_cls)
                teachers = (model.teacher_network, model.current_teacher_network)
                before = [teacher.get_weights() for teacher in teachers]
                student_before = model.network.get_weights()
                # Select the V2 classification path for both preprocessing phases.
                if wrapper_cls is DiffusionClassifierV2:
                    model._switch_train_part("discriminator")
                    model._test_part = "discriminator"
                model._preprocess_training = True
                mapped = model.prep_inputs_map(self.images, self.labels)
                model._preprocess_training = None
                self.assertIsInstance(mapped[-1], tuple)
                self.assertEqual(len(mapped[-1]), 2)
                noisy_student = mapped[1] if wrapper_cls is DiffusionClassifierV2 else mapped[7]
                self.assertGreater(float(tf.reduce_max(tf.abs(noisy_student - self.images))), 0.)
                fit = model.fit_discriminator if wrapper_cls is DiffusionClassifierV2 else model.fit
                history = fit(x=self.dataset(), validation_data=self.dataset(), epochs=1, verbose=0)
                self.assert_finite(history.history.values())
                self.assertGreater(history.history["clf_distil_loss"][0], 0.)
                self.assertTrue(any(not np.array_equal(old, new) for old, new in
                                    zip(student_before, model.network.get_weights())))
                for teacher, weights in zip(teachers, before):
                    self.assertFalse(teacher.trainable)
                    self.assertFalse({id(v) for v in teacher.weights}
                                     & {id(v) for v in model.weights})
                    for old, new in zip(weights, teacher.get_weights()):
                        np.testing.assert_array_equal(old, new)

    def test_current_only_teacher_is_legal_without_deferred_previous_teacher(self) -> None:
        """The first task can distill a current expert before any previous snapshot."""

        model = self.classifier(previous=False)
        self.assertIsNone(model.teacher_network)
        self.assertTrue(model.use_classifier_distil)
        mapped = model.prep_inputs_map(self.images, self.labels)
        self.assertIsInstance(mapped[-1], tuple)
        self.assertEqual(len(mapped[-1]), 1)
        results = tf.function(model.train_step)(mapped)
        self.assertGreater(float(results["clf_distil_loss"]), 0.)
        self.assert_finite(results.values())

    def test_dual_options_round_trip_and_reject_invalid_values(self) -> None:
        """Serialize independent role weights and reject unsupported dual scopes."""

        options = dict(dual_teacher_scope="all", previous_teacher_noise_loss_weight=.5, 
                       current_teacher_noise_loss_weight=2., previous_teacher_clf_loss_weight=3., 
                       current_teacher_clf_loss_weight=4.)
        model = self.classifier(defer_teacher=True, **options)
        config = model.get_config()
        self.assertNotIn("current_teacher_network", config)
        clone = DiffusionClassifier.from_config(config)
        for name, expected in options.items():
            self.assertEqual(getattr(clone, name), expected)
        for invalid in (-1., float("nan"), float("inf")):
            with self.subTest(weight=invalid):
                with self.assertRaises((ValueError, AssertionError)):
                    self.classifier(current_teacher_clf_loss_weight=invalid)
        with self.assertRaises((ValueError, AssertionError)):
            self.classifier(dual_teacher_scope="unknown")

    @staticmethod
    def continual_loader(indices: list[int], **kwargs: object) -> tuple:
        """Return exact class-coded real train, validation, and test rows."""

        return fit_fixtures.FitTeacherTests.continual_loader(indices, **kwargs)

    def continual_model(self, wrapper_cls: type) -> DiffusionModel:
        """Compile a dynamic student with deferred noise and optional classifier KD."""

        classifier = wrapper_cls is not DiffusionModel
        network = self.make_network(num_classes=None, seed=811) if classifier else self.noise_network(None)
        options = dict(network=network, use_ema=False, seed=811, test_steps=2, 
                       scheduler_name="clipped_cosine", p_uncond=0., defer_teacher=True, 
                       noise_distil_loss_coef=.1)
        # Classifier students additionally learn from each task's clean class targets.
        if classifier:
            options.update(clf_loss_coef=1., clf_distil_loss_coef=.1, 
                           mask_by_nulls=False, mask_by_t_threshold=False)
        model = wrapper_cls(**options)
        model.compile(optimizer=tf.keras.optimizers.SGD(.01), loss="mse", 
                      run_eagerly=True, jit_compile=False)
        return model

    def continual_options(self, model: DiffusionModel, **overrides: object) -> dict:
        """Use two finite tasks with one optimizer update per training phase."""

        options = dict(
            class_num=4, task_size=2, load_dataset_fn=self.continual_loader, 
            load_dataset_fn_kwargs={"preprocess": "diffusion"}, 
            generative_model=model, use_generative_model_classifier=isinstance(model, DiffusionClassifier), 
            generative_model_kwargs={"train_num": -1, "samples_per_class": 1}, 
            use_generative_replay=True, use_distillation=True, dual_teacher_distillation=True, 
            current_teacher_init="fresh", remove_prev_classes=False, 
            batch_size=8, epochs=1, optimizer_steps_per_epoch=1, callback_patience=0, 
            plot_results=False, verbose=0, seed=811, 
            show_generated_images=False, show_network_summary=False
        )
        options.update(overrides)
        return options

    @staticmethod
    def external_classifier(path: Path) -> None:
        """Save the small standalone classifier required by a noise-only learner."""

        inputs = tf.keras.Input((4, 4, 1))
        outputs = tf.keras.layers.Dense(2, activation="softmax")(
            tf.keras.layers.Flatten()(inputs))
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.save(path)

    def test_continual_current_teacher_receives_only_new_real_rows(self) -> None:
        """Isolate each fresh expert's data while retaining both frozen roles for students."""

        for wrapper_cls in (DiffusionModel, DiffusionClassifier, DiffusionClassifierV2):
            with self.subTest(wrapper=wrapper_cls.__name__), tempfile.TemporaryDirectory() as temporary:
                student = self.continual_model(wrapper_cls)
                current_models = []
                teacher_training = []
                student_training = []
                from common.current_task_teacher import make_current_task_teacher


                def factory(*args: object, **kwargs: object) -> tuple:
                    """Track each newly allocated expert without changing its training."""

                    teacher, output_ids = make_current_task_teacher(*args, **kwargs)
                    current_models.append((teacher, teacher.network.get_weights()))
                    return teacher, output_ids

                def train(*args: object, **kwargs: object) -> dict:
                    """Inspect actual phase datasets and assert frozen-role boundaries."""

                    model, dataset = args[1:3]
                    role_models = [entry[0] for entry in current_models]
                    is_current = any(model is candidate for candidate in role_models)
                    before_student = student.network.get_weights()
                    attached = [teacher for teacher in (
                        student.teacher_network, student.current_teacher_network
                    ) if teacher is not None]
                    frozen = [teacher.get_weights() for teacher in attached]
                    # Current experts must see only original rows for their own new labels.
                    if is_current:
                        task_index = next(i for i, candidate in enumerate(role_models) if candidate is model)
                        expected = {2 * task_index, 2 * task_index + 1}
                        for split in (dataset, kwargs.get("valset")):
                            self.assertIsNotNone(split)
                            observed = set()
                            for batch in split.as_numpy_iterator():
                                images, labels = batch[:2]
                                observed.update(labels.tolist())
                                expected_images = np.broadcast_to(
                                    (labels.astype("float32") / 2. - .75)[:, None, None, None], images.shape)
                                np.testing.assert_array_equal(images, expected_images)
                            self.assertEqual(observed, expected)
                        teacher_training.append(task_index)
                    # Student phases must retain the current expert and an old snapshot after task one.
                    elif model is student:
                        self.assertIsNotNone(student.current_teacher_network)
                        self.assertEqual(len(attached), 1 if len(current_models) == 1 else 2)
                        student_training.append(len(current_models) - 1)
                    result = train_model(*args, **kwargs)
                    for teacher, old_weights in zip(attached, frozen):
                        for old, new in zip(old_weights, teacher.get_weights()):
                            np.testing.assert_array_equal(old, new)
                    # Training an independent expert cannot mutate student parameters.
                    if is_current:
                        for old, new in zip(before_student, student.network.get_weights()):
                            np.testing.assert_array_equal(old, new)
                    return result

                options = self.continual_options(student)
                # Noise-only diffusion keeps the learner's ordinary external classifier.
                if wrapper_cls is DiffusionModel:
                    template = Path(temporary) / "template.keras"
                    self.external_classifier(template)
                    options.update(tuned_model_path=str(template), compile_args=dict(
                        optimizer=tf.keras.optimizers.SGD(.01), 
                        loss="sparse_categorical_crossentropy", metrics=["accuracy"]))
                with patch("common.learner.make_current_task_teacher", side_effect=factory), \
                     patch("common.train.train_model", side_effect=train):
                    details = _run_continual_tasks(**options)
                self.assertEqual(len(current_models), 2)
                self.assertEqual(set(teacher_training), {0, 1})
                self.assertEqual(set(student_training), {0, 1})
                self.assertIsNot(current_models[0][0].network, current_models[1][0].network)
                for model, initial in current_models:
                    self.assertTrue(any(not np.array_equal(old, new) for old, new in
                                        zip(initial, model.network.get_weights())))
                self.assertEqual(len(details["teacher_histories"]), 2)
                self.assertTrue(all(history is not None for history in details["teacher_histories"]))
                self.assertIsNone(student.current_teacher_network)

    def test_current_teacher_initialization_keeps_independent_parameters(self) -> None:
        """Fresh experts use local heads while student initialization copies all weights."""

        from common.current_task_teacher import make_current_task_teacher


        student = self.classifier()
        student.network.trainable_variables[0].assign(
            tf.ones_like(student.network.trainable_variables[0]) * .123)
        initial = student.network.get_weights()
        fresh, fresh_ids = make_current_task_teacher(
            student, [2, 3], initialization="fresh", seed=73
        )
        copied, copied_ids = make_current_task_teacher(
            student, [2, 3], initialization="student", seed=79
        )
        self.assertEqual(fresh_ids, [2, 3])
        self.assertEqual(fresh.network.num_classes, 2)
        self.assertEqual(fresh.seen_classes, {2: 0, 3: 1})
        self.assertEqual(copied_ids, [0, 1, 2, 3])
        self.assertEqual(copied.network.num_classes, 4)
        for old, new in zip(initial, copied.network.get_weights()):
            np.testing.assert_array_equal(old, new)
        for teacher in (fresh, copied):
            self.assertIsNot(teacher.optimizer, student.optimizer)
            self.assertFalse({id(v) for v in teacher.network.weights}
                             & {id(v) for v in student.network.weights})
        for old, new in zip(initial, student.network.get_weights()):
            np.testing.assert_array_equal(old, new)

    def test_task_boundary_recovery_rebuilds_the_next_current_teacher(self) -> None:
        """Use deterministic kernels to reproduce the resumed student and snapshot exactly."""

        configure_runtime(811, "float32", deterministic_ops=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = []
            for resume in (False, True):
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(811)
                model = self.continual_model(DiffusionClassifier)
                options = self.continual_options(
                    model, use_generative_replay=False, remove_prev_classes=True, 
                    save_task_checkpoints=True, deterministic_ops=True, 
                    checkpoint_dir=str(root / ("resumed" if resume else "original"))
                )
                # Restore only a completed task; its disposable current expert is rebuilt.
                if resume:
                    options["resume_from"] = str(root / "original" / "task-0000")
                _run_continual_tasks(**options)
                models.append(model)
            original, resumed = models
            self.assertIsNone(resumed.current_teacher_network)
            self.assertFalse(resumed.teacher_network.trainable)
            for expected, actual in ((original.network, resumed.network), 
                                     (original.teacher_network, resumed.teacher_network)):
                self.assertEqual(len(expected.weights), len(actual.weights))
                for old, new in zip(expected.get_weights(), actual.get_weights()):
                    np.testing.assert_allclose(new, old, rtol=0., atol=0.)
            self.assertEqual(int(original.optimizer.iterations), int(resumed.optimizer.iterations))

    def test_learner_rejects_incompatible_dual_teacher_modes(self) -> None:
        """Reject disabled KD, old trainable-teacher mode, and mid-task checkpoints."""

        for settings in (dict(use_distillation=False), dict(current_teacher_init="unknown")):
            with self.subTest(settings=settings):
                model = self.continual_model(DiffusionClassifier)
                with self.assertRaises(ValueError):
                    _run_continual_tasks(**self.continual_options(model, **settings))
        for attribute, value in (("trainable_teacher", True), ("checkpoint_interval", 1)):
            with self.subTest(attribute=attribute):
                model = self.continual_model(DiffusionClassifier)
                setattr(model, attribute, value)
                with self.assertRaises(ValueError):
                    _run_continual_tasks(**self.continual_options(model))


# Run focused regressions directly without executing them on import.
if __name__ == "__main__":
    unittest.main()
