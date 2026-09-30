"""Independent classifier/noise teachers with a previous joint student snapshot."""

from contextlib import redirect_stdout
import inspect
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.learner import continually_learn
from common.runtime import configure_runtime
from common.tests import test_clean_classifier_training as classifier_fixtures
from common.tests import test_dual_teachers as dual_fixtures
from common.train import train_model
from diffusion import DiffusionClassifier, DiffusionClassifierV2, DiffusionModel


class HeadTeacherTests(unittest.TestCase):
    """Keep separately trained current experts independent in training and KD."""

    make_network = classifier_fixtures.CleanClassifierTrainingTests.make_network
    noise_network = staticmethod(dual_fixtures.DualTeacherTests.noise_network)
    tearDown = classifier_fixtures.CleanClassifierTrainingTests.tearDown

    def setUp(self) -> None:
        """Use tiny deterministic raw pixels and preserve the precision policy."""

        classifier_fixtures.CleanClassifierTrainingTests.setUp(self)
        self.images = (self.images + 1.) * 127.5

    def make_classifier(self) -> tf.keras.Model:
        """Create an ordinary Keras teacher with frozen backbone and BN state."""

        inputs = tf.keras.Input((4, 4, 1))
        hidden = tf.keras.layers.Rescaling(1. / 255.)(inputs)
        hidden = tf.keras.layers.Conv2D(2, 1, trainable=False, name="early")(hidden)
        hidden = tf.keras.layers.BatchNormalization(trainable=False, name="normalization")(hidden)
        hidden = tf.keras.layers.Conv2D(2, 1, name="tail")(hidden)
        base = tf.keras.Model(inputs, hidden, name="base")
        model = tf.keras.Sequential([
            tf.keras.Input((4, 4, 1)), base, 
            tf.keras.layers.GlobalAveragePooling2D(), 
            tf.keras.layers.Dense(2, activation="softmax", name="head")
        ], name="image_teacher")
        model.compile(optimizer=tf.keras.optimizers.Adam(.003), 
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"], 
                      run_eagerly=False, jit_compile=False)
        return model

    def make_model(
        self, wrapper_cls: type = DiffusionClassifier, num_classes: int | None = 2, 
        run_eagerly: bool = True, compile_model: bool = True, **overrides: object
    ) -> DiffusionClassifier:
        """Attach a plain image classifier and a separate noise-only native DiT."""

        options = dict(
            network=self.make_network(num_classes=num_classes, seed=811), 
            classifier_teacher_network=self.make_classifier(), 
            noise_teacher_network=self.noise_network(None if num_classes is None else 2), 
            trainable_teacher=True, teacher_dynamic_classes=num_classes is None, 
            teacher_training="each_task", use_ema=False, preprocess_type="standardize", 
            scheduler_name="clipped_cosine", test_steps=4, 
            p_uncond=0., mask_by_nulls=False, mask_by_t_threshold=False, noise_loss_coef=0., 
            clf_loss_coef=0., noise_distil_loss_coef=1., clf_distil_loss_coef=1., 
            clf_distil_type="soft", clf_distil_temperature=1., seed=811
        )
        options.update(overrides)
        model = wrapper_cls(**options)
        # Uncompiled students test the public independent teacher compile path.
        if compile_model:
            model.compile(optimizer=tf.keras.optimizers.SGD(.001), loss="mse", 
                          run_eagerly=run_eagerly, jit_compile=False)
        return model

    def dataset(self, labels: object | None = None) -> tf.data.Dataset:
        """Return one finite raw-image batch using one private data thread."""

        labels = self.labels if labels is None else tf.convert_to_tensor(labels, tf.int32)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        return tf.data.Dataset.from_tensor_slices((self.images, labels)).batch(4).with_options(options)

    def assert_weights_equal(self, model: tf.keras.Model, before: list) -> None:
        """Require exact preservation of every source model weight."""

        self.assertEqual(len(model.get_weights()), len(before))
        for expected, actual in zip(before, model.get_weights()):
            np.testing.assert_array_equal(actual, expected)

    def test_base_wrapper_exposes_only_native_noise_teacher_roles(self) -> None:
        """Keep image-classifier ownership out of the noise-only public wrapper."""

        parameters = inspect.signature(DiffusionModel.__init__).parameters
        self.assertNotIn("classifier_teacher_network", parameters)
        self.assertNotIn("teacher_dynamic_classes", parameters)
        self.assertFalse(hasattr(DiffusionModel, "set_classifier_teacher_network"))
        model = DiffusionModel(
            network=self.noise_network(), use_ema=False, test_steps=4
        )
        self.assertEqual(model.get_teacher_names(), ("previous", "current", "noise"))
        for attribute in (
            "classifier_teacher_network", "classifier_teacher_class_ids", 
            "classifier_teacher_task_class_ids", "_classifier_teacher_model", 
            "_classifier_keras_teacher_fit_state", "teacher_dynamic_classes"
        ):
            with self.subTest(attribute=attribute):
                self.assertFalse(hasattr(model, attribute))
                self.assertNotIn(attribute, model.get_config())
        with self.assertRaises(ValueError):
            model.get_teacher_network("classifier")
        with self.assertRaises(ValueError):
            model.get_teacher_model("classifier")
        with self.assertRaises((TypeError, ValueError)):
            DiffusionModel(
                network=self.noise_network(), classifier_teacher_network=self.make_classifier(), 
                use_ema=False, test_steps=4
            )
        with self.assertRaises((TypeError, ValueError)):
            DiffusionModel(
                network=self.noise_network(), teacher_dynamic_classes=True, 
                use_ema=False, test_steps=4
            )

    def test_classifier_wrappers_own_both_specialists_and_dynamic_classifier_config(self) -> None:
        """V1 and inherited V2 retain separate current teachers after the split."""

        parameters = inspect.signature(DiffusionClassifier.__init__).parameters
        self.assertIn("classifier_teacher_network", parameters)
        self.assertIn("teacher_dynamic_classes", parameters)
        for wrapper_cls in (DiffusionClassifier, DiffusionClassifierV2):
            with self.subTest(wrapper=wrapper_cls.__name__):
                model = self.make_model(
                    wrapper_cls, compile_model=False, teacher_dynamic_classes=True
                )
                self.assertEqual(
                    model.get_teacher_names(), ("previous", "current", "classifier", "noise")
                )
                self.assertIs(model.get_teacher_network("classifier"), model.classifier_teacher_network)
                self.assertIs(model.get_teacher_network("noise"), model.noise_teacher_network)
                self.assertIs(model.get_teacher_model("classifier"), model.classifier_teacher_network)
                self.assertIs(type(model.get_teacher_model("noise")), DiffusionModel)
                self.assertTrue(model.teacher_dynamic_classes)
                self.assertTrue(model.get_config()["teacher_dynamic_classes"])
                self.assertFalse(model.classifier_teacher_network.trainable)
                self.assertFalse(model.noise_teacher_network.trainable)

    def test_base_noise_specialist_fit_and_distillation_remain_independent(self) -> None:
        """Fit one native noise teacher, then use frozen native targets in a student step."""

        previous = self.noise_network()
        noise = self.noise_network()
        model = DiffusionModel(
            network=self.noise_network(), teacher_network=previous, 
            noise_teacher_network=noise, trainable_teacher=True, use_ema=False, 
            scheduler_name="clipped_cosine", test_steps=4, p_uncond=0., preprocess_type="standardize", 
            noise_loss_coef=0., noise_distil_loss_coef=1., seed=811
        )
        model.compile(optimizer=tf.keras.optimizers.SGD(.001), loss="mse", 
                      run_eagerly=True, jit_compile=False)
        model.compile_teacher(
            teacher_name="previous", optimizer=tf.keras.optimizers.SGD(.002), loss="mse", 
            run_eagerly=True, jit_compile=False
        )
        teacher_optimizer = tf.keras.optimizers.Adam(.002)
        model.compile_teacher(
            teacher_name="noise", optimizer=teacher_optimizer, loss="mse", 
            run_eagerly=True, jit_compile=False
        )
        owner = model.get_teacher_model("noise")
        self.assertIs(type(owner), DiffusionModel)
        self.assertIs(owner.network, noise)
        self.assertIs(owner.optimizer, teacher_optimizer)
        student_before = model.network.get_weights()
        previous_before = previous.get_weights()
        noise_before = noise.get_weights()
        history = model.fit_teacher(
            self.dataset(), teacher_name="noise", epochs=1, verbose=0
        )
        self.assertIn("noise_loss", history.history)
        self.assertNotIn("classifier_loss", history.history)
        self.assertEqual(int(teacher_optimizer.iterations), 1)
        self.assertEqual(int(model.optimizer.iterations), 0)
        self.assertEqual(int(model.get_teacher_model("previous").optimizer.iterations), 0)
        self.assertTrue(any(not np.array_equal(old, new) for old, new in
                            zip(noise_before, noise.get_weights())))
        self.assert_weights_equal(model.network, student_before)
        self.assert_weights_equal(previous, previous_before)
        noise_after = noise.get_weights()
        model._preprocess_training = True
        try:
            mapped = model.prep_inputs_map(self.images, self.labels)
        finally:
            model._preprocess_training = None
        results = model.train_step(mapped)
        self.assertIn("noise_distil_loss", results)
        for value in results.values():
            self.assertTrue(np.isfinite(value.numpy()).all())
        self.assertEqual(int(model.optimizer.iterations), 1)
        self.assertEqual(int(teacher_optimizer.iterations), 1)
        self.assert_weights_equal(previous, previous_before)
        self.assert_weights_equal(noise, noise_after)
        self.assertFalse(previous.trainable)
        self.assertFalse(noise.trainable)

    def test_base_delegates_native_fit_methods_to_supplied_v2_teacher_owner(self) -> None:
        """Select supported training methods from each cached native teacher wrapper."""

        for role in ("current", "noise"):
            with self.subTest(role=role):
                teacher = DiffusionClassifierV2(
                    network=self.make_network(), use_ema=False, test_steps=4, 
                    scheduler_name="clipped_cosine", p_uncond=0.
                )
                teacher.compile(
                    optimizer=tf.keras.optimizers.SGD(.002), loss="mse", 
                    run_eagerly=True, jit_compile=False
                )
                model = DiffusionModel(
                    network=self.noise_network(), trainable_teacher=True, 
                    use_ema=False, test_steps=4, **{role + "_teacher_network": teacher}
                )
                model.compile(
                    optimizer=tf.keras.optimizers.SGD(.001), loss="mse", 
                    run_eagerly=True, jit_compile=False
                )
                self.assertIs(model.get_teacher_model(role), teacher)
                self.assertIs(model.get_teacher_network(role), teacher.network)
                dataset = self.dataset()
                expected_history = {"noise_loss": [.125]}
                with patch.object(teacher, "fit_generator", return_value=expected_history) as fit:
                    history = model.fit_teacher(
                        dataset, teacher_name=role, fit_method="fit_generator", 
                        epochs=1, verbose=0
                    )
                fit.assert_called_once_with(x=dataset, epochs=1, verbose=0)
                self.assertIs(history, expected_history)
                self.assertIs(model.get_teacher_model(role), teacher)
                self.assertIs(model.get_teacher_network(role), teacher.network)
                self.assertFalse(teacher.network.trainable)
                self.assertEqual(int(model.optimizer.iterations), 0)

    def test_compile_and_fit_selectors_only_update_the_selected_teacher(self) -> None:
        """Selected teachers retain independent optimizers, masks and fit histories."""

        for eager in (True, False):
            with self.subTest(eager=eager):
                model = self.make_model(run_eagerly=eager, compile_model=False)
                classifier = model.get_teacher_network("classifier")
                noise = model.get_teacher_network("noise")
                classifier_optimizer = tf.keras.optimizers.Adam(.004)
                noise_optimizer = tf.keras.optimizers.SGD(.002)
                model.compile_teacher(
                    teacher_name="classifier", optimizer=classifier_optimizer, 
                    loss="sparse_categorical_crossentropy", metrics=["accuracy"], 
                    run_eagerly=eager, jit_compile=False
                )
                model.compile_teacher(teacher_name="noise", optimizer=noise_optimizer, 
                                      loss="mae", run_eagerly=eager, jit_compile=False)
                self.assertFalse(model.compiled)
                self.assertIs(model.get_teacher_model("classifier"), classifier)
                self.assertIs(model.get_teacher_model("noise").network, noise)
                self.assertIsInstance(model.get_teacher_model("noise"), DiffusionModel)
                self.assertNotIsInstance(model.get_teacher_model("noise"), DiffusionClassifier)
                model.compile(optimizer=tf.keras.optimizers.SGD(.001), loss="mse", 
                              run_eagerly=eager, jit_compile=False)
                self.assertIs(model.get_teacher_model("noise").optimizer, noise_optimizer)
                self.assertIs(classifier.optimizer, classifier_optimizer)
                student_before = model.network.get_weights()
                noise_before = noise.get_weights()
                classifier_before = classifier.get_weights()
                frozen_before = classifier.get_layer("base").get_layer("normalization").get_weights()
                class_history = model.fit_teacher(self.dataset(), teacher_name="classifier", epochs=1, verbose=0)
                self.assertIn("accuracy", class_history.history)
                self.assertNotIn("noise_loss", class_history.history)
                self.assertEqual(int(classifier_optimizer.iterations), 1)
                self.assertEqual(int(noise_optimizer.iterations), 0)
                self.assertTrue(any(not np.array_equal(old, new) for old, new in
                                    zip(classifier_before, classifier.get_weights())))
                self.assert_weights_equal(noise, noise_before)
                self.assert_weights_equal(classifier.get_layer("base").get_layer("normalization"), frozen_before)
                classifier_after = classifier.get_weights()
                noise_history = model.fit_teacher(self.dataset(), teacher_name="noise", epochs=1, verbose=0)
                self.assertIn("noise_loss", noise_history.history)
                self.assertNotIn("classifier_loss", noise_history.history)
                self.assertEqual(int(noise_optimizer.iterations), 1)
                self.assertEqual(int(classifier_optimizer.iterations), 1)
                self.assertTrue(any(not np.array_equal(old, new) for old, new in zip(noise_before, noise.get_weights())))
                self.assert_weights_equal(classifier, classifier_after)
                self.assert_weights_equal(model.network, student_before)
                self.assertEqual(int(model.optimizer.iterations), 0)
                self.assertIsNone(model.teacher_network)
                self.assertFalse(classifier.trainable)
                self.assertFalse(noise.trainable)

    def test_failed_selected_fit_refreezes_teachers_and_keeps_other_roles(self) -> None:
        """A failing classifier callback must not replace or unfreeze other roles."""

        model = self.make_model()
        classifier = model.get_teacher_network("classifier")
        noise = model.get_teacher_network("noise")
        student_before = model.network.get_weights()
        noise_before = noise.get_weights()

        def fail(logs: object = None) -> None:
            """Observe the restored fine-tuning mask and fail before optimization."""

            del logs
            self.assertTrue(classifier.trainable)
            self.assertFalse(classifier.get_layer("base").get_layer("early").trainable)
            self.assertFalse(classifier.get_layer("base").get_layer("normalization").trainable)
            self.assertTrue(classifier.get_layer("base").get_layer("tail").trainable)
            self.assertFalse(noise.trainable)
            raise RuntimeError("selected teacher callback failed")

        with self.assertRaisesRegex(RuntimeError, "selected teacher callback failed"):
            model.fit_teacher(self.dataset(), teacher_name="classifier", epochs=1, callbacks=[tf.keras.callbacks.LambdaCallback(on_train_begin=fail)], 
                              verbose=0)
        self.assertIs(model.get_teacher_network("classifier"), classifier)
        self.assertIs(model.get_teacher_network("noise"), noise)
        self.assertIsNone(model.teacher_network)
        self.assertFalse(classifier.trainable)
        self.assertFalse(noise.trainable)
        self.assert_weights_equal(model.network, student_before)
        self.assert_weights_equal(noise, noise_before)
        with self.assertRaises(ValueError):
            model.fit_teacher(self.dataset(), teacher_name="unknown", epochs=1, verbose=0)
        with self.assertRaises(ValueError):
            model.fit_teacher(self.dataset(), epochs=1, verbose=0)

    def test_offline_student_routes_classifier_and_noise_to_distinct_experts(self) -> None:
        """Real eager/graph V1 and V2 updates use both frozen current specialists."""

        for wrapper_cls in (DiffusionClassifier, DiffusionClassifierV2):
            for eager in (True, False):
                with self.subTest(wrapper=wrapper_cls.__name__, eager=eager):
                    model = self.make_model(wrapper_cls, run_eagerly=eager)
                    classifier = model.get_teacher_network("classifier")
                    noise = model.get_teacher_network("noise")
                    before = [teacher.get_weights() for teacher in (classifier, noise)]
                    noise_calls, class_calls, clean_inputs = [], [], []
                    original_noise = model._predict_teacher_noise
                    original_class = model._predict_single_teacher_labels
                    original_call = classifier.call

                    def predict_noise(*args: object, **kwargs: object) -> tf.Tensor:
                        """Record the noise network used by actual target inference."""

                        noise_calls.append(kwargs.get("teacher_network"))
                        return original_noise(*args, **kwargs)

                    def predict_class(*args: object, **kwargs: object) -> tf.Tensor:
                        """Record the classifier used by actual class-target inference."""

                        class_calls.append(kwargs.get("teacher_network"))
                        return original_class(*args, **kwargs)

                    def classifier_call(inputs: tf.Tensor, **kwargs: object) -> tf.Tensor:
                        """Capture external pixels passed into the ordinary classifier."""

                        clean_inputs.append(tf.convert_to_tensor(inputs).numpy())
                        return original_call(inputs, **kwargs)

                    parts = ("generator", "discriminator") if wrapper_cls is DiffusionClassifierV2 else tuple([None])
                    for part in parts:
                        # V2 prepares a separate mapped tuple for each optimizer phase.
                        if part is not None:
                            model._switch_train_part(part)
                            model._test_part = part
                        model._preprocess_training = True
                        with patch.object(model, "_predict_teacher_noise", side_effect=predict_noise), \
                             patch.object(model, "_predict_single_teacher_labels", side_effect=predict_class), \
                             patch.object(classifier, "call", side_effect=classifier_call):
                            mapped = model.prep_inputs_map(self.images, self.labels)
                        model._preprocess_training = None
                        step = model.train_step if eager else tf.function(model.train_step)
                        results = step(mapped)
                        for value in results.values():
                            self.assertTrue(np.isfinite(value.numpy()).all())
                    self.assertTrue(noise_calls)
                    self.assertTrue(class_calls)
                    self.assertTrue(all(teacher is noise for teacher in noise_calls))
                    self.assertTrue(all(teacher is classifier for teacher in class_calls))
                    for observed in clean_inputs:
                        np.testing.assert_allclose(observed, self.images, atol=2e-5)
                    for teacher, values in zip((classifier, noise), before):
                        self.assert_weights_equal(teacher, values)
                        self.assertFalse(teacher.trainable)
                    self.assertIsNone(model.teacher_network)

    def test_previous_snapshot_and_current_specialists_keep_independent_losses(self) -> None:
        """Three networks provide two separately weighted targets for each head."""

        source = self.make_model()
        previous = source.snapshot_teacher_network()
        model = self.make_model(num_classes=4, teacher_network=previous, 
                                previous_teacher_noise_loss_weight=2., current_teacher_noise_loss_weight=3., 
                                previous_teacher_clf_loss_weight=2., current_teacher_clf_loss_weight=3.)
        classifier = model.get_teacher_network("classifier")
        noise = model.get_teacher_network("noise")
        model.set_classifier_teacher_network(classifier, class_ids=[2, 3], task_class_ids=[2, 3])
        model.set_noise_teacher_network(noise, class_ids=[2, 3], task_class_ids=[2, 3])
        self.assertEqual([spec["network"] for spec in model._classifier_teacher_specs()], [previous, classifier])
        self.assertEqual([spec["network"] for spec in model._noise_teacher_specs()], [previous, noise])
        labels = tf.constant([0, 1, 2, 3], tf.int32)
        probabilities = tf.constant([[.1, .2, .3, .4], [.4, .3, .2, .1], 
                                     [.25, .25, .1, .4], [.1, .1, .7, .1]])
        class_targets = (tf.constant([[.8, .2]] * 4), tf.constant([[.25, .75]] * 4))
        expected_class = 0.
        for target, columns, weight in zip(class_targets, ([0, 1], [2, 3]), (2., 3.)):
            q = target.numpy()
            row_losses = np.sum(q * (np.log(q) - np.log(probabilities.numpy()[:, columns])), axis=-1)
            expected_class += weight * float(np.mean(row_losses[columns]))
        student_noise = tf.ones((4, 4, 4, 1))
        noise_targets = (2. * student_noise, 5. * student_noise)
        masks = (tf.constant([True, True, False, False]), tf.constant([False, False, True, True]))

        def losses() -> tuple[tf.Tensor, tf.Tensor]:
            """Compute both objectives using the same role order in eager and graph modes."""

            classifier_loss, _ = model.compute_clf_distil_loss(
                class_targets, probabilities, classes=labels, student_logits=tf.math.log(probabilities)
            )
            noise_loss = model.compute_distil_noise_loss(noise_targets, student_noise, teacher_noise_mask=masks)
            return classifier_loss, noise_loss

        for evaluate in (losses, tf.function(losses)):
            actual_class, actual_noise = evaluate()
            np.testing.assert_allclose(actual_class, expected_class, rtol=1e-5)
            np.testing.assert_allclose(actual_noise, 50., rtol=1e-6)
        previous_before = previous.get_weights()
        model.fit_teacher(self.dataset(), teacher_name="classifier", epochs=1, verbose=0)
        self.assertIs(model.teacher_network, previous)
        self.assert_weights_equal(previous, previous_before)
        self.assertEqual(model.classifier_teacher_class_ids, (2, 3))
        self.assertEqual(model.classifier_teacher_task_class_ids, (2, 3))
        self.assertEqual(model._classifier_teacher_specs()[1]["class_ids"], (2, 3))

    def test_two_tasks_grow_both_experts_without_replacing_previous_snapshot(self) -> None:
        """Independent class vocabularies grow before the student learns a second task."""

        model = self.make_model(num_classes=None)
        first = self.dataset([7, 9, 7, 9])
        for name in ("classifier", "noise"):
            before = model.network.get_weights()
            model.fit_teacher(first, teacher_name=name, epochs=1, verbose=0)
            self.assert_weights_equal(model.network, before)
        model.fit(first, epochs=1, verbose=0)
        previous = model.snapshot_teacher_network()
        model.set_teacher_network(previous)
        previous_before = previous.get_weights()
        student_before = model.network.get_weights()
        second = self.dataset([1, 12, 1, 12])
        for name in ("classifier", "noise"):
            model.fit_teacher(second, teacher_name=name, epochs=1, verbose=0)
            self.assert_weights_equal(model.network, student_before)
            self.assert_weights_equal(previous, previous_before)
            self.assertIs(model.teacher_network, previous)
        classifier = model.get_teacher_network("classifier")
        noise = model.get_teacher_network("noise")
        expected = {7: 0, 9: 1, 1: 2, 12: 3}
        self.assertEqual(dict(classifier._diffusion_seen_classes), expected)
        self.assertEqual(model.get_teacher_model("noise").seen_classes, expected)
        self.assertEqual(classifier.output_shape[-1], 4)
        self.assertEqual(noise.num_classes, 4)
        self.assertEqual(previous.num_classes, 2)
        self.assertEqual(int(model.get_teacher_model("classifier").optimizer.iterations), 2)
        self.assertEqual(int(model.get_teacher_model("noise").optimizer.iterations), 2)
        model.fit(self.dataset([7, 9, 1, 12]), epochs=1, verbose=0)
        self.assertEqual(model.seen_classes, expected)
        self.assertEqual(model.network.num_classes, 4)
        self.assert_weights_equal(previous, previous_before)
        self.assertEqual([spec["network"] for spec in model._classifier_teacher_specs()], [previous, classifier])
        self.assertEqual([spec["network"] for spec in model._noise_teacher_specs()], [previous, noise])
        for teacher in (previous, classifier, noise):
            self.assertFalse(teacher.trainable)

    def test_train_model_forwards_head_selection_with_v2_student(self) -> None:
        """The shared training adapter fits each expert without entering student phases."""

        model = self.make_model(DiffusionClassifierV2)
        student_before = model.network.get_weights()
        for name, metric in (("classifier", "accuracy"), ("noise", "noise_loss")):
            other_name = "noise" if name == "classifier" else "classifier"
            other = model.get_teacher_network(other_name)
            other_before = other.get_weights()
            with redirect_stdout(io.StringIO()):
                history = train_model(
                    None, model, self.dataset(), fit_method="fit_teacher", 
                    fit_kwargs={"teacher_name": name}, epochs=1, results_path=None, 
                    patience=0, save_config_=False, show_images=True, 
                    save_gifs=False, report_every_epoch=False, save_weights=False, 
                    run_trainset_eval=False, run_valset_eval=False, verbose=0
                )
            self.assertIn(metric, history)
            self.assertEqual(int(model.get_teacher_model(name).optimizer.iterations), 1)
            self.assert_weights_equal(other, other_before)
            self.assert_weights_equal(model.network, student_before)
        self.assertEqual(int(model.gen_optimizer.iterations), 0)
        self.assertEqual(int(model.clf_optimizer.iterations), 0)

    @staticmethod
    def loader(indices: list[int], **kwargs: object) -> tuple:
        """Return raw class-coded tiny images with noncontiguous dataset labels."""

        del kwargs
        labels = np.repeat(np.asarray(indices, dtype="int32"), 2)
        images = np.broadcast_to((40. + labels * 10.)[:, None, None, None], 
                                 (len(labels), 4, 4, 1)).astype("float32").copy()
        return images, labels, images.copy(), labels.copy(), images.copy(), labels.copy()

    def continual_options(self, model: DiffusionClassifier, **overrides: object) -> dict:
        """Run two current-expert fits and student steps over a reordered vocabulary."""

        options = dict(
            generative_model=model, load_dataset_fn=self.loader, 
            load_dataset_fn_kwargs={"preprocess": None}, class_num=4, 
            class_order=[7, 1, 9, 3], task_size=2, use_generative_model_classifier=True, 
            generative_model_kwargs={"train_num": -1}, use_generative_replay=False, 
            use_distillation=True, dual_teacher_distillation=False, 
            remove_prev_classes=False, batch_size=4, epochs=1, optimizer_steps_per_epoch=1, 
            callback_patience=0, plot_results=False, deterministic_ops=True, show_generated_images=False, 
            show_network_summary=False, return_details=True, verbose=0, 
            seed=811
        )
        options.update(overrides)
        return options

    def test_continual_handoff_preserves_prefitted_noise_optimizer(self) -> None:
        """Entering continual training retains an already fitted native expert's slots."""

        model = self.make_model(num_classes=None)
        model.fit_teacher(self.dataset(), teacher_name="noise", epochs=1, verbose=0)
        owner = model.get_teacher_model("noise")
        optimizer = owner.optimizer
        self.assertEqual(int(optimizer.iterations), 1)
        before = [value.numpy().copy() for value in optimizer.variables]
        original_fit = model.fit_teacher
        noise_fit_observed = []

        def inspect_fit(*args: object, **kwargs: object) -> object:
            """Stop immediately before the first continual noise update after checking state."""

            # The classifier may train first, but must leave the native owner untouched.
            if kwargs.get("teacher_name") == "noise":
                noise_fit_observed.append(True)
                self.assertIs(model.get_teacher_model("noise"), owner)
                self.assertIs(owner.optimizer, optimizer)
                self.assertEqual(int(optimizer.iterations), 1)
                self.assertEqual(len(optimizer.variables), len(before))
                for expected, actual in zip(before, optimizer.variables):
                    np.testing.assert_array_equal(actual.numpy(), expected)
                raise RuntimeError("prefitted noise state retained")
            return original_fit(*args, **kwargs)

        with patch.object(model, "fit_teacher", side_effect=inspect_fit), \
             redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(RuntimeError, "prefitted noise state retained"):
            continually_learn(**self.continual_options(model))
        self.assertEqual(noise_fit_observed, [True])
        self.assertFalse(model.get_teacher_network("noise").trainable)

    def test_continual_recovery_restores_both_experts_and_previous_snapshot(self) -> None:
        """A task-one checkpoint reproduces all independent model and optimizer states."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models, details = [], []
            for resume in (False, True):
                tf.keras.backend.clear_session()
                configure_runtime(dtype_policy="float32", deterministic_ops=True, seed=811)
                model = self.make_model(num_classes=None)
                options = self.continual_options(
                    model, save_task_checkpoints=True, 
                    checkpoint_dir=str(root / ("resumed" if resume else "original"))
                )
                # Resume from the committed first task into an independent run directory.
                if resume:
                    options["resume_from"] = str(root / "original" / "task-0000")
                with redirect_stdout(io.StringIO()):
                    details.append(continually_learn(**options))
                models.append(model)
            expected, actual = models
            self.assertEqual(details[0]["task_classes"], [[7, 1], [9, 3]])
            self.assertEqual(actual.seen_classes, expected.seen_classes)
            self.assertEqual(actual.network.num_classes, 4)
            self.assert_weights_equal(actual.network, expected.network.get_weights())
            self.assertIsNotNone(actual.teacher_network)
            self.assert_weights_equal(actual.teacher_network, expected.teacher_network.get_weights())
            for name in ("classifier", "noise"):
                original = expected.get_teacher_network(name)
                restored = actual.get_teacher_network(name)
                self.assert_weights_equal(restored, original.get_weights())
                self.assertFalse(restored.trainable)
                original_model = expected.get_teacher_model(name)
                restored_model = actual.get_teacher_model(name)
                self.assertEqual(int(restored_model.optimizer.iterations), int(original_model.optimizer.iterations))
                self.assertGreaterEqual(int(restored_model.optimizer.iterations), 2)
                self.assertEqual(len(restored_model.optimizer.variables), len(original_model.optimizer.variables))
                for old, new in zip(original_model.optimizer.variables, restored_model.optimizer.variables):
                    np.testing.assert_array_equal(new.numpy(), old.numpy())
            self.assertEqual(dict(actual.get_teacher_network("classifier")._diffusion_seen_classes), 
                             dict(expected.get_teacher_network("classifier")._diffusion_seen_classes))
            self.assertEqual(actual.get_teacher_model("noise").seen_classes, 
                             expected.get_teacher_model("noise").seen_classes)
            self.assertEqual(actual.get_teacher_network("classifier").output_shape[-1], 4)
            self.assertEqual(actual.get_teacher_network("noise").num_classes, 4)
            self.assertEqual(len(expected.optimizer.variables), len(actual.optimizer.variables))
            for old, new in zip(expected.optimizer.variables, actual.optimizer.variables):
                np.testing.assert_array_equal(new.numpy(), old.numpy())
            restored_classifier = actual.get_teacher_network("classifier")
            classifier_before = restored_classifier.get_weights()

            def inspect_mask(logs: object = None) -> None:
                """Check persisted fine-tuning flags without advancing recovered weights."""

                del logs
                base = restored_classifier.get_layer("base")
                self.assertTrue(base.trainable)
                self.assertFalse(base.get_layer("early").trainable)
                self.assertFalse(base.get_layer("normalization").trainable)
                self.assertTrue(base.get_layer("tail").trainable)
                self.assertFalse(actual.get_teacher_network("noise").trainable)
                self.assertFalse(actual.teacher_network.trainable)
                raise RuntimeError("recovered mask checked")

            known_labels = list(restored_classifier._diffusion_seen_classes)[:2]
            with self.assertRaisesRegex(RuntimeError, "recovered mask checked"):
                actual.fit_teacher(self.dataset(known_labels * 2), teacher_name="classifier", epochs=1, callbacks=[tf.keras.callbacks.LambdaCallback(on_train_begin=inspect_mask)], 
                                   verbose=0)
            self.assertFalse(restored_classifier.trainable)
            self.assert_weights_equal(restored_classifier, classifier_before)


# Direct invocation runs only the focused specialist-teacher regressions.
if __name__ == "__main__":
    unittest.main()
