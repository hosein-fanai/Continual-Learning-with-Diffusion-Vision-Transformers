"""Serialized specialist identities and fixed continual development cohorts."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import Config, load_config, save_config
from common.learner import _load_continual_arrays, continually_learn
from common.specialist_teacher_artifacts import (
    load_specialist_teacher_descriptors, normalize_specialist_teacher_descriptors
)
from common.train import main
from common.tests import test_clean_classifier_training as classifier_fixtures
from diffusion import DiffusionClassifier


class SpecialistTeacherArtifactTests(unittest.TestCase):
    """Verify artifact binding without replacing teacher training or KD mathematics."""

    def test_config_round_trip_and_changed_teacher_rejected(self) -> None:
        """A sealed artifact cannot silently change between coordinator and worker."""

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "classifier.keras"
            artifact.write_bytes(b"teacher revision one")
            descriptors = normalize_specialist_teacher_descriptors({
                "classifier": {"format": "keras", "path": str(artifact)}
            })
            self.assertEqual(descriptors["classifier"]["path"], str(artifact.resolve()))
            config = Config(continually_learn={"specialist_teacher_descriptors": descriptors})
            path = Path(directory) / "student.yaml"
            save_config(config, path)
            self.assertEqual(load_config(path).continually_learn.specialist_teacher_descriptors, descriptors)
            artifact.write_bytes(b"teacher revision two")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                load_specialist_teacher_descriptors(descriptors)

    def test_native_config_binds_relative_weights_and_loads_existing_factory(self) -> None:
        """The worker receives the same native Config, optimizer and exact weights."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "noise.weights.h5"
            weights.write_bytes(b"native weight fixture")
            source = Config(
                model={"name": "unet", "wrapper_name": "diffusion_model", "weights_path": weights.name}, 
                training={"task": "generation"}
            )
            path = root / "noise.yaml"
            save_config(source, path)
            descriptor = normalize_specialist_teacher_descriptors({
                "noise": {"format": "config", "path": str(path)}
            })
            expected = object()
            with patch("common.model.get_model", return_value=expected) as factory:
                self.assertIs(load_specialist_teacher_descriptors(descriptor)["noise"], expected)
            passed = factory.call_args.args[0]
            self.assertEqual(passed.model.weights_path, str(weights.resolve()))
            self.assertEqual(passed.optimizer, source.optimizer)
            self.assertFalse(passed.model.show_network_summary)
            weights.write_bytes(b"replacement native weights")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                normalize_specialist_teacher_descriptors(descriptor)

    def test_classifier_loader_retains_compile_settings(self) -> None:
        """Use safe Keras loading and require the supplied teacher's own optimizer."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classifier.keras"
            path.write_bytes(b"compiled classifier fixture")
            descriptors = normalize_specialist_teacher_descriptors({
                "classifier": {"format": "keras", "path": str(path)}
            })
            teacher = SimpleNamespace(compiled=True, optimizer=object())
            with patch("tensorflow.keras.models.load_model", return_value=teacher) as loader:
                self.assertIs(load_specialist_teacher_descriptors(descriptors)["classifier"], teacher)
            loader.assert_called_once_with(str(path.resolve()), compile=True, safe_mode=True)
            with patch("tensorflow.keras.models.load_model", return_value=SimpleNamespace(compiled=False)):
                with self.assertRaisesRegex(ValueError, "compile settings"):
                    load_specialist_teacher_descriptors(descriptors)

    def test_main_passes_independent_specialists_and_reseeds_student(self) -> None:
        """Teacher construction is isolated from the student's initialization seed."""

        classifier, noise = object(), object()
        config = Config(
            continually_learn={"specialist_teacher_descriptors": {"classifier": {}, "noise": {}}}, 
            training={"task": "continual", "seed": 19}
        )
        bundle = {"generative_model": object(), "classifier": object()}
        with patch("common.train.configure_runtime") as configure, \
             patch("common.train.get_datasets", return_value=(lambda: None, None)), \
             patch("common.train.get_model", return_value=bundle) as factory, \
             patch("common.train.train_model", return_value={}), \
             patch("common.train.report", return_value={}), \
             patch("common.specialist_teacher_artifacts.load_specialist_teacher_descriptors", 
                   return_value={"classifier": classifier, "noise": noise}), \
             redirect_stdout(io.StringIO()):
            main(config)
        self.assertIs(factory.call_args.kwargs["classifier_teacher_network"], classifier)
        self.assertIs(factory.call_args.kwargs["noise_teacher_network"], noise)
        self.assertEqual(len(configure.call_args_list), 2)
        self.assertEqual(configure.call_args_list[0], configure.call_args_list[1])

    def test_fixed_dataset_seed_preserves_capped_cohorts_across_model_seeds(self) -> None:
        """Confirmation seeds vary model/replay RNG while capped rows remain paired."""

        labels = np.repeat(np.arange(4), 10)
        images = np.arange(len(labels), dtype="float32")[:, None]
        observed = []

        def loader(**kwargs: object) -> tuple:
            """Record the split seed without replacing the actual cap implementation."""

            observed.append(kwargs["seed"])
            return images, labels, images + 100., labels, images + 200., labels

        results = [
            _load_continual_arrays(
                loader, [3, 0, 2, 1], False, 
                {"preprocess": None, "onehot_labels": False, "seed": 42}, 
                12, 8, 0, model_seed, dataset_seed=42
            ) for model_seed in (19, 73)
        ]
        for left, right in zip(results[0][0], results[1][0]):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(observed, [42, 42])
        self.assertNotEqual(results[0][1].integers(2**31), results[1][1].integers(2**31))

    def test_loaded_fixed_width_unet_and_keras_specialists_train_only_new_task_rows(self) -> None:
        """Real artifact-loaded experts update through two V1 continual tasks.

        The UNet retains its fixed ten-class conditional vocabulary while the
        ordinary image classifier grows from two to four classes. Both train only
        new real rows; the joint student also retains previous-task snapshots.
        """

        original_policy = tf.keras.mixed_precision.global_policy()
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.utils.set_random_seed(811)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                classifier = tf.keras.Sequential([
                    tf.keras.Input((32, 32, 3)), tf.keras.layers.Rescaling(1. / 255.), 
                    tf.keras.layers.Conv2D(2, 1), tf.keras.layers.GlobalAveragePooling2D(), 
                    tf.keras.layers.Dense(2, activation="softmax")
                ])
                classifier.compile(
                    optimizer=tf.keras.optimizers.SGD(.01), 
                    loss="sparse_categorical_crossentropy", metrics=["accuracy"], 
                    run_eagerly=True, jit_compile=False
                )
                classifier_path = root / "classifier.keras"
                classifier.save(classifier_path)
                noise_config = Config(
                    dataset={"name": "CIFAR10"}, 
                    model={
                        "name": "unet", "wrapper_name": "diffusion_model", 
                        "show_network_summary": False, 
                        "kwargs": {
                            "num_classes": 10, "timesteps": 8, "widths": [2, 3], 
                            "block_depth": 1, "bottleneck_width": 4, "bottleneck_depth": 1, 
                            "image_embedding_dim": 2, "time_embedding_dim": 2, 
                            "label_embedding_dim": 2, "use_batch_norm": False, 
                            "compile_args": {"run_eagerly": True, "jit_compile": False}
                        }, 
                        "wrapper_kwargs": {
                            "use_ema": False, "test_network_name": "raw", "scheduler_name": "clipped_cosine", 
                            "test_steps": 4, "p_uncond": 0., "preprocess_type": "standardize"
                        }
                    }, 
                    optimizer={"name": "sgd", "schedule": "constant", "initial_learning_rate": .01}, 
                    training={"task": "generation", "seed": 811}
                )
                noise_path = root / "noise.yaml"
                save_config(noise_config, noise_path)
                specialists = load_specialist_teacher_descriptors({
                    "classifier": {"format": "keras", "path": str(classifier_path)}, 
                    "noise": {"format": "config", "path": str(noise_path)}
                }, seed=811)
                self.assertEqual(specialists["noise"].network.num_classes, 10)
                self.assertFalse(specialists["noise"].network.dynamic_num_classes)
                network = classifier_fixtures.CleanClassifierTrainingTests.make_network(
                    self, image_size=32, channels=3, patch_size=16, num_classes=None, seed=811
                )
                model = DiffusionClassifier(
                    network=network, classifier_teacher_network=specialists["classifier"], 
                    noise_teacher_network=specialists["noise"], trainable_teacher=True, 
                    teacher_dynamic_classes=True, teacher_training="each_task", use_ema=False, 
                    preprocess_type="standardize", scheduler_name="clipped_cosine", test_steps=4, 
                    p_uncond=0., mask_by_nulls=False, mask_by_t_threshold=False, 
                    clf_distil_loss_coef=1., noise_distil_loss_coef=1., 
                    clf_distil_type="soft", seed=811
                )
                model.compile(optimizer=tf.keras.optimizers.SGD(.001), loss="mse", 
                              run_eagerly=True, jit_compile=False)
                fit_rows = []
                original_fit = model.fit_teacher

                def record_teacher_fit(x: object, **kwargs: object) -> object:
                    """Inspect actual phase input labels before delegating the same fit."""

                    labels = sorted({int(label) for _, batch in x for label in batch.numpy()})
                    fit_rows.append((kwargs.get("teacher_name"), labels))
                    return original_fit(x, **kwargs)

                def loader(indices: list[int], **kwargs: object) -> tuple:
                    """Return two raw RGB examples for every requested original label."""

                    del kwargs
                    labels = np.repeat(np.asarray(indices, dtype="int32"), 2)
                    images = np.broadcast_to((40. + labels * 10.)[:, None, None, None], 
                                             (len(labels), 32, 32, 3)).astype("float32").copy()
                    return images, labels, images.copy(), labels.copy(), images.copy(), labels.copy()

                with patch.object(model, "fit_teacher", side_effect=record_teacher_fit), \
                     redirect_stdout(io.StringIO()):
                    details = continually_learn(
                        generative_model=model, load_dataset_fn=loader, 
                        load_dataset_fn_kwargs={"preprocess": None}, class_num=4, 
                        class_order=[7, 1, 9, 3], task_size=2, use_generative_model_classifier=True, 
                        generative_model_kwargs={"train_num": -1}, use_generative_replay=False, 
                        use_distillation=True, remove_prev_classes=False, batch_size=4, epochs=1, 
                        optimizer_steps_per_epoch=1, callback_patience=0, plot_results=False, 
                        deterministic_ops=True, show_generated_images=False, show_network_summary=False, 
                        return_details=True, verbose=0, seed=811
                    )
                self.assertEqual(fit_rows, [
                    ("classifier", [0, 1]), ("noise", [0, 1]), 
                    ("classifier", [2, 3]), ("noise", [2, 3])
                ])
                self.assertEqual(details["task_classes"], [[7, 1], [9, 3]])
                self.assertEqual(model.get_teacher_network("noise").num_classes, 10)
                self.assertEqual(model.get_teacher_network("classifier").output_shape[-1], 4)
                self.assertEqual(model.noise_teacher_task_class_ids, (2, 3))
                self.assertEqual(model.classifier_teacher_task_class_ids, (2, 3))
                self.assertEqual(model.teacher_network.num_classes, 4)
                for role in ("classifier", "noise"):
                    self.assertFalse(model.get_teacher_network(role).trainable)
                    self.assertGreaterEqual(int(model.get_teacher_model(role).optimizer.iterations), 2)
        finally:
            tf.keras.backend.clear_session()
            tf.keras.mixed_precision.set_global_policy(original_policy)


# Direct invocation runs only the specialist artifact and cohort regressions.
if __name__ == "__main__":
    unittest.main()
