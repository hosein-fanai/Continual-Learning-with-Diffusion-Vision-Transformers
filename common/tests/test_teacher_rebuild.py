"""Rebuild current teachers while retaining frozen weights and historical snapshots."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.learner import continually_learn
from common.runtime import configure_runtime
from common.tests import test_head_teachers as fixtures
from diffusion import DiffusionClassifier, DiffusionClassifierV2, DiffusionModel, UNet


class TeacherRebuildTests(unittest.TestCase):
    """Exercise fresh native and ordinary teacher training through public role selectors."""

    setUp = fixtures.HeadTeacherTests.setUp
    tearDown = fixtures.HeadTeacherTests.tearDown
    make_network = fixtures.HeadTeacherTests.make_network
    noise_network = staticmethod(fixtures.HeadTeacherTests.noise_network)
    make_classifier = fixtures.HeadTeacherTests.make_classifier
    make_model = fixtures.HeadTeacherTests.make_model
    loader = staticmethod(fixtures.HeadTeacherTests.loader)
    continual_options = fixtures.HeadTeacherTests.continual_options
    dataset = fixtures.HeadTeacherTests.dataset
    assert_weights_equal = fixtures.HeadTeacherTests.assert_weights_equal

    @staticmethod
    def unet(num_classes: int | None = 2) -> UNet:
        """Construct a bounded conditional denoiser with a frozen input projection."""

        network = UNet(
            num_classes=num_classes, timesteps=8, image_size=4, channels=1, 
            widths=[2], block_depth=1, bottleneck_width=4, bottleneck_depth=1, 
            image_embedding_dim=2, time_embedding_dim=2, label_embedding_dim=2, 
            use_batch_norm=False, seed=811
        )
        network.image_embedder.trainable = False
        network.image_embedder.set_weights([
            np.full_like(value, .25) for value in network.image_embedder.get_weights()
        ])
        return network

    def assert_fresh_optimizer(
        self, old: tf.keras.optimizers.Optimizer, new: tf.keras.optimizers.Optimizer
    ) -> None:
        """Require the same optimization recipe with new variables and no elapsed steps."""

        self.assertIsNot(new, old)
        self.assertEqual(new.get_config(), old.get_config())
        self.assertEqual(int(new.iterations), 0)
        self.assertFalse({id(value) for value in old.variables}
                         & {id(value) for value in new.variables})
        for value in new.variables:
            # Materialized Adam accumulation slots must start empty after rebuilding.
            if "momentum" in value.name or "velocity" in value.name:
                np.testing.assert_array_equal(value.numpy(), np.zeros(value.shape))

    def test_unet_noise_teacher_rebuild_resets_training_and_preserves_frozen_projection(self) -> None:
        """Train a native UNet, rebuild its selected role, then train its fresh output head."""

        noise = self.unet()
        model = DiffusionModel(
            network=self.noise_network(), noise_teacher_network=noise, 
            trainable_teacher=True, use_ema=False, preprocess_type="standardize", 
            scheduler_name="clipped_cosine", test_steps=4, p_uncond=0., seed=811
        )
        model.compile(
            optimizer=tf.keras.optimizers.SGD(.001), loss="mse", 
            run_eagerly=True, jit_compile=False
        )
        model.compile_teacher(
            teacher_name="noise", optimizer=tf.keras.optimizers.Adam(.003), loss="mae", 
            run_eagerly=True, jit_compile=False
        )
        previous = model.snapshot_teacher_network("raw")
        model.set_teacher_network(previous)
        previous_before = previous.get_weights()
        student_before = model.network.get_weights()
        frozen_before = noise.image_embedder.get_weights()
        model.fit_teacher(self.dataset(), teacher_name="noise", epochs=1, verbose=0)
        old_owner = model.get_teacher_model("noise")
        old_optimizer = old_owner.optimizer
        self.assertEqual(int(old_optimizer.iterations), 1)
        self.assert_weights_equal(noise.image_embedder, frozen_before)
        trained_before = noise.get_weights()
        self.assertTrue(any(np.any(value != 0.) for value in noise.output_projection.get_weights()))

        rebuilt = model.rebuild_teacher_network("noise", seed=907)

        self.assertIs(rebuilt, model.noise_teacher_network)
        self.assertIsInstance(rebuilt, UNet)
        self.assertIsNot(rebuilt, noise)
        self.assertFalse(rebuilt.trainable)
        self.assertFalse(rebuilt.image_embedder.trainable)
        self.assert_weights_equal(rebuilt.image_embedder, frozen_before)
        for value in rebuilt.output_projection.get_weights():
            np.testing.assert_array_equal(value, np.zeros_like(value))
        new_owner = model.get_teacher_model("noise")
        self.assertIsNot(new_owner, old_owner)
        self.assertIs(new_owner.network, rebuilt)
        self.assertEqual(new_owner.get_compile_config()["loss"], "mae")
        self.assert_fresh_optimizer(old_optimizer, new_owner.optimizer)
        self.assertFalse({id(value) for value in noise.weights}
                         & {id(value) for value in rebuilt.weights})
        fresh_before = rebuilt.get_weights()
        model.fit_teacher(self.dataset(), teacher_name="noise", epochs=1, verbose=0)
        self.assertEqual(int(new_owner.optimizer.iterations), 1)
        self.assertTrue(any(not np.array_equal(before, after)
                            for before, after in zip(fresh_before, rebuilt.get_weights())))
        self.assertFalse(rebuilt.trainable)
        self.assert_weights_equal(rebuilt.image_embedder, frozen_before)
        self.assert_weights_equal(noise, trained_before)
        self.assert_weights_equal(previous, previous_before)
        self.assertIs(model.teacher_network, previous)
        self.assert_weights_equal(model.network, student_before)
        self.assertEqual(int(model.optimizer.iterations), 0)

    def test_classifier_specialist_rebuild_preserves_nested_mask_grown_head_and_other_roles(self) -> None:
        """V1 and V2 keep a grown classifier vocabulary while restarting only its learned weights."""

        for wrapper_cls in (DiffusionClassifier, DiffusionClassifierV2):
            with self.subTest(wrapper=wrapper_cls.__name__):
                model = self.make_model(wrapper_cls, teacher_dynamic_classes=True)
                classifier = model.classifier_teacher_network
                base = classifier.get_layer("base")
                for name in ("early", "normalization"):
                    layer = base.get_layer(name)
                    layer.set_weights([
                        np.full_like(value, .25 * (index + 1))
                        for index, value in enumerate(layer.get_weights())
                    ])
                frozen_before = {
                    name: base.get_layer(name).get_weights()
                    for name in ("early", "normalization")
                }
                for labels in ([7, 9, 7, 9], [1, 12, 1, 12]):
                    model.fit_teacher(
                        self.dataset(labels), teacher_name="classifier", epochs=1, verbose=0
                    )
                classifier = model.classifier_teacher_network
                self.assertEqual(classifier.output_shape[-1], 4)
                expected_mapping = {7: 0, 9: 1, 1: 2, 12: 3}
                self.assertEqual(dict(classifier._diffusion_seen_classes), expected_mapping)
                model.set_classifier_teacher_network(
                    classifier, class_ids=[7, 9, 1, 12], task_class_ids=[1, 12]
                )
                previous = model.snapshot_teacher_network("raw")
                model.set_teacher_network(previous)
                previous_before = previous.get_weights()
                noise = model.noise_teacher_network
                noise_before = noise.get_weights()
                noise_owner = model.get_teacher_model("noise")
                noise_optimizer = noise_owner.optimizer
                student_before = model.network.get_weights()
                old_optimizer = classifier.optimizer
                self.assertEqual(int(old_optimizer.iterations), 2)
                # Distinct learned values make reset coverage independent of RNG coincidence.
                classifier.get_layer("head").set_weights([
                    np.full_like(value, 9.) for value in classifier.get_layer("head").get_weights()
                ])
                classifier.get_layer("base").get_layer("tail").set_weights([
                    np.full_like(value, 7.)
                    for value in classifier.get_layer("base").get_layer("tail").get_weights()
                ])
                trained_before = classifier.get_weights()

                rebuilt = model.rebuild_teacher_network("classifier", seed=919)

                self.assertIs(rebuilt, model.classifier_teacher_network)
                self.assertIs(model.get_teacher_model("classifier"), rebuilt)
                self.assertIsNot(rebuilt, classifier)
                self.assertFalse(rebuilt.trainable)
                self.assertEqual(rebuilt.output_shape[-1], 4)
                self.assertEqual(dict(rebuilt._diffusion_seen_classes), expected_mapping)
                self.assertEqual(model.classifier_teacher_class_ids, (7, 9, 1, 12))
                self.assertEqual(model.classifier_teacher_task_class_ids, (1, 12))
                self.assert_fresh_optimizer(old_optimizer, rebuilt.optimizer)
                self.assertEqual(rebuilt.get_compile_config()["loss"], "sparse_categorical_crossentropy")
                for value in rebuilt.get_layer("head").get_weights():
                    self.assertLess(float(np.max(np.abs(value))), 9.)
                for value in rebuilt.get_layer("base").get_layer("tail").get_weights():
                    self.assertLess(float(np.max(np.abs(value))), 7.)
                for name, before in frozen_before.items():
                    self.assert_weights_equal(rebuilt.get_layer("base").get_layer(name), before)
                self.assertFalse({id(value) for value in classifier.weights}
                                 & {id(value) for value in rebuilt.weights})
                observed_masks = []

                def observe(logs: object = None) -> None:
                    """Check the preserved fine-tuning mask at an actual new training boundary."""

                    del logs
                    rebuilt_base = rebuilt.get_layer("base")
                    self.assertTrue(rebuilt.trainable)
                    self.assertTrue(rebuilt_base.trainable)
                    self.assertFalse(rebuilt_base.get_layer("early").trainable)
                    self.assertFalse(rebuilt_base.get_layer("normalization").trainable)
                    self.assertTrue(rebuilt_base.get_layer("tail").trainable)
                    self.assertTrue(rebuilt.get_layer("head").trainable)
                    observed_masks.append(True)

                fresh_head = rebuilt.get_layer("head").get_weights()
                history = model.fit_teacher(
                    self.dataset([1, 12, 1, 12]), teacher_name="classifier", epochs=1, 
                    callbacks=[tf.keras.callbacks.LambdaCallback(on_train_begin=observe)], verbose=0
                )
                self.assertEqual(observed_masks, [True])
                self.assertIn("accuracy", history.history)
                self.assertEqual(int(rebuilt.optimizer.iterations), 1)
                self.assertEqual(int(old_optimizer.iterations), 2)
                self.assertTrue(any(not np.array_equal(before, after) for before, after in
                                    zip(fresh_head, rebuilt.get_layer("head").get_weights())))
                self.assertFalse(rebuilt.trainable)
                for name, before in frozen_before.items():
                    self.assert_weights_equal(rebuilt.get_layer("base").get_layer(name), before)
                self.assert_weights_equal(classifier, trained_before)
                self.assert_weights_equal(previous, previous_before)
                self.assertIs(model.teacher_network, previous)
                self.assertIs(model.noise_teacher_network, noise)
                self.assertIs(model.get_teacher_model("noise"), noise_owner)
                self.assertIs(noise_owner.optimizer, noise_optimizer)
                self.assert_weights_equal(noise, noise_before)
                self.assert_weights_equal(model.network, student_before)
                self.assertEqual(int(model.optimizer.iterations), 0)

    def test_rebuild_rejects_missing_unsupported_and_nontrainable_roles(self) -> None:
        """Invalid requests fail before replacing or unfreezing an attached teacher."""

        model = DiffusionModel(
            network=self.noise_network(), noise_teacher_network=self.unet(), 
            trainable_teacher=True, use_ema=False, test_steps=4
        )
        original = model.noise_teacher_network
        for role in ("classifier", "unknown", "previous", "current"):
            with self.subTest(role=role), self.assertRaises(ValueError):
                model.rebuild_teacher_network(role)
            self.assertIs(model.noise_teacher_network, original)
            self.assertFalse(original.trainable)
        frozen = self.make_model(trainable_teacher=False)
        with self.assertRaisesRegex(ValueError, "trainable_teacher"):
            frozen.rebuild_teacher_network("classifier")
        self.assertFalse(frozen.classifier_teacher_network.trainable)

    def test_failed_rebuild_leaves_original_teacher_and_optimizer_attached(self) -> None:
        """A failed constructor cannot partially replace a fitted teacher or its state."""

        model = DiffusionModel(
            network=self.noise_network(), noise_teacher_network=self.unet(), 
            trainable_teacher=True, use_ema=False, test_steps=4
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(.001), loss="mse", 
            run_eagerly=True, jit_compile=False
        )
        original = model.noise_teacher_network
        owner = model.get_teacher_model("noise")
        optimizer = owner.optimizer
        before = original.get_weights()
        with patch.object(UNet, "from_config", side_effect=RuntimeError("rebuild failed")), \
             self.assertRaisesRegex(RuntimeError, "rebuild failed"):
            model.rebuild_teacher_network("noise")
        self.assertIs(model.noise_teacher_network, original)
        self.assertIs(model.get_teacher_model("noise"), owner)
        self.assertIs(owner.optimizer, optimizer)
        self.assertFalse(original.trainable)
        self.assert_weights_equal(original, before)

    def test_failed_classifier_compile_does_not_attach_candidate(self) -> None:
        """Reject a replacement whose compile recipe fails without changing the trained role."""

        model = self.make_model()
        model.fit_teacher(self.dataset(), teacher_name="classifier", epochs=1, verbose=0)
        original = model.classifier_teacher_network
        optimizer = original.optimizer
        before = original.get_weights()
        with patch.object(
            tf.keras.Sequential, "compile_from_config", 
            side_effect=RuntimeError("candidate compilation failed")
        ), self.assertRaisesRegex(RuntimeError, "candidate compilation failed"):
            model.rebuild_teacher_network("classifier")
        self.assertIs(model.classifier_teacher_network, original)
        self.assertIs(model.get_teacher_model("classifier"), original)
        self.assertIs(original.optimizer, optimizer)
        self.assertEqual(int(optimizer.iterations), 1)
        self.assertFalse(original.trainable)
        self.assert_weights_equal(original, before)

    def test_continual_recovery_preserves_native_frozen_mask_and_optimizer(self) -> None:
        """Recover task two with the same partial UNet mask, frozen weights and optimizer."""

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_root = Path(temporary)
            models = []
            frozen_before = None
            for resume in (False, True):
                tf.keras.backend.clear_session()
                configure_runtime(dtype_policy="float32", deterministic_ops=True, seed=811)
                noise = self.unet(num_classes=None)
                frozen_before = noise.image_embedder.get_weights()
                model = self.make_model(num_classes=None, noise_teacher_network=noise)
                options = self.continual_options(
                    model, save_task_checkpoints=True, 
                    checkpoint_dir=str(checkpoint_root / ("resumed" if resume else "original"))
                )
                # The resumed run loads the committed task-one training state before task two.
                if resume:
                    options["resume_from"] = str(checkpoint_root / "original" / "task-0000")
                with redirect_stdout(io.StringIO()):
                    continually_learn(**options)
                models.append(model)

            expected, actual = models
            original = expected.noise_teacher_network
            restored = actual.noise_teacher_network
            original_owner = expected.get_teacher_model("noise")
            restored_owner = actual.get_teacher_model("noise")
            self.assertIsInstance(restored, UNet)
            self.assertEqual(restored.num_classes, 4)
            self.assertEqual(restored_owner.seen_classes, original_owner.seen_classes)
            self.assert_weights_equal(restored, original.get_weights())
            self.assert_weights_equal(restored.image_embedder, frozen_before)
            self.assertFalse(restored.trainable)
            self.assertEqual(
                actual._get_native_teacher_fit_state("noise")[1], 
                expected._get_native_teacher_fit_state("noise")[1]
            )
            self.assertEqual(int(restored_owner.optimizer.iterations), 2)
            self.assertEqual(int(original_owner.optimizer.iterations), 2)
            self.assertEqual(len(restored_owner.optimizer.variables), len(original_owner.optimizer.variables))
            for old, new in zip(original_owner.optimizer.variables, restored_owner.optimizer.variables):
                np.testing.assert_array_equal(new.numpy(), old.numpy())
            previous = actual.teacher_network
            previous_before = previous.get_weights()
            student_before = actual.network.get_weights()
            trained_before = restored.get_weights()
            observed_masks = []

            def observe(logs: object = None) -> None:
                """Inspect the restored native mask during an additional real teacher update."""

                del logs
                self.assertTrue(restored.trainable)
                self.assertFalse(restored.image_embedder.trainable)
                self.assertTrue(restored.output_projection.trainable)
                observed_masks.append(True)

            known_labels = list(restored_owner.seen_classes)[:2]
            history = actual.fit_teacher(
                self.dataset(known_labels * 2), teacher_name="noise", epochs=1, 
                callbacks=[tf.keras.callbacks.LambdaCallback(on_train_begin=observe)], verbose=0
            )
            self.assertEqual(observed_masks, [True])
            self.assertIn("noise_loss", history.history)
            self.assertEqual(int(restored_owner.optimizer.iterations), 3)
            self.assertIs(actual.noise_teacher_network, restored)
            self.assert_weights_equal(restored.image_embedder, frozen_before)
            self.assertTrue(any(not np.array_equal(before, after)
                                for before, after in zip(trained_before, restored.get_weights())))
            self.assertFalse(restored.trainable)
            self.assertIs(actual.teacher_network, previous)
            self.assert_weights_equal(previous, previous_before)
            self.assert_weights_equal(actual.network, student_before)


# Keep the rebuild lifecycle checks independent of the full model registry.
if __name__ == "__main__":
    unittest.main()
