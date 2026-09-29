"""Contracts for teacher guidance defaults, per-role masks, and disabled noise KD."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import tensorflow as tf

from common.tests import test_dual_teachers as fixtures
from diffusion import DiffusionClassifier, DiffusionModel


class DualTeacherContractTests(unittest.TestCase):
    """Protect public wrapper behavior when roles are added, removed, or disabled."""

    def setUp(self) -> None:
        """Reuse deterministic native teacher fixtures without duplicating their tests."""

        self.fixture = fixtures.DualTeacherTests()
        self.fixture.setUp()

    def tearDown(self) -> None:
        """Release Keras state and restore the precision policy after each contract."""

        self.fixture.tearDown()

    def test_default_preparation_uses_training_guidance_for_single_and_dual_teachers(self) -> None:
        """Only an explicit evaluation mode chooses the evaluation guidance scale."""

        for dual in (False, True):
            with self.subTest(dual=dual):
                model = self.fixture.noise_model(train_cfg_scale=2., test_cfg_scale=7.)
                # Removing the current teacher must not change preparation-mode semantics.
                if not dual:
                    model.set_current_teacher_network(None)
                for mode, expected in ((None, 2.), (True, 2.), (False, 7.)):
                    with self.subTest(mode=mode):
                        model._preprocess_training = mode
                        with patch.object(
                            model, "_predict_teacher_noise", 
                            return_value=tf.zeros_like(self.fixture.images)
                        ) as predict:
                            model.prep_inputs_map(self.fixture.images, self.fixture.labels)
                        self.assertEqual(predict.call_count, 2 if dual else 1)
                        for call in predict.call_args_list:
                            self.assertEqual(call.kwargs["scale"], expected)

    def test_tuple_noise_targets_require_per_teacher_masks(self) -> None:
        """A two-row mask cannot silently become two scalar teacher masks."""

        model = self.fixture.noise_model()
        student = tf.zeros_like(self.fixture.images[:2])
        targets = (tf.ones_like(student), 2. * tf.ones_like(student))
        with self.assertRaisesRegex(ValueError, "tuple/list"):
            model.compute_distil_noise_loss(
                targets, student, teacher_noise_mask=tf.constant([True, False])
            )
        masks = [tf.constant([True, False]), tf.constant([False, True])]
        actual = model.compute_distil_noise_loss(targets, student, teacher_noise_mask=masks)
        self.assertAlmostEqual(float(actual.numpy()), 5., places=6)

    def test_xla_classifier_losses_match_graph_for_scopes_and_temperatures(self) -> None:
        """Keep exact independent soft targets and gradients across compiled class mappings."""

        model = self.fixture.classifier(previous_teacher_clf_loss_weight=2., 
                                        current_teacher_clf_loss_weight=3.)
        logits = tf.math.log(tf.constant([
            [.1, .2, .3, .4], [.4, .3, .2, .1], 
            [.25, .25, .1, .4], [.1, .1, .7, .1]
        ]))
        targets = (tf.constant([[.8, .2]] * 4), tf.constant([[.25, .75]] * 4))
        for scope in ("task", "all"):
            model.dual_teacher_scope = scope
            for temperature in (1., 2.):
                with self.subTest(scope=scope, temperature=temperature):
                    def loss_and_gradient(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
                        """Differentiate the same mapped teacher objective in each execution mode."""

                        with tf.GradientTape() as tape:
                            tape.watch(values)
                            loss, _ = model.compute_clf_distil_loss(
                                targets, tf.nn.softmax(values), classes=self.fixture.labels, 
                                clf_distil_temperature=temperature, student_logits=values
                            )
                        return loss, tape.gradient(loss, values)

                    expected = tf.function(loss_and_gradient)(logits)
                    actual = tf.function(loss_and_gradient, jit_compile=True)(logits)
                    for graph_value, compiled_value in zip(expected, actual):
                        tf.debugging.assert_all_finite(compiled_value, "Nonfinite XLA KD result")
                        tf.debugging.assert_near(compiled_value, graph_value, atol=1e-6, rtol=1e-5)

    def test_xla_split_training_supports_both_noise_and_class_teacher_roles(self) -> None:
        """Compile one real optimizer step with both teachers represented in both row groups."""

        previous = self.fixture.make_network(num_classes=2, seed=811)
        current = self.fixture.make_network(num_classes=2, seed=812)
        model = DiffusionClassifier(
            network=self.fixture.make_network(num_classes=4, seed=811), 
            teacher_network=previous, current_teacher_network=current, 
            use_ema=False, test_steps=4, scheduler_name="clipped_cosine", 
            p_uncond=0., mask_by_nulls=False, mask_by_t_threshold=False, 
            noise_loss_coef=0., clf_loss_coef=0., 
            noise_distil_loss_coef=1., clf_distil_loss_coef=1., 
            clf_distil_type="soft", clf_distil_temperature=2., clf_train_batch_fraction=.5
        )
        model.set_current_teacher_network(current, class_ids=[2, 3])
        model.compile(optimizer=tf.keras.optimizers.SGD(.001), loss="mse", 
                      run_eagerly=False, jit_compile=True)
        model._preprocess_training = True
        prepared = model.prep_inputs_map(self.fixture.images, self.fixture.labels)
        model._preprocess_training = None
        with patch.object(model, "_classifier_batch_mask", 
                          return_value=tf.constant([True, False, True, False])):
            results = tf.function(model.train_step, jit_compile=True)(prepared)
        self.fixture.assert_finite(results.values())
        self.assertEqual(int(model.optimizer.iterations), 1)
        for role in ("previous", "current"):
            for suffix in ("noise_distil_loss_tracker", "clf_loss_tracker"):
                tracker = getattr(model, role + "_teacher_" + suffix)
                self.assertEqual(float(tracker.count), 1.)

    def test_zero_noise_role_weights_allow_teacher_free_construction_and_clearing(self) -> None:
        """A positive global coefficient does not require teachers for disabled roles."""

        model = DiffusionModel(
            network=self.fixture.noise_network(4), use_ema=False, test_network_name="raw", 
            test_steps=4, noise_distil_loss_coef=1., 
            previous_teacher_noise_loss_weight=0., current_teacher_noise_loss_weight=0., 
            defer_teacher=False
        )
        model.set_teacher_network(None)
        model.set_current_teacher_network(None)
        self.assertIsNone(model.teacher_network)
        self.assertIsNone(model.current_teacher_network)
        self.assertFalse(model.use_noise_distil_loss)
        self.assertFalse(model.map_preprocess)
        self.assertFalse(model.defer_teacher)
        restored = DiffusionModel.from_config(model.get_config())
        self.assertIsNone(restored.teacher_network)
        self.assertIsNone(restored.current_teacher_network)
        self.assertFalse(restored.use_noise_distil_loss)
        self.assertFalse(restored.map_preprocess)
        self.assertFalse(restored.defer_teacher)
        self.assertEqual(restored.previous_teacher_noise_loss_weight, 0.)
        self.assertEqual(restored.current_teacher_noise_loss_weight, 0.)


# Direct invocation runs only this focused contract module.
if __name__ == "__main__":
    unittest.main()
