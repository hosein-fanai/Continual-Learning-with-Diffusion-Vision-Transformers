"""Verify normalized previous-class CE without changing other objectives."""

import json
import unittest

import numpy as np
import tensorflow as tf

from common.config import DiffusionClassifierConfig
from common.tests import test_clean_classifier_training as fixtures
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier


class OldClassWeightTests(unittest.TestCase):
    """Exercise actual native CE tensors, gradients and constructor persistence."""

    setUp = fixtures.CleanClassifierTrainingTests.setUp
    tearDown = fixtures.CleanClassifierTrainingTests.tearDown
    make_network = fixtures.CleanClassifierTrainingTests.make_network
    make_wrapper = fixtures.CleanClassifierTrainingTests.make_wrapper

    def weighted_wrapper(self, old_ids: list[int] | None = None, weight: float = 10.) -> DiffusionClassifier:
        """Attach a previous two-column snapshot to a four-column student."""

        wrapper = self.make_wrapper(
            network=self.make_network(num_classes=4), clf_old_class_loss_weight=weight
        )
        teacher = self.make_network(num_classes=2)
        teacher.trainable = False
        if old_ids is not None:
            object.__setattr__(teacher, "_diffusion_task_class_ids", tuple(old_ids))
        wrapper.set_teacher_network(teacher)
        return wrapper

    def test_weighted_mask_and_logit_gradients_use_annotated_output_ids(self) -> None:
        """Nonprefix old support, fractional masks and empty masks use one weighted mean."""

        labels = tf.range(4, dtype=tf.int32)
        logits = tf.reshape(tf.linspace(-1., 2., 16), (4, 4))
        fractional = tf.constant([1., .25, 1., 0.])
        for weight in [0., 2., 10.]:
            wrapper = self.weighted_wrapper([1, 3], weight)
            # Primary old-class supervision must not depend on a positive KD coefficient.
            wrapper.previous_teacher_clf_loss_weight = 0.
            for mask in [None, fractional, tf.zeros_like(fractional)]:
                with self.subTest(weight=weight, masked=mask is not None):
                    rows = tf.constant([1., weight, 1., weight])
                    effective = rows if mask is None else rows * mask
                    with tf.GradientTape() as tape:
                        tape.watch(logits)
                        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                        expected = tf.math.divide_no_nan(tf.reduce_sum(ce * effective), tf.reduce_sum(effective))
                    expected_gradient = tape.gradient(expected, logits)
                    with tf.GradientTape() as tape:
                        tape.watch(logits)
                        probabilities = tf.nn.softmax(logits)
                        actual, returned = wrapper.compute_clf_loss(labels, probabilities, mask, training=True)
                    actual_gradient = tape.gradient(actual, logits)
                    self.assertIs(returned, probabilities)
                    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-7)
                    np.testing.assert_allclose(actual_gradient.numpy(), expected_gradient.numpy(), rtol=1e-6, atol=1e-7)

    def test_traced_loss_preserves_weighting(self) -> None:
        """The graph used by native fitting keeps the same normalized row objective."""

        wrapper = self.weighted_wrapper([0, 1])
        labels = tf.range(4, dtype=tf.int32)
        probabilities = tf.nn.softmax(tf.reshape(tf.linspace(-1., 2., 16), (4, 4)))
        expected = wrapper.compute_clf_loss(labels, probabilities, training=True)[0]

        @tf.function
        def loss(values: tf.Tensor) -> tf.Tensor:
            """Trace the same public native loss without introducing another model pass."""

            return wrapper.compute_clf_loss(labels, values, training=True)[0]

        np.testing.assert_allclose(loss(probabilities).numpy(), expected.numpy(), rtol=1e-6, atol=1e-7)

    def test_task_one_evaluation_and_default_weight_keep_native_loss(self) -> None:
        """No previous owner means unweighted CE; evaluation never takes the old-row branch."""

        labels = tf.constant([0, 1, 2, 3])
        probabilities = tf.nn.softmax(tf.reshape(tf.linspace(-1., 2., 16), (4, 4)))
        default = self.weighted_wrapper(weight=1.)
        weighted = self.weighted_wrapper([0, 1])
        task_one = self.make_wrapper(network=self.make_network(num_classes=4), clf_old_class_loss_weight=10.)
        expected = default.compute_clf_loss(labels, probabilities, training=True)[0]
        for model, modes in [(weighted, [False, None]), (task_one, [True, False, None])]:
            for mode in modes:
                actual, returned = model.compute_clf_loss(labels, probabilities, training=mode)
                np.testing.assert_array_equal(actual.numpy(), expected.numpy())
                self.assertIs(returned, probabilities)

    def test_kd_aggregator_keeps_original_mask_and_loss(self) -> None:
        """Changing primary CE weight leaves native old-class KL and its gradient exact."""

        wrapper = self.weighted_wrapper([0, 1], weight=1.)
        labels = tf.range(4, dtype=tf.int32)
        logits = tf.reshape(tf.linspace(-1., 2., 16), (4, 4))
        mask = tf.constant([1., .25, 1., 0.])
        teacher_probabilities = tf.constant([[.8, .2], [.3, .7], [.6, .4], [.5, .5]])
        wrapper.use_clf_kl_loss = False
        wrapper.use_clf_ctr_loss = False
        wrapper.use_clf_distil_loss = True
        wrapper.clf_distil_type = "soft"
        wrapper.clf_distil_scope = "old_classes"
        wrapper.clf_distil_temperature = 1.

        def aggregate() -> tuple:
            """Call the real primary/KD composition on one shared logits tensor."""

            return wrapper.compute_clf_kl_ctr_distil_loss(
                labels, tf.nn.softmax(logits), None, None, clf_loss_mask=mask, clf_train_type="cond", 
                teacher_labels=teacher_probabilities, logits_c={"class_logits": logits}, training=True
            )

        with tf.GradientTape() as tape:
            tape.watch(logits)
            baseline = aggregate()
        expected_gradient = tape.gradient(baseline[4], logits)
        wrapper.clf_old_class_loss_weight = 10.
        with tf.GradientTape() as tape:
            tape.watch(logits)
            weighted = aggregate()
        actual_gradient = tape.gradient(weighted[4], logits)
        np.testing.assert_array_equal(weighted[4].numpy(), baseline[4].numpy())
        np.testing.assert_array_equal(actual_gradient.numpy(), expected_gradient.numpy())
        np.testing.assert_array_equal(weighted[5].numpy(), baseline[5].numpy())
        self.assertNotAlmostEqual(float(weighted[1]), float(baseline[1]), places=6)

    def test_opt_in_rejects_ambiguous_previous_support_and_serializes(self) -> None:
        """Unknown or malformed old membership cannot silently select the wrong rows."""

        wrapper = self.weighted_wrapper()
        labels = tf.range(4, dtype=tf.int32)
        probabilities = tf.fill((4, 4), .25)
        for old_ids in [None, 2, [0, 0], [0, 4], [False, 1]]:
            object.__setattr__(wrapper.teacher_network, "_diffusion_task_class_ids", old_ids)
            with self.assertRaisesRegex(ValueError, "previous-task student output IDs"):
                wrapper.compute_clf_loss(labels, probabilities, training=True)
        self.assertEqual(DiffusionClassifierConfig().clf_old_class_loss_weight, 1.)
        config = json.loads(json.dumps(wrapper.get_config()))
        self.assertEqual(config["clf_old_class_loss_weight"], 10.)
        restored = DiffusionClassifier.from_config(config)
        self.assertEqual(restored.clf_old_class_loss_weight, 10.)


if __name__ == "__main__":
    unittest.main()
