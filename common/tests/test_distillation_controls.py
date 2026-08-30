"""Focused tests for backward-compatible continual distillation controls."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import tensorflow as tf

from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier


class _DistillationHarness:
    """Provide only the state used by the wrapper's KD loss helper."""

    compute_distil_loss = DiffusionClassifier.compute_distil_loss
    get_clf_results_dict = DiffusionClassifier.get_clf_results_dict
    _distillation_metric_mask = DiffusionClassifier._distillation_metric_mask

    def __init__(self) -> None:
        """Initialize deterministic float32 loss helpers.

        Args:
            None.

        Returns:
            None: The minimal distillation state is initialized.
        """

        self.dtype_policy = tf.keras.mixed_precision.Policy("float32")
        self.distil_type = "soft"
        self.distil_temperature = 1.
        self.distil_scope = "current_and_replay"
        self.scce_loss_fn = tf.keras.losses.sparse_categorical_crossentropy
        self.kld_loss_fn = tf.keras.losses.kullback_leibler_divergence
        self.network = SimpleNamespace(dynamic_num_classes=False)
        self.use_clf_kl_loss = False
        self.use_clf_ctr_loss = False
        self.use_distil_loss = True
        self.clf_acc_coef = 1.
        self.ctr_acc_coef = 0.
        self.distil_acc_coef = 0.
        self.clf_loss_tracker = tf.keras.metrics.Mean(name="classifier_loss")
        self.clf_kl_loss_tracker = tf.keras.metrics.Mean(name="clf_kl_loss")
        self.clf_ctr_loss_tracker = tf.keras.metrics.Mean(name="clf_ctr_loss")
        self.distil_loss_tracker = tf.keras.metrics.Mean(name="distil_loss")
        self.total_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="total_accuracy"
        )
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="classifier_accuracy"
        )
        self.clf_ctr_accuracy_tracker = (
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="clf_ctr_accuracy"
            )
        )
        self.distil_accuracy_tracker = (
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="distil_token_accuracy"
            )
        )


class DistillationControlTests(tf.test.TestCase):
    """Check temperature mathematics and continual example scopes."""

    def setUp(self) -> None:
        """Create fixed teacher and student probability batches.

        Args:
            None.

        Returns:
            None: A fresh stateless loss harness and probabilities are ready.
        """

        super().setUp()
        self.wrapper = _DistillationHarness()
        self.teacher = tf.constant([
            [.8, .2],
            [.1, .9],
            [.6, .4],
        ])
        self.student = tf.constant([
            [.6, .4],
            [.3, .7],
            [.2, .8],
        ])

    def test_default_soft_kd_is_historical_direct_kl(self) -> None:
        """The default remains mean KL on the supplied probabilities.

        Args:
            None.

        Returns:
            None: TensorFlow assertions establish exact compatibility.
        """

        actual, returned_predictions = self.wrapper.compute_distil_loss(
            self.teacher,
            self.student,
        )
        expected = tf.reduce_mean(
            tf.keras.losses.kullback_leibler_divergence(
                self.teacher,
                self.student,
            )
        )
        self.assertAllClose(actual, expected)
        self.assertIs(returned_predictions, self.student)

    def test_soft_temperature_uses_both_distributions_and_t_squared(self) -> None:
        """Non-unit soft KD follows the standard temperature-scaled formula.

        Args:
            None.

        Returns:
            None: The implementation matches an independent formula.
        """

        temperature = 2.
        actual, _ = self.wrapper.compute_distil_loss(
            self.teacher,
            self.student,
            distil_temperature=temperature,
        )
        teacher_soft = tf.nn.softmax(
            tf.math.log(self.teacher) / temperature,
            axis=-1,
        )
        student_soft = tf.nn.softmax(
            tf.math.log(self.student) / temperature,
            axis=-1,
        )
        expected = tf.reduce_mean(
            tf.keras.losses.kullback_leibler_divergence(
                teacher_soft,
                student_soft,
            )
        ) * temperature ** 2
        self.assertAllClose(actual, expected)

    def test_soft_temperature_keeps_new_student_classes_out_of_teacher_support(
        self,
    ) -> None:
        """Temperature scaling cannot invent teacher mass for unseen classes.

        Args:
            None.

        Returns:
            None: Expanded-head KD matches an explicitly zero-padded teacher.
        """

        temperature = 2.
        expanded_student = tf.constant([
            [.5, .3, .2],
            [.2, .6, .2],
            [.2, .3, .5],
        ])
        actual, _ = self.wrapper.compute_distil_loss(
            self.teacher,
            expanded_student,
            distil_temperature=temperature,
        )
        teacher_soft = tf.nn.softmax(
            tf.math.log(self.teacher) / temperature,
            axis=-1,
        )
        teacher_soft = tf.pad(teacher_soft, [[0, 0], [0, 1]])
        student_soft = tf.nn.softmax(
            tf.math.log(expanded_student) / temperature,
            axis=-1,
        )
        expected = tf.reduce_mean(
            tf.keras.losses.kullback_leibler_divergence(
                teacher_soft,
                student_soft,
            )
        ) * temperature ** 2
        self.assertAllClose(actual, expected)

    def test_hard_kd_remains_argmax_cross_entropy(self) -> None:
        """Hard KD remains available and is invariant to temperature.

        Args:
            None.

        Returns:
            None: Hard targets match teacher argmax cross-entropy.
        """

        actual, _ = self.wrapper.compute_distil_loss(
            self.teacher,
            self.student,
            distil_type="hard",
            distil_temperature=5.,
        )
        expected = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                tf.argmax(self.teacher, axis=-1),
                self.student,
            )
        )
        self.assertAllClose(actual, expected)

    def test_distillation_scopes_select_expected_rows(self) -> None:
        """Old-class, replay-only, and all-row scopes are distinct.

        Args:
            None.

        Returns:
            None: Each scoped mean matches its explicit selected rows.
        """

        row_losses = tf.keras.losses.kullback_leibler_divergence(
            self.teacher,
            self.student,
        )
        classes = tf.constant([0, 2, 1])
        replay_mask = tf.constant([False, True, True])

        old_classes, _ = self.wrapper.compute_distil_loss(
            self.teacher,
            self.student,
            classes=classes,
            distil_scope="old_classes",
        )
        replay_only, _ = self.wrapper.compute_distil_loss(
            self.teacher,
            self.student,
            replay_mask=replay_mask,
            distil_scope="replay_only",
        )
        all_examples, _ = self.wrapper.compute_distil_loss(
            self.teacher,
            self.student,
            distil_scope="current_and_replay",
        )

        self.assertAllClose(
            old_classes,
            tf.reduce_mean(tf.gather(row_losses, [0, 2])),
        )
        self.assertAllClose(
            replay_only,
            tf.reduce_mean(tf.gather(row_losses, [1, 2])),
        )
        self.assertAllClose(all_examples, tf.reduce_mean(row_losses))

    def test_replay_only_requires_provenance(self) -> None:
        """Replay-only KD fails clearly when provenance was not supplied.

        Args:
            None.

        Returns:
            None: The missing-input error is asserted.
        """

        with self.assertRaisesRegex(ValueError, "replay_mask is required"):
            self.wrapper.compute_distil_loss(
                self.teacher,
                self.student,
                distil_scope="replay_only",
            )

    def test_scoped_metrics_use_selected_example_counts(self) -> None:
        """Aggregate KD loss and accuracy over scoped rows, not batches.

        Args:
            None.

        Returns:
            None: A one-row then three-row scope has the weighted mean 2.5.
        """

        classes = tf.constant([0, 1, 0])
        predictions = tf.one_hot(classes, 2)
        first_mask = tf.constant([True, False, False])
        second_mask = tf.constant([True, True, True])
        common = {
            "clf_loss": tf.constant(0.),
            "classes": classes,
            "classes_pred": predictions,
            "clf_distil_preds": predictions,
            "use_total_loss": False,
            "use_kl_loss": False,
            "use_ctr_loss": False,
            "use_distil_loss": True,
        }
        self.wrapper.get_clf_results_dict(
            **common,
            clf_distil_loss=tf.constant(1.),
            distil_acc_mask=first_mask,
        )
        results = self.wrapper.get_clf_results_dict(
            **common,
            clf_distil_loss=tf.constant(3.),
            distil_acc_mask=second_mask,
        )

        self.assertAllClose(results["distil_loss"], 2.5)
        self.assertAllClose(results["distil_token_accuracy"], 1.)


# Support direct execution in addition to unittest discovery.
if __name__ == "__main__":
    unittest.main()
