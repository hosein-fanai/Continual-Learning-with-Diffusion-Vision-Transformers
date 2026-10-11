"""Keep continual V1 scoring aligned with its configured classifier test view.

Run in an authorized remote TensorFlow environment, separately from notebook kernels.
"""

from unittest import TestCase
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.learner import _predict_diffusion_classes, _reset_task_random_streams
from common.tests.test_clean_classifier_training import CleanClassifierTrainingTests


class ContinualClassifierEvaluationInputTests(TestCase):
    """Score only supplied evaluation rows through the intended clean/noisy route."""

    setUp = CleanClassifierTrainingTests.setUp
    tearDown = CleanClassifierTrainingTests.tearDown
    make_network = CleanClassifierTrainingTests.make_network
    make_wrapper = CleanClassifierTrainingTests.make_wrapper

    def test_cap_zero_and_64_match_wrapper_classifier_inputs_without_label_leakage(self) -> None:
        """The HPO matrix receives the same selected classifier view as native tests."""

        for cap in (0, 64):
            with self.subTest(cap=cap):
                model = self.make_wrapper(
                    network=self.make_network(timesteps=1000), 
                    clf_train_noisy_input_type="noisy", clf_test_noisified_max_timesteps=cap
                )
                predictions = []

                def bounded_noising(images: tf.Tensor, **kwargs: object) -> tuple:
                    """Provide distinct, reproducible supported views for branch inspection."""

                    self.assertEqual(kwargs["min_timesteps"], 0)
                    self.assertEqual(kwargs["max_timesteps"], cap)
                    times = tf.fill([tf.shape(images)[0]], max(0, cap - 1))
                    return images + (0.25 if cap else 0.0), tf.zeros_like(images), times


                def record_prediction(inputs: tuple, **kwargs: object) -> tf.Tensor:
                    """Record classifier inputs and return normalized view-dependent scores."""

                    images, times, conditions = inputs
                    predictions.append((np.asarray(images), np.asarray(times), np.asarray(conditions)))
                    self.assertFalse(kwargs["training"])
                    means = tf.reduce_mean(images, axis=(1, 2, 3))
                    return tf.nn.softmax(tf.stack([means, -means], axis=-1))


                times = tf.zeros([len(self.images)], dtype=tf.int32)
                prepared = (
                    self.images, tf.zeros_like(self.images), times, self.images, 
                    times, times, self.labels
                )
                with patch.object(model, "noisify", side_effect=bounded_noising), \
                     patch.object(model.network, "predict_class", side_effect=record_prediction):
                    expected = model._classifier_inputs(
                        model._append_classifier_inputs(prepared, training=False), training=False
                    )
                    observed = _predict_diffusion_classes(
                        model, self.images.numpy(), self.labels.numpy(), 3
                    )
                self.assertEqual([len(batch[0]) for batch in predictions], [3, 1])
                np.testing.assert_array_equal(np.concatenate([batch[0] for batch in predictions]), expected[0])
                np.testing.assert_array_equal(np.concatenate([batch[1] for batch in predictions]), expected[1])
                np.testing.assert_array_equal(np.concatenate([batch[2] for batch in predictions]), np.zeros(4))
                self.assertEqual(observed.shape, (4, 2))
                np.testing.assert_allclose(observed.sum(axis=1), 1.0)

    def test_real_noising_repeats_at_same_seed_and_ignores_evaluation_labels(self) -> None:
        """Replaying the same task RNG preserves noisy views without conditioning on labels."""

        model = self.make_wrapper(
            network=self.make_network(timesteps=1000), 
            clf_train_noisy_input_type="noisy", clf_test_noisified_max_timesteps=64
        )
        calls = []

        def record_prediction(inputs: tuple, **kwargs: object) -> tf.Tensor:
            """Capture actual native corruption while avoiding unrelated classifier arithmetic."""

            images, times, conditions = inputs
            calls.append((np.asarray(images), np.asarray(times), np.asarray(conditions)))
            return tf.fill([len(images), 2], 0.5)


        with patch.object(model.network, "predict_class", side_effect=record_prediction):
            _reset_task_random_streams(model, 1441)
            _predict_diffusion_classes(model, self.images.numpy(), self.labels.numpy(), 3)
            first = list(calls)
            calls.clear()
            _reset_task_random_streams(model, 1441)
            _predict_diffusion_classes(model, self.images.numpy(), 1 - self.labels.numpy(), 3)
        for before, after in zip(first, calls):
            for expected, observed in zip(before, after):
                np.testing.assert_array_equal(expected, observed)
            self.assertTrue(np.all(before[1] >= 0))
            self.assertTrue(np.all(before[1] < 64))
            np.testing.assert_array_equal(before[2], np.zeros(len(before[2])))

    def test_clean_policy_ignores_nonzero_cap_and_default_stays_clean(self) -> None:
        """Existing clean policies never acquire a corruption draw from the fix."""

        for options in (
            {"clf_train_noisy_input_type": "clean", "clf_test_noisified_max_timesteps": 64}, 
            {"clf_train_noisy_input_type": "noisy", "clf_test_noisified_max_timesteps": None}
        ):
            with self.subTest(options=options):
                model = self.make_wrapper(network=self.make_network(timesteps=1000), **options)
                with patch.object(model, "noisify", side_effect=AssertionError("Clean scoring must not noisify.")), \
                     patch.object(model.network, "predict_class", return_value=tf.fill([4, 2], 0.5)) as prediction:
                    _predict_diffusion_classes(model, self.images.numpy(), self.labels.numpy(), 4)
                images, times, conditions = prediction.call_args.args[0]
                np.testing.assert_array_equal(images, self.images)
                np.testing.assert_array_equal(times, np.zeros(4))
                np.testing.assert_array_equal(conditions, np.zeros(4))
