"""Regressions for bounding the expanded ensemble classifier batch."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from diffusion import DiTClassifier, DiffusionClassifier
from diffusion.metrics.ensemble_accuracy import EnsembleAccuracy


class EnsemblePredictionBatchTests(unittest.TestCase):
    """Keep input ordering, all prediction heads, and noise across batch caps."""

    def setUp(self) -> None:
        """Build a seeded fixture sensitive to images, timesteps, and labels."""

        self.prediction_calls = []
        self.noise_calls = []
        self.batch_limit = None
        self.images = tf.reshape(tf.linspace(-0.8, 1.1, 12), (3, 2, 2, 1))

        def rates(timesteps: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            """Return distinct schedule rates for the eight fixture timesteps."""

            signal_power = tf.gather(tf.linspace(0.9, 0.1, 8), timesteps)
            return tf.sqrt(signal_power), tf.sqrt(1.0 - signal_power)

        def q_sample(images: tf.Tensor, timesteps: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
            """Record eager noise and apply deterministic noising to each row."""

            # Capture concrete values only while executing eagerly.
            if tf.executing_eagerly():
                self.noise_calls.append((timesteps.numpy(), noise.numpy()))
            offset = tf.cast(timesteps[:, None, None, None], images.dtype) * 0.02
            return images + 0.1 * noise + offset

        def noisify(images: tf.Tensor, timesteps: tf.Tensor, seed: int | None = None) -> tuple[tf.Tensor, None]:
            """Supply the wrapper's unseeded noising interface for completeness."""

            del seed
            return q_sample(images, timesteps, tf.random.normal(tf.shape(images))), None

        def predict(inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], **kwargs: object) -> tuple[object, ...]:
            """Return independent heads while checking the actual inference batch."""

            images, timesteps, labels = inputs
            # Capture concrete values only while executing eagerly.
            if tf.executing_eagerly():
                self.prediction_calls.append(tuple(value.numpy() for value in inputs))
                self.assertIs(kwargs.get("training"), False)
            # Enforce configured caps during eager and graph execution.
            if self.batch_limit is not None:
                assertion = tf.debugging.assert_less_equal(
                    tf.shape(images)[0], self.batch_limit, 
                    message="expanded classifier batch exceeded its cap"
                )
                with tf.control_dependencies([] if assertion is None else [assertion]):
                    images = tf.identity(images)
            feature = tf.reduce_mean(images, axis=(1, 2, 3))
            time = tf.cast(timesteps, tf.float32)
            condition = tf.cast(labels, tf.float32)
            logits = tf.stack((
                feature + 0.07 * time - 0.11 * condition, 
                -0.2 * feature + 0.03 * time + 0.19 * condition, 
                0.3 * feature - 0.09 * time + 0.04 * condition * feature
            ), axis=-1)
            primary = tf.nn.softmax(logits)
            regularizers = [
                tf.nn.softmax(0.5 * logits + tf.constant([0.1, -0.3, 0.2])), 
                None, 
                tf.nn.softmax(-0.2 * logits + tf.constant([-0.2, 0.1, 0.3]))
            ]
            distillation = tf.nn.softmax(-logits + tf.constant([0.2, 0.1, -0.4]))
            return primary, None, [], regularizers, [], distillation

        network = SimpleNamespace(
            use_cfg=True, num_classes=3, num_labels=4, 
            dynamic_num_classes=False, predict_class=predict
        )
        self.wrapper = SimpleNamespace(
            timesteps=8, get_network=lambda name: network, get_noise_and_signal_rates=rates, 
            prepare_images=tf.identity, q_sample=q_sample, noisify=noisify, seed=19
        )

    def make_metric(self, **kwargs: object) -> EnsembleAccuracy:
        """Construct a metric with retained timesteps spanning a partial chunk."""

        options = dict(max_t=8, t_range_drop_rate=0.375, t_chunk_size=3)
        options.update(kwargs)
        return EnsembleAccuracy(self.wrapper, **options)

    def test_invalid_prediction_batch_sizes_are_rejected(self) -> None:
        """Reject nonpositive and nonintegral caps while accepting automatic sizing."""

        for size in (True, False, 0, -1, 1.5, 3.0, np.inf, np.nan, "3"):
            with self.subTest(size=size), self.assertRaisesRegex(
                ValueError, "prediction_batch_size"
            ):
                self.make_metric(prediction_batch_size=size)
        for size in (None, 1, np.int64(3)):
            with self.subTest(size=size):
                self.make_metric(prediction_batch_size=size)

    def test_default_caps_the_expanded_classifier_batch(self) -> None:
        """Use each original batch size before timestep and condition expansion."""

        for mode in ("batched", "chunked"):
            for separate in (False, True):
                metric = self.make_metric(compute_type=mode, separate_probas=separate)
                for size in (3, 1, 3):
                    with self.subTest(mode=mode, separate=separate, size=size):
                        self.assertIsNone(metric.prediction_batch_size)
                        self.batch_limit = size
                        self.prediction_calls.clear()
                        metric.ensemble_predict(self.images[:size], training=False)
                        sizes = [len(call[0]) for call in self.prediction_calls]
                        self.assertEqual(sizes, [size] * (5 * (4 if separate else 1)))
                        self.assertIsNone(metric.prediction_batch_size)

    def test_caps_preserve_scores_expanded_input_order_and_seeded_noise(self) -> None:
        """Match large-call scores, ordered inputs, and noise with automatic or fixed caps."""

        for mode in ("batched", "chunked"):
            for separate in (False, True):
                options = dict(
                    compute_type=mode, separate_probas=separate, weighted=True, 
                    clf_acc_coef=0.7, ctr_acc_coef=0.2, clf_distil_acc_coef=0.4
                )
                self.batch_limit = None
                self.prediction_calls.clear()
                self.noise_calls.clear()
                expected = self.make_metric(
                    prediction_batch_size=60, **options
                ).ensemble_predict(self.images, training=False).numpy()
                expected_inputs = [
                    np.concatenate([call[index] for call in self.prediction_calls])
                    for index in range(3)
                ]
                expected_noise = list(self.noise_calls)
                self.assertGreater(np.max(np.abs(expected[0] - expected[-1])), 1e-4)
                for cap in (None, 1, 3, 7):
                    with self.subTest(mode=mode, separate=separate, cap=cap):
                        effective_cap = len(self.images) if cap is None else cap
                        self.batch_limit = effective_cap
                        self.prediction_calls.clear()
                        self.noise_calls.clear()
                        tf.random.uniform(tuple([13]))
                        metric = self.make_metric(prediction_batch_size=cap, **options)
                        actual = metric.ensemble_predict(self.images, training=False).numpy()
                        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
                        self.assertEqual(
                            max(len(call[0]) for call in self.prediction_calls), effective_cap
                        )
                        self.assertEqual(metric.prediction_batch_size, cap)
                        for index, expected_input in enumerate(expected_inputs):
                            np.testing.assert_array_equal(
                                np.concatenate([call[index] for call in self.prediction_calls]), 
                                expected_input
                            )
                        self.assertEqual(len(self.noise_calls), len(expected_noise))
                        for actual_noise, original_noise in zip(
                            self.noise_calls, expected_noise
                        ):
                            for actual_value, original_value in zip(actual_noise, original_noise):
                                np.testing.assert_array_equal(actual_value, original_value)

    def test_graph_prediction_caps_conditioning_with_dynamic_batch_size(self) -> None:
        """Reuse a graph across tail batches with automatic or explicit classifier caps."""

        for mode in ("batched", "chunked"):
            for separate in (False, True):
                for cap in (None, 7):
                    with self.subTest(mode=mode, separate=separate, cap=cap):
                        options = dict(
                            compute_type=mode, separate_probas=separate, 
                            clf_acc_coef=0.7, ctr_acc_coef=0.2, clf_distil_acc_coef=0.4
                        )
                        self.batch_limit = None
                        reference = self.make_metric(prediction_batch_size=60, **options)
                        expected = {
                            size: reference.ensemble_predict(
                                self.images[:size], training=False
                            ).numpy()
                            for size in (1, 3)
                        }
                        self.batch_limit = tf.Variable(3, trainable=False, dtype=tf.int32)
                        metric = self.make_metric(prediction_batch_size=cap, **options)

                        @tf.function(input_signature=[tf.TensorSpec((None, 2, 2, 1), tf.float32)])
                        def predict(images: tf.Tensor) -> tf.Tensor:
                            """Trace inference with a dynamic leading image dimension."""

                            return metric.ensemble_predict(images, training=False)

                        for size in (3, 1, 3):
                            self.batch_limit.assign(size if cap is None else cap)
                            np.testing.assert_allclose(
                                predict(self.images[:size]), expected[size], rtol=1e-6, atol=1e-7
                            )
                            self.assertEqual(metric.prediction_batch_size, cap)
                        self.assertEqual(predict.experimental_get_tracing_count(), 1)


class EnsemblePredictionBatchIntegrationTests(unittest.TestCase):
    """Exercise the reported wrapper parameters with a real small DiT."""

    def test_wrapper_evaluation_preserves_predictions_under_the_cap(self) -> None:
        """Preserve real model predictions for the reported evaluation parameters."""

        tf.keras.backend.clear_session()
        previous_policy = tf.keras.mixed_precision.global_policy()
        tf.keras.mixed_precision.set_global_policy("float32")
        self.addCleanup(tf.keras.backend.clear_session)
        self.addCleanup(tf.keras.mixed_precision.set_global_policy, previous_policy)
        tf.keras.utils.set_random_seed(217)
        network = DiTClassifier(
            image_size=4, channels=1, patch_size=2, dim=4, depth=1, 
            mha_num_heads=1, clf_mha_num_heads=1, num_classes=2, timesteps=64
        )
        wrapper = DiffusionClassifier(
            network=network, use_ema=False, test_steps=4, seed=83
        )
        images = tf.reshape(tf.linspace(-1.0, 1.0, 32), (2, 4, 4, 1))
        dataset = [
            (images, tf.constant([0, 1], tf.int32)), 
            (images[:1], tf.constant([0], tf.int32))
        ]
        options = dict(
            separate_probas=True, t_chunk_size=2, max_t=64, 
            t_range_drop_rate=0.75, verbose=False
        )
        original_predict = network.predict_class
        calls = []

        def tracked_predict(inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], **kwargs: object) -> tuple[object, ...]:
            """Record actual network batches and predictions during wrapper evaluation."""

            output = original_predict(inputs, **kwargs)
            self.assertIs(kwargs.get("training"), False)
            calls.append((int(tf.shape(inputs[0])[0]), output[0].numpy()))
            return output

        with patch.object(network, "predict_class", new=tracked_predict):
            expected_accuracy = wrapper.evaluate_ensemble_accuracy(
                dataset, prediction_batch_size=96, **options
            )
            expected_predictions = np.concatenate([call[1] for call in calls])
            self.assertEqual([call[0] for call in calls], [12] * 8 + [6] * 8)
            for cap in (None, 3):
                with self.subTest(cap=cap):
                    calls.clear()
                    cap_options = {} if cap is None else {"prediction_batch_size": cap}
                    actual_accuracy = wrapper.evaluate_ensemble_accuracy(
                        dataset, **options, **cap_options
                    )
                    self.assertEqual(actual_accuracy, expected_accuracy)
                    expected_sizes = [2] * 48 + [1] * 48 if cap is None else [3] * 48
                    self.assertEqual([call[0] for call in calls], expected_sizes)
                    np.testing.assert_allclose(
                        np.concatenate([call[1] for call in calls]), expected_predictions, 
                        rtol=2e-5, atol=2e-6
                    )


# Permit focused execution without test discovery.
if __name__ == "__main__":
    unittest.main()
