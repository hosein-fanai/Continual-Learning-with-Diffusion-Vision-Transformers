"""Regression checks for inverse-SNR ensemble timestep dropping."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from diffusion.metrics.ensemble_accuracy import EnsembleAccuracy


class EnsembleTimestepDropTests(unittest.TestCase):
    """Check retained computation, averaging, graph execution, and randomness."""

    def setUp(self):
        self.noise_calls = []
        self.prediction_calls = []
        self.images = tf.ones((3, 2, 2, 1), dtype=tf.float32)
        self.signal_power = np.linspace(0.9, 0.1, 8).astype(np.float32)

        def rates(timesteps):
            power = tf.gather(tf.constant(self.signal_power), timesteps)
            return tf.sqrt(power), tf.sqrt(1.0 - power)

        def q_sample(images, timesteps, noise):
            if tf.executing_eagerly():
                self.noise_calls.append((timesteps.numpy(), noise.numpy()))
            return images + 0.1 * noise

        def noisify(images, timesteps, seed=None):
            return q_sample(images, timesteps, tf.random.normal(tf.shape(images))), None

        def predict(inputs, **kwargs):
            del kwargs
            if tf.executing_eagerly():
                self.prediction_calls.append(inputs[1].numpy())
            times = tf.cast(inputs[1], tf.float32)
            scores = tf.stack((
                0.1 + 0.04 * times,
                0.65 - 0.03 * times,
                0.25 - 0.01 * times,
            ), axis=-1)
            return scores, None, [], [], []

        network = SimpleNamespace(
            use_cfg=True, num_classes=3, num_labels=4,
            dynamic_num_classes=False, predict_class=predict,
        )
        self.wrapper = SimpleNamespace(
            timesteps=8, seed=19, get_network=lambda name: network,
            get_noise_and_signal_rates=rates, q_sample=q_sample, noisify=noisify,
        )

    def make_metric(self, **kwargs):
        options = dict(max_t=8, t_range_drop_rate=0.375, t_chunk_size=3)
        options.update(kwargs)
        return EnsembleAccuracy(self.wrapper, **options)

    def expected_scores(self, selected, weighted):
        rows = np.stack((
            0.1 + 0.04 * selected,
            0.65 - 0.03 * selected,
            0.25 - 0.01 * selected,
        ), axis=-1)
        snr = self.signal_power / (1.0 - self.signal_power)
        weights = snr[selected] if weighted else np.ones(len(selected))
        return np.average(rows, axis=0, weights=weights)

    def test_invalid_drop_rates_are_rejected(self):
        for rate in (-0.01, 1.01, np.inf, -np.inf, np.nan):
            with self.subTest(rate=rate), self.assertRaisesRegex(
                ValueError, "t_range_drop_rate"
            ):
                self.make_metric(t_range_drop_rate=rate)

    def test_removal_count_rounds_down_and_retains_at_least_one(self):
        for rate, expected_count in ((0.0, 8), (0.24, 7), (0.25, 6), (1.0, 1)):
            with self.subTest(rate=rate):
                selected = self.make_metric(t_range_drop_rate=rate)._select_timesteps().numpy()
                self.assertEqual(len(selected), expected_count)
                self.assertTrue(np.all(np.diff(selected) > 0))
                self.assertTrue(np.all((selected >= 0) & (selected < 8)))

    def test_zero_removal_count_needs_no_schedule_or_selection_rng(self):
        for max_t, rate in ((8, 0.0), (8, 0.01), (1, 1.0)):
            with self.subTest(max_t=max_t, rate=rate):
                metric = self.make_metric(max_t=max_t, t_range_drop_rate=rate)
                with patch.object(
                    metric, "_get_softmax_log_snr", side_effect=AssertionError("schedule")
                ), patch.object(
                    tf.random, "uniform", side_effect=AssertionError("stateful RNG")
                ), patch.object(
                    tf.random, "stateless_uniform", side_effect=AssertionError("stateless RNG")
                ):
                    np.testing.assert_array_equal(metric._select_timesteps(), np.arange(max_t))

    def test_only_retained_timesteps_are_computed_and_averaged(self):
        for mode in ("batched", "chunked"):
            for weighted in (False, True):
                for separate in (False, True):
                    with self.subTest(mode=mode, weighted=weighted, separate=separate):
                        metric = self.make_metric(
                            compute_type=mode, weighted=weighted, separate_probas=separate,
                        )
                        selected = metric._select_timesteps().numpy()
                        self.noise_calls.clear()
                        self.prediction_calls.clear()
                        scores = metric.ensemble_predict(self.images).numpy()
                        expected = self.expected_scores(selected, weighted)
                        if separate:
                            expected = np.exp(2.0 * expected)
                            expected /= expected.sum()
                        np.testing.assert_allclose(
                            scores, np.tile(expected, (3, 1)), rtol=1e-6, atol=1e-7,
                        )
                        np.testing.assert_array_equal(
                            np.stack([times for times, _ in self.noise_calls]),
                            np.repeat(selected[:, None], 3, axis=1),
                        )
                        block_sizes = [5] if mode == "batched" else [3, 2]
                        self.assertEqual(len(self.prediction_calls), len(block_sizes))
                        start = 0
                        for observed, size in zip(self.prediction_calls, block_sizes):
                            expected_ids = np.tile(selected[start:start + size], 3)
                            if separate:
                                expected_ids = np.repeat(expected_ids, 4)
                            np.testing.assert_array_equal(observed, expected_ids)
                            start += size

    def test_seeded_selection_and_original_timestep_noise_are_preserved(self):
        full = self.make_metric(t_range_drop_rate=0.0)
        full.ensemble_predict(self.images)
        original_noise = {int(times[0]): noise for times, noise in self.noise_calls}
        reference_ids = self.make_metric()._select_timesteps().numpy()
        for mode in ("batched", "chunked"):
            for chunk_size in (1, 3, 8):
                with self.subTest(mode=mode, chunk_size=chunk_size):
                    metric = self.make_metric(compute_type=mode, t_chunk_size=chunk_size)
                    tf.random.uniform((37,))
                    np.testing.assert_array_equal(metric._select_timesteps(), reference_ids)
                    self.noise_calls.clear()
                    metric.ensemble_predict(self.images)
                    for times, noise in self.noise_calls:
                        np.testing.assert_array_equal(noise, original_noise[int(times[0])])

    def test_seeded_selection_does_not_advance_stateful_rng(self):
        metric = self.make_metric()
        tf.random.set_seed(57)
        expected = tf.random.uniform((16,)).numpy()
        tf.random.set_seed(57)
        metric._select_timesteps()
        np.testing.assert_array_equal(tf.random.uniform((16,)), expected)

    def test_graph_prediction_supports_dynamic_batch_size(self):
        for mode in ("batched", "chunked"):
            for weighted in (False, True):
                with self.subTest(mode=mode, weighted=weighted):
                    metric = self.make_metric(compute_type=mode, weighted=weighted)
                    selected = metric._select_timesteps().numpy()
                    predict = tf.function(
                        metric.ensemble_predict,
                        input_signature=[tf.TensorSpec((None, 2, 2, 1), tf.float32)],
                    )
                    expected = self.expected_scores(selected, weighted)
                    for batch_size in (1, 3):
                        scores = predict(self.images[:batch_size]).numpy()
                        np.testing.assert_allclose(
                            scores, np.tile(expected, (batch_size, 1)), rtol=1e-6, atol=1e-7,
                        )
                    self.assertEqual(predict.experimental_get_tracing_count(), 1)

    def test_unseeded_sampling_follows_inverse_snr_and_advances(self):
        self.wrapper.seed = None
        snr = np.array([1.0, 4.0, 16.0], dtype=np.float32)

        def rates(timesteps):
            return (
                tf.gather(tf.sqrt(snr / (1.0 + snr)), timesteps),
                tf.gather(tf.sqrt(1.0 / (1.0 + snr)), timesteps),
            )

        self.wrapper.get_noise_and_signal_rates = rates
        metric = self.make_metric(max_t=3, t_range_drop_rate=1.0 / 3.0)

        @tf.function
        def sample_removed_ids():
            # Exactly one of IDs 0, 1, 2 is missing from each retained pair.
            return tf.map_fn(
                lambda _: 3 - tf.reduce_sum(metric._select_timesteps()),
                tf.range(2048), fn_output_signature=tf.int32, parallel_iterations=1,
            )

        tf.random.set_seed(71)
        first = sample_removed_ids().numpy()
        second = sample_removed_ids().numpy()
        self.assertFalse(np.array_equal(first, second))
        observed = np.bincount(np.concatenate((first, second)), minlength=3) / 4096.0
        expected = (1.0 / snr) / (1.0 / snr).sum()
        np.testing.assert_allclose(observed, expected, atol=0.035, rtol=0.0)

    def test_low_precision_endpoint_weights_remain_finite(self):
        def endpoint_rates(timesteps):
            return (
                tf.gather(tf.constant([1.0, 0.5, 0.0], tf.float16), timesteps),
                tf.gather(tf.constant([0.0, np.sqrt(0.75), 1.0], tf.float16), timesteps),
            )

        self.wrapper.get_noise_and_signal_rates = endpoint_rates
        for mode in ("batched", "chunked"):
            for dtype in ("float16", "float32", "float64"):
                with self.subTest(mode=mode, dtype=dtype):
                    metric = self.make_metric(
                        max_t=3, t_range_drop_rate=1.0, weighted=True,
                        compute_type=mode, dtype=dtype,
                    )
                    scores = metric.ensemble_predict(self.images).numpy()
                    self.assertTrue(np.all(np.isfinite(scores)))
                    np.testing.assert_allclose(scores.sum(axis=-1), 1.0, atol=1e-3)
                    # Even retaining only the smallest SNR must yield weight one.
                    with patch.object(metric, "_select_timesteps", return_value=tf.constant([2])):
                        scores = metric.ensemble_predict(self.images).numpy()
                    expected = np.array([0.18, 0.59, 0.23])
                    np.testing.assert_allclose(scores, np.tile(expected, (3, 1)), atol=1e-3)


if __name__ == "__main__":
    unittest.main()
