"""Regressions for wrapper-owned image preparation during ensemble evaluation."""

import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.tests import test_clean_classifier_training as fixtures
from diffusion import DiffusionClassifier
from diffusion.metrics.ensemble_accuracy import EnsembleAccuracy


class EnsembleResolutionTests(unittest.TestCase):
    """Match training image coordinates before ensemble timestep expansion."""

    setUp = fixtures.CleanClassifierTrainingTests.setUp
    tearDown = fixtures.CleanClassifierTrainingTests.tearDown
    make_network = fixtures.CleanClassifierTrainingTests.make_network

    def make_wrapper(self, **options: object) -> DiffusionClassifier:
        """Create a small seeded wrapper with the normal raw-pixel contract."""

        network = options.pop("network", None)
        return DiffusionClassifier(
            network=self.make_network() if network is None else network, 
            use_ema=False, test_steps=2, seed=811, **options
        )

    def test_preparation_resizes_grayscale_without_random_draws(self) -> None:
        """Share conversion and resize settings in eager and dynamic-batch graphs."""

        wrapper = self.make_wrapper(
            preprocess_type="min-max", resize_method="bilinear", resize_antialias=True
        )
        wrapper.set_current_resolution(2)
        pixels = tf.reshape(tf.cast(tf.range(64) * 4, tf.uint8), (4, 4, 4))
        expected = tf.image.resize(
            tf.cast(pixels[..., None], tf.float32) / 255., 
            (2, 2), method="bilinear", antialias=True
        )
        states = {
            key: stream.state.numpy().copy()
            for key, stream in wrapper._random_streams.items()
        }
        self.assertEqual(wrapper.preprocess(pixels).shape, pixels.shape)
        with patch.object(wrapper, "preprocess", wraps=wrapper.preprocess) as convert:
            actual = wrapper.prepare_images(pixels)
        convert.assert_called_once()
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        prepare = tf.function(
            wrapper.prepare_images, 
            input_signature=[tf.TensorSpec((None, 4, 4), tf.uint8)]
        )
        for batch_size in (1, 4):
            with self.subTest(batch_size=batch_size):
                np.testing.assert_allclose(
                    prepare(pixels[:batch_size]), expected[:batch_size], 
                    rtol=1e-6, atol=1e-7
                )
        self.assertEqual(prepare.experimental_get_tracing_count(), 1)
        for key, stream in wrapper._random_streams.items():
            np.testing.assert_array_equal(stream.state, states[key])

    def test_default_resolution_preserves_pixels_without_resampling(self) -> None:
        """Avoid a resize operation when the wrapper uses its configured image size."""

        wrapper = self.make_wrapper()
        pixels = tf.reshape(tf.linspace(0., 255., 64), (4, 4, 4, 1))
        expected = wrapper.preprocess(pixels)
        with patch.object(tf.image, "resize", side_effect=AssertionError("unneeded resize")):
            actual = wrapper.prepare_images(pixels)
        np.testing.assert_array_equal(actual, expected)

    def test_training_and_ensemble_share_one_image_conversion(self) -> None:
        """Give prediction the same clean images as training without consuming its RNG."""

        wrapper = self.make_wrapper()
        wrapper.set_current_resolution(2)
        pixels = tf.reshape(tf.linspace(0., 255., 64), (4, 4, 4))
        with patch.object(wrapper, "prepare_images", wraps=wrapper.prepare_images) as prepare:
            trained = wrapper.prep_inputs((pixels, self.labels))
        prepare.assert_called_once()
        states = {
            key: stream.state.numpy().copy()
            for key, stream in wrapper._random_streams.items()
        }
        metric = EnsembleAccuracy(wrapper, max_t=2, seed=19)
        with patch.object(wrapper, "prepare_images", wraps=wrapper.prepare_images) as prepare:
            with patch.object(wrapper, "preprocess", wraps=wrapper.preprocess) as convert:
                with patch.object(
                    metric, "ensemble_predict", return_value=tf.one_hot(self.labels, 2)
                ) as predict:
                    metric.test_step(self.labels, pixels)
        prepare.assert_called_once()
        convert.assert_called_once()
        self.assertEqual(predict.call_args.args[0].shape, (4, 2, 2, 1))
        np.testing.assert_array_equal(predict.call_args.args[0], trained[0])
        self.assertIs(predict.call_args.kwargs["training"], False)
        for key, stream in wrapper._random_streams.items():
            np.testing.assert_array_equal(stream.state, states[key])

    def test_cnn_dit_evaluates_original_images_at_active_resolution(self) -> None:
        """Match explicitly resized inputs for seeded chunked and batched evaluation."""

        network = self.make_network(image_size=32, channels=3, patchify_with_cnn=True)
        wrapper = self.make_wrapper(network=network)
        wrapper.set_current_resolution(16)
        pixels = tf.reshape(tf.linspace(0., 255., 2 * 32 * 32 * 3), (2, 32, 32, 3))
        resized = tf.image.resize(
            pixels, (16, 16), method=wrapper.resize_method, 
            antialias=wrapper.resize_antialias
        )
        labels = tf.constant([0, 1], tf.int32)
        original_predict = network.predict_class
        predictions = []

        def tracked_predict(inputs: tuple[tf.Tensor, ...], **kwargs: object) -> tuple:
            """Check inference image dimensions and retain actual network probabilities."""

            self.assertEqual(tuple(inputs[0].shape[1:]), (16, 16, 3))
            self.assertIs(kwargs.get("training"), False)
            output = original_predict(inputs, **kwargs)
            predictions.append(output[0].numpy())
            return output

        options = dict(
            max_t=4, t_range_drop_rate=0.25, t_chunk_size=2, 
            prediction_batch_size=3, separate_probas=True, verbose=False, seed=19
        )
        for mode in ("chunked", "batched"):
            with self.subTest(mode=mode):
                with patch.object(network, "predict_class", new=tracked_predict):
                    predictions.clear()
                    raw_accuracy = wrapper.evaluate_ensemble_accuracy(
                        [(pixels, labels)], compute_type=mode, **options
                    )
                    raw_predictions = np.concatenate(predictions)
                    predictions.clear()
                    resized_accuracy = wrapper.evaluate_ensemble_accuracy(
                        [(resized, labels)], compute_type=mode, **options
                    )
                self.assertEqual(raw_accuracy, resized_accuracy)
                np.testing.assert_allclose(
                    np.concatenate(predictions), raw_predictions, rtol=2e-5, atol=2e-6
                )


# Permit focused execution without importing unrelated test modules.
if __name__ == "__main__":
    unittest.main()
