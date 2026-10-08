"""Regression checks for native Keras summaries of convolutional denoisers."""

import math
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import tensorflow as tf

from diffusion.models.convolution.unet import UNet
from diffusion.models.convolution.unet_classifier import UNetClassifier
from diffusion.models.wrapper.diffusion_model import DiffusionModel


class UNetSummaryTests(unittest.TestCase):
    """Keep reported shapes and weights consistent with real model execution."""

    def setUp(self) -> None:
        """Prepare a small deterministic denoiser configuration."""

        self.original_policy = tf.keras.mixed_precision.global_policy()
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.utils.set_random_seed(219)
        self.options = dict(
            num_classes=2, timesteps=4, image_size=5, channels=1, 
            widths=(2, 3), block_depth=1, bottleneck_width=4, 
            bottleneck_depth=1, image_embedding_dim=2, 
            time_embedding_dim=2, label_embedding_dim=2, use_batch_norm=False
        )

    def tearDown(self) -> None:
        """Release models and restore the caller's numeric policy."""

        tf.keras.mixed_precision.set_global_policy(self.original_policy)
        tf.keras.backend.clear_session()

    def assert_native_summary(self, network: UNet) -> None:
        """Require complete native rows and consistent unique weight accounting."""

        lines = []
        network.summary(print_fn=lines.append)
        report = "\n".join(lines)
        self.assertNotIn("?", report)
        self.assertNotIn("unbuilt", report)
        self.assertTrue(all(layer.built for layer in network.layers))
        self.assertEqual(len(network.weights), len({id(value) for value in network.weights}))
        actual_count = sum(math.prod(value.shape) for value in network.weights)
        self.assertEqual(network.count_params(), actual_count)
        self.assertEqual(sum(layer.count_params() for layer in network.layers), actual_count)
        row_weights = {
            id(value)
            for layer in network.layers
            for value in layer.weights
        }
        self.assertEqual(row_weights, {id(value) for value in network.weights})

    def assert_no_unbuilt_warnings(self, captured: list[warnings.WarningMessage]) -> None:
        """Reject Keras warnings that indicate incomplete layer construction."""

        for warning in captured:
            message = str(warning.message)
            self.assertNotIn("unbuilt state", message)
            self.assertNotIn("does not have a `build()`", message)
            self.assertNotIn("is not able to trace", message)

    def make_inputs(self, batch_size: int, width: int = 5) -> tuple[tf.Tensor, ...]:
        """Return image, timestep, and label inputs with a configurable batch."""

        return (
            tf.reshape(tf.linspace(-1.0, 1.0, batch_size * 5 * width), 
                       (batch_size, 5, width, 1)), 
            tf.zeros(tuple([batch_size]), dtype=tf.int32), 
            tf.ones(tuple([batch_size]), dtype=tf.uint8)
        )

    def activate_output(self, network: UNet) -> None:
        """Make numerical comparisons sensitive to the denoiser's hidden layers."""

        kernel, bias = network.output_projection.get_weights()
        network.output_projection.set_weights([np.full_like(kernel, 0.125), bias])

    def test_notebook_configuration_and_ema_have_complete_native_summaries(self) -> None:
        """Reproduce the CIFAR10 notebook architecture and default EMA cloning."""

        with warnings.catch_warnings(record=True) as construction_warnings:
            warnings.simplefilter("always")
            network = UNet(
                num_classes=10, use_cfg=True, image_size=32, channels=3, 
                widths=(32, 64, 96), block_depth=2, 
                bottleneck_width=128, bottleneck_depth=2, seed=219
            )
            wrapper = DiffusionModel(network=network, seed=219)
        self.assert_no_unbuilt_warnings(construction_warnings)
        self.assertIs(wrapper.network, network)
        self.assertIsNot(wrapper.ema_network, network)
        for candidate in (network, wrapper.ema_network):
            with self.subTest(network=candidate.name):
                self.assert_native_summary(candidate)
                self.assertEqual(candidate.outputs.shape, (None, 32, 32, 3))
        self.assertEqual(network.count_params(), wrapper.ema_network.count_params())

    def test_symbolic_shapes_match_dynamic_batches_and_nonsquare_execution(self) -> None:
        """Compare public symbolic shapes with eager and traced model results."""

        network = UNet(**self.options)
        self.activate_output(network)
        symbolic = tf.keras.Model(network.inputs, network.outputs)
        specs = (
            tf.TensorSpec((None, 5, None, 1), tf.float32), 
            tf.TensorSpec(tuple([None]), tf.int32), 
            tf.TensorSpec(tuple([None]), tf.uint8)
        )

        @tf.function(input_signature=[specs])
        def traced(inputs: tuple[tf.Tensor, ...]) -> tf.Tensor:
            """Evaluate the denoiser with dynamic batch and image width."""

            return network(inputs, training=False)

        for batch_size in (1, 3):
            for width in (5, 7):
                with self.subTest(batch_size=batch_size, width=width):
                    inputs = self.make_inputs(batch_size, width)
                    actual, _, features, _, _ = network(
                        inputs, full_return=True, training=False
                    )
                    self.assertEqual(actual.shape, (batch_size, 5, width, 1))
                    np.testing.assert_allclose(traced(inputs), actual, rtol=1e-5, atol=1e-6)
                    # Compare recorded native-grid shapes only on the native square input.
                    if width == 5:
                        self.assertEqual(network.outputs.shape[1:], actual.shape[1:])
                        for stage, feature in zip(network.layers_dicts, features[1:]):
                            self.assertEqual(stage.output.shape[0], None)
                            self.assertEqual(stage.output.shape[1:], feature.shape[1:])
                        np.testing.assert_allclose(
                            symbolic(inputs, training=False), actual, rtol=1e-5, atol=1e-6
                        )
        before = [value.numpy().copy() for value in network.weights]
        self.assert_native_summary(network)
        for expected, actual in zip(before, network.weights):
            np.testing.assert_array_equal(actual.numpy(), expected)

    def test_mixed_precision_symbolic_graph_matches_eager_predictions(self) -> None:
        """Keep direct symbolic execution faithful under the notebook's mixed policy."""

        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        network = UNet(**self.options)
        self.activate_output(network)
        symbolic = tf.keras.Model(network.inputs, network.outputs)
        self.assert_native_summary(network)
        for batch_size in (1, 3):
            with self.subTest(batch_size=batch_size):
                inputs = self.make_inputs(batch_size)
                expected = network(inputs, training=False)
                actual = symbolic(inputs, training=False)
                self.assertEqual(actual.dtype, tf.float16)
                self.assertEqual(actual.shape, (batch_size, 5, 5, 1))
                self.assertGreater(float(tf.reduce_max(tf.abs(expected))), 0.0)
                np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)

    def test_json_and_weight_checkpoint_roundtrip_preserve_predictions(self) -> None:
        """Restore both symbolic metadata and nonzero learned predictions."""

        network = UNet(**self.options)
        self.activate_output(network)
        inputs = self.make_inputs(3)
        expected = network(inputs, training=False)
        restored = tf.keras.models.model_from_json(network.to_json())
        self.assert_native_summary(restored)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "unet.weights.h5"
            network.save_weights(checkpoint)
            restored.load_weights(checkpoint)
        self.assertEqual(len(restored.weights), len(network.weights))
        for expected_weight, actual_weight in zip(network.weights, restored.weights):
            np.testing.assert_array_equal(actual_weight.numpy(), expected_weight.numpy())
        np.testing.assert_allclose(
            restored(inputs, training=False), expected, rtol=1e-5, atol=1e-6
        )
        self.assert_native_summary(restored)

    def test_progressive_growth_preserves_existing_weights_and_builds_new_summary_rows(self) -> None:
        """Appending a supported convolution still works after symbolic construction."""

        network = UNet(**self.options)
        self.activate_output(network)
        inputs = self.make_inputs(1)
        original_variables = list(network.weights)
        original_values = [value.numpy().copy() for value in original_variables]
        original_depth = network.depth
        original_count = network.count_params()
        growth = network.add_depths("convolution_block")
        self.assertEqual(growth["network"], {
            "before": original_depth, "added": 1, "after": original_depth + 1
        })
        network.build()
        self.assert_native_summary(network)
        self.assertGreater(network.count_params(), original_count)
        self.assertEqual(network(inputs, training=False).shape, (1, 5, 5, 1))
        self.assertEqual(network.layers_dicts[-1].output.shape, (None, 5, 5, 2))
        current_ids = {id(value) for value in network.weights}
        for variable, expected in zip(original_variables, original_values):
            self.assertIn(id(variable), current_ids)
            np.testing.assert_array_equal(variable.numpy(), expected)
        restored = tf.keras.models.model_from_json(network.to_json())
        restored.set_weights(network.get_weights())
        self.assert_native_summary(restored)
        self.assertEqual(restored.depth, network.depth)
        np.testing.assert_allclose(
            restored(inputs, training=False), network(inputs, training=False), 
            rtol=1e-5, atol=1e-6
        )

    def test_classifier_subclass_keeps_default_construction_and_inference(self) -> None:
        """Preserve subclass construction when its classifier uses TensorFlow tracing."""

        network = UNetClassifier(**self.options)
        self.activate_output(network)
        symbolic = tf.keras.Model(network.inputs, network.outputs)
        restored = tf.keras.models.model_from_json(network.to_json())
        restored.set_weights(network.get_weights())
        self.assertTrue(network.built)
        self.assertEqual(network.outputs["noises"].shape, (None, 5, 5, 1))
        self.assertEqual(network.outputs["classes"].shape, (None, 2))
        for batch_size in (1, 3):
            with self.subTest(batch_size=batch_size):
                inputs = self.make_inputs(batch_size)
                expected = network(inputs, training=False)
                self.assertEqual(expected["noises"].shape, (batch_size, 5, 5, 1))
                self.assertEqual(expected["classes"].shape, (batch_size, 2))
                np.testing.assert_allclose(
                    tf.reduce_sum(expected["classes"], axis=-1), 1.0, atol=1e-6
                )
                for candidate in (symbolic, restored):
                    actual = candidate(inputs, training=False)
                    for key in ("noises", "classes"):
                        np.testing.assert_allclose(
                            actual[key], expected[key], rtol=1e-5, atol=1e-6
                        )
                np.testing.assert_allclose(
                    network.predict_class(inputs, training=False), expected["classes"], 
                    rtol=1e-5, atol=1e-6
                )

    def test_regularized_variational_stages_have_complete_native_summaries(self) -> None:
        """Retain summary support when stages return latent tuples and class heads."""

        for variational in (False, True):
            with self.subTest(variational=variational):
                with warnings.catch_warnings(record=True) as construction_warnings:
                    warnings.simplefilter("always")
                    network = UNet(
                        reshaper_kwargs={"add_kl": variational}, 
                        cls_token_regularizer_ids=[None], 
                        **self.options
                    )
                self.assert_no_unbuilt_warnings(construction_warnings)
                self.assert_native_summary(network)
                output, _, _, regularizers, latents = network(
                    self.make_inputs(1), full_return=True, training=False
                )
                self.assertEqual(output.shape, (1, 5, 5, 1))
                self.assertEqual(len(regularizers), network.depth + 1)
                for regularizer in regularizers:
                    self.assertEqual(regularizer.shape, (1, 2))
                self.assertEqual(len(latents), int(variational))
                for mean, log_variance in latents:
                    self.assertEqual(mean.shape, log_variance.shape)


# Run this focused regression module directly when requested.
if __name__ == "__main__":
    unittest.main()




