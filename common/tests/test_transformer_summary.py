"""Regression checks for symbolic summaries of unconditioned classifiers."""

from __future__ import annotations

import unittest

import numpy as np
import tensorflow as tf

from diffusion import DiTClassifier, DiffusionClassifier


class TransformerSummaryTests(unittest.TestCase):
    """Keep shape recording consistent with actual dynamic-batch execution."""

    def tearDown(self) -> None:
        """Release the models created by each independent check."""
        tf.keras.backend.clear_session()

    def assert_summary_shapes(self, network: DiTClassifier) -> None:
        """Require public child outputs and a summary without unknown shapes."""
        self.assertEqual(network.cls_token.output.shape, (None, 1, network.clf_dim))
        for stage in network.clf_layers_dicts[:network.clf_depth]:
            self.assertEqual(
                stage[network.VTB].output.shape,
                (None, network.grid_size ** 2 + 1, network.clf_dim),
            )
        lines = []
        network.summary(print_fn=lines.append)
        self.assertNotIn("?", "\n".join(lines))
        self.assertNotIn("unbuilt", "\n".join(lines))

    def test_reported_configuration_with_wrapper_records_child_shapes(self) -> None:
        """Reproduce the six-block CIFAR configuration and default EMA cloning."""
        network = DiTClassifier(
            patchify_with_cnn=True, image_size=32, channels=3,
            depth=0, cond_type=None, ln_no_adaptation=True,
            dim=128, clf_depth=6, clf_cond_type=None,
            clf_ln_no_adaptation=True, clf_droppath_rate=0.3, classifier_dropout_rate=0.3,
        )
        wrapper = DiffusionClassifier(
            network=network, clf_train_noisy_input_type="clean",
            clf_train_class_input_type="null_class_only",
            mask_by_nulls=False, mask_by_t_threshold=False,
            clf_loss_coef=1.0, noise_loss_coef=0.0,
        )
        self.assertIs(wrapper.network, network)
        self.assert_summary_shapes(network)
        self.assert_summary_shapes(wrapper.ema_network)

    def test_symbolic_outputs_match_eager_and_traced_dynamic_batches(self) -> None:
        """Preserve predictions when the condition width differs from token width."""
        network = DiTClassifier(
            num_classes=2, image_size=4, channels=1, patch_size=2,
            dim=4, cond_dim=6, depth=0, clf_depth=1,
            clf_mha_num_heads=1, cond_type=None, ln_no_adaptation=True,
            clf_cond_type=None, clf_ln_no_adaptation=True,
        )
        self.assert_summary_shapes(network)
        # With conditioning disabled, only the image input is connected.
        symbolic = tf.keras.Model(network.inputs[0], network.outputs)
        specs = (
            tf.TensorSpec((None, 4, 4, 1), tf.float32),
            tf.TensorSpec((None,), tf.int32),
            tf.TensorSpec((None,), tf.uint8),
        )

        @tf.function(input_signature=[specs])
        def traced(inputs: tuple[tf.Tensor, ...]) -> dict[str, tf.Tensor]:
            return network(inputs, training=False)

        for batch_size in (1, 3):
            with self.subTest(batch_size=batch_size):
                inputs = (
                    tf.ones((batch_size, 4, 4, 1)),
                    tf.zeros((batch_size,), dtype=tf.int32),
                    tf.zeros((batch_size,), dtype=tf.uint8),
                )
                eager_outputs = network(inputs, training=False)
                self.assertEqual(eager_outputs["noises"].shape, (batch_size, 4, 4, 1))
                self.assertEqual(eager_outputs["classes"].shape, (batch_size, 2))
                for outputs in (symbolic(inputs[0], training=False), traced(inputs)):
                    for key in ("noises", "classes"):
                        np.testing.assert_allclose(
                            outputs[key].numpy(), eager_outputs[key].numpy(),
                            rtol=1e-5, atol=1e-6,
                        )


if __name__ == "__main__":
    unittest.main()
