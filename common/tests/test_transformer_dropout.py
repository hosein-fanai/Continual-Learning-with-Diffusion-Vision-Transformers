"""Behavioral checks for ViT output/MLP dropout and attention dropout."""

from __future__ import annotations

import json
import unittest

import numpy as np
import tensorflow as tf

from common.config import DiTClassifierConfig, DiffusionTransformerConfig
from diffusion import DiTClassifier, DiTDecoder, DiffusionTransformer
from diffusion.layers.block.di_t_decoder_block import DiTDecoderBlock
from diffusion.layers.block.vision_transformer_block import VisionTransformerBlock


class TransformerDropoutTests(unittest.TestCase):
    """Verify stochastic behavior, precision, persistence, and model routing."""

    def setUp(self) -> None:
        """Use small nonconstant tokens so dropping either branch is observable."""
        self.x = tf.reshape(tf.linspace(-2., 3., 160), (4, 5, 8))
        self.cond = tf.ones((4, 8))
        self.options = dict(
            image_size=4, channels=1, patch_size=2, dim=8, depth=1,
            mha_num_heads=2, num_classes=2, timesteps=4, seed=137,
        )

    def tearDown(self) -> None:
        """Release each test's independent Keras state."""
        tf.keras.backend.clear_session()

    def test_dropout_locations_and_zero_defaults(self) -> None:
        """Only MLP/output dropout adds feature masks; attention stays independent."""
        plain = VisionTransformerBlock(dim=8, num_heads=2, ln_no_adaptation=True)
        self.assertIsNone(plain.mha_dropout)
        self.assertEqual(len(plain.mlp.layers), 2)
        np.testing.assert_allclose(
            plain((self.x, self.cond), training=True),
            plain((self.x, self.cond), training=False),
        )
        for ratio, expected_types in (
            (2., ["Dense", "Dropout", "Dense", "Dropout"]),
            (None, ["Dense", "Dropout"]),
        ):
            with self.subTest(mlp_ratio=ratio):
                block = VisionTransformerBlock(
                    dim=8, num_heads=2, mlp_ratio=ratio,
                    dropout_rate=.25, attention_dropout_rate=.125,
                )
                self.assertEqual([type(layer).__name__ for layer in block.mlp.layers],
                                 expected_types)
                self.assertEqual(block.mha_dropout.rate, .25)
                self.assertEqual(block.mha.get_config()["dropout"], .125)

    def test_each_dropout_control_is_training_only_and_seeded(self) -> None:
        """Each rate changes training outputs and reproduces a cloned random stream."""
        for block_type in (VisionTransformerBlock, DiTDecoderBlock):
            for rates in ({"dropout_rate": .5}, {"attention_dropout_rate": .5}):
                with self.subTest(block=block_type.__name__, rates=rates):
                    block = block_type(
                        dim=8, num_heads=2, ln_no_adaptation=True, seed=137, **rates,
                    )
                    clone = block_type.from_config(json.loads(json.dumps(block.get_config())))
                    evaluation = block((self.x, self.cond), training=False)
                    clone((self.x, self.cond), training=False)
                    clone.set_weights(block.get_weights())
                    first = block((self.x, self.cond), training=True)
                    np.testing.assert_allclose(first, clone((self.x, self.cond), training=True))
                    self.assertGreater(float(tf.reduce_max(tf.abs(first - evaluation))), 1e-5)
                    second = block((self.x, self.cond), training=True)
                    self.assertGreater(float(tf.reduce_max(tf.abs(second - first))), 1e-5)
                    np.testing.assert_allclose(evaluation, block((self.x, self.cond), training=False))

    def test_decoder_streams_are_independent_and_zero_gates_stay_identity(self) -> None:
        """Dropout preserves adaLN-Zero initialization and both decoder branches."""
        block = DiTDecoderBlock(
            dim=8, num_heads=2, dropout_rate=.4, attention_dropout_rate=.3, seed=137,
        )
        output = block((self.x, self.cond), training=True)
        np.testing.assert_array_equal(output, self.x)
        seeds = [block.mha._dropout_layer.seed, block.mha2._dropout_layer.seed,
                 block.mha_dropout.seed, block.mha_dropout2.seed]
        seeds.extend(layer.seed for layer in block.mlp.layers
                     if isinstance(layer, tf.keras.layers.Dropout))
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertNotIn(None, seeds)

    def test_graph_gradients_and_compute_policies(self) -> None:
        """Nonzero dropout remains differentiable in float64 and mixed precision."""
        for dtype in ("float64", "mixed_float16"):
            with self.subTest(dtype=dtype):
                block = VisionTransformerBlock(
                    dim=8, num_heads=2, ln_no_adaptation=True,
                    dropout_rate=.25, attention_dropout_rate=.25, dtype=dtype, seed=137,
                )

                @tf.function
                def step(x: tf.Tensor, cond: tf.Tensor) -> tuple:
                    """Differentiate a training-mode block in an independent graph."""
                    with tf.GradientTape() as tape:
                        tape.watch(x)
                        output = block((x, cond), training=True)
                        loss = tf.reduce_mean(tf.square(tf.cast(output, tf.float64)))
                    return output, tape.gradient(loss, [x] + block.trainable_variables)

                output, gradients = step(self.x, self.cond)
                self.assertEqual(output.dtype.name, block.compute_dtype)
                for gradient in gradients:
                    self.assertIsNotNone(gradient)
                    self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(gradient))))

    def test_branch_rates_survive_config_round_trip_and_growth(self) -> None:
        """Route distinct main/classifier settings through initial and appended blocks."""
        network = DiTClassifier(
            **self.options, clf_depth=1, clf_mha_num_heads=2,
            vit_block_dropout_rate=.1, vit_block_attention_dropout_rate=.2,
            clf_vit_block_dropout_rate=.3, clf_vit_block_attention_dropout_rate=.4,
            dropout_rate=.5,
        )
        network.add_depths({"network": "vision_transformer_block",
                            "classifier": "vision_transformer_block"})
        inputs = (tf.ones((2, 4, 4, 1)), tf.zeros((2,), tf.int32), tf.ones((2,), tf.uint8))
        network(inputs, training=False)
        clone = DiTClassifier.from_config(json.loads(json.dumps(network.get_config())))
        clone.set_weights(network.get_weights())
        for model in (network, clone):
            for stages, rates in ((model.layers_dicts, (.1, .2)),
                                  (model.clf_layers_dicts[:-1], (.3, .4))):
                self.assertEqual(len(stages), 2)
                for stage in stages:
                    block = stage[model.VTB]
                    self.assertEqual((block.dropout_rate, block.attention_dropout_rate), rates)
            self.assertEqual([layer.rate for layer in model.classifier.layers
                              if isinstance(layer, tf.keras.layers.Dropout)], [.5])
        np.testing.assert_allclose(network(inputs)["classes"], clone(inputs)["classes"])

    def test_classifier_inheritance_and_typed_configs(self) -> None:
        """Typed settings persist and classifier None values follow set_nones."""
        main_config = DiffusionTransformerConfig(vit_block_dropout_rate=.2,
                                                 vit_block_attention_dropout_rate=.1)
        self.assertEqual(main_config.kwargs()["vit_block_dropout_rate"], .2)
        config = DiTClassifierConfig(
            **{key: value for key, value in self.options.items() if key != "seed"},
            clf_depth=1, clf_mha_num_heads=2,
            vit_block_dropout_rate=.2, vit_block_attention_dropout_rate=.1,
            clf_vit_block_dropout_rate=None, clf_vit_block_attention_dropout_rate=None,
            set_nones=True,
        )
        inherited = DiTClassifier(**config.kwargs())
        block = inherited.clf_layers_dicts[0][inherited.VTB]
        self.assertEqual((block.dropout_rate, block.attention_dropout_rate), (.2, .1))
        independent = DiTClassifier(**self.options, vit_block_dropout_rate=.2,
                                    vit_block_attention_dropout_rate=.1)
        block = independent.clf_layers_dicts[0][independent.VTB]
        self.assertEqual((block.dropout_rate, block.attention_dropout_rate), (0., 0.))

    def test_decoder_model_routes_both_attention_branches(self) -> None:
        """The inherited decoder API supplies both rates to its custom stage factory."""
        decoder = DiTDecoder(
            **self.options, encoder_output_grid_size=2, encoder_output_dim=8,
            encoder_feature_grid_sizes=[2, 2], encoder_feature_dims=[8, 8],
            vit_block_dropout_rate=.2, vit_block_attention_dropout_rate=.1,
        )
        block = decoder.layers_dicts[0][decoder.VTB]
        self.assertIsInstance(block, DiTDecoderBlock)
        self.assertEqual(block.mha.get_config()["dropout"], .1)
        self.assertEqual(block.mha2.get_config()["dropout"], .1)
        self.assertEqual(block.mha_dropout2.rate, .2)

    def test_invalid_rates_are_rejected_even_without_blocks(self) -> None:
        """Reject out-of-range and nonfinite rates before a dormant branch can hide them."""
        for rate in (-.1, 1., float("nan"), float("inf")):
            for name in ("dropout_rate", "attention_dropout_rate"):
                with self.subTest(rate=rate, name=name):
                    with self.assertRaisesRegex(ValueError, name):
                        VisionTransformerBlock(**{name: rate})
                    with self.assertRaisesRegex(AssertionError, name):
                        DiffusionTransformer(depth=0, build=False, **{"vit_block_" + name: rate})
                    with self.assertRaisesRegex(AssertionError, name):
                        DiTClassifier(depth=0, build=False, **{"clf_vit_block_" + name: rate})


# Support direct execution as well as unittest discovery.
if __name__ == "__main__":
    unittest.main()
