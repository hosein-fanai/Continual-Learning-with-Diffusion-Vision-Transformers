"""Optional spatial mixing inside encoder and decoder transformer blocks."""

import unittest
from unittest.mock import patch
from typing import Callable

import numpy as np
import tensorflow as tf

from diffusion.layers.block.vision_transformer_block import VisionTransformerBlock
from diffusion.layers.block.di_t_decoder_block import DiTDecoderBlock
from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer


class DiTBlockLocalMixerTests(unittest.TestCase):
    """Check placement, gradients, serialization, widths and optional behavior."""

    def tearDown(self) -> None:
        """Release graph and naming state between independent model checks."""

        tf.keras.backend.clear_session()

    def test_disabled_block_has_original_identity_and_no_mixer_weights(self) -> None:
        """The default remains valid for nonsquare token sequences and no grid."""

        x = tf.ones([2, 3, 8])
        cond = tf.ones([2, 8])
        block = VisionTransformerBlock(dim=8, num_heads=2)
        np.testing.assert_allclose(block((x, cond)).numpy(), x.numpy(), atol=1e-6)
        self.assertIsNone(block.local_mixer)
        self.assertIsNone(block.get_config()["local_mixer_kwargs"])
        self.assertFalse(any("local_mixer" in variable.path for variable in block.weights))

    def test_mixer_variants_train_and_clone_with_finite_gradients(self) -> None:
        """Depthwise, separable and expanded channel paths differentiate correctly."""

        x = tf.random.stateless_normal([2, 4, 8], seed=[7, 11])
        cond = tf.ones([2, 8])
        for block_class in (VisionTransformerBlock, DiTDecoderBlock):
            for mixer_options in (
                {"use_pointwise": False, "kernel_size": 3}, 
                {"use_pointwise": True, "kernel_size": 5}, 
                {"use_pointwise": True, "pointwise_dim_ratio": 2, "kernel_size": 7}
            ):
                with self.subTest(block=block_class.__name__, options=mixer_options):
                    block = block_class(
                        dim=8, num_heads=2, grid_size=2, ln_no_adaptation=True, 
                        local_mixer_kwargs={**mixer_options, "zero_init": False}
                    )
                    with tf.GradientTape() as tape:
                        output = block((x, cond), training=True)
                        loss = tf.reduce_sum(tf.square(output))
                    gradients = tape.gradient(loss, block.local_mixer.trainable_variables)
                    self.assertEqual(output.shape, x.shape)
                    self.assertEqual(block.local_mixer.output_dim, 8)
                    self.assertTrue(all(gradient is not None for gradient in gradients))
                    self.assertTrue(all(bool(tf.reduce_all(tf.math.is_finite(gradient))) for gradient in gradients))
                    self.assertGreater(float(tf.linalg.global_norm(gradients)), 0.)
                    clone = block_class.from_config(block.get_config())
                    clone((x, cond), training=False)
                    clone.set_weights(block.get_weights())
                    np.testing.assert_allclose(
                        clone((x, cond), training=False).numpy(), 
                        block((x, cond), training=False).numpy(), rtol=1e-5, atol=1e-5
                    )

    def test_decoder_mixes_once_after_both_attention_branches(self) -> None:
        """Decoder ordering is self attention, cross attention, local mixer, FFN."""

        block = DiTDecoderBlock(dim=8, num_heads=2, grid_size=2, local_mixer_kwargs={})
        x = tf.ones([2, 4, 8])
        cond = tf.ones([2, 8])
        events = []

        def record(label: str, original: Callable) -> Callable:
            """Wrap one branch without changing its tensor computation."""

            def call(*args: object, **kwargs: object) -> tf.Tensor:
                """Record the branch and forward all values unchanged."""

                events.append(label)
                return original(*args, **kwargs)

            return call

        with patch.object(block, "_call_self_attention", side_effect=record("self", block._call_self_attention)), \
             patch.object(block, "_call_cross_attention", side_effect=record("cross", block._call_cross_attention)), \
             patch.object(block, "_call_local_mixer", side_effect=record("local", block._call_local_mixer)), \
             patch.object(block, "_call_mlp", side_effect=record("mlp", block._call_mlp)):
            block.call((x, cond), values=x, training=True)
        self.assertEqual(events, ["self", "cross", "local", "mlp"])

    def test_prefix_tokens_and_width_changes_preserve_reconstruction(self) -> None:
        """Selected blocks infer their own width and bypass both prefix tokens."""

        options = {"kernel_size": 3, "pointwise_dim_ratio": 2}
        model = DiffusionTransformer(
            image_size=4, channels=1, patch_size=2, dim=8, depth=3, 
            mha_num_heads=2, vit_block_mlp_output_dims={1: 8, 2: 16, 3: 8}, 
            cls_token_type="new_weight", distil_token_type="new_weight", 
            vit_block_local_mixer_kwargs=options, vit_block_local_mixer_ids=[1, -1], 
            timesteps=8, num_classes=2, use_cfg=False
        )
        inputs = (tf.zeros([2, 4, 4, 1]), tf.zeros([2], tf.int32), tf.zeros([2], tf.int32))
        self.assertEqual(model(inputs, training=True).shape, (2, 4, 4, 1))
        self.assertEqual(model.vit_block_local_mixer_ids, [1, 3])
        self.assertEqual(model.layers_dicts[0][model.VTB].local_mixer.prefix_tokens_num, 2)
        self.assertIsNone(model.layers_dicts[1][model.VTB].local_mixer)
        self.assertEqual(model.layers_dicts[2][model.VTB].local_mixer.output_dim, 16)
        self.assertEqual(options, {"kernel_size": 3, "pointwise_dim_ratio": 2})
        clone = DiffusionTransformer.from_config(model.get_config())
        clone.set_weights(model.get_weights())
        np.testing.assert_allclose(
            clone(inputs, training=False).numpy(), model(inputs, training=False).numpy(), atol=1e-6
        )

    def test_all_block_placement_survives_growth_and_round_trip(self) -> None:
        """Default all-block selection also enables newly appended ViT stages."""

        model = DiffusionTransformer(
            image_size=4, channels=1, patch_size=2, dim=8, depth=1, 
            mha_num_heads=2, vit_block_local_mixer_kwargs={"kernel_size": 3}, 
            timesteps=8, num_classes=2, use_cfg=False
        )
        model.add_depths("vision_transformer_block")
        self.assertIsNone(model.vit_block_local_mixer_ids)
        self.assertIsNotNone(model.layers_dicts[-1][model.VTB].local_mixer)
        clone = DiffusionTransformer.from_config(model.get_config())
        self.assertEqual(clone.depth, 2)
        self.assertTrue(all(stage[clone.VTB].local_mixer is not None for stage in clone.layers_dicts))

    def test_structural_mixer_changes_fail_at_block_boundary(self) -> None:
        """Block grids and widths cannot be silently changed by mixer options."""

        for options in (
            {"strides": 2}, {"padding": "valid"}, {"dim": 16}, 
            {"grid_size": 3}, {"mlp_output_dim": 16}, {"circumvent_tokens": 1}
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                VisionTransformerBlock(dim=8, num_heads=2, grid_size=2, local_mixer_kwargs=options)
        with self.assertRaises(AssertionError):
            VisionTransformerBlock(dim=8, num_heads=2, local_mixer_kwargs={})
        with self.assertRaises(AssertionError):
            DiffusionTransformer(
                depth=2, vit_block_ids=[1], vit_block_local_mixer_ids=[2], 
                vit_block_local_mixer_kwargs={}, build=False
            )


# Direct invocation runs only this focused test module.
if __name__ == "__main__":
    unittest.main()
