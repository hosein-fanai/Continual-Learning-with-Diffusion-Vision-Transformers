"""Decoder block combining causal self-attention and cross-attention."""

import tensorflow as tf
from tensorflow.keras import layers

from typing import Any

from diffusion.layers.block.vision_transformer_block import VisionTransformerBlock
from diffusion.layers.drop_path import DropPath


class DiTDecoderBlock(VisionTransformerBlock):
    """Decode tokens with self-attention, cross-attention, and an MLP.

    The first inherited attention branch always uses ``x`` for both query and
    value and may receive a causal mask. The second branch attends from the
    normalized decoder state (or explicit ``queries``) to normalized decoder
    state (or explicit ``values``). The cross-attention branch currently does
    not expose a mask. Both attention outputs and the MLP use adaptive gates
    and independent DropPath layers.

    Args:
        **kwargs: :class:`VisionTransformerBlock` constructor options such as
            ``dim``, ``query_dim``, ``num_heads``, ``key_dim``, ``value_dim``,
            ``mlp_ratio``, ``drop_prob``, ``drop_per_sample``,
            ``ln_mlp_ratio``, ``ln_no_adaptation``, ``mlp_output_dim``, and
            standard Keras layer options. ``gate_query_flag`` is fixed to false
            and must not be supplied.

    Inputs:
        Pair ``(x, cond)`` with decoder tokens ``[batch, target_tokens, dim]``
        and conditions ``[batch, condition_dim]``. Optional cross-attention
        values are ``[batch, source_tokens, source_channels]``.

    Outputs:
        Floating decoder tokens shaped
        ``[batch, target_tokens, mlp_output_dim]``.

    Serialization:
        ``from_config(get_config())`` is supported. The decoder discards the
        serialized ``gate_query_flag`` because this branch always fixes it to
        ``False``; its parent handles normalization keys similarly.
    """

    def __init__(
        self, 
        **kwargs: Any
    ) -> None:
        """Create the inherited branches and a second attention branch.

        Args:
            **kwargs (Any): Typed :class:`VisionTransformerBlock` and Keras
                layer options documented by the class contract.

        Returns:
            ``None``.
        """

        gate_query_flag = kwargs.pop("gate_query_flag", False)
        # Preserve the decoder's fixed query-gating convention.
        if gate_query_flag is not False:
            raise ValueError("DiTDecoderBlock fixes gate_query_flag to False.")
        super().__init__(
            gate_query_flag=False, 
            **kwargs
        )

        self.mha_layer_norm2 = self._create_layer_norm(
            gate_dim=self.query_dim, 
            name=f"{self.name}/mha_layer_norm_2"
        )
        self.mha2 = layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.key_dim, 
            value_dim=self.value_dim, 
            name="mha_2",
            dtype=self.dtype_policy,
        )
        self.mha_drop_path2 = DropPath(
            drop_prob=self.drop_prob, 
            per_sample=self.drop_per_sample, 
            name=f"{self.name}/mha_drop_path_2",
            dtype=self.dtype_policy,
        )

    def _call_cross_attention(
        self, 
        x: tf.Tensor, 
        cond: tf.Tensor, 
        queries: tf.Tensor | None, 
        values: tf.Tensor | None, 
        mask: tf.Tensor | None, 
        training: bool | tf.Tensor | None
    ) -> tf.Tensor:
        """Apply condition-gated cross-attention as a residual update.

        Args:
            x (tf.Tensor): Decoder residual tensor ``[batch, target_tokens, dim]``.
            cond (tf.Tensor): Per-example condition tensor ``[batch, condition_dim]``.
            queries (tf.Tensor | None): Optional query tensor
                ``[batch, target_tokens, query_dim]``. ``None`` uses normalized
                ``x``.
            values (tf.Tensor | None): Optional source tensor
                ``[batch, source_tokens, source_channels]``. ``None`` uses
                normalized ``x``, reducing this branch to another self-attention
                update.
            mask (tf.Tensor | None): Optional Keras attention mask broadcastable to
                ``[batch, target_tokens, source_tokens]``. The public
                :meth:`call` currently passes ``None``.
            training (bool | tf.Tensor | None): Optional Keras training flag.

        Returns:
            ``tf.Tensor`` with the residual shape, normally
            ``[batch, target_tokens, query_dim]``.
        """

        h, gate = self.mha_layer_norm2(
            (x, cond), 
            training=training
        )
        h = self.mha2(
            query=h if queries is None else queries, 
            value=h if values is None else values, 
            attention_mask=mask, 
            training=training
        )
        h = tf.cast(h, x.dtype)
        x = self.mha_residual_projector(
            x, 
            training=training
        ) if x.shape[-1] != h.shape[-1] else x
        x = x + self.mha_drop_path2(
            gate * h, 
            training=training
        )

        return x

    def call(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor], 
        queries: tf.Tensor | None = None, 
        values: tf.Tensor | None = None, 
        causal_mask: tf.Tensor | None = None, 
        training: bool | tf.Tensor | None = None
    ) -> tf.Tensor:
        """Run causal self-attention, cross-attention, then the MLP.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Pair ``(x, cond)`` following
                the class input contract.
            queries (tf.Tensor | None): Optional replacement queries for only the second attention
                branch; token count must match the residual sequence.
            values (tf.Tensor | None): Optional encoder/source values for the second attention
                branch. ``None`` uses the current decoder tokens.
            causal_mask (tf.Tensor | None): Optional self-attention mask broadcastable to
                ``[batch, target_tokens, target_tokens]``. A lower-triangular
                boolean mask implements autoregressive attention.
            training (bool | tf.Tensor | None): Optional training flag.

        Returns:
            tf.Tensor: Floating decoder tokens shaped
            ``[batch, target_tokens, mlp_output_dim]``.
        """

        x, cond = inputs

        x = self._call_self_attention(
            x, cond, 
            None, None, 
            causal_mask, 
            training
        )
        x = self._call_cross_attention(
            x, cond, 
            queries, 
            values, 
            None, 
            training
        )
        x = self._call_mlp(
            x, cond, 
            training
        )

        return x


def run_self_tests() -> dict[str, str]:
    """Test self/cross attention and MLP paths of :class:`DiTDecoderBlock`.

    Args:
        None.

    Returns:
        dict[str, str]: A one-entry success mapping after causal-mask, external-source,
        dimensions, normalization, stochastic-depth, gradient, constructor
        conflict, and serialization checks pass.
    """

    import numpy as np
    import tensorflow as tf


    tf.random.set_seed(909)
    x = tf.random.normal((2, 3, 4))
    condition = tf.random.normal((2, 5))
    causal_mask = tf.linalg.band_part(
        tf.ones((3, 3), dtype=tf.bool), -1, 0
    )

    identity = DiTDecoderBlock(dim=4, num_heads=2)
    for mask in (None, causal_mask, tf.ones((2, 3, 3), tf.float32)):
        for training in (False, True):
            output = identity(
                (x, condition), 
                causal_mask=mask, 
                training=training
            )
            assert output.shape == x.shape
            np.testing.assert_allclose(
                output.numpy(), x.numpy(), atol=1e-6
            )

    for drop_per_sample in (False, True):
        for mlp_ratio in (None, 2):
            decoder = DiTDecoderBlock(
                dim=4, 
                query_dim=6, 
                key_dim=1, 
                value_dim=2, 
                num_heads=2, 
                mlp_ratio=mlp_ratio, 
                mlp_output_dim=3, 
                drop_prob=0.25, 
                drop_per_sample=drop_per_sample, 
                ln_no_adaptation=True
            )
            output = decoder(
                (x, condition), 
                queries=tf.ones((2, 3, 6)), 
                values=tf.ones((2, 5, 7)), 
                causal_mask=causal_mask, 
                training=True, 
            )
            assert output.shape == (2, 3, 3)
            assert tf.reduce_all(tf.math.is_finite(output))
            assert decoder.mha_layer_norm2.gate_dim == 6
            assert decoder.mha_drop_path2.per_sample is drop_per_sample

    cross_self = identity(
        (x, condition), 
        queries=None, 
        values=tf.ones((2, 5, 4)), 
        causal_mask=causal_mask, 
        training=False, 
    )
    assert cross_self.shape == x.shape

    with tf.GradientTape() as tape:
        gradient_output = identity((x, condition), training=True)
        loss = tf.reduce_sum(gradient_output)
    gradients = tape.gradient(loss, identity.trainable_variables)
    assert gradients and all(gradient is not None for gradient in gradients)

    try:
        DiTDecoderBlock(dim=4, gate_query_flag=True)
    except ValueError:
        pass
    else:
        raise AssertionError("gate_query_flag is constructor-controlled.")
    try:
        identity(
            (x, condition), 
            causal_mask=tf.ones((2, 4, 5), dtype=tf.bool), 
        )
    except (tf.errors.InvalidArgumentError, ValueError):
        pass
    else:
        raise AssertionError("An incompatible causal mask must fail.")

    config = identity.get_config()
    restored = DiTDecoderBlock.from_config(config)
    assert restored.dim == 4 and restored.num_heads == 2

    dtype_decoder = DiTDecoderBlock(
        dim=4, 
        num_heads=2, 
        dtype="float64"
    )
    assert dtype_decoder.compute_dtype == "float64"
    dtype_output = dtype_decoder((
        tf.ones((1, 3, 4), dtype=tf.float64),
        tf.ones((1, 2), dtype=tf.float64),
    ))
    assert dtype_output.dtype == tf.float64

    return {"DiTDecoderBlock": "passed"}


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
