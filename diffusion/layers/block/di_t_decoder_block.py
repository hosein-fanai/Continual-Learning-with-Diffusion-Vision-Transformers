"""Decoder block combining causal self-attention and cross-attention."""

from tensorflow.keras import layers

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
        The saved config repeats three constructor-forced keys. Remove
        ``gate_query_flag``, ``use_layer_norm``, and ``ln_dim`` from a copied
        config before calling ``DiTDecoderBlock.from_config``.
    """

    def __init__(
        self,
        **kwargs
    ):
        """Create the inherited branches and a second attention branch.

        Arguments and accepted types are documented on the class.

        Returns:
            ``None``.
        """

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
            name="mha_2"
        )
        self.mha_drop_path2 = DropPath(
            drop_prob=self.drop_prob, 
            per_sample=self.drop_per_sample, 
            name=f"{self.name}/mha_drop_path_2"
        )

    def _call_cross_attention(
        self, 
        x, 
        cond, 
        queries, 
        values, 
        mask, 
        training
    ):
        """Apply condition-gated cross-attention as a residual update.

        Args:
            x: Decoder residual tensor ``[batch, target_tokens, dim]``.
            cond: Per-example condition tensor ``[batch, condition_dim]``.
            queries: Optional query tensor
                ``[batch, target_tokens, query_dim]``. ``None`` uses normalized
                ``x``.
            values: Optional source tensor
                ``[batch, source_tokens, source_channels]``. ``None`` uses
                normalized ``x``, reducing this branch to another self-attention
                update.
            mask: Optional Keras attention mask broadcastable to
                ``[batch, target_tokens, source_tokens]``. The public
                :meth:`call` currently passes ``None``.
            training: Optional Keras training flag, including for DropPath.

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
        inputs, 
        queries=None, 
        values=None, 
        causal_mask=None, 
        training=None
    ):
        """Run causal self-attention, cross-attention, then the MLP.

        Args:
            inputs: Pair ``(x, cond)`` following the class input contract.
            queries: Optional replacement queries for only the second attention
                branch; token count must match the residual sequence.
            values: Optional encoder/source values for the second attention
                branch. ``None`` uses the current decoder tokens.
            causal_mask: Optional self-attention mask broadcastable to
                ``[batch, target_tokens, target_tokens]``. A lower-triangular
                boolean mask implements autoregressive attention.
            training: Optional training flag forwarded to every nested layer.

        Returns:
            Floating ``tf.Tensor`` shaped
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
