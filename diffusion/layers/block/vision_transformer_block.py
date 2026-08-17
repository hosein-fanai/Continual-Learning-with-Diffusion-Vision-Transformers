"""Condition-adaptive vision-transformer residual blocks."""

from tensorflow.keras import layers

from diffusion.layers.base_layer import BaseLayer
from diffusion.layers.drop_path import DropPath


class VisionTransformerBlock(BaseLayer):
    """Apply gated attention and feed-forward residual transformations.

    Each branch starts with :class:`~diffusion.layers.AdaLNZero`, uses a
    zero-initialized condition gate, and finishes with stochastic depth. With
    adaptive normalization enabled, a newly initialized block is therefore an
    identity when its input/output widths agree. Optional residual projectors
    make attention and MLP width changes possible.

    Although normally used for self-attention, :meth:`call` can substitute
    external ``queries`` and/or ``values``. Their token counts must remain
    compatible with the residual ``x`` because the attention result is added
    back to it.

    Args:
        mlp_ratio: Hidden-width ratio for the feed-forward branch. ``4`` gives
            a ``4 * query_dim`` hidden layer; ``None`` leaves only its final
            dense projection.
        mlp_activation_func: Keras activation for the feed-forward hidden layer.
        dim: Input token width and adaptive-normalization width.
        key_dim: Per-head query/key width. ``None`` uses ``dim // num_heads``;
            the resolved value must be positive.
        value_dim: Optional per-head value width accepted by Keras
            ``MultiHeadAttention``. ``None`` uses ``key_dim``.
        query_dim: Attention residual/output width. ``None`` uses ``dim``.
        num_heads: Positive number of attention heads.
        gate_query_flag: If true, size the attention gate to ``query_dim``;
            otherwise size it to ``dim``. External-query attention normally
            needs the former, while decoder self-attention uses the latter.
        drop_prob: Stochastic-depth probability in ``[0, 1)`` for each branch.
        drop_per_sample: Use independent path masks per example when true, or
            one path decision for the full batch when false.
        **kwargs: Remaining :class:`BaseLayer`/Keras options. Supported layer
            keys include ``ln_mlp_ratio``, ``ln_no_adaptation``, and
            ``mlp_output_dim``; Keras keys include ``name``, ``dtype``, and
            ``trainable``. ``use_layer_norm``, ``ln_dim``, ``mlp_ratio``, and
            ``mlp_activation_func`` are set explicitly here and must not be
            repeated. ``ln_no_adaptation=True`` replaces zero gates with
            scalar-one gates and makes the initial block non-identity.

    Inputs:
        Pair ``(x, cond)`` where ``x`` is floating
        ``[batch, tokens, dim]`` and ``cond`` is floating
        ``[batch, condition_dim]``.

    Outputs:
        Floating token tensor ``[batch, tokens, mlp_output_dim]``; by default
        the shape matches ``x``.

    Serialization:
        The saved config includes ``use_layer_norm`` and ``ln_dim``, but this
        constructor supplies both to :class:`BaseLayer`. Remove those two keys
        from a copied config before calling
        ``VisionTransformerBlock.from_config`` to avoid duplicate-key errors.
    """

    def __init__(
        self, 
        mlp_ratio: int | None = 4, 
        mlp_activation_func: str = "gelu", 
        dim: int = 32, 
        key_dim: int | None = None, 
        value_dim: int | None = None, 
        query_dim: int | None = None, 
        num_heads: int = 4, 
        gate_query_flag: bool = True, 
        drop_prob: float = 0., 
        drop_per_sample: bool = True, 
        **kwargs
    ):
        """Build attention, feed-forward, residual, and DropPath sublayers.

        Arguments and accepted types are documented on the class.

        Returns:
            ``None``.
        """

        super().__init__(
            use_layer_norm=True, 
            ln_dim=dim, 
            mlp_ratio=mlp_ratio, 
            mlp_activation_func=mlp_activation_func, 
            **kwargs
        )
        self._save_init_args(locals())

        self.key_dim = self.dim // self.num_heads if self.key_dim is None else self.key_dim
        self.mlp_output_dim = self.dim if self.mlp_output_dim is None else self.mlp_output_dim
        self.query_dim = self.dim if self.query_dim is None else self.query_dim

        self.mha_layer_norm = self._create_layer_norm(
            gate_dim=self.query_dim if self.gate_query_flag else self.dim, 
            name=f"{self.name}/mha_layer_norm"
        )
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.key_dim, 
            value_dim=self.value_dim, 
            name="mha"
        )
        self.mha_residual_projector = layers.Dense(
            self.query_dim, 
            name="mha_residual_projector"
        ) if self.query_dim != self.dim else None
        self.mha_drop_path = DropPath(
            drop_prob=self.drop_prob, 
            per_sample=self.drop_per_sample, 
            name=f"{self.name}/mha_drop_path"
        )

        self.mlp_layer_norm = self._create_layer_norm(
            dim=self.query_dim, 
            gate_dim=self.mlp_output_dim, 
            name=f"{self.name}/mlp_layer_norm"
        )
        self.mlp = self._create_mlp(
            self.query_dim
        )
        self.mlp_residual_projector = layers.Dense(
            self.mlp_output_dim, 
            name="mlp_residual_projector"
        ) if self.mlp_output_dim != self.query_dim else None
        self.mlp_drop_path = DropPath(
            drop_prob=self.drop_prob, 
            per_sample=self.drop_per_sample, 
            name=f"{self.name}/mlp_drop_path"
        )

    def _call_self_attention(
        self, 
        x, 
        cond, 
        queries, 
        values, 
        mask, 
        training
    ): # also can perform cross attention
        """Execute the block's first attention residual branch.

        Args:
            x: Residual token tensor shaped ``[batch, tokens, dim]``.
            cond: Per-example condition tensor ``[batch, condition_dim]``.
            queries: Optional attention query tensor. ``None`` uses normalized
                ``x``. A supplied tensor should have shape
                ``[batch, tokens, query_dim]`` so residual shapes align.
            values: Optional key/value tensor ``[batch, source_tokens,
                value_channels]``. ``None`` uses normalized ``x``.
            mask: Optional boolean or numeric Keras attention mask broadcastable
                to ``[batch, query_tokens, source_tokens]``; one permits and
                zero blocks attention.
            training: Optional Keras training flag, including for DropPath.

        Returns:
            Floating ``tf.Tensor`` containing the gated attention residual,
            normally shaped ``[batch, tokens, query_dim]``.
        """

        h, gate = self.mha_layer_norm(
            (x, cond), 
            training=training
        )
        h = self.mha(
            query=h if queries is None else queries, 
            value=h if values is None else values, 
            attention_mask=mask, 
            training=training
        )
        x = self.mha_residual_projector(
            x, 
            training=training
        ) if x.shape[-1] != h.shape[-1] else x
        x = x + self.mha_drop_path(
            gate * h, 
            training=training
        )

        return x

    def _call_mlp(
        self, 
        x, 
        cond, 
        training
    ):
        """Execute the condition-gated feed-forward residual branch.

        Args:
            x: Floating token tensor shaped
                ``[batch, tokens, query_dim]``.
            cond: Floating condition tensor ``[batch, condition_dim]``.
            training: Optional Keras training flag forwarded to normalization,
                dense layers, and DropPath.

        Returns:
            ``tf.Tensor`` shaped ``[batch, tokens, mlp_output_dim]``.
        """

        h, gate = self.mlp_layer_norm(
            (x, cond), 
            training=training
        )
        h = self.mlp(
            h, 
            training=training
        )
        x = self.mlp_residual_projector(
            x, 
            training=training
        ) if x.shape[-1] != h.shape[-1] else x
        x = x + self.mlp_drop_path(
            gate * h, 
            training=training
        )

        return x

    def call(
        self, 
        inputs, 
        queries=None, 
        values=None, 
        mask=None, 
        training=None
    ):
        """Apply attention followed by the feed-forward branch.

        Args:
            inputs: Pair ``(x, cond)`` following the class input contract.
            queries: Optional replacement attention queries. ``None`` selects
                normalized ``x`` (ordinary self-attention).
            values: Optional replacement values. ``None`` selects normalized
                ``x``; supplying values enables cross-attention-like behavior.
            mask: Optional Keras attention mask broadcastable to
                ``[batch, query_tokens, value_tokens]``.
            training: Optional training flag. Stochastic depth runs only when
                this is true.

        Returns:
            Floating ``tf.Tensor`` shaped
            ``[batch, tokens, mlp_output_dim]``.
        """

        x, cond = inputs

        x = self._call_self_attention(
            x, cond, 
            queries, 
            values, 
            mask, 
            training
        )
        x = self._call_mlp(
            x, cond, 
            training
        )

        return x
