"""Condition-adaptive vision-transformer residual blocks."""

import tensorflow as tf
from tensorflow.keras import layers

from typing import Any

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
            ``mlp_activation_func`` are set explicitly here. Serialized
            ``use_layer_norm=True`` and ``ln_dim=dim`` values are accepted,
            while conflicting values are rejected. ``ln_no_adaptation=True``
            replaces zero gates with scalar-one gates and makes the initial
            block non-identity.

    Inputs:
        Pair ``(x, cond)`` where ``x`` is floating
        ``[batch, tokens, dim]`` and ``cond`` is floating
        ``[batch, condition_dim]``.

    Outputs:
        Floating token tensor ``[batch, tokens, mlp_output_dim]``; by default
        the shape matches ``x``.

    Serialization:
        ``from_config(get_config())`` is supported. Constructor-controlled
        normalization keys are discarded before the base class is initialized.
    """

    def __init__(
        self, 
        mlp_ratio: float | None = 4, 
        mlp_activation_func: str = "gelu", 
        dim: int = 32, 
        key_dim: int | None = None, 
        value_dim: int | None = None, 
        query_dim: int | None = None, 
        num_heads: int = 4, 
        gate_query_flag: bool = True, 
        drop_prob: float = 0., 
        drop_per_sample: bool = True, 
        **kwargs: Any
    ) -> None:
        """Build attention, feed-forward, residual, and DropPath sublayers.

        Args:
            mlp_ratio (float | None): Optional feed-forward hidden-width ratio.
            mlp_activation_func (str): Keras feed-forward activation name.
            dim (int): Input and normalization feature width.
            key_dim (int | None): Optional per-head query/key width.
            value_dim (int | None): Optional per-head value width.
            query_dim (int | None): Optional attention output width.
            num_heads (int): Positive attention-head count.
            gate_query_flag (bool): Whether the attention gate uses query width.
            drop_prob (float): Stochastic-depth probability in ``[0, 1)``.
            drop_per_sample (bool): Whether each example receives its own path mask.
            **kwargs (Any): Typed :class:`BaseLayer` and Keras layer options.

        Returns:
            ``None``.
        """

        temp_val = (
            kwargs.pop("use_layer_norm", True),
            kwargs.pop("ln_dim", dim),
        )
        # Preserve the block's mandatory adaptive-normalization setting.
        if temp_val[0] is not True:
            raise ValueError("VisionTransformerBlock requires use_layer_norm=True.")
        # Keep the serialized normalization width consistent with ``dim``.
        if temp_val[1] != dim:
            raise ValueError("ln_dim must equal dim for VisionTransformerBlock.")
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
            name="mha",
            dtype=self.dtype_policy
        )
        self.mha_residual_projector = layers.Dense(
            self.query_dim, 
            name="mha_residual_projector",
            dtype=self.dtype_policy
        ) if self.query_dim != self.dim else None
        self.mha_drop_path = DropPath(
            drop_prob=self.drop_prob, 
            per_sample=self.drop_per_sample, 
            name=f"{self.name}/mha_drop_path",
            dtype=self.dtype_policy
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
            name="mlp_residual_projector",
            dtype=self.dtype_policy
        ) if self.mlp_output_dim != self.query_dim else None
        self.mlp_drop_path = DropPath(
            drop_prob=self.drop_prob, 
            per_sample=self.drop_per_sample, 
            name=f"{self.name}/mlp_drop_path",
            dtype=self.dtype_policy
        )

    def _call_self_attention(
        self, 
        x: tf.Tensor, 
        cond: tf.Tensor, 
        queries: tf.Tensor | None, 
        values: tf.Tensor | None, 
        mask: tf.Tensor | None, 
        training: bool | tf.Tensor | None
    ) -> tf.Tensor: # also can perform cross attention
        """Execute the block's first attention residual branch.

        Args:
            x (tf.Tensor): Residual token tensor shaped ``[batch, tokens, dim]``.
            cond (tf.Tensor): Per-example condition tensor ``[batch, condition_dim]``.
            queries (tf.Tensor | None): Optional attention query tensor. ``None`` uses normalized
                ``x``. A supplied tensor should have shape
                ``[batch, tokens, query_dim]`` so residual shapes align.
            values (tf.Tensor | None): Optional key/value tensor ``[batch, source_tokens,
                value_channels]``. ``None`` uses normalized ``x``.
            mask (tf.Tensor | None): Optional boolean or numeric Keras attention mask broadcastable
                to ``[batch, query_tokens, source_tokens]``; one permits and
                zero blocks attention.
            training (bool | tf.Tensor | None): Optional Keras training flag.

        Returns:
            tf.Tensor: Floating gated attention residual,
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
        h = tf.cast(h, x.dtype)
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
        x: tf.Tensor, 
        cond: tf.Tensor, 
        training: bool | tf.Tensor | None
    ) -> tf.Tensor:
        """Execute the condition-gated feed-forward residual branch.

        Args:
            x (tf.Tensor): Floating token tensor shaped
                ``[batch, tokens, query_dim]``.
            cond (tf.Tensor): Floating condition tensor ``[batch, condition_dim]``.
            training (bool | tf.Tensor | None): Optional Keras training flag forwarded to normalization,
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
        inputs: tuple[tf.Tensor, tf.Tensor], 
        queries: tf.Tensor | None = None, 
        values: tf.Tensor | None = None, 
        mask: tf.Tensor | None = None, 
        training: bool | tf.Tensor | None = None
    ) -> tf.Tensor:
        """Apply attention followed by the feed-forward branch.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Pair ``(x, cond)`` following
                the class input contract.
            queries (tf.Tensor | None): Optional replacement attention queries. ``None`` selects
                normalized ``x`` (ordinary self-attention).
            values (tf.Tensor | None): Optional replacement values. ``None`` selects normalized
                ``x``; supplying values enables cross-attention-like behavior.
            mask (tf.Tensor | None): Optional Keras attention mask broadcastable to
                ``[batch, query_tokens, value_tokens]``.
            training (bool | tf.Tensor | None): Optional training flag. Stochastic depth runs only when
                this is true.

        Returns:
            tf.Tensor: Floating tokens shaped
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


def run_self_tests() -> dict[str, str]:
    """Test attention, MLP, residual, mask, and DropPath block behavior.

    Args:
        None.

    Returns:
        dict[str, str]: A one-entry mapping after every finite boolean branch and representative
        dimension, training, error, gradient, and serialization case passes.
    """

    import numpy as np
    import tensorflow as tf


    tf.random.set_seed(808)
    x = tf.random.normal((2, 3, 4))
    condition = tf.random.normal((2, 5))

    identity = VisionTransformerBlock(dim=4, num_heads=2, drop_prob=0.0)
    for training in (None, False, True):
        output = identity((x, condition), training=training)
        assert output.shape == x.shape and output.dtype == tf.float32
        np.testing.assert_allclose(output.numpy(), x.numpy(), atol=1e-6)
    assert identity.key_dim == 2 and identity.query_dim == 4
    assert identity.mha_residual_projector is None
    assert identity.mlp_residual_projector is None

    masks = (
        None, 
        tf.linalg.band_part(tf.ones((3, 3), dtype=tf.bool), -1, 0), 
        tf.ones((2, 3, 3), dtype=tf.float32),
    )
    for gate_query_flag in (False, True):
        for drop_per_sample in (False, True):
            for mlp_ratio in (None, 2):
                block = VisionTransformerBlock(
                    dim=4, 
                    key_dim=1, 
                    value_dim=2, 
                    query_dim=6, 
                    num_heads=2, 
                    gate_query_flag=gate_query_flag, 
                    drop_prob=0.25, 
                    drop_per_sample=drop_per_sample, 
                    mlp_ratio=mlp_ratio, 
                    mlp_output_dim=3, 
                    ln_no_adaptation=True, 
                )
                for mask in masks:
                    output = block(
                        (x, condition), 
                        queries=tf.ones((2, 3, 6)), 
                        values=tf.ones((2, 3, 7)), 
                        mask=mask, 
                        training=True, 
                    )
                    assert output.shape == (2, 3, 3)
                    assert tf.reduce_all(tf.math.is_finite(output))
                assert (block.mha_residual_projector is not None)
                assert (block.mlp_residual_projector is not None)
                expected_gate = 6 if gate_query_flag else 4
                assert block.mha_layer_norm.gate_dim == expected_gate

    external_values = identity(
        (x, condition), 
        values=tf.ones((2, 5, 4)), 
        training=False
    )
    assert external_values.shape == x.shape
    external_queries = identity(
        (x, condition), 
        queries=tf.ones((2, 3, 4)), 
        training=False
    )
    assert external_queries.shape == x.shape

    stochastic = VisionTransformerBlock(
        dim=4, 
        num_heads=2, 
        drop_prob=0.5, 
        drop_per_sample=False, 
        ln_no_adaptation=True, 
        mlp_activation_func="relu", 
    )
    evaluation = stochastic((x, condition), training=False)
    tf.random.set_seed(809)
    training_output = stochastic((x, condition), training=True)
    assert evaluation.shape == training_output.shape == x.shape

    with tf.GradientTape() as tape:
        gradient_output = identity((x, condition), training=True)
        loss = tf.reduce_sum(gradient_output)
    gradients = tape.gradient(loss, identity.trainable_variables)
    assert gradients and all(gradient is not None for gradient in gradients)

    for invalid_probability in (-0.1, 1.0):
        try:
            VisionTransformerBlock(dim=4, drop_prob=invalid_probability)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid stochastic-depth probabilities must fail.")
    try:
        VisionTransformerBlock(dim=4, num_heads=0)
    except (ZeroDivisionError, ValueError):
        pass
    else:
        raise AssertionError("num_heads=0 must fail.")
    try:
        identity(
            (x, condition), 
            mask=tf.ones((2, 4, 5), dtype=tf.bool), 
        )
    except (tf.errors.InvalidArgumentError, ValueError):
        pass
    else:
        raise AssertionError("An incompatible attention mask must fail.")

    config = identity.get_config()
    restored = VisionTransformerBlock.from_config(config)
    assert restored.dim == 4 and restored.num_heads == 2

    dtype_block = VisionTransformerBlock(
        dim=4, 
        num_heads=2, 
        dtype="float64"
    )
    assert dtype_block.compute_dtype == "float64"
    dtype_output = dtype_block((
        tf.ones((1, 3, 4), dtype=tf.float64),
        tf.ones((1, 2), dtype=tf.float64),
    ))
    assert dtype_output.dtype == tf.float64

    return {"VisionTransformerBlock": "passed"}


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
