"""Depthwise-convolutional local mixing for transformer token sequences.

LocalMixer introduces spatial locality through depthwise and optional pointwise
convolution over square patch grids. Shape-preserving paths add a gated residual;
resized paths return local features directly. Prefix tokens, positional merging,
and final projections follow the shared token-layer protocols.
"""

import tensorflow as tf
from tensorflow.keras import layers

from typing import Any

from common.validation import require
from diffusion.layers.embedding.base_embedding import BaseEmbedding


class LocalMixer(BaseEmbedding):
    """Inject convolutional locality into a flattened patch-token sequence.

    The layer reshapes square spatial tokens to an image, applies a depthwise
    convolution and optional 1x1 pointwise projection, then flattens the result.
    When stride and padding preserve the spatial grid, it adds the local
    correction to a residual path; otherwise it returns only the resized local
    features. Position and MLP processing occur after this mixing.

    Args:
        use_layer_norm (bool): Whether to apply condition-adaptive normalization before
            convolution. Disabled normalization uses ``x`` and a scalar-one
            gate directly.
            Defaults to ``True``.
        kernel_size (int): Positive depthwise kernel side length.
            Defaults to ``3``.
        strides (int): Positive depthwise stride. ``1`` enables a residual only when
            the configured padding and kernel preserve the grid; larger values
            spatially reduce the sequence.
            Defaults to ``1``.
        padding (str): Keras padding mode, normally ``"same"`` or ``"valid"``. A
            stride-one residual requires output and input token counts to match,
            so use ``"same"`` (or an effectively size-preserving kernel).
            Defaults to ``'same'``.
        depth_multiplier (int): Positive number of depthwise filters per input channel.
            Defaults to ``1``.
        pointwise_dim_ratio (int): Positive output multiplier for the optional 1x1
            convolution.
            Defaults to ``1``.
        use_pointwise (bool): If true, project depthwise channels to
            ``dim * pointwise_dim_ratio``; otherwise retain
            ``dim * depth_multiplier`` channels.
            Defaults to ``True``.
        zero_init (bool): Keep the initial residual correction at zero. Adaptive
            residual mixing uses its learned zero gate; other configurations
            zero-initialize the depthwise kernel.
            Defaults to ``True``.
        circumvent_tokens (bool | int): Number of leading non-spatial tokens excluded from
            convolution. ``True`` retains the original one-token mode.
            Defaults to ``False``.
        **kwargs (Any): :class:`BaseEmbedding` options. Required keys are ``dim`` and
            ``grid_size``. Positional/MLP options can change the final channel
            width. ``use_layer_norm`` is supplied here. A serialized ``ln_dim`` must
            equal ``dim`` (or be None) when present.

    Inputs:
        Pair ``(x, cond)`` with floating tokens ``[batch, tokens, dim]``. The
        spatial token count must be a perfect square after removing an optional
        leading special tokens. ``cond`` has shape ``[batch, condition_dim]``
        when adaptive normalization is enabled.

    Outputs:
        Floating token tensor at the inferred convolutional grid size. A
        stride-one, same-padded, additive-position configuration preserves the
        input shape; strides, positional concatenation, pointwise ratios, and
        MLP projection can change it.

    Serialization:
        ``from_config(get_config())`` is supported; inherited normalization
        width is reconstructed from ``dim``.

    Attributes:
        add_residual (bool): Whether configured stride and padding preserve the source grid.
        depthwise (tf.keras.layers.DepthwiseConv2D): Spatial mixing of each input channel.
        pointwise (tf.keras.layers.Conv2D | None): Optional channel mixing after depthwise
            convolution.
        output_grid_size (int): Configured spatial output side.
        output_dim (int): Final feature width after all projections and positional merging.
    """

    def __init__(
        self, 
        use_layer_norm: bool = True, 
        kernel_size: int = 3, 
        strides: int = 1, 
        padding: str = "same", 
        depth_multiplier: int = 1, 
        pointwise_dim_ratio: int = 1, 
        use_pointwise: bool = True, 
        zero_init: bool = True, # make this arg enforcing towards every layer for zero output (pos_embed)
        circumvent_tokens: bool | int = False, 
        **kwargs: Any
    ) -> None:
        """Create local convolutions, residual projections, and position data.

        Args:
            use_layer_norm (bool): Whether to normalize before convolution.
                Defaults to ``True``.
            kernel_size (int): Positive depthwise kernel size.
                Defaults to ``3``.
            strides (int): Positive depthwise stride.
                Defaults to ``1``.
            padding (str): Keras ``"same"`` or ``"valid"`` padding.
                Defaults to ``'same'``.
            depth_multiplier (int): Positive depthwise channel multiplier.
                Defaults to ``1``.
            pointwise_dim_ratio (int): Positive pointwise channel multiplier.
                Defaults to ``1``.
            use_pointwise (bool): Whether to apply the pointwise convolution.
                Defaults to ``True``.
            zero_init (bool): Whether to initialize the local correction at
                zero.
                Defaults to ``True``.
            circumvent_tokens (bool | int): Number of leading tokens to
                preserve; ``True`` preserves one.
                Defaults to ``False``.
            **kwargs (Any): Typed :class:`BaseEmbedding` and Keras options.

        Returns:
            None: No value is returned.
        """

        super().__init__(
            use_layer_norm=use_layer_norm, 
            **kwargs
        )
        self._save_init_args(locals())
        require(
            self.grid_size is not None, 
            "LocalMixer requires grid_size."
        )

        # Use the pointwise channel multiplier when enabled; otherwise keep depthwise output
        # channels.
        self.output_dim = self.dim * self.pointwise_dim_ratio if self.use_pointwise \
                        else self.dim * self.depth_multiplier
        # Same padding rounds the strided grid up; valid padding removes the
        # convolution/pooling border.
        self.output_grid_size = (
            self.grid_size + self.strides - 1
        ) // self.strides if self.padding == "same" \
        else (self.grid_size - self.kernel_size) // self.strides + 1
        self.add_residual = self.strides == 1 and \
                            self.output_grid_size == self.grid_size

        # Allocate a residual-width gate for shape-preserving mixing, or zero gate channels
        # for resized paths.
        self.layer_norm = self._create_layer_norm(
            gate_dim=self.output_dim if self.add_residual else 0, 
            return_gate=True
        )

        zero_kernel = self.zero_init and not (
            self.add_residual
            and self.layer_norm is not None
            and not self.ln_no_adaptation
        )

        # Zero-initialize the local kernel when no adaptive zero gate supplies the requested
        # zero correction.
        self.depthwise = layers.DepthwiseConv2D(
            kernel_size=self.kernel_size, 
            strides=self.strides, 
            padding=self.padding, 
            depth_multiplier=self.depth_multiplier, 
            depthwise_initializer="zeros" if zero_kernel else "glorot_uniform", 
            dtype=self.dtype_policy, 
            name=f"{self.name}/depthwise"
        )
        # Create the pointwise channel projection only when pointwise mixing is enabled.
        self.pointwise = layers.Conv2D(
            filters=self.output_dim, 
            kernel_size=1, 
            padding="same", 
            dtype=self.dtype_policy, 
            name=f"{self.name}/pointwise"
        ) if self.use_pointwise else None
        # Create a width projector only for a shape-preserving residual with changed
        # channels.
        self.residual_projector = layers.Dense(
            self.output_dim, 
            dtype=self.dtype_policy, 
            name=f"{self.name}/residual_projector"
        ) if self.dim != self.output_dim and self.add_residual else None
        self.pos_embed = self._create_embeddings(
            embed_dim=self.output_dim, 
            output_grid_size=self.output_grid_size
        )

        # Project the residual to its configured width only when a projector exists.
        residual_token_dim = self.output_dim if self.residual_projector is not None \
                            else self.dim
        # Concatenated positions double the component width; disabled or additive positions
        # preserve it.
        self.output_dim = self.output_dim * 2 if self.pos_embed_type is not None and \
                        self.pos_merger_type == "concat" else self.output_dim

        # Project preserved prefix tokens only when their post-merge channel width differs.
        self.residual_token_projector = layers.Dense(
            self.output_dim, 
            dtype=self.dtype_policy, 
            name=f"{self.name}/residual_token_projector"
        ) if residual_token_dim != self.output_dim and \
        self.circumvent_tokens else None
        self.mlp = self._create_mlp(
            self.output_dim
        )

    def call(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor | None], 
        training: bool | tf.Tensor | None = None
    ) -> tf.Tensor:
        """Mix neighboring spatial tokens and apply the configured residual.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor | None]): Pair ``(x, cond)``
                following the class input contract.
            training (bool | tf.Tensor | None): Optional Keras training flag forwarded to normalization,
                convolutions, positional projection, and MLP layers.
                Defaults to ``None``. Keras resolves the surrounding call context; this flag is
                forwarded to child layers.

        Returns:
            tf.Tensor: with floating compute dtype. For input grid side
            ``g``, ``same`` padding produces ``ceil(g / strides)``; ``valid``
            produces ``floor((g - kernel_size) / strides) + 1``. A leading
            class token is added back when configured.
        """

        x, cond = inputs

        # Use configured normalization; otherwise preserve incoming features and any
        # identity gate.
        h, gate = self.layer_norm(
            (x, cond), 
            training=training
        ) if self.layer_norm is not None else (x, 1.)
        prefix_tokens_num = int(self.circumvent_tokens)
        # Keep configured prefix tokens outside spatial processing and restore them in
        # sequence order.
        h = h[:, prefix_tokens_num:, :] if self.circumvent_tokens else h

        h_shape = tf.shape(h)
        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        input_grid_size = tf.cast(
            tf.sqrt(tf.cast(h_shape[1], dtype=stable_dtype)),
            dtype=tf.int32
        )

        h = tf.reshape(h, (
            h_shape[0], 
            input_grid_size, 
            input_grid_size, 
            self.dim
        ))
        h = self.depthwise(
            h, 
            training=training
        )
        # Mix depthwise output channels only when pointwise convolution is configured.
        h = self.pointwise(
            h, 
            training=training
        ) if self.pointwise is not None else h

        h_shape = tf.shape(h)
        output_grid_size = h_shape[1]

        h = tf.reshape(h, (
            h_shape[0], 
            output_grid_size * output_grid_size, 
            h.shape[-1]
        ))
        # Project the residual to its configured width only when a projector exists.
        x = self.residual_projector(
            x, 
            training=training
        ) if self.residual_projector is not None else x
        # Keep configured prefix tokens outside spatial processing and restore them in
        # sequence order.
        x, x_token = (
            x[:, prefix_tokens_num:, :], 
            x[:, :prefix_tokens_num, :]
        ) if self.circumvent_tokens else (x, None)
        # Add the gated local correction only when the spatial grid is preserved; otherwise
        # return resized features.
        x = x + gate * h if self.add_residual else h

        x = self._pos_merger(
            x, 
            output_grid_size=output_grid_size, 
            training=training
        )
        # Adjust preserved prefix width only when the post-merge projection exists.
        x_token = self.residual_token_projector(
            x_token,
            training=training
        ) if self.residual_token_projector is not None else x_token
        # Keep configured prefix tokens outside spatial processing and restore them in
        # sequence order.
        x = tf.concat([
            x_token,
            x
        ], axis=1) if self.circumvent_tokens else x
        # Apply the final feature projection only when an MLP is configured.
        x = self.mlp(
            x, 
            training=training
        ) if self.mlp is not None else x

        return x


def run_self_tests() -> dict[str, str]:
    """Test all finite and boolean :class:`LocalMixer` control paths.

    Args:
        None.

    Returns:
        dict[str, str]: A one-entry mapping after normalization, convolution, pointwise,
        residual, stride, class-token, position, projection, gradient,
        invalid-shape, and config tests pass.
    """

    import numpy as np


    tf.random.set_seed(5150)
    condition = tf.ones((2, 3), dtype=tf.float32)

    for use_layer_norm in (False, True):
        for use_pointwise in (False, True):
            for zero_init in (False, True):
                for circumvent_tokens in (False, True):
                    # Include a leading special token only in the prefix-preservation test
                    # case.
                    token_count = 17 if circumvent_tokens else 16
                    inputs = tf.ones((2, token_count, 2), dtype=tf.float32)
                    layer = LocalMixer(
                        dim=2, 
                        grid_size=4, 
                        pos_embed_type=None, 
                        use_layer_norm=use_layer_norm, 
                        use_pointwise=use_pointwise, 
                        zero_init=zero_init, 
                        depth_multiplier=2, 
                        pointwise_dim_ratio=2, 
                        circumvent_tokens=circumvent_tokens
                    )
                    output = layer((inputs, condition), training=True)
                    assert output.shape == (2, token_count, 4)
                    assert output.dtype == tf.float32
                    assert (layer.pointwise is not None) is use_pointwise
                    assert (layer.layer_norm is not None) is use_layer_norm
                    assert layer.add_residual

    identity_layer = LocalMixer(
        dim=2, 
        grid_size=4, 
        pos_embed_type=None, 
        use_layer_norm=False, 
        use_pointwise=True, 
        zero_init=True,
        circumvent_tokens=True
    )
    identity_input = tf.random.normal((2, 17, 2))
    np.testing.assert_allclose(
        identity_layer((identity_input, None), training=False).numpy(), 
        identity_input.numpy(), 
        atol=1e-6
    )

    adaptive_identity = LocalMixer(
        dim=2,
        grid_size=4,
        pos_embed_type=None,
        use_layer_norm=True,
        zero_init=True,
    )
    adaptive_input = tf.random.normal((2, 16, 2))
    adaptive_condition = tf.random.normal((2, 3))
    np.testing.assert_allclose(
        adaptive_identity(
            (adaptive_input, adaptive_condition),
            training=False,
        ).numpy(),
        adaptive_input.numpy(),
        atol=1e-6,
    )
    with tf.GradientTape() as tape:
        adaptive_output = adaptive_identity(
            (adaptive_input, adaptive_condition),
            training=True,
        )
        probe = tf.random.stateless_normal(
            tf.shape(adaptive_output),
            seed=(31, 37),
        )
        adaptive_loss = tf.reduce_sum(adaptive_output * probe)
    adaptive_gradients = tape.gradient(
        adaptive_loss,
        adaptive_identity.trainable_variables,
    )
    assert adaptive_gradients and all(
        gradient is not None for gradient in adaptive_gradients
    )
    assert any(
        bool(tf.reduce_any(gradient != 0.0))
        for gradient in adaptive_gradients
    )

    strided_same = LocalMixer(
        dim=2, 
        grid_size=4, 
        strides=2, 
        padding="same", 
        pos_embed_type=None, 
        use_layer_norm=False, 
        use_pointwise=False, 
        depth_multiplier=1, 
        zero_init=False
    )
    same_output = strided_same((tf.ones((2, 16, 2)), None), training=False)
    assert same_output.shape == (2, 4, 2) and not strided_same.add_residual

    strided_valid = LocalMixer(
        dim=2, 
        grid_size=4, 
        kernel_size=3, 
        strides=2, 
        padding="valid", 
        pos_embed_type=None, 
        use_layer_norm=False, 
        zero_init=False
    )
    assert strided_valid((tf.ones((1, 16, 2)), None)).shape == (1, 1, 2)
    assert strided_valid.output_grid_size == 1

    stride_one_valid = LocalMixer(
        dim=2,
        grid_size=4,
        kernel_size=3,
        strides=1,
        padding="valid",
        pos_embed_type=None,
        use_layer_norm=False,
        zero_init=False,
    )
    assert stride_one_valid((tf.ones((1, 16, 2)), None)).shape == (1, 4, 2)
    assert stride_one_valid.output_grid_size == 2
    assert not stride_one_valid.add_residual

    positioned = LocalMixer(
        dim=2, 
        grid_size=4, 
        pos_embed_type="2d_sincos", 
        pos_merger_type="concat", 
        use_layer_norm=True, 
        ln_no_adaptation=True, 
        mlp_output_dim=3, 
        mlp_ratio=2
    )
    assert positioned((tf.ones((2, 16, 2)), None)).shape == (2, 16, 3)

    with tf.GradientTape() as tape:
        gradient_output = strided_same(
            (tf.ones((1, 16, 2)), None), 
            training=True,
        )
        loss = tf.reduce_sum(gradient_output)
    gradients = tape.gradient(loss, strided_same.trainable_variables)
    assert gradients and all(gradient is not None for gradient in gradients)

    try:
        identity_layer((tf.ones((1, 15, 2)), None))
    except (tf.errors.InvalidArgumentError, ValueError):
        pass
    # This invalid case should already have raised: Non-square spatial token counts must
    # fail.
    else:
        raise AssertionError("Non-square spatial token counts must fail.")

    config = identity_layer.get_config()
    restored = LocalMixer.from_config(config)
    assert restored.kernel_size == 3 and restored.strides == 1

    dtype_layer = LocalMixer(
        dim=2, 
        grid_size=4, 
        strides=2, 
        use_layer_norm=False, 
        use_pointwise=False, 
        pos_embed_type=None, 
        dtype="float64"
    )
    dtype_output = dtype_layer((tf.ones((1, 16, 2), tf.float64), None))
    assert dtype_layer.compute_dtype == "float64"
    assert dtype_output.dtype == tf.float64

    return {"LocalMixer": "passed"}


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
