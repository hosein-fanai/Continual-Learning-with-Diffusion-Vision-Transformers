"""Spatial upsampling for flattened square token grids.

Upsample doubles a square token grid along both spatial axes through interpolation
or learned convolution. Optional prefix tokens bypass resizing and are projected
to the merged channel width before the final MLP.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from typing import Any, Literal, TypeAlias

from common.validation import require
from diffusion.layers.embedding.base_embedding import BaseEmbedding


ScalingMethod: TypeAlias = Literal[
    "cnn_transpose", 
    "interpolate", 
    "cnn_interpolate"
]
"""Available transposed-convolution and interpolation upsampling modes."""


class Upsample(BaseEmbedding):
    """Double a square patch-token grid, optionally preserving a class token.

    Spatial tokens are reshaped to an image-like grid and expanded by a factor
    of two. ``"cnn_transpose"`` learns the resize directly;
    ``"interpolate"`` performs parameter-free interpolation; and
    ``"cnn_interpolate"`` follows interpolation with a learned convolution.
    Positional merging and an optional MLP run after resizing.

    Args:
        use_layer_norm (bool): Whether to normalize/adapt tokens before resizing.
            Defaults to ``True``.
        scaling_method (ScalingMethod): One of ``"cnn_transpose"``, ``"interpolate"``, or
            ``"cnn_interpolate"``.
            Defaults to ``'cnn_transpose'``.
        scaling_interpolation_method (str): Keras ``UpSampling2D`` method used by the
            interpolation modes, commonly ``"nearest"`` or ``"bilinear"``.
            Defaults to ``'nearest'``.
        cnn_dim_ratio (int): Positive integer channel multiplier for modes containing
            a convolution. Pure interpolation preserves ``dim`` channels.
            Defaults to ``1``.
        cnn_kernel_size (int): Positive kernel side for the transposed or post-resize
            convolution.
            Defaults to ``2``.
        cnn_activation_func (str): Keras convolution activation; ``"linear"`` leaves
            results unbounded.
            Defaults to ``'linear'``.
        circumvent_tokens (bool | int): Number of leading non-spatial tokens excluded from
            resizing and prepended afterward. ``True`` preserves one token.
            Defaults to ``False``.
        **kwargs (Any): :class:`BaseEmbedding` options. ``dim`` and positive
            ``grid_size`` are required. Positional, MLP, normalization, and
            standard Keras options are accepted; ``use_layer_norm`` is supplied here. A serialized ``ln_dim`` must
            equal ``dim`` (or be None) when present.

    Inputs:
        Pair ``(x, cond)``. ``x`` is floating ``[batch, tokens, dim]`` and its
        spatial token count must be a perfect square (after removing an optional
        leading class token). ``cond`` is required only for adaptive
        normalization.

    Outputs:
        Floating tensor with four times as many spatial tokens and configured
        ``output_dim`` channels. An excluded class token is prepended unchanged
        in count; positional concatenation and an MLP may alter channel width.

    Serialization:
        ``from_config(get_config())`` is supported; inherited normalization
        width is reconstructed from ``dim``.

    Attributes:
        scaling_layer (tf.keras.layers.Layer): Factor-two spatial
            interpolation/convolution pipeline.
        output_grid_size (int): Twice the configured source-grid side.
        token_projector (tf.keras.layers.Dense | None): Optional width adjustment for
            bypassed prefix tokens.
        output_dim (int): Final channel count after position merging and optional MLP.
    """

    def __init__(
        self, 
        use_layer_norm: bool = True, 
        scaling_method: ScalingMethod = "cnn_transpose", 
        scaling_interpolation_method: str = "nearest", 
        cnn_dim_ratio: int = 1, 
        cnn_kernel_size: int = 2, 
        cnn_activation_func: str = "linear", 
        circumvent_tokens: bool | int = False, 
        **kwargs: Any
    ) -> None:
        """Create the factor-two scaler, position table, and projections.

        Args:
            use_layer_norm (bool): Whether to normalize before scaling.
                Defaults to ``True``.
            scaling_method (ScalingMethod): Learned or interpolation resize mode.
                Defaults to ``'cnn_transpose'``.
            scaling_interpolation_method (str): Keras interpolation method.
                Defaults to ``'nearest'``.
            cnn_dim_ratio (int): Positive convolutional channel multiplier.
                Defaults to ``1``.
            cnn_kernel_size (int): Positive convolution kernel size.
                Defaults to ``2``.
            cnn_activation_func (str): Keras convolution activation.
                Defaults to ``'linear'``.
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
            self.grid_size is not None and self.grid_size >= 1, 
            "grid_size must be positive."
        )

        self.output_grid_size = self.grid_size * 2

        self.layer_norm = self._create_layer_norm(
            return_gate=False
        )

        name = f"{self.name}/scaling_layer"
        # Learn spatial and channel enlargement with transposed convolution.
        if self.scaling_method == "cnn_transpose":
            self.output_dim = self.dim * self.cnn_dim_ratio
            self.scaling_layer = layers.Conv2DTranspose(
                filters=self.output_dim, 
                kernel_size=self.cnn_kernel_size, 
                strides=2, 
                padding="same", 
                activation=self.cnn_activation_func, 
                dtype=self.dtype_policy, 
                name=name
            )
        # Preserve channels for parameter-free interpolation.
        elif self.scaling_method == "interpolate":
            self.output_dim = self.dim
            self.scaling_layer = layers.UpSampling2D(
                size=(2, 2), 
                interpolation=self.scaling_interpolation_method, 
                dtype=self.dtype_policy, 
                name=name
            )
        # Refine interpolated features with a learned channel projection.
        elif self.scaling_method == "cnn_interpolate":
            self.output_dim = self.dim * self.cnn_dim_ratio
            self.scaling_layer = models.Sequential([
                layers.UpSampling2D(
                    size=(2, 2), 
                    interpolation=self.scaling_interpolation_method, 
                    dtype=self.dtype_policy, 
                    name=f"{name}_interpolation"
                ), 
                layers.Conv2D(
                    filters=self.output_dim, 
                    kernel_size=self.cnn_kernel_size, 
                    padding="same", 
                    activation=self.cnn_activation_func, 
                    dtype=self.dtype_policy, 
                    name=f"{name}_convolution"
                )
            ], name=name)
        # Reject any scaling strategy outside the implemented modes.
        else:
            raise ValueError(
                f"scaling_method method can only be one of {ScalingMethod}."
            )

        self.pos_embed = self._create_embeddings(
            embed_dim=self.output_dim, 
            output_grid_size=self.output_grid_size
        )

        # Concatenated positions double the component width; disabled or additive positions
        # preserve it.
        self.output_dim = self.output_dim * 2 if self.pos_embed_type is not None and \
                        self.pos_merger_type == "concat" else self.output_dim

        # Project bypassed tokens only when spatial processing changes their channel width.
        self.token_projector = layers.Dense(
            self.output_dim, 
            dtype=self.dtype_policy, 
            name=f"{self.name}/token_projector"
        ) if self.dim != self.output_dim else None
        self.mlp = self._create_mlp(
            self.output_dim
        )

    def call(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor | None], 
        training: bool | tf.Tensor | None = None
    ) -> tf.Tensor:
        """Upsample a token grid by a factor of two per spatial axis.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor | None]): Pair ``(x, cond)``
                following the class input contract.
            training (bool | tf.Tensor | None): Optional Keras training flag forwarded to every nested
                normalization, convolution, and dense layer.
                Defaults to ``None``. Keras resolves the surrounding call context; this flag is
                forwarded to child layers.

        Returns:
            tf.Tensor: Floating tokens shaped
            ``[batch, (2 * grid) ** 2, output_dim]``, plus one token when
            ``circumvent_tokens=True`` (or the configured integer count).
            ``grid`` is inferred dynamically from the patch-token count.
        """

        x, cond = inputs
    
        # Use configured normalization; otherwise preserve incoming features and any
        # identity gate.
        x = self.layer_norm(
            (x, cond), 
            training=training
        ) if self.layer_norm is not None else x
        prefix_tokens_num = int(self.circumvent_tokens)
        # Keep configured prefix tokens outside spatial processing and restore them in
        # sequence order.
        x, token = (
            x[:, prefix_tokens_num:, :], 
            x[:, :prefix_tokens_num, :]
        ) if self.circumvent_tokens else (x, None)

        x_shape = tf.shape(x)
        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        input_grid_size = tf.cast(
            tf.sqrt(tf.cast(x_shape[1], dtype=stable_dtype)),
            dtype=tf.int32
        )
    
        x = tf.reshape(x, (
            x_shape[0], 
            input_grid_size, 
            input_grid_size, 
            self.dim
        ))
        x = self.scaling_layer(
            x, 
            training=training
        )

        x_shape = tf.shape(x)
        output_grid_size = x_shape[1]

        x = tf.reshape(x, (
            x_shape[0], 
            output_grid_size * output_grid_size, 
            x.shape[-1]
        ))
        x = self._pos_merger(
            x, 
            output_grid_size=output_grid_size,
            training=training
        )
        # Keep configured prefix tokens outside spatial processing and restore them in
        # sequence order.
        # Match prefix-token width to spatial output only when projection is required.
        x = tf.concat([
            self.token_projector(
                token, 
                training=training
            ) if self.token_projector is not None else token, 
            x
        ], axis=1) if self.circumvent_tokens else x
        # Apply the final feature projection only when an MLP is configured.
        x = self.mlp(
            x, 
            training=training
        ) if self.mlp is not None else x

        return x


def run_self_tests() -> dict[str, str]:
    """Test every finite and boolean :class:`Upsample` control path.

    Args:
        None.

    Returns:
        dict[str, str]: A one-entry mapping after all scaling modes, interpolation choices,
        normalization, class-token, position, projection, gradient,
        validation, malformed-shape, dtype, and config checks pass.
    """

    tf.random.set_seed(717)
    condition = tf.ones((2, 3), dtype=tf.float32)
    modes = ("cnn_transpose", "interpolate", "cnn_interpolate")
    for mode in modes:
        for interpolation in ("nearest", "bilinear"):
            for use_layer_norm in (False, True):
                for circumvent_tokens in (False, True):
                    # Include a leading special token only in the prefix-preservation test
                    # case.
                    token_count = 5 if circumvent_tokens else 4
                    layer = Upsample(
                        dim=2, 
                        grid_size=2, 
                        scaling_method=mode, 
                        scaling_interpolation_method=interpolation, 
                        cnn_dim_ratio=2, 
                        cnn_kernel_size=3, 
                        cnn_activation_func="relu", 
                        circumvent_tokens=circumvent_tokens,
                        use_layer_norm=use_layer_norm, 
                        pos_embed_type=None, 
                    )
                    output = layer(
                        (tf.ones((2, token_count, 2)), condition), training=True,
                    )
                    # Pure interpolation preserves channels; convolution modes apply the
                    # configured multiplier.
                    expected_channels = 2 if mode == "interpolate" else 4
                    # Include a leading special token only in the prefix-preservation test
                    # case.
                    expected_tokens = 17 if circumvent_tokens else 16
                    assert output.shape == (2, expected_tokens, expected_channels)
                    assert (layer.layer_norm is not None) is use_layer_norm
                    assert layer.output_grid_size == 4

    positioned = Upsample(
        dim=2, 
        grid_size=2, 
        scaling_method="cnn_interpolate", 
        scaling_interpolation_method="nearest", 
        cnn_dim_ratio=2, 
        use_layer_norm=True, 
        ln_no_adaptation=True, 
        circumvent_tokens=True, 
        pos_embed_type="1d_sincos", 
        pos_merger_type="concat", 
        mlp_ratio=2, 
        mlp_output_dim=3, 
    )
    positioned_output = positioned((tf.ones((1, 5, 2)), None), training=False)
    assert positioned_output.shape == (1, 17, 3)

    pure_interpolation = Upsample(
        dim=1, 
        grid_size=2, 
        scaling_method="interpolate", 
        scaling_interpolation_method="nearest", 
        use_layer_norm=False, 
        pos_embed_type=None, 
    )
    source = tf.reshape(tf.constant([1.0, 2.0, 3.0, 4.0]), (1, 4, 1))
    nearest = pure_interpolation((source, None))
    assert nearest.shape == (1, 16, 1)
    assert nearest[0, 0, 0] == 1.0 and nearest[0, -1, 0] == 4.0

    trainable = Upsample(
        dim=2, 
        grid_size=2, 
        scaling_method="cnn_transpose", 
        use_layer_norm=False, 
        pos_embed_type=None, 
    )
    with tf.GradientTape() as tape:
        gradient_output = trainable(
            (tf.ones((1, 4, 2)), None), 
            training=True
        )
        loss = tf.reduce_sum(gradient_output)
    gradients = tape.gradient(loss, trainable.trainable_variables)
    assert gradients and all(gradient is not None for gradient in gradients)

    for invalid_grid in (-1, 0):
        try:
            Upsample(dim=2, grid_size=invalid_grid, pos_embed_type=None)
        except AssertionError:
            pass
        # This invalid case should already have raised: Upsampling requires a positive grid
        # size.
        else:
            raise AssertionError("Upsampling requires a positive grid size.")
    for invalid_mode in ("pixel_shuffle", "", None):
        try:
            Upsample(
                dim=2, grid_size=2, 
                scaling_method=invalid_mode, 
                pos_embed_type=None
            )
        except ValueError:
            pass
        # This invalid case should already have raised: Unknown upsampling modes must fail.
        else:
            raise AssertionError("Unknown upsampling modes must fail.")

    try:
        pure_interpolation((tf.ones((1, 3, 1)), None))
    except (tf.errors.InvalidArgumentError, ValueError):
        pass
    # This invalid case should already have raised: Non-square token grids must fail.
    else:
        raise AssertionError("Non-square token grids must fail.")

    dtype_layer = Upsample(
        dim=2, 
        grid_size=2, 
        scaling_method="interpolate", 
        use_layer_norm=False, 
        pos_embed_type=None, 
        dtype="float64"
    )
    dtype_output = dtype_layer((tf.ones((1, 4, 2), tf.float64), None))
    assert dtype_output.shape == (1, 16, 2)
    assert dtype_layer.compute_dtype == "float64"
    assert dtype_output.dtype == tf.float64

    config = pure_interpolation.get_config()
    restored = Upsample.from_config(config)
    assert restored.scaling_method == "interpolate"
    assert restored.scaling_interpolation_method == "nearest"

    return {"Upsample": "passed"}


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
