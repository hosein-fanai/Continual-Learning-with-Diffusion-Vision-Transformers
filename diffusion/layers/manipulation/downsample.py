"""Spatial downsampling for flattened square token grids."""

import tensorflow as tf
from tensorflow.keras import layers

from typing import Literal, TypeAlias

from diffusion.layers.embedding.base_embedding import BaseEmbedding


ScalingMethod: TypeAlias = Literal[
    "avg_pooling", 
    "max_pooling", 
    "cnn_stride"
]
"""Available pooling and strided-convolution downsampling strategies."""


class Downsample(BaseEmbedding):
    """Reduce a square patch-token grid and optionally preserve a class token.

    The layer optionally normalizes tokens, removes a leading class token,
    reshapes spatial tokens to ``[batch, grid, grid, channels]``, applies the
    selected spatial reduction, restores a sequence, merges a positional table,
    restores/project the class token, and optionally applies an MLP.

    Args:
        use_layer_norm: Whether to create condition-adaptive normalization
            before spatial scaling. If enabled with normal adaptation,
            :meth:`call` requires ``cond``.
        scaling_method: ``"avg_pooling"`` or ``"max_pooling"`` preserves
            ``dim`` channels; ``"cnn_stride"`` learns a convolution and emits
            ``dim * cnn_dim_ratio`` channels.
        strides: Positive integer spatial stride. Pooling uses Keras's default
            2x2 window. Convolution uses ``cnn_kernel_size``.
        padding: Keras 2-D padding mode, normally ``"same"`` or ``"valid"``.
        cnn_dim_ratio: Positive integer channel multiplier used only by
            ``"cnn_stride"``.
        cnn_kernel_size: Positive convolution kernel side used only by
            ``"cnn_stride"``.
        cnn_activation_func: Keras activation for the strided convolution;
            ``"linear"`` leaves it unbounded.
        circumvent_cls_token: Treat token 0 as a non-spatial class token,
            exclude it from downsampling, project it when widths change, and
            prepend it again.
        **kwargs: :class:`BaseEmbedding` options. Required keys are ``dim`` and
            ``grid_size`` (at least 2). Positional keys include
            ``pos_embed_type``, ``pos_merger_type``, and
            ``pos_interpolation_method``; MLP keys include ``mlp_ratio`` and
            ``mlp_output_dim``. ``use_layer_norm`` and ``ln_dim`` are supplied
            here and must not be repeated.

    Inputs:
        Pair ``(x, cond)``. ``x`` is floating ``[batch, tokens, dim]``. The
        spatial token count must be a perfect square; with
        ``circumvent_cls_token=True``, ``tokens - 1`` must be square. ``cond``
        is ``[batch, condition_dim]`` when adaptive normalization is active.

    Outputs:
        Floating tensor ``[batch, reduced_tokens, output_dim]``. A leading
        class token adds one to ``reduced_tokens``. Positional concatenation
        doubles the pre-MLP width; ``mlp_output_dim`` can replace it.

    Serialization:
        ``get_config()`` includes inherited ``ln_dim`` even though
        :class:`BaseEmbedding` supplies it from ``dim``. Remove ``ln_dim`` from
        a copied config before calling ``Downsample.from_config``.
    """

    def __init__(
        self, 
        use_layer_norm: bool = True, 
        scaling_method: ScalingMethod = "avg_pooling", 
        strides: int = 2, 
        padding: str = "same", 
        cnn_dim_ratio: int = 1, 
        cnn_kernel_size: int = 3, 
        cnn_activation_func: str = "linear", 
        circumvent_cls_token: bool = False, 
        **kwargs
    ):
        """Create the selected scaler, positional table, and projections.

        Arguments and accepted types are documented on the class.

        Returns:
            ``None``.
        """

        super().__init__(
            use_layer_norm=use_layer_norm, 
            **kwargs
        )
        self._save_init_args(locals())

        assert self.grid_size >= 2, \
            "grid_size must be at least 2 for downsampling."


        self.output_grid_size = (
            self.grid_size + self.strides - 1
        ) // self.strides if self.padding == "same" \
                        else self.grid_size // self.strides

        self.layer_norm = self._create_layer_norm(
            return_gate=False
        )

        name = f"{self.name}/scaling_layer"
        if self.scaling_method == "avg_pooling":
            self.output_dim = self.dim
            self.scaling_layer = layers.AveragePooling2D(
                strides=self.strides, 
                padding=self.padding, 
                name=name
            )
        elif self.scaling_method == "max_pooling":
            self.output_dim = self.dim
            self.scaling_layer = layers.MaxPooling2D(
                strides=self.strides, 
                padding=self.padding, 
                name=name
            )
        elif self.scaling_method == "cnn_stride":
            self.output_dim = self.dim * self.cnn_dim_ratio
            self.scaling_layer = layers.Conv2D(
                filters=self.output_dim, 
                kernel_size=self.cnn_kernel_size, 
                strides=self.strides, 
                padding=self.padding, 
                activation=self.cnn_activation_func, 
                name=name
            )
        else:
            raise ValueError(f"pooling method can only be one of {ScalingMethod}.")

        self.pos_embed = self._create_embeddings(
            embed_dim=self.output_dim, 
            output_grid_size=self.output_grid_size
        )

        self.output_dim = self.output_dim * 2 if self.pos_embed_type is not None and \
                        self.pos_merger_type == "concat" else self.output_dim

        self.token_projector = layers.Dense(
            self.output_dim, 
            name=f"{self.name}/token_projector"
        ) if self.dim != self.output_dim else None
        self.mlp = self._create_mlp(
            self.output_dim
        )

    def call(self, inputs, training=None):
        """Downsample a token grid.

        Args:
            inputs: Pair ``(x, cond)`` following the class input contract.
            training: Optional Keras training flag forwarded to normalization,
                convolution, positional projection, and MLP layers.

        Returns:
            ``tf.Tensor`` with floating compute dtype. For an actual input grid
            ``g``, ``same`` padding yields side ``ceil(g / strides)``. With
            ``valid`` padding, pooling yields
            ``floor((g - 2) / strides) + 1`` and convolution yields
            ``floor((g - cnn_kernel_size) / strides) + 1``. The configured
            positional size assumes the common matching stride/window setup.
        """

        x, cond = inputs

        x = self.layer_norm(
            (x, cond), 
            training=training
        ) if self.layer_norm is not None else x
        x, token = (
            x[:, 1:, :], x[:, 0: 1, :]
        ) if self.circumvent_cls_token else (x, None)

        x_shape = tf.shape(x)
        input_grid_size = tf.cast(
            tf.sqrt(tf.cast(x_shape[1], dtype=tf.float32)), 
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
        x = tf.concat([
            self.token_projector(
                token, 
                training=training
            ) if self.token_projector is not None else token, 
            x
        ], axis=1) if self.circumvent_cls_token else x
        x = self.mlp(
            x, 
            training=training
        ) if self.mlp is not None else x

        return x


def run_self_tests() -> dict[str, str]:
    """Test every :class:`Downsample` scaling and boolean branch.

    Args:
        None.

    Returns:
        A one-entry mapping after pooling/convolution, padding, stride,
        normalization, class-token, position, projection, dtype, gradient,
        validation, shape-error, and serialization checks pass.
    """

    import numpy as np


    tf.random.set_seed(616)
    condition = tf.ones((2, 3), dtype=tf.float32)
    modes = ("avg_pooling", "max_pooling", "cnn_stride")
    for mode in modes:
        for use_layer_norm in (False, True):
            for circumvent_cls_token in (False, True):
                token_count = 17 if circumvent_cls_token else 16
                layer = Downsample(
                    dim=2, 
                    grid_size=4, 
                    scaling_method=mode, 
                    strides=2, 
                    padding="same", 
                    cnn_dim_ratio=2, 
                    cnn_kernel_size=3, 
                    cnn_activation_func="relu", 
                    circumvent_cls_token=circumvent_cls_token, 
                    use_layer_norm=use_layer_norm, 
                    pos_embed_type=None
                )
                output = layer(
                    (tf.ones((2, token_count, 2)), condition), 
                    training=True
                )
                expected_channels = 4 if mode == "cnn_stride" else 2
                expected_tokens = 5 if circumvent_cls_token else 4
                assert output.shape == (2, expected_tokens, expected_channels)
                assert (layer.layer_norm is not None) is use_layer_norm
                assert layer.output_grid_size == 2

    values = tf.reshape(tf.range(16, dtype=tf.float32), (1, 16, 1))
    average = Downsample(
        dim=1, grid_size=4, 
        scaling_method="avg_pooling", 
        strides=2, padding="valid", 
        use_layer_norm=False, 
        pos_embed_type=None
    )((values, None))
    maximum = Downsample(
        dim=1, grid_size=4, 
        scaling_method="max_pooling", 
        strides=2, padding="valid", 
        use_layer_norm=False, 
        pos_embed_type=None
    )((values, None))
    np.testing.assert_allclose(average.numpy().reshape(-1), [2.5, 4.5, 10.5, 12.5])
    np.testing.assert_array_equal(maximum.numpy().reshape(-1), [5.0, 7.0, 13.0, 15.0])

    valid_convolution = Downsample(
        dim=2, 
        grid_size=4, 
        scaling_method="cnn_stride", 
        strides=2, 
        padding="valid", 
        cnn_kernel_size=3, 
        use_layer_norm=False, 
        pos_embed_type=None
    )
    assert valid_convolution((tf.ones((1, 16, 2)), None)).shape == (1, 1, 2)

    positioned = Downsample(
        dim=2, 
        grid_size=4, 
        scaling_method="cnn_stride", 
        cnn_dim_ratio=2, 
        use_layer_norm=True, 
        ln_no_adaptation=True, 
        circumvent_cls_token=True, 
        pos_embed_type="2d_sincos", 
        pos_merger_type="concat", 
        mlp_ratio=2, 
        mlp_output_dim=3
    )
    positioned_output = positioned((tf.ones((1, 17, 2)), None), training=False)
    assert positioned_output.shape == (1, 5, 3)

    with tf.GradientTape() as tape:
        convolution_output = valid_convolution(
            (tf.ones((1, 16, 2)), None), training=True
        )
        loss = tf.reduce_sum(convolution_output)
    gradients = tape.gradient(loss, valid_convolution.trainable_variables)
    assert gradients and all(gradient is not None for gradient in gradients)

    for invalid_grid in (0, 1):
        try:
            Downsample(dim=2, grid_size=invalid_grid, pos_embed_type=None)
        except AssertionError:
            pass
        else:
            raise AssertionError("Downsampling requires grid_size >= 2.")
    for invalid_mode in ("median_pooling", "", None):
        try:
            Downsample(
                dim=2, grid_size=2, 
                scaling_method=invalid_mode,
                pos_embed_type=None
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Unknown downsampling modes must fail.")

    malformed = Downsample(
        dim=2, grid_size=4, 
        use_layer_norm=False, 
        pos_embed_type=None
    )
    try:
        malformed((tf.ones((1, 15, 2)), None))
    except (tf.errors.InvalidArgumentError, ValueError):
        pass
    else:
        raise AssertionError("Non-square token grids must fail.")

    dtype_layer = Downsample(
        dim=2, grid_size=2, 
        use_layer_norm=False, 
        pos_embed_type=None, 
        dtype="float64",
    )
    dtype_output = dtype_layer((tf.ones((1, 4, 2), tf.float64), None))
    assert dtype_layer.compute_dtype == "float64"
    # TensorFlow 2.10's nested pooling layer retains its default float32 policy.
    assert dtype_output.dtype == tf.float32

    config = malformed.get_config()
    try:
        Downsample.from_config(config)
    except TypeError:
        pass
    else:
        raise AssertionError("The documented duplicate-ln_dim limit changed.")
    filtered_config = dict(config)
    filtered_config.pop("ln_dim")
    restored = Downsample.from_config(filtered_config)
    assert restored.scaling_method == "avg_pooling" and restored.strides == 2

    return {"Downsample": "passed"}


if __name__ == "__main__":
    print(run_self_tests())
