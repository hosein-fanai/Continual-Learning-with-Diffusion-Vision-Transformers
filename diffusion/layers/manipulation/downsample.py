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
