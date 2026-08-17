"""Spatial upsampling for flattened square token grids."""

import tensorflow as tf
from tensorflow.keras import layers, models

from typing import Literal, TypeAlias

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
        use_layer_norm: Whether to normalize/adapt tokens before resizing.
        scaling_method: One of ``"cnn_transpose"``, ``"interpolate"``, or
            ``"cnn_interpolate"``.
        scaling_interpolation_method: Keras ``UpSampling2D`` method used by the
            interpolation modes, commonly ``"nearest"`` or ``"bilinear"``.
        cnn_dim_ratio: Positive integer channel multiplier for modes containing
            a convolution. Pure interpolation preserves ``dim`` channels.
        cnn_kernel_size: Positive kernel side for the transposed or post-resize
            convolution.
        cnn_activation_func: Keras convolution activation; ``"linear"`` leaves
            results unbounded.
        circumvent_cls_token: Exclude token 0 from spatial resizing, project it
            if necessary, and prepend it to the resized sequence.
        **kwargs: :class:`BaseEmbedding` options. ``dim`` and positive
            ``grid_size`` are required. Positional, MLP, normalization, and
            standard Keras options are accepted; ``use_layer_norm`` and
            ``ln_dim`` are supplied internally and must not be repeated.

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
        ``get_config()`` includes inherited ``ln_dim`` even though
        :class:`BaseEmbedding` supplies it from ``dim``. Remove ``ln_dim`` from
        a copied config before calling ``Upsample.from_config``.
    """

    def __init__(
        self, 
        use_layer_norm: bool = True, 
        scaling_method: ScalingMethod = "cnn_transpose", 
        scaling_interpolation_method: str = "nearest", 
        cnn_dim_ratio: int = 1, 
        cnn_kernel_size: int = 2, 
        cnn_activation_func: str = "linear", 
        circumvent_cls_token: bool = False, 
        **kwargs
    ):
        """Create the factor-two scaler, position table, and projections.

        Arguments and accepted types are documented on the class.

        Returns:
            ``None``.
        """

        super().__init__(
            use_layer_norm=use_layer_norm, 
            **kwargs
        )
        self._save_init_args(locals())

        assert self.grid_size >= 1, \
            "grid_size must be positive."


        self.output_grid_size = self.grid_size * 2

        self.layer_norm = self._create_layer_norm(
            return_gate=False
        )

        name = f"{self.name}/scaling_layer"
        if self.scaling_method == "cnn_transpose":
            self.output_dim = self.dim * self.cnn_dim_ratio
            self.scaling_layer = layers.Conv2DTranspose(
                filters=self.output_dim, 
                kernel_size=self.cnn_kernel_size, 
                strides=2, 
                padding="same", 
                activation=self.cnn_activation_func, 
                name=name
            )
        elif self.scaling_method == "interpolate":
            self.output_dim = self.dim
            self.scaling_layer = layers.UpSampling2D(
                size=(2, 2), 
                interpolation=self.scaling_interpolation_method, 
                name=name
            )
        elif self.scaling_method == "cnn_interpolate":
            self.output_dim = self.dim * self.cnn_dim_ratio
            self.scaling_layer = models.Sequential([
                layers.UpSampling2D(
                    size=(2, 2), 
                    interpolation=self.scaling_interpolation_method, 
                    name=f"{name}_interpolation"
                ), 
                layers.Conv2D(
                    filters=self.output_dim, 
                    kernel_size=self.cnn_kernel_size, 
                    padding="same", 
                    activation=self.cnn_activation_func, 
                    name=f"{name}_convolution"
                )
            ], name=name)
        else:
            raise ValueError(f"scaling_method method can only be one of {ScalingMethod}.")

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
        """Upsample a token grid by a factor of two per spatial axis.

        Args:
            inputs: Pair ``(x, cond)`` following the class input contract.
            training: Optional Keras training flag forwarded to every nested
                normalization, convolution, and dense layer.

        Returns:
            Floating ``tf.Tensor`` shaped
            ``[batch, (2 * grid) ** 2, output_dim]``, plus one token when
            ``circumvent_cls_token=True``. ``grid`` is inferred dynamically
            from the input token count.
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
