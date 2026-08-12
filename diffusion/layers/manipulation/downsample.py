from typing import Literal, TypeAlias

import tensorflow as tf
from tensorflow.keras import layers

from diffusion.layers.embedding.base_embedding import BaseEmbedding


ScalingMethod: TypeAlias = Literal[
    "avg_pooling", 
    "max_pooling", 
    "cnn_stride"
]


class Downsample(BaseEmbedding):
    """
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
        super().__init__(
            use_layer_norm=use_layer_norm, 
            **kwargs
        )
        self._save_init_args(locals())

        assert self.grid_size >= 2, \
            "grid_size must be at least 2 for downsampling."


        self.output_grid_size = (
            self.grid_size + self.self.strides - 1
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
