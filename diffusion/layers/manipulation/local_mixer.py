import tensorflow as tf
from tensorflow.keras import layers

from diffusion.layers.embedding.base_embedding import BaseEmbedding


class LocalMixer(BaseEmbedding):
    """
    Lightweight residual local token mixer.

    Input:
        x: patch tokens with shape (B, H*W, dim)

    Output:
        x + local spatial correction, same shape as input.

    It reshapes tokens to (B, H, W, dim), applies depthwise spatial conv,
    optional pointwise conv, then reshapes back to tokens.
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
        circumvent_cls_token: bool = False, 
        **kwargs
    ):
        super().__init__(
            use_layer_norm=use_layer_norm, 
            **kwargs
        )
        self._save_init_args(locals())

        self.output_dim = self.dim * self.pointwise_dim_ratio if self.use_pointwise \
                        else self.dim * self.depth_multiplier
        self.output_grid_size = (
            self.grid_size + self.strides - 1
        ) // self.strides if self.padding == "same" \
                        else self.grid_size // self.strides
        self.add_residual = self.strides == 1

        self.layer_norm = self._create_layer_norm(
            gate_dim=self.output_dim if self.add_residual else 0, 
            return_gate=True
        )
        self.depthwise = layers.DepthwiseConv2D(
            kernel_size=self.kernel_size, 
            strides=self.strides, 
            padding=self.padding, 
            depth_multiplier=self.depth_multiplier, 
            depthwise_initializer="zeros" if self.zero_init else "glorot_uniform", 
            name=f"{self.name}/depthwise"
        )
        self.pointwise = layers.Conv2D(
            filters=self.output_dim, 
            kernel_size=1, 
            padding="same", 
            name=f"{self.name}/pointwise"
        ) if self.use_pointwise else None
        self.residual_projector = layers.Dense(
            self.output_dim, 
            name=f"{self.name}/residual_projector"
        ) if self.dim != self.output_dim and self.add_residual else None
        self.pos_embed = self._create_embeddings(
            embed_dim=self.output_dim, 
            output_grid_size=self.output_grid_size
        )

        residual_token_dim = self.output_dim \
                            if self.residual_projector is not None else self.dim
        self.output_dim = self.output_dim * 2 if self.pos_embed_type is not None and \
                        self.pos_merger_type == "concat" else self.output_dim

        self.token_projector = layers.Dense(
            self.output_dim, 
            name=f"{self.name}/token_projector"
        ) if self.dim != self.output_dim else None
        self.residual_token_projector = layers.Dense(
            self.output_dim, 
            name=f"{self.name}/residual_token_projector"
        ) if residual_token_dim != self.output_dim and \
            self.circumvent_cls_token else None
        self.mlp = self._create_mlp(
            self.output_dim
        )

    def call(self, inputs, training=None):
        x, cond = inputs

        h, gate = self.layer_norm(
            (x, cond), 
            training=training
        ) if self.layer_norm is not None else (x, 1.)
        h, h_token = (
            h[:, 1:, :], h[:, 0: 1, :]
        ) if self.circumvent_cls_token else (h, None)

        h_shape = tf.shape(h)
        input_grid_size = tf.cast(
            tf.sqrt(tf.cast(h_shape[1], dtype=tf.float32)), 
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
        x = self.residual_projector(
            x, 
            training=training
        ) if self.residual_projector is not None else x
        x, x_token = (
            x[:, 1:, :], x[:, 0: 1, :]
        ) if self.circumvent_cls_token else (x, None)
        x = x + gate * h if self.add_residual else h

        x = self._pos_merger(
            x, 
            output_grid_size=output_grid_size, 
            training=training
        )
        x_token = self.residual_token_projector(
            x_token, 
            training=training
        ) if self.residual_token_projector is not None else x_token
        x = tf.concat([
            (x_token + self.token_projector(
                h_token, 
                training=training
            )) if self.token_projector is not None else (x_token + h_token), 
            x
        ], axis=1) if self.circumvent_cls_token else x
        x = self.mlp(
            x, 
            training=training
        ) if self.mlp is not None else x

        return x
