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
        self.output_grid_size = self.grid_size // self.strides
        self.add_residual = self.strides == 1 # and \
                            # self.use_layer_norm and \
                            # not self.ln_no_adaptation

        self.layer_norm = self._create_layer_norm(
            gate_dim=self.output_dim if self.add_residual else 0, 
            return_gate=True
        )

        self.depthwise = layers.DepthwiseConv2D(
            kernel_size=self.kernel_size, 
            strides=self.strides, 
            padding="same", 
            depth_multiplier=self.depth_multiplier, 
            depthwise_initializer="zeros" if self.zero_init else "glorot_uniform", 
            name="depthwise"
        )
        self.pointwise = layers.Conv2D(
            filters=self.output_dim, 
            kernel_size=1, 
            padding="same", 
            name="pointwise"
        ) if self.use_pointwise else None
        self.residual_projector = layers.Dense(
            self.output_dim, 
            name="residual_projector"
        ) if self.dim != self.output_dim and self.add_residual else None
        self.pos_embed = self._create_embeddings(
            embed_dim=self.output_dim, 
            output_grid_size=self.output_grid_size
        )

        self.output_dim = self.output_dim * 2 if self.pos_embed_type is not None and \
                        self.pos_merger_type == "concat" and self.mlp is None else self.output_dim

        self.token_projector = layers.Dense(
            self.output_dim, 
            name="token_projector"
        ) if self.dim != self.output_dim else None
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
        h = tf.reshape(h, (
            -1, 
            self.grid_size, 
            self.grid_size, 
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
        h = tf.reshape(h, (
            -1, 
            self.output_grid_size * self.output_grid_size, 
            self.prev_output_dim
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
            training=training
        )
        x = tf.concat([
            x_token + self.token_projector(
                h_token, 
                training=training
            ) if self.token_projector is not None else x_token + h_token, 
            x
        ], axis=1) if self.circumvent_cls_token else x
        x = self.mlp(
            x, 
            training=training
        ) if self.mlp is not None else x

        return x
