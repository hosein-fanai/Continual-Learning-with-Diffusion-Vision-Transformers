"""Depthwise-convolutional local mixing for transformer token sequences."""

import tensorflow as tf
from tensorflow.keras import layers

from diffusion.layers.embedding.base_embedding import BaseEmbedding


class LocalMixer(BaseEmbedding):
    """Inject convolutional locality into a flattened patch-token sequence.

    The layer reshapes square spatial tokens to an image, applies a depthwise
    convolution and optional 1x1 pointwise projection, then flattens the result.
    With ``strides == 1`` it adds the local correction to a residual path; with
    larger strides it returns only the reduced local features. Position and MLP
    processing occur after this mixing.

    Args:
        use_layer_norm: Whether to apply condition-adaptive normalization before
            convolution. Disabled normalization uses ``x`` and a scalar-one
            gate directly.
        kernel_size: Positive depthwise kernel side length.
        strides: Positive depthwise stride. ``1`` enables a residual; larger
            values spatially reduce the sequence.
        padding: Keras padding mode, normally ``"same"`` or ``"valid"``. A
            stride-one residual requires output and input token counts to match,
            so use ``"same"`` (or an effectively size-preserving kernel).
        depth_multiplier: Positive number of depthwise filters per input channel.
        pointwise_dim_ratio: Positive output multiplier for the optional 1x1
            convolution.
        use_pointwise: If true, project depthwise channels to
            ``dim * pointwise_dim_ratio``; otherwise retain
            ``dim * depth_multiplier`` channels.
        zero_init: Zero-initialize the depthwise kernel so the initial local
            correction is zero. This covers the convolutional path, while an
            adaptive normalization gate also starts at zero by default.
        circumvent_cls_token: Exclude token 0 from spatial convolution and
            combine it with its normalized/projected counterpart afterward.
        **kwargs: :class:`BaseEmbedding` options. Required keys are ``dim`` and
            ``grid_size``. Positional/MLP options can change the final channel
            width. ``use_layer_norm`` and ``ln_dim`` are supplied internally
            and must not be repeated.

    Inputs:
        Pair ``(x, cond)`` with floating tokens ``[batch, tokens, dim]``. The
        spatial token count must be a perfect square after removing an optional
        leading class token. ``cond`` has shape ``[batch, condition_dim]`` when
        adaptive normalization is enabled.

    Outputs:
        Floating token tensor at the inferred convolutional grid size. A
        stride-one, same-padded, additive-position configuration preserves the
        input shape; strides, positional concatenation, pointwise ratios, and
        MLP projection can change it.

    Serialization:
        ``get_config()`` includes inherited ``ln_dim`` even though
        :class:`BaseEmbedding` supplies it from ``dim``. Remove ``ln_dim`` from
        a copied config before calling ``LocalMixer.from_config``.
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
        """Create local convolutions, residual projections, and position data.

        Arguments and accepted types are documented on the class.

        Returns:
            ``None``.
        """

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
        """Mix neighboring spatial tokens and apply the configured residual.

        Args:
            inputs: Pair ``(x, cond)`` following the class input contract.
            training: Optional Keras training flag forwarded to normalization,
                convolutions, positional projection, and MLP layers.

        Returns:
            ``tf.Tensor`` with floating compute dtype. For input grid side
            ``g``, ``same`` padding produces ``ceil(g / strides)``; ``valid``
            produces ``floor((g - kernel_size) / strides) + 1``. A leading
            class token is added back when configured.
        """

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
