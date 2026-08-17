"""Convert image feature maps into transformer patch-token sequences."""

import tensorflow as tf
from tensorflow.keras import layers, models

from diffusion.layers.embedding.base_embedding import BaseEmbedding
from diffusion.layers.single_token_layer import SingleTokenLayer


class PatchEmbedding(BaseEmbedding):
    """Patchify a channels-last image and merge spatial position features.

    Standard patchification uses one ``patch_size`` convolution with matching
    stride and ``valid`` padding. CNN patchification first applies a 3x3 Swish
    convolution and then a 3x3 stride-``patch_size`` convolution with ``same``
    padding. Both paths flatten the resulting square feature map in row-major
    order.

    Args:
        patch_size: Positive integer convolution stride and, for standard
            patchification, kernel side length.
        patchify_with_cnn: Select the two-convolution ``same``-padding stem
            instead of the single ``valid``-padding projection.
        shift_right_token: Prepend a learned BOS token and discard the last
            patch token, preserving sequence length. The internal BOS layer has
            positional embeddings disabled and currently retains batch size
            one, so this option is directly shape-compatible only with
            single-example calls unless the token implementation is batched by
            the caller.
        **kwargs: :class:`BaseEmbedding` options. Required keys are ``dim`` and
            ``grid_size``. Common keys include ``pos_embed_type``,
            ``pos_merger_type``, ``pos_interpolation_method``,
            ``embed_freq_dim``, ``mlp_ratio``, and ``mlp_output_dim``. An
            additive table must match the patch projection width; concatenation
            allocates ``dim // 2`` channels to content before merging.

    Inputs:
        Floating image tensor shaped ``[batch, height, width, channels]``. The
        projected height and width must be equal because tokens are treated as
        a square grid. At the native size they should equal ``grid_size`` so
        positional token counts match.

    Outputs:
        Floating tensor shaped ``[batch, output_grid_size ** 2, output_dim]``.
        ``output_dim`` is normally ``dim`` and may be changed by concatenation
        or an explicit MLP output width.

    Serialization:
        The saved config contains an inherited ``ln_dim`` key that duplicates
        the value supplied internally by :class:`BaseEmbedding`. Remove that
        key from a copied config before calling ``PatchEmbedding.from_config``.

    """

    def __init__(
        self, 
        patch_size: int = 2, 
        patchify_with_cnn: bool = False, 
        shift_right_token: bool = False, 
        **kwargs
    ):
        """Create the convolutional patch projector and positional table.

        Arguments and accepted types are documented on the class.

        Returns:
            ``None``.
        """

        super().__init__(**kwargs)
        self._save_init_args(locals())

        self.mlp_ratio = 1 if self.mlp_ratio is None and self.embed_freq_dim is not None \
                        else self.mlp_ratio
        self.hidden_dim = self.dim // 2 if self.pos_merger_type == "concat" else self.dim
        self.mlp_output_dim = self.hidden_dim if self.mlp_output_dim is None \
                            and self.embed_freq_dim is not None else self.mlp_output_dim
        self.embed_dim = self.hidden_dim if self.embed_freq_dim is None else self.embed_freq_dim
        self.output_grid_size = self.grid_size

        if self.patchify_with_cnn:
            self.patch_projector = models.Sequential([
                layers.Conv2D(
                    self.hidden_dim // 2, 
                    kernel_size=3, 
                    strides=1, 
                    padding="same", 
                    activation="swish", 
                    name=f"{self.name}/patch_projector/conv_1"
                ), 
                layers.Conv2D(
                    self.hidden_dim, 
                    kernel_size=3, 
                    strides=self.patch_size, 
                    padding="same", 
                    name=f"{self.name}/patch_projector/conv_2"
                )
            ], name="patch_projector")
        else:
            self.patch_projector = layers.Conv2D(
                self.hidden_dim, 
                self.patch_size, 
                strides=self.patch_size, 
                name="patch_projector"
            )

        self.shift_right_token = SingleTokenLayer(
            dim=self.hidden_dim, 
            with_pos_embed=False, 
            name=f"{self.name}/bos_token"
        ) if self.shift_right_token else None
        self.pos_embed = self._create_embeddings(
            output_grid_size=self.grid_size, 
        ) if self.pos_merger_type is not None else None
        self.pos_embed_mlp = self._create_mlp(
            self.embed_dim
        )

    def call(self, x: tf.Tensor, 
            output_grid_size: int | None = None, 
            training: bool | None = None):
        """Project an image to patch tokens.

        Args:
            x: Floating TensorFlow tensor shaped
                ``[batch, height, width, channels]``.
            output_grid_size: Optional positive target side for positional-table
                resizing. It does not resize image patches and must equal the
                projected spatial side for elementwise addition/concatenation.
                ``None`` uses the configured/native positional behavior.
            training: Optional Keras training flag forwarded to convolutions,
                token handling, and positional projection layers.

        Returns:
            ``tf.Tensor`` of patch tokens with floating compute dtype and shape
            ``[batch, projected_height * projected_width, output_dim]``.
            Standard projection uses ``floor(height / patch_size)`` for
            divisible inputs; the CNN stem uses ``ceil(height / patch_size)``.
        """

        x = self.patch_projector(
            x, 
            training=training
        )

        x_shape = tf.shape(x)

        x = tf.reshape(x, (
            x_shape[0], 
            x_shape[1] * x_shape[2], 
            x_shape[3]
        ))
        x = tf.concat([
            self.shift_right_token(
                (x, None), 
                training=training
            ), 
            x[:, :-1, :]
        ], axis=1) if self.shift_right_token is not None else x
        x = self._pos_merger(
            x, 
            output_grid_size=output_grid_size, 
            training=training
        )

        return x
