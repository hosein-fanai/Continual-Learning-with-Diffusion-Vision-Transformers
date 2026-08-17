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
            patch token, preserving sequence length. The learned token is
            repeated across the runtime batch, so every example receives the
            same trainable BOS value.
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
            tf.repeat(
                self.shift_right_token(
                    (x, None),
                    training=training
                ),
                tf.shape(x)[0],
                axis=0,
            ), 
            x[:, :-1, :]
        ], axis=1) if self.shift_right_token is not None else x
        x = self._pos_merger(
            x, 
            output_grid_size=output_grid_size, 
            training=training
        )

        return x


def run_self_tests() -> dict[str, str]:
    """Run small image-to-token tests for :class:`PatchEmbedding`.

    Args:
        None.

    Returns:
        A one-entry success mapping after both projector stems, positional
        merges, BOS shifting, projection, resize, dtype, gradient, invalid
        shape, and serialization checks pass.
    """

    import numpy as np


    tf.random.set_seed(404)
    images = tf.reshape(tf.range(2 * 8 * 8, dtype=tf.float32), (2, 8, 8, 1))

    standard = PatchEmbedding(
        dim=4, 
        grid_size=4, 
        patch_size=2, 
        patchify_with_cnn=False, 
        shift_right_token=False, 
        pos_embed_type="new_weight", 
        pos_merger_type="add"
    )
    tokens = standard(images, training=False)
    assert tokens.shape == (2, 16, 4) and tokens.dtype == tf.float32
    assert standard.patch_projector.kernel_size == (2, 2)

    cnn = PatchEmbedding(
        dim=4, 
        grid_size=4, 
        patch_size=2, 
        patchify_with_cnn=True, 
        pos_embed_type="2d_sincos"
    )
    cnn_tokens = cnn(images[:, :7, :7, :], training=True)
    assert cnn_tokens.shape == (2, 16, 4)
    assert len(cnn.patch_projector.layers) == 2

    no_position = PatchEmbedding(
        dim=3, 
        grid_size=4, 
        patch_size=2, 
        pos_embed_type=None
    )
    assert no_position(images).shape == (2, 16, 3)
    assert no_position.pos_embed is None

    concatenated = PatchEmbedding(
        dim=6, 
        grid_size=4, 
        patch_size=2, 
        pos_embed_type="1d_sincos", 
        pos_merger_type="concat", 
    )
    assert concatenated(images).shape == (2, 16, 6)

    projected = PatchEmbedding(
        dim=6, 
        grid_size=4, 
        patch_size=2, 
        embed_freq_dim=4, 
        pos_embed_type="1d_sincos"
    )
    assert projected(images).shape == (2, 16, 6)
    explicit_projection = PatchEmbedding(
        dim=6, 
        grid_size=4, 
        patch_size=2, 
        embed_freq_dim=4, 
        pos_embed_type="1d_sincos", 
        pos_merger_type="concat", 
        mlp_output_dim=3, 
        mlp_ratio=2, 
    )
    assert explicit_projection(images).shape == (2, 16, 6)
    incompatible_projection = PatchEmbedding(
        dim=6, 
        grid_size=4, 
        patch_size=2, 
        embed_freq_dim=4, 
        pos_embed_type="1d_sincos", 
        mlp_output_dim=3, 
    )
    try:
        incompatible_projection(images)
    except (tf.errors.InvalidArgumentError, ValueError):
        pass
    else:
        raise AssertionError("Additive content and position widths must match.")

    resized_positions = PatchEmbedding(
        dim=4, 
        grid_size=4, 
        patch_size=2, 
        pos_embed_type="2d_interpolate", 
        pos_interpolation_method="bilinear", 
    )
    smaller = resized_positions(
        tf.ones((1, 4, 4, 1)), 
        output_grid_size=2,
    )
    assert smaller.shape == (1, 4, 4)

    shifted = PatchEmbedding(
        dim=4, 
        grid_size=4, 
        patch_size=2, 
        shift_right_token=True, 
        pos_embed_type=None, 
    )
    unshifted_one = no_position(images[:1])
    shifted_one = shifted(images[:1])
    assert shifted_one.shape == (1, 16, 4)
    np.testing.assert_allclose(
        shifted_one[:, 1:, :].numpy(), 
        shifted.patch_projector(images[:1]).numpy().reshape(1, 16, 4)[:, :-1, :], 
        atol=1e-6, 
    )
    assert unshifted_one.shape == (1, 16, 3)

    shifted_batch = shifted(images)
    projected_batch = shifted.patch_projector(images).numpy().reshape(2, 16, 4)
    assert shifted_batch.shape == (2, 16, 4)
    np.testing.assert_allclose(
        shifted_batch[:, 1:, :].numpy(), 
        projected_batch[:, :-1, :], 
        atol=1e-6, 
    )
    np.testing.assert_allclose(
        shifted_batch[0, 0, :].numpy(), 
        shifted_batch[1, 0, :].numpy(), 
        atol=1e-6, 
    )

    cnn_shifted_positioned = PatchEmbedding(
        dim=4, 
        grid_size=4, 
        patch_size=2, 
        patchify_with_cnn=True, 
        shift_right_token=True, 
        pos_embed_type="new_weight", 
        pos_merger_type="add", 
    )
    cnn_shifted_tokens = cnn_shifted_positioned(images, training=True)
    assert cnn_shifted_tokens.shape == (2, 16, 4)
    assert cnn_shifted_positioned.shift_right_token is not None
    assert cnn_shifted_positioned.pos_embed is not None

    with tf.GradientTape() as tape:
        result = standard(images[:1], training=True)
        loss = tf.reduce_sum(result)
    gradients = tape.gradient(loss, standard.trainable_variables)
    assert gradients and all(gradient is not None for gradient in gradients)

    try:
        standard(tf.ones((1, 6, 6, 1)))
    except (tf.errors.InvalidArgumentError, ValueError):
        pass
    else:
        raise AssertionError("A native position table must reject wrong token counts.")
    try:
        PatchEmbedding(dim=4, grid_size=2, pos_embed_type="invalid")
    except AssertionError:
        pass
    else:
        raise AssertionError("Invalid positional modes must fail.")

    dtype_layer = PatchEmbedding(
        dim=4, 
        grid_size=4, 
        patch_size=2, 
        pos_embed_type=None, 
        dtype="float64", 
    )
    dtype_output = dtype_layer(tf.ones((1, 8, 8, 1), dtype=tf.float64))
    assert dtype_layer.compute_dtype == "float64"
    # The nested convolution keeps its default TensorFlow 2.10 float32 policy.
    assert dtype_output.dtype == tf.float32

    config = standard.get_config()
    try:
        PatchEmbedding.from_config(config)
    except TypeError:
        pass
    else:
        raise AssertionError("The documented duplicate-ln_dim limit changed.")
    filtered_config = dict(config)
    filtered_config.pop("ln_dim")
    restored = PatchEmbedding.from_config(filtered_config)
    assert restored.patch_size == 2 and not restored.patchify_with_cnn
    assert restored(images[:1]).shape == (1, 16, 4)

    return {"PatchEmbedding": "passed"}


if __name__ == "__main__":
    print(run_self_tests())
