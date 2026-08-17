"""Discrete timestep and class-condition embedding layers."""

from diffusion.layers.embedding.base_embedding import BaseEmbedding


class ConditionEmbedding(BaseEmbedding):
    """Map integer condition IDs to learned or sinusoidal vectors.

    The practical condition-table modes are ``"new_weight"`` and
    ``"1d_sincos"``. The former is a trainable Keras embedding initialized in
    the usual way. The latter initializes all ``embed_steps`` rows from fixed
    frequencies and remains fixed unless ``embed_trainable=True``. Spatial and
    interpolation modes produce rank-three/four tables and are therefore not
    valid weights for this rank-two lookup layer.

    When ``embed_freq_dim`` differs from ``dim``, the class defaults to a
    one-hidden-layer projection with ratio ``1`` and output width ``dim``.
    Explicit ``mlp_ratio`` or ``mlp_output_dim`` values override those defaults.

    Args:
        **kwargs: :class:`BaseEmbedding` constructor arguments. ``dim`` and
            positive ``embed_steps`` are required. Set
            ``pos_embed_type="new_weight"`` for learned labels, or
            ``pos_embed_type="1d_sincos"`` for timestep-style initialization.
            ``embed_freq_dim`` selects the raw table width; ``embed_trainable``
            controls only non-new initialized tables. Standard Keras ``name``,
            ``dtype``, and ``trainable`` options are also accepted.

    Inputs:
        An integer ``tf.Tensor`` of any shape. Every value must be in
        ``[0, embed_steps)``.

    Outputs:
        A floating tensor with the input shape followed by the embedding or MLP
        output width, normally ``[..., dim]``.

    Serialization:
        The saved config contains an inherited ``ln_dim`` key that duplicates
        the value supplied internally by :class:`BaseEmbedding`. Remove that
        key from a copied config before calling ``ConditionEmbedding.from_config``.
    """

    def __init__(
        self, 
        **kwargs
    ):
        """Create the lookup table and optional frequency projection MLP.

        Arguments and accepted types are documented on the class.

        Returns:
            ``None``.
        """

        super().__init__(**kwargs)
        self._save_init_args(locals())

        self.mlp_ratio = 1 if self.mlp_ratio is None and self.embed_freq_dim is not None \
                        else self.mlp_ratio
        self.mlp_output_dim = self.dim if self.mlp_output_dim is None \
                            and self.embed_freq_dim is not None \
                            else self.mlp_output_dim

        self.embed = self._create_embedding_layer()
        self.embed_mlp = self._create_mlp(
            self.embed_dim
        )

    def call(self, x, training=None):
        """Look up and optionally project discrete conditions.

        Args:
            x: Integer TensorFlow tensor with values in
                ``[0, self.embed_steps)``. For diffusion batches it is commonly
                shaped ``[batch]``.
            training: Optional Keras training flag forwarded to the lookup and
                projection layers. No stochastic operation is introduced here.

        Returns:
            ``tf.Tensor`` with floating compute dtype and shape
            ``x.shape + (output_dim,)``. ``output_dim`` is the raw
            ``embed_dim`` when no MLP exists and otherwise ``mlp_output_dim``.
        """

        x = self.embed(
            x, 
            training=training
        )
        x = self.embed_mlp(
            x, 
            training=training
        ) if self.embed_mlp is not None else x

        return x


def run_self_tests() -> dict[str, str]:
    """Test learned and sinusoidal :class:`ConditionEmbedding` contracts.

    Args:
        None.

    Returns:
        A one-entry success mapping after mode, rank, dtype, projection,
        trainability, boundary-error, gradient, and serialization checks.
    """

    import tensorflow as tf


    learned = ConditionEmbedding(
        dim=4, 
        embed_steps=5, 
        pos_embed_type="new_weight", 
        embed_trainable=False, 
        dtype="float32", 
    )
    ids = tf.constant([0, 2, 4], dtype=tf.int32)
    output = learned(ids, training=False)
    assert output.shape == (3, 4) and output.dtype == tf.float32
    assert learned.embed.trainable
    matrix_output = learned(tf.constant([[0, 1], [3, 4]], dtype=tf.int64))
    assert matrix_output.shape == (2, 2, 4)

    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(learned(ids, training=True))
    gradients = tape.gradient(loss, learned.trainable_variables)
    assert gradients and all(gradient is not None for gradient in gradients)

    for trainable in (False, True):
        sinusoidal = ConditionEmbedding(
            dim=5, 
            embed_steps=6, 
            pos_embed_type="1d_sincos", 
            embed_trainable=trainable, 
        )
        values = sinusoidal(tf.constant([0, 5], dtype=tf.int32), training=True)
        assert values.shape == (2, 5)
        assert sinusoidal.embed.trainable is trainable

    automatic_projection = ConditionEmbedding(
        dim=6, 
        embed_freq_dim=4, 
        embed_steps=5, 
        pos_embed_type="1d_sincos", 
    )
    assert automatic_projection.mlp_ratio == 1
    assert automatic_projection.mlp_output_dim == 6
    assert automatic_projection(tf.constant([1, 2])).shape == (2, 6)

    explicit_projection = ConditionEmbedding(
        dim=6, 
        embed_freq_dim=4, 
        embed_steps=5, 
        pos_embed_type="new_weight", 
        mlp_ratio=2, 
        mlp_output_dim=3, 
        mlp_activation_func="relu", 
    )
    assert explicit_projection(tf.constant([1, 2]), training=False).shape == (2, 3)

    try:
        ConditionEmbedding(dim=4, embed_steps=5, pos_embed_type=None)
    except ValueError:
        pass
    else:
        raise AssertionError("A condition lookup requires an initialized table mode.")

    dtype_layer = ConditionEmbedding(
        dim=4, 
        embed_steps=5, 
        pos_embed_type="1d_sincos", 
        dtype="float64", 
    )
    dtype_output = dtype_layer(tf.constant([0, 1], dtype=tf.int32))
    assert dtype_layer.compute_dtype == "float64"
    # The nested Keras Embedding retains its TensorFlow 2.10 float32 policy.
    assert dtype_output.dtype == tf.float32

    for invalid_id in (5, -1):
        try:
            invalid_output = learned(tf.constant([invalid_id], dtype=tf.int32))
        except tf.errors.InvalidArgumentError:
            pass
        else:
            # TensorFlow's Embedding gather documents device-dependent handling
            # for invalid indices; GPU kernels can return a finite row.
            assert invalid_output.shape == (1, 4)
            assert tf.reduce_all(tf.math.is_finite(invalid_output))

    for unsupported_spatial_mode in (
        "1d_interpolate", "1d_learned_interpolate", "2d_sincos",
        "2d_interpolate", "2d_learned_interpolate",
    ):
        try:
            ConditionEmbedding(
                dim=4, 
                grid_size=2, 
                embed_steps=4, 
                pos_embed_type=unsupported_spatial_mode, 
            )
        except (TypeError, ValueError, tf.errors.InvalidArgumentError):
            pass
        else:
            raise AssertionError(
                f"Spatial table mode {unsupported_spatial_mode!r} must not "
                "silently initialize a rank-two condition lookup."
            )

    config = learned.get_config()
    try:
        ConditionEmbedding.from_config(config)
    except TypeError:
        pass
    else:
        raise AssertionError("The documented duplicate-ln_dim limit changed.")
    filtered_config = dict(config)
    filtered_config.pop("ln_dim")
    restored = ConditionEmbedding.from_config(filtered_config)
    assert restored.dim == 4 and restored.embed_steps == 5
    assert restored(tf.constant([0])).shape == (1, 4)

    return {"ConditionEmbedding": "passed"}


if __name__ == "__main__":
    print(run_self_tests())
