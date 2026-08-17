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
