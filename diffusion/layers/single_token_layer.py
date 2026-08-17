"""Learned or input-provided single-token embeddings."""

import tensorflow as tf
from tensorflow.keras import initializers

from diffusion.layers.embedding.base_embedding import BaseEmbedding


class SingleTokenLayer(BaseEmbedding):
    """Produce one token for a class marker, BOS marker, or condition.

    The token is either a trainable weight or a per-example vector supplied at
    call time. An optional learned positional vector is merged by addition or
    concatenation. With concatenation, ``dim // 2`` channels are allocated to
    each half, so an even ``dim`` is required to retain exactly ``dim`` output
    channels.

    Args:
        with_pos_embed: If true, force a learned ``"new_weight"`` positional
            vector and batch-broadcast the trainable token through the merge.
            If false, disable positional embeddings.
        input_as_token: If true, use the second call input shaped
            ``[batch, token_dim]``. If false, ignore that input and use one
            trainable token shaped ``[1, 1, embed_dim]``.
        **kwargs: :class:`BaseEmbedding` arguments. Required ``dim`` sets the
            target width. Common keys are ``pos_merger_type``,
            ``embed_freq_dim``, ``mlp_ratio``, ``mlp_output_dim``, ``name``,
            and ``dtype``. ``pos_embed_type`` is overridden by
            ``with_pos_embed``.

    Inputs:
        Pair ``(images, token)``. ``images`` may be any tensor with batch as
        axis 0; only its batch size is read. ``token`` must be a floating
        ``[batch, token_dim]`` tensor when ``input_as_token=True`` and may be
        ``None`` otherwise.

    Outputs:
        Floating token tensor shaped ``[batch, 1, output_dim]`` when using an
        input token or a positional embedding. With a trainable token and
        ``with_pos_embed=False``, the current implementation leaves its batch
        dimension at ``1``; callers concatenating it to a larger batch must
        explicitly account for that shape.

    Serialization:
        The saved config contains an inherited ``ln_dim`` key that duplicates
        the value supplied internally by :class:`BaseEmbedding`. Remove that
        key from a copied config before calling ``SingleTokenLayer.from_config``.
    """

    def __init__(
        self, 
        with_pos_embed: bool = True, 
        input_as_token: bool = False, 
        **kwargs
    ):
        """Create the trainable token, positional vector, and projections.

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
        self.pos_embed_type = "new_weight" if self.with_pos_embed else None
        self.embed_dim = self.dim // 2 if self.pos_merger_type == "concat" else self.dim

        self.token = self.add_weight(
            shape=(1, 1, self.embed_dim), 
            initializer=initializers.RandomNormal(stddev=1e-6), # or "zeros"
            trainable=True, 
            name=f"{self.name}/token_embeddings"
        ) if not self.input_as_token else None
        self.token_mlp = self._create_mlp(
            self.embed_dim
        ) if self.token is not None else None
        self.pos_embed = self._create_embeddings(
            output_grid_size=1, 
        )
        self.pos_embed_mlp = self._create_mlp(
            self.embed_dim
        ) if self.pos_embed is not None else None

    def call(self, inputs, training=None):
        """Resolve and return the configured single token.

        Args:
            inputs: Pair ``(images, token)`` described by the class contract.
            training: Optional Keras training flag forwarded to token and
                positional projection MLPs.

        Returns:
            ``tf.Tensor`` of token embeddings. Its last dimension is ``dim``
            for the usual additive setup and, for concatenation, twice
            ``dim // 2`` unless an MLP overrides the component widths.
        """

        images, token = inputs

        x = token[:, None, :] if self.input_as_token else self.token
        x = self.token_mlp(
            x, 
            training=training
        ) if self.token_mlp is not None else x
        x = self._pos_merger(
            x, 
            batch_size=tf.shape(images)[0], 
            training=training
        )

        return x
