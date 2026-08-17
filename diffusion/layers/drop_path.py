"""Stochastic-depth regularization for complete residual paths."""

import tensorflow as tf

from common.argument_saver import ArgumentSaverLayer


class DropPath(ArgumentSaverLayer):
    """Drop complete residual paths during training.

    Unlike elementwise dropout, every non-batch axis shares the same mask. A
    kept path can be divided by its keep probability so its expected magnitude
    matches inference.

    Args:
        drop_prob: Float in ``[0, 1)`` giving the drop probability.
        scale_by_keep: Divide kept paths by ``1 - drop_prob`` when true.
        per_sample: If true, use a ``[batch, 1, ..., 1]`` mask so each example
            is independent. If false, one scalar-like mask is shared by the
            entire batch.
        **kwargs: Standard ``tf.keras.layers.Layer`` options such as ``name``,
            ``dtype``, and ``trainable``.

    Inputs:
        A floating ``tf.Tensor`` of any rank at least one. Common shapes are
        ``[batch, tokens, channels]`` and ``[batch, height, width, channels]``.

    Outputs:
        A tensor with exactly the input shape and dtype. Evaluation is an
        identity operation; training returns either zero or a retained path.
    """

    def __init__(
        self, 
        drop_prob: float = 0., 
        scale_by_keep: bool = True, 
        per_sample: bool = True, 
        **kwargs
    ):
        """Initialize stochastic-depth probability and mask semantics.

        Arguments and accepted types are documented on the class.

        Returns:
            ``None``.
        """

        super().__init__(**kwargs)
        self._save_init_args(locals())

        assert 0. <= self.drop_prob < 1., \
            "drop_prob must satisfy 0.0 <= drop_prob < 1.0 ."

    def call(self, x, training=None):
        """Apply a training-only path mask.

        Args:
            x: Floating ``tf.Tensor`` of shape ``[batch, ...]``. Its dtype is
                also used to generate the random mask.
            training: Optional boolean training flag. ``None`` is treated as
                false; false returns ``x`` unchanged.

        Returns:
            ``tf.Tensor`` with the same shape and dtype as ``x``. With
            ``scale_by_keep=True``, retained values are divided by the keep
            probability.
        """

        training = False if training is None else training

        if not training or self.drop_prob == 0.:
            return x

        keep_prob = 1. - self.drop_prob

        # Use dynamic rank so this works for:
        #   [B, tokens, channels]
        #   [B, H, W, C]
        #   [B, ...]
        x_shape = tf.shape(x)
        rank = tf.rank(x)

        if self.per_sample:
            # Shape: [B, 1, 1, ..., 1]
            mask_shape = tf.concat(
                [x_shape[:1], tf.ones((rank - 1,), dtype=tf.int32)],
                axis=0
            )
        else:
            # Shape: [1, 1, 1, ..., 1], one decision for whole batch.
            mask_shape = tf.ones((rank,), dtype=tf.int32)

        random_tensor = keep_prob + tf.random.uniform(
            mask_shape, 
            minval=0., 
            maxval=1., 
            dtype=x.dtype
        )
        binary_mask = tf.floor(random_tensor)

        if self.scale_by_keep:
            binary_mask = binary_mask / keep_prob

        return x * binary_mask
