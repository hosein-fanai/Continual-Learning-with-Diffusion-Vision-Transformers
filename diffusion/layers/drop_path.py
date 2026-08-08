import tensorflow as tf

from common.argument_saver import ArgumentSaverLayer


class DropPath(ArgumentSaverLayer):
    """
    DropPath / Stochastic Depth.

    During training, randomly drops an entire residual branch.

    Modes:
        per_sample=True:
            Modern DropPath behavior used in many ViT implementations.
            Each sample in the batch independently keeps/drops the path.

        per_sample=False:
            Batchwise stochastic depth, closer to the original paper's
            "drop this residual layer for the current mini-batch" behavior.

    Args:
        drop_prob:
            Probability of dropping the residual path.
        scale_by_keep:
            If True, divide kept paths by keep_prob so the expected
            residual magnitude is unchanged.
        per_sample:
            If True, use independent masks per sample. If False, use one
            mask for the whole batch.
    """

    def __init__(
        self, 
        drop_prob: float = 0., 
        scale_by_keep: bool = True, 
        per_sample: bool = True, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self._save_init_args(locals())

        assert 0. <= self.drop_prob < 1., \
            "drop_prob must satisfy 0.0 <= drop_prob < 1.0 ."

    def call(self, x, training=None):
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
