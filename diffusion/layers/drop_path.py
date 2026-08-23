"""Stochastic-depth regularization for complete residual paths."""

import tensorflow as tf

from typing import Any

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
        **kwargs: Any
    ) -> None:
        """Initialize stochastic-depth probability and mask semantics.

        Args:
            drop_prob (float): Path-drop probability in ``[0, 1)``.
            scale_by_keep (bool): Whether retained paths are divided by their
                keep probability.
            per_sample (bool): Whether examples receive independent masks.
            **kwargs (Any): Standard Keras layer options.

        Returns:
            ``None``.
        """

        super().__init__(**kwargs)
        self._save_init_args(locals())

        # Keep the path-drop probability within its valid half-open interval.
        if not 0. <= self.drop_prob < 1.:
            raise ValueError("drop_prob must satisfy 0.0 <= drop_prob < 1.0.")

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        """Apply a training-only path mask.

        Args:
            x (tf.Tensor): Floating tensor of shape ``[batch, ...]``. Its dtype is
                also used to generate the random mask.
            training (bool | None): Optional boolean training flag. ``None`` is treated as
                false; false returns ``x`` unchanged.

        Returns:
            ``tf.Tensor`` with the same shape and dtype as ``x``. With
            ``scale_by_keep=True``, retained values are divided by the keep
            probability.
        """

        training = False if training is None else training

        # Skip stochastic masking during evaluation or when dropping is disabled.
        if not training or self.drop_prob == 0.:
            return x

        keep_prob = 1. - self.drop_prob

        # Use dynamic rank so this works for:
        #   [B, tokens, channels]
        #   [B, H, W, C]
        #   [B, ...]
        x_shape = tf.shape(x)
        rank = tf.rank(x)

        # Draw one broadcastable path decision per example when requested.
        if self.per_sample:
            # Shape: [B, 1, 1, ..., 1]
            mask_shape = tf.concat(
                [x_shape[:1], tf.ones((rank - 1,), dtype=tf.int32)],
                axis=0
            )
        # Otherwise share one path decision across the entire batch.
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

        # Rescale retained paths to preserve their expected magnitude.
        if self.scale_by_keep:
            binary_mask = binary_mask / keep_prob

        return x * binary_mask


def run_self_tests() -> dict[str, str]:
    """Run deterministic tests for every :class:`DropPath` mode.

    Args:
        None.

    Returns:
        dict[str, str]: A one-entry success mapping after probability boundaries, rank/dtype,
        train/evaluation, scaling, mask-sharing, and config checks pass.
    """

    import numpy as np


    for invalid_probability in (-0.01, 1.0, 2.0):
        try:
            DropPath(drop_prob=invalid_probability)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid drop probabilities must be rejected.")

    x = tf.ones((64, 3, 2), dtype=tf.float32)
    identity = DropPath(drop_prob=0.0)
    for training in (None, False, True):
        assert identity(x, training=training) is x

    tf.random.set_seed(31415)
    per_sample = DropPath(
        drop_prob=0.5, 
        scale_by_keep=True, 
        per_sample=True, 
    )(x, training=True)
    assert per_sample.shape == x.shape and per_sample.dtype == x.dtype
    unique_values = set(np.unique(per_sample.numpy()).tolist())
    assert unique_values.issubset({0.0, 2.0})
    assert np.all(per_sample.numpy() == per_sample.numpy()[:, :1, :1])

    tf.random.set_seed(2718)
    unscaled = DropPath(
        drop_prob=0.5, 
        scale_by_keep=False, 
        per_sample=True, 
    )(x, training=tf.constant(True))
    assert set(np.unique(unscaled.numpy()).tolist()).issubset({0.0, 1.0})

    tf.random.set_seed(9)
    shared = DropPath(
        drop_prob=0.5, 
        scale_by_keep=False, 
        per_sample=False, 
        dtype="float64", 
    )(tf.ones((4, 2, 2, 3), dtype=tf.float64), training=True)
    assert shared.dtype == tf.float64 and shared.shape == (4, 2, 2, 3)
    assert np.all(shared.numpy() == shared.numpy().reshape(-1)[0])

    tf.random.set_seed(10)
    shared_scaled = DropPath(
        drop_prob=0.5, 
        scale_by_keep=True, 
        per_sample=False, 
    )(tf.ones((4, 2, 3), dtype=tf.float32), training=True)
    assert shared_scaled.shape == (4, 2, 3)
    assert set(np.unique(shared_scaled.numpy()).tolist()).issubset({0.0, 2.0})
    assert np.all(shared_scaled.numpy() == shared_scaled.numpy().reshape(-1)[0])

    rank_one = DropPath(drop_prob=0.25, per_sample=True)(
        tf.ones((8,), dtype=tf.float32), 
        training=True
    )
    assert rank_one.shape == (8,)

    layer = DropPath(drop_prob=0.25, scale_by_keep=False, per_sample=False)
    restored = DropPath.from_config(layer.get_config())
    assert restored.drop_prob == 0.25
    assert not restored.scale_by_keep and not restored.per_sample
    assert restored(tf.ones((1, 1)), training=False).shape == (1, 1)

    return {"DropPath": "passed"}


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
