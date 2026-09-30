"""Stochastic-depth regularization for complete residual paths.

DropPath implements stochastic depth by sharing a Bernoulli mask across each
residual path, either independently per sample or across the batch. Training may
rescale retained paths by inverse keep probability; evaluation preserves inputs.
"""

import tensorflow as tf

from functools import partial

from typing import Any

from common.argument_saver import ArgumentSaverLayer
from common.random import SeedStream


class DropPath(ArgumentSaverLayer):
    """Drop complete residual paths during training.

    Unlike elementwise dropout, every non-batch axis shares the same mask. A
    kept path can be divided by its keep probability so its expected magnitude
    matches inference.

    Args:
        drop_prob (float): Float in ``[0, 1)`` giving the drop probability.
            Defaults to ``0.0``.
        scale_by_keep (bool): Divide kept paths by ``1 - drop_prob`` when true.
            Defaults to ``True``.
        per_sample (bool): If true, use a ``[batch, 1, ..., 1]`` mask so each example
            is independent. If false, one scalar-like mask is shared by the
            entire batch.
            Defaults to ``True``.
        **kwargs (Any): Standard ``tf.keras.layers.Layer`` options such as ``name``,
            ``dtype``, and ``trainable``.

    Inputs:
        A floating ``tf.Tensor`` of any rank at least one. Common shapes are
        ``[batch, tokens, channels]`` and ``[batch, height, width, channels]``.

    Outputs:
        A tensor with exactly the input shape and dtype. Evaluation is an
        identity operation; training returns either zero or a retained path.

    Attributes:
        drop_prob (float): Drop probability retained from construction.
        scale_by_keep (bool): Whether retained paths preserve their expectation.
        per_sample (bool): Whether mask decisions are independent across batch rows.
        seed (int | None): Optional initial seed for the saved path-mask stream.
    """

    def __init__(
        self, 
        drop_prob: float = 0., 
        scale_by_keep: bool = True, 
        per_sample: bool = True, 
        seed: int | None = None, 
        **kwargs: Any
    ) -> None:
        """Initialize stochastic-depth probability and mask semantics.

        Args:
            drop_prob (float): Caller-supplied path-drop probability in ``[0, 1)``.
                Stored without a project range check. Defaults to ``0.0``.
            scale_by_keep (bool): Whether retained paths are divided by their
                keep probability.
                Defaults to ``True``.
            per_sample (bool): Whether examples receive independent masks.
                Defaults to ``True``.
            seed (int | None): Initial stream seed. Defaults to ``None`` for a
                randomly initialized stream. Masks advance on every training call;
                reset_seed restores a sequence and weight checkpoints retain its counter.
            **kwargs (Any): Standard Keras layer options.

        Returns:
            None: No value is returned.

        Raises:
            ValueError: If seed is outside the shared NumPy/TensorFlow supported
                seed interval.
        """

        super().__init__(**kwargs)
        self._save_init_args(locals())

        self.seed = None if self.seed is None else int(self.seed)
        self.random_stream = SeedStream(self.seed, name=f"{self.name}__path_random")

    def _apply_mask(self, x: tf.Tensor) -> tf.Tensor:
        """Apply one stochastic path mask to a training tensor.

        Args:
            x (tf.Tensor): Floating tensor shaped ``[batch, ...]``.

        Returns:
            tf.Tensor: Masked tensor with the same shape and dtype.

        Side Effects:
            Advances the checkpointed counter used for this path mask.

        Raises:
            TypeError: x has a dtype that cannot participate in the floating mask
                arithmetic.
            tf.errors.InvalidArgumentError: Per-sample masking receives a scalar x,
                so the requested rank-minus-one mask shape is invalid.
        """

        keep_prob = 1. - self.drop_prob
        x_shape = tf.shape(x)
        rank = tf.rank(x)

        # Draw an independent path decision for each example when per-sample masking is
        # enabled.
        if self.per_sample:
            mask_shape = tf.concat([
                x_shape[:1], 
                tf.ones(tuple([rank - 1]), dtype=tf.int32)
            ], axis=0)
        # Share one path decision across the full batch in batchwise masking mode.
        else:
            mask_shape = tf.ones(tuple([rank]), dtype=tf.int32)

        random_tensor = keep_prob + tf.random.stateless_uniform(
            mask_shape, 
            minval=0., 
            maxval=1., 
            seed=self.random_stream.next_seed(), 
            dtype=x.dtype
        )
        binary_mask = tf.floor(random_tensor)
        # Rescale retained paths to preserve their expected magnitude when requested.
        if self.scale_by_keep:
            binary_mask = binary_mask / keep_prob

        return x * binary_mask

    def reset_seed(self, seed: int | None) -> None:
        """Start the task's new path-mask sequence without changing weights.

        Args:
            seed (int | None): Integer base seed restarts the sequence. None
                marks the configuration unseeded while preserving the live
                stream's saved base seed and advancing counter.

        Returns:
            result (None): Updates seed metadata and, for an integer seed,
                resets the checkpointed counter. Learned weights are unchanged.

        Raises:
            ValueError: SeedStream rejects an integer outside the supported seed
                interval; invalid values are not clipped.
        """

        self.seed = seed
        self.random_stream.reset_seed(seed)

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        """Apply a training-only path mask.

        Args:
            x (tf.Tensor): Floating tensor of shape ``[batch, ...]``. Its dtype is
                also used to generate the random mask.
            training (bool | tf.Tensor | None): Training flag. Defaults to ``None``, returning identity when
                this method receives None. False also preserves inputs; True or a true scalar tensor samples
                a mask. Keras __call__ may supply a surrounding training context before invoking this
                method.

        Returns:
            tf.Tensor: with the same shape and dtype as ``x``. With
            ``scale_by_keep=True``, retained values are divided by the keep
            probability. Active training advances the checkpointed random counter;
            inference and a zero drop probability do not draw a mask.

        Raises:
            TypeError: Active masking receives an input dtype incompatible with
                floating random masks.
            ValueError or tf.errors.InvalidArgumentError: A tensor training flag is not scalar,
                or per-sample masking receives a rank-zero input.
        """

        # Return identity for zero drop probability 
        # or an unspecified training flag.
        if self.drop_prob == 0. or training is None:
            return x

        # Resolve ordinary Python training flags directly; 
        # tensor flags use graph control flow below.
        if not tf.is_tensor(training):
            # Apply stochastic depth for training; 
            # preserve the input for evaluation.
            return self._apply_mask(x) if training else x
        return tf.cond(
            tf.cast(training, tf.bool), 
            partial(self._apply_mask, x), 
            partial(tf.identity, x)
        )


def run_self_tests() -> dict[str, str]:
    """Run deterministic tests for every :class:`DropPath` mode.

    Args:
        None.

    Returns:
        dict[str, str]: A one-entry success mapping after configuration retention, rank/dtype,
        train/evaluation, scaling, mask-sharing, and config checks pass.
    """

    import numpy as np


    for configured_probability in (-0.01, 1.0, 2.0):
        configured = DropPath(drop_prob=configured_probability)
        assert configured.drop_prob == configured_probability

    x = tf.ones((64, 3, 2), dtype=tf.float32)
    identity = DropPath(drop_prob=0.0)
    for training in (None, False, True):
        assert identity(x, training=training) is x

    near_one = DropPath(drop_prob=0.999, seed=313)
    assert near_one(x, training=False) is x
    near_one_output = near_one(x, training=True)
    assert near_one_output.shape == x.shape
    assert bool(tf.reduce_all(tf.math.is_finite(near_one_output)))

    tf.random.set_seed(31415)
    per_sample = DropPath(
        drop_prob=0.5, 
        scale_by_keep=True, 
        per_sample=True 
    )(x, training=True)
    assert per_sample.shape == x.shape and per_sample.dtype == x.dtype
    unique_values = set(np.unique(per_sample.numpy()).tolist())
    assert unique_values.issubset({0.0, 2.0})
    assert np.all(per_sample.numpy() == per_sample.numpy()[:, :1, :1])

    tf.random.set_seed(2718)
    unscaled = DropPath(
        drop_prob=0.5, 
        scale_by_keep=False, 
        per_sample=True 
    )(x, training=tf.constant(True))
    assert set(np.unique(unscaled.numpy()).tolist()).issubset({0.0, 1.0})

    tf.random.set_seed(9)
    shared = DropPath(
        drop_prob=0.5, 
        scale_by_keep=False, 
        per_sample=False, 
        dtype="float64" 
    )(tf.ones((4, 2, 2, 3), dtype=tf.float64), training=True)
    assert shared.dtype == tf.float64 and shared.shape == (4, 2, 2, 3)
    assert np.all(shared.numpy() == shared.numpy().reshape(-1)[0])

    tf.random.set_seed(10)
    shared_scaled = DropPath(
        drop_prob=0.5, 
        scale_by_keep=True, 
        per_sample=False 
    )(tf.ones((4, 2, 3), dtype=tf.float32), training=True)
    assert shared_scaled.shape == (4, 2, 3)
    assert set(np.unique(shared_scaled.numpy()).tolist()).issubset({0.0, 2.0})
    assert np.all(shared_scaled.numpy() == shared_scaled.numpy().reshape(-1)[0])

    rank_one = DropPath(drop_prob=0.25, per_sample=True)(
        tf.ones(tuple([8]), dtype=tf.float32), 
        training=True
    )
    assert rank_one.shape == tuple([8])

    layer = DropPath(drop_prob=0.25, scale_by_keep=False, per_sample=False)
    restored = DropPath.from_config(layer.get_config())
    assert restored.drop_prob == 0.25
    assert not restored.scale_by_keep and not restored.per_sample
    assert restored(tf.ones((1, 1)), training=False).shape == (1, 1)

    return {"DropPath": "passed"}


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
