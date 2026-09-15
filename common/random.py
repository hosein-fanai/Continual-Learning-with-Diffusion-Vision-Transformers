"""Checkpointed random streams for TensorFlow graphs and XLA compilation."""

from __future__ import annotations

import tensorflow as tf

import random

from common.keras_registry import register_canonical_keras_serializable
from common.runtime import effective_seed


@register_canonical_keras_serializable()
class SeedStream(tf.keras.layers.Layer):
    """Advance saved seeds atomically without XLA's stateful-RNG seed loss.

    Separate instances isolate independent random operations. Explicit draw
    seeds select a different seed while retaining the advancing counter;
    ``reset_seed`` starts a reproducible new sequence. Draw sequences differ
    from legacy TensorFlow stateful RNG, whose seeds XLA ignores.
    Concurrent calls receive distinct counters, but their assignment to input
    rows depends on execution order; use serial mapping for reproducible rows.
    """

    def __init__(self, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.seed = effective_seed(None, seed)
        initial_seed = self.seed if self.seed is not None else random.randrange(2**31)
        # PHILOX stores a 128-bit counter followed by its 64-bit key. The key
        # retains our base seed; skip(1) advances the low counter by 256.
        self.state = self.add_weight(
            name="seed_state", 
            shape=(3,), 
            dtype="int64", 
            trainable=False, 
            autocast=False, 
            initializer=tf.keras.initializers.Constant([0, 0, initial_seed])
        )
        self._generator = tf.random.Generator(
            state=self.state.value, 
            alg=tf.random.Algorithm.PHILOX,
        )
        self.built = True

    def next_seed(self, seed: int | None = None) -> tf.Tensor:
        """Return an int32 stateless-RNG seed and advance this stream once."""

        seed = effective_seed(None, seed)
        # A Keras read followed by assign_add loses increments in parallel
        # Dataset.map calls. PHILOX skip returns its old state atomically and
        # is supported by XLA, while sharing the checkpointed Keras weight.
        state = self._generator.skip(1)
        counter = tf.bitwise.right_shift(state[0], tf.constant(8, tf.int64))
        base_seed = state[2] if seed is None else tf.cast(seed, tf.int64)

        return tf.cast(tf.stack((base_seed, counter)), tf.int32)

    def reset_seed(self, seed: int | None) -> None:
        """Reset an explicitly seeded stream; None retains unseeded state."""

        self.seed = effective_seed(None, seed)
        if self.seed is not None:
            self.state.assign([0, 0, self.seed])

    def get_config(self):
        return {**super().get_config(), "seed": self.seed}
