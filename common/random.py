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

    def __init__(self, seed: int | None = None, **kwargs: object) -> None:
        """Create a saved PHILOX counter and base seed for stateless draws.

        Args:
            seed (int | None): Nonnegative seed below ``2**32``. None obtains
                a base seed from Python's current random stream.
            **kwargs (object): Keras Layer options such as name and dtype.
                The RNG state always uses int64, independently of compute dtype.

        Returns:
            result (None): The layer owns one nontrainable int64 state of shape
                ``(3,)``; no random tensor has yet been drawn.

        Raises:
            ValueError: If the seed or Keras layer options are invalid.
        """

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
        """Return a stateless-RNG seed and advance this stream atomically once.

        Args:
            seed (int | None): Optional base-seed override for this draw, in
                ``[0, 2**32)``. None uses the saved base seed. An override does
                not replace that saved seed or reset the advancing counter.

        Returns:
            draw_seed (tf.Tensor): Int32 tensor of shape ``(2,)`` containing
                the base seed and draw counter, suitable for stateless TF RNGs.
                Concurrent calls receive distinct counters; row assignment
                depends on scheduling. The int32 counter wraps after 2**32 draws.

        Raises:
            ValueError: If an explicit seed is outside its supported interval.
        """

        seed = effective_seed(None, seed)
        # A Keras read followed by assign_add loses increments in parallel
        # Dataset.map calls. PHILOX skip returns its old state atomically and
        # is supported by XLA, while sharing the checkpointed Keras weight.
        state = self._generator.skip(1)
        counter = tf.bitwise.right_shift(state[0], tf.constant(8, tf.int64))
        base_seed = state[2] if seed is None else tf.cast(seed, tf.int64)

        return tf.cast(tf.stack((base_seed, counter)), tf.int32)

    def reset_seed(self, seed: int | None) -> None:
        """Restart an explicitly seeded stream while retaining unseeded state.

        Args:
            seed (int | None): New seed in ``[0, 2**32)``. An integer sets the
                saved counter to zero. None marks configuration as unseeded
                and leaves the live counter/base seed unchanged.

        Returns:
            result (None): Integer seeds restart subsequent draws reproducibly;
                None preserves the next draw of the existing stream.

        Raises:
            ValueError: If the seed is outside its supported interval.
        """

        self.seed = effective_seed(None, seed)
        # An absent seed intentionally leaves the advancing counter unchanged.
        if self.seed is not None:
            self.state.assign([0, 0, self.seed])

    def get_config(self) -> dict[str, object]:
        """Describe reconstruction independently of the current random counter.

        Returns:
            config (dict[str, object]): Keras layer settings plus the configured
                integer/None seed. Restore layer weights as well when continuing
                an existing random sequence; config alone creates a fresh stream.
        """

        return {**super().get_config(), "seed": self.seed}
