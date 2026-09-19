"""Retain float64 attention-scale precision with native Keras attention."""

from __future__ import annotations

from tensorflow.keras import layers

import numpy as np

from common.keras_registry import register_canonical_keras_serializable


@register_canonical_keras_serializable(package="continual_learning")
class PolicyMultiHeadAttention(layers.MultiHeadAttention):
    """Correct float64 scaling while inheriting native Keras attention behavior.

    Keras 3 propagates dtype policies and manages seeded dropout itself. Only
    the attention scaling constant needs a float64 correction. The registered
    class name is retained for loading existing serialized configurations.
    """

    def _build_attention(self, rank: int) -> None:
        """Build native attention and preserve its float64 scaling constant."""

        super()._build_attention(rank)
        # Keras 3 casts a Python float through float32 before float64. Keep the
        # cached scale typed so the existing double-precision equation is exact.
        if self.variable_dtype == "float64":
            self._inverse_sqrt_key_dim = np.float64(self._inverse_sqrt_key_dim)


def run_self_tests() -> dict[str, str]:
    """Check analytical single-head attention and registered config reconstruction."""

    import tensorflow as tf

    attention = PolicyMultiHeadAttention(num_heads=1, key_dim=2, use_bias=False,
                                         dtype="float64", name="attention_policy_probe")
    inputs = tf.constant([[[1., 0.], [0., 1.]]], dtype=tf.float64)
    attention(inputs, inputs, training=False)
    assert all(weight.dtype == tf.float64 for weight in attention.weights)
    projections = [np.eye(2).reshape(2, 1, 2)] * 3 + [np.eye(2).reshape(1, 2, 2)]
    attention.set_weights(projections)
    diagonal = np.exp(1. / np.sqrt(2.))
    expected = np.array([[[diagonal, 1.], [1., diagonal]]]) / (diagonal + 1.)
    np.testing.assert_allclose(attention(inputs, inputs).numpy(), expected, atol=1e-12, rtol=1e-12)
    clone = tf.keras.layers.deserialize(tf.keras.layers.serialize(attention))
    clone(inputs, inputs, training=False)
    clone.set_weights(attention.get_weights())
    assert [weight.name for weight in clone.weights] == [weight.name for weight in attention.weights]
    np.testing.assert_array_equal(clone(inputs, inputs).numpy(), attention(inputs, inputs).numpy())
    return {"PolicyMultiHeadAttention": "passed"}
