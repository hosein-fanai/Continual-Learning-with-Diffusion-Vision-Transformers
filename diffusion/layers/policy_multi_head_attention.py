"""Preserve the parent's numerical policy inside TensorFlow 2.10 attention."""

from __future__ import annotations

from tensorflow.keras import layers

from common.keras_registry import register_canonical_keras_serializable


@register_canonical_keras_serializable(package="continual_learning")
class PolicyMultiHeadAttention(layers.MultiHeadAttention):
    """Apply the configured dtype to projections and attention normalization.

    Keras 2.10's attention factory omits dtype when constructing internal
    projection, softmax and dropout layers. Their global-policy default can
    silently change saved model precision when a config is restored. These
    factory overrides preserve the standard attention equations, names and
    weight topology while making its child policy explicit.
    """

    def _get_common_kwargs_for_sublayer(self) -> dict[str, object]:
        """Add the attention policy to Keras' projection-layer constructor options."""

        options = super()._get_common_kwargs_for_sublayer()
        options["dtype"] = self.dtype_policy

        return options

    def _build_attention(self, rank: int) -> None:
        """Keep attention-score normalization and dropout in the same compute policy."""

        super()._build_attention(rank)
        self._softmax = layers.Softmax(
            axis=self._softmax.axis, 
            dtype=self.dtype_policy
        )
        self._dropout_layer = layers.Dropout(
            rate=self._dropout, 
            dtype=self.dtype_policy
        )


def run_self_tests() -> dict[str, str]:
    """Check analytical single-head attention and registered config reconstruction."""

    import numpy as np
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
