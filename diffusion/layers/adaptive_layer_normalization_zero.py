"""Conditioned, zero-initialized adaptive layer-normalization primitives."""

import tensorflow as tf
from tensorflow.keras import layers, models

from common.argument_saver import ArgumentSaverLayer


class AdaLNZero(ArgumentSaverLayer):
    """Normalize token features and modulate them from a condition vector.

    The layer performs non-affine layer normalization on ``x`` and predicts a
    per-example shift, scale, and optionally residual gate from ``cond``.  The
    final conditioning projection is initialized to zero, so a newly created
    adaptive layer initially returns ordinary normalized features and a zero
    gate.  Transformer blocks use that gate to introduce their residual branch
    gradually during training.

    Args:
        dim: Size of the last axis of ``x`` and of the shift/scale vectors.
        gate_dim: Number of gate channels. ``None`` uses ``dim``. It may differ
            from ``dim`` when the gated branch projects to another width.
        mlp_ratio: If provided, insert a dense hidden layer of width
            ``int(dim * mlp_ratio)`` before the zero-initialized projection.
            ``None`` applies only a Swish activation before that projection.
        return_gate: Whether :meth:`call` returns ``(features, gate)`` instead
            of only the modulated features.
        no_adaptation: If true, omit the conditioning MLP, ignore ``cond``, and
            return plain normalized features. The accompanying gate, when
            requested, is the scalar ``1.0`` rather than a learned tensor.
        epsilon: Small positive float added to the normalization variance.
        **kwargs: Standard ``tf.keras.layers.Layer`` options, including
            ``name``, ``dtype``, and ``trainable``.

    Inputs:
        A pair ``(x, cond)``. ``x`` is normally a floating tensor shaped
        ``[batch, tokens, dim]``. ``cond`` is a floating tensor shaped
        ``[batch, condition_dim]``; it is unused when ``no_adaptation=True``.

    Outputs:
        A floating tensor shaped like ``x``. If ``return_gate=True``, the
        output is ``(features, gate)`` where an adaptive gate has shape
        ``[batch, 1, gate_dim]`` and is intended to broadcast over tokens.
    """

    def __init__(
        self, 
        dim: int, 
        gate_dim: int | None = None, 
        mlp_ratio: float | None = None, 
        return_gate: bool = True, 
        no_adaptation: bool = False, 
        epsilon: float = 1e-6, 
        **kwargs
    ):
        """Initialize the normalization and optional conditioning network.

        Arguments and accepted types are documented on the class.

        Returns:
            ``None``.
        """

        super().__init__(**kwargs)
        self._save_init_args(locals())

        self.gate_dim = self.gate_dim if self.gate_dim is not None else self.dim

        self.norm = layers.LayerNormalization(
            center=False, 
            scale=False, 
            epsilon=self.epsilon, 
            name="layer_norm"
        )

        self.mlp_output_dim = self.dim * 2 + (
            self.gate_dim if self.return_gate else 0
        )
        if self.mlp_ratio is None:
            mlp_first_layer = layers.Activation(
                "swish", 
                name=f"{self.name}/mlp/first_layer"
            )
        else:
            mlp_first_layer = layers.Dense(
                int(self.dim * self.mlp_ratio), 
                activation="swish", 
                # kernel_initializer="zeros", 
                name=f"{self.name}/mlp/first_layer"
            )
        self.mlp = models.Sequential([
            mlp_first_layer, 
            layers.Dense(
                self.mlp_output_dim, 
                kernel_initializer="zeros", 
                name=f"{self.name}/mlp/final_layer"
            )
        ], name="mlp") if not self.no_adaptation else None

    def call(self, inputs, training=None):
        """Apply conditional normalization.

        Args:
            inputs: Pair ``(x, cond)`` following the class-level input
                contract. The feature dtype must be supported by Keras layer
                normalization; the condition must be compatible with the
                layer's compute dtype.
            training: Optional Python boolean or Keras training flag forwarded
                to the nested normalization and dense layers. This layer has no
                stochastic training-only operation.

        Returns:
            ``tf.Tensor`` with the shape and dtype of normalized ``x``, or a
            ``tuple[tf.Tensor, tf.Tensor | float]`` when ``return_gate`` is
            enabled. See the class-level output contract for gate shapes.
        """

        x, cond = inputs

        h = self.norm(x, training=training)

        if self.no_adaptation:
            if self.return_gate:
                return h, 1.
            return h

        params = self.mlp(cond, training=training)
        params = tf.expand_dims(params, 1)

        if self.return_gate:
            shift, scale, gate = tf.split(
                params,
                [self.dim, self.dim, self.gate_dim],
                axis=-1
            )
            return h * (1 + scale) + shift, gate

        shift, scale = tf.split(params, 2, axis=-1)
        return h * (1 + scale) + shift
