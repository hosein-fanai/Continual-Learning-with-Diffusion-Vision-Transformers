"""Conditioned, zero-initialized adaptive layer-normalization primitives.

AdaLNZero combines non-affine layer normalization with condition-derived shifts,
scales, and optional residual gates. Its final conditioning projection starts at
zero; plain-normalization mode omits adaptation and supplies an identity gate.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from typing import Any

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
        dim (int | None): Size of the last axis of ``x`` and of the shift/scale vectors.
            It may be ``None`` only when ``no_adaptation=True``.
        gate_dim (int | None): Number of gate channels. ``None`` uses ``dim``. It may differ
            from ``dim`` when the gated branch projects to another width.
            Defaults to ``None``.
        mlp_ratio (float | None): If provided, insert a dense hidden layer of width
            ``int(dim * mlp_ratio)`` before the zero-initialized projection.
            ``None`` applies only a Swish activation before that projection.
            Defaults to ``None``.
        return_gate (bool): Whether :meth:`call` returns ``(features, gate)`` instead
            of only the modulated features.
            Defaults to ``True``.
        no_adaptation (bool): If true, omit the conditioning MLP, ignore ``cond``, and
            return plain normalized features. The accompanying gate, when
            requested, is the scalar ``1.0`` rather than a learned tensor.
            Defaults to ``False``.
        epsilon (float): Small positive float added to the normalization variance.
            Defaults to ``1e-06``.
        **kwargs (Any): Standard ``tf.keras.layers.Layer`` options, including
            ``name``, ``dtype``, and ``trainable``.

    Inputs:
        A pair ``(x, cond)``. ``x`` is normally a floating tensor shaped
        ``[batch, tokens, dim]``. ``cond`` is a floating tensor shaped
        ``[batch, condition_dim]``; it is unused when ``no_adaptation=True``.

    Outputs:
        A floating tensor shaped like ``x``. If ``return_gate=True``, the
        output is ``(features, gate)`` where an adaptive gate has shape
        ``[batch, 1, gate_dim]`` and is intended to broadcast over tokens.

    Attributes:
        norm (tf.keras.layers.LayerNormalization): Non-affine normalizer using the layer
            policy.
        mlp (tf.keras.Sequential | None): Condition-to-shift/scale/gate projection, absent
            in plain mode.
        gate_dim (int | None): Resolved residual-gate width; unused when adaptation is
            disabled.
    """

    def __init__(
        self, 
        dim: int | None,
        gate_dim: int | None = None, 
        mlp_ratio: float | None = None, 
        return_gate: bool = True, 
        no_adaptation: bool = False, 
        epsilon: float = 1e-6, 
        **kwargs: Any
    ) -> None:
        """Initialize the normalization and optional conditioning network.

        Args:
            dim (int | None): Feature width normalized and modulated by the
                layer, or ``None`` for inferred-width plain normalization.
            gate_dim (int | None): Optional residual-gate width; ``None`` uses
                ``dim``.
                Defaults to ``None``.
            mlp_ratio (float | None): Conditioning-MLP hidden-width ratio. Defaults to ``None``, using Swish
                and the final zero-initialized projection without a hidden Dense layer.
            return_gate (bool): Whether :meth:`call` returns a residual gate.
                Defaults to ``True``.
            no_adaptation (bool): Whether to omit condition-driven modulation.
                Defaults to ``False``.
            epsilon (float): Positive variance stabilizer for normalization.
                Defaults to ``1e-06``.
            **kwargs (Any): Standard Keras layer options.

        Returns:
            None: No value is returned.

        Raises:
            TypeError: If adaptive mode receives dim=None and cannot size its projection.
            ValueError: If Keras rejects feature widths or normalization settings.
        """

        super().__init__(**kwargs)
        self._save_init_args(locals())

        # Honor an explicit residual-gate width; otherwise match the normalized feature
        # width.
        self.gate_dim = self.gate_dim if self.gate_dim is not None else self.dim

        self.norm = layers.LayerNormalization(
            center=False, 
            scale=False, 
            epsilon=self.epsilon, 
            name="layer_norm", 
            dtype=self.dtype_policy
        )

        # Plain normalization infers its width from the input and has no MLP.
        if self.no_adaptation:
            self.mlp_output_dim = None
            self.mlp = None
        # Adaptive normalization constructs the condition-to-modulation network.
        else:
            # Reserve extra conditioning outputs only when a learned residual gate is
            # requested.
            self.mlp_output_dim = self.dim * 2 + (
                self.gate_dim if self.return_gate else 0
            )
            # Use activation-only conditioning when no hidden-width ratio is set.
            if self.mlp_ratio is None:
                mlp_first_layer = layers.Activation(
                    "swish",
                    name=f"{self.name}/mlp/first_layer",
                    dtype=self.dtype_policy,
                )
            # Otherwise add the configured hidden conditioning projection.
            else:
                mlp_first_layer = layers.Dense(
                    int(self.dim * self.mlp_ratio),
                    activation="swish",
                    # kernel_initializer="zeros",
                    name=f"{self.name}/mlp/first_layer",
                    dtype=self.dtype_policy,
                )
            self.mlp = models.Sequential([
                mlp_first_layer,
                layers.Dense(
                    self.mlp_output_dim,
                    kernel_initializer="zeros",
                    name=f"{self.name}/mlp/final_layer",
                    dtype=self.dtype_policy,
                )
            ], name="mlp")

    def call(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor | None], 
        training: bool | tf.Tensor | None = None
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor | float]:
        """Apply conditional normalization.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor | None]): Pair ``(x, cond)``
                following the class-level input
                contract. The feature dtype must be supported by Keras layer
                normalization; the condition must be compatible with the
                layer's compute dtype.
            training (bool | tf.Tensor | None): Optional Python boolean or
                symbolic Keras training flag forwarded to the nested layers.
                Defaults to ``None``. Keras resolves the surrounding call context; this flag is
                forwarded to child layers.

        Returns:
            tf.Tensor: with the shape and dtype of normalized ``x``, or a
            ``tuple[tf.Tensor, tf.Tensor | float]`` when ``return_gate`` is
            enabled. See the class-level output contract for gate shapes.
        """

        x, cond = inputs

        h = self.norm(x, training=training)

        # Bypass condition-driven modulation in plain-normalization mode.
        if self.no_adaptation:
            # Supply an identity residual gate when the caller requests one.
            if self.return_gate:
                return h, 1.
            return h

        params = self.mlp(cond, training=training)
        params = tf.expand_dims(params, 1)

        # Split out the learned residual gate when gated output is enabled.
        if self.return_gate:
            shift, scale, gate = tf.split(
                params,
                [self.dim, self.dim, self.gate_dim],
                axis=-1
            )
            return h * (1 + scale) + shift, gate

        shift, scale = tf.split(params, 2, axis=-1)
        return h * (1 + scale) + shift


def run_self_tests() -> dict[str, str]:
    """Run deterministic, CPU-small tests for :class:`AdaLNZero`.

    Args:
        None.

    Returns:
        dict[str, str]: A one-entry mapping whose value is ``"passed"`` after all adaptive,
        non-adaptive, gated, ungated, shape, dtype, gradient, and
        serialization checks succeed.
    """

    import numpy as np


    tf.random.set_seed(1701)
    x = tf.constant([
        [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], 
         [[0.0, 1.0, 0.0, 1.0], [2.0, 4.0, 6.0, 8.0]]
    ], dtype=tf.float32)
    cond = tf.ones((2, 3), dtype=tf.float32)

    for mlp_ratio in (None, 2.0):
        layer = AdaLNZero(dim=4, gate_dim=2, mlp_ratio=mlp_ratio)
        normalized, gate = layer((x, cond), training=True)
        expected = layer.norm(x, training=False)
        np.testing.assert_allclose(normalized.numpy(), expected.numpy(), atol=1e-6)
        np.testing.assert_array_equal(gate.numpy(), np.zeros((2, 1, 2)))
        assert normalized.shape == x.shape
        assert normalized.dtype == tf.float32
        assert layer.mlp.layers[-1].kernel.shape[-1] == 10

        with tf.GradientTape() as tape:
            result, result_gate = layer((x, cond), training=True)
            loss = tf.reduce_sum(result) + tf.reduce_sum(result_gate)
        gradients = tape.gradient(loss, layer.trainable_variables)
        assert gradients and all(gradient is not None for gradient in gradients)

    ungated = AdaLNZero(dim=4, return_gate=False)
    ungated_result = ungated((x, cond), training=False)
    assert isinstance(ungated_result, tf.Tensor)
    assert ungated_result.shape == x.shape
    assert ungated.mlp_output_dim == 8

    plain_gated = AdaLNZero(dim=4, no_adaptation=True, return_gate=True)
    plain_result, plain_gate = plain_gated((x, None), training=True)
    np.testing.assert_allclose(plain_result.numpy(), plain_gated.norm(x).numpy())
    assert plain_gate == 1.0
    assert plain_gated.mlp is None

    plain_ungated = AdaLNZero(
        dim=4, 
        no_adaptation=True, 
        return_gate=False, 
        epsilon=1e-4, 
        dtype="float64", 
    )
    x64 = tf.cast(x[:, 0, :], tf.float64)
    plain_ungated_result = plain_ungated((x64, None))
    assert plain_ungated_result.shape == (2, 4)
    assert plain_ungated.compute_dtype == "float64"
    assert plain_ungated_result.dtype == tf.float64
    assert plain_ungated.norm.epsilon == 1e-4

    inferred_plain = AdaLNZero(
        dim=None,
        no_adaptation=True,
        return_gate=False,
    )
    inferred_plain_result = inferred_plain((x, None))
    assert inferred_plain_result.shape == x.shape
    assert inferred_plain.mlp is None and inferred_plain.mlp_output_dim is None
    inferred_gated = AdaLNZero(dim=None, no_adaptation=True)
    inferred_gated_result, inferred_gate = inferred_gated((x, None))
    assert inferred_gated_result.shape == x.shape and inferred_gate == 1.0

    config = AdaLNZero(dim=4, gate_dim=3, mlp_ratio=1.5).get_config()
    restored = AdaLNZero.from_config(config)
    assert restored.dim == 4 and restored.gate_dim == 3
    assert restored.mlp_ratio == 1.5 and restored.return_gate

    incompatible = AdaLNZero(dim=4)
    try:
        incompatible((x, tf.ones((3, 2))))
    except (tf.errors.InvalidArgumentError, ValueError):
        pass
    # This invalid case should already have raised: Mismatched feature and condition batches
    # must fail.
    else:
        raise AssertionError("Mismatched feature and condition batches must fail.")

    return {"AdaLNZero": "passed"}


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
