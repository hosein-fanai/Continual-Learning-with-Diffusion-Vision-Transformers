"""Shared factories for condition-aware normalization and feed-forward layers."""

from tensorflow.keras import layers, models

from common.argument_saver import ArgumentSaverLayer

from diffusion.layers.adaptive_layer_normalization_zero import AdaLNZero


class BaseLayer(ArgumentSaverLayer):
    """Base class for configurable diffusion token-processing layers.

    ``BaseLayer`` centralizes construction of :class:`AdaLNZero` and the small
    dense networks used throughout the transformer, embedding, and token
    manipulation layers. It is a factory-bearing base class and does not
    implement ``call`` itself.

    Args:
        use_layer_norm: Whether :meth:`_create_layer_norm` creates an adaptive
            normalization layer. If false, the factory returns ``None``.
        ln_dim: Default normalized feature width. It is required when layer
            normalization is enabled unless ``ln_no_adaptation=True``.
        ln_mlp_ratio: Optional hidden-width ratio for the conditioning MLP in
            each adaptive normalization layer. ``None`` uses only Swish and a
            final projection.
        ln_no_adaptation: Make created normalizers ordinary non-affine layer
            normalization layers and ignore their condition input.
        mlp_ratio: Optional hidden-width ratio for :meth:`_create_mlp`. ``None``
            produces a single output projection when ``mlp_output_dim`` is set.
        mlp_activation_func: Keras activation name or callable for the optional
            hidden dense layer, for example ``"swish"``, ``"gelu"``, or
            ``"relu"``.
        mlp_output_dim: Default output width. ``None`` disables the MLP and
            makes the factory represent an identity operation.
        **kwargs: Standard ``tf.keras.layers.Layer`` options such as ``name``,
            ``dtype``, and ``trainable``.

    Attributes:
        prev_output_dim: Input width recorded by the latest MLP factory call.
        output_dim: Effective output width recorded by that call.

    Inputs:
        No direct tensor input; subclasses call the protected layer factories.

    Outputs:
        No direct tensor output. Factories return ``AdaLNZero``, Keras
        ``Sequential``, or ``None`` as documented on each method.
    """

    def __init__(
        self, 
        use_layer_norm: bool = False, 
        ln_dim: int | None = None, 
        ln_mlp_ratio: int | None = None, 
        ln_no_adaptation: bool = False, 
        mlp_ratio: float | None = None, 
        mlp_activation_func: str = "swish", 
        mlp_output_dim: int | None = None, 
        **kwargs
    ):
        """Store shared layer configuration and validate normalization use.

        Arguments and accepted types are documented on the class.

        Returns:
            ``None``.
        """

        super().__init__(**kwargs)
        self._check_assertions(locals())
        self._save_init_args(locals())

    def _check_assertions(self, local_vars):
        """Validate base constructor arguments.

        Args:
            local_vars: Constructor-local mapping containing at least
                ``use_layer_norm``, ``ln_no_adaptation``, and ``ln_dim``.

        Returns:
            ``None``. An ``AssertionError`` is raised when adaptive
            normalization is requested without a feature width.
        """

        if local_vars["use_layer_norm"] and \
        not local_vars["ln_no_adaptation"]:
            assert local_vars["ln_dim"] is not None, \
                "ln_dim cannot be None when use_layer_norm is true."

    def _create_layer_norm(
        self, 
        dim: int | None = None, 
        gate_dim: int | None = None, 
        mlp_ratio: float | None = None, 
        return_gate: bool = True, 
        no_adaptation: bool | None = None, 
        use_layer_norm: bool | None = None, 
        name: str | None = None, 
    ):
        """Create a configured adaptive normalizer or return ``None``.

        Args:
            dim: Normalized channel width. ``None`` uses ``self.ln_dim``.
            gate_dim: Gate width. ``None`` uses ``self.ln_dim``; callers that
                override ``dim`` and need a matching gate should pass it too.
            mlp_ratio: Conditioning hidden-width ratio. ``None`` uses
                ``self.ln_mlp_ratio``.
            return_gate: Whether the normalizer also returns a residual gate.
            no_adaptation: Override the instance default. ``True`` ignores the
                condition and makes the gate the scalar ``1.0``.
            use_layer_norm: Override whether a layer is created at all.
            name: Optional Keras name; ``None`` derives one from this layer.

        Returns:
            ``AdaLNZero | None``. The layer consumes ``(features, condition)``;
            ``None`` tells the caller to leave features unchanged.
        """

        dim = self.ln_dim if dim is None else dim
        gate_dim = self.ln_dim if gate_dim is None else gate_dim
        mlp_ratio = None if mlp_ratio is None and self.ln_mlp_ratio is None \
                    else mlp_ratio or self.ln_mlp_ratio
        no_adaptation = self.ln_no_adaptation if no_adaptation is None else no_adaptation
        use_layer_norm = self.use_layer_norm if use_layer_norm is None else use_layer_norm
        name = f"{self.name}/layer_norm" if name is None else name

        layer_norm = AdaLNZero(
            dim=dim, 
            gate_dim=gate_dim, 
            mlp_ratio=mlp_ratio, 
            return_gate=return_gate, 
            no_adaptation=no_adaptation, 
            name=name
        ) if use_layer_norm else None

        return layer_norm

    def _create_mlp(
        self, 
        prev_output_dim: int, 
        mlp_ratio: float | None = None, 
        mlp_activation_func: str | None = None, 
        mlp_output_dim: int | None = None, 
    ):
        """Create an optional dense projection network.

        Args:
            prev_output_dim: Positive integer size of the input's last axis.
            mlp_ratio: Hidden-width ratio overriding ``self.mlp_ratio``. When
                non-``None``, the first dense layer has
                ``int(prev_output_dim * mlp_ratio)`` units.
            mlp_activation_func: Keras activation for the hidden dense layer.
            mlp_output_dim: Final width overriding ``self.mlp_output_dim``.
                ``None`` disables the MLP entirely.

        Returns:
            ``tf.keras.Sequential | None``. A configured ratio produces
            ``Dense(hidden, activation) -> Dense(output)``; a ``None`` ratio
            produces one ``Dense(output)``; a ``None`` output width returns
            ``None``. The factory also records ``prev_output_dim`` and the
            effective ``output_dim`` on this object.
        """

        mlp_ratio = self.mlp_ratio if mlp_ratio is None else mlp_ratio
        mlp_activation_func = self.mlp_activation_func if mlp_activation_func is None \
                            else mlp_activation_func
        mlp_output_dim = self.mlp_output_dim if mlp_output_dim is None else mlp_output_dim

        self.prev_output_dim = int(prev_output_dim)
        if mlp_output_dim is not None:
            self.output_dim = mlp_output_dim
            mlp = models.Sequential(name=f"{self.name}/mlp")
            if mlp_ratio is not None:
                mlp.add(layers.Dense(
                    int(prev_output_dim * mlp_ratio), 
                    activation=mlp_activation_func, 
                    name=f"{mlp.name}/first_layer"
                ))

            mlp.add(layers.Dense(
                mlp_output_dim, 
                name=f"{mlp.name}/final_layer"
            ))
        else:
            self.output_dim = prev_output_dim
            mlp = None

        return mlp


def run_self_tests() -> dict[str, str]:
    """Test all factories and validation paths exposed by :class:`BaseLayer`.

    Args:
        None.

    Returns:
        ``{"BaseLayer": "passed"}`` after factory, override, config, shape,
        and expected abstract-call checks pass.
    """

    import tensorflow as tf


    try:
        BaseLayer(use_layer_norm=True)
    except AssertionError:
        pass
    else:
        raise AssertionError("Adaptive normalization requires ln_dim.")

    disabled = BaseLayer(use_layer_norm=False, ln_dim=4)
    assert disabled._create_layer_norm() is None
    assert disabled._create_layer_norm(use_layer_norm=False, dim=2) is None

    adaptive = BaseLayer(
        use_layer_norm=True, 
        ln_dim=4, 
        ln_mlp_ratio=2, 
        mlp_ratio=2, 
        mlp_activation_func="relu", 
        mlp_output_dim=3, 
        name="base_factory_test", 
    )
    normalizer = adaptive._create_layer_norm(gate_dim=2, return_gate=True)
    normalized, gate = normalizer((tf.ones((2, 3, 4)), tf.ones((2, 5))))
    assert normalized.shape == (2, 3, 4) and gate.shape == (2, 1, 2)
    overridden_plain = adaptive._create_layer_norm(
        dim=3, 
        gate_dim=1, 
        mlp_ratio=1, 
        return_gate=False, 
        no_adaptation=True, 
        name="plain_override", 
    )
    assert overridden_plain((tf.ones((1, 2, 3)), None)).shape == (1, 2, 3)

    two_layer_mlp = adaptive._create_mlp(prev_output_dim=4)
    assert len(two_layer_mlp.layers) == 2
    assert two_layer_mlp(tf.ones((2, 4))).shape == (2, 3)
    assert adaptive.prev_output_dim == 4 and adaptive.output_dim == 3

    single_layer_mlp = adaptive._create_mlp(
        prev_output_dim=3, 
        mlp_ratio=None, 
        mlp_output_dim=2, 
    )
    # A ``None`` override inherits the instance ratio by design.
    assert len(single_layer_mlp.layers) == 2
    one_projection = BaseLayer(mlp_output_dim=2)._create_mlp(3)
    assert len(one_projection.layers) == 1
    assert one_projection(tf.ones((1, 3))).shape == (1, 2)

    no_mlp = BaseLayer()._create_mlp(prev_output_dim=5)
    assert no_mlp is None
    no_projection_layer = BaseLayer()
    assert no_projection_layer._create_mlp(5) is None
    assert no_projection_layer.prev_output_dim == 5
    assert no_projection_layer.output_dim == 5

    restored = BaseLayer.from_config(adaptive.get_config())
    assert restored.ln_dim == 4 and restored.mlp_output_dim == 3
    direct_input = tf.ones((1, 4))
    # TensorFlow 2.10's base Keras Layer call is an identity when a subclass
    # does not override it; subclasses in this project add the real behavior.
    assert restored(direct_input) is direct_input

    dtype_layer = BaseLayer(mlp_output_dim=2, dtype="float64")
    dtype_projection = dtype_layer._create_mlp(4)
    dtype_output = dtype_projection(tf.ones((1, 4), dtype=tf.float64))
    assert dtype_layer.compute_dtype == "float64"
    # Factory-created nested Dense layers retain the global float32 policy.
    assert dtype_output.dtype == tf.float32

    return {"BaseLayer": "passed"}


if __name__ == "__main__":
    print(run_self_tests())
