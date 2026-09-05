"""Shared factories for condition-aware normalization and feed-forward layers.

BaseLayer stores shared normalization and MLP settings and creates policy-aware
AdaLNZero or dense projection sublayers for token processors. Optional overrides
resolve against instance defaults, and disabled factories return None to denote
an identity path.
"""

from tensorflow.keras import layers, models

from typing import Any

from common.argument_saver import ArgumentSaverLayer

from diffusion.layers.adaptive_layer_normalization_zero import AdaLNZero


class BaseLayer(ArgumentSaverLayer):
    """Base class for configurable diffusion token-processing layers.

    ``BaseLayer`` centralizes construction of :class:`AdaLNZero` and the small
    dense networks used throughout the transformer, embedding, and token
    manipulation layers. It is a factory-bearing base class and does not
    implement ``call`` itself.

    Args:
        use_layer_norm (bool): Whether :meth:`_create_layer_norm` creates an adaptive
            normalization layer. If false, the factory returns ``None``.
            Defaults to ``False``.
        ln_dim (int | None): Default normalized feature width. It is required when layer
            normalization is enabled unless ``ln_no_adaptation=True``.
            Defaults to ``None``.
        ln_mlp_ratio (float | None): Optional hidden-width ratio for the conditioning MLP in
            each adaptive normalization layer. ``None`` uses only Swish and a
            final projection.
            Defaults to ``None``.
        ln_no_adaptation (bool): Make created normalizers ordinary non-affine layer
            normalization layers and ignore their condition input.
            Defaults to ``False``.
        mlp_ratio (float | None): Optional hidden-width ratio for :meth:`_create_mlp`. ``None``
            produces a single output projection when ``mlp_output_dim`` is set.
            Defaults to ``None``.
        mlp_activation_func (str): Keras activation name or callable for the optional
            hidden dense layer, for example ``"swish"``, ``"gelu"``, or
            ``"relu"``.
            Defaults to ``'swish'``.
        mlp_output_dim (int | None): Default output width. ``None`` disables the MLP and
            makes the factory represent an identity operation.
            Defaults to ``None``.
        **kwargs (Any): Standard ``tf.keras.layers.Layer`` options such as ``name``,
            ``dtype``, and ``trainable``.

    Attributes:
        prev_output_dim (int | None): Input width recorded by the latest MLP factory call.
        output_dim (int | None): Effective output width recorded by that call.

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
        ln_mlp_ratio: float | None = None,
        ln_no_adaptation: bool = False, 
        mlp_ratio: float | None = None, 
        mlp_activation_func: str = "swish", 
        mlp_output_dim: int | None = None, 
        **kwargs: Any
    ) -> None:
        """Store shared layer configuration and validate normalization use.

        Args:
            use_layer_norm (bool): Whether normalization factories create
                :class:`AdaLNZero` layers.
                Defaults to ``False``.
            ln_dim (int | None): Normalized feature width. Defaults to ``None``, valid when normalization is
                disabled or plain normalization infers width; adaptive normalization requires an explicit
                width.
            ln_mlp_ratio (float | None): Conditioning hidden-width ratio. Defaults to ``None``, using Swish
                followed directly by the modulation projection.
            ln_no_adaptation (bool): Whether created normalizers ignore their
                condition input.
                Defaults to ``False``.
            mlp_ratio (float | None): Feed-forward hidden-width ratio. Defaults to ``None``, omitting the
                hidden layer while retaining any requested final projection.
            mlp_activation_func (str): Keras hidden-layer activation name.
                Defaults to ``'swish'``.
            mlp_output_dim (int | None): Feed-forward output width. Defaults to ``None``, so the factory
                returns None for an identity transformation.
            **kwargs (Any): Standard Keras layer options.

        Returns:
            None: No value is returned.
        """

        super().__init__(**kwargs)
        self._check_assertions(locals())
        self._save_init_args(locals())

    def _check_assertions(self, local_vars: dict[str, Any]) -> None:
        """Validate base constructor arguments.

        Args:
            local_vars (dict[str, Any]): Constructor-local mapping containing
                at least
                ``use_layer_norm``, ``ln_no_adaptation``, and ``ln_dim``.

        Returns:
            None: Valid settings return normally.

        Raises:
            ValueError: If adaptive normalization is requested without a feature width.
        """

        # Adaptive normalization requires an explicitly known feature width.
        if local_vars["use_layer_norm"] and \
        not local_vars["ln_no_adaptation"]:
            # Reject the missing width before constructing adaptive layers.
            if local_vars["ln_dim"] is None:
                raise ValueError(
                    "ln_dim cannot be None when use_layer_norm is true."
                )

    def _create_layer_norm(
        self, 
        dim: int | None = None, 
        gate_dim: int | None = None, 
        mlp_ratio: float | None = None, 
        return_gate: bool = True, 
        no_adaptation: bool | None = None, 
        use_layer_norm: bool | None = None, 
        name: str | None = None, 
    ) -> AdaLNZero | None:
        """Create a configured adaptive normalizer or return ``None``.

        Args:
            dim (int | None): Normalized channel width. ``None`` uses
                ``self.ln_dim``.
                Defaults to ``None``.
            gate_dim (int | None): Gate width. ``None`` uses ``self.ln_dim``; callers that
                override ``dim`` and need a matching gate should pass it too.
                Defaults to ``None``.
            mlp_ratio (float | None): Conditioning hidden-width ratio. ``None`` uses
                ``self.ln_mlp_ratio``.
                Defaults to ``None``.
            return_gate (bool): Whether the normalizer also returns a residual gate.
                Defaults to ``True``.
            no_adaptation (bool | None): Plain-normalization override. Defaults to ``None``, inheriting
                self.ln_no_adaptation; True ignores conditions and uses scalar-one gates.
            use_layer_norm (bool | None): Whether to construct a normalizer. Defaults to ``None``,
                inheriting self.use_layer_norm; False returns None.
            name (str | None): Optional Keras name; ``None`` derives one from this layer.
                Defaults to ``None``.

        Returns:
            AdaLNZero | None:. The layer consumes ``(features, condition)``;
            ``None`` tells the caller to leave features unchanged.
        """

        # Inherit the stored normalization width when no per-call width override is
        # supplied.
        dim = self.ln_dim if dim is None else dim
        # Inherit the stored gate width when no per-call override is supplied.
        gate_dim = self.ln_dim if gate_dim is None else gate_dim
        # Omit the conditioning hidden layer only when neither call nor instance supplies
        # its ratio.
        mlp_ratio = None if mlp_ratio is None and self.ln_mlp_ratio is None \
                    else mlp_ratio or self.ln_mlp_ratio
        # Inherit plain/adaptive normalization mode unless the caller explicitly overrides
        # it.
        no_adaptation = self.ln_no_adaptation if no_adaptation is None else no_adaptation
        # Inherit the normalization toggle unless the caller explicitly overrides it.
        use_layer_norm = self.use_layer_norm if use_layer_norm is None else use_layer_norm
        # Derive the child name from its owner when no explicit name is supplied.
        name = f"{self.name}/layer_norm" if name is None else name

        # Create a normalizer only when enabled; None denotes an identity path.
        layer_norm = AdaLNZero(
            dim=dim, 
            gate_dim=gate_dim, 
            mlp_ratio=mlp_ratio, 
            return_gate=return_gate, 
            no_adaptation=no_adaptation, 
            name=name,
            dtype=self.dtype_policy,
        ) if use_layer_norm else None

        return layer_norm

    def _create_mlp(
        self, 
        prev_output_dim: int | None,
        mlp_ratio: float | None = None, 
        mlp_activation_func: str | None = None, 
        mlp_output_dim: int | None = None, 
    ) -> models.Sequential | None:
        """Create an optional dense projection network.

        Args:
            prev_output_dim (int | None): Positive input width. ``None`` is
                permitted only when no output projection is configured.
            mlp_ratio (float | None): Hidden-width ratio. Defaults to ``None``, inheriting self.mlp_ratio;
                if both are None, only the final projection is created. A resolved ratio gives
                int(prev_output_dim * ratio) hidden units.
            mlp_activation_func (str | None): Hidden-layer activation. Defaults to ``None``, inheriting
                self.mlp_activation_func; unused when the resolved hidden ratio is None.
            mlp_output_dim (int | None): Final projection width. Defaults to ``None``, inheriting
                self.mlp_output_dim; the MLP is disabled only if the resolved instance/call width is None.

        Returns:
            tf.keras.Sequential | None:. A configured ratio produces
            ``Dense(hidden, activation) -> Dense(output)``; a ``None`` ratio
            produces one ``Dense(output)``; a ``None`` output width returns
            ``None``. The factory also records ``prev_output_dim`` and the
            effective ``output_dim`` on this object.
        """

        # Use the stored feed-forward ratio when the call leaves it unspecified.
        mlp_ratio = self.mlp_ratio if mlp_ratio is None else mlp_ratio
        # Use the stored hidden activation when the call leaves it unspecified.
        mlp_activation_func = self.mlp_activation_func if mlp_activation_func is None \
                            else mlp_activation_func
        # Use the stored output width when the call leaves it unspecified.
        mlp_output_dim = self.mlp_output_dim if mlp_output_dim is None else mlp_output_dim

        # Treat a missing input width as valid only for a disabled MLP.
        if prev_output_dim is None:
            # A requested projection cannot be built without its input width.
            if mlp_output_dim is not None:
                raise ValueError(
                    "prev_output_dim is required when mlp_output_dim is set."
                )

            self.prev_output_dim = None
            self.output_dim = None

            return None

        self.prev_output_dim = prev_output_dim
        # Build the requested output projection when an output width is set.
        if mlp_output_dim is not None:
            self.output_dim = mlp_output_dim
            mlp = models.Sequential(name=f"{self.name}/mlp")
            # Add a hidden dense layer when a hidden-width ratio is configured.
            if mlp_ratio is not None:
                mlp.add(layers.Dense(
                    int(prev_output_dim * mlp_ratio), 
                    activation=mlp_activation_func, 
                    name=f"{mlp.name}/first_layer",
                    dtype=self.dtype_policy,
                ))

            mlp.add(layers.Dense(
                mlp_output_dim, 
                dtype=self.dtype_policy, 
                name=f"{mlp.name}/final_layer"
            ))
        # Otherwise expose an identity transformation with unchanged width.
        else:
            self.output_dim = prev_output_dim
            mlp = None

        return mlp


def run_self_tests() -> dict[str, str]:
    """Test all factories and validation paths exposed by :class:`BaseLayer`.

    Args:
        None.

    Returns:
        dict[str, str]: ``{"BaseLayer": "passed"}`` after factory, override, config, shape,
        and expected abstract-call checks pass.
    """

    import tensorflow as tf


    try:
        BaseLayer(use_layer_norm=True)
    except ValueError:
        pass
    # This invalid case should already have raised: Adaptive normalization requires ln_dim.
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
    inferred_plain = BaseLayer(
        use_layer_norm=True,
        ln_no_adaptation=True,
    )._create_layer_norm(return_gate=False)
    assert inferred_plain((tf.ones((1, 2, 5)), None)).shape == (1, 2, 5)

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
    assert dtype_output.dtype == tf.float64

    return {"BaseLayer": "passed"}


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
