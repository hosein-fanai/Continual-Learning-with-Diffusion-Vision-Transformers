"""Keras serialization mixins that retain constructor arguments.

The concrete layer and model bases in this module let project components save
their constructor configuration without duplicating ``get_config`` and
``from_config`` implementations.
"""

from __future__ import annotations

from tensorflow.keras import layers, models

from copy import deepcopy

from collections.abc import Collection, Mapping


class ArgumentSaver:
    """Record constructor values for Keras-compatible serialization.

    Subclasses call ``self._save_init_args(locals())`` from ``__init__`` after
    their superclass has been initialized.  The mixin stores each selected
    value both as an instance attribute and in ``_init_config``.  Mutable
    ``list``, ``set``, and ``dict`` values are independently copied for the
    attribute and saved config.  Caller or instance mutations therefore cannot
    change another object's defaults or the serialized constructor input.

    Attributes:
        _init_config (dict[str, object]): Constructor argument names mapped to
            their constructor values.  Mutable values are defensive copies.
            The mapping is created by the first call to
            :meth:`_save_init_args` and extended by later calls, which supports
            subclasses that save both base and derived constructor arguments.
    """

    def __setattr__(self, name: str, value: object) -> None:
        """Keep mutable constructor metadata outside the checkpoint graph.

        Args:
            name (str): Instance attribute name being assigned.
            value (object): New attribute value.

        Returns:
            None: The attribute is assigned through the next MRO class.
        """

        saved_config = self.__dict__.get("_init_config", {})
        # Reassign saved mutable metadata without creating TF dependencies.
        if name in saved_config \
        and isinstance(value, (list, set, dict)) \
        and hasattr(self, "_no_dependency"):
            value = self._no_dependency(value)
        super().__setattr__(name, value)

    def _save_init_args(
        self: ArgumentSaver, 
        local_vars: Mapping[str, object], 
        exclude: Collection[str] = (
            "self", "kwargs", 
            "__class__", "temp_val"
        ), 
        rename: Mapping[str, str] | None = None
    ) -> dict[str, object]:
        """Save selected local constructor variables as state and config.

        Args:
            local_vars (Mapping[str, object]): Usually ``locals()`` from a
                constructor.  Each non-excluded entry becomes an attribute.
            exclude (Collection[str]): Names not to save.  The default skips
                ``self``, catch-all ``kwargs``, ``__class__``, and the temporary
                name ``temp_val``.  Supply a different collection to retain a
                normally excluded name.
            rename (Mapping[str, str] | None): Attribute-only renames. ``None``
                uses ``{"build": "build_"}``, which creates ``self.build_``
                while retaining the constructor key ``"build"`` in config.

        Returns:
            dict[str, object]: The cumulative ``_init_config`` dictionary.

        Example:
            ``self._save_init_args(locals(), exclude=("self",),
            rename={"enabled": "is_enabled"})`` saves the constructor key
            ``enabled`` and exposes its value as ``self.is_enabled``.
        """

        rename = {"build": "build_"} if rename is None else rename
        # Initialize cumulative configuration storage on the first save.
        if not hasattr(self, "_init_config"):
            # Keep serialization metadata out of TensorFlow's object graph.
            if hasattr(self, "_no_dependency"):
                self._init_config = self._no_dependency({})
            # Plain mixin instances have no Trackable dependency API.
            else:
                self._init_config = {}

        for name, value in local_vars.items():
            # Omit constructor locals explicitly excluded from persistence.
            if name in exclude:
                continue

            # Isolate mutable state from both callers and serialized config.
            if isinstance(value, (list, set, dict)):
                attribute_value = deepcopy(value)
                config_value = deepcopy(value)
            # Preserve immutable or object-valued arguments by identity.
            else:
                attribute_value = value
                config_value = value

            # Mutable constructor metadata must not become checkpoint state.
            if isinstance(value, (list, set, dict)) \
            and hasattr(self, "_no_dependency"):
                attribute_value = self._no_dependency(attribute_value)

            setattr(
                self, 
                rename.get(name, name), 
                attribute_value
            )

            self._init_config[name] = config_value

        return self._init_config

    def get_config(self: ArgumentSaver) -> dict[str, object]:
        """Return the superclass config plus saved constructor arguments.

        Returns:
            dict[str, object]: A Keras serialization mapping.  Saved values
            override same-named entries returned by ``super().get_config()``.

        Raises:
            AttributeError: If the subclass never called
                :meth:`_save_init_args` and therefore has no ``_init_config``.
        """

        config = super().get_config()
        # TensorFlow 2.10 omits these fields for subclassed Model instances.
        config.setdefault("name", self.name)
        config.setdefault("trainable", self.trainable)
        config.setdefault("dtype", self.dtype_policy.name)
        config.setdefault("dynamic", self.dynamic)
        saved_config = {
            name: (
                deepcopy(value)
                if isinstance(value, (list, set, dict))
                else value
            )
            for name, value in self._init_config.items()
        }
        config.update(saved_config)

        return config

    @classmethod
    def from_config(
        cls: type[ArgumentSaver], 
        config: Mapping[str, object]
    ) -> ArgumentSaver:
        """Reconstruct an instance from a Keras configuration mapping.

        Args:
            config (Mapping[str, object]): Keyword arguments accepted by
                ``cls.__init__``.  The mapping is deep-copied before use.

        Returns:
            ArgumentSaver: A new ``cls`` instance initialized with ``config``.
        """

        config = deepcopy(config)

        return cls(**config)


class ArgumentSaverLayer(ArgumentSaver, layers.Layer):
    """Keras ``Layer`` base with automatic constructor-argument persistence.

    The class adds no computation.  Layer implementations inherit from this
    base and call :meth:`ArgumentSaver._save_init_args`; Keras then uses the
    inherited ``get_config``/``from_config`` pair for round-trip serialization.
    """


class ArgumentSaverModel(ArgumentSaver, models.Model):
    """Keras ``Model`` base with automatic constructor-argument persistence.

    The class adds no forward pass.  Project networks and wrappers derive from
    it so nested Keras objects and their constructor settings can be recreated
    from a saved config.
    """


def run_self_tests() -> dict[str, str]:
    """Smoke-test saved arguments and Keras reconstruction.

    Returns:
        dict[str, str]: Passing markers for all three public classes.
    """

    saver = ArgumentSaver()
    source = {"items": [1, 2]}
    saved = saver._save_init_args(
        {"self": saver, "payload": source, "build": False}
    )
    source["items"].append(3)
    assert saved == {"payload": {"items": [1, 2]}, "build": False}
    assert saver.payload == {"items": [1, 2]} and saver.build_ is False

    class LayerProbe(ArgumentSaverLayer):
        """Minimal serializable layer used by the smoke test."""

        def __init__(self, value: int = 1, **kwargs: object) -> None:
            """Save one layer value.

            Args:
                value (int): Value retained in the Keras configuration.
                **kwargs (object): Standard Keras layer options.

            Returns:
                None.
            """

            super().__init__(**kwargs)
            self._save_init_args(locals())

    class ModelProbe(ArgumentSaverModel):
        """Minimal serializable model used by the smoke test."""

        def __init__(self, width: int = 2, **kwargs: object) -> None:
            """Save one model width.

            Args:
                width (int): Width retained in the Keras configuration.
                **kwargs (object): Standard Keras model options.

            Returns:
                None.
            """

            super().__init__(**kwargs)
            self._save_init_args(locals())

    layer = LayerProbe(value=4, name="probe_layer", trainable=False)
    layer_clone = LayerProbe.from_config(layer.get_config())
    assert layer_clone.value == 4 and layer_clone.trainable is False

    model = ModelProbe(width=8, name="probe_model")
    model_clone = ModelProbe.from_config(model.get_config())
    assert model_clone.width == 8 and model_clone.name == "probe_model"

    return {
        "ArgumentSaver": "passed", 
        "ArgumentSaverLayer": "passed", 
        "ArgumentSaverModel": "passed"
    }
