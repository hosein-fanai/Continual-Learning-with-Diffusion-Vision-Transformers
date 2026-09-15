"""Keras serialization mixins that retain constructor arguments.

The concrete layer and model bases in this module let project components save
their constructor configuration without duplicating ``get_config`` and
``from_config`` implementations.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from collections.abc import Collection, Mapping

from common.keras_compat import display_name


def _copy_config_containers(value: object) -> object:
    """Copy nested metadata containers without cloning their runtime object leaves.

    Args:
        value (object): Constructor metadata containing dicts, lists, tuples,
            sets, and arbitrary leaves. Keras tracked containers are accepted.

    Returns:
        copied (object): Ordinary independent containers with the same contents.
            Noncontainer objects, including layers/tensors, retain identity.
            Removing container trackers avoids copying the owning model graph.

    Raises:
        RecursionError: If constructor metadata contains a container cycle.
    """

    # Each supported container is rebuilt without its Keras tracking metadata.
    if isinstance(value, dict):
        return {key: _copy_config_containers(item) for key, item in value.items()}
    # Lists retain order while discarding the wrapper's owning tracker.
    if isinstance(value, list):
        return [_copy_config_containers(item) for item in value]
    # Tuples may contain mutable nested metadata that also needs copying.
    if isinstance(value, tuple):
        return tuple(_copy_config_containers(item) for item in value)
    # Sets preserve membership without retaining a tracking wrapper.
    if isinstance(value, set):
        return {_copy_config_containers(item) for item in value}
    return value


class ArgumentSaver:
    """Record constructor values for Keras-compatible serialization.

    Subclasses call ``self._save_init_args(locals())`` from ``__init__`` after
    their superclass has been initialized.  The mixin stores each selected
    value both as an instance attribute and in ``_init_config``.  Mutable
    ``list``, ``set``, and ``dict`` values, including those nested in tuples,
    are independently copied for the attribute and saved config. Runtime
    leaves such as layers retain identity. Container mutations therefore cannot
    change another object's defaults or the serialized constructor input.

    Attributes:
        _init_config (dict[str, object]): Constructor argument names mapped to
            their constructor values.  Mutable values are defensive copies.
            The mapping is created by the first call to
            :meth:`_save_init_args` and extended by later calls, which supports
            subclasses that save both base and derived constructor arguments.
    """

    def __init__(
        self, 
        *args: object, 
        dynamic: bool = False, 
        **kwargs: object
    ) -> None:
        """Initialize the next base while retaining historical dynamic metadata.

        Args:
            *args (object): Positional arguments forwarded to the next base class.
            dynamic (bool): Historical flag retained only for serialization;
                it does not select eager execution in Keras 3. Defaults to False.
            **kwargs (object): Base-class options. Explicit names use ``__``
                in place of slashes; other options pass through unchanged.

        Returns:
            result (None): The instance and its saved dynamic flag are initialized.

        Raises:
            TypeError: If the next base rejects supplied arguments.
            ValueError: If Keras rejects a constructor setting.
        """

        # Keras 3 removed dynamic. Retain it as serialization metadata only.
        object.__setattr__(self, "_legacy_dynamic", bool(dynamic))

        # Normalize only explicit names; Keras supplies omitted names itself.
        if kwargs.get("name") is not None:
            kwargs["name"] = display_name(kwargs["name"])

        super().__init__(*args, **kwargs)

    @property
    def dynamic(self) -> bool:
        """Expose historical metadata without changing Keras execution mode.

        Returns:
            dynamic (bool): Constructor flag saved for configuration round trips.
        """

        return self._legacy_dynamic

    def add_weight(self, *args: object, **kwargs: object) -> object:
        """Normalize explicit project weight names before Keras creates them.

        Args:
            *args (object): Positional Keras add_weight arguments, normally shape.
            **kwargs (object): Keras weight options including dtype, initializer,
                and name. Explicit names replace slashes with ``__``.

        Returns:
            weight (object): Registered Keras variable with the requested dtype,
                or the layer variable dtype when no dtype is supplied.

        Raises:
            ValueError: If Keras rejects shape, dtype, initializer, or state creation.
        """

        # Automatically generated names stay under framework ownership.
        
        if kwargs.get("name") is not None:
            kwargs["name"] = display_name(kwargs["name"])

        return super().add_weight(*args, **kwargs)

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
        if name in saved_config and isinstance(value, (list, set, dict)) \
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
                Defaults to ``None``.

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
            if isinstance(value, (list, tuple, set, dict)):
                attribute_value = _copy_config_containers(value)
                config_value = _copy_config_containers(value)
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
        # Retain constructor metadata even when a subclass omits base fields.
        config.setdefault("name", self.name)
        config.setdefault("trainable", self.trainable)
        config.setdefault("dtype", self.dtype_policy.name)
        config.setdefault("dynamic", self.dynamic)
        # Copy mutable constructor metadata; preserve immutable/object-valued arguments.
        saved_config = {
            name: (
                _copy_config_containers(value)
                if isinstance(value, (list, tuple, set, dict))
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
                ``cls.__init__``. Nested metadata containers are copied before
                use; runtime object leaves retain identity.

        Returns:
            ArgumentSaver: A new ``cls`` instance initialized with ``config``.
        """

        config = _copy_config_containers(dict(config))

        return cls(**config)


class ArgumentSaverLayer(ArgumentSaver, layers.Layer):
    """Keras ``Layer`` base with automatic constructor-argument persistence.

    The class adds no computation.  Layer implementations inherit from this
    base and call :meth:`ArgumentSaver._save_init_args`; Keras then uses the
    inherited ``get_config``/``from_config`` pair for round-trip serialization.
    """

    def __call__(
        self, inputs: object, 
        *args: object, 
        **kwargs: object
    ) -> object:
        """Preserve optional inputs through Keras first-build inspection.

        Args:
            inputs (object): Tensor or nested tensors with optional None leaves.
                For an unbuilt layer, missing leaves trigger its existing build
                method using corresponding shapes and None placeholders.
            *args (object): Additional positional arguments for the layer call.
            **kwargs (object): Keras call options such as training and mask.

        Returns:
            outputs (object): Subclass call result with its declared structure
                and computation dtype. Already built layers use ordinary dispatch.

        Raises:
            AttributeError: If a present first-build input has no shape attribute.
            ValueError: If the layer build/call rejects supplied shapes or options.
        """

        # Keras 3.4's automatic shape inspection rejects optional None inputs.
        # Run the existing build contract first; its state lock stays intact.
        if not self.built and any(x is None for x in tf.nest.flatten(inputs)):
            self.build(tf.nest.map_structure(
                lambda x: None if x is None else x.shape, inputs
            ))

        return super().__call__(inputs, *args, **kwargs)


class ArgumentSaverModel(ArgumentSaver, models.Model):
    """Keras ``Model`` base with automatic constructor-argument persistence.

    The class adds no forward pass.  Project networks and wrappers derive from
    it so nested Keras objects and their constructor settings can be recreated
    from a saved config.
    """
