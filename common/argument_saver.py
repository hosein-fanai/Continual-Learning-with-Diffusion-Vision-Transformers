"""Keras serialization mixins that retain constructor arguments.

The concrete layer and model bases in this module let project components save
their constructor configuration without duplicating ``get_config`` and
``from_config`` implementations.
"""

from tensorflow.keras import layers, models

from copy import deepcopy


class ArgumentSaver:
    """Record constructor values for Keras-compatible serialization.

    Subclasses call ``self._save_init_args(locals())`` from ``__init__`` after
    their superclass has been initialized.  The mixin stores each selected
    value both as an instance attribute and in ``_init_config``.  Mutable
    ``list``, ``set``, and ``dict`` values are copied for the saved config so
    later in-place mutations do not change the serialized constructor input.

    Attributes:
        _init_config (dict[str, object]): Constructor argument names mapped to
            their original values.  It is created by the first call to
            :meth:`_save_init_args` and extended by later calls, which supports
            subclasses that save both base and derived constructor arguments.
    """

    def _save_init_args(
        self, 
        local_vars, 
        exclude=("self", "kwargs", "__class__", "temp_val"), 
        rename={"build": "build_"}, 
    ):
        """Save selected local constructor variables as state and config.

        Args:
            local_vars (Mapping[str, object]): Usually ``locals()`` from a
                constructor.  Each non-excluded entry becomes an attribute.
            exclude (Collection[str]): Names not to save.  The default skips
                ``self``, catch-all ``kwargs``, ``__class__``, and the temporary
                name ``temp_val``.  Supply a different collection to retain a
                normally excluded name.
            rename (Mapping[str, str]): Attribute-only renames.  For example,
                the default ``{"build": "build_"}`` creates ``self.build_``
                while retaining the constructor key ``"build"`` in the config.

        Returns:
            dict[str, object]: The cumulative ``_init_config`` dictionary.

        Example:
            ``self._save_init_args(locals(), exclude=("self",),
            rename={"enabled": "is_enabled"})`` saves the constructor key
            ``enabled`` and exposes its value as ``self.is_enabled``.
        """
        if not hasattr(self, "_init_config"):
            self._init_config = {}

        for name, value in local_vars.items():
            if name in exclude:
                continue

            setattr(
                self, 
                rename.get(name, name), 
                value
            )

            self._init_config[name] = deepcopy(value) if isinstance(value, (list, set, dict)) else value

        return self._init_config

    def get_config(self):
        """Return the superclass config plus saved constructor arguments.

        Returns:
            dict[str, object]: A Keras serialization mapping.  Saved values
            override same-named entries returned by ``super().get_config()``.

        Raises:
            AttributeError: If the subclass never called
                :meth:`_save_init_args` and therefore has no ``_init_config``.
        """
        config = super().get_config()
        config.update(self._init_config)

        return config

    @classmethod
    def from_config(cls, config):
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
