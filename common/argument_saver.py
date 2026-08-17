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

            self._init_config[name] = deepcopy(
                value
            ) if isinstance(value, (list, set, dict)) else value

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


def run_self_tests() -> dict[str, str]:
    """Run deterministic serialization tests for every class in this module.

    The checks cover exclusions and renames, cumulative saves, defensive
    copying of mutable configuration values, superclass Keras configuration,
    and ``from_config`` reconstruction for both layer and model subclasses.

    Args:
        None.

    Returns:
        dict[str, str]: Exactly one ``"passed"`` entry for
        :class:`ArgumentSaver`, :class:`ArgumentSaverLayer`, and
        :class:`ArgumentSaverModel` when every assertion succeeds.
    """

    saver = ArgumentSaver()
    source_list = [1, {"nested": 2}]
    source_dict = {"enabled": True}
    source_set = {1, 2}
    saved = saver._save_init_args({
        "self": saver, 
        "items": source_list, 
        "options": source_dict, 
        "members": source_set, 
        "build": "deferred", 
        "kwargs": {"ignored": True}, 
        "temp_val": "ignored", 
        "__class__": ArgumentSaver, 
    })
    assert saved is saver._init_config
    assert saver.items is source_list
    assert saver.options is source_dict
    assert saver.members is source_set
    assert saver.build_ == "deferred" and not hasattr(saver, "build")
    assert set(saved) == {"items", "options", "members", "build"}
    source_list[1]["nested"] = 99
    source_dict["enabled"] = False
    source_set.add(3)
    assert saved["items"] == [1, {"nested": 2}]
    assert saved["options"] == {"enabled": True}
    assert saved["members"] == {1, 2}
    cumulative = saver._save_init_args(
        {"self": saver, "value": 7, "skip": 8}, 
        exclude=("self", "skip"), 
        rename={"value": "renamed_value"}, 
    )
    assert cumulative is saved
    assert saver.renamed_value == 7 and not hasattr(saver, "value")
    assert cumulative["value"] == 7 and "skip" not in cumulative
    try:
        saver.get_config()
    except AttributeError:
        pass
    else:
        raise AssertionError("Bare ArgumentSaver must require a configurable superclass.")


    def layer_probe_init(
        self: ArgumentSaverLayer, 
        value: int = 1, 
        payload: dict[str, object] | None = None, 
        **kwargs: object, 
    ) -> None:
        """Initialize the dynamic layer probe used by this self-test.

        Args:
            self (ArgumentSaverLayer): Probe instance being initialized.
            value (int): Scalar constructor value to preserve.
            payload (dict[str, object] | None): Mutable value to preserve.
            **kwargs (object): Standard Keras ``Layer`` constructor options.

        Returns:
            None.
        """

        layers.Layer.__init__(self, **kwargs)
        payload = {} if payload is None else payload
        self._save_init_args(locals())


    layer_probe_type = type(
        "ArgumentSaverLayerProbe", 
        (ArgumentSaverLayer,), 
        {"__init__": layer_probe_init}, 
    )
    layer = layer_probe_type(
        value=4, 
        payload={"items": [1, 2]}, 
        name="argument_saver_layer_probe", 
        trainable=False, 
        dtype="float64", 
    )
    layer_config = layer.get_config()
    assert layer_config["value"] == 4
    assert layer_config["payload"] == {"items": [1, 2]}
    assert layer_config["name"] == "argument_saver_layer_probe"
    assert layer_config["trainable"] is False
    assert layer_config["dtype"] == "float64"
    layer_clone = layer_probe_type.from_config(layer_config)
    assert isinstance(layer_clone, ArgumentSaverLayer)
    assert layer_clone.value == 4
    assert layer_clone.payload == {"items": [1, 2]}
    layer_config["payload"]["items"].append(3)
    assert layer_clone.payload == {"items": [1, 2]}


    def model_probe_init(
        self: ArgumentSaverModel, 
        width: int = 2, 
        metadata: dict[str, object] | None = None, 
        **kwargs: object, 
    ) -> None:
        """Initialize the dynamic model probe used by this self-test.

        Args:
            self (ArgumentSaverModel): Probe instance being initialized.
            width (int): Scalar constructor value to preserve.
            metadata (dict[str, object] | None): Mutable value to preserve.
            **kwargs (object): Standard Keras ``Model`` constructor options.

        Returns:
            None.
        """

        models.Model.__init__(self, **kwargs)
        metadata = {} if metadata is None else metadata
        self._save_init_args(locals())


    model_probe_type = type(
        "ArgumentSaverModelProbe", 
        (ArgumentSaverModel,), 
        {"__init__": model_probe_init}, 
    )
    model = model_probe_type(
        width=8, 
        metadata={"labels": {1, 2}}, 
        name="argument_saver_model_probe", 
        trainable=True, 
    )
    model_config = model.get_config()
    assert model_config["width"] == 8
    assert model_config["metadata"] == {"labels": {1, 2}}
    assert "name" not in model_config, (
        "TensorFlow 2.10's subclassed Model.get_config does not include the "
        "base name, unlike Layer.get_config."
    )
    model_clone = model_probe_type.from_config(model_config)
    assert isinstance(model_clone, ArgumentSaverModel)
    assert model_clone.width == 8
    assert model_clone.metadata == {"labels": {1, 2}}
    assert model_clone is not model

    return {
        "ArgumentSaver": "passed", 
        "ArgumentSaverLayer": "passed", 
        "ArgumentSaverModel": "passed", 
    }


if __name__ == "__main__":
    print(run_self_tests())
