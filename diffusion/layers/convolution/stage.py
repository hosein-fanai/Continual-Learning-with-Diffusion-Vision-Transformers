"""Trackable mapping container for depth-wise Keras layers."""

import tensorflow as tf
from tensorflow.keras import layers

from collections.abc import Iterator, Mapping


class LayerDict(layers.Layer):
    """Store named child layers with mapping access and Keras variable tracking.

    ``LayerDict`` intentionally leaves execution to its owning model. It exists
    to provide dictionary-style stage access while exposing all child weights
    through the ordinary Keras ``trainable_variables`` property.
    """

    def __init__(
        self, 
        layers_dict: Mapping[str, layers.Layer] | None = None, 
        execution_order: list[str] | tuple[str, ...] | None = None, 
        **kwargs
    ):
        """Track the supplied layers in a stable public key order."""

        super().__init__(**kwargs)
        source = {} if layers_dict is None else dict(layers_dict)
        if any(not isinstance(key, str) or not key for key in source):
            raise ValueError("LayerDict keys must be non-empty strings.")
        if any(not isinstance(value, layers.Layer) for value in source.values()):
            raise TypeError("LayerDict values must be Keras layers or models.")

        order = list(source) if execution_order is None else list(execution_order)
        if len(order) != len(set(order)) or set(order) != set(source):
            raise ValueError("execution_order must contain every layer key exactly once.")

        self._execution_order = []
        self._layers_dict = {}
        self._tracked_attribute_names = {}
        for key in order:
            self[key] = source[key]

    @property
    def execution_order(self) -> tuple[str, ...]:
        """Return component keys in their stable execution order."""

        return tuple(self._execution_order)

    def __setitem__(self, key: str, value: layers.Layer) -> None:
        """Add or replace one tracked child layer."""

        if not isinstance(key, str) or not key:
            raise ValueError("LayerDict keys must be non-empty strings.")
        if not isinstance(value, layers.Layer):
            raise TypeError("LayerDict values must be Keras layers or models.")

        if key in self._layers_dict:
            attribute_name = self._tracked_attribute_names[key]
        else:
            attribute_name = f"_tracked_layer_{len(self._execution_order)}"
            self._execution_order.append(key)
            self._tracked_attribute_names[key] = attribute_name

        setattr(self, attribute_name, value)
        self._layers_dict[key] = value

    def update(self, values: Mapping[str, layers.Layer]) -> None:
        """Add or replace the supplied components in mapping order."""

        for key, value in values.items():
            self[key] = value

    def __getitem__(self, key: str) -> layers.Layer:
        return self._layers_dict[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._execution_order)

    def __len__(self) -> int:
        return len(self._execution_order)

    def __contains__(self, key: object) -> bool:
        return key in self._layers_dict

    def keys(self):
        return tuple(self._execution_order)

    def values(self):
        return tuple(self._layers_dict[key] for key in self._execution_order)

    def items(self):
        return tuple((key, self._layers_dict[key]) for key in self._execution_order)

    def get(self, key: str, default=None):
        return self._layers_dict.get(key, default)

    def get_config(self) -> dict:
        """Serialize the stable order and every contained Keras layer."""

        config = super().get_config()
        config.update({
            "layers_dict": {
                key: tf.keras.layers.serialize(self._layers_dict[key])
                for key in self._execution_order
            }, 
            "execution_order": list(self._execution_order)
        })

        return config

    @classmethod
    def from_config(cls, config):
        """Deserialize child layers before reconstructing the container."""

        config = dict(config)
        serialized_layers = config.pop("layers_dict")
        config["layers_dict"] = {
            key: tf.keras.layers.deserialize(value)
            for key, value in serialized_layers.items()
        }

        return cls(**config)


if __name__ != "__main__":
    tf.keras.utils.register_keras_serializable(
        package="continual_learning"
    )(LayerDict)


def run_self_tests() -> dict[str, str]:
    """Check mapping behavior, tracking, validation, and serialization."""

    first = layers.Dense(4, name="first")
    second = layers.Dense(2, name="second")
    stage = LayerDict(
        {"first": first, "second": second},
        execution_order=("second", "first"),
        name="stage_probe",
    )
    assert list(stage) == ["second", "first"]
    assert stage["first"] is first and "second" in stage
    assert stage.get("missing") is None
    stage["third"] = layers.Dense(1, name="third")
    assert list(stage) == ["second", "first", "third"]

    x = tf.ones((2, 3))
    first(x)
    second(first(x))
    assert len(stage.trainable_variables) == 4

    clone = LayerDict.from_config(stage.get_config())
    assert list(clone) == ["second", "first", "third"]
    assert isinstance(clone["first"], layers.Dense)

    try:
        LayerDict({"a": first}, execution_order=("missing",))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid execution orders must fail.")

    return {"LayerDict": "passed"}


if __name__ == "__main__":
    print(run_self_tests())
