"""Trackable mapping container for depth-wise Keras layers."""

import tensorflow as tf
from tensorflow.keras import layers

from collections.abc import Iterator, Mapping
from typing import Any

from common.argument_saver import ArgumentSaverLayer


@tf.keras.utils.register_keras_serializable(package="continual_learning")
class LayerDict(ArgumentSaverLayer):
    """Store named child layers with mapping access and Keras variable tracking.

    ``LayerDict`` intentionally leaves execution to its owning model. It exists
    to provide dictionary-style stage access while exposing all child weights
    through the ordinary Keras ``trainable_variables`` property.
    """

    def __init__(
        self, 
        layers_dict: Mapping[str, layers.Layer] | None = None, 
        execution_order: list[str] | tuple[str, ...] | None = None, 
        **kwargs: Any
    ) -> None:
        """Track the supplied layers in a stable public key order.

        Args:
            layers_dict (Mapping[str, layers.Layer] | None): Optional mapping of
                public keys to child Keras layers.
            execution_order (list[str] | tuple[str, ...] | None): Optional exact
                key order; ``None`` preserves mapping insertion order.
            **kwargs (Any): Standard Keras layer options.

        Returns:
            None: Initialization mutates only the new container.
        """

        super().__init__(**kwargs)
        source = {} if layers_dict is None else dict(layers_dict)
        # Require every layer key to be a non-empty string.
        if any(not isinstance(key, str) or not key for key in source):
            raise ValueError("LayerDict keys must be non-empty strings.")

        for key, value in source.items():
            # Recreate layers supplied by the inherited from_config method.
            if isinstance(value, Mapping):
                source[key] = tf.keras.layers.deserialize(dict(value))
            # Require every remaining stored value to be a Keras layer.
            elif not isinstance(value, layers.Layer):
                raise TypeError("LayerDict values must be Keras layers or models.")

        order = list(source) if execution_order is None else list(execution_order)
        # Require the execution order to contain each key exactly once.
        if len(order) != len(set(order)) or set(order) != set(source):
            raise ValueError("execution_order must contain every layer key exactly once.")

        self._execution_order = []
        self._layers_dict = {}
        self._tracked_attribute_names = {}
        for key in order:
            self[key] = source[key]
        self._save_serialization_config()

    def _save_serialization_config(self) -> None:
        """Save the current child-layer mapping through ArgumentSaver.

        Returns:
            None: The inherited constructor configuration is updated in place.
        """

        self._save_init_args(
            {
                "layers_dict": {
                    key: tf.keras.layers.serialize(self._layers_dict[key])
                    for key in self._execution_order
                },
                "execution_order": list(self._execution_order),
            }, 
            rename={
                "layers_dict": "_config_layers_dict",
                "execution_order": "_config_execution_order",
            },
        )

    @property
    def execution_order(self) -> tuple[str, ...]:
        """Return component keys in their stable execution order.

        Args:
            None.

        Returns:
            tuple[str, ...]: Immutable public-key order.
        """

        return tuple(self._execution_order)

    def __setitem__(self, key: str, value: layers.Layer) -> None:
        """Add or replace one tracked child layer.

        Args:
            key (str): Non-empty public layer key.
            value (layers.Layer): Keras child layer or model to track.

        Returns:
            None: The container is updated in place.
        """

        # Require a non-empty string for each newly assigned key.
        if not isinstance(key, str) or not key:
            raise ValueError("LayerDict keys must be non-empty strings.")
        # Preserve Keras tracking by accepting layer values only.
        if not isinstance(value, layers.Layer):
            raise TypeError("LayerDict values must be Keras layers or models.")

        # Reuse the existing trackable attribute when replacing a key.
        if key in self._layers_dict:
            attribute_name = self._tracked_attribute_names[key]
        # Allocate a stable new trackable attribute for a new key.
        else:
            attribute_name = f"_tracked_layer_{len(self._execution_order)}"
            self._execution_order.append(key)
            self._tracked_attribute_names[key] = attribute_name

        setattr(self, attribute_name, value)
        self._layers_dict[key] = value

        # Keep constructor config current for layers added after initialization.
        if hasattr(self, "_init_config"):
            self._save_serialization_config()

    def update(self, values: Mapping[str, layers.Layer]) -> None:
        """Add or replace the supplied components in mapping order.

        Args:
            values (Mapping[str, layers.Layer]): Components to insert.

        Returns:
            None: The container is updated in place.
        """

        for key, value in values.items():
            self[key] = value

    def __getitem__(self, key: str) -> layers.Layer:
        """Return one child layer.

        Args:
            key (str): Public child-layer key.

        Returns:
            layers.Layer: Tracked layer stored under ``key``.
        """

        return self._layers_dict[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over public keys in execution order.

        Args:
            None.

        Returns:
            Iterator[str]: Iterator over stable public keys.
        """

        return iter(self._execution_order)

    def __len__(self) -> int:
        """Return the number of tracked child layers.

        Args:
            None.

        Returns:
            int: Number of public keys.
        """

        return len(self._execution_order)

    def __contains__(self, key: object) -> bool:
        """Report whether a public key is present.

        Args:
            key (object): Candidate mapping key.

        Returns:
            bool: ``True`` when ``key`` exists.
        """

        return key in self._layers_dict

    def keys(self) -> tuple[str, ...]:
        """Return public keys in execution order.

        Args:
            None.

        Returns:
            tuple[str, ...]: Stable key sequence.
        """

        return tuple(self._execution_order)

    def values(self) -> tuple[layers.Layer, ...]:
        """Return child layers in execution order.

        Args:
            None.

        Returns:
            tuple[layers.Layer, ...]: Stable child-layer sequence.
        """

        return tuple(self._layers_dict[key] for key in self._execution_order)

    def items(self) -> tuple[tuple[str, layers.Layer], ...]:
        """Return key-layer pairs in execution order.

        Args:
            None.

        Returns:
            tuple[tuple[str, layers.Layer], ...]: Stable mapping items.
        """

        return tuple((key, self._layers_dict[key]) for key in self._execution_order)

    def get(self, key: str, default: Any | None = None) -> layers.Layer | Any:
        """Return a child layer or a caller-supplied default.

        Args:
            key (str): Public child-layer key.
            default (Any | None): Value returned when ``key`` is absent.

        Returns:
            layers.Layer | Any: Stored child layer or ``default``.
        """

        return self._layers_dict.get(key, default)


def run_self_tests() -> dict[str, str]:
    """Check mapping behavior, tracking, validation, and serialization.

    Args:
        None.

    Returns:
        dict[str, str]: One success entry after all checks pass.
    """

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


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
