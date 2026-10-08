"""Trackable mapping container for depth-wise Keras layers.

LayerDict provides stable mapping access to child layers while exposing their
variables to Keras checkpoints. Insertions and replacements refresh serialized
constructor metadata. Owners can execute a selected component through the
container so Keras records the stage's real build state and output shape.
"""

import tensorflow as tf
from tensorflow.keras import layers

from collections.abc import Iterator, Mapping
from typing import Any

from common.argument_saver import ArgumentSaverLayer
from common.keras_registry import register_canonical_keras_serializable


@register_canonical_keras_serializable(package="continual_learning")
class LayerDict(ArgumentSaverLayer):
    """Store named child layers with mapping access and Keras variable tracking.

    The owning model selects which component to execute. Calling the container
    delegates that operation to the selected child and records an ordinary Keras
    call node without changing the mapping or checkpoint ownership hierarchy.

    Attributes:
        _execution_order (list[str]): Mutable insertion order underlying the public tuple
            property.
        _layers_dict (dict[str, tf.keras.layers.Layer]): Live child layers keyed by public
            names.
        _tracked_attribute_names (dict[str, str]): Stable Keras attribute name for each
            public key.
    """

    def __init__(
        self, 
        layers_dict: Mapping[str, layers.Layer] | None = None, 
        execution_order: list[str] | tuple[str, ...] | None = None, 
        **kwargs: Any
    ) -> None:
        """Track the supplied layers in a stable public key order.

        Args:
            layers_dict (Mapping[str, tf.keras.layers.Layer | Mapping] | None): Child layers keyed by public
                names; serialized layer mappings are also deserialized. Defaults to ``None``, creating an
                empty container. The source mapping is copied, while live layer objects remain shared.
            execution_order (list[str] | tuple[str, ...] | None): Optional exact
                key order; ``None`` preserves mapping insertion order.
                Defaults to ``None``.
            **kwargs (Any): Standard Keras layer options.

        Returns:
            None: Initialization mutates only the new container.

        Raises:
            ValueError: execution_order repeats a key or does not contain exactly
                the source keys, or Keras cannot deserialize a supplied layer config.
            TypeError: layers_dict is not convertible to a mapping or a child layer
                cannot be serialized into the saved constructor configuration.
        """

        super().__init__(**kwargs)
        # Start an empty stage without supplied layers; otherwise copy the layer mapping.
        source = {} if layers_dict is None else dict(layers_dict)
        for key, value in source.items():
            # Recreate layers supplied by the inherited from_config method.
            if isinstance(value, Mapping):
                source[key] = tf.keras.layers.deserialize(dict(value))

        # Use mapping insertion order unless an explicit execution order is supplied.
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

    def call(
        self, 
        inputs: Any, 
        layer_key: str, 
        child_kwargs: Mapping[str, Any] | None = None, 
        training: bool | None = None
    ) -> Any:
        """Execute one tracked component through its owning stage.

        Args:
            inputs (Any): Tensor or nested tensor inputs accepted by the child.
            layer_key (str): Public key of the component to execute.
            child_kwargs (Mapping[str, Any] | None): Additional child call
                arguments. Defaults to ``None``.
            training (bool | None): Keras execution mode forwarded to the child.
                Defaults to ``None``.

        Returns:
            Any: The selected child's unchanged tensor output structure.

        Raises:
            KeyError: The selected component is absent.
            ValueError: The child rejects its input structure or shape.
        """

        child_kwargs = {} if child_kwargs is None else child_kwargs
        return self[layer_key](inputs, training=training, **child_kwargs)

    def _save_serialization_config(self) -> None:
        """Serialize the current ordered children into the saved constructor configuration.

        Returns:
            None: _init_config and its private serialized mapping/order attributes
                are replaced. Live child layers are retained; no tensors are copied
                and execution order is unchanged.

        Raises:
            TypeError: A child layer exposes a configuration Keras cannot serialize.
        """

        self._save_init_args(
            {
                "layers_dict": {
                    key: tf.keras.layers.serialize(self._layers_dict[key])
                    for key in self._execution_order
                }, 
                "execution_order": list(self._execution_order)
            }, 
            rename={
                "layers_dict": "_config_layers_dict", 
                "execution_order": "_config_execution_order"
            }
        )

    @property
    def execution_order(self) -> tuple[str, ...]:
        """Return component keys in their stable execution order.

        Args:
            None.

        Returns:
            tuple[str, ...]: Immutable public-key order.

        Raises:
            None: Returning a tuple copy does not change the tracked order.
        """

        return tuple(self._execution_order)

    def __setitem__(self, key: str, value: layers.Layer) -> None:
        """Add or replace one tracked child layer.

        Args:
            key (str): Public layer key, stored as supplied; no local nonempty
                string check is performed.
            value (layers.Layer): Keras child layer or model to track.

        Returns:
            None: An existing key keeps its execution position; a new key appends
                at the end and receives a stable tracked attribute. Replacing a layer
                preserves its key/attribute name and refreshes saved configuration.

        Raises:
            TypeError: key is unhashable or the inserted child's configuration cannot
                be serialized by Keras.
            ValueError: Keras rejects attaching new tracked state after this container
                has been built.
        """

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
            None: Each entry is assigned through __setitem__ in mapping order.
                Earlier successful entries remain installed if a later assignment
                fails; this update is not transactional.

        Raises:
            TypeError: A supplied key is unhashable or a child cannot be serialized.
            ValueError: Keras rejects adding tracked layers to a built container.
        """

        for key, value in values.items():
            self[key] = value

    def __getitem__(self, key: str) -> layers.Layer:
        """Return one child layer.

        Args:
            key (str): Public child-layer key.

        Returns:
            layers.Layer: Tracked layer stored under ``key``.

        Raises:
            KeyError: If no child layer is stored under key.
        """

        return self._layers_dict[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over public keys in execution order.

        Args:
            None.

        Returns:
            Iterator[str]: Iterator over stable public keys.

        Raises:
            None: Constructing an iterator does not mutate the key list; subsequent
                container mutation follows ordinary Python list-iterator behavior.
        """

        return iter(self._execution_order)

    def __len__(self) -> int:
        """Return the number of tracked child layers.

        Args:
            None.

        Returns:
            int: Number of public keys.

        Raises:
            None: The number of stored public keys is read without mutation.
        """

        return len(self._execution_order)

    def __contains__(self, key: object) -> bool:
        """Report whether a public key is present.

        Args:
            key (object): Candidate mapping key.

        Returns:
            bool: ``True`` when ``key`` exists.

        Raises:
            TypeError: key is unhashable, following ordinary dictionary membership.
        """

        return key in self._layers_dict

    def keys(self) -> tuple[str, ...]:
        """Return public keys in execution order.

        Args:
            None.

        Returns:
            tuple[str, ...]: Stable key sequence.

        Raises:
            None: Returning a tuple copy does not expose the mutable internal list.
        """

        return tuple(self._execution_order)

    def values(self) -> tuple[layers.Layer, ...]:
        """Return child layers in execution order.

        Args:
            None.

        Returns:
            tuple[layers.Layer, ...]: Stable child-layer sequence.

        Raises:
            None: All keys in the maintained execution order already have stored layers;
                the returned tuple shares those live layer objects without copying weights.
        """

        return tuple(self._layers_dict[key] for key in self._execution_order)

    def items(self) -> tuple[tuple[str, layers.Layer], ...]:
        """Return key-layer pairs in execution order.

        Args:
            None.

        Returns:
            tuple[tuple[str, layers.Layer], ...]: Stable mapping items.

        Raises:
            None: The maintained key/layer mapping is read without mutation;
                returned pairs share the live child layers.
        """

        return tuple((key, self._layers_dict[key]) for key in self._execution_order)

    def get(self, key: str, default: Any | None = None) -> layers.Layer | Any:
        """Return a child layer or a caller-supplied default.

        Args:
            key (str): Public child-layer key.
            default (Any | None): Value returned when key is absent. Defaults to ``None``; returned as
                supplied without inserting a new layer.

        Returns:
            layers.Layer | Any: Stored child layer or ``default``.

        Raises:
            TypeError: key is unhashable, following ordinary dictionary lookup.
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
        name="stage_probe"
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
        LayerDict({"a": first}, execution_order=tuple(["missing"]))
    except ValueError:
        pass
    # This invalid case should already have raised: Invalid execution orders must fail.
    else:
        raise AssertionError("Invalid execution orders must fail.")

    return {"LayerDict": "passed"}


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
