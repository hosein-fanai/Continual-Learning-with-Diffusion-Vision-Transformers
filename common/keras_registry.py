"""Lazy Keras custom-object registration for package-level deserialization."""

from __future__ import annotations

import tensorflow as tf

import sys

from importlib import import_module

from collections.abc import Mapping
from typing import Callable


def register_canonical_keras_serializable(
    package: str = "continual_learning", 
    name: str | None = None
) -> Callable[[type], type]:
    """Return a Keras decorator that can replace this project's lazy proxy.

    Args:
        package (str): Keras serialization namespace used before ``>``.
        name (str | None): Optional registered name; the class name is used
            when omitted.

    Returns:
        Callable[[type], type]: Decorator that installs the canonical class.
    """

    def decorator(target: type) -> type:
        """Replace a matching proxy, then perform standard Keras registration.

        Args:
            target (type): Canonical Keras model or layer class.

        Returns:
            type: The same class after standard Keras registration.
        """

        object_name = name or target.__name__
        registered_name = f"{package}>{object_name}"
        custom_objects = tf.keras.utils.get_custom_objects()
        existing = custom_objects.get(registered_name)
        target_module = target.__module__

        # Resolve the canonical spec name during ``python -m package.module``.
        if target_module == "__main__":
            main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
            target_module = getattr(main_spec, "name", target_module)

        expected_target = (target_module, target.__name__)

        # Remove only the proxy installed for this exact canonical class.
        if getattr(existing, "_continual_lazy_target", None) == expected_target:
            del custom_objects[registered_name]

        return tf.keras.utils.register_keras_serializable(
            package=package, 
            name=name
        )(target)

    return decorator


def register_lazy_keras_serializable(
    module_name: str, 
    attribute_name: str, 
    package: str = "continual_learning", 
    aliases: tuple[str, ...] = ()
) -> None:
    """Register a class proxy that imports its implementation on demand.

    The proxy lets ``import autoencoder`` or ``import diffusion`` prepare Keras
    deserialization without importing every implementation module.  When Keras
    asks the proxy to construct an object, it imports the canonical class and
    delegates construction; that module's ordinary serialization decorator
    then replaces the proxy in the global registry.

    Args:
        module_name (str): Absolute module containing the canonical class.
        attribute_name (str): Class name exported by ``module_name``.
        package (str): Keras serialization namespace used before ``>``.
        aliases (tuple[str, ...]): Additional legacy registry keys that should
            resolve through the same proxy.

    Returns:
        None: The proxy is installed, or an existing registration is retained.
    """

    registered_name = f"{package}>{attribute_name}"
    custom_objects = tf.keras.utils.get_custom_objects()
    # Preserve an actual class or deliberate caller override already present.
    if registered_name in custom_objects:
        return


    class LazySerializableProxy:
        """Stand in for one registered class until Keras needs to construct it."""

        def __new__(cls, *args: object, **kwargs: object) -> object:
            """Construct the canonical class for name-only deserialization.

            Args:
                *args (object): Positional constructor values.
                **kwargs (object): Keyword constructor values.

            Returns:
                object: Instance of the lazily imported canonical class.
            """

            target = getattr(import_module(module_name), attribute_name)

            return target(*args, **kwargs)

        @classmethod
        def from_config(
            cls, 
            config: Mapping[str, object], 
            custom_objects: Mapping[str, object] | None = None
        ) -> object:
            """Delegate config reconstruction to the canonical class.

            Args:
                config (Mapping[str, object]): Serialized constructor state.
                custom_objects (Mapping[str, object] | None): Custom-object
                    scope supplied by Keras for nested deserialization.

            Returns:
                object: Canonical class instance reconstructed from ``config``.
            """

            target = getattr(import_module(module_name), attribute_name)
            scope = dict(custom_objects or {})
            with tf.keras.utils.custom_object_scope(scope):
                return target.from_config(dict(config))


    LazySerializableProxy.__name__ = attribute_name
    LazySerializableProxy.__qualname__ = attribute_name
    LazySerializableProxy.__module__ = module_name
    LazySerializableProxy._continual_lazy_target = (
        module_name, 
        attribute_name
    )

    custom_objects[registered_name] = LazySerializableProxy
    for alias in aliases:
        # Preserve an existing canonical or caller-provided legacy alias.
        if alias not in custom_objects:
            custom_objects[alias] = LazySerializableProxy
