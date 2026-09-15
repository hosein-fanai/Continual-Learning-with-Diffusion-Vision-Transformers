"""Small Keras 3 bridges shared by project models and diagnostics."""

import tensorflow as tf

from copy import deepcopy

from collections.abc import Sequence


NAME_SEPARATOR = "__"


def display_name(name: object) -> str:
    """Render framework paths with the project's readable hierarchy separator."""

    return str(name).replace("/", NAME_SEPARATOR)


def variable_path(variable: object) -> str:
    """Return the full Keras variable path, including its owning layer."""

    return getattr(variable, "path", None) or variable.name


def format_variable_name(variable: object) -> str:
    """Format both Keras variables and raw TensorFlow variables consistently."""

    return display_name(variable_path(variable))


def optimizer_iterations(optimizer: object) -> object:
    """Return actual update counts, including Keras 3.4 loss-scale wrappers."""

    while hasattr(optimizer, "inner_optimizer"):
        optimizer = optimizer.inner_optimizer

    return getattr(optimizer, "iterations", None)


def register_optimizer_variables(
    optimizer: tf.keras.optimizers.Optimizer, 
    variables: Sequence[object]
) -> tf.keras.optimizers.Optimizer:
    """Return an optimizer covering ``variables`` without discarding old slots.

    Keras 3 optimizers cannot extend their variable registry after ``build``.
    Recreate only when a new variable appears, retaining matching optimizer
    state (including iterations, loss scaling, and existing moment estimates).
    Slots for a changed shape start at the optimizer's default. This helper
    does not permit adding layers to an already built Keras model.
    """

    variables = list(dict((id(v), v) for v in variables).values())

    # Empty phases need no optimizer registry.
    if not variables:
        return optimizer

    # Initial registration uses the public optimizer build API.
    if not optimizer.built:
        optimizer.build(variables)

        return optimizer

    owner = getattr(optimizer, "inner_optimizer", optimizer)
    known = {id(v) for v in owner._trainable_variables}

    # Keep the existing instance when all selected variables are registered.
    if all(id(v) in known for v in variables):
        return optimizer

    replacement = type(optimizer).from_config(deepcopy(optimizer.get_config()))
    replacement.build(variables)

    def copy_state(source: object, destination: object) -> None:
        """Transfer unambiguous local optimizer state through nested wrappers."""

        # A LossScaleOptimizer may build its inner optimizer under a temporary
        # outer name scope. Match local state names separately at each level.
        source_inner = getattr(source, "inner_optimizer", None)
        old_values = source._variables if source_inner is not None else source.variables
        new_values = destination._variables if source_inner is not None else destination.variables
        old_state = {}
        for value in old_values:
            key = value.name, tuple(value.shape)

            if key in old_state:
                raise ValueError(f"Ambiguous optimizer state name: {key[0]}")

            old_state[key] = value
        for value in new_values:
            old_value = old_state.get((value.name, tuple(value.shape)))
            # New or reshaped variable slots retain their initialized values.
            if old_value is not None:
                value.assign(old_value)

        # Preserve inner slots independently of the wrapper's own counters.
        if source_inner is not None:
            copy_state(source_inner, destination.inner_optimizer)

    copy_state(optimizer, replacement)

    return replacement
