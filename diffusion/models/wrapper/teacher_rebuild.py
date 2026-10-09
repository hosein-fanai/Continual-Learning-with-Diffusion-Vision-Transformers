"""Rebuild teacher parameters while retaining deliberate layer freezing."""


import tensorflow as tf

from copy import deepcopy

from common.runtime import derive_seed


def _teacher_graph(network: tf.keras.layers.Layer) -> tuple[dict, dict]:
    """Index distinct layers and every parent edge, preserving shared-layer identity."""

    result = {}
    paths = {}
    children = {}


    def visit(layer: tf.keras.layers.Layer, path: str) -> str:
        """Record the first path and retain edges from every additional parent."""

        # Shared graph layers own one fine-tuning flag and one set of values.
        if id(layer) in paths:
            return paths[id(layer)]

        paths[id(layer)] = path
        result[path] = layer
        children[path] = []
        for index, child in enumerate(layer._flatten_layers(include_self=False, recursive=False)):
            child_path = visit(child, f"{path}/{index}:{type(child).__name__}")
            # Duplicate direct references still represent one parent-child edge.
            if child_path not in children[path]:
                children[path].append(child_path)
        
        return path


    visit(network, "root")

    return result, children


def _teacher_layer_order(
    layers: dict, 
    children: dict
) -> list[str]:
    """Order every parent before a shared child so trainable setters cannot overwrite it."""

    pending = dict.fromkeys(layers, 0)
    for descendants in children.values():
        for child in descendants:
            pending[child] += 1

    order = [path for path, count in pending.items() if count == 0]
    for path in order:
        for child in children[path]:
            pending[child] -= 1
            # A shared child becomes safe only after all its parents have been restored.
            if pending[child] == 0:
                order.append(child)
    
    # Cyclic ownership cannot define a reliable parent-first fine-tuning policy.
    if len(order) != len(layers):
        raise ValueError("Teacher layer ownership must be acyclic.")
    
    return order


def teacher_layer_trainability(
    network: tf.keras.layers.Layer, 
    layer_states: tuple | list | None = None
) -> dict[str, bool]:
    """Capture structural layer flags, optionally using pre-attachment object/flag pairs.

    Supplied states avoid reading the globally frozen flags of an attached teacher.
    Shared layer objects are recorded once; no source flags or values are changed.
    """

    overrides = None if layer_states is None else {id(layer): bool(flag) for layer, flag in layer_states}
    result = {}
    layers, _ = _teacher_graph(network)
    for path, layer in layers.items():
        # A missing saved layer would silently change the fine-tuning policy.
        if overrides is not None and id(layer) not in overrides:
            raise ValueError(
                "Teacher fine-tuning state does not cover its current topology."
            )
        
        result[path] = bool(layer.trainable) if overrides is None else overrides[id(layer)]
    
    return result


def materialize_teacher_trainability(network: tf.keras.layers.Layer) -> dict[str, bool]:
    """Build a native teacher before its initial mask is captured, preserving frozen parents.

    Progressive native networks can be marked built while appended layers still
    create children lazily. Existing flags are retained by object identity; newly
    created descendants inherit any frozen ancestor before the complete mask is
    restored. This is an initial attachment operation, not a relaxed restore path.
    """

    original_layers, _ = _teacher_graph(network)
    original_flags = {
        id(layer): bool(layer.trainable) 
        for layer in original_layers.values()
    }
    network.build()
    layers, children = _teacher_graph(network)
    state = {}
    inherited = {path: set() for path in layers}
    inherited["root"].add(True)
    for path in _teacher_layer_order(layers, children):
        layer = layers[path]
        state[path] = original_flags.get(id(layer), bool(layer.trainable) and all(inherited[path]))
        active = {flag and state[path] for flag in inherited[path]}
        for child in children[path]:
            inherited[child].update(active)
    
    restore_teacher_trainability(network, state)
    
    return state


def restore_teacher_trainability(
    network: tf.keras.layers.Layer, 
    state: dict[str, bool]
) -> None:
    """Restore parent-before-child flags only when the complete layer topology matches."""

    layers, children = _teacher_graph(network)
    # Reattachment must not apply an old mask to a different topology.
    if set(layers) != set(state):
        raise ValueError(
            "Teacher fine-tuning state does not match its current topology."
        )
    
    for path in _teacher_layer_order(layers, children):
        layers[path].trainable = state[path]


def copy_frozen_teacher_weights(
    source: tf.keras.layers.Layer, 
    candidate: tf.keras.layers.Layer, 
    state: dict[str, bool]
) -> None:
    """Retain all variables owned by frozen layers and leave other initial values fresh.

    Structural paths distinguish identically named nested layers. Direct variable
    ownership prevents a trainable parent from hiding a frozen child's batch-normalization
    statistics, or from copying a trainable child's weights. Shape and shared-variable
    correspondence are checked before assigning values to the unattached candidate.
    """

    source_layers, source_children = _teacher_graph(source)
    candidate_layers, candidate_children = _teacher_graph(candidate)
    # Structural keys also expose changed sharing or additional nested layers.
    if set(source_layers) != set(candidate_layers) or set(state) != set(source_layers) \
    or source_children != candidate_children:
        raise ValueError("Rebuilt teacher layer topology differs from the source.")
    
    source_aliases = {}
    candidate_aliases = {}
    frozen_variables = {}
    assignments = []
    effective = {path: set() for path in source_layers}
    effective["root"].add(True)
    for path in _teacher_layer_order(source_layers, source_children):
        layer = source_layers[path]
        replacement = candidate_layers[path]
        # Matching positions must still denote the same layer implementations.
        if type(layer) is not type(replacement):
            raise ValueError("Rebuilt teacher layer types differ from the source.")

        source_values = [*layer._trainable_variables, *layer._non_trainable_variables]
        candidate_values = [*replacement._trainable_variables, *replacement._non_trainable_variables]
        # Copying a prefix could otherwise leave a partially preserved frozen layer.
        if len(source_values) != len(candidate_values):
            raise ValueError("Rebuilt teacher variable counts differ from the source.")

        active = {parent_active and state[path] for parent_active in effective[path]}
        for child in source_children[path]:
            effective[child].update(active)

        # A shared variable cannot be both retained and freshly initialized for different parents.
        if source_values and len(active) > 1:
            raise ValueError("Shared teacher weights have conflicting frozen parent policies.")

        frozen = active == {False}
        for index, (original, fresh) in enumerate(zip(source_values, candidate_values)):
            key = (path, index)
            source_alias = source_aliases.setdefault(id(original), key)
            candidate_alias = candidate_aliases.setdefault(id(fresh), key)
            # Preserve semantic variable pairing and shared-variable identity.
            if original is fresh or source_alias != candidate_alias \
            or tuple(original.shape) != tuple(fresh.shape) or str(original.dtype) != str(fresh.dtype):
                raise ValueError("Rebuilt teacher variable structure differs from the source.")

            previous_policy = frozen_variables.setdefault(id(original), frozen)
            # A variable registered by several layers needs one consistent reset policy.
            if previous_policy != frozen:
                raise ValueError("Shared teacher weights have conflicting frozen layer policies.")

            # Trainable layers retain constructor values, including running statistics.
            if frozen:
                assignments.append((original, fresh))

    restore_teacher_trainability(candidate, state)
    for original, fresh in assignments:
        fresh.assign(original)


def fresh_teacher_config(
    config: object, 
    seed: int | None = None
) -> object:
    """Copy configuration and optionally derive explicit seeds without changing global RNG.

    None preserves each configured initializer seed. A supplied seed replaces nested
    scalar seed settings deterministically, including native model and Keras initializer
    seeds; all other constructor settings retain their configured values.
    """


    def copy_value(
        value: object, 
        path: tuple
    ) -> object:
        """Recursively copy configuration containers and replace declared seed fields."""

        # Serialized models and initializers declare their own scalar seed fields.
        if isinstance(value, dict):
            return {
                key: derive_seed(seed, "teacher_rebuild", str(path), key)
                if key == "seed" and seed is not None and not isinstance(item, (dict, list, tuple))
                else copy_value(item, (*path, key))
                for key, item in value.items()
            }
        # Rebuild containers without retaining the original metadata tracker.
        if isinstance(value, list):
            return [copy_value(item, (*path, index)) for index, item in enumerate(value)]
        # Tuple-valued shape/configuration fields retain their container type.
        if isinstance(value, tuple):
            return tuple(copy_value(item, (*path, index)) for index, item in enumerate(value))
        return deepcopy(value)


    return copy_value(config, ())
