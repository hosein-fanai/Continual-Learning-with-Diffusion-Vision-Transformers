"""Share wrapper selectors and topology-aware copying for diffusion networks.

Wrappers own schedules, noising, losses, Keras training steps, EMA snapshots,
and sampling; compatible transformer, convolutional, and composite networks
own architecture and features. The aliases define raw/EMA selection,
conditional/unconditional auxiliary objectives, and curriculum clustering.

copy_network_weights_by_layer aligns active layers after class or depth
growth, including renamed composite decoder and terminal classifier layers.
Importing this module does not create models or copy any weights.
"""

from tensorflow.keras import models

from typing import Literal, TypeAlias


NetworkName: TypeAlias = Literal["ema", "raw"]
"""Selectable prediction copy: exponential-moving-average or trainable raw."""

TrainType: TypeAlias = Literal["cond", "uncond"]
"""Select conditional or null-label branch values for an auxiliary loss."""

ClusteringType: TypeAlias = Literal["uniform", "log_snr"]
"""Progressive timestep partitioning strategy."""


def copy_network_weights_by_layer(
    source_network: models.Model, 
    target_network: models.Model
) -> None:
    """Copy all model weights through stable active-layer names.

    Dynamic class expansion can change ``Model.weights`` ordering after layer
    replacement. Copying each active layer separately keeps expanded label and
    classifier heads aligned without depending on that global order.

    Pairs active top-level layers by name, then additionally pairs composite
    decoder internals and terminal classifier components. Each matched pair must
    have the same ordered weight shapes; both models must receive complete coverage.
    The operation changes target values in place without cloning layers, changing
    topology, or transferring optimizer state. A later mismatch can leave earlier
    pairs already copied; copying is not rolled back on failure.

    Args:
        source_network (tf.keras.Model): Built source network with the weights to preserve;
            transformer,
            convolutional, or compatible composite implementations are accepted.
        target_network (tf.keras.Model): Built clone with topology-equivalent active layers
            and assignable
            variables. Source and target architecture objects are not replaced.

    Returns:
        None: Every target weight receives its matching source value.

    Raises:
        ValueError: Matched layers have different ordered weight shapes or matched
            active layers do not cover every model weight in both networks.
    """

    source_layers = {
        layer.name: layer for layer in source_network.layers
    }
    target_layers = {
        layer.name: layer for layer in target_network.layers
    }
    # Pair only active layer names shared by the source and target.
    layer_pairs = [
        (source_layers[name], target_layer)
        for name, target_layer in target_layers.items()
        if name in source_layers
    ]

    source_decoder = getattr(source_network, "decoder", None)
    target_decoder = getattr(target_network, "decoder", None)

    # Composite classifiers can reconstruct their nested decoder model under a
    # different automatic model name; pair its active layers directly.
    if source_decoder is not None and target_decoder is not None:
        source_decoder_layers = {
            layer.name: layer for layer in source_decoder.layers
        }
        # Pair decoder internals by shared names when outer model names differ.
        layer_pairs.extend(
            (source_decoder_layers[name], target_layer)
            for name, target_layer in {
                layer.name: layer for layer in target_decoder.layers
            }.items()
            if name in source_decoder_layers
        )

    # Progressive classifiers retain their terminal connector under its old
    # layer name, while a config clone names it from the current depth.
    if getattr(source_network, "clf_layers_dicts", None) and getattr(
        target_network, "clf_layers_dicts", None
    ):
        source_terminal = source_network.clf_layers_dicts[-1]
        target_terminal = target_network.clf_layers_dicts[-1]
        # Match terminal classifier components that survive depth-dependent renaming.
        layer_pairs.extend(
            (source_terminal[name], target_layer)
            for name, target_layer in target_terminal.items()
            if name in source_terminal
        )

    copied_source_ids: set[int] = set()
    copied_target_ids: set[int] = set()
    copied_layer_pairs: set[tuple[int, int]] = set()
    for source_layer, target_layer in layer_pairs:
        pair_id = (id(source_layer), id(target_layer))
        # Avoid copying a same-named terminal layer twice.
        if pair_id in copied_layer_pairs:
            continue

        copied_layer_pairs.add(pair_id)
        source_shapes = [tuple(weight.shape) for weight in source_layer.weights]
        target_shapes = [tuple(weight.shape) for weight in target_layer.weights]

        # Reject a same-named layer whose reconstructed topology is different.
        if source_shapes != target_shapes:
            raise ValueError(
                "Teacher snapshot layer mismatch for "
                f"{source_layer.name!r}/{target_layer.name!r}: "
                f"{source_shapes} != {target_shapes}."
            )

        target_layer.set_weights(source_layer.get_weights())
        copied_source_ids.update(id(weight) for weight in source_layer.weights)
        copied_target_ids.update(id(weight) for weight in target_layer.weights)

    source_weight_ids = {id(weight) for weight in source_network.weights}
    target_weight_ids = {id(weight) for weight in target_network.weights}

    # Require complete coverage so a partial teacher can never be returned.
    if not source_weight_ids <= copied_source_ids \
    or not target_weight_ids <= copied_target_ids:
        raise ValueError(
            "Teacher snapshot could not match every network weight "
            f"(source {len(copied_source_ids & source_weight_ids)}/"
            f"{len(source_weight_ids)}, target "
            f"{len(copied_target_ids & target_weight_ids)}/"
            f"{len(target_weight_ids)})."
        )
