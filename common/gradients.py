"""Policy-aware gradient application for custom TensorFlow training steps."""

from __future__ import annotations

import tensorflow as tf

from collections.abc import Sequence


Gradient = tf.Tensor | tf.IndexedSlices


def apply_policy_gradients(
    tape: tf.GradientTape, 
    optimizer: tf.keras.optimizers.Optimizer, 
    loss: tf.Tensor, 
    variables: Sequence[tf.Variable]
) -> list[tuple[Gradient, tf.Variable]]:
    """Differentiate and apply an unscaled objective under any Keras policy.

    TensorFlow's mixed-float16 ``Model.compile`` wraps an optimizer in
    :class:`tf.keras.mixed_precision.LossScaleOptimizer`.  Custom training
    loops must then scale the loss before differentiation and unscale the
    resulting gradients exactly once.  Ordinary optimizers retain the direct
    differentiation path.  Variables disconnected from ``loss`` are omitted
    rather than forwarding ``None`` gradients to an optimizer.

    Args:
        tape (tf.GradientTape): Unconsumed tape that recorded ``loss``.
        optimizer (tf.keras.optimizers.Optimizer): Ordinary optimizer or Keras
            loss-scale wrapper used for the update.
        loss (tf.Tensor): Unscaled scalar training objective.
        variables (Sequence[tf.Variable]): Candidate variables to update.

    Returns:
        list[tuple[tf.Tensor | tf.IndexedSlices, tf.Variable]]: Non-``None``
        gradient-variable pairs that were applied, or an empty list when no
        variable is connected to ``loss``.
    """

    selected_variables = list(variables)

    # Avoid asking GradientTape or an optimizer to process an empty selection.
    if not selected_variables:
        return []

    loss_scale_type = tf.keras.mixed_precision.LossScaleOptimizer
    uses_loss_scaling = isinstance(optimizer, loss_scale_type)

    # Re-enter the still-unconsumed tape so loss scaling itself is recorded.
    if uses_loss_scaling:
        with tape:
            gradient_loss = optimizer.get_scaled_loss(loss)
    # Ordinary optimizers differentiate the original loss directly.
    else:
        gradient_loss = loss

    gradients = tape.gradient(gradient_loss, selected_variables)

    # Convert scaled gradients back exactly once before optimizer application.
    if uses_loss_scaling:
        gradients = optimizer.get_unscaled_gradients(gradients)

    # Let the None-valued grads get into the optimizer, so Keras shows a warning (good for noticing mistakes).
    pairs = zip(gradients, selected_variables)
    optimizer.apply_gradients(pairs)

    return pairs
