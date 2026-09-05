"""Apply policy-aware gradients in custom TensorFlow training steps.

``apply_policy_gradients`` differentiates an unscaled objective, handles the
TensorFlow 2.10 LossScaleOptimizer protocol, and updates the selected variables.
Ordinary and mixed-precision callers share one API; empty selections are no-ops,
and disconnected objectives are reported before applying an update.
"""

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
    resulting gradients exactly once. Ordinary optimizers retain the direct
    differentiation path. Partially disconnected variables are forwarded with
    ``None`` gradients so Keras emits its standard warning, while the returned
    list contains only connected pairs submitted to the optimizer. Dynamic
    loss scaling may skip a nonfinite update according to optimizer policy.

    Args:
        tape (tf.GradientTape): Unconsumed tape that recorded ``loss``.
        optimizer (tf.keras.optimizers.Optimizer): Ordinary optimizer or Keras
            loss-scale wrapper used for the update.
        loss (tf.Tensor): Unscaled scalar training objective.
        variables (Sequence[tf.Variable]): Candidate variables to update.

    Returns:
        list[tuple[tf.Tensor | tf.IndexedSlices, tf.Variable]]: Non-``None``
        gradient-variable pairs submitted to the optimizer. An explicitly empty
        variable selection remains a no-op.

    Raises:
        ValueError: If variables were selected but the objective is completely
            disconnected from them.
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

    gradient_variable_pairs = list(zip(gradients, selected_variables))
    # Exclude disconnected variables from the returned gradient-variable pairs.
    pairs = [
        (gradient, variable)
        for gradient, variable in gradient_variable_pairs
        if gradient is not None
    ]
    # Reject a selected objective that is disconnected from every variable.
    if not pairs:
        raise ValueError(
            "The loss is disconnected from every selected variable."
        )

    optimizer.apply_gradients(gradient_variable_pairs)

    return pairs
