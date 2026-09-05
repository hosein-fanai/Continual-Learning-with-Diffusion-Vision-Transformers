"""Serializable MAE/MSE losses for comparing predictions with target prefixes.

``MaskedLoss`` truncates only the final target dimension to the prediction width,
checks that examples/features remain aligned, and averages non-batch dimensions.
Keras then applies sample weights and the configured batch reduction. Importing
this module registers the loss under the ``continual_learning`` Keras namespace.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import losses


@tf.keras.utils.register_keras_serializable(package="continual_learning")
class MaskedLoss(losses.Loss):
    """Compute MAE or MSE after truncating the target's last dimension.

    For classification-style tensors, ``N`` is
    ``tf.shape(y_pred)[-1]`` and the loss compares ``y_pred`` with
    ``y_true[..., :N]``.  Thus a target of shape ``[batch, 10]`` and prediction
    of shape ``[batch, 4]`` uses only target columns ``0`` through ``3``.  Equal
    widths behave like the selected standard Keras loss. Both tensors must have
    the same rank of at least two and matching dimensions except for the last;
    the target cannot be narrower than the prediction. Empty prediction widths
    have an undefined mean, as with ordinary MAE/MSE.

    Attributes:
        loss (Callable): Elementwise ``tf.math.abs`` for ``"mae"`` or
            ``tf.math.square`` for ``"mse"``.
        loss_type (str): Serializable loss-mode identifier.
    """

    def __init__(
        self: MaskedLoss, 
        loss_type: str = "mae", 
        name: str = "masked_loss", 
        reduction: str = losses.Reduction.AUTO
    ) -> None:
        """Initialize the selected elementwise regression loss.

        Args:
            loss_type (str): Exactly ``"mae"`` or ``"mse"``.  The default is
                mean absolute error.
            name (str): Keras loss name; defaults to ``"masked_loss"``.
            reduction (str): Keras loss reduction. The TensorFlow 2.10 default
                is ``tf.keras.losses.Reduction.AUTO``.

        Returns:
            None.

        Raises:
            ValueError: If ``loss_type`` is unsupported.
        """

        super(MaskedLoss, self).__init__(name=name, reduction=reduction)

        # Use absolute elementwise error for MAE mode.
        if loss_type == "mae":
            self.loss = tf.math.abs
        # Use squared elementwise error for MSE mode.
        elif loss_type == "mse":
            self.loss = tf.math.square
        # Reject loss modes outside the two documented choices.
        else:
            raise ValueError("loss_type needs to be one of 'mae' or 'mse'.")

        self.loss_type = loss_type

    def call(
        self: MaskedLoss, 
        y_true: tf.Tensor, 
        y_pred: tf.Tensor
    ) -> tf.Tensor:
        """Evaluate the configured loss on the target prefix.

        Args:
            y_true (tf.Tensor): Ground-truth values, normally shaped
                ``[batch, target_width]`` with ``target_width >= pred_width``.
            y_pred (tf.Tensor): Predictions shaped ``[batch, pred_width]``.

        Returns:
            tf.Tensor: One MAE or MSE value per batch row, averaged over every
            non-batch axis after truncating the target's last dimension. Keras
            then applies the configured reduction and sample weights.

        Raises:
            ValueError: If a statically shaped tensor has rank below two.
            tf.errors.InvalidArgumentError: If a dynamically checked rank is
                below two, the ranks or non-last dimensions differ, or the
                target's last dimension is narrower than the prediction.
        """

        prefix = tf.cast(y_true[..., :tf.shape(y_pred)[-1]], y_pred.dtype)
        # Prevent broadcasting from changing which examples/features are paired.
        checks = (
            tf.debugging.assert_rank_at_least(y_pred, 2),
            tf.debugging.assert_equal(tf.rank(prefix), tf.rank(y_pred)),
            tf.debugging.assert_equal(
                tf.shape(prefix), tf.shape(y_pred),
                message="The target prefix must match the prediction shape.",
            ),
        )
        # Attach graph assertion operations; omit eager assertions that return None.
        with tf.control_dependencies([check for check in checks if check is not None]):
            error = self.loss(prefix - y_pred)
            return tf.reduce_mean(error, axis=tf.range(1, tf.rank(error)))

    def get_config(self: MaskedLoss) -> dict[str, object]:
        """Return a Keras-serializable configuration.

        Returns:
            dict[str, object]: Base loss settings plus ``loss_type``.
        """

        return {**super().get_config(), "loss_type": self.loss_type}
