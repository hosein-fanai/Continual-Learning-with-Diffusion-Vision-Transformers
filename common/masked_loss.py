"""Serializable losses for comparing predictions with target prefixes."""

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
    the prediction width must be positive, and the target's last dimension
    cannot be narrower than the prediction's.

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

        rank_true = tf.debugging.assert_rank_at_least(y_true, 2)
        rank_pred = tf.debugging.assert_rank_at_least(y_pred, 2)
        same_rank = tf.debugging.assert_equal(
            tf.rank(y_true),
            tf.rank(y_pred),
            message="y_true and y_pred must have the same rank.",
        )
        same_leading_shape = tf.debugging.assert_equal(
            tf.shape(y_true)[:-1],
            tf.shape(y_pred)[:-1],
            message="y_true and y_pred must match outside the last dimension.",
        )
        width_assertion = tf.debugging.assert_greater_equal(
            tf.shape(y_true)[-1],
            tf.shape(y_pred)[-1],
            message="y_true must be at least as wide as y_pred.",
        )
        positive_width = tf.debugging.assert_positive(
            tf.shape(y_pred)[-1],
            message="y_pred must have a positive last dimension.",
        )
        assertions = [
            assertion for assertion in (
                rank_true,
                rank_pred,
                same_rank,
                same_leading_shape,
                width_assertion,
                positive_width,
            ) if assertion is not None
        ]

        with tf.control_dependencies(assertions):
            n = tf.shape(y_pred)[-1]
            y_true_last = tf.cast(y_true[..., :n], y_pred.dtype)
            error = y_true_last - y_pred

        non_batch_axes = tf.range(1, tf.rank(error))
        return tf.reduce_mean(self.loss(error), axis=non_batch_axes)

    def get_config(self: MaskedLoss) -> dict[str, object]:
        """Return a Keras-serializable configuration.

        Returns:
            dict[str, object]: Base loss settings plus ``loss_type``.
        """

        return {**super().get_config(), "loss_type": self.loss_type}
