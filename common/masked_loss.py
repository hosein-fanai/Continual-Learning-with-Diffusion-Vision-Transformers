"""Loss functions for comparing a prediction with a prefix of its target."""

import tensorflow as tf
from tensorflow.keras import losses


class MaskedLoss(losses.Loss):
    """Compute MAE or MSE after truncating the target's last dimension.

    For rank-two classification-style tensors, ``N`` is
    ``tf.shape(y_pred)[1]`` and the loss compares ``y_pred`` with
    ``y_true[..., :N]``.  Thus a target of shape ``[batch, 10]`` and prediction
    of shape ``[batch, 4]`` uses only target columns ``0`` through ``3``.  Equal
    widths behave like the selected standard Keras loss.  The implementation is
    intended for tensors of rank at least two; for ranks above two, ``N`` still
    comes from prediction axis 1 while slicing occurs on the target's last axis.

    Attributes:
        loss (tf.keras.losses.Loss): ``MeanAbsoluteError`` when ``loss_type`` is
            ``"mae"`` or ``MeanSquaredError`` when it is ``"mse"``.
    """

    def __init__(self, loss_type="mae", name="masked_loss"):
        """Initialize the selected elementwise regression loss.

        Args:
            loss_type (str): Exactly ``"mae"`` or ``"mse"``.  The default is
                mean absolute error.
            name (str): Keras loss name; defaults to ``"masked_loss"``.

        Returns:
            None.

        Raises:
            TypeError: If ``loss_type`` is unsupported.  The current
                implementation attempts to raise a descriptive string, which
                Python surfaces as ``TypeError``.
        """
        super(MaskedLoss, self).__init__(name=name)

        if loss_type == "mae":
            self.loss = losses.MeanAbsoluteError()
        elif loss_type == "mse":
            self.loss = losses.MeanSquaredError()
        else:
            raise("loss_type needs to be one of mae or mse.")

    def call(self, y_true, y_pred):
        """Evaluate the configured loss on the target prefix.

        Args:
            y_true (tf.Tensor): Ground-truth values, normally shaped
                ``[batch, target_width]`` with ``target_width >= pred_width``.
            y_pred (tf.Tensor): Predictions shaped ``[batch, pred_width]``.

        Returns:
            tf.Tensor: Scalar mean MAE or MSE using
            ``y_true[..., :tf.shape(y_pred)[1]]``.
        """
        n = tf.shape(y_pred)[1]

        y_true_last = y_true[..., :n]
        y_pred_last = y_pred

        return self.loss(y_true_last, y_pred_last)
