import tensorflow as tf
from tensorflow.keras import losses


class MaskedLoss(losses.Loss):
    """
    Error computed only on the first N logits in order to match the shape of the labels and logits.
    """

    def __init__(self, name="masked_loss"):
        super(MaskedLoss, self).__init__(name=name)
        self.loss = losses.MeanAbsoluteError()

    def call(self, y_true, y_pred):
        n = tf.shape(y_pred)[1]

        y_true_last = y_true[..., :n]
        y_pred_last = y_pred

        return self.loss(y_true_last, y_pred_last)

