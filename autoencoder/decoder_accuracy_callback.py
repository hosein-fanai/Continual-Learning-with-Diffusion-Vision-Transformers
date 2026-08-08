import tensorflow as tf
from tensorflow.keras import callbacks


class DecoderAccuracyCallback(callbacks.Callback):

    def __init__(
        self,
        classifier,
        samples_per_class=500,
    ):
        super().__init__()
        self.samples_per_class = samples_per_class
        self.classifier = classifier

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        x_gen, y_true = self.model.generate(
            samples_per_class=self.samples_per_class, 
            onehot_y_output=False
        )

        y_pred = self.classifier(x_gen, training=False)
        y_pred = tf.argmax(y_pred, axis=1)

        corrects = tf.cast(y_pred == y_true, dtype=tf.float16)
        acc = tf.reduce_mean(corrects)

        logs["decoder_accuracy"] = acc.numpy()
