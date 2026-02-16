import tensorflow as tf
from tensorflow.keras import callbacks, models

import numpy as np


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
            onehot_labels=False
        )

        y_pred = self.classifier.predict(x_gen, verbose=0)
        y_pred = np.argmax(y_pred, axis=1)

        acc = np.mean(y_pred == y_true)

        logs["decoder_accuracy"] = acc

