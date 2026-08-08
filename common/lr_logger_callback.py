from tensorflow.keras import callbacks
from tensorflow.keras import backend as K


class LrLoggerCallback(callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if callable(lr):
            lr = lr(self.model.optimizer.iterations)

        logs = logs or {}
        logs["learning_rate"] = float(K.get_value(lr))
