from tensorflow.keras import callbacks


class RawNetworkValidationCallback(callbacks.Callback):

    def __init__(
        self, 
        val_x, 
        val_y=None
    ):
        super().__init__()

        self.val_x = val_x
        self.val_y = val_y

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        raw_results = self.model.evaluate(
            self.val_x, 
            self.val_y, 
            network_name="raw", 
            verbose=0, 
            return_dict=True,
        )

        for name, value in raw_results.items():
            logs[f"val_raw_{name}"] = value
