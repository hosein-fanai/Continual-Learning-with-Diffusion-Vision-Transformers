"""Epoch-end validation against raw rather than EMA diffusion weights."""

from tensorflow.keras import callbacks


class RawNetworkValidationCallback(callbacks.Callback):
    """Add raw-network validation metrics to Keras epoch logs.

    Diffusion wrappers normally evaluate their EMA network. This callback makes
    an additional call with ``network_name="raw"`` and prefixes every returned
    metric with ``"val_raw_"``. The bound model must therefore implement the
    project's extended ``evaluate`` signature; a plain Keras model is not
    sufficient.

    Args:
        val_x: Validation input accepted by the bound model's ``evaluate``
            method, such as an image tensor/NumPy array or a ``tf.data.Dataset``.
            When it is a dataset yielding ``(x, y)``, leave ``val_y`` as
            ``None``.
        val_y: Optional validation targets accepted by ``model.evaluate``.

    Inputs:
        Stored validation inputs/targets plus Keras epoch indices and log
        mappings. Tensor and dataset dtypes follow the bound wrapper's normal
        evaluation contract.

    Outputs:
        Callback hooks return ``None`` and add scalar ``val_raw_*`` entries to
        a truthy epoch ``logs`` mapping. A passed empty mapping is replaced
        locally by the current ``logs or {}`` expression and is not observably
        mutated by its caller.
    """

    def __init__(
        self, 
        val_x, 
        val_y=None
    ):
        """Store validation data for reuse at every epoch boundary.

        Arguments and accepted types are documented on the class.

        Returns:
            ``None``.
        """

        super().__init__()

        self.val_x = val_x
        self.val_y = val_y

    def on_epoch_end(self, epoch, logs=None):
        """Evaluate raw weights and append the results to epoch logs.

        Args:
            epoch: Zero-based integer epoch index supplied by Keras; unused by
                this callback.
            logs: Optional mutable mapping of epoch metrics. Each raw evaluation
                result named ``key`` is assigned as ``logs["val_raw_" + key]``.
                If ``None`` or empty, a temporary mapping is created.

        Returns:
            ``None``. The bound model's ``evaluate`` call is executed with
            ``verbose=0`` and ``return_dict=True``.
        """

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


def run_self_tests() -> dict[str, str]:
    """Test raw-network evaluation and log mutation semantics.

    Args:
        None.

    Returns:
        A one-entry mapping after tensor/dataset-style target storage,
        evaluation arguments, metric prefixing, and empty-log behavior checks.
    """

    from types import SimpleNamespace
    from unittest.mock import Mock

    validation_x = [1, 2]
    validation_y = [0, 1]
    evaluate = Mock(return_value={"loss": 0.25, "accuracy": 0.75})
    callback = RawNetworkValidationCallback(validation_x, validation_y)
    callback.set_model(SimpleNamespace(evaluate=evaluate))
    logs = {"loss": 1.0}
    result = callback.on_epoch_end(3, logs)
    assert result is None
    assert logs == {
        "loss": 1.0, 
        "val_raw_loss": 0.25, 
        "val_raw_accuracy": 0.75, 
    }
    evaluate.assert_called_once_with(
        validation_x, 
        validation_y, 
        network_name="raw", 
        verbose=0, 
        return_dict=True, 
    )

    dataset_like = [("x", "y")]
    dataset_evaluate = Mock(return_value={"noise_loss": 0.5})
    dataset_callback = RawNetworkValidationCallback(dataset_like)
    dataset_callback.set_model(SimpleNamespace(evaluate=dataset_evaluate))
    assert dataset_callback.on_epoch_end(0, None) is None
    dataset_evaluate.assert_called_once_with(
        dataset_like, 
        None, 
        network_name="raw", 
        verbose=0, 
        return_dict=True, 
    )

    empty_logs = {}
    dataset_callback.on_epoch_end(1, empty_logs)
    assert empty_logs == {}

    return {"RawNetworkValidationCallback": "passed"}


if __name__ == "__main__":
    print(run_self_tests())
