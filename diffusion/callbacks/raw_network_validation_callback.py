"""Epoch-end validation against raw rather than EMA diffusion weights.

RawNetworkValidationCallback performs an additional evaluation with raw network
weights and appends val_raw_* metrics to epoch logs. It relies on diffusion
wrappers accepting keyword evaluation inputs and the network_name selector.
"""

from tensorflow.keras import callbacks

from typing import Any


class RawNetworkValidationCallback(callbacks.Callback):
    """Add raw-network validation metrics to Keras epoch logs.

    Diffusion wrappers normally evaluate their EMA network. This callback makes
    an additional call with ``network_name="raw"`` and prefixes every returned
    metric with ``"val_raw_"``. The bound model must therefore implement the
    project's extended ``evaluate`` signature; a plain Keras model is not
    sufficient.

    Args:
        val_x (Any): Validation input accepted by the bound model's ``evaluate``
            method, such as an image tensor/NumPy array or a ``tf.data.Dataset``.
            When it is a dataset yielding ``(x, y)``, leave ``val_y`` as
            ``None``.
        val_y (Any | None): Optional validation targets accepted by ``model.evaluate``.
            Defaults to ``None``.

    Inputs:
        Stored validation inputs/targets plus Keras epoch indices and log
        mappings. Tensor and dataset dtypes follow the bound wrapper's normal
        evaluation contract.

    Outputs:
        Callback hooks return ``None`` and add scalar ``val_raw_*`` entries to
        the supplied epoch ``logs`` mapping. Both populated and empty mappings
        are mutated in place; when Keras supplies ``None``, results are computed
        but no external mapping exists to observe them.

    Attributes:
        val_x (Any): Validation inputs retained by reference for each epoch evaluation.
        val_y (Any | None): Optional separate targets; None when inputs already include
            targets.
    """

    def __init__(
        self, 
        val_x: Any, 
        val_y: Any | None = None
    ) -> None:
        """Store validation data for reuse at every epoch boundary.

        Args:
            val_x (Any): Validation inputs or a dataset accepted by the bound
                model's ``evaluate`` method.
            val_y (Any | None): Optional validation targets. Leave this as
                ``None`` when ``val_x`` already yields input-target pairs.
                Defaults to ``None``.

        Returns:
            None: No value is returned.
        """

        super().__init__()

        self.val_x = val_x
        self.val_y = val_y

    def on_epoch_end(
        self, 
        epoch: int, 
        logs: dict[str, Any] | None = None
    ) -> None:
        """Evaluate raw weights and append the results to epoch logs.

        Args:
            epoch (int): Zero-based epoch index supplied by Keras; unused by
                this callback.
            logs (dict[str, Any] | None): Optional mutable epoch-metric mapping.
                Each raw result named ``key`` is stored under
                ``"val_raw_" + key``. A passed mapping, including an empty one,
                is mutated in place.
                Defaults to ``None``. No caller-owned log mapping is available in that case.

        Returns:
            None: The bound model's ``evaluate`` call is executed with
            ``verbose=0`` and ``return_dict=True``.
        """

        # Create a local empty mapping only when Keras supplies no log mapping.
        logs = {} if logs is None else logs

        raw_results = self.model.evaluate(
            x=self.val_x,
            y=self.val_y,
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
        dict[str, str]: A one-entry mapping after tensor/dataset-style target storage,
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
        x=validation_x,
        y=validation_y,
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
        x=dataset_like,
        y=None,
        network_name="raw", 
        verbose=0, 
        return_dict=True, 
    )

    empty_logs = {}
    dataset_callback.on_epoch_end(1, empty_logs)
    assert empty_logs == {"val_raw_noise_loss": 0.5}

    class V2StyleModel:
        """Record evaluation arguments using the classifier-V2 positional control layout.

        Attributes:
            call (tuple[bool, str | None, dict[str, Any]] | None): Last evaluation
                control values and keyword data, or None before evaluation.
        """

        def __init__(self) -> None:
            """Initialize an empty evaluation-call record for the callback regression.

            Returns:
                None: The new model double has call=None.
            """

            self.call = None

        def evaluate(
            self,
            eval_both: bool = False,
            test_part: str | None = None,
            **kwargs: Any
        ) -> dict[str, float]:
            """Record controls and keyword inputs, then return a fixed validation loss.

            Args:
                eval_both (bool): Whether the caller requests both model branches.
                    Defaults to False; recorded without changing the fixed output.
                    Defaults to ``False``.
                test_part (str | None): Optional evaluated branch selector. Defaults to
                    None; recorded without selecting a real network.
                    Defaults to ``None``.
                **kwargs (Any): Evaluation data and Keras options, retained in call for
                    assertions about keyword forwarding.

            Returns:
                dict[str, float]: A new mapping containing loss=0.125.
            """

            self.call = (eval_both, test_part, kwargs)
            return {"loss": 0.125}

    v2_model = V2StyleModel()
    v2_callback = RawNetworkValidationCallback(validation_x, validation_y)
    v2_callback.set_model(v2_model)
    v2_logs = {}
    v2_callback.on_epoch_end(0, v2_logs)
    assert v2_model.call == (
        False,
        None,
        {
            "x": validation_x,
            "y": validation_y,
            "network_name": "raw",
            "verbose": 0,
            "return_dict": True,
        },
    )
    assert v2_logs == {"val_raw_loss": 0.125}

    return {"RawNetworkValidationCallback": "passed"}


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
