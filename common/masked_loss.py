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


def run_self_tests() -> dict[str, str]:
    """Run numerical, shape, weighting, and serialization loss tests.

    Both supported modes are checked with equal and wider target widths,
    eager and graph execution, rank-three inputs, inherited sample weights,
    invalid modes/ranks/shapes, custom names, and the current serialization
    limitations: Keras emits an unsupported ``reduction`` key and does not
    persist a nondefault ``loss_type``.

    Args:
        None.

    Returns:
        dict[str, str]: ``{"MaskedLoss": "passed"}`` after all assertions
        succeed.
    """

    y_true = tf.constant([
        [1.0, 3.0, 99.0], 
        [2.0, 4.0, -99.0]
    ], dtype=tf.float32)
    y_pred = tf.constant([
        [0.0, 1.0], 
        [2.0, 2.0]
    ], dtype=tf.float32)

    mae = MaskedLoss()
    assert isinstance(mae.loss, losses.MeanAbsoluteError)
    assert mae.name == "masked_loss"
    tf.debugging.assert_near(mae(y_true, y_pred), tf.constant(1.25))
    tf.debugging.assert_near(
        mae(y_true, y_pred, sample_weight=tf.constant(0.5)), 
        tf.constant(0.625), 
    )
    try:
        mae(y_true, y_pred, sample_weight=tf.constant([1.0, 0.0]))
    except tf.errors.InvalidArgumentError:
        pass
    else:
        raise AssertionError(
            "Per-example weights are incompatible because call() returns an "
            "already reduced scalar from its nested Keras loss."
        )

    mse = MaskedLoss(loss_type="mse", name="masked_mse")
    assert isinstance(mse.loss, losses.MeanSquaredError)
    assert mse.name == "masked_mse"
    tf.debugging.assert_near(mse(y_true, y_pred), tf.constant(2.25))

    equal_true = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    equal_pred = tf.identity(equal_true)
    tf.debugging.assert_near(mae(equal_true, equal_pred), tf.constant(0.0))
    tf.debugging.assert_near(mse(equal_true, equal_pred), tf.constant(0.0))

    rank_three_true = tf.reshape(tf.range(6, dtype=tf.float32), (1, 2, 3))
    rank_three_pred = rank_three_true[..., :2]
    tf.debugging.assert_near(
        mae(rank_three_true, rank_three_pred), tf.constant(0.0)
    )


    @tf.function
    def graph_loss(
        target: tf.Tensor,
        prediction: tf.Tensor,
    ) -> tf.Tensor:
        """Evaluate the MAE test instance in a TensorFlow graph.

        Args:
            target (tf.Tensor): Rank-two target tensor.
            prediction (tf.Tensor): Rank-two prediction tensor.

        Returns:
            tf.Tensor: Scalar masked MAE.
        """

        return mae(target, prediction)


    tf.debugging.assert_near(graph_loss(y_true, y_pred), tf.constant(1.25))

    mae_config = mae.get_config()
    assert mae_config["name"] == "masked_loss"
    assert "reduction" in mae_config and "loss_type" not in mae_config
    try:
        MaskedLoss.from_config(mae_config)
    except TypeError:
        pass
    else:
        raise AssertionError(
            "Keras's emitted reduction key is not accepted by this constructor."
        )
    serialized = tf.keras.losses.serialize(mae)
    try:
        tf.keras.losses.deserialize(
            serialized, 
            custom_objects={"MaskedLoss": MaskedLoss}
        )
    except TypeError:
        pass
    else:
        raise AssertionError("The current Keras loss round trip must expose its limitation.")
    sanitized_clone = MaskedLoss(name=mse.get_config()["name"])
    assert isinstance(sanitized_clone.loss, losses.MeanAbsoluteError), (
        "Even after removing the unsupported reduction key, loss_type is "
        "absent and reconstruction therefore uses the MAE default."
    )

    for invalid_mode in ("MAE", "unknown", None):
        try:
            MaskedLoss(loss_type=invalid_mode)
        except TypeError:
            pass
        else:
            raise AssertionError("Unsupported loss modes must raise TypeError.")

    try:
        mae(tf.constant([1.0, 2.0]), tf.constant([1.0, 2.0]))
    except (tf.errors.InvalidArgumentError, IndexError):
        pass
    else:
        raise AssertionError("Rank-one predictions must fail at axis-one lookup.")

    tf.debugging.assert_near(
        mae(tf.constant([[1.0]]), tf.constant([[1.0, 2.0]])), 
        tf.constant(0.5), 
    )
    try:
        mae(tf.ones((1, 0)), tf.ones((1, 2)))
    except (tf.errors.InvalidArgumentError, ValueError):
        pass
    else:
        raise AssertionError("A zero-width target cannot broadcast to predictions.")

    return {"MaskedLoss": "passed"}


if __name__ == "__main__":
    print(run_self_tests())
