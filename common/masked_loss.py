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


def run_self_tests() -> dict[str, str]:
    """Run numerical, shape, weighting, and serialization loss tests.

    Both supported modes are checked with equal and wider target widths,
    eager and graph execution, rank-three inputs, inherited sample weights,
    invalid modes/ranks/shapes, custom names, per-example sample weighting, and
    Keras configuration/serialization round trips.

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
    assert mae.loss is tf.math.abs
    assert mae.name == "masked_loss"
    tf.debugging.assert_near(mae(y_true, y_pred), tf.constant(1.25))
    tf.debugging.assert_near(
        mae(y_true, y_pred, sample_weight=tf.constant(0.5)), 
        tf.constant(0.625), 
    )
    tf.debugging.assert_near(
        mae(y_true, y_pred, sample_weight=tf.constant([1.0, 0.0])),
        tf.constant(0.75),
    )

    mse = MaskedLoss(loss_type="mse", name="masked_mse")
    assert mse.loss is tf.math.square
    assert mse.name == "masked_mse"
    tf.debugging.assert_near(mse(y_true, y_pred), tf.constant(2.25))

    equal_true = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    equal_pred = tf.identity(equal_true)
    tf.debugging.assert_near(mae(equal_true, equal_pred), tf.constant(0.0))
    tf.debugging.assert_near(mse(equal_true, equal_pred), tf.constant(0.0))

    rank_three_true = tf.constant([
        [[1., 3., 99.], [5., 7., 99.]],
        [[2., 2., 99.], [2., 2., 99.]],
    ])
    rank_three_pred = tf.zeros((2, 2, 2))
    tf.debugging.assert_near(
        mae.call(rank_three_true, rank_three_pred), tf.constant([4., 2.])
    )
    tf.debugging.assert_near(
        mae(
            rank_three_true,
            rank_three_pred,
            sample_weight=tf.constant([1., 0.]),
        ),
        tf.constant(2.),
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
    assert mae_config["loss_type"] == "mae" and "reduction" in mae_config
    config_clone = MaskedLoss.from_config(mae_config)
    assert config_clone.loss_type == "mae"
    serialized = tf.keras.losses.serialize(mae)
    serialized_clone = tf.keras.losses.deserialize(
        serialized
    )
    assert isinstance(serialized_clone, MaskedLoss)
    assert serialized_clone.loss_type == "mae"
    mse_clone = MaskedLoss.from_config(mse.get_config())
    assert mse_clone.loss_type == "mse"

    for invalid_mode in ("MAE", "unknown", None):
        try:
            MaskedLoss(loss_type=invalid_mode)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported loss modes must raise ValueError.")

    try:
        mae(tf.constant([1.0, 2.0]), tf.constant([1.0, 2.0]))
    except (tf.errors.InvalidArgumentError, ValueError):
        pass
    else:
        raise AssertionError("Rank-one predictions must fail clearly.")

    try:
        mae(tf.constant([[1.0]]), tf.constant([[1.0, 2.0]]))
    except tf.errors.InvalidArgumentError:
        pass
    else:
        raise AssertionError("A target narrower than its prediction must fail.")
    try:
        mae(tf.ones((1, 0)), tf.ones((1, 2)))
    except (tf.errors.InvalidArgumentError, ValueError):
        pass
    else:
        raise AssertionError("A zero-width target cannot broadcast to predictions.")
    try:
        mae(tf.ones((1, 0)), tf.ones((1, 0)))
    except tf.errors.InvalidArgumentError:
        pass
    else:
        raise AssertionError("A zero-width prediction has no defined mean loss.")
    try:
        mae.call(tf.ones((1, 3)), tf.ones((2, 2)))
    except tf.errors.InvalidArgumentError:
        pass
    else:
        raise AssertionError("Mismatched batch dimensions must not broadcast.")
    try:
        mae.call(tf.ones((1, 1, 3)), tf.ones((1, 2)))
    except tf.errors.InvalidArgumentError:
        pass
    else:
        raise AssertionError("Mismatched target and prediction ranks must fail.")
    try:
        mae.call(tf.ones((2, 1, 3)), tf.ones((2, 2, 2)))
    except tf.errors.InvalidArgumentError:
        pass
    else:
        raise AssertionError(
            "Mismatched intermediate dimensions must not broadcast."
        )

    tf.debugging.assert_near(
        mae(tf.constant([[1, 2]], tf.int32), tf.constant([[0., 0.]])),
        tf.constant(1.5),
    )

    return {"MaskedLoss": "passed"}


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
