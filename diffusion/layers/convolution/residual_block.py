"""Condition-aware residual convolution blocks for image feature maps."""

import tensorflow as tf
from tensorflow.keras import layers

from common.argument_saver import ArgumentSaverLayer


def _split_inputs(inputs):
    """Return an image tensor and its optional rank-two condition."""

    if isinstance(inputs, (tuple, list)):
        if len(inputs) != 2:
            raise ValueError("Convolution layer inputs must be x or (x, condition).")

        return inputs[0], inputs[1]

    return inputs, None


class ResidualConvBlock(ArgumentSaverLayer):
    """Apply two spatial convolutions with residual and condition projections.

    Inputs may be an image feature map ``x`` or ``(x, condition)``. ``x`` is a
    channels-last rank-four tensor. A supplied condition is rank two and is
    projected to ``filters`` channels before it is broadcast over the image.
    """

    def __init__(
        self, 
        filters: int, 
        condition_dim: int | None = None, 
        kernel_size: int = 3, 
        activation_func: str = "swish", 
        use_batch_norm: bool = True, 
        dropout_rate: float = 0.0, 
        zero_init: bool = False, 
        **kwargs
    ):
        """Create the convolution, residual, and optional condition paths."""

        super().__init__(**kwargs)
        self._check_arguments(locals())
        self._save_init_args(locals())

        self.output_dim = self.filters
        self.normalization = layers.BatchNormalization(
            center=False, 
            scale=False, 
            dtype=self.dtype_policy, 
            name=f"{self.name}/normalization", 
        ) if self.use_batch_norm else None
        self.first_convolution = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="same",
            activation=self.activation_func,
            dtype=self.dtype_policy,
            name=f"{self.name}/first_convolution",
        )
        self.dropout = layers.SpatialDropout2D(
            rate=self.dropout_rate, 
            dtype=self.dtype_policy, 
            name=f"{self.name}/dropout"
        ) if self.dropout_rate > 0.0 else None
        self.second_convolution = layers.Conv2D(
            filters=self.filters, 
            kernel_size=self.kernel_size, 
            padding="same", 
            kernel_initializer="zeros" if self.zero_init else "glorot_uniform", 
            bias_initializer="zeros", 
            dtype=self.dtype_policy, 
            name=f"{self.name}/second_convolution"
        )
        self.condition_projector = layers.Dense(
            self.filters, 
            dtype=self.dtype_policy, 
            name=f"{self.name}/condition_projector"
        )
        self.residual_projector = None

    @staticmethod
    def _check_arguments(local_vars: dict) -> None:
        """Validate dimensions and probabilities used by the block."""

        for name in ("filters", "kernel_size"):
            value = local_vars[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        condition_dim = local_vars["condition_dim"]
        if condition_dim is not None and (
            not isinstance(condition_dim, int)
            or isinstance(condition_dim, bool)
            or condition_dim < 1
        ):
            raise ValueError("condition_dim must be None or a positive integer.")

        if not 0.0 <= local_vars["dropout_rate"] < 1.0:
            raise ValueError("dropout_rate must be in the range [0, 1).")

    def build(self, input_shape) -> None:
        """Create the residual projection when input channels differ."""

        image_shape = input_shape[0] if (
            isinstance(input_shape, (tuple, list))
            and len(input_shape) == 2
            and tf.TensorShape(input_shape[0]).rank == 4
        ) else input_shape
        image_shape = tf.TensorShape(image_shape)
        input_dim = image_shape[-1]
        if input_dim is None:
            raise ValueError("ResidualConvBlock requires known input channels.")

        if int(input_dim) != self.filters:
            self.residual_projector = layers.Conv2D(
                filters=self.filters, 
                kernel_size=1, 
                dtype=self.dtype_policy, 
                name=f"{self.name}/residual_projector"
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Transform an image feature map and add its residual representation."""

        x, condition = _split_inputs(inputs)
        residual = self.residual_projector(
            x, 
            training=training
        ) if self.residual_projector is not None else x

        h = self.normalization(
            x,
            training=training,
        ) if self.normalization is not None else x
        h = self.first_convolution(
            h, 
            training=training
        )

        if condition is not None:
            condition = tf.cast(condition, h.dtype)
            condition = self.condition_projector(
                condition, 
                training=training, 
            )
            condition = tf.cast(condition, h.dtype)
            condition = condition[:, None, None, :]
            h = h + condition

        h = self.dropout(
            h, 
            training=training
        ) if self.dropout is not None else h
        h = self.second_convolution(
            h, 
            training=training
        )

        return residual + h


class ResidualConvStack(ArgumentSaverLayer):
    """Run a fixed number of `ResidualConvBlock` objects in sequence."""

    def __init__(
        self, 
        filters: int, 
        depth: int = 1, 
        condition_dim: int | None = None, 
        kernel_size: int = 3, 
        activation_func: str = "swish", 
        use_batch_norm: bool = True, 
        dropout_rate: float = 0.0, 
        zero_init: bool = False, 
        **kwargs
    ):
        """Create ``depth`` equally configured residual blocks."""

        super().__init__(**kwargs)

        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
            raise ValueError("depth must be a positive integer.")
        ResidualConvBlock._check_arguments(locals())
        self._save_init_args(locals())

        self.output_dim = self.filters
        self.blocks = [
            ResidualConvBlock(
                filters=self.filters, 
                condition_dim=self.condition_dim, 
                kernel_size=self.kernel_size, 
                activation_func=self.activation_func, 
                use_batch_norm=self.use_batch_norm, 
                dropout_rate=self.dropout_rate, 
                zero_init=self.zero_init and block_id == self.depth - 1, 
                dtype=self.dtype_policy, 
                name=f"{self.name}/block_{block_id + 1}"
            )
            for block_id in range(self.depth)
        ]

    def call(self, inputs, training=None):
        """Apply every residual block while reusing the optional condition."""

        x, condition = _split_inputs(inputs)

        for block in self.blocks:
            x = block(
                (x, condition) if condition is not None else x, 
                training=training
            )

        return x


if __name__ != "__main__":
    tf.keras.utils.register_keras_serializable(
        package="continual_learning"
    )(ResidualConvBlock)
    tf.keras.utils.register_keras_serializable(
        package="continual_learning"
    )(ResidualConvStack)


def run_self_tests() -> dict[str, str]:
    """Run small shape, gradient, conditioning, and config checks."""

    tf.random.set_seed(101)
    x = tf.random.normal((2, 8, 8, 3))
    condition = tf.random.normal((2, 5))

    block = ResidualConvBlock(
        filters=4, 
        condition_dim=5, 
        dropout_rate=0.1, 
        name="residual_probe", 
    )
    y = block(
        (x, condition), 
        training=True
    )
    assert y.shape == (2, 8, 8, 4)
    assert len(block.trainable_variables) > 0
    assert ResidualConvBlock.from_config(block.get_config())(
        (x, condition), 
        training=False
    ).shape == y.shape

    stack = ResidualConvStack(
        filters=6, 
        depth=2, 
        use_batch_norm=False, 
        name="stack_probe"
    )
    with tf.GradientTape() as tape:
        tape.watch(x)
        stack_output = stack(x, training=True)
        loss = tf.reduce_sum(stack_output)
    assert stack_output.shape == (2, 8, 8, 6)
    assert tape.gradient(loss, x) is not None

    for invalid_kwargs in (
        {"filters": 0}, 
        {"filters": 4, "kernel_size": 0}, 
        {"filters": 4, "dropout_rate": 1.0}, 
    ):
        try:
            ResidualConvBlock(**invalid_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid residual block arguments must fail.")

    return {
        "ResidualConvBlock": "passed", 
        "ResidualConvStack": "passed"
    }


if __name__ == "__main__":
    print(run_self_tests())
