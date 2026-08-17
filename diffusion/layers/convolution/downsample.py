"""Channels-last image downsampling layers for convolutional networks."""

import tensorflow as tf
from tensorflow.keras import layers

from common.argument_saver import ArgumentSaverLayer
from diffusion.layers.convolution.residual_block import _split_inputs


class ImageDownsample(ArgumentSaverLayer):
    """Reduce both spatial image dimensions by a configurable integer stride."""

    def __init__(
        self, 
        filters: int | None = None, 
        scaling_method: str = "avg_pooling", 
        kernel_size: int = 3, 
        strides: int = 2, 
        activation_func: str = "linear", 
        **kwargs,
    ):
        """Create a pooling or strided-convolution downsampler."""

        super().__init__(**kwargs)
        self._check_arguments(locals())
        self._save_init_args(locals())

        self.output_dim = self.filters
        self.projection = None
        if self.scaling_method == "avg_pooling":
            self.scaling_layer = layers.AveragePooling2D(
                pool_size=self.strides, 
                strides=self.strides, 
                padding="same", 
                dtype=self.dtype_policy, 
                name=f"{self.name}/scaling_layer", 
            )
        elif self.scaling_method == "max_pooling":
            self.scaling_layer = layers.MaxPooling2D(
                pool_size=self.strides, 
                strides=self.strides, 
                padding="same", 
                dtype=self.dtype_policy, 
                name=f"{self.name}/scaling_layer", 
            )
        else:
            self.scaling_layer = None

    @staticmethod
    def _check_arguments(local_vars: dict) -> None:
        """Validate the scaling method and dimensions."""

        if local_vars["scaling_method"] not in (
            "avg_pooling", 
            "max_pooling", 
            "cnn_stride", 
        ):
            raise ValueError(
                "scaling_method must be avg_pooling, max_pooling, or cnn_stride."
            )
        for name in ("kernel_size", "strides"):
            value = local_vars[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        filters = local_vars["filters"]
        if filters is not None and (
            not isinstance(filters, int) or 
            isinstance(filters, bool) or filters < 1
        ):
            raise ValueError("filters must be None or a positive integer.")

    def build(self, input_shape) -> None:
        """Resolve an omitted output width and create learned projections."""

        image_shape = input_shape[0] if (
            isinstance(input_shape, (tuple, list))
            and len(input_shape) == 2
            and tf.TensorShape(input_shape[0]).rank == 4
        ) else input_shape
        input_dim = tf.TensorShape(image_shape)[-1]
        if input_dim is None:
            raise ValueError("ImageDownsample requires known input channels.")

        self.output_dim = int(input_dim) if self.filters is None else self.filters
        if self.scaling_method == "cnn_stride":
            self.scaling_layer = layers.Conv2D(
                filters=self.output_dim, 
                kernel_size=self.kernel_size, 
                strides=self.strides, 
                padding="same", 
                activation=self.activation_func, 
                dtype=self.dtype_policy, 
                name=f"{self.name}/scaling_layer", 
            )
        elif self.output_dim != int(input_dim):
            self.projection = layers.Conv2D(
                filters=self.output_dim, 
                kernel_size=1, 
                activation=self.activation_func, 
                dtype=self.dtype_policy, 
                name=f"{self.name}/projection", 
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Downsample the image component of ``x`` or ``(x, condition)``."""

        x, _ = _split_inputs(inputs)

        x = self.scaling_layer(
            x, 
            training=training
        )
        x = self.projection(
            x, 
            training=training
        ) if self.projection is not None else x

        return x


if __name__ != "__main__":
    tf.keras.utils.register_keras_serializable(
        package="continual_learning"
    )(ImageDownsample)


def run_self_tests() -> dict[str, str]:
    """Exercise every scaling method, projection, condition, and config path."""

    x = tf.random.normal((2, 9, 7, 3))
    condition = tf.random.normal((2, 4))
    methods = ("avg_pooling", "max_pooling", "cnn_stride")
    for method in methods:
        layer = ImageDownsample(
            filters=5, 
            scaling_method=method, 
            name=f"down_{method}", 
        )
        output = layer((x, condition))
        assert output.shape == (2, 5, 4, 5)
        assert layer.output_dim == 5
        clone = ImageDownsample.from_config(layer.get_config())
        assert clone(x).shape == output.shape

    inferred = ImageDownsample(filters=None)
    assert inferred(x).shape == (2, 5, 4, 3)
    assert inferred.output_dim == 3

    try:
        ImageDownsample(scaling_method="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown downsampling methods must fail.")

    return {"ImageDownsample": "passed"}


if __name__ == "__main__":
    print(run_self_tests())
