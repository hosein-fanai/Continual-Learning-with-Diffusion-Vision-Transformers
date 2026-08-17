"""Channels-last image upsampling layers for convolutional networks."""

import tensorflow as tf
from tensorflow.keras import layers, models

from common.argument_saver import ArgumentSaverLayer

from diffusion.layers.convolution.residual_block import _split_inputs


class ImageUpsample(ArgumentSaverLayer):
    """Increase both spatial image dimensions by a configurable integer stride."""

    def __init__(
        self, 
        filters: int | None = None, 
        scaling_method: str = "interpolate", 
        interpolation: str = "bilinear", 
        kernel_size: int = 3, 
        strides: int = 2, 
        activation_func: str = "linear", 
        **kwargs,
    ):
        """Create interpolation or learned image upsampling components."""

        super().__init__(**kwargs)
        self._check_arguments(locals())
        self._save_init_args(locals())

        self.output_dim = self.filters
        self.projection = None
        self.interpolator = layers.UpSampling2D(
            size=(self.strides, self.strides), 
            interpolation=self.interpolation, 
            dtype=self.dtype_policy, 
            name=f"{self.name}/interpolator"
        ) if self.scaling_method != "cnn_transpose" else None
        self.scaling_layer = self.interpolator if self.scaling_method == "interpolate" \
                            else None

    @staticmethod
    def _check_arguments(local_vars: dict) -> None:
        """Validate scaling selection and positive dimensions."""

        if local_vars["scaling_method"] not in (
            "interpolate", 
            "cnn_interpolate", 
            "cnn_transpose"
        ):
            raise ValueError(
                "scaling_method must be interpolate, cnn_interpolate, or "
                "cnn_transpose."
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
        """Resolve the output width and create the selected learned scaler."""

        image_shape = input_shape[0] if (
            isinstance(input_shape, (tuple, list))
            and len(input_shape) == 2
            and tf.TensorShape(input_shape[0]).rank == 4
        ) else input_shape
        input_dim = tf.TensorShape(image_shape)[-1]
        if input_dim is None:
            raise ValueError("ImageUpsample requires known input channels.")

        self.output_dim = int(input_dim) if self.filters is None else self.filters
        if self.scaling_method == "cnn_transpose":
            self.scaling_layer = layers.Conv2DTranspose(
                filters=self.output_dim, 
                kernel_size=self.kernel_size, 
                strides=self.strides, 
                padding="same", 
                activation=self.activation_func, 
                dtype=self.dtype_policy, 
                name=f"{self.name}/scaling_layer"
            )
        elif self.scaling_method == "cnn_interpolate":
            self.scaling_layer = models.Sequential([
                self.interpolator, 
                layers.Conv2D(
                    filters=self.output_dim, 
                    kernel_size=self.kernel_size, 
                    padding="same", 
                    activation=self.activation_func, 
                    dtype=self.dtype_policy, 
                    name=f"{self.name}/convolution"
                ),
            ], name=f"{self.name}/scaling_layer")
        elif self.output_dim != int(input_dim):
            self.projection = layers.Conv2D(
                filters=self.output_dim, 
                kernel_size=1, 
                activation=self.activation_func, 
                dtype=self.dtype_policy, 
                name=f"{self.name}/projection"
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Upsample the image component of ``x`` or ``(x, condition)``."""

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
    )(ImageUpsample)


def run_self_tests() -> dict[str, str]:
    """Exercise every scaling method, projection, condition, and config path."""

    x = tf.random.normal((2, 4, 5, 3))
    condition = tf.random.normal((2, 4))
    methods = ("interpolate", "cnn_interpolate", "cnn_transpose")
    for method in methods:
        layer = ImageUpsample(
            filters=5, 
            scaling_method=method, 
            name=f"up_{method}"
        )
        output = layer((x, condition))
        assert output.shape == (2, 8, 10, 5)
        assert layer.output_dim == 5
        clone = ImageUpsample.from_config(layer.get_config())
        assert clone(x).shape == output.shape

    inferred = ImageUpsample(filters=None)
    assert inferred(x).shape == (2, 8, 10, 3)
    assert inferred.output_dim == 3

    try:
        ImageUpsample(scaling_method="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown upsampling methods must fail.")

    return {"ImageUpsample": "passed"}


if __name__ == "__main__":
    print(run_self_tests())
