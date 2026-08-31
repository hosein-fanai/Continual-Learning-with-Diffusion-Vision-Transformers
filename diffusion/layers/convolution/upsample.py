"""Channels-last image upsampling layers for convolutional networks."""

import tensorflow as tf
from tensorflow.keras import layers, models

from typing import Any

from common.argument_saver import ArgumentSaverLayer
from common.keras_registry import register_canonical_keras_serializable

from diffusion.layers.convolution.residual_block import _split_inputs

@register_canonical_keras_serializable(package="continual_learning")
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
        **kwargs: Any
    ) -> None:
        """Create interpolation or learned image upsampling components.

        Args:
            filters (int | None): Output channels; ``None`` preserves input width.
            scaling_method (str): ``"interpolate"``, ``"cnn_interpolate"``, or
                ``"cnn_transpose"``.
            interpolation (str): Keras image-interpolation method.
            kernel_size (int): Positive learned-scaler kernel size.
            strides (int): Positive spatial enlargement factor.
            activation_func (str): Keras activation for learned projections.
            **kwargs (Any): Standard Keras layer options.

        Returns:
            None: Initialization mutates only the new layer instance.
        """

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
    def _check_arguments(local_vars: dict[str, Any]) -> None:
        """Validate scaling selection and positive dimensions.

        Args:
            local_vars (dict[str, Any]): Constructor arguments to validate.

        Returns:
            None: Valid arguments complete without a value.
        """

        # Restrict scaling to the three implemented upsampling strategies.
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
            # Require positive integer kernel and enlargement dimensions.
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        filters = local_vars["filters"]
        # Require a positive integer output width when filters is specified.
        if filters is not None and (
            not isinstance(filters, int) or 
            isinstance(filters, bool) or filters < 1
        ):
            raise ValueError("filters must be None or a positive integer.")

    def build(self, input_shape: Any) -> None:
        """Resolve the output width and create the selected learned scaler.

        Args:
            input_shape (Any): TensorShape-compatible image shape, optionally
                paired with a condition shape.

        Returns:
            None: Keras build state is updated in place.
        """

        image_shape = input_shape[0] if (
            isinstance(input_shape, (tuple, list))
            and len(input_shape) == 2
            and tf.TensorShape(input_shape[0]).rank == 4
        ) else input_shape
        input_dim = tf.TensorShape(image_shape)[-1]
        # Require statically known channels to resolve learned projections.
        if input_dim is None:
            raise ValueError("ImageUpsample requires known input channels.")

        self.output_dim = int(input_dim) if self.filters is None else self.filters
        # Build transposed convolution for directly learned upsampling.
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
        # Follow interpolation with convolution in the hybrid mode.
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
        # Project interpolated channels only when the requested width changes.
        elif self.output_dim != int(input_dim):
            self.projection = layers.Conv2D(
                filters=self.output_dim, 
                kernel_size=1, 
                activation=self.activation_func, 
                dtype=self.dtype_policy, 
                name=f"{self.name}/projection"
            )

        super().build(input_shape)

    def call(
        self, 
        inputs: tf.Tensor | tuple[tf.Tensor, tf.Tensor], 
        training: bool | tf.Tensor | None = None
    ) -> tf.Tensor:
        """Upsample the image component of ``x`` or ``(x, condition)``.

        Args:
            inputs (tf.Tensor | tuple[tf.Tensor, tf.Tensor]): Image tensor or
                image-condition pair; the condition is ignored by this layer.
            training (bool | tf.Tensor | None): Optional Keras training flag.

        Returns:
            tf.Tensor: Spatially enlarged channels-last image features.
        """

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


def run_self_tests() -> dict[str, str]:
    """Exercise every scaling method, projection, condition, and config path.

    Args:
        None.

    Returns:
        dict[str, str]: One success entry after all checks pass.
    """

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


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
