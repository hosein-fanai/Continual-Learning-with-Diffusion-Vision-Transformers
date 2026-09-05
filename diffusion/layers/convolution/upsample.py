"""Channels-last image upsampling layers for convolutional networks.

ImageUpsample enlarges image grids through interpolation, interpolation followed
by convolution, or transposed convolution. Deferred building resolves output
channels and adds learned projection only for the selected mode.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from typing import Any

from common.argument_saver import ArgumentSaverLayer
from common.keras_registry import register_canonical_keras_serializable

from diffusion.layers.convolution.residual_block import _split_inputs

@register_canonical_keras_serializable(package="continual_learning")
class ImageUpsample(ArgumentSaverLayer):
    """Increase both spatial image dimensions by a configurable integer stride.

    Attributes:
        output_dim (int | None): Requested filters, resolved from input channels at build
            time when omitted.
        scaling_layer (tf.keras.layers.Layer | None): Selected interpolator/convolution
            pipeline, completed during build.
        projection (tf.keras.layers.Conv2D | None): Optional 1x1 channel projection in
            interpolation-only mode.

    Inputs:
        Floating channels-last image features [B, H, W, C], or an image/condition
        pair whose condition is ignored. Channels must be known at build time.

    Outputs:
        Floating image features [B, H * strides, W * strides, output_dim] under the layer's compute policy.
        Constructor filters=None preserves C; interpolation otherwise adds
        a 1x1 projection when learned spatial scaling does not already set width.
    """

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
                Defaults to ``None``.
            scaling_method (str): ``"interpolate"``, ``"cnn_interpolate"``, or
                ``"cnn_transpose"``.
                Defaults to ``'interpolate'``.
            interpolation (str): Keras image-interpolation method.
                Defaults to ``'bilinear'``.
            kernel_size (int): Positive learned-scaler kernel size.
                Defaults to ``3``.
            strides (int): Positive spatial enlargement factor.
                Defaults to ``2``.
            activation_func (str): Keras activation for learned projections.
                Defaults to ``'linear'``.
            **kwargs (Any): Standard Keras layer options.

        Returns:
            None: Initialization mutates only the new layer instance.
        """

        super().__init__(**kwargs)
        self._check_arguments(locals())
        self._save_init_args(locals())

        self.output_dim = self.filters
        self.projection = None
        # Interpolation modes need an interpolator; transposed convolution performs its own
        # resize.
        self.interpolator = layers.UpSampling2D(
            size=(self.strides, self.strides), 
            interpolation=self.interpolation, 
            dtype=self.dtype_policy, 
            name=f"{self.name}/interpolator"
        ) if self.scaling_method != "cnn_transpose" else None
        # Pure interpolation is immediately usable; learned scaling is completed during
        # build.
        self.scaling_layer = self.interpolator if self.scaling_method == "interpolate" \
                            else None

    @staticmethod
    def _check_arguments(local_vars: dict[str, Any]) -> None:
        """Validate the scaling selection.

        Args:
            local_vars (dict[str, Any]): Constructor arguments to validate.

        Returns:
            None: Valid arguments complete without a value.

        Raises:
            ValueError: If scaling_method does not name an implemented strategy.
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

    def build(self, input_shape: Any) -> None:
        """Resolve the output width and create the selected learned scaler.

        Args:
            input_shape (Any): TensorShape-compatible image shape, optionally
                paired with a condition shape.

        Returns:
            None: Keras build state is updated in place.
        """

        # Extract image shape from a paired image/condition signature; otherwise use the
        # tensor shape directly.
        image_shape = input_shape[0] if (
            isinstance(input_shape, (tuple, list))
            and len(input_shape) == 2
            and tf.TensorShape(input_shape[0]).rank == 4
        ) else input_shape
        input_dim = tf.TensorShape(image_shape)[-1]
        # Require statically known channels to resolve learned projections.
        if input_dim is None:
            raise ValueError("ImageUpsample requires known input channels.")

        # Preserve input channels when filters is omitted; otherwise use the requested
        # width.
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
                Defaults to ``None``. Keras resolves the surrounding call context; this flag is
                forwarded to child layers.

        Returns:
            tf.Tensor: Spatially enlarged channels-last image features.

        Notes:
            The image input is [B, H, W, C]; the result is [B, H * strides, W * strides, output_dim]. Outputs follow the
            layer compute policy and condition tensors never influence this scaler.
        """

        x, _ = _split_inputs(inputs)

        x = self.scaling_layer(
            x, 
            training=training
        )
        # Apply the optional channel projection; otherwise retain the resized features.
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
    # This invalid case should already have raised: Unknown upsampling methods must fail.
    else:
        raise AssertionError("Unknown upsampling methods must fail.")

    return {"ImageUpsample": "passed"}


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
