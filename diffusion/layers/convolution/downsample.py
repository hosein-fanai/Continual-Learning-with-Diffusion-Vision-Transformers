"""Channels-last image downsampling layers for convolutional networks."""

import tensorflow as tf
from tensorflow.keras import layers

from typing import Any

from common.argument_saver import ArgumentSaverLayer
from common.keras_registry import register_canonical_keras_serializable

from diffusion.layers.convolution.residual_block import _split_inputs


@register_canonical_keras_serializable(package="continual_learning")
class ImageDownsample(ArgumentSaverLayer):
    """Reduce both spatial image dimensions by a configurable integer stride."""

    def __init__(
        self, 
        filters: int | None = None, 
        scaling_method: str = "avg_pooling", 
        kernel_size: int = 3, 
        strides: int = 2, 
        activation_func: str = "linear", 
        **kwargs: Any
    ) -> None:
        """Create a pooling or strided-convolution downsampler.

        Args:
            filters (int | None): Output channels; ``None`` preserves the input
                width.
            scaling_method (str): ``"avg_pooling"``, ``"max_pooling"``, or
                ``"cnn_stride"``.
            kernel_size (int): Positive convolution kernel size.
            strides (int): Positive spatial reduction factor.
            activation_func (str): Keras activation used by learned projections.
            **kwargs (Any): Standard Keras layer options.

        Returns:
            None: Initialization mutates only the new layer instance.
        """

        super().__init__(**kwargs)
        self._check_arguments(locals())
        self._save_init_args(locals())

        self.output_dim = self.filters
        self.projection = None
        # Construct average pooling immediately because it is channel-agnostic.
        if self.scaling_method == "avg_pooling":
            self.scaling_layer = layers.AveragePooling2D(
                pool_size=self.strides, 
                strides=self.strides, 
                padding="same", 
                dtype=self.dtype_policy, 
                name=f"{self.name}/scaling_layer", 
            )
        # Construct max pooling immediately because it is channel-agnostic.
        elif self.scaling_method == "max_pooling":
            self.scaling_layer = layers.MaxPooling2D(
                pool_size=self.strides, 
                strides=self.strides, 
                padding="same", 
                dtype=self.dtype_policy, 
                name=f"{self.name}/scaling_layer", 
            )
        # Defer the learned strided convolution until input channels are known.
        else:
            self.scaling_layer = None

    @staticmethod
    def _check_arguments(local_vars: dict[str, Any]) -> None:
        """Validate the scaling method.

        Args:
            local_vars (dict[str, Any]): Constructor arguments to validate.

        Returns:
            None: Valid arguments complete without a value.
        """

        # Restrict scaling to the three implemented downsampling strategies.
        if local_vars["scaling_method"] not in (
            "avg_pooling", 
            "max_pooling", 
            "cnn_stride", 
        ):
            raise ValueError(
                "scaling_method must be avg_pooling, max_pooling, or cnn_stride."
            )

    def build(self, input_shape: Any) -> None:
        """Resolve an omitted output width and create learned projections.

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
        # Require statically known channels to resolve projections.
        if input_dim is None:
            raise ValueError("ImageDownsample requires known input channels.")

        self.output_dim = int(input_dim) if self.filters is None else self.filters
        # Build the learned strided convolution at the resolved output width.
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
        # Project pooled channels only when the requested width changes.
        elif self.output_dim != int(input_dim):
            self.projection = layers.Conv2D(
                filters=self.output_dim, 
                kernel_size=1, 
                activation=self.activation_func, 
                dtype=self.dtype_policy, 
                name=f"{self.name}/projection", 
            )

        super().build(input_shape)

    def call(
        self, 
        inputs: tf.Tensor | tuple[tf.Tensor, tf.Tensor], 
        training: bool | tf.Tensor | None = None
    ) -> tf.Tensor:
        """Downsample the image component of ``x`` or ``(x, condition)``.

        Args:
            inputs (tf.Tensor | tuple[tf.Tensor, tf.Tensor]): Image tensor or
                image-condition pair; the condition is ignored by this layer.
            training (bool | tf.Tensor | None): Optional Keras training flag.

        Returns:
            tf.Tensor: Spatially downsampled channels-last image features.
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


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
