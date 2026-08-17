"""Public tensor and dtype contracts for convolutional diffusion models."""

import tensorflow as tf

from typing import TypeAlias


UNetInputs = tuple[tf.Tensor, tf.Tensor, tf.Tensor]
"""``(noisy_images, timesteps, labels)`` tensor tuple accepted by ``UNet``."""

DTypeLike: TypeAlias = (
    str | tf.dtypes.DType | tf.keras.mixed_precision.Policy
)
"""Keras-compatible string, TensorFlow dtype, or mixed-precision policy."""

UNetFullOutput = tuple[
    tf.Tensor, 
    tf.Tensor, 
    list[tf.Tensor | None], 
    list[tf.Tensor | None], 
    tuple[tf.Tensor | None, tf.Tensor | None], 
]
"""Five-item full return contract shared with the diffusion wrapper."""
