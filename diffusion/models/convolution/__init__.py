"""Public convolutional diffusion model exports and shared tensor aliases.

UNet and UNetClassifier are imported lazily through __getattr__ so raw models,
wrappers, and layer registries can share contracts without import cycles.
UNetInputs names the image/timestep/label triple; UNetFullOutput describes the
five-part noise, condition, feature, regularizer, and latent-statistics return.
DTypeLike admits a dtype string, TensorFlow dtype, or Keras dtype policy.
"""

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
    list[tuple[tf.Tensor, tf.Tensor]],
]
"""Five-item full return contract shared with the diffusion wrapper."""


__all__ = (
    "UNet", 
    "UNetClassifier", 
    "UNetInputs", 
    "UNetFullOutput", 
    "DTypeLike", 
)


def __getattr__(name: str) -> type:
    """Load a concrete convolutional model without creating import cycles.

    Args:
        name (str): Public class name, ``"UNet"`` or ``"UNetClassifier"``.

    Returns:
        type: Requested model class.

    Raises:
        AttributeError: If ``name`` is not a lazily exported model.
    """

    # Import UNet only when callers request that public symbol.
    if name == "UNet":
        from .unet import UNet

        return UNet

    # Import UNetClassifier only when callers request that public symbol.
    if name == "UNetClassifier":
        from .unet_classifier import UNetClassifier

        return UNetClassifier

    raise AttributeError(name)
