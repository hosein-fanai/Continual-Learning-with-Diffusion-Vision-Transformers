import tensorflow as tf

from typing import TypeAlias


UNetInputs = tuple[tf.Tensor, tf.Tensor, tf.Tensor]
DTypeLike: TypeAlias = (
    str | tf.dtypes.DType | tf.keras.mixed_precision.Policy
)
UNetFullOutput = tuple[
    tf.Tensor, 
    tf.Tensor, 
    list[tf.Tensor | None], 
    list[tf.Tensor | None], 
    tuple[tf.Tensor | None, tf.Tensor | None], 
]
