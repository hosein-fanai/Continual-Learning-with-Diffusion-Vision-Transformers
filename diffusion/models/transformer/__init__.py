"""Shared public type aliases for diffusion-transformer network APIs.

The concrete raw networks live beside this module.  They define tensor
transformations only; ``diffusion.models.wrapper`` contains their training,
evaluation, EMA, scheduling, and sampling orchestration.
"""

import tensorflow as tf

from typing import Literal, TypeAlias


CondType: TypeAlias = Literal[
    "time_label", 
    "time", 
    "label"
]
"""Condition selection: combined time/label, time only, or label only."""

TokenType: TypeAlias = Literal[
    "new_weight", 
    "time_label", 
    "time", 
    "label"
]
"""Class-token source: learned weight or one of the condition selections."""

IdsType: TypeAlias = list[int | None] | tuple[int | None, ...]
"""Depth-ID sequence; ``None`` expands all IDs and negatives are relative."""

IdsDictType: TypeAlias = dict[int, IdsType]
"""Mapping from a target depth to source/component depth IDs."""


def select_first_token(x: tf.Tensor) -> tf.Tensor:
    """Select the class token from a token sequence.

    Args:
        x (tf.Tensor): Float token tensor of shape ``[B, tokens, features]``.

    Returns:
        tf.Tensor: First-token features of shape ``[B, features]``.
    """

    return x[:, 0, :]
