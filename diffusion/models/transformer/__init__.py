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


def select_second_token(x: tf.Tensor) -> tf.Tensor:
    """Select the distillation token after a leading class token.

    Args:
        x (tf.Tensor): Float token tensor of shape ``[B, tokens, features]``.

    Returns:
        tf.Tensor: Second-token features of shape ``[B, features]``.
    """

    return x[:, 1, :]


def remove_first_token(x: tf.Tensor) -> tf.Tensor:
    """Remove a leading distillation token before global average pooling.

    Args:
        x (tf.Tensor): Float token tensor of shape ``[B, tokens, features]``.

    Returns:
        tf.Tensor: All tokens after the first, preserving batch and features.
    """

    return x[:, 1:, :]


def remove_second_token(x: tf.Tensor) -> tf.Tensor:
    """Remove a distillation token while retaining a leading class token.

    Args:
        x (tf.Tensor): Float token tensor ordered as class, distillation, then
            patch tokens.

    Returns:
        tf.Tensor: The class token followed by every patch token.
    """

    return tf.concat((x[:, :1, :], x[:, 2:, :]), axis=1)
