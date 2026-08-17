"""Public type contracts shared by diffusion embedding layers.

``PosEmbedType`` enumerates positional-table construction strategies, while
``MergeType`` controls whether a positional table is added to or concatenated
with content embeddings.
"""

from typing import TypeAlias, Literal


PosEmbedType: TypeAlias = Literal[
    "new_weight", 
    "1d_sincos", 
    "1d_interpolate", 
    "1d_learned_interpolate", 
    "2d_sincos", 
    "2d_interpolate", 
    "2d_learned_interpolate", 
]
"""Supported learned, sinusoidal, and interpolated positional table modes."""

MergeType: TypeAlias = Literal["concat", "add"]
"""Supported positional/content merge operations."""
