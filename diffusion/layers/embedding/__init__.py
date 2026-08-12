from typing import TypeAlias, Literal


PosEmbedType: TypeAlias = Literal[
    "new_weight", 
    "interpolate", 
    "learned_interpolate", 
    "2d_sincos", 
    "1d_sincos", 
]

MergeType: TypeAlias = Literal["concat", "add"]
