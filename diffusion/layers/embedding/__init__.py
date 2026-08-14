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

MergeType: TypeAlias = Literal["concat", "add"]
