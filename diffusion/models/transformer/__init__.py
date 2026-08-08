from typing import Literal, TypeAlias


CondType: TypeAlias = Literal[
    "time_label", 
    "time", 
    "label"
]

TokenType: TypeAlias = Literal[
    "new_weight", 
    "time_label", 
    "time", 
    "label"
]

IdsType: TypeAlias = list[int | None]

IdsDictType: TypeAlias = dict[IdsType]
