from typing import Literal, TypeAlias


NetworkName: TypeAlias = Literal["ema", "raw"]

TrainType: TypeAlias = Literal["cond", "uncond"]

ClusteringType: TypeAlias = Literal["uniform", "log_snr"]
