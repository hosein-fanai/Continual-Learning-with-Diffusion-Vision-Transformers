"""Shared type aliases for diffusion training and sampling wrappers.

Wrappers orchestrate raw networks from ``diffusion.models.transformer`` rather
than replacing them: they own schedules, noising, losses, Keras steps, EMA, and
sampling while raw networks own architecture and tensor features.
"""

from typing import Literal, TypeAlias


NetworkName: TypeAlias = Literal["ema", "raw"]
"""Selectable prediction copy: exponential-moving-average or trainable raw."""

TrainType: TypeAlias = Literal["cond", "uncond"]
"""Select conditional or null-label branch values for an auxiliary loss."""

ClusteringType: TypeAlias = Literal["uniform", "log_snr"]
"""Progressive timestep partitioning strategy."""
