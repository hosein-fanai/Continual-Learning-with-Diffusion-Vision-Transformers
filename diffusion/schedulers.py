"""Diffusion noise schedules.

This module implements a practical set of diffusion *noise schedules* used in
image generation and editing papers / libraries.

Covered schedules:
- linear (DDPM)
- scaled_linear (Diffusers variant)
- squaredcos_cap_v2 (cosine schedule, Nichol & Dhariwal style)
- clipped_cosine
- sigmoid
- quadratic
- ve (variance-exploding / sigma-space schedule)
- karras (Karras rho schedule in sigma-space)
- sub_vp (flow-matching / rectified-flow style)
- logistic
- shifted variants via log-SNR shifting

The module exposes a common API:
- betas = generate_betas(...)
- sigmas = generate_sigmas(...)
- alpha_bar = betas_to_alpha_bar(betas)
- timesteps = schedule_timesteps(...)

Notes:
- For discrete DDPM-style schedules, betas are typically used for training.
- For sigma-space samplers, sigmas are typically used for inference.
- Some schedules are naturally defined in continuous time; here we provide a
  numerically useful discretization for a chosen number of steps.
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass

from enum import Enum

from math import pi

from typing import Iterable, Literal, TypeAlias


class ScheduleKind(str, Enum):
    LINEAR = "linear"
    SCALED_LINEAR = "scaled_linear"
    COSINE = "squaredcos_cap_v2"
    CLIPPED_COSINE = "clipped_cosine"
    SIGMOID = "sigmoid"
    QUADRATIC = "quadratic"
    VE = "ve"
    KARRAS = "karras"
    SUB_VP = "sub_vp"
    LOGISTIC = "logistic"


@dataclass(frozen=True)
class ScheduleConfig:
    kind: ScheduleKind = ScheduleKind.LINEAR
    num_steps: int = 1000

    # Generic bounds used by many schedules.
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # Cosine schedule offset.
    cosine_s: float = 0.008

    # Clipped cosine schedule bounds on sqrt(alpha_bar).
    min_sqrt_alpha_bar: float = 0.02
    max_sqrt_alpha_bar: float = 0.95

    # Sigma-space schedules.
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0

    # Optional SNR shift. Positive values make the schedule noisier earlier.
    snr_shift: float = 0.0

    # Logistic schedule sharpness (larger -> steeper transition).
    logistic_k: float = 10.0

    # Safety clamp.
    clip_min: float = 1e-8
    clip_max: float = 0.999


def _as_float64(x: np.ndarray | Iterable[float]) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def betas_to_alpha_bar(betas: np.ndarray) -> np.ndarray:
    """Convert per-step betas to cumulative alpha products."""
    betas = _as_float64(betas)
    if np.any(betas <= 0) or np.any(betas >= 1):
        raise ValueError("betas must lie strictly in (0, 1).")
    alphas = 1.0 - betas
    return np.cumprod(alphas)


def alpha_bar_to_betas(
    alpha_bar: np.ndarray,
    clip_min: float = 1e-8,
    clip_max: float = 0.999,
) -> np.ndarray:
    """Convert a desired alpha_bar trajectory to per-step betas.

    beta[0] is chosen so that cumprod(1 - beta)[0] == alpha_bar[0].
    For later steps:
        beta[t] = 1 - alpha_bar[t] / alpha_bar[t - 1]
    """
    alpha_bar = np.clip(_as_float64(alpha_bar), 1e-12, 1.0)

    if np.any(np.diff(alpha_bar) > 1e-12):
        raise ValueError("alpha_bar must be monotonically non-increasing.")

    betas = np.empty_like(alpha_bar)
    betas[0] = 1.0 - alpha_bar[0]
    betas[1:] = 1.0 - alpha_bar[1:] / alpha_bar[:-1]

    return np.clip(betas, clip_min, clip_max)


def alpha_bar_to_sigmas(alpha_bar: np.ndarray) -> np.ndarray:
    """Convert alpha_bar to sigma values: sigma = sqrt(1 - alpha_bar)."""
    alpha_bar = np.clip(_as_float64(alpha_bar), 0.0, 1.0)
    return np.sqrt(np.maximum(1.0 - alpha_bar, 0.0))


def sigmas_to_betas(sigmas: np.ndarray) -> np.ndarray:
    """Convert a sigma schedule to betas via alpha_bar = 1 - sigma^2.

    This is a convenience mapping for schedules where sigma is interpreted as
    the corruption standard deviation under the VP parameterization.
    """
    sigmas = np.clip(_as_float64(sigmas), 0.0, 1.0 - 1e-12)
    alpha_bar = 1.0 - sigmas**2
    alpha_bar = np.clip(alpha_bar, 1e-12, 1.0)
    betas = alpha_bar_to_betas(alpha_bar)
    return np.clip(betas, 1e-8, 0.999)


def _apply_snr_shift(alpha_bar: np.ndarray, snr_shift: float) -> np.ndarray:
    """Shift a schedule in log-SNR space.

    If SNR = alpha^2 / sigma^2 = alpha_bar / (1 - alpha_bar), then
    logSNR' = logSNR + snr_shift.
    """
    if snr_shift == 0.0:
        return alpha_bar

    alpha_bar = np.clip(_as_float64(alpha_bar), 1e-12, 1.0 - 1e-12)
    snr = alpha_bar / np.maximum(1.0 - alpha_bar, 1e-12)
    shifted = snr * np.exp(snr_shift)
    return shifted / (1.0 + shifted)


def _cosine_alpha_bar(t: np.ndarray, s: float = 0.008) -> np.ndarray:
    f = np.cos(((t + s) / (1.0 + s)) * (pi / 2.0)) ** 2
    return f / f[0]


def _sigmoid01(t: np.ndarray, k: float = 10.0) -> np.ndarray:
    """Smooth map from [0,1] -> [0,1]."""
    x = np.clip(t, 0.0, 1.0)
    z = k * (x - 0.5)
    return 1.0 / (1.0 + np.exp(-z))


def generate_betas(config: ScheduleConfig) -> np.ndarray:
    """Generate a discrete beta schedule.

    For sigma-space schedules (VE/Karras/Sub-VP/Logistic), the result is a
    beta-equivalent discretization suitable for VP-style code paths.
    """
    n = int(config.num_steps)
    if n < 2:
        raise ValueError("num_steps must be >= 2")

    t = np.linspace(0.0, 1.0, n, dtype=np.float64)

    if config.kind == ScheduleKind.LINEAR:
        betas = np.linspace(config.beta_start, config.beta_end, n, dtype=np.float64)
        return np.clip(betas, config.clip_min, config.clip_max)

    if config.kind == ScheduleKind.SCALED_LINEAR:
        # Common Diffusers-style schedule: linearly interpolate sqrt(beta),
        # then square to get a gentler early ramp.
        betas = np.linspace(np.sqrt(config.beta_start), np.sqrt(config.beta_end), n) ** 2
        return np.clip(betas, config.clip_min, config.clip_max)

    if config.kind == ScheduleKind.QUADRATIC:
        betas = config.beta_start + (config.beta_end - config.beta_start) * (t**2)
        return np.clip(betas, config.clip_min, config.clip_max)

    if config.kind == ScheduleKind.COSINE:
        alpha_bar = _cosine_alpha_bar(t, s=config.cosine_s)
        alpha_bar = _apply_snr_shift(alpha_bar, config.snr_shift)
        alpha_bar = np.clip(alpha_bar, 1e-12, 1.0)
        return alpha_bar_to_betas(
            alpha_bar,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
        )

    if config.kind == ScheduleKind.CLIPPED_COSINE:
        # Matches:
        #   min_sqrt_alpha_bar = 0.02
        #   max_sqrt_alpha_bar = 0.95
        #   start_angle = tf.acos(max_sqrt_alpha_bar)
        #   end_angle = tf.acos(min_sqrt_alpha_bar)
        #
        #   diffusion_times = tf.range(0, timesteps, dtype=tf.float32) / timesteps
        #   diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        #
        #   sqrt_alpha_bar = tf.cos(diffusion_angles)
        #   sqrt_one_minus_alpha_bar = tf.sin(diffusion_angles)
        #
        # The bounds are directly imposed on sqrt(alpha_bar), not alpha_bar.
        min_sqrt_alpha = float(config.min_sqrt_alpha_bar)
        max_sqrt_alpha = float(config.max_sqrt_alpha_bar)

        if not (0.0 < min_sqrt_alpha < max_sqrt_alpha < 1.0):
            raise ValueError(
                "Expected 0 < min_sqrt_alpha_bar < "
                "max_sqrt_alpha_bar < 1."
            )

        start_angle = np.arccos(max_sqrt_alpha)
        end_angle = np.arccos(min_sqrt_alpha)
        diffusion_times = np.arange(n, dtype=np.float64) / float(n)
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        sqrt_alpha_bar = np.cos(diffusion_angles)
        alpha_bar = sqrt_alpha_bar**2

        alpha_bar = _apply_snr_shift(alpha_bar, config.snr_shift)
        return alpha_bar_to_betas(
            alpha_bar,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
        )

    if config.kind == ScheduleKind.SIGMOID:
        # Sigmoid-shaped alpha_bar decay: slow early/late, steeper mid-way.
        alpha_bar = 1.0 - _sigmoid01(t, k=config.logistic_k)
        alpha_bar = _apply_snr_shift(alpha_bar, config.snr_shift)
        alpha_bar = np.clip(alpha_bar, 1e-12, 1.0)
        return alpha_bar_to_betas(
            alpha_bar,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
        )

    if config.kind == ScheduleKind.LOGISTIC:
        # A smoother alternative used in editing work: alpha_bar = sigmoid(-k*(t-0.5)).
        alpha_bar = 1.0 / (1.0 + np.exp(config.logistic_k * (t - 0.5)))
        alpha_bar = _apply_snr_shift(alpha_bar, config.snr_shift)
        alpha_bar = np.clip(alpha_bar, 1e-12, 1.0)
        return alpha_bar_to_betas(
            alpha_bar,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
        )

    if config.kind in {ScheduleKind.VE, ScheduleKind.KARRAS, ScheduleKind.SUB_VP}:
        sigmas = generate_sigmas(config)
        betas = sigmas_to_betas(sigmas)
        return np.clip(betas, config.clip_min, config.clip_max)

    raise ValueError(f"Unsupported schedule kind: {config.kind}")


def generate_sigmas(config: ScheduleConfig) -> np.ndarray:
    """Generate a sigma schedule.

    For VE/Karras schedules, sigmas are the natural representation.
    For VP-style schedules, sigmas are derived from alpha_bar.
    """
    n = int(config.num_steps)
    if n < 2:
        raise ValueError("num_steps must be >= 2")

    t = np.linspace(0.0, 1.0, n, dtype=np.float64)

    if config.kind == ScheduleKind.VE:
        # Variance exploding: sigma grows from sigma_min to sigma_max.
        sigmas = config.sigma_min * (config.sigma_max / config.sigma_min) ** t
        return np.clip(sigmas, config.clip_min, None)

    if config.kind == ScheduleKind.KARRAS:
        # Karras rho schedule in sigma-space.
        inv_rho = 1.0 / config.rho
        sigmas = (
            config.sigma_min**inv_rho
            + t * (config.sigma_max**inv_rho - config.sigma_min**inv_rho)
        ) ** config.rho
        return np.clip(sigmas, config.clip_min, None)

    if config.kind == ScheduleKind.SUB_VP:
        # α(t) = 1 - σ(t) is a common sub-VP/flow-matching parameterization.
        # We use a simple monotone corruption path in [0,1).
        # A cosine-shaped corruption path works well in practice.
        alpha_bar = _cosine_alpha_bar(t, s=config.cosine_s)
        sigma = 1.0 - np.sqrt(np.clip(alpha_bar, 0.0, 1.0))
        sigma = _apply_snr_shift(1.0 - sigma**2, config.snr_shift)
        sigma = np.sqrt(np.clip(1.0 - sigma, 0.0, 1.0))
        return np.clip(sigma, config.clip_min, 1.0 - config.clip_min)

    # For VP-style schedules, derive sigmas from alpha_bar.
    betas = generate_betas(
        ScheduleConfig(
            kind=config.kind,
            num_steps=config.num_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            cosine_s=config.cosine_s,
            min_sqrt_alpha_bar=config.min_sqrt_alpha_bar,
            max_sqrt_alpha_bar=config.max_sqrt_alpha_bar,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            rho=config.rho,
            snr_shift=config.snr_shift,
            logistic_k=config.logistic_k,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
        )
    )
    alpha_bar = betas_to_alpha_bar(betas)
    return alpha_bar_to_sigmas(alpha_bar)


def schedule_timesteps(config: ScheduleConfig) -> np.ndarray:
    """Return normalized timesteps in [0,1]."""
    return np.linspace(0.0, 1.0, int(config.num_steps), dtype=np.float64)


def make_schedule(
    kind: str,
    num_steps: int = 1000,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Convenience wrapper that returns betas, alpha_bar, and sigmas.

    Parameters
    ----------
    kind:
        One of: linear, scaled_linear, squaredcos_cap_v2, clipped_cosine,
        sigmoid, quadratic, ve, karras, sub_vp, logistic.
    num_steps:
        Number of discrete steps.
    kwargs:
        Optional parameters such as beta_start, beta_end, sigma_min, sigma_max,
        rho, snr_shift, cosine_s, logistic_k,
        min_sqrt_alpha_bar, max_sqrt_alpha_bar.
    """
    cfg = ScheduleConfig(kind=ScheduleKind(kind), num_steps=num_steps, **kwargs)
    betas = generate_betas(cfg)
    alpha_bar = betas_to_alpha_bar(betas)
    sigmas = alpha_bar_to_sigmas(alpha_bar)
    return {
        "betas": betas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": np.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": np.sqrt(1.0 - alpha_bar),
        "sigmas": sigmas,
        "timesteps": schedule_timesteps(cfg),
    }


# A small registry for downstream callers.
SCHEDULE_REGISTRY = {
    kind.value: kind for kind in ScheduleKind
}


SchedulerName: TypeAlias = Literal[
    "linear",
    "scaled_linear",
    "squaredcos_cap_v2",
    "clipped_cosine",
    "sigmoid",
    "quadratic",
    "ve",
    "karras",
    "sub_vp",
    "logistic",
]


if __name__ == "__main__":
    # Example usage.
    for name in SCHEDULE_REGISTRY.values():
        out = make_schedule(name, num_steps=8)
        print(f"\n{name}")
        print("betas      :", np.round(out["betas"], 6))
        print("sigmas     :", np.round(out["sigmas"], 6))
        print("alpha_bar  :", np.round(out["alpha_bar"], 6))
        print("sqrt_alpha :", np.round(out["sqrt_alpha_bar"], 6))
