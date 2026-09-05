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
- sub_vp (sub-variance-preserving marginal standard deviation)
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

from pathlib import Path

from typing import Iterable, Literal, TypeAlias


class ScheduleKind(str, Enum):
    """Names of the ten supported forward-noise schedule families.

    Members also behave as strings and can be passed directly to make_schedule.
    Their values are the accepted serialized scheduler names; no schedule arrays
    are allocated by selecting an enum value.

    Attributes:
        LINEAR (ScheduleKind): ``"linear"`` interpolates beta endpoints directly.
        SCALED_LINEAR (ScheduleKind): ``"scaled_linear"`` interpolates square-root
            beta endpoints and squares the result.
        COSINE (ScheduleKind): ``"squaredcos_cap_v2"`` discretizes normalized
            squared-cosine signal power over num_steps + 1 interval edges.
        CLIPPED_COSINE (ScheduleKind): ``"clipped_cosine"`` sweeps cosine angles
            between configured signal-amplitude bounds on an endpoint-exclusive grid.
        SIGMOID (ScheduleKind): ``"sigmoid"`` interpolates beta through a centered
            logistic transition; its endpoints approach the configured bounds.
        QUADRATIC (ScheduleKind): ``"quadratic"`` increases beta quadratically in time.
        VE (ScheduleKind): ``"ve"`` increases native sigma geometrically between endpoints.
        KARRAS (ScheduleKind): ``"karras"`` interpolates native sigma in inverse-rho space.
        SUB_VP (ScheduleKind): ``"sub_vp"`` uses cosine signal power and the native
            sub-VP marginal standard deviation ``1 - alpha_bar``.
        LOGISTIC (ScheduleKind): ``"logistic"`` applies a centered logistic decay
            directly to cumulative signal power rather than beta.
    """

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
    """Store immutable parameters for one discretized forward-noise schedule.

    Construction records values without validation; schedule generation and
    schedule_timesteps call _validate_config before using them. Every numerical
    field is validated, including fields irrelevant to the selected family.
    Changing an irrelevant valid field therefore leaves that family's curve
    unchanged. Create a new configuration to change a frozen instance.

    Attributes:
        kind (ScheduleKind): Schedule family; its matching string value also compares
            equal to the enum. Defaults to ``ScheduleKind.LINEAR``.
        num_steps (int): Number of discrete output points/transitions; requires at
            least two in generation APIs. Defaults to ``1000``.
        beta_start (float): Positive lower beta bound for linear, scaled-linear,
            quadratic, and sigmoid families. Defaults to ``0.0001``.
        beta_end (float): Upper beta bound satisfying beta_start <= beta_end < 1.
            Sigmoid approaches these bounds on its finite grid. Defaults to ``0.02``.
        cosine_s (float): Nonnegative time offset used by cosine and sub-VP
            signal-power curves. Defaults to ``0.008``.
        min_sqrt_alpha_bar (float): Lower clipped-cosine signal-amplitude endpoint;
            requires 0 < min_sqrt_alpha_bar < max_sqrt_alpha_bar < 1.
            Defaults to ``0.02``.
        max_sqrt_alpha_bar (float): Upper clipped-cosine signal-amplitude endpoint.
            Defaults to ``0.95``.
        sigma_min (float): Positive first native VE/Karras noise-to-data ratio.
            Defaults to ``0.002``.
        sigma_max (float): Last native VE/Karras ratio, at least sigma_min.
            Defaults to ``80.0``.
        rho (float): Positive Karras interpolation curvature. Defaults to ``7.0``.
        snr_shift (float): Additive log-SNR shift; positive preserves more signal,
            negative adds noise, zero preserves the base curve. Direct native
            VE/Karras sigma generation ignores it; beta-equivalent generation
            applies it. Defaults to ``0.0``.
        logistic_k (float): Positive sigmoid/logistic sharpness; larger values
            concentrate change near the midpoint. Defaults to ``10.0``.
        clip_min (float): Lower final beta clamp, strictly above zero and below
            clip_max. Defaults to ``1e-08``.
        clip_max (float): Upper final beta clamp, strictly below one.
            Defaults to ``0.999``.
    """

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

    # Optional SNR shift. Positive values preserve more signal at each step.
    snr_shift: float = 0.0

    # Logistic schedule sharpness (larger -> steeper transition).
    logistic_k: float = 10.0

    # Safety clamp.
    clip_min: float = 1e-8
    clip_max: float = 0.999


def _validate_config(config: ScheduleConfig) -> None:
    """Validate shared numerical invariants of a schedule configuration.

    Args:
        config (ScheduleConfig): Immutable schedule parameters to validate.

    Returns:
        None: Valid configurations complete without a value.

    Raises:
        ValueError: If a count, bound, endpoint, or curve parameter is
            invalid or nonfinite.
    """

    # Require at least two discrete steps.
    if config.num_steps < 2:
        raise ValueError("num_steps must be at least 2.")

    numeric_values = (
        config.beta_start, config.beta_end, config.cosine_s,
        config.min_sqrt_alpha_bar, config.max_sqrt_alpha_bar,
        config.sigma_min, config.sigma_max, config.rho,
        config.snr_shift, config.logistic_k, config.clip_min, config.clip_max,
    )
    # Reject nonfinite curve parameters before they contaminate generated schedule arrays.
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError("All numeric schedule parameters must be finite.")

    # Keep numerical beta clamps strictly inside the probability interval.
    if not 0.0 < config.clip_min < config.clip_max < 1.0:
        raise ValueError("Expected 0 < clip_min < clip_max < 1.")
    # Require an ordered, probabilistically valid beta range.
    if not 0.0 < config.beta_start <= config.beta_end < 1.0:
        raise ValueError("Expected 0 < beta_start <= beta_end < 1.")
    # Negative cosine offsets violate the configured monotone forward-process domain.
    if config.cosine_s < 0.0:
        raise ValueError("cosine_s must be non-negative.")
    # Require ordered clipped-cosine signal bounds inside (0, 1).
    if not (
        0.0 < config.min_sqrt_alpha_bar
        < config.max_sqrt_alpha_bar < 1.0
    ):
        raise ValueError(
            "Expected 0 < min_sqrt_alpha_bar < max_sqrt_alpha_bar < 1."
        )
    # Require a positive, ordered sigma range for sigma-space schedules.
    if not 0.0 < config.sigma_min <= config.sigma_max:
        raise ValueError("Expected 0 < sigma_min <= sigma_max.")
    # Require positive Karras curvature.
    if config.rho <= 0.0:
        raise ValueError("rho must be positive.")
    # Require positive sigmoid/logistic transition sharpness.
    if config.logistic_k <= 0.0:
        raise ValueError("logistic_k must be positive.")


def _as_float64(x: np.ndarray | Iterable[float]) -> np.ndarray:
    """Convert NumPy-compatible numeric values to float64 without scaling.

    Args:
        x (np.ndarray | Iterable[float]): Numeric array or materialized sequence of
            any shape. A lazy generator must first be materialized for np.asarray.

    Returns:
        np.ndarray: Same-shaped float64 values. An existing float64 array may be
        returned without copying; input values are never modified.
    """

    return np.asarray(x, dtype=np.float64)


def betas_to_alpha_bar(betas: np.ndarray) -> np.ndarray:
    """Convert discrete variances to cumulative retained-signal power.

    Args:
        betas (np.ndarray): Nonempty finite one-dimensional variances strictly
            inside (0, 1); for example [0.1, 0.2].

    Returns:
        np.ndarray: Float64 cumulative products ``cumprod(1 - betas)`` shaped
        like betas; the example produces [0.9, 0.72]. Inputs are unchanged.

    Raises:
        ValueError: The input is empty, nonfinite, not one-dimensional, or contains
            a value outside the open probability interval.
    """

    betas = _as_float64(betas)
    # Reject NaN/Inf because range comparisons do not catch them.
    if betas.ndim != 1 or betas.size == 0 or not np.all(np.isfinite(betas)):
        raise ValueError("betas must be a non-empty finite one-dimensional array.")
    # Keep every discrete noise variance strictly between zero and one.
    if np.any(betas <= 0) or np.any(betas >= 1):
        raise ValueError("betas must lie strictly in (0, 1).")
    alphas = 1.0 - betas

    return np.cumprod(alphas)


def alpha_bar_to_betas(
    alpha_bar: np.ndarray, 
    clip_min: float = 1e-8, 
    clip_max: float = 0.999
) -> np.ndarray:
    """Convert a nonincreasing signal-power trajectory to discrete variances.

    Values are clipped to [1e-12, 1] before checking monotonicity. The first beta
    is 1 - alpha_bar[0]; subsequent betas are 1 - alpha_bar[t]/alpha_bar[t-1].
    Final beta clipping means inverse conversion may only approximate the source
    curve at a clamp boundary. The input is not modified.

    Args:
        alpha_bar (np.ndarray): Nonempty finite one-dimensional cumulative signal
            powers; increases above 1e-12 after clipping are rejected.
        clip_min (float): Positive lower beta clamp below clip_max.
            Defaults to ``1e-08``.
        clip_max (float): Upper beta clamp strictly below one.
            Defaults to ``0.999``.

    Returns:
        np.ndarray: Float64 beta values of the same shape in [clip_min, clip_max].
        For alpha_bar=[0.9, 0.72], returns approximately [0.1, 0.2].

    Raises:
        ValueError: Shape, finiteness, monotonicity, or 0 < clip_min < clip_max < 1
            is violated.
    """

    alpha_bar = _as_float64(alpha_bar)
    # Reject NaN/Inf before clipping, which preserves nonfinite entries.
    if alpha_bar.ndim != 1 or alpha_bar.size == 0 \
    or not np.all(np.isfinite(alpha_bar)):
        raise ValueError(
            "alpha_bar must be a non-empty finite one-dimensional array."
        )
    alpha_bar = np.clip(alpha_bar, 1e-12, 1.0)
    # Keep derived-beta clamps strictly inside the probability interval.
    if not 0.0 < clip_min < clip_max < 1.0:
        raise ValueError("Expected 0 < clip_min < clip_max < 1.")

    # Require cumulative signal power to decay monotonically over time.
    if np.any(np.diff(alpha_bar) > 1e-12):
        raise ValueError("alpha_bar must be monotonically non-increasing.")

    betas = np.empty_like(alpha_bar)
    betas[0] = 1.0 - alpha_bar[0]
    betas[1:] = 1.0 - alpha_bar[1:] / alpha_bar[:-1]

    return np.clip(betas, clip_min, clip_max)


def alpha_bar_to_sigmas(alpha_bar: np.ndarray) -> np.ndarray:
    """Convert VP retained-signal power to bounded corruption amplitude.

    Args:
        alpha_bar (np.ndarray): Numeric values of any shape. Values below zero or
            above one are clipped; NaN values propagate through the conversion.

    Returns:
        np.ndarray: Same-shaped float64 ``sqrt(1 - clip(alpha_bar, 0, 1))``.
        Negative inputs become one and inputs above one become zero. These are
        bounded VP amplitudes, not unbounded VE/Karras noise-to-data ratios.
    """

    alpha_bar = np.clip(_as_float64(alpha_bar), 0.0, 1.0)

    return np.sqrt(np.maximum(1.0 - alpha_bar, 0.0))


def sigmas_to_betas(
    sigmas: np.ndarray, 
    clip_min: float = 1e-8, 
    clip_max: float = 0.999
) -> np.ndarray:
    """Convert native noise-to-data ratios to an equivalent VP beta trajectory.

    The relation ``alpha_bar = 1 / (1 + sigma**2)`` preserves ratios above one.
    Their bounded VP corruption amplitude is sigma/sqrt(1 + sigma**2).

    Args:
        sigmas (np.ndarray): Nonempty, finite, nonnegative, one-dimensional,
            nondecreasing native noise-to-data ratios.
        clip_min (float): Positive lower beta clamp below clip_max.
            Defaults to ``1e-08``.
        clip_max (float): Upper beta clamp strictly below one.
            Defaults to ``0.999``.

    Returns:
        np.ndarray: Same-shaped float64 beta values clipped to [clip_min, clip_max].
        Clipping cumulative powers/betas can make inverse reconstruction approximate.

    Raises:
        ValueError: Shape, finiteness, sigma ordering/nonnegativity, or beta bounds
            violate the conversion contract.
    """

    sigmas = _as_float64(sigmas)
    # Reject NaN/Inf because ordering comparisons do not catch them.
    if sigmas.ndim != 1 or sigmas.size == 0 \
    or not np.all(np.isfinite(sigmas)):
        raise ValueError("sigmas must be a non-empty finite one-dimensional array.")
    # Require non-negative noise scales that do not decrease over time.
    if np.any(sigmas < 0.0) or np.any(np.diff(sigmas) < 0.0):
        raise ValueError("sigmas must be non-negative and non-decreasing.")
    alpha_bar = 1.0 / (1.0 + np.square(sigmas))
    betas = alpha_bar_to_betas(alpha_bar, clip_min, clip_max)

    return betas


def _apply_snr_shift(alpha_bar: np.ndarray, snr_shift: float) -> np.ndarray:
    """Translate a signal curve in log-SNR space with a stable logistic inverse.

    Args:
        alpha_bar (np.ndarray): Retained-signal powers of any shape. A nonzero
            shift first converts to float64 and clips to [1e-12, 1 - 1e-12].
        snr_shift (float): Log-SNR offset satisfying SNR_new = SNR * exp(snr_shift).
            Positive values preserve more signal; negative values add noise.

    Returns:
        np.ndarray: Same-shaped shifted float64 signal power for a nonzero shift.
        Zero returns the original object unchanged, preserving its dtype and values.
    """

    # Preserve the original signal curve when no SNR shift is requested.
    if snr_shift == 0.0:
        return alpha_bar

    alpha_bar = np.clip(_as_float64(alpha_bar), 1e-12, 1.0 - 1e-12)
    log_snr = np.log(alpha_bar) - np.log1p(-alpha_bar)
    shifted_log_snr = log_snr + snr_shift

    # Evaluate sigmoid(shifted_log_snr) without overflowing for large shifts.
    return np.exp(-np.logaddexp(0.0, -shifted_log_snr))


def _cosine_alpha_bar(t: np.ndarray, s: float = 0.008) -> np.ndarray:
    """Evaluate a squared-cosine signal curve normalized at the first sample.

    Args:
        t (np.ndarray): Nonempty time array, normally a one-dimensional float64 grid
            in [0, 1]. Times are not clipped by this helper.
        s (float): Time offset in ``cos(((t+s)/(1+s))*pi/2)**2``; generation APIs
            require it to be nonnegative.
            Defaults to ``0.008``.

    Returns:
        np.ndarray: Squared-cosine values divided by the first entry, with t's
        shape. Schedule grids produce float64 values beginning at one.
    """

    f = np.cos(((t + s) / (1.0 + s)) * (pi / 2.0)) ** 2

    return f / f[0]


def _sigmoid01(t: np.ndarray, k: float = 10.0) -> np.ndarray:
    """Evaluate a stable centered logistic transition on clipped normalized time.

    Args:
        t (np.ndarray): Numeric times of any shape; values outside [0, 1] use the
            nearest endpoint.
        k (float): Transition steepness. Zero returns 0.5 everywhere; positive
            values rise and negative values fall. Generation APIs require k > 0.
            Defaults to ``10.0``.

    Returns:
        np.ndarray: Same-shaped ``sigmoid(k * (clip(t, 0, 1) - 0.5))``. Finite
        endpoints need not equal zero/one. Float64 schedule grids retain float64.
    """

    x = np.clip(t, 0.0, 1.0)
    z = k * (x - 0.5)

    return np.exp(-np.logaddexp(0.0, -z))


def generate_betas(config: ScheduleConfig) -> np.ndarray:
    """Generate discrete forward variances for any supported schedule family.

    Linear, scaled-linear, quadratic, and sigmoid families interpolate beta.
    Cosine and sub-VP sample num_steps + 1 cumulative-power interval edges;
    clipped cosine samples angles on arange(num_steps)/num_steps, excluding its
    final angle. Logistic samples a declining cumulative-power sigmoid.
    VE/Karras convert native sigma through alpha_bar = 1/(1 + sigma**2).
    Every family supports a cumulative log-SNR shift before final beta conversion.

    Args:
        config (ScheduleConfig): Complete curve controls. Meaningful fields are
            documented on ScheduleConfig; all numerical fields are validated,
            even those not used by the chosen family.

    Returns:
        np.ndarray: Float64 variances shaped [config.num_steps], clipped to
        [clip_min, clip_max]. Cosine and sub-VP return the same beta-equivalent
        signal curve, although their native sigma definitions differ.

    Raises:
        ValueError: Shared configuration invariants, the selected family, or a
            derived signal-power trajectory are invalid.

    Examples:
        ``generate_betas(ScheduleConfig(num_steps=4, beta_end=0.01))`` creates
        a four-point linear ramp. Choosing ScheduleKind.KARRAS instead makes
        sigma_min, sigma_max, and rho the active native-curve controls.
    """

    _validate_config(config)
    n = config.num_steps

    t = np.linspace(0.0, 1.0, n, dtype=np.float64)

    # Construct schedules defined directly in beta space.
    if config.kind in {
        ScheduleKind.LINEAR,
        ScheduleKind.SCALED_LINEAR,
        ScheduleKind.QUADRATIC,
    }:
        # Interpolate beta directly for the standard linear schedule.
        if config.kind == ScheduleKind.LINEAR:
            betas = np.linspace(
                config.beta_start,
                config.beta_end,
                n,
                dtype=np.float64,
            )
        # Interpolate in square-root beta space for a gentler early ramp.
        elif config.kind == ScheduleKind.SCALED_LINEAR:
            betas = np.linspace(
                np.sqrt(config.beta_start),
                np.sqrt(config.beta_end),
                n,
            ) ** 2
        # Increase beta quadratically across normalized time.
        else:
            betas = config.beta_start + (
                config.beta_end - config.beta_start
            ) * (t**2)

        betas = np.clip(betas, config.clip_min, config.clip_max)
        # An unshifted beta-space ramp needs no cumulative-alpha round trip.
        if config.snr_shift == 0.0:
            return betas

        return alpha_bar_to_betas(
            _apply_snr_shift(
                betas_to_alpha_bar(betas),
                config.snr_shift,
            ),
            clip_min=config.clip_min,
            clip_max=config.clip_max,
        )

    # Discretize the cosine cumulative signal curve at interval edges.
    if config.kind == ScheduleKind.COSINE:
        edge_times = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
        alpha_bar_edges = _cosine_alpha_bar(edge_times, s=config.cosine_s)
        alpha_bar_edges = _apply_snr_shift(
            alpha_bar_edges,
            config.snr_shift,
        )
        alpha_bar_edges = np.clip(alpha_bar_edges, 1e-12, 1.0)

        return alpha_bar_to_betas(
            alpha_bar_edges[1:],
            clip_min=config.clip_min, 
            clip_max=config.clip_max, 
        )

    # Sweep between explicit square-root signal bounds for clipped cosine.
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

    # Interpolate beta directly through a centered sigmoid ramp.
    if config.kind == ScheduleKind.SIGMOID:
        # This is the generalized Diffusers sigmoid-beta construction.  It is
        # intentionally distinct from LOGISTIC, which shapes cumulative
        # alpha_bar instead. A sharpness of 12 spans the conventional logits
        # [-6, 6]; logistic_k keeps that span configurable.
        betas = config.beta_start + (
            config.beta_end - config.beta_start
        ) * _sigmoid01(t, k=config.logistic_k)
        betas = np.clip(betas, config.clip_min, config.clip_max)

        # Preserve the canonical beta ramp when no log-SNR shift is requested.
        if config.snr_shift == 0.0:
            return betas

        alpha_bar = _apply_snr_shift(
            betas_to_alpha_bar(betas),
            config.snr_shift,
        )

        return alpha_bar_to_betas(
            alpha_bar,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
        )

    # Convert the centered logistic signal curve into discrete betas.
    if config.kind == ScheduleKind.LOGISTIC:
        # A smoother alternative used in editing work: alpha_bar = sigmoid(-k*(t-0.5)).
        alpha_bar = _sigmoid01(1.0 - t, k=config.logistic_k)
        alpha_bar = _apply_snr_shift(alpha_bar, config.snr_shift)
        alpha_bar = np.clip(alpha_bar, 1e-12, 1.0)

        return alpha_bar_to_betas(
            alpha_bar, 
            clip_min=config.clip_min, 
            clip_max=config.clip_max, 
        )

    # Map VE/Karras raw sigma scales to equivalent VP signal power.
    if config.kind in {ScheduleKind.VE, ScheduleKind.KARRAS}:
        raw_sigmas = generate_sigmas(config)
        alpha_bar = 1.0 / (1.0 + np.square(raw_sigmas))
        alpha_bar = _apply_snr_shift(alpha_bar, config.snr_shift)

        return alpha_bar_to_betas(
            alpha_bar,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
        )

    # Discretize the shifted sub-VP cumulative signal curve at interval edges.
    if config.kind == ScheduleKind.SUB_VP:
        edge_times = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
        alpha_bar_edges = _cosine_alpha_bar(edge_times, s=config.cosine_s)
        alpha_bar_edges = _apply_snr_shift(
            alpha_bar_edges,
            config.snr_shift,
        )
        alpha_bar_edges = np.clip(alpha_bar_edges, 1e-12, 1.0)

        return alpha_bar_to_betas(
            alpha_bar_edges[1:],
            clip_min=config.clip_min,
            clip_max=config.clip_max,
        )

    raise ValueError(f"Unsupported schedule kind: {config.kind}")


def generate_sigmas(config: ScheduleConfig) -> np.ndarray:
    """Generate native or VP corruption amplitudes in increasing-time order.

    VE uses geometric interpolation and Karras uses inverse-rho interpolation
    between their exact native sigma endpoints; these two direct paths ignore
    snr_shift. Sub-VP returns 1 - alpha_bar on an endpoint-inclusive cosine grid
    after any SNR shift. Other families convert generate_betas output through
    cumulative alpha to sqrt(1 - alpha_bar).

    Args:
        config (ScheduleConfig): Curve controls, validated in full before selecting
            a generation path. See ScheduleConfig for all fields and defaults.

    Returns:
        np.ndarray: Float64 array shaped [config.num_steps]. VE/Karras preserve
        unbounded native noise-to-data ratios; sub-VP and other families return
        amplitudes in [0, 1]. The grid and native sub-VP marginal can differ from
        make_schedule's beta-equivalent VP outputs.

    Raises:
        ValueError: Shared invariants or delegated beta-generation requirements fail.
    """

    _validate_config(config)
    n = config.num_steps

    t = np.linspace(0.0, 1.0, n, dtype=np.float64)

    # Grow VE sigma geometrically between its configured endpoints.
    if config.kind == ScheduleKind.VE:
        # Variance exploding: sigma grows from sigma_min to sigma_max.
        sigmas = config.sigma_min * (config.sigma_max / config.sigma_min) ** t

        return sigmas

    # Interpolate Karras sigma in inverse-rho space.
    if config.kind == ScheduleKind.KARRAS:
        # Karras rho schedule in sigma-space.
        inv_rho = 1.0 / config.rho
        sigmas = (
            config.sigma_min**inv_rho
            + t * (config.sigma_max**inv_rho - config.sigma_min**inv_rho)
        ) ** config.rho

        return sigmas

    # Return the official sub-VP marginal standard deviation.
    if config.kind == ScheduleKind.SUB_VP:
        # The sub-VP marginal standard deviation is 1 - alpha_bar(t), not the
        # VP standard deviation sqrt(1 - alpha_bar(t)).
        alpha_bar = _cosine_alpha_bar(t, s=config.cosine_s)
        alpha_bar = _apply_snr_shift(alpha_bar, config.snr_shift)
        sigma = 1.0 - np.clip(alpha_bar, 0.0, 1.0)

        return sigma

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
            clip_max=config.clip_max
        )
    )
    alpha_bar = betas_to_alpha_bar(betas)

    return alpha_bar_to_sigmas(alpha_bar)


def schedule_timesteps(config: ScheduleConfig) -> np.ndarray:
    """Return normalized increasing times with both zero and one included.

    Args:
        config (ScheduleConfig): Schedule controls; all shared numerical invariants
            are checked, though only num_steps determines the returned grid.

    Returns:
        np.ndarray: Float64 ``linspace(0, 1, config.num_steps)`` shaped [num_steps].
        These plotting coordinates are separate from integer wrapper timestep IDs.

    Raises:
        ValueError: The schedule configuration violates shared numerical invariants.
    """

    _validate_config(config)

    return np.linspace(0.0, 1.0, config.num_steps, dtype=np.float64)


def make_schedule(
    kind: str | ScheduleKind, 
    num_steps: int = 1000, 
    **kwargs: float
) -> dict[str, np.ndarray]:
    """Build six consistent VP schedule arrays from a serialized family name.

    All outputs are reconstructed from one clipped beta trajectory. In particular,
    sigmas is the bounded VP amplitude, including for VE/Karras and sub-VP;
    use generate_sigmas directly when the native marginal is required.

    Args:
        kind (str | ScheduleKind): Case-sensitive family value: linear,
            scaled_linear, squaredcos_cap_v2, clipped_cosine, sigmoid, quadratic,
            ve, karras, sub_vp, or logistic. Enum members are also accepted.
        num_steps (int): Number of entries in each output, at least two.
            Defaults to ``1000``.
        **kwargs (float): ScheduleConfig overrides, empty by default: beta_start,
            beta_end, cosine_s, min_sqrt_alpha_bar, max_sqrt_alpha_bar, sigma_min,
            sigma_max, rho, snr_shift, logistic_k, clip_min, and clip_max. Omitted
            fields retain their ScheduleConfig defaults; kind/num_steps cannot
            be supplied twice.

    Returns:
        dict[str, np.ndarray]: Six float64 arrays shaped [num_steps]: betas,
        alpha_bar=cumprod(1-betas), sqrt_alpha_bar, sqrt_one_minus_alpha_bar,
        sigmas (equal to sqrt_one_minus_alpha_bar), and normalized timesteps.

    Raises:
        ValueError: The family name or numerical schedule parameters are invalid.
        TypeError: A keyword is unknown or supplied more than once.

    Examples:
        ``make_schedule('linear', 100, beta_end=0.01)`` customizes a beta ramp.
        ``make_schedule('karras', 20, sigma_min=0.01, sigma_max=10, rho=5)``
        returns bounded VP-equivalent amplitudes derived from the native ratios.
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
        "timesteps": schedule_timesteps(cfg)
    }


def save_schedule_plots(
    output_dir: str | Path = "schedule_plots", 
    num_steps: int = 1000, 
    dpi: int = 150, 
    metrics: Iterable[str] | None = None, 
    **schedule_kwargs: float
) -> dict[str, Path]:
    """Save side-by-side comparisons of every supported noise schedule.

    By default, one PNG is written per available statistic; ``metrics`` can
    select a smaller set. Each image contains a 5-by-2 grid with one subplot
    for every :class:`ScheduleKind`, which keeps coincident curves visible
    while preserving a common scale for direct comparison. Available images
    cover every numeric schedule output except the x-axis timesteps, plus
    per-step alpha, common complements, native sigma, SNR, log-SNR, a combined
    signal/noise-coefficient view, and a combined beta/one-minus-beta view.

    Supported metric names are betas, alphas, alpha_bar, one_minus_alpha_bar,
    sqrt_alpha_bar, sqrt_one_minus_alpha_bar, sigmas, native_sigmas, snr, log_snr,
    sqrt_alpha_bar_with_sqrt_one_minus_alpha_bar, and betas_with_one_minus_betas.
    Repeated names are written once in first-requested order. Matplotlib figures
    use an offscreen canvas and are cleared after saving.

    Args:
        output_dir (str | pathlib.Path): Directory in which the PNG files are
            created. Missing parent directories are created. Existing files
            with the generated names are replaced.
            Defaults to ``'schedule_plots'``.
        num_steps (int): Number of points in each schedule; defaults to 1000
            and must satisfy :class:`ScheduleConfig` validation.
            Defaults to ``1000``.
        dpi (int): Positive output resolution in dots per inch; defaults to
            150.
            Defaults to ``150``.
        metrics (Iterable[str] | None): Statistic names to save, in the
            requested order. ``None`` saves every available statistic. A
            single string is also accepted as one statistic name.
            Defaults to ``None``.
        schedule_kwargs (float): Optional :class:`ScheduleConfig` field
            overrides shared by every schedule, such as ``beta_end``,
            ``sigma_max``, or ``snr_shift``. Do not repeat ``kind`` or
            ``num_steps`` here.

    Returns:
        dict[str, pathlib.Path]: Statistic names mapped to the PNG files that
        were written, in plotting order.

    Raises:
        TypeError: If a schedule override is unknown or conflicts with an
            explicit argument.
        ValueError: If ``metrics`` or a schedule configuration is invalid.
        OSError: If the output directory or an image file cannot be written.
        ImportError: If Matplotlib is unavailable.

    Notes:
        ``sigmas`` is the bounded VP-equivalent amplitude returned by
        :func:`make_schedule`. ``native_sigmas`` comes from
        :func:`generate_sigmas` and therefore preserves VE/Karras magnitudes
        and the sub-VP marginal. Standalone beta and SNR plots use logarithmic
        y-axes; native sigma uses a symmetric-log y-axis so its exact zero is
        visible.
    """

    metric_specs = (
        (
            "sqrt_alpha_bar_with_sqrt_one_minus_alpha_bar",
            "Signal and noise coefficients",
            "bounded",
            (
                ("sqrt_alpha_bar", r"$\sqrt{\bar{\alpha}_t}$", "-"),
                (
                    "sqrt_one_minus_alpha_bar",
                    r"$\sqrt{1 - \bar{\alpha}_t}$",
                    "--",
                ),
            ),
        ),
        (
            "betas_with_one_minus_betas",
            "Per-step beta and one minus beta",
            "bounded",
            (
                ("betas", r"$\beta_t$", "-"),
                ("one_minus_betas", r"$1 - \beta_t$ ($\alpha_t$)", "--"),
            ),
        ),
        (
            "betas",
            r"Per-step beta ($\beta_t$)",
            "log",
            (("betas", r"$\beta_t$", "-"),),
        ),
        (
            "alphas",
            r"Per-step alpha ($\alpha_t$)",
            "bounded",
            (("alphas", r"$\alpha_t$", "-"),),
        ),
        (
            "alpha_bar",
            r"Cumulative alpha ($\bar{\alpha}_t$)",
            "bounded",
            (("alpha_bar", r"$\bar{\alpha}_t$", "-"),),
        ),
        (
            "one_minus_alpha_bar",
            r"$1 - \bar{\alpha}_t$",
            "bounded",
            (("one_minus_alpha_bar", r"$1 - \bar{\alpha}_t$", "-"),),
        ),
        (
            "sqrt_alpha_bar",
            r"$\sqrt{\bar{\alpha}_t}$",
            "bounded",
            (("sqrt_alpha_bar", r"$\sqrt{\bar{\alpha}_t}$", "-"),),
        ),
        (
            "sqrt_one_minus_alpha_bar",
            r"$\sqrt{1 - \bar{\alpha}_t}$",
            "bounded",
            (
                (
                    "sqrt_one_minus_alpha_bar",
                    r"$\sqrt{1 - \bar{\alpha}_t}$",
                    "-",
                ),
            ),
        ),
        (
            "sigmas",
            "VP-equivalent sigma",
            "bounded",
            (("sigmas", "VP-equivalent sigma", "-"),),
        ),
        (
            "native_sigmas",
            "Native sigma",
            "symlog",
            (("native_sigmas", "Native sigma", "-"),),
        ),
        (
            "snr",
            "Signal-to-noise ratio",
            "log",
            (("snr", "SNR", "-"),),
        ),
        (
            "log_snr",
            "Log signal-to-noise ratio",
            "linear",
            (("log_snr", "Log-SNR", "-"),),
        ),
    )
    metric_specs_by_name = {
        metric_spec[0]: metric_spec for metric_spec in metric_specs
    }

    # Select every statistic when the caller did not request a subset.
    if metrics is None:
        selected_metric_names = tuple(metric_specs_by_name)
    # Treat one string as one name instead of an iterable of characters.
    elif isinstance(metrics, str):
        selected_metric_names = (metrics,)
    # Materialize arbitrary iterables once while preserving their order.
    else:
        selected_metric_names = tuple(metrics)

    # Require at least one output statistic.
    if not selected_metric_names:
        raise ValueError("metrics must contain at least one statistic name.")
    # Report only requested plot metrics absent from the supported metric registry.
    unknown_metrics = tuple(
        metric_name
        for metric_name in selected_metric_names
        if metric_name not in metric_specs_by_name
    )
    # Reject misspelled names before generating schedules or creating files.
    if unknown_metrics:
        available_metrics = ", ".join(metric_specs_by_name)
        raise ValueError(
            f"Unknown metrics {unknown_metrics}; choose from {available_metrics}."
        )
    selected_metric_specs = tuple(
        metric_specs_by_name[metric_name]
        for metric_name in dict.fromkeys(selected_metric_names)
    )

    epsilon = np.finfo(np.float64).eps
    schedule_values: dict[str, dict[str, np.ndarray]] = {}

    for kind in ScheduleKind:
        config = ScheduleConfig(
            kind=kind,
            num_steps=num_steps,
            **schedule_kwargs,
        )
        schedule = make_schedule(kind, num_steps=num_steps, **schedule_kwargs)
        betas = schedule["betas"]
        alphas = 1.0 - betas
        alpha_bar = schedule["alpha_bar"]
        safe_alpha_bar = np.clip(alpha_bar, epsilon, 1.0 - epsilon)

        schedule_values[kind.value] = {
            "timesteps": schedule["timesteps"],
            "betas": betas,
            "alphas": alphas,
            "one_minus_betas": alphas,
            "alpha_bar": alpha_bar,
            "one_minus_alpha_bar": 1.0 - alpha_bar,
            "sqrt_alpha_bar": schedule["sqrt_alpha_bar"],
            "sqrt_one_minus_alpha_bar": schedule[
                "sqrt_one_minus_alpha_bar"
            ],
            "sigmas": schedule["sigmas"],
            "native_sigmas": generate_sigmas(config),
            "snr": safe_alpha_bar / (1.0 - safe_alpha_bar),
            "log_snr": (
                np.log(safe_alpha_bar) - np.log1p(-safe_alpha_bar)
            ),
        }

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure


    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}

    for (
        metric_name,
        metric_title,
        y_scale,
        series_specs,
    ) in selected_metric_specs:
        figure = Figure(figsize=(14.0, 18.0), constrained_layout=True)
        FigureCanvasAgg(figure)
        axes = np.asarray(
            figure.subplots(5, 2, sharex=True, sharey=True)
        ).ravel()
        try:
            for axis, (schedule_name, values) in zip(
                axes,
                schedule_values.items(),
            ):
                for series_name, series_label, line_style in series_specs:
                    axis.plot(
                        values["timesteps"],
                        values[series_name],
                        label=series_label,
                        linestyle=line_style,
                        linewidth=1.8,
                    )
                axis.set_title(schedule_name)
                axis.grid(True, which="both", alpha=0.25)

            # Label the shared color/style mapping once on combined plots.
            if len(series_specs) > 1:
                axes[0].legend(loc="center right")

            # Reveal positive quantities that span several orders of magnitude.
            if y_scale == "log":
                axes[0].set_yscale("log")
            # Retain zero while compressing the wide native-sigma range.
            elif y_scale == "symlog":
                axes[0].set_yscale("symlog", linthresh=1e-3)

            # Give every probability/amplitude panel the same complete range.
            if y_scale == "bounded":
                axes[0].set_ylim(-0.02, 1.02)

            figure.suptitle(f"{metric_title} across diffusion schedules")
            figure.supxlabel("Normalized timestep")
            figure.supylabel(metric_title)
            image_path = destination / f"{metric_name}.png"
            figure.savefig(image_path, dpi=dpi, bbox_inches="tight")
            saved_paths[metric_name] = image_path
        finally:
            axes[0].set_yscale("linear")
            figure.clear()
            figure.set_canvas(None)

    return saved_paths


# A small registry for downstream callers.
SCHEDULE_REGISTRY = {
    kind.value: kind for kind in ScheduleKind
}
"""Mapping from each accepted schedule string to its enum member."""


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
"""Static string type accepted by wrapper ``scheduler_name`` arguments."""


def run_self_tests() -> dict[str, str]:
    """Exercise every schedule enum member and configuration behavior.

    The checks cover enum/string round trips, registry completeness, frozen
    dataclass defaults and overrides, every schedule family at its minimum and
    a representative multi-step size, native sigma endpoints, conversion
    identities, clipping, log-SNR shifts, invalid bounds, and public helper
    error contracts.  Small NumPy arrays keep this suitable for the aggregate
    project self-test runner.

    Args:
        None.

    Returns:
        dict[str, str]: Exactly one ``"passed"`` result for each class defined
        by this module: :class:`ScheduleKind` and :class:`ScheduleConfig`.

    Raises:
        AssertionError: If any class invariant, numerical contract, or expected
            exception differs from the implementation.
    """

    from dataclasses import FrozenInstanceError, asdict, replace


    expected_names = {
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
    }
    assert {kind.value for kind in ScheduleKind} == expected_names
    assert set(SCHEDULE_REGISTRY) == expected_names
    for kind in ScheduleKind:
        assert isinstance(kind, str)
        assert kind == kind.value
        assert ScheduleKind(kind.value) is kind
        assert SCHEDULE_REGISTRY[kind.value] is kind
    with np.testing.assert_raises(ValueError):
        ScheduleKind("unknown")

    default = ScheduleConfig()
    assert default == ScheduleConfig(kind=ScheduleKind.LINEAR)
    assert hash(default) == hash(ScheduleConfig())
    assert asdict(default) == {
        "kind": ScheduleKind.LINEAR, 
        "num_steps": 1000, 
        "beta_start": 1e-4, 
        "beta_end": 2e-2, 
        "cosine_s": 0.008, 
        "min_sqrt_alpha_bar": 0.02, 
        "max_sqrt_alpha_bar": 0.95, 
        "sigma_min": 0.002, 
        "sigma_max": 80.0, 
        "rho": 7.0, 
        "snr_shift": 0.0, 
        "logistic_k": 10.0, 
        "clip_min": 1e-8, 
        "clip_max": 0.999, 
    }
    custom = ScheduleConfig(
        kind=ScheduleKind.KARRAS,
        num_steps=5, 
        beta_start=0.001, 
        beta_end=0.01, 
        cosine_s=0.01, 
        min_sqrt_alpha_bar=0.1, 
        max_sqrt_alpha_bar=0.9, 
        sigma_min=0.01, 
        sigma_max=10.0, 
        rho=3.0, 
        snr_shift=0.5, 
        logistic_k=4.0, 
        clip_min=1e-6, 
        clip_max=0.95, 
    )
    assert replace(custom, num_steps=6).num_steps == 6
    assert custom.kind is ScheduleKind.KARRAS
    assert custom.sigma_max == 10.0
    with np.testing.assert_raises(FrozenInstanceError):
        custom.num_steps = 7
    with np.testing.assert_raises(TypeError):
        ScheduleConfig(unknown_field=True)

    output_keys = {
        "betas", 
        "alpha_bar", 
        "sqrt_alpha_bar", 
        "sqrt_one_minus_alpha_bar", 
        "sigmas", 
        "timesteps", 
    }
    for kind in ScheduleKind:
        for num_steps in (2, 7):
            config = ScheduleConfig(kind=kind, num_steps=num_steps)
            betas = generate_betas(config)
            sigmas = generate_sigmas(config)
            times = schedule_timesteps(config)
            assert betas.shape == sigmas.shape == times.shape == (num_steps,)
            assert betas.dtype == sigmas.dtype == times.dtype == np.float64
            assert np.all(np.isfinite(betas)) and np.all(np.isfinite(sigmas))
            assert np.all((betas >= config.clip_min) & (betas <= config.clip_max))
            assert np.all(sigmas >= 0.0)
            np.testing.assert_allclose(times[[0, -1]], [0.0, 1.0])

            schedule = make_schedule(kind, num_steps=num_steps)
            assert set(schedule) == output_keys
            for value in schedule.values():
                assert value.shape == (num_steps,)
                assert value.dtype == np.float64
                assert np.all(np.isfinite(value))
            np.testing.assert_allclose(schedule["betas"], betas)
            np.testing.assert_allclose(
                schedule["sqrt_alpha_bar"] ** 2, 
                schedule["alpha_bar"], 
                atol=1e-12, 
            )
            np.testing.assert_allclose(
                schedule["sqrt_one_minus_alpha_bar"] ** 2, 
                1.0 - schedule["alpha_bar"], 
                atol=1e-12, 
            )
            np.testing.assert_allclose(
                schedule["sigmas"], 
                schedule["sqrt_one_minus_alpha_bar"], 
            )

    linear = ScheduleConfig(
        num_steps=5, 
        beta_start=0.001, 
        beta_end=0.02, 
        clip_min=1e-9, 
        clip_max=0.9, 
    )
    np.testing.assert_allclose(generate_betas(linear)[[0, -1]], [0.001, 0.02])
    scaled = replace(linear, kind=ScheduleKind.SCALED_LINEAR)
    np.testing.assert_allclose(generate_betas(scaled)[[0, -1]], [0.001, 0.02])
    quadratic = replace(linear, kind=ScheduleKind.QUADRATIC)
    quadratic_betas = generate_betas(quadratic)
    assert quadratic_betas[1] < generate_betas(linear)[1]
    for direct_beta_config in (linear, scaled, quadratic):
        shifted_direct = replace(direct_beta_config, snr_shift=0.75)
        assert np.all(
            betas_to_alpha_bar(generate_betas(shifted_direct))
            > betas_to_alpha_bar(generate_betas(direct_beta_config))
        )

    sigmoid = replace(
        linear,
        kind=ScheduleKind.SIGMOID,
        logistic_k=12.0,
    )
    sigmoid_times = np.linspace(0.0, 1.0, sigmoid.num_steps)
    reference_sigmoid_betas = sigmoid.beta_start + (
        sigmoid.beta_end - sigmoid.beta_start
    ) * _sigmoid01(sigmoid_times, sigmoid.logistic_k)
    np.testing.assert_allclose(
        generate_betas(sigmoid),
        reference_sigmoid_betas,
        rtol=1e-13,
        atol=1e-13,
    )
    assert np.all(np.diff(generate_betas(sigmoid)) > 0.0)
    logistic = replace(sigmoid, kind=ScheduleKind.LOGISTIC)
    assert not np.allclose(generate_betas(sigmoid), generate_betas(logistic))
    shifted_sigmoid = replace(sigmoid, snr_shift=0.75)
    assert np.all(
        betas_to_alpha_bar(generate_betas(shifted_sigmoid))
        > betas_to_alpha_bar(generate_betas(sigmoid))
    )

    cosine_config = ScheduleConfig(
        kind=ScheduleKind.COSINE,
        num_steps=4,
        cosine_s=0.008,
    )
    cosine_edges = _cosine_alpha_bar(
        np.arange(cosine_config.num_steps + 1, dtype=np.float64)
        / cosine_config.num_steps,
        cosine_config.cosine_s,
    )
    reference_cosine_betas = np.clip(
        1.0 - cosine_edges[1:] / cosine_edges[:-1],
        cosine_config.clip_min,
        cosine_config.clip_max,
    )
    np.testing.assert_allclose(
        generate_betas(cosine_config),
        reference_cosine_betas,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        generate_betas(replace(cosine_config, kind=ScheduleKind.SUB_VP)),
        reference_cosine_betas,
        rtol=1e-13,
        atol=1e-13,
    )

    clipped = ScheduleConfig(
        kind=ScheduleKind.CLIPPED_COSINE, 
        num_steps=9, 
        min_sqrt_alpha_bar=0.1, 
        max_sqrt_alpha_bar=0.9, 
    )
    clipped_alpha = betas_to_alpha_bar(generate_betas(clipped))
    assert np.all(np.diff(clipped_alpha) <= 0.0)
    for invalid_bounds in ((0.0, 0.9), (0.5, 0.5), (0.8, 0.2), (0.1, 1.0)):
        with np.testing.assert_raises(ValueError):
            generate_betas(replace(
                clipped, 
                min_sqrt_alpha_bar=invalid_bounds[0], 
                max_sqrt_alpha_bar=invalid_bounds[1], 
            ))

    ve = ScheduleConfig(
        kind=ScheduleKind.VE, 
        num_steps=6, 
        sigma_min=0.01, 
        sigma_max=12.0, 
    )
    np.testing.assert_allclose(generate_sigmas(ve)[[0, -1]], [0.01, 12.0])
    karras = replace(ve, kind=ScheduleKind.KARRAS, rho=3.0)
    np.testing.assert_allclose(generate_sigmas(karras)[[0, -1]], [0.01, 12.0])
    assert np.all(np.diff(generate_sigmas(ve)) > 0.0)
    assert np.all(np.diff(generate_sigmas(karras)) > 0.0)
    raw_ve_sigmas = generate_sigmas(ve)
    ve_alpha = betas_to_alpha_bar(generate_betas(ve))
    np.testing.assert_allclose(
        ve_alpha,
        1.0 / (1.0 + raw_ve_sigmas**2),
        rtol=1e-10,
        atol=1e-10,
    )

    sub_vp = ScheduleConfig(kind=ScheduleKind.SUB_VP, num_steps=9)
    sub_times = schedule_timesteps(sub_vp)
    sub_alpha = _cosine_alpha_bar(sub_times, sub_vp.cosine_s)
    np.testing.assert_allclose(generate_sigmas(sub_vp), 1.0 - sub_alpha)
    shifted_sub_vp = replace(sub_vp, snr_shift=0.75)
    np.testing.assert_allclose(
        generate_sigmas(shifted_sub_vp),
        1.0 - _apply_snr_shift(sub_alpha, 0.75),
    )

    alpha_bar = np.array([0.9, 0.72, 0.504], dtype=np.float64)
    betas = alpha_bar_to_betas(alpha_bar)
    np.testing.assert_allclose(betas, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(betas_to_alpha_bar(betas), alpha_bar)
    raw_sigmas = np.sqrt((1.0 - alpha_bar) / alpha_bar)
    np.testing.assert_allclose(sigmas_to_betas(raw_sigmas), betas)
    vp_sigmas = alpha_bar_to_sigmas(alpha_bar)
    np.testing.assert_allclose(
        raw_sigmas / np.sqrt(1.0 + raw_sigmas**2),
        vp_sigmas,
    )
    assert alpha_bar_to_sigmas(np.array([-1.0, 2.0])).tolist() == [1.0, 0.0]
    with np.testing.assert_raises(ValueError):
        betas_to_alpha_bar(np.array([0.0, 0.1]))
    with np.testing.assert_raises(ValueError):
        betas_to_alpha_bar(np.array([0.1, 1.0]))
    with np.testing.assert_raises(ValueError):
        alpha_bar_to_betas(np.array([0.5, 0.6]))
    with np.testing.assert_raises(ValueError):
        sigmas_to_betas(np.array([0.8, 0.2]))
    with np.testing.assert_raises(ValueError):
        sigmas_to_betas(np.array([-0.1, 0.2]))
    for converter, values in (
        (betas_to_alpha_bar, np.array([0.1, np.nan])),
        (alpha_bar_to_betas, np.array([0.9, np.nan])),
        (sigmas_to_betas, np.array([0.1, np.inf])),
    ):
        with np.testing.assert_raises(ValueError):
            converter(values)

    base_alpha = np.array([0.2, 0.5, 0.8], dtype=np.float64)
    assert _apply_snr_shift(base_alpha, 0.0) is base_alpha
    assert np.all(_apply_snr_shift(base_alpha, 1.0) > base_alpha)
    assert np.all(_apply_snr_shift(base_alpha, -1.0) < base_alpha)
    assert np.all(np.isfinite(_apply_snr_shift(base_alpha, 1_000.0)))
    assert np.all(np.isfinite(_apply_snr_shift(base_alpha, -1_000.0)))
    assert np.all(np.isfinite(_sigmoid01(
        np.asarray([0.0, 1.0]), k=1_000.0
    )))
    np.testing.assert_allclose(_sigmoid01(np.array([-1.0, 0.5, 2.0]), 0.0), 0.5)
    cosine = _cosine_alpha_bar(np.linspace(0.0, 1.0, 5))
    assert cosine[0] == 1.0 and cosine[-1] < 1e-20

    for generator in (generate_betas, generate_sigmas):
        with np.testing.assert_raises(ValueError):
            generator(ScheduleConfig(num_steps=1))
        with np.testing.assert_raises(ValueError):
            generator(ScheduleConfig(kind="unsupported", num_steps=3))
    with np.testing.assert_raises(ValueError):
        make_schedule("unsupported", num_steps=3)
    with np.testing.assert_raises(TypeError):
        make_schedule("linear", num_steps=3, unsupported=True)
    for invalid_config in (
        ScheduleConfig(num_steps=0),
        ScheduleConfig(num_steps=1),
        ScheduleConfig(cosine_s=-0.1),
        ScheduleConfig(snr_shift=float("nan")),
        ScheduleConfig(beta_start=0.1, beta_end=0.01),
        ScheduleConfig(sigma_min=0.0),
        ScheduleConfig(rho=0.0),
        ScheduleConfig(logistic_k=0.0),
        ScheduleConfig(clip_min=0.9, clip_max=0.1),
    ):
        with np.testing.assert_raises(ValueError):
            schedule_timesteps(invalid_config)

    return {"ScheduleKind": "passed", "ScheduleConfig": "passed"}


# Run scheduler regression tests when this module is invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
