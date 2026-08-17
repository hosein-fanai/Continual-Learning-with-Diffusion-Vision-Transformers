"""Create NumPy diffusion-noise schedules.

The public workflow is to construct :class:`ScheduleConfig`, pass it to
:func:`generate_betas` or :func:`generate_sigmas`, and use the conversion
helpers when another parameterization is required.  :func:`make_schedule` is
the compact string-based interface used by callers that want every common
array at once.

All returned arrays are one-dimensional ``numpy.ndarray`` objects with dtype
``float64`` and length ``num_steps``.  Discrete variance-preserving (VP)
schedules are naturally represented by beta values.  ``ve`` and ``karras``
are naturally represented in sigma space; their beta results are a clipped VP
equivalent for code paths that require betas.
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass

from enum import Enum

from math import pi

from typing import Iterable, Literal, TypeAlias


class ScheduleKind(str, Enum):
    """Names of the supported schedule families.

    Values
    ------
    LINEAR
        Linearly interpolate beta from ``beta_start`` to ``beta_end``.
    SCALED_LINEAR
        Linearly interpolate ``sqrt(beta)`` and square the result.
    COSINE
        Follow the ``squaredcos_cap_v2`` cumulative-alpha curve.
    CLIPPED_COSINE
        Interpolate cosine angles between configured signal bounds.
    SIGMOID, LOGISTIC
        Use smooth S-shaped cumulative-alpha decay curves.
    QUADRATIC
        Increase beta quadratically over normalized time.
    VE
        Increase sigma geometrically from ``sigma_min`` to ``sigma_max``.
    KARRAS
        Use a rho-shaped interpolation in sigma space.
    SUB_VP
        Use the module's cosine-derived sub-VP corruption path.

    Notes
    -----
    Members inherit from ``str``.  Consequently either
    ``ScheduleKind.LINEAR`` or its value ``"linear"`` can be passed through
    string-oriented APIs such as :func:`make_schedule`.
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
    """Immutable parameters for one discretized noise schedule.

    Parameters
    ----------
    kind : ScheduleKind, default=ScheduleKind.LINEAR
        Schedule family.  Pass an enum member when constructing this class;
        :func:`make_schedule` accepts the corresponding string instead.
    num_steps : int, default=1000
        Number of discrete points.  Generation functions require at least 2.
    beta_start, beta_end : float, default=1e-4 and 2e-2
        Endpoints used by ``linear``, ``scaled_linear``, and ``quadratic``.
        Values are expected to describe valid variances; final values are
        clipped to ``[clip_min, clip_max]``.
    cosine_s : float, default=0.008
        Offset in the cosine cumulative-alpha curve.  Larger values change
        the amount of early-time noise.
    min_sqrt_alpha_bar, max_sqrt_alpha_bar : float, default=0.02 and 0.95
        Signal-amplitude bounds used only by ``clipped_cosine``.  They must
        satisfy ``0 < min < max < 1``.
    sigma_min, sigma_max : float, default=0.002 and 80.0
        Sigma endpoints used by ``ve`` and ``karras``.  Both should be
        positive; ``sigma_max`` normally exceeds ``sigma_min``.
    rho : float, default=7.0
        Curvature for ``karras``.  It must be nonzero because its reciprocal
        is evaluated; positive values are the intended input.
    snr_shift : float, default=0.0
        Additive log-SNR shift applied by compatible schedules.  ``0`` leaves
        the curve unchanged, a positive value preserves more signal at a
        given step, and a negative value introduces more noise.
    logistic_k : float, default=10.0
        Steepness of ``sigmoid`` and ``logistic`` transitions.  Larger
        positive values concentrate their change near the midpoint.
    clip_min, clip_max : float, default=1e-8 and 0.999
        Inclusive numerical bounds for beta-like results.  Use values inside
        ``(0, 1)`` to keep conversion helpers valid.

    Notes
    -----
    Fields irrelevant to the selected ``kind`` are retained but ignored.  For
    example, changing ``rho`` has no effect on a ``linear`` schedule.
    Instances are frozen, so create a new config rather than assigning a
    field after construction.
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


def _as_float64(x: np.ndarray | Iterable[float]) -> np.ndarray:
    """Convert an array or finite iterable to a ``float64`` NumPy array.

    Parameters
    ----------
    x : numpy.ndarray or iterable of float
        Numeric values of any NumPy-compatible shape.  Integer values are
        converted rather than rounded or normalized.

    Returns
    -------
    numpy.ndarray
        An array with the same shape as ``x`` and dtype ``float64``.
    """

    return np.asarray(x, dtype=np.float64)


def betas_to_alpha_bar(betas: np.ndarray) -> np.ndarray:
    """Convert per-step variances to cumulative retained-signal power.

    Parameters
    ----------
    betas : numpy.ndarray
        One-dimensional beta values, each strictly inside ``(0, 1)``.  For
        example, ``[0.1, 0.2]`` represents per-step alpha values ``[0.9,
        0.8]``.

    Returns
    -------
    numpy.ndarray
        ``float64`` cumulative products of ``1 - betas`` with the same shape;
        the example input returns ``[0.9, 0.72]``.

    Raises
    ------
    ValueError
        If any beta is zero, negative, or at least one.
    """
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
    """Convert a cumulative signal trajectory to per-step beta values.

    Parameters
    ----------
    alpha_bar : numpy.ndarray
        One-dimensional, monotonically non-increasing retained-signal values.
        Inputs are clipped to ``[1e-12, 1]`` before validation.  The first
        beta is ``1 - alpha_bar[0]``; later betas are
        ``1 - alpha_bar[t] / alpha_bar[t - 1]``.
    clip_min, clip_max : float, default=1e-8 and 0.999
        Bounds applied to every returned beta.  They prevent exact zero and
        one values by default.

    Returns
    -------
    numpy.ndarray
        ``float64`` beta schedule with the same shape as ``alpha_bar``.

    Raises
    ------
    ValueError
        If the clipped trajectory increases between adjacent steps.

    Examples
    --------
    ``alpha_bar_to_betas(np.array([0.9, 0.72]))`` returns approximately
    ``array([0.1, 0.2])``.
    """
    alpha_bar = np.clip(_as_float64(alpha_bar), 1e-12, 1.0)

    if np.any(np.diff(alpha_bar) > 1e-12):
        raise ValueError("alpha_bar must be monotonically non-increasing.")

    betas = np.empty_like(alpha_bar)
    betas[0] = 1.0 - alpha_bar[0]
    betas[1:] = 1.0 - alpha_bar[1:] / alpha_bar[:-1]

    return np.clip(betas, clip_min, clip_max)


def alpha_bar_to_sigmas(alpha_bar: np.ndarray) -> np.ndarray:
    """Convert retained-signal power to VP corruption amplitudes.

    Parameters
    ----------
    alpha_bar : numpy.ndarray
        Numeric cumulative-alpha values.  Values outside ``[0, 1]`` are
        clipped, so negative entries become sigma ``1`` and entries above one
        become sigma ``0``.

    Returns
    -------
    numpy.ndarray
        ``float64`` values computed as ``sqrt(1 - alpha_bar)`` with the same
        shape as the input.
    """
    alpha_bar = np.clip(_as_float64(alpha_bar), 0.0, 1.0)
    return np.sqrt(np.maximum(1.0 - alpha_bar, 0.0))


def sigmas_to_betas(sigmas: np.ndarray) -> np.ndarray:
    """Convert VP corruption amplitudes to an equivalent beta schedule.

    Parameters
    ----------
    sigmas : numpy.ndarray
        One-dimensional sigma values interpreted through
        ``alpha_bar = 1 - sigma**2``.  Inputs are clipped to ``[0, 1)``;
        therefore natural VE/Karras sigmas above one lose their original
        magnitude in this beta-equivalent representation.

    Returns
    -------
    numpy.ndarray
        Beta values in ``[1e-8, 0.999]`` with the same shape as ``sigmas``.

    Raises
    ------
    ValueError
        If the implied cumulative-alpha curve increases, which can happen
        when ``sigmas`` decreases.
    """
    sigmas = np.clip(_as_float64(sigmas), 0.0, 1.0 - 1e-12)
    alpha_bar = 1.0 - sigmas**2
    alpha_bar = np.clip(alpha_bar, 1e-12, 1.0)
    betas = alpha_bar_to_betas(alpha_bar)
    return np.clip(betas, 1e-8, 0.999)


def _apply_snr_shift(alpha_bar: np.ndarray, snr_shift: float) -> np.ndarray:
    """Add a scalar offset to a curve in log signal-to-noise space.

    Parameters
    ----------
    alpha_bar : numpy.ndarray
        Retained-signal powers.  With a nonzero shift, values are clipped away
        from exactly zero and one before conversion.
    snr_shift : float
        Offset applied as ``SNR' = SNR * exp(snr_shift)``.  Zero returns the
        original object unchanged; positive values increase ``alpha_bar`` and
        negative values decrease it.

    Returns
    -------
    numpy.ndarray
        Shifted retained-signal powers with the input shape.
    """
    if snr_shift == 0.0:
        return alpha_bar

    alpha_bar = np.clip(_as_float64(alpha_bar), 1e-12, 1.0 - 1e-12)
    snr = alpha_bar / np.maximum(1.0 - alpha_bar, 1e-12)
    shifted = snr * np.exp(snr_shift)
    return shifted / (1.0 + shifted)


def _cosine_alpha_bar(t: np.ndarray, s: float = 0.008) -> np.ndarray:
    """Evaluate the normalized squared-cosine signal curve.

    Parameters
    ----------
    t : numpy.ndarray
        Typically normalized times in ``[0, 1]``.  Values are not clipped.
    s : float, default=0.008
        Time offset that softens the curve near zero.

    Returns
    -------
    numpy.ndarray
        ``cos(((t + s) / (1 + s)) * pi / 2) ** 2``, normalized by its first
        entry.  The output has the same shape as ``t``.
    """

    f = np.cos(((t + s) / (1.0 + s)) * (pi / 2.0)) ** 2
    return f / f[0]


def _sigmoid01(t: np.ndarray, k: float = 10.0) -> np.ndarray:
    """Evaluate a centered logistic map after clipping time to ``[0, 1]``.

    Parameters
    ----------
    t : numpy.ndarray
        Times of any shape.  Values below zero behave as zero and values above
        one behave as one.
    k : float, default=10.0
        Transition steepness.  ``k=0`` returns ``0.5`` everywhere; positive
        values rise with time and negative values fall.

    Returns
    -------
    numpy.ndarray
        Logistic values with the same shape as ``t``.
    """
    x = np.clip(t, 0.0, 1.0)
    z = k * (x - 0.5)
    return 1.0 / (1.0 + np.exp(-z))


def generate_betas(config: ScheduleConfig) -> np.ndarray:
    """Generate per-step beta values for a configured schedule.

    Parameters
    ----------
    config : ScheduleConfig
        Complete schedule configuration.  ``kind`` selects the algorithm:
        beta endpoints affect ``linear``, ``scaled_linear``, and ``quadratic``;
        cosine fields affect cosine-based schedules; ``logistic_k`` affects
        S-shaped schedules; and sigma/rho fields affect ``ve`` and ``karras``.

    Returns
    -------
    numpy.ndarray
        One-dimensional ``float64`` array of length ``config.num_steps``.
        Values are clipped to the configured beta bounds.  For ``ve``,
        ``karras``, and ``sub_vp``, the result is a VP beta-equivalent of the
        sigma curve rather than the natural sigma-space schedule.

    Raises
    ------
    ValueError
        If fewer than two steps are requested, ``clipped_cosine`` signal
        bounds are unordered/outside ``(0, 1)``, the selected kind is not
        supported, or a derived cumulative-alpha trajectory is increasing.

    Examples
    --------
    A two-endpoint linear ramp can be created with
    ``generate_betas(ScheduleConfig(num_steps=4, beta_start=1e-4,
    beta_end=2e-2))``.  Selecting ``ScheduleKind.KARRAS`` instead makes
    ``sigma_min``, ``sigma_max``, and ``rho`` the meaningful controls.
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
    """Generate corruption amplitudes for a configured schedule.

    Parameters
    ----------
    config : ScheduleConfig
        Complete schedule configuration.  ``ve`` geometrically interpolates
        ``sigma_min`` to ``sigma_max``; ``karras`` additionally uses ``rho``;
        and ``sub_vp`` creates a bounded cosine-derived path.  Other kinds are
        generated as betas and converted through cumulative alpha.

    Returns
    -------
    numpy.ndarray
        One-dimensional ``float64`` array of length ``config.num_steps``.
        VE and Karras values may exceed one and are lower-bounded by
        ``clip_min``.  Sub-VP and beta-derived values stay in approximately
        ``[0, 1)``.

    Raises
    ------
    ValueError
        If fewer than two steps are requested or delegated beta generation
        rejects the configuration.

    Notes
    -----
    ``generate_sigmas`` preserves natural VE/Karras magnitudes.  By contrast,
    ``make_schedule`` reports sigmas reconstructed from its beta-equivalent
    curve, so call this function directly when a sigma-space sampler needs
    values such as ``sigma_max=80``.
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
    """Return evenly spaced normalized times, including both endpoints.

    Parameters
    ----------
    config : ScheduleConfig
        Only ``num_steps`` is read.  Unlike the generation functions, this
        helper does not explicitly reject values below two; NumPy determines
        the resulting empty, singleton, or multi-point array.

    Returns
    -------
    numpy.ndarray
        ``float64`` array equivalent to
        ``numpy.linspace(0, 1, config.num_steps)``.
    """
    return np.linspace(0.0, 1.0, int(config.num_steps), dtype=np.float64)


def make_schedule(
    kind: str,
    num_steps: int = 1000,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Build all common VP arrays from a schedule name.

    Parameters
    ----------
    kind : str
        One of ``"linear"``, ``"scaled_linear"``,
        ``"squaredcos_cap_v2"``, ``"clipped_cosine"``, ``"sigmoid"``,
        ``"quadratic"``, ``"ve"``, ``"karras"``, ``"sub_vp"``, or
        ``"logistic"``.  Enum members also work because
        :class:`ScheduleKind` inherits from ``str``.
    num_steps : int, default=1000
        Number of entries in every returned array; it must be at least two.
    **kwargs : float
        Optional :class:`ScheduleConfig` fields.  Allowed keys are
        ``beta_start``, ``beta_end``, ``cosine_s``,
        ``min_sqrt_alpha_bar``, ``max_sqrt_alpha_bar``, ``sigma_min``,
        ``sigma_max``, ``rho``, ``snr_shift``, ``logistic_k``, ``clip_min``,
        and ``clip_max``.  Values are numeric and have the meanings documented
        on :class:`ScheduleConfig`.  ``kind`` and ``num_steps`` must be passed
        as the named parameters, not repeated in ``kwargs``.  Unknown keys
        cause the dataclass constructor to raise ``TypeError``.

    Returns
    -------
    dict[str, numpy.ndarray]
        A dictionary with exactly six ``float64`` arrays of shape
        ``(num_steps,)``: ``"betas"``, ``"alpha_bar"`` (cumulative
        ``1-beta``), ``"sqrt_alpha_bar"``,
        ``"sqrt_one_minus_alpha_bar"``, ``"sigmas"`` (identical to the
        preceding noise-amplitude array), and normalized ``"timesteps"``.

    Raises
    ------
    ValueError
        If ``kind`` is unknown or schedule parameters fail validation.
    TypeError
        If ``kwargs`` contains a key not declared by :class:`ScheduleConfig`.

    Examples
    --------
    ``make_schedule("linear", 100, beta_end=0.01)`` customizes a beta-space
    ramp.  ``make_schedule("karras", 20, sigma_min=0.01, sigma_max=10,
    rho=5)`` accepts sigma-space controls, but its returned ``"sigmas"`` are
    reconstructed from clipped beta-equivalent values; use
    :func:`generate_sigmas` to retain the natural maximum of ``10``.
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


if __name__ == "__main__":
    # Example usage.
    for name in SCHEDULE_REGISTRY.values():
        out = make_schedule(name, num_steps=8)
        print(f"\n{name}")
        print("betas      :", np.round(out["betas"], 6))
        print("sigmas     :", np.round(out["sigmas"], 6))
        print("alpha_bar  :", np.round(out["alpha_bar"], 6))
        print("sqrt_alpha :", np.round(out["sqrt_alpha_bar"], 6))
