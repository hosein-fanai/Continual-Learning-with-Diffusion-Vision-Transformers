"""Fixed-checkpoint inference controls shared by both cognitive routes.

Noisy inference delegates to the project's ``EnsembleAccuracy`` predictor;
calibration diagnostics delegate to ``common.mechanistic``. A clean primary
prediction is the reference. Ensembles change inference work, never training.
Entropy is descriptive predictive entropy, not calibrated epistemic uncertainty.

Temperature scaling follows Guo et al., ICML 2017
(https://proceedings.mlr.press/v70/guo17a.html). For a probability mixture its
canonical logits are log probabilities, not the individual heads' logits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import numpy as np
import tensorflow as tf

from common.mechanistic import _probability_matrix, calibration_metrics
from common.runtime import derive_seed, effective_seed
from diffusion.metrics.ensemble_accuracy import EnsembleAccuracy


@dataclass(frozen=True)
class EnsembleEvaluationSettings:
    """Prespecified inference treatments and validation calibration controls.

    Horizons are exclusive: horizon four evaluates schedule indices 0,1,2,3.
    Even schedule index zero is noised using q_sample; the clean reference
    bypasses q_sample entirely. Head order is primary, distillation, regularizer.
    Positive weights are normalized before prediction. All regularizer heads
    share their branch weight, as in EnsembleAccuracy.

    A positive calibration_fraction partitions validation rows within each
    represented class. Test evaluation instead requires separately supplied
    validation arrays. Different batch partitions may change stateless noise;
    batch size, ordering, seed, and software/hardware must be held fixed.
    """

    enabled: bool = False
    horizons: tuple[int, ...] = (1, 4)
    noise_draws: int = 1
    batch_size: int = 32
    network_name: str = "raw"
    compute_type: str = "chunked"
    t_chunk_size: int = 16
    head_weights: tuple[float, float, float] = (1., 0., 0.)
    separate_probas: bool = False
    weighted: bool = False
    seed: int = 0
    calibration_fraction: float = 0.
    ece_bins: int = 15
    temperature_bounds: tuple[float, float] = (0.05, 20.)

    def __post_init__(self) -> None:
        """Reject ambiguous coefficients, nonfinite bounds and invalid budgets.

        Returns:
            validated (None): None; frozen dataclass fields are normalized to numeric tuples and
                validated.

        Raises:
            ValueError: If flags, budgets, weights, seed, horizon, calibration fraction or
                temperature bounds are invalid.
        """

        for name in ("enabled", "separate_probas", "weighted"):
            # Strings such as 'false' must not accidentally enable inference work.
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"evaluation.{name} must be boolean.")
        for name in ("noise_draws", "batch_size", "t_chunk_size", "ece_bins"):
            value = getattr(self, name)
            # Each count describes a nonempty exact inference budget.
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"evaluation.{name} must be a positive integer.")
        # Empty treatment sets cannot compare any noisy-view ensembles.
        if isinstance(self.horizons, (str, bytes)) or not self.horizons:
            raise ValueError("evaluation.horizons must be a nonempty sequence.")
        horizons = tuple(self.horizons)
        # Horizon indices must remain positive integer schedule boundaries.
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in horizons):
            raise ValueError("evaluation.horizons must contain positive integers.")
        # Repeating a horizon would duplicate the same named treatment.
        if len(set(horizons)) != len(horizons):
            raise ValueError("evaluation.horizons must be distinct.")
        object.__setattr__(self, "horizons", horizons)
        weights = np.asarray(self.head_weights, dtype="float64")
        # Negative or nonfinite coefficients are not probability-mixture weights.
        if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights < 0.):
            raise ValueError("evaluation.head_weights requires three finite nonnegative weights.")
        # Unit-mass normalization requires a finite strictly positive denominator.
        if not np.isfinite(weights.sum()) or weights.sum() <= 0.:
            raise ValueError("evaluation.head_weights must have a finite positive sum.")
        object.__setattr__(self, "head_weights", tuple(float(value) for value in weights))
        # Only existing raw and EMA inference branches are supported.
        if self.network_name not in ("raw", "ema"):
            raise ValueError("evaluation.network_name must be raw or ema.")
        # These are precisely the two existing EnsembleAccuracy execution modes.
        if self.compute_type not in ("chunked", "batched"):
            raise ValueError("evaluation.compute_type must be chunked or batched.")
        # An unseeded treatment could not reproduce its fixed noisy views.
        if self.seed is None:
            raise ValueError("evaluation.seed must be explicit for fixed noisy views.")
        effective_seed(seed=self.seed)
        fraction = float(self.calibration_fraction)
        # A split must retain some rows for evaluation after calibration fitting.
        if not math.isfinite(fraction) or not 0. <= fraction < 1.:
            raise ValueError("evaluation.calibration_fraction must lie in [0, 1).")
        object.__setattr__(self, "calibration_fraction", fraction)
        bounds = tuple(float(value) for value in self.temperature_bounds)
        # Positive bounds must include the unchanged predictor T=1.
        if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds) or not 0. < bounds[0] <= 1. <= bounds[1] or bounds[0] == bounds[1]:
            raise ValueError("evaluation.temperature_bounds must bracket one with positive distinct bounds.")
        object.__setattr__(self, "temperature_bounds", bounds)

    @property
    def normalized_head_weights(self) -> tuple[float, float, float]:
        """Return a nonnegative unit-mass mixture over requested head branches.

        Returns:
            weights (tuple[float, float, float]): Tuple of three Python floats summing to one
                for primary, distillation and regularizer branches.

        Raises:
            None: Construction already requires a finite positive weight sum.
        """

        total = sum(self.head_weights)
        return tuple(value / total for value in self.head_weights)


def _targets(labels: object, count: int, classes: int) -> np.ndarray:
    """Require dense sparse integer targets; never silently truncate labels.

    Args:
        labels (object): Numeric sparse IDs [N] or [N, 1]; finite integer-valued floats are
            accepted without truncation.
        count (int): Exact integer number of expected or selected rows.
        classes (int): Positive integer classifier support width C.

    Returns:
        targets (np.ndarray): Dense int64 vector [N] without fractional truncation or one-
            hot decoding.

    Raises:
        ValueError: If target shape, finiteness, integer values or class support are
            invalid.
    """

    values = np.asarray(labels)
    # Sparse column labels have the same meaning as sparse vectors.
    if values.shape == (count, 1):
        values = values[:, 0]
    # Misaligned or nonfinite labels cannot index a categorical distribution.
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError("Evaluation labels must be finite sparse IDs aligned with samples.")
    # Reject fractional IDs and labels outside the current seen-class support.
    if np.any(values != np.floor(values)) or np.any(values < 0) or np.any(values >= classes):
        raise ValueError("Evaluation labels must be dense integer IDs in the seen classifier support.")
    return values.astype("int64")


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply softmax(log(max(p, 1e-12))/T) to normalized probabilities.

    T=1 preserves the input exactly. Positive T preserves class ordering.
    The numerical floor is declared because exact zeros lack finite logits.

    Args:
        probabilities (np.ndarray): Float-compatible nonnegative matrix [N, C] with unit row
            mass; exact zeros are preserved at T=1.
        temperature (float): Finite positive scalar controlling softmax sharpness; smaller
            values sharpen the distribution.

    Returns:
        scaled (np.ndarray): Float64 normalized matrix [N, C]; T=1 copies exact input zeros
            and other values.

    Raises:
        ValueError: If probabilities are invalid or temperature is nonfinite/nonpositive.
    """

    probs = _probability_matrix(probabilities)
    # Nonpositive temperatures reverse or destroy the class-order interpretation.
    if not math.isfinite(float(temperature)) or temperature <= 0.:
        raise ValueError("temperature must be finite and positive.")
    # Preserve exact input zeros when no calibration transform is requested.
    if temperature == 1.:
        return probs.copy()
    logits = np.log(np.maximum(probs, 1e-12)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(
    probabilities: np.ndarray,
    labels: object,
    *,
    split: str = "validation",
    bounds: tuple[float, float] = (0.05, 20.),
) -> dict[str, object]:
    """Fit one positive scalar using validation NLL only, without model updates.

    NLL is convex in inverse temperature beta: its derivative is the mean
    predicted logit minus the target logit. Bisection finds its bounded global
    minimum (80 iterations); boundary optima are included explicitly. No test
    labels or evaluation metric are consulted. A uniform predictor leaves T=1.

    Args:
        probabilities (np.ndarray): Finite nonnegative prediction matrix [N, C] with unit
            row mass, aligned with sparse labels when supplied.
        labels (object): Sparse integer label vector aligned with the image rows; the label
            convention for this operation is described above.
        split (str): Declared data split; supported training, validation or test access is
            constrained by this operation.
        bounds (tuple[float, float]): Two finite positive temperature bounds bracketing T=1,
            with distinct endpoints.

    Returns:
        fit (dict[str, object]): Dict containing Python float temperature, before/after NLL,
            bounds and validation sample metadata.

    Raises:
        ValueError: If the split is not validation, data are empty/invalid or bounds do not
            bracket one.
    """

    # Test outcomes cannot enter the calibration fitting objective.
    if split != "validation":
        raise ValueError("Temperature fitting is restricted to held-out validation data.")
    checked = EnsembleEvaluationSettings(temperature_bounds=bounds)
    probs = _probability_matrix(probabilities)
    targets = _targets(labels, len(probs), probs.shape[1])
    # Mean validation NLL is undefined for an empty calibration set.
    if not len(probs):
        raise ValueError("Temperature fitting requires nonempty validation data.")
    logits = np.log(np.maximum(probs, 1e-12))
    target_logits = logits[np.arange(len(targets)), targets]

    def derivative(beta: float) -> float:
        """Evaluate the convex NLL derivative with respect to inverse temperature.

        Args:
            beta (float): Positive scalar inverse temperature, beta=1/T, for the convex NLL
                derivative.

        Returns:
            gradient (float): Python float mean NLL derivative with respect to inverse
                temperature.

        Raises:
            None: The enclosing fitter supplies finite logits and bounded positive beta.
        """

        scaled = beta * logits
        scaled -= scaled.max(axis=1, keepdims=True)
        weights = np.exp(scaled)
        weights /= weights.sum(axis=1, keepdims=True)
        return float(np.mean(np.sum(weights * logits, axis=1) - target_logits))

    lower, upper = 1. / checked.temperature_bounds[1], 1. / checked.temperature_bounds[0]
    # A uniform predictor supplies no information for identifying temperature.
    if np.all(np.ptp(logits, axis=1) == 0.):
        beta = 1.
    # An increasing objective throughout its domain is minimized at the left bound.
    elif derivative(lower) >= 0.:
        beta = lower
    # A decreasing objective throughout its domain is minimized at the right bound.
    elif derivative(upper) <= 0.:
        beta = upper
    # Otherwise the monotone derivative crosses zero within the bounds.
    else:
        for _ in range(80):
            midpoint = (lower + upper) / 2.
            # A positive derivative places the minimizer below the midpoint.
            if derivative(midpoint) > 0.:
                upper = midpoint
            # A nonpositive derivative retains the upper half of the interval.
            else:
                lower = midpoint
        beta = (lower + upper) / 2.
    temperature = 1. / beta
    return {
        "temperature": temperature,
        "fit_split": "validation",
        "sample_count": len(probs),
        "class_counts": _class_counts(targets),
        "missing_class_ids": sorted(set(range(probs.shape[1])) - set(targets.tolist())),
        "bounds": list(checked.temperature_bounds),
        "at_boundary": bool(np.isclose(temperature, checked.temperature_bounds).any()),
        "nll_before": calibration_metrics(probs, targets)["nll"],
        "nll_after": calibration_metrics(temperature_scale(probs, temperature), targets)["nll"],
        "log_probability_floor": 1e-12,
    }


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    """Keep reported class composition JSON-safe and explicit.

    Args:
        labels (np.ndarray): Sparse integer label vector aligned with the image rows; the
            label convention for this operation is described above.

    Returns:
        counts (dict[str, int]): Dict mapping string class IDs to integer observation
            counts.

    Raises:
        TypeError: If labels cannot be compared by NumPy unique.
    """

    ids, counts = np.unique(labels, return_counts=True)
    return {str(class_id): int(count) for class_id, count in zip(ids, counts)}


def _validation_partition(labels: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Make disjoint stratified calibration/report rows without singleton reuse.

    Args:
        labels (np.ndarray): Sparse integer label vector aligned with the image rows; the
            label convention for this operation is described above.
        fraction (float): Positive fraction below one used to choose a disjoint calibration
            subset within each class.
        seed (int): Explicit integer random seed; local or derived streams preserve
            reproducibility without reseeding caller-owned generators.

    Returns:
        indices (tuple[np.ndarray, np.ndarray]): (calibration, evaluation): sorted integer
            arrays of disjoint row indices, both containing each represented class.

    Raises:
        ValueError: If any represented class has fewer than two rows or seed is invalid.
    """

    rng = np.random.default_rng(derive_seed(seed, "section10", "validation_partition"))
    calibration, evaluation = [], []
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        # Every represented class needs a distinct fitting and reporting row.
        if len(indices) < 2:
            raise ValueError("Calibration splitting requires at least two validation rows per represented class.")
        indices = rng.permutation(indices)
        count = min(len(indices) - 1, max(1, int(math.floor(len(indices) * fraction))))
        calibration.extend(indices[:count])
        evaluation.extend(indices[count:])
    return np.sort(calibration), np.sort(evaluation)


def _predict(
    wrapper: Any,
    samples: np.ndarray,
    settings: EnsembleEvaluationSettings,
    horizon: int | None,
    stream: str,
    *,
    verbose: bool | int | str = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Predict with existing APIs and account for each classifier invocation.

    Example-forwards count rows processed by predict_class, including the
    C+1 candidate conditions. They are not full denoising trajectories or
    FLOPs. Batch-call count separately exposes vectorization/chunking.
    Latency includes noising, aggregation and synchronized tensor transfers,
    excludes predictor construction/calibration fitting, and has no warm-up.

    Args:
        wrapper (Any): Live diffusion classifier exposing its raw network, class mapping,
            schedules and existing training or inference APIs.
        samples (np.ndarray): Finite numeric image array with shape [N, H, W, C] in the
            saved model-input scale.
        settings (EnsembleEvaluationSettings): Validated settings instance for this
            component; its fields select the behavior described above.
        horizon (int | None): Exclusive positive diffusion timestep bound; None selects
            exactly clean primary-head inference.
        stream (str): Named random-stream component separating calibration and reporting
            draws.
        verbose (bool | int | str): Keras-style progress verbosity: False/0 is quiet,
            True/1 or "auto" updates progress, and 2 prints only completion.

    Returns:
        prediction (tuple[np.ndarray, dict[str, object]]): (probabilities, cost): float64
            [N, C] probabilities and a dict of example-forwards, calls and synchronized
            latency.

    Raises:
        ValueError: If network selection, inputs, outputs or requested ensemble heads are
            invalid.
    """

    network = wrapper.get_network(settings.network_name)
    predictor = None
    # Clean inference bypasses the noisy-view predictor entirely.
    if horizon is not None:
        primary, distillation, regularizer = settings.normalized_head_weights
        predictor = EnsembleAccuracy(
            wrapper, network_name=settings.network_name, compute_type=settings.compute_type,
            weighted=settings.weighted, max_t=horizon, t_chunk_size=settings.t_chunk_size,
            clf_acc_coef=primary, clf_distil_acc_coef=distillation, ctr_acc_coef=regularizer,
            separate_probas=settings.separate_probas, seed=settings.seed,
        )
    chunks = 1 if horizon is None or settings.compute_type == "batched" else math.ceil(horizon / settings.t_chunk_size)
    draws = 1 if horizon is None else settings.noise_draws
    factor = network.num_classes + 1 if horizon is not None and settings.separate_probas else 1
    batches = math.ceil(len(samples) / settings.batch_size)
    probabilities = []
    progress = tf.keras.utils.Progbar(
        len(samples), verbose=1 if verbose == "auto" else int(verbose), unit_name="sample"
    ) if verbose else None
    started = time.perf_counter()
    for start in range(0, len(samples), settings.batch_size):
        batch = tf.convert_to_tensor(samples[start:start + settings.batch_size], dtype=network.compute_dtype)
        # The clean reference makes one unconditional primary prediction per batch.
        if predictor is None:
            zero = tf.zeros((len(batch),), dtype=tf.int32)
            output = network.predict_class((batch, zero, zero), max_encoder_num=None, training=False)
            probability = output.numpy()
        # Each noise draw uses the existing head and timestep aggregation API.
        else:
            draw_probabilities = []
            for draw in range(draws):
                # Horizons share corresponding draws; repeated batches do not.
                predictor.seed = derive_seed(settings.seed, "section10", stream, "batch", start, "draw", draw)
                draw_probabilities.append(predictor.ensemble_predict(batch, training=False).numpy())
            probability = np.mean(np.stack(draw_probabilities).astype("float64"), axis=0)
        # Validate before correcting only floating-point row-sum roundoff.
        probability = _probability_matrix(probability)
        probabilities.append(probability / probability.sum(axis=1, keepdims=True))
        # Quiet evaluation has no progress display to advance.
        if progress is not None:
            progress.update(min(start + settings.batch_size, len(samples)))
    elapsed = time.perf_counter() - started
    return np.concatenate(probabilities), {
        "sample_count": len(samples),
        "batch_size": settings.batch_size,
        "network_forward_calls": batches * draws * chunks,
        "example_forwards": len(samples) * (1 if horizon is None else horizon) * draws * factor,
        "candidate_condition_factor": factor,
        "latency_seconds": elapsed,
        "seconds_per_example": elapsed / len(samples),
        "latency_protocol": "no warmup; prediction/noising/aggregation/host synchronization; excludes setup and temperature fitting",
    }


def evaluate_checkpoint(
    wrapper: Any,
    samples: np.ndarray,
    labels: object,
    settings: EnsembleEvaluationSettings,
    *,
    split: str = "validation",
    calibration_samples: np.ndarray | None = None,
    calibration_labels: object = None,
    calibration_split: str = "validation",
    old_class_count: int | None = None,
) -> dict[str, object]:
    """Compare clean and fixed-cost ensembles using the same selected network.

    Labels must already be dense classifier IDs, without any oracle task
    condition. The network must expose precisely its seen-class support.
    For validation calibration, an internal split guarantees disjoint rows.
    External calibration arrays must originate in separately held-out
    validation data; their provenance/disjointness is the caller's contract.
    Test rows are never passed to fit_temperature, even when temperature
    scaling is requested. Returned dictionaries contain no model or tensors.

    Args:
        wrapper (Any): Live diffusion classifier exposing its raw network, class mapping,
            schedules and existing training or inference APIs.
        samples (np.ndarray): Finite numeric image array with shape [N, H, W, C] in the
            saved model-input scale.
        labels (object): Sparse integer label vector aligned with the image rows; the label
            convention for this operation is described above.
        settings (EnsembleEvaluationSettings): Validated settings instance for this
            component; its fields select the behavior described above.
        split (str): Declared data split; supported training, validation or test access is
            constrained by this operation.
        calibration_samples (np.ndarray | None): Optional separately held-out numeric
            validation images [M, H, W, C] matching evaluation geometry.
        calibration_labels (object): Optional sparse integer validation labels [M], required
            together with calibration_samples.
        calibration_split (str): Provenance label for external calibration arrays; must be
            validation.
        old_class_count (int | None): Optional integer boundary K separating old columns [0,
            K) from newly introduced columns.

    Returns:
        evaluation (dict[str, object]): JSON-compatible dict of clean/ensemble variants,
            float outcomes, calibration and work counts; disabled settings return an empty
            variant list.

    Raises:
        TypeError: If settings are not EnsembleEvaluationSettings.
        ValueError: If data, split, support, EMA, horizon or disjoint calibration
            requirements are violated.
    """

    # Require fully validated controls instead of accepting unchecked mappings.
    if not isinstance(settings, EnsembleEvaluationSettings):
        raise TypeError("settings must be EnsembleEvaluationSettings.")
    # Disabled diagnostics allocate no predictors and perform no model work.
    if not settings.enabled:
        return {"enabled": False, "variants": []}
    # Training rows are not an allowed held-out reliability evaluation split.
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test.")
    # An EMA label must identify actual EMA weights rather than silently using raw.
    if settings.network_name == "ema" and not getattr(wrapper, "use_ema", False):
        raise ValueError("EMA evaluation requires an actual EMA network; raw fallback is not this treatment.")
    network = wrapper.get_network(settings.network_name)
    # Null condition zero is unconditional only for a CFG-enabled network.
    if not getattr(network, "use_cfg", False):
        raise ValueError("Unconditional evaluation requires the existing CFG null label.")
    # Noisy views cannot use schedule entries absent during model construction.
    if max(settings.horizons) > wrapper.timesteps:
        raise ValueError("Evaluation horizons cannot exceed the trained diffusion schedule.")
    seen = getattr(wrapper, "seen_classes", None)
    # Future logits must not enter a class-incremental reliability report.
    if seen and sorted(seen.values()) != list(range(network.num_classes)):
        raise ValueError("Evaluation requires an expanded head containing exactly the dense seen-class support.")
    x = np.asarray(samples)
    # Network inputs need a complete finite image batch including channels.
    if x.ndim != 4 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Evaluation requires nonempty finite image arrays [N,H,W,C].")
    y = _targets(labels, len(x), network.num_classes)
    # Partial calibration input could silently misalign fitting images and targets.
    if (calibration_samples is None) != (calibration_labels is None):
        raise ValueError("Supply calibration_samples and calibration_labels together.")
    calibration_indices, evaluation_indices = None, np.arange(len(x))
    cx, cy = None, None
    # Explicit calibration supports an independent validation set during test scoring.
    if calibration_samples is not None:
        # Caller-declared provenance must identify validation rather than test.
        if calibration_split != "validation":
            raise ValueError("Calibration arrays must be held-out validation data.")
        # Reject direct array reuse; independent provenance remains caller-owned.
        if calibration_samples is samples or np.shares_memory(np.asarray(calibration_samples), x):
            raise ValueError("Calibration and evaluation arrays must be separately held-out, disjoint data.")
        cx = np.asarray(calibration_samples)
        # The same checkpoint requires compatible finite calibration images.
        if cx.ndim != 4 or cx.shape[1:] != x.shape[1:] or not len(cx) or not np.isfinite(cx).all():
            raise ValueError("Calibration images must be nonempty, finite and match evaluation image shape.")
        cy = _targets(calibration_labels, len(cx), network.num_classes)
    # Without external calibration, divide only the supplied validation rows.
    elif settings.calibration_fraction > 0.:
        # Test rows are never repartitioned to create calibration examples.
        if split != "validation":
            raise ValueError("Test temperature scaling requires separate held-out validation arrays.")
        calibration_indices, evaluation_indices = _validation_partition(y, settings.calibration_fraction, settings.seed)
        cx, cy = x[calibration_indices], y[calibration_indices]
        x, y = x[evaluation_indices], y[evaluation_indices]
    variants = []
    for horizon in (None, *settings.horizons):
        probabilities, cost = _predict(wrapper, x, settings, horizon, split)
        record = {
            "name": "clean" if horizon is None else f"ensemble_{horizon}",
            "timesteps": [] if horizon is None else list(range(horizon)),
            "noise_draws": 0 if horizon is None else settings.noise_draws,
            "head_weights": [1., 0., 0.] if horizon is None else list(settings.normalized_head_weights),
            "probability_transform": "primary probability" if horizon is None else (
                "mean over draws of softmax(weighted timestep mean of null plus candidate diagonal)"
                if settings.separate_probas else "unit-mass head and timestep mixture; mean over draws"
            ),
            "metrics": calibration_metrics(probabilities, y, bins=settings.ece_bins),
            "evaluation_cost": cost,
            "temperature_fit": None,
        }
        # A supplied old-class boundary enables retention and plasticity decomposition.
        if old_class_count is not None:
            from semantic_consolidation.experimental import classification_outcomes
            record["class_outcomes"] = classification_outcomes(probabilities, y, old_class_count, settings.ece_bins)
        # Fit one separate temperature per prespecified inference treatment.
        if cx is not None:
            calibration_probabilities, calibration_cost = _predict(wrapper, cx, settings, horizon, "calibration")
            started = time.perf_counter()
            fitted = fit_temperature(calibration_probabilities, cy, split="validation", bounds=settings.temperature_bounds)
            fitted["fit_seconds"] = time.perf_counter() - started
            record["temperature_fit"] = fitted
            record["calibration_cost"] = calibration_cost
            record["calibrated_metrics"] = calibration_metrics(
                temperature_scale(probabilities, fitted["temperature"]), y, bins=settings.ece_bins,
            )
        variants.append(record)
    serialized_settings = asdict(settings)
    for name in ("horizons", "head_weights", "temperature_bounds"):
        serialized_settings[name] = list(serialized_settings[name])
    return {
        "enabled": True,
        "split": split,
        "settings": serialized_settings,
        "seen_class_count": int(network.num_classes),
        "evaluation_class_counts": _class_counts(y),
        "evaluation_missing_class_ids": sorted(set(range(network.num_classes)) - set(y.tolist())),
        "evaluation_indices": evaluation_indices.tolist(),
        "calibration_indices": None if calibration_indices is None else calibration_indices.tolist(),
        "calibration_source": None if cx is None else ("validation partition" if calibration_indices is not None else "caller-supplied held-out validation"),
        "ece_binning": f"{settings.ece_bins} equal-width confidence bins",
        "uncertainty_interpretation": "Predictive entropy is descriptive; no epistemic calibration claim.",
        "variants": variants,
    }
