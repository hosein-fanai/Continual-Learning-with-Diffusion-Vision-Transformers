"""Matched-view replay selection shared by the two optional thesis routes.

Drift is the mean of per-view Jensen--Shannon divergences in nats, *not* the
divergence of view-averaged predictions. The old teacher is zero-padded onto
the student's dense class support; new-class probability is never discarded.
Selection runs after real current-data updates against the prior-task teacher.
This module does not manufacture drift using a fresh copy of the student.

The quality gate measures teacher label consistency, not perceptual fidelity.
Its training-candidate quantile is a self-consistency heuristic. A threshold
chosen with independent validation information must be declared separately.
MIR uses a reversible actual optimizer update and the subsequent increase in
candidate classification loss; it is separate from accumulated drift.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time

import numpy as np


@dataclass
class ReplaySelectionSettings:
    """Finite selection budgets and declared, reproducible scoring controls."""

    strategy: str = "drift"
    candidate_multiplier: int = 2
    batch_size: int = 32
    noise_levels: tuple[int, ...] = (0,)
    quality_quantile: float = 0.0
    quality_threshold: float | None = None
    quality_threshold_split: str = "validation"
    class_coverage: bool = True
    min_per_class: int = 1
    quota_policy: str = "class_floor"

    def __post_init__(self) -> None:
        """Normalize scalar/sequence controls and reject ambiguous protocols.

        Returns:
            validated (None): None; normalizes views and scalar quality settings in place.

        Raises:
            ValueError: If strategy, quotas, seed-independent budgets, quality controls or view
                indices are invalid.
        """

        # Reject names that could silently select a different scientific control.
        if self.strategy not in {"drift", "random", "confidence", "label_surprisal", "mir"}:
            raise ValueError("Replay strategy must be drift, random, confidence, label_surprisal, or mir.")
        for name in ("candidate_multiplier", "batch_size", "min_per_class"):
            value = getattr(self, name)
            # Generation and scoring must both have positive integer bounds.
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"replay.{name} must be a positive integer.")
        # A string or empty view sequence cannot define a noise average.
        if isinstance(self.noise_levels, (str, bytes)) or not self.noise_levels:
            raise ValueError("replay.noise_levels must contain nonnegative integer indices.")
        self.noise_levels = tuple(self.noise_levels)
        # Schedule indices are discrete and zero is reserved for the clean view.
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in self.noise_levels):
            raise ValueError("replay.noise_levels must contain nonnegative integer indices.")
        # Repeated levels would silently change the declared weighting of views.
        if len(set(self.noise_levels)) != len(self.noise_levels):
            raise ValueError("Duplicate replay noise levels are unsupported.")
        self.quality_quantile = float(self.quality_quantile)
        # Quantiles are defined only on the closed unit interval.
        if not math.isfinite(self.quality_quantile) or not 0 <= self.quality_quantile <= 1:
            raise ValueError("replay.quality_quantile must lie in [0,1].")
        # An externally fitted threshold bypasses the candidate quantile fit.
        if self.quality_threshold is not None:
            self.quality_threshold = float(self.quality_threshold)
            # The gate compares a probability, so its threshold has the same range.
            if not math.isfinite(self.quality_threshold) or not 0 <= self.quality_threshold <= 1:
                raise ValueError("replay.quality_threshold must lie in [0,1].")
        # Test information may never tune the replay quality gate.
        if self.quality_threshold_split not in {"training", "validation"}:
            raise ValueError("Quality thresholds may only use training or validation information.")
        # Require an explicit coverage treatment instead of truthy configuration text.
        if not isinstance(self.class_coverage, bool):
            raise ValueError("replay.class_coverage must be boolean.")
        # Fixed quotas isolate within-class ranking from changes in class allocation.
        if self.quota_policy not in {"class_floor", "fixed_per_class"}:
            raise ValueError("replay.quota_policy must be class_floor or fixed_per_class.")
        # Exact positive class quotas cannot simultaneously disable coverage.
        if self.quota_policy == "fixed_per_class" and not self.class_coverage:
            raise ValueError("Fixed per-class quotas require class_coverage=true.")


def fixed_class_quotas(budget: int, old_classes: list[int], seed: int, min_per_class: int = 1) -> dict[int, int]:
    """Assign equal integer quotas, then a seeded, score-independent remainder.

    Sorted dense class IDs are permuted using ``seed``; the first B modulo K
    receive one extra row. The caller records this seed and reuses it across
    ranking treatments. Positive budgets must cover every class's requested
    minimum, including the semantic route's two-row positive-pair minimum.

    Args:
        budget (int): Exact nonnegative integer number of distinct candidate rows to retain.
        old_classes (list[int]): Declared nonnegative dense old-class IDs; replay candidates
            must belong to this support.
        seed (int): Explicit integer random seed; local or derived streams preserve
            reproducibility without reseeding caller-owned generators.
        min_per_class (int): Positive integer minimum retained row count per covered class.

    Returns:
        quotas (dict[int, int]): Dict from integer old-class ID to integer retained quota; a
            zero budget gives zero quotas.

    Raises:
        ValueError: If budgets, seed, class IDs or positive per-class feasibility are
            invalid.
    """

    # Reject ambiguous numeric input before division or random permutation.
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < 0:
        raise ValueError("Fixed-quota budget must be a nonnegative integer.")
    # Positive row floors must be exact integer counts.
    if isinstance(min_per_class, bool) or not isinstance(min_per_class, (int, np.integer)) or min_per_class < 1:
        raise ValueError("Fixed-quota min_per_class must be a positive integer.")
    # Match the selector's seed domain for a reproducible remainder stream.
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2 ** 32:
        raise ValueError("Fixed-quota seed must be an integer in [0,2**32).")
    values = np.asarray(old_classes)
    classes = sorted(set(_labels(values, len(values)).tolist()))
    # Empty old-class support is meaningful only before any replay is needed.
    if not classes:
        # Positive replay cannot be assigned without a class vocabulary.
        if budget:
            raise ValueError("A positive fixed-quota budget requires old classes.")
        return {}
    base, remainder = divmod(int(budget), len(classes))
    # No class's quota may be redistributed to hide an insufficient row floor.
    if budget and base < min_per_class:
        raise ValueError(
            f"Fixed per-class quotas require budget >= K * min_per_class; "
            f"budget={budget}, K={len(classes)}, min_per_class={min_per_class}."
        )
    quotas = {class_id: base for class_id in classes}
    for class_id in np.random.default_rng(int(seed)).permutation(classes)[:remainder]:
        quotas[int(class_id)] += 1
    return quotas


def _candidate_identities(images: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> dict:
    """Bind row ordinals to exact candidate contents, retaining duplicate pixels.

    Pixel duplicates may legitimately be generated. Their content hashes can
    coincide, but their pool-hash/ordinal identities remain distinct. Hashing
    records dtype and shape, and canonical dense int64 labels, in C order.

    Args:
        images (np.ndarray): Numeric sample-major images in the configured model-input
            scale, normally float32 NHWC values in [-1, 1].
        labels (np.ndarray): Sparse integer label vector aligned with the image rows; the
            label convention for this operation is described above.
        indices (np.ndarray): Integer row ordinals identifying selected occurrences in the
            supplied candidate pool.

    Returns:
        identities (dict): Dict of pool/content SHA-256 strings and selected occurrence
            identities; duplicate pixels retain distinct ordinals.

    Raises:
        IndexError: If a selected row ordinal is outside the supplied pool.
    """

    image_header = json.dumps({"dtype": images.dtype.str, "shape": list(images.shape)}, sort_keys=True).encode("utf-8")
    pool_hash = hashlib.sha256(image_header + images.tobytes(order="C") + labels.tobytes(order="C")).hexdigest()
    row_hashes = [hashlib.sha256(np.asarray(row).tobytes(order="C") + labels[index:index + 1].tobytes()).hexdigest()
                  for index, row in enumerate(images)]
    row_ids = [f"{pool_hash}:{index}" for index in range(len(images))]
    return {
        "candidate_identity_sha256": pool_hash,
        "candidate_identity_encoding": "UTF8 sorted JSON image dtype/shape + C-order pixels + dense int64 labels; row ID=pool hash:ordinal",
        "candidate_row_ids": row_ids, "candidate_row_sha256": row_hashes,
        "selected_row_ids": [row_ids[index] for index in indices],
        "selected_row_sha256": [row_hashes[index] for index in indices],
        "selection_identity_sha256": hashlib.sha256("\n".join(row_ids[index] for index in indices).encode("utf-8")).hexdigest(),
    }


def probability_rows(values: np.ndarray) -> np.ndarray:
    """Normalize finite, nonnegative prediction rows without clipping zeros.

    Args:
        values (np.ndarray): Finite nonnegative numeric matrix [N, C] with C positive and
            positive mass in every row.

    Returns:
        probabilities (np.ndarray): Float64 row-normalized matrix preserving exact zeros and
            avoiding finite-input sum overflow.

    Raises:
        ValueError: If inputs are not a matrix with positive class width, contain invalid
            mass or include an all-zero row.
    """

    probabilities = np.asarray(values, dtype="float64")
    # Divergence requires a matrix with a nonempty class support.
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise ValueError("Predictions must have shape [examples, positive class count].")
    # Invalid scores cannot be interpreted as probability mass.
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Predictions must be finite and nonnegative.")
    # Scale first to avoid overflow for finite, unnormalized head mixtures.
    maximum = probabilities.max(axis=1, keepdims=True)
    # All-zero rows do not define a probability distribution.
    if np.any(maximum <= 0):
        raise ValueError("Every prediction must have positive total mass.")
    scaled = np.divide(probabilities, maximum, out=np.zeros_like(probabilities), where=maximum > 0)
    return scaled / scaled.sum(axis=1, keepdims=True)


def padded_jensen_shannon(teacher: np.ndarray, student: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rowwise JS in [0,log(2)] and student new-class probability.

    Class columns must follow common's dense append-only class mapping. If
    teacher support has width K and student width L>=K, the comparison is
    JS((p_1,...,p_K,0,...,0), q). Zero times log zero is defined as zero.

    Args:
        teacher (np.ndarray): Teacher probability matrix [N, K] on old dense class support;
            rows may be unnormalized positive masses.
        student (np.ndarray): Current prediction matrix [N, L], where L is at least the
            teacher class width.

    Returns:
        divergences (tuple[np.ndarray, np.ndarray]): (js, invasion): float64 vectors [N] of
            JS in [0, log(2)] and current probability on new-class columns.

    Raises:
        ValueError: If probability rows are invalid, row counts differ or teacher support
            exceeds student support.
    """

    prior, current = probability_rows(teacher), probability_rows(student)
    # Dense class mappings are append-only and candidates remain row-matched.
    if len(prior) != len(current) or prior.shape[1] > current.shape[1]:
        raise ValueError("Teacher/student batches must match and class support may only grow.")
    old_width = prior.shape[1]
    prior = np.pad(prior, ((0, 0), (0, current.shape[1] - old_width)))
    midpoint = (prior + current) / 2.

    def divergence(distribution: np.ndarray) -> np.ndarray:
        """Compute KL to the midpoint using the zero-mass limiting convention.

        Args:
            distribution (np.ndarray): Normalized float64 probabilities [N, L] on the common
                padded support.

        Returns:
            kl (np.ndarray): Float64 row vector [N] of KL divergence to the captured midpoint,
                using zero-times-log-zero equals zero.

        Raises:
            None: The enclosing function supplies validated normalized rows on matching support.
        """

        ratio = np.divide(distribution, midpoint, out=np.ones_like(distribution), where=distribution > 0)
        return np.sum(distribution * np.log(ratio), axis=1)

    js = (divergence(prior) + divergence(current)) / 2.
    return np.clip(js, 0., np.log(2.)), current[:, old_width:].sum(axis=1)


def _labels(values: np.ndarray, count: int) -> np.ndarray:
    """Validate dense replay IDs, including an explicitly empty vector.

    Args:
        values (np.ndarray): One-dimensional sparse nonnegative integer IDs with exactly
            count entries; an empty sequence is accepted.
        count (int): Exact integer number of expected or selected rows.

    Returns:
        labels (np.ndarray): Int64 vector of dense replay IDs, with an empty vector allowed
            when count is zero.

    Raises:
        ValueError: If labels are misaligned, not one-dimensional, noninteger or negative.
    """

    labels = np.asarray(values)
    # Empty Python sequences otherwise acquire a misleading floating dtype.
    if labels.ndim == 1 and count == 0 and not len(labels):
        return labels.astype("int64")
    # Label IDs must match candidate rows without implicit one-hot decoding.
    if labels.ndim != 1 or len(labels) != count or labels.dtype.kind not in "iu" or np.any(labels < 0):
        raise ValueError("Candidate labels must be aligned nonnegative dense integer IDs.")
    return labels.astype("int64", copy=False)


def _view(wrapper: object, images: object, level: int, seed: int) -> tuple:
    """Use common's forward diffusion with stateless noise and a clean zero view.

    Args:
        wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
            schedules and existing training or inference APIs.
        images (object): Numeric sample-major images in the configured model-input scale,
            normally float32 NHWC values in [-1, 1].
        level (int): Nonnegative integer noise level; zero means exactly clean and positive
            values index the diffusion schedule.
        seed (int): Explicit integer random seed; local or derived streams preserve
            reproducibility without reseeding caller-owned generators.

    Returns:
        view (tuple): (images, times): original floating image dtype and tf.int32 timestep
            vector; positive levels use stateless forward noise.

    Raises:
        ValueError: If a positive level does not index the wrapper diffusion schedule.
    """

    import tensorflow as tf

    # Clean controls bypass the potentially noisy schedule entry at timestep zero.
    if level == 0:
        clean, _, times = wrapper.noisify(images, min_timesteps=0, max_timesteps=0, seed=seed)
        return clean, times
    # Positive noising levels must index the existing diffusion schedule.
    if not 0 < level < int(wrapper.timesteps):
        raise ValueError("Replay noise level must index the wrapper's diffusion schedule.")
    times = tf.fill((tf.shape(images)[0],), tf.cast(level, tf.int32))
    noise = tf.random.stateless_normal(tf.shape(images), seed=(int(seed) % (2 ** 31 - 1), level), dtype=images.dtype)
    return wrapper.q_sample(images, times, noise), times


def _predict(network: object, images: object, times: object) -> np.ndarray:
    """Use the existing raw primary classifier under the null CFG condition.

    Args:
        network (object): Raw classifier network exposing predict_class and the existing
            primary classifier layers.
        images (object): Numeric sample-major images in the configured model-input scale,
            normally float32 NHWC values in [-1, 1].
        times (object): Integer diffusion timestep vector of shape [N], normally tf.int32.

    Returns:
        probabilities (np.ndarray): Float64 normalized [N, C] primary-head probabilities
            under null conditioning.

    Raises:
        ValueError: If raw output rows are nonfinite, negative, empty in class support or
            have zero mass.
    """

    import tensorflow as tf

    return probability_rows(np.asarray(network.predict_class(
        (images, times, tf.zeros_like(times)), max_encoder_num=None, training=False,
    )))


def _check_fixed_inference(network: object) -> None:
    """Reject the platform's latent samplers, which also sample at inference.

    Args:
        network (object): Raw classifier network exposing predict_class and the existing
            primary classifier layers.

    Returns:
        validated (None): None; accepts deterministic latent paths.

    Raises:
        ValueError: If a flattening path enables variational inference-time sampling.
    """

    for prefix in ("", "clf_"):
        reshapers = getattr(network, f"{prefix}reshaper_ids_dict", {}) or {}
        options = getattr(network, f"{prefix}reshaper_kwargs", {}) or {}
        # Variational flattening samples even when predict_class uses training=False.
        if "flatten" in reshapers.values() and options.get("add_kl", False):
            raise ValueError("Fixed-view replay scoring requires deterministic latent features; disable variational flattening.")


class DriftReplaySelector:
    """Score bounded fixed candidates and rotate coverage over selection calls.

    Reuse one instance for repeated replay selections within an increment.
    Each scheduled class receives ``min_per_class`` distinct floor rows.
    When the budget is smaller than K times that floor, coverage rotates over
    ceil(K/floor(budget/min_per_class)) calls with fixed candidates. Any leftover
    slots rank globally and do not replace those guaranteed complete blocks.
    Semantic phases requiring positives in every represented class must use a
    full-size retained pool (budget >= K*min_per_class), as the route validates.
    The quality-filtered pool must supply each class's floor when requested.
    Infeasible budgets or coverage fail explicitly instead of weakening the
    threshold, repeating low-quality rows, or implying unavailable coverage.
    Optional ``fixed_per_class`` quotas instead allocate all class counts before
    ranking; they cannot rotate or globally redistribute any remainder slots.
    """

    def __init__(self, settings: ReplaySelectionSettings, seed: int) -> None:
        """Create task-local threshold state and a reproducible coverage cursor.

        Args:
            settings (ReplaySelectionSettings): Validated settings instance for this component;
                its fields select the behavior described above.
            seed (int): Explicit integer random seed; local or derived streams preserve
                reproducibility without reseeding caller-owned generators.

        Returns:
            initialized (None): None; stores validated settings, fixed threshold metadata and an
                initial coverage cursor.

        Raises:
            TypeError: If settings are not ReplaySelectionSettings.
            ValueError: If the explicit seed is outside [0, 2**32) or not an integer.
        """

        # Keep field validation in the shared dataclass instead of duck typing.
        if not isinstance(settings, ReplaySelectionSettings):
            raise TypeError("settings must be ReplaySelectionSettings.")
        # Stateless TensorFlow views need a concrete, bounded seed.
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2 ** 32:
            raise ValueError("Replay seed must be an integer in [0,2**32).")
        self.settings = settings
        self.seed = int(seed)
        self.selection_calls = 0
        self.coverage_cursor = 0
        self.threshold = settings.quality_threshold
        self.threshold_record = {
            "threshold": self.threshold, "source": "configured" if self.threshold is not None else "unfitted",
            "split": settings.quality_threshold_split if self.threshold is not None else None,
            "criterion": "teacher_conditioning_label_probability",
        }

    def score(self, wrapper: object, images: np.ndarray, labels: np.ndarray, teacher: object = None) -> dict:
        """Measure prior-task/current drift on exactly matched fixed image/noise views.

        Inputs use model diffusion space and common's dense class IDs. Clean
        teacher confidence/surprisal remain the same control definitions as
        common.mechanistic.select_replay_candidates. The full scorer runs for
        every strategy, keeping ordinary selection-control scoring work equal.
        Seeds depend on batch position and level, so report fixed batch size and
        candidate order; invariance to repartitioning examples is not claimed.

        Args:
            wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
                schedules and existing training or inference APIs.
            images (np.ndarray): Numeric sample-major images in the configured model-input
                scale, normally float32 NHWC values in [-1, 1].
            labels (np.ndarray): Sparse integer label vector aligned with the image rows; the
                label convention for this operation is described above.
            teacher (object): Independent frozen prior-task raw network; None uses
                wrapper.teacher_network when this operation accepts a wrapper.

        Returns:
            scores (dict): Dict of float64 per-row drift, invasion, confidence, label
                probability/surprisal, student NLL and measured scoring work.

        Raises:
            ValueError: If images, class IDs, teacher independence, deterministic inference or
                predictions are invalid.
        """

        import tensorflow as tf
        from common.runtime import derive_seed

        x = np.asarray(images, dtype="float32")
        y = _labels(labels, len(x))
        # Nonfinite generated pixels invalidate every probability diagnostic.
        if not np.isfinite(x).all():
            raise ValueError("Replay images must be finite.")
        prior = wrapper.teacher_network if teacher is None else teacher
        # A live or missing target cannot measure accumulated historical change.
        if prior is None or prior is wrapper.network or getattr(prior, "trainable", True):
            raise ValueError("Drift requires an independent frozen prior-task teacher.")
        _check_fixed_inference(prior)
        _check_fixed_inference(wrapper.network)
        old_width = int(prior.num_classes)
        # The frozen teacher must represent every candidate's conditioning label.
        if np.any(y >= old_width):
            raise ValueError("Replay conditioning labels must belong to prior-task classes.")
        started = time.perf_counter()
        drifts, invasions, clean_predictions, student_losses = [], [], [], []
        teacher_forwards = student_forwards = 0
        for start in range(0, len(x), self.settings.batch_size):
            batch = tf.convert_to_tensor(x[start:start + self.settings.batch_size])
            batch_y = y[start:start + len(batch)]
            per_view_js, per_view_invasion, per_view_loss = [], [], []
            clean_teacher = None
            for level in self.settings.noise_levels:
                noised, times = _view(wrapper, batch, level, derive_seed(self.seed, "replay_view", start, level))
                p, q = _predict(prior, noised, times), _predict(wrapper.network, noised, times)
                js, invasion = padded_jensen_shannon(p, q)
                per_view_js.append(js)
                per_view_invasion.append(invasion)
                per_view_loss.append(-np.log(np.maximum(q[np.arange(len(batch_y)), batch_y], 1e-12)))
                teacher_forwards += len(batch)
                student_forwards += len(batch)
                # Reuse the clean view for common's confidence/surprisal controls.
                if level == 0:
                    clean_teacher = p
            # Noise-only drift still compares controls using clean teacher predictions.
            if clean_teacher is None:
                clean_teacher = _predict(prior, batch, tf.zeros((len(batch),), tf.int32))
                teacher_forwards += len(batch)
            drifts.append(np.mean(per_view_js, axis=0))
            invasions.append(np.mean(per_view_invasion, axis=0))
            student_losses.append(np.mean(per_view_loss, axis=0))
            clean_predictions.append(clean_teacher)
        probabilities = np.concatenate(clean_predictions) if len(x) else np.empty((0, old_width))
        label_probability = probabilities[np.arange(len(y)), y]
        return {
            "drift": np.concatenate(drifts) if len(x) else np.empty(0),
            "new_class_invasion": np.concatenate(invasions) if len(x) else np.empty(0),
            "confidence": probabilities.max(axis=1),
            "label_surprisal": -np.log(np.maximum(label_probability, 1e-12)),
            "teacher_label_probability": label_probability,
            "teacher_probabilities": probabilities,
            "student_label_loss": np.concatenate(student_losses) if len(x) else np.empty(0),
            "diagnostics": {
                "definition": "mean_of_per_view_zero_padded_JS", "units": "nats",
                "conditioning": "same_null_CFG_ID", "noise_levels": list(self.settings.noise_levels),
                "seed": self.seed, "batch_size": self.settings.batch_size,
                "teacher_example_forwards": teacher_forwards, "student_example_forwards": student_forwards,
                "noisy_image_draws": len(x) * sum(level > 0 for level in self.settings.noise_levels),
                "seconds": time.perf_counter() - started,
            },
        }

    def fit_quality_threshold(
        self, wrapper: object, images: np.ndarray, labels: np.ndarray,
        split: str = "training", teacher: object = None, scored: dict | None = None,
    ) -> dict:
        """Fit a teacher-label probability quantile using training/validation only.

        Repeated fitting is rejected: the threshold must remain fixed throughout
        an increment. Supplying a configured threshold records its provenance
        without silently fitting another value. Training candidate labels are
        generated condition IDs, so this is explicitly self-labeled consistency.

        Args:
            wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
                schedules and existing training or inference APIs.
            images (np.ndarray): Numeric sample-major images in the configured model-input
                scale, normally float32 NHWC values in [-1, 1].
            labels (np.ndarray): Sparse integer label vector aligned with the image rows; the
                label convention for this operation is described above.
            split (str): Declared data split; supported training, validation or test access is
                constrained by this operation.
            teacher (object): Independent frozen prior-task raw network; None uses
                wrapper.teacher_network when this operation accepts a wrapper.
            scored (dict | None): Per-candidate score mapping from score(); quality and ranking
                vectors must align with the supplied rows.

        Returns:
            threshold_record (dict): Dict describing the fixed Python float quality threshold
                and training/validation provenance.

        Raises:
            ValueError: If the fitting split is test or aligned finite probability scores are
                unavailable.
            RuntimeError: If an already-fitted quantile is fitted again within the increment.
        """

        # Quality calibration cannot inspect the test split.
        if split not in {"training", "validation"}:
            raise ValueError("Quality fitting may not use test data.")
        # Keep the quality rule fixed while the current learner changes.
        if self.threshold_record["source"] == "fitted_quantile":
            raise RuntimeError("Fit the replay quality threshold only once per increment.")
        # Respect an explicitly configured threshold and its declared source split.
        if self.threshold is not None:
            return dict(self.threshold_record)
        scored = self.score(wrapper, images, labels, teacher) if scored is None else scored
        values = np.asarray(scored["teacher_label_probability"], dtype="float64")
        # A quantile needs aligned, nonempty, valid label probabilities.
        if values.ndim != 1 or not len(values) or len(values) != len(images) or not np.isfinite(values).all() \
        or np.any((values < 0) | (values > 1)):
            raise ValueError("Quality fitting requires finite, aligned, nonempty training/validation scores.")
        self.threshold = float(np.quantile(values, self.settings.quality_quantile))
        self.threshold_record = {
            "threshold": self.threshold, "source": "fitted_quantile", "split": split,
            "quantile": self.settings.quality_quantile, "examples": len(values),
            "criterion": "teacher_conditioning_label_probability",
            "interpretation": "teacher_self_consistency" if split == "training" else "validation_label_consistency",
        }
        return dict(self.threshold_record)

    def select(
        self, images: np.ndarray, labels: np.ndarray, budget: int,
        old_classes: list[int], scored: dict, strategy: str | None = None,
        interference: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Filter, apply the declared class allocation, and rank distinct rows.

        A failed quality gate never silently backfills the budget. The selected
        pool has exactly ``budget`` distinct candidate rows or selection raises
        with the available counts. Smaller retained budgets use rotating complete
        class-floor blocks across calls; generation must still cover all classes.
        Candidate generation remains the existing learner's responsibility.
        Fixed quotas use a separate, task-stable seed for class remainders;
        ranking ties retain the existing seeded strategy behavior.

        Args:
            images (np.ndarray): Numeric sample-major images in the configured model-input
                scale, normally float32 NHWC values in [-1, 1].
            labels (np.ndarray): Sparse integer label vector aligned with the image rows; the
                label convention for this operation is described above.
            budget (int): Exact nonnegative integer number of distinct candidate rows to retain.
            old_classes (list[int]): Declared nonnegative dense old-class IDs; replay candidates
                must belong to this support.
            scored (dict): Per-candidate score mapping from score(); quality and ranking vectors
                must align with the supplied rows.
            strategy (str | None): Optional ranking control overriding the validated default:
                drift, random, confidence, label_surprisal or mir.
            interference (np.ndarray | None): Optional finite float vector [N] of candidate NLL
                increases after one real virtual update; required for MIR.

        Returns:
            selection (tuple[np.ndarray, np.ndarray, dict]): (images, labels, audit): retained
                image dtype unchanged, dense int64 IDs and JSON-compatible quality/quota/score
                metadata.

        Raises:
            ValueError: If class IDs, budget, scores, quality feasibility, quotas or MIR
                interference are invalid.
            RuntimeError: If the threshold is unfitted or an exact quota invariant fails.
        """

        from common.mechanistic import select_replay_candidates
        from common.runtime import derive_seed

        x = np.asarray(images)
        y = _labels(labels, len(x))
        # Selection operates on an exact integer row budget, including zero.
        if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < 0:
            raise ValueError("Replay retained budget must be a nonnegative integer.")
        expected_array = np.asarray(old_classes)
        expected = sorted(set(_labels(expected_array, len(expected_array)).tolist()))
        # Declared coverage classes must include every generated candidate class.
        if set(y) - set(expected):
            raise ValueError("Candidate labels must belong to the declared old classes.")
        strategy = self.settings.strategy if strategy is None else strategy
        # Per-call controls obey the same vocabulary as configured strategies.
        if strategy not in {"drift", "random", "confidence", "label_surprisal", "mir"}:
            raise ValueError("Unknown replay strategy.")
        quality = np.asarray(scored["teacher_label_probability"], dtype="float64")
        # Quality filtering must have one valid probability for every candidate.
        if quality.shape != (len(x),) or not np.isfinite(quality).all() or np.any((quality < 0) | (quality > 1)):
            raise ValueError("Candidate quality probabilities are invalid.")
        for name in ("drift", "new_class_invasion", "confidence", "label_surprisal"):
            values = np.asarray(scored[name])
            # Validate diagnostics before advancing the persistent coverage cursor.
            if values.shape != (len(x),) or not np.isfinite(values).all():
                raise ValueError(f"Invalid replay diagnostic scores: {name}.")
        # Do not silently learn a new threshold at every post-wake selection.
        if self.threshold is None:
            raise RuntimeError("Fit or explicitly configure the quality threshold before selecting replay.")
        call_seed = derive_seed(self.seed, "replay_rank", self.selection_calls)
        rng = np.random.default_rng(call_seed)
        eligible = quality >= self.threshold
        available_classes = [class_id for class_id in expected if np.any(y == class_id)]
        floor = self.settings.min_per_class
        quotas = None
        quota_seed = None
        eligible_class_counts = {class_id: int(np.sum(eligible & (y == class_id))) for class_id in expected}
        # Quota allocation never reads scores or the selection-call counter.
        if self.settings.quota_policy == "fixed_per_class":
            quota_seed = derive_seed(self.seed, "replay_fixed_quotas")
            quotas = fixed_class_quotas(budget, expected, quota_seed, floor)
            deficits = {class_id: {"target": target, "eligible": eligible_class_counts[class_id]}
                        for class_id, target in quotas.items() if eligible_class_counts[class_id] < target}
            # Keep quality unchanged and fail before ranking or mutating state.
            if deficits:
                raise ValueError(
                    f"Fixed per-class quotas are infeasible after quality screening: {deficits}; "
                    f"target quotas={quotas}, eligible counts={eligible_class_counts}. "
                    "No redistribution, quality relaxation, or repeated rows is permitted."
                )
        # A positive selection must fit at least one complete class-floor block.
        if self.settings.class_coverage and 0 < budget < floor:
            raise ValueError(f"Retained budget {budget} cannot fit min_per_class={floor}; increase the budget.")
        # Exact replay budgets may not be met by repeating rejected candidates.
        if int(np.sum(eligible)) < budget:
            raise ValueError(
                f"Quality threshold {self.threshold:g} leaves {int(np.sum(eligible))} eligible "
                f"candidates for retained budget {budget}; adjust the training/validation threshold or pool budget."
            )
        missing_eligible = [class_id for class_id in expected if eligible_class_counts[class_id] < floor]
        # Quality and class coverage must both hold; an empty class is infeasible.
        if self.settings.class_coverage and budget and missing_eligible:
            raise ValueError(
                f"Class coverage is infeasible: old classes {missing_eligible} have fewer than "
                f"min_per_class={floor} quality-eligible candidates; eligible counts={eligible_class_counts}. "
                "Increase the candidate pool or adjust the threshold using training/validation only."
            )
        if strategy in {"random", "confidence", "label_surprisal"}:
            # Preserve the common API's clean-teacher control definitions and
            # tie/random ranking behavior. Row IDs are its sample payload.
            ranked, _, _ = select_replay_candidates(
                np.arange(len(x)), y, len(x),
                strategy="surprise" if strategy == "label_surprisal" else strategy,
                probabilities=scored["teacher_probabilities"], seed=call_seed,
            )
            ranked = np.asarray(ranked, dtype="int64")
        # Drift and MIR have distinct externally computed candidate scores.
        else:
            values = np.asarray(interference if strategy == "mir" else scored["drift"], dtype="float64")
            # MIR must supply real virtual-update losses rather than a proxy score.
            if values.shape != (len(x),) or not np.isfinite(values).all():
                raise ValueError("Drift/MIR requires one finite score per candidate; MIR needs a real virtual update.")
            ties = rng.permutation(len(x))
            ranked = ties[np.argsort(-values[ties], kind="stable")]
        ranked = ranked[eligible[ranked]]
        selected = []
        covered = []
        classes_per_call = int(budget) // floor
        # Reserve complete class-floor blocks before filling slots by global rank.
        if quotas is not None:
            covered = [class_id for class_id, quota in quotas.items() if quota]
            for class_id, quota in quotas.items():
                selected.extend(ranked[y[ranked] == class_id][:quota].astype("int64").tolist())
        # The default policy retains its original rotating floors and global fill.
        elif self.settings.class_coverage and available_classes and budget:
            order = available_classes[self.coverage_cursor:] + available_classes[:self.coverage_cursor]
            covered = order[:min(classes_per_call, len(order))]
            for class_id in covered:
                selected.extend(ranked[y[ranked] == class_id][:floor].astype("int64").tolist())
            self.coverage_cursor = (self.coverage_cursor + len(covered)) % len(available_classes)
        selected_set = set(selected)
        for index in ranked:
            # Stop at the declared retained budget even if more candidates pass.
            if len(selected) >= budget:
                break
            # Floor-selected rows must not be repeated in the global-rank fill.
            if int(index) not in selected_set:
                selected.append(int(index))
                selected_set.add(int(index))
        indices = np.asarray(selected, dtype="int64")
        # Strict quota failures cannot silently become undersized or duplicate pools.
        if quotas is not None and (len(indices) != budget or len(set(selected)) != budget):
            raise RuntimeError("Fixed per-class quota selection did not return the exact distinct-row budget.")
        self.selection_calls += 1
        counts = {str(class_id): int(np.sum(y[indices] == class_id)) for class_id in expected}
        diagnostics = {
            "strategy": strategy, "selection_call": self.selection_calls,
            "candidate_count": len(x), "requested_count": int(budget), "selected_count": len(indices),
            "candidate_multiplier": self.settings.candidate_multiplier,
            "actual_candidate_to_budget_ratio": len(x) / int(budget) if budget else None,
            "candidate_pool_enlarged": len(x) > int(budget),
            "quality": dict(self.threshold_record), "quality_pass_count": int(np.sum(quality >= self.threshold)),
            "quality_floor_exception_indices": [],
            "undersized_by": max(0, int(budget) - len(indices)),
            "class_counts": counts, "class_coverage": self.settings.class_coverage,
            "min_per_class": floor,
            "class_floor_ids_this_call": covered,
            "unavailable_candidate_classes": sorted(set(expected) - set(available_classes)),
            "coverage_window_calls": math.ceil(len(available_classes) / classes_per_call)
            if self.settings.class_coverage and classes_per_call and available_classes else None,
            "coverage_window_assumption": "fixed candidate class set and positive constant retained budget",
            "selected_indices": indices.tolist(), "scoring": dict(scored.get("diagnostics", {})),
        }
        # The optional study policy adds full reproducibility metadata only when enabled.
        if quotas is not None:
            diagnostics.update(
                quota_policy=self.settings.quota_policy,
                target_quotas={str(class_id): quota for class_id, quota in quotas.items()},
                eligible_class_counts={str(class_id): count for class_id, count in eligible_class_counts.items()},
                actual_selected_counts=counts,
                selector_seed=self.seed, quota_seed=quota_seed, ranking_seed=call_seed,
                quota_remainder_rule="seeded permutation of sorted old class IDs; first budget % K receive one extra",
                quota_seed_scope="task-stable and independent of ranking strategy, scores and selection call",
                coverage_window_calls=1 if budget else None,
                coverage_window_assumption="fixed exact quotas for every old class in this retained pool",
            )
            diagnostics.update(_candidate_identities(x, y, indices))
        for name in ("drift", "new_class_invasion", "confidence", "label_surprisal"):
            values = np.asarray(scored[name])
            diagnostics[f"candidate_mean_{name}"] = float(values.mean()) if len(values) else None
            diagnostics[f"selected_mean_{name}"] = float(values[indices].mean()) if len(indices) else None
        # Keep prospective loss change separate from accumulated drift summaries.
        if strategy == "mir":
            diagnostics["selected_mean_interference"] = float(np.mean(np.asarray(interference)[indices])) if len(indices) else None
        return x[indices], y[indices], diagnostics


def prepare_virtual_current_batch(wrapper: object, batch: tuple) -> tuple:
    """Apply the actual fit-time training bounds and teacher-target preparation.

    ``DiffusionModel.fit`` restores its entry bounds after every block, and
    sets ``_preprocess_training`` only while tracing its mapped training data.
    Calling ``prep_inputs_map`` directly between fits would therefore sample
    the wrong timestep range under restricted training bounds, and could use
    test CFG scale for noise-only KD. This adapter mirrors that existing fit
    context, then restores it. Random preparation occurs once before MIR's
    reversible numerical update and is additional baseline work.

    Args:
        wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
            schedules and existing training or inference APIs.
        batch (tuple): Raw current image/label pair or triple with replay provenance, using
            wrapper-facing original labels.

    Returns:
        prepared (tuple): Mapped tensor tuple from the existing fit-time preprocessing API,
            including teacher targets when configured.

    Raises:
        ValueError: If batch is not a raw pair or provenance triple, or inherited
            timestep/preprocessing controls are invalid.
    """

    # A prospective update must prepare a raw current batch exactly once.
    if len(batch) not in (2, 3):
        raise ValueError("MIR preparation requires a raw current image/label pair or provenance triple.")
    bounds = wrapper._active_min_timestep, wrapper._active_max_timestep
    previous_mode = wrapper._preprocess_training
    try:
        wrapper.set_timestep_bounds(wrapper.train_noisified_min_timesteps, wrapper.train_noisified_max_timesteps)
        wrapper._preprocess_training = True
        return wrapper.prep_inputs_map(*batch)
    finally:
        wrapper._preprocess_training = previous_mode
        wrapper.set_timestep_bounds(*bounds)


def virtual_update_interference(
    wrapper: object, prepared_current_batch: tuple, selector: DriftReplaySelector,
    images: np.ndarray, labels: np.ndarray, teacher: object = None,
    before_scores: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Compute MIR loss increase using the live optimizer, then restore state.

    The candidate objective is mean matched-view conditioning-label NLL. The
    virtual update is the platform's actual joint ``train_step`` with its
    current optimizer slots, schedule, clipping, and enabled losses. This is
    MIR's anticipated loss-change criterion adapted to the joint platform,
    not a reproduction of the paper's entire experimental setup.

    Prepare the current batch once using ``prepare_virtual_current_batch``
    BEFORE calling this function. Preparation owns stochastic noising/CFG draws;
    this function accepts mapped tensors only and rejects stochastic training
    layers, EMA, and uninitialized optimizers. It therefore consumes no hidden
    legacy TensorFlow RNG stream during the reversible update. Existing model,
    optimizer and metric variables are restored even if scoring raises.

    Args:
        wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
            schedules and existing training or inference APIs.
        prepared_current_batch (tuple): Mapped tensor tuple produced once by
            prepare_virtual_current_batch; raw image/label tuples are rejected.
        selector (DriftReplaySelector): DriftReplaySelector defining the matched-view
            candidate objective and its fixed scoring settings.
        images (np.ndarray): Numeric sample-major images in the configured model-input
            scale, normally float32 NHWC values in [-1, 1].
        labels (np.ndarray): Sparse integer label vector aligned with the image rows; the
            label convention for this operation is described above.
        teacher (object): Independent frozen prior-task raw network; None uses
            wrapper.teacher_network when this operation accepts a wrapper.
        before_scores (dict | None): Optional cached score() result from the current
            unchanged checkpoint and exact candidate rows.

    Returns:
        interference (tuple[np.ndarray, dict]): (increase, audit): float64 candidate NLL
            increase [N] and numerical-restoration/work metadata; no optimizer update
            remains committed.

    Raises:
        ValueError: If eager execution, float32, deterministic layers, prepared input or
            initialized optimizer requirements fail.
        RuntimeError: If one virtual update is not observed, new mutable state appears or
            saved numerical state is not restored.
    """

    import tensorflow as tf

    from common.keras_compat import optimizer_iterations

    # Reversible live assignments require eager execution and no secondary EMA update.
    if not tf.executing_eagerly() or getattr(wrapper, "use_ema", False):
        raise ValueError("Exact MIR currently requires eager execution and EMA disabled.")
    # Raw batches would advance legacy stochastic preprocessing inside the virtual step.
    if not getattr(wrapper, "map_preprocess", False) or len(prepared_current_batch) <= 3:
        raise ValueError("MIR requires an already mapped current batch; call prepare_virtual_current_batch first.")
    # Dynamic loss scaling introduces optimizer skip behavior outside this contract.
    if getattr(wrapper, "dtype_policy", None).name != "float32":
        raise ValueError("Exact MIR currently requires float32 precision.")
    _check_fixed_inference(wrapper.network)
    layers = getattr(wrapper.network, "submodules", None)
    # Keras 3 provides recursive flattening in place of the older submodules view.
    if layers is None:
        layers = wrapper.network._flatten_layers(include_self=False, recursive=True)
    for layer in layers:
        for name in (
            "rate", "drop_prob", "droppath_rate", "clf_droppath_rate",
            "dropout_rate", "classifier_dropout_rate", "dropout", "recurrent_dropout"
        ):
            value = getattr(layer, name, 0.)
            # Active dropout/stochastic depth would advance unrestorable legacy RNG.
            if isinstance(value, (int, float)) and value > 0:
                raise ValueError("MIR requires deterministic training layers; disable dropout/stochastic depth.")
        # Latent sampling and random perturbation layers require a separate RNG protocol.
        if getattr(layer, "add_kl", False) or isinstance(layer, (tf.keras.layers.GaussianNoise, tf.keras.layers.GaussianDropout)):
            raise ValueError("MIR does not support stochastic variational/noise layers.")
    optimizer = wrapper.optimizer
    # A real wake step must initialize the exact optimizer slots before they are copied.
    if int(optimizer_iterations(optimizer).numpy()) < 1:
        raise ValueError("MIR requires a real wake update first so optimizer slots already exist.")

    def state_variables() -> list:
        """Deduplicate all existing numerical model, optimizer, and metric state.

        Returns:
            variables (list): Identity-deduplicated ordered list of existing model, optimizer
                and metric numerical variables.

        Raises:
            AttributeError: If an optimizer or metric does not expose its variable collection.
        """

        optimizer_values = optimizer.variables() if callable(optimizer.variables) else optimizer.variables
        all_variables = list(wrapper.weights) + list(optimizer_values)
        for metric in wrapper.metrics:
            all_variables.extend(metric.variables)
        unique = {}
        for variable in all_variables:
            unique[id(variable)] = variable
        return list(unique.values())

    variables = state_variables()
    state = [variable.numpy().copy() for variable in variables]
    started = time.perf_counter()
    before = selector.score(wrapper, images, labels, teacher) if before_scores is None else before_scores
    iteration = int(optimizer_iterations(optimizer).numpy())
    try:
        wrapper.train_step(prepared_current_batch)
        # Skipped or repeated optimizer steps would change the MIR criterion.
        if int(optimizer_iterations(optimizer).numpy()) != iteration + 1:
            raise RuntimeError("MIR expected exactly one virtual optimizer update.")
        # Lazily created variables were not part of the reversible pre-update snapshot.
        if [id(variable) for variable in state_variables()] != [id(variable) for variable in variables]:
            raise RuntimeError("MIR created new mutable state; build all trainable paths during wake first.")
        after = selector.score(wrapper, images, labels, teacher)
        interference = np.asarray(after["student_label_loss"]) - np.asarray(before["student_label_loss"])
    finally:
        for variable, value in zip(variables, state):
            variable.assign(value)
    # Require bitwise restoration, including Adam moments and optimizer iteration.
    if any(not np.array_equal(variable.numpy(), value) for variable, value in zip(variables, state)):
        raise RuntimeError("MIR failed to restore numerical model/optimizer/metric state.")
    return interference, {
        "criterion": "candidate_mean_view_NLL_after_minus_before_actual_virtual_joint_update",
        "virtual_optimizer_updates": 1, "committed_optimizer_updates": 0,
        "optimizer": type(optimizer).__name__, "restored_state_bytes": sum(value.nbytes for value in state),
        "virtual_current_examples": int(len(prepared_current_batch[0])),
        "state_restored": True, "stochastic_preprocessing": "prepared_once_before_virtual_update",
        "after_scoring": after["diagnostics"],
        "before_scoring_reused": before_scores is not None,
        "seconds": time.perf_counter() - started,
    }
