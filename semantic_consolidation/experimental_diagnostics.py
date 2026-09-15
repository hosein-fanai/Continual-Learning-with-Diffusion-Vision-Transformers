"""Section-11 held-out hidden probes and class-conditioned generated-memory audit.

Prediction reliability and replay summaries reuse ``common.mechanistic``;
hidden projections reuse ``semantic_consolidation.phases.semantic_features``.
The cubic-kernel unbiased MMD estimator follows the KID construction in
https://arxiv.org/abs/1801.01401. The default *pixel* feature extractor makes
this a KID-style pixel-distribution diagnostic, not standard Inception KID.
Linear CKA uses the existing implementation of the statistic discussed in
https://proceedings.mlr.press/v97/kornblith19a.html.

These are descriptive measurements. Classifier agreement does not certify a
generated label, and no distribution score establishes replay usefulness.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
import hashlib
from pathlib import Path

import numpy as np

from common.mechanistic import (
    _probability_matrix, class_centroid_drift, linear_cka, replay_quality_metrics,
)
from common.runtime import derive_seed


def _positive(value: int, name: str) -> int:
    """Reject fractional and boolean sample budgets before allocating arrays.

    Args:
        value (int): Python or NumPy integer required to be positive; booleans and
            fractional values are rejected.
        name (str): Human-readable field name included in validation error messages.

    Returns:
        count (int): Validated positive Python int.

    Raises:
        ValueError: If value is boolean, fractional, noninteger or less than one.
    """

    # Diagnostic sampling requires a positive integer row budget.
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _seed(value: int) -> int:
    """Require reproducible diagnostic sampling without entropy-seeded fallbacks.

    Args:
        value (int): Python or NumPy integer in [0, 2**32); booleans are rejected.

    Returns:
        seed (int): Validated Python int in [0, 2**32).

    Raises:
        ValueError: If value is boolean, noninteger or outside the seed domain.
    """

    # seed must be an explicit integer in [0, 2**32).
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < 2 ** 32:
        raise ValueError("seed must be an explicit integer in [0, 2**32).")
    return int(value)


def _labels(values: object, count: int) -> np.ndarray:
    """Require aligned sparse nonnegative integer class IDs, preserving originals.

    Args:
        values (object): Aligned numeric sparse IDs [N] or [N, 1]; finite nonnegative
            integer-valued inputs are accepted.
        count (int): Exact integer number of expected or selected rows.

    Returns:
        labels (np.ndarray): One-dimensional int64 class IDs preserving row order.

    Raises:
        ValueError: If labels are misaligned, nonnumeric, nonfinite, negative or fractional.
    """

    labels = np.asarray(values)
    # A sparse column vector has the same label meaning as a sparse flat vector.
    if labels.shape == (count, 1):
        labels = labels[:, 0]
    # Labels must be aligned sparse numeric class IDs.
    if labels.shape != (count,) or not np.issubdtype(labels.dtype, np.number):
        raise ValueError("Labels must be aligned sparse numeric class IDs.")
    # Labels must be finite nonnegative integers.
    if not np.isfinite(labels).all() or np.any(labels < 0) or np.any(labels != np.floor(labels)):
        raise ValueError("Labels must be finite nonnegative integers.")
    return labels.astype("int64")


def _images(values: object) -> np.ndarray:
    """Require the project's finite NHWC input arrays without changing their scale.

    Args:
        values (object): Numeric image array [N, H, W, C], cast to float32 without
            rescaling; N may be zero.

    Returns:
        images (np.ndarray): Float32 NHWC ndarray at the supplied scale; empty sample axes
            are allowed.

    Raises:
        ValueError: If rank, spatial/channel dimensions or finiteness are invalid.
    """

    images = np.asarray(values, dtype="float32")
    # Images must be finite NHWC arrays with nonempty spatial/channel dimensions.
    if images.ndim != 4 or any(size < 1 for size in images.shape[1:]) or not np.isfinite(images).all():
        raise ValueError("Images must be finite NHWC arrays with nonempty spatial/channel dimensions.")
    return images


def _features(values: object) -> np.ndarray:
    """Flatten finite nonempty feature axes, retaining the row correspondence.

    Args:
        values (object): Finite numeric array [N, ...] with N and the flattened feature
            width both positive.

    Returns:
        features (np.ndarray): Finite nonempty float64 matrix [N, D] with flattened feature
            axes.

    Raises:
        ValueError: If rows, feature dimensions or numerical values are invalid.
    """

    features = np.asarray(values, dtype="float64")
    # Features require nonempty finite rows and a feature axis.
    if features.ndim < 2 or not len(features) or not np.isfinite(features).all():
        raise ValueError("Features require nonempty finite rows and a feature axis.")
    features = features.reshape((len(features), -1))
    # Features require at least one coordinate.
    if not features.shape[1]:
        raise ValueError("Features require at least one coordinate.")
    return features


def _json_finite(value: object) -> object:
    """Keep undefined measurements as JSON null, never nonstandard NaN tokens.

    Args:
        value (object): Nested dict/list/tuple with ordinary values and NumPy scalar
            numbers; nonfinite floats become None.

    Returns:
        converted (object): Nested Python scalar/sequence/mapping values, with nonfinite
            measurements represented by None.

    Raises:
        RecursionError: If a cyclic or excessively deep structure cannot be traversed.
    """

    # Recursively normalize mapping keys and unavailable nested measurements for JSON.
    if isinstance(value, dict):
        return {str(key): _json_finite(item) for key, item in value.items()}
    # Preserve sequence order while converting nested NumPy scalars.
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    # Unavailable floating measurements are represented as JSON null.
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    # Use built-in integers so diagnostic records serialize without a custom encoder.
    if isinstance(value, np.integer):
        return int(value)
    return value


def _image_hash(image: np.ndarray) -> str:
    """Identify fixed float32 observations including their spatial/channel shape.

    Args:
        image (np.ndarray): One finite numeric image whose shape and float32 pixel bytes
            identify the observation.

    Returns:
        digest (str): SHA-256 hexadecimal string of image shape and canonical float32 pixel
            bytes.

    Raises:
        ValueError: If image values cannot be converted to float32.
    """

    digest = hashlib.sha256(str(image.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(image, dtype="float32").tobytes())
    return digest.hexdigest()


def _sample_indices(images: np.ndarray, labels: np.ndarray, class_id: int,
                    count: int, seed: int) -> np.ndarray:
    """Select uniform per-class rows reproducibly after canonical content ordering.

    Args:
        images (np.ndarray): Numeric sample-major images in the configured model-input
            scale, normally float32 NHWC values in [-1, 1].
        labels (np.ndarray): Sparse integer label vector aligned with the image rows; the
            label convention for this operation is described above.
        class_id (int): Nonnegative integer class ID selecting one gate or one class cohort.
        count (int): Exact integer number of expected or selected rows.
        seed (int): Explicit integer random seed; local or derived streams preserve
            reproducibility without reseeding caller-owned generators.

    Returns:
        indices (np.ndarray): Int64 vector of at most count distinct class-row indices,
            reproducible after content ordering.

    Raises:
        ValueError: If seed is invalid or images/labels cannot be aligned.
    """

    available = np.flatnonzero(labels == class_id)
    ordered = np.asarray(sorted(available, key=lambda index: _image_hash(images[index])), dtype="int64")
    return np.random.default_rng(seed).permutation(ordered)[:count]


def extract_hidden(network: object, images: object, batch_size: int = 32) -> np.ndarray:
    """Read the actual clean hidden projection with no oracle condition or updates.

    The shared semantic-feature API returns the projection immediately before
    the primary classification output. It performs inference-mode calls with
    unconditional class input zero and exactly clean images at time zero.

    Args:
        network (object): Raw classifier network exposing predict_class and the existing
            primary classifier layers.
        images (object): Numeric sample-major images in the configured model-input scale,
            normally float32 NHWC values in [-1, 1].
        batch_size (int): Positive integer maximum number of rows per batch; any additional
            phase-specific minimum is described above.

    Returns:
        features (np.ndarray): Float32 hidden matrix [N, D] immediately before the primary
            output layer.

    Raises:
        ValueError: If batch size or input images are invalid/empty or the network
            projection is incompatible.
    """

    import tensorflow as tf
    from semantic_consolidation.phases import semantic_features

    batch_size = _positive(batch_size, "batch_size")
    values = _images(images)
    # Hidden extraction requires at least one held-out image.
    if not len(values):
        raise ValueError("Hidden extraction requires at least one held-out image.")
    batches = []
    for start in range(0, len(values), batch_size):
        batch = tf.convert_to_tensor(values[start:start + batch_size], dtype=network.compute_dtype)
        hidden, _ = semantic_features(network, batch, tf.zeros((len(batch),), dtype=tf.int32))
        batches.append(hidden.numpy().reshape((len(batch), -1)))
    return _features(np.concatenate(batches)).astype("float32")


def descriptive_linear_cka(previous: object, current: object) -> dict:
    """Measure aligned temporal CKA only when centering has more than one degree of freedom.

    With two nonconstant observations, centered linear Gram matrices are
    proportional and CKA is one regardless of feature changes. Three rows is
    only an availability threshold; even an eight-row cohort is descriptive,
    not strong evidence of representation preservation. Invalid/misaligned
    feature arrays are rejected without dropping or replacing observations.

    Args:
        previous (object): Finite sample-aligned earlier feature array [N, ...], converted
            to a float64 matrix.
        current (object): Finite current feature array with the same rows/order as previous;
            flattened widths may differ.

    Returns:
        alignment (dict): Dict with sample_count, optional Python float linear_cka and an
            explicit unavailable reason.

    Raises:
        ValueError: If feature arrays are invalid or rows are not aligned.
    """

    before, after = _features(previous), _features(current)
    # Row identity is a prerequisite for temporal feature alignment.
    if len(before) != len(after):
        raise ValueError("Hidden CKA requires the same fixed rows in the same order.")
    value, reason = None, None
    # Two-row centered Gram matrices make otherwise unrelated features align perfectly.
    if len(before) <= 2:
        reason = "fewer_than_three_aligned_observations"
    # A constant centered representation has zero HSIC normalization.
    elif np.all(before == before[:1]) or np.all(after == after[:1]):
        reason = "constant_centered_representation"
    # Evaluate only nondegenerate aligned cohorts.
    else:
        measured = linear_cka(before, after)
        # Preserve valid measured alignment without fabricating unavailable values.
        if np.isfinite(measured):
            value = float(measured)
        # Numerical degeneracy remains an explicit missing measurement.
        else:
            reason = "undefined_centered_alignment"
    return {"sample_count": len(before), "linear_cka": value,
            "linear_cka_unavailable_reason": reason}


def hidden_feature_change(previous: object, current: object, labels: object) -> dict:
    """Describe aligned-row drift and reuse common CKA and centroid calculations.

    CKA allows distinct feature widths. Coordinate-dependent drift requires
    the same width, and relative Frobenius drift additionally requires a
    nonzero reference norm. CKA is unavailable for fewer than three rows or a
    constant centered representation. Negative numerical roundoff is handled
    by the common CKA implementation, not by changing the drift observations.

    Args:
        previous (object): Finite sample-aligned earlier feature array [N, ...], converted
            to a float64 matrix.
        current (object): Finite current feature array with the same rows/order as previous;
            coordinate drift requires equal widths.
        labels (object): Aligned sparse nonnegative original class IDs used to compare class
            centroids.

    Returns:
        change (dict): JSON-safe dict of CKA, optional float coordinate drift and class-
            centroid changes; undefined metrics are None.

    Raises:
        ValueError: If features or labels are invalid or rows differ between checkpoints.
    """

    before, after = _features(previous), _features(current)
    targets = _labels(labels, len(before))
    # Hidden drift requires the same fixed rows in the same order.
    if len(before) != len(after):
        raise ValueError("Hidden drift requires the same fixed rows in the same order.")
    result = {**descriptive_linear_cka(before, after),
              "before_feature_width": before.shape[1], "after_feature_width": after.shape[1],
              "mean_sample_l2_drift": None, "relative_frobenius_drift": None,
              "centroid_drift": None}
    # Coordinate-dependent drift requires equal feature widths; CKA does not.
    if before.shape[1] == after.shape[1]:
        difference = after - before
        denominator = float(np.linalg.norm(before))
        result.update(
            mean_sample_l2_drift=float(np.linalg.norm(difference, axis=1).mean()),
            relative_frobenius_drift=float(np.linalg.norm(difference) / denominator) if denominator else None,
            centroid_drift=class_centroid_drift(before, targets, after, targets),
        )
    return _json_finite(result)


class FixedHiddenProbe:
    """Track bounded validation cohorts fixed at each class's first observation.

    By default the observer retains its selected historical validation pixels
    exclusively for diagnostics; callers must disclose this information and
    byte count. With ``retain_images=False`` it stores only fingerprints and
    numeric feature snapshots, and reports missing historical rows instead of
    substituting different samples. Neither mode exposes cached data to fits.
    Acquisition comparisons are per class because their reference checkpoints
    differ; a pooled CKA across acquisition checkpoints would be misleading.
    """

    def __init__(self, per_class: int = 8, batch_size: int = 32, seed: int = 0,
                 network_name: str = "raw", retain_images: bool = True) -> None:
        """Validate cohort budgets and initialize optional retained validation information.

        Args:
            per_class (int): Positive integer maximum number of fixed validation rows retained
                for each represented class.
            batch_size (int): Positive integer maximum number of rows per batch; any additional
                phase-specific minimum is described above.
            seed (int): Explicit integer random seed; local or derived streams preserve
                reproducibility without reseeding caller-owned generators.
            network_name (str): Existing raw or ema branch name; requesting EMA requires actual
                EMA weights where validated.
            retain_images (bool): Whether to retain the selected held-out images; False retains
                fingerprints/features and reports missing later rows.

        Returns:
            initialized (None): None; starts empty fixed class cohorts and no previous
                observation.

        Raises:
            ValueError: If sample caps, network branch, seed or retain_images flag are invalid.
        """
        self.per_class = _positive(per_class, "per_class")
        self.batch_size = _positive(batch_size, "batch_size")
        # network_name must be raw or ema.
        if network_name not in ("raw", "ema"):
            raise ValueError("network_name must be raw or ema.")
        # retain_images must be boolean.
        if not isinstance(retain_images, bool):
            raise ValueError("retain_images must be boolean.")
        self.seed, self.network_name, self.retain_images = _seed(seed), network_name, retain_images
        self.cohorts: dict[int, dict] = {}
        self.last_task: int | None = None

    @property
    def retained_bytes(self) -> dict[str, int]:
        """Count persistent numeric probe arrays and serialized fingerprint bytes.

        Returns:
            bytes_by_kind (dict[str, int]): Dict of integer retained image, feature and binary-
                fingerprint payload bytes, excluding Python overhead.

        Raises:
            KeyError: If internal cohort records are incomplete.
        """

        images = sum(group["images"].nbytes for group in self.cohorts.values() if group["images"] is not None)
        features = sum(group[name].nbytes for group in self.cohorts.values()
                       for name in ("acquisition_features", "previous_features"))
        hashes = sum(32 * len(group["hashes"]) for group in self.cohorts.values())
        return {"image_bytes": images, "feature_bytes": features, "fingerprint_payload_bytes": hashes,
                "numeric_and_fingerprint_payload_bytes": images + features + hashes}

    def observe(self, wrapper: object, images: object, original_labels: object,
                task_index: int, *, split: str = "validation") -> dict:
        """Observe one completed checkpoint on supplied permitted held-out rows.

        Args:
            wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
                schedules and existing training or inference APIs.
            images (object): Numeric sample-major images in the configured model-input scale,
                normally float32 NHWC values in [-1, 1].
            original_labels (object): Sparse nonnegative integer original class IDs [N], aligned
                with supplied validation images.
            task_index (int): Nonnegative integer checkpoint index, strictly increasing for one
                probe instance.
            split (str): Declared data split; supported training, validation or test access is
                constrained by this operation.

        Returns:
            observation (dict): JSON-compatible task dict with per-class fixed-cohort changes,
                explicit availability and retained bytes.

        Raises:
            ValueError: If split, task order, label support, images or requested EMA branch is
                invalid.
        """

        # Fixed probes accept held-out validation rows only.
        if split != "validation":
            raise ValueError("Fixed probes accept held-out validation rows only.")
        # task_index must be a nonnegative integer.
        if isinstance(task_index, bool) or not isinstance(task_index, (int, np.integer)) or task_index < 0:
            raise ValueError("task_index must be a nonnegative integer.")
        # Probe task indices must be strictly increasing.
        if self.last_task is not None and task_index <= self.last_task:
            raise ValueError("Probe task indices must be strictly increasing.")
        values = _images(images)
        labels = _labels(original_labels, len(values))
        # Probe labels contain future or unmapped classes.
        if not set(labels).issubset(set(wrapper.seen_classes)):
            raise ValueError("Probe labels contain future or unmapped classes.")
        # EMA hidden probes require an actual EMA network; raw fallback changes the treatment.
        if self.network_name == "ema" and not getattr(wrapper, "use_ema", False):
            raise ValueError("EMA hidden probes require an actual EMA network; raw fallback changes the treatment.")
        network = wrapper.get_network(self.network_name)
        per_class = {}
        for class_id in sorted(set(self.cohorts) | set(labels.tolist())):
            group = self.cohorts.get(class_id)
            # Choose a class cohort only when that class is first observed.
            if group is None:
                selected = _sample_indices(values, labels, class_id, self.per_class,
                                           derive_seed(self.seed, "hidden_cohort", class_id))
                fixed = values[selected]
                hashes = [_image_hash(image) for image in fixed]
            # Retained validation images give the exact original cohort at later checkpoints.
            elif self.retain_images:
                fixed, hashes = group["images"], group["hashes"]
            else:
                # Match multiplicities as well as hashes when identical pixels recur.
                available = defaultdict(list)
                for index in np.flatnonzero(labels == class_id):
                    available[_image_hash(values[index])].append(index)
                selected = []
                for fingerprint in group["hashes"]:
                    # Consume one matching occurrence to preserve duplicate-image multiplicities.
                    if available[fingerprint]:
                        selected.append(available[fingerprint].pop())
                # Missing historical rows cannot be replaced by a different available cohort.
                if len(selected) != len(group["hashes"]):
                    per_class[str(class_id)] = {
                        "availability": "fixed_validation_rows_unavailable",
                        "required_rows": len(group["hashes"]), "available_rows": len(selected),
                        "acquisition_task_index": group["acquisition_task"],
                    }
                    continue
                fixed, hashes = values[selected], group["hashes"]
            features = extract_hidden(network, fixed, self.batch_size)
            fixed_labels = np.full(len(fixed), class_id, dtype="int64")
            record = {"availability": "measured", "sample_count": len(fixed), "sample_sha256": hashes,
                      "since_acquisition": None, "since_previous_observation": None}
            # Record this class acquisition checkpoint as the immutable feature reference.
            if group is None:
                group = {"images": fixed.copy() if self.retain_images else None, "hashes": hashes,
                         "acquisition_features": features.copy(), "acquisition_task": int(task_index)}
                self.cohorts[class_id] = group
                record["availability"] = "acquisition_reference_established"
            # Later observations compare with both acquisition and the previous measured checkpoint.
            else:
                record["since_acquisition"] = hidden_feature_change(group["acquisition_features"], features, fixed_labels)
                record["since_previous_observation"] = hidden_feature_change(group["previous_features"], features, fixed_labels)
                record["previous_observation_task_index"] = group["previous_task"]
            record["acquisition_task_index"] = group["acquisition_task"]
            group["previous_features"], group["previous_task"] = features.copy(), int(task_index)
            per_class[str(class_id)] = record
        self.last_task = int(task_index)
        effective_network = "raw" if network is getattr(wrapper, "network", None) else self.network_name
        return {"task_index": int(task_index), "split": split, "network_name": self.network_name,
                "effective_network_name": effective_network,
                "feature_extractor": "phases.semantic_features: clean hidden projection before primary output",
                "cka_estimator": "centered linear Gram alignment; ordinary biased-HSIC normalization, not debiased CKA",
                "cka_minimum_aligned_observations": 3,
                "cka_interpretation": "small fixed cohorts are descriptive; high CKA alone is not strong representation-preservation evidence",
                "selection": "class-first-observation uniform content-canonical sample, fixed thereafter",
                "per_class_limit": self.per_class, "seed": self.seed, "per_class": per_class,
                "retains_historical_validation_images": self.retain_images,
                "retained_bytes": self.retained_bytes,
                "memory_scope": "array and binary fingerprint payload; Python/container overhead excluded",
                "use": "diagnostics only; never gradient fitting or hyperparameter selection"}


def polynomial_kid(real_features: object, generated_features: object) -> dict:
    """Compute one all-pairs unbiased cubic-kernel MMD-squared estimate.

    For k(x,y)=(x.T@y/d+1)^3 this is
    sum(i!=j)k(x_i,x_j)/(m(m-1)) + sum(i!=j)k(y_i,y_j)/(n(n-1))
    - 2*sum(i,j)k(x_i,y_j)/(mn). The estimate may legitimately be negative.
    Population unbiasedness assumes independent IID real/generated samples;
    selected replay pools instead have a descriptive finite-pool meaning.
    No repeated-subset standard deviation is presented as training uncertainty.

    Args:
        real_features (object): Finite real-data feature array [M, ...], flattened to
            float64 with M at least two.
        generated_features (object): Finite generated-data feature array [N, ...] with N at
            least two and the same flattened width as real_features.

    Returns:
        estimate (dict): Dict containing Python float unbiased MMD-squared, sample counts
            and kernel metadata; negative estimates are retained.

    Raises:
        ValueError: If features are invalid, widths differ, either population has fewer than
            two rows or the polynomial kernel overflows.
    """

    real, generated = _features(real_features), _features(generated_features)
    # Real and generated features must have the same width.
    if real.shape[1] != generated.shape[1]:
        raise ValueError("Real and generated features must have the same width.")
    # Unbiased polynomial KID requires at least two rows in each population.
    if len(real) < 2 or len(generated) < 2:
        raise ValueError("Unbiased polynomial KID requires at least two rows in each population.")
    width = real.shape[1]
    with np.errstate(over="raise", invalid="raise"):
        try:
            xx = (real @ real.T / width + 1.) ** 3
            yy = (generated @ generated.T / width + 1.) ** 3
            xy = (real @ generated.T / width + 1.) ** 3
            value = ((xx.sum() - np.trace(xx)) / (len(real) * (len(real) - 1))
                     + (yy.sum() - np.trace(yy)) / (len(generated) * (len(generated) - 1))
                     - 2. * xy.mean())
        except FloatingPointError as error:
            raise ValueError("Polynomial kernel overflow; use a fixed, disclosed feature scale.") from error
    return {"value": float(value), "real_sample_count": len(real), "generated_sample_count": len(generated),
            "feature_width": width, "kernel": "(dot(x,y)/feature_width + 1)^3",
            "estimator": "unbiased_two_sample_U_statistic_off_diagonal_within_all_cross_pairs",
            "negative_estimates_clipped": False}


def generated_memory_diagnostics(
    images: object, labels: object, expected_classes: Sequence[int], *,
    real_images: object | None = None, real_labels: object | None = None,
    probabilities: object | None = None, seed: int = 0, max_per_class: int = 128,
    representatives_per_class: int = 4, feature_extractor: Callable | None = None,
    feature_metadata: dict | None = None, artifact_path: str | Path | None = None,
) -> dict:
    """Audit the supplied actual replay pool with bounded per-class distribution work.

    Images use the same fixed preprocessing on both populations, normally the
    project's [-1,1] input scale. Sparse labels and probability columns use
    dense *seen-head* IDs. ``real_images`` must be held-out permitted validation
    data supplied by the caller, never future-class training data. The default
    extractor is identity flattening with no pretrained weights. A custom
    callable must be frozen and identical across methods and supply ``name``,
    ``pretraining``, ``preprocessing`` and ``identity_sha256`` metadata. This
    function does not download, fit, select, or calibrate any evaluator.

    Representatives are seeded uniform rows, not favorable score-ranked rows.
    If requested, the compressed NPZ stores their pixels, conditioning labels,
    and source-pool indices. Full-pool coverage/consistency remain distinct
    from bounded diversity/KID feature comparisons; all sample counts are saved.

    Args:
        images (object): Numeric sample-major images in the configured model-input scale,
            normally float32 NHWC values in [-1, 1].
        labels (object): Sparse integer label vector aligned with the image rows; the label
            convention for this operation is described above.
        expected_classes (Sequence[int]): Distinct nonnegative dense class IDs expected in
            the replay population.
        real_images (object | None): Optional caller-supplied held-out validation images in
            the same geometry and fixed scale as generated images.
        real_labels (object | None): Optional aligned sparse integer validation labels,
            required together with real_images.
        probabilities (object | None): Finite nonnegative prediction matrix [N, C] with unit
            row mass, aligned with sparse labels when supplied.
        seed (int): Explicit integer random seed; local or derived streams preserve
            reproducibility without reseeding caller-owned generators.
        max_per_class (int): Integer per-class cap of at least two for bounded diversity and
            unbiased two-sample distribution calculations.
        representatives_per_class (int): Positive integer cap on uniformly sampled
            representative images exported for each class.
        feature_extractor (Callable | None): Optional frozen callable preserving row order
            and count; None uses identity flattened pixels without learned weights.
        feature_metadata (dict | None): Required identity/pretraining/preprocessing metadata
            for a custom frozen extractor; None with the pixel default.
        artifact_path (str | Path | None): Optional .npz destination for representative
            pixels, labels and source-row indices.

    Returns:
        diagnostics (dict): JSON-safe replay summary, per-class distribution/consistency
            statistics and optional representative-artifact metadata.

    Raises:
        ValueError: If sample budgets, labels, shapes, custom extractor metadata or output
            suffix are invalid.
        OSError: If the requested representative artifact cannot be written.
    """

    max_per_class = _positive(max_per_class, "max_per_class")
    representatives_per_class = _positive(representatives_per_class, "representatives_per_class")
    seed = _seed(seed)
    # max_per_class must be at least two for unbiased KID.
    if max_per_class < 2:
        raise ValueError("max_per_class must be at least two for unbiased KID.")
    values = _images(images)
    targets = _labels(labels, len(values))
    expected_values = np.asarray(list(expected_classes))
    expected = _labels(expected_values, len(expected_values)).tolist()
    # expected_classes must not contain duplicates.
    if len(set(expected)) != len(expected):
        raise ValueError("expected_classes must not contain duplicates.")
    # Generated labels must belong to expected_classes.
    if not set(targets).issubset(set(expected)):
        raise ValueError("Generated labels must belong to expected_classes.")
    probs = None if probabilities is None else _probability_matrix(probabilities)
    # Probabilities must align with samples and contain their dense class IDs.
    if probs is not None and (len(probs) != len(values) or (len(targets) and targets.max() >= probs.shape[1])):
        raise ValueError("Probabilities must align with samples and contain their dense class IDs.")
    # real_images and real_labels must be supplied together.
    if (real_images is None) != (real_labels is None):
        raise ValueError("real_images and real_labels must be supplied together.")
    real = _images(real_images) if real_images is not None else np.empty((0, *values.shape[1:]), dtype="float32")
    reference_labels = _labels(real_labels, len(real)) if real_labels is not None else np.empty(0, dtype="int64")
    # Real and generated image shapes and preprocessing must match.
    if real.shape[1:] != values.shape[1:]:
        raise ValueError("Real and generated image shapes and preprocessing must match.")
    # The default evaluator uses disclosed pixel features without an external image prior.
    if feature_extractor is None:
        # Custom feature metadata requires a custom feature_extractor.
        if feature_metadata is not None:
            raise ValueError("Custom feature metadata requires a custom feature_extractor.")
        metadata = {"name": "identity_flattened_pixels", "pretraining": "none; no learned parameters",
                    "preprocessing": "project model-input values; no rescaling or normalization inside evaluator",
                    "identity_sha256": hashlib.sha256(b"identity_flattened_pixels:v1").hexdigest(),
                    "interpretation": "pixel-space polynomial MMD; KID-style diagnostic, not Inception KID"}

        def feature_extractor(batch: np.ndarray) -> np.ndarray:
            """Keep the input-space metric constant across checkpoints and methods.

            Args:
                batch (np.ndarray): Finite float32 image batch [N, H, W, C] already at the disclosed
                    input scale.

            Returns:
                features (np.ndarray): Float32 matrix [N, H*W*C] retaining input values and row
                    order.

            Raises:
                ValueError: If an empty or incompatible batch cannot be reshaped as requested.
            """

            return batch.reshape((len(batch), -1))
    # Custom frozen feature extractors must supply reproducible identity and preprocessing metadata.
    else:
        required = ("name", "pretraining", "preprocessing", "identity_sha256")
        # Custom features require name, pretraining, preprocessing, and identity_sha256 metadata.
        if not isinstance(feature_metadata, dict) or any(
            not isinstance(feature_metadata.get(key), str) or not feature_metadata[key].strip() for key in required
        ):
            raise ValueError("Custom features require name, pretraining, preprocessing, and identity_sha256 metadata.")
        metadata = dict(feature_metadata)
        # Feature identity_sha256 must be a SHA-256 hex digest.
        if len(metadata["identity_sha256"]) != 64 or any(character not in "0123456789abcdef" for character in metadata["identity_sha256"].lower()):
            raise ValueError("Feature identity_sha256 must be a SHA-256 hex digest.")
    summary = replay_quality_metrics(values, targets, expected, probabilities=probs,
                                     max_diversity_samples=max_per_class, seed=seed)
    per_class, representatives = {}, []
    measured_scores = []
    for class_id in sorted(expected):
        class_rows = np.flatnonzero(targets == class_id)
        class_probs = probs[class_rows] if probs is not None else None
        report = replay_quality_metrics(values[class_rows], targets[class_rows], [class_id],
                                        probabilities=class_probs, max_diversity_samples=max_per_class,
                                        seed=derive_seed(seed, "generated_diversity", class_id))
        chosen = _sample_indices(values, targets, class_id, max_per_class,
                                 derive_seed(seed, "generated_kid", class_id))
        reference = _sample_indices(real, reference_labels, class_id, max_per_class,
                                    derive_seed(seed, "real_kid", class_id))
        sample_rows = _sample_indices(values, targets, class_id, representatives_per_class,
                                      derive_seed(seed, "representative_samples", class_id))
        representatives.extend(sample_rows.tolist())
        report["representative_source_indices"] = sample_rows.tolist()
        report["distribution_sample_counts"] = {"real_available": int(np.sum(reference_labels == class_id)),
                                                 "generated_available": len(class_rows),
                                                 "real_used": len(reference), "generated_used": len(chosen)}
        report["polynomial_kid"] = None
        report["distribution_availability"] = "fewer_than_two_real_or_generated_rows"
        # The unbiased two-sample estimator needs distinct within-population pairs.
        if len(reference) >= 2 and len(chosen) >= 2:
            reference_features = _features(feature_extractor(real[reference]))
            generated_features = _features(feature_extractor(values[chosen]))
            # Feature extractors must preserve sample count and ordering.
            if len(reference_features) != len(reference) or len(generated_features) != len(chosen):
                raise ValueError("Feature extractors must preserve sample count and ordering.")
            report["polynomial_kid"] = polynomial_kid(reference_features, generated_features)
            report["distribution_availability"] = "measured"
            measured_scores.append(report["polynomial_kid"]["value"])
        per_class[str(class_id)] = report
    representative_rows = np.asarray(representatives, dtype="int64")
    artifact = None
    # Persist representative pixels only when an explicit artifact path is supplied.
    if artifact_path is not None:
        destination = Path(artifact_path)
        # Representative artifact_path must end in .npz.
        if destination.suffix.lower() != ".npz":
            raise ValueError("Representative artifact_path must end in .npz.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, images=values[representative_rows], labels=targets[representative_rows],
                            source_indices=representative_rows)
        artifact = {"path": str(destination), "file_bytes": destination.stat().st_size,
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    return _json_finite({
        "summary": summary, "per_class": per_class, "feature_extractor": metadata,
        "distribution_reference": "caller-supplied held-out permitted validation images",
        "distribution_sampling": "independent uniform no-replacement bounded sample within each population and class",
        "max_per_class": max_per_class, "seed": seed,
        "macro_polynomial_kid": float(np.mean(measured_scores)) if measured_scores else None,
        "kid_classes_measured": len(measured_scores), "kid_classes_expected": len(expected),
        "kid_macro_scope": "equal weight over measured classes only; missing classes remain explicit",
        "uncertainty": "no image-resampling statistic is a training-replication confidence interval",
        "consistency_interpretation": "agreement with caller's classifier; not independent semantic ground truth",
        "reliability_protocol": {
            "ece_bins": 15, "binning": "equal-width confidence bins; left-closed, right-open except final bin includes one",
            "weighting": "sample fraction in each occupied bin", "nll_probability_floor": 1e-12,
            "calibration_fit": "none; supplied probabilities are evaluated unchanged against conditioning labels",
            "implementation": "common.mechanistic.replay_quality_metrics -> calibration_metrics defaults",
        },
        "population_interpretation": "unbiased for independent IID populations; selected replay is descriptive of the supplied pool",
        "representative_selection": "seeded uniform per-class rows; no quality ranking",
        "representative_count": len(representative_rows), "representative_artifact": artifact,
    })
