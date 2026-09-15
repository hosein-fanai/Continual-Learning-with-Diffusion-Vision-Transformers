"""Bounded held-out probes; these measurements never participate in training."""

from __future__ import annotations

import hashlib

import numpy as np
import tensorflow as tf

from common.runtime import derive_seed
from semantic_consolidation.objectives import normalized_features


def balanced_probe(images: np.ndarray, labels: np.ndarray, settings: object) -> tuple:
    """Canonicalize rows, sample per class, and interleave shuffled class rounds.

    Content ordering makes the selected observations and their order independent
    of the input dataset's iteration order. Labels balance the sample only;
    gate selection is separate and every selected gate sees every probe batch.

    Args:
        images (np.ndarray): Nonempty numeric image array [N, ...], indexed without changing
            its dtype.
        labels (np.ndarray): Aligned integer ndarray [N] containing at least one represented
            class.
        settings (object): Validated settings instance for this component; its fields select
            the behavior described above.

    Returns:
        probe (tuple): (images, labels): selected arrays preserving input dtypes and a
            deterministic mixed-class order.

    Raises:
        ValueError: If images and labels cannot be indexed consistently.
        ZeroDivisionError: If no class is represented; callers require a nonempty validation
            pool.
    """

    classes = np.unique(labels)
    per_class = max(2, settings.probe_batches * settings.batch_size // len(classes))
    selected = {}
    for class_id in classes:
        available = sorted(np.flatnonzero(labels == class_id), key=lambda i: hashlib.sha256(
            np.ascontiguousarray(images[i]).tobytes()
        ).digest())
        rng = np.random.default_rng(derive_seed(settings.seed, "validation_probe", int(class_id)))
        selected[int(class_id)] = rng.permutation(available)[:per_class].tolist()
    rng = np.random.default_rng(derive_seed(settings.seed, "probe_row_mixing"))
    indices = []
    for row in range(max(map(len, selected.values()))):
        for class_id in rng.permutation(classes):
            # Exhausted classes contribute no fabricated validation rows.
            if row < len(selected[int(class_id)]):
                indices.append(selected[int(class_id)][row])
    return images[indices], labels[indices]


def probe_batches(size: int, batch_size: int) -> list[np.ndarray]:
    """Partition all rows once, balancing batch sizes to avoid a singleton tail.

    Args:
        size (int): Integer number of available rows in the finite pool.
        batch_size (int): Positive integer maximum number of rows per batch; any additional
            phase-specific minimum is described above.

    Returns:
        batches (list[np.ndarray]): List of integer index arrays partitioning every row
            exactly once with approximately equal sizes.

    Raises:
        ZeroDivisionError: If batch_size is zero.
        ValueError: If a size cannot define a valid array partition.
    """

    return list(np.array_split(np.arange(size), max(1, int(np.ceil(size / batch_size)))))


def gate_coverage(available: list[int], old: set[int], limit: int, seed: int) -> dict:
    """Choose all gates, or a reproducible balanced old/new subset without labels.

    Args:
        available (list[int]): Available dense integer gate IDs, independent of labels in
            any individual batch.
        old (set[int]): Set of dense class IDs introduced before the current task.
        limit (int): Maximum integer number of gates to measure; positive route settings
            preserve old/new coverage when available.
        seed (int): Explicit integer random seed; local or derived streams preserve
            reproducibility without reseeding caller-owned generators.

    Returns:
        coverage (dict): Dict containing selected/available gate IDs, old/new counts,
            fractions and exact coverage scope.

    Raises:
        ValueError: If seed cannot initialize the NumPy random generator.
    """

    available = sorted(available)
    rng = np.random.default_rng(seed)
    groups = [rng.permutation([c for c in available if c in old]).tolist(),
              rng.permutation([c for c in available if c not in old]).tolist()]
    selected = []
    for index in range(max(map(len, groups), default=0)):
        for group in groups:
            # Transfer unused slots to the group with remaining gates.
            if index < len(group):
                selected.append(group[index])
    selected = sorted(selected[:limit])
    scope = "all_available_gates"
    # Empty banks do not imply that any gate was functionally evaluated.
    if not available:
        scope = "no_available_gates"
    # Partial coverage must distinguish single-group and old/new subsets.
    elif len(selected) < len(available):
        scope = "balanced_old_new_gate_subset" if all(groups) else (
            "old_gate_subset" if groups[0] else "new_gate_subset"
        )
    result = {
        "available_gate_ids": available, "gate_ids": selected,
        "selection_seed": seed, "max_gates": limit,
        "scope": scope,
        "aggregation": "selected gates only; no extrapolation to unmeasured gates",
    }
    for name, is_old in (("old", True), ("new", False)):
        ids = [c for c in available if (c in old) == is_old]
        measured = [c for c in selected if (c in old) == is_old]
        result[name] = {
            "available": len(ids), "measured": len(measured), "gate_ids": measured,
            "fraction": len(measured) / len(ids) if ids else None,
        }
    return result


def diagnostic_view(wrapper: object, images: tf.Tensor, level: int, seed: int) -> tuple:
    """Use the public forward-diffusion API with stateless diagnostic-only noise.

    This avoids consuming the training RNG. Cache the returned tensors for the
    before/after comparison; level zero retains the route's exactly-clean rule.

    Args:
        wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
            schedules and existing training or inference APIs.
        images (tf.Tensor): Numeric sample-major images in the configured model-input scale,
            normally float32 NHWC values in [-1, 1].
        level (int): Nonnegative integer noise level; zero means exactly clean and positive
            values index the diffusion schedule.
        seed (int): Explicit integer random seed; local or derived streams preserve
            reproducibility without reseeding caller-owned generators.

    Returns:
        view (tuple): (images, times): same floating image dtype as input and int32 timestep
            vector; zero returns clean pixels.

    Raises:
        ValueError: If the requested noise level or image shape is unsupported by the
            wrapper.
        tf.errors.InvalidArgumentError: If a schedule index is out of range.
    """

    times = tf.fill((len(images),), tf.cast(level, tf.int32))
    # Route zero denotes an exactly clean input, regardless of schedule alpha.
    if level == 0:
        return images, times
    noise = tf.random.stateless_normal(tf.shape(images), seed=[seed, 0], dtype=images.dtype)
    return wrapper.q_sample(images, times, noise), times


def _mean(values: np.ndarray) -> float | None:
    """Keep missing pair sets distinct from zero-valued distances.

    Args:
        values (np.ndarray): Numeric ndarray containing a pair family; an empty array has no
            defined mean.

    Returns:
        mean (float | None): Python float arithmetic mean, or None when the pair array is
            empty.

    Raises:
        TypeError: If values cannot be averaged as numbers.
    """

    return float(np.mean(values)) if values.size else None


def one_vs_rest(features: np.ndarray, labels: np.ndarray, focus: int) -> dict:
    """Acquisition's two components on all permitted held-out rows (no predictor).

    Args:
        features (np.ndarray): Finite numeric matrix of shape [N, D]; objective and
            classifier paths compute in float32.
        labels (np.ndarray): Sparse integer label vector aligned with the image rows; the
            label convention for this operation is described above.
        focus (int): Integer selected-class ID; its observations define positives and all
            other classes define negatives.

    Returns:
        geometry (dict): Dict of pair counts, float distances and availability labels; empty
            pair families have None means.

    Raises:
        ValueError: If labels and feature rows cannot be aligned.
        tf.errors.InvalidArgumentError: If features are empty or nonfinite.
    """

    z = normalized_features(features).numpy().astype("float64")
    positives, negatives = z[labels == focus], z[labels != focus]
    count = len(positives)
    within = (1. - np.clip(positives @ positives.T, -1., 1.))[~np.eye(count, dtype=bool)]
    cross = np.clip(positives @ negatives.T, -1., 1.)
    return {
        "positive_examples": count, "negative_examples": len(negatives),
        "positive_ordered_pairs": within.size, "positive_negative_pairs": cross.size,
        "within_class_cosine_distance": _mean(within),
        "one_vs_rest_squared_cosine": _mean(cross ** 2),
        "one_vs_rest_cosine_distance": _mean(1. - cross),
        "availability": "complete" if within.size and cross.size else "insufficient_positive_or_negative_rows",
    }


def class_geometry(features: np.ndarray, labels: np.ndarray, old: set[int]) -> dict:
    """Pair-weighted normalized hidden geometry, with explicit old/new scopes.

    Cosine distance is 1-cosine (smaller within, larger between indicates greater
    class separation); squared cosine separately matches acquisition's goal of
    orthogonality. Zero features remain zero under the objective's normalization.

    Args:
        features (np.ndarray): Finite numeric matrix of shape [N, D]; objective and
            classifier paths compute in float32.
        labels (np.ndarray): Sparse integer label vector aligned with the image rows; the
            label convention for this operation is described above.
        old (set[int]): Set of dense class IDs introduced before the current task.

    Returns:
        geometry (dict): Nested dict of all/old/new/cross-group and per-class pair-weighted
            float statistics with explicit counts.

    Raises:
        ValueError: If labels and feature rows cannot be aligned.
        tf.errors.InvalidArgumentError: If features are empty or nonfinite.
    """

    z = normalized_features(features).numpy().astype("float64")
    cosine = np.clip(z @ z.T, -1., 1.)
    same = labels[:, None] == labels[None, :]
    off_diagonal = ~np.eye(len(labels), dtype=bool)
    is_old = np.isin(labels, list(old))
    scopes = {
        "all": np.ones_like(same),
        "old": is_old[:, None] & is_old[None, :],
        "new": ~is_old[:, None] & ~is_old[None, :],
        "old_new": is_old[:, None] != is_old[None, :],
    }
    result = {"aggregation": "mean over ordered pairs in each scope; self-pairs excluded"}
    for name, mask in scopes.items():
        within, between = mask & same & off_diagonal, mask & ~same
        result[name] = {
            "within_pairs": int(within.sum()), "between_pairs": int(between.sum()),
            "within_class_cosine_distance": _mean(1. - cosine[within]),
            "between_class_cosine_distance": _mean(1. - cosine[between]),
            "between_class_squared_cosine": _mean(cosine[between] ** 2),
        }
    result["per_class"] = {int(c): one_vs_rest(features, labels, int(c)) for c in np.unique(labels)}
    return result


def numeric_changes(before: dict, after: dict) -> dict:
    """After-minus-before numeric metric deltas; preserve missingness recursively.

    Args:
        before (dict): Earlier nested metric dict, with None for unavailable measurements.
        after (dict): Later nested metric dict; only matching numeric fields are subtracted.

    Returns:
        changes (dict): Dict of Python float after-minus-before differences; corresponding
            unavailable endpoints remain None.

    Raises:
        RecursionError: If cyclic metric dictionaries cannot be recursively traversed.
    """

    result = {}
    for key in before.keys() & after.keys():
        first, last = before[key], after[key]
        # Preserve metric nesting when comparing corresponding scopes.
        if isinstance(first, dict) and isinstance(last, dict):
            result[key] = numeric_changes(first, last)
        # A missing endpoint never supplies an invented numeric delta.
        elif first is None or last is None:
            result[key] = None
        # Scope strings and other metadata are not arithmetic measurements.
        elif isinstance(first, (int, float, np.number)) and isinstance(last, (int, float, np.number)):
            result[key] = float(last - first)
    return result
