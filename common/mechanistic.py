"""Mechanistic and replay-quality measurements for continual experiments.

The functions in this module are NumPy-only and do not mutate models or data.
They are therefore suitable for optional post-task reporting and for offline
analysis of cached activations or replay pools.
"""

from __future__ import annotations

import numpy as np

from collections.abc import Sequence
from numbers import Integral, Real


def _label_ids(labels: np.ndarray | Sequence[int]) -> np.ndarray:
    """Convert sparse or one-hot labels to a one-dimensional integer array.

    Args:
        labels (numpy.ndarray | Sequence[int]): Sparse labels or a rank-two
            one-hot/probability matrix.

    Returns:
        numpy.ndarray: Integer label IDs with shape ``[samples]``.

    Raises:
        ValueError: If labels are not rank one/two or contain nonintegral IDs.
    """

    values = np.asarray(labels)
    # Convert matrix labels by selecting their represented class.
    if values.ndim == 2:
        # Argmax would silently turn NaN/Inf matrix entries into valid-looking IDs.
        if not np.all(np.isfinite(values)):
            raise ValueError("labels must contain only finite matrix values.")

        values = np.argmax(values, axis=-1)
    # Reject structures that cannot identify one class per sample.
    elif values.ndim != 1:
        raise ValueError("labels must be sparse rank-one or one-hot rank-two.")

    # Require integer-valued sparse labels before casting them.
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError("labels must contain finite integer class IDs.")

    return values.astype("int64", copy=False)


def _probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    """Validate and return a floating multiclass probability matrix.

    Args:
        probabilities (numpy.ndarray): Candidate probabilities shaped
            ``[samples, classes]``.

    Returns:
        numpy.ndarray: Validated ``float64`` probabilities.

    Raises:
        ValueError: If rank, width, finiteness, range, or row sums are invalid.
    """

    values = np.asarray(probabilities, dtype="float64")

    # A one-class first task is a valid (degenerate) continual distribution.
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("probabilities must have shape [samples, classes>=1].")
    # Reject NaN, infinity, and values outside the probability interval.
    if not np.all(np.isfinite(values)) or np.any(values < 0.) or np.any(values > 1.):
        raise ValueError("probabilities must be finite values in [0, 1].")
    # Refuse unnormalized logits so calibration metrics keep their meaning.
    if not np.allclose(np.sum(values, axis=1), 1., rtol=1e-5, atol=1e-7):
        raise ValueError("each probability row must sum to one.")

    return values


def calibration_metrics(
    probabilities: np.ndarray, 
    labels: np.ndarray | Sequence[int], 
    bins: int = 15, 
    epsilon: float = 1e-12
) -> dict[str, float]:
    """Compute accuracy, entropy, NLL, multiclass Brier score, and ECE.

    Expected calibration error uses equal-width confidence bins and weights
    each occupied bin by its sample fraction. The Brier score is the mean sum
    of squared multiclass probability errors per sample.

    Args:
        probabilities (numpy.ndarray): Normalized class probabilities shaped
            ``[samples, classes]``.
        labels (numpy.ndarray | Sequence[int]): Matching sparse or one-hot
            ground-truth labels.
        bins (int): Positive number of equal-width ECE bins.
        epsilon (float): Positive lower probability bound used only inside
            logarithms.

    Returns:
        dict[str, float]: ``accuracy``, ``entropy``, ``nll``, ``brier`` and
        ``ece`` averaged over independent examples.

    Raises:
        TypeError: If ``bins`` is not a non-boolean integer.
        ValueError: If inputs are empty, misaligned, or outside their domains.
    """

    probs = _probability_matrix(probabilities)
    targets = _label_ids(labels)

    # Reject nonintegral bin counts explicitly rather than truncating them.
    if isinstance(bins, bool) or not isinstance(bins, Integral):
        raise TypeError("bins must be a non-boolean integer.")
    # Keep ECE binning and logarithms mathematically defined.
    if int(bins) < 1 or not isinstance(epsilon, Real) or float(epsilon) <= 0.:
        raise ValueError("bins and epsilon must be positive.")
    # Require a nonempty one-to-one probability/target alignment.
    if len(probs) == 0 or len(targets) != len(probs):
        raise ValueError("probabilities and labels must be nonempty and aligned.")
    # Keep label indexing within the represented probability width.
    if np.any(targets < 0) or np.any(targets >= probs.shape[1]):
        raise ValueError("labels exceed the probability matrix class range.")

    rows = np.arange(len(targets))
    predictions = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    correct = (predictions == targets).astype("float64")
    clipped = np.clip(probs, float(epsilon), 1.)
    entropy = -np.sum(probs * np.log(clipped), axis=1)
    nll = -np.log(clipped[rows, targets])
    onehot = np.eye(probs.shape[1], dtype="float64")[targets]
    brier = np.sum(np.square(probs - onehot), axis=1)

    bin_ids = np.minimum((confidence * int(bins)).astype("int64"), int(bins) - 1)
    ece = 0.
    for bin_id in range(int(bins)):
        mask = bin_ids == bin_id
        # Add only occupied bins to avoid means over empty arrays.
        if np.any(mask):
            weight = float(np.mean(mask))
            ece += weight * abs(
                float(np.mean(correct[mask])) - 
                float(np.mean(confidence[mask]))
            )

    return {
        "accuracy": float(np.mean(correct)), 
        "entropy": float(np.mean(entropy)), 
        "nll": float(np.mean(nll)), 
        "brier": float(np.mean(brier)), 
        "ece": float(ece)
    }


def linear_cka(first: np.ndarray, second: np.ndarray) -> float:
    """Compute linear centered-kernel alignment between two representations.

    Samples occupy rows and arbitrary remaining representation dimensions are
    flattened. This feature-space identity avoids materializing quadratic Gram
    matrices and is therefore suitable for moderately large probe sets.

    Args:
        first (numpy.ndarray): First representation with a sample axis.
        second (numpy.ndarray): Second representation with the same sample count.

    Returns:
        float: Linear CKA in ``[0, 1]`` or ``NaN`` for a constant representation.

    Raises:
        ValueError: If fewer than two aligned samples are supplied.
    """

    x = np.asarray(first, dtype="float64")
    y = np.asarray(second, dtype="float64")

    # Require two aligned observations so centering is meaningful.
    if x.ndim < 2 or y.ndim < 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("representations require at least two aligned samples.")

    x = x.reshape((len(x), -1))
    y = y.reshape((len(y), -1))
    x -= np.mean(x, axis=0, keepdims=True)
    y -= np.mean(y, axis=0, keepdims=True)
    cross_norm = float(np.sum(np.square(x.T @ y)))
    x_norm = float(np.sum(np.square(x.T @ x)))
    y_norm = float(np.sum(np.square(y.T @ y)))
    denominator = np.sqrt(x_norm * y_norm)

    # Mark a constant representation as unavailable instead of dividing by zero.
    if denominator == 0.:
        return float("nan")

    return float(np.clip(cross_norm / denominator, 0., 1.))


def class_centroid_drift(
    previous: np.ndarray, 
    previous_labels: np.ndarray | Sequence[int], 
    current: np.ndarray, 
    current_labels: np.ndarray | Sequence[int]
) -> dict[str, object]:
    """Measure Euclidean representation-centroid movement for shared classes.

    Args:
        previous (numpy.ndarray): Earlier representations with samples in rows.
        previous_labels (numpy.ndarray | Sequence[int]): Earlier class labels.
        current (numpy.ndarray): Later representations with samples in rows.
        current_labels (numpy.ndarray | Sequence[int]): Later class labels.

    Returns:
        dict[str, object]: Mean drift and a string-keyed per-class mapping.

    Raises:
        ValueError: If arrays and labels are empty, misaligned, or incompatible.
    """

    old_x = np.asarray(previous, dtype="float64")
    new_x = np.asarray(current, dtype="float64")
    old_y = _label_ids(previous_labels)
    new_y = _label_ids(current_labels)

    # Require aligned nonempty arrays with a common flattened feature width.
    if len(old_x) == 0 or len(new_x) == 0 or len(old_x) != len(old_y) or len(new_x) != len(new_y):
        raise ValueError("representations and labels must be nonempty and aligned.")
    old_x = old_x.reshape((len(old_x), -1))
    new_x = new_x.reshape((len(new_x), -1))
    # Reject incomparable representation spaces.
    if old_x.shape[1] != new_x.shape[1]:
        raise ValueError("previous and current feature widths must match.")

    shared = sorted(set(old_y.tolist()) & set(new_y.tolist()))
    per_class = {
        str(class_id): float(np.linalg.norm(
            np.mean(new_x[new_y == class_id], axis=0)
            - np.mean(old_x[old_y == class_id], axis=0)
        ))
        for class_id in shared
    }
    values = list(per_class.values())

    return {
        "mean_centroid_drift": float(np.mean(values)) if values else float("nan"), 
        "per_class_centroid_drift": per_class
    }


def _balanced_indices(
    labels: np.ndarray, 
    budget: int, 
    rng: np.random.Generator
) -> np.ndarray:
    """Select a near-equal number of candidate indices from every class.

    Args:
        labels (numpy.ndarray): Rank-one integer candidate labels.
        budget (int): Number of indices to return.
        rng (numpy.random.Generator): Local generator used to randomize ties.

    Returns:
        numpy.ndarray: Unique candidate indices with length ``budget``.
    """

    queues = {
        int(class_id): list(rng.permutation(np.flatnonzero(labels == class_id)))
        for class_id in sorted(np.unique(labels).tolist())
    }

    selected: list[int] = []
    while len(selected) < budget:
        progressed = False
        for class_id in queues:
            # Take one candidate per class per round while one remains.
            if queues[class_id] and len(selected) < budget:
                selected.append(int(queues[class_id].pop()))
                progressed = True

        # Stop when every class queue has been exhausted.
        if not progressed:
            break

    return np.asarray(selected, dtype="int64")


def select_replay_candidates(
    samples: np.ndarray, 
    labels: np.ndarray | Sequence[int], 
    budget: int, 
    strategy: str = "all", 
    probabilities: np.ndarray | None = None, 
    seed: int | None = None, 
    surprise_weight: float = 0.5
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Select one replay budget from a common candidate pool.

    ``confidence`` keeps the most confidently recognized candidates;
    ``surprise`` keeps the largest teacher NLL; and
    ``confidence_surprise`` ranks a weighted sum of standardized confidence and
    surprise. ``uniform`` is class-balanced, while ``random`` is the matched
    uninformative control. ``all`` preserves candidate order and is the legacy
    no-gating behavior.

    Args:
        samples (numpy.ndarray): Candidate replay examples.
        labels (numpy.ndarray | Sequence[int]): Candidate conditioning labels.
        budget (int): Nonnegative maximum selected example count.
        strategy (str): ``all``, ``uniform``, ``random``, ``confidence``,
            ``surprise``, or ``confidence_surprise``.
        probabilities (numpy.ndarray | None): Teacher probabilities required by
            confidence/surprise strategies.
        seed (int | None): Local selection seed.
        surprise_weight (float): Weight in ``[0, 1]`` assigned to standardized
            surprise in the combined score.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, dict[str, object]]: Selected samples,
        selected integer labels, and gate-allocation diagnostics.

    Raises:
        TypeError: If ``budget`` is not a non-boolean integer.
        ValueError: If inputs or strategy parameters are invalid.
    """

    x = np.asarray(samples)
    y = _label_ids(labels)
    # Reject booleans/nonintegral replay budgets without silent coercion.
    if isinstance(budget, bool) or not isinstance(budget, Integral):
        raise TypeError("budget must be a non-boolean integer.")

    budget = int(budget)
    # Require aligned inputs and valid score interpolation.
    if budget < 0 or len(x) != len(y) or not isinstance(surprise_weight, Real) \
    or not 0. <= float(surprise_weight) <= 1.:
        raise ValueError("candidate inputs, budget, or surprise_weight are invalid.")

    strategy = str(strategy).lower()
    valid = {
        "all", "uniform", "random", "confidence", "surprise",
        "confidence_surprise",
    }
    # Reject misspelled treatments rather than silently changing the experiment.
    if strategy not in valid:
        raise ValueError(f"strategy must be one of {sorted(valid)}.")
    selected_num = min(budget, len(x))
    rng = np.random.default_rng(seed)

    # Preserve exact legacy order when no candidate reduction is requested.
    if strategy == "all":
        indices = np.arange(selected_num, dtype="int64")
        scores = None
    # Allocate the replay budget approximately equally across represented labels.
    elif strategy == "uniform":
        indices = _balanced_indices(y, selected_num, rng)
        scores = None
    # Draw the matched uninformative control from the same candidate pool.
    elif strategy == "random":
        indices = rng.choice(len(x), size=selected_num, replace=False) \
            if selected_num else np.empty((0,), dtype="int64")
        scores = None
    # Rank candidates with frozen-teacher probabilities.
    else:
        # Require a valid teacher distribution for cognitive gates.
        if probabilities is None:
            raise ValueError(f"{strategy} selection requires probabilities.")
        probs = _probability_matrix(probabilities)
        # Keep the teacher scores aligned and able to represent every label.
        if len(probs) != len(x) or np.any(y < 0) or np.any(y >= probs.shape[1]):
            raise ValueError("teacher probabilities and candidate labels are incompatible.")
        confidence = np.max(probs, axis=1)
        surprise = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.))
        # Use the requested single cognitive signal directly.
        if strategy == "confidence":
            scores = confidence
        # Select high prediction error for surprise-only consolidation.
        elif strategy == "surprise":
            scores = surprise
        # Combine scale-free confidence and surprise with an explicit weight.
        else:
            confidence_z = (confidence - np.mean(confidence)) / (np.std(confidence) + 1e-12)
            surprise_z = (surprise - np.mean(surprise)) / (np.std(surprise) + 1e-12)
            scores = (
                (1. - float(surprise_weight)) * confidence_z
                + float(surprise_weight) * surprise_z
            )

        tie_order = rng.permutation(len(x))
        ranked = tie_order[np.argsort(scores[tie_order], kind="stable")[::-1]]
        indices = ranked[:selected_num]

    selected_labels = y[indices]
    class_counts = {
        str(class_id): int(np.sum(selected_labels == class_id))
        for class_id in sorted(np.unique(y).tolist())
    }
    diagnostics: dict[str, object] = {
        "strategy": strategy,
        "candidate_count": int(len(x)),
        "selected_count": int(len(indices)),
        "class_counts": class_counts,
    }

    # Report selected gate-score moments only for scored strategies.
    if scores is not None and len(indices):
        diagnostics["selected_score_mean"] = float(np.mean(scores[indices]))
        diagnostics["selected_score_std"] = float(np.std(scores[indices]))

    return x[indices], selected_labels, diagnostics


def _mean_pairwise_distance(values: np.ndarray) -> float:
    """Compute mean Euclidean distance over unique pairs of flattened rows.

    Args:
        values (numpy.ndarray): At least zero samples with arbitrary features.

    Returns:
        float: Mean pair distance or ``NaN`` when fewer than two rows exist.
    """

    x = np.asarray(values, dtype="float64")

    # Pairwise diversity is undefined for fewer than two examples.
    if len(x) < 2:
        return float("nan")

    x = x.reshape((len(x), -1))
    squared = np.maximum(
        np.sum(np.square(x), axis=1)[:, None]
        + np.sum(np.square(x), axis=1)[None, :]
        - 2. * x @ x.T,
        0.,
    )
    upper = np.triu_indices(len(x), k=1)

    return float(np.mean(np.sqrt(squared[upper])))


def replay_quality_metrics(
    samples: np.ndarray, 
    labels: np.ndarray | Sequence[int], 
    expected_classes: Sequence[int], 
    probabilities: np.ndarray | None = None, 
    previous_samples: np.ndarray | None = None, 
    previous_labels: np.ndarray | Sequence[int] | None = None, 
    max_diversity_samples: int = 512, 
    seed: int | None = None
) -> dict[str, object]:
    """Measure replay consistency, coverage, diversity, and distribution drift.

    Args:
        samples (numpy.ndarray): Selected replay examples.
        labels (numpy.ndarray | Sequence[int]): Their conditioning labels.
        expected_classes (Sequence[int]): Old classes that replay should cover.
        probabilities (numpy.ndarray | None): Optional evaluator probabilities
            used for label consistency and calibration.
        previous_samples (numpy.ndarray | None): Optional earlier replay pool.
        previous_labels (numpy.ndarray | Sequence[int] | None): Labels aligned
            with ``previous_samples`` for centroid drift.
        max_diversity_samples (int): Positive cap on quadratic diversity work.
        seed (int | None): Local seed for diversity subsampling.

    Returns:
        dict[str, object]: Coverage, normalized label entropy, diversity,
        optional calibration/consistency, and optional centroid drift.

    Raises:
        TypeError: If ``max_diversity_samples`` is not an integer.
        ValueError: If replay inputs, expected classes, or prior pairs conflict.
    """

    x = np.asarray(samples)
    y = _label_ids(labels)
    expected = sorted({int(class_id) for class_id in expected_classes})

    # Enforce an explicit bounded quadratic diagnostic budget.
    if isinstance(max_diversity_samples, bool) or not isinstance(max_diversity_samples, Integral):
        raise TypeError("max_diversity_samples must be a non-boolean integer.")
    # Require aligned replay and a positive diversity cap.
    if len(x) != len(y) or int(max_diversity_samples) < 1:
        raise ValueError("samples/labels must align and diversity cap must be positive.")

    present = sorted(np.unique(y).tolist()) if len(y) else []
    # Keep the normalized entropy denominator tied to the declared class set.
    unexpected = sorted(set(present) - set(expected))
    # Reject replay labels that would make coverage and entropy incomparable.
    if unexpected:
        raise ValueError(
            "replay labels must belong to expected_classes; "
            f"unexpected labels: {unexpected}."
        )

    coverage = float(
        len(set(present) & set(expected)) / len(expected)
    ) if expected else float("nan")
    counts = np.asarray([
        np.sum(y == class_id) 
        for class_id in present
    ], dtype="float64")

    # Normalize class entropy to one for a uniform distribution over expected classes.
    if len(counts) and len(expected) > 1:
        frequencies = counts / np.sum(counts)
        label_entropy = float(-np.sum(frequencies * np.log(frequencies)) / np.log(len(expected)))
    # A single expected class has complete balance by construction.
    elif len(counts) and len(expected) == 1:
        label_entropy = 1.
    # Keep an empty pool explicitly unavailable.
    else:
        label_entropy = float("nan")

    rng = np.random.default_rng(seed)
    diversity_indices = rng.choice(
        len(x), 
        size=min(len(x), int(max_diversity_samples)), 
        replace=False
    ) if len(x) else np.empty((0,), dtype="int64")
    result: dict[str, object] = {
        "sample_count": int(len(x)), 
        "class_coverage": coverage, 
        "normalized_label_entropy": label_entropy, 
        "pixel_diversity": _mean_pairwise_distance(x[diversity_indices]), 
        "class_counts": {str(class_id): int(np.sum(y == class_id)) for class_id in expected}
    }

    # Add evaluator-based consistency and calibration when probabilities exist.
    if probabilities is not None:
        probs = _probability_matrix(probabilities)
        # Keep evaluator rows aligned with selected replay examples.
        if len(probs) != len(x):
            raise ValueError("probabilities must align with replay samples.")
        result["label_consistency"] = float(np.mean(
            np.argmax(probs, axis=1) == y
        )) if len(y) else float("nan")

        result["calibration"] = calibration_metrics(probs, y) if len(y) else {}

    has_previous_samples = previous_samples is not None
    has_previous_labels = previous_labels is not None

    # Reject partially specified prior pools because drift would be ambiguous.
    if has_previous_samples != has_previous_labels:
        raise ValueError("previous_samples and previous_labels must be supplied together.")
    # Add class-centroid replay-distribution drift when both pools are nonempty.
    if has_previous_samples and len(np.asarray(previous_samples)) and len(x):
        result["distribution_drift"] = class_centroid_drift(
            np.asarray(previous_samples), 
            previous_labels, 
            x, y
        )

    return result
