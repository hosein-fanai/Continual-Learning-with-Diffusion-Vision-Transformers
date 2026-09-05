"""Mechanistic and replay-quality measurements for continual experiments.

The functions in this module are NumPy-only and do not mutate models or data.
They are therefore suitable for optional post-task reporting and for offline
analysis of cached activations or replay pools.
"""

from __future__ import annotations

import numpy as np

from collections.abc import Sequence


def _label_ids(
    labels: np.ndarray | Sequence[int],
    class_num: int | None = None,
) -> np.ndarray:
    """Convert sparse vectors/columns or one-hot labels to integer IDs.

    Args:
        labels (numpy.ndarray | Sequence[int]): Sparse vector/column labels or
            a rank-two one-hot/probability matrix with multiple columns.
        class_num (int | None): Known prediction width, used to recognize a
            one-column one-hot matrix for a one-class classifier.
            Defaults to ``None``, interpreting one-column input as sparse IDs.

    Returns:
        numpy.ndarray: ``int64`` label IDs with shape ``[samples]``; an existing
        sparse int64 array may share storage. One-hot probabilities are decoded
        by argmax without checking their normalization.

    Raises:
        ValueError: If labels are not rank one/two or contain nonfinite IDs.
    """

    values = np.asarray(labels)
    # Reject label structures that cannot identify one class per sample.
    if values.ndim not in (1, 2):
        raise ValueError("labels must be sparse rank-one or one-hot rank-two.")
    # Check before argmax, which otherwise hides invalid probability entries.
    if not np.all(np.isfinite(values)):
        raise ValueError("labels must contain finite class IDs.")

    # Decode probability columns, including a known one-class one-hot target.
    if values.ndim == 2 and (values.shape[1] != 1 or class_num == 1):
        values = np.argmax(values, axis=-1)

    return values.reshape(-1).astype("int64", copy=False)


def _probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    """Validate and return a floating multiclass probability matrix.

    Values must be finite and within ``[0, 1]``. Every row must sum to one
    within ``rtol=1e-5`` and ``atol=1e-7``; logits are never renormalized.
    A single class is supported for the first task of a continual schedule.

    Args:
        probabilities (numpy.ndarray): Candidate probabilities shaped
            ``[samples, classes]``.

    Returns:
        numpy.ndarray: Validated ``float64`` array with unchanged shape. The
        result may share storage with a float64 input; values are not mutated.

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
            Defaults to ``15``.
        epsilon (float): Positive lower probability bound used only inside
            logarithms.
            Defaults to ``1e-12``.

    Returns:
        dict[str, float]: ``accuracy``, ``entropy``, ``nll``, ``brier`` and
        ``ece`` averaged over independent examples.

    Raises:
        ValueError: If inputs are empty, misaligned, or outside their domains.
    """

    probs = _probability_matrix(probabilities)
    targets = _label_ids(labels, class_num=probs.shape[1])

    bins = int(bins)
    epsilon = float(epsilon)
    # Keep ECE binning and logarithms mathematically defined.
    if bins < 1 or epsilon <= 0.:
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
    clipped = np.clip(probs, epsilon, 1.)
    entropy = -np.sum(probs * np.log(clipped), axis=1)
    nll = -np.log(clipped[rows, targets])
    onehot = np.eye(probs.shape[1], dtype="float64")[targets]
    brier = np.sum(np.square(probs - onehot), axis=1)

    bin_ids = np.minimum((confidence * bins).astype("int64"), bins - 1)
    ece = 0.
    for bin_id in range(bins):
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
    flattened. Use feature or sample matrices according to which dimensions
    are smaller, and leave the supplied representations unchanged.

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
    x = x - np.mean(x, axis=0, keepdims=True)
    y = y - np.mean(y, axis=0, keepdims=True)
    # Use sample matrices when the representations are wider than the batch.
    if max(x.shape[1], y.shape[1]) > len(x):
        xx = x @ x.T
        yy = y @ y.T
        cross_norm = float(np.sum(xx * yy))
        x_norm = float(np.sum(np.square(xx)))
        y_norm = float(np.sum(np.square(yy)))
    # Narrow representations use the equivalent feature-space calculation.
    else:
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

    Each representation is converted to float64 and flattened after its sample
    axis. For each class present in both snapshots, compute the L2 norm of the
    difference between its sample-mean feature vectors. Classes contribute
    equally to the overall mean, regardless of their sample counts.

    Args:
        previous (numpy.ndarray): Earlier finite representations shaped
            ``[N_old, ...]`` with at least one row.
        previous_labels (numpy.ndarray | Sequence[int]): Aligned sparse labels
            shaped ``[N_old]``/``[N_old, 1]`` or multi-column one-hot labels.
        current (numpy.ndarray): Later finite representations shaped
            ``[N_new, ...]`` with the same flattened feature width; sample
            counts and row identities may differ between snapshots.
        current_labels (numpy.ndarray | Sequence[int]): Labels for the current
            rows in the same formats as ``previous_labels``.

    Returns:
        dict[str, object]: ``mean_centroid_drift`` (float) and
        ``per_class_centroid_drift`` (string class ID to float distance).
        With no shared classes the mapping is empty and the mean is ``NaN``.

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

    # Average shared-class drift when available; mark no shared classes as undefined.
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

    Shuffle candidates independently within each class, then consume one per
    class per round in sorted class-ID order. Exhausted classes are skipped;
    a partial final round favors earlier sorted classes. The local generator
    advances, while labels remain unchanged.

    Args:
        labels (numpy.ndarray): Rank-one integer candidate labels.
        budget (int): Nonnegative maximum number of indices to return. Zero
            returns an empty array; a budget beyond available rows returns all.
        rng (numpy.random.Generator): Local generator used to randomize ties.

    Returns:
        numpy.ndarray: Unique ``int64`` candidate indices in selection order,
        with length ``min(budget, len(labels))`` for valid nonnegative budgets.
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
        samples (numpy.ndarray): Candidate replay examples shaped ``[N, ...]``;
            non-sample dimensions and dtype are preserved in the selection.
        labels (numpy.ndarray | Sequence[int]): Aligned sparse ``[N]``/
            ``[N, 1]`` labels or multi-column one-hot conditioning labels.
        budget (int): Nonnegative maximum selected example count.
        strategy (str): ``all``, ``uniform``, ``random``, ``confidence``,
            ``surprise``, or ``confidence_surprise``.
            Defaults to ``'all'``.
        probabilities (numpy.ndarray | None): Teacher probabilities required by
            confidence/surprise strategies, shaped ``[N, classes]`` with class
            columns indexed by label ID. Defaults to ``None``, valid only for
            ``all``, ``uniform``, and ``random``; those modes ignore this input.
        seed (int | None): Local selection seed.
            Defaults to ``None`` to initialize a local generator from entropy.
            Global NumPy RNG state is not changed.
        surprise_weight (float): Weight in ``[0, 1]`` assigned to standardized
            surprise in the combined score.
            Defaults to ``0.5``.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, dict[str, object]]: Selected samples,
        selected ``int64`` labels, and diagnostics containing strategy, candidate/
        selected counts, and per-class allocations (string class IDs). Scored
        strategies add selected score mean/std when the selection is nonempty.
        At most ``min(budget, N)`` rows are returned; zero budget returns empty
        arrays with preserved trailing sample dimensions.

    Raises:
        ValueError: If inputs or strategy parameters are invalid.
    """

    x = np.asarray(samples)
    y = _label_ids(labels)
    budget = int(budget)
    surprise_weight = float(surprise_weight)
    # Require aligned inputs and valid score interpolation.
    if budget < 0 or len(x) != len(y) \
    or not 0. <= surprise_weight <= 1.:
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
        # Draw random candidates for a nonzero budget; return no indices for zero budget.
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

    Uses float64 squared norms and a Gram matrix, clips negative roundoff to
    zero, and averages distances above the diagonal. Each unordered distinct
    row pair receives equal weight; the calculation uses quadratic memory.

    Args:
        values (numpy.ndarray): Numeric finite samples shaped ``[N, ...]``;
            every non-sample dimension is flattened into the feature vector.

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
        samples (numpy.ndarray): Selected finite replay examples shaped
            ``[N, ...]``; spatial/feature dimensions are flattened for distances.
        labels (numpy.ndarray | Sequence[int]): Aligned sparse ``[N]``/
            ``[N, 1]`` or multi-column one-hot conditioning labels.
        expected_classes (Sequence[int]): Old classes that replay should cover.
        probabilities (numpy.ndarray | None): Optional evaluator probabilities
            shaped ``[N, classes]``, used for label consistency and calibration.
            Defaults to ``None``, omitting those result fields.
        previous_samples (numpy.ndarray | None): Optional earlier replay pool.
            Defaults to ``None``, disabling centroid drift. When supplied,
            ``previous_labels`` must also be supplied and flattened feature
            width must match the current pool when both are nonempty.
        previous_labels (numpy.ndarray | Sequence[int] | None): Labels aligned
            with ``previous_samples`` for centroid drift.
            Defaults to ``None``; required exactly when the previous pool is
            supplied, in the same label formats as ``labels``.
        max_diversity_samples (int): Positive cap on quadratic diversity work.
            Defaults to ``512``.
        seed (int | None): Local seed for diversity subsampling.
            Defaults to ``None`` for entropy-seeded local sampling, without
            changing global NumPy RNG state.

    Returns:
        dict[str, object]: ``sample_count``, fraction ``class_coverage``,
        ``normalized_label_entropy`` (label entropy divided by log expected
        class count), ``pixel_diversity`` (mean pairwise Euclidean distance), and
        string-keyed ``class_counts``. One nonempty expected class has entropy
        1; empty pools have undefined entropy/diversity, and no expected classes
        gives undefined coverage (all represented by ``NaN``). Optional evaluator
        input adds ``label_consistency`` and ``calibration``; nonempty current
        and prior pools add ``distribution_drift`` from ``class_centroid_drift``.

    Raises:
        ValueError: If replay inputs, expected classes, or prior pairs conflict.
    """

    x = np.asarray(samples)
    y = _label_ids(labels)
    expected = sorted({int(class_id) for class_id in expected_classes})

    max_diversity_samples = int(max_diversity_samples)
    # Require aligned replay and a positive diversity cap.
    if len(x) != len(y) or max_diversity_samples < 1:
        raise ValueError("samples/labels must align and diversity cap must be positive.")

    # List represented classes for a nonempty replay pool; otherwise keep the list empty.
    present = sorted(np.unique(y).tolist()) if len(y) else []
    # Keep the normalized entropy denominator tied to the declared class set.
    unexpected = sorted(set(present) - set(expected))
    # Reject replay labels that would make coverage and entropy incomparable.
    if unexpected:
        raise ValueError(
            "replay labels must belong to expected_classes; "
            f"unexpected labels: {unexpected}."
        )

    # Normalize coverage by expected classes; leave an empty class universe undefined.
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
    # Subsample diversity inputs when replay exists; skip sampling an empty pool.
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
        # Measure label consistency for nonempty replay; mark an empty pool unavailable.
        result["label_consistency"] = float(np.mean(
            np.argmax(probs, axis=1) == y
        )) if len(y) else float("nan")

        # Compute calibration only when replay examples exist.
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
