"""Numerical objectives for semantic modulation acquisition and consolidation.

The acquisition objective is an OPL-inspired, selected-class separation loss;
it is not a claimed reproduction of TMCL. Consolidation uses asymmetric,
instance-matched InfoNCE with every other target row as a negative. Callers
average losses over their configured noise draws and freeze the target network
itself: stopping target gradients here does not freeze a shared, changing model.
All reductions and similarity calculations use float32. The separately named
``orthogonal_projection_loss`` and ``multiview_barlow_loss`` implement the
default OPL and jointly normalized multiview Barlow objectives from the pinned
official TMCL source. They do not change the legacy objectives above.

Source: Dendritic-Learning-Group/tmcl, commit
2cb306cc5cbe6d13a3dd4991998ee3e7c2948c8b, tmcl/nn/losses/contrastive_opl.py
and tmcl/nn/losses/mv_barlow.py. Paper: https://arxiv.org/html/2505.14125v3.
"""

from __future__ import annotations

import tensorflow as tf


def _feature_matrix(features: tf.Tensor, name: str) -> tf.Tensor:
    """Validate a nonempty, finite batch of feature vectors in float32.

    Args:
        features (tf.Tensor): Finite numeric matrix of shape [N, D]; objective and
            classifier paths compute in float32.
        name (str): Human-readable field name included in validation error messages.

    Returns:
        features (tf.Tensor): Finite nonempty tf.float32 matrix [N, D].

    Raises:
        ValueError: If statically known rank is not two.
        tf.errors.InvalidArgumentError: If feature dimensions are empty, rank is invalid at
            runtime or values are nonfinite.
    """

    values = tf.cast(tf.convert_to_tensor(features), tf.float32)
    checks = [
        tf.debugging.assert_rank(values, 2, message=f"{name} must have rank 2"), 
        tf.debugging.assert_all_finite(values, f"{name} must be finite"), 
        tf.debugging.assert_positive(
            tf.shape(values), message=f"{name} must have nonempty dimensions"
        )
    ]
    with tf.control_dependencies(checks):
        return tf.identity(values)


def _scalar(value: float | tf.Tensor, name: str) -> tf.Tensor:
    """Return a validated, finite float32 scalar.

    Args:
        value (float | tf.Tensor): Numeric scalar cast to tf.float32; vectors, NaN and
            infinity are invalid.
        name (str): Human-readable field name included in validation error messages.

    Returns:
        scalar (tf.Tensor): Finite scalar tf.float32 tensor.

    Raises:
        ValueError: If the statically known input rank is not scalar.
        tf.errors.InvalidArgumentError: If runtime rank or finiteness checks fail.
    """

    value = tf.cast(tf.convert_to_tensor(value), tf.float32)
    checks = [
        tf.debugging.assert_rank(value, 0, message=f"{name} must be scalar"), 
        tf.debugging.assert_all_finite(value, f"{name} must be finite")
    ]
    with tf.control_dependencies(checks):
        return tf.identity(value)


def normalized_features(features: tf.Tensor) -> tf.Tensor:
    """Return finite, row-normalized float32 features, with zero rows left zero.

    Row scaling before normalization avoids overflow for large finite values.
    The usual ``l2_normalize`` epsilon is applied after scaling; every nonzero
    row then has at least one element of magnitude one and a unit output norm.

    Args:
        features (tf.Tensor): Finite numeric matrix of shape [N, D]; objective and
            classifier paths compute in float32.

    Returns:
        normalized (tf.Tensor): Tf.float32 matrix [N, D] with unit nonzero rows and exact
            zero rows preserved.

    Raises:
        ValueError: If the input has statically invalid rank.
        tf.errors.InvalidArgumentError: If features are empty or nonfinite.
    """

    return _normalize_matrix(_feature_matrix(features, "features"))


def _normalize_matrix(values: tf.Tensor) -> tf.Tensor:
    """Normalize an already validated feature matrix without repeating its guards.

    Args:
        values (tf.Tensor): Finite nonempty tf.float32 matrix [N, D] accepted by
            _feature_matrix; nonzero rows may have any finite magnitude.

    Returns:
        normalized (tf.Tensor): tf.float32 matrix [N, D] with unit nonzero rows
            and exact zero rows. Gradients remain connected to values.

    Raises:
        ValueError: If a statically known input rank cannot supply axis one.
        TypeError: If the input dtype is incompatible with the floating-point
            normalization operations.
        tf.errors.InvalidArgumentError: If a malformed input has an invalid
            dynamic reduction axis. This internal helper deliberately relies on
            _feature_matrix for rank, nonempty, dtype, and finiteness validation.
    """

    scale = tf.reduce_max(tf.abs(values), axis=1, keepdims=True)
    scaled = tf.math.divide_no_nan(values, scale)
    return tf.math.l2_normalize(scaled, axis=1, epsilon=1e-12)


def _paired_features(
    student_features: tf.Tensor, 
    target_features: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """Validate pair correspondence and detach target features.

    Args:
        student_features (tf.Tensor): Finite student feature matrix [N, D], convertible to
            tf.float32; row order must match the target.
        target_features (tf.Tensor): Finite paired target feature matrix [N, D], convertible
            to tf.float32; target gradients are stopped.

    Returns:
        paired (tuple[tf.Tensor, tf.Tensor]): (student, target): normalized tf.float32
            matrices of identical shape; target is detached.

    Raises:
        ValueError: If either feature rank is invalid.
        tf.errors.InvalidArgumentError: If shapes differ or values/dimensions are invalid.
    """

    student = _feature_matrix(student_features, "student_features")
    target = _feature_matrix(target_features, "target_features")
    with tf.control_dependencies([
        tf.debugging.assert_equal(
            tf.shape(student), tf.shape(target), 
            message="student and target must have identical batch/feature shapes"
        )
    ]):
        return _normalize_matrix(student), tf.stop_gradient(
            _normalize_matrix(target)
        )


def _weighted_mean(
    per_row_loss: tf.Tensor, 
    row_weights: tf.Tensor | None
) -> tf.Tensor:
    """Average bounded, detached weighted losses over all selected rows.

    Args:
        per_row_loss (tf.Tensor): Float32 vector [N] of losses whose weighted arithmetic
            mean is required.
        row_weights (tf.Tensor | None): Optional finite float vector [N] in [0, 1]; weights
            are detached and the denominator remains N.

    Returns:
        loss (tf.Tensor): Scalar tf.float32 arithmetic mean over N rows; optional weights
            change strength without sum-of-weights renormalization.

    Raises:
        ValueError: If weights have statically invalid rank.
        tf.errors.InvalidArgumentError: If weights are misaligned, nonfinite or outside [0,
            1].
    """

    # Without reliability weights, every selected row contributes equally.
    if row_weights is None:
        return tf.reduce_mean(per_row_loss)
    weights = tf.cast(tf.convert_to_tensor(row_weights), tf.float32)
    checks = [
        tf.debugging.assert_rank(weights, 1, message="row_weights must have rank 1"), 
        tf.debugging.assert_equal(
            tf.shape(weights), tf.shape(per_row_loss), 
            message="row_weights must provide exactly one weight per batch row"
        ), 
        tf.debugging.assert_all_finite(weights, "row_weights must be finite"), 
        tf.debugging.assert_greater_equal(
            weights, 0.0, message="row_weights must lie in [0, 1]"
        ), 
        tf.debugging.assert_less_equal(
            weights, 1.0, message="row_weights must lie in [0, 1]"
        )
    ]
    with tf.control_dependencies(checks):
        return tf.reduce_mean(per_row_loss * tf.stop_gradient(weights))


def modulation_separation_loss(
    modulated_features: tf.Tensor, 
    positive_mask: tf.Tensor, 
    orthogonality_weight: float = 1.0
) -> tf.Tensor:
    """Cluster selected-class positives and orthogonalize them to negatives.

    For normalized features ``z``, positive set ``P`` and negative set ``N``:

    ``mean_{i,j in P; i != j}(1 - z_i @ z_j)``
    ``+ orthogonality_weight * mean_{i in P,j in N}(z_i @ z_j)**2``.

    Each pair family is normalized by its own size. The positive term counts
    ordered pairs and excludes self-pairs. Squared cross-class cosine penalizes
    similarity without cancellation between positive and negative signs. A
    constant nonzero output has loss ``orthogonality_weight``, while perfectly
    aligned positives orthogonal to negatives have loss zero.

    Args:
        modulated_features (tf.Tensor): Finite feature matrix [N, D] after applying the same
            selected-class gate to every row, including negatives.
        positive_mask (tf.Tensor): tf.bool vector [N] with at least two positive entries and
            at least one negative entry.
        orthogonality_weight (float): Finite positive scalar multiplying the mean squared
            positive-versus-negative cosine.
            Defaults to ``1.0``.

    Returns:
        loss (tf.Tensor): Scalar tf.float32 attraction plus weighted squared-cosine
            separation; input-feature gradients remain available.

    Raises:
        TypeError: If positive_mask is not boolean.
        ValueError: If static feature/mask rank is invalid.
        tf.errors.InvalidArgumentError: If shapes, finite values, positive pairs, negatives
            or orthogonality coefficient are invalid.
    """

    features = normalized_features(modulated_features)
    mask = tf.convert_to_tensor(positive_mask)
    # positive_mask must be a boolean vector
    if mask.dtype != tf.bool:
        raise TypeError("positive_mask must be a boolean vector")
    weight = _scalar(orthogonality_weight, "orthogonality_weight")
    checks = [
        tf.debugging.assert_rank(mask, 1, message="positive_mask must have rank 1"), 
        tf.debugging.assert_equal(
            tf.shape(mask)[0], tf.shape(features)[0], 
            message="positive_mask must match the feature batch"
        ), 
        tf.debugging.assert_positive(
            weight, message="orthogonality_weight must be strictly positive"
        )
    ]
    with tf.control_dependencies(checks):
        positives = tf.boolean_mask(features, mask)
        negatives = tf.boolean_mask(features, tf.logical_not(mask))
    count = tf.shape(positives)[0]
    with tf.control_dependencies([
        tf.debugging.assert_greater_equal(
            count, 2, message="acquisition requires at least two positive examples"
        ), 
        tf.debugging.assert_positive(
            tf.shape(negatives)[0], 
            message="acquisition requires at least one negative example"
        )
    ]):
        positive_cosines = tf.matmul(positives, positives, transpose_b=True)
        off_diagonal = tf.logical_not(tf.eye(count, dtype=tf.bool))
        positive_distance = tf.boolean_mask(1.0 - positive_cosines, off_diagonal)
        # Unit-vector roundoff can otherwise give a tiny negative distance.
        attraction = tf.reduce_mean(tf.maximum(positive_distance, 0.0))
        negative_cosines = tf.matmul(positives, negatives, transpose_b=True)
        separation = tf.reduce_mean(tf.square(negative_cosines))
        return attraction + weight * separation


def orthogonal_projection_loss(
    positive_features: tf.Tensor, 
    negative_features: tf.Tensor, 
    negative_weight: float = 1.0, 
    square_loss: bool = False
) -> tf.Tensor:
    """Match the official TMCL selected-class OPL objective in float32.

    The default is ``1 - mean(offdiag(P @ P.T))`` plus ``negative_weight``
    times ``mean(abs(P @ N.T))`` for row-normalized positive/negative features.
    Absolute value precedes the mean; opposite cross-class cosines cannot
    cancel. There is no negative-negative term. The official square variant
    replaces the positive term by ``mean((1-cos)**2)`` and the negative term
    by ``mean(cos**2)``. No target gradients are stopped.

    Args:
        positive_features (tf.Tensor): Finite nonzero feature rows [P, D],
            with at least two positives and positive width D.
        negative_features (tf.Tensor): Finite nonzero feature rows [N, D],
            with at least one negative and the same feature width.
        negative_weight (float): Multiplier for the negative-pair term.
            Defaults to 1.0, matching the official acquisition configuration.
        square_loss (bool): Select the official squared-pair variant instead
            of the default linear-positive/absolute-negative loss.
            Defaults to False.

    Returns:
        loss (tf.Tensor): Scalar float32 pair-family means with gradients to
            both supplied feature matrices. Zero-norm cosine rows are rejected
            rather than assigned an arbitrary direction.

    Raises:
        ValueError: A static feature rank is invalid.
        tf.errors.InvalidArgumentError: Features are empty, nonfinite,
            zero-norm, misaligned or contain fewer than two positives.
    """

    positive = _feature_matrix(positive_features, "positive_features")
    negative = _feature_matrix(negative_features, "negative_features")
    checks = [
        tf.debugging.assert_greater_equal(
            tf.shape(positive)[0], 2, message="OPL requires two positive rows"
        ), 
        tf.debugging.assert_equal(
            tf.shape(positive)[1], tf.shape(negative)[1], 
            message="OPL positive and negative widths must match"
        ), 
        tf.debugging.assert_positive(
            tf.reduce_max(tf.abs(positive), axis=1), 
            message="OPL cosine similarity requires nonzero positive rows"
        ), 
        tf.debugging.assert_positive(
            tf.reduce_max(tf.abs(negative), axis=1), 
            message="OPL cosine similarity requires nonzero negative rows"
        )
    ]
    with tf.control_dependencies(checks):
        positive = _normalize_matrix(positive)
        negative = _normalize_matrix(negative)
    positive_cosines = tf.matmul(positive, positive, transpose_b=True)
    positive_pairs = tf.boolean_mask(
        positive_cosines, tf.logical_not(tf.eye(tf.shape(positive)[0], dtype=tf.bool))
    )
    cross_cosines = tf.matmul(positive, negative, transpose_b=True)
    # The optional official variant squares individual pair errors before averaging.
    if square_loss:
        positive_loss = tf.reduce_mean(tf.square(1. - positive_pairs))
        negative_loss = tf.reduce_mean(tf.square(cross_cosines))
    # Default OPL uses the mean absolute cross-cosine, not an absolute mean.
    else:
        positive_loss = 1. - tf.reduce_mean(positive_pairs)
        negative_loss = tf.reduce_mean(tf.abs(cross_cosines))
    return positive_loss + tf.cast(negative_weight, tf.float32) * negative_loss


def multiview_barlow_loss(
    views: list[tf.Tensor] | tuple[tf.Tensor, ...], 
    off_diagonal_weight: float = 0.005, 
    scale: float = 0.024, 
    epsilon: float = 1e-5
) -> tf.Tensor:
    """Match the official TMCL jointly normalized multiview Barlow loss.

    Stack K aligned [B, D] views and normalize each feature using population
    mean/variance over all K*B rows, without affine parameters or running
    statistics. Let Z_v be a normalized view and M the per-example mean over
    views. For every view, C_v = transpose(M) @ Z_v / B. Return the SUM over
    views of ``scale * (sum((diag(C_v)-1)**2) + off_diagonal_weight *
    sum(offdiag(C_v)**2))``. This is neither pairwise BT nor an average over
    views, and it is distinct from the official GhoshBarlowLoss alternative.

    The pinned official nonsynchronized implementation ignores its constructor
    normalize_eps/norm_strategy and uses functional BatchNorm's epsilon 1e-5.
    This implementation exposes epsilon explicitly; its default matches that
    effective path. Scale 0.024 matches the source default/launcher, whereas
    the paper lists 0.1 for ConViT. Callers must record which scale they choose.
    Single-device batches are supported; distributed cross-correlation and
    synchronized normalization are not implemented here.

    Args:
        views (list[tf.Tensor] | tuple[tf.Tensor, ...]): At least two aligned,
            finite nonempty [B, D] projected feature matrices. All share shape.
            Gradients flow through every supplied view and their joint moments;
            callers must stop target tensors before this call for MI or SI.
        off_diagonal_weight (float): Redundancy-reduction multiplier.
            Defaults to 0.005.
        scale (float): Scalar multiplying each view's full objective.
            Defaults to 0.024, matching the pinned official source.
        epsilon (float): Additive population-variance stabilizer, normally
            positive. Defaults to 1e-5. Zero is meaningful only when every
            jointly normalized feature has nonzero variance.

    Returns:
        loss (tf.Tensor): Finite scalar float32 sum over view correlations.
            Constant features give one diagonal penalty per feature per view.

    Raises:
        ValueError: The static view collection or a feature rank is invalid.
        tf.errors.InvalidArgumentError: Feature shapes differ, data are
            nonfinite, or the requested normalization produces nonfinite values.
    """

    # A static collection gives an explicit view count and avoids treating rows as views.
    if not isinstance(views, (list, tuple)) or len(views) < 2:
        raise ValueError("Multiview Barlow requires a list or tuple of at least two views.")
    matrices = [_feature_matrix(view, f"view_{index}") for index, view in enumerate(views)]
    checks = [
        tf.debugging.assert_equal(
            tf.shape(matrix), tf.shape(matrices[0]), 
            message="Multiview Barlow views must have aligned batch and feature shapes"
        )
        for matrix in matrices[1:]
    ]
    with tf.control_dependencies(checks):
        stacked = tf.stack(matrices, axis=0)
    mean, variance = tf.nn.moments(stacked, axes=[0, 1], keepdims=True)
    normalized = (stacked - mean) * tf.math.rsqrt(variance + tf.cast(epsilon, tf.float32))
    with tf.control_dependencies([
        tf.debugging.assert_all_finite(variance, "Multiview Barlow variance overflowed"), 
        tf.debugging.assert_all_finite(normalized, "Multiview Barlow normalization is nonfinite")
    ]):
        expected = tf.reduce_mean(normalized, axis=0)
    batch_size = tf.cast(tf.shape(stacked)[1], tf.float32)
    losses = []
    for index in range(len(matrices)):
        correlation = tf.matmul(expected, normalized[index], transpose_a=True) / batch_size
        diagonal = tf.linalg.diag_part(correlation)
        off_diagonal = tf.linalg.set_diag(correlation, tf.zeros_like(diagonal))
        loss = tf.reduce_sum(tf.square(diagonal - 1.)) + (
            tf.cast(off_diagonal_weight, tf.float32) * tf.reduce_sum(tf.square(off_diagonal))
        )
        losses.append(tf.cast(scale, tf.float32) * loss)
    result = tf.add_n(losses)
    with tf.control_dependencies([
        tf.debugging.assert_all_finite(result, "Multiview Barlow loss is nonfinite")
    ]):
        return tf.identity(result)


def _aligned_views(views: list[tf.Tensor] | tuple[tf.Tensor, ...]) -> list[tf.Tensor]:
    """Validate aligned views for branch-normalized Barlow objectives."""

    # Branch BatchNorm needs multiple examples and an explicit collection of views.
    if not isinstance(views, (list, tuple)) or len(views) < 2:
        raise ValueError("Barlow requires at least two explicitly paired views.")
    matrices = [_feature_matrix(value, f"view_{index}") for index, value in enumerate(views)]
    checks = [tf.debugging.assert_greater_equal(
        tf.shape(matrices[0])[0], 2, message="Branch normalization needs at least two examples"
    )]
    checks.extend(tf.debugging.assert_equal(
        tf.shape(value), tf.shape(matrices[0]), message="Barlow views must have identical shapes"
    ) for value in matrices[1:])
    with tf.control_dependencies(checks):
        return [tf.identity(value) for value in matrices]


def _standardize_branch(matrix: tf.Tensor, epsilon: float) -> tf.Tensor:
    """Apply source-equivalent population BatchNorm to one already checked branch."""

    mean, variance = tf.nn.moments(matrix, axes=[0], keepdims=True)
    normalized = (matrix - mean) * tf.math.rsqrt(variance + tf.cast(epsilon, tf.float32))
    with tf.control_dependencies([
        tf.debugging.assert_all_finite(normalized, "Barlow branch normalization is nonfinite")
    ]):
        return tf.identity(normalized)


def barlow_twins_loss(
    first_features: tf.Tensor, second_features: tf.Tensor, 
    off_diagonal_weight: float = 0.005, scale: float = 0.024, 
    normalization: str = "per_branch", epsilon: float = 1e-5
) -> tf.Tensor:
    """Compute the pinned two-branch Barlow loss without implicit target stops.

    ``normalization='per_branch'`` matches the official TMCL SI caller and
    optional two-view VI/MI caller. ``'joint'`` matches the source class default.
    Each normalized branch has shape [B,D]; C = first.T @ second / B.
    Return scale times the summed squared diagonal error plus weighted summed
    squared off-diagonal correlations. Both inputs remain differentiable.

    The effective nonsynchronized source epsilon is 1e-5. This helper exposes
    it explicitly for independent mathematical checks. Distributed reduction
    is not implied. Source: pinned tmcl/nn/losses/barlow.py.
    """

    first, second = _aligned_views([first_features, second_features])
    # SI and the released two-view VI option use independent branch statistics.
    if normalization == "per_branch":
        first = _standardize_branch(first, epsilon)
        second = _standardize_branch(second, epsilon)
    # Joint statistics are a distinct explicitly selected source option.
    elif normalization == "joint":
        joined = _standardize_branch(tf.concat([first, second], axis=0), epsilon)
        first, second = tf.split(joined, 2, axis=0)
    # A misspelled normalization policy must not silently change the loss.
    else:
        raise ValueError("normalization must be 'per_branch' or 'joint'.")
    correlation = tf.matmul(first, second, transpose_a=True) / tf.cast(tf.shape(first)[0], tf.float32)
    diagonal = tf.linalg.diag_part(correlation)
    off_diagonal = tf.linalg.set_diag(correlation, tf.zeros_like(diagonal))
    result = tf.cast(scale, tf.float32) * (
        tf.reduce_sum(tf.square(diagonal - 1.))
        + tf.cast(off_diagonal_weight, tf.float32) * tf.reduce_sum(tf.square(off_diagonal))
    )
    with tf.control_dependencies([tf.debugging.assert_all_finite(result, "Barlow loss is nonfinite")]):
        return tf.identity(result)


def ghosh_barlow_loss(
    views: list[tf.Tensor] | tuple[tf.Tensor, ...], 
    off_diagonal_weight: float = 0.005, scale: float = 0.024, epsilon: float = 1e-5
) -> tf.Tensor:
    """Match the distinct GhoshBarlowLoss alternative in the pinned source.

    Normalize each branch independently. With M the mean normalized view,
    average the summed squared diagonal errors of E[Z_v*M] over views.
    Redundancy uses the off-diagonal squared entries of M.T @ M / B once.
    This differs from the joint-normalized, view-summed multiview objective.
    No input is implicitly stopped. Source: tmcl/nn/losses/mv_barlow.py.
    """

    normalized = [_standardize_branch(value, epsilon) for value in _aligned_views(views)]
    stacked = tf.stack(normalized, axis=0)
    expected = tf.reduce_mean(stacked, axis=0)
    diagonal = tf.reduce_mean(stacked * expected[None, :, :], axis=1)
    invariance = tf.reduce_sum(tf.square(diagonal - 1.)) / tf.cast(len(normalized), tf.float32)
    covariance = tf.matmul(expected, expected, transpose_a=True) / tf.cast(tf.shape(expected)[0], tf.float32)
    off_diagonal = tf.linalg.set_diag(covariance, tf.zeros_like(tf.linalg.diag_part(covariance)))
    result = tf.cast(scale, tf.float32) * (
        invariance + tf.cast(off_diagonal_weight, tf.float32) * tf.reduce_sum(tf.square(off_diagonal))
    )
    with tf.control_dependencies([tf.debugging.assert_all_finite(result, "Ghosh Barlow loss is nonfinite")]):
        return tf.identity(result)


def state_invariance_loss(
    current_prediction_views: list[tf.Tensor] | tuple[tf.Tensor, ...], 
    frozen_views: list[tf.Tensor] | tuple[tf.Tensor, ...], 
    method: str = "cassle", objective: str = "barlow", 
    off_diagonal_weight: float = 0.0051, scale: float = 0.1, 
    negative_weight: float = 1.0, epsilon: float = 1e-5
) -> tf.Tensor:
    """Compute CaSSLe-style SI or corrected PNR on aligned past/current views.

    Targets are always stopped here. The owning phase must additionally freeze
    the past network and projector, including nontrainable normalization state.
    As in pinned main_tmcl.py, ``barlow`` averages same-view independently
    normalized two-branch Barlow losses for views 0 and 1 only. ``ghosh``
    averages the same-view two-branch Ghosh loss over all supplied views.

    PNR adds negative_weight times mean cross-view cosine: 0->1 and 1->0
    for Barlow; additionally 2->3 and 3->2 for Ghosh (exactly four views).
    This is repulsion when minimizing, because source simsiam_loss_func is
    negative cosine. Cosine uses norm floor 1e-8, matching that source call.
    The pinned Barlow PNR branch computes/logs but accidentally omits adding
    this term; this implementation deliberately corrects that omission.
    It is not bitwise compatibility with that bug. The correction follows
    PNR's explicit pseudo-negative regularization, Cha et al., ICML 2024,
    https://proceedings.mlr.press/v235/cha24a.html.

    Args are finite paired [B,D] view collections with B>=2. Modes and view
    counts are structural identities; unsupported choices fail explicitly.
    Source configuration defaults are lambda .0051 and scale .1 for SI,
    independently of the VI/MI loss coefficients.
    """

    # Method identity controls whether pseudo-negative repulsion is present.
    if method not in ("cassle", "pnr") or objective not in ("barlow", "ghosh"):
        raise ValueError("SI supports cassle/pnr with barlow/ghosh objectives.")
    # Pairing cannot silently truncate a missing past or current augmentation.
    if (not isinstance(current_prediction_views, (list, tuple))
            or not isinstance(frozen_views, (list, tuple))
            or len(current_prediction_views) != len(frozen_views)):
        raise ValueError("SI requires equal explicit current and past view collections.")
    count = len(current_prediction_views)
    matrices = _aligned_views(list(current_prediction_views) + list(frozen_views))
    # At least two image augmentations define the source's crossed SI construction.
    if count < 2:
        raise ValueError("SI requires at least two paired views.")
    # The released Ghosh PNR pairing is defined for these four views only.
    if method == "pnr" and objective == "ghosh" and count != 4:
        raise ValueError("Ghosh PNR requires exactly four paired views.")
    predictions = matrices[:count]
    frozen = [tf.stop_gradient(value) for value in matrices[count:]]
    # Released Barlow SI deliberately uses only the first two augmentations.
    if objective == "barlow":
        losses = [barlow_twins_loss(
            predictions[index], frozen[index], off_diagonal_weight=off_diagonal_weight, 
            scale=scale, normalization="per_branch", epsilon=epsilon
        ) for index in range(2)]
    # Ghosh SI includes every matched view pair in its mean.
    else:
        losses = [ghosh_barlow_loss(
            [prediction, target], off_diagonal_weight=off_diagonal_weight, scale=scale, epsilon=epsilon
        ) for prediction, target in zip(predictions, frozen)]
    result = tf.add_n(losses) / tf.cast(len(losses), tf.float32)
    # Corrected PNR adds the cross-view pseudo-negative term for either objective.
    if method == "pnr":
        pairs = [(0, 1), (1, 0)] if objective == "barlow" else [(0, 1), (1, 0), (2, 3), (3, 2)]
        repulsion = []
        for current, past in pairs:
            left = tf.math.l2_normalize(predictions[current], axis=1, epsilon=1e-16)
            right = tf.math.l2_normalize(frozen[past], axis=1, epsilon=1e-16)
            repulsion.append(tf.reduce_mean(tf.reduce_sum(left * right, axis=1)))
        result += tf.cast(negative_weight, tf.float32) * tf.add_n(repulsion) / tf.cast(len(pairs), tf.float32)
    with tf.control_dependencies([tf.debugging.assert_all_finite(result, "SI loss is nonfinite")]):
        return tf.identity(result)


def contrastive_alignment_loss(
    student_features: tf.Tensor, 
    target_features: tf.Tensor, 
    temperature: float = 0.1, 
    row_weights: tf.Tensor | None = None
) -> tf.Tensor:
    """Align matched inputs using normalized, asymmetric InfoNCE.

    Row ``i`` of the student must correspond to row ``i`` of the frozen target.
    All target rows ``j != i`` are negatives, including same-class examples:
    this is instance discrimination, not supervised class contrast. The exact
    reduction is ``mean_i w_i [logsumexp_j(s_ij / T) - s_ii / T]`` where
    ``s_ij`` is cosine similarity. No sum-of-weights renormalization is used;
    reliability therefore changes the term's strength. Callers average over
    noise draws separately. Fully collapsed nonzero features give ``log(B)``
    without weights, so low loss alone should not replace collapse diagnostics.

    Args:
        student_features (tf.Tensor): Finite student feature matrix [N, D], convertible to
            tf.float32; row order must match the target.
        target_features (tf.Tensor): Finite paired target feature matrix [N, D], convertible
            to tf.float32; target gradients are stopped.
        temperature (float): Finite positive scalar controlling softmax sharpness; smaller
            values sharpen the distribution.
            Defaults to ``0.1``.
        row_weights (tf.Tensor | None): Optional finite float vector [N] in [0, 1]; weights
            are detached and the denominator remains N.
            Defaults to ``None``.

    Returns:
        loss (tf.Tensor): Scalar tf.float32 asymmetric instance InfoNCE, averaged over rows
            with optional detached reliability weights.

    Raises:
        ValueError: If a static tensor rank is invalid.
        tf.errors.InvalidArgumentError: If paired features/weights are invalid, fewer than
            two rows are supplied or temperature produces nonfinite logits.
    """

    student, target = _paired_features(student_features, target_features)
    temperature = _scalar(temperature, "temperature")
    checks = [
        tf.debugging.assert_greater_equal(
            tf.shape(student)[0], 2, 
            message="contrastive alignment requires at least two examples"
        ), 
        tf.debugging.assert_positive(
            temperature, message="temperature must be strictly positive"
        )
    ]
    with tf.control_dependencies(checks):
        similarities = tf.matmul(student, target, transpose_b=True)
        # Subtract the matched similarity before temperature scaling. This is
        # algebraically logsumexp(logits) - logits_ii, without subtracting two
        # large, nearly equal float32 values at small temperatures.
        centered_logits = (
            similarities - tf.linalg.diag_part(similarities)[:, None]
        ) / temperature
        with tf.control_dependencies([
            tf.debugging.assert_all_finite(
                centered_logits, 
                "temperature is too small for finite float32 contrastive logits"
            )
        ]):
            per_row_loss = tf.reduce_logsumexp(centered_logits, axis=1)
            return _weighted_mean(per_row_loss, row_weights)


def normalized_feature_distillation_loss(
    student_features: tf.Tensor, 
    target_features: tf.Tensor, 
    row_weights: tf.Tensor | None = None
) -> tf.Tensor:
    """Return weighted mean squared error between normalized paired features.

    This control uses ``mean_i w_i mean_d (student_id - target_id)**2``.
    As for InfoNCE, weights are detached and the denominator is the full batch
    size, not the sum of weights. Target gradients are stopped. Unlike InfoNCE,
    a batch of one is valid because this objective uses no negatives.

    Args:
        student_features (tf.Tensor): Finite student feature matrix [N, D], convertible to
            tf.float32; row order must match the target.
        target_features (tf.Tensor): Finite paired target feature matrix [N, D], convertible
            to tf.float32; target gradients are stopped.
        row_weights (tf.Tensor | None): Optional finite float vector [N] in [0, 1]; weights
            are detached and the denominator remains N.
            Defaults to ``None``.

    Returns:
        loss (tf.Tensor): Scalar tf.float32 mean over rows and dimensions of normalized
            feature squared differences; a one-row batch is valid.

    Raises:
        ValueError: If a static tensor rank is invalid.
        tf.errors.InvalidArgumentError: If paired features or reliability weights violate
            shape, finiteness or range requirements.
    """

    student, target = _paired_features(student_features, target_features)
    return _weighted_mean(tf.reduce_mean(tf.square(student - target), axis=1), row_weights)


def reliability_weights(
    alpha_bar: tf.Tensor, 
    floor: float = 0.05
) -> tf.Tensor:
    """Return detached semantic reliability ``clip(alpha_bar, floor, 1)``.

    ``alpha_bar`` is the cumulative diffusion signal power from the existing
    schedule, not a timestep or signal-to-noise ratio. The floor lies in [0, 1].
    Scalar or batched inputs are allowed; the loss APIs expect a vector when
    these values are used as row weights. This rule is a bounded experimental
    heuristic, not an estimated probability of semantic correctness.

    Args:
        alpha_bar (tf.Tensor): Finite cumulative diffusion signal-power scalar or tensor,
            cast to tf.float32 before clipping.
        floor (float): Finite scalar lower clipping bound in [0, 1].
            Defaults to ``0.05``.

    Returns:
        weights (tf.Tensor): Detached tf.float32 tensor with the input alpha_bar shape,
            clipped to [floor, 1].

    Raises:
        ValueError: If floor has a statically nonscalar shape.
        tf.errors.InvalidArgumentError: If alpha_bar/floor is nonfinite or floor lies
            outside [0, 1].
    """

    values = tf.cast(tf.convert_to_tensor(alpha_bar), tf.float32)
    floor = _scalar(floor, "floor")
    checks = [
        tf.debugging.assert_all_finite(values, "alpha_bar must be finite"), 
        tf.debugging.assert_greater_equal(floor, 0.0, message="floor must lie in [0, 1]"), 
        tf.debugging.assert_less_equal(floor, 1.0, message="floor must lie in [0, 1]")
    ]
    with tf.control_dependencies(checks):
        return tf.stop_gradient(tf.clip_by_value(values, floor, 1.0))
