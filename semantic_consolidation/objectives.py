"""Numerical objectives for semantic modulation acquisition and consolidation.

The acquisition objective is an OPL-inspired, selected-class separation loss;
it is not a claimed reproduction of TMCL. Consolidation uses asymmetric,
instance-matched InfoNCE with every other target row as a negative. Callers
average losses over their configured noise draws and freeze the target network
itself: stopping target gradients here does not freeze a shared, changing model.
All reductions and similarity calculations use float32.
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
        ),
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
        tf.debugging.assert_all_finite(value, f"{name} must be finite"),
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

    values = _feature_matrix(features, "features")
    scale = tf.reduce_max(tf.abs(values), axis=1, keepdims=True)
    scaled = tf.math.divide_no_nan(values, scale)
    return tf.math.l2_normalize(scaled, axis=1, epsilon=1e-12)


def _paired_features(
    student_features: tf.Tensor,
    target_features: tf.Tensor,
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
            message="student and target must have identical batch/feature shapes",
        )
    ]):
        return normalized_features(student), tf.stop_gradient(
            normalized_features(target)
        )


def _weighted_mean(
    per_row_loss: tf.Tensor,
    row_weights: tf.Tensor | None,
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
            message="row_weights must provide exactly one weight per batch row",
        ),
        tf.debugging.assert_all_finite(weights, "row_weights must be finite"),
        tf.debugging.assert_greater_equal(
            weights, 0.0, message="row_weights must lie in [0, 1]"
        ),
        tf.debugging.assert_less_equal(
            weights, 1.0, message="row_weights must lie in [0, 1]"
        ),
    ]
    with tf.control_dependencies(checks):
        return tf.reduce_mean(per_row_loss * tf.stop_gradient(weights))


def modulation_separation_loss(
    modulated_features: tf.Tensor,
    positive_mask: tf.Tensor,
    orthogonality_weight: float = 1.0,
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
            message="positive_mask must match the feature batch",
        ),
        tf.debugging.assert_positive(
            weight, message="orthogonality_weight must be strictly positive"
        ),
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
            message="acquisition requires at least one negative example",
        ),
    ]):
        positive_cosines = tf.matmul(positives, positives, transpose_b=True)
        off_diagonal = tf.logical_not(tf.eye(count, dtype=tf.bool))
        positive_distance = tf.boolean_mask(1.0 - positive_cosines, off_diagonal)
        # Unit-vector roundoff can otherwise give a tiny negative distance.
        attraction = tf.reduce_mean(tf.maximum(positive_distance, 0.0))
        negative_cosines = tf.matmul(positives, negatives, transpose_b=True)
        separation = tf.reduce_mean(tf.square(negative_cosines))
        return attraction + weight * separation


def contrastive_alignment_loss(
    student_features: tf.Tensor,
    target_features: tf.Tensor,
    temperature: float = 0.1,
    row_weights: tf.Tensor | None = None,
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
        row_weights (tf.Tensor | None): Optional finite float vector [N] in [0, 1]; weights
            are detached and the denominator remains N.

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
            message="contrastive alignment requires at least two examples",
        ),
        tf.debugging.assert_positive(
            temperature, message="temperature must be strictly positive"
        ),
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
                "temperature is too small for finite float32 contrastive logits",
            )
        ]):
            per_row_loss = tf.reduce_logsumexp(centered_logits, axis=1)
            return _weighted_mean(per_row_loss, row_weights)


def normalized_feature_distillation_loss(
    student_features: tf.Tensor,
    target_features: tf.Tensor,
    row_weights: tf.Tensor | None = None,
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
    floor: float = 0.05,
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
        tf.debugging.assert_less_equal(floor, 1.0, message="floor must lie in [0, 1]"),
    ]
    with tf.control_dependencies(checks):
        return tf.stop_gradient(tf.clip_by_value(values, floor, 1.0))
