"""Post-generation measurements for image replay candidate pools.

These NumPy-only measurements describe contrast and sample variation; they do
not establish image quality or downstream replay usefulness.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np


def _merge_chunk_moments(
    previous: tuple[int, np.ndarray, np.ndarray] | None,
    chunk: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Merge a nonempty float64 chunk into per-coordinate population moments.

    Args:
        previous: Existing sample count, mean, and sum of squared deviations,
            or ``None`` before the first chunk. Existing arrays are updated.
        chunk: Two-dimensional rows of flattened images in float64.

    Returns:
        Updated sample count, coordinate means, and squared-deviation sums.
    """
    batch_count = len(chunk)
    batch_mean = np.mean(chunk, axis=0)
    centered = chunk - batch_mean
    batch_m2 = np.einsum("ij,ij->j", centered, centered)
    if previous is None:
        return batch_count, batch_mean, batch_m2

    count, mean, m2 = previous
    total = count + batch_count
    delta = batch_mean - mean
    m2 += batch_m2 + np.square(delta) * (count * batch_count / total)
    mean += delta * (batch_count / total)
    return total, mean, m2


def generated_sample_variation(
    samples: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int = 128,
    value_range: float = 1.0,
) -> dict[str, int | float | None]:
    """Measure contrast and diversity over a complete generated candidate pool.

    Population standard deviations use ``ddof=0``. Per-coordinate moments are
    merged across chunks, so partial batches and batch boundaries do not change
    the weighting. Only the current chunk is converted to float64; accumulated
    moments occupy one image-sized vector pair globally and per class. Samples
    are neither clipped nor rescaled in place.

    Args:
        samples: Numeric image array with samples along the first axis. Remaining
            axes, including channels, are treated as image coordinates.
        labels: One class label per sample; shape ``(N,)`` or ``(N, 1)``.
        batch_size: Positive number of samples per statistics chunk.
        value_range: Positive finite reference intensity range dividing all
            standard deviations, such as 2 for diffusion values in ``[-1, 1]``.
            Use 1 to report native units. Values outside the range are retained.

    Returns:
        Sample and class counts, ``mean_image_std`` (mean within-image standard
        deviation), ``mean_pixel_std`` (across-image standard deviation at each
        coordinate, averaged over coordinates), ``within_class_pixel_std``
        (the same calculation within each class, averaged equally over classes
        with at least two samples), and ``eligible_class_count``. All three
        measurements are ``None`` for an empty pool; within-class variation is
        also ``None`` when no class contains at least two samples.

    Raises:
        ValueError: If dimensions, labels, batch size, reference range, or sample
            values are invalid. Nonfinite and complex samples are rejected.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    value_range = float(value_range)
    if not np.isfinite(value_range) or value_range <= 0:
        raise ValueError("value_range must be finite and positive")

    samples = np.asarray(samples)
    labels = np.asarray(labels)
    if samples.ndim == 0 or samples.dtype.kind not in "biuf":
        raise ValueError("samples must be a real numeric array with a sample axis")
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1 or len(labels) != len(samples):
        raise ValueError("labels must contain one class label per sample")
    coordinate_count = int(np.prod(samples.shape[1:], dtype=np.int64))
    if coordinate_count == 0:
        raise ValueError("each sample must contain at least one image coordinate")

    sample_count = len(samples)
    classes, class_indices = np.unique(labels, return_inverse=True)
    class_count = len(classes)
    result: dict[str, int | float | None] = {
        "sample_count": sample_count,
        "class_count": class_count,
        "mean_image_std": None,
        "mean_pixel_std": None,
        "within_class_pixel_std": None,
        "eligible_class_count": 0,
    }
    if not sample_count:
        return result

    global_moments = None
    class_moments: list[tuple[int, np.ndarray, np.ndarray] | None] = [None] * class_count
    image_std_sum = 0.0
    for start in range(0, sample_count, int(batch_size)):
        stop = min(start + int(batch_size), sample_count)
        chunk = np.asarray(samples[start:stop], dtype=np.float64).reshape(
            stop - start, coordinate_count
        )
        if not np.all(np.isfinite(chunk)):
            raise ValueError("samples must contain only finite values")
        image_std_sum += float(np.sum(np.std(chunk, axis=1, ddof=0)))
        global_moments = _merge_chunk_moments(global_moments, chunk)
        chunk_classes = class_indices[start:stop]
        for class_index in np.unique(chunk_classes):
            class_moments[class_index] = _merge_chunk_moments(
                class_moments[class_index], chunk[chunk_classes == class_index]
            )

    assert global_moments is not None
    count, _, m2 = global_moments
    result["mean_image_std"] = image_std_sum / sample_count / value_range
    result["mean_pixel_std"] = float(np.mean(np.sqrt(m2 / count))) / value_range
    within_class_values = [
        float(np.mean(np.sqrt(moments[2] / moments[0]))) / value_range
        for moments in class_moments
        if moments is not None and moments[0] >= 2
    ]
    result["eligible_class_count"] = len(within_class_values)
    if within_class_values:
        result["within_class_pixel_std"] = float(np.mean(within_class_values))
    return result
