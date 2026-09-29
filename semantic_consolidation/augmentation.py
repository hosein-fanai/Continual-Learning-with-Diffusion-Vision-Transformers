"""Stateless TensorFlow image augmentation from TMCL, Appendix A.

Reference: https://arxiv.org/html/2505.14125v3#A1. Acquisition uses only
horizontal flips. Consolidation uses four independently sampled RGB views by
default; solarization is eligible on views 2, 4, ... (paper's one-based view
numbering). Operations run in pixel space before any diffusion corruption;
public inputs and outputs use this project's float32 NHWC model scale [-1, 1].

Kornia's ``RandomSolarize(thresholds=0.0, additions=0.0)`` means a fixed
*pixel threshold of 0.5*: the first zero is the sampling half-width around
0.5, not a pixel threshold of zero. ColorJitter uses multiplicative brightness,
luminance-mean contrast, grayscale-blend saturation and HSV hue shifts. Its
random order is shared across the batch, while factors and application masks
are independent per image, matching Kornia's ColorJitter semantics.

Source audit (the original tran-khoa/tmcl repository redirects here):
https://github.com/Dendritic-Learning-Group/tmcl/blob/2cb306cc5cbe6d13a3dd4991998ee3e7c2948c8b/tmcl/datasets/kornia_ssl.py
https://github.com/Dendritic-Learning-Group/tmcl/blob/2cb306cc5cbe6d13a3dd4991998ee3e7c2948c8b/tmcl/main_tmcl.py
https://github.com/kornia/kornia/blob/v0.8.0/kornia/augmentation/_2d/intensity/color_jitter.py
https://github.com/kornia/kornia/blob/v0.8.0/kornia/augmentation/_2d/intensity/solarize.py
https://github.com/kornia/kornia/blob/v0.8.0/kornia/enhance/adjust.py
The paper
governs two source discrepancies: its acquisition has no padded crop, and its
solarization alternates even views rather than grouping the last half of views.

This is a native TensorFlow policy implementation, not bitwise Kornia replay.
Random-resized crop uses ten rejection attempts and the standard central
aspect-ratio fallback, not Kornia v0.8.0's crop-generator edge-case quirks.
Bicubic resize aligns corners as Kornia does. RNG and interpolation/HSV
rounding differ between backends. Bicubic overshoots are retained unless a
subsequent color operation clips them; the output range is therefore nominal.
No dataset-specific normalization is added to the project's existing scaling.
"""

from __future__ import annotations

from numbers import Integral

import tensorflow as tf


__all__ = ["acquisition_augmentation", "consolidation_views"]

_CROP_SCALE = (0.08, 1.0)
_CROP_RATIO = (3.0 / 4.0, 4.0 / 3.0)
_COLOR_JITTER_PROBABILITY = 0.8
_GRAYSCALE_PROBABILITY = 0.2
_FLIP_PROBABILITY = 0.5
_SOLARIZE_PROBABILITY = 0.2


def _seed_pair(seed: int | tf.Tensor) -> tf.Tensor:
    """Accept an explicit integer scalar or two-element stateless seed.

    Args:
        seed (int | tf.Tensor): Integer scalar or statically shaped integer vector [2].

    Returns:
        tf.Tensor: Int64 seed [2]; scalar seeds receive zero as the second word.

    Raises:
        ValueError: If the dtype is not integer or the static shape is not scalar/[2]."""

    seed = tf.convert_to_tensor(seed)
    # Floating seeds would silently discard reproducibility information.
    if not seed.dtype.is_integer:
        raise ValueError("augmentation seed must contain integers")
    # A scalar names a two-word stateless stream with zero second word.
    if seed.shape.rank == 0:
        seed = tf.stack((tf.cast(seed, tf.int64), tf.constant(0, tf.int64)))
    # Reject all vector shapes except the explicit stateless seed pair.
    elif seed.shape.rank != 1 or seed.shape[0] != 2:
        raise ValueError("augmentation seed must be an integer or shape [2]")
    return tf.cast(seed, tf.int64)


def _fold(seed: tf.Tensor, stream: int | tf.Tensor) -> tf.Tensor:
    """Derive a deterministic child stream without reading global RNG state.

    Args:
        seed (tf.Tensor): Integer stateless seed vector [2].
        stream (int | tf.Tensor): Scalar integer child-stream identifier.

    Returns:
        tf.Tensor: Integer stateless seed vector [2] from TensorFlow fold_in."""

    return tf.random.experimental.stateless_fold_in(seed, tf.cast(stream, tf.int64))


def _uniform(seed: tf.Tensor, stream: int, shape: object = ()) -> tf.Tensor:
    """Draw independent float32 uniforms from a named local stream.

    Args:
        seed (tf.Tensor): Integer stateless seed vector [2].
        stream (int): Child-stream identifier folded into seed.
        shape (object): TensorFlow-compatible integer output shape; () returns a scalar.

    Returns:
        tf.Tensor: Float32 values in [0,1), with the requested shape."""

    return tf.random.stateless_uniform(shape, _fold(seed, stream), dtype=tf.float32)


def _images(images: tf.Tensor) -> tf.Tensor:
    """Check model-scale RGB images before converting them to pixel space.

    Args:
        images (tf.Tensor): Numeric NHWC tensor [N,H,W,3] in [-1,1], with positive dimensions.

    Returns:
        tf.Tensor: Float32 NHWC images with identical geometry and values after casting.

    Raises:
        ValueError: If a statically known rank cannot represent NHWC images.
        tf.errors.InvalidArgumentError: If dimensions, channels, finiteness or model scale are invalid.
            Scale checks prevent silently applying pixel transforms to already rescaled/noised data."""

    images = tf.cast(tf.convert_to_tensor(images), tf.float32)
    tf.debugging.assert_rank(images, 4, message="augmentation requires NHWC images")
    tf.debugging.assert_equal(tf.shape(images)[-1], 3, message="TMCL requires RGB images")
    tf.debugging.assert_positive(tf.shape(images)[:3], message="image dimensions must be positive")
    tf.debugging.assert_all_finite(images, "augmentation images must be finite")
    tf.debugging.assert_greater_equal(images, -1., message="expected model inputs in [-1, 1]")
    tf.debugging.assert_less_equal(images, 1., message="expected model inputs in [-1, 1]")
    return images


def _flip(images: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    """Apply independent Bernoulli(0.5) horizontal flips to each image.

    Args:
        images (tf.Tensor): Float32 NHWC tensor [N,H,W,C], in either model or pixel scale.
        seed (tf.Tensor): Integer stateless seed vector [2].

    Returns:
        tf.Tensor: Same dtype/shape/pixels; selected rows reverse only the width axis."""

    mask = _uniform(seed, 0, tuple([tf.shape(images)[0]])) < _FLIP_PROBABILITY
    return tf.where(mask[:, None, None, None], tf.reverse(images, axis=[2]), images)


def _crop_box(shape: tf.Tensor, seed: tf.Tensor) -> tuple[tf.Tensor, ...]:
    """Sample crop area uniformly and aspect ratio log-uniformly with ten attempts.

    Args:
        shape (tf.Tensor): Int32 image shape with positive height/width in its first two entries.
        seed (tf.Tensor): Integer stateless seed vector [2], unique to the image/view.

    Returns:
        tuple[tf.Tensor, ...]: Scalar int32 (top, left, height, width). The first valid
            proposal uses uniform integer origins; if all ten fail, return a centered
            crop constrained to the configured aspect-ratio range and image bounds."""

    height, width = shape[0], shape[1]
    height_f, width_f = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
    area = height_f * width_f * (
        _CROP_SCALE[0] + (_CROP_SCALE[1] - _CROP_SCALE[0]) * _uniform(seed, 0, tuple([10]))
    )
    log_min = tf.math.log(tf.constant(_CROP_RATIO[0], tf.float32))
    log_max = tf.math.log(tf.constant(_CROP_RATIO[1], tf.float32))
    ratio = tf.exp(log_min + (log_max - log_min) * _uniform(seed, 1, tuple([10])))
    widths = tf.cast(tf.round(tf.sqrt(area * ratio)), tf.int32)
    heights = tf.cast(tf.round(tf.sqrt(area / ratio)), tf.int32)
    valid = (widths > 0) & (widths <= width) & (heights > 0) & (heights <= height)

    def sampled() -> tuple[tf.Tensor, ...]:
        """Use the first valid proposal and a uniformly sampled crop origin."""

        index = tf.argmax(tf.cast(valid, tf.int32), output_type=tf.int32)
        crop_height, crop_width = heights[index], widths[index]
        top = tf.cast(_uniform(seed, 2) * tf.cast(height - crop_height + 1, tf.float32), tf.int32)
        left = tf.cast(_uniform(seed, 3) * tf.cast(width - crop_width + 1, tf.float32), tf.int32)
        return top, left, crop_height, crop_width

    def central() -> tuple[tf.Tensor, ...]:
        """Fall back to a centered crop constrained by the aspect-ratio bounds."""

        input_ratio = width_f / height_f
        crop_width = tf.where(
            input_ratio > _CROP_RATIO[1], 
            tf.cast(tf.round(height_f * _CROP_RATIO[1]), tf.int32), width
        )
        crop_height = tf.where(
            input_ratio < _CROP_RATIO[0], 
            tf.cast(tf.round(width_f / _CROP_RATIO[0]), tf.int32), height
        )
        crop_width = tf.clip_by_value(crop_width, 1, width)
        crop_height = tf.clip_by_value(crop_height, 1, height)
        return (height - crop_height) // 2, (width - crop_width) // 2, crop_height, crop_width

    return tf.cond(tf.reduce_any(valid), sampled, central)


def _random_resized_crop(images: tf.Tensor, seed: tf.Tensor, size: tuple[int, int]) -> tf.Tensor:
    """Crop images independently and resize using corner-aligned bicubic interpolation.

    Args:
        images (tf.Tensor): Float32 RGB pixel-space tensor [N,H,W,3].
        seed (tf.Tensor): Integer stateless seed [2]; row indices select independent crop streams.
        size (tuple[int, int]): Positive output height/width, fixed for map_fn's output signature.

    Returns:
        tf.Tensor: Float32 [N,size[0],size[1],3]. Bicubic overshoot is retained."""

    def crop(row: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Resize one sampled crop using the image-index-specific seed.

        Args:
            row (tuple[tf.Tensor, tf.Tensor]): A scalar int32 row index and its
                float32 RGB image [H,W,3]. The index is folded into the enclosing
                stateless seed to select this image's independent crop.

        Returns:
            tf.Tensor: Float32 RGB image [size[0],size[1],3], resized with
                corner-aligned bicubic interpolation; overshoot is retained.
        """

        index, image = row
        top, left, height, width = _crop_box(tf.shape(image), _fold(seed, index))
        image = tf.slice(image, (top, left, 0), (height, width, 3))
        return tf.raw_ops.ResizeBicubic(
            images=image[None], size=size, align_corners=True, half_pixel_centers=False
        )[0]

    return tf.map_fn(
        crop, (tf.range(tf.shape(images)[0]), images), 
        fn_output_signature=tf.TensorSpec((*size, 3), tf.float32)
    )


def _gray(images: tf.Tensor) -> tf.Tensor:
    """Compute fixed RGB luminance while retaining a singleton channel.

    Args:
        images (tf.Tensor): Float32 RGB tensor [N,H,W,3] in pixel space.

    Returns:
        tf.Tensor: Float32 luminance [N,H,W,1] using coefficients 0.299, 0.587, 0.114."""

    return tf.reduce_sum(images * tf.constant([0.299, 0.587, 0.114]), axis=-1, keepdims=True)


def _color_operation(images: tf.Tensor, index: tf.Tensor, factor: tf.Tensor) -> tf.Tensor:
    """Apply one color-jitter operation with independent per-image factors.

    Args:
        images (tf.Tensor): Float32 RGB pixel tensor [N,H,W,3].
        index (tf.Tensor): Int32 scalar 0=brightness, 1=contrast, 2=saturation, 3=hue.
        factor (tf.Tensor): Float32 per-image factors [N,1,1,1]; multiplicative except
            hue, whose signed value denotes a fraction of one HSV revolution.

    Returns:
        tf.Tensor: Float32 RGB with unchanged shape. Brightness, contrast and saturation
            clip to [0,1]; hue wraps modulo one and retains the remaining HSV channels."""

    def brightness() -> tf.Tensor:
        """Scale intensity multiplicatively and clip to pixel range."""

        return tf.clip_by_value(images * factor, 0., 1.)

    def contrast() -> tf.Tensor:
        """Blend with the image's scalar mean luminance, not channel means."""

        mean = tf.reduce_mean(_gray(images), axis=(1, 2), keepdims=True)
        return tf.clip_by_value(images * factor + mean * (1. - factor), 0., 1.)

    def saturation() -> tf.Tensor:
        """Blend each pixel with its grayscale value."""

        return tf.clip_by_value(images * factor + _gray(images) * (1. - factor), 0., 1.)

    def hue() -> tf.Tensor:
        """Rotate HSV hue by per-image fractions of a full revolution."""

        hsv = tf.image.rgb_to_hsv(images)
        hue_channel = tf.math.floormod(hsv[..., :1] + factor, 1.)
        return tf.image.hsv_to_rgb(tf.concat((hue_channel, hsv[..., 1:]), axis=-1))

    return tf.switch_case(index, (brightness, contrast, saturation, hue))


def _color_jitter(images: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    """Jitter with independent factors/masks and one random batch-shared operation order.

    Args:
        images (tf.Tensor): Float32 pixel-space RGB [N,H,W,3].
        seed (tf.Tensor): Integer stateless seed [2] selecting factors, ordering and masks.

    Returns:
        tf.Tensor: Float32 RGB of identical shape. Each row is transformed with p=0.8;
            unselected rows preserve their input exactly. Factor ranges are brightness
            and contrast [0.6,1.4), saturation [0.8,1.2), hue [-0.1,0.1)."""

    count = tf.shape(images)[0]
    apply = _uniform(seed, 0, tuple([count])) < _COLOR_JITTER_PROBABILITY
    lows = tf.constant([0.6, 0.6, 0.8, -0.1])[:, None]
    spans = tf.constant([0.8, 0.8, 0.4, 0.2])[:, None]
    factors = (lows + spans * _uniform(seed, 1, (4, count)))[:, :, None, None, None]
    order = tf.random.experimental.stateless_shuffle(tf.range(4), _fold(seed, 2))
    jittered = images
    for position in range(4):
        index = order[position]
        jittered = _color_operation(jittered, index, factors[index])
    return tf.where(apply[:, None, None, None], jittered, images)


def _solarize(images: tf.Tensor) -> tf.Tensor:
    """Clip pixel values, then invert intensities at or above 0.5.

    Args:
        images (tf.Tensor): Float32 pixel-space image tensor of any shape.

    Returns:
        tf.Tensor: Same shape/dtype; clipped values below 0.5 stay unchanged, others become 1-value."""

    images = tf.clip_by_value(images, 0., 1.)
    return tf.where(images < 0.5, images, 1. - images)


def _consolidation_view(
    pixels: tf.Tensor, seed: tf.Tensor, view_index: int, size: tuple[int, int]
) -> tf.Tensor:
    """Generate one independent pixel-space view with a zero-based view index.

    Args:
        pixels (tf.Tensor): Float32 clean RGB [N,H,W,3] in [0,1].
        seed (tf.Tensor): Integer stateless seed [2] unique to this view.
        view_index (int): Zero-based view position; odd positions permit solarization.
        size (tuple[int, int]): Positive resized output height/width.

    Returns:
        tf.Tensor: Float32 [N,size[0],size[1],3] after crop, color jitter, grayscale
            (p=0.2), flip (p=0.5), then solarization (p=0.2 on eligible views).
            Outputs remain in pixel scale with possible unclipped bicubic overshoot."""

    images = _random_resized_crop(pixels, _fold(seed, 0), size)
    images = _color_jitter(images, _fold(seed, 1))
    gray = _uniform(seed, 2, tuple([tf.shape(images)[0]])) < _GRAYSCALE_PROBABILITY
    images = tf.where(gray[:, None, None, None], _gray(images), images)
    images = _flip(images, _fold(seed, 3))
    # Paper views 2, 4, ... include solarization; view_index is zero-based.
    if (view_index + 1) % 2 == 0:
        solarize = _uniform(seed, 4, tuple([tf.shape(images)[0]])) < _SOLARIZE_PROBABILITY
        images = tf.where(solarize[:, None, None, None], _solarize(images), images)
    return images


def acquisition_augmentation(images: tf.Tensor, seed: int | tf.Tensor) -> tf.Tensor:
    """Apply TMCL's acquisition policy: independent horizontal flips with p=0.5.

    Args:
        images: Clean float RGB images shaped [N, H, W, 3] in [-1, 1].
        seed: Explicit integer or integer tensor [2]. Derive it from the phase
            and optimizer step in the caller for reproducible training/resume.

    Returns:
        Float32 images with identical geometry and exact input pixel values.
        No crop, intensity transform or diffusion noise is applied here.
    """

    return _flip(_images(images), _fold(_seed_pair(seed), 100))


def consolidation_views(
    images: tf.Tensor, seed: int | tf.Tensor, num_views: int = 4, 
    image_size: int | None = 32
) -> tuple[tf.Tensor, ...]:
    """Generate TMCL Appendix A consolidation views before diffusion noising.

    Each view applies random resized crop (scale [0.08, 1], aspect ratio
    [3/4, 4/3], bicubic), ColorJitter (brightness/contrast/saturation/hue
    0.4/0.4/0.2/0.1, p=0.8), grayscale (p=0.2), horizontal flip (p=0.5),
    then solarization (p=0.2) on one-based even views 2, 4, ... only.

    Args:
        images: Clean float RGB images shaped [N, H, W, 3] in [-1, 1].
        seed: Explicit integer or integer tensor [2], normally derived from
            phase and optimizer step. Views and examples have independent
            parameter draws; calls never mutate global RNG state.
        num_views: Number of independent views; the paper uses four.
        image_size: Output side length; the paper uses 32. ``None`` explicitly
            preserves a statically known input geometry for small experiments.
            Using any other geometry is a departure from the paper recipe.

    Returns:
        A tuple of ``num_views`` float32 NHWC tensors. List position 0 is paper
        view 1. Seeds of earlier views are unaffected by requesting more views.
        Outputs use model scale [-1, 1], with possible bicubic overshoots when
        no subsequent clipping color operation is selected.

    Raises:
        ValueError: If seed, view count or output geometry is invalid.
        tf.errors.InvalidArgumentError: If images are not finite RGB NHWC
            tensors in the clean model-input range.
    """

    # View counts must be explicit positive integers, excluding booleans.
    if isinstance(num_views, bool) or not isinstance(num_views, Integral) or num_views < 1:
        raise ValueError("num_views must be a positive integer")
    images = _images(images)
    # Explicit geometry preservation supports small experimental models only.
    if image_size is None:
        size = tuple(images.shape[1:3])
        # Dynamic dimensions cannot define map_fn's output tensor signature.
        if any(dimension is None for dimension in size):
            raise ValueError("image_size=None requires statically known image geometry")
    # A square output defaults to the paper's 32 by 32 crop size.
    else:
        # Reject dimensions that TensorFlow would otherwise coerce silently.
        if isinstance(image_size, bool) or not isinstance(image_size, Integral) or image_size < 1:
            raise ValueError("image_size must be a positive integer or None")
        size = (int(image_size), int(image_size))
    seed = _fold(_seed_pair(seed), 200)
    pixels = (images + 1.) * 0.5
    return tuple(
        2. * _consolidation_view(pixels, _fold(seed, index), index, size) - 1.
        for index in range(int(num_views))
    )
