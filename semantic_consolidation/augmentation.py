"""Stateless TensorFlow image augmentation from TMCL, Appendix A.

Reference: https://arxiv.org/html/2505.14125v3#A1. Acquisition uses only
horizontal flips. Consolidation uses four independently sampled RGB views by
default; solarization is eligible on views 2, 4, ... (paper's one-based view
numbering). Operations run in pixel space before any diffusion corruption;
inputs use NHWC model coordinates and are validated/cast to float32 in [-1, 1].
Acquisition returns float32; consolidation restores the wrapper's configured
model coordinates and variable dtype through its public preprocessing API.

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


def _fold(stream: int | tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    """Derive a deterministic child seed without reading global RNG state.

    The stream identifier is cast to int64 before TensorFlow folds it into the
    parent. Reusing the same parent and identifier reproduces the same child;
    this function neither advances nor stores a random generator.

    Args:
        stream (int | tf.Tensor): Scalar child-stream identifier, converted to
            int64. Callers supply integers to avoid truncating fractional values.
        seed (tf.Tensor): Int32 or int64 stateless seed with shape [2].

    Returns:
        tf.Tensor: Int64 child seed with shape [2], matching the cast stream
        dtype rather than necessarily the parent seed dtype.

    Raises:
        TypeError: If TensorFlow cannot convert the stream or rejects the seed
            dtype.
        ValueError: If a statically known seed/stream shape is incompatible.
        tf.errors.InvalidArgumentError: If a runtime seed or stream shape is
            invalid for TensorFlow's stateless random operation.
    """

    return tf.random.experimental.stateless_fold_in(seed, tf.cast(stream, tf.int64))


def _uniform(stream: int, seed: tf.Tensor, shape: object = ()) -> tf.Tensor:
    """Draw float32 uniforms from a named stateless child stream.

    The same stream, parent seed, and shape reproduce the same tensor. Draws
    do not advance global RNG state; callers choose distinct stream identifiers
    when independent draws are required.

    Args:
        stream (int): Scalar integer child-stream identifier, cast to int64
            before it is folded into the parent seed.
        seed (tf.Tensor): Int32 or int64 stateless seed with shape [2].
        shape (object): TensorFlow-compatible integer shape, such as a tuple
            of nonnegative dimensions or a rank-one int32/int64 tensor.
            Defaults to (), which requests a scalar.

    Returns:
        tf.Tensor: Float32 values in [0, 1), with the requested shape.

    Raises:
        TypeError: If TensorFlow rejects a shape/seed dtype or cannot convert
            an argument.
        ValueError: If a statically known shape or seed is incompatible.
        tf.errors.InvalidArgumentError: If a runtime seed is not length two,
            the output shape is not an integer vector, or a dimension is negative.
    """

    return tf.random.stateless_uniform(shape, _fold(stream=stream, seed=seed), dtype=tf.float32)


def _images(images: tf.Tensor) -> tf.Tensor:
    """Check model-scale RGB images before converting them to pixel space.

    Args:
        images (tf.Tensor): Numeric NHWC tensor [N,H,W,3] in [-1,1], with positive dimensions.

    Returns:
        tf.Tensor: Float32 NHWC images with identical geometry and values after casting.

    Raises:
        ValueError: If a statically known rank cannot represent NHWC images.
        tf.errors.InvalidArgumentError: If dimensions, channels, finiteness or model scale are invalid.
            The checks establish numeric bounds only; they do not identify whether
            in-range data was previously rescaled or noised.
    """

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

    Flips are stateless and preserve values exactly. The helper assumes an
    NHWC input; public augmentation entry points own the image validation.

    Args:
        images (tf.Tensor): Float32 NHWC tensor [N,H,W,C], in either model or pixel scale.
        seed (tf.Tensor): Integer stateless seed vector [2].

    Returns:
        tf.Tensor: Same dtype/shape/pixels; selected rows reverse only the width axis.

    Raises:
        TypeError: If TensorFlow rejects the seed dtype.
        ValueError: If statically known image/seed dimensions cannot satisfy
            the NHWC broadcast and reversal operations.
        tf.errors.InvalidArgumentError: If incompatible ranks, dimensions, or
            seed shapes are discovered at runtime.
    """

    mask = _uniform(stream=0, shape=tuple([tf.shape(images)[0]]), seed=seed) < _FLIP_PROBABILITY
    return tf.where(mask[:, None, None, None], tf.reverse(images, axis=[2]), images)


def _crop_box(shape: tf.Tensor, seed: tf.Tensor) -> tuple[tf.Tensor, ...]:
    """Sample crop area uniformly and aspect ratio log-uniformly with ten attempts.

    The caller supplies positive image dimensions. This helper adds no input
    validation and uses only local stateless streams; no image data is changed.

    Args:
        shape (tf.Tensor): Int32 image shape with positive height/width in its first two entries.
        seed (tf.Tensor): Integer stateless seed vector [2], unique to the image/view.

    Returns:
        tuple[tf.Tensor, ...]: Scalar int32 (top, left, height, width). The first valid
            proposal uses uniform integer origins; if all ten fail, return a centered
            crop constrained to the configured aspect-ratio range and image bounds.

    Raises:
        TypeError: If shape or seed dtypes are incompatible with the int32
            crop calculations and stateless RNG.
        ValueError: If statically known shape/seed dimensions are incompatible.
        tf.errors.InvalidArgumentError: If a runtime shape cannot supply the
            height/width entries or the stateless seed is invalid.
    """

    height, width = shape[0], shape[1]
    height_f, width_f = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
    area = height_f * width_f * (
        _CROP_SCALE[0] + (_CROP_SCALE[1] - _CROP_SCALE[0]) * _uniform(stream=0, shape=tuple([10]), seed=seed)
    )
    log_min = tf.math.log(tf.constant(_CROP_RATIO[0], tf.float32))
    log_max = tf.math.log(tf.constant(_CROP_RATIO[1], tf.float32))
    ratio = tf.exp(log_min + (log_max - log_min) * _uniform(stream=1, shape=tuple([10]), seed=seed))
    widths = tf.cast(tf.round(tf.sqrt(area * ratio)), tf.int32)
    heights = tf.cast(tf.round(tf.sqrt(area / ratio)), tf.int32)
    valid = (widths > 0) & (widths <= width) & (heights > 0) & (heights <= height)

    def sampled() -> tuple[tf.Tensor, ...]:
        """Use the first valid proposal and independently sample its origin.

        This zero-argument branch closes over the enclosing int32 proposal arrays
        and image dimensions, Boolean validity mask, and stateless seed. The
        parent calls it only when at least one of its ten proposals fits the image.
        Independent folded streams choose top and left coordinates, including the
        last legal origin.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: Int32 scalars
            (top, left, height, width) defining an in-bounds nonempty crop.

        Raises:
            tf.errors.InvalidArgumentError: If directly executing the closure with
                malformed captured tensors makes indexing or stateless sampling
                invalid. The branch adds no validation beyond the parent's contract.
        """

        index = tf.argmax(tf.cast(valid, tf.int32), output_type=tf.int32)
        crop_height, crop_width = heights[index], widths[index]
        top = tf.cast(_uniform(stream=2, seed=seed) * tf.cast(height - crop_height + 1, tf.float32), tf.int32)
        left = tf.cast(_uniform(stream=3, seed=seed) * tf.cast(width - crop_width + 1, tf.float32), tf.int32)
        return top, left, crop_height, crop_width

    def central() -> tuple[tf.Tensor, ...]:
        """Return the centered fallback when no sampled crop proposal fits.

        This zero-argument branch reads the enclosing positive int32 height/width
        and their float32 copies. It keeps the full image when its aspect ratio is
        within the configured bounds; otherwise it shortens the oversized axis,
        rounds to integer pixels, and clips each extent to the image bounds.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: Int32 scalars
            (top, left, height, width). Integer division places any odd leftover
            margin after the crop.

        Raises:
            None: No explicit exception is raised for the parent's positive scalar
                dimensions; this branch performs no independent shape/range checks.
        """

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


def _random_resized_crop(images: tf.Tensor, size: tuple[int, int], seed: tf.Tensor) -> tf.Tensor:
    """Crop images independently and resize using corner-aligned bicubic interpolation.

    Args:
        images (tf.Tensor): Numeric RGB pixel-space tensor [N,H,W,3]. ResizeBicubic
            converts each supported numeric image dtype to float32.
        size (tuple[int, int]): Positive output height/width, fixed for map_fn's output signature.
        seed (tf.Tensor): Integer stateless seed [2]; row indices select independent crop streams.

    Returns:
        tf.Tensor: Float32 [N,size[0],size[1],3]. Bicubic overshoot is retained.

    Raises:
        TypeError: If TensorFlow rejects the image, output-size, or seed dtype.
        ValueError: If a statically known image/output geometry is incompatible
            with RGB map_fn output or bicubic resize.
        tf.errors.InvalidArgumentError: If runtime slicing, seed, or resize
            dimensions are invalid.
    """

    def crop(row: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Resize one sampled crop using the image-index-specific seed.

        Args:
            row (tuple[tf.Tensor, tf.Tensor]): A scalar int32 row index and its
                numeric RGB image [H,W,3]. The index is folded into the enclosing
                stateless seed to select this image's independent crop.

        Returns:
            tf.Tensor: Float32 RGB image [size[0],size[1],3], resized with
                corner-aligned bicubic interpolation; overshoot is retained.

        Raises:
            ValueError: If the row cannot unpack into an index/image pair or the
                captured output size has incompatible dimensions.
            TypeError: If TensorFlow rejects an image, seed, or size dtype.
            tf.errors.InvalidArgumentError: If the sampled slice or runtime resize
                geometry is invalid.
        """

        index, image = row
        top, left, height, width = _crop_box(tf.shape(image), _fold(stream=index, seed=seed))
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

    The fixed coefficients are float32, so this internal helper expects the
    float32 output of the crop stage. It does not clip, normalize, or mutate
    its input.

    Args:
        images (tf.Tensor): Float32 RGB tensor [N,H,W,3] in pixel space.

    Returns:
        tf.Tensor: Float32 luminance [N,H,W,1] using coefficients 0.299, 0.587, 0.114.

    Raises:
        TypeError: If the input dtype cannot multiply the float32 luminance
            coefficients without an explicit cast.
        ValueError: If a statically known channel dimension cannot broadcast
            with the three RGB coefficients.
        tf.errors.InvalidArgumentError: If incompatible channels or rank are
            discovered at runtime.
    """

    return tf.reduce_sum(images * tf.constant([0.299, 0.587, 0.114]), axis=-1, keepdims=True)


def _color_operation(images: tf.Tensor, index: tf.Tensor, factor: tf.Tensor) -> tf.Tensor:
    """Apply one color-jitter operation with independent per-image factors.

    The branch index is expected to be in [0,3]; the helper does not validate
    that range separately from TensorFlow switch_case. The input tensors are
    not modified, and no RNG is used in this single-operation helper.

    Args:
        images (tf.Tensor): Float32 RGB pixel tensor [N,H,W,3].
        index (tf.Tensor): Int32 scalar 0=brightness, 1=contrast, 2=saturation, 3=hue.
        factor (tf.Tensor): Float32 per-image factors [N,1,1,1]; multiplicative except
            hue, whose signed value denotes a fraction of one HSV revolution.

    Returns:
        tf.Tensor: Float32 RGB with unchanged shape. Brightness, contrast and saturation
            clip to [0,1]; hue wraps modulo one and retains the remaining HSV channels.

    Raises:
        TypeError: If TensorFlow rejects the branch-index dtype or arithmetic
            operands are not compatible with the float32 color operations.
        ValueError: If a static image/channel/factor shape is incompatible.
        tf.errors.InvalidArgumentError: If an invalid dynamic RGB shape or
            broadcasting combination reaches a selected color operation.
    """

    def brightness() -> tf.Tensor:
        """Multiply the captured RGB image by per-image intensity factors.

        This zero-argument switch branch reads float32 images [N,H,W,3] and
        float32 factors [N,1,1,1] from the enclosing call. It clips the multiplied
        pixels to [0,1] without changing the captured tensors.

        Returns:
            tf.Tensor: Float32 RGB tensor [N,H,W,3] with the transformed pixels.

        Raises:
            ValueError: If statically known image/factor shapes cannot broadcast.
            tf.errors.InvalidArgumentError: If incompatible shapes are discovered
                during TensorFlow execution.
        """

        return tf.clip_by_value(images * factor, 0., 1.)

    def contrast() -> tf.Tensor:
        """Blend captured pixels around each image's scalar mean luminance.

        The zero-argument branch converts the captured float32 [N,H,W,3] images
        to luminance, averages over both spatial axes, and blends around that mean
        with float32 [N,1,1,1] factors. The result is clipped to [0,1]; no batch
        statistics are shared or stored.

        Returns:
            tf.Tensor: Float32 RGB tensor [N,H,W,3] with unchanged geometry.

        Raises:
            ValueError: If statically known image/factor shapes cannot broadcast
                or the image does not have the expected spatial axes.
            tf.errors.InvalidArgumentError: If invalid shape/axis combinations are
                discovered during TensorFlow execution.
        """

        mean = tf.reduce_mean(_gray(images), axis=(1, 2), keepdims=True)
        return tf.clip_by_value(images * factor + mean * (1. - factor), 0., 1.)

    def saturation() -> tf.Tensor:
        """Blend captured RGB pixels with their per-pixel luminance.

        This zero-argument branch reads float32 images [N,H,W,3] and float32
        factors [N,1,1,1]. A zero factor gives grayscale and one preserves the
        input before clipping to [0,1]. The enclosing tensors are not modified.

        Returns:
            tf.Tensor: Float32 RGB tensor [N,H,W,3] with transformed saturation.

        Raises:
            ValueError: If statically known image/factor shapes cannot broadcast.
            tf.errors.InvalidArgumentError: If incompatible shapes are discovered
                during TensorFlow execution.
        """

        return tf.clip_by_value(images * factor + _gray(images) * (1. - factor), 0., 1.)

    def hue() -> tf.Tensor:
        """Rotate captured HSV hue by each image's signed fractional offset.

        This zero-argument branch converts the captured float32 RGB images
        [N,H,W,3] to HSV, adds float32 [N,1,1,1] factors to hue modulo one, and
        converts back to RGB. Saturation and value are retained; there is no
        additional clipping or mutation of the captured tensors.

        Returns:
            tf.Tensor: Float32 RGB tensor [N,H,W,3] with the shifted hues.

        Raises:
            ValueError: If a known channel dimension is not three or the factors
                cannot broadcast to the hue channel.
            tf.errors.InvalidArgumentError: If incompatible channel/shape
                dimensions are discovered during TensorFlow execution.
        """

        hsv = tf.image.rgb_to_hsv(images)
        hue_channel = tf.math.floormod(hsv[..., :1] + factor, 1.)
        return tf.image.hsv_to_rgb(tf.concat((hue_channel, hsv[..., 1:]), axis=-1))

    return tf.switch_case(index, (brightness, contrast, saturation, hue))


def _color_jitter(images: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    """Jitter with independent factors/masks and one random batch-shared operation order.

    All factors, the operation permutation, and selection masks come from
    separate folded stateless streams. There is no mutation of input tensors
    or global RNG state; this internal helper assumes validated RGB geometry.

    Args:
        images (tf.Tensor): Float32 pixel-space RGB [N,H,W,3].
        seed (tf.Tensor): Integer stateless seed [2] selecting factors, ordering and masks.

    Returns:
        tf.Tensor: Float32 RGB of identical shape. Each row is transformed with p=0.8;
            unselected rows preserve their input exactly. Factor ranges are brightness
            and contrast [0.6,1.4), saturation [0.8,1.2), hue [-0.1,0.1).

    Raises:
        TypeError: If TensorFlow rejects the seed or image dtype.
        ValueError: If a statically known RGB, factor, or seed shape is invalid.
        tf.errors.InvalidArgumentError: If incompatible channel/broadcast shapes
            or seed dimensions are discovered during the color operations.
    """

    count = tf.shape(images)[0]
    apply = _uniform(stream=0, shape=tuple([count]), seed=seed) < _COLOR_JITTER_PROBABILITY
    lows = tf.constant([0.6, 0.6, 0.8, -0.1])[:, None]
    spans = tf.constant([0.8, 0.8, 0.4, 0.2])[:, None]
    factors = (lows + spans * _uniform(stream=1, shape=(4, count), seed=seed))[:, :, None, None, None]
    order = tf.random.experimental.stateless_shuffle(tf.range(4), _fold(stream=2, seed=seed))
    jittered = images
    for position in range(4):
        index = order[position]
        jittered = _color_operation(jittered, index, factors[index])
    return tf.where(apply[:, None, None, None], jittered, images)


def _solarize(images: tf.Tensor) -> tf.Tensor:
    """Clip pixel values, then invert intensities at or above 0.5.

    This deterministic transform does not change its input tensor or require
    an RNG. Floating values outside [0,1] are clipped before the threshold test.

    Args:
        images (tf.Tensor): Float32 pixel-space image tensor of any shape.

    Returns:
        tf.Tensor: Same shape/dtype; clipped values below 0.5 stay unchanged, others become 1-value.

    Raises:
        TypeError: If the input dtype is incompatible with floating pixel
            arithmetic.
        tf.errors.InvalidArgumentError: If TensorFlow rejects the input during
            clipping or comparison. No shape/range validation is added here.
    """

    images = tf.clip_by_value(images, 0., 1.)
    return tf.where(images < 0.5, images, 1. - images)


def _consolidation_view(
    pixels: tf.Tensor, view_index: int, size: tuple[int, int], seed: tf.Tensor
) -> tf.Tensor:
    """Generate one independent pixel-space view with a zero-based view index.

    Separate child streams select each transform's parameters; repeating the
    same inputs, view index, size, and seed reproduces the result. Inputs and
    global RNG state are unchanged.

    Args:
        pixels (tf.Tensor): Numeric clean RGB [N,H,W,3] in [0,1]. The crop resize
            converts supported numeric dtypes to float32.
        view_index (int): Zero-based view position; odd positions permit solarization.
        size (tuple[int, int]): Positive resized output height/width.
        seed (tf.Tensor): Integer stateless seed [2] unique to this view.

    Returns:
        tf.Tensor: Float32 [N,size[0],size[1],3] after crop, color jitter, grayscale
            (p=0.2), flip (p=0.5), then solarization (p=0.2 on eligible views).
            Outputs remain in pixel scale with possible unclipped bicubic overshoot.

    Raises:
        TypeError: If the view index cannot support integer arithmetic, or
            TensorFlow rejects an image/seed/size dtype.
        ValueError: If a statically known crop, RGB, seed, or output shape is
            incompatible.
        tf.errors.InvalidArgumentError: If a dynamic crop, resize, color, or seed
            shape is invalid. The public caller owns image/count/size validation.
    """

    images = _random_resized_crop(pixels, size=size, seed=_fold(stream=0, seed=seed))
    images = _color_jitter(images, _fold(stream=1, seed=seed))
    gray = _uniform(stream=2, shape=tuple([tf.shape(images)[0]]), seed=seed) < _GRAYSCALE_PROBABILITY
    images = tf.where(gray[:, None, None, None], _gray(images), images)
    images = _flip(images, _fold(stream=3, seed=seed))
    # Paper views 2, 4, ... include solarization; view_index is zero-based.
    if (view_index + 1) % 2 == 0:
        solarize = _uniform(stream=4, shape=tuple([tf.shape(images)[0]]), seed=seed) < _SOLARIZE_PROBABILITY
        images = tf.where(solarize[:, None, None, None], _solarize(images), images)
    return images


def acquisition_augmentation(images: tf.Tensor, seed: int | tf.Tensor) -> tf.Tensor:
    """Apply independent horizontal flips before acquisition diffusion noise.

    Each row is flipped with probability 0.5 using a stateless stream derived
    from the explicit seed. Inputs are checked and cast to float32 first;
    the same images and seed reproduce the same result without advancing
    global RNG state. No crop, intensity transform, or diffusion noise is
    applied, and this helper does not call a wrapper preprocessing API.

    Args:
        images (tf.Tensor): Numeric clean RGB images [N,H,W,3] in [-1,1].
            All dimensions must be positive and all values finite. Non-float32
            numeric inputs are converted to float32 before validation.
        seed (int | tf.Tensor): Explicit integer scalar or integer tensor [2].
            A scalar becomes an int64 seed pair with a zero second word.
            Derive this required seed from the phase and optimizer step for
            reproducible training and recovery.

    Returns:
        tf.Tensor: Float32 images [N,H,W,3]. Geometry and input pixel values
        after casting are preserved; selected rows reverse only the width axis.

    Raises:
        TypeError: If images or seed cannot be converted to TensorFlow tensors.
        ValueError: If the seed is not integer scalar/[2], or a statically known
            image rank cannot represent NHWC images.
        tf.errors.InvalidArgumentError: If the images have an empty dimension,
            non-RGB channels, nonfinite values, or values outside [-1,1].
    """

    return _flip(_images(images), _fold(stream=100, seed=_seed_pair(seed)))


def consolidation_views(
    images: tf.Tensor, wrapper: object, seed: int | tf.Tensor, num_views: int = 4, 
    image_size: int | None = 32
) -> tuple[tf.Tensor, ...]:
    """Generate independent consolidation views before diffusion corruption.

    Each view applies a random resized crop (area scale [0.08,1], aspect ratio
    [3/4,4/3], corner-aligned bicubic resize), color jitter with probability
    0.8, grayscale with probability 0.2, and a horizontal flip with probability
    0.5. One-based even views additionally permit solarization with probability
    0.2. Color-jitter factors use brightness/contrast ranges [0.6,1.4),
    saturation [0.8,1.2), and hue offsets [-0.1,0.1).

    The wrapper's public postprocess/preprocess pair converts its configured
    model coordinates through raw pixels into min-max [0,1] pixel space.
    After augmentation, the inverse route restores the wrapper's configured
    model coordinates. These conversions belong to the wrapper; no independent
    dataset normalizer is applied here. Internal resizing/color operations use
    float32; the final wrapper conversion determines the returned dtype.
    Random streams are stateless and do not change the global RNG.

    Args:
        images (tf.Tensor): Numeric clean RGB images [N,H,W,3] in the wrapper's
            configured model coordinates. This helper first casts to float32
            and requires finite values in [-1,1] and positive dimensions.
            The standardize setting therefore expects [-1,1], while min-max
            expects [0,1]; these are model inputs, not general raw [0,255] images.
        wrapper (object): Diffusion wrapper exposing public preprocess and
            postprocess methods with the preprocess_type override. Its saved
            preprocess_type must match the incoming coordinates; its variable
            dtype policy controls the final output dtype.
        seed (int | tf.Tensor): Required integer scalar or integer tensor [2],
            normally derived from phase and optimizer step. Views and examples
            receive independent child streams. Reusing the same seed and data
            reproduces the same views.
        num_views (int): Positive number of views, excluding booleans.
            Defaults to 4. Requesting extra views does not change earlier ones.
        image_size (int | None): Positive square output side length, excluding
            booleans. Defaults to 32. None preserves the statically known input
            height and width, including nonsquare geometry.

    Returns:
        tuple[tf.Tensor, ...]: num_views NHWC tensors [N,out_h,out_w,3], in
        wrapper.dtype_policy.variable_dtype and its configured model
        coordinates. Position zero is view one. The nominal range is [-1,1]
        for standardize or [0,1] for min-max; bicubic overshoots can remain
        when a later clipping color operation is not selected.

    Raises:
        TypeError: If images/seed cannot be converted to TensorFlow tensors
            or the wrapper does not accept the required preprocessing arguments.
        AttributeError: If the wrapper lacks the public preprocessing methods.
        ValueError: If seed, view count, or output size is invalid; image rank
            is statically incompatible; image_size=None has unknown spatial
            dimensions; or the wrapper rejects its configured preprocessing mode.
        tf.errors.InvalidArgumentError: If images are not finite, nonempty
            RGB NHWC tensors in [-1,1], or a dynamic crop/resize shape is invalid.
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
    seed = _fold(stream=200, seed=_seed_pair(seed))
    pixels = wrapper.preprocess(wrapper.postprocess(images), preprocess_type="min-max")
    return tuple(
        wrapper.preprocess(wrapper.postprocess(
            _consolidation_view(pixels, view_index=index, size=size, seed=_fold(stream=index, seed=seed)), 
            preprocess_type="min-max"
        ))
        for index in range(int(num_views))
    )
