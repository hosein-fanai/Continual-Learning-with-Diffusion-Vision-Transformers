"""Display generated replay without changing training pools or random streams."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from common.runtime import derive_seed


def _sample_null_preview(model: object, verbose: bool, seed: int | None) -> np.ndarray:
    """Sample the CFG null condition while preserving checkpointed random state.

    Args:
        model (DiffusionModel): CFG-enabled diffusion wrapper.
        verbose (bool): Whether to print reverse-diffusion progress.
        seed (int | None): Independent preview seed; None uses the model seed.

    Returns:
        numpy.ndarray: One floating [H, W, C] image converted explicitly
        from raw sampler pixels to display [0,1] by wrapper.preprocess. Its dtype
        follows the wrapper conversion. All tracked SeedStream weights are
        restored in a finally block; the source replay pool is not touched.

    Raises:
        ValueError: If the selected network, seed, or sampling settings are invalid. Stream weights are restored before the error propagates.
    """

    from common.random import SeedStream
    from diffusion.models.wrapper.diffusion_model import DiffusionModel


    streams = [
        (layer, layer.get_weights())
        for layer in model._flatten_layers()
        if isinstance(layer, SeedStream)
    ]
    try:
        # A display-only null sample must bypass subclass replay-capture hooks:
        # it is neither an old-class candidate nor a training example.
        pixels = DiffusionModel.sample(
            model, network_name=model.test_network_name, labels=[0], 
            scale=1.0, verbose=verbose, seed=seed
        )
        return model.preprocess(pixels, "min-max").numpy()[0]
    finally:
        for stream, weights in streams:
            stream.set_weights(weights)


def show_generated_replay(
    samples: np.ndarray, 
    labels: np.ndarray, 
    original_labels: Mapping[int, object], 
    generative_model: object | None = None, 
    data_min: float = 0.0, 
    data_range: float = 1.0, 
    verbose: bool = False, 
    seed: int | None = None
) -> None:
    """Show a random replay example per represented class and a CFG null preview.

    Args:
        samples (np.ndarray): Nonempty replay pool in raw ``[0,255]`` coordinates
            for diffusion, otherwise loader coordinates; NHW grayscale or NHWC
            with one, three, or four channels.
        labels (np.ndarray): Aligned dense class IDs, before the CFG offset.
        original_labels (Mapping[int, object]): Dense IDs to dataset labels.
        generative_model (object | None): Optional diffusion wrapper. A null
            sample is drawn only when this wrapper supports CFG. None uses
            the generic data_min/data_range display transform without a null sample.
            Defaults to ``None``.
        data_min (float): Shared loader-space lower pixel bound.
            Defaults to ``0.0``.
        data_range (float): Positive shared loader pixel range.
            Defaults to ``1.0``.
        verbose (bool): Print progress for the additional null sample.
            Defaults to ``False``.
        seed (int | None): Local preview seed, independent of replay selection; None
            creates an unseeded selection RNG and defers null sampling to the model seed.
            Defaults to ``None``.

    Returns:
        None: Displays and closes a Matplotlib figure. Empty or non-image pools
        produce no figure; non-image pools print a short explanation.

    Raises:
        ValueError: Labels do not align or the display scale is invalid.
        KeyError: A represented class has no original-label mapping.
    """

    images = np.asarray(samples)
    ids = np.asarray(labels).reshape(-1)
    # Prevent displayed class titles from being paired with different image rows.
    if len(images) != len(ids):
        raise ValueError("Replay preview images and labels must align.")
    # An empty replay pool has no images to display.
    if not len(images):
        return
    # Treat NHW grayscale input as a one-channel image batch.
    if images.ndim == 3:
        images = images[..., None]
    # Feature vectors and unsupported channel counts have no image preview.
    if images.ndim != 4 or images.shape[-1] not in (1, 3, 4):
        print("Generated image preview unavailable for non-image replay.", flush=True)
        return
    # The display transform must not invert or corrupt the image intensity range.
    if not np.isfinite(data_min) or not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("Replay preview requires a finite, positive pixel range.")

    from diffusion.models.wrapper.diffusion_model import DiffusionModel


    is_diffusion = isinstance(generative_model, DiffusionModel)
    rng = np.random.default_rng(derive_seed(seed, "replay_preview_selection"))
    previews, titles = [], []
    for label in np.unique(ids):
        index = int(rng.choice(np.flatnonzero(ids == label)))
        previews.append(generative_model.preprocess(images[index], "min-max").numpy()
                        if is_diffusion else (images[index].astype(np.float64) - data_min) / data_range)
        titles.append(f"Class {original_labels[int(label)]}")

    # Add an unconditional preview only when the model has a CFG null embedding.
    if is_diffusion and generative_model.use_cfg:
        # Print progress only for explicitly verbose preview requests.
        if verbose:
            print("Generating null-conditioned image preview...", flush=True)
        previews.insert(0, _sample_null_preview(
            generative_model, verbose=verbose, seed=derive_seed(seed, "replay_preview_null")
        ))
        titles.insert(0, "Null (unconditional)")

    from matplotlib import pyplot as plt


    columns = min(6, len(previews))
    rows = (len(previews) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
    try:
        for axis, sample, title in zip(axes.flat, previews, titles):
            display = np.clip(sample, 0.0, 1.0)
            # Render single-channel images with fixed grayscale intensity limits.
            if display.shape[-1] == 1:
                axis.imshow(display[..., 0], cmap="gray", vmin=0.0, vmax=1.0)
            # Render RGB/RGBA samples using their supplied color channels.
            else:
                axis.imshow(display)
            axis.set_title(title)
        for axis in axes.flat:
            axis.axis("off")
        figure.suptitle("Generated replay examples and null preview" if titles[0].startswith("Null")
                         else "Generated replay examples")
        figure.tight_layout()
        plt.show()
    finally:
        plt.close(figure)
