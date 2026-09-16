"""Display generated replay without changing training pools or random streams."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from common.runtime import derive_seed


def _sample_null_preview(model: object, seed: int | None, verbose: bool) -> np.ndarray:
    """Sample the CFG null condition while preserving checkpointed random state.

    Args:
        model (DiffusionModel): CFG-enabled diffusion wrapper.
        seed (int | None): Independent preview seed; None uses the model seed.
        verbose (bool): Whether to print reverse-diffusion progress.

    Returns:
        np.ndarray: One unconditional image in the sampler's [0, 1] coordinates.
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
        return DiffusionModel.sample(
            model, network_name=model.test_network_name, labels=[0],
            scale=1.0, seed=seed, verbose=verbose,
        ).numpy()[0]
    finally:
        for stream, weights in streams:
            stream.set_weights(weights)


def show_generated_replay(
    samples: np.ndarray,
    labels: np.ndarray,
    original_labels: Mapping[int, object],
    *,
    generative_model: object | None = None,
    data_min: float = 0.0,
    data_range: float = 1.0,
    seed: int | None = None,
    verbose: bool = False,
) -> None:
    """Show a random replay example per represented class and a CFG null preview.

    Args:
        samples (np.ndarray): Nonempty replay pool in loader coordinates, NHW
            grayscale or NHWC with one, three, or four channels.
        labels (np.ndarray): Aligned dense class IDs, before the CFG offset.
        original_labels (Mapping[int, object]): Dense IDs to dataset labels.
        generative_model (object | None): Optional diffusion wrapper. A null
            sample is drawn only when this wrapper supports CFG.
        data_min (float): Shared loader-space lower pixel bound.
        data_range (float): Positive shared loader pixel range.
        seed (int | None): Local preview seed, independent of replay selection.
        verbose (bool): Print progress for the additional null sample.

    Returns:
        None: Displays and closes a Matplotlib figure. Empty or non-image pools
        produce no figure; non-image pools print a short explanation.

    Raises:
        ValueError: Labels do not align or the display scale is invalid.
        KeyError: A represented class has no original-label mapping.
    """
    images = np.asarray(samples)
    ids = np.asarray(labels).reshape(-1)
    if len(images) != len(ids):
        raise ValueError("Replay preview images and labels must align.")
    if not len(images):
        return
    if images.ndim == 3:
        images = images[..., None]
    if images.ndim != 4 or images.shape[-1] not in (1, 3, 4):
        print("Generated image preview unavailable for non-image replay.", flush=True)
        return
    if not np.isfinite(data_min) or not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("Replay preview requires a finite, positive pixel range.")

    rng = np.random.default_rng(derive_seed(seed, "replay_preview_selection"))
    previews, titles = [], []
    for label in np.unique(ids):
        index = int(rng.choice(np.flatnonzero(ids == label)))
        previews.append((images[index].astype(np.float64) - data_min) / data_range)
        titles.append(f"Class {original_labels[int(label)]}")

    from diffusion.models.wrapper.diffusion_model import DiffusionModel

    if isinstance(generative_model, DiffusionModel) and generative_model.use_cfg:
        if verbose:
            print("Generating null-conditioned image preview...", flush=True)
        previews.insert(0, _sample_null_preview(
            generative_model, derive_seed(seed, "replay_preview_null"), verbose,
        ))
        titles.insert(0, "Null (unconditional)")

    from matplotlib import pyplot as plt

    columns = min(6, len(previews))
    rows = (len(previews) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
    try:
        for axis, sample, title in zip(axes.flat, previews, titles):
            display = np.clip(sample, 0.0, 1.0)
            if display.shape[-1] == 1:
                axis.imshow(display[..., 0], cmap="gray", vmin=0.0, vmax=1.0)
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
