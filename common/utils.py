"""Initialize notebook paths, plot outputs, extract features, and persist samples.

The plotting helpers display figures or save static images, CSV histories, and
GIF trajectories. Feature extraction uses a frozen Xception trunk and can save
label-split metadata. Numeric sample archives default to non-pickled storage;
legacy object arrays require explicit trust opt-in when loading. File writers
use their supplied paths and generally expect parent directories to exist.
"""

from __future__ import annotations

import numpy as np

import os

import json

import warnings

from pathlib import Path
from runpy import run_path

from collections.abc import Iterable, Mapping, Sequence


models_path = "./files/models"
hyperas_path = os.path.join(models_path, "hyperas")

best_score = -float("inf")
i = 1


def init() -> None:
    """Set notebook import paths and load TensorFlow for an existing checkout.

    Resolve the checkout from this module, change the working directory to its
    root, and add the root and ``files/notebooks/thesis`` to ``sys.path`` without
    duplicate entries. Reuse the shared standard-library notebook path setup
    before importing TensorFlow through Python's normal module cache.

    Call this after preparing dependencies. It does not install packages, change
    device visibility, limit GPU memory, set a dtype policy, or seed randomness.

    Returns:
        None: Checkout paths are configured and TensorFlow is imported.

    Raises:
        OSError: The shared initializer cannot be read or the checkout cannot
            become the working directory.
        ImportError: TensorFlow or one of its dependencies is unavailable.
    """

    run_path(
        str(Path(__file__).resolve().parents[1] / "files" / "notebooks" / "init.py"), 
        run_name="init"
    )

    import tensorflow as tf


    # # Enable incremental allocation only when a GPU is available.
    # if gpus:=tf.config.list_physical_devices("GPU"):
    #     try:
    #         tf.config.set_logical_device_configuration(gpus[0], [
    #             tf.config.LogicalDeviceConfiguration(memory_limit=6144)
    #         ])
    #     except RuntimeError as e:
    #         print(e)
    #         print("Could not limit gpu memory.")


def extract_features(
    dataset_list: Iterable[object], 
    batch_size: int = 128, 
    file_name: str | os.PathLike[str] | None = None, 
    split_seed: int = 42, 
    validation_ratio: float = 0.2
) -> list[np.ndarray]:
    """Extract 2,048-wide Xception features for multiple sample arrays.

    Args:
        dataset_list (Iterable[numpy.ndarray | tf.Tensor]): Image arrays, each
            normally shaped ``[samples, 32, 32, 3]`` with numeric raw
            [0,255] pixels; the selected backbone owns float casting and scaling.  Every array is passed to
            a frozen resize/preprocess/Xception/global-pooling model.
        batch_size (int): Positive prediction batch size; defaults to 128.
        file_name (str | os.PathLike | None): Optional base path without
            ``.npy``. When supplied, the feature arrays are saved as an ordered,
            non-pickled NumPy container via :func:`save_samples`.
            Defaults to ``None``, returning features without saving an archive.
        split_seed (int): Seed that produced the label-aligned train/validation
            feature split. It is saved beside three-array archives so loading
            can reconstruct the identical label ordering.
            Defaults to ``42``.
        validation_ratio (float): Fraction assigned to the saved validation
            feature array. It is stored with ``split_seed``.
            Defaults to ``0.2``.

    Returns:
        list[numpy.ndarray]: One floating feature array per input dataset,
        normally shaped ``[samples, 2048]`` in the extractor's compute
        dtype. Constructing the extractor can download pretrained weights; prediction
        runs with frozen backbone weights and can print Keras progress output.

    Raises:
        ValueError: If an input image shape is incompatible with the pretrained feature model or a saved bundle contains unsupported arrays.
        OSError: If pretrained weights cannot be read/downloaded or requested feature/metadata files cannot be written.
    """

    from tensorflow.keras import models

    from common.model import get_model


    conv_base = models.Sequential(
        get_model(10, model_type="pretrained", verbose=0).layers[:4]
    )
    conv_base.trainable = False

    features_list = []
    for dataset in dataset_list:
        features = conv_base.predict(dataset, batch_size=batch_size)
        features_list.append(features)

    del conv_base

    # Persist extracted features when an output prefix is supplied.
    if file_name is not None:
        feature_bundle = np.empty(len(features_list), dtype=object)
        feature_bundle[:] = features_list
        save_samples(feature_bundle, file_name, ".npy")

        # The project feature loader consumes exactly train/validation/test
        # archives. Record their split contract beside the safe array container.
        if len(features_list) == 3:
            save_feature_split_metadata(
                file_name, 
                split_seed=split_seed, 
                validation_ratio=validation_ratio
            )

    return features_list


def save_feature_split_metadata(
    path: str | os.PathLike[str], 
    split_seed: int, 
    validation_ratio: float = 0.2
) -> Path:
    """Save the label-split contract beside a train/val/test feature archive.

    Args:
        path (str | os.PathLike): Feature-archive base path without ``.npy``.
        split_seed (int): Seed used for the stratified train/validation split.
        validation_ratio (float): Fraction assigned to validation features.
            Defaults to ``0.2``.

    Returns:
        pathlib.Path: Written ``.metadata.json`` sidecar path.

    Raises:
        OSError: If the sidecar parent directory is absent or the file cannot be opened/written.
    """

    metadata_path = Path(os.fspath(path) + ".metadata.json")
    payload = {
        "format_version": 1, 
        "label_split": {
            "random_state": int(split_seed), 
            "validation_ratio": float(validation_ratio)
        }
    }

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(payload, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")

    return metadata_path


def load_feature_split_metadata(
    path: str | os.PathLike[str]
) -> tuple[int, float] | None:
    """Load a feature archive's label-split seed and ratio when available.

    Legacy NPY archives have no sidecar and return ``None`` so callers can
    retain their historical seed-42 interpretation.

    Args:
        path (str | os.PathLike): Feature-archive base path without ``.npy``.

    Returns:
        tuple[int, float] | None: Stored split seed and validation fraction, or
        ``None`` when a legacy archive has no metadata sidecar.

    Raises:
        ValueError: If a sidecar has an unsupported or incomplete schema, or
            contains an invalid split seed/fraction.
        OSError: If an existing sidecar cannot be opened.
    """

    metadata_path = Path(os.fspath(path) + ".metadata.json")
    # Preserve the established interpretation of metadata-free archives.
    if not metadata_path.is_file():
        return None

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        payload = json.load(metadata_file)

    # Accept only the versioned schema written by the paired save helper.
    if not isinstance(payload, Mapping) or payload.get("format_version") != 1:
        raise ValueError("Unsupported feature split metadata format.")

    label_split = payload.get("label_split")

    # Require a nested mapping before looking up its fields.
    if not isinstance(label_split, Mapping):
        raise ValueError("Feature metadata must contain a label_split mapping.")
    # Reject partial metadata rather than guessing an archive layout.
    if "random_state" not in label_split \
    or "validation_ratio" not in label_split:
        raise ValueError(
            "Feature label_split metadata requires random_state and "
            "validation_ratio."
        )

    return int(label_split["random_state"]), float(label_split["validation_ratio"])


def CL_plot(
    class_num: int, 
    pairs: Iterable[tuple[Sequence[float], str]], 
    class_counts: Sequence[int] | None = None
) -> None:
    """Plot continual-learning accuracy against the number of seen classes.

    Args:
        class_num (int): Final class count.  The x-axis is integer values from
            2 through this value inclusive.
        pairs (Iterable[tuple[Sequence[float], str]]): Accuracy series and
            legend labels.  Each series should contain ``class_num - 1``
            values; multiple pairs create comparison curves.
        class_counts (Sequence[int] | None): Optional seen-class count for each
            configured task. ``None`` preserves the legacy two-through-N axis.
            Defaults to ``None``.

    Returns:
        None: Matplotlib displays the figure interactively.

    Raises:
        ValueError: If an accuracy series length differs from the x-axis length.
    """

    from matplotlib import pyplot as plt


    # Use the legacy two-through-N class axis unless explicit task class counts were supplied.
    x_values = list(range(2, class_num+1)) if class_counts is None else list(class_counts)

    for accs, label in pairs:
        plt.plot(x_values, accs, label=label)

    plt.legend()
    plt.xlabel("#classes")
    plt.ylabel("accuracy")
    plt.show()


def _training_history_metric(name: str) -> str | None:
    """Identify the training counterpart of an ordinary or merged validation metric.

    Args:
        name (str): Keras metric key, optionally prefixed by val_, generator_val_,
            or discriminator_val_ as produced by the project's V2 history merger.

    Returns:
        str | None: Matching training key, or None for a nonvalidation metric.

    Raises:
        None.
    """

    # Ordinary Keras validation metrics prepend val_ to their training names.
    if name.startswith("val_"):
        return name[4:]

    for prefix in ("generator_", "discriminator_"):
        # The V2 merger places its phase prefix before the Keras validation prefix.
        if name.startswith(prefix + "val_"):
            return prefix + name[len(prefix) + 4:]

    return None


def _history_metric_epochs(
    history: Mapping[str, Sequence[float]], 
    metric_epochs: Mapping[str, Sequence[int]] | None
) -> dict[str, np.ndarray]:
    """Resolve one-based epoch coordinates without inferring sparse validation cadence.

    Args:
        history (Mapping[str, Sequence[float]]): Metric values ordered by observation.
            Nonvalidation series default to consecutive epochs starting at one.
        metric_epochs (Mapping[str, Sequence[int]] | None): Optional per-metric
            one-dimensional positive, strictly increasing integer-valued coordinates.
            Each supplied sequence must have exactly one entry per recorded value.

    Returns:
        dict[str, np.ndarray]: Int64 epoch vector for every history key. An unspecified
            validation series inherits its training partner's axis only when lengths
            match; empty validation series have empty axes.

    Raises:
        ValueError: If coordinates are malformed, keys are unknown, or sparse/standalone
            validation observations lack explicit epoch coordinates.
    """

    provided = {} if metric_epochs is None else dict(metric_epochs)
    # A misspelled metric key would silently discard the caller's intended alignment.
    if set(provided) - set(history):
        raise ValueError("metric_epochs contains keys absent from history.")

    resolved = {}
    for name, coordinates in provided.items():
        values = np.asarray(coordinates)
        # Epoch coordinates must preserve observation identity and chronological order.
        if (values.shape != tuple([len(history[name])]) or values.dtype.kind not in "iuf"
                or not np.isfinite(values).all() or np.any(values < 1)
                or np.any(values >= 2 ** 63) or np.any(values != np.floor(values))
                or np.any(values[1:] <= values[:-1])):
            raise ValueError("metric_epochs must contain aligned, increasing positive integer epochs.")

        resolved[name] = values.astype(np.int64)

    for name, values in history.items():
        # Dense training observations keep their historical one-based default axis.
        if name not in resolved and _training_history_metric(name) is None:
            resolved[name] = np.arange(1, len(values) + 1, dtype=np.int64)

    for name, values in history.items():
        # Explicit coordinates and resolved training series need no further inference.
        if name in resolved:
            continue

        partner = _training_history_metric(name)

        # An empty validation series has no observation coordinates to invent.
        if not len(values):
            resolved[name] = np.empty(0, dtype=np.int64)
        # Equal-length validation and training observations share the same epoch axis.
        elif partner in resolved and len(values) == len(resolved[partner]):
            resolved[name] = resolved[partner].copy()
        # Sparse or standalone validation cannot be dated from its value count alone.
        else:
            raise ValueError(
                f"Explicit metric_epochs are required for validation metric {name!r}."
            )

    return resolved


def plot_history(
    history: Mapping[str, Sequence[float]], 
    range_: tuple[int | None, ...] = (0, None), 
    metrics: Sequence[str] | None = None, 
    row: int | None = None, 
    col: int = 3, 
    figsize: tuple[float, float] | None = None, 
    x_ticks_rotation: float = 90, 
    y_ticks_rotation: float = 0, 
    show_all_x_ticks: bool = True, 
    y_ticks_num: int | None = None, 
    show_plots: bool = True, 
    plot_path: str | os.PathLike[str] | None = None, 
    csv_path: str | os.PathLike[str] | None = None, 
    metric_epochs: Mapping[str, Sequence[int]] | None = None
) -> None:
    """Plot metric observations at their actual epochs and optionally export aligned CSV.

    Training and matching validation metrics share a subplot. Ordinary val_ keys and
    merged generator_val_/discriminator_val_ keys are recognized. Sparse validation
    requires explicit epoch metadata; its frequency is never inferred from lengths.

    Args:
        history (Mapping[str, Sequence[float]]): Numeric scalar observations ordered
            by epoch for each metric. Dense nonvalidation series default to epochs
            1..N; equal-length validation partners inherit that axis.
        range_ (tuple[int | None, ...]): Two or three slice arguments selecting training
            observation positions, not absolute epoch labels. Dense paired validation
            retains the same slice and step. Sparse paired validation retains points
            within the selected training epoch interval, including when only validation
            is requested. A start beyond a short phase's length retains that phase.
            Standalone validation with explicit coordinates slices its own observations.
            CSV always contains the complete unsliced history.
            Defaults to ``(0, None)``.
        metrics (Sequence[str] | None): Requested metric keys; None selects all.
            Validation partners are overlaid even when omitted from this sequence.
            Defaults to ``None``.
        row (int | None): Subplot rows; None chooses enough rows for all selected metrics.
            Defaults to ``None``.
        col (int): Subplot columns, default 3.
        figsize (tuple[float, float] | None): Figure width/height in inches; None uses
            (20, row * 5).
            Defaults to ``None``.
        x_ticks_rotation (float): X tick label rotation in degrees, default 90.
        y_ticks_rotation (float): Y tick label rotation in degrees, default 0.
        show_all_x_ticks (bool): Label all displayed epochs when at most 50 are shown.
            Defaults to ``True``.
        y_ticks_num (int | None): Optional number of evenly spaced y ticks across the
            displayed values; None preserves Matplotlib's automatic ticks.
            Defaults to ``None``.
        show_plots (bool): Display the figure when true; otherwise close it after saving.
            Defaults to ``True``.
        plot_path (str | os.PathLike | None): Optional figure destination; None skips saving.
            Defaults to ``None``.
        csv_path (str | os.PathLike | None): Optional CSV destination. Its epoch column
            is the sorted union of actual coordinates; unobserved cells remain missing.
            None skips CSV output.
            Defaults to ``None``.
        metric_epochs (Mapping[str, Sequence[int]] | None): Per-metric
            one-dimensional positive, strictly increasing integer-valued epochs with
            one coordinate per value. Supply sparse, irregular, resumed or standalone
            validation coordinates explicitly. None uses no additional per-metric
            coordinates; dense training starts at one and sparse validation still
            requires epoch metadata.
            Defaults to ``None``.

    Returns:
        None: Creates a figure and optional files without changing history or coordinates.

    Raises:
        KeyError: If a requested metric is absent.
        ValueError: If epoch metadata is ambiguous/invalid, the subplot grid is
            insufficient, no metric remains, or a selected reference range is empty.
    """

    import matplotlib


    # Select a noninteractive backend for file-only rendering.
    if not show_plots:
        matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    import pandas as pd


    epochs_by_metric = _history_metric_epochs(history, metric_epochs)
    range_ = slice(*range_)
    # Plot every recorded metric when no subset is supplied.
    if metrics is None:
        metrics = list(history.keys())

    plotted_metrics = []
    for metric in metrics:
        # Avoid plotting the same metric more than once.
        if metric in plotted_metrics:
            continue

        partner = _training_history_metric(metric)
        # Validation selected beside its training counterpart shares that subplot.
        if partner is not None and partner in metrics:
            continue

        plotted_metrics.append(metric)

    # Require at least one history series to plot.
    if not plotted_metrics:
        raise ValueError("At least one history metric is required for plotting.")

    # Infer the minimum row count needed for all selected metrics.
    if row is None:
        row = -(len(plotted_metrics) // -col)

    # Require a positive grid large enough for every metric.
    elif row <= 0 or row * col < len(plotted_metrics):
        raise ValueError("row and col must provide a cell for every metric.")

    # Scale the default figure size with the subplot grid.
    if figsize is None:
        figsize = (20, row * 5)

    fig, axes = plt.subplots(row, col, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for i, metric in enumerate(plotted_metrics):
        ax = axes[i]
        partner = _training_history_metric(metric)
        reference = partner if partner in history else metric
        metric_range = range_
        # Short phase histories retain observations when the global start lies beyond them.
        if range_.start is not None and range_.start >= len(history[reference]):
            metric_range = slice(None)

        reference_epochs = epochs_by_metric[reference][metric_range]
        # An empty reference range cannot define the intended displayed interval.
        if not len(reference_epochs):
            raise ValueError("The selected history range contains no reference epochs.")

        full_epochs = epochs_by_metric[metric]
        # Validation-only views use the same training interval as paired views.
        if reference != metric and not np.array_equal(full_epochs, epochs_by_metric[reference]):
            selected = (full_epochs >= min(reference_epochs)) & (full_epochs <= max(reference_epochs))
            epochs = full_epochs[selected]
            values = np.asarray(history[metric])[selected]
        # Dense series and standalone validation retain positional slicing and its step.
        else:
            epochs = full_epochs[metric_range]
            values = np.asarray(history[metric])[metric_range]

        shown_values = list(values)
        ax.plot(
            epochs, 
            values, 
            label="Validation" if partner is not None else "Training", 
            marker="o" if len(values) == 1 else None
        )
        for val_metric, val_values in history.items():
            # Overlay only nonempty validation partners of this training metric.
            if _training_history_metric(val_metric) != metric or not len(val_values):
                continue

            val_epochs = epochs_by_metric[val_metric]
            # Fully aligned observations retain the training slice, including its step.
            if np.array_equal(val_epochs, full_epochs):
                selected_epochs = val_epochs[metric_range]
                selected_values = np.asarray(val_values)[metric_range]
            # Sparse observations are selected using their actual epoch coordinates.
            else:
                selected = (val_epochs >= min(reference_epochs)) & (val_epochs <= max(reference_epochs))
                selected_epochs = val_epochs[selected]
                selected_values = np.asarray(val_values)[selected]
            shown_values.extend(selected_values)
            ax.plot(
                selected_epochs, selected_values, label="Validation", 
                marker="o" if len(selected_values) == 1 else None
            )

        ax.legend()
        ax.set_xlabel("epochs")
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.tick_params(axis="x", rotation=x_ticks_rotation)
        ax.tick_params(axis="y", rotation=y_ticks_rotation)

        # Label every reference epoch on the x-axis when requested.
        if show_all_x_ticks and len(reference_epochs) <= 50:
            ax.set_xticks(reference_epochs)

        # Empty sparse views contain no measured range from which to derive y ticks.
        if y_ticks_num and shown_values:
            ax.set_yticks(np.linspace(min(shown_values), max(shown_values), y_ticks_num))

    for i in range(len(plotted_metrics), row * col):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Save the rendered history figure when a path is supplied.
    if plot_path:
        fig.savefig(plot_path, dpi=500, bbox_inches="tight")

    # Display the history figure in interactive mode.
    if show_plots:
        plt.show()
    # Release the file-only figure without opening a window.
    else:
        plt.close(fig)

    # Use the same actual epoch coordinates for unsliced CSV and plotted observations.
    if csv_path:
        history_df = pd.DataFrame({
            name: pd.Series(values, index=epochs_by_metric[name], dtype=float)
            for name, values in history.items()
        }).sort_index()
        history_df.index.name = "epoch"
        history_df.to_csv(csv_path, index=True)


def create_gif(
    output_path: str | os.PathLike[str], 
    images1: Iterable[np.ndarray], 
    images2: Iterable[np.ndarray] | None = None, 
    duration: int = 100, 
    loop: int = 0, 
    verbose: bool | int = 1
) -> None:
    """Tile diffusion trajectories and write an animated RGBA GIF.

    Args:
        output_path (str | os.PathLike): Destination including the ``.gif``
            suffix; parent directories must exist.
        images1 (Iterable[numpy.ndarray]): Nonempty frame sequence.  Each frame
            is shaped ``[samples, height, width, channels]`` with display values
            expected in ``[0, 1]``; grayscale and RGB samples are tiled
            horizontally.
        images2 (Iterable[numpy.ndarray] | None): Optional second trajectory.
            Paired frames (using truncating ``zip``) are stacked vertically per
            sample with a 10-pixel white separator before horizontal tiling.
            Defaults to ``None``, rendering only the first trajectory.
        duration (int): Milliseconds per frame passed to Pillow.
            Defaults to ``100``.
        loop (int): GIF repeat count; ``0`` requests infinite looping.
            Defaults to ``0``.
        verbose (bool | int): Print the destination when truthy.
            Defaults to ``1``.

    Returns:
        None.

    Raises:
        ValueError: If the resulting frame sequence is empty or paired frame
            shapes cannot be concatenated.
        OSError: If Pillow cannot write the destination.
    """

    from PIL import Image


    # Animate the single supplied image sequence by itself.
    if images2 is None:
        images = images1
    # Concatenate paired sequences horizontally for comparison.
    else:
        images = []
        for image1, image2 in zip(images1, images2):
            images.append(
                np.concatenate([
                    image1, 
                    np.ones((
                        image1.shape[0], 
                        10, 
                        image1.shape[2], 
                        image1.shape[3]
                    )), 
                    image2
                ], axis=1)
            )

    frames = []
    for image in images:
        image = np.asarray(image)
        # Require nonempty rank-four image sequences with displayable channels.
        if image.ndim != 4 or image.shape[0] == 0 \
        or image.shape[-1] not in (1, 3, 4):
            raise ValueError(
                "Every GIF frame must be a nonempty [samples, H, W, C] array "
                "with 1, 3, or 4 channels."
            )
        # Remove the singleton channel expected by grayscale rendering.
        if image.shape[-1] == 1:
            image = image[..., 0]

        image = (np.clip(image, 0., 1.) * 255).astype("uint8")
        image = np.concatenate(image, axis=1)
        image = Image.fromarray(image)

        frames.append(image.convert("RGBA"))

    # Reject an empty frame sequence before selecting the first GIF frame.
    if not frames:
        raise ValueError("At least one GIF frame is required.")

    frames[0].save(
        output_path, 
        save_all=True, 
        append_images=frames[1:], 
        duration=duration, 
        loop=loop
    )

    # Report the written GIF path when requested.
    if verbose:
        print(f"GIF saved to '{output_path}'.")


def show_img(
    x: object, 
    y: Sequence[object] | None = None
) -> None:
    """Display one image without axes and optionally add a label title.

    Args:
        x (numpy.ndarray | tf.Tensor): Matplotlib-compatible image shaped
            ``[height, width]`` or ``[height, width, channels]``.
        y (Sequence[object] | None): Optional indexable label container.
            ``y[0]`` is displayed when present and non-``None``.
            Defaults to ``None``.

    Returns:
        None: The image is shown interactively.

    Raises:
        TypeError: If Matplotlib cannot display the image shape or numeric dtype.
    """

    from matplotlib import pyplot as plt


    plt.imshow(x)
    plt.axis("off")

    # Add a label title when the caller supplied one.
    if y is not None and len(y) > 0 and y[0] is not None:
        plt.title(f"Label: {y[0]}")

    plt.show()


def plot_images(
    imgs: np.ndarray, 
    row: int = 1, 
    col: int = 11, 
    has_null_label: bool = False, 
    show_images: bool = True, 
    save_path: str | os.PathLike[str] | None = None, 
    titles: Sequence[str] | None = None
) -> None:
    """Display or save a grayscale batch as a labeled subplot grid.

    Args:
        imgs (numpy.ndarray): Images shaped
            ``[samples, height, width, channels]`` with one, three, or four
            channels.
        row (int): Positive minimum subplot row count. Additional rows are
            added when needed to fit every image.
            Defaults to ``1``.
        col (int): Positive maximum number of subplot columns.
            Defaults to ``11``.
        has_null_label (bool): Whether the first image is a null-condition
            preview. Titles then start at -1, followed by zero-based indices
            for the remaining images. Defaults to ``False``.
        show_images (bool): Display the figure interactively.
            Defaults to ``True``.
        save_path (str | os.PathLike | None): Optional image destination.  At
            least one of ``show_images`` or ``save_path`` must be enabled.
            Defaults to ``None``, skipping figure-file output.
        titles (Sequence[str] | None): Optional title for each image, overriding
            sample-index titles. Defaults to ``None``.

    Returns:
        None.

    Raises:
        ValueError: If neither display nor saving is requested or ``imgs`` has
            an invalid shape or the title count differs from the image count.

    Note:
        Default titles are sample indices starting at 0, or -1 with ``has_null_label``;
        class IDs are not inferred from image content or sampling labels.
    """

    import matplotlib


    # Select a noninteractive backend for file-only rendering.
    if not show_images:
        matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt


    # Require either a visible display or an output file.
    if not show_images and save_path is None:
        raise ValueError("Enable image display or provide save_path.")
    imgs = np.asarray(imgs)
    # Require a nonempty rank-four batch with displayable channels.
    if imgs.ndim != 4 or len(imgs) == 0 or imgs.shape[-1] not in (1, 3, 4):
        raise ValueError(
            "imgs must be a nonempty [samples, H, W, C] array with "
            "1, 3, or 4 channels."
        )

    # Custom labels must identify every image in the grid.
    if titles is not None and len(titles) != len(imgs):
        raise ValueError("titles must contain one title per image.")

    col = min(col, len(imgs))
    row = max(row, -(len(imgs) // -col))
    fig, axes = plt.subplots(row, col, figsize=(20, max(6, 2 * row)))
    axes = np.atleast_1d(axes).ravel()

    for i in range(len(imgs)):
        # Remove a singleton grayscale channel; preserve RGB/RGBA channels.
        image = imgs[i, :, :, 0] if imgs.shape[-1] == 1 else imgs[i]
        # Use a grayscale colormap for one-channel images and native colors otherwise.
        axes[i].imshow(image, cmap="gray" if imgs.shape[-1] == 1 else None)
        axes[i].set_title(
            titles[i] if titles is not None else f"{i - int(has_null_label)}"
        )
        axes[i].axis("off")

    for j in range(len(imgs), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    # Save the image grid when an output path is supplied.
    if save_path:
        fig.savefig(
            save_path, 
            dpi=200, 
            bbox_inches="tight"
        )

    # Display the image grid in interactive mode.
    if show_images:
        plt.show()
    # Release the file-only figure without opening a window.
    else:
        plt.close(fig)


def plot_noisy_images(
    model: object, 
    imgs: object, 
    interval: int = 10, 
    show_images: bool = True, 
    save_path: str | os.PathLike[str] | None = None, 
    col: int = 10, 
    seed: int = 42
) -> None:
    """Plot one image through diffusion, reading left to right and top to bottom.

    Every panel uses the same source image and Gaussian draw. The first panel
    in reading order is t=0 and the last is t=timesteps-1. With 1000 timesteps,
    interval=10, and col=10, the grid has 100 panels: t=0 at the top left and
    t=999 at the bottom right. Including both endpoints gives gaps of 10 or 11
    timesteps. Timestep zero follows the schedule and need not be clean.

    Args:
        model (DiffusionModel): Wrapper owning the schedule and pixel preprocessing.
        imgs (numpy.ndarray | tf.Tensor): A single raw ``[1, H, W, C]`` image
            in ``[0,255]``, with 1, 3, or 4 channels.
        interval (int): Positive nominal spacing. The panel count is
            ``max(2, ceil(timesteps / interval))``; integer timesteps are
            evenly spaced over the full inclusive range, retaining both ends.
            Defaults to ``10``.
        show_images (bool): Display the grid; defaults to ``True``.
        save_path (str | os.PathLike | None): Optional grid image destination;
            defaults to ``None``. Required when ``show_images`` is false.
        col (int): Positive maximum columns per row; defaults to 10.
            A partially filled final row occupies its leftmost cells.
        seed (int): Stateless Gaussian seed, default 42. Repeated calls reuse
            the same noise without advancing a training model's random stream.

    Returns:
        None: The existing image grid helper displays or saves the result.

    Raises:
        ValueError: The schedule, interval, column count, image, or output is invalid.
        TypeError: Timestep count, interval, or column count is not an integer.
    """

    import tensorflow as tf


    # Reject empty or backwards timestep selections before generating noise.
    if interval <= 0:
        raise ValueError("interval must be a positive integer.")
    # Require a usable row width for the timestep grid.
    if col <= 0:
        raise ValueError("col must be a positive integer.")

    timesteps = model.timesteps
    steps = range(0, timesteps, interval)
    imgs = np.asarray(imgs)

    # Each grid follows exactly one source image through the entire schedule.
    if imgs.ndim != 4 or len(imgs) != 1 or imgs.shape[-1] not in (1, 3, 4):
        raise ValueError(
            "imgs must contain one [1, H, W, C] image with 1, 3, or 4 channels."
        )

    steps = np.linspace(0, timesteps - 1, max(2, len(steps)), dtype=np.int32)
    images = model.preprocess(imgs)
    noise = tf.random.stateless_normal(tf.shape(images), seed=[seed, 0], dtype=images.dtype)
    noisy_images = model.q_sample(
        tf.repeat(images, len(steps), axis=0), 
        tf.constant(steps, dtype=tf.int32), 
        tf.repeat(noise, len(steps), axis=0)
    )
    plot_images(
        model.preprocess(model.postprocess(noisy_images, clip=True), "min-max").numpy(), 
        col=col, 
        show_images=show_images, 
        save_path=save_path, 
        titles=[f"t={step}" for step in steps]
    )


def save_samples(
    arr: object, 
    path: str | os.PathLike[str], 
    type_: str
) -> None:
    """Save a NumPy-compatible array as CSV or NPY using a base path.

    Args:
        arr (numpy.ndarray | array-like): Values to persist. CSV generally
            requires a one- or two-dimensional numeric array. A numeric NPY
            array is saved normally; a one-dimensional object array is treated
            as an ordered heterogeneous bundle whose members must each convert
            to a non-object NumPy array. Such bundles use safe NPZ container
            content while retaining the public ``.npy`` filename convention.
        path (str | os.PathLike): Base path without an extension.
        type_ (str): Exactly ``".csv"`` or ``".npy"``.

    Returns:
        None.

    Raises:
        ValueError: If ``type_`` is unsupported or an object bundle is not a
            one-dimensional sequence of non-object arrays.
        OSError: If the destination cannot be opened or written.
    """

    path = os.fspath(path)

    # Serialize tabular values as comma-separated text.
    if type_ == ".csv":
        np.savetxt(path+type_, arr, delimiter=',')
    # Serialize numeric arrays or safe heterogeneous numeric bundles.
    elif type_ == ".npy":
        array = np.asarray(arr)
        members = None

        # Validate pickle-backed inputs before opening/truncating the destination.
        if array.dtype.hasobject:
            # One outer axis is required to preserve member boundaries safely.
            if array.ndim != 1:
                raise ValueError(
                    "Object sample bundles must be one-dimensional."
                )

            members = [np.asarray(member) for member in array]

            # No member may smuggle a pickle-backed object payload.
            if any(member.dtype.hasobject for member in members):
                raise ValueError(
                    "Object sample bundle members must have non-object dtypes."
                )

        with open(path+type_, "wb") as file:
            # Replace pickle-backed object arrays with an ordered ZIP container.
            if members is not None:
                np.savez(file, *members)
            # Ordinary homogeneous arrays retain the original NPY encoding.
            else:
                np.save(file, array, allow_pickle=False)
    # Reject artifact formats outside CSV and NPY.
    else:
        raise ValueError("type_ must be '.csv' or '.npy'.")


def load_samples(
    path: str | os.PathLike[str], 
    type_: str, 
    allow_pickle: bool = False
) -> np.ndarray:
    """Load a CSV or NPY sample archive from a base path.

    Args:
        path (str | os.PathLike): Base path without an extension.
        type_ (str): ``".csv"`` loads comma-delimited numeric data;
            ``".npy"`` loads NumPy binary data.
        allow_pickle (bool): NPY trust policy. The safe default ``False`` rejects
            legacy object arrays. ``True`` explicitly permits a trusted legacy
            pickle payload and emits ``RuntimeWarning`` with a migration path.
            New heterogeneous bundles are ordered NPZ containers whose members
            are always loaded with pickle disabled. CSV loading ignores this
            option after its type has been validated.

    Returns:
        numpy.ndarray: Loaded values.

    Raises:
        ValueError: If ``type_`` is unsupported, CSV contents are invalid, or a
            strict NPY load encounters an object array, or a safe bundle has
            malformed keys/object-valued members.
        OSError: If the selected path cannot be opened.

    Warns:
        RuntimeWarning: If explicit compatibility mode enables pickle for a
            trusted legacy object-array archive. Re-save the returned array with
            :func:`save_samples` to migrate it to the safe container format.
    """

    path = os.fspath(path)

    # Parse comma-separated numeric values.
    if type_ == ".csv":
        arr = np.loadtxt(path + type_, delimiter=",")
    # Restore a binary NumPy array under the selected trust policy.
    elif type_ == ".npy":
        with open(path+type_, "rb") as file:
            magic = file.read(4)
            file.seek(0)
            is_safe_bundle = magic == b"PK\x03\x04"
            # Make every legacy pickle load an explicit, visible trust decision.
            if allow_pickle and not is_safe_bundle:
                warnings.warn(
                    "allow_pickle=True can execute code from this archive. "
                    "Only load trusted legacy files, then migrate by calling "
                    "save_samples(loaded, new_path, '.npy').", 
                    RuntimeWarning, 
                    stacklevel=2
                )

            try:
                # Always disable pickle for safe bundles; use the explicit trust flag for legacy NPY.
                loaded = np.load(
                    file, 
                    allow_pickle=False if is_safe_bundle else allow_pickle
                )
            except ValueError as error:
                is_legacy_object_archive = (
                    "Object arrays" in str(error)
                    and "allow_pickle=False" in str(error)
                )
                # Explain the safe migration path without silently executing code.
                if is_legacy_object_archive:
                    raise ValueError(
                        "Legacy object-array NPY loading is disabled. If and "
                        "only if the file is trusted, retry with "
                        "allow_pickle=True and re-save it with save_samples."
                    ) from error
                raise

            # Safe heterogeneous bundles use ordered default NPZ member names.
            if isinstance(loaded, np.lib.npyio.NpzFile):
                try:
                    expected_keys = [
                        f"arr_{index}" for index in range(len(loaded.files))
                    ]
                    # Reject missing, reordered, duplicated, or injected members.
                    if loaded.files != expected_keys:
                        raise ValueError(
                            "Sample bundle members must be ordered arr_0..arr_n."
                        )
                    try:
                        members = [loaded[key] for key in expected_keys]
                    except ValueError as error:
                        # Convert NumPy's lazy object-member error into our contract.
                        if "allow_pickle=False" in str(error):
                            raise ValueError(
                                "Sample bundle members must have non-object dtypes."
                            ) from error
                        raise
                    # Container members remain non-pickled even under legacy mode.
                    if any(member.dtype.hasobject for member in members):
                        raise ValueError(
                            "Sample bundle members must have non-object dtypes."
                        )
                    arr = np.empty(len(members), dtype=object)
                    arr[:] = members
                finally:
                    loaded.close()
            # Ordinary numeric and explicitly trusted legacy NPY arrays pass through.
            else:
                arr = loaded
    # Reject artifact formats outside CSV and NPY.
    else:
        raise ValueError("type_ must be '.csv' or '.npy'.")

    return arr


def save_logs(
    model_name: str, 
    i: int, 
    search_space: Sequence[object] | None = None, 
    names: Sequence[str] | None = None, 
    metrics: Mapping[str, object] | None = None, 
    where_to: str = "file"
) -> None:
    """Format one hyperparameter-search record and write and/or print it.

    Args:
        model_name (str): Filename stem under ``./files/models/hyperas/logs``.
        i (int): Optimization iteration number included when both search-space
            values and names are nonempty.
        search_space (Sequence[object] | None): Selected hyperparameter values.
            Defaults to ``None``, treated as an empty sequence and omitting
            the parameter/iteration block.
        names (Sequence[str] | None): Corresponding names. ``zip`` silently
            truncates to the shorter sequence. Defaults to ``None``, treated
            as an empty sequence and omitting the parameter/iteration block.
        metrics (Mapping[str, object] | None): Metric names and printable values.
            Defaults to ``None``, treated as an empty mapping; ``None`` or an
            empty mapping omits the metric line.
        where_to (str): ``"file"`` appends to disk, ``"print"`` writes to
            stdout, and ``"both"`` does both.  Other values perform neither.
            Defaults to ``'file'``.

    Returns:
        None.

    Raises:
        OSError: If file output is selected and the fixed log directory does
            not exist or is not writable.
    """

    # Normalize an omitted search space to an empty sequence.
    search_space = () if search_space is None else search_space
    # Normalize omitted parameter names to an empty sequence.
    names = () if names is None else names
    # Normalize omitted metrics to an empty mapping.
    metrics = {} if metrics is None else metrics
    txt = ""

    # Format named search-space values for the log message.
    if search_space and names:
        txt += f"----Optimization Iteration {i}:\n"
        for ss, name in zip(search_space, names):
            txt += f"{name}: {ss}\n"

    # Append reported metrics when present.
    if metrics:
        txt += "----("
        for metric_name, metric_value in metrics.items():
            txt += f"{metric_name}={metric_value}, "

        txt = txt[:-2] + ")\n\n"

    # Append the message to the configured log file when requested.
    if where_to == "file" or where_to == "both":
        with open(f"./files/models/hyperas/logs/{model_name}.txt", "at") as f:
            f.write(txt)

    # Print the message to standard output when requested.
    if where_to == "print" or where_to == "both":
        print(txt)
