"""Plotting, feature extraction, sample persistence, and experiment helpers."""

from __future__ import annotations

import numpy as np

import os

import json

import warnings

from pathlib import Path

from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral, Real


models_path = "./models"
hyperas_path = os.path.join(models_path, "hyperas")

best_score = -float("inf")
i = 1


def init() -> None:
    """Limit the first visible TensorFlow GPU to 6,144 MiB of logical memory.

    The setting is applied only when at least one physical GPU is visible and
    must run before TensorFlow initializes that device.  CPU-only execution is
    left unchanged.  A late configuration attempt prints TensorFlow's
    ``RuntimeError`` and a short warning.

    Returns:
        None.
    """

    import tensorflow as tf


    # Enable incremental allocation only when a GPU is available.
    if gpus:=tf.config.list_physical_devices("GPU"):
        try:
            tf.config.set_logical_device_configuration(gpus[0], [
                tf.config.LogicalDeviceConfiguration(memory_limit=6144)
            ])
        except RuntimeError as e:
            print(e)
            print("Could not limit gpu memory.")


def extract_features(
    dataset_list: Iterable[object], 
    batch_size: int = 128, 
    file_name: str | os.PathLike[str] | None = None,
    split_seed: int = 42,
    validation_ratio: float = 0.2,
) -> list[np.ndarray]:
    """Extract 2,048-wide Xception features for multiple sample arrays.

    Args:
        dataset_list (Iterable[numpy.ndarray | tf.Tensor]): Image arrays, each
            normally shaped ``[samples, 32, 32, 3]``.  Every array is passed to
            a frozen resize/preprocess/Xception/global-pooling model.
        batch_size (int): Positive prediction batch size; defaults to 128.
        file_name (str | os.PathLike | None): Optional base path without
            ``.npy``. When supplied, the feature arrays are saved as an ordered,
            non-pickled NumPy container via :func:`save_samples`.
        split_seed (int): Seed that produced the label-aligned train/validation
            feature split. It is saved beside three-array archives so loading
            can reconstruct the identical label ordering.
        validation_ratio (float): Fraction assigned to the saved validation
            feature array. It is stored with ``split_seed``.

    Returns:
        list[numpy.ndarray]: One floating feature array per input dataset,
        normally shaped ``[samples, 2048]``.

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


def _normalize_feature_split_metadata(
    split_seed: int, 
    validation_ratio: float
) -> tuple[int, float]:
    """Validate the reproducible label split stored with feature archives.

    Args:
        split_seed (int): Non-boolean NumPy-compatible random seed.
        validation_ratio (float): Fraction assigned to validation features.

    Returns:
        tuple[int, float]: Normalized Python seed and validation fraction.

    Raises:
        TypeError: If either value has an unsupported type.
        ValueError: If the seed or ratio lies outside its valid interval.
    """

    # Require an integral seed shared by NumPy and scikit-learn.
    if isinstance(split_seed, bool) or not isinstance(split_seed, Integral):
        raise TypeError("split_seed must be a non-boolean integer.")
    split_seed = int(split_seed)
    # Restrict the seed to the unsigned interval NumPy accepts.
    if not 0 <= split_seed < 2 ** 32:
        raise ValueError("split_seed must be in [0, 2**32).")
    # Require a real validation fraction rather than a truthy flag.
    if isinstance(validation_ratio, bool) \
    or not isinstance(validation_ratio, Real):
        raise TypeError("validation_ratio must be a non-boolean real number.")
    validation_ratio = float(validation_ratio)
    # Both saved training and validation partitions must remain nonempty.
    if not np.isfinite(validation_ratio) \
    or not 0. < validation_ratio < 1.:
        raise ValueError("validation_ratio must lie strictly between 0 and 1.")

    return split_seed, validation_ratio


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

    Returns:
        pathlib.Path: Written ``.metadata.json`` sidecar path.
    """

    split_seed, validation_ratio = _normalize_feature_split_metadata(
        split_seed, 
        validation_ratio
    )
    metadata_path = Path(os.fspath(path) + ".metadata.json")
    payload = {
        "format_version": 1, 
        "label_split": {
            "random_state": split_seed, 
            "validation_ratio": validation_ratio
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

    return _normalize_feature_split_metadata(
        label_split["random_state"], 
        label_split["validation_ratio"]
    )


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

    Returns:
        None: Matplotlib displays the figure interactively.

    Raises:
        ValueError: If an accuracy series length differs from the x-axis length.
    """

    from matplotlib import pyplot as plt


    x_values = list(range(2, class_num+1)) if class_counts is None \
            else list(class_counts)

    for accs, label in pairs:
        plt.plot(x_values, accs, label=label)

    plt.legend()
    plt.xlabel("#classes")
    plt.ylabel("accuracy")
    plt.show()


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
    csv_path: str | os.PathLike[str] | None = None
) -> None:
    """Plot Keras epoch metrics, optionally saving the figure and raw CSV.

    A training metric such as ``"loss"`` and its ``"val_loss"`` counterpart
    share one subplot.  If both names occur in ``metrics``, the explicit
    validation entry is skipped to avoid a duplicate subplot.

    Args:
        history (Mapping[str, Sequence[float]]): Metric names mapped to
            per-epoch values, typically ``History.history``. Series may have
            different lengths, but each training/validation pair must align.
        range_ (tuple[int | None, ...]): Two or three arguments expanded into
            ``slice(*range_)``.  ``(0, None)`` plots all epochs, ``(5, 20)``
            plots zero-based entries 5--19, and ``(None, None, 2)`` plots every
            other epoch.  CSV output is not sliced.
        metrics (Sequence[str] | None): Keys to plot; ``None`` considers every
            history key.
        row (int | None): Positive subplot rows. ``None`` uses
            ``ceil(number_of_plots / col)``; an explicit value must provide
            enough cells for every requested metric.
        col (int): Positive subplot columns; defaults to 3.
        figsize (tuple[float, float] | None): Matplotlib figure size in inches.
            ``None`` uses ``(20, row * 5)``.
        x_ticks_rotation (float): Epoch tick-label rotation in degrees.
        y_ticks_rotation (float): Value tick-label rotation in degrees.
        show_all_x_ticks (bool): Show one tick per epoch unless a plotted range
            contains more than 50 epochs.
        y_ticks_num (int | None): If truthy, place this many evenly spaced ticks
            between the combined training/validation minimum and maximum.
        show_plots (bool): Display the figure; false closes it after saving.
        plot_path (str | os.PathLike | None): Optional image destination.
        csv_path (str | os.PathLike | None): Optional CSV destination containing
            an added one-based ``epoch`` column and all unsliced history keys;
            shorter series are padded with missing values.

    Returns:
        None.

    Raises:
        KeyError: If a requested metric is absent.
        ValueError: If a grid size is invalid or insufficient, no metric
            remains to plot, a selected metric range is empty, or metric
            lengths are incompatible.
    """

    import matplotlib
    # Select a noninteractive backend for file-only rendering.
    if not show_plots:
        matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    import pandas as pd


    range_ = slice(*range_)

    # Plot every recorded metric when no subset is supplied.
    if metrics is None:
        metrics = list(history.keys())
    # Require at least one subplot column.
    if col <= 0:
        raise ValueError("col must be positive.")

    plotted_metrics = []
    for metric in metrics:
        # Avoid plotting the same metric more than once.
        if metric in plotted_metrics:
            continue

        # Pair validation series with their requested training metric.
        if metric.startswith("val_") and \
            metric[4:] in metrics:
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
        figsize = (20, row*5)

    fig, axes = plt.subplots(row, col, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for i, metric in enumerate(plotted_metrics):
        # Fail clearly when a requested training series is absent.
        if metric not in history:
            raise KeyError(f"History has no metric named {metric!r}.")
        ax = axes[i]
        epochs = range(1, len(history[metric])+1)[range_]
        show_metric_x_ticks = show_all_x_ticks and len(epochs) <= 50

        values = np.asarray(history[metric])[range_]
        # Reject empty metric histories with no plottable epochs.
        if len(values) == 0:
            raise ValueError(f"Metric {metric!r} has no values in the selected range.")
        min_ = min(values)
        max_ = max(values)

        ax.plot(
            epochs, 
            values, 
            label="Training" if not metric.startswith("val_") else "Validation"
        )

        val_values = history.get("val_" + metric)
        # Align and plot a nonempty validation series when available.
        if val_values is not None and len(val_values) > 0:
            selected_val_values = np.asarray(val_values)[range_]
            # Keep training and validation epoch coordinates aligned.
            if len(selected_val_values) != len(epochs):
                raise ValueError(
                    f"Training and validation lengths differ for {metric!r}."
                )
            min_ = min([min_, *selected_val_values])
            max_ = max([max_, *selected_val_values])

            ax.plot(epochs, selected_val_values, label="Validation")

        ax.legend()
        ax.set_xlabel("epochs")
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.tick_params(axis='x', rotation=x_ticks_rotation)
        ax.tick_params(axis='y', rotation=y_ticks_rotation)

        # Label every epoch on the x-axis when requested.
        if show_metric_x_ticks:
            ax.set_xticks(epochs)

        # Replace automatic y ticks with the requested uniform count.
        if y_ticks_num:
            ax.set_yticks(np.linspace(min_, max_, y_ticks_num))    

    for i in range(len(plotted_metrics), row*col):
        axes[i].set_visible(False)
    
    plt.tight_layout()

    # Save the rendered history figure when a path is supplied.
    if plot_path:
        fig.savefig(
            plot_path, 
            dpi=200,
            bbox_inches="tight"
        )

    # Display the history figure in interactive mode.
    if show_plots:
        plt.show()
    # Release the file-only figure without opening a window.
    else:
        plt.close(fig)

    # Export aligned history series when a CSV path is supplied.
    if csv_path:
        # Preserve unequal generator/discriminator history lengths with NaN.
        history_df = pd.DataFrame({
            name: pd.Series(values) for name, values in history.items()
        })
        history_df.insert(0, "epoch", range(1, len(history_df) + 1))
        history_df.to_csv(csv_path, index=False)


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
        duration (int): Milliseconds per frame passed to Pillow.
        loop (int): GIF repeat count; ``0`` requests infinite looping.
        verbose (bool | int): Print the destination when truthy.

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

    # Reject empty sequences before GIF encoding.
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


def show_img(x: object, y: Sequence[object] | None = None) -> None:
    """Display one image without axes and optionally add a label title.

    Args:
        x (numpy.ndarray | tf.Tensor): Matplotlib-compatible image shaped
            ``[height, width]`` or ``[height, width, channels]``.
        y (Sequence[object] | None): Optional indexable label container.
            ``y[0]`` is displayed when present and non-``None``.

    Returns:
        None: The image is shown interactively.

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
    show_images: bool = True, 
    save_path: str | os.PathLike[str] | None = None
) -> None:
    """Display or save a grayscale batch as a labeled subplot grid.

    Args:
        imgs (numpy.ndarray): Images shaped
            ``[samples, height, width, channels]`` with one, three, or four
            channels.
        row (int): Positive minimum subplot row count. Additional rows are
            added when needed to fit every image.
        col (int): Positive maximum number of subplot columns.
        show_images (bool): Display the figure interactively.
        save_path (str | os.PathLike | None): Optional image destination.  At
            least one of ``show_images`` or ``save_path`` must be enabled.

    Returns:
        None.

    Raises:
        ValueError: If neither display nor saving is requested, a grid
            dimension is nonpositive, or ``imgs`` has an invalid shape.

    Note:
        Titles are zero-based sample indices, not inferred class labels.
    """

    import matplotlib
    # Select a noninteractive backend for file-only rendering.
    if not show_images:
        matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt


    # Require either a visible display or an output file.
    if not show_images and save_path is None:
        raise ValueError("Enable image display or provide save_path.")
    # Require a positive image-grid shape.
    if row <= 0 or col <= 0:
        raise ValueError("row and col must be positive.")
    imgs = np.asarray(imgs)
    # Require a nonempty rank-four batch with displayable channels.
    if imgs.ndim != 4 or len(imgs) == 0 or imgs.shape[-1] not in (1, 3, 4):
        raise ValueError(
            "imgs must be a nonempty [samples, H, W, C] array with "
            "1, 3, or 4 channels."
        )

    col = min(col, len(imgs))
    row = max(row, -(len(imgs) // -col))
    fig, axes = plt.subplots(row, col, figsize=(20, 6))
    axes = np.atleast_1d(axes).ravel()

    for i in range(len(imgs)):
        image = imgs[i, :, :, 0] if imgs.shape[-1] == 1 else imgs[i]
        axes[i].imshow(image, cmap="gray" if imgs.shape[-1] == 1 else None)
        axes[i].set_title(f"{i}")
        axes[i].axis("off")

    for j in range(len(imgs), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    # Save the image grid when an output path is supplied.
    if save_path:
        fig.savefig(
            save_path, 
            dpi=200,
            bbox_inches="tight", 
        )

    # Display the image grid in interactive mode.
    if show_images:
        plt.show()
    # Release the file-only figure without opening a window.
    else:
        plt.close(fig)


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
        TypeError: If ``allow_pickle`` is not boolean.
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

    # Keep the trust decision explicit and reject truthy stand-ins such as 1.
    if not isinstance(allow_pickle, bool):
        raise TypeError("allow_pickle must be boolean.")

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
        model_name (str): Filename stem under ``./models/hyperas/logs``.
        i (int): Optimization iteration number included when both search-space
            values and names are nonempty.
        search_space (Sequence[object]): Selected hyperparameter values.
        names (Sequence[str]): Corresponding names.  ``zip`` silently truncates
            to the shorter sequence.
        metrics (Mapping[str, object]): Metric names and printable values.  An
            empty mapping omits the metric line.
        where_to (str): ``"file"`` appends to disk, ``"print"`` writes to
            stdout, and ``"both"`` does both.  Other values perform neither.

    Returns:
        None.

    Raises:
        OSError: If file output is selected and the fixed log directory does
            not exist or is not writable.
    """

    search_space = () if search_space is None else search_space
    names = () if names is None else names
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
        with open(f"./models/hyperas/logs/{model_name}.txt", "at") as f: 
            f.write(txt)

    # Print the message to standard output when requested.
    if where_to == "print" or where_to == "both":
        print(txt)
