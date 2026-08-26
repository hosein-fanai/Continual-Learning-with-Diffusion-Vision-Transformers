"""Plotting, feature extraction, sample persistence, and experiment helpers."""

from __future__ import annotations

import os

import numpy as np

from collections.abc import Iterable, Mapping, Sequence


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
    file_name: str | os.PathLike[str] | None = None
) -> list[np.ndarray]:
    """Extract 2,048-wide Xception features for multiple sample arrays.

    Args:
        dataset_list (Iterable[numpy.ndarray | tf.Tensor]): Image arrays, each
            normally shaped ``[samples, 32, 32, 3]``.  Every array is passed to
            a frozen resize/preprocess/Xception/global-pooling model.
        batch_size (int): Positive prediction batch size; defaults to 128.
        file_name (str | os.PathLike | None): Optional base path without
            ``.npy``.  When supplied, an object array containing all feature
            arrays is saved via :func:`save_samples`.

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
        save_samples(np.array(features_list, dtype="object"), 
                    file_name, ".npy")

    return features_list


def CL_plot(
    class_num: int, 
    pairs: Iterable[tuple[Sequence[float], str]]
) -> None:
    """Plot continual-learning accuracy against the number of seen classes.

    Args:
        class_num (int): Final class count.  The x-axis is integer values from
            2 through this value inclusive.
        pairs (Iterable[tuple[Sequence[float], str]]): Accuracy series and
            legend labels.  Each series should contain ``class_num - 1``
            values; multiple pairs create comparison curves.

    Returns:
        None: Matplotlib displays the figure interactively.

    Raises:
        ValueError: If an accuracy series length differs from the x-axis length.
    """

    from matplotlib import pyplot as plt


    for accs, label in pairs:
        plt.plot(list(range(2, class_num+1)), accs, label=label)

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
            metric.replace("val_", '') in metrics:
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
        arr (numpy.ndarray | array-like): Values to persist.  CSV generally
            requires a one- or two-dimensional numeric array; NPY supports
            arbitrary shapes and object arrays.
        path (str | os.PathLike): Base path without an extension.
        type_ (str): Exactly ``".csv"`` or ``".npy"``.

    Returns:
        None.

    Raises:
        ValueError: If ``type_`` is unsupported.
        OSError: If the destination cannot be opened or written.
    """

    path = os.fspath(path)
    # Serialize tabular values as comma-separated text.
    if type_ == ".csv":
        np.savetxt(path+type_, arr, delimiter=',')
    # Serialize arbitrary NumPy arrays in binary format.
    elif type_ == ".npy":
        with open(path+type_, "wb") as file:
            np.save(file ,arr)
    # Reject artifact formats outside CSV and NPY.
    else:
        raise ValueError("type_ must be '.csv' or '.npy'.")


def load_samples(
    path: str | os.PathLike[str], 
    type_: str
) -> np.ndarray:
    """Load a CSV or NPY sample archive from a base path.

    Args:
        path (str | os.PathLike): Base path without an extension.
        type_ (str): ``".csv"`` loads comma-delimited numeric data;
            ``".npy"`` loads with ``allow_pickle=True``.

    Returns:
        numpy.ndarray: Loaded values.

    Raises:
        ValueError: If ``type_`` is unsupported or CSV contents are invalid.
        OSError: If the selected path cannot be opened.
    """

    path = os.fspath(path)
    # Parse comma-separated numeric values.
    if type_ == ".csv":
        arr = np.loadtxt(path + type_, delimiter=",")
    # Restore a binary NumPy array without pickle objects.
    elif type_ == ".npy":
        with open(path+type_, "rb") as file:
            arr = np.load(file, allow_pickle=True)
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
