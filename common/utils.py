"""Plotting, feature extraction, sample persistence, and experiment helpers."""

import os


models_path = "./models"
hyperas_path = os.path.join(models_path, "hyperas")

best_score = -float("inf")
i = 1


def init():
    """Limit the first visible TensorFlow GPU to 6,144 MiB of logical memory.

    The setting is applied only when at least one physical GPU is visible and
    must run before TensorFlow initializes that device.  CPU-only execution is
    left unchanged.  A late configuration attempt prints TensorFlow's
    ``RuntimeError`` and a short warning.

    Returns:
        None.
    """
    import tensorflow as tf


    if gpus:=tf.config.list_physical_devices("GPU"):
        try:
            tf.config.set_logical_device_configuration(gpus[0], [
                tf.config.LogicalDeviceConfiguration(memory_limit=6144)
            ])
        except RuntimeError as e:
            print(e)
            print("Could not limit gpu memory.")


def extract_features(dataset_list, batch_size=128, 
                    file_name=None):
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

    Raises:
        NameError: In the current module unless ``get_model`` has been injected
            into its globals; the function references that factory without
            importing it.  Importing ``common.model.get_model`` into the caller
            namespace alone does not resolve this module-global name.
    """
    from tensorflow.keras import models

    import numpy as np


    conv_base = models.Sequential(
        get_model(10, model_type="pretrained", verbose=0).layers[:4]
    )
    conv_base.trainable = False

    features_list = []
    for dataset in dataset_list:
        features = conv_base.predict(dataset, batch_size=batch_size)
        features_list.append(features)

    del conv_base

    if file_name is not None:
        save_samples(np.array(features_list, dtype="object"), 
                    file_name, ".npy")

    return features_list


def CL_plot(class_num, pairs):
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
    history, 
    range_=(0, None), 
    metrics=None, 
    row=None, col=3, 
    figsize=None, 
    x_ticks_rotation=90, 
    y_ticks_rotation=0, 
    show_all_x_ticks=True, 
    y_ticks_num=None, 
    show_plots=True, 
    plot_path=None, 
    csv_path=None
):
    """Plot Keras epoch metrics, optionally saving the figure and raw CSV.

    A training metric such as ``"loss"`` and its ``"val_loss"`` counterpart
    share one subplot.  If both names occur in ``metrics``, the explicit
    validation entry is skipped to avoid a duplicate subplot.

    Args:
        history (Mapping[str, Sequence[float]]): Metric names mapped to
            per-epoch values, typically ``History.history``.  Series should
            have equal, nonzero lengths and validation values should be lists.
        range_ (tuple[int | None, ...]): Two or three arguments expanded into
            ``slice(*range_)``.  ``(0, None)`` plots all epochs, ``(5, 20)``
            plots zero-based entries 5--19, and ``(None, None, 2)`` plots every
            other epoch.  CSV output is not sliced.
        metrics (Sequence[str] | None): Keys to plot; ``None`` considers every
            history key.
        row (int | None): Subplot rows.  ``None`` uses
            ``ceil(number_of_plots / col)``.
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
            an added one-based ``epoch`` column and all unsliced history keys.

    Returns:
        None.

    Raises:
        TypeError: If a requested metric is absent or cannot be indexed.
        AttributeError: For a one-cell subplot layout, because the current code
            expects Matplotlib's axes object to provide ``flatten``.
    """
    from matplotlib import pyplot as plt

    import numpy as np

    import pandas as pd


    range_ = slice(*range_)

    if metrics is None:
        metrics = list(history.keys())

    plotted_metrics = []
    for metric in metrics:
        if metric in plotted_metrics:
            continue

        if metric.startswith("val_") and \
            metric.replace("val_", '') in metrics:
                continue

        plotted_metrics.append(metric)

    if row is None:
        row = -(len(plotted_metrics) // -col)

    if figsize is None:
        figsize = (20, row*5)

    fig, axes = plt.subplots(row, col, figsize=figsize)
    axes = axes.flatten()

    for i, metric in enumerate(plotted_metrics):
        ax = axes[i]
        epochs = range(1, len(history.get(metric))+1)[range_]
        show_all_x_ticks = False if len(epochs) > 50 else show_all_x_ticks

        values = history.get(metric)[range_]
        min_ = min(values)
        max_ = max(values)

        ax.plot(epochs, values, 
            label="Training" if not metric.startswith("val_") else "Validation")

        if values:=history.get("val_"+metric, None):
            min_ = min([min_]+values)
            max_ = max([max_]+values)

            ax.plot(epochs, values[range_], label="Validation")            

        ax.legend()
        ax.set_xlabel("epochs")
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.tick_params(axis='x', rotation=x_ticks_rotation)
        ax.tick_params(axis='y', rotation=y_ticks_rotation)

        if show_all_x_ticks:
            ax.set_xticks(epochs)

        if y_ticks_num:
            ax.set_yticks(np.linspace(min_, max_, y_ticks_num))    

    for i in range(len(plotted_metrics), row*col):
        axes[i].set_visible(False)
    
    plt.tight_layout()

    if plot_path:
        fig.savefig(
            plot_path, 
            dpi=1_000, 
            bbox_inches="tight"
        )

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    if csv_path:
        history_df = pd.DataFrame(history)
        history_df.insert(0, "epoch", range(1, len(history_df) + 1))
        history_df.to_csv(csv_path, index=False)


def create_gif(
    output_path, 
    images1, 
    images2=None, 
    duration=100, 
    loop=0, 
    verbose=1
):
    """Tile diffusion trajectories and write an animated RGBA GIF.

    Args:
        output_path (str | os.PathLike): Destination including the ``.gif``
            suffix; parent directories must exist.
        images1 (Iterable[numpy.ndarray]): Nonempty frame sequence.  Each frame
            is shaped ``[samples, height, width, channels]`` with display values
            expected in ``[0, 1]``; only channel 0 is used and sample images are
            tiled horizontally.
        images2 (Iterable[numpy.ndarray] | None): Optional second trajectory.
            Paired frames (using truncating ``zip``) are stacked vertically per
            sample with a 10-pixel white separator before horizontal tiling.
        duration (int): Milliseconds per frame passed to Pillow.
        loop (int): GIF repeat count; ``0`` requests infinite looping.
        verbose (bool | int): Print the destination when truthy.

    Returns:
        None.

    Raises:
        IndexError: If the resulting frame sequence is empty.
        ValueError: If paired frame shapes cannot be concatenated.
        OSError: If Pillow cannot write the destination.
    """
    import numpy as np
    
    from PIL import Image


    if images2 is None:
        images = images1
    else:
        images = []
        for image1, image2 in zip(images1, images2):
            images.append(
                np.concatenate([
                    image1, 
                    np.ones((image1.shape[0], 10, image1.shape[2], image1.shape[3])), 
                    image2
                ], axis=1)
            )

    frames = []
    for image in images:
        img = (image*255).astype("uint8")[..., 0]
        img = np.concatenate(img, axis=1)
        img = Image.fromarray(img)

        frames.append(img.convert("RGBA"))

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )

    if verbose:
        print(f"GIF saved to {output_path}")


def show_img(x, y=None):
    """Display one image without axes and optionally add a label title.

    Args:
        x (numpy.ndarray | tf.Tensor): Matplotlib-compatible image shaped
            ``[height, width]`` or ``[height, width, channels]``.
        y (Sequence[object] | None): Indexable label container.  ``y[0]`` is
            displayed unless it is ``None``.  Despite the default, passing
            ``None`` currently fails because the function indexes it.

    Returns:
        None: The image is shown interactively.

    Raises:
        TypeError: If ``y`` is ``None`` or otherwise not indexable.
    """
    from matplotlib import pyplot as plt


    plt.imshow(x)
    plt.axis("off")

    if y[0] is not None:
        plt.title(f"Label: {y[0]}")

    plt.show()


def plot_images(
    imgs, 
    row=1, col=11, 
    show_images=True, 
    save_path=None
):
    """Display or save a grayscale batch as a labeled subplot grid.

    Args:
        imgs (numpy.ndarray | tf.Tensor): Images shaped
            ``[samples, height, width, channels]``; only channel 0 is rendered.
        row (int): Positive subplot row count.  The implementation always sets
            columns to ``len(imgs)``, so values above 1 create unused panels.
        col (int): Accepted for API compatibility but overwritten with
            ``len(imgs)``.
        show_images (bool): Display the figure interactively.
        save_path (str | os.PathLike | None): Optional image destination.  At
            least one of ``show_images`` or ``save_path`` must be enabled.

    Returns:
        None.

    Raises:
        AssertionError: If neither display nor saving is requested.
        AttributeError: For a single image in a one-row layout, because the
            scalar Matplotlib axes object has no ``flatten`` method.

    Note:
        Titles are ``i - 1``: the first image is labeled ``-1`` (the CFG null
        slot), followed by labels 0, 1, and so on.
    """
    from matplotlib import pyplot as plt


    assert show_images or save_path is not None


    col = len(imgs) # // row
    fig, axes = plt.subplots(row, col, figsize=(20, 6))
    axes = axes.flatten()

    for i in range(len(imgs)):
        axes[i].imshow(imgs[i, :, :, 0], cmap="gray")
        axes[i].set_title(f"{i-1}") 
        axes[i].axis("off")

    for j in range(len(imgs), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(
            save_path, 
            dpi=1_000, 
            bbox_inches="tight", 
        )

    if show_images:
        plt.show()
    else:
        plt.close(fig)


def save_samples(arr, path, type_):
    """Save a NumPy-compatible array as CSV or NPY using a base path.

    Args:
        arr (numpy.ndarray | array-like): Values to persist.  CSV generally
            requires a one- or two-dimensional numeric array; NPY supports
            arbitrary shapes and object arrays.
        path (str | os.PathLike): Base path without an extension.
        type_ (str): Exactly ``".csv"`` or ``".npy"``.  Any other value only
            prints ``"Wrong type!"`` and performs no write.

    Returns:
        None.
    """
    import numpy as np


    if type_ == ".csv":
        np.savetxt(path+type_, arr, delimiter=',')
    elif type_ == ".npy":
        with open(path+type_, "wb") as file:
            np.save(file ,arr)
    else:
        print("Wrong type!")


def load_samples(path, type_):
    """Load an NPY sample archive from a base path.

    Args:
        path (str | os.PathLike): Base path without an extension.
        type_ (str): ``".npy"`` loads with ``allow_pickle=True``.  Unsupported
            values return ``None``.  Although ``".csv"`` is recognized by a
            branch, CSV loading is not implemented.

    Returns:
        numpy.ndarray | None: Loaded NPY array, or ``None`` for an unrecognized
        type.

    Raises:
        UnboundLocalError: If ``type_ == ".csv"`` because no array is assigned.
        OSError: If the NPY path cannot be opened.
    """
    import numpy as np


    if type_ == ".csv":
        pass
    elif type_ == ".npy":
        with open(path+type_, "rb") as file:
            arr = np.load(file, allow_pickle=True)
    else:
        return None
    
    return arr


def save_logs(model_name, i, search_space=[], 
            names=[], metrics={}, where_to="file"):
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
    txt = ""

    if search_space and names:
        txt += f"----Optimization Iteration {i}:\n"
        for ss, name in zip(search_space, names):
            txt += f"{name}: {ss}\n"

    if metrics:
        txt += "----("
        for metric_name, metric_value in metrics.items():
            txt += f"{metric_name}={metric_value}, "

        txt = txt[:-2] + ")\n\n"

    if where_to == "file" or where_to == "both":
        with open(f"./models/hyperas/logs/{model_name}.txt", "at") as f: 
            f.write(txt)

    if where_to == "print" or where_to == "both":
        print(txt)
