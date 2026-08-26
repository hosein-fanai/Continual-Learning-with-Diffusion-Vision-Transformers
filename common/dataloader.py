"""MNIST/CIFAR loading, preprocessing, limiting, and ``tf.data`` helpers."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import models

import numpy as np

from collections.abc import Callable, Mapping, Sequence
from numbers import Integral, Real

from .config import Config


DatasetArrays = tuple[
    np.ndarray, 
    np.ndarray, 
    np.ndarray | None, 
    np.ndarray | None, 
    np.ndarray, 
    np.ndarray
]
DatasetLoader = Callable[..., DatasetArrays]


_DIFFUSION_MODELS = {
    "diffusion_transformer", "dit_classifier", "dit_decoder",
    "dit_encoder_decoder", "dit_encoder_decoder_classifier",
    "unet", "unet_classifier"
}
_DENSE_MODELS = {"vae", "variational_autoencoder", "vae_classifier", "dnn"}


def _pad_images(
    x: np.ndarray,
    pad: int,
    value: float | int = 0,
) -> np.ndarray:
    """Apply symmetric constant padding to a NumPy image batch.

    Args:
        x (numpy.ndarray): Images shaped ``[N, H, W]`` or ``[N, H, W, C]``.
        pad (int): Non-boolean, nonnegative padding width.
        value (float | int): Constant border value in the array's current
            preprocessing space.

    Returns:
        numpy.ndarray: Padded images with unchanged dtype and leading/channel
        dimensions. ``pad=0`` returns ``x`` unchanged.

    Raises:
        TypeError: If ``pad`` is not a non-boolean integer.
        ValueError: If ``pad`` is negative or ``x`` is not a rank-three/four
            image batch.
    """

    # Reject booleans and non-integral padding widths.
    if isinstance(pad, bool) or not isinstance(pad, Integral):
        raise TypeError("pad must be a non-boolean integer.")
    # Keep spatial padding nonnegative.
    if pad < 0:
        raise ValueError("pad must be nonnegative.")
    # Accept only image batches with optional channel dimensions.
    if x.ndim not in (3, 4):
        raise ValueError("Padding requires [N, H, W] or [N, H, W, C] images.")
    # Preserve the original array when padding is disabled.
    if pad == 0:
        return x

    spatial_padding = ((0, 0), (int(pad), int(pad)), (int(pad), int(pad)))
    channel_padding = ((0, 0),) if x.ndim == 4 else ()

    return np.pad(
        x,
        spatial_padding + channel_padding,
        constant_values=value,
    )


def _limit_samples(
    x: np.ndarray, 
    y: np.ndarray, 
    max_samples: int | None, 
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Limit rows while retaining at least one example of every present class.

    Args:
        x (numpy.ndarray): Samples with leading row dimension.
        y (numpy.ndarray): Sparse or one-hot labels aligned with ``x``.
        max_samples (int | None): Positive retained-row limit, or ``None`` to
            keep every row.
        rng (numpy.random.Generator): Random generator used for selection.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Aligned limited samples and labels.

    Raises:
        ValueError: If the limit is nonpositive or cannot retain every class.
    """

    # Require a positive limit whenever limiting is enabled.
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided.")

    # Preserve all rows when they already fit within the limit.
    if max_samples is None or len(x) <= max_samples:
        return x, y

    labels = np.asarray(y)
    label_ids = np.argmax(labels, axis=-1) if labels.ndim > 1 and labels.shape[-1] > 1 \
                else labels.reshape(-1)
    classes = np.unique(label_ids)

    # Retain capacity for at least one example of every represented class.
    if max_samples < len(classes):
        raise ValueError(
            "A sample limit must be at least the number of represented classes."
        )

    selected = [
        int(rng.choice(np.flatnonzero(label_ids == class_id)))
        for class_id in classes
    ]

    remaining = max_samples - len(selected)
    # Fill unused capacity from rows not selected for class coverage.
    if remaining:
        available = np.setdiff1d(
            np.arange(len(x)), 
            np.asarray(selected), 
            assume_unique=True
        )
        selected.extend(rng.choice(
            available, 
            remaining, 
            replace=False
        ).tolist())

    selected = np.asarray(selected, dtype=int)
    rng.shuffle(selected)

    return x[selected], y[selected]


def _map_inputs(
    dataset: tf.data.Dataset,
    transform: Callable[[tf.Tensor], tf.Tensor],
    paired: bool,
    num_parallel_calls: object,
) -> tf.data.Dataset:
    """Apply one input transform while preserving optional labels.

    Args:
        dataset (tf.data.Dataset): Batched input-only or supervised pipeline.
        transform (Callable[[tf.Tensor], tf.Tensor]): Input transformation.
        paired (bool): Whether dataset elements are ``(inputs, labels)`` pairs.
        num_parallel_calls (object): Value forwarded to ``Dataset.map``.

    Returns:
        tf.data.Dataset: Dataset with transformed inputs and unchanged labels.
    """

    def transform_pair(
        inputs: tf.Tensor,
        labels: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Transform paired inputs without changing their labels.

        Args:
            inputs (tf.Tensor): One batched input tensor.
            labels (tf.Tensor): Labels paired with the input batch.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Transformed inputs and original labels.
        """

        return transform(inputs), labels


    # Preserve labels for supervised and conditional pipelines.
    if paired:
        return dataset.map(
            transform_pair,
            num_parallel_calls=num_parallel_calls,
        )

    return dataset.map(
        transform,
        num_parallel_calls=num_parallel_calls,
    )


def get_dataset_spec(
    dataset_name: str, 
    return_features: bool = False
) -> tuple[int, tuple[int, int, int], int]:
    """Return the class count, image shape, and flattened input size.

    Args:
        dataset_name (str): MNIST, Fashion-MNIST, CIFAR-10, or CIFAR-100 name.
        return_features (bool): Return the fixed saved-feature width when true.

    Returns:
        tuple[int, tuple[int, int, int], int]: Class count, image shape, and
        flattened image/feature width.
    """

    dataset_name = dataset_name.lower()
    # Select the MNIST shape and class count.
    if dataset_name == "mnist":
        class_num, image_shape = 10, (28, 28, 1)
    # Select the Fashion-MNIST shape and class count.
    elif dataset_name == "fmnist":
        class_num, image_shape = 10, (28, 28, 1)
    # Select the CIFAR-10 shape and class count.
    elif dataset_name == "cifar10":
        class_num, image_shape = 10, (32, 32, 3)
    # Select the CIFAR-100 shape and class count.
    elif dataset_name == "cifar100":
        class_num, image_shape = 100, (32, 32, 3)
    # Reject dataset names outside the four supported families.
    else:
        raise ValueError(
            "dataset_name must be 'mnist', "
            "'fmnist', 'cifar10', or 'cifar100'."
        )

    flat_dim = int(
        np.prod(image_shape)
    ) if not return_features else 2_048

    return class_num, image_shape, flat_dim


def sort_filter_labels(
    labels_list: Sequence[np.ndarray], 
    indices: Sequence[int]
) -> list[list[int]]:
    """Find row indices whose labels match requested classes, class by class.

    Args:
        labels_list (Iterable[numpy.ndarray]): Label arrays.  Each array usually
            has shape ``[samples]`` or ``[samples, 1]`` and contains integer
            class IDs.
        indices (Iterable[int]): Class IDs to retain, in the desired grouping
            order.  For ``indices=[2, 0]``, rows labeled ``2`` precede rows
            labeled ``0`` in the returned index list.  Repeated class IDs also
            repeat their rows; unknown IDs contribute no rows.

    Returns:
        list[list[int]]: One flat list of zero-based row indices per input label
        array.  Within a class, original row order is preserved.
    """

    labels_set_list = []
    for labels_set in labels_list:
        allowed_instances_list = []
        for index in indices:
            allowed_instances = np.where(labels_set == index)[0].tolist()
            allowed_instances_list.extend(allowed_instances)

        labels_set_list.append(allowed_instances_list)
    
    return labels_set_list


def preprocess_dataset(
    x_train: np.ndarray, 
    y_train: np.ndarray, 
    x_test: np.ndarray, 
    y_test: np.ndarray, 
    class_num: int, 
    indices: Sequence[int], 
    validation_ratio: float, 
    preprocess: str | None, 
    return_features: bool, 
    features_path: str | None, 
    onehot_labels: bool, 
    seed: int | None, 
    verbose: bool | int
) -> DatasetArrays:
    """Prepare filtered train, validation, and test NumPy arrays.

    Class filtering is applied in the order supplied by ``indices``. Training
    data is then split reproducibly using ``validation_ratio`` and stratified
    by label. Preprocessing statistics always come from that final
    training partition.

    Args:
        x_train (numpy.ndarray): Training samples shaped ``[N, ...]``.  For raw
            CIFAR this is ``[N, 32, 32, 3]``; it is ignored and replaced with
            saved feature arrays when ``return_features=True``.
        y_train (numpy.ndarray): Integer training labels shaped ``[N]`` or
            ``[N, 1]``.
        x_test (numpy.ndarray): Test samples shaped ``[M, ...]``; replaced in
            feature mode.
        y_test (numpy.ndarray): Integer test labels shaped ``[M]`` or ``[M, 1]``.
        class_num (int): Total output class count used for one-hot encoding.
            Feature mode resets it to 10 or 100 based on ``features_path``.
        indices (Sequence[int]): Class IDs to retain.  ``[3]`` creates a
            single-class subset; ``[0, 2, 1]`` groups data in that class order.
        validation_ratio (float): Training fraction reserved for validation.
            ``0.0`` disables the validation split.
        preprocess (str | None): ``"min-max"`` applies one global training-set
            minimum and maximum; ``"normalize"`` applies elementwise mean and
            standard deviation over axis 0; and ``"standardize"`` or
            ``"diffusion"`` maps the training extrema to ``[-1, 1]``. Held-out
            values are transformed with the same statistics and are not
            clipped, so they can lie outside the nominal interval. Any other
            value, including ``None`` or ``""``, performs no scaling. In
            raw-image mode unscaled arrays are cast to ``uint8``.
        return_features (bool): If true, load pre-extracted arrays instead of
            returning images. ``features_path`` must identify MNIST,
            Fashion-MNIST, CIFAR-10, or CIFAR-100 and contain three arrays in
            train/validation/test order.
        features_path (str | None): Base path without the ``.npy`` suffix. Its
            text must identify MNIST, Fashion-MNIST, CIFAR-10, or CIFAR-100 in
            feature mode; :func:`common.utils.load_samples` appends the suffix.
        onehot_labels (bool): If true, convert each label array to ``float32``
            one-hot rows shaped ``[samples, class_num]``; otherwise retain its
            original integer-label rank and dtype.
        seed (int | None): Random seed used for the stratified split.
        verbose (bool | int): Truthy values print shapes and simple label
            frequencies; falsy values suppress output.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray | None,
        numpy.ndarray | None, numpy.ndarray, numpy.ndarray]: ``(x_train,
        y_train, x_val, y_val, x_test, y_test)`` after filtering, splitting,
        preprocessing, and optional one-hot conversion. The validation arrays
        are ``None`` when ``validation_ratio`` is ``0.0``.

    Raises:
        TypeError: If ``validation_ratio`` is not a non-boolean real number.
        ValueError: If feature mode cannot infer a supported dataset from its
            path, the validation ratio is non-finite or outside ``[0, 1)``,
            requested classes cannot support the stratified split, or
            feature/label lengths differ.
    """

    from tensorflow.keras.utils import to_categorical

    from sklearn.model_selection import train_test_split

    from common.utils import load_samples

    # Require a non-boolean numeric validation ratio.
    if isinstance(validation_ratio, (bool, np.bool_)) \
    or not isinstance(validation_ratio, Real):
        raise TypeError("validation_ratio must be a non-boolean real number.")
    # Keep the validation fraction finite and within its valid interval.
    if not np.isfinite(validation_ratio) or not 0. <= validation_ratio < 1.:
        raise ValueError("validation_ratio must lie in [0, 1).")
    validation_ratio = float(validation_ratio)
    # Load saved feature vectors instead of raw images.
    if return_features:
        # Require an archive path whenever saved features are requested.
        if not features_path:
            raise ValueError("features_path is required when return_features=True.")
        normalized_features_path = str(features_path).lower()
        # Infer a ten-class label width from MNIST or CIFAR-10 archives.
        if ("cifar10_" in normalized_features_path or
        "mnist_" in normalized_features_path or
        "fmnist_" in normalized_features_path):
            class_num = 10
        # Infer the full CIFAR-100 label width from its archive name.
        elif "cifar100_" in normalized_features_path:
            class_num = 100
        # Reject feature paths whose dataset cannot be inferred.
        else:
            raise ValueError(
                "features_path has to identify mnist, "
                "fmnist, cifar10, or cifar100 features."
            )

        labels_set_list = sort_filter_labels(
            [y_train, y_test], 
            list(range(class_num))
        )
        y_train = y_train[labels_set_list[0]]
        y_test = y_test[labels_set_list[1]]

        y_train, y_val = train_test_split(
            y_train, 
            test_size=0.2, 
            stratify=y_train, 
            random_state=42
        )

        x_train, x_val, x_test = load_samples(features_path, ".npy")

        x_train = np.concatenate([x_train, x_val], axis=0)
        y_train = np.concatenate([y_train, y_val], axis=0)

    labels_set_list = sort_filter_labels(
        [y_train, y_test], 
        indices
    )
    x_train, y_train = x_train[labels_set_list[0]], y_train[labels_set_list[0]]
    x_test, y_test = x_test[labels_set_list[1]], y_test[labels_set_list[1]]

    # Reserve a stratified validation partition when requested.
    if validation_ratio > 0.:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, 
            test_size=validation_ratio, 
            stratify=y_train, 
            random_state=seed
        )
    # Omit validation arrays when no split is requested.
    else:
        x_val, y_val = None, None

    # Scale data to the [0, 1] interval.
    if preprocess == "min-max":
        min_ = x_train.min()
        max_ = x_train.max()
        value_range = max_ - min_
        # Avoid division by zero for constant inputs.
        if value_range == 0.:
            value_range = 1.

        x_train = (x_train.astype("float32") - min_) / value_range
        # Apply the training extrema to validation inputs when present.
        if x_val is not None:
            x_val = (x_val.astype("float32") - min_) / value_range
        x_test = (x_test.astype("float32") - min_) / value_range
    # Normalize each feature to zero mean/unit variance.
    elif preprocess == "normalize":
        x_train = x_train.astype("float32")
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std = np.where(std == 0., 1., std)

        x_train = (x_train - mean) / std
        # Apply training normalization statistics to validation inputs.
        if x_val is not None:
            x_val = (x_val.astype("float32") - mean) / std
        x_test = (x_test.astype("float32") - mean) / std
    # Scale diffusion inputs to [-1, 1].
    elif preprocess in ("standardize", "diffusion"):
        min_ = x_train.min()
        max_ = x_train.max()
        value_range = max_ - min_
        # Avoid division by zero for constant inputs.
        if value_range == 0.:
            value_range = 1.

        x_train = (x_train.astype("float32") - min_) / value_range
        # Apply the training extrema to validation inputs when present.
        if x_val is not None:
            x_val = (x_val.astype("float32") - min_) / value_range
        x_test = (x_test.astype("float32") - min_) / value_range

        x_train = (x_train * 2.) - 1.
        # Map validation inputs into diffusion space when present.
        if x_val is not None:
            x_val = (x_val * 2.) - 1.
        x_test = (x_test * 2.) - 1.
    # Preserve values when no preprocessing is requested.
    else:
        # Keep raw image storage compact.
        if not return_features:
            x_train = x_train.astype("uint8")
            # Preserve compact storage for raw validation images.
            if x_val is not None:
                x_val = x_val.astype("uint8")
            x_test = x_test.astype("uint8")

    # Convert integer labels to full-width categorical rows.
    if onehot_labels:
        y_train = to_categorical(y_train, num_classes=class_num)
        # Convert validation labels when a validation split exists.
        if y_val is not None:
            y_val = to_categorical(y_val, num_classes=class_num)
        y_test = to_categorical(y_test, num_classes=class_num)

    # Report prepared split shapes and label frequencies.
    if verbose:
        print("Trainset:", x_train.shape, y_train.shape)
        # Report validation shapes only when that split exists.
        if x_val is not None:
            print("Validation set:", x_val.shape, y_val.shape)
        print("Testset:", x_test.shape, y_test.shape)

        for set_id, dataset in enumerate((y_train, y_val, y_test)):
            # Skip an omitted validation-label array.
            if dataset is None:
                continue

            print(f"---{set_id}")
            label_ids = np.argmax(dataset, axis=-1) \
                if dataset.ndim > 1 and dataset.shape[-1] > 1 \
                else dataset.reshape(-1)
            class_ids, counts = np.unique(label_ids, return_counts=True)
            for clss_id, count in zip(class_ids, counts):
                print(clss_id, count / len(label_ids))
            
            print()

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_mnist(
    indices: Sequence[int] = tuple(range(10)), 
    validation_ratio: float = 0.2, 
    preprocess: str | None = None, 
    features_path: str | None = (
        "./data/mnist_xception_gavgpooled_features_train_val_test"
    ), 
    return_features: bool = False, 
    onehot_labels: bool = False, 
    seed: int | None = 42, 
    verbose: bool | int = 1
) -> DatasetArrays:
    """Load and prepare MNIST images or saved features.

    Args:
        indices (Sequence[int]): Retained class IDs from ``0`` through ``9``.
            Defaults to every class; ordering controls grouping before split.
        validation_ratio (float): Training fraction reserved for validation;
            ``0.0`` returns ``None`` validation arrays.
        preprocess (str | None): ``"min-max"``, ``"normalize"``,
            ``"standardize"``/``"diffusion"``, or any other value for no
            scaling; see :func:`preprocess_dataset` for exact behavior.
        features_path (str | None): Base path for a ``.npy`` archive of
            train/validation/test features.  Used only when
            ``return_features=True`` and must contain ``"mnist_"``.
        return_features (bool): Return saved feature vectors rather than raw
            ``28 x 28`` images.
        onehot_labels (bool): Return 10-column one-hot labels when true; return
            integer class IDs when false.
        seed (int | None): Random seed used for the stratified split.
        verbose (bool | int): Whether to print dataset summaries.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray | None,
        numpy.ndarray | None, numpy.ndarray, numpy.ndarray]: Training,
        optional validation, and test feature/label pairs in that order.

    Example:
        ``load_mnist(indices=[0, 1], preprocess="min-max",
        onehot_labels=True)`` returns only two classes, but one-hot rows still
        have width 10 because MNIST's full class count is retained.
    """

    from tensorflow.keras.datasets import mnist


    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    return preprocess_dataset(x_train, y_train, x_test, y_test, 10, 
                            indices, validation_ratio, preprocess, 
                            return_features, features_path, 
                            onehot_labels, seed, verbose)


def load_fmnist(
    indices: Sequence[int] = tuple(range(10)), 
    validation_ratio: float = 0.2, 
    preprocess: str | None = None, 
    features_path: str | None = (
        "./data/fmnist_xception_gavgpooled_features_train_val_test"
    ), 
    return_features: bool = False, 
    onehot_labels: bool = False, 
    seed: int | None = 42, 
    verbose: bool | int = 1
) -> DatasetArrays:
    """Load and prepare Fashion MNIST images or saved features.

    Args:
        indices (Sequence[int]): Retained class IDs from ``0`` through ``9``.
            Defaults to every class; ordering controls grouping before split.
        validation_ratio (float): Training fraction reserved for validation;
            ``0.0`` returns ``None`` validation arrays.
        preprocess (str | None): ``"min-max"``, ``"normalize"``,
            ``"standardize"``/``"diffusion"``, or any other value for no
            scaling; see :func:`preprocess_dataset` for exact behavior.
        features_path (str | None): Base path for a ``.npy`` archive of
            train/validation/test features.  Used only when
            ``return_features=True`` and must contain ``"fmnist_"``.
        return_features (bool): Return saved feature vectors rather than raw
            ``28 x 28`` images.
        onehot_labels (bool): Return 10-column one-hot labels when true; return
            integer class IDs when false.
        seed (int | None): Random seed used for the stratified split.
        verbose (bool | int): Whether to print dataset summaries.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray | None,
        numpy.ndarray | None, numpy.ndarray, numpy.ndarray]: Training,
        optional validation, and test feature/label pairs in that order.

    Example:
        ``load_fmnist(indices=[0, 1], preprocess="min-max",
        onehot_labels=True)`` returns only two classes, but one-hot rows still
        have width 10 because Fashion-MNIST's full class count is retained.
    """

    from tensorflow.keras.datasets import fashion_mnist


    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    return preprocess_dataset(x_train, y_train, x_test, y_test, 10, 
                            indices, validation_ratio, preprocess, 
                            return_features, features_path, 
                            onehot_labels, seed, verbose)


def load_cifar10(
    indices: Sequence[int] = tuple(range(10)), 
    validation_ratio: float = 0.2, 
    preprocess: str | None = None, 
    features_path: str | None = (
        "./data/cifar10_xception_gavgpooled_features_train_val_test"
    ), 
    return_features: bool = False, 
    onehot_labels: bool = False, 
    seed: int | None = 42, 
    verbose: bool | int = 1
) -> DatasetArrays:
    """Load and prepare CIFAR-10 images or saved features.

    Args:
        indices (Sequence[int]): Retained class IDs from ``0`` through ``9``.
            Defaults to every class; ordering controls grouping before split.
        validation_ratio (float): Training fraction reserved for validation;
            ``0.0`` returns ``None`` validation arrays.
        preprocess (str | None): ``"min-max"``, ``"normalize"``,
            ``"standardize"``/``"diffusion"``, or any other value for no
            scaling; see :func:`preprocess_dataset` for exact behavior.
        features_path (str | None): Base path for a ``.npy`` archive of
            train/validation/test features.  Used only when
            ``return_features=True`` and must contain ``"cifar10_"``.
        return_features (bool): Return saved feature vectors rather than raw
            ``32 x 32 x 3`` images.
        onehot_labels (bool): Return 10-column one-hot labels when true; return
            integer class IDs when false.
        seed (int | None): Random seed used for the stratified split.
        verbose (bool | int): Whether to print dataset summaries.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray | None,
        numpy.ndarray | None, numpy.ndarray, numpy.ndarray]: Training,
        optional validation, and test feature/label pairs in that order.

    Example:
        ``load_cifar10(indices=[0, 1], preprocess="min-max",
        onehot_labels=True)`` returns only two classes, but one-hot rows still
        have width 10 because CIFAR-10's full class count is retained.
    """

    from tensorflow.keras.datasets import cifar10


    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    return preprocess_dataset(x_train, y_train, x_test, y_test, 10, 
                            indices, validation_ratio, preprocess, 
                            return_features, features_path, 
                            onehot_labels, seed, verbose)


def load_cifar100(
    indices: Sequence[int] = tuple(range(100)), 
    validation_ratio: float = 0.2, 
    preprocess: str | None = None, 
    features_path: str | None = (
        "./data/cifar100_xception_gavgpooled_features_train_val_test"
    ), 
    return_features: bool = False, 
    onehot_labels: bool = False, 
    seed: int | None = 42, 
    verbose: bool | int = 1
) -> DatasetArrays:
    """Load and prepare CIFAR-100 fine-label images or saved features.

    Args:
        indices (Sequence[int]): Retained fine-label IDs from ``0`` through
            ``99``.  Defaults to all classes.
        validation_ratio (float): Training fraction reserved for validation;
            ``0.0`` returns ``None`` validation arrays.
        preprocess (str | None): ``"min-max"``, ``"normalize"``,
            ``"standardize"``/``"diffusion"``, or another value for no
            scaling; see :func:`preprocess_dataset`.
        features_path (str | None): Base path for a ``.npy`` archive of
            train/validation/test features.  Used only in feature mode and must
            contain ``"cifar100_"``.
        return_features (bool): Return saved feature vectors instead of raw
            images when true.
        onehot_labels (bool): Return 100-column one-hot labels when true.
        seed (int | None): Random seed used for the stratified split.
        verbose (bool | int): Whether to print dataset summaries.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray | None,
        numpy.ndarray | None, numpy.ndarray, numpy.ndarray]: Training,
        optional validation, and test feature/label pairs in that order.
    """

    from tensorflow.keras.datasets import cifar100


    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    return preprocess_dataset(x_train, y_train, x_test, y_test, 100, 
                            indices, validation_ratio, preprocess, 
                            return_features, features_path, 
                            onehot_labels, seed, verbose)


def get_dataset(
    x: np.ndarray | tf.Tensor, 
    y: np.ndarray | tf.Tensor | None = None, 
    pad: int = 0, 
    cache: str | bool = False, 
    shuffle_buffer: int = 10_000, 
    batch_size: int = 128, 
    drop_remainder: bool = True, 
    augment_fn: Callable | None = None,
    conv_base: models.Model | None = None, 
    num_parallel_calls: int | None = None, 
    prefetch: bool = False, 
    seed: int | None = None
) -> tf.data.Dataset:
    """Create one batched ``tf.data`` pipeline from inputs or input-label pairs.

    Args:
        x (numpy.ndarray | tf.Tensor): Input samples shaped ``[samples, ...]``.
            Rank-three image batches receive a final singleton channel axis.
        y (numpy.ndarray | tf.Tensor | None): Optional labels aligned with
            ``x``. ``None`` creates an input-only dataset.
        pad (int): Nonnegative zero-padding width applied to both image axes
            after batching. ``0`` disables padding.
        cache (str | bool): ``True`` caches in memory, a nonempty string caches
            at that path, and ``False`` disables caching.
        shuffle_buffer (int): Positive shuffle capacity; ``0`` or a
            negative value preserves input order.
        batch_size (int): Positive examples per batch.
        drop_remainder (bool): Whether to omit a final undersized batch.
        augment_fn (Callable | None): Optional callable applied to each batch of
            inputs while labels, when present, remain unchanged.
        conv_base (tf.keras.Model | None): Optional feature extractor applied to
            each input batch with ``training=False``.
        num_parallel_calls (int | None): Parallel mapping count. ``None`` uses
            ``AUTOTUNE`` mapping and prefetching.
        prefetch (bool): Whether to append a prefetch operation.
        seed (int | None): Optional deterministic seed for shuffling.

    Returns:
        tf.data.Dataset: Batched inputs when ``y`` is ``None``; otherwise
        batched ``(inputs, labels)`` pairs.

    Raises:
        TypeError: If ``pad`` is not a non-boolean integer.
        ValueError: If ``pad`` is negative or ``batch_size`` is not positive.
    """

    from tensorflow.keras import layers


    num_parallel_calls = tf.data.AUTOTUNE if num_parallel_calls is None \
                        else num_parallel_calls

    # Reject booleans and non-integral padding widths.
    if isinstance(pad, (bool, np.bool_)) or not isinstance(pad, Integral):
        raise TypeError("pad must be a non-boolean integer.")
    # Keep spatial padding nonnegative.
    if pad < 0:
        raise ValueError("pad must be nonnegative.")
    # Require a positive number of examples per batch.
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    # Add the channel dimension expected by image models.
    if len(x.shape) == 3:
        x = x[..., None]

    # Build an input-only pipeline for unconditional models.
    if y is None:
        dataset = tf.data.Dataset.from_tensor_slices(x)
    # Keep supervised or conditional inputs paired with their labels.
    else:
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

    # Cache transformed elements in memory when explicitly requested.
    if cache is True:
        dataset = dataset.cache()
    # Cache transformed elements at the requested filesystem path.
    elif cache:
        dataset = dataset.cache(str(cache))

    # Randomize element order when shuffling is enabled.
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(
            shuffle_buffer, 
            seed=seed
        )

    dataset = dataset.batch(
        batch_size, 
        drop_remainder=drop_remainder
    )

    # Add spatial zero padding when requested.
    if pad > 0:
        padder = layers.ZeroPadding2D((pad, pad))
        dataset = _map_inputs(
            dataset,
            padder,
            paired=y is not None,
            num_parallel_calls=num_parallel_calls,
        )

    # Apply the configured input augmentation.
    if augment_fn is not None:
        dataset = _map_inputs(
            dataset,
            augment_fn,
            paired=y is not None,
            num_parallel_calls=num_parallel_calls,
        )

    # Extract fixed features from every input batch.
    if conv_base is not None:
        def extract_inputs(inputs: tf.Tensor) -> tf.Tensor:
            """Run the configured feature extractor in inference mode.

            Args:
                inputs (tf.Tensor): One batched model input tensor.

            Returns:
                tf.Tensor: Extracted feature tensor.
            """

            return conv_base(inputs, training=False)


        dataset = _map_inputs(
            dataset,
            extract_inputs,
            paired=y is not None,
            num_parallel_calls=num_parallel_calls,
        )

    # Overlap input preparation with model execution when requested.
    if prefetch:
        dataset = dataset.prefetch(num_parallel_calls)

    return dataset


def _resolve_dataset_options(
    config: Config | None,
    kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Resolve direct or configured dataset orchestration values once.

    Args:
        config (Config | None): Typed project configuration, when supplied.
        kwargs (Mapping[str, object]): Direct-mode dataset options.

    Returns:
        dict[str, object]: Flat values consumed by :func:`get_datasets`.
    """

    # Keep the legacy direct defaults when no typed configuration is supplied.
    if config is None:
        model_name = kwargs.get(
            "model_name",
            kwargs.get("model_type", kwargs.get("name", "diffusion_transformer")),
        )
        model_name = str(model_name).lower()
        default_preprocess = None if model_name == "pretrained" \
                             else "standardize"

        return {
            "dataset_name": kwargs.get("dataset_name", "mnist"),
            "model_name": model_name,
            "preprocess": kwargs.get("preprocess", default_preprocess),
            "indices": kwargs.get("indices"),
            "validation_ratio": kwargs.get("validation_ratio", 0.),
            "return_features": kwargs.get("return_features", False),
            "features_path": kwargs.get("features_path", ""),
            "onehot_labels": kwargs.get("onehot_labels", False),
            "batch_size": kwargs.get("batch_size", 128),
            "shuffle_buffer": kwargs.get("shuffle_buffer", 10_000),
            "pad": kwargs.get("pad", 0),
            "max_train_samples": kwargs.get("max_train_samples"),
            "max_val_samples": kwargs.get("max_val_samples"),
            "use_valset": kwargs.get("use_valset", True),
            "seed": kwargs.get("seed"),
            "task": kwargs.get("task", "legacy"),
        }

    model_name = config.model.name or (
        "dit_classifier" if config.model.with_classifier
        else "diffusion_transformer"
    )
    model_name = str(model_name).lower()

    return {
        "dataset_name": config.dataset.name,
        "model_name": model_name,
        "preprocess": config.dataset.preprocess,
        "indices": config.dataset.indices,
        "validation_ratio": config.dataset.validation_ratio,
        "return_features": config.dataset.return_features,
        "features_path": config.dataset.features_path,
        "onehot_labels": config.dataset.onehot_labels,
        "batch_size": config.dataset.batch_size,
        "shuffle_buffer": config.dataset.shuffle_buffer,
        "pad": config.dataset.pad,
        "max_train_samples": config.dataset.max_train_samples,
        "max_val_samples": config.dataset.max_val_samples,
        "use_valset": config.training.use_valset,
        "seed": config.training.seed,
        "task": config.training.task,
    }


def get_datasets(
    config: Config | None = None, 
    **kwargs: object
) -> tuple[tf.data.Dataset | DatasetLoader, tf.data.Dataset | None]:
    """Build the selected training and validation datasets.

    Settings may come from a :class:`common.config.Config` object or directly
    from keyword arguments. Config values take precedence when both are
    supplied.

    Args:
        config (Config | None): Optional typed project configuration. When
            provided, its dataset, model, and training sections supply every
            setting and direct keyword options are ignored.
        **kwargs (object): Direct options used only when ``config`` is ``None``:
            ``dataset_name`` (``"mnist"``, ``"fmnist"``, ``"cifar10"``, or
            ``"cifar100"``), ``model_name`` (str), ``preprocess``
            (str | None), ``indices`` (Sequence[int] | None),
            ``validation_ratio`` (float), ``return_features`` (bool),
            ``features_path`` (str), ``onehot_labels`` (bool), ``batch_size``
            (int), ``shuffle_buffer`` (int), ``pad`` (int),
            ``max_train_samples`` and ``max_val_samples`` (int | None),
            ``use_valset`` (bool), ``seed`` (int | None), and ``task`` (str).

    Returns:
        tuple[tf.data.Dataset | DatasetLoader, tf.data.Dataset | None]: Training
            and optional validation inputs. Continual tasks return the selected
            NumPy-array loader and ``None``; their configured limits, padding,
            shuffle capacity, and seed are reapplied by the continual trainer.
            All other tasks return batched ``tf.data.Dataset`` objects, with
            validation optionally disabled.

    Side Effects:
        Config mode records ``dataset.trainset_len``. A missing preprocessing
        mode is also resolved to ``"standardize"`` for diffusion families so
        the returned continual loader receives the same effective setting.
        Direct pretrained calls default to raw images because Xception owns
        their rescaling; other direct families retain standardization.

    Raises:
        TypeError: If ``pad`` is not a non-boolean integer.
        ValueError: If ``pad`` is negative or incompatible with saved features
            or a pretrained image model.
    """

    options = _resolve_dataset_options(config, kwargs)
    dataset_name = options["dataset_name"]
    model_name = options["model_name"]
    preprocess = options["preprocess"]
    indices = options["indices"]
    validation_ratio = options["validation_ratio"]
    return_features = options["return_features"]
    features_path = options["features_path"]
    onehot_labels = options["onehot_labels"]
    batch_size = options["batch_size"]
    shuffle_buffer = options["shuffle_buffer"]
    pad = options["pad"]
    max_train_samples = options["max_train_samples"]
    max_val_samples = options["max_val_samples"]
    use_valset = options["use_valset"]
    seed = options["seed"]
    task = options["task"]

    # Reject booleans and non-integral padding widths before loading data.
    if isinstance(pad, (bool, np.bool_)) or not isinstance(pad, Integral):
        raise TypeError("pad must be a non-boolean integer.")
    # Keep spatial padding nonnegative.
    if pad < 0:
        raise ValueError("pad must be nonnegative.")

    dataset_name = dataset_name.lower()
    model_name = model_name.lower()
    task = task.lower()

    # Prevent image padding from being applied to saved feature vectors.
    if pad and return_features:
        raise ValueError("pad is not supported for saved feature inputs.")
    # Keep pretrained image geometry unchanged.
    if pad and model_name in {"pretrained", "hp-tuned"}:
        raise ValueError("pad is not supported for pretrained/hp-tuned models.")
    # Keep continual pretrained classifiers on their required image geometry.
    if config is not None and pad and task == "continual" \
    and str(config.model.classifier_name).lower() in {
        "pretrained", "hp-tuned"
    }:
        raise ValueError(
            "pad is not supported for pretrained/hp-tuned classifiers."
        )

    # Supply diffusion-safe scaling when no preprocessing mode was chosen.
    if preprocess is None and model_name in _DIFFUSION_MODELS:
        preprocess = "standardize"
        # Record the resolved preprocessing mode in typed configuration.
        if config is not None:
            config.dataset.preprocess = preprocess

    class_num, _, _ = get_dataset_spec(
        dataset_name, 
        return_features
    )
    # Include every dataset class by default.
    if indices is None:
        indices = list(range(class_num))
    # Restrict configured continual runs to their requested leading classes.
    if config is not None and task == "continual" \
    and config.continually_learn.class_num is not None:
        continual_class_num = config.continually_learn.class_num
        # Keep the continual class count within dataset bounds.
        if not 2 <= continual_class_num <= class_num:
            raise ValueError(
                "continually_learn.class_num must be between "
                "2 and the selected dataset's class count."
            )

        indices = list(range(continual_class_num))

    loaders = {
        "mnist": load_mnist, 
        "fmnist": load_fmnist, 
        "cifar10": load_cifar10, 
        "cifar100": load_cifar100
    }
    loader = loaders[dataset_name]

    x_train, y_train, x_val, y_val, x_test, y_test = loader(
        indices=indices, 
        validation_ratio=validation_ratio, 
        preprocess=preprocess, 
        features_path=features_path, 
        return_features=return_features, 
        onehot_labels=onehot_labels, 
        seed=seed, 
        verbose=0
    )
    # Prefer the requested validation partition.
    if x_val is not None:
        x_eval, y_eval = x_val, y_val
    # Fall back to test data when the loader omitted validation data.
    else:
        x_eval, y_eval = x_test, y_test

    # Flatten sparse labels into the shape expected by Keras losses.
    if not onehot_labels:
        y_train = np.asarray(y_train).reshape(-1)
        y_eval = np.asarray(y_eval).reshape(-1)

    rng = np.random.default_rng(seed)
    x_train, y_train = _limit_samples(
        x_train, y_train, 
        max_train_samples, 
        rng
    )
    x_eval, y_eval = _limit_samples(
        x_eval, y_eval, 
        max_val_samples, 
        rng
    )

    # Pad raw images before any dense-model flattening.
    if pad > 0:
        pad_value = -1. if str(preprocess).lower() in (
            "standardize", "diffusion"
        ) else 0.
        x_train = _pad_images(np.asarray(x_train), pad, value=pad_value)
        x_eval = _pad_images(np.asarray(x_eval), pad, value=pad_value)

    # Flatten inputs for dense classifiers and autoencoders.
    if model_name in _DENSE_MODELS:
        x_train = x_train.reshape((len(x_train), -1))
        x_eval = x_eval.reshape((len(x_eval), -1))

    trainset = get_dataset(
        x_train, 
        y_train, 
        pad=0, 
        shuffle_buffer=shuffle_buffer, 
        batch_size=batch_size, 
        drop_remainder=(task != "continual" and len(x_train) >= batch_size), 
        seed=seed
    )

    # Record the built training pipeline length.
    if config is not None:
        config.dataset.trainset_len = len(trainset)

    valset = get_dataset(
        x_eval, 
        y_eval, 
        pad=0, 
        shuffle_buffer=0, 
        batch_size=batch_size, 
        drop_remainder=False
    ) if use_valset else None # Build a deterministic validation pipeline.

    # Defer per-task loading to the continual learner.
    if task == "continual":
        return loader, None
    return trainset, valset
