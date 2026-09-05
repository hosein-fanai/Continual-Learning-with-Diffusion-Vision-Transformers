"""MNIST/CIFAR loading, preprocessing, limiting, and TensorFlow dataset helpers.

The array loaders preserve original class IDs and derive preprocessing statistics
from the training partition. They can return raw images or saved feature vectors,
with optional stratified validation and one-hot labels. ``get_dataset`` adds
batching and transforms; ``get_datasets`` resolves Config/direct options and
returns either prepared datasets or a deferred loader for continual learning.
Dataset downloads and feature reads occur when the selected loader is invoked.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import models

import numpy as np

from collections.abc import Callable, Mapping, Sequence

from .config import Config, normalize_training_task, resolve_continual_schedule


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
_DATASET_SPECS = {
    "mnist": (10, (28, 28, 1)), 
    "fmnist": (10, (28, 28, 1)), 
    "cifar10": (10, (32, 32, 3)), 
    "cifar100": (100, (32, 32, 3))
}
_LEGACY_FEATURE_SPLIT_SEED = 42
_LEGACY_FEATURE_VALIDATION_RATIO = 0.2


def _policy_numpy_dtype() -> np.dtype:
    """Return the active policy's stable floating NumPy dtype.

    Returns:
        numpy.dtype: NumPy equivalent of the active policy's ``variable_dtype``;
        for example float32 under mixed_float16 and float64 under float64.
        Reading the policy does not alter runtime settings.
    """

    variable_dtype = tf.keras.mixed_precision.global_policy().variable_dtype

    return np.dtype(tf.as_dtype(variable_dtype).as_numpy_dtype)


def _pad_images(
    x: np.ndarray, 
    pad: int, 
    value: float | int = 0
) -> np.ndarray:
    """Apply symmetric constant padding to a NumPy image batch.

    Args:
        x (numpy.ndarray): Images shaped ``[N, H, W]`` or ``[N, H, W, C]``.
        pad (int): Nonnegative padding width after integer normalization.
        value (float | int): Constant border value in the array's current
            preprocessing space.
            Defaults to ``0``.

    Returns:
        numpy.ndarray: Padded images with unchanged dtype and leading/channel
        dimensions. ``pad=0`` returns ``x`` unchanged.
    """

    # Preserve the original array when padding is disabled.
    if pad == 0:
        return x

    spatial_padding = ((0, 0), (int(pad), int(pad)), (int(pad), int(pad)))
    # Preserve a channel axis only for rank-four image batches.
    channel_padding = ((0, 0),) if x.ndim == 4 else ()

    return np.pad(
        x, 
        spatial_padding + channel_padding, 
        constant_values=value
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
        ValueError: If the limit cannot retain every represented class.
    """

    # Preserve all rows when they already fit within the limit.
    if max_samples is None or len(x) <= max_samples:
        return x, y

    labels = np.asarray(y)
    # Decode multi-column one-hot labels; flatten sparse vectors or columns.
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
    num_parallel_calls: object
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
        *metadata: tf.Tensor
    ) -> tuple[tf.Tensor, ...]:
        """Transform paired inputs without changing labels or metadata.

        Args:
            inputs (tf.Tensor): One batched input tensor.
            labels (tf.Tensor): Labels paired with the input batch.
            *metadata (tf.Tensor): Optional additional per-example tensors.

        Returns:
            tuple[tf.Tensor, ...]: Transformed inputs, labels, and metadata.
        """

        return transform(inputs), labels, *metadata


    # Preserve labels for supervised and conditional pipelines.
    if paired:
        return dataset.map(
            transform_pair, 
            num_parallel_calls=num_parallel_calls
        )

    return dataset.map(
        transform, 
        num_parallel_calls=num_parallel_calls
    )


def get_dataset_spec(
    dataset_name: str, 
    return_features: bool = False
) -> tuple[int, tuple[int, int, int], int]:
    """Return the class count, image shape, and flattened input size.

    Args:
        dataset_name (str): Case-insensitive ``"mnist"``, ``"fmnist"``,
            ``"cifar10"``, or ``"cifar100"``.
        return_features (bool): Use width 2,048 for saved features when true;
            false uses the product of the raw image dimensions. The returned
            image shape remains the dataset's raw shape in either mode.
            Defaults to ``False``.

    Returns:
        tuple[int, tuple[int, int, int], int]: Class count (10 or 100),
        ``(height, width, channels)`` (``(28, 28, 1)`` or ``(32, 32, 3)``), and
        flattened image/feature width. No dataset files are loaded.

    Raises:
        ValueError: If the dataset name is not one of the supported names.
    """

    spec = _DATASET_SPECS.get(dataset_name.lower())

    # Reject dataset names outside the supported families.
    if spec is None:
        raise ValueError(
            "dataset_name must be 'mnist', "
            "'fmnist', 'cifar10', or 'cifar100'."
        )

    class_num, image_shape = spec
    # Use the saved-feature width for feature inputs, otherwise flatten image geometry.
    flat_dim = 2_048 if return_features else int(np.prod(image_shape))

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

    return [
        [
            row
            for class_id in indices
            for row in np.where(labels == class_id)[0].tolist()
        ]
        for labels in labels_list
    ]


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
        onehot_labels (bool): If true, convert each label array to one-hot rows
            in the active policy's stable variable dtype, shaped
            ``[samples, class_num]``; otherwise retain its original
            integer-label rank and dtype.
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
        ValueError: If feature mode cannot infer a supported dataset from its
            path, the validation ratio is outside ``[0, 1)``,
            requested classes cannot support the stratified split, or
            feature/label lengths differ.
    """

    from tensorflow.keras.utils import to_categorical

    from sklearn.model_selection import train_test_split

    from common.utils import load_feature_split_metadata, load_samples


    stable_dtype = _policy_numpy_dtype()

    validation_ratio = float(validation_ratio)
    # Keep the validation fraction within its mathematical interval.
    if not 0. <= validation_ratio < 1.:
        raise ValueError("validation_ratio must lie in [0, 1).")
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

        # New archives declare the exact seed/ratio used to align their saved
        # train and validation features with labels. Archives created before
        # metadata support retain the historical 42/0.2 layout.
        feature_split_metadata = load_feature_split_metadata(features_path)
        # Use saved split metadata when available; otherwise preserve the legacy split.
        feature_split_seed, feature_validation_ratio = (
            feature_split_metadata
            if feature_split_metadata is not None
            else (
                _LEGACY_FEATURE_SPLIT_SEED,
                _LEGACY_FEATURE_VALIDATION_RATIO,
            )
        )
        y_train, y_val = train_test_split(
            y_train, 
            test_size=feature_validation_ratio,
            stratify=y_train, 
            random_state=feature_split_seed,
        )

        x_train, x_val, x_test = load_samples(features_path, ".npy")

        # Fail before concatenation if metadata and feature arrays describe a
        # different split, which would otherwise silently corrupt label pairs.
        feature_label_lengths = (
            ("train", len(x_train), len(y_train)),
            ("validation", len(x_val), len(y_val)),
            ("test", len(x_test), len(y_test)),
        )
        # Report only feature splits whose row counts disagree with reconstructed labels.
        mismatched_lengths = [
            f"{name}: features={feature_count}, labels={label_count}"
            for name, feature_count, label_count in feature_label_lengths
            if feature_count != label_count
        ]
        # Reject feature archives whose metadata does not align with labels.
        if mismatched_lengths:
            raise ValueError(
                "Saved feature split metadata does not align with labels ("
                + "; ".join(mismatched_lengths)
                + ")."
            )

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

    # Scale from training extrema to [0, 1] or diffusion's [-1, 1].
    if preprocess in ("min-max", "standardize", "diffusion"):
        min_ = x_train.min()
        max_ = x_train.max()
        value_range = max_ - min_
        # Avoid division by zero for constant inputs.
        if value_range == 0.:
            value_range = 1.

        x_train = (x_train.astype(stable_dtype) - min_) / value_range
        # Apply the training extrema to validation inputs when present.
        if x_val is not None:
            x_val = (x_val.astype(stable_dtype) - min_) / value_range
        x_test = (x_test.astype(stable_dtype) - min_) / value_range

        # Map normalized inputs into diffusion space when requested.
        if preprocess != "min-max":
            x_train = (x_train * 2.) - 1.
            # Apply the same mapping to validation inputs when present.
            if x_val is not None:
                x_val = (x_val * 2.) - 1.
            x_test = (x_test * 2.) - 1.
    # Normalize each feature to zero mean/unit variance.
    elif preprocess == "normalize":
        x_train = x_train.astype(stable_dtype)
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std = np.where(std == 0., 1., std)

        x_train = (x_train - mean) / std
        # Apply training normalization statistics to validation inputs.
        if x_val is not None:
            x_val = (x_val.astype(stable_dtype) - mean) / std
        x_test = (x_test.astype(stable_dtype) - mean) / std
    # Preserve values when no preprocessing is requested.
    else:
        # Align saved floating features with the active stable dtype.
        if return_features:
            x_train = x_train.astype(stable_dtype)
            # Align validation features when that split exists.
            if x_val is not None:
                x_val = x_val.astype(stable_dtype)
            x_test = x_test.astype(stable_dtype)
        # Keep raw image storage compact.
        else:
            x_train = x_train.astype("uint8")
            # Preserve compact storage for raw validation images.
            if x_val is not None:
                x_val = x_val.astype("uint8")
            x_test = x_test.astype("uint8")

    # Convert integer labels to full-width categorical rows.
    if onehot_labels:
        y_train = to_categorical(
            y_train,
            num_classes=class_num,
        ).astype(stable_dtype)
        # Convert validation labels when a validation split exists.
        if y_val is not None:
            y_val = to_categorical(
                y_val,
                num_classes=class_num,
            ).astype(stable_dtype)
        y_test = to_categorical(
            y_test,
            num_classes=class_num,
        ).astype(stable_dtype)

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
            # Decode one-hot labels for frequencies; retain sparse IDs for other label shapes.
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
    """Load MNIST and return filtered train, validation, and test arrays.

    The Keras loader may download missing source data to its usual local cache.
    Class IDs remain in their original dataset numbering. Validation is split
    from the filtered training data, and scaling statistics are computed from
    the remaining training partition. Saved-feature inputs follow the same
    filtering and re-splitting contract as raw images.

    Args:
        indices (Sequence[int]): Class IDs to retain in the requested grouping
            order. Defaults to ``tuple(range(10))`` (every dataset class).
        validation_ratio (float): Fraction of filtered training rows reserved
            by a stratified split. Defaults to ``0.2``; ``0.0`` disables
            validation. Must lie in ``[0, 1)``.
        preprocess (str | None): Defaults to ``None`` for no scaling.
            ``"min-max"`` uses scalar training extrema for ``[0, 1]`` scaling;
            ``"standardize"``/``"diffusion"`` maps those extrema to ``[-1, 1]``;
            ``"normalize"`` uses elementwise training mean/std. Other values
            preserve unscaled storage. Held-out values are not clipped.
        features_path (str | None): Base path without ``.npy`` for a saved
            train/validation/test feature archive. Defaults to
            ``"./data/mnist_xception_gavgpooled_features_train_val_test"``.
            Ignored for raw images; feature mode requires a dataset-identifying
            path and uses its optional metadata sidecar to reconstruct labels.
        return_features (bool): Defaults to ``False`` for images. ``True``
            replaces image arrays with the saved feature splits.
        onehot_labels (bool): Defaults to ``False`` for sparse labels.
            ``True`` returns rows of width ``10`` in the active policy's
            stable variable dtype.
        seed (int | None): Defaults to ``42`` for reproducible validation
            splitting. ``None`` allows a stochastic split.
        verbose (bool | int): Defaults to ``1`` to print split shapes and label
            frequencies; zero/False suppresses this output.

    Returns:
        DatasetArrays: ``(x_train, y_train, x_val, y_val, x_test, y_test)``.
        Raw images have shape ``[N, 28, 28]`` and sparse labels ``[N]``
        for each split; features have their archive-defined non-sample axes.
        Unscaled images are uint8. Scaled inputs, unscaled floating features,
        and one-hot labels use float32, or float64 under the float64 policy.
        Validation arrays are both None when validation is disabled.

    Raises:
        ValueError: If feature metadata/row counts are incompatible, the
            validation fraction is invalid, or a stratified split is infeasible.
        OSError: If required data or feature artifacts cannot be read/downloaded.
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
    """Load Fashion-MNIST and return filtered train, validation, and test arrays.

    The Keras loader may download missing source data to its usual local cache.
    Class IDs remain in their original dataset numbering. Validation is split
    from the filtered training data, and scaling statistics are computed from
    the remaining training partition. Saved-feature inputs follow the same
    filtering and re-splitting contract as raw images.

    Args:
        indices (Sequence[int]): Class IDs to retain in the requested grouping
            order. Defaults to ``tuple(range(10))`` (every dataset class).
        validation_ratio (float): Fraction of filtered training rows reserved
            by a stratified split. Defaults to ``0.2``; ``0.0`` disables
            validation. Must lie in ``[0, 1)``.
        preprocess (str | None): Defaults to ``None`` for no scaling.
            ``"min-max"`` uses scalar training extrema for ``[0, 1]`` scaling;
            ``"standardize"``/``"diffusion"`` maps those extrema to ``[-1, 1]``;
            ``"normalize"`` uses elementwise training mean/std. Other values
            preserve unscaled storage. Held-out values are not clipped.
        features_path (str | None): Base path without ``.npy`` for a saved
            train/validation/test feature archive. Defaults to
            ``"./data/fmnist_xception_gavgpooled_features_train_val_test"``.
            Ignored for raw images; feature mode requires a dataset-identifying
            path and uses its optional metadata sidecar to reconstruct labels.
        return_features (bool): Defaults to ``False`` for images. ``True``
            replaces image arrays with the saved feature splits.
        onehot_labels (bool): Defaults to ``False`` for sparse labels.
            ``True`` returns rows of width ``10`` in the active policy's
            stable variable dtype.
        seed (int | None): Defaults to ``42`` for reproducible validation
            splitting. ``None`` allows a stochastic split.
        verbose (bool | int): Defaults to ``1`` to print split shapes and label
            frequencies; zero/False suppresses this output.

    Returns:
        DatasetArrays: ``(x_train, y_train, x_val, y_val, x_test, y_test)``.
        Raw images have shape ``[N, 28, 28]`` and sparse labels ``[N]``
        for each split; features have their archive-defined non-sample axes.
        Unscaled images are uint8. Scaled inputs, unscaled floating features,
        and one-hot labels use float32, or float64 under the float64 policy.
        Validation arrays are both None when validation is disabled.

    Raises:
        ValueError: If feature metadata/row counts are incompatible, the
            validation fraction is invalid, or a stratified split is infeasible.
        OSError: If required data or feature artifacts cannot be read/downloaded.
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
    """Load CIFAR-10 and return filtered train, validation, and test arrays.

    The Keras loader may download missing source data to its usual local cache.
    Class IDs remain in their original dataset numbering. Validation is split
    from the filtered training data, and scaling statistics are computed from
    the remaining training partition. Saved-feature inputs follow the same
    filtering and re-splitting contract as raw images.

    Args:
        indices (Sequence[int]): Class IDs to retain in the requested grouping
            order. Defaults to ``tuple(range(10))`` (every dataset class).
        validation_ratio (float): Fraction of filtered training rows reserved
            by a stratified split. Defaults to ``0.2``; ``0.0`` disables
            validation. Must lie in ``[0, 1)``.
        preprocess (str | None): Defaults to ``None`` for no scaling.
            ``"min-max"`` uses scalar training extrema for ``[0, 1]`` scaling;
            ``"standardize"``/``"diffusion"`` maps those extrema to ``[-1, 1]``;
            ``"normalize"`` uses elementwise training mean/std. Other values
            preserve unscaled storage. Held-out values are not clipped.
        features_path (str | None): Base path without ``.npy`` for a saved
            train/validation/test feature archive. Defaults to
            ``"./data/cifar10_xception_gavgpooled_features_train_val_test"``.
            Ignored for raw images; feature mode requires a dataset-identifying
            path and uses its optional metadata sidecar to reconstruct labels.
        return_features (bool): Defaults to ``False`` for images. ``True``
            replaces image arrays with the saved feature splits.
        onehot_labels (bool): Defaults to ``False`` for sparse labels.
            ``True`` returns rows of width ``10`` in the active policy's
            stable variable dtype.
        seed (int | None): Defaults to ``42`` for reproducible validation
            splitting. ``None`` allows a stochastic split.
        verbose (bool | int): Defaults to ``1`` to print split shapes and label
            frequencies; zero/False suppresses this output.

    Returns:
        DatasetArrays: ``(x_train, y_train, x_val, y_val, x_test, y_test)``.
        Raw images have shape ``[N, 32, 32, 3]`` and sparse labels ``[N, 1]``
        for each split; features have their archive-defined non-sample axes.
        Unscaled images are uint8. Scaled inputs, unscaled floating features,
        and one-hot labels use float32, or float64 under the float64 policy.
        Validation arrays are both None when validation is disabled.

    Raises:
        ValueError: If feature metadata/row counts are incompatible, the
            validation fraction is invalid, or a stratified split is infeasible.
        OSError: If required data or feature artifacts cannot be read/downloaded.
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
    """Load CIFAR-100 and return filtered train, validation, and test arrays.

    The Keras loader may download missing source data to its usual local cache.
    Class IDs remain in their original dataset numbering. Validation is split
    from the filtered training data, and scaling statistics are computed from
    the remaining training partition. Saved-feature inputs follow the same
    filtering and re-splitting contract as raw images.

    Args:
        indices (Sequence[int]): Class IDs to retain in the requested grouping
            order. Defaults to ``tuple(range(100))`` (every dataset class).
        validation_ratio (float): Fraction of filtered training rows reserved
            by a stratified split. Defaults to ``0.2``; ``0.0`` disables
            validation. Must lie in ``[0, 1)``.
        preprocess (str | None): Defaults to ``None`` for no scaling.
            ``"min-max"`` uses scalar training extrema for ``[0, 1]`` scaling;
            ``"standardize"``/``"diffusion"`` maps those extrema to ``[-1, 1]``;
            ``"normalize"`` uses elementwise training mean/std. Other values
            preserve unscaled storage. Held-out values are not clipped.
        features_path (str | None): Base path without ``.npy`` for a saved
            train/validation/test feature archive. Defaults to
            ``"./data/cifar100_xception_gavgpooled_features_train_val_test"``.
            Ignored for raw images; feature mode requires a dataset-identifying
            path and uses its optional metadata sidecar to reconstruct labels.
        return_features (bool): Defaults to ``False`` for images. ``True``
            replaces image arrays with the saved feature splits.
        onehot_labels (bool): Defaults to ``False`` for sparse labels.
            ``True`` returns rows of width ``100`` in the active policy's
            stable variable dtype.
        seed (int | None): Defaults to ``42`` for reproducible validation
            splitting. ``None`` allows a stochastic split.
        verbose (bool | int): Defaults to ``1`` to print split shapes and label
            frequencies; zero/False suppresses this output.

    Returns:
        DatasetArrays: ``(x_train, y_train, x_val, y_val, x_test, y_test)``.
        Raw images have shape ``[N, 32, 32, 3]`` and sparse labels ``[N, 1]``
        for each split; features have their archive-defined non-sample axes.
        Unscaled images are uint8. Scaled inputs, unscaled floating features,
        and one-hot labels use float32, or float64 under the float64 policy.
        Validation arrays are both None when validation is disabled.

    Raises:
        ValueError: If feature metadata/row counts are incompatible, the
            validation fraction is invalid, or a stratified split is infeasible.
        OSError: If required data or feature artifacts cannot be read/downloaded.
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
    seed: int | None = None, 
    metadata: np.ndarray | tf.Tensor | None = None
) -> tf.data.Dataset:
    """Create one batched input, supervised, or metadata-bearing pipeline.

    Args:
        x (numpy.ndarray | tf.Tensor): Input samples shaped ``[samples, ...]``.
            Rank-three image batches receive a final singleton channel axis.
        y (numpy.ndarray | tf.Tensor | None): Optional labels aligned with
            ``x``. ``None`` creates an input-only dataset.
            Defaults to ``None``.
        pad (int): Nonnegative zero-padding width applied to both image axes
            after batching. ``0`` disables padding.
            Defaults to ``0``.
        cache (str | bool): ``True`` caches in memory, a nonempty string caches
            at that path, and ``False`` disables caching.
            Defaults to ``False``.
        shuffle_buffer (int): Positive shuffle capacity; ``0`` or a
            negative value preserves input order.
            Defaults to ``10000``.
        batch_size (int): Positive examples per batch.
            Defaults to ``128``.
        drop_remainder (bool): Whether to omit a final undersized batch.
            Defaults to ``True``.
        augment_fn (Callable | None): Optional callable applied to each batch of
            inputs while labels, when present, remain unchanged.
            Defaults to ``None``, skipping augmentation.
        conv_base (tf.keras.Model | None): Optional feature extractor applied to
            each input batch with ``training=False``.
            Defaults to ``None``, returning the transformed inputs directly.
        num_parallel_calls (int | None): Parallel mapping count. ``None`` uses
            ``AUTOTUNE`` mapping and prefetching.
            Defaults to ``None``.
        prefetch (bool): Whether to append a prefetch operation.
            Defaults to ``False``.
        seed (int | None): Optional deterministic seed for shuffling.
            Defaults to ``None``, leaving the dataset shuffle seed unspecified.
        metadata (numpy.ndarray | tf.Tensor | None): Optional third tensor
            aligned with ``x`` and ``y``. Continual distillation uses it for a
            replay-provenance mask. It requires non-``None`` labels.
            Defaults to ``None``, omitting the third dataset component.

    Returns:
        tf.data.Dataset: Batched inputs, ``(inputs, labels)`` pairs, or
        ``(inputs, labels, metadata)`` triples.

    Raises:
        ValueError: If metadata is supplied without labels.
    """

    from tensorflow.keras import layers


    # Use automatic mapping parallelism unless the caller supplies a value.
    num_parallel_calls = tf.data.AUTOTUNE if num_parallel_calls is None \
                        else num_parallel_calls

    # Add the channel dimension expected by image models.
    if len(x.shape) == 3:
        x = x[..., None]

    # Metadata has no meaning without the labels whose rows it describes.
    if metadata is not None and y is None:
        raise ValueError("metadata requires non-None labels.")
    # Build an input-only pipeline for unconditional models.
    if y is None:
        dataset = tf.data.Dataset.from_tensor_slices(x)
    # Retain optional per-example provenance beside supervised inputs.
    elif metadata is not None:
        dataset = tf.data.Dataset.from_tensor_slices((x, y, metadata))
    # Keep ordinary supervised or conditional inputs paired with their labels.
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
    kwargs: Mapping[str, object]
) -> dict[str, object]:
    """Resolve direct or configured dataset orchestration values once.

    Config mode copies authoritative dataset/training/continual settings and
    resolves the effective seed. Direct mode supplies the aliases, options,
    and defaults documented in ``get_datasets``; its pretrained inputs default
    to no preprocessing, while other model families default to standardization.

    Args:
        config (Config | None): Typed project configuration; ``None`` selects
            direct keyword resolution. This function does not mutate it.
        kwargs (Mapping[str, object]): Direct-mode dataset options consumed
            only when ``config`` is ``None``; the input mapping is not mutated.

    Returns:
        dict[str, object]: Flat dataset/model names; preprocessing, class/filter,
        feature, label, validation, batching, shuffle, padding, sample-cap,
        validation-toggle, seed, task, and VAE-input settings consumed by
        ``get_datasets``. Loading and TensorFlow pipeline construction are deferred.
    """

    # Keep the legacy direct defaults when no typed configuration is supplied.
    if config is None:
        model_name = kwargs.get(
            "model_name", 
            kwargs.get("model_type", kwargs.get("name", "diffusion_transformer"))
        )
        model_name = str(model_name).lower()
        # Leave pretrained images raw; standardize other direct model families.
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
            "task": normalize_training_task(kwargs.get("task", "legacy"))
        }

    # Infer the diffusion family from with_classifier when no model name is set.
    model_name = config.model.name or (
        "dit_classifier" if config.model.with_classifier
        else "diffusion_transformer"
    )
    model_name = str(model_name).lower()
    task = normalize_training_task(config.training.task)

    # Use a continual seed override for continual runs; otherwise use training.seed.
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
        "seed": (
            config.continually_learn.seed
            if task == "continual"
            and config.continually_learn.seed is not None
            else config.training.seed
        ), 
        "task": task,
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
            Defaults to ``None``, resolving the direct keyword options below.
        **kwargs (object): Direct options used only when ``config`` is ``None``:
            ``dataset_name`` (``"mnist"``, ``"fmnist"``, ``"cifar10"``, or
            ``"cifar100"``), ``model_name`` (str), ``preprocess``
            (str | None), ``indices`` (Sequence[int] | None),
            ``validation_ratio`` (float), ``return_features`` (bool),
            ``features_path`` (str), ``onehot_labels`` (bool), ``batch_size``
            (int), ``shuffle_buffer`` (int), ``pad`` (int),
            ``max_train_samples`` and ``max_val_samples`` (int | None),
            ``use_valset`` (bool), ``seed`` (int | None), and ``task`` (str).

    Direct Defaults:
        ``dataset_name="mnist"`` and ``model_name="diffusion_transformer"``
        select the loader and representation. ``model_type``/``name`` are
        fallback aliases for the model name. ``preprocess`` defaults to None
        for pretrained models and ``"standardize"`` otherwise. ``indices=None``
        selects every dataset class; ``validation_ratio=0.0`` creates no
        validation partition. ``return_features=False``, ``features_path=""``,
        and ``onehot_labels=False`` select raw images and sparse labels before
        any required VAE conditioning adjustment. ``batch_size=128``,
        ``shuffle_buffer=10000``, and ``pad=0`` define training batching,
        shuffle capacity, and spatial padding. ``max_train_samples=None`` and
        ``max_val_samples=None`` retain all rows; positive caps preserve at
        least one row per represented class. ``use_valset=True`` returns an
        existing validation partition, ``seed=None`` leaves selection unseeded,
        and ``task="legacy"`` selects ordinary dataset construction.
        ``model_kwargs`` (alias ``kwargs``, default empty mapping) supplies an
        optional VAE ``conditioned`` choice, defaulting to true for continual
        VAE runs and false for ordinary standalone VAEs.

    Returns:
        tuple[tf.data.Dataset | DatasetLoader, tf.data.Dataset | None]: Training
            and optional validation inputs. Continual tasks return the selected
            NumPy-array loader and ``None``; their configured limits, padding,
            shuffle capacity, and seed are reapplied by the continual trainer.
            All other tasks return batched ``tf.data.Dataset`` objects, with
            validation optionally disabled.

    Side Effects:
        Config mode records ``dataset.trainset_len``. A missing preprocessing
        mode is resolved to ``"standardize"`` for diffusion families. For VAE
        families it follows the reconstruction activation: ``tanh`` uses
        ``"standardize"``, ``sigmoid`` uses ``"min-max"``, and linear/``None``
        uses ``"normalize"``. The returned continual loader receives the same
        recorded setting. Direct pretrained calls default to raw images because
        Xception owns their rescaling; other direct families retain
        standardization. Conditional VAEs also record ``onehot_labels=True``.
        Continual mode loads and sizes the selected training pool for optimizer
        setup, then defers task-specific dataset creation to the learner.

    Raises:
        ValueError: If ``task`` is unsupported, or if ``pad`` is incompatible
            with saved features or a pretrained image model.
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

    # Conditional VAEs consume full-width labels as model inputs. Keep dataset
    # construction aligned with the effective factory setting rather than
    # requiring a redundant one-hot override from every caller.
    vae_conditioned = False
    # Resolve conditioning for standalone VAE families.
    if model_name in {"vae", "variational_autoencoder"}:
        # Direct VAE calls read conditioning from their model keyword mapping.
        if config is None:
            direct_model_kwargs = kwargs.get(
                "model_kwargs", kwargs.get("kwargs", {})
            ) or {}
            vae_conditioned = bool(direct_model_kwargs.get(
                "conditioned", task == "continual"
            ))
        # Configured generic model options override the typed VAE section.
        elif config.model.kwargs:
            vae_conditioned = bool(config.model.kwargs.get(
                "conditioned", task == "continual"
            ))
        # Without generic options, use typed conditioning or the continual default.
        else:
            vae_conditioned = bool(
                config.model.variational_autoencoder.conditioned
                or task == "continual"
            )

    # Attached classifiers and conditioned VAEs both require one-hot model inputs.
    if model_name == "vae_classifier" or vae_conditioned:
        onehot_labels = True
        # Persist the inferred label representation when a Config is available.
        if config is not None:
            config.dataset.onehot_labels = True

    dataset_name = dataset_name.lower()

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

    # Match an unspecified VAE input space to its effective output activation.
    if preprocess is None and model_name in {
        "vae", "variational_autoencoder", "vae_classifier"
    }:
        vae_activation = "tanh"
        # Respect generic model options before the typed family section.
        if config is not None:
            # Select the joint or standalone VAE section to resolve reconstruction scaling.
            typed_vae_config = config.model.vae_classifier \
                if model_name == "vae_classifier" \
                else config.model.variational_autoencoder
            # Prefer a generic reconstruction activation; otherwise use the typed activation.
            vae_activation = config.model.kwargs.get(
                "last_activation", "tanh"
            ) if config.model.kwargs else typed_vae_config.last_activation

        activation_name = getattr(
            vae_activation, "__name__", 
            vae_activation
        )
        # Normalize a named activation while preserving the explicit linear None value.
        activation_name = str(activation_name).lower() if activation_name is not None \
                        else None
        preprocess = {
            "tanh": "standardize", 
            "sigmoid": "min-max", 
            "linear": "normalize", 
            None: "normalize"
        }.get(activation_name)

        # Record a recognized activation's resolved preprocessing mode.
        if config is not None and preprocess is not None:
            config.dataset.preprocess = preprocess

    class_num, _, _ = get_dataset_spec(
        dataset_name, 
        return_features
    )

    # Include every dataset class by default.
    if indices is None:
        indices = list(range(class_num))
    # Resolve configured class order before the continual loader is deferred.
    if config is not None and task == "continual":
        indices, _ = resolve_continual_schedule(
            config.continually_learn.class_num, 
            config.continually_learn.class_order, 
            config.continually_learn.task_groups, 
            available_class_num=class_num, 
            task_size=config.continually_learn.task_size, 
            class_order_mode=config.continually_learn.class_order_mode, 
            task_order_mode=config.continually_learn.task_order_mode, 
            seed=options["seed"]
        )

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
    # Flatten sparse labels into the shape expected by Keras losses.
    if not onehot_labels:
        y_train = np.asarray(y_train).reshape(-1)
        # Flatten validation labels only when a validation split exists.
        y_val = np.asarray(y_val).reshape(-1) if y_val is not None else None

    rng = np.random.default_rng(seed)
    x_train, y_train = _limit_samples(
        x_train, y_train, 
        max_train_samples, 
        rng
    )
    # Continual training constructs its own task pipelines. Only the update
    # count is needed here to resolve configured optimizer schedules.
    if task == "continual":
        # Record the deferred pipeline length only for configured execution.
        if config is not None:
            config.dataset.trainset_len = (len(x_train) + batch_size - 1) // batch_size
        return loader, None

    # Limit only an independently created validation partition.
    if x_val is not None:
        x_val, y_val = _limit_samples(
            x_val, y_val, 
            max_val_samples, 
            rng
        )

    # Pad raw images before any dense-model flattening.
    if pad > 0:
        # Use a -1 border in diffusion space and a zero border in other input spaces.
        pad_value = -1. if str(preprocess).lower() in (
            "standardize", "diffusion"
        ) else 0.
        x_train = _pad_images(np.asarray(x_train), pad, value=pad_value)
        # Pad validation inputs only when a real validation partition exists.
        if x_val is not None:
            x_val = _pad_images(np.asarray(x_val), pad, value=pad_value)

    # Flatten inputs for dense classifiers and autoencoders.
    if model_name in _DENSE_MODELS:
        x_train = x_train.reshape((len(x_train), -1))
        # Flatten validation inputs only when a real partition exists.
        if x_val is not None:
            x_val = x_val.reshape((len(x_val), -1))

    trainset = get_dataset(
        x_train, 
        y_train, 
        pad=0, 
        shuffle_buffer=shuffle_buffer, 
        batch_size=batch_size, 
        drop_remainder=len(x_train) >= batch_size,
        seed=seed
    )

    # Record the built training pipeline length.
    if config is not None:
        config.dataset.trainset_len = len(trainset)

    # Build validation batches only when validation is enabled and arrays exist.
    valset = get_dataset(
        x_val,
        y_val,
        pad=0, 
        shuffle_buffer=0, 
        batch_size=batch_size, 
        drop_remainder=False
    ) if use_valset and x_val is not None else None

    return trainset, valset
