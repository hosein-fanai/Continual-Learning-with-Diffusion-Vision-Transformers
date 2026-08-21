"""MNIST and CIFAR loading, class filtering, preprocessing, and ``tf.data`` helpers."""

import tensorflow as tf
from tensorflow.keras import models

import numpy as np

from collections.abc import Callable, Sequence

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


def _limit_samples(
    x: np.ndarray, 
    y: np.ndarray, 
    max_samples: int | None, 
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Limit rows while retaining at least one example of every present class."""

    if max_samples is None or len(x) <= max_samples:
        return x, y

    labels = np.asarray(y)
    label_ids = np.argmax(labels, axis=-1) if labels.ndim > 1 and labels.shape[-1] > 1 \
                else labels.reshape(-1)
    classes = np.unique(label_ids)

    if max_samples < len(classes):
        raise ValueError(
            "A sample limit must be at least the number of represented classes."
        )

    selected = [
        int(rng.choice(np.flatnonzero(label_ids == class_id)))
        for class_id in classes
    ]

    remaining = max_samples - len(selected)
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


def get_dataset_spec(
    dataset_name: str, 
    return_features: bool = False
):
    """Return the class count, image shape, and flattened input size."""

    dataset_name = dataset_name.lower()
    if dataset_name == "mnist":
        class_num, image_shape = 10, (28, 28, 1)
    elif dataset_name == "fmnist":
        class_num, image_shape = 10, (28, 28, 1)
    elif dataset_name == "cifar10":
        class_num, image_shape = 10, (32, 32, 3)
    elif dataset_name == "cifar100":
        class_num, image_shape = 100, (32, 32, 3)
    else:
        raise ValueError(
            "dataset_name must be 'mnist', "
            "'fmnist', 'cifar10', or 'cifar100'."
        )

    flat_dim = int(
        np.prod(image_shape)
    ) if not return_features else 2_048

    return class_num, image_shape, flat_dim


def sort_filter_labels(labels_list, indices):
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
            ``"diffusion"`` scales values to ``[-1, 1]``. Any other value,
            including ``None`` or ``""``, performs no scaling. In raw-image
            mode the unscaled training and test arrays are cast to ``uint8``.
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
        Exception: If feature mode cannot infer MNIST, Fashion-MNIST, CIFAR-10,
            or CIFAR-100 from the path.
        ValueError: If the requested classes cannot support the configured
            stratified validation split, or if feature and label lengths differ.
    """

    from tensorflow.keras.utils import to_categorical

    from sklearn.model_selection import train_test_split

    from common.utils import load_samples


    if return_features: # Load saved feature vectors instead of raw images.
        if ("cifar10_" in features_path or # Select ten-class feature archives.
        "mnist_" in features_path or 
        "fmnist_" in features_path):
            class_num = 10
        elif "cifar100_" in features_path: # Select the 100-class archive.
            class_num = 100
        else: # Reject feature paths whose dataset cannot be inferred.
            raise Exception(
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

    if validation_ratio > 0.: # Reserve a stratified validation partition.
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, 
            test_size=validation_ratio, 
            stratify=y_train, 
            random_state=seed
        )
    else: # Keep every training row when validation is disabled.
        x_val, y_val = None, None

    if preprocess == "min-max": # Scale data to the [0, 1] interval.
        min_ = x_train.min()
        max_ = x_train.max()
        value_range = max_ - min_
        if value_range == 0.: # Avoid division by zero for constant inputs.
            value_range = 1.

        x_train = (x_train.astype("float32") - min_) / value_range
        if x_val is not None:
            x_val = (x_val.astype("float32") - min_) / value_range
        x_test = (x_test.astype("float32") - min_) / value_range
    elif preprocess == "normalize": # Normalize each feature to zero mean/unit variance.
        x_train = x_train.astype("float32")
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std = np.where(std == 0., 1., std)

        x_train = (x_train - mean) / std
        if x_val is not None:
            x_val = (x_val.astype("float32") - mean) / std
        x_test = (x_test.astype("float32") - mean) / std
    elif preprocess in ("standardize", "diffusion"): # Scale diffusion inputs to [-1, 1].
        min_ = x_train.min()
        max_ = x_train.max()
        value_range = max_ - min_
        if value_range == 0.: # Avoid division by zero for constant inputs.
            value_range = 1.

        x_train = (x_train.astype("float32") - min_) / value_range
        if x_val is not None:
            x_val = (x_val.astype("float32") - min_) / value_range
        x_test = (x_test.astype("float32") - min_) / value_range

        x_train = (x_train * 2.) - 1.
        if x_val is not None:
            x_val = (x_val * 2.) - 1.
        x_test = (x_test * 2.) - 1.
    else: # Preserve values when no preprocessing is requested.
        if not return_features: # Keep raw image storage compact.
            x_train = x_train.astype("uint8")
            x_test = x_test.astype("uint8")

    if onehot_labels: # Convert integer labels to full-width categorical rows.
        y_train = to_categorical(y_train, num_classes=class_num)
        if y_val is not None:
            y_val = to_categorical(y_val, num_classes=class_num)
        y_test = to_categorical(y_test, num_classes=class_num)

    if verbose: # Report prepared split shapes and label frequencies.
        print("Trainset:", x_train.shape, y_train.shape)
        if x_val is not None:
            print("Validation set:", x_val.shape, y_val.shape)
        print("Testset:", x_test.shape, y_test.shape)

        for set_id, dataset in enumerate((y_train, y_val, y_test)):
            if dataset is None: # Skip an omitted validation-label array.
                continue

            print(f"---{set_id}")
            for clss_id in np.unique(dataset):
                print(clss_id, sum(dataset == clss_id) / len(dataset))
            
            print()

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_mnist(
    indices: Sequence[int] = list(range(10)), 
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
    indices: Sequence[int] = list(range(10)), 
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
    indices: Sequence[int] = list(range(10)), 
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
    indices: Sequence[int] = list(range(100)), 
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
    seed: int | None = None, 
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
        num_parallel_calls (int | None): Parallel mapping count and prefetch
            buffer size. ``None`` uses TensorFlow's synchronous/default value.
        prefetch (bool): Whether to append a prefetch operation.
        seed (int | None): Optional deterministic seed for shuffling.

    Returns:
        tf.data.Dataset: Batched inputs when ``y`` is ``None``; otherwise
        batched ``(inputs, labels)`` pairs.
    """

    from tensorflow.keras import layers


    if len(x.shape) == 3: # Add the channel dimension expected by image models.
        x = x[..., None]

    if y is None: # Build an input-only pipeline for unconditional models.
        dataset = tf.data.Dataset.from_tensor_slices(x)
    else: # Keep supervised or conditional inputs paired with their labels.
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

    if cache is True: # Cache transformed elements in memory.
        dataset = dataset.cache()
    elif cache: # Cache transformed elements in the requested file.
        dataset = dataset.cache(str(cache))

    if shuffle_buffer > 0: # Randomize training element order.
        dataset = dataset.shuffle(
            shuffle_buffer, 
            seed=seed
        )

    dataset = dataset.batch(
        batch_size, 
        drop_remainder=drop_remainder
    )

    if pad > 0: # Add spatial zero padding when requested.
        padder = layers.ZeroPadding2D((pad, pad))
        if y is None: # Transform input-only dataset elements.
            dataset = dataset.map(
                lambda x: padder(x), 
                num_parallel_calls=num_parallel_calls
            )
        else: # Preserve labels while padding paired inputs.
            dataset = dataset.map(
                lambda x, y: (padder(x), y), 
                num_parallel_calls=num_parallel_calls
            )

    if augment_fn is not None: # Apply the configured input augmentation.
        if y is None: # Transform input-only dataset elements.
            dataset = dataset.map(
                lambda x: augment_fn(x), 
                num_parallel_calls=num_parallel_calls
            )
        else: # Preserve labels while augmenting paired inputs.
            dataset = dataset.map(
                lambda x, y: (augment_fn(x), y), 
                num_parallel_calls=num_parallel_calls
            )

    if conv_base is not None: # Extract fixed features from every input batch.
        if y is None: # Transform input-only dataset elements.
            dataset = dataset.map(
                lambda x: conv_base(x, training=False),
                num_parallel_calls=num_parallel_calls
            )
        else: # Preserve labels while transforming paired inputs.
            dataset = dataset.map(
                lambda x, y: (conv_base(x, training=False), y), 
                num_parallel_calls=num_parallel_calls
            )

    if prefetch: # Overlap input preparation with model execution.
        dataset = dataset.prefetch(num_parallel_calls)

    return dataset


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
    """

    if config is None: # Resolve settings from direct keyword arguments.
        dataset_name = kwargs.get("dataset_name", "mnist")
        model_name = kwargs.get("model_name", "diffusion_transformer")
        preprocess = kwargs.get("preprocess", "standardize")
        indices = kwargs.get("indices")
        validation_ratio = kwargs.get("validation_ratio", 0.)
        return_features = kwargs.get("return_features", False)
        features_path = kwargs.get("features_path", "")
        onehot_labels = kwargs.get("onehot_labels", False)
        batch_size = kwargs.get("batch_size", 128)
        shuffle_buffer = kwargs.get("shuffle_buffer", 10_000)
        pad = kwargs.get("pad", 0)
        max_train_samples = kwargs.get("max_train_samples")
        max_val_samples = kwargs.get("max_val_samples")
        use_valset = kwargs.get("use_valset", True)
        seed = kwargs.get("seed")
        task = kwargs.get("task", "legacy")
    else: # Resolve settings from the typed configuration.
        dataset_name = config.dataset.name
        model_name = config.model.name
        if model_name is None: # Select the legacy diffusion model variant.
            if config.model.with_classifier: # Select the classifier network.
                model_name = "dit_classifier"
            else: # Select the generator-only network.
                model_name = "diffusion_transformer"
        preprocess = config.dataset.preprocess
        indices = config.dataset.indices
        validation_ratio = config.dataset.validation_ratio
        return_features = config.dataset.return_features
        features_path = config.dataset.features_path
        onehot_labels = config.dataset.onehot_labels
        batch_size = config.dataset.batch_size
        shuffle_buffer = config.dataset.shuffle_buffer
        pad = config.dataset.pad
        max_train_samples = config.dataset.max_train_samples
        max_val_samples = config.dataset.max_val_samples
        use_valset = config.training.use_valset
        seed = config.training.seed
        task = config.training.task

    dataset_name = dataset_name.lower()
    model_name = model_name.lower()
    task = task.lower()

    if pad and return_features:
        raise ValueError("pad is not supported for saved feature inputs.")
    if pad and model_name in {"pretrained", "hp-tuned"}:
        raise ValueError("pad is not supported for pretrained/hp-tuned models.")
    if config is not None and pad and task == "continual" \
    and str(config.model.classifier_name).lower() in {
        "pretrained", "hp-tuned"
    }:
        raise ValueError(
            "pad is not supported for pretrained/hp-tuned classifiers."
        )

    if preprocess is None and model_name in { 
        "diffusion_transformer", "dit_classifier", "dit_decoder", 
        "dit_encoder_decoder", "dit_encoder_decoder_classifier", 
        "unet", "unet_classifier"
    }: # Supply diffusion-safe scaling by default.
        preprocess = "standardize"
        if config is not None:
            config.dataset.preprocess = preprocess

    class_num, _, _ = get_dataset_spec(
        dataset_name, 
        return_features
    )
    if indices is None: # Include every dataset class by default.
        indices = list(range(class_num))
    if config is not None and task == "continual" \
    and config.continually_learn.class_num is not None:
        continual_class_num = config.continually_learn.class_num
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
    if x_val is not None: # Prefer the requested validation partition.
        x_eval, y_eval = x_val, y_val
    else: # Fall back to test data when validation is disabled.
        x_eval, y_eval = x_test, y_test

    if not onehot_labels: # Flatten sparse labels for Keras losses.
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

    if model_name in (
        "vae", 
        "variational_autoencoder", 
        "vae_classifier", 
        "dnn"
    ): # Flatten dense-model inputs.
        x_train = x_train.reshape((len(x_train), -1))
        x_eval = x_eval.reshape((len(x_eval), -1))

    trainset = get_dataset(
        x_train, 
        y_train, 
        pad=pad, 
        shuffle_buffer=shuffle_buffer, 
        batch_size=batch_size, 
        drop_remainder=task != "continual", 
        seed=seed
    )

    if config is not None: # Record the built training pipeline length.
        config.dataset.trainset_len = len(trainset)

    if task == "continual": # Defer per-task loading to the continual learner.
        return loader, None

    valset = get_dataset(
        x_eval, 
        y_eval, 
        pad=pad, 
        shuffle_buffer=0, 
        batch_size=batch_size, 
        drop_remainder=False
    ) if use_valset else None # Build a deterministic validation pipeline.

    return trainset, valset
