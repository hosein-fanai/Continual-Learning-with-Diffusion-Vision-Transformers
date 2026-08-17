"""CIFAR loading, class filtering, preprocessing, and ``tf.data`` helpers."""


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
    import numpy as np


    labels_set_list = []
    for labels_set in labels_list:
        allowed_instances_list = []
        for index in indices:
            allowed_instances = np.where(labels_set == index)[0].tolist()
            allowed_instances_list.extend(allowed_instances)

        labels_set_list.append(allowed_instances_list)
    
    return labels_set_list


def get_data(x_train, y_train, x_test, y_test, 
            class_num, indices, preprocess, 
            return_features, features_path, 
            onehot_labels, verbose):
    """Prepare filtered train, validation, and test NumPy arrays.

    Class filtering is applied in the order supplied by ``indices``.  Training
    data is then split reproducibly into 80% training and 20% validation data,
    stratified by label.  Normalization statistics always come from that final
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
        preprocess (str | None): ``"min-max"`` applies one global training-set
            minimum and maximum; ``"normalize"`` applies elementwise mean and
            standard deviation over axis 0.  Any other value, including
            ``None`` or ``""``, performs no scaling.  In raw-image mode the
            unscaled training and test arrays are cast to ``uint8``.
        return_features (bool): If true, load pre-extracted arrays instead of
            returning images.  ``features_path`` must identify CIFAR-10 or
            CIFAR-100 and contain three arrays in train/validation/test order.
        features_path (str | os.PathLike): Base path without the ``.npy``
            suffix.  Its text must contain ``"cifar10_"`` or ``"cifar100_"``
            in feature mode; :func:`common.utils.load_samples` appends the
            suffix.
        onehot_labels (bool): If true, convert each label array to ``float32``
            one-hot rows shaped ``[samples, class_num]``; otherwise retain its
            original integer-label rank and dtype.
        verbose (bool | int): Truthy values print shapes and simple label
            frequencies; falsy values suppress output.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray]: ``(x_train, y_train, x_val, y_val,
        x_test, y_test)`` after filtering, splitting, preprocessing, and
        optional one-hot conversion.

    Raises:
        Exception: If feature mode cannot infer CIFAR-10 or CIFAR-100 from the
            path.
        ValueError: If the requested classes cannot support a stratified 80/20
            split, or if loaded feature and label lengths are incompatible.
    """
    from tensorflow.keras.utils import to_categorical

    from sklearn.model_selection import train_test_split

    import numpy as np

    from common.utils import load_samples


    if return_features:
        if "cifar10_" in features_path:
            class_num = 10
        elif "cifar100_" in features_path:
            class_num = 100
        else:
            raise Exception("features_path has to contain cifar10_ or cifar100_.")

        labels_set_list = sort_filter_labels([y_train, y_test], list(range(class_num)))
        y_train = y_train[labels_set_list[0]]
        y_test = y_test[labels_set_list[1]]

        y_train, y_val = train_test_split(y_train, test_size=0.2, 
                                        stratify=y_train, random_state=42)

        x_train, x_val, x_test = load_samples(features_path, ".npy")

        x_train = np.concatenate([x_train, x_val], axis=0)
        y_train = np.concatenate([y_train, y_val], axis=0)

    labels_set_list = sort_filter_labels([y_train, y_test], indices)
    x_train, y_train = x_train[labels_set_list[0]], y_train[labels_set_list[0]]
    x_test, y_test = x_test[labels_set_list[1]], y_test[labels_set_list[1]]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, 
                                                    stratify=y_train, random_state=42)

    if preprocess == "min-max":
        min_ = x_train.min()
        max_ = x_train.max()

        x_train = (x_train - min_) / (max_ - min_)
        x_val = (x_val - min_) / (max_ - min_)
        x_test = (x_test - min_) / (max_ - min_)
    elif preprocess == "normalize":
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)

        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std
        x_test = (x_test - mean) / std
    else: # no preprocess
        if not return_features:
            x_train = x_train.astype("uint8")
            x_test = x_test.astype("uint8")

    if onehot_labels:
        y_train = to_categorical(y_train, num_classes=class_num)
        y_val = to_categorical(y_val, num_classes=class_num)
        y_test = to_categorical(y_test, num_classes=class_num)

    if verbose:
        print("Trainset:", x_train.shape, y_train.shape)
        print("Validation set:", x_val.shape, y_val.shape)
        print("Testset:", x_test.shape, y_test.shape)

        for set_id, dataset in enumerate((y_train, y_val, y_test)):
            print(f"---{set_id}")
            for clss_id in np.unique(dataset):
                print(clss_id, sum(dataset == clss_id) / len(dataset))
            
            print()

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_cifar10(indices=list(range(10)), preprocess=None, 
                features_path="./data/cifar10_xception_gavgpooled_features_train_val_test", 
                return_features=False, onehot_labels=False, verbose=1):
    """Load and prepare CIFAR-10 images or saved Xception features.

    Args:
        indices (Sequence[int]): Retained class IDs from ``0`` through ``9``.
            Defaults to every class; ordering controls grouping before split.
        preprocess (str | None): ``"min-max"``, ``"normalize"``, or any other
            value for no scaling; see :func:`get_data` for exact behavior.
        features_path (str | os.PathLike): Base path for a ``.npy`` archive of
            train/validation/test features.  Used only when
            ``return_features=True`` and must contain ``"cifar10_"``.
        return_features (bool): Return saved feature vectors rather than raw
            ``32 x 32 x 3`` images.
        onehot_labels (bool): Return 10-column one-hot labels when true; return
            integer class IDs when false.
        verbose (bool | int): Whether to print dataset summaries.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray]: Training, validation, and test feature/
        label pairs in that order.

    Example:
        ``load_cifar10(indices=[0, 1], preprocess="min-max",
        onehot_labels=True)`` returns only two classes, but one-hot rows still
        have width 10 because CIFAR-10's full class count is retained.
    """
    from tensorflow.keras.datasets import cifar10


    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    return get_data(x_train, y_train, x_test, y_test, 10, 
                indices, preprocess, return_features, 
                features_path, onehot_labels, verbose)


def load_cifar100(indices=list(range(100)), preprocess=None, 
                features_path="./data/cifar100_xception_gavgpooled_features_train_val_test", 
                return_features=False, onehot_labels=False, verbose=1):
    """Load and prepare CIFAR-100 fine-label images or saved features.

    Args:
        indices (Sequence[int]): Retained fine-label IDs from ``0`` through
            ``99``.  Defaults to all classes.
        preprocess (str | None): ``"min-max"``, ``"normalize"``, or another
            value for no scaling; see :func:`get_data`.
        features_path (str | os.PathLike): Base path for a ``.npy`` archive of
            train/validation/test features.  Used only in feature mode and must
            contain ``"cifar100_"``.
        return_features (bool): Return saved feature vectors instead of raw
            images when true.
        onehot_labels (bool): Return 100-column one-hot labels when true.
        verbose (bool | int): Whether to print dataset summaries.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray]: Training, validation, and test feature/
        label pairs in that order.
    """
    from tensorflow.keras.datasets import cifar100


    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    
    return get_data(x_train, y_train, x_test, y_test, 100,
                indices, preprocess, return_features, 
                features_path, onehot_labels, verbose)


def get_dataset(X, Y, conv_base=None, batch_size=128):
    """Create a cached, batched TensorFlow dataset from aligned arrays.

    Args:
        X (numpy.ndarray | tf.Tensor): Samples shaped ``[N, ...]``.
        Y (numpy.ndarray | tf.Tensor): Labels whose first dimension is ``N``.
        conv_base (Callable | None): Optional feature extractor called as
            ``conv_base(x_batch)`` after batching.  ``None`` leaves samples
            unchanged.  Labels always pass through unchanged.
        batch_size (int): Positive examples per batch.  The last partial batch
            is retained; no shuffling is performed.

    Returns:
        tf.data.Dataset: Elements are ``(x_batch, y_batch)``.  ``x_batch`` is
        either the original batched sample tensor or ``conv_base`` output.  The
        pipeline maps, prefetches with ``AUTOTUNE``, and caches on first full
        iteration.
    """
    import tensorflow as tf


    def preprocess_func(x, y):
        """Optionally transform one batch while retaining its labels.

        Args:
            x (tf.Tensor): A sample batch shaped ``[batch, ...]``.
            y (tf.Tensor): Its aligned label batch.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: ``(conv_base(x), y)`` when a feature
            extractor was supplied, otherwise ``(x, y)``.
        """
        if conv_base is not None:
            x = conv_base(x)

        return x, y

    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(preprocess_func)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.cache()

    return dataset

