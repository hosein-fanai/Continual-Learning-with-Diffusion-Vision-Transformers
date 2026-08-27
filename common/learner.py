"""Configurable class-incremental learning with replay alternatives."""

from __future__ import annotations

import tensorflow as tf

from sklearn.metrics import accuracy_score

import numpy as np

from collections.abc import Callable, Sequence
from typing import Any

from common.config import Config, resolve_continual_schedule
from common.utils import CL_plot
from common.model import get_model, copy_model, get_callbacks
from common.replay_buffer import ReplayBuffer
from common.dataloader import get_dataset, _limit_samples, _pad_images

from autoencoder import VariationalAutoencoder, VAEClassifier

from diffusion import (
    DiffusionModel,
    DiffusionClassifier,
    DiffusionClassifierV2,
    DiffusionTransformer,
    DiTClassifier,
    DiTDecoder,
    DiTEncoderDecoder,
    DiTEncoderDecoderClassifier,
    UNet,
    UNetClassifier
)


DatasetArrays = tuple[
    np.ndarray, np.ndarray, np.ndarray | None,
    np.ndarray | None, np.ndarray, np.ndarray
]
DatasetLoader = Callable[..., DatasetArrays]


def _prepare_diffusion_x(
    x: np.ndarray,
    data_min: float,
    data_range: float
) -> np.ndarray:
    """Map a supported loader representation to diffusion model space.

    Args:
        x (numpy.ndarray): Preprocessed image batch ``[samples, H, W, C]``.
        data_min (float): Minimum of the shared real training input.
        data_range (float): Nonzero maximum-minus-minimum training range.

    Returns:
        numpy.ndarray: Float32 inputs mapped so the training extrema are
        ``-1`` and ``1``. Held-out values are not clipped.
    """

    x = np.asarray(x, dtype="float32")

    return ((x - data_min) / data_range * 2.) - 1.


def _select_classes(
    x: np.ndarray,
    y: np.ndarray,
    classes: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Select rows for class IDs without changing preprocessing space.

    Args:
        x (numpy.ndarray): Preprocessed samples shaped ``[samples, ...]``.
        y (numpy.ndarray): Sparse or one-hot labels for ``x``.
        classes (Sequence[int]): Integer class IDs to retain.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Filtered sample and label arrays
        in their original dtypes and preprocessing coordinates.
    """

    labels = np.asarray(y)
    # Decode one-hot or probability rows into class identifiers.
    if labels.ndim > 1 and labels.shape[-1] > 1:
        label_ids = np.argmax(labels, axis=-1)
    # Flatten sparse scalar or column-shaped labels.
    else:
        label_ids = labels.reshape(-1)

    selected = np.isin(label_ids, classes)

    return np.asarray(x)[selected], labels[selected]


def _remap_continual_labels(
    labels: np.ndarray | None,
    class_order: Sequence[int],
    onehot_labels: bool,
) -> np.ndarray | None:
    """Map original dataset labels to their contiguous schedule positions.

    Args:
        labels (numpy.ndarray | None): Sparse or one-hot labels from a loader.
        class_order (Sequence[int]): Original labels in introduction order.
        onehot_labels (bool): Whether to return selected-width one-hot rows.

    Returns:
        numpy.ndarray | None: Labels encoded from zero through selected classes
        minus one, or ``None`` when the input split is absent.

    Raises:
        ValueError: If a loader returns a label outside ``class_order``.
    """

    # Preserve absent validation arrays without fabricating held-out data.
    if labels is None:
        return None

    label_array = np.asarray(labels)
    label_ids = np.argmax(label_array, axis=-1) \
        if label_array.ndim > 1 and label_array.shape[-1] > 1 \
        else label_array.reshape(-1)
    mapping = {label: index for index, label in enumerate(class_order)}
    try:
        remapped = np.asarray([mapping[label.item()] for label in label_ids])
    except KeyError as error:
        raise ValueError(
            f"Dataset returned unscheduled class label {error.args[0]!r}."
        ) from error

    # Rebuild categorical targets at the selected continual output width.
    if onehot_labels:
        return np.eye(len(class_order), dtype=label_array.dtype)[remapped]

    return remapped.astype(label_array.dtype).reshape(label_array.shape)


def _continual_metrics(
    accuracy_matrix: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Compute task-balanced summary metrics from a continual accuracy matrix.

    Args:
        accuracy_matrix (Sequence[Sequence[float]]): Lower-triangular matrix
            where row ``t`` contains performance on tasks learned through
            ``t`` and unavailable future entries are NaN.

    Returns:
        dict[str, float]: Final average accuracy, average incremental accuracy,
        average forgetting, and backward transfer. Single-task forgetting and
        backward transfer are zero because no prior task exists.
    """

    matrix = np.asarray(accuracy_matrix, dtype="float64")
    task_num = len(matrix)
    # Keep an empty schedule representable for defensive direct callers.
    if task_num == 0:
        return {
            "final_average_accuracy": np.nan,
            "average_incremental_accuracy": np.nan,
            "average_forgetting": np.nan,
            "backward_transfer": np.nan,
        }

    row_averages = [np.nanmean(matrix[index, :index + 1])
                    for index in range(task_num)]
    final_average_accuracy = float(np.nanmean(matrix[-1, :task_num]))
    # Forgetting and BWT are defined only over tasks learned before the last.
    if task_num == 1:
        average_forgetting = 0.
        backward_transfer = 0.
    # Compare every earlier task with its learned and best historical scores.
    else:
        final_old = matrix[-1, :task_num - 1]
        maxima = np.asarray([
            np.nanmax(matrix[index:, index])
            for index in range(task_num - 1)
        ])
        diagonal = np.asarray([
            matrix[index, index] for index in range(task_num - 1)
        ])
        average_forgetting = float(np.nanmean(maxima - final_old))
        backward_transfer = float(np.nanmean(final_old - diagonal))

    return {
        "final_average_accuracy": final_average_accuracy,
        "average_incremental_accuracy": float(np.nanmean(row_averages)),
        "average_forgetting": average_forgetting,
        "backward_transfer": backward_transfer,
    }


def _task_accuracy_summaries(
    accuracy_matrix: Sequence[Sequence[float]],
) -> tuple[list[float], list[float]]:
    """Extract current-task and prior-task macro accuracy after every task.

    Args:
        accuracy_matrix (Sequence[Sequence[float]]): Lower-triangular continual
            task accuracy matrix.

    Returns:
        tuple[list[float], list[float]]: Matrix diagonal and row-wise mean of
        earlier columns. The first old-task value is NaN by definition.
    """

    new_task_accuracy = [
        float(accuracy_matrix[index][index])
        for index in range(len(accuracy_matrix))
    ]
    old_task_accuracy = [
        np.nan if index == 0 else float(np.nanmean(
            accuracy_matrix[index][:index]
        ))
        for index in range(len(accuracy_matrix))
    ]

    return new_task_accuracy, old_task_accuracy


def _predict_diffusion_classes(
    model: DiffusionClassifier,
    x: np.ndarray,
    y: np.ndarray,
    data_min: float,
    data_range: float,
    batch_size: int
) -> np.ndarray:
    """Predict class scores from a diffusion classifier in data batches.

    Args:
        model (DiffusionClassifier): Trained diffusion classifier wrapper.
        x (numpy.ndarray): Preprocessed images ``[samples, H, W, C]``.
        y (numpy.ndarray): Integer labels ``[samples]`` used by V2 noising.
        data_min (float): Current task training-input minimum.
        data_range (float): Current nonzero training-input range.
        batch_size (int): Positive prediction batch size.

    Returns:
        numpy.ndarray: Class scores shaped ``[samples, class_num]``.
    """

    x = _prepare_diffusion_x(x, data_min, data_range)
    y = np.asarray(y).reshape(-1)
    network = model.get_network(model.test_network_name)
    predictions = []

    for start in range(0, len(x), batch_size):
        end = start + batch_size
        x_batch = x[start:end]

        # Build V2's configured noisy classifier input.
        if isinstance(model, DiffusionClassifierV2):
            t_batch, x_batch, null_labels, _ = model.prep_clfv2_inputs(
                (x_batch, y[start: end]),
                model.clf_test_noisified_max_timesteps
            )
        # Evaluate standard diffusion classifiers at clean timestep zero.
        else:
            t_batch = np.zeros((len(x_batch),), dtype="int32")
            null_labels = np.zeros((len(x_batch),), dtype="uint8")

        predictions.append(network.predict_class(
            (x_batch, t_batch, null_labels),
            training=False
        ).numpy())

    return np.concatenate(predictions, axis=0)


def _reported_accuracy(
    evaluations: object,
    ensemble: bool = False
) -> float | None:
    """Find a classifier accuracy value in nested report output.

    Args:
        evaluations (object): Possibly nested report result.
        ensemble (bool): Search only for the exact
            ``"ensemble_accuracy"`` key when true.

    Returns:
        float | None: First recognized scalar accuracy, if present.
    """

    # Treat nonmapping evaluation output as unavailable for named lookup.
    if not isinstance(evaluations, dict):
        return None

    preferred = ("ensemble_accuracy",) if ensemble else (
        "total_accuracy",
        "accuracy",
        "classifier_accuracy",
        "cls_token_accuracy",
        "avg_pooling_accuracy",
        "clf_accuracy",
        "discriminator_accuracy"
    )

    for name in preferred:
        # Prefer explicitly named accuracy metrics in priority order.
        if name in evaluations:
            return float(evaluations[name])

    # Preserve the broad legacy fallback only for ordinary accuracy.
    if not ensemble:
        for name, value in evaluations.items():
            # Fall back to any scalar metric whose name denotes accuracy.
            if "accuracy" in name.lower() and np.isscalar(value):
                return float(value)

    for value in evaluations.values():
        accuracy = _reported_accuracy(value, ensemble=ensemble)
        # Return the first usable nested accuracy value.
        if accuracy is not None:
            return accuracy

    return None


def _copy_classifier_prefix(
    source_model: tf.keras.Model,
    target_model: tf.keras.Model
) -> None:
    """Copy shared weights and the target-width classifier head prefix.

    Args:
        source_model (tf.keras.Model): Built full-width source classifier.
        target_model (tf.keras.Model): Built task-width target classifier.

    Returns:
        None: ``target_model`` is updated in place.
    """

    # Require matching layer structures before copying a classifier prefix.
    if len(source_model.layers) != len(target_model.layers):
        raise ValueError(
            "Initial and continual classifiers must have matching layers."
        )

    for source_layer, target_layer in zip(
        source_model.layers[:-1],
        target_model.layers[:-1]
    ):
        target_layer.set_weights(source_layer.get_weights())

    source_kernel, source_bias = source_model.layers[-1].get_weights()
    target_kernel, target_bias = target_model.layers[-1].get_weights()
    target_width = target_bias.shape[0]

    # Require the source head to cover every target output column.
    if source_bias.shape[0] < target_width:
        raise ValueError(
            "Initial classifier output is narrower than a continual task."
        )

    target_kernel[...] = source_kernel[..., :target_width]
    target_bias[...] = source_bias[:target_width]
    target_model.layers[-1].set_weights([target_kernel, target_bias])


def _has_positive_distillation_objective(
    model: DiffusionClassifier
) -> bool:
    """Return whether a diffusion classifier has an active teacher objective.

    Args:
        model (DiffusionClassifier): Wrapper whose token and regularizer losses
            are inspected before the first continual task.

    Returns:
        bool: True when token distillation or a teacher-backed classifier-token
        regularizer has a positive coefficient.
    """

    distil_loss_coef = float(tf.keras.backend.get_value(
        model.distil_loss_coef
    ))
    ctr_loss_coef = float(tf.keras.backend.get_value(model.ctr_loss_coef))
    regularizer_kwargs = getattr(
        model.network, "clf_cls_token_regularizer_kwargs", None
    )
    # U-Net classifiers keep the same metadata on the inherited attribute.
    if regularizer_kwargs is None:
        regularizer_kwargs = getattr(
            model.network, "cls_token_regularizer_kwargs", {}
        )

    uses_teacher_regularizer = (
        ctr_loss_coef > 0.
        and regularizer_kwargs.get("train_type", "normal") in (
            "distil", "both"
        )
    )
    return distil_loss_coef > 0. or uses_teacher_regularizer


def _load_continual_arrays(
    load_dataset_fn: DatasetLoader,
    class_num: int,
    class_order: Sequence[int],
    return_features: bool,
    load_dataset_fn_kwargs: dict[str, object],
    max_train_samples: int | None,
    max_val_samples: int | None,
    pad: int,
    dataset_seed: int | None
) -> tuple[DatasetArrays, np.random.Generator]:
    """Load and prepare the shared array view used by every task.

    Args:
        load_dataset_fn (DatasetLoader): Full-array dataset loader.
        class_num (int): Number of selected classes used by the run.
        class_order (Sequence[int]): Original dataset labels in task order.
        return_features (bool): Whether the loader should return saved features.
        load_dataset_fn_kwargs (dict[str, object]): Loader preprocessing options.
        max_train_samples (int | None): Optional shared training-row limit.
        max_val_samples (int | None): Optional validation-row limit.
        pad (int): Symmetric image padding width.
        dataset_seed (int | None): Seed for limiting and later task sampling.

    Returns:
        tuple[DatasetArrays, numpy.random.Generator]: Prepared arrays and the
        run-local generator after shared sample limiting.
    """

    (all_x_train, all_y_train, all_x_val, all_y_val,
     all_x_test, all_y_test) = load_dataset_fn(
        indices=list(class_order),
        return_features=return_features,
        **load_dataset_fn_kwargs,
        verbose=0
    )

    rng = np.random.default_rng(dataset_seed)
    all_x_train, all_y_train = _limit_samples(
        all_x_train,
        all_y_train,
        max_train_samples,
        rng
    )
    # Limit the validation split once in the shared preprocessing space.
    if all_x_val is not None:
        all_x_val, all_y_val = _limit_samples(
            all_x_val,
            all_y_val,
            max_val_samples,
            rng
        )

    # Pad raw images once in the same coordinate space as the loader.
    if pad > 0:
        pad_value = -1. if str(
            load_dataset_fn_kwargs["preprocess"]
        ).lower() in ("standardize", "diffusion") else 0.
        all_x_train = _pad_images(
            np.asarray(all_x_train), pad, value=pad_value
        )
        all_x_test = _pad_images(
            np.asarray(all_x_test), pad, value=pad_value
        )
        # Apply matching padding to an available validation split.
        if all_x_val is not None:
            all_x_val = _pad_images(
                np.asarray(all_x_val), pad, value=pad_value
            )

    # Map arbitrary original labels into classifier positions in schedule order.
    label_arrays = [
        _remap_continual_labels(
            labels,
            class_order,
            load_dataset_fn_kwargs["onehot_labels"],
        )
        for labels in (all_y_train, all_y_val, all_y_test)
    ]
    all_y_train, all_y_val, all_y_test = label_arrays

    return (
        (
            all_x_train, all_y_train, all_x_val, all_y_val,
            all_x_test, all_y_test
        ),
        rng
    )


def _run_continual_tasks(
    class_num: int,
    load_dataset_fn: DatasetLoader,
    class_order: Sequence[int] | None = None,
    task_groups: Sequence[Sequence[int]] | None = None,
    load_dataset_fn_kwargs: dict[str, object] | None = None, 
    remove_prev_classes: bool = True, 
    keep_same_model: bool = True, 
    tuned_model_path: str = "", 
    compile_args: dict[str, object] | None = None, 
    use_loaded_opt: bool = False, 
    batch_size: int = 128, 
    epochs: int = 100, 
    fit_method: str = "fit",
    fit_kwargs: dict[str, object] | None = None,
    use_buffer: bool = False, 
    buffer_kwargs: dict[str, object] | None = None, 
    plot_results: bool = True, 
    verbose: bool | int = True, 
    generative_model: tf.keras.Model | None = None, 
    teacher_network: tf.keras.Model | None = None,
    generative_model_compile_args: dict[str, object] | None = None, 
    generative_model_kwargs: dict[str, int] | None = None, 
    use_generative_model_classifier: bool = False, 
    train_classifier_separately: bool = False, 
    use_distillation: bool = False,
    evaluate_ensemble_accuracy: bool = False, 
    ensemble_accuracy_kwargs: dict[str, object] | None = None, 
    callbacks_list: Sequence[tf.keras.callbacks.Callback] | None = None, 
    return_details: bool = True, 
    use_valset: bool = True, 
    return_features: bool | None = None, 
    max_train_samples: int | None = None, 
    max_val_samples: int | None = None, 
    shuffle_buffer: int | None = None, 
    pad: int = 0, 
    dataset_seed: int | None = None, 
    initial_classifier: tf.keras.Model | None = None, 
    callback_patience: int | None = None, 
    callback_monitor: str | None = None, 
    callback_monitor_mode: str | None = None
) -> list[float] | dict[str, object]:
    """Run classifier training over a configurable class-incremental schedule.

    Standalone classifier heads are expanded between tasks. A diffusion
    classifier initialized with ``num_classes=None`` instead grows its attached
    head in place as labels are discovered. Optional replay comes either from a
    fixed-size sample buffer or a conditional generative model; the two modes
    are mutually exclusive.

    Args:
        class_num (int): Total selected class count ``N``.
        class_order (Sequence[int] | None): Original dataset labels in their
            introduction order. ``None`` uses ``range(N)``.
        task_groups (Sequence[Sequence[int]] | None): Original labels introduced
            per task. ``None`` uses ``[0, 1]`` and then singleton tasks. When
            both schedule arguments are supplied, flattened groups must match
            ``class_order`` exactly.
        load_dataset_fn (Callable[..., tuple[numpy.ndarray, ...]]): Loader called
            with ``indices``, ``return_features``, ``preprocess``,
            ``onehot_labels``, and ``verbose``.  It must return exactly
            ``(x_train, y_train, x_val, y_val, x_test, y_test)``.  The built-in
            :func:`common.dataloader.load_cifar10` and ``load_cifar100`` satisfy
            this interface.
        load_dataset_fn_kwargs (dict[str, object] | None): Loader options merged
            over ``{"preprocess": None, "onehot_labels": False}``. For built-in
            loaders optional keys also include ``features_path``,
            ``validation_ratio`` (float), and ``seed`` (int | None). Valid
            ``preprocess`` values are ``"normalize"``, ``"min-max"``,
            ``"standardize"``/``"diffusion"``, or ``""``/``None`` for no
            scaling; ``onehot_labels`` is bool.
            Do not include ``indices``, ``return_features``, or ``verbose``
            because this function passes them explicitly.  A custom loader may
            accept other keys. VAE replay requires ``onehot_labels=True`` and
            accepts every listed preprocessing value.
        remove_prev_classes (bool): If true, training receives classes 0 and 1
            at the first task and only the newly introduced class thereafter;
            validation/test still contain all seen classes.  If false, every
            split contains all classes seen so far.
        keep_same_model (bool): Copy learned non-head weights and old class-head
            columns into the next, one-class-wider model when true.  False uses
            a freshly cloned tuned architecture at each task. Ignored when the
            generative model's classifier is selected.
        tuned_model_path (str): Nonempty saved Keras model path.
            If its case-insensitive text contains ``"dnn"``, the loader is asked
            for saved 2,048-wide features; otherwise it is asked for images.
        compile_args (dict[str, object] | None): Overrides the classifier defaults
            accepted by ``tf.keras.Model.compile``, such as ``optimizer``,
            ``loss``, ``metrics``, ``loss_weights``, or ``run_eagerly``.
            ``None`` uses the existing defaults. Example:
            ``{"optimizer": "adam", "metrics": ["accuracy"]}``.
        use_loaded_opt (bool): Inherit the optimizer deserialized from the tuned
            model instead of ``compile_args["optimizer"]``.
        batch_size (int): Positive batch size used by each newly built
            ``tf.data.Dataset``; defaults to 128.
        epochs (int): Positive maximum epochs per ordinary fit phase; defaults
            to 100. Progressive diffusion replay instead uses ``stage_epochs``
            and ``final_epochs`` from ``fit_kwargs``.
        fit_method (str): ``"fit"`` or ``"fit_progressively"`` for the
            diffusion replay-model phase. Standalone classifier and VAE phases
            retain their established fit methods. For
            ``DiffusionClassifierV2``, progressive selection affects its
            generator phase while the discriminator remains an ordinary fit.
        fit_kwargs (dict[str, object] | None): Extra arguments copied into each
            diffusion replay-model fit. Progressive arguments are those of
            ``DiffusionModel.fit_progressively``, including ``stage_tasks``,
            ``stage_epochs``, and ``final_epochs``. The curriculum repeats for
            every continual task; timestep/resolution state is restored after
            each call, while requested depth growth intentionally persists.
        use_buffer (bool): Enable fixed-capacity replay.  It must not be true
            together with ``generative_model``.
        buffer_kwargs (dict[str, object] | None): Replay controls merged over
            ``{"maxlen": 10000, "sample_num": 1000, "insert_num": 1000,
            "seed": None}``.  ``maxlen`` is deque capacity; ``sample_num`` is
            the maximum prior pairs concatenated before a task; ``insert_num``
            is the number sampled from that task's augmented training arrays
            after fitting; and ``seed`` initializes the buffer's private random
            generator without changing global random state. Extra keys are
            retained but unused. Example: ``{"maxlen": 5000,
            "sample_num": 500, "insert_num": 500, "seed": 42}``.
        plot_results (bool): Plot accuracy against the number of seen classes
            after all tasks.
        verbose (bool | int): Print task summaries, Keras progress, history
            figures, classification reports, and confusion matrices when
            truthy.
        generative_model (tf.keras.Model | None): Optional already-created VAE,
            diffusion wrapper, or raw diffusion network used for generative
            replay. Raw classifier networks are connected to
            ``DiffusionClassifier`` and all other supported diffusion networks
            to ``DiffusionModel``. Pass a compiled wrapper directly
            when custom wrapper or optimizer settings are needed. This cannot
            be combined with ``use_buffer``. Diffusion replay requires image
            data, requires a network initialized with ``num_classes=None``, and
            accepts every loader preprocessing value.
        teacher_network (tf.keras.Model | None): Runtime-only frozen teacher
            used for the first task when a diffusion classifier is wrapped.
            Automatic continual distillation replaces it with the completed
            task-one student before task two.
        generative_model_compile_args (dict[str, object] | None): Compilation
            values used when this function wraps a raw diffusion network.
            Values override ``{"optimizer": "adam", "loss": "mse"}``.
            Already-wrapped models keep their existing compilation. ``None``
            uses the defaults.
        generative_model_kwargs (dict[str, int] | None): Generative replay controls
            merged over ``{"train_num": 1000, "samples_per_class": 1000}``.
            ``samples_per_class`` sets prior generations per seen class.
            ``train_num=-1`` fits current data without resampling; any positive
            value samples exactly that many current-task rows with replacement.
        use_generative_model_classifier (bool): Use the classifier attached to
            a ``VAEClassifier`` or the classifier branch of a
            ``DiffusionClassifier`` as the continually learned model. A VAE
            classifier keeps its fixed-width head; a diffusion classifier adds
            one output for each newly observed label. A joint-only VAE task
            reports reconstruction-based classifier accuracy through the VAE;
            a separately trained classifier reports its direct-input accuracy.
        train_classifier_separately (bool): Give the selected classifier its
            own training step in addition to generative training. This remains
            optional for ``VAEClassifier`` and requires its classifier to be
            compiled. It must be false for ``DiffusionClassifier`` and true for
            ``DiffusionClassifierV2`` because V2 separates its generator and
            classifier variables. It has no effect when
            ``use_generative_model_classifier`` is false.
        use_distillation (bool): Use an independent frozen snapshot of each
            completed raw student as the next task's teacher. The replay model
            must be a diffusion classifier with a distillation token and a
            positive teacher objective. Task one is teacher-free unless
            ``teacher_network`` is supplied explicitly. A raw classifier
            network receives the wrapper's established ``8.6e-3`` classifier
            coefficient for its distillation objective.
        evaluate_ensemble_accuracy (bool): Also evaluate the attached
            diffusion classifier by ensembling predictions across timesteps
            after every task. Ordinary task accuracy is still retained.
        ensemble_accuracy_kwargs (dict[str, object] | None): Options forwarded
            to ``DiffusionClassifier.evaluate_ensemble_accuracy``. The report
            selects raw and EMA networks itself.
        callbacks_list (Sequence[tf.keras.callbacks.Callback] | None): Extra
            callbacks appended to each enabled incremental classifier fit and
            passed to generative-model fits. This is primarily useful for
            experiment logging; ``None`` preserves the original callback
            behavior.
        return_details (bool): Return accuracies, histories, and the final
            classifier/generator objects when true. The default keeps the
            original accuracy-list return value.
        use_valset (bool): Build and use a fresh validation dataset for every
            task when true and the loader created an explicit validation split.
            False disables task validation. Test rows are never substituted.
        return_features (bool | None): Internal factory override for configured
            runs. ``None`` preserves direct mode's legacy path-name inference.
        max_train_samples (int | None): Internal configured limit applied once
            to the loader's full training arrays before task selection.
        max_val_samples (int | None): Internal configured limit applied only to
            independently created validation arrays.
        shuffle_buffer (int | None): Internal configured training shuffle
            capacity. ``None`` preserves the legacy full-task shuffle.
        pad (int): Internal configured symmetric image padding applied before
            task selection and replay.
        dataset_seed (int | None): Seed for configured limiting and shuffling.
        initial_classifier (tf.keras.Model | None): Optional configured
            classifier whose trunk and visible head columns initialize tasks.
        callback_patience (int | None): Internal configured early-stopping
            patience. ``None`` preserves direct mode's legacy value of 5;
            ``0`` disables early stopping.
        callback_monitor (str | None): Internal configured metric override.
        callback_monitor_mode (str | None): Internal configured Keras monitor
            direction. ``None`` preserves each phase's legacy direction.
    Returns:
        list[float] | dict[str, object]: Cumulative test accuracy after each
        configured task. When ``return_details=True``, returns
        those accuracies plus optional ensemble accuracies, task histories,
        report evaluations, the task accuracy matrix, standard continual
        metrics, schedule metadata, and final model objects.

    Raises:
        ValueError: If buffer and generative replay are both enabled,
            ``fit_method`` is unsupported, or progressive fitting is selected
            without a diffusion replay model and ``stage_tasks``.
        TypeError: If a forwarded dictionary contains a conflicting or
            unsupported keyword, or ``generative_model`` is unsupported.
        ValueError: If dataset shapes/labels cannot support the requested
            task, replay, or classifier loss.
    """

    class_order, original_task_groups = resolve_continual_schedule(
        class_num,
        list(class_order) if class_order is not None else None,
        [list(group) for group in task_groups]
        if task_groups is not None else None,
    )
    class_num = len(class_order)
    group_sizes = [len(group) for group in original_task_groups]
    boundaries = np.cumsum([0, *group_sizes])
    internal_task_groups = [
        list(range(int(boundaries[index]), int(boundaries[index + 1])))
        for index in range(len(group_sizes))
    ]

    # Restrict the shared selector to the two supported diffusion fit paths.
    if fit_method not in ("fit", "fit_progressively"):
        raise ValueError(
            "fit_method must be 'fit' or 'fit_progressively'."
        )

    fit_kwargs = dict(fit_kwargs or {})
    reserved_fit_keys = {
        "x", "y", "epochs", "initial_epoch", "validation_data",
        "callbacks", "verbose",
    }
    conflicting_fit_keys = sorted(reserved_fit_keys.intersection(fit_kwargs))
    # Keep task-owned data, epoch, callback, and verbosity values unambiguous.
    if conflicting_fit_keys:
        raise ValueError(
            "fit_kwargs cannot override continual orchestration arguments: "
            + str(conflicting_fit_keys)
        )
    # Require the curriculum description used by every progressive task.
    if fit_method == "fit_progressively" \
    and fit_kwargs.get("stage_tasks") is None:
        raise ValueError(
            "fit_kwargs must include stage_tasks for fit_progressively."
        )

    # Keep fixed-buffer and generative replay mutually exclusive.
    if use_buffer and generative_model is not None:
        raise ValueError(
            "The replay buffer and a generative model cannot be used together."
        )

    # Import lazily to avoid the train -> learner -> train module cycle.
    from common.train import report, train_model


    def train_task_model(
        model: tf.keras.Model, 
        trainset: object, 
        valset: object | None = None, 
        task_callbacks: Sequence[tf.keras.callbacks.Callback] | None = None, 
        fit_method: str = "fit", 
        fit_kwargs: dict[str, object] | None = None, 
    ) -> dict[str, list[float]]:
        """Train one continual phase through the shared training API.

        Args:
            model (tf.keras.Model): Compiled phase model.
            trainset (object): Training input accepted by the selected method.
            valset (object | None): Optional validation input.
            task_callbacks (Sequence[tf.keras.callbacks.Callback] | None):
                Extra callbacks for this phase.
            fit_method (str): Model method selected by ``train_model``.
            fit_kwargs (dict[str, object] | None): Extra selected-method
                arguments.

        Returns:
            dict[str, list[float]]: Per-epoch metric history.
        """

        return train_model(
            None, 
            model, 
            trainset, 
            valset=valset, 
            save_config_=False, 
            extra_callbacks=task_callbacks, 
            epochs=epochs, 
            verbose=verbose, 
            results_path=None, 
            show_images=True, 
            save_gifs=False, 
            report_every_epoch=False, 
            save_weights=False, 
            fit_method=fit_method, 
            fit_kwargs=fit_kwargs or {}
        )


    def report_task_model(
        history: dict[str, list[float]], 
        model: tf.keras.Model, 
        trainset: object, 
        testset: object
    ) -> dict[str, object]:
        """Report one continual phase through the shared reporting API.

        Args:
            history (dict[str, list[float]]): Phase metric history.
            model (tf.keras.Model): Trained phase model.
            trainset (object): Phase training input.
            testset (object): Phase evaluation input.

        Returns:
            dict[str, object]: Evaluation values and report metadata.
        """

        return report(
            None, 
            history, 
            model, 
            trainset, 
            valset=testset, 
            results_path=None, 
            save_history_plot=False, 
            save_csv=False, 
            show_history_plot=bool(verbose), 
            plot_without_20percent=False, 
            run_trainset_eval=False, 
            run_valset_eval=True, 
            save_final_images=False, 
            show_final_images=False, 
            save_final_gifs=False, 
            evaluate_ensemble_accuracy=(
                evaluate_ensemble_accuracy and 
                isinstance(model, DiffusionClassifier)
            ), 
            ensemble_accuracy_kwargs=ensemble_accuracy_kwargs, 
            verbose=verbose
        )


    def phase_callbacks(
        default_monitor: str, 
        default_mode: str = "max", 
        legacy_patience: int = 0
    ) -> list[tf.keras.callbacks.Callback]:
        """Build phase-specific early stopping plus caller callbacks.

        Args:
            default_monitor (str): Metric used without an explicit override.
            default_mode (str): Keras monitor direction.
            legacy_patience (int): Direct-mode fallback patience.

        Returns:
            list[tf.keras.callbacks.Callback]: Newly assembled callbacks.
        """

        patience = legacy_patience if callback_patience is None else callback_patience
        selected = []

        # Add early stopping when a positive patience was requested.
        if patience > 0:
            selected = get_callbacks(
                monitor=callback_monitor or default_monitor, 
                mode=callback_monitor_mode or default_mode, 
                patience=patience, 
                verbose=verbose
            )

        # Append caller-provided callbacks after shared defaults.
        if callbacks_list is not None:
            selected += list(callbacks_list)

        return selected


    load_dataset_fn_kwargs_default = {
        "preprocess": None, 
        "onehot_labels": False
    }
    load_dataset_fn_kwargs = {
        **load_dataset_fn_kwargs_default, 
        **(load_dataset_fn_kwargs or {})
    }

    compile_args = dict(compile_args or {})
    ensemble_accuracy_kwargs = dict(ensemble_accuracy_kwargs or {})
    generative_model_compile_args = {
        "optimizer": "adam", 
        "loss": "mse", 
        **(generative_model_compile_args or {})
    }

    buffer_kwargs_default = {
        "maxlen": 10_000, 
        "sample_num": 1_000, 
        "insert_num": 1_000, 
        "seed": None
    }
    buffer_kwargs = {
        **buffer_kwargs_default, 
        **(buffer_kwargs or {})
    }

    generative_model_kwargs_default = {
        "train_num": 1_000, 
        "samples_per_class": 1_000
    }
    generative_model_kwargs = {
        **generative_model_kwargs_default, 
        **(generative_model_kwargs or {})
    }

    # Infer legacy saved-feature use from the tuned model path.
    if return_features is None:
        return_features = "dnn" in str(tuned_model_path).lower()
    # Normalize an explicit feature-return switch.
    else:
        return_features = bool(return_features)

    # Prevent image padding from being applied to saved feature vectors.
    if pad and return_features:
        raise ValueError("pad is not supported for saved feature inputs.")

    # Create fixed-capacity replay storage when requested.
    if use_buffer:
        buffer = ReplayBuffer(
            maxlen=buffer_kwargs["maxlen"], 
            seed=buffer_kwargs["seed"]
        )

    # Wrap raw diffusion classifiers for joint replay training.
    if isinstance(generative_model, (
        DiTClassifier, 
        DiTEncoderDecoderClassifier, 
        UNetClassifier
    )):
        generative_model = DiffusionClassifier(
            network=generative_model, 
            teacher_network=teacher_network,
            defer_teacher=use_distillation,
            distil_loss_coef=8.6e-3 if use_distillation else 0.,
            mask_by_nulls=generative_model.use_cfg, 
            test_steps=min(50, generative_model.timesteps)
        )
        generative_model.compile(
            **generative_model_compile_args
        )
    # Wrap raw diffusion generator networks for training and replay.
    elif isinstance(generative_model, (
        DiTDecoder, 
        DiTEncoderDecoder, 
        DiffusionTransformer, 
        UNet
    )): # Wrap raw generator-only diffusion networks.
        generative_model = DiffusionModel(
            network=generative_model, 
            test_steps=min(50, generative_model.timesteps)
        )
        generative_model.compile(**generative_model_compile_args)

    # Install an optional first-task teacher on an existing wrapper.
    elif teacher_network is not None and isinstance(
        generative_model, DiffusionClassifier
    ) and getattr(generative_model, "teacher_network", None) is not teacher_network:
        generative_model.set_teacher_network(teacher_network)
    # Reject unsupported replay-model types before task construction.
    elif generative_model is not None and not isinstance(
        generative_model, 
        (VariationalAutoencoder, DiffusionModel)
    ):  # Reject objects that cannot provide supported generative replay.
        raise TypeError(
            "generative_model must be a supported VAE, "
            "diffusion network, or diffusion wrapper."
        )

    # Reject teachers for replay models without a distillation-capable wrapper.
    if teacher_network is not None and not isinstance(
        generative_model, DiffusionClassifier
    ):
        raise ValueError(
            "teacher_network requires a diffusion classifier generative_model."
        )

    # Restrict automatic self-distillation to classifier diffusion wrappers.
    if use_distillation and not isinstance(
        generative_model, DiffusionClassifier
    ):
        raise ValueError(
            "use_distillation requires a diffusion classifier generative_model."
        )
    # Require an independent student head for continual self-distillation.
    if use_distillation and getattr(
        generative_model.network, "distil_token", None
    ) is None:
        raise ValueError(
            "use_distillation requires the diffusion classifier to have a "
            "distil_token."
        )
    # Avoid retaining snapshots when no configured objective can consume them.
    if use_distillation and not _has_positive_distillation_objective(
        generative_model
    ):
        raise ValueError(
            "use_distillation requires a positive distillation objective."
        )

    # Prevent a progressive selector from being silently ignored by VAE,
    # classifier-only, or fixed-buffer continual training.
    if fit_method == "fit_progressively" and not isinstance(
        generative_model, DiffusionModel
    ):
        raise ValueError(
            "fit_progressively requires a diffusion replay model."
        )

    # Continual diffusion always uses the dynamic class-vocabulary API.
    if isinstance(generative_model, DiffusionModel) and not getattr(
        generative_model.network, "dynamic_num_classes", False
    ):
        raise ValueError(
            "Continual diffusion networks must be initialized with "
            "num_classes=None."
        )

    # Require the classifier wrapper that owns ensemble evaluation when enabled.
    if evaluate_ensemble_accuracy and not isinstance(
        generative_model, DiffusionClassifier
    ):
        raise ValueError(
            "evaluate_ensemble_accuracy requires a DiffusionClassifier "
            "or DiffusionClassifierV2 generative_model."
        )

    uses_attached_classifier = use_generative_model_classifier
    use_diffusion_classifier = uses_attached_classifier and isinstance(
        generative_model, DiffusionClassifier
    )
    # Force image inputs for diffusion classifier branches.
    if use_diffusion_classifier:
        return_features = False

    # Validate conditional VAE replay inputs.
    if isinstance(generative_model, VariationalAutoencoder):
        # Require class conditioning for VAE replay generation.
        if not generative_model.conditioned:
            raise ValueError("A replay VAE must be conditioned.")
        # Require one-hot labels consumed by conditional VAEs.
        if not load_dataset_fn_kwargs["onehot_labels"]:
            raise ValueError("VAE replay requires one-hot labels.")
    # Validate image input for diffusion replay.
    elif isinstance(generative_model, DiffusionModel):
        # Prevent diffusion networks from consuming saved flat features.
        if return_features:
            raise ValueError("Diffusion replay requires image data.")

    # Require a supported classifier-bearing replay model when selected.
    if uses_attached_classifier and not isinstance(
        generative_model, 
        (VAEClassifier, DiffusionClassifier)
    ): # Validate requests to reuse a generator-attached classifier.
        raise ValueError(
            "use_generative_model_classifier requires "
            "a VAEClassifier or DiffusionClassifier."
        )

    # Select and validate a diffusion classifier head.
    if use_diffusion_classifier:
        # Require separate classifier training for the V2 diffusion wrapper.
        if isinstance(generative_model, DiffusionClassifierV2) \
        and not train_classifier_separately:  # V2 trains discriminator variables separately.
            raise ValueError(
                "train_classifier_separately must "
                "be True for DiffusionClassifierV2."
            )
        # Keep separate classifier fitting disabled for joint diffusion wrappers.
        if not isinstance(generative_model, DiffusionClassifierV2) \
        and train_classifier_separately:  # Standard wrappers train both parts jointly.
            raise ValueError(
                "train_classifier_separately must "
                "be False for DiffusionClassifier."
            )

        prev_model = generative_model.network.classifier
    # Select and validate a VAE classifier head.
    elif uses_attached_classifier:
        # Require a Keras classifier when a separate fit phase is requested.
        if not isinstance(generative_model.classifier, tf.keras.Model):
            raise TypeError(
                "The VAEClassifier classifier must be a Keras model."
            )
        # Require the selected VAE classifier to be compiled before separate fit.
        if train_classifier_separately and getattr(
            generative_model.classifier, 
            "optimizer", None
        ) is None:  # A separate classifier fit requires prior compilation.
            raise ValueError(
                "The VAEClassifier classifier must be compiled "
                "before its separate training step."
            )

        prev_model = generative_model.classifier
    # Build the initial standalone continual classifier.
    else:
        prev_model = get_model(
            1, 
            model_type="hp-tuned", 
            model_path=tuned_model_path, 
            compile_args=compile_args, 
            use_loaded_opt=use_loaded_opt, 
            verbose=0
        )

    train_direct_classifier = not uses_attached_classifier or (
        train_classifier_separately and not use_diffusion_classifier
    )
    score_from_generator = use_diffusion_classifier or (
        uses_attached_classifier and not train_classifier_separately
    )

    acc_list = []
    ensemble_acc_list = []
    histories = []
    generative_histories = []
    classifier_evaluations_list = []
    generative_evaluations_list = []
    accuracy_matrix = []
    dataset_arrays = None
    for task_index, new_classes in enumerate(internal_task_groups):
        seen_classes = [
            label
            for group in internal_task_groups[:task_index + 1]
            for label in group
        ]
        seen_class_num = len(seen_classes)
        # Print the classes visible in the current continual task.
        if verbose:
            print(
                75*'-' + " Classes:",
                [
                    label
                    for group in original_task_groups[:task_index + 1]
                    for label in group
                ],
            )

        # Freeze the preceding student before the next class can expand it.
        if use_distillation and task_index > 0:
            previous_teacher = generative_model.snapshot_teacher_network(
                network_name="raw"
            )
            generative_model.set_teacher_network(previous_teacher)

        # Reuse the diffusion wrapper's classifier.
        if use_diffusion_classifier:
            new_model = generative_model.network.classifier
        # Reuse the VAE's classifier.
        elif uses_attached_classifier:
            new_model = generative_model.classifier
        # Build a classifier head sized for all classes seen in this task.
        else:
            new_model = get_model(
                seen_class_num, model_type="hp-tuned",
                model_path=tuned_model_path, 
                compile_args=compile_args, 
                use_loaded_opt=use_loaded_opt, 
                verbose=0
            )

            # Seed a fresh task head from the configured initial classifier.
            if initial_classifier is not None and (
                task_index == 0 or not keep_same_model
            ):
                _copy_classifier_prefix(initial_classifier, new_model)

            # Carry learned trunk weights and visible head columns forward.
            elif keep_same_model:
                copy_model(prev_model, new_model)

        # Fit one shared preprocessing space for all continual tasks.
        if dataset_arrays is None:
            dataset_arrays, rng = _load_continual_arrays(
                load_dataset_fn,
                class_num,
                class_order,
                return_features,
                load_dataset_fn_kwargs,
                max_train_samples,
                max_val_samples,
                pad,
                dataset_seed
            )

        (all_x_train, all_y_train, all_x_val,
        all_y_val, all_x_test, all_y_test) = dataset_arrays
        # Train later tasks on only their newly introduced group.
        if remove_prev_classes and task_index > 0:
            train_classes = new_classes
        # Train the first task, or every cumulative task, on all seen classes.
        else:
            train_classes = seen_classes

        x_train, y_train = _select_classes(
            all_x_train, 
            all_y_train, 
            train_classes
        )
        x_test, y_test = _select_classes(
            all_x_test, 
            all_y_test, 
            seen_classes
        )
        # Select seen validation rows in the shared space.
        if all_x_val is not None and all_y_val is not None:
            x_val, y_val = _select_classes(
                all_x_val, 
                all_y_val, 
                seen_classes
            )
        # Record that the loader did not create a validation split.
        else:
            x_val, y_val = None, None

        # Respect explicit validation disabling.
        if not use_valset:
            x_val, y_val = None, None

        # Flatten image inputs for dense VAE replay.
        if isinstance(generative_model, VariationalAutoencoder) \
        and not return_features: # Match configured dense VAE input shapes.
            x_train = x_train.reshape((len(x_train), -1))
            x_test = x_test.reshape((len(x_test), -1))
            # Flatten matching validation inputs when present.
            if x_val is not None:
                x_val = x_val.reshape((len(x_val), -1))

        diffusion_data_min = float(np.min(all_x_train))
        diffusion_data_range = float(
            np.max(all_x_train) - diffusion_data_min
        )
        # Keep constant-valued tasks numerically valid.
        if diffusion_data_range == 0.:
            diffusion_data_range = 1.

        # Add fixed-buffer samples from earlier tasks.
        if use_buffer:
            current_x_train, current_y_train = x_train, y_train
            x_buffer, y_buffer = buffer.sample_buffer_and_prepare_dataset(
                buffer_kwargs["sample_num"]
            )
            # buffer.sample_dataset_and_extend_buffer(
            #     (x_train, y_train), 
            #     buffer_kwargs["insert_num"]
            # )

            # Append replay only when the buffer is nonempty.
            if len(x_buffer) > 0:
                x_train = np.concatenate([x_train, x_buffer], axis=0)
                y_train = np.concatenate([y_train, y_buffer], axis=0)

        # Generate replay after the first task.
        if generative_model is not None and task_index > 0:
            classes = [
                label
                for group in internal_task_groups[:task_index]
                for label in group
            ]
            # Generate VAE replay in loader label format.
            if isinstance(generative_model, VariationalAutoencoder):
                x_buffer, y_buffer = generative_model.generate(
                    classes=classes, 
                    samples_per_class=generative_model_kwargs[
                        "samples_per_class"
                    ], 
                    onehot_y_output=load_dataset_fn_kwargs["onehot_labels"]
                )
            # Generate conditional diffusion replay for prior classes.
            else:
                y_buffer = np.repeat(
                    classes, 
                    generative_model_kwargs["samples_per_class"]
                )
                x_buffer = generative_model.sample(
                    network_name=generative_model.test_network_name, 
                    labels=y_buffer + int(generative_model.use_cfg)
                ).numpy() if len(y_buffer) != 0 else []

                # Restore generated images to classifier preprocessing space.
                if len(x_buffer) > 0 and not return_features:
                    x_buffer = (
                        x_buffer * diffusion_data_range + diffusion_data_min
                    )
                    x_buffer = x_buffer.astype(x_train.dtype)

                # Match generated class IDs to one-hot loader labels.
                if load_dataset_fn_kwargs["onehot_labels"]:
                    y_buffer = np.eye(
                        class_num, 
                        dtype=y_train.dtype
                    )[y_buffer]
                # Match generated IDs to column-shaped sparse labels.
                elif y_train.ndim > 1:
                    y_buffer = y_buffer[:, None]

                y_buffer = y_buffer.astype(y_train.dtype)

            # Append nonempty generated replay to real data.
            if len(x_buffer) > 0:
                x_train = np.concatenate([x_train, x_buffer], axis=0)
                y_train = np.concatenate([y_train, y_buffer], axis=0)

        classifier_x_train = x_train
        classifier_x_val = x_val
        classifier_x_test = x_test
        classifier_input_shape = getattr(new_model, "input_shape", None)

        # Reshape generated image replay for a standalone image classifier.
        if not use_diffusion_classifier and isinstance(
            classifier_input_shape, tuple
        ) and len(classifier_input_shape) == 2:
            classifier_x_train = x_train.reshape((len(x_train), -1))
            classifier_x_test = x_test.reshape((len(x_test), -1))
            classifier_x_val = x_val.reshape((len(x_val), -1)) if x_val is not None else None

        classifier_y_train = y_train
        classifier_y_val = y_val
        classifier_y_test = y_test
        # Choose categorical or sparse loss to match loader labels.
        if load_dataset_fn_kwargs["onehot_labels"]:
            loss = getattr(
                new_model, 
                "loss", 
                compile_args.get(
                    "loss", 
                    "sparse_categorical_crossentropy"
                )
            )
            loss_name = getattr(
                loss, 
                "name", 
                getattr(loss, "__name__", str(loss))
            ).lower()

            # Convert labels to integer IDs for sparse classifier losses.
            if "sparse" in loss_name:
                classifier_y_train = np.argmax(y_train, axis=-1)
                classifier_y_val = np.argmax(y_val, axis=-1) if y_val is not None else None
                classifier_y_test = np.argmax(y_test, axis=-1)
            # Trim one-hot targets to a newly expanded standalone head.
            elif not uses_attached_classifier:
                classifier_y_train = y_train[..., :seen_class_num]
                classifier_y_val = y_val[..., :seen_class_num] \
                    if y_val is not None else None
                classifier_y_test = y_test[..., :seen_class_num]

        task_shuffle_buffer = len(x_train) if shuffle_buffer is None else shuffle_buffer
        trainset = get_dataset(
            classifier_x_train, 
            classifier_y_train, 
            shuffle_buffer=task_shuffle_buffer, 
            batch_size=batch_size, 
            drop_remainder=False, 
            seed=dataset_seed
        )
        valset = get_dataset(
            classifier_x_val, 
            classifier_y_val, 
            shuffle_buffer=0, 
            batch_size=batch_size, 
            drop_remainder=False
        ) if x_val is not None else None
        testset = get_dataset(
            classifier_x_test, 
            classifier_y_test, 
            shuffle_buffer=0, 
            batch_size=batch_size, 
            drop_remainder=False
        )

        history = {}
        # Attach standalone-classifier callbacks only when that fit phase runs.
        if train_direct_classifier:
            task_callbacks = phase_callbacks(
                "val_accuracy" if valset is not None else "accuracy", 
                legacy_patience=5,
            )

            history = train_task_model(
                new_model, 
                trainset, 
                valset, 
                task_callbacks=task_callbacks
            )

        prev_model = new_model

        # Retain completed-task examples for future fixed-buffer replay.
        if use_buffer:
            buffer.sample_dataset_and_extend_buffer(
                (current_x_train, current_y_train),
                buffer_kwargs["insert_num"]
            )

        generative_trainset = None
        generative_testset = None
        # Train the joint VAE/classifier through the shared API.
        if isinstance(generative_model, VAEClassifier):
            generative_history = train_task_model(
                generative_model, 
                x_train, 
                (x_val, y_val) if x_val is not None else None, 
                task_callbacks=phase_callbacks(
                    "val_clf_accuracy" if x_val is not None \
                        else "clf_accuracy"
                ), 
                fit_method="train", 
                fit_kwargs={
                    "y": y_train, 
                    "train_num": generative_model_kwargs["train_num"], 
                    "batch_size": batch_size, 
                    "shuffle_buffer": task_shuffle_buffer, 
                    "seed": dataset_seed
                }
            )
            generative_trainset = get_dataset(
                x_train, y_train, 
                shuffle_buffer=task_shuffle_buffer, 
                batch_size=batch_size, 
                drop_remainder=False, 
                seed=dataset_seed
            )
            generative_testset = get_dataset(
                x_test, y_test, 
                shuffle_buffer=0, 
                batch_size=batch_size, 
                drop_remainder=False
            )
        # Train a conditional replay VAE through the shared API.
        elif isinstance(generative_model, VariationalAutoencoder):
            generative_history = train_task_model(
                generative_model, 
                x_train, 
                (x_val, y_val) if x_val is not None else None, 
                task_callbacks=phase_callbacks(
                    "val_loss" if x_val is not None else "loss", 
                    default_mode="min", 
                ), 
                fit_method="train", 
                fit_kwargs={
                    "y": y_train, 
                    "train_num": generative_model_kwargs["train_num"], 
                    "batch_size": batch_size, 
                    "clf": new_model, 
                    "shuffle_buffer": task_shuffle_buffer, 
                    "seed": dataset_seed
                }
            )
            generative_trainset = get_dataset(
                x_train, y_train, 
                shuffle_buffer=task_shuffle_buffer, 
                batch_size=batch_size, 
                drop_remainder=False, 
                seed=dataset_seed
            )
            generative_testset = get_dataset(
                x_test, y_test, 
                shuffle_buffer=0, 
                batch_size=batch_size, 
                drop_remainder=False
            )
        # Train diffusion replay from fresh task datasets.
        elif isinstance(generative_model, DiffusionModel):
            generative_x = _prepare_diffusion_x(
                x_train, 
                diffusion_data_min, 
                diffusion_data_range
            )
            generative_y = np.argmax(y_train, axis=-1) if load_dataset_fn_kwargs["onehot_labels"] \
                        else np.asarray(y_train).reshape(-1)

            diffusion_classifier_x = generative_x
            diffusion_classifier_y = generative_y

            train_num = generative_model_kwargs["train_num"]
            # Resample diffusion training rows to the configured exact count.
            if train_num != -1:
                indices = rng.integers(
                    0, 
                    len(generative_x), 
                    (train_num,)
                )
                generative_x = generative_x[indices]
                generative_y = generative_y[indices]

            generative_trainset = get_dataset(
                generative_x, 
                generative_y, 
                shuffle_buffer=len(generative_x) if shuffle_buffer is None else shuffle_buffer,
                batch_size=batch_size, 
                drop_remainder=False, 
                seed=dataset_seed
            )
            generative_y_val = np.argmax(y_val, axis=-1) if load_dataset_fn_kwargs["onehot_labels"] \
                            and y_val is not None else np.asarray(y_val).reshape(-1) \
                            if y_val is not None else None
            generative_valset = get_dataset(
                _prepare_diffusion_x(
                    x_val, 
                    diffusion_data_min, 
                    diffusion_data_range
                ), 
                generative_y_val, 
                shuffle_buffer=0, 
                batch_size=batch_size,  
                drop_remainder=False
            ) if x_val is not None else None
            generative_y_test = np.argmax(y_test, axis=-1) if load_dataset_fn_kwargs["onehot_labels"] \
                                else np.asarray(y_test).reshape(-1)

            generative_testset = get_dataset(
                _prepare_diffusion_x(
                    x_test, 
                    diffusion_data_min, 
                    diffusion_data_range
                ), 
                generative_y_test, 
                shuffle_buffer=0, 
                batch_size=batch_size, 
                drop_remainder=False
            )

            # Select V2's phase-aware generator entry point.
            if isinstance(generative_model, DiffusionClassifierV2):
                generative_fit_method = "fit_generator_progressively" \
                    if fit_method == "fit_progressively" \
                    else "fit_generator"
            # Use the combined replay-wrapper method for every other variant.
            else:
                generative_fit_method = fit_method
            generative_history = train_task_model(
                generative_model, 
                generative_trainset, 
                generative_valset, 
                task_callbacks=phase_callbacks(
                    "val_loss" if generative_valset is not None else "loss", 
                    default_mode="min"
                ), 
                fit_method=generative_fit_method,
                fit_kwargs=dict(fit_kwargs or {})
            )

            # Run a separate V2 classifier fit after diffusion-generator training.
            if use_diffusion_classifier and isinstance(
                generative_model, DiffusionClassifierV2
            ): # Train V2 classifier variables in their required separate phase.
                task_callbacks = phase_callbacks(
                    (
                        "val_total_accuracy" if generative_valset is not None
                        else "total_accuracy"
                    ) if generative_model.use_total_accuracy else (
                        "val_" + generative_model.accuracy_tracker.name
                        if generative_valset is not None
                        else generative_model.accuracy_tracker.name
                    ),
                    legacy_patience=5
                )

                diffusion_classifier_trainset = get_dataset(
                    diffusion_classifier_x, 
                    diffusion_classifier_y, 
                    shuffle_buffer=len(diffusion_classifier_x) if shuffle_buffer is None else shuffle_buffer, 
                    batch_size=batch_size, 
                    drop_remainder=False, 
                    seed=dataset_seed
                )
                history = train_task_model(
                    generative_model, 
                    diffusion_classifier_trainset, 
                    generative_valset, 
                    task_callbacks=task_callbacks, 
                    fit_method="fit_discriminator"
                )
        # Record that this task has no generative replay phase.
        else:
            generative_history = None

        generative_evaluations = {}
        # Evaluate the generative phase when one ran for this task.
        if generative_history is not None:
            generative_evaluations = report_task_model(
                generative_history, 
                generative_model, 
                generative_trainset, 
                generative_testset
            )

        classifier_evaluations = {}
        classifier_report_history = history or generative_history
        # Evaluate a separately fitted standalone classifier when available.
        if not use_diffusion_classifier and classifier_report_history \
        and train_direct_classifier:
            classifier_evaluations = report_task_model(
                classifier_report_history, 
                new_model, 
                trainset, 
                testset
            )

        accuracy_source = generative_evaluations if score_from_generator \
                        else classifier_evaluations
        # Score the same raw/EMA network selected by the diffusion wrapper.
        if use_diffusion_classifier:
            selected_evaluation_name = (
                "valset_ema_eval"
                if generative_model.test_network_name == "ema"
                else "valset_network_eval"
            )
            selected_evaluation = generative_evaluations.get(
                selected_evaluation_name
            )
            # Fall back to the complete report when the selected entry is absent.
            if isinstance(selected_evaluation, dict):
                accuracy_source = selected_evaluation

        acc = _reported_accuracy(accuracy_source)
        ensemble_acc = None
        # Read only the explicit ensemble metric from diffusion reports.
        if evaluate_ensemble_accuracy:
            ensemble_acc = _reported_accuracy(
                accuracy_source,
                ensemble=True
            )
            # Fail clearly instead of silently substituting ordinary accuracy.
            if ensemble_acc is None:
                raise KeyError(
                    "ensemble_accuracy was not returned by the task report."
                )

        y_test_ids = np.argmax(y_test, axis=-1) if load_dataset_fn_kwargs["onehot_labels"] \
                    else np.asarray(y_test).reshape(-1)

        # Obtain one common prediction vector for cumulative and per-task scores.
        if use_diffusion_classifier:
            predictions = _predict_diffusion_classes(
                generative_model,
                x_test,
                y_test_ids,
                diffusion_data_min,
                diffusion_data_range,
                batch_size
            )
        # Evaluate joint-only VAE classification on reconstructed inputs.
        elif uses_attached_classifier and not train_classifier_separately:
            _, _, predictions = generative_model(
                (x_test, y_test),
                training=False,
            )
        # Use direct predictions from the standalone classifier.
        else:
            predictions = new_model.predict(
                classifier_x_test,
                verbose=verbose
            )

        prediction_ids = np.argmax(predictions, axis=-1)
        prediction_accuracy = float(accuracy_score(y_test_ids, prediction_ids))
        # Preserve the legacy report-derived cumulative accuracy when available.
        if acc is None:
            acc = prediction_accuracy
        matrix_row = [np.nan] * len(internal_task_groups)
        # Score every learned task group separately without future-class rows.
        for learned_index, learned_classes in enumerate(
            internal_task_groups[:task_index + 1]
        ):
            group_mask = np.isin(y_test_ids, learned_classes)
            matrix_row[learned_index] = float(accuracy_score(
                y_test_ids[group_mask],
                prediction_ids[group_mask],
            ))
        accuracy_matrix.append(matrix_row)

        # Expose joint generative history as task history when no separate fit ran.
        if uses_attached_classifier and not history \
        and generative_history is not None:
            history = generative_history

        histories.append(history)
        generative_histories.append(generative_history)
        classifier_evaluations_list.append(classifier_evaluations)
        generative_evaluations_list.append(generative_evaluations)
        acc_list.append(acc)
        # Retain ensemble scores beside the backward-compatible normal scores.
        if ensemble_acc is not None:
            ensemble_acc_list.append(ensemble_acc)

        # Print the completed task's accuracy when requested.
        if verbose:
            print(f"Task test accuracy: {acc:.4f}")
            # Print the optional ensemble result on the same task summary.
            if ensemble_acc is not None:
                print(f"Task ensemble accuracy: {ensemble_acc:.4f}")
            print(75*'-'+'\n')

    # Plot accuracy across completed continual tasks when requested.
    if plot_results:
        CL_plot(
            class_num,
            [(acc_list, " ")],
            class_counts=[
                sum(group_sizes[:index + 1])
                for index in range(len(group_sizes))
            ],
        )

    continual_metrics = _continual_metrics(accuracy_matrix)
    new_task_accuracy, old_task_accuracy = _task_accuracy_summaries(
        accuracy_matrix
    )

    # Return histories and final objects for orchestration callers.
    if return_details:
        return {
            "accuracies": acc_list, 
            "ensemble_accuracies": ensemble_acc_list,
            "class_order": class_order,
            "task_classes": original_task_groups,
            "accuracy_matrix": accuracy_matrix,
            "new_task_accuracy": new_task_accuracy,
            "old_task_accuracy": old_task_accuracy,
            "continual_metrics": continual_metrics,
            "dataset_seed": dataset_seed,
            "histories": histories, 
            "generative_histories": generative_histories, 
            "classifier_evaluations": classifier_evaluations_list, 
            "generative_evaluations": generative_evaluations_list, 
            "model": prev_model, 
            "generative_model": generative_model
        }
    return acc_list


def run_protocol_self_tests() -> dict[str, str]:
    """Exercise schedule, label, metric, and validation-isolation contracts.

    Args:
        None.

    Returns:
        dict[str, str]: Passing markers for schedule resolution, sparse and
        one-hot remapping, continual metrics, and validation isolation.
    """

    from unittest.mock import Mock, patch

    from common import dataloader
    from common.hpo import _objective_values


    # Verify the legacy two-class start followed by singleton experiences.
    assert resolve_continual_schedule(4) == (
        [0, 1, 2, 3],
        [[0, 1], [2], [3]],
    )

    class_order = [3, 1, 2, 0]
    sparse = np.asarray([[3], [1], [2], [0]], dtype="uint8")
    sparse_remapped = _remap_continual_labels(
        sparse,
        class_order,
        onehot_labels=False,
    )
    assert np.array_equal(
        sparse_remapped,
        np.asarray([[0], [1], [2], [3]], dtype="uint8"),
    )
    onehot = np.eye(4, dtype="float32")[[3, 1, 2, 0]]
    onehot_remapped = _remap_continual_labels(
        onehot,
        class_order,
        onehot_labels=True,
    )
    assert np.array_equal(onehot_remapped, np.eye(4, dtype="float32"))

    matrix = [
        [0.8, np.nan, np.nan],
        [0.7, 0.9, np.nan],
        [0.6, 0.85, 0.75],
    ]
    metrics = _continual_metrics(matrix)
    assert np.isclose(metrics["final_average_accuracy"], 0.7333333333333334)
    assert np.isclose(metrics["average_incremental_accuracy"], 0.7777777777777778)
    assert np.isclose(metrics["average_forgetting"], 0.125)
    assert np.isclose(metrics["backward_transfer"], -0.125)
    new_accuracy, old_accuracy = _task_accuracy_summaries(matrix)
    assert np.allclose(new_accuracy, [0.8, 0.9, 0.75])
    assert np.allclose(old_accuracy, [np.nan, 0.7, 0.725], equal_nan=True)

    x_train = np.arange(24, dtype="uint8").reshape((6, 2, 2, 1))
    y_train = np.asarray([3, 1, 3, 1, 3, 1], dtype="uint8")
    x_test = np.arange(16, dtype="uint8").reshape((4, 2, 2, 1))
    y_test = np.asarray([3, 1, 3, 1], dtype="uint8")
    loader = Mock(return_value=(
        x_train, y_train, None, None, x_test, y_test
    ))
    arrays, _ = _load_continual_arrays(
        loader,
        class_num=2,
        class_order=[3, 1],
        return_features=False,
        load_dataset_fn_kwargs={
            "preprocess": None,
            "onehot_labels": False,
        },
        max_train_samples=None,
        max_val_samples=1,
        pad=0,
        dataset_seed=7,
    )
    _, remapped_train, x_val, y_val, retained_test, remapped_test = arrays
    assert x_val is None and y_val is None
    assert len(retained_test) == len(x_test)
    assert set(remapped_train.reshape(-1)) == {0, 1}
    assert np.array_equal(remapped_test, np.asarray([0, 1, 0, 1]))

    ordinary_loader = Mock(return_value=(
        x_train,
        np.asarray([0, 1, 0, 1, 0, 1], dtype="uint8"),
        None,
        None,
        x_test,
        np.asarray([0, 1, 0, 1], dtype="uint8"),
    ))
    # Replace only the built-in loader long enough to inspect absent validation.
    with patch.object(dataloader, "load_mnist", ordinary_loader):
        _, valset = dataloader.get_datasets(
            dataset_name="mnist",
            model_name="cnn",
            preprocess=None,
            validation_ratio=0.,
            use_valset=True,
            batch_size=2,
        )
    assert valset is None

    validation_objective = _objective_values(
        "continual",
        "cnn",
        {
            "task_val_accuracy": [0.2, 0.4],
            "continual_accuracy": [0.99, 0.99],
        },
    )
    assert np.isclose(validation_objective, 0.3)
    # Reject test-only ensemble feedback in the continual HPO path.
    try:
        _objective_values(
            "continual",
            "dit_classifier",
            {"task_val_accuracy": [0.5]},
            use_ensemble_accuracy=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Continual HPO must reject test ensemble feedback.")

    return {
        "schedule": "passed",
        "label_remapping": "passed",
        "continual_metrics": "passed",
        "validation_isolation": "passed",
    }


def continually_learn(
    config: Config | dict[str, object] | None = None, 
    teacher_network: tf.keras.Model | None = None,
    **kwargs: object
) -> list[float] | dict[str, object]:
    """Run class-incremental learning from a config or direct keyword inputs.

    Exactly one input style is used. With ``config=None``, every setting comes
    from ``kwargs`` and ``class_num`` plus ``load_dataset_fn`` are required.
    With a :class:`common.config.Config` (or a compatible root mapping), direct
    keywords except ``teacher_network`` are ignored:
    :func:`common.dataloader.get_datasets` creates the
    loader, :func:`common.model.get_model` creates the classifier/replay-model
    bundle, and :func:`common.train.train_model` plus
    :func:`common.train.report` run the configured training and reporting.
    Config mode requires ``config.training.task == "continual"``.

    Args:
        config (Config | dict[str, object] | None): Optional complete project
            configuration. A mapping is normalized with ``Config(**config)``.
        teacher_network (tf.keras.Model | None): Runtime-only teacher passed to
            a newly constructed diffusion classifier for task one. Automatic
            continual distillation replaces it after that task. It is never
            stored in the serializable configuration.
        **kwargs (object): Direct-mode inputs, used only when ``config`` is
            ``None``. The possible keys are:

            - ``class_num`` (int, required): Total selected class count.
            - ``class_order`` (Sequence[int] | None): Original dataset labels
              in introduction order; defaults to the natural order.
            - ``task_groups`` (Sequence[Sequence[int]] | None): Labels added
              per task; defaults to two classes then singleton tasks.
            - ``load_dataset_fn`` (Callable, required): Loader returning
              ``(x_train, y_train, x_val, y_val, x_test, y_test)``.
            - ``load_dataset_fn_kwargs`` (dict | None): Loader overrides.
              Built-in keys are ``preprocess``, ``onehot_labels``,
              ``features_path``, ``validation_ratio``, and ``seed``; do not
              supply ``indices``, ``return_features``, or ``verbose``.
            - ``remove_prev_classes`` (bool, default ``True``): Later tasks
              train only on the new class instead of all seen classes.
            - ``keep_same_model`` (bool, default ``True``): Copy learned
              classifier weights into each expanded head.
            - ``tuned_model_path`` (str, default ``""``): Saved Keras
              classifier template. Paths containing ``"dnn"`` select saved
              features; other paths select image input.
            - ``compile_args`` (dict | None): Classifier ``Model.compile``
              overrides.
            - ``use_loaded_opt`` (bool, default ``False``): Reuse the optimizer
              stored in ``tuned_model_path``.
            - ``batch_size`` (int, default ``128``): Per-task batch size.
            - ``epochs`` (int, default ``100``): Maximum epochs per ordinary
              phase. Progressive diffusion replay uses ``stage_epochs`` and
              ``final_epochs`` from ``fit_kwargs`` instead.
            - ``fit_method`` (str, default ``"fit"``): Select
              ``"fit_progressively"`` only for diffusion replay-model
              training. V2 maps this to its progressive generator method and
              keeps its separate discriminator fit ordinary.
            - ``fit_kwargs`` (dict | None): Extra diffusion-fit arguments,
              copied for every task. A progressive curriculum therefore
              repeats per task, and any requested depth additions persist.
            - ``use_buffer`` (bool, default ``False``): Enable bounded sample
              replay; it is mutually exclusive with ``generative_model``.
            - ``buffer_kwargs`` (dict | None): ``maxlen``, ``sample_num``,
              ``insert_num``, and ``seed``; defaults are ``10000``, ``1000``,
              ``1000``, and ``None`` respectively.
            - ``plot_results`` (bool, default ``True``): Plot task accuracy.
            - ``verbose`` (bool | int, default ``True``): Training/reporting
              verbosity.
            - ``generative_model`` (tf.keras.Model | None): Conditional VAE,
              raw diffusion network, or diffusion wrapper used for replay.
              Diffusion networks must be initialized with ``num_classes=None``.
            - ``teacher_network`` is the explicit argument above and may be
              used with a raw diffusion classifier.
            - ``generative_model_compile_args`` (dict | None): Compile
              overrides used only when a raw diffusion network is wrapped;
              defaults to Adam and MSE.
            - ``generative_model_kwargs`` (dict | None): ``train_num`` and
              ``samples_per_class``, both defaulting to ``1000``;
              ``train_num=-1`` disables replay-model resampling.
            - ``use_generative_model_classifier`` (bool, default ``False``):
              Use a classifier attached to the replay model.
            - ``train_classifier_separately`` (bool, default ``False``): Add a
              classifier phase for ``VAEClassifier``; it must be true for
              ``DiffusionClassifierV2`` and false for ``DiffusionClassifier``.
            - ``use_distillation`` (bool, default ``False``): Snapshot each
              completed diffusion-classifier student for use as the next
              task's teacher; requires a token and positive teacher objective.
            - ``evaluate_ensemble_accuracy`` (bool, default ``False``): Also
              evaluate timestep-ensembled accuracy for a diffusion-classifier
              replay model after every task.
            - ``ensemble_accuracy_kwargs`` (dict | None): Options forwarded to
              ``DiffusionClassifier.evaluate_ensemble_accuracy``.
            - ``callbacks_list`` (Sequence[Callback] | None): Extra callbacks
              forwarded through :func:`common.train.train_model`.
            - ``return_details`` (bool, default ``False``): Return task
              histories and final model objects in addition to accuracies.
            - ``use_valset`` (bool, default ``True``): Use an explicit loader
              validation split; a missing split remains disabled.

    Returns:
        list[float] | dict[str, object]: Test accuracy for each task. With
        ``return_details=True`` (or its configured equivalent), the mapping
        also contains ``ensemble_accuracies``, classifier/generative histories,
        their per-task report outputs, and final models; configured details
        additionally contain aggregate evaluations.

    Raises:
        TypeError: If direct mode omits a required key, includes an unknown
            key, or config is not a ``Config``/mapping.
        ValueError: If configured mode is not a continual task or a requested
            model/dataset/replay combination is invalid.
    """

    # Resolve the legacy direct keyword interface when no config is supplied.
    if config is None:
        options = dict(kwargs)
        defaults = {
            "class_order": None,
            "task_groups": None,
            "load_dataset_fn_kwargs": None, 
            "remove_prev_classes": True, 
            "keep_same_model": True, 
            "tuned_model_path": "", 
            "compile_args": None, 
            "use_loaded_opt": False, 
            "batch_size": 128, 
            "epochs": 100, 
            "fit_method": "fit",
            "fit_kwargs": None,
            "use_buffer": False, 
            "buffer_kwargs": None, 
            "plot_results": True, 
            "verbose": True, 
            "generative_model": None, 
            "generative_model_compile_args": None, 
            "generative_model_kwargs": None, 
            "use_generative_model_classifier": False, 
            "train_classifier_separately": False, 
            "use_distillation": False,
            "evaluate_ensemble_accuracy": False, 
            "ensemble_accuracy_kwargs": None, 
            "callbacks_list": None, 
            "return_details": False, 
            "use_valset": True
        }
        allowed = {"class_num", "load_dataset_fn", *defaults}

        unknown = sorted(set(options) - allowed)
        # Reject direct options outside the documented continual API.
        if unknown:
            raise TypeError(
                "Unsupported continually_learn options: " + str(unknown)
            )

        missing = [
            name for name in ("class_num", "load_dataset_fn")
            if name not in options
        ]
        # Require class count and dataset loader in direct mode.
        if missing:
            raise TypeError(
                "Missing required continually_learn options: " + str(missing)
            )

        return _run_continual_tasks(
            teacher_network=teacher_network,
            **{**defaults, **options}
        )

    # Convert compatible mappings into typed configuration.
    if isinstance(config, dict):
        config = Config(**config)

    # Reject unsupported configuration root types.
    if not isinstance(config, Config):
        raise TypeError("config must be a Config, mapping, or None.")

    # Restrict this entry point to continual-learning configurations.
    if config.training.task.lower() != "continual":
        raise ValueError(
            "continually_learn(config) requires training.task='continual'."
        )


    # Reuse the shared project pipeline instead of duplicating its four stages.
    from common.train import main


    run = main(config, teacher_network=teacher_network)
    model = run["model"]
    history = run["history"]

    # Require the model bundle produced for continual tasks.
    if not isinstance(model, dict):
        raise TypeError(
            "A continual config must create a model bundle."
        )

    details = model.get("continual_details")
    # Normalize legacy accuracy-list results into a detail mapping.
    if details is None:
        details = {
            "accuracies": list(history.get("continual_accuracy", [])), 
            "ensemble_accuracies": list(history.get(
                "continual_ensemble_accuracy", []
            )), 
            "histories": [], 
            "generative_histories": [], 
            "classifier_evaluations": [], 
            "generative_evaluations": [], 
            "class_order": [],
            "task_classes": [],
            "accuracy_matrix": [],
            "new_task_accuracy": [],
            "old_task_accuracy": [],
            "continual_metrics": {},
            "dataset_seed": config.training.seed,
            "model": model.get("classifier"), 
            "generative_model": model.get("generative_model")
        }
    details["evaluations"] = run["evaluations"]

    # Return full task details only when configured by the caller.
    if config.continually_learn.return_details:
        return details
    return details["accuracies"]
