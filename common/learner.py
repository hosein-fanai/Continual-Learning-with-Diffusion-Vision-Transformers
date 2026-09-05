"""Class-incremental experiment orchestration shared by Config and HPO APIs.

This module loads one common array representation, resolves a class schedule, expands
classifier vocabularies, constructs per-task training pools, and fits standalone,
VAE-replay, or diffusion-classifier experiments. A V2 wrapper runs its generator phase and,
when its attached classifier is selected, a separate discriminator phase. Optional
previous-task teachers, replay provenance, deterministic random streams, and committed task
checkpoints keep those phases consistent across task transitions and resumed runs.

Inputs normally arrive through ``continually_learn(Config(...))``; direct calls accept a
dataset loader, model objects, and the controls documented by ``_run_continual_tasks``.
``DatasetArrays`` names the six-array loader return contract ``(x_train, y_train, x_val,
y_val, x_test, y_test)``; validation arrays may be ``None``. ``DatasetLoader`` is a callable
returning that tuple.

Outputs are either a task-accuracy trajectory or a detailed mapping containing trained
models, histories, task matrices, diagnostics, and recovery metadata. Development mode
evaluates validation data only. Legacy and authenticated confirmation modes also evaluate
the held-out test split. Enabling ``use_ensemble_accuracy`` makes task-balanced
EnsembleAccuracy matrices the source of continual metrics; ordinary cumulative accuracy is
example-weighted.

The task runner mutates model weights and runtime random/dtype state. Plotting, callback
artifacts, and checkpoint writes depend on their explicit controls. Importing this module
defines these APIs; it does not start an experiment.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from copy import deepcopy

import numpy as np
import tensorflow as tf

from common.config import Config, normalize_training_task, resolve_continual_schedule
from common.utils import CL_plot
from common.model import get_model, copy_model, get_callbacks
from common.replay_buffer import (
    ReplayBuffer,
    _balanced_generation_labels,
    _cached_replay_candidates,
    _replay_cache_path,
    _restore_replay_label_shape,
    _sample_exact_rows
)
from common.dataloader import get_dataset, _limit_samples, _pad_images
from common.mechanistic import (
    calibration_metrics,
    class_centroid_drift,
    linear_cka,
    replay_quality_metrics,
    select_replay_candidates
)
from common.continual_reporting import (
    continual_metrics as _continual_metrics,
    observed_mean as _observed_mean,
    task_accuracy_summaries as _task_accuracy_summaries
)
from common.runtime import configure_runtime, derive_seed
from common.recovery import (
    _array_recovery_descriptor,
    _artifact_recovery_descriptor,
    _model_topology_descriptor,
    _model_weight_descriptor,
    _progressive_depth_specs,
    _qualified_name,
    _recovery_descriptor,
    _trackable_topology_descriptor,
    capture_rng_state,
    fingerprint_state,
    load_task_checkpoint,
    restore_replay_buffer,
    restore_rng_state,
    save_task_checkpoint
)

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


def _fresh_optimizer(value: object) -> object:
    """Clone a configured Keras optimizer for an independent continual run.

    Only instances are cloned. Names and other specifications are forwarded to Keras
    unchanged. Serialization preserves optimizer configuration, not learned slot values or
    iteration state; the caller's optimizer is not trained in place.

    Args:
        value (object): Optimizer instance or another accepted compile specification, such
            as an optimizer name.

    Returns:
        object: A newly deserialized optimizer for a Keras instance; otherwise the original
        value.

    Raises:
        Exception: Keras serialization/deserialization errors propagate when the
            configured optimizer cannot be reconstructed.
    """

    # Pass optimizer names and other non-instance specifications through unchanged.
    if not isinstance(value, tf.keras.optimizers.Optimizer):
        return value

    return tf.keras.optimizers.deserialize(tf.keras.optimizers.serialize(value))


def _optimizer_iteration_metrics(
    classifier: tf.keras.Model | None,
    generative_model: tf.keras.Model | None
) -> dict[str, int]:
    """Read optimizer update counters for the active model roles.

    Counters are read eagerly without changing model or optimizer state. Shared optimizer
    objects may appear under multiple role names.

    Args:
        classifier (tf.keras.Model | None): Standalone or attached classifier head; None
            omits this role.
        generative_model (tf.keras.Model | None): Replay model or wrapper; None omits this
            role.

    Returns:
        dict[str, int]: Counters keyed as ``<role>_<attribute>``, where role is
        ``classifier`` or ``replay`` and attribute is an available ``optimizer``,
        ``gen_optimizer``, or ``clf_optimizer``. Missing counters contribute no entry.
    """

    counters = {}
    for role, model in (("classifier", classifier), ("replay", generative_model)):
        # An absent classifier or generator has no optimizer counters to record.
        if model is None:
            continue

        for attribute in ("optimizer", "gen_optimizer", "clf_optimizer"):
            optimizer = getattr(model, attribute, None)
            iterations = getattr(optimizer, "iterations", None)
            # Skip optimizer attributes that expose no iteration counter.
            if iterations is None:
                continue

            counters[f"{role}_{attribute}"] = int(iterations.numpy())

    return counters


def _finite_fixed_step_dataset(
    dataset: tf.data.Dataset,
    steps_per_epoch: int,
    epochs: int,
) -> tf.data.Dataset:
    """Repeat a finite dataset to supply a requested fit-batch budget.

    The requested budget is at least one batch. A known finite source retains at least one
    complete pass, even when the requested batch count is smaller, so wrapper label
    discovery can see every source class. Unknown cardinality uses the requested budget. An
    empty source remains empty. A source configured to reshuffle on iteration can change
    order on repetition. This helper does not batch, introduce shuffling, or materialize
    data. Fit callbacks may stop before the requested budget is consumed.

    Args:
        dataset (tf.data.Dataset): Re-iterable finite batched training dataset; repetition
            preserves its element structure and shuffle policy.
        steps_per_epoch (int): Requested fit batches per epoch; used to compute the repeated
            batch budget.
        epochs (int): Epoch count for the active ordinary or progressive phase.

    Returns:
        tf.data.Dataset: A repeated stream bounded by take(), with the original element
        specification.
    """

    required_batches = max(int(steps_per_epoch) * max(int(epochs), 1), 1)
    source_batches = int(tf.data.experimental.cardinality(dataset).numpy())
    # A known finite source must retain a full pass for label discovery.
    if source_batches >= 0:
        required_batches = max(required_batches, source_batches)
    return dataset.repeat().take(required_batches)


def _validate_supplied_model_runtime(
    model: tf.keras.Model | None,
    seed: int | None,
    role: str
) -> None:
    """Check that a caller-built model agrees with the requested master seed.

    Only an explicit ``seed`` attribute is checked. A missing or mismatched value on such a
    component cannot be repaired by reseeding after weight initialization. This helper
    checks seeds, not dtype policy or model architecture.

    Args:
        model (tf.keras.Model | None): Prebuilt model or None; a wrapped Keras network is
            inspected as well.
        seed (int | None): Required experiment seed, or None to impose no seed contract.
        role (str): Human-readable model role included in an incompatibility message.

    Returns:
        None: Models without declared seeds, absent models, and matching models pass without
        mutation.

    Raises:
        ValueError: An explicitly seeded component does not match the requested seed.
    """

    # An unseeded experiment imposes no model-seed contract.
    if model is None or seed is None:
        return

    seeded_components = [(role, model)]
    raw_network = getattr(model, "network", None)
    # Wrapped models also carry a raw network whose seed must agree.
    if isinstance(raw_network, tf.keras.Model):
        seeded_components.append((role + " raw network", raw_network))

    for component_role, component in seeded_components:
        # Components without an explicit seed do not declare a seed contract.
        if not hasattr(component, "seed"):
            continue

        component_seed = getattr(component, "seed")
        # Reject a missing or mismatched seed on an explicitly seeded component.
        if component_seed is None or int(component_seed) != int(seed):
            raise ValueError(
                f"{component_role} was built with seed {component_seed!r}, "
                f"but the continual experiment requires seed {seed}. Build "
                "the model through get_model/config or reconstruct it with "
                "the same seed before starting continual learning."
            )


def _derive_generative_callback_seed(
    task_seed: int | None,
    callback_seed: int | None,
    task_index: int,
    callback_index: int,
    callback: object,
) -> int | None:
    """Derive a stable task-local random stream for one generative callback.

    Experiment-seeded and callback-only runs use distinct derivation namespaces. The
    callback object itself is not changed.

    Args:
        task_seed (int | None): Derived experiment task seed; when present it takes
            precedence over the callback seed.
        callback_seed (int | None): Callback-owned base seed used only when the experiment
            supplies no task seed.
        task_index (int): Zero-based task position, included in the callback-only seed
            route.
        callback_index (int): Stable position in the configured generative callback list.
        callback (object): Callback object whose qualified type name separates random
            streams.

    Returns:
        int | None: Deterministic child seed, or None when neither seed source is specified.
    """

    callback_name = _qualified_name(callback)
    # Preserve the established experiment-seeded callback stream exactly.
    if task_seed is not None:
        return derive_seed(
            task_seed,
            "generative_callback",
            callback_index,
            callback_name,
        )
    # Without an experiment seed, derive the stream from the callback's own seed.
    if callback_seed is not None:
        return derive_seed(
            callback_seed,
            "continual_task",
            task_index,
            "generative_callback",
            callback_index,
            callback_name,
        )

    return None


def _reset_task_random_streams(
    model: tf.keras.Model | None,
    task_seed: int | None
) -> None:
    """Reset cached execution functions and stochastic component streams for a task.

    Traversal follows layers and TensorFlow tracked modules while visiting shared objects
    once. Keras train/test/predict functions are cleared; explicit task seeds reset legacy
    random-layer generators and repository layer seeds. The wrapper seed and VAE
    reparameterization seed are also updated when present. Weights and optimizer slots are
    retained. This compensates for Python tracing counters that TensorFlow checkpoints do
    not save.

    Args:
        model (tf.keras.Model | None): Model tree to reset; None performs no work.
        task_seed (int | None): Task master seed; None clears model execution caches without
            installing derived seeds.

    Returns:
        None: The supplied model, nested models, and discovered stochastic layers are
        updated in place.
    """

    # An absent optional model has no random streams to reset.
    if model is None:
        return

    components = []
    pending = [model]
    discovered: set[int] = set()
    while pending:
        component = pending.pop()
        # Skip shared components already visited through another parent.
        if id(component) in discovered:
            continue
        discovered.add(id(component))
        components.append(component)
        children = list(getattr(component, "layers", ()))
        # Traverse tracked TensorFlow modules; ignore scalar and metadata trackables.
        children.extend(
            child for child in getattr(component, "_self_tracked_trackables", ())
            if isinstance(child, tf.Module)
        )
        pending.extend(reversed(children))

    for component_index, component in enumerate(components):
        # Keras models need cached functions cleared before task-specific retracing.
        if isinstance(component, tf.keras.Model):
            component.train_function = None
            component.test_function = None
            component.predict_function = None

        component_seed = derive_seed(
            task_seed,
            "keras_random_component",
            component_index,
            _qualified_name(component)
        )
        random_generator = getattr(component, "_random_generator", None)

        # Seeded Keras random layers carry a private generator counter to reset.
        if random_generator is not None and component_seed is not None:
            # Legacy Keras random layers increment this private Python counter
            # while tracing; checkpoints intentionally omit it.
            random_generator._seed = component_seed
            generator = getattr(random_generator, "_generator", None)
            # Reset the underlying TensorFlow generator when that reset API exists.
            if generator is not None and hasattr(generator, "reset_from_seed"):
                generator.reset_from_seed(component_seed)

        # Reset repository stochastic layers such as DropPath, which use their
        # public seed directly instead of Keras BaseRandomLayer.
        if component_seed is not None and random_generator is None \
        and hasattr(component, "seed"):
            component.seed = component_seed

    # Propagate a defined task seed to wrappers exposing a public seed.
    if task_seed is not None and hasattr(model, "seed"):
        model.seed = int(task_seed)
    # VAEs with a reparameterization stream receive a separate derived seed.
    if task_seed is not None and hasattr(model, "reparameterization_seed"):
        model.reparameterization_seed = derive_seed(
            task_seed,
            "vae",
            "reparameterization"
        )


def _prepare_diffusion_x(
    x: np.ndarray,
    data_min: float,
    data_range: float
) -> np.ndarray:
    """Convert loader-space arrays to the diffusion model input representation.

    The transformation is affine and does not clip out-of-range values. A loader already
    producing [-1, 1] uses data_min=-1 and data_range=2. Mixed precision uses stable
    variable precision rather than the lower compute dtype.

    Args:
        x (np.ndarray): Numeric input array, normally NHWC images or NHW grayscale images.
        data_min (float): Lower endpoint of the shared loader preprocessing scale.
        data_range (float): Nonzero width of the shared loader preprocessing scale.

    Returns:
        np.ndarray: ``2 * (x - data_min) / data_range - 1`` in the active Keras policy's
        variable dtype. NHW input gains a final singleton channel axis; other ranks retain
        their shape.
    """

    variable_dtype = tf.keras.mixed_precision.global_policy().variable_dtype
    numpy_dtype = tf.as_dtype(variable_dtype).as_numpy_dtype
    x = np.asarray(x, dtype=numpy_dtype)
    # Grayscale batches without a channel axis need a trailing singleton axis.
    if x.ndim == 3:
        x = x[..., None]

    return ((x - data_min) / data_range * 2.) - 1.


def _label_ids(labels: np.ndarray | None) -> np.ndarray | None:
    """Read sparse or one-hot labels as a flat class-ID vector.

    This helper only changes label representation. It does not remap class vocabulary,
    validate probability values, or force an integer dtype.

    Args:
        labels (np.ndarray | None): Sparse vector, sparse single-column array, multi-column
            one-hot/probability array, or None.

    Returns:
        np.ndarray | None: Argmax IDs for multiple columns, flattened values otherwise, or
        None for missing labels.
    """

    # An absent split has no labels to convert.
    if labels is None:
        return None
    labels = np.asarray(labels)
    # Decode one-hot rows with argmax; flatten sparse and column-vector labels.
    return labels.argmax(axis=-1) if labels.ndim > 1 and labels.shape[-1] > 1 \
        else labels.reshape(-1)


def _select_classes(
    x: np.ndarray,
    y: np.ndarray,
    classes: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Select examples belonging to specified classes without changing their representation.

    Class membership uses sparse IDs or the argmax of multi-column labels; preprocessing and
    class numbering remain unchanged.

    Args:
        x (np.ndarray): Input array with one row per label.
        y (np.ndarray): Aligned sparse or one-hot label array.
        classes (Sequence[int]): Class IDs to retain in the labels' current vocabulary.

    Returns:
        tuple[np.ndarray, np.ndarray]: Selected inputs and labels in original row order,
        preserving feature shape and label representation. No matches return empty arrays
        with the corresponding trailing dimensions.
    """

    selected = np.isin(_label_ids(y), classes)
    return np.asarray(x)[selected], np.asarray(y)[selected]


def _remap_continual_labels(
    labels: np.ndarray | None,
    class_order: Sequence[int],
    onehot_labels: bool
) -> np.ndarray | None:
    """Map original dataset classes to contiguous positions in the continual schedule.

    All supplied labels must belong to the schedule. This establishes the same dense
    vocabulary for training, replay, and task evaluation.

    Args:
        labels (np.ndarray | None): Sparse label array, or a multi-column label array when
            onehot_labels=True; None denotes an absent split.
        class_order (Sequence[int]): Original class IDs in their definitive introduction
            order.
        onehot_labels (bool): True emits one-hot rows of total scheduled width; False
            restores the input sparse shape and dtype.

    Returns:
        np.ndarray | None: Remapped labels in schedule order, or None for an absent split.
        One-hot output uses the input label dtype.

    Raises:
        ValueError: A dataset row contains an original label absent from class_order.
    """

    # Keep missing validation labels absent during schedule remapping.
    if labels is None:
        return None

    label_array = np.asarray(labels)
    label_ids = _label_ids(label_array)
    mapping = {label: index for index, label in enumerate(class_order)}
    try:
        remapped = np.asarray([mapping[label.item()] for label in label_ids])
    except KeyError as error:
        raise ValueError(
            f"Dataset returned unscheduled class label {error.args[0]!r}."
        ) from error

    # Recreate one-hot rows only when the loader requested one-hot labels.
    if onehot_labels:
        return np.eye(len(class_order), dtype=label_array.dtype)[remapped]

    return remapped.astype(label_array.dtype).reshape(label_array.shape)


def _predict_diffusion_classes(
    model: DiffusionClassifier,
    x: np.ndarray,
    y: np.ndarray,
    data_min: float,
    data_range: float,
    batch_size: int
) -> np.ndarray:
    """Predict primary-head class scores using the wrapper-selected evaluation network.

    V1 predicts at timestep zero with null labels. V2 applies its configured classifier
    test-noising cap through prep_clfv2_inputs. Both select raw or EMA through
    model.test_network_name and call predict_class with training=False. Inputs must be
    nonempty; task orchestration skips empty evaluation splits. This is ordinary inference,
    not timestep ensembling or a combined-head metric.

    Args:
        model (DiffusionClassifier): DiffusionClassifier or V2 wrapper with its current
            vocabulary and test-network selector.
        x (np.ndarray): Loader-space image array with N rows.
        y (np.ndarray): Aligned sparse class IDs; V2 uses these while preparing classifier
            inputs.
        data_min (float): Loader-space lower endpoint passed to diffusion input preparation.
        data_range (float): Loader-space scale width passed to diffusion input preparation.
        batch_size (int): Maximum examples per inference call; must permit positive range
            steps.

    Returns:
        np.ndarray: Concatenated primary class scores shaped (N, current_class_count).
    """

    x = _prepare_diffusion_x(x, data_min, data_range)
    y = np.asarray(y).reshape(-1)
    network = model.get_network(model.test_network_name)
    predictions = []

    for start in range(0, len(x), batch_size):
        end = start + batch_size
        x_batch = x[start:end]

        # V2 prepares classifier inputs with its configured test-time noising limit.
        if isinstance(model, DiffusionClassifierV2):
            t_batch, x_batch, null_labels, _ = model.prep_clfv2_inputs(
                (x_batch, y[start: end]),
                model.clf_test_noisified_max_timesteps
            )
        # V1 ordinary accuracy uses clean inputs at timestep zero and CFG-null labels.
        else:
            t_batch = np.zeros((len(x_batch),), dtype="int32")
            null_labels = np.zeros((len(x_batch),), dtype="uint8")

        predictions.append(network.predict_class(
            (x_batch, t_batch, null_labels),
            max_encoder_num=None,
            training=False
        ).numpy())

    return np.concatenate(predictions, axis=0)


def _ensemble_accuracy_row(
    model: DiffusionClassifier,
    x: np.ndarray,
    y: np.ndarray,
    learned_groups: Sequence[Sequence[int]],
    total_group_num: int,
    data_min: float,
    data_range: float,
    batch_size: int,
    options: dict[str, object],
    seed: int | None,
    verbose: bool | int
) -> list[float]:
    """Evaluate timestep-ensemble accuracy separately for the learned task groups.

    Each group gets an unshuffled diffusion-space dataset. Options are copied before a
    fallback seed is inserted, so the caller mapping is unchanged. The returned row is later
    macro-averaged across tasks, not weighted by group size.

    Args:
        model (DiffusionClassifier): Diffusion classifier wrapper exposing
            evaluate_ensemble_accuracy.
        x (np.ndarray): Loader-space images from one evaluation split.
        y (np.ndarray): Aligned sparse or one-hot labels in the dense scheduled vocabulary.
        learned_groups (Sequence[Sequence[int]]): Already introduced task groups, in
            schedule order.
        total_group_num (int): Width of the full continual accuracy matrix, including future
            tasks.
        data_min (float): Lower endpoint used to transform evaluation images.
        data_range (float): Scale width used to transform evaluation images.
        batch_size (int): Evaluation dataset batch size; final partial batches are retained.
        options (dict[str, object]): EnsembleAccuracy options such as network_name, max_t,
            weighting, head coefficients, compute mode, and seed.
        seed (int | None): Fallback ensemble seed used only when options has no non-None
            seed.
        verbose (bool | int): Whether each wrapper ensemble evaluation prints progress.

    Returns:
        list[float]: One scalar per scheduled task. Observed learned groups receive their
        actual ensemble accuracy; future or empty groups remain NaN.
    """

    label_ids = _label_ids(y)
    row = [np.nan] * total_group_num
    ensemble_options = dict(options)

    # Use the task-derived ensemble seed only when no explicit seed was supplied.
    if ensemble_options.get("seed") is None:
        ensemble_options["seed"] = seed

    for group_index, group in enumerate(learned_groups):
        selected = np.isin(label_ids, group)
        # A task group absent from this split retains its unavailable accuracy cell.
        if not np.any(selected):
            continue

        dataset = get_dataset(
            _prepare_diffusion_x(x[selected], data_min, data_range),
            label_ids[selected],
            shuffle_buffer=0,
            batch_size=batch_size,
            drop_remainder=False
        )
        row[group_index] = model.evaluate_ensemble_accuracy(
            dataset,
            verbose=bool(verbose),
            **ensemble_options
        )

    return row


def _has_positive_distillation_objective(model: DiffusionModel) -> bool:
    """Determine whether a diffusion wrapper has an active teacher-dependent loss.

    Regularizer training modes distil and both require teacher output; normal mode does not.
    Classifier-specific regularizer kwargs take precedence over the raw network's generic
    token-regularizer settings. Coefficients are read eagerly and are not changed.

    Args:
        model (DiffusionModel): Compiled diffusion wrapper with noise, classification, and
            token-regularizer coefficients.

    Returns:
        bool: True for positive noise or classifier distillation, or a positive
        teacher-trained classifier regularizer.
    """

    noise_distil_loss_coef = float(tf.keras.backend.get_value(
        model.noise_distil_loss_coef
    ))
    clf_distil_loss_coef = float(tf.keras.backend.get_value(
        getattr(model, "clf_distil_loss_coef", 0.)
    ))
    ctr_loss_coef = float(tf.keras.backend.get_value(model.ctr_loss_coef))
    regularizer_kwargs = getattr(
        model.network, "clf_cls_token_regularizer_kwargs", None
    )
    # A missing classifier-specific regularizer mapping uses the shared mapping.
    if regularizer_kwargs is None:
        regularizer_kwargs = getattr(model.network, "cls_token_regularizer_kwargs", {})

    uses_teacher_regularizer = (
        isinstance(model, DiffusionClassifier)
        and
        ctr_loss_coef > 0.
        and regularizer_kwargs.get("train_type", "normal") in ("distil", "both")
    )
    return noise_distil_loss_coef > 0. or clf_distil_loss_coef > 0. \
        or uses_teacher_regularizer


def _flatten_example_rows(values: np.ndarray) -> np.ndarray:
    """Flatten feature axes while preserving the example axis, including empty splits.

    An explicit feature count allows N=0, unlike an ambiguous (0, -1) reshape. Values are
    not scaled, copied deliberately, or shuffled.

    Args:
        values (np.ndarray): Array with a leading example axis and trailing feature
            dimensions.

    Returns:
        np.ndarray: View or reshape result shaped (N, product(feature_dimensions)),
        preserving dtype.
    """

    array = np.asarray(values)
    feature_count = int(np.prod(array.shape[1:], dtype=np.int64))

    return array.reshape((len(array), feature_count))


def _load_continual_arrays(
    load_dataset_fn: DatasetLoader,
    class_order: Sequence[int],
    return_features: bool,
    load_dataset_fn_kwargs: dict[str, object],
    max_train_samples: int | None,
    max_val_samples: int | None,
    pad: int,
    seed: int | None
) -> tuple[DatasetArrays, np.random.Generator]:
    """Load, cap, pad, and relabel the shared arrays used by every continual task.

    Padding uses -1 for standardize/diffusion preprocessing and zero otherwise. Labels are
    remapped to schedule positions while preserving the requested sparse/one-hot
    representation. The returned RNG is subsequently used by the task runner and is included
    in recovery state. Sample caps preserve at least one row per present class.

    Args:
        load_dataset_fn (DatasetLoader): Loader accepting indices, return_features, verbose,
            and the supplied loader options.
        class_order (Sequence[int]): Original class IDs in resolved schedule order, used for
            filtering and dense-label remapping.
        return_features (bool): Whether to request stored features instead of ordinary
            images.
        load_dataset_fn_kwargs (dict[str, object]): Loader options including explicit
            preprocess and onehot_labels entries.
        max_train_samples (int | None): Global training-row cap; None retains all selected
            training rows.
        max_val_samples (int | None): Global validation-row cap; None retains the selected
            validation split.
        pad (int): Symmetric image border width; positive values pad images, zero leaves
            shapes unchanged.
        seed (int | None): NumPy sampling seed; None uses an independently initialized
            generator.

    Returns:
        tuple[DatasetArrays, np.random.Generator]: Prepared six-array loader tuple and its
        advanced local generator. Validation may remain absent; test rows are neither
        sample-limited nor used to compute these caps.

    Raises:
        ValueError: A requested sample cap cannot retain every present class, or a label is
        outside the schedule.
    """

    (all_x_train, all_y_train, all_x_val, all_y_val,
     all_x_test, all_y_test) = load_dataset_fn(
        indices=list(class_order),
        return_features=return_features,
        **load_dataset_fn_kwargs,
        verbose=0
    )

    rng = np.random.default_rng(seed)
    all_x_train, all_y_train = _limit_samples(
        all_x_train,
        all_y_train,
        max_train_samples,
        rng
    )
    # Limit validation rows only when a validation split exists.
    if all_x_val is not None:
        all_x_val, all_y_val = _limit_samples(
            all_x_val,
            all_y_val,
            max_val_samples,
            rng
        )

    # Apply spatial padding only when a nonzero border was requested.
    if pad > 0:
        # Use -1 for diffusion-scaled borders; other preprocessing uses zero.
        pad_value = -1. if str(
            load_dataset_fn_kwargs["preprocess"]
        ).lower() in ("standardize", "diffusion") else 0.
        all_x_train = _pad_images(np.asarray(all_x_train), pad, value=pad_value)
        all_x_test = _pad_images(np.asarray(all_x_test), pad, value=pad_value)
        # Pad validation images only when that optional split exists.
        if all_x_val is not None:
            all_x_val = _pad_images(np.asarray(all_x_val), pad, value=pad_value)

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
        (all_x_train, all_y_train, all_x_val, all_y_val, all_x_test, all_y_test),
        rng
    )


def _sample_diffusion_replay(
    generative_model: DiffusionModel,
    labels: np.ndarray,
    batch_size: int,
    seed: int | None,
    empty_samples: np.ndarray,
) -> np.ndarray:
    """Generate aligned replay images in bounded label-conditioned batches.

    Each sampling call uses model.test_network_name and adds one to class labels when CFG
    reserves zero for the null label. No generator call occurs for an empty request.

    Args:
        generative_model (DiffusionModel): Diffusion wrapper whose selected test network
            supplies replay samples.
        labels (np.ndarray): Flat old-class IDs in the learner vocabulary, before the
            optional CFG offset.
        batch_size (int): Maximum labels per sampling call.
        seed (int | None): Task/candidate seed from which an independent stream is derived
            for each chunk. None is forwarded to sample, which falls back to the model seed.
        empty_samples (np.ndarray): Correctly shaped empty loader array returned by identity
            when no labels are requested.

    Returns:
        np.ndarray: Generated samples concatenated in label order, or the supplied
        empty_samples object. The wrapper sample API supplies [0, 1] image values;
        conversion back to loader space is performed by the task runner.
    """

    chunks = []
    for chunk_index, start in enumerate(range(0, len(labels), batch_size)):
        chunk_labels = labels[start:start + batch_size]
        chunks.append(generative_model.sample(
            network_name=generative_model.test_network_name,
            labels=chunk_labels + int(generative_model.use_cfg),
            seed=derive_seed(seed, "replay_sample_chunk", chunk_index),
        ).numpy())

    # Concatenate generated replay chunks; preserve the empty sample shape otherwise.
    return np.concatenate(chunks, axis=0) if chunks else empty_samples


def _predict_teacher_probabilities(
    teacher: tf.keras.Model,
    x: np.ndarray,
    data_min: float,
    data_range: float,
    batch_size: int
) -> np.ndarray:
    """Score replay or probe images with a fixed diffusion classifier network.

    Images are transformed to diffusion space and evaluated at timestep zero with null
    labels and training=False. This helper does not change teacher weights, apply
    temperature scaling, or ensemble classifier heads.

    Args:
        teacher (tf.keras.Model): Raw network exposing predict_class, normally a frozen
            previous-task snapshot.
        x (np.ndarray): Loader-space images to score, including a valid empty array.
        data_min (float): Lower endpoint of the loader preprocessing scale.
        data_range (float): Nonzero width of the loader preprocessing scale.
        batch_size (int): Maximum examples in each inference call.

    Returns:
        np.ndarray: Primary class probabilities shaped (N, teacher_class_count). For N=0, an
        empty float32 matrix uses teacher.num_classes when available.
    """

    diffusion_x = _prepare_diffusion_x(x, data_min, data_range)
    predictions = []
    for start in range(0, len(diffusion_x), batch_size):
        batch = diffusion_x[start:start + batch_size]
        timesteps = np.zeros((len(batch),), dtype="int32")
        null_labels = np.zeros((len(batch),), dtype="uint8")
        predictions.append(np.asarray(teacher.predict_class(
            (batch, timesteps, null_labels),
            max_encoder_num=None,
            training=False
        )))

    # An empty probe returns an empty matrix with the teacher's class width.
    if not predictions:
        output_width = int(getattr(teacher, "num_classes", 0) or 0)

        return np.empty((0, output_width), dtype="float32")

    return np.concatenate(predictions, axis=0)


def _resolve_baseline_controls(
    baseline: str | None,
    generative_model: tf.keras.Model | None,
    use_buffer: bool,
    buffer_kwargs: dict[str, object],
    remove_prev_classes: bool,
    use_generative_replay: bool,
    use_generative_model_classifier: bool,
    use_distillation: bool,
) -> tuple[str | None, bool, bool, bool, bool, bool, dict[str, object]]:
    """Translate a named continual baseline into the existing experiment switches.

    Sequential and cumulative use only a standalone classifier; cumulative keeps old real
    examples. Reservoir ER uses a bounded reservoir. VAE/diffusion replay use generator-only
    models with a standalone classifier. LwF enables an attached diffusion classifier and
    teacher learning without generated rehearsal. Joint none/replay/KD/both select the
    corresponding rehearsal and distillation switches. Without a named baseline, inputs are
    returned as supplied.

    Args:
        baseline (str | None): Named baseline or None to retain all caller-selected
            switches.
        generative_model (tf.keras.Model | None): Replay generator or attached diffusion
            classifier whose family must match a named baseline.
        use_buffer (bool): Caller-selected fixed-buffer replay switch, retained only when no
            baseline overrides it.
        buffer_kwargs (dict[str, object]): Buffer settings; named reservoir_er forces
            strategy=reservoir on a copied mapping.
        remove_prev_classes (bool): Whether task training is restricted to newly introduced
            classes before replay.
        use_generative_replay (bool): Whether previously learned classes receive generated
            rehearsal rows.
        use_generative_model_classifier (bool): Whether the attached diffusion classifier
            supplies task predictions.
        use_distillation (bool): Whether completed diffusion students become frozen teachers
            for subsequent tasks.

    Returns:
        tuple: ``(baseline, remove_prev_classes, use_buffer, use_generative_replay,
        use_generative_model_classifier, use_distillation, buffer_kwargs)``. baseline is
        normalized to lowercase unless it is None.

    Raises:
        ValueError: The baseline name is unsupported or its model family contradicts the
        requested control.
    """

    # Without a named baseline, retain the caller's individual continual switches.
    if baseline is None:
        return (
            None, remove_prev_classes, use_buffer, use_generative_replay,
            use_generative_model_classifier, use_distillation, buffer_kwargs,
        )

    supported = {
        "sequential", "cumulative", "reservoir_er", "lwf", "vae_replay",
        "diffusion_replay", "joint_none", "joint_replay", "joint_kd",
        "joint_both"
    }
    baseline = str(baseline).lower()
    # Reject a baseline name outside the implemented research controls.
    if baseline not in supported:
        raise ValueError(f"Unsupported continual baseline: {baseline!r}.")

    buffer_kwargs = dict(buffer_kwargs)
    remove_prev_classes = baseline != "cumulative"

    use_buffer = baseline == "reservoir_er"
    use_generative_replay = baseline in (
        "vae_replay", "diffusion_replay", "joint_replay", "joint_both"
    )
    use_generative_model_classifier = baseline in (
        "lwf", "joint_none", "joint_replay", "joint_kd", "joint_both"
    )
    use_distillation = baseline in ("lwf", "joint_kd", "joint_both")
    # The reservoir baseline fixes the buffer strategy to reservoir sampling.
    if use_buffer:
        buffer_kwargs["strategy"] = "reservoir"

    # Sequential, cumulative, and reservoir controls use a standalone classifier.
    if baseline in ("sequential", "cumulative", "reservoir_er"):
        # A generator would add an undeclared treatment to a classifier-only baseline.
        if generative_model is not None:
            raise ValueError(f"{baseline} requires no generative_model.")
    # LwF and joint controls use the classifier attached to a diffusion model.
    elif use_generative_model_classifier:
        # Reject models without a supported diffusion classifier for these controls.
        if not isinstance(generative_model, (
            DiffusionClassifier, DiTClassifier,
            DiTEncoderDecoderClassifier, UNetClassifier,
        )):
            raise ValueError(f"{baseline} requires a DiffusionClassifier.")
    # VAE replay uses a generator-only VAE with a separate classifier.
    elif baseline == "vae_replay":
        # Keep this baseline generator-only; VAEClassifier adds an unmatched
        # auxiliary classifier objective and belongs in a separate ablation.
        if not isinstance(generative_model, VariationalAutoencoder) \
        or isinstance(generative_model, VAEClassifier):
            raise ValueError("vae_replay requires a generator-only VAE.")
    # Diffusion replay uses a generator-only diffusion model and separate classifier.
    elif baseline == "diffusion_replay":
        # Reject non-diffusion generators and classifier-bearing diffusion variants here.
        if not isinstance(generative_model, (
            DiffusionModel, DiTDecoder, DiTEncoderDecoder,
            DiffusionTransformer, UNet,
        )) or isinstance(generative_model, (
            DiffusionClassifier, DiTClassifier,
            DiTEncoderDecoderClassifier, UNetClassifier,
        )):
            raise ValueError(
                "diffusion_replay requires a generator-only DiffusionModel."
            )
    return (
        baseline, remove_prev_classes, use_buffer, use_generative_replay,
        use_generative_model_classifier, use_distillation, buffer_kwargs
    )


def _run_continual_tasks(
    class_num: int,
    load_dataset_fn: DatasetLoader,
    class_order: Sequence[int] | None = None,
    task_groups: Sequence[Sequence[int]] | None = None,
    task_size: int = 1,
    class_order_mode: str = "fixed",
    task_order_mode: str = "fixed",
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
    baseline: str | None = None,
    plot_results: bool = True,
    verbose: bool | int = True,
    generative_model: tf.keras.Model | None = None,
    teacher_network: tf.keras.Model | None = None,
    generative_model_compile_args: dict[str, object] | None = None,
    generative_model_kwargs: dict[str, int] | None = None,
    use_generative_replay: bool = True,
    replay_budget_mode: str = "legacy",
    replay_old_examples: int | None = None,
    replay_current_examples: int | None = None,
    replay_candidate_multiplier: int = 1,
    replay_selection: str = "all",
    replay_surprise_weight: float = 0.5,
    replay_cache_dir: str | None = None,
    replay_cache_mode: str = "off",
    mechanistic_metrics: bool = False,
    mechanistic_max_samples: int = 512,
    use_generative_model_classifier: bool = False,
    train_classifier_separately: bool = False,
    use_distillation: bool = False,
    snapshot_network_name: str = "raw",
    use_ensemble_accuracy: bool = False,
    evaluate_ensemble_accuracy: bool = False,
    ensemble_accuracy_kwargs: dict[str, object] | None = None,
    callbacks_list: Sequence[tf.keras.callbacks.Callback] | None = None,
    generative_callbacks_list: Sequence[tf.keras.callbacks.Callback] | None = None,
    return_details: bool = True,
    use_valset: bool = True,
    return_features: bool | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    shuffle_buffer: int | None = None,
    pad: int = 0,
    deterministic_ops: bool = False,
    initial_classifier: tf.keras.Model | None = None,
    callback_patience: int | None = None,
    callback_monitor: str | None = None,
    callback_monitor_mode: str | None = None,
    save_task_checkpoints: bool = False,
    checkpoint_dir: str | None = None,
    resume_from: str | None = None,
    experiment_phase: str = "legacy",
    experiment_manifest_path: str | None = None,
    experiment_manifest_hash: str | None = None,
    experiment_run_id: str | None = None,
    optimizer_steps_per_epoch: int | None = None,
    dtype_policy: str | None = None,
    seed: int | None = None
) -> list[float] | dict[str, object]:
    """Train and evaluate a class-incremental schedule with optional replay, distillation, and recovery.

    The schedule establishes dense class order before any generator subsampling. Each task
    selects current real rows, optionally adds buffered or generated old rows, trains the
    appropriate phases, evaluates learned groups, and snapshots the completed student when
    distillation is enabled. V1 trains jointly. V2 fits its generator and, when
    use_generative_model_classifier=True, then fits its discriminator with a separate
    optimizer. Otherwise an external classifier is trained alongside the V2 generator-only
    phase. Replay-only classifier distillation carries an explicit row-provenance mask.

    Diffusion input scaling is shared across tasks and replay. Literal preprocess values
    'standardize'/'diffusion' use data_min=-1 and data_range=2; 'min-max' uses 0 and 1.
    Other values infer the minimum and range from the first scheduled task's capped, padded
    training rows, substituting 1 for a zero observed range. The resulting affine map
    converts that loader scale to diffusion-space [-1, 1] without clipping later rows to the
    first task's observed bounds.

    Task matrices use per-task accuracies, so derived continual metrics weight tasks
    equally. Ordinary cumulative accuracy weights examples; ensemble trajectory values
    average observed task scores. A one-class softmax's initial classification score is
    unavailable. Development excludes locked test arrays from model calls and recovery
    fingerprints. Confirmation validates its frozen schedule and seed before loading the
    experiment data.

    Models, optimizer state, callback state, and global random/dtype configuration are
    mutated. Plotting and checkpoint writes follow their controls. Resumption reconstructs
    dynamic topology and restores the last committed task boundary; an interrupted task is
    rerun with its original task-local random streams.

    Args:
        class_num (int): Number of scheduled classes; must agree with the resolved schedule
            or checkpoint.
        load_dataset_fn (DatasetLoader): Six-array loader accepting indices,
            return_features, verbose, and loader-specific options.
        class_order (Sequence[int] | None): Original label introduction order; None derives
            it from task_groups or range(class_num). Defaults to ``None``.
        task_groups (Sequence[Sequence[int]] | None): Explicit groups of original labels;
            None partitions class_order by task_size. Defaults to ``None``.
        task_size (int): Classes per automatically formed task; a final short group is
            allowed. Defaults to ``1``.
        class_order_mode (str): fixed preserves class identity order; random permutes
            classes using the master seed. Defaults to ``'fixed'``.
        task_order_mode (str): fixed preserves group order; random permutes complete task
            groups. Defaults to ``'fixed'``.
        load_dataset_fn_kwargs (dict[str, object] | None): Extra loader options; None starts
            with preprocess=None and onehot_labels=False. The preprocess value also
            determines the shared diffusion input scale as described below. Defaults to
            ``None``.
        remove_prev_classes (bool): True trains on new-task real rows plus replay; False
            retains all seen real classes. Defaults to ``True``.
        keep_same_model (bool): True copies the preceding standalone classifier into each
            expanded head; False starts each task from the template or initial_classifier.
            Defaults to ``True``.
        tuned_model_path (str): Saved classifier architecture used for standalone head
            expansion; ignored when the attached diffusion classifier is selected. Defaults
            to ``''``.
        compile_args (dict[str, object] | None): Standalone classifier compile options; None
            uses factory defaults. Explicit optimizer instances are cloned for this run.
            Defaults to ``None``.
        use_loaded_opt (bool): Whether standalone classifier construction rebuilds the
            optimizer configuration from its saved template. Defaults to ``False``.
        batch_size (int): Number of examples per training/evaluation/replay-generation
            batch; partial task batches are retained. Defaults to ``128``.
        epochs (int): Ordinary per-phase epochs; progressive generator phases instead use
            their stage/final budgets. Defaults to ``100``.
        fit_method (str): fit uses ordinary training; fit_progressively requires a diffusion
            model and a stage_tasks curriculum. Defaults to ``'fit'``.
        fit_kwargs (dict[str, object] | None): Diffusion generator-fit options only; VAE and
            classifier phases use their own options. None uses an empty mapping.
            Orchestration-owned arguments cannot be overridden; use
            optimizer_steps_per_epoch for a budget shared by all phases. Defaults to
            ``None``.
        use_buffer (bool): Enable bounded real-example replay; mutually exclusive with a
            generative model. Defaults to ``False``.
        buffer_kwargs (dict[str, object] | None): Buffer options. None resolves
            maxlen=10000, sample_num=1000, insert_num=1000, seed=None, and strategy=fifo.
            Defaults to ``None``.
        baseline (str | None): Optional named treatment resolved by
            _resolve_baseline_controls; None preserves explicit switches. Defaults to
            ``None``.
        plot_results (bool): Whether to display the final accuracy trajectory after
            completing tasks. Defaults to ``True``.
        verbose (bool | int): Training/reporting verbosity and whether task summaries and
            history plots are displayed. Defaults to ``True``.
        generative_model (tf.keras.Model | None): Conditioned VAE, raw supported diffusion
            network, or diffusion wrapper; None selects standalone classification without a
            generator. Defaults to ``None``.
        teacher_network (tf.keras.Model | None): Optional first-task frozen diffusion
            teacher. None supplies no new teacher and preserves any teacher already attached
            to a supplied wrapper. Later automatic teachers come from completed students.
            Defaults to ``None``.
        generative_model_compile_args (dict[str, object] | None): Compile settings for raw
            diffusion networks wrapped here; None supplies optimizer=adam and loss=mse.
            Existing compiled wrappers retain their own compilation. Defaults to ``None``.
        generative_model_kwargs (dict[str, int] | None): Generator exposure settings. None
            supplies train_num=1000 and samples_per_class=1000. In legacy budgeting,
            diffusion generator train_num resamples the combined pool with replacement; -1
            retains it. fixed_total forces -1. V2 discriminator training retains the full
            combined pool independently of generator resampling. Defaults to ``None``.
        use_generative_replay (bool): Whether a supplied generator produces rehearsal rows
            for previously introduced classes. Defaults to ``True``.
        replay_budget_mode (str): legacy uses per-class generated counts or the buffer
            sample count; fixed_total uses explicit old/current exposure counts. Defaults to
            ``'legacy'``.
        replay_old_examples (int | None): Exact old-row budget in fixed_total mode; None is
            valid only for legacy budgeting. Defaults to ``None``.
        replay_current_examples (int | None): Current-row exposure budget in fixed_total
            mode; None retains the available current pool. Defaults to ``None``.
        replay_candidate_multiplier (int): Multiplier applied to the old-row budget before
            candidate selection. Defaults to ``1``.
        replay_selection (str): all preserves candidates up to the budget; uniform selects
            by class-balanced quotas; random samples without class balancing. confidence,
            surprise, and confidence_surprise rank teacher scores and require a diffusion
            classifier teacher. Defaults to ``'all'``.
        replay_surprise_weight (float): Relative surprise contribution in
            confidence_surprise selection. Must lie in [0, 1] whenever the selector is
            called; other strategies do not use it in their score. Defaults to ``0.5``.
        replay_cache_dir (str | None): Directory for authenticated generated-candidate
            archives; None disables a cache destination when caching is off. Defaults to
            ``None``.
        replay_cache_mode (str): off generates without caching; read requires a matching
            archive; write stores a pool; read_write reuses a matching pool or generates
            one. Defaults to ``'off'``.
        mechanistic_metrics (bool): Whether to compute bounded replay-quality, old-class
            probability drift, and held-out calibration diagnostics. Defaults to ``False``.
        mechanistic_max_samples (int): Maximum retained/probe rows for optional diagnostics
            and quadratic diversity comparisons. Defaults to ``512``.
        use_generative_model_classifier (bool): True uses the attached diffusion classifier
            for prediction; False trains an expanding external classifier. Defaults to
            ``False``.
        train_classifier_separately (bool): Compatibility input. Actual phase selection is
            inferred from the V2 wrapper rather than this value. Defaults to ``False``.
        use_distillation (bool): Whether each completed diffusion student becomes the next
            task's frozen teacher; requires an active teacher-dependent objective. Defaults
            to ``False``.
        snapshot_network_name (str): raw or ema branch used for automatic teachers and
            teacher-scored replay; ema requires an EMA-enabled wrapper. Defaults to
            ``'raw'``.
        use_ensemble_accuracy (bool): Make ensemble task matrices authoritative for
            continual metrics and public accuracy; also enables ensemble evaluation.
            Defaults to ``False``.
        evaluate_ensemble_accuracy (bool): Record ensemble matrices as diagnostics without
            changing the authoritative metric unless use_ensemble_accuracy is True. Defaults
            to ``False``.
        ensemble_accuracy_kwargs (dict[str, object] | None): Options forwarded to
            evaluate_ensemble_accuracy, including network, timesteps, weighting, heads,
            compute mode, and optional seed; None uses wrapper defaults. Defaults to
            ``None``.
        callbacks_list (Sequence[tf.keras.callbacks.Callback] | None): Callbacks shared by
            active fit phases; None adds no caller callbacks. Defaults to ``None``.
        generative_callbacks_list (Sequence[tf.keras.callbacks.Callback] | None): Additional
            generator-phase callbacks; None adds none. Supported artifact prefixes and seeds
            are reset per task. Defaults to ``None``.
        return_details (bool): True returns models, matrices, histories, and metadata; False
            returns only the selected task-accuracy trajectory. Defaults to ``True``.
        use_valset (bool): Whether to use the validation arrays returned by the loader;
            False discards them. This switch is not passed to the loader to create a split.
            Development mode requires a nonempty validation set. Defaults to ``True``.
        return_features (bool | None): True requests saved features; False uses images; None
            infers features from dnn in the tuned model path. Attached diffusion
            classification requires images. Defaults to ``None``.
        max_train_samples (int | None): Global cap applied once to scheduled training rows
            before task selection; None keeps all selected rows. Defaults to ``None``.
        max_val_samples (int | None): Global cap on the validation split before task
            selection; None keeps all validation rows. Defaults to ``None``.
        shuffle_buffer (int | None): Training shuffle capacity; None uses the active pool
            size, while zero disables shuffling. Defaults to ``None``.
        pad (int): Symmetric image-border width; zero leaves dimensions unchanged. Saved
            features do not support image padding. Defaults to ``0``.
        deterministic_ops (bool): Whether initial runtime setup requests deterministic
            TensorFlow operations. Defaults to ``False``.
        initial_classifier (tf.keras.Model | None): Optional source weights for the first
            standalone head, or every head when keep_same_model=False; None uses the saved
            template/previous head. Defaults to ``None``.
        callback_patience (int | None): Early-stopping patience override; None uses the
            phase default (five for classifier fits, zero for generators). Nonpositive
            values add no built-in early stopping. Defaults to ``None``.
        callback_monitor (str | None): Metric monitored by built-in phase callbacks; None
            uses the phase-specific accuracy or loss name. Defaults to ``None``.
        callback_monitor_mode (str | None): min/max comparison mode override; None follows
            the phase's accuracy/loss default. Defaults to ``None``.
        save_task_checkpoints (bool): Whether to commit state after each completed task when
            checkpoint_dir is available. Defaults to ``False``.
        checkpoint_dir (str | None): Task-checkpoint root; None is inferred from resume_from
            when resuming and otherwise supplies no save destination. Defaults to ``None``.
        resume_from (str | None): Checkpoint root or committed task directory; None starts a
            new experiment. Recovery validates descriptor, schedule, topology, and optimizer
            state. Defaults to ``None``.
        experiment_phase (str): development evaluates validation only; legacy also evaluates
            test; confirmation additionally authenticates the frozen manifest/run identity.
            Defaults to ``'legacy'``.
        experiment_manifest_path (str | None): Frozen experiment-manifest path required for
            confirmation; None is unused in other phases. Defaults to ``None``.
        experiment_manifest_hash (str | None): Expected frozen-manifest digest required for
            confirmation; None supplies no authentication digest. Defaults to ``None``.
        experiment_run_id (str | None): Unique planned condition/stream run identifier
            required for confirmation; None is unused outside that mode. Defaults to
            ``None``.
        optimizer_steps_per_epoch (int | None): Optional requested fit-batch budget per
            epoch for each active phase; early stopping can shorten training. None normally
            uses finite pool lengths, with fit_kwargs.steps_per_epoch able to override the
            diffusion generator alone. Cannot be combined with fit_kwargs.steps_per_epoch.
            Defaults to ``None``.
        dtype_policy (str | None): Keras precision policy; None retains the current global
            policy. Prebuilt components must already use compatible precision. Defaults to
            ``None``.
        seed (int | None): Master seed for task, data, replay, callback, model, and ensemble
            streams; None leaves stochastic initialization uncontrolled by a master seed.
            Defaults to ``None``.

    Returns:
        list[float] | dict[str, object]: With return_details=False, the selected per-task
        accuracy trajectory. Otherwise a mapping containing accuracies,
        ordinary/validation/ensemble accuracy matrices, phase histories/evaluations,
        resource/mechanistic diagnostics, original schedule, task seeds, continual metrics,
        trained classifier and generator references, run descriptor, and checkpoint
        paths/cursor. accuracy_matrix selects validation in development and test otherwise,
        using the ensemble matrix when requested. Development leaves test matrices empty and
        continual_metrics empty; its optimization targets live in
        validation_continual_metrics. Undefined singleton/future or unobserved task cells
        remain NaN rather than fabricated accuracies.

    Raises:
        ValueError: Schedule, baseline, split, teacher, replay, curriculum, or
            checkpoint compatibility requirements are violated.
        TypeError: The supplied generator is not a supported model family. Exception:
        Dataset, Keras training, callback, artifact, and checkpoint errors
            propagate from their owning APIs.
    """

    # A committed checkpoint owns the already-materialized stochastic
    # schedule. Reading it before schedule construction avoids changing class
    # order merely because NumPy's permutation implementation changes between
    # supported environments. Explicit caller schedules must still agree.
    if resume_from is not None:
        schedule_checkpoint = load_task_checkpoint(resume_from)
        saved_order = list(schedule_checkpoint.class_order)
        saved_groups = [list(group) for group in schedule_checkpoint.task_groups]
        # The requested class count must match the checkpoint's complete schedule.
        if int(class_num) != len(saved_order):
            raise ValueError(
                "Requested class_num differs from the checkpoint schedule."
            )

        # Compare an explicit order only when no stochastic mode transforms it.
        if class_order is not None and class_order_mode == "fixed" and \
        task_order_mode == "fixed" and fingerprint_state(
            list(class_order)
        ) != fingerprint_state(saved_order):
            raise ValueError(
                "Requested class_order differs from the checkpoint schedule."
            )

        # Explicit fixed task groups must match the saved task boundaries and order.
        if task_groups is not None and task_order_mode == "fixed" \
        and fingerprint_state([
            list(group) for group in task_groups
        ]) != fingerprint_state(saved_groups):
            raise ValueError(
                "Requested task_groups differ from the checkpoint schedule."
            )

        class_order, original_task_groups = saved_order, saved_groups
    # Materialize the requested schedule only for a new experiment.
    else:
        # Copy explicit schedules into lists; leave omitted schedules for automatic
        # resolution.
        class_order, original_task_groups = resolve_continual_schedule(
            class_num,
            list(class_order) if class_order is not None else None,
            [list(group) for group in task_groups]
            if task_groups is not None else None,
            task_size=task_size,
            class_order_mode=class_order_mode,
            task_order_mode=task_order_mode,
            seed=seed
        )

    authenticated_manifest_hash = None
    # Test evaluation is reachable only through a frozen externally verified
    # confirmation manifest and one exact planned stream/run identity.
    if str(experiment_phase).lower() == "confirmation":
        # Confirmation requires the manifest path, trusted hash, and selected run ID.
        if not all((
            experiment_manifest_path,
            experiment_manifest_hash,
            experiment_run_id,
        )):
            raise ValueError(
                "confirmation requires experiment_manifest_path, "
                "experiment_manifest_hash, and experiment_run_id."
            )
        from common.experiment import (
            materialize_run_plan,
            read_experiment_manifest,
            validate_frozen_confirmation,
        )

        frozen_manifest = read_experiment_manifest(
            experiment_manifest_path,
            expected_hash=experiment_manifest_hash,
        )
        frozen_manifest = validate_frozen_confirmation(
            frozen_manifest,
            expected_hash=experiment_manifest_hash,
        )
        # Select only the frozen run whose ID matches the requested confirmation run.
        matching_runs = [
            run for run in materialize_run_plan(
                frozen_manifest,
                expected_hash=experiment_manifest_hash,
            )
            if run["run_id"] == experiment_run_id
        ]
        # A run identifier must name exactly one frozen condition-stream cell.
        if len(matching_runs) != 1:
            raise ValueError("experiment_run_id is not unique in the frozen manifest.")
        planned_run = matching_runs[0]
        planned_stream = planned_run["stream"]
        # Refuse test access if the materialized schedule differs at all.
        if fingerprint_state(planned_stream["class_order"]) \
        != fingerprint_state(list(class_order)) \
        or fingerprint_state(planned_stream["task_groups"]) \
        != fingerprint_state(original_task_groups):
            raise ValueError(
                "Confirmation schedule differs from the frozen manifest run."
            )
        # The confirmation seed must match the preregistered stream seed.
        if planned_stream["stream_seed"] != seed:
            raise ValueError("Confirmation seed differs from the frozen manifest run.")
        authenticated_manifest_hash = frozen_manifest["manifest_hash"]

    dtype_policy = dtype_policy or tf.keras.mixed_precision.global_policy().name
    configure_runtime(seed, dtype_policy, deterministic_ops)
    _validate_supplied_model_runtime(generative_model, seed, "continual replay model")

    # Reject VAEClassifier because its auxiliary objective is not this replay protocol.
    if isinstance(generative_model, VAEClassifier):
        raise ValueError(
            "Continual VAEClassifier is unsupported because its fixed "
            "full-class head exposes future logits. Use a conditioned "
            "VariationalAutoencoder with the expanding external classifier."
        )

    class_num = len(class_order)
    group_sizes = [len(group) for group in original_task_groups]
    boundaries = np.cumsum([0, *group_sizes])
    internal_task_groups = [
        list(range(int(boundaries[index]), int(boundaries[index + 1])))
        for index in range(len(group_sizes))
    ]

    # Only ordinary and progressive fitting are supported by the continual loop.
    if fit_method not in ("fit", "fit_progressively"):
        raise ValueError("fit_method must be 'fit' or 'fit_progressively'.")

    fit_kwargs = dict(fit_kwargs or {})
    reserved_fit_keys = {
        "x", "y", "epochs", "initial_epoch",
        "validation_data", "callbacks", "verbose"
    }
    conflicting_fit_keys = sorted(reserved_fit_keys.intersection(fit_kwargs))
    # Reject fit arguments whose values are owned by task orchestration.
    if conflicting_fit_keys:
        raise ValueError(
            "fit_kwargs cannot override continual orchestration arguments: "
            + str(conflicting_fit_keys)
        )

    # Progressive fitting needs an explicitly configured stage curriculum.
    if fit_method == "fit_progressively" \
    and fit_kwargs.get("stage_tasks") is None:
        raise ValueError("fit_kwargs must include stage_tasks for fit_progressively.")

    from common.train import report, train_model


    def _train_task_model(
        model: tf.keras.Model,
        trainset: object,
        valset: object | None = None,
        task_callbacks: Sequence[tf.keras.callbacks.Callback] | None = None,
        fit_method: str = "fit",
        fit_kwargs: dict[str, object] | None = None,
    ) -> dict[str, list[float]]:
        """Fit one task phase through the shared training API.

        The closure uses outer epochs, verbosity, and optimizer_steps_per_epoch. A requested
        fit-batch budget repeats finite datasets without exposing an infinite public stream;
        progressive stages use their stage/final epoch budgets. Automatic per-call config,
        weight, GIF, and periodic report writes are disabled. Supplied callbacks may still
        perform their configured side effects. Model training and optimizer updates occur in
        place. Early stopping can finish before the requested fit-batch budget has been
        consumed.

        Args:
            model (tf.keras.Model): Standalone classifier, VAE, or diffusion wrapper for
                this phase.
            trainset (object): tf.data.Dataset for ordinary/diffusion training, or VAE array
                input.
            valset (object | None): Optional phase validation dataset or VAE (images,
                labels) pair; None disables validation. Defaults to ``None``.
            task_callbacks (Sequence[tf.keras.callbacks.Callback] | None): Callbacks
                attached to this fit; None supplies no extra phase callbacks. Defaults to
                ``None``.
            fit_method (str): Shared trainer method selector, including fit, train,
                fit_generator, fit_discriminator, or progressive variants. Defaults to
                ``'fit'``.
            fit_kwargs (dict[str, object] | None): Phase-specific Keras/VAE/curriculum
                options; None starts from an empty mapping. Defaults to ``None``.

        Returns:
            dict[str, list[float]]: Epoch histories returned by train_model with its
            normalized metric names.
        """

        phase_fit_kwargs = dict(fit_kwargs or {})
        phase_trainset = trainset
        # Repeat a finite task dataset only for the explicit fixed-update
        # protocol; VAE's array-based ``train`` method owns its own repetition.
        if optimizer_steps_per_epoch is not None:
            phase_fit_kwargs["steps_per_epoch"] = optimizer_steps_per_epoch
            # Dataset inputs can be repeated to supply the requested optimizer-step budget.
            if isinstance(phase_trainset, tf.data.Dataset):
                fixed_step_epochs = epochs
                # Progressive stages create separate fit iterators and own
                # their epoch budgets instead of using the outer value.
                if "progressively" in fit_method:
                    stage_epochs = int(phase_fit_kwargs.get("stage_epochs", 1))
                    final_epochs = phase_fit_kwargs.get("final_epochs")
                    # Use the stage epoch budget for an omitted final budget; otherwise use
                    # the explicit value.
                    final_epochs = stage_epochs if final_epochs is None \
                        else int(final_epochs)
                    fixed_step_epochs = max(stage_epochs, final_epochs, 1)
                phase_trainset = _finite_fixed_step_dataset(
                    phase_trainset,
                    optimizer_steps_per_epoch,
                    fixed_step_epochs,
                )

        return train_model(
            None,
            model,
            phase_trainset,
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
            fit_kwargs=phase_fit_kwargs
        )


    def _report_task_model(
        history: dict[str, list[float]],
        model: tf.keras.Model,
        trainset: object,
        evaluation_set: object,
        split_name: str = "testset",
    ) -> dict[str, object]:
        """Evaluate one task phase without producing ordinary report files.

        The closure uses outer verbosity. It disables CSV, final images/GIFs, and duplicate
        generic ensemble evaluation because the learner builds ensemble task matrices
        separately. Verbose mode may still display a history plot.

        Args:
            history (dict[str, list[float]]): Phase metric histories used by optional
                interactive history plots.
            model (tf.keras.Model): Trained phase model passed to the common reporter.
            trainset (object): Training input retained for the reporter signature; train-set
                evaluation is disabled.
            evaluation_set (object): Validation or allowed test split to evaluate through
                the reporter's valset parameter.
            split_name (str): Prefix replacing valset in returned report keys; development
                supplies valset and other phases use testset. Defaults to ``'testset'``.

        Returns:
            dict[str, object]: Nested phase evaluation values with split-appropriate key
            prefixes.
        """

        reported = report(
            None,
            history,
            model,
            trainset,
            valset=evaluation_set,
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
            # Per-task ensemble matrices are evaluated explicitly below; the
            # generic cumulative report would duplicate that expensive pass.
            evaluate_ensemble_accuracy=False,
            ensemble_accuracy_kwargs=ensemble_accuracy_kwargs,
            verbose=verbose
        )
        # The generic reporter always calls its second split ``valset``. Rename
        # it only when this is the locked confirmatory/legacy test split.
        # Rename validation-prefixed report keys for the selected split; preserve other keys.
        return {
            key.replace("valset", split_name, 1)
            if key.startswith("valset") else key: value
            for key, value in reported.items()
        }


    def _phase_callbacks(
        default_monitor: str,
        default_mode: str = "max",
        legacy_patience: int = 0,
        include_generative: bool = False
    ) -> list[tf.keras.callbacks.Callback]:
        """Assemble the callbacks appropriate to one active training phase.

        Positive effective patience creates built-in callbacks; nonpositive patience does
        not. Caller callback objects are reused, not cloned, and their order is retained.

        Args:
            default_monitor (str): Accuracy or loss metric used when the caller did not
                override callback_monitor.
            default_mode (str): Monitor direction used when callback_monitor_mode is absent.
                Defaults to ``'max'``.
            legacy_patience (int): Phase patience used only when the outer callback_patience
                is None. Defaults to ``0``.
            include_generative (bool): Whether to append generator-specific callbacks after
                shared caller callbacks. Defaults to ``False``.

        Returns:
            list[tf.keras.callbacks.Callback]: Built-in patience callbacks followed by the
            requested shared and generator callbacks.
        """

        # Use legacy patience only when no explicit callback patience was provided.
        patience = legacy_patience if callback_patience is None else callback_patience
        selected = []

        # Positive patience enables early stopping for this training phase.
        if patience > 0:
            selected = get_callbacks(
                monitor=callback_monitor or default_monitor,
                mode=callback_monitor_mode or default_mode,
                patience=patience,
                verbose=verbose
            )

        # Append caller callbacks when a callback collection was supplied.
        if callbacks_list is not None:
            selected += list(callbacks_list)

        # Generator phases also receive their optional generative callbacks.
        if include_generative and generative_callbacks_list is not None:
            selected += list(generative_callbacks_list)

        return selected


    load_dataset_fn_kwargs_default = {"preprocess": None, "onehot_labels": False}
    load_dataset_fn_kwargs = {
        **load_dataset_fn_kwargs_default,
        **(load_dataset_fn_kwargs or {})
    }

    compile_args = dict(compile_args or {})
    # Clone an explicitly configured optimizer before the task loop mutates it.
    if "optimizer" in compile_args:
        compile_args["optimizer"] = _fresh_optimizer(compile_args["optimizer"])

    ensemble_accuracy_kwargs = dict(ensemble_accuracy_kwargs or {})
    snapshot_network_name = str(snapshot_network_name).lower()
    # Restrict teacher snapshots to actual wrapper network branches.
    if snapshot_network_name not in ("raw", "ema"):
        raise ValueError("snapshot_network_name must be 'raw' or 'ema'.")

    # The legacy flag requests computation; the new flag additionally makes
    # ensemble results authoritative for continual metrics.
    evaluate_ensemble_accuracy = bool(
        evaluate_ensemble_accuracy or use_ensemble_accuracy
    )
    generative_model_compile_args = {
        "optimizer": "adam",
        "loss": "mse",
        **(generative_model_compile_args or {})
    }
    generative_model_compile_args["optimizer"] = _fresh_optimizer(
        generative_model_compile_args["optimizer"]
    )

    buffer_kwargs_default = {
        "maxlen": 10_000,
        "sample_num": 1_000,
        "insert_num": 1_000,
        "seed": None,
        "strategy": "fifo",
    }
    buffer_kwargs = {**buffer_kwargs_default, **(buffer_kwargs or {})}

    generative_model_kwargs_default = {"train_num": 1_000, "samples_per_class": 1_000}
    generative_model_kwargs = {
        **generative_model_kwargs_default,
        **(generative_model_kwargs or {})
    }

    (
        baseline,
        remove_prev_classes,
        use_buffer,
        use_generative_replay,
        use_generative_model_classifier,
        use_distillation,
        buffer_kwargs
    ) = _resolve_baseline_controls(
        baseline,
        generative_model,
        use_buffer,
        buffer_kwargs,
        remove_prev_classes,
        use_generative_replay,
        use_generative_model_classifier,
        use_distillation
    )

    experiment_phase = str(experiment_phase).lower()
    # Reject phases outside legacy, development, and confirmation reporting.
    if experiment_phase not in ("legacy", "development", "confirmation"):
        raise ValueError(
            "experiment_phase must be 'legacy', "
            "'development', or 'confirmation'."
        )

    replay_budget_mode = str(replay_budget_mode).lower()
    # Keep replay accounting under a known per-class or fixed-total contract.
    if replay_budget_mode not in ("legacy", "fixed_total"):
        raise ValueError("replay_budget_mode must be 'legacy' or 'fixed_total'.")

    # Normalize an explicit shared optimizer-step budget before routing it to phases.
    if optimizer_steps_per_epoch is not None:
        optimizer_steps_per_epoch = int(optimizer_steps_per_epoch)
        # Keep the dedicated all-phase control unambiguous with the historical
        # diffusion-only fit mapping.
        if "steps_per_epoch" in fit_kwargs:
            raise ValueError(
                "Set optimizer_steps_per_epoch or fit_kwargs.steps_per_epoch, "
                "not both."
            )

    # A matched old-data budget must be named explicitly in fixed-total mode.
    if replay_budget_mode == "fixed_total" and replay_old_examples is None:
        raise ValueError("fixed_total replay requires replay_old_examples to be set.")

    # Fixed-total replay uses its explicit count; legacy replay uses the buffer or per-class
    # generator count.
    old_replay_count = replay_old_examples \
        if replay_budget_mode == "fixed_total" else (
            buffer_kwargs["sample_num"] if use_buffer
            else generative_model_kwargs["samples_per_class"]
        )
    buffer_can_replay = use_buffer \
        and buffer_kwargs["maxlen"] > 0 \
        and (baseline == "reservoir_er" or buffer_kwargs["insert_num"] > 0)
    generator_can_replay = generative_model is not None \
        and use_generative_replay
    has_old_replay = old_replay_count > 0 \
        and (buffer_can_replay or generator_can_replay)
    # A discarded singleton first task needs old-data replay to become learnable later.
    if len(original_task_groups[0]) == 1 \
    and remove_prev_classes and not has_old_replay:
        raise ValueError(
            "A new-only singleton-first schedule cannot learn its one-way "
            "softmax before discarding that class. Use cumulative or a "
            "positive buffered/generative replay exposure instead."
        )
    replay_candidate_multiplier = int(replay_candidate_multiplier)
    replay_selection = str(replay_selection).lower()
    replay_selection_names = {
        "all", "uniform", "random", "confidence", "surprise",
        "confidence_surprise",
    }
    # Reject replay selectors outside the implemented candidate-selection methods.
    if replay_selection not in replay_selection_names:
        raise ValueError(
            "replay_selection must be one of "
            f"{sorted(replay_selection_names)}."
        )
    replay_surprise_weight = float(replay_surprise_weight)
    # Treat False as disabled caching; normalize named cache modes to lowercase.
    replay_cache_mode = "off" if replay_cache_mode is False \
        else str(replay_cache_mode).lower()
    mechanistic_max_samples = int(mechanistic_max_samples)
    # Reject simultaneous buffered and generated replay sources.
    if use_buffer and generative_model is not None:
        raise ValueError(
            "The replay buffer and a generative model cannot be used together."
        )
    # A positive fixed old-data budget requires an enabled replay source.
    if replay_budget_mode == "fixed_total" \
    and replay_old_examples > 0 \
    and not (use_buffer or (generative_model is not None and use_generative_replay)):
        raise ValueError(
            "A positive fixed old-example budget requires buffer or "
            "generative replay."
        )

    # Infer feature inputs from the tuned classifier name when the option is omitted.
    if return_features is None:
        return_features = "dnn" in str(tuned_model_path).lower()
    # An explicit feature-input option takes precedence over filename inference.
    else:
        return_features = bool(return_features)

    # Saved feature vectors cannot receive spatial image padding.
    if pad and return_features:
        raise ValueError("pad is not supported for saved feature inputs.")

    # Construct a replay buffer only for buffered-replay runs.
    if use_buffer:
        # Derive the buffer seed from the experiment when available; otherwise retain its
        # configured seed.
        buffer = ReplayBuffer(
            maxlen=buffer_kwargs["maxlen"],
            seed=derive_seed(seed, "replay_buffer")
                if seed is not None else buffer_kwargs["seed"],
            strategy=buffer_kwargs["strategy"]
        )

    # Wrap raw classifier-capable diffusion networks before training or replay.
    if isinstance(generative_model, (
        DiTClassifier,
        DiTEncoderDecoderClassifier,
        UNetClassifier
    )):
        # Enable the initial classifier distillation coefficient only for distilled runs.
        generative_model = DiffusionClassifier(
            network=generative_model,
            teacher_network=teacher_network,
            defer_teacher=use_distillation,
            clf_distil_loss_coef=8.6e-3 if use_distillation else 0.,
            mask_by_nulls=generative_model.use_cfg,
            test_steps=min(50, generative_model.timesteps),
            seed=seed,
        )
        generative_model.compile(**generative_model_compile_args)
    # Wrap raw generator-only diffusion networks.
    elif isinstance(generative_model, (
        DiTDecoder,
        DiTEncoderDecoder,
        DiffusionTransformer,
        UNet
    )):
        # Enable the initial noise distillation coefficient only for distilled runs.
        generative_model = DiffusionModel(
            network=generative_model,
            teacher_network=teacher_network,
            defer_teacher=use_distillation,
            noise_distil_loss_coef=1. if use_distillation else 0.,
            test_steps=min(50, generative_model.timesteps),
            seed=seed
        )
        generative_model.compile(**generative_model_compile_args)
    # Install an optional first-task teacher on an existing wrapper.
    elif teacher_network is not None and isinstance(
        generative_model,
        DiffusionModel
    ) and getattr(generative_model, "teacher_network", None) is not teacher_network:
        generative_model.set_teacher_network(teacher_network)
    # Reject supplied generators outside the supported VAE and diffusion families.
    elif generative_model is not None and not isinstance(
        generative_model,
        (VariationalAutoencoder, DiffusionModel)
    ):  # Reject objects that cannot provide supported generative replay.
        raise TypeError(
            "generative_model must be a supported VAE, "
            "diffusion network, or diffusion wrapper."
        )

    # Reject teachers for replay models without a diffusion wrapper.
    if teacher_network is not None and not isinstance(generative_model, DiffusionModel):
        raise ValueError("teacher_network requires a diffusion generative_model.")

    # Restrict automatic self-distillation to diffusion wrappers.
    if use_distillation and not isinstance(generative_model, DiffusionModel):
        raise ValueError("use_distillation requires a diffusion generative_model.")

    scored_replay_selection = replay_selection in {
        "confidence",
        "surprise",
        "confidence_surprise"
    }
    # Confidence and surprise selection require a diffusion classifier teacher.
    if scored_replay_selection and not isinstance(
        generative_model, DiffusionClassifier
    ):
        raise ValueError(
            "confidence/surprise replay selection "
            "requires a DiffusionClassifier teacher."
        )

    # Candidate caching is available only for enabled generated replay.
    if replay_cache_mode != "off" and not (
        generative_model is not None and use_generative_replay
    ):
        raise ValueError("Replay candidate caching requires enabled generative replay.")

    # Reject a silent raw fallback when an EMA-teacher ablation was requested.
    needs_snapshot_teacher = use_distillation or scored_replay_selection or (
        mechanistic_metrics and isinstance(generative_model, DiffusionClassifier)
    )
    # EMA treatments must never fall back silently to the raw student branch.
    if needs_snapshot_teacher and snapshot_network_name == "ema" \
    and getattr(generative_model, "ema_network", None) is None:
        raise ValueError("snapshot_network_name='ema' requires EMA to be enabled.")

    # Require an independent student head for continual self-distillation.
    if use_distillation \
    and isinstance(generative_model, DiffusionClassifier) \
    and float(tf.keras.backend.get_value(generative_model.clf_distil_loss_coef)) > 0. \
    and getattr(generative_model.network, "distil_token", None) is None:
        raise ValueError(
            "use_distillation requires the diffusion classifier to have a "
            "distil_token."
        )
    # A distilled run must actually enable at least one teacher loss.
    if use_distillation and not _has_positive_distillation_objective(generative_model):
        raise ValueError("use_distillation requires a positive distillation objective.")
    # Replay-only KD is undefined when the continual treatment has no source
    # of replay rows; fail instead of reporting an identically zero objective.
    if use_distillation \
    and isinstance(generative_model, DiffusionClassifier) \
    and generative_model.clf_distil_scope == "replay_only" \
    and not use_generative_replay:
        raise ValueError(
            "clf_distil_scope='replay_only' requires use_generative_replay=True."
        )
    # Prevent a progressive selector from being silently ignored by VAE,
    # classifier-only, or fixed-buffer continual training.
    if fit_method == "fit_progressively" and not isinstance(
        generative_model, DiffusionModel
    ):
        raise ValueError("fit_progressively requires a diffusion replay model.")

    # Continual diffusion models need a vocabulary that can grow at task boundaries.
    if isinstance(generative_model, DiffusionModel) and not getattr(
        generative_model.network, "dynamic_num_classes", False
    ):
        raise ValueError(
            "Continual diffusion networks must be initialized with "
            "num_classes=None."
        )

    # Ensemble accuracy requires the class predictions of a diffusion classifier.
    if evaluate_ensemble_accuracy and not isinstance(
        generative_model, DiffusionClassifier
    ):
        raise ValueError(
            "evaluate_ensemble_accuracy requires a DiffusionClassifier "
            "or DiffusionClassifierV2 generative_model."
        )

    # Using the generator's classifier requires a classifier-bearing diffusion wrapper.
    if use_generative_model_classifier and not isinstance(
        generative_model, DiffusionClassifier
    ):
        raise ValueError(
            "use_generative_model_classifier requires a "
            "DiffusionClassifier."
        )
    use_diffusion_classifier = use_generative_model_classifier
    # Attached diffusion classifiers consume images instead of saved dense features.
    if use_diffusion_classifier:
        return_features = False

    # VAE replay has conditioning and label-format requirements of its own.
    if isinstance(generative_model, VariationalAutoencoder):
        # An unconditional VAE cannot generate replay for requested old classes.
        if not generative_model.conditioned:
            raise ValueError("A replay VAE must be conditioned.")
        # The conditioned VAE replay path expects one-hot labels.
        if not load_dataset_fn_kwargs["onehot_labels"]:
            raise ValueError("VAE replay requires one-hot labels.")
    # Diffusion replay uses spatial image inputs.
    elif isinstance(generative_model, DiffusionModel):
        # Reject saved feature vectors for the image-space diffusion path.
        if return_features:
            raise ValueError("Diffusion replay requires image data.")

    # V2 determines its own two-phase training protocol.
    train_classifier_separately = isinstance(generative_model, DiffusionClassifierV2)
    # Use the diffusion network's existing classifier head as the initial head.
    if use_diffusion_classifier:
        prev_model = generative_model.network.classifier
    # Standalone classifiers start from the saved tuned architecture.
    else:
        prev_model = get_model(
            1,
            model_type="hp-tuned",
            model_path=tuned_model_path,
            compile_args=compile_args,
            use_loaded_opt=use_loaded_opt,
            verbose=0,
            seed=seed
        )


    def _recovery_trackables() -> dict[str, object]:
        """Collect the current models, optimizers, and frozen teacher for checkpointing.

        The closure reads the current prev_model and generative_model references. V2
        generator and discriminator optimizers are included independently, and the untracked
        teacher is explicitly attached to the checkpoint mapping. Objects are referenced
        directly; this helper neither saves nor clones their state.

        Args:
            None.

        Returns:
            dict[str, object]: Stable role names mapped to TensorFlow trackables. The
            classifier is always represented; absent optimizer, generator, or teacher roles
            are omitted.
        """

        objects: dict[str, object] = {"classifier": prev_model}
        classifier_optimizer = getattr(prev_model, "optimizer", None)
        # Checkpoint the standalone classifier optimizer when one is attached.
        if classifier_optimizer is not None:
            objects["classifier_optimizer"] = classifier_optimizer

        # Include replay-model state only when a generator exists.
        if generative_model is not None:
            objects["replay_model"] = generative_model
            replay_optimizer = getattr(generative_model, "optimizer", None)
            # Checkpoint the replay model's primary optimizer when available.
            if replay_optimizer is not None:
                objects["replay_optimizer"] = replay_optimizer

            generator_optimizer = getattr(generative_model, "gen_optimizer", None)
            classifier_phase_optimizer = getattr(
                generative_model,
                "clf_optimizer",
                None
            )

            # V2's generator optimizer has separate state to checkpoint.
            if generator_optimizer is not None:
                objects["generator_optimizer"] = generator_optimizer

            # V2's classifier optimizer also has separate state to checkpoint.
            if classifier_phase_optimizer is not None:
                objects["classifier_phase_optimizer"] = (classifier_phase_optimizer)

            teacher = getattr(generative_model, "teacher_network", None)
            # Save the frozen next-task teacher outside wrapper tracking.
            if teacher is not None:
                objects["teacher"] = teacher

        return objects


    def _prepare_optimizer_slots() -> None:
        """Materialize optimizer variables needed for exact checkpoint save or restore.

        The closure reads prev_model and generative_model. Diffusion wrappers first register
        their dynamic variables; V2 additionally prepares both phase-specific variable
        groups. Shared optimizer aliases are handled once. Legacy optimizers use
        _create_all_weights, newer ones use build. No gradient update is applied.

        Args:
            None.

        Returns:
            None: Existing optimizer objects acquire missing slot variables for the
            currently constructed model topology.
        """

        optimizer_variables = []
        classifier_optimizer = getattr(prev_model, "optimizer", None)
        # Prepare slots for a standalone classifier optimizer when present.
        if classifier_optimizer is not None:
            optimizer_variables.append(
                (classifier_optimizer, prev_model.trainable_variables)
            )

        # Diffusion wrappers register their dynamically expanded optimizer variables.
        if isinstance(generative_model, DiffusionModel):
            generative_model._register_optimizer_variables()
            # V2 needs slots for both generator and classifier variable groups.
            if isinstance(generative_model, DiffusionClassifierV2):
                optimizer_variables.extend([
                    (
                        generative_model.gen_optimizer,
                        generative_model.gen_trainable_variables or []
                    ),
                    (
                        generative_model.clf_optimizer,
                        generative_model.clf_trainable_variables or [],
                    ),
                ])
        # Other supplied generators use their ordinary optimizer variable list.
        elif generative_model is not None:
            replay_optimizer = getattr(generative_model, "optimizer", None)
            # Prepare generator slots only when an optimizer is attached.
            if replay_optimizer is not None:
                optimizer_variables.append(
                    (replay_optimizer, generative_model.trainable_variables)
                )

        prepared = set()
        for optimizer, variables in optimizer_variables:
            # Skip absent optimizers and aliases already prepared through another role.
            if optimizer is None or id(optimizer) in prepared:
                continue

            prepared.add(id(optimizer))
            variables = list(variables)

            # Legacy Keras optimizers create slots through _create_all_weights.
            if hasattr(optimizer, "_create_all_weights"):
                optimizer._create_all_weights(variables)
            # Newer optimizer APIs create their slots through build.
            elif hasattr(optimizer, "build"):
                optimizer.build(variables)


    # Load the shared arrays once. Their local generator state is checkpointed
    # after every task and restored before the first resumed task.
    dataset_arrays, rng = _load_continual_arrays(
        load_dataset_fn,
        class_order,
        return_features,
        load_dataset_fn_kwargs,
        max_train_samples,
        max_val_samples,
        pad,
        seed
    )

    # Development requires nonempty validation data and cannot substitute locked test rows.
    if experiment_phase == "development" and (
        not use_valset or
        dataset_arrays[2] is None or
        len(dataset_arrays[2]) == 0
    ):
        raise ValueError(
            "development experiments require a nonempty validation split; "
            "the locked test split is never used as a substitute."
        )

    # Remove locked-test contents before fingerprints or task preprocessing can
    # inspect them during development. Empty copies preserve the loader's
    # shapes/dtypes without retaining views into the original test allocation.
    if experiment_phase == "development":
        dataset_arrays = (
            *dataset_arrays[:4],
            np.asarray(dataset_arrays[4])[:0].copy(),
            np.asarray(dataset_arrays[5])[:0].copy()
        )
    # Non-diffusion runs do not need a conversion into diffusion image space.
    if not isinstance(generative_model, DiffusionModel):
        diffusion_data_min, diffusion_data_range = 0., 1.
    # Diffusion runs derive image-space conversion from the loader's preprocessing.
    else:
        initial_x, _ = _select_classes(
            dataset_arrays[0],
            dataset_arrays[1],
            internal_task_groups[0]
        )
        preprocess = load_dataset_fn_kwargs.get("preprocess")

        # Already standardized diffusion inputs span the nominal [-1, 1] range.
        if preprocess in ("standardize", "diffusion"):
            diffusion_data_min, diffusion_data_range = -1., 2.
        # Min-max inputs span the nominal [0, 1] range.
        elif preprocess == "min-max":
            diffusion_data_min, diffusion_data_range = 0., 1.
        # Other loader representations use their observed training minimum and range.
        else:
            diffusion_data_min = float(np.min(initial_x))
            diffusion_data_range = float(np.max(initial_x) - diffusion_data_min) or 1.

    initial_generator_weight_descriptor = _model_weight_descriptor(generative_model)
    dataset_names = ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test")
    loader_descriptor = _recovery_descriptor(load_dataset_fn)
    loader_kwargs_descriptor = _recovery_descriptor(load_dataset_fn_kwargs)
    array_descriptors = {
        name: _array_recovery_descriptor(array)
        for name, array in zip(dataset_names, dataset_arrays)
    }
    generator_topology_descriptor = _model_topology_descriptor(generative_model)
    diffusion_scale_descriptor = {
        "data_min": diffusion_data_min,
        "data_range": diffusion_data_range
    }
    replay_cache_context = fingerprint_state({
        "loader": loader_descriptor,
        "loader_kwargs": loader_kwargs_descriptor,
        "train_inputs": array_descriptors["x_train"],
        "train_labels": array_descriptors["y_train"],
        "generator_topology": generator_topology_descriptor,
        "generator_initial_weights": initial_generator_weight_descriptor,
        "dtype_policy": dtype_policy,
        "diffusion_scale": diffusion_scale_descriptor
    })

    # Seal every behavior-defining input used after a task boundary. Paths and
    # presentation-only switches are excluded, while model artifacts, configs,
    # data bytes, callbacks, schedules, precision, and replay policy are all
    # represented without process-local repr strings.
    run_descriptor = {
        "schema": 4,
        "schedule": {
            "class_num": class_num,
            "class_order": class_order,
            "task_groups": original_task_groups,
            "task_size": task_size,
            "class_order_mode": class_order_mode,
            "task_order_mode": task_order_mode,
        },
        "runtime": {
            "seed": seed,
            "dtype_policy": dtype_policy,
            "deterministic_ops": bool(deterministic_ops),
        },
        "data": {
            "loader": loader_descriptor,
            "loader_kwargs": loader_kwargs_descriptor,
            "return_features": bool(return_features),
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
            "shuffle_buffer": shuffle_buffer,
            "pad": pad,
            "use_valset": bool(use_valset),
            "diffusion_scale": diffusion_scale_descriptor,
            "arrays": array_descriptors,
        },
        "training": {
            "remove_prev_classes": bool(remove_prev_classes),
            "keep_same_model": bool(keep_same_model),
            "use_loaded_opt": bool(use_loaded_opt),
            "batch_size": batch_size,
            "epochs": epochs,
            "fit_method": fit_method,
            "fit_kwargs": _recovery_descriptor(fit_kwargs),
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "compile_args": _recovery_descriptor(compile_args),
            "generative_compile_args": _recovery_descriptor(
                generative_model_compile_args
            ),
            "callbacks": _recovery_descriptor(list(callbacks_list or [])),
            "generative_callbacks": _recovery_descriptor(
                list(generative_callbacks_list or [])
            ),
            "callback_patience": callback_patience,
            "callback_monitor": callback_monitor,
            "callback_monitor_mode": callback_monitor_mode,
        },
        "replay": {
            "baseline": baseline,
            "use_buffer": bool(use_buffer),
            "use_generative_replay": bool(use_generative_replay),
            "buffer_kwargs": _recovery_descriptor(buffer_kwargs),
            "generative_model_kwargs": _recovery_descriptor(generative_model_kwargs),
            "budget_mode": replay_budget_mode,
            "old_examples": replay_old_examples,
            "current_examples": replay_current_examples,
            "candidate_multiplier": replay_candidate_multiplier,
            "selection": replay_selection,
            "surprise_weight": replay_surprise_weight,
            "cache_dir": replay_cache_dir,
            "cache_mode": replay_cache_mode,
            "use_generative_model_classifier": bool(use_generative_model_classifier),
            "train_classifier_separately": bool(train_classifier_separately),
        },
        "distillation_and_metrics": {
            "use_distillation": bool(use_distillation),
            "snapshot_network_name": snapshot_network_name,
            "mechanistic_metrics": bool(mechanistic_metrics),
            "mechanistic_max_samples": mechanistic_max_samples,
            "experiment_phase": experiment_phase,
            "use_ensemble_accuracy": bool(use_ensemble_accuracy),
            "evaluate_ensemble_accuracy": bool(evaluate_ensemble_accuracy),
            "ensemble_accuracy_kwargs": _recovery_descriptor(ensemble_accuracy_kwargs),
            "experiment_manifest_hash": authenticated_manifest_hash,
            "experiment_run_id": experiment_run_id,
        },
        "models": {
            "template_artifact": _artifact_recovery_descriptor(tuned_model_path),
            "classifier": _model_topology_descriptor(prev_model),
            "classifier_initial_weights": _model_weight_descriptor(prev_model),
            "replay": generator_topology_descriptor,
            "replay_initial_weights": initial_generator_weight_descriptor,
            "initial_classifier": _model_topology_descriptor(initial_classifier),
            "initial_classifier_weights": _model_weight_descriptor(initial_classifier),
            "initial_teacher": _model_topology_descriptor(teacher_network),
            "initial_teacher_weights": _model_weight_descriptor(teacher_network),
        },
    }
    run_fingerprint = fingerprint_state(run_descriptor)

    task_state = {
        "accuracies": [],
        "ensemble_accuracies": [],
        "histories": [],
        "generative_histories": [],
        "classifier_evaluations": [],
        "generative_evaluations": [],
        "ordinary_accuracy_matrix": [],
        "validation_accuracy_matrix": [],
        "ensemble_accuracy_matrix": [],
        "validation_ensemble_accuracy_matrix": [],
        "task_seeds": [],
        "task_resource_metrics": [],
        "task_mechanistic_metrics": []
    }
    acc_list = task_state["accuracies"]
    ensemble_acc_list = task_state["ensemble_accuracies"]
    histories = task_state["histories"]
    generative_histories = task_state["generative_histories"]
    classifier_evaluations_list = task_state["classifier_evaluations"]
    generative_evaluations_list = task_state["generative_evaluations"]
    ordinary_accuracy_matrix = task_state["ordinary_accuracy_matrix"]
    validation_accuracy_matrix = task_state["validation_accuracy_matrix"]
    ensemble_accuracy_matrix = task_state["ensemble_accuracy_matrix"]
    validation_ensemble_accuracy_matrix = task_state[
        "validation_ensemble_accuracy_matrix"
    ]
    task_seeds = task_state["task_seeds"]
    task_resource_metrics = task_state["task_resource_metrics"]
    task_mechanistic_metrics = task_state["task_mechanistic_metrics"]
    previous_replay_samples = None
    previous_replay_labels = None
    checkpoint_paths = []
    start_task_index = 0
    generative_callback_base_seeds = [
        getattr(callback, "base_seed", getattr(callback, "seed", None))
        for callback in generative_callbacks_list or ()
    ]

    # Resume only from an immutable completed-task boundary. The saved
    # canonical schedule is compared before any weights are accepted.
    if resume_from is not None:
        recovered = load_task_checkpoint(
            resume_from,
            expected_class_order=class_order,
            expected_task_groups=original_task_groups,
            expected_fingerprint=run_fingerprint,
        )
        saved = recovered.experiment_state
        saved_run_descriptor = saved.get("run_descriptor")
        # Reject missing or changed run descriptors before restoring checkpoint state.
        if saved_run_descriptor is None or fingerprint_state(
            saved_run_descriptor
        ) != run_fingerprint:
            raise ValueError(
                "Checkpoint is incompatible: its complete experiment "
                "descriptor is missing or differs from the run fingerprint."
            )
        start_task_index = recovered.next_task_index
        # Keep a recovery cursor within the authoritative saved schedule.
        if start_task_index > len(internal_task_groups):
            raise ValueError("Checkpoint task cursor exceeds this schedule.")

        completed_groups = internal_task_groups[:start_task_index]
        # Recreate the exact dynamic topology before object restoration. Class
        # and persistent depth growth are replayed task by task so optimizer
        # slot dependencies follow the uninterrupted structural sequence.
        if isinstance(generative_model, DiffusionModel):
            # Restore stage depth specifications only for progressive training.
            depth_specs = _progressive_depth_specs(fit_kwargs) \
                if fit_method == "fit_progressively" else []
            for completed_group in completed_groups:
                generative_model._check_new_labels(
                    y=np.asarray(completed_group),
                    verbose=False,
                )
                for depth_spec in depth_specs:
                    try:
                        generative_model._add_depths(deepcopy(depth_spec))
                    except Exception as error:
                        raise ValueError(
                            "Checkpoint topology is incompatible: persistent "
                            "progressive depth growth could not be reconstructed "
                            f"after {start_task_index} completed task(s)."
                        ) from error
            # After diffusion topology recovery, reconnect the attached classifier head.
            if use_diffusion_classifier:
                prev_model = generative_model.network.classifier
        # Standalone classifier recovery rebuilds the saved classifier topology.
        else:
            completed_width = 0
            for completed_index, completed_group in enumerate(completed_groups):
                completed_width += len(completed_group)
                prev_model = get_model(
                    completed_width,
                    model_type="hp-tuned",
                    model_path=tuned_model_path,
                    compile_args=compile_args,
                    use_loaded_opt=use_loaded_opt,
                    verbose=0,
                    seed=derive_seed(seed, "task", completed_index)
                )
                # A caller-supplied optimizer instance is intentionally shared
                # across task heads within one run. Recreate each historical
                # slot group so its saved object graph can be consumed exactly.
                _prepare_optimizer_slots()

        # Persisted teachers represent the completed student and therefore
        # have the same topology as the restored model.
        # Recreate its object graph before loading the saved teacher weights.
        if use_distillation and start_task_index > 0:
            restored_teacher = generative_model.snapshot_teacher_network(
                network_name=snapshot_network_name
            )
            generative_model.set_teacher_network(restored_teacher)

        _prepare_optimizer_slots()
        reconstructed_trackables = _recovery_trackables()
        reconstructed_topology = _trackable_topology_descriptor(
            reconstructed_trackables
        )
        expected_topology_fingerprint = saved.get("trackable_topology_fingerprint")
        # Reject any missing model, EMA, teacher, or optimizer dependency.
        if expected_topology_fingerprint is None or fingerprint_state(
            reconstructed_topology
        ) != expected_topology_fingerprint:
            raise ValueError(
                "Checkpoint topology is incompatible with the reconstructed "
                "models, EMA branch, teacher, or optimizer slots."
            )

        recovered = load_task_checkpoint(
            resume_from,
            trackables=reconstructed_trackables,
            expected_class_order=class_order,
            expected_task_groups=original_task_groups,
            expected_fingerprint=run_fingerprint,
            assert_consumed=True
        )
        restore_rng_state(recovered.rng_state, numpy_generator=rng)
        # Restore saved buffer contents and RNG state only for buffered replay.
        if use_buffer and recovered.replay_state is not None:
            restore_replay_buffer(buffer, recovered.replay_state)

        for name, values in task_state.items():
            values.extend(saved.get(name, []))

        previous_replay_samples = saved.get("previous_replay_samples")
        previous_replay_labels = saved.get("previous_replay_labels")
        checkpoint_paths.append(str(recovered.task_dir))

    # Resumed runs without a new checkpoint root continue beside the recovered task.
    if checkpoint_dir is None and resume_from is not None:
        checkpoint_dir = str(recovered.task_dir.parent)

    for task_index in range(start_task_index, len(internal_task_groups)):
        task_wall_start = time.perf_counter()
        # Record buffered, generated, or absent replay according to the active source.
        task_resource = {
            "task_index": task_index,
            "task_classes": list(original_task_groups[task_index]),
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "replay": {
                "source": "buffer" if use_buffer else (
                    "generated" if generative_model is not None \
                    and use_generative_replay else "none"
                ),
                "selection": replay_selection,
                "candidate_count": 0,
                "selected_count": 0,
                "storage_offered_count": 0,
                "cache_path": None
            },
            "seconds": {
                "generator_sampling": 0.,
                "teacher_scoring": 0.,
                "classifier_fit": 0.,
                "generator_fit": 0.,
            }
        }
        task_mechanistic = {}
        new_classes = internal_task_groups[task_index]
        task_seed = derive_seed(seed, "task", task_index)
        task_seeds.append(task_seed)


        def _task_dataset(
            x: np.ndarray | None,
            y: np.ndarray | None,
            training: bool = False,
            metadata: np.ndarray | None = None
        ) -> tf.data.Dataset | None:
            """Create a phase dataset using the current task seed and batching settings.

            The closure captures batch_size, shuffle_buffer, and task_seed. A missing
            shuffle_buffer uses the training pool length. Metadata travels with its
            corresponding examples through shuffling and batching.

            Args:
                x (np.ndarray | None): Example array; None returns no dataset for an absent
                    split.
                y (np.ndarray | None): Aligned labels accepted by get_dataset, or None for
                    an unlabeled input.
                training (bool): True enables the configured shuffle buffer and task seed;
                    False uses an unshuffled evaluation stream. Defaults to ``False``.
                metadata (np.ndarray | None): Aligned per-row metadata such as the replay
                    provenance mask; None creates ordinary dataset elements. Defaults to
                    ``None``.

            Returns:
                tf.data.Dataset | None: Batched inputs, (inputs, labels), or (inputs,
                labels, metadata), with partial batches retained. NHW inputs gain a
                singleton channel axis. Returns None when x is absent.

            Raises:
                ValueError: Metadata is provided without labels for a present input array.
            """

            # An absent optional split produces no task dataset.
            if x is None:
                return None
            # Training shuffles with the explicit buffer or full row count and task seed;
            # evaluation stays ordered.
            return get_dataset(
                x, y,
                shuffle_buffer=(
                    len(x) if shuffle_buffer is None else shuffle_buffer
                ) if training else 0,
                batch_size=batch_size,
                drop_remainder=False,
                seed=task_seed if training else None,
                metadata=metadata
            )


        for callback_index, callback in enumerate(generative_callbacks_list or ()):
            prefix_setter = getattr(callback, "set_artifact_prefix", None)
            # Callbacks exposing artifact prefixes receive a task-and-class-specific prefix.
            if callable(prefix_setter):
                class_text = "-".join(
                    str(label)
                    for label in original_task_groups[task_index]
                )
                prefix_setter(f"task-{task_index + 1}_classes-{class_text}")

            # Give sampling/report callbacks a task-isolated reproducible stream.
            if hasattr(callback, "seed"):
                callback.seed = _derive_generative_callback_seed(
                    task_seed,
                    generative_callback_base_seeds[callback_index],
                    task_index,
                    callback_index,
                    callback
                )

        # A restarted incomplete task receives the same process-wide streams,
        # independent of how much randomness the preceding process consumed.
        configure_runtime(task_seed, dtype_policy, False)
        # Reset cached Keras random-op counters as well as wrapper/layer seeds;
        # TensorFlow intentionally excludes these private counters from normal
        # checkpoints, so every task must start from its own derived stream.
        _reset_task_random_streams(generative_model, task_seed)
        seen_classes = [
            label
            for group in internal_task_groups[:task_index + 1]
            for label in group
        ]
        seen_class_num = len(seen_classes)
        classification_objective_defined = seen_class_num > 1
        task_resource["classification_objective_defined"] = classification_objective_defined

        # Verbose runs print the classes introduced through the current task.
        if verbose:
            print(
                75*'-' + " Classes:",
                [
                    label
                    for group in original_task_groups[:task_index + 1]
                    for label in group
                ]
            )

        previous_teacher = None
        # Tasks after the first can use a snapshot of the previous completed student.
        if needs_snapshot_teacher and task_index > 0:
            # Distillation reuses its frozen teacher; other teacher-backed diagnostics take a
            # snapshot now.
            previous_teacher = generative_model.teacher_network if use_distillation else (
                generative_model.snapshot_teacher_network(
                    network_name=snapshot_network_name
                )
            )

        # Expand diffusion vocabularies from the task schedule before subsampling training
        # rows.
        if isinstance(generative_model, DiffusionModel):
            # The schedule owns vocabulary order. A sampled generator pool may
            # omit a new class; discovering it later would reorder class logits.
            generative_model._check_new_labels(y=np.asarray(new_classes), verbose=verbose)

        # Attached diffusion classifiers reuse the wrapper's newly expanded head.
        if use_diffusion_classifier:
            new_model = generative_model.network.classifier
        # Standalone classifiers rebuild a head sized for all classes seen so far.
        else:
            new_model = get_model(
                seen_class_num, model_type="hp-tuned",
                model_path=tuned_model_path,
                compile_args=compile_args,
                use_loaded_opt=use_loaded_opt,
                verbose=0,
                seed=task_seed
            )
            _reset_task_random_streams(new_model, task_seed)

            # Use the supplied initial classifier for the first task or independently
            # restarted heads.
            if initial_classifier is not None and (
                task_index == 0 or not keep_same_model
            ):
                copy_model(initial_classifier, new_model, allow_truncate=True)
            # Continuing heads inherit the preceding task's learned classifier weights.
            elif keep_same_model:
                copy_model(prev_model, new_model)

        optimizer_iterations_before = _optimizer_iteration_metrics(
            new_model,
            generative_model
        )

        (all_x_train, all_y_train, all_x_val,
        all_y_val, all_x_test, all_y_test) = dataset_arrays
        # New-only training after the first task excludes original rows from earlier classes.
        if remove_prev_classes and task_index > 0:
            train_classes = new_classes
        # The first task and cumulative training use all classes introduced so far.
        else:
            train_classes = seen_classes

        x_train, y_train = _select_classes(all_x_train, all_y_train, train_classes)
        task_resource["current_examples_available"] = int(len(x_train))

        # Fixed-total designs match current-data exposure across treatments.
        if replay_budget_mode == "fixed_total":
            x_train, y_train = _sample_exact_rows(
                x_train,
                y_train,
                replay_current_examples,
                np.random.default_rng(derive_seed(task_seed, "current_exposure"))
            )

        task_resource["current_examples_exposed"] = int(len(x_train))
        x_test, y_test = _select_classes(all_x_test, all_y_test, seen_classes)

        # Select seen-class validation rows when both validation arrays exist.
        if all_x_val is not None and all_y_val is not None:
            x_val, y_val = _select_classes(all_x_val, all_y_val, seen_classes)
        # Without a complete validation split, keep both validation arrays absent.
        else:
            x_val, y_val = None, None

        # An explicitly disabled validation set is not passed to training or evaluation.
        if not use_valset:
            x_val, y_val = None, None

        # Image-based VAEs flatten pixels to match their dense input architecture.
        if isinstance(generative_model, VariationalAutoencoder) \
        and not return_features: # Match configured dense VAE input shapes.
            x_train = _flatten_example_rows(x_train)
            x_test = _flatten_example_rows(x_test)
            # Flatten validation images only when a validation split exists.
            if x_val is not None:
                x_val = _flatten_example_rows(x_val)

        # Track the origin of each training row. Diffusion KD consumes this
        # metadata only for the optional replay-only scope.
        replay_mask = np.zeros((len(x_train),), dtype=bool)

        old_classes = [
            label
            for group in internal_task_groups[:task_index]
            for label in group
        ]
        old_original_classes = [
            label
            for group in original_task_groups[:task_index]
            for label in group
        ]
        selected_replay_x = x_train[:0]
        selected_replay_y = y_train[:0]
        candidate_probabilities = None

        # Buffered replay samples stored examples from previously learned classes.
        if use_buffer:
            current_x_train, current_y_train = x_train, y_train
            # The first task has no replay; later tasks use the fixed-total or legacy buffer
            # count.
            target_replay_count = int(
                replay_old_examples
                if replay_budget_mode == "fixed_total"
                else buffer_kwargs["sample_num"]
            ) if task_index > 0 else 0
            candidate_count = target_replay_count * replay_candidate_multiplier
            x_buffer, y_buffer = buffer.sample_buffer_and_prepare_dataset(
                candidate_count
            )
            # Fixed-total exposure is exact even before a small reservoir fills;
            # repeated rows are explicit optimizer exposures, not new storage.
            if replay_budget_mode == "fixed_total" and candidate_count:
                x_buffer, y_buffer = _sample_exact_rows(
                    x_buffer,
                    y_buffer,
                    candidate_count,
                    np.random.default_rng(derive_seed(task_seed, "buffer_candidates")),
                )
            candidate_ids = _label_ids(y_buffer)
            selected_replay_x, selected_ids, gate_diagnostics = (
                select_replay_candidates(
                    x_buffer,
                    candidate_ids,
                    target_replay_count,
                    strategy=replay_selection,
                    seed=derive_seed(task_seed, "replay_selection"),
                    surprise_weight=replay_surprise_weight,
                )
            )
            selected_replay_y = _restore_replay_label_shape(selected_ids, y_train)
            task_resource["replay"].update(gate_diagnostics)
        # Generated replay begins after the first task when a generator and replay are
        # enabled.
        elif use_generative_replay and \
        generative_model is not None \
        and task_index > 0:
            # Use the fixed old-example total or multiply legacy samples per class by old-
            # class count.
            target_replay_count = int(
                replay_old_examples
                if replay_budget_mode == "fixed_total"
                else generative_model_kwargs["samples_per_class"]
                * len(old_classes)
            )
            candidate_count = target_replay_count * replay_candidate_multiplier
            legacy_candidate_path = (
                replay_budget_mode == "legacy"
                and replay_candidate_multiplier == 1
                and replay_selection == "all"
                and replay_cache_mode == "off"
            )
            # Preserve legacy candidate randomness; other candidate pools use a derived replay
            # seed.
            candidate_seed = task_seed if legacy_candidate_path else derive_seed(
                task_seed, "replay_candidates"
            )
            # Legacy unexpanded pools repeat each old label equally; other pools balance the
            # requested total.
            expected_candidate_ids = (
                np.repeat(old_classes, generative_model_kwargs["samples_per_class"])
                if replay_budget_mode == "legacy"
                and replay_candidate_multiplier == 1
                else _balanced_generation_labels(
                    old_classes,
                    candidate_count,
                    np.random.default_rng(candidate_seed),
                )
            )
            expected_candidate_y = _restore_replay_label_shape(
                expected_candidate_ids, y_train
            )
            # Create a candidate-cache path only when caching is enabled.
            cache_file = _replay_cache_path(
                replay_cache_dir,
                task_index,
                old_original_classes,
                candidate_count,
                context_fingerprint=replay_cache_context,
            ) if replay_cache_mode != "off" else None
            read_cached_pool = replay_cache_mode == "read" or (
                replay_cache_mode == "read_write"
                and cache_file is not None and cache_file.is_file()
            )
            sample_started = time.perf_counter()

            # Read mode, or an existing automatic cache, supplies the candidate pool from
            # disk.
            if read_cached_pool:
                # The read branch ignores candidate x bytes; only exact count,
                # stream seed, class set, and archive checksums authenticate it.
                x_buffer, y_buffer, cache_path = _cached_replay_candidates(
                    np.empty((candidate_count, 0), dtype=x_train.dtype),
                    expected_candidate_y,
                    replay_cache_dir,
                    replay_cache_mode,
                    task_index,
                    old_original_classes,
                    candidate_seed,
                    replay_cache_context,
                )
            # Without a readable candidate cache, generate a fresh replay pool.
            else:
                # VAE generation exposes a per-class API; reduce only the
                # opt-in non-divisible fixed-total case to an exact pool.
                # Preserve a correctly shaped pool for an explicit zero budget.
                if candidate_count == 0:
                    x_buffer = x_train[:0]
                    y_buffer = expected_candidate_y
                # VAE replay generates conditioned samples through the VAE API.
                elif isinstance(generative_model, VariationalAutoencoder):
                    per_class = int(np.ceil(candidate_count / len(old_classes)))
                    x_buffer, y_buffer = generative_model.generate(
                        classes=old_classes,
                        samples_per_class=per_class,
                        onehot_y_output=load_dataset_fn_kwargs["onehot_labels"],
                        seed=candidate_seed,
                    )
                    # Reduce a non-divisible per-class draw to the exact pool size.
                    if len(x_buffer) != candidate_count:
                        generated_ids = _label_ids(y_buffer)
                        x_buffer, generated_ids, _ = select_replay_candidates(
                            x_buffer,
                            generated_ids,
                            candidate_count,
                            strategy="uniform",
                            seed=derive_seed(candidate_seed, "vae_exact_pool"),
                        )
                        y_buffer = _restore_replay_label_shape(generated_ids, y_train)
                # Diffusion sampling accepts the exact candidate label vector.
                # Use direct label-conditioned diffusion for every non-VAE model.
                else:
                    y_buffer_ids = expected_candidate_ids
                    x_buffer = _sample_diffusion_replay(
                        generative_model,
                        y_buffer_ids,
                        batch_size,
                        candidate_seed,
                        x_train[:0],
                    )
                    # Restore images to the shared loader preprocessing space.
                    # Convert generated diffusion values only when rows exist.
                    if len(x_buffer) and not return_features:
                        x_buffer = (
                            x_buffer * diffusion_data_range + diffusion_data_min
                        ).astype(x_train.dtype)
                    y_buffer = _restore_replay_label_shape(y_buffer_ids, y_train)
                x_buffer, y_buffer, cache_path = _cached_replay_candidates(
                    x_buffer,
                    y_buffer,
                    replay_cache_dir,
                    replay_cache_mode,
                    task_index,
                    old_original_classes,
                    candidate_seed,
                    replay_cache_context,
                )
            # Remove a generated singleton channel when current grayscale inputs have no
            # channel axis.
            if (x_train.ndim == 3 and x_buffer.ndim == 4 and x_buffer.shape[-1] == 1):
                x_buffer = x_buffer[..., 0]
            task_resource["seconds"]["generator_sampling"] = float(
                time.perf_counter() - sample_started
            )
            task_resource["replay"]["cache_path"] = cache_path
            candidate_ids = _label_ids(y_buffer)

            # Confidence and surprise selectors score candidates with the previous teacher.
            if scored_replay_selection:
                scoring_started = time.perf_counter()
                candidate_probabilities = _predict_teacher_probabilities(
                    previous_teacher,
                    x_buffer,
                    diffusion_data_min,
                    diffusion_data_range,
                    batch_size,
                )
                task_resource["seconds"]["teacher_scoring"] += float(
                    time.perf_counter() - scoring_started
                )
            selected_replay_x, selected_ids, gate_diagnostics = (
                select_replay_candidates(
                    x_buffer,
                    candidate_ids,
                    target_replay_count,
                    strategy=replay_selection,
                    probabilities=candidate_probabilities,
                    seed=derive_seed(task_seed, "replay_selection"),
                    surprise_weight=replay_surprise_weight,
                )
            )
            selected_replay_y = _restore_replay_label_shape(selected_ids, y_train)
            task_resource["replay"].update(gate_diagnostics)

        # Append replay rows and their origin mask only when selection retained examples.
        if len(selected_replay_x):
            x_train = np.concatenate([x_train, selected_replay_x], axis=0)
            y_train = np.concatenate([y_train, selected_replay_y], axis=0)
            replay_mask = np.concatenate([
                replay_mask,
                np.ones((len(selected_replay_x),), dtype=bool),
            ])

            # Expensive replay diagnostics remain fully opt-in.
            if mechanistic_metrics:
                selected_probabilities = None
                # Add teacher consistency/calibration only when a trace exists.
                if previous_teacher is not None:
                    scoring_started = time.perf_counter()
                    selected_probabilities = _predict_teacher_probabilities(
                        previous_teacher,
                        selected_replay_x,
                        diffusion_data_min,
                        diffusion_data_range,
                        batch_size,
                    )
                    task_resource["seconds"]["teacher_scoring"] += float(
                        time.perf_counter() - scoring_started
                    )
                selected_ids = _label_ids(selected_replay_y)
                task_mechanistic = replay_quality_metrics(
                    selected_replay_x,
                    selected_ids,
                    old_classes,
                    probabilities=selected_probabilities,
                    previous_samples=previous_replay_samples,
                    previous_labels=previous_replay_labels,
                    max_diversity_samples=mechanistic_max_samples,
                    seed=derive_seed(task_seed, "mechanistic_metrics"),
                )
                retained_count = min(len(selected_replay_x), mechanistic_max_samples)
                previous_replay_samples, previous_replay_labels = (
                    _sample_exact_rows(
                        selected_replay_x,
                        selected_ids,
                        retained_count,
                        np.random.default_rng(derive_seed(
                            task_seed, "mechanistic_replay_reference"
                        )),
                    )
                )

        task_resource["training_examples_total"] = int(len(x_train))

        classifier_x_train = x_train
        classifier_x_val = x_val
        classifier_x_test = x_test
        classifier_input_shape = getattr(new_model, "input_shape", None)

        # Flatten image rows when a standalone classifier expects a vector input.
        if not use_diffusion_classifier and isinstance(
            classifier_input_shape, tuple
        ) and len(classifier_input_shape) == 2:
            classifier_x_train = _flatten_example_rows(x_train)
            classifier_x_test = _flatten_example_rows(x_test)
            # Flatten validation rows only when validation inputs exist.
            classifier_x_val = _flatten_example_rows(x_val) \
                if x_val is not None else None

        classifier_y_train = y_train
        classifier_y_val = y_val
        classifier_y_test = y_test
        # Adapt one-hot loader labels to the classifier's configured loss and current width.
        if load_dataset_fn_kwargs["onehot_labels"]:
            loss = getattr(
                new_model,
                "loss",
                compile_args.get("loss", "sparse_categorical_crossentropy")
            )
            loss_name = getattr(
                loss,
                "name",
                getattr(loss, "__name__", str(loss))
            ).lower()

            # Sparse classifier losses need integer target IDs instead of one-hot rows.
            if "sparse" in loss_name:
                classifier_y_train = np.argmax(y_train, axis=-1)
                # Decode validation targets only when their optional label array exists.
                classifier_y_val = np.argmax(y_val, axis=-1) if y_val is not None else None
                classifier_y_test = np.argmax(y_test, axis=-1)
            # Other standalone classifiers retain one-hot targets clipped to the seen-class
            # width.
            elif not use_diffusion_classifier:
                classifier_y_train = y_train[..., :seen_class_num]
                # Clip validation one-hot targets only when validation labels exist.
                classifier_y_val = y_val[..., :seen_class_num] \
                    if y_val is not None else None
                classifier_y_test = y_test[..., :seen_class_num]

        # An omitted shuffle buffer spans the task rows; an explicit size is preserved.
        task_shuffle_buffer = len(x_train) if shuffle_buffer is None \
            else shuffle_buffer
        trainset = _task_dataset(classifier_x_train, classifier_y_train, training=True)
        valset = _task_dataset(classifier_x_val, classifier_y_val)
        # Only legacy and confirmation runs construct a locked-test classifier dataset.
        testset = _task_dataset(classifier_x_test, classifier_y_test) \
            if experiment_phase != "development" else None

        history = {}
        # Attach standalone-classifier callbacks only when that fit phase runs.
        if not use_diffusion_classifier:
            # Monitor validation accuracy when available; otherwise monitor training accuracy.
            task_callbacks = _phase_callbacks(
                "val_accuracy" if valset is not None else "accuracy",
                legacy_patience=5,
            )

            fit_started = time.perf_counter()
            history = _train_task_model(
                new_model,
                trainset,
                valset,
                task_callbacks=task_callbacks
            )
            task_resource["seconds"]["classifier_fit"] += float(
                time.perf_counter() - fit_started
            )

        prev_model = new_model

        # Retain completed-task examples for future fixed-buffer replay.
        if use_buffer:
            # The named ER control is Algorithm R over the complete exposed
            # current stream. Independent reservoir ablations retain the
            # historical sampled-insertion ``insert_num`` behavior.
            if baseline == "reservoir_er":
                buffer.extend(zip(current_x_train, current_y_train))
                task_resource["replay"]["storage_offered_count"] = int(
                    len(current_x_train)
                )
            # Other buffer treatments retain the configured sampled insertion policy.
            else:
                buffer.sample_dataset_and_extend_buffer(
                    (current_x_train, current_y_train),
                    buffer_kwargs["insert_num"]
                )
                task_resource["replay"]["storage_offered_count"] = int(
                    min(len(current_x_train), buffer_kwargs["insert_num"])
                )

        generative_trainset = None
        generative_valset = None
        generative_testset = None
        # Fixed-total replay preserves all exposed rows; legacy mode uses the generator row
        # budget.
        phase_train_num = -1 if replay_budget_mode == "fixed_total" \
                        else generative_model_kwargs["train_num"]
        # VAE generators train through their custom array-based training API.
        if isinstance(generative_model, VariationalAutoencoder):
            vae_fit_kwargs = {
                "y": y_train,
                "train_num": phase_train_num,
                "batch_size": batch_size,
                "shuffle_buffer": task_shuffle_buffer,
                "seed": task_seed,
                "clf": new_model
            }

            fit_started = time.perf_counter()
            # Pass and monitor validation loss when validation exists; otherwise train without
            # validation.
            generative_history = _train_task_model(
                generative_model,
                x_train,
                (x_val, y_val) if x_val is not None else None,
                task_callbacks=_phase_callbacks(
                    "val_loss" if x_val is not None else "loss",
                    default_mode="min",
                    include_generative=True
                ),
                fit_method="train",
                fit_kwargs=vae_fit_kwargs
            )
            task_resource["seconds"]["generator_fit"] += float(
                time.perf_counter() - fit_started
            )
            generative_trainset = _task_dataset(x_train, y_train, training=True)
            generative_valset = _task_dataset(x_val, y_val)
            # Build a VAE test dataset only outside development experiments.
            generative_testset = _task_dataset(x_test, y_test) \
                                if experiment_phase != "development" else None
        # Diffusion generators prepare noised training through dataset-based wrapper APIs.
        elif isinstance(generative_model, DiffusionModel):
            generative_x = _prepare_diffusion_x(
                x_train,
                diffusion_data_min,
                diffusion_data_range
            )
            generative_y = _label_ids(y_train)

            diffusion_classifier_x = generative_x
            diffusion_classifier_y = generative_y
            diffusion_classifier_replay_mask = replay_mask
            replay_only_distillation = isinstance(
                generative_model, DiffusionClassifier
            ) and generative_model.clf_distil_scope == "replay_only"
            generative_replay_mask = replay_mask

            # Fixed-total experiments must not resample away their exact
            # current/replay exposure ratio; the combined set defines updates.
            train_num = phase_train_num
            # Resample diffusion training rows to the configured exact count.
            if train_num != -1:
                indices = rng.integers(0, len(generative_x), (train_num,))
                generative_x = generative_x[indices]
                generative_y = generative_y[indices]
                generative_replay_mask = generative_replay_mask[indices]

            # Attach replay-origin metadata only when distillation is restricted to replay
            # rows.
            generative_trainset = _task_dataset(
                generative_x,
                generative_y,
                training=True,
                metadata=generative_replay_mask if replay_only_distillation else None
            )
            generative_y_val = _label_ids(y_val)
            # Build validation data only when present; replay-only KD marks all validation
            # rows as non-replay.
            generative_valset = _task_dataset(
                _prepare_diffusion_x(x_val, diffusion_data_min, diffusion_data_range),
                generative_y_val,
                metadata=np.zeros((len(x_val),), dtype=bool) if replay_only_distillation else None
            ) if x_val is not None else None

            # Locked-test diffusion preprocessing is confirmation-only.
            if experiment_phase != "development":
                generative_y_test = _label_ids(y_test)
                # Replay-only KD marks locked-test rows as non-replay; other scopes need no
                # mask metadata.
                generative_testset = _task_dataset(
                    _prepare_diffusion_x(
                        x_test,
                        diffusion_data_min,
                        diffusion_data_range
                    ),
                    generative_y_test,
                    metadata=np.zeros((len(x_test),), dtype=bool) if replay_only_distillation else None
                )

            # V2 selects a generator-specific fitting method before its classifier phase.
            if isinstance(generative_model, DiffusionClassifierV2):
                # Use V2's progressive generator method for curricula and its ordinary
                # generator method otherwise.
                generative_fit_method = "fit_generator_progressively" \
                    if fit_method == "fit_progressively" \
                    else "fit_generator"
            # Other diffusion wrappers use the configured ordinary or progressive fit method
            # directly.
            else:
                generative_fit_method = fit_method

            fit_started = time.perf_counter()
            # Monitor validation loss when available and training loss otherwise.
            generative_history = _train_task_model(
                generative_model,
                generative_trainset,
                generative_valset,
                task_callbacks=_phase_callbacks(
                    "val_loss" if generative_valset is not None else "loss",
                    default_mode="min",
                    include_generative=True,
                ),
                fit_method=generative_fit_method,
                fit_kwargs=dict(fit_kwargs or {})
            )
            task_resource["seconds"]["generator_fit"] += float(
                time.perf_counter() - fit_started
            )

            # V2's attached classifier trains in a separate discriminator phase.
            if use_diffusion_classifier and isinstance(
                generative_model, DiffusionClassifierV2
            ): # Train V2 classifier variables in their required separate phase.
                # Monitor total accuracy when active, otherwise primary-head accuracy; prefix
                # validation metrics when available.
                task_callbacks = _phase_callbacks(
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

                # Classifier replay-only KD receives row-origin metadata; other scopes omit
                # it.
                diffusion_classifier_trainset = _task_dataset(
                    diffusion_classifier_x,
                    diffusion_classifier_y,
                    training=True,
                    metadata=(
                        diffusion_classifier_replay_mask
                        if replay_only_distillation else None
                    ),
                )
                fit_started = time.perf_counter()
                history = _train_task_model(
                    generative_model,
                    diffusion_classifier_trainset,
                    generative_valset,
                    task_callbacks=task_callbacks,
                    fit_method="fit_discriminator"
                )
                task_resource["seconds"]["classifier_fit"] += float(
                    time.perf_counter() - fit_started
                )
        # Runs without a generator have no generative training history.
        else:
            generative_history = None

        # Compare the frozen slow/previous trace with the updated student on
        # a bounded old-class training probe. This is optional and never reads
        # the locked test split during development.
        if mechanistic_metrics and previous_teacher is not None and old_classes:
            probe_x, probe_y = _select_classes(all_x_train, all_y_train, old_classes)
            probe_x, probe_y = _sample_exact_rows(
                probe_x,
                probe_y,
                min(len(probe_x), mechanistic_max_samples),
                np.random.default_rng(derive_seed(task_seed, "representation_probe"))
            )
            probe_ids = _label_ids(probe_y)
            scoring_started = time.perf_counter()
            teacher_probe_probabilities = _predict_teacher_probabilities(
                previous_teacher,
                probe_x,
                diffusion_data_min,
                diffusion_data_range,
                batch_size
            )
            student_probe_probabilities = _predict_teacher_probabilities(
                generative_model.network,
                probe_x,
                diffusion_data_min,
                diffusion_data_range,
                batch_size
            )
            task_resource["seconds"]["teacher_scoring"] += float(
                time.perf_counter() - scoring_started
            )
            old_width = teacher_probe_probabilities.shape[1]
            student_old_probabilities = student_probe_probabilities[:, :old_width]
            # Compute CKA with at least two probe rows; smaller probes leave it unavailable.
            task_mechanistic["representation"] = {
                "old_class_probability_cka": linear_cka(
                    teacher_probe_probabilities,
                    student_old_probabilities
                ) if len(probe_x) >= 2 else float("nan"),
                "old_class_probability_centroid_drift": class_centroid_drift(
                    teacher_probe_probabilities,
                    probe_ids,
                    student_old_probabilities,
                    probe_ids
                )
            }

            # Calibration requires held-out labels; never estimate it from the
            # training probe used for prediction-space diagnostics.
            if use_valset and all_x_val is not None and all_y_val is not None:
                calibration_x, calibration_y = _select_classes(
                    all_x_val,
                    all_y_val,
                    old_classes
                )
                calibration_x, calibration_y = _sample_exact_rows(
                    calibration_x,
                    calibration_y,
                    min(len(calibration_x), mechanistic_max_samples),
                    np.random.default_rng(derive_seed(task_seed, "calibration_probe"))
                )

                # Estimate teacher calibration only when old-class validation examples remain.
                if len(calibration_x):
                    calibration_ids = _label_ids(calibration_y)
                    scoring_started = time.perf_counter()
                    calibration_probabilities = (
                        _predict_teacher_probabilities(
                            previous_teacher,
                            calibration_x,
                            diffusion_data_min,
                            diffusion_data_range,
                            batch_size,
                        )
                    )
                    task_resource["seconds"]["teacher_scoring"] += float(
                        time.perf_counter() - scoring_started
                    )
                    task_mechanistic["teacher_heldout_probe"] = (
                        calibration_metrics(calibration_probabilities, calibration_ids)
                    )

        generative_evaluations = {}
        # Development reports the validation split; other phases report the locked test split.
        evaluation_split_name = (
            "valset" if experiment_phase == "development" else "testset"
        )
        # Route the generator's validation dataset in development and test dataset otherwise.
        generative_evaluation_set = (
            generative_valset
            if experiment_phase == "development"
            else generative_testset
        )
        # Report a generator phase only when both its history and evaluation dataset exist.
        if generative_history is not None \
        and generative_evaluation_set is not None:
            generative_evaluations = _report_task_model(
                generative_history,
                generative_model,
                generative_trainset,
                generative_evaluation_set,
                split_name=evaluation_split_name,
            )

        classifier_evaluations = {}
        classifier_report_history = history or generative_history
        # Standalone classifiers report their own phase when a training history is available.
        if not use_diffusion_classifier and classifier_report_history:
            # Use validation data for development reporting and locked-test data otherwise.
            classifier_evaluations = _report_task_model(
                classifier_report_history,
                new_model,
                trainset,
                valset if experiment_phase == "development" else testset,
                split_name=evaluation_split_name,
            )

        # Both splits follow the same scoring path. Development never adds
        # the locked test split. Task metrics are macro means of these rows;
        # ordinary cumulative accuracy remains weighted by example count.
        splits = []
        # Only legacy and confirmation phases include locked-test accuracy scoring.
        if experiment_phase != "development":
            splits.append((
                "test", x_test, y_test, classifier_x_test,
                ordinary_accuracy_matrix, ensemble_accuracy_matrix,
            ))
        # Include validation accuracy scoring when the validation split exists.
        if x_val is not None:
            splits.append((
                "validation", x_val, y_val, classifier_x_val,
                validation_accuracy_matrix, validation_ensemble_accuracy_matrix,
            ))

        acc, ensemble_acc = float("nan"), None
        # Development selects validation accuracy as its trajectory; other phases select test
        # accuracy.
        selected_split = "validation" if experiment_phase == "development" else "test"
        learned_groups = internal_task_groups[:task_index + 1]
        for split, inputs, labels, classifier_inputs, matrix, ensemble_matrix in splits:
            row = [np.nan] * len(internal_task_groups)
            ordinary_acc = float("nan")
            # A one-class softmax has trivial accuracy and zero CE gradient.
            # Missing split rows likewise carry no accuracy observation.
            if classification_objective_defined and len(inputs):
                label_ids = _label_ids(labels)
                # Diffusion classifiers use wrapper-aware prediction; standalone classifiers
                # use Keras predict.
                predictions = _predict_diffusion_classes(
                    generative_model, inputs, label_ids,
                    diffusion_data_min, diffusion_data_range, batch_size,
                ) if use_diffusion_classifier else new_model.predict(
                    classifier_inputs, verbose=verbose,
                )
                correct = np.argmax(predictions, axis=-1) == label_ids
                ordinary_acc = float(np.mean(correct))
                for group_index, group in enumerate(learned_groups):
                    selected = np.isin(label_ids, group)
                    # Record a task-group mean only when that group has evaluation examples.
                    if np.any(selected):
                        row[group_index] = float(np.mean(correct[selected]))
            matrix.append(row)

            split_ensemble_acc = None
            # Ensemble-enabled runs also build a task-group ensemble accuracy row.
            if evaluate_ensemble_accuracy:
                # Evaluate ensembles only for a nontrivial class vocabulary; singleton rows
                # remain unavailable.
                ensemble_row = _ensemble_accuracy_row(
                    generative_model, inputs, labels, learned_groups,
                    len(internal_task_groups), diffusion_data_min,
                    diffusion_data_range, batch_size, ensemble_accuracy_kwargs,
                    derive_seed(seed, "ensemble", task_index, split), verbose,
                ) if classification_objective_defined else [np.nan] * len(internal_task_groups)
                ensemble_matrix.append(ensemble_row)
                split_ensemble_acc = _observed_mean(ensemble_row[:task_index + 1])

            # Only the phase-selected split supplies the public task accuracy trajectory.
            if split == selected_split:
                ensemble_acc = split_ensemble_acc
                # Use ensemble accuracy when requested and ordinary accuracy otherwise.
                acc = ensemble_acc if use_ensemble_accuracy else ordinary_acc

        # Joint diffusion fits reuse the generator history when no separate classifier fit
        # ran.
        if use_diffusion_classifier and not history \
        and generative_history is not None:
            history = generative_history

        optimizer_iterations_after = _optimizer_iteration_metrics(
            new_model,
            generative_model
        )
        task_resource["optimizer_updates"] = {
            name: int(value - optimizer_iterations_before.get(name, 0))
            for name, value in optimizer_iterations_after.items()
        }
        # Record the teacher branch only when this task actually used a previous teacher.
        task_resource["snapshot_network_name"] = snapshot_network_name if previous_teacher is not None \
                                                else None
        task_resource["seconds"]["task_total"] = float(
            time.perf_counter() - task_wall_start
        )

        histories.append(history)
        generative_histories.append(generative_history)
        classifier_evaluations_list.append(classifier_evaluations)
        generative_evaluations_list.append(generative_evaluations)
        task_resource_metrics.append(task_resource)
        task_mechanistic_metrics.append(task_mechanistic)
        acc_list.append(acc)
        # Append ensemble trajectory values only when ensemble evaluation produced a result.
        if ensemble_acc is not None:
            ensemble_acc_list.append(ensemble_acc)

        # Store a frozen copy of the just-completed student, not the narrower
        # teacher that was consumed during this task. It is immediately valid
        # for the next task and has a restorable topology.
        # Create it before saving so teacher weights exist in the checkpoint.
        if use_distillation:
            completed_teacher = generative_model.snapshot_teacher_network(
                network_name=snapshot_network_name
            )
            generative_model.set_teacher_network(completed_teacher)

        # Persist task-boundary state when checkpointing is enabled and has a destination.
        if save_task_checkpoints and checkpoint_dir is not None:
            # Materialize every optimizer slot at the save boundary. This
            # makes the checkpoint object graph complete even when a phase did
            # not happen to receive gradients in the just-finished task.
            _prepare_optimizer_slots()
            checkpoint_trackables = _recovery_trackables()
            trackable_topology = _trackable_topology_descriptor(checkpoint_trackables)
            checkpoint_state = {
                "class_order": class_order,
                "task_groups": original_task_groups,
                **task_state,
                "previous_replay_samples": previous_replay_samples,
                "previous_replay_labels": previous_replay_labels,
                "fingerprint": run_fingerprint,
                "run_descriptor": run_descriptor,
                "trackable_topology": trackable_topology,
                "trackable_topology_fingerprint": fingerprint_state(trackable_topology)
            }
            # Include replay-buffer state for buffered runs; other runs save no buffer.
            checkpoint_path = save_task_checkpoint(
                checkpoint_dir,
                task_index,
                checkpoint_state,
                checkpoint_trackables,
                rng_state=capture_rng_state(numpy_generator=rng),
                replay_buffer=buffer if use_buffer else None,
                fingerprint=run_fingerprint
            )
            checkpoint_paths.append(str(checkpoint_path))

        # Verbose runs print the completed task's selected accuracy.
        if verbose:
            # Label the reported trajectory as validation in development and test otherwise.
            split_label = "validation" if experiment_phase == "development" else "test"
            print(f"Task {split_label} accuracy: {acc:.4f}")

            # Print the ensemble score only when ensemble evaluation ran.
            if ensemble_acc is not None:
                print(f"Task ensemble accuracy: {ensemble_acc:.4f}")

            print(75*'-'+'\n')

    # Draw the continual accuracy curve only when plotting is enabled.
    if plot_results:
        CL_plot(
            class_num,
            [(acc_list, " ")],
            class_counts=[
                sum(group_sizes[:index + 1])
                for index in range(len(group_sizes))
            ]
        )

    # Select ensemble test metrics when requested; otherwise select ordinary test metrics.
    selected_test_matrix = ensemble_accuracy_matrix \
                        if use_ensemble_accuracy else ordinary_accuracy_matrix
    # Select ensemble validation metrics when requested; otherwise use ordinary validation
    # metrics.
    selected_validation_matrix = validation_ensemble_accuracy_matrix \
                                if use_ensemble_accuracy else validation_accuracy_matrix
    # Expose the validation matrix in development and the selected test matrix otherwise.
    accuracy_matrix = selected_validation_matrix \
                    if experiment_phase == "development" else selected_test_matrix
    # Compute locked-test summaries only outside development; development has no test
    # summaries.
    continual_metrics = _continual_metrics(selected_test_matrix) \
                        if experiment_phase != "development" else {}
    # Compute validation summaries only when a validation matrix was collected.
    validation_continual_metrics = _continual_metrics(
        selected_validation_matrix
    ) if selected_validation_matrix else {}
    new_task_accuracy, old_task_accuracy = _task_accuracy_summaries(accuracy_matrix)

    # Detailed callers receive models, matrices, histories, and recovery metadata.
    if return_details:
        return {
            **task_state,
            "class_order": class_order,
            "task_classes": original_task_groups,
            "accuracy_matrix": accuracy_matrix,
            "new_task_accuracy": new_task_accuracy,
            "old_task_accuracy": old_task_accuracy,
            "continual_metrics": continual_metrics,
            "validation_continual_metrics": validation_continual_metrics,
            "use_ensemble_accuracy": use_ensemble_accuracy,
            "baseline": baseline,
            "experiment_phase": experiment_phase,
            "experiment_manifest_hash": authenticated_manifest_hash,
            "experiment_run_id": experiment_run_id,
            "test_evaluated": experiment_phase != "development",
            "snapshot_network_name": snapshot_network_name,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_paths": checkpoint_paths,
            "resumed_from": resume_from,
            "next_task_index": len(internal_task_groups),
            "run_descriptor": run_descriptor,
            "model": prev_model,
            "generative_model": generative_model,
            "dtype_policy": dtype_policy,
            "seed": seed
        }
    return acc_list


def continually_learn(
    config: Config | dict[str, object] | None = None,
    teacher_network: tf.keras.Model | None = None,
    **kwargs: object
) -> list[float] | dict[str, object]:
    """Run continual learning through a typed Config, configuration mapping, or direct keywords.

    Config mode requires training.task='continual' and delegates model/data construction,
    fitting, reporting, and artifact handling to common.train.main. Direct mode calls the
    task runner with existing objects and does not merge keyword overrides into a supplied
    Config. Model mutations, split isolation, random streams, plotting, and checkpoint
    effects follow the selected mode.

    Args:
        config (Config | dict[str, object] | None): Config object or section mapping passed
            to Config(**config); None selects direct _run_continual_tasks mode. Defaults to
            ``None``.
        teacher_network (tf.keras.Model | None): Optional runtime first-task diffusion
            teacher, forwarded independently of serializable configuration. None supplies no
            override for a teacher already attached to a direct-mode wrapper. Defaults to
            ``None``.
        **kwargs (object): Direct-mode arguments documented by _run_continual_tasks,
            including required class_num and load_dataset_fn. Empty by default; ignored when
            config is supplied.

    Returns:
        list[float] | dict[str, object]: Direct mode returns only accuracies unless
        return_details=True is passed. Config mode follows
        config.continually_learn.return_details; detailed output additionally includes the
        common main pipeline's evaluations mapping.

    Raises:
        ValueError: A supplied configuration selects a non-continual task. Other
        construction/training errors propagate from the delegated pipeline.
    """

    # Direct keyword calls bypass Config construction and invoke the task loop.
    if config is None:
        kwargs.setdefault("return_details", False)

        return _run_continual_tasks(teacher_network=teacher_network, **kwargs)

    # Convert mapping-based configuration into the typed Config API.
    if isinstance(config, dict):
        config = Config(**config)

    task = normalize_training_task(config.training.task)

    # Reject configurations that select a non-continual training task.
    if task != "continual":
        raise ValueError(
            "continually_learn(config) requires training.task='continual'."
        )


    from common.train import main


    run = main(config, teacher_network=teacher_network)
    model = run["model"]

    details = model["continual_details"]
    details["evaluations"] = run["evaluations"]

    # Configured detailed output returns the full run; other calls return the accuracy
    # trajectory.
    if config.continually_learn.return_details:
        return details
    return details["accuracies"]
