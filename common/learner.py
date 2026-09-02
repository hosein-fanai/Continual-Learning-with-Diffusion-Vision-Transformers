"""Configurable class-incremental learning with replay alternatives."""

from __future__ import annotations

import tensorflow as tf

from sklearn.metrics import accuracy_score

import numpy as np

import time

from copy import deepcopy

from collections.abc import Callable, Sequence

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
from common.runtime import (
    configure_runtime, 
    derive_seed, 
    validate_model_dtype_policy
)
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
    """Clone a configured optimizer so a run never mutates caller state."""

    if not isinstance(value, tf.keras.optimizers.Optimizer):
        return value

    try:
        return tf.keras.optimizers.deserialize(
            tf.keras.optimizers.serialize(value)
        )
    except Exception as error:
        raise ValueError(
            "Configured optimizer must support Keras "
            "serialization for exact continual recovery."
        ) from error


def _optimizer_iteration_metrics(
    classifier: tf.keras.Model | None, 
    generative_model: tf.keras.Model | None
) -> dict[str, int]:
    """Read named optimizer iteration counters without mutating either model."""

    counters = {}
    for role, model in (
        ("classifier", classifier), 
        ("replay", generative_model)
    ):
        if model is None:
            continue

        for attribute in ("optimizer", "gen_optimizer", "clf_optimizer"):
            optimizer = getattr(model, attribute, None)
            iterations = getattr(optimizer, "iterations", None)
            if iterations is None:
                continue

            try:
                counters[f"{role}_{attribute}"] = int(iterations.numpy())
            except (AttributeError, TypeError, ValueError):
                continue

    return counters


def _validate_supplied_model_runtime(
    model: tf.keras.Model | None, 
    seed: int | None, 
    dtype_policy: str, 
    role: str
) -> None:
    """Validate a caller-built continual model before it can be mutated."""

    if model is None:
        return

    validate_model_dtype_policy(model, dtype_policy, role=role)
    # An unseeded experiment imposes only the numerical-policy contract.
    if seed is None:
        return

    seeded_components = [(role, model)]
    raw_network = getattr(model, "network", None)
    if isinstance(raw_network, tf.keras.Model):
        seeded_components.append((role + " raw network", raw_network))

    for component_role, component in seeded_components:
        if not hasattr(component, "seed"):
            continue

        component_seed = getattr(component, "seed")
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
    """Return one stable task-local seed for a generative callback."""

    callback_name = _qualified_name(callback)
    # Preserve the established experiment-seeded callback stream exactly.
    if task_seed is not None:
        return derive_seed(
            task_seed,
            "generative_callback",
            callback_index,
            callback_name,
        )
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
    """Reset stochastic Keras/model streams at a reproducible task boundary."""

    if model is None:
        return

    components = []
    pending = [model]
    discovered: set[int] = set()
    while pending:
        component = pending.pop()
        if id(component) in discovered:
            continue
        discovered.add(id(component))
        components.append(component)
        children = list(getattr(component, "layers", ()))
        children.extend(
            child for child in getattr(
                component, "_self_tracked_trackables", ()
            )
            if isinstance(child, tf.Module)
        )
        pending.extend(reversed(children))

    visited: set[int] = set()
    for component_index, component in enumerate(components):
        # Reset shared layers exactly once even when multiple paths reach them.
        if id(component) in visited:
            continue

        visited.add(id(component))

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

        if random_generator is not None and component_seed is not None:
            # Legacy Keras random layers increment this private Python counter
            # while tracing; checkpoints intentionally omit it.
            random_generator._seed = component_seed
            generator = getattr(random_generator, "_generator", None)
            if generator is not None and hasattr(generator, "reset_from_seed"):
                generator.reset_from_seed(component_seed)

        # Reset repository stochastic layers such as DropPath, which use their
        # public seed directly instead of Keras BaseRandomLayer.
        if component_seed is not None and random_generator is None \
        and hasattr(component, "seed"):
            component.seed = component_seed

    if task_seed is not None and hasattr(model, "seed"):
        model.seed = int(task_seed)
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
    """Map a supported loader representation to diffusion model space."""

    variable_dtype = tf.keras.mixed_precision.global_policy().variable_dtype
    numpy_dtype = tf.as_dtype(variable_dtype).as_numpy_dtype
    x = np.asarray(x, dtype=numpy_dtype)
    if x.ndim == 3:
        x = x[..., None]

    return ((x - data_min) / data_range * 2.) - 1.


def _select_classes(
    x: np.ndarray, 
    y: np.ndarray, 
    classes: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Select rows for class IDs without changing preprocessing space."""

    labels = np.asarray(y)
    if labels.ndim > 1 and labels.shape[-1] > 1:
        label_ids = np.argmax(labels, axis=-1)
    else:
        label_ids = labels.reshape(-1)

    selected = np.isin(label_ids, classes)

    return np.asarray(x)[selected], labels[selected]


def _remap_continual_labels(
    labels: np.ndarray | None, 
    class_order: Sequence[int], 
    onehot_labels: bool
) -> np.ndarray | None:
    """Map original dataset labels to their contiguous schedule positions."""

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
    """Predict class scores from a diffusion classifier in data batches."""

    x = _prepare_diffusion_x(x, data_min, data_range)
    y = np.asarray(y).reshape(-1)
    network = model.get_network(model.test_network_name)
    predictions = []

    for start in range(0, len(x), batch_size):
        end = start + batch_size
        x_batch = x[start:end]

        if isinstance(model, DiffusionClassifierV2):
            t_batch, x_batch, null_labels, _ = model.prep_clfv2_inputs(
                (x_batch, y[start: end]),
                model.clf_test_noisified_max_timesteps
            )
        else:
            t_batch = np.zeros((len(x_batch),), dtype="int32")
            null_labels = np.zeros((len(x_batch),), dtype="uint8")

        predictions.append(network.predict_class(
            (x_batch, t_batch, null_labels),
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
    """Evaluate timestep-ensemble accuracy separately for learned tasks."""

    labels = np.asarray(y)
    label_ids = np.argmax(labels, axis=-1) if labels.ndim > 1 and labels.shape[-1] > 1 \
                else labels.reshape(-1)
    row = [np.nan] * total_group_num
    ensemble_options = dict(options)

    if ensemble_options.get("seed") is None:
        ensemble_options["seed"] = seed

    for group_index, group in enumerate(learned_groups):
        selected = np.isin(label_ids, group)
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


def _reported_accuracy(
    evaluations: object,
    ensemble: bool = False
) -> float | None:
    """Find a classifier accuracy value in nested report output."""

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
        if name in evaluations:
            return float(evaluations[name])

    if not ensemble:
        for name, value in evaluations.items():
            if "accuracy" in name.lower() and np.isscalar(value):
                return float(value)

    for value in evaluations.values():
        accuracy = _reported_accuracy(value, ensemble=ensemble)
        if accuracy is not None:
            return accuracy

    return None


def _has_positive_distillation_objective(model: DiffusionModel) -> bool:
    """Return whether a diffusion wrapper has an active teacher objective."""

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
    if regularizer_kwargs is None:
        regularizer_kwargs = getattr(
            model.network, "cls_token_regularizer_kwargs", {}
        )

    uses_teacher_regularizer = (
        isinstance(model, DiffusionClassifier)
        and
        ctr_loss_coef > 0.
        and regularizer_kwargs.get("train_type", "normal") in (
            "distil", "both"
        )
    )
    return noise_distil_loss_coef > 0. or clf_distil_loss_coef > 0. \
        or uses_teacher_regularizer


def _flatten_example_rows(values: np.ndarray) -> np.ndarray:
    """Flatten trailing feature axes while retaining an empty row dimension."""

    array = np.asarray(values)
    if array.ndim < 2:
        raise ValueError("Continual inputs must include a feature axis.")

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
    """Load and prepare the shared array view used by every task."""

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
    if all_x_val is not None:
        all_x_val, all_y_val = _limit_samples(
            all_x_val,
            all_y_val,
            max_val_samples,
            rng
        )

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


def _sample_diffusion_replay(
    generative_model: DiffusionModel,
    labels: np.ndarray,
    batch_size: int,
    seed: int | None,
    empty_samples: np.ndarray,
) -> np.ndarray:
    """Sample aligned diffusion replay without materializing one large batch."""

    chunks = []
    for chunk_index, start in enumerate(range(0, len(labels), batch_size)):
        chunk_labels = labels[start:start + batch_size]
        chunks.append(generative_model.sample(
            network_name=generative_model.test_network_name,
            labels=chunk_labels + int(generative_model.use_cfg),
            seed=derive_seed(seed, "replay_sample_chunk", chunk_index),
        ).numpy())

    return np.concatenate(chunks, axis=0) if chunks else empty_samples


def _predict_teacher_probabilities(
    teacher: tf.keras.Model, 
    x: np.ndarray, 
    data_min: float, 
    data_range: float, 
    batch_size: int
) -> np.ndarray:
    """Evaluate one frozen diffusion classifier on replay candidates."""

    predict_class = getattr(teacher, "predict_class", None)
    # Cognitive gates require an actual classifier teacher.
    if not callable(predict_class):
        raise TypeError("replay scoring requires a teacher with predict_class().")

    diffusion_x = _prepare_diffusion_x(x, data_min, data_range)
    predictions = []
    for start in range(0, len(diffusion_x), batch_size):
        batch = diffusion_x[start:start + batch_size]
        timesteps = np.zeros((len(batch),), dtype="int32")
        null_labels = np.zeros((len(batch),), dtype="uint8")
        predictions.append(np.asarray(predict_class(
            (batch, timesteps, null_labels), training=False
        )))

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
    """Resolve one optional named baseline into existing continual switches."""

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
    if baseline not in supported:
        raise ValueError(f"Unsupported continual baseline: {baseline!r}.")

    buffer_kwargs = dict(buffer_kwargs)
    remove_prev_classes = baseline != "cumulative"

    if baseline in ("sequential", "cumulative"):
        use_buffer = False
        use_generative_replay = False
        use_generative_model_classifier = False
        use_distillation = False
        if generative_model is not None:
            raise ValueError(f"{baseline} baseline requires no generative_model.")
    elif baseline == "reservoir_er":
        use_buffer = True
        use_generative_replay = False
        use_generative_model_classifier = False
        use_distillation = False
        buffer_kwargs["strategy"] = "reservoir"
        if generative_model is not None:
            raise ValueError("reservoir_er requires no generative_model.")
    # Configure LwF-style teacher learning without generated rehearsal.
    elif baseline == "lwf":
        use_buffer = False
        use_generative_replay = False
        use_generative_model_classifier = True
        use_distillation = True
        # The current teacher API belongs to diffusion classifiers.
        if not isinstance(generative_model, (
            DiffusionClassifier, DiTClassifier,
            DiTEncoderDecoderClassifier, UNetClassifier,
        )):
            raise ValueError("lwf requires a DiffusionClassifier model.")
    elif baseline == "vae_replay":
        use_buffer = False
        use_generative_replay = True
        use_generative_model_classifier = False
        use_distillation = False
        # Keep this baseline generator-only; VAEClassifier adds an unmatched
        # auxiliary classifier objective and belongs in a separate ablation.
        if not isinstance(generative_model, VariationalAutoencoder) \
        or isinstance(generative_model, VAEClassifier):
            raise ValueError("vae_replay requires a generator-only VAE.")
    elif baseline == "diffusion_replay":
        use_buffer = False
        use_generative_replay = True
        use_generative_model_classifier = False
        use_distillation = False
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
    else:
        use_buffer = False
        use_generative_model_classifier = True
        use_generative_replay = baseline in ("joint_replay", "joint_both")
        use_distillation = baseline in ("joint_kd", "joint_both")
        if not isinstance(generative_model, (
            DiffusionClassifier, DiTClassifier,
            DiTEncoderDecoderClassifier, UNetClassifier,
        )):
            raise ValueError(f"{baseline} requires a DiffusionClassifier.")

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
    generative_callbacks_list: Sequence[
        tf.keras.callbacks.Callback
    ] | None = None, 
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
    """Train a class-incremental schedule with optional replay and recovery."""

    # A committed checkpoint owns the already-materialized stochastic
    # schedule. Reading it before schedule construction avoids changing class
    # order merely because NumPy's permutation implementation changes between
    # supported environments. Explicit caller schedules must still agree.
    if resume_from is not None:
        schedule_checkpoint = load_task_checkpoint(resume_from)
        saved_order = list(schedule_checkpoint.class_order)
        saved_groups = [list(group) for group in schedule_checkpoint.task_groups]
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
        matching_runs = [
            run for run in materialize_run_plan(
                frozen_manifest,
                expected_hash=experiment_manifest_hash,
            )
            if run["run_id"] == experiment_run_id
        ]
        # A run identifier must name exactly one frozen condition-stream cell.
        if len(matching_runs) != 1:
            raise ValueError(
                "experiment_run_id is not unique in the frozen manifest."
            )
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
        if planned_stream["stream_seed"] != seed:
            raise ValueError(
                "Confirmation seed differs from the frozen manifest run."
            )
        authenticated_manifest_hash = frozen_manifest["manifest_hash"]

    dtype_policy = dtype_policy or tf.keras.mixed_precision.global_policy().name
    configure_runtime(seed, dtype_policy, deterministic_ops)
    _validate_supplied_model_runtime(
        generative_model, 
        seed, 
        dtype_policy, 
        "continual replay model"
    )

    if isinstance(generative_model, VAEClassifier):
        raise ValueError(
            "Continual VAEClassifier is unsupported because its fixed "
            "full-class head exposes future logits. Use a conditioned "
            "VariationalAutoencoder with the expanding external classifier."
        )

    if initial_classifier is not None:
        validate_model_dtype_policy(
            initial_classifier,
            dtype_policy,
            role="initial continual classifier",
        )

    # A runtime teacher must use the same numerical policy as its student.
    if teacher_network is not None:
        validate_model_dtype_policy(
            teacher_network,
            dtype_policy,
            role="initial continual teacher",
        )

    class_num = len(class_order)
    group_sizes = [len(group) for group in original_task_groups]
    boundaries = np.cumsum([0, *group_sizes])
    internal_task_groups = [
        list(range(int(boundaries[index]), int(boundaries[index + 1])))
        for index in range(len(group_sizes))
    ]

    if fit_method not in ("fit", "fit_progressively"):
        raise ValueError(
            "fit_method must be 'fit' or 'fit_progressively'."
        )

    fit_kwargs = dict(fit_kwargs or {})
    reserved_fit_keys = {
        "x", "y", "epochs", "initial_epoch", 
        "validation_data", "callbacks", "verbose"
    }
    conflicting_fit_keys = sorted(reserved_fit_keys.intersection(fit_kwargs))
    if conflicting_fit_keys:
        raise ValueError(
            "fit_kwargs cannot override continual orchestration arguments: "
            + str(conflicting_fit_keys)
        )

    if fit_method == "fit_progressively" \
    and fit_kwargs.get("stage_tasks") is None:
        raise ValueError(
            "fit_kwargs must include stage_tasks for fit_progressively."
        )

    from common.train import report, train_model


    def _train_task_model(
        model: tf.keras.Model, 
        trainset: object, 
        valset: object | None = None, 
        task_callbacks: Sequence[tf.keras.callbacks.Callback] | None = None, 
        fit_method: str = "fit", 
        fit_kwargs: dict[str, object] | None = None, 
    ) -> dict[str, list[float]]:
        phase_fit_kwargs = dict(fit_kwargs or {})
        phase_trainset = trainset
        # Repeat a finite task dataset only for the explicit fixed-update
        # protocol; VAE's array-based ``train`` method owns its own repetition.
        if optimizer_steps_per_epoch is not None:
            phase_fit_kwargs["steps_per_epoch"] = optimizer_steps_per_epoch
            if isinstance(phase_trainset, tf.data.Dataset):
                phase_trainset = phase_trainset.repeat()

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
        patience = legacy_patience if callback_patience is None else callback_patience
        selected = []

        if patience > 0:
            selected = get_callbacks(
                monitor=callback_monitor or default_monitor, 
                mode=callback_monitor_mode or default_mode, 
                patience=patience, 
                verbose=verbose
            )

        if callbacks_list is not None:
            selected += list(callbacks_list)

        if include_generative and generative_callbacks_list is not None:
            selected += list(generative_callbacks_list)

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
    if "optimizer" in compile_args:
        compile_args["optimizer"] = _fresh_optimizer(
            compile_args["optimizer"]
        )

    ensemble_accuracy_kwargs = dict(ensemble_accuracy_kwargs or {})
    snapshot_network_name = str(snapshot_network_name).lower()
    # Restrict teacher snapshots to actual wrapper network branches.
    if snapshot_network_name not in ("raw", "ema"):
        raise ValueError(
            "snapshot_network_name must be 'raw' or 'ema'."
        )

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
    if experiment_phase not in ("legacy", "development", "confirmation"):
        raise ValueError(
            "experiment_phase must be 'legacy', "
            "'development', or 'confirmation'."
        )

    replay_budget_mode = str(replay_budget_mode).lower()
    # Keep replay accounting under a known per-class or fixed-total contract.
    if replay_budget_mode not in ("legacy", "fixed_total"):
        raise ValueError(
            "replay_budget_mode must be 'legacy' or 'fixed_total'."
        )

    if optimizer_steps_per_epoch is not None:
        optimizer_steps_per_epoch = int(optimizer_steps_per_epoch)
        if optimizer_steps_per_epoch <= 0:
            raise ValueError("optimizer_steps_per_epoch must be positive.")
        # Keep the dedicated all-phase control unambiguous with the historical
        # diffusion-only fit mapping.
        if "steps_per_epoch" in fit_kwargs:
            raise ValueError(
                "Set optimizer_steps_per_epoch or fit_kwargs.steps_per_epoch, "
                "not both."
            )

    # A matched old-data budget must be named explicitly in fixed-total mode.
    if replay_budget_mode == "fixed_total" and replay_old_examples is None:
        raise ValueError(
            "fixed_total replay requires replay_old_examples to be set."
        )

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
    if len(original_task_groups[0]) == 1 \
    and remove_prev_classes and not has_old_replay:
        raise ValueError(
            "A new-only singleton-first schedule cannot learn its one-way "
            "softmax before discarding that class. Use cumulative or a "
            "positive buffered/generative replay exposure instead."
        )
    replay_candidate_multiplier = int(replay_candidate_multiplier)
    if replay_candidate_multiplier <= 0:
        raise ValueError("replay_candidate_multiplier must be a positive integer.")
    replay_selection = str(replay_selection).lower()
    replay_selection_names = {
        "all", "uniform", "random", "confidence", "surprise",
        "confidence_surprise",
    }
    if replay_selection not in replay_selection_names:
        raise ValueError(
            "replay_selection must be one of "
            f"{sorted(replay_selection_names)}."
        )
    replay_surprise_weight = float(replay_surprise_weight)
    replay_cache_mode = "off" if replay_cache_mode is False \
        else str(replay_cache_mode).lower()
    mechanistic_max_samples = int(mechanistic_max_samples)
    if use_buffer and generative_model is not None:
        raise ValueError(
            "The replay buffer and a generative model cannot be used together."
        )
    if replay_budget_mode == "fixed_total" \
    and replay_old_examples > 0 \
    and not (use_buffer or (
        generative_model is not None and use_generative_replay
    )):
        raise ValueError(
            "A positive fixed old-example budget requires buffer or "
            "generative replay."
        )

    if return_features is None:
        return_features = "dnn" in str(tuned_model_path).lower()
    else:
        return_features = bool(return_features)

    if pad and return_features:
        raise ValueError("pad is not supported for saved feature inputs.")

    if use_buffer:
        buffer = ReplayBuffer(
            maxlen=buffer_kwargs["maxlen"], 
            seed=derive_seed(seed, "replay_buffer")
                if seed is not None else buffer_kwargs["seed"], 
            strategy=buffer_kwargs["strategy"]
        )

    if isinstance(generative_model, (
        DiTClassifier, 
        DiTEncoderDecoderClassifier, 
        UNetClassifier
    )):
        generative_model = DiffusionClassifier(
            network=generative_model, 
            teacher_network=teacher_network,
            defer_teacher=use_distillation,
            clf_distil_loss_coef=8.6e-3 if use_distillation else 0.,
            mask_by_nulls=generative_model.use_cfg, 
            test_steps=min(50, generative_model.timesteps),
            seed=seed,
        )
        generative_model.compile(
            **generative_model_compile_args
        )
    # Wrap raw generator-only diffusion networks.
    elif isinstance(generative_model, (
        DiTDecoder, 
        DiTEncoderDecoder, 
        DiffusionTransformer, 
        UNet
    )):
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
    ) and getattr(
        generative_model, 
        "teacher_network", None
    ) is not teacher_network:
        generative_model.set_teacher_network(teacher_network)
    elif generative_model is not None and not isinstance(
        generative_model, 
        (VariationalAutoencoder, DiffusionModel)
    ):  # Reject objects that cannot provide supported generative replay.
        raise TypeError(
            "generative_model must be a supported VAE, "
            "diffusion network, or diffusion wrapper."
        )

    # Reject teachers for replay models without a diffusion wrapper.
    if teacher_network is not None and not isinstance(
        generative_model, 
        DiffusionModel
    ):
        raise ValueError(
            "teacher_network requires a diffusion generative_model."
        )

    # Restrict automatic self-distillation to diffusion wrappers.
    if use_distillation and not isinstance(
        generative_model, 
        DiffusionModel
    ):
        raise ValueError(
            "use_distillation requires a diffusion generative_model."
        )

    scored_replay_selection = replay_selection in {
        "confidence", 
        "surprise", 
        "confidence_surprise"
    }
    if scored_replay_selection and not isinstance(
        generative_model, DiffusionClassifier
    ):
        raise ValueError(
            "confidence/surprise replay selection "
            "requires a DiffusionClassifier teacher."
        )

    if replay_cache_mode != "off" and not (
        generative_model is not None and use_generative_replay
    ):
        raise ValueError(
            "Replay candidate caching requires enabled generative replay."
        )

    # Reject a silent raw fallback when an EMA-teacher ablation was requested.
    needs_snapshot_teacher = use_distillation or scored_replay_selection or (
        mechanistic_metrics and isinstance(generative_model, DiffusionClassifier)
    )
    # EMA treatments must never fall back silently to the raw student branch.
    if needs_snapshot_teacher and snapshot_network_name == "ema" \
    and getattr(generative_model, "ema_network", None) is None:
        raise ValueError(
            "snapshot_network_name='ema' requires EMA to be enabled."
        )

    # Require an independent student head for continual self-distillation.
    if use_distillation \
    and isinstance(generative_model, DiffusionClassifier) \
    and float(tf.keras.backend.get_value(
        generative_model.clf_distil_loss_coef
    )) > 0. \
    and getattr(generative_model.network, "distil_token", None) is None:
        raise ValueError(
            "use_distillation requires the diffusion classifier to have a "
            "distil_token."
        )
    if use_distillation and not _has_positive_distillation_objective(
        generative_model
    ):
        raise ValueError(
            "use_distillation requires a positive distillation objective."
        )
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
        raise ValueError(
            "fit_progressively requires a diffusion replay model."
        )

    if isinstance(generative_model, DiffusionModel) and not getattr(
        generative_model.network, "dynamic_num_classes", False
    ):
        raise ValueError(
            "Continual diffusion networks must be initialized with "
            "num_classes=None."
        )

    if evaluate_ensemble_accuracy and not isinstance(
        generative_model, DiffusionClassifier
    ):
        raise ValueError(
            "evaluate_ensemble_accuracy requires a DiffusionClassifier "
            "or DiffusionClassifierV2 generative_model."
        )

    if use_generative_model_classifier and not isinstance(
        generative_model, DiffusionClassifier
    ):
        raise ValueError(
            "use_generative_model_classifier requires a "
            "DiffusionClassifier."
        )
    use_diffusion_classifier = use_generative_model_classifier
    if use_diffusion_classifier:
        return_features = False

    if isinstance(generative_model, VariationalAutoencoder):
        if not generative_model.conditioned:
            raise ValueError("A replay VAE must be conditioned.")
        if not load_dataset_fn_kwargs["onehot_labels"]:
            raise ValueError("VAE replay requires one-hot labels.")
    elif isinstance(generative_model, DiffusionModel):
        if return_features:
            raise ValueError("Diffusion replay requires image data.")

    if use_diffusion_classifier:
        # V2 trains discriminator variables separately.
        if isinstance(generative_model, DiffusionClassifierV2) \
        and not train_classifier_separately:  
            raise ValueError(
                "train_classifier_separately must "
                "be True for DiffusionClassifierV2."
            )
        # Standard wrappers train both parts jointly.
        if not isinstance(generative_model, DiffusionClassifierV2) \
        and train_classifier_separately:  
            raise ValueError(
                "train_classifier_separately must "
                "be False for DiffusionClassifier."
            )

        prev_model = generative_model.network.classifier
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

    train_direct_classifier = not use_diffusion_classifier
    score_from_generator = use_diffusion_classifier


    def _recovery_trackables() -> dict[str, object]:
        objects: dict[str, object] = {"classifier": prev_model}
        classifier_optimizer = getattr(prev_model, "optimizer", None)
        if classifier_optimizer is not None:
            objects["classifier_optimizer"] = classifier_optimizer

        if generative_model is not None:
            objects["replay_model"] = generative_model
            replay_optimizer = getattr(generative_model, "optimizer", None)
            if replay_optimizer is not None:
                objects["replay_optimizer"] = replay_optimizer

            generator_optimizer = getattr(
                generative_model, 
                "gen_optimizer", 
                None
            )
            classifier_phase_optimizer = getattr(
                generative_model, 
                "clf_optimizer", 
                None
            )

            if generator_optimizer is not None:
                objects["generator_optimizer"] = generator_optimizer

            if classifier_phase_optimizer is not None:
                objects["classifier_phase_optimizer"] = (
                    classifier_phase_optimizer
            )

            teacher = getattr(generative_model, "teacher_network", None)
            # Save the frozen next-task teacher outside wrapper tracking.
            if teacher is not None:
                objects["teacher"] = teacher

        return objects


    def _prepare_optimizer_slots() -> None:
        optimizer_variables = []
        classifier_optimizer = getattr(prev_model, "optimizer", None)
        if classifier_optimizer is not None:
            optimizer_variables.append(
                (classifier_optimizer, prev_model.trainable_variables)
            )

        if isinstance(generative_model, DiffusionModel):
            generative_model._register_optimizer_variables()
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
        elif generative_model is not None:
            replay_optimizer = getattr(generative_model, "optimizer", None)
            if replay_optimizer is not None:
                optimizer_variables.append(
                    (replay_optimizer, generative_model.trainable_variables)
                )

        prepared = set()
        for optimizer, variables in optimizer_variables:
            if optimizer is None or id(optimizer) in prepared:
                continue

            prepared.add(id(optimizer))
            variables = list(variables)

            if hasattr(optimizer, "_create_all_weights"):
                optimizer._create_all_weights(variables)
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
    if not isinstance(generative_model, DiffusionModel):
        diffusion_data_min, diffusion_data_range = 0., 1.
    else:
        initial_x, _ = _select_classes(
            dataset_arrays[0], 
            dataset_arrays[1], 
            internal_task_groups[0]
        )
        preprocess = load_dataset_fn_kwargs.get("preprocess")

        if preprocess in ("standardize", "diffusion"):
            diffusion_data_min, diffusion_data_range = -1., 2.
        elif preprocess == "min-max":
            diffusion_data_min, diffusion_data_range = 0., 1.
        else:
            diffusion_data_min = float(np.min(initial_x))
            diffusion_data_range = float(
                np.max(initial_x) - diffusion_data_min
            ) or 1.

    initial_generator_weight_descriptor = _model_weight_descriptor(
        generative_model
    )
    dataset_names = (
        "x_train", "y_train", 
        "x_val", "y_val", 
        "x_test", "y_test"
    )
    loader_descriptor = _recovery_descriptor(load_dataset_fn)
    loader_kwargs_descriptor = _recovery_descriptor(load_dataset_fn_kwargs)
    array_descriptors = {
        name: _array_recovery_descriptor(array)
        for name, array in zip(dataset_names, dataset_arrays)
    }
    generator_topology_descriptor = _model_topology_descriptor(
        generative_model
    )
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
        "schema": 3,
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
            "generative_model_kwargs": _recovery_descriptor(
                generative_model_kwargs
            ),
            "budget_mode": replay_budget_mode,
            "old_examples": replay_old_examples,
            "current_examples": replay_current_examples,
            "candidate_multiplier": replay_candidate_multiplier,
            "selection": replay_selection,
            "surprise_weight": replay_surprise_weight,
            "cache_dir": replay_cache_dir,
            "cache_mode": replay_cache_mode,
            "use_generative_model_classifier": bool(
                use_generative_model_classifier
            ),
            "train_classifier_separately": bool(
                train_classifier_separately
            ),
        },
        "distillation_and_metrics": {
            "use_distillation": bool(use_distillation),
            "snapshot_network_name": snapshot_network_name,
            "mechanistic_metrics": bool(mechanistic_metrics),
            "mechanistic_max_samples": mechanistic_max_samples,
            "experiment_phase": experiment_phase,
            "use_ensemble_accuracy": bool(use_ensemble_accuracy),
            "evaluate_ensemble_accuracy": bool(evaluate_ensemble_accuracy),
            "ensemble_accuracy_kwargs": _recovery_descriptor(
                ensemble_accuracy_kwargs
            ),
            "experiment_manifest_hash": authenticated_manifest_hash,
            "experiment_run_id": experiment_run_id,
        },
        "models": {
            "template_artifact": _artifact_recovery_descriptor(
                tuned_model_path
            ),
            "classifier": _model_topology_descriptor(prev_model),
            "classifier_initial_weights": _model_weight_descriptor(prev_model),
            "replay": generator_topology_descriptor,
            "replay_initial_weights": initial_generator_weight_descriptor,
            "initial_classifier": _model_topology_descriptor(
                initial_classifier
            ),
            "initial_classifier_weights": _model_weight_descriptor(
                initial_classifier
            ),
            "initial_teacher": _model_topology_descriptor(teacher_network),
            "initial_teacher_weights": _model_weight_descriptor(
                teacher_network
            ),
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
            if use_diffusion_classifier:
                prev_model = generative_model.network.classifier
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
        expected_topology_fingerprint = saved.get(
            "trackable_topology_fingerprint"
        )
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
        if use_buffer and recovered.replay_state is not None:
            restore_replay_buffer(buffer, recovered.replay_state)

        for name, values in task_state.items():
            values.extend(saved.get(name, []))

        previous_replay_samples = saved.get("previous_replay_samples")
        previous_replay_labels = saved.get("previous_replay_labels")
        checkpoint_paths.append(str(recovered.task_dir))

    if checkpoint_dir is None and resume_from is not None:
        recovery_path = load_task_checkpoint(resume_from).task_dir
        checkpoint_dir = str(recovery_path.parent)

    for task_index in range(start_task_index, len(internal_task_groups)):
        task_wall_start = time.perf_counter()
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
            if x is None:
                return None
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


        for callback_index, callback in enumerate(
            generative_callbacks_list or ()
        ):
            prefix_setter = getattr(callback, "set_artifact_prefix", None)
            if callable(prefix_setter):
                class_text = "-".join(
                    str(label) 
                    for label in original_task_groups[task_index]
                )
                prefix_setter(
                    f"task-{task_index + 1}_classes-{class_text}"
                )

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
        if needs_snapshot_teacher and task_index > 0:
            previous_teacher = generative_model.teacher_network if use_distillation else (
                generative_model.snapshot_teacher_network(
                    network_name=snapshot_network_name
                )
            )

        if use_diffusion_classifier:
            new_model = generative_model.network.classifier
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

            if initial_classifier is not None and (
                task_index == 0 or not keep_same_model
            ):
                copy_model(initial_classifier, new_model, allow_truncate=True)
            elif keep_same_model:
                copy_model(prev_model, new_model)

        optimizer_iterations_before = _optimizer_iteration_metrics(
            new_model, 
            generative_model
        )

        (all_x_train, all_y_train, all_x_val,
        all_y_val, all_x_test, all_y_test) = dataset_arrays
        if remove_prev_classes and task_index > 0:
            train_classes = new_classes
        else:
            train_classes = seen_classes

        x_train, y_train = _select_classes(
            all_x_train, 
            all_y_train, 
            train_classes
        )
        task_resource["current_examples_available"] = int(len(x_train))

        # Fixed-total designs match current-data exposure across treatments.
        if replay_budget_mode == "fixed_total":
            x_train, y_train = _sample_exact_rows(
                x_train, 
                y_train, 
                replay_current_examples, 
                np.random.default_rng(derive_seed(
                    task_seed, 
                    "current_exposure"
                ))
            )

        task_resource["current_examples_exposed"] = int(len(x_train))
        x_test, y_test = _select_classes(
            all_x_test, 
            all_y_test, 
            seen_classes
        )

        if all_x_val is not None and all_y_val is not None:
            x_val, y_val = _select_classes(
                all_x_val, 
                all_y_val, 
                seen_classes
            )
        else:
            x_val, y_val = None, None

        if not use_valset:
            x_val, y_val = None, None

        if isinstance(generative_model, VariationalAutoencoder) \
        and not return_features: # Match configured dense VAE input shapes.
            x_train = _flatten_example_rows(x_train)
            x_test = _flatten_example_rows(x_test)
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

        if use_buffer:
            current_x_train, current_y_train = x_train, y_train
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
                    np.random.default_rng(derive_seed(
                        task_seed, "buffer_candidates"
                    )),
                )
            candidate_ids = np.argmax(y_buffer, axis=-1) \
                if np.asarray(y_buffer).ndim == 2 \
                and np.asarray(y_buffer).shape[1] > 1 \
                else np.asarray(y_buffer).reshape(-1)
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
            selected_replay_y = _restore_replay_label_shape(
                selected_ids, y_train
            )
            task_resource["replay"].update(gate_diagnostics)
        elif use_generative_replay and \
        generative_model is not None \
        and task_index > 0:
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
            candidate_seed = task_seed if legacy_candidate_path else derive_seed(
                task_seed, "replay_candidates"
            )
            expected_candidate_ids = (
                np.repeat(
                    old_classes,
                    generative_model_kwargs["samples_per_class"],
                )
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
            else:
                # VAE generation exposes a per-class API; reduce only the
                # opt-in non-divisible fixed-total case to an exact pool.
                # Preserve a correctly shaped pool for an explicit zero budget.
                if candidate_count == 0:
                    x_buffer = x_train[:0]
                    y_buffer = expected_candidate_y
                elif isinstance(generative_model, VariationalAutoencoder):
                    per_class = int(np.ceil(
                        candidate_count / len(old_classes)
                    )) if candidate_count else 0
                    x_buffer, y_buffer = generative_model.generate(
                        classes=old_classes,
                        samples_per_class=per_class,
                        onehot_y_output=load_dataset_fn_kwargs["onehot_labels"],
                        seed=candidate_seed,
                    )
                    # Reduce a non-divisible per-class draw to the exact pool size.
                    if len(x_buffer) != candidate_count:
                        generated_ids = np.argmax(y_buffer, axis=-1) \
                            if np.asarray(y_buffer).ndim == 2 \
                            and np.asarray(y_buffer).shape[1] > 1 \
                            else np.asarray(y_buffer).reshape(-1)
                        x_buffer, generated_ids, _ = select_replay_candidates(
                            x_buffer,
                            generated_ids,
                            candidate_count,
                            strategy="uniform",
                            seed=derive_seed(candidate_seed, "vae_exact_pool"),
                        )
                        y_buffer = _restore_replay_label_shape(
                            generated_ids, y_train
                        )
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
                    y_buffer = _restore_replay_label_shape(
                        y_buffer_ids, y_train
                    )
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
                if (
                    x_train.ndim == 3
                    and x_buffer.ndim == 4
                    and x_buffer.shape[-1] == 1
                ):
                    x_buffer = x_buffer[..., 0]
            task_resource["seconds"]["generator_sampling"] = float(
                time.perf_counter() - sample_started
            )
            task_resource["replay"]["cache_path"] = cache_path
            candidate_ids = np.argmax(y_buffer, axis=-1) \
                if np.asarray(y_buffer).ndim == 2 \
                and np.asarray(y_buffer).shape[1] > 1 \
                else np.asarray(y_buffer).reshape(-1)

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
            selected_replay_y = _restore_replay_label_shape(
                selected_ids, y_train
            )
            task_resource["replay"].update(gate_diagnostics)

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
                selected_ids = np.argmax(selected_replay_y, axis=-1) \
                    if selected_replay_y.ndim == 2 \
                    and selected_replay_y.shape[1] > 1 \
                    else selected_replay_y.reshape(-1)
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
                retained_count = min(
                    len(selected_replay_x), mechanistic_max_samples
                )
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

        if not use_diffusion_classifier and isinstance(
            classifier_input_shape, tuple
        ) and len(classifier_input_shape) == 2:
            classifier_x_train = _flatten_example_rows(x_train)
            classifier_x_test = _flatten_example_rows(x_test)
            classifier_x_val = _flatten_example_rows(x_val) \
                if x_val is not None else None

        classifier_y_train = y_train
        classifier_y_val = y_val
        classifier_y_test = y_test
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

            if "sparse" in loss_name:
                classifier_y_train = np.argmax(y_train, axis=-1)
                classifier_y_val = np.argmax(y_val, axis=-1) if y_val is not None else None
                classifier_y_test = np.argmax(y_test, axis=-1)
            elif not use_diffusion_classifier:
                classifier_y_train = y_train[..., :seen_class_num]
                classifier_y_val = y_val[..., :seen_class_num] \
                    if y_val is not None else None
                classifier_y_test = y_test[..., :seen_class_num]

        task_shuffle_buffer = len(x_train) if shuffle_buffer is None \
            else shuffle_buffer
        trainset = _task_dataset(
            classifier_x_train, classifier_y_train, training=True
        )
        valset = _task_dataset(classifier_x_val, classifier_y_val)
        testset = _task_dataset(classifier_x_test, classifier_y_test) \
            if experiment_phase != "development" else None

        history = {}
        # Attach standalone-classifier callbacks only when that fit phase runs.
        if train_direct_classifier:
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
        phase_train_num = -1 if replay_budget_mode == "fixed_total" \
                        else generative_model_kwargs["train_num"]
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
            generative_trainset = _task_dataset(
                x_train, y_train, training=True
            )
            generative_valset = _task_dataset(x_val, y_val)
            generative_testset = _task_dataset(x_test, y_test) \
                                if experiment_phase != "development" else None
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
                indices = rng.integers(
                    0, 
                    len(generative_x), 
                    (train_num,)
                )
                generative_x = generative_x[indices]
                generative_y = generative_y[indices]
                generative_replay_mask = generative_replay_mask[indices]

            generative_trainset = _task_dataset(
                generative_x, 
                generative_y, 
                training=True, 
                metadata=generative_replay_mask if replay_only_distillation else None
            )
            generative_y_val = np.argmax(y_val, axis=-1) if load_dataset_fn_kwargs["onehot_labels"] \
                            and y_val is not None else np.asarray(y_val).reshape(-1) \
                            if y_val is not None else None
            generative_valset = _task_dataset(
                _prepare_diffusion_x(
                    x_val, 
                    diffusion_data_min, 
                    diffusion_data_range
                ), 
                generative_y_val, 
                metadata=np.zeros((len(x_val),), dtype=bool) if replay_only_distillation else None
            ) if x_val is not None else None

            # Locked-test diffusion preprocessing is confirmation-only.
            if experiment_phase != "development":
                generative_y_test = np.argmax(
                    y_test, 
                    axis=-1
                ) if load_dataset_fn_kwargs["onehot_labels"] else np.asarray(
                    y_test
                ).reshape(-1)
                generative_testset = _task_dataset(
                    _prepare_diffusion_x(
                        x_test, 
                        diffusion_data_min, 
                        diffusion_data_range
                    ), 
                    generative_y_test, 
                    metadata=np.zeros((len(x_test),), dtype=bool) if replay_only_distillation else None
                )

            if isinstance(generative_model, DiffusionClassifierV2):
                generative_fit_method = "fit_generator_progressively" \
                    if fit_method == "fit_progressively" \
                    else "fit_generator"
            else:
                generative_fit_method = fit_method

            fit_started = time.perf_counter()
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

            if use_diffusion_classifier and isinstance(
                generative_model, DiffusionClassifierV2
            ): # Train V2 classifier variables in their required separate phase.
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
        else:
            generative_history = None

        # Compare the frozen slow/previous trace with the updated student on
        # a bounded old-class training probe. This is optional and never reads
        # the locked test split during development.
        if mechanistic_metrics and previous_teacher is not None and old_classes:
            probe_x, probe_y = _select_classes(
                all_x_train, 
                all_y_train, 
                old_classes
            )
            probe_x, probe_y = _sample_exact_rows(
                probe_x, 
                probe_y, 
                min(len(probe_x), mechanistic_max_samples), 
                np.random.default_rng(derive_seed(
                    task_seed, 
                    "representation_probe"
                ))
            )
            probe_ids = np.argmax(probe_y, axis=-1) \
                        if probe_y.ndim == 2 and probe_y.shape[1] > 1 \
                        else probe_y.reshape(-1)
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
                    np.random.default_rng(derive_seed(
                        task_seed, 
                        "calibration_probe"
                    ))
                )

                if len(calibration_x):
                    calibration_ids = np.argmax(
                        calibration_y, axis=-1
                    ) if calibration_y.ndim == 2 \
                        and calibration_y.shape[1] > 1 \
                        else calibration_y.reshape(-1)
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
                        calibration_metrics(
                            calibration_probabilities,
                            calibration_ids,
                        )
                    )

        generative_evaluations = {}
        evaluation_split_name = (
            "valset" if experiment_phase == "development" else "testset"
        )
        generative_evaluation_set = (
            generative_valset
            if experiment_phase == "development"
            else generative_testset
        )
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
        if not use_diffusion_classifier and classifier_report_history \
        and train_direct_classifier:
            classifier_evaluations = _report_task_model(
                classifier_report_history, 
                new_model, 
                trainset, 
                valset if experiment_phase == "development" else testset,
                split_name=evaluation_split_name,
            )

        accuracy_source = generative_evaluations if score_from_generator \
                        else classifier_evaluations
        # Score the same raw/EMA network selected by the diffusion wrapper.
        if use_diffusion_classifier:
            evaluation_prefix = evaluation_split_name
            selected_evaluation_name = (
                f"{evaluation_prefix}_ema_eval"
                if generative_model.test_network_name == "ema"
                else f"{evaluation_prefix}_network_eval"
            )
            selected_evaluation = generative_evaluations.get(
                selected_evaluation_name
            )
            if isinstance(selected_evaluation, dict):
                accuracy_source = selected_evaluation

        acc = _reported_accuracy(accuracy_source)
        ensemble_acc = None

        # Development runs never execute a model on the locked test split.
        if experiment_phase != "development":
            y_test_ids = np.argmax(y_test, axis=-1) \
                if load_dataset_fn_kwargs["onehot_labels"] \
                else np.asarray(y_test).reshape(-1)

            if use_diffusion_classifier:
                predictions = _predict_diffusion_classes(
                    generative_model,
                    x_test,
                    y_test_ids,
                    diffusion_data_min,
                    diffusion_data_range,
                    batch_size
                )
            else:
                predictions = new_model.predict(
                    classifier_x_test,
                    verbose=verbose
                )

            prediction_ids = np.argmax(predictions, axis=-1)
            prediction_accuracy = float(accuracy_score(
                y_test_ids, prediction_ids
            ))
            if acc is None:
                acc = prediction_accuracy
            matrix_row = [np.nan] * len(internal_task_groups)
            # Score learned groups separately without future-class rows.
            if classification_objective_defined:
                for learned_index, learned_classes in enumerate(
                    internal_task_groups[:task_index + 1]
                ):
                    group_mask = np.isin(y_test_ids, learned_classes)
                    matrix_row[learned_index] = float(accuracy_score(
                        y_test_ids[group_mask],
                        prediction_ids[group_mask],
                    ))
            ordinary_accuracy_matrix.append(matrix_row)

        # Build a validation matrix independently of the locked test matrix so
        # HPO can optimize continual metrics without consulting test results.
        if x_val is not None:
            y_val_ids = np.argmax(y_val, axis=-1) \
                if load_dataset_fn_kwargs["onehot_labels"] \
                else np.asarray(y_val).reshape(-1)
            if use_diffusion_classifier:
                validation_predictions = _predict_diffusion_classes(
                    generative_model,
                    x_val,
                    y_val_ids,
                    diffusion_data_min,
                    diffusion_data_range,
                    batch_size,
                )
            else:
                validation_predictions = new_model.predict(
                    classifier_x_val,
                    verbose=verbose,
                )
            validation_ids = np.argmax(validation_predictions, axis=-1)
            validation_row = [np.nan] * len(internal_task_groups)
            if classification_objective_defined:
                for learned_index, learned_classes in enumerate(
                    internal_task_groups[:task_index + 1]
                ):
                    group_mask = np.isin(y_val_ids, learned_classes)
                    # Leave NaN when split limiting removed every group example.
                    if np.any(group_mask):
                        validation_row[learned_index] = float(accuracy_score(
                            y_val_ids[group_mask],
                            validation_ids[group_mask],
                        ))
            validation_accuracy_matrix.append(validation_row)
            if experiment_phase == "development":
                acc = float(accuracy_score(y_val_ids, validation_ids))

        # Ensemble matrices use actual ensemble evaluations for each task,
        # rather than treating one cumulative scalar as a CL matrix.
        if evaluate_ensemble_accuracy:
            # Keep the locked test ensemble completely outside development.
            if experiment_phase != "development":
                ensemble_row = _ensemble_accuracy_row(
                    generative_model,
                    x_test,
                    y_test,
                    internal_task_groups[:task_index + 1],
                    len(internal_task_groups),
                    diffusion_data_min,
                    diffusion_data_range,
                    batch_size,
                    ensemble_accuracy_kwargs,
                    derive_seed(seed, "ensemble", task_index, "test"),
                    verbose,
                )
                if not classification_objective_defined:
                    ensemble_row = [np.nan] * len(internal_task_groups)
                ensemble_accuracy_matrix.append(ensemble_row)
                # Test ensemble values remain authoritative outside development.
                ensemble_acc = _observed_mean(
                    ensemble_row[:task_index + 1]
                )
                if use_ensemble_accuracy:
                    acc = ensemble_acc
            if x_val is not None:
                validation_ensemble_row = _ensemble_accuracy_row(
                    generative_model,
                    x_val,
                    y_val,
                    internal_task_groups[:task_index + 1],
                    len(internal_task_groups),
                    diffusion_data_min,
                    diffusion_data_range,
                    batch_size,
                    ensemble_accuracy_kwargs,
                    derive_seed(seed, "ensemble", task_index, "validation"),
                    verbose,
                )
                if not classification_objective_defined:
                    validation_ensemble_row = [
                        np.nan
                    ] * len(internal_task_groups)
                validation_ensemble_accuracy_matrix.append(
                    validation_ensemble_row
                )
                if experiment_phase == "development":
                    ensemble_acc = _observed_mean(
                        validation_ensemble_row[:task_index + 1]
                    )
                    if use_ensemble_accuracy:
                        acc = ensemble_acc

        # A one-class softmax has zero cross-entropy gradient and 100% trivial
        # accuracy, so task-one classification is explicitly unavailable.
        if not classification_objective_defined:
            acc = float("nan")
            if ensemble_acc is not None:
                ensemble_acc = float("nan")

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

        if save_task_checkpoints and checkpoint_dir is not None:
            # Materialize every optimizer slot at the save boundary. This
            # makes the checkpoint object graph complete even when a phase did
            # not happen to receive gradients in the just-finished task.
            _prepare_optimizer_slots()
            checkpoint_trackables = _recovery_trackables()
            trackable_topology = _trackable_topology_descriptor(
                checkpoint_trackables
            )
            checkpoint_state = {
                "class_order": class_order, 
                "task_groups": original_task_groups, 
                **task_state, 
                "previous_replay_samples": previous_replay_samples, 
                "previous_replay_labels": previous_replay_labels, 
                "fingerprint": run_fingerprint, 
                "run_descriptor": run_descriptor, 
                "trackable_topology": trackable_topology, 
                "trackable_topology_fingerprint": fingerprint_state(
                    trackable_topology
                )
            }
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

        if verbose:
            split_label = "validation" if experiment_phase == "development" else "test"
            print(f"Task {split_label} accuracy: {acc:.4f}")

            if ensemble_acc is not None:
                print(f"Task ensemble accuracy: {ensemble_acc:.4f}")

            print(75*'-'+'\n')

    if plot_results:
        CL_plot(
            class_num, 
            [(acc_list, " ")], 
            class_counts=[
                sum(group_sizes[:index + 1])
                for index in range(len(group_sizes))
            ]
        )

    selected_test_matrix = ensemble_accuracy_matrix \
                        if use_ensemble_accuracy else ordinary_accuracy_matrix
    selected_validation_matrix = validation_ensemble_accuracy_matrix \
                                if use_ensemble_accuracy else validation_accuracy_matrix
    accuracy_matrix = selected_validation_matrix \
                    if experiment_phase == "development" else selected_test_matrix
    continual_metrics = _continual_metrics(selected_test_matrix) \
                        if experiment_phase != "development" else {}
    validation_continual_metrics = _continual_metrics(
        selected_validation_matrix
    ) if selected_validation_matrix else {}
    new_task_accuracy, old_task_accuracy = _task_accuracy_summaries(
        accuracy_matrix
    )

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
    """Run class-incremental learning through config or direct keyword mode."""

    if config is None:
        kwargs.setdefault("return_details", False)

        return _run_continual_tasks(
            teacher_network=teacher_network, 
            **kwargs
        )

    if isinstance(config, dict):
        config = Config(**config)

    if not isinstance(config, Config):
        raise TypeError("config must be a Config, mapping, or None.")

    task = normalize_training_task(config.training.task)

    if task != "continual":
        raise ValueError(
            "continually_learn(config) requires training.task='continual'."
        )


    from common.train import main


    run = main(
        config, 
        teacher_network=teacher_network
    )
    model = run["model"]

    if not isinstance(model, dict):
        raise TypeError(
            "A continual config must create a model bundle."
        )

    details = model["continual_details"]
    details["evaluations"] = run["evaluations"]

    if config.continually_learn.return_details:
        return details
    return details["accuracies"]
