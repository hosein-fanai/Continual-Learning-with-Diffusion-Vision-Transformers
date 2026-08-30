"""Configurable class-incremental learning with replay alternatives."""

from __future__ import annotations

import tensorflow as tf

from sklearn.metrics import accuracy_score

import numpy as np

from pathlib import Path

from copy import deepcopy

import hashlib

import json

import time

import uuid

from collections.abc import Callable, Sequence
from numbers import Integral, Real

from common.config import Config, resolve_continual_schedule
from common.utils import CL_plot
from common.model import get_model, copy_model, get_callbacks
from common.replay_buffer import ReplayBuffer
from common.dataloader import get_dataset, _limit_samples, _pad_images
from common.mechanistic import (
    calibration_metrics, 
    class_centroid_drift, 
    linear_cka, 
    replay_quality_metrics, 
    select_replay_candidates
)
from common.runtime import (
    configure_runtime, 
    derive_seed, 
    validate_model_dtype_policy
)
from common.recovery import (
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


def _qualified_name(value: object) -> str:
    """Return a stable module-qualified type or callable name.

    Args:
        value (object): Type, callable, or instance to identify.

    Returns:
        str: Import-style module and qualified name without instance identity.
    """

    candidate = value if isinstance(value, type) else type(value)
    # Prefer a function/method's own definition identity over its metatype.
    if callable(value) and not isinstance(value, type):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)

        # Use the callable identity only when both stable components exist.
        if module is not None and qualname is not None:
            return f"{module}.{qualname}"

    return f"{candidate.__module__}.{candidate.__qualname__}"


def _array_recovery_descriptor(value: object) -> dict[str, object] | None:
    """Describe array content compactly for compatibility checks.

    Args:
        value (object): Array-compatible value, or ``None``.

    Returns:
        dict[str, object] | None: Shape, dtype, and content hash, or ``None``.

    Raises:
        TypeError: If an object-dtype array cannot be hashed portably.
    """

    # Preserve a missing optional validation split in the descriptor.
    if value is None:
        return None

    array = np.ascontiguousarray(np.asarray(value))
    # Reject arrays whose bytes contain process-local Python object pointers.
    if array.dtype.hasobject:
        raise TypeError("Object-dtype datasets cannot be recovery-fingerprinted.")

    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    # An empty split has no payload bytes, and Python rejects casting a
    # zero-sized multidimensional memoryview. Shape and dtype still distinguish
    # every valid empty descriptor without allocating a temporary byte string.
    if array.size:
        digest.update(memoryview(array).cast("B"))

    return {
        "shape": list(array.shape), 
        "dtype": array.dtype.str, 
        "sha256": digest.hexdigest()
    }


def _artifact_recovery_descriptor(path: str) -> dict[str, object] | None:
    """Hash a model template without binding it to a filesystem location.

    Args:
        path (str): Model file/directory path, or an empty string.

    Returns:
        dict[str, object] | None: Content signature, missing marker, or ``None``.
    """

    # An empty tuned-model path denotes no external template artifact.
    if not path:
        return None

    artifact = Path(path)

    # Hash one-file formats such as HDF5 without loading them into memory.
    if artifact.is_file():
        digest = hashlib.sha256()
        with artifact.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)

        return {
            "kind": "file", 
            "size": artifact.stat().st_size, 
            "sha256": digest.hexdigest()
        }

    # Hash every relative file for directory formats such as SavedModel.
    if artifact.is_dir():
        files = []
        for child in sorted(item for item in artifact.rglob("*") if item.is_file()):
            digest = hashlib.sha256()
            with child.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)

            files.append({
                "path": child.relative_to(artifact).as_posix(), 
                "size": child.stat().st_size, 
                "sha256": digest.hexdigest()
            })

        return {"kind": "directory", "files": files}

    return {"kind": "missing"}


def _model_weight_descriptor(model: object) -> list[dict[str, object]] | None:
    """Hash configured initial weights that may seed future tasks.

    Args:
        model (object): Keras-compatible model, or ``None``.

    Returns:
        list[dict[str, object]] | None: Ordered weight hashes, or ``None``.
    """

    # Preserve the absence of an optional initial classifier or teacher.
    if model is None:
        return None

    return [
        _array_recovery_descriptor(weight.numpy())
        for weight in list(getattr(model, "weights", []) or [])
    ]


def _recovery_descriptor(
    value: object, 
    active_ids: set[int] | None = None, 
    strip_config_names: bool = False
) -> object:
    """Convert configuration objects to stable, JSON-safe descriptions.

    The descriptor deliberately records behavior-defining configs and callable
    identities, never ``repr`` output containing process-local memory addresses.
    It is used only for compatibility validation; TensorFlow variables remain
    authoritative in the object checkpoint.

    Args:
        value (object): Configuration value or Keras object to describe.
        active_ids (set[int] | None): Active recursion path for cycle handling.
        strip_config_names (bool): Omit generated Keras ``name`` fields.

    Returns:
        object: JSON-safe deterministic representation of ``value``.

    Raises:
        ValueError: If a configurable object cannot return its configuration.
    """

    # Emit ordinary scalar values directly.
    if value is None or isinstance(value, (bool, str, int)):
        return value

    # Canonicalize round trips through float32-backed Keras variables.
    if isinstance(value, float):
        # Keras may materialize the same constructor scalar through float32 on
        # a compiled object (for example 0.9 -> 0.899999976). Seven significant
        # digits preserve float32 semantics while keeping the descriptor stable.
        return float(format(value, ".7g"))

    # Convert NumPy scalar wrappers into their Python scalar counterparts.
    if isinstance(value, np.generic):
        return _recovery_descriptor(
            value.item(), active_ids, strip_config_names
        )

    # Keep array descriptors compact regardless of array size.
    if isinstance(value, np.ndarray):
        return _array_recovery_descriptor(value)

    # Record TensorFlow dtypes by their portable registered name.
    if isinstance(value, tf.dtypes.DType):
        return {"type": "tensorflow.DType", "name": value.name}
    
    # Record symbolic shapes without depending on TensorFlow repr output.
    if isinstance(value, tf.TensorShape):
        return {"type": "tensorflow.TensorShape", "shape": value.as_list()}

    active_ids = set() if active_ids is None else active_ids
    object_id = id(value)
    # Stop only a cycle on the active recursion path, not repeated values.
    if object_id in active_ids:
        return {"type": _qualified_name(value), "cycle": True}

    active_ids.add(object_id)
    try:
        # Preserve mapping semantics while normalizing key order.
        if isinstance(value, dict):
            # Ordinary string-key mappings remain readable in the checkpoint.
            # Use direct JSON objects when every mapping key is a string.
            if all(isinstance(key, str) for key in value):
                return {
                    key: _recovery_descriptor(
                        value[key], 
                        active_ids, 
                        strip_config_names
                    )
                    for key in sorted(value)
                    if not (strip_config_names and key == "name")
                }

            # Layer-routing configs sometimes use integer depth keys.
            entries = [
                {
                    "key": _recovery_descriptor(
                        key, active_ids, strip_config_names
                    ),
                    "value": _recovery_descriptor(
                        item, active_ids, strip_config_names
                    ),
                }
                for key, item in value.items()
            ]
            entries.sort(key=fingerprint_state)

            return {"mapping": entries}

        # Preserve sequence order while normalizing every member.
        if isinstance(value, (list, tuple)):
            return [
                _recovery_descriptor(item, active_ids, strip_config_names)
                for item in value
            ]

        # Sort unordered collections by their already-stable fingerprints.
        if isinstance(value, (set, frozenset)):
            items = [
                _recovery_descriptor(item, active_ids, strip_config_names)
                for item in value
            ]
            items.sort(key=fingerprint_state)

            return {"set": items}

        get_config = getattr(value, "get_config", None)
        # Prefer semantic object configuration to a transient object repr.
        if callable(get_config):
            try:
                config = get_config()
            except Exception as error:
                raise ValueError(
                    f"Cannot fingerprint {_qualified_name(value)} config."
                ) from error
            return {
                "type": _qualified_name(value),
                "config": _recovery_descriptor(
                    config,
                    active_ids,
                    isinstance(value, (
                        tf.keras.Model, 
                        tf.keras.layers.Layer, 
                        tf.keras.optimizers.Optimizer
                    ))
                )
            }

        # Identify plain callables by definition rather than memory address.
        if callable(value):
            return {"callable": _qualified_name(value)}

        # Unknown runtime objects contribute their stable type, not transient
        # instance identity. All supported Keras objects expose get_config().
        return {"type": _qualified_name(value)}
    finally:
        active_ids.remove(object_id)


def _model_topology_descriptor(model: object) -> dict[str, object] | None:
    """Describe a Keras topology without including mutable values.

    Args:
        model (object): Keras-compatible model, or ``None``.

    Returns:
        dict[str, object] | None: Config and ordered variable specifications.
    """

    # Preserve absent optional model branches explicitly.
    if model is None:
        return None

    weights = list(getattr(model, "weights", []) or [])

    return {
        "object": _recovery_descriptor(model),
        "weights": [{
            "shape": weight.shape.as_list(), 
            "dtype": tf.as_dtype(weight.dtype).name, 
            "trainable": bool(getattr(weight, "trainable", False))
        } for weight in weights]
    }


def _trackable_topology_descriptor(
    trackables: dict[str, object]
) -> dict[str, object]:
    """Describe model and optimizer structures before strict restore.

    Args:
        trackables (dict[str, object]): Named TensorFlow checkpoint objects.

    Returns:
        dict[str, object]: Stable config and variable specifications by name.
    """

    result: dict[str, object] = {}
    for name in sorted(trackables):
        value = trackables[name]
        variables_attr = getattr(value, "variables", None)
        variables = variables_attr() if callable(variables_attr) \
                    else list(variables_attr or [])
        result[name] = {
            "object": _recovery_descriptor(value),
            "variables": [{
                "shape": variable.shape.as_list(), 
                "dtype": tf.as_dtype(variable.dtype).name, 
                "trainable": bool(getattr(variable, "trainable", False))
            } for variable in variables]
        }

    return result


def _progressive_depth_specs(fit_kwargs: dict[str, object]) -> list[object]:
    """Resolve persistent depth additions made by one progressive task.

    Args:
        fit_kwargs (dict[str, object]): Validated progressive fit arguments.

    Returns:
        list[object]: Depth specifications in their post-stage application order.

    Raises:
        ValueError: If a requested depth stage lacks its companion value.
    """

    stage_tasks = fit_kwargs.get("stage_tasks")
    depths = fit_kwargs.get("depths")

    # A depth-only shorthand applies each supplied specification in order.
    if stage_tasks == "depths_only":
        return list(depths or [])

    # Timestep- and resolution-only curricula never alter persistent topology.
    if stage_tasks in ("timesteps_only", "resolutions_only"):
        return []

    # Non-progressive or malformed selectors carry no reconstructable growth.
    if not isinstance(stage_tasks, Sequence) or isinstance(stage_tasks, str):
        return []

    resolved = []
    for stage_index, task in enumerate(stage_tasks):
        has_depth = False
        depth_spec = None

        # Resolve a named depth update through the stage-indexed values.
        if task == "depth":
            has_depth = True
        # Prefer a dictionary's inline depth value when supplied.
        elif isinstance(task, dict) and "depth" in task:
            has_depth = True
            depth_spec = task["depth"]
        # A set names simultaneous updates and uses companion value arrays.
        elif isinstance(task, (set, frozenset)) and "depth" in task:
            has_depth = True
        # A two-item depth pair carries its value inline.
        elif isinstance(task, (tuple, list)) and len(task) == 2 \
        and task[0] == "depth":
            has_depth = True
            depth_spec = task[1]

        # Ignore stages that do not request persistent structural growth.
        if not has_depth:
            continue

        # Resolve an omitted inline value from the stage-aligned depth list.
        if depth_spec is None:
            # Fail rather than guessing a topology that might partially restore.
            if depths is None or stage_index >= len(depths):
                raise ValueError(
                    "Cannot reconstruct progressive topology: a depth "
                    "stage has no stage-indexed depth specification."
                )

            depth_spec = depths[stage_index]

        resolved.append(depth_spec)

    return resolved


def _fresh_optimizer(value: object) -> object:
    """Clone a configured optimizer so a run never mutates caller state.

    Args:
        value (object): Optimizer instance or non-optimizer compile value.

    Returns:
        object: Fresh equivalent optimizer, or the unchanged non-optimizer.

    Raises:
        ValueError: If an optimizer cannot round-trip through Keras config.
    """

    # Leave optimizer names/config mappings for Keras to construct normally.
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
    """Read named optimizer iteration counters without mutating either model.

    Args:
        classifier (tf.keras.Model | None): Current classifier model.
        generative_model (tf.keras.Model | None): Optional replay wrapper.

    Returns:
        dict[str, int]: Stable role/attribute names and completed update counts.
    """

    counters = {}
    for role, model in (
        ("classifier", classifier), 
        ("replay", generative_model)
    ):
        # Skip an absent optional role without fabricating a counter.
        if model is None:
            continue

        for attribute in ("optimizer", "gen_optimizer", "clf_optimizer"):
            optimizer = getattr(model, attribute, None)
            iterations = getattr(optimizer, "iterations", None)
            # Ignore model roles that do not own this optimizer variant.
            if iterations is None:
                continue

            try:
                counters[f"{role}_{attribute}"] = int(iterations.numpy())
            except (AttributeError, TypeError, ValueError):
                # Custom optimizers may expose symbolic counters before build.
                continue

    return counters


def _validate_supplied_model_runtime(
    model: tf.keras.Model | None, 
    seed: int | None, 
    dtype_policy: str, 
    role: str
) -> None:
    """Validate a caller-built continual model before it can be mutated.

    A seed cannot retroactively control weights or stochastic-layer
    configuration created before this function was called. Direct continual
    APIs therefore accept seeded prebuilt repository models only when their
    top-level seed (and a wrapped raw network's seed) matches the experiment.
    Config-built models already satisfy this contract.

    Args:
        model (tf.keras.Model | None): Optional caller-built replay model.
        seed (int | None): Effective continual master seed.
        dtype_policy (str): Required Keras numerical policy name.
        role (str): Human-readable model role for validation errors.

    Returns:
        None: Compatible models are left unchanged.

    Raises:
        ValueError: If a seeded experiment receives a repository model built
            with a missing/different seed, or any nested layer has an
            incompatible dtype policy.
    """

    # Preserve classifier-only runs without a replay-model validation target.
    if model is None:
        return

    validate_model_dtype_policy(model, dtype_policy, role=role)
    # An unseeded experiment imposes only the numerical-policy contract.
    if seed is None:
        return

    seeded_components = [(role, model)]
    raw_network = getattr(model, "network", None)
    # Validate a wrapper and its already-built raw architecture independently.
    if isinstance(raw_network, tf.keras.Model):
        seeded_components.append((role + " raw network", raw_network))

    for component_role, component in seeded_components:
        # Custom third-party Keras models may expose no repository seed API.
        if not hasattr(component, "seed"):
            continue

        component_seed = getattr(component, "seed")
        # Initialization cannot be repaired after a seeded model was built.
        if component_seed is None or int(component_seed) != int(seed):
            raise ValueError(
                f"{component_role} was built with seed {component_seed!r}, "
                f"but the continual experiment requires seed {seed}. Build "
                "the model through get_model/config or reconstruct it with "
                "the same seed before starting continual learning."
            )


def _reset_task_random_streams(
    model: tf.keras.Model | None, 
    task_seed: int | None
) -> None:
    """Reset stochastic Keras/model streams at a reproducible task boundary.

    TensorFlow 2.10 deliberately excludes each Keras random layer's private
    generator/counter from ordinary checkpoints. Resetting those counters and
    retracing cached Keras functions at the start of *every* task makes the
    next task depend only on its derived seed, whether the preceding task ran
    in this process or was restored from a committed checkpoint.

    Args:
        model (tf.keras.Model | None): Model tree whose stochastic components
            and cached graph functions should be reset.
        task_seed (int | None): Derived task seed. ``None`` preserves unseeded
            behavior while still clearing cached graph functions.

    Returns:
        None: Random-layer state and model caches are updated in place.
    """

    # Classifier-only continual runs have no persistent generative model tree.
    if model is None:
        return

    components = list(getattr(model, "submodules", ()))
    # Include subclassed models whose submodule enumeration omits the root.
    if not any(component is model for component in components):
        components.insert(0, model)

    visited: set[int] = set()
    for component_index, component in enumerate(components):
        # Reset shared layers exactly once even when multiple paths reach them.
        if id(component) in visited:
            continue

        visited.add(id(component))

        # Force Keras to build fresh stateful-random graph ops for this task.
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

        # Reset the Keras random-layer stream only in explicitly seeded mode.
        if random_generator is not None and component_seed is not None:
            # Legacy Keras random layers increment this private Python counter
            # while tracing; checkpoints intentionally omit it.
            random_generator._seed = component_seed
            generator = getattr(random_generator, "_generator", None)
            # Newer stateful Keras RNG mode stores a Generator instead.
            if generator is not None and hasattr(generator, "reset_from_seed"):
                generator.reset_from_seed(component_seed)

        # Reset repository stochastic layers such as DropPath, which use their
        # public seed directly instead of Keras BaseRandomLayer.
        if component_seed is not None and random_generator is None \
        and hasattr(component, "seed"):
            component.seed = component_seed

    # Wrappers and VAEs use their top-level seed for noising/reparameterization.
    if task_seed is not None and hasattr(model, "seed"):
        model.seed = int(task_seed)
    # VAE latent sampling uses a distinct stream from its generation API.
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
    """Map a supported loader representation to diffusion model space.

    Args:
        x (numpy.ndarray): Preprocessed image batch ``[samples, H, W, C]``.
        data_min (float): Minimum of the shared real training input.
        data_range (float): Nonzero maximum-minus-minimum training range.

    Returns:
        numpy.ndarray: Inputs in the active policy's variable dtype, mapped so
        the training extrema are ``-1`` and ``1``. Held-out values are not
        clipped.
    """

    variable_dtype = tf.keras.mixed_precision.global_policy().variable_dtype
    numpy_dtype = tf.as_dtype(variable_dtype).as_numpy_dtype
    x = np.asarray(x, dtype=numpy_dtype)

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
    onehot_labels: bool
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


def _observed_mean(values: Sequence[float] | np.ndarray) -> float:
    """Return the mean of observed values without all-NaN warnings.

    Args:
        values (Sequence[float] | numpy.ndarray): Numeric values which may
            contain unavailable entries represented by NaN.

    Returns:
        float: Mean of non-NaN entries, or NaN when none are observed.
    """

    array = np.asarray(values, dtype="float64")
    observed = array[~np.isnan(array)]
    # Preserve an unavailable summary when no observation exists.
    if observed.size == 0:
        return float("nan")
    return float(np.mean(observed))


def _observed_max(values: Sequence[float] | np.ndarray) -> float:
    """Return the maximum observed value without all-NaN warnings.

    Args:
        values (Sequence[float] | numpy.ndarray): Numeric values which may
            contain unavailable entries represented by NaN.

    Returns:
        float: Maximum non-NaN entry, or NaN when none are observed.
    """

    array = np.asarray(values, dtype="float64")
    observed = array[~np.isnan(array)]

    # Preserve an unavailable summary when no observation exists.
    if observed.size == 0:
        return float("nan")
    return float(np.max(observed))


def _continual_metrics(
    accuracy_matrix: Sequence[Sequence[float]]
) -> dict[str, float]:
    """Compute task-balanced summary metrics from a continual accuracy matrix.

    Args:
        accuracy_matrix (Sequence[Sequence[float]]): Lower-triangular matrix
            where row ``t`` contains performance on tasks learned through
            ``t`` and unavailable future entries are NaN.

    Returns:
        dict[str, float]: Final average accuracy, average incremental accuracy,
        signed average forgetting, and backward transfer. Forgetting compares
        the best score before the final evaluation with the final score, so
        improvement can produce negative forgetting. Single-task forgetting
        and backward transfer are zero because no prior task exists.
    """

    matrix = np.asarray(accuracy_matrix, dtype="float64")
    task_num = len(matrix)
    # Keep an empty schedule representable for defensive direct callers.
    if task_num == 0:
        return {
            "final_average_accuracy": np.nan, 
            "average_incremental_accuracy": np.nan, 
            "average_forgetting": np.nan, 
            "backward_transfer": np.nan
        }

    row_averages = [_observed_mean(matrix[index, :index + 1])
                    for index in range(task_num)]
    final_average_accuracy = _observed_mean(matrix[-1, :task_num])
    # Forgetting and BWT are defined only over tasks learned before the last.
    if task_num == 1:
        average_forgetting = 0.
        backward_transfer = 0.
    # Compare every earlier task with its learned and best historical scores.
    else:
        final_old = matrix[-1, :task_num - 1]
        maxima = np.asarray([
            _observed_max(matrix[index:task_num - 1, index])
            for index in range(task_num - 1)
        ])
        diagonal = np.asarray([
            matrix[index, index] 
            for index in range(task_num - 1)
        ])
        average_forgetting = _observed_mean(maxima - final_old)
        backward_transfer = _observed_mean(final_old - diagonal)

    return {
        "final_average_accuracy": final_average_accuracy, 
        "average_incremental_accuracy": _observed_mean(row_averages), 
        "average_forgetting": average_forgetting, 
        "backward_transfer": backward_transfer
    }


def _task_accuracy_summaries(
    accuracy_matrix: Sequence[Sequence[float]]
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
        np.nan if index == 0 else _observed_mean(
            accuracy_matrix[index][:index]
        )
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


def _predict_vae_classes(
    model: VAEClassifier, 
    x: np.ndarray
) -> np.ndarray:
    """Predict from a VAE classifier without supplying ground-truth labels.

    Args:
        model (VAEClassifier): Joint model with a raw-input classifier branch.
        x (numpy.ndarray): Samples shaped ``[samples, data_dim]``.

    Returns:
        numpy.ndarray: Class probabilities shaped ``[samples, class_num]``.
    """

    classifier = model.classifier
    # Keras classifiers receive an explicit inference flag; generic callables
    # retain their documented one-argument protocol.
    predictions = classifier(x, training=False) if isinstance(classifier, tf.keras.layers.Layer) \
                else classifier(x)

    return np.asarray(predictions)


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
    """Evaluate timestep-ensemble accuracy separately for learned tasks.

    Per-task evaluation is required for forgetting and backward-transfer
    metrics; a single cumulative accuracy cannot reconstruct those values.

    Args:
        model (DiffusionClassifier): Wrapper owning ensemble evaluation.
        x (numpy.ndarray): Clean preprocessed images for one split.
        y (numpy.ndarray): Sparse or one-hot labels aligned with ``x``.
        learned_groups (Sequence[Sequence[int]]): Learned internal class groups.
        total_group_num (int): Total number of schedule groups/row columns.
        data_min (float): Shared loader-space training minimum.
        data_range (float): Nonzero shared loader-space training range.
        batch_size (int): Evaluation batch size.
        options (dict[str, object]): Ensemble metric keyword options.
        seed (int | None): Derived split/task ensemble-noising seed.
        verbose (bool | int): Whether ensemble evaluation reports progress.

    Returns:
        list[float]: Full-width row with learned-task accuracies and NaNs for
        unavailable future tasks.
    """

    labels = np.asarray(y)
    label_ids = np.argmax(labels, axis=-1) if labels.ndim > 1 and labels.shape[-1] > 1 \
                else labels.reshape(-1)
    row = [np.nan] * total_group_num
    ensemble_options = dict(options)

    # Supply deterministic noising unless the caller selected another seed.
    if ensemble_options.get("seed") is None:
        ensemble_options["seed"] = seed

    for group_index, group in enumerate(learned_groups):
        selected = np.isin(label_ids, group)
        # A limited split can legitimately contain no member of a task.
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


def _flatten_example_rows(values: np.ndarray) -> np.ndarray:
    """Flatten trailing feature axes while retaining an empty row dimension.

    Args:
        values (numpy.ndarray): Example-major array with at least two axes.

    Returns:
        numpy.ndarray: Matrix shaped ``[example_count, feature_count]``.

    Raises:
        ValueError: If ``values`` does not contain a feature axis.
    """

    array = np.asarray(values)
    # A feature matrix needs one leading row axis and at least one feature axis.
    if array.ndim < 2:
        raise ValueError("Continual inputs must include a feature axis.")

    feature_count = int(np.prod(array.shape[1:], dtype=np.int64))

    return array.reshape((len(array), feature_count))


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


def _sample_exact_rows(
    x: np.ndarray, 
    y: np.ndarray, 
    count: int | None, 
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Select an exact seeded exposure count from aligned arrays.

    Sampling is without replacement when enough rows exist and with replacement
    only when an explicitly requested matched exposure exceeds the available
    rows. ``None`` preserves every row and its order.

    Args:
        x (numpy.ndarray): Input rows.
        y (numpy.ndarray): Aligned labels.
        count (int | None): Requested exposure count, or ``None`` for all rows.
        rng (numpy.random.Generator): Local deterministic generator.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Selected aligned arrays.

    Raises:
        TypeError: If ``count`` is neither ``None`` nor a non-boolean integer.
        ValueError: If arrays are misaligned, empty for a positive request, or
            ``count`` is negative.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    # Require one label row for every input row.
    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of rows.")

    # Preserve the legacy complete current-data exposure.
    if count is None:
        return x, y

    # Refuse boolean and fractional exposure counts.
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError("exposure counts must be non-boolean integers or None.")

    count = int(count)
    # Keep explicit exposure counts nonnegative.
    if count < 0:
        raise ValueError("exposure counts must be nonnegative.")

    # A positive exposure cannot be drawn from an empty task.
    if count > 0 and len(x) == 0:
        raise ValueError("cannot sample positive exposure from an empty task.")

    # Return correctly shaped empty views without asking NumPy to sample.
    if count == 0:
        return x[:0], y[:0]

    indices = rng.choice(len(x), size=count, replace=count > len(x))

    return x[indices], y[indices]


def _restore_replay_label_shape(
    label_ids: np.ndarray, 
    reference_labels: np.ndarray
) -> np.ndarray:
    """Represent selected integer IDs like the loader's original labels.

    Args:
        label_ids (numpy.ndarray): One-dimensional selected class IDs.
        reference_labels (numpy.ndarray): Sparse-column, sparse-vector, or
            one-hot labels whose shape/dtype define the result representation.

    Returns:
        numpy.ndarray: Selected labels compatible with ``reference_labels``.
    """

    ids = np.asarray(label_ids, dtype="int64").reshape(-1)
    reference = np.asarray(reference_labels)
    # Recreate one-hot labels at the loader's fixed vocabulary width.
    if reference.ndim == 2 and reference.shape[1] > 1:
        return np.eye(reference.shape[1], dtype=reference.dtype)[ids]

    # Preserve legacy column-shaped sparse labels when loaders return them.
    if reference.ndim > 1:
        return ids[:, None].astype(reference.dtype, copy=False)

    return ids.astype(reference.dtype, copy=False)


def _balanced_generation_labels(
    classes: Sequence[int], 
    count: int, 
    rng: np.random.Generator
) -> np.ndarray:
    """Allocate an exact generated-replay count nearly equally by class.

    Args:
        classes (Sequence[int]): Old zero-based class IDs.
        count (int): Nonnegative total candidate count.
        rng (numpy.random.Generator): Local generator used to randomize remainder
            allocation and candidate order.

    Returns:
        numpy.ndarray: Integer labels with exactly ``count`` rows.

    Raises:
        ValueError: If count is negative or positive without any old class.
    """

    classes = [int(class_id) for class_id in classes]
    # Keep total generation counts within their mathematical domain.
    if count < 0 or (count > 0 and not classes):
        raise ValueError("generation count requires a nonempty class set.")

    # Preserve an explicit zero replay budget.
    if count == 0:
        return np.empty((0,), dtype="int64")

    base, remainder = divmod(int(count), len(classes))
    shuffled = list(np.asarray(classes)[rng.permutation(len(classes))])
    labels = np.concatenate([
        np.repeat(class_id, base + int(index < remainder))
        for index, class_id in enumerate(shuffled)
    ]).astype("int64", copy=False)
    rng.shuffle(labels)

    return labels


def _predict_teacher_probabilities(
    teacher: tf.keras.Model, 
    x: np.ndarray, 
    data_min: float, 
    data_range: float, 
    batch_size: int
) -> np.ndarray:
    """Evaluate one frozen diffusion classifier on replay candidates.

    Args:
        teacher (tf.keras.Model): Frozen raw/EMA classifier network exposing
            ``predict_class``.
        x (numpy.ndarray): Candidate images in loader preprocessing space.
        data_min (float): Loader-space minimum used for diffusion conversion.
        data_range (float): Nonzero loader-space range.
        batch_size (int): Positive inference batch size.

    Returns:
        numpy.ndarray: Teacher probabilities shaped ``[samples, old_classes]``.

    Raises:
        TypeError: If the teacher does not expose ``predict_class``.
    """

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

    # Retain a rank-two empty output for a zero candidate pool.
    if not predictions:
        output_width = int(getattr(teacher, "num_classes", 0) or 0)

        return np.empty((0, output_width), dtype="float32")

    return np.concatenate(predictions, axis=0)


def _cache_digest(value: np.ndarray) -> str:
    """Hash one non-object array with dtype and shape metadata.

    Args:
        value (numpy.ndarray): Array to fingerprint.

    Returns:
        str: Lowercase SHA-256 digest.

    Raises:
        TypeError: If an object-dtype array is supplied.
    """

    array = np.ascontiguousarray(np.asarray(value))
    # Object bytes contain process-specific pointers and cannot be trusted.
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot be used in a replay cache.")

    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))

    # Empty replay pools contribute metadata but have no content bytes.
    if array.size:
        digest.update(memoryview(array).cast("B"))

    return digest.hexdigest()


def _cached_replay_candidates(
    x: np.ndarray, 
    y: np.ndarray, 
    cache_dir: str | None, 
    cache_mode: str, 
    task_index: int, 
    old_classes: Sequence[int], 
    seed: int | None, 
    context_fingerprint: str | None = None
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Read or atomically write one matched replay candidate pool.

    ``write`` refuses to replace an existing pool. ``read_write`` reuses an
    existing pool and otherwise creates it. Pool names deliberately omit the
    experimental condition so raw/EMA and gate treatments can consume exactly
    the same candidates.

    Args:
        x (numpy.ndarray): Newly generated candidate samples.
        y (numpy.ndarray): Candidate labels.
        cache_dir (str | None): Shared cache directory.
        cache_mode (str): ``off``, ``write``, ``read``, or ``read_write``.
        task_index (int): Zero-based continual task index.
        old_classes (Sequence[int]): Classes represented by the pool.
        seed (int | None): Candidate-generation seed recorded in metadata.
        context_fingerprint (str | None): Dataset/generator context shared by
            legitimate treatment variants but different for incompatible runs.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, str | None]: Candidate arrays and
        concrete cache path, or ``None`` when caching is off.

    Raises:
        ValueError: If mode/path/metadata/checksum is invalid.
        FileExistsError: If write-only mode would overwrite a pool.
        FileNotFoundError: If read-only mode cannot find the pool.
    """

    cache_mode = str(cache_mode).lower()
    # Keep cache behavior explicit and typo-safe.
    if cache_mode not in ("off", "write", "read", "read_write"):
        raise ValueError("replay_cache_mode must be off, write, read, or read_write.")
    # Preserve the legacy no-I/O path exactly.
    if cache_mode == "off":
        return np.asarray(x), np.asarray(y), None
    # Every enabled cache mode requires a concrete destination.
    if cache_dir is None or not str(cache_dir).strip():
        raise ValueError("replay_cache_dir is required when cache mode is enabled.")

    path = _replay_cache_path(
        cache_dir, 
        task_index, 
        old_classes, 
        len(y), 
        context_fingerprint=context_fingerprint, 
        create_root=True
    )

    should_read = cache_mode == "read" or (
        cache_mode == "read_write" and path.exists()
    )
    # Load and validate a previously frozen candidate pool.
    if should_read:
        # Read-only treatments must fail loudly when the reference pool is absent.
        if not path.is_file():
            raise FileNotFoundError(f"Replay cache does not exist: {path}")

        with np.load(path, allow_pickle=False) as archive:
            cached_x = archive["x"]
            cached_y = archive["y"]
            metadata = json.loads(str(archive["metadata"].item()))
        expected = {
            "schema_version": 1,
            "task_index": int(task_index),
            "old_classes": [int(class_id) for class_id in old_classes],
            "candidate_count": int(len(y)),
            "seed": seed,
            "context_fingerprint": context_fingerprint,
        }

        # Refuse a cache created for another stochastic stream or replay budget.
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Replay cache metadata differs from this experiment.")
        # Verify archive integrity before a cached pool can enter training.
        if metadata.get("x_sha256") != _cache_digest(cached_x) \
        or metadata.get("y_sha256") != _cache_digest(cached_y):
            raise ValueError("Replay cache checksum validation failed.")

        return cached_x, cached_y, str(path)

    x = np.asarray(x)
    y = np.asarray(y)
    # A retried write may reuse the exact authenticated pool left by an
    # interrupted task, but it must never replace different candidate bytes.
    if cache_mode == "write" and path.exists():
        # Reject directories or other non-archive filesystem objects.
        if not path.is_file():
            raise FileExistsError(f"Replay cache path is not a file: {path}")

        with np.load(path, allow_pickle=False) as archive:
            cached_x = archive["x"]
            cached_y = archive["y"]
            metadata = json.loads(str(archive["metadata"].item()))

        expected = {
            "schema_version": 1, 
            "task_index": int(task_index), 
            "old_classes": [int(class_id) for class_id in old_classes], 
            "candidate_count": int(len(y)), 
            "seed": seed, 
            "context_fingerprint": context_fingerprint
        }
        # Authenticate metadata, stored checksums, and the regenerated pool.
        valid_metadata = all(
            metadata.get(key) == value 
            for key, value in expected.items()
        )
        valid_archive = (
            metadata.get("x_sha256") == _cache_digest(cached_x)
            and metadata.get("y_sha256") == _cache_digest(cached_y)
        )
        same_candidates = (
            _cache_digest(cached_x) == _cache_digest(x)
            and _cache_digest(cached_y) == _cache_digest(y)
        )

        # Refuse a conflicting writer so the common candidate pool is immutable.
        if not (valid_metadata and valid_archive and same_candidates):
            raise FileExistsError(
                "Replay cache already exists with incompatible candidates: "
                f"{path}"
            )

        return cached_x, cached_y, str(path)

    metadata = {
        "schema_version": 1, 
        "task_index": int(task_index), 
        "old_classes": [int(class_id) for class_id in old_classes], 
        "candidate_count": int(len(y)), 
        "seed": seed, 
        "context_fingerprint": context_fingerprint, 
        "x_sha256": _cache_digest(x), 
        "y_sha256": _cache_digest(y)
    }
    temporary = path.with_name(
        "." + path.name + ".tmp-" + uuid.uuid4().hex
    )

    try:
        with open(temporary, "xb") as stream:
            np.savez_compressed(
                stream, 
                x=x, 
                y=y, 
                metadata=np.asarray(json.dumps(metadata, sort_keys=True))
            )
        temporary.replace(path)
    finally:
        # Remove only this invocation's private partial archive after failure.
        if temporary.exists():
            temporary.unlink()

    return x, y, str(path)


def _replay_cache_path(
    cache_dir: str,
    task_index: int,
    old_classes: Sequence[int],
    candidate_count: int,
    context_fingerprint: str | None = None,
    create_root: bool = False,
) -> Path:
    """Return the condition-independent path for one replay candidate pool.

    Args:
        cache_dir (str): Shared candidate-pool directory.
        task_index (int): Zero-based continual task index.
        old_classes (Sequence[int]): Classes represented by the pool.
        candidate_count (int): Exact candidate count.
        context_fingerprint (str | None): Dataset/preprocessing/generator
            namespace. The filename uses its first 16 characters while cache
            metadata stores and verifies the complete fingerprint.
        create_root (bool): Create the cache directory when true.

    Returns:
        pathlib.Path: Stable cache archive path.
    """

    root = Path(cache_dir)
    # Create cache roots only for an actual read/write operation.
    if create_root:
        root.mkdir(parents=True, exist_ok=True)
    class_text = "-".join(str(int(class_id)) for class_id in old_classes)
    context_text = "legacy" if context_fingerprint is None else str(
        context_fingerprint
    )[:16]
    return root / (
        f"context-{context_text}_task-{task_index + 1:04d}_"
        f"classes-{class_text}_"
        f"candidates-{int(candidate_count)}.npz"
    )


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
    """Resolve one optional named baseline into existing continual switches.

    Args:
        baseline (str | None): Named baseline or ``None``.
        generative_model (tf.keras.Model | None): Constructed replay model.
        use_buffer (bool): Existing episodic-replay switch.
        buffer_kwargs (dict[str, object]): Mutable replay-buffer options.
        remove_prev_classes (bool): Existing sequential/cumulative switch.
        use_generative_replay (bool): Existing generated-replay switch.
        use_generative_model_classifier (bool): Attached-classifier switch.
        use_distillation (bool): Teacher-loss switch.

    Returns:
        tuple[str | None, bool, bool, bool, bool, bool, dict[str, object]]:
        Canonical name, remove-previous, buffer, generated-replay,
        attached-classifier, distillation, and buffer options.

    Raises:
        ValueError: If a named baseline is unsupported or its model family is
            incompatible.
    """

    # Preserve every independent legacy option when no ladder cell is selected.
    if baseline is None:
        return (
            None, remove_prev_classes, use_buffer, use_generative_replay,
            use_generative_model_classifier, use_distillation, buffer_kwargs,
        )
    aliases = {
        "sequential": "sequential",
        "sequential_finetuning": "sequential",
        "cumulative": "cumulative",
        "cumulative_upper_bound": "cumulative",
        "reservoir_er": "reservoir_er",
        "lwf": "lwf",
        "vae_replay": "vae_replay",
        "diffusion_replay": "diffusion_replay",
        "joint_none": "joint_none",
        "joint_replay": "joint_replay",
        "joint_kd": "joint_kd",
        "joint_both": "joint_both",
    }
    key = str(baseline).lower()
    # Reject ambiguous research labels before training starts.
    if key not in aliases:
        raise ValueError(f"Unsupported continual baseline: {baseline!r}.")
    baseline = aliases[key]
    buffer_kwargs = dict(buffer_kwargs)

    # Configure the two no-replay real-data controls.
    if baseline in ("sequential", "cumulative"):
        remove_prev_classes = baseline == "sequential"
        use_buffer = False
        use_generative_replay = False
        use_generative_model_classifier = False
        use_distillation = False
        # A named pure classifier control must not hide a configured generator.
        if generative_model is not None:
            raise ValueError(f"{baseline} baseline requires no generative_model.")
    # Configure unbiased episodic replay with the established standalone head.
    elif baseline == "reservoir_er":
        remove_prev_classes = True
        use_buffer = True
        use_generative_replay = False
        use_generative_model_classifier = False
        use_distillation = False
        buffer_kwargs["strategy"] = "reservoir"
        # ER and generative replay are mutually exclusive treatments.
        if generative_model is not None:
            raise ValueError("reservoir_er requires no generative_model.")
    # Configure LwF-style teacher learning without generated rehearsal.
    elif baseline == "lwf":
        remove_prev_classes = True
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
    # Configure VAE replay feeding the separate classifier baseline.
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
    # Configure generator-only diffusion replay with a separate classifier.
    elif baseline == "diffusion_replay":
        use_buffer = False
        use_generative_replay = True
        use_generative_model_classifier = False
        use_distillation = False
        # Any diffusion wrapper can provide images to a separate classifier.
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
    # Configure the four joint-DiT factorial cells.
    else:
        use_buffer = False
        use_generative_model_classifier = True
        use_generative_replay = baseline in ("joint_replay", "joint_both")
        use_distillation = baseline in ("joint_kd", "joint_both")
        # Joint cells require a diffusion classifier rather than a generator-only model.
        if not isinstance(generative_model, (
            DiffusionClassifier, DiTClassifier,
            DiTEncoderDecoderClassifier, UNetClassifier,
        )):
            raise ValueError(f"{baseline} requires a DiffusionClassifier.")
    return (
        baseline, remove_prev_classes, use_buffer, use_generative_replay,
        use_generative_model_classifier, use_distillation, buffer_kwargs,
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
    teacher_network_name: str | None = None,
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
    dataset_seed: int | None = None, 
    seed: int | None = None,
    dtype_policy: str | None = None,
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
            per task. ``None`` partitions the order by ``task_size`` (one
            class per task by default). When
            both schedule arguments are supplied, flattened groups must match
            ``class_order`` exactly.
        task_size (int): Positive classes per automatically constructed task.
        class_order_mode (str): ``"fixed"`` or seeded ``"random"`` class order.
        task_order_mode (str): ``"fixed"`` or seeded ``"random"`` whole-task
            order, preserving the order inside each group.
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
        remove_prev_classes (bool): If true, training receives the initial task
            group first and only each newly introduced group thereafter;
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
            "seed": None}``. ``strategy`` optionally selects ``"fifo"``
            (the unchanged default), ``"reservoir"``, or ``"class_balanced"``.
            ``maxlen`` is capacity; ``sample_num`` is
            the maximum prior pairs concatenated before a task; ``insert_num``
            is the number sampled from that task's augmented training arrays
            after fitting; and ``seed`` initializes the buffer's private random
            generator without changing global random state. Extra keys are
            retained but unused. Example: ``{"maxlen": 5000,
            "sample_num": 500, "insert_num": 500, "seed": 42}``.
        baseline (str | None): Optional named baseline ladder cell: sequential,
            cumulative, reservoir_er, lwf, vae_replay, diffusion_replay, or one
            of joint_none/joint_replay/joint_kd/joint_both. ``None`` preserves
            the independent legacy controls.
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
        use_generative_replay (bool): Generate old examples after task one.
            ``True`` preserves historical behavior when a generator is present;
            ``False`` supports joint/KD controls without generated rehearsal.
        replay_budget_mode (str): ``"legacy"`` retains historical buffer/per-class
            counts; ``"fixed_total"`` enforces explicit total old and optional
            current exposure counts for budget-matched replay comparisons.
        replay_old_examples (int | None): Exact old-example count per later task
            in fixed-total mode.
        replay_current_examples (int | None): Exact current-example exposure in
            fixed-total mode; ``None`` keeps all current rows.
        optimizer_steps_per_epoch (int | None): Optional positive number of
            optimizer updates for every active training phase in each epoch.
            Its default ``None`` preserves finite-dataset Keras behavior. When
            set, the already-selected task pool repeats as needed; this makes
            update budgets comparable without changing old/current pool sizes.
        replay_candidate_multiplier (int): Positive multiplier producing one
            common candidate pool before optional gate selection.
        replay_selection (str): ``all``, ``uniform``, ``random``, ``confidence``,
            ``surprise``, or ``confidence_surprise`` candidate selection.
        replay_surprise_weight (float): Combined-gate surprise weight in
            ``[0, 1]``.
        replay_cache_dir (str | None): Optional shared candidate-pool directory.
        replay_cache_mode (str): ``off``, ``write``, ``read``, or ``read_write``.
        mechanistic_metrics (bool): Record teacher calibration and replay
            consistency, coverage, diversity, drift, allocation, and resources.
        mechanistic_max_samples (int): Positive diversity/drift reporting cap.
        use_generative_model_classifier (bool): Use the classifier attached to
            a ``VAEClassifier`` or the classifier branch of a
            ``DiffusionClassifier`` as the continually learned model. A VAE
            classifier keeps its fixed-width head; a diffusion classifier adds
            one output for each newly observed label. A joint-only VAE task
            reports raw-input classifier accuracy without exposing target
            labels; a separately trained classifier uses the same protocol.
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
        snapshot_network_name (str): ``"raw"`` or ``"ema"`` student branch
            cloned as the next task's teacher. EMA selection fails when EMA is
            disabled so an intended ablation cannot silently use raw weights.
        teacher_network_name (str | None): Optional alias for
            ``snapshot_network_name``. An explicit alias overrides the legacy
            field; ``None`` preserves it.
        use_ensemble_accuracy (bool): Make per-task timestep-ensemble values
            authoritative for the test/validation matrices and derived CL
            metrics. It also enables ensemble evaluation.
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
        generative_callbacks_list (Sequence[tf.keras.callbacks.Callback] |
            None): Callbacks appended only to diffusion/VAE generative fit
            phases. This keeps sampling callbacks away from incompatible
            standalone classifier fits.
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
        seed (int | None): Canonical continual master seed. ``dataset_seed`` is
            retained as an equal-valued backward-compatible alias.
        dtype_policy (str | None): Keras global floating-point policy. ``None``
            preserves the policy already installed by the caller.
        deterministic_ops (bool): Request deterministic TensorFlow kernels.
        initial_classifier (tf.keras.Model | None): Optional configured
            classifier whose trunk and visible head columns initialize tasks.
        callback_patience (int | None): Internal configured early-stopping
            patience. ``None`` preserves direct mode's legacy value of 5;
            ``0`` disables early stopping.
        callback_monitor (str | None): Internal configured metric override.
        callback_monitor_mode (str | None): Internal configured Keras monitor
            direction. ``None`` preserves each phase's legacy direction.
        save_task_checkpoints (bool): Save an immutable checkpoint after every
            completely trained, evaluated, and recorded task.
        checkpoint_dir (str | None): Task-checkpoint output root. ``None``
            disables writes when no configured orchestration root is supplied.
        resume_from (str | None): Checkpoint root or committed task directory
            to restore before continuing with the next unfinished task.
        experiment_phase (str): ``legacy`` retains test evaluation;
            ``development`` disables every test-set evaluation and reports
            validation-only outcomes; ``confirmation`` enables frozen final
            test reporting.
        experiment_manifest_path (str | None): Frozen confirmation manifest.
        experiment_manifest_hash (str | None): Trusted external manifest hash.
        experiment_run_id (str | None): Planned run whose schedule and seed
            must match this invocation before test data can be evaluated.
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

    # ``seed`` is the canonical continual seed; retain ``dataset_seed`` as a
    # backward-compatible direct-API alias.
    if seed is not None and dataset_seed is not None and seed != dataset_seed:
        raise ValueError("seed and dataset_seed must match when both are set.")
    seed = dataset_seed if seed is None else seed
    dataset_seed = seed

    # A committed checkpoint owns the already-materialized stochastic
    # schedule. Reading it before schedule construction avoids changing class
    # order merely because NumPy's permutation implementation changes between
    # supported environments. Explicit caller schedules must still agree.
    if resume_from is not None:
        schedule_checkpoint = load_task_checkpoint(resume_from)
        saved_order = list(schedule_checkpoint.class_order)
        saved_groups = [list(group) for group in schedule_checkpoint.task_groups]
        # Keep the caller's requested class count compatible with saved data.
        if int(class_num) != len(saved_order):
            raise ValueError(
                "Requested class_num differs from the checkpoint schedule."
            )
        # Compare an explicit order only when no stochastic mode transforms it.
        if class_order is not None \
        and class_order_mode == "fixed" and task_order_mode == "fixed" \
        and fingerprint_state(
            list(class_order)
        ) != fingerprint_state(saved_order):
            raise ValueError(
                "Requested class_order differs from the checkpoint schedule."
            )
        # Compare explicit groups when whole-task ordering itself stays fixed.
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
            seed=seed,
        )
    authenticated_manifest_hash = None
    # Test evaluation is reachable only through a frozen externally verified
    # confirmation manifest and one exact planned stream/run identity.
    if str(experiment_phase).lower() == "confirmation":
        # Require all three parts of the externally authenticated run identity.
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
        # Split/order/initialization blocks share one authoritative stream seed.
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
        "continual replay model",
    )
    # Validate caller-provided classifier artifacts before copying their state.
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

        phase_fit_kwargs = dict(fit_kwargs or {})
        phase_trainset = trainset
        # Repeat a finite task dataset only for the explicit fixed-update
        # protocol; VAE's array-based ``train`` method owns its own repetition.
        if optimizer_steps_per_epoch is not None:
            phase_fit_kwargs["steps_per_epoch"] = optimizer_steps_per_epoch
            # Only dataset inputs need orchestration-level repetition.
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


    def report_task_model(
        history: dict[str, list[float]], 
        model: tf.keras.Model, 
        trainset: object, 
        evaluation_set: object,
        split_name: str = "testset",
    ) -> dict[str, object]:
        """Report one continual phase through the shared reporting API.

        Args:
            history (dict[str, list[float]]): Phase metric history.
            model (tf.keras.Model): Trained phase model.
            trainset (object): Phase training input.
            evaluation_set (object): Phase evaluation input.
            split_name (str): Protocol role, ``"testset"`` or ``"valset"``.

        Returns:
            dict[str, object]: Evaluation values and report metadata.
        """

        # Validate the protocol name before evaluating any model or dataset.
        if split_name not in ("testset", "valset"):
            raise ValueError("split_name must be 'testset' or 'valset'.")
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


    def phase_callbacks(
        default_monitor: str, 
        default_mode: str = "max", 
        legacy_patience: int = 0,
        include_generative: bool = False,
    ) -> list[tf.keras.callbacks.Callback]:
        """Build phase-specific early stopping plus caller callbacks.

        Args:
            default_monitor (str): Metric used without an explicit override.
            default_mode (str): Keras monitor direction.
            legacy_patience (int): Direct-mode fallback patience.
            include_generative (bool): Append callbacks reserved for a
                compatible generative-model fit.

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
        # Sampling callbacks must never be attached to standalone classifiers.
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
    # Isolate a caller-owned optimizer from task and retry mutations.
    if "optimizer" in compile_args:
        compile_args["optimizer"] = _fresh_optimizer(
            compile_args["optimizer"]
        )
    ensemble_accuracy_kwargs = dict(ensemble_accuracy_kwargs or {})
    # The optional, more readable teacher alias overrides the legacy name.
    if teacher_network_name is not None:
        snapshot_network_name = teacher_network_name
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
        buffer_kwargs,
    ) = _resolve_baseline_controls(
        baseline,
        generative_model,
        use_buffer,
        buffer_kwargs,
        remove_prev_classes,
        use_generative_replay,
        use_generative_model_classifier,
        use_distillation,
    )

    experiment_phase = str(experiment_phase).lower()
    # Restrict evaluation behavior to the three documented protocol roles.
    if experiment_phase not in ("legacy", "development", "confirmation"):
        raise ValueError(
            "experiment_phase must be 'legacy', 'development', or "
            "'confirmation'."
        )
    replay_budget_mode = str(replay_budget_mode).lower()
    # Keep replay accounting under a known per-class or fixed-total contract.
    if replay_budget_mode not in ("legacy", "fixed_total"):
        raise ValueError(
            "replay_budget_mode must be 'legacy' or 'fixed_total'."
        )
    for name, value in (
        ("replay_old_examples", replay_old_examples),
        ("replay_current_examples", replay_current_examples),
    ):
        # Exact exposure controls accept only nonnegative integer counts.
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
        ):
            raise ValueError(f"{name} must be a nonnegative integer or None.")
    # Fixed update budgets are opt-in and must define at least one batch.
    if optimizer_steps_per_epoch is not None and (
        isinstance(optimizer_steps_per_epoch, bool)
        or not isinstance(optimizer_steps_per_epoch, Integral)
        or optimizer_steps_per_epoch <= 0
    ):
        raise ValueError(
            "optimizer_steps_per_epoch must be a positive integer or None."
        )
    # Normalize the validated optional update budget once for every task.
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
        raise ValueError(
            "fixed_total replay requires replay_old_examples to be set."
        )
    # Candidate pools must be a positive multiple of the selected replay set.
    if isinstance(replay_candidate_multiplier, bool) \
    or not isinstance(replay_candidate_multiplier, Integral) \
    or replay_candidate_multiplier <= 0:
        raise ValueError("replay_candidate_multiplier must be a positive integer.")
    replay_candidate_multiplier = int(replay_candidate_multiplier)
    replay_selection = str(replay_selection).lower()
    replay_selection_names = {
        "all", "uniform", "random", "confidence", "surprise",
        "confidence_surprise",
    }
    # Refuse unsupported gates rather than silently selecting another treatment.
    if replay_selection not in replay_selection_names:
        raise ValueError(
            "replay_selection must be one of "
            f"{sorted(replay_selection_names)}."
        )
    # The combined cognitive gate uses a convex surprise/confidence weight.
    if isinstance(replay_surprise_weight, bool) \
    or not isinstance(replay_surprise_weight, Real) \
    or not 0. <= float(replay_surprise_weight) <= 1.:
        raise ValueError("replay_surprise_weight must be a number in [0, 1].")
    replay_surprise_weight = float(replay_surprise_weight)
    # Accept PyYAML 1.1's boolean representation of an unquoted ``off`` token.
    replay_cache_mode = "off" if replay_cache_mode is False \
        else str(replay_cache_mode).lower()
    # Limit candidate cache behavior to explicit immutable/read-through modes.
    if replay_cache_mode not in ("off", "write", "read", "read_write"):
        raise ValueError(
            "replay_cache_mode must be 'off', 'write', 'read', or 'read_write'."
        )
    # Every enabled cache policy requires a concrete shared directory.
    if replay_cache_mode != "off" and not replay_cache_dir:
        raise ValueError(
            "replay_cache_dir is required when replay_cache_mode is enabled."
        )
    # Bound quadratic diversity/representation diagnostics by a positive cap.
    if isinstance(mechanistic_max_samples, bool) \
    or not isinstance(mechanistic_max_samples, Integral) \
    or mechanistic_max_samples <= 0:
        raise ValueError("mechanistic_max_samples must be a positive integer.")
    mechanistic_max_samples = int(mechanistic_max_samples)
    # Keep real-memory and generated-replay sources mutually exclusive.
    if use_buffer and generative_model is not None:
        raise ValueError(
            "The replay buffer and a generative model cannot be used together."
        )
    # A positive old-data exposure needs an enabled source that can supply it.
    if replay_budget_mode == "fixed_total" \
    and replay_old_examples > 0 \
    and not (use_buffer or (
        generative_model is not None and use_generative_replay
    )):
        raise ValueError(
            "A positive fixed old-example budget requires buffer or "
            "generative replay."
        )

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
            seed=derive_seed(seed, "replay_buffer")
            if seed is not None else buffer_kwargs["seed"],
            strategy=buffer_kwargs["strategy"],
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
            test_steps=min(50, generative_model.timesteps),
            seed=seed,
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
            test_steps=min(50, generative_model.timesteps),
            seed=seed,
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
    scored_replay_selection = replay_selection in {
        "confidence", "surprise", "confidence_surprise",
    }
    # Confidence/surprise scores require a frozen diffusion classifier trace.
    if scored_replay_selection and not isinstance(
        generative_model, DiffusionClassifier
    ):
        raise ValueError(
            "confidence/surprise replay selection requires a "
            "DiffusionClassifier teacher."
        )
    # Candidate archives are meaningful only when old images are generated.
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
    # Replay-only KD is undefined when the continual treatment has no source
    # of replay rows; fail instead of reporting an identically zero objective.
    if use_distillation \
    and generative_model.distil_scope == "replay_only" \
    and not use_generative_replay:
        raise ValueError(
            "distil_scope='replay_only' requires use_generative_replay=True."
        )
    # V2 has a separate discriminator pipeline that does not consume replay
    # provenance. Keep this unsupported scope explicit; its other scopes work.
    if use_distillation \
    and isinstance(generative_model, DiffusionClassifierV2) \
    and generative_model.distil_scope == "replay_only":
        raise ValueError(
            "DiffusionClassifierV2 does not support replay_only distillation; "
            "use old_classes or current_and_replay."
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
            verbose=0,
            seed=seed,
        )

    train_direct_classifier = not uses_attached_classifier or (
        train_classifier_separately and not use_diffusion_classifier
    )
    score_from_generator = use_diffusion_classifier or (
        uses_attached_classifier and not train_classifier_separately
    )

    def recovery_trackables() -> dict[str, object]:
        """Return stable names for every mutable TensorFlow object.

        Returns:
            dict[str, object]: Models, teachers, and optimizers saved together.
        """

        objects: dict[str, object] = {"classifier": prev_model}
        classifier_optimizer = getattr(prev_model, "optimizer", None)
        # Save a standalone or attached classifier optimizer explicitly.
        if classifier_optimizer is not None:
            objects["classifier_optimizer"] = classifier_optimizer
        # Save the complete replay wrapper and its phase-specific optimizers.
        if generative_model is not None:
            objects["replay_model"] = generative_model
            replay_optimizer = getattr(generative_model, "optimizer", None)
            # Preserve the ordinary compiled replay optimizer when available.
            if replay_optimizer is not None:
                objects["replay_optimizer"] = replay_optimizer
            generator_optimizer = getattr(
                generative_model, "gen_optimizer", None
            )
            classifier_phase_optimizer = getattr(
                generative_model, "clf_optimizer", None
            )
            # Preserve V2's independent generator optimizer.
            if generator_optimizer is not None:
                objects["generator_optimizer"] = generator_optimizer
            # Preserve V2's independent classifier optimizer.
            if classifier_phase_optimizer is not None:
                objects["classifier_phase_optimizer"] = (
                    classifier_phase_optimizer
            )
            teacher = getattr(generative_model, "teacher_network", None)
            # Save the frozen next-task teacher outside wrapper tracking.
            if teacher is not None:
                objects["teacher"] = teacher
        return objects


    def prepare_optimizer_slots() -> None:
        """Create optimizer variables before strict object restoration.

        Returns:
            None: Every available optimizer is registered with its variables.
        """

        optimizer_variables = []
        classifier_optimizer = getattr(prev_model, "optimizer", None)
        # Register classifier slots against the reconstructed classifier width.
        if classifier_optimizer is not None:
            optimizer_variables.append(
                (classifier_optimizer, prev_model.trainable_variables)
            )
        # Let diffusion wrappers register dynamic network variables first.
        if isinstance(generative_model, DiffusionModel):
            # The wrapper knows how to register dynamic raw-network variables.
            generative_model._register_optimizer_variables()
            # V2 owns two distinct variable partitions and optimizer states.
            if isinstance(generative_model, DiffusionClassifierV2):
                optimizer_variables.extend([
                    (
                        generative_model.gen_optimizer,
                        generative_model.gen_trainable_variables or [],
                    ),
                    (
                        generative_model.clf_optimizer,
                        generative_model.clf_trainable_variables or [],
                    ),
                ])
        # Register the single optimizer used by VAE replay models.
        elif generative_model is not None:
            replay_optimizer = getattr(generative_model, "optimizer", None)
            # Skip uncompiled replay models without optimizer state.
            if replay_optimizer is not None:
                optimizer_variables.append(
                    (replay_optimizer, generative_model.trainable_variables)
                )

        prepared = set()
        for optimizer, variables in optimizer_variables:
            # Avoid duplicate slot creation for aliased optimizer attributes.
            if optimizer is None or id(optimizer) in prepared:
                continue
            prepared.add(id(optimizer))
            variables = list(variables)
            # Use the TensorFlow 2.10 legacy optimizer registration API.
            if hasattr(optimizer, "_create_all_weights"):
                optimizer._create_all_weights(variables)
            # Fall back to the modern optimizer registration API.
            elif hasattr(optimizer, "build"):
                optimizer.build(variables)


    # Load the shared arrays once. Their local generator state is checkpointed
    # after every task and restored before the first resumed task.
    dataset_arrays, rng = _load_continual_arrays(
        load_dataset_fn,
        class_num,
        class_order,
        return_features,
        load_dataset_fn_kwargs,
        max_train_samples,
        max_val_samples,
        pad,
        dataset_seed,
    )
    # Development selection requires its own nonempty held-out partition.
    if experiment_phase == "development" and (
        not use_valset or dataset_arrays[2] is None or len(dataset_arrays[2]) == 0
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
            np.asarray(dataset_arrays[5])[:0].copy(),
        )
    initial_generator_weight_descriptor = _model_weight_descriptor(
        generative_model
    )
    replay_cache_context = fingerprint_state({
        "loader": _recovery_descriptor(load_dataset_fn),
        "loader_kwargs": _recovery_descriptor(load_dataset_fn_kwargs),
        "train_inputs": _array_recovery_descriptor(dataset_arrays[0]),
        "train_labels": _array_recovery_descriptor(dataset_arrays[1]),
        "generator_topology": _model_topology_descriptor(generative_model),
        "generator_initial_weights": initial_generator_weight_descriptor,
        "dtype_policy": dtype_policy,
    })

    # Seal every behavior-defining input used after a task boundary. Paths and
    # presentation-only switches are excluded, while model artifacts, configs,
    # data bytes, callbacks, schedules, precision, and replay policy are all
    # represented without process-local repr strings.
    dataset_names = (
        "x_train", "y_train", "x_val", "y_val", "x_test", "y_test"
    )
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
            "loader": _recovery_descriptor(load_dataset_fn),
            "loader_kwargs": _recovery_descriptor(load_dataset_fn_kwargs),
            "return_features": bool(return_features),
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
            "shuffle_buffer": shuffle_buffer,
            "pad": pad,
            "use_valset": bool(use_valset),
            "arrays": {
                name: _array_recovery_descriptor(array)
                for name, array in zip(dataset_names, dataset_arrays)
            },
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
            "replay": _model_topology_descriptor(generative_model),
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

    acc_list = []
    ensemble_acc_list = []
    histories = []
    generative_histories = []
    classifier_evaluations_list = []
    generative_evaluations_list = []
    ordinary_accuracy_matrix = []
    validation_accuracy_matrix = []
    ensemble_accuracy_matrix = []
    validation_ensemble_accuracy_matrix = []
    task_seeds = []
    task_resource_metrics = []
    task_mechanistic_metrics = []
    previous_replay_samples = None
    previous_replay_labels = None
    checkpoint_paths = []
    start_task_index = 0

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
        # Require the human-inspectable descriptor to authenticate its hash.
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
        completed_classes = [
            label for group in completed_groups for label in group
        ]
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
            # Refresh the attached classifier reference after class growth.
            if use_diffusion_classifier:
                prev_model = generative_model.network.classifier
        # Rebuild a standalone head at the completed visible class width.
        elif not uses_attached_classifier:
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
                    seed=derive_seed(seed, "task", completed_index),
                )
                # A caller-supplied optimizer instance is intentionally shared
                # across task heads within one run. Recreate each historical
                # slot group so its saved object graph can be consumed exactly.
                prepare_optimizer_slots()

        # Persisted teachers represent the completed student and therefore
        # have the same topology as the restored model.
        # Recreate its object graph before loading the saved teacher weights.
        if use_distillation and start_task_index > 0:
            restored_teacher = generative_model.snapshot_teacher_network(
                network_name=snapshot_network_name
            )
            generative_model.set_teacher_network(restored_teacher)

        prepare_optimizer_slots()
        reconstructed_trackables = recovery_trackables()
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
            assert_consumed=True,
        )
        restore_rng_state(recovered.rng_state, numpy_generator=rng)
        # Restore bounded replay contents and its independent Python RNG.
        if use_buffer and recovered.replay_state is not None:
            restore_replay_buffer(buffer, recovered.replay_state)

        acc_list = list(saved.get("accuracies", []))
        ensemble_acc_list = list(saved.get("ensemble_accuracies", []))
        histories = list(saved.get("histories", []))
        generative_histories = list(saved.get("generative_histories", []))
        classifier_evaluations_list = list(
            saved.get("classifier_evaluations", [])
        )
        generative_evaluations_list = list(
            saved.get("generative_evaluations", [])
        )
        ordinary_accuracy_matrix = list(
            saved.get("ordinary_accuracy_matrix", [])
        )
        validation_accuracy_matrix = list(
            saved.get("validation_accuracy_matrix", [])
        )
        ensemble_accuracy_matrix = list(
            saved.get("ensemble_accuracy_matrix", [])
        )
        validation_ensemble_accuracy_matrix = list(
            saved.get("validation_ensemble_accuracy_matrix", [])
        )
        task_seeds = list(saved.get("task_seeds", []))
        task_resource_metrics = list(
            saved.get("task_resource_metrics", [])
        )
        task_mechanistic_metrics = list(
            saved.get("task_mechanistic_metrics", [])
        )
        previous_replay_samples = saved.get("previous_replay_samples")
        previous_replay_labels = saved.get("previous_replay_labels")
        checkpoint_paths.append(str(recovered.task_dir))

    # Continue checkpoint commits beside the supplied recovery boundary.
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
                "cache_path": None,
            },
            "seconds": {
                "generator_sampling": 0.,
                "teacher_scoring": 0.,
                "classifier_fit": 0.,
                "generator_fit": 0.,
            },
        }
        task_mechanistic = {}
        new_classes = internal_task_groups[task_index]
        task_seed = derive_seed(seed, "task", task_index)
        task_seeds.append(task_seed)
        for callback_index, callback in enumerate(
            generative_callbacks_list or ()
        ):
            prefix_setter = getattr(callback, "set_artifact_prefix", None)
            # Namespace continual image/GIF files by task and original classes.
            if callable(prefix_setter):
                class_text = "-".join(
                    str(label) for label in original_task_groups[task_index]
                )
                prefix_setter(
                    f"task-{task_index + 1}_classes-{class_text}"
                )
            # Give sampling/report callbacks a task-isolated reproducible stream.
            if hasattr(callback, "seed"):
                callback.seed = derive_seed(
                    task_seed,
                    "generative_callback",
                    callback_index,
                    _qualified_name(callback),
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
        previous_teacher = None
        # Snapshot only after at least one complete student has been learned.
        if needs_snapshot_teacher and task_index > 0:
            previous_teacher = generative_model.snapshot_teacher_network(
                network_name=snapshot_network_name
            )
            # Only KD installs the snapshot as a train-time teacher. Scoring
            # and diagnostics otherwise keep it as a read-only local model.
            # Select the train-time branch only when an objective consumes it.
            if use_distillation:
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
                verbose=0,
                seed=task_seed,
            )
            _reset_task_random_streams(new_model, task_seed)

            # Seed a fresh task head from the configured initial classifier.
            if initial_classifier is not None and (
                task_index == 0 or not keep_same_model
            ):
                _copy_classifier_prefix(initial_classifier, new_model)

            # Carry learned trunk weights and visible head columns forward.
            elif keep_same_model:
                copy_model(prev_model, new_model)

        optimizer_iterations_before = _optimizer_iteration_metrics(
            new_model, generative_model
        )

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
        task_resource["current_examples_available"] = int(len(x_train))
        # Fixed-total designs match current-data exposure across treatments.
        if replay_budget_mode == "fixed_total":
            x_train, y_train = _sample_exact_rows(
                x_train,
                y_train,
                replay_current_examples,
                np.random.default_rng(derive_seed(
                    task_seed, "current_exposure"
                )),
            )
        task_resource["current_examples_exposed"] = int(len(x_train))
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
            x_train = _flatten_example_rows(x_train)
            x_test = _flatten_example_rows(x_test)
            # Flatten matching validation inputs when present.
            if x_val is not None:
                x_val = _flatten_example_rows(x_val)

        diffusion_data_min = float(np.min(all_x_train))
        diffusion_data_range = float(
            np.max(all_x_train) - diffusion_data_min
        )
        # Keep constant-valued tasks numerically valid.
        if diffusion_data_range == 0.:
            diffusion_data_range = 1.

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

        # Add real examples retained from earlier tasks.
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

        # Generate old examples only when the explicit replay switch is on.
        elif use_generative_replay and generative_model is not None \
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
            # Reuse an authenticated common pool when this treatment is a reader.
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
            # Otherwise create the candidate pool under the configured generator.
            else:
                # VAE generation exposes a per-class API; reduce only the
                # opt-in non-divisible fixed-total case to an exact pool.
                # Preserve a correctly shaped pool for an explicit zero budget.
                if candidate_count == 0:
                    x_buffer = x_train[:0]
                    y_buffer = expected_candidate_y
                # Generate conditional VAE candidates through its per-class API.
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
                    x_buffer = generative_model.sample(
                        network_name=generative_model.test_network_name,
                        labels=y_buffer_ids + int(generative_model.use_cfg),
                        seed=candidate_seed,
                    ).numpy() if len(y_buffer_ids) else x_train[:0]
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
            task_resource["seconds"]["generator_sampling"] = float(
                time.perf_counter() - sample_started
            )
            task_resource["replay"]["cache_path"] = cache_path
            candidate_ids = np.argmax(y_buffer, axis=-1) \
                if np.asarray(y_buffer).ndim == 2 \
                and np.asarray(y_buffer).shape[1] > 1 \
                else np.asarray(y_buffer).reshape(-1)

            # Compute cognitive gate scores once for the shared candidate pool.
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

        # Append selected replay and preserve a row-level source indicator.
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

        # Reshape generated image replay for a standalone image classifier.
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
            seed=task_seed
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
        ) if experiment_phase != "development" else None

        history = {}
        # Attach standalone-classifier callbacks only when that fit phase runs.
        if train_direct_classifier:
            task_callbacks = phase_callbacks(
                "val_accuracy" if valset is not None else "accuracy", 
                legacy_patience=5,
            )

            fit_started = time.perf_counter()
            history = train_task_model(
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
            # Preserve sampled insertion for manually selected buffer policies.
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
        # Train the joint VAE/classifier through the shared API.
        if isinstance(generative_model, VAEClassifier):
            fit_started = time.perf_counter()
            generative_history = train_task_model(
                generative_model, 
                x_train, 
                (x_val, y_val) if x_val is not None else None, 
                task_callbacks=phase_callbacks(
                    "val_clf_accuracy" if x_val is not None \
                        else "clf_accuracy",
                    include_generative=True,
                ), 
                fit_method="train", 
                fit_kwargs={
                    "y": y_train, 
                    "train_num": phase_train_num,
                    "batch_size": batch_size, 
                    "shuffle_buffer": task_shuffle_buffer, 
                    "seed": task_seed
                }
            )
            task_resource["seconds"]["generator_fit"] += float(
                time.perf_counter() - fit_started
            )
            generative_trainset = get_dataset(
                x_train, y_train, 
                shuffle_buffer=task_shuffle_buffer, 
                batch_size=batch_size, 
                drop_remainder=False, 
                seed=task_seed
            )
            generative_valset = get_dataset(
                x_val, y_val,
                shuffle_buffer=0,
                batch_size=batch_size,
                drop_remainder=False,
            ) if x_val is not None else None
            generative_testset = get_dataset(
                x_test, y_test, 
                shuffle_buffer=0, 
                batch_size=batch_size, 
                drop_remainder=False
            ) if experiment_phase != "development" else None
        # Train a conditional replay VAE through the shared API.
        elif isinstance(generative_model, VariationalAutoencoder):
            fit_started = time.perf_counter()
            generative_history = train_task_model(
                generative_model, 
                x_train, 
                (x_val, y_val) if x_val is not None else None, 
                task_callbacks=phase_callbacks(
                    "val_loss" if x_val is not None else "loss", 
                    default_mode="min", 
                    include_generative=True,
                ), 
                fit_method="train", 
                fit_kwargs={
                    "y": y_train, 
                    "train_num": phase_train_num,
                    "batch_size": batch_size, 
                    "clf": new_model, 
                    "shuffle_buffer": task_shuffle_buffer, 
                    "seed": task_seed
                }
            )
            task_resource["seconds"]["generator_fit"] += float(
                time.perf_counter() - fit_started
            )
            generative_trainset = get_dataset(
                x_train, y_train, 
                shuffle_buffer=task_shuffle_buffer, 
                batch_size=batch_size, 
                drop_remainder=False, 
                seed=task_seed
            )
            generative_valset = get_dataset(
                x_val, y_val,
                shuffle_buffer=0,
                batch_size=batch_size,
                drop_remainder=False,
            ) if x_val is not None else None
            generative_testset = get_dataset(
                x_test, y_test, 
                shuffle_buffer=0, 
                batch_size=batch_size, 
                drop_remainder=False
            ) if experiment_phase != "development" else None
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
            diffusion_classifier_replay_mask = replay_mask
            replay_only_distillation = isinstance(
                generative_model, DiffusionClassifier
            ) and generative_model.distil_scope == "replay_only"
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

            generative_trainset = get_dataset(
                generative_x, 
                generative_y, 
                shuffle_buffer=len(generative_x) if shuffle_buffer is None else shuffle_buffer,
                batch_size=batch_size, 
                drop_remainder=False, 
                seed=task_seed,
                metadata=generative_replay_mask
                if replay_only_distillation else None,
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
                drop_remainder=False,
                metadata=np.zeros((len(x_val),), dtype=bool)
                if replay_only_distillation else None,
            ) if x_val is not None else None
            # Locked-test diffusion preprocessing is confirmation-only.
            if experiment_phase != "development":
                generative_y_test = np.argmax(
                    y_test, axis=-1
                ) if load_dataset_fn_kwargs["onehot_labels"] else np.asarray(
                    y_test
                ).reshape(-1)
                generative_testset = get_dataset(
                    _prepare_diffusion_x(
                        x_test,
                        diffusion_data_min,
                        diffusion_data_range
                    ),
                    generative_y_test,
                    shuffle_buffer=0,
                    batch_size=batch_size,
                    drop_remainder=False,
                    metadata=np.zeros((len(x_test),), dtype=bool)
                    if replay_only_distillation else None,
                )

            # Select V2's phase-aware generator entry point.
            if isinstance(generative_model, DiffusionClassifierV2):
                generative_fit_method = "fit_generator_progressively" \
                    if fit_method == "fit_progressively" \
                    else "fit_generator"
            # Use the combined replay-wrapper method for every other variant.
            else:
                generative_fit_method = fit_method
            fit_started = time.perf_counter()
            generative_history = train_task_model(
                generative_model, 
                generative_trainset, 
                generative_valset, 
                task_callbacks=phase_callbacks(
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
                    seed=task_seed,
                    metadata=diffusion_classifier_replay_mask
                    if replay_only_distillation else None,
                )
                fit_started = time.perf_counter()
                history = train_task_model(
                    generative_model, 
                    diffusion_classifier_trainset, 
                    generative_valset, 
                    task_callbacks=task_callbacks, 
                    fit_method="fit_discriminator"
                )
                task_resource["seconds"]["classifier_fit"] += float(
                    time.perf_counter() - fit_started
                )
        # Record that this task has no generative replay phase.
        else:
            generative_history = None

        # Compare the frozen slow/previous trace with the updated student on
        # a bounded old-class training probe. This is optional and never reads
        # the locked test split during development.
        if mechanistic_metrics and previous_teacher is not None and old_classes:
            probe_x, probe_y = _select_classes(
                all_x_train, all_y_train, old_classes
            )
            probe_x, probe_y = _sample_exact_rows(
                probe_x,
                probe_y,
                min(len(probe_x), mechanistic_max_samples),
                np.random.default_rng(derive_seed(
                    task_seed, "representation_probe"
                )),
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
                batch_size,
            )
            student_probe_probabilities = _predict_teacher_probabilities(
                generative_model.network,
                probe_x,
                diffusion_data_min,
                diffusion_data_range,
                batch_size,
            )
            task_resource["seconds"]["teacher_scoring"] += float(
                time.perf_counter() - scoring_started
            )
            old_width = teacher_probe_probabilities.shape[1]
            student_old_probabilities = student_probe_probabilities[:, :old_width]
            task_mechanistic["representation"] = {
                "old_class_probability_cka": linear_cka(
                    teacher_probe_probabilities,
                    student_old_probabilities,
                ) if len(probe_x) >= 2 else float("nan"),
                "old_class_probability_centroid_drift": class_centroid_drift(
                    teacher_probe_probabilities,
                    probe_ids,
                    student_old_probabilities,
                    probe_ids,
                ),
            }
            # Calibration requires held-out labels; never estimate it from the
            # training probe used for representation-drift diagnostics.
            if use_valset and all_x_val is not None and all_y_val is not None:
                calibration_x, calibration_y = _select_classes(
                    all_x_val,
                    all_y_val,
                    old_classes,
                )
                calibration_x, calibration_y = _sample_exact_rows(
                    calibration_x,
                    calibration_y,
                    min(len(calibration_x), mechanistic_max_samples),
                    np.random.default_rng(derive_seed(
                        task_seed, "calibration_probe"
                    )),
                )
                # Split limiting can remove every old-class validation row.
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
        # Evaluate the generative phase when one ran for this task.
        if generative_history is not None \
        and generative_evaluation_set is not None:
            generative_evaluations = report_task_model(
                generative_history, 
                generative_model, 
                generative_trainset, 
                generative_evaluation_set,
                split_name=evaluation_split_name,
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
            # Fall back to the complete report when the selected entry is absent.
            if isinstance(selected_evaluation, dict):
                accuracy_source = selected_evaluation

        acc = _reported_accuracy(accuracy_source)
        ensemble_acc = None

        # Development runs never execute a model on the locked test split.
        if experiment_phase != "development":
            y_test_ids = np.argmax(y_test, axis=-1) \
                if load_dataset_fn_kwargs["onehot_labels"] \
                else np.asarray(y_test).reshape(-1)

            # Obtain one common prediction vector for cumulative/task scores.
            if use_diffusion_classifier:
                predictions = _predict_diffusion_classes(
                    generative_model,
                    x_test,
                    y_test_ids,
                    diffusion_data_min,
                    diffusion_data_range,
                    batch_size
                )
            # Evaluate the VAE classifier without target conditioning.
            elif uses_attached_classifier and not train_classifier_separately:
                predictions = _predict_vae_classes(generative_model, x_test)
            # Use direct predictions from the standalone classifier.
            else:
                predictions = new_model.predict(
                    classifier_x_test,
                    verbose=verbose
                )

            prediction_ids = np.argmax(predictions, axis=-1)
            prediction_accuracy = float(accuracy_score(
                y_test_ids, prediction_ids
            ))
            # Preserve legacy report-derived cumulative accuracy when available.
            if acc is None:
                acc = prediction_accuracy
            matrix_row = [np.nan] * len(internal_task_groups)
            # Score learned groups separately without future-class rows.
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
            # Use the diffusion classifier's clean/noisy test input protocol.
            if use_diffusion_classifier:
                validation_predictions = _predict_diffusion_classes(
                    generative_model,
                    x_val,
                    y_val_ids,
                    diffusion_data_min,
                    diffusion_data_range,
                    batch_size,
                )
            # Keep VAE validation predictions independent of target labels.
            elif uses_attached_classifier and not train_classifier_separately:
                validation_predictions = _predict_vae_classes(
                    generative_model,
                    x_val,
                )
            # Evaluate standalone/direct classifier predictions otherwise.
            else:
                validation_predictions = new_model.predict(
                    classifier_x_val,
                    verbose=verbose,
                )
            validation_ids = np.argmax(validation_predictions, axis=-1)
            validation_row = [np.nan] * len(internal_task_groups)
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
            # Publish validation accuracy as the development task trajectory.
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
                ensemble_accuracy_matrix.append(ensemble_row)
                # Test ensemble values remain authoritative outside development.
                ensemble_acc = _observed_mean(
                    ensemble_row[:task_index + 1]
                )
                # Let the explicit ensemble policy replace ordinary test scores.
                if use_ensemble_accuracy:
                    acc = ensemble_acc
            # Build a validation ensemble matrix only for a real held-out split.
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
                validation_ensemble_accuracy_matrix.append(
                    validation_ensemble_row
                )
                # Development exposes the validation ensemble as its task score.
                if experiment_phase == "development":
                    ensemble_acc = _observed_mean(
                        validation_ensemble_row[:task_index + 1]
                    )
                    # Make the validation ensemble authoritative when requested.
                    if use_ensemble_accuracy:
                        acc = ensemble_acc

        # Expose joint generative history as task history when no separate fit ran.
        # This keeps classifier metrics present for joint-only attached heads.
        if uses_attached_classifier and not history \
        and generative_history is not None:
            history = generative_history

        optimizer_iterations_after = _optimizer_iteration_metrics(
            new_model, generative_model
        )
        task_resource["optimizer_updates"] = {
            name: int(value - optimizer_iterations_before.get(name, 0))
            for name, value in optimizer_iterations_after.items()
        }
        task_resource["teacher_network_name"] = (
            snapshot_network_name if previous_teacher is not None else None
        )
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
        # Retain ensemble scores beside the backward-compatible normal scores.
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

        # Commit only complete task state and never a partially trained phase.
        if save_task_checkpoints and checkpoint_dir is not None:
            # Materialize every optimizer slot at the save boundary. This
            # makes the checkpoint object graph complete even when a phase did
            # not happen to receive gradients in the just-finished task.
            prepare_optimizer_slots()
            checkpoint_trackables = recovery_trackables()
            trackable_topology = _trackable_topology_descriptor(
                checkpoint_trackables
            )
            checkpoint_state = {
                "class_order": class_order,
                "task_groups": original_task_groups,
                "accuracies": acc_list,
                "ensemble_accuracies": ensemble_acc_list,
                "histories": histories,
                "generative_histories": generative_histories,
                "classifier_evaluations": classifier_evaluations_list,
                "generative_evaluations": generative_evaluations_list,
                "ordinary_accuracy_matrix": ordinary_accuracy_matrix,
                "validation_accuracy_matrix": validation_accuracy_matrix,
                "ensemble_accuracy_matrix": ensemble_accuracy_matrix,
                "validation_ensemble_accuracy_matrix": (
                    validation_ensemble_accuracy_matrix
                ),
                "task_seeds": task_seeds,
                "task_resource_metrics": task_resource_metrics,
                "task_mechanistic_metrics": task_mechanistic_metrics,
                "previous_replay_samples": previous_replay_samples,
                "previous_replay_labels": previous_replay_labels,
                "fingerprint": run_fingerprint,
                "run_descriptor": run_descriptor,
                "trackable_topology": trackable_topology,
                "trackable_topology_fingerprint": fingerprint_state(
                    trackable_topology
                ),
            }
            checkpoint_path = save_task_checkpoint(
                checkpoint_dir,
                task_index,
                checkpoint_state,
                checkpoint_trackables,
                rng_state=capture_rng_state(numpy_generator=rng),
                replay_buffer=buffer if use_buffer else None,
                fingerprint=run_fingerprint,
            )
            checkpoint_paths.append(str(checkpoint_path))

        # Print the completed task's accuracy when requested.
        if verbose:
            split_label = "validation" \
                if experiment_phase == "development" else "test"
            print(f"Task {split_label} accuracy: {acc:.4f}")
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

    # Return histories and final objects for orchestration callers.
    if return_details:
        return {
            "accuracies": acc_list, 
            "ensemble_accuracies": ensemble_acc_list,
            "class_order": class_order,
            "task_classes": original_task_groups,
            "accuracy_matrix": accuracy_matrix,
            "ordinary_accuracy_matrix": ordinary_accuracy_matrix,
            "validation_accuracy_matrix": validation_accuracy_matrix,
            "ensemble_accuracy_matrix": ensemble_accuracy_matrix,
            "validation_ensemble_accuracy_matrix": (
                validation_ensemble_accuracy_matrix
            ),
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
            "dataset_seed": dataset_seed,
            "seed": seed,
            "task_seeds": task_seeds,
            "dtype_policy": dtype_policy,
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_paths": checkpoint_paths,
            "resumed_from": resume_from,
            "next_task_index": len(internal_task_groups),
            "histories": histories, 
            "generative_histories": generative_histories, 
            "classifier_evaluations": classifier_evaluations_list, 
            "generative_evaluations": generative_evaluations_list, 
            "task_resource_metrics": task_resource_metrics,
            "task_mechanistic_metrics": task_mechanistic_metrics,
            "run_descriptor": run_descriptor,
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


    # Verify the default singleton class-incremental schedule.
    assert resolve_continual_schedule(4) == (
        [0, 1, 2, 3],
        [[0], [1], [2], [3]],
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
    improved_metrics = _continual_metrics([
        [0.5, np.nan],
        [0.7, 0.6],
    ])
    assert np.isclose(improved_metrics["average_forgetting"], -0.2)
    assert np.isclose(improved_metrics["backward_transfer"], 0.2)
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
        {},
        {
            "validation_continual_metrics": {
                "final_average_accuracy": 0.3,
            },
            "continual_accuracy": [0.99, 0.99],
        },
    )
    assert np.isclose(validation_objective, 0.3)
    # Ensemble HPO must use the authoritative validation matrix as well.
    ensemble_objective = _objective_values(
        "continual",
        "dit_classifier",
        {},
        {
            "validation_continual_metrics": {
                "final_average_accuracy": 0.5,
            },
            "continual_accuracy": [0.99],
        },
        use_ensemble_accuracy=True,
    )
    assert np.isclose(ensemble_objective, 0.5)

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
              per task; defaults to singleton tasks.
            - ``task_size`` (int, default ``1``): Classes per automatically
              constructed task.
            - ``class_order_mode``/``task_order_mode`` (str): ``"fixed"`` or
              seeded ``"random"`` class/whole-task ordering.
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
              ``insert_num``, ``seed``, and ``strategy``; FIFO is the unchanged
              default, with reservoir and class-balanced storage opt-in.
            - ``baseline`` (str | None): Optional named baseline-ladder cell;
              ``None`` preserves all independent legacy switches.
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
            - ``use_generative_replay`` (bool, default ``True``): Disable old
              generation for no-replay/KD-only joint controls when false.
            - ``replay_budget_mode`` (str, default ``"legacy"``): Select
              historical counts or exact ``"fixed_total"`` exposure.
            - ``replay_old_examples``/``replay_current_examples`` (int | None):
              Exact fixed-total old/current row counts.
            - ``optimizer_steps_per_epoch`` (int | None): Optional positive
              update count applied to every active phase per epoch. The
              selected task pool repeats only when this is set.
            - ``replay_candidate_multiplier`` (int, default ``1``): Candidate
              pool multiple used before an optional replay gate.
            - ``replay_selection`` (str, default ``"all"``): All, uniform,
              random, confidence, surprise, or combined selection.
            - ``replay_cache_dir``/``replay_cache_mode``: Optional matched-pool
              cache location and off/write/read/``read_write`` policy.
            - ``mechanistic_metrics`` (bool, default ``False``): Enable bounded
              teacher, representation, replay-quality, gate, and resource data.
            - ``use_generative_model_classifier`` (bool, default ``False``):
              Use a classifier attached to the replay model.
            - ``train_classifier_separately`` (bool, default ``False``): Add a
              classifier phase for ``VAEClassifier``; it must be true for
              ``DiffusionClassifierV2`` and false for ``DiffusionClassifier``.
            - ``use_distillation`` (bool, default ``False``): Snapshot each
              completed diffusion-classifier student for use as the next
              task's teacher; requires a token and positive teacher objective.
            - ``snapshot_network_name`` (str, default ``"raw"``): Select the
              raw or EMA branch used for previous-task teachers.
            - ``teacher_network_name`` (str | None): Optional readable alias
              that overrides ``snapshot_network_name`` when supplied.
            - ``use_ensemble_accuracy`` (bool, default ``False``): Make
              per-task ensemble accuracies authoritative for CL metrics.
            - ``evaluate_ensemble_accuracy`` (bool, default ``False``): Also
              evaluate timestep-ensembled accuracy for a diffusion-classifier
              replay model after every task.
            - ``ensemble_accuracy_kwargs`` (dict | None): Options forwarded to
              ``DiffusionClassifier.evaluate_ensemble_accuracy``.
            - ``callbacks_list`` (Sequence[Callback] | None): Extra callbacks
              forwarded through :func:`common.train.train_model`.
            - ``generative_callbacks_list`` (Sequence[Callback] | None):
              Callbacks used only for generative phases.
            - ``return_details`` (bool, default ``False``): Return task
              histories and final model objects in addition to accuracies.
            - ``use_valset`` (bool, default ``True``): Use an explicit loader
              validation split; a missing split remains disabled.
            - ``seed`` (int | None): Master seed for every continual stream.
            - ``dtype_policy`` (str | None): Optional Keras numeric policy.
            - ``deterministic_ops`` (bool): Request deterministic TF kernels.
            - ``checkpoint_dir``/``resume_from`` (str | None): Task-boundary
              recovery output/input directories.
            - ``experiment_phase`` (str, default ``"legacy"``): Use
              ``"development"`` to prohibit test evaluation or
              ``"confirmation"`` after freezing the manifest/configuration.
            - ``experiment_manifest_path``/``experiment_manifest_hash``/
              ``experiment_run_id``: Required frozen-manifest identity for a
              confirmation run; its schedule and seed are verified before any
              locked-test access.

    Returns:
        list[float] | dict[str, object]: Protocol-selected accuracy for each
        task (validation in development, otherwise test). With
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
            "task_size": 1,
            "class_order_mode": "fixed",
            "task_order_mode": "fixed",
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
            "baseline": None,
            "plot_results": True, 
            "verbose": True, 
            "generative_model": None, 
            "generative_model_compile_args": None, 
            "generative_model_kwargs": None, 
            "use_generative_replay": True,
            "replay_budget_mode": "legacy",
            "replay_old_examples": None,
            "replay_current_examples": None,
            "optimizer_steps_per_epoch": None,
            "replay_candidate_multiplier": 1,
            "replay_selection": "all",
            "replay_surprise_weight": 0.5,
            "replay_cache_dir": None,
            "replay_cache_mode": "off",
            "mechanistic_metrics": False,
            "mechanistic_max_samples": 512,
            "use_generative_model_classifier": False, 
            "train_classifier_separately": False, 
            "use_distillation": False,
            "snapshot_network_name": "raw",
            "teacher_network_name": None,
            "use_ensemble_accuracy": False,
            "evaluate_ensemble_accuracy": False, 
            "ensemble_accuracy_kwargs": None, 
            "callbacks_list": None, 
            "generative_callbacks_list": None,
            "return_details": False, 
            "use_valset": True,
            "return_features": None,
            "max_train_samples": None,
            "max_val_samples": None,
            "shuffle_buffer": None,
            "pad": 0,
            "seed": None,
            "dataset_seed": None,
            "dtype_policy": None,
            "deterministic_ops": False,
            "initial_classifier": None,
            "callback_patience": None,
            "callback_monitor": None,
            "callback_monitor_mode": None,
            "save_task_checkpoints": False,
            "checkpoint_dir": None,
            "resume_from": None,
            "experiment_phase": "legacy",
            "experiment_manifest_path": None,
            "experiment_manifest_hash": None,
            "experiment_run_id": None,
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
            "dataset_seed": (
                config.continually_learn.seed
                if config.continually_learn.seed is not None
                else config.training.seed
            ),
            "model": model.get("classifier"), 
            "generative_model": model.get("generative_model")
        }
    details["evaluations"] = run["evaluations"]

    # Return full task details only when configured by the caller.
    if config.continually_learn.return_details:
        return details
    return details["accuracies"]
