"""Atomic task-boundary recovery primitives for continual experiments.

The continual learner owns *when* a task is complete and which model objects
belong to a run.  This module owns the durable representation of that state:

* TensorFlow trackables are written with :class:`tf.train.Checkpoint`, so model,
  EMA, teacher, optimizer, and explicit ``tf.random.Generator`` variables can be
  restored together.
* Python/NumPy generator state and arbitrary JSON-compatible experiment state
  are encoded without pickle.
* Replay-buffer samples are stored in an ``allow_pickle=False`` NPZ archive.
* A task directory becomes visible only after its ``COMMITTED`` marker is
  written and the temporary directory is atomically renamed.  ``latest.json``
  is merely an index; discovery falls back to the newest valid committed task
  if that index is absent, stale, or corrupt.

The checkpoint boundary is deliberately a *completed task*.  If a process is
interrupted inside a task, callers restore the preceding committed boundary and
restart the incomplete task with its deterministic task seed.
"""

from __future__ import annotations

import tensorflow as tf

import numpy as np

import base64

import hashlib

import json

import math

import os

import random

import re

import shutil

import uuid

from pathlib import Path

from dataclasses import dataclass
from collections.abc import Mapping, Sequence


SCHEMA_VERSION = 1
"""On-disk task-checkpoint schema version."""

_TASK_DIRECTORY_PATTERN = re.compile(r"^task-(\d{4,})$")
_TRACKABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_COMMITTED_NAME = "COMMITTED"
_STATE_NAME = "state.json"
_LATEST_NAME = "latest.json"
_REPLAY_NAME = "replay.npz"
_TF_DIRECTORY_NAME = "tf_checkpoint"
_TF_PREFIX_NAME = "ckpt"


@dataclass(frozen=True)
class TaskCheckpoint:
    """Hold decoded state from one immutable committed task checkpoint.

    The frozen dataclass fixes the top-level field bindings; contained mappings and
    arrays remain ordinary mutable objects. ``load_task_checkpoint`` constructs it
    for read-only inspection or after optional TensorFlow restoration. The ``state``
    property adds the cursor, schedule, RNG state, and fingerprint to experiment data.

    Attributes:
        task_dir (Path): Concrete directory containing the committed task state.
        completed_task_index (int): Zero-based index of the last completed task.
        next_task_index (int): First task still to execute, normally completed index + 1.
        class_order (tuple[object, ...]): Authoritative original class introduction order.
        task_groups (tuple[tuple[object, ...], ...]): Classes introduced by each task.
        experiment_state (dict[str, object]): Caller-owned histories, measurements,
            and orchestration metadata after removing schema-owned fields.
        rng_state (dict[str, object]): Portable RNG snapshot, or an empty mapping
            when no snapshot was supplied to the writer.
        replay_state (dict[str, object] | None): Decoded replay items, capacity, RNG,
            and optional strategy metadata; None for runs without replay storage.
        fingerprint (str | None): Optional immutable run/configuration identity.
        restore_status (object | None): TensorFlow checkpoint status after restoration.
            Defaults to None for inspection without TensorFlow trackables.
    """

    task_dir: Path
    completed_task_index: int
    next_task_index: int
    class_order: tuple[object, ...]
    task_groups: tuple[tuple[object, ...], ...]
    experiment_state: dict[str, object]
    rng_state: dict[str, object]
    replay_state: dict[str, object] | None
    fingerprint: str | None
    restore_status: object | None = None

    @property
    def state(self) -> dict[str, object]:
        """Return one integration-friendly mapping with cursor and schedule.

        Returns:
            dict[str, object]: Experiment state augmented with the recovery
            cursor, resolved schedule, RNG snapshot, and run fingerprint.
        """

        return {
            **self.experiment_state,
            "completed_task_index": self.completed_task_index,
            "next_task_index": self.next_task_index,
            "class_order": list(self.class_order),
            "task_groups": [list(group) for group in self.task_groups],
            "rng_state": self.rng_state,
            "fingerprint": self.fingerprint
        }


def _encode_json(value: object) -> object:
    """Encode supported Python/NumPy state into JSON-compatible data without pickle.

    Ordinary scalars, lists, and string-key mappings remain readable JSON. NumPy
    scalars normalize to Python values; arrays retain dtype/shape/base64 bytes.
    Paths, bytes, tuples, sets, and nonfinite floats use explicit type tags. Literal
    mappings containing the reserved tag key are escaped to preserve their meaning.
    Input containers and arrays are not mutated.

    Args:
        value (object): A supported scalar, non-object ndarray, Path, bytes, tuple,
            set, list, or string-key mapping; nested values follow the same rules.

    Returns:
        object: A tree containing only JSON-compatible primitives, lists, and
        dictionaries. Tagged values can be reconstructed by ``_decode_json``.

    Raises:
        TypeError: If arrays have object dtype, mappings have non-string keys, or
            an unsupported runtime object is encountered.
    """

    # Preserve scalar values that JSON represents directly.
    if value is None or isinstance(value, (bool, str, int)):
        return value

    # Encode exceptional floating values using explicit recovery tags.
    if isinstance(value, float):
        # Strict JSON cannot represent non-finite IEEE values.
        if math.isnan(value):
            return {"__recovery_type__": "float", "value": "nan"}

        # Preserve positive versus negative infinity in the float tag.
        if math.isinf(value):
            # Choose the infinity tag from the original value's sign.
            return {
                "__recovery_type__": "float",
                "value": "inf" if value > 0 else "-inf"
            }

        return value

    # Convert NumPy scalars through their equivalent Python values.
    if isinstance(value, np.generic):
        return _encode_json(value.item())

    # Encode array dtype, shape, and contiguous bytes together.
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)

        # Reject object arrays whose bytes contain process-local object references.
        if contiguous.dtype.hasobject:
            raise TypeError(
                "Object-dtype arrays are not recovery-serializable."
            )

        return {
            "__recovery_type__": "ndarray",
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "data": base64.b64encode(contiguous.tobytes()).decode("ascii")
        }

    # Store filesystem paths with a recoverable path type tag.
    if isinstance(value, Path):
        return {
            "__recovery_type__": "path",
            "value": str(value)
        }

    # Encode byte strings as base64 for JSON storage.
    if isinstance(value, bytes):
        return {
            "__recovery_type__": "bytes",
            "data": base64.b64encode(value).decode("ascii")
        }

    # Preserve tuple identity with a tagged sequence.
    if isinstance(value, tuple):
        return {
            "__recovery_type__": "tuple",
            "items": [_encode_json(item) for item in value]
        }

    # Sort encoded set members to obtain deterministic serialized order.
    if isinstance(value, set):
        encoded_items = [_encode_json(item) for item in value]
        encoded_items.sort(key=_stable_json_dumps)

        return {
            "__recovery_type__": "set",
            "items": encoded_items
        }

    # Encode list elements while retaining ordinary JSON list structure.
    if isinstance(value, list):
        return [_encode_json(item) for item in value]

    # Encode mapping values while preserving string keys.
    if isinstance(value, Mapping):
        encoded = {}
        for key, item in value.items():
            # Reject mapping keys that cannot be represented as JSON object keys.
            if not isinstance(key, str):
                raise TypeError("Recovery mapping keys must be strings.")

            encoded[key] = _encode_json(item)

        # Escape literal mappings that use the serializer's reserved tag key.
        if "__recovery_type__" in encoded:
            return {
                "__recovery_type__": "mapping",
                "items": encoded,
            }

        return encoded

    raise TypeError(
        "Unsupported recovery value type: " + type(value).__qualname__
    )


def _decode_json(value: object) -> object:
    """Reconstruct Python/NumPy state emitted by the recovery JSON encoder.

    Tagged arrays are decoded into owned writable copies after verifying payload
    size against shape and dtype. Tagged tuples, sets, paths, bytes, and exceptional
    floats recover their original kinds. Untagged mappings/lists are decoded
    recursively, and escaped literal mappings do not interpret their own tag key.

    Args:
        value (object): Parsed JSON-compatible value produced by ``_encode_json``.

    Returns:
        object: Reconstructed primitive/container/NumPy value. Arrays retain the
        encoded shape and dtype and do not share the encoded byte buffer.

    Raises:
        ValueError: If an array's encoded size is inconsistent or its type tag is
            unknown, or if a malformed encoded value cannot be reconstructed.
        KeyError: If a tagged record lacks a field required by its encoding.
    """

    # Decode ordinary lists element by element.
    if isinstance(value, list):
        return [_decode_json(item) for item in value]

    # Return scalar JSON values without tag interpretation.
    if not isinstance(value, dict):
        return value

    type_name = value.get("__recovery_type__")

    # Decode untagged dictionaries as ordinary mappings.
    if type_name is None:
        return {
            key: _decode_json(item)
            for key, item in value.items()
        }

    # Restore a tagged NaN or signed infinity.
    if type_name == "float":
        return {
            "nan": float("nan"),
            "inf": float("inf"),
            "-inf": float("-inf")
        }[value["value"]]

    # Reconstruct tagged arrays from their dtype, shape, and byte payload.
    if type_name == "ndarray":
        dtype = np.dtype(value["dtype"])
        shape = tuple(int(size) for size in value["shape"])
        raw = base64.b64decode(value["data"].encode("ascii"))
        expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize

        # Reject array payloads whose bytes do not match the declared shape.
        if len(raw) != expected_size:
            raise ValueError(
                "Encoded ndarray byte length does not match its shape."
            )

        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()

    # Reconstruct a tagged filesystem path.
    if type_name == "path":
        return Path(value["value"])

    # Decode a tagged byte string from base64.
    if type_name == "bytes":
        return base64.b64decode(value["data"].encode("ascii"))

    # Reconstruct a tagged tuple without changing its container type.
    if type_name == "tuple":
        return tuple(_decode_json(item) for item in value["items"])

    # Reconstruct a tagged set from its decoded members.
    if type_name == "set":
        return set(_decode_json(item) for item in value["items"])

    # Restore an escaped literal mapping without interpreting its own tag key.
    if type_name == "mapping":
        return {
            key: _decode_json(item)
            for key, item in value["items"].items()
        }

    raise ValueError(f"Unknown recovery JSON type tag: {type_name!r}.")


def _stable_json_dumps(value: object) -> str:
    """Serialize already encoded recovery state to canonical compact JSON text.

    Keys are sorted and separators contain no extra whitespace. Unicode text is
    retained directly. Nonfinite floats must already be represented by encoder tags;
    raw NaN/Inf values are rejected so fingerprints never depend on nonstandard JSON.

    Args:
        value (object): JSON-compatible primitive/container tree, normally returned
            by ``_encode_json``. Raw arrays and other custom objects are unsupported.

    Returns:
        str: Deterministic JSON text with no trailing newline.

    Raises:
        TypeError: If a value is not JSON-serializable.
        ValueError: If the tree contains raw nonfinite floats or circular references.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":")
    )


def fingerprint_state(value: object) -> str:
    """Return a stable SHA-256 identity for supported serializable experiment state.

    The value is encoded without pickle, serialized as canonical sorted JSON, and
    hashed as UTF-8. Array dtype/shape/content and tagged container types participate
    in the identity; no files are read or written and inputs are not mutated.

    Args:
        value (object): Python/NumPy state accepted by ``_encode_json``, including
            nested string-key mappings, sequences, non-object arrays, and scalars.

    Returns:
        str: A 64-character lowercase hexadecimal SHA-256 digest.

    Raises:
        TypeError: If the value contains unsupported objects, object arrays, or
            mappings whose keys are not strings.
    """

    encoded = _encode_json(value)

    return hashlib.sha256(
        _stable_json_dumps(encoded).encode("utf-8")
    ).hexdigest()


def _qualified_name(value: object) -> str:
    """Return a stable module-qualified identity for a class, instance, or callable.

    Functions with module/qualified-name metadata use their own identity; ordinary
    instances and other callable objects fall back to their class identity. This
    avoids embedding process-dependent repr strings in recovery fingerprints.

    Args:
        value (object): Class, callable, or instance whose semantic type is described.

    Returns:
        str: ``"module.qualified_name"`` for the callable or object's type.
    """

    # Use a class directly; use its type when describing an instance.
    candidate = value if isinstance(value, type) else type(value)
    # Prefer the callable's own identity when the value is a function or bound callable.
    if callable(value) and not isinstance(value, type):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)

        # Use the callable's module and qualified name when both are available.
        if module is not None and qualname is not None:
            return f"{module}.{qualname}"

    return f"{candidate.__module__}.{candidate.__qualname__}"


def _array_recovery_descriptor(value: object) -> dict[str, object] | None:
    """Describe an optional numeric array by shape, dtype, and a SHA-256 content digest.

    The hash includes the NumPy dtype string, shape, and contiguous payload bytes,
    so equal bytes with different shapes/dtypes remain distinct. Empty arrays retain
    shape/dtype identity without attempting to cast an empty memoryview. No file is
    written and the supplied array is not mutated.

    Args:
        value (object): NumPy-compatible array or None for an absent input. Object
            dtypes are unsupported because their bytes contain Python pointers.

    Returns:
        dict[str, object] | None: ``shape`` as a dimension list, ``dtype`` as its
        NumPy string, and ``sha256`` as a hexadecimal digest; None for absent input.

    Raises:
        TypeError: If the normalized array has an object dtype.
    """

    # Preserve an absent optional array descriptor.
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
    """Fingerprint a model artifact without binding its identity to its absolute path.

    Files are hashed incrementally. Directories are represented by sorted relative
    file paths, sizes, and content hashes; empty directories contribute no records.
    This function reads artifacts but does not create, load as code, or modify them.

    Args:
        path (str): Model file/directory path. An empty string means no artifact.

    Returns:
        dict[str, object] | None: A file descriptor with ``kind``, ``size``, and
        ``sha256``; a directory descriptor with ``kind`` and a ``files`` list; a
        ``{"kind": "missing"}`` descriptor for a nonexistent nonempty path; or
        None when no path was supplied.

    Raises:
        OSError: If an existing artifact cannot be inspected or read.
    """

    # Leave an unspecified model artifact out of the recovery descriptor.
    if not path:
        return None

    artifact = Path(path)

    # Fingerprint a single model artifact file by size and content.
    if artifact.is_file():
        return {
            "kind": "file",
            "size": artifact.stat().st_size,
            "sha256": _sha256_file(artifact)
        }

    # Fingerprint directory artifacts using their ordered relative file records.
    if artifact.is_dir():
        files = []
        # Include regular files only, leaving directory entries out of the artifact digest.
        for child in sorted(item for item in artifact.rglob("*") if item.is_file()):
            files.append({
                "path": child.relative_to(artifact).as_posix(),
                "size": child.stat().st_size,
                "sha256": _sha256_file(child)
            })

        return {"kind": "directory", "files": files}

    return {"kind": "missing"}


def _model_weight_descriptor(model: object) -> list[dict[str, object]] | None:
    """Fingerprint a model's currently materialized weights in their existing order.

    Only initialized weight values are inspected. The helper does not build a model,
    create optimizer slots, or mutate variables; caller-owned initializers and
    teachers can therefore contribute exact content to a run fingerprint.

    Args:
        model (object): Keras-like model exposing eager ``weights``, or None.

    Returns:
        list[dict[str, object]] | None: One shape/dtype/SHA-256 descriptor per weight,
        an empty list for a model without weights, or None for an absent model.
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
    """Convert configuration objects into deterministic, compact recovery descriptions.

    Primitive values retain their meaning; arrays are content-hashed, containers are
    traversed recursively, and Keras objects use their constructor configuration.
    Sets and non-string-key mappings are ordered by stable fingerprints. Active
    reference cycles are marked rather than traversed repeatedly. Unknown objects
    contribute a qualified type name, not full state or process-local identity.

    Args:
        value (object): Configuration value, array, container, callable, or object
            exposing ``get_config`` to describe.
        active_ids (set[int] | None): Recursion guard. Defaults to None to create
            a fresh set; nested calls temporarily add/remove IDs in the supplied set.
        strip_config_names (bool): Defaults to False. True omits mapping entries
            named ``name`` and rounds floating configuration scalars to seven
            significant digits to normalize Keras float32 constructor round trips.
            Keras model/layer/optimizer configs enable this normalization internally.

    Returns:
        object: Nested primitive/list/dictionary description suitable for
        ``fingerprint_state``. This is compatibility metadata, not a reconstruction
        format or a substitute for a TensorFlow model checkpoint.

    Raises:
        TypeError: If an array has an unsupported object dtype.
        ValueError: If a provided ``get_config`` method cannot produce its config.
    """

    # Keep directly serializable scalar configuration values unchanged.
    if value is None or isinstance(value, (bool, str, int)):
        return value

    # Canonicalize only float32-backed Keras configuration round trips.
    if isinstance(value, float):
        # Keras may materialize the same constructor scalar through float32 on
        # a compiled object (for example 0.9 -> 0.899999976). Seven significant
        # digits preserve float32 semantics while keeping the descriptor stable.
        # Round Keras float32 configuration scalars only during name-stripped normalization.
        return float(format(value, ".7g")) if strip_config_names else value

    # Normalize NumPy scalar configuration values through their Python equivalents.
    if isinstance(value, np.generic):
        return _recovery_descriptor(
            value.item(), active_ids, strip_config_names
        )

    # Fingerprint array configuration values by content instead of embedding them.
    if isinstance(value, np.ndarray):
        return _array_recovery_descriptor(value)

    # Record TensorFlow dtypes by their portable registered name.
    if isinstance(value, tf.dtypes.DType):
        return {"type": "tensorflow.DType", "name": value.name}

    # Represent TensorFlow shapes by their portable dimension list.
    if isinstance(value, tf.TensorShape):
        return {"type": "tensorflow.TensorShape", "shape": value.as_list()}

    # Create a recursion guard for the first call; reuse it for nested objects.
    active_ids = set() if active_ids is None else active_ids
    object_id = id(value)
    # Represent a repeated active object as a cycle instead of recursing again.
    if object_id in active_ids:
        return {"type": _qualified_name(value), "cycle": True}

    active_ids.add(object_id)
    try:
        # Preserve mapping semantics while normalizing key order.
        if isinstance(value, dict):
            # Ordinary string-key mappings remain readable in the checkpoint.
            # Use direct JSON objects when every mapping key is a string.
            if all(isinstance(key, str) for key in value):
                # Omit generated Keras names only when name-stripped configuration is requested.
                return {
                    key: _recovery_descriptor(
                        value[key],
                        active_ids,
                        strip_config_names
                    )
                    for key in sorted(value)
                    if not (strip_config_names and key == "name")
                }

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

        # Describe ordered sequences recursively in their existing order.
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

        # Record other callables by qualified name when they have no configuration API.
        if callable(value):
            return {"callable": _qualified_name(value)}

        # Unknown runtime objects contribute their stable type, not transient
        # instance identity. All supported Keras objects expose get_config().
        return {"type": _qualified_name(value)}
    finally:
        active_ids.remove(object_id)


def _model_topology_descriptor(model: object) -> dict[str, object] | None:
    """Describe a model's constructor configuration and initialized variable topology.

    The result records shapes, dtypes, and trainability without reading mutable
    weight values. It does not build missing layers or change the supplied model.

    Args:
        model (object): Keras-like object with optional ``weights``, or None.

    Returns:
        dict[str, object] | None: ``object`` contains its recovery configuration;
        ``weights`` lists shape/dtype/trainable records in weight order. None
        preserves an absent optional model.
    """

    # Preserve an absent optional model topology.
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
    """Describe named model/optimizer structures before strict checkpoint restoration.

    Names are processed in sorted order. Both method-based and property-based Keras
    ``variables`` APIs are accepted, and no variable values are changed or recorded.

    Args:
        trackables (dict[str, object]): Dependency names mapped to built models,
            optimizers, or other objects exposing a variable collection.

    Returns:
        dict[str, object]: Each name maps to an ``object`` recovery description and
        an ordered ``variables`` list of shape/dtype/trainable records. Objects
        without variables contribute an empty list.
    """

    result: dict[str, object] = {}
    for name in sorted(trackables):
        value = trackables[name]
        variables_attr = getattr(value, "variables", None)
        # Call method-based variable APIs; read property-based variable collections directly.
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
    """Resolve persistent depth additions made by a progressive training specification.

    ``depths_only`` uses the entire ``depths`` sequence; timestep/resolution-only
    curricula add no depth. Explicit stage sequences accept ``"depth"``, depth
    mappings, sets containing depth, and ``("depth", specification)`` pairs.
    Missing inline specifications use the corresponding stage index in ``depths``.

    Args:
        fit_kwargs (dict[str, object]): Progressive fit arguments. Relevant keys
            are ``stage_tasks`` and optional stage-aligned ``depths``.

    Returns:
        list[object]: Depth specifications in execution order, excluding stages
        that do not alter persistent topology. No source configuration is mutated.

    Raises:
        ValueError: If a depth stage has neither an inline nor a stage-indexed spec.
    """

    stage_tasks = fit_kwargs.get("stage_tasks")
    depths = fit_kwargs.get("depths")

    # A depth-only curriculum uses the complete configured depth sequence.
    if stage_tasks == "depths_only":
        return list(depths or [])

    # Timestep- and resolution-only curricula never alter persistent topology.
    if stage_tasks in ("timesteps_only", "resolutions_only"):
        return []

    # Ignore non-sequence curricula that do not describe explicit depth stages.
    if not isinstance(stage_tasks, Sequence) or isinstance(stage_tasks, str):
        return []

    resolved = []
    for stage_index, task in enumerate(stage_tasks):
        has_depth = False
        depth_spec = None

        # A bare depth stage obtains its specification from the separate depths sequence.
        if task == "depth":
            has_depth = True
        # A depth mapping carries its own optional depth specification.
        elif isinstance(task, dict) and "depth" in task:
            has_depth = True
            depth_spec = task["depth"]
        # A set containing depth requests the corresponding stage-indexed specification.
        elif isinstance(task, (set, frozenset)) and "depth" in task:
            has_depth = True
        # A two-item depth pair supplies its specification inline.
        elif isinstance(task, (tuple, list)) and len(task) == 2 \
        and task[0] == "depth":
            has_depth = True
            depth_spec = task[1]

        # Skip stages that do not add persistent depth.
        if not has_depth:
            continue

        # Resolve an unspecified depth from its stage-indexed configuration.
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


def _write_json(path: Path, value: object) -> None:
    """Encode, write, and flush one recovery JSON file to durable storage.

    The destination is opened for replacement, written as canonical UTF-8 JSON plus
    one newline, flushed, and fsynced. Parent directories must already exist. Atomic
    publication is the calling checkpoint writer's responsibility.

    Args:
        path (Path): Destination file, normally inside a private task directory.
        value (object): State accepted by ``_encode_json``; unsupported values are
            rejected before the destination is opened.

    Returns:
        None: The JSON file is written; no in-memory input is changed.

    Raises:
        TypeError: If the state cannot be encoded safely.
        OSError: If opening, writing, flushing, or syncing the destination fails.
    """

    encoded = _encode_json(value)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(_stable_json_dumps(encoded))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> object:
    """Read a UTF-8 recovery JSON file and reconstruct its supported Python values.

    This helper parses/decodes the file only. Schema, task identity, path, and
    checksum checks belong to the checkpoint validation functions.

    Args:
        path (Path): Existing recovery JSON file to open in text mode.

    Returns:
        object: Decoded Python/NumPy state returned by ``_decode_json``.

    Raises:
        OSError: If the file cannot be opened/read.
        ValueError: If JSON text or a tagged recovery value is malformed.
        KeyError: If an encoded tagged value omits a required field.
    """

    with path.open("r", encoding="utf-8") as stream:
        return _decode_json(json.load(stream))


def _sha256_file(path: Path) -> str:
    """Hash an existing file incrementally without loading its complete contents.

    The file is read in 1 MiB chunks and is not modified. The resulting identity
    covers its bytes only, excluding path, timestamps, and other filesystem metadata.

    Args:
        path (Path): Readable file whose raw content should be hashed.

    Returns:
        str: A 64-character lowercase hexadecimal SHA-256 content digest.

    Raises:
        OSError: If the file cannot be opened or read.
    """

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def _validate_schedule(
    completed_task_index: int,
    class_order: Sequence[object],
    task_groups: Sequence[Sequence[object]]
) -> tuple[list[object], list[list[object]]]:
    """Normalize a resolved class schedule and validate a completed-task cursor.

    The class order and every task group must be nonempty, the cursor must identify
    an existing task, and flattening task groups must equal the unique class order.
    Comparison uses the recovery encoding so supported NumPy scalar labels remain
    portable. This helper does not shuffle, regroup, or mutate the input schedule.

    Args:
        completed_task_index (int): Zero-based index of a completed task; normalized
            with int and required to lie in ``[0, len(task_groups))``.
        class_order (Sequence[object]): Already resolved unique class introduction
            order, using labels supported by the recovery serializer.
        task_groups (Sequence[Sequence[object]]): Ordered nonempty class groups whose
            concatenation equals class_order exactly.

    Returns:
        tuple[list[object], list[list[object]]]: Portable class order and task groups,
        with NumPy scalar labels converted through their Python equivalents.

    Raises:
        ValueError: If the schedule is empty, duplicated, inconsistent, or has an
            invalid completed-task cursor.
        TypeError: If labels cannot be represented by the recovery serializer.
    """

    completed_task_index = int(completed_task_index)

    normalized_order = list(class_order)
    normalized_groups = [list(group) for group in task_groups]

    # Reject a checkpoint schedule with no classes.
    if not normalized_order:
        raise ValueError("class_order must not be empty.")

    # Reject a schedule with no tasks or with an empty task group.
    if not normalized_groups or any(not group for group in normalized_groups):
        raise ValueError("task_groups must contain only nonempty groups.")

    # Require the completed cursor to identify one of the scheduled tasks.
    if completed_task_index < 0 or completed_task_index >= len(normalized_groups):
        raise ValueError(
            "completed_task_index must identify a task in task_groups."
        )

    flattened = [label for group in normalized_groups for label in group]
    encoded_order = [
        _stable_json_dumps(_encode_json(item)) for item in normalized_order
    ]
    encoded_flattened = [
        _stable_json_dumps(_encode_json(item)) for item in flattened
    ]

    # Require task-group expansion to match the declared class order exactly.
    if encoded_flattened != encoded_order:
        raise ValueError("Flattening task_groups must equal class_order exactly.")

    # Reject repeated classes in the introduction schedule.
    if len(set(encoded_order)) != len(encoded_order):
        raise ValueError("class_order must contain unique labels.")

    # Round-trip NumPy scalars into portable Python values before persistence.
    normalized_order = _decode_json(_encode_json(normalized_order))
    normalized_groups = _decode_json(_encode_json(normalized_groups))

    return normalized_order, normalized_groups


def capture_rng_state(
    *,
    numpy_generator: np.random.Generator | None = None,
    python_rng: random.Random | None = None,
    include_globals: bool = True,
    tensorflow_generator: object | None = None,
    include_tensorflow_global: bool = False
) -> dict[str, object]:
    """Capture Python, NumPy, and optional TensorFlow generator state without advancing it.

    Legacy stateful ``tf.random.*`` counters and live tf.data iterators are not
    representable by this JSON snapshot. Exact task-boundary recovery therefore
    reseeds each restarted incomplete task from its derived task seed. Explicit
    TensorFlow generators are trackable and can also have their state recorded here.

    Args:
        numpy_generator (np.random.Generator | None): Local NumPy generator to
            snapshot. Defaults to None to omit local NumPy state.
        python_rng (random.Random | None): Local Python RNG to snapshot. Defaults
            to None to omit local Python state.
        include_globals (bool): Defaults to True to capture process-wide Python
            and NumPy RNGs; False records only requested local/TensorFlow state.
        tensorflow_generator (object | None): TensorFlow Generator-like object
            exposing state and algorithm. Defaults to None to omit TensorFlow
            state unless global-generator capture is requested.
        include_tensorflow_global (bool): Defaults to False. True resolves the
            TensorFlow global generator and uses it instead of any explicitly
            supplied tensorflow_generator.

    Returns:
        dict[str, object]: Versioned snapshot containing only requested available
        entries: ``python_global``/``python_local`` RNG tuples, ``numpy_global``
        legacy RandomState fields, ``numpy_local`` bit-generator name/state, and
        ``tensorflow_generator`` algorithm ID/state array. The state is accepted by
        the recovery serializer; it is not written to disk by this function.

    Raises:
        RuntimeError: If global TensorFlow capture is requested but its API is
            unavailable, or TensorFlow cannot initialize/access its global generator.
    """

    state: dict[str, object] = {"schema_version": SCHEMA_VERSION}

    # Capture Python and NumPy process-wide RNGs when global state is requested.
    if include_globals:
        state["python_global"] = random.getstate()
        numpy_state = np.random.get_state()
        state["numpy_global"] = {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1],
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4]
        }

    # Capture the supplied local Python RNG without changing global state.
    if python_rng is not None:
        state["python_local"] = python_rng.getstate()

    # Capture the supplied NumPy generator and its bit-generator type.
    if numpy_generator is not None:
        state["numpy_local"] = {
            "bit_generator": type(numpy_generator.bit_generator).__name__,
            "state": numpy_generator.bit_generator.state
        }

    # Resolve TensorFlow's global generator only when explicitly requested.
    if include_tensorflow_global:
        experimental = getattr(tf.random, "experimental", None)
        getter = getattr(experimental, "get_global_generator", None)

        # Report TensorFlow versions that lack the global-generator getter.
        if getter is None:
            raise RuntimeError(
                "This TensorFlow version has no global Generator API."
            )

        tensorflow_generator = getter()

    # Capture TensorFlow state when an explicit or global generator is available.
    if tensorflow_generator is not None:
        generator_state = np.asarray(tensorflow_generator.state.numpy())
        algorithm = tensorflow_generator.algorithm
        # Read tensor-valued algorithm IDs through numpy; convert scalar IDs directly.
        algorithm = int(algorithm.numpy()) if hasattr(algorithm, "numpy") \
                    else int(algorithm)
        state["tensorflow_generator"] = {
            "algorithm": algorithm,
            "state": generator_state
        }

    return state


def restore_rng_state(
    state: Mapping[str, object],
    *,
    numpy_generator: np.random.Generator | None = None,
    python_rng: random.Random | None = None,
    restore_globals: bool = True,
    tensorflow_generator: object | None = None,
    restore_tensorflow_global: bool = False
) -> dict[str, object]:
    """Restore a portable RNG snapshot and return its local generator objects.

    Supplied local objects are updated in place. Missing local objects are created
    only when their corresponding state is present. Global Python/NumPy restoration
    is independently selectable, and TensorFlow global installation is opt-in.
    Missing snapshot entries are left unchanged; no random draws are generated.

    Args:
        state (Mapping[str, object]): Versioned snapshot from ``capture_rng_state``
            or its decoded checkpoint representation.
        numpy_generator (np.random.Generator | None): Existing local generator to
            restore. Defaults to None to construct the saved bit-generator family
            when local NumPy state exists.
        python_rng (random.Random | None): Existing local Python RNG to restore.
            Defaults to None to create one when local Python state exists.
        restore_globals (bool): Defaults to True to restore available global
            Python/NumPy entries; False leaves both process-wide RNGs unchanged.
        tensorflow_generator (object | None): Existing TensorFlow generator to
            restore. Defaults to None to create one from saved algorithm/state when
            TensorFlow state exists. A supplied generator must use the same algorithm.
        restore_tensorflow_global (bool): Defaults to False. True also installs a
            restored TensorFlow generator globally; absent TensorFlow state is ignored.

    Returns:
        dict[str, object]: Restored local objects under ``python_rng``,
        ``numpy_generator``, and/or ``tensorflow_generator``. Keys appear only for
        snapshot entries that existed. Global Python/NumPy states are not returned.

    Raises:
        ValueError: If the snapshot schema, NumPy bit-generator family, TensorFlow
            algorithm, or supplied RNG state is incompatible.
        RuntimeError: If TensorFlow global installation is requested but unavailable.

    Side Effects:
        Selected global RNGs and supplied local generators are mutated. Restoration
        is sequential; this function does not roll back earlier RNGs if a later
        malformed entry fails validation.
    """

    # Reject RNG snapshots written with an unsupported schema.
    if int(state.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Unsupported RNG-state schema version.")

    restored: dict[str, object] = {}

    # Restore Python's global RNG only when requested and present in the snapshot.
    if restore_globals and "python_global" in state:
        random.setstate(state["python_global"])

    # Restore NumPy's global RNG only when requested and present in the snapshot.
    if restore_globals and "numpy_global" in state:
        numpy_state = state["numpy_global"]
        np.random.set_state((
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"])
        ))

    # Restore a local Python RNG when the snapshot contains one.
    if "python_local" in state:
        # Create a missing local RNG; otherwise restore the supplied instance in place.
        python_rng = random.Random() if python_rng is None else python_rng
        python_rng.setstate(state["python_local"])
        restored["python_rng"] = python_rng

    # Restore a local NumPy generator when the snapshot contains one.
    if "numpy_local" in state:
        numpy_state = state["numpy_local"]
        bit_generator_name = str(numpy_state["bit_generator"])

        # Construct a missing NumPy generator using the saved bit-generator family.
        if numpy_generator is None:
            bit_generator_type = getattr(np.random, bit_generator_name, None)
            # Reject snapshots naming a bit-generator family unavailable in NumPy.
            if bit_generator_type is None:
                raise ValueError(
                    f"NumPy has no bit generator named {bit_generator_name!r}."
                )

            numpy_generator = np.random.Generator(bit_generator_type())
        # Reject a supplied NumPy generator whose algorithm differs from the snapshot.
        elif type(numpy_generator.bit_generator).__name__ != bit_generator_name:
            raise ValueError(
                "Supplied NumPy generator uses a different bit-generator type."
            )

        numpy_generator.bit_generator.state = numpy_state["state"]
        restored["numpy_generator"] = numpy_generator

    # Restore TensorFlow generator state when it was included in the snapshot.
    if "tensorflow_generator" in state:
        tf_state = state["tensorflow_generator"]
        algorithm = int(tf_state["algorithm"])
        values = tf.convert_to_tensor(tf_state["state"], dtype=tf.int64)
        # Construct a missing TensorFlow generator from the saved algorithm and state.
        if tensorflow_generator is None:
            tensorflow_generator = tf.random.Generator.from_state(
                values,
                alg=algorithm
            )
        # Restore a supplied TensorFlow generator after checking its algorithm.
        else:
            current_algorithm = tensorflow_generator.algorithm
            # Normalize tensor-valued algorithm IDs through numpy and scalar IDs directly.
            current_algorithm = int(current_algorithm.numpy()) if hasattr(current_algorithm, "numpy") \
                                else int(current_algorithm)
            # Reject restoration into a TensorFlow generator using a different algorithm.
            if current_algorithm != algorithm:
                raise ValueError(
                    "Supplied TensorFlow generator uses a different algorithm."
                )

            tensorflow_generator.state.assign(values)

        restored["tensorflow_generator"] = tensorflow_generator

        # Install the restored generator globally only when requested.
        if restore_tensorflow_global:
            experimental = getattr(tf.random, "experimental", None)
            setter = getattr(experimental, "set_global_generator", None)

            # Report TensorFlow versions that lack the global-generator setter.
            if setter is None:
                raise RuntimeError(
                    "This TensorFlow version has no global Generator API."
                )

            setter(tensorflow_generator)

    return restored


def _validate_trackables(
    trackables: Mapping[str, object] | None,
) -> dict[str, object]:
    """Normalize named checkpoint dependencies and reject unstable dependency names.

    None-valued entries are omitted. Remaining names must be strings matching a
    Python-style identifier; actual TensorFlow trackability is checked when the
    TensorFlow checkpoint object is constructed, not by this normalization helper.

    Args:
        trackables (Mapping[str, object] | None): Dependency names mapped to model,
            optimizer, variable, or RNG objects. None means no TensorFlow payload.

    Returns:
        dict[str, object]: Fresh mapping of non-None dependency values by validated
        name. The dependency objects themselves are retained by reference.

    Raises:
        ValueError: If a remaining dependency name is not a valid identifier.
    """

    # Exclude absent optional objects from the TensorFlow dependency mapping.
    normalized = {
        name: value
        for name, value in dict(trackables or {}).items()
        if value is not None
    }
    for name in normalized:
        # Reject dependency names that are not stable Python-style identifiers.
        if not isinstance(name, str) or not _TRACKABLE_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"Invalid TensorFlow checkpoint dependency name: {name!r}."
            )

    return normalized


def _write_replay_archive(
    task_dir: Path,
    replay_buffer: object
) -> dict[str, object]:
    """Write replay samples as a non-pickled NPZ and return their manifest metadata.

    Each sample-label component is stacked independently, so items within each
    component must have homogeneous shape. Empty buffers use empty placeholder
    arrays. A state_dict-capable buffer contributes insertion-strategy counters and
    class allocations without duplicating samples, capacity, or RNG fields.

    Args:
        task_dir (Path): Existing private task directory that receives ``replay.npz``.
        replay_buffer (object): Buffer exposing ``buffer`` sample-label pairs,
            ``maxlen``, and a private ``_rng`` with getstate. Optional state_dict
            provides reservoir/class-balanced continuation metadata.

    Returns:
        dict[str, object]: ``path``, ``count``, ``maxlen``, and ``rng_state`` fields
        plus optional ``strategy_state``. The array payload is stored separately in
        the NPZ under ``x`` and ``y`` with their original stacked dtypes/shapes.

    Raises:
        TypeError: If the buffer interface/items are unsupported or arrays contain
            object dtypes.
        ValueError: If replay items cannot be stacked into homogeneous arrays.
        OSError: If the archive cannot be written.
    """

    # Require replay storage, capacity, and private RNG before writing an archive.
    if not hasattr(replay_buffer, "buffer") \
    or not hasattr(replay_buffer, "maxlen") \
    or not hasattr(replay_buffer, "_rng"):
        raise TypeError(
            "replay_buffer must expose buffer, maxlen, and a private _rng."
        )

    items = list(replay_buffer.buffer)

    # Reject replay entries that are not sample-label pairs.
    if any(not isinstance(item, (tuple, list)) or len(item) != 2 for item in items):
        raise TypeError("Every replay-buffer item must be an (x, y) pair.")

    # Stack nonempty replay items into homogeneous sample and label arrays.
    if items:
        try:
            x_values = np.stack([np.asarray(item[0]) for item in items])
            y_values = np.stack([np.asarray(item[1]) for item in items])
        except ValueError as error:
            raise ValueError(
                "Replay-buffer x and y items must have homogeneous shapes."
            ) from error
    # Represent an empty replay buffer with empty placeholder arrays.
    else:
        # Dtypes/shapes are immaterial for an empty buffer and become defined by
        # the first post-resume insertion.
        x_values = np.empty((0,), dtype=np.float32)
        y_values = np.empty((0,), dtype=np.uint8)

    # Reject replay arrays that would require object serialization.
    if x_values.dtype.hasobject or y_values.dtype.hasobject:
        raise TypeError("Object-dtype replay items cannot be saved safely.")

    archive_path = task_dir / _REPLAY_NAME
    np.savez_compressed(archive_path, x=x_values, y=y_values)
    replay_metadata = {
        "path": _REPLAY_NAME,
        "count": len(items),
        "maxlen": replay_buffer.maxlen,
        "rng_state": replay_buffer._rng.getstate()
    }
    state_getter = getattr(replay_buffer, "state_dict", None)

    # Preserve strategy-specific counters when the buffer exposes versioned state.
    if callable(state_getter):
        replay_state = dict(state_getter())
        # Samples and RNG are already stored in their established archive and
        # manifest fields. The remaining metadata describes insertion policy.
        replay_state.pop("items", None)
        replay_state.pop("rng_state", None)
        replay_state.pop("maxlen", None)
        replay_metadata["strategy_state"] = replay_state

    return replay_metadata


def _read_replay_archive(
    task_dir: Path,
    metadata: Mapping[str, object] | None
) -> dict[str, object] | None:
    """Load numeric replay arrays and combine them with their checkpoint metadata.

    The archive name must be local to the task directory, pickle loading is disabled,
    and both arrays must match the declared count. Per-item array values are copied
    so returned items own their mutable data independently of the loaded arrays.

    Args:
        task_dir (Path): Existing committed checkpoint directory containing the NPZ.
        metadata (Mapping[str, object] | None): Manifest fields ``path``, ``count``,
            ``maxlen``, and ``rng_state`` with optional ``strategy_state``. None
            represents a checkpoint without replay storage.

    Returns:
        dict[str, object] | None: ``items`` as sample-label pairs, ``maxlen``,
        ``rng_state``, and optional ``strategy_state``; None when metadata is absent.

    Raises:
        ValueError: If the archive name is unsafe, row counts differ from metadata,
            or NumPy cannot load the archive without pickle.
        OSError: If the replay archive cannot be opened or read.
    """

    # Return no replay state when the checkpoint has no replay metadata.
    if metadata is None:
        return None

    archive_name = str(metadata["path"])

    # Reject replay archive names that escape the checkpoint directory.
    if Path(archive_name).name != archive_name:
        raise ValueError("Replay archive path must be a local filename.")
    with np.load(task_dir / archive_name, allow_pickle=False) as archive:
        x_values = np.asarray(archive["x"])
        y_values = np.asarray(archive["y"])

    expected_count = int(metadata["count"])

    # Require both replay arrays to match the manifest's sample count.
    if len(x_values) != expected_count or len(y_values) != expected_count:
        raise ValueError("Replay archive count does not match its metadata.")

    items = []
    for index in range(expected_count):
        x_item = x_values[index]
        y_item = y_values[index]
        # Copy array-valued samples; keep immutable scalar samples as stored.
        x_item = x_item.copy() if hasattr(x_item, "copy") else x_item
        # Copy array-valued labels; keep immutable scalar labels as stored.
        y_item = y_item.copy() if hasattr(y_item, "copy") else y_item
        items.append((x_item, y_item))

    replay_state = {
        "items": items,
        "maxlen": metadata["maxlen"],
        "rng_state": metadata["rng_state"],
    }
    # Checkpoints written before replay strategies existed omit this field and
    # are interpreted as the historical FIFO state during restoration.
    if "strategy_state" in metadata:
        replay_state["strategy_state"] = metadata["strategy_state"]
    return replay_state


def restore_replay_buffer(
    replay_buffer: object,
    replay_state: Mapping[str, object]
) -> object:
    """Restore retained items and private RNG/strategy state into an existing buffer.

    Modern buffers receive one complete load_state_dict call, preserving reservoir
    counters without reinserting saved items. Older FIFO-compatible buffers use their
    clear/extend/private-RNG interface. Saved capacity must match the target; an old
    FIFO-only interface cannot restore a non-FIFO strategy checkpoint.

    Args:
        replay_buffer (object): Target ReplayBuffer or compatible legacy FIFO object.
            Modern targets expose load_state_dict; legacy targets require clear,
            extend, maxlen, and a private RNG supporting setstate.
        replay_state (Mapping[str, object]): Decoded ``items``, ``maxlen``, and
            ``rng_state`` fields plus optional ``strategy_state``. Missing strategy
            metadata is interpreted as schema-version-1 FIFO state.

    Returns:
        object: The same replay_buffer object after in-place restoration.

    Raises:
        TypeError: If the target lacks a required interface or cannot restore the
            checkpoint's insertion strategy.
        ValueError: If capacity, strategy metadata, counters, items, or RNG state
            are incompatible with the target buffer.
    """

    strategy_state = replay_state.get("strategy_state")
    state_loader = getattr(replay_buffer, "load_state_dict", None)

    # New replay buffers restore counters directly rather than replaying saved
    # items through a reservoir policy, which would change their probabilities.
    if callable(state_loader):
        # Use saved strategy metadata when present; interpret older checkpoints as FIFO.
        combined_state = {
            "maxlen": replay_state["maxlen"],
            "items": replay_state["items"],
            "rng_state": replay_state["rng_state"],
            **(dict(strategy_state) if strategy_state is not None else {
                "schema_version": 1,
                "strategy": "fifo",
                "items_seen": len(replay_state["items"]),
                "classes": []
            })
        }
        state_loader(combined_state)

        return replay_buffer

    # Reject non-FIFO recovery when the target buffer lacks strategy-state loading.
    if strategy_state is not None \
    and str(strategy_state.get("strategy", "fifo")) != "fifo":
        raise TypeError(
            "This replay buffer cannot restore a non-FIFO strategy checkpoint."
        )
    # Require the legacy mutation and RNG interface before changing the buffer.
    if not hasattr(replay_buffer, "clear") \
    or not hasattr(replay_buffer, "extend") \
    or not hasattr(replay_buffer, "_rng"):
        raise TypeError(
            "replay_buffer must expose clear, extend, maxlen, and _rng."
        )
    # Require matching capacity before restoring a legacy replay buffer.
    if getattr(replay_buffer, "maxlen", None) != replay_state["maxlen"]:
        raise ValueError("Replay-buffer capacity differs from the checkpoint.")

    replay_buffer.clear()
    replay_buffer.extend(replay_state["items"])
    replay_buffer._rng.setstate(replay_state["rng_state"])
    return replay_buffer


def _task_directory_name(task_index: int) -> str:
    """Format a zero-based task index as a canonical sortable checkpoint directory name.

    At least four decimal digits are retained, with larger indices growing beyond
    four digits. The enclosing checkpoint APIs validate that the index is usable;
    this formatting helper does not inspect or create directories.

    Args:
        task_index (int): Zero-based completed-task index, such as 0 or 12.

    Returns:
        str: ``"task-0000"`` for zero, ``"task-0012"`` for twelve, and the
        corresponding zero-padded name for other indices.
    """

    return f"task-{task_index:04d}"


def save_task_checkpoint(
    checkpoint_root: str | os.PathLike[str],
    completed_task_index: int,
    state: Mapping[str, object],
    trackables: Mapping[str, object] | None = None,
    *,
    class_order: Sequence[object] | None = None,
    task_groups: Sequence[Sequence[object]] | None = None,
    rng_state: Mapping[str, object] | None = None,
    replay_buffer: object | None = None,
    fingerprint: str | None = None
) -> Path:
    """Atomically commit one completed continual-learning task.

    The primary API is ``save_task_checkpoint(root, task_index, state,
    trackables)``. ``state`` must contain the already-resolved ``class_order``
    and ``task_groups`` unless those are supplied explicitly as keywords.  They
    are materialized schedule values, not stochastic schedule options. A
    committed task directory is immutable; an existing directory with the same
    task index raises ``FileExistsError``.

    Creates the root if necessary, writes manifest/replay/TensorFlow payloads
    in a unique temporary directory, seals their checksums, and renames the
    directory into place before updating ``latest.json``. The input state and
    live RNGs are not mutated. A failure while updating the latest pointer can
    leave a valid committed task discoverable by directory scanning.

    Args:
        checkpoint_root (str | os.PathLike[str]): Root checkpoint directory.
        completed_task_index (int): Zero-based index of the completed task.
        state (Mapping[str, object]): Experiment state supported by the recovery
            JSON codec. Reserved cursor, schedule, RNG, and fingerprint entries
            are stored in schema fields; other entries become experiment state.
        trackables (Mapping[str, object] | None): Named TensorFlow models,
            optimizers, or other checkpoint dependencies. Defaults to ``None``;
            ``None`` or an empty mapping omits the TensorFlow payload.
        class_order (Sequence[object] | None): Optional explicit class order.
            Defaults to ``None``, reading the required value from ``state``.
        task_groups (Sequence[Sequence[object]] | None): Optional explicit task
            grouping whose flattened order must match ``class_order``.
            Defaults to ``None``, reading the required grouping from ``state``.
        rng_state (Mapping[str, object] | None): Snapshot from
            ``capture_rng_state``. Defaults to ``None``, using
            ``state.get("rng_state")`` or an empty mapping when absent. This
            function does not capture the current RNGs automatically.
        replay_buffer (object | None): Buffer exposing the replay ``state_dict``
            protocol. Defaults to ``None``, omitting the replay archive.
        fingerprint (str | None): Immutable run fingerprint to persist.
            Defaults to ``None``, reading ``state.get("fingerprint")``;
            if that is also absent, no run fingerprint is recorded.

    Returns:
        Path: Newly committed ``task-NNNN`` directory under ``checkpoint_root``.

    Raises:
        TypeError: If state, dependencies, fingerprint, or encoded values have
            unsupported types.
        ValueError: If the schedule, task cursor, dependency names, or replay
            state is invalid.
        FileExistsError: If the completed task directory already exists.
        OSError: If checkpoint payloads or commit metadata cannot be written.
    """

    # Reject checkpoint state that is not a mapping.
    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping.")

    state = dict(state)
    # Prefer an explicitly supplied class order; otherwise read it from state.
    class_order = state.get("class_order") if class_order is None else class_order
    # Prefer explicit task groups; otherwise read them from state.
    task_groups = state.get("task_groups") if task_groups is None else task_groups

    # Require a resolved class order and task grouping before saving.
    if class_order is None or task_groups is None:
        raise ValueError(
            "state must contain resolved class_order and task_groups."
        )

    # Use RNG state embedded in experiment state when no separate snapshot was supplied.
    if rng_state is None:
        rng_state = state.get("rng_state")

    # Use the embedded run fingerprint when no explicit fingerprint was supplied.
    if fingerprint is None:
        fingerprint = state.get("fingerprint")

    # Cursor, schedule, and RNG are represented once in schema-owned fields.
    experiment_state = dict(state)
    for reserved_name in (
        "completed_task_index",
        "next_task_index",
        "class_order",
        "task_groups",
        "rng_state",
        "fingerprint"
    ):
        experiment_state.pop(reserved_name, None)

    normalized_order, normalized_groups = _validate_schedule(
        completed_task_index,
        class_order,
        task_groups
    )
    normalized_trackables = _validate_trackables(trackables)

    # Reject non-string run fingerprints while allowing an omitted fingerprint.
    if fingerprint is not None and not isinstance(fingerprint, str):
        raise TypeError("fingerprint must be a string or None.")

    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / _task_directory_name(int(completed_task_index))

    # Refuse to replace an already committed task directory.
    if target.exists():
        raise FileExistsError(f"Task checkpoint already exists: {target}")

    temporary = root / (
        "." + target.name + ".tmp-" + uuid.uuid4().hex
    )
    temporary.mkdir()

    try:
        checkpoint_prefix = None
        # Write a TensorFlow checkpoint when model or optimizer dependencies exist.
        if normalized_trackables:
            tf_directory = temporary / _TF_DIRECTORY_NAME
            tf_directory.mkdir()
            checkpoint = tf.train.Checkpoint(**normalized_trackables)
            written_prefix = checkpoint.write(
                str(tf_directory / _TF_PREFIX_NAME)
            )
            checkpoint_prefix = Path(written_prefix).relative_to(
                temporary
            ).as_posix()

        replay_metadata = None
        # Write replay samples only when the run has a replay buffer.
        if replay_buffer is not None:
            replay_metadata = _write_replay_archive(temporary, replay_buffer)

        # Hash every external payload before the manifest is sealed.
        payload_files = {}
        for path in sorted(temporary.rglob("*")):
            # Hash files only, excluding temporary subdirectory entries from the payload manifest.
            if path.is_file():
                relative = path.relative_to(temporary).as_posix()
                payload_files[relative] = _sha256_file(path)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "completed_task_index": int(completed_task_index),
            "next_task_index": int(completed_task_index) + 1,
            "class_order": normalized_order,
            "task_groups": normalized_groups,
            "schedule_fingerprint": fingerprint_state({
                "class_order": normalized_order,
                "task_groups": normalized_groups
            }),
            "fingerprint": fingerprint,
            "trackable_names": sorted(normalized_trackables),
            "checkpoint_prefix": checkpoint_prefix,
            "experiment_state": experiment_state,
            "rng_state": dict(rng_state or {}),
            "replay": replay_metadata,
            "payload_sha256": payload_files
        }
        state_path = temporary / _STATE_NAME
        _write_json(state_path, manifest)
        state_sha256 = _sha256_file(state_path)

        # This marker is intentionally the final file written inside the temp
        # directory.  The directory rename then makes the whole task visible.
        _write_json(
            temporary / _COMMITTED_NAME,
            {
                "schema_version": SCHEMA_VERSION,
                "state_sha256": state_sha256
            }
        )
        os.replace(str(temporary), str(target))

        latest = {
            "schema_version": SCHEMA_VERSION,
            "completed_task_index": int(completed_task_index),
            "task_dir": target.name,
            "state_sha256": state_sha256
        }
        latest_temporary = root / (
            "." + _LATEST_NAME + ".tmp-" + uuid.uuid4().hex
        )
        _write_json(latest_temporary, latest)
        os.replace(str(latest_temporary), str(root / _LATEST_NAME))

        return target
    except BaseException:
        # Clean only the unique directory created by this call.  A previously
        # committed task is never removed or replaced.
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _validate_committed_task(task_dir: Path) -> dict[str, object]:
    """Authenticate a committed task directory and return its decoded manifest.

    Validation covers the directory/commit marker, state checksum and schema,
    completed/next cursors, exact resolved schedule, dependency names, TensorFlow
    prefix and shard presence, replay declaration, and exact payload file set.
    Every declared payload must remain local, regular, and checksum-matching. Files
    are read for validation but never loaded into a live TensorFlow object graph.

    Args:
        task_dir (Path): Candidate ``task-NNNN`` directory containing state.json,
            COMMITTED, and any declared TensorFlow/replay payloads.

    Returns:
        dict[str, object]: Decoded manifest with cursor, schedule, fingerprint,
        trackable/prefix declarations, experiment/RNG state, replay metadata, and
        payload hashes. No files or model/RNG objects are changed.

    Raises:
        ValueError: If commitment, schema, names, paths, checksums, or state
            invariants are invalid.
        OSError: If required checkpoint files cannot be read.
        KeyError: If a malformed manifest omits required schema fields.
    """

    match = _TASK_DIRECTORY_PATTERN.fullmatch(task_dir.name)

    # Reject paths that are not genuine task checkpoint directories.
    if match is None or not task_dir.is_dir() or task_dir.is_symlink():
        raise ValueError(f"Not a task checkpoint directory: {task_dir}")

    state_path = task_dir / _STATE_NAME
    committed_path = task_dir / _COMMITTED_NAME

    # Reject checkpoints missing either the state file or commit marker.
    if not state_path.is_file() or not committed_path.is_file():
        raise ValueError(f"Task checkpoint is not committed: {task_dir}")

    committed = _read_json(committed_path)

    # Reject commit markers written with an unsupported schema version.
    if int(committed.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Unsupported COMMITTED schema version.")

    state_sha256 = _sha256_file(state_path)

    # Reject state files whose checksum disagrees with the commit marker.
    if committed.get("state_sha256") != state_sha256:
        raise ValueError("Task checkpoint state checksum is invalid.")

    manifest = _read_json(state_path)

    # Reject checkpoint manifests written with an unsupported schema version.
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Unsupported task-checkpoint schema version.")

    directory_task_index = int(match.group(1))

    # Require the manifest's completed task to match the directory name.
    if int(manifest["completed_task_index"]) != directory_task_index:
        raise ValueError("Task directory index differs from its manifest.")

    # Require the next-task cursor to follow the completed task exactly.
    if int(manifest["next_task_index"]) != directory_task_index + 1:
        raise ValueError("Task cursor is inconsistent in the manifest.")

    normalized_order, normalized_groups = _validate_schedule(
        directory_task_index,
        manifest["class_order"],
        manifest["task_groups"]
    )
    schedule_fingerprint = fingerprint_state({
        "class_order": normalized_order,
        "task_groups": normalized_groups
    })

    # Reject a schedule whose fingerprint disagrees with its recorded contents.
    if manifest.get("schedule_fingerprint") != schedule_fingerprint:
        raise ValueError("Task schedule fingerprint is invalid.")

    # Validate the object-graph declaration before a caller is allowed to use
    # this directory as a recovery boundary.  In particular, a marker plus a
    # hand-written/minimal manifest must never be mistaken for a TensorFlow
    # checkpoint whose index or data shard was only partly written.
    trackable_names = manifest.get("trackable_names")

    # Reject malformed, duplicate, or invalid TensorFlow dependency names.
    if not isinstance(trackable_names, list) or any(
        not isinstance(name, str)
        or _TRACKABLE_NAME_PATTERN.fullmatch(name) is None
        for name in trackable_names
    ) or len(set(trackable_names)) != len(trackable_names):
        raise ValueError("Task checkpoint trackable names are invalid.")

    # Require dependency names in their canonical sorted order.
    if trackable_names != sorted(trackable_names):
        raise ValueError(
            "Task checkpoint trackable names are not canonical."
        )

    checkpoint_prefix = manifest.get("checkpoint_prefix")

    # Validate the TensorFlow prefix when the checkpoint declares one.
    if checkpoint_prefix is not None:
        # Reject TensorFlow prefixes that are not strings.
        if not isinstance(checkpoint_prefix, str):
            raise ValueError("TensorFlow checkpoint prefix is invalid.")
        prefix_path = Path(checkpoint_prefix)
        # Reject TensorFlow prefixes that escape the task directory.
        if prefix_path.is_absolute() or ".." in prefix_path.parts:
            raise ValueError("TensorFlow checkpoint prefix escapes task directory.")

    # Require TensorFlow dependencies and a checkpoint prefix to be declared together.
    if bool(trackable_names) != (checkpoint_prefix is not None):
        raise ValueError(
            "TensorFlow checkpoint prefix and trackable declaration disagree."
        )

    # Require experiment state to use the serializer's mapping representation.
    if not isinstance(manifest.get("experiment_state"), dict):
        raise ValueError("Task checkpoint experiment state must be a mapping.")

    # Require RNG state to use the serializer's mapping representation.
    if not isinstance(manifest.get("rng_state"), dict):
        raise ValueError("Task checkpoint RNG state must be a mapping.")

    # Reject a non-string run fingerprint while allowing an absent fingerprint.
    if manifest.get("fingerprint") is not None \
    and not isinstance(manifest.get("fingerprint"), str):
        raise ValueError("Task checkpoint run fingerprint is invalid.")

    payload_hashes = manifest.get("payload_sha256")
    # Reject malformed payload names or SHA-256 entries in the file manifest.
    if not isinstance(payload_hashes, dict) or any(
        not isinstance(relative, str) or not isinstance(expected_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
        for relative, expected_hash in payload_hashes.items()
    ):
        raise ValueError("Task checkpoint payload manifest is invalid.")

    # The manifest is closed over the exact set of external payloads.  This
    # catches both deleted shards and unlisted partial/foreign files instead of
    # accepting whichever subset happens to remain on disk.
    # Collect files only when comparing the actual and declared payload sets.
    actual_files = {
        path.relative_to(task_dir).as_posix()
        for path in task_dir.rglob("*")
        if path.is_file()
    }
    actual_payloads = actual_files - {_STATE_NAME, _COMMITTED_NAME}

    # Reject missing payload files and unlisted extra files.
    if set(payload_hashes) != actual_payloads:
        raise ValueError("Task checkpoint payload set differs from its manifest.")

    # Check TensorFlow shard completeness when a checkpoint prefix exists.
    if checkpoint_prefix is not None:
        index_name = checkpoint_prefix + ".index"
        data_prefix = checkpoint_prefix + ".data-"
        # Reject TensorFlow checkpoints missing an index or data shard.
        if index_name not in payload_hashes or not any(
            relative.startswith(data_prefix) for relative in payload_hashes
        ):
            raise ValueError("TensorFlow checkpoint is missing an index or data shard.")

    replay = manifest.get("replay")
    # Validate replay metadata when a replay archive is declared.
    if replay is not None:
        # Reject malformed replay metadata or a non-string replay path.
        if not isinstance(replay, dict) or not isinstance(replay.get("path"), str):
            raise ValueError("Replay checkpoint metadata is invalid.")
        replay_name = replay["path"]
        # Require the replay archive to be a declared local payload file.
        if Path(replay_name).name != replay_name or replay_name not in payload_hashes:
            raise ValueError("Replay checkpoint payload is missing or unsafe.")

    for relative, expected_hash in payload_hashes.items():
        relative_path = Path(relative)
        # Reject payload names that escape the task directory.
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Checkpoint payload path escapes its task directory.")
        payload_path = task_dir / relative_path
        # Reject symlinked, missing, or checksum-mismatched payload files.
        if payload_path.is_symlink() or not payload_path.is_file() \
        or _sha256_file(payload_path) != expected_hash:
            raise ValueError(
                f"Task checkpoint payload checksum is invalid: {relative}"
            )

    return manifest


def find_latest_task_checkpoint(
    checkpoint_path: str | os.PathLike[str]
) -> Path:
    """Resolve a task directory or find the newest valid committed child.

    A stale/corrupt ``latest.json`` never hides a valid older checkpoint.
    Corrupt or incomplete task directories are ignored during fallback scanning.
    Args:
        checkpoint_path (str | os.PathLike[str]): Task directory, checkpoint
            root, or ``latest.json`` path to resolve.

    Returns:
        Path: Newest valid committed task directory.
    """

    supplied = Path(checkpoint_path)
    # Treat an explicit latest.json path as a request to search its parent directory.
    if supplied.name == _LATEST_NAME and supplied.is_file():
        supplied = supplied.parent

    # Validate and return a directly supplied task directory.
    if _TASK_DIRECTORY_PATTERN.fullmatch(supplied.name):
        _validate_committed_task(supplied)
        return supplied

    # Reject checkpoint roots that do not exist as directories.
    if not supplied.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {supplied}")

    latest_path = supplied / _LATEST_NAME
    # Try the latest-task index first when an index file exists.
    if latest_path.is_file():
        try:
            latest = _read_json(latest_path)
            # Reject unsupported index schemas and fall back to directory discovery.
            if int(latest.get("schema_version", -1)) != SCHEMA_VERSION:
                raise ValueError("Unsupported latest-index schema version.")

            child_name = str(latest["task_dir"])

            # Reject unsafe or malformed task paths in the latest-task index.
            if Path(child_name).name != child_name \
            or _TASK_DIRECTORY_PATTERN.fullmatch(child_name) is None:
                raise ValueError("latest.json contains an unsafe task path.")

            candidate = supplied / child_name
            manifest = _validate_committed_task(candidate)

            # Reject index entries whose task cursor disagrees with the checkpoint manifest.
            if int(manifest["completed_task_index"]) \
            != int(latest["completed_task_index"]):
                raise ValueError("latest.json task index is inconsistent.")

            # Reject index entries whose recorded state checksum is stale or corrupt.
            if _sha256_file(candidate / _STATE_NAME) != latest["state_sha256"]:
                raise ValueError("latest.json state checksum is inconsistent.")
            # A crash can commit a newer task before updating latest.json.
            if not any(
                _TASK_DIRECTORY_PATTERN.fullmatch(child.name)
                and int(child.name[5:]) > int(latest["completed_task_index"])
                for child in supplied.iterdir()
            ):
                return candidate
        except Exception:
            # Scan below. latest.json is an optimization, not the source of
            # truth for whether a checkpoint is committed.
            pass

    candidates = []
    for child in supplied.iterdir():
        match = _TASK_DIRECTORY_PATTERN.fullmatch(child.name)

        # Consider only directories whose names identify completed task slots.
        if match is not None and child.is_dir():
            candidates.append((int(match.group(1)), child))

    for _, candidate in sorted(candidates, reverse=True):
        try:
            _validate_committed_task(candidate)
            return candidate
        except Exception:
            continue

    raise FileNotFoundError(
        f"No valid committed task checkpoint exists below: {supplied}"
    )


def load_task_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    trackables: Mapping[str, object] | None = None,
    expected_class_order: Sequence[object] | None = None,
    expected_task_groups: Sequence[Sequence[object]] | None = None,
    expected_fingerprint: str | None = None,
    assert_consumed: bool = True
) -> TaskCheckpoint:
    """Inspect and optionally restore the newest committed task checkpoint.

    Callers that need saved topology before model construction may first call
    this function without ``trackables``, construct/register all optimizer slot
    variables, and call it again with the complete mapping.  Under TensorFlow
    2.10 legacy optimizers normally require ``_create_all_weights(var_list)``
    before ``assert_consumed=True`` restoration.

    Validates commitment and payload checksums before optional TensorFlow
    restoration. Supplying dependencies changes their live variables; a later
    restore assertion or replay-decoding failure does not roll those changes
    back. RNG and replay state are returned for explicit restoration with
    ``restore_rng_state`` and ``restore_replay_buffer``.

    Args:
        checkpoint_path (str | os.PathLike[str]): Checkpoint root, committed
            task directory, or ``latest.json`` path. A root resolves to its
            newest valid committed task.
        trackables (Mapping[str, object] | None): Optional TensorFlow objects to
            restore, keyed by exactly the saved dependency names. Defaults to
            ``None`` for inspection without live-object restoration. An empty
            mapping is rejected when the checkpoint has TensorFlow dependencies.
        expected_class_order (Sequence[object] | None): Optional required class
            order. Defaults to ``None``, disabling schedule matching when
            ``expected_task_groups`` is also ``None``; supply both together.
        expected_task_groups (Sequence[Sequence[object]] | None): Optional
            required task grouping. Defaults to ``None``; when provided,
            ``expected_class_order`` must also be supplied and the complete
            resolved schedule must match the saved schedule.
        expected_fingerprint (str | None): Optional required run fingerprint.
            Defaults to ``None``, skipping run-fingerprint matching.
        assert_consumed (bool): Whether to require every checkpoint value to be
            consumed by the supplied object graph. Defaults to ``True``.
            ``False`` still requires every supplied existing object to match,
            while permitting unmatched saved values. Ignored during inspection.

    Returns:
        TaskCheckpoint: Resolved directory, completed/next task indices, tuple
        schedule, experiment/RNG mappings, optional replay state/fingerprint,
        and TensorFlow restore status (``None`` for inspection).

    Raises:
        FileNotFoundError: If no valid committed task can be resolved.
        ValueError: If the checkpoint, requested schedule/fingerprint, dependency
            names, or replay payload is incompatible or malformed.
        TypeError: If supplied dependencies do not meet the mapping protocol.
        AssertionError: If TensorFlow restore matching fails under the selected
            ``assert_consumed`` mode.
        OSError: If checkpoint files cannot be read.
    """

    task_dir = find_latest_task_checkpoint(checkpoint_path)
    manifest = _validate_committed_task(task_dir)
    class_order = tuple(manifest["class_order"])
    task_groups = tuple(tuple(group) for group in manifest["task_groups"])

    # Require expected class order and task grouping together.
    if (expected_class_order is None) != (expected_task_groups is None):
        raise ValueError(
            "Expected class order and task groups must be supplied together."
        )

    # Validate the expected schedule when the caller supplies one.
    if expected_class_order is not None:
        _, expected_groups = _validate_schedule(
            int(manifest["completed_task_index"]),
            expected_class_order,
            expected_task_groups
        )
        expected_order = _decode_json(_encode_json(list(expected_class_order)))

        # Reject restoration when the requested schedule differs from the saved schedule.
        if fingerprint_state({
            "class_order": expected_order,
            "task_groups": expected_groups
        }) != manifest["schedule_fingerprint"]:
            raise ValueError("Requested continual schedule differs from checkpoint.")

    # Reject restoration when an expected run fingerprint does not match.
    if expected_fingerprint is not None \
    and manifest.get("fingerprint") != expected_fingerprint:
        raise ValueError("Run fingerprint differs from the checkpoint.")

    normalized_trackables = _validate_trackables(trackables)
    saved_trackable_names = set(manifest.get("trackable_names", []))
    restore_status = None

    # Restore TensorFlow state when the caller supplies dependencies.
    if normalized_trackables:
        # Require supplied TensorFlow dependency names to match the saved object graph.
        if set(normalized_trackables) != saved_trackable_names:
            raise ValueError(
                "TensorFlow trackable names differ from the checkpoint: "
                f"saved={sorted(saved_trackable_names)}, "
                f"supplied={sorted(normalized_trackables)}."
            )

        prefix = manifest.get("checkpoint_prefix")

        # Reject a TensorFlow restore with no usable checkpoint prefix.
        if not isinstance(prefix, str):
            raise ValueError("Checkpoint manifest has no TensorFlow prefix.")

        prefix_path = Path(prefix)

        # Reject a TensorFlow restore prefix that escapes the task directory.
        if prefix_path.is_absolute() or ".." in prefix_path.parts:
            raise ValueError("TensorFlow checkpoint prefix escapes task directory.")

        checkpoint = tf.train.Checkpoint(**normalized_trackables)
        restore_status = checkpoint.read(str(task_dir / prefix_path))

        # Require every saved value to be consumed for strict restoration.
        if assert_consumed:
            restore_status.assert_consumed()
        # For partial restoration, still require every supplied object to match.
        else:
            restore_status.assert_existing_objects_matched()
    # Reject explicitly empty dependencies when the checkpoint contains TensorFlow state.
    elif saved_trackable_names and trackables is not None:
        raise ValueError("The checkpoint requires nonempty TensorFlow trackables.")

    replay_state = _read_replay_archive(task_dir, manifest.get("replay"))

    return TaskCheckpoint(
        task_dir=task_dir,
        completed_task_index=int(manifest["completed_task_index"]),
        next_task_index=int(manifest["next_task_index"]),
        class_order=class_order,
        task_groups=task_groups,
        experiment_state=dict(manifest.get("experiment_state", {})),
        rng_state=dict(manifest.get("rng_state", {})),
        replay_state=replay_state,
        fingerprint=manifest.get("fingerprint"),
        restore_status=restore_status
    )


__all__ = [
    "SCHEMA_VERSION",
    "TaskCheckpoint",
    "capture_rng_state",
    "find_latest_task_checkpoint",
    "fingerprint_state",
    "load_task_checkpoint",
    "restore_replay_buffer",
    "restore_rng_state",
    "save_task_checkpoint"
]
