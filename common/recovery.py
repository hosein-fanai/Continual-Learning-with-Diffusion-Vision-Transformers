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
    """Decoded state from one committed task checkpoint.

    Attributes:
        task_dir: Concrete committed checkpoint directory.
        completed_task_index: Zero-based index of the last completed task.
        next_task_index: First task that still needs to run.
        class_order: Authoritative resolved class order for the run.
        task_groups: Authoritative resolved groups introduced per task.
        experiment_state: Caller-owned serializable metrics and orchestration
            state.
        rng_state: State returned by :func:`capture_rng_state`, when supplied.
        replay_state: Decoded replay samples, capacity, and private RNG state;
            ``None`` when the run has no replay buffer.
        fingerprint: Optional caller-defined run/configuration fingerprint.
        restore_status: TensorFlow checkpoint restore status, or ``None`` when
            the checkpoint was inspected without trackables.
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
    """Encode supported Python/NumPy values into strict JSON data.

    Args:
        value (object): Supported value to encode.

    Returns:
        object: Strictly JSON-compatible encoded value.
    """

    # Select the recovery action required by this condition.
    if value is None or isinstance(value, (bool, str, int)):
        return value

    # Select the recovery action required by this condition.
    if isinstance(value, float):
        # Strict JSON cannot represent non-finite IEEE values.
        if math.isnan(value):
            return {"__recovery_type__": "float", "value": "nan"}

        # Select the recovery action required by this condition.
        if math.isinf(value):
            return {
                "__recovery_type__": "float",
                "value": "inf" if value > 0 else "-inf"
            }

        return value

    # Select the recovery action required by this condition.
    if isinstance(value, np.generic):
        return _encode_json(value.item())

    # Select the recovery action required by this condition.
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)

        # Select the recovery action required by this condition.
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

    # Select the recovery action required by this condition.
    if isinstance(value, Path):
        return {
            "__recovery_type__": "path",
            "value": str(value)
        }

    # Select the recovery action required by this condition.
    if isinstance(value, bytes):
        return {
            "__recovery_type__": "bytes",
            "data": base64.b64encode(value).decode("ascii")
        }

    # Select the recovery action required by this condition.
    if isinstance(value, tuple):
        return {
            "__recovery_type__": "tuple",
            "items": [_encode_json(item) for item in value]
        }

    # Select the recovery action required by this condition.
    if isinstance(value, set):
        encoded_items = [_encode_json(item) for item in value]
        encoded_items.sort(key=_stable_json_dumps)

        return {
            "__recovery_type__": "set",
            "items": encoded_items
        }

    # Select the recovery action required by this condition.
    if isinstance(value, list):
        return [_encode_json(item) for item in value]

    # Select the recovery action required by this condition.
    if isinstance(value, Mapping):
        encoded = {}
        for key, item in value.items():
            # Select the recovery action required by this condition.
            if not isinstance(key, str):
                raise TypeError("Recovery mapping keys must be strings.")

            encoded[key] = _encode_json(item)

        return encoded

    raise TypeError(
        "Unsupported recovery value type: " + type(value).__qualname__
    )


def _decode_json(value: object) -> object:
    """Decode values emitted by :func:`_encode_json`.

    Args:
        value (object): JSON-compatible encoded value.

    Returns:
        object: Decoded Python or NumPy value.
    """

    # Select the recovery action required by this condition.
    if isinstance(value, list):
        return [_decode_json(item) for item in value]

    # Select the recovery action required by this condition.
    if not isinstance(value, dict):
        return value

    type_name = value.get("__recovery_type__")

    # Select the recovery action required by this condition.
    if type_name is None:
        return {
            key: _decode_json(item)
            for key, item in value.items()
        }

    # Select the recovery action required by this condition.
    if type_name == "float":
        return {
            "nan": float("nan"),
            "inf": float("inf"),
            "-inf": float("-inf")
        }[value["value"]]

    # Select the recovery action required by this condition.
    if type_name == "ndarray":
        dtype = np.dtype(value["dtype"])
        shape = tuple(int(size) for size in value["shape"])
        raw = base64.b64decode(value["data"].encode("ascii"))
        expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize

        # Select the recovery action required by this condition.
        if len(raw) != expected_size:
            raise ValueError(
                "Encoded ndarray byte length does not match its shape."
            )

        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()

    # Select the recovery action required by this condition.
    if type_name == "path":
        return Path(value["value"])

    # Select the recovery action required by this condition.
    if type_name == "bytes":
        return base64.b64decode(value["data"].encode("ascii"))

    # Select the recovery action required by this condition.
    if type_name == "tuple":
        return tuple(_decode_json(item) for item in value["items"])

    # Select the recovery action required by this condition.
    if type_name == "set":
        return set(_decode_json(item) for item in value["items"])

    raise ValueError(f"Unknown recovery JSON type tag: {type_name!r}.")


def _stable_json_dumps(value: object) -> str:
    """Return a deterministic, strict JSON representation.

    Args:
        value (object): JSON-compatible value to serialize.

    Returns:
        str: Canonically ordered compact JSON text.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":")
    )


def fingerprint_state(value: object) -> str:
    """Return a stable SHA-256 fingerprint for serializable state.

    Args:
        value (object): Supported recovery state to fingerprint.

    Returns:
        str: Lowercase hexadecimal SHA-256 digest.
    """

    encoded = _encode_json(value)

    return hashlib.sha256(
        _stable_json_dumps(encoded).encode("utf-8")
    ).hexdigest()


def _qualified_name(value: object) -> str:
    """Return a stable module-qualified type or callable name."""

    candidate = value if isinstance(value, type) else type(value)
    if callable(value) and not isinstance(value, type):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)

        if module is not None and qualname is not None:
            return f"{module}.{qualname}"

    return f"{candidate.__module__}.{candidate.__qualname__}"


def _array_recovery_descriptor(value: object) -> dict[str, object] | None:
    """Describe array content compactly for compatibility checks."""

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
    """Hash a model template without binding it to a filesystem location."""

    if not path:
        return None

    artifact = Path(path)

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
    """Hash configured initial weights that may seed future tasks."""

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
    """Convert configuration objects to stable, JSON-safe descriptions."""

    if value is None or isinstance(value, (bool, str, int)):
        return value

    # Canonicalize only float32-backed Keras configuration round trips.
    if isinstance(value, float):
        # Keras may materialize the same constructor scalar through float32 on
        # a compiled object (for example 0.9 -> 0.899999976). Seven significant
        # digits preserve float32 semantics while keeping the descriptor stable.
        return float(format(value, ".7g")) if strip_config_names else value

    if isinstance(value, np.generic):
        return _recovery_descriptor(
            value.item(), active_ids, strip_config_names
        )

    if isinstance(value, np.ndarray):
        return _array_recovery_descriptor(value)

    # Record TensorFlow dtypes by their portable registered name.
    if isinstance(value, tf.dtypes.DType):
        return {"type": "tensorflow.DType", "name": value.name}

    if isinstance(value, tf.TensorShape):
        return {"type": "tensorflow.TensorShape", "shape": value.as_list()}

    active_ids = set() if active_ids is None else active_ids
    object_id = id(value)
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

        if callable(value):
            return {"callable": _qualified_name(value)}

        # Unknown runtime objects contribute their stable type, not transient
        # instance identity. All supported Keras objects expose get_config().
        return {"type": _qualified_name(value)}
    finally:
        active_ids.remove(object_id)


def _model_topology_descriptor(model: object) -> dict[str, object] | None:
    """Describe a Keras topology without including mutable values."""

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
    """Describe model and optimizer structures before strict restore."""

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
    """Resolve persistent depth additions made by one progressive task."""

    stage_tasks = fit_kwargs.get("stage_tasks")
    depths = fit_kwargs.get("depths")

    if stage_tasks == "depths_only":
        return list(depths or [])

    # Timestep- and resolution-only curricula never alter persistent topology.
    if stage_tasks in ("timesteps_only", "resolutions_only"):
        return []

    if not isinstance(stage_tasks, Sequence) or isinstance(stage_tasks, str):
        return []

    resolved = []
    for stage_index, task in enumerate(stage_tasks):
        has_depth = False
        depth_spec = None

        if task == "depth":
            has_depth = True
        elif isinstance(task, dict) and "depth" in task:
            has_depth = True
            depth_spec = task["depth"]
        elif isinstance(task, (set, frozenset)) and "depth" in task:
            has_depth = True
        elif isinstance(task, (tuple, list)) and len(task) == 2 \
        and task[0] == "depth":
            has_depth = True
            depth_spec = task[1]

        if not has_depth:
            continue

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
    """Write strict JSON and flush it before returning.

    Args:
        path (Path): Destination JSON path.
        value (object): Supported value to encode and write.

    Returns:
        None.
    """

    encoded = _encode_json(value)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(_stable_json_dumps(encoded))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> object:
    """Read and decode one recovery JSON file.

    Args:
        path (Path): Recovery JSON path to read.

    Returns:
        object: Decoded recovery value.
    """

    with path.open("r", encoding="utf-8") as stream:
        return _decode_json(json.load(stream))


def _sha256_file(path: Path) -> str:
    """Hash one file without loading it fully into memory.

    Args:
        path (Path): File whose bytes should be hashed.

    Returns:
        str: Lowercase hexadecimal SHA-256 digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            # Select the recovery action required by this condition.
            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _validate_schedule(
    completed_task_index: int,
    class_order: Sequence[object],
    task_groups: Sequence[Sequence[object]]
) -> tuple[list[object], list[list[object]]]:
    """Normalize and validate an authoritative resolved task schedule.

    Args:
        completed_task_index (int): Zero-based completed-task cursor.
        class_order (Sequence[object]): Resolved class introduction order.
        task_groups (Sequence[Sequence[object]]): Resolved classes per task.

    Returns:
        tuple[list[object], list[list[object]]]: Portable normalized class order
        and task groups.
    """

    # Select the recovery action required by this condition.
    if isinstance(completed_task_index, bool) or not isinstance(
        completed_task_index, (int, np.integer)
    ):
        raise TypeError("completed_task_index must be an integer.")

    completed_task_index = int(completed_task_index)

    normalized_order = list(class_order)
    normalized_groups = [list(group) for group in task_groups]

    # Select the recovery action required by this condition.
    if not normalized_order:
        raise ValueError("class_order must not be empty.")

    # Select the recovery action required by this condition.
    if not normalized_groups or any(not group for group in normalized_groups):
        raise ValueError("task_groups must contain only nonempty groups.")

    # Select the recovery action required by this condition.
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

    # Select the recovery action required by this condition.
    if encoded_flattened != encoded_order:
        raise ValueError("Flattening task_groups must equal class_order exactly.")

    # Select the recovery action required by this condition.
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
    """Capture Python, NumPy, and optional TensorFlow generator state.

    TensorFlow's legacy stateful ``tf.random.*`` operation counters and a live
    ``tf.data`` iterator are not representable by this JSON snapshot.  Exact
    task-boundary recovery should therefore reseed every incomplete task from a
    derived task seed.  An explicit ``tf.random.Generator`` is trackable and may
    also be passed here for a portable diagnostic snapshot.
    Args:
        numpy_generator (np.random.Generator | None): Optional local NumPy
            generator to snapshot.
        python_rng (random.Random | None): Optional local Python RNG to snapshot.
        include_globals (bool): Whether to capture Python and NumPy global RNGs.
        tensorflow_generator (object | None): Optional TensorFlow generator to
            snapshot.
        include_tensorflow_global (bool): Whether to resolve and capture the
            TensorFlow global generator.

    Returns:
        dict[str, object]: Portable RNG-state snapshot.
    """

    state: dict[str, object] = {"schema_version": SCHEMA_VERSION}

    # Select the recovery action required by this condition.
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

    # Select the recovery action required by this condition.
    if python_rng is not None:
        state["python_local"] = python_rng.getstate()

    # Select the recovery action required by this condition.
    if numpy_generator is not None:
        state["numpy_local"] = {
            "bit_generator": type(numpy_generator.bit_generator).__name__,
            "state": numpy_generator.bit_generator.state
        }

    # Select the recovery action required by this condition.
    if include_tensorflow_global:
        experimental = getattr(tf.random, "experimental", None)
        getter = getattr(experimental, "get_global_generator", None)

        # Select the recovery action required by this condition.
        if getter is None:
            raise RuntimeError(
                "This TensorFlow version has no global Generator API."
            )

        tensorflow_generator = getter()

    # Select the recovery action required by this condition.
    if tensorflow_generator is not None:
        generator_state = np.asarray(tensorflow_generator.state.numpy())
        algorithm = tensorflow_generator.algorithm
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
    """Restore a snapshot from :func:`capture_rng_state`.

    Missing local generator objects are constructed and returned.  Supplied
    objects are restored in place so existing learner/buffer references remain
    valid.
    Args:
        state (Mapping[str, object]): Snapshot from :func:`capture_rng_state`.
        numpy_generator (np.random.Generator | None): Optional NumPy generator
            to restore in place.
        python_rng (random.Random | None): Optional Python RNG to restore in
            place.
        restore_globals (bool): Whether to restore Python and NumPy globals.
        tensorflow_generator (object | None): Optional TensorFlow generator to
            restore in place.
        restore_tensorflow_global (bool): Whether to install the restored
            TensorFlow generator as the global generator.

    Returns:
        dict[str, object]: Local generators that were restored or constructed.
    """

    # Select the recovery action required by this condition.
    if int(state.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Unsupported RNG-state schema version.")

    restored: dict[str, object] = {}

    # Select the recovery action required by this condition.
    if restore_globals and "python_global" in state:
        random.setstate(state["python_global"])

    # Select the recovery action required by this condition.
    if restore_globals and "numpy_global" in state:
        numpy_state = state["numpy_global"]
        np.random.set_state((
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"])
        ))

    # Select the recovery action required by this condition.
    if "python_local" in state:
        python_rng = random.Random() if python_rng is None else python_rng
        python_rng.setstate(state["python_local"])
        restored["python_rng"] = python_rng

    # Select the recovery action required by this condition.
    if "numpy_local" in state:
        numpy_state = state["numpy_local"]
        bit_generator_name = str(numpy_state["bit_generator"])

        # Select the recovery action required by this condition.
        if numpy_generator is None:
            bit_generator_type = getattr(np.random, bit_generator_name, None)
            # Select the recovery action required by this condition.
            if bit_generator_type is None:
                raise ValueError(
                    f"NumPy has no bit generator named {bit_generator_name!r}."
                )

            numpy_generator = np.random.Generator(bit_generator_type())
        # Select the recovery action required by this condition.
        elif type(numpy_generator.bit_generator).__name__ != bit_generator_name:
            raise ValueError(
                "Supplied NumPy generator uses a different bit-generator type."
            )

        numpy_generator.bit_generator.state = numpy_state["state"]
        restored["numpy_generator"] = numpy_generator

    # Select the recovery action required by this condition.
    if "tensorflow_generator" in state:
        tf_state = state["tensorflow_generator"]
        algorithm = int(tf_state["algorithm"])
        values = tf.convert_to_tensor(tf_state["state"], dtype=tf.int64)
        # Select the recovery action required by this condition.
        if tensorflow_generator is None:
            tensorflow_generator = tf.random.Generator.from_state(
                values,
                alg=algorithm
            )
        # Handle the complementary recovery case.
        else:
            current_algorithm = tensorflow_generator.algorithm
            current_algorithm = int(current_algorithm.numpy()) if hasattr(current_algorithm, "numpy") \
                                else int(current_algorithm)
            # Select the recovery action required by this condition.
            if current_algorithm != algorithm:
                raise ValueError(
                    "Supplied TensorFlow generator uses a different algorithm."
                )

            tensorflow_generator.state.assign(values)

        restored["tensorflow_generator"] = tensorflow_generator

        # Select the recovery action required by this condition.
        if restore_tensorflow_global:
            experimental = getattr(tf.random, "experimental", None)
            setter = getattr(experimental, "set_global_generator", None)

            # Select the recovery action required by this condition.
            if setter is None:
                raise RuntimeError(
                    "This TensorFlow version has no global Generator API."
                )

            setter(tensorflow_generator)

    return restored


def _validate_trackables(
    trackables: Mapping[str, object] | None,
) -> dict[str, object]:
    """Normalize checkpoint objects and reject unstable dependency names.

    Args:
        trackables (Mapping[str, object] | None): Named TensorFlow checkpoint
            dependencies.

    Returns:
        dict[str, object]: Non-null trackables with validated names.
    """

    normalized = {
        name: value
        for name, value in dict(trackables or {}).items()
        if value is not None
    }
    for name in normalized:
        # Select the recovery action required by this condition.
        if not isinstance(name, str) or not _TRACKABLE_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"Invalid TensorFlow checkpoint dependency name: {name!r}."
            )

    return normalized


def _write_replay_archive(
    task_dir: Path,
    replay_buffer: object
) -> dict[str, object]:
    """Write a homogeneous replay buffer to a non-pickle NumPy archive.

    Args:
        task_dir (Path): Temporary task-checkpoint directory.
        replay_buffer (object): Replay buffer exposing samples, capacity, and
            private RNG state.

    Returns:
        dict[str, object]: Replay archive metadata for the task manifest.
    """

    # Select the recovery action required by this condition.
    if not hasattr(replay_buffer, "buffer") \
    or not hasattr(replay_buffer, "maxlen") \
    or not hasattr(replay_buffer, "_rng"):
        raise TypeError(
            "replay_buffer must expose buffer, maxlen, and a private _rng."
        )

    items = list(replay_buffer.buffer)

    # Select the recovery action required by this condition.
    if any(not isinstance(item, (tuple, list)) or len(item) != 2 for item in items):
        raise TypeError("Every replay-buffer item must be an (x, y) pair.")

    # Select the recovery action required by this condition.
    if items:
        try:
            x_values = np.stack([np.asarray(item[0]) for item in items])
            y_values = np.stack([np.asarray(item[1]) for item in items])
        except ValueError as error:
            raise ValueError(
                "Replay-buffer x and y items must have homogeneous shapes."
            ) from error
    # Handle the complementary recovery case.
    else:
        # Dtypes/shapes are immaterial for an empty buffer and become defined by
        # the first post-resume insertion.
        x_values = np.empty((0,), dtype=np.float32)
        y_values = np.empty((0,), dtype=np.uint8)

    # Select the recovery action required by this condition.
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
    """Read replay samples and metadata from one committed task directory.

    Args:
        task_dir (Path): Committed task-checkpoint directory.
        metadata (Mapping[str, object] | None): Replay manifest metadata.

    Returns:
        dict[str, object] | None: Decoded replay state, or ``None`` when absent.
    """

    # Select the recovery action required by this condition.
    if metadata is None:
        return None

    archive_name = str(metadata["path"])

    # Select the recovery action required by this condition.
    if Path(archive_name).name != archive_name:
        raise ValueError("Replay archive path must be a local filename.")
    with np.load(task_dir / archive_name, allow_pickle=False) as archive:
        x_values = np.asarray(archive["x"])
        y_values = np.asarray(archive["y"])

    expected_count = int(metadata["count"])

    # Select the recovery action required by this condition.
    if len(x_values) != expected_count or len(y_values) != expected_count:
        raise ValueError("Replay archive count does not match its metadata.")

    items = []
    for index in range(expected_count):
        x_item = x_values[index]
        y_item = y_values[index]
        x_item = x_item.copy() if hasattr(x_item, "copy") else x_item
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
    """Restore replay contents and private RNG state into an existing buffer.

    Args:
        replay_buffer (object): Existing replay buffer to mutate in place.
        replay_state (Mapping[str, object]): Decoded checkpoint replay state.

    Returns:
        object: The restored replay buffer.
    """

    strategy_state = replay_state.get("strategy_state")
    state_loader = getattr(replay_buffer, "load_state_dict", None)

    # New replay buffers restore counters directly rather than replaying saved
    # items through a reservoir policy, which would change their probabilities.
    if callable(state_loader):
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

    # Select the recovery action required by this condition.
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
    # Select the recovery action required by this condition.
    if getattr(replay_buffer, "maxlen", None) != replay_state["maxlen"]:
        raise ValueError("Replay-buffer capacity differs from the checkpoint.")

    replay_buffer.clear()
    replay_buffer.extend(replay_state["items"])
    replay_buffer._rng.setstate(replay_state["rng_state"])
    return replay_buffer


def _task_directory_name(task_index: int) -> str:
    """Return the canonical sortable directory name for one task.

    Args:
        task_index (int): Zero-based task index.

    Returns:
        str: Canonical sortable task-directory name.
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
    Args:
        checkpoint_root (str | os.PathLike[str]): Root checkpoint directory.
        completed_task_index (int): Zero-based index of the completed task.
        state (Mapping[str, object]): Caller-owned experiment and schedule state.
        trackables (Mapping[str, object] | None): Named TensorFlow dependencies.
        class_order (Sequence[object] | None): Optional explicit class order.
        task_groups (Sequence[Sequence[object]] | None): Optional explicit task
            grouping.
        rng_state (Mapping[str, object] | None): Optional RNG-state snapshot.
        replay_buffer (object | None): Optional replay buffer to archive.
        fingerprint (str | None): Optional immutable run fingerprint.

    Returns:
        Path: Newly committed task-checkpoint directory.
    """

    # Select the recovery action required by this condition.
    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping.")

    state = dict(state)
    class_order = state.get("class_order") if class_order is None else class_order
    task_groups = state.get("task_groups") if task_groups is None else task_groups

    # Select the recovery action required by this condition.
    if class_order is None or task_groups is None:
        raise ValueError(
            "state must contain resolved class_order and task_groups."
        )

    # Select the recovery action required by this condition.
    if rng_state is None:
        rng_state = state.get("rng_state")

    # Select the recovery action required by this condition.
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

    # Select the recovery action required by this condition.
    if fingerprint is not None and not isinstance(fingerprint, str):
        raise TypeError("fingerprint must be a string or None.")

    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / _task_directory_name(int(completed_task_index))

    # Select the recovery action required by this condition.
    if target.exists():
        raise FileExistsError(f"Task checkpoint already exists: {target}")

    temporary = root / (
        "." + target.name + ".tmp-" + uuid.uuid4().hex
    )
    temporary.mkdir()

    try:
        checkpoint_prefix = None
        # Select the recovery action required by this condition.
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
        # Select the recovery action required by this condition.
        if replay_buffer is not None:
            replay_metadata = _write_replay_archive(temporary, replay_buffer)

        # Hash every external payload before the manifest is sealed.
        payload_files = {}
        for path in sorted(temporary.rglob("*")):
            # Select the recovery action required by this condition.
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
    """Validate hashes/schema for one task directory and return its manifest.

    Args:
        task_dir (Path): Candidate committed task directory.

    Returns:
        dict[str, object]: Validated task manifest.
    """

    match = _TASK_DIRECTORY_PATTERN.fullmatch(task_dir.name)

    # Select the recovery action required by this condition.
    if match is None or not task_dir.is_dir() or task_dir.is_symlink():
        raise ValueError(f"Not a task checkpoint directory: {task_dir}")

    state_path = task_dir / _STATE_NAME
    committed_path = task_dir / _COMMITTED_NAME

    # Select the recovery action required by this condition.
    if not state_path.is_file() or not committed_path.is_file():
        raise ValueError(f"Task checkpoint is not committed: {task_dir}")

    committed = _read_json(committed_path)

    # Select the recovery action required by this condition.
    if int(committed.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Unsupported COMMITTED schema version.")

    state_sha256 = _sha256_file(state_path)

    # Select the recovery action required by this condition.
    if committed.get("state_sha256") != state_sha256:
        raise ValueError("Task checkpoint state checksum is invalid.")

    manifest = _read_json(state_path)

    # Select the recovery action required by this condition.
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Unsupported task-checkpoint schema version.")

    directory_task_index = int(match.group(1))

    # Select the recovery action required by this condition.
    if int(manifest["completed_task_index"]) != directory_task_index:
        raise ValueError("Task directory index differs from its manifest.")

    # Select the recovery action required by this condition.
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

    # Select the recovery action required by this condition.
    if manifest.get("schedule_fingerprint") != schedule_fingerprint:
        raise ValueError("Task schedule fingerprint is invalid.")

    # Validate the object-graph declaration before a caller is allowed to use
    # this directory as a recovery boundary.  In particular, a marker plus a
    # hand-written/minimal manifest must never be mistaken for a TensorFlow
    # checkpoint whose index or data shard was only partly written.
    trackable_names = manifest.get("trackable_names")

    # Select the recovery action required by this condition.
    if not isinstance(trackable_names, list) or any(
        not isinstance(name, str)
        or _TRACKABLE_NAME_PATTERN.fullmatch(name) is None
        for name in trackable_names
    ) or len(set(trackable_names)) != len(trackable_names):
        raise ValueError("Task checkpoint trackable names are invalid.")

    # Select the recovery action required by this condition.
    if trackable_names != sorted(trackable_names):
        raise ValueError(
            "Task checkpoint trackable names are not canonical."
        )

    checkpoint_prefix = manifest.get("checkpoint_prefix")

    # Select the recovery action required by this condition.
    if checkpoint_prefix is not None:
        # Select the recovery action required by this condition.
        if not isinstance(checkpoint_prefix, str):
            raise ValueError("TensorFlow checkpoint prefix is invalid.")
        prefix_path = Path(checkpoint_prefix)
        # Select the recovery action required by this condition.
        if prefix_path.is_absolute() or ".." in prefix_path.parts:
            raise ValueError("TensorFlow checkpoint prefix escapes task directory.")

    # Select the recovery action required by this condition.
    if bool(trackable_names) != (checkpoint_prefix is not None):
        raise ValueError(
            "TensorFlow checkpoint prefix and trackable declaration disagree."
        )

    # Select the recovery action required by this condition.
    if not isinstance(manifest.get("experiment_state"), dict):
        raise ValueError("Task checkpoint experiment state must be a mapping.")

    # Select the recovery action required by this condition.
    if not isinstance(manifest.get("rng_state"), dict):
        raise ValueError("Task checkpoint RNG state must be a mapping.")

    # Select the recovery action required by this condition.
    if manifest.get("fingerprint") is not None \
    and not isinstance(manifest.get("fingerprint"), str):
        raise ValueError("Task checkpoint run fingerprint is invalid.")

    payload_hashes = manifest.get("payload_sha256")
    # Select the recovery action required by this condition.
    if not isinstance(payload_hashes, dict) or any(
        not isinstance(relative, str) or not isinstance(expected_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
        for relative, expected_hash in payload_hashes.items()
    ):
        raise ValueError("Task checkpoint payload manifest is invalid.")

    # The manifest is closed over the exact set of external payloads.  This
    # catches both deleted shards and unlisted partial/foreign files instead of
    # accepting whichever subset happens to remain on disk.
    actual_files = {
        path.relative_to(task_dir).as_posix()
        for path in task_dir.rglob("*")
        if path.is_file()
    }
    actual_payloads = actual_files - {_STATE_NAME, _COMMITTED_NAME}

    # Select the recovery action required by this condition.
    if set(payload_hashes) != actual_payloads:
        raise ValueError("Task checkpoint payload set differs from its manifest.")

    # Select the recovery action required by this condition.
    if checkpoint_prefix is not None:
        index_name = checkpoint_prefix + ".index"
        data_prefix = checkpoint_prefix + ".data-"
        # Select the recovery action required by this condition.
        if index_name not in payload_hashes or not any(
            relative.startswith(data_prefix) for relative in payload_hashes
        ):
            raise ValueError("TensorFlow checkpoint is missing an index or data shard.")

    replay = manifest.get("replay")
    # Select the recovery action required by this condition.
    if replay is not None:
        # Select the recovery action required by this condition.
        if not isinstance(replay, dict) or not isinstance(replay.get("path"), str):
            raise ValueError("Replay checkpoint metadata is invalid.")
        replay_name = replay["path"]
        # Select the recovery action required by this condition.
        if Path(replay_name).name != replay_name or replay_name not in payload_hashes:
            raise ValueError("Replay checkpoint payload is missing or unsafe.")

    for relative, expected_hash in payload_hashes.items():
        relative_path = Path(relative)
        # Select the recovery action required by this condition.
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Checkpoint payload path escapes its task directory.")
        payload_path = task_dir / relative_path
        # Select the recovery action required by this condition.
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
    # Select the recovery action required by this condition.
    if supplied.name == _LATEST_NAME and supplied.is_file():
        supplied = supplied.parent

    # Select the recovery action required by this condition.
    if _TASK_DIRECTORY_PATTERN.fullmatch(supplied.name):
        _validate_committed_task(supplied)
        return supplied

    # Select the recovery action required by this condition.
    if not supplied.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {supplied}")

    latest_path = supplied / _LATEST_NAME
    # Select the recovery action required by this condition.
    if latest_path.is_file():
        try:
            latest = _read_json(latest_path)
            # Select the recovery action required by this condition.
            if int(latest.get("schema_version", -1)) != SCHEMA_VERSION:
                raise ValueError("Unsupported latest-index schema version.")

            child_name = str(latest["task_dir"])

            # Select the recovery action required by this condition.
            if Path(child_name).name != child_name \
            or _TASK_DIRECTORY_PATTERN.fullmatch(child_name) is None:
                raise ValueError("latest.json contains an unsafe task path.")

            candidate = supplied / child_name
            manifest = _validate_committed_task(candidate)

            # Select the recovery action required by this condition.
            if int(manifest["completed_task_index"]) \
            != int(latest["completed_task_index"]):
                raise ValueError("latest.json task index is inconsistent.")

            # Select the recovery action required by this condition.
            if _sha256_file(candidate / _STATE_NAME) != latest["state_sha256"]:
                raise ValueError("latest.json state checksum is inconsistent.")
            return candidate
        except Exception:
            # Scan below. latest.json is an optimization, not the source of
            # truth for whether a checkpoint is committed.
            pass

    candidates = []
    for child in supplied.iterdir():
        match = _TASK_DIRECTORY_PATTERN.fullmatch(child.name)

        # Select the recovery action required by this condition.
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
    Args:
        checkpoint_path (str | os.PathLike[str]): Checkpoint path to resolve.
        trackables (Mapping[str, object] | None): Optional TensorFlow objects to
            restore.
        expected_class_order (Sequence[object] | None): Optional required class
            order.
        expected_task_groups (Sequence[Sequence[object]] | None): Optional
            required task grouping.
        expected_fingerprint (str | None): Optional required run fingerprint.
        assert_consumed (bool): Whether to require every checkpoint value to be
            consumed by the supplied object graph.

    Returns:
        TaskCheckpoint: Decoded task state and TensorFlow restore status.
    """

    task_dir = find_latest_task_checkpoint(checkpoint_path)
    manifest = _validate_committed_task(task_dir)
    class_order = tuple(manifest["class_order"])
    task_groups = tuple(tuple(group) for group in manifest["task_groups"])

    # Select the recovery action required by this condition.
    if (expected_class_order is None) != (expected_task_groups is None):
        raise ValueError(
            "Expected class order and task groups must be supplied together."
        )

    # Select the recovery action required by this condition.
    if expected_class_order is not None:
        _, expected_groups = _validate_schedule(
            int(manifest["completed_task_index"]),
            expected_class_order,
            expected_task_groups
        )
        expected_order = _decode_json(_encode_json(list(expected_class_order)))

        # Select the recovery action required by this condition.
        if fingerprint_state({
            "class_order": expected_order,
            "task_groups": expected_groups
        }) != manifest["schedule_fingerprint"]:
            raise ValueError("Requested continual schedule differs from checkpoint.")

    # Select the recovery action required by this condition.
    if expected_fingerprint is not None \
    and manifest.get("fingerprint") != expected_fingerprint:
        raise ValueError("Run fingerprint differs from the checkpoint.")

    normalized_trackables = _validate_trackables(trackables)
    saved_trackable_names = set(manifest.get("trackable_names", []))
    restore_status = None

    # Select the recovery action required by this condition.
    if normalized_trackables:
        # Select the recovery action required by this condition.
        if set(normalized_trackables) != saved_trackable_names:
            raise ValueError(
                "TensorFlow trackable names differ from the checkpoint: "
                f"saved={sorted(saved_trackable_names)}, "
                f"supplied={sorted(normalized_trackables)}."
            )

        prefix = manifest.get("checkpoint_prefix")

        # Select the recovery action required by this condition.
        if not isinstance(prefix, str):
            raise ValueError("Checkpoint manifest has no TensorFlow prefix.")

        prefix_path = Path(prefix)

        # Select the recovery action required by this condition.
        if prefix_path.is_absolute() or ".." in prefix_path.parts:
            raise ValueError("TensorFlow checkpoint prefix escapes task directory.")

        checkpoint = tf.train.Checkpoint(**normalized_trackables)
        restore_status = checkpoint.read(str(task_dir / prefix_path))

        # Select the recovery action required by this condition.
        if assert_consumed:
            restore_status.assert_consumed()
        # Handle the complementary recovery case.
        else:
            restore_status.assert_existing_objects_matched()
    # Select the recovery action required by this condition.
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
