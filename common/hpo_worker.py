"""Train one saved HPO configuration in a fresh TensorFlow process.

The coordinator owns Optuna. This executable knows only YAML input, a JSON
result, and a parent-liveness pipe; it never reads or writes study storage.
TensorFlow imports occur after memory/thread settings and the liveness watcher.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import sys
import threading
import traceback


def _watch_parent(ready: threading.Event) -> None:
    """Acknowledge startup, then exit when the coordinator's pipe closes."""

    try:
        # Only the launcher can authorize TensorFlow startup after Popen returns.
        if os.read(sys.stdin.fileno(), 1) != b"\x01":
            return
        ready.set()
        while os.read(sys.stdin.fileno(), 1):
            pass
    finally:
        os._exit(1)


def _json_value(value: object) -> object:
    """Convert metric trees to JSON without transporting runtime/model objects.

    Python JSON preserves nonfinite float values so the coordinator's existing
    final-objective guard can recognize and prune them. NumPy/TensorFlow values
    are reduced through their public conversion methods, without importing
    either package merely to serialize ordinary Python metrics.
    """

    # Ordinary scalar results need no conversion.
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    # Artifact paths cross the process boundary as plain strings.
    if isinstance(value, Path):
        return str(value)
    # Recursive metric dictionaries may contain NumPy or TensorFlow leaves.
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    # Histories and vector evaluations retain their original ordering.
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    # Eager TensorFlow tensors expose NumPy conversion without serialization.
    if callable(getattr(value, "numpy", None)):
        return _json_value(value.numpy())
    # NumPy arrays and scalars expose JSON-compatible Python containers.
    if callable(getattr(value, "tolist", None)):
        return _json_value(value.tolist())
    raise TypeError(f"Unsupported worker metric type: {type(value).__name__}")


def _write_result(output_path: Path, payload: dict[str, object]) -> None:
    """Atomically publish one complete JSON envelope beside its temporary file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(_json_value(payload), indent=2), encoding="utf-8")
    temporary.replace(output_path)


def run_worker(
    config_path: Path,
    output_path: Path,
    *,
    gpu_memory_limit_mb: float | None = None,
    threads: int = 1,
) -> int:
    """Load, train, and publish one trial; return zero only for a completed run.

    All GPU configuration happens before importing the training pipeline or
    creating any TensorFlow tensors. TrainingDiverged becomes ``pruned``;
    ResourceExhaustedError becomes ``oom``; other exceptions become ``error``.
    Tracebacks remain in stdout/stderr, captured in the trial's worker log.
    """

    config_path = Path(config_path).resolve()
    output_path = Path(output_path).resolve()
    payload: dict[str, object] = {
        "status": "error",
        "config_path": str(config_path),
        "results_path": None,
        "history": {},
        "evaluations": {},
        "error": None,
        "divergence": None,
        "divergence_path": None,
    }
    tensorflow = None
    training_diverged = None
    config = None
    try:
        # Direct callers receive the same argument validation as the launcher.
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer.")
        # Invalid memory caps must fail before TensorFlow initializes a device.
        if gpu_memory_limit_mb is not None and (
            isinstance(gpu_memory_limit_mb, bool)
            or not isinstance(gpu_memory_limit_mb, (int, float))
            or not math.isfinite(gpu_memory_limit_mb)
            or gpu_memory_limit_mb <= 0
        ):
            raise ValueError("gpu_memory_limit_mb must be a positive finite number or None.")
        os.environ.update({
            "TF_NUM_INTRAOP_THREADS": str(threads),
            "TF_NUM_INTEROP_THREADS": str(threads),
            "OMP_NUM_THREADS": str(threads),
            "MPLBACKEND": "Agg",
            "TF_FORCE_GPU_ALLOW_GROWTH": "true" if gpu_memory_limit_mb is None else "false",
        })
        import tensorflow as tensorflow

        tensorflow.config.threading.set_intra_op_parallelism_threads(threads)
        tensorflow.config.threading.set_inter_op_parallelism_threads(threads)
        for device in tensorflow.config.list_physical_devices("GPU"):
            # Growth keeps each worker from reserving all free device memory.
            if gpu_memory_limit_mb is None:
                tensorflow.config.experimental.set_memory_growth(device, True)
            # Logical-device caps give each worker an explicit upper bound.
            else:
                tensorflow.config.set_logical_device_configuration(
                    device,
                    [tensorflow.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_limit_mb)],
                )
        from common.callbacks.hpo_guard import TrainingDiverged
        from common.config import load_config, save_config
        from common.train import main as train

        training_diverged = TrainingDiverged
        config = load_config(config_path)
        result = train(config)
        results_path = Path(result["results_path"]).resolve()
        resolved_config_path = results_path / "config.yaml"
        save_config(config, resolved_config_path)
        payload.update({
            "status": "complete",
            "config_path": str(resolved_config_path),
            "results_path": str(results_path),
            "history": result["history"],
            "evaluations": result["evaluations"],
        })
        # Force conversion inside the guarded block so unsupported data is an error.
        payload = _json_value(payload)
    except Exception as error:
        traceback.print_exc()
        payload.update({
            "status": "error", "error": f"{type(error).__name__}: {error}",
            "history": {}, "evaluations": {},
        })
        # The pipeline may have created an artifact directory before failing.
        if config is not None:
            payload["results_path"] = str(config.training.results_path)
        # Numeric divergence is a scientific pruning decision, not a crash.
        if training_diverged is not None and isinstance(error, training_diverged):
            payload.update({
                "status": "pruned",
                "divergence": error.evidence,
                "divergence_path": str(error.evidence_path) if error.evidence_path is not None else None,
            })
        # A resource limit failure is kept separate for actionable diagnostics.
        elif tensorflow is not None and isinstance(error, tensorflow.errors.ResourceExhaustedError):
            payload["status"] = "oom"
    _write_result(output_path, payload)
    return 0 if payload["status"] == "complete" else 1


def main() -> int:
    """Parse the worker CLI and install its optional parent-liveness watcher."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-memory-limit-mb", type=float, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--watch-parent", action="store_true")
    arguments = parser.parse_args()
    # Only launcher-managed workers treat stdin EOF as coordinator death.
    if arguments.watch_parent:
        ready = threading.Event()
        threading.Thread(target=_watch_parent, args=(ready,), daemon=True, name="hpo-parent-watch").start()
        # An interrupted Popen constructor may retain an otherwise orphaned pipe.
        if not ready.wait(timeout=10.0):
            print("HPO coordinator did not acknowledge worker startup.", file=sys.stderr)
            return 1
    return run_worker(
        arguments.config,
        arguments.output,
        gpu_memory_limit_mb=arguments.gpu_memory_limit_mb,
        threads=arguments.threads,
    )


# Importing the transport module must never start training or a watcher thread.
if __name__ == "__main__":
    raise SystemExit(main())
