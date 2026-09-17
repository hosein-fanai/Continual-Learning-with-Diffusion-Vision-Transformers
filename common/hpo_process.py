"""Small process and lock helpers for a single Optuna coordinator.

Only the coordinator touches study storage. Training workers receive saved YAML
and return JSON, keeping TensorFlow runtime state out of the notebook process.
This module deliberately imports only the Python standard library.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import BinaryIO, Iterable, Iterator


@contextmanager
def study_lock(study_root: Path, *, blocking: bool = False) -> Iterator[None]:
    """Hold an OS lock until the coordinator exits or crashes.

    The persistent lock file is never removed: removing it could let another
    process lock a different inode while the original coordinator still runs.
    Independent study directories can run concurrently. Network file systems
    must support the host's advisory file locks. Coordinators reject contention
    by default; cache preparation uses ``blocking=True`` to wait for its owner.
    """

    study_root = Path(study_root)
    study_root.mkdir(parents=True, exist_ok=True)
    with (study_root / ".coordinator.lock").open("a+b") as lock_file:
        # Windows locks one existing byte; writing the same byte is harmless.
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            # Initialize the byte once without truncating a held lock file.
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            while True:
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    # Permission/device errors must not become infinite retries.
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    # Study coordinators fail promptly instead of waiting.
                    if not blocking:
                        raise RuntimeError(
                            f"Another HPO coordinator already holds the study lock: {study_root}"
                        ) from error
                    time.sleep(0.1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        # POSIX flock is released automatically when its owning process dies.
        else:
            import fcntl

            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except OSError as error:
                # Only genuine lock contention is reported as another coordinator.
                if error.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                raise RuntimeError(
                    f"Another HPO coordinator already holds the study lock: {study_root}"
                ) from error
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def dataset_load_lock(dataset_name: str) -> Iterator[None]:
    """Serialize CIFAR download/extraction/decoding before concurrent training.

    A separate blocking lock coordinates simultaneous notebook studies for the
    same user and dataset. The lock is deliberately broader than KERAS_HOME so
    read-only-cache fallbacks need no private Keras API. Training workers then
    train concurrently once their loader leaves this context. Keras can extract
    an existing archive on every call, so warm caches require the same lock.
    """

    # This helper is deliberately limited to the supported parallel profile.
    if dataset_name not in ("cifar10", "cifar100"):
        raise ValueError("Parallel dataset loading supports cifar10 and cifar100 only.")

    identity = os.path.normcase(str(Path.home().resolve())) + ":" + dataset_name
    cache_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    lock_root = Path(tempfile.gettempdir()) / "continual-learning-hpo-cache" / cache_key
    with study_lock(lock_root, blocking=True):
        yield


@dataclass
class WorkerHandle:
    """A training child and the streams/files owned by its coordinator."""

    process: subprocess.Popen[bytes]
    output_path: Path
    log_path: Path
    log_file: BinaryIO


def start_worker(
    config_path: Path,
    output_path: Path,
    log_path: Path,
    *,
    gpu_memory_limit_mb: float | None,
    threads: int = 1,
) -> WorkerHandle:
    """Launch one isolated training process, logging both output streams.

    GPU visibility is inherited. Each visible GPU uses memory growth unless an
    explicit per-worker MB cap is supplied. Parent stdin remains open solely as
    a startup/liveness pipe; EOF makes the worker exit after a coordinator crash.
    Invalid settings and launch errors propagate without leaving open files.
    """

    # Reject booleans and fractional thread counts before launching anything.
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer.")
    # TensorFlow caps must be positive, finite numbers and never booleans.
    if gpu_memory_limit_mb is not None and (
        isinstance(gpu_memory_limit_mb, bool)
        or not isinstance(gpu_memory_limit_mb, (int, float))
        or not math.isfinite(gpu_memory_limit_mb)
        or gpu_memory_limit_mb <= 0
    ):
        raise ValueError("gpu_memory_limit_mb must be a positive finite number or None.")
    config_path = Path(config_path).resolve()
    output_path = Path(output_path).resolve()
    log_path = Path(log_path).resolve()
    # Fail before creating artifacts when the coordinator supplied no config.
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    # Distinct paths prevent a log or result from overwriting the trial input.
    if len({config_path, output_path, log_path}) != 3:
        raise ValueError("Worker config, output, and log paths must be distinct.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment.update({
        "TF_NUM_INTRAOP_THREADS": str(threads),
        "TF_NUM_INTEROP_THREADS": str(threads),
        "OMP_NUM_THREADS": str(threads),
        "MPLBACKEND": "Agg",
        "PYTHONUNBUFFERED": "1",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true" if gpu_memory_limit_mb is None else "false",
    })
    command = [
        sys.executable, "-u", "-m", "common.hpo_worker",
        "--config", str(config_path), "--output", str(output_path),
        "--threads", str(threads), "--watch-parent",
    ]
    # An omitted cap explicitly selects growth instead of a logical GPU limit.
    if gpu_memory_limit_mb is not None:
        command.extend(("--gpu-memory-limit-mb", str(gpu_memory_limit_mb)))
    log_file = log_path.open("wb")
    process = None
    try:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            close_fds=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        # A child must not train if interruption prevents Popen from returning.
        process.stdin.write(b"\x01")
        process.stdin.flush()
        return WorkerHandle(process, output_path, log_path, log_file)
    except BaseException:
        # An interruption after process creation must not orphan its child.
        if process is not None:
            stop_workers([WorkerHandle(process, output_path, log_path, log_file)])
        # A launch failure owns only the log file.
        else:
            log_file.close()
        raise


def _close_worker_streams(handle: WorkerHandle) -> None:
    """Close coordinator-owned descriptors after a child is reaped or stopped."""

    try:
        # Test doubles and failed launches need not expose a liveness pipe.
        if handle.process.stdin is not None:
            handle.process.stdin.close()
    except OSError:
        pass
    finally:
        try:
            handle.log_file.close()
        except OSError:
            pass


def finish_worker(handle: WorkerHandle) -> dict[str, object]:
    """Read and validate a finished worker's result and close its streams.

    A missing/malformed result, a successful payload from a crashed process, or
    an unsupported status raises RuntimeError with the worker log location.
    Training failures with valid error payloads are returned to the coordinator.
    """

    exit_code = handle.process.poll()
    # Reading a live process could observe an old or incomplete result.
    if exit_code is None:
        raise RuntimeError("Cannot finish a worker that is still running.")
    try:
        payload = json.loads(handle.output_path.read_text(encoding="utf-8"))
        # Only the versionless, explicit result envelope is accepted.
        if not isinstance(payload, dict) or payload.get("status") not in {
            "complete", "pruned", "oom", "error"
        }:
            raise ValueError("Missing or unsupported worker status.")
        # Every outcome identifies the saved config used for reproducibility.
        if not isinstance(payload.get("config_path"), str) or not payload["config_path"]:
            raise ValueError("Worker config_path must be a nonempty string.")
        # Completed results must contain the metric structures, never a model.
        if payload["status"] == "complete":
            # A process crashing after publishing success has not finished safely.
            if exit_code != 0:
                raise ValueError(f"Worker published success but exited with code {exit_code}.")
            # The main pipeline reports a dictionary of history and evaluations.
            if not isinstance(payload.get("history"), (dict, list)) or not isinstance(
                payload.get("evaluations"), dict
            ):
                raise ValueError("Worker history/evaluations are missing or malformed.")
            # HPO always configures a concrete artifact directory.
            if not isinstance(payload.get("results_path"), str) or not payload["results_path"]:
                raise ValueError("Worker results_path must be a nonempty string.")
        # Failure payloads must explain their cause rather than silently pass.
        elif not isinstance(payload.get("error"), str) or not payload["error"]:
            raise ValueError("Worker failure has no error message.")
        # Divergence carries the guard's evidence for reproducible pruning.
        if payload["status"] == "pruned" and not isinstance(payload.get("divergence"), dict):
            raise ValueError("Pruned worker has no divergence evidence.")
        return payload
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(
            f"Worker result unavailable or invalid (exit {exit_code}); "
            f"see {handle.log_path}: {error}"
        ) from error
    finally:
        _close_worker_streams(handle)


def stop_workers(handles: Iterable[WorkerHandle]) -> None:
    """Terminate all children together, then kill stragglers and close files.

    Cleanup uses one five-second grace period for the group instead of waiting
    serially for every worker. A second interrupt does not skip later children.
    Calling this again after cleanup is harmless.
    """

    handles = list(handles)
    for handle in handles:
        try:
            # Completed processes only need their descriptors closed below.
            if handle.process.poll() is None:
                handle.process.terminate()
        except (OSError, KeyboardInterrupt):
            pass
    deadline = time.monotonic() + 5.0
    for handle in handles:
        try:
            handle.process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except (subprocess.TimeoutExpired, OSError, KeyboardInterrupt):
            try:
                handle.process.kill()
                handle.process.wait(timeout=5.0)
            except (subprocess.TimeoutExpired, OSError, KeyboardInterrupt):
                pass
        finally:
            _close_worker_streams(handle)
