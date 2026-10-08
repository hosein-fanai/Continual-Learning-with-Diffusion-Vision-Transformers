"""Small process and lock helpers for a single Optuna coordinator.

Only the coordinator touches study storage. Training workers receive saved YAML
and return JSON, keeping TensorFlow runtime state out of the notebook process.
This module deliberately imports only the Python standard library.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack, contextmanager
from collections.abc import Sequence
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from numbers import Integral
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, BinaryIO, Callable, Iterable, Iterator

from common.hpo_pruning import (
    create_exchange, decision_path, read_report, validate_report, write_atomic_json
)


@contextmanager
def study_lock(study_root: Path, blocking: bool = False) -> Iterator[None]:
    """Hold an OS lock until the coordinator exits or crashes.

    The persistent lock file is never removed: removing it could let another
    process lock a different inode while the original coordinator still runs.
    Independent study directories can run concurrently. Network file systems
    must support the host's advisory file locks. Coordinators reject contention
    by default; cache preparation uses ``blocking=True`` to wait for its owner.

    Args:
        study_root (Path): Study directory, created along with its persistent lock file.
        blocking (bool): False raises on another owner; True waits for the OS lock.
            Defaults to ``False``.

    Yields:
        None: The exclusive coordinator lock is held until context exit.

    Raises:
        RuntimeError: A nonblocking request encounters another lock owner.
        OSError: Directory, lock-file, or operating-system lock access fails.
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

    Args:
        dataset_name (str): Exactly cifar10 or cifar100; included in the per-user
            temporary lock identity so different datasets can prepare concurrently.

    Yields:
        None: A blocking dataset lock is held for the enclosed loading operation.

    Raises:
        ValueError: The name is not a supported parallel-profile dataset.
        OSError: The temporary lock directory or OS lock is inaccessible.
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
    resource_stack: ExitStack | None = None
    pruning_exchange: dict[str, object] | None = None
    pruning_last_report: dict[str, object] | None = None
    pruning_last_decision: bool | None = None


def normalize_worker_gpu_ids(worker_gpu_ids: Sequence[int | str] | None) -> tuple[str, ...] | None:
    """Normalize an explicit ordered GPU selection without probing or initializing CUDA.

    Args:
        worker_gpu_ids (Sequence[int | str] | None): Distinct physical GPU indices
            or CUDA GPU/MIG identifiers. None retains inherited device visibility.

    Returns:
        tuple[str, ...] | None: Nonempty normalized selection or inherited visibility.

    Raises:
        ValueError: A selection is empty, duplicated, malformed, or not a sequence.
    """

    # An omitted selection preserves existing callers and CPU-only worker tests.
    if worker_gpu_ids is None:
        return None
    # A string is one selector, not the sequence accepted by the public API.
    if isinstance(worker_gpu_ids, (str, bytes)) or not isinstance(worker_gpu_ids, Sequence) or not worker_gpu_ids:
        raise ValueError("worker_gpu_ids must be a nonempty sequence of distinct GPU indices or UUIDs.")
    normalized = []
    for gpu_id in worker_gpu_ids:
        # Integer selectors address physical device indices before child startup.
        if isinstance(gpu_id, Integral) and not isinstance(gpu_id, bool) and gpu_id >= 0:
            selector = str(int(gpu_id))
        # CUDA UUIDs and numeric strings remain explicit single-device selectors.
        elif isinstance(gpu_id, str) and gpu_id and gpu_id == gpu_id.strip() and (
            gpu_id.isascii() and gpu_id.isdecimal()
            or gpu_id.startswith(("GPU-", "MIG-")) and len(gpu_id) > 4
            and all(character.isascii() and (character.isalnum() or character in "-/") for character in gpu_id)
        ):
            selector = str(int(gpu_id)) if gpu_id.isdecimal() else gpu_id
        # Reject device masks and hidden-device sentinels instead of changing routing.
        else:
            raise ValueError("Each worker_gpu_ids entry must select one GPU index or UUID.")
        # Aliased integer spellings cannot create duplicate capacity for one device.
        if selector in normalized:
            raise ValueError("worker_gpu_ids must contain distinct GPU selectors.")
        normalized.append(selector)
    return tuple(normalized)


def start_worker(
    config_path: Path, 
    output_path: Path, 
    log_path: Path, 
    gpu_memory_limit_mb: float | None, 
    threads: int = 1, 
    worker_context: Callable[[], AbstractContextManager[dict[str, Any]]] | None = None, 
    gpu_id: int | str | None = None, 
    gpu_worker_context: Callable[[str], AbstractContextManager[dict[str, Any]]] | None = None, 
    pruning_monitor: str | None = None, 
    pruning_trial_number: int | None = None
) -> WorkerHandle:
    """Launch one isolated training process, logging both output streams.

    GPU visibility is inherited unless gpu_id selects one GPU. Each device uses memory growth unless an
    explicit per-worker MB cap is supplied. Parent stdin remains open solely as
    a startup/liveness pipe; EOF makes the worker exit after a coordinator crash.
    Invalid settings and launch errors propagate without leaving open files.

    Args:
        config_path (Path): Existing saved YAML trial configuration.
        output_path (Path): JSON result destination; an old result is removed before
            launch. Must differ from the input and log paths.
        log_path (Path): Combined stdout/stderr destination, opened for replacement.
        gpu_memory_limit_mb (float | None): Positive finite per-visible-device cap;
            None enables TensorFlow memory growth. Device visibility is inherited.
        threads (int): Positive CPU intra/inter-op and OpenMP thread count.
            Defaults to ``1``.
        worker_context (Callable | None): Optional zero-argument context factory
            yielding an environment mapping and register(pid) callback. Registration
            precedes startup authorization; cleanup waits for the child to exit.
        gpu_id (int | str | None): Physical GPU index or CUDA UUID selected before
            TensorFlow imports. None inherits visibility for existing callers.
        gpu_worker_context (Callable | None): Admission factory receiving the
            normalized selected GPU string. Requires gpu_id and excludes
            worker_context; environment overrides must agree with the selection.
        pruning_monitor (str | None): Epoch metric reported to the coordinator;
            None preserves workers without performance pruning.
        pruning_trial_number (int | None): Owning Optuna trial number, required
            with pruning_monitor to bind reports to the coordinator trial.

    Returns:
        WorkerHandle: Child process, resolved output/log paths, and open log stream.
        The coordinator owns cleanup and the startup/liveness stdin pipe. The child
        runs from this checkout using the same Python executable as the coordinator.

    Raises:
        ValueError: Thread/memory settings or path separation are invalid.
        FileNotFoundError: The input configuration does not exist.
        OSError: Files or child creation fail; already created resources are closed.
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
    selected_gpu = None if gpu_id is None else normalize_worker_gpu_ids([gpu_id])[0]
    # Admission factories have distinct contracts and cannot both own one worker.
    if worker_context is not None and gpu_worker_context is not None:
        raise ValueError("worker_context and gpu_worker_context are mutually exclusive.")
    # GPU-aware admission cannot run without an explicit selected device.
    if gpu_worker_context is not None and (selected_gpu is None or not callable(gpu_worker_context)):
        raise ValueError("gpu_worker_context must be callable and requires gpu_id.")
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
        "TF_FORCE_GPU_ALLOW_GROWTH": "true" if gpu_memory_limit_mb is None else "false"
    })
    command = [
        sys.executable, "-u", "-m", "common.hpo_worker", 
        "--config", str(config_path), "--output", str(output_path), 
        "--threads", str(threads), "--watch-parent"
    ]
    # An omitted cap explicitly selects growth instead of a logical GPU limit.
    if gpu_memory_limit_mb is not None:
        command.extend(("--gpu-memory-limit-mb", str(gpu_memory_limit_mb)))
    resources = ExitStack()
    pruning_exchange = None
    environment.pop("HPO_PRUNING_EXCHANGE", None)
    resource = None
    log_file = None
    process = None
    try:
        # Only enabled trials receive a private token-bound pruning exchange.
        if pruning_monitor is not None:
            directory = resources.enter_context(tempfile.TemporaryDirectory(
                prefix=".hpo-pruning-", dir=output_path.parent
            ))
            pruning_exchange = create_exchange(Path(directory), pruning_monitor, pruning_trial_number)
        # An unpaired trial number is an invalid transport configuration.
        elif pruning_trial_number is not None:
            raise ValueError("pruning_trial_number requires pruning_monitor.")
        # Resource admission must precede child startup and remain held until exit.
        if worker_context is not None:
            resource = resources.enter_context(worker_context())
        # GPU-aware admission reserves the same device the scheduler selected.
        elif gpu_worker_context is not None:
            resource = resources.enter_context(gpu_worker_context(selected_gpu))
        # Resource hooks may add provenance but cannot redirect an explicit device.
        if resource is not None:
            reserved_gpu = resource["environment"].get("CUDA_VISIBLE_DEVICES")
            # A conflicting reservation cannot safely authorize this worker.
            if selected_gpu is not None and reserved_gpu is not None and reserved_gpu != selected_gpu:
                raise ValueError("Resource CUDA_VISIBLE_DEVICES does not match the selected worker GPU.")
            environment.update(resource["environment"])
        # Explicit routing takes effect before the child imports any framework.
        if selected_gpu is not None:
            environment["CUDA_VISIBLE_DEVICES"] = selected_gpu
            environment["HPO_WORKER_GPU_ID"] = selected_gpu
        # An inherited routing marker must not change the legacy transport contract.
        else:
            environment.pop("HPO_WORKER_GPU_ID", None)
        # Admission provenance cannot replace this launch's private pruning identity.
        environment.pop("HPO_PRUNING_EXCHANGE", None)
        # Only the coordinator-created exchange reaches the new child process.
        if pruning_exchange is not None:
            environment["HPO_PRUNING_EXCHANGE"] = json.dumps(pruning_exchange)
        log_file = log_path.open("wb")
        process = subprocess.Popen(
            command, 
            cwd=Path(__file__).resolve().parents[1], 
            env=environment, 
            stdin=subprocess.PIPE, 
            stdout=log_file, 
            stderr=subprocess.STDOUT, 
            close_fds=True, 
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        # Registration must finish before the worker receives training authorization.
        if resource is not None:
            resource["register"](process.pid)
        # A child must not train if interruption prevents Popen from returning.
        process.stdin.write(b"\x01")
        process.stdin.flush()
        return WorkerHandle(process, output_path, log_path, log_file, resources, pruning_exchange)
    except BaseException:
        # An interruption after process creation must not orphan its child.
        if process is not None:
            stop_workers([WorkerHandle(process, output_path, log_path, log_file, resources, pruning_exchange)])
        # A launch failure owns only the log file.
        else:
            # Resource entry or log creation can fail before a stream exists.
            if log_file is not None:
                log_file.close()
            resources.close()
        raise



def read_pruning_report(handle: WorkerHandle) -> dict[str, object] | None:
    """Return the next unanswered epoch report for this worker, without Optuna."""

    # Legacy workers never create a report channel or perform polling I/O.
    if handle.pruning_exchange is None:
        return None
    report = read_report(handle.pruning_exchange)
    # Atomic publication may not have happened before this scheduler poll.
    if report is None:
        return None
    previous = handle.pruning_last_report
    # Re-reading the exact last request is harmless after its answer was sent.
    if previous is not None and report == previous:
        return None
    # Another request cannot replay or move backwards through training epochs.
    if previous is not None and report["step"] <= previous["step"]:
        raise ValueError("HPO pruning epoch report is stale or replayed.")
    return report


def answer_pruning_report(handle: WorkerHandle, report: dict[str, object], prune: bool) -> None:
    """Atomically answer one pending epoch after the sole coordinator ranks it."""

    # Only a real pending request can receive a performance-pruning decision.
    if handle.pruning_exchange is None or not isinstance(prune, bool):
        raise ValueError("HPO pruning answer requires an exchange and a boolean decision.")
    report = validate_report(handle.pruning_exchange, report)
    pending = read_pruning_report(handle)
    # A changed report must not receive the previous epoch's or another trial's answer.
    if pending is None or pending != report:
        raise ValueError("HPO pruning answer does not match the pending epoch report.")
    write_atomic_json(decision_path(handle.pruning_exchange, report), {"report": report, "prune": prune})
    handle.pruning_last_report = report
    handle.pruning_last_decision = prune


def _close_worker_streams(handle: WorkerHandle) -> None:
    """Close coordinator-owned descriptors after a child is reaped or stopped.

    Args:
        handle (WorkerHandle): Finished/stopped child whose stdin and log the caller owns.

    Returns:
        None: Closes available streams; repeated close/I/O failures are tolerated.

    Raises:
        None.
    """

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


def _release_worker_resources(handle: WorkerHandle) -> None:
    """Release admission only after the caller has confirmed child termination."""

    # Handles from ordinary callers need no external resource cleanup.
    if handle.resource_stack is not None:
        handle.resource_stack.close()
        handle.resource_stack = None


def finish_worker(handle: WorkerHandle) -> dict[str, object]:
    """Read and validate a finished worker's result and close its streams.

    A missing/malformed result, a successful payload from a crashed process, or
    an unsupported status raises RuntimeError with the worker log location.
    Training failures with valid error payloads are returned to the coordinator.

    Args:
        handle (WorkerHandle): Worker whose process has already exited and whose
            output_path should contain the complete result envelope.

    Returns:
        dict[str, object]: Validated complete/pruned/oom/error payload. Successful
        results require exit zero plus history, evaluations, config and result paths.
        Failure payloads require an error; pruning requires divergence evidence
        or a matching coordinator decision and partial performance evidence.

    Raises:
        RuntimeError: The process is live or its result/exit status is inconsistent.
            Completed-process streams are closed even when result validation fails.
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
        # Numerical divergence and finite performance pruning keep separate evidence.
        if payload["status"] == "pruned" and not isinstance(payload.get("divergence"), dict):
            evidence = payload.get("pruning")
            # A child cannot claim performance pruning without the parent's decision.
            if not isinstance(evidence, dict) or handle.pruning_exchange is None \
                    or handle.pruning_last_decision is not True \
                    or evidence.get("reason") != "performance_pruning" \
                    or validate_report(handle.pruning_exchange, evidence.get("report")) != handle.pruning_last_report \
                    or not isinstance(evidence.get("partial_history"), list):
                raise ValueError("Pruned worker has no matching divergence or performance evidence.")
        return payload
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(
            f"Worker result unavailable or invalid (exit {exit_code}); "
            f"see {handle.log_path}: {error}"
        ) from error
    finally:
        _close_worker_streams(handle)
        _release_worker_resources(handle)


def stop_workers(handles: Iterable[WorkerHandle]) -> None:
    """Terminate all children together, then kill stragglers and close files.

    Cleanup uses one five-second grace period for the group instead of waiting
    serially for every worker. A second interrupt does not skip later children.
    Calling this again after cleanup is harmless.

    Args:
        handles (Iterable[WorkerHandle]): Coordinator-owned children to reap, consumed
            once into a list; completed children still have their streams closed.

    Returns:
        None: Attempts graceful group termination, force-kills remaining children,
        and closes descriptors. Cleanup tolerates OS errors and repeated interrupts.

    Raises:
        None.
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
        reaped = False
        try:
            handle.process.wait(timeout=max(0.0, deadline - time.monotonic()))
            reaped = True
        except (subprocess.TimeoutExpired, OSError, KeyboardInterrupt):
            try:
                handle.process.kill()
                handle.process.wait(timeout=5.0)
                reaped = True
            except (subprocess.TimeoutExpired, OSError, KeyboardInterrupt):
                pass
        finally:
            _close_worker_streams(handle)
            # A surviving child must retain its GPU reservation after failed cleanup.
            if reaped:
                _release_worker_resources(handle)
