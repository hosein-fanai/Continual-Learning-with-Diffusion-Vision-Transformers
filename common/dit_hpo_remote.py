"""Keep DiT HPO computation on admitted workers in the supplied GPU pool.

The notebook is a standard-library coordinator. It owns a resource lease while
one registered subprocess uses TensorFlow. Do not nest this helper inside the
campaign notebook runner, which already owns the notebook kernel's lease.
"""

from contextlib import contextmanager
import csv
import ctypes
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Iterator


CAMPAIGN = Path("files/results/joint_classifier_v1_audit_20261007")
POOL_PATH = CAMPAIGN / "runtime_routing_20261007/gpu_pool.json"
ALLOCATOR_PATH = CAMPAIGN / "support/remote_resource_slots.py"
TF_MEMORY_MIB = 12288
RESERVED_MEMORY_MIB = 13312
POLL_SECONDS = 15
VERIFIED_HOSTS = {
    "b917b7e3195e": "GPU-938d1ee4-8591-b274-4110-d469503b692c", 
    "ea77e5605c7a": "GPU-cb5492c6-af24-d4d8-864c-238f14012c80", 
    "c91ca2e8f176": "GPU-086f60c4-da46-3b0d-5c9c-fa603a2c7d0a", 
    "849815d34935": "GPU-e663e5cb-6203-f55b-3c79-4d044bda261e", 
    "9ad6066c9c0a": "GPU-3f57cb0a-a694-4a39-cfef-1fdbc77eeef7"
}


def _remote_root(checkout_root: str | Path) -> tuple[Path, str]:
    """Reject unsupported machines before filesystem or runtime inspection."""

    hostname = socket.gethostname()
    # Reject any machine outside the supplied remote container pool.
    if platform.system() != "Linux" or hostname not in VERIFIED_HOSTS:
        raise RuntimeError("DiT HPO requires a supplied remote H100/A100 container; laptop execution is prohibited.")
    root = Path(checkout_root).resolve()
    # Require the authoritative remote checkout before reading project configuration.
    if not root.is_relative_to(Path("/workspace")) or not (root / "common/hpo.py").is_file():
        raise RuntimeError("Use the verified project checkout under /workspace on the supplied container.")
    return root, hostname


def _query(arguments: list[str]) -> list[list[str]]:
    """Read NVIDIA inventory without importing a GPU framework."""

    result = subprocess.check_output(["nvidia-smi", *arguments], text=True, timeout=15)
    return list(csv.reader(result.splitlines(), skipinitialspace=True))


def _source_hashes(root: Path) -> dict[str, str]:
    """Record the model, HPO implementation, and admission source identities."""

    paths = [ALLOCATOR_PATH]
    paths.extend(path.relative_to(root) for path in sorted((root / "common").glob("*.py")))
    for directory in ["diffusion", "models"]:
        model_directory = root / directory
        # Fingerprint each available maintained model package.
        if model_directory.is_dir():
            paths.extend(path.relative_to(root) for path in sorted(model_directory.rglob("*.py")))
    return {
        path.as_posix(): hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in paths if (root / path).is_file()
    }


def inspect_remote(checkout_root: str | Path) -> dict[str, Any]:
    """Return a serializable identity and live inventory without importing TF."""

    root, hostname = _remote_root(checkout_root)
    pool = json.loads((root / POOL_PATH).read_text())
    [gpu] = _query([
        "-i", "0", "--query-gpu=uuid,name,memory.free,memory.total", 
        "--format=csv,noheader,nounits"
    ])
    uuid, model, free_mib, total_mib = gpu
    # Bind the physical GPU to its previously verified container identity.
    if uuid != VERIFIED_HOSTS[hostname] or not any(name in model.upper() for name in ["H100", "A100"]):
        raise RuntimeError("The supplied container's GPU UUID/model does not match its verified identity.")
    matches = [
        name for name, host in pool.get("hosts", {}).items()
        if host.get("gpu_uuid") == uuid and host.get("expected_hostname", hostname) == hostname
    ]
    # Require a unique active-pool entry for this verified GPU.
    if len(matches) != 1:
        raise RuntimeError("The verified host/GPU is absent or ambiguous in the active GPU pool.")
    # Preserve the coordinated two-slot and memory-headroom policy.
    if pool.get("maximum_workloads_per_gpu") != 2 or pool.get("minimum_headroom_mib") != 2048:
        raise RuntimeError("The GPU pool no longer has the verified two-slot/headroom policy.")
    versions = {name: importlib.metadata.version(name) for name in ["tensorflow", "keras", "optuna"]}
    # Keep the mandated TensorFlow and Keras versions unchanged.
    if versions["tensorflow"] != "2.20.0" or versions["keras"] != "3.11.2":
        raise RuntimeError("DiT HPO requires TensorFlow 2.20.0 and Keras 3.11.2; no runtime replacement is performed.")
    processes = []
    for row in _query([
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", 
        "--format=csv,noheader,nounits"
    ]):
        # Report only live compute processes on the selected physical GPU.
        if len(row) == 4 and row[0] == uuid:
            processes.append({"pid": int(row[1]), "command": row[2], "memory_mib": row[3]})
    return {
        "checkout_root": str(root), "hostname": hostname, "pool_host": matches[0], 
        "gpu_uuid": uuid, "gpu_name": model, "free_mib": int(free_mib), 
        "total_mib": int(total_mib), "processes": processes, "versions": versions, 
        "python": platform.python_version(), "source_sha256": _source_hashes(root)
    }


def _verify_snapshot(checkout_root: str | Path, expected_identity: dict[str, Any]) -> dict[str, Any]:
    """Reject checkout, framework, host, or GPU changes since preflight."""

    current = inspect_remote(checkout_root)
    keys = ["checkout_root", "hostname", "pool_host", "gpu_uuid", "versions", "python", "source_sha256"]
    for key in keys:
        # Stop if the recorded execution identity changed after preflight.
        if current[key] != expected_identity.get(key):
            raise RuntimeError("Remote HPO identity changed after preflight: " + key)
    return current


def _allocator(root: Path) -> ModuleType:
    """Load the deployed admission helper without importing project models."""

    specification = importlib.util.spec_from_file_location("dit_hpo_resource_slots", root / ALLOCATOR_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _process_identity(pid: int) -> dict[str, Any]:
    """Return a PID identity immune to PID reuse, matching the deployed helper."""

    fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": int(pid), "start_ticks": fields[19], "parent_pid": int(fields[1])}


def _in_existing_allocation(pid: int) -> bool:
    """Detect a notebook already managed by a different resource lease."""

    identity = _process_identity(pid)
    for slot in [0, 1]:
        path = Path("/tmp") / ("joint-notebook-gpu-0-slot" + str(slot) + ".json")
        # Skip slots that have no published allocation record.
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        for item in [record["owner"], *record.get("workers", [])]:
            # Recognize only the exact recorded PID and process start identity.
            if item["pid"] == pid and item["start_ticks"] == identity["start_ticks"]:
                return True
    return False


@contextmanager
def launch_worker(command: list[str], checkout_root: str | Path, expected_identity: dict[str, Any], log_path: str | Path | None = None) -> Iterator[subprocess.Popen[str]]:
    """Own one lease until its registered, gated subprocess has fully exited.

    Run this in a direct remote Jupyter kernel, outside run_notebooks.py. Capacity
    shortages queue without a global cutoff; unknown ownership fails closed. The
    yielded process can be polled or waited upon while output goes to log_path.
    """

    current = _verify_snapshot(checkout_root, expected_identity)
    root = Path(current["checkout_root"])
    # Require admission before any TensorFlow or Keras imports.
    if "tensorflow" in sys.modules or "keras" in sys.modules:
        raise RuntimeError("Use a fresh coordinator kernel without TensorFlow/Keras imports.")
    # Refuse a nested coordinator that already belongs to another lease.
    if _in_existing_allocation(os.getpid()):
        raise RuntimeError("This kernel already belongs to a GPU lease. Open the notebook directly on the remote Jupyter server, outside run_notebooks.py.")
    allocator = _allocator(root)
    lease = None
    while lease is None:
        _verify_snapshot(root, expected_identity)
        try:
            lease = allocator.acquire(memory_mb=RESERVED_MEMORY_MIB, gpu=0, max_jobs=2)
        except allocator.AdmissionBlocked as error:
            message = str(error)
            # Queue only known slot or memory shortages; other admission failures remain fatal.
            if not message.startswith(("Both remote workload slots are occupied", "Insufficient free GPU memory")):
                raise
            print("DiT HPO queued: " + message, flush=True)
            time.sleep(POLL_SECONDS)
    process = None
    read_fd = None
    write_fd = None
    output = None
    try:
        read_fd, write_fd = os.pipe()
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment["DIT_HPO_GATE_FD"] = str(read_fd)
        environment["DIT_HPO_LEASE_SLOT"] = str(lease.record["slot"])
        environment["DIT_HPO_LEASE_OWNER"] = json.dumps(lease.record["owner"])
        environment["PYTHONUNBUFFERED"] = "1"
        # Stream worker output into the requested persistent log.
        if log_path is not None:
            destination = Path(log_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            output = destination.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command, cwd=root, env=environment, stdout=output, stderr=subprocess.STDOUT, 
            pass_fds=tuple([read_fd]), text=True
        )
        os.close(read_fd)
        read_fd = None
        lease.register_worker(process.pid)
        os.write(write_fd, b"1")
        os.close(write_fd)
        write_fd = None
        print("DiT HPO worker " + str(process.pid) + " admitted on " + current["pool_host"] + "; log: " + str(log_path), flush=True)
        yield process
    finally:
        # Stop and reap only this owned child if it is still running.
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for descriptor in [read_fd, write_fd]:
            # Close pipe endpoints retained after incomplete worker startup.
            if descriptor is not None:
                os.close(descriptor)
        # Flush and close the owned worker log before releasing admission.
        if output is not None:
            output.close()
        lease.close()


def _arm_parent_exit() -> None:
    """Terminate this Linux worker if its coordinator process disappears."""

    library = ctypes.CDLL(None, use_errno=True)
    # Fail if the worker cannot arrange termination when its coordinator exits.
    if library.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Could not arm worker termination on coordinator exit.")


def _validate_worker_lease(expected_identity: dict[str, Any]) -> None:
    """Verify a live parent-owned lease includes this exact descendant process."""

    try:
        descriptor = int(os.environ.pop("DIT_HPO_GATE_FD"))
        slot = int(os.environ.pop("DIT_HPO_LEASE_SLOT"))
        expected_owner = json.loads(os.environ.pop("DIT_HPO_LEASE_OWNER"))
    except (KeyError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("GPU workers must be started through launch_worker().") from error
    try:
        # Require the parent registration handshake before GPU initialization.
        if os.read(descriptor, 1) != b"1":
            raise RuntimeError("The coordinator did not authorize this worker after slot registration.")
    finally:
        os.close(descriptor)
    # Reject a slot outside the deployed two-slot allocator.
    if slot not in [0, 1]:
        raise RuntimeError("The worker lease slot is invalid.")
    lock_root = Path("/tmp")
    stem = "joint-notebook-gpu-0"
    record_path = lock_root / (stem + "-slot" + str(slot) + ".json")
    lock_path = lock_root / (stem + (".lock" if slot == 0 else "-slot1.lock"))
    with (lock_root / (stem + "-admission.lock")).open("a") as mutex:
        fcntl.flock(mutex.fileno(), fcntl.LOCK_EX)
        with lock_path.open("a") as lifetime_lock:
            try:
                fcntl.flock(lifetime_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise RuntimeError("The coordinator's GPU lease is no longer held.")
        record = json.loads(record_path.read_text())
        identity = _process_identity(os.getpid())
        owner = _process_identity(expected_owner["pid"])
        # Reject PID reuse or loss of the verified direct parent.
        if owner["start_ticks"] != expected_owner["start_ticks"] or identity["parent_pid"] != owner["pid"]:
            raise RuntimeError("The GPU worker's parent ownership could not be verified.")
        registered = any(
            worker["pid"] == identity["pid"] and worker["start_ticks"] == identity["start_ticks"]
            for worker in record.get("workers", [])
        )
        # Require the exact worker identity in its parent-owned allocation record.
        if not registered or record.get("owner") != expected_owner:
            raise RuntimeError("This GPU worker is not registered to its coordinator's lease.")
        # Enforce both the physical GPU identity and the declared memory reservation.
        if record.get("gpu_uuid") != expected_identity["gpu_uuid"] or record.get("memory_mb") != RESERVED_MEMORY_MIB:
            raise RuntimeError("The GPU worker's reservation or GPU identity does not match the HPO policy.")


@contextmanager
def managed_worker(checkout_root: str | Path, expected_identity: dict[str, Any]) -> Iterator[None]:
    """Validate admission and cap TensorFlow before importing any model code.

    The parent coordinator releases the lease only after this subprocess exits.
    Leaving this context does not pretend TensorFlow has released GPU memory.
    """

    _verify_snapshot(checkout_root, expected_identity)
    # Require admission before any TensorFlow or Keras imports.
    if "tensorflow" in sys.modules or "keras" in sys.modules:
        raise RuntimeError("GPU admission must precede all TensorFlow/Keras imports.")
    _arm_parent_exit()
    _validate_worker_lease(expected_identity)
    os.environ.setdefault("KERAS_HOME", "/workspace/.keras")
    import tensorflow as tf


    # Check the imported runtime against the preflight package version.
    if tf.__version__ != "2.20.0":
        raise RuntimeError("The imported TensorFlow runtime differs from the verified version.")
    physical = tf.config.list_physical_devices("GPU")
    # Expose exactly one admitted physical GPU to the worker.
    if len(physical) != 1:
        raise RuntimeError("The worker must expose exactly one admitted GPU.")
    tf.config.set_logical_device_configuration(
        physical[0], [tf.config.LogicalDeviceConfiguration(memory_limit=TF_MEMORY_MIB)]
    )
    # Fail closed if the admitted GPU cannot initialize.
    if len(tf.config.list_logical_devices("GPU")) != 1:
        raise RuntimeError("The admitted GPU did not initialize successfully.")
    yield
