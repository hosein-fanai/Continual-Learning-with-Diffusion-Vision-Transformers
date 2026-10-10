"""Keep DiT HPO computation on admitted pool or managed online GPU workers.

The notebook is a standard-library coordinator. Its admitted child runs one
TensorFlow workload or coordinates separately reserved HPO workers. Do not
nest this helper inside the
campaign notebook runner, which already owns the notebook kernel's lease.
"""

from contextlib import contextmanager, ExitStack
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
from typing import Any, Callable, Iterator


CAMPAIGN = Path("files/results/joint_classifier_v1_audit_20261007")
POOL_PATH = CAMPAIGN / "runtime_routing_20261007/gpu_pool.json"
ALLOCATOR_PATH = CAMPAIGN / "support/remote_resource_slots.py"
HOSTED_ALLOCATOR_PATH = Path("common/gpu_resource_slots.py")
TF_MEMORY_MIB = 12288
RESERVED_MEMORY_MIB = 13312
POLL_SECONDS = 15
LOCK_ROOT = Path("/tmp")
_ADMITTED_HOSTED_RUNTIME: str | None = None
VERIFIED_HOSTS = {
    "b917b7e3195e": "GPU-938d1ee4-8591-b274-4110-d469503b692c", 
    "ea77e5605c7a": "GPU-cb5492c6-af24-d4d8-864c-238f14012c80", 
    "c91ca2e8f176": "GPU-086f60c4-da46-3b0d-5c9c-fa603a2c7d0a", 
    "849815d34935": "GPU-e663e5cb-6203-f55b-3c79-4d044bda261e", 
    "9ad6066c9c0a": "GPU-3f57cb0a-a694-4a39-cfef-1fdbc77eeef7"
}


def _runtime_kind(hostname: str) -> str:
    """Recognize supplied hosts or online services without trusting an override."""

    # Explicit pool identities always retain their shared two-slot policy.
    if hostname in VERIFIED_HOSTS:
        return "runpod"
    # Kaggle can inherit Colab package markers, so its live marker takes precedence.
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    # A gated child inherits provider identity only after validating its parent lease.
    if _ADMITTED_HOSTED_RUNTIME in {"colab", "kaggle"}:
        return _ADMITTED_HOSTED_RUNTIME
    # The runtime override alone cannot establish a managed online service.
    if os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules:
        return "colab"
    # Generic remote containers need independent container evidence and GPU preflight.
    if Path("/.dockerenv").is_file() or Path("/run/.containerenv").is_file():
        return "container"
    raise RuntimeError("DiT HPO requires a remote Linux container, Colab, or Kaggle; laptop execution is prohibited.")


def _remote_root(checkout_root: str | Path) -> tuple[Path, str]:
    """Reject unsupported machines before filesystem or runtime inspection."""

    hostname = socket.gethostname()
    # Reject desktop kernels before inspecting their checkout or installed runtime.
    if platform.system() != "Linux":
        raise RuntimeError("DiT HPO requires an online Linux GPU runtime; laptop execution is prohibited.")
    runtime_kind = _runtime_kind(hostname)
    root = Path(checkout_root).resolve()
    # The established pool keeps its authoritative /workspace checkout restriction.
    if runtime_kind == "runpod" and not root.is_relative_to(Path("/workspace")):
        raise RuntimeError("Use the verified project checkout under /workspace on the supplied container.")
    # Hosted notebooks must also use a complete project checkout.
    if not (root / "common/hpo.py").is_file():
        raise RuntimeError("Use a complete project checkout containing common/hpo.py.")
    return root, hostname


def _query(arguments: list[str]) -> list[list[str]]:
    """Read NVIDIA inventory without importing a GPU framework."""

    result = subprocess.check_output(["nvidia-smi", *arguments], text=True, timeout=15)
    return list(csv.reader(result.splitlines(), skipinitialspace=True))


def _source_hashes(root: Path) -> dict[str, str]:
    """Record the model, HPO implementation, and admission source identities."""

    paths = [ALLOCATOR_PATH] if _runtime_kind(socket.gethostname()) == "runpod" else []
    paths.extend(path.relative_to(root) for path in sorted((root / "common").glob("*.py")))
    paths.extend([
        Path("common/callbacks/hpo_guard.py"), 
        Path("common/callbacks/hpo_pruning.py")
    ])
    for directory in ["diffusion", "models", "semantic_consolidation"]:
        model_directory = root / directory
        # Fingerprint each available maintained model package.
        if model_directory.is_dir():
            paths.extend(path.relative_to(root) for path in sorted(model_directory.rglob("*.py")))
    return {
        path.as_posix(): hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in paths if (root / path).is_file()
    }


def inspect_remote(checkout_root: str | Path, concurrent_trials: int = 1, gpu_ids: list[int] | None = None, worker_gpu_memory_limit_mb: int | None = None) -> dict[str, Any]:
    """Inspect selected devices and divide the total worker count round-robin.

    gpu_ids contains distinct physical NVIDIA indices, defaults to [0], and may
    not contain more devices than concurrent_trials. Each trial uses one device.
    An explicit worker_gpu_memory_limit_mb reserves that exact TensorFlow budget
    plus per-worker overhead; it must fit every selected GPU's assigned quota.
    None preserves the existing automatically bounded per-worker policy.
    """

    # Exact counts and device identities must not accept boolean or duplicate aliases.
    if type(concurrent_trials) is not int or concurrent_trials < 1:
        raise ValueError("concurrent_trials must be a positive integer.")
    # Reservation sizes require exact positive units, not boolean or fractional aliases.
    if worker_gpu_memory_limit_mb is not None and (
        type(worker_gpu_memory_limit_mb) is not int or worker_gpu_memory_limit_mb <= 0
    ):
        raise ValueError("worker_gpu_memory_limit_mb must be a positive integer reservation size.")
    selected = [0] if gpu_ids is None else list(gpu_ids)
    # Every selected device must receive at least one worker reservation.
    if not selected or any(type(item) is not int or item < 0 for item in selected) \
    or len(set(selected)) != len(selected) or len(selected) > concurrent_trials:
        raise ValueError("gpu_ids must contain distinct nonnegative physical GPU indices, at most concurrent_trials entries.")
    root, hostname = _remote_root(checkout_root)
    runtime_kind = _runtime_kind(hostname)
    counts = {gpu: concurrent_trials // len(selected) + int(position < concurrent_trials % len(selected)) for position, gpu in enumerate(selected)}
    # Supplied containers retain their established identity and two-worker policy.
    if runtime_kind == "runpod" and (selected != [0] or concurrent_trials > 2):
        raise ValueError("The supplied GPU pool supports only one or two concurrent trials on its verified GPU 0.")
    devices = []
    pool_host = runtime_kind
    for gpu_id in selected:
        [gpu] = _query([
            "-i", str(gpu_id), "--query-gpu=uuid,name,memory.free,memory.total", 
            "--format=csv,noheader,nounits"
        ])
        uuid, model, free_mib, total_mib = gpu
        # Inventory rows must represent independent, usable NVIDIA devices.
        if not uuid.startswith("GPU-") or uuid in {item["gpu_uuid"] for item in devices}:
            raise RuntimeError("The selected GPU inventory has an invalid or duplicate UUID.")
        # The established pool uses its verified allocator and fixed reservations.
        if runtime_kind == "runpod":
            pool = json.loads((root / POOL_PATH).read_text())
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
            pool_host = matches[0]
            policy = {"max_jobs": 2, "tf_memory_mib": TF_MEMORY_MIB, "reserved_memory_mib": RESERVED_MEMORY_MIB}
        # Other remote runtimes size budgets from each measured GPU capacity.
        else:
            workers = counts[gpu_id]
            capacity_mib = (int(total_mib) - (2048 + workers * 1024 + 1024)) // workers
            tf_memory_mib = min(TF_MEMORY_MIB, (capacity_mib // 256) * 256)
            # Leave per-worker overhead and allocator headroom on every selected GPU.
            if tf_memory_mib <= 0:
                raise RuntimeError("The selected GPU lacks sufficient memory for its worker policy.")
            policy = {"max_jobs": workers, "tf_memory_mib": tf_memory_mib, "reserved_memory_mib": tf_memory_mib + 1024}
        # Explicit budgets must fit the complete per-device quota without clipping.
        if worker_gpu_memory_limit_mb is not None:
            reserved_mib = worker_gpu_memory_limit_mb + 1024
            # Reject quotas that exceed measured capacity after overhead and headroom.
            if counts[gpu_id] * reserved_mib + 3072 > int(total_mib):
                raise ValueError("The requested worker GPU memory budget and concurrency exceed the selected GPU capacity.")
            policy["tf_memory_mib"] = worker_gpu_memory_limit_mb
            policy["reserved_memory_mib"] = reserved_mib
        devices.append({
            "gpu_id": gpu_id, "gpu_uuid": uuid, "gpu_name": model, "free_mib": int(free_mib), 
            "total_mib": int(total_mib), "concurrent_trials": counts[gpu_id], "worker_policy": policy, "processes": []
        })
    common_cap = min(item["worker_policy"]["tf_memory_mib"] for item in devices)
    for device in devices:
        device["worker_policy"]["tf_memory_mib"] = common_cap
        device["worker_policy"]["reserved_memory_mib"] = common_cap + 1024
    versions = {name: importlib.metadata.version(name) for name in ["tensorflow", "keras", "optuna"]}
    # Keep the mandated TensorFlow and Keras versions unchanged.
    if versions["tensorflow"] != "2.20.0" or versions["keras"] != "3.11.2":
        raise RuntimeError("DiT HPO requires TensorFlow 2.20.0 and Keras 3.11.2; no runtime replacement is performed.")
    for row in _query([
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", 
        "--format=csv,noheader,nounits"
    ]):
        for device in devices:
            # Preserve live process inventory separately for each physical GPU.
            if len(row) == 4 and row[0] == device["gpu_uuid"]:
                device["processes"].append({"pid": int(row[1]), "command": row[2], "memory_mib": row[3]})
    first = devices[0]
    identity = {
        "checkout_root": str(root), "hostname": hostname, "pool_host": pool_host, 
        "gpu_uuid": first["gpu_uuid"], "gpu_name": first["gpu_name"], "free_mib": first["free_mib"], 
        "total_mib": first["total_mib"], "processes": first["processes"], "versions": versions, 
        "python": platform.python_version(), "source_sha256": _source_hashes(root), 
        "runtime_kind": runtime_kind, "worker_policy": first["worker_policy"], 
        "concurrent_trials": concurrent_trials, "gpu_ids": selected, "gpus": devices
    }
    # Omitted requests retain the historical identity shape for existing callers.
    if worker_gpu_memory_limit_mb is not None:
        identity["worker_gpu_memory_limit_mb"] = worker_gpu_memory_limit_mb
    return identity


def _device_identity(identity: dict[str, Any], gpu_id: int | str | None = None) -> dict[str, Any]:
    """Select one authenticated device while preserving the complete plan identity."""

    # The core HPO transport normalizes physical indices to decimal strings.
    if isinstance(gpu_id, str) and gpu_id.isascii() and gpu_id.isdecimal():
        gpu_id = int(gpu_id)
    devices = identity.get("gpus", [])
    # Historical single-GPU records remain valid at the internal lease boundary.
    if not devices:
        # Explicit routing cannot address an unrecorded device.
        if gpu_id not in {None, 0, identity.get("gpu_uuid")}:
            raise RuntimeError("The requested GPU is absent from the admitted HPO identity.")
        return identity
    selected = devices[0] if gpu_id is None else next((item for item in devices if gpu_id in {item["gpu_id"], item["gpu_uuid"]}), None)
    # A scheduler may only choose a device from the exact preflight selection.
    if selected is None:
        raise RuntimeError("The requested GPU is absent from the admitted HPO identity.")
    return {**identity, **{key: selected[key] for key in ["gpu_id", "gpu_uuid", "gpu_name", "free_mib", "total_mib", "processes", "worker_policy"]}}


def _verify_snapshot(checkout_root: str | Path, expected_identity: dict[str, Any]) -> dict[str, Any]:
    """Reject checkout, framework, host, GPU assignment, or budget changes."""

    options = {}
    # Reinspection must retain an explicitly requested worker reservation.
    if "worker_gpu_memory_limit_mb" in expected_identity:
        options["worker_gpu_memory_limit_mb"] = expected_identity["worker_gpu_memory_limit_mb"]
    current = inspect_remote(
        checkout_root, concurrent_trials=expected_identity.get("concurrent_trials", 1), 
        gpu_ids=expected_identity.get("gpu_ids"), **options
    )
    keys = ["checkout_root", "hostname", "pool_host", "gpu_uuid", "versions", "python", "source_sha256", 
            "runtime_kind", "worker_policy", "concurrent_trials"]
    for key in keys:
        # Stop if the recorded execution identity changed after preflight.
        if current[key] != expected_identity.get(key):
            raise RuntimeError("Remote HPO identity changed after preflight: " + key)
    # Measured free memory and live processes may change while reservations queue.
    if "gpus" in expected_identity:
        stable_keys = ["gpu_id", "gpu_uuid", "gpu_name", "total_mib", "concurrent_trials", "worker_policy"]
        expected_devices = [{key: device[key] for key in stable_keys} for device in expected_identity["gpus"]]
        current_devices = [{key: device[key] for key in stable_keys} for device in current["gpus"]]
        # Ordering determines round-robin worker quotas and is part of execution identity.
        if current_devices != expected_devices or current["gpu_ids"] != expected_identity["gpu_ids"]:
            raise RuntimeError("Remote HPO identity changed after preflight: gpus")
    return current

def serial_worker_identity(checkout_root: str | Path, expected_identity: dict[str, Any], gpu_id: int | None = None) -> dict[str, Any]:
    """Reverify the search plan before admitting one job on a selected GPU.

    Confirmation keeps the scientific plan unchanged while reserving only the
    single device it uses. The original complete source and device snapshot must
    still match before deriving this separate runtime-only reservation identity.
    """

    verified = _verify_snapshot(checkout_root, expected_identity)
    selected = verified.get("gpu_ids", [0])[0] if gpu_id is None else gpu_id
    # Confirmation must remain on a device selected by the verified search plan.
    if selected not in verified.get("gpu_ids", [0]):
        raise ValueError("The confirmation GPU is absent from the verified search plan.")
    options = {}
    # Confirmation preserves explicit search memory instead of reverting to the default.
    if "worker_gpu_memory_limit_mb" in verified:
        options["worker_gpu_memory_limit_mb"] = verified["worker_gpu_memory_limit_mb"]
    return inspect_remote(checkout_root, concurrent_trials=1, gpu_ids=[selected], **options)

def _allocator(root: Path) -> ModuleType:
    """Load the deployed admission helper without importing project models."""

    path = ALLOCATOR_PATH if _runtime_kind(socket.gethostname()) == "runpod" else HOSTED_ALLOCATOR_PATH
    specification = importlib.util.spec_from_file_location("dit_hpo_resource_slots", root / path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _process_identity(pid: int) -> dict[str, Any]:
    """Return a PID identity immune to PID reuse, matching the deployed helper."""

    fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": int(pid), "start_ticks": fields[19], "parent_pid": int(fields[1])}


def _in_existing_allocation(pid: int, gpu_ids: list[int] | None = None) -> bool:
    """Reject nested workers and owner overlap on the requested devices."""

    identity = _process_identity(pid)
    for path in LOCK_ROOT.glob("joint-notebook-gpu-*-slot*.json"):
        try:
            record = json.loads(path.read_text())
        except FileNotFoundError:
            # A sibling may finish and release its lease after directory listing.
            continue
        for item in record.get("workers", []):
            # A registered GPU worker cannot become a coordinator on any device.
            if item["pid"] == pid and item["start_ticks"] == identity["start_ticks"]:
                return True
        owner = record["owner"]
        if owner["pid"] == pid and owner["start_ticks"] == identity["start_ticks"]:
            # Legacy callers retain the conservative all-device ownership check.
            if gpu_ids is None:
                return True
            gpu = record.get("gpu")
            # Older serialized leases can store a numeric device index as text.
            if isinstance(gpu, str) and gpu.isdecimal():
                gpu = int(gpu)
            # Only a known different device permits this CPU owner's sibling launch.
            if type(gpu) is not int or gpu < 0 or gpu in gpu_ids:
                return True
    return False

def _check_deadline(deadline: float | None, cancel_event: Any | None = None) -> None:
    """Stop an admission attempt after its optional absolute UTC cutoff."""

    # Threaded confirmation cancellation must also interrupt queued admission.
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("The DiT worker group was cancelled.")
    # An omitted deadline preserves the established unlimited admission behavior.
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("The DiT experiment time budget has expired.")


def _wait_for_capacity(deadline: float | None, cancel_event: Any | None = None) -> None:
    """Bound capacity polling by the remaining campaign time."""

    _check_deadline(deadline, cancel_event=cancel_event)
    remaining = POLL_SECONDS if deadline is None else max(0.0, deadline - time.time())
    # An event wait wakes immediately when another confirmation is cancelled.
    if cancel_event is None:
        time.sleep(min(POLL_SECONDS, remaining))
    # Keep cancellation responsive without changing the legacy sleep path.
    else:
        cancel_event.wait(min(POLL_SECONDS, remaining))
    _check_deadline(deadline, cancel_event=cancel_event)

@contextmanager
def _parallel_launch_guard(checkout_root: str | Path, expected_identity: dict[str, Any], deadline: float | None = None, cancel_event: Any | None = None) -> Iterator[None]:
    """Lock selected GPUs in physical-index order before acquiring reservations."""

    # Serial launches continue to share the allocator without a coordinator lock.
    if expected_identity.get("concurrent_trials", 1) <= 1:
        yield
        return
    _verify_snapshot(checkout_root, expected_identity)
    with ExitStack() as stack:
        for gpu_id in sorted(expected_identity.get("gpu_ids", [0])):
            handle = stack.enter_context((LOCK_ROOT / ("dit-hpo-parallel-gpu-" + str(gpu_id) + ".lock")).open("a"))
            acquired = False
            while not acquired:
                _check_deadline(deadline, cancel_event=cancel_event)
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    print("DiT parallel HPO queued: another coordinator owns GPU " + str(gpu_id) + "'s reservation turn.", flush=True)
                    _wait_for_capacity(deadline, cancel_event=cancel_event)
                    _verify_snapshot(checkout_root, expected_identity)
                # Ordered coordinator locks prevent overlapping groups from deadlocking.
                else:
                    acquired = True
        yield


@contextmanager
def launch_worker(command: list[str], checkout_root: str | Path, expected_identity: dict[str, Any], log_path: str | Path | None = None, deadline: float | None = None, cancel_event: Any | None = None) -> Iterator[subprocess.Popen[str]]:
    """Own admitted GPU reservations until the gated subprocess fully exits."""

    options = {} if deadline is None else {"deadline": deadline}
    # Forward cancellation only when callers opt into grouped execution.
    if cancel_event is not None:
        options["cancel_event"] = cancel_event
    with _parallel_launch_guard(checkout_root, expected_identity, **options):
        with _launch_worker(command, checkout_root, expected_identity, log_path=log_path, **options) as process:
            yield process


@contextmanager
def _launch_worker(command: list[str], checkout_root: str | Path, expected_identity: dict[str, Any], log_path: str | Path | None = None, deadline: float | None = None, cancel_event: Any | None = None) -> Iterator[subprocess.Popen[str]]:
    """Reserve all selected devices before starting the authenticated CPU child.

    Multi-GPU acquisition rolls back every reservation before retrying capacity
    shortages. Existing single-GPU launches retain their established handshake.
    """

    current = _verify_snapshot(checkout_root, expected_identity)
    root = Path(current["checkout_root"])
    # Require admission before any TensorFlow or Keras imports.
    if "tensorflow" in sys.modules or "keras" in sys.modules:
        raise RuntimeError("Use a fresh coordinator kernel without TensorFlow/Keras imports.")
    # Refuse a nested coordinator that already belongs to another lease.
    if _in_existing_allocation(os.getpid(), gpu_ids=current.get("gpu_ids", [0])):
        raise RuntimeError("This kernel already belongs to a GPU lease. Open the notebook directly on the remote Jupyter server, outside run_notebooks.py.")
    allocator = _allocator(root)
    devices = current.get("gpus", [])
    multiple_devices = len(devices) > 1
    leases = []
    while not leases:
        _check_deadline(deadline, cancel_event=cancel_event)
        _verify_snapshot(root, expected_identity)
        try:
            # Cross-device groups hold no partial reservation while awaiting capacity.
            if multiple_devices:
                for device in sorted(devices, key=lambda item: item["gpu_id"]):
                    policy = device["worker_policy"]
                    leases.extend(allocator.acquire_many(
                        memory_mb=policy["reserved_memory_mib"], gpu=device["gpu_id"], 
                        max_jobs=policy["max_jobs"], count=device["concurrent_trials"]
                    ))
            # Single-GPU launches retain the established first-lease handshake.
            else:
                policy = current["worker_policy"]
                gpu_id = current.get("gpu_ids", [0])[0]
                leases = [allocator.acquire(memory_mb=policy["reserved_memory_mib"], gpu=gpu_id, max_jobs=policy["max_jobs"])]
        except BaseException as error:
            for retained in reversed(leases):
                retained.close()
            leases = []
            # Only known capacity shortages queue; unknown ownership remains fatal.
            if not isinstance(error, allocator.AdmissionBlocked) or not str(error).startswith(("Both remote workload slots are occupied", "Insufficient free GPU memory")):
                raise
            print("DiT HPO queued: " + str(error), flush=True)
            _wait_for_capacity(deadline, cancel_event=cancel_event)
    first_gpu = current.get("gpu_ids", [0])[0]
    lease = next((item for item in leases if int(item.record.get("gpu", 0)) == first_gpu), leases[0])
    process = None
    read_fd = None
    write_fd = None
    output = None
    try:
        _check_deadline(deadline, cancel_event=cancel_event)
        read_fd, write_fd = os.pipe()
        environment = os.environ.copy()
        # The child admission loop shares the same absolute campaign cutoff.
        if deadline is not None:
            environment["DIT_HPO_DEADLINE_UTC"] = str(deadline)
        device_identity = _device_identity(current, first_gpu)
        environment["CUDA_VISIBLE_DEVICES"] = str(device_identity.get("gpu_uuid", first_gpu))
        # Pass provider identity for authenticated children whose parent used an SDK marker.
        if current["runtime_kind"] in {"colab", "kaggle"}:
            environment["CONTINUAL_RUNTIME"] = current["runtime_kind"]
        environment["DIT_HPO_GATE_FD"] = str(read_fd)
        environment["DIT_HPO_LEASE_SLOT"] = str(lease.record["slot"])
        environment["DIT_HPO_LEASE_GPU"] = str(first_gpu)
        environment["DIT_HPO_LEASE_OWNER"] = json.dumps(lease.record["owner"])
        # All group records are reauthenticated under their individual lifetime locks.
        if multiple_devices:
            environment["DIT_HPO_GROUP_LEASES"] = json.dumps([
                {"gpu_id": int(item.record["gpu"]), "slot": item.record["slot"], "owner": item.record["owner"]}
                for item in leases
            ])
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
        # Stop and reap only this owned child before releasing its GPU reservations.
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
        for retained in reversed(leases):
            retained.close()

def _arm_parent_exit() -> None:
    """Terminate this Linux worker if its coordinator process disappears."""

    library = ctypes.CDLL(None, use_errno=True)
    # Fail if the worker cannot arrange termination when its coordinator exits.
    if library.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Could not arm worker termination on coordinator exit.")


def _validate_worker_lease(expected_identity: dict[str, Any]) -> dict[str, Any]:
    """Verify a live parent-owned lease includes this exact descendant process."""

    try:
        selected_gpu = int(os.environ.pop("DIT_HPO_LEASE_GPU", "0"))
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
    # The coordinator gate authorizes only the first selected device's parent lease.
    if selected_gpu != expected_identity.get("gpu_ids", [0])[0]:
        raise RuntimeError("The coordinator lease GPU differs from the verified first device.")
    # Restrict workers to the declared hosted or supplied-pool slot policy.
    if slot not in range(expected_identity["worker_policy"]["max_jobs"]):
        raise RuntimeError("The worker lease slot is invalid.")
    lock_root = LOCK_ROOT
    stem = "joint-notebook-gpu-" + str(expected_identity.get("gpu_id", expected_identity.get("gpu_ids", [0])[0]))
    record_path = lock_root / (stem + "-slot" + str(slot) + ".json")
    lock_path = lock_root / (stem + (".lock" if slot == 0 else "-slot" + str(slot) + ".lock"))
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
        if record.get("gpu_uuid") != expected_identity["gpu_uuid"] \
        or record.get("memory_mb") != expected_identity["worker_policy"]["reserved_memory_mib"]:
            raise RuntimeError("The GPU worker's reservation or GPU identity does not match the HPO policy.")
    return record

@contextmanager
def _locked_lease_record(slot: int, expected_owner: dict[str, Any], expected_identity: dict[str, Any]) -> Iterator[tuple[dict[str, Any], Path]]:
    """Read and modify only a live lease under its shared admission mutex."""

    # Reject unsupported slot identities before examining shared records.
    if slot not in range(expected_identity["worker_policy"]["max_jobs"]):
        raise RuntimeError("The parallel worker lease slot is invalid.")
    lock_root = LOCK_ROOT
    stem = "joint-notebook-gpu-" + str(expected_identity.get("gpu_id", expected_identity.get("gpu_ids", [0])[0]))
    record_path = lock_root / (stem + "-slot" + str(slot) + ".json")
    lock_path = lock_root / (stem + (".lock" if slot == 0 else "-slot" + str(slot) + ".lock"))
    with (lock_root / (stem + "-admission.lock")).open("a") as mutex:
        fcntl.flock(mutex.fileno(), fcntl.LOCK_EX)
        with lock_path.open("a") as lifetime_lock:
            try:
                fcntl.flock(lifetime_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise RuntimeError("The parallel worker's GPU lease is no longer held.")
        record = json.loads(record_path.read_text())
        owner = _process_identity(expected_owner["pid"])
        # Preserve exact ownership across namespace aliases and PID reuse.
        if owner["start_ticks"] != expected_owner["start_ticks"] or record.get("owner") != expected_owner:
            raise RuntimeError("The parallel worker's lease owner changed.")
        # Tie every registration to the measured physical GPU and worker reservation.
        if record.get("slot") != slot or record.get("gpu_uuid") != expected_identity["gpu_uuid"] \
        or record.get("memory_mb") != expected_identity["worker_policy"]["reserved_memory_mib"]:
            raise RuntimeError("The parallel worker's GPU reservation changed.")
        yield record, record_path


def _descendant_identity(pid: int, expected_owner: dict[str, Any], allocator: ModuleType) -> dict[str, Any]:
    """Verify ancestry and namespace identities before recording a worker."""

    worker = allocator._identity(pid)
    ancestor = worker
    visited = set()
    while ancestor["pid"] != expected_owner["pid"]:
        # A cyclic or detached ancestry cannot establish the requested lease owner.
        if ancestor["pid"] in visited or ancestor["parent_pid"] <= 0:
            raise RuntimeError("The parallel worker is not a descendant of its lease owner.")
        visited.add(ancestor["pid"])
        ancestor = allocator._identity(ancestor["parent_pid"])
    # Matching a PID is insufficient if its process lifetime changed.
    if ancestor["start_ticks"] != expected_owner["start_ticks"]:
        raise RuntimeError("The parallel worker's ancestor PID was reused.")
    return worker


def _edit_lease_worker(slot: int, owner: dict[str, Any], worker: dict[str, Any], expected_identity: dict[str, Any], allocator: ModuleType, add: bool) -> None:
    """Edit one exact worker registration without replacing other live entries."""

    with _locked_lease_record(slot, owner, expected_identity) as (record, path):
        retained = [
            item for item in record.get("workers", [])
            if item["pid"] != worker["pid"] or item["start_ticks"] != worker["start_ticks"]
        ]
        # Registration requires live ancestry; removing a reaped identity does not.
        if add:
            verified = _descendant_identity(worker["pid"], owner, allocator)
            # Refuse to replace a registration using a reused PID.
            if verified["start_ticks"] != worker["start_ticks"]:
                raise RuntimeError("The proposed parallel worker PID was reused.")
            retained.append(verified)
        record["workers"] = retained
        allocator._write_record(path, record)


def _stop_owned_worker(worker: dict[str, Any]) -> None:
    """Stop and reap an owned child before its resource context is released."""

    try:
        current = _process_identity(worker["pid"])
    except FileNotFoundError:
        return
    # A reaped child whose PID was reused is no longer this coordinator's process.
    if current["start_ticks"] != worker["start_ticks"]:
        return
    # Never signal another coordinator's child during cleanup.
    if current["parent_pid"] != os.getpid():
        raise RuntimeError("Cannot clean up a parallel worker with changed parent ownership.")
    os.kill(worker["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            reaped, _status = os.waitpid(worker["pid"], os.WNOHANG)
        except ChildProcessError:
            return
        # A completed wait establishes that the owned worker no longer consumes memory.
        if reaped == worker["pid"]:
            return
        time.sleep(0.05)
    os.kill(worker["pid"], signal.SIGKILL)
    os.waitpid(worker["pid"], 0)


def _inherit_provider(expected_identity: dict[str, Any]) -> None:
    """Inherit an online provider only after a valid parent lease is established."""

    global _ADMITTED_HOSTED_RUNTIME

    # Supplied pool identities remain independent of hosted environment markers.
    if expected_identity["runtime_kind"] in {"colab", "kaggle"}:
        # The verified parent supplies this marker to authenticated descendants.
        if os.environ.get("CONTINUAL_RUNTIME") != expected_identity["runtime_kind"]:
            raise RuntimeError("The hosted worker's inherited provider differs from its verified parent.")
        _ADMITTED_HOSTED_RUNTIME = expected_identity["runtime_kind"]


@contextmanager
def managed_parallel_coordinator(checkout_root: str | Path, expected_identity: dict[str, Any]) -> Iterator[Callable[..., Any]]:
    """Provide authenticated per-device resource contexts to the shared HPO API.

    Multi-GPU groups are fully reserved by the notebook before this CPU child
    starts. Single-GPU groups preserve the existing outer and child ownership.
    """

    # Admission and process startup must precede framework imports.
    if "tensorflow" in sys.modules or "keras" in sys.modules:
        raise RuntimeError("Parallel admission must precede all TensorFlow/Keras imports.")
    _arm_parent_exit()
    first_record = _validate_worker_lease(expected_identity)
    _inherit_provider(expected_identity)
    current = _verify_snapshot(checkout_root, expected_identity)
    root = Path(current["checkout_root"])
    allocator = _allocator(root)
    coordinator = allocator._identity(os.getpid())
    first_slot = first_record["slot"]
    first_owner = first_record["owner"]
    first_gpu = current.get("gpu_ids", [0])[0]
    first_identity = _device_identity(current, first_gpu)
    deadline_value = os.environ.get("DIT_HPO_DEADLINE_UTC")
    deadline = None if deadline_value is None else float(deadline_value)
    extra_leases = []
    detached = False
    active: dict[tuple[int, int], dict[str, Any]] = {}
    previous_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    previous_termination = signal.getsignal(signal.SIGTERM)

    def interrupt_coordinator(signum: int, frame: Any) -> None:
        """Unwind the process pool so workers are reaped before lease release."""

        raise KeyboardInterrupt("The parallel HPO coordinator received termination.")

    signal.signal(signal.SIGTERM, interrupt_coordinator)
    try:
        _edit_lease_worker(first_slot, first_owner, coordinator, first_identity, allocator, add=False)
        detached = True
        policy = current["worker_policy"]
        concurrent_trials = current["concurrent_trials"]
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        allocations = [{"gpu_id": first_gpu, "slot": first_slot, "owner": first_owner}]
        # Every cross-device reservation was acquired before this process was spawned.
        if len(current.get("gpus", [])) > 1:
            try:
                allocations = json.loads(os.environ.pop("DIT_HPO_GROUP_LEASES"))
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                raise RuntimeError("Multi-GPU HPO requires the verified complete parent reservation group.") from error
            for allocation in allocations:
                selected = _device_identity(current, allocation["gpu_id"])
                # The notebook must own every preallocated group reservation.
                if allocation["owner"] != first_owner:
                    raise RuntimeError("The multi-GPU group does not belong to its verified notebook owner.")
                with _locked_lease_record(allocation["slot"], first_owner, selected):
                    pass
        # One explicitly routed worker needs no additional reservation.
        elif concurrent_trials > 1:
            while not extra_leases:
                _check_deadline(deadline)
                _verify_snapshot(root, expected_identity)
                try:
                    # The deployed pool's established two-worker API remains unchanged.
                    if concurrent_trials == 2:
                        extra_leases = [allocator.acquire(memory_mb=policy["reserved_memory_mib"], gpu=first_gpu, max_jobs=policy["max_jobs"])]
                    # Larger hosted groups reserve their remaining slots atomically.
                    else:
                        extra_leases = allocator.acquire_many(
                            memory_mb=policy["reserved_memory_mib"], gpu=first_gpu, 
                            max_jobs=policy["max_jobs"], count=concurrent_trials - 1
                        )
                except allocator.AdmissionBlocked as error:
                    # Only known capacity shortages may wait; ownership failures stay fatal.
                    if not str(error).startswith(("Both remote workload slots are occupied", "Insufficient free GPU memory")):
                        raise
                    print("DiT parallel HPO queued: " + str(error), flush=True)
                    _wait_for_capacity(deadline)
            allocations.extend({"gpu_id": first_gpu, "slot": lease.record["slot"], "owner": lease.record["owner"]} for lease in extra_leases)
        # GPU and slot together identify a unique reservation across the full group.
        if len(allocations) != concurrent_trials or len({(item["gpu_id"], item["slot"]) for item in allocations}) != concurrent_trials \
        or any(lease.record["owner"] != coordinator for lease in extra_leases):
            raise RuntimeError("The parallel HPO leases do not have the required verified owners and distinct slots.")
        for device in current.get("gpus", []):
            # The fixed scheduler quota must match the actual per-device lease count.
            if sum(item["gpu_id"] == device["gpu_id"] for item in allocations) != device["concurrent_trials"]:
                raise RuntimeError("The multi-GPU reservation counts differ from the HPO routing policy.")

        @contextmanager
        def worker_context(gpu_id: int | str | None = None) -> Iterator[dict[str, Any]]:
            """Bind a gated worker to a free reservation on its requested GPU."""

            selected = _device_identity(current, gpu_id)
            selected_id = selected.get("gpu_id", first_gpu)
            available = [item for item in allocations if (item["gpu_id"], item["slot"]) not in active and (gpu_id is None or item["gpu_id"] == selected_id)]
            # Another worker would exceed this device's verified reservation count.
            if not available:
                raise RuntimeError("All admitted DiT HPO worker contexts for the selected GPU are already in use.")
            allocation = available[0]
            selected = _device_identity(current, allocation["gpu_id"])
            slot = allocation["slot"]
            key = (allocation["gpu_id"], slot)
            state: dict[str, Any] = {"worker": None}
            active[key] = state

            def register(pid: int) -> None:
                """Publish one exact child identity before the core stdin gate opens."""

                # A context represents one process lifetime, never a shared slot handle.
                if state["worker"] is not None:
                    raise RuntimeError("The admitted HPO worker context already registered a process.")
                worker = _descendant_identity(pid, allocation["owner"], allocator)
                # Only children created by this process pool may use its reservations.
                if worker["parent_pid"] != coordinator["pid"]:
                    raise RuntimeError("The HPO worker is not a direct child of the parallel coordinator.")
                # A process lifetime belongs to exactly one active GPU reservation.
                if any(item["worker"] is not None and item["worker"]["pid"] == worker["pid"] \
                and item["worker"]["start_ticks"] == worker["start_ticks"] for item in active.values()):
                    raise RuntimeError("The HPO worker is already registered to another admitted context.")
                _edit_lease_worker(slot, allocation["owner"], worker, selected, allocator, add=True)
                state["worker"] = worker

            visibility = str(gpu_id) if gpu_id is not None else str(selected["gpu_uuid"] if "gpus" in current else allocation["gpu_id"])
            environment = {
                "CUDA_VISIBLE_DEVICES": visibility, "PYTHONUNBUFFERED": "1", 
                "DIT_HPO_PARALLEL_IDENTITY": json.dumps(current), 
                "DIT_HPO_PARALLEL_COORDINATOR": json.dumps(coordinator), 
                "DIT_HPO_LEASE_GPU": str(allocation["gpu_id"]), "DIT_HPO_LEASE_SLOT": str(slot), 
                "DIT_HPO_LEASE_OWNER": json.dumps(allocation["owner"])
            }
            # Propagate module-only online identity through the validated handshake.
            if current["runtime_kind"] in {"colab", "kaggle"}:
                environment["CONTINUAL_RUNTIME"] = current["runtime_kind"]
            try:
                yield {"environment": environment, "register": register}
            finally:
                worker = state["worker"]
                # Reap a failed or interrupted child before deleting its registration.
                if worker is not None:
                    _stop_owned_worker(worker)
                    _edit_lease_worker(slot, allocation["owner"], worker, selected, allocator, add=False)
                active.pop(key, None)

        yield worker_context
    finally:
        for state in list(active.values()):
            worker = state["worker"]
            # Defensive cleanup retains reservations until every owned child is reaped.
            if worker is not None:
                _stop_owned_worker(worker)
        for lease in reversed(extra_leases):
            lease.close()
        # The outer notebook retains its leases until the CPU child fully exits.
        if detached:
            _edit_lease_worker(first_slot, first_owner, coordinator, first_identity, allocator, add=True)
        signal.signal(signal.SIGTERM, previous_termination)
        # Restore visibility without initializing any GPU in the CPU coordinator.
        if previous_visibility is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # Preserve an explicit visibility value supplied by the outer launcher.
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_visibility

def prepare_hpo_worker(gpu_memory_limit_mb: int | None) -> None:
    """Authenticate an admitted process-pool child before the HPO API imports TF."""

    # Framework initialization belongs to the existing core HPO worker after admission.
    if "tensorflow" in sys.modules or "keras" in sys.modules:
        raise RuntimeError("Parallel worker admission must precede TensorFlow/Keras imports.")
    try:
        expected = json.loads(os.environ.pop("DIT_HPO_PARALLEL_IDENTITY"))
        coordinator = json.loads(os.environ.pop("DIT_HPO_PARALLEL_COORDINATOR"))
        slot = int(os.environ.pop("DIT_HPO_LEASE_SLOT"))
        owner = json.loads(os.environ.pop("DIT_HPO_LEASE_OWNER"))
    except (KeyError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("Parallel HPO workers require a verified resource context.") from error
    # The existing HPO configuration must enforce the exact admitted per-worker cap.
    if gpu_memory_limit_mb != expected["worker_policy"]["tf_memory_mib"]:
        raise RuntimeError("The HPO worker's GPU memory cap differs from its admission policy.")
    _arm_parent_exit()
    identity = _process_identity(os.getpid())
    parent = _process_identity(coordinator["pid"])
    # Authenticate the process-pool parent as well as the notebook lease owner.
    if identity["parent_pid"] != coordinator["pid"] or parent["start_ticks"] != coordinator["start_ticks"]:
        raise RuntimeError("The parallel HPO coordinator identity changed.")
    selected_gpu = int(os.environ.pop("DIT_HPO_LEASE_GPU", "0"))
    selected = _device_identity(expected, selected_gpu)
    # Device visibility and the authenticated reservation must select the same GPU.
    if "gpus" in expected and os.environ.get("CUDA_VISIBLE_DEVICES") not in {str(selected_gpu), selected["gpu_uuid"]}:
        raise RuntimeError("The HPO worker's GPU visibility differs from its admitted reservation.")
    with _locked_lease_record(slot, owner, selected) as (record, _path):
        registered = any(
            worker["pid"] == identity["pid"] and worker["start_ticks"] == identity["start_ticks"]
            for worker in record.get("workers", [])
        )
        # The core stdin handshake opens only after this exact identity is published.
        if not registered:
            raise RuntimeError("The parallel HPO worker is not registered to its GPU lease.")
    _inherit_provider(expected)
    current = _verify_snapshot(expected["checkout_root"], expected)
    _descendant_identity(os.getpid(), owner, _allocator(Path(current["checkout_root"])))
    cache_path = "/workspace/.keras" if current["runtime_kind"] == "runpod" else str(Path(current["checkout_root"]).parent / ".keras")
    os.environ.setdefault("KERAS_HOME", cache_path)

@contextmanager
def managed_worker(checkout_root: str | Path, expected_identity: dict[str, Any]) -> Iterator[None]:
    """Validate admission and cap TensorFlow before importing any model code.

    The parent coordinator releases the lease only after this subprocess exits.
    Leaving this context does not pretend TensorFlow has released GPU memory.
    """

    global _ADMITTED_HOSTED_RUNTIME

    # Require admission before any TensorFlow or Keras imports.
    if "tensorflow" in sys.modules or "keras" in sys.modules:
        raise RuntimeError("GPU admission must precede all TensorFlow/Keras imports.")
    _arm_parent_exit()
    _validate_worker_lease(expected_identity)
    # Only an authenticated child can inherit a module-only hosted parent marker.
    if expected_identity["runtime_kind"] in {"colab", "kaggle"}:
        # Require the explicit provider value passed by the verified coordinator.
        if os.environ.get("CONTINUAL_RUNTIME") != expected_identity["runtime_kind"]:
            raise RuntimeError("The hosted worker's inherited provider differs from its verified parent.")
        _ADMITTED_HOSTED_RUNTIME = expected_identity["runtime_kind"]
    _verify_snapshot(checkout_root, expected_identity)
    cache_path = "/workspace/.keras" if expected_identity["runtime_kind"] == "runpod" else str(Path(checkout_root).parent / ".keras")
    os.environ.setdefault("KERAS_HOME", cache_path)
    import tensorflow as tf


    # Check the imported runtime against the preflight package version.
    if tf.__version__ != "2.20.0":
        raise RuntimeError("The imported TensorFlow runtime differs from the verified version.")
    physical = tf.config.list_physical_devices("GPU")
    # Expose exactly one admitted physical GPU to the worker.
    if len(physical) != 1:
        raise RuntimeError("The worker must expose exactly one admitted GPU.")
    tf.config.set_logical_device_configuration(
        physical[0], [tf.config.LogicalDeviceConfiguration(memory_limit=expected_identity["worker_policy"]["tf_memory_mib"])]
    )
    # Fail closed if the admitted GPU cannot initialize.
    if len(tf.config.list_logical_devices("GPU")) != 1:
        raise RuntimeError("The admitted GPU did not initialize successfully.")
    yield
