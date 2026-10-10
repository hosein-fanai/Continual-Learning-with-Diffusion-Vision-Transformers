"""Run bounded admitted UNet train/test smoke checks without datasets or HPO.

Invoke only on a supplied online GPU container. The parent imports no framework;
each sequential child authenticates its lease and installs its TensorFlow cap
before model imports. Synthetic batch-eight results establish limited execution
compatibility, never full-batch capacity, training quality or HPO throughput.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


CASES = [
    {
        "case": "deep_three_levels", "widths": "64-128-256", 
        "block_depth": 3, "bottleneck_depth": 3, "bottleneck_mult": 2.0, 
        "embedding_dim": 192, "embedding_layout": "time_rich", 
        "batch_norm": True, "dropout": 0.2, "activation_func": "gelu", 
        "downsampling_method": "cnn_stride", "upsampling_method": "cnn_transpose"
    }, 
    {
        "case": "four_levels", "widths": "32-64-128-256", 
        "block_depth": 3, "bottleneck_depth": 3, "bottleneck_mult": 2.0, 
        "embedding_dim": 192, "embedding_layout": "balanced", 
        "batch_norm": False, "dropout": 0.1, "activation_func": "swish", 
        "downsampling_method": "max_pooling", "upsampling_method": "cnn_interpolate", 
        "upsampling_interpolation": "nearest"
    }
]


def write_json(path: Path, value: Any) -> None:
    """Publish a finite receipt atomically without replacing unrelated artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def nvidia_snapshot() -> dict[str, str]:
    """Read physical GPU and process memory without importing a framework."""

    return {
        "devices": subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", 
            "--format=csv,noheader,nounits"
        ], text=True, timeout=15).strip(), 
        "compute_processes": subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", 
            "--format=csv,noheader,nounits"
        ], text=True, timeout=15).strip()
    }


def finite_metrics(values: dict[str, Any]) -> dict[str, float]:
    """Authenticate real scalar train/test metrics before serializing a pass."""

    resolved = {}
    for key, tensor in values.items():
        value = float(tensor.numpy())
        # Invalid wrapper metrics cannot establish successful execution.
        if not math.isfinite(value):
            raise ValueError("Nonfinite smoke metric: " + key)
        resolved[key] = value
    # The notebook ranks the actual noise-loss metric, not an unrelated output.
    if "noise_loss" not in resolved:
        raise ValueError("The UNet smoke check did not report noise_loss.")
    return resolved


def run_case(request: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """Exercise a public HPO config and real compiled model on synthetic pixels."""

    import keras
    import optuna
    import tensorflow as tf

    from common.config import save_config
    from common.hpo import _build_trial_config
    from common.model import get_model
    from common.unet_hpo import SEARCH_SPACE_OVERRIDES


    # Framework package metadata and actual imported implementations must agree.
    if tf.__version__ != "2.20.0" or keras.__version__ != "3.11.2":
        raise RuntimeError("UNet smoke requires TensorFlow 2.20.0 / Keras 3.11.2.")
    started = time.time()
    tf.keras.backend.clear_session()
    tf.config.experimental.reset_memory_stats("GPU:0")
    options = deepcopy(SEARCH_SPACE_OVERRIDES)
    for key, value in case.items():
        # Case names identify artifacts and are not model hyperparameters.
        if key != "case":
            options[key] = [value]
    options["optimizer"] = ["adamw"]
    parameters = {
        key: choices[0] if isinstance(choices, list) else choices["low"]
        for key, choices in options.items()
    }
    destination = Path(request["output"]) / case["case"]
    destination.mkdir(parents=True, exist_ok=False)
    config = _build_trial_config(
        optuna.trial.FixedTrial(parameters), "generation", "unet", "cifar10", 1, 
        results_path=destination, search_space_overrides=options, 
        validation_source="split", validation_ratio=0.2, seed=42
    )
    config.dataset.trainset_len = 2
    config.model.show_network_summary = False
    save_config(config, destination / "smoke_config.yaml")
    model = get_model(config)
    variable_devices = sorted({str(variable.value.device) for variable in model.trainable_variables})
    # Real model variables must reside on the admitted logical GPU.
    if not variable_devices or not all("GPU:0" in device for device in variable_devices):
        raise RuntimeError("UNet trainable variables were not placed on the admitted GPU.")
    images = tf.cast(tf.random.stateless_uniform(
        [8, 32, 32, 3], minval=0, maxval=256, seed=[42, 7], dtype=tf.int32
    ), tf.uint8)
    labels = tf.range(8, dtype=tf.int32)
    batch = (images, labels)
    train_step = tf.function(model.train_step, jit_compile=False)
    test_step = tf.function(model.test_step, jit_compile=False)
    before = int(model.optimizer.iterations.numpy())
    train_metrics = []
    for step in range(2):
        train_metrics.append(finite_metrics(train_step(batch)))
    after = int(model.optimizer.iterations.numpy())
    # Two completed optimizer updates distinguish training from a forward smoke.
    if after - before != 2:
        raise RuntimeError("UNet smoke did not perform both expected optimizer updates.")
    model.reset_metrics()
    validation_metrics = finite_metrics(test_step(batch))
    result = {
        "case": case["case"], "status": "PASS", 
        "synthetic_batch_shape": [8, 32, 32, 3], "synthetic_dtype": "uint8", 
        "configured_search_batch_size": config.dataset.batch_size, 
        "actual_synthetic_batch_size": 8, "optimizer_updates": after - before, 
        "train_metrics": train_metrics, "ema_test_metrics": validation_metrics, 
        "test_network": config.model.wrapper_kwargs["test_network_name"], 
        "model_parameters": int(model.count_params()), "variable_devices": variable_devices, 
        "model_kwargs": config.model.kwargs, 
        "tensorflow_memory_bytes": tf.config.experimental.get_memory_info("GPU:0"), 
        "nvidia_snapshot": nvidia_snapshot(), 
        "tensorflow": tf.__version__, "keras": keras.__version__, 
        "elapsed_seconds": time.time() - started, 
        "scope": "synthetic train/test compatibility; no capacity or throughput claim"
    }
    write_json(destination / "result.json", result)
    del train_step, test_step, model, batch, images, labels
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def worker(request_path: Path) -> None:
    """Authenticate this child and run both cases within one admitted GPU lease."""

    request = json.loads(request_path.read_text(encoding="utf-8"))
    root = Path(request["checkout"])
    sys.path.insert(0, str(root))
    os.chdir(root)
    # The executed helper must match the parent-authenticated local source bytes.
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != request["helper_sha256"]:
        raise RuntimeError("The UNet smoke helper changed between parent and worker.")

    from common.dit_hpo_remote import managed_worker


    result = {"status": "RUNNING", "gpu_id": request["gpu_id"], "cases": []}
    try:
        with managed_worker(root, request["identity"]):
            for case in CASES:
                # Keep this validation's bound separate from every experiment clock.
                if time.time() >= request["deadline_unix"]:
                    raise TimeoutError("UNet smoke reached its validation deadline.")
                result["cases"].append(run_case(request, case))
                write_json(Path(request["output"]) / "worker_result.json", result)
        result["status"] = "PASS"
    except BaseException as error:
        result.update({"status": "FAIL", "error": type(error).__name__ + ": " + str(error)})
        write_json(Path(request["output"]) / "worker_result.json", result)
        raise
    write_json(Path(request["output"]) / "worker_result.json", result)


def run_check(arguments: argparse.Namespace) -> dict[str, Any]:
    """Launch sequential bounded workers and preserve identity and memory evidence."""

    root = Path(arguments.checkout).resolve()
    output = Path(arguments.output).resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)

    from common.dit_hpo_remote import inspect_remote, launch_worker


    # Refuse to overwrite completed or interrupted validation evidence.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory to preserve earlier validation evidence.")
    started = time.time()
    deadline = started + arguments.max_seconds
    helper_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = {
        "status": "RUNNING", "started_at_unix": started, "deadline_unix": deadline, 
        "checkout": str(root), "helper_sha256": helper_digest, "devices": [], 
        "synthetic_data_only": True, "hpo_started": False, "experiment_clock_started": False, 
        "scope": "bounded synthetic train/test compatibility, not capacity or throughput"
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "receipt.json", result)
    try:
        for gpu_id in arguments.gpu_ids:
            identity = inspect_remote(
                root, concurrent_trials=1, gpu_ids=[gpu_id], 
                worker_gpu_memory_limit_mb=arguments.memory_mib
            )
            # Recycled SSH endpoints cannot silently substitute another container.
            if arguments.expected_hostname is not None and identity["hostname"] != arguments.expected_hostname:
                raise RuntimeError("UNet smoke reached an unexpected container hostname.")
            destination = output / f"gpu-{gpu_id}"
            request = {
                "checkout": str(root), "output": str(destination), "identity": identity, 
                "gpu_id": gpu_id, "deadline_unix": deadline, "helper_sha256": helper_digest
            }
            request_path = destination / "request.json"
            write_json(request_path, request)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(request_path)]
            with launch_worker(
                command, root, identity, log_path=destination / "worker.log", deadline=deadline
            ) as process:
                while process.poll() is None:
                    # The context terminates and reaps only this owned child on timeout.
                    if time.time() >= deadline:
                        raise TimeoutError("Bounded UNet deployment smoke expired.")
                    time.sleep(1)
                # Failed or terminated children never count as successful checks.
                if process.returncode != 0:
                    raise RuntimeError(f"UNet smoke worker on GPU {gpu_id} exited {process.returncode}.")
            observed = json.loads((destination / "worker_result.json").read_text(encoding="utf-8"))
            # Both requested architectures must complete before this device passes.
            if observed["status"] != "PASS" or len(observed["cases"]) != len(CASES):
                raise RuntimeError("UNet smoke worker has incomplete compatibility evidence.")
            result["devices"].append({"identity": identity, "result": observed})
            write_json(output / "receipt.json", result)
        result.update({"status": "PASS", "finished_at_unix": time.time(), "elapsed_seconds": time.time() - started})
    except BaseException as error:
        result.update({"status": "FAIL", "error": type(error).__name__ + ": " + str(error), "finished_at_unix": time.time()})
        write_json(output / "receipt.json", result)
        raise
    write_json(output / "receipt.json", result)
    return result


def main() -> None:
    """Parse the parent request or dispatch an authenticated internal worker."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", default="/workspace/UNet-HPO")
    parser.add_argument("--output")
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--memory-mib", type=int, default=24576)
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--expected-hostname", default="cf2f09c572e4")
    parser.add_argument("--worker")
    arguments = parser.parse_args()
    # Only launch_worker supplies the lease required by this internal child path.
    if arguments.worker:
        worker(Path(arguments.worker))
    # Parent execution needs a caller-selected evidence directory.
    else:
        # Missing evidence placement must fail before inspection or admission.
        if arguments.output is None:
            parser.error("--output is required for parent execution")
        result = run_check(arguments)
        print(json.dumps({"status": result["status"], "devices": len(result["devices"]), "output": arguments.output}))


# Direct execution coordinates remote compatibility checks, never HPO studies.
if __name__ == "__main__":
    main()
