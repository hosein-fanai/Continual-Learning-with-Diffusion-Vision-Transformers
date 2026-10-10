"""Measure admitted CIFAR-100 follow-up workers without starting production HPO."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


BRANCHES = [
    "cross_dense", "feature_dense", "local_hybrid", "u_cross", 
    "plain_missing", "u_skip", "feature_ladder", "cross_ladder"
]


def write_json(path: str | Path, value: object) -> None:
    """Publish this check's receipt atomically."""

    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def capacity_options(branches: list[str]) -> dict[str, list[Any]]:
    """Fix large legal routed, U and mixer cases using public search overrides."""

    return {
        "dit_followup_branch": list(dict.fromkeys(branches)), 
        "followup_missing_axis": ["dim"], "followup_missing_dim": [256], 
        "dim": [256], "depth": [10], "mha_num_heads": [8], 
        "mha_key_dim": ["dim"], "batch_size": [128], "patch_size": [2], 
        "mlp_ratio": [4.0], "patchify_with_cnn": [True], "use_cfg": [True], 
        "patches_pos_embed_type": ["2d_sincos"], "patches_pos_merger_type": ["concat"], 
        "time_freq_dim": [8], "time_embed_trainable": [True], 
        "time_mlp_ratio": [4], "label_embed_type": ["new_weight"], 
        "label_freq_dim": [8], "label_mlp_ratio": [4], 
        "conds_merger_type": ["concat"], "ln_mlp_ratio": [4], 
        "use_refiner_cnn": [True], "final_activation_func": ["linear"], 
        "loss_function": ["mse"], "timesteps": [5000], "optimizer": ["adam"], 
        "local_mixer_variant": ["expanded_pointwise"], "local_mixer_kernel_size": [7], 
        "local_mixer_placement": ["every"], "feature_merge": ["concat"], 
        "cross_merge": ["concat"], "cross_plug_type": ["values", "queries"], 
        "resampling_pos_embed_type": ["new_weight"]
    }


def initial_trials(branches: list[str], count: int) -> list[dict[str, str]]:
    """Cover each requested topology and alternate cross-attention direction."""

    result = []
    for index in range(count):
        branch = branches[index % len(branches)]
        trial = {"dit_followup_branch": branch}
        # Exercise both supported directions without adding inactive parameters.
        if branch in ["cross_dense", "cross_ladder", "u_cross"]:
            trial["cross_plug_type"] = "values" if (index // len(branches)) % 2 == 0 else "queries"
        result.append(trial)
    return result


def verify_artifacts(study: Any, study_root: Path, expected_trials: int) -> list[dict[str, Any]]:
    """Require finite completed trials and readable final reports and events."""

    from PIL import Image


    trials = study.get_trials(deepcopy=False)
    states = [trial.state.name for trial in trials]
    # Pruned, failed and unfinished configurations do not establish safe capacity.
    if len(trials) != expected_trials or any(state != "COMPLETE" for state in states):
        raise RuntimeError("Capacity check did not finish every trial: " + str(states))
    artifacts = []
    for trial in trials:
        # A successful process must also yield a meaningful validation objective.
        if trial.value is None or not math.isfinite(trial.value):
            raise RuntimeError("Capacity check has a nonfinite objective.")
        payload_path = study_root / "workers" / f"trial-{trial.number:04d}.json"
        payload = json.loads(payload_path.read_text())
        # The completed database trial must agree with its worker's outcome.
        if payload.get("status") != "complete":
            raise RuntimeError("Missing successful worker payload.")
        run = Path(payload["results_path"])
        for pattern in ["final*.png", "final*.gif"]:
            pictures = list(run.glob(pattern))
            # Sample-generation compatibility requires both still and animated reports.
            if not pictures:
                raise RuntimeError("Missing final image report: " + pattern)
            for picture in pictures:
                with Image.open(picture) as image:
                    for frame in range(getattr(image, "n_frames", 1)):
                        image.seek(frame)
                        image.load()
        for filename in ["train history.csv", "evals history.csv", "objectives.csv", "config.yaml", "model.weights.h5"]:
            artifact = run / filename
            # Empty checkpoint/config/history files are incomplete output.
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise RuntimeError("Missing completed artifact: " + str(artifact))
        trial_events = [
            path for path in (study_root / "tensorboard").glob(f"t{trial.number:04d}*")
            if path.is_dir() and (path.name == f"t{trial.number:04d}" or path.name.startswith(f"t{trial.number:04d}-"))
        ]
        # Current APIs append a configuration hash to each TensorBoard run name.
        if len(trial_events) != 1:
            raise RuntimeError("Missing or ambiguous training TensorBoard run for trial " + str(trial.number))
        event_directories = [
            trial_events[0] / "train", trial_events[0] / "validation", 
            study_root / "tensorboard" / f"trial-{trial.number:04d}" / "outcome"
        ]
        for tensorboard in event_directories:
            # All three event streams must be present after a completed run.
            if not list(tensorboard.glob("events.out.tfevents.*")):
                raise RuntimeError("Missing TensorBoard events: " + str(tensorboard))
        artifacts.append({
            "trial": trial.number, "branch": trial.params["dit_followup_branch"], 
            "objective": trial.value, "run_directory": str(run), 
            "worker_gpu_id": trial.user_attrs.get("worker_gpu_id"), 
            "duration_seconds": (trial.datetime_complete - trial.datetime_start).total_seconds()
        })
    return artifacts


def child(request_path: str | Path) -> None:
    """Execute public HPO APIs only after the parent admission handshake."""

    request = json.loads(Path(request_path).read_text())
    root = Path(request["checkout"])
    sys.path.insert(0, str(root))
    os.chdir(root)
    from common.dit_hpo_remote import managed_parallel_coordinator


    with managed_parallel_coordinator(root, request["identity"]) as context:
        from common.hpo import run_hpo


        study = run_hpo(
            task="generation", model_name="diffusion_transformer", dataset_name="CIFAR100", 
            n_trials=request["trials"], epochs=1, results_path=request["results"], 
            timeout=max(1, request["deadline"] - time.time() - 120), 
            fit_kwargs={"steps_per_epoch": 2}, 
            objective_metrics=["generation_loss"], objective_directions=["minimize"], 
            dtype_policy="float32", n_startup_trials=40, 
            search_space_overrides=capacity_options(request["branches"]), 
            validation_source="test", validation_ratio=0.0, 
            concurrent_trials=request["concurrent_trials"], 
            worker_gpu_memory_limit_mb=request["memory_mib"], 
            worker_gpu_ids=[gpu["gpu_uuid"] for gpu in request["identity"]["gpus"]], 
            gpu_worker_context=context, 
            pruning={
                "type": "percentile", "monitor": "val_noise_loss", "percentile": 75.0, 
                "n_startup_trials": 40, "n_warmup_steps": 9, "interval_steps": 5, "n_min_trials": 10
            }, 
            stop_active_on_timeout=True, 
            initial_trials=initial_trials(request["branches"], request["trials"]), 
            seed=42
        )
        study_root = Path(request["results"]) / "generation" / "diffusion_transformer" / "cifar100"
        trials = study.get_trials(deepcopy=False)
        write_json(Path(request["output"]) / "trial_states.json", [
            {"number": trial.number, "state": trial.state.name, "user_attrs": trial.user_attrs}
            for trial in trials
        ])
        artifacts = verify_artifacts(study, study_root, request["trials"])
        write_json(Path(request["output"]) / "worker_result.json", {"status": "PASS", "artifacts": artifacts})


def observe_gpus(status: dict[str, Any], identity: dict[str, Any]) -> None:
    """Measure physical GPU usage and simultaneous workers on every device."""

    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu", 
        "--format=csv,noheader,nounits"
    ], text=True, timeout=15).splitlines()
    selected = {gpu["gpu_uuid"]: str(gpu["gpu_id"]) for gpu in identity["gpus"]}
    for row in rows:
        uuid, memory, utilization = [part.strip() for part in row.split(",")]
        # Ignore devices outside this explicitly reserved measurement.
        if uuid in selected:
            gpu = selected[uuid]
            status["gpu_peak_used_mib"][gpu] = max(status["gpu_peak_used_mib"].get(gpu, 0), int(memory))
            status["gpu_peak_utilization_percent"][gpu] = max(status["gpu_peak_utilization_percent"].get(gpu, 0), int(utilization))
    rows = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", 
        "--format=csv,noheader,nounits"
    ], text=True, timeout=15).splitlines()
    counts = {str(gpu["gpu_id"]): 0 for gpu in identity["gpus"]}
    for row in rows:
        uuid, pid, memory = [part.strip() for part in row.split(",")]
        # Count processes only on the GPUs whose ownership admission verified.
        if uuid in selected:
            counts[selected[uuid]] += 1
            # NVIDIA may report unavailable memory for short-lived processes.
            if memory.isdigit():
                status["gpu_process_peak_used_mib"][pid] = max(status["gpu_process_peak_used_mib"].get(pid, 0), int(memory))
    status["max_observed_gpu_processes"] = max(status["max_observed_gpu_processes"], sum(counts.values()))
    for gpu, count in counts.items():
        status["gpu_peak_worker_count"][gpu] = max(status["gpu_peak_worker_count"].get(gpu, 0), count)
    # Device-by-device peaks alone cannot prove simultaneous full-container load.
    if all(counts[str(gpu["gpu_id"])] >= gpu["concurrent_trials"] for gpu in identity["gpus"]):
        status["observed_full_simultaneous_quota"] = True
    for relative in ["memory.current", "memory.peak", "memory.max"]:
        path = Path("/sys/fs/cgroup") / relative
        # Record effective cgroup-v2 memory limits when this host exposes them.
        if path.exists():
            raw = path.read_text().strip()
            # The literal 'max' represents an unbounded controller setting.
            if raw.isdigit():
                status["cgroup_memory_bytes"][relative] = max(status["cgroup_memory_bytes"].get(relative, 0), int(raw))


def run_check(arguments: argparse.Namespace) -> dict[str, Any]:
    """Reserve selected quotas and cache only an exact completed measurement."""

    started = time.time()
    deadline = started + arguments.max_seconds
    absolute_deadline = getattr(arguments, "deadline_unix", None)
    # A notebook clock includes this preflight within its existing search budget.
    if absolute_deadline is not None:
        deadline = min(deadline, absolute_deadline)
    root = Path(arguments.checkout).resolve()
    output = Path(arguments.output).resolve()
    gpu_ids = list(arguments.gpu_ids)
    concurrent_trials = len(gpu_ids) * arguments.trials_per_gpu
    branches = list(arguments.branches)
    trials = max(concurrent_trials, len(branches))
    sys.path.insert(0, str(root))
    os.chdir(root)
    from common.dit_hpo_remote import inspect_remote, launch_worker


    identity = inspect_remote(
        root, concurrent_trials=concurrent_trials, gpu_ids=gpu_ids, 
        worker_gpu_memory_limit_mb=arguments.memory_mib
    )
    # Reject a recycled or mistyped SSH endpoint rather than benchmarking another host.
    if arguments.expected_hostname is not None and identity["hostname"] != arguments.expected_hostname:
        raise RuntimeError("Capacity check is connected to a different container.")
    # This deployment's measured concurrency applies to the supplied A100 inventory.
    if any("A100" not in gpu["gpu_name"] for gpu in identity["gpus"]):
        raise RuntimeError("This capacity profile requires the selected A100 GPUs.")
    profile = {
        "dataset": "CIFAR100", "source_sha256": identity["source_sha256"], 
        "hostname": identity["hostname"], "versions": identity["versions"], 
        "python": identity["python"], "interpreter": str(Path(sys.executable).resolve()), 
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 
        "gpu_uuids": [gpu["gpu_uuid"] for gpu in identity["gpus"]], 
        "gpu_ids": gpu_ids, "concurrent_trials": concurrent_trials, 
        "trials_per_gpu": arguments.trials_per_gpu, "memory_mib": arguments.memory_mib, 
        "options": capacity_options(branches), "initial_trials": initial_trials(branches, trials), 
        "epochs": 1, "steps_per_epoch": 2, "validation_samples": 10000, 
        "train_samples_loaded": 50000, "dtype_policy": "float32"
    }
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    receipt = output / "receipt.json"
    # Repeated notebook execution may reuse only the exact successful measurement.
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        # Preserve failures and changed deployments as separate evidence.
        if previous.get("status") == "PASS" and previous.get("profile_sha256") == digest:
            print("Reusing the completed matching capacity check.", flush=True)
            return previous
        raise RuntimeError("Previous capacity evidence differs or did not pass; preserve it and choose a new output directory.")
    # Do not overwrite an interrupted measurement that lacks a final receipt.
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Capacity output contains evidence without a final receipt.")
    output.mkdir(parents=True, exist_ok=True)
    request = {
        "checkout": str(root), "identity": identity, "results": str(output / "results"), 
        "output": str(output), "deadline": deadline, "trials": trials, 
        "concurrent_trials": concurrent_trials, "memory_mib": arguments.memory_mib, "branches": branches
    }
    write_json(output / "request.json", request)
    status = {
        "status": "RUNNING", "profile_sha256": digest, "profile": profile, 
        "started_at_unix": started, "deadline_unix": deadline, 
        "gpu_peak_used_mib": {}, "gpu_peak_utilization_percent": {}, 
        "gpu_process_peak_used_mib": {}, "gpu_peak_worker_count": {}, 
        "cgroup_memory_bytes": {}, "max_observed_gpu_processes": 0, 
        "observed_full_simultaneous_quota": False
    }
    write_json(receipt, status)
    try:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", str(output / "request.json")]
        with launch_worker(command, root, identity, log_path=output / "worker.log", deadline=deadline) as process:
            while process.poll() is None:
                # Stop owned work before the caller's bounded deadline expires.
                if time.time() >= deadline:
                    raise TimeoutError("Bounded capacity measurement expired.")
                observe_gpus(status, identity)
                write_json(output / "progress.json", status)
                time.sleep(2)
            # A process failure cannot publish a successful capacity result.
            if process.returncode != 0:
                raise RuntimeError("Admitted capacity worker exited " + str(process.returncode))
        result = json.loads((output / "worker_result.json").read_text())
        # Both end-to-end completion and concurrent occupancy are required.
        if result.get("status") != "PASS" or not status["observed_full_simultaneous_quota"]:
            raise RuntimeError("Missing completed trials or simultaneous execution of every reserved GPU quota.")
        status.update({"status": "PASS", "result": result})
    except BaseException as error:
        status.update({"status": "FAIL", "error": type(error).__name__ + ": " + str(error)})
        raise
    finally:
        status["finished_at_unix"] = time.time()
        status["elapsed_seconds"] = status["finished_at_unix"] - started
        write_json(receipt, status)
    print(json.dumps(status, indent=2), flush=True)
    return status


def stop_handler(signum: int, frame: Any) -> None:
    """Unwind admission leases when the standalone parent receives SIGTERM."""

    raise InterruptedError("Capacity parent received signal " + str(signum))


def main() -> None:
    """Dispatch a CPU coordinator or authenticated child on the supplied host."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--trials-per-gpu", type=int, default=1)
    parser.add_argument("--memory-mib", type=int, default=18432)
    parser.add_argument("--branches", choices=BRANCHES, nargs="+", default=BRANCHES)
    parser.add_argument("--max-seconds", type=int, default=1200)
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--expected-hostname", default="81eff9282ac2")
    arguments = parser.parse_args()
    os.environ.update({
        "KERAS_HOME": "/workspace/.keras", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", 
        "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1", "MPLBACKEND": "Agg"
    })
    # Only the parent-created child may enter the authenticated HPO context.
    if arguments.child is not None:
        child(arguments.child)
    # Standalone invocation remains a CPU-only controller until admission succeeds.
    else:
        # Explicit paths prevent accidental work in an unrelated checkout or study.
        if arguments.checkout is None or arguments.output is None:
            parser.error("--checkout and --output are required")
        signal.signal(signal.SIGTERM, stop_handler)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        run_check(arguments)


# Importing this module for a notebook must not launch any work.
if __name__ == "__main__":
    main()
