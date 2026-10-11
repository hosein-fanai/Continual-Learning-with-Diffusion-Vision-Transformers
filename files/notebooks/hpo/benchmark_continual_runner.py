"""Run one admitted remote continual pilot with its own bounded execution clock.

This uses real CIFAR10, five tasks, the largest joint student/classifier recipe,
and the selected UNet teacher. Defaults retain the costly replay stress case:
two epochs per phase, 5000 old/current examples, candidate multiplier two,
50 replay steps, and a thirty-minute clock. Explicit CLI options can shorten
replay for a task-growth probe. Recorded timing describes only the selected pilot
settings. All output is preserved in a new /tmp directory; the twelve-hour
campaign and its clock are untouched. Optional concurrent repetitions measure
single-GPU admission and throughput for the same fixed recipe and seed; they are
capacity observations rather than independent HPO evidence.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback


class _Tee:
    """Preserve coordinator output while retaining the launcher's console stream."""

    def __init__(self, terminal: object, logfile: object) -> None:
        """Keep both externally owned streams open for the entire pilot."""

        self.terminal = terminal
        self.logfile = logfile

    def write(self, value: str) -> int:
        """Write the same coordinator text to the terminal and persistent log."""

        self.terminal.write(value)
        self.logfile.write(value)
        self.flush()
        return len(value)

    def flush(self) -> None:
        """Flush both destinations so a timeout retains recent progress."""

        self.terminal.flush()
        self.logfile.flush()


def _utc_now() -> str:
    """Return an explicit UTC timestamp independent of the container timezone."""

    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value: dict) -> None:
    """Atomically publish one finite control/result artifact in the pilot directory."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _telemetry(path: Path, gpu_uuid: str, stop: threading.Event) -> None:
    """Record device memory/utilization without taking ownership of any process."""

    with path.open("a", encoding="utf-8") as stream:
        while not stop.is_set():
            row = {"timestamp_utc": _utc_now(), "gpu_uuid": gpu_uuid}
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi", "-i", gpu_uuid, 
                        "--query-gpu=uuid,memory.used,memory.free,utilization.gpu", 
                        "--format=csv,noheader,nounits"
                    ], 
                    capture_output=True, text=True, timeout=10, check=True
                )
                row["gpu_csv"] = result.stdout.strip()
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            stop.wait(15)


def _timing_artifacts(output: Path, epochs: int) -> dict:
    """Collect CSV and committed-checkpoint task timers without extrapolating."""

    records = []
    for path in sorted(output.rglob("task_metrics.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                # Keep actual phase timers separate from losses and other diagnostics.
                if row.get("phase") == "resource" and row.get("metric", "").startswith("seconds/"):
                    records.append({"source_path": str(path), **row})
    checkpoint_records = {}
    checkpoint_read_errors = []
    checkpoint_paths = sorted(
        [path for path in output.rglob("state.json") if path.parent.name.startswith("task")], 
        key=lambda path: (path.stat().st_mtime_ns, str(path))
    )
    for path in checkpoint_paths:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            for record in state.get("experiment_state", {}).get("task_resource_metrics", []):
                # Later states supersede the same task within its own trial checkpoint root.
                if isinstance(record, dict) and "task_index" in record:
                    checkpoint_root = str(path.parent.parent)
                    checkpoint_records[(checkpoint_root, record["task_index"])] = {
                        **record, "source_path": str(path), "checkpoint_root": checkpoint_root, 
                        "checkpoint_state_modified_ns": path.stat().st_mtime_ns
                    }
        except (OSError, ValueError, TypeError, AttributeError) as error:
            checkpoint_read_errors.append({"source_path": str(path), "error": f"{type(error).__name__}: {error}"})
    return {
        "task_timing_rows": records, 
        "checkpoint_task_resource_metrics": [checkpoint_records[index] for index in sorted(checkpoint_records)], 
        "checkpoint_read_errors": checkpoint_read_errors, 
        "pilot_epochs_per_task": epochs, "campaign_epochs_per_task": 20, 
        "twenty_epoch_trial_estimate_seconds": None, 
        "estimate_status": "not_extrapolated", 
        "reason": "Recorded phase timers describe only the selected pilot settings, including initialization/compilation; no campaign-duration extrapolation is inferred."
    }


def main() -> int:
    """Verify this supplied A100 and execute one bounded existing-API pilot."""

    # Reject laptop execution before inspecting the checkout or creating artifacts.
    if platform.system() != "Linux" or not any(
        path.is_file() for path in (Path("/.dockerenv"), Path("/run/.containerenv"))
    ):
        raise RuntimeError("This benchmark runs only on the supplied online Linux container.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, help="New output directory below /tmp; default is a unique directory.")
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-hostname", default="c8854e910483")
    parser.add_argument("--epochs", type=int, default=2, help="Positive epoch count per student/specialist task phase.")
    parser.add_argument("--replay-old-examples", type=int, default=5000, help="Fixed-total old examples from the supported replay search choices.")
    parser.add_argument("--replay-current-examples", type=int, default=5000, help="Fixed-total current examples from the supported replay search choices.")
    parser.add_argument("--replay-candidate-multiplier", type=int, default=2, help="Supported confidence-selection overgeneration multiplier.")
    parser.add_argument("--replay-steps", type=int, default=50, help="Supported reverse-sampling step choice.")
    parser.add_argument("--hours", type=float, default=0.5, help="Positive independent pilot wall-clock budget in hours.")
    parser.add_argument("--concurrent-trials", type=int, default=1, help="Admitted simultaneous fixed-recipe capacity workers on GPU 0.")
    parser.add_argument("--target-completed", type=int, default=1, help="Requested fixed-recipe completions; also the allocation ceiling.")
    options = parser.parse_args()
    root = options.checkout_root.resolve(strict=True)
    # A supplied identity is an execution boundary, not a hint for device fallback.
    if socket.gethostname() != options.expected_hostname:
        raise RuntimeError("The container hostname differs from the requested pilot host.")
    # The admitted runner requires the complete existing project implementation.
    if not (root / "common/hpo.py").is_file():
        raise RuntimeError("The pilot requires the complete supplied project checkout.")
    # Framework imports belong to admitted children rather than this coordinator.
    if "tensorflow" in sys.modules or "keras" in sys.modules:
        raise RuntimeError("Use a fresh coordinator process without TensorFlow or Keras imports.")
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("KERAS_HOME", "/workspace/.keras")
    # Default output receives a unique directory without replacing prior evidence.
    if options.output is None:
        output = Path(tempfile.mkdtemp(prefix="dit-continual-pilot-", dir="/tmp"))
    # Explicit output paths retain the same isolated temporary-root constraint.
    else:
        output = options.output.resolve()
        # The pilot may not write into the campaign or another filesystem root.
        if output == Path("/tmp") or not output.is_relative_to(Path("/tmp")):
            raise ValueError("Pilot output must be a new directory strictly below /tmp.")
        output.mkdir(parents=True, exist_ok=False)
    print(f"Pilot output: {output}", flush=True)
    started = time.monotonic()
    receipt = {
        "status": "starting", "started_at_utc": _utc_now(), "checkout_root": str(root), 
        "output_root": str(output), "expected_hostname": options.expected_hostname, 
        "expected_gpu_uuid": options.expected_gpu_uuid, "pid": os.getpid(), 
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 
        "keras_home": os.environ["KERAS_HOME"], "campaign_clock_used": False, 
        "pilot_parameters": {
            "epochs_per_task": options.epochs, "replay_old_examples": options.replay_old_examples, 
            "replay_current_examples": options.replay_current_examples, 
            "replay_candidate_multiplier": options.replay_candidate_multiplier, 
            "replay_steps": options.replay_steps, "experiment_hours": options.hours, 
            "concurrent_trials": options.concurrent_trials, "target_completed": options.target_completed
        }
    }
    _write(output / "result.json", receipt)
    stop = threading.Event()
    monitor = None
    plan = None
    exit_code = 2
    with (output / "coordinator.log").open("w", encoding="utf-8") as logfile, \
    redirect_stdout(_Tee(sys.stdout, logfile)), redirect_stderr(_Tee(sys.stderr, logfile)):
        try:
            from common.dit_continual_candidates import candidate_profile
            from common.dit_continual_runner import budget_summary, make_plan, run_search, search_summary, start_experiment
            from common.dit_hpo_remote import inspect_remote


            identity = inspect_remote(
                root, concurrent_trials=options.concurrent_trials, gpu_ids=[0], worker_gpu_memory_limit_mb=24576
            )
            _write(output / "runtime_before.json", identity)
            # Admission must bind the requested physical A100 and its observed memory.
            if identity["gpu_uuid"] != options.expected_gpu_uuid or "A100" not in identity["gpu_name"] \
            or identity["total_mib"] < 80000:
                raise RuntimeError("GPU UUID/model/memory differs from the supplied A100 80 GB pilot identity.")
            profile = candidate_profile(task_seed=42)
            for group, selected in (
                ("student_candidates", "joint165_noisy"), 
                ("classifier_teacher_candidates", "joint165_noisy"), 
                ("noise_teacher_candidates", "unet23")
            ):
                profile[group] = {selected: profile[group][selected]}
            overrides = {
                "student_config": ["joint165_noisy"], 
                "classifier_teacher_descriptor": ["joint165_noisy"], 
                "noise_teacher_descriptor": ["unet23"], 
                "classifier_teacher_source": ["both"], "noise_teacher_source": ["both"], 
                "continual_strategy": ["generative_replay"], "replay_budget_mode": ["fixed_total"], 
                "replay_old_examples": [options.replay_old_examples], 
                "replay_current_examples": [options.replay_current_examples], 
                "replay_candidate_multiplier": [options.replay_candidate_multiplier], "replay_selection": ["confidence"], 
                "test_steps": [options.replay_steps], "test_eta": [0.0], "optimizer": ["adam"], 
                "learning_rate": {"low": 0.001, "high": 0.001, "log": True}, 
                "batch_size": [128], "test_cfg_scale": {"low": 3.0, "high": 3.0}, 
                "p_uncond": [0.1], "clipnorm": [None], "global_clipnorm": [1.0]
            }
            _write(output / "request.json", {
                "epochs_per_task": options.epochs, "task_count": 5, 
                "target_completed": options.target_completed, "max_attempts": options.target_completed, 
                "experiment_hours": options.hours, "confirmation_reserve_hours": 0.0, 
                "worker_gpu_memory_limit_mb": 24576, "concurrent_trials": options.concurrent_trials, 
                "profile": profile, "search_space_overrides": overrides
            })
            plan = make_plan(
                root, output / "results", dataset_name="CIFAR10", epochs=options.epochs, n_startup_trials=1, 
                concurrent_trials=options.concurrent_trials, gpu_ids=[0], pruning=None, validation_source="test", validation_ratio=0.0, 
                experiment_hours=options.hours, confirmation_reserve_hours=0.0, search_space_overrides=overrides, 
                worker_gpu_memory_limit_mb=24576, search_profile="dit_continual_runner", 
                continual_profile=profile, seed=42
            )
            # Authenticate the identity sealed immediately before admitted execution.
            if plan["identity"]["gpu_uuid"] != options.expected_gpu_uuid:
                raise RuntimeError("The selected GPU identity changed while constructing the pilot plan.")
            _write(output / "plan.json", plan)
            receipt["study_root"] = plan["study_root"]
            receipt["time_budget"] = start_experiment(plan)
            receipt["status"] = "running"
            _write(output / "result.json", receipt)
            monitor = threading.Thread(
                target=_telemetry, args=(output / "gpu_telemetry.jsonl", options.expected_gpu_uuid, stop), 
                daemon=True, name="continual-pilot-telemetry"
            )
            monitor.start()
            search_started = time.monotonic()
            result = run_search(
                plan, target_completed=options.target_completed, max_attempts=options.target_completed, 
                batch_trials=options.target_completed
            )
            receipt["search_elapsed_seconds"] = time.monotonic() - search_started
            receipt["search_result"] = result
            receipt["status"] = "complete" if result["target_reached"] else (
                "timeout" if result["time_budget_exhausted"] else "incomplete"
            )
            exit_code = 0 if receipt["status"] == "complete" else 3
        except BaseException as error:
            receipt["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            receipt["error"] = f"{type(error).__name__}: {error}"
            receipt["traceback"] = traceback.format_exc()
            traceback.print_exc()
        finally:
            stop.set()
            # Join only a monitor that was created after successful pilot setup.
            if monitor is not None:
                monitor.join(timeout=15)
            receipt["finished_at_utc"] = _utc_now()
            receipt["total_elapsed_seconds"] = time.monotonic() - started
            receipt["twenty_epoch_capacity_proven"] = False
            # Failed setup has no valid study paths or durable execution clock.
            if plan is not None:
                try:
                    receipt["final_search_summary"] = search_summary(plan)
                    receipt["final_budget"] = budget_summary(plan)
                except Exception as error:
                    receipt["summary_error"] = f"{type(error).__name__}: {error}"
            try:
                _write(output / "timing.json", _timing_artifacts(output, options.epochs))
            except Exception as error:
                receipt["timing_collection_error"] = f"{type(error).__name__}: {error}"
            _write(output / "result.json", receipt)
            print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)
    return exit_code


# Importing the script defines the pilot without starting any workload.
if __name__ == "__main__":
    raise SystemExit(main())
