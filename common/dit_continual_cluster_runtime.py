"""Run two admitted continual HPO shards from one persistent shared start request."""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import socket
import threading
import time

from common import dit_continual_cluster as cluster
from common import dit_continual_runner as continual
from common import dit_hpo_runner as shared
from common.dit_continual_candidates import candidate_profile
from common.dit_hpo_remote import inspect_remote
from common.hpo_process import study_lock
from common.dit_continual_storage import prepare_storage


def campaign_settings() -> dict:
    """Return independent editable defaults for the two eight-A100 hosts."""

    return {
        "hosts": {"a": "231305a26aca", "b": "86f2a28e21aa"}, 
        "epochs": 10, "search_hours": 8.0, "confirmation_hours": 4.0, 
        "gpu_ids": list(range(8)), "concurrent_trials": 24, "worker_gpu_memory_limit_mb": 24576, 
        "n_startup_trials": 10, "task_seed": 42, "search_seeds": {"a": 42, "b": 43}, 
        "planning_trials_per_shard": 48, "completion_ceiling_per_shard": 96, 
        "max_attempts_per_shard": 192, "batch_trials": 192, "top_k": 4, 
        "confirmation_seeds": {"a": [101, 202], "b": [303]}, 
        "search_space_overrides": {
            "continual_strategy": ["generative_replay"], "replay_budget_mode": ["fixed_total"], 
            "batch_size": [128], "replay_old_examples": [100, 500, 1000], 
            "replay_current_examples": [1000, 2500, 5000], "replay_candidate_multiplier": [1, 2], 
            "test_steps": [20, 50], "test_eta": [0.0], 
            "replay_selection": ["all", "uniform", "confidence"], 
            "learning_rate": {"low": 3e-4, "high": 2e-3, "log": True}
        }
    }


def prepare_cluster(campaign_root: str | Path, checkout_root: str | Path, settings: dict | None = None) -> dict:
    """Freeze a campaign request without starting its clock or any training."""

    root = Path(campaign_root).resolve()
    settings = campaign_settings() if settings is None else deepcopy(settings)
    # The fixed paired experiment requires both uniquely identified hosts and seeds.
    if set(settings["hosts"]) != {"a", "b"} or len(set(settings["hosts"].values())) != 2 \
    or set(settings["search_seeds"]) != {"a", "b"} or set(settings["confirmation_seeds"]) != {"a", "b"}:
        raise ValueError("The campaign requires distinct a/b hosts and both seed allocations.")
    record = cluster._seal({
        "version": 1, "campaign_root": str(root), "checkout_root": str(Path(checkout_root).resolve()), 
        "settings": settings
    })
    with study_lock(root, blocking=True):
        return cluster._publish_once(root / "request.json", record)


def _request(root: Path) -> dict:
    """Authenticate the shared request at every lifecycle boundary."""

    request = cluster._read(root / "request.json")
    # Relocated control files cannot silently redirect a previously sealed campaign.
    if request["campaign_root"] != str(root.resolve()):
        raise ValueError("Campaign path differs from the immutable request.")
    return request


def _plan(request: dict, shard_id: str) -> dict:
    """Prepare the local independent study using one shared scientific recipe."""

    settings = request["settings"]
    return continual.make_plan(
        request["checkout_root"], Path(request["campaign_root"]) / "shards" / shard_id, 
        epochs=settings["epochs"], n_startup_trials=settings["n_startup_trials"], 
        concurrent_trials=settings["concurrent_trials"], gpu_ids=settings["gpu_ids"], 
        validation_source="test", validation_ratio=0.0, 
        experiment_hours=settings["search_hours"] + settings["confirmation_hours"], 
        confirmation_reserve_hours=settings["confirmation_hours"], 
        search_space_overrides=settings["search_space_overrides"], 
        worker_gpu_memory_limit_mb=settings["worker_gpu_memory_limit_mb"], 
        search_profile="dit_continual_runner", continual_profile=candidate_profile(task_seed=settings["task_seed"]), 
        seed=settings["search_seeds"][shard_id]
    )


def _idle_identity(identity: dict, settings: dict) -> None:
    """Require eight empty A100 80GB devices and enough measured free reservation memory."""

    devices = identity["gpus"]
    # Readiness is an exact host inventory, not a model-name capacity inference.
    if len(devices) != len(settings["gpu_ids"]) or len({gpu["gpu_uuid"] for gpu in devices}) != len(devices):
        raise RuntimeError("Readiness needs distinct selected physical GPUs.")
    for gpu in devices:
        required = gpu["concurrent_trials"] * gpu["worker_policy"]["reserved_memory_mib"] + 2048
        # Existing processes are preserved, and a busy host cannot advertise readiness.
        if "A100" not in gpu["gpu_name"].upper() or gpu["total_mib"] < 80000 \
        or gpu["processes"] or gpu["free_mib"] < required:
            raise RuntimeError("Selected GPUs must be idle A100 80GB devices with sufficient free memory.")


def _enable_storage(plan: dict, shard_id: str) -> dict:
    """Initialize SQLite only on node-local disk before publishing immutable snapshots."""

    local = Path("/tmp/dit_continual_cluster_sqlite") / shard_id
    return prepare_storage(plan, local_root=local)


def _readiness(root: Path, request: dict, shard_id: str, plan: dict, phase: str) -> dict:
    """Publish live service identity and, while waiting, freshly inspect all devices."""

    identity = plan["identity"]
    # Waiting services repeatedly verify inventory; heartbeats do not merely replay old free memory.
    if phase == "ready":
        settings = request["settings"]
        identity = inspect_remote(
            request["checkout_root"], concurrent_trials=settings["concurrent_trials"], 
            gpu_ids=settings["gpu_ids"], worker_gpu_memory_limit_mb=settings["worker_gpu_memory_limit_mb"]
        )
        _idle_identity(identity, settings)
        for key in ["hostname", "source_sha256", "versions", "python"]:
            # Editing the frozen source while waiting requires a new verified service.
            if identity[key] != plan["identity"][key]:
                raise RuntimeError("Waiting service identity changed: " + key)
    record = {
        "shard_id": shard_id, "phase": phase, "heartbeat_unix": time.time(), "pid": os.getpid(), 
        "request_sha256": request["sha256"], "identity": identity, "recipe": cluster._recipe(plan)
    }
    shared._write(root / "services" / shard_id / "status.json", record)
    return record


def launch_cluster(campaign_root: str | Path, started_at_unix: float | None = None) -> dict:
    """Start both freshly ready hosts once; repeated launches never extend the clock."""

    root = Path(campaign_root).resolve()
    request = _request(root)
    # A recovered pre-training launch can retain an earlier deadline, never extend it into the future.
    if started_at_unix is not None and (not math.isfinite(started_at_unix) or started_at_unix > time.time()):
        raise ValueError("A recovered campaign start must be finite and no later than now.")
    with study_lock(root, blocking=True):
        start_path = root / "START.json"
        # Idempotent notebook reruns keep the original start time even after services finish.
        if start_path.exists():
            start = cluster._read(start_path)
            # Existing launch requests must retain the frozen campaign identity.
            if start["request_sha256"] != request["sha256"]:
                raise ValueError("The start command belongs to a different campaign request.")
            # Replaying an explicit recovery cannot change an already published start.
            if started_at_unix is not None and start["started_at_unix"] != float(started_at_unix):
                raise ValueError("The recovered start differs from the existing campaign clock.")
            return start
        statuses = [shared._read(root / "services" / shard / "status.json") for shard in ["a", "b"]]
        uuids = []
        for shard, status in zip(["a", "b"], statuses):
            identity = status["identity"]
            # A stale heartbeat or wrong host cannot cause a partially ready campaign launch.
            if status["phase"] != "ready" or not 0 <= time.time() - status["heartbeat_unix"] <= 60 \
            or status["request_sha256"] != request["sha256"] or status["shard_id"] != shard \
            or identity["hostname"] != request["settings"]["hosts"][shard]:
                raise RuntimeError("Both expected services must have fresh matching readiness.")
            _idle_identity(identity, request["settings"])
            uuids.extend(gpu["gpu_uuid"] for gpu in identity["gpus"])
        # Independent studies may differ in proposal seeds, never in scoring or source/runtime identity.
        if statuses[0]["recipe"] != statuses[1]["recipe"] or len(set(uuids)) != 16:
            raise RuntimeError("Hosts require matching recipes and 16 distinct physical GPUs.")
        start = cluster._seal({
            "version": 1, "request_sha256": request["sha256"], 
            "started_at_unix": time.time() if started_at_unix is None else float(started_at_unix), 
            "hostnames": request["settings"]["hosts"], "gpu_uuids": uuids
        })
        return cluster._publish_once(start_path, start)


def cluster_status(campaign_root: str | Path) -> dict:
    """Read published service and experiment state without touching SQLite or GPU workers."""

    root = Path(campaign_root).resolve()
    result = {}
    for name in ["request.json", "START.json", "campaign_clock.json", "global_finalists.json", "global_confirmation_summary.json"]:
        path = root / name
        # Missing lifecycle artifacts describe a pending phase rather than fabricated completion.
        if path.exists():
            result[name.removesuffix(".json")] = shared._read(path)
    result["services"] = {}
    for shard in ["a", "b"]:
        path = root / "services" / shard / "status.json"
        # A missing service is explicitly distinguishable from an idle ready host.
        if path.exists():
            result["services"][shard] = shared._read(path)
    return result


def _wait_files(root: Path, paths: list[Path], deadline: float) -> None:
    """Wait within the shared deadline and stop promptly after a peer failure."""

    while not all(path.exists() for path in paths):
        for shard in ["a", "b"]:
            path = root / "services" / shard / "status.json"
            # A failed peer cannot leave the other host waiting for an impossible artifact.
            if path.exists() and shared._read(path).get("phase") == "failed":
                raise RuntimeError("Peer service failed; inspect its status and log: " + shard)
        # Cleanup needs time within the same absolute campaign budget.
        if time.time() >= deadline - 5:
            raise TimeoutError("Shared campaign deadline reached while waiting for peer artifacts.")
        time.sleep(2)


def serve_shard(campaign_root: str | Path, checkout_root: str | Path, shard_id: str, expected_hostname: str) -> None:
    """Wait ready without training, then execute search and paired global confirmations."""

    root = Path(campaign_root).resolve()
    request = _request(root)
    settings = request["settings"]
    # Service routing is bound to the actual remote container, not a notebook label.
    if shard_id not in settings["hosts"] or socket.gethostname() != expected_hostname \
    or settings["hosts"][shard_id] != expected_hostname or str(Path(checkout_root).resolve()) != request["checkout_root"]:
        raise RuntimeError("Service hostname, shard or checkout does not match the sealed campaign.")
    service_root = root / "services" / shard_id
    with study_lock(service_root):
        phase = {"value": "preparing"}
        stopped = threading.Event()
        publication_lock = threading.Lock()
        plan = _plan(request, shard_id)
        _enable_storage(plan, shard_id)
        shared._write(service_root / "plan.json", plan)

        def publish() -> None:
            """Serialize status writes from lifecycle and heartbeat threads."""

            with publication_lock:
                _readiness(root, request, shard_id, plan, phase["value"])

        def heartbeat() -> None:
            """Keep executing services visible while their admitted workers run."""

            while not stopped.wait(15):
                # Ready polling owns its own live inspection before START.
                if phase["value"] != "ready":
                    publish()

        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            phase["value"] = "ready"
            while not (root / "START.json").exists():
                publish()
                stopped.wait(5)
            start = cluster._read(root / "START.json")
            # The shared start's identity and clock are immutable across service restarts.
            if start["request_sha256"] != request["sha256"]:
                raise ValueError("START belongs to another request.")
            # The leader publishes the first clock without relying on cross-host flock semantics.
            if shard_id == "b":
                end = start["started_at_unix"] + 3600 * (settings["search_hours"] + settings["confirmation_hours"])
                _wait_files(root, [root / "campaign_clock.json"], end)
            clock = cluster.start_shared_campaign(plan, root, shard_id, started_at_unix=start["started_at_unix"])
            deadline = clock["budget"]["deadline_unix"]
            phase["value"] = "search"
            publish()
            continual.run_search(
                plan, target_completed=settings["completion_ceiling_per_shard"], 
                max_attempts=settings["max_attempts_per_shard"], batch_trials=settings["batch_trials"]
            )
            export_path = root / f"search_{shard_id}.json"
            cluster.export_search(plan, shard_id, export_path)
            phase["value"] = "waiting_for_search_exports"
            publish()
            exports = [root / f"search_{shard}.json" for shard in ["a", "b"]]
            _wait_files(root, exports, deadline)
            manifest_path = root / "global_finalists.json"
            # One designated host freezes global ranks; the second verifies its immutable manifest.
            if shard_id == "a":
                cluster.freeze_global_finalists(
                    exports, manifest_path, top_k=settings["top_k"], seeds_by_shard=settings["confirmation_seeds"]
                )
            _wait_files(root, [manifest_path], deadline)
            cluster.install_global_finalists(plan, manifest_path, shard_id)
            phase["value"] = "confirmations"
            publish()
            results = continual.run_confirmations(plan)
            shared._write(root / f"confirmations_{shard_id}.json", {"shard_id": shard_id, "results": results})
            phase["value"] = "waiting_for_confirmation_exports"
            publish()
            _wait_files(root, [root / f"confirmations_{shard}.json" for shard in ["a", "b"]], deadline)
            # Receipts preserve missing seed pairs instead of presenting them as complete.
            if shard_id == "a":
                plans = {shard: shared._read(root / "services" / shard / "plan.json") for shard in ["a", "b"]}
                cluster.merge_global_confirmations(plans, manifest_path, root / "global_confirmation_summary.json")
            phase["value"] = "complete"
            publish()
        except BaseException as error:
            phase["value"] = "failed"
            stopped.set()
            worker.join(timeout=20)
            with publication_lock:
                status = _readiness(root, request, shard_id, plan, "failed")
                shared._write(service_root / "status.json", {**status, "error": repr(error)})
            raise
        finally:
            stopped.set()
            worker.join(timeout=20)


def main() -> None:
    """Run one durable remote host service from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--checkout-root", required=True)
    parser.add_argument("--shard-id", choices=["a", "b"], required=True)
    parser.add_argument("--expected-hostname", required=True)
    options = parser.parse_args()
    serve_shard(options.campaign_root, options.checkout_root, options.shard_id, options.expected_hostname)


# Importing the runtime exposes notebook APIs without starting a service.
if __name__ == "__main__":
    main()
