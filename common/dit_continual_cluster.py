"""Coordinate independent continual HPO shards through immutable shared artifacts.

Each host retains its own SQLite study, process locks and GPU reservations. SHA-256
envelopes detect artifact changes; they are integrity checks, not signatures.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from common import dit_continual_runner as continual
from common import dit_hpo_runner as shared
from common.hpo_process import study_lock


def _hash(value: dict) -> str:
    """Hash a finite, deterministic JSON representation."""

    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _seal(value: dict) -> dict:
    """Attach an integrity digest without mutating the supplied payload."""

    return {**value, "sha256": _hash(value)}


def _read(value: dict | str | Path) -> dict:
    """Load an envelope and reject any changed payload or missing digest."""

    record = deepcopy(value) if isinstance(value, dict) else shared._read(value)
    digest = record.pop("sha256", None)
    # Every transferred artifact passes an integrity check before fields are consumed.
    if digest != _hash(record):
        raise ValueError("Cross-host artifact SHA-256 mismatch.")
    return {**record, "sha256": digest}


def _recipe(plan: dict) -> dict:
    """Keep scientific settings and source identity while excluding shard execution."""

    continual._validate_plan(plan)
    execution = {"seed", "results_path", "concurrent_trials", "worker_gpu_ids", "worker_gpu_memory_limit_mb"}
    recipe = {
        "version": plan["version"], 
        "hpo": {key: deepcopy(value) for key, value in plan["hpo"].items() if key not in execution}, 
        "source_sha256": deepcopy(plan["identity"]["source_sha256"]), 
        "versions": deepcopy(plan["identity"]["versions"]), 
        "python": plan["identity"]["python"]
    }
    # Compare the persisted JSON form, including numeric feature-map keys and tuple values.
    return json.loads(json.dumps(recipe, allow_nan=False))


def _publish_once(path: str | Path, record: dict) -> dict:
    """Publish an immutable record or verify an identical completed publication."""

    path = Path(path)
    # A restart may reuse the exact artifact but cannot silently select new inputs.
    if path.exists():
        # Existing immutable content must match the complete proposed artifact.
        if shared._read(path) != record:
            raise ValueError(f"An immutable campaign artifact changed: {path}")
    # The owning coordinator has already excluded competing writers.
    else:
        shared._write(path, record)
    return record


def start_shared_campaign(
    plan: dict, campaign_root: str | Path, shard_id: str, started_at_unix: float | None = None
) -> dict:
    """Create or adopt one persistent wall clock before either shard launches work.

    An explicit start binds the clock to the shared START command. Otherwise the
    first caller starts it. Both hosts must finish readiness checks before calling.
    Repeated calls never extend time.
    """

    root = Path(campaign_root)
    root.mkdir(parents=True, exist_ok=True)
    policy = plan.get("time_budget")
    # An untimed shard cannot participate in a shared finite campaign.
    if policy is None:
        raise ValueError("A shared campaign requires a finite experiment time budget.")
    recipe = _recipe(plan)
    with study_lock(root, blocking=True):
        clock_path = root / "campaign_clock.json"
        # An established campaign clock is authoritative on both hosts.
        if clock_path.exists():
            clock = _read(clock_path)
            # Hosts can share the clock only when science and phase budgets agree.
            if clock["recipe"] != recipe or clock["budget"]["policy"] != policy:
                raise ValueError("Shared campaign recipe or time policy changed.")
            # A replayed START command cannot move an existing campaign clock.
            if started_at_unix is not None and clock["budget"]["started_at_unix"] != float(started_at_unix):
                raise ValueError("Shared campaign START time changed.")
        # Only the first admitted setup allocates an absolute start time.
        else:
            started = time.time() if started_at_unix is None else float(started_at_unix)
            deadline = started + 3600.0 * policy["experiment_hours"]
            clock = _seal({
                "version": 1, "recipe": recipe, 
                "budget": {
                    "version": 1, "policy": deepcopy(policy), "started_at_unix": started, 
                    "deadline_unix": deadline, 
                    "search_deadline_unix": deadline - 3600.0 * policy["confirmation_reserve_hours"], 
                    "cleanup_seconds": 60.0
                }
            })
            _publish_once(clock_path, clock)
        with shared._coordinator(plan):
            _publish_once(Path(plan["control_root"]) / "budget.json", clock["budget"])
            _publish_once(Path(plan["control_root"]) / "cluster_campaign.json", {
                "shard_id": shard_id, "clock_sha256": clock["sha256"], 
                "campaign_root": str(root.resolve())
            })
    return clock


def export_search(plan: dict, shard_id: str, output_path: str | Path) -> dict:
    """Export finite completed candidates and exact YAML bytes after a shard stops."""

    with shared._coordinator(plan):
        recipe = _recipe(plan)
        campaign = shared._read(Path(plan["control_root"]) / "cluster_campaign.json")
        # Shard labels are immutable provenance, not merely output filenames.
        if campaign["shard_id"] != shard_id:
            raise ValueError("Search export has a different shard identity.")
        study = shared._load_study(plan)
        trials = [] if study is None else study.get_trials(deepcopy=False)
        # Frozen exports must never race active trials that could improve selection.
        if any(trial.state.name == "RUNNING" for trial in trials):
            raise ValueError("Stop all search workers before exporting a shard.")
        waiting = [trial.number for trial in trials if trial.state.name == "WAITING"]
        deadline = shared._phase_deadline(plan, "search")
        # Unstarted queued hints can be excluded only after the search cutoff.
        if waiting and (deadline is None or time.time() < deadline - 60.0):
            raise ValueError("Queued trials remain before the search deadline.")
        candidates = []
        for trial in shared._ranked(study, direction="maximize"):
            path = Path(plan["study_root"]) / "configs" / f"trial-{trial.number:04d}.yaml"
            config_text = path.read_bytes().decode("utf-8")
            candidates.append({
                "source_shard": shard_id, "source_trial_number": trial.number, 
                "params": deepcopy(trial.params), "search_final_average_accuracy": float(trial.value), 
                "config_text": config_text, 
                "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest()
            })
        record = _seal({
            "version": 1, "kind": "continual_search_shard", "shard_id": shard_id, 
            "search_seed": plan["hpo"]["seed"], "recipe": recipe, 
            "clock_sha256": campaign["clock_sha256"], "budget": shared._budget_state(plan), 
            "summary": continual.search_summary(plan), "unstarted_trial_numbers": waiting, 
            "candidates": candidates
        })
        return _publish_once(output_path, record)


def freeze_global_finalists(
    exports: list[dict | str | Path], output_path: str | Path, top_k: int = 4, 
    seeds_by_shard: dict[str, list[int]] | None = None
) -> dict:
    """Rank both completed studies globally and freeze distinct recipes and paired seeds."""

    records = [_read(value) for value in exports]
    seeds_by_shard = {"a": [101, 202], "b": [303]} if seeds_by_shard is None else deepcopy(seeds_by_shard)
    shards = [record["shard_id"] for record in records]
    # Global selection requires exactly one completed export from every planned shard.
    if not records or len(set(shards)) != len(shards) or set(shards) != set(seeds_by_shard):
        raise ValueError("Export shards must match the unique confirmation shard allocation.")
    reference = records[0]
    for record in records:
        # Scores are comparable only under identical scientific and timing contracts.
        if record["kind"] != "continual_search_shard" or any(
            record[key] != reference[key] for key in ["recipe", "clock_sha256", "budget"]
        ):
            raise ValueError("Shard recipes, source/runtime identities or campaign clocks differ.")
    seeds = [seed for shard_seeds in seeds_by_shard.values() for seed in shard_seeds]
    # Each candidate receives distinct paired repeats unrelated to search initialization.
    if any(not values for values in seeds_by_shard.values()) or any(type(seed) is not int for seed in seeds) \
    or len(set(seeds)) != len(seeds) or any(record["search_seed"] in seeds for record in records):
        raise ValueError("Confirmation seeds must be distinct integers separate from search seeds.")
    candidates = []
    for record in records:
        trial_numbers = set()
        for candidate in record["candidates"]:
            score = candidate["search_final_average_accuracy"]
            # Envelope integrity does not replace the finite objective and YAML contracts.
            if candidate["source_shard"] != record["shard_id"] or isinstance(score, bool) \
            or not isinstance(score, (int, float)) or not math.isfinite(score) \
            or type(candidate["source_trial_number"]) is not int \
            or candidate["source_trial_number"] in trial_numbers \
            or hashlib.sha256(candidate["config_text"].encode("utf-8")).hexdigest() != candidate["config_sha256"]:
                raise ValueError("Invalid candidate objective, shard or configuration identity.")
            trial_numbers.add(candidate["source_trial_number"])
            candidates.append(candidate)
    candidates.sort(key=lambda row: (-row["search_final_average_accuracy"], row["source_shard"], row["source_trial_number"]))
    selected = []
    seen = set()
    for candidate in candidates:
        identity = _hash(candidate["params"])
        # Replicated parameter configurations receive only one global finalist slot.
        if identity not in seen:
            seen.add(identity)
            selected.append({**candidate, "global_candidate_id": len(selected) + 1})
    # The stated finalist count cannot silently shrink after failed searches.
    if type(top_k) is not int or top_k < 1 or len(selected) < top_k:
        raise ValueError("Not enough distinct finite completed configurations for global selection.")
    manifest = _seal({
        "version": 1, "kind": "continual_global_finalists", 
        "recipe": reference["recipe"], "clock_sha256": reference["clock_sha256"], 
        "budget": reference["budget"], "top_k": top_k, "seeds_by_shard": seeds_by_shard, 
        "exports_sha256": {record["shard_id"]: record["sha256"] for record in records}, 
        "search_seeds": {record["shard_id"]: record["search_seed"] for record in records}, 
        "candidates": selected[:top_k]
    })
    with study_lock(Path(output_path).parent, blocking=True):
        return _publish_once(output_path, manifest)


def install_global_finalists(plan: dict, manifest: dict | str | Path, shard_id: str) -> dict:
    """Install portable global recipes with this host's paired-seed assignment."""

    global_manifest = _read(manifest)
    campaign = shared._read(Path(plan["control_root"]) / "cluster_campaign.json")
    # Host routing remains local while scientific settings and deadlines stay global.
    if global_manifest["kind"] != "continual_global_finalists" or global_manifest["recipe"] != _recipe(plan) \
    or global_manifest["clock_sha256"] != campaign["clock_sha256"] \
    or global_manifest["budget"] != shared._budget_state(plan) or campaign["shard_id"] != shard_id \
    or global_manifest["search_seeds"].get(shard_id) != plan["hpo"]["seed"]:
        raise ValueError("Global finalists do not match this shard's sealed campaign.")
    with shared._coordinator(plan):
        candidates = []
        for candidate in global_manifest["candidates"]:
            number = candidate["global_candidate_id"]
            destination = Path(plan["control_root"]) / "finalist_configs" / f"global-{number:04d}.yaml"
            data = candidate["config_text"].encode("utf-8")
            # Portable YAMLs preserve their original source trial and initialization recipe.
            if hashlib.sha256(data).hexdigest() != candidate["config_sha256"]:
                raise ValueError("Global candidate YAML SHA-256 mismatch.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Resume accepts only the previously installed identical YAML bytes.
            if destination.exists():
                # Integrity is rechecked even when the file was installed earlier.
                if destination.read_bytes() != data:
                    raise ValueError("An installed global finalist configuration changed.")
            # Publish a previously absent configuration atomically.
            else:
                temporary = destination.with_suffix(".tmp")
                temporary.write_bytes(data)
                temporary.replace(destination)
            candidates.append({
                "rank": number, "trial_number": number, "global_candidate_id": number, 
                "source_shard": candidate["source_shard"], "source_trial_number": candidate["source_trial_number"], 
                "params": candidate["params"], "search_final_average_accuracy": candidate["search_final_average_accuracy"], 
                "input_config_path": str(destination.resolve()), "config_sha256": candidate["config_sha256"]
            })
        local = {
            "study_name": plan["study_name"], "global_manifest_sha256": global_manifest["sha256"], 
            "shard_id": shard_id, "top_k": global_manifest["top_k"], 
            "seeds": global_manifest["seeds_by_shard"][shard_id], 
            "dataset_seed": plan["hpo"]["continual_profile"]["task_seed"], 
            "task_groups": plan["hpo"]["continual_profile"]["task_groups"], 
            "selection_policy": {
                "objective_metric": continual.METRIC, "objective_direction": "maximize", 
                "objective_network": "raw", "evaluation_source": "official_test"
            }, 
            "candidates": candidates
        }
        return _publish_once(Path(plan["control_root"]) / "finalists.json", local)


def merge_global_confirmations(
    plans_by_shard: dict[str, dict], manifest: dict | str | Path, output_path: str | Path
) -> dict:
    """Authenticate local receipts and report global paired means without filling gaps."""

    global_manifest = _read(manifest)
    # Partial reporting may omit receipts, but never an entire planned host identity.
    if set(plans_by_shard) != set(global_manifest["seeds_by_shard"]):
        raise ValueError("Global summary requires the plans for every campaign shard.")
    scores = {candidate["global_candidate_id"]: {} for candidate in global_manifest["candidates"]}
    for shard_id, plan in plans_by_shard.items():
        local = install_global_finalists(plan, global_manifest, shard_id)
        manifest_path = Path(plan["control_root"]) / "finalists.json"
        for candidate in local["candidates"]:
            for seed in local["seeds"]:
                path = Path(plan["control_root"]) / "confirmations" / f"trial-{candidate['trial_number']:04d}" / f"seed-{seed}" / "completed.json"
                # Unfinished confirmations remain absent rather than becoming zero scores.
                if not path.exists():
                    continue
                expected = {
                    "manifest_sha256": shared._digest(manifest_path), 
                    "config_sha256": candidate["config_sha256"], 
                    "trial_number": candidate["trial_number"], "training_seed": seed
                }
                record = shared._completed_record(path, expected, plan=plan)
                result = record["result"]
                # The local receipt must retain the global YAML and its original source identity.
                if result.get("source_input_config_sha256") != candidate["config_sha256"] \
                or result.get("source_trial_number") != candidate["source_trial_number"] \
                or result.get("training_seed") != seed \
                or result.get("dataset_seed") != local["dataset_seed"]:
                    raise ValueError("Confirmation receipt has a different global source or paired seed.")
                scores[candidate["global_candidate_id"]][seed] = float(result["objective"])
    required = sum(len(values) for values in global_manifest["seeds_by_shard"].values())
    rows = []
    for candidate in global_manifest["candidates"]:
        observed = scores[candidate["global_candidate_id"]]
        values = list(observed.values())
        rows.append({
            "global_candidate_id": candidate["global_candidate_id"], 
            "source_shard": candidate["source_shard"], "source_trial_number": candidate["source_trial_number"], 
            "search_final_average_accuracy": candidate["search_final_average_accuracy"], 
            "scores_by_seed": {str(seed): score for seed, score in sorted(observed.items())}, 
            "completed_seeds": len(values), "required_seeds": required, 
            "mean_final_average_accuracy": statistics.mean(values) if values else None, 
            "std_final_average_accuracy": statistics.stdev(values) if len(values) > 1 else None, 
            "all_seeds_complete": len(values) == required, "comparable": len(values) == required
        })
    result = _seal({
        "version": 1, "global_manifest_sha256": global_manifest["sha256"], 
        "evaluation_source": "official_test", "all_pairs_complete": all(row["all_seeds_complete"] for row in rows), 
        "candidates": rows
    })
    with study_lock(Path(output_path).parent, blocking=True):
        shared._write(output_path, result)
    return result
