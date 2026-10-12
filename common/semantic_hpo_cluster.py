"""Exchange sealed semantic HPO results between independent remote host studies.

Only JSON envelopes and frozen YAMLs cross hosts. SQLite transactions, GPU
admission and worker locks remain local. All hosts adopt the same absolute
clock and rank complete validation streams before allocating paired repeats.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import time

from common import dit_hpo_runner as shared
from common import semantic_hpo_runner as semantic
from common.config import Config, _safe_load_unique_yaml
from common.dit_continual_cluster import _hash, _publish_once, _read, _seal
from common.hpo_process import study_lock
from common.hpo_sqlite import database_path
from common.semantic_hpo import validate_semantic_config


def _recipe(plan: dict, anchors_by_shard: dict[str, list[dict]] | None = None) -> dict:
    """Compare scientific inputs and implementation while excluding host routing."""

    semantic._validate_plan(plan)
    execution = {
        "sampler_seed", "results_path", "concurrent_trials", "worker_gpu_ids", "worker_gpu_memory_limit_mb"
    }
    # Queue partitioning is an explicit complete-campaign allocation, not dropped coverage.
    if anchors_by_shard is not None:
        from common.semantic_hpo_candidates import semantic_initial_trials


        master = semantic_initial_trials(plan["hpo"]["semantic_profile"])
        # Every host receives a concrete nonempty queue before the clock can start.
        if not isinstance(anchors_by_shard, dict) or len(anchors_by_shard) < 2 or any(
            not isinstance(key, str) or not key or not isinstance(values, list) or not values
            for key, values in anchors_by_shard.items()
        ):
            raise ValueError("Declare nonempty shard queues for every host in the complete canonical anchor set.")
        assigned = [anchor for anchors in anchors_by_shard.values() for anchor in anchors]
        identities = [_hash(anchor) for anchor in assigned]
        # Exact coverage prevents duplicate pilots or omitted native/semantic treatments.
        if len(set(identities)) != len(identities) or sorted(identities) != sorted(_hash(anchor) for anchor in master):
            raise ValueError("Shard anchors must cover the canonical full master exactly once without duplicates.")
        execution.add("initial_trials")
    recipe = {
        "version": plan["version"], 
        "hpo": {key: deepcopy(value) for key, value in plan["hpo"].items() if key not in execution}, 
        "source_sha256": deepcopy(plan["identity"]["source_sha256"]), 
        "versions": deepcopy(plan["identity"]["versions"]), "python": plan["identity"]["python"]
    }
    # The entire partition remains scientific provenance on every host.
    if anchors_by_shard is not None:
        recipe["anchors_by_shard"] = deepcopy(anchors_by_shard)
    return json.loads(json.dumps(recipe, allow_nan=False))


def _weighted_partition(items: list, shard_weights: dict[str, int]) -> dict[str, list]:
    """Allocate each discrete workload once, with an initial workload on every host."""

    # Weights are exact measured worker counts, not model-name capacity guesses.
    if not isinstance(shard_weights, dict) or len(shard_weights) < 2 or any(
        not isinstance(shard, str) or not shard or Path(shard).name != shard
        or type(weight) is not int or weight < 1 for shard, weight in shard_weights.items()
    ) or len(items) < len(shard_weights):
        raise ValueError("Provide positive integer worker counts and at least one workload per shard.")
    shards = sorted(shard_weights)
    allocation = {shard: [] for shard in shards}
    for index, item in enumerate(items):
        shard = shards[index] if index < len(shards) else min(
            shards, key=lambda key: ((len(allocation[key]) + 1) / shard_weights[key], key)
        )
        allocation[shard].append(deepcopy(item))
    return allocation


def partition_anchors(profile: dict, shard_weights: dict[str, int]) -> dict[str, list[dict]]:
    """Partition the complete canonical pilot queue once across measured host slots."""

    from common.semantic_hpo_candidates import semantic_initial_trials


    return _weighted_partition(semantic_initial_trials(profile), shard_weights)


def _candidate_seed_allocation(manifest: dict) -> dict[str, dict[str, list[int]]]:
    """Normalize legacy seed shards and authenticate all nine candidate-seed pairs."""

    ids = [str(candidate["global_candidate_id"]) for candidate in manifest["candidates"]]
    # Explicit per-candidate routing supports more hosts than confirmation seeds.
    if "seeds_by_candidate_by_shard" in manifest:
        allocation = deepcopy(manifest["seeds_by_candidate_by_shard"])
    # Older two-host manifests assign the same seed subset to every finalist.
    else:
        allocation = {
            shard: {identifier: list(seeds) for identifier in ids}
            for shard, seeds in manifest["seeds_by_shard"].items()
        }
    expected_shards = set(manifest["exports_sha256"])
    # Every shard and candidate must remain represented even when a local list is empty.
    if set(allocation) != expected_shards or any(
        not isinstance(values, dict) or set(values) != set(ids) for values in allocation.values()
    ):
        raise ValueError("Confirmation allocation differs from the complete shard and candidate sets.")
    for identifier in ids:
        rows = [allocation[shard][identifier] for shard in allocation]
        # Exact seed coverage precludes duplicated or omitted learned/control pairs.
        if any(not isinstance(values, list) for values in rows):
            raise ValueError("Confirmation allocation must contain per-candidate seed lists.")
        seeds = [seed for values in rows for seed in values]
        # Every global finalist retains precisely the three prescribed independent seeds.
        if any(type(seed) is not int for seed in seeds) or sorted(seeds) != [101, 202, 303]:
            raise ValueError("Allocate confirmation seeds 101, 202 and 303 exactly once for every candidate.")
    return allocation


def _local_seed_assignment(manifest: dict, shard_id: str) -> dict[str, list[int]]:
    """Read one authenticated host's explicit candidate-seed assignment."""

    return _candidate_seed_allocation(manifest)[shard_id]


def _storage_identity(plan: dict) -> dict:
    """Require a prepared node-local SQLite cache owned by this runtime host."""

    root = Path(plan["study_root"])
    marker = shared._read(root / "sqlite_local.json")
    path = database_path(root)
    # A worker cannot use a cache owned by another host or inside shared study storage.
    if marker.get("hostname") != plan["identity"]["hostname"] \
    or marker.get("database_path") != str(path) or path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Semantic shards require separately prepared host-local SQLite caches.")
    return {
        "hostname": marker["hostname"], "database_path": str(path), 
        "cache_id": marker["cache_id"], "study_root": str(root.resolve())
    }


def start_shared_campaign(
    plan: dict, campaign_root: str | Path, shard_id: str, started_at_unix: float | None = None, 
    anchors_by_shard: dict[str, list[dict]] | None = None
) -> dict:
    """Adopt one immutable twelve-hour clock after all hosts finish readiness.

    With separate volumes, transport campaign_clock.json to each other host
    before calling this function there. A shared mounted directory also works.
    Repeated START calls cannot move an existing deadline or rename a shard.
    """

    # Shard labels identify outputs without introducing path traversal.
    if not isinstance(shard_id, str) or not shard_id or Path(shard_id).name != shard_id:
        raise ValueError("Use one nonempty filename-safe shard identifier.")
    policy = plan.get("time_budget")
    # This campaign has one user-specified finite twelve-hour budget.
    if policy is None or policy.get("experiment_hours") != 12.0:
        raise ValueError("The semantic shared campaign requires its twelve-hour policy.")
    storage = _storage_identity(plan)
    root = Path(campaign_root)
    root.mkdir(parents=True, exist_ok=True)
    with study_lock(root, blocking=True):
        clock_path = root / "campaign_clock.json"
        # Restarting either host adopts the existing absolute clock.
        if clock_path.exists():
            clock = _read(clock_path)
            allocation = clock["recipe"].get("anchors_by_shard") if anchors_by_shard is None else anchors_by_shard
            recipe = _recipe(plan, allocation)
            # Runtime/source and anchor identities must agree before local adoption.
            if clock.get("kind") != "semantic_campaign_clock" or clock["recipe"] != recipe \
            or clock["budget"]["policy"] != policy:
                raise ValueError("Shared semantic recipe, source identity or time policy changed.")
            # A repeated START cannot extend the experiment.
            if started_at_unix is not None and clock["budget"]["started_at_unix"] != float(started_at_unix):
                raise ValueError("Shared campaign START time changed.")
        # Only the first valid campaign publication allocates a start time.
        else:
            allocation = anchors_by_shard
            recipe = _recipe(plan, allocation)
            started = time.time() if started_at_unix is None else float(started_at_unix)
            # Invalid timestamps cannot produce enforceable absolute deadlines.
            if not math.isfinite(started):
                raise ValueError("A shared START requires a finite absolute timestamp.")
            deadline = started + 3600.0 * policy["experiment_hours"]
            clock = _seal({
                "version": 1, "kind": "semantic_campaign_clock", "recipe": recipe, 
                "budget": {
                    "version": 1, "policy": deepcopy(policy), "started_at_unix": started, 
                    "deadline_unix": deadline, 
                    "search_deadline_unix": deadline - 3600.0 * policy["confirmation_reserve_hours"], 
                    "cleanup_seconds": 60.0
                }
            })
        # The local sealed queue must match the declared assignment before publication.
        if allocation is not None and plan["hpo"].get("initial_trials") != allocation.get(shard_id):
            raise ValueError("This host's sealed initial_trials differ from its declared shard anchor queue.")
        _publish_once(clock_path, clock)
        with shared._coordinator(plan):
            _publish_once(Path(plan["control_root"]) / "budget.json", clock["budget"])
            _publish_once(Path(plan["control_root"]) / "cluster_campaign.json", {
                "shard_id": shard_id, "clock_sha256": clock["sha256"], 
                "campaign_root": str(root.resolve()), "storage": storage, "anchors_by_shard": allocation
            })
    return clock


def _validate_candidate(candidate: dict, recipe: dict, shard_id: str) -> Config:
    """Authenticate portable native fields, sampled settings and objective evidence."""

    score = candidate.get("search_objective")
    text = candidate.get("config_text")
    # Envelope checks supplement finite objective and exact YAML identity checks.
    if candidate.get("source_shard") != shard_id or type(candidate.get("source_trial_number")) is not int \
    or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) \
    or not isinstance(text, str) or hashlib.sha256(text.encode("utf-8")).hexdigest() != candidate.get("config_sha256"):
        raise ValueError("Invalid semantic candidate objective, shard or configuration identity.")
    config = Config(**_safe_load_unique_yaml(io.StringIO(text)))
    profile = recipe["hpo"]["semantic_profile"]
    validate_semantic_config(config, profile)
    metric = profile.get("selection_metric", semantic.METRIC)
    # A different sampler stream must never change model/data seed or search domains.
    if config.hpo.get("params") != candidate.get("params") \
    or config.hpo.get("trial_number") != candidate["source_trial_number"] \
    or config.hpo.get("semantic_search_space_overrides") != recipe["hpo"].get("search_space_overrides", {}) \
    or config.training.seed != profile["student_config"]["training"]["seed"] \
    or candidate.get("objective_metric") != metric:
        raise ValueError("Semantic candidate differs from its sampled parameters or common training seed.")
    return config


def _validate_export_closure(record: dict) -> None:
    """Bind a portable export to all finite complete trials at admission close."""

    profile = record["recipe"]["hpo"]["semantic_profile"]
    closed = record["search_closed"]
    expected = {
        "fixed_recipe_sha256": profile["fixed_recipe_sha256"], 
        "objective_metric": profile.get("selection_metric", semantic.METRIC), 
        "task_seed": profile["task_seed"], "dataset_seed": profile["dataset_seed"], 
        "task_groups": profile["task_groups"]
    }
    snapshot = closed.get("trial_state_fingerprint", [])
    # Closed study fingerprints cannot contain live or duplicate allocations.
    if any(closed.get(key) != value for key, value in expected.items()) \
    or closed.get("allocated_trials") != len(snapshot) \
    or any(row["state"] == "RUNNING" for row in snapshot) \
    or len({row["number"] for row in snapshot}) != len(snapshot):
        raise ValueError("Exported search closure has a different protocol or active/duplicated trials.")
    waiting = sorted(row["number"] for row in snapshot if row["state"] == "WAITING")
    # Unstarted hints are recorded explicitly and excluded consistently.
    if waiting != closed.get("unstarted_trial_numbers") or waiting != sorted(record["unstarted_trial_numbers"]):
        raise ValueError("Exported unstarted hints differ from the closed search evidence.")
    complete = {
        row["number"]: row for row in snapshot if row["state"] == "COMPLETE" and row["value"] is not None
    }
    # Global ranking receives every finite completed result from each stopped shard.
    if set(complete) != {candidate["source_trial_number"] for candidate in record["candidates"]}:
        raise ValueError("Search export must retain every finite complete candidate in its closed snapshot.")
    for candidate in record["candidates"]:
        row = complete[candidate["source_trial_number"]]
        # Portable scores and parameters must still match the closed local observation.
        if row["params"] != candidate["params"] or row["value"] != candidate["search_objective"]:
            raise ValueError("Exported candidate differs from its closed study observation.")


def export_search(plan: dict, shard_id: str, output_path: str | Path) -> dict:
    """Export complete candidates only after the admitted search has explicitly closed."""

    semantic.close_search(plan, reason="global_finalist_export")
    with shared._coordinator(plan):
        campaign = shared._read(Path(plan["control_root"]) / "cluster_campaign.json")
        recipe = _recipe(plan, campaign.get("anchors_by_shard"))
        # Host identity is provenance, not merely an export filename.
        if campaign["shard_id"] != shard_id or campaign["storage"] != _storage_identity(plan):
            raise ValueError("Search export has a different shard or local-storage identity.")
        closed = semantic.validate_search_closed(plan)
        # Export cannot silently freeze a study that still admits new work.
        if closed is None:
            raise ValueError("An explicit authenticated search_closed marker is required before export.")
        study = shared._load_study(plan)
        trials = [] if study is None else study.get_trials(deepcopy=False)
        # An in-flight candidate could change the global ranking.
        if any(trial.state.name == "RUNNING" for trial in trials):
            raise ValueError("Finish or recover all running search trials before export.")
        waiting = [trial.number for trial in trials if trial.state.name == "WAITING"]
        candidates = []
        for trial in shared._ranked(study, direction="maximize"):
            path = Path(plan["study_root"]) / "configs" / f"trial-{trial.number:04d}.yaml"
            data = path.read_bytes()
            candidate = {
                "source_shard": shard_id, "source_trial_number": trial.number, 
                "params": deepcopy(trial.params), "objective_metric": semantic._metric(plan), 
                "search_objective": float(trial.value), "config_text": data.decode("utf-8"), 
                "config_sha256": hashlib.sha256(data).hexdigest()
            }
            _validate_candidate(candidate, recipe, shard_id)
            candidates.append(candidate)
        record = _seal({
            "version": 1, "kind": "semantic_search_shard", "shard_id": shard_id, 
            "sampler_seed": plan["hpo"].get("sampler_seed", plan["hpo"]["seed"]), 
            "recipe": recipe, "storage": campaign["storage"], "clock_sha256": campaign["clock_sha256"], 
            "budget": shared._budget_state(plan), "search_closed": closed, 
            "summary": semantic.search_summary(plan), "unstarted_trial_numbers": waiting, "candidates": candidates
        })
        _validate_export_closure(record)
        return _publish_once(output_path, record)


def freeze_global_finalists(
    exports: list[dict | str | Path], output_path: str | Path, top_k: int = 3, 
    seeds_by_shard: dict[str, list[int]] | None = None, shard_weights: dict[str, int] | None = None
) -> dict:
    """Freeze globally ranked distinct recipes and three paired confirmation seeds."""

    records = [_read(value) for value in exports]
    # Legacy whole-seed partitions and weighted candidate-seed partitions are exclusive.
    if seeds_by_shard is not None and shard_weights is not None:
        raise ValueError("Choose legacy seed routing or weighted candidate-seed routing, not both.")
    allocation = {"a": [101, 202], "b": [303]} if seeds_by_shard is None else deepcopy(seeds_by_shard)
    shards = [record["shard_id"] for record in records]
    planned_shards = set(shard_weights) if shard_weights is not None else set(allocation)
    seeds = [101, 202, 303] if shard_weights is not None else [seed for values in allocation.values() for seed in values]
    # Global selection cannot proceed from one host or duplicate exports.
    if len(records) < 2 or len(set(shards)) != len(records) or set(shards) != planned_shards:
        raise ValueError("Global semantic selection requires all distinct planned shard exports.")
    # Every finalist receives the same three distinct paired repeats exactly once.
    if (shard_weights is None and any(not values for values in allocation.values())) or any(type(seed) is not int for seed in seeds) \
    or sorted(seeds) != [101, 202, 303]:
        raise ValueError("Allocate confirmation seeds 101, 202 and 303 exactly once across shards.")
    reference = records[0]
    expected_anchors = reference["recipe"].get("anchors_by_shard")
    # Five-host selection cannot omit a shard that received part of the sealed pilot.
    if expected_anchors is not None and set(expected_anchors) != planned_shards:
        raise ValueError("Global selection must include every shard in the sealed anchor partition.")
    candidates = []
    storages = set()
    for record in records:
        # Only complete studies with matching source/runtime/protocol clocks are comparable.
        if record.get("kind") != "semantic_search_shard" or any(
            record[key] != reference[key] for key in ("recipe", "clock_sha256", "budget")
        ) or not isinstance(record.get("search_closed"), dict):
            raise ValueError("Semantic shard recipe, runtime identity, clock or closed-search evidence differs.")
        _validate_export_closure(record)
        storage = record["storage"]
        storage_id = (storage["hostname"], storage["database_path"], storage["cache_id"])
        # Searches must use independent node-local transactions.
        if storage_id in storages:
            raise ValueError("Semantic exports cannot share one SQLite cache.")
        storages.add(storage_id)
        numbers = set()
        for candidate in record["candidates"]:
            _validate_candidate(candidate, record["recipe"], record["shard_id"])
            # A repeated trial cannot consume a second ranking position.
            if candidate["source_trial_number"] in numbers:
                raise ValueError("A shard export duplicated one trial number.")
            numbers.add(candidate["source_trial_number"])
            candidates.append(candidate)
    training_seed = reference["recipe"]["hpo"]["semantic_profile"]["student_config"]["training"]["seed"]
    # Confirmation is new model initialization, not a replay of the search seed.
    if training_seed in seeds:
        raise ValueError("Confirmation seeds must differ from the common search training seed.")
    candidates.sort(key=lambda value: (-value["search_objective"], value["source_shard"], value["source_trial_number"]))
    distinct = []
    identities = set()
    for candidate in candidates:
        identity = _hash(candidate["params"])
        # The better of duplicated parameter observations owns the sole finalist slot.
        if identity not in identities:
            identities.add(identity)
            distinct.append({**candidate, "global_candidate_id": len(distinct) + 1})
    # Insufficient completed diversity remains a visible campaign failure.
    if type(top_k) is not int or top_k < 1 or len(distinct) < top_k:
        raise ValueError("Not enough distinct finite completed semantic candidates for global selection.")
    routing = {"seeds_by_shard": allocation}
    # Weighted bundles support more shards than the three prescribed seeds.
    if shard_weights is not None:
        jobs = [
            {"global_candidate_id": candidate["global_candidate_id"], "seed": seed}
            for candidate in distinct[:top_k] for seed in seeds
        ]
        partition = _weighted_partition(jobs, shard_weights)
        routing = {
            "shard_weights": deepcopy(shard_weights), 
            "seeds_by_candidate_by_shard": {
                shard: {
                    str(candidate["global_candidate_id"]): [
                        job["seed"] for job in assigned if job["global_candidate_id"] == candidate["global_candidate_id"]
                    ] for candidate in distinct[:top_k]
                } for shard, assigned in partition.items()
            }
        }
    manifest = _seal({
        "version": 1, "kind": "semantic_global_finalists", "recipe": reference["recipe"], 
        "clock_sha256": reference["clock_sha256"], "budget": reference["budget"], 
        "top_k": top_k, **routing, 
        "exports_sha256": {record["shard_id"]: record["sha256"] for record in records}, 
        "sampler_seeds": {record["shard_id"]: record["sampler_seed"] for record in records}, 
        "candidates": distinct[:top_k]
    })
    _candidate_seed_allocation(manifest)
    with study_lock(Path(output_path).parent, blocking=True):
        return _publish_once(output_path, manifest)


def install_global_finalists(plan: dict, manifest: dict | str | Path, shard_id: str) -> dict:
    """Install globally unique output IDs while retaining original YAML trial identity."""

    global_manifest = _read(manifest)
    seed_assignment = _local_seed_assignment(global_manifest, shard_id)
    campaign = shared._read(Path(plan["control_root"]) / "cluster_campaign.json")
    # Local routing may differ; science, source identity and absolute clock may not.
    if global_manifest.get("kind") != "semantic_global_finalists" \
    or global_manifest["recipe"] != _recipe(plan, global_manifest["recipe"].get("anchors_by_shard")) \
    or global_manifest["clock_sha256"] != campaign["clock_sha256"] \
    or global_manifest["budget"] != shared._budget_state(plan) or campaign["shard_id"] != shard_id \
    or global_manifest["sampler_seeds"].get(shard_id) != plan["hpo"].get("sampler_seed", plan["hpo"]["seed"]):
        raise ValueError("Global semantic finalists do not match this shard's sealed campaign.")
    with shared._coordinator(plan):
        # Installing finalists must stop further local search allocations.
        if semantic.validate_search_closed(plan) is None:
            raise ValueError("Close the local search before installing global finalists.")
        candidates = []
        expected_ids = list(range(1, global_manifest["top_k"] + 1))
        # Unique consecutive IDs prevent confirmation-output collisions across source studies.
        if [candidate.get("global_candidate_id") for candidate in global_manifest["candidates"]] != expected_ids:
            raise ValueError("Global candidate IDs must uniquely enumerate every frozen finalist.")
        for candidate in global_manifest["candidates"]:
            _validate_candidate(candidate, global_manifest["recipe"], candidate["source_shard"])
            number = candidate["global_candidate_id"]
            path = Path(plan["control_root"]) / "finalist_configs" / f"global-{number:04d}.yaml"
            data = candidate["config_text"].encode("utf-8")
            path.parent.mkdir(parents=True, exist_ok=True)
            # Recovery cannot accept a replaced frozen configuration.
            if path.exists() and path.read_bytes() != data:
                raise ValueError("An installed semantic finalist configuration changed.")
            # Previously absent configurations are published atomically once.
            if not path.exists():
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(data)
                temporary.replace(path)
            candidates.append({
                "rank": number, "trial_number": number, "global_candidate_id": number, 
                "source_shard": candidate["source_shard"], "source_trial_number": candidate["source_trial_number"], 
                "params": candidate["params"], "objective_metric": candidate["objective_metric"], 
                "search_objective": candidate["search_objective"], 
                "source_input_config_path": str(path.resolve()), 
                "source_input_config_sha256": candidate["config_sha256"], 
                "input_config_path": str(path.resolve()), "config_sha256": candidate["config_sha256"]
            })
            # An empty local list means another shard owns this candidate's repeats.
            if "seeds_by_candidate_by_shard" in global_manifest:
                candidates[-1]["seeds"] = seed_assignment[str(number)]
        profile = plan["hpo"]["semantic_profile"]
        local = {
            "study_name": plan["study_name"], "global_manifest_sha256": global_manifest["sha256"], 
            "shard_id": shard_id, "top_k": global_manifest["top_k"], 
            "seeds": sorted({seed for values in seed_assignment.values() for seed in values}), 
            "search_profile": semantic.PROFILE, "fixed_recipe_sha256": profile["fixed_recipe_sha256"], 
            "dataset_seed": profile["dataset_seed"], "task_seed": profile["task_seed"], 
            "task_groups": deepcopy(profile["task_groups"]), 
            "selection_policy": {
                "objective_metric": semantic._metric(plan), "objective_direction": "maximize", 
                "objective_network": "raw", "evaluation_source": "held_out_training_validation"
            }, 
            "candidates": candidates
        }
        # Persist the portable seal so launch-time checks authenticate candidate routing.
        if "seeds_by_candidate_by_shard" in global_manifest:
            frozen_path = Path(plan["control_root"]) / "global_finalists.json"
            _publish_once(frozen_path, global_manifest)
            local["global_manifest_path"] = str(frozen_path.resolve())
        return _publish_once(Path(plan["control_root"]) / "finalists.json", local)


def validate_installed_finalists(plan: dict, local: dict) -> None:
    """Authenticate local candidate routing against the frozen portable global manifest."""

    path = Path(plan["control_root"]) / "global_finalists.json"
    # A routing manifest must be installed beside this shard's immutable finalist receipt.
    if local.get("global_manifest_path") != str(path.resolve()):
        raise ValueError("Candidate seed routing is missing its installed global manifest.")
    frozen = _read(path)
    assigned = _local_seed_assignment(frozen, local["shard_id"])
    # Local scheduling cannot alter the globally selected scientific recipe or seed union.
    if local.get("global_manifest_sha256") != frozen["sha256"] \
    or frozen["recipe"] != _recipe(plan, frozen["recipe"].get("anchors_by_shard")) \
    or len(local["candidates"]) != len(frozen["candidates"]) \
    or local["seeds"] != sorted({seed for values in assigned.values() for seed in values}):
        raise ValueError("Installed finalist routing differs from the frozen global recipe.")
    for candidate, selected in zip(local["candidates"], frozen["candidates"]):
        identifier = selected["global_candidate_id"]
        expected = {
            "trial_number": identifier, "global_candidate_id": identifier, 
            "source_shard": selected["source_shard"], "source_trial_number": selected["source_trial_number"], 
            "config_sha256": selected["config_sha256"], "params": selected["params"], 
            "seeds": assigned[str(identifier)]
        }
        # The source trial and exact local seed subset belong to this global output ID.
        if any(candidate.get(key) != value for key, value in expected.items()):
            raise ValueError("Installed candidate seeds or source identity differ from the global allocation.")


def merge_confirmations(
    plans_by_shard: dict[str, dict], manifest: dict | str | Path, output_path: str | Path
) -> dict:
    """Merge authenticated paired repeats without assigning scores to missing runs.

    Receipt/config trees may be transported to one host for this read-and-report
    phase, retaining the plan's referenced paths and original source identities.
    """

    from common.semantic_hpo_cluster_controls import merge_controls


    global_manifest = _read(manifest)
    allocation = _candidate_seed_allocation(global_manifest)
    # A missing host cannot be silently treated as an entirely unstarted set of seeds.
    if set(plans_by_shard) != set(allocation):
        raise ValueError("Global reporting requires a plan for each expected semantic shard.")
    scores = {candidate["global_candidate_id"]: {} for candidate in global_manifest["candidates"]}
    for shard_id, plan in plans_by_shard.items():
        manifest_path = Path(plan["control_root"]) / "finalists.json"
        local = shared._read(manifest_path)
        profile = plan["hpo"]["semantic_profile"]
        assigned = allocation[shard_id]
        assigned_union = sorted({seed for values in assigned.values() for seed in values})
        # Reporting reads transported receipts without opening another host's SQLite.
        if global_manifest.get("kind") != "semantic_global_finalists" \
        or _recipe(plan, global_manifest["recipe"].get("anchors_by_shard")) != global_manifest["recipe"] \
        or local.get("global_manifest_sha256") != global_manifest["sha256"] \
        or local.get("shard_id") != shard_id or local.get("seeds") != assigned_union \
        or local.get("fixed_recipe_sha256") != profile["fixed_recipe_sha256"] \
        or local.get("task_groups") != profile["task_groups"] \
        or local.get("dataset_seed") != profile["dataset_seed"] or local.get("task_seed") != profile["task_seed"]:
            raise ValueError("Installed finalist manifest differs from its global semantic campaign.")
        # Partial results may miss receipts, never frozen candidate identities.
        if len(local["candidates"]) != len(global_manifest["candidates"]):
            raise ValueError("Installed finalist count differs from the global manifest.")
        for candidate, frozen in zip(local["candidates"], global_manifest["candidates"]):
            expected = {
                "trial_number": frozen["global_candidate_id"], "global_candidate_id": frozen["global_candidate_id"], 
                "source_shard": frozen["source_shard"], "source_trial_number": frozen["source_trial_number"], 
                "params": frozen["params"], "config_sha256": frozen["config_sha256"], 
                "objective_metric": frozen["objective_metric"], "search_objective": frozen["search_objective"]
            }
            # Transported YAML bytes and local source routing must retain global identity.
            if any(candidate.get(key) != value for key, value in expected.items()) \
            or shared._digest(candidate["input_config_path"]) != frozen["config_sha256"] \
            or candidate.get("seeds", local["seeds"]) != assigned[str(frozen["global_candidate_id"])]:
                raise ValueError("Installed candidate configuration differs from global selection.")
        for candidate in local["candidates"]:
            for seed in candidate.get("seeds", local["seeds"]):
                path = Path(plan["control_root"]) / "confirmations" / f"trial-{candidate['trial_number']:04d}" / f"seed-{seed}" / "completed.json"
                # Missing repeats stay incomplete instead of receiving fabricated zero scores.
                if not path.exists():
                    continue
                expected = {
                    "manifest_sha256": shared._digest(manifest_path), "config_sha256": candidate["config_sha256"], 
                    "trial_number": candidate["trial_number"], "training_seed": seed
                }
                result = shared._completed_record(path, expected, plan=plan)["result"]
                # A receipt must belong to this selected native recipe and paired seed.
                if result.get("source_input_config_sha256") != candidate["config_sha256"] \
                or result.get("source_trial_number") != candidate["source_trial_number"] \
                or result.get("training_seed") != seed or result.get("dataset_seed") != local["dataset_seed"] \
                or result.get("task_seed") != local["task_seed"]:
                    raise ValueError("Confirmation differs from its global source or paired seed identity.")
                scores[candidate["global_candidate_id"]][seed] = float(result["objective"])
    metric = global_manifest["recipe"]["hpo"]["semantic_profile"].get("selection_metric", semantic.METRIC)
    rows = []
    for candidate in global_manifest["candidates"]:
        observed = scores[candidate["global_candidate_id"]]
        values = list(observed.values())
        rows.append({
            "global_candidate_id": candidate["global_candidate_id"], "source_shard": candidate["source_shard"], 
            "source_trial_number": candidate["source_trial_number"], "objective_metric": metric, 
            "search_objective": candidate["search_objective"], 
            "scores_by_seed": {str(seed): score for seed, score in sorted(observed.items())}, 
            "completed_seeds": len(values), "required_seeds": 3, 
            "mean_objective": statistics.mean(values) if values else None, 
            "std_objective": statistics.stdev(values) if len(values) > 1 else None, 
            "all_seeds_complete": len(values) == 3, "comparable": len(values) == 3
        })
    result = {
        "version": 1, "global_manifest_sha256": global_manifest["sha256"], 
        "evaluation_source": "held_out_training_validation", "objective_metric": metric, 
        "all_pairs_complete": all(row["all_seeds_complete"] for row in rows), "candidates": rows
    }
    result["controls"] = merge_controls(plans_by_shard, global_manifest, result)
    result["all_campaign_pairs_complete"] = result["all_pairs_complete"] and result["controls"]["all_pairs_complete"]
    result = _seal(result)
    with study_lock(Path(output_path).parent, blocking=True):
        shared._write(output_path, result)
    return result
