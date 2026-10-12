"""Authenticate and aggregate matched controls across semantic campaign shards."""

from copy import deepcopy
from pathlib import Path
import statistics

from common import dit_hpo_runner as shared
from common import semantic_hpo_runner as semantic


def merge_controls(
    plans_by_shard: dict[str, dict], manifest: dict | str | Path, learned_summary: dict
) -> dict:
    """Read matched control receipts without opening foreign SQLite databases.

    Every paired difference is checked against the corresponding learned receipt,
    even when its score was supplied by the authenticated confirmation merger.
    Missing receipts remain missing; this function does not publish artifacts.
    """

    from common import semantic_hpo_cluster as cluster


    frozen = cluster._read(manifest)
    allocation = cluster._candidate_seed_allocation(frozen)
    metric = frozen["recipe"]["hpo"]["semantic_profile"].get("selection_metric", semantic.METRIC)
    # Both the seed allocation and the learned report must name this exact campaign.
    if frozen.get("kind") != "semantic_global_finalists" \
    or set(plans_by_shard) != set(allocation) \
    or learned_summary.get("global_manifest_sha256") != frozen["sha256"] \
    or learned_summary.get("objective_metric") != metric:
        raise ValueError("Control merge requires the matching global campaign and learned summary.")
    learned_rows = {row["global_candidate_id"]: row for row in learned_summary["candidates"]}
    candidate_ids = {row["global_candidate_id"] for row in frozen["candidates"]}
    # Duplicate or missing report rows cannot stand in for a frozen finalist.
    if set(learned_rows) != candidate_ids or len(learned_rows) != len(learned_summary["candidates"]):
        raise ValueError("Learned summary candidates differ from the global finalists.")
    observed = {identifier: {condition: [] for condition in (
        "baseline", "unmodulated_feature_distillation"
    )} for identifier in candidate_ids}
    for shard_id, plan in plans_by_shard.items():
        local_path = Path(plan["control_root"]) / "finalists.json"
        local = shared._read(local_path)
        profile = plan["hpo"]["semantic_profile"]
        assigned = allocation[shard_id]
        assigned_union = sorted({seed for values in assigned.values() for seed in values})
        # Transported control trees retain the global recipe, source and schedule.
        if cluster._recipe(plan, frozen["recipe"].get("anchors_by_shard")) != frozen["recipe"] \
        or local.get("global_manifest_sha256") != frozen["sha256"] or local.get("shard_id") != shard_id \
        or local.get("seeds") != assigned_union \
        or local.get("fixed_recipe_sha256") != profile["fixed_recipe_sha256"] \
        or local.get("task_groups") != profile["task_groups"] \
        or local.get("dataset_seed") != profile["dataset_seed"] or local.get("task_seed") != profile["task_seed"] \
        or len(local["candidates"]) != len(frozen["candidates"]):
            raise ValueError("Installed control manifest differs from the global campaign.")
        local_digest = shared._digest(local_path)
        for candidate, selected in zip(local["candidates"], frozen["candidates"]):
            identifier = selected["global_candidate_id"]
            expected_candidate = {
                "trial_number": identifier, "global_candidate_id": identifier, 
                "source_shard": selected["source_shard"], "source_trial_number": selected["source_trial_number"], 
                "params": selected["params"], "config_sha256": selected["config_sha256"], 
                "objective_metric": metric, "search_objective": selected["search_objective"]
            }
            # Output IDs, native source trials and actual YAML bytes stay bound together.
            if any(candidate.get(key) != value for key, value in expected_candidate.items()) \
            or shared._digest(candidate["input_config_path"]) != selected["config_sha256"] \
            or candidate.get("seeds", local["seeds"]) != assigned[str(identifier)]:
                raise ValueError("Control finalist configuration differs from global selection.")
            for seed in candidate.get("seeds", local["seeds"]):
                identity = {
                    "manifest_sha256": local_digest, "config_sha256": candidate["config_sha256"], 
                    "trial_number": identifier, "training_seed": seed
                }
                tail = Path(f"trial-{identifier:04d}") / f"seed-{seed}" / "completed.json"
                learned_path = Path(plan["control_root"]) / "confirmations" / tail
                learned_score = None
                reported_score = learned_rows[identifier]["scores_by_seed"].get(str(seed))
                # Pairing uses a completed learned receipt with the same source and seed.
                if learned_path.exists():
                    result = shared._completed_record(learned_path, identity, plan=plan)["result"]
                    _validate_source(result, candidate, seed)
                    learned_score = float(result["objective"])
                    # A summary cannot replace or alter the authenticated per-seed outcome.
                    if reported_score != learned_score:
                        raise ValueError("Learned summary score differs from its paired completion receipt.")
                # A report claiming a missing learned receipt is not portable evidence.
                elif reported_score is not None:
                    raise ValueError("Learned summary includes a missing paired completion receipt.")
                for condition in observed[identifier]:
                    path = Path(plan["control_root"]) / "controls" / condition / tail
                    # Interrupted or unstarted controls receive no fabricated outcome.
                    if not path.exists():
                        continue
                    expected = {**identity, "control_condition": condition}
                    result = shared._completed_record(path, expected)["result"]
                    semantic._validate_control_result(plan, result, expected)
                    _validate_source(result, candidate, seed)
                    score = float(result["objective"])
                    observed[identifier][condition].append({
                        "source_shard": shard_id, "training_seed": seed, "objective": score, 
                        "learned_minus_control": None if learned_score is None else learned_score - score, 
                        "validation_continual_metrics": deepcopy(result.get("validation_continual_metrics", {}))
                    })
    required_seeds = 3
    rows = []
    for candidate in frozen["candidates"]:
        for condition, per_seed in observed[candidate["global_candidate_id"]].items():
            per_seed.sort(key=lambda row: row["training_seed"])
            values = [row["objective"] for row in per_seed]
            differences = [row["learned_minus_control"] for row in per_seed if row["learned_minus_control"] is not None]
            rows.append({
                "global_candidate_id": candidate["global_candidate_id"], "control_condition": condition, 
                "objective_metric": metric, "source_shard": candidate["source_shard"], 
                "source_trial_number": candidate["source_trial_number"], 
                "completed_seeds": len(values), "required_seeds": required_seeds, "paired_seeds": len(differences), 
                "mean_objective": statistics.mean(values) if values else None, 
                "std_objective": statistics.stdev(values) if len(values) > 1 else None, 
                "mean_learned_minus_control": statistics.mean(differences) if differences else None, 
                "std_learned_minus_control": statistics.stdev(differences) if len(differences) > 1 else None, 
                "per_seed": per_seed, "mean_validation_metrics": semantic._mean_validation_metrics(per_seed), 
                "all_seeds_complete": len(values) == required_seeds, "comparable": len(differences) == required_seeds
            })
    return {
        "global_manifest_sha256": frozen["sha256"], "objective_metric": metric, 
        "evaluation_source": "held_out_training_validation", 
        "completed_runs": sum(row["completed_seeds"] for row in rows), 
        "required_runs": len(rows) * required_seeds, "all_runs_complete": all(row["all_seeds_complete"] for row in rows), 
        "all_pairs_complete": all(row["comparable"] for row in rows), "candidates": rows
    }


def _validate_source(result: dict, candidate: dict, seed: int) -> None:
    """Require a completed stream to match the original native YAML and paired seed."""

    # Global output IDs must never replace the source trial number inside YAML.
    if result.get("source_input_config_sha256") != candidate["config_sha256"] \
    or result.get("source_trial_number") != candidate["source_trial_number"] or result.get("training_seed") != seed:
        raise ValueError("Control or learned receipt differs from its global source and paired seed.")
