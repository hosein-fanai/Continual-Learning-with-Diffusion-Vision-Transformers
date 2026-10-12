"""Semantic HPO lifecycle over a sealed catalogue of native continual recipes.

The coordinator uses the existing admitted worker, immutable recipe, search and
paired-confirmation lifecycle. Learning happens only in isolated remote workers.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from common import dit_hpo_runner as shared


PROFILE = "semantic_consolidation_runner"
METRIC = "final_average_accuracy"
make_plan = shared.make_plan
budget_summary = shared.budget_summary
start_experiment = shared.start_experiment


def _metric(plan: dict) -> str:
    """Read the objective sealed with the native semantic recipe catalogue."""

    return plan["hpo"]["semantic_profile"].get("selection_metric", METRIC)


def _validate_plan(plan: dict) -> None:
    """Require semantic tuning with the complete raw validation objective."""

    settings = plan["hpo"]
    profile = settings.get("semantic_profile")
    metric = profile.get("selection_metric", METRIC) if isinstance(profile, dict) else None
    dataset = profile.get("student_config", {}).get("dataset", {}) if isinstance(profile, dict) else {}
    # Generic generation or continual-tuning plans cannot be relabeled semantic.
    if (
        settings.get("search_profile") != PROFILE or settings.get("task") != "continual"
        or settings.get("model_name") != "dit_classifier"
        or metric not in {METRIC, "retention_balanced_accuracy"}
        or settings.get("objective_metrics") != [metric]
        or settings.get("objective_directions") != ["maximize"]
        or not isinstance(profile, dict)
        or dataset.get("validation_source") != "split"
        or dataset.get("validation_ratio", 0.0) <= 0.0
    ):
        raise ValueError("Semantic runner requires its sealed native profile and raw full-stream validation maximization.")


def search_summary(plan: dict) -> dict:
    """Report completed full-stream accuracy without interpreting pruned scores."""

    _validate_plan(plan)
    study = shared._load_study(plan)
    trials = [] if study is None else study.get_trials(deepcopy=False)
    ranked = shared._ranked(study, direction="maximize")
    states = {}
    for trial in trials:
        states[trial.state.name] = states.get(trial.state.name, 0) + 1
    earlier = sorted(ranked, key=lambda trial: trial.number)[:-50]
    best = float(ranked[0].value) if ranked else None
    summary = {
        "allocated_trials": len(trials), "completed_finite_trials": len(ranked), 
        "states": states, "best_trial": ranked[0].number if ranked else None, 
        "objective_metric": _metric(plan), "best_validation_objective": best, 
        "improvement_last_50_valid_trials": best - max(float(trial.value) for trial in earlier) if earlier else None, 
        "task_groups": deepcopy(plan["hpo"]["semantic_profile"]["task_groups"]), 
        "fixed_recipe_sha256": plan["hpo"]["semantic_profile"]["fixed_recipe_sha256"]
    }
    summary["best_validation_metrics"] = deepcopy(
        getattr(ranked[0], "user_attrs", {}).get("validation_metrics", {})
    ) if ranked else {}
    summary["candidate_coverage"] = {
        parameter: {
            candidate: {
                "allocated": sum(trial.params.get(parameter) == candidate for trial in trials), 
                "complete": sum(trial.params.get(parameter) == candidate for trial in ranked)
            } for candidate in plan["hpo"]["semantic_profile"].get(catalog, {})
        } for parameter, catalog in (("base_recipe", "recipe_candidates"), ("semantic_budget", "phase_budget_candidates"))
    }
    # Historical accuracy field names remain accurate for the original objective.
    if _metric(plan) == METRIC:
        summary["best_validation_final_average_accuracy"] = best
    return summary


def _trial_snapshot(trials: list) -> list[dict]:
    """Record allocation, state, objective and parameter identity at admission close."""

    return [{
        "number": trial.number, "state": trial.state.name, "params": dict(trial.params), 
        "value": float(trial.value) if trial.value is not None and math.isfinite(float(trial.value)) else None
    } for trial in sorted(trials, key=lambda item: item.number)]


def validate_search_closed(plan: dict) -> dict | None:
    """Authenticate a durable admission stop against its exact scientific recipe."""

    path = Path(plan["control_root"]) / "search_closed.json"
    # An absent marker leaves staged search admissions open.
    if not path.exists():
        return None
    marker = shared._read(path)
    profile = plan["hpo"]["semantic_profile"]
    expected = {
        "study_name": plan["study_name"], "fixed_recipe_sha256": profile["fixed_recipe_sha256"], 
        "objective_metric": _metric(plan), "task_seed": profile["task_seed"], 
        "dataset_seed": profile["dataset_seed"], "task_groups": profile["task_groups"]
    }
    # A closure from another task or split cannot authenticate this study.
    if any(marker.get(key) != value for key, value in expected.items()):
        raise ValueError("Search admission closure belongs to a different semantic recipe.")
    study = shared._load_study(plan)
    trials = [] if study is None else study.get_trials(deepcopy=False)
    # Closure freezes allocation and outcomes, including unstarted anchor records.
    if _trial_snapshot(trials) != marker["trial_state_fingerprint"] or len(trials) != marker["allocated_trials"] or sorted(
        trial.number for trial in trials if trial.state.name == "WAITING"
    ) != marker["unstarted_trial_numbers"]:
        raise ValueError("Trial allocation changed after semantic search admission closed.")
    return marker


def search_closed(plan: dict) -> bool:
    """Report whether an authenticated unchanged study has stopped new admissions."""

    return validate_search_closed(plan) is not None


def _close_search(plan: dict, reason: str) -> dict:
    """Persist the final admission boundary while holding the coordinator lock."""

    existing = validate_search_closed(plan)
    # Repeated closure calls retain the original boundary and timestamp.
    if existing is not None:
        return existing
    study = shared._load_study(plan)
    trials = [] if study is None else study.get_trials(deepcopy=False)
    # Live workers must finish or recover before immutable selection.
    if any(trial.state.name == "RUNNING" for trial in trials):
        raise ValueError("Finish or recover running trials before closing semantic search.")
    profile = plan["hpo"]["semantic_profile"]
    marker = {
        "study_name": plan["study_name"], "fixed_recipe_sha256": profile["fixed_recipe_sha256"], 
        "objective_metric": _metric(plan), "task_seed": profile["task_seed"], 
        "dataset_seed": profile["dataset_seed"], "task_groups": profile["task_groups"], 
        "closed_at_unix": time.time(), "allocated_trials": len(trials), 
        "trial_state_fingerprint": _trial_snapshot(trials), 
        "unstarted_trial_numbers": sorted(trial.number for trial in trials if trial.state.name == "WAITING"), 
        "reason": reason
    }
    shared._write(Path(plan["control_root"]) / "search_closed.json", marker)
    return marker


def close_search(plan: dict, reason: str = "selection_requested") -> dict:
    """Explicitly finish staged search before local or global finalist selection."""

    _validate_plan(plan)
    with shared._coordinator(plan):
        return _close_search(plan, reason)


def freeze_finalists(plan: dict, seeds: Iterable[int], top_k: int = 3) -> dict:
    """Freeze distinct full-stream winners under the sealed validation protocol."""

    _validate_plan(plan)
    profile = plan["hpo"]["semantic_profile"]
    metric = _metric(plan)
    seeds = list(seeds)
    # Paired repeats use distinct new model seeds rather than the search seed.
    if not seeds or any(type(value) is not int for value in seeds) \
    or len(set(seeds)) != len(seeds) or plan["hpo"]["seed"] in seeds:
        raise ValueError("Use distinct integer confirmation seeds different from the search seed.")
    # The requested number is an exact positive candidate count.
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer candidate count.")
    policy = {
        "objective_metric": metric, "objective_direction": "maximize", "objective_network": "raw", 
        "evaluation_source": "held_out_training_validation"
    }
    identity = {
        "study_name": plan["study_name"], "top_k": top_k, "seeds": seeds, 
        "search_profile": PROFILE, "fixed_recipe_sha256": profile["fixed_recipe_sha256"], 
        "dataset_seed": profile["dataset_seed"], "task_seed": profile["task_seed"], 
        "task_groups": deepcopy(profile["task_groups"]), "selection_policy": policy
    }
    with shared._coordinator(plan):
        path = Path(plan["control_root"]) / "finalists.json"
        # Resume authenticates the already frozen selection instead of reranking.
        if path.exists():
            manifest = shared._read(path)
            # Changing the scientific or repeat design requires a new campaign.
            if any(manifest.get(key) != value for key, value in identity.items()):
                raise ValueError("Frozen semantic finalists have a different recipe, selection or seed design.")
            # Per-candidate routing remains bound to the portable global selection seal.
            if "global_manifest_path" in manifest or any("seeds" in candidate for candidate in manifest["candidates"]):
                from common.semantic_hpo_cluster import validate_installed_finalists


                validate_installed_finalists(plan, manifest)
            for candidate in manifest["candidates"]:
                shared._candidate_seeds(manifest, candidate)
                # A stored path is insufficient if its frozen bytes were replaced.
                if shared._digest(candidate["input_config_path"]) != candidate["config_sha256"]:
                    raise ValueError("A frozen finalist configuration has changed.")
            return manifest
        study = shared._load_study(plan)
        ranked = shared._ranked(study, direction="maximize")
        distinct = []
        parameters = []
        for trial in ranked:
            # Repeated observations cannot occupy multiple finalist positions.
            if trial.params not in parameters:
                parameters.append(dict(trial.params))
                distinct.append(trial)
        # Incomplete diversity remains visible rather than silently shrinking top_k.
        if len(distinct) < top_k:
            raise ValueError("Not enough distinct finite completed configurations to select finalists.")
        trials = study.get_trials(deepcopy=False)
        # No running trial may be excluded from the immutable comparison.
        if any(trial.state.name == "RUNNING" for trial in trials):
            raise ValueError("Finish or recover running trials before freezing finalists.")
        waiting = [trial.number for trial in trials if trial.state.name == "WAITING"]
        deadline = shared._phase_deadline(plan, "search")
        # Calling the final selector explicitly closes admissions. Queued hints
        # remain unstarted evidence; no live worker can be excluded above.
        _close_search(plan, "finalists_requested")
        closed = search_closed(plan)
        candidates = []
        for rank, trial in enumerate(distinct[:top_k], start=1):
            original = Path(plan["study_root"]) / "configs" / f"trial-{trial.number:04d}.yaml"
            source_digest = shared._digest(original)
            frozen = Path(plan["control_root"]) / "finalist_configs" / original.name
            frozen.parent.mkdir(parents=True, exist_ok=True)
            frozen.write_bytes(original.read_bytes())
            # Copying must preserve identity even if the source changes concurrently.
            if shared._digest(frozen) != source_digest or shared._digest(original) != source_digest:
                raise ValueError("A finalist source configuration changed while being frozen.")
            candidate = {
                "rank": rank, "trial_number": trial.number, "params": dict(trial.params), 
                "objective_metric": metric, "search_objective": float(trial.value), 
                "source_input_config_path": str(original), "source_input_config_sha256": source_digest, 
                "input_config_path": str(frozen), "config_sha256": source_digest
            }
            # Legacy field names remain limited to their actual accuracy metric.
            if metric == METRIC:
                candidate["search_final_average_accuracy"] = float(trial.value)
            candidates.append(candidate)
        manifest = {**identity, "study_allocated_trials": len(trials), "candidates": candidates}
        # Queued hints remain recorded as unstarted rather than completed trials.
        if waiting:
            manifest["unstarted_trials_at_search_close"] = {
                "trial_numbers": waiting, "admission_closed": closed, 
                "search_execution_deadline_unix": None if deadline is None else deadline - 60.0
            }
        shared._write(path, manifest)
        return manifest


def validate_confirmation_objective(plan: dict, result: dict) -> None:
    """Bind finite confirmation scores to the same native stream and semantic route."""

    _validate_plan(plan)
    # Mechanistic controls never count as learned treatment confirmations.
    if result.get("control_condition") is not None:
        raise ValueError("A control treatment cannot count as a learned semantic confirmation.")
    profile = plan["hpo"]["semantic_profile"]
    expected = {
        "objective_metric": _metric(plan), "objective_direction": "maximize", "objective_network": "raw", 
        "task_groups": profile["task_groups"], "search_profile": PROFILE, 
        "fixed_recipe_sha256": profile["fixed_recipe_sha256"]
    }
    objective = result.get("objective")
    # Completion must not accept a plain continual run or a different frozen stream.
    if any(result.get(key) != value for key, value in expected.items()) or isinstance(objective, bool) \
    or not isinstance(objective, (int, float)) or not math.isfinite(float(objective)):
        raise ValueError("Confirmation does not match the finite semantic objective and frozen task stream.")
    for name in ("dataset_seed", "task_seed"):
        # New retention receipts bind both seeds; older accuracy receipts remain readable.
        if name in result or _metric(plan) != METRIC:
            # Recorded schedule and split seeds must match the sealed profile.
            if result.get(name) != profile[name]:
                raise ValueError("Confirmation does not match the frozen task stream seed identity.")
    metrics = result.get("validation_continual_metrics")
    # An objective cannot disagree with its own validation metric receipt.
    if metrics is not None and metrics.get(_metric(plan)) != objective:
        raise ValueError("Confirmation objective differs from its recorded validation metric.")


def confirmation_summary(plan: dict) -> list[dict]:
    """Summarize authenticated paired accuracy means and incomplete repeat counts."""

    _validate_plan(plan)
    manifest_path = Path(plan["control_root"]) / "finalists.json"
    manifest = shared._read(manifest_path)
    freeze_finalists(plan, manifest["seeds"], top_k=manifest["top_k"])
    manifest_digest = shared._digest(manifest_path)
    rows = []
    for candidate in manifest["candidates"]:
        scores = []
        per_seed = []
        candidate_seeds = shared._candidate_seeds(manifest, candidate)
        for training_seed in candidate_seeds:
            path = Path(plan["control_root"]) / "confirmations" / f"trial-{candidate['trial_number']:04d}" / f"seed-{training_seed}" / "completed.json"
            # Partial reports never invent outcomes for missing seeds.
            if path.exists():
                expected = {
                    "manifest_sha256": manifest_digest, "config_sha256": candidate["config_sha256"], 
                    "trial_number": candidate["trial_number"], "training_seed": training_seed
                }
                result = shared._completed_record(path, expected, plan=plan)["result"]
                scores.append(float(result["objective"]))
                per_seed.append({
                    "training_seed": training_seed, "objective": float(result["objective"]), 
                    "validation_continual_metrics": deepcopy(result.get("validation_continual_metrics", {}))
                })
        row = {
            "trial_number": candidate["trial_number"], 
            "objective_metric": _metric(plan), 
            "search_objective": candidate.get("search_objective", candidate.get("search_final_average_accuracy")), 
            "completed_seeds": len(scores), "required_seeds": len(candidate_seeds), 
            "mean_objective": statistics.mean(scores) if scores else None, 
            "std_objective": statistics.stdev(scores) if len(scores) > 1 else None, 
            "per_seed": per_seed, "mean_validation_metrics": _mean_validation_metrics(per_seed), 
            "all_seeds_complete": len(scores) == len(candidate_seeds), 
            "comparable": bool(candidate_seeds) and len(scores) == len(candidate_seeds)
        }
        # Preserve historical names only for the original average-accuracy objective.
        if _metric(plan) == METRIC:
            row.update({
                "search_final_average_accuracy": row["search_objective"], 
                "mean_final_average_accuracy": row["mean_objective"], 
                "std_final_average_accuracy": row["std_objective"]
            })
        rows.append(row)
    shared._write(Path(plan["control_root"]) / "confirmation_summary.json", rows)
    return rows


def _mean_validation_metrics(per_seed: list[dict]) -> dict:
    """Average only finite metrics present for every completed seed in a summary."""

    names = set.intersection(*(set(row["validation_continual_metrics"]) for row in per_seed)) if per_seed else set()
    means = {}
    for name in sorted(names):
        values = [row["validation_continual_metrics"][name] for row in per_seed]
        # Structured, boolean and divergent values do not define a scalar mean.
        if all(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            means[name] = statistics.mean(values)
    return means


def run_confirmations(plan: dict) -> list[dict]:
    """Authenticate the semantic finalist design before admitted paired repeats."""

    _validate_plan(plan)
    manifest = shared._read(Path(plan["control_root"]) / "finalists.json")
    freeze_finalists(plan, manifest["seeds"], top_k=manifest["top_k"])
    return shared.run_confirmations(plan)


def _validate_control_result(plan: dict, result: dict, expected: dict) -> None:
    """Authenticate a paired native control without relabeling it learned HPO."""

    # A control receipt belongs to one treatment, frozen recipe and model seed.
    if result.get("control_condition") != expected["control_condition"] \
    or result.get("training_seed") != expected["training_seed"] \
    or result.get("source_input_config_sha256") != expected["config_sha256"]:
        raise ValueError("Semantic control receipt does not match its frozen treatment, source and seed.")
    ordinary = dict(result)
    ordinary.pop("control_condition")
    validate_confirmation_objective(plan, ordinary)


def run_controls(plan: dict) -> list[dict]:
    """Run fresh baseline and unmodulated-distillation controls for each paired seed.

    Controls use independent native streams, admitted workers, the campaign's
    persistent hard deadline, and separate completion receipts. They never enter
    Optuna. The baseline omits semantic phases; feature MSE is a mechanistic
    control using the finalist's coefficients, not an independently tuned rival.
    """

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    from threading import Event


    _validate_plan(plan)
    manifest_path = Path(plan["control_root"]) / "finalists.json"
    manifest = shared._read(manifest_path)
    freeze_finalists(plan, manifest["seeds"], top_k=manifest["top_k"])
    with shared._coordinator(plan):
        manifest_digest = shared._digest(manifest_path)
        records = {}
        jobs = []
        index = 0
        for candidate in manifest["candidates"]:
            for condition in ("baseline", "unmodulated_feature_distillation"):
                for training_seed in shared._candidate_seeds(manifest, candidate):
                    destination = Path(plan["control_root"]) / "controls" / condition / f"trial-{candidate['trial_number']:04d}" / f"seed-{training_seed}"
                    expected = {
                        "manifest_sha256": manifest_digest, "config_sha256": candidate["config_sha256"], 
                        "trial_number": candidate["trial_number"], "training_seed": training_seed, 
                        "control_condition": condition
                    }
                    completed = destination / "completed.json"
                    # Completed controls are authenticated before being skipped.
                    if completed.exists():
                        record = shared._completed_record(completed, expected)
                        _validate_control_result(plan, record["result"], expected)
                        records[index] = record
                    # Missing controls require a fresh isolated stream.
                    else:
                        jobs.append({
                            "index": index, "candidate": candidate, "destination": destination, 
                            "expected": expected, "completed": completed
                        })
                    index += 1
        deadline = shared._phase_deadline(plan, "confirmation", start=bool(jobs))
        duration = capacity_plan(plan, target=1, confirmation_runs=0, control_runs=0)["conservative_full_trial_seconds"]
        gpu_ids = [gpu["gpu_id"] for gpu in plan["identity"].get("gpus", [])] or [None]
        cancel = Event()
        active = {}
        free_gpus = list(gpu_ids)
        next_job = 0
        exhausted = False
        admission_stopped = False
        failure = None
        with ThreadPoolExecutor(max_workers=len(gpu_ids), thread_name_prefix="semantic-control") as executor:
            try:
                while next_job < len(jobs) or active:
                    # The outer campaign deadline cancels outstanding owned workers.
                    if deadline is not None and time.time() >= deadline:
                        exhausted = True
                        cancel.set()
                    # Stop admissions early enough for an observed complete stream.
                    if deadline is not None and duration is not None and time.time() + duration >= deadline - 60.0:
                        admission_stopped = True
                    while free_gpus and next_job < len(jobs) and not exhausted and not admission_stopped and failure is None:
                        job = jobs[next_job]
                        next_job += 1
                        destination = job["destination"]
                        destination.mkdir(parents=True, exist_ok=True)
                        attempt = 1
                        while (destination / f"attempt-{attempt:03d}").exists():
                            attempt += 1
                        expected = job["expected"]
                        payload = {
                            "kind": "semantic_control", "input_config_path": job["candidate"]["input_config_path"], 
                            "output_root": str(destination / f"attempt-{attempt:03d}"), 
                            "training_seed": expected["training_seed"], "expected_config_sha256": expected["config_sha256"], 
                            "control_condition": expected["control_condition"]
                        }
                        gpu_id = free_gpus.pop(0)
                        tag = f"control-{expected['control_condition']}-{expected['trial_number']:04d}-{expected['training_seed']}"
                        future = executor.submit(
                            shared._launch, plan, payload, tag, deadline=deadline, 
                            gpu_id=gpu_id, cancel_event=cancel
                        )
                        active[future] = (job, gpu_id)
                    # Nothing remains to drain when all workers are idle.
                    if not active:
                        break
                    finished, _ = wait(active, timeout=0.2, return_when=FIRST_COMPLETED)
                    for future in finished:
                        job, gpu_id = active.pop(future)
                        free_gpus.append(gpu_id)
                        try:
                            result = future.result()
                            _validate_control_result(plan, result, job["expected"])
                            record = {"identity": job["expected"], "result": result}
                            shared._write(job["completed"], record)
                            records[job["index"]] = record
                        except TimeoutError:
                            exhausted = True
                            cancel.set()
                        except Exception as error:
                            # Preserve the first actual failure while cancelling peers.
                            if failure is None and not exhausted:
                                failure = error
                            cancel.set()
            except BaseException:
                cancel.set()
                raise
        shared._write(Path(plan["control_root"]) / "controls_status.json", {
            "completed_pairs": len(records), "required_pairs": index, "all_pairs_complete": len(records) == index, 
            "time_budget_exhausted": exhausted, "admission_stopped_for_full_trial": admission_stopped, 
            "deadline": deadline, "gpu_ids": gpu_ids
        })
        # Partial receipts survive a visible worker failure.
        if failure is not None:
            raise failure
        return [records[key] for key in sorted(records)]


def controls_summary(plan: dict) -> list[dict]:
    """Report paired control scores and learned-minus-control differences."""

    _validate_plan(plan)
    manifest_path = Path(plan["control_root"]) / "finalists.json"
    manifest = shared._read(manifest_path)
    freeze_finalists(plan, manifest["seeds"], top_k=manifest["top_k"])
    digest = shared._digest(manifest_path)
    rows = []
    for candidate in manifest["candidates"]:
        for condition in ("baseline", "unmodulated_feature_distillation"):
            scores = []
            differences = []
            per_seed = []
            candidate_seeds = shared._candidate_seeds(manifest, candidate)
            for training_seed in candidate_seeds:
                expected = {
                    "manifest_sha256": digest, "config_sha256": candidate["config_sha256"], 
                    "trial_number": candidate["trial_number"], "training_seed": training_seed
                }
                tail = Path(f"trial-{candidate['trial_number']:04d}") / f"seed-{training_seed}" / "completed.json"
                path = Path(plan["control_root"]) / "controls" / condition / tail
                # Unstarted and interrupted controls have no reported score.
                if not path.exists():
                    continue
                control_identity = {**expected, "control_condition": condition}
                record = shared._completed_record(path, control_identity)
                _validate_control_result(plan, record["result"], control_identity)
                score = float(record["result"]["objective"])
                scores.append(score)
                observed = {
                    "training_seed": training_seed, "objective": score, 
                    "validation_continual_metrics": deepcopy(record["result"].get("validation_continual_metrics", {})), 
                    "learned_minus_control": None
                }
                learned_path = Path(plan["control_root"]) / "confirmations" / tail
                # Paired differences require the corresponding completed learned seed.
                if learned_path.exists():
                    learned = shared._completed_record(learned_path, expected, plan=plan)
                    differences.append(float(learned["result"]["objective"]) - score)
                    observed["learned_minus_control"] = differences[-1]
                per_seed.append(observed)
            rows.append({
                "trial_number": candidate["trial_number"], "control_condition": condition, 
                "objective_metric": _metric(plan), "completed_seeds": len(scores), 
                "required_seeds": len(candidate_seeds), "paired_seeds": len(differences), 
                "mean_objective": statistics.mean(scores) if scores else None, 
                "std_objective": statistics.stdev(scores) if len(scores) > 1 else None, 
                "mean_learned_minus_control": statistics.mean(differences) if differences else None, 
                "per_seed": per_seed, "mean_validation_metrics": _mean_validation_metrics(per_seed), 
                "all_seeds_complete": len(scores) == len(candidate_seeds), 
                "comparable": bool(candidate_seeds) and len(differences) == len(candidate_seeds)
            })
    shared._write(Path(plan["control_root"]) / "controls_summary.json", rows)
    return rows


def cost_estimate(
    plan: dict, 
    remaining_trials: int = 120, 
    confirmation_runs: int = 9, 
    workers: int | None = None, 
    confirmation_workers: int | None = None
) -> dict:
    """Estimate remaining wall time from completed full-stream worker durations.

    Durations include the measured contention and all training/evaluation phases.
    Scaling assumes comparable recipes, hardware and concurrency; it is not a
    measured memory-capacity guarantee. Failed attempts are reported separately.
    """

    _validate_plan(plan)
    study = shared._load_study(plan)
    trials = [] if study is None else study.get_trials(deepcopy=False)
    workers = plan["hpo"]["concurrent_trials"] if workers is None else workers
    confirmation_workers = max(1, len(plan["identity"].get("gpus", []))) if confirmation_workers is None else confirmation_workers
    # Counts describe whole streams and admitted simultaneous workers exactly.
    if any(type(value) is not int or value < minimum for value, minimum in (
        (remaining_trials, 0), (confirmation_runs, 0), (workers, 1), (confirmation_workers, 1)
    )):
        raise ValueError("Trial/repeat counts must be nonnegative integers and workers a positive integer.")
    durations = []
    failed_seconds = 0.0
    for trial in trials:
        started = getattr(trial, "datetime_start", None)
        completed = getattr(trial, "datetime_complete", None)
        # Running/queued jobs have no final full-stream timing observation.
        if started is None or completed is None:
            continue
        elapsed = (completed - started).total_seconds()
        # Negative timestamps cannot define a measured duration.
        if elapsed <= 0:
            continue
        # Only finite complete streams estimate successful candidate durations.
        if trial.state.name == "COMPLETE" and trial.value is not None and math.isfinite(float(trial.value)):
            durations.append(elapsed / 3600.0)
        # Failed and pruned attempts contribute separate historical worker time.
        elif trial.state.name in {"FAIL", "PRUNED"}:
            failed_seconds += elapsed
    durations.sort()
    median = statistics.median(durations) if durations else None
    p90 = durations[max(0, math.ceil(0.9 * len(durations)) - 1)] if durations else None
    # The existing confirmation scheduler admits at most one repeat per GPU.
    batches = math.ceil(remaining_trials / workers) + math.ceil(confirmation_runs / confirmation_workers)
    return {
        "completed_timed_trials": len(durations), "median_full_trial_hours": median, 
        "p90_full_trial_hours": p90, "failed_or_pruned_worker_hours": failed_seconds / 3600.0, 
        "remaining_search_trials": remaining_trials, "confirmation_runs": confirmation_runs, 
        "assumed_workers": workers, "assumed_confirmation_workers": confirmation_workers, 
        "estimated_remaining_wall_hours_median": batches * median if median is not None else None, 
        "estimated_remaining_wall_hours_p90": batches * p90 if p90 is not None else None, 
        "assumption": "Same GPU class, sealed native stream/catalogue, semantic search distribution and measured concurrency; excludes future failed attempts and queue waits."
    }


def capacity_plan(
    plan: dict, 
    target: int = 64, 
    confirmation_runs: int = 9, 
    control_runs: int = 18, 
    safety_factor: float = 1.2, 
    paired_queue: bool = False
) -> dict:
    """Estimate sustainable full-stream admissions using measured p90 duration.

    No pilot duration or throughput is invented. The configured confirmation
    reserve is a minimum; measured confirmation and control waves can enlarge
    it without extending the persistent deadline. Legacy confirmations and
    controls use separate single-worker-per-device waves. paired_queue uses one
    combined admitted queue with the plan's verified worker concurrency. These are
    conditional timing estimates, not guarantees against unseen slow recipes.
    """

    _validate_plan(plan)
    # Capacity counts describe exact complete streams rather than fractional runs.
    if any(type(value) is not int or value < minimum for value, minimum in (
        (target, 1), (confirmation_runs, 0), (control_runs, 0)
    )):
        raise ValueError("Use a positive integer target and nonnegative repeat/control counts.")
    summary = search_summary(plan)
    workers = plan["hpo"]["concurrent_trials"]
    repeat_workers = workers if paired_queue else max(1, len(plan["identity"].get("gpus", [])))
    estimate = cost_estimate(
        plan, remaining_trials=max(0, target - summary["completed_finite_trials"]), 
        confirmation_runs=confirmation_runs + control_runs, workers=workers, 
        confirmation_workers=repeat_workers
    )
    clock = shared._budget_state(plan)
    now = time.time()
    p90 = estimate["p90_full_trial_hours"]
    duration = None if p90 is None else p90 * 3600.0 * safety_factor
    repeat_waves = math.ceil((confirmation_runs + control_runs) / repeat_workers) if paired_queue else (
        math.ceil(confirmation_runs / repeat_workers) + math.ceil(control_runs / repeat_workers)
    )
    policy = plan.get("time_budget") or {}
    configured_reserve = float(policy.get("confirmation_reserve_hours", 0.0)) * 3600.0
    measured_reserve = None if duration is None else repeat_waves * duration
    reserve = None if duration is None else max(configured_reserve, measured_reserve)
    cutoff = None
    admission_stop = None
    remaining = None
    sustainable = target if clock is None else None
    # Untimed legacy plans have no artificial completion deadline.
    if clock is not None:
        # Both the persistent outer watchdog and inner worker cleanup retain time.
        hard_cutoff = float(clock["deadline_unix"]) - float(clock["cleanup_seconds"]) - 60.0
        cutoff = float(clock["search_deadline_unix"]) - float(clock["cleanup_seconds"]) - 60.0
        # Measured repeat costs can enlarge the minimum reserve without extending time.
        if reserve is not None:
            cutoff = min(cutoff, hard_cutoff - reserve)
            admission_stop = cutoff - duration
            remaining = max(0.0, cutoff - now)
            sustainable = summary["completed_finite_trials"] + math.floor(remaining / duration) * workers
    return {
        **estimate, "requested_target": target, "control_runs": control_runs, 
        "confirmation_runs": confirmation_runs, "safety_factor": safety_factor, "paired_queue": paired_queue, 
        "conservative_full_trial_seconds": duration, "repeat_waves": repeat_waves, 
        "configured_reserve_seconds": configured_reserve, 
        "estimated_repeat_reserve_seconds": measured_reserve, "reserved_seconds": reserve, 
        "search_completion_cutoff_unix": cutoff, "admission_stop_unix": admission_stop, 
        "remaining_search_seconds": remaining, "sustainable_target": sustainable, 
        "pilot_required": duration is None, 
        "can_admit_full_trial": None if duration is None else (admission_stop is None or now < admission_stop), 
        "timing_scope": "Complete finite five-task-or-native-full-stream trials at measured search concurrency; repeat/control cost conservatively uses the same duration."
    }


def run_search(
    plan: dict, 
    target_completed: int = 64, 
    max_attempts: int = 256, 
    batch_trials: int | None = None, 
    confirmation_runs: int = 9, 
    control_runs: int = 18, 
    safety_factor: float = 1.2, 
    adapt_target: bool = True, 
    finalize_search: bool = False, 
    paired_queue: bool = False
) -> dict:
    """Measure one pilot wave, then refill workers until buffered admission stops.

    The first wave measures complete streams. Later invocations can refill every
    worker through the requested target; a separate admission deadline leaves
    the measured p90 allowance for active streams to finish. batch_trials=None
    imposes no additional post-pilot chunk barrier. A timed search may lower its
    target using observed throughput. Reaching a pilot or staged target leaves
    admissions open unless finalize_search is requested. Immutable finalists and
    shared hard watchdogs remain authoritative.
    """

    _validate_plan(plan)
    workers = plan["hpo"]["concurrent_trials"]
    # Admission targets, limits and optional chunks are exact positive trial counts.
    if any(type(value) is not int for value in (target_completed, max_attempts)) \
    or target_completed < 1 or max_attempts < target_completed \
    or (batch_trials is not None and (type(batch_trials) is not int or batch_trials < 1)):
        raise ValueError("Use integer budgets: target >= 1, attempts >= target, batch >= 1.")
    with shared._coordinator(plan):
        frozen = (Path(plan["control_root"]) / "finalists.json").exists()
        closed = search_closed(plan)
        # Reading a completed stage never creates or extends an execution clock.
        deadline = shared._phase_deadline(plan, "search", start=not frozen and not closed)
        timed_out = False
        admission_stopped = False
        while True:
            summary = search_summary(plan)
            complete = summary["completed_finite_trials"]
            allocated = summary["allocated_trials"]
            study = shared._load_study(plan)
            trials = [] if study is None else study.get_trials(deepcopy=False)
            waiting = sorted(trial.number for trial in trials if trial.state.name == "WAITING" and trial.number < max_attempts)
            capacity = capacity_plan(
                plan, target=target_completed, confirmation_runs=confirmation_runs, 
                control_runs=control_runs, safety_factor=safety_factor, paired_queue=paired_queue
            )
            target = target_completed
            # Adaptation may shrink a stage but cannot skip its explicit review boundary.
            if adapt_target and capacity["admission_stop_unix"] is not None:
                target = min(target_completed, max_attempts, capacity["sustainable_target"])
            exhausted = timed_out or (deadline is not None and time.time() >= deadline)
            buffered_stop = admission_stopped or capacity["can_admit_full_trial"] is False
            # Stages finish at their requested target or any immutable budget boundary.
            if frozen or closed or complete >= target or (allocated >= max_attempts and not waiting) or exhausted or buffered_stop:
                summary.update({
                    "target_completed": target, "initial_target_completed": target_completed, 
                    "target_reached": complete >= target, "time_budget_exhausted": exhausted, 
                    "admission_stopped_for_full_trial": buffered_stop, "finalists_frozen": frozen, 
                    "capacity_plan": capacity, "time_budget": budget_summary(plan)
                })
                should_close = finalize_search or exhausted or buffered_stop or (allocated >= max_attempts and not waiting)
                # Pilot completion alone does not permanently close future staged search.
                if not frozen and not closed and should_close and not summary["states"].get("RUNNING", 0):
                    _close_search(plan, "full_stream_admission_buffer" if buffered_stop else "search_finished")
                shared._write(Path(plan["control_root"]) / "search_status.json", summary)
                print(json.dumps(summary, indent=2), flush=True)
                return summary
            pilot = capacity["conservative_full_trial_seconds"] is None
            chunk_limit = workers if pilot else target - complete
            # An explicit chunk bound overrides unrestricted measured refilling.
            if batch_trials is not None:
                chunk_limit = min(chunk_limit, batch_trials)
            wave = min(chunk_limit, target - complete, max(0, max_attempts - allocated) + len(waiting))
            # The shared total-budget engine executes queued records below its
            # numeric target in addition to new allocations. Bound that prefix
            # so previously enqueued anchors cannot silently add extra waves.
            allowance = waiting[wave - 1] + 1 if len(waiting) >= wave else allocated + wave - len(waiting)
            payload = {"kind": "search", "allocated_target": allowance}
            completion_deadline = deadline
            # Measured streams stop new admissions while active workers drain to cutoff.
            if not pilot and capacity["admission_stop_unix"] is not None:
                payload["admission_deadline"] = capacity["admission_stop_unix"]
                completion_deadline = capacity["search_completion_cutoff_unix"]
            options = {} if completion_deadline is None else {"deadline": completion_deadline}
            try:
                result = shared._launch(
                    plan, payload, 
                    f"search-{allowance:04d}", **options
                )
            except TimeoutError:
                # An untimed launch failure is not an expected campaign cutoff.
                if deadline is None:
                    raise
                timed_out = True
                continue
            # The hard watchdog preserves partial evidence without counting completion.
            if result.get("time_budget_exhausted", False):
                timed_out = True
                continue
            # Startup can consume the remaining admission window without state changes.
            if result.get("admission_stopped", False):
                admission_stopped = True
                continue
            updated = search_summary(plan)
            # A successful child must otherwise change an observable trial state.
            if updated == summary:
                raise RuntimeError("HPO made no observable trial-state progress; inspect its logs.")
            print(json.dumps(updated, indent=2), flush=True)


def run_confirmation(
    input_config_path: str | Path, 
    output_root: str | Path, 
    training_seed: int, 
    expected_config_sha256: str, 
    control_condition: str | None = None
) -> dict:
    """Repeat the complete semantic stream with new training and semantic RNGs."""

    source = Path(input_config_path).resolve()
    # The supported mechanistic controls are explicit native route conditions.
    if control_condition not in {None, "baseline", "unmodulated_feature_distillation"}:
        raise ValueError("Unsupported semantic confirmation control treatment.")
    # Verify frozen bytes before importing the learning implementation.
    if shared._digest(source) != expected_config_sha256:
        raise ValueError("Finalist input config does not match its frozen SHA-256.")
    destination = Path(output_root).resolve()
    confirmation_input = destination / "confirmation-input.yaml"
    # An interrupted attempt remains evidence rather than being overwritten.
    if confirmation_input.exists():
        raise FileExistsError(f"Confirmation attempt already exists: {confirmation_input}")

    from common.config import load_config, save_config
    from common.semantic_hpo import normalize_semantic_profile, reseed_semantic_config, run_semantic_trial, validate_semantic_config


    config = load_config(source)
    # Loading must not race replacement of the selected source YAML.
    if shared._digest(source) != expected_config_sha256:
        raise ValueError("Finalist input config changed while it was being loaded.")
    profile = normalize_semantic_profile(
        config.hpo["semantic_profile"], dataset_name=config.dataset.name, seed=config.training.seed
    )
    validate_semantic_config(config, semantic_profile=profile)
    source_trial = config.hpo.get("trial_number")
    split_seed = config.hpo["continual_dataset_seed"]
    task_groups = deepcopy(config.continually_learn.task_groups)
    # A finite repeat cannot silently change the frozen data/task schedule.
    if task_groups != profile["task_groups"] or split_seed != profile["dataset_seed"]:
        raise ValueError("Semantic confirmation requires the frozen task stream and dataset seed.")
    reseed_semantic_config(config, seed=training_seed)
    config.continually_learn.resume_from = None
    config.continually_learn.checkpoint_dir = str(destination / "task-checkpoints")
    config.training.results_path = str(destination / "runs")
    config.training.project_tag = f"semantic-confirmation-t{source_trial}-s{training_seed}"
    config.training.tensorboard = True
    config.training.tensorboard_path = str(destination / "tensorboard")
    config.training.tensorboard_run_name = config.training.project_tag
    for key in (
        "objectives", "checkpoint_dir", "resume_original_trial_number", "resolved_config_path", 
        "classifier_weights_path", "execution", "pruning", "pruning_exchange", "pruning_monitor"
    ):
        config.hpo.pop(key, None)
    config.hpo.update({
        "seed": training_seed, "input_config_path": str(confirmation_input), 
        "confirmation": {
            "source_input_config_path": str(source), "source_input_config_sha256": expected_config_sha256, 
            "source_trial_number": source_trial, "dataset_seed": split_seed, 
            "task_groups": task_groups, "training_seed": training_seed, 
            "initial_weights_path": config.model.weights_path
        }
    })
    validate_semantic_config(config, semantic_profile=profile)
    destination.mkdir(parents=True, exist_ok=True)
    # Controls derive a distinct treatment identity after validating the learned recipe.
    if control_condition is not None:
        config.hpo["semantic_control"] = {
            "condition": control_condition, "source_semantic_trial_sha256": config.hpo.get("semantic_trial_sha256"), 
            "source_input_config_sha256": expected_config_sha256, "training_seed": training_seed
        }
        config.hpo["semantic_consolidation"]["condition"] = control_condition
        config.hpo["semantic_trial_sha256"] = hashlib.sha256(json.dumps({
            "control": config.hpo["semantic_control"], "route": config.hpo["semantic_consolidation"]
        }, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    save_config(config, confirmation_input)
    # Learned confirmations retain the semantic HPO adapter's exact route contract.
    if control_condition is None:
        result = run_semantic_trial(config)
    # Controls use the native route with their separately authenticated condition.
    else:
        from common.semantic_hpo import _append_retention_metrics
        from semantic_consolidation.config import RouteConfig, RouteSettings
        from semantic_consolidation.runner import run


        # Native controls deliberately bypass HPO's learned-only treatment boundary
        # after the frozen learned recipe has passed both identity validations.
        result = run(RouteConfig(common=config, route=RouteSettings(**config.hpo["semantic_consolidation"])))
        # New retention campaigns append the same audited endpoint objective for controls.
        if profile.get("selection_metric") == "retention_balanced_accuracy":
            _append_retention_metrics(config, result)
    metrics = result["evaluations"]["validation_continual_metrics"]
    metric = profile.get("selection_metric", METRIC)
    value = metrics[metric]
    # Structured or divergent results never count as completed paired repeats.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("Final semantic validation accuracy must be a finite scalar.")
    config.hpo["objectives"] = [float(value)]
    resolved = Path(result["results_path"]) / "config.yaml"
    save_config(config, resolved)
    receipt = {
        "objective": float(value), "objective_metric": metric, "objective_direction": "maximize", 
        "objective_network": "raw", "search_profile": PROFILE, 
        "fixed_recipe_sha256": profile["fixed_recipe_sha256"], 
        "validation_continual_metrics": metrics, "task_groups": task_groups, 
        "source_trial_number": source_trial, "source_input_config_path": str(source), 
        "source_input_config_sha256": expected_config_sha256, "training_seed": training_seed, 
        "dataset_seed": split_seed, "task_seed": profile["task_seed"], 
        "confirmation_input_config_path": str(confirmation_input), 
        "resolved_config_path": str(resolved), "results_path": result["results_path"]
    }
    # Treatment receipts preserve their separate identity for paired reporting.
    if control_condition is not None:
        receipt["control_condition"] = control_condition
        receipt["derived_control_sha256"] = config.hpo["semantic_trial_sha256"]
    return receipt


def run_control(
    input_config_path: str | Path, 
    output_root: str | Path, 
    training_seed: int, 
    expected_config_sha256: str, 
    control_condition: str
) -> dict:
    """Execute one admitted native control from an authenticated learned finalist."""

    # Dispatch cannot relabel an unsupported route as a mechanistic control.
    if control_condition not in {"baseline", "unmodulated_feature_distillation"}:
        raise ValueError("Unsupported semantic confirmation control treatment.")
    return run_confirmation(
        input_config_path, output_root, training_seed, expected_config_sha256, 
        control_condition=control_condition
    )
