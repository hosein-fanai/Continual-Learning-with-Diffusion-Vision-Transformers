"""Semantic-only HPO lifecycle over one fixed native continual DiT recipe.

The coordinator uses the existing admitted worker, immutable recipe, search and
paired-confirmation lifecycle. Learning happens only in isolated remote workers.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
import math
from pathlib import Path
import statistics

from common import dit_continual_runner as continual
from common import dit_hpo_runner as shared


PROFILE = "semantic_consolidation_runner"
METRIC = "final_average_accuracy"
make_plan = shared.make_plan
run_search = shared.run_search
run_confirmations = shared.run_confirmations
budget_summary = shared.budget_summary
start_experiment = shared.start_experiment


def _validate_plan(plan: dict) -> None:
    """Require semantic-only tuning with the complete raw continual objective."""

    settings = plan["hpo"]
    # Generic generation or continual-tuning plans cannot be relabeled semantic.
    if (
        settings.get("search_profile") != PROFILE or settings.get("task") != "continual"
        or settings.get("model_name") != "dit_classifier"
        or settings.get("objective_metrics") != [METRIC]
        or settings.get("objective_directions") != ["maximize"]
        or not isinstance(settings.get("semantic_profile"), dict)
    ):
        raise ValueError("Semantic runner requires its fixed native profile and raw final_average_accuracy maximization.")


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
    return {
        "allocated_trials": len(trials), "completed_finite_trials": len(ranked), 
        "states": states, "best_trial": ranked[0].number if ranked else None, 
        "best_validation_final_average_accuracy": best, 
        "improvement_last_50_valid_trials": best - max(float(trial.value) for trial in earlier) if earlier else None, 
        "task_groups": deepcopy(plan["hpo"]["semantic_profile"]["task_groups"]), 
        "fixed_recipe_sha256": plan["hpo"]["semantic_profile"]["fixed_recipe_sha256"]
    }


def freeze_finalists(plan: dict, seeds: Iterable[int], top_k: int = 3) -> dict:
    """Reuse immutable scalar-accuracy selection with paired fresh training seeds.

    The private view adapts only the selector's profile vocabulary. Study paths,
    source configurations, task identities, locks and seed checks remain exact.
    Search and worker execution always use the semantic profile.
    """

    _validate_plan(plan)
    selection = dict(plan)
    profile = plan["hpo"]["semantic_profile"]
    # The generic selector names its data seed task_seed; native semantic inputs may separate them.
    selection["hpo"] = {
        **plan["hpo"], "search_profile": continual.PROFILE, 
        "continual_profile": {**profile, "task_seed": profile["dataset_seed"]}
    }
    return continual.freeze_finalists(selection, seeds, top_k=top_k)


def validate_confirmation_objective(plan: dict, result: dict) -> None:
    """Bind finite confirmation scores to the same native stream and semantic route."""

    _validate_plan(plan)
    profile = plan["hpo"]["semantic_profile"]
    expected = {
        "objective_metric": METRIC, "objective_direction": "maximize", "objective_network": "raw", 
        "task_groups": profile["task_groups"], "search_profile": PROFILE, 
        "fixed_recipe_sha256": profile["fixed_recipe_sha256"]
    }
    objective = result.get("objective")
    # Completion must not accept a plain continual run or a different frozen stream.
    if any(result.get(key) != value for key, value in expected.items()) or isinstance(objective, bool) \
    or not isinstance(objective, (int, float)) or not math.isfinite(float(objective)):
        raise ValueError("Confirmation does not match the finite semantic objective and frozen task stream.")


def confirmation_summary(plan: dict) -> list[dict]:
    """Summarize authenticated paired accuracy means and incomplete repeat counts."""

    _validate_plan(plan)
    manifest_path = Path(plan["control_root"]) / "finalists.json"
    manifest = shared._read(manifest_path)
    manifest_digest = shared._digest(manifest_path)
    rows = []
    for candidate in manifest["candidates"]:
        scores = []
        for training_seed in manifest["seeds"]:
            path = Path(plan["control_root"]) / "confirmations" / f"trial-{candidate['trial_number']:04d}" / f"seed-{training_seed}" / "completed.json"
            # Partial reports never invent outcomes for missing seeds.
            if path.exists():
                expected = {
                    "manifest_sha256": manifest_digest, "config_sha256": candidate["config_sha256"], 
                    "trial_number": candidate["trial_number"], "training_seed": training_seed
                }
                scores.append(float(shared._completed_record(path, expected, plan=plan)["result"]["objective"]))
        rows.append({
            "trial_number": candidate["trial_number"], 
            "search_final_average_accuracy": candidate["search_final_average_accuracy"], 
            "completed_seeds": len(scores), "required_seeds": len(manifest["seeds"]), 
            "mean_final_average_accuracy": statistics.mean(scores) if scores else None, 
            "std_final_average_accuracy": statistics.stdev(scores) if len(scores) > 1 else None, 
            "all_seeds_complete": len(scores) == len(manifest["seeds"]), 
            "comparable": len(scores) == len(manifest["seeds"])
        })
    shared._write(Path(plan["control_root"]) / "confirmation_summary.json", rows)
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
        "assumption": "Same GPU class, fixed native stream, semantic search distribution and measured concurrency; excludes future failed attempts and queue waits."
    }


def run_confirmation(
    input_config_path: str | Path, 
    output_root: str | Path, 
    training_seed: int, 
    expected_config_sha256: str
) -> dict:
    """Repeat the complete semantic stream with new training and semantic RNGs."""

    source = Path(input_config_path).resolve()
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
    save_config(config, confirmation_input)
    result = run_semantic_trial(config)
    metrics = result["evaluations"]["validation_continual_metrics"]
    value = metrics[METRIC]
    # Structured or divergent results never count as completed paired repeats.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("Final semantic validation accuracy must be a finite scalar.")
    config.hpo["objectives"] = [float(value)]
    resolved = Path(result["results_path"]) / "config.yaml"
    save_config(config, resolved)
    return {
        "objective": float(value), "objective_metric": METRIC, "objective_direction": "maximize", 
        "objective_network": "raw", "search_profile": PROFILE, 
        "fixed_recipe_sha256": profile["fixed_recipe_sha256"], 
        "validation_continual_metrics": metrics, "task_groups": task_groups, 
        "source_trial_number": source_trial, "source_input_config_path": str(source), 
        "source_input_config_sha256": expected_config_sha256, "training_seed": training_seed, 
        "dataset_seed": split_seed, "confirmation_input_config_path": str(confirmation_input), 
        "resolved_config_path": str(resolved), "results_path": result["results_path"]
    }
