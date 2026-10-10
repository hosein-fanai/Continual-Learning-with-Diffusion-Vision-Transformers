"""Continual DiT notebook lifecycle using the shared admitted HPO runner.

The coordinator imports no learning framework. Search and paired confirmations
use the public HPO/training APIs inside the existing isolated GPU workers.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
import math
from pathlib import Path
import statistics
import time

from common import dit_hpo_runner as shared


PROFILE = "dit_continual_runner"
METRIC = "final_average_accuracy"
make_plan = shared.make_plan
run_search = shared.run_search
run_confirmations = shared.run_confirmations
budget_summary = shared.budget_summary
start_experiment = shared.start_experiment


def _validate_plan(plan: dict) -> None:
    """Require the exact raw continual objective before interpreting receipts."""

    settings = plan["hpo"]
    # A scalar generation plan must never be interpreted as continual accuracy.
    if (
        settings.get("search_profile") != PROFILE or settings.get("task") != "continual"
        or settings.get("model_name") != "dit_classifier"
        or settings.get("objective_metrics") != [METRIC]
        or settings.get("objective_directions") != ["maximize"]
    ):
        raise ValueError("Continual runner requires raw final_average_accuracy maximization.")


def search_summary(plan: dict) -> dict:
    """Report maximize-directed progress and the requested teacher-source cells."""

    from common.dit_continual_hpo import baseline_hints


    _validate_plan(plan)
    study = shared._load_study(plan)
    trials = [] if study is None else study.get_trials(deepcopy=False)
    ranked = shared._ranked(study, direction="maximize")
    states = {}
    overrides = plan["hpo"].get("search_space_overrides", {})
    coverage = {
        f"{route['classifier_teacher_source']}/{route['noise_teacher_source']}": {"allocated": 0, "complete": 0}
        for route in baseline_hints(search_space_overrides=overrides)
    }
    for trial in trials:
        states[trial.state.name] = states.get(trial.state.name, 0) + 1
        classifier = trial.params.get("classifier_teacher_source")
        noise = trial.params.get("noise_teacher_source")
        key = f"{classifier}/{noise}"
        # Partially allocated trials may not yet have both source suggestions.
        if key in coverage:
            coverage[key]["allocated"] += 1
            # Failed or resource-pruned attempts do not establish successful coverage.
            if trial in ranked:
                coverage[key]["complete"] += 1
    earlier = sorted(ranked, key=lambda trial: trial.number)[:-50]
    best = float(ranked[0].value) if ranked else None
    return {
        "allocated_trials": len(trials), "completed_finite_trials": len(ranked), 
        "states": states, "best_trial": ranked[0].number if ranked else None, 
        "best_validation_final_average_accuracy": best, 
        "improvement_last_50_valid_trials": best - max(float(trial.value) for trial in earlier) if earlier else None, 
        "teacher_source_coverage": coverage, 
        "all_teacher_modes_completed": all(item["complete"] > 0 for item in coverage.values()), 
        "task_groups": plan["hpo"]["continual_profile"]["task_groups"]
    }


def freeze_finalists(plan: dict, seeds: Iterable[int], top_k: int = 3) -> dict:
    """Freeze distinct maximum-accuracy recipes with a paired fresh-seed design."""

    _validate_plan(plan)
    seeds = list(seeds)
    # Confirmations require exact independent paired seed identities.
    if not seeds or any(type(value) is not int for value in seeds) \
    or len(set(seeds)) != len(seeds) or plan["hpo"]["seed"] in seeds:
        raise ValueError("Use distinct integer confirmation seeds different from the search seed.")
    # The requested selection is an exact nonempty number of recipes.
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer candidate count.")
    policy = {"objective_metric": METRIC, "objective_direction": "maximize", "objective_network": "raw"}
    with shared._coordinator(plan):
        path = Path(plan["control_root"]) / "finalists.json"
        # Reruns retain the original immutable selection and seed pairing.
        if path.exists():
            manifest = shared._read(path)
            # A changed confirmation design belongs to a new experiment.
            if manifest["seeds"] != seeds or manifest["top_k"] != top_k or manifest.get("selection_policy") != policy:
                raise ValueError("Frozen continual finalists have a different selection or seed design.")
            for candidate in manifest["candidates"]:
                # Authenticate source bytes before skipping a completed freeze.
                if shared._digest(candidate["input_config_path"]) != candidate["config_sha256"]:
                    raise ValueError("A frozen finalist configuration has changed.")
            return manifest
        study = shared._load_study(plan)
        ranked = shared._ranked(study, direction="maximize")
        distinct = []
        configurations = []
        for trial in ranked:
            # Repeated parameters receive only one finalist slot.
            if trial.params not in configurations:
                configurations.append(dict(trial.params))
                distinct.append(trial)
        # Do not silently shrink the requested paired comparison.
        if len(distinct) < top_k:
            raise ValueError("Not enough distinct finite completed configurations to select finalists.")
        trials = study.get_trials(deepcopy=False)
        # Running workers can still publish a superior candidate.
        if any(trial.state.name == "RUNNING" for trial in trials):
            raise ValueError("Finish or recover running trials before freezing finalists.")
        waiting = [trial.number for trial in trials if trial.state.name == "WAITING"]
        deadline = shared._phase_deadline(plan, "search")
        # Only an expired persistent deadline may leave queued hints unstarted.
        if waiting and (deadline is None or time.time() < deadline - 60.0):
            raise ValueError("Finish or recover queued trials before freezing finalists.")
        candidates = []
        for rank, trial in enumerate(distinct[:top_k], start=1):
            original = Path(plan["study_root"]) / "configs" / f"trial-{trial.number:04d}.yaml"
            frozen = Path(plan["control_root"]) / "finalist_configs" / original.name
            frozen.parent.mkdir(parents=True, exist_ok=True)
            frozen.write_bytes(original.read_bytes())
            candidates.append({
                "rank": rank, "trial_number": trial.number, "params": dict(trial.params), 
                "search_final_average_accuracy": float(trial.value), 
                "input_config_path": str(frozen), "config_sha256": shared._digest(frozen)
            })
        manifest = {
            "study_name": plan["study_name"], "study_allocated_trials": len(trials), 
            "top_k": top_k, "seeds": seeds, "dataset_seed": plan["hpo"]["continual_profile"]["task_seed"], 
            "task_groups": deepcopy(plan["hpo"]["continual_profile"]["task_groups"]), 
            "selection_policy": policy, "candidates": candidates
        }
        # Preserve excluded queued identities without inventing outcomes.
        if waiting:
            manifest["unstarted_trials_at_search_deadline"] = {
                "trial_numbers": waiting, "search_execution_deadline_unix": deadline - 60.0
            }
        shared._write(path, manifest)
        return manifest


def validate_confirmation_objective(plan: dict, result: dict) -> None:
    """Reject mismatched task streams, networks, directions or partial objectives."""

    _validate_plan(plan)
    expected = {
        "objective_metric": METRIC, "objective_direction": "maximize", "objective_network": "raw", 
        "task_groups": plan["hpo"]["continual_profile"]["task_groups"]
    }
    objective = result.get("objective")
    # A finite result must identify the same data stream and scoring protocol.
    if any(result.get(key) != value for key, value in expected.items()) or isinstance(objective, bool) \
    or not isinstance(objective, (int, float)) or not math.isfinite(float(objective)):
        raise ValueError("Confirmation does not match the finite raw continual objective and frozen task stream.")


def confirmation_summary(plan: dict) -> list[dict]:
    """Summarize authenticated paired accuracy means, retaining partial progress."""

    _validate_plan(plan)
    manifest_path = Path(plan["control_root"]) / "finalists.json"
    manifest = shared._read(manifest_path)
    manifest_digest = shared._digest(manifest_path)
    rows = []
    for candidate in manifest["candidates"]:
        scores = []
        for training_seed in manifest["seeds"]:
            path = Path(plan["control_root"]) / "confirmations" / f"trial-{candidate['trial_number']:04d}" / f"seed-{training_seed}" / "completed.json"
            # Missing seeds remain absent while successful partial progress is retained.
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


def run_confirmation(
    input_config_path: str | Path, 
    output_root: str | Path, 
    training_seed: int, 
    expected_config_sha256: str
) -> dict:
    """Replay a fixed continual recipe through the existing complete training API.

    Architecture, original initialization artifact, task groups, dataset split,
    distillation constants and specialist recipes remain fixed. Only training
    random streams change. No searched checkpoint initializes a confirmation.
    """

    source = Path(input_config_path).resolve()
    # Reject source replacement before importing the training stack.
    if shared._digest(source) != expected_config_sha256:
        raise ValueError("Finalist input config does not match its frozen SHA-256.")
    destination = Path(output_root).resolve()
    confirmation_input = destination / "confirmation-input.yaml"
    # Failed attempts remain separate evidence instead of being overwritten.
    if confirmation_input.exists():
        raise FileExistsError(f"Confirmation attempt already exists: {confirmation_input}")

    from common.config import load_config, save_config
    from common.dit_continual_hpo import normalize_continual_profile
    from common.train import main


    config = load_config(source)
    # Loading must not race replacement of the frozen YAML.
    if shared._digest(source) != expected_config_sha256:
        raise ValueError("Finalist input config changed while it was being loaded.")
    profile = normalize_continual_profile(config.hpo["continual_profile"], dataset_name="cifar10", seed=config.training.seed)
    # Authenticate the model family, objective and validation source.
    if (
        config.hpo.get("search_profile") != PROFILE or config.training.task != "continual"
        or config.model.name != "dit_classifier" or config.model.wrapper_name != "diffusion_classifier"
        or config.model.wrapper_kwargs.get("use_ema") is not False
        or config.model.wrapper_kwargs.get("test_network_name") != "raw"
        or config.hpo.get("objective_metrics") != [METRIC]
        or config.hpo.get("objective_directions") != ["maximize"]
        or not config.training.use_valset or config.dataset.validation_source not in ("split", "test")
    ):
        raise ValueError("Confirmation requires the fixed raw V1 continual validation recipe.")
    split_seed = config.hpo.get("continual_dataset_seed", config.training.seed)
    source_trial = config.hpo.get("trial_number")
    task_groups = deepcopy(config.continually_learn.task_groups)
    # Paired confirmation seeds must not resample the task partition or order.
    if task_groups != profile["task_groups"]:
        raise ValueError("Confirmation requires the frozen five two-class tasks.")
    config.training.seed = training_seed
    config.model.kwargs["seed"] = training_seed
    config.model.wrapper_kwargs["seed"] = training_seed
    config.continually_learn.seed = training_seed
    config.continually_learn.resume_from = None
    config.continually_learn.checkpoint_dir = None
    config.training.results_path = str(destination / "runs")
    config.training.project_tag = f"confirmation-t{source_trial}-s{training_seed}"
    config.training.tensorboard = True
    config.training.tensorboard_path = str(destination / "tensorboard")
    config.training.tensorboard_run_name = config.training.project_tag
    for key in (
        "objectives", "checkpoint_dir", "resume_original_trial_number", "resolved_config_path", 
        "classifier_weights_path", "execution", "pruning", "pruning_exchange", "pruning_monitor"
    ):
        config.hpo.pop(key, None)
    config.hpo.update({
        "seed": training_seed, "continual_dataset_seed": split_seed, 
        "input_config_path": str(confirmation_input), 
        "confirmation": {
            "source_input_config_path": str(source), "source_input_config_sha256": expected_config_sha256, 
            "source_trial_number": source_trial, "dataset_seed": split_seed, 
            "task_groups": task_groups, "training_seed": training_seed, 
            "initial_weights_path": config.model.weights_path
        }
    })
    destination.mkdir(parents=True, exist_ok=True)
    save_config(config, confirmation_input)
    result = main(config)
    metrics = result["evaluations"]["validation_continual_metrics"]
    value = metrics[METRIC]
    # Divergent or structured values cannot count as completed seed repeats.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("Final continual validation accuracy must be a finite scalar.")
    config.hpo["objectives"] = [float(value)]
    resolved = Path(result["results_path"]) / "config.yaml"
    save_config(config, resolved)
    return {
        "objective": float(value), "objective_metric": METRIC, "objective_direction": "maximize", 
        "objective_network": "raw", "validation_continual_metrics": metrics, "task_groups": task_groups, 
        "source_trial_number": source_trial, "source_input_config_path": str(source), 
        "source_input_config_sha256": expected_config_sha256, "training_seed": training_seed, 
        "dataset_seed": split_seed, "confirmation_input_config_path": str(confirmation_input), 
        "resolved_config_path": str(resolved), "results_path": result["results_path"]
    }
