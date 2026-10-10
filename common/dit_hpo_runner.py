"""Resumable notebook orchestration for generation and named DiT HPO protocols.

The notebook reads study metadata. Admitted child processes use the existing
HPO and training APIs. Targets count finite COMPLETE trials; attempt ceilings
include every allocated trial, including failures and interruptions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time

from subprocess import TimeoutExpired

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from common.hpo_sqlite import database_path


def _read(path: str | Path) -> Any:
    """Read one UTF-8 JSON artifact."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str | Path, value: Any) -> None:
    """Atomically publish a finite JSON artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _digest(path: str | Path) -> str:
    """Return the SHA-256 identity of an artifact."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def _coordinator(plan: dict) -> Iterator[None]:
    """Hold the existing process lock in a separate runner control directory."""

    from common.hpo_process import study_lock


    root = Path(plan["control_root"])
    root.mkdir(parents=True, exist_ok=True)
    with study_lock(root):
        yield


def make_plan(
    checkout_root: str | Path, 
    results_path: str | Path, 
    dataset_name: str = "CIFAR10", 
    epochs: int = 50, 
    n_startup_trials: int = 40, 
    concurrent_trials: int = 1, 
    gpu_ids: list[int] | None = None, 
    pruning: dict[str, object] | None = None, 
    validation_source: str = "split", 
    validation_ratio: float = 0.2, 
    experiment_hours: float | None = None, 
    confirmation_reserve_hours: float = 2.0, 
    search_space_overrides: dict[str, object] | None = None, 
    worker_gpu_memory_limit_mb: int | None = None, 
    transfer_manifest: dict[str, object] | None = None, 
    search_profile: str | None = None, 
    continual_profile: dict[str, object] | None = None, 
    model_name: str = "diffusion_transformer", 
    seed: int = 42
) -> dict:
    """Seal the scientific recipe while leaving trial targets adjustable.

    Args:
        checkout_root: Verified remote project directory.
        results_path: Experiment root, absolute or relative to the checkout.
        dataset_name: One dataset per independent study.
        epochs: Maximum epochs for every search trial.
        n_startup_trials: Random observations before adaptive sampling.
        concurrent_trials: Number of isolated HPO trial workers; one is serial.
            This execution setting does not change the scientific recipe.
        gpu_ids: Selected physical GPU indices. None preserves the original GPU-zero
            execution path; an explicit list uses isolated, device-assigned trials.
            concurrent_trials is the total across selected devices.
        pruning: Optional shared HPO pruning policy. None keeps performance pruning
            disabled. This scientific setting is sealed for study recovery. Classifier
            Pareto plans require None; OOM and numerical guards remain enabled.
        validation_source: Existing HPO validation protocol, split or test.
        validation_ratio: Existing HPO validation fraction; use zero for test.
        experiment_hours: Optional persistent wall-clock budget, started by search,
            confirmations or an explicit start_experiment call before GPU preflight.
        confirmation_reserve_hours: Portion of that budget reserved for confirmations.
        search_space_overrides: Optional public HPO search-space options, copied and
            sealed with the scientific recipe. None preserves the existing search space.
        worker_gpu_memory_limit_mb: Optional exact per-worker TensorFlow cap in MiB.
            Admission reserves overhead separately; this runtime control is not sealed
            in the scientific recipe. None retains the existing automatic budget.
        transfer_manifest: Optional frozen source-study provenance. Follow-up plans
            derive partial starting suggestions, seal this manifest and preserve
            all source artifacts. None retains the existing runner recipe.
        search_profile: None retains ordinary generation. dit_classifier_runner
            selects joint raw DiT classification with maximized accuracy
            and minimized noise loss; the older joint profile stays separate. Its
            default search space queues the archive baseline as a fresh trial.
            Explicit nonempty overrides omit that suggestion so fixed hints
            cannot fall outside the requested distributions.
        continual_profile: Fixed student configuration, specialist teacher recipes,
            and task seed for dit_continual_runner. The named continual profile
            freezes five CIFAR-10 tasks and maximizes final average accuracy.
        model_name: Ordinary generation denoiser, diffusion_transformer or unet.
            Named DiT profiles retain their existing classifier model; UNet cannot
            use those profiles or DiT transfer manifests.
        seed: Shared search, initialization and dataset-split seed.

    Returns:
        dict: Paths, public HPO arguments and the remote worker identity.

    Raises:
        RuntimeError: The runtime is not an authorized remote host.
        ValueError: Existing runner provenance differs from this recipe.
    """

    from common.dit_hpo_remote import inspect_remote


    # Named classifier plans cannot silently become a legacy Pareto study.
    if search_profile not in (None, "dit_classifier_runner", "dit_continual_runner"):
        raise ValueError("Unknown DiT runner search_profile.")
    # Architecture-specific profiles and transfer hints cannot change model families.
    if model_name not in ("diffusion_transformer", "unet"):
        raise ValueError("Generation runner model_name must be 'diffusion_transformer' or 'unet'.")
    # UNet has no compatible classifier/continual profile or transformer transfer hints.
    if model_name == "unet" and (search_profile is not None or transfer_manifest is not None):
        raise ValueError("UNet generation does not support named DiT profiles or DiT transfer manifests.")
    continual = search_profile == "dit_continual_runner"
    # Architecture and specialist inputs belong only to the explicit continual recipe.
    if not continual and continual_profile is not None:
        raise ValueError("continual_profile requires search_profile='dit_continual_runner'.")
    # Normalize serialized inputs before remote admission or study publication.
    if continual:
        from common.dit_continual_hpo import baseline_hints, normalize_continual_profile, validate_continual_search


        continual_profile = normalize_continual_profile(continual_profile, dataset_name=dataset_name, seed=seed)
        validate_continual_search(continual_profile, search_space_overrides=search_space_overrides)
        # Epoch counters restart across continual phases and are not comparable.
        if pruning is not None:
            raise ValueError("Continual task-phase HPO requires pruning=None.")
    # Scalar epoch pruning cannot decide a two-objective classifier tradeoff.
    if search_profile == "dit_classifier_runner" and pruning is not None:
        raise ValueError("Classifier Pareto HPO requires pruning=None; OOM and numerical-divergence pruning remain enabled.")
    # Generation transfer provenance cannot describe the classifier objective.
    if search_profile is not None and transfer_manifest is not None:
        raise ValueError("Generation transfer manifests do not apply to classifier plans.")
    root = Path(checkout_root).resolve()
    options = {} if worker_gpu_memory_limit_mb is None else {
        "worker_gpu_memory_limit_mb": worker_gpu_memory_limit_mb
    }
    # Omitted controls retain the established inspection call contract.
    if gpu_ids is None:
        identity = inspect_remote(root, concurrent_trials=concurrent_trials, **options)
    # Explicit devices are validated and bound to their measured UUIDs.
    else:
        identity = inspect_remote(root, concurrent_trials=concurrent_trials, gpu_ids=gpu_ids, **options)
    results = Path(results_path)
    results = (root / results).resolve() if not results.is_absolute() else results.resolve()
    classifier = search_profile == "dit_classifier_runner"
    task = "continual" if continual else ("joint" if classifier else "generation")
    model_name = "dit_classifier" if classifier or continual else model_name
    study_root = results / task / model_name / dataset_name.lower()
    # Separate named-profile storage retains the public HPO path contract.
    if classifier or continual:
        study_root = study_root / search_profile
    plan = {
        "version": 2, 
        "checkout_root": str(root), 
        "study_root": str(study_root), 
        "control_root": str(study_root / "notebook_runner"), 
        "study_name": f"{task}-{model_name}-" + dataset_name.lower() + (f"-{search_profile}" if classifier or continual else ""), 
        "hpo": {
            "task": task, 
            "model_name": model_name, 
            "dataset_name": dataset_name, 
            "epochs": epochs, 
            "results_path": str(results), 
            "fit_method": "fit", 
            "objective_metrics": ["final_average_accuracy"] if continual else (["classification_accuracy", "noise_loss"] if classifier else ["generation_loss"]), 
            "objective_directions": ["maximize"] if continual else (["maximize", "minimize"] if classifier else ["minimize"]), 
            "dtype_policy": "float32", 
            "n_startup_trials": n_startup_trials, 
            "trial_budget_mode": "total", 
            "validation_source": validation_source, 
            "validation_ratio": validation_ratio, 
            "concurrent_trials": concurrent_trials, 
            "seed": seed
        }, 
        "identity": identity
    }
    # Omitted profiles retain the exact original scientific recipe.
    if classifier or continual:
        plan["hpo"]["search_profile"] = search_profile
        # Default-space baselines are fresh suggestions, never transferred scores.
        if classifier and not search_space_overrides:
            from common.dit_classifier_hpo import baseline_hints


            plan["hpo"]["initial_trials"] = baseline_hints()
    # Queue source coverage while retaining one native continual model family.
    if continual:
        plan["hpo"].update({
            "continual_profile": continual_profile, "task_size": 2, 
            "use_distillation": True, "use_ensemble_accuracy": False, 
            "initial_trials": baseline_hints(search_space_overrides=search_space_overrides)
        })
    # A coupled phase allocation must leave time for both search and cleanup.
    if experiment_hours is not None:
        total = float(experiment_hours)
        reserve = float(confirmation_reserve_hours)
        # Both finite phase lengths must fit the shared wall-clock budget.
        if not math.isfinite(total) or not math.isfinite(reserve) or not 0 <= reserve < total:
            raise ValueError("The confirmation reserve must be finite and smaller than the experiment budget.")
        plan["time_budget"] = {"experiment_hours": total, "confirmation_reserve_hours": reserve}
    # Pruning changes candidate selection and belongs to the sealed scientific recipe.
    if pruning is not None:
        plan["hpo"]["pruning"] = dict(pruning)
    # Candidate distributions are scientific settings, independent of worker routing.
    if search_space_overrides is not None:
        plan["hpo"]["search_space_overrides"] = copy.deepcopy(search_space_overrides)
    # Explicit routing assigns one physical GPU to each isolated trial process.
    if gpu_ids is not None:
        plan["hpo"]["worker_gpu_ids"] = [gpu["gpu_uuid"] for gpu in identity["gpus"]]
    # Only isolated HPO workers consume the shared API's per-worker memory cap.
    if concurrent_trials > 1 or gpu_ids is not None or pruning is not None or worker_gpu_memory_limit_mb is not None:
        plan["hpo"]["worker_gpu_memory_limit_mb"] = identity["worker_policy"]["tf_memory_mib"]
    # A separate follow-up may import suggestions, never source scores or storage.
    if transfer_manifest is not None:
        from common.dit_hpo_followup import followup_initial_trials, validate_followup_protocol


        frozen = validate_followup_protocol(transfer_manifest, plan["hpo"])
        source_results = Path(frozen["request"]["source_results_path"]).resolve()
        # Output placement must not write anywhere inside the upstream result tree.
        if results == source_results or results.is_relative_to(source_results):
            raise ValueError("Follow-up results must be outside the source results directory.")
        branches = plan["hpo"].get("search_space_overrides", {}).get("dit_followup_branch")
        # Explicit branch choices authenticate the intended missing-space search.
        if not isinstance(branches, list):
            raise ValueError("Transferred DiT plans require explicit dit_followup_branch choices.")
        plan["transfer_manifest"] = frozen
        plan["hpo"]["initial_trials"] = followup_initial_trials(frozen, branches)
    scientific = {
        "version": plan["version"], 
        "hpo": {
            key: value for key, value in plan["hpo"].items()
            if key not in {"concurrent_trials", "worker_gpu_memory_limit_mb", "worker_gpu_ids"}
        }, 
        "source_sha256": identity["source_sha256"], 
        "versions": identity["versions"], 
        "python": identity["python"]
    }
    # Source provenance is immutable even when its live study later grows.
    if transfer_manifest is not None:
        scientific["transfer_manifest"] = plan["transfer_manifest"]
    with _coordinator(plan):
        path = Path(plan["control_root"]) / "recipe.json"
        # Existing recipes cannot silently change on notebook restart.
        if path.exists() and _read(path) != scientific:
            raise ValueError("Runner recipe changed; use a fresh RESULTS_PATH.")
        _write(path, scientific)
    return plan


def _budget_state(plan: dict, start: bool = False) -> dict | None:
    """Read the durable execution clock, creating it only under the coordinator lock."""

    path = Path(plan["control_root"]) / "budget.json"
    policy = plan.get("time_budget")
    # Existing clocks remain authoritative even if a later setup omits the budget.
    if path.exists():
        state = _read(path)
        # Restarting cannot silently extend or redistribute an active clock.
        if policy is not None and state["policy"] != policy:
            raise ValueError("Experiment time budget changed; use a fresh RESULTS_PATH.")
        return state
    # Setup and legacy untimed callers never create a clock.
    if policy is None or not start:
        return None
    started = time.time()
    deadline = started + 3600.0 * policy["experiment_hours"]
    state = {
        "version": 1, 
        "policy": dict(policy), 
        "started_at_unix": started, 
        "deadline_unix": deadline, 
        "search_deadline_unix": deadline - 3600.0 * policy["confirmation_reserve_hours"], 
        "cleanup_seconds": 60.0
    }
    _write(path, state)
    return state


def _phase_deadline(plan: dict, phase: str, start: bool = False) -> float | None:
    """Return the absolute execution cutoff, leaving time for process cleanup."""

    state = _budget_state(plan, start=start)
    # An unstarted or untimed experiment has no absolute cutoff.
    if state is None:
        return None
    field = "search_deadline_unix" if phase == "search" else "deadline_unix"
    return float(state[field]) - float(state["cleanup_seconds"])


def budget_summary(plan: dict) -> dict:
    """Describe the persistent experiment clock without starting or resetting it."""

    state = _budget_state(plan)
    # An unstarted or untimed experiment has no absolute cutoff.
    if state is None:
        return {"configured": plan.get("time_budget") is not None, "started": False}
    now = time.time()
    return {
        **state, 
        "configured": True, 
        "started": True, 
        "remaining_seconds": max(0.0, state["deadline_unix"] - now), 
        "search_time_budget_exhausted": now >= state["search_deadline_unix"] - state["cleanup_seconds"], 
        "time_budget_exhausted": now >= state["deadline_unix"] - state["cleanup_seconds"]
    }


def start_experiment(plan: dict) -> dict:
    """Start or resume the shared clock before optional compatibility work.

    Call this immediately before a GPU preflight to include that work in the
    same budget as search and confirmations. The preflight must enforce the
    returned search_deadline_unix minus cleanup_seconds as its absolute cutoff.
    Subsequent calls retain the original start and deadlines; untimed plans
    remain untimed. Ordinary search still starts its clock automatically when
    callers do not use this entry point.

    Args:
        plan: Fixed recipe returned by make_plan.

    Returns:
        dict: The persistent budget_summary, including absolute phase deadlines
            for timed plans. No study trials or GPU workers are created.
    """

    with _coordinator(plan):
        _budget_state(plan, start=True)
        return budget_summary(plan)


def _load_study(plan: dict) -> Any:
    """Read the existing Optuna study without creating its database."""

    import optuna


    database = database_path(plan["study_root"])
    # An unstarted experiment has no allocated trials.
    if not database.exists():
        return None
    return optuna.load_study(
        study_name=plan["study_name"], 
        storage="sqlite:///" + database.resolve().as_posix()
    )


def _classifier_plan(plan: dict) -> bool:
    """Recognize only the explicit two-objective raw classifier recipe."""

    options = plan["hpo"]
    classifier = options.get("search_profile") == "dit_classifier_runner"
    # Old scalar plans cannot be resumed or interpreted as Pareto experiments.
    if classifier and (
        options.get("task") != "joint" or options.get("model_name") != "dit_classifier"
        or options.get("objective_metrics") != ["classification_accuracy", "noise_loss"]
        or options.get("objective_directions") != ["maximize", "minimize"]
    ):
        raise ValueError("Classifier runner requires accuracy maximization and noise_loss minimization.")
    return classifier


def _finite_pair(values: Any) -> bool:
    """Recognize two finite real JSON-compatible objective values in fixed order."""

    return isinstance(values, (list, tuple)) and len(values) == 2 and all(
        not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in values
    )


def _classifier_trials(study: Any) -> list[Any]:
    """Collect complete accuracy/noise pairs without reading Optuna's scalar value property."""

    trials = [] if study is None else study.get_trials(deepcopy=False)
    return [trial for trial in trials if trial.state.name == "COMPLETE" and _finite_pair(trial.values)]


def _pareto_trials(trials: list[Any]) -> list[Any]:
    """Return nondominated trials ordered by decreasing accuracy, increasing noise, then trial number."""

    return sorted([
        trial for trial in trials if not any(
            other.values[0] >= trial.values[0] and other.values[1] <= trial.values[1]
            and (other.values[0] > trial.values[0] or other.values[1] < trial.values[1])
            for other in trials
        )
    ], key=lambda trial: (-float(trial.values[0]), float(trial.values[1]), trial.number))


def _pareto_subset(trials: list[Any], top_k: int) -> list[Any]:
    """Select evenly spaced accuracy-ordered front positions, retaining both endpoints.

    With one slot, choose maximum accuracy. With two or more, keep accuracy and
    noise extremes plus evenly spaced tradeoffs. This is a display/coverage
    policy, not a weighted fitness or a claim that one tradeoff is universally best.
    """

    count = min(top_k, len(trials))
    # Small fronts retain every available distinct nondominated configuration.
    if count == len(trials):
        return trials
    # A single requested representative explicitly prioritizes maximum accuracy.
    if count == 1:
        return trials[:1]
    return [trials[round(index * (len(trials) - 1) / (count - 1))] for index in range(count)]


def _validate_confirmation_objective(plan: dict, result: dict) -> None:
    """Authenticate classifier objective order, directions, network and finite paired scores."""

    # Continual receipts authenticate scalar accuracy and the frozen task stream.
    if plan["hpo"].get("search_profile") == "dit_continual_runner":
        from common.dit_continual_runner import validate_confirmation_objective


        validate_confirmation_objective(plan, result)
        return
    # Legacy generation receipts retain their original identity schema.
    if not _classifier_plan(plan):
        return
    expected = {
        "objective_metrics": ["classification_accuracy", "noise_loss"], 
        "objective_directions": ["maximize", "minimize"], "objective_network": "raw"
    }
    # A partial, scalar or nonfinite result is not a completed Pareto confirmation.
    if not _finite_pair(result.get("objectives")):
        raise ValueError("Classifier confirmation requires two finite accuracy/noise objectives.")
    # EMA, swapped metrics and altered directions cannot be relabeled as raw objectives.
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Confirmation objective identity does not match the raw classifier Pareto recipe.")


def _ranked(study: Any, direction: str = "minimize") -> list[Any]:
    """Order finite completed scalar trials by their directed score then trial number."""

    trials = [] if study is None else study.get_trials(deepcopy=False)
    return sorted(
        [
            trial for trial in trials
            if trial.state.name == "COMPLETE" and trial.value is not None
            and math.isfinite(float(trial.value))
        ], 
        key=lambda trial: ((-1.0 if direction == "maximize" else 1.0) * float(trial.value), trial.number)
    )


def search_summary(plan: dict) -> dict:
    """Describe finite progress and objective-specific tradeoffs without training."""

    # Continual accuracy uses maximum-directed ranking and teacher coverage.
    if plan["hpo"].get("search_profile") == "dit_continual_runner":
        from common.dit_continual_runner import search_summary as continual_summary


        return continual_summary(plan)
    study = _load_study(plan)
    trials = [] if study is None else study.get_trials(deepcopy=False)
    classifier = _classifier_plan(plan)
    ranked = _classifier_trials(study) if classifier else _ranked(study)
    chronological = sorted(ranked, key=lambda trial: trial.number)
    states = {}
    branches = {}
    for trial in trials:
        states[trial.state.name] = states.get(trial.state.name, 0) + 1
    for trial in ranked:
        branch = trial.params.get("classifier_route", "plain") if classifier else trial.params.get(
            "dit_followup_branch", trial.params.get("dit_architecture_grid4", trial.params.get("dit_architecture_plain", "unknown"))
        )
        # UNet widths identify its sampled architecture family in notebook progress.
        if plan["hpo"]["model_name"] == "unet":
            branch = trial.params.get("widths", "unknown")
        branches[branch] = branches.get(branch, 0) + 1
    summary = {
        "allocated_trials": len(trials), "completed_finite_trials": len(ranked), 
        "states": states, "architecture_counts": branches
    }
    earlier = chronological[:-50]
    # Pareto progress has separate extrema and an explicit front, never a scalar best trial.
    if classifier:
        front = _pareto_trials(ranked)
        accuracy = max(float(trial.values[0]) for trial in ranked) if ranked else None
        noise = min(float(trial.values[1]) for trial in ranked) if ranked else None
        summary.update({
            "pareto_front_size": len(front), "pareto_trial_numbers": [trial.number for trial in front], 
            "pareto_front": [
                {"trial_number": trial.number, "accuracy": float(trial.values[0]), "noise_loss": float(trial.values[1])}
                for trial in front
            ], 
            "max_validation_accuracy": accuracy, "min_validation_noise_loss": noise, 
            "accuracy_improvement_last_50_valid_trials": accuracy - max(float(trial.values[0]) for trial in earlier) if earlier else None, 
            "noise_loss_improvement_last_50_valid_trials": min(float(trial.values[1]) for trial in earlier) - noise if earlier else None
        })
        return summary
    previous_best = min(float(trial.value) for trial in earlier) if earlier else None
    best = float(ranked[0].value) if ranked else None
    summary.update({
        "best_trial": ranked[0].number if ranked else None, 
        "best_validation_noise_loss": best, 
        "improvement_last_50_valid_trials": previous_best - best if earlier else None
    })
    return summary


def _launch(
    plan: dict, payload: dict, tag: str, deadline: float | None = None, gpu_id: int | None = None, 
    cancel_event: Any = None
) -> dict:
    """Run one admitted child and authenticate its receipt within the phase deadline."""

    from common.dit_hpo_remote import launch_worker


    # Cancellation must also prevent work from entering the admission queue.
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("DiT worker launch cancelled.")
    # Expired phases cannot queue or launch another worker.
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("DiT experiment phase deadline reached before worker admission.")
    worker_plan = plan
    # Confirmation workers each reserve one selected GPU with a serial memory budget.
    if payload["kind"] == "confirmation" and (gpu_id is not None or len(plan["identity"].get("gpus", [])) > 1):
        from common.dit_hpo_remote import serial_worker_identity


        options = {} if gpu_id is None else {"gpu_id": gpu_id}
        worker_plan = {
            **plan, "identity": serial_worker_identity(plan["checkout_root"], plan["identity"], **options)
        }
    jobs = Path(plan["control_root"]) / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    attempt = 1
    # Preserve requests and logs from failed or interrupted child processes.
    while (jobs / f"{tag}-{attempt:03d}.json").exists():
        attempt += 1
    request_path = jobs / f"{tag}-{attempt:03d}.json"
    receipt_path = request_path.with_suffix(".result.json")
    log_path = request_path.with_suffix(".log")
    request = {"plan": worker_plan, "payload": payload, "receipt_path": str(receipt_path)}
    options = {}
    # Persist the same absolute deadline seen by admission and the child.
    if deadline is not None:
        request["deadline"] = deadline
        options["deadline"] = deadline
    # Admission polls this shared event before entering a worker slot.
    if cancel_event is not None:
        options["cancel_event"] = cancel_event
    _write(request_path, request)
    command = [sys.executable, "-m", "common.dit_hpo_runner", "--worker", str(request_path)]
    print(f"Worker log: {log_path}", flush=True)
    try:
        with launch_worker(command, worker_plan["checkout_root"], worker_plan["identity"], log_path, **options) as process:
            # A thread-owned confirmation must notice cancellation from the notebook thread.
            if cancel_event is not None:
                while True:
                    # Notebook interruption cancels each thread-owned child promptly.
                    if cancel_event.is_set():
                        raise InterruptedError("DiT worker execution cancelled.")
                    remaining = None if deadline is None else deadline - time.time()
                    # Expired confirmations leave their context to reap the owned process.
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError("DiT experiment phase deadline reached during worker execution.")
                    try:
                        returncode = process.wait(timeout=0.2 if remaining is None else min(0.2, remaining))
                        break
                    except TimeoutExpired:
                        continue
            # Untimed legacy launches preserve their original blocking wait contract.
            elif deadline is None:
                returncode = process.wait()
            # Timed searches have one outer watchdog in addition to the HPO deadline.
            else:
                returncode = process.wait(timeout=max(0.0, deadline - time.time()))
    except TimeoutExpired as error:
        raise TimeoutError("DiT experiment phase deadline reached during worker execution.") from error
    # Failed workers cannot publish a successful-looking partial result.
    if returncode != 0 or not receipt_path.exists():
        raise RuntimeError(f"Worker failed (exit {returncode}); inspect {log_path}.")
    receipt = _read(receipt_path)
    # Receipts must authenticate the precise request consumed by the child.
    if receipt.get("request_sha256") != _digest(request_path):
        raise ValueError("Worker receipt does not match its request.")
    return receipt["result"]


def run_search(plan: dict, target_completed: int = 200, max_attempts: int = 400, batch_trials: int = 10) -> dict:
    """Reach a valid-trial target through bounded calls to the public HPO API.

    Args:
        plan: Fixed recipe returned by make_plan.
        target_completed: Desired positive count of finite COMPLETE trials.
        max_attempts: Total allocated-trial ceiling, including failures.
        batch_trials: Maximum new trial allocations per child process.

    Returns:
        dict: Search summary, target_reached and time_budget_exhausted flags.

    Raises:
        ValueError: Exact counts or coupled budget limits are incompatible.
        RuntimeError: A child fails or the study makes no progress.
    """

    counts = (target_completed, max_attempts, batch_trials)
    # Counts and the coupled attempt ceiling define the authorized experiment.
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts) \
    or target_completed < 1 or max_attempts < target_completed or batch_trials < 1:
        raise ValueError("Use integer budgets: target >= 1, attempts >= target, batch >= 1.")
    with _coordinator(plan):
        timed_out = False
        # Selection must remain fixed once confirmation begins.
        if (Path(plan["control_root"]) / "finalists.json").exists():
            deadline = _phase_deadline(plan, "search")
            status_path = Path(plan["control_root"]) / "search_status.json"
            status = _read(status_path) if status_path.exists() else {}
            timed_out = deadline is not None and (
                time.time() >= deadline - 60.0 or status.get("time_budget_exhausted", False)
            )
            # Expired campaigns permit status-only replay so confirmation cells remain reachable.
            if search_summary(plan)["completed_finite_trials"] < target_completed and not timed_out:
                raise ValueError("Finalists are frozen; use a new experiment for further search.")
        deadline = _phase_deadline(plan, "search", start=True)
        while True:
            summary = search_summary(plan)
            complete = summary["completed_finite_trials"]
            allocated = summary["allocated_trials"]
            # Re-running a completed stage must allocate no new trials.
            exhausted = timed_out or (deadline is not None and time.time() >= deadline)
            # Count ceilings and phase expiry are normal terminal search states.
            if complete >= target_completed or allocated >= max_attempts or exhausted:
                summary["target_completed"] = target_completed
                summary["target_reached"] = complete >= target_completed
                summary["time_budget_exhausted"] = exhausted
                summary["time_budget"] = budget_summary(plan)
                _write(Path(plan["control_root"]) / "search_status.json", summary)
                print(json.dumps(summary, indent=2), flush=True)
                return summary
            allowance = min(max_attempts, allocated + min(batch_trials, target_completed - complete))
            options = {} if deadline is None else {"deadline": deadline}
            try:
                result = _launch(plan, {"kind": "search", "allocated_target": allowance}, f"search-{allowance:04d}", **options)
            except TimeoutError:
                # Only a configured clock may turn admission or process expiry into a normal stop.
                if deadline is None:
                    raise
                timed_out = True
                continue
            # The inner HPO deadline can finish during the reserved cleanup interval.
            if result.get("time_budget_exhausted", False):
                timed_out = True
                continue
            updated = search_summary(plan)
            # Recovery that makes no progress must not cause an infinite loop.
            if updated == summary:
                raise RuntimeError("HPO made no observable trial-state progress; inspect its logs.")
            print(json.dumps(updated, indent=2), flush=True)


def freeze_finalists(plan: dict, seeds: Iterable[int], top_k: int = 3) -> dict:
    """Freeze distinct scalar winners or representative Pareto configurations.

    Running trials always block initial selection. Queued suggestions may remain
    unstarted after a persisted search deadline; retain those records unchanged
    and exclude them from selection once their execution window has elapsed.

    Args:
        plan: Fixed study recipe.
        seeds: Distinct integer training seeds excluding the original search seed.
        top_k: Exact generation finalist count; maximum classifier finalist count.
            Classifier fronts use evenly spaced decreasing-accuracy positions,
            keeping both extremes when at least two slots are requested. One
            slot chooses maximum accuracy. Display order is not a fitness rank.

    Returns:
        dict: Immutable finalist manifest reused on notebook restart.

    Raises:
        ValueError: Seed identity, candidate availability or frozen inputs differ.
    """

    # Continual finalists preserve their own scalar objective and stream identity.
    if plan["hpo"].get("search_profile") == "dit_continual_runner":
        from common.dit_continual_runner import freeze_finalists as continual_finalists


        return continual_finalists(plan, seeds, top_k=top_k)
    classifier = _classifier_plan(plan)
    seeds = list(seeds)
    # Fresh, unique seed identities are part of the confirmation design.
    if not seeds or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds) \
    or len(set(seeds)) != len(seeds) or plan["hpo"]["seed"] in seeds:
        raise ValueError("Use distinct integer confirmation seeds different from the search seed.")
    # Candidate count must identify an exact nonempty selection.
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer candidate count.")
    policy = {
        "method": "pareto_accuracy_ordered_even_spacing", 
        "single_slot": "maximum_accuracy", "keep_extremes": top_k >= 2, 
        "objective_metrics": ["classification_accuracy", "noise_loss"], 
        "objective_directions": ["maximize", "minimize"], "order_is_fitness_rank": False
    }
    with _coordinator(plan):
        path = Path(plan["control_root"]) / "finalists.json"
        # The first selection remains authoritative across later cell executions.
        if path.exists():
            manifest = _read(path)
            # Reusing a freeze requires the identical paired design.
            if manifest["seeds"] != seeds or manifest["top_k"] != top_k:
                raise ValueError("Finalists were frozen with different seeds or candidate count.")
            # Old scalar selections cannot become a representative Pareto freeze.
            if classifier and manifest.get("selection_policy") != policy:
                raise ValueError("Frozen classifier selection does not match the Pareto policy.")
            for candidate in manifest["candidates"]:
                # Reject edits to an already frozen candidate.
                if _digest(candidate["input_config_path"]) != candidate["config_sha256"]:
                    raise ValueError("A frozen finalist configuration has changed.")
            return manifest
        study = _load_study(plan)
        ranked = _pareto_trials(_classifier_trials(study)) if classifier else _ranked(study)
        distinct = []
        configurations = set()
        for trial in ranked:
            configuration = json.dumps(trial.params, sort_keys=True)
            # Duplicate sampled settings do not merit another finalist slot.
            if configuration in configurations:
                continue
            configurations.add(configuration)
            distinct.append(trial)
        # Pareto confirmation needs one candidate; scalar selection retains its exact count.
        if not distinct or (not classifier and len(distinct) < top_k):
            raise ValueError("Not enough distinct finite completed configurations to select finalists.")
        trials = study.get_trials(deepcopy=False)
        # A still-running trial can publish an outcome even after its time limit.
        if any(trial.state.name == "RUNNING" for trial in trials):
            raise ValueError("Finish or recover pending trials before freezing finalists.")
        waiting = [trial.number for trial in trials if trial.state.name == "WAITING"]
        search_execution_deadline = None
        # Unstarted hints are preserved when the durable clock forbids new work.
        if waiting:
            deadline = _phase_deadline(plan, "search")
            search_execution_deadline = None if deadline is None else deadline - 60.0
            # Match run_search's finalist replay cutoff, including child cleanup.
            if search_execution_deadline is None or time.time() < search_execution_deadline:
                raise ValueError("Finish or recover pending trials before freezing finalists.")
        selected = _pareto_subset(distinct, top_k) if classifier else distinct[:top_k]
        candidates = []
        for rank, trial in enumerate(selected, start=1):
            original = Path(plan["study_root"]) / "configs" / f"trial-{trial.number:04d}.yaml"
            frozen = Path(plan["control_root"]) / "finalist_configs" / original.name
            frozen.parent.mkdir(parents=True, exist_ok=True)
            frozen.write_bytes(original.read_bytes())
            scores = {
                "display_order": rank, "search_accuracy": float(trial.values[0]), 
                "search_noise_loss": float(trial.values[1])
            } if classifier else {"rank": rank, "search_noise_loss": float(trial.value)}
            candidates.append({
                **scores, "trial_number": trial.number, "params": dict(trial.params), 
                "input_config_path": str(frozen), "config_sha256": _digest(frozen)
            })
        manifest = {
            "study_name": plan["study_name"], 
            "study_allocated_trials": len(study.get_trials(deepcopy=False)), 
            "top_k": top_k, "seeds": seeds, "dataset_seed": plan["hpo"]["seed"], 
            "candidates": candidates
        }
        # Record excluded suggestions without cancelling them or inventing scores.
        if waiting:
            manifest["unstarted_trials_at_search_deadline"] = {
                "trial_numbers": waiting, "search_execution_deadline_unix": search_execution_deadline
            }
        # Explicit Pareto provenance distinguishes maximum capacity from actual front size.
        if classifier:
            manifest.update({
                "selection_policy": policy, "distinct_pareto_configurations": len(distinct), 
                "selected_candidates": len(candidates)
            })
        _write(path, manifest)
        return manifest


def _completed_record(path: str | Path, expected: dict, plan: dict | None = None) -> dict:
    """Authenticate a saved confirmation identity and its finite objective contract."""

    record = _read(path)
    # Completed results belong to exactly one frozen candidate and seed.
    if record["identity"] != expected:
        raise ValueError("A completed confirmation belongs to different inputs.")
    # Pareto classifier receipts require both named finite raw objectives.
    if plan is not None and (
        _classifier_plan(plan) or plan["hpo"].get("search_profile") == "dit_continual_runner"
    ):
        _validate_confirmation_objective(plan, record["result"])
    # Legacy scalar generation retains its established validation behavior.
    elif not math.isfinite(float(record["result"]["objective"])):
        raise ValueError("A completed confirmation has a nonfinite objective.")
    return record


def run_confirmations(plan: dict) -> list[dict]:
    """Run frozen finalist/seed pairs with one admitted worker per selected GPU.

    Authenticate completed receipts before skipping them. Fresh attempts retain
    their source configuration and paired seeds. The persistent experiment
    deadline stops new launches and cancels admitted or waiting workers; only
    finite completed results receive completion receipts. Legacy plans without
    an explicit GPU inventory retain synchronous confirmation execution.

    Args:
        plan: Study recipe with a previously frozen finalist manifest.

    Returns:
        list[dict]: Authenticated successful receipts in finalist/seed order.
            A reached deadline returns the completed subset for safe resumption.

    Raises:
        ValueError: Frozen inputs, receipt identities or finite scores changed.
        RuntimeError: A confirmation child process failed unexpectedly.
        KeyboardInterrupt: Cancellation closes active admitted worker contexts.
    """

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    from threading import Event
    import time


    with _coordinator(plan):
        manifest_path = Path(plan["control_root"]) / "finalists.json"
        manifest = _read(manifest_path)
        manifest_digest = _digest(manifest_path)
        records = {}
        jobs = []
        index = 0
        for candidate in manifest["candidates"]:
            # Authenticate every frozen candidate before launching any work.
            if _digest(candidate["input_config_path"]) != candidate["config_sha256"]:
                raise ValueError("A frozen finalist configuration has changed.")
            for training_seed in manifest["seeds"]:
                destination = Path(plan["control_root"]) / "confirmations" / f"trial-{candidate['trial_number']:04d}" / f"seed-{training_seed}"
                completed_path = destination / "completed.json"
                expected = {
                    "manifest_sha256": manifest_digest, 
                    "config_sha256": candidate["config_sha256"], 
                    "trial_number": candidate["trial_number"], 
                    "training_seed": training_seed
                }
                # Completed paired repeats retain their exact authenticated identity.
                if completed_path.exists():
                    records[index] = _completed_record(completed_path, expected, plan=plan)
                # Only missing successes need another independently preserved attempt.
                else:
                    jobs.append({
                        "index": index, "candidate": candidate, "seed": training_seed, 
                        "destination": destination, "completed_path": completed_path, 
                        "expected": expected
                    })
                index += 1
        deadline = _phase_deadline(plan, "confirmation", start=bool(jobs))
        gpu_ids = [gpu["gpu_id"] for gpu in plan["identity"].get("gpus", [])] or [None]
        exhausted = False
        cancel_event = Event()

        def request(job: dict) -> tuple[dict, str]:
            """Allocate the next preserved attempt for one missing paired repeat."""

            destination = job["destination"]
            destination.mkdir(parents=True, exist_ok=True)
            attempt = 1
            while (destination / f"attempt-{attempt:03d}").exists():
                attempt += 1
            candidate = job["candidate"]
            payload = {
                "kind": "confirmation", 
                "input_config_path": candidate["input_config_path"], 
                "output_root": str(destination / f"attempt-{attempt:03d}"), 
                "training_seed": job["seed"], 
                "expected_config_sha256": candidate["config_sha256"]
            }
            return payload, f"confirm-{candidate['trial_number']:04d}-{job['seed']}"

        def publish(job: dict, result: dict) -> None:
            """Persist a finite completed result in the sole coordinator thread."""

            _validate_confirmation_objective(plan, result)
            # Scalar generation retains its finite-score guard; classifier validates the full pair above.
            if not _classifier_plan(plan) and not math.isfinite(float(result["objective"])):
                raise ValueError("Confirmation objective is not finite.")
            record = {"identity": job["expected"], "result": result}
            _write(job["completed_path"], record)
            records[job["index"]] = record

        # Preserve synchronous legacy callers and their ordinary interrupt behavior.
        if gpu_ids == [None]:
            for job in jobs:
                # An expired absolute budget must never allocate another attempt.
                if deadline is not None and time.time() >= deadline:
                    exhausted = True
                    break
                payload, tag = request(job)
                options = {} if deadline is None else {"deadline": deadline}
                try:
                    publish(job, _launch(plan, payload, tag, **options))
                except TimeoutError:
                    exhausted = True
                    break
        # Explicit devices receive independent workers while each keeps one slot.
        else:
            active = {}
            free_gpus = list(gpu_ids)
            next_job = 0
            failure = None
            with ThreadPoolExecutor(max_workers=len(gpu_ids), thread_name_prefix="dit-confirmation") as executor:
                try:
                    while next_job < len(jobs) or active:
                        # Time remaining is shared across every GPU and paired seed.
                        if deadline is not None and time.time() >= deadline:
                            exhausted = True
                            cancel_event.set()
                        while free_gpus and next_job < len(jobs) and not exhausted and failure is None:
                            job = jobs[next_job]
                            next_job += 1
                            gpu_id = free_gpus.pop(0)
                            payload, tag = request(job)
                            future = executor.submit(
                                _launch, plan, payload, tag, deadline=deadline, 
                                gpu_id=gpu_id, cancel_event=cancel_event
                            )
                            active[future] = (job, gpu_id)
                        # No active work remains after completion, timeout or failure.
                        if not active:
                            break
                        finished, _ = wait(active, timeout=0.2, return_when=FIRST_COMPLETED)
                        for future in finished:
                            job, gpu_id = active.pop(future)
                            free_gpus.append(gpu_id)
                            try:
                                publish(job, future.result())
                            except TimeoutError:
                                exhausted = True
                                cancel_event.set()
                            except Exception as error:
                                # Keep the first real failure while cancelled siblings drain.
                                if failure is None and not exhausted:
                                    failure = error
                                cancel_event.set()
                except BaseException:
                    cancel_event.set()
                    raise
            # Successful siblings are already durable before an error is surfaced.
            if failure is not None:
                raise failure
        _write(Path(plan["control_root"]) / "confirmation_status.json", {
            "completed_pairs": len(records), "required_pairs": index, 
            "all_pairs_complete": len(records) == index, 
            "time_budget_exhausted": exhausted, "deadline": deadline, 
            "gpu_ids": gpu_ids
        })
        return [records[key] for key in sorted(records)]


def confirmation_summary(plan: dict) -> list[dict]:
    """Summarize authenticated paired repeats without scalarizing classifier tradeoffs."""

    # Continual scores retain accuracy units and paired seeds.
    if plan["hpo"].get("search_profile") == "dit_continual_runner":
        from common.dit_continual_runner import confirmation_summary as continual_summary


        return continual_summary(plan)

    import statistics


    manifest_path = Path(plan["control_root"]) / "finalists.json"
    manifest = _read(manifest_path)
    manifest_digest = _digest(manifest_path)
    classifier = _classifier_plan(plan)
    rows = []
    for candidate in manifest["candidates"]:
        losses = []
        accuracies = []
        for training_seed in manifest["seeds"]:
            path = Path(plan["control_root"]) / "confirmations" / f"trial-{candidate['trial_number']:04d}" / f"seed-{training_seed}" / "completed.json"
            # Summaries may report partial confirmation progress.
            if path.exists():
                expected = {
                    "manifest_sha256": manifest_digest, 
                    "config_sha256": candidate["config_sha256"], 
                    "trial_number": candidate["trial_number"], "training_seed": training_seed
                }
                result = _completed_record(path, expected, plan=plan)["result"]
                # Every completed classifier seed contributes both objectives together.
                if classifier:
                    accuracies.append(float(result["objectives"][0]))
                    losses.append(float(result["objectives"][1]))
                # Generation continues to expose its original scalar loss summary.
                else:
                    losses.append(float(result["objective"]))
        row = {
            "trial_number": candidate["trial_number"], "search_noise_loss": candidate["search_noise_loss"], 
            "completed_seeds": len(losses), "required_seeds": len(manifest["seeds"]), 
            "mean_noise_loss": statistics.mean(losses) if losses else None, 
            "std_noise_loss": statistics.stdev(losses) if len(losses) > 1 else None, 
            "all_seeds_complete": len(losses) == len(manifest["seeds"])
        }
        # Both paired means remain visible; incomplete seed sets are not comparable outcomes.
        if classifier:
            row.update({
                "search_accuracy": candidate["search_accuracy"], 
                "mean_accuracy": statistics.mean(accuracies) if accuracies else None, 
                "std_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else None, 
                "comparable": row["all_seeds_complete"]
            })
        rows.append(row)
    _write(Path(plan["control_root"]) / "confirmation_summary.json", rows)
    return rows


def _worker(request_path: str | Path) -> None:
    """Execute one gated child using existing public HPO/training APIs."""

    from common.dit_hpo_remote import managed_parallel_coordinator, managed_worker


    request = _read(request_path)
    plan = request["plan"]
    payload = request["payload"]
    deadline = request.get("deadline")
    routed = plan["hpo"].get("worker_gpu_ids") is not None
    parallel = payload["kind"] == "search" and (
        plan["hpo"]["concurrent_trials"] > 1 or routed or plan["hpo"].get("pruning") is not None
    )
    manager = managed_parallel_coordinator if parallel else managed_worker
    with manager(plan["checkout_root"], plan["identity"]) as worker_context:
        # Search uses the existing public HPO engine.
        if payload["kind"] == "search":
            from common.hpo import run_hpo


            arguments = dict(plan["hpo"])
            arguments["n_trials"] = payload["allocated_target"]
            # The HPO coordinator reaps active trials before the outer process cutoff.
            if deadline is not None:
                arguments["timeout"] = max(0.0, deadline - time.time() - 60.0)
                arguments["stop_active_on_timeout"] = True
            # The existing process scheduler owns asks/tells and trial completion.
            if parallel:
                context_key = "gpu_worker_context" if routed else "worker_context"
                arguments[context_key] = worker_context
            # Explicit recovery retains the HPO sampler/checkpoint protocol.
            if database_path(plan["study_root"]).exists():
                arguments["resume_from"] = plan["study_root"]
            run_hpo(**arguments)
            result = search_summary(plan)
            # Propagate the inner cutoff even if cleanup finished before the outer deadline.
            if deadline is not None:
                result["time_budget_exhausted"] = time.time() >= deadline - 60.0
        # Confirmation reuses each selected recipe with paired fresh seeds.
        elif payload["kind"] == "confirmation":
            # Continual replay uses the existing complete sequential training API.
            if plan["hpo"].get("search_profile") == "dit_continual_runner":
                from common.dit_continual_runner import run_confirmation


            # Generation and joint repeats retain their established worker.
            else:
                from common.dit_hpo_confirmation import run_confirmation


            arguments = {key: value for key, value in payload.items() if key != "kind"}
            result = run_confirmation(**arguments)
        # Unknown actions cannot fall through to a successful receipt.
        else:
            raise ValueError("Unknown DiT notebook worker action.")
        _write(request["receipt_path"], {"request_sha256": _digest(request_path), "result": result})


def main() -> None:
    """Run the internal child entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    arguments = parser.parse_args()
    _worker(arguments.worker)


# The module entry point is reserved for admitted child requests.
if __name__ == "__main__":
    main()
