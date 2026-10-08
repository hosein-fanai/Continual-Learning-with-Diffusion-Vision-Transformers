"""Resumable notebook orchestration for ordinary DiT generation HPO.

The notebook reads study metadata. Admitted child processes use the existing
HPO and training APIs. Targets count finite COMPLETE trials; attempt ceilings
include every allocated trial, including failures and interruptions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


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
    seed: int = 42
) -> dict:
    """Seal the scientific recipe while leaving trial targets adjustable.

    Args:
        checkout_root: Verified remote project directory.
        results_path: Experiment root, absolute or relative to the checkout.
        dataset_name: One dataset per independent study.
        epochs: Maximum epochs for every search trial.
        n_startup_trials: Random observations before adaptive sampling.
        seed: Shared search, initialization and dataset-split seed.

    Returns:
        dict: Paths, public HPO arguments and the remote worker identity.

    Raises:
        RuntimeError: The runtime is not an authorized remote host.
        ValueError: Existing runner provenance differs from this recipe.
    """

    from common.dit_hpo_remote import inspect_remote


    root = Path(checkout_root).resolve()
    identity = inspect_remote(root)
    results = Path(results_path)
    results = (root / results).resolve() if not results.is_absolute() else results.resolve()
    study_root = results / "generation" / "diffusion_transformer" / dataset_name.lower()
    plan = {
        "version": 1, 
        "checkout_root": str(root), 
        "study_root": str(study_root), 
        "control_root": str(study_root / "notebook_runner"), 
        "study_name": "generation-diffusion_transformer-" + dataset_name.lower(), 
        "hpo": {
            "task": "generation", 
            "model_name": "diffusion_transformer", 
            "dataset_name": dataset_name, 
            "epochs": epochs, 
            "results_path": str(results), 
            "fit_method": "fit", 
            "objective_metrics": ["generation_loss"], 
            "objective_directions": ["minimize"], 
            "dtype_policy": "float32", 
            "n_startup_trials": n_startup_trials, 
            "trial_budget_mode": "total", 
            "validation_source": "split", 
            "validation_ratio": 0.2, 
            "concurrent_trials": 1, 
            "seed": seed
        }, 
        "identity": identity
    }
    scientific = {
        "version": plan["version"], 
        "hpo": plan["hpo"], 
        "source_sha256": identity["source_sha256"], 
        "versions": identity["versions"], 
        "python": identity["python"]
    }
    with _coordinator(plan):
        path = Path(plan["control_root"]) / "recipe.json"
        # Existing recipes cannot silently change on notebook restart.
        if path.exists() and _read(path) != scientific:
            raise ValueError("Runner recipe changed; use a fresh RESULTS_PATH.")
        _write(path, scientific)
    return plan


def _load_study(plan: dict) -> Any:
    """Read the existing Optuna study without creating its database."""

    import optuna


    database = Path(plan["study_root"]) / "study.db"
    # An unstarted experiment has no allocated trials.
    if not database.exists():
        return None
    return optuna.load_study(
        study_name=plan["study_name"], 
        storage="sqlite:///" + database.resolve().as_posix()
    )


def _ranked(study: Any) -> list[Any]:
    """Order finite completed scalar trials by loss then trial number."""

    trials = [] if study is None else study.get_trials(deepcopy=False)
    return sorted(
        [
            trial for trial in trials
            if trial.state.name == "COMPLETE" and trial.value is not None
            and math.isfinite(float(trial.value))
        ], 
        key=lambda trial: (float(trial.value), trial.number)
    )


def search_summary(plan: dict) -> dict:
    """Describe current progress without starting training.

    Args:
        plan: The dictionary returned by make_plan.

    Returns:
        dict: Attempt/state counts, valid architecture counts, best loss and
        improvement over the last 50 valid trials. This is descriptive evidence,
        not a statistical stopping test.
    """

    study = _load_study(plan)
    trials = [] if study is None else study.get_trials(deepcopy=False)
    ranked = _ranked(study)
    chronological = sorted(ranked, key=lambda trial: trial.number)
    states = {}
    branches = {}
    for trial in trials:
        states[trial.state.name] = states.get(trial.state.name, 0) + 1
    for trial in ranked:
        branch = trial.params.get("dit_architecture_grid4", trial.params.get("dit_architecture_plain", "unknown"))
        branches[branch] = branches.get(branch, 0) + 1
    earlier = chronological[:-50]
    previous_best = min(float(trial.value) for trial in earlier) if earlier else None
    best = float(ranked[0].value) if ranked else None
    return {
        "allocated_trials": len(trials), 
        "completed_finite_trials": len(ranked), 
        "states": states, 
        "architecture_counts": branches, 
        "best_trial": ranked[0].number if ranked else None, 
        "best_validation_noise_loss": best, 
        "improvement_last_50_valid_trials": previous_best - best if earlier else None
    }


def _launch(plan: dict, payload: dict, tag: str) -> dict:
    """Run one admitted child and authenticate its completion receipt."""

    from common.dit_hpo_remote import launch_worker


    jobs = Path(plan["control_root"]) / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    attempt = 1
    # Preserve requests and logs from failed or interrupted child processes.
    while (jobs / f"{tag}-{attempt:03d}.json").exists():
        attempt += 1
    request_path = jobs / f"{tag}-{attempt:03d}.json"
    receipt_path = request_path.with_suffix(".result.json")
    log_path = request_path.with_suffix(".log")
    _write(request_path, {"plan": plan, "payload": payload, "receipt_path": str(receipt_path)})
    command = [sys.executable, "-m", "common.dit_hpo_runner", "--worker", str(request_path)]
    print(f"Worker log: {log_path}", flush=True)
    with launch_worker(command, plan["checkout_root"], plan["identity"], log_path) as process:
        returncode = process.wait()
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
        target_completed: Desired finite COMPLETE trials, at most 300.
        max_attempts: Total allocated-trial ceiling, including failures.
        batch_trials: Maximum new trial allocations per child process.

    Returns:
        dict: Search summary and target_reached flag.

    Raises:
        ValueError: Exact counts or coupled budget limits are incompatible.
        RuntimeError: A child fails or the study makes no progress.
    """

    counts = (target_completed, max_attempts, batch_trials)
    # Counts and the coupled attempt ceiling define the authorized experiment.
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts) \
    or not 1 <= target_completed <= 300 or max_attempts < target_completed or batch_trials < 1:
        raise ValueError("Use integer budgets: 1 <= target <= 300, attempts >= target, batch >= 1.")
    with _coordinator(plan):
        # Selection must remain fixed once confirmation begins.
        if (Path(plan["control_root"]) / "finalists.json").exists():
            # An existing freeze permits status reads, but no additional search.
            if search_summary(plan)["completed_finite_trials"] < target_completed:
                raise ValueError("Finalists are frozen; use a new experiment for further search.")
        while True:
            summary = search_summary(plan)
            complete = summary["completed_finite_trials"]
            allocated = summary["allocated_trials"]
            # Re-running a completed stage must allocate no new trials.
            if complete >= target_completed or allocated >= max_attempts:
                summary["target_completed"] = target_completed
                summary["target_reached"] = complete >= target_completed
                _write(Path(plan["control_root"]) / "search_status.json", summary)
                print(json.dumps(summary, indent=2), flush=True)
                return summary
            allowance = min(max_attempts, allocated + min(batch_trials, target_completed - complete))
            _launch(plan, {"kind": "search", "allocated_target": allowance}, f"search-{allowance:04d}")
            updated = search_summary(plan)
            # Recovery that makes no progress must not cause an infinite loop.
            if updated == summary:
                raise RuntimeError("HPO made no observable trial-state progress; inspect its logs.")
            print(json.dumps(updated, indent=2), flush=True)


def freeze_finalists(plan: dict, seeds: Iterable[int], top_k: int = 3) -> dict:
    """Freeze input configurations and paired fresh seeds for confirmation.

    Args:
        plan: Fixed study recipe.
        seeds: Distinct integer training seeds excluding the original search seed.
        top_k: Number of finite completed candidates selected by validation loss.

    Returns:
        dict: Immutable finalist manifest reused on notebook restart.

    Raises:
        ValueError: Seed identity, candidate count or frozen inputs differ.
    """

    seeds = list(seeds)
    # Fresh, unique seed identities are part of the confirmation design.
    if not seeds or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds) \
    or len(set(seeds)) != len(seeds) or plan["hpo"]["seed"] in seeds:
        raise ValueError("Use distinct integer confirmation seeds different from the search seed.")
    # Candidate count must identify an exact nonempty selection.
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer candidate count.")
    with _coordinator(plan):
        path = Path(plan["control_root"]) / "finalists.json"
        # The first selection remains authoritative across later cell executions.
        if path.exists():
            manifest = _read(path)
            # Reusing a freeze requires the identical paired design.
            if manifest["seeds"] != seeds or manifest["top_k"] != top_k:
                raise ValueError("Finalists were frozen with different seeds or candidate count.")
            for candidate in manifest["candidates"]:
                # Reject edits to an already frozen candidate.
                if _digest(candidate["input_config_path"]) != candidate["config_sha256"]:
                    raise ValueError("A frozen finalist configuration has changed.")
            return manifest
        study = _load_study(plan)
        ranked = _ranked(study)
        distinct = []
        configurations = set()
        for trial in ranked:
            configuration = json.dumps(trial.params, sort_keys=True)
            # Duplicate sampled settings do not merit another finalist slot.
            if configuration in configurations:
                continue
            configurations.add(configuration)
            distinct.append(trial)
        # Selection requires enough successful, distinct sampled configurations.
        if len(distinct) < top_k:
            raise ValueError("Not enough distinct finite completed configurations to select finalists.")
        # Pending work can still alter the ranking and must finish first.
        if any(trial.state.name in ("RUNNING", "WAITING") for trial in study.get_trials(deepcopy=False)):
            raise ValueError("Finish or recover pending trials before freezing finalists.")
        candidates = []
        for rank, trial in enumerate(distinct[:top_k], start=1):
            original = Path(plan["study_root"]) / "configs" / f"trial-{trial.number:04d}.yaml"
            frozen = Path(plan["control_root"]) / "finalist_configs" / original.name
            frozen.parent.mkdir(parents=True, exist_ok=True)
            frozen.write_bytes(original.read_bytes())
            candidates.append({
                "rank": rank, 
                "trial_number": trial.number, 
                "search_noise_loss": float(trial.value), 
                "params": dict(trial.params), 
                "input_config_path": str(frozen), 
                "config_sha256": _digest(frozen)
            })
        manifest = {
            "study_name": plan["study_name"], 
            "study_allocated_trials": len(study.get_trials(deepcopy=False)), 
            "top_k": top_k, 
            "seeds": seeds, 
            "dataset_seed": plan["hpo"]["seed"], 
            "candidates": candidates
        }
        _write(path, manifest)
        return manifest


def _completed_record(path: str | Path, expected: dict) -> dict:
    """Authenticate a saved confirmation identity and finite objective."""

    record = _read(path)
    # Completed results belong to exactly one frozen candidate and seed.
    if record["identity"] != expected:
        raise ValueError("A completed confirmation belongs to different inputs.")
    # Nonfinite outcomes are not scientific confirmation successes.
    if not math.isfinite(float(record["result"]["objective"])):
        raise ValueError("A completed confirmation has a nonfinite objective.")
    return record


def run_confirmations(plan: dict) -> list[dict]:
    """Run each frozen finalist/seed once and retain interrupted attempts.

    Args:
        plan: Study recipe with a previously frozen finalist manifest.

    Returns:
        list[dict]: Successful receipts for all selected candidates and seeds.

    Raises:
        ValueError: Frozen inputs or completed receipt identities changed.
        RuntimeError: A confirmation child process failed.
    """

    with _coordinator(plan):
        manifest_path = Path(plan["control_root"]) / "finalists.json"
        manifest = _read(manifest_path)
        manifest_digest = _digest(manifest_path)
        records = []
        for candidate in manifest["candidates"]:
            # Authenticate the candidate before loading or launching results.
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
                # Re-running all notebook cells never repeats a successful confirmation.
                if completed_path.exists():
                    record = _completed_record(completed_path, expected)
                # Missing success receipts require a fresh attempt directory.
                else:
                    destination.mkdir(parents=True, exist_ok=True)
                    attempt = 1
                    while (destination / f"attempt-{attempt:03d}").exists():
                        attempt += 1
                    output = destination / f"attempt-{attempt:03d}"
                    payload = {
                        "kind": "confirmation", 
                        "input_config_path": candidate["input_config_path"], 
                        "output_root": str(output), 
                        "training_seed": training_seed, 
                        "expected_config_sha256": candidate["config_sha256"]
                    }
                    result = _launch(plan, payload, f"confirm-{candidate['trial_number']:04d}-{training_seed}")
                    # A failed validation score cannot become a completed result.
                    if not math.isfinite(float(result["objective"])):
                        raise ValueError("Confirmation objective is not finite.")
                    record = {"identity": expected, "result": result}
                    _write(completed_path, record)
                records.append(record)
        return records


def confirmation_summary(plan: dict) -> list[dict]:
    """Summarize authenticated paired repeats separately from search trials."""

    import statistics


    manifest_path = Path(plan["control_root"]) / "finalists.json"
    manifest = _read(manifest_path)
    manifest_digest = _digest(manifest_path)
    rows = []
    for candidate in manifest["candidates"]:
        losses = []
        for training_seed in manifest["seeds"]:
            path = Path(plan["control_root"]) / "confirmations" / f"trial-{candidate['trial_number']:04d}" / f"seed-{training_seed}" / "completed.json"
            # Summaries may report partial confirmation progress.
            if path.exists():
                expected = {
                    "manifest_sha256": manifest_digest, 
                    "config_sha256": candidate["config_sha256"], 
                    "trial_number": candidate["trial_number"], 
                    "training_seed": training_seed
                }
                losses.append(float(_completed_record(path, expected)["result"]["objective"]))
        rows.append({
            "trial_number": candidate["trial_number"], 
            "search_noise_loss": candidate["search_noise_loss"], 
            "completed_seeds": len(losses), 
            "required_seeds": len(manifest["seeds"]), 
            "mean_noise_loss": statistics.mean(losses) if losses else None, 
            "std_noise_loss": statistics.stdev(losses) if len(losses) > 1 else None, 
            "all_seeds_complete": len(losses) == len(manifest["seeds"])
        })
    _write(Path(plan["control_root"]) / "confirmation_summary.json", rows)
    return rows


def _worker(request_path: str | Path) -> None:
    """Execute one gated child using existing public HPO/training APIs."""

    from common.dit_hpo_remote import managed_worker


    request = _read(request_path)
    plan = request["plan"]
    payload = request["payload"]
    with managed_worker(plan["checkout_root"], plan["identity"]):
        # Search uses the existing public HPO engine.
        if payload["kind"] == "search":
            from common.hpo import run_hpo


            arguments = dict(plan["hpo"])
            arguments["n_trials"] = payload["allocated_target"]
            # Explicit recovery retains the HPO sampler/checkpoint protocol.
            if (Path(plan["study_root"]) / "study.db").exists():
                arguments["resume_from"] = plan["study_root"]
            run_hpo(**arguments)
            result = search_summary(plan)
        # Confirmation reuses each selected recipe with paired fresh seeds.
        elif payload["kind"] == "confirmation":
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
