"""Explicit, backed-up reconciliation of stopped DiT search allocations.

This coordinator-only API never trains, imports TensorFlow, extends a clock,
retries a trial, or interprets partial artifacts as completed objectives.
"""

from __future__ import annotations

from collections import Counter
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import time
from typing import Any
import uuid

from common import dit_hpo_runner as runner
from common.dit_hpo_transfer import _fingerprint
from common.hpo_process import study_lock


def _process_snapshot(proc_root: Path = Path("/proc")) -> dict[int, str]:
    """Read live command lines, tolerating only processes that exit during inspection."""

    processes = {}
    for directory in proc_root.iterdir():
        # Only numeric proc entries identify processes.
        if not directory.name.isdecimal():
            continue
        try:
            command = (directory / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        except (FileNotFoundError, ProcessLookupError):
            continue
        processes[int(directory.name)] = command
    return processes


def _require_stopped(trials: list[Any]) -> dict:
    """Reject GPU processes, HPO owners, other kernels and surviving recorded worker PIDs."""

    output = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"
    ], text=True, timeout=15)
    gpu_processes = [line.strip() for line in output.splitlines() if line.strip()]
    # Any GPU owner can still publish work, including an unmapped host PID.
    if gpu_processes:
        raise RuntimeError("Stop active GPU processes before search recovery: " + repr(gpu_processes))
    processes = _process_snapshot()
    caller = os.getpid()
    blockers = {}
    for pid, command in processes.items():
        # The invoking notebook itself may remain alive while owning both locks.
        if pid == caller:
            continue
        # Other notebook kernels cannot be silently adopted as idle coordinators.
        if any(marker in command for marker in (
            "common.hpo_worker", "common.dit_hpo_runner", "hpo_worker.py", 
            "dit_hpo_runner.py", "ipykernel_launcher", "ipykernel.kernelapp"
        )):
            blockers[pid] = command
    for trial in trials:
        attributes = trial.user_attrs
        for key in ("worker_pid", "coordinator_pid", "pid"):
            recorded = attributes.get(key)
            # Recycled PIDs remain ambiguous without a saved start-time identity.
            if isinstance(recorded, int) and not isinstance(recorded, bool) and recorded in processes:
                blockers[recorded] = processes[recorded]
    # Fail before storage changes whenever process ownership is uncertain.
    if blockers:
        raise RuntimeError("Other kernels or HPO workers block search recovery: " + repr(blockers))
    return {"caller_pid": caller, "gpu_processes": [], "other_hpo_or_kernel_processes": []}


def _validate_identity(plan: dict, study: Any) -> dict:
    """Authenticate the recipe, live source and both persistent study identities."""

    from common.dit_hpo_remote import _verify_snapshot


    _verify_snapshot(plan["checkout_root"], plan["identity"])
    control = Path(plan["control_root"])
    recipe = runner._read(control / "recipe.json")
    expected = {
        "version": plan["version"], 
        "hpo": {key: value for key, value in plan["hpo"].items() if key not in {
            "concurrent_trials", "worker_gpu_memory_limit_mb", "worker_gpu_ids"
        }}, 
        "source_sha256": plan["identity"]["source_sha256"], 
        "versions": plan["identity"]["versions"], "python": plan["identity"]["python"]
    }
    # Follow-up recipes additionally seal their transferred source provenance.
    if "transfer_manifest" in plan:
        expected["transfer_manifest"] = plan["transfer_manifest"]
    # Runtime routing may differ, but scientific and source identity cannot.
    if recipe != expected:
        raise ValueError("Stopped-search recovery recipe differs from the supplied plan.")
    envelope = runner._read(Path(plan["study_root"]) / "study_spec.json")
    spec = envelope.get("spec")
    # Authenticate the file before trusting any stored study fields.
    if not isinstance(spec, dict) or envelope.get("fingerprint") != _fingerprint(spec):
        raise ValueError("Stopped-search recovery study specification checksum is invalid.")
    # SQLite and sidecar must name the same exact study as the notebook.
    if study.study_name != plan["study_name"] or spec.get("study_name") != plan["study_name"]:
        raise ValueError("Stopped-search recovery study name differs from the plan.")
    # Both independent persistent identity representations must agree.
    if study.user_attrs.get("study_spec") != spec or study.user_attrs.get("study_spec_fingerprint") != envelope["fingerprint"]:
        raise ValueError("Stopped-search recovery storage identity differs from its sidecar.")
    for key in (
        "task", "model_name", "epochs", "seed", "fit_method", "objective_metrics", 
        "objective_directions", "dtype_policy", "n_startup_trials", "search_profile"
    ):
        # These ordinary protocol controls share their public API representation.
        if spec.get(key) != plan["hpo"].get(key):
            raise ValueError("Stopped-search recovery protocol differs for " + key + ".")
    # The storage dataset spelling is canonical lowercase.
    if spec.get("dataset_name") != str(plan["hpo"]["dataset_name"]).lower():
        raise ValueError("Stopped-search recovery dataset differs from the plan.")
    for key in ("search_space_overrides", "pruning"):
        # Empty omitted controls retain their original legacy identity.
        if (spec.get(key) or {}) != (plan["hpo"].get(key) or {}):
            raise ValueError("Stopped-search recovery protocol differs for " + key + ".")
    data_selection = spec.get("data_selection", {})
    selection = data_selection.get("resolved", {})
    for key in ("validation_source", "validation_ratio"):
        # Never accept different test/holdout feedback in a stopped study.
        if key in plan["hpo"] and selection.get(key) != plan["hpo"][key]:
            raise ValueError("Stopped-search recovery data selection differs for " + key + ".")
    directions = [direction.name.lower() for direction in study.directions]
    # Direction changes would alter finalist meaning even with unchanged scores.
    if directions != plan["hpo"]["objective_directions"]:
        raise ValueError("Stopped-search recovery objective directions differ from the plan.")
    return {"recipe_sha256": runner._digest(control / "recipe.json"), "study_spec_fingerprint": envelope["fingerprint"]}


def _trial_record(trial: Any) -> dict:
    """Preserve states, parameters, objective values and original recovery attributes."""

    from optuna.distributions import distribution_to_json


    return {
        "number": trial.number, "state": trial.state.name, "values": trial.values, 
        "params": trial.params, "user_attrs": trial.user_attrs, "system_attrs": trial.system_attrs, 
        "distributions": {key: distribution_to_json(value) for key, value in trial.distributions.items()}, 
        "intermediate_values": {str(key): value for key, value in trial.intermediate_values.items()}, 
        "datetime_start": None if trial.datetime_start is None else trial.datetime_start.isoformat(), 
        "datetime_complete": None if trial.datetime_complete is None else trial.datetime_complete.isoformat()
    }


def _trial_identity(trials: list[Any]) -> str:
    """Compare persisted trial evidence, retaining exceptional diagnostic floats as strings."""

    # Optuna intermediate diagnostics may legitimately contain NaN or infinity.
    encoded = json.dumps([_trial_record(trial) for trial in trials], sort_keys=True, default=str)
    return _fingerprint(encoded)


def _backup(study_root: Path, control: Path, trials: list[Any], identity: dict, evidence: dict) -> Path:
    """Write a consistent SQLite backup and provenance before changing any trial."""

    destination = control / "stopped_search_recovery" / uuid.uuid4().hex
    destination.mkdir(parents=True, exist_ok=False)
    database = study_root / "study.db"
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(destination / "study.db")) as backup:
            source.backup(backup)
    for path in (
        control / "recipe.json", control / "budget.json", control / "search_status.json", 
        study_root / "trials.csv", study_root / "study_spec.json"
    ):
        # Older studies may not have a summary or aggregate CSV yet.
        if path.is_file():
            shutil.copyfile(path, destination / path.name)
    # Preserve exceptional intermediate diagnostics in standard JSON-compatible strings.
    records = json.loads(json.dumps([_trial_record(trial) for trial in trials], default=str), parse_constant=str)
    runner._write(destination / "before.json", {
        "created_at_unix": time.time(), "identity": identity, "process_evidence": evidence, 
        "trials": records, "pending_trial_numbers": [trial.number for trial in trials if trial.state.name == "RUNNING"]
    })
    return destination


def _copy_database(source_path: Path, destination_path: Path, existing_destination: bool = False) -> None:
    """Use one SQLite backup transaction, bounding lock retries without replacing files."""

    started = time.monotonic()

    def progress(status: int, remaining: int, total: int) -> None:
        """Abort a prolonged busy/locked backup so SQLite rolls its transaction back."""

        # Python otherwise retries busy destinations indefinitely.
        if status in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED} and time.monotonic() - started >= 30.0:
            raise TimeoutError("SQLite recovery publication remained locked for 30 seconds.")

    target = destination_path.as_uri() + ("?mode=rw" if existing_destination else "?mode=rwc")
    with closing(sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(target, uri=True, timeout=5.0)) as destination:
            source.backup(destination, pages=-1, progress=progress, sleep=0.1)


def _stage_recovery(database: Path, plan: dict, pending: list[Any], backup: Path, deadline: float) -> list[Any]:
    """Apply ordinary Optuna APIs to a private node-local study snapshot."""

    import optuna


    storage = optuna.storages.RDBStorage("sqlite:///" + database.as_posix())
    study = optuna.load_study(study_name=plan["study_name"], storage=storage)
    try:
        for position, frozen in enumerate(pending, start=1):
            trial = optuna.trial.Trial(study, frozen._trial_id)
            # Expiry alone is not evidence that an older worker died at its cutoff.
            reason = "deadline" if frozen.user_attrs.get("stop_reason") == "deadline" and isinstance(
                frozen.user_attrs.get("deadline"), dict
            ) else "interrupted"
            trial.set_user_attr("stopped_search_recovery", {
                "previous_state": "RUNNING", "reason": reason, "backup_path": str(backup), 
                "recovered_at_unix": time.time(), "search_deadline_unix": deadline, 
                "result_not_promoted": True
            })
            trial.set_user_attr("stop_reason", reason)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            # These changes remain private until the entire copy passes verification.
            if position % 10 == 0 or position == len(pending):
                print(f"Prepared {position}/{len(pending)} interrupted trials.", flush=True)
        return study.get_trials(deepcopy=True)
    finally:
        storage.remove_session()
        storage.engine.dispose()


def recover_stopped_search(plan: dict) -> dict:
    """Fail verified orphan RUNNING trials after the persisted search deadline.

    The caller must be the only remaining notebook kernel. Both coordinator
    locks, the remote source identity and an idle GPU/process inspection must
    succeed. A remote backup precedes every mutation. Optuna edits a private
    node-local copy; a verified SQLite backup transaction publishes that copy
    into the existing database file without replacing its inode. Completed,
    pruned and queued trials and the original clock remain unchanged. No work launches.
    With no RUNNING records this returns without requiring an expired deadline.

    Args:
        plan: Existing recipe returned by the DiT notebook's make_plan call.

    Returns:
        dict: Recovery receipt, including the backup path and affected trials.

    Raises:
        RuntimeError: Remote execution, exclusive ownership or idle-process checks fail.
        ValueError: Scientific identity differs or the search deadline has not elapsed.
        OSError: A required backup or receipt cannot be written.
    """

    from common.dit_hpo_remote import _remote_root


    _remote_root(plan["checkout_root"])
    study_root = Path(plan["study_root"]).resolve()
    control = Path(plan["control_root"]).resolve()
    # Both acquired locks must protect the actual selected study.
    if control != study_root / "notebook_runner":
        raise ValueError("Stopped-search recovery control directory does not belong to its study.")
    # A first notebook execution has nothing to recover and starts no clock.
    if not (study_root / "study.db").is_file():
        return {"status": "no_running_trials", "recovered_trial_numbers": [], "study_started": False}
    with runner._coordinator(plan), study_lock(study_root):
        study = runner._load_study(plan)
        before = study.get_trials(deepcopy=True)
        study_attributes = dict(study.user_attrs)
        pending = [trial for trial in before if trial.state.name == "RUNNING"]
        receipt_path = control / "stopped_search_recovery.json"
        # Ordinary completed searches need neither expiry nor recovery mutations.
        if not pending:
            # Repeated clicks retain the original successful recovery receipt.
            if receipt_path.is_file():
                receipt = runner._read(receipt_path)
                # Return it only while it still describes the exact current study.
                if receipt.get("all_trials_sha256_after") == _trial_identity(before):
                    return receipt
            return {"status": "no_running_trials", "recovered_trial_numbers": [], "study_started": True}
        deadline = runner._phase_deadline(plan, "search")
        # Recovery cannot cancel unexpired work or create an implicit deadline.
        if deadline is None or time.time() < deadline:
            raise ValueError("Stopped-search recovery requires the persisted search deadline to have elapsed.")
        # A frozen selection must never be silently reinterpreted after recovery.
        if (control / "finalists.json").exists():
            raise ValueError("Do not reconcile running trials after finalists have been frozen.")
        identity = _validate_identity(plan, study)
        evidence = _require_stopped(pending)
        unchanged = [trial for trial in before if trial.state.name != "RUNNING"]
        unchanged_hash = _trial_identity(unchanged)
        budget_hash = runner._digest(control / "budget.json")
        print(f"Backing up the study before recovering {len(pending)} interrupted trials.", flush=True)
        backup = _backup(study_root, control, before, identity, evidence)
        database = study_root / "study.db"
        original_inode = database.stat().st_ino
        with tempfile.TemporaryDirectory(prefix="dit-stopped-search-", dir="/tmp") as directory:
            staged_database = Path(directory) / "study.db"
            _copy_database(backup / "study.db", staged_database)
            staged = _stage_recovery(staged_database, plan, pending, backup, deadline)
            # All completed, pruned, queued and previously failed evidence is immutable.
            if _trial_identity([trial for trial in staged if trial.number not in {item.number for item in pending}]) != unchanged_hash:
                raise RuntimeError("Unrelated trial evidence changed in staged recovery; the original study is unchanged.")
            # Every abandoned allocation must be terminal before any publication.
            if any(trial.state.name == "RUNNING" for trial in staged):
                raise RuntimeError("Running trials remain in staged recovery; the original study is unchanged.")
            current_study = runner._load_study(plan)
            current = current_study.get_trials(deepcopy=True)
            # Never overwrite a study updated by an owner bypassing coordinator locks.
            if _trial_identity(current) != _trial_identity(before) or current_study.user_attrs != study_attributes:
                raise RuntimeError("The original study changed during recovery preparation; refusing publication.")
            # Recovery spends no new experimental time allowance.
            if runner._digest(control / "budget.json") != budget_hash:
                raise RuntimeError("The experiment clock changed during stopped-search recovery.")
            _require_stopped(pending)
            print("Publishing verified recovery into the existing study database.", flush=True)
            _copy_database(staged_database, database, existing_destination=True)
            study = runner._load_study(plan)
            after = study.get_trials(deepcopy=True)
            # Existing SQLite connections must continue to reference the same file.
            if database.stat().st_ino != original_inode or _trial_identity(after) != _trial_identity(staged):
                raise RuntimeError("Published recovery differs from its verified snapshot; inspect the preserved backup.")
            print(f"Published recovery for {len(pending)} interrupted trials.", flush=True)
        temporary = study_root / "trials.csv.recovery.tmp"
        study.trials_dataframe().to_csv(temporary, index=False)
        temporary.replace(study_root / "trials.csv")
        status_path = control / "search_status.json"
        status = runner._read(status_path) if status_path.is_file() else {}
        runner._write(status_path, {**status, **runner.search_summary(plan)})
        receipt = {
            "status": "recovered", "study_name": plan["study_name"], "backup_path": str(backup), 
            "recovered_trial_numbers": [trial.number for trial in pending], 
            "states_before": dict(Counter(trial.state.name for trial in before)), 
            "states_after": dict(Counter(trial.state.name for trial in after)), 
            "unaffected_trials_sha256": unchanged_hash, "budget_sha256": budget_hash, 
            "all_trials_sha256_after": _trial_identity(after), "identity": identity, 
            "process_evidence": evidence, "finished_at_unix": time.time(), 
            "training_started": False, "deadline_extended": False, 
            "publication": "sqlite_backup_into_existing_file", "database_inode_preserved": True
        }
        runner._write(backup / "receipt.json", receipt)
        runner._write(receipt_path, receipt)
        return receipt
