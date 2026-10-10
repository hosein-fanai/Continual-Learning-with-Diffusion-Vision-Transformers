"""Prepare authenticated fresh UNet studies with node-local SQLite transactions.

The caller verifies that local_root is on node-local storage. Durable snapshots
and the immutable bootstrap receipt stay beside the ordinary study artifacts.
This module never imports TensorFlow, allocates trials or starts an experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from common import dit_hpo_runner as runner
from common.hpo_process import study_lock
from common.hpo_sqlite import database_path, enable_local_sqlite


_RECEIPT = "local_sqlite_bootstrap.json"


def _identity(plan: dict) -> dict:
    """Authenticate the ordinary UNet plan against its sealed source recipe."""

    options = plan["hpo"]
    # The bootstrap exception is specific to the ordinary teacher-free objective.
    if options.get("task") != "generation" or options.get("model_name") != "unet" \
    or options.get("use_distillation", False) or options.get("teacher_network") is not None \
    or options.get("search_profile") is not None or options.get("continual_profile") is not None \
    or plan.get("transfer_manifest") is not None \
    or options.get("objective_metrics") != ["generation_loss"] \
    or options.get("objective_directions") != ["minimize"]:
        raise ValueError("Local UNet storage requires ordinary teacher-free generation.")
    root = Path(plan["study_root"]).resolve()
    control = Path(plan["control_root"]).resolve()
    expected_name = "generation-unet-" + options["dataset_name"].lower()
    # Paths and study names must identify the same notebook-controlled study.
    if control != root / "notebook_runner" or plan["study_name"] != expected_name:
        raise ValueError("UNet storage paths or study name differ from the plan.")
    recipe_path = control / "recipe.json"
    recipe_bytes = recipe_path.read_bytes()
    recipe = json.loads(recipe_bytes)
    expected = {
        "version": plan["version"], 
        "hpo": {
            key: value for key, value in options.items()
            if key not in {"concurrent_trials", "worker_gpu_memory_limit_mb", "worker_gpu_ids"}
        }, 
        "source_sha256": plan["identity"]["source_sha256"], 
        "versions": plan["identity"]["versions"], "python": plan["identity"]["python"]
    }
    # Runtime concurrency controls do not change the immutable scientific recipe.
    if recipe != expected:
        raise ValueError("UNet storage recipe differs from the sealed plan.")
    checkout = Path(plan["checkout_root"]).resolve()
    for relative, digest in recipe["source_sha256"].items():
        source = (checkout / relative).resolve()
        # Refuse foreign paths or modified implementations before storage writes.
        if not source.is_relative_to(checkout) or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError("UNet storage source identity changed: " + relative)
    return {
        "version": 1, "study_root": str(root), "study_name": expected_name, 
        "directions": ["minimize"], "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest()
    }


def _receipt(plan: dict, identity: dict) -> bool:
    """Validate an existing immutable receipt without creating one."""

    path = Path(plan["control_root"]) / _RECEIPT
    # Only a deliberately published receipt can authorize fresh initialization.
    if not path.exists():
        return False
    # Preserve the original receipt rather than blessing changed settings.
    if runner._read(path) != identity:
        raise ValueError("UNet SQLite bootstrap receipt differs from its current recipe.")
    return True


def _require_empty(database: Path, identity: dict) -> None:
    """Require the sole correctly directed study to have no scientific state."""

    import optuna


    storage = optuna.storages.RDBStorage("sqlite:///" + database.as_posix())
    try:
        summaries = optuna.get_all_study_summaries(storage=storage)
        # Unexpected extra studies cannot share this bootstrap exemption.
        if len(summaries) != 1 or summaries[0].study_name != identity["study_name"]:
            raise ValueError("Fresh UNet SQLite must contain exactly its expected study.")
        study = optuna.load_study(study_name=identity["study_name"], storage=storage)
        # A differently directed objective is a different experiment.
        if [direction.name.lower() for direction in study.directions] != identity["directions"]:
            raise ValueError("Fresh UNet SQLite objective directions changed.")
        study_id = storage.get_study_id_from_name(identity["study_name"])
        # Partial initialization or allocated trials require strict recovery.
        if study.get_trials(deepcopy=False) or study.user_attrs or storage.get_study_system_attrs(study_id):
            raise ValueError("Unsealed UNet SQLite contains existing trial or study state.")
    finally:
        storage.remove_session()
        storage.engine.dispose()


def is_fresh_storage(plan: dict) -> bool:
    """Identify only an authenticated empty bootstrap, preserving strict resume.

    The caller owns the notebook coordinator lock. A sealed study always uses
    ordinary recovery. Malformed or nonempty unsealed local stores fail closed.
    """

    root = Path(plan["study_root"])
    # Ordinary shared-storage studies retain the original runner behavior.
    if not (root / "sqlite_local.json").is_file():
        return False
    identity = _identity(plan)
    authenticated = _receipt(plan, identity)
    # The HPO engine validates all sealed state under its existing resume rules.
    if (root / "study_spec.json").is_file():
        return False
    # Missing receipts cannot be reconstructed from an existing database.
    if not authenticated:
        raise ValueError("Unsealed local UNet SQLite has no bootstrap receipt.")
    _require_empty(database_path(root), identity)
    return True


def prepare_storage(plan: dict, local_root: str | Path = "/tmp/unet-hpo-sqlite") -> dict:
    """Prepare or restore local transactions without starting trials or a clock.

    Use only a verified node-local directory. Existing unmarked databases are
    never adopted or replaced; migration of an established study remains an
    explicit separate operation. Existing caches use the shared recovery API.
    """

    identity = _identity(plan)
    root = Path(plan["study_root"]).resolve()
    local = Path(local_root).resolve()
    # Cache and durable snapshots must not overlap in the directory hierarchy.
    if local == root or local.is_relative_to(root) or root.is_relative_to(local):
        raise ValueError("Local SQLite cache must be separate from durable study storage.")
    with runner._coordinator(plan), study_lock(root):
        identity = _identity(plan)
        has_receipt = _receipt(plan, identity)
        # Established local stores use the existing explicit restoration protocol.
        if (root / "sqlite_local.json").exists():
            # A failed bootstrap never silently authorizes a future initialization.
            if not has_receipt and not (root / "study_spec.json").is_file():
                raise ValueError("Unsealed local UNet SQLite has no bootstrap receipt.")
            marker = enable_local_sqlite(root, local)
            is_fresh_storage(plan)
            return marker
        # Existing unmarked artifacts require explicit migration, not replacement.
        if has_receipt or (root / "study.db").exists() or (root / "study_spec.json").exists():
            raise ValueError("Preserve existing UNet storage; only a fresh study can be initialized.")
        import optuna


        local.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="unet-bootstrap-", dir=local) as temporary:
            source = Path(temporary) / "study.db"
            storage = optuna.storages.RDBStorage("sqlite:///" + source.as_posix())
            try:
                optuna.create_study(
                    study_name=identity["study_name"], storage=storage, 
                    directions=identity["directions"]
                )
            finally:
                storage.remove_session()
                storage.engine.dispose()
            _require_empty(source, identity)
            marker = enable_local_sqlite(root, local, source_database=source)
        runner._write(Path(plan["control_root"]) / _RECEIPT, identity)
        is_fresh_storage(plan)
        return marker
