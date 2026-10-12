"""Authenticate semantic studies before creating node-local SQLite transactions.

Choose a verified node-local cache such as /tmp on each container. Shared study
directories retain immutable database snapshots and the scientific recipe;
separate hosts use separate study roots. Preparation allocates no trials and
starts no experiment clock. Existing stores follow the shared recovery API.
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
    """Bind the semantic objective, native profile, runner recipe and live source."""

    from common.semantic_hpo_runner import _validate_plan


    _validate_plan(plan)
    options = plan["hpo"]
    # Live teacher objects and other profile inputs cannot bypass semantic sealing.
    if options.get("teacher_network") is not None or options.get("continual_profile") is not None \
    or plan.get("transfer_manifest") is not None:
        raise ValueError("Local semantic storage requires only its sealed native semantic profile.")
    root = Path(plan["study_root"]).resolve()
    control = Path(plan["control_root"]).resolve()
    expected_name = "continual-dit_classifier-" + options["dataset_name"].lower() + "-semantic_consolidation_runner"
    # A sealed plan must address only its own runner controls and study identity.
    if control != root / "notebook_runner" or plan["study_name"] != expected_name:
        raise ValueError("Semantic storage paths or study name differ from the plan.")
    recipe_bytes = (control / "recipe.json").read_bytes()
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
    # Sampler seed is immutable within a shard, even when another shard differs.
    if recipe != json.loads(json.dumps(expected, allow_nan=False)):
        raise ValueError("Semantic storage recipe differs from the sealed plan.")
    checkout = Path(plan["checkout_root"]).resolve()
    for relative, digest in recipe["source_sha256"].items():
        source = (checkout / relative).resolve()
        # Refuse relocated or changed source before any storage mutation.
        if not source.is_relative_to(checkout) or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError("Semantic storage source identity changed: " + relative)
    return {
        "version": 1, "study_root": str(root), "study_name": expected_name, 
        "objective_metrics": list(options["objective_metrics"]), "directions": ["maximize"], 
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest()
    }


def _receipt(plan: dict, identity: dict) -> bool:
    """Read an immutable bootstrap receipt without reconstructing missing evidence."""

    path = Path(plan["control_root"]) / _RECEIPT
    # An absent receipt is valid only before this API has bootstrapped a store.
    if not path.exists():
        return False
    # Present receipts cannot authorize a different scientific recipe.
    if runner._read(path) != identity:
        raise ValueError("Semantic SQLite bootstrap receipt differs from its current recipe.")
    return True


def _require_empty(database: Path, identity: dict) -> None:
    """Permit first initialization only for one correctly directed empty study."""

    import optuna


    storage = optuna.storages.RDBStorage("sqlite:///" + database.as_posix())
    try:
        summaries = optuna.get_all_study_summaries(storage=storage)
        # Fresh storage must not conceal an unrelated or additional study.
        if len(summaries) != 1 or summaries[0].study_name != identity["study_name"]:
            raise ValueError("Fresh semantic SQLite must contain exactly its expected study.")
        study = optuna.load_study(study_name=identity["study_name"], storage=storage)
        # Objective directions belong to the sealed scientific identity.
        if [direction.name.lower() for direction in study.directions] != identity["directions"]:
            raise ValueError("Fresh semantic SQLite objective directions changed.")
        study_id = storage.get_study_id_from_name(identity["study_name"])
        # Any trial or metadata marks a partial initialization requiring recovery.
        if study.get_trials(deepcopy=False) or study.user_attrs or storage.get_study_system_attrs(study_id):
            raise ValueError("Unsealed semantic SQLite contains existing trial or study state.")
    finally:
        storage.remove_session()
        storage.engine.dispose()


def is_fresh_storage(plan: dict) -> bool:
    """Recognize an authenticated empty cache without weakening sealed-study resume.

    The caller owns the coordinator lock. A missing cache or foreign host is
    never restored implicitly; prepare_storage performs that explicit action.
    """

    root = Path(plan["study_root"])
    # Ordinary storage without a local marker follows the existing resume path.
    if not (root / "sqlite_local.json").is_file():
        return False
    identity = _identity(plan)
    authenticated = _receipt(plan, identity)
    # A published specification ends the empty-store bootstrap exception.
    if (root / "study_spec.json").is_file():
        return False
    # Unsealed caches must retain their original bootstrap authorization.
    if not authenticated:
        raise ValueError("Unsealed local semantic SQLite has no bootstrap receipt.")
    _require_empty(database_path(root), identity)
    return True


def prepare_storage(plan: dict, local_root: str | Path = "/tmp/semantic-hpo-sqlite") -> dict:
    """Create or explicitly restore a host-local cache before any campaign START.

    The caller verifies local_root is on node-local storage, outside shared
    /workspace. Existing same-host transactions are preserved; migration of an
    unmarked established database is intentionally outside this bootstrap API.
    Call independently for each host's distinct study root. Shared artifacts
    contain snapshots, not a live database used concurrently by both hosts.
    """

    identity = _identity(plan)
    root = Path(plan["study_root"]).resolve()
    local = Path(local_root).resolve()
    # Cache and durable output trees must not contain one another.
    if local == root or local.is_relative_to(root) or root.is_relative_to(local):
        raise ValueError("Local SQLite cache must be separate from durable study storage.")
    with runner._coordinator(plan), study_lock(root):
        identity = _identity(plan)
        has_receipt = _receipt(plan, identity)
        # Marked stores can reuse or explicitly restore their authenticated cache.
        if (root / "sqlite_local.json").exists():
            # Neither an original receipt nor a finished seal may be reconstructed.
            if not has_receipt and not (root / "study_spec.json").is_file():
                raise ValueError("Unsealed local semantic SQLite has no bootstrap receipt.")
            marker = enable_local_sqlite(root, local)
            is_fresh_storage(plan)
            return marker
        # Existing evidence without a marker must never be overwritten or adopted.
        if has_receipt or (root / "study.db").exists() or (root / "study_spec.json").exists():
            raise ValueError("Preserve existing semantic storage; only a fresh study can be initialized.")
        import optuna


        local.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="semantic-bootstrap-", dir=local) as temporary:
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
