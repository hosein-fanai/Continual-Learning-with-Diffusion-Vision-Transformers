"""Freeze completed DiT results as parameter hints for a separate HPO study.

Only a read-only SQLite connection touches the source database. Optuna reads a
transactionally consistent temporary backup. Scores and checkpoints are never
inserted into the destination study: every queued hint starts a new trial.
"""

from __future__ import annotations

from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

from common.hpo_process import study_lock, write_atomic_json
from common.hpo_sqlite import database_path


TRANSFER_VERSION = 1
STUDY_NAME = "generation-diffusion_transformer-cifar10"
TRANSFER_PARAMETERS = frozenset({
    "learning_rate", "optimizer", "weight_decay", "clipnorm", 
    "global_clipnorm", "learning_rate_schedule", "ema_decay", "schedule", 
    "p_uncond", "image_loss_coef", "timesteps", "use_cfg", 
    "patches_pos_embed_type", "patches_pos_merger_type", "conds_merger_type", 
    "time_freq_dim", "time_embed_trainable", "time_mlp_ratio", 
    "label_embed_type", "label_freq_dim", "label_mlp_ratio", 
    "final_activation_func", "loss_function", "droppath_rate"
})
PROTOCOL = {
    "task": "generation", "model_name": "diffusion_transformer", 
    "dataset_name": "cifar10", "epochs": 50, "fit_method": "fit", 
    "dtype_policy": "float32", "validation_source": "test", 
    "validation_ratio": 0.0, "objective_metrics": ["generation_loss"], 
    "objective_directions": ["minimize"], "evaluation_loss": "mse", 
    "objective_network": "ema"
}


def _canonical_dataset(dataset_name: str) -> str:
    """Resolve the two source datasets supported by the maintained DiT notebooks."""

    canonical = str(dataset_name).lower()
    # Transfer protocol ownership excludes unreviewed datasets and label semantics.
    if canonical not in {"cifar10", "cifar100"}:
        raise ValueError("DiT transfer supports only CIFAR10 or CIFAR100.")
    return canonical


def _protocol(dataset_name: str) -> dict:
    """Retain the shared scientific protocol with the requested dataset identity."""

    return {**PROTOCOL, "dataset_name": dataset_name}


def _study_name(dataset_name: str) -> str:
    """Construct the public HPO study name for the authenticated dataset."""

    return "generation-diffusion_transformer-" + dataset_name


def _fingerprint(value: Any) -> str:
    """Hash a finite plain JSON tree using the repository's canonical encoding."""

    encoded = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read(path: Path) -> dict:
    """Read one mapping-shaped JSON artifact."""

    value = json.loads(path.read_text(encoding="utf-8"))
    # JSON sidecars must retain their documented mapping envelope.
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON mapping: " + str(path))
    return value


def _require(actual: dict, expected: dict, label: str) -> None:
    """Reject a different source scientific protocol at its owning boundary."""

    for key, value in expected.items():
        # Reject changed scientific controls before using any source score.
        if actual.get(key) != value:
            raise ValueError(f"{label} differs for {key}: expected {value!r}.")


def _source_protocol(recipe: dict, spec: dict, dataset_name: str) -> None:
    """Authenticate the shared scientific controls before ranking source trials."""

    common = {
        key: value for key, value in _protocol(dataset_name).items()
        if key not in {"evaluation_loss", "objective_network", "dataset_name", 
                       "validation_source", "validation_ratio"}
    }
    _require(recipe.get("hpo", {}), common, "Source runner recipe")
    _require(spec, common, "Source study specification")
    # Both persisted identities must refer to this notebook's dataset.
    if str(recipe["hpo"].get("dataset_name", "")).lower() != dataset_name \
    or spec.get("dataset_name") != dataset_name:
        raise ValueError("Transfer requires the " + dataset_name.upper() + " source study.")
    _require(recipe["hpo"], {
        "validation_source": "test", "validation_ratio": 0.0
    }, "Source runner recipe")
    _require(spec.get("data_selection", {}).get("resolved", {}), {
        "validation_source": "test", "validation_ratio": 0.0, 
        "effective_validation_ratio": 0.0, "drop_remainder": False
    }, "Source validation selection")
    _require(spec, {
        "study_name": _study_name(dataset_name), "max_train_samples": None, 
        "max_val_samples": None, "fit_kwargs": {}, "effective_distillation": False, 
        "use_ensemble_accuracy": False, "model_overrides": {}, "wrapper_overrides": {}
    }, "Source study specification")
    for key in ("seed", "n_startup_trials", "search_space_overrides", "pruning"):
        actual = recipe["hpo"].get(key)
        expected = spec.get(key)
        # Legacy unexpanded studies seal omission as an empty override mapping.
        if key == "search_space_overrides":
            actual, expected = actual or {}, expected or {}
        # File recipe and storage identity must describe the same search.
        if actual != expected:
            raise ValueError("Source recipe and study specification differ for " + key + ".")


def _config_receipt(study_root: Path, trial: Any, dataset_name: str) -> dict:
    """Verify a completed trial's resolved configuration without loading weights."""

    from common.config import load_config


    reference = trial.user_attrs.get("resolved_config_path", trial.user_attrs.get("config_path"))
    # COMPLETE storage state alone is insufficient without a resolved config.
    if not reference:
        raise ValueError(f"Completed source trial {trial.number} has no resolved configuration.")
    path = Path(reference).resolve()
    # Do not follow trial metadata into another experiment's artifacts.
    if not path.is_relative_to(study_root):
        raise ValueError(f"Source trial {trial.number} configuration is outside its study directory.")
    config_bytes = path.read_bytes()
    config = load_config(path)
    _require(vars(config.dataset), {
        "validation_source": "test", "validation_ratio": 0.0, 
        "drop_remainder": False, "max_train_samples": None, "max_val_samples": None
    }, "Source trial dataset")
    # Retain the same dataset when older configs vary name capitalization.
    if config.dataset.name.lower() != dataset_name:
        raise ValueError("Source trial dataset is not " + dataset_name.upper() + ".")
    _require(vars(config.training), {
        "task": "generation", "epochs": 50, "fit_method": "fit", 
        "fit_kwargs": {}, "dtype_policy": "float32"
    }, "Source trial training")
    _require(vars(config.model), {"name": "diffusion_transformer"}, "Source trial model")
    _require(config.model.wrapper_kwargs, {
        "test_network_name": "ema", "use_ema": True
    }, "Source trial wrapper")
    # Reconstruction targets are not comparable to noise-prediction MSE.
    if config.model.wrapper_kwargs.get("swap_noise_image", False):
        raise ValueError("Transfer requires noise-prediction source objectives.")
    evaluation_loss = config.model.kwargs.get("compile_args", {}).get(
        "evaluation_loss", config.model.loss_function
    )
    # MAE-trained candidates remain eligible only with fixed MSE evaluation.
    if evaluation_loss != "mse":
        raise ValueError("Transfer requires source objectives evaluated with MSE.")
    _require(config.hpo, {
        "trial_number": trial.number, "params": trial.params, 
        "objectives": [trial.value], "objective_metrics": ["generation_loss"], 
        "objective_directions": ["minimize"]
    }, "Source trial HPO metadata")
    # Completed configurations are immutable while the source adds other trials.
    if path.read_bytes() != config_bytes:
        raise RuntimeError("Source configuration changed while freezing transfer: " + str(path))
    return {
        "path": str(path), "sha256": hashlib.sha256(config_bytes).hexdigest()
    }


def validate_transfer_manifest(manifest: dict) -> dict:
    """Authenticate a frozen transfer before it can seed a destination study.

    The returned mapping contains only JSON data. This validates its internal
    provenance; it deliberately does not reread a growing upstream experiment.
    """

    # Versioned manifests prevent silently interpreting an unknown transfer shape.
    if not isinstance(manifest, dict) or manifest.get("version") != TRANSFER_VERSION:
        raise ValueError("Unsupported DiT transfer manifest.")
    unsigned = {key: value for key, value in manifest.items() if key != "transfer_sha256"}
    # A frozen destination cannot accept edits to its upstream snapshot.
    if manifest.get("transfer_sha256") != _fingerprint(unsigned):
        raise ValueError("DiT transfer manifest checksum differs.")
    protocol = manifest.get("protocol", {})
    # Malformed protocol envelopes are incompatible manifests, not dataset aliases.
    if not isinstance(protocol, dict):
        raise ValueError("DiT transfer manifest protocol differs.")
    dataset_name = _canonical_dataset(protocol.get("dataset_name", ""))
    # Every follow-up uses the same scientific objective as its source ranking.
    if protocol != _protocol(dataset_name):
        raise ValueError("DiT transfer manifest protocol differs.")
    request = manifest.get("request", {})
    # Requests retain their mapping envelope across both supported schema shapes.
    if not isinstance(request, dict):
        raise ValueError("DiT transfer manifest request differs.")
    # Version-one CIFAR10 manifests predate an explicit requested dataset field.
    if request.get("dataset_name", "cifar10") != dataset_name:
        raise ValueError("DiT transfer manifest request dataset differs.")
    source = manifest.get("source", {})
    # Source identity must be structurally complete before reading dataset fields.
    if not isinstance(source, dict) or not isinstance(source.get("recipe"), dict) \
    or not isinstance(source["recipe"].get("hpo"), dict):
        raise ValueError("DiT transfer manifest source dataset differs.")
    # Internal source identities must agree even when an edited manifest is re-signed.
    if source.get("study_name") != _study_name(dataset_name) \
    or str(source.get("recipe", {}).get("hpo", {}).get("dataset_name", "")).lower() != dataset_name:
        raise ValueError("DiT transfer manifest source dataset differs.")
    expected_root = Path(request.get("source_results_path", "")) / "generation" / "diffusion_transformer" / dataset_name
    # A dataset-specific snapshot cannot be silently redirected to another study.
    if Path(source.get("study_root", "")) != expected_root:
        raise ValueError("DiT transfer manifest source dataset hierarchy differs.")
    selected = manifest.get("selected", [])
    hints = manifest.get("initial_trials", [])
    # Each hint needs exactly one finite completed source provenance entry.
    if not selected or len(selected) != len(hints):
        raise ValueError("DiT transfer requires completed source trials and matching hints.")
    seen = set()
    for source, hint in zip(selected, hints):
        # Architecture and capacity are always sampled afresh in the new study.
        if not isinstance(hint, dict) or not hint or not set(hint).issubset(TRANSFER_PARAMETERS):
            raise ValueError("DiT transfer hints contain unsupported or architectural parameters.")
        fingerprint = _fingerprint(hint)
        # Deduplication and hint integrity persist across notebook restarts.
        if source.get("hint_sha256") != fingerprint or fingerprint in seen:
            raise ValueError("DiT transfer hints are inconsistent or duplicated.")
        # Interrupted, pruned and nonfinite candidates never supply hints.
        if not math.isfinite(float(source["value"])) or source.get("state") != "COMPLETE":
            raise ValueError("DiT transfer source trials must be finite and COMPLETE.")
        seen.add(fingerprint)
    return json.loads(json.dumps(manifest, allow_nan=False))


def _snapshot_source(study_root: Path, top_k: int, dataset_name: str) -> dict:
    """Read a consistent Optuna backup and return authenticated top parameter hints."""

    import optuna


    database = database_path(study_root)
    recipe_path = study_root / "notebook_runner" / "recipe.json"
    spec_path = study_root / "study_spec.json"
    # The follow-up cannot freeze an experiment that has not begun producing results.
    if not all(path.is_file() for path in (database, recipe_path, spec_path)):
        raise FileNotFoundError(
            "Source DiT HPO has no completed study yet. Run the first notebook until "
            "at least one finite trial is COMPLETE, then rerun this setup cell: " + str(study_root)
        )
    recipe_bytes = recipe_path.read_bytes()
    spec_bytes = spec_path.read_bytes()
    recipe = json.loads(recipe_bytes)
    envelope = json.loads(spec_bytes)
    spec = envelope.get("spec")
    # Authenticate the source sidecar before opening its SQLite database.
    if not isinstance(spec, dict) or envelope.get("fingerprint") != _fingerprint(spec):
        raise ValueError("Source study specification checksum is invalid.")
    _source_protocol(recipe, spec, dataset_name)
    with tempfile.TemporaryDirectory(prefix="dit-hpo-transfer-") as temporary:
        snapshot = Path(temporary) / "study.db"
        # SQLite's backup handles WAL and concurrent source commits transactionally.
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as source:
            with closing(sqlite3.connect(snapshot)) as destination:
                source.backup(destination)
        snapshot_sha256 = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        storage = optuna.storages.RDBStorage("sqlite:///" + snapshot.as_posix())
        try:
            study = optuna.load_study(study_name=_study_name(dataset_name), storage=storage)
            _require(study.user_attrs, {
                "study_spec": spec, "study_spec_fingerprint": envelope["fingerprint"]
            }, "Source Optuna study")
            # Confirm storage direction rather than relying only on copied metadata.
            if [direction.name for direction in study.directions] != ["MINIMIZE"]:
                raise ValueError("Source Optuna objective directions differ.")
            trials = study.get_trials(deepcopy=False)
            ranked = sorted([
                trial for trial in trials if trial.state.name == "COMPLETE"
                and trial.value is not None and math.isfinite(float(trial.value))
            ], key=lambda trial: (float(trial.value), trial.number))
            selected = []
            initial_trials = []
            seen = set()
            for trial in ranked:
                hint = {key: value for key, value in trial.params.items() if key in TRANSFER_PARAMETERS}
                hint_sha256 = _fingerprint(hint)
                # Equivalent settings from different architectures consume one hint.
                if not hint or hint_sha256 in seen:
                    continue
                receipt = _config_receipt(study_root, trial, dataset_name)
                selected.append({
                    "trial_number": trial.number, "state": "COMPLETE", "value": float(trial.value), 
                    "params_sha256": _fingerprint(trial.params), "hint_sha256": hint_sha256, 
                    "config": receipt
                })
                initial_trials.append(hint)
                seen.add(hint_sha256)
                # Freeze only the requested number of unique source configurations.
                if len(selected) >= top_k:
                    break
            # Never silently start an unconditioned follow-up when results are missing.
            if not selected:
                raise ValueError(
                    "Source DiT HPO has no finite COMPLETE trials with transferable settings. "
                    "Finish at least one trial in the first notebook, then rerun this setup cell."
                )
            state_counts = {}
            for trial in trials:
                state_counts[trial.state.name] = state_counts.get(trial.state.name, 0) + 1
        finally:
            storage.remove_session()
            storage.engine.dispose()
    # Concurrent source identity edits invalidate the cross-file snapshot.
    if recipe_path.read_bytes() != recipe_bytes or spec_path.read_bytes() != spec_bytes:
        raise RuntimeError("Source scientific identity changed while freezing transfer.")
    return {
        "source": {
            "study_root": str(study_root), "study_name": _study_name(dataset_name), 
            "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(), 
            "recipe": recipe, "study_spec_fingerprint": envelope["fingerprint"], 
            "sqlite_snapshot_sha256": snapshot_sha256, "snapshot_trial_states": state_counts
        }, 
        "selected": selected, "initial_trials": initial_trials
    }


def freeze_transfer(
    source_results_path: str | Path, manifest_path: str | Path, top_k: int = 12, 
    dataset_name: str = "CIFAR10"
) -> dict:
    """Freeze read-only source results as immutable partial hints for a new study.

    Args:
        source_results_path: The first notebook's RESULTS_PATH, containing its
            generation/diffusion_transformer/<dataset> study hierarchy.
        manifest_path: JSON file inside a separate follow-up results directory.
            An existing validated file is reused without consulting upstream.
        top_k: Maximum unique conditioning/optimizer settings, ordered by finite
            source loss then trial number. Ordinary usage is a positive integer.
        dataset_name: CIFAR10 or CIFAR100, case-insensitive. Source results and
            destination protocol must retain this dataset. CIFAR10 preserves the
            original version-one request shape for existing frozen manifests.

    Returns:
        dict: Authenticated provenance and initial_trials parameter dictionaries.
            Every destination trial must be trained and evaluated anew. No source
            objective, fitted sampler history, checkpoint or model is imported.

    Raises:
        FileNotFoundError: The first notebook has not initialized its source study.
        ValueError: Completed results are absent or scientific identity differs.
        RuntimeError: Source identity changes during the snapshot, or another
            process already holds the destination manifest lock.
    """

    dataset_name = _canonical_dataset(dataset_name)
    source_root = Path(source_results_path).resolve()
    path = Path(manifest_path).resolve()
    # The source experiment remains strictly read-only during transfer.
    if path.is_relative_to(source_root):
        raise ValueError("The follow-up transfer manifest must be outside the source results directory.")
    request = {"source_results_path": str(source_root), "top_k": top_k}
    # Preserve existing CIFAR10 manifests byte-for-byte while sealing new datasets.
    if dataset_name != "cifar10":
        request["dataset_name"] = dataset_name
    with study_lock(path.parent / ".transfer-lock"):
        # Resume uses the original snapshot even when upstream has more winners.
        if path.is_file():
            manifest = validate_transfer_manifest(_read(path))
            # A new upstream selection requires its own destination experiment.
            if manifest.get("request") != request:
                raise ValueError("Frozen DiT transfer request changed; use a fresh follow-up RESULTS_PATH.")
            return manifest
        study_root = source_root / "generation" / "diffusion_transformer" / dataset_name
        manifest = {
            "version": TRANSFER_VERSION, "request": request, "protocol": _protocol(dataset_name), 
            **_snapshot_source(study_root, top_k, dataset_name)
        }
        manifest["transfer_sha256"] = _fingerprint(manifest)
        validate_transfer_manifest(manifest)
        write_atomic_json(path, manifest)
        return manifest
