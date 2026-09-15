"""Notebook staging of existing Route One APIs; no new training or metric logic.

Run each training stream in its own kernel. An unfinished stream resumes its
latest authenticated native checkpoint; completed streams remain immutable.
The separate frozen record binds the production manifests and notebook sources.
"""

from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import BinaryIO, TYPE_CHECKING

# Import annotation-only types without changing the runtime backend.
if TYPE_CHECKING:
    import pandas as pd
    from semantic_consolidation.config import RouteConfig
    from common.dataloader import DatasetLoader
    import tensorflow as tf
import re
import time

import numpy as np

from allocation_study.artifacts import source_fingerprint, validate_study_source
from notebooks.thesis.completion import publish_completion, reconcile_completions
from common.config import resolve_continual_schedule
from common.experiment import (
    collect_final_stream_metrics, materialize_run_plan, paired_run_statistics,
    read_experiment_manifest, read_long_results,
)
from semantic_consolidation.config import load_route_config, save_route_settings, validate_route_config
from semantic_consolidation.study import (
    _completed_metrics, _read_study_manifest, analyze_study, prepare_study,
    validate_planned_config,
)


CONDITIONS = {
    "cifar10": {
        "baseline": {"route": {"condition": "baseline"}},
        "extra_joint": {"route": {"condition": "extra_joint"}},
        "learned": {"route": {"condition": "learned"}},
    },
    "cifar100": {
        "baseline": {"route": {"condition": "baseline"}},
        "extra_joint": {"route": {"condition": "extra_joint"}},
        "learned": {"route": {"condition": "learned"}},
        "random": {"route": {"condition": "random"}},
        "ce_only": {"route": {"condition": "no_consolidation"}},
    },
}

CONFIRMATION_SEEDS = [1103, 2207, 3301]
CAMPAIGN_VERSION = "minimum_v4_tf220"


def check_runtime() -> dict:
    """Check the maintained notebook runtime before preparing or starting work.

    Returns:
        runtime (dict): TensorFlow/Keras version strings and integer GPU count; no training is
            run.

    Raises:
        RuntimeError: If TensorFlow is not 2.20, Keras is not version 3, or its
            backend is not TensorFlow.
    """
    import tensorflow as tf
    import keras
    # Match the maintained TensorFlow 2.20 and native Keras 3 runtime contract.
    if tf.__version__.split(".")[:2] != ["2", "20"] or keras.__version__.split(".")[0] != "3" \
            or keras.backend.backend() != "tensorflow":
        raise RuntimeError("These notebooks require TensorFlow 2.20 and Keras 3 with the TensorFlow backend.")
    return {"TensorFlow": tf.__version__, "Keras": keras.__version__,
            "GPUs": len(tf.config.list_physical_devices("GPU"))}


def campaign_checklist(record_path: str | Path, *, save: bool=True) -> pd.DataFrame:
    """Return the frozen execution order with its notebook and paired seed.

    Args:
        record_path (str | Path): Externally retained frozen_design.json; its identities and
            source hashes must still match.
        save (bool): True also writes the immutable checklist; False only returns its table.

    Returns:
        checklist (pd.DataFrame): One row per declared stream in the frozen randomized execution
            order; seed and run IDs are not averaged.

    Raises:
        ValueError: If frozen identities or an existing checklist disagree.
        OSError: If required files cannot be read or written.
    """
    import pandas as pd
    _, manifests = _campaign(record_path)
    names = ["02_CIFAR10_platform.ipynb", "03_CIFAR10_extra_joint.ipynb", "04_CIFAR10_learned.ipynb",
             "05_CIFAR100_platform.ipynb", "06_CIFAR100_extra_joint.ipynb", "07_CIFAR100_learned.ipynb",
             "08_CIFAR100_random.ipynb", "09_CIFAR100_ce_only.ipynb"]
    notebook_map = dict(zip(((dataset, condition) for dataset, conditions in CONDITIONS.items()
                             for condition in conditions), names))
    rows = [{"dataset": dataset, "condition": entry["condition"],
             "seed": entry["stream"]["stream_seed"], "paired_stream": entry["block_id"],
             "notebook": notebook_map[dataset, entry["condition"]], "run_id": entry["run_id"]}
            for dataset, manifest in manifests.items() for entry in materialize_run_plan(manifest)]
    table = pd.DataFrame(rows, index=pd.RangeIndex(1, len(rows) + 1, name="execution_step"))
    # Write the checklist only when requested.
    if save:
        path = Path(record_path).parent / "execution_checklist.csv"
        text = table.to_csv(lineterminator="\n")
        # Use existing evidence only when the corresponding artifact is present.
        if path.exists():
            # Existing checklist differs from the frozen plan; preserve and
            # inspect it.
            if path.read_text(encoding="utf-8") != text:
                raise ValueError("Existing checklist differs from the frozen plan; preserve and inspect it.")
        # Handle the complementary supported case without inventing observations.
        else:
            with path.open("x", encoding="utf-8", newline="") as stream:
                stream.write(text)
    return table


def _digest(path: Path) -> str:
    """Ignore notebook execution products while binding every cell's source.

    Args:
        path (str | Path): File to read or write; relative paths use the current working
            directory.

    Returns:
        sha256 (str): Hexadecimal SHA-256; notebook cell types and sources are bound, while
            execution outputs and metadata are excluded.

    Raises:
        OSError: If the source cannot be read.
        ValueError: If notebook JSON cannot be decoded.
    """
    data = path.read_bytes()
    # Bind notebook source without binding execution products.
    if path.suffix == ".ipynb":
        cells = json.loads(data)["cells"]
        data = json.dumps([
            {"cell_type": cell["cell_type"], "source": "".join(cell["source"])}
            for cell in cells
        ], sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict:
    """Read the saved artifact.

    Args:
        path (str | Path): File to read or write; relative paths use the current working
            directory.

    Returns:
        record (dict): Decoded JSON mapping without modifying its file.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the JSON is malformed.
    """
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_new(path: Path, value: dict) -> None:
    """None; creates the JSON file exclusively and refuses an existing file.

    Args:
        path (str | Path): File to read or write; relative paths use the current working
            directory.
        value (object): JSON-like object or scalar. Missing and nonfinite values are handled as
            described in the return contract.

    Returns:
        written (None): None; creates the JSON file exclusively and refuses an existing file.

    Raises:
        FileExistsError: If the destination already exists.
        ValueError: If nonfinite values cannot be serialized.
        OSError: If publication fails.
    """
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def prepare_campaign(campaign_dir: str | Path, templates: dict[str, Path], seeds: list[int]) -> Path:
    """Freeze both fresh confirmation studies without loading data or training.

    Settle development choices first. Keep the returned frozen_design.json unchanged and retain
    a separate copy; its hashes authenticate the manifests.

    Args:
        campaign_dir (str | Path): Fresh destination for both paired studies; an existing
            campaign is preserved.
        templates (dict[str, Path]): Exactly cifar10 and cifar100, mapped to their central
            recipe YAML files.
        seeds (list[int]): The three declared independent stream seeds; development seed 17 is
            separate.

    Returns:
        frozen_record (Path): Path to the new frozen_design.json binding both manifests, recipes
            and notebook/helper sources; no data is loaded.

    Raises:
        FileExistsError: If the campaign exists.
        ValueError: If seeds, templates or planned scientific settings are invalid.
        OSError: If source binding or publication fails.
    """
    runtime = check_runtime()
    directory = Path(campaign_dir).resolve()
    # Campaign already exists: the selected artifact; use its frozen record.
    if directory.exists():
        raise FileExistsError(f"Campaign already exists: {directory}; use its frozen record.")
    # The revised minimum campaign requires seeds the selected artifact; development seed 17
    # stays separate.
    if seeds != CONFIRMATION_SEEDS:
        raise ValueError(f"The revised minimum campaign requires seeds {CONFIRMATION_SEEDS}; development seed 17 stays separate.")
    # Templates must contain exactly cifar10 and cifar100.
    if set(templates) != set(CONDITIONS):
        raise ValueError("Templates must contain exactly cifar10 and cifar100.")
    paths = {name: Path(path).resolve() for name, path in templates.items()}
    configs = {name: load_route_config(path) for name, path in paths.items()}
    for name, config in configs.items():
        # Template dataset differs from the selected artifact.
        if config.common.dataset.name != name:
            raise ValueError(f"Template dataset differs from {name}.")
    notebook_dir = Path(__file__).resolve().parent
    notebooks = sorted(path for path in notebook_dir.glob("*.ipynb")
                       if re.match(r"^0[2-9][ _-]", path.name))
    # Expected the eight numbered training notebooks 02 through 09.
    if len(notebooks) != 8:
        raise ValueError("Expected the eight numbered training notebooks 02 through 09.")
    # Bind recovery and extraction procedures along with the frozen scientific recipe.
    helpers = sorted(notebook_dir.glob("*.py"))
    rationale = [notebook_dir / name for name in ("HYPERPARAMETER_RATIONALE.md", "recipe_sources.json")
                 if (notebook_dir / name).is_file()]
    bound_files = [*helpers, *notebooks, *paths.values(), *rationale]
    source_root = Path(__file__).resolve().parents[2]
    fingerprints = {path.relative_to(source_root).as_posix(): _digest(path) for path in bound_files}
    # Each native preparation validates seeds, all conditions, and source identity.
    studies = {}
    for name, config in configs.items():
        manifest_path = prepare_study(config, directory / name, seeds,
                                      conditions=CONDITIONS[name], phase="confirmation")
        manifest = read_experiment_manifest(manifest_path)
        studies[name] = {"manifest_path": manifest_path.relative_to(directory).as_posix(),
                         "manifest_hash": manifest["manifest_hash"]}
    record = {"schema_version": 2, "phase": "confirmation", "seeds": list(seeds),
              "campaign_version": CAMPAIGN_VERSION, "declared_stream_count": 24, "runtime": runtime,
              "studies": studies, "bound_files": fingerprints,
              "notebook_hash_scope": "Every cell type/source; execution outputs and metadata excluded."}
    record_path = directory / "frozen_design.json"
    _write_new(record_path, record)
    _campaign(record_path)
    return record_path


def _campaign(record_path: str | Path) -> tuple[dict, dict]:
    """Frozen record with resolved artifact locators and dataset-to-manifest mapping, after source
    and identity validation.

    Args:
        record_path (str | Path): Externally retained frozen_design.json; its identities and
            source hashes must still match.

    Returns:
        campaign (tuple[dict, dict]): Frozen record with resolved artifact locators and
            dataset-to-manifest mapping, after source and identity validation.

    Raises:
        ValueError: If schema, locators, manifest identity or source hashes disagree.
        OSError: If bound artifacts cannot be read.
    """
    record_path = Path(record_path).resolve()
    record = _read_json(record_path)
    # Expected a separately retained confirmation frozen_design.
    if record.get("schema_version") not in (1, 2) or record.get("phase") != "confirmation":
        raise ValueError("Expected a separately retained confirmation frozen_design.json.")
    # Frozen campaign must contain both declared datasets.
    if set(record["studies"]) != set(CONDITIONS):
        raise ValueError("Frozen campaign must contain both declared datasets.")
    for filename, digest in record["bound_files"].items():
        path = Path(filename)
        # Resolve portable version-two artifact locators.
        if record["schema_version"] == 2:
            path = _relative_path(Path(__file__).resolve().parents[2], filename)
        # Frozen notebook/helper/template source changed: the selected artifact.
        if _digest(path) != digest:
            raise ValueError(f"Frozen notebook/helper/template source changed: {filename}.")
    # Resolve portable version-two artifact locators.
    if record["schema_version"] == 2:
        for study in record["studies"].values():
            study["manifest_path"] = str(_relative_path(record_path.parent, study["manifest_path"]))
    manifests = {name: _read_study_manifest(Path(study["manifest_path"]), study["manifest_hash"])
                 for name, study in record["studies"].items()}
    return record, manifests


def _relative_path(root: Path, filename: str) -> Path:
    """Resolve a portable locator without accepting absolute or escaping paths.

    Args:
        root (Path): Root against which the relative locator is resolved.
        filename (str): Portable POSIX relative file locator; absolute and escaping locators are
            rejected.

    Returns:
        resolved (Path): Absolute path contained within the supplied root.

    Raises:
        ValueError: If the locator is absolute, uses a drive/backslash or escapes the root.
    """
    path = Path(filename)
    # Portable campaign paths must be relative POSIX paths.
    if path.is_absolute() or "\\" in filename or ":" in filename:
        raise ValueError("Portable campaign paths must be relative POSIX paths.")
    resolved = (root / path).resolve()
    # Portable campaign path escapes its source or campaign directory.
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Portable campaign path escapes its source or campaign directory.")
    return resolved


def _selected_config(config_path: str | Path, manifest_path: Path, entry: dict, relocatable: bool) -> RouteConfig:
    """Validated live route configuration; only permitted artifact locators can differ from the
    sealed YAML.

    Args:
        config_path (str | Path): Central or materialized route YAML, loaded through the
            existing strict configuration API.
        manifest_path (Path): Native paired manifest identifying the study and adjacent run
            artifacts.
        entry (dict): One materialized condition/seed entry with its exact schedule, run ID and
            manifest hash.
        relocatable (bool): Whether only artifact locators may be resolved relative to a
            relocated frozen campaign.

    Returns:
        config (RouteConfig): Validated live route configuration; only permitted artifact
            locators can differ from the sealed YAML.

    Raises:
        ValueError: If run identity or scientific settings differ from the plan.
        OSError: If the configuration or manifest cannot be read.
    """
    config = load_route_config(config_path)
    identity = config.common.continually_learn
    # Selected run YAML identity differs from the frozen plan.
    if identity.experiment_run_id != entry["run_id"] or identity.experiment_manifest_hash != entry["manifest_hash"]:
        raise ValueError("Selected run YAML identity differs from the frozen plan.")
    if relocatable:
        # Only locators change: the externally bound manifest, run identity and
        # every scientific setting still pass the native frozen-plan validator.
        identity.experiment_manifest_path = str(manifest_path)
        config.common.training.results_path = str(manifest_path.parent / "runs")
    # Selected run YAML identity differs from the frozen plan.
    elif Path(identity.experiment_manifest_path).resolve() != manifest_path:
        raise ValueError("Selected run YAML identity differs from the frozen plan.")
    validate_planned_config(config)
    return config


def _outputs(manifest_path: Path, manifest: dict, *, complete: bool = False) -> dict:
    """Validated run-ID mapping; an incomplete subset is allowed only when complete=False.

    Args:
        manifest_path (Path): Native paired manifest identifying the study and adjacent run
            artifacts.
        manifest (dict): Authenticated native study manifest with materialized condition and
            stream definitions.
        complete (bool): True requires every declared stream; False permits a validated
            completed subset.

    Returns:
        completed (dict): Validated run-ID mapping; an incomplete subset is allowed only when
            complete=False.

    Raises:
        ValueError: If native evidence is invalid or required declared streams are missing.
        OSError: If completion evidence cannot be read or reconciled.
    """
    outputs = reconcile_completions(manifest_path, expected_hash=manifest["manifest_hash"])
    planned = {entry["run_id"]: entry for entry in materialize_run_plan(manifest)}
    # Completed records must cover the declared runs; finish every planned stream
    # before analysis.
    if not set(outputs) <= set(planned) or complete and set(outputs) != set(planned):
        raise ValueError("Completed records must cover the declared runs; finish every planned stream before analysis.")
    return outputs


def _initialize(config: RouteConfig, context: dict) -> tuple:
    """Live config/context pair with seeded runtime, source provenance and untouched controller
    slots; native input arrays keep their configured dtype.

    Args:
        config (RouteConfig): RouteConfig containing the live common and semantic settings for
            one stream.
        context (dict): Live notebook context returned by stream selection, including frozen
            identities and controller ownership.

    Returns:
        selected (tuple[RouteConfig, dict]): Live config/context pair with seeded runtime,
            source provenance and untouched controller slots; native input arrays keep their
            configured dtype.

    Raises:
        RuntimeError: If the TensorFlow/Keras runtime is unsupported.
        ValueError: If runtime configuration is invalid.
    """
    check_runtime()
    from common.runtime import configure_runtime, effective_seed
    from semantic_consolidation.provenance import source_provenance
    seed = effective_seed(config.common)
    # Use the common runtime seed when no route seed was specified.
    if config.route.seed is None:
        config.route.seed = seed
    config.common.hpo["semantic_consolidation"] = asdict(config.route)
    configure_runtime(seed, config.common.training.dtype_policy, config.common.training.deterministic_ops)
    provenance = source_provenance()
    lease_path = Path(config.common.continually_learn.checkpoint_dir).with_suffix(".running.lock")
    lease = _acquire_stream_lease(lease_path)
    context.update(config=config, settings=config.route, provenance=provenance, lease=lease,
                   started=time.perf_counter(), controller=None, observer=None, finished=False,
                   started_utc=datetime.now(timezone.utc).isoformat(),
                   runtime_config=asdict(config))
    return config, context


def _acquire_stream_lease(path: Path) -> BinaryIO:
    """Reserve one live stream across local kernels without blocking another run.

    Args:
        path (Path): Stream-specific lock file outside its checkpoint root. The file may
            persist; ownership is the OS lock, not file existence.

    Returns:
        lease (BinaryIO): Open binary file holding an exclusive nonblocking lock. Closing it or
            exiting the process releases ownership.

    Raises:
        RuntimeError: If another kernel currently owns this stream.
        OSError: If the lock directory or file cannot be created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lease = path.open("a+b", buffering=0)
    try:
        # Use the same native platform locks as completion publication.
        if os.name == "nt":
            import msvcrt
            lease.seek(0)
            msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
        # Handle the complementary supported case without inventing observations.
        else:
            import fcntl
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        lease.close()
        raise RuntimeError("This stream is running in another kernel; use its live session or wait for it to exit.") from error
    return lease


def _configure_recovery(config: RouteConfig, checkpoint_dir: Path) -> str | None:
    """Select the newest native checkpoint without changing the sealed recipe.

    Args:
        config (RouteConfig): Live route configuration. Checkpoint saving must already be
            enabled in its scientific recipe.
        checkpoint_dir (Path): Dedicated root for this one seed and treatment. A missing or
            empty root starts the stream. Recognized unpublished native temporaries
            are retained; published evidence must identify a valid native checkpoint.

    Returns:
        resume_from (str | None): Absolute checkpoint locator, or None for a fresh stream. Only
            runtime locators on config are updated.

    Raises:
        ValueError: If saving is disabled or existing checkpoint evidence is corrupt or
            incomplete. Evidence is preserved for inspection.
        FileNotFoundError: If published evidence contains no valid committed state.
        OSError: If checkpoint discovery cannot read the root.
    """
    from common.recovery import find_latest_task_checkpoint
    continual = config.common.continually_learn
    # Recovery must be part of the frozen recipe, rather than enabled afterward.
    if not continual.save_task_checkpoints:
        raise ValueError("Enable save_task_checkpoints before freezing this recipe.")
    continual.checkpoint_dir = str(checkpoint_dir.resolve())
    continual.resume_from = None
    # Unpublished native task temporaries can survive a process interruption.
    # Preserve them, but they do not constitute a committed training boundary.
    existing = list(checkpoint_dir.iterdir()) if checkpoint_dir.exists() else []
    published = [path for path in existing if not (
        path.is_dir() and re.fullmatch(r"(?:\.task-\d+\.tmp-|\.progress-)[0-9a-f]{32}", path.name)
        or path.is_file() and re.fullmatch(r"\.progress-index-[0-9a-f]{32}", path.name))]
    initial = checkpoint_dir / ".initial"
    # A failed initial publication may contain only native unpublished directories.
    if initial in published and initial.is_dir() and all(
            path.is_dir() and re.fullmatch(r"\.task-\d+\.tmp-[0-9a-f]{32}", path.name)
            for path in initial.iterdir()):
        published.remove(initial)
    # Resolve existing published state through the native authenticated selector.
    if published:
        continual.resume_from = str(find_latest_task_checkpoint(checkpoint_dir))
    return continual.resume_from


def load_run(record_path: str | Path, dataset: str, condition: str, repeat_index: int | None=0) -> tuple:
    """Select one frozen stream; None chooses its first uncompleted repeat.

    Rerun in a fresh kernel to resume the first unfinished repeat. A completed stream is never
    rerun, and a failed stream is never silently skipped.

    Args:
        record_path (str | Path): Externally retained frozen_design.json; its identities and
            source hashes must still match.
        dataset (str): cifar10 or cifar100, selecting the corresponding frozen study.
        condition (str): Declared treatment name; CE-only maps to the native no_consolidation
            control.
        repeat_index (int | None): Zero-based declared repeat, or None to choose the first
            unfinished repeat without skipping failures.

    Returns:
        selected (tuple[RouteConfig, dict]): Live configuration/context for one declared
            unfinished stream; resume_from selects its newest valid native checkpoint, or None
            starts it.

    Raises:
        FileExistsError: If the selected repeat or all repeats are complete.
        ValueError: If selection, source identity, start receipt or checkpoint evidence is
            invalid.
        OSError: If artifacts cannot be read.
    """
    record_path = Path(record_path).resolve()
    record, manifests = _campaign(record_path)
    # Unknown dataset/condition in this minimum campaign.
    if dataset not in CONDITIONS or condition not in CONDITIONS[dataset]:
        raise ValueError("Unknown dataset/condition in this minimum campaign.")
    manifest = manifests[dataset]
    manifest_path = Path(record["studies"][dataset]["manifest_path"])
    outputs = _outputs(manifest_path, manifest)
    plan = materialize_run_plan(manifest)
    entries = sorted((entry for entry in plan if entry["condition"] == condition), key=lambda entry: entry["block_id"])
    # Choose the first unfinished repeat without dropping failed streams.
    if repeat_index is None:
        entries = [entry for entry in entries if entry["run_id"] not in outputs]
        # All planned repeats for the selected artifact/the selected artifact are complete.
        if not entries:
            raise FileExistsError(f"All planned repeats for {dataset}/{condition} are complete.")
        entry = entries[0]
    # Handle the complementary supported case without inventing observations.
    else:
        # Repeat_index must select a declared zero-based repeat, or be None.
        if isinstance(repeat_index, bool) or not isinstance(repeat_index, int) or not 0 <= repeat_index < len(entries):
            raise ValueError("repeat_index must select a declared zero-based repeat, or be None.")
        entry = entries[repeat_index]
    # Run already completed: the selected artifact.
    if entry["run_id"] in outputs:
        raise FileExistsError(f"Run already completed: {entry['run_id']}.")
    config_path = manifest_path.parent / f"{entry['run_id']}.yaml"
    relocatable = record["schema_version"] == 2
    config = _selected_config(config_path, manifest_path, entry, relocatable)
    marker = manifest_path.parent / f"{entry['run_id']}.started.json"
    # A start receipt binds retries to the same declared stream.
    if marker.exists():
        started = _read_json(marker)
        # Started receipt differs from the frozen stream identity.
        if started.get("run_id") != entry["run_id"] or started.get("manifest_hash") != entry["manifest_hash"]:
            raise ValueError("Started receipt differs from the frozen stream identity.")
    _configure_recovery(config, manifest_path.parent / "checkpoints" / entry["run_id"])
    validate_planned_config(config)
    return _initialize(config, {"record_path": record_path, "dataset": dataset,
                               "manifest_path": manifest_path, "entry": entry, "config_path": config_path,
                               "relocatable": relocatable,
                               "started_marker": marker})


def _development_identity(config: RouteConfig) -> str:
    """Separate pilots when resolved settings or executable sources change.

    Args:
        config (RouteConfig): Resolved recipe after applying the pilot seed,
            treatment and complete class schedule, before runtime initialization.

    Returns:
        identity (str): Short SHA-256 of settings and source. Recovery locators
            are excluded because they do not define the scientific recipe.

    Raises:
        OSError: If executable sources cannot be read.
        ValueError: If source fingerprinting or finite JSON encoding fails.
    """
    settings = asdict(config)
    for field in ("checkpoint_dir", "resume_from"):
        settings["common"]["continually_learn"].pop(field, None)
    payload = {"settings": settings, "source": source_fingerprint()["sha256"],
               "workflow": _digest(Path(__file__))}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def load_development(config_path: str | Path, condition: str="baseline", seed: int=17) -> tuple:
    """Prepare one fresh, validation-only pilot using a seeded full class order.

    Args:
        config_path (str | Path): Central or materialized route YAML, loaded through the
            existing strict configuration API.
        condition (str): Declared treatment name; CE-only maps to the native no_consolidation
            control.
        seed (int): Integer stream seed in [0, 2**32); 17 is the development default.

    Returns:
        selected (tuple[RouteConfig, dict]): One validation-only config/context with a fixed
            seeded complete class order and recipe-specific recovery root.

    Raises:
        ValueError: If dataset, treatment, seed, recipe or existing checkpoint evidence is
            invalid.
        OSError: If configuration or checkpoint files cannot be read.
    """
    config = load_route_config(config_path)
    dataset = config.common.dataset.name
    # Unknown dataset/condition for the minimum development plan.
    if dataset not in CONDITIONS or condition not in CONDITIONS[dataset]:
        raise ValueError("Unknown dataset/condition for the minimum development plan.")
    # Seed must be an integer in [0, 2**32).
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    continual = config.common.continually_learn
    order, groups = resolve_continual_schedule(continual.class_num, continual.class_order,
                                               continual.task_groups, task_size=continual.task_size, seed=seed)
    order = np.random.default_rng(seed).permutation(order).tolist()
    boundaries = np.cumsum([0, *map(len, groups)])
    continual.class_order = order
    continual.task_groups = [order[start:stop] for start, stop in zip(boundaries[:-1], boundaries[1:])]
    continual.class_order_mode = continual.task_order_mode = "fixed"
    continual.seed = config.common.training.seed = config.route.seed = seed
    continual.experiment_phase = "development"
    continual.experiment_manifest_path = continual.experiment_manifest_hash = continual.experiment_run_id = None
    config.route.condition = CONDITIONS[dataset][condition]["route"]["condition"]
    config.common.training.project_tag = f"development-{dataset}-{condition}-seed-{seed}"
    # Bind inherited settings and executable sources, not only the leaf YAML.
    checkpoint_dir = Path(config.common.training.results_path) / "checkpoints" / f"{condition}-{seed}-{_development_identity(config)}"
    _configure_recovery(config, checkpoint_dir)
    validate_route_config(config)
    validate_planned_config(config)
    return _initialize(config, {"record_path": None, "dataset": dataset})


def _check_context(context: dict) -> None:
    """None; rechecks frozen source and configuration without changing training state.

    Args:
        context (dict): Live notebook context returned by stream selection, including frozen
            identities and controller ownership.

    Returns:
        validated (None): None; rechecks frozen source and configuration without changing
            training state.

    Raises:
        ValueError: If source or planned identity changed.
        OSError: If frozen evidence cannot be read.
    """
    # Recheck or publish the frozen confirmation identity.
    if context["record_path"] is not None:
        _campaign(context["record_path"])
        _selected_config(context["config_path"], context["manifest_path"], context["entry"],
                         context.get("relocatable", False))


def attach_route(context: dict, bundle: dict) -> None:
    """Attach the existing controller to the exact factory-created live model.

    Args:
        context (dict): Live notebook context returned by stream selection, including frozen
            identities and controller ownership.
        bundle (dict): Dictionary returned by common.model.get_model, with the live model and
            native continual results.

    Returns:
        attached (None): None; wraps the same live model with the existing route controller and
            optional observer, preserving its weights and optimizer.

    Raises:
        RuntimeError: If a controller is already attached.
        ValueError: If scientific configuration changed or unsupported extensions were enabled.
        OSError: If a new start receipt cannot be written.
    """
    from semantic_consolidation.controller import RouteController
    from semantic_consolidation.model import adapt_model
    _check_context(context)
    # Route is already attached; restart this notebook in a fresh kernel.
    if context["controller"] is not None:
        raise RuntimeError("Route is already attached; restart this notebook in a fresh kernel.")
    expected, actual = deepcopy(context["runtime_config"]), asdict(context["config"])
    for values in (expected, actual):
        values["common"]["dataset"].pop("trainset_len", None)
        values["common"]["optimizer"].pop("decay_steps", None)
    # Runtime scientific settings changed after selecting the run.
    if expected != actual:
        raise ValueError("Runtime scientific settings changed after selecting the run.")
    settings = context["settings"]
    # The minimum notebooks do not enable scheduling/replay extensions.
    if settings.extensions:
        raise ValueError("The minimum notebooks do not enable scheduling/replay extensions.")
    controller = RouteController(settings)
    bundle["generative_model"] = adapt_model(bundle["generative_model"], controller)
    context["controller"] = controller
    # Attach validation/resource observations only when enabled in the recipe.
    if settings.experimental.get("enabled", False):
        from semantic_consolidation.experimental import ExperimentalController
        observer = ExperimentalController(context["config"].common, settings.experimental, settings.seed, bundle=bundle)
        object.__setattr__(bundle["generative_model"], "experimental_controller", observer)
        context["observer"] = observer
    # Use existing evidence only when the corresponding artifact is present.
    if context["record_path"] is not None and not context["started_marker"].exists():
        entry = context["entry"]
        _write_new(context["started_marker"], {
            "run_id": entry["run_id"], "manifest_hash": entry["manifest_hash"],
            "started_utc": context["started_utc"],
            "status": "started; only completed_runs.json establishes completion",
        })


def close_run(context: dict, *, release: bool = False) -> None:
    """Release the optional monitor, including when the training cell raises.

    Args:
        context (dict): Live notebook context returned by stream selection, including frozen
            identities and controller ownership.
        release (bool): True also releases the stream lease after failed training. False keeps
            ownership until finish_run publishes completion.

    Returns:
        closed (None): None; closes the optional resource observer, including on training
            failure. A context without an observer is unchanged.

    Raises:
        None: This wrapper declares no additional validation failures; observer close errors
            propagate.
    """
    # Release the optional resource observer.
    if context.get("observer") is not None:
        context["observer"].close()
    # Release stream ownership after failure or completed publication.
    if release and context.get("lease") is not None:
        context["lease"].close()


def finish_run(context: dict, config: RouteConfig, bundle: dict, history: dict, trainset: tf.data.Dataset | DatasetLoader, valset: tf.data.Dataset | None) -> dict:
    """Save one stream; a completed same-context retry returns its saved metrics.

    Args:
        context (dict): Live notebook context returned by stream selection, including frozen
            identities and controller ownership.
        config (RouteConfig): RouteConfig containing the live common and semantic settings for
            one stream.
        bundle (dict): Dictionary returned by common.model.get_model, with the live model and
            native continual results.
        history (dict): Native training-history mapping returned by common.train.train_model.
        trainset (tf.data.Dataset | DatasetLoader): Native training input from get_datasets;
            continual loaders preserve NumPy image/label dtypes.
        valset (tf.data.Dataset | None): Native held-out validation dataset, or None when the
            continual loader constructs task validation internally.

    Returns:
        evaluations (dict): Native report metrics after saving route evidence and publishing
            completion; a same-context publication retry reuses authenticated saved metrics.

    Raises:
        RuntimeError: If the context does not own this model or was already finished without
            reusable evidence.
        ValueError: If frozen or completed evidence conflicts.
        OSError: If native reporting or publication fails.
    """
    _check_context(context)
    # Finish_run requires this context's one trained, unfinished model.
    if context["controller"] is None or config is not context["config"]:
        raise RuntimeError("finish_run requires this context's one trained, unfinished model.")
    # A publication failure can leave valid native reports and a complete artifact.
    # Reconcile before report/controller.save: a retry must not overwrite evidence
    # or run evaluation again. A fresh context cannot adopt somebody else's run.
    if context["record_path"] is not None:
        entry, manifest_path = context["entry"], context["manifest_path"]
        manifest = _read_study_manifest(manifest_path, entry["manifest_hash"])
        saved = _outputs(manifest_path, manifest).get(entry["run_id"])
        # Reuse authenticated completion after an interrupted publication.
        if saved is not None:
            # Completed stream belongs to a different live run context; preserve
            # its saved results.
            if Path(saved["results_path"]).resolve() != Path(config.common.training.results_path).resolve() \
                    or saved.get("started_utc") != context["started_utc"]:
                raise ValueError("Completed stream belongs to a different live run context; preserve its saved results.")
            context["finished"] = True
            close_run(context, release=True)
            return context.get("evaluations", dict(saved["metrics"]))
    # Finish_run requires this context's one trained, unfinished model.
    if context["finished"]:
        raise RuntimeError("finish_run requires this context's one trained, unfinished model.")
    from common.train import report
    from semantic_consolidation.controller import _json_value
    from semantic_consolidation.provenance import save_provenance
    controller, observer = context["controller"], context["observer"]
    result_path = Path(config.common.training.results_path).resolve()
    bundle["continual_details"]["semantic_consolidation"] = controller.records
    save_provenance(context["provenance"], result_path)
    save_route_settings(context["settings"], result_path / "route.settings.yaml")
    controller.save(result_path)
    evaluations = report(config.common, history, bundle, trainset, valset=valset)
    context["evaluations"] = evaluations
    # Save the observer records alongside the native stream results.
    if observer is not None:
        observer.save(result_path)
        bundle["continual_details"]["section11_experimental"] = observer.records
    _check_context(context)
    # Recheck or publish the frozen confirmation identity.
    if context["record_path"] is not None:
        entry, manifest_path = context["entry"], context["manifest_path"]
        manifest = _read_study_manifest(manifest_path, entry["manifest_hash"])
        outputs = _outputs(manifest_path, manifest)
        # This stream already has a completed record.
        if entry["run_id"] in outputs:
            raise FileExistsError("This stream already has a completed record.")
        matrix = bundle["continual_details"]["ordinary_accuracy_matrix"]
        metrics = _completed_metrics(matrix, len(entry["stream"]["task_groups"]))
        completed = {"manifest_hash": entry["manifest_hash"], "run_id": entry["run_id"],
                     "condition": entry["condition"], "results_path": str(result_path),
                     "seconds": time.perf_counter() - context["started"],
                     "started_utc": context["started_utc"], "completed_utc": datetime.now(timezone.utc).isoformat(),
                     "total_updates": sum(row["total_updates"] for row in controller.records),
                     "accuracy_matrix": _json_value(matrix), "accuracy_matrix_source": "ordinary_accuracy_matrix",
                     "metrics": metrics}
        publish_completion(manifest_path, completed, expected_hash=entry["manifest_hash"])
    context["finished"] = True
    close_run(context, release=True)
    return evaluations


def analyze_campaign(record_path: str | Path) -> dict:
    """Analyze all declared streams; safely reuse previously verified statistics.

    Args:
        record_path (str | Path): Externally retained frozen_design.json; its identities and
            source hashes must still match.

    Returns:
        statistics (dict): Native paired final-test statistics by dataset, after all declared
            streams authenticate; existing identical analysis is reused.

    Raises:
        ValueError: If a stream is absent or saved analysis differs from recomputed native
            statistics.
        OSError: If artifacts cannot be read or written.
    """
    from semantic_consolidation.controller import _json_value
    record, manifests = _campaign(record_path)
    prepared = {}
    # Check both datasets before writing any analysis artifacts.
    for dataset, manifest in manifests.items():
        path = Path(record["studies"][dataset]["manifest_path"])
        prepared[dataset] = (path, _outputs(path, manifest, complete=True))
    results = {}
    for dataset, (path, outputs) in prepared.items():
        manifest = manifests[dataset]
        saved_path = path.parent / "paired_statistics.json"
        # Use existing evidence only when the corresponding artifact is present.
        if not saved_path.exists():
            results[dataset] = _json_value(analyze_study(path, expected_hash=manifest["manifest_hash"]))
        # Handle the complementary supported case without inventing observations.
        else:
            spec = manifest["spec"]["analysis_spec"]
            rows = collect_final_stream_metrics(manifest, {
                run_id: result["metrics"][spec["primary_metric"]] for run_id, result in outputs.items()
            }, expected_hash=manifest["manifest_hash"])
            checked = _json_value(paired_run_statistics(
                rows, condition_a=spec["condition_a"], condition_b=spec["condition_b"],
                metric=spec["primary_metric"], manifest=manifest, expected_hash=manifest["manifest_hash"],
            ))
            saved = _read_json(saved_path)
            # Saved paired statistics differ from authenticated complete-stream
            # outcomes.
            if any(saved.get(key) != value for key, value in checked.items()):
                raise ValueError("Saved paired statistics differ from authenticated complete-stream outcomes.")
            # Saved paired result rows differ from authenticated complete-stream
            # outcomes.
            if read_long_results(path.parent / "paired_results.csv") != rows:
                raise ValueError("Saved paired result rows differ from authenticated complete-stream outcomes.")
            # Saved analysis has a different executable source identity.
            if saved.get("artifact_validation", {}).get("source") != validate_study_source(manifest, "semantic_consolidation"):
                raise ValueError("Saved analysis has a different executable source identity.")
            results[dataset] = saved
    _campaign(record_path)
    return results
