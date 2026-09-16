"""Evaluate a saved cognitive-route checkpoint without retraining.

Both routes share this command. Validation permits explicit exploratory
inference settings. Test evaluation requires the unchanged inference design,
data protocol and route treatment of an authenticated confirmation manifest.
All calibration rows come from the original training-stream validation split.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from common.config import Config, _safe_load_unique_yaml, load_config, resolve_continual_schedule
from common.runtime import configure_runtime, effective_seed
from semantic_consolidation.evaluation import EnsembleEvaluationSettings, _validation_partition, evaluate_checkpoint


def _route_settings(config: Config) -> tuple[str, dict]:
    """Read the route identity and complete settings saved by either runner.

    Args:
        config (Config): Validated configuration tree, with the data, model and experiment
            settings consumed by this operation.

    Returns:
        identity (tuple[str, dict]): (route_name, settings): saved route string and an
            independent top-level settings dict.

    Raises:
        ValueError: If saved metadata does not contain exactly one supported route mapping.
    """

    names = [name for name in ("semantic_consolidation", "gist_memory") if name in config.hpo]
    # A saved route must identify one unambiguous data source and treatment.
    if len(names) != 1 or not isinstance(config.hpo[names[0]], Mapping):
        raise ValueError("Saved config.hpo must contain exactly one cognitive route settings mapping.")
    return names[0], dict(config.hpo[names[0]])


def _evaluation_settings(
    config: Config, settings_path: str | Path | None, split: str,
) -> EnsembleEvaluationSettings:
    """Resolve saved or explicit YAML settings without changing saved artifacts.

    YAML may be a bare evaluation mapping, extensions mapping, route settings,
    or complete route config. Explicit settings remain subject to the frozen
    manifest comparison for test evaluation. Validation invocation enables an
    otherwise absent/disabled diagnostic as an exploratory evaluation only.

    Args:
        config (Config): Validated configuration tree, with the data, model and experiment
            settings consumed by this operation.
        settings_path (str | Path | None): Optional YAML path overriding inference settings;
            test access still requires agreement with the frozen design.
        split (str): Declared data split; supported training, validation or test access is
            constrained by this operation.

    Returns:
        settings (EnsembleEvaluationSettings): Validated EnsembleEvaluationSettings;
            explicit validation evaluation enables an otherwise disabled diagnostic.

    Raises:
        OSError: If an override YAML cannot be read.
        TypeError: If selected settings are not a mapping or contain unknown fields.
        ValueError: If inference controls are invalid.
    """

    _, route = _route_settings(config)
    values = route.get("extensions", {}).get("evaluation", {})
    # An explicit file takes precedence during validation exploration.
    if settings_path is not None:
        with Path(settings_path).open(encoding="utf-8") as stream:
            values = _safe_load_unique_yaml(stream)
        # These enclosing mappings are present in the supported artifact formats.
        for key in ("route", "extensions", "evaluation"):
            # Descend only through an actually present wrapper.
            if isinstance(values, Mapping) and key in values:
                values = values[key]
    # Reject sequences and null rather than silently guessing settings.
    if not isinstance(values, Mapping):
        raise TypeError("Evaluation settings must be a YAML mapping.")
    settings = EnsembleEvaluationSettings(**values)
    # Running validation explicitly requests this diagnostic even for old runs.
    if split == "validation" and not settings.enabled:
        settings = replace(settings, enabled=True)
    return settings


def _resolved_schedule(config: Config) -> tuple[list[int], list[list[int]]]:
    """Resolve the saved original-label order using the existing schedule API.

    Args:
        config (Config): Validated configuration tree, with the data, model and experiment
            settings consumed by this operation.

    Returns:
        schedule (tuple[list[int], list[list[int]]]): (order, groups): original integer
            class order and task-group lists.

    Raises:
        ValueError: If saved class order, groups or randomization controls are invalid.
    """

    continual = config.continually_learn
    return resolve_continual_schedule(
        continual.class_num, continual.class_order, continual.task_groups,
        task_size=continual.task_size, class_order_mode=continual.class_order_mode,
        task_order_mode=continual.task_order_mode, seed=effective_seed(config),
    )


def _confirmation_contract(config: Config, settings: EnsembleEvaluationSettings) -> dict:
    """Authenticate the run and bind all inputs consumed by this evaluator.

    The existing route planned_config resolves the manifest's complete defaults
    and validates its training design. Saved network architecture/weight paths
    are runtime artifacts; this check binds evaluation/data/route settings and
    stream identity, not an independently authenticated training provenance.

    Args:
        config (Config): Validated configuration tree, with the data, model and experiment
            settings consumed by this operation.
        settings (EnsembleEvaluationSettings): Validated settings instance for this
            component; its fields select the behavior described above.

    Returns:
        contract (dict): Dict binding the trusted manifest, run, condition, phase and source
            identity.

    Raises:
        ValueError: If confirmation identity, source, route, inference, data or stream
            settings differ from the frozen design.
        OSError: If required manifest/source files cannot be read.
    """

    from common.experiment import materialize_run_plan, read_experiment_manifest, validate_frozen_confirmation

    continual = config.continually_learn
    # Test access requires the same explicit contract as the shared learner.
    if continual.experiment_phase != "confirmation" or not all((
        continual.experiment_manifest_path, continual.experiment_manifest_hash,
        continual.experiment_run_id,
    )):
        raise ValueError("Test evaluation requires a saved frozen confirmation manifest, trusted hash and run ID.")
    manifest = read_experiment_manifest(
        continual.experiment_manifest_path, expected_hash=continual.experiment_manifest_hash,
    )
    manifest = validate_frozen_confirmation(manifest, expected_hash=continual.experiment_manifest_hash)
    matching = [entry for entry in materialize_run_plan(manifest, expected_hash=continual.experiment_manifest_hash)
                if entry["run_id"] == continual.experiment_run_id]
    # One run ID identifies exactly one condition and seed/order block.
    if len(matching) != 1:
        raise ValueError("The saved run ID must occur exactly once in its frozen manifest.")
    route_name, actual_route = _route_settings(config)
    from common.study_artifacts import validate_study_source
    source = validate_study_source(manifest, route_name)
    # The two route studies resolve their own treatment-specific configuration.
    if route_name == "gist_memory":
        from gist_memory.study import planned_config
    # Semantic modulation has a separate manifest adapter over the common format.
    else:
        from semantic_consolidation.study import planned_config
    planned = planned_config(matching[0], continual.experiment_manifest_path)
    expected_route = asdict(planned.route)
    # Compare through strict JSON so tuple/list YAML representations are equivalent.
    if json.dumps(actual_route, sort_keys=True) != json.dumps(expected_route, sort_keys=True):
        raise ValueError("Saved route settings differ from the frozen confirmation treatment.")
    expected_settings = EnsembleEvaluationSettings(**expected_route.get("extensions", {}).get("evaluation", {}))
    # Inference horizons/weights/calibration must be fixed before test access.
    if not settings.enabled or settings != expected_settings:
        raise ValueError("Test inference settings must match the enabled evaluation design in the frozen manifest.")
    order, groups = _resolved_schedule(config)
    stream = matching[0]["stream"]
    # Bind the realized order and all random dataset-split decisions.
    if order != stream["class_order"] or groups != stream["task_groups"] or effective_seed(config) != stream["stream_seed"]:
        raise ValueError("Saved schedule or seed differs from the frozen confirmation stream.")
    expected_data, actual_data = asdict(planned.common.dataset), asdict(config.dataset)
    expected_data.pop("trainset_len", None)
    actual_data.pop("trainset_len", None)
    # Only the derived training-batch count may differ from the planned dataset.
    if actual_data != expected_data or config.training.dtype_policy != planned.common.training.dtype_policy:
        raise ValueError("Saved preprocessing, data split, sample limits or precision differs from the frozen manifest.")
    return {"manifest_hash": manifest["manifest_hash"], "run_id": continual.experiment_run_id,
            "condition": matching[0]["condition"], "phase": "confirmation", "source": source}


def _load_arrays(config: Config, route_name: str, route: dict) -> tuple:
    """Reconstruct the exact original split/caps before selecting seen classes.

    Re-filtering original classes before splitting would change validation
    membership. The full saved schedule is loaded first, using the learner's
    existing preprocessing, cap RNG, padding and introduction-label conversion.

    Args:
        config (Config): Validated configuration tree, with the data, model and experiment
            settings consumed by this operation.
        route_name (str): Saved route identity used to resolve the matching configuration
            and data adapter.
        route (dict): Complete saved route settings mapping, including its data source and
            extensions.

    Returns:
        arrays (tuple): Tuple of original training/validation/test image and sparse label
            arrays from the saved loader protocol.

    Raises:
        ValueError: If the saved loader, preprocessing or schedule is unsupported.
        OSError: If a required local dataset cannot be read or downloaded by the loader.
    """

    from common.dataloader import get_datasets
    from common.learner import _load_continual_arrays

    # Controlled gist runs use CIFAR geometry but never load CIFAR pixels.
    if route_name == "gist_memory" and route.get("data_source") == "controlled":
        from gist_memory.config import RouteSettings
        from gist_memory.data import get_controlled_loader

        loader = get_controlled_loader(RouteSettings(**route))
    # Standard datasets retain their existing common loader selection.
    else:
        loader, _ = get_datasets(config)
    dataset, seed = config.dataset, effective_seed(config)
    order, _ = _resolved_schedule(config)
    arrays, _ = _load_continual_arrays(
        loader, order, dataset.return_features,
        {"preprocess": dataset.preprocess, "onehot_labels": dataset.onehot_labels,
         "validation_ratio": dataset.validation_ratio, "features_path": dataset.features_path,
         "seed": seed},
        dataset.max_train_samples, dataset.max_val_samples, dataset.pad, seed,
    )
    return arrays


def _seen_arrays(images: object, labels: object, seen: dict) -> tuple[np.ndarray, np.ndarray]:
    """Select introduced labels and use the wrapper's saved dense classifier map.

    Args:
        images (object): Numeric sample-major images in the configured model-input scale,
            normally float32 NHWC values in [-1, 1].
        labels (object): Sparse integer label vector aligned with the image rows; the label
            convention for this operation is described above.
        seen (dict): Saved mapping from original integer labels to dense seen-class
            classifier columns.

    Returns:
        arrays (tuple[np.ndarray, np.ndarray]): (images, labels): selected image ndarray
            with grayscale channels added as needed and dense int64 labels.

    Raises:
        ValueError: If the requested held-out split is absent or contains no checkpoint
            classes.
    """

    from common.dataloader import get_dataset

    # Never replace an absent held-out split with training or test examples.
    if images is None or labels is None:
        raise ValueError("The requested held-out split is absent from the saved data protocol.")
    x, y = np.asarray(images), np.asarray(labels).reshape(-1)
    selected = np.isin(y, list(seen))
    # A valid evaluation needs at least one permitted seen-class example.
    if not np.any(selected):
        raise ValueError("The held-out split contains no classes represented by the checkpoint.")
    mapped = np.asarray([seen[int(label)] for label in y[selected]], dtype=np.int64)
    # The shared dataset API also adds grayscale channels as during training.
    dataset = get_dataset(x[selected], mapped, batch_size=128, shuffle_buffer=0, drop_remainder=False)
    batches = list(dataset.as_numpy_iterator())
    return np.concatenate([batch[0] for batch in batches]), np.concatenate([batch[1] for batch in batches])


def evaluate_saved_checkpoint(
    config_path: str | Path,
    *,
    split: str = "validation",
    settings_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """Reload one completed checkpoint, evaluate it and write an exclusive report.

    No fit/train API is called. Validation settings can be explored explicitly;
    test settings must match the authenticated frozen design. Positive saved
    calibration_fraction fits temperature on the original validation arrays
    when evaluating test, with no test labels used for fitting.

    Args:
        config_path (str | Path): Saved common configuration path containing the model
            architecture, seen-class mapping and checkpoint location.
        split (str): Declared data split; supported training, validation or test access is
            constrained by this operation.
        settings_path (str | Path | None): Optional YAML path overriding inference settings;
            test access still requires agreement with the frozen design.
        output_path (str | Path | None): Optional exclusive JSON report destination; None
            chooses a report beside the saved checkpoint configuration.

    Returns:
        report (dict): JSON-compatible dict of checkpoint identity, held-out outcomes,
            calibration and measured inference work; also writes an exclusive report file.

    Raises:
        FileExistsError: If the report destination already exists.
        OSError: If checkpoint/configuration inputs cannot be read or output cannot be
            written.
        ValueError: If checkpoint, split, calibration or confirmation contracts are invalid.
    """

    from common.model import get_model
    from semantic_consolidation.controller import weight_digest

    path = Path(config_path).resolve()
    project = load_config(path)
    # This entry point is deliberately limited to saved continual route images.
    if split not in ("validation", "test") or project.training.task != "continual":
        raise ValueError("Use a saved continual config and split validation or test.")
    # Loading an untrained factory model would not be checkpoint evaluation.
    if not project.model.weights_path:
        raise ValueError("Saved config.model.weights_path is required; checkpoint evaluation never trains a model.")
    # Route validation assumes sparse unpadded uint8-derived diffusion images.
    if project.dataset.onehot_labels or project.dataset.return_features or project.dataset.preprocess != "fixed-standardize":
        raise ValueError("Checkpoint evaluation supports the routes' sparse raw-image fixed-standardize protocol.")
    route_name, route = _route_settings(project)
    settings = _evaluation_settings(project, settings_path, split)
    confirmation = _confirmation_contract(project, settings) if split == "test" else None
    destination = Path(output_path).resolve() if output_path is not None else path.parent / f"checkpoint.{split}.ensemble.json"
    # Preserve earlier outcomes instead of silently replacing them on a rerun.
    if destination.exists():
        raise FileExistsError(f"Evaluation report already exists: {destination}")
    configure_runtime(effective_seed(project), project.training.dtype_policy, project.training.deterministic_ops)
    arrays = _load_arrays(project, route_name, route)
    bundle = get_model(project)
    wrapper = bundle["generative_model"] if isinstance(bundle, dict) else bundle
    seen = dict(wrapper.seen_classes)
    # A final saved route classifier must retain its introduction-label map.
    if not seen or sorted(seen.values()) != list(range(wrapper.network.num_classes)):
        raise ValueError("The checkpoint lacks a complete dense seen-class map.")
    order, _ = _resolved_schedule(project)
    # Introduction IDs are positions in the saved original class schedule.
    if any(isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(order) for label in seen):
        raise ValueError("Checkpoint seen classes are outside the saved introduction schedule.")
    offset = 2 if split == "validation" else 4
    samples, labels = _seen_arrays(arrays[offset], arrays[offset + 1], seen)
    calibration = {}
    # Test calibration reuses only independently held-out validation arrays.
    if split == "test" and settings.calibration_fraction > 0.:
        cx, cy = _seen_arrays(arrays[2], arrays[3], seen)
        calibration_indices, _ = _validation_partition(cy, settings.calibration_fraction, settings.seed)
        calibration = {"calibration_samples": cx[calibration_indices], "calibration_labels": cy[calibration_indices],
                       "calibration_split": "validation"}
    del arrays
    before = weight_digest(wrapper.network.weights)
    old_class_count = None
    # Include the prespecified old/new decomposition only when the experimental observer is enabled.
    if route.get("experimental", {}).get("enabled", False):
        _, groups = _resolved_schedule(project)
        old_class_count = len(order) - len(groups[-1])
    result = evaluate_checkpoint(wrapper, samples, labels, settings, split=split,
                                 old_class_count=old_class_count, **calibration)
    unchanged = before == weight_digest(wrapper.network.weights)
    # Inference-only evaluation must not alter the restored checkpoint.
    if not unchanged:
        raise RuntimeError("Checkpoint evaluation changed model weights.")
    result.update({
        "config_path": str(path), "checkpoint_path": str(project.model.weights_path),
        "checkpoint_weight_digest": before, "checkpoint_weights_unchanged": unchanged,
        "route": route_name, "data_source": route.get("data_source", "standard"),
        "original_class_to_classifier_id": {str(order[int(label)]): int(index) for label, index in seen.items()},
        "confirmation": confirmation, "output_path": str(destination),
        "calibration_protocol": "fixed stratified calibration_fraction of original training-stream validation; never test-fitted",
        "comparison_scope": "inference treatments on one saved checkpoint; final examples are not independent training replications",
    })
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    return result


def main(argv: list[str] | None = None) -> None:
    """Run the shared saved-checkpoint evaluation command for either route.

    Args:
        argv (list[str] | None): Command-line strings; None reads the current process
            arguments.

    Returns:
        completed (None): None; parses command-line arguments, executes the requested
            operation and prints its artifact location.

    Raises:
        SystemExit: If arguments are invalid or help is requested.
        OSError: If requested input/output artifacts cannot be accessed.
        ValueError: If the selected configuration or experiment contract is invalid.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Saved common config.yaml with model weights")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--settings", type=Path, help="Evaluation YAML or saved route.settings.yaml")
    parser.add_argument("--output", type=Path, help="New JSON output file; existing files are never replaced")
    args = parser.parse_args(argv)
    result = evaluate_saved_checkpoint(args.config, split=args.split, settings_path=args.settings, output_path=args.output)
    print(f"Saved {args.split} checkpoint comparison: {result['output_path']}")


# Execute only when invoked as a module or script.
if __name__ == "__main__":
    main()
