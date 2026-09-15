"""End-to-end route-one execution through the shared project pipeline APIs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from common.config import load_config
from semantic_consolidation.config import (
    RouteConfig, load_route_config, save_route_settings, validate_route_config,
)


def run(config: RouteConfig | str | Path) -> dict[str, object]:
    """Train a complete continual stream and save common and route artifacts.

    Args:
        config (RouteConfig | str | Path): RouteConfig or route YAML input path; paths are
            loaded and validated before data/model construction.

    Returns:
        result (dict[str, object]): Dict with model bundle, histories, common evaluations,
            resolved results_path and semantic/experimental records.

    Raises:
        OSError: If configuration, data or result artifacts cannot be read/written.
        ValueError: If configuration, source/manifest, model or data contracts fail.
        RuntimeError: If training budgets or frozen-state invariants fail.
    """

    from common.dataloader import get_datasets
    from common.model import get_model
    from common.runtime import configure_runtime, effective_seed
    from common.train import report, train_model
    from semantic_consolidation.controller import RouteController
    from semantic_consolidation.model import adapt_model
    from semantic_consolidation.provenance import source_provenance, save_provenance
    from semantic_consolidation.study import validate_planned_config

    # Load path inputs through the same validated configuration API used by the CLI.
    if not isinstance(config, RouteConfig):
        config = load_route_config(config)
    validate_route_config(config)
    validate_planned_config(config)
    project, settings = config.common, config.route
    seed = effective_seed(project)
    # An omitted semantic seed inherits the explicit continual master seed.
    if settings.seed is None:
        settings.seed = seed
    # Shared immutable/resolved Config artifacts retain the full intervention.
    project.hpo["semantic_consolidation"] = asdict(settings)
    configure_runtime(seed, project.training.dtype_policy, project.training.deterministic_ops)
    provenance = source_provenance()
    trainset, valset = get_datasets(project)
    bundle = get_model(project)
    controller = RouteController(settings)
    extensions = None
    # Construct the scheduling controller only when an extension is requested.
    if settings.extensions:
        from semantic_consolidation.extensions import ExtensionController
        extensions = ExtensionController(project, settings.extensions, settings.seed)
    bundle["generative_model"] = adapt_model(bundle["generative_model"], controller, extensions=extensions)
    observer = None
    # Attach held-out diagnostics only for an explicitly enabled observation protocol.
    if settings.experimental.get("enabled", False):
        from semantic_consolidation.experimental import ExperimentalController
        observer = ExperimentalController(project, settings.experimental, settings.seed, bundle=bundle)
        object.__setattr__(bundle["generative_model"], "experimental_controller", observer)
    try:
        history = train_model(project, bundle, trainset, valset=valset)
    finally:
        # Release the background monitor even if training fails.
        if observer is not None:
            observer.close()
    bundle["continual_details"]["semantic_consolidation"] = controller.records
    result_path = Path(project.training.results_path)
    save_provenance(provenance, result_path)
    save_route_settings(settings, result_path / "route.settings.yaml")
    controller.save(result_path)
    # Save optional schedule and replay evidence alongside the ordinary route records.
    if extensions is not None:
        extensions.save(result_path)
        bundle["continual_details"]["section10_extensions"] = extensions.records
    evaluations = report(project, history, bundle, trainset, valset=valset)
    # Export completed held-out observations after the shared reporting path.
    if observer is not None:
        observer.save(result_path)
        bundle["continual_details"]["section11_experimental"] = observer.records
    return {
        "model": bundle,
        "history": history,
        "evaluations": evaluations,
        "results_path": str(result_path),
        "route_records": controller.records,
        "experimental_records": observer.records if observer is not None else [],
    }


def load_inference_model(config_path: str | Path) -> object:
    """Reload a saved common config and its ordinary unmodulated joint model.

    Modulators and predictors are not used for final inference. This loader is
    intentionally not a continual-training resume operation.

    Args:
        config_path (str | Path): Saved common configuration path containing the model
            architecture, seen-class mapping and checkpoint location.

    Returns:
        model (object): Ordinary unmodulated joint wrapper loaded through common.model; this
            does not restore phase/controller state for resumption.

    Raises:
        ValueError: If saved weights_path is absent or checkpoint topology is incompatible.
        OSError: If configuration or checkpoint files cannot be read.
    """

    from common.model import get_model

    config = load_config(config_path)
    # The saved common configuration must include a model weights path.
    if config.model.weights_path is None:
        raise ValueError("The saved common configuration must include a model weights path.")
    result = get_model(config)
    return result["generative_model"] if isinstance(result, dict) else result


def main(argv: list[str] | None = None) -> None:
    """Run ``python -m semantic_consolidation --config FILE [--dry-run]``.

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

    parser = argparse.ArgumentParser(description="Semantic modulation consolidation for joint continual learning")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print resolved settings without training")
    args = parser.parse_args(argv)
    config = load_route_config(args.config)
    # A dry run validates and prints the complete design without loading data or training.
    if args.dry_run:
        data = {"common": asdict(config.common), "route": asdict(config.route)}
        data["route"]["noise_levels"] = list(config.route.noise_levels)
        print(yaml.safe_dump(data, sort_keys=True))
        return
    result = run(config)
    print(f"Completed semantic consolidation stream: {result['results_path']}")
