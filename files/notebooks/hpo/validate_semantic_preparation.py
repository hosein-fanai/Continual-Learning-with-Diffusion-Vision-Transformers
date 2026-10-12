"""Validate the prepared semantic notebook and every anchor on a remote container.

This checks syntax and real configuration contracts without training a model.
It is intentionally not executed on the laptop. Numerical route tests and GPU
capacity checks are separate steps in SEMANTIC_HPO_CAMPAIGN_20261012.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = Path(__file__).with_name("Semantic_Consolidation_HPO_Runner.ipynb")


def validate() -> dict:
    """Compile all cells and build every queued recipe through the real adapter."""

    # Preparation computation belongs on an authorized online Linux container.
    if platform.system() != "Linux" or not (
        Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()
    ):
        raise RuntimeError("Run preparation validation only in the supplied online container.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import optuna

    from common.semantic_hpo import build_semantic_config, normalize_semantic_profile, validate_semantic_search
    from common.semantic_hpo_candidates import semantic_candidate_profile, semantic_initial_trials


    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    identifiers = [cell["id"] for cell in notebook["cells"]]
    # Stable cell names must identify exactly one configuration source.
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate notebook cell IDs.")
    compiled = {}
    for cell in notebook["cells"]:
        # Only code cells have executable syntax or execution artifacts.
        if cell["cell_type"] == "code":
            compiled[cell["id"]] = compile("".join(cell["source"]), f"{NOTEBOOK.name}:{cell['id']}", "exec")
            # The delivered notebook must not contain stale experimental claims.
            if cell["execution_count"] is not None or cell["outputs"]:
                raise ValueError("Prepared notebook contains execution artifacts.")
    options = {}
    exec(compiled["settings"], options)
    profile = normalize_semantic_profile(semantic_candidate_profile())
    overrides = options["SEARCH_SPACE_OVERRIDES"]
    validate_semantic_search(profile, overrides)
    anchors = semantic_initial_trials(profile)
    pairs = []
    for number, params in enumerate(anchors):
        trial = optuna.trial.FixedTrial(params, number=number)
        config = build_semantic_config(
            trial, "cifar10", 10, "/tmp/semantic-preparation-unused", profile, 
            search_space_overrides=overrides, 
            objective_metrics=["retention_balanced_accuracy"], objective_directions=["maximize"], 
            seed=42
        )
        pairs.append({
            "recipe": params["base_recipe"], "budget": params["semantic_budget"], 
            "old_replay": config.continually_learn.replay_old_examples
        })
    return {
        "status": "passed", "scope": "cell syntax and all native candidate anchor contracts; no model execution", 
        "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(), 
        "compiled_code_cells": len(compiled), "validated_anchors": pairs, 
        "fixed_recipe_sha256": profile["fixed_recipe_sha256"]
    }


def main() -> None:
    """Write a receipt only after all preparation checks succeed remotely."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/semantic-preparation-validation.json"))
    options = parser.parse_args()
    report = validate()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


# This entry point checks preparation and cannot start a campaign.
if __name__ == "__main__":
    main()
