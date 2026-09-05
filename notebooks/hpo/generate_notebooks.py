"""Build the task/model matrix of thin configuration-driven HPO notebooks.

NOTEBOOKS maps task names to model names and (epochs, rationale) pairs; ROOT
is this generator's directory. Builders return nbformat-4 dictionaries with
Python source and markdown. They do not execute the generated experiments.
The main function verifies NOTEBOOKS against common.hpo.SEARCH_SPACES and
writes one notebook per entry, replacing existing files at those paths.

Generated experiments use CIFAR10, 30 trials, seed 42, and results/hpo by
default. Each setup cell exposes these values for editing. The result cell
displays a Pareto set for multiple objectives or the best value/parameters for
a single objective. Direct script execution writes notebooks; importing this
module only defines constants and builder functions.
"""

from __future__ import annotations

import json

from pathlib import Path


ROOT = Path(__file__).resolve().parent


NOTEBOOKS = {
    "generation": {
        "diffusion_transformer": (
            50, 
            "Tune conditional image synthesis with a patch-based diffusion transformer. "
            "The space varies transformer capacity and diffusion optimization while "
            "preserving valid patch and attention dimensions."
        ), 
        "dit_decoder": (
            50, 
            "Tune conditional image synthesis with a standalone DiT decoder. The space "
            "keeps decoder conditioning and routing compatible and varies only capacity "
            "and diffusion training choices."
        ),
        "dit_encoder_decoder": (
            50, 
            "Tune conditional image synthesis with a DiT encoder-decoder. The space uses "
            "validated encoder-decoder topology templates so capacity can change without "
            "breaking routed feature shapes."
        ), 
        "unet": (
            50, 
            "Tune conditional image synthesis with a convolutional U-Net. The space "
            "balances multiscale width, depth, bottleneck capacity, and diffusion "
            "optimization for a compact image-model baseline."
        ), 
        "vae": (
            30, 
            "Tune flattened-image generation with a variational autoencoder. The space "
            "balances latent compression, reconstruction quality, KL regularization, and "
            "hidden capacity while fixing the output domain to the dataset representation."
        ), 
    },
    "joint": {
        "dit_classifier": (
            50, 
            "Tune joint diffusion generation and classification with a DiT classifier. "
            "The space balances generator capacity, classifier features, and the two loss "
            "scales while retaining classifier-free conditioning."
        ), 
        "dit_encoder_decoder_classifier": (
            50, 
            "Tune joint generation and classification with a DiT encoder-decoder "
            "classifier. Validated routing templates keep encoder, decoder, and classifier "
            "feature shapes aligned during the search."
        ), 
        "unet_classifier": (
            50, 
            "Tune joint generation and classification with a U-Net classifier. The space "
            "varies multiscale capacity and classifier depth together with the "
            "classification-to-diffusion loss balance."
        ), 
        "vae_classifier": (
            30, 
            "Tune flattened-image reconstruction, latent regularization, and classification with "
            "a VAE classifier. The space couples latent capacity, beta, and classification "
            "weight under a fixed feature representation."
        )
    },
    "classification": {
        "cnn": (
            30, 
            "Tune the project's convolutional image-classification baseline. The compact "
            "space covers staged capacity, regularization, pooling, and optimization through "
            "the public model factory."
        ), 
        "dnn": (
            30, 
            "Tune classification from normalized flattened raw pixels with the project's "
            "DNN baseline. The space compares compact hidden templates, activation, "
            "normalization, dropout, and optimization choices."
        ), 
        "pretrained": (
            30, 
            "Tune transfer learning with the pretrained Xception classifier. The space "
            "balances fine-tuning depth, head dropout, and learning rate to control "
            "adaptation without erasing pretrained features."
        )
    },
    "continual": {
        "cnn": (
            20,
            "Tune the convolutional classifier under sequential, cumulative, "
            "or reservoir-replay continual protocols without a generator."
        ),
        "dnn": (
            20,
            "Tune the dense classifier under sequential, cumulative, or "
            "reservoir-replay continual protocols on flattened inputs."
        ),
        "pretrained": (
            20,
            "Tune Xception transfer learning under sequential, cumulative, or "
            "reservoir-replay continual protocols for CIFAR images."
        ),
        "diffusion_transformer": (
            20, 
            "Tune a conditional DiffusionTransformer replay buffer for continual "
            "classification. The objective measures retention and plasticity while replay "
            "and generator-update budgets use the shared candidate set."
        ), 
        "dit_decoder": (
            20, 
            "Tune a standalone DiTDecoder replay buffer for continual classification. The "
            "space preserves decoder-compatible conditioning while jointly evaluating "
            "replay quality and the shared compute-budget choices."
        ),
        "dit_encoder_decoder": (
            20, 
            "Tune a DiTEncoderDecoder replay buffer for continual classification. The "
            "space uses shape-safe routing templates and optimizes conditional replay for "
            "average accuracy across experiences."
        ),
        "unet": (
            20, 
            "Tune a conditional U-Net replay buffer for continual classification. The "
            "space balances multiscale capacity and replay fidelity while drawing replay "
            "and generator-update budgets from a common candidate set."
        ),
        "vae": (
            20, 
            "Tune a conditional VAE replay buffer over dataset features. The objective "
            "balances latent compression and replay fidelity against continual retention "
            "while also selecting the generated samples per class."
        ),
        "dit_classifier": (
            20, 
            "Tune a DiTClassifier as a conditional replay generator for continual "
            "classification. The space can use its auxiliary classifier signal while the "
            "continual objective remains average performance across experiences."
        ),
        "dit_encoder_decoder_classifier": (
            20, 
            "Tune a DiTEncoderDecoderClassifier replay buffer for continual learning. "
            "Validated feature routes keep all branches compatible while the search "
            "balances replay fidelity and auxiliary classification."
        ), 
        "unet_classifier": (
            20, 
            "Tune a U-Net classifier replay buffer for continual learning. The space "
            "balances conditional synthesis, classifier depth, and retention while replay "
            "budgets use the shared candidate set."
        )
    }
}


def markdown_cell(task: str, model: str, rationale: str) -> dict:
    """Build a task/model overview as a fresh markdown cell mapping.

    Model underscores become title-case spaces. The joint task receives the
    explicit heading Generation + Classification; all other task strings use
    title case. The rationale is inserted verbatim, without escaping markdown.

    Args:
        task (str): HPO task key such as generation, joint, or continual.
        model (str): Model-family key used only to form the displayed heading.
        rationale (str): Experiment motivation placed beneath the heading.

    Returns:
        dict[str, object]: Markdown cell with id='overview', empty metadata, and
        a list of newline-preserving source strings. No file is written.
    """

    title = model.replace("_", " ").title()
    # Label joint optimization by both tasks; title-case every other task key.
    task_title = "Generation + Classification" \
                if task == "joint" else task.title()

    return {
        "cell_type": "markdown", 
        "id": "overview", 
        "metadata": {}, 
        "source": [
            f"# {task_title}: {title}\n", 
            "\n", 
            f"{rationale}\n"
        ]
    }


def code_cell(source: str, cell_id: str) -> dict:
    """Package Python source as a fresh, unexecuted notebook code cell.

    This function does not parse or run source. splitlines(keepends=True) preserves
    the caller's line endings and missing final newline. Empty source becomes an
    empty list, and the caller supplies the stable cell identifier.

    Args:
        source (str): Python code to store verbatim as source lines.
        cell_id (str): Cell identifier, expected to be unique within its notebook.

    Returns:
        dict[str, object]: Code cell with the supplied id, empty metadata/outputs,
        execution_count=None, and a list-valued source field.
    """

    return {
        "cell_type": "code", 
        "execution_count": None, 
        "id": cell_id, 
        "metadata": {}, 
        "outputs": [], 
        "source": source.splitlines(keepends=True)
    }


def make_notebook(task: str, model: str, 
                epochs: int, rationale: str) -> dict:
    """Build a five-cell notebook that inspects and executes one HPO study.

    The overview is followed by setup, search-space inspection, run_hpo invocation,
    and results. Setup fixes DATASET='CIFAR10', N_TRIALS=30, SEED=42, and
    RESULTS_PATH='results/hpo', leaving them visible for subsequent notebook edits.
    The supplied epochs is written into setup. No validation or training occurs
    while building; search-space keys are resolved only when cells are executed.

    Args:
        task (str): HPO task key interpolated into the setup cell and heading.
        model (str): Model key accepted for that task by common.hpo.SEARCH_SPACES.
        epochs (int): Requested training epochs per trial, inserted as Python code.
        rationale (str): Markdown explanation passed to the overview builder.

    Returns:
        dict[str, object]: nbformat=4, nbformat_minor=5 document with five fresh
        cells and Python 3.10/tf_env kernel metadata. Results show best_trials when
        multiple directions exist, otherwise best_value and best_params.
    """

    setup = f'''"""Configuration-driven HPO experiment using the shared task/model APIs.

TASK and MODEL select a constrained search space; DATASET names the input
dataset. The setup defaults to CIFAR10, 30 trials, seed 42, and results/hpo;
EPOCHS is supplied by the notebook generator. Edit these visible constants
before executing the study. The run cell constructs and trains trial models
through Config, may download data, and writes configured study/report artifacts.

Outputs are an Optuna study, trial dataframe, and best-trial view. Multiple
objectives expose the Pareto set; a single objective exposes its best scalar
value and parameters. Defining the setup does not itself launch training.
"""

from common.hpo import SEARCH_SPACES, run_hpo

TASK = "{task}"
MODEL = "{model}"
DATASET = "CIFAR10"
N_TRIALS = 30
EPOCHS = {epochs}
SEED = 42
RESULTS_PATH = "results/hpo"
'''
    inspect_space = '''# Inspect the constrained, task-specific search space before running.
SEARCH_SPACES[TASK][MODEL]
'''
    run = '''study = run_hpo(
    task=TASK,
    model_name=MODEL,
    dataset_name=DATASET,
    n_trials=N_TRIALS,
    epochs=EPOCHS,
    seed=SEED,
    results_path=RESULTS_PATH,
)
'''
    results = '''trials = study.trials_dataframe()
# Multiple objectives expose the Pareto set; a single objective exposes its best trial.
best = (
    study.best_trials
    if len(study.directions) > 1
    else {"value": study.best_value, "params": study.best_params}
)
trials, best, RESULTS_PATH
'''

    return {
        "cells": [
            markdown_cell(task, model, rationale), 
            code_cell(setup, "setup"), 
            code_cell(inspect_space, "inspect-space"), 
            code_cell(run, "run-study"), 
            code_cell(results, "show-results")
        ], 
        "metadata": {
            "kernelspec": {
                "display_name": "Python (tf_env)", 
                "language": "python", 
                "name": "tf_env"
            }, 
            "language_info": {
                "name": "python", 
                "version": "3.10"
            }, 
        }, 
        "nbformat": 4, 
        "nbformat_minor": 5
    }


def main() -> None:
    """Write every declared HPO notebook beneath its task directory.

    The declared task/model set must match SEARCH_SPACES before any output is
    written. Missing task directories are created. Existing notebooks at the
    generated paths are replaced by new unexecuted documents, including their
    metadata and outputs. Other paths are not enumerated for deletion.

    Args:
        None. ROOT, NOTEBOOKS, and imported SEARCH_SPACES determine all output paths.

    Returns:
        None: UTF-8 JSON notebooks with one-space indentation and final newlines
        have been written; the written count equals the supported task/model count.

    Raises:
        RuntimeError: The declared search-space matrix or final count disagrees.
        OSError: A directory cannot be created or an output notebook cannot be written.
    """

    from common.hpo import SEARCH_SPACES

    declared = {
        (task, model)
        for task, models in NOTEBOOKS.items()
        for model in models
    }
    supported = {
        (task, model)
        for task, models in SEARCH_SPACES.items()
        for model in models
    }
    # Reject a missing or extra task/model entry before creating output files.
    if declared != supported:
        raise RuntimeError(
            "Notebook/search-space mismatch: missing="
            f"{sorted(supported - declared)}, extra={sorted(declared - supported)}"
        )

    count = 0
    for task, models in NOTEBOOKS.items():
        task_dir = ROOT / task
        task_dir.mkdir(parents=True, exist_ok=True)
        for model, (epochs, rationale) in models.items():
            path = task_dir / f"{model}.ipynb"
            notebook = make_notebook(task, model, epochs, rationale)
            path.write_text(
                json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", 
                encoding="utf-8"
            )
            count += 1

    # Reject a completed write count that does not cover the supported matrix.
    if count != len(supported):
        raise RuntimeError(
            f"Expected {len(supported)} notebooks, generated {count}."
        )


# Generate the notebook matrix when this helper is invoked directly.
if __name__ == "__main__":
    main()
