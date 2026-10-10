"""Maintain semantic-only HPO using the generation runner's remote bootstrap.

Run this authoring utility only on an authorized online container. It writes an
unexecuted notebook and does not load models, allocate trials or start a campaign.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _cell(kind: str, cell_id: str, source: str) -> dict:
    """Return one clean notebook cell with a stable identifier."""

    cell = {"cell_type": kind, "id": cell_id, "metadata": {}, "source": source.splitlines(keepends=True)}
    # Markdown has no execution state.
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def make_notebook() -> dict:
    """Build the maintained semantic runner without executing its control cells."""

    template = json.loads((ROOT / "DiT_Generation_HPO_Runner.ipynb").read_text(encoding="utf-8"))
    setup = deepcopy(next(cell for cell in template["cells"] if cell["id"] == "remote-guard"))
    setup["source"] = [line.replace("from common.dit_hpo_runner import (", "from common.semantic_hpo_runner import (") for line in setup["source"]]
    setup["execution_count"] = None
    setup["outputs"] = []
    cells = [
        _cell("markdown", "overview", """# Semantic consolidation HPO in continual learning

Optimize **semantic acquisition and consolidation** around one supplied native
**DiT CLF architecture and complete continual-learning recipe**. Architecture,
task stream, replay, teachers/distillation, joint optimizer, joint batch size,
epochs and diffusion process stay fixed. This isolates the useful semantic
settings without letting architecture or replay changes explain the result.

One trial runs the **entire sequential task stream**, using the native semantic
runner. Maximize held-out training **validation final average accuracy**. CIFAR-10
and CIFAR-100 use their supplied class groups; this notebook does not replace
them with an invented schedule. The official test set does not select trials.

The workflow matches `DiT_Generation_HPO_Runner.ipynb`: verified remote setup,
sealed inputs, admitted isolated workers, persistent Optuna search, TensorBoard,
resumption, frozen finalists and paired fresh-seed confirmations. The full field
catalog below is exhaustive for the implemented `RouteSettings` interface;
**200 TPE trials do not exhaust the continuous or Cartesian search space**.

Read [SEMANTIC_HPO_GUIDE.md](SEMANTIC_HPO_GUIDE.md) for the full search design,
mechanistic controls, resource plan, duration formulas and limitations. All code
cells must run in an authorized online container. This notebook is unexecuted.
"""), 
        _cell("markdown", "runtime-notes", """## Remote runtime and measured admission

Use the supplied remote checkout with **TensorFlow 2.20.0 / Keras 3.11.2** and its
verified kernel. The inherited coordinator checks the checkout/runtime and hides
its GPUs; training occurs only in admitted child processes. All computation,
including preparation and reports, belongs on the online container.

This notebook requires the **matching complete checkout**, including the new
`common/semantic_hpo.py`, `common/semantic_hpo_runner.py`, their shared HPO/worker
changes, and `semantic_consolidation` integration. Upload/synchronize those
changes to an isolated remote checkout before setup. A notebook uploaded alone
is insufficient. While these changes are unpublished, the inherited bootstrap's
GitHub `main` clone does not contain them; use the prepared matching checkout
rather than assuming a fresh public clone supplies this implementation.

Start with **one worker on one A100 80 GB or H100 80 GB**, requesting a **24 GiB
TensorFlow cap**. This request is an initial measurement setting, not a capacity
certificate. The allocator also reserves worker overhead and device headroom,
checks live ownership and preserves existing jobs. Unknown ownership blocks
admission. Student, previous snapshot, semantic targets/modulators, replay and
augmentation views all affect memory. Measure the largest configured case and
late tasks before increasing concurrency. Independent trials can share a GPU
only after measured admission; tasks inside one trial remain sequential.

The generation runner's 51-worker result does not establish semantic-consolidation
capacity. GPUs do not pool their VRAM. `EXPERIMENT_HOURS=None` imposes no global
cutoff. Optional explicit budgets survive restarts and do not stop rental billing.
"""), 
        setup, 
        _cell("markdown", "scope-notes", """## Complete search registry before supplying model inputs

The executable table below is generated from `common.semantic_hpo.SEARCH_SPACE`
and `FIELD_CATALOG`. Every native route field is classified as sampled,
conditional, fixed or a separate ablation. Unlisted architecture/CL controls
cannot become search axes through an override.

Noise indices must be below the supplied model's fixed diffusion horizon. The
normalizer filters unsupported index choices; it does not change that horizon.
`noise_levels` uses comma-separated categorical encodings that resolve to exact
tuples. A clean view has index zero. Reliability and its floor are inactive for
an entirely clean alignment; view count is sampled only with `tmcl` augmentation.
"""), 
        _cell("code", "search-catalog", """from IPython.display import Markdown

from common.semantic_hpo import FIELD_CATALOG, SEARCH_SPACE


catalog_rows = ["| Field | Classification | Domain or fixed source | Condition | Purpose |", "| --- | --- | --- | --- | --- |"]
for field_name, specification in FIELD_CATALOG.items():
    domain = SEARCH_SPACE.get(field_name, "Supplied route constant; see purpose")
    catalog_rows.append(
        "| " + " | ".join(str(value).replace("|", "/").replace("\\n", " ") for value in (
            field_name, specification["role"], domain, specification["condition"], specification["reason"]
        )) + " |"
    )
display(Markdown("\\n".join(catalog_rows)))
print("Native route fields documented:", len(FIELD_CATALOG))
print("Potential semantic search axes:", len(SEARCH_SPACE))
"""), 
        _cell("markdown", "input-notes", """## Supply the fixed architecture and continual-learning recipe

`STUDENT_CONFIG` is a native `common.config.Config`, mapping, or remote YAML path.
It includes the **whole** scientific recipe: dataset/split, task order, student
architecture, wrapper and classifier projection, replay pools, active teachers
and distillation, joint optimizer, joint batch size and per-task joint epochs.
These constants are sealed. Supply the intended compatible recipe rather than
the architecture alone. The adapter rejects incompatible inputs instead of
silently rewriting their scientific settings.

Use a fresh native DiT classifier with V1 `diffusion_classifier`, a positive
classifier semantic projection, CFG enabled, standardized preprocessing, raw
network evaluation and EMA disabled. The audited semantic route uses float32,
complete joint phases (`patience=0`), no real replay buffer, and `fixed_total` or
`match_current` replay. Retained old modulators require enough generated old
examples for positive pairs. HPO requires `experiment_phase="development"`,
`validation_source="split"`, and a nonzero training holdout. The native route
currently starts fresh; a trained student checkpoint is not an HPO initializer.

An output config from another HPO profile is usable when it satisfies these
semantic-route contracts. Fixed specialist artifacts belong in the native
`continually_learn.specialist_teacher_descriptors` field. Their identities and
all supported native teacher/KD settings remain fixed to the supplied input.

`ROUTE_SETTINGS` accepts a mapping, `RouteSettings`, or a standalone route-settings
YAML (raw route fields or a `route:` mapping). The suggested fixed mechanism is
learned acquisition, contrastive objective, semantic-only consolidation and
retained modulators. Phase settings listed in the registry are searched; fixed
mechanistic controls and extensions retain their input values. No model is
invented by this template. Fill the native input before running the plan cell.
"""), 
        _cell("code", "model-inputs", """STUDENT_CONFIG = None  # Native complete DiT CLF continual Config or remote YAML.
ROUTE_SETTINGS = {
    "condition": "learned", 
    "acquisition_objective": "contrastive", 
    "consolidation_scope": "semantic", 
    "retain_modulators": True, 
    "extensions": {}, 
    "experimental": {}
}
"""), 
        _cell("markdown", "search-design", """## Search budget and statistical scope

Use **12 successful full-stream trials** as a timing and failure pilot, review at
**40**, and target **200** successful trials. TPE has a 40-observation startup
setting; the pilot is exploratory, not evidence of convergence. `MAX_ATTEMPTS=800`
also counts failed and OOM-pruned attempts. Search performance pruning is disabled:
early-task scores cannot reliably stand in for final forgetting/accuracy.

The semantic budget ranges vary added optimizer updates. Thus the selected result
answers which semantic recipe works best in this allowed budget range, not which
mechanism is best at equal compute. Inspect full-stream duration and phase timing
alongside accuracy; a separate measured time-matched control is needed for a
claim beyond extra training. The baseline, random/identity gates, no consolidation,
feature distillation and extra/time-matched joint treatments are **separate
ablations**, not categorical ways for TPE to turn the mechanism off.

Freeze the top three distinct recipes and rerun each at seeds 101/202/303 with
the same dataset split and task order. The full stream restarts from fresh model
initialization; winning trained weights are not reused. This is nine additional
streams. A scientific comparison against no semantic phases needs **three more
paired baseline streams**, separately run and budgeted. They do not run in the
finalist cell. Validation confirmations remain validation estimates; reserve an
untouched final test protocol after selection.
"""), 
        _cell("code", "settings", """RESULTS_PATH = "files/results/semantic_consolidation_hpo_v1"
GPU_IDS = [0]
CONCURRENT_TRIALS = 1
WORKER_GPU_MEMORY_LIMIT_MB = 24576
N_STARTUP_TRIALS = 40
SEARCH_SEED = 42
SEARCH_SPACE_OVERRIDES = {}  # Restrictions on supported semantic axes only.

PILOT_TARGET = 12
REVIEW_TARGET = 40
SEARCH_TARGET = 200
EXTENSION_TARGET = 400
RUN_EXTENSION = False
MAX_ATTEMPTS = 800
BATCH_TRIALS = max(4, 2 * CONCURRENT_TRIALS)
EXPERIMENT_HOURS = None
CONFIRMATION_RESERVE_HOURS = 0.0  # If adding a deadline, replace using pilot timings.
TOP_K = 3
CONFIRMATION_SEEDS = [101, 202, 303]
"""), 
        _cell("code", "plan", """from collections.abc import Mapping

from common.config import Config, load_config
from common.semantic_hpo import normalize_semantic_profile


# The full native scientific recipe is required before making a study.
if STUDENT_CONFIG is None:
    raise ValueError("Supply the native DiT CLF continual recipe in STUDENT_CONFIG first.")

# Preview supported input types without loading TensorFlow or a model.
if isinstance(STUDENT_CONFIG, Config):
    fixed_config = STUDENT_CONFIG
elif isinstance(STUDENT_CONFIG, Mapping):
    fixed_config = Config(**dict(STUDENT_CONFIG))
else:
    fixed_config = load_config(STUDENT_CONFIG)

DATASET = fixed_config.dataset.name
SEMANTIC_PROFILE = normalize_semantic_profile(
    {"student_config": STUDENT_CONFIG, "route_settings": ROUTE_SETTINGS}, DATASET, seed=SEARCH_SEED
)
EPOCHS = SEMANTIC_PROFILE["student_config"]["training"]["epochs"]
VALIDATION_RATIO = SEMANTIC_PROFILE["student_config"]["dataset"]["validation_ratio"]
plan = make_plan(
    CHECKOUT_ROOT, RESULTS_PATH, dataset_name=DATASET, epochs=EPOCHS, 
    n_startup_trials=N_STARTUP_TRIALS, concurrent_trials=CONCURRENT_TRIALS, 
    gpu_ids=GPU_IDS, pruning=None, validation_source="split", 
    validation_ratio=VALIDATION_RATIO, experiment_hours=EXPERIMENT_HOURS, 
    confirmation_reserve_hours=CONFIRMATION_RESERVE_HOURS, 
    search_space_overrides=SEARCH_SPACE_OVERRIDES, 
    worker_gpu_memory_limit_mb=WORKER_GPU_MEMORY_LIMIT_MB, 
    search_profile="semantic_consolidation_runner", semantic_profile=SEMANTIC_PROFILE, 
    seed=SEARCH_SEED
)
display(plan["hpo"]["semantic_profile"])
display(plan["identity"]["gpus"])
display(budget_summary(plan))
print("Persistent study:", plan["study_root"])
"""), 
        _cell("markdown", "plan-notes", """The displayed sealed profile is the reviewable record of architecture,
complete native CL constants, dataset/task seeds, resolved class groups and
semantic settings. Inspect it before search. Changing the native recipe,
mechanism, dataset split, distribution, code or package identity requires a fresh
results directory. Operational target counts and measured worker placement can
change while resuming an otherwise identical study.
"""), 
        _cell("code", "tensorboard", """TENSORBOARD_DIRECTORY = Path(plan["study_root"])
get_ipython().run_line_magic("load_ext", "tensorboard")
get_ipython().run_line_magic("tensorboard", f"--logdir {TENSORBOARD_DIRECTORY} --port 6006")
"""), 
        _cell("markdown", "pilot-notes", """## Pilot complete streams and estimate the required rental time

Run cells individually to inspect this pilot before the main target. Run All
continues through the main search and confirmations. Twelve finite completions
are a timing sample; check failures and the largest retained-memory cases too.
Search OOMs retain failure evidence and do not count as successful coverage.
"""), 
        _cell("code", "search-pilot", """pilot = run_search(plan, target_completed=PILOT_TARGET, max_attempts=MAX_ATTEMPTS, batch_trials=BATCH_TRIALS)
display(pilot)
"""), 
        _cell("code", "timing-estimate", """from common.semantic_hpo_runner import cost_estimate


progress = search_summary(plan)
remaining = max(0, SEARCH_TARGET - progress["completed_finite_trials"])
display(cost_estimate(plan, remaining_trials=remaining, confirmation_runs=TOP_K * len(CONFIRMATION_SEEDS)))
print("Separate scientific baseline comparison adds", len(CONFIRMATION_SEEDS), "full streams.")
print("No pilot duration means no measured runtime estimate. Use the guide's formulas.")
"""), 
        _cell("markdown", "timing-notes", """The timing helper uses finite completed trial durations on this recipe. Its
median/p90 estimates assume comparable future hardware and measured concurrency;
future failure, admission wait, preparation, interruptions and final test work
are additional. The helper accounts for confirmations using one worker per
selected GPU, which may be slower in aggregate than multiworker search. If using
an explicit deadline, reserve the confirmation batches plus a measured allowance.

For scale only: if a full stream takes **2 hours** and one effective worker runs
on each GPU, 200 search + 9 confirmation + 3 separate baseline streams require
**424 GPU-hours**: ideally 17.7 days on one GPU, 5.9 days on three, or 3.5 days on
five. A 25% planning allowance gives approximately 22.1, 7.4 or 4.4 days. These
are arithmetic illustrations, **not measured CIFAR runtimes or speedup claims**.
At 30 minutes per stream divide by four; at 8 hours multiply by four. CIFAR-100
needs its own pilot because task count, replay and class-conditioned state change
costs. No fixed H100/A100 speed ratio or rental price is assumed.
"""), 
        _cell("code", "search-review", """review = run_search(plan, target_completed=REVIEW_TARGET, max_attempts=MAX_ATTEMPTS, batch_trials=BATCH_TRIALS)
display(review)
"""), 
        _cell("markdown", "main-notes", """## Continue the same persistent study

Inspect failures, per-axis coverage, late-task accuracy/forgetting and timing
before extending. Targets are total finite completions; rerunning an achieved
target does not repeat completed training. Interruptions preserve committed
study and recovery artifacts. Stranded `RUNNING` trials require explicit
reconciliation; the notebook does not automatically restart an interrupted
semantic worker or resume it mid-phase. A reached 200-trial budget is not proof
of a global optimum or coverage of every discrete branch and interaction.
"""), 
        _cell("code", "search-main", """search = run_search(plan, target_completed=SEARCH_TARGET, max_attempts=MAX_ATTEMPTS, batch_trials=BATCH_TRIALS)
display(search)
"""), 
        _cell("code", "search-optional", """# The optional extension is explicitly disabled in the delivered notebook.
if RUN_EXTENSION:
    display(run_search(plan, target_completed=EXTENSION_TARGET, max_attempts=MAX_ATTEMPTS, batch_trials=BATCH_TRIALS))
display(search_summary(plan))
"""), 
        _cell("markdown", "confirmation-notes", """## Freeze and confirm the top distinct semantic recipes

Freezing ends this study's search. Select only finite completed maximum-validation
recipes, then run all paired fresh-seed streams. Seeds change learning randomness
while the resolved task groups and dataset split remain fixed. Confirmations do
not use performance pruning. Incomplete seed sets must remain explicitly
incomplete and cannot support paired mean comparisons. Report all three seeds,
mean/spread, per-task matrices and forgetting; three seeds offer limited precision.

The learned finalists alone do not demonstrate semantic benefit. Compare the
selected frozen recipe with `condition="baseline"` in separate native route runs
using the same original CL constants, class groups, split and training seeds.
Additional mechanism or compute-matched controls are described in the guide.
"""), 
        _cell("code", "freeze-finalists", """finalists = freeze_finalists(plan, CONFIRMATION_SEEDS, top_k=TOP_K)
display(finalists)
"""), 
        _cell("code", "confirm-finalists", """confirmation_records = run_confirmations(plan)
display(confirmation_summary(plan))
"""), 
        _cell("markdown", "artifacts", """## Evidence and follow-up experiments

Preserve the immutable native/semantic recipe, source/environment identities,
Optuna parameters/states and storage, input and resolved configs, route settings,
per-task recovery checkpoints, phase timings, validation matrices, TensorBoard,
worker logs, frozen finalist manifests and confirmation receipts. Do not copy
only the best model file. Check final-average accuracy together with forgetting,
backward transfer and time; none is implied by a different metric.

The exhaustive implemented field catalog does not imply that every scientific
variant has been optimized. Zero phase/loss ablations, acquisition objective,
consolidation scope, retained-gate policy, extension schedulers/replay selection,
experimental diagnostics, joint architecture and CL recipe require explicitly
separate studies. Keep the selected model and test protocol frozen before using
official test data. Preparation/CPU tests do not establish GPU fit, convergence
or real dataset timing. No full campaign was run to create this notebook.
""")
    ]
    return {"cells": cells, "metadata": deepcopy(template["metadata"]), "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    """Write the maintained semantic HPO notebook with empty execution outputs."""

    destination = ROOT / "Semantic_Consolidation_HPO_Runner.ipynb"
    destination.write_text(json.dumps(make_notebook(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination)


# Importing the generator never rewrites notebook files.
if __name__ == "__main__":
    main()
