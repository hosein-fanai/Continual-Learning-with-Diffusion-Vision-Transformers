"""Author the continual runner on an authorized remote container without training."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _cell(kind: str, cell_id: str, source: str) -> dict:
    """Build one unexecuted notebook cell from readable maintained source."""

    cell = {"cell_type": kind, "id": cell_id, "metadata": {}, "source": source.splitlines(keepends=True)}
    # Only executable cells carry execution counts and output containers.
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def make_notebook() -> dict:
    """Compose the established remote bootstrap with the candidate search recipe."""

    template = json.loads((ROOT / "DiT_Generation_HPO_Runner.ipynb").read_text(encoding="utf-8"))
    setup = deepcopy(next(
        cell for cell in template["cells"]
        if cell["cell_type"] == "code" and "from common.dit_hpo_runner import (" in "".join(cell["source"])
        and "prepare_notebook" in "".join(cell["source"])
    ))
    setup["id"] = "remote-guard"
    case_comments = {
        'if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):': "# Prefer Kaggle's live marker when hosted-provider markers overlap.", 
        'elif os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules:': "# Colab identifies the managed runtime when Kaggle is absent.", 
        'elif Path("/.dockerenv").is_file() or Path("/run/.containerenv").is_file():': "# A supplied Linux container uses the existing remote-container setup.", 
        "else:": "# Reject an unrecognized environment before setup or GPU admission.", 
        "if str(CHECKOUT_ROOT) not in sys.path:": "# Add the verified checkout only when it is not already importable."
    }
    setup_lines = ['"""Coordinate remote-only continual HPO with a sealed official-test protocol."""\n', "\n"]
    for line in setup["source"]:
        # Add branch intent while leaving the shared bootstrap's behavior intact.
        if line.rstrip("\r\n") in case_comments:
            setup_lines.append(case_comments[line.rstrip("\r\n")] + "\n")
        setup_lines.append(line.replace("from common.dit_hpo_runner import (", "from common.dit_continual_runner import ("))
    setup["source"] = setup_lines
    setup["execution_count"] = None
    setup["outputs"] = []
    cells = [
        _cell("markdown", "overview", """# DiT continual HPO: official-test selection in twelve hours

Each trial trains **five sequential two-class CIFAR-10 tasks**. The class partition
and order are randomized once with `TASK_SEED`, sealed, and reused by every trial
and fresh-seed confirmation. Trial seeds never reshuffle the task stream.

Optuna chooses the student configuration and active current classifier/noise
specialists from the maintained evidence-backed catalog. The student is both the
replay generator and classifier through the native V1 `DiffusionClassifier`.
Clean train/test and noisy train/test joint recipes are distinct candidates.
Snapshots, replay and evaluation use the raw network with EMA disabled.

**Every evaluation uses official test rows.** No training holdout is created.
The full official training split remains available; searched exposure budgets may
select subsets. The native scoring slot is named `validation_continual_metrics`,
but it contains official-test scores here, without a duplicate test matrix.
Optuna maximizes task-balanced `final_average_accuracy` after task five.
These test-tuned scores are selection estimates, including after repeated seeds;
they are not an untouched estimate of generalization.

The persistent **12-hour** execution budget allocates **9 hours to search** and
**3 hours to three finalists at three fresh seeds each**. `RUN_HPO=False` keeps
training disabled while setup, candidate recipes and measured capacity are reviewed.
"""), 
        _cell("markdown", "runtime-notes", """## Remote runtime and provisional GPU recommendation

Run this entire notebook on an online container with **TensorFlow 2.20.0 / Keras
3.11.2**. The coordinator does not import either framework; admitted isolated
workers perform computation. Verify checkout, GPU UUIDs, free memory and live jobs.
Preserve other jobs. Generation-only H100 concurrency measurements do not establish
capacity for continual learning with snapshots, specialist fitting and replay.

The provisional conservative search allocation is **32 A100 80 GB GPUs** with a **24,576 MiB
TensorFlow cap per worker** plus allocator overhead. Three simultaneous workers
completed five tasks at two epochs per task in 36–40 minutes, using approximately
73.3 GiB together. An explicit warm-epoch/replay work model projects 4.36–5.12 hours
per concurrent 20-epoch trial at the revised replay maximum; this is an estimate,
not a measured bound. Aggregate GPU-hours suggest 23 devices, but 24 devices would
need a second wave beyond the nine-hour allowance if every trial took 5.12 hours.
32 devices supply 96 simultaneous workers. Eight GPUs support 24 completions
under that uniform-duration wave scenario; about 33 is only an aggregate-work
estimate that ignores the completion tail. Full-duration
trials, other candidates and multi-GPU contention remain unmeasured. The defaults `GPU_IDS=[0]` and
`WORKERS_PER_GPU=1` are for readiness inspection. Configure the full campaign only
after verifying every device and measuring the intended worker concurrency.

Use complete-trial waves as well as aggregate GPU-hours in capacity planning.
The factor 0.8 reserves 20% for startup, variation and failed attempts. Eight GPUs
at one worker each require a representative **complete five-task trial in at most
36 minutes**; at three measured workers each the corresponding limit is **108
minutes per concurrent worker**. Single-worker speed cannot be substituted for
concurrent speed. Measure costly candidates including both specialists, replay, test
evaluation and checkpoints. A bounded smoke test cannot establish this duration.
If it is longer, increase GPU count or explicitly reduce the completion target.
Confirmations use **one worker per GPU**, independently of search concurrency.
Eight GPUs therefore need two waves for nine confirmations, even with 24 search
workers. Measure their one-worker duration separately, or conservatively reuse
the concurrent-search duration in the confirmation-capacity gate below.
With 32 GPUs, nine confirmations fit one wave, but each must finish within
**2.4 hours** to preserve the three-hour reserve and margin. This has not been
measured; the projected concurrent duration cannot pass that gate. The complete
12-hour campaign is therefore not yet certified, and `RUN_HPO` remains false.
"""), 
        setup, 
        _cell("markdown", "inputs-notes", """## Candidate provenance and initialization

`common.dit_continual_candidates.candidate_profile` supplies the maintained catalog
and source evidence. It distinguishes measured best/near-best configurations from
motivated transfers; review its study/trial/configuration identities before launch.
Student identity and the two specialist identities are categorical hyperparameters.
Specialist identities are sampled only when their current/both route is active.
Previous teachers remain snapshots of the selected student.

- **Four students:** joint165 clean/noisy and cheaper joint155 clean/noisy.
- **Three classifier teachers:** joint165 clean/noisy and a tentative joint135
  architecture transfer with zero classifier DropPath.
- **Three noise teachers:** confirmed UNet trial23, CIFAR10 DiT trial1351 and
  joint165 noisy. Their epsilon processes are aligned to the student protocol.

CIFAR100 and follow-up winners were reviewed but excluded unchanged because their
dataset, 5,000-step linear schedule and CFG convention do not match this narrowed
budget. No completed local CIFAR10 follow-up result archive was found; a notebook
alone is not evidence of a winning model. Catalog notes preserve those decisions.

Historical configurations initialize fresh continual models. Full-dataset class
heads and condition embeddings must not silently initialize a randomly relabeled
task stream. Expandable vocabularies use dense task IDs; the settings cell displays
the original-label mapping. Any explicit checkpoint must match its native config
and label vocabulary and is hashed. Specialists train on real new-task rows through
the existing teacher API. Their native optimizer/noise semantics are retained.

Clean and noisy joint candidates retain their paired training/testing settings;
`classifier_noise` is not independently suggested in candidate mode. Loss constants
come from each selected native recipe; the source choices gate their active terms.
Every student and active specialist has **20 epochs per task**. Epoch counts are
equal; data exposure, optimizer updates and total compute can vary with the
searched training/replay budgets and the number of active specialists.
The batch choices are 64/128; each selected combination must fit measured memory.
"""), 
        _cell("markdown", "search-space", """## Search size and treatment coverage

Target **96 finite COMPLETE trials**, with **24 startup observations**, a review
at **32**, and at most **192 allocated attempts**. This is a practical screening
budget for a narrowed conditional search, not a proven sufficient number or an
exhaustive search. Continuous distributions rule out exhaustive enumeration.
Review improvement, failures and candidate coverage; the time limit takes priority.

| Dimension | Search |
| --- | --- |
| Student / current classifier / current noise | Catalog candidate IDs, conditionally active |
| Each teacher source | none, previous, current, both: 16 source pairs |
| Continual strategy | generative replay, cumulative, new-only |
| Replay budget | fixed total; old 100 / 500 / 1,000; current 1,000 / 2,500 / 5,000 |
| Replay selection / candidate multiplier | all, uniform, confidence / 1 or 2 |
| Sampling | 20 or 50 steps; CFG [2.5, 5]; eta 0 |
| Batch / epochs | 64 or 128 / 20 per task |
| Optimizer / learning rate | Adam or AdamW / log [3e-4, 2e-3] |
| AdamW decay | log [1e-6, 1e-3] |
| Clipping | per-variable None/0.5/1/5; global only without per-variable clipping |
| Unconditional-label probability | 0.05, 0.1, 0.2 |

The expensive pilot generated 10,000 candidates per replay transition and was
still in its first transition after twenty minutes. The default old-replay ceiling
is therefore 1,000 selected examples (2,000 candidates with oversampling), reducing
maximum sampling volume fivefold. This changes the searched learning protocol;
equivalent accuracy to a 5,000-example replay buffer is not assumed.

With the default catalog, the initial queue has **19 partial warm-start hints**:
16 teacher-source pairs plus three additional student/no-distillation anchors.
These hints count toward the 24 startup observations. Their remaining parameters
are sampled, so source/candidate gains are HPO comparisons rather than matched
causal ablations. Before strategy, replay and optimizer choices, there are already
**256 student/source/active-teacher combinations**:
`4 * (1 + 1 + 3 + 3) * (1 + 1 + 3 + 3)`.

A failed or OOM-pruned attempt does not establish successful coverage.
Current specialists apply to new
examples; previous snapshots apply to old examples from task two. New-only is
excluded when a previous route needs old rows. Replay-only controls are absent in
data-only strategies. Adaptive candidate sampling does not guarantee equal coverage
of every candidate combination within 96 observations.
"""), 
        _cell("code", "settings", """from math import ceil, floor

from common.config import resolve_continual_schedule
from common.dit_continual_candidates import CATALOG_EVIDENCE_NOTES, candidate_profile
from common.dit_continual_runner import start_experiment


DATASET = "CIFAR10"
RESULTS_PATH = "files/results/dit_continual_hpo_v2_test_12h"
VALIDATION_SOURCE = "test"
VALIDATION_RATIO = 0.0
GPU_IDS = [0]  # Readiness only; configure verified campaign devices after the pilot.
WORKERS_PER_GPU = 1  # Increase only after measured aggregate throughput and memory checks.
CONCURRENT_TRIALS = len(GPU_IDS) * WORKERS_PER_GPU
WORKER_GPU_MEMORY_LIMIT_MB = 24576
EPOCHS = 20  # Equal per-task student/specialist budget; five tasks per trial.
N_STARTUP_TRIALS = 24
SEARCH_SEED = 42
TASK_SEED = 42
RUN_HPO = False  # Enable after candidate, memory and complete-trial timing review.
MEASURED_FULL_TRIAL_HOURS = None  # Conservative measured complete-trial duration.
MEASURED_CONFIRMATION_TRIAL_HOURS = None  # One worker/GPU; None reuses the conservative search time.

SEARCH_SPACE_OVERRIDES = {
    "replay_budget_mode": ["fixed_total"], 
    "replay_old_examples": [100, 500, 1000], 
    "replay_current_examples": [1000, 2500, 5000], 
    "replay_selection": ["all", "uniform", "confidence"], 
    "replay_candidate_multiplier": [1, 2], 
    "test_steps": [20, 50], "test_eta": [0.0], "batch_size": [64, 128], 
    "learning_rate": {"low": 3e-4, "high": 2e-3, "log": True}
}
REVIEW_TARGET = 32
SEARCH_TARGET = 96
MAX_ATTEMPTS = 192
BATCH_TRIALS = max(16, 2 * CONCURRENT_TRIALS)
EXPERIMENT_HOURS = 12.0
CONFIRMATION_RESERVE_HOURS = 3.0
TOP_K = 3
CONFIRMATION_SEEDS = [101, 202, 303]

CONTINUAL_PROFILE = candidate_profile(task_seed=TASK_SEED)
preview_class_order, preview_task_groups = resolve_continual_schedule(
    10, available_class_num=10, task_size=2, class_order_mode="random", seed=TASK_SEED
)
display({"dense_to_original_label": dict(enumerate(preview_class_order)), "task_groups": preview_task_groups})
display([
    {"role": role, "candidate": candidate_id, **item["provenance"]}
    for role in ("student_candidates", "classifier_teacher_candidates", "noise_teacher_candidates")
    for candidate_id, item in CONTINUAL_PROFILE[role].items()
])
display(CATALOG_EVIDENCE_NOTES)
# A completed representative timing measurement permits a throughput calculation.
if MEASURED_FULL_TRIAL_HOURS is not None:
    measured_workers_per_gpu = CONCURRENT_TRIALS / len(GPU_IDS)
    complete_trials_per_worker = floor(
        (EXPERIMENT_HOURS - CONFIRMATION_RESERVE_HOURS) * 0.8 / MEASURED_FULL_TRIAL_HOURS
    )
    # A full trial must fit inside the search allowance before devices can help.
    if complete_trials_per_worker < 1:
        raise RuntimeError("A measured full trial exceeds the search allowance; revise the phase budget or trial recipe.")
    recommended_gpus = ceil(SEARCH_TARGET / (complete_trials_per_worker * measured_workers_per_gpu))
    display({"required_gpus_from_measurement": recommended_gpus, "configured_gpus": len(GPU_IDS)})
    confirmation_waves = ceil(TOP_K * len(CONFIRMATION_SEEDS) / len(GPU_IDS))
    confirmation_trial_hours = (
        MEASURED_FULL_TRIAL_HOURS if MEASURED_CONFIRMATION_TRIAL_HOURS is None
        else MEASURED_CONFIRMATION_TRIAL_HOURS
    )
    confirmation_hours_required = confirmation_waves * confirmation_trial_hours / 0.8
    display({"confirmation_hours_required": confirmation_hours_required, "reserved_hours": CONFIRMATION_RESERVE_HOURS})
"""), 
        _cell("markdown", "validation-notes", """## Seal the task schedule and persistent clock

Changing candidate recipes, task seed, objective or distributions requires a fresh
results directory. Worker placement and total completion targets can change while
preserving that scientific recipe. Restarting cannot reset the persistent clock.
Performance pruning is disabled because epoch numbers repeat across tasks/teachers.
OOM/nonfinite handling remains active. Admission waiting consumes the budget.
Search stops at nine hours and confirmations at twelve hours, with owned children
stopped/reaped and partial checkpoints preserved. No partial trial counts as complete.
"""), 
        _cell("code", "plan", """from common.dit_continual_hpo import SEARCH_SPACE


plan = make_plan(
    CHECKOUT_ROOT, RESULTS_PATH, dataset_name=DATASET, epochs=EPOCHS, 
    n_startup_trials=N_STARTUP_TRIALS, concurrent_trials=CONCURRENT_TRIALS, 
    gpu_ids=GPU_IDS, pruning=None, validation_source=VALIDATION_SOURCE, 
    validation_ratio=VALIDATION_RATIO, experiment_hours=EXPERIMENT_HOURS, 
    confirmation_reserve_hours=CONFIRMATION_RESERVE_HOURS, 
    search_space_overrides=SEARCH_SPACE_OVERRIDES, 
    worker_gpu_memory_limit_mb=WORKER_GPU_MEMORY_LIMIT_MB, 
    search_profile="dit_continual_runner", continual_profile=CONTINUAL_PROFILE, 
    seed=SEARCH_SEED
)
display({"base_space": SEARCH_SPACE, "active_overrides": SEARCH_SPACE_OVERRIDES})
display(plan["hpo"]["continual_profile"]["task_groups"])
display(plan["identity"]["gpus"])
print("Persistent study:", plan["study_root"])
display(budget_summary(plan))
# Explicit campaign enablement starts the persistent clock only after capacity review.
if RUN_HPO:
    # Unmeasured duration cannot support the requested twelve-hour completion target.
    if MEASURED_FULL_TRIAL_HOURS is None:
        raise RuntimeError("Record a representative complete-trial duration before enabling the twelve-hour campaign.")
    # The selected device count must support the target under that recorded duration.
    if len(GPU_IDS) < recommended_gpus:
        raise RuntimeError("Configured GPUs cannot meet the target under the measured throughput budget.")
    # Finalist waves need their own measured capacity within the same persistent clock.
    if confirmation_hours_required > CONFIRMATION_RESERVE_HOURS:
        raise RuntimeError("Measured confirmation waves exceed the reserve; revise phase allocation or capacity before launch.")
    display(start_experiment(plan))
# Setup-only execution leaves all training phases disabled.
else:
    print("Setup only: HPO is disabled. Review verified capacity and the candidate catalog first.")
"""), 
        _cell("markdown", "tensorboard-notes", """## TensorBoard

Native task, teacher and replay summaries remain visible. The evaluation namespace
may say `validation` for the official-test scoring slot; consult data-source metadata.
"""), 
        _cell("code", "tensorboard", """# Open TensorBoard only for an enabled campaign.
if RUN_HPO:
    TENSORBOARD_DIRECTORY = Path(plan["study_root"])
    get_ipython().run_line_magic("load_ext", "tensorboard")
    get_ipython().run_line_magic("tensorboard", f"--logdir {TENSORBOARD_DIRECTORY} --port 6006")
"""), 
        _cell("markdown", "review-notes", """## Review after 32 completed trials

Review teacher/candidate coverage, failures, duration and memory before continuing.
Targets count total finite completions; rerunning reached targets repeats no work.
Interruptions preserve the study and task checkpoints.
"""), 
        _cell("code", "search-review", """# Run the initial review stage only after campaign enablement.
if RUN_HPO:
    review = run_search(plan, target_completed=REVIEW_TARGET, max_attempts=MAX_ATTEMPTS, batch_trials=BATCH_TRIALS)
    display(review)
    display(budget_summary(plan))
"""), 
        _cell("markdown", "main-search-notes", """## Continue toward 96 completed trials

Run this cell after review. The same nine-hour search deadline applies if the
target is not reached. Report actual completed trials and source/candidate coverage.
"""), 
        _cell("code", "search-main", """# Continue the same timed study only when campaign execution is enabled.
if RUN_HPO:
    search = run_search(plan, target_completed=SEARCH_TARGET, max_attempts=MAX_ATTEMPTS, batch_trials=BATCH_TRIALS)
    display(search)
    display(budget_summary(plan))
"""), 
        _cell("markdown", "confirmation-notes", """## Freeze three finalists and repeat three fresh seeds

Freeze the top three distinct configurations by **official-test final average
accuracy** after search completes or its phase deadline expires. The selection is
immutable. Incomplete coverage must remain visible in any conclusions.
Each finalist repeats the complete five-task sequence from its original recipe at
seeds 101, 202 and 303. Task groups, test rows and recipes stay fixed. Report means
and standard deviations with completed-seed counts; complete paired seed sets
support comparative means. These remain test-tuned selection estimates.
"""), 
        _cell("code", "freeze-finalists", """# Freeze selection only for the enabled campaign after its search stage.
if RUN_HPO:
    finalists = freeze_finalists(plan, CONFIRMATION_SEEDS, top_k=TOP_K)
    display(finalists)
"""), 
        _cell("code", "confirm-finalists", """# Execute paired seed confirmations only for the enabled frozen campaign.
if RUN_HPO:
    confirmation_records = run_confirmations(plan)
    display(confirmation_summary(plan))
    display(budget_summary(plan))
"""), 
        _cell("markdown", "artifacts", """## Saved evidence

Keep candidate/source identities, sealed tasks, Optuna parameters/states, input and
resolved YAML, task checkpoints, official-test matrices in the native validation
slot, forgetting/backward-transfer metrics, TensorBoard, logs, persistent clock,
finalist manifests and authenticated seed receipts. Report actual hardware,
completed trials, seed coverage and wall time alongside every result.
""")
    ]
    metadata = {
        key: deepcopy(value) for key, value in template["metadata"].items()
        if key in {"kernelspec", "language_info"}
    }
    metadata.update({"protocol_version": 2, "execution": "remote_only", "unexecuted": True})
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    """Write only the maintained continual runner notebook."""

    destination = ROOT / "DiT_Continual_HPO_Runner.ipynb"
    destination.write_text(json.dumps(make_notebook(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination)


# Importing the builder never rewrites the maintained notebook.
if __name__ == "__main__":
    main()
