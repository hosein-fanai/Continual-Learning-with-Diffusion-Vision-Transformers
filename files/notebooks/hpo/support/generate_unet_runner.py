"""Build the UNet notebook from the maintained DiT runner's control cells.

Run on a supplied remote container. This writes notebook source only and never
starts HPO. It preserves the shared runner APIs and clears inherited outputs.
"""

from copy import deepcopy
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[4]
DIRECTORY = ROOT / "files/notebooks/hpo"


def build_notebook() -> object:
    """Return the unexecuted UNet runner with a bounded, two-GPU recipe."""

    template = nbformat.read(DIRECTORY / "DiT_Generation_HPO_Runner.ipynb", as_version=4)
    inherited = {cell.id: cell for cell in template.cells}
    cells = []

    def markdown(identifier: str, source: str) -> None:
        """Append one explanatory cell with a stable identifier."""

        cells.append(nbformat.v4.new_markdown_cell(source.strip(), id=identifier))

    def code(identifier: str, source: str) -> None:
        """Append one executable cell without execution state."""

        cells.append(nbformat.v4.new_code_cell(source.strip(), id=identifier))

    def reuse(identifier: str) -> None:
        """Copy a control cell while removing all prior execution state."""

        cell = deepcopy(inherited[identifier])
        # Only executable cells carry an execution counter and outputs.
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
        branch_comments = {
            "remote-guard": {
                'if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):': "# Prefer Kaggle's explicit runtime marker when selecting the setup policy.", 
                'elif os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules:': "# Use Colab setup only when its runtime marker is present.", 
                'elif Path("/.dockerenv").is_file() or Path("/run/.containerenv").is_file():': "# Recognize remote Linux containers through their filesystem markers.", 
                "else:": "# Reject runtimes that lack supported online execution evidence.", 
                "if str(CHECKOUT_ROOT) not in sys.path:": "# Reuse the checkout import path without adding duplicate entries."
            }, 
            "search-first-100": {
                'if not first_stage["target_reached"] and not first_stage["time_budget_exhausted"]:': "# An exhausted attempt ceiling needs inspection before the main search."
            }, 
            "search-to-200": {
                'if not main_stage["target_reached"] and not main_stage["time_budget_exhausted"]:': "# Distinguish an attempt ceiling from the normal experiment deadline."
            }, 
            "confirmation-table": {
                "if not summary.empty:": "# Separate completed seed sets from partial confirmation evidence.", 
                "    if not complete.empty:": "    # Rank only candidates with every required confirmation seed complete.", 
                "    if not partial.empty:": "    # Display incomplete candidates without assigning a final ranking.", 
                "else:": "# An empty table means no confirmation results are available yet."
            }
        }
        for branch, comment in branch_comments.get(identifier, {}).items():
            cell.source = cell.source.replace(branch, comment + "\n" + branch)
        cells.append(cell)

    markdown("overview", """
# UNet denoiser HPO: economical search and seed confirmation

The same resumable workflow as `DiT_Generation_HPO_Runner.ipynb`, using the
repository's conditional convolutional **UNet** and shared `common.hpo` API.
Optimize **validation EMA noise MSE**, with TensorBoard, persistent Optuna storage,
performance/OOM pruning, frozen finalists and paired fresh-seed confirmations.

**Default budget: two A100 80 GB GPUs, at most 12 hours of experiment time.** Search
gets up to 9 hours and confirmation retains 3 hours. The targets are 12 successful
trials for review and 60 for the main search, with an optional extension to 100.
They are ceilings, not a promise of completion. Every trial retains at most
50 epochs and early-stopping patience 5. Two finalists receive three fresh seeds
each. No exhaustive search or globally best model is claimed.

**Data:** CIFAR-10, with a seeded 20% holdout from the official training split
(approximately 40,000 training / 10,000 validation images). The official test
set is excluded from selection. This intentionally improves the validation
protocol over the reference notebook's test-guided search. To reproduce that
protocol, set `VALIDATION_SOURCE="test"`, `VALIDATION_RATIO=0.0` before starting
a fresh study; its test scores then become tuning scores, not held-out estimates.

The notebook is delivered unexecuted. Run it only in an online GPU container.
Run cells individually if you want to inspect the pilot before the main search.
Run All starts the full bounded search and confirmations; the optional extension
is disabled unless explicitly enabled below.
""")
    markdown("runtime-notes", """
## Runtime and cost controls

Use the complete matching checkout with **TensorFlow 2.20.0 / Keras 3.11.2**.
Select **Python (UNet TF2.20)**, or a kernel containing those versions, and restart
before setup. The coordinator hides GPUs; isolated admitted children train.
Container setup verifies packages without replacing them. The inherited hosted
bootstrap can install the pinned environment on Colab/Kaggle, but this notebook
and its matching helpers must be present in the checkout before use.

Use `GPU_IDS=[0, 1]`, `CONCURRENT_TRIALS=2`, and a **24 GiB** TensorFlow cap
per worker: one trial on each GPU.
This is a conservative requested reservation, **not a measured UNet capacity
certificate**. Admission adds per-worker overhead and device headroom, checks
process ownership and preserves existing jobs. Unknown owners block execution.
Recognized search OOMs are recorded and pruned, never silently downsized.

After representative large models fit, two workers on one 80 GB GPU may improve
throughput, but benchmark before increasing concurrency. The DiT notebook's
17-workers-per-H100 result does not apply to UNet. More GPUs reduce waiting time
only when their observed throughput justifies the extra rental cost. Confirmations
use one worker per selected GPU. VRAM is never pooled across devices.

The persistent clock starts at first search, includes pauses/restarts, and cannot
be extended by rerunning a cell. Setup precedes that clock. Save the entire
RESULTS_PATH, including SQLite snapshots, sampler state, configs and notebook_runner.
Use a new directory after scientific settings, source or package versions change.

The database uses container-local transactions under `/tmp/unet-hpo-sqlite`,
with verified immutable snapshots in the persistent study directory. Setup can
restore the last committed snapshot after container replacement. Do not open
the compatibility `study.db` directly; all notebook readers resolve the active
database through the storage helper. A lost container can lose writes after
its most recent snapshot; a same-container kernel restart retains local writes.
""")
    reuse("remote-guard")
    code("settings", '''from copy import deepcopy

from common.unet_hpo import SEARCH_SPACE_OVERRIDES as UNET_SEARCH_SPACE
from common.unet_hpo_storage import prepare_storage


DATASET = "CIFAR10"
RESULTS_PATH = "files/results/unet_generation_hpo_v1"
VALIDATION_SOURCE = "split"
VALIDATION_RATIO = 0.2
GPU_IDS = [0, 1]
CONCURRENT_TRIALS = 2
WORKER_GPU_MEMORY_LIMIT_MB = 24576
EPOCHS = 50
N_STARTUP_TRIALS = 12
SEARCH_SEED = 42
SEARCH_SPACE_OVERRIDES = deepcopy(UNET_SEARCH_SPACE)

PRUNING = {
    "type": "percentile", 
    "monitor": "val_noise_loss", 
    "percentile": 50.0, 
    "n_startup_trials": 12, 
    "n_warmup_steps": 9, 
    "interval_steps": 5, 
    "n_min_trials": 5
}

REVIEW_TARGET = 12
SEARCH_TARGET = 60
EXTENSION_TARGET = 100
ENABLE_EXTENSION = False
MAX_ATTEMPTS = 240
BATCH_TRIALS = max(12, 2 * CONCURRENT_TRIALS)
EXPERIMENT_HOURS = 12.0
CONFIRMATION_RESERVE_HOURS = 3.0
TOP_K = 2
CONFIRMATION_SEEDS = [101, 202, 303]

plan = make_plan(
    CHECKOUT_ROOT, RESULTS_PATH, 
    dataset_name=DATASET, epochs=EPOCHS, n_startup_trials=N_STARTUP_TRIALS, 
    concurrent_trials=CONCURRENT_TRIALS, gpu_ids=GPU_IDS, pruning=PRUNING, 
    validation_source=VALIDATION_SOURCE, validation_ratio=VALIDATION_RATIO, 
    experiment_hours=EXPERIMENT_HOURS, 
    confirmation_reserve_hours=CONFIRMATION_RESERVE_HOURS, 
    search_space_overrides=SEARCH_SPACE_OVERRIDES, 
    worker_gpu_memory_limit_mb=WORKER_GPU_MEMORY_LIMIT_MB, 
    model_name="unet", seed=SEARCH_SEED
)
storage_status = prepare_storage(plan)
display(storage_status)
display(plan["hpo"])
print("Persistent study:", plan["study_root"])
TENSORBOARD_DIRECTORY = Path(plan["study_root"])
display(plan["identity"]["gpus"])
display(budget_summary(plan))''')
    markdown("space", """
## Thorough, conditional search space

All settings flow through the maintained API. The explicit `denoiser_v1` switch
leaves older UNet studies' distributions unchanged. This is a broad finite domain
sampled by TPE, not a Cartesian grid. Limited runs cannot cover every interaction.

| Group | Domain and condition |
| --- | --- |
| Encoder widths | 32-64; 32-64-96; 32-64-128; 48-96-192; 64-96-128; 64-128-256; 32-64-128-256 |
| Residual depth | 1 / 2 / 3 blocks at each encoder/decoder level |
| Bottleneck | 1 / 1.5 / 2 times widest encoder width; 1 / 2 / 3 blocks |
| Embedding channels | Total 64 / 96 / 128 / 192; balanced or time-rich allocation |
| Nonlinearity | swish / gelu / relu |
| Normalization | Batch normalization enabled / disabled; no unsupported GroupNorm knob |
| Spatial dropout | 0 / 0.05 / 0.1 / 0.2 |
| Downsampling | average pooling / max pooling / learned strided convolution |
| Upsampling | interpolation / convolution plus interpolation / transpose convolution |
| Interpolation | nearest / bilinear; sampled only for interpolation-based upsampling |
| Optimizer | Adam / AdamW |
| Learning rate | 1e-4 to 1e-3, log-uniform; cosine decay |
| Weight decay | 1e-6 to 1e-3, log-uniform; AdamW only |
| Batch size | 32 / 64 / 128 |
| Gradient clipping | Global norm None / 1 / 5; per-variable clipping disabled |
| EMA decay | 0.995 / 0.999 |
| CFG label dropout | 0.05 / 0.1 / 0.2; class conditioning and CFG remain enabled |

Fixed controls keep the objective comparable: ordinary epsilon prediction,
**1,000 timesteps, clipped-cosine noise schedule, MSE, linear output, encoder
skips enabled, zero auxiliary image loss**. Different noise schedules change
the prediction task, so their raw MSEs should not be treated as equivalent
architecture scores. Compare schedules later in separate controlled studies
using a common image-quality evaluation. Model constructors do not expose
attention or convolution-kernel choices here; those are not fictitious HPO axes.

The 2–4 downsamplings divide 32 evenly, reaching 8, 4 or 2 pixels. Width templates
avoid invalid or needlessly huge hierarchies. A balanced embedding allocates
roughly one third to image/time/label; time-rich allocates roughly one quarter,
one half, one quarter, preserving the selected total exactly.

Sampling steps, guidance scale and eta do not change this loss-based HPO objective,
so they are not searched. Existing reports use fixed sample settings. Tune those
only after selecting a denoiser, and assess image quality/diversity and downstream
replay usefulness separately. A better denoising score need not mean better images.
""")
    markdown("pruning-policy", """
## Pruning, recovery and TensorBoard

TPE begins with 12 startup observations. The median/50th-percentile rule starts
after 12 completed reference trials, no earlier than epoch 10, then every five
epochs when at least five reference observations exist. This economical policy
can discard slow starters; finalists keep the same 50-epoch limit and early
stopping but disable performance pruning. Numerical/OOM guards are separate.
Pruned/failed trials consume attempts but do not count toward successful targets.

The shared runner persists input YAML, histories, reports, sampler state, database,
source identities and worker receipts. Reached targets add no work on rerun.
Deadline cancellations preserve partial evidence without claiming completion.
Unknown crashes remain errors. Source/recipe guards prevent accidental reuse of
another experiment. The unchanged shared TensorBoard callback writes metrics
under the study and confirmation trees; no duplicate training loop is introduced.
""")
    reuse("tensorboard-live")
    reuse("initial-status")
    reuse("search-first-100")
    markdown("pilot-review", """
## Review the pilot before spending the remaining budget

Inspect completed/pruned counts, width and resampling coverage, failure logs and
duration. The estimate below uses elapsed pilot throughput on **your current
hardware and concurrency**, including pruning overhead. It is a planning estimate:
later TPE suggestions can be larger, and confirmation has performance pruning
disabled. If fewer than two distinct configurations complete, selection cannot
produce two finalists. Do not interpret an incomplete confirmation table as a win.

The two-GPU default has the same 24 GPU-hour allowance as one GPU for 24 hours.
For a cheaper first look, before first execution choose 6 total hours with a
2-hour confirmation reserve and SEARCH_TARGET=30 in a fresh results directory.
""")
    code("trial-table", '''import optuna

from common.hpo_sqlite import database_path


database = database_path(plan["study_root"])
study = None
# Admission or a resumed deadline can prevent the first allocation entirely.
if database.is_file():
    study = optuna.load_study(
        study_name=plan["study_name"], storage="sqlite:///" + database.as_posix()
    )
    trial_table = study.trials_dataframe()
    # No scalar objective exists until at least one trial is allocated.
    if "value" in trial_table.columns:
        display(trial_table.sort_values("value").head(15))
    # Retain useful status columns even before the first score is recorded.
    else:
        display(trial_table)
# A missing database is an unstarted search, not an empty successful study.
else:
    print("No study database yet; inspect admission and budget status.")
display(search_summary(plan))''')
    code("measured-budget", '''from statistics import median


finished = [trial for trial in study.trials if trial.datetime_complete is not None] if study is not None else []
successful = [trial for trial in finished if trial.state.name == "COMPLETE" and trial.value is not None]
# Use the observed wall span, so concurrent trials do not masquerade as serial time.
if successful:
    span_hours = (
        max(trial.datetime_complete for trial in finished)
        - min(trial.datetime_start for trial in finished if trial.datetime_start is not None)
    ).total_seconds() / 3600
    successful_per_hour = len(successful) / max(span_hours, 1e-9)
    typical_trial_hours = median(trial.duration.total_seconds() / 3600 for trial in successful)
    display({
        "observed_successful_trials_per_hour": round(successful_per_hour, 2), 
        "estimated_additional_search_hours": round(max(0, SEARCH_TARGET - len(successful)) / successful_per_hour, 2), 
        "rough_confirmation_hours": round(TOP_K * len(CONFIRMATION_SEEDS) * typical_trial_hours / len(GPU_IDS), 2), 
        "warning": "Pilot extrapolation only; finalist size and early stopping can change runtime."
    })
# A time-limited pilot can end before its first successful result.
else:
    print("No successful pilot trial yet; inspect logs before estimating throughput.")
display(budget_summary(plan))''')
    reuse("search-to-200")
    code("optional-extension", '''# Extra trials use the existing clock and attempt ceiling, never extra hours.
if ENABLE_EXTENSION:
    final_stage = run_search(
        plan, target_completed=EXTENSION_TARGET, max_attempts=MAX_ATTEMPTS, 
        batch_trials=BATCH_TRIALS
    )
    display(final_stage)
# Keep the economical 60-success target when the extension is disabled.
else:
    print("Optional extension disabled.")''')
    markdown("confirmation-design", """
## Freeze two finalists and confirm on three paired seeds

The top two distinct finite COMPLETE configurations are frozen with source hashes.
After freezing, search cannot be extended. Each is retrained from scratch with
seeds 101, 202 and 303, keeping the same training/validation split and shuffle seed.
Six runs are outside the search-trial count, with no performance pruning.
Early stopping and the epoch maximum remain. Completed confirmations are reused;
failed/incomplete attempts retain their artifacts and are not successful repeats.

All candidates need every required seed before a fair mean comparison. Three seeds
provide a modest stability check, not proof of optimality. If the shared deadline
prevents completion, the summary marks that limitation explicitly. The recovery
helper below only repairs authenticated orphan records after a stopped search;
it refuses live workers and never restarts completed training.
""")
    code("freeze-finalists", '''from common.dit_hpo_recovery import recover_stopped_search


recovery = recover_stopped_search(plan)
display(recovery)
finalists = freeze_finalists(plan, CONFIRMATION_SEEDS, top_k=TOP_K)
display(finalists)''')
    reuse("run-confirmations")
    reuse("confirmation-table")
    markdown("budget", """
## Simple GPU budget

| Plan | Hardware | Experiment cap | Intended use |
| --- | --- | --- | --- |
| Cheapest first pass | 2 x A100 80 GB | 6 h: 4 search + 2 confirmation | Up to 30 successful trials; broad coverage will be limited |
| Recommended starting allowance | 2 x A100 80 GB | 12 h: 9 search + 3 confirmation | Default target 60 plus six seed confirmations |

These are **spending limits and unbenchmarked planning allowances**, not measured
completion times. The code stops at its persistent deadline even if targets or
confirmations are unfinished. Provider setup, data download, idle rental time,
storage and cleanup are outside the nominal experiment charge. Ending this
notebook does not stop billing or shut down your container.

Runpod's public pricing page viewed 2026-10-10 lists A100 80 GB at **$1.79/GPU-h**:
Two GPUs for 6 hours (12 GPU-hours) is about **$21.48**; two GPUs for 12 hours
(24 GPU-hours) about **$42.96**, excluding extras.
Use your actual quoted rate: `GPUs x hours x price_per_GPU_hour`. Availability,
cloud tier and the live console quote can differ. H100 SXM is listed at $3.99/h;
it needs more than approximately 2.23 times this workload's A100 throughput to
beat that A100 hourly price per completed workload. No UNet speedup is measured
here, so renting several H100s is not the default recommendation.

Sources: [Runpod GPU pricing](https://www.runpod.io/pricing),
[Optuna conditional search spaces](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html),
[Optuna PercentilePruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.PercentilePruner.html).

Use the pilot's measured rate to judge whether the desired successful-trial count
fits. More time should follow evidence of useful improvement, not a grid-size
calculation. Keep the official test set untouched until the full recipe is chosen.
""")
    notebook = nbformat.v4.new_notebook(cells=cells, metadata=deepcopy(template.metadata))
    notebook.metadata.pop("dit_hpo_runner", None)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python (UNet TF2.20)", "language": "python", "name": "unet-tf220"
    }
    notebook.metadata["unet_hpo_runner"] = {
        "execution": "remote_only", "protocol_version": 1, "unexecuted": True
    }
    return notebook


def main() -> None:
    """Write and schema-check the unexecuted notebook artifact."""

    notebook = build_notebook()
    nbformat.validate(notebook)
    destination = DIRECTORY / "UNet_Generation_HPO_Runner.ipynb"
    nbformat.write(notebook, destination)
    print(destination)


# Script execution generates notebook source without starting a study.
if __name__ == "__main__":
    main()
