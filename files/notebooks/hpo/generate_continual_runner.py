"""Maintain the continual runner using the generation notebook's online bootstrap.

Run this authoring utility only on an authorized remote container. It writes an
unexecuted notebook; it never starts an HPO campaign or loads model objects.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _cell(kind: str, cell_id: str, source: str) -> dict:
    """Build an unexecuted nbformat cell from readable maintained source."""

    cell = {"cell_type": kind, "id": cell_id, "metadata": {}, "source": source.splitlines(keepends=True)}
    # Only executable cells carry notebook execution state.
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def make_notebook() -> dict:
    """Compose the existing bootstrap with the fixed-architecture continual plan."""

    template = json.loads((ROOT / "DiT_Generation_HPO_Runner.ipynb").read_text(encoding="utf-8"))
    setup = deepcopy(next(cell for cell in template["cells"] if cell["id"] == "remote-guard"))
    setup["source"] = [line.replace("from common.dit_hpo_runner import (", "from common.dit_continual_runner import (") for line in setup["source"]]
    setup["execution_count"] = None
    setup["outputs"] = []
    cells = [
        _cell("markdown", "overview", """# DiT CLF continual-learning HPO runner

Use one input **DiTClassifier** as both replay generator and classifier through
the native **V1 `DiffusionClassifier`**. The student architecture and all active
distillation types, loss coefficients and accuracy coefficients are fixed inputs.
`use_ema=False`; snapshots, replay and evaluation use the raw network.

The public HPO API maximizes validation **final_average_accuracy** over **five
two-class CIFAR-10 tasks**. A seeded random class partition/order is materialized
once, written into the recipe, and reused by every trial and confirmation.

This mirrors the generation runner: remote setup → explicit inputs → admission
and immutable recipe → review search → extended search → frozen finalists →
paired fresh-seed confirmations. All stages are resumable. The notebook contains
no replacement training loop, replay sampler, distillation loss or GPU scheduler.
"""), 
        _cell("markdown", "runtime-notes", """## Remote execution and measured concurrency

Run this entire notebook on the supplied online GPU container with TensorFlow
**2.20.0** and Keras **3.11.2**. The coordinator does not import either framework;
the existing admitted child workers perform training. Select the verified remote
kernel (for the three-H100 checkout, `Python (DiT TF2.20)`).

`GPU_IDS` selects physical devices; `CONCURRENT_TRIALS` is the total number of
independent full continual trials. Each trial trains its tasks sequentially on
one GPU. A single Optuna coordinator owns study writes. Reservations account for
per-worker memory plus overhead, bind GPU UUIDs and preserve existing jobs.

The initial setting is one worker because the student and teacher inputs are
not supplied yet. Measure the largest student + previous snapshot + both current
specialists + replay workload before increasing concurrency. Generation-only
51-worker measurements do not establish capacity for this workload. Multiple
workers on one GPU and across GPUs use the same public HPO process scheduler.
Recognized OOMs are preserved as pruned attempts; they are not successful modes.
No global deadline is imposed unless `EXPERIMENT_HOURS` is explicitly set.
"""), 
        setup, 
        _cell("markdown", "inputs-notes", """## Supply the fixed student and task specialist recipes

Set `STUDENT_CONFIG` to a native project configuration mapping or YAML path.
Its model must be `dit_classifier` with wrapper `diffusion_classifier` (V1),
including its intended classifier head. Set the architecture's
classifier/noise distillation type, temperature, loss weights and accuracy
weights there; these values are **not** optimization dimensions. An initial
student checkpoint may be included in the configuration and is held fixed.
No layers are added: the existing wrapper uses the configured distillation-token
head when present, and its existing primary-head fallback otherwise. A separate
distillation token is not required.

In native `model.wrapper_kwargs`, these include `clf_distil_type`,
`clf_distil_temperature`, `clf_distil_loss_coef`, `clf_distil_acc_coef`,
`noise_distil_loss_coef`, `previous_teacher_clf_loss_weight`,
`current_teacher_clf_loss_weight`, `previous_teacher_noise_loss_weight` and
`current_teacher_noise_loss_weight`. The selected loss-bearing terms must have
active coefficients in the supplied model; the notebook never invents them.
Use the existing model's noise loss/compile configuration for noise distillation.

Supply a trained/loadable EfficientNet classifier descriptor and a native U-Net
configuration descriptor. Current-task specialists are trained by the existing
teacher-training API on the current task's training rows. Each uses its own
supplied optimizer/compile settings and the notebook's per-task epoch budget.
The previous-task teacher is a raw student snapshot.
Teacher architecture/checkpoint identities are sealed with the study. The inputs
below are intentionally unset; no substitute student or teachers are invented.

Supported descriptor shapes (use remote paths):
`{"format": "keras", "path": "/workspace/models/efficientnet.keras"}` and
`{"format": "config", "path": "/workspace/models/unet.yaml", "weights_path": "/workspace/models/unet.weights.h5"}`.
The U-Net weights path is optional. The normalizer adds artifact hashes.

For EfficientNet, supply the pretrained backbone with a fresh or appropriately
initialized **expandable final Dense head**, retaining its optimizer/loss/metrics.
Compile that head for sparse class labels, as used by the existing teacher API.
The learner uses dense labels in the resolved task order: dense label `i` means
original CIFAR-10 label `class_order[i]`. An existing trained head is compatible
only when its output columns already follow that vocabulary. Original CIFAR-10
column order is not automatically remapped to the random task order.

For the native U-Net, `model.unet.num_classes=None` (or `model.kwargs.num_classes=None`
when using generic constructor kwargs) lets its vocabulary grow with the task
stream. A fixed vocabulary, such as ten conditions, requires an initial checkpoint
whose condition IDs already match the same resolved dense-label mapping. The
settings cell below displays this mapping before model-input validation.
Set the U-Net configuration's `use_ema=False` and `test_network_name="raw"`
in its diffusion wrapper settings too. The profile validates these supplied
settings; it does not silently alter the teacher recipe. Its noise schedule,
preprocessing and optimizer come from that native configuration, while its
per-task training epoch and batch budgets follow the student trial.
"""), 
        _cell("code", "model-inputs", """STUDENT_CONFIG = None  # Native Config mapping or remote YAML path.
CLASSIFIER_TEACHER_DESCRIPTOR = None  # EfficientNet Keras artifact descriptor.
NOISE_TEACHER_DESCRIPTOR = None  # Native U-Net Config descriptor.

# Save EfficientNet with its optimizer/loss/metrics and express the U-Net's
# optimizer/loss in its native Config, with use_ema=False/test_network_name="raw".
# Their own compile settings are preserved.
# Set distillation type/coefficient/accuracy settings in the student model's
# wrapper_kwargs; every active route keeps those supplied constants.
"""), 
        _cell("markdown", "search-space", """## Search design

The architecture, diffusion timestep count/noise schedule, V1 wrapper, raw-network
selection and active distillation coefficients stay fixed to the input.
Two behavior controls remain worth testing under continual replay: classifier
training noise (`clean`, `noisy32`, `noisy128`, `full`) and unconditional-label
probability `{0.05, 0.1, 0.2}`. They change the learning/replay balance without
changing the student graph. Evaluation remains clean raw-network classification.

| Dimension | Search |
| --- | --- |
| Classifier teacher source | none, previous student, current EfficientNet, both |
| Noise teacher source | none, previous student, current U-Net, both |
| Continual strategy | generative replay, cumulative, new-only |
| Replay budget | legacy per class or fixed total |
| Legacy samples per old class | 100, 500, 1000, 2500, 5000 |
| Legacy training exposure | -1, 1000, 2500, 5000, 7500, 10000 |
| Fixed total old/current exposure | independently 100, 500, 1000, 2500, 5000 |
| Replay selection | all, uniform, confidence, surprise, confidence_surprise |
| Candidate multiplier | 1, 2, 4 when selection filters candidates |
| Combined surprise weight | uniform [0, 1] |
| Replay sampling | steps 20/50/100; CFG uniform [2.5, 5]; eta 0/1 |
| Batch size | 32, 64, 128 |
| Optimizer | Adam or AdamW |
| Learning rate | logarithmic [3e-4, 5e-3], constant schedule |
| AdamW decay | logarithmic [1e-6, 1e-3] |
| Clipping | per-variable None/0.5/1/5; global same choices only without per-variable clipping |

The two independent teacher axes cover **all 16 combinations**, including no
distillation, classifier-only, noise-only and both. Previous teachers apply to
old examples; current specialists apply to new examples. Both combines these
disjoint row scopes. Task one has no previous snapshot, so previous-only terms
activate from task two. Inactive terms are disabled; their zero contribution is
not a newly searched coefficient. Search dimensions are conditional: replay
controls have no effect for a strategy without generated replay, and source
choices activate only their required teachers. New-only is excluded when a
previous-teacher route is active, because that treatment requires old-task rows.

The initial queue includes each requested source pair once. The coverage table
reports attempts and **finite COMPLETE** outcomes separately: a queued, failed or
OOM-pruned mode is not evidence that the mode was evaluated successfully.
"""), 
        _cell("code", "settings", """DATASET = "CIFAR10"
RESULTS_PATH = "files/results/dit_continual_hpo_v1"
VALIDATION_SOURCE = "split"
VALIDATION_RATIO = 0.2
GPU_IDS = [0]
CONCURRENT_TRIALS = 1
WORKER_GPU_MEMORY_LIMIT_MB = 12288
EPOCHS = 50  # Per task, for the student and each active current specialist.
N_STARTUP_TRIALS = 40
SEARCH_SEED = 42
TASK_SEED = 42

SEARCH_SPACE_OVERRIDES = {}  # Only supported behavioral/replay/optimization dimensions.
REVIEW_TARGET = 200
SEARCH_TARGET = 1000
EXTENSION_TARGET = SEARCH_TARGET + 50
MAX_ATTEMPTS = 5000
BATCH_TRIALS = max(16, 4 * CONCURRENT_TRIALS)
EXPERIMENT_HOURS = None
CONFIRMATION_RESERVE_HOURS = 2.0  # Used only with an explicit experiment budget.
TOP_K = 3
CONFIRMATION_SEEDS = [101, 202, 303]

from common.config import resolve_continual_schedule


preview_class_order, preview_task_groups = resolve_continual_schedule(
    10, available_class_num=10, task_size=2, class_order_mode="random", seed=TASK_SEED
)
display({"dense_to_original_label": dict(enumerate(preview_class_order)), "task_groups": preview_task_groups})
"""), 
        _cell("markdown", "validation-notes", """## Seal inputs, task order and validation protocol

The default stratified training holdout supplies HPO feedback. Official test
scores do not choose trials. Changing model inputs, task seed, split, objective
or search distributions requires a fresh results directory. Trial targets and
measured worker placement can change when resuming the same scientific recipe.
Performance pruning is disabled because epoch numbers are reused across tasks
and specialist phases. The shared OOM handling and finite final-objective checks
remain active; checkpoints and failure logs are preserved.
"""), 
        _cell("code", "plan", """from common.dit_continual_hpo import SEARCH_SPACE


# Missing inputs must stop before allocating any study trials.
if STUDENT_CONFIG is None or CLASSIFIER_TEACHER_DESCRIPTOR is None or NOISE_TEACHER_DESCRIPTOR is None:
    raise ValueError("Fill the student, EfficientNet and U-Net input cells before creating the study.")

CONTINUAL_PROFILE = {
    "student_config": STUDENT_CONFIG, 
    "specialist_teacher_descriptors": {
        "classifier": CLASSIFIER_TEACHER_DESCRIPTOR, "noise": NOISE_TEACHER_DESCRIPTOR
    }, 
    "task_seed": TASK_SEED
}
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
display(SEARCH_SPACE)
display(plan["hpo"]["continual_profile"]["task_groups"])
display(plan["hpo"]["continual_profile"]["artifact_sha256"])
display(plan["hpo"]["continual_profile"]["specialist_teacher_descriptors"])
display(plan["identity"]["gpus"])
print("Persistent study:", plan["study_root"])
display(budget_summary(plan))
"""), 
        _cell("markdown", "tensorboard-notes", """## TensorBoard

Search and confirmation workers use the existing TensorBoard callbacks and
continual task/class/phase summaries. Teacher and replay phases remain visible
in their native artifacts. The notebook does not create a separate event writer.
"""), 
        _cell("code", "tensorboard", """TENSORBOARD_DIRECTORY = Path(plan["study_root"])
get_ipython().run_line_magic("load_ext", "tensorboard")
get_ipython().run_line_magic("tensorboard", f"--logdir {TENSORBOARD_DIRECTORY} --port 6006")
"""), 
        _cell("markdown", "review-notes", """## Initial search and review

Only finite COMPLETE trials count toward the target; every allocated attempt
counts toward `MAX_ATTEMPTS`. Interrupting preserves study/checkpoint evidence.
Rerun this cell to resume. Review all teacher-mode coverage and failures before
expanding the search; successful coverage is not guaranteed by the attempt cap.
"""), 
        _cell("code", "search-review", """review = run_search(plan, target_completed=REVIEW_TARGET, max_attempts=MAX_ATTEMPTS, batch_trials=BATCH_TRIALS)
display(review)
"""), 
        _cell("markdown", "main-search-notes", """## Extend the same study

Run the next cell after inspecting the initial results. Targets are total finite
completions, so rerunning a reached target creates no duplicate trials.
"""), 
        _cell("code", "search-main", """search = run_search(plan, target_completed=SEARCH_TARGET, max_attempts=MAX_ATTEMPTS, batch_trials=BATCH_TRIALS)
display(search)
"""), 
        _cell("code", "search-optional", """RUN_EXTENSION = False
# The extension is optional after reviewing the main target.
if RUN_EXTENSION:
    display(run_search(plan, target_completed=EXTENSION_TARGET, max_attempts=MAX_ATTEMPTS, batch_trials=BATCH_TRIALS))
display(search_summary(plan))
"""), 
        _cell("markdown", "confirmation-notes", """## Freeze and confirm finalists

Freeze the top distinct configurations by **maximum validation final average
accuracy** only after the desired search is complete. Selection becomes immutable;
further search then requires a new experiment. Inspect incomplete teacher modes
before making comparative claims.

Each finalist uses the same fresh training seeds, architecture/initial artifact,
specialist recipes, validation split and five task groups. The existing complete
training API repeats the entire continual sequence. Confirmations use one admitted
worker per selected GPU, preserving completed pair receipts on restart. Only
complete paired seed sets support mean comparisons. These repeated validation
scores remain selection estimates, not a new untouched test evaluation.
"""), 
        _cell("code", "freeze-finalists", """finalists = freeze_finalists(plan, CONFIRMATION_SEEDS, top_k=TOP_K)
display(finalists)
"""), 
        _cell("code", "confirm-finalists", """confirmation_records = run_confirmations(plan)
display(confirmation_summary(plan))
"""), 
        _cell("markdown", "artifacts", """## Saved evidence

The study keeps immutable recipe/input hashes, frozen task groups, Optuna
parameters and trial states, input/resolved YAML, per-task recovery checkpoints,
validation/test accuracy matrices, continual metrics, TensorBoard, worker logs,
finalist manifests and authenticated confirmation receipts. Inspect the native
continual matrices for forgetting and backward transfer alongside the selected
final-average-accuracy objective. No training has been executed in this template.
""")
    ]
    return {"cells": cells, "metadata": deepcopy(template["metadata"]), "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    """Write only the maintained continual runner notebook."""

    destination = ROOT / "DiT_Continual_HPO_Runner.ipynb"
    destination.write_text(json.dumps(make_notebook(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination)


# Importing the builder never rewrites the maintained notebook.
if __name__ == "__main__":
    main()
