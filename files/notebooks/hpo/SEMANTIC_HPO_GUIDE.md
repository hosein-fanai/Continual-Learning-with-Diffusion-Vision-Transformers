# Semantic consolidation HPO in continual learning

[Semantic_Consolidation_HPO_Runner.ipynb](Semantic_Consolidation_HPO_Runner.ipynb)
searches semantic acquisition/consolidation around a **fixed native DiT CLF and
a fixed complete continual-learning recipe**. This is the recommended first
study: allowing replay, joint learning rate or teacher sources to change would
make the best result harder to attribute to semantic consolidation. Those
settings can be investigated later in an explicitly separate interaction study.

The notebook follows the generation runner's remote setup, admitted isolated
workers, persistent Optuna study, TensorBoard, immutable recipe, resumable targets,
frozen finalists and paired seed confirmations. It uses the existing semantic
runner through the shared HPO API. It does not introduce another training loop.
Generate it remotely with:

```bash
python files/notebooks/hpo/generate_semantic_runner.py
```

The notebook requires the **matching complete repository checkout** on the
online container. Before setup, upload or synchronize the unpublished changes,
including `common/semantic_hpo.py`, `common/semantic_hpo_runner.py`, their shared
HPO/worker/confirmation integration, and the matching `semantic_consolidation`
sources. Use an isolated checkout so this does not overwrite another campaign's
live code or outputs. Uploading only the notebook does not install its helpers.
Until these changes are published, cloning GitHub `main` through the inherited
bootstrap will not supply them. Start the notebook from the prepared matching
checkout and verify its source/runtime identity before search.

## Inputs and fixed scientific protocol

Supply `STUDENT_CONFIG` as a native `common.config.Config`, mapping or remote YAML
path containing both the architecture and the entire CL recipe. Inputs are
intentionally unset in the delivered notebook. No substitute student, replay
budget or task schedule is chosen on the user's behalf.

The following remain fixed across search trials:

| Group | Fixed inputs |
| --- | --- |
| Student | DiT width, depth, patches, attention, embeddings, classifier projection/head, dropout, conditioning and every other architecture setting |
| Diffusion | Horizon, noise schedule, prediction/loss configuration and raw-network policy; EMA stays disabled as required by the native semantic route |
| Task protocol | Dataset, class vocabulary, resolved class order/task groups, task seed, dataset split and source, original training/validation rows |
| Continual memory | Generated replay enabled/disabled where supported, old/current exposure, replay mode/selection/candidate budgets, buffer policy and snapshots |
| Joint training | Per-task epochs, joint batch size, optimizer, learning-rate schedule, decay, clipping, loss/KD coefficients and teacher settings |
| Mechanism | Learned route condition, acquisition objective, consolidation update scope and retention policy |
| Reporting | Raw validation final average accuracy as selection objective; diagnostics and checkpoint policy remain declared constants |

The native semantic implementation accepts CIFAR-10 and CIFAR-100 here. It
requires a compatible V1 `diffusion_classifier`, a positive semantic classifier
projection, CFG, standardized preprocessing, float32 and fresh training.
`training.task="continual"`, `fit_method="fit"`, `patience=0`, complete task
epochs, no real replay buffer, and replay mode `fixed_total` or `match_current`
are native requirements. Retained old-class modulators require positive replay
pairs. A trained student checkpoint or an unrelated continual-HPO recipe is not
silently converted into an admissible semantic experiment.

Fixed specialist teacher artifacts are supplied through native
`continually_learn.specialist_teacher_descriptors`; their architecture/artifact
identities and supported teacher/KD settings are retained. Fresh student training
does not prohibit using the declared fixed teacher artifacts.

The HPO profile requires `experiment_phase="development"`,
`validation_source="split"`, positive `validation_ratio`, and validation enabled.
The notebook derives epochs and validation fraction from the native input. It
does not impose the generation runner's epoch count. Dataset/task seeds remain
fixed while training randomness uses the search seed and then fresh confirmation
seeds. Inspect the sealed profile printed before launching search.

`ROUTE_SETTINGS` accepts a `RouteSettings`, mapping or standalone settings YAML
(raw fields or a single `route:` envelope). A combined `base_config/common/route`
YAML is a different native format; do not pass it as standalone settings. The
notebook's suggested fixed mechanism is:

```yaml
condition: learned
acquisition_objective: contrastive
consolidation_scope: semantic
retain_modulators: true
extensions: {}
experimental: {}
```

The semantic phase's batch size and Adam learning rate are distinct from the
frozen joint-training batch and optimizer. Semantic-only consolidation updates
the native classifier projection/head and temporary predictor; its shared
diffusion backbone remains fixed during that phase. Joint training still follows
the supplied native recipe.

## Exhaustive implemented search catalog

The authoritative registry is
[`common/semantic_hpo.py`](../../../common/semantic_hpo.py): `SEARCH_SPACE`
defines distributions and `FIELD_CATALOG` classifies **every** current
`RouteSettings` field. The notebook renders this registry before the missing-input
guard, so its full catalog can be inspected without supplying a model. The tables
below enumerate all 18 potential axes. Conditional axes may be absent from a
particular trial. Ranges are broad development choices, not literature-derived
optimal values or a claim that the entire space has been evaluated.

| Native route field | Actual sampled domain | Condition and interpretation |
| --- | --- | --- |
| `acquisition_steps` | 50, 100, 200, 400, 1000 | Positive semantic acquisition optimizer updates **per task**; zero is a separate removal control |
| `consolidation_steps` | 100, 200, 400, 800, 2000 | Positive consolidation updates per task; independent of joint epochs |
| `batch_size` | 16, 32, 64, 128 | Semantic phases only; native balanced sampling may reduce effective size when positive pairs are scarce |
| `learning_rate` | Log-uniform `[1e-5, 3e-3]` | Shared semantic Adam rate; joint optimizer untouched |
| `temperature` | Log-uniform `[0.03, 0.5]` | Contrastive semantic alignment temperature |
| `alignment_weight` | Log-uniform `[0.1, 10]` | Alignment contribution; zero removes the term and belongs to an ablation |
| `ce_weight` | Log-uniform `[0.1, 10]` | Supervised consolidation contribution; positive within this method |
| `orthogonality_weight` | Log-uniform `[0.1, 10]` | Sampled only with fixed `acquisition_objective="contrastive"`; inactive for fixed true-class CE acquisition |
| `gain_limit` | Log-uniform `[0.1, 2]` | Positive affine modulation gain bound |
| `bias_limit` | Log-uniform `[0.1, 2]` | Positive affine modulation bias bound |
| `modulation_init_std` | Log-uniform `[0.001, 0.1]` | Initialization scale of semantic modulators |
| `acquisition_noise_level` | 0, 10, 50, 100, 250, 500 | Keep only indices below the fixed diffusion horizon; 0 is truly clean |
| `ce_noise_level` | 0, 10, 50, 100, 250, 500 | Independent consolidation CE noise index, filtered by the same horizon |
| `noise_levels` | `0`; `10`; `50`; `100`; `250`; `500`; `0,10,50`; `0,50,100`; `0,100,250`; `0,250,500` | Categorical comma-separated encodings resolve to integer level sequences; each index must fit the fixed horizon |
| `image_augmentation` | `none`, `tmcl` | Native same-input alignment or the native RGB32 image-view policy; no invented augmentation pipeline |
| `augmentation_views` | 2, 4, 8 | Sampled only for `image_augmentation="tmcl"`; view count changes computation |
| `reliability` | `uniform`, `alpha_bar` | Sampled only if any consolidation alignment noise level is positive; both give unit reliability on exclusively clean views |
| `reliability_floor` | 0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99 | Sampled only for positive alignment noise and `alpha_bar`; a floor below all selected schedule weights has no effective clipping |

These domains include clean/noisy acquisition, independent supervised noise,
single/multiple alignment noise views, reliability policies, augmentation,
semantic capacity through affine bounds, objective balance and phase budget.
They preserve the model graph. Absolute diffusion indices are intentional:
changing the supplied noise schedule changes their signal-to-noise meaning,
and therefore requires a new study. The adapter filters indices rather than
silently rescaling them.

Importance plots require care. A floor that never binds is behaviorally
equivalent to another floor, semantic batches can collapse to the same effective
size, and categorical branches have different parameter sets. Inspect resolved
configs, effective phase statistics and sampled coverage before interpreting
parameter importance as a mechanistic effect. Varying alignment/CE weights and
learning rate can also create similar effective updates.

All remaining native route fields are explicitly accounted for:

| Field | Classification | Policy in this notebook |
| --- | --- | --- |
| `condition` | Fixed method | `learned`; baseline, random gates, feature-distillation and joint-only controls are separate studies |
| `extra_joint_seconds` | Fixed measured ablation | Unused by the learned method; time-matched controls require actual measured per-task allowances |
| `consolidation_scope` | Fixed ablation | Suggested `semantic`; `backbone` changes the intervention and must be separately declared |
| `acquisition_objective` | Fixed ablation | Suggested `contrastive`; `true_class_ce` is a separate declared comparison |
| `retain_modulators` | Fixed ablation | Suggested `true`; changing retention changes the memory protocol |
| `probe_batches` | Fixed diagnostic | Number of observed probe batches; input/default preserved |
| `probe_max_gates` | Fixed diagnostic | Limits measured gate coverage, not the size of the training gate bank |
| `seed` | Provenance, never optimized | Search training seed; fresh paired training seeds for confirmations |
| `extensions` | Fixed mapping | Native scheduling, replay, evaluation, reselection, quality-split and KD-allocation treatments preserved; not extra hidden HPO axes |
| `experimental` | Fixed diagnostic mapping | Declared experimental diagnostics stay fixed and cannot make held-out rows training data |
| `checkpoint_interval` | Fixed operational | Native recovery frequency; not an accuracy dimension. Explicit new HPO attempts start fresh; interrupted `RUNNING` states require reconciliation |

Use `SEARCH_SPACE_OVERRIDES` only to restrict supported semantic domains, with
a new results directory if distributions change. Architecture, CL, optimizer
and arbitrary extension keys are not accepted as additional semantic axes.
The proposed main study keeps extension/experimental mappings empty; enabling
them creates an explicitly different fixed experimental context. The exhaustive
catalog accounts for these mappings as whole fixed inputs; it does not claim
to tune every nested scheduling, replay, allocation or diagnostic option.

## Why this is not a full Cartesian grid

With horizon at least 501 and the default contrastive mechanism, the discrete
choices alone have

`5 * 5 * 4 * 6 * 6 * (1 + 9 * (1 + 8)) * (1 + 3)`

possible combinations: **1,180,800** before the eight
continuous dimensions. The factors account for phase budgets, batch, two
independent noise indices, clean versus noisy alignment/reliability branches,
and conditional augmentation views. Discretizing each continuous dimension to
just three points gives `3**8 = 6,561` and **7,747,228,800** settings. Some are
behaviorally equivalent because a reliability floor may not bind. The continuous
space itself has no finite exhaustive enumeration.

An exhaustive **catalog** is useful; evaluating every combination is infeasible.
Use the shared TPE sampler, whose documented define-by-run support accommodates
categorical and conditional domains. A 200-success budget is a practical staged
search proposal, not a coverage or convergence guarantee. The relevant behavior
is documented in the [Optuna sampler reference](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html).

## Search, confirmation and controls

| Stage | Default | What it establishes |
| --- | --- | --- |
| Pilot | First 12 finite completed full streams | Real timing, phase cost and failures; not convergence or all-branch coverage |
| Review | 40 total successful trials | Initial branch/metric review near the 40-observation TPE startup setting |
| Main | 200 total successful trials | Broad conditional optimization; review coverage and diminishing gains |
| Attempt ceiling | 800 total attempts | Bounds repeated failures/OOMs; does not guarantee the success target |
| Optional extension | 400 total successes, disabled by default | Continue the identical sealed study before freezing finalists |
| Finalist confirmation | Top 3 distinct recipes × seeds 101/202/303 | Nine fresh full-stream validation repeats with matched split/task order |
| Separate no-semantic baseline | Same three training seeds | Three additional streams needed to compare semantic benefit; not launched automatically |

Each trial completes its sequential task stream. `final_average_accuracy` is
computed from the final validation accuracy matrix. Inspect forgetting and
backward transfer alongside this primary metric; a higher final average does
not alone establish reduced forgetting for every task. Do not prune a trial on
an early task's accuracy: joint epochs and semantic phases are not interchangeable
resources and forgetting appears later. Performance pruning is disabled;
recognized OOMs retain their resource-pruning evidence, and nonfinite objectives
are not successful trials.

Search targets resume from committed study results; interrupted `RUNNING` trials
require explicit reconciliation before continuing. The notebook does not
automatically restart a stranded worker or restore an interrupted semantic trial
mid-phase. Preserve the native recovery artifacts and failure evidence. An
explicitly authorized new HPO attempt runs the complete stream from fresh
initialization.

Freeze finalists only after search is complete. Confirmations restart from
fresh initialization and use the same fixed task groups and dataset seed. They
vary training and semantic randomness, not task order or validation membership.
All three seeds must finish before comparing paired means. Report each seed,
mean, standard deviation, per-task matrices and compute. Three seeds give limited
precision; later robustness work can add independent task-order blocks in a new
declared experiment.

The finalists are all learned semantic recipes. Their success is not evidence
that semantic consolidation beats the same CL system without semantic phases.
Run a separate `RouteConfig` with `condition="baseline"` through the existing
native semantic runner, paired on frozen architecture/CL settings, class groups,
dataset partition and the three training seeds. Keep output paths separate.
The notebook's confirmation cell does **not** run this baseline.

For stronger interpretation, add selected controls supported by the native
route: random/identity modulation, no consolidation, modulated or unmodulated
feature distillation, true-class CE acquisition, changed update scope or gate
retention. Positive-only HPO domains deliberately exclude term/phase removal.
Do not optimize controls jointly with the learned method and then treat their
unequal tuning budgets as a balanced mechanism comparison.

Phase budgets vary within HPO. A winning recipe may spend more optimizer
updates or use more augmented/noise views. Compare a measured time-matched
joint-only treatment before attributing improvement specifically to semantic
structure. Native `semantic_consolidation.controls` can prepare control studies;
its time-matched allowances must come from actual matching pilot phase records.
Its broader seed/task-order design must be reviewed separately rather than
assumed identical to this HPO's fixed-order confirmations. Official test data
remain untouched until a final selected recipe and evaluation protocol are frozen.

## GPU choice and honest duration planning

Begin with **one available A100 80 GB or H100 80 GB, one admitted worker**, using
the notebook's requested **24 GiB TensorFlow cap**. These are available
large-memory GPU classes in the user's recorded remote pool. NVIDIA specifies
80 GB HBM2e for A100 80 GB and 80 GB for H100 SXM; those capacities alone do not
establish model fit or concurrency. See the primary
[A100 specifications](https://www.nvidia.com/en-us/data-center/a100/) and
[H100 specifications](https://www.nvidia.com/en-sg/data-center/h100/).

Before new execution, verify the live endpoint, GPU UUID/model, checkout/source
and framework identity, free GPU/container RAM, existing jobs and coordinated
reservations. Keep TensorFlow 2.20.0/Keras 3.11.2. Historical ready/idle readings
are not current capacity. All task computation, including CPU validation, runs
on the supplied online containers. No local laptop fallback is allowed.

The 24 GiB cap is **unmeasured for the supplied input**. The real peak includes
the DiT CLF, optimizer state, previous teacher snapshot, semantic target model,
modulator bank, feature/predictor state, replay generation and multi-view batches.
Larger late-task class banks, batch 128, eight augmentation views and multiple
noise levels need representative checks. CPU memory and storage can also limit
parallelism. If a requested budget fails, retain the failure and deliberately
revise the resource plan; do not quietly shrink the scientific recipe.

After measured checks, use more independent workers on the same GPU or spread
trials over additional supplied GPUs, preserving other jobs. There is no fixed
numerical per-GPU cap; measured fit, ownership, overhead and throughput decide.
The shared confirmation scheduler runs one repeat per selected GPU, so packed
search throughput is not automatically confirmation throughput. A single
coordinator owns a study's database. Do not start competing coordinators against
shared storage or assume that `GPU_IDS` selects GPUs in other SSH containers.
Multiple containers require the established coordinated execution setup; the
notebook directly addresses selected devices visible to its own container.

There is **no measured semantic-HPO runtime yet**: input model/CL recipes are
unset, and preparation validation is not a real-data GPU benchmark. Neither the
generation runner's 51-worker H100 result nor its full-stream timing applies.
H100's peak hardware numbers cannot establish an application speed ratio over
A100. Benchmark each selected hardware class under actual concurrency.

Measure these on the pilot:

1. Full stream elapsed time, plus joint training, replay, acquisition,
   consolidation, evaluation and checkpoint phases.
2. Peak GPU and container RAM, especially the final task and largest retained
   semantic batch/view combination.
3. Aggregate finite-completion throughput at the intended simultaneous worker
   count; summing isolated theoretical capacities misses contention.
4. Failed/pruned worker hours, queue time and artifact/storage overhead.

The notebook's `cost_estimate` reads finite completed full-stream trial durations
and reports median/p90, consumed failed/pruned worker time, and remaining search
plus confirmation estimates. Before any completed pilot duration exists it
returns no invented timing. Its forecasts assume comparable hardware and
concurrency and omit future failures, queues and interruptions. Confirmations
use the selected GPU count when estimating batches.

For identical workers, a rough planning formula is:

`remaining hours ≈ ceil(remaining successes / measured search workers) * stream hours`

`+ ceil(finalist repeats / confirmation workers) * confirmation stream hours`

`+ separately scheduled baseline hours + failed-attempt/queue/setup allowance`.

Prefer directly measured aggregate successes/hour when duration varies strongly
by branch. At heterogeneous GPU rates, add measured full-stream completion
rates for a rough aggregate rate; keep separate rates for differently sized
trials and do not pool VRAM. Pilot median and p90 give useful planning scenarios,
not confidence limits on future completion time.

For an **illustration only**, assume two hours per full stream, one effective
worker per GPU, and identical throughput. The entire main design is 200 search
+ 9 finalist + 3 separately executed baseline streams = 424 GPU-hours. A simple
25% allowance produces 530 GPU-hours:

| GPUs at the assumed rate | Ideal 424 GPU-hours | With illustrative 25% allowance |
| --- | --- | --- |
| 1 | 17.7 days | 22.1 days |
| 3 | 5.9 days | 7.4 days |
| 5 | 3.5 days | 4.4 days |

The first 12 pilot trials are included in the 200, not additional. Thirty-minute
streams divide these times by four; eight-hour streams multiply them by four.
Real trial variation, wave rounding, lower confirmation concurrency, admission
waits and failures can increase elapsed time. These are neither CIFAR-10 nor
CIFAR-100 benchmarks. CIFAR-100 needs a separate pilot because its supplied task
count, growing classifier/modulator state, replay exposure and evaluation work
can differ substantially; dataset name alone is insufficient for a multiplier.

For a constrained budget, complete the 12/40 stages, inspect evidence, then set
a smaller final success target before freezing. Do not reduce epochs, task
count, replay or architecture halfway through the same study. The notebook has
`EXPERIMENT_HOURS=None` by default, respecting the removed campaign cutoff.
If explicitly adding a persistent wall budget, reserve confirmation time using
pilot observations; `CONFIRMATION_RESERVE_HOURS=0` is only a placeholder while
no deadline is active. An experiment budget does not shut down a rented container.
No rental-price quote is supplied; use a current provider quote and include idle
setup, storage and cleanup when converting measured GPU-hours to cost.

## Validation and evidence boundaries

Regenerate and validate notebook/source on an authorized online container with
the pinned framework. Confirm that schema and all code cells compile, the
registry covers every route field, inputs stay immutable, generated routes run
through the actual adapter, trial/confirmation metrics match, and missing model
inputs stop before any study launch. Use focused remote tests in a separate
process. These preparation checks establish integration behavior, not scientific
improvement, an exhaustive sweep, GPU capacity or wall-time performance.

Preserve the complete results directory: immutable recipe, source/environment
identity, study database and recovery artifacts, input/resolved configs, route
settings, per-task matrices/checkpoints, phase timings, failure logs, TensorBoard,
frozen finalists and paired confirmation receipts. Changes to scientific inputs,
search distributions or fixed task/split identity require a new directory.
