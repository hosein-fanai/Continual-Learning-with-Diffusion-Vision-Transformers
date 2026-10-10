# Hyperparameter optimization notebooks

## UNet denoiser campaign runner

[UNet_Generation_HPO_Runner.ipynb](UNet_Generation_HPO_Runner.ipynb) mirrors the
DiT generation runner's persistent search, TensorBoard, pruning, recovery and
fresh-seed confirmation workflow for the native convolutional UNet. Its opt-in
`denoiser_v1` domain is defined in `common/unet_hpo.py`; the notebook documents
all sampled and fixed controls. Existing generic UNet and DiT distributions
remain unchanged. Deploy the matching `common/hpo.py`, `common/unet_hpo.py`,
`common/dit_hpo_runner.py`, `common/dit_hpo_confirmation.py`,
`common/unet_hpo_storage.py` and the matching shared storage/recovery helpers
with the notebook. The dedicated deployment contains all dependent packages.

The domain covers seven 2–4-level width hierarchies, residual/bottleneck depths,
embedding budgets and allocations, BatchNorm, spatial dropout, activations,
independent down/up sampling with conditional interpolation, Adam/AdamW,
learning rate/decay, batch size, global clipping, EMA and CFG label dropout.
Skips and linear output stay enabled; fixed clipped-cosine/1,000-step noising,
MSE and zero auxiliary image loss keep denoising objectives comparable.
Unsupported attention/GroupNorm/kernel-size axes are not invented. Sampling
settings require separate finalist image-quality evaluation rather than loss HPO.

Defaults use CIFAR-10 with a seeded 20% training holdout, keeping the official
test set out of tuning. Search reviews 12 successful trials and targets 60,
with an explicitly disabled optional extension to 100 and 240 total attempts.
Trials keep the 50-epoch maximum and patience 5. Median pruning begins after
12 completed references, epoch 10, then every five epochs. The top two recipes
receive paired fresh seeds 101/202/303, without performance pruning.

The current default uses **two A100 80 GB GPUs for at most 12 hours**:
nine hours available to search and a three-hour confirmation reserve.
`GPU_IDS=[0, 1]` and `CONCURRENT_TRIALS=2` place one independent worker on each
device, with a requested **24-GiB TensorFlow cap per worker**. These are spending
allowances, not completion guarantees; the notebook estimates throughput from
its pilot. Both deployed A100s passed bounded synthetic UNet train/test checks;
full-batch capacity and search throughput remain unmeasured.
Increase concurrency only after representative measured checks. At the Runpod
public A100 rate of $1.79/GPU-h viewed 2026-10-10, 24 GPU-hours would cost about
$42.96, excluding setup, idle rental, storage and cleanup. Verify the live quote
at [Runpod pricing](https://www.runpod.io/pricing). The notebook never shuts down
a rented container.

The dedicated deployment is `/workspace/UNet-HPO`; select kernel
**Python (UNet TF2.20)** (`unet-tf220`) using `/opt/unet-tf220/bin/python`.
Its `/workspace` storage uses geesefs, so notebook setup prepares active SQLite
under node-local `/tmp/unet-hpo-sqlite` and publishes durable snapshots to the
shared study directory. Keep the complete result directory and its snapshots;
container replacement requires setup to restore the committed snapshot before
resuming. No HPO has been launched on this deployment.

Deployment validation passed 94 focused tests, notebook schema/setup, and full
source contracts for nine support files plus all 12 notebook code cells.
Each A100 passed two deep UNet architectures, each with two training updates and
one EMA evaluation on synthetic batch-eight data. These checks used
`tf.function(..., jit_compile=False)` and do not certify full HPO or XLA behavior.
Evidence: `files/results/unet_hpo_deployment_20261010/REPORT.md`.

Earlier preparation validation used a separate remote copy, Python 3.12.3,
TensorFlow 2.20.0, Keras 3.11.2 and Optuna 5.0.0, with GPUs hidden. All 64
focused UNet/runner/confirmation tests passed; all 19 cells passed schema checks
and all 12 code cells compiled and passed argument/spacing checks. Actual
setup/settings passed without importing TensorFlow, creating a study database
or starting its clock. Nine native scaler combinations each passed one small
eager CPU training/evaluation step through the public model factory. This is
not a full HPO, GPU/XLA capacity test, exhaustive architecture test or runtime
benchmark. Six focused source files passed their full static contracts; the
shared hpo.py snapshot also contained 11 existing missing case comments in
concurrent continual-profile changes, recorded separately in the evidence.

Evidence: `files/results/unet_hpo_preparation_20261010/`. Regenerate notebook
source remotely with `python files/notebooks/hpo/support/generate_unet_runner.py`.
The generator reuses maintained DiT control cells and does not start a study.

These notebooks are thin, reproducible entry points to the shared
`common.hpo` API. Each notebook explains one supported model/task
pair, exposes the same editable constants, displays its constrained search
space, runs the study, and reports the best trial or Pareto front. Outputs are
initially empty. Generate the 24 notebooks from the maintained template with
`python files/notebooks/hpo/generate_notebooks.py`; the generated task directories
are local artifacts. The generator embeds the shared runtime setup before
importing `common.hpo`.

Current studies use search-space version 14 and seal training-semantics
version 5. Search version 14 names transformer stochastic depth and classifier-head
dropout separately; version 13 introduced conditional `global_clipnorm` choices
alongside per-variable `clipnorm`. Training-semantics version 3 introduced isolated,
paired seeded final diffusion evaluation; version 4 also excludes nonfinite final
objectives from every study mode, retaining divergence evidence before pruning. Version 5
records recognized memory exhaustion as resource pruning. Earlier study specifications cannot
resume into the current search and evaluation contracts. Start a
new study in a fresh `RESULTS_PATH` (for example,
`files/results/hpo_semantics5`) and preserve its predecessor. The generic API's
stable default path does not authorize reusing an incompatible existing study;
do not change old `study_spec.json` fields to authorize mixed-semantics trials.

## Fixed-recipe semantic consolidation runner

[Semantic_Consolidation_HPO_Runner.ipynb](Semantic_Consolidation_HPO_Runner.ipynb)
adds semantic acquisition/consolidation HPO through the native semantic runner,
with the same remote setup, admitted isolated workers, persistent study,
TensorBoard and paired finalist confirmation lifecycle as the generation runner.
Supply a compatible native DiT CLF configuration containing the architecture and
complete continual-learning recipe. Architecture, task stream, replay,
teachers/KD, joint optimizer/batch and epochs stay fixed; only the semantic
phase controls are optimized. CIFAR-10 and CIFAR-100 use their supplied task
groups. Inputs intentionally remain unset.

The executable notebook table covers every `RouteSettings` field. Eighteen
potential semantic axes cover phase update budgets, semantic batch/rate,
temperature, objective weights, modulation bounds/initialization, independent
noise views, augmentation and conditional reliability. Mechanistic removals,
scope/retention changes, extension settings and diagnostics are explicitly fixed
or assigned to separate ablations. An exhaustive field catalog does not mean
that the continuous search space is exhaustively evaluated.

Defaults pilot 12 full-stream successes, review 40 and target 200, then confirm
the top three distinct recipes with three paired fresh training seeds while
retaining the dataset split and task order. Search maximizes held-out training
validation final average accuracy; official test scores do not select trials.
No-semantic baseline runs are separately required for efficacy comparisons and
are not silently included in finalist confirmation. Epoch pruning is disabled;
OOM/nonfinite outcomes retain failure evidence.

Start with one remote A100/H100 80 GB worker and a requested 24-GiB cap, then
measure fit and full-stream throughput. This is not a GPU capacity certificate.
The pilot timing helper forecasts remaining search and confirmation time from
actual completed trials. There is no global deadline by default and no claimed
CIFAR runtime before the fixed input has been benchmarked. The
[full search and GPU guide](SEMANTIC_HPO_GUIDE.md) includes every distribution,
conditional branch, fixed/ablation field, time formula and illustrative budget.
Regenerate remotely with
`python files/notebooks/hpo/generate_semantic_runner.py`.

Preparation passed 155 distinct remote tests, including two real isolated
semantic workers and a fresh-seed full-stream confirmation on tiny synthetic
data. All 13 notebook code cells and 13 Python source files passed their
applicable contracts. This does not establish production GPU fit or duration.
See [validation evidence](../../results/semantic_hpo_preparation_20261010/REPORT.md).

## Fixed-architecture DiT continual runner

`DiT_Continual_HPO_Runner.ipynb` uses the same remote bootstrap, immutable recipe,
admission, isolated workers, resumable search targets and paired finalist
confirmation lifecycle as the generation runner. Regenerate only this notebook
with `python files/notebooks/hpo/generate_continual_runner.py` on an authorized
remote container. The generic matrix generator does not replace it.

Supply the native DiT classifier configuration, compiled EfficientNet `.keras`
artifact and native generation/U-Net YAML in the explicit input cells. Inputs
are intentionally unset. Artifact hashes, architecture and active distillation
type/loss/accuracy coefficients are fixed. The student is the generator and
classifier through V1; EMA is disabled. Current specialists train on real new-task
rows through the existing `fit_teacher` API, keeping their own compile/optimizer
settings. `EPOCHS` applies to each task's student and active specialist fits.
Use a fresh or appropriately initialized expandable final Dense head on the
EfficientNet backbone. Existing trained head columns must already match the
resolved dense task order, where column `i` refers to original label
`class_order[i]`; arbitrary CIFAR-10 column order is not remapped automatically.
The notebook displays that mapping before validating the model inputs. Native
U-Net `num_classes=None` supports a growing vocabulary; a fixed condition
vocabulary requires the same known dense-label mapping in its initial weights.
The U-Net input must explicitly select `use_ema=False` and
`test_network_name="raw"` in its diffusion wrapper configuration; this is
validated rather than silently changing the supplied teacher recipe.

One seeded random partition fixes all ten CIFAR-10 labels into five two-class
tasks across trials and confirmations. The default validation protocol is a
training holdout (ratio 0.2); final validation average accuracy is maximized.
Confirmations vary training seeds while retaining the task groups and dataset
seed. They restart from the supplied original model initialization, not the
winning trial's trained weights.

Independent classifier/noise teacher sources `{none, previous, current, both}`
cover sixteen combinations. The initial queue contains all requested pairs;
the review separately counts allocated and finite-completed coverage. The
previous source uses a raw student snapshot on old-class rows; current sources
are EfficientNet and U-Net specialists on new-class rows. New-only is excluded
when a previous-teacher source needs old rows. Task one has no previous snapshot.
Disabling a term gates its contribution to zero without optimizing its input
coefficient. No layers are added to the input architecture: the wrapper uses its
existing distillation-token head when present and its existing primary-head
fallback otherwise.

The profile searches strategy, conditional replay budgets/exposures, selection,
candidate multiplication, surprise weighting, generation steps/CFG/eta,
batch/optimizer/learning rate/weight decay/clipping, classifier training noise
and unconditional-label probability. Diffusion horizon/schedule and graph
architecture remain fixed. `common.dit_continual_hpo.SEARCH_SPACE` is the
authoritative distribution table displayed by the notebook. Unsupported
architecture overrides fail explicitly.

Continual HPO can use multiple isolated trials per GPU and across selected
GPUs. Each trial's tasks remain sequential. Defaults are one worker with a
12-GiB TensorFlow reservation until the user supplies and measures the actual
student + snapshot + specialists + replay workload. Generation-only capacity
measurements do not establish continual capacity. The same allocator preserves
existing jobs and fails closed on uncertain ownership; no local computation is
permitted. There is no experiment cutoff by default. Epoch performance pruning
is disabled across task phases; recognized OOMs retain their pruning evidence,
and nonfinite final metrics do not count as completed trials.

**Validation:** 158 distinct focused tests passed on the authorized remote
Python 3.12.3 / TensorFlow 2.20.0 / Keras 3.11.2 runtime with GPUs hidden.
These include two real concurrent five-task V1 workers using synthetic data,
artifact-loaded per-task teacher training, recovery, and shared runner/SQLite
integration. The 20-cell notebook and 15 edited Python files passed schema/code
and source-contract checks. Inputs remain unset; no full HPO or model-capacity
benchmark ran. Commands, limitations and logs are in the
[validation report](../../results/dit_continual_hpo_implementation_20261010/README.md).

## DiT generation campaign runner

`DiT_Generation_HPO_Runner.ipynb` is a maintained runner for standalone
`diffusion_transformer` generation, using `common.dit_hpo_runner` and the public
`common.hpo.run_hpo` API. The generic notebook generator does not replace it.

The supplied three-H100 container targets `GPU_IDS=[0,1,2]` and 51 total search
workers (17 per GPU), with `WORKER_GPU_MEMORY_LIMIT_MB=3584` (3.5 GiB)
TensorFlow caps. Admission reserves another 1 GiB per worker and device
headroom; the requested cap must fit every selected GPU quota. Confirmations
retain this cap with one worker per GPU. The largest retained plain-DiT model
passed at batch 128 with all 51 workers: two bounded epochs/four steps each,
full official-test validation, TensorBoard and final reports. All 871 artifact
checks passed. Peak recorded TensorFlow allocation was 3457.67 MiB. This is a
bounded check, not a full 50-epoch or exhaustive guarantee; see the
[v10 capacity report](../../results/dit_hpo_v10_readiness_20261009/REPORT.md).

Separate one-worker screens at the same cap found TensorFlow OOMs when restoring
six heads, eight heads or depth7 at the largest batch128/patch2 settings. These
choices therefore remain excluded from this51-worker space.

The previous mixed-backbone v9 full-concurrency check **failed**: all 51 workers reached training, then
a largest U-DiT candidate at batch 128 raised a recognized TensorFlow OOM. The
benchmark stopped its own workers at the first failure and released all slots.
Six workers had passed at the same cap, including full official-test validation,
TensorBoard and reports; this does not establish 51-worker capacity. The requested
batch sizes remain `[32,64,128]`. Search OOMs are pruned, so some large candidates
may be excluded from comparison. Confirmation errors remain incomplete and
surface to the caller. See [the v9 capacity report](../../results/dit_hpo_v9_readiness_20261009/REPORT.md).
The historical 51-worker pass used a smaller space; v8's larger models instead
passed nine simultaneous largest-plain probes at 24 GiB. Preserve both records.

A single Optuna coordinator owns the database. One GPU, multiple trials on one
GPU, and multiple trials across GPUs remain supported. Set device selection and
concurrency to the measured capacity of other runtimes. Each trial uses one GPU.

The notebook targets 200 successful trials for review, 1,000 for the main search,
and an optional 50 more, with a separate 5,000-attempt ceiling. A persistent
20-hour wall-time budget starts at first search execution, permits at most
18 hours of search and reserves 2 hours for confirmations. It includes admission
waits and survives restarts; setup does not start the clock. Actual completed
counts depend on throughput and pruning. Completed results persist at cutoff;
active deadline cancellations are recorded separately from OOM pruning and are
not resumed as interrupted training. Cleanup receives a short time allowance.
The execution clock is stored in `notebook_runner/budget.json` separately from
the scientific recipe. Trial targets can change without restarting the clock.

The runner explicitly passes `SEARCH_SPACE_OVERRIDES` through `make_plan` to
`run_hpo`. Its search-space table lists the expanded CFG, patch/time/label
embeddings, position/condition merging, width/depth/attention, refiner,
adaptive-normalization MLP, final activation, MSE/MAE training-loss choices and
500/1,000/2,500/5,000 diffusion timesteps.
Optimizer, batch, diffusion-schedule, EMA and auxiliary-loss searches remain
in the shared API. The sampled compile loss may be MSE or MAE; validation,
early stopping, pruning and final HPO feedback remain EMA noise MSE using the
shared compile API's fixed evaluation loss, preserving comparable objectives.

The v10 plain-only restriction uses widths 16/32/64/128, plain depths 3/4/5/6,
four attention heads and `mha_key_dim=[None]` for the native default key width.
The explicit `batch_size=[32,64,128]` retains all three requested batch choices.
`dit_architecture_grid4=["plain"]` restricts both CIFAR-10 patch grids to plain
DiT; U-DiT is excluded. Time/label MLP ratios are sampled only when their frequency
width is explicit, because frequency None disables that embedding MLP. An
active embedding MLP treats None and 1 as the same one-times width. Adaptive
normalization instead omits its hidden Dense layer when `ln_mlp_ratio=None`.
The overrides are copied and sealed with the scientific recipe. Start v10 in its
fresh directory; preserve v9, v8 and recovered v7 without changing their distributions.

All runs use at most 50 epochs and early-stopping patience 5. TPE has a
40-observation random-startup threshold. The top three configurations receive
three paired fresh seeds, with one confirmation at a time on each selected GPU,
subject to the final deadline. Completed repeats are authenticated and skipped.
Only candidates with every required confirmation complete support a full seed
mean comparison; partial results remain available without pretending completion.

`VALIDATION_SOURCE="test"` and `VALIDATION_RATIO=0.0` train on all 50,000 official
CIFAR-10 training images and use the 10,000 official test images for epoch
validation, early stopping, pruning, final HPO feedback and confirmations.
Test scores participate in tuning and are not an untouched generalization
estimate. Other `make_plan` callers retain the `split`/0.2 default.

The shared API supplies Optuna percentile pruning on validation EMA noise MSE:
75th percentile, 40 completed reference trials, epoch10 warmup, five-epoch
intervals, and at least10 reference values at the epoch. Recognized TensorFlow
`ResourceExhaustedError` and Python `MemoryError` independently prune search
trials, preserve error/partial artifacts and TensorBoard diagnostics, and release
the slot for a replacement. Unclassified failures remain errors. Confirmations
retain early stopping but disable performance pruning; confirmation failures
remain incomplete and must not be treated as successful seed repeats.

TensorBoard uses the same notebook extension as `13_CIFAR10_Joint_HPO.ipynb`,
port6006, and existing training/HPO logging APIs. Its root includes the study
and confirmation attempt directories. No duplicate training or event writer is
implemented in the notebook. The new results path is
`files/results/dit_generation_hpo_v10`; preserve v9, v8, v7 and older studies.

Colab and Kaggle buttons open the published GitHub main revision. Publish the
notebook and matching source helpers together before using those URLs for a new
revision. The prepared remote checkout includes the local updates; select its
**Python 3 (ipykernel)** kernel, or another kernel with the required packages.
Setup verifies TensorFlow 2.20.0 / Keras 3.11.2
without replacing container packages. Keep the coordinator free of framework
imports. Save the complete results directory, including SQLite, sampler state,
configurations, artifacts and `notebook_runner`, to resume with matching source
and environment versions. Hosted runtime limits still apply.

## CIFAR-100 DiT generation campaign runner

[DiT_Generation_HPO_CIFAR100_Runner.ipynb](DiT_Generation_HPO_CIFAR100_Runner.ipynb)
mirrors the main CIFAR-10 generation runner using **CIFAR-100's 100 fine classes**.
It retains the same plain-DiT search space, training and evaluation APIs,
TensorBoard, pruning, recovery and paired confirmations. Both datasets contain
32 x 32 RGB images, so no search-space change is required. The shared dataset
and model APIs resolve the 100-class label conditioning automatically.

All 50,000 official training images are used for fitting. The 10,000 official
test images provide validation, early stopping, pruning, HPO feedback and
confirmations (`VALIDATION_SOURCE="test"`, `VALIDATION_RATIO=0.0`). These scores
participate in tuning and are not an untouched generalization estimate.

The defaults remain **51 workers across three H100s**, 17 per device with
3,584 MiB TensorFlow caps and coordinated admission. The CIFAR-10 capacity pass
does not establish CIFAR-100 capacity; the corresponding largest-model,
batch-128 check is pending while the container's GPUs are occupied. Existing
jobs are preserved. Recognized search OOMs are pruned through the shared API;
failed confirmations remain incomplete.

Successful-trial targets are 200, 1,000 and an optional 1,050, with 5,000 attempts,
a persistent 20-hour clock and two hours reserved for confirmations. All runs
keep a 50-epoch maximum, early-stopping patience 5, the existing percentile
pruner, and the same top-three/fresh-seed confirmation protocol. TensorBoard
uses **port 6006**; use another free port if a dashboard already occupies it.

Use fresh `files/results/dit_generation_hpo_cifar100_v1` and the separate
`Continual-Learning-with-Diffusion-Vision-Transformers-cifar100` checkout.
CIFAR-10 results cannot resume as CIFAR-100 results. The notebook is initially
unexecuted and is not replaced by the generic notebook generator. Its Colab
and Kaggle buttons target the published GitHub main revision; publish the new
notebook with matching APIs before using those links.

## DiT follow-up: missing capacities and new backbones

`DiT_Generation_HPO_Followup.ipynb` is a separate maintained study driven by the
first runner's completed results. It uses the existing training, TensorBoard,
pruning, confirmation and multi-GPU admission APIs. The default source is
`files/results/dit_generation_hpo_v10` in the original remote checkout; configure
its absolute location with `SOURCE_RESULTS_PATH`. At least one compatible,
finite, completed source trial is required. The source need not have reached its
final target, and it is never modified by transfer.

`freeze_transfer` authenticates and snapshots source results, then selects up to
eight distinct best conditioning/optimizer parameter hints. The immutable
`source_transfer.json` preserves that selection for resumes. Each hint seeds
four missing-capacity plain routes and seven new topology routes: at most
**88 initial suggestions**, in source-rank order. Unspecified capacity and
architecture parameters remain searchable. Each suggestion trains a fresh model;
source scores, checkpoints and weights are not inserted into the new study.
TPE learns from new outcomes, then continues throughout the declared search space.

| Branch | Difference from the first notebook |
| --- | --- |
| `plain_missing` | Forces width 256, depth 7-10, 6/8 heads, or explicit per-head key width equal to `dim`; other capacity axes retain their full domains. |
| `u_skip` | Restores the nine-stage, two-scale U-DiT with feature skips into stages 7 and 9. |
| `local_hybrid` | Adds native spatial mixers inside selected ViT blocks, after attention and before the feed-forward sublayer. |
| `feature_ladder` | Fuses stage 1 and the preceding stage at each target from stage 3 onward. |
| `feature_dense` | Fuses every preceding transformer-stage output at each target from stage 3 onward. |
| `cross_ladder` | Decoder blocks retain self-attention, then attend to the output two stages earlier. |
| `cross_dense` | Decoder blocks attend to multiple earlier outputs, excluding their current query stream. |
| `u_cross` | Adds cross-attention routes 7<-3 and 9<-1 to the U-DiT feature-skip backbone. |

Local mixers search depthwise, separable and expanded-pointwise convolutions,
3/5/7 kernels, and every/alternating/late block placement. Expanded channels are
projected back to the attention residual width. Feature and cross-attention
sources search additive or concatenative merging; cross attention also searches
query-side or value-side connections. Every source precedes its target, feature
routes retain the immediate predecessor, and U routes connect matching grids.

Capacity domains restore `dim=[16,32,64,128,256]`, plain/routed depths 3-10,
`mha_num_heads=[4,6,8]`, `mha_key_dim=[None,"dim"]`, and batches 32/64/128.
U backbones keep their fixed nine-stage graph. All previously requested CFG,
embedding, merger, adaptive-normalization, refiner, final-activation, MSE/MAE
training-loss and diffusion-timestep choices remain available. Every plain
follow-up candidate exceeds at least one v10 restriction; every other branch
introduces a topology excluded from v10.

The follow-up writes to `files/results/dit_generation_hpo_followup_v1` and has
its own 20-hour budget, with two hours reserved for confirmations. Targets are
100 successful trials for review, 400 for the main search and an optional 50
more, with a 2,500-attempt ceiling. It keeps the original full CIFAR-10 training
and official-test feedback protocol; selected test scores are model-selection
results, not untouched test estimates. TensorBoard uses **port 6007**.

The initial resource setting is **three workers total**, one on each selected
H100, with `WORKER_GPU_MEMORY_LIMIT_MB=73728` (72 GiB). Three simultaneous largest
selected cases completed bounded training and full test validation, peaking at
34.49/24.50/22.17 GiB. All workers wrote completion payloads, but the parent
notebook kernel exited before the coordinator recorded every outcome; the overall
check remains interrupted, not an end-to-end pass. This is not a maximum-
concurrency claim or an exhaustive capacity guarantee. Device memory is not
pooled across trials. Recognized search
OOMs are pruned through the shared API. Do not reuse the first notebook's
51-worker setting for this larger space.

On the supplied container, use the separate
`Continual-Learning-with-Diffusion-Vision-Transformers-followup` checkout and a
fresh kernel. Keep the original checkout and results intact so the first study
can resume with its matching source identity. Publish the follow-up notebook and
all matching helper/model changes together before using its Colab/Kaggle buttons.

## CIFAR-100 DiT follow-up

[DiT_Generation_HPO_CIFAR100_Followup.ipynb](DiT_Generation_HPO_CIFAR100_Followup.ipynb)
mirrors the CIFAR-10 follow-up's eight backbone families and complete larger
search space, using **CIFAR-100's 100 fine classes**. It consumes completed
results from the CIFAR-100 main runner at
`/workspace/Continual-Learning-with-Diffusion-Vision-Transformers-cifar100/files/results/dit_generation_hpo_cifar100_v1`.
At least one compatible finite COMPLETE source trial is required; the source
study need not have reached its final target. No CIFAR-10 observations or
weights are imported. The shared transfer API receives `dataset_name=DATASET`
and rejects an incompatible source dataset while retaining its CIFAR-10 default
for existing callers.

The best eight distinct source parameter hints produce up to **88 fresh initial
suggestions**, followed by TPE throughout the declared space. All 50,000 training
images are used for fitting and all 10,000 official test images supply validation
and HPO feedback. Trial targets remain **100 -> 400 -> optional 450**, with
2,500 attempts, 50 maximum epochs, the same pruning and paired confirmation
seeds, and a separate **20-hour clock** including two hours for confirmations.

The defaults are **three workers total**, one per H100, with **73,728 MiB** caps;
CIFAR-100 capacity is unmeasured. A bounded largest-model, batch-128 check must
also cover full validation and the 100-class PNG/GIF reports when the GPUs are
available. Existing jobs retain their reservations. The larger space must not
inherit the main notebook's 51-worker concurrency. TensorBoard uses **port 6007**;
choose another port if needed.

Use the separate
`Continual-Learning-with-Diffusion-Vision-Transformers-cifar100-followup` checkout
and fresh `files/results/dit_generation_hpo_cifar100_followup_v1`. Preserve both
source and CIFAR-10 studies. The frozen source manifest authenticates the
CIFAR-100 selection and is reused on resume. This maintained notebook has empty
outputs and is not replaced by the generic notebook generator. Its online
buttons require the notebook and matching APIs to be published together.

## Notebook matrix

| Task | Notebook | Model role | Representation | Default epochs |
| --- | --- | --- | --- | ---: |
| Generation | `generation/diffusion_transformer.ipynb` | Conditional DiT generator | Images | 50 |
| Generation | `generation/dit_decoder.ipynb` | Standalone conditional DiT decoder | Images | 50 |
| Generation | `generation/dit_encoder_decoder.ipynb` | Conditional DiT encoder-decoder | Images | 50 |
| Generation | `generation/unet.ipynb` | Conditional convolutional generator | Images | 50 |
| Generation | `generation/vae.ipynb` | Variational generator | Flattened images | 30 |
| Generation + classification | `joint/dit_classifier.ipynb` | Joint DiT generator/classifier | Images | 50 |
| Generation + classification | `joint/dit_encoder_decoder_classifier.ipynb` | Joint DiT encoder-decoder/classifier | Images | 50 |
| Generation + classification | `joint/unet_classifier.ipynb` | Joint U-Net generator/classifier | Images | 50 |
| Generation + classification | `joint/vae_classifier.ipynb` | Joint variational generator/classifier | Flattened images | 30 |
| Classification | `classification/cnn.ipynb` | Convolutional baseline | Images | 30 |
| Classification | `classification/dnn.ipynb` | Dense baseline | Flattened images | 30 |
| Classification | `classification/pretrained.ipynb` | Xception transfer learning | Images | 30 |
| Continual learning | `continual/cnn.ipynb` | Classifier-only sequential/cumulative/replay baseline | Images | 20 |
| Continual learning | `continual/dnn.ipynb` | Classifier-only sequential/cumulative/replay baseline | Flattened images | 20 |
| Continual learning | `continual/pretrained.ipynb` | Classifier-only sequential/cumulative/replay baseline | Images | 20 |
| Continual learning | `continual/diffusion_transformer.ipynb` | Conditional replay buffer | Images | 20 |
| Continual learning | `continual/dit_decoder.ipynb` | Conditional replay buffer | Images | 20 |
| Continual learning | `continual/dit_encoder_decoder.ipynb` | Conditional replay buffer | Images | 20 |
| Continual learning | `continual/unet.ipynb` | Conditional replay buffer | Images | 20 |
| Continual learning | `continual/vae.ipynb` | Conditional replay buffer | Feature vectors | 20 |
| Continual learning | `continual/dit_classifier.ipynb` | Joint-model replay buffer | Images | 20 |
| Continual learning | `continual/dit_encoder_decoder_classifier.ipynb` | Joint-model replay buffer | Images | 20 |
| Continual learning | `continual/unet_classifier.ipynb` | Joint-model replay buffer | Images | 20 |

The generated `continual/diffusion_classifier.ipynb`
uses `diffusion_classifier` to search the three diffusion-classifier families
in one conditional study. Family-specific
parameter names are prefixed, so Optuna can compare DiT, encoder-decoder DiT,
and U-Net candidates without incompatible conditional distributions.

Continual `vae_classifier` search is intentionally omitted. Its attached
classifier is a fixed full-class, raw-input branch independent of the VAE
latent path, so tuning its loss weight would neither define a growing
class-incremental head nor improve the replay representation. Continual VAE
studies instead use the ordinary conditional VAE with the learner's expanding
external DNN.
The old `continual/vae_classifier.ipynb` is retained with an archival notice;
it is outside the 24 supported notebooks and the generator's output matrix.

## Execution notes

1. Use a remote TensorFlow 2.20.0 / Keras 3.11.2 runtime. The DiT campaign
   runner supports Colab and Kaggle GPU sessions with the admission rules
   above. Local notebook computation is prohibited.
2. Open one notebook and edit only its setup constants as needed. The defaults
   use `CIFAR10`, 30 trials, and seed 42.
3. Inspect `SEARCH_SPACES[TASK][MODEL]` before starting a study. Spaces are
   task-specific and keep divisibility, routing, conditioning, and tensor-shape
   constraints valid.
4. Run the study cell. It calls only:

   ```python
   run_hpo(
       task=TASK, 
       model_name=MODEL, 
       dataset_name=DATASET, 
       n_trials=N_TRIALS, 
       epochs=EPOCHS, 
       results_path=RESULTS_PATH, 
       # Use model_name="diffusion_classifier" to search every diffusion
       # classifier family. Enable the previous-task teacher lifecycle with:
       use_distillation=True, 
       # Optional bounded plumbing-study controls (sealed in study_spec.json):
       max_train_samples=512, 
       max_val_samples=256, 
       n_startup_trials=2, 
       search_space_overrides={"timesteps": [500], "test_steps": [20, 50]}, 
       # Diffusion-classifier joint/continual studies only:
       use_ensemble_accuracy=False, 
       ensemble_accuracy_kwargs={"weighted": True, "max_t": 128}, 
       # Optional runtime-only teacher for the same classifier families:
       # teacher_network=teacher,
       # Optional diffusion curriculum:
       fit_method="fit_progressively", 
       fit_kwargs={
           "stage_tasks": "timesteps_only", 
           "stages_num": 4, 
           "stage_epochs": 5, 
           "final_epochs": 5
       }, 
       seed=SEED
   )
   ```

5. The final cell displays the trial table and either the best single-objective
   trial or the Pareto-optimal joint trials. Study artifacts are written below
   `files/results/hpo/<task>/<model>/<dataset>/` by default.

Each successful trial saves its resolved YAML config, final model weights,
history and evaluation CSV files, plots, available trajectory GIFs, and
TensorBoard events. The
TensorBoard event suffix lists every sampled value in alphabetical parameter
name order; the complete name-to-value mapping is also stored in the trial
config and TensorBoard text summary. Compact logs live below
`files/results/hpo/_tb/` for other generic workflows. Ordinary DiT generation
uses its study directory's `tensorboard` tree for both fresh and resumed trials.
`study.db` permits resuming a study, while `trials.csv`
gives a study-level table.

Optuna feedback comes from the post-training validation evaluation of the same
saved/restored model state, not from a historical best or the last row of the
pre-restoration Keras history. Diffusion objectives use the EMA branch;
ordinary classifiers and VAEs use `valset_eval`. The semantic
defaults are fixed reconstruction MSE for VAEs, unweighted `noise_loss` for
diffusion, a generation/accuracy Pareto pair for joint
models, validation accuracy for standalone classifiers, and validation
`final_average_accuracy` for continual studies. `objective_metrics` can name
other scalar validation metrics with matching or inferred directions. Feedback uses
the selected validation data: a training holdout by default, or official test data
when `validation_source="test"` is explicit. The DiT campaign notebook now selects
official test data; report-only test metrics are not substituted for `valset` scores.
Final objectives are scored only after a completed fit. The optional DiT percentile-pruning policy uses intermediate validation EMA
noise loss to stop poor trials before final scoring; disabled pruning preserves
post-fit-only feedback. VAE generation early stopping
uses `val_mean_squared_error`. Reconstruction and denoising metrics are proxies
for generation quality; they do not establish sample diversity or replay utility.

Reverse-process `test_steps`, CFG scale, and eta are sampled only in continual
diffusion studies, where generated examples enter later replay tasks and can
change the objective. Generation and joint studies optimize validation losses,
so these otherwise inert dimensions are fixed to at most 50 steps, CFG 4, and
eta 0. Final visualization settings remain reporting choices rather than HPO
parameters.

`swap_noise_image=True` is an immutable wrapper override for direct x_t/VAE
prediction. These studies fix `image_loss_coef=0`, tune `kl_loss_coef` unless
the override supplies one, and train with `x_t` prediction plus weighted
main-latent KL. Their default validation objective is unweighted reconstruction
MSE, so changing the sampled KL coefficient cannot directly improve a score.
HPO validates the variational structure and ratio-list
cardinality but passes a fixed KL coefficient through unchanged. These studies
do not request denoising GIFs. DiT, DiT encoder-decoder, and DiT
encoder-decoder-classifier studies require an immutable main-network topology;
a single-level example is
`model_overrides={"vit_block_ids": [1, 4], "use_decoder_ids": [4],
"reshaper_ids_dict": {2: "flatten", 3: "unflatten"}, "reshaper_kwargs":
{"add_kl": True}}`; U-Net and U-Net classifier studies require
`model_overrides={"reshaper_kwargs": {"add_kl": True}}`. Standalone
`dit_classifier` and `dit_decoder` x_t studies are rejected because their raw
call contracts cannot resume main-latent decoding. Noise distillation is also
incompatible with x_t prediction.

An x_t DiT depth is fixed by `model_overrides["depth"]`, when supplied, or
derived from the greatest positive stage ID in the fixed topology. An explicit
depth must cover every referenced stage, and neither form creates a sampled
2--6 depth parameter. Every topology contains actual encoder computation
(`vit`, local mixer, or downsampling) before its bridge and decoder computation
(`vit`, local mixer, or upsampling) after it. Transformer/local-mixer stages
stay out of the bridge, all downsampling precedes it, and all upsampling and
decoder blocks follow it. Multiple adjacent pairs must additionally form one
uninterrupted central bridge. This is the minimal Copy35-style routing pattern
(depth derives to 15):

```python
model_overrides = {
    "vit_block_ids": [1, 3, 5, 13, 15], 
    "use_decoder_ids": [13, 15], 
    "connection_ids_dict": {8: [3], 10: [1], 12: [7]}, 
    "cross_attention_ids_dict": {13: [9], 15: [11]}, 
    "downsample_ids": [2, 4], 
    "reshaper_ids_dict": {
        6: "flatten", 7: "unflatten", 
        8: "flatten", 9: "unflatten", 
        10: "flatten", 11: "unflatten"
    }, 
    "reshaper_kwargs": {
        "add_kl": True, 
        "latent_dim_ratio": [1 / 32, 1 / 128, 1 / 256]
    }, 
    "upsample_ids": [12, 14]
}
```

DiT classifier studies focus `classifier_architecture` on `linear`,
`connection`, and `u_shape`. On grids divisible by four, `u_vae` and
`u_multilevel_vae` are also available. The variational templates follow the
network order demonstrated in `DiT mini copy 35.ipynb`: all encoder blocks and
downsampling operations precede a central variational bridge, every
flatten/unflatten pair occurs inside that bridge, and all decoder blocks and
upsampling operations follow it. This makes the encoder/latent/decoder
boundary and the occurrence order used by the ratio list unambiguous.
All-depth feature aggregation always projects the concatenated main features
to the configured classifier width before entering any of these templates.
The multilevel template also retains the mandatory terminal `-1: [-1]`
classifier connection used to feed its final depth to the classification head.

`reshaper_kwargs["latent_dim_ratio"]` is always a list in flatten occurrence
order, with exactly one member per flatten/unflatten pair. HPO samples an
independent absolute latent width of 16, 32, or 64 for each occurrence and
divides it by that occurrence's flattened width. It does not repeat one ratio:
doing so made the later, larger feature maps produce latent widths of 128 and
256 in the incomplete multi-level notebook run and caused an extreme KL and
parameter increase. These models are variational bottlenecks, not
conditional-prior hierarchical VAEs.

The zero-inclusive `ctr_loss_coef` search applies to joint DiT and U-Net
classifiers. A positive value adds a regularizer at the final classifier depth,
or at the central latent depth for a U-VAE classifier.
It uses normal labels without a teacher; teacher-backed studies can select
`normal`, `distil`, or `both`, with hard or soft teacher targets for the latter
two modes. Accuracy coefficients are balanced across the active classifier,
distillation, and regularizer predictions through `clf_acc_coef`,
`clf_distil_acc_coef`, and `ctr_acc_coef`.

With `task_size=1`, the schedule starts from one class and then adds one class
per task; `task_size=2` starts from two and adds two, including a shorter final
task when necessary. A one-output softmax has trivial 100% accuracy and zero
classification gradient, so singleton classifier searches exclude a new-only
sequential protocol unless positive replay exists. The first acquisition row
is reported as unavailable. Consequently, forgetting and backward-transfer
cannot be HPO objectives for a singleton first task; final validation average
accuracy remains well defined.

Dataset, split, class/task schedule, seed, epoch/trial budget, and objective are
sealed experimental controls rather than hyperparameters: changing them across
trials would make validation scores incomparable. The umbrella continual
diffusion-classifier space instead searches the model family, architecture,
optimizer, diffusion/noise schedule, teacher snapshot branches,
V1/V2 wrapper behavior, hard/soft distillation temperature and scope, optional
noise distillation, and continual/replay policy. `search_space_overrides` may
bound expensive dimensions for a plumbing study, but it becomes part of the
immutable study identity and cannot be changed on resume.

Umbrella categorical overrides are checked against each active conditional
domain, and sampling steps must not exceed training timesteps. Raw
`model_overrides`, `wrapper_overrides`, and a live external teacher require an
exact model family because their tensor topology and diffusion process must
match. The umbrella instead supports teacher-free continual distillation from
the selected raw/EMA snapshot of the preceding task.

V1 classifier trials compare the ordinary conditional classifier prediction
with the notebook's unconditional branch at CFG scale 1. Timestep masking is
limited to 50%, 70%, or 90%; null, timestep, combined, and unmasked recipes
remain available.

For V2 DiT classifiers, `clf_vars_recipe` selects one of three coupled,
notebook-supported variable assignments: `separate` maps to embedding/noise
IDs `([], [])`, `conditions` maps to `([1, 2], [])`, and `notebook` maps to
`([0, 1, 2, 3], [1])`. Coupling the assignments avoids unsupported Cartesian
combinations. Classifier input noising is limited to clean inputs or caps of
64 and 256 timesteps when below the selected diffusion horizon. V2 performs a
generator fit and then a separate classifier fit, each using `epochs`; compare
V1 and V2 only with that difference in compute budget made explicit.

## Notebook-informed search-space limits

Search-space version 9 uses the stored notebook outputs as practical anchors,
not as definitive benchmarks. Most comparisons are single stochastic runs,
some notebooks are incomplete, and the selected legacy CIFAR VAE objective is
circular; consequently, failed or weak runs are used to exclude implausible
regions rather than to assert a precise optimum.

For DiT, the default space emphasizes four-head capacities, MLP ratios 2 and
4, MSE, adaptive normalization, global 2D sinusoidal positions, 500 or 1,000
timesteps, and Adam/AdamW. Ordinary generation/joint fitting uses cosine
decay; continual and progressive fits retain a constant rate because their
complete update count is not known up front. The plain backbone is joined by
the compact symmetric two-level feature-skip U-DiT on compatible grids.
Resampling positions are retained. Continual sampling concentrates on 20, 50,
or 100 reverse steps, CFG 2.5Ã¢â‚¬â€œ5, and eta 0 or 1.

The CNN space includes the saved CIFAR stage shape
`(64, 128, 128, 256)` with depths `(1, 2, 2, 1)`, dropout 0.15 and 0.20, max
intermediate pooling, and global-average pooling. Transfer learning includes
the 12-layer Xception tail used by both CIFAR notebooks. Dense classifier
templates include a linear head and the saved two-layer widths, but the legacy
DNN results used frozen Xception features and must not be interpreted as
evidence for the same learning rates on raw flattened pixels.

Reservoir replay independently samples capacity (2,500/5,000/10,000), replay
sample count (500/1,000/2,500), and insertion count (500/1,000). Capacity is a
storage limit; tying it to both per-update counts excluded the saved
10,000-capacity/1,000-sample setup and needlessly coupled memory with compute.

For joint or continual `dit_classifier`,
`dit_encoder_decoder_classifier`, and `unet_classifier` studies, set
`use_ensemble_accuracy=True` to use validation/task ensemble accuracy as the
Optuna feedback signal. Ordinary accuracy is still reported. These trials use
an `ensemble_accuracy` subdirectory so they cannot mix with an existing normal
accuracy study.

Set `fit_method="fit_progressively"` only for diffusion model families and
provide `stage_tasks` in `fit_kwargs`. The mapping accepts the existing
`DiffusionModel.fit_progressively` controls: stage count and verbosity, stage
and final epoch budgets, timestep boundaries and clustering, resolutions,
depths, pacing and early-stopping settings. Remaining Keras keys such as
`steps_per_epoch` and `validation_steps` are forwarded unchanged. Progressive
`DiffusionClassifierV2` trials apply the curriculum to the generator and then
run their existing ordinary discriminator phase. Values must be YAML-safe
because every trial config is serialized and reloaded before training.

Progressive studies use a `fit_progressively` subdirectory and a separate
SQLite study name, so resuming them cannot mix their trials with ordinary-fit
studies. As with other HPO settings such as epoch count, use a different
`results_path` when comparing different progressive curricula.

Passing `teacher_network` enables conditional hard/soft distillation sampling
for joint or continual `dit_classifier`, `dit_encoder_decoder_classifier`, and
`unet_classifier` studies. The student token/head and wrapper loss settings are
written to each trial config, while the live teacher is passed directly to
`main` and never serialized. Distillation trials use their own `distillation`
study directory/name and prefer `total_accuracy`; they can also use
`fit_progressively`.

With ordinary `fit`, `epochs` is the per-fit budget. With progressive fitting,
the diffusion budget is the epochs actually run across all curriculum stages:
at most `stage_epochs * number_of_stages + final_epochs` under fixed pacing.
The `epochs` argument remains the budget for ordinary classifier phases in a
continual study and must therefore still be positive. It also retains the
existing role of sizing a sampled cosine learning-rate schedule; set the HPO
epoch value to a progressive budget representative of the curriculum when
cosine decay is enabled.

For fair comparisons, keep the dataset, seed, trial count, epoch budget, and
continual replay-budget candidate set fixed across competing model families. Diffusion and
joint studies are substantially more expensive than CNN, DNN, or VAE studies;
run a small smoke study before committing to all 30 trials. Recognized TensorFlow
`ResourceExhaustedError` or Python `MemoryError` during a trial is recorded as
Optuna **PRUNED** for resource exhaustion, and the study continues within its
budget. This resource handling is independent of performance-pruner settings.
Other programming/configuration errors and unexplained worker crashes still
surface instead of being relabeled as OOM.

The notebooks can be regenerated after intentional template changes with
`python files/notebooks/hpo/generate_notebooks.py`.

## DiT classifier campaign runner

[DiT_Classifier_HPO_Runner.ipynb](DiT_Classifier_HPO_Runner.ipynb) uses version2 of
`dit_classifier_runner` and the same orchestration APIs as generation. It uses
**both raw classification_accuracy (maximize) and raw noise_loss (minimize)** as
separate Optuna objectives. Both values drive the sampler and Pareto front; there
is no weighted sum or accuracy-only HPO ranking. The older
`joint_dit_classifier` Pareto profile keeps its existing recipe.

The user selected official test feedback. All CIFAR training rows are used for
fitting; official test rows supply epoch validation, both HPO objectives and
confirmations. These are tuning results, not an untouched generalization estimate.

The archive-informed space retains widths64/128/256, generator depths4/6/8,
classifier depths2/4/6/8, batch32/64/128, readout/noise/loss/regularization choices
and native feature/local/cross-attention routes. The complete23-distribution table
is in the notebook and [profile source](../../../common/dit_classifier_hpo.py).
Copies37–42 and the later classifier archives motivated the capacity, corruption
and CE axes; Copies48/49/64/66/71/75/76/80/90/95/97 motivated feature routing,
readout, conditioning and regularization. Saved adaptive comparisons and KD runs
are not evidence that any teacher-free setting is universally best. The baseline
hint contains parameters only and is retrained; no saved outcome is imported.

Completed trials compare **final raw weights after the same100-epoch budget**.
Accuracy-only early stopping and best-accuracy restoration are disabled. Per-epoch
validation and TensorBoard remain active. `PRUNING=None`: the existing Optuna
percentile report/should_prune API is scalar and is not used for this Pareto study.
Recognized OOMs, nonfinite training losses and a nonfinite value in either final
objective still prune only the affected search trial.

Finalists are distinct finite nondominated configurations. TOP_K=3 is a maximum:
smaller fronts use all candidates; larger fronts use evenly spaced entries in
accuracy order, retaining the highest-accuracy and lowest-noise extremes when
TOP_K>=2. TOP_K=1 explicitly chooses the highest-accuracy Pareto candidate.
This determines confirmation allocation, not HPO fitness. Each candidate receives
paired fresh seeds101/202/303. Receipts authenticate both raw objectives and
directions; summaries show mean/std for accuracy and noise loss. Partial seed
sets cannot support a final comparison across all candidates.

Initial defaults remain3 total workers on3H100s with72GiB per-worker caps,
100/400/450 completion targets,2500 attempts and a20-hour clock with2hours reserved
for confirmations. Counts and completion of every confirmation are not guaranteed.
These are **unmeasured classifier settings**, without the generation runner's
51-worker capacity evidence. Smaller hosted GPUs need a lower/automatic memory
cap. TensorBoard uses port6008. Shared APIs retain GPU admission, isolated workers,
persistent clocks, YAML/SQLite/sampler recovery and paired confirmations.

Use fresh `files/results/dit_classifier_hpo_v2_pareto`. Scalar version1 studies
and receipts cannot be reinterpreted as the new objective pair. Existing active
generation/recovery/follow-up source checkouts must remain unchanged. This
maintained notebook is not overwritten by the generic notebook generator.

**Validation status:** source review only. Focused profile, HPO dispatch and
runner/confirmation regression tests are authored but unexecuted. Other supplied
containers were unavailable; the occupied three-H100 container was not contacted
or altered. No local Python or model computation was performed.

On an available authorized online runtime, first verify host/GPU identity, live
ownership, memory, checkout and TensorFlow2.20.0/Keras3.11.2. In a separate process
and isolated checkout, start with remote CPU checks (`CUDA_VISIBLE_DEVICES=-1`):

```bash
python -m unittest common.tests.test_dit_classifier_profile common.tests.test_dit_classifier_hpo_dispatch common.tests.test_dit_classifier_runner
python -m unittest common.tests.test_dit_hpo_runner common.tests.test_dit_hpo_confirmation common.tests.test_joint_hpo_profile common.tests.test_source_contracts
```

Then use coordinated admission for a bounded largest-model, batch128, full
feedback/report/TensorBoard and paired-confirmation GPU check before increasing
concurrency. GPU discovery or CPU tests alone do not establish H100 capacity.
