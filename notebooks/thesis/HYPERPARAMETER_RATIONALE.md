# Registered Route One recipe — TensorFlow 2.20

## User-selected revision, 19 September 2026

The maintained CIFAR-10 and CIFAR-100 YAMLs now register the user's requested
recipe for a **test-informed fixed benchmark**. Earlier official-test HPO
results informed recipe discussion; new seeds and a freeze cannot make those
test rows an independent confirmation set. These are explicit experimental
choices, not an HPO optimum or a claim of reproducing JDCL/TMCL. `dim=True` was
clarified to integer width **128**.

| Component | Registered value |
|---|---|
| DiT architecture | Width 128, backbone depth 6, classifier depth 6, CNN patchification; existing patch 4, four heads and FFN ratio 2 retained |
| Classifier training | Noisy images, null-only conditioning, `mask_by_nulls=false`; batch fraction 0 retains all rows for both objectives |
| Joint optimization | Adam, initial LR 0.001, zero weight decay, one whole-stream cosine, batch 128, 50 epochs/task, no early stopping |
| Joint cosine duration | 49,950 joint-optimizer updates for CIFAR-10; 116,150 for CIFAR-100; same declared horizon across each dataset's paired conditions |
| Primary accuracy | Raw primary-head timestep ensemble: `max_t=256`, `t_range_drop_rate=0.75`, `clf_acc_coef=1.0`, `clf_distil_acc_coef=0.0`; uniform averaging over retained timesteps (`weighted=false`) |
| Replay | `match_current`: every old class receives the same number of generated rows as each permitted current real class; no old raw training rows |
| Replay sampling | 1,000 reverse steps, eta 1, CFG scale 3 |
| Distillation | Soft classifier KD, temperature 2, replay-only classifier scope; classifier/noise KD loss coefficients 0.1. Noise KD selects teacher-supported conditioning IDs, including null conditioning |
| Semantic phases | Acquisition/consolidation: 200/400 updates per CIFAR-10 task, 1000/2000 per CIFAR-100 task; Adam 0.0003, batch 32, TMCL four-view augmentation, InfoNCE temperature 0.2, alignment 0.1, CE/orthogonality 1 |

The initial joint LR was reduced from 0.005 to 0.001 as an explicitly approved,
conservative optimization choice. It has not been validated as optimal for this
full recipe; source-model rates and historical HPO results use different models
or training settings. This LR choice does not change the main campaign's
registered cosine durations or KD loss coefficients.

With the existing stratified 20% validation split, current classes provide
4,000 real examples/class on CIFAR-10 and 400/class on CIFAR-100. The generated
old pool grows to 32,000/36,000 rows at the final task; every seen class has the
same training count. Unequal current class counts are rejected rather than
silently truncating real data or claiming balance. No generative replay is
introduced in the offline/naive supplemental references.

Here, "no old raw training rows" means that later-task training receives current
real images and generated old-class replay. The simulator can still retain old
arrays in host memory, and held-out historical validation rows support diagnostics;
neither is an old training-image rehearsal pool. With `mask_by_nulls=false` and
`clf_train_batch_fraction=0`, the current classifier loss uses all joint rows with
null conditioning. The historical half-row CE description below does not apply
to this recipe.

Retained partial batches give CIFAR-10 task batch counts
`[63,125,188,250,313]` and CIFAR-100
`[32,63,94,125,157,188,219,250,282,313]`. Fifty epochs give 46,950/86,150
ordinary joint updates. The global cosine horizons add the maximum fixed
extra-joint allowance: 3,000/30,000. Semantic phase optimizers use separate
clocks; extra-joint advances the joint clock, including the learning rates in
later tasks. This control matches additional optimizer applications, not
task-relative learning rates, images, FLOPs or wall time. Changing the split,
batch, task schedule, epochs or phase allowance requires reviewing these
explicit horizons. Arbitrary time-matched or extension budgets are not covered.

Only extra-joint uses the full declared joint-optimizer horizon. Baseline and
learned runs perform 46,950/86,150 joint updates; learned semantic updates advance
their own optimizers and do not advance the joint cosine. Thus the shared horizon
does not imply that every condition ends at zero joint learning rate.

### Reference cosine durations

Notebooks 11 and 12 resolve their own cosine durations in `prepare_reference`,
after the native training/validation split and sample caps, before model and
optimizer creation. They do not use the replay campaign's 49,950/116,150 horizons.
At 50 epochs, batch size 128, a 20% validation split and no sample cap:

| Reference | CIFAR-10 joint updates | CIFAR-100 joint updates |
|---|---:|---:|
| Offline joint, notebook 11 | 15,650 | 15,650 |
| Naive sequential, notebook 12 | 15,750 | 16,000 |

Offline duration is `epochs * ceil(total_training_rows / batch_size)`. Naive
duration sums each task's partial-batch-inclusive epoch length before multiplying
by epochs. Both start at LR 0.001 and reach the cosine endpoint after their planned
updates. The saved `reference_plan.json` records the resolved row counts, update
budget and optimizer settings. Conflicting epoch/step overrides are rejected.
These references remain supplemental: their notebook defaults are seed 17 and
validation evaluation, and they are not included in the frozen 21-run plan.

### Prediction and interpretation

`max_t` is an exclusive upper timestep bound, not a count of independent votes.
The existing ensemble drops 75% of the 256 candidate timesteps using its
seeded inverse-SNR timestep-removal policy, retaining 64 timesteps. Predictions
at the retained timesteps are averaged uniformly (`weighted=false`); removal
probabilities and prediction weights are separate operations. Only the primary
classifier head contributes to inference; ordinary
clean metrics remain separate diagnostics. The distillation head remains in
training with its existing KD loss. It has no prior teacher at the first task,
and KD is disabled throughout the supplemental offline/naive references, so
giving it zero inference weight keeps the primary endpoint on a supervised head
in every task and reference. Fixed recipes do not prove that timestep averaging
or the replay images are useful.

The inference coefficients above are distinct from `clf_distil_loss_coef=0.1`
and `noise_distil_loss_coef=0.1`. Giving the distillation head zero inference
weight does not disable its training loss when a prior teacher exists. Serialized
typed defaults for other model options do not override the explicit thesis
`model.wrapper_kwargs` or `continually_learn.ensemble_accuracy_kwargs`.

The current campaign is `minimum_v6_tf220_21streams`. Its prepared
[`frozen_design.json`](../../results/thesis_route_one/minimum_v6_tf220_21streams/frozen_design.json)
binds two native manifests and 21 run configurations with seeds 1103, 2207 and
3301. The user selected notebooks **03–09 only** before starting these runs:
CIFAR-10 extra-joint and learned (six streams), and all five CIFAR-100 conditions
(15 streams). Notebook 02 and its code remain available, but its three CIFAR-10
platform streams are outside this plan. This scope retains the primary paired
learned-minus-extra-joint comparison on both datasets; it does not provide a
CIFAR-10 platform comparison. Each training notebook selects one unfinished seed
per launch, so complete three launches per notebook using fresh kernels.

The manually prepared record uses the same preparation API as notebook 01;
running notebook 01 is unnecessary. Preparation is separate from training
completion: optional final collection requires all 21 verified streams. The
earlier `minimum_v5_tf220` 24-stream design and its source archives are preserved
as historical records, not silently relabelled as a completed reduced study.
Reproducing that older design requires its matching archived source. Preserve prior campaigns,
results and HPO selection provenance. Development can assess learning and cost;
software checks alone do not establish either. The native `benchmark` phase
records `test_informed=true` and `independent_confirmation=false` in the bound
selection provenance; the strict `confirmation` phase remains separate. Future
benchmark test outcomes must not alter this recipe, the seeds,
stopping rules or the set of reported streams.

Implementation references: [CIFAR-10 recipe](configs/cifar10.yaml),
[CIFAR-100 recipe](configs/cifar100.yaml),
[reference budget resolution](reference_benchmarks.py), and
[timestep selection and averaging](../../diffusion/metrics/ensemble_accuracy.py).
The [saved freeze validation](../../results/thesis_route_one/minimum_v6_tf220_21streams/freeze_validation.json)
records bounded software checks, not completed thesis experiments or convergence.

## Previous reduced recipe and source comparison (historical)

All subsections below record the earlier starting recipe and source audit. Its architecture, optimizer,
batch, replay budget, sampler and primary clean endpoint are superseded above;
unchanged semantic settings and the distinctions from official sources remain
relevant.

Runtime and primary sources were reviewed 15 September 2026 for TensorFlow 2.20 / Keras 3. These were local starting values, not demonstrated convergence or an exact reproduction. The earlier plan called its three paired seeds `[1103, 2207, 3301]` confirmation seeds; the current plan retains those seeds but is explicitly a test-informed benchmark. Development uses 17.

### Inspected primary sources

- **J:** [JDCL arXiv v3, 6 October 2025](https://arxiv.org/html/2411.08224v3), methods and Appendix B; [official repository](https://github.com/pskiers/Joint-Diffusion-in-Latent-Space/tree/ffa5303ef156c69505ff358982529d410621b5a3), commit `ffa5303ef156c69505ff358982529d410621b5a3` (21 November 2024).
- **T:** [TMCL arXiv v3, 26 January 2026](https://arxiv.org/html/2505.14125v3), methods and Appendices A/E; [official repository](https://github.com/Dendritic-Learning-Group/tmcl/tree/2cb306cc5cbe6d13a3dd4991998ee3e7c2948c8b), commit `2cb306cc5cbe6d13a3dd4991998ee3e7c2948c8b`.
- [recipe_sources.json](recipe_sources.json) records exact inspected files, URLs and SHA-256 values. The source scope is these two methods and their official implementations/dependencies, not a systematic search for optimal CIFAR settings. The supplied DOCX guide was directly extracted; its hash is recorded there. Its illustrative budgets are not validated defaults.

Code locators below are relative to the pinned repositories. **J10/J100** mean `configs/standard_diffusion/continual_learning/joint_diffusion_pooling/cifar10/` and `.../cifar100/10_tasks/`; **T-launch** means `slurm/submit_cifar100_s5.py`. YAMLs are complete J configs, without hidden YAML inheritance; Python subclass/default inheritance was inspected.

### Paper, executable source and earlier adaptation

| Setting | Paper value | Official code value / location | Chosen local value, units and reason |
|---|---|---|---|
| Architecture | J: U-Net joint model; T: ConViT, 384 dimensions, six blocks | J10/J100: channels 128, residual blocks 3, multipliers `[1,2,2,2]`; T-launch: `convit_small_dytox` | Supported deterministic DiTClassifier/V1, dim **128**, depth **4**, patch **4**, 4 attention heads, MLP ratio 2, classifier depth 1; float32, zero dropout, raw network, no EMA. A reduced local capacity choice; model summary supplies actual parameter counts. No architectural transplant. |
| Joint optimizer/rate/decay | J: AdamW, 0.0002; T: AdamW, feedforward 0.001, decay0.0001 | J `JointDiffusionNoisyClassifier.configure_optimizers`: AdamW(lr), implicit decay0.01 in declared PyTorch2.4.1; T-launch overrides generic defaults with 0.001/0.0001 | Native Keras 3 **Adam**, LR **0.0002**, explicit decay **0**, constant schedule on both datasets. A conservative local no-decay choice retains the source-informed rate without importing an unvalidated regularization strength. AdamW also passes current wrapper-growth checks; switching the scientific recipe would require development evidence. Keras defaults beta=(0.9,0.999), epsilon1e-7 differ from PyTorch epsilon1e-8. |
| Joint batch/duration | J: batch256; CIFAR10 initial50k/subsequent30k; CIFAR100 initial100k/subsequent70k updates | `cifar10.sh`, `cifar100_10t.sh`: local/global calls; J10 local50k/global30k; J100 local YAML**120k**/global70k. Launcher checkpoint names include100k/60k and do not override YAML `max_steps` | Batch **64**; CIFAR10 **40 epochs/task**, CIFAR100 **60 epochs/task**; complete finite task datasets, no step cap. Approx. 30,120/55,080 ordinary updates per stream, much less than J. No local/global reproduction or classification warmup. |
| Current/replay data | J: replay1000/class CIFAR10,400/class CIFAR100 | J10/J100 KD YAMLs and `train_joint_diffusion_cl.py`: generated local and old model data | All permitted current training rows (`null`), validation20% excluded: **8000/4000 current rows per task**; fixed **2048 old generated rows** at later tasks. No old raw training gradients. Candidate multiplier1, uniform selection. Diversity improves over repeating1024 originals; replay remains bounded as classes grow. |
| Diffusion/sampling | J: horizon1000, linear beta, DDIM250 steps | J YAML `sqrt_linear`, beta0.001–0.02; dependency `make_beta_schedule` implements this name as linearly spaced betas; replay DDIM250 | Horizon **1000**, existing **clipped_cosine**, **50** reverse steps, eta **0**, CFG scale **3**, null probability **0.5**. Reduces replay cost; replay fidelity must be measured. No claim of matching J sampling. |
| Joint losses | J: classification/KD scales0.001; joint diffusion beta0.01 | J local CE0.001, KD-stage CE**0.00001**, classification-KD0.001, diffusion-KD1, denoising0.01. `JointDiffusionKnowledgeDistillation.p_losses` combines old and new teachers | CE **1**, noise MSE **1**, classifier KD **0.1**, denoising KD **0.1**. Local selected-row reductions and teacher flow differ, so paper coefficients are not transplanted. Soft classifier KD **T=2**, replay-only; all-seen student support preserved. |
| Semantic phase duration | T: acquisition100/consolidation200 epochs/session, pretraining250 epochs | T-launch actually sets acquisition**50**/consolidation200, pretraining `(250,200)`; `data_setups.py` and `main_tmcl.py` interpret these as epochs. Generic Config defaults `(100,100)` are overridden | CIFAR10 **200 acquisition +400 consolidation updates/task**; CIFAR100 **1000+2000**. Exactly100 acquisition visits per new gate; final retained-gate consolidation minimum40/20 visits. These are optimizer applications, not T epochs. |
| Semantic optimizer and modulation | T: modulationLR0.01/feedforwardLR0.001, cosine with10 warmup epochs | T-launch matches those rates; `main_tmcl.py` scales by batch/256 and implements AdamW/schedules. Generic defaults0.015/0.0015 differ. Layerwise modulation decay0.4→0.04 | Existing shared phase **Adam LR0.0003**, constant, **zero weight decay** in both phases; local API exposes one shared rate. Raw gate initialization SD**0.02**, gain/bias tanh limits**1**, orthogonality weight**1**. No unimplemented phase schedules or layerwise decay. |
| Alignment | T: multi-view Barlow Twins, lambda0.005, scale0.1 | T-launch/main loss: four views, lambda0.005, **scale0.024**; sums correlation penalties. SupCon's0.1 temperature is a different comparator | Existing asymmetric instance **InfoNCE T0.2**, weight**0.1**, CE weight**1**, phase batch**32**, **four independently augmented image views**, no diffusion noise `[0]`, uniform reliability. Average InfoNCE over three frozen-target views; CE uses a separate clean input. Approximate collapsed alignment contribution0.1×log32=0.347 versus CE log(number of classes). This scale argument is not empirical validation. |
| Diagnostics | T studies CDNV | `main_tmcl.py` calls `represent/neural_collapse.py:class_distance_normalized_variance` on class-split features | **8 fixed validation examples/class**, observer batch 32, replay audit4/class; phase probe2 batches, max4 gates. Existing cosine geometry, rank, hidden drift and CKA remain their own measurements, **not CDNV**. Eight is a local cost choice. |

The paper/code disagreements above remain unresolved historical differences. The complete launcher-linked YAML inventory adds two exceptions: **CIFAR10 global task3 uses LR0.00002**, and **CIFAR100 global task2 uses104,000 updates** (later global tasks use70,000). The local constant0.0002 rate and reduced epochs deliberately do not emulate either exception. Neither filename checkpoint steps nor generic config defaults define the effective launcher budget. JDCL's dependency requirements point to moving `latent-diffusion@main`; we separately inspected dependency commit `e544c57f8537e60515fc938d1c337774aa934717`. This verifies its current implementation, **not** the dependency state used for published experiments. No official run logs were retrieved to reconcile those differences.

TensorFlow 2.20 uses the supported wrapper lifecycle to reconstruct the fixed-depth network at class boundaries. Compatible optimizer slots and old-class parameters transfer; resized slots start from their default initialization. Raw post-build `add_class`/`add_depths` calls are not the supported route. Model summaries include tracked random-state values as well as floating parameters, so they must not be interpreted as process-memory measurements. See the [maintained growth contract](../../diffusion/README.md) and the bounded validation record linked above; neither establishes convergence.

### Historical objective and exposure accounting

The local factory uses mean MSE for denoising. In the earlier masked recipe, class CE averaged CFG-null selected rows; with `p_uncond=0.5`, about half the joint rows contributed, rather than receiving an extra factor0.5 in the selected-row mean. Classifier KD computes T²-scaled KL with the previous teacher distribution, zero-padded for new classes, against the student's full seen-class vocabulary. Its mask intersects explicit replay provenance with the classifier mask. Denoising KD instead selects teacher-supported **conditioning IDs**, including null conditions: it must not be described as replay-only. It uses a selected-exposure-normalized MSE. Both KD losses are absent without a prior teacher. See local `diffusion_classifier.py:compute_clf_distil_loss`, `diffusion_model.py:compute_distil_noise_loss`, and `common/model.py`.

Acquisition averages positive-pair attraction plus squared positive-to-rest cosine; official TMCL's default uses absolute cross-cosine. Consolidation averages row-wise instance InfoNCE and then views, with same-class neighbors possible negatives. It updates only the classifier projection/head and temporary predictor; the backbone, gate bank and independent acquired target remain fixed. These objective/mask differences rule out interpreting identical coefficients as identical optimization. See `semantic_consolidation/objectives.py`, `phases.py`, `controller.py`.

The local gates act on semantic features and use bounded gain/bias transforms. Official TMCL applies unbounded affine modulation to attention and MLP operations (`tmcl/nn/modulations/build.py`, `task_modulations.py`). Its stopped-gradient views share the evolving backbone (`main_tmcl.py`); this route uses a separate frozen acquired target. The experiments therefore evaluate a TMCL-inspired semantic extension of the project's joint diffusion platform. The endpoint is supervised class-incremental accuracy using all permitted current labels. Their interpretation must retain these architectural and objective differences.

For the earlier batch-64, fixed-2048-replay recipe, retained partial batches gave expected joint counts of CIFAR10 5000 then6280/task and CIFAR100 3780 then5700/task. Its semantic/extra-joint allowance was **600/3000 updates per task**, giving totals of **33,120/85,080 updates per stream** for those methods. Semantic batches had32 rows, ordinary batches up to64: equal updates were not equal images, FLOPs or wall time. The phase sampler balanced16 focus-positive and16 other-class examples, unlike the earlier joint pool proportions (20.38%/33.86% replay). At CIFAR100 task10, 2048 replay rows distributed roughly22–23 per old class. These exposure counts are historical; the current batch-128, match-current counts appear above.

Notebook00 records actual exposure, phase coverage, held-out before/after classifier and predictor-free observations, replay self-consistency, each task's runtime, sampled process RSS and allocator measurements. Inspect the full ten-task CIFAR100 stream before freezing. Tensor inventories are storage accounting, sampled RSS can miss peaks, and allocator peaks are not total device occupancy. Sum only disjoint `seconds.task_total`; notebook elapsed time may include pauses. Do not infer useful learning, strong preservation, accuracy improvement or thesis sufficiency from software smoke tests.

### Historical recovery and reporting policy

The earlier recipes enabled native recovery every 200 completed updates and at phase boundaries; this recovery interval is retained in the current recipes. The latest two intermediate snapshots and immutable completed-task boundaries are retained. The interval is identical across paired methods; measure checkpoint time and storage separately. Neither the historical 40/60-epoch budgets nor the current 50-epoch budgets and 21-stream design establish a measured wall-time promise. Default reporting presents final/incremental accuracy, signed forgetting, local backward transfer, optimizer work and complete task time, with the native primary paired interval; detailed diagnostics are optional.
