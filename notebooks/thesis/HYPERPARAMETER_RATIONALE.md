# Registered Route One recipe — TensorFlow 2.20

## Registered recipe

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
null conditioning.

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
[`frozen_design.json`](../../files/results/thesis_route_one/minimum_v6_tf220_21streams/frozen_design.json)
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
The [saved freeze validation](../../files/results/thesis_route_one/minimum_v6_tf220_21streams/freeze_validation.json)
records bounded software checks, not completed thesis experiments or convergence.

The [recipe source ledger](recipe_sources.json) records the inspected primary
sources. The [scientific basis](../../semantic_consolidation/SCIENTIFIC_BASIS.md)
describes the current semantic objectives and their differences from the
source methods.
