# TMCL representation learning with the native DiT

`TMCLTrainer` is the opt-in representation-learning path in this package. It
reuses the project's DiT model, TensorFlow training loop, image augmentations,
gradient application and task-boundary recovery. Its phases implement view
invariance (VI), class-modulation acquisition, modulation invariance (MI) and
optional previous-model state invariance (SI).
They do not run joint classifier cross-entropy, classifier distillation or
diffusion training.

This is a **TMCL-derived DiT adaptation**. CIFAR-10 experiments, the native DiT
and its declared block-feature endpoints differ from the paper's continual-learning
benchmark and ConViT architecture. An implemented loss or passing correctness
test does not establish benchmark reproduction or improved retention. The
[experiment protocol](../experiments/paper_replication_diagnostics/TMCL_PROTOCOL.md)
defines the proposed paired experiments, evaluation and compute gates.

## Relationship to the existing semantic route

The original `RouteController`, `RouteSettings`, YAML runner and HPO notebooks
remain the joint-diffusion semantic route described in [README.md](README.md).
Its `image_augmentation: tmcl` setting selects image transforms only. It does
**not** select `TMCLTrainer`, multiview Barlow VI/MI, or internal transformer
modulations. Its original InfoNCE/feature-distillation objectives and existing
checkpoints retain their meaning.

| Component | Existing joint-diffusion route | New `TMCLTrainer` path |
|---|---|---|
| Base learning | Joint classifier/denoiser fits precede extra phases | VI pretraining, then acquisition and VI/MI consolidation |
| Acquisition loss | Existing OPL-inspired squared cross-cosine separation | Official OPL with mean absolute cross-cosine separation |
| Modulation locations | Hidden classifier-projection features | Native attention Q/K/V/output and MLP linear outputs throughout the DiT |
| Consolidation objective | InfoNCE or feature MSE, with route-specific CE controls | Separate VI and MI multiview Barlow objectives |
| Training images | Supplied current/generated-replay pool | Supplied current-session RGB images only |
| Retained memory | Route-specific old gates and native teachers | Class-specific layer modulations |
| Classification endpoint | Existing trained classifier | Separately fitted frozen-feature readout |

## Entry point and inputs

Use `TMCLTrainer(wrapper, task_classes, settings)` from
[`tmcl.py`](tmcl.py), with [`TMCLSettings`](tmcl_config.py). The wrapper must
already contain a built, float32 native `DiTClassifier` with CFG enabled.
The adapter uses clean inputs, timestep zero and the CFG null condition.
Class IDs select modulations; they are not supplied as class-conditioning
inputs to the native feature network.

The following is an integration sketch for a remote runner, not a dataset
loader or a standalone campaign command. `wrapper`, `task_classes` and the
three current-session arrays must already exist under the frozen protocol.

```python
from semantic_consolidation.tmcl import TMCLTrainer
from semantic_consolidation.tmcl_config import TMCLSettings


settings = TMCLSettings(seed=101)
trainer = TMCLTrainer(wrapper, task_classes, settings)
record = trainer.run_session(
    wrapper.preprocess(current_raw_images), 
    wrapper.preprocess(current_labelled_raw_images), 
    current_labelled_class_ids, verbose=0
)
boundary = trainer.save("artifacts/tmcl/seed-101")
```

Call `run_session` once for each scheduled task in order. The full
`current_raw_images` array contains that session's permitted training images;
the labelled subset supplies acquisition only. Labels must be aligned integer
IDs consistent with the declared disjoint schedule. Acquisition needs at least
two new classes and at least two labelled examples per class. The caller owns
the split, label fraction, subset selection and image provenance: tensor
geometry checks cannot establish that a supplied image came from the current
task. Do not concatenate historical images into this no-replay input.

Arrays passed directly to `run_session` are in the wrapper's model coordinates.
Use `wrapper.preprocess` once on raw pixels; do not preprocess an already
normalized array again. No official-test images belong in these training
arrays. Data loading and evaluation are external to this trainer.

The reference DiT recipe should set `ln_no_adaptation=True` and
`clf_ln_no_adaptation=True` when constructing its native model, as specified in
the experiment protocol. This avoids interpreting initially closed native
condition-adaptive residual branches as learned class modulation. Changing
that architecture setting is a separately recorded adaptation.

## Phases and trainable state

1. **Initial pretraining:** only the first session receives VI pretraining.
   Augmented current images pass through the unmodulated encoder and the VI
   projector. Gradients update the connected native feedforward parameters and
   VI projector.
2. **Acquisition:** a uniformly sampled new class selects one class-specific
   bank of internal affine modulations. One-versus-rest batches use current
   labelled images. The encoder runs in inference mode; its weights and
   normalization state stay fixed. Only the selected new-class bank changes.
   Old banks stay fixed, including through optimizer momentum and decay.
3. **Consolidation:** VI and MI use current images without labels. VI uses all
   unmodulated augmented views. MI passes the first unmodulated view through its
   own projector and predictor. Other views receive independently sampled
   retained class modulations; their current encoder/projector outputs are
   gradient-stopped targets. All retained banks stay fixed. The connected
   feedforward parameters and active representation heads learn.

MI targets come from the **current** encoder, rather than a frozen copy of the
previous session. Stopping gradients does not switch off train-mode BatchNorm
updates in the MI projector; this follows the released `no_grad` call pattern.
Optional past-model state invariance (SI) is a different branch. With
`si_weight != 0`, consolidation from session two onward compares current VI
projections, through a separate predictor, with the previous session's frozen
unmodulated encoder and VI projector. The previous snapshot is taken **before**
session-end head resets, then kept in inference mode. SI neither reads old
images nor changes that snapshot. The default `si_weight=0` excludes its
predictor, previous-model memory and forward-pass cost.

For the VI-only baseline, construct an independent, identically initialized
wrapper with `TMCLSettings(mi_weight=0.0, seed=101)` and otherwise identical
settings/data. This disables acquisition and unused modulation memory while
retaining the VI pretraining and consolidation budgets. MI has additional
acquisition and target-forward cost; equal VI epochs do not imply equal total
compute. Do not reuse an already trained wrapper as the baseline's initial
state.

## Objective definitions and provenance

The primary references are [TMCL paper v3](https://arxiv.org/html/2505.14125v3)
and official commit
[`2cb306cc5cbe6d13a3dd4991998ee3e7c2948c8b`](https://github.com/Dendritic-Learning-Group/tmcl/commit/2cb306cc5cbe6d13a3dd4991998ee3e7c2948c8b).
The TensorFlow implementation in [`objectives.py`](objectives.py) follows the
pinned [OPL source](https://github.com/Dendritic-Learning-Group/tmcl/blob/2cb306cc5cbe6d13a3dd4991998ee3e7c2948c8b/tmcl/nn/losses/contrastive_opl.py)
and [multiview Barlow source](https://github.com/Dendritic-Learning-Group/tmcl/blob/2cb306cc5cbe6d13a3dd4991998ee3e7c2948c8b/tmcl/nn/losses/mv_barlow.py).
No official PyTorch training backend is introduced.

For row-normalized positive features `P` and negative features `N`, default OPL
is `1 - mean(offdiag(P @ P.T)) + mean(abs(P @ N.T))`. Positive self-pairs are
excluded; there is no negative-negative attraction. The absolute value occurs
before averaging, so opposite signed cross-class cosines do not cancel.
Zero-norm cosine rows are rejected. The separately available squared variant
is not selected by the default TMCL phase.

This guard does not replace a safe-zero convention in the inspected reference:
TMCL calls TorchMetrics pairwise cosine, and its minimum declared version
[1.6.3 divides directly by each row norm without an epsilon](https://github.com/Lightning-AI/torchmetrics/blob/v1.6.3/src/torchmetrics/functional/pairwise/cosine.py).
The native guard reports undefined zero-row inputs before NaNs propagate.
The official dependency requirement is a lower bound, not an exact environment
lock; this source check is not a claim about every possible newer dependency.

For `K` aligned projected views of shape `[B, D]`, Barlow normalization computes
one population mean and variance over all `K*B` rows. If `Z_v` is a normalized
view and `M` is the per-example mean over views, each correlation is
`C_v = M.T @ Z_v / B`. The objective sums over views:

```text
scale * [sum((diag(C_v) - 1)^2) + lambda * sum(offdiag(C_v)^2)]
```

This is not pairwise BT or a mean over view losses. Its default normalization
epsilon is `1e-5`, matching the effective nonsynchronized official path. All
supplied views are differentiable unless the caller explicitly stops them;
the MI phase owns its target stops. Constant features retain a diagonal
penalty, but a low training loss still does not establish useful features.

Optional SI makes its loss choice explicit: `si_method="cassle"` or `"pnr"`,
and `si_objective="barlow"` or `"ghosh"`. The Barlow choice uses per-branch
normalization and averages the first two same-view current/past losses. The
Ghosh choice separately normalizes each branch, correlates each view with their
mean, averages the diagonal penalties and applies one redundancy penalty to
the mean's covariance; SI averages this two-branch loss over all supplied views.
These are different objectives from the jointly normalized main VI/MI loss.
Defaults for SI are redundancy `0.0051` and scale `0.1`.

PNR additionally penalizes the mean cosine similarity between crossed current
predictions and stopped past features. It pairs views `0/1`; the Ghosh variant
also pairs `2/3` and requires exactly four views. A positive cosine penalty is
minimized, so these crossed pairs are repelled. Unlike OPL's inspected pairwise
cosine implementation, PNR's reference cosine uses a safe norm floor; the
native objective preserves its `1e-8` floor. This follows the intended
[PNR regularizer](https://proceedings.mlr.press/v235/cha24a.html) and explicitly
corrects an upstream discrepancy: the pinned Barlow-PNR branch computes the
repulsion but omits adding it to the SI loss, whereas its Ghosh branch adds it.
See the pinned [SI call site](https://github.com/Dendritic-Learning-Group/tmcl/blob/2cb306cc5cbe6d13a3dd4991998ee3e7c2948c8b/tmcl/main_tmcl.py#L919)
and [PNR cosine helper](https://github.com/Dendritic-Learning-Group/tmcl/blob/2cb306cc5cbe6d13a3dd4991998ee3e7c2948c8b/tmcl/nn/losses/pnr.py).
This correction is not claimed to reproduce the buggy branch's outputs.

## Paper/code choices in the native implementation

The released launcher predates paper v3. The following choices are explicit;
do not label their mixture exact released-code parity. See
[Appendix A](https://arxiv.org/html/2505.14125v3#A1), the
[pinned launcher](https://github.com/Dendritic-Learning-Group/tmcl/blob/2cb306cc5cbe6d13a3dd4991998ee3e7c2948c8b/slurm/submit_cifar100_s5.py)
and [training implementation](https://github.com/Dendritic-Learning-Group/tmcl/blob/2cb306cc5cbe6d13a3dd4991998ee3e7c2948c8b/tmcl/main_tmcl.py).

| Choice | Paper v3 / pinned source distinction | Native setting or policy |
|---|---|---|
| Epoch budgets | Paper acquisition 100; launcher 50 | First-session pretraining 250; acquisition 100 and consolidation 200 per session |
| Barlow scale | Paper ConViT 0.1; source/launcher 0.024 | `TMCLSettings.barlow_scale=0.1`; standalone objective default remains 0.024 |
| Redundancy and views | Both specify lambda 0.005 and four views | `redundancy_weight=0.005`, `augmentation_views=4` |
| Projector width | Paper prints first width 2028; source uses 2048 | Three 2048-wide layers, with hidden BatchNorm/ReLU; separate VI and MI projectors |
| Session-end resets | Paper resets projectors and predictors; launcher disables reset, and source reset branch omits the predictor | Default resets both projectors and enabled predictors in place; optimizer state is retained; an SI snapshot precedes reset |
| Representation optimizer | Source preserves optimizer state across sessions | Persistent `TMCLAdamW` state for backbone/VI, MI/predictor and optional SI predictor; one-dimensional bias/normalization parameters excluded from feedforward decay |
| Representation LR | Source supplies session-one schedule concatenation and batch scaling | Session-one pretraining/consolidation share a cosine curve; later sessions restart the curve, preserving moments; peak base LR scaled by `batch_size/256`, minimum `1e-6` |
| Acquisition optimizer | Source has persistent modulation optimizer and depth-dependent decay | A separate optimizer for each new class; only its selected updates advance its moments; old gates remain exact; cosine depth decay `0.4` to `0.04`, minimum LR `1e-5` |
| Modulation decay | Paper names affine gain; released code stores its residual around identity | Decay stored gain residual toward zero (effective gain toward one) and affine bias toward zero; no decay toward zero effective gain |
| Acquisition sampling | Paper uses an independent half-probability class mixture; released balanced iterator has different finite sampling details | Uniform new-class focus and Bernoulli `0.5` positive draws, with replacement; redraw/count batches without two positives or a negative, with bounded failure |
| Head details | Lightly projector's final linear has a bias; PyTorch BN stores unbiased running variance | Native Dense layers match bias placement and uniform initialization distributions; `TMCLBatchNormalization` uses population training variance and unbiased evaluation running variance |
| Optional SI | Source offers CaSSLe/PNR with Barlow/Ghosh; Barlow-PNR omits its computed repulsion | Explicit named options, frozen pre-reset past state and separate predictor; PNR repulsion is actually added |
| Image transforms | Paper and released acquisition/view ordering differ | Existing TensorFlow paper-derived transforms; interpolation/sampling are backend adaptations |

The paper does not specify whether resetting a projector also resets optimizer
moments. Resetting head values in place while retaining their optimizer state
is recorded here explicitly. Separate acquisition optimizers make the selected
class's Adam update count independent of other classes' updates; this matters
because Keras optimizer iterations are global, while the source's optimizer
maintains per-parameter step state. New classes start with new moments, and no
old class is trained again by acquisition.

[`TMCLAdamW`](tmcl_optimizers.py) is a registered native Keras AdamW subclass,
not another training framework. The reference update adds epsilon `1e-8` after
second-moment bias correction:

```text
lr * [m / (1 - beta1**t)] / [sqrt(v / (1 - beta2**t)) + epsilon]
```

Keras 3.11.2's ordinary AdamW uses the epsilon-hat form; setting its epsilon to
`1e-8` alone does not fix that difference. Native decoupled weight decay,
exclusions and optimizer-state serialization are retained. AMSGrad is
supported; repeated sparse embedding indices are summed into a dense gradient
before the second moment is formed. Float32 is the supported parameter mode.
The reference per-parameter clock is represented by separate optimizers for
parameters with the same update frequency; this does not claim equivalence
for arbitrary intermittently omitted parameters within one group. Numerical
kernel order and random streams remain backend differences.

The native head initializers reproduce uniform `[-1/sqrt(fan_in),
1/sqrt(fan_in)]` distributions, including enabled biases, using TensorFlow
random streams. BN training uses population variance, while its running
variance update uses `B/(B-1)` times that variance, matching the reference's
evaluation-state convention. These are mathematical and distributional
matches, not identical random weights across frameworks.

Default batch size is 256. A virtual epoch has
`ceil(current_images / batch_size)` updates, even when acquisition labels cover
only a small fraction of those images. Batches have fixed size and use sampling
with replacement; virtual epochs are not guaranteed one-pass dataset sweeps.
Zero phase epochs are explicit ablations. A shortened qualification must also
shorten or disable the ten-epoch warmup so each executed acquisition or combined
representation trajectory is longer than its warmup. These are substantial
reference budgets, not measured optimal
CIFAR-10 hyperparameters.

## Readout, recovery and supported scope

Representation phases do not train the native classifier output head. Its
ordinary accuracy is therefore not a valid learned-readout endpoint for this
path. [`tmcl_evaluation.py`](tmcl_evaluation.py) provides separate
`linear_evaluate` and `knn_evaluate` functions. Both take the adapter, training
images/labels, evaluation images/labels and the declared benchmark vocabulary.
They freeze and verify every native/modulation variable. The caller owns split
provenance and must record offline access to old training labels separately
from the no-replay representation-training protocol. Paper-style offline
readouts use the **full benchmark's** training labels at every representation
boundary, including classes not yet encountered in representation training.
For the CIFAR-10 adaptation this means all ten classes. A seen-only vocabulary
is a separately labelled adaptation and cannot populate the paper's future-task
transfer matrix.

The linear readout concatenates pooled outputs of the last four actual blocks
on the classifier's sequential ancestor path, including the classifier suffix.
It requires four available blocks and reports native CLS/GAP pooling through
`adapter.readout_spec(4)`. This is an explicit DiT endpoint adaptation, not a
claim that native blocks are ConViT blocks. The main VI/MI feature endpoint is
unchanged by these readout taps.

The default linear probe uses all supplied training labels, a fresh linear
head, 100 epochs, batch 1024, shuffled full batches and no feature L2
normalization. Its base learning rate `0.1` is scaled by `batch_size/256`, giving
`0.4` at the default batch, then decayed on an **epoch** cosine clock. The native
`TMCLSGD` subclass stores an unscaled gradient-momentum buffer with momentum
`0.9` and zero decay. Ordinary Keras SGD stores learning-rate-scaled velocity
and therefore differs when this schedule changes; the subclass applies the
current rate to the full buffer. Clean/flipped frozen features are cached, and
each shuffled training row independently selects a horizontal flip with
probability `0.5` per epoch. Evaluation uses clean features only.

Weighted kNN uses the last block, L2-normalized clean features, all supplied
training-label bank rows, `k=20` and temperature `0.07`. Exponential cosine
weights are summed by class. Both readouts predict over **all** declared classes;
task metrics select evaluation rows, never restrict prediction columns. Their
defaults follow the pinned outer training script's readout calls and
[linear](https://github.com/Dendritic-Learning-Group/tmcl/blob/2cb306cc5cbe6d13a3dd4991998ee3e7c2948c8b/tmcl/main_linear.py)/[kNN](https://github.com/Dendritic-Learning-Group/tmcl/blob/2cb306cc5cbe6d13a3dd4991998ee3e7c2948c8b/tmcl/main_knn.py)
implementations. Readout fitting remains a remote compute task.

`paper_transfer_metrics` implements the paper's Appendix B equations 9 and 10
from a complete sessions-by-tasks accuracy matrix and independent task-only
VI/Barlow baseline accuracies, all using that same full-vocabulary kNN protocol.
Each baseline is trained from scratch on its one task; it is neither the
sequential run's diagonal accuracy nor an untrained encoder. The helper cannot
verify those scientific identities from arrays alone. Do not substitute a
conventional final-minus-diagonal forgetting score for these paper metrics.

`save` and `restore` use `common.recovery` and authenticate the source,
configuration, architecture and fixed class schedule. Restore into a fresh,
matching `TMCLTrainer`. Only completed sessions are recovery boundaries; a
failed partial session must restore the preceding boundary. This is not
optimizer-step recovery or a promise to resume midway through acquisition.
Completed boundaries retain the native network, gates, representation heads,
persistent representation optimizers, optional frozen SI state and RNG state. Session records identify
input-array digests, phase budgets, actual update counts and gradient-variable
names; the outer runner must still bind those arrays to its dataset manifest.

Supported execution is single-device eager TensorFlow 2.20.0 / Keras 3.11.2,
float32, current RGB images and the native DiT adapter. Scoped modulation
hooks reject overlapping calls on one network and tracing. UNet semantic
modulation, ConViT positional-prior modulation, variational reshapers,
distributed/synchronized Barlow statistics and old-image replay are not
implemented in this path. A UNet replay generator is a separate experimental
axis; its availability does not make a UNet semantic adapter available.

## Validation and experiment status

The latest targeted qualification on 2026-10-13 passed **127 tests in 110.110 seconds**, with zero
failures, errors or skips. Its receipt timestamp is
`2026-10-13T03:08:27.919425+00:00`. Source contracts also passed across all 63
semantic Python files: 91 classes, 755 functions and 922 branches. This
qualification includes the augmentation-only `tf.function` optimization;
feature hooks, sampling, gradient tapes and optimizer application remain
eager. Three added tests check seed/view parity, one dynamic-batch trace and
state/RNG preservation without tracing the adapter.

On the verified A100, a standalone batch-256/four-view augmentation profile
decreased from `12.3559` to `0.970161` seconds, a `12.74x` component speedup,
with parity within the stated `2e-6` tolerance. This is **not** a measured
`12.74x` training speedup. The completed full-size phase timings were1.84038s
for VI,0.74513s for acquisition,3.29473s for VI+MI and4.30833s for VI+MI+SI,
per update over five measurements after two warmups. See the
[timing record](../experiments/paper_replication_diagnostics/TIMING.md) for
memory, training-only budget projections and their limitations.

On 2026-10-13, 13 new objective tests and 18 unchanged legacy objective tests
passed in a separate CPU-only process on remote host `480cfb8aed52`, using
`/opt/tf220/bin/python`. There were 31 passes, no failures and no skips. Checks
cover independent analytic/numeric loss values, finite-difference gradients,
graph/eager objective agreement, target-stop ownership and preservation of the
legacy losses. The remote log is
`/tmp/tmcl_revision_20261013/.tmp/objective-tests.log`.

That earlier run predates the SI, optimizer, head-state, sampling and readout
extensions. The subsequent pre-graph qualification on 2026-10-13 passed **124 tests
in 91.287 seconds**, with zero failures, errors or skips. It ran on remote host
`df84424aab32`, A100 80 GB UUID
`GPU-daa7d297-ec68-bbe6-2b2f-6ad86f43e89c`, in the isolated
`/workspace/tmcl_revision_20261013b` checkout using `/opt/tf220/bin/python`,
TensorFlow 2.20.0, Keras 3.11.2 and NumPy 2.1.2. The environment's `pip check`
also passed. This is a targeted semantic qualification, not the full
repository test suite.

The initial 124-case attempt passed 123 tests and failed one exact resumed-state
comparison by approximately `1.8e-12`. The successful rerun enabled the
project's existing `deterministic_ops=True` setting; it did not relax the
comparison tolerance. The tiny native fixture also explicitly uses
`ln_no_adaptation=True` and `clf_ln_no_adaptation=True`, matching the stated
recipe. Exact checkpoint continuation is established for that deterministic
fixture/environment, not promised for nondeterministic kernels or arbitrary
devices. Historical failures and their tested source remain preserved in the
[validation record](../experiments/paper_replication_diagnostics/validation_20261013/README.md).

That pre-graph qualification passed source contracts across all 63 semantic Python
files: 90 classes, 747 functions and 923 branches. Its result receipt was
written at `2026-10-13T02:43:10.406067+00:00`; the exact command and log paths
are in the validation record. Its source and logs are preserved in the
downloaded `qualification_20261013b.tar.gz`; this archive predates the graph
optimization and does not establish its 127-test result. No completed
full-run duration, GPU concurrency capacity, benchmark accuracy or retention
gain is claimed here. The protocol's measured cost/duration gate still precedes
renting further GPUs or launching a full campaign.
