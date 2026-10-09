# Diffusion model wrappers

This directory contains the stateful training and sampling layer around the raw
networks in [`diffusion.models.transformer`](../transformer/README.md) and
[`diffusion.models.convolution`](../convolution/README.md).

The division of responsibility is intentional:

| Raw-network packages | Wrapper package |
| --- | --- |
| Patch/condition embeddings | Forward noising and timestep sampling |
| Attention, feature routing, spatial layers | Loss composition and metrics |
| Noise/image and class heads | Optimizer steps and EMA updates |
| Intermediate features, regularizers, latent statistics | Evaluation selection and reverse sampling |
| Architectural depth growth | Progressive curriculum orchestration |

Compile, fit, evaluate, and sample through a wrapper. Call the raw network
directly when you specifically need its tensor/intermediate-feature API.

## Public classes

- `DiffusionModel`: general denoising wrapper with raw/EMA networks, DDIM/DDPM
  sampling, VAE bottleneck sampling, conditional/CFG vocabularies, noise teachers,
  and progressive curricula.
- `DiffusionClassifier`: joint denoising/classification wrapper for
  `DiTClassifier`, `DiTEncoderDecoderClassifier`, and `UNetClassifier`; owns
  classifier teachers, ordinary Keras classifier fitting, and teacher-head growth.
- `DiffusionClassifierV2`: alternating generator/discriminator variant with two
  optimizer states and explicit variable ownership.

The package aliases `NetworkName` (`"raw" | "ema"`), `TrainType`
(`"cond" | "uncond"`), and `ClusteringType` (`"uniform" | "log_snr"`).

## Data and label conventions

Keras `fit`/`evaluate` datasets normally yield:

- images: raw `[0,255]` pixels `[B, H, W, channels]`; integer and floating
  inputs are accepted, and the wrapper owns model-space conversion;
- classes: integer `tf.Tensor` `[B]`; fixed-width models use zero-based IDs in
  `0..num_classes-1`, while dynamic models map observed dataset IDs.

Set `preprocess_type="standardize"` on the wrapper (the default) for fixed
raw-pixel conversion to `[-1,1]`; `"min-max"` maps raw pixels to `[0,1]`.
Constructor `None` keeps values unchanged. These are the only supported modes,
and the setting is saved with the wrapper. The public `preprocess(x, preprocess_type=None)` and
`postprocess(x, preprocess_type=None, clip=False)` methods use that saved mode
when the override is `None`; postprocessing performs the inverse conversion
to raw floating pixel units and clips to `[0,255]` only when requested for
`"standardize"` or `"min-max"`. Passthrough preserves
values in both directions, even with `clip=True`, because they configure no
pixel bounds.

Training, evaluation, and ensemble-evaluation input paths share
`prepare_images(images)`: it calls `preprocess` once, adds a channel axis to
grayscale batches, and resizes with `resize_method` and `resize_antialias` when
the active resolution differs from `image_size`. It does not sample timesteps,
noise, or conditioning labels. `preprocess` alone continues to preserve shape.
Low-level `q_sample`, `noisify`, and direct raw-network calls consume model
coordinates: use `model.prepare_images(raw_images)` when active-resolution
preparation is also needed.

For fixed-width models with classifier-free guidance, `prep_inputs` shifts
dataset classes by one. Network label 0 is null and real network labels are
1..`num_classes`. Direct raw-network calls and `sample(labels=...)` expect
these network IDs; wrapper training data uses dataset IDs.

With raw `num_classes=None`, the wrapper discovers real labels during `fit` and
stores their consecutive zero-based classifier targets in `seen_classes`.
`prep_inputs` adds the CFG offset before a raw-network call, and default
sampling does the same; explicit `sample(labels=...)` values remain network
condition IDs. The dictionary is retained by reference in the wrapper
initialization config, so each discovery updates the serializable state
immediately. Passing a saved nonempty mapping to the constructor restores
dynamic growth and expands a smaller raw/EMA topology before checkpoint
weights are loaded.

## Basic training and sampling

```python
import tensorflow as tf

from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer
from diffusion.models.wrapper.diffusion_model import DiffusionModel


network = DiffusionTransformer(
    num_classes=10, 
    image_size=28, 
    channels=1, 
    patch_size=2, 
    dim=64, 
    cond_dim=64, 
    depth=4 
)

model = DiffusionModel(
    network=network, 
    preprocess_type="standardize", 
    use_ema=True, 
    scheduler_name="clipped_cosine", 
    p_uncond=0.1, 
    test_cfg_scale=4.0 
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4), 
    loss="mse" 
)

# dataset yields (raw images [B,28,28,1] in [0,255], classes [B])
history = model.fit(dataset, epochs=10, validation_data=validation_dataset)

# Sampling labels are network IDs: 1 and 2 correspond to dataset classes 0 and 1.
images = model.sample(labels=[1, 2], steps=50, eta=0.0)

# Three consecutive samples for condition 1, then three for condition 2.
images = model.sample(labels=[1, 2], samples_per_label=3, steps=50, eta=0.0)

# Include CFG null condition 0 before the default real-class conditions.
images = model.sample(add_null_label=True, steps=50, eta=0.0)
```

With `"standardize"` or `"min-max"` preprocessing, `sample` returns clipped raw
float images `[B,H,W,C]` in `[0,255]`. Passthrough modes return values unchanged
without clipping; the caller retains responsibility for their units.
Use `model.preprocess(images, "min-max")` to plot raw pixels in `[0,1]`. Set
`return_x_ts=True` and/or `return_x0s=True` to receive a list containing final
images plus step-wise NumPy trajectories in the same raw pixel units. `eta=0` is deterministic DDIM;
`0 < eta < 1` is stochastic DDIM; `eta=1` is DDPM-equivalent only when using
the full consecutive schedule.

Both samplers accept `add_null_label=False` immediately after `labels`, before
`samples_per_label`. Every argument can be passed by position or keyword. When
`labels=None`, enabling it prepends null condition ID 0 for a CFG network.
The remaining defaults are observed class conditions for a dynamic model and
all real class conditions for a fixed-width model. Without CFG, the flag has
no effect. Explicit `labels`, including `[]`, override default label selection;
to request a particular null-conditioned sample, include 0 explicitly.
Positional calls must include the null-label flag before the sample count.

`sample` and `sample_vae` accept `samples_per_label=1`. In ordinary diffusion
sampling and direct `sample_vae` calls, an integer greater than one repeats
each selected condition ID contiguously, including the null condition when
selected. Supplied `x_t` or latent
`z` batches must already have one row for each repeated condition ID.
Only labels are repeated. The positional argument order begins
`network_name, labels, add_null_label, samples_per_label`, followed by `x_t`
in `sample` or `z` in `sample_vae`.

With `swap_noise_image=True`, `sample(...)` forwards both `add_null_label` and
`samples_per_label` to `sample_vae(...)`, and passes `x_t` as the latent `z`.

Repeat counts must be positive integers; boolean, fractional, zero and negative
counts are rejected. The existing label helper uses `tf.repeat`, preserving
empty requests and supporting symbolic label tensors. Supplied image/latent
batches must still match the repeated labels.

`network_name="ema"` is the default for sampling. Use `"raw"` whenever
`use_ema=False`. With EMA enabled, a deferred raw network and its clone are
built before their initial weight copy, so `build=False` raw configurations are
supported.

## `DiffusionModel` configuration

Important constructor controls are:

- `scheduler_name`: `linear`, `scaled_linear`, `squaredcos_cap_v2`,
  `clipped_cosine`, `sigmoid`, `quadratic`, `ve`, `karras`, `sub_vp`, or
  `logistic`;
- `p_uncond`: probability that a shifted training label becomes null ID 0;
- `train_cfg_scale=None`: one conditional training pass; a numeric value also
  runs an unconditional pass and applies CFG;
- `test_cfg_scale`: evaluation/sampling guidance;
- `test_steps` and `test_eta`: default sampler discretization/stochasticity;
- `noise_loss_coef`, `noise_distil_loss_coef`, `image_loss_coef`, `kl_loss_coef`,
  `ctr_loss_coef`: loss
  multipliers; a zero auxiliary coefficient disables that objective;
- `show_separate_noise_losses=False`: keep the usual `noise_loss` progress
  metric. When true, rename it to `total_noise_loss` and also show
  `cond_noise_loss` for non-null rows and `uncond_noise_loss` for null rows.
  The two split metrics are reporting-only and custom callbacks should monitor
  the renamed total metric;
- `kl_train_type` and `ctr_train_type`: choose conditional or unconditional
  branch values for auxiliary losses;
- `*_noisified_min_timesteps` / `*_noisified_max_timesteps`: half-open train and
  test ranges `[minimum, maximum)`; a base-wrapper maximum of `None` or `0`
  selects clean-only inputs, while `-1` uses the full timestep count;
- `resize_method`, `resize_antialias`: active-resolution input resizing;
- `map_preprocess=False`: keep `prep_inputs` in the train/test step. When true,
  `fit`, `evaluate`, validation, and progressive stages map each
  `tf.data.Dataset` through `prep_inputs_map` in the CPU input pipeline;
- `swap_noise_image=True`: trains a direct noisy-`x_t` prediction and makes
  `sample` use the network's KL-enabled flatten bottleneck via `sample_vae`.

### VAE sampling topology

A VAE-capable raw network configures `latent_dim_ratio` as a list with exactly
one positive entry per flatten/unflatten pair, ordered by ascending flatten
depth; omission selects a full-width ratio of `1.0` for every pair.
`sample_vae(...)` starts at the first flatten boundary and injects one sampled
or supplied latent for every pair in that same order. The immediately
following unflatten restores the feature consumed downstream.

A transformer VAE can arrange the adjacent pair or pairs as one central
bridge after real encoder computation and before real decoder/up-sampling
computation. This applies to single- and multi-level bottlenecks.
Keep the bridge contiguous and free of transformer/local-mixer processing,
with downsampling before it and upsampling after it. Block class does not
define encoder versus decoder placement; depth
relative to the bridge does. Normal training executes an encoder-feature route
attached to a flatten stage before computing that pair's posterior. Sampling
bypasses the whole flatten stage—including that training-only route—and injects
its latent directly. Convolutional multiscale U-Net instead places stochastic
pairs at successive encoder scales; its decoder skip routes source their
unflattened outputs.

For one pair, `z` may be a single rank-2 tensor or a one-element sequence. For
multiple pairs, it must be a sequence in the same ascending-depth order; when
`z=None`, the wrapper draws the latents independently. Decoder routes must use
every stochastic unflattened feature—otherwise a later flatten overwrites the
stream and leaves the earlier latent disconnected—and must not select
pre-latent encoder features in a way that bypasses the variational path.
`sample_vae` does not check network topology. These layout recommendations also
apply when `swap_noise_image=True` delegates `sample(...)` to `sample_vae(...)`.
Even for one pair, an attached decoder needs to consume a post-flatten encoder
feature for the sampled latent to affect its output.

`compile(loss=..., **kwargs)` forwards `optimizer`, `run_eagerly`,
`steps_per_execution`, supported `jit_compile`, metrics, weighted metrics, and
loss weights to Keras. `fit(**kwargs)` and `evaluate(**kwargs)` accept the normal
Keras batch/epoch/callback/validation/step arguments documented in their
docstrings. `summary(**kwargs)` forwards Keras summary display options to the
raw network.

`compile_teacher(teacher_name="previous", **kwargs)` compiles a selected
attached teacher separately when `trainable_teacher=True`. `DiffusionModel`
supports `"previous"` for `teacher_network`, `"current"` for the shared
`current_teacher_network`, and `"noise"` for `noise_teacher_network`.
`DiffusionClassifier` and V2 additionally support `"classifier"` for
`classifier_teacher_network`. `get_teacher_names()` lists the wrapper's supported
selectors. `fit_teacher(..., teacher_name=...)` uses the same selector; omitting
it retains the previous-teacher behavior.
`get_teacher_network(teacher_name)` returns the selected raw model and
`get_teacher_model(teacher_name)` returns its cached native training wrapper.
Classifier wrappers also return ordinary Keras classifier owners through this API
and retain their original fine-tuning masks. Each selected teacher keeps independent
compilation and optimizer state. Native teachers accept their wrapper's normal
compile options; ordinary classifiers receive options through their own `compile`.
Student weights, optimizer state, and compile settings are unchanged. Teacher and student optimizers must be
independent. This method may run before student `compile`, but `fit_teacher`
still requires student compilation. Recompiling follows the teacher's normal
compile behavior; use a fresh optimizer to reset its state.

For native teachers, explicit `compile_teacher` settings survive later student
`compile` calls and are cleared when the teacher is replaced. Without an
explicit override, student compilation gives the native teacher matching
compile settings and a separate optimizer.

`rebuild_teacher_network(teacher_name, seed=None)` replaces an explicitly selected
teacher with a newly initialized training graph. It supports the same roles as
`get_teacher_names()` and requires an attached teacher with
`trainable_teacher=True`. The current topology, class vocabulary, resolution and
role mappings are preserved. Trainable layers use their configured initializers;
layers intentionally frozen in the original fine-tuning mask retain their weights
and state. The temporary frozen state used for teacher inference does not define
that mask. Ordinary classifier graphs must be built Functional or Sequential
models, including nested pretrained backbones. The saved compile recipe creates
a fresh optimizer, and the student and other teacher roles retain their state.
The candidate is constructed before replacing the attachment and is frozen after
attachment. The return value is the replacement raw network.

Call this method explicitly before a task's specialist training phase:

```python
model.rebuild_teacher_network("noise", seed=task_seed)
model.rebuild_teacher_network("classifier", seed=task_seed)
model.fit_teacher(new_task_trainset, teacher_name="classifier", epochs=10)
```

Only classifier wrappers expose the `"classifier"` role. Rebuilding preserves
existing class-column and task-support metadata; a manual task loop updates the
upcoming task's support through `set_classifier_teacher_network` or
`set_noise_teacher_network` as usual. Automatic continual orchestration keeps its
existing persistent-specialist behavior until the caller explicitly requests a
rebuild.

Omitting `evaluate(network_name=...)` inherits `test_network_name`, including
validation inside `fit`. Explicit `"raw"`/`"ema"` overrides apply only to that
evaluation call; without EMA, both selectors resolve to the raw network.

## Noising and forward APIs

Useful programmatic interfaces are:

```python
x_t, noise, t = model.noisify(x0)
x_t_at_10 = model.q_sample(x0, tf.fill([batch_size], 10), noise)

x0_pred, eps, regularizers, latent_stats = model.forward(
    "ema", 
    x_t, 
    t, 
    t, 
    cond_labels, 
    uncond_labels, 
    scale=4.0, 
    training=False 
)
```

`set_timestep_bounds(minimum, maximum)` changes the active half-open draw range.
`set_current_resolution(size)` synchronizes the wrapper, raw network, and EMA
network. The resolution must be positive and divisible by the raw network's
patch size; encoder-decoder networks additionally validate the attached
decoder patch size. `None` restores the configured branch size or sizes.

## Progressive training

`fit_progressively` can change timestep ranges, resolution, and architecture in
the same ordered curriculum. A string changes one field, a `(name, value)` pair
provides an inline value, a set combines field names using companion sequences,
and a dictionary combines names with inline or `None` values.

```python
history = model.fit_progressively(
    stage_tasks=[
        {"timesteps": (700, 1000), "resolution": 14}, 
        ("timesteps", (300, 1000)), 
        {
            "resolution": 28, 
            "depth": "vision_transformer_block" 
        } 
    ], 
    stage_epochs=2, 
    final_epochs=1, 
    pacing_type="fixed", 
    x=dataset, 
    validation_data=validation_dataset 
)
```

Timestep/resolution changes apply before a stage. A depth addition applies after
its stage and first trains in the next stage or `final_epochs`. Exact raw-network
layer names are `feature_connector`, `cross_attention_connector`,
`vision_transformer_block`, `local_mixer`, `downsampler`, `upsampler`,
`reshaper`, and `cls_token_regularizer`. Connector specs accept `{"ids": [...]}`;
block specs accept `use_decoder` and `mlp_output_dim`; reshapers use
`"flatten"`/`"unflatten"`. See the transformer README for full syntax.

Shorthands are also available:

```python
model.fit_progressively("timesteps_only", stages_num=4, x=dataset)
model.fit_progressively(
    "resolutions_only", resolutions=[7, 14, 28], x=dataset
)
```

Generated timestep clusters are `uniform` or `log_snr`. Fixed pacing runs every
allocated epoch; plateau pacing uses epoch-wise Keras early stopping or the
project's batch-wise plateau callback. `stopper_mode` selects `"min"`, `"max"`,
or `"auto"` for either callback; use `"max"` when monitoring accuracy.
The returned `History` includes a
`progressive_stages` record and the resolved schedules. Timestep bounds and
resolution are restored on exit. Depth additions run after their training stage
and remain in the model. The full depth schedule is checked before fitting;
growth retains trained layers, optimizer state and old EMA values while new EMA
weights start from the corresponding raw weights. See the
[compatibility guide](../../../compatibility_migration.md).

## Joint diffusion classification

Use a `DiTClassifier` with `DiffusionClassifier`:

```python
from diffusion.models.transformer.di_t_classifier import DiTClassifier
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier


network = DiTClassifier(
    depth=4, 
    clf_depth=2, 
    feature_aggregation_ids_dict={1: [-1]} 
)
model = DiffusionClassifier(
    network=network, 
    clf_train_type="cond", 
    clf_loss_coef=8.6e-3, 
    mask_by_nulls=True, 
    p_uncond=0.1 
)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")
```

`mask_by_nulls=True` selects only examples whose dropped CFG label equals 0 for
classifier loss and accuracy. `mask_by_t_threshold=True` intersects that mask
with `t <= ceil(mask_t_percentage / 100 * timesteps) - 1`; zero percent selects
no timesteps. Set either switch false to remove that selection criterion.
`clf_train_type="uncond"` requires a numeric
`train_cfg_scale` because it needs the explicit unconditional pass.

Use `model.evaluate_ensemble_accuracy(dataset, weighted=True)` to
average classifier predictions across diffusion timesteps. It defaults to the
configured test network, accepts the `EnsembleAccuracy` options, and is also
inherited by `DiffusionClassifierV2`; pass `network_name="raw"` or `"ema"` to
select a network explicitly.

The classifier depth specification can grow both branches on built models.
The wrapper preserves existing layers, registers the appended variables with
the optimizer, and initializes new EMA weights from their raw counterparts.
Classifier growth requires a positive initial `clf_depth` and must preserve
the retained classifier head's feature width:

```python
depths = [{
    "network": "vision_transformer_block", 
    "classifier": {
        "feature_connector": {"ids": [-1]}, 
        "vision_transformer_block": True 
    } 
}]
history = model.fit_progressively(
    "depths_only", depths=depths, final_epochs=1, x=dataset
)
```

Classifier-specific progressive names also include `feature_aggregator` and
`cross_attention_aggregator`.

## Distillation training

`DiffusionModel` owns native/noise teacher attachment and training. Conditional
label mapping and CFG remain part of this denoising API. Set
`noise_distil_loss_coef > 0` to match the student's epsilon prediction to a frozen
teacher on the same `x_t`. Native diffusion teachers also receive the same
timestep, condition IDs, and CFG scale. The teacher
is run with `training=False`, its outputs are stopped, and its variables are
never optimized. `defer_teacher=True` permits task one to train before
continual learning snapshots the first completed denoiser.
The reported `noise_distil_loss` is an eligible-row population mean: batches
contribute their teacher-mask weight, and empty eligible batches contribute zero
weight. The differentiated per-batch KD objective is unchanged; `total_loss`
continues to aggregate complete batch objectives.

An ordinary callable Keras epsilon teacher can return a single noise tensor.
`teacher_noise_input_type` selects its call input: `"images"` passes `x_t`,
`"images_timesteps"` passes `(x_t, t)`, and the default
`"images_timesteps_labels"` passes `(x_t, t, labels)`. These selectors apply
only to callable teachers without the native full-return interface; native
teachers retain their existing diffusion call contract. For example, attach a
trained two-input Keras noise model as follows:

```python
model = DiffusionModel(
    network=student_denoiser, 
    teacher_network=noise_teacher,  # noise_teacher((x_t, t), training=False)
    teacher_noise_input_type="images_timesteps", 
    noise_distil_loss_coef=1.0
)
```

An image-only epsilon teacher instead uses `teacher_noise_input_type="images"`;
its images are still `x_t`, since it is teaching noise prediction.

On `DiffusionClassifier` and V2, classifier distillation is enabled when
`teacher_network` is supplied and
`clf_distil_loss_coef > 0`. It uses the distillation-token head when present,
otherwise the primary classifier head. Teacher-trained classifier regularizers also enable the
same inherited `map_preprocess` path when `ctr_loss_coef > 0` and their
`train_type` is `distil` or `both`. Pass a compatible raw classifier, another
wrapper whose `.network` is that classifier, or an ordinary image-only Keras
classifier as the programmatic teacher. The teacher is deliberately not a
YAML/dataclass field because it is a live Keras object. The wrapper unwraps and
freezes the effective raw network, calls it with `training=False`, and stops
gradients through its probabilities.

A teacher without callable `predict_class` is called as
`teacher(postprocess(clean_x0), training=False)` in both V1 and V2. It receives
clean raw pixel units even when the student classifier uses noised images;
timesteps and condition labels are not passed. The inverse uses the wrapper's
saved `preprocess_type` without clipping. This matches
`get_model(..., model_type="pretrained")`, which owns its application-specific
resizing and scaling. Native diffusion teachers continue receiving model coordinates.
Its single output must have shape `[batch, classes]`.
By default this output contains class probabilities. Set
`teacher_classifier_from_logits=True` when it contains logits so the wrapper
applies softmax before constructing distillation targets. For a trained Keras
classifier with a linear output layer:

```python
model = DiffusionClassifier(
    network=student, 
    teacher_network=image_classifier, 
    teacher_classifier_from_logits=True, 
    clf_distil_type="soft", 
    clf_distil_loss_coef=1.0, 
    noise_distil_loss_coef=0.0
)
```

A built image-only Keras classifier can also be trained through
`DiffusionClassifier.fit_teacher` or `DiffusionClassifierV2.fit_teacher`.
Attach it with `teacher_network=classifier` and `trainable_teacher=True`,
compile the student, then call `student.fit_teacher(...)`. An uncompiled
classifier first needs `student.compile_teacher(...)`; an already compiled
classifier retains its own optimizer, classification loss, and metrics unless
explicitly recompiled. For an ordinary classifier, pass the classification loss
and metrics to `compile_teacher` because omitted options use standard Keras
compile defaults. Only `fit_method="fit"` is supported for these teachers;
arrays or datasets,
validation arguments, and callbacks are forwarded to its Keras fit method.
Supply training and validation images in the classifier's own raw input range;
`fit_teacher` forwards these inputs directly without diffusion preprocessing.
See the [EfficientNetV2L example](../../../common/README.md#training-an-image-classifier-teacher).

The classifier's original nested layer trainability is restored during teacher
compilation and fitting, including a selected fine-tuning tail and frozen
BatchNormalization layers. It is frozen again afterward, including when
compilation or fitting raises an error.
Repeated fits retain its optimizer state, and student weights and optimizer
state are unchanged.

For class-incremental Keras classifier training, additionally set
`teacher_dynamic_classes=True` on `DiffusionClassifier` or V2 and retain
`teacher_training="each_task"` (the default). This option belongs to classifier
wrappers; native diffusion conditioning grows through `num_classes=None` and its
ordinary label discovery instead.
Both explicit `fit_teacher` calls and automatic `common.learner` orchestration
then discover sparse integer dataset labels, append new head columns when
needed, and keep a persistent label-to-column mapping. Distillation aligns this
mapping to the student's vocabulary, even when their class orders differ;
unused head capacity is excluded from teacher targets. Arrays and finite
datasets are supported. Validation labels must already be known or present in
the training labels; validation never creates classes. Sample weights and
callbacks retain their Keras meaning, while `class_weight` keys use dataset IDs.

Dynamic growth supports built, serializable Functional/Sequential image
classifiers ending directly in a standard biased Dense layer, including the
models returned by `get_model(..., conv_base_name="EfficientNetV2L")`. It preserves
backbone weights, the existing output columns, fine-tuning masks, and compatible
optimizer slots/counters; new columns and slot tails retain their initializers.
Growth replaces the selected attachment with the expanded Keras model, so use
`model.get_teacher_network(teacher_name)` to access the latest teacher
(`teacher_name="previous"` for the original `teacher_network` workflow). Other custom head architectures
require their own growth implementation; they retain fixed-head fitting with
`teacher_dynamic_classes=False`.

Newly discovered dataset IDs are sorted within each call and appended after
previously known IDs. For an already fine-tuned initial head, declare its column
meaning on the first call, for example
`model.fit_teacher(x, y, teacher_class_ids=[9, 7], ...)` for a two-column head
representing dataset classes 9 and 7. The declaration must cover every initial
column exactly once and cannot reinterpret an established mapping. Without the
dynamic option, callers retain the existing fixed-head target encoding.

Automatic continual fitting uses the scheduled current/replay pool, declares
scheduled classes before sampling, and checkpoints the teacher architecture,
mapping, fine-tuning mask and training state for recovery. Learning old classes
still depends on the supplied replay/distillation protocol; retaining head
weights alone does not prevent forgetting.

Other ordinary callables and callable epsilon teachers remain inference-only.
A plain single-output teacher cannot supply both classification and noise
distillation targets simultaneously. Attach separate
`classifier_teacher_network` and `noise_teacher_network` specialists for this
combination, or use a compatible native multi-output shared teacher. `teacher_noise_input_type`,
`teacher_classifier_from_logits`, and wrapper `preprocess_type` are
serialized configuration fields; the attached teacher remains a runtime-only
object.

Continual learning can instead construct the teacher automatically. With
`defer_teacher=True`, task one is allowed to run without a teacher;
`snapshot_teacher_network("raw")` makes an independent frozen copy of the
completed student and `set_teacher_network(...)` activates it for the next
task. The continual API performs both calls before discovering and adding the
new class. A supplied task-one teacher is replaced by the first student
snapshot. Runtime teachers are excluded from wrapper configuration.

Native wrapper teachers and snapshots retain their `preprocess_type` metadata.
Both previous- and current-teacher attachment reject known input-coordinate
mismatches before distillation: the teacher and student must use the same
`preprocess_type`, including `None` for passthrough. A raw native teacher without
this metadata remains caller-owned: its model input coordinates must match the student.

```python
teacher = DiTClassifier(
    num_classes=10, 
    feature_aggregation_ids_dict={1: [-1]}
)
student = DiTClassifier(
    num_classes=10, 
    feature_aggregation_ids_dict={1: [-1]}, 
    clf_distil_token_type="new_weight"
)
model = DiffusionClassifier(
    network=student, 
    teacher_network=teacher, 
    clf_distil_type="soft", 
    clf_distil_loss_coef=1.0, 
    clf_acc_coef=0.5, 
    clf_distil_acc_coef=0.5, 
    ctr_acc_coef=0.0
)
model.compile(optimizer="adam", loss="mse")
model.fit(dataset, validation_data=validation_dataset, epochs=10)
```

Automatic distillation preparation requires a two-component `tf.data.Dataset`
yielding `(x0, dataset_labels)`. `fit`, `evaluate`, validation datasets, and
progressive-fit stages map it lazily through the classifier's override of
`prep_inputs_map`. The ordinary seven-value result is extended to:

```text
(x0, noise, t, x_t, cfg_labels, uncond_labels, classes, teacher_labels)
```

V1 accepts `clf_train_noisified_max_timesteps` and
`clf_test_noisified_max_timesteps` when `clf_train_noisy_input_type="noisy"`.
Their default `None` preserves shared diffusion inputs in training and clean
classifier inputs in evaluation. Explicit `0` selects clean images, `-1` uses
the full horizon, and positive values sample `[0, cap)` independently of the
diffusion bounds. With `clf_train_noisy_input_type="clean"`, both caps are ignored.
Explicit caps add a classifier pass during full-batch training or replace only
classifier rows when `clf_train_batch_fraction > 0`. CFG and timestep masks
still select rows using the original diffusion inputs. Ensemble-loss replacement
cannot be combined with explicit V1 caps because it performs its own noising.

During training a native teacher with callable `predict_class` receives
`(x_t, t, selected_labels)`, where
`selected_labels` is the conditional or unconditional prepared branch selected
by the classifier input policy. With an explicit cap, the mapped batch caches
the classifier image/time pair immediately after the seven diffusion tensors;
teacher and student consume that same corruption. During validation/evaluation
the native teacher receives the same clean or capped input and unconditional
label as the student. Its `predict_class` method supplies probabilities;
`teacher_classifier_from_logits` does not alter that interface. An ordinary
image-only teacher always receives `postprocess(clean_x0)` instead. The custom train/test
steps consume the prepared tuple without noising
or shifting it a second time. Array inputs, separate `x`/`y`, validation tuples,
and already-prepared datasets are not automatically adapted by this path.
Progressive resolution changes are mirrored to compatible teachers. For a
narrower past teacher, old conditional IDs are retained and a new-task ID is
replaced by the safe zero/null condition before lookup. When continual
growth gives teacher and student different class widths, teacher probabilities
are restricted to shared columns and normalized before either loss is computed.
The heads must share at least one class. Every row selected by the final scope
and classifier mask must retain positive teacher mass on that shared support;
a valid full teacher distribution with all its mass outside that support cannot
define a target. Such selected rows raise an error. Excluded rows contribute zero.

`clf_distil_type="hard"` takes `argmax(teacher_labels)` and applies sparse
categorical cross-entropy. `"soft"` applies KL divergence in the
teacher-to-student direction. Its positive `clf_distil_temperature` defaults to
`1.0`; values above one soften both distributions, and values below one sharpen
them. The KL term is scaled by the temperature squared. Hard KD ignores this
temperature. Teacher temperature normalization happens in log space on retained
support, preserving exact zeros and tiny positive probabilities before newly
added classes are padded with zero target mass. Python role coefficients and
temperatures are constructed in the policy's variable dtype, preserving float64
precision; scalar Tensor coefficients retain their supplied values before casting.
The scalar optimization objective adds
`clf_distil_loss_coef * clf_distil_loss` to the existing diffusion and classifier
terms. The coefficient defaults to `0.0`; at zero, classifier distillation loss and
its metrics are disabled. Teacher mapping remains enabled only when a
classifier regularizer independently requests it.

Soft KD evaluates student `log_softmax(logits / temperature)` from the same
network pass that produced the probabilities. It never clips student
probabilities to reconstruct saturated logits. All student classes participate
in normalization; newly added classes have exactly zero teacher target mass.
Auxiliary heads retain their mean-probability mixture, computed in log space
before temperature scaling. Raw `DiTClassifier`, `DiTEncoderDecoderClassifier`,
and `UNetClassifier` expose this additive metadata through `return_logits=True`.
Normal probability returns and checkpoint weights are unchanged. Custom raw
networks used with soft KD must implement that full-return interface; compilation
rejects probability-only implementations. Direct calls to the KD helper may
omit `student_logits` only for strictly positive probability inputs.

Classifier regularizers read `train_type` and `distil_type` from
`clf_cls_token_regularizer_kwargs`, falling back to
`cls_token_regularizer_kwargs` for networks such as `UNetClassifier`.
`train_type="normal"` keeps the existing ground-truth cross-entropy,
`"distil"` uses the selected hard/soft teacher loss, and `"both"` averages the
two losses before applying the existing `ctr_loss_coef`.

When both class and distillation tokens are active, `DiffusionClassifier`
reports `clf_distil_loss` plus three classification accuracies:
`cls_token_accuracy`, `clf_distil_acc`, and `total_accuracy` for their
`clf_acc_coef`/`clf_distil_acc_coef` combination. When classifier regularizers are
active, `ctr_acc_coef` also adds their averaged probabilities to
`total_accuracy`. A distillation token paired
with global average pooling reports `avg_pooling_accuracy` for the ordinary
head instead. Without a distillation token, the existing
`classifier_accuracy` name and output behavior remain unchanged. The raw
classifier always leaves `classes` and `distil_classes` independent; only the
wrapper forms the coefficient-weighted prediction used by `total_accuracy`.
This overall prediction uses the same coefficients for every example, independently
of true labels and replay provenance. KD scope masks select loss eligibility and
scoped head diagnostics; they never choose components of the overall prediction.
`EnsembleAccuracy` accepts the same `clf_acc_coef`, `clf_distil_acc_coef`, and
`ctr_acc_coef` values and applies them at every ensembled timestep.

`DiffusionModel` owns noise-teacher attachment, mapped noise prediction/mask,
and the noise-distillation loss. `DiffusionClassifier` adds the classifier
teacher slot, ordinary Keras fitting/fine-tuning masks, dynamic classifier-head
metadata, classifier-teacher probabilities, and hard/soft classifier losses and
their trackers; V2 inherits this separation.
`DiffusionClassifierV2` assigns the effective distillation token and its softmax
head to the classifier variable group and applies distillation only in the
discriminator train/test phase. Its generator map returns the ordinary seven
diffusion tensors, plus the noise-teacher prediction/mask when noise
distillation is active. Its discriminator map returns
`(t, x_t, null_labels, classes, x0, teacher_labels)`. A native classifier teacher
and student see the same clean or bounded-noise phase-specific input; an
ordinary image-only teacher receives the clean `x0` field.

## Split generator/discriminator training

`DiffusionClassifierV2` owns two optimizer instances. Shared-variable selectors
use these IDs:

| Selector | Meaning |
| --- | --- |
| `clf_vars_embedding_ids=0` | Patch embedder |
| `1` | Time embedder |
| `2` | Label embedder |
| `3` | Main depth-0 label regularizer |
| `4` | Shared main class token when present |
| `clf_vars_noise_part_ids=-1` | Final main-network depth |
| positive/other negative depth IDs | Absolute/relative main depths as detailed in `__init__` |

All classifier stages, its token/regularizer, and its final head are always in
the classifier group. Remaining variables form the generator group.

```python
from diffusion.models.wrapper.diffusion_classifier_v2 import DiffusionClassifierV2


model = DiffusionClassifierV2(
    network=network, 
    clf_vars_embedding_ids=[1, 2], 
    clf_vars_noise_part_ids=[-1], 
    clf_train_noisified_max_timesteps=250 
)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")

gen_history = model.fit_generator(x=dataset, epochs=5)
clf_history = model.fit_discriminator(x=dataset, epochs=5)
merged = model.merge_result_dicts((gen_history.history, clf_history.history))

# Alternatively, run both phases and merge their histories in one call.
merged = model.fit(
    gen_kwargs={"x": dataset, "epochs": 5}, 
    clf_kwargs={"x": dataset, "epochs": 5} 
)
```

`merge_result_dicts(dicts, names=("generator", "discriminator"))` combines
history or evaluation dictionaries. Unique metric names stay unchanged; names
shared by multiple dictionaries receive their phase prefix, such as
`generator_loss` and `discriminator_loss`. The returned dictionary is a shallow
merge: input dictionaries are unchanged, and their values are reused.
Provide one name per input, including any `None` entries, which are skipped.
Custom names support additional phases even when a metric appears in only
some phases. Ambiguous final metric names raise `ValueError` instead of
overwriting results.

When a classifier noising maximum is `None`, that phase uses clean images and
timestep 0. A numeric maximum draws noise below that exclusive bound.
`fit_generator_progressively` and `fit_discriminator_progressively` accept the
same progressive arguments as `fit_progressively`. Use `evaluate_generator`,
`evaluate_discriminator`, or `evaluate(eval_both=True, ...)` for phase-specific
metrics. `clf_vars_names` and `gen_vars_names` expose the resolved variable split
after compilation.

## Encoder-decoder training

`DiffusionModel` provides the complete training pipeline for
`DiTEncoderDecoder`. Its raw-network call is `(x_t, t, labels)`; the
`DiTEncoderDecoder` three-input adapter reuses `x_t` as the decoder image. The
same convention is therefore used by training, evaluation, classifier-free
guidance, and reverse sampling, while sampled `noise` remains only the loss
target. Active-resolution resizing and the configured noise, image, KL, and
regularizer loss coefficients work exactly as in the base wrapper. Decoder
blocks at depths 1..N cross-attend to the final encoder feature. Decoder depth
0 has no context-attention block and uses the condition plus decoder image. The
raw model's `full_return=True` output is the usual five-item
`DiffusionTransformer` tuple, so auxiliary encoder regularizers and latent
statistics remain available to inherited wrapper logic.

```python
from diffusion import DiTEncoderDecoder, DiffusionModel


network = DiTEncoderDecoder(
    encoder_kwargs={"image_size": 32, "channels": 3, "depth": 4}, 
    decoder_kwargs={"depth": 2, "use_unpatchify": True} 
)
model = DiffusionModel(network=network, use_ema=True)
model.compile(optimizer="adam", loss="mse")
model.fit(dataset, epochs=10)
images = model.sample(labels=[1, 2], steps=50)
```

Configure the attached decoder with `use_unpatchify=True` and an output image
shape matching the sampled noise target. Token-only decoder output is valid for
direct raw-network calls but not for this wrapper's image-shaped diffusion
target pipeline, even when `image_loss_coef=0`. Progressive
depth changes grow the encoder by default; targeted `{"decoder": ...}` specs
grow the attached decoder. Resolution changes are synchronized across both branches and a
non-None value must be divisible by both patch sizes.
Raw-network `get_config`/`from_config` reconstructs the configured topology;
learned raw and EMA values still come from checkpoint weights. For continual
checkpoints, use the project-level paired `config.yaml` and `.weights.h5`: the
factory passes raw `num_classes=None`, the wrapper uses saved `seen_classes` to
restore the grown topology, and only then are weights loaded. Project training
requires a `Config` for dynamic diffusion weight saving and writes this paired
config even when `save_config_=False`.

The wrapper also accepts a standalone, context-free `DiTDecoder` when it uses
`decoder_separate_cond=True`, `shift_inputs=False`, and no encoder-feature
aggregation mappings. In that mode the wrapper supplies empty encoder context,
so the decoder uses its own time/label embeddings and decoder blocks fall back
to self-attention.

A standalone `DiTDecoder` or composed `DiTEncoderDecoder` needs a KL-enabled
flatten bottleneck and decoder routes that work from the sampled boundary
to use the inherited VAE sampler.

For `DiTEncoderDecoderClassifier`, prefer `DiffusionClassifier` for ordinary
training, evaluation, and sampling. Its three-input wrapper calls automatically
reuse the noisy encoder image as the decoder input and receive the standard
`{"noises": ..., "classes": ...}` result. The raw network also accepts
`(encoder_images, timesteps, labels, decoder_images)` for explicit teacher
forcing, but no stock wrapper supplies that fourth tensor; use a direct call or
a custom `train_step` for that workflow. The classifier's inherited depth API
grows its encoder and classifier branches, and targeted `{"decoder": ...}`
specs can also grow the attached decoder.

## Previous-task and current-task teachers

A previous completed student snapshot can teach both heads while two separate
current-task specialists teach classification and noise. Attach a classifier
such as the model returned by `get_model(..., conv_base_name="EfficientNetV2L")`
and a native noise DiT through the dedicated slots:

```python
model = DiffusionClassifier(
    network=student_network, 
    classifier_teacher_network=classifier_teacher, 
    noise_teacher_network=noise_teacher, 
    trainable_teacher=True, 
    teacher_dynamic_classes=True, 
    defer_teacher=True, 
    clf_distil_loss_coef=1.0, 
    noise_distil_loss_coef=1.0
)
model.compile(optimizer="adam", loss="mse")
model.compile_teacher(
    teacher_name="classifier", 
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)
model.compile_teacher(teacher_name="noise", optimizer="adam", loss="mse")
model.fit_teacher(trainset, teacher_name="classifier", epochs=1)
model.fit_teacher(trainset, teacher_name="noise", epochs=1)
model.fit(trainset, epochs=1)
```

For continuing class growth, construct the native student and noise networks
with `num_classes=None` and keep `teacher_dynamic_classes=True` so the ordinary
classifier head grows too. For an offline fixed-class experiment, fixed native
class counts and `teacher_dynamic_classes=False` are sufficient.

Each `fit_teacher` call updates only its selected teacher. The classifier receives raw images;
the noise teacher uses its diffusion wrapper preprocessing. Both stay frozen
during student fitting. `teacher_network` remains the independent previous
student snapshot and does not change when fitting either specialist.
`current_teacher_network` is the legacy shared current teacher; it cannot be
combined with either specialist. `DiffusionClassifierV2` supports the same
attachments and selectors.

For automatic continual learning with supplied specialists, use
`use_distillation=True` and leave `dual_teacher_distillation=False`. With
`trainable_teacher=True`, the learner trains each attached specialist on the new
task's real data before fitting the student, retaining its optimizer and class
growth state across tasks. `teacher_training="each_task"` is the default;
`"first_task"` trains the specialists only once. Set `trainable_teacher=False`
for already trained frozen specialists. The completed student becomes the
previous teacher after every task independently of the specialists.
Task-boundary recovery restores all attached roles and their training state;
optimizer-step checkpoints are unsupported for this mode.

Alternatively, the continual learner can construct one shared current teacher
automatically alongside the previous snapshot:

```python
continually_learn = dict(
    use_distillation=True, 
    dual_teacher_distillation=True, 
    current_teacher_init="fresh"
)
```

Use these options in the existing continual-learning configuration or learner
call. For each task, the learner trains a separate current-task teacher using
only that task's real training data and validation data. It then freezes this
teacher and distills the student from both it and the previous completed
student snapshot. Task one uses just the current teacher. Replay generation keeps
the existing continual-learning sampler.

`current_teacher_init="fresh"` initializes new weights and a task-local class
head. `"student"` makes an independent copy of the expanded student and trains
that copy only on the new data. Each teacher has its own optimizer. The learner
maps local teacher outputs and conditional labels into the student's vocabulary,
including an existing nonidentity `seen_classes` mapping. Its keys use the
learner's remapped dataset IDs; its values identify output columns. Continual-learning
wrappers must use dynamic vocabularies (`num_classes=None` at construction);
fixed-width teacher examples below also work with manual wrapper training.

Configure the student's ordinary KD coefficients and optional role weights:

```python
model = DiffusionClassifier(
    network=network, 
    defer_teacher=True, 
    noise_distil_loss_coef=1.0, 
    clf_distil_loss_coef=1.0, 
    previous_teacher_noise_loss_weight=1.0, 
    current_teacher_noise_loss_weight=1.0, 
    previous_teacher_clf_loss_weight=1.0, 
    current_teacher_clf_loss_weight=1.0, 
    dual_teacher_scope="task"
)
```

Noise-only `DiffusionModel` uses the noise options; both `DiffusionClassifier`
and `DiffusionClassifierV2` support classification and noise together. Zero
disables an individual teacher's objective. Each enabled teacher contributes
its own normalized loss, multiplied by its role weight; their sum is multiplied
by the existing corresponding KD coefficient. Classification supports both hard
and soft targets. The teachers' distributions are not averaged together.

The default `dual_teacher_scope="task"` applies each teacher only to examples
from its taught classes, using original class IDs before CFG label dropout.
Use `"all"` to apply both teachers to every example, subject to the existing
noise CFG mask and classifier row masks. The existing `clf_distil_scope` applies
to the previous teacher: `"replay_only"` preserves its replay restriction while
the current teacher can still teach new real examples. Without a null condition, noise KD
also excludes class conditions outside the teacher's vocabulary. Each classification
teacher targets only its taught output columns; the student's softmax denominator
still includes all student classes. With task scope, the previous teacher contributes
only when old-task real or replay examples are present. Use all scope to obtain
previous-teacher classification targets on current-only data.

For manually trained teachers, keep the previous teacher in `teacher_network`
and attach the current teacher before fitting:

```python
model.set_current_teacher_network(
    current_teacher, 
    class_ids=[4, 5],       # teacher columns 0, 1 map to student classes 4, 5
    task_class_ids=[4, 5]  # classes taught by this teacher
)
```

A full-width current teacher can use `class_ids=list(range(6))` with
`task_class_ids=[4, 5]`. Ordinary callable image classifiers receive clean images
in both teacher roles, without `predict_class`; an image-only classifier in
a shared slot needs its role's noise weight set to zero when noise KD is enabled.
A dedicated classifier specialist imposes no restriction on the independent
noise specialist. Callable epsilon
teachers use `teacher_noise_input_type` as documented above.

Use `set_noise_teacher_network(...)` on any diffusion wrapper.
`DiffusionClassifier` and V2 additionally expose `set_classifier_teacher_network(...)`.
Both setters accept optional `class_ids` and `task_class_ids` arguments for
manual specialist attachment. Their mappings are
independent: classifier columns and noise conditions can follow different class
orders. The current role's classification weight applies to the classifier
specialist and its noise weight applies to the noise specialist.

The shared current teacher created by `dual_teacher_distillation=True` is
discarded after each task and rebuilt after task-boundary recovery.
Task-boundary checkpoints are supported; optimizer-step/mid-task checkpoints
are rejected in this mode. This automatic shared-teacher mode cannot be combined
with `trainable_teacher=True` or supplied specialists. Supplied specialists use
the persistent selected-teacher lifecycle described above.

## Teacher construction and weight-only reload

For native HDF5 weight-only reload, reconstruct the same teacher attachment
topology used when saving. Passing `teacher_network` to the wrapper constructor
and attaching it later with `set_teacher_network` can produce different Keras
weighted-layer layouts. A constructor-attached teacher therefore needs the same
constructor attachment when reloading; this is an existing compatibility boundary.
Common task-boundary recovery has its own authenticated reconstruction protocol.
