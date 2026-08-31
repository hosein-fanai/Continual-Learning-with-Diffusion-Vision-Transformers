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
  sampling, VAE bottleneck sampling, and progressive curricula.
- `DiffusionClassifier`: joint denoising/classification wrapper for
  `DiTClassifier`, `DiTEncoderDecoderClassifier`, and `UNetClassifier`.
- `DiffusionClassifierV2`: alternating generator/discriminator variant with two
  optimizer states and explicit variable ownership.

The package aliases `NetworkName` (`"raw" | "ema"`), `TrainType`
(`"cond" | "uncond"`), and `ClusteringType` (`"uniform" | "log_snr"`).

## Data and label conventions

Keras `fit`/`evaluate` datasets normally yield:

- images: float `tf.Tensor` `[B, H, W, channels]`, normally normalized to
  `[-1, 1]`;
- classes: integer `tf.Tensor` `[B]`; fixed-width models use zero-based IDs in
  `0..num_classes-1`, while dynamic models map observed dataset IDs.

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
    depth=4, 
)

model = DiffusionModel(
    network=network, 
    use_ema=True, 
    scheduler_name="clipped_cosine", 
    p_uncond=0.1, 
    test_cfg_scale=4.0, 
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4), 
    loss="mse", 
)

# dataset yields (images [B,28,28,1], classes [B])
history = model.fit(dataset, epochs=10, validation_data=validation_dataset)

# Sampling labels are network IDs: 1 and 2 correspond to dataset classes 0 and 1.
images = model.sample(labels=[1, 2], steps=50, eta=0.0)
```

`sample` returns postprocessed float images `[B,H,W,C]` in `[0,1]`. Set
`return_x_ts=True` and/or `return_x0s=True` to receive a list containing final
images plus step-wise NumPy trajectories. `eta=0` is deterministic DDIM;
`0 < eta < 1` is stochastic DDIM; `eta=1` is DDPM-equivalent only when using
the full consecutive schedule.

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
- `noise_loss_coef`, `image_loss_coef`, `kl_loss_coef`, `ctr_loss_coef`: loss
  multipliers; a zero auxiliary coefficient disables that objective;
- `show_separate_noise_losses=False`: keep the usual `noise_loss` progress
  metric. When true, rename it to `total_noise_loss` and also show
  `cond_noise_loss` for non-null rows and `uncond_noise_loss` for null rows.
  The two split metrics are reporting-only and custom callbacks should monitor
  the renamed total metric;
- `kl_train_type` and `ctr_train_type`: choose conditional or unconditional
  branch values for auxiliary losses;
- `*_noisified_min_timesteps` / `*_noisified_max_timesteps`: half-open train and
  test ranges `[minimum, maximum)`; a base-wrapper maximum of `None` or `-1`
  uses the network's full timestep count;
- `resize_method`, `resize_antialias`: active-resolution input resizing;
- `map_preprocess=False`: keep `prep_inputs` in the train/test step. When true,
  `fit`, `evaluate`, validation, and progressive stages map each
  `tf.data.Dataset` through `prep_inputs_map` in the CPU input pipeline;
- `swap_noise_image=True`: trains against `x_t` and makes `sample` use the
  network's KL-enabled flatten bottleneck via `sample_vae`.

`compile(loss=..., **kwargs)` forwards `optimizer`, `run_eagerly`,
`steps_per_execution`, supported `jit_compile`, metrics, weighted metrics, and
loss weights to Keras. `fit(**kwargs)` and `evaluate(**kwargs)` accept the normal
Keras batch/epoch/callback/validation/step arguments documented in their
docstrings. `summary(**kwargs)` forwards Keras summary display options to the
raw network.

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
    training=False, 
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
            "depth": "vision_transformer_block", 
        }, 
    ], 
    stage_epochs=2, 
    final_epochs=1, 
    pacing_type="fixed", 
    x=dataset, 
    validation_data=validation_dataset, 
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
model.fit_progressively(
    "depths_only", 
    depths=["vision_transformer_block", "local_mixer"], 
    final_epochs=1, 
    x=dataset, 
)
```

Generated timestep clusters are `uniform` or `log_snr`. Fixed pacing runs every
allocated epoch; plateau pacing uses epoch-wise Keras early stopping or the
project's batch-wise plateau callback. The returned `History` includes a
`progressive_stages` record and the resolved schedules. Timestep bounds and
resolution are restored on exit; completed depth growth remains.

## Joint diffusion classification

Use a `DiTClassifier` with `DiffusionClassifier`:

```python
from diffusion.models.transformer.di_t_classifier import DiTClassifier
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier

network = DiTClassifier(
    depth=4, 
    clf_depth=2, 
    feature_aggregation_ids_dict={1: [-1]}, 
)
model = DiffusionClassifier(
    network=network, 
    clf_train_type="cond", 
    clf_loss_coef=8.6e-3, 
    mask_by_nulls=True, 
    p_uncond=0.1, 
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
select a network explicitly. The historical `netwrok_name` spelling remains a
compatibility alias.

Classifier progressive depth can target both branches:

```python
depths = [{
    "network": "vision_transformer_block", 
    "classifier": {
        "feature_connector": {"ids": [-1]}, 
        "vision_transformer_block": True, 
    }, 
}]
history = model.fit_progressively(
    "depths_only", depths=depths, final_epochs=1, x=dataset
)
```

Classifier-specific progressive names also include `feature_aggregator` and
`cross_attention_aggregator`.

## Distillation training

Distillation-token loss is enabled only when the wrapped classifier has a
distillation token, `teacher_network` is supplied, and
`distil_loss_coef > 0`. Teacher-trained classifier regularizers also enable the
same inherited `map_preprocess` path when `ctr_loss_coef > 0` and their
`train_type` is `distil` or `both`. Pass a
compatible raw classifier, or another wrapper whose `.network` is that
classifier, as the programmatic teacher. The teacher is deliberately not a
YAML/dataclass field because it is a live Keras object. The wrapper unwraps and
freezes the effective raw network, calls it with `training=False`, and stops
gradients through its probabilities.

Continual learning can instead construct the teacher automatically. With
`defer_teacher=True`, task one is allowed to run without a teacher;
`snapshot_teacher_network("raw")` makes an independent frozen copy of the
completed student and `set_teacher_network(...)` activates it for the next
task. The continual API performs both calls before discovering and adding the
new class. A supplied task-one teacher is replaced by the first student
snapshot. Runtime teachers are excluded from wrapper configuration.

```python
teacher = DiTClassifier(
    num_classes=10,
    feature_aggregation_ids_dict={1: [-1]},
)
student = DiTClassifier(
    num_classes=10,
    feature_aggregation_ids_dict={1: [-1]},
    clf_distil_token_type="new_weight",
)
model = DiffusionClassifier(
    network=student,
    teacher_network=teacher,
    distil_type="soft",
    distil_loss_coef=1.0,
    clf_acc_coef=0.5,
    distil_acc_coef=0.5,
    ctr_acc_coef=0.0,
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

During training the teacher receives `(x_t, t, selected_labels)`, where
`selected_labels` is the conditional or unconditional prepared branch selected
by `clf_train_type`. During validation/evaluation it receives the same clean
`x0`, zero timestep, and unconditional label used by the student's established
classification path. If it provides `predict_class`, that method supplies the
target; otherwise its call result is used directly, with `result["classes"]`
selected from mappings. The custom train/test steps consume the prepared tuple
without noising or shifting it a second time. Array inputs, separate `x`/`y`,
validation tuples, and already-prepared datasets are not automatically adapted
by this path. Progressive resolution changes are mirrored to compatible
teachers. For a narrower past teacher, old conditional IDs are retained and a
new-task ID is replaced by the safe zero/null condition before lookup. When continual
growth gives teacher and student different class widths, teacher probabilities
are restricted or zero-padded to the student's current width and normalized
before either loss is computed.

`distil_type="hard"` takes `argmax(teacher_labels)` and applies sparse
categorical cross-entropy. `"soft"` applies KL divergence in the
teacher-to-student direction. Its positive `distil_temperature` defaults to
`1.0`; other values soften teacher and student distributions consistently and
scale the KL term by the temperature squared. The scalar optimization objective adds
`distil_loss_coef * distil_loss` to the existing diffusion and classifier
terms. The coefficient defaults to `0.0`; at zero, distillation-token loss and
its metrics are disabled. Teacher mapping remains enabled only when a
classifier regularizer independently requests it.

Classifier regularizers read `train_type` and `distil_type` from
`clf_cls_token_regularizer_kwargs`, falling back to
`cls_token_regularizer_kwargs` for networks such as `UNetClassifier`.
`train_type="normal"` keeps the existing ground-truth cross-entropy,
`"distil"` uses the selected hard/soft teacher loss, and `"both"` averages the
two losses before applying the existing `ctr_loss_coef`.

When both class and distillation tokens are active, `DiffusionClassifier`
reports `distil_loss` plus three classification accuracies:
`cls_token_accuracy`, `distil_token_accuracy`, and `total_accuracy` for their
`clf_acc_coef`/`distil_acc_coef` combination. When classifier regularizers are
active, `ctr_acc_coef` also adds their averaged probabilities to
`total_accuracy`. A distillation token paired
with global average pooling reports `avg_pooling_accuracy` for the ordinary
head instead. Without a distillation token, the existing
`classifier_accuracy` name and output behavior remain unchanged. The raw
classifier always leaves `classes` and `distil_classes` independent; only the
wrapper forms the coefficient-weighted prediction used by `total_accuracy`.
`EnsembleAccuracy` accepts the same `clf_acc_coef`, `distil_acc_coef`, and
`ctr_acc_coef` values and applies them at every ensembled timestep.

`DiffusionModel` owns only the generic dataset-map switch and
`prep_inputs_map` hook. Teacher calls, hard/soft losses, and distillation
trackers live on `DiffusionClassifier` and are inherited by V2.
`DiffusionClassifierV2` assigns the effective distillation token and its softmax
head to the classifier variable group and applies distillation only in the
discriminator train/test phase. Its generator map returns only the ordinary
seven prepared diffusion tensors and never calls the teacher. Its discriminator
map returns `(t, x_t, null_labels, classes, x0, teacher_labels)`, so the teacher and
student see the same clean or bounded-noise phase-specific input.

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
    clf_train_noisified_max_timesteps=250, 
)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")

gen_history = model.fit_generator(x=dataset, epochs=5)
clf_history = model.fit_discriminator(x=dataset, epochs=5)

# Convenience API: runs both and returns a merged history dictionary.
merged = model.fit(
    gen_kwargs={"x": dataset, "epochs": 5}, 
    clf_kwargs={"x": dataset, "epochs": 5}, 
)
```

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
    decoder_kwargs={"depth": 2, "use_unpatchify": True}, 
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
depth changes grow the encoder only; decoder depth is fixed at raw-network
construction. Resolution changes are synchronized across both branches and a
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

A standalone `DiTDecoder` cannot use `swap_noise_image=True`. A composed
`DiTEncoderDecoder` can use the inherited VAE sampler when its reshaper and
decoder routes satisfy the base wrapper's `sample_vae` constraints.

For `DiTEncoderDecoderClassifier`, prefer `DiffusionClassifier` for ordinary
training, evaluation, and sampling. Its three-input wrapper calls automatically
reuse the noisy encoder image as the decoder input and receive the standard
`{"noises": ..., "classes": ...}` result. The raw network also accepts
`(encoder_images, timesteps, labels, decoder_images)` for explicit teacher
forcing, but no stock wrapper supplies that fourth tensor; use a direct call or
a custom `train_step` for that workflow. The classifier's inherited depth API
grows its encoder and classifier branches, while the attached decoder remains
at its constructor depth.
