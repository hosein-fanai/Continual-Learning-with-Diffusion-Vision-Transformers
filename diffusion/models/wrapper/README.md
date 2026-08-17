# Diffusion model wrappers

This directory contains the stateful training and sampling layer around the raw
networks in [`diffusion.models.transformer`](../transformer/README.md).

The division of responsibility is intentional:

| Transformer package | Wrapper package |
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
  `DiTClassifier`.
- `DiffusionClassifierV2`: alternating generator/discriminator variant with two
  optimizer states and explicit variable ownership.
- `DiffusionEncoderDecoderModel`: experimental teacher-forced training override
  for `DiTEncoderDecoder`.

The package aliases `NetworkName` (`"raw" | "ema"`), `TrainType`
(`"cond" | "uncond"`), and `ClusteringType` (`"uniform" | "log_snr"`).

## Data and label conventions

Keras `fit`/`evaluate` datasets normally yield:

- images: float `tf.Tensor` `[B, H, W, channels]`, normally normalized to
  `[-1, 1]`;
- classes: zero-based integer `tf.Tensor` `[B]` in `0..num_classes-1`.

When classifier-free guidance is enabled, `prep_inputs` shifts dataset classes
by one. Network label 0 is null and real network labels are 1..`num_classes`.
Direct raw-network calls and `sample(labels=...)` expect these already-shifted
network IDs; wrapper training data remains zero-based.

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
`use_ema=False`.

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
- `kl_train_type` and `ctr_train_type`: choose conditional or unconditional
  branch values for auxiliary losses;
- `*_noisified_min_timesteps` / `*_noisified_max_timesteps`: half-open train and
  test ranges `[minimum, maximum)`;
- `resize_method`, `resize_antialias`: active-resolution input resizing;
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
patch size; `None` restores the base image size.

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
with `t <= int(mask_t_percentage / 100 * timesteps)`. Set either switch false to
remove that selection criterion. `clf_train_type="uncond"` requires a numeric
`train_cfg_scale` because it needs the explicit unconditional pass.

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
| `clf_vars_noise_part_ids=-1` | Final main-transformer depth |
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

## Experimental encoder/decoder wrapper

`DiffusionEncoderDecoderModel.train_step` forwards `(x_t, t, labels, noise)` to
its raw network for teacher forcing. The underlying encoder/decoder classes
retain documented legacy mismatches between structured returns and call
signatures, so this path should be treated as an extension point until tested
and integrated for the desired configuration.
