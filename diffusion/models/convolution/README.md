# Convolutional diffusion models

This directory provides `UNet` and `UNetClassifier`. They are convolutional raw
networks with the same depth, feature-return, serialization, and progressive
training conventions as the transformer models. Diffusion wrappers still own
noising, losses, EMA state, optimization, and sampling.

Use the public imports:

```python
from diffusion import UNet, UNetClassifier
# Equivalent raw-model imports:
from diffusion.models.convolution import UNet, UNetClassifier
```

Compile and fit a wrapper, not the bare raw network, for diffusion training.

## U-Net hierarchy and depth contract

The standard hierarchy is:

```text
projected image + broadcast time/label condition       depth 0
  -> residual stack -> downsample                      one pair per width
  -> bottleneck residual stack
  -> upsample -> encoder skip concat -> residual stack one group per width
  -> zero-initialized noise projection
```

`widths` sets the encoder levels and decoder widths. `block_depth` is the
number of residual blocks inside each encoder/decoder stack;
`bottleneck_depth` applies to the central stack. Each operation above is a
separate tracked `LayerDict`, so `layers_dicts[i]` produces feature depth
`i + 1`. Encoder residual-stack outputs are the standard skip sources. Decoder
features are resized to the exact source shape before concatenation, including
at odd or progressively changed resolutions.

Depth `0` is the 1x1-projected image concatenated with spatially broadcast
timestep and label embeddings. `encode(...)` returns
`(x, condition, features_list, regs_list, z_vals)`, where feature and
regularizer index equals depth. `call(..., full_return=True)` returns:

```python
predicted_noise, condition, features_list, regs_list, z_vals = network(
    (x_t, timesteps, labels), 
    full_return=True, 
    training=False, 
)
assert len(features_list) == len(regs_list) == network.depth + 1
```

`features_list` contains real intermediate tensors, `regs_list` contains
configured auxiliary predictions or `None`, and `z_vals` is `(mean, log_var)`
for a KL flatten stage or `(None, None)` otherwise. `layers_dicts`,
`connection_ids_dict`, `reshaper_ids_dict`, and the depth-indexed lists provide
the same inspection points expected by the wrappers.

## Constructing and calling `UNet`

```python
import tensorflow as tf

from diffusion import UNet

network = UNet(
    num_classes=10, 
    use_cfg=True, 
    timesteps=1_000, 
    image_size=32, 
    channels=3, 
    widths=(32, 64, 96), 
    block_depth=2, 
    bottleneck_width=128, 
    bottleneck_depth=2, 
    downsampling_method="avg_pooling", 
    upsampling_method="interpolate", 
)

x_t = tf.random.normal([8, 32, 32, 3])
t = tf.range(8, dtype=tf.int32)
labels = tf.range(8, dtype=tf.int32)
noise = network((x_t, t, labels), training=False)
# noise: [8, 32, 32, 3]
```

With classifier-free guidance (CFG), direct calls use label `0` for the null
condition and `1..num_classes` for real classes. Wrappers accept zero-based
dataset classes and shift them while training. `sample(labels=...)` and
`sample_vae(labels=...)` consume the already-shifted network IDs.

Downsampling methods are `avg_pooling`, `max_pooling`, and `cnn_stride`.
Upsampling methods are `interpolate`, `cnn_interpolate`, and `cnn_transpose`;
`upsampling_interpolation` selects the interpolation algorithm where relevant.
The reusable implementations are documented in the
[convolution layer guide](../../layers/convolution/README.md).

## Diffusion and variational training

An ordinary `UNet` enables encoder skips by default. Enable the hierarchical
variational bottleneck with only its configuration:

```python
import tensorflow as tf

from diffusion import DiffusionModel, UNet

vae_network = UNet(
    num_classes=10, 
    image_size=32, 
    channels=3, 
    widths=(32, 64, 96), 
    reshaper_kwargs={"add_kl": True, "latent_dim_ratio": 0.5}, 
)
model = DiffusionModel(
    network=vae_network, 
    kl_loss_coef=1e-4, 
    test_steps=50, 
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4), 
    loss=tf.keras.losses.MeanSquaredError(), 
)

# Dataset elements are (images [B,H,W,C] in [-1,1], zero-based labels [B]).
# model.fit(train_dataset, validation_data=validation_dataset, epochs=20)

images = model.sample_vae(
    network_name="ema", 
    labels=[1, 2, 3], 
    seed=7, 
)
# images: [3, 32, 32, 3] in [0,1]
```

`add_kl=True` computes and inserts the consecutive flatten/unflatten depths;
callers do not need to calculate `reshaper_ids_dict`. With
`use_skip_connections=None` (the default), any reshaped bottleneck disables
skips automatically so a decoded latent cannot bypass the encoder. Explicitly
combining a bottleneck with `use_skip_connections=True` is rejected.

Normal `DiffusionModel` training retains the diffusion noise target and adds
the configured KL loss. Setting `swap_noise_image=True` selects the wrapper's
alternate image/noise target and also routes `model.sample(...)` to
`sample_vae(...)`. A supplied latent `z` must match the configured latent
width; omitting it draws a standard-normal latent.

## Progressive resolution and depth

`set_current_resolution(size)` updates the active square resolution. U-Net
decoder stages align their tensors at runtime, so the same weights support
progressive resolution and odd intermediate sizes.

`add_depths(...)` appends shape-preserving stages after the base decoder:

```python
report = network.add_depths("convolution_block")
# {"network": {"before": ..., "added": 1, "after": ...}}

network.add_depths([
    "convolution_block",
    ("convolution_block", "cls_token_regularizer"),
])
```

A string adds one stage, a tuple/set/mapping combines components in one stage,
and an outer list adds separate stages. Supported components are
`convolution_block` (also `residual_block`) and `cls_token_regularizer`;
`vision_transformer_block` remains a compatibility alias for a residual stack.
The wrapper can grow these stages during training:

```python
history = model.fit_progressively(
    "depths_only", 
    depths=["convolution_block", "convolution_block"], 
    final_epochs=1, 
    x=train_dataset, 
)
```

Each depth is appended after its stage, then trained by the following or final
stage. Raw and EMA copies grow together and the wrapper registers the new
variables with its optimizer.

## `UNetClassifier`

`UNetClassifier` inherits the complete denoising hierarchy and adds a small
feature-based classifier. `feature_aggregation_ids_dict` maps classifier depth
IDs to inherited U-Net feature depths; depth `0` is the embedded input,
positive IDs are main stages, and negative IDs are relative to the final main
depth. Selected feature maps are spatially aligned, projected, processed by
`clf_depth` residual stages, globally pooled, and classified with a float32
softmax head. `aggregate_from_noises=True` instead classifies the predicted
noise. `classifier_only_distil_token=True` adds a wrapper-compatible tracked
placeholder and an independent second softmax head over the same final pooled
feature; it does not add a sequence token to the convolutional network.

Its normal call returns a wrapper-compatible mapping:

```python
from diffusion import UNetClassifier

network = UNetClassifier(
    num_classes=10, 
    image_size=32, 
    channels=3, 
    widths=(32, 64, 96), 
    feature_aggregation_ids_dict={1: (1, -1)}, 
    clf_dim=96, 
    clf_depth=2, 
    clf_block_depth=1, 
)
outputs = network((x_t, t, labels), training=False)
# outputs["noises"]: [B,32,32,3]
# outputs["classes"]: [B,10]
# outputs["distil_classes"]: [B,10] when classifier_only_distil_token=True
```

Selected main features are spatially aligned and normalized to `clf_dim`, so
relative selectors remain shape-stable when the main network grows.
`full_return=True` additionally provides the main `cond`, `features_list`,
`regs_list`, and `z_vals`, plus their `clf_` counterparts. Classifier-side KL
statistics can be enabled with
`clf_reshaper_kwargs={"add_kl": True, "latent_dim_ratio": ...}`.
`predict_noise(...)` runs only the denoiser, while `predict_class(...)` executes
only the main depths required by the selected classifier features. Its ordinary
result remains `classes`; `full_return=True` appends `distil_classes` as the
sixth item when enabled. The wrappers own teacher loss and any weighted
combination of these two independent predictions.

The inherited main U-Net can also use `reshaper_kwargs={"add_kl": True, ...}`.
In that configuration, `DiffusionClassifier.sample_vae(...)` resumes only the
noise branch after the latent boundary; the classifier branch is not needed
while decoding.

### Training with both classifier wrappers

Use one raw model per wrapper:

```python
import tensorflow as tf

from diffusion import (
    DiffusionClassifier,
    DiffusionClassifierV2,
    UNetClassifier,
)

joint = DiffusionClassifier(
    network=UNetClassifier(image_size=32, channels=3, num_classes=10),
    clf_loss_coef=8.6e-3,
)
joint.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")
# joint.fit(train_dataset, epochs=20)

split = DiffusionClassifierV2(
    network=UNetClassifier(image_size=32, channels=3, num_classes=10),
)
split.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")
# split.fit_generator(x=train_dataset, epochs=10)
# split.fit_discriminator(x=train_dataset, epochs=10)
# Or: split.fit(gen_kwargs={"x": train_dataset, "epochs": 10},
#               clf_kwargs={"x": train_dataset, "epochs": 10})
```

Both wrappers consume datasets of `(clean_images, zero_based_class_labels)`.
`DiffusionClassifier` optimizes denoising and classification jointly;
`DiffusionClassifierV2` maintains independent generator and classifier
optimizers and exposes the phase-specific fit methods shown above.

Classifier progression accepts ordinary main-network specifications or a
targeted mapping:

```python
network.add_depths({
    "network": "convolution_block",
    "classifier": ["convolution_block", "convolution_block"],
})
```

The targeted form reports `network` and `classifier` growth separately. A
classifier outer list adds separate fixed-width residual stages; a tuple,
set, or mapping describes one classifier stage.

## Configuration and serialization

Constructor values and appended depth specifications are included in
`get_config()`. `UNet.from_config(...)` and `UNetClassifier.from_config(...)`
rebuild the topology used for EMA cloning; learned state still transfers
through normal Keras weight/model saving or `set_weights`.

For a dynamically grown U-Net, each `add_class()` also updates the saved
`num_classes` to the current width and expands every enabled main/classifier
auxiliary softmax plus the final classifier and distillation heads. Continual
restoration additionally requires the wrapper's persisted zero-based
`seen_classes` mapping; it restores label identity and enables later growth.

`cls_token_regularizer_ids` and classifier `clf_` names are retained for
compatibility with the existing wrappers even though the convolutional model
has spatial features rather than transformer class tokens.
`cls_token_regularizer_kwargs` also accepts `train_type` (`normal`, `distil`,
or `both`) and `distil_type` (`hard` or `soft`); their defaults are `normal`
and `hard`.
