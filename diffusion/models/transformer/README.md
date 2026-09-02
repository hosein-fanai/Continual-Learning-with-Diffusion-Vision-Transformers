# Diffusion transformer networks

This directory contains the raw TensorFlow/Keras networks. They turn image,
timestep, and label tensors into noise/image predictions and, for
`DiTClassifier`, class probabilities. They do not create a diffusion schedule,
sample forward noise, optimize losses, maintain EMA weights, or run reverse
sampling. Those responsibilities belong to the sibling
[`wrapper`](../wrapper/README.md) package.

## Public classes

- `DiffusionTransformer`: configurable patch-based denoising transformer.
- `DiTClassifier`: `DiffusionTransformer` plus a feature-routing classifier
  branch and softmax head.
- `DiTDecoder`: experimental shifted/causal decoder specialization.
- `DiTEncoderDecoder`: wrapper-compatible transformer encoder with an adapted
  decoder and optional teacher forcing.
- `DiTEncoderDecoderClassifier`: classifier encoder plus an adapted
  context-aware decoder, with optional teacher forcing.

The package also defines `CondType`, `TokenType`, `IdsType`, and `IdsDictType`
type aliases.

## Tensor interface

The standard transformer call is:

```python
outputs = network((noisy_images, timesteps, labels), training=False)
```

| Input | Type and shape | Meaning |
| --- | --- | --- |
| `noisy_images` | `tf.Tensor`, float, `[B, H, W, channels]` | Model-space image, normally in `[-1, 1]` |
| `timesteps` | `tf.Tensor`, integer, `[B]` | Discrete IDs in `0..timesteps-1` |
| `labels` | `tf.Tensor`, integer, `[B]` | Network label IDs; with CFG, 0 is null and real classes are 1..`num_classes` |

`DiffusionTransformer` returns `[B, H, W, channels]` when
`use_unpatchify=True`, or `[B, tokens, features]` otherwise. With
`full_return=True`, it returns
`(output, cond, features_list, regs_list, z_vals_list)`, where `z_vals_list` is an
ordered list of `(mean, log_variance)` pairs.
`DiTClassifier` instead returns `{"noises": ..., "classes": ...}` and adds both
branches' intermediate values when `full_return=True`. With a distillation
token, it also returns the independent `distil_classes` head. `classes` remains
the class-token/average-pooling result in both training and inference.

`DiTEncoderDecoder` accepts the standard three tensors and reuses
`noisy_images` as the decoder image. A fourth float image tensor supplies an
explicit teacher-forcing input. Its ordinary result is the decoder tensor. Its
`full_return=True` result preserves the standard five-item transformer contract:
`(noises, encoder_cond, encoder_features, encoder_regs, encoder_z_vals_list)`.

`DiTEncoderDecoderClassifier` accepts the same three tensors for wrapper
compatibility. It uses `noisy_images` as both the encoder and decoder image in
that form. A fourth float tensor `[B, decoder_height, decoder_width,
decoder_channels]` selects explicit teacher forcing and must use the same batch
size as the other inputs:

```python
outputs = network(
    (encoder_images, timesteps, labels, decoder_images), 
    training=False, 
)
```

Its ordinary result has the same two keys as `DiTClassifier`, but `"noises"`
is the decoder prediction and `"classes"` is computed from encoder features
(or from the decoder prediction when `aggregate_from_noises=True`). With
`full_return=True`, the result also contains `cond`, `features_list`,
`regs_list`, `z_vals_list`, `clf_cond`, `clf_features_list`, `clf_regs_list`,
`clf_z_vals_list`, `decoder_cond`, `decoder_features_list`, `encoder_cond`, and
`encoder_features_list`. The last two are explicit aliases of `cond` and
`features_list` for decoder-oriented code.

## Class and distillation tokens

`distil_token_type` enables a second optional prefix token and accepts the same
`None`, `"new_weight"`, `"time"`, `"label"`, and `"time_label"` values as
`cls_token_type`. Its `distil_token_freq_dim`, `distil_token_mlp_ratio`, and
`distil_token_pos_merger_type` options mirror the corresponding `cls_token_*`
options. When both tokens exist, every retained token feature is ordered as:

```text
[class token, distillation token, patch tokens...]
```

With only a distillation token it occupies position 0. Local mixers, scalers,
and reshapers preserve all active prefix tokens outside spatial patch-grid
operations. `features_list` retains those prefixes, while the final
`DiffusionTransformer`/`DiTDecoder` denoising output removes them before
unpatchification or token return. These raw denoisers only propagate the
tokens; they do not classify them or expose separate token predictions.

`DiTClassifier` adds the distillation head. With
`classifier_only_distil_token=True`, set `clf_distil_token_type`. With
`classifier_only_distil_token=False`, the classifier uses the main
transformer's `distil_token_type`. The remaining distillation-token shape/merge
options are the inherited `distil_token_*` values.

The ordinary classifier head reads the class token when present. If no class
token exists, or `force_global_avg_pooling=True`, it averages every token except
the distillation token; forced pooling therefore still includes an existing
class token. The distillation head reads the distillation position directly.
Whenever distillation is active, calls expose the primary `"classes"` and
independent `"distil_classes"` distributions in both modes; without it, the
established `"classes"` contract is unchanged. The wrapper combines these
distributions only when computing `total_accuracy`. Dynamic `add_class()`
growth expands both softmax heads.

## Depth and ID conventions

Depth 0 is not a transformer block. It is the patch-embedded input, after any
patch/condition merge and optional prefix tokens. Depths 1 through N are the
outputs of `layers_dicts[0]` through `layers_dicts[N-1]`.

An ID dictionary maps a target depth to source feature depths:

```python
connection_ids_dict = {
    2: [0, 1],       # at depth 2, combine input and depth-1 features
    4: [-2],         # for total depth 4, -2 normalizes to depth 3
}
```

- `None` in a component-ID list expands to every eligible ID. Thus
  `vit_block_ids=[None]` enables a block at every depth.
- Negative IDs normalize as `id + total_depth + 1`: at depth N, `-1` is N and
  `-(N+1)` is 0.
- A routed source must exist when the target executes and tensors merged by a
  connector must have compatible non-merge dimensions.
- In `DiTClassifier`, feature/cross-attention aggregators read main-transformer
  depths, while `clf_connection_ids_dict` and
  `clf_cross_attention_ids_dict` read classifier depths.
- Classifier depth 0 is its first aggregated input. Processing depths are
  1..`clf_depth`; a mandatory terminal connector is stored at
  `clf_depth + 1`. Constructor key `-1` configures this terminal connector.

## Stage contents

A main depth can contain these components, executed in order:

1. feature connector;
2. cross-attention connector;
3. vision-transformer or decoder block;
4. local spatial mixer;
5. downsampler;
6. upsampler;
7. flatten/unflatten reshaper;
8. class-token regularizer.

Each component does only its local job. Connectors select and add/concatenate
existing features. Transformer blocks perform global attention and an FFN.
Local mixers perform depthwise spatial convolution on square patch grids.
Scalers resize token grids. Reshapers create or reverse vector bottlenecks. A
regularizer applies a `num_classes` softmax to a configured flattened token
slice. Setting `mlp_ratio` inside `cls_token_regularizer_kwargs` adds a Dense
hidden projection before that softmax. Its `activation_function` defaults to
`"tanh"` when omitted. The classifier branch uses the same keys in
`clf_cls_token_regularizer_kwargs`. Wrapper training also reads `train_type`
(`normal`, `distil`, or `both`) and `distil_type` (`hard` or `soft`); these
metadata keys do not change the raw network layers.
Components omitted from a depth are identity/no-op paths.

When several KL-enabled flatten stages execute, the network keeps their
posterior statistics separate and the wrapper sums their standard-normal KL
terms. `sample_vae` draws or accepts one latent per flatten stage and injects
them in depth order, so U-shaped skips can originate from stochastic
unflattened features. This is a multiscale factorized VAE, not a conditional
hierarchical VAE: the code does not define learned top-down priors
`p(z_l | z_{l+1})`. A patch grid must be divisible by `2**L` for `L`
factor-two down/up levels.

`DiTClassifier` adds main-feature aggregators and cross-attention aggregators
before the analogous classifier components. Its final extraction behavior,
including distillation-token exclusion from global averaging, is described
above.

## Configuration dictionaries

The following whitelists are enforced. These dictionaries are shared by every
selected depth of that component type; they are not keyed per depth.

| Argument | Allowed keys and important values |
| --- | --- |
| `connection_kwargs`, `cross_attention_kwargs` | `connect_axis: int`; `connect_type`: `"concat"` or `"add"`; `use_layer_norm: bool`; `ln_dim: int or None`; `ln_mlp_ratio: float or None`; `ln_no_adaptation: bool`; `mlp_output_dim: int or None`; `mlp_ratio: float or None`; `mlp_activation_func: Keras activation` |
| `local_mixer_kwargs` | `embed_temperature: float`; `dim: int`; `grid_size: int`; `use_layer_norm: bool`; `ln_mlp_ratio`; `ln_no_adaptation`; `kernel_size: int`; `strides: int`; `depth_multiplier: int`; `use_pointwise: bool`; `pointwise_dim_ratio: int`; `zero_init: bool`; `pos_embed_type`; `pos_interpolation_method`; `pos_merger_type`: `"add"` or `"concat"`; `mlp_ratio`; `mlp_activation_func`; `mlp_output_dim` |
| `downsample_kwargs` | Common embedding/norm/position/MLP keys above plus `scaling_method`: `"avg_pooling"`, `"max_pooling"`, or `"cnn_stride"`; `cnn_dim_ratio: int`; `cnn_kernel_size: int`; `cnn_activation_func` |
| `upsample_kwargs` | Common embedding/norm/position/MLP keys plus `scaling_method`: `"cnn_transpose"`, `"interpolate"`, or `"cnn_interpolate"`; `scaling_interpolation_method`; `cnn_dim_ratio`; `cnn_kernel_size`; `cnn_activation_func` |
| `reshaper_kwargs` | `add_kl: bool`; `latent_dim_ratio: positive float` |
| `cls_token_regularizer_kwargs`, `clf_cls_token_regularizer_kwargs` | `start: int`; `end: int`; these are Python token-slice bounds before flattening; optional `mlp_ratio: positive float or None` adds a hidden Dense layer; optional `activation_function: Keras activation` defaults to `"tanh"`; `train_type`: `"normal"`, `"distil"`, or `"both"`; `distil_type`: `"hard"` or `"soft"` |

`pos_embed_type` is `None` or one of `new_weight`, `1d_sincos`,
`1d_interpolate`, `1d_learned_interpolate`, `2d_sincos`, `2d_interpolate`, and
`2d_learned_interpolate`. Interpolation methods are values supported by the
underlying TensorFlow/Keras resize layer. Spatial patch/mixer/scaler embeddings
support this full set. Discrete time/label `ConditionEmbedding` tables should use
`new_weight` or `1d_sincos`; spatial/interpolation modes have incompatible table
rank for lookup.

Classifier `feature_aggregation_kwargs` and
`cross_attention_aggregation_kwargs` use the connector whitelist.
`clf_connection_kwargs`, `clf_cross_attention_kwargs`,
`clf_local_mixer_kwargs`, `clf_downsample_kwargs`, and `clf_upsample_kwargs`
use their main-branch whitelist. Passing `None` makes most of these inherit the
corresponding main mapping; passing `{}` explicitly uses inferred/layer defaults.
See `DiTClassifier.__init__` for the few `clf_*` values that intentionally do
not inherit.

## Basic denoiser example

```python
import tensorflow as tf

from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer

network = DiffusionTransformer(
    image_size=32, 
    channels=3, 
    patch_size=4, 
    dim=128, 
    cond_dim=128, 
    depth=4, 
    vit_block_ids=[None], 
    connection_ids_dict={2: [0, 1]}, 
    connection_kwargs={
        "connect_type": "concat", 
        "use_layer_norm": True, 
        "mlp_output_dim": 128, 
    }, 
    local_mixer_ids=[3], 
    local_mixer_kwargs={"kernel_size": 3, "zero_init": True}, 
)

images = tf.zeros([8, 32, 32, 3], tf.float32)
times = tf.zeros([8], tf.int32)
labels = tf.ones([8], tf.uint8)  # real class 0 after CFG shifting
predicted_noise = network((images, times, labels), training=False)
```

## Classifier example and `clf_*` defaults

```python
from diffusion.models.transformer.di_t_classifier import DiTClassifier

network = DiTClassifier(
    depth=4, 
    clf_depth=2, 
    feature_aggregation_ids_dict={1: [2, 4]}, 
    feature_aggregation_kwargs={
        "connect_type": "concat", 
        "mlp_output_dim": 64, 
    }, 
    clf_dim=64, 
    clf_dim_forced=True, 
    clf_connection_ids_dict={2: [0, 1], -1: [-1]}, 
)
```

Important initial values are: `clf_depth=1`, `clf_dim=None` (resolved to the
first aggregation width), `clf_dim_forced=False`,
`clf_cond_type="time_label"`, `clf_cls_token_type="new_weight"`,
`classifier_only_distil_token=True`, `clf_distil_token_type=None`,
`clf_vit_block_ids=[None]`, and all classifier mixer/scaler/reshaper/regularizer
ID collections empty. The terminal connection defaults to `{-1: (-1,)}`.
`clf_mha_num_heads`, block MLP ratio, normalization mode, dropout, and most
component kwargs inherit the main branch when passed as `None`; key/value widths
and `clf_ln_mlp_ratio` remain `None` unless explicitly set.

## Encoder-decoder denoiser API

`DiTEncoderDecoder` is a complete `DiffusionTransformer` encoder with an
attached `DiTDecoder`. Put inherited transformer settings in `encoder_kwargs`
and decoder settings in `decoder_kwargs`; flat constructor arguments override
equal values in `encoder_kwargs`. When omitted, the decoder receives the
encoder's class count, CFG mode, timestep count, image/channel/patch settings,
`dim`, and `cond_dim`. Its `encoder_output_grid_size` and
`encoder_output_dim` are inferred from the encoder's final feature. The
composite owns symbolic construction, so a decoder `build` value is ignored.

```python
from diffusion import DiTEncoderDecoder

network = DiTEncoderDecoder(
    encoder_kwargs={
        "image_size": 32, 
        "channels": 3, 
        "patch_size": 4, 
        "dim": 128, 
        "depth": 4, 
    }, 
    decoder_kwargs={
        "depth": 2, 
        "use_decoder_ids": [None], 
        "shift_inputs": False, 
        "use_unpatchify": True, 
    },
)

# Standard denoising: the encoder image also feeds the decoder.
predicted_noise = network((noisy_images, timesteps, labels), training=False)

# Explicit teacher forcing.
predicted_noise = network(
    (noisy_images, timesteps, labels, target_images), 
    training=True, 
)
```

`network.encoder` is a read-only alias for `network`. Consequently
`embed_conditions`, `embed_inputs`, `prepend_cls_token`,
`prepend_distil_token`,
`slice_and_flatten_tokens`, `encode`, `add_depths`, and variable inspection all
belong to the encoder. The composite forces the encoder's `use_unpatchify`
setting to `False` because only the decoder owns the final noise/image head.
`network.depth` and progressive `add_depths` refer only to encoder depth;
`network.decoder.depth` remains fixed after construction.
At decoder depths 1..N, each decoder block cross-attends to the encoder's final
feature by default, so the encoder image supplies actual context. Decoder depth
0 has no attention block and therefore uses only the selected condition and
decoder image.
`set_current_resolution` synchronizes both branches. A non-None size must be
positive and divisible by both branch patch sizes; `None` restores each
branch's configured image size. `build_model` always exposes four symbolic
inputs even though eager three-input calls are valid. Configuration round trips
preserve both nested dictionaries and standard Keras model state.

The raw model may return decoder tokens when `use_unpatchify=False`. The
`DiffusionModel` wrapper instead requires an unpatchified decoder image with
the same shape as its sampled noise target. During latent resume the composite
uses that image as its decoder input, so `sample_vae` supports the same ordered
single- or multiscale-latent contract as the standalone transformer models.

## Encoder-decoder classifier API

`DiTEncoderDecoderClassifier` initializes itself as the complete
`DiTClassifier` encoder and attaches a separately configured `DiTDecoder`.
Place all inherited transformer and classifier options in `encoder_kwargs` and
all decoder-specific options in `decoder_kwargs`. Flat constructor arguments
override equal values in `encoder_kwargs`. The decoder inherits shared class,
timestep, image, channel, patch, token-width, and condition-width settings from
the encoder when they are omitted. `encoder_output_grid_size` and
`encoder_output_dim` are inferred from the encoder's final feature. The
composite manages decoder building, so a `build` entry in `decoder_kwargs` is
ignored. Noise-based classification
(`aggregate_from_noises=True`) requires decoder unpatchification and matching
encoder/decoder image sizes and channel counts; decoder scaling stages must
also restore the encoder's configured input image grid.

The composite defaults decoder `shift_inputs` to `False`, which makes the
three-input wrapper path a direct denoising call. Set it to `True` for
right-shifted teacher forcing; the learned BOS token is shared across the batch.

```python
from diffusion import DiTEncoderDecoderClassifier

network = DiTEncoderDecoderClassifier(
    encoder_kwargs={
        "image_size": 32, 
        "channels": 3, 
        "patch_size": 4, 
        "dim": 128, 
        "depth": 4, 
        "clf_depth": 2, 
        "feature_aggregation_ids_dict": {1: [-1]}, 
    }, 
    decoder_kwargs={
        "depth": 2, 
        "use_decoder_ids": [None], 
        "shift_inputs": False, 
        "use_unpatchify": True, 
    }, 
)

# Wrapper-compatible fallback: encoder images also feed the decoder.
joint = network((noisy_images, timesteps, labels), training=False)

# Explicit teacher forcing.
teacher_forced = network(
    (noisy_images, timesteps, labels, target_images), 
    training=True, 
)
```

`network.encoder` is a read-only alias for `network`; it is not a second model.
The inherited `encode`, embedding/token helpers, `compute_class`,
`set_max_encoder_num`, `get_variables_names`, and classifier routing state all
belong to this encoder/classifier side. The unused encoder `use_unpatchify`
head is disabled, including for noise-based classification, because decoder
noise is the classifier input. `predict_noise` is decoder-aware and
accepts either input form. `predict_class` also accepts either form; the fourth
image is used only for noise-based classification. When
`decoder_separate_cond=True`, the attached decoder embeds its own timestep and
label condition; otherwise it receives the encoder condition. A conditionless
encoder is represented by a zero decoder-context tensor.

`set_current_resolution` updates both encoder and decoder resolutions. A
non-None value must be positive and divisible by both patch sizes; `None`
restores each branch's configured image size. `add_depths` retains the
`DiTClassifier` syntax and grows only the `"network"` and `"classifier"`
branches. Decoder depth and routing are fixed after construction; access
`network.decoder` for decoder-specific inspection. `build_model` exposes four
symbolic inputs even though eager three-input calls remain supported.
`get_config`/`from_config` preserve the nested dictionaries plus Keras `name`,
`trainable`, `dtype`, and `dynamic` state.

## Progressive depth API

`add_depths` appends supported components without replacing existing weights.
Exact main-network names are `feature_connector`,
`cross_attention_connector`, `vision_transformer_block`, `local_mixer`,
`downsampler`, `upsampler`, `reshaper`, and `cls_token_regularizer`.

```python
growth = network.add_depths([
    "vision_transformer_block", 
    {
        "feature_connector": {"ids": [-1]}, 
        "local_mixer": True, 
    }, 
])
```

For `DiTClassifier`, pass a targeted mapping to grow either branch:

```python
growth = classifier_network.add_depths({
    "network": "vision_transformer_block", 
    "classifier": {
        "feature_connector": {"ids": [-1]}, 
        "vision_transformer_block": {"use_decoder": False}, 
    }, 
})
```

Classifier-only additional names are `feature_aggregator` and
`cross_attention_aggregator`. Added sequences must preserve the feature width
expected by the already-created output/classifier head.

Dynamic class growth follows the same serialization rule across transformer
variants: each `add_class()` updates `get_config()["num_classes"]` to the
current width and expands every configured auxiliary softmax plus the final
classifier output, including attached decoder configuration where applicable.
Optional regularizer hidden layers are preserved. That integer records the
checkpoint's final topology. Continual restoration still starts from
`num_classes=None` and replays the wrapper's persisted zero-based
`seen_classes`, which retains the real-label mapping and future growth intent.

## Encoder/decoder status

`DiTEncoderDecoder` adapts the decoder's separate condition and feature inputs
to the standard denoiser return contract. Use `DiffusionModel`
for aligned denoising training, evaluation, and sampling, subject to the
image-output and VAE limits above. Direct four-input calls remain available for
custom teacher-forcing workflows.

`DiTEncoderDecoderClassifier` adapts the decoder's separate condition and
feature-list inputs and is the supported choice when classification and the
standard three-input wrapper contract are required. Use
`DiffusionClassifier` for ordinary training and sampling; use a direct
four-input call or a custom training step for teacher forcing.

The standalone `DiTDecoder` uses
`decoder((images, times, labels), encoder_cond, encoder_features_list)` and its
symbolic builder exposes those values as five inputs. Eager calls accept the
complete depth-indexed encoder feature list; the symbolic fifth input is one
final feature tensor, wrapped internally as a one-item list. Separate
conditions, class tokens, shifted batches, and the image output head are
supported. A changed active resolution must be positive and divisible by the
decoder patch size, and `None` restores its configured image size. Its
encoder-feature and cross-attention aggregation mappings remain reserved rather
than materialized; each decoder block still cross-attends to the final encoder
feature as its default values tensor. Depth 0 has no block and is
condition/decoder-image only. Encoder-style blocks selected with
`use_decoder_ids=[]` receive the same context without a causal mask.
`get_config`/`from_config` preserve standard Keras `name`, `trainable`, `dtype`,
and `dynamic` state in addition to decoder options.
For context-free diffusion replay, `DiffusionModel` can supply
empty encoder context when `decoder_separate_cond=True`, `shift_inputs=False`,
and both encoder aggregation mappings are empty.

The compositions above isolate callers from the decoder's structured context
arguments; the remaining standalone decoder limitations still apply to their
selected decoder configurations.
