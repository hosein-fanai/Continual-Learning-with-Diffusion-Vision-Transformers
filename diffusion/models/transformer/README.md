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
- `DiTEncoderDecoder`: experimental encoder/decoder composition.
- `DiTEncoderDecoderClassifier`: compatibility multiple-inheritance marker; it
  does not currently initialize a classifier branch itself.

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
`(output, cond, features_list, regs_list, (z_mean, z_log_var))`.
`DiTClassifier` instead returns `{"noises": ..., "classes": ...}` and adds both
branches' intermediate values when `full_return=True`.

## Depth and ID conventions

Depth 0 is not a transformer block. It is the patch-embedded input, after any
patch/condition merge and optional prefix token. Depths 1 through N are the
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
slice. Components omitted from a depth are identity/no-op paths.

`DiTClassifier` adds main-feature aggregators and cross-attention aggregators
before the analogous classifier components. Its final extractor uses the first
token when a class token exists, otherwise global average pooling; setting
`force_global_avg_pooling=True` always chooses averaging.

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
| `cls_token_regularizer_kwargs` | `start: int`; `end: int`; these are Python token-slice bounds before flattening |

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
`clf_vit_block_ids=[None]`, and all classifier mixer/scaler/reshaper/regularizer
ID collections empty. The terminal connection defaults to `{-1: (-1,)}`.
`clf_mha_num_heads`, block MLP ratio, normalization mode, dropout, and most
component kwargs inherit the main branch when passed as `None`; key/value widths
and `clf_ln_mlp_ratio` remain `None` unless explicitly set.

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

## Experimental encoder/decoder path

`DiTDecoder`, `DiTEncoderDecoder`, and `DiTEncoderDecoderClassifier` preserve an
older structured decoder API. Their module docstrings identify current
integration limitations, including saved-but-unmaterialized aggregation options
and mismatched structured call outputs. Treat them as extension points and test
a configuration end to end before using the wrapper.
