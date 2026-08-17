# Token manipulation layers

These layers move between a flattened transformer sequence and a local square
grid. They let raw diffusion transformers add convolutional locality or change
spatial resolution while retaining the `[batch, tokens, channels]` interface.

Every call takes `(x, cond)`. `x` is floating `[B,T,D]`; `cond` is floating
`[B,C]` when adaptive normalization is enabled. Without a class token, `T` must
be a perfect square. With `circumvent_cls_token=True`, token 0 is preserved
outside the spatial operation and `T - 1` must be square.

## `Downsample`

```python
from diffusion.layers.manipulation.downsample import Downsample

down = Downsample(
    dim=64, 
    grid_size=8, 
    scaling_method="avg_pooling", 
    strides=2, 
    pos_embed_type=None, 
    name="downsample_1", 
)
y = down((x, condition), training=True)  # [B,64,64] -> [B,16,64]
```

Allowed scaling modes are:

| Mode | Spatial operation | Pre-position output channels |
|---|---|---|
| `"avg_pooling"` | 2x2 average pool | `dim` |
| `"max_pooling"` | 2x2 max pool | `dim` |
| `"cnn_stride"` | learned `cnn_kernel_size` convolution | `dim * cnn_dim_ratio` |

With `same` padding, an input side `G` becomes `ceil(G / strides)`. With
`valid` padding, use the normal pooling/convolution output formula; the stored
positional size assumes the common setup where window/kernel and stride align.

## `Upsample`

```python
from diffusion.layers.manipulation.upsample import Upsample

up = Upsample(
    dim=64, 
    grid_size=4, 
    scaling_method="cnn_interpolate", 
    scaling_interpolation_method="bilinear", 
    cnn_dim_ratio=2, 
    cnn_kernel_size=3, 
    pos_embed_type=None, 
)
y = up((x, condition))          # [B,16,64] -> [B,64,128]
```

Allowed scaling modes are:

| Mode | Spatial operation | Pre-position output channels |
|---|---|---|
| `"cnn_transpose"` | stride-2 transposed convolution | `dim * cnn_dim_ratio` |
| `"interpolate"` | 2x Keras upsampling | `dim` |
| `"cnn_interpolate"` | 2x upsampling then convolution | `dim * cnn_dim_ratio` |

All modes double both spatial axes and therefore quadruple spatial token count.
Interpolation commonly accepts `"nearest"` or `"bilinear"`.

## `LocalMixer`

`LocalMixer` applies a depthwise spatial convolution and optionally a 1x1
pointwise convolution:

```python
from diffusion.layers.manipulation.local_mixer import LocalMixer

mixer = LocalMixer(
    dim=128, 
    grid_size=8, 
    kernel_size=3, 
    strides=1, 
    padding="same", 
    depth_multiplier=1, 
    use_pointwise=True, 
    pointwise_dim_ratio=1, 
    zero_init=True, 
    pos_embed_type=None, 
)
y = mixer((x, condition), training=True)  # [B,64,128] -> [B,64,128]
```

`strides=1` enables a residual update. Its spatial shapes must match, so use
`padding="same"` unless the kernel is otherwise size-preserving. Larger strides
disable the residual and return the reduced convolutional output. Without a
pointwise convolution the width is `dim * depth_multiplier`; with one it is
`dim * pointwise_dim_ratio`. `zero_init=True` makes the initial local correction
zero.

## Class-token and output-width behavior

When `circumvent_cls_token=True`, the leading token bypasses the spatial
reshape and is projected only when required to match the output. For example,
downsampling `[B,65,64]` from an 8x8 grid produces `[B,17,64]`.

After the spatial operation:

- `pos_merger_type="add"` preserves width;
- `pos_merger_type="concat"` appends positional channels;
- a configured `mlp_output_dim` replaces the merged width with that value.

`dim` and `grid_size` are required embedding arguments. Constructors also
accept positional options, `ln_mlp_ratio`, `ln_no_adaptation`, `mlp_ratio`,
`mlp_activation_func`, `mlp_output_dim`, and Keras `name`, `dtype`, and
`trainable`. Each manipulation layer supplies `use_layer_norm` and `ln_dim`
internally, so do not repeat those names in `**kwargs`.

The inherited argument saver nevertheless records `ln_dim` in `get_config()`.
Before manually reconstructing `Downsample`, `Upsample`, or `LocalMixer` with
`from_config`, copy the config and remove `ln_dim`; the constructor restores it
from `dim`.
