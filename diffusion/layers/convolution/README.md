# Convolution layers

These channels-last Keras components build `UNet` and `UNetClassifier`. They
are public from either package path:

```python
from diffusion import (
    ImageDownsample,
    ImageUpsample,
    LayerDict,
    ResidualConvBlock,
    ResidualConvStack,
    VariationalReshaper,
)
from diffusion.layers.convolution import ResidualConvBlock
```

## Catalog

| Class | Main configuration | Behavior |
|---|---|---|
| `ResidualConvBlock` | `filters`, optional `condition_dim`, `kernel_size`, activation, batch norm, spatial dropout, `zero_init` | Two same-padded convolutions plus an identity or 1x1-projected residual |
| `ResidualConvStack` | the block options plus `depth` | Reuses one condition across a fixed sequence of residual blocks |
| `ImageDownsample` | optional `filters`, `scaling_method`, `kernel_size`, `strides`, activation | `avg_pooling`, `max_pooling`, or `cnn_stride` |
| `ImageUpsample` | optional `filters`, `scaling_method`, interpolation, `kernel_size`, `strides`, activation | `interpolate`, `cnn_interpolate`, or `cnn_transpose` |
| `VariationalReshaper` | `reshape_type`, `source_shape`, `add_kl`, `latent_dim_ratio` | Static flatten/unflatten with optional reparameterized latent statistics |
| `LayerDict` | `layers_dict`, optional `execution_order` | Serializable mapping container that tracks child variables; its owner controls execution |

All convolution operators accept a rank-four image map `x [B,H,W,C]`.
Residual blocks also accept `(x, condition [B,D])`; they project and broadcast
the condition after the first convolution. Downsamplers and upsamplers accept
the same tuple for stage-call compatibility and ignore its condition.

## Residual and scaling example

```python
import tensorflow as tf

from diffusion.layers.convolution import (
    ImageDownsample,
    ImageUpsample,
    ResidualConvStack,
)

x = tf.random.normal([4, 17, 15, 8])
condition = tf.random.normal([4, 12])

encoder = ResidualConvStack(
    filters=16,
    depth=2,
    condition_dim=12,
    dropout_rate=0.1,
)
down = ImageDownsample(filters=16, scaling_method="cnn_stride")
up = ImageUpsample(filters=16, scaling_method="cnn_interpolate")

h = encoder((x, condition), training=True)  # [4,17,15,16]
h = down((h, condition), training=True)     # [4,9,8,16]
h = up((h, condition), training=True)       # [4,18,16,16]
```

Pooling and strided-convolution downsampling use same padding. Upsampling
multiplies each spatial dimension by `strides`; owning U-Net stages resize to
the exact skip shape when odd dimensions do not invert exactly. When `filters`
is omitted, a scaling layer preserves its input channel width.

`zero_init=True` zero-initializes the second convolution of a block. On a
stack, only the final block receives that option. `training` is forwarded to
batch normalization, spatial dropout, convolutions, and nested layers.

## Variational reshaping

`VariationalReshaper` always returns `(x, mean, log_var)`. A plain flatten or
unflatten performs only the requested static reshape. A KL-enabled flatten
projects the flattened feature to mean/log-variance vectors, samples with the
reparameterization rule, and projects back when `latent_dim_ratio != 1`:

```python
from diffusion import VariationalReshaper

flatten = VariationalReshaper(
    reshape_type="flatten",
    source_shape=(4, 4, 32),
    add_kl=True,
    latent_dim_ratio=0.5,
    name="depth_4_reshaper",
)
z_for_decoder, z_mean, z_log_var = flatten(feature_map)

unflatten = VariationalReshaper(
    reshape_type="unflatten",
    source_shape=(4, 4, 32),
)
restored, _, _ = unflatten(z_for_decoder)
```

`source_shape` must be fully known and positive. `latent_dim_ratio` must be
positive and must produce at least one latent value. The optional projection
is named `<reshaper-name>/z`, which is the stable lookup used by
`DiffusionModel.sample_vae`.

## Tracked stage mappings

`LayerDict` gives a model dictionary-style access while keeping every child
visible to Keras variable tracking and serialization:

```python
from diffusion import LayerDict, ResidualConvStack

stage = LayerDict(
    {"convolution_block": ResidualConvStack(filters=32, depth=2)},
    execution_order=("convolution_block",),
)
block = stage["convolution_block"]
keys = stage.execution_order
```

It implements stable `keys`, `values`, `items`, `get`, membership, iteration,
assignment, and `update`. `LayerDict` intentionally has no stage-wide `call`;
`UNet` executes contained components according to their public keys.

All six classes support `get_config()`/`from_config()` and common Keras
`name`, `dtype`, and `trainable` options. Learned values still require normal
Keras weight or model saving.
