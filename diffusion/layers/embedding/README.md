# Embedding layers

Embedding layers convert discrete conditions and image patches into the token
representations consumed by diffusion transformers. They inherit shared
normalization/MLP factories from `BaseLayer` and standard Keras configuration
from `tf.keras.layers.Layer`.

## Positional modes

`PosEmbedType` accepts the following exact strings or `None`:

| Value | Table created | Typical shape | Learned? |
|---|---|---|---|
| `None` | no position table | — | — |
| `"new_weight"` | zero-initialized table at target size | `[1,O²,D]` | yes |
| `"1d_sincos"` | flattened-position or discrete-step sine/cosine table | `[1,O²,D]` or `[steps,D]` | fixed unless installed in a trainable lookup |
| `"2d_sincos"` | separate row/column sine/cosine features | `[1,O²,D]` | no |
| `"1d_interpolate"` | fixed flattened source table | `[1,G²,D]` | no |
| `"2d_interpolate"` | fixed row/column source table | `[1,G²,D]` | no |
| `"1d_learned_interpolate"` | learned flattened source table | `[1,G²,D]` | yes |
| `"2d_learned_interpolate"` | learned two-dimensional source table | `[1,G,G,D]` | yes |

Here `G` is `grid_size`, `O` is `output_grid_size`, and `D` is the raw
embedding width. Interpolation modes resize the source table in `_pos_merger`.
`pos_interpolation_method` is passed to `tf.image.resize`; common valid values
include `"nearest"`, `"bilinear"`, `"bicubic"`, and `"area"`.

`pos_merger_type="add"` preserves channels and requires content and positional
widths to match. `"concat"` appends position channels. Concrete layers often
allocate half of `dim` to each part so concatenation returns `dim`; use even
dimensions for that configuration.

## `ConditionEmbedding`

Use a learned table for class IDs:

```python
from diffusion.layers.embedding.condition_embedding import ConditionEmbedding

labels = ConditionEmbedding(
    dim=64, 
    embed_steps=11,              # e.g. null label + 10 real classes
    pos_embed_type="new_weight", 
    embed_trainable=True, 
    name="label_embedding", 
    dtype="float32", 
)
y = labels(label_ids)           # label_ids [B] -> y [B,64]
```

Use sinusoidal initialization for timesteps:

```python
times = ConditionEmbedding(
    dim=128, 
    embed_steps=1000, 
    pos_embed_type="1d_sincos", 
    embed_trainable=False, 
    embed_temperature=10_000.0, 
)
y = times(t)                    # t int tensor [B], values 0..999
```

Only `"new_weight"` and `"1d_sincos"` produce the rank-two table expected by
`ConditionEmbedding`. Spatial/interpolation modes are for token positions, and
`None` does not create lookup weights. Integer IDs may have any shape and the
output appends the embedding width. IDs must be in `[0, embed_steps)`.

Set `embed_freq_dim` to build a raw table at a different width. For example,
`dim=128, embed_freq_dim=32` defaults to a ratio-1 hidden projection ending at
128 channels. Explicit `mlp_ratio` and `mlp_output_dim` override that behavior.

## `PatchEmbedding`

```python
from diffusion.layers.embedding.patch_embedding import PatchEmbedding

patches = PatchEmbedding(
    dim=128, 
    grid_size=8, 
    patch_size=4, 
    patchify_with_cnn=False, 
    pos_embed_type="2d_sincos", 
    pos_merger_type="add", 
    name="patch_embedding", 
)
tokens = patches(images)        # images [B,32,32,C] -> [B,64,128]
```

The standard path is a `patch_size`-kernel, `patch_size`-stride convolution with
`valid` padding. `patchify_with_cnn=True` uses a 3x3 feature convolution followed
by a 3x3 stride convolution with `same` padding. The projected feature map must
be square.

For a changed runtime resolution, pass the actual projected side so the
position table is resized:

```python
tokens = patches(images_48, output_grid_size=12)  # patch_size=4
```

`output_grid_size` affects only positional embeddings; it does not resize the
image or convolution result. It must match that result's side. With
`shift_right_token=True`, the layer prepends a learned BOS token and removes the
last patch. The same trainable BOS value is repeated across the runtime batch,
so shifted token sequences preserve both the input batch size and token count.

## BaseEmbedding factory keyword contract

`BaseEmbedding` is an internal utility and has no public `call`. Its
`_create_embedding_layer(**kwargs)` factory recognizes:

| Key | Valid value | Effect |
|---|---|---|
| `pos_embed_type` | `"new_weight"` or `"1d_sincos"` for lookup use | selects initialization |
| `embed_steps` | positive integer | lookup row count |
| `embed_dim` | positive integer | lookup width |
| `grid_size` | positive integer | source side for spatial modes |
| `output_grid_size` | positive integer | target side for spatial modes |
| `temperature` | positive float | sinusoidal wavelengths |
| `name` | string or `None` | initialization tensor/weight name |

For example, the effective private factory call for a fixed timestep table is:

```python
layer._create_embedding_layer(
    pos_embed_type="1d_sincos", 
    embed_steps=1000, 
    embed_dim=64, 
    temperature=10_000.0, 
)
```

Set `embed_trainable` on the public constructor. Passing it directly through
this private `**kwargs` factory is not valid for non-new tables because the key
is also forwarded to `_create_embeddings`. Unknown keys likewise raise
`TypeError` when an initialized table is created.

## Forwarded configuration

Embedding constructors accept BaseLayer settings (`use_layer_norm`,
`ln_mlp_ratio`, `ln_no_adaptation`, `mlp_ratio`, `mlp_activation_func`, and
`mlp_output_dim` where the subclass permits them) and Keras settings such as
`name`, `dtype`, and `trainable`. Direct
`Class.from_config(layer.get_config())` round trips are supported. The base
constructor ignores a serialized `ln_dim` and consistently derives it from
`dim`.
