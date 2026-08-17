# Diffusion layers

This directory contains the reusable Keras layers from which the diffusion
transformers are assembled. The classes under `diffusion.models.transformer`
choose, order, and configure these layers to form a raw prediction network. The
classes under `diffusion.models.wrapper` do not replace those layers: they own a
raw network (and usually an EMA clone) and add noising, optimization,
classifier-free guidance, evaluation, and sampling. In normal use, construct a
transformer or `UNet`, pass it to a wrapper, and compile/train the wrapper.

All tensors are channels-last. The notation used below is:

- `B`: batch size
- `T`: token count
- `D`: feature width
- `C`: condition width
- `G`: square spatial-grid side, so `T = G * G`

## Layer catalog

| Layer | Purpose | Input | Output |
|---|---|---|---|
| `AdaLNZero` | Conditioned layer normalization with zero-initialized shift, scale, and residual gate | `(x [B,T,D], cond [B,C])` | `[B,T,D]`, optionally with gate `[B,1,Gd]` |
| `DropPath` | Drop a complete residual branch during training | `[B,...]` | same shape/dtype |
| `FeatureHandler` | Select saved features by ID, concatenate/add them, then optionally normalize/project | lists of compatible tensors | merged tensor or `None` |
| `SingleTokenLayer` | Create a learned single token or wrap a supplied vector as one token | `(reference, token)` | normally `[B,1,D]` |
| `BaseLayer` | Internal factory for adaptive normalization and dense MLPs | construction utility | `AdaLNZero`, `Sequential`, or `None` |

Subdirectories provide [attention blocks](block/README.md),
[embeddings](embedding/README.md), and
[spatial token manipulation](manipulation/README.md).

## Common Keras options

Every concrete layer ultimately derives from `tf.keras.layers.Layer`. Its
`**kwargs` therefore accepts common Keras layer options such as:

```python
layer = DropPath(
    drop_prob=0.1,
    name="encoder_2/drop_path",
    dtype="float32",
    trainable=True,
)
```

Layers based on `BaseLayer` additionally accept shared factory arguments:
`use_layer_norm`, `ln_dim`, `ln_mlp_ratio`, `ln_no_adaptation`, `mlp_ratio`,
`mlp_activation_func`, and `mlp_output_dim`. A subclass may supply some of these
itself; its class docstring states which keys must not be repeated.

Constructor arguments are saved by `ArgumentSaverLayer`, so `get_config()`
includes Keras configuration and constructor settings. Direct round trips work
for `AdaLNZero`, `DropPath`, and `FeatureHandler`. Several composed layers also
save keys that their constructor supplies internally: all `BaseEmbedding`
derivatives duplicate `ln_dim`, `VisionTransformerBlock` duplicates
`use_layer_norm` and `ln_dim`, and `DiTDecoderBlock` additionally duplicates
`gate_query_flag`. Remove those exact keys from a copied config before calling
the corresponding `from_config`. Runtime weights always require normal Keras
weight/model saving.

## Conditioning and zero gates

`AdaLNZero(dim=D)` first computes non-affine layer normalization. Its condition
MLP predicts `shift [B,D]`, `scale [B,D]`, and optionally `gate [B,Gd]`; these
are expanded across tokens. The final projection starts at zero, so the initial
modulated features equal ordinary normalized features and the initial gate is
zero. Transformer residual branches therefore begin as exact identities when
their residual widths agree.

```python
norm = AdaLNZero(dim=64, gate_dim=128, return_gate=True)
h, gate = norm((tokens, condition), training=True)
# h: [B, T, 64]; gate: [B, 1, 128]
```

Set `no_adaptation=True` to ignore `condition`; the layer then returns ordinary
normalization and, if requested, scalar gate `1.0`.

## Feature IDs and merges

`FeatureHandler.ids` uses ordinary Python indices into `features_list`:

```python
handler = FeatureHandler(
    ids=[0, -1, -1],
    connect_type="concat",
    ln_dim=192,  # three selected 64-channel features
)
merged = handler([early, middle, latest])
```

The example concatenates `early`, `latest`, and `latest`, in that order.
Negative indices, repeated indices, and arbitrary ordering are valid; an
out-of-range index raises `IndexError`. With `connect_type="concat"`, every
dimension except `connect_axis` must match. With `"add"`, TensorFlow
broadcasting rules apply. Passing `ids=[]` and no secondary tensors returns
`None`. If the constructor receives `ids=None`, each call must supply an
explicit ID list.

`second_list` appends tensors after the ID-selected tensors:

```python
merged = handler(features, second_list=[skip], ids=[-1], cond=condition)
```

The current constructor requires a non-`None` `ln_dim` even when normalization
and the MLP are disabled, because it records that width in its MLP factory.
When adaptive normalization is enabled, `ln_dim` must equal the merged channel
width and `cond [B,C]` is required. `mlp_output_dim` optionally projects the
merged result to a new final width.

## Training behavior

The `training` argument is forwarded through normalization, attention,
convolution, and dense sublayers. `DropPath` is the explicitly stochastic
layer: `training=False` or `None` makes it an identity, while `training=True`
drops paths independently per sample by default. `scale_by_keep=True` preserves
the expected residual magnitude.
