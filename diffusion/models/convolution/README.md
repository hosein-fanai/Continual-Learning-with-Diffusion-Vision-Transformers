# Convolutional diffusion models

This directory provides `UNet`, a conditional convolutional noise-prediction
network that implements the same raw-network contract as
`diffusion.models.transformer.DiffusionTransformer`.

The directory roles are distinct:

- `diffusion.models.convolution` and `diffusion.models.transformer` contain raw
  networks that map `(noisy image, timestep, label)` to predicted noise.
- `diffusion.models.wrapper` owns one of those raw networks, creates the EMA
  clone, adds the diffusion schedule and noising process, implements losses,
  and provides `fit`, `evaluate`, and `sample`.

Compile and train the wrapper, not the bare `UNet`, for ordinary diffusion use.

## Architecture

`UNet` projects the image with a 1x1 convolution, embeds timestep and label,
broadcasts those conditions across the spatial grid, and concatenates them.
Each encoder level runs `block_depth` residual blocks and saves every block
output before 2x average pooling. The bottleneck runs `bottleneck_depth`
residual blocks. Decoder levels resize to the corresponding skip's exact shape,
concatenate skips in last-in/first-out order, and run matching residual blocks.
The final zero-initialized 1x1 projection returns predicted noise.

`widths` defines the number and channel width of encoder levels; for example,
`widths=(32, 64, 96)` creates three encoder and three decoder levels.
`block_depth=2` creates two residual blocks/skips at every such level.
`bottleneck_depth` applies only at the most compressed representation. These
are U-Net topology terms, not the transformer's depth-ID system: `UNet` does
not support `add_depths` or dynamic transformer layer insertion.

## Constructing and calling `UNet`

```python
import tensorflow as tf
from diffusion.models.convolution.unet import UNet

network = UNet(
    num_classes=10,
    use_cfg=True,
    timesteps=1000,
    image_size=32,
    channels=3,
    widths=(32, 64, 96),
    block_depth=2,
    bottleneck_width=128,
    bottleneck_depth=2,
    activation_func="swish",
    final_activation_func="linear",
    use_batch_norm=True,
    upsampling_interpolation="bilinear",
    build=True,
    name="unet",
    dtype="float32",
    trainable=True,
)

x_t = tf.random.normal([8, 32, 32, 3])
t = tf.constant([0, 10, 50, 100, 250, 500, 750, 999], tf.int32)
embedding_labels = tf.constant([0, 1, 2, 3, 4, 5, 6, 7], tf.int32)
eps = network((x_t, t, embedding_labels), training=False)
# eps: [8, 32, 32, 3]
```

Inputs and outputs are:

| Item | Dtype | Shape/range |
|---|---|---|
| noisy images | floating compute-compatible | `[B,H,W,channels]` |
| timesteps | integer | `[B]`, values `0 <= t < timesteps` |
| direct-call labels | integer | `[B]`, values `0 <= label < num_labels` |
| predicted noise | floating compute dtype | `[B,H,W,channels]` |

With `use_cfg=True`, direct network calls use embedding index 0 for the
unconditional label and indices 1 through `num_classes` for real classes. The
wrapper accepts ordinary dataset labels 0 through `num_classes - 1` during
training and performs the shift itself. With `use_cfg=False`, direct and dataset
labels are both ordinary zero-based IDs.

`full_return=True` returns the compatibility tuple
`(noise, condition, [], [None], (None, None))`. `condition` is `[B,
time_embedding_dim + label_embedding_dim]`; the placeholders indicate that this
plain U-Net produces no transformer feature regularizers or KL latent state.

## Training through `DiffusionModel`

```python
from diffusion.models.wrapper.diffusion_model import DiffusionModel

model = DiffusionModel(
    network=network,
    use_ema=True,
    p_uncond=0.1,
    test_steps=50,
    test_cfg_scale=4.0,
)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=20)

# For CFG, label 0 is unconditional and 1 is the first real class at sampling.
images = model.sample(labels=[1, 2, 3], steps=50, scale=4.0, eta=0.0)
```

The datasets normally yield `(clean_images, zero_based_class_labels)`. The
wrapper generates timesteps/noise and trains the prediction target internally.
`eta=0` is deterministic DDIM; positive values add sampling noise.

## Progressive resolution

Convolution kernels are reusable at other spatial sizes, and decoder levels
align to actual skip shapes. Use:

```python
network.set_current_resolution(48)
```

or let `DiffusionModel.fit_progressively` manage resolution stages. `None`
restores `image_size`. The resolution must be a positive integer. Progressive
timestep and resolution tasks are supported; progressive depth tasks call
`add_depths` and deliberately raise `NotImplementedError` for this fixed U-Net.

## Constructor reference

- `num_classes`, `timesteps`, `image_size`, `channels`, every `widths` item,
  block depths, and embedding dimensions must be positive integers.
- `image_embedding_dim` is the initial noisy-image projection width.
- `time_embedding_dim` and `label_embedding_dim` are concatenated before being
  spatially broadcast.
- `activation_func` and `final_activation_func` accept Keras activation names or
  serializable callables.
- `upsampling_interpolation` is passed to `tf.image.resize`.
- `name_prefix` prefixes internal layer names; Keras `name` names the model.
- `build=True` eagerly creates variables so a wrapper can immediately clone and
  synchronize EMA weights.
- `**kwargs` is forwarded to `tf.keras.Model`; common valid keys are `name`,
  `dtype`, and `trainable`.

Constructor values are included in `get_config()`. Reconstruct with
`UNet.from_config(network.get_config())`; transfer learned state separately with
Keras weights/model saving or `set_weights`.

