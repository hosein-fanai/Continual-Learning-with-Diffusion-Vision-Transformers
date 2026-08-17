# Diffusion model architectures

This directory separates raw neural architectures from the Keras models that
train and sample with them:

- `transformer/` contains DiT-style raw networks. They consume already-noisy
  tensors plus timestep/label conditions and return prediction tensors, with
  optional tuples of intermediate features and auxiliary outputs.
- `convolution/` contains the convolutional `UNet` alternative.
- `wrapper/` contains trainable orchestration models. A wrapper owns a raw
  network, constructs schedules and noisy examples, computes losses, maintains
  an optional exponential-moving-average (EMA) copy, and implements reverse
  diffusion and Keras training/evaluation hooks.

In normal use, instantiate a transformer or U-Net first, pass it to a wrapper,
then compile and fit the wrapper. Call the raw architecture directly only when
you have already prepared its timestep and condition inputs.

```python
import tensorflow as tf

from diffusion import DiffusionModel, DiffusionTransformer

network = DiffusionTransformer(
    image_size=28,
    channels=1,
    patch_size=2,
    dim=64,
    depth=4,
    num_classes=10,
)
model = DiffusionModel(network, scheduler_name="clipped_cosine")
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
```

See the README in each child directory for constructor dictionaries, tensor
contracts, classifier state, depth routing, training, and sampling APIs.
