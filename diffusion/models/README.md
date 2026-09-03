# Diffusion model architectures

This directory separates raw neural architectures from the Keras models that
train and sample with them:

- `transformer/` contains DiT-style raw networks. They consume already-noisy
  tensors plus timestep/label conditions and return prediction tensors, with
  optional tuples of intermediate features and auxiliary outputs.
- `convolution/` contains depth-based `UNet` and `UNetClassifier` raw
  networks assembled from reusable convolution layers.
- `wrapper/` contains trainable orchestration models. A wrapper owns a raw
  network, constructs schedules and noisy examples, computes losses, maintains
  an optional exponential-moving-average (EMA) copy, and implements reverse
  diffusion and Keras training/evaluation hooks.

In normal use, instantiate a transformer or U-Net first, pass it to a wrapper,
then compile and fit the wrapper. Call the raw architecture directly only when
you have already prepared its timestep and condition inputs.

`UNet` mirrors the transformer's tracked hierarchy: depth `0` is the embedded
image/condition tensor, depths `1..N` correspond to `layers_dicts`, and
`full_return=True` supplies aligned feature, regularizer, and latent-statistic
outputs. Its ordinary encoder/bottleneck/decoder hierarchy uses skips. Setting
`reshaper_kwargs={"add_kl": True, "latent_dim_ratio": [...]}` inserts the
variational bottleneck and disables those skips automatically. It can then be
trained by `DiffusionModel` with KL loss and decoded with `sample_vae`.
`latent_dim_ratio` contains exactly one positive ratio per contiguous
flatten/unflatten pair, ordered by ascending flatten depth; omission means
`1.0` for every pair. A convolutional multiscale U-Net places such stochastic
pairs at successive encoder scales so its decoder skips originate from their
unflattened outputs.

For a multilevel transformer U-DiT used with `sample_vae`, all adjacent pairs
form one central bridge after the complete encoder stack and before the
decoder/up-sampling computation. Training may route an encoder feature at each
flatten stage; latent sampling bypasses those routes, and the decoder consumes
the paired stochastic features without direct pre-latent encoder bypasses.

`DiTEncoderDecoder` and `DiTEncoderDecoderClassifier` accept their base
denoiser/classifier three-input contracts and additionally accept a fourth
decoder image for teacher forcing. Their inherited transformer APIs, active
resolution policy, and progressive depth belong to the encoder; resolution
updates are synchronized to the attached decoder, whose architecture keeps its
own fixed construction depth. Use `DiffusionModel` for plain encoder/decoder
denoising, `DiffusionClassifier` for the classifier's
three-input workflow, and a custom classifier training step when its fourth
tensor is required.

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

The convolutional equivalent uses the same wrapper boundary:

```python
from diffusion import DiffusionModel, UNet

network = UNet(image_size=28, channels=1, widths=(32, 64, 96))
model = DiffusionModel(network, scheduler_name="clipped_cosine")
```

`UNet.add_depths(...)` appends shape-preserving residual stages, so wrapper
progressive-depth schedules do not require a separate convolutional training
path. `UNetClassifier` adds a feature aggregation/classifier branch and is the
raw network accepted by both `DiffusionClassifier` and
`DiffusionClassifierV2`. Targeted depth specifications can grow the inherited
`network` branch, the `classifier` branch, or both.

See the README in each child directory for constructor dictionaries, tensor
contracts, classifier state, depth routing, training, and sampling APIs.
