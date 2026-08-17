# Continual learning and diffusion models

This TensorFlow 2.10 research codebase combines two related workflows:

- class-incremental CIFAR learning with ordinary fine-tuning, replay-buffer
  rehearsal, or conditional-VAE generative replay; and
- conditional image diffusion with configurable transformer/U-Net networks,
  optional joint classification, EMA evaluation, classifier-free guidance,
  sampling, and progressive training over timesteps, resolution, or depth.

The code is organized as importable research modules rather than a published
Python package. Run scripts and notebooks from the repository root so imports
such as `from diffusion import DiffusionModel` resolve consistently.

## Environment

The pinned runtime dependency is TensorFlow 2.10:

```powershell
python -m pip install -r requirements.txt
```

Dataset, reporting, and legacy continual-learning paths also import NumPy,
PyYAML, pandas, scikit-learn, Matplotlib, and Pillow. Ensure those packages are
available when using the corresponding modules.

## How the diffusion API fits together

```text
(clean image, class ID)
          |
          v
DiffusionModel / DiffusionClassifier wrapper
  schedule + noising + losses + optimizer + EMA + sampler
          |
          v
DiffusionTransformer / DiTClassifier / UNet / UNetClassifier raw network
  embeddings -> depth 1..N blocks -> prediction heads
          |
          v
reusable layers (attention, residual convolution, routing, scaling, embeddings)
```

The raw architectures in `diffusion/models/transformer/` and
`diffusion/models/convolution/` implement tensor transformations. The wrappers
in `diffusion/models/wrapper/` own the diffusion process and Keras lifecycle.
Compile and fit the wrapper; call the raw network only with already prepared
noisy images, timestep IDs, and embedding-label IDs.

### Minimal diffusion model

```python
import tensorflow as tf

from diffusion import DiffusionModel, DiffusionTransformer

network = DiffusionTransformer(
    num_classes=10, 
    use_cfg=True, 
    timesteps=1_000, 
    image_size=28, 
    channels=1, 
    patch_size=2, 
    dim=64, 
    depth=4, 
)
model = DiffusionModel(
    network=network, 
    scheduler_name="clipped_cosine", 
    test_steps=50, 
    test_cfg_scale=4.0, 
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3), 
    loss="mse"
)

# Dataset elements: float images [B, 28, 28, 1] scaled to [-1, 1],
# paired with zero-based integer class IDs [B].
# model.fit(dataset, epochs=20)

# With CFG, sampling IDs are already shifted: 0 is null and 1..10 are classes.
samples = model.sample(labels=[1, 2, 3], steps=50, scale=4.0)
```

`DiTClassifier` or `UNetClassifier` plus `DiffusionClassifier` adds a
classifier branch and joint classification loss. `DiffusionClassifierV2`
separates generator and classifier variable groups/optimizers. Their READMEs
define the branch state, progressive-depth behavior, and variable-selection
IDs.

### Architectural depth and routed IDs

For transformer models, depth `0` is the patch-embedded input before any stage.
For `UNet`, depth `0` is the projected image concatenated with broadcast time
and label embeddings. In both families, depths `1..N` are outputs of the
tracked `layers_dicts`, and `full_return=True` exposes the aligned feature and
regularizer lists. An `*_ids_dict` maps a target depth to source depths, while
an `*_ids` sequence selects depths where a component is enabled. `None`
commonly expands to every eligible depth; negative IDs are normalized relative
to the final depth.

The standard convolutional hierarchy uses encoder residual stacks and
downsampling, a bottleneck, then upsampling with encoder skips. A variational
bottleneck is enabled without manually calculating depth IDs:

```python
from diffusion import UNet

vae_network = UNet(
    image_size=32, 
    channels=3, 
    widths=(32, 64, 96), 
    reshaper_kwargs={"add_kl": True, "latent_dim_ratio": 0.5}, 
)
```

This configuration disables skips automatically so decoding cannot bypass the
latent. Train it through `DiffusionModel` with a nonzero `kl_loss_coef`, then
use `sample_vae(...)` to decode latent samples. `UNet.add_depths(...)` and the
targeted `UNetClassifier.add_depths(...)` append shape-preserving residual
stages for progressive-depth training. See the
[convolution model guide](diffusion/models/convolution/README.md) for exact
calls and supported specifications.

### Schedules

```python
from diffusion.schedulers import ScheduleConfig, ScheduleKind, generate_sigmas
from diffusion import make_schedule

vp = make_schedule("linear", 1_000, beta_start=1e-4, beta_end=2e-2)
karras = generate_sigmas(ScheduleConfig(
    kind=ScheduleKind.KARRAS, 
    num_steps=50, 
    sigma_min=0.002, 
    sigma_max=80.0, 
    rho=7.0, 
))
```

`make_schedule` returns beta, cumulative-alpha, signal/noise amplitude, sigma,
and normalized-time arrays. Use `generate_sigmas` directly for native VE or
Karras magnitudes; the all-in-one helper reports a VP beta-equivalent curve.

## Historical configuration-driven MNIST workflow

`common.config` defines nested dataclasses and YAML serialization.
`common.train` loads normalized MNIST batches, constructs a diffusion wrapper,
trains it, and writes configured reports:

```python
from common.config import load_config
from common.train import main

config = load_config("configs/default.yaml")
print(config.model.dit_classifier.kwargs())
# common.train.main(config) is the historical executor once its config schema
# and selected model constructors use the same API revision.
```

Omitted YAML fields use dataclass defaults and unknown fields are rejected.
This configuration layer predates several generalized transformer constructor
names and performs no alias translation. The tracked YAML examples therefore
need the mappings documented in `common.config` before they can construct the
current raw networks. That requires migrating the config dataclasses or
translating at the `common.train.get_model` call site; changing the YAML keys
alone is rejected by the present dataclasses. For a matching schema revision,
`common.train.main` is the repository-root executor and `__main__.py` is the
package-context command entry point.

## Class-incremental learning

The legacy continual-learning interface adds one class per task and optionally
rehearses prior classes from a bounded replay buffer or conditional VAE:

```python
from common.dataloader import load_cifar10
from common.learner import continually_learn

accuracies = continually_learn(
    class_num=10, 
    load_dataset_fn=load_cifar10, 
    tuned_model_path="models/hyperas/cifar10_cnn_model_00.h5", 
    use_buffer=True, 
    buffer_kwargs={
        "maxlen": 10_000,
        "sample_num": 1_000,
        "insert_num": 1_000,
        "seed": 42,
    }, 
)
```

Replay and VAE rehearsal are mutually exclusive. See `common/README.md` for
all loader callback requirements and dictionary keys, and
`autoencoder/README.md` for conditional labels, training, and generation.

## Directory guide

- [`common/`](common/README.md): configuration, datasets, continual learner,
  replay buffer, losses, callbacks, plotting, and the training pipeline.
- [`autoencoder/`](autoencoder/README.md): VAE and classifier-VAE models.
- [`diffusion/`](diffusion/README.md): schedules, models, layers, metrics, and
  callbacks.
- [`configs/`](configs/README.md): YAML configuration examples and schema use.
- [`data/`](data/README.md): pre-extracted CIFAR feature arrays.
- [`models/`](models/README.md): checkpoints and legacy model artifacts; this
  is not the `diffusion.models` source package.
- [`notebooks/`](notebooks/README.md): exploratory and archived experiments.
- [`results/`](results/README.md): generated run artifacts and reports.
- [`gifs/`](gifs/README.md): retained example animations.
- [`others/`](others/README.md): conceptual reference material.

Every source subdirectory has its own API README. Every Python class, method,
and function documents its accepted input types, output types, tensor shapes,
state changes, valid modes, and constrained dictionary/keyword forms in its
docstring.
