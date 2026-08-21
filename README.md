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

Install the pinned TensorFlow runtime and the reporting/HPO dependencies with:

```powershell
python -m pip install -r requirements.txt
```

The requirements include Optuna, TensorBoard, PyYAML, pandas, scikit-learn,
Matplotlib, and Pillow used by the experiment pipeline.

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

## Configuration-driven experiments

`common.config` defines nested dataclasses and YAML serialization.
`common.train` supports MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 plus every
end-to-end model family used by the project. Set `model.name` for the generic
path; leave it `null` to retain the original DiT/DiT-classifier behavior.

```python
from common.config import load_config
from common.train import run_experiment

config = load_config("configs/my-run.yaml")
result = run_experiment(config)
print(result["results_path"])
```

Generic model-specific constructor options live in `model.kwargs`; diffusion
wrapper options live in `model.wrapper_kwargs`. Omitted fields keep defaults
and unknown typed-section fields are rejected.

## Hyperparameter optimization

The 21 notebooks under [`notebooks/hpo/`](notebooks/hpo/README.md) cover all
task/model pairings supported by the HPO runner. They call one API:

```python
from common.hpo import SEARCH_SPACES, run_hpo

study = run_hpo(
    task="generation",
    model_name="unet",
    dataset_name="CIFAR10",
    n_trials=30,
    epochs=50,
    results_path="results/hpo",
)
```

Each Optuna trial first writes and reloads a YAML config, then uses
`common.train` for data, construction, training, and reports. Trial directories
contain weights, resolved config, plots, GIFs, CSVs, and objective values;
TensorBoard event filenames encode every optimized value and the event itself
records the full name/value mapping. Study state is resumable from SQLite and
is mirrored to `trials.csv`. Dataset-specific study directories prevent runs
on CIFAR-10 and CIFAR-100 from sharing state. Joint models use a two-objective
Pareto study.

## Class-incremental learning

`continually_learn` adds one class per task and accepts either a complete
`Config` or the original inputs as direct keywords. Config mode builds the
loader and model bundle through `common.dataloader.get_datasets` and
`common.model.get_model`; every classifier and replay-model phase then uses the
shared training and reporting APIs.

```python
from common.config import Config
from common.learner import continually_learn

config = Config(
    dataset={"name": "cifar10", "preprocess": "min-max"},
    model={"name": "cnn", "show_network_summary": False},
    training={"task": "continual", "epochs": 20},
    continually_learn={"use_buffer": True, "plot_results": True},
)
accuracies = continually_learn(config)
```

Direct mode remains useful with an existing classifier template or runtime
model object:

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

Each task rebuilds its training and validation inputs as `tf.data.Dataset`
pipelines. Both training and test evaluation are reported through
`common.train`. Fixed-buffer and generative rehearsal are mutually exclusive.
Configured sample limits, shuffling, raw-image padding, and seed are preserved
inside the task loop; typed replay models derive their class count and padded
dimensions from the dataset. `model.weights_path` initializes the replay model,
or the incremental classifier when no replay model is built.
See `common/README.md` for the complete direct-key list, config-field mapping,
supported raw-model/wrapper mappings, loader requirements, and dictionary keys;
see
`autoencoder/README.md` for conditional labels, training, and generation.

## Directory guide

- [`common/`](common/README.md): configuration, datasets, continual learner,
  replay buffer, losses, callbacks, plotting, and the training pipeline.
- [`autoencoder/`](autoencoder/README.md): VAE and VAE-classifier models.
- [`diffusion/`](diffusion/README.md): schedules, models, layers, metrics, and
  callbacks.
- [`configs/`](configs/README.md): YAML configuration examples and schema use.
- [`data/`](data/README.md): pre-extracted CIFAR feature arrays.
- [`models/`](models/README.md): checkpoints and legacy model artifacts; this
  is not the `diffusion.models` source package.
- [`notebooks/`](notebooks/README.md): exploratory, archived, and HPO experiments.
- [`results/`](results/README.md): generated run artifacts and reports.
- [`gifs/`](gifs/README.md): retained example animations.
- [`others/`](others/README.md): conceptual reference material.

Every source subdirectory has its own API README. Every Python class, method,
and function documents its accepted input types, output types, tensor shapes,
state changes, valid modes, and constrained dictionary/keyword forms in its
docstring.
