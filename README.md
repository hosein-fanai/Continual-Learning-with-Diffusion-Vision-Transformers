# Continual learning with Diffusion Vision Transformers

This TensorFlow 2.10 research codebase combines two related workflows:

- class-incremental CIFAR learning with ordinary fine-tuning, replay-buffer
  rehearsal, or conditional-VAE generative replay; and
- conditional image diffusion with configurable transformer/U-Net networks,
  optional joint classification, EMA evaluation, classifier-free guidance,
  sampling, and progressive training over timesteps, resolution, or depth.

The code is organized as importable research modules rather than a published
Python package. Run scripts and notebooks from the repository root so imports
such as `from diffusion import DiffusionModel` resolve consistently.

The root CLI loads a YAML configuration without training by default, or runs
the configured experiment with `--train`:

```powershell
python . configs/default.yaml
python . --train configs/default.yaml
```

Use `python . --help` for the complete command contract. The repository root
has no importable package initializer, so the CLI uses `python .` rather than a
root-package `python -m` target and the entry point uses absolute imports.
Individual `autoencoder.*` and `diffusion.*` modules can still be run with
`python -m`; their package-level public re-exports are loaded and cached lazily
so a target module is not imported twice during registered Keras self-tests.
Lazy Keras registry proxies also let package-only imports restore registered
SavedModels as their canonical Python classes.

## Environment

Use Python 3.10, which is the version exercised by this project and supported
by the pinned TensorFlow 2.10 runtime. Install the runtime and reporting/HPO
dependencies with:

```powershell
python -m pip install -r requirements.txt
```

The requirements include a TensorFlow-compatible NumPy 1.21--1.23 range plus
Optuna, TensorBoard, PyYAML, pandas, scikit-learn, Matplotlib, and Pillow used
by the experiment pipeline.

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

`DiffusionClassifierV2` requires CFG. Its classifier-only train/test timestep
caps use clean timestep 0 for `None`, the full horizon for `-1`, or an exclusive
`[0, cap)` range for a positive value; this range is independent of any active
progressive generator interval. Direct evaluation must select
`test_part="generator"` or `"discriminator"`, call a phase-specific evaluator,
or use `eval_both=True`. Shared reporting requests both phases.

Transformer classifiers can also prepend a distillation token after the class
token. `DiffusionClassifier` maps `tf.data.Dataset` batches through a frozen
teacher, supports hard cross-entropy or soft KL targets, and reports the class,
distillation, and coefficient-combined accuracies. See the
[transformer token contract](diffusion/models/transformer/README.md#class-and-distillation-tokens)
and [wrapper training contract](diffusion/models/wrapper/README.md#distillation-training)
for the exact output keys, coefficients, and dataset limitations.

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
    reshaper_kwargs={"add_kl": True, "latent_dim_ratio": [0.5]},
)
```

This configuration disables skips automatically so decoding cannot bypass the
latent. `latent_dim_ratio` is a list with exactly one positive entry for each
flatten/unflatten pair, ordered by ascending flatten depth; this
single-bottleneck example therefore has one entry. Omitting the list selects
full-width latents by default.

For a multilevel transformer U-DiT decoded through `sample_vae(...)`, arrange
all adjacent pairs as one central bridge after the complete encoder stack and
before decoder/up-sampling computation. During training, a flatten-stage route
can select its encoder feature; sampling bypasses that route to inject the
matching latent. Decoder routes must then consume the corresponding stochastic
unflattened features instead of reaching around the bridge to pre-latent
encoder features.

Train the network through `DiffusionModel` with a nonzero `kl_loss_coef`, then
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
Karras magnitudes and endpoint-sampled sub-VP marginal deviations; the
all-in-one helper reports a VP beta-equivalent curve.
`"sigmoid"` and `"logistic"` are distinct: sigmoid interpolates per-step beta
between `beta_start` and `beta_end`, while logistic shapes a decreasing
cumulative `alpha_bar` and derives beta from it. Both use `logistic_k` for
transition steepness.

## Configuration-driven experiments

`common.config` defines nested dataclasses and YAML serialization.
`common.train` supports MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 plus every
end-to-end model family used by the project. Set `model.name` for the generic
path; leave it `null` to retain the original DiT/DiT-classifier behavior.

The public orchestration surface is intentionally small:

| Stage | API |
| --- | --- |
| Data | `common.dataloader.get_datasets(config)` |
| Model | `common.model.get_model(config)` |
| Training | `common.train.train_model(config, model, trainset, valset)` |
| Reporting | `common.train.report(config, history, model, trainset, valset)` |
| All stages | `common.train.main(config)` |
| Continual learning | `common.learner.continually_learn(config)` |
| HPO | `common.hpo.run_hpo(...)` |

See [`common/README.md`](common/README.md) for the concise API guide. Exact
configuration fields and compatibility keywords remain documented in the
corresponding dataclasses and function docstrings.

Feature persistence is non-pickling by default. Homogeneous numeric `.npy`
files remain ordinary NPY; heterogeneous numeric feature splits use ordered NPZ
content behind the same public `.npy` filename. Loading a legacy object-NPY
requires explicit `allow_pickle=True`, emits a security warning, and is only
appropriate for a trusted file that will immediately be re-saved in the safe
format.

```python
from common.config import load_config
from common.train import main

config = load_config("configs/default.yaml")
result = main(config)
print(result["results_path"])
```

Generic model-specific constructor options live in `model.kwargs`; diffusion
wrapper options live in `model.wrapper_kwargs`. Omitted fields keep defaults
and unknown typed-section fields are rejected.
`training.task` is normalized case-insensitively and must be `legacy`,
`generation`, `joint`, `classification`, or `continual`. A path-like
`training.results_path` is normalized to text; `None` is accepted only when all
active consumers can run without an artifact directory, including interactive
image display during training and disabled save/TensorBoard/checkpoint outputs.
Set `training.fit_method="fit_progressively"` with `training.stage_tasks` and
the desired stage controls to use the diffusion wrapper's existing progressive
trainer; the default `"fit"` path is unchanged.

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
    fit_method="fit_progressively",
    fit_kwargs={
        "stage_tasks": "timesteps_only",
        "stages_num": 4,
        "stage_epochs": 5,
        "final_epochs": 5,
    },
)
```

Each Optuna trial first writes and reloads a YAML config, then uses
`common.train` for data, construction, training, and reports. Trial directories
contain weights, resolved config, enabled plots/images/GIFs, CSVs, and objective
values. TensorBoard event filenames encode every optimized value and the event
itself records the full name/value mapping. Continual trials additionally log
every epoch under task/class/phase namespaces and write final validation/test
continual metrics. Joint models default to a two-objective Pareto study;
`objective_metrics` and matching `objective_directions` select one or more
other objectives. Continual objectives are read only from validation continual
metrics and default to maximizing `final_average_accuracy`.

For a teacher-distilled class-incremental architecture search, use
`task="continual"`, `model_name="diffusion_classifier"`, and
`use_distillation=True`. This umbrella space conditionally searches the DiT,
encoder-decoder DiT, and U-Net classifier families together with their
optimizer, diffusion, distillation, and replay choices. Dataset, split, seed,
task schedule, training budget, and validation objective remain fixed across
trials so their scores stay comparable.

Study state, the sampler state, and per-trial continual task checkpoints are
resumable by passing the existing study directory to `resume_from`. The saved
study specification is checked before continuation, and a trial interrupted
inside a continual task restarts only that task from its preceding committed
boundary. Study summaries are mirrored to `trials.csv`. Dataset-specific study
directories prevent runs on CIFAR-10 and CIFAR-100 from sharing state.

## Class-incremental learning

`continually_learn` accepts either a complete `Config` or the original inputs
as direct keywords. Its default `task_size=1` begins with one class and adds one
class per task. Larger automatic tasks partition `class_order` by `task_size`;
`class_order_mode="random"` shuffles classes before grouping, while
`task_order_mode="random"` shuffles complete groups without changing their
contents. Explicit `task_groups` can define an exact grouped-class schedule.
Config mode builds the loader and model bundle through
`common.dataloader.get_datasets` and `common.model.get_model`; every classifier
and replay-model phase then uses the shared training and reporting APIs.

For a source-verified experimental protocol covering the baseline ladder,
matched replay budgets, six-cell replay × KD design, candidate gates,
development/confirmation manifests, paired stream-level inference, recovery,
artifacts, and expected TensorFlow retracing, consult the optional local
`others/research-grade-continual-learning.md` reference when that ignored
workspace artifact is present.

```python
from common.config import Config
from common.learner import continually_learn

config = Config(
    dataset={"name": "cifar10", "preprocess": "min-max"},
    model={"name": "cnn", "show_network_summary": False},
    training={
        "task": "continual",
        "epochs": 20,
        "dtype_policy": "mixed_float16",
        "deterministic_ops": True,
    },
    continually_learn={
        "seed": 42,
        "task_size": 1,
        "use_buffer": True,
        "buffer_kwargs": {"strategy": "fifo"},
        "plot_results": True,
    },
)
accuracies = continually_learn(config)
```

Continual diffusion replay uses the same training selector and curriculum
fields. The curriculum is applied to each replay-model task; ordinary
classifier phases still use `training.epochs`.

When `continually_learn.use_distillation=True`, a diffusion classifier with a
distillation token trains task one as the student, then uses an independent
frozen snapshot of each completed `snapshot_network_name` branch (`"raw"` or
`"ema"`) as the next task's teacher. EMA snapshots require EMA to be enabled.
This works with ordinary or progressive fitting and with both V1 and V2
wrappers; HPO enables the same lifecycle with
`run_hpo(..., use_distillation=True)`.

`continually_learn.seed` is the authoritative master seed and is propagated to
the schedule, data, model/layer initialization, task shuffling, replay,
training, sampling, and ensemble noising. `training.dtype_policy` is installed
before construction and governs models, wrappers, layers, schedules, and
optimizers, including mixed-precision loss scaling. With task checkpointing
enabled explicitly, `resume_from` accepts a checkpoint root or committed
task directory and restores models, teachers, optimizers, replay, task cursor,
metrics, and RNG state. Use the run's immutable `input_config.yaml` for this
resume; `config.yaml` is the final artifact-resolved record. When
`reporting.save_csv=True`, task/epoch metrics,
accuracy matrices, schedule, and continual summaries are saved beside the
resolved config and enabled weights/plots/images/GIFs. If
`use_ensemble_accuracy=True`, the ensemble matrix is authoritative for the
reported continual metrics. Average forgetting is signed: the best score
before the final evaluation minus the final score, so positive backward
transfer appears as negative forgetting rather than being clipped to zero.

Ensemble evaluation selects `network_name="raw"|"ema"`. Its `"chunked"` and
`"batched"` computation modes implement the same aggregation;
with a seed, stateless per-timestep noising makes results invariant to mode,
chunk size, and unrelated prior random draws. An `"ema"` selection resolves to
the raw branch when EMA is disabled.

Fixed-buffer replay uses `buffer_kwargs["strategy"]`. `"fifo"` is the exact
historical default, retaining the newest examples. `"reservoir"` gives every
example offered to the buffer equal probability of occupying fixed memory, while
`"class_balanced"` divides feasible storage nearly equally among observed
classes and uses reservoir sampling within each class. Strategy counters,
class allocation state, retained samples, and the private RNG are restored from
task checkpoints, so resumed insertion follows the same stream as an
uninterrupted run.

The optional `continually_learn.optimizer_steps_per_epoch` control fixes the
number of updates for each active task-training phase by repeating only the
already-selected pool; its `null` default changes nothing. The named
`reservoir_er` baseline offers that complete current pool to Algorithm R,
whereas an explicit `strategy: reservoir` with `baseline: null` preserves the
`insert_num` sampled-insertion ablation.

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
        "strategy": "reservoir",
    }, 
)
```

The `hp-tuned` template path keeps every learned non-output-layer weight and
replaces only the output head. With `use_loaded_opt=True`, the new model receives
a fresh optimizer reconstructed from the saved optimizer configuration; slots
and iteration state are not transferred, and an uncompiled saved model is
rejected. With it disabled, the supplied compile configuration is used.

Each task rebuilds its training and validation inputs as `tf.data.Dataset`
pipelines. Both training and test evaluation are reported through
`common.train`. Fixed-buffer and generative rehearsal are mutually exclusive.
Configured sample limits, shuffling, raw-image padding, and seed are preserved
inside the task loop. Typed replay models derive padded dimensions from the
dataset; fresh and restored continual diffusion constructors receive raw
`num_classes=None` and grow their class vocabulary as labels are observed.
`model.weights_path` initializes a continual VAE replay model, the incremental
classifier for classifier-only and buffer runs, or a continual diffusion model
when its paired config contains zero-based `seen_classes`. The wrapper restores the grown
topology before weight loading. Saving dynamic diffusion weights requires a
`Config` and always writes the paired `config.yaml`, even when ordinary config
saving was disabled.
See `common/README.md` for the orchestration guide and inspect
`help(continually_learn)` after importing it from `common.learner` for the
complete direct-key contract;
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

## Validation

Run the complete repository assessment from the project root in the supported
Conda environment:

```powershell
conda run -n tf_env python test.py
```

This command first enforces the source-wide documentation, type-annotation,
named-callable, and branch-purpose-comment contracts. It then discovers the
maintained model and layer classes and runs every registered TensorFlow 2.10
self-test. A missing class, an omitted self-test result, or any non-passing
result makes the command fail.
