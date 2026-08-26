# Common experiment APIs

`common` is the orchestration layer. It prepares data, constructs and compiles
models, runs training, performs reporting, and coordinates continual learning
and HPO. Neural-network implementations remain in `autoencoder` and
`diffusion`.

## Recommended workflow

Use a `Config` for new experiments and `main` when all stages are needed:

```python
from common.config import load_config
from common.train import main

config = load_config("configs/my-run.yaml")
run = main(config)

model = run["model"]
history = run["history"]
evaluations = run["evaluations"]
```

The same pipeline can be called one stage at a time:

```python
from common.dataloader import get_datasets
from common.model import get_model
from common.train import train_model, report

trainset, valset = get_datasets(config)
model = get_model(config)
history = train_model(config, model, trainset, valset)
evaluations = report(config, history, model, trainset, valset)
```

| Goal | API | Main result |
| --- | --- | --- |
| Complete experiment | `common.train.main` | Model, history, evaluations, result path |
| Prepare inputs | `common.dataloader.get_datasets` | Train and validation datasets |
| Construct a model | `common.model.get_model` | Compiled model or continual bundle |
| Fit a model | `common.train.train_model` | Plain metric-history mapping |
| Evaluate/save reports | `common.train.report` | Evaluation mapping |
| Continual learning | `common.learner.continually_learn` | Task accuracies or details |
| Hyperparameter search | `common.hpo.run_hpo` | Resumable Optuna study |

Config mode and direct-keyword mode remain supported. Config mode is preferred
because dimensions, optimizer schedules, paths, progressive training, and
continual settings are resolved consistently. Exact fields and edge cases are
documented by each function and configuration dataclass.

## Datasets

`load_mnist`, `load_fmnist`, `load_cifar10`, and `load_cifar100` return:

```text
(x_train, y_train, x_val, y_val, x_test, y_test)
```

They support class filtering, a stratified validation split, sparse or one-hot
labels, saved features, and these preprocessing modes:

- `"min-max"`: training-set scalar min/max scaling;
- `"normalize"`: elementwise training mean/std scaling;
- `"standardize"` or `"diffusion"`: training extrema mapped to `[-1, 1]`;
- `None` or `""`: no scaling in the direct loader functions.

`get_dataset` converts arrays into a batched `tf.data.Dataset` and can shuffle,
cache, pad, augment, extract features, and prefetch. `get_datasets` selects the
loader and input representation from the requested model:

- image tensors for diffusion, CNN, and pretrained models;
- flattened tensors for DNN and VAE models;
- saved feature tensors when `return_features=True`.

When `get_datasets` receives no preprocessing mode, it selects
`"standardize"` for diffusion models. VAE models use `"standardize"` with a
`tanh` reconstruction, `"min-max"` with `sigmoid`, and `"normalize"` with a
linear reconstruction.

For `training.task="continual"`, `get_datasets` returns the selected loader so
the learner can create class-specific datasets for each task.

## Model construction

`get_model(config)` supports these families:

| Family | Names |
| --- | --- |
| Classifiers | `cnn`, `dnn`, `pretrained`, `hp-tuned` |
| Autoencoders | `vae`, `variational_autoencoder`, `vae_classifier` |
| Diffusion transformers | `diffusion_transformer`, `dit_classifier`, `dit_decoder`, `dit_encoder_decoder`, `dit_encoder_decoder_classifier` |
| Convolutional diffusion | `unet`, `unet_classifier` |

Diffusion raw networks are automatically paired with `DiffusionModel`,
`DiffusionClassifier`, or `DiffusionClassifierV2`. Dataset-derived class counts
and image dimensions are applied automatically. A continual config returns:

```python
{
    "classifier": classifier,
    "classifier_name": "cnn",
    "generative_model": replay_model_or_none,
}
```

The legacy classifier form remains available:

```python
from common.model import get_model

classifier = get_model(10, model_type="CNN")
```

## Training and reporting

`train_model` keeps one return format for every family: a dictionary of metric
names to epoch values. It supports:

- ordinary Keras `fit`;
- VAE `train`;
- diffusion progressive fitting;
- the two generator/classifier phases of `DiffusionClassifierV2`;
- continual model bundles;
- callbacks, TensorBoard, weight saving, and paired configuration saving.

Set `training.fit_method="fit_progressively"` and provide
`training.stage_tasks` for a diffusion curriculum. Stage and final epoch fields
own the progressive budget; `training.epochs` continues to control ordinary
phases.

`report` plots/saves history, evaluates enabled raw and EMA networks, computes
optional ensemble accuracy, and generates final images or GIFs. Saving dynamic
diffusion weights also saves the class mapping and resolved topology required
to reconstruct them.

## Continual learning

`continually_learn` introduces classes from two through the requested class
count. It supports:

- no replay;
- a bounded `ReplayBuffer`;
- conditional VAE replay;
- diffusion replay;
- attached VAE/diffusion classifiers;
- previous-task student distillation and optional timestep-ensemble evaluation.

For a diffusion classifier with an active distillation token and positive
teacher loss, set `continually_learn.use_distillation=True`. Task one may start
teacher-free; each following task snapshots the completed raw student before
its class head expands and uses that frozen snapshot as the teacher. An
explicit `teacher_network` is therefore optional and, when supplied, applies to
task one only. V1 uses this lifecycle in `fit` and `fit_progressively`; V2 uses
it in the classifier phase after either its ordinary or progressive generator
phase.

Configured usage is intentionally small:

```python
from common.config import Config
from common.learner import continually_learn

config = Config(
    dataset={"name": "cifar10", "preprocess": "min-max"},
    model={"name": "cnn", "show_network_summary": False},
    training={"task": "continual", "epochs": 20},
    continually_learn={
        "class_num": 10,
        "use_buffer": True,
        "plot_results": False,
    },
)
accuracies = continually_learn(config)
```

Direct mode remains available for existing model objects and custom loaders;
use `help(continually_learn)` for that compatibility API. Fixed-buffer and
generative replay are mutually exclusive. Set `return_details=True` to receive
histories, reports, final models, and optional ensemble accuracies.

## Hyperparameter optimization

`run_hpo` supports `generation`, `joint`, `classification`, and `continual`
studies through the task/model pairs listed in `SEARCH_SPACES`:

```python
from common.hpo import run_hpo

study = run_hpo(
    task="generation",
    model_name="unet",
    dataset_name="CIFAR10",
    n_trials=30,
    epochs=50,
    results_path="results/hpo",
)
```

Each trial creates a normal `Config` and runs the same `main` pipeline. Study
state is stored in SQLite, trial summaries in CSV, and resolved run artifacts
under the selected results directory. Joint studies use generation loss and
classification accuracy as separate objectives. Progressive, ensemble, and
teacher-distillation studies use separate resumable study paths. For continual
diffusion classifiers, `run_hpo(..., use_distillation=True)` enables the token
and loss search without requiring a live teacher; each trial creates its later
teachers from its own completed task snapshots.

## Supporting modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Typed settings plus YAML load/save |
| `argument_saver.py` | Keras constructor-config persistence |
| `replay_buffer.py` | Bounded seeded sample replay |
| `masked_loss.py` | Prefix-masked MAE/MSE losses |
| `lr_logger_callback.py` | Effective learning-rate logging |
| `utils.py` | Plots, GIFs, CSV/NumPy persistence, and HPO logs |

Use the public functions above for orchestration. Private helpers beginning
with `_` implement individual stages and are not stable entry points.
