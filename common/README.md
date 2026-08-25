# Common APIs

`common` contains the project-level interfaces for configuration, MNIST/CIFAR
input preparation, model orchestration, continual learning, replay, Keras
serialization, losses, callbacks, plotting, and artifact persistence.

## Training architecture

Raw diffusion networks perform one denoising pass. Their wrappers own the noise
schedule, EMA network, training steps, evaluation, and iterative sampling.
`common.model.get_model` selects the correct wrapper for every supported raw
network. It also constructs the VAE and classifier families used by the shared
experiment and HPO paths.

Every config-driven experiment follows the same four public stages:

```python
from common.config import load_config
from common.dataloader import get_datasets
from common.model import get_model
from common.train import train_model, report

config = load_config("configs/my-run.yaml")
trainset, valset = get_datasets(config)
model = get_model(config)
history = train_model(config, model, trainset, valset)
report(config, history, model, trainset, valset)
```

`main(config)` is the equivalent one-call entry point. The original
compact DiT fields remain available when `model.name` is `None`; generic runs
select a family with `model.name`, `model.kwargs`, and `model.wrapper_kwargs`.

`get_datasets` records `config.dataset.trainset_len` and resolves a missing
diffusion preprocessing mode to `"standardize"`. `get_model` uses the recorded
length to resolve a missing cosine-decay length. `train_model` then records the
actual result directory in `config.training.results_path` and, when enabled,
the saved weights path in `config.model.weights_path`. Saving dynamic diffusion
weights requires a `Config` and always writes the paired `config.yaml`, even
when `save_config_=False`.

Direct `get_model(...)` calls default to a constant learning rate because no
dataset length is necessarily available. Requesting cosine decay directly
requires either `decay_steps` or `trainset_len`.

## Configuration API

`load_config(None)` returns pure dataclass defaults. It does **not** merge
`configs/default.yaml`. When a YAML path is provided, omitted fields receive
their dataclass defaults, nested mappings become typed section objects, and
unknown fields raise during dataclass construction.

```python
from common.config import Config, load_config, save_config

config = Config()
config.training.epochs = 50
save_config(config, "results/resolved-config.yaml")

loaded = load_config("results/resolved-config.yaml")
wrapper_kwargs = loaded.model.diffusion_model.kwargs()
```

Set `training.fit_method="fit_progressively"` to route a diffusion wrapper
through its existing curriculum trainer. `training.stage_tasks` is required;
the remaining stage, timestep, resolution, depth, pacing, and stopping fields
mirror `DiffusionModel.fit_progressively`. `stage_epochs` and `final_epochs`
own the progressive budget, while `fit_kwargs` carries only additional Keras
keys such as step counts. Ordinary runs keep `fit_method="fit"` and continue
to use `training.epochs`. Saving progressive weights forces a final config
rewrite with the post-growth network constructor state, making depth-grown
weights reconstructable.

`KwargsMixin.kwargs()` uses `dataclasses.asdict` and emits every field verbatim,
including `None` and defaults. It does not filter, validate, or rename keys.
`Config.continually_learn` is a typed `ContinuallyLearnConfig` section; it keeps
class-incremental and replay policy separate from shared dataset, model,
optimizer, fit, and reporting settings.

Set `reporting.evaluate_ensemble_accuracy=True` for a `DiffusionClassifier` or
`DiffusionClassifierV2` to add `ensemble_accuracy` beside the normal metrics in
every enabled raw/EMA train/validation evaluation. Pass metric options such as
`weighted`, `max_t`, `t_chunk_size`, or `seed` through
`reporting.ensemble_accuracy_kwargs`.

Distillation configuration stays YAML-safe. For a DiT classifier, enable
`classifier_only_distil_token` and select `clf_distil_token_type` (or share an
inherited `distil_token_type`); for a UNet classifier, enable
`classifier_only_distil_token`. Then set the wrapper's `distil_type`
(`"hard"` or `"soft"`), `distil_loss_coef`, and accuracy coefficients.
`cls_token_regularizer_kwargs` and
`clf_cls_token_regularizer_kwargs` accept `train_type` (`"normal"`,
`"distil"`, or `"both"`) and `distil_type` (`"hard"` or `"soft"`). The live
teacher is deliberately not a config field; pass it at runtime with
`get_model(config, teacher_network=teacher)` or
`main(config, teacher_network=teacher)`. Reports prefer `total_accuracy` when
the wrapper combines classifier, distillation, and regularizer predictions.

Depth numbering also belongs to the raw network API. Depth 0 is the embedding
stem that patchifies images and constructs/merges timestep and label
conditioning. `depth=N` then creates repeated processing depths 1 through N.
Thus `depth=0` is a stem/output-only network; it does not mean "use the first
transformer block."

## Dataset APIs

`load_mnist`, `load_fmnist`, `load_cifar10`, and `load_cifar100` return six
NumPy arrays:

```text
(x_train, y_train, x_val, y_val, x_test, y_test)
```

All loaders support class filtering, a configurable stratified training split,
optional one-hot labels, and these preprocessing values:

- `"min-max"`: one scalar minimum and maximum from the final training split;
- `"normalize"`: elementwise mean and standard deviation over training axis 0;
- `"standardize"` or `"diffusion"`: min-max scaling followed by mapping the
  training extrema to `[-1, 1]`;
- `None`, `""`, or another value: no scaling (raw train/test images are cast to
  `uint8`).

Validation and test arrays reuse training statistics without clipping, so
held-out values outside the observed training range can exceed the nominal
min-max/standardized interval.

```python
from common.dataloader import load_cifar10, get_dataset

x_train, y_train, x_val, y_val, x_test, y_test = load_cifar10(
    indices=[0, 1, 2],
    preprocess="normalize",
    onehot_labels=True,
    verbose=0,
)

dataset = get_dataset(x_train, y_train, batch_size=64)
```

Feature mode loads `features_path + ".npy"`; the archive must contain train,
validation, and test arrays, and its path must identify `mnist_`, `fmnist_`,
`cifar10_`, or `cifar100_`. `get_dataset` shuffles by default with a buffer of
10,000; pass `shuffle_buffer=0` for deterministic order. It can also apply a
feature extractor to whole batches, prefetch, pad, augment, and cache.

For generic configs, `common.dataloader.get_datasets` supports all four
datasets. It prepares image, flattened-image, or saved-feature inputs according
to the selected model and converts labels to sparse or one-hot form as required.
Continual runs return the matching loader for `continually_learn`. The learner
preprocesses one full-class array view so every task uses the same coordinate
system, then builds new `tf.data.Dataset` training, validation, and test
pipelines for each task.

## Continual-learning API

`continually_learn(config=None, **kwargs)` grows a classifier from two classes
through the requested total. It can train on only the newly introduced class or
on all seen classes, and can replay from a fixed buffer, conditional VAE, or
diffusion model. A configured call creates the loader and models through the
shared factories and ignores direct keywords:

```python
from common.config import Config
from common.learner import continually_learn

config = Config(
    dataset={"name": "cifar10", "preprocess": "min-max"},
    model={"name": "cnn", "show_network_summary": False},
    training={"task": "continual", "epochs": 20, "verbose": 1},
    continually_learn={
        "class_num": 10,
        "use_buffer": True,
        "buffer_kwargs": {
            "maxlen": 10_000,
            "sample_num": 1_000,
            "insert_num": 1_000,
            "seed": 42,
        },
    },
)
accuracies = continually_learn(config)
```

Config values map as follows: `dataset` selects/preprocesses the loader;
`model` builds the standalone classifier and optional replay model; `optimizer`
compiles them; `training` controls epochs, batching-related validation,
callbacks, verbosity, and persistence; `continually_learn` controls task/replay
policy; and `reporting` controls aggregate artifacts. A classifier-family
`model.name` creates a classifier-only run. A VAE or diffusion `model.name`
creates that replay model and uses `model.classifier_name` plus
`model.classifier_kwargs` for the standalone classifier. Buffer replay sets the
generative model to `None`. Typed VAE sections receive the selected dataset's
class count and padded input dimensions automatically. Typed diffusion sections
receive the padded dimensions; in continual mode their constructors always
receive `num_classes=None` so classes are added as labels are observed.
`model.weights_path` initializes a continual VAE replay model, or the
incremental classifier in classifier-only and buffer runs. It also restores a
continual diffusion checkpoint when the paired config contains its current raw
`num_classes` and zero-based wrapper `seen_classes`. Fresh and restored continual factory
construction still passes raw `num_classes=None`; the wrapper replays the
mapping to grow raw/EMA topology before loading the weights.

With `config=None`, all inputs are direct keywords:

```python
from common.dataloader import load_cifar10
from common.learner import continually_learn

accuracies = continually_learn(
    class_num=10,
    load_dataset_fn=load_cifar10,
    tuned_model_path="models/cifar10_dnn.keras",
    load_dataset_fn_kwargs={
        "preprocess": "normalize",
        "onehot_labels": False,
    },
    use_buffer=True,
    buffer_kwargs={
        "maxlen": 10_000,
        "sample_num": 1_000,
        "insert_num": 1_000,
        "seed": 42,
    },
)
```

The complete direct-key set is `class_num` and `load_dataset_fn` (required),
plus `load_dataset_fn_kwargs`, `remove_prev_classes`, `keep_same_model`,
`tuned_model_path`, `compile_args`, `use_loaded_opt`, `batch_size`, `epochs`,
`fit_method`, `fit_kwargs`, `use_buffer`, `buffer_kwargs`, `plot_results`,
`verbose`, `generative_model`, `generative_model_compile_args`,
`generative_model_kwargs`,
`use_generative_model_classifier`, `train_classifier_separately`,
`use_distillation`, `evaluate_ensemble_accuracy`,
`ensemble_accuracy_kwargs`, `callbacks_list`, `return_details`, and
`use_valset`; `teacher_network` is a separate runtime-only argument. Their
types, defaults, and nested dictionary keys are listed by
`help(continually_learn)`.

The loader mapping may contain `preprocess`, `onehot_labels`, `validation_ratio`,
`seed`, and, for built-in loaders, `features_path`; do not repeat `indices`,
`return_features`, or `verbose`. Buffer keys are exactly the four shown above.
Pass an already-created
`VariationalAutoencoder` or `VAEClassifier` through `generative_model`; VAE
replay requires one-hot labels. `generative_model_kwargs` controls `train_num`
and `samples_per_class`. See [`../autoencoder/README.md`](../autoencoder/README.md)
for VAE construction and resampling behavior.

For diffusion replay, raw classifier variants receive `DiffusionClassifier`;
all other supported raw diffusion models, including `DiTDecoder` and
`DiTEncoderDecoder`, receive `DiffusionModel`. An already-created diffusion
wrapper is preserved. Raw networks and wrappers passed directly must have been
initialized with `num_classes=None`. `generative_model_kwargs` uses the same
`train_num` and `samples_per_class` controls. Diffusion replay accepts every
loader preprocessing value, including `normalize`; the learner converts
diffusion data as needed and returns generated samples to the classifier's input
representation. When a raw diffusion network is passed directly,
`generative_model_compile_args` overrides the default Adam/MSE compilation. An
already-wrapped model remains compiled as provided.

For diffusion replay, direct calls may set
`fit_method="fit_progressively"` and pass the existing curriculum arguments in
`fit_kwargs`, including the required `stage_tasks`. The mapping is copied for
each continual task. V2 uses `fit_generator_progressively` for its generator
and retains the ordinary separate discriminator fit; non-diffusion replay
models reject the progressive selector.

For continual distillation, set
`continually_learn.use_distillation=True`, configure the student and wrapper as
above, and call `continually_learn(config, teacher_network=teacher)`. The flag
records serializable intent and requires the runtime teacher; the teacher object
never enters saved YAML. Distillation uses the same ordinary or progressive
fit selection, including V2's progressive-generator/ordinary-classifier split.

For dynamic diffusion, `seen_classes` is retained by reference in the
wrapper's initialization config as a zero-based target mapping, so later label discovery is immediately
serializable. Each raw transformer/U-Net `add_class()` also updates its saved
`num_classes` to the current width. A dynamic weight checkpoint is therefore a
pair: the `.weights.h5` file and the automatically updated `config.yaml`.

For `VAEClassifier`, set `use_generative_model_classifier=True` to use its
attached classifier as the continually learned model. By default that classifier
is updated only by the joint generative-model step, and task accuracy is the
VAE's reconstruction-based `clf_accuracy`; set
`train_classifier_separately=True` to run the ordinary classifier fit as an
additional step for every task and report its direct-input accuracy. The
attached classifier must be a Keras model and must already be compiled for the
separate step.

The same classifier-selection flag supports `DiffusionClassifier` and
`DiffusionClassifierV2`. For the standard wrapper,
`train_classifier_separately` must be false because its existing `fit` step
trains generation and classification jointly. For V2 it must be true; the
learner uses `fit_generator` followed by the existing `fit_discriminator` API.
Each newly observed label grows the dynamic diffusion classifier's final layer
by one output while preserving existing weights. Evaluation uses the selected
raw/EMA network's `predict_class` path.
Set `continually_learn.evaluate_ensemble_accuracy=True` to evaluate the same
task test data with timestep ensembling. Normal task scores remain in
`accuracies`; with `return_details=True`, ensemble scores are returned in
`ensemble_accuracies` and are also saved as `continual_ensemble_accuracy`.
Options are forwarded through `continually_learn.ensemble_accuracy_kwargs`.

Set `return_details=True` when an orchestration layer needs the final models,
per-task histories, and classifier/generative evaluation mappings in addition
to the accuracy list. `callbacks_list` forwards
ordinary Keras callbacks to the classifier and replay-model fits. Every enabled
classifier and generative phase is trained by `common.train.train_model` and
reported by `common.train.report`; returned task accuracies come from those test
reports, with prediction as a fallback for custom reports. All continual phases
receive newly built `tf.data.Dataset` objects for each task. Set
`use_valset=False` to disable training-time task validation explicitly; test
reporting remains enabled. The configured `max_train_samples`,
`max_val_samples`, `shuffle_buffer`, `pad`, and training seed are also applied
inside the continual task loop; padding is available for raw images, not saved
feature or pretrained/hp-tuned classifier inputs. Sample limits retain at least
one row per represented class and therefore cannot be smaller than that class
count. Configured patience/monitor settings are rebuilt for each enabled phase.

## Hyperparameter optimization

`common.hpo.run_hpo` creates a resolved `Config` for every Optuna trial, writes
it to YAML, reloads it through `load_config`, and calls `main`. Search
spaces live in `SEARCH_SPACES[task][model]`; architecture-dependent choices are
conditional so invalid tensor shapes and wrapper combinations are not sampled.

DiT classifier trials sample `classifier_architecture` from `linear`,
`local_mixer`, `connection`, `cross_attention`, `cross_attention_decoder`,
`cross_attention_aggregation`, `u_shape`, and `u_vae`. These choices reuse the
existing classifier routing, attention, scaling, and reshaping arguments. The
`u_vae` choice also samples a positive `kl_loss_coef` for its KL-enabled
flatten/unflatten bottleneck.

Joint DiT and U-Net classifier trials sample a zero-inclusive `ctr_loss_coef`.
A positive value enables a regularizer at the final classifier depth. Without
a teacher it uses normal labels; teacher-backed studies also sample `normal`,
`distil`, or `both` training and hard or soft targets for the distillation
modes. `clf_acc_coef`, `distil_acc_coef`, and `ctr_acc_coef` equally weight the
active prediction heads and remain zero for inactive heads.

V2 DiT classifier trials independently sample `clf_vars_embedding_recipe` from
`none`, `label`, `conditions`, `core`, and `notebook`, and
`clf_vars_noise_recipe` from `none`, `first`, `last`, and `last_two`. Embedding
recipes map respectively to `[]`, `[2]`, `[1, 2]`, `[0, 1, 2]`, and
`[0, 1, 2, 3]`; noise recipes map to `[]`, `[1]`, `[-1]`, and `[-2, -1]`.
The negative final IDs remain relative to the end after progressive depth
growth.

Successful trials contain the resolved config, final weights, metric CSV files,
history/sample plots, a GIF, and TensorBoard events. The study directory also
contains persistent SQLite state and `trials.csv`. Joint generation and
classification studies are multi-objective; all other studies use one scalar
objective. `run_hpo(..., use_ensemble_accuracy=True)` replaces the accuracy
feedback with post-training ensemble accuracy for joint or continual diffusion
classifier studies while retaining ordinary accuracy in their reports. Each
resolved config records the selection in `hpo.use_ensemble_accuracy` and
enables the matching reporting or continual evaluation option. See
[`../notebooks/hpo/README.md`](../notebooks/hpo/README.md).

HPO accepts the same `fit_method` selector and a `fit_kwargs` mapping.
Progressive studies are limited to diffusion families and require
`stage_tasks`; named curriculum values are written into the explicit
`TrainingConfig` fields before every trial config is serialized and reloaded.
The values must therefore be YAML-safe. Progressive trials use their own study
subdirectory and SQLite identity rather than resuming an ordinary-fit study.

Supplying `teacher_network` enables hard/soft distillation suggestions only for
joint or continual `dit_classifier`, `dit_encoder_decoder_classifier`, and
`unet_classifier` studies. Those trials use a separate `distillation`
subdirectory, keep the teacher out of trial YAML, and prefer `total_accuracy`
for ordinary classification feedback. Distillation and progressive fitting can
be enabled together.

## Supporting APIs

- `ReplayBuffer(maxlen, seed)` stores `(x, y)` pairs in a bounded deque.
  `sample_buffer_and_prepare_dataset` returns `float32` samples and `uint8`
  labels. Sampling uses a private seeded generator and does not remove entries.
- `ArgumentSaverLayer` and `ArgumentSaverModel` combine Keras bases with
  constructor-config persistence. Subclasses call
  `self._save_init_args(locals())`; mutable list/set/dict config values are
  copied.
- `MaskedLoss("mae" | "mse")` compares predictions with the matching target
  prefix, returns one loss per batch row, and supports Keras sample weighting
  and serialization.
- `LrLoggerCallback` adds the effective optimizer learning rate to epoch logs.
- `common.model` supplies the legacy CNN/DNN/Xception factories, early stopping,
  and one-class output-head expansion.
- `common.utils` supplies history/sample plotting, GIF creation, CSV/NPY
  persistence, and search logging. Each function docstring states its expected
  array shape and its display/file side effects.

All public callables have detailed docstrings; use `help(object)` for exact
types, return forms, accepted dictionary keys, state mutations, and edge-case
behavior.
