# Common APIs

`common` contains the project-level interfaces for configuration, MNIST/CIFAR
input preparation, model orchestration, continual learning, replay, Keras
serialization, losses, callbacks, plotting, and artifact persistence.

## Training architecture

The two diffusion model directories have deliberately different roles:

1. `diffusion/models/transformer` defines the raw neural network. A transformer
   receives a noised image, timestep, and label and performs one forward pass.
2. `diffusion/models/wrapper` owns the process around that network: the noise
   schedule, classifier-free label masking, EMA weights, custom training and
   evaluation steps, progressive training, and iterative sampling.
3. `common.train.get_model` constructs one object from each directory and
   compiles the wrapper. With classification enabled, it pairs
   `DiTClassifier` with `DiffusionClassifier`; otherwise it pairs
   `DiffusionTransformer` with `DiffusionModel`.

The orchestration sequence is below. With the checked-in dataclasses it is
conceptual: the call site must translate the legacy keys listed under “Current
constructor compatibility” before `get_model`, or the dataclass schema itself
must first be migrated. YAML alone cannot supply the replacement names because
unknown dataclass fields are rejected and `kwargs()` still emits legacy defaults.

```python
from common.config import load_config
from common.train import get_datasets, get_model, train_model, report

config = load_config("configs/my-run.yaml")
trainset, valset = get_datasets(config)
model = get_model(config)
history = train_model(config, model, trainset, valset)
report(config, history, model, trainset, valset)
```

`get_datasets` mutates `config.dataset.trainset_len`. `get_model` uses that
value to resolve a missing cosine-decay length. `train_model` then records the
actual result directory in `config.reporting.results_path` and, when enabled,
the saved weights path in `config.model.weights_path`.

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

`KwargsMixin.kwargs()` uses `dataclasses.asdict` and emits every field verbatim,
including `None` and defaults. It does not filter, validate, or rename keys.

### Current constructor compatibility

The compact transformer dataclasses preserve an earlier API revision. Their
following keys are not accepted by the current constructors and are forwarded
unchanged by `common.train.get_model`:

| Config key | Current constructor control |
| --- | --- |
| `project_with_cnn` | `patchify_with_cnn` |
| `pos_plug_method` | `patches_pos_merger_type` |
| `conds_plug_method` | No single alias: choose `conds_merger_type` for time/label merging and `patches_conds_merger_type` for patch/condition merging |
| `t_emb_freq_dim` | `time_freq_dim` |
| `t_emb_mlp_ratio` | `time_mlp_ratio` |
| `num_heads` | `mha_num_heads` |
| `mha_mlp_ratio` | `vit_block_mlp_ratio` |
| `use_final_cnn` | `use_refiner_cnn` |
| `final_cnn_hidden_dim` | `refiner_cnn_hidden_dim` |
| `final_cnn_residual` | `refiner_cnn_residual` |
| `prepend_cls_token` | No direct boolean alias; configure `cls_token_type`, `classifier_only_cls_token`, and/or `clf_cls_token_type` |
| `cls_token_pos_plug_method` | `cls_token_pos_merger_type` |
| `clf_dropout_rate` | `dropout_rate` |
| `lambda_` | `clf_loss_coef` |

The raw `DiffusionModelConfig` fields and the classifier masking fields other
than `lambda_` match the current wrapper signatures. Until the schema is
migrated, treat the current transformer/wrapper constructor docstrings as the
authoritative interface and do not expect the default compact config to build
without replacing legacy keys at the call site.

Depth numbering also belongs to the raw network API. Depth 0 is the embedding
stem that patchifies images and constructs/merges timestep and label
conditioning. `depth=N` then creates repeated processing depths 1 through N.
Thus `depth=0` is a stem/output-only network; it does not mean “use the first
transformer block.”

## Dataset APIs

`load_cifar10` and `load_cifar100` return six NumPy arrays:

```text
(x_train, y_train, x_val, y_val, x_test, y_test)
```

Both loaders support class filtering, an 80/20 stratified training split,
optional one-hot labels, and these preprocessing values:

- `"min-max"`: one scalar minimum and maximum from the final training split;
- `"normalize"`: elementwise mean and standard deviation over training axis 0;
- `None`, `""`, or another value: no scaling (raw train/test images are cast to
  `uint8`).

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
validation, and test arrays, and its path must contain `cifar10_` or
`cifar100_`. `get_dataset` batches without shuffling, optionally applies a
feature extractor to whole batches, prefetches, and caches.

`common.train.get_datasets` is separate: it always loads MNIST, maps images to
`float32 [batch, 28, 28, 1]` in `[-1, 1]`, drops incomplete training batches,
and optionally uses the MNIST test split as validation data.

## Continual-learning API

`continually_learn` grows a legacy classifier from two classes through the
requested total. It can train on only the newly introduced class or on all
seen classes, and can optionally replay from a fixed buffer or conditional VAE.

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

The loader mapping may contain `preprocess`, `onehot_labels`, and, for built-in
loaders, `features_path`; do not repeat `indices`, `return_features`, or
`verbose`. Buffer keys are exactly the four shown above. VAE replay instead
uses `vae_init_kwargs` (the `VariationalAutoencoder` constructor controls except
`conditioned` and `class_num`) and `vae_kwargs` with `train_num` and
`samples_per_class`. VAE replay requires one-hot labels. See
[`../autoencoder/README.md`](../autoencoder/README.md) for the nested
`hiddens_kwargs` contract and resampling behavior.

## Supporting APIs

- `ReplayBuffer(maxlen, seed)` stores `(x, y)` pairs in a bounded deque.
  `sample_buffer_and_prepare_dataset` returns `float32` samples and `uint8`
  labels. Sampling does not remove entries.
- `ArgumentSaverLayer` and `ArgumentSaverModel` combine Keras bases with
  constructor-config persistence. Subclasses call
  `self._save_init_args(locals())`; mutable list/set/dict config values are
  copied.
- `MaskedLoss("mae" | "mse")` compares predictions with the first matching
  target columns.
- `LrLoggerCallback` adds the effective optimizer learning rate to epoch logs.
- `common.model` supplies the legacy CNN/DNN/Xception factories, early stopping,
  and one-class output-head expansion.
- `common.utils` supplies history/sample plotting, GIF creation, NPY persistence,
  and search logging. Each function docstring states its expected array shape
  and its display/file side effects.

All public callables have detailed docstrings; use `help(object)` for exact
types, return forms, accepted dictionary keys, state mutations, and edge-case
behavior.
