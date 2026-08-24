# Training configurations

YAML files in this directory provide keyword overrides for the dataclasses in
`common.config`. `load_config(path)` converts nested mappings into a `Config`
tree; omitted sections and fields keep their dataclass defaults. Unknown keys
are rejected by the corresponding dataclass constructor.

The top-level sections are:

- `dataset`: MNIST, Fashion-MNIST, or CIFAR selection; batching,
  preprocessing, class filtering, optional features, and smoke-test limits.
- `model`: set `name` for the generic model factory, then supply constructor
  values through `kwargs`, wrapper values through `wrapper_kwargs`, and an
  optional attached/continual classifier through `classifier_*`. Leaving
  `name: null` selects the original typed DiT fields. When `kwargs` is empty,
  the field matching `name` supplies typed settings (for example `unet`,
  `dit_decoder`, `dit_encoder_decoder`, `variational_autoencoder`, or
  `vae_classifier`). The same rule applies to a named diffusion wrapper when
  `wrapper_kwargs` is empty.
- `optimizer`: optimizer family, learning rate/schedule, decay, momentum, and
  optional clipping.
- `training`: task, epoch/validation settings, result directory, early
  stopping, TensorBoard, verbosity, and weight persistence.
- `continually_learn`: optional class count, cumulative/new-class behavior,
  fixed-buffer or generative replay controls, classifier reuse, optional
  diffusion ensemble evaluation, and result detail/accuracy plotting switches.
- `reporting`: history plots/CSV, final sample controls, and train/validation
  evaluation switches, including optional raw/EMA ensemble accuracy.
- `hpo`: resolved trial metadata and selected accuracy feedback signal,
  normally written by `common.hpo`.

```python
from common.config import Config, load_config, save_config

config = Config(
    dataset={"name": "cifar10", "preprocess": "diffusion"}, 
    model={"name": "unet", "kwargs": {"widths": [32, 64, 96]}}, 
    training={"task": "generation", "epochs": 50}
)
save_config(config, "results/resolved-config.yaml")
config = load_config("results/resolved-config.yaml")
```

A classifier-only continual configuration is equally direct:

```python
from common.config import Config
from common.learner import continually_learn

config = Config(
    dataset={"name": "cifar10", "preprocess": "min-max"}, 
    model={"name": "cnn", "show_network_summary": False}, 
    training={"task": "continual", "epochs": 20}
    continually_learn={
        "class_num": 10, 
        "use_buffer": True, 
        "buffer_kwargs": {
            "maxlen": 10_000, 
            "sample_num": 1_000, 
            "insert_num": 1_000, 
            "seed": 42
        }
    }
)
accuracies = continually_learn(config)
```

For generative replay, set `model.name` to a conditional VAE or diffusion
family and select the continual classifier with `model.classifier_name` and
`model.classifier_kwargs`. The factories compile both models from `optimizer`;
`continually_learn` trains and reports each enabled phase through
`common.train`. Fixed-buffer replay suppresses generative-model construction.
Dataset sample limits, shuffle capacity, raw-image padding, and the training
seed carry into every task. Typed replay-model sections automatically use the
selected dataset's padded dimensions. Continual diffusion constructors always
receive `num_classes: null`; their class vocabulary grows as labels are seen.

Nested model dataclasses expose `kwargs() -> dict[str, Any]`; those dictionaries
are forwarded to the corresponding transformer or wrapper constructor. Use the
constructor docstrings as the authoritative valid-value reference. The
`mnist_config copy*.yaml` files and `default.yaml` record experiments from
earlier API revisions and contain legacy key names. They are archival records,
not HPO bases. New studies generate current YAML files below
`results/hpo/<task>/<model>/<dataset>/configs/` and reload every one before
training.

The YAML examples are experiment inputs, not an additional default layer:
`load_config(None)` uses only dataclass defaults. For new files, prefer the
generic `model.name`/`model.kwargs` path or save a `Config` object directly.

`project_tag: null` lets the image callback use only a timestamp. A non-null
tag is appended to that timestamp. `weights_path: null` starts with newly
initialized weights; otherwise it must identify a Keras-compatible weights
file for the selected architecture. Continual VAE replay uses it to initialize
the replay model; classifier-only and fixed-buffer continual runs use it to
initialize the incremental classifier and visible head columns. For continual
diffusion, pass an already initialized dynamic wrapper so its `seen_classes`
mapping and grown layers remain aligned.
