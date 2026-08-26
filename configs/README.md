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
- `training`: task, ordinary/progressive fit selection, curriculum and
  epoch/validation settings, result directory, early stopping, TensorBoard,
  verbosity, and weight persistence.
- `continually_learn`: optional class count, cumulative/new-class behavior,
  fixed-buffer or generative replay controls, classifier reuse, optional
  diffusion distillation/ensemble evaluation, and result detail/accuracy
  plotting switches.
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

`save_config` keeps its full-output behavior by default. Pass `shorten=True` to
write only values that differ from their dataclass defaults; omitted values are
restored normally by `load_config`.

Diffusion wrappers can use their existing progressive trainer directly from
the same config tree:

```python
config = Config(
    dataset={"name": "cifar10", "preprocess": "diffusion"},
    model={"name": "unet"},
    training={
        "task": "generation",
        "fit_method": "fit_progressively",
        "stage_tasks": "timesteps_only",
        "stages_num": 4,
        "stages_verbose": True,
        "stage_epochs": 5,
        "final_epochs": 5,
        "timestep_clustering_type": "log_snr",
        "pacing_type": "fixed",
    },
)
```

`fit_method` is either `"fit"` (the default) or `"fit_progressively"`.
Progressive mode requires `stage_tasks`; `stages_num`, `timestep_boundaries`,
`resolutions`, and `depths` describe the stages. `stage_epochs` and
`final_epochs` replace the ordinary `epochs` budget for the diffusion phase.
`stages_verbose`, `timestep_clustering_type`, `pacing_type`,
`earlystopping_type`, `progressive_monitor`, `progressive_patience`,
`min_delta`, and `stopper_mode` map to the identically purposed wrapper
arguments. Extra Keras fit values such as `steps_per_epoch` and
`validation_steps` belong in `training.fit_kwargs`; data, callbacks,
verbosity, and epoch counters remain owned by `train_model`.
When progressive weights are saved, `train_model` rewrites the final config
even if `save_config_=False` and records the post-growth network constructor
state. A depth curriculum's YAML and weights therefore rebuild the same final
topology.

A classifier-only continual configuration is equally direct:

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
selected dataset's padded dimensions. Fresh and restored continual diffusion
constructors always receive raw `num_classes: null`; on restoration the wrapper
uses saved zero-based `seen_classes` to regrow the topology before loading weights.
For a diffusion replay model, the same progressive training fields run one
curriculum per continual task. A `DiffusionClassifierV2` applies the curriculum
to its generator phase and keeps its required discriminator phase on ordinary
`fit`; standalone classifiers, buffers, and VAEs do not accept the progressive
selector.

Student distillation settings are serializable model fields. A DiT classifier
sets `classifier_only_distil_token: true` together with a non-null
`clf_distil_token_type` (or shares a non-null inherited `distil_token_type`); a
UNet classifier sets `classifier_only_distil_token: true`. Set
`distil_type`, `distil_loss_coef`, `clf_acc_coef`, `distil_acc_coef`, and
`ctr_acc_coef` in its diffusion-classifier wrapper.
Token regularizer mappings accept `train_type: normal|distil|both` and
`distil_type: hard|soft`. Continual configs additionally set
`continually_learn.use_distillation: true`. Task one trains without a teacher by
default. Before each later task, the completed raw student is copied into an
independent frozen teacher, then the student is allowed to grow for the new
class. An optional live teacher can initialize task one through
`continually_learn(config, teacher_network=teacher)` or
`main(config, teacher_network=teacher)`; it is never placed in YAML. The same
lifecycle works with ordinary and progressive fitting and with both diffusion
classifier wrappers.

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
initialize the incremental classifier and visible head columns. Continual
diffusion uses it with the paired config's current raw `num_classes` and wrapper
zero-based `seen_classes`; the wrapper rebuilds the grown raw/EMA topology before loading
the checkpoint. Dynamic diffusion weight saving requires a `Config` and writes
this paired `config.yaml` even when `save_config_=False`. Progressive weight
saving follows the same final-rewrite rule so permanent depth additions are
represented by the saved constructor mapping.
