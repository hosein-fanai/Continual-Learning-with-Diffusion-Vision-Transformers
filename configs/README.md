# Training configurations

The root `default.yaml` is a current, CLI-loadable example whose mappings provide
keyword overrides for the dataclasses in
`common.config`. `load_config(path)` converts nested mappings into a `Config`
tree; omitted sections and fields keep their dataclass defaults. Unknown keys
are rejected by the corresponding dataclass constructor.
Loading uses safe YAML constructors and rejects duplicate explicit mapping keys
at every nesting level; ordinary YAML `<<` merge overrides remain supported.

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
  optional positive finite `clipnorm`, which clips each variable's gradient
  tensor independently (it is not a global-gradient norm).
- `training`: task, ordinary/progressive fit selection, curriculum and
  epoch/validation settings, result directory, early stopping, TensorBoard,
  verbosity, weight persistence, global dtype policy, and deterministic-kernel
  selection.
- `continually_learn`: optional class count, cumulative/new-class behavior,
  fixed-buffer or generative replay controls, classifier reuse, optional
  seeded class/task scheduling, diffusion distillation/ensemble evaluation,
  task-boundary recovery, and result detail/accuracy plotting switches.
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

`training.task` accepts only `legacy`, `generation`, `joint`, `classification`,
or `continual` (case-insensitive). A path-like `training.results_path` is
normalized to text. `common.config.normalize_training_task(value)` exposes the
same validation to direct callers. Set `results_path` to `null` only when every
active runtime path can work without a directory: disable file, TensorBoard,
weight/config, and checkpoint outputs, and use interactive image display during
training.

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
    training={
        "task": "continual",
        "epochs": 20,
        "dtype_policy": "mixed_float16",
        "deterministic_ops": True,
    },
    continually_learn={
        "class_num": 10,
        "task_size": 1,
        "class_order_mode": "random",
        "task_order_mode": "fixed",
        "seed": 42,
        "use_buffer": True, 
        "buffer_kwargs": {
            "maxlen": 10_000, 
            "sample_num": 1_000, 
            "insert_num": 1_000, 
            "seed": 42,
            "strategy": "fifo"
        }
    }
)
accuracies = continually_learn(config)
```

`buffer_kwargs.strategy` selects how examples occupy a full fixed memory:
`fifo` is the backward-compatible newest-example policy, `reservoir` is
Algorithm R over the stream offered to storage, and `class_balanced` assigns
near-equal feasible class quotas with a reservoir inside each class. The
default is `fifo`. Replay checkpoints preserve the selected policy, insertion
counters, class-allocation state, retained samples, and private RNG state.
The named `baseline: reservoir_er` offers every current row selected for the
task to Algorithm R. For an insertion-count ablation instead, leave
`baseline: null`, set `strategy: reservoir`, and control `insert_num`.

`continually_learn.optimizer_steps_per_epoch` is an optional positive update
budget applied to every active continual training phase. When set, each
already-selected task pool repeats as needed and Keras executes exactly that
many steps per epoch; `null` preserves the existing finite-dataset behavior.
Use it together with fixed old/current row pools when update count is a matched
experimental control. Do not also set `training.fit_kwargs.steps_per_epoch`.

For generative replay, set `model.name` to a conditional VAE or diffusion
family and select the continual classifier with `model.classifier_name` and
`model.classifier_kwargs`. The factories compile both models from `optimizer`;
`continually_learn` trains and reports each enabled phase through
`common.train`. Fixed-buffer replay suppresses generative-model construction.
Dataset sample limits, shuffle capacity, raw-image padding, and the effective
continual seed carry into every task. `continually_learn.seed` is authoritative
and falls back to `training.seed` only when omitted. It controls schedule
resolution, split/limiting, initialization, wrappers, task shuffling, replay,
stochastic layers, generation, and ensemble noising. The final config
materializes the resolved `class_order` and `task_groups`. Automatic groups
contain `task_size` classes; `class_order_mode: random` shuffles labels before
grouping and `task_order_mode: random` shuffles whole groups while preserving
each group's contents. Explicit `task_groups` overrides automatic grouping.
`training.dtype_policy` is installed before datasets, models, layers, wrappers,
schedules, and optimizers are constructed; mixed policies also use optimizer
loss scaling. Directly supplied prebuilt repository models must already match
the requested seed and dtype policy. Typed replay-model
sections automatically use the
selected dataset's padded dimensions. Fresh and restored continual diffusion
constructors always receive raw `num_classes: null`; on restoration the wrapper
uses saved zero-based `seen_classes` to regrow the topology before loading weights.
For a diffusion replay model, the same progressive training fields run one
curriculum per continual task. A `DiffusionClassifierV2` applies the curriculum
to its generator phase and keeps its required discriminator phase on ordinary
`fit`; standalone classifiers, buffers, and VAEs do not accept the progressive
selector.

The V2 fields `clf_train_noisified_max_timesteps` and
`clf_test_noisified_max_timesteps` select clean timestep 0 with `null`, the full
horizon with `-1`, or an exclusive `[0, cap)` range with a positive integer.
Classifier caps are independent of progressive generator intervals. Direct V2
evaluation must select `generator`/`discriminator` or request both; shared
reporting evaluates both phases.

Student distillation settings are serializable model fields. A DiT classifier
sets `classifier_only_distil_token: true` together with a non-null
`clf_distil_token_type` (or shares a non-null inherited `distil_token_type`); a
UNet classifier sets `classifier_only_distil_token: true`. Set
`distil_type`, `distil_loss_coef`, `clf_acc_coef`, `distil_acc_coef`, and
`ctr_acc_coef` in its diffusion-classifier wrapper.
Token regularizer mappings accept `train_type: normal|distil|both` and
`distil_type: hard|soft`. Continual configs additionally set
`continually_learn.use_distillation: true`. Task one trains without a teacher by
default. Before each later task, the completed `snapshot_network_name`
(`raw` or `ema`) student is copied into an independent frozen teacher, then the
student is allowed to grow for the new class. EMA selection requires an
EMA-enabled wrapper. An optional live teacher can initialize task one through
`continually_learn(config, teacher_network=teacher)` or
`main(config, teacher_network=teacher)`; it is never placed in YAML. The same
lifecycle works with ordinary and progressive fitting and with both diffusion
classifier wrappers.

`distil_scope` accepts `old_classes`, `replay_only`, or the default
`current_and_replay`. Replay-only scope requires generative replay; the learner
adds row-level replay provenance to training batches, V2 retains it through
mapped preprocessing, and teacher-targeted losses/metrics select only those
rows.

With `continually_learn.save_task_checkpoints: true` (an opt-in setting), a configured
run commits `checkpoints/task-NNNN` only after a task has trained, evaluated,
and updated its matrices. The checkpoint contains
raw/EMA and classifier/replay models, the frozen next-task teacher, optimizers,
replay samples and private RNG, global/local RNG state, task cursor, resolved
schedule, and metric histories. Set `continually_learn.resume_from` to either
the checkpoint root or a committed task directory. An interruption inside a
task restarts that task from the preceding committed boundary; completed tasks
are never repeated. Long-form `epoch_metrics.csv`, `task_metrics.csv`,
`accuracy_matrices.csv`, `schedule.csv`, and `summary.csv` are written when
`reporting.save_csv` is enabled, beside the resolved config and any enabled
weights, plots, images, and GIFs. Continual TensorBoard logs use separate
task/class/phase namespaces for every epoch's generator loss and
classifier/distillation accuracies, and include final validation and test CL
summaries. When `continually_learn.use_ensemble_accuracy: true`, the ensemble
matrix is used for all derived continual metrics.

`run_hpo` accepts the same seed, dtype, deterministic-kernel, schedule,
distillation snapshot, and ensemble controls as arguments. For continual HPO,
`objective_metrics` names one metric or an ordered metric list from
`validation_continual_metrics`; `objective_directions` supplies matching
`minimize`/`maximize` directions (or directions are inferred). The default is
to maximize `final_average_accuracy`. Pass the existing study directory to
`resume_from` to validate and restore its study specification, SQLite/TPE
state, and per-trial committed task checkpoints. HPO always enables CSV and
TensorBoard reporting; continual trials retain the task/class/phase metric
namespaces described above.

Nested model dataclasses expose `kwargs() -> dict[str, Any]`; those dictionaries
are forwarded to the corresponding transformer or wrapper constructor. Use the
constructor docstrings as the authoritative valid-value reference. The
YAML files under `old/` record experiments from earlier API revisions and
contain legacy key names. They are archival records, not CLI examples or HPO
bases. New studies generate current YAML files below
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
