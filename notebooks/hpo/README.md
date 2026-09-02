# Hyperparameter optimization notebooks

These notebooks are thin, reproducible entry points to the shared
`common.hpo` API. Each notebook explains one scientifically valid model/task
pair, exposes the same editable constants, displays its constrained search
space, runs the study, and reports the best trial or Pareto front. Outputs are
intentionally empty in version control.

## Notebook matrix

| Task | Notebook | Model role | Representation | Default epochs |
| --- | --- | --- | --- | ---: |
| Generation | [diffusion_transformer](generation/diffusion_transformer.ipynb) | Conditional DiT generator | Images | 50 |
| Generation | [dit_decoder](generation/dit_decoder.ipynb) | Standalone conditional DiT decoder | Images | 50 |
| Generation | [dit_encoder_decoder](generation/dit_encoder_decoder.ipynb) | Conditional DiT encoder-decoder | Images | 50 |
| Generation | [unet](generation/unet.ipynb) | Conditional convolutional generator | Images | 50 |
| Generation | [vae](generation/vae.ipynb) | Variational generator | Flattened images | 30 |
| Generation + classification | [dit_classifier](joint/dit_classifier.ipynb) | Joint DiT generator/classifier | Images | 50 |
| Generation + classification | [dit_encoder_decoder_classifier](joint/dit_encoder_decoder_classifier.ipynb) | Joint DiT encoder-decoder/classifier | Images | 50 |
| Generation + classification | [unet_classifier](joint/unet_classifier.ipynb) | Joint U-Net generator/classifier | Images | 50 |
| Generation + classification | [vae_classifier](joint/vae_classifier.ipynb) | Joint variational generator/classifier | Flattened images | 30 |
| Classification | [cnn](classification/cnn.ipynb) | Convolutional baseline | Images | 30 |
| Classification | [dnn](classification/dnn.ipynb) | Dense baseline | Flattened images | 30 |
| Classification | [pretrained](classification/pretrained.ipynb) | Xception transfer learning | Images | 30 |
| Continual learning | [cnn](continual/cnn.ipynb) | Classifier-only sequential/cumulative/replay baseline | Images | 20 |
| Continual learning | [dnn](continual/dnn.ipynb) | Classifier-only sequential/cumulative/replay baseline | Flattened images | 20 |
| Continual learning | [pretrained](continual/pretrained.ipynb) | Classifier-only sequential/cumulative/replay baseline | Images | 20 |
| Continual learning | [diffusion_transformer](continual/diffusion_transformer.ipynb) | Conditional replay buffer | Images | 20 |
| Continual learning | [dit_decoder](continual/dit_decoder.ipynb) | Conditional replay buffer | Images | 20 |
| Continual learning | [dit_encoder_decoder](continual/dit_encoder_decoder.ipynb) | Conditional replay buffer | Images | 20 |
| Continual learning | [unet](continual/unet.ipynb) | Conditional replay buffer | Images | 20 |
| Continual learning | [vae](continual/vae.ipynb) | Conditional replay buffer | Feature vectors | 20 |
| Continual learning | [dit_classifier](continual/dit_classifier.ipynb) | Joint-model replay buffer | Images | 20 |
| Continual learning | [dit_encoder_decoder_classifier](continual/dit_encoder_decoder_classifier.ipynb) | Joint-model replay buffer | Images | 20 |
| Continual learning | [unet_classifier](continual/unet_classifier.ipynb) | Joint-model replay buffer | Images | 20 |

The API-only continual model name `diffusion_classifier` searches those last
three diffusion-classifier families in one conditional study. Family-specific
parameter names are prefixed, so Optuna can compare DiT, encoder-decoder DiT,
and U-Net candidates without incompatible conditional distributions.

Continual `vae_classifier` search is intentionally omitted. Its attached
classifier is a fixed full-class, raw-input branch independent of the VAE
latent path, so tuning its loss weight would neither define a growing
class-incremental head nor improve the replay representation. Continual VAE
studies instead use the ordinary conditional VAE with the learner's expanding
external DNN.

## Execution notes

1. Start Jupyter from the repository root with the `tf_env` kernel available.
2. Open one notebook and edit only its setup constants as needed. The defaults
   use `CIFAR10`, 30 trials, and seed 42.
3. Inspect `SEARCH_SPACES[TASK][MODEL]` before starting a study. Spaces are
   task-specific and keep divisibility, routing, conditioning, and tensor-shape
   constraints valid.
4. Run the study cell. It calls only:

   ```python
   run_hpo(
       task=TASK,
       model_name=MODEL,
       dataset_name=DATASET,
       n_trials=N_TRIALS,
       epochs=EPOCHS,
       seed=SEED,
       results_path=RESULTS_PATH,
       # Use model_name="diffusion_classifier" to search every diffusion
       # classifier family. Enable the previous-task teacher lifecycle with:
       use_distillation=True,
       # Optional bounded plumbing-study controls (sealed in study_spec.json):
       max_train_samples=512,
       max_val_samples=256,
       n_startup_trials=2,
       search_space_overrides={"timesteps": [250], "test_steps": [10, 20]},
       # Diffusion-classifier joint/continual studies only:
       use_ensemble_accuracy=False,
       ensemble_accuracy_kwargs={"weighted": True, "max_t": 128},
       # Optional runtime-only teacher for the same classifier families:
       # teacher_model=teacher,
       # Optional diffusion curriculum:
       fit_method="fit_progressively",
       fit_kwargs={
           "stage_tasks": "timesteps_only",
           "stages_num": 4,
           "stage_epochs": 5,
           "final_epochs": 5,
       },
   )
   ```

5. The final cell displays the trial table and either the best single-objective
   trial or the Pareto-optimal joint trials. Study artifacts are written below
   `results/hpo/<task>/<model>/<dataset>/` by default.

Each successful trial saves its resolved YAML config, final model weights,
history and evaluation CSV files, plots, available trajectory GIFs, and
TensorBoard events. The
TensorBoard event suffix lists every sampled value in alphabetical parameter
name order; the complete name-to-value mapping is also stored in the trial
config and TensorBoard text summary. Compact logs live below
`results/hpo/_tb/`. `study.db` permits resuming a study, while `trials.csv`
gives a study-level table.

Optuna feedback comes from the post-training validation evaluation of the same
saved/restored model state, not from a historical best or the last row of the
pre-restoration Keras history. Diffusion objectives use the configured raw or
EMA branch; ordinary classifiers and VAEs use `valset_eval`. The semantic
defaults are generation loss, a generation/accuracy Pareto pair for joint
models, validation accuracy for standalone classifiers, and validation
`final_average_accuracy` for continual studies. `objective_metrics` can name
other scalar validation metrics with matching or inferred directions. Test-set
metrics are never HPO feedback, and trials are scored only after the complete
fit; there is no intermediate pruning contract.

Reverse-process `test_steps`, CFG scale, and eta are sampled only in continual
diffusion studies, where generated examples enter later replay tasks and can
change the objective. Generation and joint studies optimize validation losses,
so these otherwise inert dimensions are fixed to at most 50 steps, CFG 4, and
eta 0. Final visualization settings remain reporting choices rather than HPO
parameters.

`swap_noise_image=True` is an immutable wrapper override for direct x0/VAE
prediction. These studies fix `image_loss_coef=0`, tune a positive
`kl_loss_coef` unless the override supplies one, and optimize reconstruction
plus weighted main-latent KL. They do not request denoising GIFs. The immutable
main-network topology must also be sampleable: DiT, DiT encoder-decoder, and
DiT encoder-decoder-classifier studies require
`model_overrides={"vit_block_ids": [1], "reshaper_ids_dict": {1: "flatten",
2: "unflatten"}, "reshaper_kwargs": {"add_kl": True}}`; U-Net and U-Net
classifier studies require
`model_overrides={"reshaper_kwargs": {"add_kl": True}}`. Standalone
`dit_classifier` and `dit_decoder` x0 studies are rejected because their raw
call contracts cannot resume main-latent decoding. Noise distillation is also
incompatible with x0 prediction.

DiT classifier studies sample `classifier_architecture` from `linear`,
`local_mixer`, `connection`, `cross_attention`, `cross_attention_decoder`,
`cross_attention_aggregation`, `u_shape`, and `u_vae`. Each choice uses the
model's existing classifier-layer arguments; `u_vae` additionally samples a
positive `kl_loss_coef` for its KL-enabled variational bottleneck. On grids
divisible by four, `u_multilevel_vae` adds two independent variational scales;
it is not advertised as a conditional-prior hierarchical VAE.

The zero-inclusive `ctr_loss_coef` search applies to joint DiT and U-Net
classifiers. A positive value adds a regularizer at the final classifier depth.
It uses normal labels without a teacher; teacher-backed studies can select
`normal`, `distil`, or `both`, with hard or soft teacher targets for the latter
two modes. Accuracy coefficients are balanced across the active classifier,
distillation, and regularizer predictions through `clf_acc_coef`,
`clf_distil_acc_coef`, and `ctr_acc_coef`.

With `task_size=1`, the schedule starts from one class and then adds one class
per task; `task_size=2` starts from two and adds two, including a shorter final
task when necessary. A one-output softmax has trivial 100% accuracy and zero
classification gradient, so singleton classifier searches exclude a new-only
sequential protocol unless positive replay exists. The first acquisition row
is reported as unavailable. Consequently, forgetting and backward-transfer
cannot be HPO objectives for a singleton first task; final validation average
accuracy remains well defined.

Dataset, split, class/task schedule, seed, epoch/trial budget, and objective are
sealed experimental controls rather than hyperparameters: changing them across
trials would make validation scores incomparable. The umbrella continual
diffusion-classifier space instead searches the model family, architecture,
optimizer, diffusion/noise schedule, raw/EMA evaluation and teacher branches,
V1/V2 wrapper behavior, hard/soft distillation temperature and scope, optional
noise distillation, and continual/replay policy. `search_space_overrides` may
bound expensive dimensions for a plumbing study, but it becomes part of the
immutable study identity and cannot be changed on resume.

Umbrella categorical overrides are checked against each active conditional
domain, and sampling steps must not exceed training timesteps. Raw
`model_overrides`, `wrapper_overrides`, and a live external teacher require an
exact model family because their tensor topology and diffusion process must
match. The umbrella instead supports teacher-free continual distillation from
the selected raw/EMA snapshot of the preceding task.

For V2 DiT classifiers, `clf_vars_embedding_recipe` independently selects
`none`, `label`, `conditions`, `core`, or `notebook`, while
`clf_vars_noise_recipe` selects `none`, `first`, `last`, or `last_two`. These
map to embedding IDs `[]`, `[2]`, `[1, 2]`, `[0, 1, 2]`, `[0, 1, 2, 3]` and
noise-part IDs `[]`, `[1]`, `[-1]`, `[-2, -1]`, respectively. Negative final
IDs continue to refer to the final stages when progressive fitting grows the
network.

For joint or continual `dit_classifier`,
`dit_encoder_decoder_classifier`, and `unet_classifier` studies, set
`use_ensemble_accuracy=True` to use validation/task ensemble accuracy as the
Optuna feedback signal. Ordinary accuracy is still reported. These trials use
an `ensemble_accuracy` subdirectory so they cannot mix with an existing normal
accuracy study.

Set `fit_method="fit_progressively"` only for diffusion model families and
provide `stage_tasks` in `fit_kwargs`. The mapping accepts the existing
`DiffusionModel.fit_progressively` controls: stage count and verbosity, stage
and final epoch budgets, timestep boundaries and clustering, resolutions,
depths, pacing and early-stopping settings. Remaining Keras keys such as
`steps_per_epoch` and `validation_steps` are forwarded unchanged. Progressive
`DiffusionClassifierV2` trials apply the curriculum to the generator and then
run their existing ordinary discriminator phase. Values must be YAML-safe
because every trial config is serialized and reloaded before training.

Progressive studies use a `fit_progressively` subdirectory and a separate
SQLite study name, so resuming them cannot mix their trials with ordinary-fit
studies. As with other HPO settings such as epoch count, use a different
`results_path` when comparing different progressive curricula.

Passing `teacher_model` enables conditional hard/soft distillation sampling
for joint or continual `dit_classifier`, `dit_encoder_decoder_classifier`, and
`unet_classifier` studies. The student token/head and wrapper loss settings are
written to each trial config, while the live teacher is passed directly to
`main` and never serialized. Distillation trials use their own `distillation`
study directory/name and prefer `total_accuracy`; they can also use
`fit_progressively`.

With ordinary `fit`, `epochs` is the per-fit budget. With progressive fitting,
the diffusion budget is the epochs actually run across all curriculum stages:
at most `stage_epochs * number_of_stages + final_epochs` under fixed pacing.
The `epochs` argument remains the budget for ordinary classifier phases in a
continual study and must therefore still be positive. It also retains the
existing role of sizing a sampled cosine learning-rate schedule; set the HPO
epoch value to a progressive budget representative of the curriculum when
cosine decay is enabled.

For fair comparisons, keep the dataset, seed, trial count, epoch budget, and
continual replay-budget candidate set fixed across competing model families. Diffusion and
joint studies are substantially more expensive than CNN, DNN, or VAE studies;
run a small smoke study before committing to all 30 trials. A TensorFlow
out-of-memory trial is recorded as failed and the study continues.

The notebooks can be regenerated after intentional template changes with
`python notebooks/hpo/generate_notebooks.py`.
