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
| Continual learning | [diffusion_transformer](continual/diffusion_transformer.ipynb) | Conditional replay buffer | Images | 20 |
| Continual learning | [dit_decoder](continual/dit_decoder.ipynb) | Conditional replay buffer | Images | 20 |
| Continual learning | [dit_encoder_decoder](continual/dit_encoder_decoder.ipynb) | Conditional replay buffer | Images | 20 |
| Continual learning | [unet](continual/unet.ipynb) | Conditional replay buffer | Images | 20 |
| Continual learning | [vae](continual/vae.ipynb) | Conditional replay buffer | Feature vectors | 20 |
| Continual learning | [dit_classifier](continual/dit_classifier.ipynb) | Joint-model replay buffer | Images | 20 |
| Continual learning | [dit_encoder_decoder_classifier](continual/dit_encoder_decoder_classifier.ipynb) | Joint-model replay buffer | Images | 20 |
| Continual learning | [unet_classifier](continual/unet_classifier.ipynb) | Joint-model replay buffer | Images | 20 |
| Continual learning | [vae_classifier](continual/vae_classifier.ipynb) | Joint-model replay buffer | Feature vectors | 20 |

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
       # Diffusion-classifier joint/continual studies only:
       use_ensemble_accuracy=False,
       ensemble_accuracy_kwargs={"weighted": True, "max_t": 128},
       # Optional runtime-only teacher for the same classifier families:
       # teacher_network=teacher,
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
history and evaluation CSV files, plots, a GIF, and TensorBoard events. The
TensorBoard event suffix lists every sampled value in alphabetical parameter
name order; the complete name-to-value mapping is also stored in the trial
config and TensorBoard text summary. Compact logs live below
`results/hpo/_tb/`. `study.db` permits resuming a study, while `trials.csv`
gives a study-level table.

DiT classifier studies sample `classifier_architecture` from `linear`,
`local_mixer`, `connection`, `cross_attention`, `cross_attention_decoder`,
`cross_attention_aggregation`, `u_shape`, and `u_vae`. Each choice uses the
model's existing classifier-layer arguments; `u_vae` additionally samples a
positive `kl_loss_coef` for its KL-enabled variational bottleneck.

The zero-inclusive `ctr_loss_coef` search applies to joint DiT and U-Net
classifiers. A positive value adds a regularizer at the final classifier depth.
It uses normal labels without a teacher; teacher-backed studies can select
`normal`, `distil`, or `both`, with hard or soft teacher targets for the latter
two modes. Accuracy coefficients are balanced across the active classifier,
distillation, and regularizer predictions through `clf_acc_coef`,
`distil_acc_coef`, and `ctr_acc_coef`.

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

Passing `teacher_network` enables conditional hard/soft distillation sampling
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
