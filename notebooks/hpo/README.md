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
       search_space_overrides={"timesteps": [500], "test_steps": [20, 50]},
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
history and evaluation CSV files, plots, available trajectory GIFs, and
TensorBoard events. The
TensorBoard event suffix lists every sampled value in alphabetical parameter
name order; the complete name-to-value mapping is also stored in the trial
config and TensorBoard text summary. Compact logs live below
`results/hpo/_tb/`. `study.db` permits resuming a study, while `trials.csv`
gives a study-level table.

Optuna feedback comes from the post-training validation evaluation of the same
saved/restored model state, not from a historical best or the last row of the
pre-restoration Keras history. Diffusion objectives use the EMA branch;
ordinary classifiers and VAEs use `valset_eval`. The semantic
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

`swap_noise_image=True` is an immutable wrapper override for direct x_t/VAE
prediction. These studies fix `image_loss_coef=0`, tune `kl_loss_coef` unless
the override supplies one, and optimize `x_t` prediction plus weighted
main-latent KL. HPO validates the variational structure and ratio-list
cardinality but passes a fixed KL coefficient through unchanged. These studies
do not request denoising GIFs. DiT, DiT encoder-decoder, and DiT
encoder-decoder-classifier studies require an immutable main-network topology;
a single-level example is
`model_overrides={"vit_block_ids": [1, 4], "use_decoder_ids": [4],
"reshaper_ids_dict": {2: "flatten", 3: "unflatten"}, "reshaper_kwargs":
{"add_kl": True}}`; U-Net and U-Net classifier studies require
`model_overrides={"reshaper_kwargs": {"add_kl": True}}`. Standalone
`dit_classifier` and `dit_decoder` x_t studies are rejected because their raw
call contracts cannot resume main-latent decoding. Noise distillation is also
incompatible with x_t prediction.

An x_t DiT depth is fixed by `model_overrides["depth"]`, when supplied, or
derived from the greatest positive stage ID in the fixed topology. An explicit
depth must cover every referenced stage, and neither form creates a sampled
2--6 depth parameter. Every topology contains actual encoder computation
(`vit`, local mixer, or downsampling) before its bridge and decoder computation
(`vit`, local mixer, or upsampling) after it. Transformer/local-mixer stages
stay out of the bridge, all downsampling precedes it, and all upsampling and
decoder blocks follow it. Multiple adjacent pairs must additionally form one
uninterrupted central bridge. This is the minimal Copy35-style routing pattern
(depth derives to 15):

```python
model_overrides = {
    "vit_block_ids": [1, 3, 5, 13, 15],
    "use_decoder_ids": [13, 15],
    "connection_ids_dict": {8: [3], 10: [1], 12: [7]},
    "cross_attention_ids_dict": {13: [9], 15: [11]},
    "downsample_ids": [2, 4],
    "reshaper_ids_dict": {
        6: "flatten", 7: "unflatten",
        8: "flatten", 9: "unflatten",
        10: "flatten", 11: "unflatten",
    },
    "reshaper_kwargs": {
        "add_kl": True,
        "latent_dim_ratio": [1 / 32, 1 / 128, 1 / 256],
    },
    "upsample_ids": [12, 14],
}
```

DiT classifier studies focus `classifier_architecture` on `linear`,
`connection`, and `u_shape`. On grids divisible by four, `u_vae` and
`u_multilevel_vae` are also available. The variational templates follow the
network order demonstrated in `DiT mini copy 35.ipynb`: all encoder blocks and
downsampling operations precede a central variational bridge, every
flatten/unflatten pair occurs inside that bridge, and all decoder blocks and
upsampling operations follow it. This makes the encoder/latent/decoder
boundary and the occurrence order used by the ratio list unambiguous.
All-depth feature aggregation always projects the concatenated main features
to the configured classifier width before entering any of these templates.
The multilevel template also retains the mandatory terminal `-1: [-1]`
classifier connection used to feed its final depth to the classification head.

`reshaper_kwargs["latent_dim_ratio"]` is always a list in flatten occurrence
order, with exactly one member per flatten/unflatten pair. HPO samples an
independent absolute latent width of 16, 32, or 64 for each occurrence and
divides it by that occurrence's flattened width. It does not repeat one ratio:
doing so made the later, larger feature maps produce latent widths of 128 and
256 in the incomplete multi-level notebook run and caused an extreme KL and
parameter increase. These models are variational bottlenecks, not
conditional-prior hierarchical VAEs.

The zero-inclusive `ctr_loss_coef` search applies to joint DiT and U-Net
classifiers. A positive value adds a regularizer at the final classifier depth,
or at the central latent depth for a U-VAE classifier.
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
optimizer, diffusion/noise schedule, teacher snapshot branches,
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

V1 classifier trials compare the ordinary conditional classifier prediction
with the notebook's unconditional branch at CFG scale 1. Timestep masking is
limited to 50%, 70%, or 90%; null, timestep, combined, and unmasked recipes
remain available.

For V2 DiT classifiers, `clf_vars_recipe` selects one of three coupled,
notebook-supported variable assignments: `separate` maps to embedding/noise
IDs `([], [])`, `conditions` maps to `([1, 2], [])`, and `notebook` maps to
`([0, 1, 2, 3], [1])`. Coupling the assignments avoids unsupported Cartesian
combinations. Classifier input noising is limited to clean inputs or caps of
64 and 256 timesteps when below the selected diffusion horizon. V2 performs a
generator fit and then a separate classifier fit, each using `epochs`; compare
V1 and V2 only with that difference in compute budget made explicit.

## Notebook-informed search-space limits

Search-space version 9 uses the stored notebook outputs as practical anchors,
not as definitive benchmarks. Most comparisons are single stochastic runs,
some notebooks are incomplete, and the selected legacy CIFAR VAE objective is
circular; consequently, failed or weak runs are used to exclude implausible
regions rather than to assert a precise optimum.

For DiT, the default space emphasizes four-head capacities, MLP ratios 2 and
4, MSE, adaptive normalization, global 2D sinusoidal positions, 500 or 1,000
timesteps, and Adam/AdamW. Ordinary generation/joint fitting uses cosine
decay; continual and progressive fits retain a constant rate because their
complete update count is not known up front. The plain backbone is joined by
the compact symmetric two-level feature-skip U-DiT on compatible grids.
Resampling positions are retained. Continual sampling concentrates on 20, 50,
or 100 reverse steps, CFG 2.5–5, and eta 0 or 1.

The CNN space includes the saved CIFAR stage shape
`(64, 128, 128, 256)` with depths `(1, 2, 2, 1)`, dropout 0.15 and 0.20, max
intermediate pooling, and global-average pooling. Transfer learning includes
the 12-layer Xception tail used by both CIFAR notebooks. Dense classifier
templates include a linear head and the saved two-layer widths, but the legacy
DNN results used frozen Xception features and must not be interpreted as
evidence for the same learning rates on raw flattened pixels.

Reservoir replay independently samples capacity (2,500/5,000/10,000), replay
sample count (500/1,000/2,500), and insertion count (500/1,000). Capacity is a
storage limit; tying it to both per-update counts excluded the saved
10,000-capacity/1,000-sample setup and needlessly coupled memory with compute.

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
