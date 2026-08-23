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
| Classification | [dnn](classification/dnn.ipynb) | Dense baseline | Feature vectors | 30 |
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

For joint or continual `dit_classifier`,
`dit_encoder_decoder_classifier`, and `unet_classifier` studies, set
`use_ensemble_accuracy=True` to use validation/task ensemble accuracy as the
Optuna feedback signal. Ordinary accuracy is still reported. These trials use
an `ensemble_accuracy` subdirectory so they cannot mix with an existing normal
accuracy study.

For fair comparisons, keep the dataset, seed, trial count, epoch budget, and
continual replay-budget candidate set fixed across competing model families. Diffusion and
joint studies are substantially more expensive than CNN, DNN, or VAE studies;
run a small smoke study before committing to all 30 trials. A TensorFlow
out-of-memory trial is recorded as failed and the study continues.

The notebooks can be regenerated after intentional template changes with
`python notebooks/hpo/generate_notebooks.py`.
