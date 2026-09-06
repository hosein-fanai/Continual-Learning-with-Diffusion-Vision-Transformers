# Experiment notebooks

The maintained small execution entry point is
[`thesis_development.ipynb`](thesis_development.ipynb). Start a fresh `tf_env`
kernel and run every cell. Its default runs real two-task optimization and replay
on synthetic MNIST-shaped arrays, uses a seeded split of training rows for
validation, checks the saved full matrix and reloads the label-free classifier.
The notebook supports semantic consolidation and gist memory; changing its
explicit data mode to CIFAR starts a real development experiment. Preparation
and synthetic checks do not establish benchmark performance.

[`NOTEBOOK_STATUS.json`](NOTEBOOK_STATUS.json) records the maintained entry,
24 generated HPO templates and 74 historical archives, excluding editor
checkpoints. Historical notebooks now begin with an archival notice; original
code cells, outputs and execution metadata are preserved. Fifty-five have an
explicit notice that official test arrays supplied validation. Selection from
those displays makes the observations development data, permanently; these
repairs do not confer untouched-test status.

The named `cifar10 main.ipynb` and `cifar100 main.ipynb` are historical archives
with obsolete APIs, not executable replacements for the maintained notebook.
Do not use them to produce current thesis results. The same rule applies to
other archived notebooks even when some individual cells still run.

The following describes the historical layout and helper behavior.

These notebooks explore MNIST/CIFAR continual learning, variational
autoencoders, U-Net diffusion, diffusion transformers, and joint DiT
classification.

When a notebook starts with its working directory set to `notebooks/`, the
local helper can initialize repository imports:

```python
import init
```

That import moves the process one level upward and applies the project's
TensorFlow GPU-memory initialization. If the working directory is already the repository root, import
`autoencoder`, `common`, and `diffusion` directly instead.

For reproducible experiments, move stable settings into a YAML file under
`configs/`, call `common.config.load_config`, and use `common.train` or the
documented model APIs. Notebook outputs can be large and may embed results from
older constructor versions.

The reproducible HPO entry points are indexed in [`hpo/README.md`](hpo/README.md).
They are intentionally thin: each displays a constrained scientific search
space and calls `common.hpo.run_hpo`, which writes/reloads trial configs and
uses the standard `common.train` pipeline.
