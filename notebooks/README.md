# Experiment notebooks

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
