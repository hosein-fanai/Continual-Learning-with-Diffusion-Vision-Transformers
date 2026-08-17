# Saved model artifacts

This directory stores trained weights and historical Keras models. It is
distinct from `diffusion/models/`, which contains Python architecture source.

- `DiT/` and `DiTCLF/` pair experiment configuration fragments with HDF5
  weights.
- root-level `*.weights.h5` files are Keras weight checkpoints for named
  experiments.
- `hyperas/` contains older CIFAR hyperparameter-search models and logs.

Load a current diffusion checkpoint only into an architecture created with the
same constructor configuration:

```python
network = DiffusionTransformer(...)
model = DiffusionModel(network, ...)
model.load_weights("models/DiT/model.weights.h5")
```

Keras weights do not encode every Python constructor choice. Treat the nearby
YAML as provenance, and note that historical configuration keys may predate
the current APIs. Do not confuse this artifact directory with the import path
`diffusion.models`.
