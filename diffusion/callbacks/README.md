# Diffusion callbacks

These Keras callbacks add progressive-stage stopping, qualitative sampling, and
raw-network validation to the diffusion wrappers. They rely on project-specific
wrapper attributes and are not all compatible with an arbitrary Keras model.

## `BatchLossPlateau`

`BatchLossPlateau` monitors a scalar after every training batch and sets
`model.stop_training=True` after sustained non-improvement:

```python
from diffusion.callbacks.batch_loss_plateau import BatchLossPlateau

stopper = BatchLossPlateau(
    monitor="noise_loss",
    patience=200,
    min_delta=1e-4,
)
model.fit(dataset, callbacks=[stopper])
```

An improvement is strictly `current < best - min_delta`. Missing monitor keys
are ignored. The stop condition is `wait > patience`, so `patience=200` stops
after 201 consecutive non-improving batches. `fit_progressively` creates a new
callback for each stage, giving each stage independent `best=inf` and `wait=0`
state.

## `ImageGeneratorCallback`

This callback reads `model.test_steps`, `model.test_cfg_scale`, and
`model.test_eta`, calls `model.sample(...)` at epoch end, and renders the result.
The current validation permits two configurations.

Display only:

```python
from diffusion.callbacks.image_generator_callback import ImageGeneratorCallback

preview = ImageGeneratorCallback(show_images=True)
```

Save PNG and GIF artifacts, optionally also display them:

```python
artifacts = ImageGeneratorCallback(
    show_images=False,
    save_gifs=True,
    results_path="results",
    project_tag="mnist",
)
```

The saving constructor immediately creates
`results/YYYY-MM-DD_HH-MM-SS project_tag/images` and `.../gifs`. GIF mode asks
`sample` for both noisy-state and predicted-clean frame sequences. Filenames
record the one-based epoch, sampling steps, guidance scale, and eta.

`results_path` with `save_gifs=False` is rejected by the current API; saving
PNGs alone is not an accepted configuration. If `results_path=None`,
`show_images` must be true.

## `RawNetworkValidationCallback`

Diffusion wrappers commonly validate EMA weights. This callback performs a
second validation pass against raw trainable weights and inserts prefixed
results into epoch logs:

```python
from diffusion.callbacks.raw_network_validation_callback import (
    RawNetworkValidationCallback,
)

raw_validation = RawNetworkValidationCallback(
    val_x=validation_dataset,
    val_y=None,
)
model.fit(train_dataset, callbacks=[raw_validation])
```

The bound model must implement
`evaluate(val_x, val_y, network_name="raw", verbose=0, return_dict=True)`.
For each returned key such as `noise_loss`, the callback writes
`val_raw_noise_loss` into the Keras `logs` mapping. `val_x` may instead be an
array/tensor with separate `val_y`, following that wrapper's normal evaluation
contract. Keras normally supplies a nonempty logs mapping. If a caller manually
passes `{}`, the callback's `logs or {}` expression creates a replacement and
the original empty dictionary is not mutated.
