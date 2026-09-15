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

With `mode="min"` (the default), improvement means `current < best - min_delta`.
With `mode="max"`, it means `current > best + min_delta`; `mode="auto"` maximizes
accuracy/acc/AUC monitors and minimizes other names. Missing monitor keys
are ignored. The stop condition is `wait >= patience`, so `patience=200` stops
after 200 consecutive non-improving batches. `fit_progressively` creates a new
callback for each stage, giving each stage independent best and patience state.
Its `stopper_mode` argument now selects direction for both batch-wise and
epoch-wise pacing. Use `stopper_mode="max"` for an accuracy monitor.

## `ImageGenerator`

This callback reads `model.test_steps`, `model.test_cfg_scale`, `model.test_eta`,
and `model.test_network_name`, calls `model.sample(...)` at epoch end, and
renders the result.

`ImageGenerator(add_null_label=True)` forwards that flag to the sampler and
omits explicit labels. The wrapper chooses observed classes for a dynamic
model or all real classes for a fixed-width model. With CFG enabled, the default
preview starts with null condition ID 0; set `add_null_label=False` to omit it.
For a model without CFG, the flag has no effect and class 0 appears once.
These rules also apply when sampling delegates to the VAE path.
The callback passes `has_null_label=True` to the image plotter only when the
grid contains a CFG null preview, giving it title `-1` and the following images
titles `0`, `1`, and so on. Other grids start at `0`. These are grid indices,
not recovered dataset class IDs.

The current validation permits two configurations.

Display only:

```python
from diffusion.callbacks.image_generator import ImageGenerator

preview = ImageGenerator(show_images=True)
```

Save PNG and GIF artifacts, optionally also display them:

```python
artifacts = ImageGenerator(
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

Supplying `results_path` saves PNGs whether or not GIF output is enabled. If
`results_path=None`, `show_images` must be true; `save_gifs=True` always
requires a result path.

New result directories are reserved exclusively using timestamp and tag, with a
unique suffix on collision. Every writer for the same run shares that reservation.

## `RawNetworkValidation`

Diffusion wrappers commonly validate EMA weights. This callback performs a
second validation pass against raw trainable weights and inserts prefixed
results into epoch logs:

```python
from diffusion.callbacks.raw_network_validation import (
    RawNetworkValidation,
)

raw_validation = RawNetworkValidation(
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
contract. Both populated and empty mappings supplied by Keras or a caller are
updated in place. If the evaluator exposes `eval_both`, the callback enables
it so V2 reports both generator and discriminator phases.

Use the direct imports above or the lazy package exports
`diffusion.ImageGenerator` and `diffusion.RawNetworkValidation`.
Strict recovery recognizes the current image callback class and its stable
configuration. See the [current audit](../../research_audit.md).
