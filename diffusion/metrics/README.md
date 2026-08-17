# Diffusion metrics

This directory contains metrics that need more context than a single raw model
prediction. `EnsembleAccuracy` evaluates a diffusion classifier across multiple
noise levels and averages its class scores before measuring sparse categorical
accuracy.

## Required model interface

Pass a `DiffusionClassifier`-compatible wrapper, not a bare
`DiTClassifier`. The wrapper must expose:

- `timesteps`;
- `noisify(images, timesteps, seed=...)`;
- `network` and, when selected, `ema_network`;
- inner-network `predict_class((images, timesteps, labels), training=...)` and
  `num_classes`.

The constructor annotation retains a historical raw-network type, but the
runtime API above is authoritative.

## Usage

```python
from diffusion.metrics.ensemble_accuracy import EnsembleAccuracy

metric = EnsembleAccuracy(
    diffusion_clf=model, 
    netwrok_name="ema",       # spelling is part of the current public API
    compute_type="chunked", 
    weighted=True, 
    max_t=128, 
    t_chunk_size=16, 
    random_seed=42, 
    name="ensemble_accuracy", 
    dtype="float32", 
)

metric.test_step(labels, images)
accuracy = metric.result()     # scalar tf.Tensor
metric.reset_state()
```

`max_t` means timesteps `0` through `max_t - 1`; it must be positive and no
larger than `model.timesteps`. Every image is independently noised at each
selected timestep. Unconditional label 0 is passed to the classifier branch.

`weighted=False` computes a uniform mean. `weighted=True` gives timestep `t`
weight `1 - t / max_t`, emphasizing cleaner inputs. Prediction values may be
logits or probabilities because sparse categorical accuracy uses `argmax`.

## Batched versus chunked

`compute_type="batched"` materializes `[B * max_t, H, W, C]` and calls the
network once. It is simple and fast when memory permits. `"chunked"` materializes
at most `[B * t_chunk_size, H, W, C]`, accumulates float32 score sums, and is the
default. `t_chunk_size` must be positive; a value at least `max_t` gives one
chunk.

## Dataset convenience loop

```python
metric.reset_state()
value = metric.evaluate(validation_dataset)
```

The dataset must be sized (`len(dataset)` must work) and yield `(images,
labels)`. `evaluate` prints batch progress and returns a NumPy scalar. Metric
state is cumulative: neither `evaluate` nor `test_step` resets prior results.
The custom `update_state(y_true, y_pred)` does not accept `sample_weight`.

`netwrok_name` is intentionally documented with its existing misspelling.
Use exactly `"ema"` or `"raw"`; the implementation selects EMA only for the
former and otherwise falls back to the raw network.
