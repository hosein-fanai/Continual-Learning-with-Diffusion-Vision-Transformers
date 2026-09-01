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
- `get_network("raw" | "ema")`;
- inner-network `predict_class((images, timesteps, labels), training=...)` and
  `num_classes`.

The constructor uses `Any` because the wrapper protocol spans several model
families; the runtime checks enforce the interface above.

## Usage

```python
from diffusion.metrics.ensemble_accuracy import EnsembleAccuracy

metric = EnsembleAccuracy(
    diffusion_clf=model, 
    network_name="ema",
    compute_type="chunked", 
    weighted=True, 
    max_t=128, 
    t_chunk_size=16, 
    seed=42,
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
With `separate_probas=True`, CFG models instead evaluate labels `0..num_classes`,
add the complete null-label score vector to the matching score from each real
label, and apply softmax after timestep aggregation.

`weighted=False` computes a uniform mean. `weighted=True` uses normalized SNR
weights from the wrapper's existing diffusion schedule, emphasizing cleaner
inputs. Prediction values may be logits or probabilities because sparse
categorical accuracy uses `argmax`.

## Batched versus chunked

`compute_type="batched"` materializes `[B * max_t, H, W, C]` and calls the
network once. It is simple and fast when memory permits. `"chunked"` materializes
at most `[B * t_chunk_size, H, W, C]`, accumulates score sums in the metric's
configured dtype, and is the default. `t_chunk_size` must be positive; a value
at least `max_t` gives one chunk.

## Dataset convenience loop

```python
metric.reset_state()
value = metric.evaluate(validation_dataset)
```

The dataset must be sized (`len(dataset)` must work) and yield `(images,
labels)` or `(images, labels, sample_weight)`. `evaluate` prints batch progress
and returns a NumPy scalar. Metric state is cumulative: neither `evaluate` nor
`test_step` resets prior results. Both `test_step` and `update_state` accept
per-example `sample_weight`.

Use `network_name="ema"` or `network_name="raw"`; any other selector raises
`ValueError`.
