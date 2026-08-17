# Diffusion package

The `diffusion` package provides conditional image-diffusion architectures,
Keras training/sampling wrappers, reusable transformer and spatial layers,
noise schedules, callbacks, and metrics. The package targets TensorFlow 2.10
and uses channels-last image tensors.

## Architecture and wrapper roles

The two model directories form one API boundary:

1. `models/transformer/` and `models/convolution/` define raw neural networks.
   They transform noisy images and conditions into predictions, but do not own
   the diffusion process.
2. `models/wrapper/` accepts a raw network and supplies schedule arrays,
   noising, classifier-free label masking, objectives, optimizer steps, EMA
   synchronization, Keras `train_step`/`test_step`, and reverse sampling.

Compile and train the wrapper. Inspect or call its `network` when direct raw
predictions are needed; use `ema_network` for the smoothed copy when EMA is
enabled. The transformer and wrapper READMEs document their complete input
dictionaries and state.

## Basic conditional model

```python
import tensorflow as tf

from diffusion import DiffusionModel, DiffusionTransformer

network = DiffusionTransformer(
    num_classes=10,
    use_cfg=True,
    timesteps=1_000,
    image_size=28,
    channels=1,
    patch_size=2,
    dim=64,
    depth=4,
)
model = DiffusionModel(
    network,
    scheduler_name="clipped_cosine",
    test_steps=50,
    test_cfg_scale=4.0,
)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")

# A dataset element is (images, class_ids): float [B, 28, 28, 1] in [-1, 1]
# and integer [B]. The wrapper prepares noisy images and shifted CFG IDs.
# model.fit(dataset, epochs=...)

images = model.sample(labels=[1, 2, 3], steps=50, scale=4.0)
```

When classifier-free guidance is active, wrappers reserve condition ID `0` as
the null label and shift real zero-based dataset labels to `1..num_classes`
during training. `sample(labels=...)` and direct raw-network calls consume
those already-shifted embedding IDs, so the example IDs generate real classes
0, 1, and 2. Without CFG, use the ordinary IDs `0..num_classes-1`.

## Schedule API

```python
from diffusion import make_schedule

schedule = make_schedule(
    "clipped_cosine",
    num_steps=1_000,
    min_sqrt_alpha_bar=0.02,
    max_sqrt_alpha_bar=0.95,
)
```

`make_schedule` returns six `float64` NumPy arrays of shape `(num_steps,)`:
`betas`, `alpha_bar`, `sqrt_alpha_bar`, `sqrt_one_minus_alpha_bar`, `sigmas`,
and normalized `timesteps`. Supported names are `linear`, `scaled_linear`,
`squaredcos_cap_v2`, `clipped_cosine`, `sigmoid`, `quadratic`, `ve`, `karras`,
`sub_vp`, and `logistic`. See `schedulers.py` for every valid keyword and the
difference between native sigma-space and beta-equivalent outputs.

## Package map

- `layers/`: patch/condition embeddings, DiT blocks, adaptive normalization,
  feature routing, stochastic depth, and spatial resolution operators.
- `models/`: raw architectures and training/sampling wrappers.
- `callbacks/`: image generation, validation, and batch-loss control hooks.
- `metrics/`: ensemble classification accuracy.
- `schedulers.py`: NumPy schedule generation and conversion.
- `old/`: archived, non-public experiments retained for reproducibility.

The package root re-exports the main supported classes so applications can use
imports such as `from diffusion import DiTClassifier, DiffusionClassifier`.
