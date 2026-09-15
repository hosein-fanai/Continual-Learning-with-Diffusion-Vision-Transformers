# Autoencoder APIs

`autoencoder` provides dense variational autoencoders for feature replay in
continual learning. These models operate on flat vectors such as 2,048-wide
Xception features; they do not contain convolutional image encoders.
The supported runtime is TensorFlow 2.20 with native Keras 3.

## `VariationalAutoencoder`

The encoder produces `z_mean`, `z_log_var`, and a reparameterized sample `z`.
The decoder reconstructs the original vector. Training minimizes:

```text
reconstruction_loss + beta * KL(q(z | x, y) || N(0, I))
```

Conditioning is optional:

| Mode | Constructor | Call/training input | `sample` return |
| --- | --- | --- | --- |
| Unconditional | `conditioned=False, class_num=None` | `x: [B, data_dim]` | `x_gen: [samples_per_label, data_dim]` |
| Conditional | `conditioned=True, class_num=C` | `(x, one_hot_y)` with labels `[B, C]` | `(x_gen, y_gen)` with `samples_per_label * len(labels)` rows |

```python
from autoencoder.variational_autoencoder import VariationalAutoencoder

vae = VariationalAutoencoder(
    data_dim=2048, 
    latent_dim=16, 
    hiddens_dims=(256, 64), 
    hiddens_kwargs={
        "actv": "relu", 
        "use_batch_norm": False, 
        "kernel_init": "glorot_uniform", 
    },
    last_activation="linear", 
    beta=0.25, 
    conditioned=True, 
    class_num=10, 
    compile_args={"optimizer": "adam", "loss": "mean_squared_error"}, 
    name="feature_vae", 
)

history = vae.train(
    x_train, one_hot_y_train, 
    train_num=-1, 
    epochs=20, 
    batch_size=256, 
    validation_data=(x_val, one_hot_y_val), 
)

x_replay, y_replay = vae.sample(
    labels=[0, 3],
    samples_per_label=500,
    onehot_y_output=True, 
)
# x_replay: [1000, 2048]; y_replay: [1000, 10]
```

### Dictionary and keyword contracts

`hiddens_kwargs` is forwarded to every hidden block and accepts only:

- `actv`: a Keras activation name/callable; `"prelu"` creates a `PReLU` layer;
- `use_batch_norm`: bool; when true, Dense bias is disabled and the implemented
  order is Dense, activation, then batch normalization;
- `kernel_init`: a Keras initializer name or object.

Do not include `units`; widths come from `hiddens_dims`.
Each Dense block receives an independent clone of the selected initializer;
this prevents one reused unseeded initializer object from repeating the same
draw across encoder and decoder layers.

`compile_args` starts with Nadam at learning rate `0.1` and MSE, then accepts
any `tf.keras.Model.compile` key such as `optimizer`, `loss`, `metrics`,
`run_eagerly`, or `jit_compile` when supported by the installed TensorFlow.
Top-level `**kwargs` goes to `tf.keras.Model`, so common valid examples are
`name`, `dtype`, and `trainable`; unknown keys fail in Keras.

### Serialization

`VariationalAutoencoder` is registered with Keras and its `get_config()` records
the complete encoder/decoder topology, conditioning, activation/initializer,
dtype/name, and seed settings. `from_config(...)` and `clone_model(...)` rebuild
an independent, uncompiled architecture. Architecture config deliberately does
not carry learned weights, optimizer slots/iterations, or compile arguments;
use normal model/weight persistence for learned state and compile a direct clone
explicitly.
Observed conditional `seen_classes` IDs are retained as replay metadata, so full
model loading preserves the default `sample(labels=None)` class set. Older
artifacts without this metadata require explicit generation class IDs.

The model package exports are lazy and cached: `from autoencoder import
VariationalAutoencoder, VAEClassifier` avoids eagerly importing every
registered Keras module. The decoder callback now lives at
`common.callbacks.decoder_accuracy.DecoderAccuracy`; the compatibility export
`autoencoder.DecoderAccuracyCallback` resolves to that same class.
Importing the package installs lazy Keras registry proxies for both model
classes. Consequently, `import autoencoder` before `load_model(...)` restores
the canonical Python class (including `isinstance` and custom methods), while
direct `python -m autoencoder.<module>` execution still loads each module once.

`VAEClassifier` follows the same rule and additionally serializes its nested
classifier through Keras object serialization while omitting the inherited
`conditioned` key that its constructor fixes internally. Its classifier must
therefore be a registered/serializable Keras object for a portable config
round-trip; an arbitrary callable may still run but is not promised to
deserialize.

`train` accepts `train_num`, `epochs`, `batch_size`, `shuffle_buffer`, `seed`,
`validation_data`, `callbacks_list`, `callbacks_monitor`, `clf`, `verbose`, and
the optional positive `steps_per_epoch`. Setting the latter repeats the
prepared dataset and fixes the optimizer updates in each epoch; its default
`None` preserves finite-dataset fitting.
Its `train_num` behavior is:

- `-1`: use each supplied row once, with no manual resampling;
- any positive value: sample exactly that many rows with replacement.

Counts and architecture dimensions must be integers; fractions and booleans are
rejected. `epochs`, `batch_size`, and an explicit `steps_per_epoch` must be
positive. `shuffle_buffer=0` disables shuffling. Invalid counts fail before
fitting or changing observed-class metadata.

Automatic early stopping minimizes loss monitors and maximizes accuracy monitors.
For a custom metric with another direction, supply an explicit callback list.
Reconstruction losses are computed in the policy's variable dtype before
reduction, so mixed-float16 training avoids half-precision squared-error overflow.
Optional sample weights are relative, finite, nonnegative row weights normalized
to mean one within each batch. Rescaling them therefore preserves the balance of
reconstruction, KL and classification; an all-zero mask gives zero data losses.
Loss trackers summarize batch objectives using batch size. Keras regularization
penalties remain independent of the row mask.

VAE HPO evaluates its default `generation_loss` using held-out reconstruction
`mean_squared_error`, with fixed preprocessing within each study. This keeps
validation units comparable when training samples MSE versus MAE or different
KL weights. It is a reconstruction proxy, not a measure of prior-sample quality;
continual studies select validation continual accuracy. Explicit objective names
such as `generative_loss` remain available for controlled experiments.

Conditional training records argmax label IDs in `seen_classes`. Calling
`sample(labels=None)` replays all recorded classes. An explicit empty class
list returns two empty Python lists; unconditional generation returns only the
sample array and ignores `labels`/`onehot_y_output`. Pass an integer
`samples_per_label`; the sampler no longer coerces floating counts to integers.
Seeded sampling now derives its stream using `"vae", "sample"`, so the same
seed and weights do not reproduce the previous `generate` stream.

Use `sample(labels=..., samples_per_label=...)` for generation. Replay,
decoder-accuracy callbacks, and final-image reporting use this API. The old
`generate(classes=..., samples_per_class=...)` spelling has no compatibility
alias. Conditional class IDs must be integers in `[0, class_num)`; sampling
never truncates a fractional class ID. A zero sampling count returns an empty
batch, while a negative count is rejected.

## `VAEClassifier`

`VAEClassifier` fixes conditional mode, attaches a classifier, and adds
`alpha * categorical_crossentropy` plus accuracy trackers. Its VAE reconstructs
from `(x, y)`, but its classifier predicts directly from `x`; ground-truth
labels therefore cannot leak into predictions through a label-conditioned
reconstruction. Classification cross-entropy is averaged over each batch.

```python
from autoencoder import VAEClassifier

model = VAEClassifier(
    class_num=10, 
    classifier=classifier, 
    alpha=0.01, 
    data_dim=2048, 
    latent_dim=8, 
    compile_args={"optimizer": "adam"}, 
)
history = model.train(x_train, one_hot_y_train, epochs=10, train_num=-1)
```

Its `**kwargs` accepts the VAE architecture keys, `compile_args`, `compile`, and
Keras model keys. Do not pass `conditioned` or `class_num`. Set `compile=False`
to construct without compiling, for example before supplying a custom compile
configuration later.

`VAEClassifier.train(**kwargs)` accepts only the base training controls
`train_num`, `epochs`, `batch_size`, `shuffle_buffer`, `seed`,
`validation_data`, `callbacks_list`, `verbose`, and `steps_per_epoch`. It
supplies `clf` and monitors `val_clf_accuracy` when validation is supplied,
otherwise `clf_accuracy`.

## Decoder accuracy callback

`common.callbacks.decoder_accuracy.DecoderAccuracy(classifier,
samples_per_label=500, seed=None)` is attached by `train(..., clf=...)` and
`VAEClassifier.train`. It samples every observed class, classifies
the generated features, and add `decoder_accuracy` to Keras logs.

The callback calls `sample(samples_per_label=...)`. Keras classifiers receive
`training=False`; ordinary callables receive the generated vectors. Empty
generation and incompatible label/prediction batches are rejected before
accuracy is computed. Epoch-derived seeds keep callback sampling independent
from the encoder's reparameterization stream.

## Continual-learning integration

In direct mode, pass a conditional `VariationalAutoencoder` to
`common.learner.continually_learn` with `generative_model=model`. In config
mode, select `model.name="vae"`; `get_model` creates the VAE and expanding
standalone classifier, while `get_datasets` supplies one-hot labels through the
loader. The continual loop generates prior-class features before each task and
routes both VAE training and evaluation through `common.train`.

The dataset loader must return one-hot labels, and `data_dim` must match the
loaded image/feature width. `VAEClassifier` is intentionally unsupported in the
continual loop: its fixed full-class head exposes future logits. Use the
generator-only conditional VAE with the learner's expanding external
classifier instead.

See individual class and method docstrings with `help(...)` for tracker initial
state, exact tensor shapes, return dictionaries, callback composition, and
failure conditions.
