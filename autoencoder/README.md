# Autoencoder APIs

`autoencoder` provides dense variational autoencoders for feature replay in
continual learning. These models operate on flat vectors such as 2,048-wide
Xception features; they do not contain convolutional image encoders.

## `VariationalAutoencoder`

The encoder produces `z_mean`, `z_log_var`, and a reparameterized sample `z`.
The decoder reconstructs the original vector. Training minimizes:

```text
reconstruction_loss + beta * KL(q(z | x, y) || N(0, I))
```

Conditioning is optional:

| Mode | Constructor | Call/training input | `generate` return |
| --- | --- | --- | --- |
| Unconditional | `conditioned=False, class_num=None` | `x: [B, data_dim]` | `x_gen: [samples_per_class, data_dim]` |
| Conditional | `conditioned=True, class_num=C` | `(x, one_hot_y)` with labels `[B, C]` | `(x_gen, y_gen)` with `samples_per_class * len(classes)` rows |

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

x_replay, y_replay = vae.generate(
    classes=[0, 3], 
    samples_per_class=500, 
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

`compile_args` starts with Nadam at learning rate `0.1` and MSE, then accepts
any `tf.keras.Model.compile` key such as `optimizer`, `loss`, `metrics`,
`run_eagerly`, or `jit_compile` when supported by the installed TensorFlow.
Top-level `**kwargs` goes to `tf.keras.Model`, so common valid examples are
`name`, `dtype`, and `trainable`; unknown keys fail in Keras.

`train` accepts `train_num`, `epochs`, `batch_size`, `shuffle_buffer`, `seed`,
`validation_data`, `callbacks_list`, `callbacks_monitor`, `clf`, and `verbose`.
Its `train_num` behavior is:

- `-1`: use each supplied row once, with no manual resampling;
- any positive value: sample exactly that many rows with replacement.

Conditional training records argmax label IDs in `seen_classes`. Calling
`generate(classes=None)` replays all recorded classes. An explicit empty class
list returns two empty Python lists; unconditional generation returns only the
sample array and ignores `classes`/`onehot_y_output`.

## `VAEClassifier`

`VAEClassifier` fixes conditional mode, attaches a classifier, and adds
`alpha * categorical_crossentropy` plus accuracy trackers. Forward, training,
and evaluation all classify reconstructed vectors and average categorical
cross-entropy over each batch.

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
`validation_data`, `callbacks_list`, and `verbose`. It supplies `clf` and
monitors `val_clf_accuracy` itself.

## Decoder accuracy callback

`DecoderAccuracyCallback(classifier, samples_per_class=500)` is attached to a
conditional VAE. At each epoch end it generates examples for every class in
`model.seen_classes`, classifies them, and adds `decoder_accuracy` to Keras
logs. Keras layers/models receive `classifier(x_gen, training=False)`; a plain
callable may accept only `classifier(x_gen)`. In either case it must return
`[samples, class_num]` scores. `samples_per_class` must be positive, and at
least one class must have been recorded before the callback runs.

## Continual-learning integration

In direct mode, pass a conditional `VariationalAutoencoder` or `VAEClassifier`
to `common.learner.continually_learn` with `generative_model=model`. In config
mode, select `model.name="vae"` or `"vae_classifier"`; `get_model` creates the
VAE and standalone classifier, while `get_datasets` supplies the loader. The
continual loop generates prior-class features before each task and routes both
VAE training and evaluation through `common.train`.

The dataset loader must return one-hot labels, and `data_dim` must match the
loaded image/feature width. With `use_generative_model_classifier=True`, a
`VAEClassifier`'s attached classifier becomes the continual model; setting
`train_classifier_separately=True` gives it an additional classifier-only fit
and requires that classifier to already be compiled.

See individual class and method docstrings with `help(...)` for tracker initial
state, exact tensor shapes, return dictionaries, callback composition, and
failure conditions.
