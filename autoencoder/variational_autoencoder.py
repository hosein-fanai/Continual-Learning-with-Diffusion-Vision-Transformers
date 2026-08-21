"""Dense variational autoencoder with optional class conditioning and replay."""

import tensorflow as tf
from tensorflow.keras import metrics, layers, models, optimizers

import numpy as np

from collections.abc import Callable, Sequence

from common.dataloader import get_dataset
from common.model import get_callbacks

from autoencoder.decoder_accuracy_callback import DecoderAccuracyCallback


class VariationalAutoencoder(models.Model):
    """Encode dense samples into a Gaussian latent space and reconstruct them.

    The model is designed for flat feature vectors rather than image tensors.
    In unconditional mode it maps ``x`` directly through the encoder and
    decoder.  In conditional mode both networks concatenate a one-hot class
    vector to their input, so calls and training data use ``(x, y)`` pairs.
    Training minimizes reconstruction loss plus ``beta * KL``.

    Attributes:
        latent_dim (int): Width of ``z_mean``, ``z_log_var``, and sampled ``z``;
            initialized from ``latent_dim``.
        beta (float): KL-divergence multiplier initialized from ``beta``.
        conditioned (bool): Whether one-hot labels condition both networks.
        class_num (int | None): One-hot label width in conditional mode;
            otherwise ``None``.
        encoder (tf.keras.Model): Maps ``x`` or ``(x, y)`` to
            ``(z_mean, z_log_var, z)``.
        decoder (tf.keras.Model): Maps ``z`` or ``(z, y)`` to reconstructed
            vectors.
        seen_classes (list[int]): Unique class IDs observed by :meth:`train`;
            initialized empty and used by :meth:`generate` when classes are not
            specified.
        total_loss_tracker (tf.keras.metrics.Mean): Running total-loss mean,
            initialized with value/count zero.
        kl_loss_tracker (tf.keras.metrics.Mean): Running KL-loss mean,
            initialized with value/count zero.
        recon_loss_tracker (tf.keras.metrics.Mean): Running reconstruction-loss
            mean, initialized with value/count zero.
    """

    def __init__(
        self, 
        data_dim=2048, 
        latent_dim=8, 
        hiddens_dims=(16,), 
        hiddens_kwargs={}, 
        last_activation="tanh", 
        beta=0.25, 
        conditioned=False, 
        class_num=None, 
        compile=True, 
        compile_args={}, 
        **kwargs
    ):
        """Build the encoder, decoder, metric state, and optional optimizer.

        Args:
            data_dim (int): Positive feature width for input and reconstruction;
                the expected sample shape is ``[batch, data_dim]``.
            latent_dim (int): Positive number of Gaussian latent features.
            hiddens_dims (Sequence[int]): Encoder hidden widths in forward
                order.  The decoder uses the reverse order.  ``()`` creates
                direct latent/output projections with no hidden block.
            hiddens_kwargs (Mapping[str, object]): Options forwarded to every
                :meth:`_dense_layer`.  Allowed keys are ``actv`` (a Keras
                activation name/callable, including the special ``"prelu"``),
                ``use_batch_norm`` (bool), and ``kernel_init`` (a Keras
                initializer name/object).  Example:
                ``{"actv": "relu", "use_batch_norm": False,
                "kernel_init": "glorot_uniform"}``.  ``units`` is not valid
                because each width comes from ``hiddens_dims``.
            last_activation (str | Callable | None): Keras activation for the
                final reconstruction.  Use ``"tanh"`` for data scaled to
                ``[-1, 1]``, ``"sigmoid"`` for ``[0, 1]``, or ``None``/
                ``"linear"`` for unbounded features.
            beta (float): Scalar KL-loss multiplier.  ``0`` trains an ordinary
                stochastic autoencoder objective; larger values regularize the
                latent distribution more strongly.
            conditioned (bool): Require one-hot labels and concatenate them to
                encoder/decoder inputs when true.
            class_num (int | None): Positive one-hot width when conditioned.
                It must be ``None`` when ``conditioned=False`` and non-``None``
                when ``conditioned=True``.
            compile (bool): Whether to call ``Model.compile`` during
                construction.  False leaves compilation to the caller.
            compile_args (Mapping[str, object]): Overrides/extends defaults
                ``{"optimizer": Nadam(learning_rate=0.1, decay=0.0),
                "loss": "mean_squared_error"}``.  Keys may be any accepted by
                ``Model.compile``, for example ``{"optimizer": "adam",
                "run_eagerly": True}``.
            **kwargs (object): Standard ``tf.keras.Model`` constructor options,
                such as ``name``, ``trainable``, and ``dtype``.

        Returns:
            None.

        Raises:
            AssertionError: If ``conditioned`` and ``class_num`` are
                inconsistent.
            TypeError: If either keyword mapping contains unsupported keys.
        """

        super().__init__(**kwargs)

        assert (conditioned and class_num is not None) \
        or (not conditioned and class_num is None), \
            "When conditioned is True, class_num cannot be None, " \
            "and when conditioned is False, class_num needs to be None."


        self.latent_dim = latent_dim
        self.beta = beta
        self.conditioned = conditioned
        self.class_num = class_num

        self.encoder = self._build_encoder(data_dim, latent_dim, hiddens_dims, 
                                        hiddens_kwargs, class_num)
        self.decoder = self._build_decoder(data_dim, latent_dim, hiddens_dims[::-1], 
                                        hiddens_kwargs, class_num, last_activation)

        self.seen_classes = []

        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        self.recon_loss_tracker = metrics.Mean(name="recon_loss")

        compile_args_default = {
            "optimizer": optimizers.Nadam(learning_rate=0.1, decay=0.),
            "loss": "mean_squared_error",
        }
        compile_args = {**compile_args_default, **compile_args}

        if compile:
            self.compile(**compile_args)

    def _dense_layer(self, units, actv="selu", 
                    use_batch_norm=True, 
                    kernel_init="he_normal"):
        """Create one dense activation/normalization block.

        Args:
            units (int): Positive output feature width.
            actv (str | Callable): Keras activation.  ``"prelu"`` creates a
                trainable ``PReLU`` layer; other values use ``Activation`` when
                batch normalization is enabled or the Dense activation itself
                when it is disabled.
            use_batch_norm (bool): Append ``BatchNormalization`` and omit the
                Dense bias when true.  The implemented order is Dense,
                activation, then batch normalization.
            kernel_init (str | tf.keras.initializers.Initializer): Dense kernel
                initializer accepted by Keras.

        Returns:
            tf.keras.Sequential: An unbuilt block mapping ``[..., input_dim]``
            to ``[..., units]``.
        """

        dlayer = models.Sequential()

        dlayer.add(layers.Dense(units, activation=actv if not(use_batch_norm or actv == "prelu") else "linear", 
                                kernel_initializer=kernel_init, use_bias=not use_batch_norm))
        dlayer.add(layers.Activation(actv)) if (use_batch_norm and actv != "prelu") else None
        dlayer.add(layers.PReLU()) if actv == "prelu" else None
        dlayer.add(layers.BatchNormalization()) if use_batch_norm else None

        return dlayer

    def _build_encoder(self, input_dim, latent_dim, 
                    hiddens_dims, hiddens_kwargs={}, 
                    class_num=None):
        """Build the functional Gaussian encoder.

        Args:
            input_dim (int): Width of each flat input sample.
            latent_dim (int): Width of each latent statistic/sample.
            hiddens_dims (Sequence[int]): Hidden block widths in encoder order.
            hiddens_kwargs (Mapping[str, object]): Per-block keys ``actv``,
                ``use_batch_norm``, and/or ``kernel_init``; see
                :meth:`_dense_layer`.
            class_num (int | None): Conditional label width.  Used only when
                ``self.conditioned`` is true.

        Returns:
            tf.keras.Model: A model accepting ``x`` shaped
            ``[batch, input_dim]`` or ``(x, y)`` with ``y`` shaped
            ``[batch, class_num]``.  It returns three tensors, each shaped
            ``[batch, latent_dim]``: mean, log variance, and sampled latent.
        """

        x_inputs = layers.Input(shape=(input_dim,), name="x_input")

        if self.conditioned:
            y_inputs = layers.Input(shape=(class_num,), name="y_input")
            x = layers.Concatenate()([x_inputs, y_inputs])
            inputs = [x_inputs, y_inputs]
        else:
            x = x_inputs
            inputs = x_inputs

        for hidden_dim in hiddens_dims:
            x = self._dense_layer(hidden_dim, **hiddens_kwargs)(x)

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = VariationalAutoencoder.compute_z(z_mean, z_log_var)

        encoder = models.Model(
            inputs, 
            [z_mean, z_log_var, z], 
            name="encoder"
        )

        return encoder

    def _build_decoder(self, output_dim, latent_dim, 
                    hiddens_dims, hiddens_kwargs, 
                    class_num, last_activation):
        """Build the functional reconstruction decoder.

        Args:
            output_dim (int): Reconstructed vector width.
            latent_dim (int): Input latent-vector width.
            hiddens_dims (Sequence[int]): Hidden widths in decoder order,
                normally the reversed encoder widths.
            hiddens_kwargs (Mapping[str, object]): Per-block keys accepted by
                :meth:`_dense_layer`.
            class_num (int | None): One-hot label width in conditional mode.
            last_activation (str | Callable | None): Final Dense activation.

        Returns:
            tf.keras.Model: A model accepting ``z`` shaped
            ``[batch, latent_dim]`` or conditional ``(z, y)`` and returning a
            tensor shaped ``[batch, output_dim]``.
        """

        z_inputs = layers.Input(shape=(latent_dim,), name="z_input")

        if self.conditioned:
            y_inputs = layers.Input(shape=(class_num,), name="y_input")
            z = layers.Concatenate()([z_inputs, y_inputs])
            inputs = [z_inputs, y_inputs]
        else:
            z = z_inputs
            inputs = z_inputs

        for hidden_dim in hiddens_dims:
            z = self._dense_layer(hidden_dim, **hiddens_kwargs)(z)

        outputs = layers.Dense(
            output_dim, 
            activation=last_activation
        )(z)

        decoder = models.Model(
            inputs, 
            outputs, 
            name="decoder"
        )

        return decoder

    @staticmethod
    def compute_z(z_mean, z_log_var):
        """Sample a latent vector with the reparameterization trick.

        Args:
            z_mean (tf.Tensor): ``float32`` Gaussian means shaped
                ``[batch, latent_dim]``.
            z_log_var (tf.Tensor): Matching ``float32`` elementwise log
                variances.

        Returns:
            tf.Tensor: ``float32`` values computed as
            ``z_mean + exp(0.5*z_log_var) * epsilon`` with the same shape;
            ``epsilon`` is newly drawn from a standard normal distribution on
            every call.
        """

        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        return z

    @staticmethod
    def compute_kl(z_mean, z_log_var):
        """Compute mean KL divergence from the unit Gaussian prior.

        Args:
            z_mean (tf.Tensor): Rank-two means shaped ``[batch, latent_dim]``.
            z_log_var (tf.Tensor): Matching log variances.

        Returns:
            tf.Tensor: Scalar floating tensor.  Divergence is summed over
            latent axis 1 and averaged across the batch.
        """

        return -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )

    @property
    def metrics(self):
        """Expose loss trackers that Keras resets between epochs/evaluations.

        Returns:
            list[tf.keras.metrics.Mean]: Total, KL, and reconstruction trackers,
            in that order.
        """

        return [
            self.total_loss_tracker, 
            self.kl_loss_tracker, 
            self.recon_loss_tracker, 
        ]

    def call(self, inputs, training=False):
        """Encode, sample, and reconstruct a batch.

        Args:
            inputs (tf.Tensor | tuple[tf.Tensor, tf.Tensor]): In unconditional
                mode, feature tensor ``x`` shaped ``[batch, data_dim]``.  In
                conditional mode, ``(x, y)`` where ``y`` is one-hot and shaped
                ``[batch, class_num]``.
            training (bool | tf.Tensor): Keras training flag forwarded to both
                submodels, affecting batch normalization.

        Returns:
            tuple[tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
            ``((z_mean, z_log_var, z), reconstruction)``.  Latent tensors have
            shape ``[batch, latent_dim]`` and reconstruction matches
            ``[batch, data_dim]``.
        """

        z_mean, z_log_var, z = self.encoder(inputs, training=training)

        if self.conditioned:
            _, y = inputs
            decoder_inputs = (z, y)
        else:
            decoder_inputs = z

        return ((z_mean, z_log_var, z), 
                self.decoder(decoder_inputs, training=training))

    def train_step(self, data):
        """Run one optimizer step for reconstruction plus beta-weighted KL.

        Args:
            data (tf.Tensor | tuple[tf.Tensor, tf.Tensor]): Unconditional
                feature batch, or conditional ``(x, one_hot_y)`` as supplied by
                Keras ``fit``.

        Returns:
            dict[str, tf.Tensor]: Scalar running means under ``loss``,
            ``kl_loss``, and ``recon_loss``.
        """

        if self.conditioned:
            x, _ = data
        else:
            x = data

        with tf.GradientTape() as tape:
            (z_mean, z_log_var, _), x_recon = self(data, training=True)

            recon_loss = self.compiled_loss(
                x,
                x_recon,
                regularization_losses=self.losses,
            )

            kl_loss = VariationalAutoencoder.compute_kl(z_mean, z_log_var)

            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(), 
            "kl_loss": self.kl_loss_tracker.result(), 
            "recon_loss": self.recon_loss_tracker.result(),
        }

    def test_step(self, data):
        """Evaluate reconstruction and KL losses without updating weights.

        Args:
            data (tf.Tensor | tuple[tf.Tensor, tf.Tensor]): Unconditional
                feature batch or conditional ``(x, one_hot_y)``.

        Returns:
            dict[str, tf.Tensor]: Scalar running means under ``loss``,
            ``kl_loss``, and ``recon_loss``.
        """

        if self.conditioned:
            x, _ = data
        else:
            x = data

        (z_mean, z_log_var, _), x_recon = self(data, training=False)

        recon_loss = self.compiled_loss(
            x,
            x_recon,
            regularization_losses=self.losses,
        )

        kl_loss = VariationalAutoencoder.compute_kl(z_mean, z_log_var)

        total_loss = self.beta * kl_loss + recon_loss

        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.recon_loss_tracker.update_state(recon_loss)

        return {
            "loss": self.total_loss_tracker.result(), 
            "kl_loss": self.kl_loss_tracker.result(), 
            "recon_loss": self.recon_loss_tracker.result(),
        }

    def generate(self, classes=None, 
                samples_per_class=500, 
                onehot_y_output=False):
        """Decode random normal latents into synthetic replay samples.

        Args:
            classes (Sequence[int] | None): Conditional class IDs.  ``None``
                uses ``seen_classes``; ``[]`` returns ``([], [])`` immediately.
                IDs should lie in ``[0, class_num)``.  In unconditional mode
                this argument is ignored.
            samples_per_class (int): Nonnegative number of examples per class
                in conditional mode, or total examples in unconditional mode.
            onehot_y_output (bool): In conditional mode, return labels as
                ``float32`` one-hot rows when true or NumPy integer IDs when
                false.  Ignored in unconditional mode.

        Returns:
            numpy.ndarray | tuple[numpy.ndarray, numpy.ndarray] | tuple[list,
            list]: Unconditional mode returns samples shaped
            ``[samples_per_class, data_dim]``.  Conditional mode returns
            ``(x, y)`` with ``len(classes) * samples_per_class`` rows, ordered
            in contiguous class groups.  With no conditional classes, both
            outputs are empty Python lists rather than arrays.
        """

        if self.conditioned:
            if classes is None:
                classes = self.seen_classes

            if len(classes) == 0:
                return [], []

            z = tf.random.normal(shape=(samples_per_class*len(classes), self.latent_dim))
            y = tf.concat([tf.one_hot(tf.cast([i]*samples_per_class, tf.uint8), 
                                    depth=self.class_num) for i in classes], 
                                axis=0)
            x = self.decoder((z, y), training=False)

            x = x.numpy()
            y = y.numpy()

            if not onehot_y_output:
                y = np.argmax(y, axis=-1)

            return x, y

        z = tf.random.normal(
            shape=(samples_per_class, self.latent_dim)
        )
        x = self.decoder(z, training=False)

        x = x.numpy()
        
        return x

    def train(
        self, 
        x: np.ndarray | tf.Tensor, 
        y: np.ndarray | tf.Tensor | None = None, 
        train_num: int = 10_000, 
        epochs: int = 10, 
        batch_size: int = 512, 
        shuffle_buffer: int = 10_000, 
        seed: int | None = None, 
        validation_data: (
            tf.data.Dataset
            | np.ndarray
            | tf.Tensor
            | tuple[np.ndarray | tf.Tensor, np.ndarray | tf.Tensor]
            | None
        ) = None, 
        callbacks_list: Sequence[tf.keras.callbacks.Callback] | None = None, 
        callbacks_monitor: str = "", 
        clf: models.Model | Callable | None = None, 
        verbose: bool | int = 1
    ) -> dict[str, list[float]]:
        """Fit the VAE and optionally monitor generated-sample accuracy.

        Args:
            x (numpy.ndarray | tf.Tensor): Flat samples shaped
                ``[samples, data_dim]``.
            y (numpy.ndarray | tf.Tensor | None): Required only in conditional
                mode; one-hot labels shaped ``[samples, class_num]``.  Observed
                argmax class IDs are added to ``seen_classes``.
            train_num (int): ``-1`` uses the input once without resampling.
                Any other value samples indices *with replacement*.  The actual
                sample count is ``max(train_num, len(x))``: values smaller than
                the input length therefore resample ``len(x)`` rows rather than
                creating a smaller subset, while larger values oversample to
                the requested size.
            epochs (int): Positive maximum number of Keras training epochs.
            batch_size (int): Positive examples per ``tf.data.Dataset`` batch.
            shuffle_buffer (int): Training shuffle capacity passed to
                :func:`common.dataloader.get_dataset`; ``0`` disables shuffling.
            seed (int | None): Optional training-dataset shuffle seed.
            validation_data (tf.data.Dataset | numpy.ndarray | tf.Tensor |
                tuple[numpy.ndarray | tf.Tensor, numpy.ndarray | tf.Tensor] |
                None): Validation input. A passed dataset is preserved; raw
                conditional ``(x_val, y_val)`` arrays or an unconditional
                ``x_val`` array/tensor are converted to a fresh dataset.
            callbacks_list (Sequence[tf.keras.callbacks.Callback] | None): Exact
                callbacks to pass to ``fit``.  When omitted, :func:`get_callbacks`
                adds early stopping; when ``clf`` is supplied, a
                :class:`DecoderAccuracyCallback` is prepended.
            callbacks_monitor (str): Metric name for automatically constructed
                callbacks.  With ``clf`` and an empty string it becomes
                ``"decoder_accuracy"``.  Without ``clf``, the empty default is
                passed through unchanged.
            clf (tf.keras.Model | Callable | None): Classifier mapping generated
                vectors ``[batch, data_dim]`` to class scores
                ``[batch, class_num]``.  It is used only by the decoder-accuracy
                callback and does not enter the VAE loss.
            verbose (int | bool): Keras verbosity and callback verbosity.

        Returns:
            dict[str, list[float]]: ``History.history`` with per-epoch total,
            KL, reconstruction, validation, learning-rate, and/or callback
            metrics depending on the supplied data and callbacks.

        Raises:
            AssertionError: If label presence does not match conditional mode.
        """

        assert (self.conditioned and (y is not None)) \
        or (not self.conditioned and (y is None)) 


        callbacks_list = list(callbacks_list) if callbacks_list is not None else None

        if train_num != -1: # Resample to the requested minimum training size.
            input_size = len(x)
            train_num = max(train_num, input_size)

            indices = np.random.randint(
                0, 
                input_size, 
                (train_num,)
            )
            x = tf.gather(
                x, 
                indices
            ) if isinstance(x, tf.Tensor) else x[indices]
            y = tf.gather(# Keep conditional labels aligned with samples.
                y, 
                indices
            ) if isinstance(y, tf.Tensor) and y is not None else y[indices]

        if y is not None: # Record classes available for conditional generation.
            new_classes = np.unique(np.argmax(y, axis=-1))
            self.seen_classes.extend(new_classes)
            self.seen_classes = list(set(self.seen_classes))

        if clf is not None and callbacks_list is None: # Build classifier-aware defaults.
            if callbacks_monitor == "": # Select the decoder metric by default.
                callbacks_monitor = "decoder_accuracy"

            callbacks_list = [
                DecoderAccuracyCallback(classifier=clf), 
                *get_callbacks(
                    monitor=callbacks_monitor, 
                    verbose=verbose
                )
            ]
        elif clf is not None and callbacks_list is not None: # Add decoder evaluation to user callbacks.
            callbacks_list = [
                DecoderAccuracyCallback(classifier=clf), 
                *callbacks_list
            ]
        elif clf is None and callbacks_list is None: # Build ordinary VAE callbacks.
            callbacks_list = get_callbacks(
                monitor=callbacks_monitor, 
                verbose=verbose
            )

        trainset = get_dataset(
            x, y, 
            shuffle_buffer=shuffle_buffer, 
            batch_size=batch_size, 
            drop_remainder=False, 
            seed=seed
        )
        if isinstance(validation_data, tf.data.Dataset): # Preserve prepared validation pipelines.
            valset = validation_data
        elif isinstance(validation_data, tuple): # Convert paired validation arrays.
            valset = get_dataset(
                validation_data[0], 
                validation_data[1], 
                shuffle_buffer=0, 
                batch_size=batch_size, 
                drop_remainder=False
            )
        elif validation_data is not None: # Convert unconditional validation inputs.
            valset = get_dataset(
                validation_data, 
                shuffle_buffer=0, 
                batch_size=batch_size, 
                drop_remainder=False
            )
        else: # Train without validation when no input was supplied.
            valset = None

        history = self.fit(
            trainset, 
            epochs=epochs, 
            validation_data=valset,
            callbacks=callbacks_list, 
            verbose=verbose
        ).history

        return history


def run_self_tests() -> dict[str, str]:
    """Run CPU-small tests for every VAE mode and public behavior.

    The suite covers valid and invalid conditioning, compilation choices,
    every dense-block activation/normalization branch, empty/nonempty hidden
    stacks, latent sampling and KL mathematics, conditional/unconditional
    calls, real eager train/test steps, metric state, generation boundaries,
    weight persistence, and every callback/resampling branch of :meth:`train`.
    ``fit`` orchestration is mocked so this module never downloads data or
    starts a long training job.

    Args:
        None.

    Returns:
        dict[str, str]: ``{"VariationalAutoencoder": "passed"}`` after all
        assertions succeed.
    """

    from pathlib import Path
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest import mock
    import sys


    tf.keras.backend.clear_session()
    tf.random.set_seed(2026)
    np.random.seed(2026)

    for conditioned, class_num in ((True, None), (False, 2)):
        try:
            VariationalAutoencoder(
                data_dim=4, 
                latent_dim=2, 
                hiddens_dims=(), 
                conditioned=conditioned, 
                class_num=class_num, 
                compile=False, 
            )
        except AssertionError:
            pass
        else:
            raise AssertionError("Conditioning and class_num must agree.")

    try:
        VariationalAutoencoder(
            data_dim=4, 
            latent_dim=2, 
            hiddens_dims=(3,), 
            hiddens_kwargs={"unknown_option": True}, 
            compile=False, 
        )
    except TypeError:
        pass
    else:
        raise AssertionError("Unknown dense-block options must be rejected.")

    uncompiled = VariationalAutoencoder(
        data_dim=4, 
        latent_dim=2, 
        hiddens_dims=(), 
        last_activation=None, 
        beta=0.0, 
        compile=False, 
        name="uncompiled_vae", 
        trainable=False, 
    )
    assert uncompiled.name == "uncompiled_vae"
    assert uncompiled.trainable is False
    assert uncompiled._is_compiled is False
    assert uncompiled.latent_dim == 2 and uncompiled.beta == 0.0
    assert uncompiled.conditioned is False and uncompiled.class_num is None
    assert len(uncompiled.encoder.layers) > 0 and len(uncompiled.decoder.layers) > 0

    unconditioned = VariationalAutoencoder(
        data_dim=4, 
        latent_dim=2, 
        hiddens_dims=(5,), 
        hiddens_kwargs={
            "actv": "relu", 
            "use_batch_norm": False, 
            "kernel_init": "glorot_uniform", 
        }, 
        last_activation=None, 
        beta=0.5, 
        compile_args={
            "optimizer": tf.keras.optimizers.SGD(learning_rate=0.01), 
            "loss": "mean_squared_error", 
            "run_eagerly": True, 
        },
        name="unconditional_vae",
    )
    assert unconditioned._is_compiled is True
    assert isinstance(unconditioned.optimizer, tf.keras.optimizers.SGD)
    assert unconditioned.run_eagerly is True

    relu_no_bn = unconditioned._dense_layer(
        3, 
        actv="relu", 
        use_batch_norm=False, 
        kernel_init="ones"
    )
    assert len(relu_no_bn.layers) == 1
    assert isinstance(relu_no_bn.layers[0], layers.Dense)
    assert relu_no_bn.layers[0].use_bias is True
    assert relu_no_bn.layers[0].activation is tf.keras.activations.relu
    tf.debugging.assert_equal(
        tf.shape(relu_no_bn(tf.ones((2, 4)))), 
        tf.constant([2, 3])
    )

    relu_with_bn = unconditioned._dense_layer(
        3, 
        actv="relu", 
        use_batch_norm=True, 
        kernel_init="ones"
    )
    assert [type(layer) for layer in relu_with_bn.layers] == [
        layers.Dense,
        layers.Activation,
        layers.BatchNormalization,
    ]
    assert relu_with_bn.layers[0].use_bias is False
    tf.debugging.assert_equal(
        tf.shape(relu_with_bn(tf.ones((2, 4)), training=True)), 
        tf.constant([2, 3]), 
    )

    prelu_no_bn = unconditioned._dense_layer(
        3, 
        actv="prelu", 
        use_batch_norm=False
    )
    assert [type(layer) for layer in prelu_no_bn.layers] == [
        layers.Dense, 
        layers.PReLU, 
    ]
    assert prelu_no_bn.layers[0].activation is tf.keras.activations.linear
    assert prelu_no_bn.layers[0].use_bias is True
    tf.debugging.assert_equal(
        tf.shape(prelu_no_bn(tf.ones((2, 4)))), 
        tf.constant([2, 3])
    )

    prelu_with_bn = unconditioned._dense_layer(
        3, 
        actv="prelu", 
        use_batch_norm=True
    )
    assert [type(layer) for layer in prelu_with_bn.layers] == [
        layers.Dense, 
        layers.PReLU, 
        layers.BatchNormalization, 
    ]
    assert prelu_with_bn.layers[0].use_bias is False
    tf.debugging.assert_equal(
        tf.shape(prelu_with_bn(tf.ones((2, 4)))), 
        tf.constant([2, 3])
    )

    x = tf.constant([
        [0.0, 0.25, 0.5, 0.75], 
        [1.0, 0.75, 0.5, 0.25]
    ], dtype=tf.float32)
    (z_mean, z_log_var, z), reconstruction = unconditioned(x, training=False)
    for latent in (z_mean, z_log_var, z):
        assert latent.shape == (2, 2) and latent.dtype == tf.float32
        assert bool(tf.reduce_all(tf.math.is_finite(latent)))
    assert reconstruction.shape == (2, 4)
    assert reconstruction.dtype == tf.float32

    zero_latents = tf.zeros((3, 2), tf.float32)
    tf.debugging.assert_near(
        VariationalAutoencoder.compute_kl(zero_latents, zero_latents), 
        tf.constant(0.0), 
    )
    tf.debugging.assert_near(
        VariationalAutoencoder.compute_kl(tf.ones((3, 2)), zero_latents), 
        tf.constant(1.0), 
    )
    tf.random.set_seed(77)
    sampled_a = VariationalAutoencoder.compute_z(zero_latents, zero_latents)
    tf.random.set_seed(77)
    sampled_b = VariationalAutoencoder.compute_z(zero_latents, zero_latents)
    tf.debugging.assert_near(sampled_a, sampled_b)
    assert sampled_a.shape == zero_latents.shape
    assert bool(tf.reduce_any(tf.not_equal(sampled_a, zero_latents)))
    try:
        VariationalAutoencoder.compute_z(
            tf.zeros((1, 2), tf.float64), 
            tf.zeros((1, 2), tf.float64)
        )
    except (tf.errors.InvalidArgumentError, TypeError):
        pass
    else:
        raise AssertionError("compute_z currently requires float32 latent inputs.")

    metric_names = [metric.name for metric in unconditioned.metrics]
    assert metric_names == ["total_loss", "kl_loss", "recon_loss"]
    unconditioned.reset_metrics()
    assert all(float(metric.result()) == 0.0 for metric in unconditioned.metrics)
    weights_before_train = [
        weight.numpy().copy() 
        for weight in unconditioned.trainable_weights
    ]
    train_result = unconditioned.train_step(x)
    assert set(train_result) == {"loss", "kl_loss", "recon_loss"}
    assert all(bool(tf.math.is_finite(value)) for value in train_result.values())
    assert any(
        not np.array_equal(before, after.numpy())
        for before, after in zip(weights_before_train, 
                                unconditioned.trainable_weights)
    )
    weights_before_test = [
        weight.numpy().copy() 
        for weight in unconditioned.trainable_weights
    ]
    test_result = unconditioned.test_step(x)
    assert set(test_result) == {"loss", "kl_loss", "recon_loss"}
    assert all(bool(tf.math.is_finite(value)) for value in test_result.values())
    for before, after in zip(weights_before_test, 
                            unconditioned.trainable_weights):
        np.testing.assert_array_equal(before, after.numpy())

    generated = unconditioned.generate(classes=[999], samples_per_class=3)
    assert isinstance(generated, np.ndarray)
    assert generated.shape == (3, 4) and generated.dtype == np.float32
    generated_zero = unconditioned.generate(samples_per_class=0)
    assert generated_zero.shape == (0, 4)
    try:
        unconditioned.generate(samples_per_class=-1)
    except (tf.errors.InvalidArgumentError, ValueError):
        pass
    else:
        raise AssertionError("A negative generation count must fail.")

    sigmoid_vae = VariationalAutoencoder(
        data_dim=3, 
        latent_dim=1, 
        hiddens_dims=(), 
        last_activation="sigmoid", 
        compile=False, 
    )
    sigmoid_samples = sigmoid_vae.generate(samples_per_class=2)
    assert sigmoid_samples.shape == (2, 3)
    assert np.all(sigmoid_samples >= 0.0) and np.all(sigmoid_samples <= 1.0)

    conditioned = VariationalAutoencoder(
        data_dim=4, 
        latent_dim=2, 
        hiddens_dims=(4,), 
        hiddens_kwargs={"actv": "prelu", "use_batch_norm": True}, 
        last_activation="tanh", 
        beta=0.25, 
        conditioned=True, 
        class_num=3, 
        compile_args={
            "optimizer": tf.keras.optimizers.SGD(learning_rate=0.01), 
            "loss": "mean_squared_error", 
            "run_eagerly": True, 
        },
    )
    y = tf.one_hot([0, 2], depth=3)
    (cond_mean, cond_log_var, cond_z), cond_reconstruction = conditioned(
        (x, y), 
        training=True
    )
    assert cond_mean.shape == cond_log_var.shape == cond_z.shape == (2, 2)
    assert cond_reconstruction.shape == (2, 4)
    assert bool(tf.reduce_all(cond_reconstruction <= 1.0))
    assert bool(tf.reduce_all(cond_reconstruction >= -1.0))
    conditioned.reset_metrics()
    cond_train_result = conditioned.train_step((x, y))
    assert set(cond_train_result) == {"loss", "kl_loss", "recon_loss"}
    cond_test_result = conditioned.test_step((x, y))
    assert set(cond_test_result) == {"loss", "kl_loss", "recon_loss"}

    assert conditioned.generate(classes=[], samples_per_class=2) == ([], [])
    conditioned.seen_classes = [1]
    seen_x, seen_y = conditioned.generate(classes=None, samples_per_class=2)
    assert seen_x.shape == (2, 4)
    np.testing.assert_array_equal(seen_y, np.array([1, 1]))
    explicit_x, explicit_y = conditioned.generate(
        classes=[2, 0], 
        samples_per_class=2, 
        onehot_y_output=False
    )
    assert explicit_x.shape == (4, 4)
    np.testing.assert_array_equal(explicit_y, np.array([2, 2, 0, 0]))
    onehot_x, onehot_y = conditioned.generate(
        classes=[0, 2], 
        samples_per_class=1, 
        onehot_y_output=True
    )
    assert onehot_x.shape == (2, 4)
    assert onehot_y.shape == (2, 3) and onehot_y.dtype == np.float32
    np.testing.assert_array_equal(onehot_y, np.eye(3, dtype=np.float32)[[0, 2]])
    zero_cond_x, zero_cond_y = conditioned.generate(
        classes=[1], 
        samples_per_class=0, 
        onehot_y_output=True
    )
    assert zero_cond_x.shape == (0, 4) and zero_cond_y.shape == (0, 3)
    invalid_id_x, invalid_id_y = conditioned.generate(
        classes=[3], 
        samples_per_class=1, 
        onehot_y_output=True
    )
    assert invalid_id_x.shape == (1, 4)
    np.testing.assert_array_equal(invalid_id_y, np.zeros((1, 3), np.float32))

    with TemporaryDirectory() as temp_dir:
        weights_path = Path(temp_dir) / "vae.weights.h5"
        unconditioned.save_weights(weights_path)
        weight_clone = VariationalAutoencoder(
            data_dim=4, 
            latent_dim=2, 
            hiddens_dims=(5,), 
            hiddens_kwargs={
                "actv": "relu", 
                "use_batch_norm": False, 
                "kernel_init": "glorot_uniform", 
            }, 
            last_activation=None, 
            beta=0.5, 
            compile=False, 
        )
        weight_clone(x, training=False)
        weight_clone.load_weights(weights_path)
        source_mean, source_log_var, _ = unconditioned.encoder(x, training=False)
        clone_mean, clone_log_var, _ = weight_clone.encoder(x, training=False)
        tf.debugging.assert_near(source_mean, clone_mean)
        tf.debugging.assert_near(source_log_var, clone_log_var)
        fixed_z = tf.zeros((2, 2), tf.float32)
        tf.debugging.assert_near(
            unconditioned.decoder(fixed_z, training=False), 
            weight_clone.decoder(fixed_z, training=False), 
        )

    fit_history = SimpleNamespace(history={"loss": [1.0]})
    sentinel_callback = tf.keras.callbacks.Callback()
    classifier = tf.keras.Sequential([
        layers.Input(shape=(4,)), 
        layers.Dense(3, activation="softmax", kernel_initializer="zeros"), 
    ])
    module = sys.modules[__name__]
    x_numpy = x.numpy()
    y_numpy = y.numpy()

    with mock.patch.object(
        VariationalAutoencoder, "fit", autospec=True, return_value=fit_history
    ) as fit_mock, mock.patch.object(
        module, "get_callbacks", return_value=[sentinel_callback]
    ) as callbacks_mock:
        history = unconditioned.train(
            x_numpy, 
            train_num=-1, 
            epochs=2, 
            batch_size=1, 
            validation_data=x_numpy, 
            callbacks_monitor="custom_metric", 
            verbose=0, 
        )
        assert history == {"loss": [1.0]}
        callbacks_mock.assert_called_once_with(monitor="custom_metric", verbose=0)
        fit_args, fit_kwargs = fit_mock.call_args
        assert fit_args[0] is unconditioned
        assert isinstance(fit_args[1], tf.data.Dataset)
        train_values = np.concatenate(
            list(fit_args[1].as_numpy_iterator()),
            axis=0
        )
        train_values = train_values[np.argsort(train_values[:, 0])]
        expected_values = x_numpy[np.argsort(x_numpy[:, 0])]
        np.testing.assert_array_equal(train_values, expected_values)
        assert fit_kwargs["epochs"] == 2 and "batch_size" not in fit_kwargs
        assert isinstance(fit_kwargs["validation_data"], tf.data.Dataset)
        validation_values = np.concatenate(
            list(fit_kwargs["validation_data"].as_numpy_iterator()),
            axis=0
        )
        np.testing.assert_array_equal(validation_values, x_numpy)
        assert fit_kwargs["callbacks"] == [sentinel_callback]

    explicit_callback = tf.keras.callbacks.Callback()
    explicit_valset = get_dataset(
        x_numpy,
        shuffle_buffer=0,
        batch_size=2,
        drop_remainder=False
    )
    with mock.patch.object(
        VariationalAutoencoder, "fit", autospec=True, return_value=fit_history
    ) as fit_mock, mock.patch.object(module, "get_callbacks") as callbacks_mock:
        history = unconditioned.train(
            x_numpy, 
            train_num=-1, 
            epochs=1, 
            validation_data=explicit_valset,
            callbacks_list=[explicit_callback], 
            verbose=0, 
        )
        assert history == {"loss": [1.0]}
        callbacks_mock.assert_not_called()
        assert fit_mock.call_args.kwargs["callbacks"] == [explicit_callback]
        assert fit_mock.call_args.kwargs["validation_data"] is explicit_valset

    deterministic_indices = np.array([1, 0, 1, 0, 1], dtype=np.int64)
    conditioned.seen_classes = []
    with mock.patch.object(
        VariationalAutoencoder, "fit", autospec=True, return_value=fit_history
    ) as fit_mock, mock.patch.object(
        module, "get_callbacks", return_value=[sentinel_callback]
    ) as callbacks_mock, mock.patch.object(
        np.random, "randint", return_value=deterministic_indices
    ) as randint_mock:
        history = conditioned.train(
            x_numpy, 
            y_numpy, 
            train_num=5, 
            epochs=1, 
            batch_size=2, 
            clf=classifier, 
            verbose=0, 
        )
        assert history == {"loss": [1.0]}
        randint_mock.assert_called_once_with(0, 2, (5,))
        callbacks_mock.assert_called_once_with(
            monitor="decoder_accuracy", verbose=0
        )
        fit_args, fit_kwargs = fit_mock.call_args
        assert isinstance(fit_args[1], tf.data.Dataset)
        train_batches = list(fit_args[1].as_numpy_iterator())
        actual_x = np.concatenate([batch[0] for batch in train_batches], axis=0)
        actual_y = np.concatenate([batch[1] for batch in train_batches], axis=0)
        actual_rows = np.concatenate([actual_x, actual_y], axis=-1)
        expected_rows = np.concatenate([
            x_numpy[deterministic_indices],
            y_numpy[deterministic_indices]
        ], axis=-1)
        actual_rows = actual_rows[np.lexsort(actual_rows.T[::-1])]
        expected_rows = expected_rows[np.lexsort(expected_rows.T[::-1])]
        np.testing.assert_array_equal(actual_rows, expected_rows)
        assert isinstance(fit_kwargs["callbacks"][0], DecoderAccuracyCallback)
        assert fit_kwargs["callbacks"][0].classifier is classifier
        assert fit_kwargs["callbacks"][1] is sentinel_callback
        assert set(int(item) for item in conditioned.seen_classes) == {0, 2}

    lower_count_indices = np.array([0, 1], dtype=np.int64)
    with mock.patch.object(
        VariationalAutoencoder, "fit", autospec=True, return_value=fit_history
    ) as fit_mock, mock.patch.object(module, "get_callbacks") as callbacks_mock, \
            mock.patch.object(
                np.random, "randint", return_value=lower_count_indices
            ) as randint_mock:
        conditioned.train(
            x_numpy, 
            y_numpy, 
            train_num=1, 
            callbacks_list=[explicit_callback], 
            clf=classifier, 
            verbose=0, 
        )
        randint_mock.assert_called_once_with(0, 2, (2,))
        callbacks_mock.assert_not_called()
        assert isinstance(fit_mock.call_args.args[1], tf.data.Dataset)
        fit_callbacks = fit_mock.call_args.kwargs["callbacks"]
        assert isinstance(fit_callbacks[0], DecoderAccuracyCallback)
        assert fit_callbacks[1] is explicit_callback

    try:
        unconditioned.train(x_numpy, y_numpy, train_num=-1, verbose=0)
    except AssertionError:
        pass
    else:
        raise AssertionError("Unconditional training must reject labels.")
    try:
        conditioned.train(x_numpy, None, train_num=-1, verbose=0)
    except AssertionError:
        pass
    else:
        raise AssertionError("Conditional training must require labels.")
    try:
        unconditioned.train(np.empty((0, 4), np.float32), train_num=0, verbose=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Resampling an empty input must fail.")

    tf.keras.backend.clear_session()
    return {"VariationalAutoencoder": "passed"}


if __name__ == "__main__":
    print(run_self_tests())
