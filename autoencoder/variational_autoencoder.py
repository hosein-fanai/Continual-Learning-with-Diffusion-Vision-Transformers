"""Dense variational autoencoder with optional class conditioning and replay."""

import tensorflow as tf
from tensorflow.keras import metrics, layers, models, optimizers

import numpy as np

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

        encoder = models.Model(inputs, [z_mean, z_log_var, z], name="encoder")

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

        outputs = layers.Dense(output_dim, activation=last_activation)(z)

        decoder = models.Model(inputs, outputs, name="decoder")

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

        return (z_mean, z_log_var, z), self.decoder(decoder_inputs, training=training)

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

        z = tf.random.normal(shape=(samples_per_class, self.latent_dim))
        x = self.decoder(z, training=False)

        x = x.numpy()
        
        return x

    def train(self, x, y=None, 
            train_num=10_000, 
            epochs=10, batch_size=512, 
            validation_data=None, 
            callbacks_list=None, 
            callbacks_monitor="",
            clf=None, verbose=1):
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
            batch_size (int): Positive examples per NumPy-input batch.
            validation_data (object | None): Keras validation input.  For a
                conditional VAE this is normally ``(x_val, y_val)``; for an
                unconditional VAE it may be an ``x_val`` tensor/dataset.
            callbacks_list (list[tf.keras.callbacks.Callback] | None): Exact
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
        assert (self.conditioned and (y is not None)) or (not self.conditioned and (y is None)) 

        if train_num != -1:
            input_size = len(x)
            train_num = max(train_num, input_size)

            indices = np.random.randint(0, input_size, (train_num,))
            x = x[indices]
            if y is not None:
                y = y[indices]

        if y is not None:
            new_classes = np.unique(np.argmax(y, axis=-1))
            self.seen_classes.extend(new_classes)
            self.seen_classes = list(set(self.seen_classes))

        if clf is not None and callbacks_list is None:
            callbacks_monitor = "decoder_accuracy" if callbacks_monitor == "" else callbacks_monitor

            callbacks_list = [
                DecoderAccuracyCallback(classifier=clf)
            ] + get_callbacks(monitor=callbacks_monitor, verbose=verbose)
        elif clf is not None and callbacks_list is not None:
            callbacks_list = [
                DecoderAccuracyCallback(classifier=clf)
            ] + callbacks_list
        elif clf is None and callbacks_list is None:
            callbacks_list = get_callbacks(monitor=callbacks_monitor, verbose=verbose)

        history = self.fit(
            x, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data, 
            callbacks=callbacks_list, 
            verbose=verbose,
        ).history

        return history


if __name__ == "__main__":
    from common.dataloader import load_cifar10
    from common.utils import init


    init()

    x_train, y_train, *_ = load_cifar10(return_features=True, 
                                        onehot_labels=True, 
                                        preprocess="normalize", 
                                        verbose=0)

    vae = VariationalAutoencoder(conditioned=True, class_num=10)

    vae.train(
        x_train, y_train, 
        train_num=-1, 
        clf=models.load_model("./models/hyperas/cifar10_dnn_model_00B.h5")
    )
