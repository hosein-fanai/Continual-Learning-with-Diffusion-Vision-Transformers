"""Dense variational autoencoder with optional class conditioning and replay."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import metrics, layers, models, optimizers

import numpy as np

from collections.abc import Callable, Mapping, Sequence

from common.dataloader import get_dataset
from common.gradients import apply_policy_gradients
from common.keras_registry import register_canonical_keras_serializable
from common.model import get_callbacks
from common.runtime import derive_seed

from autoencoder.decoder_accuracy_callback import DecoderAccuracyCallback


@register_canonical_keras_serializable(package="continual_learning")
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
        self: VariationalAutoencoder, 
        data_dim: int = 2048, 
        latent_dim: int = 8, 
        hiddens_dims: Sequence[int] = (16,), 
        hiddens_kwargs: Mapping[str, object] | None = None, 
        last_activation: str | Callable | None = "tanh", 
        beta: float = 0.25, 
        conditioned: bool = False, 
        class_num: int | None = None, 
        compile: bool = True, 
        compile_args: Mapping[str, object] | None = None, 
        seed: int | None = None,
        **kwargs: object
    ) -> None:
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
            beta (float): Finite, nonnegative KL-loss multiplier applied to
                the latent regularization term.
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
            seed (int | None): Optional experiment seed for the VAE's explicit
                reparameterization operation and default generation/training
                streams. Continual task-level global reseeding combines with
                this component seed to reproduce its stateful draw sequence.
            **kwargs (object): Standard ``tf.keras.Model`` constructor options,
                such as ``name``, ``trainable``, and ``dtype``.

        Returns:
            None.

        Raises:
            ValueError: If conditioning settings are inconsistent, a model
                width is nonpositive, or ``beta`` would invalidate the loss.
            TypeError: If either keyword mapping contains unsupported keys.
        """

        super().__init__(**kwargs)

        conditioned = bool(conditioned)
        compile = bool(compile)
        # Keep class metadata consistent with the conditioning mode.
        if (conditioned and class_num is None) \
        or (not conditioned and class_num is not None):
            raise ValueError(
                "When conditioned is True, class_num cannot be None, and when "
                "conditioned is False, class_num needs to be None."
            )
        data_dim = int(data_dim)
        latent_dim = int(latent_dim)
        hiddens_dims = tuple(int(hidden_dim) for hidden_dim in hiddens_dims)
        class_num = int(class_num) if class_num is not None else None
        if data_dim <= 0 or latent_dim <= 0 \
        or any(hidden_dim <= 0 for hidden_dim in hiddens_dims) \
        or (conditioned and class_num <= 0):
            raise ValueError("VAE dimensions must be positive.")
        if not np.isfinite(beta) or beta < 0.:
            raise ValueError("beta must be finite and nonnegative.")

        hiddens_kwargs = dict(hiddens_kwargs or {})
        allowed_hidden_kwargs = {
            "actv", "use_batch_norm", "kernel_init"
        }
        unknown_hidden_kwargs = set(hiddens_kwargs) - allowed_hidden_kwargs
        # Validate the public hidden-block mapping even when no hidden layer is
        # requested; otherwise an empty architecture silently accepts typos.
        if unknown_hidden_kwargs:
            raise TypeError(
                "Unsupported hiddens_kwargs: "
                f"{sorted(unknown_hidden_kwargs)}."
            )
        # Normalize NumPy boolean scalars into a stable Python config value.
        if "use_batch_norm" in hiddens_kwargs:
            hiddens_kwargs["use_batch_norm"] = bool(
                hiddens_kwargs["use_batch_norm"]
            )


        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hiddens_dims = hiddens_dims
        # Constructor callables/configuration are metadata, not additional
        # checkpoint dependencies beyond the layers built from them below.
        self.hiddens_kwargs = self._no_dependency(dict(hiddens_kwargs))
        self.last_activation = self._no_dependency(last_activation)
        self.beta = beta
        self.conditioned = conditioned
        self.class_num = class_num
        # Keep reparameterization's operation seed independent from dataset
        # shuffling and callback generation. A stateful op remains compatible
        # with TF 2.10 Functional KerasTensors; task-level runtime reseeding
        # resets its deterministic sequence at each recovery boundary.
        self.reparameterization_seed = derive_seed(
            seed,
            "vae",
            "reparameterization",
        )
        self.seed = None if seed is None else int(seed)

        self.encoder = self._build_encoder(data_dim, latent_dim, hiddens_dims, 
                                        hiddens_kwargs, class_num)
        self.decoder = self._build_decoder(data_dim, latent_dim, hiddens_dims[::-1], 
                                        hiddens_kwargs, class_num, last_activation)

        self.seen_classes = []

        stable_dtype = self.dtype_policy.variable_dtype
        self.total_loss_tracker = metrics.Mean(
            name="total_loss",
            dtype=stable_dtype,
        )
        self.kl_loss_tracker = metrics.Mean(
            name="kl_loss",
            dtype=stable_dtype,
        )
        self.recon_loss_tracker = metrics.Mean(
            name="recon_loss",
            dtype=stable_dtype,
        )

        compile_args_default = {
            "optimizer": optimizers.Nadam(learning_rate=0.1, decay=0.),
            "loss": "mean_squared_error",
        }
        compile_args = {**compile_args_default, **(compile_args or {})}

        # Compile immediately when requested by the caller.
        if compile:
            self.compile(**compile_args)

    @staticmethod
    def _serialize_activation(
        activation: str | Callable | None,
    ) -> object:
        """Serialize one activation while preserving simple public values.

        Args:
            activation (str | Callable | None): Constructor activation value.

        Returns:
            object: JSON-compatible Keras activation configuration.
        """

        # Strings (including the special hidden-block ``prelu`` spelling) and
        # None are already stable constructor values.
        if activation is None or isinstance(activation, str):
            return activation

        return tf.keras.activations.serialize(activation)

    @staticmethod
    def _deserialize_activation(config: object) -> object:
        """Restore an activation serialized by :meth:`_serialize_activation`.

        Args:
            config (object): Serialized activation configuration.

        Returns:
            object: Activation value accepted by Keras Dense/Activation.
        """

        # Preserve already-stable public activation spellings unchanged.
        if config is None or isinstance(config, str):
            return config

        return tf.keras.activations.deserialize(config)

    def get_config(self: VariationalAutoencoder) -> dict[str, object]:
        """Return a complete, architecture-preserving Keras configuration.

        Compilation state is intentionally excluded from architecture config;
        Keras serializes it separately for full-model formats. Direct
        ``from_config`` and ``clone_model`` reconstruction therefore produce an
        uncompiled model with the same encoder/decoder topology.

        Returns:
            dict[str, object]: Standard model state and every constructor value
            required to recreate the VAE architecture.
        """

        config = super().get_config()
        # TensorFlow 2.10 can omit these fields for subclassed models.
        config.setdefault("name", self.name)
        config.setdefault("trainable", self.trainable)
        config.setdefault("dtype", self.dtype_policy.name)
        config.setdefault("dynamic", self.dynamic)

        hidden_kwargs = dict(self.hiddens_kwargs)
        # Encode an optional callable activation for Keras reconstruction.
        if "actv" in hidden_kwargs:
            hidden_kwargs["actv"] = self._serialize_activation(
                hidden_kwargs["actv"]
            )
        # Encode an optional initializer through Keras' registered format.
        if "kernel_init" in hidden_kwargs:
            hidden_kwargs["kernel_init"] = tf.keras.initializers.serialize(
                tf.keras.initializers.get(hidden_kwargs["kernel_init"])
            )

        config.update({
            "data_dim": self.data_dim,
            "latent_dim": self.latent_dim,
            "hiddens_dims": list(self.hiddens_dims),
            "hiddens_kwargs": hidden_kwargs,
            "last_activation": self._serialize_activation(
                self.last_activation
            ),
            "beta": float(self.beta),
            "conditioned": self.conditioned,
            "class_num": self.class_num,
            "compile": False,
            "compile_args": None,
            "seed": self.seed,
        })

        return config

    @classmethod
    def _deserialize_constructor_config(
        cls: type[VariationalAutoencoder],
        config: Mapping[str, object],
    ) -> dict[str, object]:
        """Normalize serialized activation/initializer values for construction.

        Args:
            config (Mapping[str, object]): Output of :meth:`get_config`.

        Returns:
            dict[str, object]: Independent keyword mapping accepted by
            :meth:`__init__`.
        """

        restored = dict(config)
        hidden_kwargs = dict(restored.get("hiddens_kwargs") or {})
        # Restore an encoded hidden activation before constructor dispatch.
        if "actv" in hidden_kwargs:
            hidden_kwargs["actv"] = cls._deserialize_activation(
                hidden_kwargs["actv"]
            )
        # Restore an encoded initializer while retaining stable string names.
        if "kernel_init" in hidden_kwargs:
            initializer_config = hidden_kwargs["kernel_init"]
            hidden_kwargs["kernel_init"] = (
                initializer_config
                if isinstance(initializer_config, str)
                else tf.keras.initializers.deserialize(initializer_config)
            )
        restored["hiddens_kwargs"] = hidden_kwargs
        # Restore the output activation when the serialized field is present.
        if "last_activation" in restored:
            restored["last_activation"] = cls._deserialize_activation(
                restored["last_activation"]
            )

        return restored

    @classmethod
    def from_config(
        cls: type[VariationalAutoencoder],
        config: Mapping[str, object],
    ) -> VariationalAutoencoder:
        """Recreate a VAE from its complete architecture configuration.

        Args:
            config (Mapping[str, object]): Output of :meth:`get_config`.

        Returns:
            VariationalAutoencoder: Independent uncompiled architecture clone.
        """

        return cls(**cls._deserialize_constructor_config(config))

    def _dense_layer(
        self: VariationalAutoencoder, 
        units: int, 
        actv: str | Callable = "selu", 
        use_batch_norm: bool = True, 
        kernel_init: object = "he_normal"
    ) -> tf.keras.Sequential:
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
        # Give every Dense layer an independent initializer instance. Keras
        # 2.10 otherwise repeats draws when one deserialized unseeded
        # initializer object is reused across encoder and decoder blocks.
        dense_initializer = tf.keras.initializers.deserialize(
            tf.keras.initializers.serialize(
                tf.keras.initializers.get(kernel_init)
            )
        )

        dlayer.add(layers.Dense(
            units, 
            activation=actv if not(use_batch_norm or actv == "prelu") else "linear", 
            kernel_initializer=dense_initializer,
            use_bias=not use_batch_norm,
            dtype=self.dtype_policy,
        ))
        dlayer.add(
            layers.Activation(actv, dtype=self.dtype_policy)
        ) if (use_batch_norm and actv != "prelu") else None
        dlayer.add(
            layers.PReLU(dtype=self.dtype_policy)
        ) if actv == "prelu" else None
        dlayer.add(
            layers.BatchNormalization(dtype=self.dtype_policy)
        ) if use_batch_norm else None

        return dlayer

    def _build_encoder(
        self: VariationalAutoencoder, 
        input_dim: int, 
        latent_dim: int, 
        hiddens_dims: Sequence[int], 
        hiddens_kwargs: Mapping[str, object] | None = None, 
        class_num: int | None = None
    ) -> tf.keras.Model:
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

        hiddens_kwargs = dict(hiddens_kwargs or {})
        x_inputs = layers.Input(
            shape=(input_dim,),
            dtype=self.compute_dtype,
            name="x_input",
        )

        # Concatenate one-hot labels into a conditional encoder input.
        if self.conditioned:
            y_inputs = layers.Input(
                shape=(class_num,),
                dtype=self.compute_dtype,
                name="y_input",
            )
            x = layers.Concatenate(dtype=self.dtype_policy)([x_inputs, y_inputs])
            inputs = [x_inputs, y_inputs]
        # Use only data features for an unconditional encoder.
        else:
            x = x_inputs
            inputs = x_inputs

        for hidden_dim in hiddens_dims:
            x = self._dense_layer(hidden_dim, **hiddens_kwargs)(x)

        z_mean = layers.Dense(
            latent_dim,
            dtype=self.dtype_policy,
            name="z_mean",
        )(x)
        z_log_var = layers.Dense(
            latent_dim,
            dtype=self.dtype_policy,
            name="z_log_var",
        )(x)
        z = VariationalAutoencoder.compute_z(
            z_mean,
            z_log_var,
            seed=self.reparameterization_seed,
            dtype=self.dtype_policy.variable_dtype,
        )

        encoder = models.Model(
            inputs, 
            [z_mean, z_log_var, z], 
            name="encoder"
        )

        return encoder

    def _build_decoder(
        self: VariationalAutoencoder, 
        output_dim: int, 
        latent_dim: int, 
        hiddens_dims: Sequence[int], 
        hiddens_kwargs: Mapping[str, object], 
        class_num: int | None, 
        last_activation: str | Callable | None
    ) -> tf.keras.Model:
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

        z_inputs = layers.Input(
            shape=(latent_dim,),
            dtype=self.compute_dtype,
            name="z_input",
        )

        # Concatenate one-hot labels into a conditional decoder input.
        if self.conditioned:
            y_inputs = layers.Input(
                shape=(class_num,),
                dtype=self.compute_dtype,
                name="y_input",
            )
            z = layers.Concatenate(dtype=self.dtype_policy)([z_inputs, y_inputs])
            inputs = [z_inputs, y_inputs]
        # Decode directly from latent samples in unconditional mode.
        else:
            z = z_inputs
            inputs = z_inputs

        for hidden_dim in hiddens_dims:
            z = self._dense_layer(hidden_dim, **hiddens_kwargs)(z)

        outputs = layers.Dense(
            output_dim, 
            activation=last_activation,
            dtype=self.dtype_policy,
        )(z)

        decoder = models.Model(
            inputs, 
            outputs, 
            name="decoder"
        )

        return decoder

    @staticmethod
    def compute_z(
        z_mean: tf.Tensor, 
        z_log_var: tf.Tensor,
        seed: int | None = None,
        dtype: tf.dtypes.DType | str | None = None,
    ) -> tf.Tensor:
        """Sample a latent vector with the reparameterization trick.

        Args:
            z_mean (tf.Tensor): Floating Gaussian means shaped
                ``[batch, latent_dim]``.
            z_log_var (tf.Tensor): Matching floating elementwise log variances.
            seed (int | None): Optional stateful TensorFlow operation seed.
            dtype (tf.dtypes.DType | str | None): Stable calculation dtype.
                None preserves ``z_mean.dtype``. Mixed-precision callers should
                pass their policy's variable dtype.

        Returns:
            tf.Tensor: Values cast back to ``z_mean``'s dtype, computed as
            ``z_mean + exp(0.5*z_log_var) * epsilon`` with the same shape;
            ``epsilon`` is newly drawn from a standard normal distribution on
            every call.
        """

        output_dtype = z_mean.dtype
        stable_dtype = tf.as_dtype(dtype or output_dtype)
        stable_mean = tf.cast(z_mean, stable_dtype)
        stable_log_var = tf.cast(z_log_var, stable_dtype)
        epsilon = tf.random.normal(
            shape=tf.shape(stable_mean),
            dtype=stable_dtype,
            seed=seed,
        )
        z = stable_mean + tf.exp(0.5 * stable_log_var) * epsilon

        return tf.cast(z, output_dtype)

    @staticmethod
    def compute_kl(
        z_mean: tf.Tensor | Sequence[tuple[tf.Tensor, tf.Tensor]], 
        z_log_var: tf.Tensor | None = None, 
        sample_weight: tf.Tensor | None = None, 
        dtype: tf.dtypes.DType | str | None = None
    ) -> tf.Tensor:
        """Compute weighted KL divergence from one or more Gaussian latents.

        Args:
            z_mean (tf.Tensor | Sequence[tuple[tf.Tensor, tf.Tensor]]): One
                rank-two mean tensor, or an ordered sequence of
                ``(mean, log_variance)`` pairs for multiple latent sites.
            z_log_var (tf.Tensor | None): Log variances matching a single
                ``z_mean`` tensor. Leave None when ``z_mean`` is a sequence.
            sample_weight (tf.Tensor | None): Optional per-row weights. The
                weighted result divides by their sum and is zero when every
                weight is zero.
            dtype (tf.dtypes.DType | str | None): Stable calculation dtype.
                None preserves the first mean tensor's dtype.

        Returns:
            tf.Tensor: Scalar floating tensor. Divergence is summed over every
            latent dimension and site, then averaged across batch rows.
            An empty sequence returns zero.
        """

        z_vals_list = tuple(z_mean) if z_log_var is None else ((z_mean, z_log_var),)
        stable_dtype = tf.as_dtype(
            dtype or (
                z_vals_list[0][0].dtype
                if z_vals_list else tf.keras.backend.floatx()
            )
        )

        if not z_vals_list:
            return tf.constant(0., dtype=stable_dtype)

        kl_rows = tf.add_n([
            -0.5 * tf.reduce_sum(
                1. + tf.cast(log_var, stable_dtype)
                - tf.square(tf.cast(mean, stable_dtype))
                - tf.exp(tf.cast(log_var, stable_dtype)), 
                axis=-1
            )
            for mean, log_var in z_vals_list
        ])

        if sample_weight is None:
            return tf.reduce_mean(kl_rows)

        sample_weight = tf.cast(
            tf.reshape(sample_weight, (-1,)),
            stable_dtype
        )

        return tf.math.divide_no_nan(
            tf.reduce_sum(kl_rows * sample_weight), 
            tf.reduce_sum(sample_weight)
        )

    @property
    def metrics(
        self: VariationalAutoencoder
    ) -> list[tf.keras.metrics.Metric]:
        """Expose loss trackers that Keras resets between epochs/evaluations.

        Returns:
            list[tf.keras.metrics.Metric]: Total, KL, and reconstruction
            trackers followed by any reconstruction metrics supplied to
            :meth:`compile`.
        """

        compiled_metrics = self.compiled_metrics.metrics \
            if self.compiled_metrics is not None else []

        return [
            self.total_loss_tracker, 
            self.kl_loss_tracker, 
            self.recon_loss_tracker,
            *compiled_metrics
        ]

    def call(
        self: VariationalAutoencoder, 
        inputs: tf.Tensor | tuple[tf.Tensor, tf.Tensor], 
        training: bool | tf.Tensor = False, 
    ) -> tuple[tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
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

        # Reattach labels to latent samples for conditional decoding.
        if self.conditioned:
            _, y = inputs
            decoder_inputs = (z, y)
        # Decode only the latent sample in unconditional mode.
        else:
            decoder_inputs = z

        return ((z_mean, z_log_var, z), 
                self.decoder(decoder_inputs, training=training))

    def train_step(
        self: VariationalAutoencoder, 
        data: tf.Tensor | tuple[tf.Tensor, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """Run one optimizer step for reconstruction plus beta-weighted KL.

        Args:
            data (tf.Tensor | tuple[tf.Tensor, tf.Tensor]): Unconditional
                feature batch (with optional ignored labels), or conditional
                ``(x, one_hot_y)`` as supplied by Keras ``fit``.

        Returns:
            dict[str, tf.Tensor]: Scalar running means under ``loss``,
            ``kl_loss``, and ``recon_loss``, plus configured reconstruction
            metrics.
        """

        # Extract reconstruction targets from conditional training pairs.
        if self.conditioned:
            x, _ = data
        # Ignore optional supervised labels in unconditional Keras pipelines.
        else:
            x = data[0] if isinstance(data, (tuple, list)) else data

        with tf.GradientTape() as tape:
            model_inputs = data if self.conditioned else x
            (z_mean, z_log_var, _), x_recon = self(
                model_inputs, training=True
            )

            stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
            recon_loss = tf.cast(self.compiled_loss(
                x, 
                x_recon, 
                regularization_losses=self.losses
            ), stable_dtype)
            kl_loss = VariationalAutoencoder.compute_kl(
                z_mean, 
                z_log_var,
                dtype=stable_dtype,
            )

            total_loss = recon_loss + tf.cast(self.beta, stable_dtype) * kl_loss

        apply_policy_gradients(
            tape,
            self.optimizer,
            total_loss,
            self.trainable_weights,
        )

        batch_weight = tf.cast(tf.shape(x)[0], stable_dtype)
        self.total_loss_tracker.update_state(
            total_loss, sample_weight=batch_weight
        )
        self.recon_loss_tracker.update_state(
            recon_loss, sample_weight=batch_weight
        )
        self.kl_loss_tracker.update_state(
            kl_loss, sample_weight=batch_weight
        )
        self.compiled_metrics.update_state(x, x_recon)

        results = {
            "loss": self.total_loss_tracker.result(), 
            "kl_loss": self.kl_loss_tracker.result(), 
            "recon_loss": self.recon_loss_tracker.result()
        }
        results.update({
            metric.name: metric.result()
            for metric in self.compiled_metrics.metrics
        })

        return results

    def test_step(
        self: VariationalAutoencoder, 
        data: tf.Tensor | tuple[tf.Tensor, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """Evaluate reconstruction and KL losses without updating weights.

        Args:
            data (tf.Tensor | tuple[tf.Tensor, tf.Tensor]): Unconditional
                feature batch (with optional ignored labels), or conditional
                ``(x, one_hot_y)``.

        Returns:
            dict[str, tf.Tensor]: Scalar running means under ``loss``,
            ``kl_loss``, and ``recon_loss``, plus configured reconstruction
            metrics.
        """

        # Extract reconstruction targets from conditional evaluation pairs.
        if self.conditioned:
            x, _ = data
        # Ignore optional supervised labels in unconditional Keras pipelines.
        else:
            x = data[0] if isinstance(data, (tuple, list)) else data

        model_inputs = data if self.conditioned else x
        (z_mean, z_log_var, _), x_recon = self(
            model_inputs, training=False
        )

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        recon_loss = tf.cast(self.compiled_loss(
            x, 
            x_recon, 
            regularization_losses=self.losses
        ), stable_dtype)
        kl_loss = VariationalAutoencoder.compute_kl(
            z_mean, 
            z_log_var,
            dtype=stable_dtype,
        )

        total_loss = tf.cast(self.beta, stable_dtype) * kl_loss + recon_loss

        batch_weight = tf.cast(tf.shape(x)[0], stable_dtype)
        self.total_loss_tracker.update_state(total_loss, sample_weight=batch_weight)
        self.kl_loss_tracker.update_state(kl_loss, sample_weight=batch_weight)
        self.recon_loss_tracker.update_state(
            recon_loss, sample_weight=batch_weight
        )
        self.compiled_metrics.update_state(x, x_recon)

        results = {
            "loss": self.total_loss_tracker.result(), 
            "kl_loss": self.kl_loss_tracker.result(), 
            "recon_loss": self.recon_loss_tracker.result()
        }
        results.update({
            metric.name: metric.result()
            for metric in self.compiled_metrics.metrics
        })

        return results

    def generate(
        self: VariationalAutoencoder, 
        classes: Sequence[int] | None = None, 
        samples_per_class: int = 500, 
        onehot_y_output: bool = False,
        seed: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[list, list]:
        """Decode random normal latents into synthetic replay samples.

        Args:
            classes (Sequence[int] | None): Conditional
                class IDs. ``None``
                uses ``seen_classes``; ``[]`` returns ``([], [])`` immediately.
                IDs should lie in ``[0, class_num)``.  In unconditional mode
                this argument is ignored.
            samples_per_class (int): Number of examples per class in
                conditional mode, or total examples in unconditional mode.
            onehot_y_output (bool): In conditional mode, return labels as
                one-hot rows in the policy's stable variable dtype when true,
                or NumPy integer IDs when false. Ignored in unconditional mode.
            seed (int | None): Optional stateless latent-sampling seed. The
                same seed and arguments reproduce the same latent batch.

        Returns:
            numpy.ndarray | tuple[numpy.ndarray, numpy.ndarray] | tuple[list,
            list]: Unconditional mode returns samples shaped
            ``[samples_per_class, data_dim]``.  Conditional mode returns
            ``(x, y)`` with ``len(classes) * samples_per_class`` rows, ordered
            in contiguous class groups.  With no conditional classes, both
            outputs are empty Python lists rather than arrays.

        Raises:
            ValueError: If a class ID is outside ``[0, class_num)``.
        """

        # Resolve the model seed only when the caller does not provide a
        # task/callback-specific stream.
        generation_seed = self.seed if seed is None else seed
        conditional_seed = derive_seed(
            generation_seed,
            "vae",
            "generate",
            "conditional",
        )
        unconditional_seed = derive_seed(
            generation_seed,
            "vae",
            "generate",
            "unconditional",
        )

        samples_per_class = int(samples_per_class)

        # Generate and label each requested class in conditional mode.
        if self.conditioned:
            # Default to classes observed during training.
            if classes is None:
                classes = self.seen_classes

            # Return an empty result when no class is available to generate.
            if len(classes) == 0:
                return [], []

            classes = [int(class_id) for class_id in classes]
            # Keep class identifiers within the configured output range.
            if any(
                class_id < 0 or class_id >= int(self.class_num)
                for class_id in classes
            ):
                raise ValueError("Every class ID must lie in [0, class_num).")

            latent_shape = (samples_per_class * len(classes), self.latent_dim)
            stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
            z = tf.random.normal(
                shape=latent_shape,
                dtype=stable_dtype,
            ) if conditional_seed is None else tf.random.stateless_normal(
                latent_shape,
                seed=[conditional_seed, 0],
                dtype=stable_dtype,
            )
            y = tf.concat([tf.one_hot(tf.cast([i]*samples_per_class, tf.int32),
                                    depth=self.class_num,
                                    dtype=stable_dtype) for i in classes],
                                axis=0)
            x = self.decoder((z, y), training=False)

            x = x.numpy()
            y = y.numpy()

            # Convert generated labels to sparse IDs when requested.
            if not onehot_y_output:
                y = np.argmax(y, axis=-1)

            return x, y

        latent_shape = (samples_per_class, self.latent_dim)
        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        z = tf.random.normal(
            shape=latent_shape,
            dtype=stable_dtype,
        ) if unconditional_seed is None else tf.random.stateless_normal(
            latent_shape,
            seed=[unconditional_seed, 0],
            dtype=stable_dtype,
        )
        x = self.decoder(z, training=False)

        x = x.numpy()
        
        return x

    def train(
        self: VariationalAutoencoder, 
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
        verbose: bool | int = 1,
        steps_per_epoch: int | None = None,
    ) -> dict[str, list[float]]:
        """Fit the VAE and optionally monitor generated-sample accuracy.

        Args:
            x (numpy.ndarray | tf.Tensor): Flat samples shaped
                ``[samples, data_dim]``.
            y (numpy.ndarray | tf.Tensor | None): Required only in conditional
                mode; one-hot labels shaped ``[samples, class_num]``.  Observed
                argmax class IDs are added to ``seen_classes``.
            train_num (int): ``-1`` uses the input once
                without resampling.
                Any positive value samples exactly that many indices with
                replacement, so smaller values downsample and larger values
                oversample the supplied rows.
            epochs (int): Positive maximum number of Keras training
                epochs.
            batch_size (int): Positive examples per
                ``tf.data.Dataset`` batch.
            shuffle_buffer (int): Nonnegative training shuffle
                capacity passed to :func:`common.dataloader.get_dataset`; ``0``
                disables shuffling.
            seed (int | None): Optional task seed for resampling, dataset
                shuffling, and decoder-accuracy sampling. ``None`` uses the
                model constructor seed when available.
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
            steps_per_epoch (int | None): Optional fixed update count
                per epoch. When supplied, the prepared finite dataset repeats
                so Keras can always complete the requested number of steps.

        Returns:
            dict[str, list[float]]: ``History.history`` with per-epoch total,
            KL, reconstruction, validation, learning-rate, and/or callback
            metrics depending on the supplied data and callbacks.

        Raises:
            ValueError: If label presence does not match conditional mode, the
                input is empty, or ``x``/``y`` lengths differ.
        """

        # Publish the normalized Python integer to the Keras fit call.
        if steps_per_epoch is not None:
            steps_per_epoch = int(steps_per_epoch)

        train_num = int(train_num)
        epochs = int(epochs)
        batch_size = int(batch_size)
        shuffle_buffer = int(shuffle_buffer)
        seed = self.seed if seed is None else seed
        # Validate and normalize the effective task seed.
        derive_seed(seed, "vae", "train", "validation")
        # Normalize a supplied seed for downstream APIs.
        if seed is not None:
            seed = int(seed)
        # Keep label presence consistent with the conditioning mode.
        if (self.conditioned and y is None) \
        or (not self.conditioned and y is not None):
            raise ValueError("Label presence must match the VAE conditioning mode.")
        # Reject an empty training population before sampling or fitting.
        if len(x) == 0:
            raise ValueError("Training input must contain at least one sample.")
        # Keep conditional labels aligned with their input samples.
        if y is not None and len(x) != len(y):
            raise ValueError("x and y must contain the same number of samples.")

        callbacks_list = list(callbacks_list) if callbacks_list is not None else None

        # Resample to the requested training size.
        if train_num != -1:
            input_size = len(x)
            rng = np.random.default_rng(seed)
            indices = rng.integers(0, input_size, size=train_num)
            x = tf.gather(
                x, 
                indices
            ) if isinstance(x, tf.Tensor) else x[indices]
            # Apply the same sampled indices to conditional labels.
            if y is not None:
                y = tf.gather(
                    y,
                    indices
                ) if isinstance(y, tf.Tensor) else y[indices]

        # Record classes available for conditional generation.
        if y is not None:
            new_classes = np.unique(np.argmax(y, axis=-1))
            self.seen_classes = sorted(set(
                [*self.seen_classes, *new_classes.tolist()]
            ))

        # Build classifier-aware defaults.
        if clf is not None and callbacks_list is None:
            # Select the decoder metric by default.
            if callbacks_monitor == "":
                callbacks_monitor = "decoder_accuracy"

            callbacks_list = [
                DecoderAccuracyCallback(classifier=clf, seed=seed),
                *get_callbacks(
                    monitor=callbacks_monitor, 
                    verbose=verbose
                )
            ]
        # Add decoder evaluation to user callbacks.
        elif clf is not None and callbacks_list is not None:
            callbacks_list = [
                DecoderAccuracyCallback(classifier=clf, seed=seed),
                *callbacks_list
            ]
        # Build ordinary VAE callbacks.
        elif clf is None and callbacks_list is None:
            # Select a validation-aware reconstruction monitor by default.
            if callbacks_monitor == "":
                callbacks_monitor = "val_loss" if validation_data is not None \
                    else "loss"
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
        # Repeat only under the explicit update-count protocol; the default
        # still consumes the prepared finite dataset once per epoch.
        if steps_per_epoch is not None:
            trainset = trainset.repeat()
        # Preserve prepared validation pipelines.
        if isinstance(validation_data, tf.data.Dataset):
            valset = validation_data
        # Convert paired validation arrays.
        elif isinstance(validation_data, tuple):
            # Accept paired validation arrays only for conditional models.
            if not self.conditioned:
                raise ValueError(
                    "Tuple validation_data is only valid in conditional mode."
                )
            valset = get_dataset(
                validation_data[0], 
                validation_data[1], 
                shuffle_buffer=0, 
                batch_size=batch_size, 
                drop_remainder=False
            )
        # Convert unconditional validation inputs.
        elif validation_data is not None:
            valset = get_dataset(
                validation_data, 
                shuffle_buffer=0, 
                batch_size=batch_size, 
                drop_remainder=False
            )
        # Train without validation when no input was supplied.
        else:
            valset = None

        keras_fit_kwargs = {}
        # Avoid adding a new Keras keyword in the default path so existing
        # mocked calls and third-party subclasses remain byte-for-byte stable.
        if steps_per_epoch is not None:
            keras_fit_kwargs["steps_per_epoch"] = steps_per_epoch
        history = self.fit(
            trainset, 
            epochs=epochs, 
            validation_data=valset,
            callbacks=callbacks_list, 
            verbose=verbose,
            **keras_fit_kwargs,
        ).history

        return history


# TensorFlow 2.10 may emit the plain root name for subclassed-model JSON.
tf.keras.utils.get_custom_objects()[
    "VariationalAutoencoder"
] = VariationalAutoencoder


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
        except ValueError:
            pass
        else:
            raise AssertionError("Conditioning and class_num must agree.")

    for invalid_options in (
        {"latent_dim": 0},
        {"hiddens_dims": (0,)},
        {"beta": float("nan")},
    ):
        options = {
            "data_dim": 4,
            "latent_dim": 2,
            "hiddens_dims": (),
            "compile": False,
            **invalid_options,
        }
        try:
            VariationalAutoencoder(**options)
        except ValueError:
            pass
        else:
            raise AssertionError("Degenerate VAE dimensions/losses must fail.")

    try:
        VariationalAutoencoder(
            data_dim=4, 
            latent_dim=2, 
            hiddens_dims=(),
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
            "metrics": [
                tf.keras.metrics.MeanAbsoluteError(name="recon_mae")
            ],
            "run_eagerly": True, 
        },
        name="unconditional_vae",
    )
    assert unconditioned._is_compiled is True
    assert isinstance(unconditioned.optimizer, tf.keras.optimizers.SGD)
    assert unconditioned.run_eagerly is True
    architecture_config = unconditioned.get_config()
    assert architecture_config["data_dim"] == 4
    assert architecture_config["latent_dim"] == 2
    assert architecture_config["hiddens_dims"] == [5]
    assert architecture_config["hiddens_kwargs"]["actv"] == "relu"
    assert architecture_config["last_activation"] is None
    assert architecture_config["beta"] == 0.5
    assert architecture_config["compile"] is False
    architecture_clone = VariationalAutoencoder.from_config(
        architecture_config
    )
    assert architecture_clone.data_dim == unconditioned.data_dim
    assert architecture_clone.latent_dim == unconditioned.latent_dim
    assert architecture_clone.hiddens_dims == unconditioned.hiddens_dims
    assert architecture_clone.hiddens_kwargs["actv"] == "relu"
    assert architecture_clone.last_activation is None
    assert architecture_clone.beta == unconditioned.beta
    assert architecture_clone.conditioned is False
    assert architecture_clone._is_compiled is False

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

    shared_initializer = tf.keras.initializers.GlorotUniform()
    initialized_block_a = unconditioned._dense_layer(
        3,
        use_batch_norm=False,
        kernel_init=shared_initializer,
    )
    initialized_block_b = unconditioned._dense_layer(
        3,
        use_batch_norm=False,
        kernel_init=shared_initializer,
    )
    assert initialized_block_a.layers[0].kernel_initializer \
        is not shared_initializer
    assert initialized_block_a.layers[0].kernel_initializer \
        is not initialized_block_b.layers[0].kernel_initializer

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
    weighted_means = tf.constant([
        [0., 0.],
        [1., 1.],
    ])
    weighted_log_vars = tf.zeros_like(weighted_means)
    tf.debugging.assert_near(
        VariationalAutoencoder.compute_kl(
            weighted_means, weighted_log_vars
        ),
        tf.constant(0.5),
    )
    tf.debugging.assert_near(
        VariationalAutoencoder.compute_kl(
            weighted_means, weighted_log_vars,
            sample_weight=tf.constant([1., 0.]),
        ),
        tf.constant(0.),
    )
    tf.debugging.assert_near(
        VariationalAutoencoder.compute_kl(
            weighted_means, weighted_log_vars,
            sample_weight=tf.constant([0., 1.]),
        ),
        tf.constant(1.),
    )
    tf.debugging.assert_near(
        VariationalAutoencoder.compute_kl(
            weighted_means, weighted_log_vars,
            sample_weight=tf.zeros((2,)),
        ),
        tf.constant(0.),
    )
    latent_sites = [
        (weighted_means[:, :1], weighted_log_vars[:, :1]),
        (weighted_means[:, 1:], weighted_log_vars[:, 1:]),
    ]
    tf.debugging.assert_near(
        VariationalAutoencoder.compute_kl(latent_sites),
        VariationalAutoencoder.compute_kl(
            weighted_means, weighted_log_vars
        ),
    )
    tf.debugging.assert_near(
        VariationalAutoencoder.compute_kl(
            latent_sites,
            sample_weight=tf.constant([0., 1.]),
        ),
        tf.constant(1.),
    )
    assert VariationalAutoencoder.compute_kl(
        [], dtype=tf.float64
    ).dtype == tf.float64
    gradient_means = [
        tf.Variable([[0.5], [1.0]]),
        tf.Variable([[1.5], [2.0]]),
    ]
    with tf.GradientTape() as tape:
        gradient_kl = VariationalAutoencoder.compute_kl([
            (mean, tf.zeros_like(mean)) for mean in gradient_means
        ])
    gradients = tape.gradient(gradient_kl, gradient_means)
    assert all(
        gradient is not None
        and bool(tf.reduce_all(tf.math.is_finite(gradient)))
        for gradient in gradients
    )
    tf.random.set_seed(77)
    sampled_a = VariationalAutoencoder.compute_z(zero_latents, zero_latents)
    tf.random.set_seed(77)
    sampled_b = VariationalAutoencoder.compute_z(zero_latents, zero_latents)
    tf.debugging.assert_near(sampled_a, sampled_b)
    assert sampled_a.shape == zero_latents.shape
    assert bool(tf.reduce_any(tf.not_equal(sampled_a, zero_latents)))
    float64_sample = VariationalAutoencoder.compute_z(
        tf.zeros((1, 2), tf.float64),
        tf.zeros((1, 2), tf.float64)
    )
    assert float64_sample.dtype == tf.float64
    assert float64_sample.shape == (1, 2)

    metric_names = [metric.name for metric in unconditioned.metrics]
    assert metric_names == ["total_loss", "kl_loss", "recon_loss"]
    unconditioned.reset_metrics()
    assert all(float(metric.result()) == 0.0 for metric in unconditioned.metrics)
    weights_before_train = [
        weight.numpy().copy() 
        for weight in unconditioned.trainable_weights
    ]
    train_result = unconditioned.train_step(x)
    assert set(train_result) == {
        "loss", "kl_loss", "recon_loss", "recon_mae"
    }
    assert [metric.name for metric in unconditioned.metrics][-1] == "recon_mae"
    assert all(bool(tf.math.is_finite(value)) for value in train_result.values())
    assert any(
        not np.array_equal(before, after.numpy())
        for before, after in zip(weights_before_train, 
                                unconditioned.trainable_weights)
    )
    unconditioned.reset_metrics()
    assert all(float(metric.result()) == 0.0 for metric in unconditioned.metrics)
    weights_before_test = [
        weight.numpy().copy() 
        for weight in unconditioned.trainable_weights
    ]
    test_result = unconditioned.test_step(x)
    assert set(test_result) == set(train_result)
    assert all(bool(tf.math.is_finite(value)) for value in test_result.values())
    for before, after in zip(weights_before_test, 
                            unconditioned.trainable_weights):
        np.testing.assert_array_equal(before, after.numpy())

    paired_labels = tf.constant([0, 1], tf.int32)
    paired_train_result = unconditioned.train_step((x, paired_labels))
    paired_test_result = unconditioned.test_step((x, paired_labels))
    assert set(paired_train_result) == set(train_result)
    assert set(paired_test_result) == set(test_result)

    generated = unconditioned.generate(classes=[999], samples_per_class=3)
    assert isinstance(generated, np.ndarray)
    assert generated.shape == (3, 4) and generated.dtype == np.float32
    generated_zero = unconditioned.generate(samples_per_class=0)
    assert generated_zero.shape == (0, 4)
    normalized_count_samples = unconditioned.generate(samples_per_class=1.5)
    assert normalized_count_samples.shape == (1, 4)
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
    try:
        conditioned.generate(
            classes=[3],
            samples_per_class=1,
            onehot_y_output=True
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-range conditional class IDs must fail.")
    normalized_class_x, normalized_class_y = conditioned.generate(
        classes=[1.5], samples_per_class=1
    )
    assert normalized_class_x.shape == (1, 4)
    np.testing.assert_array_equal(normalized_class_y, np.array([1]))
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

    deterministic_seed = 19
    deterministic_indices = np.random.default_rng(deterministic_seed).integers(
        0, 2, size=5
    )
    conditioned.seen_classes = []
    with mock.patch.object(
        VariationalAutoencoder, "fit", autospec=True, return_value=fit_history
    ) as fit_mock, mock.patch.object(
        module, "get_callbacks", return_value=[sentinel_callback]
    ) as callbacks_mock:
        history = conditioned.train(
            x_numpy, 
            y_numpy, 
            train_num=5, 
            epochs=1, 
            batch_size=2, 
            clf=classifier, 
            seed=deterministic_seed,
            verbose=0, 
        )
        assert history == {"loss": [1.0]}
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

    lower_count_seed = 23
    lower_count_indices = np.random.default_rng(lower_count_seed).integers(
        0, 2, size=1
    )
    with mock.patch.object(
        VariationalAutoencoder, "fit", autospec=True, return_value=fit_history
    ) as fit_mock, mock.patch.object(module, "get_callbacks") as callbacks_mock:
        conditioned.train(
            x_numpy, 
            y_numpy, 
            train_num=1, 
            seed=lower_count_seed,
            callbacks_list=[explicit_callback], 
            clf=classifier, 
            verbose=0, 
        )
        callbacks_mock.assert_not_called()
        assert isinstance(fit_mock.call_args.args[1], tf.data.Dataset)
        fit_callbacks = fit_mock.call_args.kwargs["callbacks"]
        assert isinstance(fit_callbacks[0], DecoderAccuracyCallback)
        assert fit_callbacks[1] is explicit_callback

    try:
        unconditioned.train(x_numpy, y_numpy, train_num=-1, verbose=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Unconditional training must reject labels.")
    try:
        conditioned.train(x_numpy, None, train_num=-1, verbose=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Conditional training must require labels.")
    try:
        unconditioned.train(np.empty((0, 4), np.float32), train_num=0, verbose=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Resampling an empty input must fail.")

    with mock.patch.object(
        VariationalAutoencoder, "fit", autospec=True, return_value=fit_history
    ) as fit_mock:
        unconditioned.train(
            x_numpy,
            train_num=-1,
            batch_size=1,
            steps_per_epoch=3,
            callbacks_list=[],
            verbose=0,
        )
        repeated_dataset = fit_mock.call_args.args[1]
        assert len(list(repeated_dataset.take(3).as_numpy_iterator())) == 3
        assert fit_mock.call_args.kwargs["steps_per_epoch"] == 3

    with mock.patch.object(
        VariationalAutoencoder, "fit", autospec=True, return_value=fit_history
    ) as fit_mock:
        unconditioned.train(
            x_numpy,
            train_num=3,
            seed=31,
            callbacks_list=[],
            verbose=0,
        )
        unconditional_dataset = fit_mock.call_args.args[1]
        unconditional_rows = np.concatenate(
            list(unconditional_dataset.as_numpy_iterator()), axis=0
        )
        assert unconditional_rows.shape == (3, 4)

    tf.keras.backend.clear_session()
    return {"VariationalAutoencoder": "passed"}


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
