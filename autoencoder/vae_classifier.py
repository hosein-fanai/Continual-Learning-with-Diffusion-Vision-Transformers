"""Joint conditional variational autoencoder and classifier model.

VAEClassifier jointly optimizes conditional reconstruction, Gaussian KL, and
classification from the original input. It extends the dense VAE generation and
training APIs while keeping the discriminative branch independent of the supplied
one-hot condition. Import registers the model with Keras serialization.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import metrics, losses

import numpy as np

from collections.abc import Callable

from common.gradients import apply_policy_gradients
from common.keras_registry import register_canonical_keras_serializable

from autoencoder.variational_autoencoder import VariationalAutoencoder


@register_canonical_keras_serializable(package="continual_learning")
class VAEClassifier(VariationalAutoencoder):
    """Train a conditional dense VAE alongside a classification objective.

    The conditional VAE reconstructs from ``(x, y)``, while the classifier
    predicts directly from ``x``.  Keeping the discriminative branch independent
    of the supplied target prevents ground-truth labels from leaking through a
    label-conditioned reconstruction. Reconstruction and KL losses train the
    VAE; classification loss trains a nested classifier when it is trainable.
    Labels must always be one-hot.

    Attributes:
        classifier (tf.keras.Model | Callable): Class-score model initialized
            from ``classifier``.
        alpha (float): Classification-loss multiplier initialized from
            ``alpha``.
            Defaults to ``1.0``.
        generative_loss_tracker (tf.keras.metrics.Mean): Running mean of the
            reconstruction plus beta-weighted KL objective.
        clf_loss_tracker (tf.keras.metrics.Mean): Running mean categorical
            loss, initialized with zero total/count.
        clf_accuracy_tracker (tf.keras.metrics.CategoricalAccuracy): Running
            sample-level categorical accuracy, initialized with zero
            total/count.
        latent_dim, beta, conditioned, class_num, encoder, decoder,
            seen_classes: Inherited VAE state.  ``conditioned`` is always true
            and ``class_num`` is supplied directly to this constructor.
    """

    def __init__(
        self: VAEClassifier, 
        class_num: int, 
        classifier: tf.keras.Model | Callable[..., tf.Tensor], 
        alpha: float = 1., 
        **kwargs: object
    ) -> None:
        """Build a conditional VAE, attach a classifier, and compile it.

        Args:
            class_num (int): Positive one-hot label width and classifier output
                width.
            classifier (tf.keras.Model | Callable): Maps input vectors shaped
                ``[batch, data_dim]`` to class probabilities shaped
                ``[batch, class_num]``. It is registered as a nested Keras
                component; its trainable weights participate in optimization.
            alpha (float): Finite, nonnegative coefficient applied to mean
                categorical cross-entropy.
                Defaults to ``1.0``.
            **kwargs (object): VAE options ``data_dim``, ``latent_dim``,
                ``hiddens_dims``, ``hiddens_kwargs``, ``last_activation``,
                ``beta``, and ``compile_args``, plus Keras model options such as
                ``name``, ``trainable``, and ``dtype``.  Do not provide
                ``conditioned`` or ``class_num``; this class fixes them.  The
                ``compile_args`` mapping accepts ``Model.compile`` keys and
                overrides ``{"optimizer": "adam", "loss":
                "mean_squared_error"}``, for example
                ``compile_args={"optimizer": Adam(1e-4), "run_eagerly": True}``.

        Returns:
            None.

        Raises:
            TypeError: If ``conditioned`` or ``class_num`` is included in
                ``kwargs``, or another unsupported key is supplied.
            ValueError: If ``alpha`` would invalidate the training objective.

        Notes:
            compile defaults to True; False skips compilation. All remaining VAE
            constructor defaults follow VariationalAutoencoder.__init__, including
            seed=None and beta=0.25. This wrapper replaces the default optimizer with
            Adam and fixes conditioning to True with the supplied class_num.
        """

        # Keep conditioning and class width controlled by this wrapper.
        if "conditioned" in kwargs or "class_num" in kwargs:
            raise TypeError("VAEClassifier fixes conditioned=True and class_num.")
        alpha = float(alpha)
        # Reject classification weights that would make the joint objective invalid.
        if not np.isfinite(alpha) or alpha < 0.:
            raise ValueError("alpha must be finite and nonnegative.")
        compile_model = kwargs.pop("compile", True)
        compile_model = bool(compile_model)
        super().__init__(
            compile=False, 
            conditioned=True, 
            class_num=class_num, 
            **kwargs
        )

        self.classifier = classifier
        self.alpha = alpha

        stable_dtype = self.dtype_policy.variable_dtype
        self.generative_loss_tracker = metrics.Mean(
            name="generative_loss",
            dtype=stable_dtype,
        )
        self.clf_loss_tracker = metrics.Mean(
            name="clf_loss",
            dtype=stable_dtype,
        )
        self.clf_accuracy_tracker = metrics.CategoricalAccuracy(
            name="clf_accuracy",
            dtype=stable_dtype,
        )

        compile_args_default = {
            "optimizer": "adam", 
            "loss": "mean_squared_error"
        }
        compile_args = {
            **compile_args_default, 
            **(kwargs.get("compile_args") or {})
        }

        # Compile immediately when requested by the caller.
        if compile_model:
            self.compile(**compile_args)

    def get_config(self: VAEClassifier) -> dict[str, object]:
        """Return the VAE architecture plus classifier branch configuration.

        Returns:
            dict[str, object]: JSON-compatible constructor configuration. The
            nested classifier is represented through Keras object
            serialization and compilation remains separate from architecture.
        """

        config = super().get_config()
        # This subclass fixes conditioning internally; forwarding the base
        # keyword would be rejected by its constructor.
        config.pop("conditioned", None)
        config.update({
            "class_num": self.class_num,
            "classifier": tf.keras.utils.serialize_keras_object(
                self.classifier
            ),
            "alpha": self.alpha,
            "compile": False,
            "compile_args": None,
        })

        return config

    @classmethod
    def from_config(
        cls: type[VAEClassifier],
        config: dict[str, object],
    ) -> VAEClassifier:
        """Recreate the joint model and deserialize its classifier branch.

        Args:
            config (dict[str, object]): Output of :meth:`get_config`.

        Returns:
            VAEClassifier: Independent uncompiled architecture clone.
        """

        restored = VariationalAutoencoder._deserialize_constructor_config(
            config
        )
        classifier_config = restored.pop("classifier")
        # Keras 2.10's generic deserializer does not expose the built-in
        # Sequential/Functional model classes through its default object map.
        # Route model configs through the model-aware factory while retaining
        # the generic path for registered callable classifiers.
        if (
            isinstance(classifier_config, dict)
            and classifier_config.get("class_name")
            in {"Functional", "Model", "Sequential"}
        ):
            classifier = tf.keras.models.model_from_config(classifier_config)
        # Preserve support for registered callable classifier objects.
        else:
            classifier = tf.keras.utils.deserialize_keras_object(
                classifier_config
            )

        return cls(classifier=classifier, **restored)

    def _compute_accuracy(
        self: VAEClassifier, 
        y_true: tf.Tensor, 
        y_pred: tf.Tensor
    ) -> tf.Tensor:
        """Compute unweighted categorical accuracy for one batch.

        Args:
            y_true (tf.Tensor): One-hot targets shaped ``[batch, class_num]``.
            y_pred (tf.Tensor): Matching class probabilities.

        Returns:
            tf.Tensor: Scalar fraction in the policy's stable variable dtype.
        """

        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        
        corrects = tf.cast(
            y_true == y_pred,
            dtype=tf.as_dtype(self.dtype_policy.variable_dtype),
        )
        accuracy = tf.reduce_mean(corrects)

        return accuracy

    @property
    def metrics(self: VAEClassifier) -> list[tf.keras.metrics.Metric]:
        """Expose all VAE and classifier trackers to the Keras loop.

        Returns:
            list[tf.keras.metrics.Metric]: Total, generative, KL,
            reconstruction, classification-loss, and classification-accuracy
            trackers in that order, followed by configured reconstruction
            metrics. Keras resets them between epochs/evaluations.
        """

        # Include compiled metrics once their container exists; otherwise expose only local
        # trackers.
        compiled_metrics = self.compiled_metrics.metrics \
            if self.compiled_metrics is not None else []

        return [
            self.total_loss_tracker,
            self.generative_loss_tracker,
            self.kl_loss_tracker, 
            self.recon_loss_tracker, 
            self.clf_loss_tracker, 
            self.clf_accuracy_tracker,
            *compiled_metrics
        ]

    def call(
        self: VAEClassifier, 
        inputs: tuple[tf.Tensor, tf.Tensor], 
        training: bool | tf.Tensor = False, 
    ) -> tuple[tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]:
        """Reconstruct conditionally while classifying the unconditioned input.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): ``(x, one_hot_y)`` with shapes
                ``[batch, data_dim]`` and ``[batch, class_num]``.
            training (bool | tf.Tensor): Keras training flag forwarded to the
                VAE and to a Keras-layer classifier.
                Defaults to ``False``.

        Returns:
            tuple[tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]:
            Latent statistics/sample shaped ``[batch, latent_dim]``,
            reconstruction shaped ``[batch, data_dim]``, and label-independent
            classifier probabilities from ``x`` shaped ``[batch, class_num]``.
        """

        x, _ = inputs
        (z_mean, z_log_var, z), reconstructed = super().call(inputs, training)
        # Classify x directly: reconstructed is conditioned on the true label
        # and therefore must never be used as the discriminative input.
        if isinstance(self.classifier, tf.keras.layers.Layer):
            prediction = self.classifier(x, training=training)
        # Support generic callables that do not accept a training argument.
        else:
            prediction = self.classifier(x)

        return (z_mean, z_log_var, z), reconstructed, prediction

    def train_step(
        self: VAEClassifier, 
        inputs: tuple[tf.Tensor, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """Optimize VAE and classifier losses for one conditional batch.

        Classification cross-entropy is computed from direct predictions on
        ``x`` and averaged across the batch. The reconstruction loss follows
        the compiled Keras reduction and KL is a batch mean. Consequently, the
        classification term cannot exploit the label supplied to the
        conditional encoder/decoder.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Feature vectors and one-hot
                labels shaped ``[batch, data_dim]`` and
                ``[batch, class_num]``.

        Returns:
            dict[str, tf.Tensor]: Scalar running means under ``loss``,
            ``generative_loss``, ``kl_loss``, ``recon_loss``, ``clf_loss``, and
            ``clf_accuracy``. Configured reconstruction metrics are included as
            additional keys.

        Notes:
            A third sample_weight component is accepted and broadcast over batch rows.
            Classification and KL use weight-normalized means (zero for all-zero
            weights); reconstruction follows the compiled Keras reduction. The total
            is reconstruction + beta * KL + alpha * classification. Running loss
            trackers weight batch objectives by batch size; accuracy and reconstruction
            metrics receive row weights. test_step updates metrics and samples the
            latent without applying gradients; train_step also updates trainable model
            weights and training-mode normalization statistics.
        """

        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(inputs)
        model_inputs = (x, y)

        with tf.GradientTape() as tape:
            (z_mean, z_log_var, _), x_recon, y_pred = self(
                model_inputs, training=True
            )

            stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
            # Use unweighted rows when no weights are supplied; otherwise broadcast weights
            # across the batch.
            row_sample_weight = None if sample_weight is None else tf.broadcast_to(
                tf.reshape(tf.cast(sample_weight, stable_dtype), (-1,)),
                tf.shape(x)[:1],
            )
            recon_loss = tf.cast(self.compiled_loss(
                x, 
                x_recon, 
                sample_weight=row_sample_weight,
                regularization_losses=self.losses,
            ), stable_dtype)
            kl_loss = VariationalAutoencoder.compute_kl(
                z_mean, 
                z_log_var,
                sample_weight=row_sample_weight,
                dtype=stable_dtype,
            )
            clf_rows = losses.categorical_crossentropy(
                tf.cast(y, stable_dtype),
                tf.cast(y_pred, stable_dtype),
            )
            # Average unweighted class losses; otherwise normalize their weighted sum by
            # total weight.
            clf_loss = tf.reduce_mean(clf_rows) if row_sample_weight is None \
                else tf.math.divide_no_nan(
                    tf.reduce_sum(clf_rows * row_sample_weight),
                    tf.reduce_sum(row_sample_weight),
                )

            generative_loss = (
                recon_loss + tf.cast(self.beta, stable_dtype) * kl_loss
            )
            total_loss = generative_loss + (
                tf.cast(self.alpha, stable_dtype) * clf_loss
            )
        
        apply_policy_gradients(
            tape,
            self.optimizer,
            total_loss,
            self.trainable_weights,
        )

        batch_weight = tf.cast(tf.shape(x)[0], stable_dtype)
        self.total_loss_tracker.update_state(total_loss, sample_weight=batch_weight)
        self.generative_loss_tracker.update_state(
            generative_loss,
            sample_weight=batch_weight,
        )
        self.kl_loss_tracker.update_state(kl_loss, sample_weight=batch_weight)
        self.recon_loss_tracker.update_state(recon_loss, sample_weight=batch_weight)
        self.clf_loss_tracker.update_state(clf_loss, sample_weight=batch_weight)
        self.clf_accuracy_tracker.update_state(
            y, y_pred, sample_weight=row_sample_weight
        )
        self.compiled_metrics.update_state(
            x, x_recon, sample_weight=row_sample_weight
        )

        results = {
            "loss": self.total_loss_tracker.result(),
            "generative_loss": self.generative_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(), 
            "recon_loss": self.recon_loss_tracker.result(), 
            "clf_loss": self.clf_loss_tracker.result(), 
            "clf_accuracy": self.clf_accuracy_tracker.result()
        }
        results.update({
            metric.name: metric.result()
            for metric in self.compiled_metrics.metrics
        })

        return results

    def test_step(
        self: VAEClassifier, 
        inputs: tuple[tf.Tensor, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """Evaluate conditional reconstruction and direct classification.

        Class predictions depend only on ``x``; ``y`` is used as a target and
        as the VAE reconstruction condition, never as classifier input.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Feature vectors and one-hot
                labels shaped ``[batch, data_dim]`` and
                ``[batch, class_num]``.

        Returns:
            dict[str, tf.Tensor]: Scalar running means under ``loss``,
            ``generative_loss``, ``kl_loss``, ``recon_loss``, ``clf_loss``, and
            ``clf_accuracy``. Configured reconstruction metrics are included as
            additional keys.

        Notes:
            A third sample_weight component is accepted and broadcast over batch rows.
            Classification and KL use weight-normalized means (zero for all-zero
            weights); reconstruction follows the compiled Keras reduction. The total
            is reconstruction + beta * KL + alpha * classification. Running loss
            trackers weight batch objectives by batch size; accuracy and reconstruction
            metrics receive row weights. test_step updates metrics and samples the
            latent without applying gradients; train_step also updates trainable model
            weights and training-mode normalization statistics.
        """

        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(inputs)
        model_inputs = (x, y)

        (z_mean, z_log_var, _), x_recon, y_pred = self(
            model_inputs, training=False
        )

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        # Use unweighted rows when no weights are supplied; otherwise broadcast weights
        # across the batch.
        row_sample_weight = None if sample_weight is None else tf.broadcast_to(
            tf.reshape(tf.cast(sample_weight, stable_dtype), (-1,)),
            tf.shape(x)[:1],
        )
        recon_loss = tf.cast(self.compiled_loss(
            x, 
            x_recon, 
            sample_weight=row_sample_weight,
            regularization_losses=self.losses,
        ), stable_dtype)
        kl_loss = VariationalAutoencoder.compute_kl(
            z_mean, 
            z_log_var,
            sample_weight=row_sample_weight,
            dtype=stable_dtype,
        )
        clf_rows = losses.categorical_crossentropy(
            tf.cast(y, stable_dtype),
            tf.cast(y_pred, stable_dtype),
        )
        # Average unweighted class losses; otherwise normalize their weighted sum by total
        # weight.
        clf_loss = tf.reduce_mean(clf_rows) if row_sample_weight is None \
            else tf.math.divide_no_nan(
                tf.reduce_sum(clf_rows * row_sample_weight),
                tf.reduce_sum(row_sample_weight),
            )

        generative_loss = (
            recon_loss + tf.cast(self.beta, stable_dtype) * kl_loss
        )
        total_loss = generative_loss + (
            tf.cast(self.alpha, stable_dtype) * clf_loss
        )

        batch_weight = tf.cast(tf.shape(x)[0], stable_dtype)
        self.total_loss_tracker.update_state(total_loss, sample_weight=batch_weight)
        self.generative_loss_tracker.update_state(
            generative_loss,
            sample_weight=batch_weight,
        )
        self.kl_loss_tracker.update_state(kl_loss, sample_weight=batch_weight)
        self.recon_loss_tracker.update_state(recon_loss, sample_weight=batch_weight)
        self.clf_loss_tracker.update_state(clf_loss, sample_weight=batch_weight)
        self.clf_accuracy_tracker.update_state(
            y, y_pred, sample_weight=row_sample_weight
        )
        self.compiled_metrics.update_state(
            x, x_recon, sample_weight=row_sample_weight
        )

        results = {
            "loss": self.total_loss_tracker.result(),
            "generative_loss": self.generative_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(), 
            "recon_loss": self.recon_loss_tracker.result(), 
            "clf_loss": self.clf_loss_tracker.result(), 
            "clf_accuracy": self.clf_accuracy_tracker.result()
        }
        results.update({
            metric.name: metric.result()
            for metric in self.compiled_metrics.metrics
        })

        return results

    def train(
        self: VAEClassifier, 
        x: tf.Tensor | np.ndarray, 
        y: tf.Tensor | np.ndarray, 
        **kwargs: object
    ) -> dict[str, list[float]]:
        """Fit with decoder-accuracy monitoring from the attached classifier.

        Args:
            x (numpy.ndarray | tf.Tensor): Samples shaped
                ``[samples, data_dim]``.
            y (numpy.ndarray | tf.Tensor): One-hot labels shaped
                ``[samples, class_num]``.
            **kwargs (object): Options accepted by
                :meth:`VariationalAutoencoder.train`: ``train_num``, ``epochs``,
                ``batch_size``, ``shuffle_buffer``, ``seed``,
                ``steps_per_epoch``,
                ``validation_data``, ``callbacks_list``, and ``verbose``.
                ``x``, ``y``, ``clf``, and ``callbacks_monitor`` are forbidden
                because this method supplies them. Example:
                ``train(x, y, epochs=20, train_num=-1,
                validation_data=(x_val, y_val))``.

        Returns:
            dict[str, list[float]]: Keras epoch history.  Automatically created
            early stopping monitors ``"val_clf_accuracy"`` when validation is supplied,
            otherwise ``"clf_accuracy"``; a decoder-accuracy
            callback is always prepended unless callback construction fails.

        Raises:
            TypeError: If a reserved or unsupported training keyword is
                supplied.
        """

        reserved = {"x", "y", "clf", "callbacks_monitor"}.intersection(kwargs)
        # Prevent forwarded fit options from overriding managed train arguments.
        if reserved:
            raise TypeError(
                "Reserved VAEClassifier.train options: " + str(sorted(reserved))
            )

        # Monitor validation classification accuracy when validation exists, or training
        # accuracy otherwise.
        monitor = "val_clf_accuracy" if kwargs.get("validation_data") is not None \
                else "clf_accuracy"

        return super().train(x, y, clf=self.classifier,
                            callbacks_monitor=monitor,
                            **kwargs)


# TensorFlow 2.10 may emit the plain root name for subclassed-model JSON.
tf.keras.utils.get_custom_objects()["VAEClassifier"] = VAEClassifier


def run_self_tests() -> dict[str, str]:
    """Run joint VAE/classifier construction, step, and API tests.

    The tests cover constructor guards, classifier registration, nonzero and
    zero classification coefficients, trainable and frozen classifiers,
    forward shapes, label-leakage isolation, exact/partial accuracy, real eager
    train/test steps, metric reset, inherited conditional generation, weight
    persistence, invalid label/output shapes, callable classifiers, and every
    reserved or forwarded :meth:`train` keyword.

    Args:
        None.

    Returns:
        dict[str, str]: ``{"VAEClassifier": "passed"}`` after every assertion
        succeeds.
    """

    from pathlib import Path
    from tempfile import TemporaryDirectory
    from unittest import mock


    tf.keras.backend.clear_session()
    tf.random.set_seed(303)
    classifier = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)), 
        tf.keras.layers.Dense(
            3, 
            activation="softmax", 
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1), 
        ), 
    ], name="self_test_classifier")

    try:
        VAEClassifier(
            3, 
            classifier, 
            conditioned=True, 
            data_dim=4, 
            latent_dim=2, 
            hiddens_dims=(), 
        )
    except TypeError:
        pass
    # This invalid case should already have raised: VAEClassifier must reject a conditioned
    # override.
    else:
        raise AssertionError("VAEClassifier must reject a conditioned override.")

    compile_disabled = VAEClassifier(
        3,
        classifier,
        compile=False,
        data_dim=4,
        latent_dim=2,
        hiddens_dims=(),
    )
    assert compile_disabled._is_compiled is False
    compile_args_none = VAEClassifier(
        3,
        classifier,
        compile=False,
        compile_args=None,
        data_dim=4,
        latent_dim=2,
        hiddens_dims=(),
    )
    assert compile_args_none._is_compiled is False

    for invalid_alpha in (-0.1, float("nan"), float("inf")):
        try:
            VAEClassifier(
                3,
                classifier,
                alpha=invalid_alpha,
                compile=False,
                data_dim=4,
                latent_dim=2,
                hiddens_dims=(),
            )
        except ValueError:
            pass
        # This invalid case should already have raised: Invalid classification loss weights
        # must fail.
        else:
            raise AssertionError("Invalid classification loss weights must fail.")

    try:
        VAEClassifier(
            3,
            classifier,
            data_dim=4,
            latent_dim=2,
            hiddens_dims=(),
            unsupported_option=True,
        )
    except TypeError:
        pass
    # This invalid case should already have raised: Unknown Keras model options must be
    # rejected.
    else:
        raise AssertionError("Unknown Keras model options must be rejected.")

    model = VAEClassifier(
        class_num=3, 
        classifier=classifier, 
        alpha=0.5, 
        data_dim=4, 
        latent_dim=2, 
        hiddens_dims=(4,), 
        hiddens_kwargs={"actv": "relu", "use_batch_norm": False}, 
        last_activation=None, 
        beta=0.25, 
        compile_args={
            "optimizer": tf.keras.optimizers.SGD(learning_rate=0.01), 
            "loss": "mean_squared_error", 
            "metrics": [
                tf.keras.metrics.MeanAbsoluteError(name="recon_mae")
            ],
            "run_eagerly": True, 
        }, 
        name="vae_classifier", 
    )
    assert model.conditioned is True and model.class_num == 3
    assert model.classifier is classifier and model.alpha == 0.5
    assert model.name == "vae_classifier" and model._is_compiled is True
    assert isinstance(model.optimizer, tf.keras.optimizers.SGD)
    assert model.run_eagerly is True
    joint_config = model.get_config()
    assert joint_config["class_num"] == 3
    assert joint_config["alpha"] == 0.5
    assert joint_config["data_dim"] == 4
    assert joint_config["latent_dim"] == 2
    assert joint_config["hiddens_dims"] == [4]
    assert "conditioned" not in joint_config
    assert isinstance(joint_config["classifier"], dict)
    joint_clone = VAEClassifier.from_config(joint_config)
    assert joint_clone.class_num == model.class_num
    assert joint_clone.alpha == model.alpha
    assert joint_clone.data_dim == model.data_dim
    assert joint_clone.latent_dim == model.latent_dim
    assert joint_clone.hiddens_dims == model.hiddens_dims
    assert joint_clone._is_compiled is False
    assert isinstance(joint_clone.classifier, tf.keras.Model)
    assert joint_clone.classifier is not model.classifier
    assert len(joint_clone.classifier.weights) == len(model.classifier.weights)
    assert all(
        left.shape == right.shape
        for left, right in zip(
            joint_clone.classifier.weights,
            model.classifier.weights,
        )
    )

    x = tf.constant([
        [0.0, 0.25, 0.5, 0.75], [1.0, 0.75, 0.5, 0.25]
    ], dtype=tf.float32)
    y = tf.one_hot([0, 2], depth=3)
    (z_mean, z_log_var, z), reconstruction, prediction = model(
        (x, y), training=False
    )
    clone_latents, clone_reconstruction, clone_prediction = joint_clone(
        (x, y), training=False
    )
    assert z_mean.shape == z_log_var.shape == z.shape == (2, 2)
    assert reconstruction.shape == (2, 4)
    assert prediction.shape == (2, 3)
    assert all(value.shape == (2, 2) for value in clone_latents)
    assert clone_reconstruction.shape == (2, 4)
    assert clone_prediction.shape == (2, 3)
    tf.debugging.assert_near(
        tf.reduce_sum(prediction, axis=1), 
        tf.ones((2,), tf.float32)
    )
    assert all(
        bool(tf.reduce_all(tf.math.is_finite(value)))
        for value in (z_mean, z_log_var, z, reconstruction, prediction)
    )

    # Regression: changing only the reconstruction condition may alter the VAE
    # output, but it must not alter predictions for an otherwise identical x.
    protocol_classifier = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(3, activation="softmax", use_bias=False),
    ])
    protocol_model = VAEClassifier(
        class_num=3,
        classifier=protocol_classifier,
        data_dim=4,
        latent_dim=1,
        hiddens_dims=(),
        last_activation=None,
        compile=False,
    )
    protocol_classifier.layers[-1].set_weights([np.array([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)])
    protocol_model.decoder.layers[-1].set_weights([
        np.array([
            [0.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
        ], dtype=np.float32),
        np.zeros((4,), dtype=np.float32),
    ])
    same_x = tf.repeat(x[:1], repeats=2, axis=0)
    labels_a = tf.one_hot([0, 0], depth=3)
    labels_b = tf.one_hot([1, 1], depth=3)
    _, reconstruction_a, prediction_a = protocol_model(
        (same_x, labels_a), training=False
    )
    _, reconstruction_b, prediction_b = protocol_model(
        (same_x, labels_b), training=False
    )
    tf.debugging.assert_near(prediction_a, prediction_b)
    tf.debugging.assert_near(
        prediction_a,
        protocol_classifier(same_x, training=False),
    )
    assert bool(tf.reduce_any(tf.not_equal(
        reconstruction_a,
        reconstruction_b,
    ))), "Conditional reconstruction should remain label-sensitive."

    tf.debugging.assert_near(
        model._compute_accuracy(
            tf.one_hot([0, 1], 3), 
            tf.one_hot([0, 1], 3)
        ), 
        tf.constant(1.0), 
    )
    tf.debugging.assert_near(
        model._compute_accuracy(
            tf.one_hot([0, 1], 3), 
            tf.one_hot([0, 2], 3)
        ), 
        tf.constant(0.5), 
    )
    tf.debugging.assert_near(
        model._compute_accuracy(
            tf.one_hot([0, 1], 3), 
            tf.one_hot([2, 2], 3)
        ),
        tf.constant(0.0), 
    )

    metric_names = [metric.name for metric in model.metrics]
    assert metric_names == [
        "total_loss",
        "generative_loss",
        "kl_loss", 
        "recon_loss", 
        "clf_loss", 
        "clf_accuracy", 
    ]
    model.reset_metrics()
    assert all(float(metric.result()) == 0.0 for metric in model.metrics)

    weights_before_train = [
        weight.numpy().copy() 
        for weight in model.trainable_weights
    ]
    train_result = model.train_step((x, y))
    assert set(train_result) == {
        "loss",
        "generative_loss",
        "kl_loss", 
        "recon_loss", 
        "clf_loss", 
        "clf_accuracy", 
        "recon_mae",
    }
    assert all(bool(tf.math.is_finite(value)) for value in train_result.values())
    tf.debugging.assert_near(
        train_result["generative_loss"],
        train_result["recon_loss"] + model.beta * train_result["kl_loss"],
    )
    tf.debugging.assert_near(
        train_result["loss"],
        train_result["generative_loss"] + model.alpha * train_result["clf_loss"],
    )
    assert any(
        not np.array_equal(before, after.numpy())
        for before, after in zip(weights_before_train, 
                            model.trainable_weights)
    )
    model.reset_metrics()
    assert all(float(metric.result()) == 0.0 for metric in model.metrics)

    weights_before_test = [
        weight.numpy().copy() 
        for weight in model.trainable_weights
    ]
    test_result = model.test_step((x, y))
    assert set(test_result) == set(train_result)
    assert all(bool(tf.math.is_finite(value)) for value in test_result.values())
    tf.debugging.assert_near(
        test_result["generative_loss"],
        test_result["recon_loss"] + model.beta * test_result["kl_loss"],
    )
    tf.debugging.assert_near(
        test_result["loss"],
        test_result["generative_loss"] + model.alpha * test_result["clf_loss"],
    )
    for before, after in zip(weights_before_test, model.trainable_weights):
        np.testing.assert_array_equal(before, after.numpy())

    model.seen_classes = [0, 2]
    generated_x, generated_y = model.generate(
        samples_per_class=1, 
        onehot_y_output=False
    )
    assert generated_x.shape == (2, 4)
    np.testing.assert_array_equal(
        generated_y, 
        np.array([0, 2])
    )

    frozen_classifier = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)), 
        tf.keras.layers.Dense(3, activation="softmax"), 
    ])
    frozen_classifier.trainable = False
    zero_alpha_model = VAEClassifier(
        3, 
        frozen_classifier, 
        alpha=0.0, 
        data_dim=4, 
        latent_dim=1, 
        hiddens_dims=(), 
        beta=0.0, 
        compile_args={
            "optimizer": tf.keras.optimizers.SGD(0.01), 
            "loss": "mean_squared_error", 
            "run_eagerly": True, 
        },
    )
    zero_alpha_result = zero_alpha_model.train_step((x, y))
    assert all(bool(tf.math.is_finite(value)) for value in zero_alpha_result.values())
    assert zero_alpha_model.alpha == 0.0
    assert frozen_classifier.trainable is False


    def callable_classifier(inputs: tf.Tensor) -> tf.Tensor:
        """Return deterministic uniform three-class probabilities.

        Args:
            inputs (tf.Tensor): Feature batch shaped ``[batch, 4]``.

        Returns:
            tf.Tensor: Uniform probabilities shaped ``[batch, 3]``.
        """

        return tf.fill(
            (tf.shape(inputs)[0], 3), 
            tf.constant(1.0 / 3.0, tf.float32)
        )


    callable_model = VAEClassifier(
        3, 
        callable_classifier, 
        alpha=0.25,
        data_dim=4, 
        latent_dim=1, 
        hiddens_dims=(), 
        compile_args={
            "optimizer": "sgd", 
            "loss": "mean_squared_error"
        }, 
    )
    assert callable_model.alpha == 0.25
    assert callable_model((x, y), training=False)[2].shape == (2, 3)
    callable_result = callable_model.test_step((x, y))
    assert all(bool(tf.math.is_finite(value)) for value in callable_result.values())

    bad_classifier = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)), 
        tf.keras.layers.Dense(2, activation="softmax"), 
    ])
    bad_model = VAEClassifier(
        3, 
        bad_classifier, 
        data_dim=4, 
        latent_dim=1, 
        hiddens_dims=(), 
        compile_args={
            "optimizer": "sgd", 
            "loss": "mean_squared_error"
        },
    )
    try:
        bad_model.test_step((x, y))
    except (ValueError, tf.errors.InvalidArgumentError):
        pass
    # This invalid case should already have raised: Classifier output width must match
    # one-hot labels.
    else:
        raise AssertionError("Classifier output width must match one-hot labels.")

    try:
        model.test_step((x, tf.constant([0, 2])))
    except (ValueError, tf.errors.InvalidArgumentError):
        pass
    # This invalid case should already have raised: VAEClassifier train/test labels must be
    # one-hot.
    else:
        raise AssertionError("VAEClassifier train/test labels must be one-hot.")

    with TemporaryDirectory() as temp_dir:
        weights_path = Path(temp_dir) / "vae_classifier.weights.h5"
        model.save_weights(weights_path)
        clone_classifier = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(4,)), 
            tf.keras.layers.Dense(3, activation="softmax"), 
        ])
        clone = VAEClassifier(
            3, 
            clone_classifier, 
            alpha=0.5, 
            data_dim=4, 
            latent_dim=2, 
            hiddens_dims=(4,), 
            hiddens_kwargs={"actv": "relu", "use_batch_norm": False}, 
            last_activation=None, 
            beta=0.25, 
            compile_args={
                "optimizer": "sgd", 
                "loss": "mean_squared_error"
            },
        )
        clone((x, y), training=False)
        clone.load_weights(weights_path)
        source_mean, source_log_var, _ = model.encoder((x, y), training=False)
        clone_mean, clone_log_var, _ = clone.encoder((x, y), training=False)
        tf.debugging.assert_near(source_mean, clone_mean)
        tf.debugging.assert_near(source_log_var, clone_log_var)
        tf.debugging.assert_near(classifier(x), clone_classifier(x))

    delegated_history = {"loss": [0.75]}
    with mock.patch.object(
        VariationalAutoencoder, 
        "train", 
        autospec=True, 
        return_value=delegated_history, 
    ) as base_train:
        returned_history = model.train(
            x.numpy(), 
            y.numpy(), 
            epochs=1, 
            batch_size=2, 
            train_num=-1, 
            callbacks_list=[], 
            verbose=0, 
        )
        assert returned_history is delegated_history
        call_args, call_kwargs = base_train.call_args
        assert call_args[0] is model
        np.testing.assert_array_equal(call_args[1], x.numpy())
        np.testing.assert_array_equal(call_args[2], y.numpy())
        assert call_kwargs["clf"] is classifier
        assert call_kwargs["callbacks_monitor"] == "clf_accuracy"
        assert call_kwargs["epochs"] == 1
        assert call_kwargs["callbacks_list"] == []

    for reserved_name, reserved_value in (
        ("clf", classifier),
        ("callbacks_monitor", "loss"),
    ):
        try:
            model.train(
                x.numpy(), 
                y.numpy(), 
                **{reserved_name: reserved_value}
            )
        except TypeError:
            pass
        # This invalid case should already have raised: Reserved training key must fail.
        else:
            raise AssertionError(
                f"Reserved training key {reserved_name!r} must fail."
            )

    for duplicate_name, duplicate_value in (
        ("x", x.numpy()), 
        ("y", y.numpy())
    ):
        try:
            model.train(
                x.numpy(), y.numpy(), 
                **{duplicate_name: duplicate_value}
            )
        except TypeError:
            pass
        # This invalid case should already have raised: Duplicate positional training key
        # must fail.
        else:
            raise AssertionError(
                f"Duplicate positional training key {duplicate_name!r} must fail."
            )

    tf.keras.backend.clear_session()
    return {"VAEClassifier": "passed"}


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
