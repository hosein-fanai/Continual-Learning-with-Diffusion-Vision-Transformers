"""Joint conditional variational autoencoder and classifier model."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import metrics, losses

import numpy as np

from collections.abc import Callable
from numbers import Real

from autoencoder.variational_autoencoder import VariationalAutoencoder


class VAEClassifier(VariationalAutoencoder):
    """Train a conditional dense VAE alongside a classification objective.

    The forward pass and custom training/test steps classify reconstructed
    vectors.  Thus reconstruction, KL, and classification losses jointly train
    the VAE, while classification loss can also train a nested classifier when
    it is trainable. Labels must always be one-hot.

    Attributes:
        classifier (tf.keras.Model | Callable): Class-score model initialized
            from ``classifier``.
        alpha (float): Classification-loss multiplier initialized from
            ``alpha``.
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
            classifier (tf.keras.Model | Callable): Maps vectors shaped
                ``[batch, data_dim]`` to class probabilities shaped
                ``[batch, class_num]``. It is registered as a nested Keras
                component; its trainable weights participate in optimization.
            alpha (float): Nonnegative coefficient applied to mean categorical
                cross-entropy. ``0`` removes that term.
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
            ValueError: If ``alpha`` is non-finite or negative.
        """

        # Keep conditioning and class width controlled by this wrapper.
        if "conditioned" in kwargs or "class_num" in kwargs:
            raise TypeError("VAEClassifier fixes conditioned=True and class_num.")
        # Require a callable classifier for the reconstruction branch.
        if not callable(classifier):
            raise TypeError("classifier must be callable.")
        # Reject booleans and nonnumeric classification-loss weights.
        if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
            raise TypeError("alpha must be a real number.")
        # Keep the classification-loss coefficient finite and nonnegative.
        if not np.isfinite(alpha) or alpha < 0.:
            raise ValueError("alpha must be finite and nonnegative.")

        compile_model = kwargs.pop("compile", True)
        # Require an explicit boolean compilation switch.
        if not isinstance(compile_model, (bool, np.bool_)):
            raise TypeError("compile must be boolean.")
        compile_model = bool(compile_model)
        super().__init__(
            compile=False, 
            conditioned=True, 
            class_num=class_num, 
            **kwargs
        )

        self.classifier = classifier
        self.alpha = float(alpha)

        self.clf_loss_tracker = metrics.Mean(name="clf_loss")
        self.clf_accuracy_tracker = metrics.CategoricalAccuracy(
            name="clf_accuracy"
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
            tf.Tensor: Scalar ``float32`` fraction whose argmax class IDs match.
        """

        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        
        corrects = tf.cast(y_true == y_pred, dtype=tf.float32)
        accuracy = tf.reduce_mean(corrects)

        return accuracy

    @property
    def metrics(self: VAEClassifier) -> list[tf.keras.metrics.Metric]:
        """Expose all VAE and classifier trackers to the Keras loop.

        Returns:
            list[tf.keras.metrics.Metric]: Total, KL, reconstruction,
            classification-loss, and classification-accuracy trackers in that
            order, followed by configured reconstruction metrics. Keras resets
            them between epochs/evaluations.
        """

        compiled_metrics = self.compiled_metrics.metrics \
            if self.compiled_metrics is not None else []

        return [
            self.total_loss_tracker, 
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
        """Reconstruct conditional inputs and classify the reconstruction.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): ``(x, one_hot_y)`` with shapes
                ``[batch, data_dim]`` and ``[batch, class_num]``.
            training (bool | tf.Tensor): Keras training flag forwarded to the
                VAE and to a Keras-layer classifier.

        Returns:
            tuple[tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]:
            Latent statistics/sample shaped ``[batch, latent_dim]``,
            reconstruction shaped ``[batch, data_dim]``, and classifier
            probabilities shaped ``[batch, class_num]``.
        """

        (z_mean, z_log_var, z), reconstructed = super().call(inputs, training)
        # Forward training state to Keras classifiers.
        if isinstance(self.classifier, tf.keras.layers.Layer):
            prediction = self.classifier(reconstructed, training=training)
        # Support generic callables that do not accept a training argument.
        else:
            prediction = self.classifier(reconstructed)

        return (z_mean, z_log_var, z), reconstructed, prediction

    def train_step(
        self: VAEClassifier, 
        inputs: tuple[tf.Tensor, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """Optimize VAE and classifier losses for one conditional batch.

        Classification cross-entropy is computed from the classifier's output
        for reconstructed vectors and averaged across the batch. The
        reconstruction loss follows the compiled Keras reduction and KL is a
        batch mean.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Feature vectors and one-hot
                labels shaped ``[batch, data_dim]`` and
                ``[batch, class_num]``.

        Returns:
            dict[str, tf.Tensor]: Scalar running means under ``loss``,
            ``kl_loss``, ``recon_loss``, ``clf_loss``, and ``clf_accuracy``.
            Configured reconstruction metrics are included as additional keys.
        """

        x, y = inputs

        with tf.GradientTape() as tape:
            (z_mean, z_log_var, _), x_recon, y_pred = self(
                inputs, training=True
            )

            recon_loss = self.compiled_loss(
                x, 
                x_recon, 
                regularization_losses=self.losses,
            )
            kl_loss = VariationalAutoencoder.compute_kl(
                z_mean, 
                z_log_var
            )
            clf_loss = tf.reduce_mean(
                losses.categorical_crossentropy(y, y_pred),
            )

            total_loss = (
                recon_loss + 
                self.beta * kl_loss + 
                self.alpha * clf_loss
            )
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        batch_weight = tf.cast(tf.shape(x)[0], tf.float32)
        self.total_loss_tracker.update_state(total_loss, sample_weight=batch_weight)
        self.kl_loss_tracker.update_state(kl_loss, sample_weight=batch_weight)
        self.recon_loss_tracker.update_state(recon_loss, sample_weight=batch_weight)
        self.clf_loss_tracker.update_state(clf_loss, sample_weight=batch_weight)
        self.clf_accuracy_tracker.update_state(y, y_pred)
        self.compiled_metrics.update_state(x, x_recon)

        results = {
            "loss": self.total_loss_tracker.result(), 
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
        """Evaluate reconstruction-based VAE/classifier losses without updates.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Feature vectors and one-hot
                labels shaped ``[batch, data_dim]`` and
                ``[batch, class_num]``.

        Returns:
            dict[str, tf.Tensor]: Scalar running means under ``loss``,
            ``kl_loss``, ``recon_loss``, ``clf_loss``, and ``clf_accuracy``.
            Configured reconstruction metrics are included as additional keys.
        """

        x, y = inputs

        (z_mean, z_log_var, _), x_recon, y_pred = self(
            inputs, training=False
        )

        recon_loss = self.compiled_loss(
            x, 
            x_recon, 
            regularization_losses=self.losses,
        )
        kl_loss = VariationalAutoencoder.compute_kl(
            z_mean, 
            z_log_var
        )
        clf_loss = tf.reduce_mean(
            losses.categorical_crossentropy(y, y_pred)
        )

        total_loss = (
            recon_loss + 
            self.beta * kl_loss + 
            self.alpha * clf_loss
        )

        batch_weight = tf.cast(tf.shape(x)[0], tf.float32)
        self.total_loss_tracker.update_state(total_loss, sample_weight=batch_weight)
        self.kl_loss_tracker.update_state(kl_loss, sample_weight=batch_weight)
        self.recon_loss_tracker.update_state(recon_loss, sample_weight=batch_weight)
        self.clf_loss_tracker.update_state(clf_loss, sample_weight=batch_weight)
        self.clf_accuracy_tracker.update_state(y, y_pred)
        self.compiled_metrics.update_state(x, x_recon)

        results = {
            "loss": self.total_loss_tracker.result(), 
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
                ``validation_data``, ``callbacks_list``, and ``verbose``.
                ``x``, ``y``, ``clf``, and ``callbacks_monitor`` are forbidden
                because this method supplies them. Example:
                ``train(x, y, epochs=20, train_num=-1,
                validation_data=(x_val, y_val))``.

        Returns:
            dict[str, list[float]]: Keras epoch history.  Automatically created
            early stopping monitors ``"val_clf_accuracy"``; a decoder-accuracy
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

        monitor = "val_clf_accuracy" if kwargs.get("validation_data") is not None \
                else "clf_accuracy"

        return super().train(x, y, clf=self.classifier,
                            callbacks_monitor=monitor,
                            **kwargs)


def run_self_tests() -> dict[str, str]:
    """Run joint VAE/classifier construction, step, and API tests.

    The tests cover constructor guards, classifier registration, nonzero and
    zero classification coefficients, trainable and frozen classifiers,
    forward shapes, exact/partial accuracy, real eager train/test steps,
    metric reset, inherited conditional generation, weight persistence,
    invalid label/output shapes, callable classifiers, and every reserved or
    forwarded :meth:`train` keyword.

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

    for invalid_alpha in (True, "1", -0.1, float("nan"), float("inf")):
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
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("Invalid classification coefficients must fail.")

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
    assert "alpha" not in model.get_config(), (
        "The current subclassed-model config does not persist custom "
        "VAEClassifier constructor values."
    )

    x = tf.constant([
        [0.0, 0.25, 0.5, 0.75], [1.0, 0.75, 0.5, 0.25]
    ], dtype=tf.float32)
    y = tf.one_hot([0, 2], depth=3)
    (z_mean, z_log_var, z), reconstruction, prediction = model(
        (x, y), training=False
    )
    assert z_mean.shape == z_log_var.shape == z.shape == (2, 2)
    assert reconstruction.shape == (2, 4)
    assert prediction.shape == (2, 3)
    tf.debugging.assert_near(
        tf.reduce_sum(prediction, axis=1), 
        tf.ones((2,), tf.float32)
    )
    assert all(
        bool(tf.reduce_all(tf.math.is_finite(value)))
        for value in (z_mean, z_log_var, z, reconstruction, prediction)
    )

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
        "kl_loss", 
        "recon_loss", 
        "clf_loss", 
        "clf_accuracy", 
        "recon_mae",
    }
    assert all(bool(tf.math.is_finite(value)) for value in train_result.values())
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
    else:
        raise AssertionError("Classifier output width must match one-hot labels.")

    try:
        model.test_step((x, tf.constant([0, 2])))
    except (ValueError, tf.errors.InvalidArgumentError):
        pass
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
        else:
            raise AssertionError(
                f"Duplicate positional training key {duplicate_name!r} must fail."
            )

    tf.keras.backend.clear_session()
    return {"VAEClassifier": "passed"}


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
