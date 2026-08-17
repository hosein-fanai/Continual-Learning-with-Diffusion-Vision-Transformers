"""Class-conditioned VAE jointly carrying a downstream classifier."""

import tensorflow as tf
from tensorflow.keras import metrics, losses

from autoencoder.variational_autoencoder import VariationalAutoencoder


class ClassifierVAE(VariationalAutoencoder):
    """Train a conditional dense VAE alongside a classification objective.

    The forward pass classifies reconstructed vectors, while the custom
    training and test steps calculate categorical loss/accuracy from
    ``classifier(x)`` on the original vectors.  Thus reconstruction and KL
    losses train the VAE; classification loss can train the nested classifier
    when it is trainable.  Labels must always be one-hot.

    Attributes:
        classifier (tf.keras.Model | Callable): Class-score model initialized
            from ``classifier``.
        alpha (float): Classification-loss multiplier initialized from
            ``alpha``.
        clf_loss_tracker (tf.keras.metrics.Mean): Running summed categorical
            loss, initialized with zero total/count.
        clf_accuracy_tracker (tf.keras.metrics.Mean): Running batch-accuracy
            mean, initialized with zero total/count.
        latent_dim, beta, conditioned, class_num, encoder, decoder,
            seen_classes: Inherited VAE state.  ``conditioned`` is always true
            and ``class_num`` is supplied directly to this constructor.
    """

    def __init__(
        self, 
        class_num, 
        classifier, 
        alpha=1., 
        **kwargs
    ):
        """Build a conditional VAE, attach a classifier, and compile it.

        Args:
            class_num (int): Positive one-hot label width and classifier output
                width.
            classifier (tf.keras.Model | Callable): Maps vectors shaped
                ``[batch, data_dim]`` to class probabilities or logits shaped
                ``[batch, class_num]``.  It is registered as a nested Keras
                component; its trainable weights participate in optimization.
            alpha (float): Scalar coefficient applied to the batch-summed
                categorical cross-entropy.  ``0`` removes that term.
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
            AssertionError: If ``conditioned`` or ``class_num`` is included in
                ``kwargs``.
            TypeError: If ``compile`` is included in ``kwargs`` (the base call
                already supplies ``compile=False``) or another unsupported key
                is supplied.
        """
        assert "conditioned" not in kwargs
        assert "class_num" not in kwargs


        super().__init__(compile=False, conditioned=True, class_num=class_num, **kwargs)

        self.classifier = classifier
        self.alpha = alpha

        self.clf_loss_tracker = metrics.Mean(name="clf_loss")
        self.clf_accuracy_tracker = metrics.Mean(name="clf_accuracy")

        compile_args_default = {
            "optimizer": "adam",
            "loss": "mean_squared_error",
        }
        compile_args = {**compile_args_default, **kwargs.get("compile_args", {})}

        if kwargs.get("compile", True):
            self.compile(**compile_args)

    def _compute_accuracy(self, y_true, y_pred):
        """Compute unweighted categorical accuracy for one batch.

        Args:
            y_true (tf.Tensor): One-hot targets shaped ``[batch, class_num]``.
            y_pred (tf.Tensor): Matching scores or probabilities.

        Returns:
            tf.Tensor: Scalar ``float32`` fraction whose argmax class IDs match.
        """
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        
        corrects = tf.cast(y_true == y_pred, dtype=tf.float32)
        accuracy = tf.reduce_mean(corrects)

        return accuracy

    @property
    def metrics(self):
        """Expose all VAE and classifier trackers to the Keras loop.

        Returns:
            list[tf.keras.metrics.Mean]: Total, KL, reconstruction,
            classification-loss, and classification-accuracy trackers in that
            order.  Keras resets them between epochs/evaluations.
        """
        return [
            self.total_loss_tracker, 
            self.kl_loss_tracker, 
            self.recon_loss_tracker, 
            self.clf_loss_tracker, 
            self.clf_accuracy_tracker, 
        ]

    def call(self, inputs, training=False):
        """Reconstruct conditional inputs and classify the reconstruction.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): ``(x, one_hot_y)`` with shapes
                ``[batch, data_dim]`` and ``[batch, class_num]``.
            training (bool | tf.Tensor): Keras training flag forwarded to the
                VAE.  The classifier is invoked without an explicit flag.

        Returns:
            tuple[tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]:
            Latent statistics/sample shaped ``[batch, latent_dim]``,
            reconstruction shaped ``[batch, data_dim]``, and classifier output
            shaped ``[batch, class_num]``.
        """
        (z_mean, z_log_var, z), reconstructed = super().call(inputs, training)
        prediction = self.classifier(reconstructed)

        return (z_mean, z_log_var, z), reconstructed, prediction

    def train_step(self, inputs):
        """Optimize VAE and classifier losses for one conditional batch.

        Classification cross-entropy is computed from ``classifier(x)`` and
        reduced with ``sum`` across the batch, whereas the reconstruction loss
        follows the compiled Keras reduction and KL is a batch mean.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Feature vectors and one-hot
                labels shaped ``[batch, data_dim]`` and
                ``[batch, class_num]``.

        Returns:
            dict[str, tf.Tensor]: Scalar running means under ``loss``,
            ``kl_loss``, ``recon_loss``, ``clf_loss``, and ``clf_accuracy``.
        """
        x, y = inputs

        with tf.GradientTape() as tape:
            (z_mean, z_log_var, _), x_recon, _ = self(inputs, training=True)
            y_pred = self.classifier(x)

            kl_loss = self.compute_kl(z_mean, z_log_var)

            recon_loss = self.compiled_loss(
                x, 
                x_recon, 
                regularization_losses=self.losses,
            )

            clf_loss = tf.reduce_sum(
                losses.categorical_crossentropy(y, y_pred),
            )

            total_loss = self.beta * kl_loss + recon_loss + self.alpha * clf_loss
        
        clf_acc = self._compute_accuracy(y, y_pred)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.clf_loss_tracker.update_state(clf_loss)
        self.clf_accuracy_tracker.update_state(clf_acc)

        return {
            "loss": self.total_loss_tracker.result(), 
            "kl_loss": self.kl_loss_tracker.result(), 
            "recon_loss": self.recon_loss_tracker.result(),
            "clf_loss": self.clf_loss_tracker.result(), 
            "clf_accuracy": self.clf_accuracy_tracker.result(),
        }

    def test_step(self, inputs):
        """Evaluate VAE and original-input classifier losses without updates.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Feature vectors and one-hot
                labels shaped ``[batch, data_dim]`` and
                ``[batch, class_num]``.

        Returns:
            dict[str, tf.Tensor]: Scalar running means under ``loss``,
            ``kl_loss``, ``recon_loss``, ``clf_loss``, and ``clf_accuracy``.
        """
        x, y = inputs

        (z_mean, z_log_var, _), x_recon, _ = self(inputs, training=False)
        y_pred = self.classifier(x)

        kl_loss = self.compute_kl(z_mean, z_log_var)

        recon_loss = self.compiled_loss(
            x, 
            x_recon, 
            regularization_losses=self.losses,
        )

        clf_loss = tf.reduce_sum(
            losses.categorical_crossentropy(y, y_pred)
        )

        total_loss = self.beta * kl_loss + recon_loss + self.alpha * clf_loss
        clf_acc = self._compute_accuracy(y, y_pred)

        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.clf_loss_tracker.update_state(clf_loss)
        self.clf_accuracy_tracker.update_state(clf_acc)

        return {
            "loss": self.total_loss_tracker.result(), 
            "kl_loss": self.kl_loss_tracker.result(), 
            "recon_loss": self.recon_loss_tracker.result(),
            "clf_loss": self.clf_loss_tracker.result(), 
            "clf_accuracy": self.clf_accuracy_tracker.result(),
        }

    def train(self, x, y, **kwargs):
        """Fit with decoder-accuracy monitoring from the attached classifier.

        Args:
            x (numpy.ndarray | tf.Tensor): Samples shaped
                ``[samples, data_dim]``.
            y (numpy.ndarray | tf.Tensor): One-hot labels shaped
                ``[samples, class_num]``.
            **kwargs (object): Options accepted by
                :meth:`VariationalAutoencoder.train`: ``train_num``, ``epochs``,
                ``batch_size``, ``validation_data``, ``callbacks_list``, and
                ``verbose``.  ``x``, ``y``, ``clf``, and ``callbacks_monitor``
                are forbidden because this method supplies them.  Example:
                ``train(x, y, epochs=20, train_num=-1,
                validation_data=(x_val, y_val))``.

        Returns:
            dict[str, list[float]]: Keras epoch history.  Automatically created
            early stopping monitors ``"val_clf_accuracy"``; a decoder-accuracy
            callback is always prepended unless callback construction fails.

        Raises:
            AssertionError: If a reserved key is supplied in ``kwargs``.
            TypeError: If an unsupported training keyword is supplied.
        """
        assert "x" not in kwargs
        assert "y" not in kwargs
        assert "clf" not in kwargs
        assert "callbacks_monitor" not in kwargs


        return super().train(x, y, clf=self.classifier, 
                            callbacks_monitor="val_clf_accuracy", 
                            **kwargs)
