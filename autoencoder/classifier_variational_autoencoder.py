import tensorflow as tf
from tensorflow.keras import metrics, losses

from autoencoder.variational_autoencoder import VariationalAutoencoder


class ClassifierVAE(VariationalAutoencoder):

    def __init__(
        self, 
        class_num, 
        classifier, 
        alpha=1., 
        **kwargs
    ):
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
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        
        corrects = tf.cast(y_true == y_pred, dtype=tf.float32)
        accuracy = tf.reduce_mean(corrects)

        return accuracy

    @property
    def metrics(self):
        return [
            self.total_loss_tracker, 
            self.kl_loss_tracker, 
            self.recon_loss_tracker, 
            self.clf_loss_tracker, 
            self.clf_accuracy_tracker, 
        ]

    def call(self, inputs, training=False):
        (z_mean, z_log_var, z), reconstructed = super().call(inputs, training)
        prediction = self.classifier(reconstructed)

        return (z_mean, z_log_var, z), reconstructed, prediction

    def train_step(self, inputs):
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
        assert "x" not in kwargs
        assert "y" not in kwargs
        assert "clf" not in kwargs
        assert "callbacks_monitor" not in kwargs


        return super().train(x, y, clf=self.classifier, 
                            callbacks_monitor="val_clf_accuracy", 
                            **kwargs)
