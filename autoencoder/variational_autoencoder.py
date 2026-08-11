import tensorflow as tf
from tensorflow.keras import metrics, layers, models, optimizers

import numpy as np

from common.model import get_callbacks

from autoencoder.decoder_accuracy_callback import DecoderAccuracyCallback


class VariationalAutoencoder(models.Model):

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
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        return z

    @staticmethod
    def compute_kl(z_mean, z_log_var):
        return -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker, 
            self.kl_loss_tracker, 
            self.recon_loss_tracker, 
        ]

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)

        if self.conditioned:
            _, y = inputs
            decoder_inputs = (z, y)
        else:
            decoder_inputs = z

        return (z_mean, z_log_var, z), self.decoder(decoder_inputs, training=training)

    def train_step(self, data):
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
