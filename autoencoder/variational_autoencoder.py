import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np

from autoencoder.decoder_accuracy_callback import DecoderAccuracyCallback
from common.utils import get_callbacks


class VariationalAutoencoder(models.Model):

    def __init__(
        self, 
        data_dim=2048, 
        latent_dim=16, 
        hiddens_dims=(512, 256, 256), 
        last_activation="linear",
        beta=1., 
        conditioned=False, 
        class_num=None, 
        compile_args=None, 
        **kwargs
    ):
        super().__init__(**kwargs)

        self.latent_dim = latent_dim
        self.beta = beta
        self.conditioned = conditioned
        self.class_num = class_num

        self.encoder = self._build_encoder(data_dim, latent_dim, hiddens_dims, class_num)
        self.decoder = self._build_decoder(data_dim, latent_dim, hiddens_dims[::-1], 
                                        class_num, last_activation)

        self.seen_classes = []

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        if compile_args is None:
            compile_args = {
                "optimizer": "adam",
                "loss": "binary_crossentropy",
            }
            
        self.compile(**compile_args)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
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

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )

            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
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

        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )

        total_loss = recon_loss + self.beta * kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def generate(self, samples_per_class=500, 
                onehot_labels=True, verbose=0):
        if self.conditioned:
            if len(self.seen_classes) == 0:
                return [], []

            z = tf.random.normal(shape=(samples_per_class*len(self.seen_classes), self.latent_dim))
            y = tf.concat([tf.one_hot(tf.cast([i]*samples_per_class, tf.uint8), 
                                    depth=self.class_num) for i in self.seen_classes], 
                                axis=0)
            x = self.decoder.predict((z, y), verbose=verbose)

            x = np.array(x, dtype="float32")
            y = np.array(y, dtype="uint8")

            if not onehot_labels:
                y = np.argmax(y, axis=-1)

            return x, y

        z = tf.random.normal(shape=(samples_per_class, self.latent_dim))
        x = self.decoder.predict(z, verbose=verbose)

        x = np.array(x, dtype="float32")
        
        return x

    def train(self, x, y=None, train_num=1_000, 
            epochs=5, batch_size=256, validation_data=None, 
            callbacks_list=None, clf=None, verbose=1):
        if train_num != -1:
            indices = np.random.randint(0, len(x), (train_num,))
            x = x[indices]
            if y is not None:
                y = y[indices]

        new_classes = np.unique(np.argmax(y, axis=-1))
        self.seen_classes.extend(new_classes)
        list(set(self.seen_classes))

        if callbacks_list is None:
            callbacks_list = get_callbacks(monitor="decoder_accuracy")
        else:
            callbacks_list = callbacks_list

        history = self.fit(
            x, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data, 
            callbacks=[
                DecoderAccuracyCallback(classifier=clf)
            ]+callbacks_list, 
            verbose=verbose,
        ).history

        return history

    def update_seen_classes(self, cls):
        self.seen_classes.append(cls)

    def _build_encoder(self, input_dim, latent_dim, 
                    hiddens_dims, class_num=None):
        x_inputs = layers.Input(shape=(input_dim,), name="x_input")

        if self.conditioned:
            y_inputs = layers.Input(shape=(class_num,), name="y_input")
            x = layers.Concatenate()([x_inputs, y_inputs])
            inputs = [x_inputs, y_inputs]
        else:
            x = x_inputs
            inputs = x_inputs

        for hidden_dim in hiddens_dims:
            x = layers.Dense(hidden_dim, activation="relu")(x)

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        encoder = models.Model(inputs, [z_mean, z_log_var, z], name="encoder")

        return encoder

    def _build_decoder(self, output_dim, latent_dim, 
                    hiddens_dims, class_num, 
                    last_activation):
        z_inputs = layers.Input(shape=(latent_dim,), name="z_input")

        if self.conditioned:
            y_inputs = layers.Input(shape=(class_num,), name="y_input")
            z = layers.Concatenate()([z_inputs, y_inputs])
            inputs = [z_inputs, y_inputs]
        else:
            z = z_inputs
            inputs = z_inputs

        for hidden_dim in hiddens_dims:
            z = layers.Dense(hidden_dim, activation="relu")(z)

        outputs = layers.Dense(output_dim, activation=last_activation)(z)

        decoder = models.Model(inputs, outputs, name="decoder")

        return decoder


if __name__ == "__main__":
    from common.utils import init, load_cifar10


    init()

    x_train, y_train, x_val, y_val, *_ = load_cifar10(return_features=True, onehot_labels=True, verbose=0) # , preprocess="normalize"

    vae = VariationalAutoencoder(conditioned=True) # , last_activation="tanh"

    vae.train(
        x_train, y_train, 
        train_num=-1, 
        epochs=5, 
        batch_size=256, 
        validation_data=(x_val, y_val), 
        clf=models.load_model("./models/hyperas/cifar10_dnn_model_00.h5")
    )

