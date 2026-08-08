from typing import TypeAlias, Literal

import tensorflow as tf
from tensorflow.keras import metrics

from diffusion.models.transformer.di_t_classifier import DiTClassifier
from diffusion.models.wrapper import NetworkName


ComputeType: TypeAlias = Literal[
    "chunked", 
    "batched"
]


class EnsembleAccuracy(metrics.Metric):

    def __init__(
        self, 
        diffusion_clf: DiTClassifier, 
        netwrok_name: NetworkName = "ema", 
        compute_type: ComputeType = "chunked", 
        weighted: bool = False, 
        max_t: int = 128, 
        t_chunk_size: int = 16, 
        random_seed: int | None = None, 
        name: str | None = "ensemble_accuracy", 
        **kwargs
    ):
        super().__init__(
            name=name, 
            **kwargs
        )

        assert max_t <= diffusion_clf.timesteps, \
            "max_t cannot be more than timesteps."


        if compute_type == "chunked":
            self.ensemble_predict = self.ensemble_predict_chunked
        elif compute_type == "batched":
            self.ensemble_predict = self.ensemble_predict_batched
        else:
            raise ValueError("compute_type can either be chunked or batched.")

        self.diffusion_clf = diffusion_clf
        self.network = self.diffusion_clf.ema_network if netwrok_name == "ema" \
                    else self.diffusion_clf.network
        self.weighted = weighted
        self.max_t = int(max_t)
        self.t_chunk_size = int(t_chunk_size)
        self.random_seed = random_seed

        self.tracker = metrics.SparseCategoricalAccuracy(name="tracker")

    def ensemble_predict_batched(self, x, training=None):
        batch_size = tf.shape(x)[0]
        ts = tf.range(self.max_t, dtype=tf.int32)

        x_rep = tf.repeat(x, repeats=self.max_t, axis=0)
        t_rep = tf.tile(ts, multiples=[batch_size])
        uncond_labels = tf.zeros((batch_size * self.max_t,), dtype=tf.uint8)

        x_rep, *_ = self.diffusion_clf.noisify(
            x_rep, 
            t_rep, 
            seed=self.random_seed
        )

        cls_pred = self.network.predict_class(
            (x_rep, t_rep, uncond_labels), 
            training=training
        )
        cls_pred = tf.reshape(
            cls_pred,
            (batch_size, self.max_t, -1)
        )

        denominator = tf.cast(self.max_t, tf.float32)
        if self.weighted:
            weights = 1.0 - tf.cast(ts, tf.float32) / tf.cast(self.max_t, tf.float32)
            weights = tf.reshape(weights, (1, self.max_t, 1))
            cls_pred = cls_pred * weights

            denominator = tf.reduce_sum(weights)              

        return tf.reduce_sum(cls_pred, axis=1) / denominator

    def ensemble_predict_chunked(self, x, training=None):
        batch_size = tf.shape(x)[0]
        num_classes = self.network.num_classes

        pred_sum = tf.zeros(
            (batch_size, num_classes), 
            dtype=tf.float32
        )
        for start in range(0, self.max_t, self.t_chunk_size):
            chunk_t = min(self.t_chunk_size, self.max_t - start)
            ts_chunk = tf.range(start, start + chunk_t, dtype=tf.int32)
            t_rep = tf.tile(ts_chunk, multiples=[batch_size])
            uncond_labels = tf.zeros(
                (batch_size * chunk_t,), 
                dtype=tf.uint8
            )

            x_rep = tf.repeat(x, repeats=chunk_t, axis=0)
            x_rep, *_ = self.diffusion_clf.noisify(
                x_rep, t_rep, 
                seed=self.random_seed
            )

            cls_pred = self.network.predict_class(
                (x_rep, t_rep, uncond_labels), 
                training=training
            )
            cls_pred = tf.reshape(
                cls_pred, 
                (batch_size, chunk_t, num_classes)
            )

            if self.weighted:
                weights = 1.0 - tf.cast(ts_chunk, tf.float32) / tf.cast(self.max_t, tf.float32)
                weights = tf.reshape(weights, (1, chunk_t, 1))
                cls_pred = cls_pred * weights

            pred_sum += tf.reduce_sum(cls_pred, axis=1)

        denominator = tf.cast(self.max_t, tf.float32)
        if self.weighted:
            denominator = tf.reduce_sum(
                1. - tf.range(self.max_t, dtype=tf.float32) / tf.cast(self.max_t, tf.float32)
            )

        return pred_sum / denominator

    def test_step(self, y_true, x):
        y_pred = self.ensemble_predict(x)
        self.update_state(y_true, y_pred)

        return self.result()

    def evaluate(self, dataset):
        dataset_len = len(dataset)
        acc = 0.

        for i, (x, y)  in enumerate(dataset):
            print(f"\rStep ({i+1}/{dataset_len}) --- Ensemble Accuracy: {acc:.4f}", end='')

            acc = self.test_step(y, x)

        return self.result().numpy()

    def update_state(self, y_true, y_pred):
        self.tracker.update_state(y_true, y_pred)

    def result(self):
        return self.tracker.result()

    def reset_state(self):
        self.tracker.reset_state()
