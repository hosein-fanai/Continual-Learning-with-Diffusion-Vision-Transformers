"""Accuracy based on class predictions ensembled across diffusion noise levels."""

from typing import TypeAlias, Literal

import tensorflow as tf
from tensorflow.keras import metrics

from diffusion.models.transformer.di_t_classifier import DiTClassifier
from diffusion.models.wrapper import NetworkName


ComputeType: TypeAlias = Literal[
    "chunked", 
    "batched"
]
"""Memory-oriented chunked evaluation or single-call batched evaluation."""


class EnsembleAccuracy(metrics.Metric):
    """Measure classifier accuracy after averaging predictions over timesteps.

    For every clean input, this metric creates noisy versions at integer
    timesteps ``0`` through ``max_t - 1``, obtains unconditional class
    predictions, averages them, and delegates accuracy tracking to
    ``SparseCategoricalAccuracy``. ``"batched"`` evaluates all replicas in one
    network call; ``"chunked"`` performs smaller calls and has lower peak
    memory use while computing the same aggregate.

    Despite the historical ``DiTClassifier`` annotation, ``diffusion_clf`` must
    be the trained classifier *wrapper*: it must expose ``timesteps``,
    ``noisify``, ``network``, and ``ema_network``. Each selected inner network
    must expose ``num_classes`` and ``predict_class``.

    Args:
        diffusion_clf: A ``DiffusionClassifier``-compatible wrapper.
        netwrok_name: Historical misspelling retained by the public API.
            ``"ema"`` selects ``ema_network`` and ``"raw"`` selects
            ``network``. Only these values are valid; any non-``"ema"`` value
            currently falls back to the raw network.
        compute_type: ``"chunked"`` for bounded memory or ``"batched"`` for
            one larger network call.
        weighted: If true, weight timestep ``t`` by ``1 - t / max_t`` so clean
            and lightly noised samples contribute more. If false, use a uniform
            mean.
        max_t: Positive number of evaluated timesteps, no greater than the
            wrapper's total ``timesteps``. Timestep ``max_t`` itself is excluded.
        t_chunk_size: Positive number of timesteps per call in chunked mode.
            Values larger than ``max_t`` simply produce one chunk.
        random_seed: Optional seed passed to the wrapper's Gaussian noising
            operation, or ``None`` for its configured/default randomness.
        name: Keras metric name.
        **kwargs: Standard ``tf.keras.metrics.Metric`` options, notably
            ``dtype``.

    Inputs:
        Clean floating images ``x`` shaped
        ``[batch, height, width, channels]`` and sparse integer labels
        ``y_true`` shaped ``[batch]`` (or ``[batch, 1]``).

    Outputs:
        Ensemble prediction tensors shaped ``[batch, num_classes]`` and a
        scalar floating accuracy from :meth:`result`.
    """

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
        """Bind a classifier wrapper and initialize the accuracy tracker.

        Arguments and accepted types are documented on the class.

        Returns:
            ``None``.
        """

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
        """Average predictions for all examples and timesteps in one call.

        Args:
            x: Clean floating image tensor
                ``[batch, height, width, channels]``.
            training: Optional flag forwarded to ``network.predict_class``.
                Normally false for metric evaluation.

        Returns:
            Floating ``tf.Tensor`` shaped ``[batch, num_classes]`` containing
            the uniform or linearly weighted timestep mean.
        """

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
        """Average timestep predictions using bounded-size network calls.

        Args:
            x: Clean floating image tensor
                ``[batch, height, width, channels]``.
            training: Optional flag forwarded to ``network.predict_class``.

        Returns:
            ``tf.Tensor`` of dtype ``tf.float32`` and shape
            ``[batch, num_classes]``. Only ``batch * t_chunk_size`` noised
            images are materialized per iteration.
        """

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
        """Update accuracy from one labeled image batch.

        Args:
            y_true: Sparse integer labels shaped ``[batch]`` or
                ``[batch, 1]``.
            x: Clean floating images shaped
                ``[batch, height, width, channels]``.

        Returns:
            Scalar floating ``tf.Tensor`` containing cumulative accuracy.
        """

        y_pred = self.ensemble_predict(x)
        self.update_state(y_true, y_pred)

        return self.result()

    def evaluate(self, dataset):
        """Evaluate a finite iterable of ``(images, labels)`` batches.

        This convenience loop prints progress and does not reset existing
        metric state. Call :meth:`reset_state` first when an independent result
        is required.

        Args:
            dataset: Sized iterable yielding pairs of a floating image tensor
                and sparse integer label tensor. ``len(dataset)`` must work.

        Returns:
            NumPy scalar containing cumulative sparse categorical accuracy.
        """

        dataset_len = len(dataset)
        acc = 0.

        for i, (x, y)  in enumerate(dataset):
            print(f"\rStep ({i+1}/{dataset_len}) --- Ensemble Accuracy: {acc:.4f}", end='')

            acc = self.test_step(y, x)

        return self.result().numpy()

    def update_state(self, y_true, y_pred):
        """Accumulate sparse categorical accuracy statistics.

        Args:
            y_true: Sparse integer labels shaped ``[batch]`` or
                ``[batch, 1]``.
            y_pred: Floating scores shaped ``[batch, num_classes]``. Scores may
                be logits or probabilities because accuracy uses ``argmax``.

        Returns:
            ``None``. This override does not accept sample weights.
        """

        self.tracker.update_state(y_true, y_pred)

    def result(self):
        """Return cumulative accuracy.

        Returns:
            Scalar floating ``tf.Tensor`` from the internal sparse categorical
            accuracy tracker.
        """

        return self.tracker.result()

    def reset_state(self):
        """Reset correct-example and example-count accumulators to zero.

        Returns:
            ``None``.
        """

        self.tracker.reset_state()
