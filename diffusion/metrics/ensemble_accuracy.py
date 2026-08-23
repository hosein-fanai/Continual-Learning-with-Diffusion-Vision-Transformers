"""Accuracy based on class predictions ensembled across diffusion noise levels."""

import tensorflow as tf
from tensorflow.keras import metrics

import numpy as np

from typing import Any, TypeAlias, Literal

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
    ``noisify``, ``network``, and ``ema_network``. Weighted evaluation also
    requires ``get_noise_and_signal_rates``. Each selected inner network must
    expose ``num_classes`` and ``predict_class``.

    Args:
        diffusion_clf: A ``DiffusionClassifier``-compatible wrapper.
        netwrok_name: Historical misspelling retained by the public API.
            ``"ema"`` selects ``ema_network`` and ``"raw"`` selects
            ``network``. Only these two values are valid.
        compute_type: ``"chunked"`` for bounded memory or ``"batched"`` for
            one larger network call.
        weighted: If true, use normalized signal-to-noise-ratio weights so
            cleaner timesteps contribute more. If false, use a uniform mean.
        max_t: Positive number of evaluated timesteps, no greater than the
            wrapper's total ``timesteps``. Timestep ``max_t`` itself is excluded.
        t_chunk_size: Positive number of timesteps per call in chunked mode.
            Values larger than ``max_t`` simply produce one chunk.
        seed: Optional seed passed to the wrapper's Gaussian noising
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
        diffusion_clf: Any, 
        netwrok_name: NetworkName = "ema", 
        compute_type: ComputeType = "chunked", 
        weighted: bool = False, 
        max_t: int = 128, 
        t_chunk_size: int = 16, 
        seed: int | None = None, 
        name: str | None = "ensemble_accuracy", 
        **kwargs: Any
    ) -> None:
        """Bind a classifier wrapper and initialize the accuracy tracker.

        Args:
            diffusion_clf (Any): Diffusion-classifier wrapper exposing
                ``timesteps``, ``noisify``, ``network``, and ``ema_network``.
            netwrok_name (NetworkName): ``"ema"`` or ``"raw"`` network selector.
            compute_type (ComputeType): ``"chunked"`` or ``"batched"``.
            weighted (bool): Whether timesteps use normalized SNR weights.
            max_t (int): Positive number of timesteps to ensemble.
            t_chunk_size (int): Positive timesteps per chunked network call.
            seed (int | None): Optional noising seed.
            name (str | None): Keras metric name.
            **kwargs (Any): Standard Keras metric options.

        Returns:
            ``None``.
        """

        super().__init__(
            name=name, 
            **kwargs
        )

        required = ("timesteps", "noisify", "network", "ema_network")
        missing = [name for name in required if not hasattr(diffusion_clf, name)]
        # Require the wrapper protocol attributes used by this metric.
        if missing:
            raise TypeError(
                "diffusion_clf is missing required attributes: " + ", ".join(missing)
            )
        # Require a callable forward-noising operation.
        if not callable(diffusion_clf.noisify):
            raise TypeError("diffusion_clf.noisify must be callable.")
        # SNR weighting uses the wrapper's existing schedule-rate lookup.
        if weighted and not callable(
            getattr(diffusion_clf, "get_noise_and_signal_rates", None)
        ):
            raise TypeError(
                "Weighted evaluation requires callable "
                "diffusion_clf.get_noise_and_signal_rates."
            )
        # Require a positive integer diffusion horizon from the wrapper.
        if not isinstance(diffusion_clf.timesteps, int) \
        or isinstance(diffusion_clf.timesteps, bool) \
        or diffusion_clf.timesteps < 1:
            raise ValueError("diffusion_clf.timesteps must be a positive integer.")
        # Require a positive integer ensemble horizon.
        if not isinstance(max_t, int) or isinstance(max_t, bool) or max_t < 1:
            raise ValueError("max_t must be a positive integer.")
        # Keep the ensemble horizon within the wrapper's trained horizon.
        if max_t > diffusion_clf.timesteps:
            raise ValueError("max_t cannot exceed diffusion_clf.timesteps.")
        # Require a positive integer timestep chunk size.
        if not isinstance(t_chunk_size, int) or isinstance(t_chunk_size, bool) \
        or t_chunk_size < 1:
            raise ValueError("t_chunk_size must be a positive integer.")
        # Restrict prediction to the wrapper's raw or EMA network.
        if netwrok_name not in ("ema", "raw"):
            raise ValueError("netwrok_name must be 'ema' or 'raw'.")

        # Use bounded-memory prediction when timesteps should be chunked.
        if compute_type == "chunked":
            self.ensemble_predict = self.ensemble_predict_chunked
        # Use one vectorized prediction when all timesteps fit in one batch.
        elif compute_type == "batched":
            self.ensemble_predict = self.ensemble_predict_batched
        # Reject unknown ensemble-computation strategies.
        else:
            raise ValueError("compute_type can either be chunked or batched.")

        self.diffusion_clf = diffusion_clf
        self.network = self.diffusion_clf.get_network(netwrok_name)
        # Require the selected network's class-prediction interface.
        if self.network is None or not callable(
            getattr(self.network, "predict_class", None)
        ) or not isinstance(getattr(self.network, "num_classes", None), int) \
        or isinstance(getattr(self.network, "num_classes", None), bool) \
        or self.network.num_classes < 1:
            raise TypeError(
                "The selected network must expose a positive integer num_classes "
                "and callable predict_class."
            )
        self.weighted = weighted
        self.max_t = int(max_t)
        self.t_chunk_size = int(t_chunk_size)
        self.seed = seed

        self.tracker = metrics.SparseCategoricalAccuracy(
            name="tracker",
            dtype=self.dtype,
        )

    def _get_timestep_weights(self) -> tf.Tensor:
        """Return uniform or normalized SNR weights for all ensemble steps.

        Returns:
            tf.Tensor: One nonnegative weight per timestep, shaped ``[max_t]``.
        """

        # Preserve a uniform mean when schedule-aware weighting is disabled.
        if not self.weighted:
            return tf.ones((self.max_t,), dtype=self.dtype)

        timesteps = tf.range(self.max_t, dtype=tf.int32)
        signal_rates, noise_rates = self.diffusion_clf.get_noise_and_signal_rates(
            timesteps
        )

        signal_power = tf.cast(
            tf.square(signal_rates), 
            tf.float32
        )
        noise_power = tf.cast(
            tf.square(noise_rates), 
            tf.float32
        )
        epsilon = tf.cast(
            tf.keras.backend.epsilon(), 
            tf.float32
        )
        log_snr = (
            tf.math.log(tf.maximum(signal_power, epsilon))
            - tf.math.log(tf.maximum(noise_power, epsilon))
        )

        return tf.cast(tf.nn.softmax(log_snr), self.dtype)

    def reset_state(self) -> None:
        """Reset correct-example and example-count accumulators to zero.

        Returns:
            ``None``.
        """

        self.tracker.reset_state()

    def update_state(
        self, 
        y_true: tf.Tensor, 
        y_pred: tf.Tensor, 
        sample_weight: tf.Tensor | None = None
    ) -> None:
        """Accumulate sparse categorical accuracy statistics.

        Args:
            y_true (tf.Tensor): Sparse integer labels shaped ``[batch]`` or
                ``[batch, 1]``.
            y_pred (tf.Tensor): Floating scores shaped ``[batch, num_classes]``. Scores may
                be logits or probabilities because accuracy uses ``argmax``.
            sample_weight (tf.Tensor | None): Optional per-example weights.

        Returns:
            ``None``. Internal correct and total counts are updated in place.
        """

        self.tracker.update_state(
            y_true, y_pred, 
            sample_weight=sample_weight
        )

    def result(self) -> tf.Tensor:
        """Return cumulative accuracy.

        Returns:
            tf.Tensor: Scalar floating value from the internal sparse categorical
            accuracy tracker.
        """

        return self.tracker.result()

    def ensemble_predict_batched(
        self, 
        x: tf.Tensor, 
        training: bool | tf.Tensor | None = None
    ) -> tf.Tensor:
        """Average predictions for all examples and timesteps in one call.

        Args:
            x (tf.Tensor): Clean floating image tensor
                ``[batch, height, width, channels]``.
            training (bool | tf.Tensor | None): Optional flag forwarded to ``network.predict_class``.
                Normally false for metric evaluation.

        Returns:
            tf.Tensor: Floating scores shaped ``[batch, num_classes]`` containing
            the uniform or SNR-weighted timestep mean.
        """

        batch_size = tf.shape(x)[0]
        ts = tf.range(self.max_t, dtype=tf.int32)

        x_rep = tf.repeat(x, repeats=self.max_t, axis=0)
        t_rep = tf.tile(ts, multiples=[batch_size])
        uncond_labels = tf.zeros(
            (batch_size * self.max_t,), 
            dtype=tf.uint8
        )

        x_rep, *_ = self.diffusion_clf.noisify(
            x_rep, 
            t_rep, 
            seed=self.seed
        )

        cls_pred = self.network.predict_class(
            (x_rep, t_rep, uncond_labels), 
            training=training
        )
        cls_pred = tf.cast(cls_pred, self.dtype)
        cls_pred = tf.reshape(
            cls_pred,
            (batch_size, self.max_t, -1)
        )

        weights = self._get_timestep_weights()
        denominator = tf.reduce_sum(weights)
        cls_pred = cls_pred * tf.reshape(weights, (1, self.max_t, 1))

        return tf.reduce_sum(cls_pred, axis=1) / denominator

    def ensemble_predict_chunked(
        self, 
        x: tf.Tensor, 
        training: bool | tf.Tensor | None = None
    ) -> tf.Tensor:
        """Average timestep predictions using bounded-size network calls.

        Args:
            x (tf.Tensor): Clean floating image tensor
                ``[batch, height, width, channels]``.
            training (bool | tf.Tensor | None): Optional flag forwarded to ``network.predict_class``.

        Returns:
            ``tf.Tensor`` in the metric's configured dtype and shape
            ``[batch, num_classes]``. Only ``batch * t_chunk_size`` noised
            images are materialized per iteration.
        """

        batch_size = tf.shape(x)[0]
        num_classes = self.network.num_classes
        weights = self._get_timestep_weights()

        pred_sum = tf.zeros(
            (batch_size, num_classes), 
            dtype=self.dtype,
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
                x_rep, 
                t_rep, 
                seed=self.seed
            )

            cls_pred = self.network.predict_class(
                (x_rep, t_rep, uncond_labels), 
                training=training
            )
            cls_pred = tf.cast(cls_pred, self.dtype)
            cls_pred = tf.reshape(
                cls_pred, 
                (batch_size, chunk_t, num_classes)
            )

            chunk_weights = tf.reshape(
                weights[start: start + chunk_t], 
                (1, chunk_t, 1)
            )
            cls_pred = cls_pred * chunk_weights

            pred_sum += tf.reduce_sum(cls_pred, axis=1)

        denominator = tf.reduce_sum(weights)

        return pred_sum / denominator

    def test_step(
        self, 
        y_true: tf.Tensor, 
        x: tf.Tensor, 
        sample_weight: tf.Tensor | None = None
    ) -> tf.Tensor:
        """Update accuracy from one labeled image batch.

        Args:
            y_true (tf.Tensor): Sparse integer labels shaped ``[batch]`` or
                ``[batch, 1]``.
            x (tf.Tensor): Clean floating images shaped
                ``[batch, height, width, channels]``.
            sample_weight (tf.Tensor | None): Optional per-example weights.

        Returns:
            tf.Tensor: Scalar floating cumulative accuracy.
        """

        y_pred = self.ensemble_predict(x)
        self.update_state(
            y_true, y_pred, 
            sample_weight=sample_weight
        )

        return self.result()

    def evaluate(
        self, 
        dataset: Any, 
        verbose: bool = True
    ) -> np.generic:
        """Evaluate a finite iterable of ``(images, labels)`` batches.

        This convenience loop prints progress and does not reset existing
        metric state. Call :meth:`reset_state` first when an independent result
        is required.

        Args:
            dataset (Any): Sized iterable yielding ``(images, labels)`` or
                ``(images, labels, sample_weight)`` batches. ``len(dataset)``
                must work.
            verbose (bool): Print batch progress when true.

        Returns:
            np.generic: NumPy scalar containing cumulative sparse categorical
            accuracy.
        """

        dataset_len = len(dataset)
        acc = 0.

        for i, batch in enumerate(dataset):
            # Print the current cumulative value when progress is enabled.
            if verbose:
                print(
                    f"\rStep ({i+1}/{dataset_len}) --- "
                    f"Ensemble Accuracy: {acc:.4f}", 
                    end=''
                )

            # Treat two-item batches as unweighted examples and labels.
            if len(batch) == 2:
                x, y = batch
                sample_weight = None
            # Forward the third batch item as per-example sample weights.
            elif len(batch) == 3:
                x, y, sample_weight = batch
            # Reject dataset batches outside the supported Keras tuple forms.
            else:
                raise ValueError(
                    "dataset batches must contain two or three values."
                )

            acc = self.test_step(
                y, x, 
                sample_weight=sample_weight
            )

        # Finish the in-place progress line when one was printed.
        if verbose:
            print()

        return self.result().numpy()


def run_self_tests() -> dict[str, str]:
    """Test all ensembling and state paths of :class:`EnsembleAccuracy`.

    Args:
        None.

    Returns:
        dict[str, str]: A one-entry mapping after network selection, both compute strategies,
        weighting, chunk boundaries, seeds, training flags, metric state,
        evaluation looping, invalid modes, and numeric boundaries pass.
    """

    import contextlib
    import io
    import numpy as np
    from types import SimpleNamespace


    predict_calls = []
    noisify_calls = []


    def predict_class(
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
        training: bool | None = None
    ) -> tf.Tensor:
        """Return deterministic timestep-dependent three-class scores.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Replicated images,
                timesteps, and labels.
            training (bool | None): Optional flag recorded for forwarding
                verification.

        Returns:
            tf.Tensor: Float32 one-hot scores shaped ``[batch, 3]``.
        """

        _, timesteps, labels = inputs
        predict_calls.append((timesteps.numpy(), labels.numpy(), training))

        return tf.one_hot(tf.math.floormod(timesteps, 3), 3, dtype=tf.float32)


    def noisify(
        images: tf.Tensor, 
        timesteps: tf.Tensor, 
        seed: int | None = None
    ) -> tuple[tf.Tensor]:
        """Record noising arguments while returning images unchanged.

        Args:
            images (tf.Tensor): Replicated clean image tensor.
            timesteps (tf.Tensor): Integer timestep tensor paired with
                ``images``.
            seed (int | None): Optional random seed supplied by the metric.

        Returns:
            tuple[tf.Tensor]: A one-item tuple containing ``images``.
        """

        noisify_calls.append((tuple(images.shape), timesteps.numpy(), seed))

        return (images,)


    raw_network = SimpleNamespace(num_classes=3, predict_class=predict_class)
    ema_network = SimpleNamespace(num_classes=3, predict_class=predict_class)
    network_by_name = {"raw": raw_network, "ema": ema_network}
    alpha_bar_values = np.array(
        [0.8, 0.5, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625],
        dtype=np.float32,
    )


    def get_noise_and_signal_rates(
        timesteps: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Return deterministic schedule amplitudes for metric tests.

        Args:
            timesteps (tf.Tensor): Schedule indices to gather.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Signal and noise amplitudes.
        """

        alpha_bar = tf.gather(
            tf.constant(alpha_bar_values, dtype=tf.float32),
            timesteps,
        )

        return tf.sqrt(alpha_bar), tf.sqrt(1.0 - alpha_bar)


    def expected_prediction(max_t: int, weighted: bool) -> np.ndarray:
        """Aggregate the deterministic class IDs with expected weights.

        Args:
            max_t (int): Number of leading timesteps to aggregate.
            weighted (bool): Whether to normalize by timestep SNR.

        Returns:
            np.ndarray: Expected three-class prediction vector.
        """

        weights = np.ones((max_t,), dtype=np.float32)
        # Match the metric's normalized SNR weighting when enabled.
        if weighted:
            weights = alpha_bar_values[:max_t] / (
                1.0 - alpha_bar_values[:max_t]
            )
        weights /= np.sum(weights)

        return np.bincount(
            np.arange(max_t) % 3,
            weights=weights,
            minlength=3,
        ).astype(np.float32)


    wrapper = SimpleNamespace(
        timesteps=8, 
        network=raw_network, 
        ema_network=ema_network, 
        noisify=noisify, 
        get_noise_and_signal_rates=get_noise_and_signal_rates,
        get_network=network_by_name.__getitem__,
    )
    images = tf.ones((2, 2, 2, 1), dtype=tf.float32)

    for weighted in (False, True):
        predict_calls.clear()
        noisify_calls.clear()
        batched = EnsembleAccuracy(
            wrapper, 
            netwrok_name="ema", 
            compute_type="batched", 
            weighted=weighted, 
            max_t=5, 
            t_chunk_size=2, 
            seed=17, 
        )
        batched_prediction = batched.ensemble_predict(images, training=True)
        assert batched.network is ema_network
        assert batched_prediction.shape == (2, 3)
        assert batched_prediction.dtype == tf.float32
        assert len(predict_calls) == 1 and predict_calls[0][2] is True
        assert np.all(predict_calls[0][1] == 0)
        assert noisify_calls[0][0] == (10, 2, 2, 1)
        assert noisify_calls[0][2] == 17
        expected_row = expected_prediction(5, weighted)
        np.testing.assert_allclose(
            batched_prediction.numpy(), 
            np.repeat(expected_row[None, :], 2, axis=0), 
            atol=1e-6,
        )

        predict_calls.clear()
        noisify_calls.clear()
        chunked = EnsembleAccuracy(
            wrapper, 
            netwrok_name="raw", 
            compute_type="chunked", 
            weighted=weighted, 
            max_t=5, 
            t_chunk_size=2, 
            seed=17, 
        )
        chunked_prediction = chunked.ensemble_predict(images, training=False)
        assert chunked.network is raw_network
        np.testing.assert_allclose(
            chunked_prediction.numpy(), 
            batched_prediction.numpy(), 
            atol=1e-6
        )
        assert len(predict_calls) == 3
        assert [call[0].shape[0] for call in predict_calls] == [4, 4, 2]
        assert all(call[2] is False for call in predict_calls)

    oversized_chunk = EnsembleAccuracy(
        wrapper, 
        compute_type="chunked", 
        max_t=4, 
        t_chunk_size=99,
    )
    predict_calls.clear()
    assert oversized_chunk.ensemble_predict(images).shape == (2, 3)
    assert len(predict_calls) == 1

    try:
        EnsembleAccuracy(
            wrapper,
            netwrok_name="not-ema",
            compute_type="batched",
            max_t=1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown network names must fail.")

    stateful = EnsembleAccuracy(wrapper, compute_type="batched", max_t=4)
    labels = tf.zeros((2,), dtype=tf.int32)
    assert float(stateful.test_step(labels, images).numpy()) == 1.0
    assert float(stateful.result().numpy()) == 1.0
    stateful.reset_state()
    assert float(stateful.result().numpy()) == 0.0
    stateful.update_state(
        tf.constant([0, 1]), 
        tf.constant([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]]),
        sample_weight=tf.constant([1.0, 0.0]),
    )
    assert float(stateful.result().numpy()) == 1.0
    stateful.reset_state()
    with contextlib.redirect_stdout(io.StringIO()):
        evaluated = stateful.evaluate([(images, labels), (images, labels)])
    assert float(evaluated) == 1.0

    try:
        EnsembleAccuracy(wrapper, max_t=9)
    except ValueError:
        pass
    else:
        raise AssertionError("max_t above wrapper timesteps must fail.")
    try:
        EnsembleAccuracy(wrapper, compute_type="all-at-once", max_t=2)
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown compute strategies must fail.")

    for invalid_kwargs in (
        {"max_t": 0},
        {"max_t": -1},
        {"max_t": 2.9},
        {"max_t": 2, "t_chunk_size": 0},
        {"max_t": 2, "t_chunk_size": -1},
        {"max_t": 4, "t_chunk_size": 2.9},
    ):
        try:
            EnsembleAccuracy(wrapper, **invalid_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid horizons and chunk sizes must fail.")

    for weighted in (False, True):
        full_horizon = EnsembleAccuracy(
            wrapper, 
            compute_type="chunked", 
            weighted=weighted, 
            max_t=wrapper.timesteps, 
            t_chunk_size=3, 
        )
        assert full_horizon.max_t == wrapper.timesteps
        full_horizon_output = full_horizon.ensemble_predict(images)
        expected_full_row = expected_prediction(wrapper.timesteps, weighted)
        np.testing.assert_allclose(
            full_horizon_output.numpy(), 
            np.repeat(expected_full_row[None, :], 2, axis=0), 
            atol=1e-6,
        )

    named = EnsembleAccuracy(
        wrapper, 
        max_t=1, 
        name="custom_ensemble", 
        dtype="float64"
    )
    assert named.name == "custom_ensemble" and named.dtype == "float64"
    assert named.ensemble_predict(images).dtype == tf.float64

    def get_zero_noise_rates(
        timesteps: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Return a schedule whose first timestep is exactly noiseless.

        Args:
            timesteps (tf.Tensor): Schedule indices to gather.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Signal and noise amplitudes.
        """

        signal = tf.gather(tf.constant([1.0, 0.5]), timesteps)
        noise = tf.gather(tf.constant([0.0, np.sqrt(0.75)]), timesteps)

        return signal, noise


    zero_noise_wrapper = SimpleNamespace(
        timesteps=2,
        network=raw_network,
        ema_network=ema_network,
        noisify=noisify,
        get_noise_and_signal_rates=get_zero_noise_rates,
        get_network=network_by_name.__getitem__,
    )
    zero_noise_metric = EnsembleAccuracy(
        zero_noise_wrapper,
        compute_type="batched",
        weighted=True,
        max_t=2,
    )
    zero_noise_prediction = zero_noise_metric.ensemble_predict(images)
    assert bool(tf.reduce_all(tf.math.is_finite(zero_noise_prediction)))
    tf.debugging.assert_near(
        tf.reduce_sum(zero_noise_prediction, axis=-1),
        tf.ones((2,)),
        atol=1e-6,
    )
    assert bool(tf.reduce_all(
        zero_noise_prediction[:, 0] > zero_noise_prediction[:, 1]
    ))

    invalid_wrapper = SimpleNamespace(
        timesteps=8,
        network=raw_network,
        ema_network=ema_network,
        noisify=None,
    )
    try:
        EnsembleAccuracy(invalid_wrapper, max_t=1)
    except TypeError:
        pass
    else:
        raise AssertionError("The wrapper noising operation must be callable.")

    return {"EnsembleAccuracy": "passed"}


# Run the metric's focused regression tests when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
