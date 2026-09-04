"""Accuracy based on class predictions ensembled across diffusion noise levels."""

import tensorflow as tf
from tensorflow.keras import metrics

import numpy as np

from typing import Any, TypeAlias, Literal, get_args

from common.runtime import derive_seed, effective_seed

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
    predictions, optionally combines their primary, classifier-regularizer,
    and distillation heads, averages them, and delegates accuracy tracking to
    ``SparseCategoricalAccuracy``. ``"batched"`` evaluates all replicas in one
    network call; ``"chunked"`` performs smaller calls and has lower peak
    memory use while computing the same aggregate.

    Despite the historical ``DiTClassifier`` annotation, ``diffusion_clf`` must
    be the trained classifier *wrapper*: it must expose ``timesteps``,
    ``noisify``, ``q_sample``, ``network``, ``ema_network``, and
    ``get_network``. Weighted
    evaluation also requires ``get_noise_and_signal_rates``. Each selected inner
    network must expose ``num_classes`` and the project's five-or-six-value
    ``predict_class(full_return=True)`` interface. Seeded mode additionally
    requires ``q_sample`` so stateless per-timestep noise can be applied without
    advancing TensorFlow's stateful RNG counters.

    Args:
        diffusion_clf: A ``DiffusionClassifier``-compatible wrapper exposing
            callable ``noisify``, ``q_sample``, and ``get_network`` methods.
        network_name: ``"ema"`` or ``"raw"`` network selector.
        compute_type: ``"chunked"`` for bounded memory or ``"batched"`` for
            one larger network call.
        weighted: If true, use normalized signal-to-noise-ratio weights so
            cleaner timesteps contribute more. If false, use a uniform mean.
        max_t: Positive number of evaluated timesteps, no greater than the
            wrapper's total ``timesteps``. Timestep ``max_t`` itself is excluded.
        t_chunk_size: Positive number of timesteps per call in chunked mode.
            Values larger than ``max_t`` simply produce one chunk.
        seed: Optional master seed for mode-invariant per-timestep Gaussian
            noising streams, or ``None`` for configured/default randomness.
        clf_acc_coef: Nonnegative coefficient for primary class predictions.
        clf_distil_acc_coef: Nonnegative coefficient for distillation-head
            predictions. A positive value requires that optional output.
        ctr_acc_coef: Nonnegative coefficient for the mean of available
            classifier-regularizer predictions. A positive value requires at
            least one such output.
        name: Keras metric name.
        separate_probas: If true, predict once per CFG label. The null-label
            scores contribute to every class and each real label contributes
            only to its corresponding class before a final softmax.
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
        network_name: NetworkName = "ema", 
        compute_type: ComputeType = "chunked", 
        weighted: bool = False, 
        max_t: int = 128, 
        t_chunk_size: int = 16, 
        clf_acc_coef: float = 1., 
        clf_distil_acc_coef: float = 0., 
        ctr_acc_coef: float = 0., 
        separate_probas: bool = False, 
        seed: int | None = None, 
        name: str | None = "ensemble_accuracy", 
        **kwargs: Any
    ) -> None:
        """Bind a classifier wrapper and initialize the accuracy tracker.

        Args:
            diffusion_clf (Any): Diffusion-classifier wrapper exposing
                ``timesteps``, callable ``noisify``/``q_sample``, raw/EMA
                members, and ``get_network``.
            network_name (NetworkName): ``"ema"`` or ``"raw"`` selector.
            compute_type (ComputeType): ``"chunked"`` or ``"batched"``.
            weighted (bool): Whether timesteps use normalized SNR weights.
            max_t (int): Positive number of timesteps to ensemble.
            t_chunk_size (int): Positive timesteps per chunked network call.
            clf_acc_coef (float): Coefficient for primary class predictions.
            clf_distil_acc_coef (float): Coefficient for distillation-head
                predictions.
            ctr_acc_coef (float): Coefficient for averaged classifier
                regularizer predictions.
            separate_probas (bool): Whether to combine separate null and
                class-conditioned CFG predictions.
            seed (int | None): Optional master noising seed.
            name (str | None): Keras metric name.
            **kwargs (Any): Standard Keras metric options.

        Returns:
            ``None``.
        """

        kwargs.setdefault(
            "dtype", 
            getattr(
                getattr(diffusion_clf, "dtype_policy", None), 
                "variable_dtype", 
                tf.keras.mixed_precision.global_policy().variable_dtype
            ),
        )
        super().__init__(
            name=name, 
            **kwargs
        )

        if network_name not in get_args(NetworkName):
            raise ValueError(
                f"network_name must be {get_args(NetworkName)}."
            )
        # Keep the ensemble horizon within the wrapper's trained horizon.
        if max_t > diffusion_clf.timesteps:
            raise ValueError("max_t cannot exceed diffusion_clf.timesteps.")
        # Require at least one prediction head to contribute to the ensemble.
        if clf_acc_coef + clf_distil_acc_coef + ctr_acc_coef <= 0.:
            raise ValueError("At least one accuracy coefficient must be positive.")

        self.diffusion_clf = diffusion_clf
        self.network_name = network_name
        self.network = self.diffusion_clf.get_network(self.network_name)
        # Label zero is unconditional only when the classifier reserves a CFG row.
        if not getattr(self.network, "use_cfg", False):
            raise ValueError(
                "EnsembleAccuracy requires use_cfg=True so label 0 is "
                "an unconditional condition."
            )
        self.compute_type = compute_type
        self.separate_probas = bool(separate_probas)
        self.weighted = weighted
        self.max_t = int(max_t)
        self.t_chunk_size = int(t_chunk_size)
        self.clf_acc_coef = float(clf_acc_coef)
        self.clf_distil_acc_coef = float(clf_distil_acc_coef)
        self.ctr_acc_coef = float(ctr_acc_coef)
        self.seed = effective_seed(seed=(
            getattr(diffusion_clf, "seed", None) 
            if seed is None else seed
        ))
        self.tracker = metrics.SparseCategoricalAccuracy(
            name="tracker", 
            dtype=self.dtype
        )

        # Use bounded-memory prediction when timesteps should be chunked.
        if self.compute_type == "chunked":
            self.ensemble_predict = self.ensemble_predict_chunked
        # Use one vectorized prediction when all timesteps fit in one batch.
        elif self.compute_type == "batched":
            self.ensemble_predict = self.ensemble_predict_batched
        # Reject unknown ensemble-computation strategies.
        else:
            raise ValueError(
                "compute_type can either be chunked or batched."
            )

        if self.separate_probas and (
            not getattr(self.network, "use_cfg", False)
            or getattr(
                self.network, 
                "num_labels", 
                None
            ) != self.network.num_classes + 1
        ):
            raise ValueError(
                "separate_probas requires one CFG null label "
                "in addition to the classifier classes."
            )

    def _predict_classes(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
        training: bool | tf.Tensor | None = None
    ) -> tf.Tensor:
        """Return the configured combination of classifier predictions.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Noisy images,
                timesteps, and unconditional labels.
            training (bool | tf.Tensor | None): Mode forwarded to
                ``network.predict_class``.

        Returns:
            tf.Tensor: Coefficient-weighted class scores shaped
            ``[batch, num_classes]``.

        Raises:
            TypeError: If the network does not return the documented
                classifier full-return tuple.
            ValueError: If a positively weighted optional head is unavailable.
        """

        batch_size = tf.shape(inputs[0])[0]
        num_labels = self.network.num_classes + 1
        if self.separate_probas:
            # Evaluate every noised row under the null and each real CFG label.
            inputs = (
                tf.repeat(inputs[0], num_labels, axis=0), 
                tf.repeat(inputs[1], num_labels, axis=0), 
                tf.tile(
                    tf.cast(tf.range(num_labels), inputs[2].dtype), 
                    [batch_size]
                )
            )

        outputs = self.network.predict_class(
            inputs, 
            max_encoder_num=None,
            full_return=True, 
            training=training
        )

        # Require the common classifier full-return structure.
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 5:
            raise TypeError(
                "network.predict_class(full_return=True) must "
                "return at least five classifier outputs."
            )

        classes_pred = tf.cast(outputs[0], self.dtype)
        total_pred = classes_pred * self.clf_acc_coef

        # Include classifier-token regularizers when their coefficient is positive.
        if self.ctr_acc_coef > 0.:
            regs_list = outputs[3]

            # Require the documented regularizer-head collection.
            if not isinstance(regs_list, (tuple, list)):
                raise TypeError(
                    "Classifier regularizer predictions must be a list or tuple."
                )

            ctr_preds = [
                tf.cast(pred, self.dtype)
                for pred in regs_list
                if pred is not None
            ]

            # Require at least one usable regularizer prediction.
            if not ctr_preds:
                raise ValueError(
                    "ctr_acc_coef > 0 requires at least "
                    "one classifier regularizer prediction."
                )

            total_pred += tf.add_n(ctr_preds) / len(ctr_preds) * self.ctr_acc_coef

        # Include the distillation head when its coefficient is positive.
        if self.clf_distil_acc_coef > 0.:
            # Require a usable distillation prediction for a positive weight.
            if len(outputs) < 6 or outputs[5] is None:
                raise ValueError(
                    "clf_distil_acc_coef > 0 requires a distillation prediction."
                )

            total_pred += tf.cast(
                outputs[5], 
                self.dtype
            ) * self.clf_distil_acc_coef

        if self.separate_probas:
            total_pred = tf.reshape(total_pred, (
                    batch_size, 
                    num_labels, 
                    self.network.num_classes
            ))
            # The null row scores every class; real label j scores class j - 1.
            total_pred = (
                total_pred[:, 0, :] + 
                tf.linalg.diag_part(total_pred[:, 1:, :])
            )

        return total_pred

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

        stable_dtype = tf.as_dtype(self.dtype)
        signal_power = tf.cast(
            tf.square(signal_rates), 
            stable_dtype,
        )
        noise_power = tf.cast(
            tf.square(noise_rates), 
            stable_dtype,
        )
        epsilon = tf.cast(
            tf.keras.backend.epsilon(), 
            stable_dtype,
        )
        log_snr = (
            tf.math.log(tf.maximum(signal_power, epsilon)) - 
            tf.math.log(tf.maximum(noise_power, epsilon))
        )

        return tf.cast(tf.nn.softmax(log_snr), self.dtype)

    def _noisify_timestep_block(
        self, 
        x: tf.Tensor, 
        start: int, 
        count: int
    ) -> tf.Tensor:
        """Create one chunk-boundary-invariant block of noisy replicas.

        Each logical timestep owns a child seed derived only from the master
        seed and timestep ID. Batched and chunked prediction therefore request
        identical noising streams regardless of their network-call grouping.

        Args:
            x (tf.Tensor): Clean images shaped ``[batch,height,width,channels]``.
            start (int): First timestep in the block.
            count (int): Positive number of consecutive timesteps.

        Returns:
            tf.Tensor: Noisy replicas shaped
            ``[batch,count,height,width,channels]``.
        """

        batch_shape = tf.reshape(tf.shape(x)[0], (1,))
        noised_by_timestep = []
        for timestep in range(start, start + count):
            timestep_batch = tf.fill(batch_shape, timestep)
            timestep_seed = derive_seed(
                self.seed, 
                "ensemble_accuracy", 
                "timestep", 
                timestep
            )

            # Seeded metrics use counter-free noise so compute mode and prior
            # random calls cannot change a logical timestep's realization.
            if timestep_seed is not None:
                noises = tf.random.stateless_normal(
                    tf.shape(x), 
                    seed=tf.constant((timestep_seed, 0), dtype=tf.int32), 
                    dtype=x.dtype
                )
                x_t = self.diffusion_clf.q_sample(
                    x, 
                    timestep_batch, 
                    noises
                )
            # Preserve ordinary advancing randomness for explicitly unseeded use.
            else:
                x_t, *_ = self.diffusion_clf.noisify(
                    x, 
                    timestep_batch, 
                    seed=None
                )

            noised_by_timestep.append(x_t)

        return tf.stack(noised_by_timestep, axis=1)

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
            y_true (tf.Tensor): Sparse dataset labels shaped ``[batch]`` or
                ``[batch, 1]``. Dynamic classifiers map their observed labels
                to the wrapper's zero-based class IDs here.
            y_pred (tf.Tensor): Floating scores shaped ``[batch, num_classes]``. Scores may
                be logits or probabilities because accuracy uses ``argmax``.
            sample_weight (tf.Tensor | None): Optional per-example weights.

        Returns:
            ``None``. Internal correct and total counts are updated in place.
        """

        # TensorFlow 2.10 misreads a one-column prediction as binary output.
        if getattr(self.network, "dynamic_num_classes", False) \
        and y_pred.shape[-1] == 1:
            y_pred = tf.concat([y_pred, tf.zeros_like(y_pred)], axis=-1)

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

        x_rep = self._noisify_timestep_block(
            x, 
            start=0, 
            count=self.max_t
        )
        x_rep = tf.reshape(
            x_rep, 
            tf.concat(([-1], tf.shape(x)[1:]), axis=0)
        )
        t_rep = tf.tile(ts, multiples=[batch_size])
        uncond_labels = tf.zeros(
            (batch_size * self.max_t,), 
            dtype=tf.int32
        )

        cls_pred = self._predict_classes(
            (x_rep, t_rep, uncond_labels), 
            training=training
        )
        cls_pred = tf.reshape(
            cls_pred, 
            (batch_size, self.max_t, -1)
        )

        weights = self._get_timestep_weights()
        denominator = tf.reduce_sum(weights)
        cls_pred = cls_pred * tf.reshape(weights, (1, self.max_t, 1))

        total_pred = tf.reduce_sum(cls_pred, axis=1) / denominator

        return tf.nn.softmax(
            total_pred, 
            axis=-1
        ) if self.separate_probas else total_pred

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
            dtype=self.dtype
        )
        for start in range(0, self.max_t, self.t_chunk_size):
            chunk_t = min(self.t_chunk_size, self.max_t - start)
            ts_chunk = tf.range(start, start + chunk_t, dtype=tf.int32)
            t_rep = tf.tile(ts_chunk, multiples=[batch_size])
            uncond_labels = tf.zeros(
                (batch_size * chunk_t,), 
                dtype=tf.int32
            )

            x_rep = self._noisify_timestep_block(
                x, 
                start=start, 
                count=chunk_t
            )
            x_rep = tf.reshape(
                x_rep, 
                tf.concat(([-1], tf.shape(x)[1:]), axis=0)
            )

            cls_pred = self._predict_classes(
                (x_rep, t_rep, uncond_labels), 
                training=training
            )
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

        total_pred = pred_sum / denominator

        return tf.nn.softmax(
            total_pred, 
            axis=-1
        ) if self.separate_probas else total_pred

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
        if getattr(self.network, "dynamic_num_classes", False):
            y_true = self.diffusion_clf._map_classes(y_true)
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
        dict[str, str]: A one-entry mapping after network selection, both compute
        strategies, head coefficients, missing-head errors, weighting, chunk
        boundaries, seeds, training flags, metric state, evaluation looping,
        invalid modes, and numeric boundaries pass.
    """

    import contextlib
    import io
    import numpy as np
    from types import SimpleNamespace


    predict_calls = []
    noisify_calls = []
    q_sample_calls = []


    def predict_class(
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
        max_encoder_num: int | None = -1,
        full_return: bool = False,
        training: bool | None = None
    ) -> tf.Tensor | tuple:
        """Return deterministic primary, regularizer, and distillation scores.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Replicated images,
                timesteps, and labels.
            full_return (bool): Return the common classifier output tuple.
            training (bool | None): Optional flag recorded for forwarding
                verification.

        Returns:
            tf.Tensor | tuple: Primary scores, or the six-value classifier
            output tuple with regularizer and distillation predictions.
        """

        _, timesteps, labels = inputs
        predict_calls.append((
            timesteps.numpy(), labels.numpy(), training, full_return
        ))

        classes = tf.one_hot(
            tf.math.floormod(timesteps, 3), 3, dtype=tf.float32
        )
        # Return only the primary scores for the compact prediction interface.
        if not full_return:
            return classes

        ctr_classes = tf.one_hot(
            tf.math.floormod(timesteps + 1, 3), 3, dtype=tf.float32
        )
        distil_classes = tf.one_hot(
            tf.math.floormod(timesteps + 2, 3), 3, dtype=tf.float32
        )

        return (
            classes, None, [], [ctr_classes, None],
            (None, None), distil_classes
        )


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


    def q_sample(
        images: tf.Tensor,
        timesteps: tf.Tensor,
        noises: tf.Tensor,
    ) -> tf.Tensor:
        """Record and apply deterministic stateless test noise.

        Args:
            images (tf.Tensor): Clean image tensor.
            timesteps (tf.Tensor): Per-row timestep IDs.
            noises (tf.Tensor): Stateless Gaussian noise tensor.

        Returns:
            tf.Tensor: Images shifted by the supplied noise.
        """

        noised_images = images + noises
        q_sample_calls.append((
            timesteps.numpy().copy(),
            noises.numpy().copy(),
            noised_images.numpy().copy(),
        ))

        return noised_images


    raw_network = SimpleNamespace(
        num_classes=3,
        dynamic_num_classes=False,
        use_cfg=True,
        predict_class=predict_class,
    )
    ema_network = SimpleNamespace(
        num_classes=3,
        dynamic_num_classes=False,
        use_cfg=True,
        predict_class=predict_class,
    )
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


    def expected_prediction(
        max_t: int,
        weighted: bool,
        offset: int = 0
    ) -> np.ndarray:
        """Aggregate the deterministic class IDs with expected weights.

        Args:
            max_t (int): Number of leading timesteps to aggregate.
            weighted (bool): Whether to normalize by timestep SNR.
            offset (int): Class-ID offset for an auxiliary prediction head.

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
            (np.arange(max_t) + offset) % 3,
            weights=weights,
            minlength=3,
        ).astype(np.float32)


    wrapper = SimpleNamespace(
        timesteps=8, 
        network=raw_network, 
        ema_network=ema_network, 
        noisify=noisify, 
        q_sample=q_sample,
        get_noise_and_signal_rates=get_noise_and_signal_rates,
        get_network=network_by_name.__getitem__,
    )
    images = tf.ones((2, 2, 2, 1), dtype=tf.float32)

    for weighted in (False, True):
        predict_calls.clear()
        noisify_calls.clear()
        q_sample_calls.clear()
        batched = EnsembleAccuracy(
            wrapper, 
            network_name="ema",
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
        assert predict_calls[0][3] is True
        assert np.all(predict_calls[0][1] == 0)
        assert noisify_calls == []
        assert len(q_sample_calls) == 5
        expected_timestep_seeds = [
            derive_seed(
                17, "ensemble_accuracy", "timestep", timestep
            )
            for timestep in range(5)
        ]
        assert [call[0].tolist() for call in q_sample_calls] == [
            [timestep, timestep] for timestep in range(5)
        ]
        for timestep, call in enumerate(q_sample_calls):
            expected_noise = tf.random.stateless_normal(
                tf.shape(images),
                seed=tf.constant((
                    expected_timestep_seeds[timestep], 0
                ), dtype=tf.int32),
                dtype=images.dtype,
            )
            np.testing.assert_array_equal(
                call[1], expected_noise.numpy()
            )
            np.testing.assert_array_equal(
                call[2], (images + expected_noise).numpy()
            )
        batched_q_sample_calls = [
            tuple(value.copy() for value in call) for call in q_sample_calls
        ]
        expected_row = expected_prediction(5, weighted)
        np.testing.assert_allclose(
            batched_prediction.numpy(), 
            np.repeat(expected_row[None, :], 2, axis=0), 
            atol=1e-6,
        )

        predict_calls.clear()
        noisify_calls.clear()
        q_sample_calls.clear()
        chunked = EnsembleAccuracy(
            wrapper, 
            network_name="raw",
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
        comparison_labels = tf.zeros((2,), dtype=tf.int32)
        batched.update_state(comparison_labels, batched_prediction)
        chunked.update_state(comparison_labels, chunked_prediction)
        tf.debugging.assert_near(batched.result(), chunked.result())
        assert len(predict_calls) == 3
        assert [call[0].shape[0] for call in predict_calls] == [4, 4, 2]
        assert all(call[2] is False for call in predict_calls)
        assert all(call[3] is True for call in predict_calls)
        assert noisify_calls == []
        assert len(set(expected_timestep_seeds)) == 5
        assert len(q_sample_calls) == len(batched_q_sample_calls)
        for batched_call, chunked_call in zip(
            batched_q_sample_calls, q_sample_calls
        ):
            for batched_value, chunked_value in zip(
                batched_call, chunked_call
            ):
                np.testing.assert_array_equal(batched_value, chunked_value)

    combined_kwargs = {
        "weighted": True,
        "max_t": 5,
        "clf_acc_coef": 0.2,
        "ctr_acc_coef": 0.3,
        "clf_distil_acc_coef": 0.5,
    }
    combined_batched = EnsembleAccuracy(
        wrapper,
        compute_type="batched",
        **combined_kwargs
    ).ensemble_predict(images)
    combined_chunked = EnsembleAccuracy(
        wrapper,
        compute_type="chunked",
        t_chunk_size=2,
        **combined_kwargs
    ).ensemble_predict(images)
    expected_combined = (
        expected_prediction(5, True) * 0.2
        + expected_prediction(5, True, offset=1) * 0.3
        + expected_prediction(5, True, offset=2) * 0.5
    )
    np.testing.assert_allclose(
        combined_batched.numpy(),
        np.repeat(expected_combined[None, :], 2, axis=0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        combined_chunked.numpy(), combined_batched.numpy(), atol=1e-6
    )

    conditioned_calls = []
    conditioned_scores = tf.reshape(
        tf.range(110, dtype=tf.float32) / 100.,
        (11, 10)
    )


    def predict_conditioned_class(
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        max_encoder_num: int | None = -1,
        full_return: bool = False,
        training: bool | None = None
    ) -> tf.Tensor | tuple:
        """Return label-dependent scores for the separate-probability test.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Images, timesteps,
                and CFG label IDs.
            full_return (bool): Return the common classifier output tuple.
            training (bool | None): Unused prediction-mode flag.

        Returns:
            tf.Tensor | tuple: Scores selected by CFG label, optionally wrapped
            in the classifier full-return structure.
        """

        del training
        timesteps = tf.cast(inputs[1], tf.int32)
        labels = tf.cast(inputs[2], tf.int32)
        conditioned_calls.append(labels.numpy())
        classes = (
            tf.gather(conditioned_scores, labels)
            + tf.one_hot(
                tf.math.floormod(timesteps, 10),
                10,
                dtype=tf.float32
            ) * 0.5
        )

        return (
            classes, None, [], [None], (None, None)
        ) if full_return else classes


    conditioned_network = SimpleNamespace(
        num_classes=10,
        num_labels=11,
        use_cfg=True,
        dynamic_num_classes=False,
        predict_class=predict_conditioned_class,
    )
    conditioned_networks = {
        "raw": conditioned_network,
        "ema": conditioned_network,
    }
    conditioned_wrapper = SimpleNamespace(
        timesteps=8,
        network=conditioned_network,
        ema_network=conditioned_network,
        noisify=noisify,
        q_sample=q_sample,
        get_network=conditioned_networks.__getitem__,
    )
    separate_kwargs = {
        "separate_probas": True,
        "max_t": 2,
        "seed": 23,
    }
    separate_batched = EnsembleAccuracy(
        conditioned_wrapper,
        compute_type="batched",
        **separate_kwargs
    ).ensemble_predict(images)
    np.testing.assert_array_equal(
        conditioned_calls.pop(),
        np.tile(np.arange(11, dtype=np.uint8), 4)
    )
    separate_chunked = EnsembleAccuracy(
        conditioned_wrapper,
        compute_type="chunked",
        t_chunk_size=1,
        **separate_kwargs
    ).ensemble_predict(images)
    assert len(conditioned_calls) == 2
    for labels in conditioned_calls:
        np.testing.assert_array_equal(
            labels,
            np.tile(np.arange(11, dtype=np.uint8), 2)
        )
    total_scores = (
        conditioned_scores[0]
        + tf.linalg.diag_part(conditioned_scores[1:])
        + tf.one_hot([0, 1], 10, dtype=tf.float32)[0] * 0.5
        + tf.one_hot([0, 1], 10, dtype=tf.float32)[1] * 0.5
    )
    expected_separate = tf.nn.softmax(total_scores).numpy()
    np.testing.assert_allclose(
        separate_batched.numpy(),
        np.repeat(expected_separate[None, :], 2, axis=0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        separate_chunked.numpy(), separate_batched.numpy(), atol=1e-6
    )
    np.testing.assert_allclose(
        tf.reduce_sum(separate_batched, axis=-1).numpy(),
        np.ones((2,), dtype=np.float32),
        atol=1e-6,
    )

    try:
        EnsembleAccuracy(wrapper, separate_probas=True, max_t=1)
    except ValueError:
        pass
    else:
        raise AssertionError("separate_probas must require CFG labels.")

    oversized_chunk = EnsembleAccuracy(
        wrapper, 
        compute_type="chunked", 
        max_t=4, 
        t_chunk_size=99,
    )
    predict_calls.clear()
    assert oversized_chunk.ensemble_predict(images).shape == (2, 3)
    assert len(predict_calls) == 1

    corrected_selector = EnsembleAccuracy(
        wrapper,
        network_name="raw",
        max_t=1,
    )
    default_selector = EnsembleAccuracy(wrapper, max_t=1)
    assert corrected_selector.network is raw_network
    assert corrected_selector.network_name == "raw"
    assert default_selector.network is ema_network
    assert default_selector.network_name == "ema"
    try:
        EnsembleAccuracy(
            wrapper,
            network_name="not-ema",
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

    dynamic_map_calls = []

    def map_dynamic_classes(classes: tf.Tensor) -> tf.Tensor:
        """Map observed dataset labels to contiguous classifier indices."""

        dynamic_map_calls.append(classes.numpy().copy())
        return tf.where(
            tf.equal(classes, 5),
            tf.zeros_like(classes),
            tf.ones_like(classes),
        )

    dynamic_network = SimpleNamespace(
        num_classes=3,
        dynamic_num_classes=True,
        use_cfg=True,
        predict_class=predict_class,
    )
    dynamic_wrapper = SimpleNamespace(
        timesteps=8,
        network=dynamic_network,
        ema_network=dynamic_network,
        noisify=noisify,
        q_sample=q_sample,
        get_network=lambda name: dynamic_network,
        _map_classes=map_dynamic_classes,
    )
    dataset_labels = tf.fill((2,), 5)
    direct_dynamic = EnsembleAccuracy(
        dynamic_wrapper,
        compute_type="batched",
        max_t=4,
    )
    assert float(direct_dynamic.test_step(dataset_labels, images)) == 1.0
    evaluated_dynamic = EnsembleAccuracy(
        dynamic_wrapper,
        compute_type="batched",
        max_t=4,
    )
    assert float(evaluated_dynamic.evaluate(
        [(images, dataset_labels)],
        verbose=False,
    )) == 1.0
    assert len(dynamic_map_calls) == 2
    for mapped_input in dynamic_map_calls:
        np.testing.assert_array_equal(mapped_input, [5, 5])

    no_cfg_network = SimpleNamespace(
        num_classes=3,
        dynamic_num_classes=False,
        use_cfg=False,
        predict_class=predict_class,
    )
    no_cfg_wrapper = SimpleNamespace(
        timesteps=8,
        get_network=lambda name: no_cfg_network,
    )
    try:
        EnsembleAccuracy(no_cfg_wrapper, max_t=1)
    except ValueError as error:
        assert "use_cfg=True" in str(error)
    else:
        raise AssertionError(
            "An ensemble without an unconditional CFG label must fail."
        )

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
        {"max_t": 1, "clf_acc_coef": -1.0},
        {
            "max_t": 1,
            "clf_acc_coef": 0.0,
            "clf_distil_acc_coef": 0.0,
            "ctr_acc_coef": 0.0,
        },
    ):
        try:
            EnsembleAccuracy(wrapper, **invalid_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid ensemble math must fail.")

    for invalid_seed in (-1, 2 ** 32):
        try:
            EnsembleAccuracy(wrapper, max_t=1, seed=invalid_seed)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("Invalid ensemble seeds must fail.")

    def missing_optional_predict_class(
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        max_encoder_num: int | None = -1,
        full_return: bool = False,
        training: bool | None = None
    ) -> tf.Tensor | tuple:
        """Return primary output without usable optional prediction heads.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Images, timesteps,
                and labels used to build deterministic test predictions.
            full_return (bool): Return the common classifier output tuple.
            training (bool | None): Unused prediction-mode flag.

        Returns:
            tf.Tensor | tuple: Primary scores, or a classifier tuple whose
            optional prediction heads are unavailable.
        """

        del training
        classes = tf.one_hot(
            tf.math.floormod(inputs[1], 3), 3, dtype=tf.float32
        )

        return (
            classes, None, [], [None], (None, None)
        ) if full_return else classes


    missing_network = SimpleNamespace(
        num_classes=3,
        dynamic_num_classes=False,
        use_cfg=True,
        predict_class=missing_optional_predict_class,
    )

    def get_missing_network(name: str) -> SimpleNamespace:
        """Return the test network without selecting between raw and EMA copies.

        Args:
            name (str): Requested network name, unused by this test double.

        Returns:
            SimpleNamespace: Network whose optional prediction heads are absent.
        """

        del name
        return missing_network

    missing_wrapper = SimpleNamespace(
        timesteps=8,
        network=missing_network,
        ema_network=missing_network,
        noisify=noisify,
        q_sample=q_sample,
        get_network=get_missing_network,
    )
    for coefficient in ("ctr_acc_coef", "clf_distil_acc_coef"):
        missing_metric = EnsembleAccuracy(
            missing_wrapper,
            max_t=1,
            clf_acc_coef=0.0,
            **{coefficient: 1.0}
        )
        try:
            missing_metric.ensemble_predict(images)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"A missing head weighted by {coefficient} must fail."
            )

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
        q_sample=q_sample,
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

    return {"EnsembleAccuracy": "passed"}


# Run the metric's focused regression tests when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
