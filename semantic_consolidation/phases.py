"""Acquisition and consolidation using the platform's existing semantic head.

The default gradient boundary is the Dense projection and classifier output
head. The shared denoising network always runs in inference mode in these
phases. No class or task condition is supplied to that network.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from common.gradients import apply_policy_gradients
from common.keras_compat import format_variable_name
from common.runtime import derive_seed
from semantic_consolidation.memory import affine_modulation
from semantic_consolidation.objectives import (
    contrastive_alignment_loss,
    modulation_separation_loss,
    normalized_feature_distillation_loss,
    reliability_weights,
)


def semantic_features(
    network: tf.keras.Model,
    images: tf.Tensor,
    times: tf.Tensor,
    stop_backbone: bool = False,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return the actual hidden projection and primary-head probabilities.

    The public raw-network feature API supplies tokens. Reapplying the existing
    classifier layers exposes the projection without replacing its prediction
    path. Dropout/state updates are disabled; gradients remain available.

    Args:
        network (tf.keras.Model): Raw classifier network exposing predict_class and the
            existing primary classifier layers.
        images (tf.Tensor): Numeric sample-major images in the configured model-input scale,
            normally float32 NHWC values in [-1, 1].
        times (tf.Tensor): Integer diffusion timestep vector of shape [N], normally
            tf.int32.
        stop_backbone (bool): Whether to detach shared features before applying the existing
            classifier projection.

    Returns:
        outputs (tuple[tf.Tensor, tf.Tensor]): (features, probabilities): tf.float32 tensors
            [N, D] and [N, C] from the existing unmodulated primary classifier.

    Raises:
        AttributeError: If the raw network lacks the supported feature/classifier API.
        ValueError: If images, times or classifier projection shapes are incompatible.
    """

    outputs = network.predict_class(
        (images, times, tf.zeros_like(times)),
        max_encoder_num=None, full_return=True, training=False,
    )
    features = network.classifier_feature_extractor(outputs[2][-1], training=False)
    # Semantic-only phases detach shared features before the eligible classifier projection.
    if stop_backbone:
        features = tf.stop_gradient(features)
    for layer in network.classifier.layers[:-1]:
        features = layer(features, training=False)
    probabilities = network.classifier.layers[-1](features, training=False)
    return tf.cast(features, tf.float32), tf.cast(probabilities, tf.float32)


def paired_view(
    wrapper: tf.keras.Model, images: tf.Tensor, level: int, seed: int
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Produce one view used unchanged by both student and frozen target.

    Route level 0 means an exactly clean image, even for schedules with a noisy
    timestep zero. Positive levels use the platform's forward diffusion API.

    Args:
        wrapper (tf.keras.Model): Live diffusion classifier exposing its raw network, class
            mapping, schedules and existing training or inference APIs.
        images (tf.Tensor): Numeric sample-major images in the configured model-input scale,
            normally float32 NHWC values in [-1, 1].
        level (int): Nonnegative integer noise level; zero means exactly clean and positive
            values index the diffusion schedule.
        seed (int): Explicit integer random seed; local or derived streams preserve
            reproducibility without reseeding caller-owned generators.

    Returns:
        view (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): (images, times, alpha_bar):
            tf.float32 images, tf.int32 times and scalar tf.float32 reliability input; clean
            level zero returns signal power one.

    Raises:
        ValueError: If image geometry or noising controls are invalid.
        tf.errors.InvalidArgumentError: If level is outside the diffusion schedule.
    """

    # Route level zero denotes an exactly clean view even when schedule index zero is noisy.
    if level == 0:
        noised, _, times = wrapper.noisify(
            images, min_timesteps=0, max_timesteps=0, seed=seed
        )
        return noised, times, tf.constant(1., tf.float32)
    times = tf.fill((tf.shape(images)[0],), tf.cast(level, tf.int32))
    noised, _, times = wrapper.noisify(images, t=times, seed=seed)
    return noised, times, tf.cast(wrapper.schedules["alpha_bar"][level], tf.float32)


class RoutePhase(tf.keras.Model):
    """One explicit phase, fitted through common.train.train_model.

    Input batches are step IDs from common.dataloader.get_dataset. Each step
    draws a class-balanced batch from the same finite current/replay pool used
    by joint fitting. This eager sampler supports uneven class counts without
    padding examples into the contrastive denominator.
    """

    def __init__(
        self, wrapper: tf.keras.Model, bank: object, pool: object,
        settings: object, phase: str, classes: list[int], seed: int,
        target: tf.keras.Model | None = None,
        frozen_bank: dict | None = None,
    ) -> None:
        """Create one isolated phase with its own sampler, counters, and temporary predictor.

        Args:
            wrapper (tf.keras.Model): Live diffusion classifier exposing its raw network, class
                mapping, schedules and existing training or inference APIs.
            bank (object): ModulationBank containing the class-specific float32 gain and bias
                variables.
            pool (object): ClassBalancedPool of the finite current/replay observations supplied
                to joint fitting.
            settings (object): Validated settings instance for this component; its fields select
                the behavior described above.
            phase (str): acquisition or consolidation, supplied by the route controller.
            classes (list[int]): Nonempty list of dense focus-class IDs. Acquisition receives
                new classes; consolidation receives all retained gates.
            seed (int): Explicit integer random seed; local or derived streams preserve
                reproducibility without reseeding caller-owned generators.
            target (tf.keras.Model | None): Independent frozen raw network used only to
                construct consolidation targets; None is valid only without target alignment.
            frozen_bank (dict | None): Mapping from dense class IDs to detached float32
                gain/bias tensors [D].

        Returns:
            initialized (None): None; creates independent counters/sampler and, for
                consolidation, an identity-initialized float32 predictor.

        Raises:
            ValueError: If seed or projection dimensions cannot initialize the phase.
        """
        super().__init__(name=f"route_{phase}", dtype="float32")
        self.wrapper = wrapper
        self.bank = bank
        self.pool = pool
        self.settings = settings
        self.phase = phase
        self.classes = classes
        self.phase_seed = seed
        self.rng = np.random.default_rng(seed)
        self.target = target
        self.frozen_bank = frozen_bank
        self.step_number = 0
        self.focus_cycle: list[int] = []
        self.focus_counts = {class_id: 0 for class_id in classes}
        self.updated_names: set[str] = set()
        self.example_draws = 0
        self.view_draws = 0
        self.trace: list[dict[str, float]] = []
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.ce_tracker = tf.keras.metrics.Mean(name="ce")
        self.semantic_tracker = tf.keras.metrics.Mean(name="semantic_loss")
        self.predictor = None
        if phase == "consolidation":
            # A fresh predictor per increment is training-only state.
            self.predictor = tf.keras.layers.Dense(
                bank.dimension, use_bias=False,
                kernel_initializer=tf.keras.initializers.Identity(),
                dtype="float32", name="consolidation_predictor",
            )
            self.predictor(tf.zeros((1, bank.dimension), dtype=tf.float32))

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        """Keras-owned mean trackers, reset at each explicit fit.

        Returns:
            metrics (list[tf.keras.metrics.Metric]): List of Keras Mean trackers for loss, CE
                and semantic loss, reset by the normal fit lifecycle.

        Raises:
            None: The trackers are created during initialization.
        """

        return [self.loss_tracker, self.ce_tracker, self.semantic_tracker]

    def fit(self, x: tf.data.Dataset, **kwargs: object) -> tf.keras.callbacks.History:
        """Fit the existing phase objective, optionally resuming committed updates.

        Args:
            x (tf.data.Dataset): Finite integer step-ID batches; the local seeded
                sampler supplies the actual float32 image pool.
            kwargs (object): Existing Keras epoch, validation and callback controls.

        Returns:
            history (tf.keras.callbacks.History): Full phase history, including
                committed updates from an interrupted attempt when enabled.

        Raises:
            ValueError: If saved sampler, variable or iterator state is incompatible.
            Exception: Propagates existing objective, optimizer or checkpoint errors.
        """
        recovery = getattr(self.wrapper, "fit_checkpoint", None)
        # Standalone phases and ordinary runs retain the original Keras fit.
        if recovery is None:
            return super().fit(x, **kwargs)
        return recovery.fit(self, x, kwargs)

    def train_step(self, data: object) -> dict[str, tf.Tensor]:
        """Apply one phase-specific update with an explicit variable allowlist.

        Args:
            data (object): Keras step-ID batch used to drive one eager optimizer update; actual
                images are drawn from the phase pool.

        Returns:
            metrics (dict[str, tf.Tensor]): Dict of scalar float32 running means; commits one
                optimizer update and appends Python scalar trace/resource fields.

        Raises:
            ValueError: If the selected class cannot form a valid batch or required phase state
                is absent.
            tf.errors.InvalidArgumentError: If objective values, gradients or feature shapes are
                invalid.
        """

        # Modern optimizers must see every eligible gate before a focus changes.
        # The gradient allowlist below still updates only the selected class.
        if self.phase == "acquisition" and self.step_number == 0:
            build_optimizer = getattr(self.optimizer, "build", None)
            # Keras 3 registers all later gate variables before the first application.
            if callable(build_optimizer):
                build_optimizer([variable for class_id in self.classes
                                 for variable in self.bank.vectors[class_id]])
        # Refresh the shuffled focus cycle only after every class has received its turn.
        if not self.focus_cycle:
            self.focus_cycle = self.rng.permutation(self.classes).tolist()
        focus = int(self.focus_cycle.pop())
        self.focus_counts[focus] += 1
        images, labels, positive = self.pool.draw(focus, self.settings.batch_size, self.rng)
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        semantic_losses = []
        ce_losses = []
        with tf.GradientTape() as tape:
            levels = (self.settings.acquisition_noise_level,) if self.phase == "acquisition" else (
                self.settings.noise_levels
            )
            if self.phase == "consolidation":
                # Keep supervised augmentation fixed while varying semantic noise bands.
                ce_images, ce_times, _ = paired_view(
                    self.wrapper, images, self.settings.ce_noise_level,
                    derive_seed(self.phase_seed, self.step_number, "ce_noise"),
                )
                _, ce_probabilities = semantic_features(
                    self.wrapper.network, ce_images, ce_times,
                    stop_backbone=self.settings.consolidation_scope == "semantic",
                )
                ce_losses.append(tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(labels, ce_probabilities)
                ))
            for draw, level in enumerate(levels):
                noised, times, alpha_bar = paired_view(
                    self.wrapper, images, level,
                    derive_seed(self.phase_seed, self.step_number, draw, "noise"),
                )
                # Acquisition trains the gate against an immutable feature backbone.
                if self.phase == "acquisition":
                    features, _ = semantic_features(self.wrapper.network, noised, times)
                    features = tf.stop_gradient(features)
                    if self.settings.acquisition_objective == "true_class_ce":
                        # Deliberately shortcut-prone control: labels choose each row's gate.
                        rows = []
                        for row, label in enumerate(labels.numpy()):
                            rows.append(self.bank.apply(features[row:row + 1], int(label)))
                        modulated = tf.concat(rows, axis=0)
                        probabilities = self.wrapper.network.classifier.layers[-1](
                            modulated, training=False
                        )
                        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                            labels, probabilities
                        ))
                    # The selected-class separation loss applies one gate to positives and negatives
                    # alike.
                    else:
                        loss = modulation_separation_loss(
                            self.bank.apply(features, focus), positive,
                            orthogonality_weight=self.settings.orthogonality_weight,
                        )
                    semantic_losses.append(loss)
                    ce_losses.append(tf.constant(0., tf.float32))
                # Consolidation learns the unmodulated projection against an independent frozen target.
                else:
                    features, _ = semantic_features(
                        self.wrapper.network, noised, times,
                        stop_backbone=self.settings.consolidation_scope == "semantic",
                    )
                    target_features, _ = semantic_features(self.target, noised, times)
                    # Only the explicitly unmodulated control removes target modulation.
                    if self.settings.condition != "unmodulated_feature_distillation":
                        gain, bias = self.frozen_bank[focus]
                        target_features = affine_modulation(
                            target_features, gain, bias, self.settings
                        )
                    target_features = tf.stop_gradient(target_features)
                    predicted = self.predictor(features, training=True)
                    weight = reliability_weights(
                        alpha_bar, floor=self.settings.reliability_floor
                    ) if self.settings.reliability == "alpha_bar" else tf.constant(1., tf.float32)
                    row_weights = tf.fill((tf.shape(images)[0],), weight)
                    # Feature-distillation controls use pointwise normalized MSE instead of negatives.
                    if self.settings.condition in (
                        "feature_distillation", "unmodulated_feature_distillation"
                    ):
                        semantic = normalized_feature_distillation_loss(
                            predicted, target_features, row_weights=row_weights
                        )
                    # The main consolidation objective treats matched target rows as the InfoNCE
                    # positives.
                    else:
                        semantic = contrastive_alignment_loss(
                            predicted, target_features,
                            temperature=self.settings.temperature, row_weights=row_weights,
                        )
                    semantic_losses.append(semantic)
            semantic_loss = tf.reduce_mean(tf.stack(semantic_losses))
            ce_loss = tf.reduce_mean(tf.stack(ce_losses))
            # Acquisition uses only the class-separation objective.
            if self.phase == "acquisition":
                loss = semantic_loss
            # Consolidation combines fixed CE with the declared semantic alignment coefficient.
            else:
                alignment_weight = 0. if self.settings.condition == "no_consolidation" else (
                    self.settings.alignment_weight
                )
                loss = self.settings.ce_weight * ce_loss + alignment_weight * semantic_loss
            tf.debugging.assert_all_finite(loss, "Nonfinite route objective.")

        # Only eligible new-class gate variables may change during acquisition.
        if self.phase == "acquisition":
            selected = [focus] if self.settings.acquisition_objective == "contrastive" else self.classes
            variables = [v for class_id in selected for v in self.bank.vectors[class_id]]
        # Consolidation uses the declared network scope and the temporary predictor.
        else:
            variables = list(self.wrapper.network.classifier.trainable_variables) if (
                self.settings.consolidation_scope == "semantic"
            ) else list(self.wrapper.network.trainable_variables)
            variables += self.predictor.trainable_variables
        pairs = apply_policy_gradients(tape, self.optimizer, loss, variables)
        self.updated_names.update(format_variable_name(variable) for _, variable in pairs)
        self.example_draws += len(images)
        self.view_draws += len(images) * len(levels)
        self.step_number += 1
        self.loss_tracker.update_state(loss)
        self.ce_tracker.update_state(ce_loss)
        self.semantic_tracker.update_state(semantic_loss)
        self.trace.append({
            "step": self.step_number, "focus_class": focus,
            "examples": len(images), "loss": float(loss.numpy()),
            "ce": float(ce_loss.numpy()), "semantic_loss": float(semantic_loss.numpy()),
        })
        return {metric.name: metric.result() for metric in self.metrics}
