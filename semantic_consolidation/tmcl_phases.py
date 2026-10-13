"""Current-image TMCL phases using the native TensorFlow DiT feature path.

Acquisition updates only the selected new-class modulation with one-versus-rest
OPL. Pretraining uses view invariance (VI). Consolidation uses VI plus modulation
invariance (MI): the first view is unmodulated and differentiable, while the
remaining views use independently sampled retained gates and stopped CURRENT
encoder/projector outputs. Neither phase invokes classifier CE or diffusion loss.

The protocol follows tmcl/main_tmcl.py at upstream commit
2cb306cc5cbe6d13a3dd4991998ee3e7c2948c8b. Native TensorFlow image augmentation,
initializers and image sampling are declared adaptations, not bitwise PyTorch
replay. Acquisition follows paper Equation 8: iid uniform focus tasks and a
Bernoulli(.5) positive count with uniform current positive/negative row draws.
Undefined tiny batches are resampled, counted, and fail after 1024 rejections.
Optional state invariance compares current predictions with the frozen previous
session's unmodulated encoder and VI projector, before that session's head reset.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from common.gradients import apply_policy_gradients
from common.keras_compat import format_variable_name
from common.runtime import derive_seed
from semantic_consolidation.augmentation import acquisition_augmentation, consolidation_views
from semantic_consolidation.objectives import (
    multiview_barlow_loss, orthogonal_projection_loss, state_invariance_loss
)


@tf.keras.utils.register_keras_serializable(package="SemanticConsolidation")
class TMCLBatchNormalization(tf.keras.layers.BatchNormalization):
    """Native BN forward with PyTorch's unbiased running-variance update.

    PyTorch normalizes training outputs using population variance, as Keras
    does, but updates its evaluation variance with the unbiased sample estimate.
    The heads use two-dimensional unmasked batches, fixed EMA momentum .9
    (PyTorch update coefficient .1), and float32. No cumulative-average counter
    is required by that fixed-momentum protocol.
    """

    def call(self, inputs: tf.Tensor, mask: tf.Tensor | None = None, 
             training: bool | None = None) -> tf.Tensor:
        """Preserve native differentiable normalization and correct only its EMA state."""

        # Projection heads have no temporal mask or sequence-weighted BN semantics.
        if mask is not None:
            raise ValueError("TMCL projection BN supports unmasked matrix inputs only.")
        updating = training is True and self.trainable
        # Only a training call updates the running-variance estimate.
        if updating:
            tf.debugging.assert_rank(inputs, 2)
            count = tf.cast(tf.shape(inputs)[0], tf.float32)
            tf.debugging.assert_greater(count, 1., message="TMCL BN requires at least two training rows.")
            previous_variance = tf.identity(self.moving_variance)
        outputs = super().call(inputs, mask=mask, training=training)
        # Inference and frozen previous-session heads leave every state variable unchanged.
        if updating:
            _, variance = tf.nn.moments(tf.cast(inputs, tf.float32), axes=[0])
            unbiased = variance * count / (count - 1.)
            self.moving_variance.assign(self.momentum * previous_variance + (1. - self.momentum) * unbiased)
        return outputs


def _linear(units: int, input_dim: int, use_bias: bool, seed: int, name: str) -> tf.keras.layers.Dense:
    """Match PyTorch Linear's initialization distribution using native seeded RNG."""

    bound = float(1. / np.sqrt(input_dim))
    return tf.keras.layers.Dense(
        units, use_bias=use_bias, 
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-bound, maxval=bound, seed=derive_seed(seed, "kernel")
        ), 
        bias_initializer=tf.keras.initializers.RandomUniform(
            minval=-bound, maxval=bound, seed=derive_seed(seed, "bias")
        ), dtype="float32", name=name
    )


def _projector(dimension: int, hidden: int, output: int, seed: int, name: str) -> tf.keras.Model:
    """Build a three-layer Barlow projector with two BN/ReLU hidden stages.

    BatchNorm momentum .9 matches the released PyTorch update coefficient .1.
    Lightly's final linear has a bias; hidden linears followed by BN do not.
    Initializer distributions match PyTorch Linear, with TensorFlow RNG streams.
    """

    layers = [tf.keras.layers.InputLayer(shape=tuple([dimension]), dtype="float32")]
    for index, width in enumerate((hidden, hidden, output)):
        layers.append(_linear(
            width, dimension if index == 0 else hidden, use_bias=index == 2, 
            seed=derive_seed(seed, index, "dense"), name=f"linear_{index}"
        ))
        # Only the two hidden layers have normalization and activation.
        if index < 2:
            layers.append(TMCLBatchNormalization(
                momentum=0.9, epsilon=1e-5, dtype="float32", name=f"bn_{index}"
            ))
            layers.append(tf.keras.layers.ReLU(dtype="float32", name=f"relu_{index}"))
    return tf.keras.Sequential(layers, name=name)


def _predictor(dimension: int, hidden: int, seed: int, name: str = "tmcl_predictor") -> tf.keras.Model:
    """Build the released MI first-view predictor, including train-mode BN."""

    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=tuple([dimension]), dtype="float32"), 
        _linear(
            hidden, dimension, use_bias=True, seed=derive_seed(seed, "hidden"), name="hidden"
        ), 
        TMCLBatchNormalization(momentum=0.9, epsilon=1e-5, dtype="float32"), 
        tf.keras.layers.ReLU(dtype="float32"), 
        _linear(
            dimension, hidden, use_bias=True, seed=derive_seed(seed, "output"), name="output"
        )
    ], name=name)


def build_tmcl_heads(feature_dim: int, settings: object, seed: int) -> dict[str, tf.keras.Model]:
    """Create fully built independent VI/MI projectors and first-view predictor.

    The caller owns session-level reset and native checkpoint restoration.
    Building without training calls leaves BN running statistics untouched.
    """

    heads = {
        "vi": _projector(
            feature_dim, settings.projector_hidden, settings.projector_dim, 
            seed=derive_seed(seed, "vi"), name="tmcl_vi_projector"
        ), 
        "mi": _projector(
            feature_dim, settings.projector_hidden, settings.projector_dim, 
            seed=derive_seed(seed, "mi"), name="tmcl_mi_projector"
        ), 
        "predictor": _predictor(
            settings.projector_dim, settings.predictor_hidden, seed=derive_seed(seed, "predictor")
        )
    }
    # SI has its own predictor and optimizer because it begins only after session one.
    if settings.si_weight != 0:
        heads["si_predictor"] = _predictor(
            settings.projector_dim, settings.predictor_hidden, 
            seed=derive_seed(seed, "si_predictor"), name="tmcl_si_predictor"
        )
    return heads


def modulation_decay_vector(bank: object, start: float, end: float) -> tf.Tensor:
    """Expand the released block-depth cosine decay to every gate coordinate.

    Sites must use the adapter's ordered ``block_N/role`` identifiers. All
    attention/MLP projections in a block share its coefficient; both gain and
    bias vectors use the resulting mask. Lightly's length-one cosine returns
    the end value rather than the start value.
    """

    keys = [str(site["key"]).split("/")[0] for site in bank.sites]
    # A checkpointed native block map is required; generic layer names are ambiguous.
    if not keys or any(not key.startswith("block_") for key in keys):
        raise ValueError("TMCL gate decay requires explicit ordered block_N site identities.")
    blocks = [int(key[len("block_"):]) for key in keys]
    distinct = list(dict.fromkeys(blocks))
    # Noncontiguous or reordered block IDs would silently change the depth schedule.
    if distinct != list(range(len(distinct))):
        raise ValueError("TMCL gate decay requires contiguous ordered block IDs.")
    coefficients = []
    for block, site in zip(blocks, bank.sites):
        # Lightly's special case for one layer reaches the ending coefficient.
        if len(distinct) == 1:
            value = end
        # Multi-block schedules include both declared depth endpoints exactly.
        else:
            value = end + 0.5 * (start - end) * (1. + np.cos(np.pi * block / (len(distinct) - 1)))
        coefficients.append(np.full(int(np.prod(site["shape"])), value, dtype=np.float32))
    return tf.convert_to_tensor(np.concatenate(coefficients), dtype=tf.float32)


class TMCLPhase(tf.keras.Model):
    """One eager phase driven by step IDs through ``common.train.train_model``.

    ``images`` must contain only current-session clean images in native model
    coordinates. Labels are used only for acquisition. Consolidation samples
    images independently of all retained class IDs, so old gates need no old
    images or replay. The owning trainer validates current-data provenance.

    ``heads`` has keys ``vi``, ``mi``, ``predictor`` and optional ``si_predictor``.
    It survives initial pretraining into the same session's consolidation;
    session-level resets and checkpointing belong to the trainer.
    The trainer assigns persistent ``optimizer`` (backbone plus VI),
    ``mi_optimizer`` (MI plus predictor), ``si_optimizer`` (SI predictor) and
    per-class ``gate_optimizers``. ``previous`` contains an independent frozen
    ``adapter`` and ``projector`` from the preceding completed session.
    A directly compiled phase may use one optimizer for bounded fixtures; that
    fallback is not the released Adam-counter lifecycle across phases.
    Only stateless image augmentation is cached as a TensorFlow graph. Native
    feature hooks, sampling, gradients and optimizer updates remain eager.
    """

    def __init__(
        self, adapter: object, images: np.ndarray, labels: np.ndarray | None, 
        classes: list[int], settings: object, phase: str, seed: int, 
        heads: dict[str, tf.keras.Model] | None = None, previous: dict[str, object] | None = None
    ) -> None:
        """Bind one current cohort and phase-specific trainable allowlist.

        Args:
            adapter: Native clean-feature adapter with a retained modulation bank.
            images: Current clean NHWC RGB images; no historical replay rows.
            labels: Aligned sparse class IDs for acquisition; otherwise ignored.
            classes: New acquisition classes, or all retained MI gate IDs.
            settings: TMCLSettings containing objective, batch and head controls.
            phase: ``pretrain``, ``acquisition`` or ``consolidation``.
            seed: Explicit independent sampler/augmentation/head seed.
            heads: Optional existing projector mapping from the same session.
            previous: Frozen previous-session adapter and VI projector, or None initially.

        Raises:
            ValueError: For incompatible cohort, phase, gate or head structure.
        """

        super().__init__(dtype="float32", name=f"tmcl_{phase}")
        # Unsupported objectives must fail before sampling or parameter updates.
        if phase not in ("pretrain", "acquisition", "consolidation"):
            raise ValueError("Supported phases are VI pretrain, OPL acquisition and VI/MI/SI consolidation.")
        self.adapter = adapter
        self.settings = settings
        self.phase = phase
        self.phase_seed = seed
        self.previous = previous
        self.use_si = phase == "consolidation" and settings.si_weight != 0 and previous is not None
        # A previous-state target cannot alias the student or expose an incomplete snapshot.
        if previous is not None:
            # Both independent previous models must be frozen before entering this phase.
            if (set(previous) != {"adapter", "projector"}
                    or previous["adapter"].network is adapter.network
                    or previous["adapter"].network.trainable or previous["projector"].trainable):
                raise ValueError("SI requires an independent frozen previous encoder and VI projector.")
        self.images = np.asarray(images, dtype=np.float32)
        self.labels = None if labels is None else np.asarray(labels)
        self.classes = sorted(set(int(class_id) for class_id in classes))
        # Images must define a nontrivial clean RGB correlation cohort.
        if self.images.ndim != 4 or self.images.shape[-1] != 3 or len(self.images) < 2:
            raise ValueError("TMCL needs at least two current NHWC RGB images.")
        # Class sampling cannot silently omit a requested retained gate.
        if not set(self.classes).issubset(set(adapter.bank.classes)):
            raise ValueError("Every requested TMCL class must have a retained modulation.")
        self.positive_rows: dict[int, np.ndarray] = {}
        self.negative_rows: dict[int, np.ndarray] = {}
        # Only acquisition is allowed to interpret supervised labels.
        if phase == "acquisition":
            # One-versus-rest OPL requires aligned integral labels and both populations.
            if self.labels is None or self.labels.shape != tuple([len(self.images)]) or not self.classes:
                raise ValueError("Acquisition requires aligned labels and at least one new class.")
            # Casting nonintegral labels would change the class interpretation.
            if not np.issubdtype(self.labels.dtype, np.integer):
                raise ValueError("Acquisition labels must be integer class IDs.")
            # At least two positive rows per update define off-diagonal OPL.
            if settings.batch_size < 3:
                raise ValueError("Acquisition requires a batch containing two positives and a negative.")
            for class_id in self.classes:
                self.positive_rows[class_id] = np.flatnonzero(self.labels == class_id)
                self.negative_rows[class_id] = np.flatnonzero(self.labels != class_id)
                # The current labelled cohort must support the declared one-versus-rest task.
                if not len(self.positive_rows[class_id]) or not len(self.negative_rows[class_id]):
                    raise ValueError("Each acquisition class needs current positive and negative images.")
        # MI needs retained modulations but never their historical image cohorts.
        elif phase == "consolidation" and settings.mi_weight != 0 and not self.classes:
            raise ValueError("Modulation invariance requires at least one retained gate.")
        # Initial training has no MI branch to rescue a zero VI objective.
        if phase == "pretrain" and settings.vi_weight == 0:
            raise ValueError("VI pretraining requires nonzero vi_weight.")
        self.rng = np.random.default_rng(seed)
        self.step_number = 0
        self.example_draws = 0
        self.view_draws = 0
        self.rejected_acquisition_batches = 0
        self.focus_counts = {class_id: 0 for class_id in self.classes}
        self.updated_names: set[str] = set()
        self.trace: list[dict[str, object]] = []
        self.mi_optimizer: tf.keras.optimizers.Optimizer | None = None
        self.si_optimizer: tf.keras.optimizers.Optimizer | None = None
        self.gate_optimizers: dict[int, tf.keras.optimizers.Optimizer] = {}
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.vi_tracker = tf.keras.metrics.Mean(name="vi")
        self.mi_tracker = tf.keras.metrics.Mean(name="mi")
        self.si_tracker = tf.keras.metrics.Mean(name="si")
        self.opl_tracker = tf.keras.metrics.Mean(name="opl")
        self.gate_decay = None
        # The released acquisition optimizer uses depth-specific decoupled gate decay.
        if phase == "acquisition":
            self.gate_decay = modulation_decay_vector(
                adapter.bank, settings.modulation_decay_start, settings.modulation_decay_end
            )
        self.heads = {} if heads is None else heads
        # Acquisition has no projector objective or projector state update.
        if phase != "acquisition":
            feature = adapter.features(tf.convert_to_tensor(self.images[:2]), training=False)
            dimension = int(feature.shape[-1])
            # A missing mapping creates exactly one independent VI/MI head set.
            if heads is None:
                self.heads = build_tmcl_heads(dimension, settings, seed=seed)
            # Reused heads must represent the same native feature/projection geometry.
            expected_heads = {"vi", "mi", "predictor"} | ({"si_predictor"} if settings.si_weight != 0 else set())
            # Optional SI extends the complete head identity rather than accepting arbitrary extras.
            if set(self.heads) != expected_heads:
                raise ValueError("TMCL heads must match the configured VI/MI/SI models.")
            # Previous VI state must be independent from the current trainable head.
            if previous is not None and previous["projector"] is self.heads["vi"]:
                raise ValueError("SI previous projector cannot alias the current VI projector.")
            for key in ("vi", "mi"):
                projected = self.heads[key](feature, training=False)
                # The shared MI predictor requires identical declared output dimensions.
                if projected.shape[-1] != settings.projector_dim:
                    raise ValueError("TMCL projector output does not match settings.")
            self.heads["predictor"](projected, training=False)
            # Build/check the optional SI predictor without altering its BN state.
            if settings.si_weight != 0:
                self.heads["si_predictor"](projected, training=False)
        self._augmentation = self._build_augmentation()

    def _build_augmentation(self) -> object:
        """Cache the unchanged image policy with one dynamic tensor seed input.

        Fixed geometry, wrapper coordinates and view count belong to this phase.
        The batch dimension remains dynamic, and scalar int64 seeds avoid one
        retrace per optimizer step. Tracing is lazy so policy test doubles can
        be installed before first use. No native feature adapter is traced.
        """

        wrapper = self.adapter.wrapper
        num_views = self.settings.augmentation_views
        acquisition = self.phase == "acquisition"
        signature = [
            tf.TensorSpec([None, *self.images.shape[1:]], tf.float32), 
            tf.TensorSpec([], tf.int64)
        ]

        @tf.function(input_signature=signature, autograph=False, jit_compile=False)
        def augment(images: tf.Tensor, seed: tf.Tensor) -> tuple[tf.Tensor, ...]:
            """Apply the captured stateless policy without entering the model."""

            # Acquisition retains its original flip-only augmentation stream.
            if acquisition:
                return tuple([acquisition_augmentation(images, seed=seed)])
            return consolidation_views(
                images, wrapper=wrapper, num_views=num_views, image_size=None, seed=seed
            )

        return augment

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        """Expose objective means to the normal Keras fit/callback lifecycle."""

        return [self.loss_tracker, self.vi_tracker, self.mi_tracker, self.si_tracker, self.opl_tracker]

    def _draw(self) -> tuple[tf.Tensor, int | None, int]:
        """Draw current rows without consulting historical gate identities."""

        batch_size = self.settings.batch_size
        # Acquisition follows the paper's iid focus and Bernoulli half-positive mixture.
        if self.phase == "acquisition":
            focus = int(self.rng.choice(self.classes))
            for _ in range(1024):
                positive_count = int(self.rng.binomial(batch_size, 0.5))
                # OPL needs an off-diagonal positive pair and at least one negative.
                if 2 <= positive_count < batch_size:
                    break
                self.rejected_acquisition_batches += 1
            # Never substitute a different mixture when bounded structural sampling fails.
            else:
                raise RuntimeError("TMCL could not draw a structurally valid Bernoulli OPL batch.")
            positive = self.rng.choice(self.positive_rows[focus], size=positive_count, replace=True)
            negative = self.rng.choice(
                self.negative_rows[focus], size=batch_size - positive_count, replace=True
            )
            rows = np.concatenate((positive, negative))
            self.focus_counts[focus] += 1
            return tf.convert_to_tensor(self.images[rows]), focus, positive_count
        rows = self.rng.choice(len(self.images), size=batch_size, replace=True)
        return tf.convert_to_tensor(self.images[rows]), None, 0

    def _variables(self, focus: int | None) -> list[tf.Variable]:
        """Select gates for OPL or native feedforward/head parameters for VI/MI."""

        # Only the active new-class gate receives an acquisition gradient.
        if self.phase == "acquisition":
            return list(self.adapter.bank.variables_for([focus]))
        variables = list(self.adapter.backbone_variables)
        # VI projector variables are selected only when VI contributes to the objective.
        if self.settings.vi_weight != 0 or self.use_si:
            variables += list(self.heads["vi"].trainable_variables)
        # Pretraining has neither MI nor MI-head optimizer state updates.
        if self.phase == "consolidation" and self.settings.mi_weight != 0:
            variables += list(self.heads["mi"].trainable_variables)
            variables += list(self.heads["predictor"].trainable_variables)
        # SI predictor is untouched until a completed previous-state target exists.
        if self.use_si:
            variables += list(self.heads["si_predictor"].trainable_variables)
        return variables

    def _mi_variables(self) -> list[tf.Variable]:
        """Select the MI group whose Adam counter begins at its first MI update."""

        return list(self.heads["mi"].trainable_variables) + list(self.heads["predictor"].trainable_variables)

    def _barlow(self, views: tuple[tf.Tensor, ...]) -> tf.Tensor:
        """Apply the exact released multiview reduction with configured scaling."""

        return multiview_barlow_loss(
            views, off_diagonal_weight=self.settings.redundancy_weight, 
            scale=self.settings.barlow_scale
        )

    def train_step(self, data: object) -> dict[str, tf.Tensor]:
        """Update one declared phase without consuming labels in VI/MI.

        ``data`` is an ignored step-ID batch. Sampler state and counters are
        explicit; this initial implementation checkpoints only session boundaries.
        Stopped MI views retain training-mode BN updates as in upstream no_grad.
        """

        # The scoped native modulation adapter and sampler require eager execution.
        if not tf.executing_eagerly():
            raise RuntimeError("TMCLPhase must be compiled with run_eagerly=True.")
        images, focus, positive_count = self._draw()
        augmentation_seed = derive_seed(self.phase_seed, self.step_number, "augmentation")
        views = self._augmentation(images, tf.convert_to_tensor(augmentation_seed, dtype=tf.int64))
        variables = self._variables(focus)
        optimizer = self.gate_optimizers[focus] if self.phase == "acquisition" and self.gate_optimizers else self.optimizer
        separate_mi = self.phase == "consolidation" and self.settings.mi_weight != 0 and self.mi_optimizer is not None
        separate_si = self.use_si and self.si_optimizer is not None
        mi_variables = self._mi_variables() if separate_mi else []
        si_variables = list(self.heads["si_predictor"].trainable_variables) if separate_si else []
        separate_ids = {id(variable) for variable in mi_variables + si_variables}
        primary_variables = [variable for variable in variables if id(variable) not in separate_ids]
        # Keras optimizers must register every possible new gate before the first focus update.
        if self.phase == "acquisition" and not optimizer.built:
            build_classes = [focus] if self.gate_optimizers else self.classes
            optimizer.build(list(self.adapter.bank.variables_for(build_classes)))
        vi_loss = tf.constant(0., tf.float32)
        mi_loss = tf.constant(0., tf.float32)
        si_loss = tf.constant(0., tf.float32)
        opl_loss = tf.constant(0., tf.float32)
        gate_counts: dict[int, int] = {}
        with tf.GradientTape(persistent=separate_mi or separate_si, watch_accessed_variables=False) as tape:
            tape.watch([variable if isinstance(variable, tf.Variable) else variable.value for variable in variables])
            # Acquisition differentiates through the frozen encoder into its active internal gate.
            if self.phase == "acquisition":
                condition = tf.fill(tuple([len(images)]), tf.cast(focus, tf.int32))
                features = self.adapter.features(views[0], condition, training=False)
                opl_loss = orthogonal_projection_loss(features[:positive_count], features[positive_count:])
                loss = opl_loss
            # VI and MI share current image views while keeping their projectors independent.
            else:
                # VI sees every augmented image unmodulated in one training-mode batch.
                if self.settings.vi_weight != 0 or self.use_si:
                    features = self.adapter.features(tf.concat(views, axis=0), training=True)
                    projections = self.heads["vi"](features, training=True)
                    # The SI-only ablation still needs the current VI projection path.
                    if self.settings.vi_weight != 0:
                        vi_loss = self._barlow(tuple(tf.split(projections, len(views), axis=0)))
                # SI targets are frozen previous-session states, unlike current MI targets.
                if self.use_si:
                    predictions = self.heads["si_predictor"](projections, training=True)
                    with tape.stop_recording():
                        previous_features = self.previous["adapter"].features(tf.concat(views, axis=0), training=False)
                        previous_projections = tf.stop_gradient(self.previous["projector"](previous_features, training=False))
                    si_loss = state_invariance_loss(
                        tuple(tf.split(predictions, len(views), axis=0)), 
                        tuple(tf.split(previous_projections, len(views), axis=0)), 
                        method=self.settings.si_method, objective=self.settings.si_objective, 
                        off_diagonal_weight=self.settings.si_redundancy_weight, 
                        scale=self.settings.si_barlow_scale, negative_weight=self.settings.si_negative_weight
                    )
                # Only consolidation uses randomly retained modulations as stopped current targets.
                if self.phase == "consolidation" and self.settings.mi_weight != 0:
                    first_features = self.adapter.features(views[0], training=True)
                    first_projection = self.heads["mi"](first_features, training=True)
                    first_prediction = self.heads["predictor"](first_projection, training=True)
                    gate_ids = self.rng.choice(self.classes, size=len(images) * (len(views) - 1), replace=True)
                    gate_counts = {class_id: int(np.sum(gate_ids == class_id)) for class_id in self.classes}
                    # no_grad in the official implementation also avoids retaining target activations.
                    with tape.stop_recording():
                        target_features = self.adapter.features(
                            tf.concat(views[1:], axis=0), tf.convert_to_tensor(gate_ids, dtype=tf.int32), 
                            training=True
                        )
                        target_projection = tf.stop_gradient(self.heads["mi"](target_features, training=True))
                    targets = tuple(tf.split(target_projection, len(views) - 1, axis=0))
                    mi_loss = self._barlow(tuple([first_prediction]) + targets)
                loss = (self.settings.vi_weight * vi_loss + self.settings.mi_weight * mi_loss
                        + self.settings.si_weight * si_loss)
            tf.debugging.assert_all_finite(loss, "Nonfinite TMCL objective.")
        # Decay only the selected class; optimizer-level scalar decay must remain zero.
        if self.phase == "acquisition":
            learning_rate = tf.cast(optimizer.learning_rate, tf.float32)
            for variable in variables:
                # Released task_modulations.py stores residual gain; AdamW decays that parameter.
                variable.assign_sub(tf.cast(learning_rate * self.gate_decay, variable.dtype) * variable)
        pairs = apply_policy_gradients(tape, optimizer, loss, primary_variables)
        # Both derivatives refer to the same recorded forward, with independent Adam counters.
        if separate_mi:
            pairs += apply_policy_gradients(tape, self.mi_optimizer, loss, mi_variables)
        # SI has independent first-use bias correction beginning with its first target session.
        if separate_si:
            pairs += apply_policy_gradients(tape, self.si_optimizer, loss, si_variables)
        del tape
        self.updated_names.update(format_variable_name(variable) for _, variable in pairs)
        self.step_number += 1
        self.example_draws += len(images)
        self.view_draws += len(images) * len(views)
        for tracker, value in (
            (self.loss_tracker, loss), (self.vi_tracker, vi_loss), 
            (self.mi_tracker, mi_loss), (self.si_tracker, si_loss), (self.opl_tracker, opl_loss)
        ):
            tracker.update_state(value)
        self.trace.append({
            "step": self.step_number, "focus_class": focus, "examples": len(images), 
            "augmentation_seed": augmentation_seed, 
            "loss": float(loss.numpy()), "vi": float(vi_loss.numpy()), 
            "mi": float(mi_loss.numpy()), "si": float(si_loss.numpy()), "opl": float(opl_loss.numpy()), 
            "modulation_draw_counts": gate_counts, 
            "rejected_acquisition_batches": self.rejected_acquisition_batches
        })
        return {metric.name: metric.result() for metric in self.metrics}
