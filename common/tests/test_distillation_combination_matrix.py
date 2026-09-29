"""Constrained categorical matrices for wrapper losses and real architecture updates.

The analytic cases enumerate finite categorical factors, not arbitrary architectures,
continuous hyperparameters, datasets, or training trajectories. NumPy independently
computes CE, temperature-scaled KL, row normalization, role coefficients, and logit
gradients. Real-model cases exercise every standalone raw family with its compatible
wrapper(s), eager and graph updates, with and without an independent classifier head.
"""

from __future__ import annotations

import gc
import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from diffusion import (
    DiTClassifier, DiTDecoder, DiTEncoderDecoder, DiTEncoderDecoderClassifier, 
    DiffusionClassifier, DiffusionClassifierV2, DiffusionModel, 
    DiffusionTransformer, UNet, UNetClassifier
)


SCOPES = ("old_classes", "replay_only", "current_and_replay")
KINDS = ("hard", "soft")
WEIGHTS = (None, np.array([1., .25, 0., 2.]), np.zeros(4))
CLASSES = np.arange(4, dtype=np.int32)
REPLAY = np.array([True, False, True, False])
LOGITS = np.array([[.2, -.4, .8, .1], [-.3, .7, .1, -.2], 
                   [.4, -.2, -.5, .6], [-.1, .3, .9, -.6]], np.float64)


def _softmax(values: np.ndarray) -> np.ndarray:
    """Return stable last-axis float64 probabilities for a NumPy score matrix.

    Args:
        values (np.ndarray): Finite float64 scores [B,C].

    Returns:
        probabilities (np.ndarray): Float64 [B,C], each row summing to one.
    """

    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def _reference_term(
    teacher: np.ndarray, logits: np.ndarray, support: np.ndarray, 
    eligible: np.ndarray, weights: np.ndarray | None, kind: str, 
    temperature: float, coefficient: float = 1.
) -> tuple[float, np.ndarray]:
    """Independently evaluate one mapped teacher objective and its score gradient.

    Args:
        teacher (np.ndarray): Nonnegative teacher probabilities [B,K] with positive row mass.
        logits (np.ndarray): Float64 student scores [B,C] with K supported columns.
        support (np.ndarray): Int32 student column IDs [K], in teacher-column order.
        eligible (np.ndarray): Boolean row eligibility [B].
        weights (np.ndarray | None): Fractional row weights [B], or equal weights.
        kind (str): Hard CE or soft KL, respectively hard and soft.
        temperature (float): Softening temperature; hard CE ignores it.
        coefficient (float): Independent role multiplier, default one.

    Returns:
        result (tuple[float, np.ndarray]): Weighted scalar loss and exact [B,C]
            derivative with respect to student logits; empty exposure yields zeros.
    """

    row_weights = eligible.astype(np.float64)
    # Fractional classifier allocation intersects the role's eligibility.
    if weights is not None:
        row_weights *= weights
    denominator = np.sum(row_weights)
    # An empty objective has neither loss nor a student gradient.
    if denominator == 0.:
        return 0., np.zeros_like(logits)
    normalized = row_weights / denominator * coefficient
    target = teacher / teacher.sum(axis=-1, keepdims=True)
    # Hard distillation retains each teacher's own argmax before mapping columns.
    if kind == "hard":
        classes = support[np.argmax(target, axis=-1)]
        student = _softmax(logits)
        rows = -np.log(student[np.arange(len(student)), classes])
        gradient = student.copy()
        gradient[np.arange(len(student)), classes] -= 1.
    # Soft distillation keeps zero mass outside support and scales KL by T squared.
    else:
        softened = np.power(target, 1. / temperature)
        softened /= softened.sum(axis=-1, keepdims=True)
        student = _softmax(logits / temperature)
        entropy = np.zeros_like(softened)
        positive = softened > 0.
        entropy[positive] = softened[positive] * np.log(softened[positive])
        rows = np.sum(entropy - softened * np.log(student[:, support]), axis=-1)
        rows *= temperature ** 2
        padded = np.zeros_like(student)
        padded[:, support] = softened
        gradient = temperature * (student - padded)
    return float(np.sum(normalized * rows)), normalized[:, None] * gradient


class _LossHarness(tf.keras.Model):
    """Provide isolated wrapper state while retaining the real Keras compiled loss.

    This fixture borrows the production kernels unchanged; only architecture and
    runtime-teacher discovery are replaced by explicit metadata for analytic cases.
    """

    compute_clf_distil_loss = DiffusionClassifier.compute_clf_distil_loss
    compute_clf_distil_ctr_loss = DiffusionClassifier.compute_clf_distil_ctr_loss
    _compute_dual_classifier_distillation = DiffusionClassifier._compute_dual_classifier_distillation
    _classifier_teacher_support = DiffusionClassifier._classifier_teacher_support
    _classifier_teacher_mask = DiffusionClassifier._classifier_teacher_mask
    compute_ctr_loss = DiffusionModel.compute_ctr_loss
    compute_distil_noise_loss = DiffusionModel.compute_distil_noise_loss
    _compute_single_teacher_noise_loss = DiffusionModel._compute_single_teacher_noise_loss
    _compute_base_loss = DiffusionModel._compute_base_loss
    compute_noise_distil_image_kl_ctr_loss = DiffusionModel.compute_noise_distil_image_kl_ctr_loss

    def __init__(self, loss: str | tf.keras.losses.Loss = "mse") -> None:
        """Initialize float64 kernels and a native Keras scalar-reduction loss.

        Args:
            loss (str | tf.keras.losses.Loss): Compiled image/noise loss, default MSE.

        Returns:
            result (None): Independent loss configuration and role metadata are ready.
        """

        super().__init__(dtype="float64")
        self.compile(optimizer="sgd", loss=loss)
        self.scce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction="none", dtype="float64")
        self.clf_distil_type = "soft"
        self.clf_distil_temperature = 1.
        self.clf_distil_scope = "current_and_replay"
        self.dual_teacher_scope = "task"
        self.previous_teacher_clf_loss_weight = 1.7
        self.previous_teacher_noise_loss_weight = 1.7
        self.specifications = ()
        self.network = SimpleNamespace(num_classes=4, cls_token_regularizer_kwargs={})
        self.use_clf_ctr_loss = True
        self.use_clf_distil_ctr_loss = True
        self.show_separate_noise_losses = False
        self.kl_train_type = "cond"
        self.ctr_train_type = "cond"
        for role in ("previous", "current"):
            setattr(self, role + "_teacher_clf_loss_tracker", tf.keras.metrics.Mean(dtype="float64"))
            setattr(self, role + "_teacher_noise_distil_loss_tracker", tf.keras.metrics.Mean(dtype="float64"))

    def _classifier_teacher_specs(self) -> tuple[dict[str, object], ...]:
        """Return explicit independent role metadata for the current analytic case.

        Returns:
            specs (tuple[dict[str, object], ...]): Role, coefficient, and vocabulary mappings.
        """

        return self.specifications

    def _noise_teacher_specs(self) -> tuple[dict[str, object], ...]:
        """Return active role coefficients in the same order as supplied noise targets.

        Returns:
            specs (tuple[dict[str, object], ...]): Independent noise-teacher metadata.
        """

        return self.specifications


def _scope_rows(scope: str, width: int) -> np.ndarray:
    """Select single-teacher rows from actual labels or explicit replay provenance.

    Args:
        scope (str): One of SCOPES.
        width (int): Teacher's original leading class-vocabulary width.

    Returns:
        rows (np.ndarray): Boolean eligibility [4].
    """

    # Historical classes use actual targets, not dropped CFG conditions.
    if scope == "old_classes":
        return CLASSES < width
    # Replay provenance is independent of membership in the old vocabulary.
    if scope == "replay_only":
        return REPLAY.copy()
    return np.ones(4, bool)


def _raw_network(family: str, classifier: bool, classes: int, head: bool = False) -> tf.keras.Model:
    """Build a tiny real member of one supported standalone architecture family.

    Args:
        family (str): unet, transformer, encoder_decoder, or standalone decoder.
        classifier (bool): Include the family-specific classifier branch when true.
        classes (int): Fixed class vocabulary width.
        head (bool): Include an independent student distillation head when classifying.

    Returns:
        network (tf.keras.Model): Built float32 network for 4x4 single-channel images.
    """

    common = dict(num_classes=classes, use_cfg=True, timesteps=4, 
                  image_size=4, channels=1, seed=937)
    # Convolutional models use residual feature maps and a pooled classifier head.
    if family == "unet":
        options = dict(common, widths=[4], block_depth=1, bottleneck_depth=1, 
                       image_embedding_dim=2, time_embedding_dim=3, label_embedding_dim=2)
        # Only classifier networks accept the independent classifier-head switch.
        if classifier:
            return UNetClassifier(**options, classifier_only_distil_token=head)
        return UNet(**options)
    options = dict(common, patch_size=2, dim=4, depth=1, 
                   mha_num_heads=1, vit_block_mlp_ratio=1.)
    # Isolated decoders use their own condition and the wrapper supplies empty context.
    if family == "decoder":
        return DiTDecoder(**options, encoder_output_grid_size=2, encoder_output_dim=4, 
                          decoder_separate_cond=True, shift_inputs=False, use_causal_mask=False)
    # Transformer classifiers require explicit supported feature and query routes.
    if classifier:
        options.update(clf_mha_num_heads=1, clf_vit_block_mlp_ratio=1., 
                       feature_aggregation_ids_dict={1: [-1]}, clf_connection_ids_dict={-1: [-1]}, 
                       clf_distil_token_type="new_weight" if head else None)
    # Composite models own the required encoder context for their decoder.
    if family == "encoder_decoder":
        model_type = DiTEncoderDecoderClassifier if classifier else DiTEncoderDecoder
        return model_type(encoder_kwargs=options, 
                          decoder_kwargs={"depth": 1, "mha_num_heads": 1, 
                                          "vit_block_mlp_ratio": 1.})
    model_type = DiTClassifier if classifier else DiffusionTransformer
    return model_type(**options)


class DistillationCombinationMatrixTests(unittest.TestCase):
    """Assert independent numerical oracles and real optimizer/teacher invariants."""

    def tearDown(self) -> None:
        """Release graphs and restore the ordinary global numeric policy.

        Returns:
            result (None): No model or random stream is retained across test methods.
        """

        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.backend.clear_session()
        gc.collect()

    def test_single_teacher_loss_and_gradient_matrix(self) -> None:
        """Enumerate 324 type/scope/temperature/width/weight/execution combinations.

        Returns:
            result (None): Losses and student gradients match independent NumPy
                formulas, and every teacher target is detached from its tape.
        """

        model = _LossHarness()
        count = 0
        for kind, scope, temperature, width, mask_id, graph in itertools.product(
            KINDS, SCOPES, (.7, 1., 2.), (2, 4, 5), range(3), (False, True)
        ):
            with self.subTest(kind=kind, scope=scope, temperature=temperature, 
                              width=width, mask=mask_id, graph=graph):
                teacher = np.arange(1, 4 * width + 1, dtype=np.float64).reshape(4, width)
                teacher /= teacher.sum(axis=-1, keepdims=True)
                weights = WEIGHTS[mask_id]
                expected_loss, expected_gradient = _reference_term(
                    teacher[:, :4], LOGITS, np.arange(min(width, 4)), 
                    _scope_rows(scope, width), weights, kind, temperature, 1.7)

                def evaluate(target: tf.Tensor, scores: tf.Tensor) -> tuple[tf.Tensor | None, ...]:
                    """Differentiate one fixed categorical configuration in either execution mode.

                    Args:
                        target (tf.Tensor): Float64 teacher probabilities [4,width].
                        scores (tf.Tensor): Float64 student scores [4,4].

                    Returns:
                        values (tuple[tf.Tensor, ...]): Loss, student gradient, and absent teacher gradient.
                    """

                    with tf.GradientTape() as tape:
                        tape.watch((target, scores))
                        loss, _ = model.compute_clf_distil_loss(
                            target, tf.nn.softmax(scores), clf_distil_type=kind, 
                            clf_distil_temperature=temperature, clf_distil_scope=scope, 
                            classes=tf.constant(CLASSES), replay_mask=tf.constant(REPLAY), 
                            clf_distil_loss_mask=None if weights is None else tf.constant(weights), 
                            student_logits=scores)
                    student_gradient, teacher_gradient = tape.gradient(loss, (scores, target))
                    return loss, student_gradient, teacher_gradient

                execute = tf.function(evaluate) if graph else evaluate
                actual, gradient, teacher_gradient = execute(tf.constant(teacher), tf.constant(LOGITS))
                np.testing.assert_allclose(actual, expected_loss, atol=2e-10, rtol=2e-9)
                np.testing.assert_allclose(gradient, expected_gradient, atol=2e-10, rtol=2e-9)
                self.assertIsNone(teacher_gradient)
                count += 1
        self.assertEqual(count, 324)

    def test_dual_teacher_mapped_scope_matrix(self) -> None:
        """Enumerate 432 independent role/type/scope/temperature/mask/execution cases.

        Returns:
            result (None): Mapped role losses and full-denominator gradients equal
                their separately normalized sum; no teacher receives a gradient.
        """

        model = _LossHarness()
        previous = np.array([[.8, .2], [.3, .7], [.6, .4], [.1, .9]])
        current = np.array([[.4, .6], [.9, .1], [.2, .8], [.7, .3]])
        specifications = (
            dict(role="previous", weight=1.7, class_ids=None, task_class_ids=(0, 1)), 
            dict(role="current", weight=.6, class_ids=(3, 2), task_class_ids=(2, 3))
        )
        count = 0
        for kind, scope, temperature, dual_scope, roles, mask_id, graph in itertools.product(
            KINDS, SCOPES, (1., 2.), ("task", "all"), (tuple([0]), tuple([1]), (0, 1)), range(3), (False, True)
        ):
            with self.subTest(kind=kind, scope=scope, temperature=temperature, 
                              dual_scope=dual_scope, roles=roles, mask=mask_id, graph=graph):
                model.dual_teacher_scope = dual_scope
                model.specifications = tuple(specifications[index] for index in roles)
                weights = WEIGHTS[mask_id]
                expected_loss, expected_gradient = 0., np.zeros_like(LOGITS)
                for index in roles:
                    support = np.array((0, 1) if index == 0 else (3, 2))
                    selected_scope = scope if index == 0 else "current_and_replay"
                    rows = np.isin(CLASSES, support) if dual_scope == "task" or selected_scope == "old_classes" else np.ones(4, bool)
                    # Only the previous role retains historical replay-only restriction.
                    if selected_scope == "replay_only":
                        rows &= REPLAY
                    term, derivative = _reference_term(
                        (previous, current)[index], LOGITS, support, rows, weights, 
                        kind, temperature, (1.7, .6)[index])
                    expected_loss += term
                    expected_gradient += derivative

                def evaluate(first: tf.Tensor, second: tf.Tensor, scores: tf.Tensor) -> tuple[tf.Tensor | None, ...]:
                    """Evaluate active independent teacher roles without merging probability targets.

                    Args:
                        first (tf.Tensor): Float64 previous target [4,2].
                        second (tf.Tensor): Float64 current target [4,2].
                        scores (tf.Tensor): Float64 student scores [4,4].

                    Returns:
                        values (tuple[tf.Tensor, ...]): Loss, score gradient, and absent target gradients.
                    """

                    with tf.GradientTape() as tape:
                        tape.watch((first, second, scores))
                        targets = tuple((first, second)[index] for index in roles)
                        loss, _ = model.compute_clf_distil_loss(
                            targets, tf.nn.softmax(scores), clf_distil_type=kind, 
                            clf_distil_temperature=temperature, clf_distil_scope=scope, 
                            classes=tf.constant(CLASSES), replay_mask=tf.constant(REPLAY), 
                            clf_distil_loss_mask=None if weights is None else tf.constant(weights), 
                            student_logits=scores)
                    gradient, old_gradient, new_gradient = tape.gradient(loss, (scores, first, second))
                    return loss, gradient, old_gradient, new_gradient

                execute = tf.function(evaluate) if graph else evaluate
                actual, gradient, old_gradient, new_gradient = execute(
                    tf.constant(previous), tf.constant(current), tf.constant(LOGITS))
                np.testing.assert_allclose(actual, expected_loss, atol=2e-10, rtol=2e-9)
                np.testing.assert_allclose(gradient, expected_gradient, atol=2e-10, rtol=2e-9)
                self.assertIsNone(old_gradient)
                self.assertIsNone(new_gradient)
                count += 1
        self.assertEqual(count, 432)

    def test_auxiliary_target_mixture_matrix(self) -> None:
        """Enumerate 216 normal/distil/both token-target combinations with two heads.

        Returns:
            result (None): Each auxiliary loss equals CE/KD of the probability
                mixture, including fractional and empty scopes, in eager and graph mode.
        """

        model = _LossHarness()
        teacher = np.array([[.8, .2], [.3, .7], [.6, .4], [.1, .9]])
        first, second = _softmax(LOGITS), _softmax(-LOGITS)
        mixture = (first + second) / 2.
        count = 0
        for mode, kind, scope, temperature, mask_id, graph in itertools.product(
            ("normal", "distil", "both"), KINDS, SCOPES, (1., 2.), range(3), (False, True)
        ):
            with self.subTest(mode=mode, kind=kind, scope=scope, temperature=temperature, 
                              mask=mask_id, graph=graph):
                model.network.cls_token_regularizer_kwargs = {"train_type": mode, "distil_type": kind}
                model.use_clf_distil_ctr_loss = mode != "normal"
                model.clf_distil_temperature, model.clf_distil_scope = temperature, scope
                weights = WEIGHTS[mask_id]
                mass = np.ones(4) if weights is None else weights
                ordinary = float(np.sum(-np.log(mixture[np.arange(4), CLASSES]) * mass) / mass.sum()) if mass.sum() else 0.
                kd, _ = _reference_term(teacher, np.log(mixture), np.arange(2), 
                                        _scope_rows(scope, 2), weights, kind, temperature, 1.7)
                expected = ordinary if mode == "normal" else kd if mode == "distil" else (ordinary + kd) / 2.

                def evaluate() -> tuple[tf.Tensor, tf.Tensor]:
                    """Compute the real auxiliary probability-mixture loss for fixed local factors.

                    Returns:
                        values (tuple[tf.Tensor, tf.Tensor]): Scalar float64 loss and mixture [4,4].
                    """

                    return model.compute_clf_distil_ctr_loss(
                        tf.constant(CLASSES), [tf.constant(first), None, tf.constant(second)], 
                        teacher_labels=tf.constant(teacher), replay_mask=tf.constant(REPLAY), 
                        loss_mask=None if weights is None else tf.constant(weights), 
                        classes_logits_list=[tf.constant(LOGITS), None, tf.constant(-LOGITS)])

                execute = tf.function(evaluate) if graph else evaluate
                actual, predictions = execute()
                np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-9)
                np.testing.assert_allclose(predictions, mixture, atol=2e-10, rtol=2e-9)
                count += 1
        self.assertEqual(count, 216)

    def test_noise_loss_role_mask_and_compiled_loss_matrix(self) -> None:
        """Enumerate 54 MSE/MAE/Huber role/mask/execution combinations.

        Returns:
            result (None): Selected-row native Keras loss and exact student gradients
                equal independent formulas; all teacher targets remain detached.
        """

        student = np.linspace(-1.4, 1.6, 32).reshape(4, 2, 2, 2)
        previous, current = student * -.3 + .7, student * .4 - .2
        count = 0
        for loss_name in ("mse", "mae", "huber"):
            loss_object = tf.keras.losses.Huber(dtype="float64") if loss_name == "huber" else loss_name
            model = _LossHarness(loss_object)
            for roles, mask_id, graph in itertools.product((tuple([0]), tuple([1]), (0, 1)), range(3), (False, True)):
                with self.subTest(loss=loss_name, roles=roles, mask=mask_id, graph=graph):
                    model.specifications = tuple(dict(role=("previous", "current")[index], 
                                                       weight=(1.7, .6)[index]) for index in roles)
                    masks = (WEIGHTS[mask_id], None if WEIGHTS[mask_id] is None else WEIGHTS[mask_id][::-1].copy())
                    expected, expected_gradient = 0., np.zeros_like(student)
                    for index in roles:
                        residual = student - (previous, current)[index]
                        # MSE uses squared residuals and their exact derivative.
                        if loss_name == "mse":
                            elements, derivative = residual ** 2, 2. * residual
                        # MAE uses signed residuals away from its nondifferentiable origin.
                        elif loss_name == "mae":
                            elements, derivative = np.abs(residual), np.sign(residual)
                        # Huber's quadratic core transitions to a unit-slope linear tail.
                        else:
                            magnitude = np.abs(residual)
                            elements = np.where(magnitude <= 1., .5 * residual ** 2, magnitude - .5)
                            derivative = np.clip(residual, -1., 1.)
                        weights = np.ones(4) if masks[index] is None else masks[index]
                        normalized = weights / weights.sum() if weights.sum() else np.zeros(4)
                        coefficient = (1.7, .6)[index]
                        expected += coefficient * np.sum(elements.mean(axis=(1, 2, 3)) * normalized)
                        expected_gradient += coefficient * derivative * normalized[:, None, None, None] / 8.

                    def evaluate(first: tf.Tensor, second: tf.Tensor, values: tf.Tensor) -> tuple[tf.Tensor | None, ...]:
                        """Differentiate real compiled teacher losses with independent role masks.

                        Args:
                            first (tf.Tensor): Float64 previous epsilon [4,2,2,2].
                            second (tf.Tensor): Float64 current epsilon [4,2,2,2].
                            values (tf.Tensor): Float64 student epsilon [4,2,2,2].

                        Returns:
                            result (tuple[tf.Tensor, ...]): Loss, student derivative, and absent teacher derivatives.
                        """

                        with tf.GradientTape() as tape:
                            tape.watch((first, second, values))
                            targets = tuple((first, second)[index] for index in roles)
                            selected = tuple(None if masks[index] is None else tf.constant(masks[index]) for index in roles)
                            loss = model.compute_distil_noise_loss(targets, values, teacher_noise_mask=selected)
                        gradient, old_gradient, new_gradient = tape.gradient(loss, (values, first, second))
                        return loss, gradient, old_gradient, new_gradient

                    execute = tf.function(evaluate) if graph else evaluate
                    actual, gradient, old_gradient, new_gradient = execute(
                        tf.constant(previous), tf.constant(current), tf.constant(student))
                    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-9)
                    np.testing.assert_allclose(gradient, expected_gradient, atol=2e-10, rtol=2e-9)
                    self.assertIsNone(old_gradient)
                    self.assertIsNone(new_gradient)
                    count += 1
        self.assertEqual(count, 54)

    def test_diffusion_objective_enablement_and_branch_matrix(self) -> None:
        """Enumerate 256 loss-enable/conditional-source/execution combinations.

        Returns:
            result (None): Every weighted noise/KD/image/KL/token sum equals its
                independent analytic components, with correct conditional branch selection.
        """

        model = _LossHarness()
        images = tf.constant(np.linspace(-.7, .8, 8).reshape(4, 1, 2, 1))
        noise = images * 2.
        prediction = images + .3
        image_prediction = images * .5
        teacher = images - .4
        teacher_mask = tf.constant([1., 0., .5, 1.], tf.float64)
        means = (np.ones((4, 2)) * .2, np.ones((4, 2)) * .7)
        log_variances = (np.ones((4, 2)) * -.4, np.ones((4, 2)) * .3)
        latents = tuple([(tf.constant(mean), tf.constant(logvar))] for mean, logvar in zip(means, log_variances))
        probabilities = (_softmax(LOGITS), _softmax(-LOGITS))
        noise_expected = np.mean((noise.numpy() - prediction.numpy()) ** 2)
        image_expected = np.mean((images.numpy() - image_prediction.numpy()) ** 2)
        teacher_rows = ((teacher.numpy() - prediction.numpy()) ** 2).mean(axis=(1, 2, 3))
        teacher_expected = 1.7 * np.sum(teacher_rows * teacher_mask.numpy()) / np.sum(teacher_mask)
        kl_expected = tuple(np.mean(.5 * np.sum(mean ** 2 + np.exp(logvar) - 1. - logvar, axis=-1))
                            for mean, logvar in zip(means, log_variances))
        token_expected = tuple(np.mean(-np.log(probability[np.arange(4), CLASSES])) for probability in probabilities)
        count = 0
        for flags, kl_branch, ctr_branch, graph in itertools.product(
            itertools.product((False, True), repeat=5), (0, 1), (0, 1), (False, True)
        ):
            with self.subTest(flags=flags, kl_branch=kl_branch, ctr_branch=ctr_branch, graph=graph):
                model.use_noise_distil_loss, model.use_image_loss, model.use_kl_loss, model.use_ctr_loss = flags[1:]
                for name, coefficient, active in zip(
                    ("noise_loss_coef", "noise_distil_loss_coef", "image_loss_coef", "kl_loss_coef", "ctr_loss_coef"), 
                    (1.3, .7, .2, .4, .6), flags
                ):
                    setattr(model, name, tf.constant(coefficient if active else 0., tf.float64))
                expected_parts = (noise_expected, teacher_expected if flags[1] else 0., 
                                  image_expected if flags[2] else 0., kl_expected[kl_branch] if flags[3] else 0., 
                                  token_expected[ctr_branch] if flags[4] else 0.)
                expected_total = sum(value * coefficient * active for value, coefficient, active in
                                     zip(expected_parts, (1.3, .7, .2, .4, .6), flags))

                def evaluate() -> tuple[tf.Tensor, ...]:
                    """Evaluate the complete real diffusion loss aggregator for fixed enablement.

                    Returns:
                        output (tuple[tf.Tensor, ...]): Weighted total and unweighted loss diagnostics.
                    """

                    return model.compute_noise_distil_image_kl_ctr_loss(
                        images, noise, tf.constant(CLASSES), image_prediction, prediction, 
                        latents[0], [tf.constant(probabilities[0])], teacher_noises_pred=teacher, 
                        z_vals_list_u=latents[1], regs_list_u=[tf.constant(probabilities[1])], 
                        kl_train_type=("cond", "uncond")[kl_branch], 
                        ctr_train_type=("cond", "uncond")[ctr_branch], teacher_noise_mask=teacher_mask)

                execute = tf.function(evaluate) if graph else evaluate
                output = execute()
                np.testing.assert_allclose(output[0], expected_total, atol=2e-10, rtol=2e-9)
                for actual, expected in zip((output[1], *output[4:8]), expected_parts):
                    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-9)
                count += 1
        self.assertEqual(count, 256)

    def test_classifier_input_ensemble_and_teacher_role_matrix(self) -> None:
        """Run 21 valid input-policy/ensemble/previous-current-teacher update cases.

        V1 supports all clean/noisy and null/all-class input selections, with an
        ensemble only for noisy/all-class inputs without explicit caps. V2 supports
        its noisy/null selector with either ensemble setting; its own caps choose
        corruption. Each valid policy is crossed with previous, current, and dual
        teachers, with task-local current support mapped to student class two.

        Returns:
            result (None): Every supported policy updates a student head, uses an
                ensemble exactly when requested, and preserves both teacher weights.
        """

        policies = ((1, "noisy", "all_classes", False), (1, "noisy", "all_classes", True), 
                    (1, "noisy", "null_class_only", False), (1, "clean", "all_classes", False), 
                    (1, "clean", "null_class_only", False), (2, "noisy", "null_class_only", False), 
                    (2, "noisy", "null_class_only", True))
        images = tf.reshape(tf.linspace(-.7, .8, 48), (3, 4, 4, 1))
        labels, replay = tf.constant([0, 1, 2]), tf.constant([True, False, False])
        count = 0
        for (version, corruption, conditions, ensemble), roles in itertools.product(
            policies, ("previous", "current", "both")
        ):
            with self.subTest(version=version, corruption=corruption, conditions=conditions, 
                              ensemble=ensemble, roles=roles):
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(947)
                network = _raw_network("transformer", True, 3)
                previous = _raw_network("transformer", True, 2) if roles != "current" else None
                current = _raw_network("transformer", True, 1) if roles != "previous" else None
                wrapper_type = DiffusionClassifier if version == 1 else DiffusionClassifierV2
                model = wrapper_type(
                    network=network, teacher_network=previous, current_teacher_network=current, 
                    use_ema=False, scheduler_name="linear", test_steps=2, seed=947, 
                    p_uncond=.5, mask_by_nulls=False, mask_by_t_threshold=False, 
                    clf_train_noisy_input_type=corruption, clf_train_class_input_type=conditions, 
                    use_ensemble_loss_instead=ensemble, noise_loss_coef=0., clf_loss_coef=.3, 
                    clf_distil_loss_coef=.8, clf_distil_type="soft", clf_distil_temperature=2., 
                    clf_distil_scope="old_classes", previous_teacher_clf_loss_weight=1.7, 
                    current_teacher_clf_loss_weight=.6
                )
                # A task-local current head maps its sole output to the third student class.
                if current is not None:
                    model.set_current_teacher_network(current, class_ids=[2], task_class_ids=[2])
                model.compile(optimizer=tf.keras.optimizers.SGD(.01), loss="mse", run_eagerly=True)
                teachers = tuple(teacher for teacher in (previous, current) if teacher is not None)
                snapshots = tuple([value.numpy().copy() for value in teacher.weights] for teacher in teachers)
                # Match the two phase selectors set by the public discriminator fit lifecycle.
                if version == 2:
                    model._switch_train_part("discriminator")
                    model._switch_test_part("discriminator")
                before = [value.numpy().copy() for value in network.classifier.weights]
                model._preprocess_training = True
                prepared = model.prep_inputs_map(images, labels, replay_mask=replay)
                model._preprocess_training = None
                # Observe the real ensemble call without replacing any prediction or gradient.
                if ensemble:
                    with patch.object(model.ensemble_loss_fn, "ensemble_predict_batched", 
                                      wraps=model.ensemble_loss_fn.ensemble_predict_batched) as observed:
                        result = model.train_step(prepared)
                    self.assertEqual(observed.call_count, 1)
                # Direct classifiers preserve their explicitly selected input route.
                else:
                    self.assertIsNone(model.ensemble_loss_fn)
                    result = model.train_step(prepared)
                self.assertTrue(any(np.any(old != new.numpy()) for old, new in zip(before, network.classifier.weights)))
                self.assertGreater(float(result["clf_distil_loss"]), 0.)
                for teacher, snapshot in zip(teachers, snapshots):
                    self.assertFalse(teacher.trainable)
                    for expected, actual in zip(snapshot, teacher.weights):
                        np.testing.assert_array_equal(actual, expected)
                count += 1
                del model, network, previous, current
                gc.collect()
        self.assertEqual(count, 21)

    def test_teacher_support_softening_and_scalar_boundaries(self) -> None:
        """Check 36 zero/tiny-support, invalid-mode, and scalar-dtype boundary cases.

        Valid full teacher distributions may have no mass in an older student's
        retained vocabulary. Such selected rows are rejected; masked-out rows still
        yield zero. Softening retains exact zeros and tiny positive probabilities.

        Returns:
            result (None): All valid edge losses and score gradients match NumPy,
                invalid retained support/modes raise, and scalar Tensor weights remain accepted.
        """

        model = _LossHarness()
        scores = tf.constant(np.log([[.3, .7]]), tf.float64)
        count = 0
        for target, temperature, graph in itertools.product(
            ([[1., 0.]], [[1e-100, 1.]], [[.8, .2]]), (.7, 1., 2.), (False, True)
        ):
            with self.subTest(target=target, temperature=temperature, graph=graph):
                teacher = np.asarray(target, np.float64)
                expected, expected_gradient = _reference_term(
                    teacher, scores.numpy(), np.arange(2), np.ones(1, bool), None, 
                    "soft", temperature, 1.7)

                def evaluate(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
                    """Differentiate an exact zero-preserving temperature target.

                    Args:
                        values (tf.Tensor): Float64 student scores [1,2].

                    Returns:
                        values (tuple[tf.Tensor, tf.Tensor]): Scalar KL and student score gradient [1,2].
                    """

                    with tf.GradientTape() as tape:
                        tape.watch(values)
                        loss, _ = model.compute_clf_distil_loss(
                            tf.constant(teacher), tf.nn.softmax(values), student_logits=values, 
                            clf_distil_temperature=temperature)
                    return loss, tape.gradient(loss, values)

                execute = tf.function(evaluate) if graph else evaluate
                actual, gradient = execute(scores)
                np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-9)
                np.testing.assert_allclose(gradient, expected_gradient, atol=2e-10, rtol=2e-9)
                count += 1
        for kind, masked, graph in itertools.product(KINDS, (False, True), (False, True)):
            with self.subTest(kind=kind, masked=masked, graph=graph):
                def evaluate_support() -> tf.Tensor:
                    """Evaluate a valid teacher whose entire mass is beyond the student head.

                    Returns:
                        loss (tf.Tensor): Zero for an excluded row; eligible rows raise InvalidArgumentError.
                    """

                    return model.compute_clf_distil_loss(
                        tf.constant([[0., 0., 1.]], tf.float64), tf.nn.softmax(scores), 
                        clf_distil_type=kind, clf_distil_temperature=2., student_logits=scores, 
                        clf_distil_loss_mask=tf.constant([0. if masked else 1.], tf.float64))[0]

                execute = tf.function(evaluate_support) if graph else evaluate_support
                # Excluded teacher rows carry no objective and need no invented target.
                if masked:
                    self.assertEqual(float(execute()), 0.)
                # Active no-overlap distributions cannot define either a hard or soft target.
                else:
                    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "positive mass"):
                        execute()
                count += 1
        for graph in (False, True):
            execute = tf.function(model.compute_clf_distil_loss) if graph else model.compute_clf_distil_loss
            with self.assertRaisesRegex(ValueError, "clf_distil_type"):
                execute(tf.constant([[.8, .2]], tf.float64), tf.nn.softmax(scores), clf_distil_type="unknown")
            count += 1
            with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "shared class support"):
                execute(tf.zeros((1, 0), tf.float64), tf.nn.softmax(scores), student_logits=scores)
            count += 1
        for coefficient, graph in itertools.product(
            (1.7, tf.constant(1.7, tf.float32), tf.constant(1.7, tf.float64)), (False, True)
        ):
            execute = tf.function(model.compute_clf_distil_loss) if graph else model.compute_clf_distil_loss
            expected, _ = _reference_term(np.array([[.8, .2]]), scores.numpy(), np.arange(2), 
                                           np.ones(1, bool), None, "soft", .7, float(coefficient))
            actual, _ = execute(tf.constant([[.8, .2]], tf.float64), tf.nn.softmax(scores), 
                                student_logits=scores, clf_distil_temperature=.7, teacher_loss_weight=coefficient)
            np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-9)
            count += 1
        self.assertEqual(count, 36)

    def test_real_architecture_wrapper_update_matrix(self) -> None:
        """Run 32 compatible architecture/wrapper/head/execution configurations.

        Eight plain-generator and twenty-four classifier configurations perform 44
        optimizer steps: V2 exercises each disjoint optimizer once while its inactive
        group stays unchanged. Both primary and independent KD heads must update.
        Both losses are enabled, native teachers stay unchanged, and actual
        predictions retain image/class shapes after each eager or graph update.

        Returns:
            result (None): Supported real wrapper families update student variables,
                freeze teachers, preserve optimizer counts, and produce valid distributions.
        """

        images = tf.reshape(tf.linspace(-.8, .8, 48), (3, 4, 4, 1))
        labels = tf.constant([0, 1, 2], tf.int32)
        replay = tf.constant([True, False, True])
        count = 0
        for family, version, head, graph in itertools.product(
            ("unet", "transformer", "encoder_decoder", "decoder"), (0, 1, 2), (False, True), (False, True)
        ):
            # The isolated decoder has no classifier branch for V1/V2 wrappers.
            if family == "decoder" and version:
                continue
            # Plain denoisers have no classifier head; its presence is not an independent factor.
            if version == 0 and head:
                continue
            with self.subTest(family=family, version=version, head=head, graph=graph):
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(937)
                network = _raw_network(family, version != 0, 3, head)
                teacher = _raw_network(family, version != 0, 2)
                options = dict(network=network, teacher_network=teacher, use_ema=False, 
                               test_network_name="raw", scheduler_name="linear", test_steps=2, 
                               p_uncond=1., noise_loss_coef=.4, noise_distil_loss_coef=.7, seed=937)
                # Classifier wrappers jointly expose primary CE and same-pass soft KD.
                if version:
                    options.update(clf_loss_coef=.3, clf_distil_loss_coef=.8, 
                                   clf_distil_type="soft", clf_distil_temperature=2., 
                                   clf_distil_scope="replay_only", mask_by_nulls=False, 
                                   mask_by_t_threshold=False)
                wrapper_type = (DiffusionModel, DiffusionClassifier, DiffusionClassifierV2)[version]
                model = wrapper_type(**options)
                model.compile(optimizer=tf.keras.optimizers.SGD(.01), loss="mse", run_eagerly=not graph)
                teacher_before = [value.numpy().copy() for value in teacher.weights]
                classifier_before = [value.numpy().copy() for value in network.classifier.weights] if version else []
                distil_before = [value.numpy().copy() for value in network.distil_classifier.weights] if head else []
                self.assertFalse(teacher.trainable)
                self.assertFalse({id(value) for value in teacher.weights} &
                                 {id(value) for value in model.network.trainable_variables})
                phases = ("generator", "discriminator") if version == 2 else tuple(["joint"])
                for phase in phases:
                    # V2 requires phase-specific preparation and distinct variable/optimizer groups.
                    if version == 2:
                        model._switch_train_part(phase)
                        model._switch_test_part(phase)
                        variables = model.gen_trainable_variables if phase == "generator" else model.clf_trainable_variables
                        optimizer = model.gen_optimizer if phase == "generator" else model.clf_optimizer
                        inactive = model.clf_trainable_variables if phase == "generator" else model.gen_trainable_variables
                        self.assertFalse({id(value) for value in variables} & {id(value) for value in inactive})
                    # The plain and joint wrappers use their single raw-variable optimizer.
                    else:
                        variables, optimizer = model.network.trainable_variables, model.optimizer
                        inactive = []
                    before = [value.numpy().copy() for value in variables]
                    inactive_before = [value.numpy().copy() for value in inactive]
                    model._preprocess_training = True
                    prepared = model.prep_inputs_map(images, labels, replay_mask=replay) if version else model.prep_inputs_map(images, labels)
                    model._preprocess_training = None
                    execute = tf.function(model.train_step) if graph else model.train_step
                    result = execute(prepared)
                    self.assertEqual(int(optimizer.iterations), 1)
                    self.assertTrue(any(np.any(old != new.numpy()) for old, new in zip(before, variables)))
                    for expected, actual in zip(inactive_before, inactive):
                        np.testing.assert_array_equal(actual, expected)
                    for value in result.values():
                        self.assertTrue(np.isfinite(value.numpy()).all())
                for before, after in zip(teacher_before, teacher.weights):
                    np.testing.assert_array_equal(after, before)
                # Main classification must update its actual head independently of denoiser progress.
                if version:
                    self.assertTrue(any(np.any(old != new.numpy()) for old, new in
                                        zip(classifier_before, network.classifier.weights)))
                # The separate KD head has no supervised objective, so its movement proves KD routing.
                if head:
                    self.assertTrue(any(np.any(old != new.numpy()) for old, new in
                                        zip(distil_before, network.distil_classifier.weights)))
                output = network((images, tf.zeros(3, tf.int32), labels + 1), training=False) if version else None
                noise = output["noises"] if version else model.call_network(
                    images, tf.zeros(3, tf.int32), labels + 1, training=False)[0][0]
                self.assertEqual(tuple(noise.shape), (3, 4, 4, 1))
                # Every classifier head remains an ordinary normalized inference distribution.
                if version:
                    np.testing.assert_allclose(tf.reduce_sum(output["classes"], axis=-1), 1., atol=1e-6)
                    self.assertEqual(tuple(output["classes"].shape), (3, 3))
                # Exercise each reverse-process regime through every plain wrapper adapter.
                if version == 0:
                    for eta in (0., .5, 1.):
                        steps = 4 if eta == 1. else 2
                        generated, states, estimates = model.sample(
                            network_name="raw", labels=[1], x_t=images[:1], 
                            steps=steps, scale=1.5, eta=eta, return_x_ts=True, return_x0s=True)
                        self.assertEqual(tuple(generated.shape), (1, 4, 4, 1))
                        self.assertEqual(len(states), steps)
                        self.assertEqual(len(estimates), steps)
                        self.assertTrue(np.isfinite(generated.numpy()).all())
                        np.testing.assert_array_equal(generated, estimates[-1])
                        self.assertTrue(np.all((generated.numpy() >= 0.) & (generated.numpy() <= 1.)))
                        # Supplied initial noise makes eta-zero sampling exactly repeatable.
                        if eta == 0.:
                            repeated = model.sample(network_name="raw", labels=[1], x_t=images[:1], 
                                                    steps=steps, scale=1.5, eta=eta)
                            np.testing.assert_array_equal(repeated, generated)
                count += 1
                print(f"matrix real configuration {count}/32: {family} V{version} head={head} graph={graph}", flush=True)
                del execute, model, network, teacher
                gc.collect()
        self.assertEqual(count, 32)


# Direct execution runs the focused matrix without invoking the complete registry.
if __name__ == "__main__":
    unittest.main()
