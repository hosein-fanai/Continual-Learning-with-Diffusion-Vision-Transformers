"""Independent numerical and native-update checks for classifier audit repairs."""

from __future__ import annotations

import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.tests.test_wrapper_verified_repairs import _constant_head, _network
from diffusion import DiffusionClassifier, DiffusionClassifierV2, DiffusionModel


class _ClassifierLossHarness(tf.keras.Model):
    """Run production loss kernels without an architecture or optimizer fixture."""

    compute_clf_distil_loss = DiffusionClassifier.compute_clf_distil_loss
    compute_clf_distil_ctr_loss = DiffusionClassifier.compute_clf_distil_ctr_loss
    _compute_single_classifier_distillation = DiffusionClassifier._compute_single_classifier_distillation
    _compute_dual_classifier_distillation = DiffusionClassifier._compute_dual_classifier_distillation
    _classifier_teacher_support = DiffusionClassifier._classifier_teacher_support
    _classifier_teacher_mask = DiffusionClassifier._classifier_teacher_mask
    compute_ctr_loss = DiffusionModel.compute_ctr_loss

    def __init__(self, dtype: str = "float64") -> None:
        """Provide explicit vocabulary, objective and precision metadata."""

        super().__init__(dtype=dtype)
        self.network = SimpleNamespace(num_classes=2, cls_token_regularizer_kwargs={})
        self.scce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction="none", dtype=dtype
        )
        self.clf_distil_type = "hard"
        self.clf_distil_temperature = 1.
        self.clf_distil_scope = "current_and_replay"
        self.previous_teacher_clf_loss_weight = 1.
        self.dual_teacher_scope = "global"
        self.use_clf_ctr_loss = True
        self.use_clf_distil_ctr_loss = True
        self.specifications = ()

    def _current_teacher_spec(self, kind: str) -> None:
        """Keep these numerical fixtures on the single previous-role boundary."""

        return None

    def _uses_mapped_classifier_teachers(self) -> bool:
        """Select the mapped public boundary only when the fixture supplies a map."""

        return bool(self.specifications)

    def _classifier_teacher_specs(self) -> tuple[dict[str, object], ...]:
        """Return explicit teacher-column identities for mapped cases."""

        return self.specifications


class _SplitTokenHarness(_ClassifierLossHarness):
    """Keep batch selection isolated while evaluating the actual token CE implementation."""

    compute_batch_diffusion_losses = DiffusionClassifier.compute_batch_diffusion_losses

    def compute_noise_distil_image_kl_ctr_loss(
        self, classes: tf.Tensor, regs_list_c: list[tf.Tensor]
    ) -> tuple:
        """Return the production token loss in the inherited nine-output layout."""

        loss, predictions = self.compute_ctr_loss(classes, regs_list_c)
        zero = tf.zeros_like(loss)
        return loss, zero, None, None, zero, zero, zero, loss, predictions


class ClassifierAuditRepairTests(unittest.TestCase):
    """Check extreme gradients, unchanged moderate losses and real same-pass updates."""

    def setUp(self) -> None:
        """Keep native fixtures isolated from any caller precision policy."""

        self.original_policy = tf.keras.mixed_precision.global_policy().name
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.utils.set_random_seed(1517)

    def tearDown(self) -> None:
        """Release native model state and restore the caller's precision policy."""

        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy(self.original_policy)

    def test_effective_null_mask_is_validated_before_construction_completes(self) -> None:
        """None and True reject zero dropout; positive dropout preserves their equivalence."""

        for mask in (None, True):
            with self.subTest(mask=mask):
                with self.assertRaisesRegex(AssertionError, "requires p_uncond > 0"):
                    DiffusionClassifier(
                        network=_network(3, distil=False), use_ema=False, test_steps=2, 
                        mask_by_nulls=mask, p_uncond=0.
                    )
                model = DiffusionClassifier(
                    network=_network(3, distil=False), use_ema=False, test_steps=2, 
                    mask_by_nulls=mask, p_uncond=.5
                )
                self.assertTrue(model.mask_by_nulls)
                self.assertIs(model.get_config()["mask_by_nulls"], True)

    def test_explicit_unmasked_zero_dropout_selects_and_updates_every_row(self) -> None:
        """The supported zero-dropout CE-only recipe retains all three examples."""

        model = DiffusionClassifier(
            network=_network(3, distil=False), use_ema=False, test_steps=2, 
            mask_by_nulls=False, mask_by_t_threshold=False, p_uncond=0., 
            clf_loss_coef=1., noise_loss_coef=0., preprocess_type=None
        )
        model.compile(optimizer=tf.keras.optimizers.SGD(.1), loss="mse", 
                      run_eagerly=True, jit_compile=False)
        _constant_head(model.network.classifier, [0., 0., 0.])
        images = tf.reshape(tf.linspace(-1., 1., 48), (3, 4, 4, 1))
        result = model.train_step((images, tf.constant([0, 0, 1])))
        self.assertAlmostEqual(float(result["classifier_loss"]), np.log(3.), places=6)
        self.assertEqual(float(model.clf_loss_tracker.count), 3.)
        np.testing.assert_allclose(
            model.network.classifier.weights[-1], [.1 / 3., 0., -.1 / 3.], atol=1e-7
        )

    def test_hard_kd_extreme_scores_keep_full_mapped_denominator_and_masks(self) -> None:
        """Mapped unsupported columns still repel target mass and excluded rows have zero gradients."""

        for mapped, graph in itertools.product((False, True), repeat=2):
            with self.subTest(mapped=mapped, graph=graph):
                model = _ClassifierLossHarness()
                # The mapped target is column two; the highest score is outside teacher support.
                if mapped:
                    model.specifications = tuple([dict(
                        role="previous", weight=1., class_ids=(2, 0), task_class_ids=None
                    )])
                teacher = tf.Variable([[1., 0.], [1., 0.]], dtype=tf.float64)
                scores = tf.constant([[100., 0., -100.], [100., 0., -100.]], tf.float64) \
                    if mapped else tf.constant([[-100., 100.], [-100., 100.]], tf.float64)

                def evaluate(logits: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor | None]:
                    """Differentiate the actual loss while watching both sides of KD."""

                    with tf.GradientTape(persistent=True) as tape:
                        tape.watch(logits)
                        loss, _ = model.compute_clf_distil_loss(
                            teacher, tf.nn.softmax(logits), 
                            clf_distil_loss_mask=tf.constant([1., 0.], tf.float64), 
                            student_logits=logits
                        )
                    return loss, tape.gradient(loss, logits), tape.gradient(loss, teacher)

                execute = tf.function(evaluate) if graph else evaluate
                loss, gradient, teacher_gradient = execute(scores)
                self.assertAlmostEqual(float(loss), 200., places=10)
                expected = [[1., 0., -1.], [0., 0., 0.]] if mapped \
                    else [[-1., 1.], [0., 0.]]
                np.testing.assert_allclose(gradient, expected, atol=1e-15)
                self.assertIsNone(teacher_gradient)
                with tf.GradientTape() as tape:
                    tape.watch(scores)
                    empty, _ = model.compute_clf_distil_loss(
                        teacher, tf.nn.softmax(scores), 
                        clf_distil_loss_mask=tf.zeros(2, tf.float64), student_logits=scores
                    )
                self.assertEqual(float(empty), 0.)
                np.testing.assert_array_equal(tape.gradient(empty, scores), np.zeros(scores.shape))

    def test_hard_probability_api_matches_recoverable_scores_and_rejects_underflow(self) -> None:
        """Positive direct probabilities retain exact CE without Keras clipping."""

        model = _ClassifierLossHarness()
        scores = tf.Variable([[-100., 100.]], dtype=tf.float64)
        with tf.GradientTape() as tape:
            loss, _ = model.compute_clf_distil_loss(
                tf.constant([[1., 0.]], tf.float64), tf.nn.softmax(scores)
            )
        self.assertAlmostEqual(float(loss), 200., places=10)
        np.testing.assert_allclose(tape.gradient(loss, scores), [[-1., 1.]], atol=1e-15)
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "same-pass student_logits"):
            model.compute_clf_distil_loss(
                tf.constant([[1., 0.]], tf.float64), tf.constant([[0., 1.]], tf.float64)
            )

    def test_auxiliary_extreme_mixture_preserves_shared_and_independent_gradients(self) -> None:
        """Identical heads produce CE 200, with each independent head owning half its gradient."""

        for mode, kind, graph, shared in itertools.product(
            ("normal", "distil", "both"), ("hard", "soft"), (False, True), (False, True)
        ):
            with self.subTest(mode=mode, kind=kind, graph=graph, shared=shared):
                model = _ClassifierLossHarness()
                model.network.cls_token_regularizer_kwargs = dict(train_type=mode, distil_type=kind)
                model.use_clf_distil_ctr_loss = mode != "normal"

                def evaluate(first: tf.Tensor, second: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor | None]:
                    """Differentiate the same probability mixture consumed by each objective."""

                    with tf.GradientTape(persistent=True) as tape:
                        tape.watch([first, second])
                        logits = [first, first if shared else second]
                        loss, _ = model.compute_clf_distil_ctr_loss(
                            tf.constant([0]), [tf.nn.softmax(value) for value in logits], 
                            teacher_labels=tf.constant([[1., 0.]], tf.float64), 
                            classes_logits_list=logits
                        )
                    return loss, tape.gradient(loss, first), tape.gradient(loss, second)

                execute = tf.function(evaluate) if graph else evaluate
                loss, first_gradient, second_gradient = execute(
                    tf.constant([[-100., 100.]], tf.float64), 
                    tf.constant([[-100., 100.]], tf.float64)
                )
                self.assertAlmostEqual(float(loss), 200., places=10)
                expected = [[-1., 1.]] if shared else [[-.5, .5]]
                np.testing.assert_allclose(first_gradient, expected, atol=1e-13)
                # Reusing one score tensor accumulates both head contributions.
                if shared:
                    self.assertIsNone(second_gradient)
                # Independent heads each receive half the identical-head gradient.
                else:
                    np.testing.assert_allclose(second_gradient, expected, atol=1e-13)

    def test_moderate_auxiliary_ce_remains_probability_mixture_with_weighted_rows(self) -> None:
        """Distinct heads match a NumPy mean-probability oracle rather than averaged logits."""

        model = _ClassifierLossHarness()
        model.use_clf_distil_ctr_loss = False
        first = tf.Variable([[2., -1.], [-1., .5]], dtype=tf.float64)
        second = tf.Variable([[-.5, 1.], [3., -2.]], dtype=tf.float64)
        labels = np.array([0, 1])
        weights = np.array([1., 3.])
        probabilities = []
        for scores in (first.numpy(), second.numpy()):
            exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities.append(exp_scores / exp_scores.sum(axis=-1, keepdims=True))
        mixture = (probabilities[0] + probabilities[1]) / 2.
        expected_loss = np.sum(-np.log(mixture[np.arange(2), labels]) * weights) / weights.sum()
        with tf.GradientTape() as tape:
            loss, actual_mixture = model.compute_clf_distil_ctr_loss(
                tf.constant(labels), [tf.nn.softmax(first), tf.nn.softmax(second)], 
                loss_mask=tf.constant(weights), classes_logits_list=[first, second]
            )
        gradients = tape.gradient(loss, [first, second])
        self.assertAlmostEqual(float(loss), expected_loss, places=12)
        np.testing.assert_allclose(actual_mixture, mixture, atol=1e-14)
        for probability, gradient in zip(probabilities, gradients):
            responsibilities = probability[np.arange(2), labels] / (2. * mixture[np.arange(2), labels])
            expected = (probability - np.eye(2)[labels]) * responsibilities[:, None]
            expected *= (weights / weights.sum())[:, None]
            np.testing.assert_allclose(gradient, expected, atol=1e-14)
        with tf.GradientTape() as tape:
            empty, _ = model.compute_clf_distil_ctr_loss(
                tf.constant(labels), [tf.nn.softmax(first), tf.nn.softmax(second)], 
                loss_mask=tf.zeros(2, tf.float64), classes_logits_list=[first, second]
            )
        self.assertEqual(float(empty), 0.)
        for gradient in tape.gradient(empty, [first, second]):
            np.testing.assert_array_equal(gradient, np.zeros((2, 2)))

    def test_soft_temperature_scaling_and_normal_hard_scores_are_unchanged(self) -> None:
        """Both KD modes retain independent NumPy CE/KL values and analytical derivatives."""

        model = _ClassifierLossHarness()
        targets = np.array([[.8, .2], [.1, .9]])
        original_scores = np.array([[.4, -.7, .1], [-.5, .2, 1.]])
        for kind, temperature in itertools.product(("hard", "soft"), (1., 2.)):
            with self.subTest(kind=kind, temperature=temperature):
                scores = tf.Variable(original_scores, dtype=tf.float64)
                with tf.GradientTape() as tape:
                    loss, _ = model.compute_clf_distil_loss(
                        tf.constant(targets), tf.nn.softmax(scores), 
                        clf_distil_type=kind, clf_distil_temperature=temperature, 
                        student_logits=scores
                    )
                scale = temperature if kind == "soft" else 1.
                exponentials = np.exp(original_scores / scale)
                student = exponentials / exponentials.sum(axis=-1, keepdims=True)
                # Hard targets select exactly one supported teacher column.
                if kind == "hard":
                    target = np.eye(3)[targets.argmax(axis=-1)]
                    expected_loss = -np.log(student[np.arange(2), targets.argmax(axis=-1)]).mean()
                # Soft targets retain temperature scaling and zero new-class mass.
                else:
                    softened = targets ** (1. / temperature)
                    softened /= softened.sum(axis=-1, keepdims=True)
                    target = np.pad(softened, ((0, 0), (0, 1)))
                    expected_loss = np.sum(softened * (np.log(softened) - np.log(student[:, :2])), axis=-1).mean()
                    expected_loss *= temperature ** 2
                expected_gradient = scale * (student - target) / 2.
                self.assertAlmostEqual(float(loss), expected_loss, places=12)
                np.testing.assert_allclose(tape.gradient(loss, scores), expected_gradient, atol=1e-14)

    def test_split_diffusion_rows_preserve_same_pass_token_scores(self) -> None:
        """Selecting diffusion rows retains logits even when float32 softmax underflows."""

        model = _SplitTokenHarness(dtype="float32")
        model.use_ctr_loss = True
        model.show_separate_noise_losses = False

        def evaluate(scores: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            """Trace row selection before retrieving Keras' same-pass softmax scores."""

            with tf.GradientTape() as tape:
                tape.watch(scores)
                probabilities = tf.keras.activations.softmax(scores)
                losses = model.compute_batch_diffusion_losses(
                    tf.constant([True, False]), 
                    classes=tf.constant([0, 0]), regs_list_c=[probabilities]
                )
            return losses[0], tape.gradient(losses[0], scores)

        for graph in (False, True):
            with self.subTest(graph=graph):
                execute = tf.function(evaluate) if graph else evaluate
                loss, gradient = execute(tf.constant([[-100., 100.], [-100., 100.]]))
                self.assertAlmostEqual(float(loss), 200., places=5)
                np.testing.assert_allclose(gradient, [[-1., 1.], [0., 0.]], atol=1e-7)

    def test_native_auxiliary_ce_and_hard_kd_keep_the_same_pass(self) -> None:
        """Native V1/V2 token normal, distil and both modes update saturated auxiliary heads."""

        images = tf.zeros((2, 4, 4, 1))
        labels = tf.constant([0, 0])
        for wrapper_class, mode in itertools.product(
            (DiffusionClassifier, DiffusionClassifierV2), ("normal", "distil", "both")
        ):
            with self.subTest(wrapper=wrapper_class.__name__, mode=mode):
                student = _network(2, distil=False, auxiliary=mode)
                student.clf_cls_token_regularizer_kwargs["distil_type"] = "hard"
                model = wrapper_class(
                    network=student, teacher_network=_network(2, distil=False), 
                    use_ema=False, test_steps=2, mask_by_nulls=False, mask_by_t_threshold=False, 
                    clf_loss_coef=0., noise_loss_coef=0., clf_distil_loss_coef=0., 
                    ctr_loss_coef=1., clf_distil_scope="current_and_replay", 
                    train_cfg_scale=None, seed=1517
                )
                model.compile(optimizer=tf.keras.optimizers.SGD(.1), loss="mse", 
                              run_eagerly=True, jit_compile=False)
                _constant_head(model.teacher_network.classifier, [10., -10.])
                heads = [stage[student.CTR] for stage in student.clf_layers_dicts if student.CTR in stage]
                self.assertEqual(len(heads), 1)
                _constant_head(heads[0], [-100., 100.])
                with patch.object(student, "compute_class", wraps=student.compute_class) as calls:
                    # V2 trains classifier-owned variables through its discriminator phase.
                    if wrapper_class is DiffusionClassifierV2:
                        model._set_clf_variables()
                        result = model.discriminator_train_step((images, labels))
                    # V1 uses the shared joint training step.
                    else:
                        result = model.train_step((images, labels))
                self.assertEqual(calls.call_count, 1)
                self.assertAlmostEqual(float(result["clf_ctr_loss"]), 200., places=5)
                np.testing.assert_allclose(heads[0].weights[-1], [-99.9, 99.9], atol=5e-6)

    def test_native_hard_kd_uses_one_student_pass_and_keeps_teacher_frozen(self) -> None:
        """V1/V2 main and separate KD heads update from saturated logits without extra inference."""

        images = tf.zeros((2, 4, 4, 1))
        labels = tf.constant([0, 1])
        for wrapper_class, distil in itertools.product(
            (DiffusionClassifier, DiffusionClassifierV2), (False, True)
        ):
            with self.subTest(wrapper=wrapper_class.__name__, distil=distil):
                model = wrapper_class(
                    network=_network(2, distil=distil), teacher_network=_network(2, distil=False), 
                    use_ema=False, test_steps=2, mask_by_nulls=False, mask_by_t_threshold=False, 
                    clf_loss_coef=0., noise_loss_coef=0., clf_distil_loss_coef=1., 
                    clf_distil_type="hard", clf_distil_scope="current_and_replay", 
                    train_cfg_scale=None, seed=1517
                )
                model.compile(optimizer=tf.keras.optimizers.SGD(.1), loss="mse", 
                              run_eagerly=True, jit_compile=False)
                _constant_head(model.teacher_network.classifier, [10., -10.])
                head = model.network.distil_classifier if distil else model.network.classifier
                _constant_head(head, [-100., 100.])
                teacher_before = model.teacher_network.get_weights()
                with patch.object(model.network, "compute_class", wraps=model.network.compute_class) as calls:
                    # V2 trains classifier-owned variables through its discriminator phase.
                    if wrapper_class is DiffusionClassifierV2:
                        model._set_clf_variables()
                        result = model.discriminator_train_step((images, labels))
                    # V1 uses the shared joint training step.
                    else:
                        result = model.train_step((images, labels))
                self.assertEqual(calls.call_count, 1)
                self.assertAlmostEqual(float(result["clf_distil_loss"]), 200., places=5)
                np.testing.assert_allclose(head.weights[-1], [-99.9, 99.9], atol=5e-6)
                self.assertFalse(model.teacher_network.trainable)
                for before, after in zip(teacher_before, model.teacher_network.get_weights()):
                    np.testing.assert_array_equal(before, after)


# Execute only when this regression module is invoked as a script.
if __name__ == "__main__":
    unittest.main()
