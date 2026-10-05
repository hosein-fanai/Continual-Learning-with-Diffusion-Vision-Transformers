"""Disclose proven empty continual retention without changing ablation treatments."""

from __future__ import annotations

import unittest
from unittest.mock import Mock
import warnings

import numpy as np
import tensorflow as tf

from common.learner import _run_continual_tasks
from common.tests.test_continual_combination_matrix import tiny_network
from diffusion import DiffusionClassifier


class EmptyRetentionRepairTests(unittest.TestCase):
    """Exercise the real learner preflight and independent selected-row loss oracles."""

    def setUp(self) -> None:
        """Construct a tiny dynamic student with previous classifier KD only."""

        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")
        self.model = DiffusionClassifier(
            network=tiny_network("dit", True), use_ema=False, 
            scheduler_name="clipped_cosine", test_steps=2, defer_teacher=True, 
            noise_distil_loss_coef=0., clf_distil_loss_coef=1., 
            clf_distil_scope="old_classes", mask_by_nulls=False, 
            mask_by_t_threshold=False, seed=953
        )
        self.model.compile(optimizer="sgd", loss="mse", run_eagerly=True)

    def tearDown(self) -> None:
        """Release the isolated model and restore the default precision policy."""

        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")

    def preflight_warnings(self, **overrides: object) -> list[warnings.WarningMessage]:
        """Stop at the data boundary and return only the retention disclosures."""

        loader = Mock(side_effect=RuntimeError("Reached dataset boundary"))
        options = dict(
            class_num=4, task_size=2, load_dataset_fn=loader, 
            load_dataset_fn_kwargs={"preprocess": None}, 
            generative_model=self.model, use_generative_model_classifier=True, 
            use_distillation=True, use_generative_replay=False, 
            replay_budget_mode="fixed_total", replay_old_examples=0, 
            plot_results=False, deterministic_ops=True, 
            show_generated_images=False, show_network_summary=False, verbose=0, seed=953
        )
        options.update(overrides)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaisesRegex(RuntimeError, "Reached dataset boundary"):
                _run_continual_tasks(**options)
        loader.assert_called_once()
        return [item for item in caught if "Guaranteed-empty distillation" in str(item.message)]

    def test_previous_only_named_and_custom_controls_warn_without_changing_scope(self) -> None:
        """Named LwF/joint KD and explicit unnamed controls remain accepted but disclosed."""

        for baseline in ("lwf", "joint_kd", None):
            with self.subTest(baseline=baseline):
                caught = self.preflight_warnings(baseline=baseline)
                self.assertEqual(len(caught), 1)
                self.assertIs(caught[0].category, RuntimeWarning)
                self.assertIn("loss and gradient are zero", str(caught[0].message))
                self.assertEqual(self.model.clf_distil_scope, "old_classes")
                self.assertEqual(float(self.model.clf_distil_loss_coef), 1.)

    def test_new_only_loss_and_gradient_are_exactly_zero_but_old_rows_are_active(self) -> None:
        """Old support [0,1] excludes [2,3], while old replay rows give a real gradient."""

        teacher = tf.constant([[.8, .2], [.3, .7]], dtype=tf.float32)
        for labels, should_be_active in (([2, 3], False), ([0, 1], True)):
            with self.subTest(labels=labels):
                logits = tf.Variable([[0., 1., 2., 3.], [1., 0., 3., 2.]], dtype=tf.float32)
                with tf.GradientTape() as tape:
                    loss, _ = self.model.compute_clf_distil_loss(
                        teacher, tf.nn.softmax(logits), classes=tf.constant(labels), 
                        student_logits=logits
                    )
                gradient = tape.gradient(loss, logits)
                # Old support receives the independently expected nonzero gradient.
                if should_be_active:
                    self.assertGreater(float(loss), 0.)
                    self.assertGreater(float(tf.linalg.global_norm([gradient])), 0.)
                # New-only rows are completely excluded by the declared scope.
                else:
                    self.assertEqual(float(loss), 0.)
                    np.testing.assert_array_equal(gradient.numpy(), np.zeros((2, 4)))

    def test_real_old_rows_and_replay_do_not_warn(self) -> None:
        """Cumulative real data and positive replay retain their declared exposure."""

        controls = (
            {"remove_prev_classes": False}, 
            {"use_generative_replay": True, "replay_old_examples": 4}
        )
        for options in controls:
            with self.subTest(options=options):
                self.assertEqual(self.preflight_warnings(**options), [])

    def test_all_row_scope_and_noise_objective_do_not_warn(self) -> None:
        """Classifier responses on new images and positive noise KD are alternatives."""

        self.model.clf_distil_scope = "current_and_replay"
        self.assertEqual(self.preflight_warnings(), [])
        self.model.clf_distil_scope = "old_classes"
        self.model.noise_distil_loss_coef = 1.
        self.assertEqual(self.preflight_warnings(), [])
        self.model.previous_teacher_noise_loss_weight = 0.
        self.assertEqual(len(self.preflight_warnings()), 1)

    def test_planned_current_teacher_must_have_an_effective_role_weight(self) -> None:
        """A real current objective suppresses disclosure; zero-weight roles do not."""

        self.assertEqual(self.preflight_warnings(dual_teacher_distillation=True), [])
        self.model.current_teacher_clf_loss_weight = 0.
        self.model.current_teacher_noise_loss_weight = 0.
        self.assertEqual(len(self.preflight_warnings(dual_teacher_distillation=True)), 1)

    def test_attached_current_and_classifier_specialists_remain_legal(self) -> None:
        """Shared current and per-head teachers can teach new labels independently."""

        teacher = self.model.network.from_config({
            **self.model.network.get_config(), "num_classes": 2
        })
        self.model.set_current_teacher_network(teacher)
        self.assertEqual(self.preflight_warnings(), [])
        self.model.set_current_teacher_network(None)
        self.model.set_classifier_teacher_network(teacher)
        self.assertEqual(self.preflight_warnings(), [])

    def test_attached_noise_specialist_remains_legal(self) -> None:
        """A current noise specialist is an alternative even with old noise KD off."""

        self.model.noise_distil_loss_coef = 1.
        self.model.previous_teacher_noise_loss_weight = 0.
        self.model.set_noise_teacher_network(tiny_network("dit"))
        self.assertEqual(self.preflight_warnings(), [])

    def test_persistent_teacher_can_learn_new_labels(self) -> None:
        """An every-task trained previous role is not a frozen old-only snapshot."""

        self.model.trainable_teacher = True
        self.model.teacher_training = "each_task"
        self.assertEqual(self.preflight_warnings(), [])


# Direct execution runs this focused regression module.
if __name__ == "__main__":
    unittest.main()
