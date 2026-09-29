"""Task-boundary recovery for independently trained diffusion teachers."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from common.learner import _run_continual_tasks
from common.runtime import configure_runtime
from common.tests import test_fit_teacher as teacher_fixtures


class TeacherRecoveryTests(unittest.TestCase):
    """Compare continued teacher growth with uninterrupted task training."""

    def setUp(self) -> None:
        """Reuse the small deterministic teacher fixture without copying model setup."""

        self.fixture = teacher_fixtures.FitTeacherTests()
        self.fixture.setUp()

    def tearDown(self) -> None:
        """Release fixture models and restore the original numerical policy."""

        self.fixture.tearDown()

    def recover_teacher(self, policy: str, use_distillation: bool) -> object:
        """Compare exact recovery under deterministic kernels and one teacher policy.

        Args:
            policy (str): Constructor-selected each_task or first_task teacher schedule.
            use_distillation (bool): Enable the student's teacher losses and snapshots.

        Returns:
            model (object): Resumed wrapper after exact model and optimizer comparisons.
        """

        options = dict(
            class_num=4, task_size=2, load_dataset_fn=self.fixture.continual_loader, 
            load_dataset_fn_kwargs={"preprocess": "diffusion"}, 
            use_generative_model_classifier=True, 
            generative_model_kwargs={"train_num": -1}, 
            use_generative_replay=False, use_distillation=use_distillation, 
            batch_size=8, epochs=1, optimizer_steps_per_epoch=1, 
            callback_patience=0, plot_results=False, verbose=0, 
            seed=541, deterministic_ops=True, 
            show_generated_images=False, show_network_summary=False, 
            save_task_checkpoints=True, fit_method="fit_progressively", 
            fit_kwargs={"stage_tasks": [("depth", "vision_transformer_block")], 
                        "stage_epochs": 1, "final_epochs": 1, "stages_verbose": False}
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = []
            for resume in (False, True):
                tf.keras.backend.clear_session()
                configure_runtime(541, "float32", deterministic_ops=True)
                teacher = self.fixture.make_network(True, None)
                teacher.add_depths("vision_transformer_block")
                model = self.fixture.make_wrapper(
                    True, num_classes=None, teacher_network=teacher, 
                    teacher_training=policy, noise_distil_loss_coef=.1 if use_distillation else 0.
                )
                run_options = dict(options, generative_model=model, 
                                   checkpoint_dir=str(root / ("resumed" if resume else "original")))
                # Continue from the first committed task into an independent output root.
                if resume:
                    run_options["resume_from"] = str(root / "original" / "task-0000")
                _run_continual_tasks(**run_options)
                models.append(model)
            original, resumed = models
            self.assertEqual(resumed.network.depth, 3)
            self.assertFalse(resumed.teacher_network.trainable)
            for expected_model, actual_model in (
                (original, resumed), (original._teacher_model, resumed._teacher_model)
            ):
                self.assertEqual(len(expected_model.weights), len(actual_model.weights))
                self.assertEqual(len(expected_model.optimizer.variables), len(actual_model.optimizer.variables))
                for expected, actual in zip(expected_model.weights, actual_model.weights):
                    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
                for expected, actual in zip(expected_model.optimizer.variables, actual_model.optimizer.variables):
                    np.testing.assert_array_equal(actual.numpy(), expected.numpy())

            return resumed

    def test_persistent_teacher_resumes_vocabulary_depth_and_optimizer(self) -> None:
        """Recover a wider teacher and both optimizers before the next growing task."""

        resumed = self.recover_teacher("each_task", use_distillation=True)
        self.assertEqual(resumed._teacher_model.seen_classes, {0: 0, 1: 1, 2: 2, 3: 3})
        self.assertEqual(resumed.teacher_network.depth, 4)
        self.assertEqual(int(resumed._teacher_model.optimizer.iterations.numpy()), 4)

    def test_first_task_teacher_is_preserved_without_student_snapshots(self) -> None:
        """Restore an independently pretrained teacher without fitting it again."""

        resumed = self.recover_teacher("first_task", use_distillation=False)
        self.assertEqual(resumed._teacher_model.seen_classes, {0: 0, 1: 1})
        self.assertEqual(resumed.teacher_network.depth, 3)
        self.assertEqual(int(resumed._teacher_model.optimizer.iterations.numpy()), 2)


# Run only these recovery regressions when this module is executed directly.
if __name__ == "__main__":
    unittest.main()
