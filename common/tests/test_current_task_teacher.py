"""Current-task teacher mapping and effective objective selection contracts."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.current_task_teacher import make_current_task_teacher, student_task_class_ids
from common.learner import (
    _has_positive_distillation_objective, _predict_teacher_probabilities, 
    _sample_diffusion_replay
)
from common.tests import test_dual_teachers as dual_fixtures
from diffusion import DiffusionClassifier


class CurrentTaskTeacherTests(unittest.TestCase):
    """Keep dataset IDs distinct from raw teacher and student output columns."""

    def setUp(self) -> None:
        """Create the existing small deterministic TensorFlow fixture."""

        self.fixture = dual_fixtures.DualTeacherTests()
        self.fixture.setUp()

    def tearDown(self) -> None:
        """Restore the precision policy and release the test's Keras state."""

        self.fixture.tearDown()

    def student_with_mapping(self, labels: list[int]) -> DiffusionClassifier:
        """Discover one dataset class at a time to preserve a nontrivial order."""

        student = self.fixture.continual_model(DiffusionClassifier)
        for label in labels:
            student._check_new_labels(y=np.asarray([label]), verbose=0)
        return student

    def test_fresh_teacher_maps_local_columns_to_nonidentity_student_columns(self) -> None:
        """Fresh heads learn dataset IDs locally and report the student's column IDs."""

        student = self.student_with_mapping([1, 0, 3, 2])
        teacher, columns = make_current_task_teacher(student, [2, 3], seed=113)
        self.assertEqual(dict(student.seen_classes), {1: 0, 0: 1, 3: 2, 2: 3})
        self.assertEqual(dict(teacher.seen_classes), {2: 0, 3: 1})
        self.assertEqual(columns, [3, 2])
        self.assertEqual(teacher.network.num_classes, 2)

    def test_warm_teacher_preserves_full_column_alignment_and_dataset_mapping(self) -> None:
        """Copied full heads retain their source vocabulary and independent variables."""

        student = self.student_with_mapping([1, 0, 3, 2])
        teacher, columns = make_current_task_teacher(
            student, [2, 3], initialization="student", seed=113
        )
        self.assertEqual(columns, [0, 1, 2, 3])
        self.assertEqual(student_task_class_ids(student, [2, 3]), [3, 2])
        self.assertEqual(dict(teacher.seen_classes), dict(student.seen_classes))
        self.assertFalse({id(value) for value in student.network.weights}
                         & {id(value) for value in teacher.network.weights})
        for expected, actual in zip(student.network.get_weights(), teacher.network.get_weights()):
            np.testing.assert_array_equal(actual, expected)

    def test_dataset_ids_larger_than_head_width_use_existing_mapping(self) -> None:
        """Observed dataset labels need not themselves be valid softmax columns."""

        student = self.student_with_mapping([7, 11])
        teacher, columns = make_current_task_teacher(student, [11], seed=113)
        self.assertEqual(columns, [1])
        self.assertEqual(dict(teacher.seen_classes), {11: 0})
        self.assertEqual(teacher.network.num_classes, 1)

    def test_current_only_objective_requires_an_attached_or_scheduled_current_teacher(self) -> None:
        """Future current teachers count only when the dual lifecycle will build them."""

        student = self.student_with_mapping([0, 1])
        student.previous_teacher_noise_loss_weight = 0.
        student.previous_teacher_clf_loss_weight = 0.
        self.assertFalse(_has_positive_distillation_objective(student))
        self.assertTrue(_has_positive_distillation_objective(
            student, dual_teacher_distillation=True
        ))
        student.current_teacher_noise_loss_weight = 0.
        student.current_teacher_clf_loss_weight = 0.
        self.assertFalse(_has_positive_distillation_objective(
            student, dual_teacher_distillation=True
        ))

    def test_replay_sampling_maps_dataset_labels_before_cfg_offset(self) -> None:
        """Replay images remain aligned with their dataset labels after vocabulary reordering."""

        student = self.student_with_mapping([1, 0, 3, 2])
        with patch.object(student, "sample", side_effect=lambda **kwargs: tf.convert_to_tensor(
            np.asarray(kwargs["labels"])[:, None], dtype=tf.float32
        )) as sample:
            actual = _sample_diffusion_replay(
                student, np.asarray([2, 0, 3]), batch_size=2, seed=113, 
                empty_samples=np.empty((0, 1), dtype="float32")
            )
        np.testing.assert_array_equal(actual[:, 0], [4, 2, 3])
        self.assertEqual(sample.call_count, 2)

    def test_teacher_scoring_returns_dense_dataset_column_order(self) -> None:
        """Snapshot and live-student probabilities use the same dataset-label order."""

        mapping = {1: 0, 0: 1, 3: 2, 2: 3}
        teacher = SimpleNamespace(
            num_classes=4, _diffusion_seen_classes=mapping, 
            predict_class=lambda inputs, **kwargs: np.repeat(
                [[.1, .2, .3, .4]], len(inputs[0]), axis=0
            )
        )
        images = np.zeros((3, 4, 4, 1), dtype="float32")
        actual = _predict_teacher_probabilities(teacher, images, -1., 2., 2)
        np.testing.assert_allclose(actual, [[.2, .1, .4, .3]] * 3)
        actual = _predict_teacher_probabilities(
            teacher, images, -1., 2., 2, class_mapping=mapping
        )
        np.testing.assert_allclose(actual, [[.2, .1, .4, .3]] * 3)
        empty = _predict_teacher_probabilities(teacher, images[:0], -1., 2., 2)
        self.assertEqual(empty.shape, (0, 4))
        with self.assertRaisesRegex(ValueError, "complete dense"):
            _predict_teacher_probabilities(
                teacher, images, -1., 2., 2, class_mapping={0: 0, 1: 0, 2: 2, 3: 3}
            )

    def test_positive_previous_role_remains_an_effective_future_snapshot_objective(self) -> None:
        """The first task may defer its previous snapshot until the task boundary."""

        student = self.student_with_mapping([0, 1])
        student.current_teacher_noise_loss_weight = 0.
        student.current_teacher_clf_loss_weight = 0.
        self.assertIsNone(student.teacher_network)
        self.assertTrue(_has_positive_distillation_objective(student))


# Direct invocation uses the same focused unittest cases as module discovery.
if __name__ == "__main__":
    unittest.main()
