"""Bounded native-bfloat16 continual fit past the initial weight fingerprint."""

import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.learner import _run_continual_tasks
from common.runtime import configure_runtime
from common.tests.test_continual_combination_matrix import tiny_network
from common.train import train_model
from diffusion import DiffusionClassifier


def _raw_cohorts(indices: list[int], **kwargs: object) -> tuple:
    """Return disjoint raw uint8 train/validation/test images for four original IDs."""

    del kwargs
    labels = np.repeat(np.asarray(indices, dtype=np.int32), 2)
    images = np.broadcast_to((32 + labels * 48)[:, None, None, None], 
                             (len(labels), 4, 4, 1)).astype(np.uint8).copy()
    return images, labels, images + np.uint8(1), labels.copy(), images + np.uint8(2), labels.copy()


class Bfloat16ContinualRepairTests(unittest.TestCase):
    """Verify one native-bfloat16 two-task treatment, not the full audited matrix."""

    def setUp(self) -> None:
        """Preserve the caller's precision policy before the isolated native treatment."""

        self.original_policy = tf.keras.mixed_precision.global_policy().name

    def tearDown(self) -> None:
        """Release model state and restore the previous global precision policy."""

        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy(self.original_policy)

    def test_two_tasks_reach_fitting_with_bfloat16_weights_and_frozen_noise_teacher(self) -> None:
        """Initial fingerprints permit two real one-step tasks and previous epsilon KD."""

        configure_runtime(dtype_policy="bfloat16", deterministic_ops=True, seed=953)
        model = DiffusionClassifier(
            network=tiny_network("dit", classifier=True), 
            use_ema=False, scheduler_name="clipped_cosine", test_steps=2, 
            preprocess_type="standardize", p_uncond=1., mask_by_nulls=False, 
            mask_by_t_threshold=False, defer_teacher=True, noise_loss_coef=1., 
            noise_distil_loss_coef=.1, clf_loss_coef=1., clf_distil_loss_coef=0., 
            previous_teacher_noise_loss_weight=1., current_teacher_noise_loss_weight=0., 
            seed=953
        )
        model.compile(optimizer=tf.keras.optimizers.SGD(.05), loss="mse", 
                      run_eagerly=True, jit_compile=False)
        self.assertEqual(model.dtype_policy.name, "bfloat16")
        self.assertTrue(any(str(variable.dtype) == "bfloat16" for variable in model.network.weights))
        observed = []

        def inspect_fit(*args: object, **kwargs: object) -> dict:
            """Inspect actual native updates and prove the previous teacher stays frozen."""

            fitted = args[1]
            self.assertIs(fitted, model)
            self.assertEqual(fitted.dtype_policy.name, "bfloat16")
            teacher = fitted.teacher_network
            self.assertEqual(teacher is not None, len(observed) == 1)
            previous = None
            # Only the second task has a completed prior snapshot.
            if teacher is not None:
                self.assertFalse(teacher.trainable)
                self.assertEqual(teacher.num_classes, 2)
                self.assertTrue(fitted.use_noise_distil_loss)
                self.assertTrue({id(variable) for variable in teacher.weights}.isdisjoint(
                    {id(variable) for variable in fitted.network.weights}
                ))
                previous = teacher.get_weights()
                images, labels = next(iter(args[2]))
                prepared = fitted.prep_inputs_map(images, labels)
                # Null conditioning keeps every new-task row eligible for the old noise teacher.
                self.assertTrue(bool(tf.reduce_all(prepared[-1])))
            before = fitted.network.get_weights()
            result = train_model(*args, **kwargs)
            after = fitted.network.get_weights()
            self.assertTrue(any(not np.array_equal(old, new) for old, new in zip(before, after)))
            self.assertTrue(all(np.isfinite(value.astype(np.float32)).all() for value in after))
            # A student update must never change the frozen distillation target.
            if teacher is not None:
                self.assertFalse(teacher.trainable)
                for old, new in zip(previous, teacher.get_weights()):
                    np.testing.assert_array_equal(old, new)
            observed.append(fitted.network.num_classes)
            return result

        with patch("common.train.train_model", side_effect=inspect_fit):
            result = _run_continual_tasks(
                class_num=4, class_order=[2, 0, 3, 1], task_size=2, 
                load_dataset_fn=_raw_cohorts, load_dataset_fn_kwargs={"preprocess": None}, 
                generative_model=model, use_generative_model_classifier=True, 
                generative_model_compile_args={"optimizer": tf.keras.optimizers.SGD(.05), 
                                               "loss": "mse", "run_eagerly": True}, 
                generative_model_kwargs={"train_num": -1, "samples_per_class": 2}, 
                use_generative_replay=False, use_distillation=True, 
                replay_budget_mode="fixed_total", replay_current_examples=4, replay_old_examples=0, 
                batch_size=4, epochs=1, optimizer_steps_per_epoch=1, callback_patience=0, 
                plot_results=False, deterministic_ops=True, show_generated_images=False, 
                show_network_summary=False, save_task_checkpoints=False, 
                experiment_phase="development", dtype_policy="bfloat16", verbose=0, seed=953
            )
        self.assertEqual(observed, [2, 4])
        self.assertEqual(result["task_classes"], [[2, 0], [3, 1]])
        self.assertFalse(result["test_evaluated"])
        matrix = np.asarray(result["validation_accuracy_matrix"])
        self.assertEqual(matrix.shape, (2, 2))
        self.assertTrue(np.isfinite(matrix[np.tril_indices(2)]).all())
        for task in result["task_resource_metrics"]:
            self.assertEqual(task["replay"]["selected_count"], 0)
            self.assertTrue(all(updates == 1 for updates in task["optimizer_updates"].values()))
        self.assertFalse(model.teacher_network.trainable)
        self.assertEqual(model.teacher_network.num_classes, 4)


# Direct execution runs the bounded continual integration regression.
if __name__ == "__main__":
    unittest.main()
