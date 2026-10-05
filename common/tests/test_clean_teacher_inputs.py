"""Verify clean teacher inputs independently of student corruption."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import DiffusionClassifierConfig, DiffusionClassifierV2Config
from common.tests import test_clean_classifier_training as fixtures
from diffusion import DiffusionClassifier, DiffusionClassifierV2


class CleanTeacherInputTests(unittest.TestCase):
    """Keep classifier targets clean without changing student or epsilon inputs."""

    setUp = fixtures.CleanClassifierTrainingTests.setUp
    tearDown = fixtures.CleanClassifierTrainingTests.tearDown
    make_network = fixtures.CleanClassifierTrainingTests.make_network
    prepared_batch = fixtures.CleanClassifierTrainingTests.prepared_batch

    def make_wrapper(self, wrapper_cls: type = DiffusionClassifier, **overrides: object) -> DiffusionClassifier:
        """Build a tiny noisy student with a native classifier teacher."""

        options = dict(
            network=self.make_network(), teacher_network=self.make_network(), 
            use_ema=False, preprocess_type=None, test_steps=4, p_uncond=0., 
            mask_by_nulls=False, mask_by_t_threshold=False, clf_distil_loss_coef=1., 
            clf_train_noisified_max_timesteps=-1 if wrapper_cls is DiffusionClassifierV2 else None, 
            seed=811
        )
        options.update(overrides)
        return wrapper_cls(**options)

    def test_graph_mapping_preserves_student_noise_for_both_wrappers(self) -> None:
        """Native teachers honor clean/default modes while mapped students stay noisy."""

        for wrapper_cls in (DiffusionClassifier, DiffusionClassifierV2):
            for mode in ("noisy", "clean"):
                with self.subTest(wrapper=wrapper_cls.__name__, mode=mode):
                    wrapper = self.make_wrapper(wrapper_cls, clf_distil_noisy_input_type=mode)
                    prepared = self.prepared_batch(wrapper)
                    source, preparation = prepared, "prep_inputs"
                    time_index, image_index = 2, 3
                    conditions = prepared[4]
                    # V2 maps its separate null-conditioned classifier batch.
                    if wrapper_cls is DiffusionClassifierV2:
                        wrapper._test_part = "discriminator"
                        source = (prepared[2], prepared[3], prepared[5], self.labels, prepared[0])
                        preparation = "prep_clfv2_inputs"
                        time_index, image_index = 0, 1
                        conditions = prepared[5]
                    expected_x = self.images if mode == "clean" else prepared[3]
                    expected_t = tf.zeros_like(prepared[2]) if mode == "clean" else prepared[2]
                    probabilities = tf.constant([[0.2, 0.8]] * 4)

                    def predict(inputs: tuple[tf.Tensor, ...], **kwargs: object) -> tf.Tensor:
                        """Check actual native teacher inputs during graph execution."""

                        tf.debugging.assert_equal(inputs[0], expected_x)
                        tf.debugging.assert_equal(inputs[1], expected_t)
                        tf.debugging.assert_equal(inputs[2], conditions)
                        self.assertFalse(kwargs["training"])
                        return probabilities

                    with patch.object(wrapper, preparation, return_value=source), \
                         patch.object(wrapper.teacher_network, "predict_class", side_effect=predict):
                        mapped = tf.function(wrapper.prep_inputs_map)(self.images, self.labels)
                    np.testing.assert_array_equal(mapped[time_index], prepared[2])
                    np.testing.assert_array_equal(mapped[image_index], prepared[3])
                    np.testing.assert_array_equal(mapped[-1], probabilities)

    def test_combined_targets_keep_epsilon_noisy(self) -> None:
        """Clean or independently capped targets require a separate native teacher pass."""

        for mode, cap in (("noisy", None), ("clean", None), ("noisy", 0)):
            with self.subTest(mode=mode, cap=cap):
                wrapper = self.make_wrapper(
                    clf_distil_noisy_input_type=mode, clf_distil_train_noisified_max_timesteps=cap, 
                    noise_distil_loss_coef=1., train_cfg_scale=2.
                )
                prepared = self.prepared_batch(wrapper)
                shared = tf.constant([[0.9, 0.1]] * 4)
                clean = tf.constant([[0.2, 0.8]] * 4)
                epsilon = tf.ones_like(self.images)

                def predict(inputs: tuple[tf.Tensor, ...], **kwargs: object) -> tf.Tensor:
                    """Require exact clean images and zero timesteps for class targets."""

                    tf.debugging.assert_equal(inputs[0], self.images)
                    tf.debugging.assert_equal(inputs[1], tf.zeros_like(prepared[2]))
                    tf.debugging.assert_equal(inputs[2], prepared[4])
                    self.assertFalse(kwargs["training"])
                    return clean

                def forward(
                    network_name: str, noisy: tf.Tensor, times: tf.Tensor, 
                    previous_times: tf.Tensor, **kwargs: object
                ) -> tuple:
                    """Keep epsilon inputs and class conditioning at the student timestep."""

                    self.assertEqual(network_name, "teacher")
                    tf.debugging.assert_equal(noisy, prepared[3])
                    tf.debugging.assert_equal(times, prepared[2])
                    tf.debugging.assert_equal(previous_times, prepared[2])
                    tf.debugging.assert_equal(kwargs["cond_labels"], prepared[4])
                    self.assertEqual(kwargs["scale"], 2.)
                    self.assertFalse(kwargs["training"])
                    return (None, epsilon, None, None, (shared, shared))

                with patch.object(wrapper, "prep_inputs", return_value=prepared), \
                     patch.object(wrapper, "forward", side_effect=forward) as noise_prediction, \
                     patch.object(wrapper.teacher_network, "predict_class", side_effect=predict) as class_prediction:
                    mapped = wrapper.prep_inputs_map(self.images, self.labels)
                self.assertEqual(noise_prediction.call_count, 1)
                self.assertEqual(class_prediction.call_count, int(mode == "clean" or cap is not None))
                np.testing.assert_array_equal(mapped[2], prepared[2])
                np.testing.assert_array_equal(mapped[3], prepared[3])
                np.testing.assert_array_equal(mapped[-3], epsilon)
                np.testing.assert_array_equal(mapped[-1], clean if mode == "clean" or cap is not None else shared)

    def test_defaults_round_trips_and_invalid_or_missing_inputs(self) -> None:
        """Persist the opt-in mode for both wrappers and reject unsupported contracts."""

        for wrapper_cls, config_cls in ((DiffusionClassifier, DiffusionClassifierConfig), 
                                        (DiffusionClassifierV2, DiffusionClassifierV2Config)):
            with self.subTest(wrapper=wrapper_cls.__name__):
                self.assertEqual(config_cls().clf_distil_noisy_input_type, "noisy")
                wrapper = self.make_wrapper(wrapper_cls)
                self.assertEqual(wrapper.clf_distil_noisy_input_type, "noisy")
                configured = self.make_wrapper(wrapper_cls, clf_distil_noisy_input_type="clean")
                clone = wrapper_cls.from_config(json.loads(json.dumps(configured.get_config())))
                self.assertEqual(clone.clf_distil_noisy_input_type, "clean")
                self.assertEqual(config_cls(clf_distil_noisy_input_type="clean").kwargs()[
                    "clf_distil_noisy_input_type"], "clean")
                with self.assertRaisesRegex(ValueError, "clean_images"):
                    configured._predict_single_teacher_labels(
                        self.images + 0.5, tf.ones_like(self.labels), self.labels
                    )
        with self.assertRaisesRegex(AssertionError, "clf_distil_noisy_input_type"):
            self.make_wrapper(clf_distil_noisy_input_type="unknown")


# Support direct focused execution without running tests on import.
if __name__ == "__main__":
    unittest.main()
