"""Check classifier-teacher corruption caps without changing student inputs."""

import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import DiffusionClassifierConfig, DiffusionClassifierV2Config
from common.tests import test_clean_teacher_inputs as fixtures
from diffusion import DiffusionClassifier, DiffusionClassifierV2


class TeacherNoisingCapsTests(unittest.TestCase):
    """Keep teacher caps independent of phase-specific student corruption."""

    setUp = fixtures.CleanTeacherInputTests.setUp
    tearDown = fixtures.CleanTeacherInputTests.tearDown
    make_network = fixtures.CleanTeacherInputTests.make_network
    make_wrapper = fixtures.CleanTeacherInputTests.make_wrapper
    prepared_batch = fixtures.CleanTeacherInputTests.prepared_batch

    def test_mapped_train_and_test_caps_preserve_v1_and_v2_student_inputs(self) -> None:
        """Graph mapping chooses the phase cap independently of diffusion bounds."""

        for wrapper_cls in (DiffusionClassifier, DiffusionClassifierV2):
            wrapper = self.make_wrapper(wrapper_cls, clf_distil_train_noisified_max_timesteps=2, 
                                        clf_distil_test_noisified_max_timesteps=4)
            wrapper.set_timestep_bounds(5, 8)
            prepared = self.prepared_batch(wrapper)
            source, preparation = prepared, "prep_inputs"
            time_index, image_index = 2, 3
            # V2 maps a separate classifier tuple with null conditioning.
            if wrapper_cls is DiffusionClassifierV2:
                wrapper._test_part = "discriminator"
                source = (prepared[2], prepared[3], prepared[5], self.labels, prepared[0])
                preparation, time_index, image_index = "prep_clfv2_inputs", 0, 1
            for training, cap in ((True, 2), (False, 4)):
                with self.subTest(wrapper=wrapper_cls.__name__, training=training):
                    wrapper._preprocess_training = training

                    def predict(inputs: tuple, **kwargs: object) -> tf.Tensor:
                        """Check actual teacher times obey the selected exclusive cap."""

                        tf.debugging.assert_greater_equal(inputs[1], 0)
                        tf.debugging.assert_less(inputs[1], cap)
                        self.assertFalse(kwargs["training"])
                        return tf.constant([[.3, .7]] * 4)

                    with patch.object(wrapper, preparation, return_value=source), \
                         patch.object(wrapper.teacher_network, "predict_class", side_effect=predict), \
                         patch.object(wrapper, "noisify", wraps=wrapper.noisify) as noisify:
                        mapped = tf.function(wrapper.prep_inputs_map)(self.images, self.labels)
                    self.assertEqual(noisify.call_count, 1)
                    self.assertEqual(noisify.call_args.kwargs, dict(min_timesteps=0, max_timesteps=cap))
                    np.testing.assert_array_equal(mapped[time_index], prepared[2])
                    np.testing.assert_array_equal(mapped[image_index], prepared[3])

    def test_raw_v2_evaluation_uses_the_teacher_test_cap(self) -> None:
        """Raw discriminator evaluation selects clean teacher targets and noisy students."""

        wrapper = self.make_wrapper(
            DiffusionClassifierV2, map_preprocess=False, clf_test_noisified_max_timesteps=3, 
            clf_distil_train_noisified_max_timesteps=2, clf_distil_test_noisified_max_timesteps=0
        )
        wrapper.compile(optimizer=tf.keras.optimizers.SGD(.01), loss="mse", jit_compile=False)
        wrapper._test_part = "discriminator"
        original_predict = wrapper.network.predict_class

        def teacher_predict(inputs: tuple, **kwargs: object) -> tf.Tensor:
            """Assert exact clean teacher images even while the student is corrupted."""

            tf.debugging.assert_equal(inputs[0], self.images)
            tf.debugging.assert_equal(inputs[1], tf.zeros_like(self.labels))
            self.assertFalse(kwargs["training"])
            return tf.constant([[.3, .7]] * 4)

        def student_predict(inputs: tuple, **kwargs: object) -> tuple:
            """Keep the classifier's own evaluation corruption unchanged."""

            tf.debugging.assert_greater(tf.reduce_max(tf.abs(inputs[0] - self.images)), 0.)
            tf.debugging.assert_less(inputs[1], 3)
            return original_predict(inputs, **kwargs)

        with patch.object(wrapper.teacher_network, "predict_class", side_effect=teacher_predict), \
             patch.object(wrapper.network, "predict_class", side_effect=student_predict):
            result = tf.function(wrapper.test_step)((self.images, self.labels))
        self.assertTrue(all(np.isfinite(float(value)) for value in result.values()))

    def test_clean_and_ordinary_teacher_policies_ignore_caps_without_noising(self) -> None:
        """Ignored teacher caps must not advance corruption streams."""

        def ordinary_teacher(images: tf.Tensor, training: bool = False) -> tf.Tensor:
            """Require original clean pixels and inference for an image-only teacher."""

            tf.debugging.assert_equal(images, self.images)
            self.assertFalse(training)
            return tf.constant([[.3, .7]] * 4)

        for ordinary in (False, True):
            with self.subTest(ordinary=ordinary):
                wrapper = self.make_wrapper(
                    teacher_network=ordinary_teacher if ordinary else self.make_network(), 
                    clf_distil_noisy_input_type="noisy" if ordinary else "clean", 
                    clf_distil_train_noisified_max_timesteps=2
                )
                with patch.object(wrapper, "noisify", side_effect=AssertionError("unexpected noising")):
                    target = wrapper._predict_teacher_labels(
                        self.images + .5, tf.fill(tuple([4]), 6), self.labels, clean_images=self.images
                    )
                self.assertEqual(target.shape, (4, 2))

    def test_sentinels_share_one_draw_and_preserve_config(self) -> None:
        """Zero is exact clean and -1 spans the schedule once for both native roles."""

        names = ("clf_distil_train_noisified_max_timesteps", "clf_distil_test_noisified_max_timesteps")
        for config_cls in (DiffusionClassifierConfig, DiffusionClassifierV2Config):
            for name in names:
                self.assertIsNone(getattr(config_cls(), name))
        for cap, normalized in ((0, 0), (-1, 8)):
            with self.subTest(cap=cap):
                wrapper = self.make_wrapper(clf_distil_train_noisified_max_timesteps=cap, 
                                            clf_distil_test_noisified_max_timesteps=cap)
                wrapper.set_timestep_bounds(5, 8)
                wrapper.set_current_teacher_network(self.make_network(), class_ids=[0, 1])
                seen = []

                def predict(inputs: tuple, **kwargs: object) -> tf.Tensor:
                    """Capture native role inputs and check the sentinel's image contract."""

                    seen.append(inputs)
                    tf.debugging.assert_greater_equal(inputs[1], 0)
                    # Zero cap must preserve pixels exactly, not merely sample schedule time zero.
                    if cap == 0:
                        tf.debugging.assert_equal(inputs[0], self.images)
                        tf.debugging.assert_equal(inputs[1], tf.zeros_like(self.labels))
                    # The full schedule overrides the student's active diffusion interval.
                    else:
                        tf.debugging.assert_less(inputs[1], normalized)
                    return tf.constant([[.3, .7]] * 4)

                with patch.object(wrapper.teacher_network, "predict_class", side_effect=predict), \
                     patch.object(wrapper.current_teacher_network, "predict_class", side_effect=predict), \
                     patch.object(wrapper, "noisify", wraps=wrapper.noisify) as noisify:
                    targets = wrapper._predict_teacher_labels(
                        self.images + .5, tf.fill(tuple([4]), 6), self.labels, clean_images=self.images
                    )
                self.assertEqual(noisify.call_count, 1)
                self.assertEqual(noisify.call_args.kwargs, dict(min_timesteps=0, max_timesteps=normalized))
                self.assertEqual(len(targets), 2)
                np.testing.assert_array_equal(seen[0][0], seen[1][0])
                np.testing.assert_array_equal(seen[0][1], seen[1][1])
                config = wrapper.get_config()
                clone = DiffusionClassifier.from_config(config)
                for name in names:
                    self.assertEqual(config[name], cap)
                    self.assertEqual(getattr(clone, name), normalized)
                    self.assertEqual(DiffusionClassifierConfig(**{name: cap}).kwargs()[name], cap)
        for name, invalid in zip(names, (-2, 9)):
            with self.assertRaisesRegex(AssertionError, name):
                self.make_wrapper(**{name: invalid})


# Support direct focused execution without running tests on import.
if __name__ == "__main__":
    unittest.main()
