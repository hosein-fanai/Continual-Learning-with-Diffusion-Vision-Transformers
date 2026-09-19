"""Regression coverage for shared classifier caps and V1 input selection."""

import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import DiffusionClassifierConfig, DiffusionClassifierV2Config
from common.tests import test_clean_classifier_training as fixtures
from diffusion import DiffusionClassifier, DiffusionClassifierV2


class ClassifierNoisingCapsTests(unittest.TestCase):
    """Keep classifier corruption separate from diffusion and matched to teachers."""

    setUp = fixtures.CleanClassifierTrainingTests.setUp
    tearDown = fixtures.CleanClassifierTrainingTests.tearDown
    make_network = fixtures.CleanClassifierTrainingTests.make_network

    def make_wrapper(self, **overrides: object) -> DiffusionClassifier:
        """Use the existing tiny DiT with explicit noisy/all-class inputs."""
        options = dict(clf_train_noisy_input_type="noisy", clf_train_type="cond",
                       clf_train_class_input_type="all_classes",
                       clf_train_noisified_max_timesteps=2,
                       clf_test_noisified_max_timesteps=4)
        options.update(overrides)
        return fixtures.CleanClassifierTrainingTests.make_wrapper(self, **options)

    def test_shared_validation_normalization_and_config_round_trip(self) -> None:
        """Both wrappers inherit caps while preserving V2's coercion and sentinels."""
        for cls, config_cls in ((DiffusionClassifier, DiffusionClassifierConfig),
                                (DiffusionClassifierV2, DiffusionClassifierV2Config)):
            for cap, normalized in ((None, 0), (0, 0), (-1, 8), (2.9, 2), (True, 1)):
                with self.subTest(wrapper=cls.__name__, cap=cap):
                    options = dict(clf_train_noisified_max_timesteps=cap,
                                   clf_test_noisified_max_timesteps=cap)
                    wrapper = cls(network=self.make_network(), use_ema=False,
                                  mask_by_nulls=False, test_steps=4, **options)
                    self.assertEqual(wrapper.clf_train_noisified_max_timesteps, normalized)
                    self.assertEqual(wrapper.clf_test_noisified_max_timesteps, normalized)
                    config = wrapper.get_config()
                    self.assertEqual(config["clf_train_noisified_max_timesteps"], cap)
                    clone = cls.from_config(config)
                    self.assertEqual(clone.clf_test_noisified_max_timesteps, normalized)
                    self.assertEqual(config_cls(**options).kwargs()[
                        "clf_train_noisified_max_timesteps"], cap)
            for name in ("clf_train_noisified_max_timesteps", "clf_test_noisified_max_timesteps"):
                for invalid in (-2, 9):
                    with self.subTest(wrapper=cls.__name__, name=name, invalid=invalid):
                        with self.assertRaisesRegex(AssertionError, name):
                            cls(network=self.make_network(), use_ema=False,
                                mask_by_nulls=False, test_steps=4, **{name: invalid})

    def test_mapped_caps_select_classifier_inputs_without_changing_diffusion(self) -> None:
        """Cached caps reach real student predictions, including one-pass split batches."""
        for training, fraction, graph in ((True, 0., False), (True, 0., True),
                                          (True, .5, True), (False, 0., True)):
            with self.subTest(training=training, fraction=fraction, graph=graph):
                wrapper = self.make_wrapper(map_preprocess=True, clf_train_batch_fraction=fraction)
                wrapper.set_timestep_bounds(5, 8)
                wrapper._preprocess_training = training
                mapped = wrapper.prep_inputs_map(self.images, self.labels)
                wrapper._preprocess_training = None
                self.assertEqual(len(mapped), 9)
                self.assertTrue(bool(tf.reduce_all((mapped[2] >= 5) & (mapped[2] < 8))))
                self.assertTrue(bool(tf.reduce_all((mapped[8] >= 0) &
                                                  (mapped[8] < (2 if training else 4)))))
                allocation = tf.constant([True, False, False, True])
                original_forward = wrapper.forward
                original_predict = wrapper.network.predict_class

                def forward(name: str, x: tf.Tensor, t: tf.Tensor,
                            previous_t: tf.Tensor, **kwargs: object) -> tuple:
                    """Only classifier-owned rows may replace diffusion inputs."""
                    expected_x = tf.where(allocation[:, None, None, None], mapped[7], mapped[3]) \
                        if fraction else mapped[3]
                    expected_t = tf.where(allocation, mapped[8], mapped[2]) if fraction else mapped[2]
                    tf.debugging.assert_equal(x, expected_x)
                    tf.debugging.assert_equal(t, expected_t)
                    return original_forward(name, x, t, previous_t, **kwargs)

                def predict(inputs: tuple, **kwargs: object) -> tuple:
                    """The classifier consumes its cached corruption and correct conditions."""
                    self.assertFalse(fraction)
                    tf.debugging.assert_equal(inputs[0], mapped[7])
                    tf.debugging.assert_equal(inputs[1], mapped[8])
                    tf.debugging.assert_equal(inputs[2], mapped[4] if training else mapped[5])
                    return original_predict(inputs, **kwargs)

                with patch.object(wrapper, "forward", side_effect=forward), \
                     patch.object(wrapper.network, "predict_class", side_effect=predict), \
                     patch.object(wrapper, "_classifier_batch_mask", return_value=allocation), \
                     patch.object(wrapper, "noisify", side_effect=AssertionError("cached inputs re-noised")):
                    step = wrapper.train_step if training else wrapper.test_step
                    step = tf.function(step) if graph else step
                    result = step(mapped)
                self.assertTrue(all(bool(tf.math.is_finite(value)) for value in result.values()))
                self.assertEqual(int(wrapper.optimizer.iterations), int(training))

    def test_raw_steps_use_independent_train_and_test_caps(self) -> None:
        """Online preparation honors phase caps outside progressive diffusion bounds."""
        wrapper = self.make_wrapper()
        wrapper.set_timestep_bounds(5, 8)
        for training, cap in ((True, 2), (False, 4)):
            with self.subTest(training=training):
                with patch.object(wrapper, "noisify", wraps=wrapper.noisify) as noisify:
                    result = (wrapper.train_step if training else wrapper.test_step)((self.images, self.labels))
                self.assertEqual(noisify.call_count, 2)
                self.assertEqual(noisify.call_args.kwargs, dict(min_timesteps=0, max_timesteps=cap))
                self.assertTrue(all(bool(tf.math.is_finite(value)) for value in result.values()))

    def test_clean_policy_ignores_caps_and_none_preserves_v1_defaults(self) -> None:
        """Inactive caps do not add corruption or alter legacy mapped tuple layouts."""
        for input_type, cap in (("clean", 2), ("noisy", None)):
            wrapper = self.make_wrapper(clf_train_noisy_input_type=input_type,
                                        clf_train_noisified_max_timesteps=cap,
                                        clf_test_noisified_max_timesteps=cap, map_preprocess=True)
            for training in (True, False):
                with self.subTest(input_type=input_type, training=training):
                    wrapper._preprocess_training = training
                    with patch.object(wrapper, "noisify", wraps=wrapper.noisify) as noisify:
                        mapped = wrapper.prep_inputs_map(self.images, self.labels)
                    self.assertEqual(noisify.call_count, 1)
                    self.assertEqual(len(mapped), 7)
                    expected_x = mapped[3] if training and input_type == "noisy" else mapped[0]
                    expected_t = mapped[2] if training and input_type == "noisy" else tf.zeros_like(mapped[2])
                    actual_x, actual_t = wrapper._classifier_inputs(mapped, training)
                    np.testing.assert_array_equal(actual_x, expected_x)
                    np.testing.assert_array_equal(actual_t, expected_t)

    def test_distillation_caches_exact_classifier_inputs_and_replay_provenance(self) -> None:
        """Teacher and student share one draw with and without noise distillation."""
        replay = tf.constant([True, False, True, False])
        for training in (True, False):
            for noise_distil in (False, True):
                with self.subTest(training=training, noise_distil=noise_distil):
                    wrapper = self.make_wrapper(map_preprocess=True, train_cfg_scale=2., test_cfg_scale=3.)
                    wrapper._preprocess_training = training
                    teacher_inputs = []

                    def predict(x: tf.Tensor, t: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
                        """Capture the actual teacher image and timestep draw."""
                        teacher_inputs.append((x, t, labels))
                        return tf.constant([[.3, .7]] * 4)

                    with patch.object(wrapper, "use_classifier_distil", True), \
                         patch.object(wrapper, "use_noise_distil_loss", noise_distil), \
                         patch.object(wrapper, "_predict_teacher_labels", side_effect=predict), \
                         patch.object(wrapper, "forward", return_value=(
                             None, tf.zeros_like(self.images), None, None,
                             (tf.constant([[.9, .1]] * 4), None))):
                        mapped = wrapper.prep_inputs_map(self.images, self.labels, replay)
                        prepared, teacher, provenance = wrapper._prepare_classifier_batch(
                            mapped, use_label_dropout=training)
                    self.assertEqual(len(teacher_inputs), 1)
                    np.testing.assert_array_equal(teacher_inputs[0][0], prepared[7])
                    np.testing.assert_array_equal(teacher_inputs[0][1], prepared[8])
                    np.testing.assert_array_equal(teacher_inputs[0][2], prepared[4 if training else 5])
                    np.testing.assert_array_equal(provenance, replay)
                    np.testing.assert_array_equal(teacher, tf.constant([[.3, .7]] * 4))
                    self.assertEqual(len(prepared), 9 + 2 * int(noise_distil))

    def test_explicit_v1_caps_select_clean_or_full_horizon_inputs(self) -> None:
        """Explicit zero differs from omitted caps; -1 overrides progressive bounds."""
        for cap in (0, -1):
            wrapper = self.make_wrapper(clf_train_noisified_max_timesteps=cap,
                                        clf_test_noisified_max_timesteps=cap)
            wrapper.set_timestep_bounds(5, 8)
            for training in (True, False):
                with self.subTest(cap=cap, training=training):
                    wrapper._preprocess_training = training
                    with patch.object(wrapper, "noisify", wraps=wrapper.noisify) as noisify:
                        mapped = wrapper.prep_inputs_map(self.images, self.labels)
                    self.assertEqual(noisify.call_args.kwargs,
                                     dict(min_timesteps=0, max_timesteps=8 if cap == -1 else 0))
                    # Zero is a clean-image sentinel even when the diffusion batch is noisy.
                    if cap == 0:
                        np.testing.assert_array_equal(mapped[7], self.images)
                        np.testing.assert_array_equal(mapped[8], np.zeros(4))

    def test_ensemble_restriction_is_v1_only(self) -> None:
        """V1 cannot silently bypass explicit caps; V2 retains its ensemble policy."""
        with self.assertRaisesRegex(AssertionError, "ensemble"):
            self.make_wrapper(use_ensemble_loss_instead=True)
        wrapper = DiffusionClassifierV2(network=self.make_network(), use_ema=False,
                                        test_steps=4, clf_train_noisified_max_timesteps=2,
                                        use_ensemble_loss_instead=True)
        self.assertTrue(wrapper.use_ensemble_loss_instead)

    def test_v2_caps_keep_their_phase_semantics(self) -> None:
        """V2 still treats None/zero as clean and ignores progressive generator bounds."""
        for cap in (None, 0, -1, 3):
            with self.subTest(cap=cap):
                wrapper = DiffusionClassifierV2(network=self.make_network(), use_ema=False,
                                                test_steps=4, clf_train_noisified_max_timesteps=cap)
                wrapper.set_timestep_bounds(5, 8)
                t, x, nulls, labels = wrapper.prep_clfv2_inputs(
                    (self.images, self.labels), wrapper.clf_train_noisified_max_timesteps)
                self.assertTrue(bool(tf.reduce_all(t >= 0)))
                # A zero cap preserves clean pixels exactly rather than sampling timestep zero.
                if cap in (None, 0):
                    np.testing.assert_array_equal(x, self.images)
                    np.testing.assert_array_equal(t, np.zeros(4))
                # Positive and full-horizon caps remain exclusive classifier bounds.
                else:
                    self.assertTrue(bool(tf.reduce_all(t < (8 if cap == -1 else cap))))
                np.testing.assert_array_equal(nulls, np.zeros(4))
                np.testing.assert_array_equal(labels, self.labels)


# Support direct execution as well as unittest discovery.
if __name__ == "__main__":
    unittest.main()
