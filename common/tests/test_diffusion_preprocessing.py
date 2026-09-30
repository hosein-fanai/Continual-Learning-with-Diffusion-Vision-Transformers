"""Regression coverage for wrapper-owned diffusion image coordinates."""

import inspect
import json
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import DiffusionModelConfig, DiffusionClassifierConfig, DiffusionClassifierV2Config
from common.tests import test_clean_classifier_training as fixtures
from diffusion import DiffusionModel, DiffusionClassifier, DiffusionClassifierV2
from diffusion.metrics.ensemble_accuracy import EnsembleAccuracy


class DiffusionPreprocessingTests(unittest.TestCase):
    """Keep conversion public, invertible, serializable, and applied exactly once."""

    setUp = fixtures.CleanClassifierTrainingTests.setUp
    tearDown = fixtures.CleanClassifierTrainingTests.tearDown
    make_network = fixtures.CleanClassifierTrainingTests.make_network

    def make_wrapper(self, wrapper_cls: type = DiffusionModel, **options: object) -> DiffusionModel:
        """Build a tiny wrapper using its raw-pixel default input contract."""

        wrapper = wrapper_cls(
            network=self.make_network(), use_ema=False, test_steps=2, 
            train_noisified_max_timesteps=0, test_noisified_max_timesteps=0, 
            **options
        )
        wrapper.set_timestep_bounds(0, 0)
        return wrapper

    def test_default_and_overrides_are_fixed_inverse_transforms(self) -> None:
        """Type overrides use fixed pixel bounds without mutating the saved default."""

        wrapper = self.make_wrapper()
        pixels = tf.constant([0., 63.75, 127.5, 255.])
        for mode, expected in (
            ("standardize", [-1., -.5, 0., 1.]), 
            ("min-max", [0., .25, .5, 1.])
        ):
            with self.subTest(mode=mode):
                converted = wrapper.preprocess(pixels, mode)
                np.testing.assert_allclose(converted, expected, atol=1e-6)
                np.testing.assert_allclose(wrapper.postprocess(converted, mode), pixels)
        self.assertEqual(wrapper.preprocess_type, "standardize")
        np.testing.assert_allclose(wrapper.preprocess(pixels), [-1., -.5, 0., 1.])
        # Values outside the pixel interval retain their affine meaning unless clipping is explicit.
        np.testing.assert_allclose(wrapper.postprocess([-3., 3.]), [-255., 510.])
        np.testing.assert_allclose(wrapper.postprocess([-3., 3.], clip=True), [0., 255.])
        constant = wrapper.preprocess(tf.fill((2, 4, 4, 1), 127.5))
        np.testing.assert_array_equal(constant, np.zeros((2, 4, 4, 1)))

    def test_passthrough_and_call_default_resolve_constructor_setting(self) -> None:
        """None on either method inherits its constructor's explicit process choice."""

        for mode, expected in ((None, [0., 255.]), ("min-max", [0., 1.])):
            with self.subTest(mode=mode):
                wrapper = self.make_wrapper(preprocess_type=mode)
                np.testing.assert_allclose(wrapper.preprocess([0., 255.], None), expected)
                np.testing.assert_allclose(wrapper.postprocess(expected, None), [0., 255.])
        self.assertIsNone(inspect.signature(DiffusionModel.preprocess).parameters["preprocess_type"].default)
        self.assertIsNone(inspect.signature(DiffusionModel.postprocess).parameters["preprocess_type"].default)
        self.assertFalse(hasattr(DiffusionModel, "_preprocess"))
        self.assertFalse(hasattr(DiffusionModel, "_postprocess"))

    def test_graph_and_precision_roundtrip_preserve_shape_and_stable_dtype(self) -> None:
        """Byte images convert safely under mixed precision and float64 policies."""

        for policy, dtype in (("mixed_float16", tf.float32), ("float64", tf.float64)):
            with self.subTest(policy=policy):
                tf.keras.mixed_precision.set_global_policy(policy)
                wrapper = self.make_wrapper()
                pixels = tf.reshape(tf.constant([0, 128, 255, 64], tf.uint8), (1, 2, 2, 1))
                converted = tf.function(wrapper.preprocess)(pixels)
                restored = tf.function(wrapper.postprocess)(converted)
                self.assertEqual(converted.dtype, dtype)
                self.assertEqual(converted.shape, pixels.shape)
                np.testing.assert_allclose(restored, pixels, atol=2e-5)

    def test_all_wrapper_input_paths_preprocess_once(self) -> None:
        """Native, joint, mapped, and V2 discriminator inputs share one conversion."""

        pixels = tf.reshape(tf.linspace(0., 255., 64), (4, 4, 4, 1))
        for wrapper_cls in (DiffusionModel, DiffusionClassifier, DiffusionClassifierV2):
            for mode in ("standardize", "min-max", None):
                with self.subTest(wrapper=wrapper_cls.__name__, mode=mode):
                    wrapper = self.make_wrapper(wrapper_cls, preprocess_type=mode)
                    expected = wrapper.preprocess(pixels)
                    with patch.object(wrapper, "preprocess", wraps=wrapper.preprocess) as convert:
                        prepared = wrapper.prep_inputs((pixels, self.labels), use_label_dropout=False)
                    convert.assert_called_once()
                    np.testing.assert_allclose(prepared[0], expected)
                    np.testing.assert_allclose(prepared[3], expected)
                    mapped = wrapper.prep_inputs_map(pixels, self.labels)
                    np.testing.assert_allclose(mapped[0], expected)
                    # V2 discriminator input preparation is independent of its generator path.
                    if wrapper_cls is DiffusionClassifierV2:
                        with patch.object(wrapper, "preprocess", wraps=wrapper.preprocess) as convert:
                            classified = wrapper.prep_clfv2_inputs((pixels, self.labels), 0, return_x0=True)
                        convert.assert_called_once()
                        np.testing.assert_allclose(classified[1], expected)
                        np.testing.assert_allclose(classified[4], expected)
                    gray = wrapper.prep_inputs((pixels[..., 0], self.labels))
                    self.assertEqual(gray[0].shape, pixels.shape)

    def test_sampling_uses_inverse_output_and_ensemble_evaluation_converts_raw_input(self) -> None:
        """Generation returns external pixels and public accuracy consumes those pixels."""

        wrapper = self.make_wrapper(DiffusionClassifier)
        pixels = tf.reshape(tf.linspace(0., 255., 64), (4, 4, 4, 1))
        metric = EnsembleAccuracy(wrapper, max_t=2, seed=19)
        with patch.object(metric, "ensemble_predict", return_value=tf.one_hot(self.labels, 2)) as predict:
            metric.test_step(self.labels, pixels)
        np.testing.assert_allclose(predict.call_args.args[0], wrapper.preprocess(pixels))
        with patch.object(wrapper, "postprocess", wraps=wrapper.postprocess) as inverse:
            result = wrapper.sample(
                labels=tf.constant([0, 1]), steps=2, return_x_ts=True, 
                return_x0s=True, verbose=0, seed=19
            )
        self.assertTrue(inverse.call_args_list)
        self.assertTrue(all(call.kwargs["clip"] for call in inverse.call_args_list))
        for value in (result[0], *result[1], *result[2]):
            self.assertTrue(np.isfinite(value).all())
            self.assertGreaterEqual(float(np.min(value)), 0.)
            self.assertLessEqual(float(np.max(value)), 255.)

    def test_native_teacher_coordinates_preserve_metadata_and_reject_mismatch(self) -> None:
        """Accept matching native input modes and reject known incompatible coordinates."""

        for teacher_mode, mismatch in (
            ("standardize", "min-max"), 
            ("min-max", "standardize"), 
            (None, "standardize")
        ):
            teacher = self.make_wrapper(DiffusionClassifier, preprocess_type=teacher_mode)
            snapshot = teacher.snapshot_teacher_network()
            self.assertTrue(hasattr(snapshot, "_diffusion_preprocess_type"))
            self.assertEqual(snapshot._diffusion_preprocess_type, teacher_mode)
            for role in ("teacher_network", "current_teacher_network"):
                for supplied in (teacher, snapshot):
                    with self.subTest(mode=teacher_mode, role=role, wrapper=supplied is teacher):
                        student = self.make_wrapper(DiffusionClassifier, preprocess_type=mismatch)
                        setter = getattr(student, "set_" + role)
                        with self.assertRaisesRegex(ValueError, "preprocess_type"):
                            setter(supplied)
                        self.assertIsNone(getattr(student, role))
                        compatible = self.make_wrapper(DiffusionClassifier, preprocess_type=teacher_mode)
                        getattr(compatible, "set_" + role)(supplied)
                        attached = getattr(compatible, role)
                        self.assertEqual(attached._diffusion_preprocess_type, teacher_mode)
                        self.assertFalse(attached.trainable)
        for role in ("teacher_network", "current_teacher_network"):
            with self.subTest(role=role, metadata="unknown"):
                student = self.make_wrapper(DiffusionClassifier)
                raw_teacher = self.make_network()
                self.assertFalse(hasattr(raw_teacher, "_diffusion_preprocess_type"))
                getattr(student, "set_" + role)(raw_teacher)
                self.assertIs(getattr(student, role), raw_teacher)

    def test_modes_roundtrip_config_and_invalid_modes_fail_at_public_boundary(self) -> None:
        """Typed and Keras configurations preserve the owning wrapper option."""

        for wrapper_cls, config_cls in (
            (DiffusionModel, DiffusionModelConfig), 
            (DiffusionClassifier, DiffusionClassifierConfig), 
            (DiffusionClassifierV2, DiffusionClassifierV2Config)
        ):
            with self.subTest(wrapper=wrapper_cls.__name__):
                wrapper = self.make_wrapper(wrapper_cls, preprocess_type="min-max")
                config = json.loads(json.dumps(wrapper.get_config()))
                clone = wrapper_cls.from_config(config)
                self.assertEqual(clone.preprocess_type, "min-max")
                self.assertEqual(config_cls(preprocess_type="min-max").kwargs()["preprocess_type"], "min-max")
                for invalid in (
                    "fixed-standardize", "fixed-min-max", "diffusion", "none", "", "normalize"
                ):
                    with self.subTest(mode=invalid):
                        with self.assertRaisesRegex(ValueError, "preprocess_type"):
                            self.make_wrapper(wrapper_cls, preprocess_type=invalid)
                        for operation in (wrapper.preprocess, wrapper.postprocess):
                            with self.assertRaisesRegex(ValueError, "preprocess_type"):
                                operation([0., 255.], invalid)


# Allow focused execution without running checks on import.
if __name__ == "__main__":
    unittest.main()
