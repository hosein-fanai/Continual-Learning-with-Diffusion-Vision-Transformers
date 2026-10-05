"""Independent preprocessing and exact-byte recovery oracles from the H100 audit."""

import hashlib
import unittest

import ml_dtypes
import numpy as np

from common.dataloader import _resolve_dataset_options
from common.hpo import _build_trial_config
from common.recovery import _array_recovery_descriptor
from common.tests import test_hpo as hpo_fixtures


class AuditConfigRecoveryRepairTests(unittest.TestCase):
    """Check configuration boundaries without loading datasets or training models."""

    def _config(self, task: str, model: str) -> object:
        """Build a deterministic two-class-first-task MNIST trial."""

        return _build_trial_config(
            hpo_fixtures._SuggestionTrial(), task, model, 
            "cifar10" if model == "pretrained" else "mnist", 
            epochs=1, results_path="unused-audit-regression", 
            class_num=4 if task == "continual" else None, 
            task_size=2 if task == "continual" else 1, seed=3
        )

    def test_all_continual_diffusion_families_keep_raw_loader_coordinates(self) -> None:
        """Every declared continual diffusion family passes the actual loader guard."""

        for model in (
            "diffusion_transformer", "dit_classifier", "dit_decoder", 
            "dit_encoder_decoder", "dit_encoder_decoder_classifier", 
            "unet", "unet_classifier"
        ):
            with self.subTest(model=model):
                config = self._config("continual", model)
                self.assertIsNone(config.dataset.preprocess)
                self.assertIsNone(_resolve_dataset_options(config, {})["preprocess"])
                self.assertEqual(config.model.classifier_name, "cnn")
                config.dataset.preprocess = "min-max"
                with self.assertRaisesRegex(ValueError, "Diffusion datasets require"):
                    _resolve_dataset_options(config, {})

    def test_other_family_preprocessing_contracts_remain_unchanged(self) -> None:
        """Generator, VAE and standalone classifier configurations retain their units."""

        cases = [
            ("generation", "diffusion_transformer", None), 
            ("generation", "dit_decoder", None), 
            ("generation", "dit_encoder_decoder", None), 
            ("generation", "unet", None), 
            ("generation", "vae", "min-max"), 
            ("joint", "vae_classifier", "min-max"), 
            ("continual", "vae", "normalize"), 
            ("continual", "vae_classifier", "normalize"), 
            ("classification", "cnn", "min-max"), 
            ("classification", "dnn", "normalize"), 
            ("classification", "pretrained", None), 
            ("continual", "cnn", "min-max"), 
            ("continual", "dnn", "normalize"), 
            ("continual", "pretrained", None)
        ]
        for task, model, expected in cases:
            with self.subTest(task=task, model=model):
                config = self._config(task, model)
                self.assertEqual(config.dataset.preprocess, expected)
                self.assertEqual(_resolve_dataset_options(config, {})["preprocess"], expected)

    def _assert_descriptor(self, array: np.ndarray) -> None:
        """Assemble the documented hash independently from exact C-order bytes."""

        before = array.tobytes(order="C")
        payload = array.dtype.str.encode("ascii") + str(tuple(array.shape)).encode("ascii") + before
        self.assertEqual(_array_recovery_descriptor(array), {
            "shape": list(array.shape), 
            "dtype": array.dtype.str, 
            "sha256": hashlib.sha256(payload).hexdigest()
        })
        self.assertEqual(array.tobytes(order="C"), before)

    def test_bfloat16_hashes_preserve_scalar_strided_and_raw_bit_patterns(self) -> None:
        """Hash bfloat16 without conversion, including signed zero and NaN payloads."""

        bits = np.asarray([0, 32768, 16256, 49024, 32641, 32642], dtype=np.uint16)
        values = bits.view(ml_dtypes.bfloat16)
        for array in (
            values, values[2].reshape(()), values.reshape(2, 3).T, 
            values[::-2], np.empty((0, 2), dtype=ml_dtypes.bfloat16)
        ):
            with self.subTest(shape=array.shape, strides=array.strides):
                self._assert_descriptor(array)
        self.assertNotEqual(
            _array_recovery_descriptor(values[0])["sha256"], 
            _array_recovery_descriptor(values[1])["sha256"]
        )
        self.assertNotEqual(
            _array_recovery_descriptor(values[4])["sha256"], 
            _array_recovery_descriptor(values[5])["sha256"]
        )

    def test_existing_numeric_hashes_and_object_rejection_are_unchanged(self) -> None:
        """Retain dtype, byte order, shape, empty and pointer-array identity contracts."""

        for dtype in ("float16", "float32", "float64", "int8", "int32", "uint64", 
                      ">f4", ">i8", "complex64", "bool"):
            for array in (
                np.asarray(1, dtype=dtype), np.asarray([1, 0], dtype=dtype), 
                np.asarray([[1, 0], [0, 1]], dtype=dtype).T, 
                np.empty((0, 2), dtype=dtype)
            ):
                with self.subTest(shape=array.shape, dtype=dtype):
                    self._assert_descriptor(array)
        self.assertIsNone(_array_recovery_descriptor(None))
        for array in (np.asarray([object()], dtype=object), 
                      np.empty(0, dtype=object), np.empty(1, dtype=[("value", object)])):
            with self.assertRaisesRegex(TypeError, "Object-dtype"):
                _array_recovery_descriptor(array)


# Direct execution runs these focused regression cases.
if __name__ == "__main__":
    unittest.main()
