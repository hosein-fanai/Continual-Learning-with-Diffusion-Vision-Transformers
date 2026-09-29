"""Keep TensorFlow/NumPy evaluation scalars numeric in report APIs and CSVs."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import tensorflow as tf

from common.train import _plain_metric_values, report
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier
from diffusion.models.wrapper.diffusion_classifier_v2 import DiffusionClassifierV2


class ReportingScalarMetricTests(unittest.TestCase):
    """Preserve numeric scalar types in reporting mappings and serialized CSV metrics."""
    def test_scalar_conversion_preserves_values_and_nonscalars(self) -> None:
        """Convert TensorFlow/NumPy scalars while preserving nonscalar identity and source mappings."""

        vector = tf.constant([1.0, 2.0])
        matrix = np.array([[3.0]])
        original = {
            "tensor": tf.constant(0.125, dtype=tf.float64), 
            "bfloat16": tf.constant(0.625, dtype=tf.bfloat16), 
            "numpy": np.float32(0.25), 
            "array": np.asarray(3, dtype=np.int64), 
            "vector": vector, 
            "matrix": matrix, 
            "plain": 0.5
        }
        actual = _plain_metric_values(original)
        self.assertIs(type(actual["tensor"]), float)
        self.assertIs(type(actual["numpy"]), float)
        self.assertIs(type(actual["bfloat16"]), float)
        self.assertEqual(actual["bfloat16"], 0.625)
        self.assertIs(type(actual["array"]), int)
        self.assertEqual([actual[key] for key in ("tensor", "numpy", "array")], 
                         [0.125, 0.25, 3])
        self.assertIs(actual["vector"], vector)
        self.assertIs(actual["matrix"], matrix)
        self.assertTrue(tf.is_tensor(original["tensor"]))

    def test_report_and_csv_preserve_numeric_metrics_for_all_evaluation_paths(self) -> None:
        """Verify plain, V1 and V2 evaluation branches retain numeric values in memory and CSV."""

        for model_type in (tf.keras.Model, DiffusionClassifier, DiffusionClassifierV2):
            with self.subTest(model_type=model_type.__name__):
                model = MagicMock(spec=model_type)
                is_diffusion = model_type is not tf.keras.Model
                # Diffusion wrappers expose EMA/ensemble branches beyond ordinary Keras evaluation.
                if is_diffusion:
                    model.use_ema = True
                model.evaluate.side_effect = lambda *args, **kwargs: {
                    "noise_loss": tf.constant(0.125, dtype=tf.float64), 
                    "bfloat16_metric": tf.constant(0.625, dtype=tf.bfloat16), 
                    "classifier_accuracy": np.float32(0.75), 
                    "count": np.asarray(4, dtype=np.int64)
                }
                # Diffusion wrappers expose EMA/ensemble branches beyond ordinary Keras evaluation.
                if is_diffusion:
                    model.evaluate_ensemble_accuracy.return_value = tf.constant(0.875)
                with tempfile.TemporaryDirectory() as temporary, \
                        patch("common.train._report_final_visuals"):
                    result = report(
                        model=model, 
                        trainset=object(), 
                        valset=object(), 
                        results_path=temporary, 
                        save_csv=True, 
                        show_history_plot=False, 
                        show_final_images=False, 
                        evaluate_ensemble_accuracy=is_diffusion, 
                        verbose=False
                    )
                    written = pd.read_csv(Path(temporary) / "evals history.csv", index_col=0)
                self.assertEqual(len(result), 4 if is_diffusion else 2)
                expected = {"noise_loss": 0.125, "classifier_accuracy": 0.75, "count": 4}
                expected["bfloat16_metric"] = 0.625
                # Diffusion wrappers expose EMA/ensemble branches beyond ordinary Keras evaluation.
                if is_diffusion:
                    expected["ensemble_accuracy"] = 0.875
                for branch, metrics in result.items():
                    for name, value in expected.items():
                        self.assertIsInstance(metrics[name], (float, int))
                        self.assertEqual(metrics[name], value)
                        self.assertTrue(pd.api.types.is_numeric_dtype(written[name]))
                        self.assertEqual(written.loc[branch, name], value)
                # V2 reporting must request both internal model branches on each evaluation.
                if model_type is DiffusionClassifierV2:
                    self.assertTrue(all(call.kwargs["eval_both"]
                                        for call in model.evaluate.call_args_list))


# Run this focused test module only when invoked directly.
if __name__ == "__main__":
    unittest.main()
