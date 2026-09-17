"""Real mixed-bfloat16 profile/main integration on synthetic CIFAR-shaped rows.

The real profile is constructed before explicitly shrinking its architecture,
horizon, epoch budget, and sampling steps for this software check. Artifacts
live only in a temporary directory. No dataset download or HPO study is run.
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from common.config import load_config
from common.dataloader import get_datasets
from common.hpo_profiles import build_joint_classifier_config
from common.train import main
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier


class _FirstTrial:
    number = 0

    def __init__(self):
        self.params = {}

    def suggest_categorical(self, name, choices):
        self.params[name] = choices[0]
        return choices[0]

    def suggest_float(self, name, low, high, **kwargs):
        value = 1e-4 if low <= 1e-4 <= high else low
        self.params[name] = value
        return value


class JointHpoPipelineSmokeTests(unittest.TestCase):
    def setUp(self):
        self.previous_policy = tf.keras.mixed_precision.global_policy().name
        tf.keras.backend.clear_session()
        self.addCleanup(tf.keras.mixed_precision.set_global_policy, self.previous_policy)
        self.addCleanup(tf.keras.backend.clear_session)

    @staticmethod
    def _cifar():
        rng = np.random.default_rng(104)
        return (
            (rng.integers(0, 256, (40, 32, 32, 3), dtype=np.uint8),
             (np.arange(40) % 2).reshape(-1, 1)),
            (rng.integers(0, 256, (12, 32, 32, 3), dtype=np.uint8),
             (np.arange(12) % 2).reshape(-1, 1)),
        )

    def _run_case(self, wrapper_name, classifier_cap):
        choices = {
            "wrapper_name": [wrapper_name], "optimizer": ["adam"],
            "learning_rate_schedule": ["cosine"], "dim": [32],
            "depth": [3], "clf_depth": [1], "patch_size": [4],
            "mha_num_heads": [4], "aggregate_from_noises": [False],
            "feature_aggregation": ["last"], "clf_loss_coef": [0.25],
        }
        is_v2 = wrapper_name.endswith("_v2")
        if is_v2:
            choices["clf_train_noisified_max_timesteps"] = [classifier_cap]
        batches = {}

        def capture_datasets(*args, **kwargs):
            train, validation = get_datasets(*args, **kwargs)
            batches["train"] = [int(tf.shape(x)[0]) for x, _ in train]
            batches["validation"] = [int(tf.shape(x)[0]) for x, _ in validation]
            return train, validation

        with tempfile.TemporaryDirectory(prefix="joint-hpo-smoke-") as temporary:
            config = build_joint_classifier_config(
                _FirstTrial(), dataset_name="cifar10", epochs=1, seed=17,
                results_path=temporary, dtype_policy="mixed_bfloat16",
                validation_source="test", max_train_samples=7, max_val_samples=3,
                search_space_overrides=choices,
            )
            self.assertEqual(config.hpo["profile_version"], 6)
            self.assertEqual(config.hpo["accuracy_metric"], "classification_accuracy")
            self.assertFalse(config.hpo["use_ensemble_accuracy"])
            self.assertFalse(config.reporting.evaluate_ensemble_accuracy)
            self.assertFalse(config.training.ensemble_monitor)
            self.assertEqual(config.reporting.ensemble_accuracy_kwargs, {})
            self.assertEqual(config.hpo["ensemble_accuracy_kwargs"], {})
            self.assertNotIn("test_noisified_min_timesteps", config.model.wrapper_kwargs)
            self.assertNotIn("test_noisified_max_timesteps", config.model.wrapper_kwargs)
            # Test-only budget reductions after building the production recipe.
            config.dataset.batch_size = 4
            config.model.show_network_summary = False
            config.model.kwargs.update(dim=8, depth=1, mha_num_heads=1,
                                       clf_mha_num_heads=1, timesteps=4)
            config.model.wrapper_kwargs.update(test_steps=2)
            config.training.verbose = 0
            config.hpo["noise_evaluation_protocol"]["timestep_max_exclusive"] = 4
            config.hpo["software_check"] = "Synthetic data; reduced architecture/horizon/budget only"
            if is_v2 and classifier_cap is not None:
                config.model.wrapper_kwargs.update(clf_train_noisified_max_timesteps=2,
                                                   clf_test_noisified_max_timesteps=2)
            for mode in config.reporting.final_generation_modes:
                if mode["steps"] is not None:
                    mode["steps"] = 3

            with contextlib.redirect_stdout(io.StringIO()), \
                    patch("tensorflow.keras.datasets.cifar10.load_data", side_effect=self._cifar) as loader, \
                    patch("common.train.get_datasets", side_effect=capture_datasets), \
                    patch.object(DiffusionClassifier, "evaluate_ensemble_accuracy",
                                 side_effect=AssertionError("Ordinary HPO must not evaluate ensembles")) as ensemble:
                result = main(config)
            ensemble.assert_not_called()
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(batches, {"train": [4, 3], "validation": [3]})
            self.assertEqual(config.dataset.trainset_len, 2)
            model = result["model"]
            self.assertEqual(model.dtype_policy.name, "mixed_bfloat16")
            self.assertEqual(model.network.dtype_policy.name, "mixed_bfloat16")
            self.assertIsNot(model.network, model.ema_network)
            self.assertIsNone(model.teacher_network)
            self.assertIsNone(model.network.distil_classifier)
            self.assertIsNone(model.network.distil_token)
            self.assertEqual(config.optimizer.decay_steps, 2)
            optimizers = [model.gen_optimizer, model.clf_optimizer] if is_v2 else [model.optimizer]
            for optimizer in optimizers:
                self.assertEqual(int(optimizer.iterations.numpy()), 2)
            if is_v2:
                self.assertIsNot(optimizers[0], optimizers[1])
                self.assertIsNot(optimizers[0]._learning_rate, optimizers[1]._learning_rate)

            output = Path(result["results_path"])
            for name in ("model.weights.h5", "train history.csv", "evals history.csv",
                         "train history.png", "input_config.yaml", "config.yaml",
                         "final-generation-modes.json"):
                self.assertGreater((output / name).stat().st_size, 0)
            self.assertEqual(len(pd.read_csv(output / "train history.csv")), 1)
            metrics_csv = pd.read_csv(output / "evals history.csv", index_col=0)
            self.assertEqual(len(metrics_csv), 2)
            for branch, metrics in result["evaluations"].items():
                for name in ("noise_loss", "classifier_accuracy"):
                    self.assertIs(type(metrics[name]), float)
                    self.assertTrue(np.isfinite(metrics[name]))
                    self.assertTrue(pd.api.types.is_numeric_dtype(metrics_csv[name]))
                    self.assertAlmostEqual(metrics_csv.loc[branch, name], metrics[name])
                self.assertNotIn("ensemble_accuracy", metrics)
            self.assertNotIn("ensemble_accuracy", metrics_csv.columns)

            manifest = json.loads((output / "final-generation-modes.json").read_text())
            self.assertEqual([mode["steps"] for mode in manifest], [3, 3, 2, 2])
            self.assertEqual([mode["scale"] for mode in manifest], [3, 3, 3, 4])
            self.assertEqual([mode["eta"] for mode in manifest], [1, 0, 0, 0])
            for mode in manifest:
                self.assertEqual(mode["network_name"], "ema")
                self.assertTrue(mode["add_null_label"])
                self.assertEqual(mode["sample_count"], 11)
                for kind in ("image", "gif"):
                    with Image.open(output / mode[kind]) as image:
                        image.verify()

            metadata = config.dataset.split_metadata
            self.assertEqual(metadata["training_rows_per_epoch"], 7)
            self.assertEqual(metadata["validation_rows_selected"], 3)
            self.assertEqual(metadata["selected_validation_location"], "official_test")
            self.assertEqual(metadata["reserved_internal_validation_rows"], 0)
            self.assertFalse(metadata["independent_test_estimate"])
            self.assertEqual(load_config(output / "config.yaml").dataset.split_metadata, metadata)

            tensorboard = Path(config.training.tensorboard_path) / "t0000"
            phase_paths = [tensorboard / "generator", tensorboard / "discriminator"] if is_v2 else [tensorboard]
            for path in phase_paths:
                events = list(path.rglob("events.out.tfevents.*"))
                self.assertTrue(events)
                scalars = []
                for event_path in events:
                    for event in tf.compat.v1.train.summary_iterator(str(event_path)):
                        for value in event.summary.value:
                            if value.metadata.plugin_data.plugin_name == "scalars":
                                scalars.append((value.tag, float(tf.make_ndarray(value.tensor).item())))
                self.assertTrue(any(name == "epoch_learning_rate" for name, _ in scalars))
                self.assertTrue(all(np.isfinite(value) for _, value in scalars))

            # The actual pipeline's checkpoint contains both trained branches.
            raw, ema = model.network.weights[0], model.ema_network.weights[0]
            expected_raw, expected_ema = raw.numpy().copy(), ema.numpy().copy()
            raw.assign(tf.zeros_like(raw))
            ema.assign(tf.zeros_like(ema))
            model.load_weights(str(output / "model.weights.h5"))
            np.testing.assert_array_equal(raw.numpy(), expected_raw)
            np.testing.assert_array_equal(ema.numpy(), expected_ema)

    def test_v1_joint_pipeline(self):
        self._run_case("diffusion_classifier", None)

    def test_v2_clean_classifier_pipeline(self):
        self._run_case("diffusion_classifier_v2", None)

    def test_v2_noisy_classifier_pipeline(self):
        self._run_case("diffusion_classifier_v2", 32)


if __name__ == "__main__":
    unittest.main()
