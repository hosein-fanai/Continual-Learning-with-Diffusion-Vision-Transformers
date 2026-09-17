"""Real float32 profile/main integration on synthetic CIFAR-shaped rows.

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
from common.callbacks.plateau_lr import OffsetCosineDecay
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

    def _run_case(self, dataset_name="cifar10", aggregation="last", fraction=0.0,
                  noisy_input="clean", class_input="null_class_only"):
        choices = {
            "optimizer": ["adam"], "dim": [32],
            "depth": [3], "clf_depth": [1], "patch_size": [4],
            "mha_num_heads": [4], "feature_aggregation": [aggregation],
            "clf_train_batch_fraction": [fraction],
            "clf_train_noisy_input_type": [noisy_input],
            "clf_train_class_input_type": [class_input],
        }
        batches = {}
        calls = {"fit_active": False, "fit": 0, "evaluate": 0}
        original_fit = DiffusionClassifier.fit
        original_evaluate = DiffusionClassifier.evaluate

        def observed_fit(model: DiffusionClassifier, *args: object, **kwargs: object) -> object:
            """Check the real fit receives retained data but schedules no validation epochs."""
            self.assertEqual(kwargs["validation_freq"], [])
            self.assertIsNotNone(kwargs["validation_data"])
            self.assertEqual(kwargs["epochs"], 2)
            calls["fit"] += 1
            calls["fit_active"] = True
            try:
                history = original_fit(model, *args, **kwargs)
            finally:
                calls["fit_active"] = False
            self.assertFalse(any(name.startswith("val_") for name in history.history))
            self.assertEqual(len(history.epoch), 2)
            return history

        def observed_evaluate(model: DiffusionClassifier, *args: object, **kwargs: object) -> dict:
            """Allow only the final raw evaluation after the complete training budget."""
            self.assertFalse(calls["fit_active"], "Epoch validation must not run")
            self.assertEqual(calls["fit"], 1)
            self.assertEqual(int(model.optimizer.iterations), 4)
            self.assertEqual(kwargs["network_name"], "raw")
            calls["evaluate"] += 1
            scores = original_evaluate(model, *args, **kwargs)
            self.assertIn("noise_loss", scores)
            self.assertIn("classifier_accuracy", scores)
            return scores

        def capture_datasets(*args, **kwargs):
            train, validation = get_datasets(*args, **kwargs)
            batches["train"] = [int(tf.shape(x)[0]) for x, _ in train]
            batches["validation"] = [int(tf.shape(x)[0]) for x, _ in validation]
            return train, validation

        with tempfile.TemporaryDirectory(prefix="joint-hpo-smoke-") as temporary:
            config = build_joint_classifier_config(
                _FirstTrial(), dataset_name=dataset_name, epochs=2, seed=17,
                results_path=temporary, dtype_policy="float32",
                validation_source="test", max_train_samples=7, max_val_samples=3,
                search_space_overrides=choices,
            )
            self.assertEqual(config.hpo["profile_version"], 12)
            self.assertEqual(config.training.fit_kwargs, {"validation_freq": []})
            self.assertTrue(config.training.use_valset)
            self.assertTrue(config.reporting.run_valset_eval)
            self.assertEqual(config.training.monitor, "classifier_accuracy")
            self.assertFalse(config.hpo["fixed_recipe"]["fit_validation"])
            self.assertFalse(config.hpo["fixed_recipe"]["test_set_used_for_fit_validation"])
            self.assertEqual(config.hpo["accuracy_metric"], "classification_accuracy")
            self.assertFalse(config.hpo["use_ensemble_accuracy"])
            self.assertFalse(config.reporting.evaluate_ensemble_accuracy)
            self.assertFalse(config.training.ensemble_monitor)
            self.assertEqual(config.reporting.ensemble_accuracy_kwargs, {})
            self.assertEqual(config.hpo["ensemble_accuracy_kwargs"], {})
            self.assertNotIn("test_noisified_min_timesteps", config.model.wrapper_kwargs)
            self.assertNotIn("test_noisified_max_timesteps", config.model.wrapper_kwargs)
            self.assertNotIn("clf_train_noisified_max_timesteps", config.model.wrapper_kwargs)
            self.assertNotIn("clf_test_noisified_max_timesteps", config.model.wrapper_kwargs)
            self.assertFalse(config.optimizer.plateau_jump)
            self.assertEqual(config.training.reduce_lr_patience, 0)
            self.assertEqual(config.training.patience, 0)
            self.assertEqual(config.model.wrapper_kwargs["clf_train_noisy_input_type"], noisy_input)
            self.assertEqual(config.model.wrapper_kwargs["clf_train_class_input_type"], class_input)
            self.assertEqual(config.model.wrapper_kwargs["clf_train_batch_fraction"], fraction)
            self.assertEqual(config.model.wrapper_kwargs["clf_train_type"], "cond")
            self.assertFalse(config.model.wrapper_kwargs["mask_by_nulls"])
            self.assertEqual(config.model.wrapper_kwargs["clf_loss_coef"], 1.0)
            self.assertFalse(config.model.kwargs["aggregate_from_noises"])
            self.assertFalse(config.reporting.save_final_gifs)
            self.assertEqual(config.dataset.batch_size, 128)
            self.assertEqual(config.optimizer.schedule, "cosine")
            self.assertTrue(config.model.kwargs["patchify_with_cnn"])
            self.assertNotIn("modify_first_t", config.model.wrapper_kwargs)
            # Test-only budget reductions after building the production recipe.
            config.dataset.batch_size = 4
            config.model.show_network_summary = False
            config.model.kwargs.update(dim=8, depth=1, mha_num_heads=1,
                                       clf_mha_num_heads=1, timesteps=4,
                                       clf_dim=8 if aggregation == "all" else None)
            config.model.wrapper_kwargs.update(test_steps=2)
            config.training.verbose = 0
            config.hpo["noise_evaluation_protocol"]["timestep_max_exclusive"] = 4
            config.hpo["software_check"] = "Synthetic data; reduced architecture/horizon/budget only"
            for mode in config.reporting.final_generation_modes:
                if mode["steps"] is not None:
                    mode["steps"] = 3

            with contextlib.redirect_stdout(io.StringIO()), \
                    patch(f"tensorflow.keras.datasets.{dataset_name}.load_data", side_effect=self._cifar) as loader, \
                    patch("common.train.get_datasets", side_effect=capture_datasets), \
                    patch.object(DiffusionClassifier, "fit", new=observed_fit), \
                    patch.object(DiffusionClassifier, "evaluate", new=observed_evaluate), \
                    patch("common.train.PlateauLearningRate",
                          side_effect=AssertionError("Plateau callback must stay disabled")) as plateau, \
                    patch("common.train.callbacks.EarlyStopping",
                          side_effect=AssertionError("Early stopping must stay disabled")) as early_stop, \
                    patch.object(DiffusionClassifier, "evaluate_ensemble_accuracy",
                                 side_effect=AssertionError("Ordinary HPO must not evaluate ensembles")) as ensemble:
                result = main(config)
            ensemble.assert_not_called()
            plateau.assert_not_called()
            early_stop.assert_not_called()
            self.assertEqual(calls, {"fit_active": False, "fit": 1, "evaluate": 1})
            self.assertFalse(any(name.startswith("val_") for name in result["history"]))
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(batches, {"train": [4, 3], "validation": [3]})
            self.assertEqual(config.dataset.trainset_len, 2)
            model = result["model"]
            self.assertEqual(model.dtype_policy.name, "float32")
            self.assertEqual(model.network.dtype_policy.name, "float32")
            self.assertFalse(model.use_ema)
            self.assertEqual(model.clf_train_batch_fraction, fraction)
            self.assertEqual(model.clf_train_noisy_input_type, noisy_input)
            self.assertEqual(model.clf_train_class_input_type, class_input)
            self.assertIsNone(model.ema_network)
            self.assertIsNone(model.teacher_network)
            self.assertIsNone(model.network.distil_classifier)
            self.assertIsNone(model.network.distil_token)
            self.assertTrue(model.network.patchify_with_cnn)
            self.assertFalse(model.modify_first_t)
            optimizer = model.optimizer
            self.assertEqual(int(optimizer.iterations.numpy()), 4)
            self.assertIsNone(optimizer.clipnorm)
            self.assertIsNone(optimizer.global_clipnorm)
            self.assertIsNone(optimizer.clipvalue)
            self.assertNotIsInstance(optimizer._learning_rate, OffsetCosineDecay)
            self.assertEqual(config.optimizer.decay_steps, 4)
            self.assertIsInstance(optimizer._learning_rate, tf.keras.optimizers.schedules.CosineDecay)

            output = Path(result["results_path"])
            for name in ("model.weights.h5", "train history.csv", "evals history.csv",
                         "train history.png", "input_config.yaml", "config.yaml",
                         "final-generation-modes.json"):
                self.assertGreater((output / name).stat().st_size, 0)
            train_history = pd.read_csv(output / "train history.csv")
            self.assertEqual(len(train_history), 2)
            self.assertFalse(any(name.startswith("val_") for name in train_history.columns))
            metrics_csv = pd.read_csv(output / "evals history.csv", index_col=0)
            self.assertEqual(len(metrics_csv), 1)
            self.assertEqual(set(result["evaluations"]), {"valset_network_eval"})
            for branch, metrics in result["evaluations"].items():
                for name in ("noise_loss", "classifier_accuracy"):
                    self.assertIs(type(metrics[name]), float)
                    self.assertTrue(np.isfinite(metrics[name]))
                    self.assertTrue(pd.api.types.is_numeric_dtype(metrics_csv[name]))
                    self.assertAlmostEqual(metrics_csv.loc[branch, name], metrics[name])
                self.assertNotIn("ensemble_accuracy", metrics)
            self.assertNotIn("ensemble_accuracy", metrics_csv.columns)

            manifest = json.loads((output / "final-generation-modes.json").read_text())
            self.assertEqual([mode["steps"] for mode in manifest], [3])
            self.assertEqual([mode["scale"] for mode in manifest], [3])
            self.assertEqual([mode["eta"] for mode in manifest], [0])
            for mode in manifest:
                self.assertEqual(mode["network_name"], "raw")
                self.assertTrue(mode["add_null_label"])
                self.assertEqual(mode["sample_count"], (10 if dataset_name == "cifar10" else 100) + 1)
                self.assertIsNone(mode["gif"])
                with Image.open(output / mode["image"]) as image:
                    image.verify()

            metadata = config.dataset.split_metadata
            self.assertEqual(metadata["training_rows_per_epoch"], 7)
            self.assertEqual(metadata["validation_rows_selected"], 3)
            self.assertEqual(metadata["selected_validation_location"], "official_test")
            self.assertEqual(metadata["reserved_internal_validation_rows"], 0)
            self.assertFalse(metadata["independent_test_estimate"])
            persisted = load_config(output / "config.yaml")
            self.assertEqual(persisted.dataset.split_metadata, metadata)
            self.assertEqual(persisted.hpo["classifier_training"], config.hpo["classifier_training"])
            self.assertIn("clf_train_class_input_type", persisted.model.wrapper_kwargs)
            self.assertEqual(persisted.model.wrapper_kwargs["clf_train_class_input_type"], class_input)
            self.assertEqual(persisted.model.wrapper_kwargs["clf_train_batch_fraction"], fraction)
            self.assertEqual(persisted.training.fit_kwargs, {"validation_freq": []})
            self.assertTrue(persisted.training.use_valset)
            self.assertTrue(persisted.reporting.run_valset_eval)

            tensorboard = Path(config.training.tensorboard_path) / "t0000"
            phase_paths = [tensorboard]
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

            # The actual pipeline's checkpoint restores its trained raw network.
            raw = model.network.weights[0]
            expected_raw = raw.numpy().copy()
            raw.assign(tf.zeros_like(raw))
            model.load_weights(str(output / "model.weights.h5"))
            np.testing.assert_array_equal(raw.numpy(), expected_raw)

    def test_cifar10_raw_joint_cosine_pipeline(self):
        self._run_case()

    def test_cifar100_raw_joint_cosine_pipeline(self):
        self._run_case(dataset_name="cifar100")

    def test_all_feature_aggregation_raw_joint_pipeline(self):
        self._run_case(aggregation="all")

    def test_positive_fraction_with_all_class_conditioning_uses_real_pipeline(self) -> None:
        """A sampled split batch still completes training before its only evaluation."""
        self._run_case(fraction=0.5, noisy_input="noisy", class_input="all_classes")


if __name__ == "__main__":
    unittest.main()
