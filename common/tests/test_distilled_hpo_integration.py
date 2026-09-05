"""End-to-end regression for Config-driven distilled continual V2 HPO.

Only downloaded MNIST pixels are replaced with deterministic synthetic arrays. Two real
Optuna trials run V2 generator/discriminator training, generated replay, noise and
classification distillation, and EnsembleAccuracy. Assertions connect four validation
continual objectives to their saved matrices and verify YAML, CSV, weight reload, and seeded
ensemble compute modes. Artifacts use a temporary directory; this bounded plumbing test does
not estimate dataset benchmark quality.

Inputs are fixtures constructed by the test methods and their helpers. Tests return no
application result: unittest records assertion outcomes and errors. Run this module directly
or through ``python -m unittest`` discovery. Importing it defines fixtures and cases; it
does not itself start a test run.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf

from common.config import load_config, save_config
from common.continual_reporting import continual_metrics
from common.hpo import run_hpo
from common.model import get_model
from common.train import main
from diffusion.metrics.ensemble_accuracy import EnsembleAccuracy
from diffusion.models.wrapper.diffusion_classifier_v2 import DiffusionClassifierV2


class DistilledHpoIntegrationTests(unittest.TestCase):
    """Only replace downloaded pixels; all project APIs execute normally.

    The unittest runner executes the selected test method with its local fixtures;
    individual methods describe the configurations and failure cases they exercise. There is
    no application model or experiment result returned by constructing this test case.

    Args:
        methodName (str): Test method selected by unittest. Defaults to ``"runTest"``;
            discovery supplies each named ``test_*`` method.

    Attributes:
        _testMethodName (str): Selected method name maintained by unittest.
    """

    @staticmethod
    def _images():
        """Create deterministic synthetic MNIST-shaped train and test pixels.

        A fixed NumPy seed of 17 adds low-valued noise and a class-specific bright stripe.
        Test inversion distinguishes held-out pixels from training input. No external data
        source is consulted.

        Args:
            None.

        Returns:
            tuple: ((x_train, y_train), (x_test, y_test)) with uint8 arrays. Training has 48
            grayscale 28x28 images and 12 rows per class; test uses every third row with
            inverted pixels.
        """

        rng = np.random.default_rng(17)
        labels = np.repeat(np.arange(4, dtype="uint8"), 12)
        images = rng.integers(0, 32, (len(labels), 28, 28), dtype="uint8")
        for image, label in zip(images, labels):
            image[3 + 5 * int(label):7 + 5 * int(label), 3:25] = 224
        return (images, labels), (255 - images[::3], labels[::3])

    def test_distilled_v2_ensemble_multiobjective_config_round_trip(self):
        """Verify the complete distilled continual V2 and ensemble multi-objective pipeline.

        Runs two real trials with two two-class tasks, train_num=1, soft temperature-scaled
        replay-only classifier KD, noise KD, and SNR-weighted primary/distillation-head
        ensembling. Verifies four validation objectives, empty test matrices, frozen
        teachers, YAML round trips, CSV values, and chunked/batched plus saved-weight
        prediction equivalence.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        metrics = [
            "final_average_accuracy", "average_incremental_accuracy",
            "average_forgetting", "backward_transfer",
        ]
        runs = []

        def record_run(*args, **kwargs):
            """Execute the real training pipeline and retain its result for objective assertions.

            The spy records results but does not replace training, evaluation, or model
            construction.

            Args:
                *args (object): Positional common.train.main arguments supplied by HPO,
                    normally the trial Config; empty by default.
                **kwargs (object): Optional main keyword options forwarded without
                    modification; empty by default.

            Returns:
                dict[str, object]: The actual main result, also appended to the enclosing
                runs list.
            """

            result = main(*args, **kwargs)
            runs.append(result)
            return result

        search = {
            "batch_size": [4], "optimizer": ["adam"],
            "capacity": ["32x4"], "depth": [2], "patch_size": [4],
            "patchify_with_cnn": [False], "use_refiner_cnn": [False],
            "mlp_ratio": [2.], "drop_prob": [0.], "timesteps": [500],
            "test_steps": [2], "test_eta": [0.], "schedule": ["clipped_cosine"],
            "image_loss_coef": [0.], "classifier_architecture": ["linear"],
            "classifier_only_cls_token": [True], "clf_cls_token_type": ["new_weight"],
            "feature_aggregation": ["last"], "clf_depth": [1],
            "clf_drop_prob": [0.], "classifier_mlp_ratio": [None], "dropout_rate": [0.],
            "ctr_loss_coef": [0.], "wrapper_name": ["diffusion_classifier_v2"],
            "clf_train_noisified_max_timesteps": [None], "clf_vars_recipe": ["separate"],
            "continual_strategy": ["generative_replay"],
            "clf_distil_scope_generative_replay": ["replay_only"],
            "clf_distil_type": ["soft"],
            "clf_distil_temperature": {"low": 2., "high": 2.},
            "use_noise_distillation": [True],
            "noise_distil_loss_coef": {"low": .1, "high": .1},
            "clf_distil_loss_coef": {"low": .1, "high": .1},
            "replay_budget_mode": ["legacy"], "replay_samples": [2],
            "replay_selection": ["all"], "train_num": [1],
        }
        ensemble = {
            "max_t": 2, "t_chunk_size": 1, "compute_type": "chunked",
            "weighted": True, "clf_acc_coef": .5, "clf_distil_acc_coef": .5,
        }
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory, patch(
            "tensorflow.keras.datasets.mnist.load_data", side_effect=self._images,
        ), patch("common.hpo.main", side_effect=record_run):
            study = run_hpo(
                "continual", "dit_classifier", "MNIST", n_trials=2, epochs=1,
                seed=19, results_path=directory, use_distillation=True,
                use_ensemble_accuracy=True, ensemble_accuracy_kwargs=ensemble,
                objective_metrics=metrics, class_order=[2, 0, 3, 1],
                task_groups=[[2, 0], [3, 1]], task_size=2,
                max_train_samples=8, max_val_samples=4, n_startup_trials=1,
                model_overrides={"compile_args": {"run_eagerly": True}},
                search_space_overrides=search,
            )
            self.assertEqual(len(runs), 2)
            self.assertEqual([direction.name for direction in study.directions], [
                "MAXIMIZE", "MAXIMIZE", "MINIMIZE", "MAXIMIZE",
            ])
            self.assertTrue(all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials))
            for trial, result in zip(study.trials, runs):
                details = result["model"]["continual_details"]
                model = result["model"]["generative_model"]
                self.assertIsInstance(model, DiffusionClassifierV2)
                # The loader remaps original labels to their scheduled dense IDs.
                self.assertEqual(model.seen_classes, {0: 0, 1: 1, 2: 2, 3: 3})
                self.assertEqual(details["task_classes"], [[2, 0], [3, 1]])
                self.assertEqual(model.network.num_classes, 4)
                self.assertTrue(details["use_ensemble_accuracy"])
                matrix = np.asarray(details["validation_ensemble_accuracy_matrix"])
                self.assertEqual(matrix.shape, (2, 2))
                self.assertTrue(np.isnan(matrix[0, 1]))
                self.assertTrue(np.isfinite(matrix[np.tril_indices(2)]).all())
                expected = continual_metrics(matrix)
                np.testing.assert_allclose(trial.values, [expected[name] for name in metrics])
                self.assertEqual(result["evaluations"]["validation_continual_metrics"], expected)
                self.assertEqual(details["ordinary_accuracy_matrix"], [])
                self.assertIsNotNone(model.teacher_network)
                self.assertFalse(model.teacher_network.trainable)
                # The final snapshot is ready to supervise a subsequent task.
                self.assertEqual(model.teacher_network.num_classes, 4)
                generator_history = details["generative_histories"][1]
                classifier_history = details["histories"][1]
                self.assertGreater(classifier_history["clf_distil_loss"][-1], 0.)
                self.assertIn("noise_distil_loss", generator_history)

                config_path = Path(trial.user_attrs["config_path"])
                config = load_config(config_path)
                self.assertEqual(config.model.wrapper_name, "diffusion_classifier_v2")
                self.assertTrue(config.continually_learn.use_distillation)
                self.assertTrue(config.continually_learn.train_classifier_separately)
                self.assertEqual(config.hpo["objective_metrics"], metrics)
                self.assertEqual(config.continually_learn.ensemble_accuracy_kwargs, ensemble)
                round_trip = Path(directory) / f"round_trip_{trial.number}.yaml"
                save_config(config, round_trip)
                self.assertEqual(load_config(round_trip), config)
                output = Path(result["results_path"])
                self.assertTrue((output / "accuracy_matrices.csv").is_file())
                self.assertTrue((output / "summary.csv").is_file())
                saved = pd.read_csv(output / "accuracy_matrices.csv")
                selected = saved[saved["matrix"] == "validation_ensemble_accuracy_matrix"]
                np.testing.assert_allclose(selected["value"].to_numpy().reshape(2, 2), matrix, equal_nan=True)

                inputs = tf.reshape(tf.linspace(-1., 1., 2 * 28 * 28), (2, 28, 28, 1))
                chunked = EnsembleAccuracy(model, seed=31, **ensemble).ensemble_predict(inputs)
                batched = EnsembleAccuracy(model, seed=31, **{
                    **ensemble, "compute_type": "batched",
                }).ensemble_predict(inputs)
                np.testing.assert_allclose(chunked, batched, rtol=1e-5, atol=1e-6)
                # Reload weights once; the other trial uses the same Config path.
                if trial.number == 0:
                    restored = get_model(config)["generative_model"]
                    self.assertIsInstance(restored, DiffusionClassifierV2)
                    restored_predictions = EnsembleAccuracy(
                        restored, seed=31, **ensemble,
                    ).ensemble_predict(inputs)
                    np.testing.assert_allclose(chunked, restored_predictions, rtol=1e-5, atol=1e-6)
            self.assertTrue(study.best_trials)


# Run this module's tests when executed directly.
if __name__ == "__main__":
    unittest.main()
