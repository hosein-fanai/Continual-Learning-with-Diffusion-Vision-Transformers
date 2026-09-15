"""Recover real external-classifier tasks through the public Config factory."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import tensorflow as tf

from common.config import Config, load_config, save_config
from common.recovery import load_task_checkpoint
from common.train import main


def _pixels() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Supply a small deterministic replacement for downloaded MNIST pixels.

    Returns:
        splits (tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]):
            Training and test pairs with uint8 images shaped [24, 28, 28] and
            [8, 28, 28], and uint8 labels covering four classes. Actual project
            preprocessing, task splitting, and TensorFlow datasets remain active.
    """
    labels = np.repeat(np.arange(4, dtype="uint8"), 6)
    images = np.broadcast_to(labels[:, None, None] * 60, (24, 28, 28)).copy()
    test_labels = np.repeat(np.arange(4, dtype="uint8"), 2)
    test_images = np.broadcast_to(test_labels[:, None, None] * 60, (8, 28, 28)).copy()
    return (images, labels), (test_images, test_labels)


def _config(root: Path) -> Config:
    """Configure a small seeded classifier-only stream with real task checkpoints.

    Args:
        root (Path): Parent of independent result and checkpoint directories.

    Returns:
        config (Config): Two two-class tasks, a float32 CNN with two convolution
            filters, Adam, and two actual updates per task. Rendering and optional
            final evaluations are disabled; task validation remains enabled.
    """
    return Config(
        dataset={"name": "mnist", "batch_size": 4, "preprocess": "min-max",
                 "validation_ratio": 0.25},
        model={"name": "cnn", "show_network_summary": False, "kwargs": {
            "architecture_kwargs": {"conv_filters": [2], "conv_depths": [1],
                                    "use_batch_norm": False},
            "compile_args": {"run_eagerly": True},
        }},
        optimizer={"schedule": "constant", "initial_learning_rate": 0.001},
        training={"task": "continual", "seed": 79, "epochs": 1, "verbose": 0,
                  "results_path": str(root / "runs"), "save_weights": False,
                  "save_gifs": False, "report_every_epoch": False},
        continually_learn={"class_num": 4, "task_size": 2, "baseline": "sequential",
                           "use_generative_replay": False, "plot_results": False,
                           "optimizer_steps_per_epoch": 2, "save_task_checkpoints": True,
                           "checkpoint_dir": str(root / "checkpoints")},
        reporting={"save_history_plot": False, "save_final_images": False,
                   "save_final_gifs": False, "run_trainset_eval": False,
                   "run_valset_eval": False, "save_csv": False},
    )


def _interrupt(config: Config) -> None:
    """Commit the first real task and interrupt before the second task updates.

    Args:
        config (Config): Fresh public-factory settings, updated with actual paths.

    Returns:
        result (None): The first task's actual model and optimizer checkpoint exists.

    Raises:
        RuntimeError: With the fixture interruption marker at the second fit entry.
    """
    actual_fit = tf.keras.Model.fit
    calls = 0

    def fit(model: tf.keras.Model, *args: object, **kwargs: object) -> object:
        """Pass through the first actual fit and stop at the next task boundary.

        Args:
            model (tf.keras.Model): Current external classifier.
            *args (object): Original Keras fit positional arguments.
            **kwargs (object): Original Keras fit keyword arguments.

        Returns:
            history (object): Actual first-task Keras History.

        Raises:
            RuntimeError: Before any second-task optimizer update.
        """
        nonlocal calls
        calls += 1
        # Leave one complete task durable while simulating an external interruption.
        if calls == 2:
            raise RuntimeError("factory interruption after committed task")
        return actual_fit(model, *args, **kwargs)

    with patch.object(tf.keras.Model, "fit", fit):
        main(config)


class FactoryRecoveryTests(unittest.TestCase):
    """Retain exact initial artifacts and actual optimizer state across factory runs."""

    def tearDown(self) -> None:
        """Release Keras naming and graph state after each independent case.

        Returns:
            result (None): Global Keras construction state is cleared.
        """
        tf.keras.backend.clear_session()

    def test_public_factory_resume_matches_updates_across_hpo_and_new_checkpoint_roots(self) -> None:
        """Recover real tasks with separated input, output, and checkpoint directories.

        The reference uses the ordinary generated input config. The interrupted
        branch uses an HPO-style immutable input in another directory; its retry
        has a new trial input and saves into a fresh checkpoint root. A further
        restart authenticates and strictly restores that relocated sequence.

        Returns:
            result (None): Every final classifier weight, Adam variable, validation
                entry, task cursor, and retained template byte matches its reference.

        Raises:
            AssertionError: If actual training, restoration, or artifact reuse differs.
        """
        with tempfile.TemporaryDirectory() as directory, patch(
            "tensorflow.keras.datasets.mnist.load_data", side_effect=_pixels
        ):
            root = Path(directory)
            reference = main(_config(root / "reference"))["model"]["continual_details"]
            original = _config(root / "interrupted")
            inputs = root / "study" / "configs"
            inputs.mkdir(parents=True)
            original.hpo["input_config_path"] = str(inputs / "trial-1.yaml")
            save_config(original, original.hpo["input_config_path"])
            with self.assertRaisesRegex(RuntimeError, "factory interruption"):
                _interrupt(original)
            old_root = Path(original.continually_learn.checkpoint_dir)
            self.assertEqual(load_task_checkpoint(old_root).next_task_index, 1)
            original_bytes = (old_root / "classifier-template.h5").read_bytes()

            resumed = load_config(original.hpo["input_config_path"])
            resumed.hpo["input_config_path"] = str(inputs / "trial-2.yaml")
            resumed.training.results_path = str(root / "retry-output")
            resumed.continually_learn.resume_from = str(old_root)
            new_root = root / "new-checkpoints"
            resumed.continually_learn.checkpoint_dir = str(new_root)
            save_config(resumed, resumed.hpo["input_config_path"])
            result = main(resumed)["model"]["continual_details"]
            self.assertEqual(result["next_task_index"], 2)
            self.assertEqual((new_root / "classifier-template.h5").read_bytes(), original_bytes)
            self.assertEqual((old_root / "classifier-template.h5").read_bytes(), original_bytes)
            self.assertEqual(load_task_checkpoint(new_root).next_task_index, 2)
            np.testing.assert_array_equal(reference["ordinary_accuracy_matrix"], result["ordinary_accuracy_matrix"])
            np.testing.assert_array_equal(reference["validation_accuracy_matrix"], result["validation_accuracy_matrix"])
            self.assertTrue(np.isfinite(result["validation_accuracy_matrix"]).any())
            for expected, actual in ((reference["model"].weights, result["model"].weights),
                                     (reference["model"].optimizer.variables, result["model"].optimizer.variables)):
                self.assertEqual(len(expected), len(actual))
                for first, second in zip(expected, actual):
                    np.testing.assert_array_equal(first.numpy(), second.numpy())

            again = load_config(str(inputs / "trial-2.yaml"))
            again.continually_learn.resume_from = str(new_root)
            again.training.results_path = str(root / "complete-retry")
            restored = main(again)["model"]["continual_details"]
            self.assertEqual(restored["next_task_index"], 2)
            self.assertEqual(len(restored["model"].weights), len(result["model"].weights))
            for first, second in zip(restored["model"].weights, result["model"].weights):
                np.testing.assert_array_equal(first.numpy(), second.numpy())

    def test_resume_rejects_changed_declarations_tampering_missing_and_conflicting_templates(self) -> None:
        """Keep model/optimizer declarations and exact artifact bytes authenticated.

        Returns:
            result (None): Changed architecture, learning rate, runtime seed, HDF5
                metadata, missing artifacts, and conflicting relocation files all
                reject before any Keras fit. An existing destination is unchanged.

        Raises:
            AssertionError: If a changed run trains or a conflicting file is overwritten.
        """
        with tempfile.TemporaryDirectory() as directory, patch(
            "tensorflow.keras.datasets.mnist.load_data", side_effect=_pixels
        ):
            root = Path(directory)
            original = _config(root)
            with self.assertRaisesRegex(RuntimeError, "factory interruption"):
                _interrupt(original)
            resumed = load_config(original.hpo["input_config_path"])
            resumed.continually_learn.resume_from = original.continually_learn.checkpoint_dir
            template = Path(original.continually_learn.checkpoint_dir) / "classifier-template.h5"
            original_bytes = template.read_bytes()
            alternatives = [deepcopy(resumed) for _ in range(3)]
            alternatives[0].model.kwargs["architecture_kwargs"]["conv_filters"] = [3]
            alternatives[1].optimizer.initial_learning_rate = 0.002
            alternatives[2].training.seed = 80
            for index, changed in enumerate(alternatives):
                with self.subTest(changed_declaration=index), patch.object(tf.keras.Model, "fit") as fit:
                    with self.assertRaisesRegex(ValueError, "declaration.*differs"):
                        main(changed)
                    fit.assert_not_called()
                    self.assertEqual(template.read_bytes(), original_bytes)

            with h5py.File(template, "a") as artifact:
                artifact.attrs["tampered_metadata"] = "preserves decoded model weights"
            with patch.object(tf.keras.Model, "fit") as fit:
                with self.assertRaisesRegex(ValueError, "Run fingerprint differs"):
                    main(deepcopy(resumed))
                fit.assert_not_called()
            template.write_bytes(original_bytes)
            with h5py.File(template, "a") as artifact:
                declaration = json.loads(artifact.attrs["continual_factory_config"])
                declaration["optimizer"]["initial_learning_rate"] = 0.002
                artifact.attrs["continual_factory_config"] = json.dumps(declaration, sort_keys=True)
            with patch.object(tf.keras.Model, "fit") as fit:
                with self.assertRaisesRegex(ValueError, "Run fingerprint differs"):
                    main(deepcopy(alternatives[1]))
                fit.assert_not_called()
            template.write_bytes(original_bytes)
            with h5py.File(template, "a") as artifact:
                del artifact.attrs["continual_factory_config"]
            with self.assertRaisesRegex(ValueError, "declaration.*missing"):
                main(deepcopy(resumed))
            template.write_bytes(original_bytes)
            conflict = root / "conflicting-checkpoints"
            conflict.mkdir()
            occupied = conflict / "classifier-template.h5"
            occupied.write_bytes(b"another experiment's template")
            redirected = deepcopy(resumed)
            redirected.continually_learn.checkpoint_dir = str(conflict)
            with self.assertRaisesRegex(FileExistsError, "Conflicting immutable"):
                main(redirected)
            self.assertEqual(occupied.read_bytes(), b"another experiment's template")

            template.unlink()
            with patch.object(tf.keras.Model, "fit") as fit:
                with self.assertRaisesRegex(FileNotFoundError, "original classifier-template"):
                    main(deepcopy(resumed))
                fit.assert_not_called()


# Allow focused execution without starting tests when this module is imported.
if __name__ == "__main__":
    unittest.main()
