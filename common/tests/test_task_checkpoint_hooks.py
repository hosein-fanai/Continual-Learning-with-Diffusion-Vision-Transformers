"""Exercise optional Python task-state ownership through the real learner."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import tensorflow as tf
import h5py

from common.learner import _run_continual_tasks
from common.recovery import fingerprint_state, load_task_checkpoint, save_task_checkpoint
from common.tests import test_continual_integration as integration
from common.tests.test_learner_verified_repairs import image_loader
from common.tests.test_wrapper_verified_repairs import _network
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier
from diffusion.models.wrapper.diffusion_model import DiffusionModel


def _hooks(model: DiffusionModel) -> tuple[Mock, Mock, Mock]:
    """Attach an explicit state-owner fixture without replacing model computation.

    Args:
        model (DiffusionModel): Real diffusion wrapper whose optional Python
            checkpoint protocol is exercised independently of its TensorFlow state.

    Returns:
        hooks (tuple[Mock, Mock, Mock]): Config, state, and restore call recorders.
            State includes a float32 array to exercise concrete serialization.
    """
    config = Mock(return_value={"schema": 1, "policy": "boundary_fixture"})
    state = Mock(return_value={"values": np.array([1., 2.], dtype="float32")})
    restore = Mock()
    model.get_task_checkpoint_config = config
    model.get_task_checkpoint_state = state
    model.restore_task_checkpoint_state = restore
    return config, state, restore


class TaskCheckpointHookTests(unittest.TestCase):
    """Reject incomplete ownership without changing ordinary training behavior."""

    def tearDown(self) -> None:
        """Release model state after each isolated learner fixture.

        Returns:
            result (None): Keras graph and naming state are cleared.
        """
        tf.keras.backend.clear_session()

    @staticmethod
    def _arguments(template: Path, model: DiffusionModel) -> dict[str, object]:
        """Describe two small actual classifier/generator training tasks.

        Args:
            template (Path): Existing two-pixel classifier template.
            model (DiffusionModel): Generator with optional task-state hooks.

        Returns:
            options (dict[str, object]): Seeded two-task options using real
                float32 images, sparse integer targets, and one optimizer step
                per active phase. Checkpointing is selected by each test.
        """
        return {
            "class_num": 4, "task_size": 2,
            "load_dataset_fn": integration.ContinualIntegrationTests._loader,
            "load_dataset_fn_kwargs": {"preprocess": "min-max"},
            "tuned_model_path": str(template), "generative_model": model,
            "compile_args": {"optimizer": "adam", "loss": "sparse_categorical_crossentropy",
                             "metrics": ["accuracy"]},
            "generative_model_kwargs": {"train_num": 4},
            "use_generative_replay": False, "optimizer_steps_per_epoch": 1,
            "epochs": 1, "batch_size": 4, "callback_patience": 0,
            "plot_results": False, "verbose": 0, "seed": 31,
        }

    def test_ordinary_training_never_calls_task_checkpoint_hooks(self) -> None:
        """Keep optional state serialization entirely outside ordinary training.

        Returns:
            result (None): Two real tasks complete and none of the three hooks runs.

        Raises:
            AssertionError: If any hook is called or the task fails to complete.
        """
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "template.keras"
            integration.ContinualIntegrationTests._template(template)
            model = integration.ContinualIntegrationTests._generator()
            hooks = _hooks(model)
            result = _run_continual_tasks(**self._arguments(template, model))
            self.assertEqual(result["next_task_index"], 2)
            for hook in hooks:
                hook.assert_not_called()
            self.assertNotIn("model_task_config", result["run_descriptor"])

    def test_partial_checkpoint_protocol_fails_before_fit_or_artifacts(self) -> None:
        """Require complete ownership instead of silently dropping Python state.

        Returns:
            result (None): An incomplete protocol raises before training or
                checkpoint creation; its configuration function is not called.

        Raises:
            AssertionError: If training, hook invocation, or publication occurs.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.keras"
            integration.ContinualIntegrationTests._template(template)
            model = integration.ContinualIntegrationTests._generator()
            model.get_task_checkpoint_config = Mock(return_value={})
            checkpoint_dir = root / "checkpoints"
            with patch.object(tf.keras.Model, "fit") as fit:
                with self.assertRaisesRegex(TypeError, "all three callable"):
                    _run_continual_tasks(**self._arguments(template, model),
                        save_task_checkpoints=True, checkpoint_dir=str(checkpoint_dir))
            fit.assert_not_called()
            model.get_task_checkpoint_config.assert_not_called()
            self.assertFalse(checkpoint_dir.exists())

    def test_checkpoint_state_requires_the_matching_owner(self) -> None:
        """Reject missing or unexpected Python state before TensorFlow restoration.

        Returns:
            result (None): Both ownership mismatches raise with all destination
                model variables unchanged and no restore-hook invocation.

        Raises:
            AssertionError: If a validly encoded ownership mismatch is accepted
                or modifies the destination before rejection.
        """
        for has_owner in (False, True):
            with self.subTest(has_owner=has_owner), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                template = root / "template.keras"
                integration.ContinualIntegrationTests._template(template)
                source = integration.ContinualIntegrationTests._generator()
                # A real checkpoint either owns Python state or has no such protocol.
                if has_owner:
                    _hooks(source)
                _run_continual_tasks(**self._arguments(template, source),
                    save_task_checkpoints=True, checkpoint_dir=str(root / "source"))
                saved = load_task_checkpoint(root / "source")
                state = deepcopy(saved.experiment_state)
                state.update(class_order=saved.class_order, task_groups=saved.task_groups)
                # Remove required state from an owner-authenticated checkpoint.
                if has_owner:
                    state.pop("model_task_state")
                # Add state to a checkpoint whose identity declares no owner.
                else:
                    state["model_task_state"] = {"values": np.ones(2, dtype="float32")}
                save_task_checkpoint(root / "mismatched", saved.completed_task_index,
                    state, fingerprint=saved.fingerprint)
                destination = integration.ContinualIntegrationTests._generator()
                destination_hooks = _hooks(destination) if has_owner else ()
                before = [variable.numpy().copy() for variable in destination.variables]
                with self.assertRaisesRegex(ValueError, "model task state does not match"):
                    _run_continual_tasks(**self._arguments(template, destination),
                        resume_from=str(root / "mismatched"))
                self.assertEqual(len(before), len(destination.variables))
                for expected, actual in zip(before, destination.variables):
                    np.testing.assert_array_equal(actual.numpy(), expected)
                # Rejected ownership never reaches the destination state owner.
                if destination_hooks:
                    destination_hooks[2].assert_not_called()

    def test_template_bytes_only_authenticate_the_external_classifier(self) -> None:
        """Bind artifact bytes only when that artifact supplies the trained classifier.

        The real learner constructs each complete descriptor. A capture at its
        final fingerprint boundary stops before training; all other fingerprint
        calls use the production function. Harmless HDF5 metadata changes bytes
        while preserving decoded model topology and initial float32 weights.

        Returns:
            result (None): Attached-classifier identity is unchanged and still
                contains classifier/replay topology and initial weights. External
                identity changes only through its authenticated artifact bytes.

        Raises:
            AssertionError: If unused bytes affect the attached identity, external
                artifact bytes are ignored, or effective model identity is lost.
        """
        records = []

        def capture(value: object) -> str:
            """Capture the actual run descriptor before any task can start.

            Args:
                value (object): Concrete value passed to the learner's fingerprint
                    function, including intermediate identities and the final run.

            Returns:
                digest (str): Production SHA-256 for intermediate descriptors.

            Raises:
                RuntimeError: At the final run descriptor, after saving an
                    independent copy for the assertions below.
            """
            # Only the complete run identity ends this descriptor-level regression.
            if isinstance(value, dict) and "models" in value and "training" in value:
                records.append(deepcopy(value))
                raise RuntimeError("fixture captured run descriptor")
            return fingerprint_state(value)

        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "template.h5"
            tf.keras.utils.set_random_seed(31)
            model = tf.keras.Sequential([
                tf.keras.layers.Input((4, 4, 1)), tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4, activation="relu"),
                tf.keras.layers.Dense(2, activation="softmax"),
            ])
            model.save(template)
            for attached in (False, True):
                records.clear()
                for metadata in ("first", "different serialization metadata"):
                    with h5py.File(template, "r+") as archive:
                        archive.attrs["fixture_metadata"] = metadata
                    tf.keras.backend.clear_session()
                    tf.keras.utils.set_random_seed(37)
                    generator = DiffusionClassifier(network=_network(classes=None, distil=False),
                        seed=37, use_ema=True, test_steps=2, scheduler_name="linear")
                    generator.compile(optimizer="adam", loss="mse", run_eagerly=True)
                    arguments = self._arguments(template, generator)
                    arguments.update(load_dataset_fn=image_loader, seed=37,
                        use_generative_model_classifier=attached)
                    with patch("common.learner.fingerprint_state", side_effect=capture):
                        with self.assertRaisesRegex(RuntimeError, "fixture captured run descriptor"):
                            _run_continual_tasks(**arguments)
                self.assertEqual(len(records), 2)
                first, second = records
                for key in ("classifier", "classifier_initial_weights", "replay", "replay_initial_weights"):
                    self.assertIsNotNone(first["models"][key])
                    self.assertEqual(fingerprint_state(first["models"][key]),
                                     fingerprint_state(second["models"][key]))
                # An attached classifier never loads the independent artifact.
                if attached:
                    self.assertIsNone(first["models"]["template_artifact"])
                    self.assertEqual(fingerprint_state(first), fingerprint_state(second))
                # External models retain exact authentication of their supplied file bytes.
                else:
                    self.assertNotEqual(first["models"]["template_artifact"]["sha256"],
                                        second["models"]["template_artifact"]["sha256"])
                    self.assertNotEqual(fingerprint_state(first), fingerprint_state(second))
                    first["models"]["template_artifact"] = None
                    second["models"]["template_artifact"] = None
                    self.assertEqual(fingerprint_state(first), fingerprint_state(second))


# Direct execution uses the same regression cases as unittest discovery.
if __name__ == "__main__":
    unittest.main()
