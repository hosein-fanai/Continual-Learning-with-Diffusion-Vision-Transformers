"""Independent checks for recovery prevalidation, rollback and published evidence."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import tensorflow as tf

from common.learner import _run_continual_tasks
from common.recovery import (
    capture_rng_state, fingerprint_state, load_task_checkpoint,
    save_task_checkpoint, save_task_progress,
)
from common.tests import test_task_checkpoint_hooks as hooks_fixtures
from common.tests import test_continual_integration as integration_fixtures
from semantic_consolidation.config import RouteSettings
from semantic_consolidation.controller import RouteController
from semantic_consolidation.memory import ModulationBank
from semantic_consolidation.model import adapt_model
from semantic_consolidation.tests.test_phases import _make_wrapper


class RecoveryValidationBoundaryTests(unittest.TestCase):
    """Check real restoration failures without replacing the recovery implementation."""

    def tearDown(self) -> None:
        """Release tiny fixture models after each independent check.

        Returns:
            result (None): Clears Keras graph and naming state.

        Raises:
            None: Fixture cleanup has no additional validation conditions.
        """
        tf.keras.backend.clear_session()

    def test_semantic_payload_rejects_before_runtime_or_data_changes(self) -> None:
        """Reject a committed nonfinite gate payload before reseeding or loading data.

        Returns:
            result (None): Model, optimizer, controller and Python/NumPy/TensorFlow
                RNG values remain exact; neither runtime setup nor fitting runs.

        Raises:
            AssertionError: Invalid state reaches setup or changes live values.
            Exception: An unexpected fixture or persistence error propagates.
        """
        wrapper = adapt_model(_make_wrapper(), RouteController(RouteSettings(seed=41)))
        controller = wrapper.route_controller
        controller.records = [{"task": 1, "condition": "learned",
                               "seen_classes": [0, 1], "new_classes": [0, 1]}]
        controller.introduced = {0, 1}
        dimension = int(wrapper.network.classifier.layers[-1].kernel.shape[0])
        controller.bank = ModulationBank(controller.settings, dimension, 41)
        controller.bank.add([0, 1])
        payload = wrapper.get_task_checkpoint_state()
        payload["bank"]["gain"][0, 0] = np.nan
        tf.random.set_global_generator(tf.random.Generator.from_seed(47))
        variables = [*wrapper.variables, *wrapper.optimizer.variables]
        before = [value.numpy().copy() for value in variables]
        controller_before = fingerprint_state(wrapper.get_task_checkpoint_state())
        rng_before = fingerprint_state(capture_rng_state(include_tensorflow_global=True))
        loader = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = save_task_checkpoint(directory, 0, {
                "class_order": [0, 1], "task_groups": [[0, 1]], "model_task_state": payload,
            })
            with patch("common.learner.configure_runtime") as setup, patch.object(tf.keras.Model, "fit") as fit:
                with self.assertRaisesRegex(ValueError, "finite float32"):
                    _run_continual_tasks(class_num=2, load_dataset_fn=loader,
                        generative_model=wrapper, resume_from=str(path), seed=53)
            setup.assert_not_called()
            fit.assert_not_called()
            loader.assert_not_called()
        self.assertEqual(controller_before, fingerprint_state(wrapper.get_task_checkpoint_state()))
        self.assertEqual(rng_before, fingerprint_state(capture_rng_state(include_tensorflow_global=True)))
        for actual, expected in zip(variables, before):
            np.testing.assert_array_equal(actual.numpy(), expected)

    def test_failed_tensorflow_assertion_restores_original_values(self) -> None:
        """Roll back matching variables when a later strict graph assertion fails.

        Returns:
            result (None): The destination float32 variable retains its original
                value after a checkpoint with an extra dependency is rejected.

        Raises:
            AssertionError: Restoration succeeds unexpectedly or leaves a partial value.
            Exception: An unexpected TensorFlow or persistence error propagates.
        """
        source = tf.Module()
        source.value = tf.Variable([3., 4.], dtype=tf.float32)
        source.extra = tf.Variable([7.], dtype=tf.float32)
        destination = tf.Module()
        destination.value = tf.Variable([-2., -5.], dtype=tf.float32)
        before = destination.value.numpy().copy()
        with tempfile.TemporaryDirectory() as directory:
            path = save_task_checkpoint(directory, 0,
                {"class_order": [0], "task_groups": [[0]]}, {"model": source})
            with self.assertRaises(AssertionError):
                load_task_checkpoint(path, trackables={"model": destination})
        np.testing.assert_array_equal(destination.value.numpy(), before)
        self.assertEqual(len(destination.variables), 1)

    def test_corrupt_progress_falls_back_without_rewriting_evidence(self) -> None:
        """Prefer valid progress, then fall back to the immutable completed boundary.

        Returns:
            result (None): A corrupt newest snapshot selects its predecessor;
                malformed index variants select the completed task and stay unchanged.

        Raises:
            AssertionError: Discovery selects corruption or rewrites published bytes.
            Exception: An unexpected checkpoint I/O error propagates.
        """
        schedule = {"class_order": [0, 1], "task_groups": [[0], [1]]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundary = save_task_checkpoint(root, 0, schedule)
            immutable = {path.name: path.read_bytes() for path in boundary.iterdir() if path.is_file()}
            state = {**schedule, "active_task_index": 1, "fit_progress": {"batch": 1}}
            first = save_task_progress(root, 1, state, {"cursor": tf.Variable(1)})
            second = save_task_progress(root, 1, state, {"cursor": tf.Variable(2)})
            self.assertEqual(load_task_checkpoint(root).task_dir, second)
            (second / "state.json").write_text("corrupted published state", encoding="utf-8")
            self.assertEqual(load_task_checkpoint(root).task_dir, first)
            for malformed in (b"{", b"[]", b'{"checkpoints": null}', b'{"checkpoints": ["../outside"]}'):
                with self.subTest(index=malformed):
                    (root / "progress.json").write_bytes(malformed)
                    self.assertEqual(load_task_checkpoint(root).task_dir, boundary)
                    self.assertEqual((root / "progress.json").read_bytes(), malformed)
            self.assertEqual(immutable,
                {path.name: path.read_bytes() for path in boundary.iterdir() if path.is_file()})

    def test_initial_restart_preserves_unpublished_and_published_evidence(self) -> None:
        """Restart around an initial publication without overwriting occupied task slots.

        Returns:
            result (None): Private staging residue permits a new initial boundary;
                a malformed published slot rejects before fit setup and retains bytes.

        Raises:
            AssertionError: Recovery erases evidence, cannot restart private residue,
                or accepts an occupied malformed task slot.
            Exception: An unexpected model or checkpoint error propagates.
        """
        for published in (False, True):
            with self.subTest(published=published), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                template = root / "template.keras"
                integration_fixtures.ContinualIntegrationTests._template(template)
                model = integration_fixtures.ContinualIntegrationTests._generator()
                hooks_fixtures._hooks(model)
                model.checkpoint_interval = 1
                model.configure_fit_checkpoint = Mock(side_effect=RuntimeError("initial boundary ready"))
                checkpoint_root = root / "checkpoints"
                residue = checkpoint_root / ".initial" / ("task-0000" if published else ".private-incomplete")
                residue.mkdir(parents=True)
                marker = residue / "evidence.bin"
                marker.write_bytes(b"preserve interrupted publication")
                expected = FileExistsError if published else RuntimeError
                with self.assertRaises(expected):
                    _run_continual_tasks(**hooks_fixtures.TaskCheckpointHookTests._arguments(template, model),
                        save_task_checkpoints=True, checkpoint_dir=str(checkpoint_root))
                self.assertEqual(marker.read_bytes(), b"preserve interrupted publication")
                # An occupied public task slot must fail before fit-checkpoint setup.
                if published:
                    model.configure_fit_checkpoint.assert_not_called()
                # Private uncommitted residue does not prevent a fresh durable boundary.
                else:
                    model.configure_fit_checkpoint.assert_called_once()
                    saved = load_task_checkpoint(checkpoint_root)
                    self.assertEqual(saved.experiment_state["restart_task_index"], 0)
