"""Core task recovery: validated gate payloads and actual interrupted training.

Synthetic pixels exercise the ordinary factory, replay, KD, phases, reporting
and common checkpoint APIs; they are not benchmark measurements.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.recovery import capture_rng_state, fingerprint_state, load_task_checkpoint
from semantic_consolidation.config import RouteSettings, load_route_config, validate_route_config
from semantic_consolidation.controller import RouteController
from semantic_consolidation.memory import ModulationBank
from semantic_consolidation.model import SemanticConsolidationClassifier, adapt_model
from semantic_consolidation.runner import run
from semantic_consolidation.tests import test_integration
from semantic_consolidation.tests.test_phases import _make_wrapper


_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS = _ROOT / "semantic_consolidation/configs"


class RouteCheckpointStateTests(unittest.TestCase):
    """Keep the plain semantic payload independent of model weights and RNGs."""

    def setUp(self) -> None:
        """Build a tiny real adapter and a completed two-class bank fixture.

        Returns:
            initialized (None): None; stores the real wrapper/controller with
                float32 gain/bias vectors and an explicit global test generator.

        Raises:
            ValueError: If the shared tiny-model factory cannot build the fixture.
        """
        self.wrapper = adapt_model(_make_wrapper(), RouteController(RouteSettings(seed=41)))
        self.controller = self.wrapper.route_controller
        self.controller.records = [{"task": 1, "condition": "learned",
                                    "seen_classes": [0, 1], "new_classes": [0, 1]}]
        self.controller.introduced = {0, 1}
        dimension = int(self.wrapper.network.classifier.layers[-1].kernel.shape[0])
        self.controller.bank = ModulationBank(self.controller.settings, dimension, 41)
        self.controller.bank.add([0, 1])
        self.controller.bank.vectors[0][0].assign_add(tf.ones(dimension) * 0.2)
        tf.random.set_global_generator(tf.random.Generator.from_seed(47))
        self.addCleanup(tf.keras.backend.clear_session)

    def test_round_trip_is_detached_untracked_and_does_not_advance_rng(self) -> None:
        """Restore exact raw vectors without aliasing records or consuming global RNGs.

        Returns:
            checked (None): None; verifies independent variables/records and
                unchanged Python, NumPy and explicit TensorFlow generator states.

        Raises:
            AssertionError: If arrays, tracking, ownership or random states differ.
        """
        state = self.wrapper.get_task_checkpoint_state()
        expected = deepcopy(state)
        before = fingerprint_state(capture_rng_state(include_tensorflow_global=True))
        old_variables = {id(v) for pair in self.controller.bank.vectors.values() for v in pair}
        self.wrapper.restore_task_checkpoint_state(state, completed_tasks=1)
        self.assertEqual(fingerprint_state(self.wrapper.get_task_checkpoint_state()), fingerprint_state(expected))
        self.assertEqual(before, fingerprint_state(capture_rng_state(include_tensorflow_global=True)))
        restored_variables = {id(v) for pair in self.controller.bank.vectors.values() for v in pair}
        self.assertTrue(old_variables.isdisjoint(restored_variables))
        self.assertTrue(restored_variables.isdisjoint({id(v) for v in self.wrapper.weights}))
        state["records"][0]["seen_classes"].append(99)
        state["bank"]["gain"][0, 0] = 100.
        self.assertEqual(fingerprint_state(self.wrapper.get_task_checkpoint_state()), fingerprint_state(expected))
        config = self.wrapper.get_task_checkpoint_config()
        config["route"]["temperature"] = 999.
        self.assertNotEqual(self.controller.settings.temperature, 999.)

    def test_invalid_payload_fails_before_controller_mutation(self) -> None:
        """Reject corrupt class, cursor, schema, shape, dtype and finite-value contracts.

        Returns:
            checked (None): None; verifies each malformed variant raises while
                preserving the original completed controller state exactly.

        Raises:
            AssertionError: If corruption is accepted or changes the live controller.
        """
        original = self.wrapper.get_task_checkpoint_state()
        cases = []
        for key, value in (("schema_version", 999), ("introduced", [1, 0]),
                           ("records", []), ("bank", None)):
            changed = deepcopy(original)
            changed[key] = value
            cases.append(changed)
        for key, value in (("dimension", 999), ("classes", [0]),
                           ("gain", original["bank"]["gain"].astype("float64")),
                           ("bias", np.zeros((1, 1), dtype="float32"))):
            changed = deepcopy(original)
            changed["bank"][key] = value
            cases.append(changed)
        changed = deepcopy(original)
        changed["bank"]["gain"][0, 0] = np.nan
        cases.append(changed)
        changed = deepcopy(original)
        changed["records"][0]["new_classes"] = [1]
        cases.append(changed)
        for state in cases:
            with self.subTest(state_keys=list(state)), self.assertRaises(ValueError):
                self.wrapper.restore_task_checkpoint_state(state, completed_tasks=1)
            self.assertEqual(fingerprint_state(self.wrapper.get_task_checkpoint_state()), fingerprint_state(original))
        for cursor in (0, 2, True):
            with self.subTest(cursor=cursor), self.assertRaises(ValueError):
                self.wrapper.restore_task_checkpoint_state(original, completed_tasks=cursor)

    def test_discarded_and_platform_banks_keep_their_distinct_states(self) -> None:
        """Empty retained-state arrays and genuinely absent banks both round-trip.

        Returns:
            checked (None): None; verifies empty float32 [0, D] gate arrays for
                discarded semantic memory and None for platform-only controls.

        Raises:
            AssertionError: If bank allocation or retention semantics change.
        """
        self.controller.settings = replace(self.controller.settings, retain_modulators=False)
        self.controller.bank.vectors.clear()
        state = self.wrapper.get_task_checkpoint_state()
        self.assertEqual(state["bank"]["gain"].shape[0], 0)
        self.wrapper.restore_task_checkpoint_state(state, completed_tasks=1)
        self.assertEqual(self.controller.bank.vectors, {})
        for condition in ("baseline", "extra_joint"):
            self.controller.settings = replace(self.controller.settings, condition=condition)
            self.controller.records[0]["condition"] = condition
            self.controller.bank = None
            state = self.wrapper.get_task_checkpoint_state()
            self.wrapper.restore_task_checkpoint_state(state, completed_tasks=1)
            self.assertIsNone(self.controller.bank)

    def test_active_tasks_cannot_be_saved_as_completed_boundaries(self) -> None:
        """Fail explicitly when an in-progress cache is presented as a complete task.

        Returns:
            checked (None): None; verifies an active task cannot produce a
                misleading completed-task checkpoint.

        Raises:
            AssertionError: If an unsupported state is accepted or lacks the guard.
        """
        self.controller._boundary = {"active": True}
        with self.assertRaisesRegex(ValueError, "fully completed"):
            self.wrapper.get_task_checkpoint_state()
        self.controller._boundary = None

    def test_config_accepts_core_extension_observer_and_time_recovery(self) -> None:
        """Allow supported checkpoint controls for each concrete route treatment.

        Returns:
            checked (None): None; validates actual shipped extension/observer
                recipes, the core route and declared wall-clock controls.

        Raises:
            AssertionError: If a supported recovery configuration is rejected.
            ValueError: If the supported core recovery fixture fails validation.
        """
        config = load_route_config(_CONFIGS / "smoke.yaml")
        config.common.continually_learn.save_task_checkpoints = True
        config.common.continually_learn.checkpoint_dir = "results/task_checkpoints"
        config.common.continually_learn.resume_from = "results/task_checkpoints"
        validate_route_config(config)
        for filename in ("extensions_smoke.yaml", "section11_smoke.yaml"):
            configured = load_route_config(_CONFIGS / filename)
            configured.common.continually_learn.save_task_checkpoints = True
            with self.subTest(filename=filename):
                validate_route_config(configured)
        config.route = replace(config.route, condition="time_matched_joint", extra_joint_seconds=(1., 1.))
        validate_route_config(config)


class RouteRecoveryIntegrationTests(unittest.TestCase):
    """Compare real two-task training against restart at its committed boundary."""

    def test_interrupted_route_matches_rng_next_update_and_completed_stream(self) -> None:
        """Restore gates and common state before the same next task and joint update.

        The uninterrupted reference and restarted stream use actual synthetic
        images through the ordinary loader, generated replay, KD and all phases.
        Compare common's saved Python/NumPy streams, the next actual joint update,
        final network/teacher/optimizer/gates, phase losses and validation matrix.
        The next-update check also exercises common's per-task TensorFlow reseed.

        Returns:
            checked (None): None; verifies exact arrays and task records after an
                interruption, and rejection of a changed alignment coefficient.

        Raises:
            AssertionError: If recovery differs from uninterrupted training or
                accepts a changed treatment fingerprint.
            Exception: Propagates unexpected real training or checkpoint failures.
        """
        original_fit = SemanticConsolidationClassifier.fit
        original_step = SemanticConsolidationClassifier.train_step
        observations = []
        first_updates = []

        def observed_fit(wrapper: SemanticConsolidationClassifier, *args: object, **kwargs: object) -> object:
            """Record common's Python/NumPy RNGs at the second task's fit entry.

            Args:
                wrapper (SemanticConsolidationClassifier): Real live route adapter.
                args (object): Positional arguments forwarded to its existing fit.
                kwargs (object): Keyword arguments forwarded unchanged to fit.

            Returns:
                history (object): Original fit result after all semantic phases.

            Raises:
                Exception: Propagates actual training or phase errors.
            """
            # Both uninterrupted and resumed runs reach the same completed-task cursor.
            if len(wrapper.route_controller.records) == 1:
                observations.append(fingerprint_state(capture_rng_state()))
            return original_fit(wrapper, *args, **kwargs)

        def observed_step(wrapper: SemanticConsolidationClassifier, data: object) -> object:
            """Capture weights after the first genuine joint update of task two.

            Args:
                wrapper (SemanticConsolidationClassifier): Real live route adapter.
                data (object): Ordinary image/label/replay batch forwarded unchanged.

            Returns:
                metrics (object): Existing train_step metric mapping.

            Raises:
                Exception: Propagates optimizer, gradient or data errors.
            """
            result = original_step(wrapper, data)
            # One capture per observed second-task fit proves the next optimization agrees.
            if len(wrapper.route_controller.records) == 1 and len(first_updates) < len(observations):
                first_updates.append([v.numpy().copy() for v in wrapper.network.weights + wrapper.optimizer.variables])
            return result

        def interrupted_fit(wrapper: SemanticConsolidationClassifier, *args: object, **kwargs: object) -> object:
            """Interrupt only after common has committed the complete first task.

            Args:
                wrapper (SemanticConsolidationClassifier): Real live route adapter.
                args (object): Positional arguments forwarded to existing fit.
                kwargs (object): Keyword arguments forwarded unchanged to fit.

            Returns:
                history (object): Existing first-task fit result.

            Raises:
                RuntimeError: Intentionally interrupts at the second task's fit entry.
            """
            # Task one has been atomically saved by common before the second fit begins.
            if len(wrapper.route_controller.records) == 1:
                raise RuntimeError("TEST interruption after committed semantic task")
            return original_fit(wrapper, *args, **kwargs)

        temporary_root = _ROOT / ".tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="route-recovery-", dir=temporary_root,
                                         ignore_cleanup_errors=True) as directory, patch(
            "tensorflow.keras.datasets.mnist.load_data", side_effect=test_integration.RouteIntegrationTests._pixels,
        ):
            root = Path(directory)
            template = load_route_config(_CONFIGS / "smoke.yaml")
            template.common.continually_learn.save_task_checkpoints = True
            reference_config = deepcopy(template)
            reference_config.common.training.results_path = str(root / "reference")
            reference_config.common.continually_learn.checkpoint_dir = str(root / "reference_checkpoints")
            with patch.object(SemanticConsolidationClassifier, "fit", observed_fit), patch.object(
                SemanticConsolidationClassifier, "train_step", observed_step,
            ):
                reference = run(reference_config)
            interrupted_config = deepcopy(template)
            interrupted_config.common.training.results_path = str(root / "interrupted")
            interrupted_config.common.continually_learn.checkpoint_dir = str(root / "restart_checkpoints")
            with patch.object(SemanticConsolidationClassifier, "fit", interrupted_fit), self.assertRaisesRegex(
                RuntimeError, "TEST interruption",
            ):
                run(interrupted_config)
            checkpoint = load_task_checkpoint(root / "restart_checkpoints")
            self.assertEqual(checkpoint.next_task_index, 1)
            self.assertEqual(len(checkpoint.state["model_task_state"]["records"]), 1)
            resumed_config = deepcopy(template)
            resumed_config.common.training.results_path = str(root / "resumed")
            resumed_config.common.continually_learn.checkpoint_dir = str(root / "restart_checkpoints")
            resumed_config.common.continually_learn.resume_from = str(root / "restart_checkpoints")
            changed = deepcopy(resumed_config)
            changed.route.alignment_weight *= 2.
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                run(changed)
            with patch.object(SemanticConsolidationClassifier, "fit", observed_fit), patch.object(
                SemanticConsolidationClassifier, "train_step", observed_step,
            ):
                resumed = run(resumed_config)
            self.assertEqual(len(observations), 2)
            self.assertEqual(observations[0], observations[1])
            self.assertEqual(len(first_updates), 2)
            self.assertEqual(len(first_updates[0]), len(first_updates[1]))
            for expected, actual in zip(first_updates[0], first_updates[1]):
                np.testing.assert_array_equal(actual, expected)
            source = reference["model"]["generative_model"]
            target = resumed["model"]["generative_model"]
            for expected_variables, actual_variables in (
                (source.network.weights, target.network.weights),
                (source.teacher_network.weights, target.teacher_network.weights),
                (source.optimizer.variables, target.optimizer.variables),
            ):
                self.assertEqual(len(expected_variables), len(actual_variables))
                for expected, actual in zip(expected_variables, actual_variables):
                    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
            self.assertEqual(source.route_controller.introduced, target.route_controller.introduced)
            self.assertEqual(set(source.route_controller.bank.vectors), set(target.route_controller.bank.vectors))
            for class_id, pair in source.route_controller.bank.vectors.items():
                for expected, actual in zip(pair, target.route_controller.bank.vectors[class_id]):
                    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
            self.assertEqual(len(resumed["route_records"]), 2)
            for expected, actual in zip(reference["route_records"], resumed["route_records"]):
                for key in ("task", "new_classes", "seen_classes", "joint_updates", "total_updates", "invariants"):
                    self.assertEqual(actual[key], expected[key])
                for phase in ("acquisition", "consolidation"):
                    self.assertEqual(actual[phase]["history"], expected[phase]["history"])
            np.testing.assert_array_equal(
                resumed["model"]["continual_details"]["validation_accuracy_matrix"],
                reference["model"]["continual_details"]["validation_accuracy_matrix"],
            )


# Direct execution runs this focused state and actual-training recovery regression.
if __name__ == "__main__":
    unittest.main()
