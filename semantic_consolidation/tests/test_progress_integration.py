"""Real two-task recovery with scheduled replay and experimental observation."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from common.recovery import load_task_checkpoint, save_task_progress
from semantic_consolidation.config import load_route_config
from semantic_consolidation.runner import run
from semantic_consolidation.tests import test_integration


class ProgressIntegrationTests(unittest.TestCase):
    """Exercise native task reconstruction around a committed consolidation update."""

    def test_scheduled_observed_stream_resumes_mid_consolidation(self) -> None:
        """Restore real replay decisions, phase targets and retained observer cohorts.

        Returns:
            checked (None): None; final raw/teacher/optimizer/gates and phase
                histories equal an uninterrupted run on deterministic synthetic pixels.

        Raises:
            AssertionError: If actual resumed state, diagnostics or scheduling differ.
            Exception: Propagates native factory, training and checkpoint failures.
        """
        root = Path(__file__).resolve().parents[2]
        template = load_route_config(root / "semantic_consolidation/configs/extensions_smoke.yaml")
        observer_recipe = load_route_config(root / "semantic_consolidation/configs/section11_smoke.yaml")
        template.route.experimental = observer_recipe.route.experimental
        template.route.consolidation_steps = 2
        template.route.checkpoint_interval = 1
        template.common.continually_learn.save_task_checkpoints = True

        def interrupt(checkpoint_root: str, task_index: int, state: dict[str, object],
                      trackables: dict[str, object]) -> Path:
            """Interrupt only after task two's first consolidation update is durable.

            Args:
                checkpoint_root (str): Native recovery root selected by the runner.
                task_index (int): Zero-based active task cursor.
                state (dict[str, object]): Authenticated numeric progress payload.
                trackables (dict[str, object]): Existing TensorFlow iterator owner.

            Returns:
                checkpoint (Path): The existing common writer's committed snapshot.

            Raises:
                RuntimeError: Deliberate failure after the specified durable update.
            """
            checkpoint = save_task_progress(checkpoint_root, task_index, state, trackables)
            phases = [stage for stage in state["fit_progress"]["stages"] if stage["phase"] is not None]
            # Both local phases exist only once consolidation has started.
            if task_index == 1 and len(phases) == 2 and not phases[-1]["complete"]:
                raise RuntimeError("intentional committed consolidation interruption")
            return checkpoint

        with tempfile.TemporaryDirectory(prefix="phase-stream-", dir=root / ".tmp",
                                         ignore_cleanup_errors=True) as directory, patch(
                "tensorflow.keras.datasets.mnist.load_data", side_effect=test_integration.RouteIntegrationTests._pixels):
            location = Path(directory)
            reference_config = deepcopy(template)
            reference_config.common.training.results_path = str(location / "reference")
            reference_config.common.continually_learn.checkpoint_dir = str(location / "reference_checkpoints")
            reference = run(reference_config)
            interrupted_config = deepcopy(template)
            interrupted_config.common.training.results_path = str(location / "interrupted")
            interrupted_config.common.continually_learn.checkpoint_dir = str(location / "resume_checkpoints")
            with patch("common.learner.save_task_progress", interrupt), self.assertRaisesRegex(
                    RuntimeError, "intentional committed consolidation"):
                run(interrupted_config)
            saved = load_task_checkpoint(location / "resume_checkpoints")
            self.assertEqual(saved.experiment_state["active_task_index"], 1)
            self.assertGreater(saved.experiment_state["active_seconds"]["generator_fit"], 0.)
            resumed_config = deepcopy(interrupted_config)
            resumed_config.common.training.results_path = str(location / "resumed")
            resumed_config.common.continually_learn.resume_from = str(location / "resume_checkpoints")
            resumed = run(resumed_config)
            source, target = (result["model"]["generative_model"] for result in (reference, resumed))
            for expected, actual in ((source.network.weights, target.network.weights),
                                     (source.teacher_network.weights, target.teacher_network.weights),
                                     (source.optimizer.variables, target.optimizer.variables)):
                self.assertEqual(len(expected), len(actual))
                for before, after in zip(expected, actual):
                    np.testing.assert_array_equal(before.numpy(), after.numpy())
            for class_id, pair in source.route_controller.bank.vectors.items():
                for before, after in zip(pair, target.route_controller.bank.vectors[class_id]):
                    np.testing.assert_array_equal(before.numpy(), after.numpy())
            for expected, actual in zip(source.route_controller.records, target.route_controller.records):
                for phase in ("acquisition", "consolidation"):
                    self.assertEqual(expected[phase]["history"], actual[phase]["history"])
                self.assertEqual(expected["total_updates"], actual["total_updates"])
                self.assertEqual(expected["invariants"], actual["invariants"])
            self.assertEqual(len(target.section10_controller.records), 2)
            self.assertEqual(len(target.experimental_controller.records), 2)
            self.assertEqual(len(source.experimental_controller.curves), len(target.experimental_controller.curves))
            for class_id, cohort in source.experimental_controller.probe.cohorts.items():
                for key in ("images", "acquisition_features", "previous_features"):
                    np.testing.assert_array_equal(cohort[key], target.experimental_controller.probe.cohorts[class_id][key])
            np.testing.assert_array_equal(reference["model"]["continual_details"]["validation_accuracy_matrix"],
                                          resumed["model"]["continual_details"]["validation_accuracy_matrix"])
