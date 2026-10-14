"""First-task semantic CE selection preserves persistent configuration identity."""

from dataclasses import asdict
import unittest

from semantic_consolidation.config import RouteSettings
from semantic_consolidation.controller import RouteController


class TaskCEScheduleTests(unittest.TestCase):
    """Check task-cursor selection independently of expensive end-to-end fitting."""

    def test_default_keeps_the_original_settings_on_every_task(self) -> None:
        """Omitting the override retains the exact existing first/later-task behavior."""

        settings = RouteSettings(ce_weight=.25)
        controller = RouteController(settings)
        self.assertIs(controller._settings_for_task(), settings)
        controller.records.append({"task": 1})
        self.assertIs(controller._settings_for_task(), settings)

    def test_override_is_detached_and_resumes_from_the_completed_task_cursor(self) -> None:
        """A restored completed record switches to later CE without modifying the declaration."""

        settings = RouteSettings(ce_weight=0., task1_ce_weight=1.)
        declared = asdict(settings)
        controller = RouteController(settings)
        first = controller._settings_for_task()
        self.assertIsNot(first, settings)
        self.assertEqual(first.ce_weight, 1.)
        self.assertEqual(first.alignment_weight, settings.alignment_weight)
        self.assertEqual(asdict(settings), declared)
        # The native checkpoint restores completed records before scheduling later phases.
        controller.records = [{"task": 1}]
        self.assertIs(controller._settings_for_task(), settings)
        self.assertEqual(controller._settings_for_task().ce_weight, 0.)
        self.assertEqual(asdict(settings), declared)


if __name__ == "__main__":
    unittest.main()
