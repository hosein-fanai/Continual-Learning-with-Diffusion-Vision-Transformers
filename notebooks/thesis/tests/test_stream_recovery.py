"""Native checkpoint selection and live-stream ownership; no invented outcomes."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from common.recovery import save_task_checkpoint
from notebooks.thesis import workflow
from notebooks.thesis.workflow import _acquire_stream_lease, _configure_recovery, close_run
from semantic_consolidation.config import load_route_config


class StreamRecoveryTests(unittest.TestCase):
    """Check durable state discovery and preservation independently of model fitting."""

    def test_native_committed_boundary_is_selected_without_changing_recipe(self) -> None:
        """Use the production writer/selector and keep the frozen saving policy.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        config = load_route_config(Path(__file__).resolve().parents[1] / "configs/cifar10.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "one-stream"
            self.assertIsNone(_configure_recovery(config, root))
            initial = save_task_checkpoint(root / ".initial", 0,
                state={"class_order": [0, 1], "task_groups": [[0], [1]], "restart_task_index": 0})
            self.assertEqual(Path(_configure_recovery(config, root)), initial)
            saved = save_task_checkpoint(root, 0, state={"class_order": [0, 1], "task_groups": [[0], [1]]})
            original = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            self.assertEqual(Path(_configure_recovery(config, root)), saved)
            self.assertTrue(config.common.continually_learn.save_task_checkpoints)
            self.assertEqual(original, {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()})

    def test_interruption_before_first_commit_preserves_unpublished_evidence(self) -> None:
        """Retry an unpublished native temporary, while rejecting a damaged commit.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        config = load_route_config(Path(__file__).resolve().parents[1] / "configs/cifar10.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pending = root / (".task-0000.tmp-" + "a" * 32)
            pending.mkdir()
            evidence = pending / "partial.txt"
            evidence.write_bytes(b"interrupted unpublished payload")
            self.assertIsNone(_configure_recovery(config, root))
            self.assertEqual(evidence.read_bytes(), b"interrupted unpublished payload")
            (root / "task-0000").mkdir()
            with self.assertRaises(FileNotFoundError):
                _configure_recovery(config, root)
            self.assertTrue((root / "task-0000").is_dir())

    def test_stream_lease_rejects_competing_owner_and_releases_after_failure(self) -> None:
        """A stale lock file is harmless; an active kernel's OS lease is exclusive.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stream.running.lock"
            lease = _acquire_stream_lease(path)
            try:
                with self.assertRaisesRegex(RuntimeError, "another kernel"):
                    _acquire_stream_lease(path)
                close_run({"lease": lease}, release=True)
                replacement = _acquire_stream_lease(path)
                replacement.close()
            finally:
                lease.close()

    def test_initial_publication_interruption_is_retryable_without_deleting_evidence(self) -> None:
        """Distinguish unpublished initial payloads from malformed published state.

        Args:
            None. This case owns an isolated temporary checkpoint root.

        Returns:
            checked (None): None; native unpublished files remain byte-identical.

        Raises:
            AssertionError: If selection invents state or damages retained evidence.
        """
        config = load_route_config(Path(__file__).resolve().parents[1] / "configs/cifar10.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pending = root / ".initial" / (".task-0000.tmp-" + "b" * 32)
            pending.mkdir(parents=True)
            evidence = pending / "partial.txt"
            evidence.write_bytes(b"unpublished initial state")
            self.assertIsNone(_configure_recovery(config, root))
            self.assertEqual(evidence.read_bytes(), b"unpublished initial state")
            (root / ".initial" / "task-0000").mkdir()
            with self.assertRaises(FileNotFoundError):
                _configure_recovery(config, root)
            self.assertTrue(evidence.is_file())


class DevelopmentBranchRecoveryTests(unittest.TestCase):
    """Select real native boundaries before initializing a continuation stream."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        template = Path(__file__).resolve().parents[1] / "configs/cifar100.yaml"
        recipe = yaml.safe_load(template.read_text(encoding="utf-8"))
        recipe["base_config"] = str((template.parent / recipe["base_config"]).resolve())
        recipe["common"]["training"]["results_path"] = str(self.root / "runs")
        self.recipe = self.root / "recipe.yaml"
        self.recipe.write_text(yaml.safe_dump(recipe), encoding="utf-8")
        initialize = patch.object(workflow, "_initialize", side_effect=lambda config, context: (config, context))
        self.initialize = initialize.start()
        self.addCleanup(initialize.stop)
        source = patch.object(workflow, "source_fingerprint", return_value={"sha256": "selection-test-source"})
        source.start()
        self.addCleanup(source.stop)
        config, _ = workflow.load_development(self.recipe, condition="baseline", seed=17)
        continual = config.common.continually_learn
        self.schedule = {"class_order": continual.class_order, "task_groups": continual.task_groups}
        self.default_root = Path(continual.checkpoint_dir)
        self.initialize.reset_mock()

    def _save(self, root: Path, index: int, schedule: dict | None = None) -> Path:
        """Write an authenticated metadata-only fixture without constructing a model."""
        return save_task_checkpoint(root, index, state=self.schedule if schedule is None else schedule)

    @staticmethod
    def _files(root: Path) -> dict[Path, bytes]:
        return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    def test_exact_requested_task_is_selected_before_initialization_and_old_run_is_preserved(self) -> None:
        """Task five remains the explicit source even when its old run has task six."""
        original = self.root / "original"
        requested = self._save(original, 5)
        newer = self._save(original, 6)
        before = self._files(original)
        destination = self.root / "continuation"
        config, _ = workflow.load_development(self.recipe, condition="baseline", seed=17,
                                              resume_from=requested, checkpoint_dir=destination)
        continual = config.common.continually_learn
        self.assertEqual(Path(continual.resume_from), requested.resolve())
        self.assertNotEqual(Path(continual.resume_from), newer.resolve())
        self.assertEqual(Path(continual.checkpoint_dir), destination.resolve())
        self.initialize.assert_called_once()
        self.assertIs(self.initialize.call_args.args[0], config)
        self.assertEqual(before, self._files(original))

    def test_rerun_selects_destination_checkpoint_instead_of_rewinding_to_source(self) -> None:
        """A continuation with its own completed task advances independently of the old run."""
        original = self.root / "original"
        requested = self._save(original, 5)
        self._save(original, 6)
        destination = self.root / "continuation"
        workflow.load_development(self.recipe, condition="baseline", seed=17,
                                  resume_from=requested, checkpoint_dir=destination)
        continued = self._save(destination, 6)
        before_original, before_destination = self._files(original), self._files(destination)
        config, _ = workflow.load_development(self.recipe, condition="baseline", seed=17,
                                              resume_from=requested, checkpoint_dir=destination)
        self.assertEqual(Path(config.common.continually_learn.resume_from), continued.resolve())
        self.assertEqual(Path(config.common.continually_learn.checkpoint_dir), destination.resolve())
        self.assertEqual(before_original, self._files(original))
        self.assertEqual(before_destination, self._files(destination))

    def test_explicit_source_requires_separate_destination_before_initialization(self) -> None:
        """Neither an omitted destination nor the original root can receive a rewind."""
        original = self.root / "original"
        requested = self._save(original, 5)
        self._save(original, 6)
        before = self._files(original)
        for options in ({}, {"checkpoint_dir": original / "."}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                workflow.load_development(self.recipe, condition="baseline", seed=17,
                                          resume_from=requested, **options)
        self.initialize.assert_not_called()
        self.assertEqual(before, self._files(original))

    def test_mismatched_source_schedule_is_rejected_before_initialization(self) -> None:
        """A valid checkpoint from another class order cannot seed this experiment."""
        reversed_order = list(reversed(self.schedule["class_order"]))
        other_schedule = {"class_order": reversed_order,
                          "task_groups": [reversed_order[start:start + 10] for start in range(0, 100, 10)]}
        requested = self._save(self.root / "other-order", 5, other_schedule)
        with self.assertRaises(ValueError):
            workflow.load_development(self.recipe, condition="baseline", seed=17,
                                      resume_from=requested, checkpoint_dir=self.root / "continuation")
        self.initialize.assert_not_called()

    def test_missing_explicit_source_does_not_initialize_a_fresh_run(self) -> None:
        """A misspelled source path must not silently start training from scratch."""
        with self.assertRaisesRegex(ValueError, "Not a task checkpoint directory"):
            workflow.load_development(self.recipe, condition="baseline", seed=17,
                                      resume_from=self.root / "missing" / "task-0005",
                                      checkpoint_dir=self.root / "continuation")
        self.initialize.assert_not_called()

    def test_destination_only_uses_its_native_latest_checkpoint(self) -> None:
        """An explicit existing root supports ordinary automatic continuation."""
        destination = self.root / "continuation"
        self._save(destination, 5)
        latest = self._save(destination, 6)
        config, _ = workflow.load_development(self.recipe, condition="baseline", seed=17,
                                              checkpoint_dir=destination)
        self.assertEqual(Path(config.common.continually_learn.resume_from), latest.resolve())
        self.assertEqual(Path(config.common.continually_learn.checkpoint_dir), destination.resolve())

    def test_default_identity_root_still_resumes_its_latest_checkpoint(self) -> None:
        """The existing no-override API retains automatic recovery for its recipe root."""
        saved = self._save(self.default_root, 5)
        config, _ = workflow.load_development(self.recipe, condition="baseline", seed=17)
        self.assertEqual(Path(config.common.continually_learn.checkpoint_dir), self.default_root)
        self.assertEqual(Path(config.common.continually_learn.resume_from), saved.resolve())


# Run this isolated unittest module when invoked as a script.
if __name__ == "__main__":
    unittest.main()
