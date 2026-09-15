"""Native checkpoint selection and live-stream ownership; no invented outcomes."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from common.recovery import save_task_checkpoint
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


# Run this isolated unittest module when invoked as a script.
if __name__ == "__main__":
    unittest.main()
