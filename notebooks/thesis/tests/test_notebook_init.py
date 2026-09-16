"""Verify the centralized notebook initializer without network or model imports."""

from __future__ import annotations

import builtins
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from notebooks.thesis import bootstrap
from notebooks.thesis.tests.test_bootstrap import INITIALIZER, REPOSITORY_NAME, REPOSITORY_URL, _make_checkout


class SharedInitializerTests(unittest.TestCase):
    """Exercise the shared callable independently of a notebook's tiny loader."""

    def setUp(self) -> None:
        self.previous_cwd, self.previous_path = Path.cwd(), list(sys.path)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.addCleanup(self._restore_process)
        self.namespace = runpy.run_path(str(INITIALIZER), run_name="notebook_initializer_test")
        self.prepare = self.namespace["prepare_notebook"]
        runtime = patch.object(bootstrap, "prepare_runtime", return_value={"ready": True})
        self.runtime = runtime.start()
        self.addCleanup(runtime.stop)
        process = patch.object(bootstrap.subprocess, "run")
        self.process = process.start()
        self.addCleanup(process.stop)
        output = patch("sys.stdout", new_callable=io.StringIO)
        output.start()
        self.addCleanup(output.stop)
        real_is_dir = Path.is_dir

        def local_is_dir(path: Path) -> bool:
            return False if path in (Path("/content"), Path("/kaggle/working")) else real_is_dir(path)

        paths = patch.object(Path, "is_dir", new=local_is_dir)
        paths.start()
        self.addCleanup(paths.stop)

    def _restore_process(self) -> None:
        os.chdir(self.previous_cwd)
        sys.path[:] = self.previous_path

    def test_module_loading_does_not_change_runtime_or_import_scientific_packages(self) -> None:
        """Loading bootstrap code is safe before the environment has its dependencies."""
        previous_cwd, previous_path = Path.cwd(), list(sys.path)
        real_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".")[0] in {"tensorflow", "keras", "numpy", "common", "semantic_consolidation"}:
                raise AssertionError(f"Initializer imported {name} before preparation.")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            namespace = runpy.run_path(str(INITIALIZER), run_name="stdlib_bootstrap_probe")
        self.assertTrue(callable(namespace["prepare_notebook"]))
        self.assertEqual(Path.cwd(), previous_cwd)
        self.assertEqual(sys.path, previous_path)
        self.runtime.assert_not_called()
        self.process.assert_not_called()

    def test_existing_checkout_forwards_all_runtime_options(self) -> None:
        """Runtime policy stays explicit while discovery restores root importability."""
        root = _make_checkout(self.directory / "local checkout")
        os.chdir(root / "notebooks" / "thesis")
        result = self.prepare(checkout_name=REPOSITORY_NAME, repository=REPOSITORY_URL,
                              revision="review-branch", runtime="hosted", cuda=False, install=True)
        self.assertEqual(result, (root.resolve(), {"ready": True}))
        self.assertEqual(Path.cwd(), root.resolve())
        self.assertIn(str(root.resolve()), sys.path)
        self.runtime.assert_called_once_with(root.resolve(), runtime="hosted", cuda=False, install=True)
        self.process.assert_not_called()

    def test_missing_checkout_clones_requested_revision_then_prepares(self) -> None:
        """The shared helper owns cloning and does not silently ignore the requested revision."""
        os.chdir(self.directory)

        def clone(command: list[str], **kwargs: object) -> Mock:
            self.assertEqual(command[:2], ["git", "clone"])
            self.assertEqual(command[command.index("--branch") + 1], "review-branch")
            self.assertIn(REPOSITORY_URL, command)
            self.assertTrue(kwargs.get("check"))
            self.assertFalse(kwargs.get("shell", False))
            _make_checkout(Path(command[-1]))
            return Mock(returncode=0)

        self.process.side_effect = clone
        root, versions = self.prepare(checkout_name=REPOSITORY_NAME, repository=REPOSITORY_URL,
                                      revision="review-branch")
        self.assertEqual(root, (self.directory / REPOSITORY_NAME).resolve())
        self.assertEqual(versions, {"ready": True})
        self.process.assert_called_once()
        self.runtime.assert_called_once_with(root, runtime="auto", cuda=None, install=None)

    def test_incomplete_checkout_is_not_overwritten_or_initialized(self) -> None:
        """A colliding destination retains its files and never reaches environment setup."""
        os.chdir(self.directory)
        root = self.directory / REPOSITORY_NAME
        root.mkdir()
        retained = root / "unrelated.txt"
        retained.write_bytes(b"preserve")
        with self.assertRaises(RuntimeError):
            self.prepare(checkout_name=REPOSITORY_NAME, repository=REPOSITORY_URL)
        self.assertEqual(retained.read_bytes(), b"preserve")
        self.process.assert_not_called()
        self.runtime.assert_not_called()

    def test_path_setup_preserves_root_thesis_paths_without_duplicates(self) -> None:
        """Repeated path setup retains one entry for the root and thesis helpers."""
        root = _make_checkout(self.directory / "local checkout")
        set_paths = self.namespace["_set_paths"]
        self.assertEqual(set_paths(root), root.resolve())
        once = list(sys.path)
        self.assertEqual(set_paths(root), root.resolve())
        self.assertEqual(Path.cwd(), root.resolve())
        self.assertEqual(sys.path, once)
        self.assertEqual(sys.path.count(str(root.resolve())), 1)
        self.assertEqual(sys.path.count(str(root.resolve() / "notebooks" / "thesis")), 1)
        self.runtime.assert_not_called()

    def test_root_thesis_and_direct_import_wrappers_only_set_paths(self) -> None:
        """All import locations set paths without importing scientific packages."""
        repository_root = INITIALIZER.parent.parent.resolve()
        wrappers = ((repository_root / "init.py", "<run_path>"),
                    (INITIALIZER.parent / "thesis" / "init.py", "<run_path>"),
                    (INITIALIZER, "init"))
        real_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".")[0] in {"tensorflow", "keras", "numpy", "common", "semantic_consolidation"}:
                raise AssertionError(f"Path-only initialization imported {name}.")
            return real_import(name, *args, **kwargs)

        for path, run_name in wrappers:
            with self.subTest(wrapper=str(path), run_name=run_name):
                os.chdir(self.directory)
                with patch("builtins.__import__", side_effect=guarded_import):
                    namespace = runpy.run_path(str(path), run_name=run_name)
                self.assertEqual(namespace["REPOSITORY_ROOT"], repository_root)
                self.assertEqual(Path.cwd(), repository_root)
                self.assertIn(str(repository_root), sys.path)
                self.assertIn(str(repository_root / "notebooks" / "thesis"), sys.path)
        self.runtime.assert_not_called()
        self.process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
