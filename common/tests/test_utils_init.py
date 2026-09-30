"""Verify public notebook initialization without loading or configuring TensorFlow."""

from __future__ import annotations

import builtins
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from common import utils


class UtilsInitializationTests(unittest.TestCase):
    """Exercise source-relative path setup through the public utility entry point."""

    def test_init_prepares_paths_before_cached_tensorflow_import_and_repeats_safely(self) -> None:
        """Initialize from an unrelated directory without configuring a cached backend."""

        previous_cwd, previous_path = Path.cwd(), list(sys.path)
        root = Path(utils.__file__).resolve().parents[1]
        thesis = root / "files" / "notebooks" / "thesis"
        expected_paths = (str(root), str(thesis))
        backend = types.ModuleType("tensorflow")
        imported_backends = []
        real_import = builtins.__import__

        def checked_import(name: str, *args: object, **kwargs: object) -> object:
            """Check checkout setup at the backend import boundary and retain normal caching."""

            # Path setup must finish before either backend import.
            if name == "tensorflow":
                self.assertEqual(Path.cwd(), root)
                for entry in expected_paths:
                    self.assertEqual(sys.path.count(entry), 1)
            module = real_import(name, *args, **kwargs)
            # Record only backend imports while forwarding standard-library helpers.
            if name == "tensorflow":
                imported_backends.append(module)
            return module

        with tempfile.TemporaryDirectory() as temporary:
            try:
                directory = Path(temporary)
                sys.path[:] = [entry for entry in sys.path if entry not in expected_paths]
                os.chdir(directory)
                with patch.dict(sys.modules, {"tensorflow": backend}), \
                        patch("builtins.__import__", side_effect=checked_import), \
                        patch("subprocess.run", side_effect=AssertionError("Unexpected external command")) as run, \
                        patch("subprocess.Popen", side_effect=AssertionError("Unexpected external process")) as process:
                    self.assertIsNone(utils.init())
                    self.assertEqual(Path.cwd(), root)
                    initialized_paths = list(sys.path)
                    self.assertIs(sys.modules["tensorflow"], backend)
                    os.chdir(directory)
                    self.assertIsNone(utils.init())
                    self.assertEqual(Path.cwd(), root)
                    self.assertEqual(sys.path, initialized_paths)
                    self.assertIs(sys.modules["tensorflow"], backend)
                    self.assertEqual(imported_backends, [backend, backend])
                    run.assert_not_called()
                    process.assert_not_called()
            finally:
                os.chdir(previous_cwd)
                sys.path[:] = previous_path


# Exercise initialization independently of the broader TensorFlow test suite.
if __name__ == "__main__":
    unittest.main()
