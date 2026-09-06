"""Check source-style enforcement and the generated HPO notebook entry points.

Synthetic Python files exercise missing private docstrings and branch comments.
Notebook generation runs in a temporary directory without launching training or
overwriting the user's experiment notebooks.
"""

from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

import test as project_tests

from notebooks.hpo import generate_notebooks


class SourceContractTests(unittest.TestCase):
    """Verify that the maintained source and notebook contracts are enforced.

    Attributes:
        _testMethodName (str): Test selected by the unittest runner.
    """

    def test_private_docstrings_and_branch_comments_are_required(self) -> None:
        """Reject missing documentation even for private callables and else arms.

        Returns:
            None: Synthetic violations fail while an explained branch passes.
        """

        sources = (
            ('"""Module."""\ndef _hidden() -> None:\n    pass\n',
             "missing docstring"),
            ('"""Module."""\nif True:\n    pass\n',
             "if missing case comment"),
            ('"""Module."""\n# Handle the true case.\nif True:\n    pass\n'
             'else:\n    pass\n', "else missing case comment"),
        )
        root = Path(project_tests.__file__).resolve().parent
        # The checker requires repository-relative fixture paths on a clean checkout too.
        (root / ".tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root / ".tmp") as directory:
            path = Path(directory) / "source.py"
            with patch.object(project_tests, "_project_python_files", return_value=(path,)):
                for source, message in sources:
                    path.write_text(source, encoding="utf-8")
                    with self.subTest(message=message), self.assertRaisesRegex(
                        AssertionError, message
                    ):
                        project_tests.assert_static_contracts()

                path.write_text(
                    '"""Module."""\nif (\n    True\n):\n    # Handle the true case.\n'
                    '    pass\n# Handle the false case.\nelse:\n    pass\n',
                    encoding="utf-8",
                )
                self.assertEqual(project_tests.assert_static_contracts()["branches"], 2)

    def test_notebook_generator_covers_every_supported_search(self) -> None:
        """Generate the complete HPO matrix and execute its setup cells locally.

        Returns:
            None: All generated task/model pairs match the runtime search spaces.
        """

        from common.hpo import SEARCH_SPACES

        expected = {
            (task, model)
            for task, models in SEARCH_SPACES.items()
            for model in models
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(generate_notebooks, "ROOT", root):
                generate_notebooks.main()
            actual = set()
            for path in root.rglob("*.ipynb"):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                namespace = {}
                exec("".join(notebook["cells"][1]["source"]), namespace)
                actual.add((namespace["TASK"], namespace["MODEL"]))
                self.assertEqual(notebook["nbformat"], 4)
                for cell in notebook["cells"][1:]:
                    compile("".join(cell["source"]), str(path), "exec")
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])
            self.assertEqual(actual, expected)


# Support standalone execution as well as unittest discovery.
if __name__ == "__main__":
    unittest.main()
