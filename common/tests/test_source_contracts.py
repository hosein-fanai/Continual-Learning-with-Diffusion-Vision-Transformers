"""Check source-style enforcement and the generated HPO notebook entry points.

Synthetic Python files exercise documentation, branch comments, training order,
and token/AST formatting, import-spacing and docstring-spacing rules without rewriting strings or
changing tuple values.
Notebook generation runs in a temporary directory without launching training or
overwriting the user's experiment notebooks.
"""

from __future__ import annotations

import ast
import json
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

import test as project_tests


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
             'else:\n    pass\n', "else missing case comment")
        )
        root = Path(project_tests.__file__).resolve().parent
        # The checker requires repository-relative fixture paths on a clean checkout too.
        (root / ".tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root / ".tmp") as directory:
            path = Path(directory) / "source.py"
            with patch.object(project_tests, "_project_python_files", return_value=tuple([path])):
                for source, message in sources:
                    path.write_text(source, encoding="utf-8")
                    with self.subTest(message=message), self.assertRaisesRegex(
                        AssertionError, message
                    ):
                        project_tests.assert_static_contracts()

                path.write_text(
                    '"""Module."""\nif (\n    True\n):\n    # Handle the true case.\n'
                    '    pass\n# Handle the false case.\nelse:\n    pass\n', 
                    encoding="utf-8"
                )
                self.assertEqual(project_tests.assert_static_contracts()["branches"], 2)

    def test_training_parameter_order_covers_every_signature_kind(self) -> None:
        """Check positional-only, keyword-only, async and lambda parameters without textual guessing."""

        valid = (
            "def f(x, training=False, **kwargs): pass", 
            "def f(*args, option=None, training=False, **kwargs): pass", 
            "def f(x, /, *, option=None, training=False): pass", 
            "def f(training, /): pass", 
            "async def f(*args, training=False, **kwargs): pass", 
            "lambda *args, option=None, training=False, **kwargs: None", 
            "def f(**training): pass", 
            "def f(training_mode=False, option=None): pass"
        )
        invalid = (
            "def f(training=False, option=None): pass", 
            "def f(training, /, option=None): pass", 
            "def f(training=False, *, option=None): pass", 
            "def f(training=False, *args): pass", 
            "async def f(*args, training=False, option=None, **kwargs): pass", 
            "lambda training=False, *args: None", 
            "lambda *, training=False, option=None: None"
        )
        for source in valid:
            with self.subTest(source=source):
                self.assertEqual(project_tests._training_order_violations(ast.parse(source)), ())
        for source in invalid:
            with self.subTest(source=source):
                violations = project_tests._training_order_violations(ast.parse(source))
                self.assertEqual(len(violations), 1)
                self.assertIn("training must be the final explicit parameter", violations[0][1])

    def test_training_keyword_order_uses_actual_call_positions(self) -> None:
        """Detect keywords after training, including AST-reordered starred arguments and nested calls."""

        valid = (
            "f(1, *args, **kwargs, training=False)", 
            "f(training=other(mode=True))", 
            "f(other(training=False), mode=True)", 
            "f(**{'training': False}, mode=True)", 
            "f(training_mode=False, option=None)", 
            "obj.training(False, option=None)", 
            "description = 'f(training=False, mode=True)'"
        )
        invalid = (
            "f(training=False, option=None)", 
            "f(training=False, **kwargs)", 
            "f(training=False, *args)", 
            "f(*args, training=False, **kwargs)", 
            "Config(training=section, model=model)"
        )
        for source in valid:
            with self.subTest(source=source):
                self.assertEqual(project_tests._training_order_violations(ast.parse(source)), ())
        for source in invalid:
            with self.subTest(source=source):
                self.assertEqual(
                    project_tests._training_order_violations(ast.parse(source)), 
                    tuple([(1, "training keyword must be the last call argument")])
                )
        nested = ast.parse("f(training=False, option=g(training=True, mode=1))")
        self.assertEqual(len(project_tests._training_order_violations(nested)), 2)
        multiline = ast.parse("f(\n    training=False,\n    *args,\n)")
        self.assertEqual(
            project_tests._training_order_violations(multiline), 
            tuple([(2, "training keyword must be the last call argument")])
        )

    def test_public_checker_enforces_training_order_without_new_exclusions(self) -> None:
        """Surface both ordering diagnostics through the complete checker on isolated source fixtures."""

        valid_source = (
            '"""Module."""\n'
            'def f(*args: object, option: object = None, training: bool = False, **kwargs: object) -> None:\n'
            '    """Document the fixture."""\n\n'
            '    pass\n'
            'f(**{}, training=False)\n'
            'callback = lambda *args, training=False, **kwargs: None\n'
        )
        root = Path(project_tests.__file__).resolve().parent
        (root / ".tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root / ".tmp") as directory:
            path = Path(directory) / "training_order.py"
            with patch.object(project_tests, "_project_python_files", return_value=tuple([path])):
                path.write_text(valid_source, encoding="utf-8")
                self.assertEqual(project_tests.assert_static_contracts()["functions"], 1)
                invalid_source = valid_source.replace(
                    "option: object = None, training: bool = False", 
                    "training: bool = False, option: object = None"
                ).replace("f(**{}, training=False)", "f(training=False, **{})")
                path.write_text(invalid_source, encoding="utf-8")
                with self.assertRaises(AssertionError) as caught:
                    project_tests.assert_static_contracts()
                self.assertIn("training must be the final explicit parameter", str(caught.exception))
                self.assertIn("training keyword must be the last call argument", str(caught.exception))

    def test_argument_format_distinguishes_bare_and_named_stars(self) -> None:
        """Reject real bare separators in functions/lambdas while retaining variadic APIs."""

        valid = (
            "def f(*args, option=None, **kwargs): pass", 
            "async def f(x, /, *args, training=False, **kwargs): pass", 
            "callback = lambda *args, option=None, **kwargs: None", 
            "f(*values, **options)", 
            "values = [*left, *right]; product = 2 * 3; power = 2 ** 3"
        )
        invalid = (
            "def f(*, option): pass", 
            "async def f(x, /, *, option=None): pass", 
            "callback = lambda *, option=None: option", 
            "def outer():\n    def inner(*, value): pass\n"
        )
        for text in valid:
            with self.subTest(source=text):
                self.assertEqual(project_tests._argument_format_violations(ast.parse(text), text), ())
        for text in invalid:
            with self.subTest(source=text):
                expected_line = 2 if text.startswith("def outer") else 1
                self.assertEqual(project_tests._argument_format_violations(ast.parse(text), text), 
                                 tuple([(expected_line, "bare keyword-only * is not allowed")]))

    def test_argument_format_rejects_trailing_delimiter_commas(self) -> None:
        """Find trailing commas through nested delimiters and comments in every container kind."""

        sources = (
            "f(1,)", "f(*args, **kwargs,)", "def f(value=1,): pass", 
            "values = [1,]", "values = {'key': 1,}", "values = {1,}", 
            "values = (1,)", "values = data[1,]", "from package import (name,)", 
            "values = [f({'key': (1, 2,)}),]", 
            "f(\n    1, # Keep this explanation.\n)", 
            "f(\n    1, \n    # Keep this explanation.\n)"
        )
        for text in sources:
            with self.subTest(source=text):
                expected_line = 2 if text.startswith("f(\n") else 1
                self.assertEqual(project_tests._argument_format_violations(ast.parse(text), text), 
                                 tuple([(expected_line, "trailing comma is not allowed")]))

    def test_argument_format_requires_exact_newline_comma_space(self) -> None:
        """Accept one ASCII space with LF/CRLF and reject missing, tabbed or repeated padding."""

        for newline in ("\n", "\r\n"):
            for padding in ("", " ", "  ", "\t", " \t"):
                text = f"values = [1,{padding}{newline}          2]{newline}"
                with self.subTest(newline=repr(newline), padding=repr(padding)):
                    expected = () if padding == " " else tuple([
                        (1, "newline comma must be followed by exactly one ASCII space")
                    ])
                    self.assertEqual(project_tests._argument_format_violations(ast.parse(text), text), expected)
        text = "values = [1, # Comments are not whitespace-only suffixes.\n          2]"
        self.assertEqual(project_tests._argument_format_violations(ast.parse(text), text), ())
        text = "values = [1, \n          2, 3]"
        self.assertEqual(project_tests._argument_format_violations(ast.parse(text), text), ())

    def test_argument_format_tracks_bare_tuple_tails_and_equivalent_rewrites(self) -> None:
        """Detect unparenthesized tuple tails with UTF-8 positions and retain tuple/unpacking values."""

        pairs = (
            ("value = 7,", "value = tuple([7])"), 
            ("value = 7, 8,", "value = 7, 8"), 
            ("value, = [7]", "[value] = [7]"), 
            ("value = 'é',", "value = tuple(['é'])"), 
            ("value = ((1,), {'key': (2,)})", "value = (tuple([1]), {'key': tuple([2])})")
        )
        for original, rewritten in pairs:
            with self.subTest(original=original):
                self.assertEqual(project_tests._argument_format_violations(ast.parse(original), original), 
                                 tuple([(1, "trailing comma is not allowed")]))
                self.assertEqual(project_tests._argument_format_violations(ast.parse(rewritten), rewritten), ())
                old_namespace, new_namespace = {}, {}
                exec(original, old_namespace)
                exec(rewritten, new_namespace)
                self.assertEqual(old_namespace["value"], new_namespace["value"])
                self.assertIs(type(old_namespace["value"]), type(new_namespace["value"]))
        for text in (
            "def f(): return 1,", "def f(): yield 1,", 
            "for value, in [(1, 2)]: pass", "values = [value for value, in rows]", 
            "é, = values"
        ):
            with self.subTest(source=text):
                self.assertEqual(project_tests._argument_format_violations(ast.parse(text), text), 
                                 tuple([(1, "trailing comma is not allowed")]))

    def test_argument_format_ignores_comments_and_literal_punctuation(self) -> None:
        """Ignore code-shaped strings, multiline literal text, Unicode separators and comments."""

        text = (
            "description = 'def f(*, x): return (x,)'\n"
            'multiline = """f(\n    x,\n)"""\n'
            "unicode_text = 'a\u2028b'\n"
            "# f(*, value,)\n"
            "values = [1, \n          {'key': (2, 3)}]\n"
            "callback = lambda *args, **kwargs: (args, kwargs)\n"
        )
        self.assertEqual(project_tests._argument_format_violations(ast.parse(text), text), ())
        self.assertEqual(project_tests._argument_format_violations(ast.parse(""), ""), ())

    def test_public_checker_enforces_argument_formatting(self) -> None:
        """Report formatting violations through the public source check with exact file locations."""

        valid = '"""Module."""\nvalues = [1, \n          2]\n'
        cases = (
            (valid.replace("1, \n", "1,\n"), "newline comma must be followed"), 
            (valid.replace("2]", "2,]"), "trailing comma is not allowed"), 
            ('"""Module."""\ndef f(*, value: int) -> int:\n'
             '    """Return the provided integer."""\n\n    return value\n', 
             "bare keyword-only \\* is not allowed")
        )
        root = Path(project_tests.__file__).resolve().parent
        (root / ".tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root / ".tmp") as directory:
            path = Path(directory) / "argument_format.py"
            with patch.object(project_tests, "_project_python_files", return_value=tuple([path])):
                path.write_text(valid, encoding="utf-8")
                self.assertEqual(project_tests.assert_static_contracts()["files"], 1)
                for text, message in cases:
                    path.write_text(text, encoding="utf-8")
                    with self.subTest(message=message), self.assertRaisesRegex(AssertionError, message):
                        project_tests.assert_static_contracts()

    def test_import_spacing_measures_groups_from_multiline_import_end(self) -> None:
        """Allow comments between imports and count exactly two blanks before following comments."""

        prefix = "import os\n# The same import group continues.\nfrom pathlib import (\n    Path\n)\n"
        for blank_count in range(4):
            text = prefix + "\n" * blank_count + "# The following code owns this comment.\nvalue = Path\n"
            with self.subTest(blank_count=blank_count):
                expected = () if blank_count == 2 else tuple([
                    (5, "import group must be followed by exactly two blank lines")
                ])
                self.assertEqual(project_tests._import_spacing_violations(ast.parse(text), text), expected)
        text = "import os  # An inline comment stays attached.\n\n\nvalue = os\n"
        self.assertEqual(project_tests._import_spacing_violations(ast.parse(text), text), ())
        text = "import os\n# A following-code comment cannot precede the required blanks.\n\n\nvalue = os\n"
        self.assertEqual(project_tests._import_spacing_violations(ast.parse(text), text), tuple([
            (1, "import group must be followed by exactly two blank lines")
        ]))

    def test_import_spacing_handles_nested_and_block_boundary_code(self) -> None:
        """Require spacing before except/finally, ordinary nested code, decorators and dedents."""

        text = (
            "try:\n    import os\nexcept ImportError:\n    import pathlib\n"
            "finally:\n    complete = True\n"
        )
        self.assertEqual(project_tests._import_spacing_violations(ast.parse(text), text), (
            (2, "import group must be followed by exactly two blank lines"), 
            (4, "import group must be followed by exactly two blank lines")
        ))
        repaired = text.replace("import os\n", "import os\n\n\n").replace(
            "import pathlib\n", "import pathlib\n\n\n"
        )
        self.assertEqual(project_tests._import_spacing_violations(ast.parse(repaired), repaired), ())
        for text, expected_line in (
            ("def f():\n    import os\nvalue = 1\n", 2), 
            ("class C:\n    def f(self):\n        import os\n        return os\n", 3), 
            ("import os\n@decorator\ndef f(): pass\n", 1)
        ):
            with self.subTest(source=text):
                self.assertEqual(project_tests._import_spacing_violations(ast.parse(text), text), tuple([
                    (expected_line, "import group must be followed by exactly two blank lines")
                ]))

    def test_import_spacing_ignores_eof_and_strings_but_checks_same_line_code(self) -> None:
        """Exempt EOF groups and literal examples while detecting real same-line and Unicode imports."""

        valid = (
            "import os", "import os\n", "import os\n# An EOF explanation.\n", 
            "description = 'import os\\nvalue = 1'\n", 
            'description = """import os\nvalue = 1\n"""\n', 
            "import os; import sys\n\n\nvalue = 1\n", 
            "é = 1; import os\n\n\nvalue = 1\n", 
            "import os\r\n\r\n\r\nvalue = 1\r\n"
        )
        for text in valid:
            with self.subTest(source=text):
                self.assertEqual(project_tests._import_spacing_violations(ast.parse(text), text), ())
        for text in ("import os; value = 1\n\n\n", "import café\nvalue = 1\n"):
            with self.subTest(source=text):
                self.assertEqual(project_tests._import_spacing_violations(ast.parse(text), text), tuple([
                    (1, "import group must be followed by exactly two blank lines")
                ]))

    def test_public_checker_enforces_import_group_spacing(self) -> None:
        """Surface import-group spacing through the public checker without disturbing other contracts."""

        valid = '"""Module."""\nimport os\n\n\n# This code uses the import.\nvalue = os\n'
        root = Path(project_tests.__file__).resolve().parent
        (root / ".tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root / ".tmp") as directory:
            path = Path(directory) / "import_spacing.py"
            with patch.object(project_tests, "_project_python_files", return_value=tuple([path])):
                path.write_text(valid, encoding="utf-8")
                self.assertEqual(project_tests.assert_static_contracts()["files"], 1)
                path.write_text(valid.replace("\n\n\n#", "\n#"), encoding="utf-8")
                with self.assertRaisesRegex(AssertionError, "import group must be followed by exactly two blank lines"):
                    project_tests.assert_static_contracts()
    def test_docstring_spacing_requires_one_blank_before_following_comments(self) -> None:
        """Count zero, one or several immediate blank lines after multiline function docs."""

        prefix = 'def f():\n    """First line.\n    Second line.\n    """\n'
        for blank_count in range(4):
            text = prefix + "\n" * blank_count + "    # Explain the return.\n    return 1\n"
            with self.subTest(blank_count=blank_count):
                expected = () if blank_count == 1 else tuple([
                    (4, "function docstring must be followed by exactly one blank line")
                ])
                self.assertEqual(project_tests._docstring_spacing_violations(ast.parse(text), text), expected)
        text = 'def f():\n    """Return one."""  # Inline documentation stays here.\n\n    return 1\n'
        self.assertEqual(project_tests._docstring_spacing_violations(ast.parse(text), text), ())
        self.assertEqual(project_tests._docstring_spacing_violations(ast.parse(text.replace("\n", "\r\n")), 
                                                                    text.replace("\n", "\r\n")), ())
        text = 'def f():\n    """Return one."""\n    \n    return 1\n'
        self.assertEqual(project_tests._docstring_spacing_violations(ast.parse(text), text), ())
        text = prefix + "    # A comment is not the separating blank line.\n\n    return 1\n"
        self.assertEqual(project_tests._docstring_spacing_violations(ast.parse(text), text), tuple([
            (4, "function docstring must be followed by exactly one blank line")
        ]))

    def test_docstring_spacing_handles_decorated_async_nested_and_concatenated_docs(self) -> None:
        """Use the complete expression boundary for parenthesized concatenation and nested methods."""

        text = (
            '@decorate\nasync def outer():\n    ("first "\n     "second")\n'
            '    @decorate\n    def inner():\n        """Inner."""\n'
            '        return 1\n    return inner\n'
        )
        self.assertEqual(project_tests._docstring_spacing_violations(ast.parse(text), text), (
            (4, "function docstring must be followed by exactly one blank line"), 
            (7, "function docstring must be followed by exactly one blank line")
        ))
        repaired = text.replace('     "second")\n', '     "second")\n\n').replace(
            '        """Inner."""\n', '        """Inner."""\n\n'
        )
        self.assertEqual(project_tests._docstring_spacing_violations(ast.parse(repaired), repaired), ())
        text = 'class C:\n    @decorate\n    def method(self):\n        "Doc " "text."\n        return 1\n'
        self.assertEqual(project_tests._docstring_spacing_violations(ast.parse(text), text), tuple([
            (4, "function docstring must be followed by exactly one blank line")
        ]))

    def test_docstring_spacing_exempts_nonfunction_docs_and_detects_same_line_bodies(self) -> None:
        """Ignore module/class docs, non-docstring literals and docstring-only bodies."""

        valid = (
            '"""Module."""\nclass C:\n    """Class."""\n    def method(self): """Only docs."""\n', 
            'def f():\n    """Only docs."""\n    # No following body statement.\n', 
            'def f():\n    value = "not a docstring"\n    "ordinary string"\n    return value\n', 
            'def f():\n    b"bytes are not docs"\n    return 1\n', 
            'def f():\n    f"formatted strings are not docs"\n    return 1\n', 
            'description = "def f(): docstring; return 1"\n'
        )
        for text in valid:
            with self.subTest(source=text):
                self.assertEqual(project_tests._docstring_spacing_violations(ast.parse(text), text), ())
        for text in ('def f(): """Doc."""; return 1', 'def f(): """Doc."""; return 1\n\n'):
            with self.subTest(source=text):
                self.assertEqual(project_tests._docstring_spacing_violations(ast.parse(text), text), tuple([
                    (1, "function docstring must be followed by exactly one blank line")
                ]))

    def test_public_checker_enforces_function_docstring_spacing(self) -> None:
        """Expose missing/excess separators through the public checker with otherwise valid source."""

        prefix = '"""Module."""\ndef f() -> int:\n    """Return the fixed integer."""\n'
        suffix = '    # Return the documented value.\n    return 1\n'
        root = Path(project_tests.__file__).resolve().parent
        (root / ".tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root / ".tmp") as directory:
            path = Path(directory) / "docstring_spacing.py"
            with patch.object(project_tests, "_project_python_files", return_value=tuple([path])):
                path.write_text(prefix + "\n" + suffix, encoding="utf-8")
                self.assertEqual(project_tests.assert_static_contracts()["functions"], 1)
                for blank_count in (0, 2):
                    path.write_text(prefix + "\n" * blank_count + suffix, encoding="utf-8")
                    with self.subTest(blank_count=blank_count), self.assertRaisesRegex(
                        AssertionError, "function docstring must be followed by exactly one blank line"
                    ):
                        project_tests.assert_static_contracts()
    def test_notebook_generator_covers_every_supported_search(self) -> None:
        """Generate the complete HPO matrix and execute its setup cells locally.

        Returns:
            None: All generated task/model pairs match the runtime search spaces.
        """

        from common.hpo import SEARCH_SPACES
        from notebooks.hpo import generate_notebooks


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
                code_cells = [cell for cell in notebook["cells"]
                              if cell["cell_type"] == "code"]
                setup_cells = [cell for cell in code_cells
                               if cell["id"] in ("bootstrap", "setup")]
                self.assertEqual([cell["id"] for cell in setup_cells], 
                                 ["bootstrap", "setup"])
                for cell in setup_cells:
                    exec("".join(cell["source"]), namespace)
                actual.add((namespace["TASK"], namespace["MODEL"]))
                self.assertEqual(notebook["nbformat"], 4)
                for cell in code_cells:
                    compile("".join(cell["source"]), str(path), "exec")
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])
            self.assertEqual(actual, expected)


# Support standalone execution as well as unittest discovery.
if __name__ == "__main__":
    unittest.main()
