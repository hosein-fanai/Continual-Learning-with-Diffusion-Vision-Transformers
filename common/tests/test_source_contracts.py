"""Check source-style enforcement and the generated HPO notebook entry points.

Synthetic Python files exercise documentation, branch comments, argument order,
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

from common import test as project_tests


class SourceContractTests(unittest.TestCase):
    """Verify that the maintained source and notebook contracts are enforced.

    Attributes:
        _testMethodName (str): Test selected by the unittest runner.
    """

    def test_source_discovery_includes_moved_files_before_staging(self) -> None:
        """Audit relocated working files while ignoring their deleted index paths."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            moved = root / "files/notebooks/init.py"
            moved.parent.mkdir(parents=True)
            moved.write_text('"""Moved source."""\n', encoding="utf-8")
            inventory = "notebooks/init.py\nfiles/notebooks/init.py\n"
            with patch.object(project_tests, "__file__", str(root / "common/test.py")), \
                 patch.object(project_tests.subprocess, "check_output", return_value=inventory):
                self.assertEqual(project_tests._project_python_files(), tuple([moved]))

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
        root = Path(project_tests.__file__).resolve().parents[1]
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

    def test_parameter_order_preserves_kinds_and_required_defaults(self) -> None:
        """Order controls within movable partitions, including async and lambda signatures."""

        valid = (
            "def f(value, verbose, seed, dtype, name, training): pass", 
            "def f(seed, option=None, verbose=False, dtype=None, name=None, training=False): pass", 
            "def f(name, option=None): pass", 
            "def f(training, /, option=None, verbose=False, seed=None): pass", 
            "def f(value=None, verbose=False, seed=None, dtype=None, name=None, training=False, *args, **kwargs): pass", 
            "def f(*args, option=None, verbose=False, seed=None, dtype=None, name=None, training=False, **kwargs): pass", 
            "def f(*args, option=None, verbose, seed=None, dtype, name=None, training=False, **kwargs): pass", 
            "def f(training=False, *args, option=None): pass", 
            "async def f(*args, option=None, verbose=False, seed=None, training=False, **kwargs): pass", 
            "lambda value=None, verbose=False, seed=None, dtype=None, name=None, training=False, *args, **kwargs: None", 
            "def f(*seed, **training): pass", 
            "def f(training_mode=False, option=None): pass"
        )
        invalid = (
            "def f(seed=None, option=None): pass", 
            "def f(seed, value): pass", 
            "def f(name, value, /): pass", 
            "def f(verbose=False, option=None, training=False): pass", 
            "def f(dtype=None, seed=None): pass", 
            "def f(*args, name, option=None): pass", 
            "def f(*args, seed=None, verbose=False, **kwargs): pass", 
            "async def f(*args, training=False, option=None, **kwargs): pass", 
            "lambda training=False, option=None: None"
        )
        for source in valid:
            with self.subTest(source=source):
                self.assertEqual(project_tests._argument_order_violations(ast.parse(source)), ())
        for source in invalid:
            with self.subTest(source=source):
                violations = project_tests._argument_order_violations(ast.parse(source))
                self.assertEqual(len(violations), 1)
                self.assertIn("parameter controls must end each partition", violations[0][1])
        mixed = ast.parse("def f(name, value, /, dtype=None, seed=None, *args, training=False, option=None): pass")
        self.assertEqual(len(project_tests._argument_order_violations(mixed)), 3)

    def test_protocol_methods_preserve_only_required_name_parameters(self) -> None:
        """Keep framework positional protocols while checking the rest of each method."""

        signatures = (
            "__setattr__(self, name, value", 
            "suggest_categorical(self, name, choices", 
            "suggest_float(self, name, low, high", 
            "suggest_int(self, name, low, high", 
            "set_user_attr(self, name, value"
        )
        for signature in signatures:
            valid = f"class Protocol:\n    def {signature}, option=None, verbose=False, seed=None): pass"
            invalid = valid.replace("verbose=False, seed=None", "seed=None, verbose=False")
            with self.subTest(signature=signature):
                self.assertEqual(project_tests._argument_order_violations(ast.parse(valid)), ())
                self.assertEqual(len(project_tests._argument_order_violations(ast.parse(invalid))), 1)
        invalid = (
            "def suggest_float(name, low, high): pass", 
            "class C:\n    def outer(self):\n        def suggest_int(name, low, high): pass", 
            "class C:\n    def suggest_float(self, low, high, name=None, option=None): pass", 
            "class C:\n    def __setattr__(self, name, value):\n        f(training=False, mode=1)"
        )
        for source in invalid:
            with self.subTest(source=source):
                self.assertEqual(len(project_tests._argument_order_violations(ast.parse(source))), 1)

    def test_call_order_uses_lexical_positions_and_keeps_unpacking_last(self) -> None:
        """Check ordered control suffixes and starred calls without inspecting dictionary keys."""

        valid = (
            "f(1, option=True, verbose=False, seed=2, dtype=float, name='x', training=False, *args, **kwargs)", 
            "f(1, option=True, *left, *right, **first, **second)", 
            "f(training=other(mode=True), **kwargs)", 
            "f(other(training=False), mode=True)", 
            "f(mode=True, **{'training': False})", 
            "f(training_mode=False, option=None)", 
            "obj.training(False, option=None)", 
            "description = 'f(training=False, mode=True)'"
        )
        invalid = (
            "f(training=False, option=None)", 
            "f(seed=1, verbose=False)", 
            "f(name='x', dtype=float)", 
            "f(*args, training=False, **kwargs)", 
            "f(*args, option=None)", 
            "f(**kwargs, training=False)", 
            "f(**{'training': False}, mode=True)", 
            "f(*args, 1)", 
            "Config(training=section, model=model)"
        )
        message = (
            "call arguments must follow ordinary arguments, verbose, seed, dtype, "
            "name, training, *args, **kwargs order"
        )
        for source in valid:
            with self.subTest(source=source):
                self.assertEqual(project_tests._argument_order_violations(ast.parse(source)), ())
        for source in invalid:
            with self.subTest(source=source):
                self.assertEqual(
                    project_tests._argument_order_violations(ast.parse(source)), tuple([(1, message)])
                )
        nested = ast.parse("f(training=False, option=g(seed=1, verbose=True))")
        self.assertEqual(len(project_tests._argument_order_violations(nested)), 2)
        multiline = ast.parse("f(\n    *args,\n    training=False\n)")
        self.assertEqual(project_tests._argument_order_violations(multiline), tuple([(2, message)]))

    def test_public_checker_enforces_argument_order_without_new_exclusions(self) -> None:
        """Surface signature and call diagnostics through the complete public checker."""

        valid_source = (
            '"""Module."""\n'
            'def f(option: object = None, verbose: bool = False, seed: object = None, '
            'dtype: object = None, name: object = None, training: bool = False, '
            '*args: object, **kwargs: object) -> None:\n'
            '    """Document the fixture."""\n\n'
            '    pass\n'
            'f(verbose=False, training=False, **{})\n'
            'callback = lambda *args, option=None, verbose=False, training=False, **kwargs: None\n'
        )
        root = Path(project_tests.__file__).resolve().parents[1]
        (root / ".tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root / ".tmp") as directory:
            path = Path(directory) / "argument_order.py"
            with patch.object(project_tests, "_project_python_files", return_value=tuple([path])):
                path.write_text(valid_source, encoding="utf-8")
                self.assertEqual(project_tests.assert_static_contracts()["functions"], 1)
                invalid_source = valid_source.replace(
                    "option: object = None, verbose: bool = False", 
                    "verbose: bool = False, option: object = None"
                ).replace("f(verbose=False, training=False, **{})", "f(**{}, verbose=False, training=False)")
                path.write_text(invalid_source, encoding="utf-8")
                with self.assertRaises(AssertionError) as caught:
                    project_tests.assert_static_contracts()
                self.assertIn("parameter controls must end each partition", str(caught.exception))
                self.assertIn("call arguments must follow ordinary arguments", str(caught.exception))

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
        root = Path(project_tests.__file__).resolve().parents[1]
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
        root = Path(project_tests.__file__).resolve().parents[1]
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
        root = Path(project_tests.__file__).resolve().parents[1]
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
        from files.notebooks.hpo import generate_notebooks


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
