
"""Repository self-test registry and Python source contract inspection.

The static API parses Git-tracked and non-ignored Python sources, checks
module/class/function documentation, function annotations, and final placement
of training parameters and explicit training keywords. Token and AST checks reject
bare keyword separators, trailing commas, incorrect newline comma spacing, and
missing two-line separation after import groups, and missing one-line separation
after function/method docstrings when body code follows.
It rejects production assert statements that disappear under Python -O. It counts
statement-level branches and requires an adjacent case comment. Notebook
cells are outside this check. No TensorFlow imports are needed for static use.

run_project_self_tests additionally imports every module in
PROJECT_SELF_TEST_CLASSES, verifies exact locally defined class coverage, and
runs its embedded tests. The registry maps module-name strings to tuples of
class-name strings; importing this file defines it without running tests.
Direct execution runs the full suite, mutating random seeds and Keras state.
The public functions return coverage/result dictionaries or raise aggregated
assertion failures; verbose=True prints progress during runtime self-tests.
"""

import ast
import io
import subprocess
import tokenize

from bisect import bisect_left
from pathlib import Path


def _project_python_files() -> tuple[Path, ...]:
    """Discover repository Python sources through Git without importing them.

    The repository root is resolved from this module. Git contributes both cached
    paths and untracked paths that are not ignored; patterns select .py files only.
    A per-command safe.directory option permits this workspace without changing
    the user's Git configuration. The output is consumed in Git's reported order.

    Args:
        None.

    Returns:
        tuple[Path, ...]: Absolute Python paths, including test modules, suitable
        for read-only source parsing. No files are created or changed.

    Raises:
        subprocess.CalledProcessError: Git cannot enumerate the working tree.
        OSError: The Git executable cannot be started.
    """

    root = Path(__file__).resolve().parent
    relative_paths = subprocess.check_output(
        (
            "git", 
            "-c", 
            f"safe.directory={root.as_posix()}", 
            "ls-files", 
            "--cached", 
            "--others", 
            "--exclude-standard", 
            "--", 
            "*.py"
        ), 
        cwd=root, 
        text=True
    ).splitlines()

    return tuple(root / relative_path for relative_path in relative_paths)


def _run_static_checker_self_tests() -> None:
    """Check synthetic branches, production asserts, and training argument order.

    One in-memory source contains if/elif/else headers with expected line numbers.
    Another distinguishes a production assertion from assertions under a function
    ending in self_tests and from the same source in a tests directory. The
    Training examples distinguish explicit parameters from the unavoidable **kwargs
    tail and recover starred call argument order from AST source locations. The
    synthetic snippets are parsed, never executed or written to disk.

    Args:
        None.

    Returns:
        None: Every expected source location matched its parser result.

    Raises:
        AssertionError: A parser returns unexpected branch, assertion, or training-order locations.
    """

    branch_source = """# choose a path
if flag:
    pass
# choose another path
elif other:
    pass
# handle the fallback
else:
    pass
"""
    branch_tree = ast.parse(branch_source)
    assert _if_branch_locations(branch_tree, branch_source) == (
        (2, "if"), (5, "elif"), (8, "else")
    )

    assertion_source = """def runtime_guard():
    assert ready, "required"

def run_self_tests():
    assert exercised
"""
    assertion_tree = ast.parse(assertion_source)
    assert _production_assert_locations(
        assertion_tree, 
        Path("package/module.py")
    ) == tuple([2])
    assert _production_assert_locations(
        assertion_tree, 
        Path("common/tests/test_module.py")
    ) == ()


    training_source = """def valid(*args, training=False, **kwargs): pass
def invalid(training=False, *args): pass
valid(**options, training=False)
valid(training=False, **options)
valid(training=False, *args)
invalid_lambda = lambda training=False, mode=True: None
"""
    assert _training_order_violations(ast.parse(training_source)) == (
        (2, "invalid training must be the final explicit parameter"), 
        (4, "training keyword must be the last call argument"), 
        (5, "training keyword must be the last call argument"), 
        (6, "<lambda> training must be the final explicit parameter")
    )


def _if_branch_locations(
    tree: ast.AST, 
    source: str
) -> tuple[tuple[int, str], ...]:
    """Locate statement-level if, elif, and associated else headers.

    AST If nodes identify the branches while lexical tokens distinguish elif from
    a nested if. An else token is accepted only at its parent's indentation and
    between that branch's final body statement and its first alternative statement.
    Ternary expressions and comprehension filters are not included. Inputs are
    read without mutation and must represent the same Python source.

    Args:
        tree (ast.AST): Module tree parsed from source, with source locations.
        source (str): Original Python text used to recover keyword positions.

    Returns:
        locations (tuple[tuple[int, str], ...]): Sorted, deduplicated pairs of one-based source
        line and keyword (if, elif, or else). An unbranched module returns ().

    Raises:
        tokenize.TokenError: If the source contains incomplete lexical tokens.
        IndentationError: If tokenization encounters inconsistent indentation.
    """

    tokens = tuple(tokenize.generate_tokens(io.StringIO(source).readline))
    # Retain only lexical branch keywords; identifiers in strings are not branch tokens.
    keyword_tokens = tuple(
        token
        for token in tokens
        if token.type == tokenize.NAME and token.string in ("if", "elif", "else")
    )
    keyword_at = {token.start: token.string for token in keyword_tokens}
    locations: set[tuple[int, str]] = set()

    for node in ast.walk(tree):
        # Ignore non-branch AST nodes while locating statement branches.
        if not isinstance(node, ast.If):
            continue

        keyword = keyword_at.get((node.lineno, node.col_offset), "if")
        locations.add((node.lineno, keyword))

        # Skip branch nodes that have no elif or else arm.
        if not node.orelse:
            continue

        first_else_node = node.orelse[0]
        # Inspect the alternative keyword only for an AST If node; other statements begin an else body.
        first_keyword = (
            keyword_at.get((first_else_node.lineno, first_else_node.col_offset))
            if isinstance(first_else_node, ast.If)
            else None
        )
        # Let the nested AST node report an elif arm without inventing an else.
        if first_keyword == "elif":
            continue

        # Match else at the parent indentation and within this branch’s source interval.
        candidates = (
            token
            for token in keyword_tokens
            if token.string == "else"
            and token.start[1] == node.col_offset
            and node.body[-1].end_lineno <= token.start[0] <= first_else_node.lineno
        )
        for token in candidates:
            locations.add((token.start[0], "else"))

    return tuple(sorted(locations))


def _production_assert_locations(
    tree: ast.AST, 
    relative_path: Path
) -> tuple[int, ...]:
    """Locate assertions that would disappear from non-test code under ``-O``.

    Args:
        tree (ast.AST): Parsed Python module tree.
        relative_path (pathlib.Path): Project-relative source path.

    Returns:
        locations (tuple[int, ...]): Sorted source lines containing production assertions.
            Assertions in ``common/tests`` or beneath an executable
            ``*self_tests`` function are intentionally excluded.

    Raises:
        AttributeError: If inputs are not an AST and a path with the documented
            source-location and path attributes.
    """

    # Unit-test modules may use Python assertions as ordinary test checks.
    if "tests" in relative_path.parts:
        return ()

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    locations: list[int] = []
    for node in ast.walk(tree):
        # Continue only with optimization-sensitive assertion statements.
        if not isinstance(node, ast.Assert):
            continue

        current = node
        inside_self_test = False
        while current in parents:
            current = parents[current]
            # Embedded executable self-tests deliberately use concise asserts.
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)) \
            and current.name.endswith("self_tests"):
                inside_self_test = True
                break

        # Production guards must remain active when Python optimization is on.
        if not inside_self_test:
            locations.append(node.lineno)

    return tuple(sorted(locations))


def _training_order_violations(tree: ast.AST) -> tuple[tuple[int, str], ...]:
    """Locate training parameters and explicit keywords that precede other arguments.

    Args:
        tree (ast.AST): Parsed Python tree with source positions. Named functions,
            async functions, lambdas, and calls are inspected recursively.

    Returns:
        tuple[tuple[int, str], ...]: Sorted one-based source lines and diagnostic
            messages. Signature order includes positional-only parameters, ordinary
            parameters, *args, and keyword-only parameters; only the syntactically
            unavoidable **kwargs tail may follow training. Calls use lexical source
            positions, so named keywords, **kwargs, and even a trailing *args after
            training are detected despite AST storing positional arguments separately.
            Names hidden inside unpacked mappings and unlabeled positional bindings
            are not guessed. Strings, dictionary keys, and training_mode are ignored.
    """

    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        # Lambda arguments obey the ordering rule despite lacking annotation syntax.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            parameters = (
                tuple(node.args.posonlyargs) + tuple(node.args.args)
                + (tuple([node.args.vararg]) if node.args.vararg is not None else ())
                + tuple(node.args.kwonlyargs)
            )
            for index, parameter in enumerate(parameters):
                # Only the unavoidable **kwargs tail may follow an explicit training parameter.
                if parameter.arg == "training" and index != len(parameters) - 1:
                    owner = node.name if not isinstance(node, ast.Lambda) else "<lambda>"
                    violations.append((
                        parameter.lineno, 
                        f"{owner} training must be the final explicit parameter"
                    ))
        # AST separates starred positional arguments from keywords, losing their interleaving.
        elif isinstance(node, ast.Call):
            arguments = (*node.args, *node.keywords)
            for keyword in node.keywords:
                # Only an explicitly named training keyword establishes a known argument role.
                if keyword.arg != "training":
                    continue
                position = (keyword.lineno, keyword.col_offset)
                # Any later direct call argument violates final placement, including unpacking.
                if any((argument.lineno, argument.col_offset) > position for argument in arguments):
                    violations.append((
                        keyword.lineno, 
                        "training keyword must be the last call argument"
                    ))
    return tuple(sorted(violations))


def _argument_format_violations(
    tree: ast.AST, 
    source: str
) -> tuple[tuple[int, str], ...]:
    """Locate bare keyword separators, trailing commas, and newline comma spacing.

    Args:
        tree (ast.AST): Tree parsed from source, including UTF-8 byte-based node
            positions. Tuple nodes identify trailing unparenthesized tuple commas.
        source (str): The same valid Python text, with original physical newlines
            and whitespace retained. Comments and string contents are not code.

    Returns:
        tuple[tuple[int, str], ...]: Sorted, deduplicated one-based source lines
            and diagnostics. A bare keyword-only star is forbidden, while named
            *args and **kwargs are allowed. Commas before a closing delimiter,
            ignoring comments and layout tokens, and commas ending an AST tuple
            are forbidden, including syntax-essential singleton tuple commas.
            A comma followed only by whitespace before LF or CRLF must have
            exactly one ASCII space; commas followed by comments or code are
            outside that spacing rule. No source text or AST nodes are changed.

    Raises:
        tokenize.TokenError: The supplied source has incomplete lexical tokens.
        IndentationError: The supplied source has inconsistent indentation.
    """

    lines = io.StringIO(source).readlines()
    ignored = {
        tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, 
        tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING
    }
    tokens = tuple(
        token for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type not in ignored
    )
    positions = tuple(
        (token.start[0], len(lines[token.start[0] - 1][:token.start[1]].encode("utf-8")))
        for token in tokens
    )
    violations: set[tuple[int, str]] = set()
    for index, token in enumerate(tokens):
        # Only operator tokens can be separators; punctuation inside strings is data.
        if token.type != tokenize.OP:
            continue
        following = tokens[index + 1].string if index + 1 < len(tokens) else None
        # Valid Python uses a star immediately followed by a comma only as a bare separator.
        if token.string == "*" and following == ",":
            violations.add((token.start[0], "bare keyword-only * is not allowed"))
        # Non-comma operators have no trailing-comma or physical-line spacing contract.
        if token.string != ",":
            continue
        # Comments and layout do not make a final comma necessary before a closing delimiter.
        if following in (")", "]", "}"):
            violations.add((token.start[0], "trailing comma is not allowed"))
        line = lines[token.end[0] - 1]
        # Spacing before a newline is irrelevant on a final line with no newline.
        if not line.endswith(("\n", "\r")):
            continue
        ending_size = 2 if line.endswith("\r\n") else 1
        remainder = line[token.end[1]:-ending_size]
        # Inline comments and further code are intentionally outside the whitespace-only rule.
        if not remainder.strip() and remainder != " ":
            violations.add((token.start[0], "newline comma must be followed by exactly one ASCII space"))

    for node in ast.walk(tree):
        # Tuple spans also cover bare assignment, return, yield, and unpacking syntax.
        if not isinstance(node, ast.Tuple):
            continue
        end = (node.end_lineno, node.end_col_offset)
        index = bisect_left(positions, end) - 1
        # Empty tuples or spans without a lexical token cannot contain a trailing comma.
        if index < 0 or positions[index] < (node.lineno, node.col_offset):
            continue
        token = tokens[index]
        # Parenthesized tuples end with a delimiter and were handled lexically above.
        if token.type == tokenize.OP and token.string == ",":
            violations.add((token.start[0], "trailing comma is not allowed"))
    return tuple(sorted(violations))


def _import_spacing_violations(
    tree: ast.AST, 
    source: str
) -> tuple[tuple[int, str], ...]:
    """Require two blank lines between a completed import group and following code.

    Args:
        tree (ast.AST): Parsed source tree with locations for module, class,
            function and nested Import/ImportFrom statements.
        source (str): Matching original Python text, retaining physical newline
            and comment positions. String contents are never interpreted as imports.

    Returns:
        tuple[tuple[int, str], ...]: Sorted, deduplicated import end-line numbers
            and diagnostics. Lexically consecutive imports form one group even
            across comments, blank lines or semicolon separators. After the final
            import, exactly two whitespace-only lines must precede the first
            standalone comment or non-import code, including except/finally headers
            and dedented statements. Inline import comments stay on the import line.
            A group with no later non-import code is exempt, including comments at
            EOF. Multiline imports are measured from their final physical line.
            Neither the source nor its tree is changed.

    Raises:
        tokenize.TokenError: Source contains incomplete lexical tokens.
        IndentationError: Tokenization encounters inconsistent indentation.
    """

    imports = tuple(
        node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    lines = io.StringIO(source).readlines()
    ignored = {
        tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, 
        tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING
    }
    tokens = tuple(
        token for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type not in ignored
    )
    positions = tuple(
        (token.start[0], len(lines[token.start[0] - 1][:token.start[1]].encode("utf-8")))
        for token in tokens
    )
    import_starts = {(node.lineno, node.col_offset) for node in imports}
    violations: set[tuple[int, str]] = set()
    for node in imports:
        following = bisect_left(positions, (node.end_lineno, node.end_col_offset))
        # Semicolons separate same-line statements without becoming code themselves.
        while following < len(tokens) and tokens[following].string == ";":
            following += 1
        # EOF imports and imports followed only by another import need no separation.
        if following == len(tokens) or positions[following] in import_starts:
            continue
        cursor = node.end_lineno
        # Count physical blank lines before the first standalone comment or statement.
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        # Same-line statements cannot be separated by blank lines after their shared line.
        if tokens[following].start[0] == node.end_lineno or cursor - node.end_lineno != 2:
            violations.add((node.end_lineno, "import group must be followed by exactly two blank lines"))
    return tuple(sorted(violations))


def _docstring_spacing_violations(
    tree: ast.AST, 
    source: str
) -> tuple[tuple[int, str], ...]:
    """Require one immediately following blank line after executable function docstrings.

    Args:
        tree (ast.AST): Parsed source tree with statement locations. Synchronous,
            asynchronous, nested and decorated functions and methods are inspected.
        source (str): Matching original Python text, including physical line breaks
            and comments. Docstring text and other string contents are unchanged.

    Returns:
        tuple[tuple[int, str], ...]: Sorted, deduplicated docstring expression
            end-lines and diagnostics. Only an initial Expr(Constant(str)) is a
            docstring; module/class docstrings and docstring-only functions are
            exempt. Exactly one whitespace-only physical line must immediately
            follow the complete expression before a standalone comment or body
            code. Inline comments remain on the closing line. A same-line next
            statement is a violation. Parenthesized and concatenated docstrings
            use the expression statement's end, including its closing delimiter.
    """

    lines = io.StringIO(source).readlines()
    violations: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        # Docstring-only functions have no following body code requiring separation.
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or len(node.body) < 2:
            continue
        docstring = node.body[0]
        # Assignments, bytes, f-strings and later string expressions are not docstrings.
        if not isinstance(docstring, ast.Expr) or not isinstance(docstring.value, ast.Constant) \
                or not isinstance(docstring.value.value, str):
            continue
        cursor = docstring.end_lineno
        # Attached standalone comments follow the blank line rather than replacing it.
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        # Blank lines below a same-line body statement cannot repair that boundary.
        if node.body[1].lineno == docstring.end_lineno or cursor - docstring.end_lineno != 1:
            violations.add((docstring.end_lineno, "function docstring must be followed by exactly one blank line"))
    return tuple(sorted(violations))

def assert_static_contracts(
    exclude_paths: tuple[str, ...] = ()
) -> dict[str, int]:
    """Assert documentation, typing, branch, guard, training-order and formatting contracts.

    Lambdas are excluded only from documentation/annotation checks because Python
    cannot annotate them; their training parameter order is still checked.
    Conventional implicit self/cls parameters need no annotations. All explicit
    training parameters must follow *args and keyword-only options, immediately
    before an optional syntactic **kwargs tail. Explicit training keywords must
    follow every other call argument, including *args and **kwargs. Nested named
    functions, property methods, and calls to arbitrary constructors stay in scope.
    Actual tokens must omit bare keyword-only stars and trailing commas, including
    singleton tuple syntax. Commas followed only by whitespace before a physical
    newline require exactly one ASCII space. Completed import groups require two
    blank lines before following comments/code unless they end the file. String
    contents are ignored; named variadic parameters and grouped imports remain valid.
    Function/method docstrings with following body code require one immediately
    following blank line, before any attached standalone comments. Module/class
    docstrings and docstring-only functions are exempt from this spacing rule.

    Args:
        exclude_paths (tuple[str, ...]): Repository-relative file or directory
            paths outside this explicitly scoped assessment. The empty default
            checks every discovered source. Exclusions do not alter contracts
            on included files or the runtime class registry.

    Returns:
        counts (dict[str, int]): Counts of checked files, classes, functions, and branch
        headers when every static contract passes.

    Raises:
        AssertionError: If any tracked Python source violates a contract.
    """

    _run_static_checker_self_tests()

    failures: list[str] = []
    counts = {"files": 0, "classes": 0, "functions": 0, "branches": 0}

    for path in _project_python_files():
        relative_path = path.relative_to(Path(__file__).resolve().parent)
        # Honor caller-declared review exclusions without changing any checks.
        if any(
            relative_path == Path(excluded) or Path(excluded) in relative_path.parents
            for excluded in exclude_paths
        ):
            continue
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path), type_comments=True)
        counts["files"] += 1

        # Require every tracked Python file to explain its module-level purpose.
        if ast.get_docstring(tree) is None:
            failures.append(f"{relative_path}:1 missing module docstring")

        for node in ast.walk(tree):
            # Audit every class definition and include it in coverage totals.
            if isinstance(node, ast.ClassDef):
                counts["classes"] += 1
                # Require each class to document its responsibility.
                if ast.get_docstring(node) is None:
                    failures.append(
                        f"{relative_path}:{node.lineno} class {node.name} missing docstring"
                    )

            # Continue only with synchronous and asynchronous function definitions.
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            counts["functions"] += 1
            docstring = ast.get_docstring(node) or ""
            # Every callable explains its contract, including private helpers.
            if not docstring:
                failures.append(
                    f"{relative_path}:{node.lineno} function {node.name} missing docstring"
                )

            parameters = (
                tuple(node.args.posonlyargs)
                + tuple(node.args.args)
                + tuple(node.args.kwonlyargs)
            )
            # Implicit instance/class receivers do not require explicit annotations.
            explicit_parameters = tuple(
                parameter
                for parameter in parameters
                if parameter.arg not in ("self", "cls")
            )
            # Include a variadic positional parameter in the explicit API contract.
            if node.args.vararg is not None:
                explicit_parameters += tuple([node.args.vararg])
            # Include a variadic keyword parameter in the explicit API contract.
            if node.args.kwarg is not None:
                explicit_parameters += tuple([node.args.kwarg])

            # Collect only explicit parameters whose implementation annotation is absent.
            missing_annotations = tuple(
                parameter.arg
                for parameter in explicit_parameters
                if parameter.annotation is None
            )
            # Report explicit parameters that lack implementation annotations.
            if missing_annotations:
                failures.append(
                    f"{relative_path}:{node.lineno} {node.name} untyped parameters "
                    f"{missing_annotations}"
                )
            # Require an implementation return annotation on every function.
            if node.returns is None:
                failures.append(
                    f"{relative_path}:{node.lineno} {node.name} missing return annotation"
                )

        for line_number, message in _training_order_violations(tree):
            failures.append(f"{relative_path}:{line_number} {message}")

        for line_number, message in _argument_format_violations(tree, source):
            failures.append(f"{relative_path}:{line_number} {message}")

        for line_number, message in _import_spacing_violations(tree, source):
            failures.append(f"{relative_path}:{line_number} {message}")
        for line_number, message in _docstring_spacing_violations(tree, source):
            failures.append(f"{relative_path}:{line_number} {message}")
        branch_locations = _if_branch_locations(tree, source)
        counts["branches"] += len(branch_locations)
        lines = source.splitlines()
        # Only lexical comments count; a hash inside a string is ordinary data.
        comment_lines = {
            token.start[0]
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
        }
        # Include closing delimiters and comments before a multiline branch's first statement.
        branch_ends = {
            node.lineno: max(node.lineno, node.body[0].lineno - 1)
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
        }
        for line_number, keyword in branch_locations:
            previous = line_number - 1
            while previous > 0 and not lines[previous - 1].strip():
                previous -= 1
            end = branch_ends.get(line_number, line_number)
            following = end + 1
            while following <= len(lines) and not lines[following - 1].strip():
                following += 1
            # A case can be explained before, on, or immediately inside its header.
            if not comment_lines.intersection(
                (previous, *range(line_number, end + 1), following)
            ):
                failures.append(
                    f"{relative_path}:{line_number} {keyword} missing case comment"
                )

        for line_number in _production_assert_locations(tree, relative_path):
            failures.append(
                f"{relative_path}:{line_number} production assert disappears under -O; "
                "use an explicit validation guard"
            )

    # Fail once with the complete static-contract violation report.
    if failures:
        raise AssertionError(
            f"Static contract audit found {len(failures)} violation(s):\n"
            + "\n".join(failures)
        )

    return counts


PROJECT_SELF_TEST_CLASSES = {
    "autoencoder.vae_classifier": tuple(["VAEClassifier"]), 
    "common.callbacks.decoder_accuracy": tuple(["DecoderAccuracy"]), 
    "autoencoder.variational_autoencoder": ("_GaussianSampling", "VariationalAutoencoder"), 
    "diffusion.callbacks.batch_loss_plateau": tuple(["BatchLossPlateau"]), 
    "diffusion.callbacks.image_generator": tuple(["ImageGenerator"]), 
    "diffusion.callbacks.raw_network_validation": tuple([
        "RawNetworkValidation"
    ]), 
    "diffusion.layers.adaptive_layer_normalization_zero": tuple(["AdaLNZero"]), 
    "diffusion.layers.policy_multi_head_attention": tuple(["PolicyMultiHeadAttention"]), 
    "diffusion.layers.base_layer": tuple(["BaseLayer"]), 
    "diffusion.layers.block.di_t_decoder_block": tuple(["DiTDecoderBlock"]), 
    "diffusion.layers.block.vision_transformer_block": tuple([
        "VisionTransformerBlock" 
    ]), 
    "diffusion.layers.convolution.downsample": tuple(["ImageDownsample"]), 
    "diffusion.layers.convolution.residual_block": (
        "ResidualConvBlock", 
        "ResidualConvStack" 
    ), 
    "diffusion.layers.convolution.stage": tuple(["LayerDict"]), 
    "diffusion.layers.convolution.upsample": tuple(["ImageUpsample"]), 
    "diffusion.layers.convolution.variational_reshaper": tuple([
        "VariationalReshaper" 
    ]), 
    "diffusion.layers.drop_path": tuple(["DropPath"]), 
    "diffusion.layers.embedding.base_embedding": tuple(["BaseEmbedding"]), 
    "diffusion.layers.embedding.condition_embedding": tuple(["ConditionEmbedding"]), 
    "diffusion.layers.embedding.patch_embedding": tuple(["PatchEmbedding"]), 
    "diffusion.layers.feature_handler": tuple(["FeatureHandler"]), 
    "diffusion.layers.manipulation.downsample": tuple(["Downsample"]), 
    "diffusion.layers.manipulation.local_mixer": tuple(["LocalMixer"]), 
    "diffusion.layers.manipulation.upsample": tuple(["Upsample"]), 
    "diffusion.layers.single_token_layer": tuple(["SingleTokenLayer"]), 
    "diffusion.metrics.ensemble_accuracy": tuple(["EnsembleAccuracy"]), 
    "diffusion.models.convolution.unet": tuple(["UNet"]), 
    "diffusion.models.convolution.unet_classifier": tuple(["UNetClassifier"]), 
    "diffusion.models.transformer.di_t_classifier": tuple(["DiTClassifier"]), 
    "diffusion.models.transformer.di_t_decoder": tuple(["DiTDecoder"]), 
    "diffusion.models.transformer.di_t_encoder_decoder": tuple([
        "DiTEncoderDecoder" 
    ]), 
    "diffusion.models.transformer.di_t_encoder_decoder_classifier": tuple([
        "DiTEncoderDecoderClassifier" 
    ]), 
    "diffusion.models.transformer.diffusion_transformer": tuple([
        "DiffusionTransformer" 
    ]), 
    "diffusion.models.wrapper.diffusion_classifier": tuple(["DiffusionClassifier"]), 
    "diffusion.models.wrapper.diffusion_classifier_v2": tuple([
        "DiffusionClassifierV2" 
    ]), 
    "diffusion.models.wrapper.diffusion_model": tuple(["DiffusionModel"]), 
    "diffusion.schedulers": ("ScheduleKind", "ScheduleConfig") 
}
"""Classes still covered by embedded model and layer self-tests."""


def run_project_self_tests(
    verbose: bool = True, 
    exclude_paths: tuple[str, ...] = ()
) -> dict[str, dict[str, str]]:
    """Run and coverage-audit every class self-test in the repository.

    Each registered module is imported and its ``run_self_tests`` function is
    called in-process.  Before accepting a result, this utility discovers the
    classes whose ``__module__`` matches that module and compares them with the
    fixed registry.  It then requires the self-test result to contain exactly
    those class names and the value ``"passed"`` for each one.  Consequently a
    newly added or silently omitted class makes the project check fail instead
    of producing an incomplete success message.

    The runner deliberately continues after ordinary exceptions so one call
    reports every failing module.  Python, NumPy, and TensorFlow random seeds
    are reset before each module; Keras state and garbage are cleared after
    each module to keep the full suite deterministic and memory-efficient.

    Args:
        verbose (bool): Print one PASS/FAIL line per module and a final class
            count.  ``False`` suppresses progress output but does not suppress
            exceptions.
            Defaults to ``True``.
        exclude_paths (tuple[str, ...]): Repository-relative paths omitted only
            from the static source assessment. Defaults to no exclusions; every
            registered runtime class is tested regardless of these paths.

    Returns:
        results (dict[str, dict[str, str]]): Ordered-by-registration module results. A
        successful result covers all registered classes and every inner
        value is ``"passed"``.

    Raises:
        AssertionError: After all modules have run if a module is missing its
            runner, defined-class coverage differs from the registry, a result
            has missing/extra/non-passing entries, or any self-test raises.
    """

    import tensorflow as tf

    import numpy as np

    import gc

    import importlib

    import inspect

    import random

    import time

    import traceback


    static_counts = assert_static_contracts(exclude_paths=exclude_paths)
    results = {}
    failures = {}
    started = time.perf_counter()

    for module_name, expected_names_tuple in PROJECT_SELF_TEST_CLASSES.items():
        module_started = time.perf_counter()
        expected_names = set(expected_names_tuple)
        try:
            random.seed(1729)
            np.random.seed(1729)
            tf.random.set_seed(1729)

            module = importlib.import_module(module_name)
            # Count classes defined in this module under their own names, excluding imports and aliases.
            defined_names = {
                name
                for name, value in vars(module).items()
                if inspect.isclass(value)
                and value.__module__ == module_name
                and value.__name__ == name
            }
            assert defined_names == expected_names, (
                f"Class registry mismatch for {module_name}: defined="
                f"{sorted(defined_names)}, expected={sorted(expected_names)}"
            )

            runner = getattr(module, "run_self_tests", None)
            assert callable(runner), f"{module_name} has no callable run_self_tests"

            module_result = runner()
            assert isinstance(module_result, dict), (
                f"{module_name}.run_self_tests() returned "
                f"{type(module_result).__name__}, not dict"
            )
            assert set(module_result) == expected_names, (
                f"Self-test coverage mismatch for {module_name}: reported="
                f"{sorted(module_result)}, expected={sorted(expected_names)}"
            )
            assert all(value == "passed" for value in module_result.values()), (
                f"Non-passing self-test result from {module_name}: {module_result}"
            )
            results[module_name] = module_result

            # Report successful module timing when progress output is requested.
            if verbose:
                elapsed = time.perf_counter() - module_started
                print(
                    f"[PASS] {module_name}: {len(module_result)} "
                    f"class(es), {elapsed:.3f}s"
                )
        except Exception as error:
            failures[module_name] = {
                "error": f"{type(error).__name__}: {error}", 
                "traceback": traceback.format_exc() 
            }

            # Report failed module timing when progress output is requested.
            if verbose:
                elapsed = time.perf_counter() - module_started
                print(
                    f"[FAIL] {module_name}: {type(error).__name__}: "
                    f"{error} ({elapsed:.3f}s)"
                )
        finally:
            tf.keras.backend.clear_session()
            gc.collect()

    # Aggregate all runtime self-test tracebacks into one actionable failure.
    if failures:
        details = "\n\n".join(
            f"{module_name}\n{failure['traceback']}"
            for module_name, failure in failures.items()
        )
        raise AssertionError(
            f"{len(failures)} of {len(PROJECT_SELF_TEST_CLASSES)} module "
            f"self-test suites failed:\n\n{details}"
        )

    tested_classes = sum(len(module_result) for module_result in results.values())
    expected_classes = sum(
        len(class_names) for class_names in PROJECT_SELF_TEST_CLASSES.values()
    )
    assert tested_classes == expected_classes

    # Print the project-wide summary when progress output is requested.
    if verbose:
        elapsed = time.perf_counter() - started
        print(
            f"[PASS] Project self-tests: {tested_classes} classes across "
            f"{len(results)} modules and {static_counts['files']} statically "
            f"audited files in {elapsed:.3f}s; all enforced checks passed."
        )

    return results


# Run the complete project test registry when invoked directly.
if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude", action="append", default=[], metavar="RELATIVE_PATH", 
        help="Omit this path from static assessment; runtime registry is unchanged."
    )
    arguments = parser.parse_args()
    run_project_self_tests(exclude_paths=tuple(arguments.exclude))
