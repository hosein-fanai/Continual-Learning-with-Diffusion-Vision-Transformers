
"""Aggregate runtime self-tests and repository-wide static contract checks."""

import ast
import io
import subprocess
import tokenize

from pathlib import Path


def _project_python_files() -> tuple[Path, ...]:
    """Return tracked and non-ignored Python sources in the working tree.

    Args:
        None.

    Returns:
        tuple[Path, ...]: Absolute project ``.py`` paths in Git's stable order.
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
    """Exercise branch and runtime-assert parsing.

    Args:
        None.

    Returns:
        None.
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
        Path("package/module.py"),
    ) == (2,)
    assert _production_assert_locations(
        assertion_tree,
        Path("common/tests/test_module.py"),
    ) == ()


def _if_branch_locations(
    tree: ast.AST, 
    source: str
) -> tuple[tuple[int, str], ...]:
    """Locate statement-level if, elif, and associated else headers.

    Args:
        tree (ast.AST): Parsed Python module tree.
        source (str): Original module source used to recover keyword tokens.

    Returns:
        tuple[tuple[int, str], ...]: Sorted unique ``(line, keyword)`` pairs.
    """

    tokens = tuple(tokenize.generate_tokens(io.StringIO(source).readline))
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
        first_keyword = (
            keyword_at.get((first_else_node.lineno, first_else_node.col_offset))
            if isinstance(first_else_node, ast.If)
            else None
        )
        # Let the nested AST node report an elif arm without inventing an else.
        if first_keyword == "elif":
            continue

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
    relative_path: Path,
) -> tuple[int, ...]:
    """Locate assertions that would disappear from non-test code under ``-O``.

    Args:
        tree (ast.AST): Parsed Python module tree.
        relative_path (pathlib.Path): Project-relative source path.

    Returns:
        tuple[int, ...]: Sorted source lines containing production assertions.
            Assertions in ``common/tests`` or beneath an executable
            ``*self_tests`` function are intentionally excluded.
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


def assert_static_contracts() -> dict[str, int]:
    """Assert documentation, typing, branch, and runtime-guard contracts.

    Lambdas are excluded because Python syntax cannot annotate lambda parameters
    or returns. Conventional implicit ``self`` and ``cls`` parameters are also
    excluded; nested named functions and property methods remain in scope.

    Args:
        None.

    Returns:
        dict[str, int]: Counts of checked files, classes, functions, and branch
        headers when every static contract passes.

    Raises:
        AssertionError: If any tracked Python source violates a contract.
    """

    _run_static_checker_self_tests()

    failures: list[str] = []
    counts = {"files": 0, "classes": 0, "functions": 0, "branches": 0}

    for path in _project_python_files():
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path), type_comments=True)
        relative_path = path.relative_to(Path(__file__).resolve().parent)
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
            public_api = not node.name.startswith("_")
            # Public APIs explain their contract; private helpers rely on types.
            if public_api and not docstring:
                failures.append(
                    f"{relative_path}:{node.lineno} function {node.name} missing docstring"
                )

            parameters = (
                tuple(node.args.posonlyargs)
                + tuple(node.args.args)
                + tuple(node.args.kwonlyargs)
            )
            explicit_parameters = tuple(
                parameter
                for parameter in parameters
                if parameter.arg not in ("self", "cls")
            )
            # Include a variadic positional parameter in the explicit API contract.
            if node.args.vararg is not None:
                explicit_parameters += (node.args.vararg,)
            # Include a variadic keyword parameter in the explicit API contract.
            if node.args.kwarg is not None:
                explicit_parameters += (node.args.kwarg,)

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

        counts["branches"] += len(_if_branch_locations(tree, source))

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
    "autoencoder.vae_classifier": ("VAEClassifier",),
    "autoencoder.decoder_accuracy_callback": ("DecoderAccuracyCallback",), 
    "autoencoder.variational_autoencoder": ("VariationalAutoencoder",), 
    "common.argument_saver": (
        "ArgumentSaver", 
        "ArgumentSaverLayer", 
        "ArgumentSaverModel", 
    ), 
    "common.config": (
        "KwargsMixin", 
        "DiffusionTransformerConfig", 
        "DiTDecoderConfig",
        "DiTEncoderDecoderConfig",
        "DiTClassifierConfig", 
        "DiTEncoderDecoderClassifierConfig",
        "UNetConfig",
        "UNetClassifierConfig",
        "DiffusionModelConfig", 
        "DiffusionClassifierConfig", 
        "DiffusionClassifierV2Config",
        "VariationalAutoencoderConfig",
        "VAEClassifierConfig",
        "DatasetConfig", 
        "ModelConfig", 
        "OptimizerConfig", 
        "ContinuallyLearnConfig",
        "TrainingConfig", 
        "ReportingConfig", 
        "Config", 
    ), 
    "common.lr_logger_callback": ("LrLoggerCallback",), 
    "common.masked_loss": ("MaskedLoss",), 
    "common.recovery": ("TaskCheckpoint",),
    "common.replay_buffer": ("ReplayBuffer",), 
    "diffusion.callbacks.batch_loss_plateau": ("BatchLossPlateau",), 
    "diffusion.callbacks.image_generator_callback": ("ImageGeneratorCallback",), 
    "diffusion.callbacks.raw_network_validation_callback": (
        "RawNetworkValidationCallback", 
    ), 
    "diffusion.layers.adaptive_layer_normalization_zero": ("AdaLNZero",), 
    "diffusion.layers.base_layer": ("BaseLayer",), 
    "diffusion.layers.block.di_t_decoder_block": ("DiTDecoderBlock",), 
    "diffusion.layers.block.vision_transformer_block": (
        "VisionTransformerBlock", 
    ), 
    "diffusion.layers.convolution.downsample": ("ImageDownsample",), 
    "diffusion.layers.convolution.residual_block": (
        "ResidualConvBlock", 
        "ResidualConvStack", 
    ), 
    "diffusion.layers.convolution.stage": ("LayerDict",), 
    "diffusion.layers.convolution.upsample": ("ImageUpsample",), 
    "diffusion.layers.convolution.variational_reshaper": (
        "VariationalReshaper", 
    ), 
    "diffusion.layers.drop_path": ("DropPath",), 
    "diffusion.layers.embedding.base_embedding": ("BaseEmbedding",), 
    "diffusion.layers.embedding.condition_embedding": ("ConditionEmbedding",), 
    "diffusion.layers.embedding.patch_embedding": ("PatchEmbedding",), 
    "diffusion.layers.feature_handler": ("FeatureHandler",), 
    "diffusion.layers.manipulation.downsample": ("Downsample",), 
    "diffusion.layers.manipulation.local_mixer": ("LocalMixer",), 
    "diffusion.layers.manipulation.upsample": ("Upsample",), 
    "diffusion.layers.single_token_layer": ("SingleTokenLayer",), 
    "diffusion.metrics.ensemble_accuracy": ("EnsembleAccuracy",), 
    "diffusion.models.convolution.unet": ("UNet",), 
    "diffusion.models.convolution.unet_classifier": ("UNetClassifier",), 
    "diffusion.models.transformer.di_t_classifier": ("DiTClassifier",), 
    "diffusion.models.transformer.di_t_decoder": ("DiTDecoder",), 
    "diffusion.models.transformer.di_t_encoder_decoder": (
        "DiTEncoderDecoder", 
    ), 
    "diffusion.models.transformer.di_t_encoder_decoder_classifier": (
        "DiTEncoderDecoderClassifier", 
    ), 
    "diffusion.models.transformer.diffusion_transformer": (
        "DiffusionTransformer", 
    ), 
    "diffusion.models.wrapper.diffusion_classifier": ("DiffusionClassifier",), 
    "diffusion.models.wrapper.diffusion_classifier_v2": (
        "DiffusionClassifierV2", 
    ), 
    "diffusion.models.wrapper.diffusion_model": ("DiffusionModel",), 
    "diffusion.schedulers": ("ScheduleKind", "ScheduleConfig"), 
}
"""Classes that must be covered by each module's ``run_self_tests`` result."""


def run_project_self_tests(
    verbose: bool = True, 
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

    Returns:
        dict[str, dict[str, str]]: Ordered-by-registration module results.  A
        successful result covers all 65 registered classes and every inner
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


    static_counts = assert_static_contracts()
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
                "traceback": traceback.format_exc(), 
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
    assert tested_classes == expected_classes == 65

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
    run_project_self_tests()
