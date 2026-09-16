"""Exercise notebook startup without package installation, downloads, or training."""

from __future__ import annotations

import ast
import builtins
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from typing import Iterator
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

from notebooks.thesis import bootstrap


NOTEBOOKS = Path(__file__).resolve().parents[1]
INITIALIZER = NOTEBOOKS.parent / "init.py"
REPOSITORY_NAME = "Continual-Learning-with-Diffusion-Vision-Transformers"
REPOSITORY_URL = f"https://github.com/hosein-fanai/{REPOSITORY_NAME}.git"
NOTEBOOK_NAMES = (
    "00_Development.ipynb",
    "01_Freeze_Experiment.ipynb",
    "02_CIFAR10_platform.ipynb",
    "03_CIFAR10_extra_joint.ipynb",
    "04_CIFAR10_learned.ipynb",
    "05_CIFAR100_platform.ipynb",
    "06_CIFAR100_extra_joint.ipynb",
    "07_CIFAR100_learned.ipynb",
    "08_CIFAR100_random.ipynb",
    "09_CIFAR100_ce_only.ipynb",
    "10_Collect_Thesis_Results.ipynb",
    "11_Offline_Joint_Reference.ipynb",
    "12_Naive_Sequential_Reference.ipynb",
)
RUNTIME_ENVIRONMENT_KEYS = (
    "CONTINUAL_RUNTIME", "COLAB_RELEASE_TAG", "KAGGLE_KERNEL_RUN_TYPE",
    "BINDER_REPO_URL", "BINDER_LAUNCH_HOST", "JUPYTERHUB_USER",
)


def _notebook(name: str = NOTEBOOK_NAMES[0]) -> dict:
    """Read a maintained notebook without including personal experiment copies."""
    return json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))


def _setup_source(name: str = NOTEBOOK_NAMES[0]) -> str:
    """Return the first executable cell, independently of Markdown positioning."""
    return "".join(next(cell for cell in _notebook(name)["cells"]
                        if cell["cell_type"] == "code")["source"])


def _make_checkout(root: Path) -> Path:
    """Create only the filesystem markers needed to recognize a checkout."""
    root.mkdir(parents=True, exist_ok=True)
    for name in ("semantic_consolidation/config.py", "common/config.py", "diffusion/__init__.py",
                 "notebooks/thesis/bootstrap.py", "notebooks/init.py", "requirements.txt"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "notebooks/init.py":
            path.write_bytes(INITIALIZER.read_bytes())
        else:
            path.write_text("# Synthetic startup fixture.\n", encoding="utf-8")
    return root


@contextmanager
def _startup_environment(start: Path, clone: object = None,
                         hosted_paths: dict[str, Path] | None = None) -> Iterator[tuple[Mock, Mock]]:
    """Intercept network/runtime setup while restoring process-global imports and cwd."""
    previous_cwd, previous_path = Path.cwd(), list(sys.path)
    real_import = builtins.__import__
    real_is_dir = Path.is_dir
    runtime = Mock(return_value={"ready": True})
    helper = types.ModuleType("notebooks.thesis.bootstrap")
    helper.prepare_runtime = runtime

    class StartupPath:
        """Map hosted writable directories into the isolated filesystem fixture."""

        def __new__(cls, value: str) -> Path:
            return (hosted_paths or {}).get(value, Path(value))

        cwd = staticmethod(Path.cwd)

    def local_is_dir(path: Path) -> bool:
        """Keep synthetic clones in their temporary directory even on a Colab host."""
        return False if path in (Path("/content"), Path("/kaggle/working")) else real_is_dir(path)

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        """Reject a premature scientific import rather than loading a GPU runtime."""
        # Startup must prepare dependencies before importing a scientific backend.
        if name.split(".")[0] in {"tensorflow", "keras", "numpy", "common", "semantic_consolidation"}:
            raise AssertionError(f"Setup imported {name} before prepare_runtime.")
        if name == "pathlib":
            return types.SimpleNamespace(Path=StartupPath)
        return real_import(name, *args, **kwargs)

    try:
        os.chdir(start)
        with patch.dict(sys.modules, {helper.__name__: helper}), \
                patch("subprocess.run", side_effect=clone) as run, \
                patch("urllib.request.urlopen", side_effect=lambda *args, **kwargs:
                      io.BytesIO(INITIALIZER.read_bytes())), \
                patch.object(Path, "is_dir", new=local_is_dir), \
                patch("builtins.__import__", side_effect=guarded_import), \
                patch("sys.stdout", new_callable=io.StringIO):
            yield run, runtime
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path


class NotebookStartupTests(unittest.TestCase):
    """Execute the real entry cell against isolated repositories and a fake runtime."""

    def test_nested_checkout_is_reused_without_network_or_backend_imports(self) -> None:
        """A notebook opened below its checkout finds its parent and prepares once."""
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_checkout(Path(temporary) / "local project")
            nested = root / "notebooks" / "thesis"
            with _startup_environment(nested) as (run, runtime), \
                    patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected network access")) as download:
                namespace = {}
                exec(compile(_setup_source(), "notebook startup", "exec"), namespace)
                self.assertEqual(Path(namespace["ROOT"]).resolve(), root.resolve())
                self.assertEqual(Path.cwd(), root.resolve())
                self.assertIn(str(root.resolve()), sys.path)
                runtime.assert_called_once_with(root.resolve(), runtime="auto", cuda=None, install=None)
                run.assert_not_called()
                download.assert_not_called()

    def test_existing_named_checkout_preserves_edits_and_does_not_pull(self) -> None:
        """Repeated Run All setup reuses downloaded files without updating local work."""
        with tempfile.TemporaryDirectory() as temporary:
            start = Path(temporary)
            root = _make_checkout(start / REPOSITORY_NAME)
            edited = root / "local-edits.txt"
            edited.write_bytes(b"preserve user edits")
            with _startup_environment(start) as (run, runtime), \
                    patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected network access")) as download:
                namespace = {}
                exec(compile(_setup_source(), "notebook startup", "exec"), namespace)
                exec(compile(_setup_source(), "notebook startup rerun", "exec"), namespace)
                self.assertEqual(Path(namespace["ROOT"]).resolve(), root.resolve())
                self.assertEqual(runtime.call_count, 2)
                run.assert_not_called()
                download.assert_not_called()
                self.assertEqual(edited.read_bytes(), b"preserve user edits")

    def test_existing_named_checkout_precedes_empty_hosted_directories_offline(self) -> None:
        """A local child checkout remains usable when hosted writable directories exist."""
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            start, working, content = (base / name for name in ("session", "working", "content"))
            for directory in (start, working, content):
                directory.mkdir()
            root = _make_checkout(start / REPOSITORY_NAME)
            with _startup_environment(start, hosted_paths={
                "/kaggle/working": working, "/content": content,
            }) as (run, runtime), \
                    patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected network access")) as download:
                namespace = {}
                exec(compile(_setup_source(), "offline local notebook startup", "exec"), namespace)
                self.assertEqual(Path(namespace["ROOT"]).resolve(), root.resolve())
                self.assertEqual(Path.cwd(), root.resolve())
                runtime.assert_called_once_with(root.resolve(), runtime="auto", cuda=None, install=None)
                run.assert_not_called()
                download.assert_not_called()
                self.assertFalse((working / REPOSITORY_NAME).exists())
                self.assertFalse((content / REPOSITORY_NAME).exists())

    def test_existing_colab_checkout_precedes_empty_kaggle_directory_offline(self) -> None:
        """Prefer an existing Colab checkout over creating another copy in Kaggle storage."""
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            start, working, content = (base / name for name in ("input", "working", "content"))
            for directory in (start, working, content):
                directory.mkdir()
            root = _make_checkout(content / REPOSITORY_NAME)
            with _startup_environment(start, hosted_paths={
                "/kaggle/working": working, "/content": content,
            }) as (run, runtime), \
                    patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected network access")) as download:
                namespace = {}
                exec(compile(_setup_source(), "offline Colab notebook startup", "exec"), namespace)
                self.assertEqual(Path(namespace["ROOT"]).resolve(), root.resolve())
                self.assertEqual(Path.cwd(), root.resolve())
                runtime.assert_called_once_with(root.resolve(), runtime="auto", cuda=None, install=None)
                run.assert_not_called()
                download.assert_not_called()
                self.assertFalse((working / REPOSITORY_NAME).exists())
                self.assertFalse((start / REPOSITORY_NAME).exists())

    def test_absent_checkout_clones_checked_argument_list_then_prepares(self) -> None:
        """Cloning uses explicit arguments and completion is verified before imports."""
        with tempfile.TemporaryDirectory() as temporary:
            start = Path(temporary)

            def clone(command: list[str], **kwargs: object) -> Mock:
                """Materialize the requested synthetic checkout as a successful clone."""
                self.assertIsInstance(command, (list, tuple))
                self.assertEqual(command[:2], ["git", "clone"])
                self.assertIn(REPOSITORY_URL, command)
                self.assertTrue(kwargs.get("check"))
                self.assertFalse(kwargs.get("shell", False))
                destination = Path(command[-1])
                # Git may receive an absolute destination or one relative to cwd.
                if not destination.is_absolute():
                    destination = Path.cwd() / destination
                _make_checkout(destination)
                return Mock(returncode=0)

            with _startup_environment(start, clone) as (run, runtime), \
                    patch("urllib.request.urlopen", side_effect=lambda *args, **kwargs:
                          io.BytesIO(INITIALIZER.read_bytes())) as download:
                namespace = {}
                exec(compile(_setup_source(), "notebook startup", "exec"), namespace)
                run.assert_called_once()
                runtime.assert_called_once_with((start / REPOSITORY_NAME).resolve(),
                                                runtime="auto", cuda=None, install=None)
                download.assert_called_once_with(
                    f"https://raw.githubusercontent.com/hosein-fanai/{REPOSITORY_NAME}/main/notebooks/init.py",
                    timeout=30,
                )

    def test_kaggle_clone_uses_writable_working_directory_before_content(self) -> None:
        """A hosted notebook opened in input storage clones into Kaggle working storage."""
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            start, working, content = (base / name for name in ("input", "working", "content"))
            for directory in (start, working, content):
                directory.mkdir()

            def clone(command: list[str], **kwargs: object) -> Mock:
                self.assertTrue(kwargs.get("check"))
                self.assertEqual(Path(command[-1]), working / REPOSITORY_NAME)
                _make_checkout(Path(command[-1]))
                return Mock(returncode=0)

            with _startup_environment(start, clone, {
                "/kaggle/working": working, "/content": content,
            }) as (run, runtime):
                namespace = {}
                exec(compile(_setup_source(), "Kaggle notebook startup", "exec"), namespace)
                run.assert_called_once()
                runtime.assert_called_once_with((working / REPOSITORY_NAME).resolve(),
                                                runtime="auto", cuda=None, install=None)
                self.assertFalse((start / REPOSITORY_NAME).exists())
                self.assertFalse((content / REPOSITORY_NAME).exists())

    def test_incomplete_existing_destination_is_preserved_and_rejected(self) -> None:
        """An unrelated existing directory is never overwritten or passed to setup."""
        with tempfile.TemporaryDirectory() as temporary:
            start = Path(temporary)
            incomplete = start / REPOSITORY_NAME
            incomplete.mkdir()
            retained = incomplete / "preserve.txt"
            retained.write_bytes(b"incomplete or unrelated checkout")
            with _startup_environment(start) as (run, runtime):
                with self.assertRaises((RuntimeError, FileExistsError)):
                    exec(compile(_setup_source(), "notebook startup", "exec"), {})
                run.assert_not_called()
                runtime.assert_not_called()
                self.assertEqual(retained.read_bytes(), b"incomplete or unrelated checkout")

    def test_clone_failure_reports_recovery_without_attempting_runtime_setup(self) -> None:
        """Network or git failures stop before any dependency installation is attempted."""
        with tempfile.TemporaryDirectory() as temporary:
            start = Path(temporary)
            failure = bootstrap.subprocess.CalledProcessError(128, ["git", "clone"])
            with _startup_environment(start, failure) as (run, runtime):
                with self.assertRaisesRegex(RuntimeError, "Enable Internet access") as caught:
                    exec(compile(_setup_source(), "failed notebook clone", "exec"), {})
                self.assertIs(caught.exception.__cause__, failure)
                run.assert_called_once()
                runtime.assert_not_called()

    def test_initializer_download_failure_preserves_the_network_error(self) -> None:
        """Failure to fetch setup stops before cloning or changing the environment."""
        with tempfile.TemporaryDirectory() as temporary:
            failure = URLError("Synthetic offline session")
            with _startup_environment(Path(temporary)) as (run, runtime), \
                    patch("urllib.request.urlopen", side_effect=failure):
                with self.assertRaises(URLError) as caught:
                    exec(compile(_setup_source(), "failed initializer fetch", "exec"), {})
                self.assertIs(caught.exception, failure)
                run.assert_not_called()
                runtime.assert_not_called()

    def test_active_notebooks_use_the_shared_loader(self) -> None:
        """Notebooks outside the excluded old archive share the maintained entry point."""
        paths = sorted(path for path in NOTEBOOKS.parent.rglob("*.ipynb")
                       if ".ipynb_checkpoints" not in path.parts
                       and not path.is_relative_to(NOTEBOOKS.parent / "old"))
        self.assertGreaterEqual(len(paths), len(NOTEBOOK_NAMES))
        expected = _setup_source().rstrip("\n")
        for path in paths:
            with self.subTest(notebook=str(path.relative_to(NOTEBOOKS.parent))):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                first = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
                source = "".join(first["source"])
                self.assertEqual(source.rstrip("\n"), expected)
                ast.parse(source, filename=str(path))

    def test_all_thirteen_entry_cells_and_colab_links_match(self) -> None:
        """Maintain identical Python startup and direct Colab links in canonical notebooks."""
        expected_setup = _setup_source()
        for name in NOTEBOOK_NAMES:
            with self.subTest(notebook=name):
                notebook = _notebook(name)
                self.assertEqual(notebook["nbformat"], 4)
                # Notebook editors may omit a cell's optional terminal newline.
                self.assertEqual(_setup_source(name).rstrip("\n"), expected_setup.rstrip("\n"))
                badge = ("https://colab.research.google.com/github/hosein-fanai/"
                         f"{REPOSITORY_NAME}/blob/main/notebooks/thesis/{name}")
                markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                                     if cell["cell_type"] == "markdown")
                self.assertIn(badge, markdown)
                for index, cell in enumerate(notebook["cells"]):
                    # Plain Python cells remain executable outside IPython preprocessing.
                    if cell["cell_type"] == "code":
                        ast.parse("".join(cell["source"]), filename=f"{name}:cell{index}")


class RuntimeDetectionTests(unittest.TestCase):
    """Recognize provider-specific signals without treating every notebook as hosted."""

    def setUp(self) -> None:
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        for key in RUNTIME_ENVIRONMENT_KEYS:
            os.environ.pop(key, None)
        modules = patch.dict(sys.modules)
        modules.start()
        self.addCleanup(modules.stop)
        sys.modules.pop("google.colab", None)

    def test_provider_environment_signals(self) -> None:
        for key, runtime in (
            ("KAGGLE_KERNEL_RUN_TYPE", "kaggle"),
            ("BINDER_REPO_URL", "binder"),
            ("BINDER_LAUNCH_HOST", "binder"),
            ("COLAB_RELEASE_TAG", "colab"),
        ):
            with self.subTest(signal=key), patch.dict(os.environ, {key: "test-runtime"}):
                self.assertEqual(bootstrap.detect_runtime(), runtime)

    def test_loaded_colab_module_is_a_session_signal(self) -> None:
        sys.modules["google.colab"] = types.ModuleType("google.colab")
        self.assertEqual(bootstrap.detect_runtime(), "colab")

    def test_kaggle_wins_when_its_image_contains_colab_signals(self) -> None:
        os.environ["KAGGLE_KERNEL_RUN_TYPE"] = "Interactive"
        os.environ["COLAB_RELEASE_TAG"] = "inherited-colab-base"
        sys.modules["google.colab"] = types.ModuleType("google.colab")
        self.assertEqual(bootstrap.detect_runtime(), "kaggle")

    def test_binder_provider_signal_precedes_colab_base_image_signal(self) -> None:
        os.environ["BINDER_REPO_URL"] = "https://example.invalid/notebook.git"
        os.environ["COLAB_RELEASE_TAG"] = "inherited-colab-base"
        self.assertEqual(bootstrap.detect_runtime(), "binder")

    def test_installed_colab_package_and_generic_jupyterhub_are_not_hosted_signals(self) -> None:
        os.environ["JUPYTERHUB_USER"] = "local-user"
        with patch("importlib.util.find_spec", return_value=Mock()) as find_spec:
            self.assertEqual(bootstrap.detect_runtime(), "local")
        find_spec.assert_not_called()

    def test_explicit_runtime_beats_environment_and_detected_provider(self) -> None:
        os.environ["CONTINUAL_RUNTIME"] = "binder"
        os.environ["KAGGLE_KERNEL_RUN_TYPE"] = "Interactive"
        for runtime in ("local", "colab", "kaggle", "binder", "studiolab", "hosted"):
            with self.subTest(runtime=runtime):
                self.assertEqual(bootstrap.detect_runtime(runtime), runtime)
        self.assertEqual(bootstrap.detect_runtime(), "binder")

    def test_unknown_explicit_or_environment_runtime_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bootstrap.detect_runtime("unknown-provider")
        os.environ["CONTINUAL_RUNTIME"] = "unknown-provider"
        with self.assertRaises(ValueError):
            bootstrap.detect_runtime()


class RuntimePreparationTests(unittest.TestCase):
    """Resolve dependency metadata and installation plans without importing TensorFlow."""

    def setUp(self) -> None:
        """Provide a tiny manifest and isolate metadata, modules, environment, and pip."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "requirements.txt").write_text(
            "tensorflow==2.20.0\nkeras==3.11.2\nnumpy>=1.26,<3\n", encoding="utf-8")
        self.versions = {"tensorflow": "2.20.0", "keras": "3.11.2", "numpy": "2.3.2"}
        self.expected_cuda = True
        modules = patch.dict(sys.modules)
        modules.start()
        self.addCleanup(modules.stop)
        sys.modules.pop("tensorflow", None)
        sys.modules.pop("keras", None)
        sys.modules.pop("google.colab", None)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        for key in RUNTIME_ENVIRONMENT_KEYS:
            os.environ.pop(key, None)
        output = patch("sys.stdout", new_callable=io.StringIO)
        output.start()
        self.addCleanup(output.stop)
        metadata = patch.object(bootstrap.metadata, "version", side_effect=self._version)
        metadata.start()
        self.addCleanup(metadata.stop)
        dependencies = patch.object(bootstrap.metadata, "requires", return_value=[])
        dependencies.start()
        self.addCleanup(dependencies.stop)
        process = patch.object(bootstrap.subprocess, "run")
        self.run = process.start()
        self.addCleanup(process.stop)

    def _version(self, package: str) -> str:
        """Return installed-version fixtures through the real metadata API boundary."""
        # Missing fixture entries model a package absent from the interpreter.
        if package not in self.versions:
            raise bootstrap.metadata.PackageNotFoundError(package)
        return self.versions[package]

    def _pip(self, command: list[str], **kwargs: object) -> Mock:
        """Write a synthetic resolver report, then emulate the approved installation."""
        self.assertEqual(command[:4], [sys.executable, "-m", "pip", "install"])
        self.assertTrue(kwargs.get("check"))
        self.assertFalse(kwargs.get("shell", False))
        if not self.expected_cuda:
            # Managed runtimes resolve the shared manifest's selected requirements without
            # requesting TensorFlow's separately packaged CUDA dependencies.
            self.assertNotIn("-r", command)
            self.assertIn("tensorflow==2.20.0", command)
            self.assertIn("keras==3.11.2", command)
            self.assertIn("numpy<3,>=1.26", command)
            self.assertFalse(any("and-cuda" in argument for argument in command))
        else:
            self.assertEqual(command[command.index("-r") + 1], str(self.root / "requirements.txt"))
        # The resolver reports its plan before any package is changed.
        if "--dry-run" in command:
            report = Path(command[command.index("--report") + 1])
            report.write_text(json.dumps({"install": [{"metadata": {"name": "tensorflow"}}]}),
                              encoding="utf-8")
        # Only the subsequent installation changes the synthetic package metadata.
        else:
            self.versions["tensorflow"] = "2.20.0"
        return Mock(returncode=0)

    def test_compatible_runtime_needs_no_pip_or_scientific_import(self) -> None:
        """Both local verification and hosted setup preserve compatible installed packages."""
        real_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            """Fail instead of loading any scientific runtime during setup."""
            # Package metadata alone must suffice for compatibility decisions.
            if name.split(".")[0] in {"tensorflow", "keras", "numpy"}:
                raise AssertionError(f"Runtime preparation imported {name}.")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            local = bootstrap.prepare_runtime(self.root, install=False)
            hosted = bootstrap.prepare_runtime(self.root, install=True)
        self.assertEqual(local, hosted)
        self.assertEqual(local["tensorflow"], "2.20.0")
        self.assertEqual(local["keras"], "3.11.2")
        self.assertEqual(os.environ["KERAS_BACKEND"], "tensorflow")
        self.run.assert_not_called()

    def test_local_mismatch_reports_required_version_without_installing(self) -> None:
        """Local checks leave the user's environment intact when an upgrade is needed."""
        self.versions["tensorflow"] = "2.19.0"
        with self.assertRaisesRegex(RuntimeError, "tensorflow==2.20.0"):
            bootstrap.prepare_runtime(self.root, install=False)
        self.run.assert_not_called()

    def test_default_setup_does_not_install_outside_colab(self) -> None:
        """The notebook's argument-free call protects an ordinary local interpreter."""
        self.versions.pop("tensorflow")
        os.environ.pop("COLAB_RELEASE_TAG", None)
        sys.modules.pop("google.colab", None)
        with self.assertRaisesRegex(RuntimeError, "tensorflow==2.20.0"):
            bootstrap.prepare_runtime(self.root)
        self.run.assert_not_called()

    def test_default_colab_setup_installs_missing_requirements(self) -> None:
        """The hosted session marker enables installation before scientific imports."""
        manifest = self.root / "requirements.txt"
        manifest.write_text(
            "tensorflow[and-cuda]==2.20.0\nkeras==3.11.2\nnumpy>=1.26,<3\n",
            encoding="utf-8",
        )
        original_manifest = manifest.read_bytes()
        self.versions.pop("tensorflow")
        os.environ["COLAB_RELEASE_TAG"] = "synthetic-colab-runtime"
        self.expected_cuda = False
        self.run.side_effect = self._pip
        with patch.object(bootstrap.metadata, "packages_distributions", return_value={}):
            result = bootstrap.prepare_runtime(self.root)
        self.assertEqual(result["tensorflow"], "2.20.0")
        self.assertEqual(self.run.call_count, 2)
        self.assertEqual(manifest.read_bytes(), original_manifest)

    def test_colab_without_install_permission_still_omits_cuda_extra(self) -> None:
        """The runtime determines CUDA selection independently of install permission."""
        (self.root / "requirements.txt").write_text(
            "tensorflow[and-cuda]==2.20.0\nkeras==3.11.2\nnumpy>=1.26,<3\n",
            encoding="utf-8",
        )
        sys.modules["google.colab"] = types.ModuleType("google.colab")
        with patch.object(bootstrap.metadata, "requires") as distribution_requirements:
            result = bootstrap.prepare_runtime(self.root, install=False)
        self.assertEqual(result["tensorflow"], "2.20.0")
        # With no requested extra, installed distribution dependencies do not
        # require a second CUDA inventory just to validate the manifest.
        distribution_requirements.assert_not_called()
        self.run.assert_not_called()

    def test_hosted_providers_install_missing_packages_with_provider_cuda_defaults(self) -> None:
        """Every supported hosted provider prepares a fresh session without an install override."""
        manifest = self.root / "requirements.txt"
        manifest.write_text(
            "tensorflow[and-cuda]==2.20.0\nkeras==3.11.2\nnumpy>=1.26,<3\n",
            encoding="utf-8",
        )
        original_manifest = manifest.read_bytes()
        for runtime, cuda in (("colab", False), ("kaggle", False), ("binder", False),
                              ("studiolab", True), ("hosted", True)):
            with self.subTest(runtime=runtime):
                self.versions.pop("tensorflow", None)
                self.expected_cuda = cuda
                self.run.reset_mock()
                self.run.side_effect = self._pip
                with patch.object(bootstrap.metadata, "packages_distributions", return_value={}):
                    result = bootstrap.prepare_runtime(self.root, runtime=runtime)
                self.assertEqual(result["tensorflow"], "2.20.0")
                self.assertEqual(self.run.call_count, 2)
                self.assertEqual(manifest.read_bytes(), original_manifest)

    def test_auto_kaggle_and_binder_install_without_cuda_wheels(self) -> None:
        """Provider environment detection reaches the automatic installer policy."""
        for key in ("KAGGLE_KERNEL_RUN_TYPE", "BINDER_REPO_URL"):
            with self.subTest(signal=key), patch.dict(os.environ, {key: "test-session"}):
                self.versions.pop("tensorflow", None)
                self.expected_cuda = False
                self.run.reset_mock()
                self.run.side_effect = self._pip
                with patch.object(bootstrap.metadata, "packages_distributions", return_value={}):
                    bootstrap.prepare_runtime(self.root)
                self.assertEqual(self.run.call_count, 2)

    def test_cuda_override_controls_pip_arguments_independently_of_provider(self) -> None:
        """Users can opt out of CUDA in a generic host or request it on a managed host."""
        (self.root / "requirements.txt").write_text(
            "tensorflow[and-cuda]==2.20.0\nkeras==3.11.2\nnumpy>=1.26,<3\n",
            encoding="utf-8",
        )
        for runtime, cuda in (("local", False), ("studiolab", False),
                              ("hosted", False), ("kaggle", True)):
            with self.subTest(runtime=runtime, cuda=cuda):
                self.versions.pop("tensorflow", None)
                self.expected_cuda = cuda
                self.run.reset_mock()
                self.run.side_effect = self._pip
                with patch.object(bootstrap.metadata, "packages_distributions", return_value={}):
                    bootstrap.prepare_runtime(self.root, install=True, runtime=runtime, cuda=cuda)
                self.assertEqual(self.run.call_count, 2)

    def test_invalid_cuda_override_does_not_start_installation(self) -> None:
        """Values such as the string False cannot accidentally enable CUDA installation."""
        self.versions.pop("tensorflow")
        for cuda in ("False", 0, 1):
            with self.subTest(cuda=cuda), self.assertRaisesRegex(TypeError, "cuda"):
                bootstrap.prepare_runtime(self.root, runtime="hosted", cuda=cuda)
        self.run.assert_not_called()

    def test_local_override_prevents_automatic_installation_inside_a_detected_provider(self) -> None:
        """An explicit local policy protects a connected local environment from installation."""
        os.environ["COLAB_RELEASE_TAG"] = "test-session"
        self.versions.pop("tensorflow")
        with self.assertRaisesRegex(RuntimeError, "tensorflow==2.20.0"):
            bootstrap.prepare_runtime(self.root, runtime="local")
        self.run.assert_not_called()

    def test_install_false_prevents_package_changes_on_every_hosted_provider(self) -> None:
        """A verification request overrides automatic installation on hosted sessions."""
        self.versions.pop("tensorflow")
        for runtime in ("colab", "kaggle", "binder", "studiolab", "hosted"):
            with self.subTest(runtime=runtime), self.assertRaisesRegex(RuntimeError, "tensorflow==2.20.0"):
                bootstrap.prepare_runtime(self.root, install=False, runtime=runtime)
        self.run.assert_not_called()

    def test_resolver_plan_precedes_install_and_successful_rerun_does_not_install(self) -> None:
        """A missing package is resolved, installed, verified, and then reused."""
        self.versions.pop("tensorflow")
        self.run.side_effect = self._pip
        with patch.object(bootstrap.metadata, "packages_distributions", return_value={}):
            result = bootstrap.prepare_runtime(self.root, install=True)
        self.assertEqual(result["tensorflow"], "2.20.0")
        self.assertEqual(self.run.call_count, 2)
        self.assertIn("--dry-run", self.run.call_args_list[0].args[0])
        self.assertNotIn("--dry-run", self.run.call_args_list[1].args[0])
        bootstrap.prepare_runtime(self.root, install=True)
        self.assertEqual(self.run.call_count, 2)

    def test_local_verification_accepts_cuda_provided_by_container_system_libraries(self) -> None:
        """Read-only local startup does not demand pip CUDA wheels from a GPU image."""
        (self.root / "requirements.txt").write_text(
            "tensorflow[and-cuda]==2.20.0\nkeras==3.11.2\nnumpy>=1.26,<3\n",
            encoding="utf-8",
        )
        with patch.object(bootstrap.metadata, "requires", return_value=[
            "nvidia-cudnn-cu12>=9,<10; extra == 'and-cuda'",
        ]):
            result = bootstrap.prepare_runtime(self.root, install=False)
        self.assertEqual(result["tensorflow"], "2.20.0")
        self.run.assert_not_called()

    def test_explicit_local_install_resolves_missing_cuda_extra_even_when_tensorflow_exists(self) -> None:
        """Installed TensorFlow alone cannot make an explicitly requested CUDA install a no-op."""
        manifest = self.root / "requirements.txt"
        manifest.write_text(
            "tensorflow[and-cuda]==2.20.0\nkeras==3.11.2\nnumpy>=1.26,<3\n",
            encoding="utf-8",
        )
        original_manifest = manifest.read_bytes()

        def requires(package: str) -> list[str]:
            """Only the selected extra contributes a missing dependency."""
            if package == "tensorflow":
                return ["numpy>=1.26", "nvidia-cudnn-cu12>=9,<10; extra == 'and-cuda'",
                        "other-optional-dependency>=1; extra == 'another-extra'"]
            return []

        def installer(command: list[str], **kwargs: object) -> Mock:
            """Resolve and install the requested extra without mutating any real packages."""
            self.assertTrue(kwargs.get("check"))
            self.assertEqual(command[command.index("-r") + 1], str(manifest))
            if "--dry-run" in command:
                report = Path(command[command.index("--report") + 1])
                report.write_text(json.dumps({"install": [{"metadata": {
                    "name": "nvidia-cudnn-cu12",
                }}]}), encoding="utf-8")
            else:
                self.versions["nvidia-cudnn-cu12"] = "9.3.0"
            return Mock(returncode=0)

        self.run.side_effect = installer
        with patch.object(bootstrap.metadata, "requires", side_effect=requires), \
                patch.object(bootstrap.metadata, "packages_distributions", return_value={}):
            result = bootstrap.prepare_runtime(self.root, install=True)
            self.assertEqual(result["tensorflow"], "2.20.0")
            self.assertEqual(self.run.call_count, 2)
            bootstrap.prepare_runtime(self.root, install=True)
        self.assertEqual(self.run.call_count, 2)
        self.assertEqual(manifest.read_bytes(), original_manifest)

    def test_loaded_old_tensorflow_is_rejected_even_when_disk_metadata_is_new(self) -> None:
        """New installed metadata cannot conceal an incompatible already imported backend."""
        loaded = types.ModuleType("tensorflow")
        loaded.__version__ = "2.19.0"
        sys.modules["tensorflow"] = loaded
        with self.assertRaisesRegex(RuntimeError, "already loaded.*Restart"):
            bootstrap.prepare_runtime(self.root, install=True)
        self.run.assert_not_called()

    def test_plan_cannot_replace_an_already_loaded_dependency(self) -> None:
        """Reject a resolver's replacement before executing any real installation."""
        self.versions["tensorflow"] = "2.19.0"
        sys.modules["numpy"] = types.ModuleType("numpy")

        def resolver(command: list[str], **kwargs: object) -> Mock:
            """Report a dependency replacement that would invalidate loaded NumPy."""
            self.assertIn("--dry-run", command)
            self.assertTrue(kwargs.get("check"))
            report = Path(command[command.index("--report") + 1])
            report.write_text(json.dumps({"install": [{"metadata": {"name": "numpy"}}]}),
                              encoding="utf-8")
            return Mock(returncode=0)

        self.run.side_effect = resolver
        with patch.object(bootstrap.metadata, "packages_distributions", return_value={"numpy": ["numpy"]}):
            with self.assertRaisesRegex(RuntimeError, "already loaded packages: numpy"):
                bootstrap.prepare_runtime(self.root, install=True)
        self.run.assert_called_once()
        self.assertEqual(self.versions["tensorflow"], "2.19.0")


class RequirementsSelectionTests(unittest.TestCase):
    """Check CUDA selection against the shared manifest on each supported host."""

    def _tensorflow(self, *, system: str = "Linux", machine: str = "x86_64",
                    cuda: bool = True) -> object:
        """Evaluate the real dependency manifest for an isolated platform context."""
        from packaging.markers import default_environment

        environment = default_environment()
        environment.update(
            platform_system=system,
            platform_machine=machine,
            sys_platform={"Linux": "linux", "Windows": "win32", "Darwin": "darwin"}[system],
            os_name="nt" if system == "Windows" else "posix",
        )
        with patch("packaging.markers.default_environment", return_value=environment):
            requirements = bootstrap._requirements(NOTEBOOKS.parents[1] / "requirements.txt",
                                                   cuda=cuda)
        tensorflow = [item for item in requirements if item.name == "tensorflow"]
        self.assertEqual(len(tensorflow), 1)
        self.assertEqual(str(tensorflow[0].specifier), "==2.20.0")
        return tensorflow[0]

    def test_linux_x86_64_requests_cuda_extra_outside_colab(self) -> None:
        """Ordinary supported Linux installs retain TensorFlow's CUDA extra."""
        self.assertEqual(self._tensorflow().extras, {"and-cuda"})

    def test_linux_colab_uses_plain_tensorflow(self) -> None:
        """Colab's Linux platform must not cause a duplicate CUDA dependency install."""
        self.assertEqual(self._tensorflow(cuda=False).extras, set())

    def test_unsupported_platforms_use_plain_tensorflow(self) -> None:
        """Native Windows, macOS, and Linux ARM do not request unsupported CUDA wheels."""
        for system, machine in (("Windows", "AMD64"), ("Darwin", "arm64"),
                                ("Darwin", "x86_64"), ("Linux", "aarch64")):
            with self.subTest(system=system, machine=machine):
                self.assertEqual(self._tensorflow(system=system, machine=machine).extras, set())

    def test_colab_preserves_unrelated_extras(self) -> None:
        """Only TensorFlow's CUDA extra changes when applying hosted setup policy."""
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "requirements.txt"
            manifest.write_text("tensorflow[and-cuda,gcs-filesystem]==2.20.0\nexample[speedup]>=1\n",
                                encoding="utf-8")
            requirements = bootstrap._requirements(manifest, cuda=False)
        self.assertEqual(requirements[0].extras, {"gcs-filesystem"})
        self.assertEqual(requirements[1].extras, {"speedup"})


# Permit a focused direct run without importing model or training test modules.
if __name__ == "__main__":
    unittest.main()
