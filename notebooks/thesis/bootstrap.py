"""Prepare a hosted notebook before importing TensorFlow or project modules.

This module uses distribution metadata to inspect installed packages. Local
checkouts are verified without installing into the user's existing environment;
Recognized hosted runtimes install missing/incompatible requirements in the
active interpreter. Colab and Kaggle use their managed CUDA libraries; Binder
omits CUDA pip dependencies for CPU use.
"""

from __future__ import annotations

import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def detect_runtime(runtime: str = "auto") -> str:
    """Detect known services without importing their SDKs or scientific packages.

    Explicit selection supports other hosted services and existing Studio Lab
    accounts. A generic JupyterHub or an installed cloud SDK is not evidence
    that this interpreter belongs to a managed online service.
    """
    requested = os.environ.get("CONTINUAL_RUNTIME", "auto") if runtime == "auto" else runtime
    choices = {"auto", "local", "colab", "kaggle", "binder", "studiolab", "hosted"}
    if requested not in choices:
        raise ValueError(f"Unknown runtime {requested!r}. Choose one of {sorted(choices)}.")
    if requested != "auto":
        return requested
    # Kaggle images can inherit Colab's environment, so check Kaggle first.
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if os.environ.get("BINDER_REPO_URL") or os.environ.get("BINDER_LAUNCH_HOST"):
        return "binder"
    if "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    return "local"


def _requirements(path: Path, *, cuda: bool = True) -> list:
    """Read the shared requirements without importing scientific libraries."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    requirements = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            requirement = Requirement(line)
            if requirement.marker is None or requirement.marker.evaluate():
                if not cuda and canonicalize_name(requirement.name) == "tensorflow":
                    requirement.extras = {
                        extra for extra in requirement.extras
                        if canonicalize_name(extra) != "and-cuda"
                    }
                requirements.append(requirement)
    return requirements


def _unsatisfied(requirements: list, *, include_extras: bool = False) -> list[str]:
    """Check distributions, and selected extra dependencies before installation.

    Verify-only startup accepts system CUDA supplied by a GPU Docker image.
    Installation checks also inspect extras: an installed TensorFlow package
    does not imply that its optional CUDA pip distributions are installed.
    """
    from packaging.requirements import Requirement

    missing = []
    pending = list(requirements)
    checked = set()
    while pending:
        requirement = pending.pop(0)
        if str(requirement) in checked:
            continue
        checked.add(str(requirement))
        try:
            version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            version = None
        if version is None or not requirement.specifier.contains(version, prereleases=True):
            missing.append(str(requirement))
            continue
        if include_extras and requirement.extras:
            for value in metadata.requires(requirement.name) or []:
                dependency = Requirement(value)
                # Select dependencies activated by this extra, excluding the
                # package's ordinary dependencies and any unrelated extras.
                marker = dependency.marker
                if (marker is not None
                        and not marker.evaluate({"extra": ""})
                        and any(marker.evaluate({"extra": extra})
                                for extra in requirement.extras)):
                    dependency.marker = None
                    pending.append(dependency)
    return missing


def _loaded_distributions() -> set[str]:
    """Identify installed distributions represented in the current Python process."""
    from packaging.utils import canonicalize_name

    imported = {name.split(".", 1)[0] for name in sys.modules}
    return {
        canonicalize_name(distribution)
        for module, distributions in metadata.packages_distributions().items()
        if module in imported
        for distribution in distributions
    }


def prepare_runtime(root: Path, *, install: bool | None = None,
                    runtime: str = "auto", cuda: bool | None = None) -> dict[str, str]:
    """Check or install the hosted notebook's runtime before project imports.

    Args:
        root: Existing project checkout containing requirements.txt.
        install: None installs in detected or explicitly selected hosted runtimes.
            False verifies only; True permits installing in this interpreter.
        runtime: auto detects Kaggle, Binder, or Colab, otherwise uses local.
            Explicit hosted or studiolab enables setup on other services.
            CONTINUAL_RUNTIME supplies this choice when runtime is auto.
        cuda: None omits the CUDA extra on Colab, Kaggle, and Binder; elsewhere
            it retains the manifest's platform-dependent extra. False omits it
            for CPU or managed CUDA environments; True retains it. Verify-only
            startup also accepts system CUDA supplied by a GPU Docker image.

    Returns:
        Installed dependency versions, plus the Python version. TensorFlow and
        Keras are inspected through package metadata and are not imported here.

    Raises:
        RuntimeError: Requirements are unavailable, incompatible modules are
            already loaded, or installation did not satisfy the manifest.
        subprocess.CalledProcessError: pip cannot resolve or install requirements.
    """
    from packaging.utils import canonicalize_name

    root = Path(root).resolve()
    manifest = root / "requirements.txt"
    if not manifest.is_file():
        raise RuntimeError("This checkout is missing requirements.txt. Use the updated repository.")
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("Use a Python 3.11–3.13 runtime for this TensorFlow 2.20 notebook.")
    runtime_name = detect_runtime(runtime)
    if cuda is not None and not isinstance(cuda, bool):
        raise TypeError("cuda must be True, False, or None.")
    if cuda is None:
        cuda = runtime_name not in {"colab", "kaggle", "binder"}
    if install is None:
        install = runtime_name != "local"

    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    requirements = _requirements(manifest, cuda=cuda)

    # Detect a previous in-process pip replacement rather than trusting new disk
    # metadata while Python still holds an older TensorFlow/Keras module.
    for module_name in ("tensorflow", "keras"):
        module = sys.modules.get(module_name)
        if module is not None:
            expected = next(item for item in requirements if item.name == module_name)
            loaded_version = getattr(module, "__version__", "0")
            if not expected.specifier.contains(loaded_version, prereleases=True):
                raise RuntimeError(
                    f"{module_name} {loaded_version} is already loaded. "
                    "Restart the session, then Run all from the first cell."
                )

    missing = _unsatisfied(requirements, include_extras=bool(install))
    if missing and not install:
        raise RuntimeError(
            "Missing or incompatible notebook dependencies: " + ", ".join(missing)
            + ". Select the project's TensorFlow 2.20 kernel, or install requirements.txt "
              "in your intended environment before starting this notebook."
        )
    if missing:
        print("Preparing notebook dependencies: " + ", ".join(missing), flush=True)
        command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                   "--quiet"]
        # Passing the manifest itself would restore an omitted CUDA extra.
        # Use requirements parsed from that same file with only the extra removed.
        command.extend([str(item) for item in requirements] if not cuda
                       else ["-r", str(manifest)])
        # Resolve first so pip cannot replace a binary package already imported
        # by the notebook frontend and leave this running process inconsistent.
        with tempfile.TemporaryDirectory(prefix="continual-runtime-") as directory:
            report = Path(directory) / "install.json"
            subprocess.run([*command, "--dry-run", "--report", str(report)], check=True)
            planned = json.loads(report.read_text(encoding="utf-8"))["install"]
            loaded = _loaded_distributions()
            replacements = [item["metadata"]["name"] for item in planned
                            if canonicalize_name(item["metadata"]["name"]) in loaded]
            if replacements:
                raise RuntimeError(
                    "Installation would replace already loaded packages: " + ", ".join(replacements)
                    + ". Use a fresh Python 3.11-3.13 session with compatible kernel packages "
                      "and run setup before scientific imports. On services with selectable "
                      "runtime images, choose one with TensorFlow 2.20. No packages were changed."
                )
            subprocess.run(command, check=True)
        importlib.invalidate_caches()
        remaining = _unsatisfied(requirements, include_extras=bool(install))
        if remaining:
            raise RuntimeError("Unresolved notebook dependencies: " + ", ".join(remaining))

    versions = {item.name: metadata.version(item.name) for item in requirements}
    versions["Python"] = sys.version.split()[0]
    print(f"Repository ready: {root} (runtime: {runtime_name})", flush=True)
    print(f"Runtime packages ready: Python {versions['Python']}, "
          f"TensorFlow {versions['tensorflow']}, Keras {versions['keras']}", flush=True)
    return versions
