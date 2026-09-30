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

    Args:
        runtime (str): ``auto`` consults CONTINUAL_RUNTIME and then provider
            markers. An explicit local/colab/kaggle/binder/studiolab/hosted
            value overrides both the environment override and detection.
            Defaults to ``'auto'``.

    Returns:
        str: Selected runtime name; undetected ``auto`` resolves to ``local``.

    Raises:
        ValueError: The explicit or environment-provided name is unsupported.
    """

    requested = os.environ.get("CONTINUAL_RUNTIME", "auto") if runtime == "auto" else runtime
    choices = {"auto", "local", "colab", "kaggle", "binder", "studiolab", "hosted"}
    # Reject policy typos before they can authorize package installation.
    if requested not in choices:
        raise ValueError(f"Unknown runtime {requested!r}. Choose one of {sorted(choices)}.")
    # Explicit runtime policy overrides all provider detection.
    if requested != "auto":
        return requested
    # Kaggle images can inherit Colab's environment, so check Kaggle first.
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    # Binder markers distinguish its CPU environment from inherited Colab images.
    if os.environ.get("BINDER_REPO_URL") or os.environ.get("BINDER_LAUNCH_HOST"):
        return "binder"
    # Only a running Colab session, not an installed SDK, enables hosted policy.
    if "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    return "local"


def _requirements(path: Path, cuda: bool = True) -> list:
    """Parse active PEP 508 requirements without importing scientific packages.

    Args:
        path (Path): UTF-8 manifest with one requirement per non-comment line.
            Environment markers are evaluated for this Python/platform.
        cuda (bool): False removes only TensorFlow's ``and-cuda`` extra; True
            retains all manifest extras. The manifest on disk is unchanged.
            Defaults to ``True``.

    Returns:
        list[packaging.requirements.Requirement]: Active requirements in source
            order, retaining version specifiers and unrelated extras.

    Raises:
        OSError: The manifest cannot be read.
        packaging.requirements.InvalidRequirement: A nonempty line is malformed.
    """

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name


    requirements = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        # Empty and comment-only manifest lines have no requirement.
        if line:
            requirement = Requirement(line)
            # Ignore requirements whose platform/interpreter markers do not apply.
            if requirement.marker is None or requirement.marker.evaluate():
                # Remove only the optional pip CUDA libraries when the host supplies them.
                if not cuda and canonicalize_name(requirement.name) == "tensorflow":
                    requirement.extras = {
                        extra for extra in requirement.extras
                        if canonicalize_name(extra) != "and-cuda"
                    }
                requirements.append(requirement)
    return requirements


def _unsatisfied(requirements: list, include_extras: bool = False) -> list[str]:
    """Check distributions, and selected extra dependencies before installation.

    Verify-only startup accepts system CUDA supplied by a GPU Docker image.
    Installation checks also inspect extras: an installed TensorFlow package
    does not imply that its optional CUDA pip distributions are installed.

    Args:
        requirements (list[packaging.requirements.Requirement]): Selected
            requirements whose environment markers have already been evaluated.
        include_extras (bool): True recursively checks dependencies activated by
            selected extras; False checks only each declared distribution.
            Defaults to ``False``.

    Returns:
        list[str]: Missing or version-incompatible requirements in traversal
            order. Compatible prerelease versions are accepted by the specifier.
            Repeated requirement strings are checked once; no packages change.

    Raises:
        packaging.requirements.InvalidRequirement: If an installed extra declares malformed requirement metadata.
        packaging.version.InvalidVersion: If an installed distribution reports an invalid version. Missing distributions are returned in the result, not raised.
    """

    from packaging.requirements import Requirement


    missing = []
    pending = list(requirements)
    checked = set()
    while pending:
        requirement = pending.pop(0)
        # Avoid repeatedly traversing an extra dependency or cyclic metadata.
        if str(requirement) in checked:
            continue
        checked.add(str(requirement))
        try:
            version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            version = None
        # Missing/incompatible parent distributions must first be installed by pip.
        if version is None or not requirement.specifier.contains(version, prereleases=True):
            missing.append(str(requirement))
            continue
        # Installation must also satisfy dependencies activated by requested extras.
        if include_extras and requirement.extras:
            for value in metadata.requires(requirement.name) or []:
                dependency = Requirement(value)
                # Select dependencies activated by this extra, excluding the
                # package's ordinary dependencies and any unrelated extras.
                marker = dependency.marker
                # Traverse dependencies activated by at least one selected package extra.
                if (marker is not None
                        and not marker.evaluate({"extra": ""})
                        and any(marker.evaluate({"extra": extra})
                                for extra in requirement.extras)):
                    dependency.marker = None
                    pending.append(dependency)
    return missing


def _loaded_distributions() -> set[str]:
    """Identify distributions whose top-level import modules are already loaded.

    Returns:
        set[str]: Canonical distribution names mapped from ``sys.modules`` by
            installed package metadata. This includes notebook-frontend imports
            and lets the installer refuse binary replacements in a live process.
            No additional package modules are imported by this lookup.

    Raises:
        OSError: If installed distribution metadata cannot be read.
    """

    from packaging.utils import canonicalize_name


    imported = {name.split(".", 1)[0] for name in sys.modules}
    return {
        canonicalize_name(distribution)
        for module, distributions in metadata.packages_distributions().items()
        if module in imported
        for distribution in distributions
    }


def prepare_runtime(root: Path, install: bool | None = None, 
                    runtime: str = "auto", cuda: bool | None = None) -> dict[str, str]:
    """Check or install the hosted notebook's runtime before project imports.

    Args:
        root (Path): Existing project checkout containing requirements.txt.
        install (bool | None): None installs in detected or explicitly selected hosted runtimes.
            False verifies only; True permits installing in this interpreter.
            Defaults to ``None``.
        runtime (str): auto detects Kaggle, Binder, or Colab, otherwise uses local.
            Explicit hosted or studiolab enables setup on other services.
            CONTINUAL_RUNTIME supplies this choice when runtime is auto.
            Defaults to ``'auto'``.
        cuda (bool | None): None omits the CUDA extra on Colab, Kaggle, and Binder; elsewhere
            it retains the manifest's platform-dependent extra. False omits it
            for CPU or managed CUDA environments; True retains it. Verify-only
            startup also accepts system CUDA supplied by a GPU Docker image.
            Defaults to ``None``.

    Returns:
        dict[str, str]: Installed dependency versions, plus Python. TensorFlow and
        Keras are inspected through package metadata and are not imported here.
        Sets KERAS_BACKEND=tensorflow and supplies TF_FORCE_GPU_ALLOW_GROWTH=true
        only when unset. May run pip and prints the selected runtime and versions.

    Raises:
        RuntimeError: Requirements are unavailable, incompatible modules are
            already loaded, or installation did not satisfy the manifest.
        subprocess.CalledProcessError: pip cannot resolve or install requirements.
        TypeError: cuda or install is neither a Boolean nor None.
        ValueError: The explicit/environment runtime selector is unsupported.
        OSError: Requirements or the temporary pip report cannot be read/written.
        packaging.requirements.InvalidRequirement: A manifest or installed extra
            contains malformed dependency metadata.
    """

    from packaging.utils import canonicalize_name


    root = Path(root).resolve()
    manifest = root / "requirements.txt"
    # A complete manifest is required before making environment changes.
    if not manifest.is_file():
        raise RuntimeError("This checkout is missing requirements.txt. Use the updated repository.")
    # TensorFlow 2.20 wheels support the declared Python interpreter range.
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("Use a Python 3.11–3.13 runtime for this TensorFlow 2.20 notebook.")
    runtime_name = detect_runtime(runtime)
    # A truthy string such as "False" must not enable CUDA wheel installation.
    if cuda is not None and not isinstance(cuda, bool):
        raise TypeError("cuda must be True, False, or None.")
    # Managed CUDA providers and Binder omit separately installed CUDA wheels.
    if cuda is None:
        cuda = runtime_name not in {"colab", "kaggle", "binder"}
    # Local kernels verify only unless installation was explicitly enabled.
    if install is None:
        install = runtime_name != "local"
    # A mistaken string cannot authorize modifications to the active interpreter.
    elif not isinstance(install, bool):
        raise TypeError("install must be True, False, or None.")

    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    requirements = _requirements(manifest, cuda=cuda)

    # Detect a previous in-process pip replacement rather than trusting new disk
    # metadata while Python still holds an older TensorFlow/Keras module.
    for module_name in ("tensorflow", "keras"):
        module = sys.modules.get(module_name)
        # Compare loaded module versions as well as installed disk metadata.
        if module is not None:
            expected = next(item for item in requirements if item.name == module_name)
            loaded_version = getattr(module, "__version__", "0")
            # Disk upgrades cannot repair incompatible modules already in memory.
            if not expected.specifier.contains(loaded_version, prereleases=True):
                raise RuntimeError(
                    f"{module_name} {loaded_version} is already loaded. "
                    "Restart the session, then Run all from the first cell."
                )

    missing = _unsatisfied(requirements, include_extras=bool(install))
    # Verification-only startup reports incompatibility without calling pip.
    if missing and not install:
        raise RuntimeError(
            "Missing or incompatible notebook dependencies: " + ", ".join(missing)
            + ". Select the project's TensorFlow 2.20 kernel, or install requirements.txt "
              "in your intended environment before starting this notebook."
        )
    # Resolve and install only when the selected requirements are unsatisfied.
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
            # Replacing an imported binary dependency requires a fresh kernel.
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
        # A successful pip process still must satisfy the selected requirement set.
        if remaining:
            raise RuntimeError("Unresolved notebook dependencies: " + ", ".join(remaining))

    versions = {item.name: metadata.version(item.name) for item in requirements}
    versions["Python"] = sys.version.split()[0]
    print(f"Repository ready: {root} (runtime: {runtime_name})", flush=True)
    print(f"Runtime packages ready: Python {versions['Python']}, "
          f"TensorFlow {versions['tensorflow']}, Keras {versions['keras']}", flush=True)
    return versions
