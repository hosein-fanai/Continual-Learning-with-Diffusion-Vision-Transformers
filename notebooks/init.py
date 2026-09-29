"""Shared checkout and runtime setup for standalone repository notebooks.

This file can be fetched before the repository exists, so its top-level imports
use only the standard library. Scientific imports happen after dependency setup.
"""

from __future__ import annotations

import os

from pathlib import Path

import subprocess

import sys


CHECKOUT_NAME = "Continual-Learning-with-Diffusion-Vision-Transformers"
REPOSITORY = f"https://github.com/hosein-fanai/{CHECKOUT_NAME}.git"


def _set_paths(root: Path) -> Path:
    """Make one checkout the working directory and add its notebook import paths.

    Args:
        root (Path): Existing repository directory; relative input is resolved
            against the current working directory before changing it.

    Returns:
        Path: Absolute checkout root. The root and its ``notebooks/thesis``
            directory are prepended to ``sys.path`` only when absent. No
            scientific libraries are imported or device settings changed.

    Raises:
        OSError: The requested directory cannot become the working directory.
    """

    root = root.resolve()
    for path in (root, root / "notebooks" / "thesis"):
        # Repeated setup must not accumulate duplicate import locations.
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    os.chdir(root)

    return root


def prepare_notebook(
    checkout_name: str = CHECKOUT_NAME, 
    repository: str | None = None, 
    revision: str = "main", 
    runtime: str = "auto", 
    cuda: bool | None = None, 
    install: bool | None = None
) -> tuple[Path, dict[str, str]]:
    """Find or clone the project, then prepare dependencies before model imports.

    Existing checkouts and revisions are reused without pulling over local work.
    Local kernels are verified; hosted installation follows ``prepare_runtime``.
    This function does not import TensorFlow.

    Args:
        checkout_name (str): Single destination directory name used when cloning.
            Existing parent checkouts take precedence over named child checkouts.
        repository (str | None): Git clone URL. None builds the default GitHub
            URL from checkout_name; an existing checkout is never fetched.
        revision (str): Branch or tag passed to ``git clone --branch`` only when
            a checkout is missing. Existing revisions remain untouched.
        runtime (str): Runtime policy forwarded to ``prepare_runtime``. ``auto``
            detects supported hosted providers; ``local`` only verifies by default.
        cuda (bool | None): Whether to retain TensorFlow's CUDA pip extra.
            None uses the selected provider's default.
        install (bool | None): True permits dependency installation, False only
            verifies; None permits it only for a detected/selected hosted runtime.

    Returns:
        tuple[Path, dict[str, str]]: Absolute selected checkout root and installed
            Python/dependency versions. Changes cwd and import paths, may clone
            the repository, and may install packages under the selected policy.

    Raises:
        ValueError: checkout_name would escape its destination directory.
        RuntimeError: Cloning fails, a destination is incomplete, or dependency
            preparation requires a different/restarted interpreter.
        OSError: Checkout paths cannot be accessed.
    """

    # A custom directory name selects its matching default GitHub repository.
    if repository is None:
        repository = f"https://github.com/hosein-fanai/{checkout_name}.git"

    # Keep cloning inside the intended writable base directory.
    if not checkout_name or Path(checkout_name).name != checkout_name \
    or checkout_name in {".", ".."}:
        raise ValueError("checkout_name must be a single directory name.")

    markers = ("common/config.py", "semantic_consolidation/config.py", "diffusion/__init__.py")
    locations = (Path.cwd(), *Path.cwd().parents, Path.cwd() / checkout_name, 
                 Path("/kaggle/working") / checkout_name, Path("/content") / checkout_name)
    root = next((path for path in locations
                 if all((path / marker).is_file() for marker in markers)), None)
    # Reuse any recognized checkout before allocating a hosted/local clone.
    if root is None:
        base = next((path for path in (Path("/kaggle/working"), Path("/content"))
                     if path.is_dir()), Path.cwd())
        root = base / checkout_name
        # Clone only into an absent destination, preserving existing user files.
        if not root.exists():
            try:
                subprocess.run([
                    "git", 
                    "clone", 
                    "--depth", 
                    "1", 
                    "--branch", 
                    revision, 
                    repository, 
                    str(root)
                ], check=True)
            except (OSError, subprocess.CalledProcessError) as error:
                raise RuntimeError(
                    f"Could not download the repository into {root}. Enable Internet access "
                    "in the notebook settings (including Kaggle), ensure git is available, "
                    "and use a writable working directory. Then rerun setup."
                ) from error

        # A failed or unrelated checkout must never reach project initialization.
        if not all((root / marker).is_file() for marker in markers):
            raise RuntimeError(
                f"{root} exists but is not a complete project checkout. "
                "Choose another CHECKOUT_NAME."
            )

    root = _set_paths(root)

    from notebooks.thesis.bootstrap import prepare_runtime


    versions = prepare_runtime(
        root, 
        runtime=runtime, 
        cuda=cuda, 
        install=install
    )

    return root, versions


# Notebooks that still use ``import init`` need only the project paths.
if __name__ == "init":
    REPOSITORY_ROOT = _set_paths(
        Path(__file__).resolve().parents[1]
    )
