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
    """Use a checkout as the working directory and make project helpers importable."""

    root = root.resolve()
    for path in (root, root / "notebooks" / "thesis"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    os.chdir(root)

    return root


def prepare_notebook(
    *, 
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
    """

    if repository is None:
        repository = f"https://github.com/hosein-fanai/{checkout_name}.git"

    if not checkout_name or Path(checkout_name).name != checkout_name \
    or checkout_name in {".", ".."}:
        raise ValueError("checkout_name must be a single directory name.")

    markers = ("common/config.py", "semantic_consolidation/config.py", "diffusion/__init__.py")
    locations = (Path.cwd(), *Path.cwd().parents, Path.cwd() / checkout_name,
                 Path("/kaggle/working") / checkout_name, Path("/content") / checkout_name)
    root = next((path for path in locations
                 if all((path / marker).is_file() for marker in markers)), None)
    if root is None:
        base = next((path for path in (Path("/kaggle/working"), Path("/content"))
                     if path.is_dir()), Path.cwd())
        root = base / checkout_name
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
