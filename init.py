"""Set notebook import paths from the repository root without loading a backend.

Importing this module changes the working directory to the checkout root and
adds that root and ``notebooks/thesis`` to ``sys.path``. Dependency preparation
and TensorFlow runtime configuration belong to the notebook's explicit setup
and training stages, so a fresh hosted kernel can import this wrapper safely.
"""

from pathlib import Path

from runpy import run_path

from common import utils


REPOSITORY_ROOT = run_path(
    str(Path(__file__).resolve().parent / "notebooks" / "init.py"), 
    run_name="init"
)["REPOSITORY_ROOT"]

utils.init()
