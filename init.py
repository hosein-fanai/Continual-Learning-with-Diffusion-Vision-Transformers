"""Forward repository-root notebook imports to shared path setup."""

from pathlib import Path

from runpy import run_path

from common import utils


REPOSITORY_ROOT = run_path(
    str(Path(__file__).resolve().parent / "notebooks" / "init.py"), 
    run_name="init"
)["REPOSITORY_ROOT"]

utils.init()
