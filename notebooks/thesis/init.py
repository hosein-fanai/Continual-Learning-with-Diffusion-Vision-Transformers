"""Load shared notebook path setup when starting in this directory."""

from pathlib import Path

from runpy import run_path

REPOSITORY_ROOT = run_path(
    str(Path(__file__).resolve().parents[1] / "init.py"), run_name="init"
)["REPOSITORY_ROOT"]
