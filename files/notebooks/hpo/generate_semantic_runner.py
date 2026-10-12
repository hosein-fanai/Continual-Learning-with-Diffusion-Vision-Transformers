"""Republish the maintained semantic notebook without reverting campaign edits.

The notebook is the source of truth. Run this utility only in an authorized
online container; it validates cell structure and clears execution artifacts.
It neither executes cells nor starts training.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Semantic_Consolidation_HPO_Runner.ipynb"


def make_notebook() -> dict:
    """Read the maintained notebook and return an unexecuted validated copy."""

    notebook = deepcopy(json.loads(NOTEBOOK.read_text(encoding="utf-8")))
    identifiers = [cell["id"] for cell in notebook["cells"]]
    # Repeated IDs make stable cell references ambiguous.
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Notebook cell identities must be unique.")
    for cell in notebook["cells"]:
        # Markdown remains intact; code is checked and returned unexecuted.
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{NOTEBOOK.name}:{cell['id']}", "exec")
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook["metadata"]["dit_hpo_runner"]["unexecuted"] = True
    return notebook


def main() -> None:
    """Write the clean maintained notebook using stable JSON formatting."""

    NOTEBOOK.write_text(json.dumps(make_notebook(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# Direct execution republishes the maintained notebook without starting a kernel.
if __name__ == "__main__":
    main()
