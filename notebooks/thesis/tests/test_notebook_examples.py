"""Check notebook helper contracts and public wrapper calls without research training."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import subprocess
import unittest

import numpy as np

from notebooks.hpo.generate_notebooks import make_notebook


NOTEBOOK_ROOT = Path(__file__).resolve().parents[2]


class NotebookExampleTests(unittest.TestCase):
    """Exercise image/label transformations and shared setup in maintained examples."""

    def test_image_helpers_preserve_class_zero_dtype_and_partial_batches(self) -> None:
        """Examples scale bytes and preserve sparse labels without a class offset."""

        root = NOTEBOOK_ROOT.parent
        inventory = subprocess.check_output(
            ["git", "-c", f"safe.directory={root.as_posix()}", "ls-files", 
             "--cached", "--others", "--exclude-standard", "--", "notebooks/*.ipynb"], 
            cwd=root, text=True)
        paths = sorted({root / name for name in inventory.splitlines()
                        if Path(name).parent == Path("notebooks")})
        self.assertEqual(len(paths), 16)
        for path in paths:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            cells = ["".join(cell["source"]) for cell in notebook["cells"]
                     if cell["cell_type"] == "code"]
            source = next(source for source in cells if "def get_dataset(" in source)
            namespace = {}
            exec(compile(source, str(path), "exec"), namespace)
            grayscale = "    x = x[..., None]" in source
            shape = (3, 4, 4) if grayscale else (3, 4, 4, 3)
            images = np.zeros(shape, dtype=np.uint8)
            images[1] = 255
            labels = np.asarray([0, 1, 2], dtype=np.uint8)
            with self.subTest(notebook=path.name):
                dataset = namespace["get_dataset"](
                    images, labels, batch_size=2, shuffle_buffer=None, drop_remainder=False)
                batches = list(dataset.as_numpy_iterator())
                self.assertEqual([len(x) for x, _ in batches], [2, 1])
                result = np.concatenate([x for x, _ in batches])
                targets = np.concatenate([y for _, y in batches])
                self.assertEqual(result.dtype, np.dtype("float32"))
                self.assertEqual(result.ndim, 4)
                np.testing.assert_array_equal(targets, labels)
                self.assertEqual(targets.dtype, labels.dtype)
                np.testing.assert_array_equal(result[0], -1.)
                np.testing.assert_array_equal(result[1], 1.)
                np.testing.assert_array_equal(images[1], 255)
                dropped = namespace["get_dataset"](
                    images, labels, batch_size=2, shuffle_buffer=None, drop_remainder=True)
                self.assertEqual(sum(len(x) for x, _ in dropped.as_numpy_iterator()), 2)

    def test_v2_example_calls_match_current_public_api(self) -> None:
        """V2 fits pass x by keyword and evaluation explicitly names its data."""

        from diffusion.models.wrapper.diffusion_classifier_v2 import DiffusionClassifierV2


        for path in sorted(NOTEBOOK_ROOT.glob("*CLFV2.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                # Markdown does not contain executable wrapper calls.
                if cell["cell_type"] != "code":
                    continue
                for call in ast.walk(ast.parse("".join(cell["source"]))):
                    # Select the documented model fitting/evaluation methods only.
                    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute) \
                            or call.func.attr not in ("fit_generator", "fit_discriminator", "evaluate"):
                        continue
                    with self.subTest(notebook=path.name, method=call.func.attr):
                        keywords = {keyword.arg: object() for keyword in call.keywords}
                        inspect.signature(getattr(DiffusionClassifierV2, call.func.attr)).bind(
                            object(), *(object() for _ in call.args), **keywords)
                        self.assertIn("x", keywords)

    def test_generated_hpo_notebooks_include_runtime_preparation(self) -> None:
        """Generated studies prepare imports and select the maintained TensorFlow kernel."""

        notebook = make_notebook("classification", "cnn", 1, "Synthetic API check.")
        cells = ["".join(cell["source"]) for cell in notebook["cells"]
                 if cell["cell_type"] == "code"]
        self.assertEqual(cells[0], (NOTEBOOK_ROOT / "setup_cell.py").read_text(encoding="utf-8"))
        self.assertIn("from common.hpo import", cells[1])
        self.assertEqual(notebook["metadata"]["kernelspec"]["name"], "tensorflow-220")
        for source in cells:
            ast.parse(source)


# Run helper/API verification without launching thesis research streams.
if __name__ == "__main__":
    unittest.main()
