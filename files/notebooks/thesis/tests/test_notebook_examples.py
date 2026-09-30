"""Check notebook helper contracts and public wrapper calls without research training."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from common.dataloader import get_dataset

from files.notebooks.hpo.generate_notebooks import make_notebook


NOTEBOOK_ROOT = Path(__file__).resolve().parents[2]


class NotebookExampleTests(unittest.TestCase):
    """Exercise image/label transformations and shared setup in maintained examples."""

    def test_image_helpers_preserve_class_zero_dtype_and_partial_batches(self) -> None:
        """Execute shared notebook pipelines without downloading source datasets."""

        root = NOTEBOOK_ROOT.parents[1]
        inventory = subprocess.check_output(
            ["git", "-c", f"safe.directory={root.as_posix()}", "ls-files", 
             "--cached", "--others", "--exclude-standard", "--", "files/notebooks/*.ipynb"], 
            cwd=root, text=True)
        paths = sorted({root / name for name in inventory.splitlines()
                        if Path(name).parent == Path("files/notebooks") and (root / name).is_file()})
        self.assertEqual(len(paths), 17)
        for path in paths:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            cells = ["".join(cell["source"]) for cell in notebook["cells"]
                     if cell["cell_type"] == "code"]
            import_source = next(source for source in cells
                                 if "from common.dataloader import get_dataset" in source)
            source = next(source for source in cells if "trainset = get_dataset(" in source)
            namespace = {}
            exec(compile(import_source, str(path), "exec"), namespace)
            grayscale = "load_mnist(" in source
            padded = "layers.ZeroPadding2D" in source
            shape = (129, 4, 4) if grayscale else (129, 4, 4, 3)
            images = np.zeros(shape, dtype=np.uint8)
            images[1] = 255
            images[2] = 127
            labels = (np.arange(129) % 3).astype(np.uint8)
            source_labels = labels if grayscale else labels[:, None]
            dataset_name = "mnist" if grayscale else "cifar10"
            expected = np.pad(images[..., None], ((0, 0), (2, 2), (2, 2), (0, 0))) \
                if padded else images
            with self.subTest(notebook=path.name):
                self.assertIs(namespace["get_dataset"], get_dataset)
                with patch(f"tensorflow.keras.datasets.{dataset_name}.load_data", 
                           return_value=((images, source_labels), (images, source_labels))) as loader:
                    exec(compile(source, str(path), "exec"), namespace)
                loader.assert_called_once_with()
                np.testing.assert_array_equal(namespace["x_train"], expected)
                np.testing.assert_array_equal(namespace["x_test"], expected)
                self.assertEqual(namespace["x_test"].dtype, images.dtype)
                batches = list(namespace["valset"].as_numpy_iterator())
                self.assertEqual([len(x) for x, _ in batches], [128, 1])
                result = np.concatenate([x for x, _ in batches])
                targets = np.concatenate([y for _, y in batches])
                expected_channels = expected[..., None] if expected.ndim == 3 else expected
                np.testing.assert_array_equal(result, expected_channels)
                self.assertEqual(result.dtype, images.dtype)
                np.testing.assert_array_equal(targets, labels)
                self.assertEqual(targets.dtype, labels.dtype)
                self.assertEqual(sum(len(x) for x, _ in namespace["trainset"].as_numpy_iterator()), 128)
                dataset = namespace["get_dataset"](
                    images[:3], labels[:3], batch_size=2, shuffle_buffer=0, drop_remainder=False)
                batches = list(dataset.as_numpy_iterator())
                self.assertEqual([len(x) for x, _ in batches], [2, 1])
                result = np.concatenate([x for x, _ in batches])
                targets = np.concatenate([y for _, y in batches])
                self.assertEqual(result.dtype, images.dtype)
                self.assertEqual(result.ndim, 4)
                np.testing.assert_array_equal(targets, labels[:3])
                self.assertEqual(targets.dtype, labels.dtype)
                np.testing.assert_array_equal(result[0], 0)
                np.testing.assert_array_equal(result[1], 255)
                np.testing.assert_array_equal(images[1], 255)
                dropped = namespace["get_dataset"](
                    images[:3], labels[:3], batch_size=2, shuffle_buffer=0, drop_remainder=True)
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
