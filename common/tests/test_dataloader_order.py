"""Source-order loading without changing default class grouping or split semantics."""

from contextlib import nullcontext
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import train_test_split

from common.dataloader import load_mnist, load_fmnist, load_cifar10, load_cifar100


_LOADERS = (
    ("mnist", load_mnist), ("fashion_mnist", load_fmnist), 
    ("cifar10", load_cifar10), ("cifar100", load_cifar100)
)


class DatasetSourceOrderTests(unittest.TestCase):
    """Use deliberately interleaved synthetic rows to detect accidental regrouping."""

    @staticmethod
    def arrays(dataset_name: str) -> tuple:
        """Return distinct train/test byte images with each dataset's native label rank."""

        color = dataset_name.startswith("cifar")
        shape = (2, 2, 3) if color else (2, 2)
        train_labels = np.asarray([2, 0, 1, 2, 1, 0, 1, 2, 0, 1, 0, 2], dtype="uint8")
        test_labels = np.asarray([1, 2, 0], dtype="uint8")
        train_images = np.arange(12 * np.prod(shape), dtype="uint8").reshape((12, *shape))
        test_images = (np.arange(3 * np.prod(shape), dtype="uint8") + 180).reshape((3, *shape))
        labels_shape = (-1, 1) if color else tuple([-1])
        return (train_images, train_labels.reshape(labels_shape)), (test_images, test_labels.reshape(labels_shape))

    def assert_arrays_equal(self, actual: np.ndarray, expected: np.ndarray) -> None:
        """Check exact bytes, label alignment, dtype and rank together."""

        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_array_equal(actual, expected)

    def test_none_indices_preserve_original_official_splits_and_storage(self) -> None:
        """All four loaders can reproduce their raw Keras arrays without row reordering."""

        for dataset_name, loader in _LOADERS:
            with self.subTest(dataset=dataset_name):
                source = self.arrays(dataset_name)
                snapshots = [value.copy() for split in source for value in split]
                with patch("tensorflow.keras.datasets." + dataset_name + ".load_data", return_value=source), \
                     patch("common.dataloader.dataset_load_lock", return_value=nullcontext()), \
                     patch("sklearn.model_selection.train_test_split") as splitter:
                    prepared = loader(indices=None, validation_ratio=0., preprocess=None, 
                                      onehot_labels=False, verbose=0)
                splitter.assert_not_called()
                train_x, train_y, val_x, val_y, test_x, test_y = prepared
                self.assertIsNone(val_x)
                self.assertIsNone(val_y)
                for actual, expected in zip((train_x, train_y, test_x, test_y), snapshots):
                    self.assert_arrays_equal(actual, expected)
                for current, before in zip((value for split in source for value in split), snapshots):
                    self.assert_arrays_equal(current, before)

    def test_default_and_explicit_class_grouping_remain_unchanged(self) -> None:
        """Omitted indices still group all classes; explicit subsets keep requested order."""

        for dataset_name, loader in _LOADERS:
            source = self.arrays(dataset_name)
            for options, classes in (({}, [0, 1, 2]), ({"indices": [2, 0]}, [2, 0])):
                with self.subTest(dataset=dataset_name, classes=classes), \
                     patch("tensorflow.keras.datasets." + dataset_name + ".load_data", return_value=source), \
                     patch("common.dataloader.dataset_load_lock", return_value=nullcontext()):
                    prepared = loader(validation_ratio=0., preprocess=None, verbose=0, **options)
                    for pair, offset in zip(source, (0, 4)):
                        images, labels = pair
                        order = np.concatenate([np.flatnonzero(labels.reshape(-1) == label) for label in classes])
                        self.assert_arrays_equal(prepared[offset], images[order])
                        self.assert_arrays_equal(prepared[offset + 1], labels[order])

    def test_none_indices_split_original_order_with_existing_stratification(self) -> None:
        """Requested validation uses the same seeded splitter on the ungrouped source rows."""

        for dataset_name, loader in _LOADERS:
            with self.subTest(dataset=dataset_name):
                source = self.arrays(dataset_name)
                (images, labels), (test_images, test_labels) = source
                train_x, val_x, train_y, val_y = train_test_split(
                    images, labels, test_size=1. / 3., stratify=labels, random_state=23
                )
                with patch("tensorflow.keras.datasets." + dataset_name + ".load_data", return_value=source), \
                     patch("common.dataloader.dataset_load_lock", return_value=nullcontext()):
                    prepared = loader(indices=None, validation_ratio=1. / 3., preprocess=None, 
                                      onehot_labels=False, verbose=0, seed=23)
                expected = (train_x, train_y, val_x, val_y, test_images, test_labels)
                for actual, reference in zip(prepared, expected):
                    self.assert_arrays_equal(actual, reference)


# Direct invocation runs only the source-order loader regressions.
if __name__ == "__main__":
    unittest.main()
