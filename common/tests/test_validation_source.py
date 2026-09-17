"""Verify optional full-training/test-validation and existing internal splits."""

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import train_test_split

from common.config import Config, DatasetConfig, load_config, save_config
from common.dataloader import get_datasets


class ValidationSourceTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.tile(np.arange(2, dtype=np.uint8), 20).reshape(-1, 1)
        self.images = np.broadcast_to(
            np.arange(40, dtype=np.uint8)[:, None, None, None], (40, 32, 32, 3)
        ).copy()
        self.test_images = np.broadcast_to(
            np.arange(200, 206, dtype=np.uint8)[:, None, None, None], (6, 32, 32, 3)
        ).copy()
        self.test_labels = np.tile(np.arange(2, dtype=np.uint8), 3).reshape(-1, 1)
        # The existing loader groups selected rows by original class ID.
        self.test_order = np.argsort(self.test_labels[:, 0], kind="stable")
        self.train_order = np.argsort(self.labels[:, 0], kind="stable")
        self.train_ids, self.reserved_ids = train_test_split(
            np.arange(40), test_size=0.2, stratify=self.labels, random_state=17,
        )

    def config(self, source="split", **dataset_overrides):
        return Config(
            dataset={
                "name": "cifar10", "preprocess": "", "validation_ratio": 0.2,
                "validation_source": source, "batch_size": 4, "shuffle_buffer": 0,
                **dataset_overrides,
            },
            model={"name": "dit_classifier", "show_network_summary": False},
            training={"task": "joint", "seed": 17},
        )

    def load(self, config):
        with patch("tensorflow.keras.datasets.cifar10.load_data", return_value=(
            (self.images, self.labels), (self.test_images, self.test_labels)
        )):
            return get_datasets(config)

    @staticmethod
    def rows(dataset):
        batches = list(dataset.as_numpy_iterator())
        return np.concatenate([batch[0] for batch in batches]), np.concatenate([
            batch[1] for batch in batches
        ])

    def test_default_preserves_internal_validation_and_excludes_test(self):
        config = self.config()
        train, validation = self.load(config)
        training, _ = self.rows(train)
        validating, _ = self.rows(validation)
        np.testing.assert_array_equal(training[:, 0, 0, 0], self.train_ids)
        np.testing.assert_array_equal(validating[:, 0, 0, 0], self.reserved_ids)
        self.assertEqual(config.dataset.split_metadata, {})
        self.assertNotIn("data_split", config.hpo)
        self.assertTrue(config.dataset.drop_remainder)

    def test_test_source_uses_all_official_training_without_internal_split(self):
        config = self.config("test", batch_size=6, drop_remainder=False)
        with patch("sklearn.model_selection.train_test_split") as splitter:
            train, validation = self.load(config)
            splitter.assert_not_called()
        training, train_labels = self.rows(train)
        validating, val_labels = self.rows(validation)
        np.testing.assert_array_equal(training[:, 0, 0, 0], self.train_order)
        np.testing.assert_array_equal(train_labels, self.labels[self.train_order, 0])
        np.testing.assert_array_equal(validating, self.test_images[self.test_order])
        np.testing.assert_array_equal(val_labels, self.test_labels[self.test_order, 0])
        self.assertEqual(len(training), 40)
        self.assertEqual(config.dataset.trainset_len, 7)
        self.assertEqual(config.dataset.validation_ratio, 0.2)
        metadata = config.dataset.split_metadata
        self.assertEqual(metadata["official_training_rows"], 40)
        self.assertEqual(metadata["training_rows_selected"], 40)
        self.assertEqual(metadata["training_rows_per_epoch"], 40)
        self.assertEqual(metadata["training_source"], "official_train")
        self.assertEqual(metadata["internal_validation_ratio"], 0)
        self.assertEqual(metadata["requested_validation_ratio"], 0.2)
        self.assertEqual(metadata["reserved_internal_validation_rows"], 0)
        self.assertEqual(metadata["internal_validation_usage"], "not_created")
        self.assertIsNone(metadata["internal_validation_location"])
        self.assertFalse(metadata["drop_remainder"])
        self.assertFalse(metadata["effective_drop_remainder"])
        self.assertEqual(metadata["validation_rows_selected"], 6)
        self.assertEqual(metadata["selected_validation_location"], "official_test")
        self.assertFalse(metadata["independent_test_estimate"])
        self.assertEqual(config.hpo["data_split"], metadata)

    def test_preprocessing_fits_all_official_training_and_never_test(self):
        # A formerly reserved outlier must participate in fitted statistics;
        # official test pixels lie outside the complete training extrema.
        self.images[self.reserved_ids[0]] = 150
        config = self.config("test", preprocess="standardize")
        train, validation = self.load(config)
        training, _ = self.rows(train)
        validating, _ = self.rows(validation)
        fitted = self.images[self.train_order].astype(np.float32)
        minimum, span = fitted.min(), fitted.max() - fitted.min()
        np.testing.assert_allclose(training, 2 * (fitted - minimum) / span - 1)
        np.testing.assert_allclose(
            validating,
            2 * (self.test_images[self.test_order].astype(np.float32) - minimum) / span - 1,
        )
        self.assertGreater(float(validating.min()), 1.0)
        self.assertEqual(config.dataset.split_metadata["preprocess_fit_source"],
                         "official_train")

    def test_optional_drop_remainder_preserves_default_and_retains_validation_tail(self):
        for source, selected_count in (("split", 32), ("test", 40)):
            for drop_remainder in (True, False):
                with self.subTest(source=source, drop_remainder=drop_remainder):
                    config = self.config(source, batch_size=7, drop_remainder=drop_remainder)
                    train, validation = self.load(config)
                    expected = selected_count // 7 * 7 if drop_remainder else selected_count
                    self.assertEqual(len(self.rows(train)[0]), expected)
                    self.assertEqual(len(self.rows(validation)[0]), 8 if source == "split" else 6)
                    self.assertEqual(config.dataset.trainset_len, (expected + 6) // 7)
                    if source == "test":
                        self.assertEqual(config.dataset.split_metadata["training_rows_per_epoch"],
                                         expected)

    def test_validation_cap_applies_to_selected_test_and_metadata_round_trips(self):
        config = self.config("test", max_train_samples=9, max_val_samples=3,
                             drop_remainder=False)
        train, validation = self.load(config)
        training, _ = self.rows(train)
        validating, labels = self.rows(validation)
        self.assertEqual(len(training), 9)
        self.assertEqual(len(validating), 3)
        self.assertTrue(set(validating[:, 0, 0, 0]).issubset(set(range(200, 206))))
        self.assertEqual(set(labels), {0, 1})
        self.assertEqual(config.dataset.split_metadata["validation_rows_before_cap"], 6)
        self.assertEqual(config.dataset.split_metadata["validation_rows_selected"], 3)
        self.assertEqual(config.dataset.split_metadata["training_rows_before_cap"], 40)
        self.assertEqual(config.dataset.split_metadata["training_rows_per_epoch"], 9)
        self.assertEqual(config.dataset.split_metadata["reserved_internal_validation_rows"], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            save_config(config, path)
            restored = load_config(path)
        self.assertEqual(asdict(config), asdict(restored))

    def test_invalid_or_unsupported_sources_fail_before_loading(self):
        with self.assertRaisesRegex(ValueError, "validation_source"):
            DatasetConfig(validation_source="testing")
        with self.assertRaisesRegex(ValueError, "drop_remainder"):
            DatasetConfig(drop_remainder="False")
        configurations = (
            (self.config("test"), "ordinary training"),
            (self.config("test", return_features=True), "feature"),
            (self.config("test"), "use_valset"),
        )
        configurations[0][0].training.task = "continual"
        configurations[2][0].training.use_valset = False
        for config, message in configurations:
            with self.subTest(message=message), patch(
                "common.dataloader.load_cifar10"
            ) as loader:
                with self.assertRaisesRegex(ValueError, message):
                    get_datasets(config)
                loader.assert_not_called()
        with patch("common.dataloader.load_cifar10") as loader:
            with self.assertRaisesRegex(ValueError, "validation_source"):
                get_datasets(dataset_name="cifar10", validation_source="typo")
            loader.assert_not_called()

    def test_direct_api_selects_test_rows_without_splitting_or_dropping_training(self):
        with patch("tensorflow.keras.datasets.cifar10.load_data", return_value=(
            (self.images, self.labels), (self.test_images, self.test_labels)
        )):
            train, validation = get_datasets(
                dataset_name="cifar10", task="joint", validation_source="test",
                validation_ratio=0.2, preprocess="", seed=17, batch_size=7,
                shuffle_buffer=0, drop_remainder=False,
            )
        np.testing.assert_array_equal(self.rows(train)[0][:, 0, 0, 0], self.train_order)
        np.testing.assert_array_equal(self.rows(validation)[0], self.test_images[self.test_order])

    def test_all_ordinary_tasks_support_explicit_official_test_validation(self):
        for task in ("legacy", "generation", "classification", "joint"):
            with self.subTest(task=task):
                config = self.config("test", drop_remainder=False)
                config.training.task = task
                train, validation = self.load(config)
                self.assertEqual(len(self.rows(train)[0]), 40)
                np.testing.assert_array_equal(self.rows(validation)[0],
                                              self.test_images[self.test_order])

    def test_switching_back_to_split_clears_stale_test_provenance(self):
        config = self.config("test")
        self.load(config)
        config.dataset.validation_source = "split"
        _, validation = self.load(config)
        np.testing.assert_array_equal(self.rows(validation)[0][:, 0, 0, 0], self.reserved_ids)
        self.assertEqual(config.dataset.split_metadata, {})
        self.assertNotIn("data_split", config.hpo)


if __name__ == "__main__":
    unittest.main()
