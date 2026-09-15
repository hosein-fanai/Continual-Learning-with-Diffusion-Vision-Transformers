"""Reject ambiguous counts before training state or sampling labels change."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from autoencoder.variational_autoencoder import VariationalAutoencoder


class AutoencoderInputValidationTests(unittest.TestCase):
    """Exercise public count/class boundaries without starting long training."""

    def setUp(self) -> None:
        """Create a small conditional VAE for each boundary check.

        Returns:
            result (None): Stores an uncompiled model and aligned float32 data.
        """

        self.model = VariationalAutoencoder(
            data_dim=4, latent_dim=2, hiddens_dims=(), conditioned=True,
            class_num=3, compile=False, seed=21,
        )
        self.x = np.ones((2, 4), np.float32)
        self.y = np.eye(3, dtype=np.float32)[[0, 2]]

    def tearDown(self) -> None:
        """Release Keras test models and reset their global naming state.

        Returns:
            result (None): Clears the isolated test process's Keras session.
        """

        tf.keras.backend.clear_session()

    def test_invalid_training_counts_fail_before_fit_or_metadata_changes(self) -> None:
        """Reject zero/fractional budgets before a repeated dataset can be fitted.

        Returns:
            result (None): Passes when invalid settings leave class metadata intact.

        Raises:
            AssertionError: If a rejected setting calls fit or changes observed IDs.
        """

        invalid = {
            "steps_per_epoch": (0, -1, 1.5, True),
            "train_num": (0, -2, 1.5, True),
            "epochs": (0, -1, 1.5, True),
            "batch_size": (0, -1, 1.5, True),
            "shuffle_buffer": (-1, 1.5, True),
        }
        with mock.patch.object(self.model, "fit") as fit:
            for name, values in invalid.items():
                for value in values:
                    with self.subTest(parameter=name, value=value):
                        with self.assertRaises(ValueError):
                            self.model.train(
                                self.x, self.y, callbacks_list=[], **{name: value}
                            )
                        self.assertEqual(self.model.seen_classes, [])
            fit.assert_not_called()

    def test_sampling_rejects_ambiguous_labels_and_counts(self) -> None:
        """Keep requested classes/counts exact instead of truncating floats.

        Returns:
            result (None): Passes when fractions, booleans, and negative IDs fail.

        Raises:
            AssertionError: If the sampler silently accepts an invalid parameter.
        """

        for value in (1.5, True, -1):
            with self.subTest(count=value), self.assertRaises(ValueError):
                self.model.sample(labels=[0], samples_per_label=value)
            with self.subTest(label=value), self.assertRaises(ValueError):
                self.model.sample(labels=[value], samples_per_label=1)
        # Reject fractional sample counts even when no labels are requested.
        with self.assertRaises(ValueError):
            self.model.sample(labels=[], samples_per_label=1.5)
        samples, labels = self.model.sample(
            labels=[np.int64(2)], samples_per_label=np.int64(0)
        )
        self.assertEqual(samples.shape, (0, 4))
        self.assertEqual(labels.shape, (0,))

    def test_constructor_requires_integer_dimensions(self) -> None:
        """Reject fractional/boolean geometry before creating the encoder.

        Returns:
            result (None): Passes when every architecture dimension is exact.

        Raises:
            AssertionError: If the constructor coerces ambiguous dimensions.
        """

        invalid = (
            {"data_dim": 4.5}, {"latent_dim": True},
            {"hiddens_dims": (2.5,)}, {"conditioned": True, "class_num": 2.5},
        )
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                VariationalAutoencoder(compile=False, **options)


# Run these validation regressions when invoked directly.
if __name__ == "__main__":
    unittest.main()
