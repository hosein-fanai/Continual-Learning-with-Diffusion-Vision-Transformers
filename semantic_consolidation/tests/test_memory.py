"""Sampling and persistent-state checks for the temporary modulation bank."""

from __future__ import annotations

import unittest

import numpy as np
import tensorflow as tf

from semantic_consolidation.config import RouteSettings
from semantic_consolidation.memory import ClassBalancedPool, ModulationBank, affine_modulation


class ModulationMemoryTests(unittest.TestCase):
    """Check class-independent initialization, retained state, and valid pairs."""

    def test_class_initialization_is_independent_of_addition_order(self) -> None:
        """A given class receives reproducible state regardless of task grouping."""

        settings = RouteSettings()
        first = ModulationBank(settings, dimension=5, seed=31)
        second = ModulationBank(settings, dimension=5, seed=31)
        first.add([7, 2])
        second.add([2])
        second.add([7])
        for class_id in (2, 7):
            for actual, expected in zip(first.vectors[class_id], second.vectors[class_id]):
                np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        self.assertEqual(first.nbytes, 2 * 2 * 5 * np.dtype("float32").itemsize)

    def test_add_and_frozen_snapshot_preserve_old_values(self) -> None:
        """Adding classes preserves old variables; frozen values never alias them."""

        bank = ModulationBank(RouteSettings(), dimension=4, seed=17)
        bank.add([0])
        original_variables = bank.vectors[0]
        frozen = bank.frozen()
        old_values = [variable.numpy().copy() for variable in original_variables]
        bank.add([0, 1])
        for variable, original, expected in zip(bank.vectors[0], original_variables, old_values):
            self.assertIs(variable, original)
            np.testing.assert_array_equal(variable.numpy(), expected)
            variable.assign_add(tf.ones_like(variable))
        for value, expected in zip(frozen[0], old_values):
            self.assertNotIsInstance(value, tf.Variable)
            np.testing.assert_array_equal(value.numpy(), expected)

    def test_old_vectors_do_not_drift_from_adam_momentum(self) -> None:
        """Omitting old variables from later updates preserves them exactly."""

        bank = ModulationBank(RouteSettings(), dimension=3, seed=23)
        bank.add([0, 1])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        # Register both gate families before exercising updates on disjoint variables.
        if callable(getattr(optimizer, "build", None)):
            optimizer.build([variable for class_id in (0, 1) for variable in bank.vectors[class_id]])
        optimizer.apply_gradients((tf.ones_like(variable), variable) for variable in bank.vectors[0])
        old_values = [variable.numpy().copy() for variable in bank.vectors[0]]
        for _ in range(3):
            optimizer.apply_gradients((tf.ones_like(variable), variable) for variable in bank.vectors[1])
        for variable, expected in zip(bank.vectors[0], old_values):
            np.testing.assert_array_equal(variable.numpy(), expected)
        self.assertEqual(int(optimizer.iterations.numpy()), 4)

    def test_affine_parameters_are_bounded_and_shared_across_rows(self) -> None:
        """Large latent parameters saturate at configured channel gain/bias limits."""

        settings = RouteSettings(gain_limit=0.5, bias_limit=0.2)
        features = tf.constant([[1., 2.], [3., 4.]])
        actual = affine_modulation(features, [100., -100.], [-100., 100.], settings)
        expected = features.numpy() * [1.5, 0.5] + [-0.2, 0.2]
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6)

    def test_pool_draw_has_distinct_rows_and_balanced_negative_classes(self) -> None:
        """One selected class supplies half the batch, with distinct observations."""

        labels = np.repeat([0, 1, 2], 6)
        pool = ClassBalancedPool(np.arange(len(labels))[:, None], labels)
        images, drawn_labels, positive = pool.draw(0, 8, np.random.default_rng(7))
        indices = images[:, 0].astype(int)
        self.assertEqual(len(indices), 8)
        self.assertEqual(len(set(indices.tolist())), 8)
        np.testing.assert_array_equal(drawn_labels, labels[indices])
        np.testing.assert_array_equal(positive, drawn_labels == 0)
        np.testing.assert_array_equal(np.bincount(drawn_labels, minlength=3), [4, 2, 2])

    def test_pool_draw_is_reproducible_and_shrinks_without_replacement(self) -> None:
        """Small classes reduce batch size instead of duplicating contrastive rows."""

        labels = np.array([0, 0, 0, 1, 1, 1, 1])
        pool = ClassBalancedPool(np.arange(7)[:, None], labels)
        first = pool.draw(0, 16, np.random.default_rng(11))
        second = pool.draw(0, 16, np.random.default_rng(11))
        for actual, expected in zip(first, second):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(first[0]), 6)
        self.assertEqual(int(first[2].sum()), 3)
        self.assertEqual(len(np.unique(first[0])), 6)

    def test_pool_handles_fewer_negative_rows_than_positive_rows(self) -> None:
        """An exhausted negative class is removed without replacement or looping."""

        pool = ClassBalancedPool(np.arange(6)[:, None], [0, 0, 0, 0, 0, 1])
        images, _, positive = pool.draw(0, 8, np.random.default_rng(13))
        self.assertEqual(len(images), 5)
        self.assertEqual(int(positive.sum()), 4)
        self.assertEqual(len(np.unique(images)), 5)

    def test_pool_rejects_missing_pair_families_and_nonfinite_images(self) -> None:
        """Malformed pools fail before acquiring an invalid class relation."""

        invalid_pools = (
            (np.zeros((2, 1)), [0, 1]),
            (np.zeros((3, 1)), [0, 0, 0]),
            (np.zeros((3, 1)), [0, 1]),
            (np.array([[0.], [float("nan")], [1.]]), [0, 0, 1]),
        )
        for images, labels in invalid_pools:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                ClassBalancedPool(images, labels)
        pool = ClassBalancedPool(np.arange(3)[:, None], [0, 1, 1])
        with self.assertRaises(ValueError):
            pool.draw(0, 4, np.random.default_rng(0))

    def test_labels_cannot_be_truncated_wrapped_or_negative(self) -> None:
        """Malformed IDs must not change class membership in the loss pairs."""

        for labels in ([0.9, 0.1, 1.0], [0, 0, -1], [0, 0, 2 ** 32], [False, False, True]):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "sparse integer IDs"):
                ClassBalancedPool(np.zeros((3, 1)), labels)
        pool = ClassBalancedPool(np.zeros((4, 1)), [0, 0, 1, 1])
        for size in (True, 3, 4.5):
            with self.subTest(batch_size=size), self.assertRaises(ValueError):
                pool.draw(0, size, np.random.default_rng(0))
        for focus in (False, 0.5, 3):
            with self.subTest(focus=focus), self.assertRaises(ValueError):
                pool.draw(focus, 4, np.random.default_rng(0))


# Run this module directly while keeping imports free of execution side effects.
if __name__ == "__main__":
    unittest.main()
