"""Remote-only focused checks for frozen TMCL readout semantics and native fitting."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import tensorflow as tf

from semantic_consolidation.tmcl_evaluation import (
    classification_metrics, extract_frozen_features, knn_evaluate, 
    linear_evaluate, paper_transfer_metrics, weighted_knn_predict
)


class _EmptyBank:
    """Provide a deterministic empty memory bank for bounded readout interface checks."""

    def variables_for(self) -> list[tf.Variable]:
        """Return no gates because this fixture tests unmodulated readout ownership."""

        return []


class _Adapter:
    """Expose two distinct image-side features and a state-mutation fault switch."""

    def __init__(self, mutate: bool = False) -> None:
        """Retain a genuine TensorFlow state variable for frozen-state enforcement."""

        self.network = SimpleNamespace(variables=[tf.Variable(1., trainable=False)])
        self.bank = _EmptyBank()
        self.mutate = mutate

    def readout_spec(self, last_layers: int = 4) -> dict[str, object]:
        """Declare fixture feature width separately from production DiT topology tests."""

        return {"width": 2, "last_layers": last_layers, "endpoint": "test_two_sides"}

    def readout_features(self, images: tf.Tensor, last_layers: int = 4) -> tf.Tensor:
        """Extract horizontal side means, optionally injecting a forbidden state mutation."""

        # The fault mode verifies that apparently successful feature output is insufficient.
        if self.mutate:
            self.network.variables[0].assign_add(1.)
        return tf.reduce_mean(images, axis=[1, 3])


class TMCLEvaluationTests(unittest.TestCase):
    """Verify all-class metrics, weighted neighbors, online flip equivalence and budgets."""

    def setUp(self) -> None:
        """Use tiny explicit images without downloading a dataset or learning an encoder."""

        self.adapter = _Adapter()
        self.images = np.asarray([
            [[[1.], [.1]], [[1.], [.1]]], 
            [[[.1], [1.]], [[.1], [1.]]]
        ], dtype="float32")
        self.train = np.tile(self.images, (4, 1, 1, 1))
        self.labels = np.tile(np.asarray([7, 3], dtype="int32"), 4)

    def tearDown(self) -> None:
        """Release only this separate validation process's native probe objects."""

        tf.keras.backend.clear_session()

    def test_frozen_features_keep_rows_and_cache_exact_flip(self) -> None:
        """Caching both deterministic image views preserves the online flip alternatives."""

        clean = extract_frozen_features(self.adapter, self.train, batch_size=3)
        flipped = extract_frozen_features(self.adapter, self.train, batch_size=3, flip=True)
        np.testing.assert_array_equal(clean, self.train.mean(axis=(1, 3)))
        np.testing.assert_array_equal(flipped, clean[:, ::-1])
        self.assertEqual(float(self.adapter.network.variables[0].numpy()), 1.)

    def test_feature_extraction_detects_nontrainable_state_mutation(self) -> None:
        """A changed normalization/counter variable invalidates a supposedly frozen probe."""

        with self.assertRaisesRegex(RuntimeError, "changed"):
            extract_frozen_features(_Adapter(mutate=True), self.images)

    def test_task_metrics_use_global_argmax_without_label_masking(self) -> None:
        """Selecting task rows must never turn all-class mistakes into task-aware hits."""

        result = classification_metrics(
            np.asarray([7, 3, 7, 3]), np.asarray([7, 7, 3, 3]), [7, 3], [[7], [3]]
        )
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual([task["accuracy"] for task in result["tasks"]], [.5, .5])
        with self.assertRaisesRegex(ValueError, "partition"):
            classification_metrics(np.asarray([7]), np.asarray([7]), [7, 3], [[7], [7]])

    def test_knn_weights_cosine_neighbors_and_preserves_noncontiguous_labels(self) -> None:
        """The closest label can beat a two-row majority through temperature-weighted votes."""

        bank = np.asarray([[1., 0.], [.7, .7], [0., 1.]], dtype="float32")
        labels = np.asarray([7, 3, 3], dtype="int32")
        queries = np.asarray([[1., 0.], [0., 1.]], dtype="float32")
        prediction = weighted_knn_predict(bank, labels, queries, [7, 3], neighbors=3, batch_size=1)
        np.testing.assert_array_equal(prediction, [7, 3])
        scaled = weighted_knn_predict(bank * np.asarray([[2.], [8.], [3.]]), labels, queries * 4., [7, 3], neighbors=3)
        np.testing.assert_array_equal(scaled, prediction)
        with self.assertRaisesRegex(ValueError, "neighborhood"):
            weighted_knn_predict(bank, labels, queries, [7, 3], neighbors=4)
        zero_query = weighted_knn_predict(bank, labels, np.zeros((1, 2)), [7, 3], neighbors=3)
        np.testing.assert_array_equal(zero_query, [3])
        with self.assertRaisesRegex(ValueError, "finite"):
            weighted_knn_predict(bank, labels, np.full((1, 2), np.nan), [7, 3], neighbors=3)

    def test_native_linear_probe_has_full_budget_and_epoch_cosine(self) -> None:
        """A real project DNN head trains separately while encoder state remains exact."""

        result = linear_evaluate(
            self.adapter, self.train, self.labels, self.images, np.asarray([7, 3]), [7, 3], 
            task_classes=[[7], [3]], epochs=2, batch_size=4, feature_batch_size=3, seed=19
        )
        self.assertEqual(result["optimizer_updates"], 4)
        self.assertEqual(result["metrics"]["count"], 2)
        self.assertEqual(result["train_rows"], 8)
        self.assertEqual(result["feature_spec"]["last_layers"], 4)
        self.assertFalse(result["l2_normalize"])
        np.testing.assert_allclose(result["learning_rates"], [.1 * 4. / 256., .05 * 4. / 256.])
        self.assertEqual(len(result["head_weights"]), 2)
        self.assertEqual(float(self.adapter.network.variables[0].numpy()), 1.)

    def test_knn_endpoint_uses_last_layer_and_all_bank_rows(self) -> None:
        """The readout bank uses all rows with an independent clean endpoint."""

        result = knn_evaluate(
            self.adapter, self.train, self.labels, self.images, np.asarray([7, 3]), [7, 3], 
            neighbors=3, feature_batch_size=3
        )
        self.assertEqual(result["bank_rows"], 8)
        self.assertEqual(result["feature_spec"]["last_layers"], 1)
        self.assertEqual(result["metrics"]["accuracy"], 1.)

    def test_paper_transfer_uses_independent_baselines_and_task_means(self) -> None:
        """Independent arithmetic distinguishes the paper metric from conventional BWT."""

        matrix = np.asarray([[.8, .2, .3], [.6, .7, .4], [.4, .5, .6]])
        baseline = np.asarray([.5, .6, .7])
        result = paper_transfer_metrics(matrix, baseline)
        self.assertAlmostEqual(result["paper_bt"], -.05)
        self.assertAlmostEqual(result["paper_ft"], -.375)
        self.assertAlmostEqual(result["paper_bt_percentage_points"], -5.)
        self.assertAlmostEqual(result["paper_ft_percentage_points"], -37.5)
        self.assertNotAlmostEqual(result["paper_bt"], -.3)
        self.assertNotAlmostEqual(result["paper_bt"], -.1 / 3.)
        altered = matrix.copy()
        np.fill_diagonal(altered, [.9, .1, .2])
        self.assertEqual(paper_transfer_metrics(altered, baseline), result)

    def test_paper_transfer_requires_future_cells_and_independent_baseline_vector(self) -> None:
        """Triangular/missing endpoint data cannot silently become forward-transfer zeros."""

        with self.assertRaisesRegex(ValueError, "complete finite"):
            paper_transfer_metrics(np.asarray([[.8, np.nan], [.6, .7]]), np.asarray([.5, .6]))
        with self.assertRaisesRegex(ValueError, "complete finite"):
            paper_transfer_metrics(np.ones((2, 2)), np.asarray([.5]))
        with self.assertRaisesRegex(ValueError, "complete finite"):
            paper_transfer_metrics(np.ones((1, 1)), np.asarray([.5]))
        with self.assertRaisesRegex(ValueError, "fractions"):
            paper_transfer_metrics(np.asarray([[80., 20.], [60., 70.]]), np.asarray([.5, .6]))


# Execute only on the user's coordinated remote environment, never on the laptop.
if __name__ == "__main__":
    unittest.main()
