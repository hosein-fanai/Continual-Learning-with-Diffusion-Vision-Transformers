"""Regression tests for callback identity, label mapping, copying, and reporting.

Small eager Keras models exercise task-boundary failure isolation and callback
lifecycles without downloading datasets or changing the global numeric policy.
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import tensorflow as tf

from common.callbacks.decoder_accuracy import DecoderAccuracy
from common.argument_saver import ArgumentSaver
from common.config import resolve_continual_schedule
from common.continual_reporting import continual_metrics
from common.learner import _remap_continual_labels
from common.model import copy_model
from common.recovery import (
    _array_recovery_descriptor,
    _decode_json,
    _encode_json,
    callback_recovery_descriptor,
)


def _classifier(widths: tuple[int, ...]) -> tf.keras.Model:
    """Build a tiny dense classifier with independent trainable layers.

    Args:
        widths (tuple[int, ...]): Positive widths from hidden layers to output.

    Returns:
        model (tf.keras.Model): Built float32 model accepting rows of width two.

    Raises:
        ValueError: If Keras rejects a layer width.
    """
    return tf.keras.Sequential([
        tf.keras.Input((2,)),
        *[tf.keras.layers.Dense(width) for width in widths],
    ])


class AuditBoundaryTests(unittest.TestCase):
    """Check real boundary failures without broad training or external data."""

    def test_tracked_constructor_metadata_does_not_copy_owner_graph(self) -> None:
        """Copy tracked routing lists independently without serializing their owner.

        Returns:
            result (None): Independent plain config lists retain object leaves
                by identity and nested caller mutations cannot alter saved values.

        Raises:
            AssertionError: If copying follows a tracker or shares mutable lists.
        """
        owner = tf.keras.layers.Layer()
        owner.routes = [1, {"ids": [2, 3]}]
        # An unregistered layer would fail if deepcopy followed the tracker graph.
        owner.child = tf.keras.layers.Dense(2)
        saver = ArgumentSaver()
        saved = saver._save_init_args({"routes": owner.routes})
        owner.routes[1]["ids"].append(4)
        self.assertEqual(saved["routes"], [1, {"ids": [2, 3]}])
        self.assertIs(type(saved["routes"]), list)
        self.assertIs(type(saved["routes"][1]), dict)

    def test_eager_metric_state_and_scalar_arrays_round_trip(self) -> None:
        """Serialize Keras eager metrics and preserve scalar array shape identity.

        Returns:
            result (None): Nested metric values restore without tensor objects,
                scalar ndarrays retain rank zero, and scalar/vector hashes differ.

        Raises:
            AssertionError: If numeric checkpoint values or shape identity change.
        """
        metrics = {"evaluation": {"loss": tf.constant(.125, tf.float64),
                    "scores": tf.constant([.25, .75], tf.float64)}}
        recovered = _decode_json(_encode_json(metrics))
        self.assertEqual(recovered["evaluation"]["loss"], .125)
        np.testing.assert_array_equal(recovered["evaluation"]["scores"], [.25, .75])
        scalar = np.asarray(7, dtype=np.int64)
        restored = _decode_json(_encode_json(scalar))
        self.assertEqual(restored.shape, ())
        self.assertEqual(restored.dtype, scalar.dtype)
        self.assertEqual(restored.item(), 7)
        self.assertNotEqual(
            _array_recovery_descriptor(scalar)["sha256"],
            _array_recovery_descriptor(np.asarray([7], dtype=np.int64))["sha256"],
        )

    def test_schedule_rejects_rounded_classes_and_task_widths(self) -> None:
        """Prevent fractional or boolean choices from changing the class stream.

        Returns:
            result (None): Discrete controls reject coercion and accept NumPy ints.

        Raises:
            AssertionError: If invalid IDs/counts silently become another design.
        """
        for options in (
            {"class_order": [0, 1.9, 2]},
            {"class_order": [False, 1, 2]},
            {"task_size": 1.5},
            {"task_size": True},
            {"available_class_num": 3.0},
            {"task_groups": [[0.0], [1], [2]]},
        ):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "integer"):
                resolve_continual_schedule(3, **options)
        order, groups = resolve_continual_schedule(
            np.int64(3), class_order=np.asarray([2, 0, 1], dtype=np.int32)
        )
        self.assertEqual(order, [2, 0, 1])
        self.assertEqual(groups, [[2], [0], [1]])

    def test_empty_label_mapping_retains_onehot_and_sparse_contracts(self) -> None:
        """Keep empty held-out splits valid for every supported representation.

        Returns:
            result (None): Assertions verify empty shapes and original dtypes.

        Raises:
            AssertionError: If empty labels fail or change representation.
        """
        for labels, onehot, shape in (
            (np.empty((0, 4), dtype=np.float64), True, (0, 2)),
            (np.empty((0,), dtype=np.int32), False, (0,)),
            (np.empty((0, 1), dtype=np.int64), False, (0, 1)),
        ):
            with self.subTest(shape=labels.shape):
                result = _remap_continual_labels(labels, [3, 1], onehot)
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.dtype, labels.dtype)
        np.testing.assert_array_equal(
            _remap_continual_labels(np.eye(4)[[3, 1]], [3, 1], True),
            np.eye(2),
        )

    def test_invalid_classifier_copy_does_not_change_destination(self) -> None:
        """Validate late shape and head failures before copying early layers.

        Returns:
            result (None): Incompatible destinations keep all original weights.

        Raises:
            AssertionError: If copy accepts incompatible shapes or mutates them.
        """
        source = _classifier((3, 3, 4))
        for widths in ((3, 3, 2), (3, 5, 4)):
            destination = _classifier(widths)
            before = destination.get_weights()
            with self.subTest(widths=widths), self.assertRaises(ValueError):
                copy_model(source, destination)
            for actual, expected in zip(destination.get_weights(), before):
                np.testing.assert_array_equal(actual, expected)

    def test_expanding_classifier_preserves_prefix_and_new_initializers(self) -> None:
        """Copy a valid shared trunk and retain fresh columns after expansion.

        Returns:
            result (None): Source columns match exactly and new columns persist.

        Raises:
            AssertionError: If transfer changes old or newly initialized columns.
        """
        source = _classifier((3, 2))
        destination = _classifier((3, 4))
        initial_head = destination.layers[-1].get_weights()
        copy_model(source, destination)
        for actual, expected in zip(destination.layers[0].get_weights(), source.layers[0].get_weights()):
            np.testing.assert_array_equal(actual, expected)
        for actual, old, initial in zip(
            destination.layers[-1].get_weights(), source.layers[-1].get_weights(), initial_head
        ):
            np.testing.assert_array_equal(actual[..., :2], old)
            np.testing.assert_array_equal(actual[..., 2:], initial[..., 2:])

    def test_callback_identity_is_stable_across_real_fit(self) -> None:
        """Keep Keras lazy monitor setup outside persistent behavior identity.

        Returns:
            result (None): Early stopping and LR reduction retain their identity
                through two fits while different configured modes remain distinct.

        Raises:
            AssertionError: If runtime comparison setup changes callback identity.
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="loss", mode="min", min_delta=.01),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", mode="min", min_delta=.02),
        ]
        expected = callback_recovery_descriptor(callbacks, strict=True)
        model = _classifier((2, 1))
        model.compile(optimizer="sgd", loss="mse", run_eagerly=True)
        data = tf.data.Dataset.from_tensor_slices((
            tf.constant([[0., 1.], [1., 0.]]), tf.constant([[1.], [0.]])
        )).batch(2)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        data = data.with_options(options)
        for _ in range(2):
            model.fit(data, epochs=1, callbacks=callbacks, verbose=0)
            self.assertEqual(callback_recovery_descriptor(callbacks, strict=True), expected)
        for callback_type in (tf.keras.callbacks.EarlyStopping, tf.keras.callbacks.ReduceLROnPlateau):
            minimize = callback_type(monitor="loss", mode="min")
            maximize = callback_type(monitor="loss", mode="max")
            self.assertNotEqual(
                callback_recovery_descriptor([minimize], strict=True),
                callback_recovery_descriptor([maximize], strict=True),
            )

    def test_decoder_accuracy_rejects_broadcastable_mismatches(self) -> None:
        """Reject one score row for multiple generated labels before logging.

        Returns:
            result (None): A broadcasting-shaped error leaves logs untouched.

        Raises:
            AssertionError: If callback reports accuracy for misaligned rows.
        """
        callback = DecoderAccuracy(lambda values: tf.constant([[1., 0.]]), 1)
        callback.set_model(SimpleNamespace(sample=lambda **kwargs: (
            tf.zeros((2, 1)), tf.constant([0, 1]),
        )))
        logs = {"loss": 1.}
        with self.assertRaises(tf.errors.InvalidArgumentError):
            callback.on_epoch_end(0, logs)
        self.assertEqual(logs, {"loss": 1.})

    def test_current_image_callback_has_stable_recovery_identity(self) -> None:
        """Authenticate the public image callback under task-local seed changes.

        Returns:
            result (None): Current callback names are accepted and configuration
                changes are detected without binding identity to output paths.

        Raises:
            AssertionError: If current callback naming or seed reset breaks recovery.
        """
        from diffusion.callbacks.image_generator import ImageGenerator

        callback = ImageGenerator(show_images=True, results_path=None, seed=11)
        descriptor = callback_recovery_descriptor([callback], strict=True)
        callback.seed = 31
        callback.set_artifact_prefix("task-2")
        self.assertEqual(callback_recovery_descriptor([callback], strict=True), descriptor)
        callback.add_null_label = False
        self.assertNotEqual(callback_recovery_descriptor([callback], strict=True), descriptor)

    def test_decoder_accuracy_accepts_sparse_columns(self) -> None:
        """Compare one prediction per sparse column label without broadcasting.

        Returns:
            result (None): Two correct column labels give exact unit accuracy.

        Raises:
            AssertionError: If column labels create pairwise comparisons.
        """
        callback = DecoderAccuracy(lambda values: tf.eye(2), 1)
        callback.set_model(SimpleNamespace(sample=lambda **kwargs: (
            tf.zeros((2, 1)), tf.constant([[0], [1]]),
        )))
        logs = {}
        callback.on_epoch_end(0, logs)
        self.assertEqual(float(logs["decoder_accuracy"]), 1.)

    def test_continual_metrics_reject_rectangular_or_rankless_inputs(self) -> None:
        """Prevent silent omission of scores from malformed task matrices.

        Returns:
            result (None): Square/empty contracts hold and malformed shapes fail.

        Raises:
            AssertionError: If incompatible task axes are accepted.
        """
        for matrix in ([[.8, .2]], [[.8], [.7]], [.8], .8, np.empty((0, 3))):
            with self.subTest(matrix=matrix), self.assertRaisesRegex(ValueError, "square"):
                continual_metrics(matrix)
        self.assertTrue(all(np.isnan(value) for value in continual_metrics([]).values()))
        self.assertEqual(continual_metrics([[.8]])["final_average_accuracy"], .8)


# Run the boundary regressions when invoked as a standalone module.
if __name__ == "__main__":
    unittest.main()
