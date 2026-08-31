"""Focused integration tests for optional continual research controls."""

from __future__ import annotations

import os
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import tensorflow as tf

from common.dataloader import get_dataset
from common.experiment import (
    create_paired_block_manifest,
    materialize_run_plan,
    write_experiment_manifest,
)
from common.learner import (
    _cached_replay_candidates,
    _replay_cache_path,
    _resolve_baseline_controls,
    _run_continual_tasks,
)
from common.mechanistic import calibration_metrics, replay_quality_metrics


_TEST_SENTINEL = 77.0
"""Feature value reserved exclusively for the locked synthetic test split."""


def _assert_no_test_sentinel(data: object) -> None:
    """Raise when model prediction/evaluation receives locked test rows.

    Args:
        data (object): NumPy, TensorFlow, or dataset input passed to a model API.

    Returns:
        None.
    """

    # Ignore a missing optional evaluation input.
    if data is None:
        return
    arrays: list[np.ndarray] = []
    # Inspect every batch while preserving the re-iterable dataset itself.
    if isinstance(data, tf.data.Dataset):
        for element in data:
            inputs = element[0] if isinstance(element, tuple) else element
            arrays.append(np.asarray(inputs))
    # Inspect an ordinary eager tensor or NumPy model input directly.
    else:
        arrays.append(np.asarray(data))
    # Fail at the exact model-access boundary if a locked row leaks through.
    if any(np.any(array == _TEST_SENTINEL) for array in arrays):
        raise AssertionError("locked synthetic test rows reached a model API")


class ResearchControlTests(unittest.TestCase):
    """Exercise baseline, exposure, split-isolation, and metadata contracts."""

    @staticmethod
    def _loader(
        indices: object,
        **kwargs: object,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Return balanced train/validation rows and sentinel test rows.

        Args:
            indices (object): Class selection supplied by the continual loader.
            **kwargs (object): Loader compatibility options.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
            np.ndarray, np.ndarray]: Synthetic train, validation, and test pairs.
        """

        del indices, kwargs
        labels = np.repeat(np.asarray([0, 1], dtype="int32"), 6)
        train = (labels.astype("float32") * 0.5 + 0.1)[:, None]
        validation_labels = np.repeat(
            np.asarray([0, 1], dtype="int32"),
            2,
        )
        validation = (
            validation_labels.astype("float32") * 0.5 + 0.15
        )[:, None]
        test_labels = np.repeat(np.asarray([0, 1], dtype="int32"), 2)
        test = np.full((len(test_labels), 1), _TEST_SENTINEL, dtype="float32")
        return (
            train,
            labels,
            validation,
            validation_labels,
            test,
            test_labels,
        )

    @staticmethod
    def _write_template(path: Path) -> None:
        """Write a tiny compiled standalone classifier template.

        Args:
            path (Path): Destination Keras HDF5 path.

        Returns:
            None.
        """

        inputs = tf.keras.Input((1,))
        hidden = tf.keras.layers.Dense(
            4,
            activation="tanh",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=3),
        )(inputs)
        outputs = tf.keras.layers.Dense(
            2,
            activation="softmax",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=5),
        )(hidden)
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.save(path)

    @classmethod
    def _run_args(cls, template_path: Path) -> dict[str, object]:
        """Return fast deterministic arguments shared by continual tests.

        Args:
            template_path (Path): Saved tiny classifier template.

        Returns:
            dict[str, object]: Direct continual API arguments.
        """

        return {
            "class_num": 2,
            "load_dataset_fn": cls._loader,
            "load_dataset_fn_kwargs": {
                "preprocess": None,
                "onehot_labels": False,
            },
            "tuned_model_path": str(template_path),
            "compile_args": {
                "optimizer": tf.keras.optimizers.SGD(learning_rate=0.02),
                "loss": "sparse_categorical_crossentropy",
                "metrics": ["accuracy"],
            },
            "use_loaded_opt": False,
            "batch_size": 4,
            "epochs": 1,
            "plot_results": False,
            "verbose": 0,
            "return_features": False,
            "callback_patience": 0,
            "save_task_checkpoints": False,
            "return_details": True,
            "seed": 43,
            "experiment_phase": "development",
        }

    def test_baselines_and_reservoir_settings_resolve_exactly(self) -> None:
        """Resolve canonical baselines and preserve reservoir settings.

        Args:
            None.

        Returns:
            None.
        """

        source = {
            "maxlen": 7,
            "sample_num": 5,
            "insert_num": 3,
            "seed": 19,
            "strategy": "fifo",
        }
        sequential = _resolve_baseline_controls(
            "sequential",
            None,
            True,
            source,
            False,
            True,
            True,
            True,
        )
        self.assertEqual(sequential[:6], (
            "sequential", True, False, False, False, False,
        ))
        cumulative = _resolve_baseline_controls(
            "cumulative",
            None,
            True,
            source,
            True,
            True,
            True,
            True,
        )
        self.assertEqual(cumulative[:6], (
            "cumulative", False, False, False, False, False,
        ))
        reservoir = _resolve_baseline_controls(
            "reservoir_er",
            None,
            False,
            source,
            False,
            True,
            True,
            True,
        )
        self.assertEqual(reservoir[:6], (
            "reservoir_er", True, True, False, False, False,
        ))
        self.assertEqual(
            reservoir[6],
            {**source, "strategy": "reservoir"},
        )
        self.assertEqual(source["strategy"], "fifo")

    def test_confirmation_requires_manifest_credentials_before_data_access(
        self,
    ) -> None:
        """Refuse an unsealed confirmation request before loading any rows.

        Args:
            None.

        Returns:
            None: Missing manifest credentials fail before loader access.
        """

        loader = Mock(side_effect=AssertionError("loader must not run"))
        with self.assertRaisesRegex(ValueError, "confirmation requires"):
            _run_continual_tasks(
                class_num=2,
                load_dataset_fn=loader,
                experiment_phase="confirmation",
            )
        loader.assert_not_called()

    def test_confirmation_rejects_manifest_schedule_and_seed_mismatches(
        self,
    ) -> None:
        """Authenticate the exact frozen stream before runtime or data access.

        Args:
            None.

        Returns:
            None: Schedule and seed mismatches both fail before loading data.
        """

        manifest = create_paired_block_manifest(
            {
                "raw_teacher": {"snapshot_network_name": "raw"},
                "ema_teacher": {"snapshot_network_name": "ema"},
            },
            [
                {
                    "block_id": "stream-a",
                    "stream_seed": 43,
                    "class_order": [0, 1],
                    "task_groups": [[0], [1]],
                },
                {
                    "block_id": "stream-b",
                    "stream_seed": 47,
                    "class_order": [1, 0],
                    "task_groups": [[1], [0]],
                },
            ],
            seed=17,
            phase="confirmation",
        )
        planned_run = next(
            run for run in materialize_run_plan(manifest)
            if run["block_id"] == "stream-a"
        )

        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "confirmation.json"
            write_experiment_manifest(manifest_path, manifest)
            mismatch_cases = (
                (
                    "schedule",
                    [1, 0],
                    [[1], [0]],
                    43,
                ),
                (
                    "seed",
                    [0, 1],
                    [[0], [1]],
                    44,
                ),
            )
            for error_name, class_order, task_groups, seed in mismatch_cases:
                loader = Mock(
                    side_effect=AssertionError("loader must not run")
                )
                with self.subTest(error_name=error_name), \
                        self.assertRaisesRegex(ValueError, error_name):
                    _run_continual_tasks(
                        class_num=2,
                        load_dataset_fn=loader,
                        class_order=class_order,
                        task_groups=task_groups,
                        seed=seed,
                        experiment_phase="confirmation",
                        experiment_manifest_path=str(manifest_path),
                        experiment_manifest_hash=manifest["manifest_hash"],
                        experiment_run_id=planned_run["run_id"],
                    )
                loader.assert_not_called()

    def test_fixed_total_buffer_exposure_is_exact(self) -> None:
        """Match current and old example exposure despite a tiny reservoir.

        Args:
            None.

        Returns:
            None.
        """

        with tempfile.TemporaryDirectory() as directory:
            template_path = Path(directory) / "tiny_classifier.h5"
            self._write_template(template_path)
            details = _run_continual_tasks(
                **self._run_args(template_path),
                baseline="reservoir_er",
                replay_budget_mode="fixed_total",
                replay_current_examples=4,
                replay_old_examples=3,
                buffer_kwargs={
                    "maxlen": 2,
                    "sample_num": 99,
                    "insert_num": 2,
                    "strategy": "fifo",
                },
            )

        self.assertEqual(details["baseline"], "reservoir_er")
        resources = details["task_resource_metrics"]
        self.assertEqual(len(resources), 2)
        self.assertEqual(resources[0]["current_examples_available"], 6)
        self.assertEqual(resources[0]["current_examples_exposed"], 4)
        self.assertEqual(resources[0]["training_examples_total"], 4)
        self.assertEqual(resources[0]["replay"]["selected_count"], 0)
        self.assertEqual(resources[1]["current_examples_available"], 6)
        self.assertEqual(resources[1]["current_examples_exposed"], 4)
        self.assertEqual(resources[1]["replay"]["candidate_count"], 3)
        self.assertEqual(resources[1]["replay"]["selected_count"], 3)
        self.assertEqual(resources[1]["training_examples_total"], 7)
        # The named baseline ignores insertion subsampling and offers every
        # exposed current row to Algorithm R on both tasks.
        self.assertEqual(
            [task["replay"]["storage_offered_count"] for task in resources],
            [4, 4],
        )

    def test_explicit_reservoir_keeps_insert_num_ablation(self) -> None:
        """Keep sampled insertion available outside the named ER baseline.

        Args:
            None.

        Returns:
            None.
        """

        with tempfile.TemporaryDirectory() as directory:
            template_path = Path(directory) / "tiny_classifier.h5"
            self._write_template(template_path)
            details = _run_continual_tasks(
                **self._run_args(template_path),
                use_buffer=True,
                replay_budget_mode="fixed_total",
                replay_current_examples=4,
                replay_old_examples=2,
                buffer_kwargs={
                    "maxlen": 8,
                    "sample_num": 99,
                    "insert_num": 2,
                    "strategy": "reservoir",
                },
            )

        self.assertIsNone(details["baseline"])
        self.assertEqual(
            [
                task["replay"]["storage_offered_count"]
                for task in details["task_resource_metrics"]
            ],
            [2, 2],
        )

    def test_optimizer_steps_per_epoch_matches_updates(self) -> None:
        """Repeat selected pools only when a fixed update budget is requested.

        Args:
            None.

        Returns:
            None.
        """

        with tempfile.TemporaryDirectory() as directory:
            template_path = Path(directory) / "tiny_classifier.h5"
            self._write_template(template_path)
            details = _run_continual_tasks(
                **self._run_args(template_path),
                baseline="sequential",
                optimizer_steps_per_epoch=3,
            )

        self.assertEqual(details["optimizer_steps_per_epoch"], 3)
        for task in details["task_resource_metrics"]:
            self.assertEqual(task["optimizer_steps_per_epoch"], 3)
            self.assertEqual(task["optimizer_updates"]["classifier_optimizer"], 3)

    def test_optimizer_steps_conflict_is_rejected(self) -> None:
        """Reject two competing sources for the same Keras step budget.

        Args:
            None.

        Returns:
            None.
        """

        with tempfile.TemporaryDirectory() as directory:
            template_path = Path(directory) / "tiny_classifier.h5"
            self._write_template(template_path)
            with self.assertRaisesRegex(ValueError, "not both"):
                _run_continual_tasks(
                    **self._run_args(template_path),
                    optimizer_steps_per_epoch=2,
                    fit_kwargs={"steps_per_epoch": 2},
                )

    def test_development_never_predicts_or_evaluates_locked_test_rows(self) -> None:
        """Keep test matrices empty while producing validation CL metrics.

        Args:
            None.

        Returns:
            None.
        """

        original_predict = tf.keras.Model.predict
        original_evaluate = tf.keras.Model.evaluate
        original_get_dataset = get_dataset

        def guarded_predict(
            model: tf.keras.Model,
            x: object,
            *args: object,
            **kwargs: object,
        ) -> object:
            """Reject sentinel inputs before delegating model prediction.

            Args:
                model (tf.keras.Model): Model receiving the prediction request.
                x (object): Prediction inputs.
                *args (object): Remaining positional prediction options.
                **kwargs (object): Remaining keyword prediction options.

            Returns:
                object: Original Keras prediction result.
            """

            _assert_no_test_sentinel(x)
            return original_predict(model, x, *args, **kwargs)

        def guarded_evaluate(
            model: tf.keras.Model,
            x: object = None,
            *args: object,
            **kwargs: object,
        ) -> object:
            """Reject sentinel inputs before delegating model evaluation.

            Args:
                model (tf.keras.Model): Model receiving the evaluation request.
                x (object): Evaluation inputs.
                *args (object): Remaining positional evaluation options.
                **kwargs (object): Remaining keyword evaluation options.

            Returns:
                object: Original Keras evaluation result.
            """

            _assert_no_test_sentinel(x)
            return original_evaluate(model, x, *args, **kwargs)

        def guarded_get_dataset(
            inputs: object,
            *args: object,
            **kwargs: object,
        ) -> tf.data.Dataset:
            """Reject locked rows before any task dataset is constructed.

            Args:
                inputs (object): Candidate dataset inputs.
                *args (object): Remaining positional loader arguments.
                **kwargs (object): Remaining keyword loader arguments.

            Returns:
                tf.data.Dataset: Dataset built by the original helper.
            """

            _assert_no_test_sentinel(inputs)
            return original_get_dataset(inputs, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            template_path = Path(directory) / "tiny_classifier.h5"
            self._write_template(template_path)
            with patch.object(tf.keras.Model, "predict", new=guarded_predict), \
                    patch.object(
                        tf.keras.Model,
                        "evaluate",
                        new=guarded_evaluate,
                    ), patch(
                        "common.learner.get_dataset",
                        new=guarded_get_dataset,
                    ):
                details = _run_continual_tasks(
                    **self._run_args(template_path),
                )

        self.assertFalse(details["test_evaluated"])
        self.assertEqual(details["ordinary_accuracy_matrix"], [])
        self.assertEqual(details["ensemble_accuracy_matrix"], [])
        self.assertEqual(details["continual_metrics"], {})
        self.assertEqual(len(details["validation_accuracy_matrix"]), 2)
        self.assertEqual(
            details["accuracy_matrix"],
            details["validation_accuracy_matrix"],
        )
        self.assertIn(
            "final_average_accuracy",
            details["validation_continual_metrics"],
        )

    def test_metadata_dataset_preserves_replay_mask_as_third_tensor(self) -> None:
        """Shuffle aligned inputs while retaining the third replay-mask tensor.

        Args:
            None.

        Returns:
            None.
        """

        inputs = np.arange(12, dtype="float32").reshape((6, 2))
        labels = np.arange(6, dtype="int32")
        replay_mask = (labels % 2 == 1)
        dataset = get_dataset(
            inputs,
            labels,
            metadata=replay_mask,
            shuffle_buffer=6,
            batch_size=2,
            drop_remainder=False,
            seed=29,
        )
        observed_inputs = []
        observed_labels = []
        observed_mask = []
        for input_batch, label_batch, mask_batch in dataset:
            observed_inputs.extend(np.asarray(input_batch))
            observed_labels.extend(np.asarray(label_batch))
            observed_mask.extend(np.asarray(mask_batch))

        observed_inputs = np.asarray(observed_inputs)
        observed_labels = np.asarray(observed_labels)
        observed_mask = np.asarray(observed_mask)
        self.assertTrue(np.array_equal(
            observed_inputs[:, 0] / 2.0,
            observed_labels,
        ))
        self.assertTrue(np.array_equal(
            observed_mask,
            observed_labels % 2 == 1,
        ))

    def test_matrix_labels_reject_nonfinite_values_before_argmax(self) -> None:
        """Reject NaN and infinity instead of converting them to class IDs.

        Args:
            None.

        Returns:
            None.
        """

        probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]])
        invalid_labels = (
            np.asarray([[1., 0.], [np.nan, 1.]]),
            np.asarray([[1., 0.], [np.inf, 1.]]),
        )
        for labels in invalid_labels:
            # Every entry must be finite before a represented class is selected.
            with self.subTest(labels=labels), self.assertRaisesRegex(
                ValueError,
                "finite",
            ):
                calibration_metrics(probabilities, labels)

    def test_replay_quality_rejects_labels_outside_expected_classes(self) -> None:
        """Keep normalized replay entropy bounded by its declared class set.

        Args:
            None.

        Returns:
            None.
        """

        samples = np.arange(12, dtype="float32").reshape((3, 4))
        # An undeclared third class would make entropy/log(2) exceed one.
        with self.assertRaisesRegex(ValueError, "expected_classes"):
            replay_quality_metrics(
                samples,
                np.asarray([0, 1, 2]),
                expected_classes=[0, 1],
            )

    def test_replay_cache_is_namespaced_and_write_retry_is_idempotent(self) -> None:
        """Separate incompatible contexts and safely reuse an exact retry.

        Args:
            None.

        Returns:
            None.
        """

        samples = np.arange(12, dtype="float32").reshape((3, 4))
        labels = np.asarray([0, 1, 0], dtype="int32")
        with tempfile.TemporaryDirectory() as directory:
            first_path = _replay_cache_path(
                directory,
                1,
                [7, 5],
                3,
                context_fingerprint="a" * 64,
            )
            second_path = _replay_cache_path(
                directory,
                1,
                [7, 5],
                3,
                context_fingerprint="b" * 64,
            )
            self.assertNotEqual(first_path, second_path)
            self.assertIn("classes-7-5", first_path.name)

            written = _cached_replay_candidates(
                samples,
                labels,
                directory,
                "write",
                1,
                [7, 5],
                41,
                "a" * 64,
            )
            retried = _cached_replay_candidates(
                samples.copy(),
                labels.copy(),
                directory,
                "write",
                1,
                [7, 5],
                41,
                "a" * 64,
            )
            self.assertEqual(written[2], retried[2])
            self.assertTrue(np.array_equal(written[0], retried[0]))
            # A conflicting rerun must not replace the immutable common pool.
            with self.assertRaisesRegex(FileExistsError, "incompatible"):
                _cached_replay_candidates(
                    samples + 1,
                    labels,
                    directory,
                    "write",
                    1,
                    [7, 5],
                    41,
                    "a" * 64,
                )

    def test_replay_cache_concurrent_commit_keeps_first_complete_pool(self) -> None:
        """A colliding publisher must authenticate rather than replace a winner.

        Returns:
            None.
        """

        samples = np.arange(8, dtype="float32").reshape((2, 4))
        labels = np.asarray([0, 1], dtype="int32")
        real_link = os.link

        def commit_then_report_collision(source: Path, destination: Path) -> None:
            """Simulate another writer winning immediately before publication.

            Args:
                source (pathlib.Path): Complete private candidate archive.
                destination (pathlib.Path): Shared immutable cache path.

            Returns:
                None.
            """

            real_link(source, destination)
            raise FileExistsError(destination)

        with tempfile.TemporaryDirectory() as directory, patch(
            "common.learner.os.link",
            side_effect=commit_then_report_collision,
        ) as link_mock:
            committed = _cached_replay_candidates(
                samples,
                labels,
                directory,
                "write",
                1,
                [0, 1],
                43,
                "c" * 64,
            )

        self.assertEqual(link_mock.call_count, 1)
        np.testing.assert_array_equal(committed[0], samples)
        np.testing.assert_array_equal(committed[1], labels)


# Run the focused research-control tests when invoked directly.
if __name__ == "__main__":
    unittest.main()
