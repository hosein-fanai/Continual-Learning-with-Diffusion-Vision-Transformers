"""Integration checks for deterministic continual recovery and scheduling."""

from __future__ import annotations

import tempfile
import unittest
import warnings

from pathlib import Path

import numpy as np
import tensorflow as tf

from common.config import resolve_continual_schedule
from common.learner import (
    _continual_metrics,
    _prepare_diffusion_x,
    _run_continual_tasks,
    _validate_supplied_model_runtime,
)


class _InterruptOnSecondFit(tf.keras.callbacks.Callback):
    """Raise once at task two, while retaining a stable descriptor type."""

    def __init__(self, enabled: bool) -> None:
        """Exercise the test helper named __init__.

        Args:
            enabled (bool): Test input named enabled.

        Returns:
            None: Result produced by the test helper.
        """
        super().__init__()
        self.enabled = enabled
        self.fit_count = 0

    def on_train_begin(
        self,
        logs: dict[str, float] | None = None,
    ) -> None:
        """Exercise the test helper named on_train_begin.

        Args:
            logs (dict[str, float] | None): Test input named logs.

        Returns:
            None: Result produced by the test helper.
        """
        del logs
        self.fit_count += 1
        # Select the test action required by this condition.
        if self.enabled and self.fit_count == 2:
            raise RuntimeError("intentional task interruption")


class ContinualIntegrationTests(unittest.TestCase):
    """Exercise the orchestration seams that unit helpers cannot cover."""

    def test_diffusion_preparation_uses_policy_variable_dtype(self) -> None:
        """Diffusion arrays use stable variable precision under each policy.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        original_policy = tf.keras.mixed_precision.global_policy()
        try:
            tf.keras.mixed_precision.set_global_policy("float64")
            float64_result = _prepare_diffusion_x(
                np.asarray([0., 1.], dtype="float32"),
                0.,
                1.,
            )
            self.assertEqual(float64_result.dtype, np.dtype("float64"))

            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            mixed_result = _prepare_diffusion_x(
                np.asarray([0., 1.], dtype="float64"),
                0.,
                1.,
            )
            self.assertEqual(mixed_result.dtype, np.dtype("float32"))
        finally:
            tf.keras.mixed_precision.set_global_policy(original_policy)

    def test_all_nan_continual_metrics_are_warning_free(self) -> None:
        """Unavailable sparse task scores remain NaN without RuntimeWarning.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            metrics = _continual_metrics([
                [np.nan, np.nan],
                [np.nan, np.nan],
            ])
        self.assertTrue(all(np.isnan(value) for value in metrics.values()))

    def test_forgetting_is_signed_when_old_tasks_improve(self) -> None:
        """Preserve positive backward transfer as negative forgetting.

        Args:
            None.

        Returns:
            None: The standard signed metric identities are asserted in place.
        """

        metrics = _continual_metrics([
            [0.5, np.nan],
            [0.7, 0.6],
        ])
        self.assertAlmostEqual(metrics["average_forgetting"], -0.2)
        self.assertAlmostEqual(metrics["backward_transfer"], 0.2)

    def test_prebuilt_seeded_model_must_match_direct_runtime(self) -> None:
        """Reject a prebuilt stochastic model whose initialization seed differs.

        Returns:
            None: The mismatch and matching branches are asserted in place.
        """

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(2,)),
            tf.keras.layers.Dense(2),
        ])
        model.seed = 7
        _validate_supplied_model_runtime(model, 7, "float32", "test model")
        with self.assertRaisesRegex(ValueError, "requires seed 8"):
            _validate_supplied_model_runtime(
                model,
                8,
                "float32",
                "test model",
            )

    @staticmethod
    def _loader(
        indices: list[int] | tuple[int, ...],
        **kwargs: object,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Return a tiny balanced image dataset for requested original labels.

        Args:
            indices (list[int] | tuple[int, ...]): Test input named indices.
            kwargs (object): Test input named kwargs.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Result produced by the test helper.
        """

        del kwargs
        labels = np.repeat(np.asarray(indices, dtype="int32"), 6)
        values = labels.astype("float32")[:, None, None, None]
        images = np.broadcast_to(values, (len(labels), 2, 2, 1)).copy()
        return (
            images,
            labels,
            images[: len(indices) * 2],
            labels[: len(indices) * 2],
            images,
            labels,
        )

    @staticmethod
    def _template(path: Path, dropout_rate: float = 0.0) -> None:
        """Write the fixed classifier architecture expanded by the learner.

        Args:
            path (pathlib.Path): Destination Keras model path.
            dropout_rate (float): Optional stochastic trunk dropout rate.

        Returns:
            None: The compiled template is saved at ``path``.
        """

        inputs = tf.keras.Input((2, 2, 1))
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(4, activation="relu")(x)
        # Select the test action required by this condition.
        if dropout_rate > 0.0:
            x = tf.keras.layers.Dropout(
                dropout_rate,
                seed=11,
            )(x)
        outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-2),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.save(path)

    def test_schedule_modes_are_seeded_and_distinct(self) -> None:
        """Class and whole-task shuffling reproduce their intended scopes.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        class_random = resolve_continual_schedule(
            6,
            task_size=2,
            class_order_mode="random",
            seed=9,
        )
        self.assertEqual(
            class_random,
            resolve_continual_schedule(
                6,
                task_size=2,
                class_order_mode="random",
                seed=9,
            ),
        )
        task_random = resolve_continual_schedule(
            6,
            task_size=2,
            task_order_mode="random",
            seed=9,
        )
        self.assertCountEqual(task_random[1], [[0, 1], [2, 3], [4, 5]])
        self.assertTrue(all(
            group in ([0, 1], [2, 3], [4, 5])
            for group in task_random[1]
        ))

    def test_completed_run_restores_from_latest_task(self) -> None:
        """A resumed run restores model/optimizer/history without retraining.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_path = root / "template.h5"
            checkpoint_dir = root / "checkpoints"
            self._template(template_path)

            common = {
                "class_num": 2,
                "load_dataset_fn": self._loader,
                "tuned_model_path": str(template_path),
                "compile_args": {
                    "optimizer": tf.keras.optimizers.Adam(1e-2),
                    "loss": "sparse_categorical_crossentropy",
                    "metrics": ["accuracy"],
                },
                "use_loaded_opt": False,
                "batch_size": 4,
                "epochs": 1,
                "plot_results": False,
                "verbose": 0,
                "seed": 31,
                "checkpoint_dir": str(checkpoint_dir),
                "save_task_checkpoints": True,
                "return_details": True,
            }
            first = _run_continual_tasks(**common)
            restored = _run_continual_tasks(
                **common,
                resume_from=str(checkpoint_dir),
            )

            self.assertEqual(first["class_order"], [0, 1])
            self.assertEqual(first["task_classes"], [[0], [1]])
            self.assertEqual(restored["next_task_index"], 2)
            np.testing.assert_allclose(
                first["ordinary_accuracy_matrix"],
                restored["ordinary_accuracy_matrix"],
                equal_nan=True,
            )
            for expected, actual in zip(
                first["model"].get_weights(),
                restored["model"].get_weights(),
            ):
                np.testing.assert_allclose(expected, actual)

    def test_interrupted_buffer_run_matches_uninterrupted_next_updates(self) -> None:
        """Resume restores cursor, replay RNG, optimizer slots, and weights.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_path = root / "template.h5"
            self._template(template_path, dropout_rate=0.35)

            def arguments(
                checkpoint_dir: Path,
                callback: tf.keras.callbacks.Callback,
            ) -> dict[str, object]:
                """Exercise the test helper named arguments.

                Args:
                    checkpoint_dir (Path): Test input named checkpoint_dir.
                    callback (tf.keras.callbacks.Callback): Test input named callback.

                Returns:
                    dict[str, object]: Result produced by the test helper.
                """
                return {
                    "class_num": 3,
                    "load_dataset_fn": self._loader,
                    "tuned_model_path": str(template_path),
                    "compile_args": {
                        # An object (rather than a string) exercises preserved
                        # iterations and historical slot groups across heads.
                        "optimizer": tf.keras.optimizers.Adam(1e-2),
                        "loss": "sparse_categorical_crossentropy",
                        "metrics": ["accuracy"],
                    },
                    "use_loaded_opt": False,
                    "batch_size": 4,
                    "epochs": 1,
                    "use_buffer": True,
                    "buffer_kwargs": {
                        "maxlen": 8,
                        "sample_num": 2,
                        "insert_num": 2,
                    },
                    "callbacks_list": [callback],
                    "plot_results": False,
                    "verbose": 0,
                    "seed": 73,
                    "checkpoint_dir": str(checkpoint_dir),
                    "save_task_checkpoints": True,
                    "return_details": True,
                }

            uninterrupted = _run_continual_tasks(**arguments(
                root / "full",
                _InterruptOnSecondFit(enabled=False),
            ))

            interrupting = _InterruptOnSecondFit(enabled=True)
            interrupted_args = arguments(root / "resumed", interrupting)
            with self.assertRaisesRegex(
                RuntimeError,
                "intentional task interruption",
            ):
                _run_continual_tasks(**interrupted_args)

            # Task zero is durable; task one restarts from its derived seed.
            interrupting.enabled = False
            resumed = _run_continual_tasks(
                **interrupted_args,
                resume_from=str(root / "resumed"),
            )

            self.assertEqual(resumed["next_task_index"], 3)
            np.testing.assert_allclose(
                uninterrupted["ordinary_accuracy_matrix"],
                resumed["ordinary_accuracy_matrix"],
                equal_nan=True,
            )
            for expected, actual in zip(
                uninterrupted["model"].get_weights(),
                resumed["model"].get_weights(),
            ):
                np.testing.assert_allclose(expected, actual, rtol=0., atol=0.)
            expected_optimizer = uninterrupted["model"].optimizer
            actual_optimizer = resumed["model"].optimizer
            self.assertEqual(
                int(expected_optimizer.iterations.numpy()),
                int(actual_optimizer.iterations.numpy()),
            )
            for expected, actual in zip(
                expected_optimizer.variables(),
                actual_optimizer.variables(),
            ):
                np.testing.assert_allclose(
                    expected.numpy(), actual.numpy(), rtol=0., atol=0.
                )


# Select the test action required by this condition.
if __name__ == "__main__":
    unittest.main()
