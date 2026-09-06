"""Integration checks for continual scheduling, precision, and deterministic recovery.

Synthetic class-coded arrays and temporary classifier templates exercise the actual task
runner. Tests compare uninterrupted and resumed weights, optimizer updates, task schedules,
and metrics; callback fixtures deliberately interrupt a later fit. Seeded variants add
dropout to test restoration of stochastic behavior.

Inputs are fixtures constructed by the test methods and their helpers. Tests return no
application result: unittest records assertion outcomes and errors. Run this module directly
or through ``python -m unittest`` discovery. Importing it defines fixtures and cases; it
does not itself start a test run.
"""

from __future__ import annotations

import tempfile
import unittest
import warnings

from pathlib import Path
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import resolve_continual_schedule
from common.dataloader import load_mnist
from common.learner import (
    _continual_metrics,
    _prepare_diffusion_x,
    _recovery_descriptor,
    _reset_task_random_streams,
    _run_continual_tasks,
    _validate_supplied_model_runtime,
)
from diffusion import DiffusionModel, DiffusionTransformer


class _InterruptOnSecondFit(tf.keras.callbacks.Callback):
    """Raise once at task two, while retaining a stable descriptor type.

    This fixture implements only the interface required by its surrounding regression tests.
    Construction and mutable state are described by ``__init__``; it does not provide a
    general production replacement.

    Args:
        enabled (bool): True enables a simulated failure at the second on_train_begin call;
            False counts fits without raising.

    Returns:
        _InterruptOnSecondFit: A new local test fixture with independent instance state.
    """

    def __init__(self, enabled: bool) -> None:
        """Create a task-boundary interruption callback with a reset fit counter.

        The stable callback type is retained in checkpoint descriptors; tests can disable
        its failure flag before resuming.

        Args:
            enabled (bool): True enables a simulated failure at the second on_train_begin
                call; False counts fits without raising.

        Returns:
            None: The fixture state is initialized in place.
        """
        super().__init__()
        self.enabled = enabled
        self.fit_count = 0

    def on_train_begin(
        self,
        logs: dict[str, float] | None = None,
    ) -> None:
        """Count a new fit and optionally interrupt the second task.

        Args:
            logs (dict[str, float] | None): Optional Keras training-start metric mapping;
                ignored by this fixture. Defaults to ``None``.

        Returns:
            None: fit_count is incremented; normal execution continues except at the
            selected interruption.

        Raises:
            RuntimeError: enabled is True and this is the second fit start.
        """
        del logs
        self.fit_count += 1
        # Interrupt task two only in the simulated-failure run.
        if self.enabled and self.fit_count == 2:
            raise RuntimeError("intentional task interruption")


class ContinualIntegrationTests(unittest.TestCase):
    """Exercise orchestration seams that unit helpers cannot cover.

    The unittest runner executes the selected test method with its local fixtures;
    individual methods describe the configurations and failure cases they exercise. There is
    no application model or experiment result returned by constructing this test case.

    Args:
        methodName (str): Test method selected by unittest. Defaults to ``"runTest"``;
            discovery supplies each named ``test_*`` method.

    Attributes:
        _testMethodName (str): Selected method name maintained by unittest.
    """

    def test_task_rng_reset_uses_stable_layer_traversal(self) -> None:
        """Avoid TensorFlow weak-dictionary submodule flattening.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        class Wrapper(tf.keras.Model):
            """Expose a model with a deliberately fragile submodule property.

            This fixture implements only the interface required by its surrounding
            regression tests. Construction and mutable state are described by ``__init__``;
            it does not provide a general production replacement.

            Args:
                None.

            Returns:
                Wrapper: A new local test fixture with independent instance state.
            """

            def __init__(self) -> None:
                """Build a small stochastic network for traversal testing.

                The network contains Dropout(rate=0.1, seed=3) followed by Dense(2). Its
                intentionally failing submodules property forces the runtime helper to use
                supported layer/tracking traversal.

                Args:
                    None.

                Returns:
                    None: The fixture state is initialized in place.
                """

                super().__init__()
                self.network = tf.keras.Sequential([
                    tf.keras.layers.Dropout(0.1, seed=3),
                    tf.keras.layers.Dense(2),
                ])

            def call(self, inputs: tf.Tensor) -> tf.Tensor:
                """Apply the small wrapped network to its input tensor.

                This fixture forwards inputs directly; the enclosing Keras call context
                controls training behavior.

                Args:
                    inputs (tf.Tensor): Tensor with a batch axis and a final feature axis
                        compatible with the Dense layer.

                Returns:
                    tf.Tensor: Network output with two features per example.
                """

                return self.network(inputs)

            @property
            def submodules(self) -> tuple[object, ...]:
                """Reject access to the fragile TensorFlow submodule enumeration property.

                Args:
                    None.

                Returns:
                    tuple[object, ...]: Declared interface only; this fixture always raises
                    instead of returning modules.

                Raises:
                    ValueError: Always raised to expose traversal code that uses this
                    property.
                """

                raise ValueError("fragile TensorFlow flattening")

        wrapper = Wrapper()
        wrapper(tf.ones((1, 2)), training=True)
        _reset_task_random_streams(wrapper, 19)
        self.assertIsNone(wrapper.train_function)

    def test_diffusion_preparation_uses_policy_variable_dtype(self) -> None:
        """Diffusion arrays use stable variable precision under each policy.

        Args:
            None.

        Returns:
            None: Assertions verify the stated regression; failures are reported
                to the unittest runner.
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
            grayscale = _prepare_diffusion_x(
                np.zeros((2, 4, 4), dtype="float32"),
                0.,
                1.,
            )
            self.assertEqual(grayscale.shape, (2, 4, 4, 1))

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
            None: Assertions verify the stated regression; failures are reported
                to the unittest runner.
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

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(2,)),
            tf.keras.layers.Dense(2),
        ])
        model.seed = 7
        _validate_supplied_model_runtime(model, 7, "test model")
        with self.assertRaisesRegex(ValueError, "requires seed 8"):
            _validate_supplied_model_runtime(
                model,
                8,
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
        """Return class-coded synthetic arrays for the requested original labels.

        The fixture ignores preprocessing/feature options: every image is a float32 2x2x1
        array filled with its int32 class ID. Six examples are produced per requested label.
        Validation uses the first 2 * len(indices) rows, so it need not contain every
        requested class; this exercises missing task observations.

        Args:
            indices (list[int] | tuple[int, ...]): Original class IDs in the order used to
                construct repeated labels and images.
            **kwargs (object): Loader-compatible keyword options, accepted and ignored. No
                options are supplied by default.

        Returns:
            tuple[np.ndarray, ...]: (x_train, y_train, x_val, y_val, x_test, y_test). Train
            and test share the complete arrays; validation contains prefix views. Sparse
            label arrays have shape (N,), and images have shape (N, 2, 2, 1).
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
            dropout_rate (float): Optional stochastic trunk dropout rate. Defaults to
                ``0.0``.

        Returns:
            None: The compiled template is saved at ``path``.
        """

        inputs = tf.keras.Input((2, 2, 1))
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(4, activation="relu")(x)
        # Include stochastic trunk behavior when dropout is requested.
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
            None: Assertions verify the stated regression; failures are reported
                to the unittest runner.
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
        self.assertEqual(
            resolve_continual_schedule(
                6,
                task_size=2,
                class_order_mode="random",
                seed=1.9,
            ),
            resolve_continual_schedule(
                6,
                task_size=2,
                class_order_mode="random",
                seed=1,
            ),
        )
        self.assertEqual(
            resolve_continual_schedule(
                6,
                task_size=2,
                class_order_mode="random",
                seed="7",
            ),
            resolve_continual_schedule(
                6,
                task_size=2,
                class_order_mode="random",
                seed=7,
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

    def test_schedule_starts_at_task_size_and_requires_a_transition(self) -> None:
        """Resolve singleton/pair starts and reject a one-task experiment.

        Args:
            None.

        Returns:
            None: Assertions verify the stated regression; failures are reported
                to the unittest runner.
        """
        self.assertEqual(
            resolve_continual_schedule(4, task_size=1)[1],
            [[0], [1], [2], [3]],
        )
        self.assertEqual(
            resolve_continual_schedule(5, task_size=2)[1],
            [[0, 1], [2, 3], [4]],
        )
        randomized = resolve_continual_schedule(
            5,
            task_size=2,
            task_order_mode="random",
            seed=4,
        )
        self.assertEqual(
            randomized,
            resolve_continual_schedule(
                5,
                task_size=2,
                task_order_mode="random",
                seed=4,
            ),
        )
        self.assertEqual(len(randomized[1][0]), 2)
        self.assertCountEqual(randomized[1], [[0, 1], [2, 3], [4]])
        with self.assertRaisesRegex(ValueError, "two task groups"):
            resolve_continual_schedule(2, task_size=2)

    def test_completed_run_restores_from_latest_task(self) -> None:
        """A resumed run restores model/optimizer/history without retraining.

        Args:
            None.

        Returns:
            None: Assertions verify the stated regression; failures are reported
                to the unittest runner.
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
                "baseline": "cumulative",
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

    @staticmethod
    def _generator(noise_distillation: bool = False) -> DiffusionModel:
        """Build the same seeded dynamic generator for each recovery or cache run.

        Args:
            noise_distillation (bool): Enable deferred previous-task noise distillation.
                Defaults to False for an ordinary replay generator.

        Returns:
            DiffusionModel: A compiled two-pixel generator with independent state.
        """

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(31)
        network = DiffusionTransformer(
            num_classes=None, use_cfg=True, timesteps=4,
            image_size=2, channels=1, patch_size=1,
            dim=4, depth=1, mha_num_heads=1,
            vit_block_mlp_ratio=1., seed=31,
        )
        wrapper = DiffusionModel(
            network=network, use_ema=True, test_steps=2,
            scheduler_name="linear", seed=31,
            defer_teacher=noise_distillation,
            # Noise KD needs a positive coefficient; ordinary replay leaves it disabled.
            noise_distil_loss_coef=1. if noise_distillation else 0.,
        )
        wrapper.compile(optimizer="adam", loss="mse", run_eagerly=True)
        return wrapper

    def test_diffusion_recovery_restores_external_classifier(self) -> None:
        """Resume both the diffusion generator and its expanding external classifier.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Restored weights and the next task update match uninterrupted training.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_path = root / "template.h5"
            self._template(template_path)
            arguments = {
                "class_num": 3,
                "task_size": 2,
                "load_dataset_fn": self._loader,
                "load_dataset_fn_kwargs": {"preprocess": "min-max"},
                "tuned_model_path": str(template_path),
                "compile_args": {
                    "optimizer": tf.keras.optimizers.Adam(1e-2),
                    "loss": "sparse_categorical_crossentropy",
                    "metrics": ["accuracy"],
                },
                "remove_prev_classes": False,
                "use_generative_replay": False,
                "generative_model_kwargs": {"train_num": 4},
                "batch_size": 4,
                "epochs": 1,
                "callback_patience": 0,
                "plot_results": False,
                "verbose": 0,
                "seed": 31,
                "checkpoint_dir": str(root / "checkpoints"),
            }
            uninterrupted = _run_continual_tasks(
                **arguments, generative_model=self._generator(), save_task_checkpoints=True,
            )
            restored = _run_continual_tasks(
                **arguments, generative_model=self._generator(),
                resume_from=str(root / "checkpoints" / "task-0000"),
            )

            self.assertEqual(restored["model"].output_shape[-1], 3)
            for role in ("model", "generative_model"):
                for expected, actual in zip(
                    uninterrupted[role].get_weights(), restored[role].get_weights(),
                ):
                    np.testing.assert_allclose(expected, actual, rtol=0., atol=0.)
            np.testing.assert_allclose(
                uninterrupted["ordinary_accuracy_matrix"],
                restored["ordinary_accuracy_matrix"], equal_nan=True,
            )

    def test_fixed_pixel_modes_preserve_continual_diffusion_coordinates(self) -> None:
        """Map both fixed public scales to the same diffusion training values.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Real loader and learner pipelines retain public pixel scaling.
        """

        labels = np.repeat(np.arange(3, dtype="uint8"), 4)
        pixels = np.broadcast_to(
            ((labels + 1) * 64)[:, None, None], (len(labels), 2, 2),
        ).copy()
        with tempfile.TemporaryDirectory() as directory:
            template_path = Path(directory) / "template.h5"
            self._template(template_path)
            for mode, expected_scale in (
                ("fixed-min-max", {"data_min": 0., "data_range": 1.}),
                ("fixed-standardize", {"data_min": -1., "data_range": 2.}),
            ):
                with self.subTest(mode=mode), patch(
                    "tensorflow.keras.datasets.mnist.load_data",
                    return_value=((pixels, labels), (pixels, labels)),
                ), patch("common.train.train_model", return_value={}) as fit, patch(
                    "common.train.report", return_value={},
                ):
                    details = _run_continual_tasks(
                        class_num=3, task_size=2, load_dataset_fn=load_mnist,
                        load_dataset_fn_kwargs={"preprocess": mode, "validation_ratio": 0.},
                        tuned_model_path=str(template_path),
                        generative_model=self._generator(),
                        generative_model_kwargs={"train_num": -1},
                        use_generative_replay=False, remove_prev_classes=False,
                        batch_size=4, epochs=1, callback_patience=0,
                        plot_results=False, verbose=0, seed=31,
                    )
                    # Inspect the generator phase rather than the separate classifier fit.
                    generator_fit = next(
                        call for call in fit.call_args_list
                        if isinstance(call.args[1], DiffusionModel)
                    )
                    trained_pixels = np.concatenate([
                        batch[0] for batch in generator_fit.args[2].as_numpy_iterator()
                    ])
                self.assertEqual(
                    details["run_descriptor"]["data"]["diffusion_scale"], expected_scale,
                )
                np.testing.assert_allclose(
                    np.sort(trained_pixels.reshape(-1)),
                    np.sort((pixels[labels < 2].astype("float32") / 255. * 2. - 1.).reshape(-1)),
                    rtol=1e-6,
                )

    def test_noise_distillation_supports_classifier_free_replay_diagnostics(self) -> None:
        """Keep generator-teacher diagnostics independent of classifier-only APIs.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Noise KD trains and replay diagnostics omit unavailable class scores.
        """

        with tempfile.TemporaryDirectory() as directory:
            template_path = Path(directory) / "template.h5"
            self._template(template_path)
            details = _run_continual_tasks(
                class_num=3, task_size=2, load_dataset_fn=self._loader,
                load_dataset_fn_kwargs={"preprocess": "min-max"},
                tuned_model_path=str(template_path),
                compile_args={
                    "optimizer": "adam", "loss": "sparse_categorical_crossentropy",
                    "metrics": ["accuracy"],
                },
                generative_model=self._generator(noise_distillation=True),
                generative_model_kwargs={"train_num": -1, "samples_per_class": 1},
                use_distillation=True, mechanistic_metrics=True,
                batch_size=4, epochs=1, callback_patience=0,
                plot_results=False, verbose=0, seed=31,
            )
        diagnostics = details["task_mechanistic_metrics"][1]
        self.assertEqual(diagnostics["sample_count"], 2)
        self.assertNotIn("label_consistency", diagnostics)
        self.assertNotIn("representation", diagnostics)
        self.assertGreater(
            details["generative_histories"][1]["noise_distil_loss"][-1], 0.,
        )

    def test_replay_cache_requires_identical_learned_generator(self) -> None:
        """Reuse identical replay sources and reject changed pre-replay training.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Matching runs read the pool without sampling; changed weights miss it.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_path = root / "template.h5"
            self._template(template_path)
            arguments = {
                "class_num": 3,
                "task_size": 2,
                "load_dataset_fn": self._loader,
                "load_dataset_fn_kwargs": {"preprocess": "min-max"},
                "tuned_model_path": str(template_path),
                "compile_args": {
                    "optimizer": "adam", "loss": "sparse_categorical_crossentropy",
                    "metrics": ["accuracy"],
                },
                "generative_model_kwargs": {"train_num": 4, "samples_per_class": 1},
                "replay_cache_dir": str(root / "cache"),
                "batch_size": 4, "callback_patience": 0,
                "plot_results": False, "verbose": 0, "seed": 31,
            }
            generated = _run_continual_tasks(
                **arguments, generative_model=self._generator(),
                replay_cache_mode="write", epochs=1,
            )
            reader = self._generator()
            with patch.object(reader, "sample", side_effect=AssertionError(
                "An identical learned generator must use its cached candidates."
            )):
                cached = _run_continual_tasks(
                    **arguments, generative_model=reader,
                    replay_cache_mode="read", epochs=1,
                )
            self.assertEqual(
                generated["task_resource_metrics"][1]["replay"]["cache_path"],
                cached["task_resource_metrics"][1]["replay"]["cache_path"],
            )
            with self.assertRaises(FileNotFoundError):
                _run_continual_tasks(
                    **arguments, generative_model=self._generator(),
                    replay_cache_mode="read", epochs=2,
                )

    def test_interrupted_buffer_run_matches_uninterrupted_next_updates(self) -> None:
        """Resume restores cursor, replay RNG, optimizer slots, and weights.

        Args:
            None.

        Returns:
            None: Assertions verify the stated regression; failures are reported
                to the unittest runner.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_path = root / "template.h5"
            self._template(template_path, dropout_rate=0.35)

            def arguments(
                checkpoint_dir: Path,
            ) -> dict[str, object]:
                """Build matching fresh arguments for uninterrupted and resumed buffer runs.

                The closure reads the temporary classifier template path and class-coded
                loader. Only the checkpoint destination differs across the compared runs.

                Args:
                    checkpoint_dir (Path): Destination for committed task checkpoints in
                        this run.

                Returns:
                    dict[str, object]: Three-class, one-epoch, seeded buffer-run options,
                    including a fresh Adam optimizer and detailed checkpoint output.
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
                    "plot_results": False,
                    "verbose": 0,
                    "seed": 73,
                    "checkpoint_dir": str(checkpoint_dir),
                    "save_task_checkpoints": True,
                    "return_details": True,
                }

            uninterrupted = _run_continual_tasks(**arguments(root / "full"))
            interrupted_args = arguments(root / "resumed")
            original_fit = tf.keras.Model.fit
            fit_count = 0

            def interrupted_fit(model: tf.keras.Model, *args: object, **kwargs: object) -> object:
                """Inject an external failure without changing experiment callbacks."""
                nonlocal fit_count
                fit_count += 1
                # The first committed task remains durable while the next fit is interrupted.
                if fit_count == 2:
                    raise RuntimeError("intentional task interruption")
                return original_fit(model, *args, **kwargs)

            with patch.object(tf.keras.Model, "fit", new=interrupted_fit):
                with self.assertRaisesRegex(RuntimeError, "intentional task interruption"):
                    _run_continual_tasks(**interrupted_args)

            resumed = _run_continual_tasks(
                **interrupted_args, resume_from=str(root / "resumed"),
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


    def test_recovery_descriptor_preserves_exact_caller_floats(self) -> None:
        """Keep scientifically distinct run settings in distinct identities.

        Args:
            None.

        Returns:
            None: Assertions verify the stated regression; failures are reported
                to the unittest runner.
        """
        first = _recovery_descriptor({"coefficient": 0.123456741})
        second = _recovery_descriptor({"coefficient": 0.123456749})
        self.assertNotEqual(first, second)


# Run this module's tests when executed directly.
if __name__ == "__main__":
    unittest.main()
