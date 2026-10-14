"""Exercise fresh task optimizers through real small joint-diffusion fits."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np
import tensorflow as tf

from common.learner import _reset_task_optimizer, _run_continual_tasks
from common.tests.test_learner_verified_repairs import image_loader
from common.tests.test_wrapper_verified_repairs import _network
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier
from diffusion.models.wrapper.diffusion_classifier_v2 import DiffusionClassifierV2


class _TaskObservation(tf.keras.callbacks.Callback):
    """Observe optimizer and learned-weight continuity without replacing updates."""

    def __init__(self) -> None:
        """Keep complete references so optimizer identities cannot be recycled."""

        super().__init__()
        self.starts = []
        self.ends = []

    def _snapshot(self) -> dict[str, object]:
        """Read the active optimizer and the actual shared trainable variables."""

        return {
            "optimizer": self.model.optimizer, 
            "iterations": int(self.model.optimizer.iterations.numpy()), 
            "learning_rate": float(self.model.optimizer.learning_rate.numpy()), 
            "optimizer_config": tf.keras.optimizers.serialize(self.model.optimizer), 
            "class_count": len(self.model.seen_classes), 
            "moments": {
                (variable.name, tuple(variable.shape)): variable.numpy().copy()
                for variable in self.model.optimizer.variables
                if variable.name.endswith(("_momentum", "_velocity"))
            }, 
            "weights": [variable.numpy().copy() for variable in self.model.network.trainable_weights]
        }

    def on_train_begin(self, logs: dict | None = None) -> None:
        """Capture state after task preparation and before the first batch."""

        del logs
        self.starts.append(self._snapshot())

    def on_train_end(self, logs: dict | None = None) -> None:
        """Capture the completed task before the next task can reset its optimizer."""

        del logs
        self.ends.append(self._snapshot())


class TaskOptimizerResetTests(unittest.TestCase):
    """Check task-local budgets, state preservation, and unchanged default behavior."""

    def tearDown(self) -> None:
        """Release only this isolated test process's Keras state."""

        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")

    @staticmethod
    def _model(version: int = 1, classes: int | None = None, eager: bool = True) -> DiffusionClassifier:
        """Compile a tiny joint DiT with both objectives active and optional dynamic growth."""

        wrapper = DiffusionClassifier if version == 1 else DiffusionClassifierV2
        model = wrapper(
            network=_network(classes=classes, distil=False), 
            use_ema=False, test_network_name="raw", scheduler_name="linear", 
            test_steps=2, train_noisified_max_timesteps=None, 
            test_noisified_max_timesteps=None, clf_distil_loss_coef=0., 
            clf_loss_coef=1., noise_loss_coef=1., seed=37
        )
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=.01, decay_steps=999, alpha=.1
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=schedule, beta_1=.8, beta_2=.95, epsilon=1e-6, 
                amsgrad=True, clipnorm=1.
            ), loss="mse", 
            evaluation_loss="mae", run_eagerly=eager, jit_compile=False
        )
        return model

    @staticmethod
    def _loader(indices: list[int], **kwargs: object) -> tuple:
        """Return raw external pixels so the diffusion wrapper owns normalization."""

        arrays = list(image_loader(indices, **kwargs))
        for index in (0, 2, 4):
            arrays[index] = (arrays[index] + np.float32(1.)) * np.float32(127.5)
        return tuple(arrays)

    @staticmethod
    def _arguments(model: DiffusionClassifier, observation: _TaskObservation) -> dict:
        """Expose cumulative pools of eight and sixteen rows, retaining batch tails."""

        return {
            "class_num": 4, "task_size": 2, 
            "task_groups": [[0, 1], [2, 3]], 
            "load_dataset_fn": TaskOptimizerResetTests._loader, 
            "load_dataset_fn_kwargs": {"preprocess": None, "onehot_labels": False}, 
            "remove_prev_classes": False, "generative_model": model, 
            "generative_model_kwargs": {"train_num": -1, "samples_per_class": 2}, 
            "use_generative_model_classifier": True, "use_generative_replay": False, 
            "use_distillation": False, "batch_size": 5, "epochs": 2, 
            "generative_callbacks_list": [observation], "callback_patience": None, 
            "plot_results": False, "show_generated_images": False, 
            "show_network_summary": False, "experiment_phase": "development", 
            "return_details": True, "verbose": 0, "seed": 37
        }

    def test_real_tasks_restart_optimizer_and_preserve_learned_weights(self) -> None:
        """Use each final training pool to set its own horizon and retain prior learning."""

        model = self._model()
        observation = _TaskObservation()
        arguments = self._arguments(model, observation)
        arguments["reinstantiate_optimizer"] = True
        result = _run_continual_tasks(**arguments)
        self.assertEqual(
            [row["optimizer_config"]["config"]["learning_rate"]["config"]["decay_steps"]
             for row in observation.starts], [4, 8]
        )
        self.assertEqual([row["iterations"] for row in observation.starts], [0, 0])
        self.assertEqual([row["iterations"] for row in observation.ends], [4, 8])
        self.assertIsNot(observation.starts[0]["optimizer"], observation.starts[1]["optimizer"])
        for index in range(2):
            self.assertAlmostEqual(observation.starts[index]["learning_rate"], .01, places=7)
            self.assertAlmostEqual(observation.ends[index]["learning_rate"], .001, places=7)
            self.assertTrue(all(
                not np.any(value) for value in observation.starts[index]["moments"].values()
            ))
        previous = observation.ends[0]["weights"]
        following = observation.starts[1]["weights"]
        self.assertEqual(len(previous), len(following))
        shared = [(expected, actual) for expected, actual in zip(previous, following)
                  if expected.shape == actual.shape]
        self.assertGreater(len(shared), 0)
        for expected, actual in shared:
            np.testing.assert_array_equal(actual, expected)
        self.assertTrue(any(
            not np.array_equal(before, after)
            for before, after in zip(observation.starts[0]["weights"], previous)
        ))
        resources = result["task_resource_metrics"]
        self.assertEqual([row["optimizer_reset"]["task_index"] for row in resources], [0, 1])
        self.assertEqual([row["optimizer_reset"]["steps_per_epoch"] for row in resources], [2, 4])
        self.assertEqual([row["optimizer_reset"]["epochs"] for row in resources], [2, 2])
        self.assertEqual([row["training_examples_total"] for row in resources], [8, 16])
        self.assertEqual([row["optimizer_reset"]["maximum_updates"] for row in resources], [4, 8])
        self.assertEqual([row["optimizer_updates"]["replay_optimizer"] for row in resources], [4, 8])
        for row in resources:
            self.assertTrue(all(value >= 0 for value in row["optimizer_updates"].values()))

    def test_graph_training_restarts_after_dynamic_vocabulary_growth(self) -> None:
        """Retrace actual graph fits with new variables and a fresh optimizer at each task."""

        model = self._model(classes=None, eager=False)
        observation = _TaskObservation()
        arguments = self._arguments(model, observation)
        arguments.update(epochs=1, reinstantiate_optimizer=True)
        result = _run_continual_tasks(**arguments)
        self.assertEqual([row["class_count"] for row in observation.starts], [2, 4])
        self.assertEqual([row["iterations"] for row in observation.starts], [0, 0])
        self.assertEqual([row["iterations"] for row in observation.ends], [2, 4])
        self.assertEqual(
            [row["optimizer_config"]["config"]["learning_rate"]["config"]["decay_steps"]
             for row in observation.starts], [2, 4]
        )
        self.assertIsNot(observation.starts[0]["optimizer"], observation.starts[1]["optimizer"])
        self.assertEqual(
            [row["optimizer_updates"]["replay_optimizer"] for row in result["task_resource_metrics"]], 
            [2, 4]
        )
        for row in observation.ends:
            self.assertTrue(all(np.isfinite(value).all() for value in row["weights"]))

    def test_default_keeps_optimizer_state_across_tasks(self) -> None:
        """Retain optimizer counters and the original schedule when reset is not requested."""

        model = self._model()
        observation = _TaskObservation()
        result = _run_continual_tasks(**self._arguments(model, observation))
        self.assertEqual([row["iterations"] for row in observation.starts], [0, 4])
        self.assertEqual([row["iterations"] for row in observation.ends], [4, 12])
        self.assertEqual([row["class_count"] for row in observation.starts], [2, 4])
        previous = observation.ends[0]["moments"]
        following = observation.starts[1]["moments"]
        shared = previous.keys() & following.keys()
        self.assertGreater(len(shared), 0)
        self.assertTrue(any(np.any(previous[key] != 0) for key in shared))
        for key in shared:
            np.testing.assert_array_equal(following[key], previous[key])
        for row in result["task_resource_metrics"]:
            self.assertNotIn("optimizer_reset", row)
        for row in observation.starts:
            self.assertEqual(
                row["optimizer_config"]["config"]["learning_rate"]["config"]["decay_steps"], 999
            )

    def test_explicit_fit_batch_budgets_control_each_horizon(self) -> None:
        """Honor both supported step overrides rather than the unmodified pool cardinality."""

        cases = (
            ({"optimizer_steps_per_epoch": 3}, 3), 
            ({"fit_kwargs": {"steps_per_epoch": 1}}, 1), 
            ({"generative_model_kwargs": {"train_num": 7, "samples_per_class": 2}}, 2)
        )
        for override, steps in cases:
            with self.subTest(override=override):
                observation = _TaskObservation()
                arguments = self._arguments(self._model(), observation)
                arguments.update(override)
                arguments["reinstantiate_optimizer"] = True
                result = _run_continual_tasks(**arguments)
                self.assertEqual(
                    [row["optimizer_reset"]["steps_per_epoch"] for row in result["task_resource_metrics"]], 
                    [steps, steps]
                )
                self.assertEqual(
                    [row["optimizer_config"]["config"]["learning_rate"]["config"]["decay_steps"]
                     for row in observation.starts], [2 * steps, 2 * steps]
                )
                self.assertEqual([row["iterations"] for row in observation.ends], [2 * steps, 2 * steps])
                self.assertEqual(
                    [row["optimizer_updates"]["replay_optimizer"] for row in result["task_resource_metrics"]], 
                    [2 * steps, 2 * steps]
                )

    def test_reset_preserves_compile_and_optimizer_configuration(self) -> None:
        """Clone optimizer settings while retaining model weights and the compile contract."""

        model = self._model()
        before = model.get_compile_config()
        weights = [variable.numpy().copy() for variable in model.network.trainable_weights]
        previous_optimizer = model.optimizer
        previous_optimizer.iterations.assign(9)
        metadata = _reset_task_optimizer(model, 1, 3, 2)
        after = model.get_compile_config()
        self.assertEqual(
            {key: value for key, value in before.items() if key != "optimizer"}, 
            {key: value for key, value in after.items() if key != "optimizer"}
        )
        previous_config = before["optimizer"]["config"]
        following_config = after["optimizer"]["config"]
        self.assertEqual(
            {key: value for key, value in previous_config.items() if key != "learning_rate"}, 
            {key: value for key, value in following_config.items() if key != "learning_rate"}
        )
        previous_schedule = previous_config["learning_rate"]["config"]
        following_schedule = following_config["learning_rate"]["config"]
        self.assertEqual(
            {key: value for key, value in previous_schedule.items() if key != "decay_steps"}, 
            {key: value for key, value in following_schedule.items() if key != "decay_steps"}
        )
        self.assertEqual(previous_schedule["decay_steps"], 999)
        self.assertEqual(following_schedule["decay_steps"], 6)
        self.assertIsNot(model.optimizer, previous_optimizer)
        self.assertEqual(metadata["maximum_updates"], 6)
        self.assertEqual(int(model.optimizer.iterations.numpy()), 0)
        for expected, actual in zip(weights, model.network.trainable_weights):
            np.testing.assert_array_equal(actual.numpy(), expected)

    def test_constant_learning_rate_is_preserved(self) -> None:
        """Reset Adam state without inventing a schedule for a constant learning rate."""

        model = self._model()
        model.compile(optimizer=tf.keras.optimizers.Adam(.003, beta_1=.7), loss="mse")
        previous = model.optimizer
        before = tf.keras.optimizers.serialize(previous)
        previous.iterations.assign(17)
        _reset_task_optimizer(model, 0, 3, 2)
        self.assertIsNot(model.optimizer, previous)
        self.assertEqual(int(model.optimizer.iterations.numpy()), 0)
        self.assertEqual(tf.keras.optimizers.serialize(model.optimizer), before)
        self.assertAlmostEqual(float(model.optimizer.learning_rate.numpy()), .003, places=7)

    def test_other_schedules_are_cloned_without_retargeting(self) -> None:
        """Preserve an exponential schedule's declared parameters while resetting its clock."""

        model = self._model()
        schedule = tf.keras.optimizers.schedules.ExponentialDecay(.01, 10, .9)
        model.compile(optimizer=tf.keras.optimizers.Adam(schedule), loss="mse")
        previous = model.optimizer
        before = tf.keras.optimizers.serialize(previous)
        previous.iterations.assign(17)
        _reset_task_optimizer(model, 0, 3, 2)
        self.assertIsNot(model.optimizer, previous)
        self.assertEqual(int(model.optimizer.iterations.numpy()), 0)
        self.assertEqual(tf.keras.optimizers.serialize(model.optimizer), before)
        self.assertAlmostEqual(float(model.optimizer.learning_rate.numpy()), .01, places=7)

    def test_loss_scale_wrapper_preserves_settings_and_retargets_inner_cosine(self) -> None:
        """Clone mixed-precision state and adjust the cosine schedule owned by inner Adam."""

        model = self._model()
        wrapped = tf.keras.mixed_precision.LossScaleOptimizer(
            model.optimizer, initial_scale=128., dynamic_growth_steps=20
        )
        model.compile(optimizer=wrapped, loss="mse")
        before = tf.keras.optimizers.serialize(model.optimizer)
        wrapped.iterations.assign(17)
        _reset_task_optimizer(model, 0, 3, 2)
        self.assertIsNot(model.optimizer, wrapped)
        self.assertIsInstance(model.optimizer, tf.keras.mixed_precision.LossScaleOptimizer)
        self.assertIsNot(model.optimizer.inner_optimizer, wrapped.inner_optimizer)
        self.assertEqual(int(model.optimizer.iterations.numpy()), 0)
        before["config"]["inner_optimizer"]["config"]["learning_rate"]["config"]["decay_steps"] = 6
        self.assertEqual(tf.keras.optimizers.serialize(model.optimizer), before)

    def test_cosine_warmup_is_inside_the_total_task_budget(self) -> None:
        """Keep active warmup duration and finish cosine decay at the final task update."""

        model = self._model()
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=.001, decay_steps=99, alpha=.2, 
            warmup_target=.01, warmup_steps=2
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(schedule), loss="mse")
        _reset_task_optimizer(model, 0, 3, 2)
        configuration = tf.keras.optimizers.serialize(model.optimizer)["config"]["learning_rate"]
        restored = tf.keras.optimizers.schedules.deserialize(configuration)
        self.assertEqual(restored.decay_steps, 4)
        self.assertEqual(restored.warmup_steps, 2)
        self.assertAlmostEqual(float(restored(0).numpy()), .001, places=7)
        self.assertAlmostEqual(float(restored(2).numpy()), .01, places=7)
        self.assertAlmostEqual(float(restored(6).numpy()), .002, places=7)

    def test_inactive_warmup_does_not_shorten_cosine_decay(self) -> None:
        """Ignore warmup_steps in the horizon when CosineDecay has no warmup target."""

        model = self._model()
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=.01, decay_steps=99, alpha=.1, 
            warmup_target=None, warmup_steps=20
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(schedule), loss="mse")
        _reset_task_optimizer(model, 0, 3, 2)
        configuration = tf.keras.optimizers.serialize(model.optimizer)["config"]["learning_rate"]
        restored = tf.keras.optimizers.schedules.deserialize(configuration)
        self.assertEqual(restored.decay_steps, 6)
        self.assertEqual(restored.warmup_steps, 20)
        self.assertAlmostEqual(float(restored(0).numpy()), .01, places=7)
        self.assertAlmostEqual(float(restored(6).numpy()), .001, places=7)

    def test_impossible_warmup_budget_fails_without_mutating_optimizer(self) -> None:
        """Reject a task with no positive cosine-decay interval after active warmup."""

        for warmup_steps in (6, 7):
            with self.subTest(warmup_steps=warmup_steps):
                model = self._model()
                schedule = tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=.001, decay_steps=99, 
                    warmup_target=.01, warmup_steps=warmup_steps
                )
                model.compile(optimizer=tf.keras.optimizers.Adam(schedule), loss="mse")
                previous = model.optimizer
                before = model.get_compile_config()
                with self.assertRaisesRegex(ValueError, "warmup"):
                    _reset_task_optimizer(model, 0, 3, 2)
                self.assertIs(model.optimizer, previous)
                self.assertEqual(model.get_compile_config(), before)

    def test_unsupported_modes_fail_before_loading_data(self) -> None:
        """Reject incompatible execution and recovery contracts before starting any work."""

        cases = (
            {"fit_method": "fit_progressively"}, 
            {"save_task_checkpoints": True}, 
            {"resume_from": "unused-checkpoint"}, 
            {"generative_model": self._model(version=2)}
        )
        for override in cases:
            with self.subTest(override=override):
                loader = Mock(side_effect=AssertionError("Unsupported reset mode loaded data"))
                arguments = self._arguments(self._model(), _TaskObservation())
                arguments.update(
                    load_dataset_fn=loader, reinstantiate_optimizer=True
                )
                arguments.update(override)
                with self.assertRaises((TypeError, ValueError)):
                    _run_continual_tasks(**arguments)
                loader.assert_not_called()


# Allow focused execution in an isolated remote process.
if __name__ == "__main__":
    unittest.main()
