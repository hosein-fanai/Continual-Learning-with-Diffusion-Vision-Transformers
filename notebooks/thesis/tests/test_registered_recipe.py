"""Check registered stream budgets and native optimizer continuity without data downloads."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from common.keras_compat import register_optimizer_variables
from common.model import _make_optimizer, get_model
from diffusion.metrics.ensemble_accuracy import EnsembleAccuracy
from semantic_consolidation.config import RouteConfig, load_route_config


TEMPLATES = Path(__file__).resolve().parents[1] / "configs"


def _planned_updates(config: RouteConfig) -> tuple[list[int], int, int]:
    """Derive joint and extra-joint budgets from balanced CIFAR training exposure.

    Each CIFAR dataset has 50,000 official training rows. The configured
    stratified holdout leaves equal class counts; match_current supplies that
    many synthetic rows for every old class. Partial batches are retained.

    Args:
        config (RouteConfig): Resolved, uncapped CIFAR recipe with equal task sizes.

    Returns:
        budgets (tuple[list[int], int, int]): Batches per task, ordinary updates,
            and total updates including the fixed extra-joint control allowance.
    """
    project, route = config.common, config.route
    continual = project.continually_learn
    rows_per_class = int(50_000 / continual.class_num * (1. - project.dataset.validation_ratio))
    class_counts = range(continual.task_size, continual.class_num + 1, continual.task_size)
    batches = [math.ceil(rows_per_class * count / project.dataset.batch_size) for count in class_counts]
    ordinary = project.training.epochs * sum(batches)
    maximum = ordinary + len(batches) * (route.acquisition_steps + route.consolidation_steps)
    return batches, ordinary, maximum


class RegisteredRecipeScheduleTests(unittest.TestCase):
    """Use the real resolved recipes and tiny native optimizer state only."""

    def tearDown(self) -> None:
        """Release Keras state after each isolated optimizer test."""
        tf.keras.backend.clear_session()

    def test_resolved_ensemble_options_construct_with_real_two_head_wrapper(self) -> None:
        """Pass registered options directly to the native ensemble constructor.

        Only transformer width and depth are reduced. Keeping both actual
        classifier heads and the registered diffusion horizon catches misspelled
        metric arguments without training or running an expensive ensemble.
        """
        for dataset in ("cifar10", "cifar100"):
            with self.subTest(dataset=dataset):
                config = load_route_config(TEMPLATES / f"{dataset}.yaml")
                project = config.common
                project.model.show_network_summary = False
                project.model.kwargs.update(dim=8, depth=1, clf_depth=1,
                                           mha_num_heads=1, clf_mha_num_heads=1)
                wrapper = get_model(project)["generative_model"]
                options = dict(project.continually_learn.ensemble_accuracy_kwargs)
                self.assertNotIn("distil_acc_coef", options)
                metric = EnsembleAccuracy(wrapper, **options)
                self.assertIs(metric.network, wrapper.network)
                self.assertEqual(metric.network.clf_cls_token_type, "new_weight")
                self.assertEqual(metric.network.clf_distil_token_type, "new_weight")
                self.assertEqual(metric.max_t, 256)
                self.assertEqual(metric.t_range_drop_rate, .5)
                self.assertEqual(metric.clf_acc_coef, .5)
                self.assertEqual(metric.clf_distil_acc_coef, .5)
                selected = metric._select_timesteps().numpy()
                self.assertEqual(len(selected), 128)
                self.assertEqual(len(np.unique(selected)), 128)
                self.assertTrue(np.all((selected >= 0) & (selected < 256)))
                tf.keras.backend.clear_session()

    def test_cosine_horizon_covers_replay_growth_and_fixed_extra_joint(self) -> None:
        """Prevent the global-row shortcut from exhausting LR before stream completion."""
        for dataset in ("cifar10", "cifar100"):
            with self.subTest(dataset=dataset):
                config = load_route_config(TEMPLATES / f"{dataset}.yaml")
                project = config.common
                self.assertEqual(project.dataset.validation_source, "split")
                self.assertEqual(project.dataset.validation_ratio, .2)
                self.assertIsNone(project.dataset.max_train_samples)
                self.assertIsNone(project.continually_learn.replay_current_examples)
                self.assertIsNone(project.continually_learn.replay_old_examples)
                self.assertEqual(project.continually_learn.replay_budget_mode, "match_current")
                self.assertEqual(project.continually_learn.class_num % project.continually_learn.task_size, 0)
                batches, ordinary, maximum = _planned_updates(config)
                self.assertEqual(project.optimizer.decay_steps, maximum)
                self.assertGreater(maximum, ordinary)
                self.assertGreater(batches[-1], batches[0])
                optimizer = _make_optimizer(project)
                self.assertIsInstance(optimizer, tf.keras.optimizers.Adam)
                self.assertIsInstance(optimizer._learning_rate, tf.keras.optimizers.schedules.CosineDecay)
                self.assertEqual(optimizer._learning_rate.decay_steps, maximum)
                task_ends = np.cumsum(batches) * project.training.epochs
                sampled_steps = [0, *task_ends.tolist(), maximum]
                actual = np.asarray([float(optimizer._learning_rate(step)) for step in sampled_steps])
                expected = project.optimizer.initial_learning_rate * .5 * (
                    1. + np.cos(np.pi * np.asarray(sampled_steps) / maximum))
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-9)
                self.assertAlmostEqual(actual[0], .005, places=8)
                self.assertTrue(np.all(np.diff(actual) < 0.))
                self.assertGreater(float(optimizer._learning_rate(ordinary)), 0.)
                self.assertEqual(float(optimizer._learning_rate(maximum)), 0.)
                self.assertEqual(float(optimizer._learning_rate(maximum + 1)), 0.)

    def test_native_growth_and_checkpoint_preserve_global_clock_and_next_update(self) -> None:
        """Keep Adam slots, LR and the next update across growth and restoration."""
        config = load_route_config(TEMPLATES / "cifar10.yaml")
        project = config.common
        optimizer = _make_optimizer(project)
        original = tf.Variable([1.], name="original")
        optimizer.apply_gradients([(tf.constant([.25]), original)])
        # Simulate crossing the obsolete global-dataset horizon, without doing 15,650 updates.
        obsolete_horizon = project.training.epochs * math.ceil(
            50_000 * (1. - project.dataset.validation_ratio) / project.dataset.batch_size)
        optimizer.iterations.assign(obsolete_horizon)
        previous_rate = float(optimizer.learning_rate)
        self.assertGreater(previous_rate, 0.)
        old_values = {(value.name, tuple(value.shape)): value.numpy().copy() for value in optimizer.variables}
        added = tf.Variable([2.], name="added")
        grown = register_optimizer_variables(optimizer, [original, added])
        self.assertIsNot(grown, optimizer)
        self.assertEqual(int(grown.iterations), obsolete_horizon)
        self.assertEqual(float(grown.learning_rate), previous_rate)
        grown_values = {(value.name, tuple(value.shape)): value.numpy() for value in grown.variables}
        for key, value in old_values.items():
            np.testing.assert_array_equal(grown_values[key], value)
        with tempfile.TemporaryDirectory(prefix="synthetic-cosine-state-") as temporary:
            path = str(Path(temporary) / "optimizer")
            tf.train.Checkpoint(optimizer=grown, original=original, added=added).write(path)
            restored = _make_optimizer(project)
            restored_original = tf.Variable([0.], name="original")
            restored_added = tf.Variable([0.], name="added")
            restored.build([restored_original, restored_added])
            tf.train.Checkpoint(optimizer=restored, original=restored_original,
                                added=restored_added).read(path).assert_consumed()
            self.assertEqual(int(restored.iterations), obsolete_horizon)
            self.assertEqual(float(restored.learning_rate), previous_rate)
            grown.apply_gradients([(tf.constant([.5]), original), (tf.constant([-.25]), added)])
            restored.apply_gradients([(tf.constant([.5]), restored_original),
                                      (tf.constant([-.25]), restored_added)])
            self.assertEqual(int(restored.iterations), obsolete_horizon + 1)
            self.assertEqual(float(restored.learning_rate), float(grown.learning_rate))
            np.testing.assert_array_equal(restored_original.numpy(), original.numpy())
            np.testing.assert_array_equal(restored_added.numpy(), added.numpy())
            for expected, actual in zip(grown.variables, restored.variables):
                np.testing.assert_array_equal(expected.numpy(), actual.numpy())


# Run only these bounded checks when this module is invoked directly.
if __name__ == "__main__":
    unittest.main()
