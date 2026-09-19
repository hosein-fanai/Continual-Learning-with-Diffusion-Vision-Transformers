"""Mathematical selection and actual V1 optimizer restoration regressions."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from semantic_consolidation.replay_selection import (
    DriftReplaySelector,
    ReplaySelectionSettings,
    padded_jensen_shannon,
    prepare_virtual_current_batch,
    probability_rows,
    virtual_update_interference,
)


def _scored(labels: np.ndarray, quality: np.ndarray | None = None) -> dict:
    """Build explicit candidate score fixtures independent of model inference."""

    count = len(labels)
    quality = np.full(count, .75) if quality is None else np.asarray(quality)
    width = int(labels.max()) + 1 if count else 2
    probabilities = np.full((count, width), 1. / width)
    return {
        "drift": np.arange(count, dtype="float64") / max(count, 1),
        "new_class_invasion": np.zeros(count), "confidence": probabilities.max(axis=1),
        "label_surprisal": -np.log(np.maximum(quality, 1e-12)),
        "teacher_label_probability": quality, "teacher_probabilities": probabilities,
        "diagnostics": {},
    }


class JensenShannonTests(unittest.TestCase):
    """Check probability normalization, support padding, and view reductions."""

    def test_zero_disjoint_and_new_class_invasion(self) -> None:
        """Preserve zero mass and detect probability moving into new classes."""

        js, invasion = padded_jensen_shannon([[1., 0.], [1., 0.]], [[1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_allclose(js, [0., np.log(2.)], atol=1e-15)
        np.testing.assert_array_equal(invasion, [0., 1.])
        js, invasion = padded_jensen_shannon([[.25, .75]], [[.125, .375, .5]])
        self.assertGreater(js[0], 0.)
        self.assertEqual(invasion[0], .5)

    def test_mean_divergences_is_not_divergence_after_view_averaging(self) -> None:
        """Exhibit view changes that cancel only under the incorrect reduction."""

        teacher = np.array([[1., 0.], [0., 1.]])
        student = teacher[::-1]
        per_view, _ = padded_jensen_shannon(teacher, student)
        averaged, _ = padded_jensen_shannon(teacher.mean(axis=0, keepdims=True), student.mean(axis=0, keepdims=True))
        self.assertAlmostEqual(float(per_view.mean()), np.log(2.))
        self.assertEqual(float(averaged[0]), 0.)

    def test_normalization_and_invalid_support(self) -> None:
        """Normalize large finite mixtures and reject invalid distributions."""

        np.testing.assert_array_equal(probability_rows([[1e308, 1e308]]), [[.5, .5]])
        for bad in ([[0., 0.]], [[-1., 2.]], [[np.nan, 1.]], [1., 0.]):
            with self.assertRaises(ValueError):
                probability_rows(bad)
        with self.assertRaises(ValueError):
            padded_jensen_shannon([[.2, .3, .5]], [[.5, .5]])


class SelectionTests(unittest.TestCase):
    """Exercise quality provenance, exact budgets, coverage, and replay controls."""

    def test_fixed_training_quantile_and_no_test_leakage(self) -> None:
        """Allow one training fit per task while rejecting test-split tuning."""

        labels = np.array([0, 1, 0, 1])
        scored = _scored(labels, [.1, .4, .7, 1.])
        selector = DriftReplaySelector(ReplaySelectionSettings(quality_quantile=.5), 7)
        record = selector.fit_quality_threshold(None, np.arange(4), labels, scored=scored)
        self.assertAlmostEqual(record["threshold"], .55)
        self.assertEqual(record["interpretation"], "teacher_self_consistency")
        with self.assertRaises(RuntimeError):
            selector.fit_quality_threshold(None, np.arange(4), labels, scored=scored)
        with self.assertRaises(ValueError):
            selector.fit_quality_threshold(None, np.arange(4), labels, split="test", scored=scored)

    def test_threshold_then_ranking_and_class_floor(self) -> None:
        """Retain valid high-drift rows while reserving both class floor slots."""

        labels = np.array([0, 0, 0, 1, 1, 1])
        scored = _scored(labels, [.9, .8, .2, .9, .8, .2])
        selector = DriftReplaySelector(ReplaySelectionSettings(quality_threshold=.5), 7)
        images, selected, report = selector.select(np.arange(6), labels, 3, [0, 1], scored)
        self.assertEqual(set(images), {1, 3, 4})
        self.assertEqual(set(selected), {0, 1})
        self.assertEqual(report["quality_pass_count"], 4)
        self.assertEqual(report["quality_floor_exception_indices"], [])
        self.assertEqual(report["selected_count"], 3)

    def test_small_retained_pools_cover_all_classes_over_window(self) -> None:
        """Cover five classes across three two-row retained pools."""

        labels = np.repeat(np.arange(5), 2)
        selector = DriftReplaySelector(ReplaySelectionSettings(quality_threshold=0.), 11)
        counts = set()
        for _ in range(3):
            _, selected, report = selector.select(np.arange(10), labels, 2, list(range(5)), _scored(labels))
            counts.update(selected.tolist())
            self.assertEqual(report["coverage_window_calls"], 3)
            self.assertEqual(len(selected), 2)
        self.assertEqual(counts, set(range(5)))

    def test_infeasible_quality_and_missing_classes_do_not_silently_bypass(self) -> None:
        """Reject empty eligibility without advancing the selection state."""

        labels = np.array([0, 0, 1, 1])
        selector = DriftReplaySelector(ReplaySelectionSettings(quality_threshold=.5), 11)
        with self.assertRaisesRegex(ValueError, "eligible candidates"):
            selector.select(np.arange(4), labels, 3, [0, 1], _scored(labels, [.8, .7, .2, .1]))
        with self.assertRaisesRegex(ValueError, "quality-eligible"):
            selector.select(np.arange(4), labels, 1, [0, 1], _scored(labels, [.8, .7, .2, .1]))
        with self.assertRaisesRegex(ValueError, "quality-eligible"):
            selector.select(np.arange(4), labels, 1, [0, 1, 2], _scored(labels))
        self.assertEqual(selector.selection_calls, 0)

    def test_multiple_positive_floor_and_strict_quality_feasibility(self) -> None:
        """Keep two positives per class for semantic acquisition/consolidation."""

        labels = np.repeat(np.arange(3), 3)
        scores = _scored(labels)
        selector = DriftReplaySelector(ReplaySelectionSettings(quality_threshold=.5, min_per_class=2), 11)
        _, selected, report = selector.select(np.arange(9), labels, 7, [0, 1, 2], scores)
        self.assertEqual(len(selected), 7)
        self.assertTrue(all(np.sum(selected == class_id) >= 2 for class_id in range(3)))
        self.assertEqual(report["min_per_class"], 2)
        self.assertEqual(report["coverage_window_calls"], 1)
        scores["teacher_label_probability"][:2] = .1
        with self.assertRaisesRegex(ValueError, "min_per_class=2"):
            selector.select(np.arange(9), labels, 6, [0, 1, 2], scores)

    def test_multiple_positive_floor_rotates_complete_blocks_in_small_pools(self) -> None:
        """Rotate full two-row class floors and reject a budget below one block."""

        labels = np.repeat(np.arange(3), 3)
        scores = _scored(labels)
        selector = DriftReplaySelector(ReplaySelectionSettings(quality_threshold=0., min_per_class=2), 11)
        covered = set()
        for _ in range(3):
            _, selected, report = selector.select(np.arange(9), labels, 3, [0, 1, 2], scores)
            floor_class = report["class_floor_ids_this_call"][0]
            self.assertGreaterEqual(int(np.sum(selected == floor_class)), 2)
            self.assertEqual(len(selected), 3)
            self.assertEqual(report["coverage_window_calls"], 3)
            covered.add(floor_class)
        self.assertEqual(covered, {0, 1, 2})
        with self.assertRaisesRegex(ValueError, "cannot fit min_per_class"):
            selector.select(np.arange(9), labels, 1, [0, 1, 2], scores)

    def test_common_control_rankings_and_mir_are_separate(self) -> None:
        """Preserve existing control rankings and distinguish prospective losses."""

        labels = np.array([0, 0, 1, 1])
        scores = _scored(labels)
        scores["teacher_probabilities"] = np.array([[.99, .01], [.6, .4], [.55, .45], [.1, .9]])
        expected = {"confidence": 0, "label_surprisal": 2, "drift": 3, "mir": 1}
        for strategy, best in expected.items():
            selector = DriftReplaySelector(ReplaySelectionSettings(
                strategy=strategy, quality_threshold=0., class_coverage=False,
            ), 11)
            selected, _, _ = selector.select(np.arange(4), labels, 1, [0, 1], scores,
                                             interference=np.array([0., 2., -1., 1.]))
            self.assertEqual(selected.tolist(), [best])
        a = DriftReplaySelector(ReplaySelectionSettings(strategy="random", quality_threshold=0.), 37)
        b = DriftReplaySelector(ReplaySelectionSettings(strategy="random", quality_threshold=0.), 37)
        np.testing.assert_array_equal(
            a.select(np.arange(4), labels, 2, [0, 1], scores)[0],
            b.select(np.arange(4), labels, 2, [0, 1], scores)[0],
        )

    def test_settings_and_zero_budget(self) -> None:
        """Reject invalid settings and preserve well-defined empty selections."""

        for kwargs in ({"candidate_multiplier": 0}, {"quality_threshold": 1.1},
                       {"quality_quantile": np.nan}, {"quality_threshold_split": "test"},
                       {"noise_levels": (0, 0)}, {"batch_size": True}, {"min_per_class": 0}):
            with self.assertRaises(ValueError):
                ReplaySelectionSettings(**kwargs)
        selector = DriftReplaySelector(ReplaySelectionSettings(quality_threshold=0.), 1)
        labels = np.array([0, 1])
        selected, _, report = selector.select(np.arange(2), labels, 0, [0, 1], _scored(labels))
        self.assertEqual(len(selected), 0)
        self.assertIsNone(report["coverage_window_calls"])
        selected, _, report = selector.select(np.empty(0), np.empty(0, dtype="int32"), 0, [],
                                              _scored(np.empty(0, dtype="int32")))
        self.assertEqual(len(selected), 0)
        self.assertEqual(report["candidate_count"], 0)


class ActualNetworkReplayTests(unittest.TestCase):
    """Exercise real forward diffusion and a real Adam virtual joint update."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build and warm the real factory model without downloading image data."""

        from common.model import get_model

        cls.wrapper = get_model(
            model_name="dit_classifier", task="joint", image_shape=(4, 4, 1),
            class_num=2, seed=29, dtype_policy="float32", show_network_summary=False,
            model_kwargs={
                "timesteps": 4, "patch_size": 2, "dim": 4, "depth": 1,
                "mha_num_heads": 1, "vit_block_mlp_ratio": 1.,
                "clf_mha_num_heads": 1, "clf_vit_block_mlp_ratio": 1.,
                "classifier_mlp_ratio": 1, "classifier_dropout_rate": 0., "droppath_rate": 0.,
                "clf_droppath_rate": 0., "compile_args": {"run_eagerly": True},
            },
            wrapper_kwargs={
                "use_ema": False, "p_uncond": 1., "clf_loss_coef": 1.,
                "test_noisified_max_timesteps": 0, "test_steps": 2,
            },
        )
        cls.images = np.linspace(-1., 1., 4 * 4 * 4, dtype="float32").reshape(4, 4, 4, 1)
        cls.labels = np.array([0, 0, 1, 1], dtype="int32")
        cls.wrapper.train_step((tf.constant(cls.images), tf.constant(cls.labels)))
        cls.teacher = cls.wrapper.snapshot_teacher_network("raw")
        cls.wrapper.map_preprocess = True
        cls.prepared = cls.wrapper.prep_inputs_map(tf.constant(cls.images), tf.constant(cls.labels))

    @classmethod
    def tearDownClass(cls) -> None:
        """Release test-local Keras graph and model naming state."""

        tf.keras.backend.clear_session()

    def test_fixed_stateless_views_and_actual_accumulated_change(self) -> None:
        """Keep views fixed across calls while real wake updates create drift."""

        wrapper = self.wrapper
        selector = DriftReplaySelector(ReplaySelectionSettings(noise_levels=(0, 2), batch_size=2), 29)
        before = selector.score(wrapper, self.images, self.labels, self.teacher)
        repeat = selector.score(wrapper, self.images, self.labels, self.teacher)
        np.testing.assert_array_equal(before["drift"], repeat["drift"])
        np.testing.assert_array_equal(before["student_label_loss"], repeat["student_label_loss"])
        np.testing.assert_allclose(before["drift"], 0., atol=1e-12)
        values = [value.numpy().copy() for value in wrapper.network.weights]
        optimizer_variables = wrapper.optimizer.variables
        optimizer_variables = optimizer_variables() if callable(optimizer_variables) else optimizer_variables
        optimizer_values = [value.numpy().copy() for value in optimizer_variables]
        try:
            for _ in range(3):
                wrapper.train_step(self.prepared)
            after = selector.score(wrapper, self.images, self.labels, self.teacher)
            self.assertGreater(float(after["drift"].mean()), 0.)
            self.assertEqual(after["diagnostics"]["teacher_example_forwards"], 8)
            self.assertEqual(after["diagnostics"]["student_example_forwards"], 8)
            self.assertEqual(after["diagnostics"]["noisy_image_draws"], 4)
            np.testing.assert_array_equal(before["teacher_probabilities"], after["teacher_probabilities"])
        finally:
            for variable, value in zip(wrapper.network.weights, values):
                variable.assign(value)
            for variable, value in zip(optimizer_variables, optimizer_values):
                variable.assign(value)

    def test_teacher_student_receive_identical_views_and_null_conditions(self) -> None:
        """Inspect both raw classifier calls for exact tensor correspondence."""

        selector = DriftReplaySelector(ReplaySelectionSettings(noise_levels=(0, 2), batch_size=2), 29)
        with patch.object(self.teacher, "predict_class", wraps=self.teacher.predict_class) as teacher_call, patch.object(
            self.wrapper.network, "predict_class", wraps=self.wrapper.network.predict_class,
        ) as student_call:
            selector.score(self.wrapper, self.images, self.labels, self.teacher)
            self.assertEqual(teacher_call.call_count, student_call.call_count)
            for prior, current in zip(teacher_call.call_args_list, student_call.call_args_list):
                for prior_tensor, current_tensor in zip(prior.args[0], current.args[0]):
                    np.testing.assert_array_equal(prior_tensor.numpy(), current_tensor.numpy())
                np.testing.assert_array_equal(prior.args[0][2].numpy(), [0, 0])

    def test_mir_restores_adam_state_and_matches_an_actual_update(self) -> None:
        """Match virtual loss change to the actual V1 Adam update and restore it."""

        wrapper = self.wrapper
        selector = DriftReplaySelector(ReplaySelectionSettings(noise_levels=(0, 2)), 17)
        before = selector.score(wrapper, self.images, self.labels, self.teacher)
        optimizer_variables = wrapper.optimizer.variables
        variables = list(wrapper.weights) + list(optimizer_variables() if callable(optimizer_variables) else optimizer_variables)
        variables += [variable for metric in wrapper.metrics for variable in metric.variables]
        state = [variable.numpy().copy() for variable in variables]
        interference, report = virtual_update_interference(
            wrapper, self.prepared, selector, self.images, self.labels, self.teacher, before_scores=before,
        )
        for variable, value in zip(variables, state):
            np.testing.assert_array_equal(variable.numpy(), value)
        self.assertTrue(report["state_restored"])
        self.assertEqual(report["virtual_optimizer_updates"], 1)
        self.assertEqual(report["committed_optimizer_updates"], 0)
        try:
            wrapper.train_step(self.prepared)
            real_after = selector.score(wrapper, self.images, self.labels, self.teacher)
            np.testing.assert_allclose(interference, real_after["student_label_loss"] - before["student_label_loss"], atol=1e-10)
        finally:
            for variable, value in zip(variables, state):
                variable.assign(value)

    def test_mir_restores_on_failed_scoring_and_rejects_raw_batches(self) -> None:
        """Restore mutated values through exceptions and reject stochastic mapping."""

        selector = DriftReplaySelector(ReplaySelectionSettings(), 17)
        before = selector.score(self.wrapper, self.images, self.labels, self.teacher)
        optimizer_variables = self.wrapper.optimizer.variables
        variables = list(self.wrapper.weights) + list(optimizer_variables() if callable(optimizer_variables) else optimizer_variables)
        state = [variable.numpy().copy() for variable in variables]
        with patch.object(selector, "score", side_effect=RuntimeError("intentional scoring failure")):
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                virtual_update_interference(self.wrapper, self.prepared, selector, self.images, self.labels,
                                            self.teacher, before_scores=before)
        for variable, value in zip(variables, state):
            np.testing.assert_array_equal(variable.numpy(), value)
        with self.assertRaisesRegex(ValueError, "already mapped"):
            virtual_update_interference(self.wrapper, (self.images, self.labels), selector,
                                        self.images, self.labels, self.teacher)

    def test_stochastic_latents_are_not_misreported_as_fixed_views(self) -> None:
        """Reject features whose hidden Gaussian sampling changes during inference."""

        selector = DriftReplaySelector(ReplaySelectionSettings(), 17)
        with patch.object(self.wrapper.network, "reshaper_ids_dict", {1: "flatten"}), patch.object(
            self.wrapper.network, "reshaper_kwargs", {"add_kl": True},
        ):
            with self.assertRaisesRegex(ValueError, "deterministic latent"):
                selector.score(self.wrapper, self.images, self.labels, self.teacher)

    def test_virtual_preparation_matches_training_bounds_and_noise_teacher_scale(self) -> None:
        """Honor restricted training timesteps and training CFG in noise-KD-only MIR."""

        from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier

        configuration = dict(self.wrapper.get_config())
        configuration.update(
            network=self.wrapper.network, teacher_network=self.teacher,
            noise_distil_loss_coef=.1, clf_distil_loss_coef=0.,
            train_noisified_min_timesteps=1, train_noisified_max_timesteps=2,
            train_cfg_scale=1., test_cfg_scale=3.,
        )
        wrapper = DiffusionClassifier(**configuration)
        self.assertTrue(wrapper.use_noise_distil_loss)
        self.assertFalse(wrapper.use_classifier_distil)
        bounds = wrapper._active_min_timestep, wrapper._active_max_timestep
        batch = (tf.constant(self.images), tf.constant(self.labels), tf.zeros(4, tf.bool))
        with patch.object(wrapper, "forward", wraps=wrapper.forward) as forward:
            prepared = prepare_virtual_current_batch(wrapper, batch)
        np.testing.assert_array_equal(prepared[2].numpy(), np.ones(4, dtype="int32"))
        self.assertEqual(forward.call_args.kwargs["scale"], wrapper.train_cfg_scale)
        self.assertEqual((wrapper._active_min_timestep, wrapper._active_max_timestep), bounds)
        self.assertIsNone(wrapper._preprocess_training)
        with patch.object(wrapper, "prep_inputs_map", side_effect=RuntimeError("preparation failed")):
            with self.assertRaisesRegex(RuntimeError, "preparation failed"):
                prepare_virtual_current_batch(wrapper, batch)
        self.assertEqual((wrapper._active_min_timestep, wrapper._active_max_timestep), bounds)
        self.assertIsNone(wrapper._preprocess_training)


# Direct execution runs these bounded regressions without the repository-wide suite.
if __name__ == "__main__":
    unittest.main()
