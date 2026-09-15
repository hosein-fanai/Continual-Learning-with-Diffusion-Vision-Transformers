"""Real DiT gradient-boundary and paired-view regressions for route phases.

Tiny synthetic images and a common.Config-built joint diffusion classifier
exercise the actual feature extraction and optimizer APIs without downloading
data or claiming benchmark efficacy.
"""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import Config
from common.keras_compat import format_variable_name
from common.model import get_model
from semantic_consolidation.config import RouteSettings
from semantic_consolidation.memory import ClassBalancedPool, ModulationBank
from semantic_consolidation.phases import RoutePhase, paired_view, semantic_features


def _make_wrapper() -> tf.keras.Model:
    """Construct a two-class, four-pixel-side DiT through the shared factory."""

    config = Config()
    config.training.task = "joint"
    config.training.seed = 41
    config.training.dtype_policy = "float32"
    config.dataset.trainset_len = 2
    config.model.name = "dit_classifier"
    config.model.wrapper_name = "diffusion_classifier"
    config.model.show_network_summary = False
    config.model.kwargs = {
        "num_classes": 2, "use_cfg": True, "timesteps": 4,
        "image_size": 4, "channels": 1, "patch_size": 2,
        "dim": 8, "cond_dim": 8, "depth": 1, "mha_num_heads": 1,
        "vit_block_mlp_ratio": 1., "clf_depth": 0,
        "clf_vit_block_ids": [], "clf_cls_token_type": None,
        "feature_aggregation_ids_dict": {1: [1]},
        "force_global_avg_pooling": True, "classifier_mlp_ratio": 1,
        "dropout_rate": 0.2, "build": True,
    }
    config.model.wrapper_kwargs = {
        "use_ema": False, "test_network_name": "raw", "test_steps": 2,
        "modify_first_t": False, "test_noisified_min_timesteps": 0,
        "test_noisified_max_timesteps": 0,
    }
    return get_model(config)


def _values(variables: list[tf.Variable]) -> dict[object, np.ndarray]:
    """Copy values keyed by variable identity, independent of duplicated names."""

    return {id(variable): variable.numpy().copy() for variable in variables}


class PhaseTests(unittest.TestCase):
    """Verify the proposed acquisition and consolidation freeze boundaries."""

    def setUp(self) -> None:
        """Create isolated fixtures and preserve the caller state needed for this test."""
        self.previous_policy = tf.keras.mixed_precision.global_policy().name
        self.wrapper = _make_wrapper()
        self.images = np.random.default_rng(47).normal(size=(8, 4, 4, 1)).astype("float32")
        self.labels = np.repeat([0, 1], 4).astype("int32")
        self.settings = RouteSettings(batch_size=4, noise_levels=(0, 2), learning_rate=0.01)
        self.pool = ClassBalancedPool(self.images, self.labels)
        projection, _ = semantic_features(
            self.wrapper.network, self.images, tf.zeros(8, dtype=tf.int32)
        )
        self.bank = ModulationBank(self.settings, dimension=int(projection.shape[1]), seed=53)
        self.bank.add([0, 1])

    def tearDown(self) -> None:
        """Release fixture state and restore the caller numerical configuration."""
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy(self.previous_policy)

    def _phase(self, phase: str, **kwargs: object) -> RoutePhase:
        """Create a reproducible acquisition or consolidation fixture over the shared pool."""
        model = RoutePhase(
            self.wrapper, self.bank, self.pool, self.settings,
            phase=phase, classes=[1], seed=59, **kwargs,
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), run_eagerly=True)
        return model

    def test_semantic_features_match_actual_unmodulated_classifier(self) -> None:
        """Extracting the projection preserves raw task-ID-free probabilities."""

        times = tf.zeros(8, dtype=tf.int32)
        features, probabilities = semantic_features(self.wrapper.network, self.images, times)
        expected = self.wrapper.network.predict_class(
            (self.images, times, tf.zeros_like(times)),
            max_encoder_num=None, full_return=False, training=False,
        )
        np.testing.assert_allclose(probabilities.numpy(), expected.numpy(), atol=1e-7)
        self.assertEqual(features.shape, (8, self.bank.dimension))
        self.assertEqual(probabilities.shape, (8, 2))
        _, repeat = semantic_features(self.wrapper.network, self.images, times)
        np.testing.assert_array_equal(probabilities.numpy(), repeat.numpy())

    def test_zero_route_level_is_exactly_clean_even_when_schedule_zero_is_noisy(self) -> None:
        """Clean controls use the wrapper clean-bounds API, not explicit timestep zero."""

        self.assertLess(float(self.wrapper.schedules["alpha_bar"][0]), 1.)
        noised, times, reliability = paired_view(self.wrapper, self.images, 0, 61)
        np.testing.assert_array_equal(noised.numpy(), self.images)
        np.testing.assert_array_equal(times.numpy(), np.zeros(8, dtype="int32"))
        self.assertEqual(float(reliability), 1.)

    def test_nonzero_route_view_uses_existing_noisify_and_signal_schedule(self) -> None:
        """Positive noise levels share the platform forward-process semantics."""

        with patch.object(self.wrapper, "noisify", wraps=self.wrapper.noisify) as noisify:
            noised, times, reliability = paired_view(self.wrapper, self.images, 2, 67)
        self.assertEqual(noisify.call_count, 1)
        self.assertEqual(noisify.call_args.kwargs["seed"], 67)
        np.testing.assert_array_equal(times.numpy(), np.full(8, 2, dtype="int32"))
        self.assertAlmostEqual(float(reliability), float(self.wrapper.schedules["alpha_bar"][2]))
        self.assertFalse(np.array_equal(noised.numpy(), self.images))

    def test_acquisition_updates_selected_modulator_and_no_network_weight(self) -> None:
        """All acquired backbone/head/state values stay exact during gate learning."""

        network_before = _values(self.wrapper.network.weights)
        bank_before = _values([variable for pair in self.bank.vectors.values() for variable in pair])
        phase = self._phase("acquisition")
        metrics = phase.train_step(None)
        for variable in self.wrapper.network.weights:
            np.testing.assert_array_equal(variable.numpy(), network_before[id(variable)])
        for variable in self.bank.vectors[0]:
            np.testing.assert_array_equal(variable.numpy(), bank_before[id(variable)])
        self.assertTrue(any(
            not np.array_equal(variable.numpy(), bank_before[id(variable)])
            for variable in self.bank.vectors[1]
        ))
        self.assertEqual(phase.updated_names, {variable.name for variable in self.bank.vectors[1]})
        self.assertEqual(phase.example_draws, 4)
        self.assertEqual(phase.view_draws, 4)
        self.assertEqual(int(phase.optimizer.iterations.numpy()), 1)
        self.assertEqual(float(metrics["ce"]), 0.)
        self.assertTrue(all(bool(tf.math.is_finite(value)) for value in metrics.values()))

    def test_consolidation_only_updates_projection_head_and_predictor(self) -> None:
        """The frozen target, bank, and entire denoising path remain byte-identical."""

        target = self.wrapper.snapshot_teacher_network("raw")
        self.assertFalse(target.trainable)
        self.assertFalse({id(value) for value in target.weights} & {id(value) for value in self.wrapper.network.weights})
        phase = self._phase("consolidation", target=target, frozen_bank=self.bank.frozen())
        before = _values(self.wrapper.network.weights)
        target_before = _values(target.weights)
        bank_before = _values([variable for pair in self.bank.vectors.values() for variable in pair])
        predictor_before = _values(phase.predictor.weights)
        classifier_variables = {id(variable) for variable in self.wrapper.network.classifier.trainable_variables}
        inputs = (self.images, tf.fill((8,), 2), tf.zeros(8, dtype=tf.int32))
        denoising_before = self.wrapper.network.predict_noise(inputs, training=False).numpy().copy()
        with patch("semantic_consolidation.phases.semantic_features", wraps=semantic_features) as features:
            metrics = phase.train_step(None)
        self.assertEqual(features.call_count, 5)
        for offset in (1, 3):
            student_call, target_call = features.call_args_list[offset:offset + 2]
            self.assertIs(student_call.args[0], self.wrapper.network)
            self.assertIs(target_call.args[0], target)
            self.assertIs(student_call.args[1], target_call.args[1])
            self.assertIs(student_call.args[2], target_call.args[2])
        changed = set()
        for variable in self.wrapper.network.weights:
            # Ignore unchanged variables when auditing which gradient paths actually moved.
            if np.array_equal(variable.numpy(), before[id(variable)]):
                continue
            changed.add(id(variable))
            self.assertIn(id(variable), classifier_variables, variable.name)
            self.assertIn(format_variable_name(variable), phase.updated_names)
        self.assertTrue(changed)
        for layer in (self.wrapper.network.classifier.layers[0], self.wrapper.network.classifier.layers[-1]):
            self.assertTrue(any(id(variable) in changed for variable in layer.weights))
        self.assertTrue(any(
            not np.array_equal(variable.numpy(), predictor_before[id(variable)])
            for variable in phase.predictor.weights
        ))
        for variable in target.weights:
            np.testing.assert_array_equal(variable.numpy(), target_before[id(variable)])
        for pair in self.bank.vectors.values():
            for variable in pair:
                np.testing.assert_array_equal(variable.numpy(), bank_before[id(variable)])
        np.testing.assert_array_equal(
            self.wrapper.network.predict_noise(inputs, training=False).numpy(), denoising_before
        )
        self.assertTrue(all(bool(tf.math.is_finite(value)) for value in metrics.values()))

    def test_semantic_noise_levels_do_not_change_acquisition_updates(self) -> None:
        """Acquisition holds its view fixed when consolidation noise bands change."""

        single = replace(self.settings, noise_levels=(0,))
        repeated = replace(self.settings, noise_levels=(0, 1))
        banks = [ModulationBank(single, self.bank.dimension, 71) for _ in range(2)]
        phases = []
        for settings, bank in zip((single, repeated), banks):
            bank.add([0, 1])
            phase = RoutePhase(self.wrapper, bank, self.pool, settings, "acquisition", [1], 73)
            phase.compile(optimizer=tf.keras.optimizers.SGD(0.01), run_eagerly=True)
            phases.append(phase)

        for phase in phases:
            phase.train_step(None)
        self.assertAlmostEqual(phases[0].trace[0]["semantic_loss"], phases[1].trace[0]["semantic_loss"], places=6)
        for actual, expected in zip(banks[0].vectors[1], banks[1].vectors[1]):
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-7)
        self.assertEqual(phases[1].view_draws, phases[0].view_draws)

    def test_acquisition_rounds_cover_all_selected_classes_equally(self) -> None:
        """Each complete focus cycle updates every new class once."""

        for optimizer_type in (tf.keras.optimizers.SGD, tf.keras.optimizers.Adam):
            with self.subTest(optimizer=optimizer_type.__name__):
                phase = RoutePhase(
                    self.wrapper, self.bank, self.pool, self.settings,
                    "acquisition", [0, 1], 79,
                )
                phase.compile(optimizer=optimizer_type(0.01), run_eagerly=True)
                for _ in range(4):
                    phase.train_step(None)
                self.assertEqual(phase.focus_counts, {0: 2, 1: 2})
                for start in (0, 2):
                    self.assertEqual({row["focus_class"] for row in phase.trace[start:start + 2]}, {0, 1})

    def test_adapter_preserves_compiled_execution_group_and_live_optimizer(self) -> None:
        """Adapting the existing model retains its explicit multi-step compile policy."""
        from semantic_consolidation.model import adapt_model

        self.wrapper.compile(optimizer=self.wrapper.optimizer, loss=self.wrapper.loss,
                             run_eagerly=True, steps_per_execution=3)
        adapted = adapt_model(self.wrapper, controller=object())
        steps = getattr(adapted, "steps_per_execution", getattr(adapted, "_steps_per_execution", None))
        self.assertEqual(int(steps.numpy() if hasattr(steps, "numpy") else steps), 3)
        self.assertIs(adapted.network, self.wrapper.network)
        self.assertIs(adapted.optimizer, self.wrapper.optimizer)

    def test_averaging_identical_alignment_draws_preserves_consolidation_strength(self) -> None:
        """Duplicating a controlled semantic draw does not multiply its loss."""

        target = self.wrapper.snapshot_teacher_network("raw")
        phases = []
        for noise_levels in ((0,), (0, 1)):
            settings = replace(self.settings, noise_levels=noise_levels)
            phase = RoutePhase(
                self.wrapper, self.bank, self.pool, settings,
                "consolidation", [1], 83, target, self.bank.frozen(),
            )
            # Zero optimizer rate lets both phases evaluate exactly the same weights.
            phase.compile(optimizer=tf.keras.optimizers.SGD(0.0), run_eagerly=True)
            phases.append(phase)

        def clean_view(wrapper: tf.keras.Model, images: tf.Tensor, level: int, seed: int) -> tuple:
            """Supply a fixed clean paired view to isolate noise-averaging loss reductions."""
            return paired_view(wrapper, images, 0, seed)

        with patch("semantic_consolidation.phases.paired_view", side_effect=clean_view):
            for phase in phases:
                phase.train_step(None)
        for metric in ("ce", "semantic_loss", "loss"):
            self.assertAlmostEqual(phases[0].trace[0][metric], phases[1].trace[0][metric], places=6)
        self.assertEqual(phases[1].view_draws, 2 * phases[0].view_draws)

    def test_semantic_noise_bands_and_reliability_do_not_change_supervised_ce(self) -> None:
        """Only alignment sees noise-band/reliability ablations; CE stays clean."""

        target = self.wrapper.snapshot_teacher_network("raw")
        phases = []
        for noise_levels, reliability in (((0,), "uniform"), ((0, 2), "alpha_bar")):
            settings = replace(self.settings, noise_levels=noise_levels, reliability=reliability)
            phase = RoutePhase(
                self.wrapper, self.bank, self.pool, settings,
                "consolidation", [1], 89, target, self.bank.frozen(),
            )
            phase.compile(optimizer=tf.keras.optimizers.SGD(0.0), run_eagerly=True)
            phase.train_step(None)
            phases.append(phase)
        self.assertAlmostEqual(phases[0].trace[0]["ce"], phases[1].trace[0]["ce"], places=7)
        self.assertEqual(phases[0].trace[0]["focus_class"], phases[1].trace[0]["focus_class"])
        self.assertEqual(phases[0].trace[0]["examples"], phases[1].trace[0]["examples"])

    def test_predictor_and_modulation_bank_do_not_enter_raw_prediction(self) -> None:
        """Training-only control can be changed without changing deployed logits."""

        target = self.wrapper.snapshot_teacher_network("raw")
        phase = self._phase("consolidation", target=target, frozen_bank=self.bank.frozen())
        times = tf.zeros(8, dtype=tf.int32)
        _, before = semantic_features(self.wrapper.network, self.images, times)
        for variable in phase.predictor.weights:
            variable.assign(tf.fill(variable.shape, 100.))
        for pair in self.bank.vectors.values():
            for variable in pair:
                variable.assign(tf.fill(variable.shape, -100.))
        _, after = semantic_features(self.wrapper.network, self.images, times)
        np.testing.assert_array_equal(before.numpy(), after.numpy())

    def test_controller_rejects_all_missing_positive_pairs_before_gate_updates(self) -> None:
        """A scarce class fails before any randomly earlier class can acquire a gate."""
        from common.dataloader import get_dataset
        from semantic_consolidation.controller import RouteController

        self.wrapper.seen_classes = {0: 0, 1: 1}
        dataset = get_dataset(
            self.images[:4], np.array([0, 1, 1, 1], dtype="int32"),
            batch_size=4, shuffle_buffer=0, drop_remainder=False,
        )
        controller = RouteController(self.settings)
        before = _values(self.wrapper.network.weights)
        with patch.object(controller, "_fit") as fit_phase:
            with self.assertRaisesRegex(ValueError, "two positive rows"):
                controller.run(self.wrapper, dataset, {})
        fit_phase.assert_not_called()
        self.assertIsNone(controller.bank)
        self.assertEqual(controller.records, [])
        for variable in self.wrapper.network.weights:
            np.testing.assert_array_equal(variable.numpy(), before[id(variable)])


# Run this module directly while keeping imports free of execution side effects.
if __name__ == "__main__":
    unittest.main()
