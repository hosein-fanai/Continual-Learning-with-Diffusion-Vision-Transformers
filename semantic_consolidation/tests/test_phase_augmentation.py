"""TMCL view routing against the real TensorFlow diffusion classifier API.

Synthetic RGB inputs check transformation placement, loss scaling, and replay
of saved phase state. These checks make no benchmark-performance claim.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import Config
from common.model import get_model
from semantic_consolidation.augmentation import acquisition_augmentation, consolidation_views
from semantic_consolidation.config import RouteSettings
from semantic_consolidation.fit_recovery import _variables
from semantic_consolidation.memory import ClassBalancedPool, ModulationBank
from semantic_consolidation.objectives import contrastive_alignment_loss
from semantic_consolidation.phases import RoutePhase, paired_view, semantic_features


def _rgb_wrapper() -> tf.keras.Model:
    """Build a small genuine DiT accepting the paper's 32-by-32 RGB views."""
    config = Config()
    config.training.task = "joint"
    config.training.seed = 137
    config.training.dtype_policy = "float32"
    config.dataset.trainset_len = 2
    config.model.name = "dit_classifier"
    config.model.wrapper_name = "diffusion_classifier"
    config.model.show_network_summary = False
    config.model.kwargs = {
        "num_classes": 2, "use_cfg": True, "timesteps": 4,
        "image_size": 32, "channels": 3, "patch_size": 8,
        "dim": 8, "cond_dim": 8, "depth": 1, "mha_num_heads": 1,
        "vit_block_mlp_ratio": 1., "clf_depth": 0,
        "clf_vit_block_ids": [], "clf_cls_token_type": None,
        "feature_aggregation_ids_dict": {1: [1]},
        "force_global_avg_pooling": True, "classifier_mlp_ratio": 1,
        "classifier_dropout_rate": 0., "build": True,
    }
    config.model.wrapper_kwargs = {
        "use_ema": False, "test_network_name": "raw", "test_steps": 2,
        "modify_first_t": False, "test_noisified_min_timesteps": 0,
        "test_noisified_max_timesteps": 0,
    }
    return get_model(config)


class PhaseAugmentationTests(unittest.TestCase):
    """Verify view augmentation as used by actual semantic optimizer updates."""

    def setUp(self) -> None:
        """Create isolated RGB model, pool, modulation bank, and frozen target."""
        self.previous_policy = tf.keras.mixed_precision.global_policy().name
        self.wrapper = _rgb_wrapper()
        self.images = np.random.default_rng(139).uniform(-1., 1., (8, 32, 32, 3)).astype("float32")
        self.labels = np.repeat([0, 1], 4).astype("int32")
        self.pool = ClassBalancedPool(self.images, self.labels)
        self.settings = RouteSettings(
            batch_size=4, noise_levels=(0, 2), image_augmentation="tmcl", augmentation_views=4,
        )
        features, _ = semantic_features(self.wrapper.network, self.images, tf.zeros(8, tf.int32))
        self.bank = ModulationBank(self.settings, int(features.shape[1]), seed=149)
        self.bank.add([0, 1])
        self.target = self.wrapper.snapshot_teacher_network("raw")

    def tearDown(self) -> None:
        """Release the Keras fixture and restore the caller's dtype policy."""
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy(self.previous_policy)

    def _phase(self, phase: str = "consolidation", settings: RouteSettings | None = None,
               learning_rate: float = 0.) -> RoutePhase:
        """Compile an eager real phase over the fixture's independent target."""
        model = RoutePhase(
            self.wrapper, self.bank, self.pool, settings or self.settings,
            phase, [0, 1], seed=151, target=self.target, frozen_bank=self.bank.frozen(),
        )
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate), run_eagerly=True)
        return model

    def test_acquisition_flips_before_noise_and_preserves_pool(self) -> None:
        """Acquisition's sole transformed view enters the existing noising API."""
        settings = replace(self.settings, acquisition_noise_level=2)
        phase = self._phase("acquisition", settings)
        augmented = []

        def augment(*args: object, **kwargs: object) -> tf.Tensor:
            """Capture the genuine flipped tensor before diffusion corruption."""
            result = acquisition_augmentation(*args, **kwargs)
            augmented.append(result)
            return result

        before = self.pool.images.copy(), self.pool.labels.copy()
        with patch("semantic_consolidation.phases.acquisition_augmentation", side_effect=augment) as transform, \
                patch("semantic_consolidation.phases.consolidation_views") as multiple, \
                patch("semantic_consolidation.phases.paired_view", wraps=paired_view) as noising:
            phase.train_step(None)
        self.assertEqual(transform.call_count, 1)
        multiple.assert_not_called()
        self.assertEqual(noising.call_count, 1)
        np.testing.assert_array_equal(noising.call_args.args[1].numpy(), augmented[0].numpy())
        self.assertEqual(noising.call_args.args[2], 2)
        np.testing.assert_array_equal(self.pool.images, before[0])
        np.testing.assert_array_equal(self.pool.labels, before[1])
        self.assertEqual(phase.example_draws, 4)
        self.assertEqual(phase.view_draws, 4)

    def test_consolidation_routes_independent_views_and_reuses_them_across_levels(self) -> None:
        """One student and three frozen target views share each sampled noise band."""
        phase = self._phase()
        draws, terms = [], []

        def augment(*args: object, **kwargs: object) -> tuple:
            """Capture the one real collection of independently augmented views."""
            result = consolidation_views(*args, **kwargs)
            draws.append(result)
            return result

        def alignment(*args: object, **kwargs: object) -> tf.Tensor:
            """Record each real pairwise term without detaching its gradient."""
            result = contrastive_alignment_loss(*args, **kwargs)
            terms.append(float(result.numpy()))
            return result

        with patch("semantic_consolidation.phases.consolidation_views", side_effect=augment) as transform, \
                patch("semantic_consolidation.phases.paired_view", wraps=paired_view) as noising, \
                patch("semantic_consolidation.phases.semantic_features", wraps=semantic_features) as features, \
                patch("semantic_consolidation.phases.contrastive_alignment_loss", side_effect=alignment):
            phase.train_step(None)

        self.assertEqual(transform.call_count, 1)
        self.assertEqual(len(draws[0]), 4)
        for index in range(1, 4):
            self.assertFalse(np.array_equal(draws[0][0].numpy(), draws[0][index].numpy()))
        # The extra leading view is the fixed supervised CE input.
        self.assertEqual(noising.call_count, 9)
        self.assertEqual(features.call_count, 9)
        raw = transform.call_args.args[0]
        np.testing.assert_array_equal(noising.call_args_list[0].args[1].numpy(), raw.numpy())
        self.assertIs(features.call_args_list[0].args[0], self.wrapper.network)
        for band, level in enumerate(self.settings.noise_levels):
            for view in range(4):
                offset = 1 + 4 * band + view
                np.testing.assert_array_equal(
                    noising.call_args_list[offset].args[1].numpy(), draws[0][view].numpy(),
                )
                self.assertEqual(noising.call_args_list[offset].args[2], level)
                self.assertIs(
                    features.call_args_list[offset].args[0],
                    self.wrapper.network if view == 0 else self.target,
                )
        self.assertEqual(len(terms), 6)
        self.assertAlmostEqual(phase.trace[-1]["semantic_loss"], float(np.mean(terms)), places=6)
        self.assertEqual(phase.example_draws, 4)
        self.assertEqual(phase.view_draws, 4 * 2 * 4)

    def test_supervised_ce_is_independent_of_augmentation_and_semantic_noise_bands(self) -> None:
        """Enabling TMCL changes alignment while preserving the supervised control."""
        records = []
        for settings in (
            replace(self.settings, image_augmentation="none", noise_levels=(0,)),
            replace(self.settings, augmentation_views=2, noise_levels=(0,)),
            self.settings,
        ):
            phase = self._phase(settings=settings)
            phase.train_step(None)
            records.append(phase.trace[-1])
        for actual in records[1:]:
            self.assertEqual(actual["focus_class"], records[0]["focus_class"])
            self.assertEqual(actual["ce"], records[0]["ce"])

    def test_disabled_augmentation_preserves_the_single_paired_view_path(self) -> None:
        """The explicit historical control invokes no augmentation transforms."""
        phase = self._phase(settings=replace(self.settings, image_augmentation="none"))
        with patch("semantic_consolidation.phases.acquisition_augmentation") as acquisition, \
                patch("semantic_consolidation.phases.consolidation_views") as consolidation, \
                patch("semantic_consolidation.phases.semantic_features", wraps=semantic_features) as features:
            phase.train_step(None)
        acquisition.assert_not_called()
        consolidation.assert_not_called()
        self.assertEqual(features.call_count, 5)
        for offset in (1, 3):
            self.assertIs(features.call_args_list[offset].args[1], features.call_args_list[offset + 1].args[1])
        self.assertEqual(phase.view_draws, 4 * 2)

    def test_tmcl_updates_preserve_acquisition_and_consolidation_freeze_boundaries(self) -> None:
        """Real augmented gradients train gates or the student head without changing frozen state."""
        network_before = [value.numpy().copy() for value in self.wrapper.network.weights]
        gate_before = {
            class_id: [value.numpy().copy() for value in pair]
            for class_id, pair in self.bank.vectors.items()
        }
        acquisition = self._phase("acquisition", learning_rate=0.001)
        acquisition.train_step(None)
        for variable, before in zip(self.wrapper.network.weights, network_before):
            np.testing.assert_array_equal(variable.numpy(), before)
        selected = acquisition.trace[-1]["focus_class"]
        self.assertTrue(any(
            not np.array_equal(variable.numpy(), before)
            for variable, before in zip(self.bank.vectors[selected], gate_before[selected])
        ))
        for variable, before in zip(self.bank.vectors[1 - selected], gate_before[1 - selected]):
            np.testing.assert_array_equal(variable.numpy(), before)

        consolidation = self._phase(learning_rate=0.001)
        eligible = list(self.wrapper.network.classifier.trainable_variables) + list(consolidation.predictor.trainable_variables)
        eligible_ids = {id(value) for value in eligible}
        frozen = [value for value in self.wrapper.network.weights if id(value) not in eligible_ids]
        frozen += list(self.target.weights)
        frozen += [value for pair in self.bank.vectors.values() for value in pair]
        frozen_before = [value.numpy().copy() for value in frozen]
        eligible_before = [value.numpy().copy() for value in eligible]
        consolidation.train_step(None)
        for variable, before in zip(frozen, frozen_before):
            np.testing.assert_array_equal(variable.numpy(), before)
        self.assertTrue(any(
            not np.array_equal(variable.numpy(), before)
            for variable, before in zip(eligible, eligible_before)
        ))

    def test_saved_phase_state_replays_the_next_augmented_update_exactly(self) -> None:
        """Restoring phase counters, sampler and optimizer values reproduces views and weights."""
        phase = self._phase(learning_rate=0.001)
        phase.train_step(None)
        # These are the local fields persisted by FitCheckpointManager in addition
        # to TensorFlow variables. The augmentation itself owns no hidden RNG.
        local = {
            "step_number": phase.step_number, "focus_cycle": list(phase.focus_cycle),
            "focus_counts": dict(phase.focus_counts), "updated_names": set(phase.updated_names),
            "example_draws": phase.example_draws, "view_draws": phase.view_draws,
            "trace": [dict(row) for row in phase.trace],
        }
        rng = deepcopy(phase.rng.bit_generator.state)
        # Use the production collector so wrapper diffusion RNG counters are
        # included alongside the network, predictor, optimizer, and metrics.
        variables = _variables(phase)
        saved = [variable.numpy().copy() for variable in variables]
        draws = []

        def augment(*args: object, **kwargs: object) -> tuple:
            """Copy each real augmented draw for exact resumed-step comparison."""
            result = consolidation_views(*args, **kwargs)
            draws.append(tuple(value.numpy().copy() for value in result))
            return result

        with patch("semantic_consolidation.phases.consolidation_views", side_effect=augment):
            phase.train_step(None)
            expected = [variable.numpy().copy() for variable in variables]
            expected_trace = [dict(row) for row in phase.trace]
            for variable, value in zip(variables, saved):
                variable.assign(value)
            for name, value in local.items():
                setattr(phase, name, deepcopy(value))
            phase.rng.bit_generator.state = deepcopy(rng)
            # Unrelated TensorFlow random draws must not disturb the view stream.
            tf.random.uniform((97,))
            phase.train_step(None)
        for actual, reference in zip(draws[1], draws[0]):
            np.testing.assert_array_equal(actual, reference)
        for variable, reference in zip(variables, expected):
            np.testing.assert_array_equal(variable.numpy(), reference)
        self.assertEqual(phase.trace, expected_trace)


# Direct execution runs only the augmented phase regressions.
if __name__ == "__main__":
    unittest.main()
