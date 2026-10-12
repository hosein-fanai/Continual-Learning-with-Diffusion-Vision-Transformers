"""Equivalent semantic updates with explicit watches instead of frozen graph retention."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.gradients import apply_policy_gradients
from semantic_consolidation.config import RouteSettings
from semantic_consolidation.memory import ClassBalancedPool, ModulationBank
from semantic_consolidation.phases import RoutePhase, semantic_features
from semantic_consolidation.tests.test_phase_augmentation import _rgb_wrapper


class PhaseTapeMemoryTests(unittest.TestCase):
    """Compare legacy and bounded tapes on identical real DiT forward computations."""

    def test_loss_gradients_and_updates_match_legacy_for_every_gradient_scope(self) -> None:
        """Restrict watches without altering losses, RNG draws, gradients or one optimizer update."""

        policy = tf.keras.mixed_precision.global_policy().name
        self.addCleanup(tf.keras.mixed_precision.set_global_policy, policy)
        self.addCleanup(tf.keras.backend.clear_session)
        wrapper = _rgb_wrapper()
        images = np.random.default_rng(401).uniform(-1., 1., (8, 32, 32, 3)).astype("float32")
        labels = np.repeat([0, 1], 4).astype("int32")
        pool = ClassBalancedPool(images, labels)
        settings = RouteSettings(
            batch_size=4, noise_levels=(0, 1, 2), image_augmentation="tmcl", augmentation_views=4
        )
        features, _ = semantic_features(wrapper.network, images, tf.zeros(8, tf.int32))
        bank = ModulationBank(settings, int(features.shape[1]), seed=409)
        bank.add([0, 1])
        target = wrapper.snapshot_teacher_network("raw")
        # Wrapper state includes the checkpointed noise-stream counters as well as network weights.
        variables = list(wrapper.variables) + [value for pair in bank.vectors.values() for value in pair]
        initial = [variable.numpy().copy() for variable in variables]
        native_tape = tf.GradientTape
        treatments = [
            ("acquisition", "contrastive", "learned", "semantic"), 
            ("acquisition", "true_class_ce", "learned", "semantic"), 
            ("consolidation", "contrastive", "learned", "semantic"), 
            ("consolidation", "contrastive", "unmodulated_feature_distillation", "semantic"), 
            ("consolidation", "contrastive", "learned", "backbone")
        ]
        for phase_name, acquisition, condition, scope in treatments:
            with self.subTest(phase=phase_name, acquisition=acquisition, condition=condition, scope=scope):
                observations = []
                for legacy in (False, True):
                    for variable, value in zip(variables, initial):
                        variable.assign(value)
                    options = replace(
                        settings, acquisition_objective=acquisition, condition=condition, consolidation_scope=scope
                    )
                    phase = RoutePhase(
                        wrapper, bank, pool, options, phase_name, [0, 1], 
                        target=target, frozen_bank=bank.frozen(), seed=419
                    )
                    phase.compile(optimizer=tf.keras.optimizers.SGD(0.001), run_eagerly=True)
                    observation = {}

                    def tape_factory(*args: object, **kwargs: object) -> tf.GradientTape:
                        """Restore the old automatic-watch behavior for the reference computation."""

                        kwargs["watch_accessed_variables"] = legacy
                        return native_tape(*args, **kwargs)

                    def gradients(tape: tf.GradientTape, optimizer: object, loss: tf.Tensor, selected: list) -> list:
                        """Record actual watched variables and returned gradients before comparing updates."""

                        observation["watched"] = {id(value) for value in tape.watched_variables()}
                        observation["selected"] = {
                            id(value if isinstance(value, tf.Variable) else value.value) for value in selected
                        }
                        observation["loss"] = float(loss.numpy())
                        pairs = apply_policy_gradients(tape, optimizer, loss, selected)
                        observation["gradients"] = [tf.convert_to_tensor(gradient).numpy().copy() for gradient, _ in pairs]
                        return pairs

                    # Reset the framework stream in addition to the restored wrapper-owned counters.
                    tf.random.set_seed(431)
                    with patch("semantic_consolidation.phases.tf.GradientTape", side_effect=tape_factory), \
                            patch("semantic_consolidation.phases.apply_policy_gradients", side_effect=gradients):
                        phase.train_step(None)
                    observation["values"] = [value.numpy().copy() for value in variables]
                    observation["predictor"] = [] if phase.predictor is None else [
                        value.numpy().copy() for value in phase.predictor.weights
                    ]
                    observation["trace"] = phase.trace
                    observations.append(observation)
                bounded, reference = observations
                self.assertEqual(bounded["watched"], bounded["selected"])
                self.assertEqual(bounded["loss"], reference["loss"])
                self.assertEqual(bounded["trace"], reference["trace"])
                self.assertEqual(len(bounded["gradients"]), len(reference["gradients"]))
                for key in ("gradients", "values", "predictor"):
                    for actual, expected in zip(bounded[key], reference[key]):
                        np.testing.assert_array_equal(actual, expected)
                # Acquisition and semantic-only consolidation must exclude the live backbone tape.
                if scope == "semantic":
                    self.assertLess(len(bounded["watched"]), len(reference["watched"]))


# Direct invocation preserves the ordinary unittest lifecycle.
if __name__ == "__main__":
    unittest.main()
