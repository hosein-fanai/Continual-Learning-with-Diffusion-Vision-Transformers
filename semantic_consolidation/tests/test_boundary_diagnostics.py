"""Task-boundary diagnostics on permitted, fixed held-out examples.

Controlled feature changes distinguish immutable gate parameters from their
function. These checks do not train a benchmark or assert method efficacy.
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier
from semantic_consolidation.config import RouteSettings
from semantic_consolidation.controller import RouteController, weight_digest
from semantic_consolidation.diagnostics import class_geometry, one_vs_rest
from semantic_consolidation.memory import ModulationBank
from semantic_consolidation.model import adapt_model
from semantic_consolidation.tests.test_phases import _make_wrapper


class BoundaryAdapterTests(unittest.TestCase):
    """The route hook surrounds the existing fit, including opt-in scheduling."""

    def test_probe_precedes_joint_and_section10_extensions(self) -> None:
        """Observe pre-joint state before either common or extension fitting."""

        for use_extensions in (False, True):
            with self.subTest(extensions=use_extensions):
                events = []
                dataset = tf.data.Dataset.from_tensor_slices((
                    np.zeros((4, 4, 4, 1), dtype="float32"),
                    np.asarray([0, 0, 1, 1], dtype="int32"),
                )).batch(4)
                validation = tf.data.Dataset.from_tensor_slices((
                    np.ones((4, 4, 4, 1), dtype="float32"),
                    np.asarray([0, 0, 1, 1], dtype="int32"),
                )).batch(4)

                def before_joint(wrapper: object, supplied: object) -> None:
                    """Record the pre-joint hook and verify its held-out data source."""

                    self.assertIs(supplied, validation)
                    events.append("pre_joint")

                def route_run(wrapper: object, supplied: object, kwargs: dict) -> None:
                    """Verify the existing fit's exposure and update count reach the route."""

                    self.assertIs(supplied, dataset)
                    self.assertEqual(kwargs["route_joint_updates"], 2)
                    events.append("route")

                def joint_fit(wrapper: object, *args: object, **kwargs: object) -> object:
                    """Stand in for two optimizer applications while recording fit order."""

                    events.append("joint")
                    wrapper.optimizer.iterations.assign_add(2)
                    return SimpleNamespace(history={})

                def extension_fit(wrapper: object, supplied: object, kwargs: dict,
                                  fit_function: object) -> tuple:
                    """Forward extension scheduling through the supplied common fit."""

                    events.append("extension_fit")
                    return fit_function(x=supplied, **kwargs), supplied

                def after_task(wrapper: object, supplied: object) -> None:
                    """Record extension completion after the route intervention."""

                    self.assertIs(supplied, validation)
                    events.append("extension_after")

                controller = SimpleNamespace(before_joint=before_joint, run=route_run)
                extensions = SimpleNamespace(fit=extension_fit, after_task=after_task) \
                    if use_extensions else None
                wrapper = adapt_model(_make_wrapper(), controller, extensions=extensions)
                with patch.object(DiffusionClassifier, "fit", joint_fit):
                    wrapper.fit(dataset, validation_data=validation, epochs=1, verbose=0)
                expected = ["pre_joint", "joint", "route"]
                # Enabled extensions must retain their existing fit and completion hooks.
                if use_extensions:
                    expected = ["pre_joint", "extension_fit", "joint", "route", "extension_after"]
                self.assertEqual(events, expected)


class FunctionalGeometryTests(unittest.TestCase):
    """A fixed affine gate does not preserve class geometry under feature drift."""

    def test_small_boundary_cohort_keeps_mse_without_degenerate_cka(self) -> None:
        """Phase observations export actual counts and explicit unavailable reasons."""
        endpoint = {"unmodulated": {"class_geometry": {}}, "gates": {}}
        before, after = np.asarray([[1., 0.], [-1., 0.]]), np.asarray([[0., 10.], [0., -10.]])
        change = RouteController._functional_change(endpoint, endpoint, before, after)
        self.assertIsNone(change["hidden_feature_cka"])
        self.assertEqual(change["hidden_feature_cka_sample_count"], 2)
        self.assertEqual(change["hidden_feature_cka_unavailable_reason"], "fewer_than_three_aligned_observations")
        self.assertGreater(change["hidden_mean_squared_change"], 0.)
        missing = RouteController._functional_change(endpoint, endpoint, None, after)
        self.assertIsNone(missing["hidden_feature_cka_sample_count"])
        self.assertEqual(missing["hidden_feature_cka_unavailable_reason"], "boundary_or_validation_unavailable")

    def test_fixed_gate_can_lose_orthogonal_class_separation(self) -> None:
        """Identical gate hashes coexist with a measured loss of old-gate function."""

        settings = RouteSettings(modulation_init_std=0., seed=41)
        bank = ModulationBank(settings, dimension=2, seed=41)
        bank.add([0])
        labels = np.asarray([0, 0, 1, 1], dtype="int32")
        initial = np.asarray([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], dtype="float32")
        collapsed = np.ones_like(initial)
        variables = list(bank.vectors[0])
        digest = weight_digest(variables)
        first = one_vs_rest(bank.apply(initial, 0).numpy(), labels, 0)
        last = one_vs_rest(bank.apply(collapsed, 0).numpy(), labels, 0)
        self.assertEqual(weight_digest(variables), digest)
        self.assertAlmostEqual(first["one_vs_rest_squared_cosine"], 0., places=6)
        self.assertAlmostEqual(last["one_vs_rest_squared_cosine"], 1., places=6)
        self.assertAlmostEqual(first["within_class_cosine_distance"], 0., places=6)
        self.assertAlmostEqual(last["within_class_cosine_distance"], 0., places=6)
        geometry = class_geometry(collapsed, labels, old={0})
        self.assertAlmostEqual(geometry["all"]["between_class_cosine_distance"], 0., places=6)

    def test_unavailable_old_gate_positives_are_reported_missing(self) -> None:
        """Absent held-out historical classes must not yield invented separation."""

        result = one_vs_rest(np.eye(2, dtype="float32"), np.asarray([2, 3]), focus=0)
        self.assertEqual(result["positive_examples"], 0)
        self.assertEqual(result["negative_examples"], 2)
        self.assertIsNone(result["within_class_cosine_distance"])
        self.assertIsNone(result["one_vs_rest_squared_cosine"])
        self.assertEqual(result["availability"], "insufficient_positive_or_negative_rows")


class FunctionalBoundaryTests(unittest.TestCase):
    """Exercise controller boundary attribution with explicitly controlled features."""

    def test_missing_validation_and_discarded_gates_remain_unavailable(self) -> None:
        """A boundary without permitted images does not infer historical gate behavior."""

        settings = RouteSettings(condition="baseline", seed=41)
        controller = RouteController(settings)
        controller.introduced = {0, 1}
        controller.bank = ModulationBank(settings, dimension=2, seed=41)
        controller.bank.add([0])
        wrapper = SimpleNamespace(network=SimpleNamespace(num_classes=2), seen_classes={0: 0, 1: 1})
        controller.before_joint(wrapper, None)
        record = controller.run(wrapper, None, {})
        drift = record["functional_drift"]
        self.assertEqual(drift["unavailable_old_gate_ids"], [1])
        self.assertEqual(drift["pre_joint"]["availability"], "validation_unavailable")
        self.assertEqual(drift["post_joint"]["availability"], "validation_unavailable")
        self.assertEqual(drift["joint_change"]["availability"], "boundary_or_validation_unavailable")
        self.assertEqual(drift["pre_joint"]["gates"], {})

    def test_old_gate_mutation_during_joint_is_rejected(self) -> None:
        """Parameter preservation starts before joint fitting, not after it."""

        settings = RouteSettings(condition="baseline", seed=41)
        controller = RouteController(settings)
        controller.introduced = {0}
        controller.bank = ModulationBank(settings, dimension=2, seed=41)
        controller.bank.add([0])
        wrapper = SimpleNamespace(network=SimpleNamespace(num_classes=2), seen_classes={0: 0, 1: 1})
        controller.before_joint(wrapper, None)
        controller.bank.vectors[0][0].assign_add(tf.ones(2))
        with self.assertRaisesRegex(RuntimeError, "Joint training changed old modulation parameters"):
            controller.run(wrapper, None, {})

    def test_joint_and_consolidation_drift_use_fixed_rows_without_extra_teacher(self) -> None:
        """Hash-stable old gates lose separation in joint fitting and recover later."""

        settings = RouteSettings(
            acquisition_steps=0, consolidation_steps=1, batch_size=4,
            probe_batches=4, modulation_init_std=0., noise_levels=(0,), seed=41,
        )
        controller = RouteController(settings)
        controller.introduced = {0, 1}
        controller.bank = ModulationBank(settings, dimension=2, seed=41)
        controller.bank.add([0, 1])
        initial = np.repeat(np.asarray(
            [[1., 0.], [0., 1.], [-1., 0.], [0., -1.]], dtype="float32",
        ), 4, axis=0)

        def network(values: np.ndarray, trainable: bool) -> object:
            """Build a controllable projection lookup with explicit trainable state."""

            lookup = tf.Variable(values, trainable=trainable, name="diagnostic_projection")
            classifier = SimpleNamespace(weights=[lookup], trainable_variables=[lookup])
            return SimpleNamespace(
                lookup=lookup, weights=[lookup], trainable_variables=[lookup],
                classifier=classifier, num_classes=4, trainable=trainable,
            )

        snapshots = []
        wrapper = SimpleNamespace(
            network=network(initial, True), teacher_network=None,
            optimizer=tf.keras.optimizers.Adam(), seen_classes={i: i for i in range(4)},
        )

        def snapshot(branch: str) -> object:
            """Count independent frozen targets while copying the current projection."""

            self.assertEqual(branch, "raw")
            target = network(wrapper.network.lookup.numpy(), False)
            snapshots.append(target)
            return target

        def features(current: object, images: tf.Tensor, times: tf.Tensor,
                     stop_backbone: bool = False) -> tuple:
            """Return prescribed row features and fixed, known classifier predictions."""

            ids = tf.cast(tf.reshape(images, (len(images), -1))[:, 0], tf.int32)
            hidden = tf.gather(current.lookup, ids)
            return hidden, tf.one_hot(ids // 4, 4)

        wrapper.snapshot_teacher_network = snapshot
        images = np.arange(16, dtype="float32").reshape(-1, 1)
        labels = np.repeat(np.arange(4, dtype="int32"), 4)
        dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(4)
        original_fit = controller._fit
        original_functional = controller._functional
        cohorts = []

        def functional(current: object, probe: tuple, old: set[int], gates: list[int]) -> tuple:
            """Record each measured cohort before running the actual functional probe."""

            cohorts.append(tuple(value.copy() for value in probe))
            return original_functional(current, probe, old, gates)

        def controlled_fit(phase: object, steps: int, random_control: bool = False) -> dict:
            """Prescribe a consolidation change without running an optimizer study."""

            result = original_fit(phase, 0, random_control)
            # Only the consolidation boundary restores the prescribed initial geometry.
            if phase.phase == "consolidation":
                wrapper.network.lookup.assign(initial)
                result["updates"] = 1
                result["gradient_variable_names"] = [wrapper.network.lookup.name]
            return result

        old_variables = [v for pair in controller.bank.vectors.values() for v in pair]
        old_digest = weight_digest(old_variables)
        with patch("semantic_consolidation.controller.semantic_features", features), \
                patch.object(controller, "_functional", side_effect=functional):
            controller.before_joint(wrapper, dataset)
            self.assertEqual(snapshots, [])
            # This explicitly prescribed collapse stands in for joint-training drift.
            wrapper.network.lookup.assign(np.ones_like(initial))
            with patch.object(controller, "_fit", side_effect=controlled_fit):
                record = controller.run(wrapper, dataset, {
                    "validation_data": dataset, "route_joint_updates": 2,
                })
        self.assertEqual(len(snapshots), 1, "Only the existing consolidation target is cloned.")
        self.assertEqual(weight_digest(old_variables), old_digest)
        self.assertEqual(len(cohorts), 3)
        for cohort in cohorts[1:]:
            np.testing.assert_array_equal(cohort[0], cohorts[0][0])
            np.testing.assert_array_equal(cohort[1], cohorts[0][1])
        self.assertIsNone(controller._boundary, "Bounded diagnostic arrays are released after the task.")
        drift = record["functional_drift"]
        self.assertTrue(drift["old_gate_parameters_unchanged_since_pre_joint"])
        before = drift["pre_joint"]["gates"]["0"]["modulated"]["one_vs_rest_squared_cosine"]
        joint = drift["post_joint"]["gates"]["0"]["modulated"]["one_vs_rest_squared_cosine"]
        final = drift["post_consolidation"]["gates"]["0"]["modulated"]["one_vs_rest_squared_cosine"]
        self.assertAlmostEqual(before, 1. / 3., places=6)
        self.assertAlmostEqual(joint, 1., places=6)
        self.assertAlmostEqual(final, before, places=6)
        self.assertAlmostEqual(
            drift["joint_change"]["gates"]["0"]["modulated"]["one_vs_rest_squared_cosine"],
            joint - before, places=6,
        )
        self.assertAlmostEqual(
            drift["consolidation_change"]["gates"]["0"]["modulated"]["one_vs_rest_squared_cosine"],
            final - joint, places=6,
        )
        self.assertGreater(drift["joint_change"]["hidden_mean_squared_change"], 0.)
        self.assertLess(drift["joint_change"]["class_geometry"]["old"]["between_class_cosine_distance"], 0.)
        self.assertGreater(drift["consolidation_change"]["class_geometry"]["old"]["between_class_cosine_distance"], 0.)


# Direct execution runs only this bounded diagnostic regression module.
if __name__ == "__main__":
    unittest.main()
