"""Executable control recipes and bounded synthetic gradient checks.

Timing numbers below are explicitly invented validation fixtures, never pilot
measurements or thesis outcomes. Preparation tests do not load benchmark data.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from common.experiment import materialize_run_plan, read_experiment_manifest
from semantic_consolidation.config import RouteSettings, load_route_config
from semantic_consolidation.controller import RouteController
from semantic_consolidation.controls import control_conditions, measured_allowances, prepare_controls
from semantic_consolidation.memory import ClassBalancedPool, ModulationBank
from semantic_consolidation.phases import RoutePhase, semantic_features
from semantic_consolidation.study import prepare_study, validate_planned_config
from semantic_consolidation.tests.test_phases import _make_wrapper


_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class ControlConfigurationTests(unittest.TestCase):
    """Every condition binds the same stream and declared phase/data budgets."""

    def setUp(self) -> None:
        """Allocate isolated preparation artifacts and load the bounded template."""

        self.temporary = tempfile.TemporaryDirectory(prefix="route_SYNTHETIC_controls_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.template = load_route_config(_CONFIGS / "smoke.yaml")

    def _timing_fixture(self) -> Path:
        """Write explicitly synthetic phase timings for input-validation tests."""

        records = [
            {"fixture_kind": "SYNTHETIC_UNIT_TEST_NOT_MEASURED_TIMING", "task": task,
             "condition": "learned", "acquisition": {"updates": 2, "seconds": 1.0},
             "consolidation": {"updates": 1, "seconds": 2.0}, "target_snapshot_seconds": 0.25}
            for task in (1, 2)
        ]
        path = self.directory / "SYNTHETIC_timing_fixture.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        return path

    def test_standalone_control_yaml_matches_the_existing_recipe(self) -> None:
        """Standalone YAMLs change only the corresponding existing control knobs."""

        base = load_route_config(_CONFIGS / "cifar10.yaml")
        for name, changes in control_conditions().items():
            with self.subTest(condition=name):
                config = load_route_config(_CONFIGS / "controls" / f"{name}.yaml")
                expected = replace(base.route, **changes["route"])
                self.assertEqual(asdict(config.route), asdict(expected))
                self.assertEqual(asdict(config.common), asdict(base.common))

    def test_prepare_materializes_all_controls_with_paired_budgets_and_exposure(self) -> None:
        """Generated control runs preserve stream, model, data, and phase settings."""

        path = prepare_controls(self.template, self.directory / "design", [17, 29])
        plan = materialize_run_plan(read_experiment_manifest(path))
        self.assertEqual(len(plan), 10)
        grouped = {}
        for entry in plan:
            grouped.setdefault(entry["block_id"], []).append(entry)
            config = load_route_config(path.parent / f"{entry['run_id']}.yaml")
            validate_planned_config(config)
            self.assertEqual(config.common.training.seed, entry["stream"]["stream_seed"])
            self.assertEqual(config.route.seed, entry["stream"]["stream_seed"])
            self.assertEqual(config.common.continually_learn.class_order, entry["stream"]["class_order"])
            self.assertEqual(asdict(config.common.dataset), asdict(self.template.common.dataset))
            # The existing JSON manifest serializes integer mapping keys as strings.
            self.assertEqual(json.loads(json.dumps(asdict(config.common.model))),
                             json.loads(json.dumps(asdict(self.template.common.model))))
            for field in ("replay_budget_mode", "replay_current_examples", "replay_old_examples"):
                self.assertEqual(getattr(config.common.continually_learn, field),
                                 getattr(self.template.common.continually_learn, field))
            expected = replace(self.template.route, seed=entry["stream"]["stream_seed"],
                               **control_conditions()[entry["condition"]]["route"])
            self.assertEqual(asdict(config.route), asdict(expected))
            self.assertEqual(config.common.continually_learn.experiment_phase, "development")
        for entries in grouped.values():
            self.assertEqual({entry["condition"] for entry in entries}, set(control_conditions()))
            self.assertTrue(all(entry["stream"] == entries[0]["stream"] for entry in entries))
        self.assertFalse((path.parent / "timing_basis.json").exists())
        self.assertFalse((path.parent / "completed_runs.json").exists())

    def test_noise_recipes_preserve_clean_default_and_pair_only_declared_knobs(self) -> None:
        """Three ordinary overrides retain clean acquisition/CE and matched streams.

        Returns:
            checked (None): None; verifies shipped YAML settings, unchanged common
                configuration, paired stream seeds, and the declared primary contrast.

        Raises:
            AssertionError: If a recipe changes unrelated settings or breaks pairing.
            ValueError: If a generated supported configuration fails validation.
        """
        base = load_route_config(_CONFIGS / "cifar10.yaml")
        self.assertEqual(base.route.noise_levels, (0,))
        recipes = {
            "noisy_weighted": {"noise_levels": (0, 50, 150), "reliability": "alpha_bar"},
            "noisy_uniform": {"noise_levels": (0, 50, 150), "reliability": "uniform"},
            "clean_uniform": {"noise_levels": (0,), "reliability": "uniform"},
        }
        for name, changes in recipes.items():
            config = load_route_config(_CONFIGS / "controls" / f"{name}.yaml")
            self.assertEqual(asdict(config.route), asdict(replace(base.route, **changes)))
            self.assertEqual(asdict(config.common), asdict(base.common))
        # The tiny template uses its own valid timestep range with identical comparisons.
        conditions = {name: {"route": {**changes, "noise_levels": (0, 2) if name != "clean_uniform" else (0,)}}
                      for name, changes in recipes.items()}
        path = prepare_study(self.template, self.directory / "noise_design", [17, 29], conditions=conditions)
        manifest = read_experiment_manifest(path)
        self.assertEqual(manifest["spec"]["analysis_spec"]["condition_a"], "noisy_weighted")
        self.assertEqual(manifest["spec"]["analysis_spec"]["condition_b"], "noisy_uniform")
        plan = materialize_run_plan(manifest)
        self.assertEqual(len(plan), 6)
        for entry in plan:
            config = load_route_config(path.parent / f"{entry['run_id']}.yaml")
            validate_planned_config(config)
            expected = replace(self.template.route, seed=entry["stream"]["stream_seed"],
                               **conditions[entry["condition"]]["route"])
            self.assertEqual(asdict(config.route), asdict(expected))
            self.assertEqual(config.route.acquisition_noise_level, 0)
            self.assertEqual(config.route.ce_noise_level, 0)

    def test_optional_time_control_uses_supplied_durations_and_exports_their_origin(self) -> None:
        """Supplied fixture durations enter the manifest with recorded provenance."""

        fixture = self._timing_fixture()
        path = prepare_controls(self.template, self.directory / "timed_design", [17, 29],
                                timing_records=fixture)
        plan = materialize_run_plan(read_experiment_manifest(path))
        self.assertEqual(len(plan), 12)
        for entry in plan:
            config = load_route_config(path.parent / f"{entry['run_id']}.yaml")
            validate_planned_config(config)
            # Only the time control consumes the measured per-task allowances.
            if entry["condition"] == "time_matched_joint":
                self.assertEqual(config.route.extra_joint_seconds, (3.25, 3.25))
        basis = json.loads((path.parent / "timing_basis.json").read_text(encoding="utf-8"))
        self.assertEqual(basis["timing_records_path"], str(fixture.resolve()))
        self.assertEqual(len(basis["timing_records_sha256"]), 64)
        self.assertEqual(basis["allowance_seconds_per_task"], [3.25, 3.25])

    def test_time_source_rejects_incomplete_foreign_or_unmeasured_phase_records(self) -> None:
        """Invalid timing sources cannot silently define a comparison budget."""

        fixture = self._timing_fixture()
        original = json.loads(fixture.read_text(encoding="utf-8"))
        cases = {"missing_task": original[:1]}
        for name, key, value in (("wrong_condition", "condition", "random"),
                                 ("unordered_task", "task", 2),
                                 ("nonfinite", "target_snapshot_seconds", float("nan")),
                                 ("negative", "target_snapshot_seconds", -1.0)):
            changed = deepcopy(original)
            changed[0][key] = value
            cases[name] = changed
        changed = deepcopy(original)
        changed[0]["acquisition"]["updates"] = 20
        cases["unmatched_phase_budget"] = changed
        changed = deepcopy(original)
        del changed[0]["target_snapshot_seconds"]
        cases["missing_measurement"] = changed
        for name, records in cases.items():
            with self.subTest(kind=name):
                fixture.write_text(json.dumps(records), encoding="utf-8")
                with self.assertRaises(ValueError):
                    measured_allowances(self.template, fixture)


class ControlExecutionTests(unittest.TestCase):
    """Existing zero-rate and CE-only paths implement the named controls."""

    def setUp(self) -> None:
        """Build the shared tiny DiT fixture and an isolated synthetic image pool."""

        self.previous_policy = tf.keras.mixed_precision.global_policy().name
        self.wrapper = _make_wrapper()
        self.images = np.random.default_rng(103).normal(size=(8, 4, 4, 1)).astype("float32")
        self.pool = ClassBalancedPool(self.images, np.repeat([0, 1], 4).astype("int32"))
        hidden, _ = semantic_features(self.wrapper.network, self.images, tf.zeros(8, tf.int32))
        self.dimension = int(hidden.shape[1])

    def tearDown(self) -> None:
        """Release Keras state and restore the caller's dtype policy."""

        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy(self.previous_policy)

    def test_identity_is_exact_and_survives_zero_rate_acquisition_with_infonce(self) -> None:
        """The existing random-zero initialization implements identity InfoNCE."""

        settings = RouteSettings(condition="random", modulation_init_std=0.0, batch_size=4,
                                 noise_levels=(0,), acquisition_steps=2, consolidation_steps=1)
        bank = ModulationBank(settings, self.dimension, 107)
        bank.add([0, 1])
        probe = tf.constant(np.random.default_rng(109).normal(size=(5, self.dimension)), tf.float32)
        acquisition = RoutePhase(self.wrapper, bank, self.pool, settings, "acquisition", [0, 1], 113)
        controller = RouteController(settings)
        acquired = controller._fit(acquisition, 2, random_control=True)
        self.assertEqual(acquired["updates"], 2)
        self.assertTrue(acquired["zero_learning_rate_control"])
        for class_id, variables in bank.vectors.items():
            np.testing.assert_array_equal(bank.apply(probe, class_id).numpy(), probe.numpy())
            for variable in variables:
                np.testing.assert_array_equal(variable.numpy(), np.zeros(variable.shape))
        target = self.wrapper.snapshot_teacher_network("raw")
        phase = RoutePhase(self.wrapper, bank, self.pool, settings, "consolidation", [0, 1],
                           127, target, bank.frozen())
        record = controller._fit(phase, 1)
        row = record["trace"][0]
        self.assertGreater(row["semantic_loss"], 0.0)
        self.assertAlmostEqual(row["loss"], settings.ce_weight * row["ce"] +
                               settings.alignment_weight * row["semantic_loss"], places=5)

    def test_replacement_ce_steps_have_zero_predictor_update_and_the_declared_ce_loss(self) -> None:
        """CE-only updates retain counted semantic work without predictor changes."""

        settings = RouteSettings(condition="no_consolidation", batch_size=4, ce_weight=0.7,
                                 alignment_weight=5.0, noise_levels=(0, 2))
        bank = ModulationBank(settings, self.dimension, 131)
        bank.add([0, 1])
        phase = RoutePhase(self.wrapper, bank, self.pool, settings, "consolidation", [0, 1],
                           137, self.wrapper.snapshot_teacher_network("raw"), bank.frozen())
        before = [variable.numpy().copy() for variable in phase.predictor.weights]
        record = RouteController(settings)._fit(phase, 2)
        self.assertEqual(record["updates"], 2)
        self.assertEqual(record["supervised_view_draws"], record["example_draws"])
        self.assertEqual(record["semantic_image_noise_draws"], 2 * record["example_draws"])
        for row in record["trace"]:
            self.assertAlmostEqual(row["loss"], settings.ce_weight * row["ce"], places=6)
        for actual, expected in zip(phase.predictor.weights, before):
            np.testing.assert_array_equal(actual.numpy(), expected)


# Direct execution runs only this bounded control regression module.
if __name__ == "__main__":
    unittest.main()
