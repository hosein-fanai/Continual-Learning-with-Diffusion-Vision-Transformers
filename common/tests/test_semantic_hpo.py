"""Semantic-only search, native recipe preservation and artifact identity checks.

Run these tests only in an authorized remote container. Profile checks do not
train a model; the separate pipeline tests exercise the actual phase adapter.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

import optuna

from common.config import Config, save_config
from common.semantic_hpo import (
    FIELD_CATALOG, PROFILE, SEARCH_SPACE, build_semantic_config, normalize_semantic_profile, 
    reseed_semantic_config, run_semantic_trial, validate_semantic_config, validate_semantic_search
)
from semantic_consolidation.config import RouteSettings, load_route_config


class _Trial:
    """Record native suggestions and select the first permitted value."""

    def __init__(self, number: int = 0) -> None:
        """Create independent parameter and distribution records."""

        self.number = number
        self.params = {}
        self.user_attrs = {}
        self.distributions = {}

    def suggest_categorical(self, name: str, choices: list) -> object:
        """Honor the adapter's resolved categorical domain."""

        self.distributions[name] = list(choices)
        self.params[name] = choices[0]
        return choices[0]

    def suggest_float(self, name: str, low: float, high: float, step: float | None = None, log: bool = False) -> float:
        """Record continuous bounds and return their lower endpoint."""

        self.distributions[name] = {"low": low, "high": high, "step": step, "log": log}
        self.params[name] = low
        return low

    def set_user_attr(self, name: str, value: object) -> None:
        """Preserve metadata through the Optuna-compatible interface."""

        self.user_attrs[name] = value


class SemanticHpoProfileTests(TestCase):
    """Prevent semantic searches from altering their ordinary continual platform."""

    def setUp(self) -> None:
        """Load a native compatible recipe and create private artifact destinations."""

        temporary = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        route = load_route_config("semantic_consolidation/configs/cifar10.yaml")
        self.native = route.common
        self.native.training.epochs = 2
        self.native.model.kwargs["seed"] = 73
        self.native.model.wrapper_kwargs["seed"] = 79
        self.profile = {"student_config": asdict(self.native), "route_settings": asdict(route.route)}

    def build(self, overrides: dict | None = None, profile: dict | None = None, seed: int = 17) -> tuple:
        """Build a deterministic actual profile candidate without executing training."""

        trial = _Trial()
        selected = self.profile if profile is None else profile
        config = build_semantic_config(
            trial, dataset_name=selected["student_config"]["dataset"]["name"], epochs=2, 
            results_path=self.root, semantic_profile=selected, search_space_overrides=overrides, seed=seed
        )
        return trial, config

    def test_catalog_accounts_for_every_native_route_setting(self) -> None:
        """New route fields cannot silently become unclassified or unintended HPO knobs."""

        self.assertEqual(set(FIELD_CATALOG), {field.name for field in fields(RouteSettings)})
        active = {name for name, value in FIELD_CATALOG.items() if value["role"] in ("searched", "conditional")}
        self.assertEqual(active, set(SEARCH_SPACE))

    def test_native_scientific_recipe_and_inputs_remain_unchanged(self) -> None:
        """Architecture, teacher loss, replay, optimizer and joint budgets remain exact."""

        original = deepcopy(self.profile)
        trial, config = self.build()
        self.assertEqual(self.profile, original)
        self.assertEqual(asdict(config.optimizer), asdict(self.native.optimizer))
        self.assertEqual(asdict(config.dataset), asdict(self.native.dataset))
        for key in ("epochs", "fit_method", "fit_kwargs", "patience", "reduce_lr_patience", "deterministic_ops"):
            self.assertEqual(getattr(config.training, key), getattr(self.native.training, key))
        for key in (
            "replay_budget_mode", "replay_old_examples", "replay_current_examples", 
            "use_distillation", "specialist_teacher_descriptors", "generative_model_kwargs", 
            "use_generative_replay", "snapshot_network_name"
        ):
            self.assertEqual(getattr(config.continually_learn, key), getattr(self.native.continually_learn, key))
        for key, value in self.native.model.kwargs.items():
            # The trial seed is an operational initialization control, not a topology change.
            if key != "seed":
                self.assertEqual(config.model.kwargs[key], value)
        for key, value in self.native.model.wrapper_kwargs.items():
            # Teacher and joint objective coefficients are never optimization dimensions.
            if key != "seed":
                self.assertEqual(config.model.wrapper_kwargs[key], value)
        self.assertEqual(config.model.kwargs["seed"], 17)
        self.assertEqual(config.model.wrapper_kwargs["seed"], 17)
        self.assertEqual(config.hpo["search_profile"], PROFILE)
        self.assertTrue(set(trial.params) <= set(SEARCH_SPACE))
        self.assertEqual(config.hpo["objective_metrics"], ["final_average_accuracy"])
        self.assertEqual(config.hpo["objective_directions"], ["maximize"])
        validate_semantic_config(config)

    def test_inactive_augmentation_and_reliability_have_no_suggestions(self) -> None:
        """Clean unaugmented consolidation creates no dummy conditional parameters."""

        trial, config = self.build({"noise_levels": ["0"], "image_augmentation": ["none"]})
        self.assertEqual(config.hpo["semantic_consolidation"]["noise_levels"], tuple([0]))
        for name in ("augmentation_views", "reliability", "reliability_floor"):
            self.assertNotIn(name, trial.params)

    def test_noisy_tmcl_and_alpha_bar_activate_native_conditional_fields(self) -> None:
        """Requested valid noise sets, independent views and weights reach RouteSettings."""

        trial, config = self.build({
            "noise_levels": ["0,100,250"], "image_augmentation": ["tmcl"], 
            "augmentation_views": [8], "reliability": ["alpha_bar"], "reliability_floor": [0.75]
        })
        settings = config.hpo["semantic_consolidation"]
        self.assertEqual(settings["noise_levels"], (0, 100, 250))
        self.assertEqual(settings["augmentation_views"], 8)
        self.assertEqual(settings["reliability_floor"], 0.75)
        self.assertTrue({"augmentation_views", "reliability", "reliability_floor"} <= set(trial.params))

    def test_fixed_true_class_ce_does_not_search_orthogonality(self) -> None:
        """The declared CE acquisition ablation cannot receive an unused separation weight."""

        profile = deepcopy(self.profile)
        profile["route_settings"]["acquisition_objective"] = "true_class_ce"
        trial, config = self.build(profile=profile)
        self.assertNotIn("orthogonality_weight", trial.params)
        self.assertEqual(config.hpo["semantic_consolidation"]["acquisition_objective"], "true_class_ce")
        with self.assertRaisesRegex(ValueError, "inactive"):
            self.build({"orthogonality_weight": {"low": 0.1, "high": 1.0}}, profile=profile)

    def test_unreachable_overrides_and_architecture_changes_fail_before_draws(self) -> None:
        """Misspelled, structural or universally inactive dimensions cannot be ignored."""

        cases = [
            {"depth": [4]}, {"replay_old_examples": [1000]}, 
            {"image_augmentation": ["none"], "augmentation_views": [4]}, 
            {"noise_levels": ["0"], "reliability": ["uniform"]}, 
            {"noise_levels": ["100"], "reliability": ["uniform"], "reliability_floor": [0.5]}, 
            {"acquisition_steps": [0]}, {"alignment_weight": {"low": 0, "high": 1, "log": False}}
        ]
        profile = normalize_semantic_profile(self.profile)
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_semantic_search(profile, overrides)

    def test_fixed_horizon_filters_defaults_and_rejects_explicit_invalid_noise(self) -> None:
        """Absolute schedule indices stay inside the supplied denoising process."""

        profile = deepcopy(self.profile)
        profile["student_config"]["model"]["kwargs"]["timesteps"] = 20
        profile["student_config"]["model"]["wrapper_kwargs"]["test_steps"] = 10
        trial, config = self.build(profile=profile)
        self.assertEqual(trial.distributions["acquisition_noise_level"], [0, 10])
        self.assertEqual(trial.distributions["noise_levels"], ["0", "10"])
        self.assertEqual(config.model.kwargs["timesteps"], 20)
        with self.assertRaisesRegex(ValueError, "Unsupported semantic"):
            self.build({"noise_levels": ["0,50,100"]}, profile=profile)

    def test_mapping_seal_survives_json_but_detects_mutation(self) -> None:
        """Serialized continuation retains native settings and cannot reseal a changed recipe."""

        profile = normalize_semantic_profile(self.profile)
        restored = json.loads(json.dumps(profile))
        repeated = normalize_semantic_profile(restored, seed=99)
        self.assertEqual(repeated["fixed_recipe_sha256"], profile["fixed_recipe_sha256"])
        restored["student_config"]["optimizer"]["initial_learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "sealed native semantic recipe changed"):
            normalize_semantic_profile(restored)

    def test_native_yaml_and_route_settings_files_bind_exact_bytes(self) -> None:
        """Either changed input file invalidates a previously normalized profile."""

        native_path = self.root / "native.yaml"
        settings_path = self.root / "route.yaml"
        save_config(self.native, native_path)
        settings_path.write_text("route:\n  image_augmentation: tmcl\n", encoding="utf-8")
        profile = normalize_semantic_profile({"student_config": native_path, "route_settings": settings_path})
        self.assertEqual(len(profile["artifact_sha256"]), 2)
        settings_path.write_text("route:\n  image_augmentation: none\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            normalize_semantic_profile(profile)

    def test_fresh_confirmation_seeds_keep_partition_and_validation_seed(self) -> None:
        """Model and phase repetition does not resample the scientific comparison cohorts."""

        profile = deepcopy(self.profile)
        profile["student_config"]["continually_learn"]["class_order_mode"] = "random"
        profile["student_config"]["continually_learn"]["task_order_mode"] = "random"
        _, first = self.build(profile=profile, seed=17)
        _, repeated = self.build(profile=first.hpo["semantic_profile"], seed=29)
        self.assertEqual(first.continually_learn.task_groups, repeated.continually_learn.task_groups)
        self.assertEqual(first.hpo["continual_dataset_seed"], self.native.continually_learn.seed)
        self.assertEqual(first.hpo["continual_dataset_seed"], repeated.hpo["continual_dataset_seed"])
        reseed_semantic_config(first, seed=43)
        validate_semantic_config(first)
        self.assertEqual(first.hpo["semantic_consolidation"]["seed"], 43)

    def test_typed_wrapper_reseed_preserves_effective_configuration(self) -> None:
        """Seeding a typed wrapper must not replace it with a one-key generic mapping."""

        profile = deepcopy(self.profile)
        model = profile["student_config"]["model"]
        model["wrapper_kwargs"] = {}
        model["diffusion_classifier"].update({
            "seed": 71, "use_ema": False, "test_network_name": "raw", 
            "test_noisified_min_timesteps": 0, "test_noisified_max_timesteps": 0
        })
        _, config = self.build(profile=profile)
        self.assertEqual(config.model.wrapper_kwargs, {})
        self.assertEqual(config.model.diffusion_classifier.seed, 17)
        reseed_semantic_config(config, seed=101)
        self.assertEqual(config.model.wrapper_kwargs, {})
        self.assertEqual(config.model.diffusion_classifier.seed, 101)
        validate_semantic_config(config)

    def test_native_data_seed_override_is_preserved_independently(self) -> None:
        """A native prior-HPO recipe keeps its existing split under new training seeds."""

        profile = deepcopy(self.profile)
        profile["student_config"]["hpo"]["continual_dataset_seed"] = 203
        _, config = self.build(profile=profile, seed=17)
        self.assertEqual(config.hpo["continual_dataset_seed"], 203)
        self.assertEqual(config.hpo["semantic_profile"]["task_seed"], 41)
        self.assertEqual(config.hpo["semantic_profile"]["dataset_seed"], 203)
        reseed_semantic_config(config, seed=107)
        validate_semantic_config(config)
        self.assertEqual(config.hpo["continual_dataset_seed"], 203)

    def test_explicitly_unseeded_data_partition_is_rejected(self) -> None:
        """A null HPO data seed must not make nominally paired trials use different splits."""

        profile = deepcopy(self.profile)
        profile["student_config"]["hpo"]["continual_dataset_seed"] = None
        with self.assertRaisesRegex(ValueError, "fixed native dataset seed"):
            normalize_semantic_profile(profile)

    def test_cifar100_uses_the_supplied_ten_task_stream(self) -> None:
        """The same semantic search supports CIFAR100 without inventing task groups."""

        route = load_route_config("semantic_consolidation/configs/cifar100.yaml")
        route.common.training.epochs = 2
        trial, config = self.build(profile={"student_config": asdict(route.common), "route_settings": asdict(route.route)})
        self.assertEqual(config.dataset.name, "cifar100")
        self.assertEqual(len(config.continually_learn.task_groups), 10)
        self.assertEqual(config.continually_learn.class_order, list(range(100)))
        self.assertTrue(trial.params)

    def test_changed_frozen_field_or_sampled_evidence_is_rejected(self) -> None:
        """Worker dispatch authenticates both the fixed platform and chosen intervention."""

        _, config = self.build()
        config.continually_learn.replay_old_examples += 1
        with self.assertRaisesRegex(ValueError, "frozen native"):
            validate_semantic_config(config)
        _, config = self.build()
        config.hpo["semantic_consolidation"]["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "parameter evidence changed"):
            validate_semantic_config(config)

    def test_invalid_selection_protocol_and_resume_are_rejected(self) -> None:
        """Official-test selection, wrong treatments and partial-stream recovery fail closed."""

        cases = [
            ("dataset", "validation_source", "test"), 
            ("continually_learn", "experiment_phase", "benchmark"), 
            ("continually_learn", "resume_from", "old/checkpoint"), 
            ("continually_learn", "use_ensemble_accuracy", True), 
            ("model", "weights_path", "old.weights.h5")
        ]
        for section, key, value in cases:
            profile = deepcopy(self.profile)
            profile["student_config"][section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                normalize_semantic_profile(profile)
        profile = deepcopy(self.profile)
        profile["route_settings"]["condition"] = "baseline"
        with self.assertRaisesRegex(ValueError, "paired ablations"):
            normalize_semantic_profile(profile)

    def test_explicit_global_controls_cannot_change_joint_training(self) -> None:
        """A generic API's defaults cannot silently replace native frozen settings."""

        with self.assertRaisesRegex(ValueError, "freezes native epochs"):
            build_semantic_config(_Trial(), "cifar10", 3, self.root, self.profile)
        with self.assertRaisesRegex(ValueError, "freezes native deterministic_ops"):
            build_semantic_config(_Trial(), "cifar10", 2, self.root, self.profile, deterministic_ops=False)
        with self.assertRaisesRegex(ValueError, "requires final_average_accuracy"):
            build_semantic_config(
                _Trial(), "cifar10", 2, self.root, self.profile, 
                objective_metrics=["average_incremental_accuracy"], objective_directions=["maximize"]
            )

    def test_real_optuna_conditional_domains_remain_stable(self) -> None:
        """A single study mixes all conditional branches without dynamic-space errors."""

        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=7))
        for noise, augmentation, reliability in (
            ("0", "none", "uniform"), ("100", "none", "uniform"), 
            ("0,100,250", "tmcl", "alpha_bar"), ("10", "tmcl", "uniform")
        ):
            study.enqueue_trial({"noise_levels": noise, "image_augmentation": augmentation, "reliability": reliability})
        for index in range(4):
            trial = study.ask()
            config = build_semantic_config(trial, "cifar10", 2, self.root, self.profile, seed=17)
            validate_semantic_config(config)
            study.tell(trial, index / 10)
        self.assertEqual(len(study.trials), 4)

    def test_runtime_dispatch_attaches_the_actual_semantic_runner(self) -> None:
        """The named HPO path cannot fall through to ordinary continual training."""

        _, config = self.build()
        expected = {"evaluations": {"validation_continual_metrics": {"final_average_accuracy": 0.5}}}
        with patch("semantic_consolidation.runner.run", return_value=expected) as runner:
            self.assertIs(run_semantic_trial(config), expected)
        passed = runner.call_args.args[0]
        self.assertIs(passed.common, config)
        self.assertEqual(passed.route.condition, "learned")
        self.assertEqual(passed.route.seed, config.training.seed)

    def test_profile_import_and_normalization_do_not_import_tensorflow(self) -> None:
        """Notebook preflight remains free of framework/device initialization."""

        path = self.root / "native.yaml"
        save_config(self.native, path)
        source = (
            "import sys\nfrom common.semantic_hpo import normalize_semantic_profile\n\n\n"
            "normalize_semantic_profile({'student_config': sys.argv[1]})\n"
            "assert 'tensorflow' not in sys.modules\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", source, str(path)], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


# Direct execution uses the same assertions and independent unittest process.
if __name__ == "__main__":
    main()
