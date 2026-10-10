"""Fixed DiT continual treatments, artifact identity and conditional search checks.

Artifacts are opaque fixture bytes until the real worker loading boundary. These
tests inspect native Config recipes and Optuna suggestions without training.
Execute only in an authorized remote container.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from itertools import product
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import optuna

from common.config import Config, save_config
from common.dit_continual_hpo import (
    SEARCH_SPACE, baseline_hints, build_dit_continual_config, normalize_continual_profile, validate_continual_search
)


class _Trial:
    """Record native suggestions while choosing the first permitted value."""

    def __init__(self, number: int = 0) -> None:
        """Initialize independent parameters and distribution traces."""

        self.number = number
        self.params = {}
        self.user_attrs = {}
        self.distributions = {}

    def suggest_categorical(self, name: str, choices: list) -> object:
        """Honor the adapter's resolved categorical restriction exactly."""

        self.distributions[name] = list(choices)
        self.params[name] = choices[0]
        return choices[0]

    def suggest_float(self, name: str, low: float, high: float, step: float | None = None, log: bool = False) -> float:
        """Record all numeric bounds and choose their lower endpoint."""

        self.distributions[name] = {"low": low, "high": high, "step": step, "log": log}
        self.params[name] = low
        return low

    def set_user_attr(self, name: str, value: object) -> None:
        """Retain profile metadata through the normal Optuna-compatible method."""

        self.user_attrs[name] = value


class DitContinualProfileTests(TestCase):
    """Keep treatment routing independent from student architecture and fixed losses."""

    def setUp(self) -> None:
        """Create native input recipes and hashable files without constructing models."""

        temporary = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        classifier = self.root / "classifier.keras"
        classifier.write_bytes(b"opaque compiled classifier fixture")
        noise_weights = self.root / "noise.weights.h5"
        noise_weights.write_bytes(b"opaque unet weights fixture")
        noise_config = self.root / "noise.yaml"
        save_config(Config(
            model={
                "name": "unet", "wrapper_name": "diffusion_model", "weights_path": noise_weights.name, 
                "wrapper_kwargs": {"use_ema": False, "test_network_name": "raw"}
            }, 
            training={"task": "generation"}
        ), noise_config)
        student_weights = self.root / "student.weights.h5"
        student_weights.write_bytes(b"opaque student weights fixture")
        self.raw = {
            "image_size": 32, "channels": 3, "timesteps": 1000, "class_num": 2, 
            "use_cfg": True, "dim": 32, "depth": 2, "clf_depth": 2, "patch_size": 4, 
            "mha_num_heads": 4, "classifier_only_cls_token": True, 
            "classifier_only_distil_token": True, "classifier_dropout_rate": 0.125, 
            "feature_aggregation_ids_dict": {1: [-1]}
        }
        self.wrapper = {
            "scheduler_name": "clipped_cosine", "clf_loss_coef": 0.023, 
            "clf_distil_loss_coef": 0.007, "noise_distil_loss_coef": 0.41, 
            "clf_distil_type": "soft", "clf_distil_temperature": 3.2, 
            "clf_acc_coef": 0.61, "clf_distil_acc_coef": 0.29, 
            "ctr_acc_coef": 0.10, "ctr_loss_coef": 0.003, 
            "previous_teacher_clf_loss_weight": 0.25, "current_teacher_clf_loss_weight": 0.75, 
            "previous_teacher_noise_loss_weight": 0.6, "current_teacher_noise_loss_weight": 0.4
        }
        self.student = Config(model={
            "name": "dit_classifier", "wrapper_name": "diffusion_classifier", 
            "kwargs": deepcopy(self.raw), "wrapper_kwargs": deepcopy(self.wrapper), 
            "weights_path": str(student_weights)
        })
        self.profile = {
            "student_config": asdict(self.student), "task_seed": 42, 
            "specialist_teacher_descriptors": {
                "classifier": {"format": "keras", "path": str(classifier)}, 
                "noise": {"format": "config", "path": str(noise_config)}
            }
        }

    def build(self, overrides: dict | None = None, profile: dict | None = None, seed: int = 17) -> tuple:
        """Build one deterministic candidate from the actual profile implementation."""

        trial = _Trial()
        config = build_dit_continual_config(
            trial, dataset_name="cifar10", epochs=2, results_path=self.root, 
            continual_profile=self.profile if profile is None else profile, 
            search_space_overrides=overrides, seed=seed
        )
        return trial, config

    def test_all_sixteen_treatments_keep_architecture_and_declared_coefficients(self) -> None:
        """Only explicitly disabled teacher terms are gated; active losses retain input values."""

        original = deepcopy(self.profile)
        sources = ("none", "previous", "current", "both")
        for clf_source, noise_source in product(sources, repeat=2):
            with self.subTest(classifier=clf_source, noise=noise_source):
                trial, config = self.build({
                    "classifier_teacher_source": [clf_source], "noise_teacher_source": [noise_source]
                })
                wrapper = config.model.wrapper_kwargs
                self.assertEqual(config.model.kwargs, self.raw)
                self.assertEqual(config.model.weights_path, self.student.model.weights_path)
                self.assertEqual(config.model.wrapper_name, "diffusion_classifier")
                self.assertEqual(wrapper["clf_loss_coef"], self.wrapper["clf_loss_coef"])
                for key in ("clf_distil_type", "clf_distil_temperature", "clf_acc_coef", "ctr_acc_coef", "ctr_loss_coef", "scheduler_name"):
                    self.assertEqual(wrapper[key], self.wrapper[key])
                    self.assertNotIn(key, trial.params)
                for head, source, loss_key in (
                    ("classifier", clf_source, "clf_distil_loss_coef"), 
                    ("noise", noise_source, "noise_distil_loss_coef")
                ):
                    self.assertEqual(wrapper[loss_key], self.wrapper[loss_key] if source != "none" else 0.0)
                    role_head = "clf" if head == "classifier" else "noise"
                    for role in ("previous", "current"):
                        key = f"{role}_teacher_{role_head}_loss_weight"
                        expected = self.wrapper[key] if source in (role, "both") else 0.0
                        self.assertEqual(wrapper[key], expected)
                    self.assertEqual(head in config.continually_learn.specialist_teacher_descriptors, source in ("current", "both"))
                previous = clf_source in ("previous", "both") or noise_source in ("previous", "both")
                current = clf_source in ("current", "both") or noise_source in ("current", "both")
                self.assertEqual(config.continually_learn.use_distillation, previous)
                self.assertEqual(wrapper["defer_teacher"], previous)
                self.assertEqual(wrapper["trainable_teacher"], current)
                self.assertEqual(wrapper["teacher_training"], "each_task")
                self.assertEqual(wrapper["dual_teacher_scope"], "task")
                self.assertEqual(wrapper["clf_distil_acc_coef"], self.wrapper["clf_distil_acc_coef"] if clf_source != "none" else 0.0)
                self.assertFalse(wrapper["use_ema"])
                self.assertEqual(config.continually_learn.snapshot_network_name, "raw")
                self.assertEqual(config.hpo["objective_metrics"], ["final_average_accuracy"])
                self.assertEqual(config.hpo["objective_directions"], ["maximize"])
        self.assertEqual(self.profile, original)

    def test_five_two_class_tasks_and_validation_seed_are_common_across_model_seeds(self) -> None:
        """Changing initialization leaves the sealed task partition and validation cohort intact."""

        normalized = normalize_continual_profile(self.profile, seed=17)
        repeated = normalize_continual_profile(normalized, seed=99)
        self.assertEqual(normalized, repeated)
        self.assertEqual(len(normalized["task_groups"]), 5)
        self.assertTrue(all(len(group) == 2 for group in normalized["task_groups"]))
        self.assertEqual(sorted(normalized["class_order"]), list(range(10)))
        for seed in (17, 99):
            _, config = self.build(profile=normalized, seed=seed)
            self.assertEqual(config.continually_learn.task_groups, normalized["task_groups"])
            self.assertEqual(config.continually_learn.class_order, normalized["class_order"])
            self.assertEqual(config.continually_learn.seed, seed)
            self.assertEqual(config.hpo["continual_dataset_seed"], 42)
            self.assertEqual(config.training.seed, seed)
            self.assertEqual(config.dataset.validation_source, "split")
            self.assertEqual(config.dataset.validation_ratio, 0.2)
            self.assertFalse(config.dataset.drop_remainder)

    def test_real_optuna_all_source_pairs_keep_conditional_distributions_stable(self) -> None:
        """A single study can sample every treatment without dynamic categorical spaces."""

        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3))
        hints = baseline_hints()
        self.assertEqual(len(hints), 16)
        for hint in hints:
            study.enqueue_trial(hint)
        observed = set()
        for index in range(16):
            trial = study.ask()
            config = build_dit_continual_config(
                trial, dataset_name="cifar10", epochs=1, results_path=self.root, 
                continual_profile=self.profile, seed=17
            )
            clf_source = config.hpo["classifier_teacher_source"]
            noise_source = config.hpo["noise_teacher_source"]
            observed.add((clf_source, noise_source))
            # A previous-task source must retain old rows in its sampled protocol.
            if clf_source in ("previous", "both") or noise_source in ("previous", "both"):
                self.assertIn(config.hpo["continual_strategy"], ("generative_replay", "cumulative"))
            study.tell(trial, index / 100)
        self.assertEqual(observed, set(product(SEARCH_SPACE["classifier_teacher_source"], SEARCH_SPACE["noise_teacher_source"])))

    def test_replay_and_optimizer_dimensions_are_conditional(self) -> None:
        """Inactive replay, decay, clipping and ranking controls create no dummy parameters."""

        sources = {"classifier_teacher_source": ["none"], "noise_teacher_source": ["none"]}
        trial, config = self.build(dict(sources, continual_strategy=["new_only"], optimizer=["adam"], clipnorm=[1.0]))
        self.assertFalse(config.continually_learn.use_generative_replay)
        self.assertTrue(config.continually_learn.remove_prev_classes)
        for key in (
            "replay_budget_mode", "replay_samples", "replay_old_examples", "replay_current_examples", 
            "replay_selection", "replay_candidate_multiplier", "replay_surprise_weight", 
            "test_steps_t1000", "test_cfg_scale", "test_eta", "weight_decay", "global_clipnorm"
        ):
            self.assertNotIn(key, trial.params)
        self.assertEqual(config.optimizer.schedule, "constant")
        trial, config = self.build(dict(
            sources, continual_strategy=["generative_replay"], replay_budget_mode=["fixed_total"], 
            replay_selection=["confidence_surprise"], replay_candidate_multiplier=[4], 
            optimizer=["adamw"], clipnorm=[None], global_clipnorm=[5.0], 
            learning_rate={"low": 0.0007, "high": 0.0009, "log": True}
        ))
        self.assertEqual(config.continually_learn.replay_budget_mode, "fixed_total")
        self.assertEqual(config.continually_learn.replay_candidate_multiplier, 4)
        self.assertEqual(config.optimizer.global_clipnorm, 5.0)
        self.assertEqual(config.optimizer.initial_learning_rate, 0.0007)
        for key in ("replay_old_examples", "replay_current_examples", "replay_surprise_weight", "weight_decay", "test_steps_t1000", "test_cfg_scale", "test_eta"):
            self.assertIn(key, trial.params)
        self.assertNotIn("replay_samples", trial.params)
        self.assertNotIn("train_num", trial.params)

    def test_previous_teacher_rejects_new_only_override_with_no_old_rows(self) -> None:
        """A requested previous-teacher treatment cannot silently become a no-op."""

        with self.assertRaises(ValueError):
            self.build({
                "classifier_teacher_source": ["previous"], "noise_teacher_source": ["none"], 
                "continual_strategy": ["new_only"]
            })

    def test_missing_active_teacher_and_loss_fail_without_adding_or_changing_architecture(self) -> None:
        """Required teacher artifacts and enabled loss terms remain explicit fixed inputs."""

        missing = deepcopy(self.profile)
        missing["specialist_teacher_descriptors"].pop("classifier")
        with self.assertRaisesRegex(ValueError, "classifier specialist"):
            self.build({"classifier_teacher_source": ["current"]}, profile=missing)
        disabled = deepcopy(self.profile)
        disabled["student_config"]["model"]["wrapper_kwargs"]["clf_distil_loss_coef"] = 0.0
        with self.assertRaisesRegex(ValueError, "clf_distil_loss_coef"):
            self.build({"classifier_teacher_source": ["previous"]}, profile=disabled)
        with self.assertRaisesRegex(ValueError, "Unknown continual search"):
            self.build({"dim": [64]})
        with self.assertRaisesRegex(ValueError, "Unknown continual search"):
            self.build({"clf_distil_temperature": [1.0]})

    def test_normalization_authenticates_student_and_specialist_artifacts(self) -> None:
        """Once sealed, changed weights or specialist files invalidate the profile."""

        normalized = normalize_continual_profile(self.profile)
        paths = [
            Path(normalized["student_config"]["model"]["weights_path"]), 
            Path(normalized["specialist_teacher_descriptors"]["classifier"]["path"]), 
            Path(normalized["specialist_teacher_descriptors"]["noise"]["path"]), 
            Path(normalized["specialist_teacher_descriptors"]["noise"]["weights_path"])
        ]
        for path in paths:
            before = path.read_bytes()
            try:
                path.write_bytes(before + b"changed")
                with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, "changed"):
                    normalize_continual_profile(normalized)
            finally:
                path.write_bytes(before)

    def test_yaml_source_digest_cannot_be_overwritten_by_renormalization(self) -> None:
        """Supplying a YAML path again must verify its previously sealed digest."""

        student_path = self.root / "student.yaml"
        save_config(self.student, student_path)
        source = dict(self.profile, student_config=str(student_path))
        normalized = normalize_continual_profile(source)
        student_path.write_text(student_path.read_text(encoding="utf-8") + "\n# changed source\n", encoding="utf-8")
        resealed_path_request = dict(normalized, student_config=str(student_path))
        with self.assertRaisesRegex(ValueError, "changed"):
            normalize_continual_profile(resealed_path_request)

    def test_profile_import_and_artifact_normalization_do_not_import_tensorflow(self) -> None:
        """A fresh coordinator can seal artifact bytes without initializing TensorFlow."""

        source_path = self.root / "student.yaml"
        save_config(self.student, source_path)
        script = "\n".join([
            "import sys", 
            "from common.dit_continual_hpo import normalize_continual_profile", 
            "", "", 
            "profile = normalize_continual_profile({'student_config': sys.argv[1], 'task_seed': 42})", 
            "assert len(profile['task_groups']) == 5", 
            "assert 'tensorflow' not in sys.modules, 'Coordinator imported TensorFlow'"
        ])
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        result = subprocess.run(
            [sys.executable, "-c", script, str(source_path)], 
            cwd=Path(__file__).resolve().parents[2], env=environment, 
            capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_baseline_hints_accept_common_categorical_override_forms(self) -> None:
        """The warm-start grid honors the same scalar/list/choices restrictions as HPO."""

        expected = [{"classifier_teacher_source": "previous", "noise_teacher_source": "none"}]
        for classifier in ("previous", ["previous"], {"choices": ["previous"]}):
            with self.subTest(classifier=classifier):
                self.assertEqual(baseline_hints({
                    "classifier_teacher_source": classifier, "noise_teacher_source": ["none"]
                }), expected)

    def test_preflight_checks_all_requested_sources_before_any_trial_is_sampled(self) -> None:
        """Missing late-treatment inputs fail before an early teacher-free trial can succeed."""

        normalized = normalize_continual_profile(self.profile)
        validate_continual_search(normalized)
        no_current = deepcopy(normalized)
        no_current["specialist_teacher_descriptors"] = {}
        with self.assertRaisesRegex(ValueError, "classifier specialist"):
            validate_continual_search(no_current)
        validate_continual_search(no_current, {
            "classifier_teacher_source": {"choices": ["previous"]}, 
            "noise_teacher_source": "none"
        })
        no_loss = deepcopy(normalized)
        no_loss["student_config"]["model"]["wrapper_kwargs"]["noise_distil_loss_coef"] = 0.0
        with self.assertRaisesRegex(ValueError, "noise_distil_loss_coef"):
            validate_continual_search(no_loss)
        validate_continual_search(no_loss, {"noise_teacher_source": ["none"]})
        no_weight = deepcopy(normalized)
        no_weight["student_config"]["model"]["wrapper_kwargs"]["current_teacher_clf_loss_weight"] = 0.0
        with self.assertRaisesRegex(ValueError, "current_teacher_clf_loss_weight"):
            validate_continual_search(no_weight)
        with self.assertRaisesRegex(ValueError, "old examples"):
            validate_continual_search(normalized, {"continual_strategy": ["new_only"]})


# Permit direct execution as well as unittest discovery.
if __name__ == "__main__":
    main()
