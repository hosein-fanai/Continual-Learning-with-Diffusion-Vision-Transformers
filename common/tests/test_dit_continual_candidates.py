"""Candidate selection, recipe preservation and fixed official-test stream checks.

Run on the authorized remote container. These tests use real Optuna storage and
native Configs without fitting the full historical architectures.
"""

from copy import deepcopy
from tempfile import TemporaryDirectory
from unittest import TestCase

import optuna

from common.dit_continual_candidates import candidate_profile
from common.dit_continual_hpo import build_dit_continual_config, normalize_continual_profile, validate_continual_search


class ContinualCandidateTests(TestCase):
    """Protect scientific identity across conditional student and teacher choices."""

    def test_catalog_roundtrip_and_fixed_schedule(self) -> None:
        """Transport preserves all candidates and never reseeds the task schedule."""

        profile = normalize_continual_profile(candidate_profile(), seed=19)
        self.assertEqual(profile, normalize_continual_profile(profile, seed=999))
        self.assertEqual(len(profile["task_groups"]), 5)
        self.assertEqual(sorted(sum(profile["task_groups"], [])), list(range(10)))
        validate_continual_search(profile)
        changed = deepcopy(profile)
        changed["task_groups"].reverse()
        with self.assertRaisesRegex(ValueError, "changed"):
            normalize_continual_profile(changed)
        changed = deepcopy(profile)
        changed["noise_teacher_candidates"]["unet23"]["descriptor"]["config"]["optimizer"]["initial_learning_rate"] = 0.123
        with self.assertRaisesRegex(ValueError, "identity changed"):
            normalize_continual_profile(changed)

    def test_all_students_and_teachers_share_stable_optuna_distributions(self) -> None:
        """Real suggestions select configs, keep paired noise and skip inactive teachers."""

        profile = candidate_profile()
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=4))
        classifier_ids = list(profile["classifier_teacher_candidates"])
        noise_ids = list(profile["noise_teacher_candidates"])
        hints = []
        for index, student_id in enumerate(profile["student_candidates"]):
            hints.append({
                "student_config": student_id, "classifier_teacher_source": "current", 
                "noise_teacher_source": "current", 
                "classifier_teacher_descriptor": classifier_ids[index % len(classifier_ids)], 
                "noise_teacher_descriptor": noise_ids[index % len(noise_ids)]
            })
        hints.append({"student_config": "joint165_clean", "classifier_teacher_source": "none", "noise_teacher_source": "none"})
        for hint in hints:
            study.enqueue_trial(hint)
        with TemporaryDirectory() as directory:
            for hint in hints:
                trial = study.ask()
                config = build_dit_continual_config(
                    trial, "cifar10", 20, directory, profile, 
                    search_space_overrides={"continual_strategy": ["cumulative"], "batch_size": [128]}, seed=42
                )
                chosen = hint["student_config"]
                cap = 64 if chosen.endswith("noisy") else 0
                self.assertEqual(config.model.wrapper_kwargs["clf_train_noisified_max_timesteps"], cap)
                self.assertEqual(config.model.wrapper_kwargs["clf_test_noisified_max_timesteps"], cap)
                self.assertEqual(config.dataset.validation_source, "test")
                self.assertEqual(config.dataset.batch_size, 128)
                self.assertEqual(config.training.epochs, 20)
                self.assertEqual(config.hpo["selected_candidates"]["student_config"], chosen)
                self.assertNotIn("classifier_noise", trial.params)
                for role in ("classifier", "noise"):
                    key = f"{role}_teacher_descriptor"
                    # Active specialists must retain the explicitly queued recipe.
                    if hint[f"{role}_teacher_source"] == "current":
                        self.assertEqual(config.hpo["selected_candidates"][key], hint[key])
                        self.assertEqual(config.continually_learn.specialist_teacher_descriptors[role]["format"], "config")
                    # Inactive specialists must consume no Optuna suggestion.
                    else:
                        self.assertNotIn(key, trial.params)
                study.tell(trial, 0.5)

    def test_candidate_noise_cannot_be_silently_overridden(self) -> None:
        """A paired clean/noisy model ID must mean the same recipe in every trial."""

        profile = normalize_continual_profile(candidate_profile())
        with self.assertRaisesRegex(ValueError, "paired classifier"):
            validate_continual_search(profile, {"classifier_noise": ["clean"]})

    def test_classifier_teacher_view_follows_selected_roles_without_changing_noise_teacher(self) -> None:
        """Cross-view current KD and mixed-role common support remain explicit."""

        cases = [
            ("joint165_clean", "current", "joint165_noisy", 64, {"current": 64}), 
            ("joint165_noisy", "current", "joint165_clean", 0, {"current": 0}), 
            ("joint165_clean", "both", "joint165_noisy", 0, {"previous": 0, "current": 64}), 
            ("joint165_noisy", "both", "joint165_clean", 0, {"previous": 64, "current": 0}), 
            ("joint165_noisy", "both", "joint165_noisy", 64, {"previous": 64, "current": 64}), 
            ("joint165_clean", "previous", None, 0, {"previous": 0}), 
            ("joint165_noisy", "previous", None, 64, {"previous": 64})
        ]
        profile = candidate_profile()
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=7))
        for student, source, teacher, _, _ in cases:
            hint = {
                "student_config": student, "classifier_teacher_source": source, 
                "noise_teacher_source": "current", "noise_teacher_descriptor": "unet23"
            }
            # Previous-only treatments have no current classifier descriptor.
            if teacher is not None:
                hint["classifier_teacher_descriptor"] = teacher
            study.enqueue_trial(hint)
        with TemporaryDirectory() as directory:
            for student, source, teacher, cap, role_caps in cases:
                with self.subTest(student=student, source=source, teacher=teacher):
                    trial = study.ask()
                    config = build_dit_continual_config(
                        trial, "cifar10", 20, directory, profile, 
                        search_space_overrides={"continual_strategy": ["cumulative"]}, seed=42
                    )
                    wrapper = config.model.wrapper_kwargs
                    self.assertEqual(wrapper["clf_distil_noisy_input_type"], "clean" if cap == 0 else "noisy")
                    self.assertEqual(wrapper["clf_distil_train_noisified_max_timesteps"], cap)
                    self.assertEqual(wrapper["clf_distil_test_noisified_max_timesteps"], cap)
                    self.assertEqual(config.hpo["classifier_teacher_inputs"]["role_training_support_caps"], role_caps)
                    self.assertFalse(config.hpo["classifier_teacher_inputs"]["noise_distillation_inputs_changed"])
                    student_cap = 64 if student.endswith("noisy") else 0
                    self.assertEqual(wrapper["clf_train_noisified_max_timesteps"], student_cap)
                    self.assertEqual(wrapper["clf_test_noisified_max_timesteps"], student_cap)
                    noise = config.continually_learn.specialist_teacher_descriptors["noise"]["config"]
                    self.assertEqual(noise["model"]["name"], "unet")
                    self.assertEqual(noise["model"]["wrapper_kwargs"]["train_noisified_max_timesteps"], 1000)
                    self.assertEqual(noise["optimizer"]["name"], "adamw")
                    study.tell(trial, .5)
