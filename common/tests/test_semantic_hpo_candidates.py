"""Check candidate contracts and source transfers on an authorized remote host.

These checks build native configuration profiles; they do not train models.
"""

from copy import deepcopy
from unittest import TestCase

from common.config import Config
from common.semantic_hpo import normalize_semantic_profile
from common.semantic_hpo_candidates import (
    TASK_GROUPS, semantic_budget_profiles, semantic_candidate_profile, semantic_initial_trials
)
from semantic_consolidation.config import RouteConfig, RouteSettings, validate_route_config


class SemanticCandidateTests(TestCase):
    """Keep source recipes, shared evaluation and active teacher roles compatible."""

    def test_every_recipe_passes_native_semantic_validation(self) -> None:
        """Whole recipes must satisfy native contracts before expensive trials start."""

        profile = semantic_candidate_profile()
        self.assertEqual(len(profile["recipe_candidates"]), 9)
        for recipe_id, mapping in profile["recipe_candidates"].items():
            with self.subTest(recipe=recipe_id):
                native = Config(**deepcopy(mapping))
                route = RouteSettings(**deepcopy(profile["route_settings"]))
                validate_route_config(RouteConfig(common=native, route=route))
                normalized = normalize_semantic_profile({
                    "student_config": mapping, "route_settings": profile["route_settings"], 
                    "task_seed": 42, "dataset_seed": 42
                })
                self.assertEqual(normalized["task_groups"], TASK_GROUPS)
                self.assertEqual(native.dataset.validation_source, "split")
                self.assertEqual(native.model.wrapper_kwargs["clf_test_noisified_max_timesteps"], 0)
                self.assertEqual(native.model.wrapper_kwargs["test_noisified_max_timesteps"], 0)
                self.assertFalse(native.continually_learn.use_ensemble_accuracy)
                self.assertIsNone(native.model.weights_path)

    def test_model_seeds_and_budget_do_not_change_data_or_tasks(self) -> None:
        """Confirmation seeds may not resample the task partition or validation split."""

        first = semantic_candidate_profile()
        repeated = semantic_candidate_profile(validation_ratio=0.1, epochs=12, seed=303)
        for recipe_id, native in repeated["recipe_candidates"].items():
            self.assertEqual(native["dataset"], first["recipe_candidates"][recipe_id]["dataset"])
            self.assertEqual(native["continually_learn"]["task_groups"], TASK_GROUPS)
            self.assertEqual(native["continually_learn"]["seed"], 42)
            self.assertEqual(native["hpo"]["continual_dataset_seed"], 42)
            self.assertEqual(native["training"]["seed"], 303)
            self.assertEqual(native["training"]["epochs"], 12)
            for descriptor in native["continually_learn"]["specialist_teacher_descriptors"].values():
                self.assertEqual(descriptor["config"]["training"]["epochs"], 12)
                self.assertEqual(descriptor["config"]["dataset"]["batch_size"], 128)
                self.assertEqual(descriptor["config"]["dataset"]["validation_source"], "split")

    def test_inactive_specialists_are_absent_and_loss_roles_match(self) -> None:
        """Teacher-free noise branches must not train or retain an unused specialist."""

        candidates = semantic_candidate_profile()["recipe_candidates"]
        for recipe_id, native in candidates.items():
            wrapper = native["model"]["wrapper_kwargs"]
            descriptors = native["continually_learn"]["specialist_teacher_descriptors"]
            self.assertNotIn("classifier", descriptors)
            self.assertEqual(wrapper["previous_teacher_clf_loss_weight"], 1.0)
            self.assertEqual(wrapper["current_teacher_clf_loss_weight"], 0.0)
            # Efficient recipes omit both specialist allocation and noise-KD loss.
            if recipe_id in ("b82_efficient", "joint126_projection"):
                self.assertEqual(descriptors, {})
                self.assertEqual(wrapper["noise_distil_loss_coef"], 0.0)
                self.assertEqual(wrapper["current_teacher_noise_loss_weight"], 0.0)
                self.assertEqual(wrapper["previous_teacher_noise_loss_weight"], 0.0)
                self.assertFalse(wrapper["trainable_teacher"])
            # Other recipes actively train the declared current-noise specialist.
            else:
                self.assertEqual(set(descriptors), {"noise"})
                self.assertEqual(wrapper["noise_distil_loss_coef"], 0.5)
                self.assertTrue(wrapper["trainable_teacher"])

    def test_dit1770_retains_optimizer_and_refiner_contrast(self) -> None:
        """The near-tied denoiser is not mislabeled as a refiner-only intervention."""

        candidates = semantic_candidate_profile()["recipe_candidates"]
        teacher1351 = candidates["b7_dit1351"]["continually_learn"]["specialist_teacher_descriptors"]["noise"]["config"]
        teacher1770 = candidates["b7_dit1770"]["continually_learn"]["specialist_teacher_descriptors"]["noise"]["config"]
        self.assertEqual(teacher1351["optimizer"]["name"], "adam")
        self.assertEqual(teacher1351["optimizer"]["initial_learning_rate"], 0.004845009591308682)
        self.assertEqual(teacher1351["optimizer"]["global_clipnorm"], 1.0)
        self.assertTrue(teacher1351["model"]["kwargs"]["use_refiner_cnn"])
        self.assertEqual(teacher1770["optimizer"]["name"], "adamw")
        self.assertEqual(teacher1770["optimizer"]["initial_learning_rate"], 0.004923507985709955)
        self.assertEqual(teacher1770["optimizer"]["weight_decay"], 0.0006398310822142558)
        self.assertEqual(teacher1770["optimizer"]["clipnorm"], 5.0)
        self.assertIsNone(teacher1770["optimizer"]["global_clipnorm"])
        self.assertFalse(teacher1770["model"]["kwargs"]["use_refiner_cnn"])

    def test_projection126_and_clean165_are_explicit_architecture_transfers(self) -> None:
        """New architectures retain their declared procedure instead of inherited scores."""

        profile = semantic_candidate_profile()
        candidates = profile["recipe_candidates"]
        for recipe_id, depth, drop_path, cap in [
            ("joint126_projection", 8, 0.0, 64), ("joint165_clean", 8, 0.05, 0)
        ]:
            model = candidates[recipe_id]["model"]
            self.assertEqual(model["kwargs"]["clf_depth"], depth)
            self.assertEqual(model["kwargs"]["clf_droppath_rate"], drop_path)
            self.assertEqual(model["kwargs"]["classifier_mlp_ratio"], 1)
            self.assertEqual(model["wrapper_kwargs"]["clf_train_noisified_max_timesteps"], cap)
            self.assertFalse(profile["candidate_provenance"][recipe_id]["historical_metrics_apply_to_adapted_recipe"])
        self.assertEqual(candidates["joint126_projection"]["optimizer"], candidates["b82_efficient"]["optimizer"])
        self.assertEqual(candidates["joint165_clean"]["optimizer"], candidates["a68_clean"]["optimizer"])

    def test_profile_calls_and_reference_recipe_are_detached(self) -> None:
        """Trial edits cannot mutate sibling recipes, provenance or later searches."""

        first = semantic_candidate_profile()
        original = semantic_candidate_profile()
        first["student_config"]["model"]["kwargs"]["dim"] = 16
        first["recipe_candidates"]["b7_dit1351"]["model"]["kwargs"]["dim"] = 32
        first["candidate_provenance"]["b7_retention"]["adaptations"].append("changed")
        self.assertEqual(first["recipe_candidates"]["a68_clean"]["model"]["kwargs"]["dim"], 128)
        self.assertEqual(first["recipe_candidates"]["b7_retention"]["model"]["kwargs"]["dim"], 128)
        self.assertEqual(semantic_candidate_profile(), original)

    def test_replay2000_is_an_explicit_unconfirmed_replay_pressure_contrast(self) -> None:
        """Larger replay changes its declared exposure without importing a winning score."""

        profile = semantic_candidate_profile()
        reference = deepcopy(profile["recipe_candidates"]["b7_retention"])
        reference["continually_learn"]["replay_old_examples"] = 2000
        self.assertEqual(profile["recipe_candidates"]["b7_replay2000"], reference)
        provenance = profile["candidate_provenance"]["b7_replay2000"]
        self.assertFalse(provenance["historical_metrics_apply_to_adapted_recipe"])
        self.assertTrue(any("New retention-pressure hypothesis" in value for value in provenance["adaptations"]))
        self.assertTrue(any("sampling and joint-training cost" in value for value in provenance["adaptations"]))

    def test_semantic_budget_tiers_are_valid_for_each_recipe(self) -> None:
        """Each coupled semantic budget fits every candidate's horizon and memory contract."""

        profile = semantic_candidate_profile()
        for tier, settings in semantic_budget_profiles().items():
            for recipe_id, mapping in profile["recipe_candidates"].items():
                with self.subTest(tier=tier, recipe=recipe_id):
                    route = RouteSettings(**{**profile["route_settings"], **settings})
                    validate_route_config(RouteConfig(common=Config(**deepcopy(mapping)), route=route))

    def test_specialist_role_rng_is_not_pinned_to_the_source_model_seed(self) -> None:
        """Explicit teacher seeds must not override the loader's per-model role stream."""

        first = semantic_candidate_profile(seed=42)
        repeated = semantic_candidate_profile(seed=303)
        for recipe_id, native in repeated["recipe_candidates"].items():
            descriptors = native["continually_learn"]["specialist_teacher_descriptors"]
            self.assertEqual(
                descriptors, first["recipe_candidates"][recipe_id]["continually_learn"]["specialist_teacher_descriptors"]
            )
            for descriptor in descriptors.values():
                teacher = Config(**deepcopy(descriptor["config"]))
                self.assertIsNone(teacher.training.seed)
                self.assertIsNone(teacher.model.kwargs.get("seed"))
                self.assertIsNone(teacher.model.wrapper_kwargs.get("seed"))
                self.assertIsNone(teacher.model.diffusion_model.seed)

    def test_initial_trials_cover_every_pair_and_time_the_strong_tier_first(self) -> None:
        """Queued coverage cannot omit an expensive recipe or waste clean-tier parameters."""

        profile = semantic_candidate_profile()
        anchors = semantic_initial_trials(profile)
        self.assertEqual(len(anchors), 27)
        self.assertEqual(
            {(item["base_recipe"], item["semantic_budget"]) for item in anchors}, 
            {(recipe_id, tier) for recipe_id in profile["recipe_candidates"] for tier in ("low", "standard", "strong")}
        )
        self.assertTrue(all(item["semantic_budget"] == "strong" for item in anchors[:9]))
        self.assertEqual([item["base_recipe"] for item in anchors[:9]], list(profile["recipe_candidates"]))
        for item in anchors:
            self.assertEqual(item["learning_rate"], 0.0005)
            self.assertEqual(item["temperature"], 0.1)
            self.assertEqual(item["alignment_weight"], 1.0)
            self.assertEqual(item["acquisition_noise_level"], 0)
            self.assertEqual(item["ce_noise_level"], 0)
            self.assertNotIn("augmentation_views", item)
            self.assertNotIn("noise_levels", item)
            self.assertNotIn("condition", item)
            # Clean-only views leave reliability parameters conditionally absent.
            if item["semantic_budget"] == "low":
                self.assertNotIn("reliability", item)
                self.assertNotIn("reliability_floor", item)
            # Positive-noise tiers use the same declared diffusion reliability rule.
            else:
                self.assertEqual(item["reliability"], "alpha_bar")
                self.assertEqual(item["reliability_floor"], 0.05)
        anchors[0]["learning_rate"] = 1.0
        self.assertEqual(semantic_initial_trials(profile)[0]["learning_rate"], 0.0005)
