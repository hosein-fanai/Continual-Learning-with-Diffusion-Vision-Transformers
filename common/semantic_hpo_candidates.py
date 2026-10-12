"""Fresh semantic-ready recipes informed by completed CIFAR10 campaigns.

Historical scores motivate whole recipes, not independent winning parameter
values. All recipes add a semantic projection, share a frozen development split
and use common clean evaluation. No historical weights or Optuna observations
are loaded. The nine mappings are hypotheses requiring new paired baselines.
"""

from __future__ import annotations

from copy import deepcopy

from common.dit_continual_candidates import candidate_profile


TASK_GROUPS = [[7, 3], [2, 8], [5, 6], [9, 4], [0, 1]]
TASK_SEED = 42
_CONTINUAL_ROOT = (
    "files/results/dit_continual_hpo_archive_20261011/campaign_three_seeds/"
    "shards/{shard}/continual/dit_classifier/cifar10/dit_continual_runner/configs/"
    "trial-{trial:04d}.yaml"
)
_CONTINUAL_REPORT = "files/results/dit_continual_hpo_analysis_20261012/REPORT.md"
_JOINT_ROOT = "files/notebooks/DiT HPO Codex"
_JOINT_REPORT = (
    f"{_JOINT_ROOT}/results/joint_classifier_v1_search100_20261010/"
    "HYPERPARAMETER_EFFECTS_FINAL_20261011.md"
)
_DIT_ROOT = (
    "files/results/dit_generation_hpo_v7/generation/diffusion_transformer/cifar10/"
    "notebook_runner/phases/continue1000_20261010"
)
_RECIPES = {
    "a68_clean": {
        "source": ["a", 68], "student": "joint155_clean", "noise_source": "current", 
        "noise_teacher": "unet23", "current_examples": 5000, 
        "learning_rate": 0.0008922997544875341, "cfg_scale": 3.6860632085091245, 
        "p_uncond": 0.2, "clipnorm": 1.0, 
        "metrics": {"confirmation_mean": 0.20523333333333332, "confirmation_sd": 0.004789919971495681, "seeds": 3, "old_task_mean": 0.0255}
    }, 
    "b61_noisy": {
        "source": ["b", 61], "student": "joint165_noisy", "noise_source": "both", 
        "noise_teacher": "unet23", "current_examples": 2500, 
        "learning_rate": 0.0018365248088441294, "cfg_scale": 2.719624499157652, 
        "p_uncond": 0.1, "clipnorm": 5.0, 
        "metrics": {"confirmation_mean": 0.20573333333333332, "confirmation_sd": 0.02479724447057239, "seeds": 3, "old_task_mean": 0.04033333333333333}
    }, 
    "b7_retention": {
        "source": ["b", 7], "student": "joint155_noisy", "noise_source": "both", 
        "noise_teacher": "unet23", "current_examples": 2500, 
        "learning_rate": 0.0018860103763154862, "cfg_scale": 2.929069189794846, 
        "p_uncond": 0.05, "clipnorm": 5.0, 
        "metrics": {"search_final_accuracy": 0.2259, "search_old_task_accuracy": 0.0995, "seeds": 1}
    }, 
    "b82_efficient": {
        "source": ["b", 82], "student": "joint155_noisy", "noise_source": "none", 
        "noise_teacher": None, "current_examples": 2500, 
        "learning_rate": 0.0008266869613643858, "cfg_scale": 3.2462311803589987, 
        "p_uncond": 0.05, "clipnorm": 0.5, 
        "metrics": {"search_final_accuracy": 0.2248, "search_old_task_accuracy": 0.0745, "seeds": 1}
    }
}


def semantic_budget_profiles() -> dict:
    """Return coupled phase-cost hypotheses; these are not measured optima.

    A tier changes several semantic controls together and therefore estimates a
    recipe comparison, not the isolated effect of each phase or view count.
    """

    return {
        "low": {
            "acquisition_steps": 100, "consolidation_steps": 100, "batch_size": 32, 
            "augmentation_views": 2, "noise_levels": [0]
        }, 
        "standard": {
            "acquisition_steps": 100, "consolidation_steps": 200, "batch_size": 32, 
            "augmentation_views": 4, "noise_levels": [0, 10, 50]
        }, 
        "strong": {
            "acquisition_steps": 200, "consolidation_steps": 400, "batch_size": 32, 
            "augmentation_views": 4, "noise_levels": [0, 10, 50]
        }
    }


def _noise_candidates(catalog: dict, validation_ratio: float, epochs: int) -> dict:
    """Adapt specialist descriptors to the common current-task training protocol."""

    candidates = deepcopy(catalog["noise_teacher_candidates"])
    candidates["dit_cifar10_1770"] = deepcopy(candidates["dit_cifar10_1351"])
    alternative = candidates["dit_cifar10_1770"]
    alternative["descriptor"]["config"]["model"]["kwargs"]["use_refiner_cnn"] = False
    alternative["descriptor"]["config"]["optimizer"].update({
        "name": "adamw", "initial_learning_rate": 0.004923507985709955, 
        "weight_decay": 0.0006398310822142558, "clipnorm": 5.0, "global_clipnorm": None
    })
    alternative["provenance"] = {
        "source_config": f"{_DIT_ROOT}/finalist_configs/trial-1770.yaml", 
        "source_summary": f"{_DIT_ROOT}/confirmation_summary.csv", 
        "source_metrics": {"noise_loss_mean": 0.030541765814026196, "noise_loss_sd": 0.00031140784999894724, "seeds": 3}, 
        "evidence_status": "near_best_confirmed_denoiser_architecture_transfer", 
        "initialization": "fresh", 
        "adaptations": [
            "Historical EMA 0.999 is disabled and modify_first_t=False becomes True.", 
            "Historical 50 full-data epochs/cosine become the common current-task epochs/constant rate.", 
            "Refiner, optimizer, decay and clipping differ together from trial1351; this is not a refiner-only ablation."
        ]
    }
    for candidate in candidates.values():
        config = candidate["descriptor"]["config"]
        config["dataset"].update({
            "batch_size": 128, "validation_source": "split", "validation_ratio": validation_ratio
        })
        config["training"].update({"epochs": epochs, "dtype_policy": "float32"})
        # The specialist loader derives a role stream from each trial/model seed.
        for controls in (config["training"], config["model"]["kwargs"], config["model"]["wrapper_kwargs"]):
            controls.pop("seed", None)
        config["model"]["wrapper_kwargs"].update({
            "test_noisified_min_timesteps": 0, "test_noisified_max_timesteps": 0
        })
        candidate["provenance"]["adaptations"].extend([
            "Specialists are freshly trained on current-task real training rows only, never validation rows.", 
            "No explicit teacher seed overrides the loader's independent role stream derived from the model seed.", 
            "The learner supplies its batch128 and shared per-task epoch budget; descriptor metadata matches that effective training.", 
            "The held-out training split replaces historical official-test feedback; historical losses are provenance only."
        ])
    return candidates


def _native_recipe(
    settings: dict, catalog: dict, teachers: dict, validation_ratio: float, epochs: int, seed: int
) -> dict:
    """Construct one explicit native continual configuration without model imports."""

    config = deepcopy(catalog["student_candidates"][settings["student"]]["config"])
    network = config["model"]["kwargs"]
    network.update({"classifier_mlp_ratio": 1, "seed": seed})
    network.update(settings.get("network_overrides", {}))
    wrapper = config["model"]["wrapper_kwargs"]
    cap = wrapper["clf_train_noisified_max_timesteps"]
    noise_source = settings["noise_source"]
    current_noise = noise_source in ("current", "both")
    wrapper.update({
        "clf_test_noisified_max_timesteps": 0, 
        "test_noisified_min_timesteps": 0, "test_noisified_max_timesteps": 0, 
        "clf_distil_noisy_input_type": "clean" if cap == 0 else "noisy", 
        "clf_distil_train_noisified_max_timesteps": cap, "clf_distil_test_noisified_max_timesteps": 0, 
        "clf_distil_scope": "old_classes", "dual_teacher_scope": "task", 
        "clf_distil_loss_coef": 0.001075, "clf_distil_acc_coef": 0.0, 
        "noise_distil_loss_coef": 0.5 if noise_source != "none" else 0.0, 
        "previous_teacher_clf_loss_weight": 1.0, "current_teacher_clf_loss_weight": 0.0, 
        "previous_teacher_noise_loss_weight": 1.0 if noise_source == "both" else 0.0, 
        "current_teacher_noise_loss_weight": 1.0 if current_noise else 0.0, 
        "defer_teacher": True, "trainable_teacher": current_noise, 
        "teacher_training": "each_task", "teacher_dynamic_classes": current_noise, 
        "use_ensemble_loss_instead": False, "train_cfg_scale": None, 
        "p_uncond": settings["p_uncond"], "test_steps": 50, 
        "test_cfg_scale": settings["cfg_scale"], "test_eta": 0.0, "seed": seed
    })
    config["model"]["with_classifier"] = True
    config["dataset"].update({
        "batch_size": 128, "preprocess": None, "return_features": False, 
        "validation_source": "split", "validation_ratio": validation_ratio
    })
    config["optimizer"].update({
        "name": "adam", "initial_learning_rate": settings["learning_rate"], 
        "weight_decay": None, "clipnorm": settings["clipnorm"], "global_clipnorm": None, 
        "schedule": "constant", "plateau_jump": False
    })
    config["training"].update({
        "task": "continual", "epochs": epochs, "fit_method": "fit", "use_valset": True, 
        "patience": 0, "reduce_lr_patience": 0, "dtype_policy": "float32", 
        "show_images": False, "save_gifs": False, "report_every_epoch": False, 
        "tensorboard": True, "verbose": 2, "seed": seed
    })
    descriptors = {}
    # Only active current-noise roles require a freshly trained specialist.
    if current_noise:
        descriptors["noise"] = deepcopy(teachers[settings["noise_teacher"]]["descriptor"])
    config["continually_learn"] = {
        "class_num": 10, "class_order": [value for group in TASK_GROUPS for value in group], 
        "task_groups": deepcopy(TASK_GROUPS), "task_size": 2, 
        "class_order_mode": "fixed", "task_order_mode": "fixed", 
        "remove_prev_classes": True, "keep_same_model": True, "use_buffer": False, 
        "use_generative_replay": True, "replay_budget_mode": "fixed_total", 
        "replay_old_examples": settings.get("old_examples", 500), 
        "replay_current_examples": settings["current_examples"], 
        "replay_selection": "uniform", "replay_candidate_multiplier": 1, 
        "use_generative_model_classifier": True, "train_classifier_separately": False, 
        "use_distillation": True, "dual_teacher_distillation": False, 
        "current_teacher_init": "fresh", "snapshot_network_name": "raw", 
        "specialist_teacher_descriptors": descriptors, "optimizer_steps_per_epoch": None, 
        "use_ensemble_accuracy": False, "evaluate_ensemble_accuracy": False, 
        "generative_model_kwargs": {"samples_per_class": 0, "train_num": -1}, 
        "plot_results": False, "show_generated_images": False, "return_details": True, 
        "save_task_checkpoints": True, "experiment_phase": "development", "seed": TASK_SEED
    }
    config["reporting"].update({
        "show_history_plot": False, "save_history_plot": False, 
        "show_final_images": False, "save_final_images": False, "save_final_gifs": False, 
        "run_trainset_eval": False, "run_valset_eval": True, "save_csv": True
    })
    config["hpo"] = {"continual_dataset_seed": TASK_SEED}
    return config


def semantic_candidate_profile(
    validation_ratio: float = 0.1, epochs: int = 10, seed: int = 42
) -> dict:
    """Return nine coherent native recipes and their source/adaptation records.

    ``seed`` changes model initialization only; the task partition and training
    validation split retain seed42 across search and confirmation. Noisy recipe
    names denote classifier training below timestep64, with the same clean
    development endpoint for all candidates. Historical noisy endpoint scores
    therefore cannot be treated as measurements of these adapted recipes.

    Current noise specialists use the learner's real current-task dataset,
    batch128 and shared epoch budget. Their independent optimizer recipes remain
    explicit. No current classifier specialist or historical weights are needed.
    """

    catalog = candidate_profile(task_seed=TASK_SEED)
    teachers = _noise_candidates(catalog, validation_ratio, epochs)
    recipes = deepcopy(_RECIPES)
    recipes["joint126_projection"] = {
        **deepcopy(recipes["b82_efficient"]), "student": "joint165_noisy", 
        "network_overrides": {"clf_droppath_rate": 0.0}, "architecture_parent": 126
    }
    recipes["joint165_clean"] = {
        **deepcopy(recipes["a68_clean"]), "student": "joint165_clean", "architecture_parent": 165
    }
    recipes["b7_dit1351"] = {
        **deepcopy(recipes["b7_retention"]), "noise_teacher": "dit_cifar10_1351"
    }
    recipes["b7_dit1770"] = {
        **deepcopy(recipes["b7_retention"]), "noise_teacher": "dit_cifar10_1770"
    }
    recipes["b7_replay2000"] = {
        **deepcopy(recipes["b7_retention"]), "old_examples": 2000
    }
    native = {}
    provenance = {}
    for recipe_id, settings in recipes.items():
        native[recipe_id] = _native_recipe(settings, catalog, teachers, validation_ratio, epochs, seed=seed)
        shard, trial = settings["source"]
        adaptations = [
            "Fresh initialization; source checkpoints and past scores are not imported into Optuna.", 
            "All candidates and their paired baselines use classifier_mlp_ratio=1; this changes the historical continual architecture.", 
            "Official-test selection is replaced by the same seeded training-validation split across all candidates and confirmation seeds.", 
            "Noisy candidates retain cap64 classifier training but use clean ordinary evaluation instead of the historical noisy endpoint.", 
            "The five historical class pairs remain fixed in the same order; only model and semantic seeds change during confirmation.", 
            f"Uniform multiplier1 avoids extra replay generation; {settings.get('old_examples', 500)} is the total old-class pool, not samples per class.", 
            "Joint and specialist training share the requested per-task epochs and batch128; historical full-data budgets do not transfer.", 
            "Semantic settings are new hypotheses; historical retention remains poor and does not establish semantic efficacy."
        ]
        # The efficient source used a larger uniform candidate pool historically.
        if trial == 82:
            adaptations.append("Historical uniform replay multiplier2 becomes multiplier1; candidate pools and random draws can differ.")
        # Copy126 transfers its measured head into the efficient continual procedure.
        if settings.get("architecture_parent") == 126:
            adaptations.append("Copy126 depth8/DropPath0/projection1 replaces b82's depth4 head; b82 optimizer, replay and previous-KD procedure remain.")
        # Copy165 transfers its replicated clean head into the clean procedure.
        if settings.get("architecture_parent") == 165:
            adaptations.append("The replicated clean165 depth8/DropPath0.05 prior replaces a68's shallow155 head; a68's continual procedure remains.")
        # The DiT branches replace only the selected current-noise specialist recipe.
        if recipe_id.startswith("b7_dit"):
            adaptations.append("The current UNet23 noise specialist becomes the named DiT denoiser; all b7 student and role weights remain.")
        # Larger replay is an unmeasured intervention with a concrete cost increase.
        if recipe_id == "b7_replay2000":
            adaptations.extend([
                "New retention-pressure hypothesis: increase total old replay500 to2000 while retaining all other b7 native controls.", 
                "Fourfold generated replay increases sampling and joint-training cost; this is not a measured winner or a causal replay optimum.", 
                "A fixed total2000 is a bounded intervention, not the paper's per-class replay setting; no historical score measures this branch."
            ])
        item = {
            "source_config": _CONTINUAL_ROOT.format(shard=shard, trial=trial), 
            "source_report": _CONTINUAL_REPORT, "procedure_source": f"{shard}:{trial}", 
            "procedure_source_metrics": deepcopy(settings["metrics"]), 
            "student_prior": deepcopy(catalog["student_candidates"][settings["student"]]["provenance"]), 
            "evidence_status": "new_semantic_projection_and_protocol_transfer", 
            "historical_metrics_apply_to_adapted_recipe": False, "initialization": "fresh", 
            "classifier_teacher_source": "previous", "noise_teacher_source": settings["noise_source"], 
            "adaptations": adaptations
        }
        # Replace inherited Copy165 provenance with the actual Copy126 source.
        if settings.get("architecture_parent") == 126:
            item["student_prior"] = {
                "source_config": f"{_JOINT_ROOT}/exports/completed_wave46_20261010/results/copy126/config.json", 
                "source_report": _JOINT_REPORT, "source_parent": 126, 
                "source_metrics": {"fixed_ensemble_accuracy": 0.8096, "mean_class_mmd": 0.046241, "seeds": 1}, 
                "evidence_status": "single_seed_projection_architecture_prior"
            }
        # Inactive specialist branches have no historical specialist to cite.
        if settings["noise_teacher"] is not None:
            item["noise_teacher_prior"] = deepcopy(teachers[settings["noise_teacher"]]["provenance"])
        provenance[recipe_id] = item
    route = {
        **semantic_budget_profiles()["standard"], "learning_rate": 0.0005, "temperature": 0.1, 
        "alignment_weight": 1.0, "ce_weight": 1.0, "orthogonality_weight": 1.0, 
        "gain_limit": 1.0, "bias_limit": 1.0, "modulation_init_std": 0.02, 
        "acquisition_noise_level": 0, "ce_noise_level": 0, "image_augmentation": "tmcl", 
        "reliability": "alpha_bar", "reliability_floor": 0.05, "condition": "learned", 
        "consolidation_scope": "semantic", "acquisition_objective": "contrastive", 
        "retain_modulators": True, "probe_batches": 4, "probe_max_gates": 16, "seed": seed
    }
    return {
        "student_config": deepcopy(native["a68_clean"]), "route_settings": route, 
        "recipe_candidates": native, "candidate_provenance": provenance, 
        "phase_budget_candidates": semantic_budget_profiles(), 
        "selection_metric": "retention_balanced_accuracy", 
        "task_seed": TASK_SEED, "dataset_seed": TASK_SEED
    }


def semantic_initial_trials(profile: dict) -> list[dict]:
    """Queue each recipe at each semantic budget before adaptive optimization.

    The nine strong-tier anchors come first so initial capacity/timing coverage
    measures the largest declared phase budget. These are new trials with fixed
    reference semantic values, never imported historical Optuna observations.
    Reliability parameters are omitted for the genuinely clean low tier.
    """

    reference = {
        "learning_rate": 0.0005, "temperature": 0.1, "alignment_weight": 1.0, 
        "orthogonality_weight": 1.0, "ce_weight": 1.0, "gain_limit": 1.0, 
        "bias_limit": 1.0, "modulation_init_std": 0.02, "acquisition_noise_level": 0, 
        "ce_noise_level": 0, "image_augmentation": "tmcl"
    }
    anchors = []
    for tier in ("strong", "low", "standard"):
        budget = profile["phase_budget_candidates"][tier]
        levels = budget["noise_levels"]
        # Normalized profiles encode the same levels as a comma-separated string.
        if isinstance(levels, str):
            levels = [int(value) for value in levels.split(",")]
        for recipe_id in profile["recipe_candidates"]:
            params = {**reference, "base_recipe": recipe_id, "semantic_budget": tier}
            # Reliability has no active role when every consolidation view is clean.
            if any(level > 0 for level in levels):
                params.update({"reliability": "alpha_bar", "reliability_floor": 0.05})
            anchors.append(params)
    return anchors
