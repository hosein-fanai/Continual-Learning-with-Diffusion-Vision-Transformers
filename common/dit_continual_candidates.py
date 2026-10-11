"""Evidence-backed fresh recipes for the twelve-hour continual HPO notebook.

Historical scores select architectural priors, never imported Optuna observations
or initial weights. Every recipe below is an explicit transfer to the common
CIFAR10, five-task, epsilon, raw-network protocol. No TensorFlow import is needed.
"""

from __future__ import annotations

from copy import deepcopy


_JOINT_EXPORT = "files/notebooks/DiT HPO Codex/exports/final20_completed_20261010"
_JOINT_SCREEN = (
    "files/notebooks/DiT HPO Codex/results/joint_classifier_v1_search100_20261010/"
    "resume_science_evidence_20261010/diagnostic/results/copy155"
)
_DIT_RUNNER = (
    "files/results/dit_generation_hpo_v7/generation/diffusion_transformer/cifar10/"
    "notebook_runner/phases/continue1000_20261010"
)
_UNET_RUNNER = "files/results/unet_generation_hpo_v1/generation/unet/cifar10/notebook_runner"
CATALOG_EVIDENCE_NOTES = [
    {
        "source_summary": (
            "files/results/dit_generation_hpo_cifar100_v1/generation/diffusion_transformer/"
            "cifar100/notebook_runner/confirmation_summary.json"
        ), 
        "finding": "CIFAR100 trial 873 mean noise loss 0.013458724 across 3 seeds; width 128, depth 6, no CFG, 5000 linear steps.", 
        "decision": "Exclude unchanged: different dataset, diffusion schedule and CFG convention."
    }, 
    {
        "source_summary": (
            "files/results/dit_generation_hpo_cifar100_followup_v1/generation/diffusion_transformer/"
            "cifar100/notebook_runner/confirmation_summary.json"
        ), 
        "finding": "Follow-up trial 82 mean 0.013252885 across 3 seeds; depth 9 local hybrid, width 128, explicit key width 128.", 
        "decision": "Promising later transfer, but higher cost and 5000 linear steps without CFG do not match this 12-hour recipe."
    }, 
    {
        "source_notebook": "files/notebooks/hpo/DiT_Generation_HPO_Followup.ipynb", 
        "finding": "No completed local CIFAR10 follow-up result archive found; the notebook describes a search, not results.", 
        "decision": "Do not invent a follow-up winner or import a score from its source campaign."
    }
]
_COMMON_ADAPTATIONS = [
    "Fresh initialization: no historical checkpoints are loaded.", 
    "CIFAR10 labels use the sealed schedule's incremental vocabulary, not original class IDs.", 
    "Five tasks share the same per-task epoch budget; historical full-data epochs are not replayed.", 
    "All evaluation feedback comes from the official test set; these are test-guided scores.", 
    "Raw-network epsilon objectives use standardize, clipped_cosine, 1000 timesteps and modify_first_t=True."
]


def _wrapper(classifier_cap: int | None = None) -> dict:
    """Construct the common diffusion convention and optional joint classifier view."""

    wrapper = {
        "preprocess_type": "standardize", "scheduler_name": "clipped_cosine", 
        "modify_first_t": True, "swap_noise_image": False, "noise_loss_coef": 1.0, 
        "image_loss_coef": 0.0, "noise_distil_loss_coef": 0.0, 
        "p_uncond": 0.1, "use_ema": False, "test_network_name": "raw", 
        "test_steps": 50, "test_cfg_scale": 3.0, "test_eta": 0.0, 
        "train_noisified_max_timesteps": 1000, "trainable_teacher": False, 
        "defer_teacher": False
    }
    # Only joint recipes attach the classifier objective and its paired input views.
    if classifier_cap is not None:
        wrapper.update({
            "clf_loss_coef": 0.00215, "clf_acc_coef": 1.0, 
            "clf_distil_loss_coef": 0.0, "clf_distil_acc_coef": 0.0, 
            "clf_distil_type": "soft", "clf_distil_temperature": 2.0, 
            "clf_train_noisy_input_type": "noisy", 
            "clf_train_class_input_type": "null_class_only", "clf_train_batch_fraction": 0.0, 
            "clf_train_noisified_max_timesteps": classifier_cap, 
            "clf_test_noisified_max_timesteps": classifier_cap, 
            "mask_by_nulls": False, "mask_by_t_threshold": False
        })
    return wrapper


def _config(
    model_family: str, 
    network: dict, 
    wrapper: dict, 
    learning_rate: float = 0.001, 
    batch_size: int = 128, 
    optimizer: str = "adam", 
    weight_decay: float | None = None
) -> dict:
    """Create a native Config mapping whose teachers can train without other artifacts."""

    joint = model_family == "dit_classifier"
    return {
        "model": {
            "name": model_family, 
            "wrapper_name": "diffusion_classifier" if joint else "diffusion_model", 
            "kwargs": deepcopy(network), "wrapper_kwargs": deepcopy(wrapper), 
            "weights_path": None, "loss_function": "mse", "show_network_summary": False
        }, 
        "dataset": {
            "name": "cifar10", "batch_size": batch_size, "validation_source": "test", 
            "validation_ratio": 0.0, "onehot_labels": False, "drop_remainder": False
        }, 
        "optimizer": {
            "name": optimizer, "initial_learning_rate": learning_rate, 
            "weight_decay": weight_decay, "schedule": "constant", "plateau_jump": False
        }, 
        "training": {
            "task": "joint" if joint else "generation", "epochs": 20, "fit_method": "fit", 
            "patience": 0, "reduce_lr_patience": 0, "dtype_policy": "float32", 
            "use_valset": True
        }, 
        "reporting": {
            "run_trainset_eval": False, "run_valset_eval": True, 
            "save_final_images": False, "save_final_gifs": False
        }
    }


def _joint_network(classifier_depth: int, classifier_droppath: float) -> dict:
    """Retain the measured Copy165/155 architecture with an incremental class vocabulary."""

    return {
        "image_size": 32, "channels": 3, "num_classes": None, "use_cfg": True, 
        "timesteps": 1000, "patch_size": 2, "dim": 128, "dim_forced": True, 
        "depth": 8, "mha_num_heads": 4, "vit_block_mlp_ratio": 4.0, 
        "patchify_with_cnn": True, "patches_pos_embed_type": "2d_sincos", 
        "patches_pos_merger_type": "add", "conds_merger_type": "add", 
        "droppath_rate": 0.0, "use_refiner_cnn": False, 
        "feature_aggregation_ids_dict": {1: [-1]}, "clf_connection_ids_dict": {-1: [-1]}, 
        "clf_depth": classifier_depth, "clf_mha_num_heads": 4, 
        "clf_vit_block_mlp_ratio": 4.0, "clf_droppath_rate": classifier_droppath, 
        "clf_cls_token_type": "new_weight", "classifier_only_cls_token": True, 
        "classifier_only_distil_token": True, "force_global_avg_pooling": True, 
        "classifier_dropout_rate": 0.5, "classifier_mlp_activation_func": "tanh"
    }


def _joint_provenance(parent: int, cap: int) -> dict:
    """Describe the observed source and distinguish transferred noisy evaluation."""

    source = f"{_JOINT_EXPORT}/results/copy{'142' if cap == 0 else '136'}/config.json"
    adaptations = list(_COMMON_ADAPTATIONS)
    adaptations.extend([
        "Search samples student optimizer and batch size; source used Adam 0.001/cosine, batch 128, 100 epochs.", 
        "Active continual classifier KD is proposed soft T=2, coefficient 0.001075; noise KD is proposed 0.5.", 
        "These KD coefficients transfer a weak-classifier scale and a new noise-loss balance; they are not measured source optima."
    ])
    # Noisy transfer evaluation differs from the source's separate ensemble protocol.
    if cap:
        adaptations.append(
            "Noisy evaluation uses one sampled timestep below 64. Source native evaluation used t0; "
            "its reported noisy inference used a separate 32-view ensemble below 128. These are different estimands."
        )
    # The efficient head has screening evidence but no matched clean confirmation.
    if parent == 155:
        source = f"{_JOINT_SCREEN}/config.json"
        metrics = {"screen_fixed_ensemble_accuracy": 0.8065, "screen_mean_class_mmd": 0.044606}
        adaptations.append(
            "Copy155's depth 4 classifier is a lower-cost single-seed candidate; its clean variant is unmeasured."
        )
        status = "promising_single_seed_architecture_transfer"
    # The larger head retains evidence from paired five-seed source confirmations.
    else:
        metrics = {
            "source_clean_native_accuracy_mean": 0.85588, 
            "source_clean_native_accuracy_sd": 0.00316, 
            "source_noisy_fixed_ensemble_accuracy_mean": 0.81944, 
            "source_noisy_fixed_ensemble_accuracy_sd": 0.00279, 
            "source_confirmation_seeds": 5
        }
        status = "paired_five_seed_architecture_transfer"
    return {
        "source_parent": parent, "source_config": source, 
        "source_summary": f"{_JOINT_EXPORT}/SUMMARY.md", "evidence_status": status, 
        "source_metrics": metrics, "adaptations": adaptations, "initialization": "fresh"
    }


def candidate_profile(task_seed: int = 42) -> dict:
    """Return four student, three classifier and three epsilon-teacher candidates.

    Students pair clean train/test cap 0 with noisy train/test cap 64 at two head
    depths. All have equal per-task epoch budgets. Batch 64/128 is a new runtime
    choice around the historical 128, not a claim that every combination fits.
    Native teachers retain their own optimizers but use the learner's current
    real-task batches and epoch budget; descriptor dataset sizes do not give
    them a separate training allowance. The builder chooses classifier KD inputs
    from the shared support of selected teachers: mixed clean/noisy roles use
    clean targets, while compatible noisy roles share their supported noise cap.
    This routing does not change the independent epsilon-distillation inputs.

    CIFAR100 winners are deliberately excluded from the epsilon-teacher pool:
    their 5000-step linear, often CFG-disabled setting differs from this common
    1000-step clipped-cosine CFG protocol. Cross-dataset losses cannot rank them
    against CIFAR10 teachers. The CIFAR100 follow-up local-hybrid trial 82 had
    mean 0.013252885 over three seeds, but does not establish continual transfer.
    No completed local CIFAR10 follow-up result archive was found in this audit.
    """

    students = {}
    for parent, depth, drop_path in [(165, 8, 0.05), (155, 4, 0.0)]:
        for condition, cap in [("clean", 0), ("noisy", 64)]:
            wrapper = _wrapper(cap)
            wrapper.update({
                "clf_distil_loss_coef": 0.001075, "noise_distil_loss_coef": 0.5, 
                "previous_teacher_clf_loss_weight": 1.0, "current_teacher_clf_loss_weight": 1.0, 
                "previous_teacher_noise_loss_weight": 1.0, "current_teacher_noise_loss_weight": 1.0
            })
            students[f"joint{parent}_{condition}"] = {
                "config": _config("dit_classifier", _joint_network(depth, drop_path), wrapper), 
                "provenance": _joint_provenance(parent, cap), 
                "batch_sizes": [64, 128], "epoch_multiplier": 1.0
            }
    classifiers = {}
    for condition, cap in [("clean", 0), ("noisy", 64)]:
        provenance = _joint_provenance(165, cap)
        provenance["adaptations"] = list(_COMMON_ADAPTATIONS) + [
            "Independent current-task joint specialist has no distillation; source architecture trains from scratch.", 
            "Native classifier distillation queries the requested student timestep; it is not a teacher ensemble.", 
            "Teacher keeps Adam 0.001 with constant rate for the shorter current-task budget instead of full-data cosine."
        ]
        classifiers[f"joint165_{condition}"] = {
            "descriptor": {
                "format": "config", 
                "config": _config("dit_classifier", _joint_network(8, 0.05), _wrapper(cap))
            }, 
            "provenance": provenance
        }
    classifiers["joint135_clean_transfer"] = {
        "descriptor": {
            "format": "config", 
            "config": _config("dit_classifier", _joint_network(8, 0.0), _wrapper(0))
        }, 
        "provenance": {
            "source_config": f"{_JOINT_EXPORT}/results/copy196/config.json", 
            "source_summary": f"{_JOINT_EXPORT}/SUMMARY.md", 
            "evidence_status": "tentative_architecture_only_transfer", 
            "source_metrics": {
                "clean_native_accuracy_mean": 0.85630, "clean_native_accuracy_sd": 0.00505, 
                "mean_class_mmd": 0.041837, "confirmation_seeds": 5
            }, 
            "initialization": "fresh", "adaptations": list(_COMMON_ADAPTATIONS) + [
                "Remove source B0 distillation and progressive 20/20/20/40 schedule; no B0 artifact is supplied.", 
                "Only its zero classifier DropPath architectural prior transfers; historical accuracy does not validate this new recipe.", 
                "Teacher keeps Adam 0.001 constant and the same current-task budget as other candidates."
            ]
        }
    }
    unet_network = {
        "image_size": 32, "channels": 3, "num_classes": None, "use_cfg": True, 
        "timesteps": 1000, "widths": [64, 128, 256], "block_depth": 3, 
        "bottleneck_width": 384, "bottleneck_depth": 2, "activation_func": "gelu", 
        "image_embedding_dim": 21, "label_embedding_dim": 21, "time_embedding_dim": 22, 
        "downsampling_method": "avg_pooling", "upsampling_method": "cnn_transpose", 
        "upsampling_interpolation": "bilinear", "use_batch_norm": True, 
        "dropout_rate": 0.0, "use_skip_connections": True, "final_activation_func": "linear"
    }
    generation_wrapper = _wrapper()
    generation_wrapper["p_uncond"] = 0.2
    unet = _config(
        "unet", unet_network, generation_wrapper, learning_rate=0.0009232173954634668, 
        batch_size=64, optimizer="adamw", weight_decay=1.0569549424860102e-5
    )
    dit_network = {
        "image_size": 32, "channels": 3, "num_classes": None, "use_cfg": True, 
        "timesteps": 1000, "dim": 128, "depth": 6, "mha_num_heads": 4, 
        "patch_size": 2, "patchify_with_cnn": True, "vit_block_mlp_ratio": 4.0, 
        "droppath_rate": 0.0, "use_refiner_cnn": True, 
        "patches_pos_embed_type": "2d_sincos", "ln_no_adaptation": False
    }
    dit = _config("diffusion_transformer", dit_network, generation_wrapper, learning_rate=0.004845009591308682)
    dit["optimizer"]["global_clipnorm"] = 1.0
    noise = {
        "unet23": {
            "descriptor": {"format": "config", "config": unet}, 
            "provenance": {
                "source_config": f"{_UNET_RUNNER}/finalist_configs/trial-0023.yaml", 
                "source_summary": f"{_UNET_RUNNER}/confirmation_summary.json", 
                "source_metrics": {"noise_loss_mean": 0.029328910013039906, "noise_loss_sd": 0.00012460615226514118, "seeds": 3}, 
                "evidence_status": "near_best_confirmed_cifar10_denoiser_transfer", "initialization": "fresh", 
                "adaptations": list(_COMMON_ADAPTATIONS) + [
                    "Source EMA 0.999 is disabled and modify_first_t=False becomes True to match students.", 
                    "Source 50 epochs/cosine becomes shared per-task epochs/constant; original AdamW rate and decay retained.", 
                    "Trial 64 bilinear counterpart mean 0.029323363 is practically tied; keep one to limit a redundant search branch.", 
                    "UNet archive explicitly excludes all checkpoints; this descriptor recreates its architecture."
                ]
            }
        }, 
        "dit_cifar10_1351": {
            "descriptor": {"format": "config", "config": dit}, 
            "provenance": {
                "source_config": f"{_DIT_RUNNER}/finalist_configs/trial-1351.yaml", 
                "source_summary": f"{_DIT_RUNNER}/confirmation_summary.json", 
                "source_metrics": {"noise_loss_mean": 0.030521333838502567, "noise_loss_sd": 0.0002750712503580498, "seeds": 3}, 
                "evidence_status": "best_confirmed_cifar10_dit_transfer", "initialization": "fresh", 
                "adaptations": list(_COMMON_ADAPTATIONS) + [
                    "Source EMA 0.999 is disabled and modify_first_t=False becomes True to match students.", 
                    "Source 50 epochs/cosine becomes shared per-task epochs/constant; source Adam rate and global clipping retained.", 
                    "Near-best trial 1770 lacks the refiner and mean 0.030541766 overlaps across seeds; omit redundant variants.", 
                    "Denoising loss is a proxy, not measured replay utility or a cross-dataset image-quality ranking."
                ]
            }
        }, 
        "joint165_noisy": deepcopy(classifiers["joint165_noisy"])
    }
    noise["joint165_noisy"]["provenance"]["adaptations"].append(
        "Use the native joint model's epsilon output as the noise teacher; classifier output is not the target."
    )
    return {
        "student_candidates": students, "classifier_teacher_candidates": classifiers, 
        "noise_teacher_candidates": noise, "task_seed": task_seed
    }
