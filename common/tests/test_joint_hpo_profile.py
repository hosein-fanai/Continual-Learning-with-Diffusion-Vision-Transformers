"""Check the offline joint-classifier search contract and executable edge cases."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import load_config, save_config
from common.hpo_profiles import JOINT_CLASSIFIER_SEARCH_SPACE, build_joint_classifier_config
from diffusion.models.transformer.di_t_classifier import DiTClassifier
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier


class _Trial:
    """Record deterministic first-choice suggestions through the real adapter."""

    number = 7

    def __init__(self):
        self.params = {}

    def suggest_categorical(self, name, choices):
        self.params[name] = choices[0]
        return choices[0]

    def suggest_float(self, name, low, high, **kwargs):
        self.params[name] = low
        return low


class JointClassifierProfileTests(unittest.TestCase):
    def make_config(self, overrides=None, **kwargs):
        return build_joint_classifier_config(
            _Trial(), dataset_name="cifar10", epochs=50, seed=17,
            results_path="results/profile-test", search_space_overrides=overrides,
            **kwargs,
        )

    def test_v1_classifies_all_clean_images_without_label_conditioning(self):
        config = self.make_config()
        wrapper = config.model.wrapper_kwargs
        self.assertEqual(config.training.task, "joint")
        self.assertEqual(config.model.kwargs["num_classes"], 10)
        self.assertEqual(wrapper["clf_train_type"], "uncond")
        self.assertEqual(wrapper["clf_train_noisy_input_type"], "clean")
        self.assertEqual(wrapper["clf_train_class_input_type"], "null_class_only")
        self.assertNotIn("train_cfg_scale", wrapper)
        self.assertNotIn("test_cfg_scale", wrapper)
        self.assertFalse(wrapper["mask_by_nulls"])
        self.assertEqual(wrapper["clf_loss_coef"], 1.0)
        self.assertFalse(config.model.kwargs["aggregate_from_noises"])
        self.assertFalse(wrapper["use_ema"])
        self.assertEqual(wrapper["test_network_name"], "raw")
        self.assertFalse(wrapper["mask_by_t_threshold"])
        self.assertNotIn("test_noisified_min_timesteps", wrapper)
        self.assertNotIn("test_noisified_max_timesteps", wrapper)
        self.assertFalse(config.reporting.evaluate_ensemble_accuracy)
        self.assertFalse(config.hpo["use_ensemble_accuracy"])
        self.assertEqual(config.hpo["accuracy_metric"], "classification_accuracy")
        self.assertEqual(config.reporting.ensemble_accuracy_kwargs, {})
        self.assertEqual(config.hpo["ensemble_accuracy_kwargs"], {})
        self.assertEqual(config.hpo["profile_version"], 10)
        self.assertEqual(config.hpo["objective_network"], "raw")
        self.assertNotIn("clf_train_noisified_max_timesteps", config.hpo["params"])
        self.assertIsNone(config.optimizer.clipnorm)
        self.assertIsNone(config.optimizer.global_clipnorm)
        self.assertFalse(config.optimizer.plateau_jump)
        self.assertEqual(config.training.reduce_lr_patience, 0)
        self.assertEqual(config.training.patience, 0)
        self.assertEqual(config.training.dtype_policy, "float32")
        self.assertEqual(config.dataset.preprocess, "fixed-standardize")
        self.assertEqual(config.hpo["epoch_budget"]["maximum_total_epochs"], 50)
        self.assertEqual(config.dataset.validation_source, "split")
        self.assertEqual(config.dataset.validation_ratio, 0.2)
        self.assertFalse(config.dataset.drop_remainder)
        self.assertEqual(config.hpo["objective_metrics"], ["classification_accuracy", "noise_loss"])
        self.assertEqual(config.hpo["objective_directions"], ["maximize", "minimize"])

    def test_tmcl_search_has_only_v1_and_nonvariational_classifier_heads(self):
        self.assertEqual(JOINT_CLASSIFIER_SEARCH_SPACE["classifier_mlp_ratio"], [1, 2, 4])
        for option in ("clipnorm", "global_clipnorm", "clf_train_noisified_max_timesteps",
                       "clf_test_noisified_max_timesteps", "aggregate_from_noises", "clf_loss_coef",
                       "wrapper_name", "learning_rate_schedule", "batch_size", "patchify_with_cnn",
                       "modify_first_t"):
            self.assertNotIn(option, JOINT_CLASSIFIER_SEARCH_SPACE)
            self.assertNotIn(option, self.make_config().hpo["params"])
        for ratio in (1, 2, 4):
            with self.subTest(ratio=ratio):
                config = self.make_config({"classifier_mlp_ratio": [ratio]})
                self.assertEqual(config.model.kwargs["classifier_mlp_ratio"], ratio)
                self.assertEqual(config.model.wrapper_name, "diffusion_classifier")
                self.assertTrue(config.model.kwargs["patchify_with_cnn"])
                self.assertEqual(config.optimizer.schedule, "cosine")
                self.assertEqual(config.dataset.batch_size, 128)
                self.assertEqual(config.hpo["epoch_budget"]["maximum_total_epochs"], 50)
                self.assertEqual(config.hpo["epoch_budget"]["joint"], 50)
                for option in ("clf_train_noisified_max_timesteps", "clf_test_noisified_max_timesteps",
                               "modify_first_t"):
                    self.assertNotIn(option, config.model.wrapper_kwargs)

    def test_dropout_and_drop_path_retain_independent_middle_choices(self):
        self.assertEqual(JOINT_CLASSIFIER_SEARCH_SPACE["dropout_rate"], [0.0, 0.15, 0.25])
        self.assertEqual(JOINT_CLASSIFIER_SEARCH_SPACE["clf_drop_prob"], [0.0, 0.15, 0.25])
        for dropout in (0.0, 0.15, 0.25):
            for drop_path in (0.0, 0.15, 0.25):
                with self.subTest(dropout=dropout, drop_path=drop_path):
                    config = self.make_config({"dropout_rate": [dropout], "clf_drop_prob": [drop_path]})
                    self.assertEqual(config.model.kwargs["dropout_rate"], dropout)
                    self.assertEqual(config.model.kwargs["clf_drop_prob"], drop_path)

    def test_data_protocol_is_an_explicit_option(self):
        default = self.make_config()
        self.assertFalse(default.hpo["fixed_recipe"]["test_set_used_for_hpo"])
        for ratio in (0.0, 0.2):
            config = self.make_config(validation_source="test", validation_ratio=ratio)
            self.assertEqual(config.dataset.validation_source, "test")
            self.assertFalse(config.dataset.drop_remainder)
            self.assertTrue(config.hpo["fixed_recipe"]["test_set_used_for_hpo"])
            self.assertEqual(config.hpo["fixed_recipe"]["validation_ratio"], 0.0)
        split = self.make_config(validation_source="split", validation_ratio=0.1)
        self.assertEqual(split.dataset.validation_ratio, 0.1)
        self.assertFalse(split.hpo["fixed_recipe"]["test_set_used_for_hpo"])
        for options in ({"validation_source": "typo"}, {"validation_ratio": 0.0},
                        {"validation_ratio": -0.1}, {"validation_ratio": float("nan")},
                        {"validation_ratio": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.make_config(**options)

    def test_aggregation_uses_internal_features_and_all_features_project(self):
        config = self.make_config({"feature_aggregation": ["all"], "dim": [256]})
        self.assertEqual(config.model.kwargs["feature_aggregation_ids_dict"], {1: [None]})
        self.assertEqual(config.model.kwargs["clf_dim"], 256)
        self.assertTrue(config.model.kwargs["clf_dim_forced"])
        config = self.make_config({"feature_aggregation": ["last"]})
        self.assertEqual(config.hpo["params"]["feature_aggregation"], "last")
        self.assertFalse(config.model.kwargs["aggregate_from_noises"])
        self.assertIsNone(config.model.kwargs["clf_dim"])

    def test_requested_edges_preserve_full_classifier_weight_and_epoch_budget(self):
        config = self.make_config({
            "mha_num_heads": [6], "dim": [256], "depth": [7], "clf_depth": [5],
            "clf_cond_type": [None], "dropout_rate": [0.25], "clf_drop_prob": [0.25],
            "classifier_mlp_ratio": [4],
        })
        self.assertTrue(config.model.kwargs["clf_ln_no_adaptation"])
        self.assertEqual(config.model.kwargs["mha_num_heads"], 6)
        self.assertEqual(config.model.kwargs["clf_mha_num_heads"], 4)
        self.assertIsNone(config.optimizer.clipnorm)
        self.assertNotIn("clf_loss_coef", config.hpo["params"])
        self.assertEqual(config.model.wrapper_kwargs["clf_loss_coef"], 1.0)
        self.assertFalse(config.training.ensemble_monitor)
        self.assertEqual(config.training.patience, 0)
        self.assertEqual(config.training.epochs, 50)
        self.assertEqual(config.training.reduce_lr_patience, 0)
        self.assertEqual(config.training.monitor, "val_classifier_accuracy")
        self.assertNotIn("modify_first_t", config.model.wrapper_kwargs)
        native = self.make_config(wrapper_overrides={"modify_first_t": False})
        self.assertFalse(native.model.wrapper_kwargs["modify_first_t"])

    def test_weight_decay_is_adamw_only_and_classifier_weight_stays_fixed(self):
        adam = self.make_config()
        self.assertIsNone(adam.optimizer.weight_decay)
        self.assertNotIn("weight_decay", adam.hpo["params"])
        adamw = self.make_config({"optimizer": ["adamw"]})
        self.assertGreater(adamw.optimizer.weight_decay, 0.0)
        self.assertNotIn("clf_loss_coef", adamw.hpo["params"])
        self.assertEqual(adamw.model.wrapper_kwargs["clf_loss_coef"], 1.0)

    def test_configuration_round_trip_preserves_cifar100_and_inputs(self):
        overrides = {"classifier_mlp_ratio": [2]}
        ensemble_options = {}
        inputs = deepcopy((overrides, ensemble_options))
        config = build_joint_classifier_config(
            _Trial(), dataset_name="CIFAR100", epochs=50, seed=17,
            results_path="results/profile-test", dtype_policy="float32",
            validation_source="test", validation_ratio=0.0,
            search_space_overrides=overrides, ensemble_accuracy_kwargs=ensemble_options,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trial.yaml"
            save_config(config, path)
            restored = load_config(path)
        self.assertEqual(asdict(config), asdict(restored))
        self.assertEqual((overrides, ensemble_options), inputs)
        self.assertEqual(restored.model.kwargs["num_classes"], 100)
        self.assertTrue(restored.hpo["fixed_recipe"]["test_set_used_for_hpo"])
        self.assertFalse(restored.hpo["fixed_recipe"]["independent_test_estimate"])
        self.assertEqual(restored.dataset.validation_source, "test")
        self.assertEqual(restored.dataset.validation_ratio, 0.0)
        self.assertTrue(restored.reporting.save_final_images)
        self.assertFalse(restored.reporting.save_final_gifs)
        self.assertEqual(restored.reporting.final_generation_network_name, "raw")
        self.assertTrue(restored.reporting.final_generation_add_null_label)
        self.assertEqual(restored.reporting.final_generation_modes, [
            {"name": "quick_scale3", "steps": 50, "scale": 3.0, "eta": 0.0},
        ])
        self.assertEqual(restored.reporting.final_images_steps, 50)
        self.assertIsNone(restored.hpo["checkpoint_selection_metric"])
        self.assertEqual(restored.hpo["checkpoint_selection_policy"], "final_epoch")

    def test_profile_rejects_misspelled_or_contract_replacing_overrides(self):
        with self.assertRaisesRegex(ValueError, "Unknown.*overrides"):
            self.make_config({"headz": [6]})
        with self.assertRaisesRegex(ValueError, "replaces profile"):
            self.make_config(model_overrides={"clf_cls_token_type": None})
        with self.assertRaisesRegex(ValueError, "replaces profile"):
            self.make_config(wrapper_overrides={"noise_distil_loss_coef": 1.0})
        for options in ({"network_name": "ema"}, {"max_t": 128}, {"t_chunk_size": 8}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "ordinary.*accuracy"):
                self.make_config(ensemble_accuracy_kwargs=options)
        for override in ({"clf_train_type": "cond"}, {"train_cfg_scale": 1.0},
                         {"test_cfg_scale": 1.0}, {"swap_noise_image": True}):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "replaces profile"):
                self.make_config(wrapper_overrides=override)

    def test_tmcl_incompatible_overrides_cannot_reenter_the_profile(self):
        for override in (
            {"wrapper_name": ["diffusion_classifier_v2"]}, {"classifier_mlp_ratio": [None]},
            {"clipnorm": [1.0]}, {"clf_train_noisified_max_timesteps": [32]},
            {"clf_test_noisified_max_timesteps": [32]},
            {"aggregate_from_noises": [True]}, {"clf_loss_coef": [0.001]},
            {"wrapper_name": ["diffusion_classifier"]}, {"learning_rate_schedule": ["cosine"]},
            {"batch_size": [128]}, {"patchify_with_cnn": [True]}, {"modify_first_t": [False]},
        ):
            with self.subTest(search=override), self.assertRaises(ValueError):
                self.make_config(override)
        for policy in ("mixed_bfloat16", "mixed_float16", "float64"):
            with self.subTest(dtype_policy=policy), self.assertRaises(ValueError):
                self.make_config(dtype_policy=policy)
        for override in (
            {"use_ema": True}, {"test_network_name": "ema"},
            {"clf_train_noisified_max_timesteps": 32}, {"clf_test_noisified_max_timesteps": 32},
            {"test_noisified_min_timesteps": 1}, {"test_noisified_max_timesteps": 32},
            {"dtype": "mixed_bfloat16"},
            {"clf_train_noisy_input_type": "noisy"}, {"clf_train_type": "cond"},
            {"clf_train_class_input_type": "all_classes"},
            {"mask_by_nulls": True}, {"mask_by_t_threshold": True},
            {"clf_loss_coef": 0.001}, {"use_ensemble_loss_instead": True},
            {"modify_first_t": True},
        ):
            with self.subTest(wrapper=override), self.assertRaises(ValueError):
                self.make_config(wrapper_overrides=override)
        for override in (
            {"classifier_mlp_ratio": None},
            {"aggregate_from_noises": True},
            {"patchify_with_cnn": False},
            {"dtype": "mixed_bfloat16"},
            {"compile_args": {"optimizer": tf.keras.optimizers.Adam(clipnorm=1.0)}},
            {"compile_args": {"loss": "mae"}},
            {"compile_args": "invalid"},
            {"reshaper_ids_dict": {1: "flatten"}, "reshaper_kwargs": {"add_kl": True}},
            {"clf_reshaper_ids_dict": {1: "flatten"}, "clf_reshaper_kwargs": {"add_kl": True}},
        ):
            with self.subTest(model=override), self.assertRaises(ValueError):
                self.make_config(model_overrides=override)

    def test_real_network_edge_routes_build_without_distillation(self):
        # Exercise the combinations most likely to fail shape/condition checks,
        # while retaining the smallest allowed width/depth for a focused check.
        for aggregation in ("all", "last"):
            with self.subTest(aggregation=aggregation):
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(17)
                config = self.make_config({
                    "mha_num_heads": [6], "dim": [32], "patch_size": [4],
                    "clf_cond_type": [None], "feature_aggregation": [aggregation],
                })
                network = DiTClassifier(**config.model.kwargs, seed=17)
                probabilities = network.predict_class((
                    tf.zeros((2, 32, 32, 3)), tf.zeros((2,), tf.int32),
                    tf.zeros((2,), tf.int32),
                ), training=False)
                if isinstance(probabilities, (tuple, list)):
                    probabilities = probabilities[0]
                self.assertEqual(tuple(probabilities.shape), (2, 10))
                self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(probabilities))))
                self.assertTrue(network.clf_has_cls_token)
                self.assertFalse(network.clf_has_distil_token)
                self.assertFalse(network.dynamic_num_classes)
                self.assertTrue(network.patchify_with_cnn)

    def test_real_wrapper_uses_all_clean_rows_and_full_noise_evaluation(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(17)
        config = self.make_config({"patch_size": [4]})
        network = DiTClassifier(**config.model.kwargs, seed=17)
        wrapper = DiffusionClassifier(network=network, **config.model.wrapper_kwargs, seed=17)
        wrapper.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")
        self.assertEqual(wrapper.clf_train_type, "uncond")
        self.assertEqual(wrapper.clf_train_noisy_input_type, "clean")
        self.assertEqual(wrapper.clf_train_class_input_type, "null_class_only")
        self.assertIsNone(wrapper.train_cfg_scale)
        self.assertEqual(wrapper.test_noisified_min_timesteps, 0)
        self.assertEqual(wrapper.test_noisified_max_timesteps, network.timesteps)
        self.assertFalse(wrapper.mask_by_nulls)
        self.assertFalse(wrapper.modify_first_t)
        self.assertFalse(wrapper.use_ema)
        self.assertIsNone(wrapper.ema_network)
        images = tf.random.stateless_uniform((2, 32, 32, 3), (17, 19), -1.0, 1.0)
        classes = tf.constant([2, 7])
        with patch.object(network, "predict_class", wraps=network.predict_class) as classify:
            wrapper.train_step((images, classes))
        classify.assert_called_once()
        class_images, class_times, class_labels = classify.call_args.args[0]
        np.testing.assert_array_equal(class_images.numpy(), images.numpy())
        np.testing.assert_array_equal(class_times.numpy(), [0, 0])
        np.testing.assert_array_equal(class_labels.numpy(), [0, 0])
        self.assertTrue(classify.call_args.kwargs["training"])


if __name__ == "__main__":
    unittest.main()
