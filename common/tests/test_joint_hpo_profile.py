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
from common.hpo_profiles import build_joint_classifier_config
from diffusion.models.transformer.di_t_classifier import DiTClassifier
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier
from diffusion.models.wrapper.diffusion_classifier_v2 import DiffusionClassifierV2


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

    def test_v1_retains_native_conditioning_and_uses_null_mask(self):
        config = self.make_config()
        wrapper = config.model.wrapper_kwargs
        self.assertEqual(config.training.task, "joint")
        self.assertEqual(config.model.kwargs["num_classes"], 10)
        self.assertNotIn("clf_train_type", wrapper)
        self.assertNotIn("train_cfg_scale", wrapper)
        self.assertNotIn("test_cfg_scale", wrapper)
        self.assertTrue(wrapper["mask_by_nulls"])
        self.assertTrue(wrapper["use_ema"])
        self.assertEqual(wrapper["test_network_name"], "ema")
        self.assertFalse(wrapper["mask_by_t_threshold"])
        self.assertNotIn("test_noisified_min_timesteps", wrapper)
        self.assertNotIn("test_noisified_max_timesteps", wrapper)
        self.assertTrue(config.reporting.evaluate_ensemble_accuracy)
        self.assertEqual(config.hpo["accuracy_metric"], "ensemble_accuracy")
        self.assertEqual(config.reporting.ensemble_accuracy_kwargs["max_t"], 128)
        self.assertNotIn("clf_train_noisified_max_timesteps", config.hpo["params"])
        self.assertIsNone(config.optimizer.clipnorm)
        self.assertEqual(config.hpo["epoch_budget"]["maximum_total_epochs"], 50)
        self.assertEqual(config.dataset.validation_source, "split")
        self.assertEqual(config.dataset.validation_ratio, 0.2)
        self.assertFalse(config.dataset.drop_remainder)
        self.assertEqual(config.hpo["objective_metrics"], ["classification_accuracy", "noise_loss"])
        self.assertEqual(config.hpo["objective_directions"], ["maximize", "minimize"])

    def test_v2_caps_control_both_inputs_and_accuracy(self):
        for cap in (None, 32, 128, 256, 512):
            with self.subTest(cap=cap):
                config = self.make_config({
                    "wrapper_name": ["diffusion_classifier_v2"],
                    "clf_train_noisified_max_timesteps": [cap],
                }, ensemble_accuracy_kwargs={"max_t": 1000, "t_chunk_size": 16})
                wrapper = config.model.wrapper_kwargs
                self.assertNotIn("test_noisified_min_timesteps", wrapper)
                self.assertNotIn("test_noisified_max_timesteps", wrapper)
                self.assertEqual(wrapper["clf_train_noisified_max_timesteps"], cap)
                self.assertEqual(wrapper["clf_test_noisified_max_timesteps"], cap)
                self.assertEqual(config.reporting.evaluate_ensemble_accuracy, cap is not None)
                self.assertEqual(config.hpo["accuracy_metric"],
                                 "classification_accuracy" if cap is None else "ensemble_accuracy")
                if cap is not None:
                    self.assertEqual(config.reporting.ensemble_accuracy_kwargs["max_t"], cap)
                self.assertEqual(config.hpo["epoch_budget"]["maximum_total_epochs"], 100)
                self.assertEqual(wrapper["clf_vars_embedding_ids"], [])
                self.assertEqual(wrapper["clf_vars_noise_part_ids"], [])
                self.assertEqual(wrapper["clf_loss_coef"], 1.0)
                self.assertNotIn("clf_loss_coef", config.hpo["params"])

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

    def test_aggregation_dimension_is_conditional_and_all_features_project(self):
        config = self.make_config({"feature_aggregation": ["all"], "dim": [256]})
        self.assertEqual(config.model.kwargs["feature_aggregation_ids_dict"], {1: [None]})
        self.assertEqual(config.model.kwargs["clf_dim"], 256)
        self.assertTrue(config.model.kwargs["clf_dim_forced"])
        config = self.make_config({"aggregate_from_noises": [True]})
        self.assertNotIn("feature_aggregation", config.hpo["params"])
        self.assertIsNone(config.model.kwargs["clf_dim"])

    def test_requested_edges_include_v1_loss_search_and_first_timestep(self):
        config = self.make_config({
            "mha_num_heads": [6], "dim": [256], "depth": [7], "clf_depth": [5],
            "clf_cond_type": [None], "dropout_rate": [0.25], "clf_drop_prob": [0.25],
            "classifier_mlp_ratio": [4], "clipnorm": [1.0], "batch_size": [256],
        })
        self.assertTrue(config.model.kwargs["clf_ln_no_adaptation"])
        self.assertEqual(config.model.kwargs["mha_num_heads"], 6)
        self.assertEqual(config.model.kwargs["clf_mha_num_heads"], 4)
        self.assertEqual(config.optimizer.clipnorm, 1.0)
        self.assertEqual(config.hpo["params"]["clf_loss_coef"], 0.001)
        self.assertEqual(config.model.wrapper_kwargs["clf_loss_coef"], 0.001)
        self.assertFalse(config.training.ensemble_monitor)
        self.assertEqual(config.training.patience, 10)
        self.assertEqual(config.training.reduce_lr_patience, 5)
        self.assertEqual(config.training.monitor, "val_classifier_accuracy")
        for coefficient in (0.001, 0.01, 0.1, 0.25, 0.5, 1.0):
            for modify in (True, False):
                variant = self.make_config({"clf_loss_coef": [coefficient], "modify_first_t": [modify]})
                self.assertEqual(variant.model.wrapper_kwargs["clf_loss_coef"], coefficient)
                self.assertEqual(variant.model.wrapper_kwargs["modify_first_t"], modify)

    def test_weight_decay_is_adamw_only_and_v2_ignores_inactive_loss_dimension(self):
        adam = self.make_config()
        self.assertIsNone(adam.optimizer.weight_decay)
        self.assertNotIn("weight_decay", adam.hpo["params"])
        adamw = self.make_config({
            "optimizer": ["adamw"], "clf_loss_coef": [0.1, 0.25, 1.0],
        })
        self.assertGreater(adamw.optimizer.weight_decay, 0.0)
        self.assertEqual(adamw.hpo["params"]["clf_loss_coef"], 0.1)
        v2 = self.make_config({"wrapper_name": ["diffusion_classifier_v2"],
                               "clf_loss_coef": [0.001]})
        self.assertEqual(v2.model.wrapper_kwargs["clf_loss_coef"], 1.0)
        self.assertNotIn("clf_loss_coef", v2.hpo["params"])

    def test_configuration_round_trip_preserves_cifar100_and_inputs(self):
        overrides = {"wrapper_name": ["diffusion_classifier_v2"],
                     "clf_train_noisified_max_timesteps": [128]}
        ensemble_options = {"max_t": 1000, "t_chunk_size": 8}
        inputs = deepcopy((overrides, ensemble_options))
        config = build_joint_classifier_config(
            _Trial(), dataset_name="CIFAR100", epochs=50, seed=17,
            results_path="results/profile-test", dtype_policy="mixed_bfloat16",
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
        self.assertTrue(restored.reporting.save_final_gifs)
        self.assertEqual(restored.reporting.final_generation_network_name, "ema")
        self.assertTrue(restored.reporting.final_generation_add_null_label)
        self.assertEqual(restored.reporting.final_generation_modes, [
            {"name": "full_stochastic_scale3", "steps": 1000, "scale": 3.0, "eta": 1.0},
            {"name": "full_default_eta_scale3", "steps": 1000, "scale": 3.0, "eta": None},
            {"name": "default_scale3", "steps": None, "scale": 3.0, "eta": None},
            {"name": "default_scale4", "steps": None, "scale": 4.0, "eta": None},
        ])

    def test_profile_rejects_misspelled_or_contract_replacing_overrides(self):
        with self.assertRaisesRegex(ValueError, "Unknown.*overrides"):
            self.make_config({"headz": [6]})
        with self.assertRaisesRegex(ValueError, "replaces profile"):
            self.make_config(model_overrides={"clf_cls_token_type": None})
        with self.assertRaisesRegex(ValueError, "replaces profile"):
            self.make_config(wrapper_overrides={"noise_distil_loss_coef": 1.0})
        with self.assertRaisesRegex(ValueError, "evaluates EMA"):
            self.make_config(ensemble_accuracy_kwargs={"network_name": "raw"})
        for override in ({"clf_train_type": "uncond"}, {"train_cfg_scale": 1.0},
                         {"test_cfg_scale": 1.0}, {"swap_noise_image": True}):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "replaces profile"):
                self.make_config(wrapper_overrides=override)

    def test_real_network_edge_routes_build_without_distillation(self):
        # Exercise the combinations most likely to fail shape/condition checks,
        # while retaining the smallest allowed width/depth for a focused check.
        for aggregation, noise, cnn in (("all", False, False), ("last", True, True)):
            with self.subTest(aggregation=aggregation, noise=noise, cnn=cnn):
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(17)
                config = self.make_config({
                    "mha_num_heads": [6], "dim": [32], "patch_size": [4],
                    "clf_cond_type": [None], "feature_aggregation": [aggregation],
                    "aggregate_from_noises": [noise], "patchify_with_cnn": [cnn],
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

    def test_real_wrappers_preserve_native_v1_mask_and_null_v2_inputs(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(17)
        config = self.make_config({"patch_size": [4]})
        network = DiTClassifier(**config.model.kwargs, seed=17)
        wrapper = DiffusionClassifier(network=network, **config.model.wrapper_kwargs, seed=17)
        wrapper.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")
        self.assertEqual(wrapper.clf_train_type, "cond")
        self.assertIsNone(wrapper.train_cfg_scale)
        self.assertEqual(wrapper.test_noisified_min_timesteps, 0)
        self.assertEqual(wrapper.test_noisified_max_timesteps, network.timesteps)
        self.assertTrue(wrapper.mask_by_nulls)
        conditional = tf.one_hot([0, 1], 10) * 0.9 + 0.01
        unconditional = tf.ones((2, 10)) * 0.1
        result = wrapper.compute_clf_kl_ctr_distil_loss(
            tf.constant([0, 1]), conditional, None, None,
            classes_pred_u=unconditional, clf_loss_mask=tf.constant([1.0, 0.0]),
        )
        np.testing.assert_allclose(result[5].numpy(), conditional.numpy())
        self.assertAlmostEqual(float(result[1]), float(-np.log(0.91)), places=5)
        images, classes = tf.zeros((2, 32, 32, 3)), tf.constant([0, 1])
        prepared = list(wrapper.prep_inputs_map(images, classes))
        prepared[4] = tf.constant([0, 2])
        with patch.object(wrapper, "_prepare_classifier_batch", return_value=(tuple(prepared), None, None)), \
             patch.object(wrapper, "compute_clf_kl_ctr_distil_loss", wraps=wrapper.compute_clf_kl_ctr_distil_loss) as loss_call:
            wrapper.train_step((images, classes))
        np.testing.assert_array_equal(loss_call.call_args.kwargs["clf_loss_mask"].numpy(), [1.0, 0.0])
        v2_config = self.make_config({
            "wrapper_name": ["diffusion_classifier_v2"], "patch_size": [4],
            "clf_train_noisified_max_timesteps": [None],
        })
        v2 = DiffusionClassifierV2(
            network=network, **v2_config.model.wrapper_kwargs, seed=17
        )
        self.assertEqual(v2.test_noisified_min_timesteps, 0)
        self.assertEqual(v2.test_noisified_max_timesteps, network.timesteps)
        prepared = v2.prep_clfv2_inputs((tf.zeros((2, 32, 32, 3)), tf.constant([2, 7])), 0)
        # This API returns noisy images, timesteps, null labels, and class targets.
        self.assertTrue(bool(tf.reduce_all(prepared[2] == 0)))
        np.testing.assert_array_equal(prepared[3].numpy(), [2, 7])


if __name__ == "__main__":
    unittest.main()
