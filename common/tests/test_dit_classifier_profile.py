"""Focused contracts and native execution checks for the classifier runner recipe."""

from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import Config
from common.dit_classifier_hpo import PROFILE, VERSION, SEARCH_SPACE, baseline_hints, build_dit_classifier_config
from common.hpo_profiles import JOINT_CLASSIFIER_PROFILE_VERSION, build_joint_classifier_config
from diffusion.models.transformer.di_t_classifier import DiTClassifier
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier


class _Trial:
    """Deterministically record the first permitted suggestion and its domain."""

    number = 7

    def __init__(self) -> None:
        """Initialize independent sampled parameters and distribution receipts."""

        self.params = {}
        self.domains = {}

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        """Record a categorical domain and select its first value.

        Args:
            name (str): Parameter name required by the suggestion protocol.
            choices (list[object]): Permitted values.

        Returns:
            object: First permitted value.
        """

        self.domains[name] = deepcopy(choices)
        self.params[name] = choices[0]
        return choices[0]

    def suggest_float(self, name: str, low: float, high: float, **kwargs: object) -> float:
        """Record a numeric domain and select its lower endpoint.

        Args:
            name (str): Parameter name required by the suggestion protocol.
            low (float): Lower endpoint selected by this fixture.
            high (float): Upper endpoint retained in the receipt.
            kwargs (object): Distribution options retained in the receipt.

        Returns:
            float: Lower endpoint.
        """

        self.domains[name] = {"low": low, "high": high, **kwargs}
        self.params[name] = low
        return low


class ClassifierRunnerProfileTests(unittest.TestCase):
    """Check native recipe semantics without loading a dataset or training a study."""

    def make_config(self, overrides: dict | None = None, **kwargs: object) -> Config:
        """Build a deterministic one-hundred-epoch profile fixture.

        Args:
            overrides (dict | None): Restricted search distributions.
            kwargs (object): Additional public builder options.

        Returns:
            Config: Unexecuted experiment configuration.
        """

        return build_dit_classifier_config(
            _Trial(), dataset_name="cifar10", epochs=100, results_path="files/results/profile-test", 
            search_space_overrides=overrides, seed=17, **kwargs
        )

    def test_objectives_feedback_and_callbacks_are_coherent(self) -> None:
        """Evaluate final raw accuracy and noise loss after a common full training budget."""

        config = self.make_config()
        self.assertEqual(config.hpo["search_profile"], PROFILE)
        self.assertEqual(config.hpo["profile_version"], VERSION)
        self.assertEqual(config.hpo["objective_metrics"], ["classification_accuracy", "noise_loss"])
        self.assertEqual(config.hpo["objective_directions"], ["maximize", "minimize"])
        self.assertNotIn("noise_loss", config.hpo.get("diagnostic_metrics", []))
        self.assertIsNone(config.hpo["checkpoint_selection_metric"])
        self.assertEqual(config.hpo["checkpoint_selection_policy"], "final_epoch")
        self.assertTrue(config.hpo["prune_nonfinite_losses"])
        self.assertEqual(VERSION, 2)
        self.assertEqual(config.hpo["objective_network"], "raw")
        self.assertEqual(config.training.task, "joint")
        self.assertEqual(config.model.wrapper_name, "diffusion_classifier")
        self.assertEqual(config.model.loss_function, "mse")
        self.assertEqual(config.training.monitor, "val_loss")
        self.assertEqual(config.training.monitor_mode, "min")
        self.assertEqual(config.training.patience, 0)
        self.assertEqual(config.training.fit_kwargs, {"validation_freq": 1})
        self.assertTrue(config.training.tensorboard)
        self.assertFalse(config.reporting.evaluate_ensemble_accuracy)
        self.assertEqual(config.dataset.validation_source, "test")
        self.assertEqual(config.dataset.validation_ratio, 0.0)
        self.assertFalse(config.dataset.drop_remainder)
        self.assertTrue(config.hpo["fixed_recipe"]["test_set_used_for_fit_validation"])
        self.assertFalse(config.hpo["fixed_recipe"]["independent_test_estimate"])
        self.assertEqual(config.model.wrapper_kwargs["noise_loss_coef"], 1.0)
        self.assertFalse(config.model.wrapper_kwargs["use_ema"])
        self.assertEqual(config.model.wrapper_kwargs["clf_distil_loss_coef"], 0.0)

    def test_numeric_domains_and_conditional_weight_decay(self) -> None:
        """Seal exactly the sampled ranges and omit AdamW-only dimensions for Adam."""

        for optimizer in SEARCH_SPACE["optimizer"]:
            with self.subTest(optimizer=optimizer):
                trial = _Trial()
                config = build_dit_classifier_config(
                    trial, dataset_name="cifar10", epochs=100, results_path="unused", 
                    search_space_overrides={"optimizer": [optimizer]}, seed=17
                )
                self.assertEqual(trial.domains["learning_rate"], SEARCH_SPACE["learning_rate"])
                self.assertEqual(config.optimizer.schedule, "cosine")
                # Only AdamW samples a weight-decay distribution.
                if optimizer == "adamw":
                    self.assertEqual(trial.domains["weight_decay"], SEARCH_SPACE["weight_decay"])
                # Adam keeps weight decay absent from both sampling and configuration.
                else:
                    self.assertNotIn("weight_decay", trial.params)
                    self.assertIsNone(config.optimizer.weight_decay)

    def test_readout_and_conditioning_macros_preserve_valid_models(self) -> None:
        """Keep token ownership, pooling and condition adaptation coupled explicitly."""

        for readout in SEARCH_SPACE["classifier_readout"]:
            for condition in SEARCH_SPACE["clf_cond_type"]:
                with self.subTest(readout=readout, condition=condition):
                    model = self.make_config({
                        "classifier_readout": [readout], "clf_cond_type": [condition]
                    }).model.kwargs
                    self.assertEqual(model["force_global_avg_pooling"], readout != "cls")
                    self.assertEqual(model["clf_cls_token_type"] is None, readout == "gap_without_cls")
                    self.assertEqual(model["clf_ln_no_adaptation"], condition is None)
                    self.assertIsNone(model["cls_token_type"])
                    self.assertIsNone(model["clf_distil_token_type"])

    def test_routes_and_aggregation_use_native_stage_indices(self) -> None:
        """Resolve every structural macro against sampled depth without forward references."""

        for route in SEARCH_SPACE["classifier_route"]:
            for aggregation in SEARCH_SPACE["feature_aggregation"]:
                with self.subTest(route=route, aggregation=aggregation):
                    model = self.make_config({
                        "depth": [8], "clf_depth": [6], "classifier_route": [route], 
                        "feature_aggregation": [aggregation]
                    }).model.kwargs
                    expected = {"last": [8], "middle_last": [4, 8], "all": [None]}[aggregation]
                    self.assertEqual(model["feature_aggregation_ids_dict"], {1: expected})
                    self.assertTrue(model["clf_dim_forced"])
                    self.assertEqual(model["clf_dim"], model["dim"])
                    self.assertEqual(model["clf_local_mixer_ids"], {
                        "local_first": [1], "local_alternating": [1, 3, 5]
                    }.get(route, []))
                    self.assertEqual(model["cross_attention_aggregation_ids_dict"], 
                                     {6: [4]} if route == "generator_cross_late" else {})
                    self.assertEqual(model["clf_cross_attention_ids_dict"], 
                                     {6: [3]} if route == "classifier_cross_late" else {})
                    self.assertEqual(model["clf_use_decoder_ids"], [])

    def test_classifier_corruption_never_truncates_denoising(self) -> None:
        """Use separate classifier caps while preserving full-horizon generation and clean feedback."""

        for noise, cap in {"clean": 0, "noisy32": 32, "noisy128": 128}.items():
            with self.subTest(noise=noise):
                config = self.make_config({"classifier_noise": [noise]})
                wrapper = config.model.wrapper_kwargs
                self.assertEqual(wrapper["clf_train_noisified_max_timesteps"], cap)
                self.assertEqual(wrapper["clf_test_noisified_max_timesteps"], 0)
                self.assertEqual(wrapper["clf_train_class_input_type"], "null_class_only")
                self.assertEqual(wrapper["clf_train_batch_fraction"], 0.0)
                self.assertTrue(wrapper["modify_first_t"])
                self.assertFalse(wrapper["mask_by_nulls"])
                self.assertFalse(wrapper["mask_by_t_threshold"])
                self.assertEqual(config.model.kwargs["timesteps"], 1000)
                self.assertEqual(wrapper["train_noisified_max_timesteps"], -1)
                self.assertEqual(wrapper["test_noisified_max_timesteps"], -1)

    def test_protocol_conflicts_are_rejected(self) -> None:
        """Do not let additive overrides replace loss, teacher, precision or objective identity."""

        invalid = [
            {"model_overrides": {"dim": 32}}, 
            {"wrapper_overrides": {"clf_loss_coef": 1.0}}, 
            {"wrapper_overrides": {"teacher_network": "unverified"}}, 
            {"wrapper_overrides": {"classifier_teacher_network": "unverified"}}, 
            {"wrapper_overrides": {"train_noisified_max_timesteps": 32}}, 
            {"model_overrides": {"dtype": "mixed_float16"}}, 
            {"wrapper_overrides": {"compile_args": {"loss": "mae"}}}, 
            {"dtype_policy": "mixed_float16"}, 
            {"ensemble_accuracy_kwargs": {"timesteps": [0, 1]}}, 
            {"validation_source": "split", "validation_ratio": 0.0}
        ]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.make_config(**options)
        with self.assertRaises(ValueError):
            self.make_config({"unknown_dimension": [1]})
        with self.assertRaises(ValueError):
            self.make_config({"classifier_route": ["unverified_u_shape"]})

    def test_baseline_is_fresh_detached_and_compatible(self) -> None:
        """Queue only parameters from the archived baseline, never outcomes or shared mutable objects."""

        hints = baseline_hints()
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0]["depth"], 8)
        self.assertEqual(hints[0]["clf_depth"], 8)
        self.assertEqual(hints[0]["clf_loss_coef"], 0.0043)
        self.assertNotIn("value", hints[0])
        self.assertNotIn("checkpoint", hints[0])
        categorical = {
            key: [value] for key, value in hints[0].items()
            if isinstance(SEARCH_SPACE[key], list)
        }
        config = self.make_config(categorical)
        self.assertEqual(config.model.kwargs["classifier_dropout_rate"], 0.5)
        self.assertEqual(config.model.kwargs["depth"], 8)
        hints[0]["dim"] = 1
        self.assertEqual(baseline_hints()[0]["dim"], 128)

    def test_old_joint_profile_remains_separate(self) -> None:
        """Do not change the established two-objective profile or borrow its study identity."""

        old = build_joint_classifier_config(
            _Trial(), dataset_name="cifar10", epochs=50, results_path="unused", seed=17
        )
        self.assertEqual(old.hpo["profile_version"], JOINT_CLASSIFIER_PROFILE_VERSION)
        self.assertNotEqual(old.hpo["search_profile"], PROFILE)
        self.assertEqual(old.hpo["objective_metrics"], ["classification_accuracy", "noise_loss"])
        self.assertEqual(old.training.patience, 0)
        self.assertEqual(old.model.wrapper_kwargs["clf_loss_coef"], 1.0)

    def test_real_network_route_readout_edges(self) -> None:
        """Build every route/readout with six heads and varied conditions using native APIs."""

        for route_index, route in enumerate(SEARCH_SPACE["classifier_route"]):
            for readout_index, readout in enumerate(SEARCH_SPACE["classifier_readout"]):
                with self.subTest(route=route, readout=readout):
                    tf.keras.backend.clear_session()
                    condition = SEARCH_SPACE["clf_cond_type"][(route_index + readout_index) % 3]
                    config = self.make_config({
                        "mha_num_heads": [6], "patch_size": [4], "classifier_route": [route], 
                        "classifier_readout": [readout], "clf_cond_type": [condition], 
                        "feature_aggregation": [SEARCH_SPACE["feature_aggregation"][readout_index]]
                    })
                    network = DiTClassifier(seed=17, **config.model.kwargs)
                    probabilities = network.predict_class((
                        tf.zeros((1, 32, 32, 3)), tf.zeros(tuple([1]), tf.int32), 
                        tf.zeros(tuple([1]), tf.int32)
                    ), training=False)
                    # Legacy full-return predictions place primary class probabilities first.
                    if isinstance(probabilities, (tuple, list)):
                        probabilities = probabilities[0]
                    self.assertEqual(tuple(probabilities.shape), (1, 10))
                    self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(probabilities))))
                    np.testing.assert_allclose(tf.reduce_sum(probabilities, axis=-1).numpy(), [1.0], atol=1e-5)

    def test_real_v1_clean_update_uses_all_rows_and_null_labels(self) -> None:
        """Run one native update to verify cap-zero classification and full-horizon denoising coexist."""

        tf.keras.backend.clear_session()
        config = self.make_config({"patch_size": [4], "classifier_noise": ["clean"]})
        network = DiTClassifier(seed=17, **config.model.kwargs)
        wrapper = DiffusionClassifier(network=network, seed=17, **config.model.wrapper_kwargs)
        wrapper.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")
        images = tf.random.stateless_uniform((2, 32, 32, 3), (17, 19), 0.0, 255.0)
        classes = tf.constant([2, 7])
        with patch.object(network, "predict_class", wraps=network.predict_class) as classify:
            wrapper.train_step((images, classes))
        classify.assert_called_once()
        class_images, class_times, class_labels = classify.call_args.args[0]
        np.testing.assert_array_equal(class_images.numpy(), wrapper.preprocess(images).numpy())
        np.testing.assert_array_equal(class_times.numpy(), [0, 0])
        np.testing.assert_array_equal(class_labels.numpy(), [0, 0])
        self.assertEqual(wrapper.test_noisified_max_timesteps, 1000)
        self.assertEqual(wrapper.clf_test_noisified_max_timesteps, 0)
        self.assertFalse(wrapper.use_ema)


# Execute only when this focused module is invoked directly in an online runtime.
if __name__ == "__main__":
    unittest.main()
