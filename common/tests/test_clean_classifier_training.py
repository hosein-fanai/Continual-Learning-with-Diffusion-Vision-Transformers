"""Behavioral checks for independent V1 classifier image and condition inputs."""

import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import (
    Config, DiffusionClassifierConfig, DiffusionClassifierV2Config,
    load_config, save_config,
)
from common.model import get_model
from diffusion import DiTClassifier, DiffusionClassifier, DiffusionClassifierV2


class CleanClassifierTrainingTests(unittest.TestCase):
    """Verify selected classifier inputs, independent masks, and persistence."""

    def setUp(self) -> None:
        """Create reproducible synthetic images and preserve the caller's policy."""
        self.original_policy = tf.keras.mixed_precision.global_policy().name
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.utils.set_random_seed(811)
        self.images = tf.reshape(tf.linspace(-1.0, 1.0, 64), (4, 4, 4, 1))
        self.labels = tf.constant([0, 1, 0, 1], tf.int32)

    def tearDown(self) -> None:
        """Discard test models and restore the original precision policy."""
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy(self.original_policy)

    def make_network(self, **overrides: object) -> DiTClassifier:
        """Build a minimal image-dependent classifier with explicit overrides."""
        options = dict(
            image_size=4, channels=1, patch_size=2, dim=4, depth=1,
            mha_num_heads=1, vit_block_mlp_ratio=1.0, num_classes=2,
            timesteps=8, use_cfg=True, clf_depth=1, clf_mha_num_heads=1,
            clf_vit_block_mlp_ratio=1.0, classifier_mlp_ratio=1,
            aggregate_from_noises=False, cls_token_type=None,
            classifier_only_cls_token=True, clf_cls_token_type="new_weight",
            clf_cond_type="time_label", feature_aggregation_ids_dict={1: [-1]},
            clf_connection_ids_dict={-1: [-1]},
        )
        options.update(overrides)
        return DiTClassifier(**options)

    def make_wrapper(self, **overrides: object) -> DiffusionClassifier:
        """Compile a tiny classifier using one shared SGD optimizer."""
        network = overrides.pop("network", None)
        options = dict(
            network=network if network is not None else self.make_network(),
            use_ema=False, seed=811, scheduler_name="clipped_cosine", test_steps=4,
            clf_train_noisy_input_type="clean", clf_train_type="uncond",
            clf_train_class_input_type="null_class_only", mask_by_nulls=False,
            mask_by_t_threshold=False, train_cfg_scale=None,
            clf_loss_coef=1.0, noise_loss_coef=1.0, image_loss_coef=0.0,
            kl_loss_coef=0.0, ctr_loss_coef=0.0, clf_distil_loss_coef=0.0,
            noise_distil_loss_coef=0.0,
        )
        options.update(overrides)
        wrapper = DiffusionClassifier(**options)
        wrapper.compile(optimizer=tf.keras.optimizers.SGD(0.05), loss="mse",
                        run_eagerly=False, jit_compile=False)
        return wrapper

    def prepared_batch(self, wrapper: DiffusionClassifier) -> tuple[tf.Tensor, ...]:
        """Make corruption and CFG dropout distinguishable from clean/null inputs."""
        prepared = list(wrapper.prep_inputs((self.images, self.labels)))
        prepared[2] = tf.constant([3, 4, 5, 6], tf.int32)
        prepared[3] = self.images + 0.5
        prepared[4] = tf.constant([0, 2, 0, 2], tf.int32)
        return tuple(prepared)

    def check_selected_pass(
        self, input_type: str, class_input: str, graph: bool,
        mask_by_nulls: bool = False, mask_by_t_threshold: bool = False,
    ) -> None:
        """Assert exact inputs, one extra pass at most, and the selected CE source."""
        wrapper = self.make_wrapper(
            clf_train_noisy_input_type=input_type,
            clf_train_class_input_type=class_input, p_uncond=0.5,
            mask_by_nulls=mask_by_nulls, mask_by_t_threshold=mask_by_t_threshold,
            mask_t_percentage=50,
        )
        prepared = self.prepared_batch(wrapper)
        primary_probs = tf.constant([[0.95, 0.05], [0.05, 0.95],
                                     [0.95, 0.05], [0.05, 0.95]])
        extra_probs = tf.constant([[0.8, 0.2], [0.9, 0.1],
                                   [0.4, 0.6], [0.7, 0.3]])
        primary_calls = tf.Variable(0, dtype=tf.int32)
        extra_calls = tf.Variable(0, dtype=tf.int32)
        original_predict = wrapper.network.predict_class
        original_forward = wrapper.forward
        original_classifier_loss = wrapper.compute_clf_kl_ctr_distil_loss
        extra_required = input_type == "clean" or class_input == "null_class_only"
        selected_x = self.images if input_type == "clean" else prepared[3]
        selected_t = tf.zeros_like(prepared[2]) if input_type == "clean" else prepared[2]
        selected_labels = prepared[5] if class_input == "null_class_only" else prepared[4]

        def extra_prediction(inputs: tuple[tf.Tensor, ...], **kwargs: object) -> tuple:
            """Validate the selected image, time, and conditioning labels together."""
            tf.debugging.assert_equal(inputs[0], selected_x)
            tf.debugging.assert_equal(inputs[1], selected_t)
            tf.debugging.assert_equal(inputs[2], selected_labels)
            self.assertTrue(kwargs["training"])
            self.assertTrue(kwargs["full_return"])
            extra_calls.assign_add(1)
            outputs = list(original_predict(inputs, **kwargs))
            outputs[0] = extra_probs
            outputs[3] = [tf.constant(31.0)]
            outputs[4] = [tf.constant(41.0)]
            return tuple(outputs)

        def primary_prediction(
            network_name: str, noisy: tf.Tensor, times: tf.Tensor,
            previous_times: tf.Tensor, **kwargs: object,
        ) -> tuple:
            """Retain diffusion inputs and distinguish its classifier predictions."""
            self.assertEqual(network_name, "raw")
            tf.debugging.assert_equal(noisy, prepared[3])
            tf.debugging.assert_equal(times, prepared[2])
            tf.debugging.assert_equal(kwargs["cond_labels"], prepared[4])
            self.assertIsNone(kwargs["scale"])
            primary_calls.assign_add(1)
            outputs = list(original_forward(network_name, noisy, times, previous_times, **kwargs))
            outputs[4] = (primary_probs, tf.reverse(primary_probs, axis=[1]))
            outputs[5] = ([tf.constant(11.0)], [tf.constant(12.0)])
            outputs[6] = ([tf.constant(21.0)], [tf.constant(22.0)])
            return tuple(outputs)

        def classifier_loss(
            classes: tf.Tensor, probabilities: tf.Tensor, latents: list,
            regularizers: list, **kwargs: object,
        ) -> tuple:
            """Classifier auxiliaries must come from the same selected forward as CE."""
            tf.debugging.assert_equal(regularizers[0], 31.0 if extra_required else 11.0)
            tf.debugging.assert_equal(latents[0], 41.0 if extra_required else 21.0)
            self.assertEqual(kwargs["clf_train_type"], "cond")
            self.assertEqual(kwargs["kl_train_type"], "cond" if extra_required else None)
            self.assertEqual(kwargs["ctr_train_type"], "cond" if extra_required else None)
            return original_classifier_loss(classes, probabilities, latents, regularizers, **kwargs)

        with patch.object(wrapper, "_prepare_classifier_batch", return_value=(prepared, None, None)), \
             patch.object(wrapper.network, "predict_class", side_effect=extra_prediction), \
             patch.object(wrapper, "forward", side_effect=primary_prediction), \
             patch.object(wrapper, "compute_clf_kl_ctr_distil_loss", side_effect=classifier_loss), \
             patch.object(wrapper, "apply_grads", side_effect=lambda *args: None), \
             patch.object(wrapper, "update_ema", side_effect=lambda: None):
            step = tf.function(wrapper.train_step) if graph else wrapper.train_step
            result = step((self.images, self.labels))
        probabilities = np.array([0.8, 0.1, 0.4, 0.3] if extra_required else [0.95] * 4)
        correct = np.array([1, 0, 0, 0] if extra_required else [1] * 4)
        selected = np.ones(4, dtype=bool)
        # Null masking always uses original CFG dropout, even for null-conditioned extra passes.
        if mask_by_nulls:
            selected &= np.array([True, False, True, False])
        # Timestep masking always uses diffusion times, even when the classifier sees zero.
        if mask_by_t_threshold:
            selected &= np.array([True, False, False, False])
        self.assertAlmostEqual(float(result["classifier_loss"]),
                               float(np.mean(-np.log(probabilities[selected]))), places=6)
        self.assertAlmostEqual(float(result["classifier_accuracy"]),
                               float(np.mean(correct[selected])), places=6)
        self.assertEqual(int(wrapper.accuracy_tracker.count), int(selected.sum()))
        self.assertEqual(int(wrapper.clf_loss_tracker.count), int(selected.sum()))
        self.assertEqual(int(primary_calls), 1)
        self.assertEqual(int(extra_calls), int(extra_required))
        self.assertEqual(wrapper.mask_by_nulls, mask_by_nulls)

    def test_four_input_combinations_use_exactly_the_selected_pass_eager_and_graph(self) -> None:
        """Input enums select forward inputs and leave all four CE rows active."""
        for input_type in ("noisy", "clean"):
            for class_input in ("all_classes", "null_class_only"):
                for graph in (False, True):
                    with self.subTest(input_type=input_type, class_input=class_input, graph=graph):
                        self.check_selected_pass(input_type, class_input, graph)

    def test_row_and_timestep_masks_remain_independent_of_both_input_selectors(self) -> None:
        """Both masks select original diffusion rows for every classifier input pair."""
        for input_type in ("noisy", "clean"):
            for class_input in ("all_classes", "null_class_only"):
                for timestep_mask in (False, True):
                    with self.subTest(input_type=input_type, class_input=class_input,
                                      timestep_mask=timestep_mask):
                        self.check_selected_pass(input_type, class_input, True,
                                                 mask_by_nulls=True,
                                                 mask_by_t_threshold=timestep_mask)

    def test_explicit_null_mask_with_no_selected_rows_is_finite(self) -> None:
        """An empty row mask contributes zero CE and accuracy regardless of inputs."""
        for input_type in ("clean", "noisy"):
            for class_input in ("all_classes", "null_class_only"):
                with self.subTest(input_type=input_type, class_input=class_input):
                    wrapper = self.make_wrapper(clf_train_noisy_input_type=input_type,
                                                clf_train_class_input_type=class_input,
                                                mask_by_nulls=True, p_uncond=0.5)
                    prepared = list(self.prepared_batch(wrapper))
                    prepared[4] = self.labels + 1
                    with patch.object(wrapper, "_prepare_classifier_batch",
                                      return_value=(tuple(prepared), None, None)), \
                         patch.object(wrapper, "apply_grads", side_effect=lambda *args: None), \
                         patch.object(wrapper, "update_ema", side_effect=lambda: None):
                        result = tf.function(wrapper.train_step)((self.images, self.labels))
                    self.assertTrue(all(np.isfinite(float(value)) for value in result.values()))
                    self.assertEqual(float(result["classifier_loss"]), 0.0)
                    self.assertEqual(float(result["classifier_accuracy"]), 0.0)
                    self.assertEqual(int(wrapper.accuracy_tracker.count), 0)
                    self.assertEqual(int(wrapper.clf_loss_tracker.count), 0)

    def test_selected_classifier_gradients_update_shared_backbone_and_head(self) -> None:
        """Every selected classifier pass updates shared features without noise loss."""
        cases = [(image, labels, "float32", True)
                 for image in ("noisy", "clean")
                 for labels in ("all_classes", "null_class_only")]
        cases += [("clean", "null_class_only", "float32", False),
                  ("clean", "null_class_only", "mixed_bfloat16", True)]
        for input_type, class_input, policy, graph in cases:
            with self.subTest(input_type=input_type, class_input=class_input, policy=policy, graph=graph):
                tf.keras.mixed_precision.set_global_policy(policy)
                wrapper = self.make_wrapper(noise_loss_coef=0.0, p_uncond=0.0,
                                            clf_train_noisy_input_type=input_type,
                                            clf_train_class_input_type=class_input)
                rng = np.random.default_rng(811)
                for variable in wrapper.network.trainable_variables:
                    # Open native zero gates so one-step checks test connectivity, not warmup.
                    if not np.any(variable.numpy()):
                        variable.assign(rng.normal(0, 0.05, variable.shape))
                groups = (wrapper.network.patch_embedder.trainable_variables,
                          wrapper.network.classifier.trainable_variables)
                before = [[v.numpy().copy() for v in group] for group in groups]
                step = tf.function(wrapper.train_step) if graph else wrapper.train_step
                result = step((self.images, self.labels))
                self.assertTrue(all(np.isfinite(float(v)) for v in result.values()))
                for variables, old_values in zip(groups, before):
                    self.assertTrue(any(not np.array_equal(v.numpy(), old)
                                        for v, old in zip(variables, old_values)))
                self.assertEqual(int(wrapper.optimizer.iterations), 1)
                self.assertEqual(int(wrapper.accuracy_tracker.count), 4)

    def test_explicit_class_input_wins_legacy_branch_without_changing_masks(self) -> None:
        """Legacy cond/uncond supplies only the omitted selector's default."""
        for legacy, expected in (("cond", "all_classes"), ("uncond", "null_class_only")):
            for masked in (False, True):
                with self.subTest(legacy=legacy, masked=masked):
                    wrapper = self.make_wrapper(clf_train_noisy_input_type="noisy",
                                                clf_train_class_input_type=None,
                                                clf_train_type=legacy, train_cfg_scale=1.0,
                                                mask_by_nulls=masked, p_uncond=0.5)
                    self.assertEqual(wrapper.clf_train_class_input_type, expected)
                    self.assertEqual(wrapper.mask_by_nulls, masked)
                    opposite = "null_class_only" if expected == "all_classes" else "all_classes"
                    explicit = self.make_wrapper(clf_train_noisy_input_type="noisy",
                                                 clf_train_class_input_type=opposite,
                                                 clf_train_type=legacy, mask_by_nulls=masked,
                                                 p_uncond=0.5)
                    self.assertEqual(explicit.clf_train_class_input_type, opposite)
                    self.assertEqual(explicit.mask_by_nulls, masked)

    def test_model_factory_preserves_mask_independently_of_class_input(self) -> None:
        """Factory inference follows CFG unless the independent mask is explicit."""
        network = self.make_network()
        for class_input in ("all_classes", "null_class_only"):
            for mask in (None, False):
                with self.subTest(class_input=class_input, mask=mask):
                    kwargs = dict(clf_train_noisy_input_type="clean", clf_train_type="uncond",
                                  clf_train_class_input_type=class_input, use_ema=False,
                                  test_steps=4, p_uncond=0.5)
                    # Omission exercises the factory's ordinary CFG-based mask inference.
                    if mask is not None:
                        kwargs["mask_by_nulls"] = mask
                    wrapper = get_model(
                        model_name="dit_classifier", model_kwargs=network.get_config(),
                        wrapper_name="diffusion_classifier", wrapper_kwargs=kwargs,
                        task="joint", class_num=2, image_shape=(4, 4, 1),
                        show_network_summary=False, seed=811,
                    )
                    self.assertEqual(wrapper.clf_train_class_input_type, class_input)
                    self.assertEqual(wrapper.clf_train_noisy_input_type, "clean")
                    self.assertEqual(wrapper.mask_by_nulls, mask is None)

    def test_noisy_default_and_positional_arguments_follow_constructor_order(self) -> None:
        """Preserve the fourth input argument and only legacy uncond CFG validation."""
        params = list(inspect.signature(DiffusionClassifier.__init__).parameters)
        self.assertEqual(params[1:9], ["mask_by_nulls", "mask_by_t_threshold",
                                     "mask_t_percentage", "clf_train_noisy_input_type",
                                     "clf_train_class_input_type",
                                     "use_ensemble_loss_instead", "clf_train_type", "clf_loss_coef"])
        wrapper = DiffusionClassifier(
            True, False, 70, "noisy", None, False, "cond", 0.25,
            network=self.make_network(), test_steps=4, p_uncond=0.1, use_ema=False,
        )
        self.assertEqual(wrapper.clf_train_noisy_input_type, "noisy")
        self.assertEqual(wrapper.clf_train_class_input_type, "all_classes")
        self.assertTrue(wrapper.mask_by_nulls)
        self.assertAlmostEqual(float(wrapper.clf_loss_coef), 0.25)
        with self.assertRaisesRegex(AssertionError, "train_cfg_scale"):
            self.make_wrapper(clf_train_noisy_input_type="noisy", clf_train_type="uncond",
                              clf_train_class_input_type=None)
        explicit = self.make_wrapper(clf_train_noisy_input_type="noisy", p_uncond=0.0)
        self.assertIsNone(explicit.train_cfg_scale)
        self.assertEqual(explicit.clf_train_class_input_type, "null_class_only")

    def test_invalid_selectors_ensemble_combinations_and_v2_inputs_are_rejected(self) -> None:
        """Reject unsupported inputs while allowing independent masks and V2 null input."""
        for field in ("clf_train_noisy_input_type", "clf_train_class_input_type"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(AssertionError, field):
                    self.make_wrapper(**{field: "unknown"})
        for input_type, class_input in (("clean", "all_classes"),
                                        ("clean", "null_class_only"),
                                        ("noisy", "null_class_only")):
            with self.subTest(input_type=input_type, class_input=class_input):
                with self.assertRaisesRegex(AssertionError, "ensemble"):
                    self.make_wrapper(clf_train_noisy_input_type=input_type,
                                      clf_train_class_input_type=class_input,
                                      use_ensemble_loss_instead=True)
        primary = self.make_wrapper(clf_train_noisy_input_type="noisy",
                                    clf_train_class_input_type="all_classes",
                                    use_ensemble_loss_instead=True)
        self.assertTrue(primary.use_ensemble_loss_instead)
        with self.assertRaisesRegex(ValueError, "supported only"):
            DiffusionClassifierV2(clf_train_noisy_input_type="clean")
        with self.assertRaisesRegex(ValueError, "clf_train_class_input_type"):
            DiffusionClassifierV2(clf_train_class_input_type="all_classes")
        for class_input in (None, "null_class_only"):
            with self.subTest(v2_class_input=class_input):
                v2 = DiffusionClassifierV2(network=self.make_network(), use_ema=False,
                                          test_steps=4, clf_train_class_input_type=class_input)
                self.assertEqual(v2.clf_train_class_input_type, "null_class_only")
        # The classifier architecture itself requires CFG support.
        with self.assertRaisesRegex(AssertionError, "use_cfg"):
            self.make_network(use_cfg=False)

    def test_teacher_targets_match_selected_inputs_with_and_without_noise_distillation(self) -> None:
        """Only primary noisy/all classification can share the noisy teacher forward."""
        for input_type in ("noisy", "clean"):
            for class_input in ("all_classes", "null_class_only"):
                for noise_distil in (False, True):
                    with self.subTest(input_type=input_type, class_input=class_input,
                                      noise_distil=noise_distil):
                        wrapper = self.make_wrapper(clf_train_noisy_input_type=input_type,
                                                    clf_train_class_input_type=class_input,
                                                    train_cfg_scale=2.0, test_cfg_scale=3.0)
                        prepared = self.prepared_batch(wrapper)
                        primary = tf.constant([[0.9, 0.1]] * 4)
                        separate = tf.constant([[0.2, 0.8]] * 4)
                        expected_x = self.images if input_type == "clean" else prepared[3]
                        expected_t = tf.zeros_like(prepared[2]) if input_type == "clean" else prepared[2]
                        expected_labels = prepared[5] if class_input == "null_class_only" else prepared[4]
                        shared = noise_distil and input_type == "noisy" and class_input == "all_classes"

                        def teacher_prediction(x: tf.Tensor, t: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
                            """Verify all selected teacher inputs before returning distinct targets."""
                            np.testing.assert_array_equal(x, expected_x)
                            np.testing.assert_array_equal(t, expected_t)
                            np.testing.assert_array_equal(labels, expected_labels)
                            return separate

                        def teacher_forward(
                            network_name: str, noisy: tf.Tensor, t: tf.Tensor,
                            previous_t: tf.Tensor, **kwargs: object,
                        ) -> tuple:
                            """Teacher denoising always keeps the original corruption and labels."""
                            self.assertEqual(network_name, "teacher")
                            np.testing.assert_array_equal(noisy, prepared[3])
                            np.testing.assert_array_equal(t, prepared[2])
                            np.testing.assert_array_equal(kwargs["cond_labels"], prepared[4])
                            self.assertEqual(kwargs["scale"], 2.0)
                            self.assertFalse(kwargs["training"])
                            return (None, tf.zeros_like(self.images), None, None, (primary, separate))

                        # An unset preprocessing mode still requires training CFG for combined targets.
                        wrapper._preprocess_training = None if noise_distil else True
                        with patch.object(wrapper, "prep_inputs", return_value=prepared), \
                             patch.object(wrapper, "use_classifier_distil", True), \
                             patch.object(wrapper, "use_noise_distil_loss", noise_distil), \
                             patch.object(wrapper, "forward", side_effect=teacher_forward) as forward, \
                             patch.object(wrapper, "_predict_teacher_labels",
                                          side_effect=teacher_prediction) as predict:
                            mapped = wrapper.prep_inputs_map(self.images, self.labels)
                        self.assertEqual(forward.call_count, int(noise_distil))
                        self.assertEqual(predict.call_count, int(not shared))
                        self.assertEqual(len(mapped), 10 if noise_distil else 8)
                        np.testing.assert_array_equal(mapped[-1], primary if shared else separate)

    def test_constructor_and_yaml_round_trips_preserve_both_inputs_and_masks(self) -> None:
        """Serialize every input pair without converting conditioning into row selection."""
        self.assertEqual(DiffusionClassifierConfig().clf_train_noisy_input_type, "noisy")
        self.assertEqual(DiffusionClassifierV2Config().clf_train_noisy_input_type, "noisy")
        for input_type in ("noisy", "clean"):
            for class_input in ("null_class_only", "all_classes"):
                with self.subTest(input_type=input_type, class_input=class_input):
                    masked = class_input == "all_classes"
                    wrapper = self.make_wrapper(clf_train_noisy_input_type=input_type,
                                                clf_train_class_input_type=class_input,
                                                mask_by_nulls=masked, p_uncond=0.5)
                    serialized = json.loads(json.dumps(wrapper.get_config()))
                    clone = DiffusionClassifier.from_config(serialized)
                    self.assertEqual(clone.clf_train_noisy_input_type, input_type)
                    self.assertEqual(clone.clf_train_class_input_type, class_input)
                    self.assertEqual(clone.mask_by_nulls, masked)
                    self.assertIsNone(clone.train_cfg_scale)
                    typed = DiffusionClassifierConfig(clf_train_noisy_input_type=input_type,
                                                     clf_train_type="uncond",
                                                     clf_train_class_input_type=class_input,
                                                     mask_by_nulls=masked)
                    self.assertEqual(typed.kwargs()["clf_train_noisy_input_type"], input_type)
                    config = Config(model={"diffusion_classifier": typed.kwargs()})
                    with tempfile.TemporaryDirectory() as directory:
                        for shorten in (False, True):
                            path = Path(directory) / f"config-{shorten}.yaml"
                            save_config(config, path, shorten=shorten)
                            recovered = load_config(path).model.diffusion_classifier
                            self.assertEqual(recovered.clf_train_noisy_input_type, input_type)
                            self.assertEqual(recovered.clf_train_class_input_type, class_input)
                            self.assertEqual(recovered.mask_by_nulls, masked)

    def test_semantic_adapter_preserves_inputs_live_weights_and_optimizer(self) -> None:
        """Semantic adaptation preserves clean/null input and independent full-batch CE."""
        from semantic_consolidation.model import adapt_model

        wrapper = self.make_wrapper()
        wrapper.train_step((self.images, self.labels))
        before = wrapper.network.get_weights()
        adapted = adapt_model(wrapper, controller=None)
        self.assertIs(adapted.network, wrapper.network)
        self.assertIs(adapted.optimizer, wrapper.optimizer)
        self.assertEqual(adapted.clf_train_noisy_input_type, "clean")
        self.assertEqual(adapted.clf_train_class_input_type, "null_class_only")
        self.assertFalse(adapted.mask_by_nulls)
        self.assertEqual(adapted.get_config()["clf_train_noisy_input_type"], "clean")
        self.assertEqual(int(adapted.optimizer.iterations), 1)
        for old, current in zip(before, adapted.network.get_weights()):
            np.testing.assert_array_equal(old, current)
        result = tf.function(adapted.train_step)((self.images, self.labels))
        self.assertTrue(np.isfinite(float(result["classifier_loss"])))
        self.assertEqual(int(adapted.optimizer.iterations), 2)


# Support direct focused execution without running tests on import.
if __name__ == "__main__":
    unittest.main()
