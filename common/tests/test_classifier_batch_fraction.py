"""Verify classifier batch allocation, disjoint objectives, and saved randomness."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import Config, DiffusionClassifierConfig, load_config, save_config
from common.model import get_model
from common.tests import test_clean_classifier_training as fixtures
from diffusion import DiffusionClassifier, DiffusionClassifierV2


class ClassifierBatchFractionTests(unittest.TestCase):
    """Exercise positive fractions separately from the existing zero-fraction path."""

    setUp = fixtures.CleanClassifierTrainingTests.setUp
    tearDown = fixtures.CleanClassifierTrainingTests.tearDown
    make_network = fixtures.CleanClassifierTrainingTests.make_network
    prepared_batch = fixtures.CleanClassifierTrainingTests.prepared_batch

    def make_wrapper(self, **overrides: object) -> DiffusionClassifier:
        """Reuse the tiny real DiT with an opt-in half-batch classifier allocation."""
        options = dict(clf_train_batch_fraction=0.5, show_separate_noise_losses=True)
        options.update(overrides)
        return fixtures.CleanClassifierTrainingTests.make_wrapper(self, **options)

    def check_mixed_forward(self, input_type: str, class_input: str, graph: bool) -> None:
        """Verify one raw call and independently normalized CE/noise on disjoint rows."""
        wrapper = self.make_wrapper(clf_train_noisy_input_type=input_type,
                                    clf_train_class_input_type=class_input)
        allocation = tf.constant([True, False, False, True])
        prepared = list(self.prepared_batch(wrapper))
        prepared[1] = tf.zeros_like(self.images)
        expected_x = tf.where(allocation[:, None, None, None], self.images, prepared[3]) \
            if input_type == "clean" else prepared[3]
        expected_t = tf.where(allocation, tf.zeros_like(prepared[2]), prepared[2]) \
            if input_type == "clean" else prepared[2]
        expected_labels = tf.where(allocation, prepared[5], prepared[4]) \
            if class_input == "null_class_only" else prepared[4]
        probabilities = tf.constant([[0.8, 0.2], [0.9, 0.1], [0.4, 0.6], [0.7, 0.3]])
        noise_values = tf.reshape(tf.constant([10.0, 2.0, 3.0, 40.0]), (4, 1, 1, 1))
        network_calls = tf.Variable(0, dtype=tf.int32)
        forward_calls = tf.Variable(0, dtype=tf.int32)
        loss_calls = tf.Variable(0, dtype=tf.int32)
        original_call = wrapper.network.call
        original_forward = wrapper.forward
        original_noise_loss = wrapper.compute_noise_distil_image_kl_ctr_loss

        def network_call(inputs: tuple, **kwargs: object) -> dict:
            """Observe actual network execution rather than just the forward adapter."""
            tf.debugging.assert_equal(inputs[0], expected_x)
            tf.debugging.assert_equal(inputs[1], expected_t)
            tf.debugging.assert_equal(inputs[2], expected_labels)
            self.assertTrue(kwargs["training"])
            network_calls.assign_add(1)
            outputs = dict(original_call(inputs, **kwargs))
            outputs["classes"] = probabilities
            outputs["noises"] = tf.broadcast_to(noise_values, tf.shape(self.images))
            outputs["regs_list"] = [tf.reshape(tf.range(4, dtype=tf.float32), (4, 1))]
            outputs["z_vals_list"] = [(tf.reshape(tf.range(4, dtype=tf.float32), (4, 1)),
                                      tf.ones((4, 1)))]
            return outputs

        def forward(*args: object, **kwargs: object) -> tuple:
            """Count the single primary forward independently of the raw network call."""
            forward_calls.assign_add(1)
            return original_forward(*args, **kwargs)

        def noise_loss(
            x0: tf.Tensor, noises: tf.Tensor, classes: tf.Tensor, x0_pred: tf.Tensor,
            noises_pred: tf.Tensor, z_vals_list_c: list, regs_list_c: list, **kwargs: object,
        ) -> tuple:
            """All diffusion objectives receive only the reserved denoising complement."""
            tf.debugging.assert_equal(x0, tf.gather(self.images, [1, 2]))
            tf.debugging.assert_equal(noises, tf.zeros((2, 4, 4, 1)))
            tf.debugging.assert_equal(classes, tf.gather(self.labels, [1, 2]))
            tf.debugging.assert_equal(tf.shape(x0_pred)[0], 2)
            tf.debugging.assert_equal(noises_pred[:, 0, 0, 0], [2.0, 3.0])
            tf.debugging.assert_equal(kwargs["cond_labels"], [2, 0])
            tf.debugging.assert_equal(regs_list_c[0][:, 0], [1.0, 2.0])
            tf.debugging.assert_equal(z_vals_list_c[0][0][:, 0], [1.0, 2.0])
            loss_calls.assign_add(1)
            return original_noise_loss(x0, noises, classes, x0_pred, noises_pred,
                                       z_vals_list_c, regs_list_c, **kwargs)

        with patch.object(wrapper, "_classifier_batch_mask", return_value=allocation), \
             patch.object(wrapper, "_prepare_classifier_batch", return_value=(tuple(prepared), None, None)), \
             patch.object(wrapper.network, "call", side_effect=network_call), \
             patch.object(wrapper.network, "predict_class", side_effect=AssertionError("extra classifier pass")), \
             patch.object(wrapper, "forward", side_effect=forward), \
             patch.object(wrapper, "compute_noise_distil_image_kl_ctr_loss", side_effect=noise_loss), \
             patch.object(wrapper, "apply_grads", side_effect=lambda *args: None), \
             patch.object(wrapper, "update_ema", side_effect=lambda: None):
            step = tf.function(wrapper.train_step) if graph else wrapper.train_step
            result = step((self.images, self.labels))
        self.assertEqual(int(network_calls), 1)
        self.assertEqual(int(forward_calls), 1)
        self.assertEqual(int(loss_calls), 1)
        self.assertAlmostEqual(float(result["classifier_loss"]), float(-np.log([0.8, 0.3]).mean()), places=6)
        self.assertAlmostEqual(float(result["classifier_accuracy"]), 0.5, places=6)
        self.assertAlmostEqual(float(result[wrapper.noise_loss_tracker.name]), 6.5, places=6)
        self.assertAlmostEqual(float(result["cond_noise_loss"]), 4.0, places=6)
        self.assertAlmostEqual(float(result["uncond_noise_loss"]), 9.0, places=6)
        self.assertEqual(int(wrapper.accuracy_tracker.count), 2)
        self.assertEqual(int(wrapper.clf_loss_tracker.count), 2)
        self.assertEqual(int(wrapper.noise_loss_tracker.count), 2)
        self.assertEqual(int(wrapper.cond_noise_loss_tracker.count), 1)
        self.assertEqual(int(wrapper.uncond_noise_loss_tracker.count), 1)
        self.assertEqual(int(wrapper.total_loss_tracker.count), 4)

    def test_all_four_input_combinations_mix_only_reserved_rows_in_one_call(self) -> None:
        """The two input selectors compose within one forward in eager and graph mode."""
        for input_type in ("noisy", "clean"):
            for class_input in ("all_classes", "null_class_only"):
                for graph in (False, True):
                    with self.subTest(input_type=input_type, class_input=class_input, graph=graph):
                        self.check_mixed_forward(input_type, class_input, graph)

    def test_masks_intersect_allocation_without_returning_filtered_rows_to_denoising(self) -> None:
        """Masks use original CFG/times and cannot expand either allocated objective."""
        allocation = tf.constant([True, False, False, True])
        for null_mask, time_mask, empty in ((True, False, False), (False, True, False),
                                          (True, True, False), (True, False, True)):
            with self.subTest(null_mask=null_mask, time_mask=time_mask, empty=empty):
                wrapper = self.make_wrapper(mask_by_nulls=null_mask, mask_by_t_threshold=time_mask,
                                            mask_t_percentage=50, p_uncond=0.5)
                prepared = list(self.prepared_batch(wrapper))
                # No original null rows leaves CE empty even though selected inputs become null.
                if empty:
                    prepared[4] = self.labels + 1
                with patch.object(wrapper, "_classifier_batch_mask", return_value=allocation), \
                     patch.object(wrapper, "_prepare_classifier_batch", return_value=(tuple(prepared), None, None)), \
                     patch.object(wrapper.network, "predict_class", side_effect=AssertionError("extra classifier pass")):
                    result = tf.function(wrapper.train_step)((self.images, self.labels))
                self.assertTrue(all(np.isfinite(float(value)) for value in result.values()))
                self.assertEqual(int(wrapper.accuracy_tracker.count), 0 if empty else 1)
                self.assertEqual(int(wrapper.clf_loss_tracker.count), 0 if empty else 1)
                self.assertEqual(int(wrapper.noise_loss_tracker.count), 2)
                self.assertEqual(int(wrapper.total_loss_tracker.count), 4)
                # Empty CE must remain zero without suppressing the valid denoising objective.
                if empty:
                    self.assertEqual(float(result["classifier_loss"]), 0.0)
                    self.assertEqual(float(result["classifier_accuracy"]), 0.0)

    def test_full_fraction_and_dynamic_short_batches_are_finite(self) -> None:
        """A single-row batch and full allocation have no denoising rows or NaNs."""
        for fraction in (0.01, 0.5, 1.0):
            with self.subTest(fraction=fraction):
                wrapper = self.make_wrapper(clf_train_batch_fraction=fraction, image_loss_coef=1.0)
                step = tf.function(wrapper.train_step, input_signature=[(
                    tf.TensorSpec((None, 4, 4, 1), tf.float32), tf.TensorSpec((None,), tf.int32),
                )])
                with patch.object(wrapper.network, "predict_class", side_effect=AssertionError("extra classifier pass")):
                    for batch_size in (4, 3, 1):
                        wrapper.reset_metrics()
                        result = step((self.images[:batch_size], self.labels[:batch_size]))
                        selected = max(1, int(np.floor(batch_size * fraction)))
                        self.assertTrue(all(np.isfinite(float(value)) for value in result.values()))
                        self.assertEqual(int(wrapper.accuracy_tracker.count), selected)
                        self.assertEqual(int(wrapper.noise_loss_tracker.count), batch_size - selected)
                        self.assertEqual(int(wrapper.image_loss_tracker.count), batch_size - selected)
                        self.assertEqual(int(wrapper.total_loss_tracker.count), batch_size)
                        # No denoising rows must give exact zeros for every enabled diffusion loss.
                        if selected == batch_size:
                            self.assertEqual(float(result[wrapper.noise_loss_tracker.name]), 0.0)
                            self.assertEqual(float(result["image_loss"]), 0.0)

    def test_full_fraction_and_empty_classifier_mask_preserve_zero_losses_in_graph(self) -> None:
        """All reserved rows may be masked out without producing an invalid update."""
        wrapper = self.make_wrapper(clf_train_batch_fraction=1.0, mask_by_nulls=True,
                                    p_uncond=0.5, image_loss_coef=1.0)
        prepared = list(self.prepared_batch(wrapper))
        prepared[4] = self.labels + 1
        with patch.object(wrapper, "_prepare_classifier_batch", return_value=(tuple(prepared), None, None)), \
             patch.object(wrapper.network, "predict_class", side_effect=AssertionError("extra classifier pass")):
            result = tf.function(wrapper.train_step)((self.images, self.labels))
        self.assertTrue(all(np.isfinite(float(value)) for value in result.values()))
        self.assertEqual(float(result["classifier_loss"]), 0.0)
        self.assertEqual(float(result[wrapper.noise_loss_tracker.name]), 0.0)
        self.assertEqual(float(result["image_loss"]), 0.0)
        self.assertEqual(int(wrapper.accuracy_tracker.count), 0)
        self.assertEqual(int(wrapper.noise_loss_tracker.count), 0)
        self.assertEqual(int(wrapper.total_loss_tracker.count), 4)

    def test_allocator_rounding_randomness_reset_and_weight_restore(self) -> None:
        """A dedicated tracked stream advances reproducibly without consuming noising RNG."""
        wrapper = self.make_wrapper(clf_train_batch_fraction=0.4)
        wrapper.reset_seed(811)
        other_states = {name: stream.state.numpy().copy()
                        for name, stream in wrapper._random_streams.items() if name != "classifier_batch"}
        draw = tf.function(wrapper._classifier_batch_mask,
                           input_signature=[tf.TensorSpec((None,), tf.int32)])
        labels = tf.range(17, dtype=tf.int32)
        first = draw(labels).numpy()
        saved = wrapper.get_weights()
        second = draw(labels).numpy()
        self.assertEqual(int(first.sum()), 6)
        self.assertEqual(int(second.sum()), 6)
        self.assertFalse(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, np.arange(17) < 6))
        for name, old in other_states.items():
            np.testing.assert_array_equal(wrapper._random_streams[name].state.numpy(), old)
        wrapper.set_weights(saved)
        np.testing.assert_array_equal(draw(labels).numpy(), second)
        clone = DiffusionClassifier.from_config(json.loads(json.dumps(wrapper.get_config())))
        clone.set_weights(saved)
        np.testing.assert_array_equal(clone._classifier_batch_mask(labels).numpy(), second)
        wrapper.reset_seed(811)
        np.testing.assert_array_equal(draw(labels).numpy(), first)
        self.assertEqual(int(tf.reduce_sum(tf.cast(draw(tf.range(3)), tf.int32))), 1)
        self.assertEqual(int(tf.reduce_sum(tf.cast(draw(tf.range(1)), tf.int32))), 1)

    def test_classifier_and_denoising_losses_have_disjoint_output_gradients(self) -> None:
        """Excluded per-row logits/noises get zero gradient from the other task."""
        wrapper = self.make_wrapper()
        allocation = tf.constant([True, False, False, True])
        prepared = list(self.prepared_batch(wrapper))
        prepared[1] = tf.zeros_like(self.images)
        logits = tf.Variable([[1.0, 0.0]] * 4)
        noises = tf.Variable(tf.ones((4, 4, 4, 1)))
        original_call = wrapper.network.call
        captured = {}

        def network_call(inputs: tuple, **kwargs: object) -> dict:
            """Keep actual network computation while exposing measurable row gradients."""
            outputs = dict(original_call(inputs, **kwargs))
            outputs["classes"] = tf.nn.softmax(logits)
            outputs["noises"] = noises + 0.0
            return outputs

        def apply_gradients(tape: tf.GradientTape, loss: tf.Tensor) -> None:
            """Capture the joint objective's exact gradients without an optimizer update."""
            captured["gradients"] = tape.gradient(loss, (logits, noises))

        with patch.object(wrapper, "_classifier_batch_mask", return_value=allocation), \
             patch.object(wrapper, "_prepare_classifier_batch", return_value=(tuple(prepared), None, None)), \
             patch.object(wrapper.network, "call", side_effect=network_call), \
             patch.object(wrapper.network, "predict_class", side_effect=AssertionError("extra classifier pass")), \
             patch.object(wrapper, "apply_grads", side_effect=apply_gradients):
            wrapper.train_step((self.images, self.labels))
        classifier_gradient, noise_gradient = captured["gradients"]
        np.testing.assert_array_equal(classifier_gradient.numpy()[[1, 2]], np.zeros((2, 2)))
        np.testing.assert_array_equal(noise_gradient.numpy()[[0, 3]], np.zeros((2, 4, 4, 1)))
        self.assertTrue(np.any(classifier_gradient.numpy()[[0, 3]]))
        self.assertTrue(np.any(noise_gradient.numpy()[[1, 2]]))

    def test_real_diffusion_auxiliary_losses_use_their_partition_and_allow_empty_rows(self) -> None:
        """Actual image, KL, and token losses exclude classifier rows before reduction."""
        wrapper = self.make_wrapper(image_loss_coef=1.0, kl_loss_coef=1.0, ctr_loss_coef=1.0)
        noise_values = tf.reshape(tf.constant([10.0, 2.0, 3.0, 40.0]), (4, 1, 1, 1))
        image_values = tf.reshape(tf.constant([10.0, 3.0, 5.0, 40.0]), (4, 1, 1, 1))
        means = tf.constant([[100.0], [2.0], [4.0], [100.0]])
        probabilities = tf.constant([[0.8, 0.2], [0.1, 0.9], [0.4, 0.6], [0.7, 0.3]])
        inputs = dict(x0=tf.zeros_like(self.images), noises=tf.zeros_like(self.images),
                      classes=self.labels, x0_pred=tf.broadcast_to(image_values, tf.shape(self.images)),
                      noises_pred=tf.broadcast_to(noise_values, tf.shape(self.images)),
                      z_vals_list_c=[(means, tf.zeros_like(means))], regs_list_c=[probabilities],
                      z_vals_list_u=None, regs_list_u=None, cond_labels=tf.constant([0, 2, 0, 2]))

        def compute(mask: tf.Tensor) -> tuple:
            """Use real loss implementations with fixed auditable auxiliary outputs."""
            return wrapper._compute_batch_diffusion_losses(mask, **inputs)

        # The tiny architecture omits auxiliary heads; activate their real loss implementations.
        with patch.object(wrapper, "use_kl_loss", True), patch.object(wrapper, "use_ctr_loss", True):
            graph = tf.function(compute)
            selected = graph(tf.constant([False, True, True, False]))
            empty = graph(tf.zeros((4,), tf.bool))
        token_loss = float(-np.log([0.9, 0.4]).mean())
        self.assertAlmostEqual(float(selected[1]), 6.5, places=6)
        self.assertAlmostEqual(float(selected[5]), 17.0, places=6)
        self.assertAlmostEqual(float(selected[6]), 5.0, places=6)
        self.assertAlmostEqual(float(selected[7]), token_loss, places=6)
        self.assertAlmostEqual(float(selected[0]), 6.5 + 17.0 + 5.0 + token_loss, places=5)
        np.testing.assert_array_equal(selected[8], probabilities.numpy()[[1, 2]])
        for value in empty[:8]:
            self.assertEqual(float(value), 0.0)
        self.assertEqual(tuple(empty[8].shape), (0, 2))

    def test_classifier_subset_updates_shared_backbone_and_head(self) -> None:
        """Classifier-only gradients from the allocated rows reach the shared network."""
        wrapper = self.make_wrapper(noise_loss_coef=0.0)
        rng = np.random.default_rng(811)
        for variable in wrapper.network.trainable_variables:
            # Open zero-initialized gates to test connectivity in one training step.
            if not np.any(variable.numpy()):
                variable.assign(rng.normal(0, 0.05, variable.shape))
        groups = (wrapper.network.patch_embedder.trainable_variables,
                  wrapper.network.classifier.trainable_variables)
        before = [[variable.numpy().copy() for variable in group] for group in groups]
        with patch.object(wrapper.network, "predict_class", side_effect=AssertionError("extra classifier pass")):
            result = tf.function(wrapper.train_step)((self.images, self.labels))
        self.assertTrue(np.isfinite(float(result["classifier_loss"])))
        for variables, old_values in zip(groups, before):
            self.assertTrue(any(not np.array_equal(variable.numpy(), old)
                                for variable, old in zip(variables, old_values)))
        self.assertEqual(int(wrapper.optimizer.iterations), 1)

    def test_real_mapped_teacher_targets_and_replay_rows_remain_aligned(self) -> None:
        """Mapped KD targets retain full-batch order and train only their allocated scope."""
        teacher = self.make_network()
        network = self.make_network(classifier_only_distil_token=True, clf_distil_token_type="new_weight")
        wrapper = self.make_wrapper(network=network, teacher_network=teacher,
                                    clf_distil_loss_coef=1.0, noise_distil_loss_coef=1.0,
                                    clf_distil_type="soft", clf_distil_scope="replay_only")
        allocation = tf.constant([True, False, False, True])
        replay = tf.constant([True, True, False, False])
        prepared = self.prepared_batch(wrapper)
        wrapper._preprocess_training = True
        with patch.object(wrapper, "prep_inputs", return_value=prepared):
            mapped = wrapper.prep_inputs_map(self.images, self.labels, replay)
        self.assertEqual(len(mapped), 11)
        np.testing.assert_array_equal(mapped[-1], replay)
        self.assertEqual(tuple(mapped[-2].shape), (4, 2))
        teacher_weights = [variable.numpy().copy() for variable in teacher.weights]
        original_noise_loss = wrapper.compute_noise_distil_image_kl_ctr_loss
        original_classifier_loss = wrapper.compute_clf_kl_ctr_distil_loss
        noise_checks = tf.Variable(0, dtype=tf.int32)
        classifier_checks = tf.Variable(0, dtype=tf.int32)

        def noise_loss(*args: object, **kwargs: object) -> tuple:
            """Only denoising rows consume cached noise targets and teacher vocabulary masks."""
            tf.debugging.assert_equal(kwargs["teacher_noises_pred"], tf.gather(mapped[7], [1, 2]))
            tf.debugging.assert_equal(kwargs["teacher_noise_mask"], tf.gather(mapped[8], [1, 2]))
            noise_checks.assign_add(1)
            return original_noise_loss(*args, **kwargs)

        def classifier_loss(*args: object, **kwargs: object) -> tuple:
            """Classifier teacher and replay tensors stay in their original full-batch order."""
            tf.debugging.assert_equal(kwargs["teacher_labels"], mapped[-2])
            tf.debugging.assert_equal(kwargs["replay_mask"], replay)
            tf.debugging.assert_equal(kwargs["clf_loss_mask"], [1.0, 0.0, 0.0, 1.0])
            classifier_checks.assign_add(1)
            return original_classifier_loss(*args, **kwargs)

        with patch.object(wrapper, "_classifier_batch_mask", return_value=allocation), \
             patch.object(wrapper.network, "predict_class", side_effect=AssertionError("extra classifier pass")), \
             patch.object(wrapper, "compute_noise_distil_image_kl_ctr_loss", side_effect=noise_loss), \
             patch.object(wrapper, "compute_clf_kl_ctr_distil_loss", side_effect=classifier_loss):
            result = tf.function(wrapper.train_step)(mapped)
        self.assertTrue(all(np.isfinite(float(value)) for value in result.values()))
        self.assertEqual(int(noise_checks), 1)
        self.assertEqual(int(classifier_checks), 1)
        self.assertEqual(int(wrapper.noise_distil_loss_tracker.count), 2)
        self.assertEqual(int(wrapper.clf_distil_loss_tracker.count), 1)
        self.assertEqual(int(wrapper.clf_distil_acc_tracker.count), 1)
        self.assertEqual(int(wrapper.optimizer.iterations), 1)
        for variable, old in zip(teacher.weights, teacher_weights):
            np.testing.assert_array_equal(variable.numpy(), old)

    def test_validation_never_allocates_or_mixes_classifier_rows(self) -> None:
        """Positive training fractions retain full clean/null validation and full denoising."""
        wrapper = self.make_wrapper(clf_train_batch_fraction=1.0,
                                    clf_train_noisy_input_type="noisy",
                                    clf_train_class_input_type="all_classes")
        with patch.object(wrapper, "_classifier_batch_mask", side_effect=AssertionError("validation allocation")):
            result = tf.function(wrapper.test_step)((self.images, self.labels))
        self.assertTrue(all(np.isfinite(float(value)) for value in result.values()))
        self.assertEqual(int(wrapper.accuracy_tracker.count), 4)
        self.assertEqual(int(wrapper.noise_loss_tracker.count), 4)

    def test_zero_fraction_keeps_existing_extra_pass_and_has_no_allocation_stream(self) -> None:
        """Zero retains the full-batch classifier path and its established RNG layout."""
        wrapper = self.make_wrapper(clf_train_batch_fraction=0.0)
        self.assertNotIn("classifier_batch", wrapper._random_streams)
        original_predict = wrapper.network.predict_class
        calls = tf.Variable(0, dtype=tf.int32)

        def prediction(inputs: tuple, **kwargs: object) -> tuple:
            """Count the original clean/null extra pass on the complete batch."""
            tf.debugging.assert_equal(inputs[0], self.images)
            tf.debugging.assert_equal(inputs[1], tf.zeros((4,), tf.int32))
            tf.debugging.assert_equal(inputs[2], tf.zeros((4,), tf.int32))
            calls.assign_add(1)
            return original_predict(inputs, **kwargs)

        with patch.object(wrapper, "_classifier_batch_mask", side_effect=AssertionError("zero allocation")), \
             patch.object(wrapper.network, "predict_class", side_effect=prediction):
            result = tf.function(wrapper.train_step)((self.images, self.labels))
        self.assertTrue(np.isfinite(float(result["classifier_loss"])))
        self.assertEqual(int(calls), 1)
        self.assertEqual(int(wrapper.accuracy_tracker.count), 4)
        self.assertEqual(int(wrapper.noise_loss_tracker.count), 4)

    def test_validation_and_serialization_factory_semantic_adapter(self) -> None:
        """Validate the opt-in contract and persist fractions across public construction paths."""
        self.assertEqual(DiffusionClassifierConfig().clf_train_batch_fraction, 0.0)
        for value in (-0.1, 1.1, float("nan"), float("inf"), True, False, "0.5"):
            with self.subTest(invalid=value), self.assertRaisesRegex(AssertionError, "clf_train_batch_fraction"):
                self.make_wrapper(clf_train_batch_fraction=value)
        for options in ({"train_cfg_scale": 0.0}, {"train_cfg_scale": 2.0},
                        {"use_ensemble_loss_instead": True,
                         "clf_train_noisy_input_type": "noisy",
                         "clf_train_class_input_type": "all_classes"}):
            with self.subTest(incompatible=options), self.assertRaises(AssertionError):
                self.make_wrapper(**options)
        with self.assertRaisesRegex(ValueError, "clf_train_batch_fraction"):
            DiffusionClassifierV2(clf_train_batch_fraction=0.5)
        wrapper = self.make_wrapper(clf_train_batch_fraction=0.3)
        clone = DiffusionClassifier.from_config(json.loads(json.dumps(wrapper.get_config())))
        self.assertEqual(clone.clf_train_batch_fraction, 0.3)
        typed = DiffusionClassifierConfig(clf_train_batch_fraction=0.3)
        config = Config(model={"diffusion_classifier": typed.kwargs()})
        with tempfile.TemporaryDirectory() as directory:
            for shorten in (False, True):
                path = Path(directory) / f"fraction-{shorten}.yaml"
                save_config(config, path, shorten=shorten)
                self.assertEqual(load_config(path).model.diffusion_classifier.clf_train_batch_fraction, 0.3)
        built = get_model(model_name="dit_classifier", model_kwargs=wrapper.network.get_config(),
                          wrapper_name="diffusion_classifier",
                          wrapper_kwargs={"clf_train_batch_fraction": 0.3, "mask_by_nulls": False,
                                          "clf_train_noisy_input_type": "clean",
                                          "clf_train_class_input_type": "null_class_only",
                                          "use_ema": False, "test_steps": 4},
                          task="joint", class_num=2, image_shape=(4, 4, 1),
                          show_network_summary=False, seed=811)
        self.assertEqual(built.clf_train_batch_fraction, 0.3)
        from semantic_consolidation.model import adapt_model

        wrapper.train_step((self.images, self.labels))
        adapted = adapt_model(wrapper, controller=None)
        self.assertIs(adapted.network, wrapper.network)
        self.assertIs(adapted.optimizer, wrapper.optimizer)
        self.assertEqual(adapted.clf_train_batch_fraction, 0.3)
        self.assertEqual(adapted.get_config()["clf_train_batch_fraction"], 0.3)
        result = tf.function(adapted.train_step)((self.images, self.labels))
        self.assertTrue(np.isfinite(float(result["classifier_loss"])))
        self.assertEqual(int(adapted.optimizer.iterations), 2)


# Permit direct execution of this focused regression suite.
if __name__ == "__main__":
    unittest.main()
