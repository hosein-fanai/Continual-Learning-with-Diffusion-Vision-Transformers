"""Small numerical contracts for joint HPO wrappers and ensemble evaluation."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from diffusion import DiTClassifier, DiffusionClassifier, DiffusionClassifierV2
from diffusion.metrics.ensemble_accuracy import EnsembleAccuracy


class JointWrapperAuditTests(unittest.TestCase):
    """Check masks, EMA ownership, corruption parity, and dataset aggregation."""

    def setUp(self):
        self.policy = tf.keras.mixed_precision.global_policy().name
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.utils.set_random_seed(709)
        self.images = tf.reshape(tf.linspace(-1.0, 1.0, 64), (4, 4, 4, 1))
        self.labels = tf.constant([0, 1, 0, 1], tf.int32)

    def tearDown(self):
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy(self.policy)

    def _wrapper(self, version=1, aggregate=False, timesteps=8, **overrides):
        network = DiTClassifier(
            image_size=4, channels=1, patch_size=2, dim=4, depth=1,
            mha_num_heads=1, vit_block_mlp_ratio=1.0, num_classes=2,
            timesteps=timesteps, use_cfg=True, clf_depth=1, clf_mha_num_heads=1,
            clf_vit_block_mlp_ratio=1.0, aggregate_from_noises=aggregate,
            cls_token_type=None, classifier_only_cls_token=True,
            clf_cls_token_type="new_weight", clf_cond_type="time_label",
            feature_aggregation_ids_dict={1: [-1]},
            clf_connection_ids_dict={-1: [-1]},
        )
        options = dict(
            network=network, use_ema=True, test_network_name="ema", seed=709,
            scheduler_name="clipped_cosine", test_steps=4,
            test_noisified_min_timesteps=1, test_noisified_max_timesteps=timesteps,
            mask_by_nulls=True, mask_by_t_threshold=False,
            clf_train_type="cond", clf_loss_coef=1.0,
            noise_loss_coef=1.0, image_loss_coef=0.0,
            kl_loss_coef=0.0, ctr_loss_coef=0.0,
            clf_distil_loss_coef=0.0, noise_distil_loss_coef=0.0,
            ema_decay=0.5,
        )
        options.update(overrides)
        cls = DiffusionClassifier if version == 1 else DiffusionClassifierV2
        if version == 2:
            options.update(clf_vars_embedding_ids=[], clf_vars_noise_part_ids=[])
        wrapper = cls(**options)
        wrapper.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss="mse",
                        run_eagerly=True)
        return wrapper

    def test_noise_objective_matches_wrappers_ema_guidance_and_row_weights(self):
        """A 3+1 partition reports the per-image EMA MSE under identical corruption."""
        first = self._wrapper(version=1)
        second = self._wrapper(version=2)
        second.network.set_weights(first.network.get_weights())
        # Make the chosen EMA visibly different from the zero-initialized raw head.
        for variable in first.ema_network.unpatchifier.trainable_variables:
            variable.assign(tf.ones_like(variable) * 0.03)
        second.ema_network.set_weights(first.ema_network.get_weights())
        batches = [(self.images[:3], self.labels[:3]),
                   (self.images[3:], self.labels[3:])]

        first.set_timestep_bounds(1, 8)
        first.reset_seed(709)
        per_batch = []
        for batch in batches:
            _, noise, timesteps, noisy, labels, nulls, _ = first.prep_inputs(
                batch, use_label_dropout=False,
            )
            self.assertTrue(bool(tf.reduce_all(timesteps >= 1)))
            self.assertTrue(bool(tf.reduce_all(timesteps < 8)))
            np.testing.assert_array_equal(labels, batch[1] + 1)
            conditional = first.ema_network(
                (noisy, timesteps, labels), full_return=True, training=False,
            )["noises"]
            unconditional = first.ema_network(
                (noisy, timesteps, nulls), full_return=True, training=False,
            )["noises"]
            guided = unconditional + 4.0 * (conditional - unconditional)
            per_batch.append(float(tf.reduce_mean(tf.square(noise - guided))))
        expected = (3 * per_batch[0] + per_batch[1]) / 4
        self.assertGreater(abs(expected - np.mean(per_batch)), 1e-5)

        for wrapper in (first, second):
            with self.subTest(wrapper=type(wrapper).__name__):
                self.assertEqual(wrapper.test_cfg_scale, 4.0)
                wrapper.reset_seed(709)
                wrapper.set_timestep_bounds(1, 8)
                wrapper.reset_metrics()
                step = wrapper.test_step if wrapper is first else wrapper.generator_test_step
                for batch in batches:
                    result = step(batch)
                self.assertAlmostEqual(float(result["noise_loss"]), expected, places=6)
                self.assertEqual(int(wrapper.noise_loss_tracker.count), 4)
                self.assertEqual(int(wrapper._random_streams["cfg"].state[0]), 0)

    def test_modify_first_t_preserves_shared_noise_objective_timesteps(self):
        """The schedule ablation changes t=0 only, excluded by HPO noise evaluation."""
        ordinary = self._wrapper(modify_first_t=False)
        modified = self._wrapper(modify_first_t=True)
        noise = tf.ones_like(self.images) * 0.5
        times = tf.constant([1, 2, 4, 7], tf.int32)
        # Modified schedules recompute square roots in TF after float32 casting;
        # unchanged schedules retain NumPy-computed roots, differing by rounding.
        np.testing.assert_allclose(
            ordinary.q_sample(self.images, times, noise),
            modified.q_sample(self.images, times, noise),
            atol=1e-7, rtol=1e-6,
        )
        zeros = tf.zeros((4,), tf.int32)
        np.testing.assert_array_equal(modified.q_sample(self.images, zeros, noise), self.images)
        self.assertGreater(float(tf.reduce_max(tf.abs(
            ordinary.q_sample(self.images, zeros, noise) - self.images,
        ))), 0.0)

    def test_report_sampling_modes_have_finite_complete_seeded_trajectories(self):
        """Full/short stochastic/deterministic grids safely finish at either t=0 schedule."""
        for modify_first_t in (False, True):
            wrapper = self._wrapper(timesteps=1000, modify_first_t=modify_first_t,
                                    test_steps=50, test_eta=0.0)
            for steps, scale, eta in ((1000, 3.0, 1.0), (1000, 3.0, None),
                                      (None, 3.0, None), (None, 4.0, None)):
                with self.subTest(modify_first_t=modify_first_t,
                                  steps=steps, scale=scale, eta=eta):
                    calls = []

                    def zero_predictor(noisy, times, labels, nulls, cfg_scale,
                                       network_name, training):
                        self.assertEqual(network_name, "ema")
                        self.assertFalse(training)
                        self.assertEqual(cfg_scale, scale)
                        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(noisy))))
                        np.testing.assert_array_equal(labels, [0, 1, 2])
                        np.testing.assert_array_equal(nulls, [0, 0, 0])
                        calls.append(int(times[0]))
                        zero = tf.zeros_like(noisy)
                        return ((zero, zero), ([], []), ([], []))

                    wrapper.reset_seed(709)
                    with patch.object(wrapper, "call_network", side_effect=zero_predictor):
                        images, noisy_history, clean_history = wrapper.sample(
                            network_name="ema", steps=steps, scale=scale, eta=eta,
                            add_null_label=True, return_x_ts=True, return_x0s=True,
                            verbose=False,
                        )
                    expected_steps = steps or 50
                    self.assertEqual(len(calls), expected_steps)
                    self.assertEqual((calls[0], calls[-1]), (999, 0))
                    self.assertEqual(len(set(calls)), expected_steps)
                    self.assertEqual(len(noisy_history), expected_steps)
                    self.assertEqual(len(clean_history), expected_steps)
                    self.assertEqual(images.shape, (3, 4, 4, 1))
                    self.assertTrue(np.isfinite(images).all())
                    self.assertTrue(np.isfinite(noisy_history).all())
                    self.assertTrue(np.isfinite(clean_history).all())
                    expected_draws = expected_steps if eta == 1.0 else 1
                    self.assertEqual(int(wrapper._random_streams["sampling"].state[0]),
                                     256 * expected_draws)

    def test_v1_null_mask_selects_loss_gradient_and_metric_population(self):
        """Only CFG-null rows supervise the native conditional classifier branch."""
        wrapper = self._wrapper()
        classes = self.labels
        probabilities = tf.Variable([[0.8, 0.2], [0.9, 0.1],
                                     [0.4, 0.6], [0.7, 0.3]], dtype=tf.float32)
        mask = tf.constant([1.0, 0.0, 1.0, 0.0])
        with tf.GradientTape() as tape:
            loss, _ = wrapper.compute_clf_loss(classes, probabilities, mask)
        gradients = tape.gradient(loss, probabilities)
        self.assertAlmostEqual(float(loss), (-np.log(0.8) - np.log(0.4)) / 2, places=6)
        np.testing.assert_array_equal(tf.boolean_mask(gradients, mask == 0), np.zeros((2, 2)))
        self.assertGreater(float(tf.reduce_max(tf.abs(gradients))), 0.0)

        # Use the actual training path to ensure it constructs that same null selector.
        prepared = list(wrapper.prep_inputs((self.images, classes)))
        prepared[4] = tf.constant([0, 2, 0, 2], tf.int32)
        with patch.object(wrapper, "_prepare_classifier_batch", return_value=(
            tuple(prepared), None, None,
        )), patch.object(wrapper, "apply_grads"), patch.object(wrapper, "update_ema"):
            wrapper.train_step((self.images, classes))
        self.assertEqual(int(wrapper.clf_loss_tracker.count), 2)
        self.assertEqual(int(wrapper.accuracy_tracker.count), 2)

        wrapper.reset_metrics()
        empty_loss, preds = wrapper.compute_clf_loss(classes, probabilities, tf.zeros((4,)))
        result = wrapper.get_clf_results_dict(
            empty_loss, classes, preds, clf_acc_mask=tf.zeros((4,), tf.bool),
        )
        self.assertEqual(float(empty_loss), 0.0)
        self.assertTrue(all(bool(tf.math.is_finite(value)) for value in result.values()))
        self.assertEqual(int(wrapper.clf_loss_tracker.count), 0)

    def test_v2_generator_and_classifier_updates_preserve_other_raw_and_ema_group(self):
        """Feature/noise classifiers both keep shared generator inputs frozen in phase two."""
        for aggregate in (False, True):
            with self.subTest(aggregate_from_noises=aggregate):
                wrapper = self._wrapper(version=2, aggregate=aggregate,
                                        clf_train_noisified_max_timesteps=3,
                                        clf_test_noisified_max_timesteps=3)
                self.assertFalse(wrapper.mask_by_nulls)
                gen_ids = {id(v) for v in wrapper.gen_trainable_variables}
                clf_ids = {id(v) for v in wrapper.clf_trainable_variables}
                self.assertFalse(gen_ids & clf_ids)
                self.assertEqual(gen_ids | clf_ids,
                                 {id(v) for v in wrapper.network.trainable_variables})
                for layer in (wrapper.network.patch_embedder, wrapper.network.time_embedder,
                              wrapper.network.label_embedder):
                    self.assertTrue({id(v) for v in layer.trainable_variables} <= gen_ids)

                for phase, active in (("generator", gen_ids), ("discriminator", clf_ids)):
                    raw_before = [v.numpy().copy() for v in wrapper.network.weights]
                    ema_before = [v.numpy().copy() for v in wrapper.ema_network.weights]
                    getattr(wrapper, f"{phase}_train_step")((self.images, self.labels))
                    changed = False
                    for variable, raw, previous_raw, ema, previous_ema in zip(
                        wrapper.network.weights, wrapper.network.get_weights(), raw_before,
                        wrapper.ema_network.get_weights(), ema_before,
                    ):
                        if id(variable) in active:
                            changed |= not np.array_equal(raw, previous_raw)
                            np.testing.assert_allclose(ema, 0.5 * previous_ema + 0.5 * raw, atol=1e-7)
                        elif variable.trainable:
                            np.testing.assert_array_equal(raw, previous_raw)
                            np.testing.assert_array_equal(ema, previous_ema)
                    self.assertTrue(changed, f"{phase} did not update any owned weight")
                self.assertEqual(int(wrapper.gen_optimizer.iterations), 1)
                self.assertEqual(int(wrapper.clf_optimizer.iterations), 1)

                wrapper.set_timestep_bounds(5, 8)
                times, _, nulls, labels = wrapper.prep_clfv2_inputs(
                    (self.images, self.labels), noisified_max_timesteps=3,
                )
                self.assertTrue(bool(tf.reduce_all((times >= 0) & (times < 3))))
                np.testing.assert_array_equal(nulls, np.zeros((4,)))
                np.testing.assert_array_equal(labels, self.labels)
                times, clean, _, _ = wrapper.prep_clfv2_inputs(
                    (self.images, self.labels), noisified_max_timesteps=None,
                )
                np.testing.assert_array_equal(times, np.zeros((4,)))
                np.testing.assert_array_equal(clean, self.images)


class EnsembleAuditTests(unittest.TestCase):
    """Reject undefined averages and keep accuracy evaluation in inference mode."""

    def setUp(self):
        self.training_flags = []

        def predict(inputs, full_return=True, training=None, **kwargs):
            del full_return, kwargs
            self.training_flags.append(training)
            # Deliberately different branches make mode leakage observable.
            row = [0.1, 0.9] if training is not False else [0.9, 0.1]
            scores = tf.tile(tf.constant([row]), (tf.shape(inputs[0])[0], 1))
            return scores, None, [], [], []

        network = SimpleNamespace(use_cfg=True, num_classes=2, num_labels=3,
                                  dynamic_num_classes=False, predict_class=predict)
        self.wrapper = SimpleNamespace(
            timesteps=4, seed=81, get_network=lambda name: network,
            q_sample=lambda x, times, noise: x + 0.1 * noise,
        )
        self.images = tf.zeros((3, 2, 2, 1))

    def test_invalid_horizons_chunks_and_coefficients_fail_before_prediction(self):
        for name in ("max_t", "t_chunk_size"):
            for value in (0, -1, 0.5, 2.5, True):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    EnsembleAccuracy(self.wrapper, **{"max_t": 4, name: value})
        for name in ("clf_acc_coef", "clf_distil_acc_coef", "ctr_acc_coef"):
            for value in (-0.1, np.inf, np.nan):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    EnsembleAccuracy(self.wrapper, max_t=4, **{name: value})
        self.assertEqual(self.training_flags, [])

    def test_metric_evaluation_forces_inference_but_loss_prediction_keeps_training(self):
        for mode in ("chunked", "batched"):
            with self.subTest(mode=mode):
                metric = EnsembleAccuracy(self.wrapper, max_t=4, t_chunk_size=3,
                                          compute_type=mode)
                self.training_flags.clear()
                result = metric.test_step(tf.zeros((3,), tf.int32), self.images)
                self.assertEqual(float(result), 1.0)
                self.assertTrue(self.training_flags)
                self.assertTrue(all(flag is False for flag in self.training_flags))
                self.training_flags.clear()
                scores = metric.ensemble_predict(self.images, training=True)
                self.assertTrue(all(flag is True for flag in self.training_flags))
                np.testing.assert_array_equal(tf.argmax(scores, axis=-1), [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
