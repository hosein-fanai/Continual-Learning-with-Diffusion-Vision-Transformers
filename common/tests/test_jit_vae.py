"""VAE XLA regressions for stochastic state and checked sample weights."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from autoencoder import VariationalAutoencoder, VAEClassifier
from common.learner import _reset_task_random_streams


class VAEJitTests(unittest.TestCase):
    """Exercise the real compiled APIs, including persisted advancing RNG."""

    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.utils.set_random_seed(17)
        self.x = np.array([[.1, .2], [.3, .4]], dtype=np.float32)
        self.y = np.eye(2, dtype=np.float32)

    def tearDown(self):
        tf.keras.mixed_precision.set_global_policy("float32")

    def make_model(self, joint=False, **kwargs):
        options = dict(data_dim=2, latent_dim=2, hiddens_dims=(), seed=17)
        options.update(kwargs)
        if "compile_args" not in options:
            options["compile_args"] = {
                "optimizer": tf.keras.optimizers.SGD(.001),
                "jit_compile": True,
                "metrics": [tf.keras.metrics.MeanAbsoluteError(name="recon_mae")],
            }
        if joint:
            classifier = tf.keras.Sequential([
                tf.keras.layers.Input((2,)),
                tf.keras.layers.Dense(2, activation="softmax"),
            ])
            return VAEClassifier(class_num=2, classifier=classifier, **options)
        return VariationalAutoencoder(**options)

    def test_xla_draws_advance_and_task_reset_repeats(self):
        model = self.make_model(compile=False)
        draw = tf.function(model.encoder, jit_compile=True)
        _reset_task_random_streams(model, 31)
        first = draw(self.x)[2].numpy()
        second = draw(self.x)[2].numpy()
        self.assertFalse(np.array_equal(first, second))
        _reset_task_random_streams(model, 31)
        np.testing.assert_array_equal(draw(self.x)[2].numpy(), first)
        np.testing.assert_array_equal(draw(self.x)[2].numpy(), second)
        _reset_task_random_streams(model, 32)
        self.assertFalse(np.array_equal(draw(self.x)[2].numpy(), first))

    def test_weights_and_tensorflow_checkpoint_restore_next_draw(self):
        model = self.make_model(compile=False)
        model(self.x)
        state = model.encoder.get_layer("z_sample").seed_stream.state
        self.assertTrue(any(weight is state for weight in model.weights))
        with tempfile.TemporaryDirectory() as directory:
            weights = str(Path(directory) / "vae.weights.h5")
            checkpoint = str(Path(directory) / "vae_checkpoint")
            model.save_weights(weights)
            tf.train.Checkpoint(model=model).write(checkpoint)
            expected = model.encoder(self.x)[2].numpy()
            for use_checkpoint in (False, True):
                with self.subTest(checkpoint=use_checkpoint):
                    clone = self.make_model(compile=False)
                    clone(self.x)
                    if use_checkpoint:
                        tf.train.Checkpoint(model=clone).read(checkpoint).assert_consumed()
                    else:
                        clone.load_weights(weights)
                    np.testing.assert_array_equal(clone.encoder(self.x)[2].numpy(), expected)

    def test_compiled_train_and_evaluate_weighted_variants(self):
        for joint in (False, True):
            with self.subTest(joint=joint):
                model = self.make_model(joint=joint)
                before = [weight.numpy().copy() for weight in model.trainable_weights]
                # Positional sample weights exercise the unmodified Keras calling API.
                trained = model.train_on_batch(self.x, self.y, np.array([1., 0.]), return_dict=True)
                self.assertTrue(all(np.isfinite(value) for value in trained.values()))
                self.assertTrue(any(
                    not np.array_equal(old, weight.numpy())
                    for old, weight in zip(before, model.trainable_weights)
                ))
                evaluated = model.test_on_batch(self.x[:1], self.y[:1], return_dict=True)
                self.assertTrue(all(np.isfinite(value) for value in evaluated.values()))
                self.assertIn("recon_mae", evaluated)
                if joint:
                    self.assertIn("clf_accuracy", evaluated)
                model.reset_metrics()
                zero = model.test_on_batch(
                    self.x, self.y, sample_weight=np.zeros(2), return_dict=True)
                for name, value in zero.items():
                    if "loss" in name:
                        self.assertEqual(float(value), 0.)

    def test_mixed_precision_compiled_weighted_step(self):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            tf.keras.optimizers.SGD(.001), initial_scale=16.)
        model = self.make_model(compile_args={"optimizer": optimizer, "jit_compile": True})
        before = [weight.numpy().copy() for weight in model.trainable_weights]
        result = model.train_on_batch(self.x, self.y, sample_weight=np.ones(2), return_dict=True)
        self.assertTrue(all(np.isfinite(value) for value in result.values()))
        self.assertEqual(int(optimizer.inner_optimizer.iterations.numpy()), 1)
        self.assertTrue(any(
            not np.array_equal(old, weight.numpy())
            for old, weight in zip(before, model.trainable_weights)
        ))

    def test_invalid_weights_fail_before_compiled_batch_and_array_calls(self):
        model = self.make_model()
        sampler_state = model.encoder.get_layer("z_sample").seed_stream.state
        before = sampler_state.numpy().copy()
        for weights in ([-1., 1.], [np.nan, 1.], [np.inf, 1.]):
            for method in (model.train_on_batch, model.test_on_batch, model.fit, model.evaluate):
                with self.subTest(weights=weights, method=method.__name__):
                    with self.assertRaises(tf.errors.InvalidArgumentError):
                        method(self.x, self.y, sample_weight=np.array(weights))
                    np.testing.assert_array_equal(sampler_state.numpy(), before)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            model.fit(self.x, self.y, validation_data=(self.x, self.y, np.array([-1., 1.])))
        with self.assertRaises(tf.errors.InvalidArgumentError):
            model.fit(self.x, self.y, class_weight={0: -1., 1: 1.})

    def test_invalid_dataset_and_generator_weights_are_checked_outside_xla(self):
        model = self.make_model()
        for weights in ([-1., 1.], [np.inf, 1.]):
            data = tf.data.Dataset.from_tensor_slices((self.x, self.y, np.array(weights))).batch(2)
            options = tf.data.Options()
            options.threading.private_threadpool_size = 1
            data = data.with_options(options)
            for method in (model.fit, model.evaluate):
                with self.subTest(weights=weights, method=method.__name__):
                    with self.assertRaises(tf.errors.InvalidArgumentError):
                        method(data, verbose=0)

        def batches():
            yield self.x, self.y, np.array([-1., 1.])

        with self.assertRaises(tf.errors.InvalidArgumentError):
            model.evaluate(batches(), steps=1, verbose=0)


if __name__ == "__main__":
    unittest.main()
