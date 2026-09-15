"""VAE XLA regressions for stochastic state and checked sample weights."""

import tempfile
import unittest
from pathlib import Path
from collections.abc import Iterator

import numpy as np
import tensorflow as tf

from autoencoder import VariationalAutoencoder, VAEClassifier
from common.learner import _reset_task_random_streams


class VAEJitTests(unittest.TestCase):
    """Exercise the real compiled APIs, including persisted advancing RNG."""

    def setUp(self) -> None:
        """Reset float32 Keras state and create aligned float32 feature and one-hot label fixtures.

        Returns:
            result (None): The stated assertions complete, with failures reported to unittest.
        """

        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.utils.set_random_seed(17)
        self.x = np.array([[.1, .2], [.3, .4]], dtype=np.float32)
        self.y = np.eye(2, dtype=np.float32)

    def tearDown(self) -> None:
        """Restore the Keras session or numeric policy after this isolated test case.

        Returns:
            result (None): The stated assertions complete, with failures reported to unittest.
        """

        tf.keras.mixed_precision.set_global_policy("float32")

    def make_model(self, joint: bool = False, **kwargs: object) -> VariationalAutoencoder:
        """Build a two-feature VAE or a joint classifier with compiled metrics.

        Args:
            joint (bool): False builds a VAE; True attaches a two-class softmax
                classifier and includes classification metrics.
            **kwargs (object): VAE constructor overrides. Supplied compile_args
                replace the default XLA SGD and reconstruction-MAE configuration.

        Returns:
            model (VariationalAutoencoder): Seeded model in the current Keras
                numeric policy; compile=False returns an uncompiled instance.

        Raises:
            ValueError: If the requested VAE or compilation settings are invalid.
        """
        options = dict(data_dim=2, latent_dim=2, hiddens_dims=(), seed=17)
        options.update(kwargs)
        # Use default compiled SGD and metrics only when no override was supplied.
        if "compile_args" not in options:
            options["compile_args"] = {
                "optimizer": tf.keras.optimizers.SGD(.001),
                "jit_compile": True,
                "metrics": [tf.keras.metrics.MeanAbsoluteError(name="recon_mae")],
            }
        # Joint fixtures attach a real classifier and its additional objective.
        if joint:
            classifier = tf.keras.Sequential([
                tf.keras.layers.Input((2,)),
                tf.keras.layers.Dense(2, activation="softmax"),
            ])
            return VAEClassifier(class_num=2, classifier=classifier, **options)
        return VariationalAutoencoder(**options)

    def test_xla_draws_advance_and_task_reset_repeats(self) -> None:
        """Reproduce independent compiled VAE draws for repeated task seeds and distinguish a new seed.

        Returns:
            result (None): The stated assertions complete, with failures reported to unittest.

        Raises:
            AssertionError: If measured behavior violates a stated invariant.
        """

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

    def test_weights_and_tensorflow_checkpoint_restore_next_draw(self) -> None:
        """Restore the next latent draw through both HDF5 weights and a consumed TensorFlow checkpoint.

        Returns:
            result (None): The stated assertions complete, with failures reported to unittest.

        Raises:
            AssertionError: If measured behavior violates a stated invariant.
        """

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
                    # Consume the complete TensorFlow checkpoint including sampling state.
                    if use_checkpoint:
                        tf.train.Checkpoint(model=clone).read(checkpoint).assert_consumed()
                    # Compare the equivalent public Keras weight-file restoration path.
                    else:
                        clone.load_weights(weights)
                    np.testing.assert_array_equal(clone.encoder(self.x)[2].numpy(), expected)

    def test_compiled_train_and_evaluate_weighted_variants(self) -> None:
        """Exercise weighted VAE and joint-classifier XLA updates, including zero-weight losses.

        Returns:
            result (None): The stated assertions complete, with failures reported to unittest.

        Raises:
            AssertionError: If measured behavior violates a stated invariant.
        """

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
                # Only the joint classifier exposes classification accuracy.
                if joint:
                    self.assertIn("clf_accuracy", evaluated)
                model.reset_metrics()
                zero = model.test_on_batch(
                    self.x, self.y, sample_weight=np.zeros(2), return_dict=True)
                for name, value in zero.items():
                    # Zero sample weights suppress objectives while metric reductions retain their own rules.
                    if "loss" in name:
                        self.assertEqual(float(value), 0.)

    def test_mixed_precision_compiled_weighted_step(self) -> None:
        """Check finite mixed-precision XLA losses and one actual inner-optimizer update.

        Returns:
            result (None): The stated assertions complete, with failures reported to unittest.

        Raises:
            AssertionError: If measured behavior violates a stated invariant.
        """

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

    def test_invalid_weights_fail_before_compiled_batch_and_array_calls(self) -> None:
        """Reject negative or nonfinite array weights before advancing the VAE random stream.

        Returns:
            result (None): The stated assertions complete, with failures reported to unittest.

        Raises:
            AssertionError: If measured behavior violates a stated invariant.
        """

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

    def test_invalid_dataset_and_generator_weights_are_checked_outside_xla(self) -> None:
        """Reject invalid dataset and generator weights through the checked input adapters.

        Returns:
            result (None): The stated assertions complete, with failures reported to unittest.

        Raises:
            AssertionError: If measured behavior violates a stated invariant.
        """

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

        def batches() -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
            """Yield a negative-weight batch through the Python-generator adapter.

            Yields:
                batch (tuple[np.ndarray, np.ndarray, np.ndarray]): Float32
                    features and one-hot labels with float64 weights [-1, 1].

            Returns:
                result (None): Iteration stops after the single invalid batch.
            """
            yield self.x, self.y, np.array([-1., 1.])

        with self.assertRaises(tf.errors.InvalidArgumentError):
            model.evaluate(batches(), steps=1, verbose=0)


# Run compiled VAE regressions when invoked directly.
if __name__ == "__main__":
    unittest.main()
