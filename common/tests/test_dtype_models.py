"""End-to-end numeric-policy smokes for the custom model families."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from autoencoder import VariationalAutoencoder
from common.dataloader import preprocess_dataset
from common.model import _get_classifier_model
from common.replay_buffer import ReplayBuffer
from common.runtime import configure_runtime
from diffusion import (
    DiTClassifier,
    DiffusionClassifier,
    DiffusionClassifierV2,
    UNetClassifier,
)


def _make_dit_network() -> DiTClassifier:
    """Build the tiny classifier network shared by policy tests.

    Returns:
        DiTClassifier: Built two-class transformer under the active policy.
    """

    return DiTClassifier(
        num_classes=2,
        use_cfg=True,
        timesteps=4,
        image_size=4,
        channels=1,
        patch_size=2,
        dim=4,
        depth=0,
        mha_num_heads=1,
        vit_block_mlp_ratio=1.0,
        clf_depth=0,
        clf_vit_block_ids=[],
        feature_aggregation_ids_dict={1: [0]},
        build=True,
    )


class DtypeModelTests(unittest.TestCase):
    """Ensure policies reach submodels and custom training steps."""

    def tearDown(self) -> None:
        """Restore the conservative policy after each isolated smoke.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")

    def test_vae_mixed_float16_training_is_finite(self) -> None:
        """Train a VAE through a loss-scaled custom step.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        configure_runtime(7, "mixed_float16")
        model = VariationalAutoencoder(
            data_dim=4,
            latent_dim=2,
            hiddens_dims=(4,),
            conditioned=False,
            compile=False,
        )
        model.compile(optimizer="adam", loss="mse", run_eagerly=True)
        metrics = model.train_step(tf.zeros((2, 4), dtype=tf.float16))

        self.assertEqual(model.compute_dtype, "float16")
        self.assertTrue(all(
            bool(tf.reduce_all(tf.math.is_finite(tf.cast(value, tf.float32))))
            for value in metrics.values()
        ))

    def test_dit_classifier_mixed_float16_training_is_finite(self) -> None:
        """Train a tiny DiT classifier through its custom wrapper step.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        configure_runtime(11, "mixed_float16")
        network = _make_dit_network()
        model = DiffusionClassifier(
            network=network,
            use_ema=False,
            test_network_name="raw",
            test_steps=2,
            seed=11,
        )
        model.compile(optimizer="adam", loss="mse", run_eagerly=True)
        metrics = model.train_step((
            tf.zeros((2, 4, 4, 1), dtype=tf.float16),
            tf.constant([0, 1], dtype=tf.int32),
        ))

        self.assertEqual(network.compute_dtype, "float16")
        self.assertTrue(all(
            bool(tf.reduce_all(tf.math.is_finite(tf.cast(value, tf.float32))))
            for value in metrics.values()
        ))

    def test_float64_reaches_vae_dit_math_and_metrics(self) -> None:
        """Keep nested variables, stable math, schedules, and metrics float64.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        configure_runtime(13, "float64")
        vae = VariationalAutoencoder(
            data_dim=4,
            latent_dim=2,
            hiddens_dims=(4,),
            conditioned=False,
            compile=False,
        )
        vae.compile(optimizer="adam", loss="mse", run_eagerly=True)
        vae_metrics = vae.train_step(tf.zeros((2, 4), dtype=tf.float64))

        self.assertTrue(all(variable.dtype == tf.float64 for variable in vae.weights))
        self.assertTrue(all(value.dtype == tf.float64 for value in vae_metrics.values()))

        network = _make_dit_network()
        wrapper = DiffusionClassifier(
            network=network,
            use_ema=False,
            test_network_name="raw",
            test_steps=2,
            seed=13,
        )
        wrapper.compile(optimizer="adam", loss="mse", run_eagerly=True)
        wrapper_metrics = wrapper.train_step((
            tf.zeros((2, 4, 4, 1), dtype=tf.float64),
            tf.constant([0, 1], dtype=tf.int32),
        ))
        predictions = network((
            tf.zeros((2, 4, 4, 1), dtype=tf.float64),
            tf.zeros((2,), dtype=tf.int32),
            tf.constant([1, 2], dtype=tf.int32),
        ), full_return=True, training=False)
        sampled, sampled_states = wrapper.sample(
            network_name="raw",
            labels=[1],
            steps=2,
            eta=0.0,
            return_x_ts=True,
            seed=13,
        )

        self.assertTrue(all(value.dtype == tf.float64 for value in wrapper.schedules.values()))
        self.assertEqual(predictions["classes"].dtype, tf.float64)
        self.assertEqual(sampled.dtype, tf.float64)
        self.assertEqual(sampled_states[0].dtype, np.float64)
        self.assertTrue(all(value.dtype == tf.float64 for value in wrapper_metrics.values()))

    def test_unet_classifier_uses_compute_trunk_and_stable_softmax(self) -> None:
        """Separate UNet compute dtype from its stable probability output.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        expected = {
            "float32": (tf.float32, tf.float32),
            "float64": (tf.float64, tf.float64),
            "mixed_float16": (tf.float16, tf.float32),
        }
        for policy_name, (noise_dtype, class_dtype) in expected.items():
            with self.subTest(policy=policy_name):
                tf.keras.backend.clear_session()
                configure_runtime(17, policy_name)
                model = UNetClassifier(
                    num_classes=2,
                    use_cfg=True,
                    timesteps=4,
                    image_size=4,
                    channels=1,
                    widths=(2,),
                    block_depth=1,
                    bottleneck_width=3,
                    bottleneck_depth=1,
                    image_embedding_dim=2,
                    time_embedding_dim=3,
                    label_embedding_dim=2,
                    feature_aggregation_ids_dict={1: (-1,)},
                    build=True,
                )
                outputs = model((
                    tf.ones((2, 4, 4, 1), dtype=noise_dtype),
                    tf.constant([0, 1], dtype=tf.int32),
                    tf.constant([0, 2], dtype=tf.int32),
                ), full_return=True, training=False)

                self.assertEqual(outputs["noises"].dtype, noise_dtype)
                self.assertEqual(outputs["classes"].dtype, class_dtype)
                self.assertEqual(
                    model.classifier.layers[-1].dtype_policy.variable_dtype,
                    class_dtype.name,
                )

    def test_v2_mixed_optimizers_are_independently_loss_scaled(self) -> None:
        """Wrap both V2 phase optimizers for mixed custom-gradient steps.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        configure_runtime(19, "mixed_float16")
        model = DiffusionClassifierV2(
            network=_make_dit_network(),
            use_ema=False,
            test_network_name="raw",
            test_steps=2,
            seed=19,
        )
        model.compile(optimizer="adam", loss="mse", run_eagerly=True)

        loss_scale_type = tf.keras.mixed_precision.LossScaleOptimizer
        self.assertIsNot(model.gen_optimizer, model.clf_optimizer)
        self.assertIsInstance(model.gen_optimizer, loss_scale_type)
        self.assertIsInstance(model.clf_optimizer, loss_scale_type)

    def test_dense_classifier_keeps_mixed_trunk_and_stable_softmax(self) -> None:
        """Use compute policy in a DNN trunk and stable dtype at its output.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        expected = {
            "mixed_float16": ("mixed_float16", tf.float32),
            "float64": ("float64", tf.float64),
        }
        for policy_name, (hidden_policy, output_dtype) in expected.items():
            with self.subTest(policy=policy_name):
                tf.keras.backend.clear_session()
                configure_runtime(21, policy_name)
                model = _get_classifier_model(
                    class_num=2,
                    model_type="dnn",
                    architecture_kwargs={
                        "input_shape": (4,),
                        "hidden_dims": (3,),
                    },
                    verbose=0,
                )
                output = model(tf.ones((2, 4)), training=False)

                hidden = next(
                    layer for layer in model.layers
                    if isinstance(layer, tf.keras.layers.Dense)
                    and layer is not model.layers[-1]
                )
                self.assertEqual(hidden.dtype_policy.name, hidden_policy)
                self.assertEqual(model.layers[-1].dtype_policy.name, output_dtype.name)
                self.assertEqual(output.dtype, output_dtype)

    def test_stale_loaded_and_teacher_policies_are_rejected(self) -> None:
        """Fail before using float32 serialized models in a mixed experiment.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        configure_runtime(22, "float32")
        stale_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(3, input_shape=(4,), activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ])
        stale_classifier(tf.ones((1, 4), dtype=tf.float32))
        stale_teacher = _make_dit_network()

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "stale_classifier"
            stale_classifier.save(str(model_path), include_optimizer=False)
            configure_runtime(22, "mixed_float16")
            with self.assertRaisesRegex(ValueError, "does not retrofit"):
                _get_classifier_model(
                    class_num=2,
                    model_type="hp-tuned",
                    model_path=str(model_path),
                    verbose=0,
                )

        student = DiffusionClassifier(
            network=_make_dit_network(),
            use_ema=False,
            test_network_name="raw",
            test_steps=2,
            seed=22,
        )
        with self.assertRaisesRegex(ValueError, "teacher_network"):
            student.set_teacher_network(stale_teacher)

    def test_data_and_replay_use_policy_variable_dtype(self) -> None:
        """Prepare normalized/one-hot/replay arrays in float64 policy dtype.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        configure_runtime(23, "float64")
        x_train = np.arange(16, dtype=np.uint8).reshape((4, 2, 2, 1))
        y_train = np.asarray([0, 1, 0, 1], dtype=np.uint8)
        prepared = preprocess_dataset(
            x_train=x_train,
            y_train=y_train,
            x_test=x_train.copy(),
            y_test=y_train.copy(),
            class_num=2,
            indices=(0, 1),
            validation_ratio=0.0,
            preprocess="min-max",
            return_features=False,
            features_path=None,
            onehot_labels=True,
            seed=23,
            verbose=0,
        )
        self.assertEqual(prepared[0].dtype, np.float64)
        self.assertEqual(prepared[1].dtype, np.float64)

        replay = ReplayBuffer(maxlen=4, seed=23)
        replay.extend(zip(prepared[0][:2], np.asarray([0, 1], dtype=np.uint8)))
        replay_x, replay_y = replay.sample_buffer_and_prepare_dataset(2)
        self.assertEqual(replay_x.dtype, np.float64)
        self.assertEqual(replay_y.dtype, np.uint8)


# Select the test action required by this condition.
if __name__ == "__main__":
    unittest.main()
