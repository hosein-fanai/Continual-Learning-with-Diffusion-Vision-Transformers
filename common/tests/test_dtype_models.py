"""End-to-end numeric-policy smokes for the custom model families."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from autoencoder import VAEClassifier, VariationalAutoencoder
from common.dataloader import preprocess_dataset
from common.model import _get_classifier_model, _make_optimizer, copy_model
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

    def test_package_only_import_restores_canonical_vae_class(self) -> None:
        """Load a SavedModel canonically after only importing its package.

        Returns:
            None: A fresh process resolves the lazy Keras registration proxy
            to the real ``VariationalAutoencoder`` class.
        """

        model = VariationalAutoencoder(
            data_dim=4,
            latent_dim=2,
            hiddens_dims=(3,),
            compile=False,
            seed=13,
        )
        inputs = tf.zeros((1, 4), dtype=tf.float32)
        model(inputs, training=False)
        classifier = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(4,)),
            tf.keras.layers.Dense(2, activation="softmax"),
        ])
        joint_model = VAEClassifier(
            class_num=2,
            classifier=classifier,
            data_dim=4,
            latent_dim=2,
            hiddens_dims=(3,),
            compile=False,
            seed=17,
        )
        joint_model(
            (inputs, tf.one_hot([0], depth=2)),
            training=False,
        )
        script = (
            "import sys\n"
            "import autoencoder\n"
            "assert 'autoencoder.variational_autoencoder' not in sys.modules\n"
            "assert 'autoencoder.vae_classifier' not in sys.modules\n"
            "import tensorflow as tf\n"
            "vae = tf.keras.models.load_model(sys.argv[1] + '/vae')\n"
            "joint = tf.keras.models.load_model(sys.argv[1] + '/joint')\n"
            "assert type(vae) is autoencoder.VariationalAutoencoder\n"
            "assert type(joint) is autoencoder.VAEClassifier\n"
            "assert type(joint.classifier) is tf.keras.Sequential\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            model.save(Path(directory) / "vae", include_optimizer=False)
            joint_model.save(
                Path(directory) / "joint",
                include_optimizer=False,
            )
            child_environment = dict(os.environ)
            child_environment["CUDA_VISIBLE_DEVICES"] = "-1"
            completed = subprocess.run(
                [sys.executable, "-c", script, directory],
                cwd=Path(__file__).parents[2],
                env=child_environment,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )

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

    def test_hp_tuned_preserves_trunk_and_rebuilds_optimizer_config(self) -> None:
        """Restore learned trunk weights without reusing stale optimizer slots.

        Returns:
            None: Trunk weights, new head, optimizer reset, and one update are
            asserted.
        """

        configure_runtime(22, "float32")
        source = tf.keras.Sequential([
            tf.keras.layers.Dense(
                3,
                input_shape=(4,),
                activation="relu",
                name="learned_trunk",
            ),
            tf.keras.layers.Dense(2, activation="softmax", name="old_head"),
        ])
        source.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1.25e-3),
            loss="sparse_categorical_crossentropy",
        )
        inputs = np.asarray([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
        ], dtype=np.float32)
        labels = np.asarray([0, 1], dtype=np.int32)
        source.train_on_batch(inputs, labels)
        expected_trunk = [weight.copy() for weight in source.layers[0].get_weights()]

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "compiled_classifier"
            source.save(str(model_path), include_optimizer=True)
            restored = _get_classifier_model(
                class_num=4,
                model_type="hp-tuned",
                model_path=str(model_path),
                use_loaded_opt=True,
                verbose=0,
            )

        # The saved representation is retained while only its old head is replaced.
        for actual, expected in zip(restored.layers[0].get_weights(), expected_trunk):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(restored.output_shape[-1], 4)
        self.assertIsInstance(restored.optimizer, tf.keras.optimizers.Adam)
        self.assertIsNot(restored.optimizer, source.optimizer)
        self.assertEqual(int(restored.optimizer.iterations), 0)
        restored.train_on_batch(inputs, labels)
        self.assertEqual(int(restored.optimizer.iterations), 1)

    def test_hp_tuned_preserves_functional_branching(self) -> None:
        """Replace a Functional head without linearizing its graph."""

        configure_runtime(25, "float32")
        inputs = tf.keras.layers.Input(shape=(4,))
        left = tf.keras.layers.Dense(2, name="left_branch")(inputs)
        right = tf.keras.layers.Dense(2, name="right_branch")(inputs)
        features = tf.keras.layers.Concatenate(name="merge")([left, right])
        outputs = tf.keras.layers.Dense(
            2, activation="softmax", name="old_head"
        )(features)
        source = tf.keras.Model(inputs, outputs)
        source.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        values = tf.reshape(tf.range(8, dtype=tf.float32), (2, 4))
        expected_features = tf.keras.Model(
            source.inputs, source.layers[-1].input
        )(values, training=False)

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "functional_classifier"
            source.save(str(model_path), include_optimizer=True)
            restored = _get_classifier_model(
                class_num=3,
                model_type="hp-tuned",
                model_path=str(model_path),
                verbose=0,
            )

        restored_features = tf.keras.Model(
            restored.inputs, restored.layers[-1].input
        )(values, training=False)
        np.testing.assert_allclose(restored_features, expected_features)
        self.assertEqual(restored.output_shape[-1], 3)

    def test_vae_custom_steps_accept_sample_weights(self) -> None:
        """Apply per-row weights in conditional VAE train and test steps."""

        configure_runtime(26, "float32")
        x = tf.constant([[0., 0.], [1., 1.]], dtype=tf.float32)
        y = tf.one_hot([0, 1], depth=2)
        sample_weight = tf.constant([1., 0.], dtype=tf.float32)
        compile_args = {
            "optimizer": tf.keras.optimizers.SGD(learning_rate=1e-3),
            "loss": "mse",
            "metrics": [tf.keras.metrics.MeanAbsoluteError(name="recon_mae")],
            "run_eagerly": True,
        }
        vae = VariationalAutoencoder(
            data_dim=2,
            latent_dim=1,
            hiddens_dims=(),
            conditioned=True,
            class_num=2,
            compile=False,
        )
        vae.compile(**compile_args)

        train_result = vae.train_step((x, y, sample_weight))
        vae.reset_metrics()
        test_result = vae.test_step((x, y, sample_weight))
        self.assertIn("recon_mae", train_result)
        self.assertIn("recon_mae", test_result)

        classifier = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(2,)),
            tf.keras.layers.Dense(
                2,
                activation="softmax",
                kernel_initializer="zeros",
                bias_initializer="zeros",
            ),
        ])
        joint = VAEClassifier(
            class_num=2,
            classifier=classifier,
            data_dim=2,
            latent_dim=1,
            hiddens_dims=(),
            compile_args={
                **compile_args,
                "optimizer": tf.keras.optimizers.SGD(learning_rate=1e-3),
            },
        )
        joint_test = joint.test_step((x, y, sample_weight))
        np.testing.assert_allclose(joint_test["clf_accuracy"], 1.)
        joint.reset_metrics()
        joint_train = joint.train_step((x, y, sample_weight))
        self.assertIn("clf_loss", joint_train)

    def test_hp_tuned_loaded_optimizer_requires_compiled_saved_model(self) -> None:
        """Reject an optimizer-restoration request when no optimizer was saved.

        Returns:
            None: The missing optimizer is reported before model compilation.
        """

        configure_runtime(23, "float32")
        source = tf.keras.Sequential([
            tf.keras.layers.Dense(3, input_shape=(4,), activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ])
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "uncompiled_classifier"
            source.save(str(model_path), include_optimizer=False)
            with self.assertRaisesRegex(ValueError, "compiled optimizer"):
                _get_classifier_model(
                    class_num=3,
                    model_type="hp-tuned",
                    model_path=str(model_path),
                    use_loaded_opt=True,
                    verbose=0,
                )

    def test_optimizer_clipnorm_is_forwarded(self) -> None:
        """Forward Keras clipnorm without duplicating its validation.

        Returns:
            None: The optimizer's clipping mode is asserted.
        """

        optimizer = _make_optimizer(
            name="sgd",
            schedule="constant",
            clipnorm=2.5,
        )
        self.assertEqual(float(optimizer.clipnorm), 2.5)
        self.assertIsNone(optimizer.global_clipnorm)

    def test_copy_model_leaves_destination_optimizer_state_untouched(self) -> None:
        """Copy classifier parameters without pretending optimizer slots match.

        Returns:
            None: Weight prefixes and destination iteration state are asserted.
        """

        configure_runtime(24, "float32")
        previous = tf.keras.Sequential([
            tf.keras.layers.Dense(3, input_shape=(2,), activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ])
        expanded = tf.keras.Sequential([
            tf.keras.layers.Dense(3, input_shape=(2,), activation="relu"),
            tf.keras.layers.Dense(4, activation="softmax"),
        ])
        previous.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        expanded.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        inputs = np.asarray([[1., 0.], [0., 1.]], dtype=np.float32)
        previous.train_on_batch(inputs, np.asarray([0, 1], dtype=np.int32))
        expanded.train_on_batch(inputs, np.asarray([2, 3], dtype=np.int32))
        destination_iteration = int(expanded.optimizer.iterations)
        new_head_before = [weight.copy() for weight in expanded.layers[-1].get_weights()]

        copy_model(previous, expanded)

        self.assertEqual(int(expanded.optimizer.iterations), destination_iteration)
        # Existing class columns are copied and new class columns keep initialization.
        np.testing.assert_array_equal(
            expanded.layers[-1].get_weights()[0][:, :2],
            previous.layers[-1].get_weights()[0],
        )
        np.testing.assert_array_equal(
            expanded.layers[-1].get_weights()[0][:, 2:],
            new_head_before[0][:, 2:],
        )
        with self.assertRaisesRegex(ValueError, "cover every destination"):
            copy_model(previous, expanded, allow_truncate=True)

    def test_stale_loaded_and_teacher_policies_are_left_to_tensorflow(self) -> None:
        """Let TensorFlow cast restored models across runtime policies.

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
            loaded_classifier = _get_classifier_model(
                class_num=2,
                model_type="hp-tuned",
                model_path=str(model_path),
                verbose=0,
            )
            self.assertEqual(loaded_classifier.output_shape[-1], 2)

        student = DiffusionClassifier(
            network=_make_dit_network(),
            use_ema=False,
            test_network_name="raw",
            test_steps=2,
            seed=22,
        )
        student.set_teacher_network(stale_teacher)
        self.assertIs(student.teacher_network, stale_teacher)

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
