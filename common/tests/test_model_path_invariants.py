"""Check model invariants across configuration, numerical policy and evaluation paths."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from autoencoder import VAEClassifier, VariationalAutoencoder
from diffusion import (
    DiTClassifier,
    DiTDecoder,
    DiTEncoderDecoder,
    DiTEncoderDecoderClassifier,
    DiffusionModel,
    DiffusionTransformer,
)


def _transformer_options() -> dict[str, object]:
    """Return an independent tiny architecture with nontrivial attention and conditions."""

    return dict(
        num_classes=2, use_cfg=True, timesteps=4, image_size=4, channels=1,
        patch_size=2, dim=4, depth=1, mha_num_heads=1,
        vit_block_mlp_ratio=1., seed=43,
    )


class ModelPathInvariantTests(unittest.TestCase):
    """Exercise model contracts that require crossing more than one component boundary."""

    def tearDown(self) -> None:
        """Release model graphs and restore the default numerical policy."""

        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")

    def test_transformer_config_preserves_dtype_across_global_policy_change(self) -> None:
        """Restore all five transformer families without silently lowering child precision."""

        decoder_options = dict(depth=1, mha_num_heads=1, vit_block_mlp_ratio=1.)
        classifier_options = dict(clf_mha_num_heads=1, clf_vit_block_mlp_ratio=1.)
        variants = (
            (DiffusionTransformer, dict(use_refiner_cnn=True)),
            (DiTClassifier, classifier_options),
            (DiTDecoder, dict(encoder_output_grid_size=2, encoder_output_dim=4,
                              decoder_separate_cond=True, shift_inputs=False,
                              feature_aggregation_ids_dict={1: [0]})),
            (DiTEncoderDecoder, dict(decoder_kwargs=decoder_options)),
            (DiTEncoderDecoderClassifier,
             dict(**classifier_options, decoder_kwargs=decoder_options)),
        )
        for model_class, overrides in variants:
            with self.subTest(model=model_class.__name__):
                tf.keras.mixed_precision.set_global_policy("float64")
                original = model_class(**_transformer_options(), **overrides)
                config = original.get_config()
                tf.keras.mixed_precision.set_global_policy("float32")
                restored = model_class.from_config(config)
                self.assertEqual(restored.dtype_policy.name, "float64")
                self.assertTrue(restored.weights)
                self.assertTrue(all(weight.dtype == tf.float64 for weight in restored.weights))
                self.assertEqual(restored.patch_embedder.compute_dtype, "float64")
                restored.set_weights(original.get_weights())
                for actual, expected in zip(restored.weights, original.weights):
                    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
                inputs = (tf.reshape(tf.linspace(tf.constant(-1., tf.float64),
                                                 tf.constant(1., tf.float64), 32),
                                     (2, 4, 4, 1)), tf.constant([0, 3]), tf.constant([1, 2]))
                call_options = dict(full_return=True, training=False)
                # A standalone decoder needs an explicit encoder context, supplied by composites.
                if model_class is DiTDecoder:
                    call_options.update(encoder_cond=None,
                                        encoder_features_list=[tf.ones((2, 4, 4), tf.float64)])
                before = original(inputs, **call_options)
                after = restored(inputs, **call_options)
                for actual, expected in zip(tf.nest.flatten(after), tf.nest.flatten(before)):
                    # Full-return structures also contain absent conditions and auxiliary heads.
                    if tf.is_tensor(expected):
                        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-12, atol=1e-12)

    def test_explicit_transformer_policy_reaches_spatial_and_connection_layers(self) -> None:
        """Keep optional token, mixer, resampling and feature projections in float64."""

        tf.keras.mixed_precision.set_global_policy("float32")
        options = _transformer_options()
        options.update(
            dtype="float64", depth=2, cls_token_type="new_weight",
            local_mixer_ids=[1], downsample_ids=[1], upsample_ids=[2],
            connection_ids_dict={2: [1]},
            connection_kwargs={"use_layer_norm": True},
            cls_token_regularizer_ids=[1],
        )
        network = DiffusionTransformer(**options)
        self.assertTrue(network.weights)
        self.assertTrue(all(weight.dtype == tf.float64 for weight in network.weights))
        wrapper = DiffusionModel(network, dtype="float64", use_ema=True, test_steps=2,
                                 ctr_loss_coef=.01)
        wrapper.compile(optimizer="adam", loss="mse", run_eagerly=True)
        results = wrapper.train_step((
            tf.ones((2, 4, 4, 1), dtype=tf.float64), tf.constant([0, 1]),
        ))
        self.assertTrue(all(np.isfinite(value.numpy()) for value in results.values()))
        self.assertTrue(all(weight.dtype == tf.float64 for weight in wrapper.ema_network.weights))

    def test_relative_vae_weights_preserve_objective_and_analytical_gradients(self) -> None:
        """Scale row weights without changing reconstruction, KL or classifier balance."""

        x = tf.ones((2, 1))
        y = tf.one_hot([0, 0], 2)
        for model_class in (VariationalAutoencoder, VAEClassifier):
            for scale in (1., 7., 0.):
                with self.subTest(model=model_class.__name__, scale=scale):
                    options = dict(data_dim=1, latent_dim=1, hiddens_dims=(),
                                   last_activation="linear", beta=2., class_num=2,
                                   compile=False)
                    # The joint model supplies its own conditional mode and classifier.
                    if model_class is VAEClassifier:
                        options.update(alpha=3., classifier=tf.keras.Sequential([
                            tf.keras.layers.InputLayer(input_shape=(1,)),
                            tf.keras.layers.Dense(2, activation="softmax"),
                        ]))
                    # The generator-only fixture must request conditional mode explicitly.
                    else:
                        options["conditioned"] = True
                    model = model_class(**options)
                    model((x, y), training=False)
                    for weight in model.weights:
                        weight.assign(tf.zeros_like(weight))
                    mean_bias = model.encoder.get_layer("z_mean").bias
                    mean_bias.assign([1.])
                    output_bias = model.decoder.layers[-1].bias
                    model.compile(optimizer=tf.keras.optimizers.SGD(.01), loss="mse",
                                  run_eagerly=True)
                    weights = tf.constant([1., 3.]) * scale
                    evaluation = model.test_step((x, y, weights))
                    expected_loss = 2. + (3. * np.log(2.) if model_class is VAEClassifier else 0.)
                    expected_loss = expected_loss if scale else 0.
                    self.assertAlmostEqual(float(evaluation["loss"]), expected_loss, places=6)
                    model.reset_metrics()
                    model.train_step((x, y, weights))
                    np.testing.assert_allclose(output_bias.numpy(), [.02 if scale else 0.], atol=1e-7)
                    np.testing.assert_allclose(mean_bias.numpy(), [.98 if scale else 1.], atol=1e-7)
                    # Cross-entropy's zero-logit gradient is alpha * [-1/2, 1/2].
                    if model_class is VAEClassifier:
                        expected_bias = [.015, -.015] if scale else [0., 0.]
                        np.testing.assert_allclose(model.classifier.layers[-1].bias.numpy(),
                                                   expected_bias, atol=1e-7)

    def test_trained_wrapper_checkpoint_preserves_ema_sampling_under_new_global_policy(self) -> None:
        """Reload learned float64 raw and EMA weights under the ordinary float32 policy."""

        tf.keras.mixed_precision.set_global_policy("float64")
        source = DiffusionModel(DiffusionTransformer(**_transformer_options()),
                                test_steps=2, seed=51, use_ema=True, ema_decay=.5)
        source.compile(optimizer=tf.keras.optimizers.SGD(.01), loss="mse", run_eagerly=True)
        source.train_step((tf.ones((2, 4, 4, 1), tf.float64), tf.constant([0, 1])))
        initial = tf.ones((2, 4, 4, 1), tf.float64) * .125
        expected = source.sample(labels=[1, 2], x_t=initial, steps=2, eta=0.)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "weights")
            source.save_weights(path)
            tf.keras.mixed_precision.set_global_policy("float32")
            restored = DiffusionModel.from_config(source.get_config())
            restored.load_weights(path).expect_partial()
            self.assertTrue(all(weight.dtype == tf.float64 for weight in restored.network.weights))
            self.assertTrue(all(weight.dtype == tf.float64 for weight in restored.ema_network.weights))
            actual = restored.sample(labels=[1, 2], x_t=initial, steps=2, eta=0.)
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-12, atol=1e-12)


    def test_conditional_vae_reload_retains_observed_replay_classes(self) -> None:
        """Preserve the trained replay vocabulary and generated samples through SavedModel."""

        x = tf.ones((2, 2))
        y = tf.one_hot([2, 2], 3)
        for model_class in (VariationalAutoencoder, VAEClassifier):
            with self.subTest(model=model_class.__name__):
                options = dict(data_dim=2, latent_dim=1, hiddens_dims=(), class_num=3,
                               seed=13, compile_args={"optimizer": "adam"})
                # The joint model owns conditional mode and a direct feature classifier.
                if model_class is VAEClassifier:
                    options["classifier"] = tf.keras.Sequential([
                        tf.keras.layers.InputLayer(input_shape=(2,)),
                        tf.keras.layers.Dense(3, activation="softmax"),
                    ])
                # The generator-only model requests label conditioning explicitly.
                else:
                    options["conditioned"] = True
                source = model_class(**options)
                source.train(x, y, train_num=-1, epochs=1, batch_size=2,
                             callbacks_list=[], verbose=0)
                expected_x, expected_y = source.generate(samples_per_class=2, seed=17)
                with tempfile.TemporaryDirectory() as directory:
                    source.save(str(Path(directory) / "vae"), include_optimizer=False)
                    restored = tf.keras.models.load_model(str(Path(directory) / "vae"), compile=False)
                    self.assertEqual(list(restored.seen_classes), [2])
                    actual_x, actual_y = restored.generate(samples_per_class=2, seed=17)
                    np.testing.assert_array_equal(actual_y, expected_y)
                    np.testing.assert_allclose(actual_x, expected_x, rtol=1e-6, atol=1e-6)

    def test_legacy_keras_attention_weights_load_into_policy_attention(self) -> None:
        """Preserve checkpoint paths and learned predictions from ordinary Keras attention."""

        source = DiffusionTransformer(**_transformer_options(), build=False)
        block = source.layers_dicts[0][source.VTB]
        block.mha = tf.keras.layers.MultiHeadAttention.from_config(block.mha.get_config())
        source.build()
        for index, weight in enumerate(source.weights):
            weight.assign(tf.random.stateless_normal(weight.shape, seed=[61, index]) * .03)
        inputs = (tf.ones((2, 4, 4, 1)), tf.constant([0, 3]), tf.constant([1, 2]))
        expected = source(inputs, training=False)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "legacy")
            source.save_weights(path)
            restored = DiffusionTransformer.from_config(source.get_config())
            restored.build()
            restored.load_weights(path).assert_consumed()
            original_weights = {weight.name: weight.numpy() for weight in source.weights}
            restored_weights = {weight.name: weight.numpy() for weight in restored.weights}
            self.assertEqual(set(original_weights), set(restored_weights))
            for name, expected_weight in original_weights.items():
                np.testing.assert_array_equal(restored_weights[name], expected_weight)
            np.testing.assert_allclose(restored(inputs, training=False).numpy(), expected.numpy(),
                                       rtol=1e-6, atol=1e-7)


# Run the bounded regression module when invoked directly.
if __name__ == "__main__":
    unittest.main()
