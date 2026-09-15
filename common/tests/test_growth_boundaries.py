"""Check supported class reconstruction and rejection of built raw growth."""

from collections.abc import Iterator
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.model import validate_progressive_classifier_growth
from diffusion import (
    DiTClassifier, DiTDecoder, DiTEncoderDecoder, DiTEncoderDecoderClassifier,
    DiffusionModel, DiffusionTransformer, UNet, UNetClassifier,
)


def small_networks() -> Iterator[tuple[tf.keras.Model, object]]:
    """Construct each raw family with a small dynamic class vocabulary.

    Returns:
        networks (Iterator[tuple[tf.keras.Model, object]]): Built float32
            networks paired with a nonempty supported-shape depth request.

    Raises:
        ValueError: A fixture constructor rejects its explicit geometry.
    """
    transformer = dict(
        image_size=4, channels=1, patch_size=2, dim=4, depth=1,
        mha_num_heads=1, vit_block_mlp_ratio=1., num_classes=None,
        timesteps=4, seed=17,
    )
    classifier = dict(clf_depth=1, clf_mha_num_heads=1, clf_vit_block_mlp_ratio=1.)
    decoder = dict(depth=1, mha_num_heads=1, vit_block_mlp_ratio=1., shift_inputs=False)
    yield DiffusionTransformer(**transformer), "vision_transformer_block"
    yield DiTClassifier(**transformer, **classifier), {"classifier": "vision_transformer_block"}
    yield DiTDecoder(
        encoder_output_dim=4, encoder_output_grid_size=2, **transformer,
    ), "vision_transformer_block"
    yield DiTEncoderDecoder(
        decoder_kwargs=decoder, **transformer,
    ), {"decoder": "vision_transformer_block"}
    yield DiTEncoderDecoderClassifier(
        decoder_kwargs=decoder, **transformer, **classifier,
    ), {"decoder": "vision_transformer_block"}
    spatial = dict(
        image_size=4, channels=1, widths=(4,), block_depth=1,
        bottleneck_width=4, bottleneck_depth=1, image_embedding_dim=2,
        time_embedding_dim=1, label_embedding_dim=1, num_classes=None,
        timesteps=4, seed=17,
    )
    yield UNet(**spatial), "convolution_block"
    yield UNetClassifier(**spatial, clf_depth=1), {"classifier": "convolution_block"}


class GrowthBoundaryTests(unittest.TestCase):
    """Verify that rejected structural changes preserve trained model state."""

    def tearDown(self) -> None:
        """Release test models and restore the ordinary numeric policy.

        Returns:
            result (None): Keras global state is reset for the next test.

        Raises:
            None: No additional validation is performed.
        """
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")

    def test_built_raw_growth_rejects_without_state_changes(self) -> None:
        """Reject class and depth mutations across all seven raw families.

        Returns:
            result (None): Variable identities, values and constructor settings
                remain exactly equal after both rejected mutations.

        Raises:
            AssertionError: A mutation succeeds or changes any observed state.
        """
        for network, depth_spec in small_networks():
            with self.subTest(network=type(network).__name__):
                self.assertTrue(network.built)
                config = network.get_config()
                identities = [id(variable) for variable in network.variables]
                weights = network.get_weights()
                with self.assertRaisesRegex(ValueError, "Post-build class growth"):
                    network.add_class()
                with self.assertRaisesRegex(ValueError, "Post-build depth growth"):
                    network.add_depths(depth_spec)
                self.assertEqual(network.get_config(), config)
                self.assertEqual([id(variable) for variable in network.variables], identities)
                for actual, expected in zip(network.get_weights(), weights):
                    np.testing.assert_array_equal(actual, expected)

    def test_empty_growth_preserves_every_raw_family(self) -> None:
        """Keep empty requests valid without changing routes or variables.

        Returns:
            result (None): None, empty-list and disabled-list requests report
                zero integer additions; empty targeted branches behave likewise.

        Raises:
            AssertionError: A no-op changes configuration, weights or depth.
        """
        for network, _ in small_networks():
            with self.subTest(network=type(network).__name__):
                config = network.get_config()
                identities = [id(variable) for variable in network.variables]
                requests = [None, [], [None]]
                # Classifiers expose an optional independently targeted branch.
                if hasattr(network, "clf_depth"):
                    requests.append({"network": None, "classifier": []})
                # Composite models expose an optional decoder branch.
                if hasattr(network, "decoder"):
                    requests.append({"network": [None], "decoder": []})
                for request in requests:
                    result = network.add_depths(request)
                    self.assertTrue(all(branch["added"] == 0 for branch in result.values()))
                    self.assertEqual(network.get_config(), config)
                    self.assertEqual([id(variable) for variable in network.variables], identities)

    def test_decoder_metadata_copies_do_not_serialize_layer_owners(self) -> None:
        """Extend compatible encoder metadata without copying tracked owners.

        Returns:
            result (None): Float32 decoder weights remain identical while
                integer feature metadata and saved routes remain independent.

        Raises:
            AssertionError: Metadata leaks across copies or decoder weights change.
        """
        decoder = DiTDecoder(
            encoder_output_dim=4, encoder_output_grid_size=2,
            encoder_feature_dims=[4], encoder_feature_grid_sizes=[2],
            feature_aggregation_ids_dict={1: [0]}, image_size=4, channels=1,
            patch_size=2, dim=4, depth=1, mha_num_heads=1, num_classes=2,
            timesteps=4,
        )
        weights = decoder.get_weights()
        decoder.set_encoder_feature_metadata([4, 4], [2, 2], [False, False])
        config = decoder.get_config()
        self.assertEqual(config["encoder_feature_dims"], [4, 4])
        self.assertEqual(config["feature_aggregation_ids_dict"], {1: [0]})
        config["feature_aggregation_ids_dict"][1].append(1)
        self.assertEqual(decoder.feature_aggregation_ids_dict, {1: [0]})
        for actual, expected in zip(decoder.get_weights(), weights):
            np.testing.assert_array_equal(actual, expected)

    def test_rejected_wrapper_depth_keeps_optimizer_ema_and_class_growth(self) -> None:
        """Keep a trained wrapper usable after an unsupported depth request.

        Returns:
            result (None): Rejection preserves raw/EMA weights, optimizer and
                RNG variables; subsequent public fitting discovers a new class.

        Raises:
            AssertionError: Rejection changes state or later class fitting fails.
        """
        network = DiffusionTransformer(
            image_size=4, channels=1, patch_size=2, dim=4, depth=1,
            mha_num_heads=1, num_classes=None, timesteps=4, seed=17,
        )
        model = DiffusionModel(network, use_ema=True, seed=17, test_steps=2)
        model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="mse")
        images = tf.ones((2, 4, 4, 1))
        first = tf.data.Dataset.from_tensor_slices((images, [3, 3])).batch(2)
        model.fit(first, epochs=1, verbose=0)
        optimizer = model.optimizer
        weights = [value.numpy().copy() for value in model.variables + optimizer.variables]
        variables = [id(value) for value in model.variables + optimizer.variables]
        config = model.get_config()
        with self.assertRaisesRegex(ValueError, "Post-build depth growth"):
            model._add_depths("vision_transformer_block")
        self.assertIs(model.optimizer, optimizer)
        self.assertEqual(model.get_config(), config)
        self.assertEqual([id(value) for value in model.variables + optimizer.variables], variables)
        for actual, expected in zip(model.variables + optimizer.variables, weights):
            np.testing.assert_array_equal(actual.numpy(), expected)
        second = tf.data.Dataset.from_tensor_slices((images, [7, 7])).batch(2)
        model.fit(second, epochs=1, verbose=0)
        self.assertEqual(model.seen_classes, {3: 0, 7: 1})
        self.assertEqual(int(model.optimizer.iterations.numpy()), 2)

    def test_progressive_depth_rejects_before_fit_or_class_discovery(self) -> None:
        """Reject a later depth stage before an earlier stage can change state.

        Returns:
            result (None): No Keras fit runs and new labels remain undiscovered;
                raw/EMA variables, optimizer slots, RNG values, constructor
                metadata and active curriculum controls remain unchanged.

        Raises:
            AssertionError: A rejected schedule fits or mutates the wrapper.
        """
        network = DiffusionTransformer(
            image_size=4, channels=1, patch_size=2, dim=4, depth=1,
            mha_num_heads=1, num_classes=None, timesteps=4, seed=17,
        )
        model = DiffusionModel(network, use_ema=True, seed=17, test_steps=2)
        model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="mse")
        images = tf.ones((2, 4, 4, 1))
        first = tf.data.Dataset.from_tensor_slices((images, [3, 3])).batch(2)
        model.fit(first, epochs=1, verbose=0)
        second = tf.data.Dataset.from_tensor_slices((images, [7, 7])).batch(2)
        config = model.get_config()
        optimizer = model.optimizer
        variables = model.variables + optimizer.variables
        identities = [id(variable) for variable in variables]
        values = [variable.numpy().copy() for variable in variables]
        controls = (model._active_min_timestep, model._active_max_timestep, model._current_resolution)
        with patch.object(tf.keras.Model, "fit") as fit:
            with self.assertRaisesRegex(ValueError, "Post-build depth growth"):
                model.fit_progressively(
                    [("timesteps", (0, 2)), ("depth", "vision_transformer_block")],
                    x=second, stage_epochs=1, final_epochs=0, verbose=0,
                )
        fit.assert_not_called()
        self.assertIs(model.optimizer, optimizer)
        self.assertEqual(model.seen_classes, {3: 0})
        self.assertEqual(model.get_config(), config)
        self.assertEqual((model._active_min_timestep, model._active_max_timestep,
                          model._current_resolution), controls)
        self.assertEqual([id(variable) for variable in model.variables + optimizer.variables], identities)
        for actual, expected in zip(model.variables + optimizer.variables, values):
            np.testing.assert_array_equal(actual.numpy(), expected)

    def test_decoder_preflight_checks_targeted_branch_and_preserves_noops(self) -> None:
        """Apply the shared schedule guard to an independently targeted decoder.

        Returns:
            result (None): Disabled decoder requests are accepted while a
                nonempty decoder request is rejected even with an empty network
                branch; configuration and variables retain their exact values.

        Raises:
            AssertionError: A no-op is rejected or decoder growth escapes validation.
        """
        network = DiTEncoderDecoder(
            image_size=4, channels=1, patch_size=2, dim=4, depth=1,
            mha_num_heads=1, vit_block_mlp_ratio=1., num_classes=None,
            timesteps=4, seed=17, decoder_kwargs=dict(
                depth=1, mha_num_heads=1, vit_block_mlp_ratio=1., shift_inputs=False,
            ),
        )
        config = network.get_config()
        variables = [id(variable) for variable in network.variables]
        values = network.get_weights()
        for request in (None, [], [None]):
            validate_progressive_classifier_growth(network, {
                "stage_tasks": [("depth", {"decoder": request})],
            })
        with self.assertRaisesRegex(ValueError, "Post-build depth growth"):
            validate_progressive_classifier_growth(network, {
                "stage_tasks": [("depth", {
                    "network": [], "decoder": "vision_transformer_block",
                })],
            })
        self.assertEqual(network.get_config(), config)
        self.assertEqual([id(variable) for variable in network.variables], variables)
        for actual, expected in zip(network.get_weights(), values):
            np.testing.assert_array_equal(actual, expected)
