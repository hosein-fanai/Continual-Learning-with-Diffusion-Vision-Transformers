"""Check depth growth, class reconstruction and invalid-transition boundaries."""

from collections.abc import Iterator
import tempfile
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
    """Verify valid growth and reject invalid changes without losing trained state."""

    def tearDown(self) -> None:
        """Release test models and restore the ordinary numeric policy.

        Returns:
            result (None): Keras global state is reset for the next test.

        Raises:
            None: No additional validation is performed.
        """
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")

    def test_built_raw_depth_growth_preserves_weights_and_checkpoint_state(self) -> None:
        """Grow every raw family while preserving old variables and recovery state.

        Returns:
            result (None): Direct class mutation still rejects, supported depth
                growth preserves old variable identities and values, and a
                reconstructed checkpoint restores every variable exactly.
                Keras weights also round-trip after the depth change.

        Raises:
            AssertionError: Growth loses variables, changes old values or drops
                state from checkpoint restoration.
        """
        for network, depth_spec in small_networks():
            with self.subTest(network=type(network).__name__):
                self.assertTrue(network.built)
                config = network.get_config()
                identities = [id(variable) for variable in network.variables]
                weights = network.get_weights()
                with self.assertRaisesRegex(ValueError, "Post-build class growth"):
                    network.add_class()
                self.assertEqual(network.get_config(), config)
                self.assertEqual([id(variable) for variable in network.variables], identities)
                for actual, expected in zip(network.get_weights(), weights):
                    np.testing.assert_array_equal(actual, expected)
                old_variables = list(network.variables)
                old_values = [variable.numpy().copy() for variable in old_variables]
                growth = network.add_depths(depth_spec)
                network.build()
                self.assertTrue(any(branch["added"] > 0 for branch in growth.values()))
                self.assertTrue(set(identities) <= {id(variable) for variable in network.variables})
                for variable, expected in zip(old_variables, old_values):
                    np.testing.assert_array_equal(variable.numpy(), expected)
                clone = type(network).from_config(network.get_config())
                clone.build()
                # Distinct values detect omitted checkpoint paths even with seeded clones.
                for index, variable in enumerate(network.variables):
                    value = index + 1 if tf.as_dtype(variable.dtype).is_integer else (index + 1) / 1000.
                    variable.assign(np.full(variable.shape, value, dtype=variable.dtype))
                with tempfile.TemporaryDirectory() as directory:
                    path = tf.train.Checkpoint(network=network).save(directory + "/state")
                    tf.train.Checkpoint(network=clone).restore(path).assert_consumed()
                    self.assertEqual(len(clone.variables), len(network.variables))
                    for actual, expected in zip(clone.variables, network.variables):
                        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
                    network.save_weights(directory + "/grown.weights.h5")
                    clone.set_weights([np.zeros_like(value) for value in clone.get_weights()])
                    clone.load_weights(directory + "/grown.weights.h5")
                    for actual, expected in zip(clone.get_weights(), network.get_weights()):
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

    def test_wrapper_depth_keeps_optimizer_ema_teacher_and_class_growth(self) -> None:
        """Preserve trained state while adding depth and subsequently discovering classes.

        Returns:
            result (None): Existing raw/EMA weights, RNG variables and optimizer
                state survive; new EMA variables match raw initialization and
                the independent teacher stays unchanged through later fitting.

        Raises:
            AssertionError: Growth changes old state, fails to register new
                variables, changes the teacher or prevents later class fitting.
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
        teacher = model.snapshot_teacher_network("raw")
        teacher_values = teacher.get_weights()
        old_variables = list(model.variables)
        weights = [value.numpy().copy() for value in old_variables]
        raw_ids = {id(value) for value in model.network.weights}
        ema_ids = {id(value) for value in model.ema_network.weights}
        optimizer_state = {(value.name, tuple(value.shape)): value.numpy().copy()
                           for value in model.optimizer.variables}
        model._add_depths("vision_transformer_block")
        self.assertEqual(model.network.depth, 2)
        self.assertTrue({id(value) for value in old_variables} <= {id(value) for value in model.variables})
        for actual, expected in zip(old_variables, weights):
            np.testing.assert_array_equal(actual.numpy(), expected)
        new_state = {(value.name, tuple(value.shape)): value for value in model.optimizer.variables}
        for key, expected in optimizer_state.items():
            np.testing.assert_array_equal(new_state[key].numpy(), expected)
        raw_added = [value for value in model.network.weights if id(value) not in raw_ids]
        ema_added = [value for value in model.ema_network.weights if id(value) not in ema_ids]
        self.assertGreater(len(raw_added), 0)
        self.assertEqual(len(raw_added), len(ema_added))
        for actual, expected in zip(ema_added, raw_added):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        second = tf.data.Dataset.from_tensor_slices((images, [7, 7])).batch(2)
        model.fit(second, epochs=1, verbose=0)
        self.assertEqual(model.seen_classes, {3: 0, 7: 1})
        self.assertEqual(int(model.optimizer.iterations.numpy()), 2)
        self.assertEqual(teacher.depth, 1)
        for actual, expected in zip(teacher.get_weights(), teacher_values):
            np.testing.assert_array_equal(actual, expected)

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
            with self.assertRaisesRegex(ValueError, "Unknown progressive"):
                model.fit_progressively(
                    [("timesteps", (0, 2)), ("depth", "unknown_layer")],
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
                invalid decoder request is rejected even with an empty network
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
        validate_progressive_classifier_growth(network, {
            "stage_tasks": [("depth", {"network": [], "decoder": "vision_transformer_block"})],
        })
        with self.assertRaisesRegex(ValueError, "Unknown progressive"):
            validate_progressive_classifier_growth(network, {
                "stage_tasks": [("depth", {
                    "network": [], "decoder": "unknown_layer",
                })],
            })
        self.assertEqual(network.get_config(), config)
        self.assertEqual([id(variable) for variable in network.variables], variables)
        for actual, expected in zip(network.get_weights(), values):
            np.testing.assert_array_equal(actual, expected)

    def test_invalid_final_depth_preserves_every_raw_family(self) -> None:
        """Reject a malformed final branch or stage before retaining earlier additions.

        Returns:
            result (None): All seven families retain exact configuration,
                variable identities and values after a mixed valid/invalid request.

        Raises:
            AssertionError: A rejected request changes any live model state.
        """
        for network, specification in small_networks():
            with self.subTest(network=type(network).__name__):
                # Targeted classifiers and decoders validate all branches together.
                if isinstance(specification, dict):
                    branch = next(iter(specification))
                    block = "convolution_block" if isinstance(network, UNet) else "vision_transformer_block"
                    invalid = {"network": block, branch: "unknown_layer"}
                # An invalid last stage must undo the earlier stage's metadata planning.
                else:
                    invalid = [specification, "unknown_layer"]
                config = network.get_config()
                variables = list(network.variables)
                values = [variable.numpy().copy() for variable in variables]
                with self.assertRaises(ValueError):
                    network.add_depths(invalid)
                self.assertEqual(network.get_config(), config)
                self.assertEqual([id(variable) for variable in network.variables],
                                 [id(variable) for variable in variables])
                for variable, expected in zip(variables, values):
                    np.testing.assert_array_equal(variable.numpy(), expected)
