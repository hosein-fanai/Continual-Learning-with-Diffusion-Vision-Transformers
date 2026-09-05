"""Regression checks for transformer feature widths and spatial routing metadata.

Tiny forward passes cover channel concatenation, addition, secondary-only connectors,
encoder aggregation, and flattened features. Model constructors reject merge axes other
than -1. Run through ``python -m unittest`` discovery or directly.
"""

from __future__ import annotations

import unittest

import tensorflow as tf

from diffusion import DiTClassifier, DiTDecoder, DiffusionTransformer


class TransformerMetadataTests(unittest.TestCase):
    """Compare inferred transformer geometry with executed feature tensors.

    Attributes:
        config (dict[str, object]): Shared tiny transformer constructor options.
        inputs (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Images, times, and labels.
        decoder_config (dict[str, object]): Matching encoder metadata.
        encoder_cond (tf.Tensor): Encoder condition vectors.
        encoder_features (list[tf.Tensor]): Two square encoder feature grids.
    """

    def setUp(self) -> None:
        """Create deterministic fixtures under the standard numeric policy.

        Returns:
            None: Small model options and input tensors are stored on this case.
        """

        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.random.set_seed(109)
        self.config = {
            "num_classes": 2,
            "use_cfg": True,
            "timesteps": 4,
            "image_size": 4,
            "channels": 1,
            "patch_size": 2,
            "dim": 4,
            "mha_num_heads": 1,
            "vit_block_mlp_ratio": 1.0,
            "build": False,
        }
        self.inputs = (
            tf.reshape(tf.linspace(-1.0, 1.0, 32), (2, 4, 4, 1)),
            tf.constant([0, 3], dtype=tf.int32),
            tf.constant([1, 2], dtype=tf.uint8),
        )
        self.decoder_config = {
            "encoder_output_grid_size": 2,
            "encoder_output_dim": 4,
            "encoder_feature_grid_sizes": [2, 2],
            "encoder_feature_dims": [4, 4],
            "shift_inputs": False,
        }
        self.encoder_cond = tf.ones((2, 4))
        self.encoder_features = [
            tf.reshape(tf.linspace(-1.0, 1.0, 32), (2, 4, 4)),
            tf.ones((2, 4, 4)),
        ]

    def tearDown(self) -> None:
        """Release Keras state after each independent regression.

        Returns:
            None: The Keras session is cleared.
        """

        tf.keras.backend.clear_session()

    def test_connection_axes_preserve_actual_width_and_grid(self) -> None:
        """Resolve channel concatenation and addition at the supported axis.

        Returns:
            None: Connector and stage metadata match the executed tensors.
        """

        for connect_type, width in (("concat", 8), ("add", 4)):
            with self.subTest(connect_type=connect_type):
                model = DiffusionTransformer(
                    depth=1,
                    vit_block_ids=[],
                    connection_ids_dict={1: [0, 0]},
                    connection_kwargs={
                        "connect_axis": -1,
                        "connect_type": connect_type,
                        "use_layer_norm": True,
                    },
                    dim_forced=False,
                    use_unpatchify=False,
                    **self.config,
                )
                output = model(self.inputs, training=False)
                handler = model.layers_dicts[0][model.FC]
                self.assertEqual(output.shape, (2, 4, width))
                self.assertEqual(handler.ln_dim, width)
                self.assertEqual(handler.output_dim, width)
                self.assertEqual(handler.grid_size, 2)
                self.assertEqual(model._get_last_output_dim(0, model.layers_dicts, 4), width)
                self.assertEqual(model._get_last_grid_size(0, model.layers_dicts, 2), 2)

    def test_encoder_aggregation_axes_preserve_actual_geometry(self) -> None:
        """Track geometry when encoder features join the decoder stream.

        Returns:
            None: Encoder aggregation metadata agrees with raw decoder output.
        """

        for connect_type, width in (("concat", 8), ("add", 4)):
            with self.subTest(connect_type=connect_type):
                model = DiTDecoder(
                    depth=1,
                    vit_block_ids=[],
                    use_decoder_ids=[],
                    feature_aggregation_ids_dict={1: [0]},
                    feature_aggregation_kwargs={
                        "connect_axis": -1,
                        "connect_type": connect_type,
                        "use_layer_norm": True,
                    },
                    dim_forced=False,
                    use_unpatchify=False,
                    **self.decoder_config,
                    **self.config,
                )
                output = model(
                    self.inputs, self.encoder_cond, self.encoder_features,
                    training=False,
                )["noises"]
                handler = model.layers_dicts[0][model.FA]
                self.assertEqual(output.shape, (2, 4, width))
                self.assertEqual(handler.output_dim, width)
                self.assertEqual(handler.grid_size, 2)
                self.assertEqual(model._get_last_output_dim(0, model.layers_dicts, 4), width)
                self.assertEqual(model._get_last_grid_size(0, model.layers_dicts, 2), 2)

        for width, grid in ((1, 2), (4, 1)):
            with self.subTest(width=width, grid=grid), self.assertRaisesRegex(
                AssertionError, "equal feature dimensions and grid sizes"
            ):
                model._create_encoder_feature_handler(
                    [0], increased_dim=width, second_grid_size=grid,
                    kwargs={"connect_type": "add"},
                )

    def test_secondary_only_connectors_keep_aggregate_geometry(self) -> None:
        """Preserve a sole encoder aggregate with channel concatenation or addition.

        Returns:
            None: Empty primary selections retain secondary width and grid.
        """

        for options in (
            {"connect_axis": -1, "connect_type": "add"},
            {"connect_axis": -1, "connect_type": "concat"},
        ):
            with self.subTest(options=options):
                model = DiTDecoder(
                    depth=1,
                    vit_block_ids=[],
                    use_decoder_ids=[],
                    feature_aggregation_ids_dict={1: [0]},
                    connection_ids_dict={1: []},
                    connection_kwargs={**options, "use_layer_norm": True},
                    use_unpatchify=False,
                    **self.decoder_config,
                    **self.config,
                )
                output = model(
                    self.inputs, self.encoder_cond, self.encoder_features,
                    training=False,
                )["noises"]
                self.assertEqual(output.shape, (2, 4, 4))
                self.assertEqual(model.layers_dicts[0][model.FC].ln_dim, 4)
                self.assertEqual(model._get_last_output_dim(0, model.layers_dicts, 4), 4)
                self.assertEqual(model._get_last_grid_size(0, model.layers_dicts, 2), 2)

    def test_decoder_aggregation_grid_precedes_previous_grid(self) -> None:
        """Use the selected encoder grid after an empty decoder feature selector.

        Returns:
            None: A later upsampler receives the aggregate grid and reconstructs an image.
        """

        model = DiTDecoder(
            depth=2,
            vit_block_ids=[],
            use_decoder_ids=[],
            feature_aggregation_ids_dict={1: [0]},
            feature_aggregation_kwargs={"connect_axis": -1},
            connection_ids_dict={1: []},
            connection_kwargs={"connect_axis": -1},
            upsample_ids=[2],
            upsample_kwargs={"scaling_method": "interpolate"},
            **{
                **self.decoder_config,
                "encoder_output_grid_size": 1,
                "encoder_feature_grid_sizes": [1, 1],
            },
            **self.config,
        )
        features = [feature[:, :1] for feature in self.encoder_features]
        output = model(
            self.inputs, self.encoder_cond, features, full_return=True, training=False,
        )
        self.assertEqual(output["features_list"][1].shape, (2, 1, 4))
        self.assertEqual(model._get_last_grid_size(0, model.layers_dicts, 2), 1)
        self.assertEqual(model._get_last_grid_size(1, model.layers_dicts, 2), 2)
        self.assertEqual(output["noises"].shape, (2, 4, 4, 1))

    def test_classifier_empty_connectors_inherit_aggregate_grid(self) -> None:
        """Keep classifier aggregation geometry through empty feature selectors.

        Returns:
            None: Every classifier stage retains the observed square patch grid.
        """

        model = DiTClassifier(
            depth=2,
            clf_depth=3,
            clf_mha_num_heads=1,
            clf_vit_block_mlp_ratio=1.0,
            feature_aggregation_ids_dict={1: [None], 2: [None]},
            clf_connection_ids_dict={1: [], 2: [], 3: [None], -1: [-1]},
            classifier_only_cls_token=False,
            cls_token_type="new_weight",
            **self.config,
        )
        output = model(self.inputs, full_return=True, training=False)
        self.assertEqual(model.first_aggregated_dim, 12)
        self.assertEqual(output["classes"].shape, (2, 2))
        for index in range(len(model.clf_layers_dicts)):
            self.assertEqual(model._get_last_grid_size(index, model.clf_layers_dicts, 2), 2)

    def test_flattened_channel_axes_keep_vector_width(self) -> None:
        """Infer rank-two channel concatenation with the existing zero-grid sentinel.

        Returns:
            None: The supported channel axis produces eight-wide vectors.
        """

        model = DiffusionTransformer(depth=0, **self.config)
        features = [tf.reshape(tf.range(8, dtype=tf.float32), (2, 4))]
        handler = model._create_feature_handler(
            ids_set=[0, 0],
            layers_dicts=[],
            base_dim=4,
            base_grid_size=0,
            dim_forced=False,
            ln_mlp_ratio=1.0,
            ln_no_adaptation=True,
            kwargs={"connect_axis": -1},
        )
        output = handler(features)
        self.assertEqual(output.shape, (2, 8))
        self.assertEqual(handler.output_dim, 8)
        self.assertEqual(handler.grid_size, 0)
        tf.debugging.assert_equal(output, tf.concat(features * 2, axis=-1))

    def test_model_factories_reject_unsupported_merge_axes(self) -> None:
        """Reject axes other than -1 before building feature routing layers.

        Returns:
            None: Transformer, classifier, and decoder factories report the unsupported axis.
        """

        for axis in (1, -2, 2):
            with self.subTest(axis=axis, model="transformer"), self.assertRaisesRegex(
                (ValueError, AssertionError), "connect_axis"
            ):
                DiffusionTransformer(
                    depth=1,
                    vit_block_ids=[],
                    connection_ids_dict={1: [0, 0]},
                    connection_kwargs={"connect_axis": axis},
                    **self.config,
                )
            with self.subTest(axis=axis, model="decoder"), self.assertRaisesRegex(
                (ValueError, AssertionError), "connect_axis"
            ):
                DiTDecoder(
                    depth=1,
                    vit_block_ids=[],
                    use_decoder_ids=[],
                    feature_aggregation_ids_dict={1: [0]},
                    feature_aggregation_kwargs={"connect_axis": axis},
                    **self.decoder_config,
                    **self.config,
                )
            with self.subTest(axis=axis, model="classifier"), self.assertRaisesRegex(
                (ValueError, AssertionError), "connect_axis"
            ):
                DiTClassifier(
                    depth=1,
                    feature_aggregation_kwargs={"connect_axis": axis},
                    **self.config,
                )


# Run these regressions when the module is executed directly.
if __name__ == "__main__":
    unittest.main()
