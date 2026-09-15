"""Regressions for transformer route, reconstruction, and ensemble contracts."""

from copy import deepcopy
import itertools
import json
import unittest

import numpy as np
import tensorflow as tf

from diffusion import (
    DiTClassifier, DiTEncoderDecoderClassifier, DiffusionClassifier,
    DiffusionTransformer,
)
from diffusion.metrics.ensemble_accuracy import EnsembleAccuracy


class DiffusionRouteRepairsTests(unittest.TestCase):
    """Exercise the reported failing combinations and their supported neighbors."""

    def setUp(self) -> None:
        """Reset state and prepare deterministic CPU-sized inputs.

        Returns:
            None: A two-class, two-stage model configuration is available.
        """

        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.utils.set_random_seed(217)
        self.base = dict(
            image_size=4, channels=1, patch_size=2, dim=4, depth=2,
            mha_num_heads=1, num_classes=2, timesteps=4,
        )
        self.inputs = (
            tf.reshape(tf.linspace(-1.0, 1.0, 32), (2, 4, 4, 1)),
            tf.zeros((2,), tf.int32), tf.ones((2,), tf.int32),
        )

    def tearDown(self) -> None:
        """Restore the default numeric state after each independent fixture.

        Returns:
            None: Global Keras state is cleared.
        """

        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.backend.clear_session()

    def _classifier(self, **overrides: object) -> DiTClassifier:
        """Construct a small classifier with explicit attention head counts.

        Args:
            **overrides (object): Constructor options overriding the base configuration.

        Returns:
            model (DiTClassifier): A classifier using the requested architecture.
        """

        options = dict(self.base, clf_mha_num_heads=1)
        options.update(overrides)
        return DiTClassifier(**options)

    def test_json_width_maps_restore_for_raw_classifier_and_nested_models(self) -> None:
        """Restore actual JSON keys without modifying the caller's configuration.

        Returns:
            None: All three model families preserve widths, weights, and predictions.
        """

        encoder = dict(self.base, vit_block_mlp_output_dims={1: 8, 2: 4})
        classifier = dict(encoder, clf_depth=2, clf_mha_num_heads=1,
                          clf_vit_block_mlp_output_dims={1: 8, 2: 4})
        candidates = (
            DiffusionTransformer(**encoder),
            DiTClassifier(**classifier),
            DiTEncoderDecoderClassifier(
                encoder_kwargs=classifier,
                decoder_kwargs=dict(depth=2, mha_num_heads=1,
                                    vit_block_mlp_output_dims={1: 8, 2: 4}),
            ),
        )
        for model in candidates:
            with self.subTest(model=type(model).__name__):
                saved = json.loads(json.dumps(model.get_config()))
                untouched = deepcopy(saved)
                restored = type(model).from_config(saved)
                restored.set_weights(model.get_weights())
                self.assertEqual(saved, untouched)
                self.assertEqual(json.loads(json.dumps(restored.get_config())),
                                 json.loads(json.dumps(model.get_config())))
                for expected, actual in zip(
                    tf.nest.flatten(model(self.inputs, training=False)),
                    tf.nest.flatten(restored(self.inputs, training=False)),
                ):
                    np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_full_regularizer_depth_range_round_trips_after_growth(self) -> None:
        """Retain depth-zero plus every processing-stage auxiliary head.

        Returns:
            None: Explicit and grown depth ranges reconstruct without dropping heads.
        """

        with self.assertRaisesRegex(AssertionError, "can only be one of"):
            DiffusionTransformer(**dict(self.base, depth=1,
                                         cls_token_regularizer_ids=[0, 1, 2]))
        with self.assertRaisesRegex(AssertionError, "can only be one of"):
            self._classifier(clf_depth=1, clf_cls_token_regularizer_ids=[0, 1, 2])
        raw = DiffusionTransformer(**dict(self.base, depth=1,
                                         cls_token_regularizer_ids=[0, 1]))
        classifier = self._classifier(
            depth=1, cls_token_regularizer_ids=[0, 1],
            clf_cls_token_regularizer_ids=[0, 1],
        )
        stage = {"vision_transformer_block": True, "cls_token_regularizer": True}
        raw.add_depths(stage)
        classifier.add_depths({"network": stage, "classifier": stage})
        for model in (raw, classifier):
            with self.subTest(model=type(model).__name__):
                self.assertEqual(model.cls_token_regularizer_ids, [0, 1, 2])
                # Classifiers also retain their independent full auxiliary-head range.
                if isinstance(model, DiTClassifier):
                    self.assertEqual(model.clf_cls_token_regularizer_ids, [0, 1, 2])
                expected_outputs = model(self.inputs, training=False)
                restored = type(model).from_config(json.loads(json.dumps(model.get_config())))
                restored.set_weights(model.get_weights())
                self.assertEqual(json.loads(json.dumps(restored.get_config())),
                                 json.loads(json.dumps(model.get_config())))
                for expected, actual in zip(
                    tf.nest.flatten(expected_outputs),
                    tf.nest.flatten(restored(self.inputs, training=False)),
                ):
                    np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_later_feature_merges_preserve_class_and_distillation_positions(self) -> None:
        """Supply neutral missing prefixes while preserving shared token features.

        Returns:
            None: Add/concat and direct/connector routes keep identifiable prefix values.
        """

        for merge, cls_only, distil_only, connector in itertools.product(
            ("add", "concat"), (False, True), (False, True), (False, True)
        ):
            with self.subTest(merge=merge, cls_only=cls_only,
                              distil_only=distil_only, connector=connector):
                model = self._classifier(
                    vit_block_ids=[], clf_depth=2, clf_vit_block_ids=[],
                    cls_token_type="new_weight", clf_cls_token_type="new_weight",
                    distil_token_type="new_weight", clf_distil_token_type="new_weight",
                    classifier_only_cls_token=cls_only,
                    classifier_only_distil_token=distil_only,
                    feature_aggregation_ids_dict={1: [1], 2: [2]},
                    feature_aggregation_kwargs={"connect_type": merge},
                    clf_connection_kwargs={"connect_type": merge},
                    clf_connection_ids_dict={2: [1], -1: [-1]} if connector else {-1: [-1]},
                )
                model.cls_token.token.assign(tf.fill(model.cls_token.token.shape, 11.0))
                model.distil_token.token.assign(tf.fill(model.distil_token.token.shape, 22.0))
                features = model(self.inputs, full_return=True)["clf_features_list"][2]
                self.assertEqual(features.shape[1], 6)
                for index, only, value in ((0, cls_only, 11.0), (1, distil_only, 22.0)):
                    main = np.full((2, 4), 0.0 if only else value)
                    previous = np.full((2, 4), value)
                    # Addition preserves classifier-only prefixes through neutral zeros.
                    if merge == "add":
                        expected = main + previous
                    # A self-connector selects classifier features before its secondary main input.
                    elif connector:
                        expected = np.concatenate([previous, main], axis=-1)
                    # A direct aggregator selects main features before the classifier stream.
                    else:
                        expected = np.concatenate([main, previous], axis=-1)
                    np.testing.assert_array_equal(features[:, index], expected)

    def test_query_prefix_alignment_reuses_current_classifier_queries(self) -> None:
        """Preserve shared prefixes and fill missing queries in canonical order.

        Returns:
            None: Mixed ownership never swaps or discards a class/distillation query.
        """

        for cls_only, distil_only in itertools.product((False, True), repeat=2):
            with self.subTest(cls_only=cls_only, distil_only=distil_only):
                model = self._classifier(
                    build=False, dtype="float64",
                    cls_token_type="new_weight", clf_cls_token_type="new_weight",
                    distil_token_type="new_weight", clf_distil_token_type="new_weight",
                    classifier_only_cls_token=cls_only,
                    classifier_only_distil_token=distil_only,
                )
                main_values = ([] if cls_only else [11.0]) + \
                    ([] if distil_only else [22.0]) + [3.0, 4.0, 5.0, 6.0]
                main = tf.constant(main_values, tf.float64)[None, :, None]
                queries = tf.constant([[[101.0], [202.0]]], tf.float64)
                aligned = model._align_main_feature_prefixes(main, queries)
                expected = [101.0 if cls_only else 11.0,
                            202.0 if distil_only else 22.0, 3.0, 4.0, 5.0, 6.0]
                np.testing.assert_array_equal(aligned[0, :, 0], expected)
                self.assertEqual(aligned.dtype, tf.float64)

    def test_cross_attention_routes_support_prefixes_and_query_width_changes(self) -> None:
        """Connect both attention sides, cross connectors, and wider external queries.

        Returns:
            None: Each route builds, differentiates through images, and survives JSON cloning.
        """

        for side, connector, wide in itertools.product(
            ("queries", "values"), (False, True), (False, True)
        ):
            with self.subTest(side=side, connector=connector, wide=wide):
                model = self._classifier(
                    ln_no_adaptation=True, clf_ln_no_adaptation=True,
                    clf_distil_token_type="new_weight",
                    cross_attention_aggregation_ids_dict={1: [1, 2] if wide else [1]},
                    clf_cross_attention_ids_dict={1: [0]} if connector else {},
                    clf_cross_attention_plug_type=side,
                )
                with tf.GradientTape() as tape:
                    tape.watch(self.inputs[0])
                    outputs = model(self.inputs, training=False)
                    score = tf.reduce_sum(outputs["classes"][:, 0])
                gradient = tape.gradient(score, self.inputs[0])
                self.assertIsNotNone(gradient)
                self.assertGreater(float(tf.reduce_max(tf.abs(gradient))), 0.0)
                self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(gradient))))
                restored = type(model).from_config(json.loads(json.dumps(model.get_config())))
                restored.set_weights(model.get_weights())
                np.testing.assert_allclose(restored(self.inputs)["classes"],
                                           outputs["classes"], atol=1e-6)

    def test_zero_depth_requires_an_image_connected_feature_extractor(self) -> None:
        """Reject the constant-token default and exercise all supported alternatives.

        Returns:
            None: Pooling, no class token, and encoded shared tokens depend on images.
        """

        with self.assertRaisesRegex(ValueError, "no attention stage"):
            self._classifier(clf_depth=0)
        for options in (
            {"force_global_avg_pooling": True},
            {"clf_cls_token_type": None},
            {"classifier_only_cls_token": False, "cls_token_type": "new_weight"},
        ):
            with self.subTest(options=options):
                model = self._classifier(clf_depth=0, ln_no_adaptation=True, **options)
                with tf.GradientTape() as tape:
                    tape.watch(self.inputs[0])
                    score = tf.reduce_sum(model(self.inputs)["classes"][:, 0])
                gradient = tape.gradient(score, self.inputs[0])
                self.assertIsNotNone(gradient)
                self.assertGreater(float(tf.reduce_max(tf.abs(gradient))), 0.0)

    def test_existing_ensembles_follow_raw_and_ema_reconstruction(self) -> None:
        """Reuse ensemble metrics after class growth replaces every network copy.

        Returns:
            None: Both computation modes resolve grown heads and match fresh metrics.
        """

        wrapper = DiffusionClassifier(network=self._classifier(), use_ema=True, test_steps=4)
        metrics = [
            EnsembleAccuracy(wrapper, network_name=selector, compute_type=mode,
                             max_t=2, t_chunk_size=1, seed=83)
            for selector, mode in itertools.product(("raw", "ema"), ("batched", "chunked"))
        ]
        previous = [metric.network for metric in metrics]
        for metric in metrics:
            self.assertEqual(metric.ensemble_predict(self.inputs[0]).shape, (2, 2))
        wrapper._rebuild_classes(3)
        for old, metric in zip(previous, metrics):
            with self.subTest(selector=metric.network_name, mode=metric.compute_type):
                self.assertIsNot(metric.network, old)
                self.assertIs(metric.network, wrapper.get_network(metric.network_name))
                actual = metric.ensemble_predict(self.inputs[0])
                self.assertEqual(actual.shape, (2, 3))
                fresh = EnsembleAccuracy(
                    wrapper, network_name=metric.network_name, compute_type=metric.compute_type,
                    max_t=2, t_chunk_size=1, seed=83,
                )
                np.testing.assert_allclose(actual, fresh.ensemble_predict(self.inputs[0]), atol=1e-6)


# Permit focused execution without discovery.
if __name__ == "__main__":
    unittest.main()
