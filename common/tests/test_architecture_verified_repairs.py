"""Executable regressions for the verified transformer architecture repairs.

These small CPU fixtures check identifiable prefix semantics, transactional
classifier growth, owning numeric policies, progressive grid resizing, and
connected same-pass classifier logits without a training campaign.
"""

from __future__ import annotations

from copy import deepcopy
import itertools
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.model import validate_progressive_classifier_growth
from diffusion import DiTClassifier, DiTEncoderDecoderClassifier, DiffusionTransformer
from diffusion.layers.convolution.variational_reshaper import VariationalReshaper


class ArchitectureVerifiedRepairsTests(unittest.TestCase):
    """Exercise each previously reproduced defect and its nearby valid cases."""

    def setUp(self) -> None:
        """Prepare deterministic, small CPU inputs on the ordinary policy."""
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.utils.set_random_seed(127)
        self.config = dict(
            image_size=4, channels=1, patch_size=2, dim=4, depth=0,
            mha_num_heads=1, num_classes=2, timesteps=4,
        )
        self.inputs = (
            tf.ones((2, 4, 4, 1)),
            tf.zeros((2,), tf.int32),
            tf.ones((2,), tf.int32),
        )

    def tearDown(self) -> None:
        """Restore global numeric state after independent model fixtures."""
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.backend.clear_session()

    def _classifier(self, **kwargs: object) -> DiTClassifier:
        """Construct a tiny classifier with explicit classifier feature routing."""
        config = dict(self.config, clf_mha_num_heads=1,
                      feature_aggregation_ids_dict={1: [0]})
        config.update(kwargs)
        return DiTClassifier(**config)

    def test_prefix_order_for_all_ownership_and_presence_combinations(self) -> None:
        """Read class=11 and distillation=22 from their canonical positions."""
        for cls_only, distil_only, has_cls, has_distil, noise in itertools.product(
            (False, True), repeat=5
        ):
            with self.subTest(cls_only=cls_only, distil_only=distil_only,
                              has_cls=has_cls, has_distil=has_distil, noise=noise):
                # Omitted tokens must not consume a prefix position.
                cls_type = "new_weight" if has_cls else None
                # Omitted distillation leaves class/patch positions unchanged.
                distil_type = "new_weight" if has_distil else None
                model = self._classifier(
                    clf_depth=0, aggregate_from_noises=noise,
                    classifier_only_cls_token=cls_only,
                    classifier_only_distil_token=distil_only,
                    cls_token_type=cls_type, clf_cls_token_type=cls_type,
                    distil_token_type=distil_type, clf_distil_token_type=distil_type,
                )
                # Make the class prefix independently identifiable.
                if has_cls:
                    model.cls_token.token.assign(tf.fill(model.cls_token.token.shape, 11.0))
                # Make the distillation prefix independently identifiable.
                if has_distil:
                    model.distil_token.token.assign(tf.fill(model.distil_token.token.shape, 22.0))
                outputs = model(self.inputs, full_return=True, training=False)
                features = outputs["clf_features_list"][-1]
                self.assertEqual(features.shape[1], 4 + int(has_cls) + int(has_distil))
                # The main class head must read the class token when it exists.
                if has_cls:
                    np.testing.assert_array_equal(features[:, 0], 11.0)
                    np.testing.assert_array_equal(model.classifier_feature_extractor(features), 11.0)
                # The independent distillation head must read its own prefix token.
                if has_distil:
                    np.testing.assert_array_equal(features[:, int(has_cls)], 22.0)
                    np.testing.assert_array_equal(model.distil_feature_extractor(features), 22.0)

    def test_mixed_prefix_pooling_and_auxiliary_slice_keep_class_semantics(self) -> None:
        """Exclude distillation from pooling and feed the class token to its auxiliary head."""
        model = self._classifier(
            clf_depth=1, clf_vit_block_ids=[],
            classifier_only_cls_token=False, cls_token_type="new_weight",
            classifier_only_distil_token=True, clf_distil_token_type="new_weight",
            force_global_avg_pooling=True, clf_cls_token_regularizer_ids=[1],
            clf_cls_token_regularizer_kwargs={"start": 0, "end": 1},
        )
        model.cls_token.token.assign(tf.fill(model.cls_token.token.shape, 11.0))
        model.distil_token.token.assign(tf.fill(model.distil_token.token.shape, 22.0))
        auxiliary = model.clf_layers_dicts[0][model.CTR]
        auxiliary.kernel.assign(tf.concat([tf.ones((1, 2)), tf.zeros((3, 2))], axis=0))
        auxiliary.bias.assign(tf.zeros_like(auxiliary.bias))
        outputs = model(self.inputs, full_return=True, return_logits=True, training=False)
        features = outputs["clf_features_list"][-1]
        expected_pool = tf.reduce_mean(tf.concat([features[:, :1], features[:, 2:]], axis=1), axis=1)
        np.testing.assert_allclose(model.classifier_feature_extractor(features), expected_pool)
        np.testing.assert_array_equal(outputs["clf_regs_logits_list"][1], 11.0)

    def test_incompatible_growth_rejection_preserves_both_branches(self) -> None:
        """Reject changed terminal input/head widths without mutating live model state."""
        for combined, connector in itertools.product(
            (False, True), ({}, {"mlp_output_dim": 4}, {"use_layer_norm": True})
        ):
            with self.subTest(combined=combined, connector=connector):
                model = self._classifier(clf_depth=1, clf_connection_kwargs=connector)
                before = model(self.inputs, training=False)
                config = deepcopy(model.get_config())
                values = model.get_weights()
                identities = [id(weight) for weight in model.weights]
                layers = [id(layer) for layer in model.layers]
                terminal = model.clf_layers_dicts[-1][model.FC]
                terminal_ids = list(terminal.ids)
                encoder_limit = model.max_encoder_num
                spec = {"classifier": {"vision_transformer_block": {"mlp_output_dim": 8}}}
                # Combined requests must not mutate the valid network branch either.
                if combined:
                    spec["network"] = "vision_transformer_block"
                with self.assertRaisesRegex(ValueError, "dimension"):
                    model.add_depths(spec)
                self.assertEqual(model.get_config(), config)
                self.assertEqual(model.clf_depth, 1)
                self.assertEqual(model.depth, 0)
                self.assertEqual(model.max_encoder_num, encoder_limit)
                self.assertEqual(terminal.ids, terminal_ids)
                self.assertEqual([id(weight) for weight in model.weights], identities)
                self.assertEqual([id(layer) for layer in model.layers], layers)
                for old, current in zip(values, model.get_weights()):
                    np.testing.assert_array_equal(current, old)
                after = model(self.inputs, training=False)
                for key in before:
                    np.testing.assert_array_equal(after[key], before[key])

    def test_growth_can_restore_width_before_reaching_existing_head(self) -> None:
        """Accept a complete 4-to-8-to-4 sequence and preserve existing weight identities."""
        model = self._classifier(clf_depth=1)
        old_weights = list(model.weights)
        old_head = model.classifier
        growth = model.add_depths({"classifier": [
            {"vision_transformer_block": {"mlp_output_dim": 8}},
            {"vision_transformer_block": {"mlp_output_dim": 4}},
        ]})
        outputs = model(self.inputs, full_return=True, training=False)
        self.assertEqual(growth["classifier"], {"before": 1, "added": 2, "after": 3})
        self.assertEqual([feature.shape[-1] for feature in outputs["clf_features_list"]], [4, 4, 8, 4, 4])
        self.assertIs(model.classifier, old_head)
        self.assertTrue({id(weight) for weight in old_weights} <= {id(weight) for weight in model.weights})
        clone = DiTClassifier.from_config(model.get_config())
        clone.set_weights(model.get_weights())
        self.assertEqual([weight.shape for weight in model.weights], [weight.shape for weight in clone.weights])
        np.testing.assert_allclose(clone(self.inputs)["classes"], outputs["classes"], atol=1e-7)

    def test_zero_depth_growth_rejects_before_any_mutation(self) -> None:
        """Keep fixed depth-zero classification valid and explain unsupported growth early."""
        model = self._classifier(clf_depth=0)
        before = model(self.inputs, training=False)
        config = deepcopy(model.get_config())
        identities = [id(weight) for weight in model.weights]
        with self.assertRaisesRegex(ValueError, "clf_depth=0 is unsupported"):
            model.add_depths({"network": "vision_transformer_block", "classifier": "vision_transformer_block"})
        self.assertEqual(model.get_config(), config)
        self.assertEqual([id(weight) for weight in model.weights], identities)
        for key, value in model(self.inputs, training=False).items():
            np.testing.assert_array_equal(value, before[key])

    def test_external_preflight_rejects_real_zero_depth_classifier_variants(self) -> None:
        """Catch unsupported growth on both raw variants before orchestration starts training."""
        ordinary = self._classifier(clf_depth=0)
        composite = DiTEncoderDecoderClassifier(
            encoder_kwargs=dict(self.config, clf_depth=0, clf_mha_num_heads=1),
            decoder_kwargs={"depth": 0, "mha_num_heads": 1},
        )
        for model in (ordinary, composite):
            with self.subTest(model=type(model).__name__):
                config = deepcopy(model.get_config())
                identities = [id(weight) for weight in model.weights]
                with self.assertRaisesRegex(ValueError, "clf_depth=0 is unsupported"):
                    validate_progressive_classifier_growth(model, {
                        "stage_tasks": "depths_only",
                        "depths": [{"network": "vision_transformer_block",
                                    "classifier": "vision_transformer_block"}],
                    })
                validate_progressive_classifier_growth(model, {
                    "stage_tasks": "depths_only", "depths": ["vision_transformer_block"],
                })
                self.assertEqual(model.get_config(), config)
                self.assertEqual([id(weight) for weight in model.weights], identities)

    def test_reshaper_policy_survives_other_global_policy_and_config_clone(self) -> None:
        """Preserve owner, outputs, statistics, and variable policy in all reshape modes."""
        for policy_name, mode, kl in itertools.product(
            ("float32", "float64", "mixed_float16"), ("flatten", "unflatten"), (False, True)
        ):
            with self.subTest(policy=policy_name, mode=mode, kl=kl):
                tf.keras.mixed_precision.set_global_policy("float32")
                policy = tf.keras.mixed_precision.Policy(policy_name)
                model = VariationalReshaper(mode, (2, 2, 2), add_kl=kl, dtype=policy, name="policy_reshaper")
                # The unflatten API receives an already flat vector.
                shape = (2, 2, 2, 2) if mode == "flatten" else (2, 8)
                inputs = tf.ones(shape, dtype=policy.compute_dtype)
                outputs = model(inputs)
                self.assertEqual(model.dtype_policy.name, policy_name)
                self.assertEqual(outputs[0].dtype.name, policy.compute_dtype)
                self.assertTrue(all(weight.dtype.name == policy.variable_dtype for weight in model.weights))
                # Only variational flatten statistics are floating latent tensors.
                if mode == "flatten" and kl:
                    self.assertEqual([value.dtype.name for value in outputs], [policy.compute_dtype] * 3)
                # Deterministic reshape statistic placeholders remain int32.
                else:
                    self.assertEqual([value.dtype.name for value in outputs[1:]], ["int32", "int32"])
                tf.keras.mixed_precision.set_global_policy("float64")
                clone = VariationalReshaper.from_config(model.get_config())
                clone.set_weights(model.get_weights())
                self.assertEqual(clone.dtype_policy.name, policy_name)
                self.assertEqual(clone(inputs)[0].dtype.name, policy.compute_dtype)
                self.assertEqual([weight.name for weight in clone.weights], [weight.name for weight in model.weights])
                self.assertTrue(all(weight.dtype.name == policy.variable_dtype for weight in clone.weights))

    def test_float64_sampling_does_not_round_through_float32(self) -> None:
        """A deterministic latent limit preserves information below float32 resolution."""
        model = VariationalReshaper("flatten", (2, 2, 2), add_kl=True, dtype="float64", name="precise")
        mean = model.get_layer("precise/z_mean")
        log_var = model.get_layer("precise/z_log_var")
        precise = 1.0 + 2.0 ** -35
        mean.kernel.assign(tf.zeros_like(mean.kernel))
        mean.bias.assign(tf.fill(mean.bias.shape, tf.constant(precise, tf.float64)))
        log_var.kernel.assign(tf.zeros_like(log_var.kernel))
        log_var.bias.assign(tf.fill(log_var.bias.shape, tf.constant(-2000.0, tf.float64)))
        sample, location, _ = model(tf.ones((1, 2, 2, 2), tf.float64))
        np.testing.assert_array_equal(sample, precise)
        np.testing.assert_array_equal(sample, location)

    def test_resize_preserves_prefix_and_dtype_in_both_directions(self) -> None:
        """Resize grids with zero, one, or two untouched prefix tokens under each policy."""
        for dtype, count, grid_in, grid_out in itertools.product(
            (tf.float32, tf.float64, tf.float16), (0, 1, 2), (2, 4), (2, 4)
        ):
            with self.subTest(dtype=dtype, count=count, grid_in=grid_in, grid_out=grid_out):
                model = DiffusionTransformer(**dict(self.config, image_size=8, build=False))
                model.set_current_resolution(4)
                prefix = tf.cast(tf.reshape(tf.range(count * 4), (1, count, 4)), dtype)
                patches = tf.ones((1, grid_in * grid_in, 4), dtype=dtype)
                tokens = tf.concat([prefix, patches], axis=1)
                output = model._resize_reshaper_tokens(tokens, grid_in, grid_out, 4, count)
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(output.shape, (1, grid_out * grid_out + count, 4))
                np.testing.assert_array_equal(output[:, :count], prefix)
                np.testing.assert_array_equal(output[:, count:], 1.0)

    def test_progressive_kl_reshapers_execute_with_prefix_under_all_policies(self) -> None:
        """Backpropagate through real flatten/unflatten resizing paths with both prefixes."""
        for policy in ("float32", "float64", "mixed_float16"):
            with self.subTest(policy=policy):
                model = DiffusionTransformer(**dict(
                    self.config, image_size=8, depth=2, dtype=policy, vit_block_ids=[],
                    cls_token_type="new_weight", distil_token_type="new_weight",
                    reshaper_ids_dict={1: "flatten", 2: "unflatten"},
                    reshaper_kwargs={"add_kl": True, "latent_dim_ratio": [0.5]},
                ))
                model.set_current_resolution(4)
                inputs = (tf.ones((2, 4, 4, 1), model.compute_dtype), *self.inputs[1:])
                with tf.GradientTape() as tape:
                    tape.watch(inputs[0])
                    output, _, features, _, _ = model(inputs, full_return=True, training=True)
                    loss = tf.reduce_sum(tf.square(tf.cast(output, tf.float64)))
                    loss += tf.reduce_mean(tf.square(tf.cast(features[-1], tf.float64)))
                gradient = tape.gradient(loss, inputs[0])
                self.assertEqual(output.shape, (2, 4, 4, 1))
                self.assertEqual(output.dtype.name, model.compute_dtype)
                self.assertIsNotNone(gradient)
                self.assertTrue(np.isfinite(gradient.numpy()).all())
                self.assertGreater(float(tf.reduce_sum(tf.abs(gradient))), 0.0)

    def test_positional_training_keeps_dropout_and_probability_contracts(self) -> None:
        """Preserve legacy positional training while keeping logits an additive option."""
        model = self._classifier(dropout_rate=0.5)
        full = model(self.inputs, full_return=True, training=False)
        dropout = next(layer for layer in model.classifier.layers
                       if isinstance(layer, tf.keras.layers.Dropout))
        methods = {
            "call": (self.inputs, False, 0),
            "compute_class": (full["features_list"], full["noises"], *self.inputs[1:]),
            "predict_class": (self.inputs, -1, True),
        }
        # Exercise the existing positional training slot on each public classifier method.
        for name, arguments in methods.items():
            method = getattr(model, name)
            # Both explicit training and inference must agree with keyword calls.
            for training in (True, False):
                with self.subTest(method=name, training=training):
                    tf.keras.utils.set_random_seed(709)
                    with patch.object(dropout, "call", wraps=dropout.call) as observed:
                        positional = method(*arguments, training)
                    self.assertEqual(observed.call_count, 1)
                    self.assertEqual(observed.call_args.kwargs["training"], training)
                    tf.keras.utils.set_random_seed(709)
                    keyword = method(*arguments, training=training)
                    # Joint calls preserve their original probability mapping.
                    if name == "call":
                        self.assertEqual(set(positional), {"noises", "classes"})
                        np.testing.assert_array_equal(positional["classes"], keyword["classes"])
                    # Classification helpers preserve their five-entry full-return tuple.
                    else:
                        self.assertEqual(len(positional), 5)
                        np.testing.assert_array_equal(positional[0], keyword[0])
            with self.subTest(method=name, return_logits=True):
                with patch.object(dropout, "call", wraps=dropout.call) as observed:
                    explicit = method(*arguments, True, return_logits=True)
                self.assertEqual(observed.call_args.kwargs["training"], True)
                # Explicit logits add a mapping entry to the joint result only on request.
                if name == "call":
                    self.assertIn("class_logits", explicit)
                # Full helper outputs append the same-pass metadata after existing entries.
                else:
                    self.assertEqual(len(explicit), 6)
                    self.assertIn("class_logits", explicit[-1])

    def test_traced_same_pass_logits_keep_saturated_head_gradients(self) -> None:
        """Expose connected logits without changing probability returns or saved weights."""
        for composite, temperature in itertools.product((False, True), (1.0, 2.0)):
            with self.subTest(composite=composite, temperature=temperature):
                kwargs = dict(self.config, clf_depth=2, clf_mha_num_heads=1,
                              clf_distil_token_type="new_weight", dropout_rate=0.5,
                              clf_cls_token_regularizer_ids=[0, 1],
                              feature_aggregation_ids_dict={1: [0]})
                # The composite owns a decoder but shares the classifier heads.
                if composite:
                    model = DiTEncoderDecoderClassifier(
                        encoder_kwargs=kwargs, decoder_kwargs={"depth": 0, "mha_num_heads": 1}
                    )
                # The ordinary classifier exposes the same additive numerical interface.
                else:
                    model = DiTClassifier(**kwargs)
                model(self.inputs, training=False)
                head = model.classifier.layers[-1]
                head.kernel.assign(tf.zeros_like(head.kernel))
                head.bias.assign([30.0, 0.0])
                identities = [id(weight) for weight in model.weights]
                names = [weight.name for weight in model.weights]
                config = deepcopy(model.get_config())

                @tf.function
                def evaluate() -> tuple[dict[str, object], tf.Tensor]:
                    """Trace the real stochastic classifier call and a stable soft-target loss."""
                    with tf.GradientTape() as tape:
                        result = model(self.inputs, full_return=True, return_logits=True, training=True)
                        teacher = tf.constant([[0.0001, 0.9999]], tf.float32)
                        teacher = tf.nn.softmax(tf.math.log(teacher) / temperature)
                        loss = -temperature ** 2 * tf.reduce_mean(tf.reduce_sum(
                            teacher * tf.nn.log_softmax(result["class_logits"] / temperature), axis=-1
                        ))
                    return result, tape.gradient(loss, head.bias)

                outputs, gradient = evaluate()
                target = np.array([0.0001, 0.9999]) ** (1.0 / temperature)
                target /= target.sum()
                student = np.exp(np.array([30.0, 0.0]) / temperature - 30.0 / temperature)
                student /= student.sum()
                np.testing.assert_allclose(gradient, temperature * (student - target), atol=1e-6)
                np.testing.assert_allclose(tf.nn.softmax(outputs["class_logits"]), outputs["classes"], atol=1e-7)
                np.testing.assert_allclose(tf.nn.softmax(outputs["distil_logits"]), outputs["distil_classes"], atol=1e-7)
                for probabilities, logits in zip(outputs["clf_regs_list"], outputs["clf_regs_logits_list"]):
                    # Every active auxiliary head must expose matching same-pass logits.
                    if probabilities is not None:
                        self.assertIsNotNone(logits)
                        np.testing.assert_allclose(tf.nn.softmax(logits), probabilities, atol=1e-7)
                ordinary = model(self.inputs, training=False)
                self.assertNotIn("class_logits", ordinary)
                self.assertEqual(model.get_config(), config)
                self.assertEqual([id(weight) for weight in model.weights], identities)
                self.assertEqual([weight.name for weight in model.weights], names)
                full = model.predict_class(self.inputs, full_return=True, return_logits=True, training=False)
                self.assertEqual(len(full), 7)
                np.testing.assert_allclose(tf.nn.softmax(full[-1]["class_logits"]), full[0], atol=1e-7)


# Direct execution mirrors unittest discovery without starting application training.
if __name__ == "__main__":
    unittest.main()
