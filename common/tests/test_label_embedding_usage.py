"""Regression checks for label embeddings owned by active conditioning paths.

Small transformer, classifier, decoder, and composite fixtures distinguish label consumers
from depth-zero auxiliary heads. Tests exercise effective condition modes, token-only
regularization, shared condition widths, retained main resume heads, and decoder resumes
without new token lookups.

Inputs are local tensors constructed by the test fixture. Tests return no application
result: unittest records assertion outcomes and errors. Importing this module defines the
cases; run it through ``python -m unittest`` or directly to execute the checks.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import tensorflow as tf

from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer
from diffusion.models.transformer.di_t_classifier import DiTClassifier
from diffusion.models.transformer.di_t_decoder import DiTDecoder
from diffusion.models.transformer.di_t_encoder_decoder import DiTEncoderDecoder
from diffusion.models.transformer.di_t_encoder_decoder_classifier import DiTEncoderDecoderClassifier


class LabelEmbeddingUsageTests(tf.test.TestCase):
    """Keep auxiliary label heads subordinate to actual label consumers.

    Args:
        methodName (str): Test method selected by unittest. Defaults to ``"runTest"``;
            discovery supplies each named ``test_*`` method.

    Attributes:
        inputs (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Images, times, and valid labels.
        base (dict[str, object]): Small transformer constructor settings.
        classifier_config (dict[str, object]): Independent classifier branch settings.
        decoder_config (dict[str, object]): Shared-condition decoder settings.
    """

    def setUp(self) -> None:
        """Create deterministic tensors and small branch-specific configurations.

        Args:
            None.

        Returns:
            None: Fresh tensors and constructor dictionaries are ready for each test.
        """

        super().setUp()
        self.images = tf.reshape(tf.linspace(-1., 1., 32), (2, 4, 4, 1))
        self.times = tf.constant([0, 1], tf.int32)
        self.labels = tf.constant([1, 2], tf.uint8)
        self.inputs = (self.images, self.times, self.labels)
        self.base = {
            "num_classes": 2,
            "use_cfg": True,
            "timesteps": 4,
            "image_size": 4,
            "channels": 1,
            "patch_size": 2,
            "dim": 4,
            "depth": 1,
            "mha_num_heads": 1,
            "vit_block_mlp_ratio": 1.,
            "cond_type": None,
            "ln_no_adaptation": True,
            "cls_token_type": None,
            "distil_token_type": None,
            "cls_token_regularizer_ids": [0],
            "build": False,
            "seed": 41,
        }
        self.classifier_config = {
            **self.base,
            "clf_cond_type": None,
            "clf_ln_no_adaptation": True,
            "clf_cls_token_type": None,
            "clf_distil_token_type": None,
            "clf_cls_token_regularizer_ids": [0],
            "clf_mha_num_heads": 1,
            "clf_vit_block_mlp_ratio": 1.,
        }
        self.decoder_config = {
            **self.base,
            "cond_type": "time_label",
            "ln_no_adaptation": False,
            "decoder_separate_cond": False,
            "encoder_output_grid_size": 2,
            "encoder_output_dim": 4,
            "encoder_feature_grid_sizes": [2],
            "encoder_feature_dims": [4],
            "shift_inputs": False,
        }
        self.encoder_cond = tf.ones((2, 4))
        self.encoder_features = [tf.ones((2, 4, 4))]

    def test_regularizer_does_not_create_an_unused_label_embedder(self) -> None:
        """Unconditional and time-only networks ignore invalid unused label IDs.

        Args:
            None.

        Returns:
            None: No label lookup is created and depth-zero predictions stay absent.
        """

        for cond_type, no_adaptation in ((None, True), ("time", False)):
            with self.subTest(cond_type=cond_type):
                model = DiffusionTransformer(**{
                    **self.base,
                    "cond_type": cond_type,
                    "ln_no_adaptation": no_adaptation,
                })
                self.assertIsNone(model.label_embedder)
                self.assertIsNone(model.labels_embed_reg)
                output = model(
                    (self.images, self.times, tf.constant([255, 255], tf.uint8)),
                    full_return=True,
                    training=False,
                )
                self.assertEqual(output[0].shape, (2, 4, 4, 1))
                self.assertIsNone(output[3][0])

    def test_unused_label_heads_do_not_block_dynamic_class_growth(self) -> None:
        """Grow an unconditional vocabulary without expanding unused auxiliary heads.

        Args:
            None.

        Returns:
            None: Two class additions leave label layers absent and forward execution valid.
        """

        model = DiffusionTransformer(**{**self.base, "num_classes": None})
        model.add_class()
        model.add_class()
        self.assertEqual(model.num_classes, 2)
        self.assertIsNone(model.label_embedder)
        self.assertIsNone(model.labels_embed_reg)
        output = model(self.inputs, full_return=True, training=False)
        self.assertIsNone(output[3][0])

    def test_main_label_tokens_supply_existing_embeddings_to_regularizers(self) -> None:
        """Class and distillation tokens enable label regularization independently.

        Args:
            None.

        Returns:
            None: Token-only label consumers produce normalized auxiliary predictions.
        """

        for token_name in ("cls_token_type", "distil_token_type"):
            for token_type in ("label", "time_label"):
                with self.subTest(token_name=token_name, token_type=token_type):
                    model = DiffusionTransformer(**{
                        **self.base, token_name: token_type,
                    })
                    self.assertIsNotNone(model.label_embedder)
                    self.assertEqual(
                        model.embed_conditions(
                            self.times, None, None, full_return=True, training=False,
                        ),
                        (None, None, None),
                    )
                    output = model(self.inputs, full_return=True, training=False)
                    self.assertIsNone(output[1])
                    self.assertEqual(output[3][0].shape, (2, 2))
                    self.assertAllClose(tf.reduce_sum(output[3][0], axis=-1), [1., 1.])

    def test_main_resume_preserves_existing_label_regularization(self) -> None:
        """Resuming main features retains the existing depth-zero label head.

        Args:
            None.

        Returns:
            None: Resumed features and label predictions match the fresh execution.
        """

        model = DiffusionTransformer(**{**self.base, "cls_token_type": "label"})
        original = model.encode(self.inputs, training=False)
        resumed = model.encode(
            (original[2][-1], self.times, self.labels), min_depth=1, training=False,
        )
        self.assertAllClose(resumed[0], original[0])
        self.assertIsNotNone(resumed[3][0])
        self.assertAllClose(resumed[3][0], original[3][0])

    def test_separate_decoder_tokens_ignore_disabled_condition_mode(self) -> None:
        """Separate time and label tokens do not activate a disabled merged condition.

        Args:
            None.

        Returns:
            None: Decoder tokens and label regularization work with neutral conditioning.
        """

        model = DiTDecoder(**{
            **self.decoder_config,
            "decoder_separate_cond": True,
            "ln_no_adaptation": True,
            "cls_token_type": "time",
            "distil_token_type": "label",
        })
        self.assertIsNone(model.conds_merger)
        output = model.decode(
            self.inputs,
            self.encoder_cond,
            self.encoder_features,
            full_return=True,
            training=False,
        )
        self.assertEqual(output[0].shape, (2, 4, 4))
        self.assertAllEqual(output[1], tf.zeros((2, 4)))
        self.assertEqual(output[3][0].shape, (2, 2))

    def test_noise_classifier_tokens_ignore_disabled_condition_mode(self) -> None:
        """Noise re-embedding preserves shared tokens without activating conditions.

        Args:
            None.

        Returns:
            None: Noise classification succeeds with separate time and label tokens.
        """

        model = DiTClassifier(**{
            **self.classifier_config,
            "aggregate_from_noises": True,
            "cond_type": "time_label",
            "classifier_only_cls_token": False,
            "classifier_only_distil_token": False,
            "cls_token_type": "time",
            "distil_token_type": "label",
        })
        self.assertIsNone(model.conds_merger)
        output = model(self.inputs, full_return=True, training=False)
        self.assertIsNone(output["cond"])
        self.assertIsNone(output["clf_cond"])
        self.assertEqual(output["classes"].shape, (2, 2))
        self.assertEqual(output["distil_classes"].shape, (2, 2))

    def test_composite_shared_width_requires_an_active_encoder_condition(self) -> None:
        """Require equal condition widths only when a composite shares real conditions.

        Args:
            None.

        Returns:
            None: Both composites allow unused width differences and reject active ones.
        """

        for model_class, encoder_config in (
            (DiTEncoderDecoder, self.base),
            (DiTEncoderDecoderClassifier, self.classifier_config),
        ):
            with self.subTest(model=model_class.__name__):
                config = {
                    **encoder_config,
                    "cond_type": "time_label",
                    "cls_token_regularizer_ids": [],
                    "decoder_kwargs": {
                        "depth": 1,
                        "mha_num_heads": 1,
                        "vit_block_mlp_ratio": 1.,
                        "cond_dim": 8,
                        "ln_no_adaptation": True,
                        "decoder_separate_cond": False,
                        "shift_inputs": False,
                    },
                }
                model = model_class(**config)
                output = model.predict_noise(self.inputs, full_return=True, training=False)
                self.assertIsNone(output[1])
                self.assertEqual(output[0].shape, (2, 4, 4, 1))
                with self.assertRaisesRegex(ValueError, "same cond_dim"):
                    model_class(**{**config, "ln_no_adaptation": False})

    def test_classifier_ignores_label_tokens_without_classifier_ownership(self) -> None:
        """Disabled or ignored classifier token settings cannot create label lookups.

        Args:
            None.

        Returns:
            None: Both branches return absent depth-zero predictions without label users.
        """

        for token_type in (None, "label", "time_label"):
            with self.subTest(token_type=token_type):
                model = DiTClassifier(**{
                    **self.classifier_config,
                    "classifier_only_cls_token": False,
                    "classifier_only_distil_token": False,
                    "clf_cls_token_type": token_type,
                    "clf_distil_token_type": token_type,
                })
                self.assertIsNone(model.label_embedder)
                self.assertIsNone(model.labels_embed_reg)
                self.assertIsNone(model.clf_labels_embed_reg)
                output = model(
                    (self.images, self.times, tf.constant([255, 255], tf.uint8)),
                    full_return=True,
                    training=False,
                )
                self.assertEqual(output["classes"].shape, (2, 2))
                self.assertIsNone(output["regs_list"][0])
                self.assertIsNone(output["clf_regs_list"][0])

    def test_classifier_label_tokens_regularize_only_the_consuming_branch(self) -> None:
        """A classifier token reuses its lookup without enabling the main label head.

        Args:
            None.

        Returns:
            None: Classifier auxiliary predictions exist while the main slot stays empty.
        """

        for token_name in ("clf_cls_token_type", "clf_distil_token_type"):
            for token_type in ("label", "time_label"):
                with self.subTest(token_name=token_name, token_type=token_type):
                    model = DiTClassifier(**{
                        **self.classifier_config, token_name: token_type,
                    })
                    output = model(self.inputs, full_return=True, training=False)
                    self.assertIsNotNone(model.label_embedder)
                    self.assertIsNone(output["regs_list"][0])
                    self.assertIsNone(output["clf_cond"])
                    regularizer = output["clf_regs_list"][0]
                    self.assertEqual(regularizer.shape, (2, 2))
                    self.assertAllClose(tf.reduce_sum(regularizer, axis=-1), [1., 1.])

    def test_shared_decoder_drops_label_lookup_without_label_tokens(self) -> None:
        """Encoder-provided conditions do not require a decoder label lookup.

        Args:
            None.

        Returns:
            None: Shared conditioning works with absent local label regularization.
        """

        model = DiTDecoder(**self.decoder_config)
        self.assertIsNone(model.label_embedder)
        self.assertIsNone(model.labels_embed_reg)
        output = model.decode(
            (self.images, self.times, None),
            self.encoder_cond,
            self.encoder_features,
            full_return=True,
            training=False,
        )
        self.assertEqual(output[0].shape, (2, 4, 4))
        self.assertIsNone(output[3][0])

    def test_shared_decoder_tokens_regularize_without_forcing_resumed_lookups(self) -> None:
        """Decoder-owned tokens retain labels only while their entrance executes.

        Args:
            None.

        Returns:
            None: Active tokens regularize labels and resumed tokens perform no lookup.
        """

        for token_name in ("cls_token_type", "distil_token_type"):
            for token_type in ("label", "time_label"):
                with self.subTest(token_name=token_name, token_type=token_type):
                    model = DiTDecoder(**{
                        **self.decoder_config, token_name: token_type,
                    })
                    self.assertIsNotNone(model.label_embedder)
                    original = model.decode(
                        self.inputs,
                        self.encoder_cond,
                        self.encoder_features,
                        full_return=True,
                        training=False,
                    )
                    self.assertEqual(original[3][0].shape, (2, 2))
                    self.assertAllClose(
                        tf.reduce_sum(original[3][0], axis=-1), [1., 1.],
                    )
                    with patch.object(
                        model.label_embedder,
                        "call",
                        side_effect=AssertionError("A skipped decoder token cannot read labels."),
                    ):
                        resumed = model.decode(
                            (original[2][-1], self.times, None),
                            self.encoder_cond,
                            self.encoder_features,
                            full_return=True,
                            min_depth=1,
                            training=False,
                        )
                    self.assertAllClose(resumed[0], original[0])
                    self.assertIsNone(resumed[3][0])


# Execute the cases only when this module is run as a script.
if __name__ == "__main__":
    unittest.main()
