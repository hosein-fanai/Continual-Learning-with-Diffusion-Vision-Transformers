"""Exercise transfer-learning selection and state safety without downloading weights."""

from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import tensorflow as tf

from common.config import Config, ModelConfig, load_config, save_config
from common.model import get_model


_BACKBONES = (
    "Xception", "EfficientNetV2B0", "EfficientNetV2B1", "EfficientNetV2B2", 
    "EfficientNetV2B3", "EfficientNetV2S", "EfficientNetV2M", "EfficientNetV2L"
)


def _tiny_base(name: str, **options: object) -> tf.keras.Model:
    """Replace expensive ImageNet constructors with real, serializable Keras layers."""

    inputs = tf.keras.Input(shape=options["input_shape"])
    features = inputs
    # EfficientNet applications own their pixel normalization.
    if name.startswith("EfficientNet"):
        features = tf.keras.layers.Rescaling(
            1.0 / 127.5, offset=-1.0, name="embedded_preprocess"
        )(features)
    features = tf.keras.layers.Conv2D(
        2, 1, strides=8, use_bias=False, kernel_initializer="ones", 
        name="early_conv"
    )(features)
    features = tf.keras.layers.BatchNormalization(name="early_bn")(features)
    features = tf.keras.layers.Conv2D(
        2, 1, use_bias=False, kernel_initializer="ones", name="tail_conv"
    )(features)
    features = tf.keras.layers.BatchNormalization(name="tail_bn")(features)
    features = tf.keras.layers.Activation("relu", name="tail_activation")(features)
    return tf.keras.Model(inputs, features, name="tiny_" + name.lower())


class PretrainedBackboneTests(unittest.TestCase):
    """Check public API routing, raw pixels, fine-tuning state, and persistence."""

    def setUp(self) -> None:
        """Keep all constructors local and all numerical checks reproducible."""

        self.original_policy = tf.keras.mixed_precision.global_policy().name
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.utils.set_random_seed(913)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.constructors = {}
        for name in _BACKBONES:
            self.constructors[name] = self.stack.enter_context(patch.object(
                tf.keras.applications, name, 
                side_effect=lambda _name=name, **options: _tiny_base(_name, **options)
            ))
        self.images = tf.reshape(tf.linspace(150.0, 250.0, 4 * 32 * 32 * 3), 
                                 (4, 32, 32, 3))
        self.labels = tf.constant([0, 1, 2, 0], tf.int32)

    def tearDown(self) -> None:
        """Release test graphs and restore the caller's numerical policy."""

        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy(self.original_policy)

    def make_model(self, name: str = "Xception", **overrides: object) -> tf.keras.Model:
        """Build the original notebook API with a small fake convolutional base."""

        options = dict(model_type="pretrained", conv_base_name=name, resize=(75, 75), 
                       num_last_not_frozen=3, dropout_rate=0.0, compile_args={"optimizer": tf.keras.optimizers.SGD(0.01), 
                                     "run_eagerly": True, "jit_compile": False}, 
                       verbose=0)
        options.update(overrides)
        return get_model(3, **options)

    def base(self, model: tf.keras.Model) -> tf.keras.Model:
        """Find the nested application model without depending on layer positions."""

        return next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))

    def test_legacy_default_and_case_insensitive_supported_names(self) -> None:
        """Preserve Xception by default and resolve every supported application safely."""

        default = get_model(3, model_type="pretrained", resize=(75, 75), verbose=0)
        self.assertEqual(self.base(default).name, "tiny_xception")
        for name in _BACKBONES:
            with self.subTest(backbone=name):
                model = self.make_model(name.swapcase())
                constructor_options = self.constructors[name].call_args.kwargs
                self.assertFalse(constructor_options["include_top"])
                self.assertEqual(constructor_options["weights"], "imagenet")
                self.assertEqual(constructor_options["input_shape"], (75, 75, 3))
                # EfficientNet must keep its built-in preprocessing enabled.
                if name.startswith("EfficientNet"):
                    self.assertTrue(constructor_options["include_preprocessing"])
                self.assertIsInstance(model, tf.keras.Sequential)
                self.assertEqual(tuple(model(self.images).shape), (4, 3))

    def test_direct_and_config_selection_precedence(self) -> None:
        """Use the requested base through both direct forms and typed configuration."""

        direct = dict(model_name="pretrained", dataset_name="cifar10", class_num=3, 
                      resize=(75, 75), show_network_summary=False)
        for route in (
            {"conv_base_name": "EfficientNetV2L"}, 
            {"model_kwargs": {"conv_base_name": "EfficientNetV2L"}}
        ):
            with self.subTest(route=route):
                model = get_model(**direct, **route)
                self.assertEqual(self.base(model).name, "tiny_efficientnetv2l")
        config = Config(dataset={"name": "cifar100"}, 
                        optimizer={"schedule": "constant"}, model={
            "name": "pretrained", "conv_base_name": "EfficientNetV2L", 
            "show_network_summary": False, "kwargs": {"resize": [75, 75]}
        })
        model = get_model(config)
        self.assertEqual(self.base(model).name, "tiny_efficientnetv2l")
        self.assertEqual(model.output_shape[-1], 100)
        config.model.kwargs["conv_base_name"] = "Xception"
        overridden = get_model(config)
        self.assertEqual(self.base(overridden).name, "tiny_xception")
        self.assertEqual(config.model.conv_base_name, "EfficientNetV2L")

    def test_external_classifier_options_override_config_backbone(self) -> None:
        """Apply the shared default and an explicit override to continual classifiers."""

        config = Config(
            dataset={"name": "cifar10"}, 
            model={"name": "diffusion_transformer", "conv_base_name": "EfficientNetV2L", 
                   "classifier_name": "pretrained", "show_network_summary": False, 
                   "classifier_kwargs": {"resize": [75, 75]}}, 
            optimizer={"schedule": "constant"}, 
            continually_learn={"use_buffer": True, "class_num": 3}, 
            training={"task": "continual"}
        )
        bundle = get_model(config)
        self.assertIsNone(bundle["generative_model"])
        self.assertEqual(self.base(bundle["classifier"]).name, "tiny_efficientnetv2l")
        config.model.classifier_kwargs["conv_base_name"] = "Xception"
        overridden = get_model(config)
        self.assertEqual(self.base(overridden["classifier"]).name, "tiny_xception")

    def test_direct_external_backbone_does_not_leak_into_generative_network(self) -> None:
        """Route a top-level backbone choice solely to the external replay classifier."""

        def make_network(
            num_classes: int | None, image_size: int, channels: int, 
            timesteps: int, seed: int | None
        ) -> SimpleNamespace:
            """Accept only replay-network arguments so leaked classifier keys fail."""

            return SimpleNamespace(timesteps=timesteps)

        generator = Mock(name="generative_model")
        with patch("diffusion.DiffusionTransformer", side_effect=make_network) as constructor, \
             patch("diffusion.DiffusionModel", return_value=generator):
            bundle = get_model(
                model_name="diffusion_transformer", dataset_name="cifar10", 
                task="continual", classifier_name="pretrained", use_buffer=False, 
                conv_base_name="EfficientNetV2L", model_kwargs={"timesteps": 4}, 
                classifier_kwargs={"resize": [75, 75]}, show_network_summary=False
            )
        self.assertIs(bundle["generative_model"], generator)
        self.assertEqual(self.base(bundle["classifier"]).name, "tiny_efficientnetv2l")
        self.assertNotIn("conv_base_name", constructor.call_args.kwargs)
        generator.compile.assert_called_once()

    def test_preprocessing_reaches_first_convolution_exactly_once(self) -> None:
        """Transform raw [0, 255] pixels once for either application family."""

        pixels = tf.stack([tf.zeros((32, 32, 3)), tf.fill((32, 32, 3), 255.0)])
        expected = np.broadcast_to(np.array([-1.0, 1.0])[:, None, None, None], 
                                   (2, 75, 75, 3))
        for name in ("Xception", "EfficientNetV2L"):
            with self.subTest(backbone=name):
                model = self.make_model(name)
                base = self.base(model)
                features = pixels
                for layer in model.layers:
                    # Stop at the application input to detect accidental double preprocessing.
                    if layer is base:
                        break
                    features = layer(features, training=False)
                # The embedded EfficientNet layer must receive raw-range pixels.
                if name.startswith("EfficientNet"):
                    self.assertEqual(float(tf.reduce_max(features)), 255.0)
                    self.assertFalse(any(isinstance(layer, tf.keras.layers.Rescaling)
                                         for layer in model.layers))
                probe = tf.keras.Model(base.inputs, base.get_layer("early_conv").input)
                np.testing.assert_allclose(probe(features).numpy(), expected, atol=1e-6)

    def test_fine_tuning_updates_only_requested_non_batchnorm_weights(self) -> None:
        """Train the head/tail while keeping frozen layers and all BN state unchanged."""

        for name in ("Xception", "EfficientNetV2L"):
            for tail in (0, 3, None):
                with self.subTest(backbone=name, trainable_tail=tail):
                    model = self.make_model(name, num_last_not_frozen=tail)
                    base = self.base(model)
                    early = base.get_layer("early_conv")
                    last = base.get_layer("tail_conv")
                    self.assertEqual(early.trainable, tail is None)
                    self.assertEqual(last.trainable, tail != 0)
                    batchnorm = [layer for layer in base.layers
                                 if isinstance(layer, tf.keras.layers.BatchNormalization)]
                    self.assertTrue(all(not layer.trainable for layer in batchnorm))
                    before_base = [(variable, variable.numpy().copy()) for variable in base.weights]
                    before_head = [value.copy() for value in model.layers[-1].get_weights()]
                    logs = model.train_on_batch(self.images, self.labels, return_dict=True)
                    self.assertTrue(all(np.isfinite(value) for value in logs.values()))
                    self.assertTrue(any(not np.array_equal(before, after)
                                        for before, after in zip(before_head, 
                                                                 model.layers[-1].get_weights())))
                    trainable_ids = {id(variable) for variable in base.trainable_weights}
                    for variable, before in before_base:
                        # Frozen convolution and BN state must survive a real optimizer step.
                        if id(variable) not in trainable_ids:
                            np.testing.assert_array_equal(variable.numpy(), before)
                    for convolution in (early, last):
                        before = next(value for variable, value in before_base
                                      if variable is convolution.kernel)
                        self.assertEqual(not np.array_equal(convolution.kernel.numpy(), before), 
                                         convolution.trainable)

    def test_invalid_selection_tail_and_resize_fail_before_weight_loading(self) -> None:
        """Reject unsupported names and invalid discrete controls before constructors run."""

        for value in ("", "ResNet50", "__dict__", None, 7):
            with self.subTest(conv_base_name=value), self.assertRaises(ValueError):
                self.make_model(value)
        for value in (-1, 1.5, True, "3"):
            with self.subTest(num_last_not_frozen=value), self.assertRaises(ValueError):
                self.make_model(num_last_not_frozen=value)
        for value in (tuple([75]), (75, 75, 3), (75.5, 75), (True, 75)):
            with self.subTest(resize=value), self.assertRaises(ValueError):
                self.make_model(resize=value)

        for constructor in self.constructors.values():
            constructor.assert_not_called()

    def test_spatial_limits_are_delegated_to_native_applications(self) -> None:
        """Let each real Keras application reject spatial bounds before downloading weights."""

        self.stack.close()
        with patch("keras.src.utils.file_utils.get_file", side_effect=AssertionError(
            "Invalid geometry must fail before loading weights."
        )) as download:
            for name in _BACKBONES:
                minimum_size = 71 if name == "Xception" else 32
                constructor = getattr(tf.keras.applications, name)
                with patch.object(tf.keras.applications, name, wraps=constructor) as native:
                    for resize in ((0, 75), (-1, 75), (minimum_size - 1, 75), 
                                   (75, 0), (75, -1), (75, minimum_size - 1)):
                        with self.subTest(backbone=name, resize=resize), self.assertRaisesRegex(
                            ValueError, "Input size must be at least"
                        ):
                            self.make_model(name, resize=resize)
                    self.assertEqual(native.call_count, 6)
            download.assert_not_called()
    def test_config_yaml_round_trip_preserves_default_and_explicit_names(self) -> None:
        """Keep backbone choice in full/compact YAML and restore omitted defaults."""

        self.assertEqual(ModelConfig().conv_base_name, "Xception")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pretrained.yaml"
            for name in ("Xception", "EfficientNetV2L"):
                for shorten in (False, True):
                    with self.subTest(backbone=name, shorten=shorten):
                        config = Config(model={"name": "pretrained", "conv_base_name": name})
                        save_config(config, path, shorten=shorten)
                        restored = load_config(path)
                        self.assertEqual(restored.model.conv_base_name, name)

    def test_safe_keras_round_trip_preserves_predictions_and_frozen_bn(self) -> None:
        """Save/reload standard Keras layers safely with preprocessing and BN flags intact."""

        with tempfile.TemporaryDirectory() as directory:
            for name in ("Xception", "EfficientNetV2L"):
                with self.subTest(backbone=name):
                    model = self.make_model(name)
                    model.train_on_batch(self.images, self.labels)
                    expected = model(self.images, training=False).numpy()
                    path = Path(directory) / (name + ".keras")
                    model.save(path)
                    restored = tf.keras.models.load_model(path, safe_mode=True)
                    np.testing.assert_allclose(restored(self.images, training=False).numpy(), 
                                               expected, rtol=1e-6, atol=1e-7)
                    self.assertIsInstance(restored, tf.keras.Sequential)
                    for layer in self.base(restored).layers:
                        # Serialization must preserve inference-only BN behavior.
                        if isinstance(layer, tf.keras.layers.BatchNormalization):
                            self.assertFalse(layer.trainable)


# Allow a focused direct invocation outside discovery.
if __name__ == "__main__":
    unittest.main()
