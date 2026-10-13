"""Bounded native DiT checks for temporary internal TMCL modulation ownership."""

from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import Config
from common.model import get_model
from semantic_consolidation.tmcl_adapter import TMCLDiTAdapter


def make_wrapper(overrides: dict[str, object] | None = None) -> tf.keras.Model:
    """Build a small genuine project DiT with nonzero initial transformer paths."""

    config = Config()
    config.training.task = "joint"
    config.training.seed = 41
    config.training.dtype_policy = "float32"
    config.dataset.trainset_len = 2
    config.model.name = "dit_classifier"
    config.model.wrapper_name = "diffusion_classifier"
    config.model.show_network_summary = False
    config.model.kwargs = {
        "num_classes": 2, "use_cfg": True, "timesteps": 4, 
        "image_size": 4, "channels": 1, "patch_size": 2, 
        "dim": 8, "cond_dim": 8, "depth": 2, "mha_num_heads": 2, 
        "vit_block_mlp_ratio": 2., "ln_no_adaptation": True, 
        "clf_depth": 1, "clf_mha_num_heads": 2, "clf_ln_no_adaptation": True, 
        "clf_cls_token_type": None, "feature_aggregation_ids_dict": {1: [2]}, 
        "force_global_avg_pooling": True, "classifier_mlp_ratio": 1, 
        "classifier_dropout_rate": 0., "build": True
    }
    config.model.wrapper_kwargs = {"use_ema": False, "test_network_name": "raw", "test_steps": 2}
    config.model.kwargs.update(overrides or {})
    return get_model(config)


class TMCLAdapterTests(unittest.TestCase):
    """Protect numerical equivalence, row routing, gradient scope and native state."""

    def setUp(self) -> None:
        """Construct native layers without any external dataset or learned checkpoint."""

        self.previous_tf32 = tf.config.experimental.tensor_float_32_execution_enabled()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        self.addCleanup(tf.config.experimental.enable_tensor_float_32_execution, self.previous_tf32)
        self.wrapper = make_wrapper()
        self.adapter = TMCLDiTAdapter(self.wrapper, seed=53)
        self.adapter.bank.add([0, 1])
        self.images = tf.reshape(tf.linspace(-1., 1., 64), (4, 4, 4, 1))

    def tearDown(self) -> None:
        """Release the isolated fixture after its independent validation process."""

        tf.keras.backend.clear_session()

    def test_zero_gates_equal_native_feature_path(self) -> None:
        """Identity gates preserve the original native attention and MLP computation."""

        times = tf.zeros(4, tf.int32)
        native = self.wrapper.network.predict_class(
            (self.images, times, times), max_encoder_num=None, full_return=True, training=False
        )
        expected = self.wrapper.network.classifier_feature_extractor(native[2][-1], training=False)
        for variable in self.adapter.bank.variables_for():
            variable.assign(tf.zeros_like(variable))
        np.testing.assert_array_equal(self.adapter.features(self.images).numpy(), expected.numpy())
        np.testing.assert_allclose(
            self.adapter.features(self.images, tf.constant([0, 1, 0, 1])).numpy(), 
            expected.numpy(), rtol=1e-6, atol=1e-6
        )
        self.assertGreaterEqual(len(self.adapter.bank.sites), 12)
        head_ids = {id(value) for value in self.wrapper.network.classifier.variables}
        self.assertFalse(head_ids.intersection(id(value) for value in self.adapter.backbone_variables))

    def test_per_row_routing_matches_separate_native_forwards(self) -> None:
        """One gate ID per row must not broadcast the first selected class to all rows."""

        ids = tf.constant([0, 1, 1, 0])
        combined = self.adapter.features(self.images, ids).numpy()
        separate = tf.concat([
            self.adapter.features(self.images[index:index + 1], ids[index:index + 1])
            for index in range(4)
        ], axis=0).numpy()
        np.testing.assert_allclose(combined, separate, rtol=2e-5, atol=2e-5)
        self.assertGreater(np.max(np.abs(combined - self.adapter.features(self.images).numpy())), 1e-7)

    def test_mlp_affine_precedes_native_activation(self) -> None:
        """The hidden Dense affine gate belongs before GELU, not after it."""

        index = next(index for index, layer in enumerate(self.adapter._layers)
                     if isinstance(layer, tf.keras.layers.Dense) and layer.activation is not tf.keras.activations.linear)
        layer = self.adapter._layers[index]
        inputs = tf.ones((4, 2, int(layer.kernel.shape[0])))
        activation = layer.activation
        linear = tf.linalg.matmul(inputs, layer.kernel) + layer.bias
        ids = tf.constant([0, 1, 0, 1])
        expected = activation(self.adapter.bank.apply(linear, index, ids))
        with self.adapter.modulated(ids):
            actual = layer(inputs)
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)
        self.assertIs(layer.activation, activation)

    def test_hooks_restore_after_exception_and_reject_nested_owner(self) -> None:
        """A failed modulated forward cannot poison later native inference or tracing."""

        native = self.adapter.features(self.images).numpy()
        original = [("call" in vars(layer), layer.call, getattr(layer, "activation", None))
                    for layer in self.adapter._layers]
        with patch.object(self.wrapper.network, "predict_class", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                self.adapter.features(self.images, tf.zeros(4, tf.int32))
        for layer, (owned, call, activation) in zip(self.adapter._layers, original):
            self.assertEqual("call" in vars(layer), owned)
            self.assertEqual(layer.call, call)
            self.assertIs(getattr(layer, "activation", None), activation)
        np.testing.assert_array_equal(self.adapter.features(self.images).numpy(), native)
        second = TMCLDiTAdapter(self.wrapper, seed=59)
        with self.adapter.modulated(None):
            with self.assertRaisesRegex(RuntimeError, "Concurrent or nested"):
                second.features(self.images)
        with self.assertRaisesRegex(RuntimeError, "eager"):
            tf.function(self.adapter.features)(self.images)

    def test_gate_gradient_changes_only_selected_memory(self) -> None:
        """Explicit gate-only tape still differentiates through frozen native operators."""

        before = [value.numpy().copy() for value in self.wrapper.network.variables]
        retained = [value.numpy().copy() for value in self.adapter.bank.variables_for([0])]
        selected = self.adapter.bank.variables_for([1])
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(selected)
            features = self.adapter.features(self.images, tf.ones(4, tf.int32))
            loss = tf.reduce_mean(tf.square(features))
        gradients = tape.gradient(loss, selected)
        self.assertTrue(all(value is not None for value in gradients))
        self.assertGreater(float(tf.linalg.global_norm(gradients)), 0.)
        tf.keras.optimizers.SGD(0.01).apply_gradients(zip(gradients, selected))
        for expected, variable in zip(before, self.wrapper.network.variables):
            np.testing.assert_array_equal(expected, variable.numpy())
        for expected, variable in zip(retained, self.adapter.bank.variables_for([0])):
            np.testing.assert_array_equal(expected, variable.numpy())

    def test_bank_growth_and_snapshot_restore_preserve_prior_state(self) -> None:
        """Adding new classes cannot overwrite retained gates; corrupt layouts fail first."""

        saved = self.adapter.snapshot()
        self.adapter.bank.add([0, 1])
        for expected, variable in zip(saved["bank"]["gain"], self.adapter.bank.variables_for()[::2]):
            np.testing.assert_array_equal(expected, variable.numpy())
        self.adapter.bank.variables_for([0])[0].assign_add(tf.ones(self.adapter.bank.dimension))
        self.adapter.restore(saved)
        np.testing.assert_array_equal(saved["bank"]["gain"][0], self.adapter.bank.variables_for([0])[0].numpy())
        bad = deepcopy(saved["bank"])
        bad["sites"][0]["key"] = "wrong_coordinate"
        with self.assertRaisesRegex(ValueError, "layout"):
            self.adapter.bank.restore(bad)
        self.adapter.bank.add([2])
        np.testing.assert_array_equal(saved["bank"]["gain"][0], self.adapter.bank.variables_for([0])[0].numpy())
        with self.assertRaisesRegex(ValueError, "class identity"):
            self.adapter.bank.restore(saved["bank"])

    def test_readout_matches_ordered_native_layers_and_is_stopped(self) -> None:
        """The last-layer concatenation contains genuine ancestors in their forward order."""

        times = tf.zeros(4, tf.int32)
        encoded = self.wrapper.network.encode((self.images, times, times), training=False)
        classified = self.wrapper.network.predict_class(
            (self.images, times, times), max_encoder_num=None, full_return=True, training=False
        )
        expected = tf.concat([
            tf.reduce_mean(encoded[2][1], axis=1), 
            tf.reduce_mean(encoded[2][2], axis=1), 
            tf.reduce_mean(classified[2][-1], axis=1)
        ], axis=1)
        inputs = tf.Variable(self.images)
        with tf.GradientTape() as tape:
            actual = self.adapter.readout_features(inputs, last_layers=3)
            loss = tf.reduce_sum(actual)
        self.assertIsNone(tape.gradient(loss, inputs))
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)
        spec = self.adapter.readout_spec(3)
        self.assertEqual(spec["width"], 24)
        self.assertEqual([value["branch"] for value in spec["layers"]], ["encoder", "encoder", "classifier"])
        self.assertTrue(all(value["pooling"] == "native_gap_without_distil" for value in spec["layers"]))
        np.testing.assert_allclose(
            self.adapter.readout_features(self.images, 1).numpy(), 
            self.adapter.features(self.images).numpy(), rtol=1e-6, atol=1e-6
        )

    def test_readout_rejects_shallow_and_branched_routes(self) -> None:
        """Readout requests cannot silently change depth or use ambiguous layer ancestry."""

        with self.assertRaisesRegex(ValueError, "depth"):
            self.adapter.readout_spec(4)
        with patch.object(self.wrapper.network, "feature_aggregation_ids_dict", {1: [1, 2]}):
            with self.assertRaisesRegex(ValueError, "final-depth"):
                self.adapter.readout_spec(1)
        with patch.object(self.wrapper.network, "max_encoder_num", 1):
            with self.assertRaisesRegex(ValueError, "stop changed"):
                self.adapter.readout_spec(1)

    def test_readout_restores_block_calls_after_failure(self) -> None:
        """Even failed readout extraction leaves the training feature path unchanged."""

        original = [("call" in vars(block), block.call) for block in self.adapter._blocks]
        with patch.object(self.wrapper.network, "predict_class", side_effect=RuntimeError("readout failure")):
            with self.assertRaisesRegex(RuntimeError, "readout failure"):
                self.adapter.readout_features(self.images, 1)
        for block, (owned, call) in zip(self.adapter._blocks, original):
            self.assertEqual("call" in vars(block), owned)
            self.assertEqual(block.call, call)
        self.assertEqual(self.adapter.features(self.images).shape, (4, 8))

    def test_readout_records_native_cls_and_omits_unexecuted_main_blocks(self) -> None:
        """Classifier-only CLS and truncated main depth have explicit distinct provenance."""

        wrapper = make_wrapper({"feature_aggregation_ids_dict": {1: [1]}, 
                                "clf_cls_token_type": "new_weight", "force_global_avg_pooling": False})
        adapter = TMCLDiTAdapter(wrapper)
        spec = adapter.readout_spec(2)
        self.assertEqual(len(adapter._blocks), 2)
        self.assertEqual(spec["layers"][0]["pooling"], "native_gap_without_distil")
        self.assertEqual(spec["layers"][1]["pooling"], "native_cls")
        self.assertEqual(adapter.readout_features(self.images, 2).shape, (4, 16))
        self.assertEqual({item["key"].split("/")[0] for item in adapter.bank.sites}, {"block_0", "block_1"})


# Independent remote test execution is the supported validation entry point.
if __name__ == "__main__":
    unittest.main()
