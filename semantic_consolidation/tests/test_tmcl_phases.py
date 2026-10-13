"""TMCL phase regressions for current-data and gradient ownership contracts.

The lightweight adapter fixture isolates loss routing and stopped-target behavior.
A separate tiny native DiT test covers compatibility with real scoped injections.
These checks make no benchmark-performance or remote-capacity claim.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import Config
from common.gradients import apply_policy_gradients
from common.model import get_model
from common.runtime import derive_seed
from semantic_consolidation.augmentation import acquisition_augmentation, consolidation_views
from semantic_consolidation.tmcl_adapter import LayerModulationBank, TMCLDiTAdapter
from semantic_consolidation.tmcl_config import TMCLSettings
from semantic_consolidation.tmcl_phases import (
    TMCLBatchNormalization, TMCLPhase, build_tmcl_heads, modulation_decay_vector
)


def _views(images: tf.Tensor, **kwargs: object) -> tuple[tf.Tensor, ...]:
    """Supply deterministic distinct views while isolating phase routing tests."""

    return tuple(tf.roll(images, shift=index, axis=2) for index in range(kwargs["num_views"]))


def _values(variables: list[tf.Variable]) -> list[np.ndarray]:
    """Copy variable values without assuming names are globally unique."""

    return [variable.numpy().copy() for variable in variables]


def make_rgb_wrapper(num_classes: int = 2) -> tf.keras.Model:
    """Construct a tiny RGB DiT wrapper through the existing Config/model factory."""

    config = Config()
    config.training.task = "joint"
    config.training.seed = 41
    config.training.dtype_policy = "float32"
    config.training.deterministic_ops = True
    config.dataset.trainset_len = 2
    config.model.name = "dit_classifier"
    config.model.wrapper_name = "diffusion_classifier"
    config.model.show_network_summary = False
    config.model.kwargs = {
        "num_classes": num_classes, "use_cfg": True, "timesteps": 4, 
        "image_size": 4, "channels": 3, "patch_size": 2, 
        "dim": 8, "cond_dim": 8, "depth": 1, "mha_num_heads": 1, 
        "vit_block_mlp_ratio": 1., "clf_depth": 0, "clf_vit_block_ids": [], 
        "clf_cls_token_type": None, "feature_aggregation_ids_dict": {1: [1]}, 
        "force_global_avg_pooling": True, "classifier_mlp_ratio": 1, 
        "classifier_dropout_rate": 0., "ln_no_adaptation": True, 
        "clf_ln_no_adaptation": True, "build": True
    }
    config.model.wrapper_kwargs = {
        "use_ema": False, "test_network_name": "raw", "test_steps": 2, 
        "modify_first_t": False, "test_noisified_min_timesteps": 0, 
        "test_noisified_max_timesteps": 0
    }
    return get_model(config)


class _Adapter:
    """Small differentiable native-contract fixture with a target-only variable."""

    def __init__(self) -> None:
        """Initialize deterministic feature weights and four independent gate pairs."""

        self.wrapper = object()
        self.network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(4, 4, 3)), 
            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(
                6, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=17), 
                dtype="float32"
            )
        ])
        self.target_only = tf.Variable(np.linspace(-0.1, 0.1, 6).astype("float32"), name="target_only")
        self.bank = LayerModulationBank([{"key": "block_0/features", "shape": [6]}], seed=19)
        self.bank.add([0, 1, 2, 3])
        self.calls: list[dict[str, object]] = []

    @property
    def backbone_variables(self) -> list[tf.Variable]:
        """Include a variable connected exclusively through stopped MI targets."""

        return list(self.network.trainable_variables) + [self.target_only]

    def features(self, images: tf.Tensor, class_ids: tf.Tensor | None = None, 
                 training: bool = False) -> tf.Tensor:
        """Expose separate unmodulated and gated differentiable feature routes."""

        self.calls.append({
            "training": training, "rows": len(images), 
            "classes": None if class_ids is None else class_ids.numpy().copy()
        })
        features = self.network(images, training=training)
        # Only modulated branches use this independent target-only feature variable.
        if class_ids is not None:
            features = self.bank.apply(features + self.target_only, 0, class_ids)
        return features


class _CycleSampler:
    """Cycle every population deterministically for exact old-gate coverage tests."""

    def choice(self, population: object, size: int, replace: bool) -> np.ndarray:
        """Select only supplied population rows while covering every gate ID."""

        population = np.arange(population) if isinstance(population, int) else np.asarray(population)
        return np.resize(population, size)


class TMCLPhaseTests(unittest.TestCase):
    """Verify eager phase updates, reused heads and labelled-data isolation."""

    def setUp(self) -> None:
        """Create small fixed current images and independently initialized native-contract state."""

        self.policy = tf.keras.mixed_precision.global_policy().name
        tf.keras.mixed_precision.set_global_policy("float32")
        self.adapter = _Adapter()
        self.images = np.random.default_rng(23).uniform(-1., 1., (12, 4, 4, 3)).astype("float32")
        self.labels = np.repeat([2, 3], 6).astype("int32")
        self.settings = TMCLSettings(
            batch_size=8, augmentation_views=3, projector_hidden=12, 
            projector_dim=6, predictor_hidden=8
        )
        self.augmentation = patch("semantic_consolidation.tmcl_phases.consolidation_views", side_effect=_views)
        self.augmentation.start()

    def tearDown(self) -> None:
        """Restore global framework policy and the real image augmentation helper."""

        self.augmentation.stop()
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy(self.policy)

    def _phase(self, phase: str, **kwargs: object) -> TMCLPhase:
        """Compile one actual custom train_step with a deterministic optimizer."""

        defaults = {
            "adapter": self.adapter, "images": self.images, 
            "labels": self.labels, "classes": [2, 3] if phase == "acquisition" else [0, 1, 2, 3], 
            "settings": self.settings, "phase": phase, "seed": 29
        }
        defaults.update(kwargs)
        result = TMCLPhase(**defaults)
        result.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.03), run_eagerly=True)
        return result

    def test_acquisition_updates_only_the_selected_new_gate(self) -> None:
        """OPL differentiates through frozen feedforward state into one new modulation."""

        phase = self._phase("acquisition")
        backbone_before = _values(self.adapter.backbone_variables)
        gates_before = _values(self.adapter.bank.variables_for())
        phase.train_step(None)
        focus = phase.trace[0]["focus_class"]
        for before, variable in zip(backbone_before, self.adapter.backbone_variables):
            np.testing.assert_array_equal(variable.numpy(), before)
        for index, class_id in enumerate(self.adapter.bank.classes):
            current = self.adapter.bank.variables_for([class_id])
            # Only the active new-class task is eligible to change in this update.
            if class_id != focus:
                for before, variable in zip(gates_before[2 * index:2 * index + 2], current):
                    np.testing.assert_array_equal(variable.numpy(), before)
            # A nontrivial OPL task must connect at least one gain/bias parameter.
            else:
                self.assertTrue(any(not np.array_equal(before, variable.numpy())
                                    for before, variable in zip(gates_before[2 * index:2 * index + 2], current)))
        self.assertTrue(all(not call["training"] for call in self.adapter.calls))
        self.assertEqual(phase.trace[0]["vi"], 0.)
        self.assertEqual(phase.trace[0]["mi"], 0.)

    def test_current_only_images_can_sample_old_and_new_gates(self) -> None:
        """Consolidation ignores labels and does not require historical class images."""

        phase = self._phase("consolidation", labels=None)
        phase.rng = _CycleSampler()
        gates_before = _values(self.adapter.bank.variables_for())
        target_before = self.adapter.target_only.numpy().copy()
        network_before = _values(self.adapter.network.trainable_variables)
        self.adapter.calls.clear()
        metrics = phase.train_step(None)
        for before, variable in zip(gates_before, self.adapter.bank.variables_for()):
            np.testing.assert_array_equal(variable.numpy(), before)
        np.testing.assert_array_equal(self.adapter.target_only.numpy(), target_before)
        self.assertTrue(any(not np.array_equal(before, variable.numpy())
                            for before, variable in zip(network_before, self.adapter.network.trainable_variables)))
        self.assertEqual([call["rows"] for call in self.adapter.calls], [24, 8, 16])
        self.assertTrue(all(call["training"] for call in self.adapter.calls))
        self.assertEqual(set(self.adapter.calls[-1]["classes"]), {0, 1, 2, 3})
        self.assertEqual(sum(phase.trace[0]["modulation_draw_counts"].values()), 16)
        self.assertTrue(all(np.isfinite(float(value)) for value in metrics.values()))
        self.assertEqual((phase.step_number, phase.example_draws, phase.view_draws), (1, 8, 24))

    def test_acquisition_applies_depth_decay_only_to_selected_gate(self) -> None:
        """A zero-gradient objective exposes exact decoupled selected-gate decay."""

        phase = self._phase("acquisition")
        before = _values(self.adapter.bank.variables_for())
        backbone = _values(self.adapter.backbone_variables)

        def zero_loss(positive_features: tf.Tensor, negative_features: tf.Tensor) -> tf.Tensor:
            """Keep a connected zero derivative without contributing an optimizer step."""

            return tf.reduce_sum(positive_features * 0.) + tf.reduce_sum(negative_features * 0.)

        with patch("semantic_consolidation.tmcl_phases.orthogonal_projection_loss", side_effect=zero_loss):
            phase.train_step(None)
        focus = phase.trace[0]["focus_class"]
        for index, class_id in enumerate(self.adapter.bank.classes):
            decay = 0.03 * self.settings.modulation_decay_end if class_id == focus else 0.
            for old, variable in zip(before[2 * index:2 * index + 2], self.adapter.bank.variables_for([class_id])):
                reference = old - decay * old
                np.testing.assert_allclose(variable.numpy(), reference, rtol=1e-7, atol=1e-9)
        for old, variable in zip(backbone, self.adapter.backbone_variables):
            np.testing.assert_array_equal(variable.numpy(), old)

    def test_depth_decay_mask_uses_blocks_not_individual_sites(self) -> None:
        """Every projection in one block shares the same cosine schedule position."""

        bank = LayerModulationBank([
            {"key": "block_0/query", "shape": [2]}, 
            {"key": "block_0/mlp", "shape": [1]}, 
            {"key": "block_1/query", "shape": [2]}, 
            {"key": "block_2/mlp", "shape": [1]}
        ])
        mask = modulation_decay_vector(bank, 0.4, 0.04).numpy()
        np.testing.assert_allclose(mask, [0.4, 0.4, 0.4, 0.22, 0.22, 0.04], rtol=1e-7)

    def test_each_gate_optimizer_counts_only_its_own_focus_updates(self) -> None:
        """Alternating acquisition tasks retain independent Adam bias-correction steps."""

        phase = self._phase("acquisition")
        phase.gate_optimizers = {
            class_id: tf.keras.optimizers.AdamW(learning_rate=0.01, weight_decay=0.)
            for class_id in (2, 3)
        }
        batches = [(tf.convert_to_tensor(self.images[:8]), focus, 4) for focus in (2, 3, 2)]
        with patch.object(phase, "_draw", side_effect=batches):
            for _ in range(3):
                phase.train_step(None)
        self.assertEqual(int(phase.gate_optimizers[2].iterations.numpy()), 2)
        self.assertEqual(int(phase.gate_optimizers[3].iterations.numpy()), 1)
        self.assertEqual(int(phase.optimizer.iterations.numpy()), 0)

    def test_mi_first_update_matches_same_forward_gradient_oracle(self) -> None:
        """Delayed MI Adam starts at one and uses gradients from the pre-update forward."""

        pretrain = self._phase("pretrain", labels=None)
        representation_optimizer = tf.keras.optimizers.AdamW(learning_rate=0.01, weight_decay=0.0001)
        pretrain.compile(optimizer=representation_optimizer, run_eagerly=True)
        for _ in range(3):
            pretrain.train_step(None)
        phase = self._phase("consolidation", labels=None, heads=pretrain.heads)
        phase.compile(optimizer=representation_optimizer, run_eagerly=True)
        phase.mi_optimizer = tf.keras.optimizers.AdamW(learning_rate=0.01, weight_decay=0.0001)
        mi_variables = phase._mi_variables()
        oracle_variables = [tf.Variable(variable.numpy()) for variable in mi_variables]
        oracle_optimizer = tf.keras.optimizers.AdamW(learning_rate=0.01, weight_decay=0.0001)
        captured: list[list[tf.Tensor]] = []

        def checked_update(tape: tf.GradientTape, optimizer: tf.keras.optimizers.Optimizer, 
                           loss: tf.Tensor, variables: list[tf.Variable]) -> list[tuple[tf.Tensor, tf.Variable]]:
            """Compare post-primary differentiation against gradients captured before any update."""

            # Capture MI derivatives while every trainable still has its forward value.
            if optimizer is representation_optimizer:
                captured.append(tape.gradient(loss, mi_variables))
            # The persistent tape must retain exactly those same recorded tensor values.
            else:
                actual = tape.gradient(loss, mi_variables)
                for before, after in zip(captured[0], actual):
                    np.testing.assert_array_equal(after.numpy(), before.numpy())
            return apply_policy_gradients(tape, optimizer, loss, variables)

        with patch("semantic_consolidation.tmcl_phases.apply_policy_gradients", side_effect=checked_update):
            phase.train_step(None)
        oracle_optimizer.apply_gradients(list(zip(captured[0], oracle_variables)))
        self.assertEqual(int(representation_optimizer.iterations.numpy()), 4)
        self.assertEqual(int(phase.mi_optimizer.iterations.numpy()), 1)
        for expected, actual in zip(oracle_variables, mi_variables):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_projector_bias_and_native_linear_initializer_match_reference_structure(self) -> None:
        """Lightly omits only pre-BN biases and initializes Linear using fan-in bounds."""

        heads = build_tmcl_heads(6, self.settings, seed=53)
        for key in ("vi", "mi"):
            linears = [layer for layer in heads[key].layers if isinstance(layer, tf.keras.layers.Dense)]
            self.assertEqual([layer.use_bias for layer in linears], [False, False, True])
            for layer in linears:
                bound = 1. / np.sqrt(int(layer.kernel.shape[0]))
                self.assertLessEqual(float(np.max(np.abs(layer.kernel.numpy()))), bound)
                # Only the final projector linear owns the independently initialized bias.
                if layer.use_bias:
                    self.assertLessEqual(float(np.max(np.abs(layer.bias.numpy()))), bound)
                    self.assertTrue(np.any(layer.bias.numpy() != 0.))

    def test_batch_norm_uses_population_forward_and_unbiased_running_variance(self) -> None:
        """PyTorch-compatible EMA state remains exact under native serialization."""

        layer = TMCLBatchNormalization(momentum=0.9, epsilon=1e-5, dtype="float32")
        values = np.asarray([[1., 2.], [3., 7.], [5., 3.], [9., 8.]], dtype="float32")
        output = layer(tf.convert_to_tensor(values), training=True).numpy()
        expected = (values - values.mean(axis=0)) / np.sqrt(values.var(axis=0) + 1e-5)
        np.testing.assert_allclose(output, expected, rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(layer.moving_mean.numpy(), 0.1 * values.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(layer.moving_variance.numpy(), 0.9 + 0.1 * values.var(axis=0, ddof=1), rtol=1e-6)
        clone = tf.keras.layers.deserialize(tf.keras.layers.serialize(layer))
        clone(tf.convert_to_tensor(values), training=False)
        clone.set_weights(layer.get_weights())
        before = _values(clone.weights)
        np.testing.assert_array_equal(clone(values, training=False).numpy(), layer(values, training=False).numpy())
        for expected_state, actual in zip(before, clone.weights):
            np.testing.assert_array_equal(actual.numpy(), expected_state)

    def test_si_uses_frozen_previous_state_and_independent_first_use_optimizer(self) -> None:
        """SI updates its predictor while all previous-network/projector state stays exact."""

        settings = replace(self.settings, si_weight=1.)
        past_adapter = _Adapter()
        past_adapter.network.trainable = False
        past_projector = build_tmcl_heads(6, settings, seed=61)["vi"]
        past_projector.trainable = False
        previous = {"adapter": past_adapter, "projector": past_projector}
        phase = self._phase("consolidation", settings=settings, previous=previous)
        phase.mi_optimizer = tf.keras.optimizers.AdamW(learning_rate=0.01, weight_decay=0.)
        phase.si_optimizer = tf.keras.optimizers.AdamW(learning_rate=0.01, weight_decay=0.)
        frozen_before = _values(past_adapter.network.weights + past_projector.weights)
        predictor_before = _values(phase.heads["si_predictor"].trainable_variables)
        past_adapter.calls.clear()
        phase.train_step(None)
        for before, variable in zip(frozen_before, past_adapter.network.weights + past_projector.weights):
            np.testing.assert_array_equal(variable.numpy(), before)
        self.assertTrue(any(not np.array_equal(before, variable.numpy())
                            for before, variable in zip(predictor_before, phase.heads["si_predictor"].trainable_variables)))
        self.assertEqual(int(phase.si_optimizer.iterations.numpy()), 1)
        self.assertEqual(int(phase.mi_optimizer.iterations.numpy()), 1)
        self.assertTrue(np.isfinite(phase.trace[0]["si"]))
        self.assertNotEqual(phase.trace[0]["si"], 0.)
        self.assertTrue(all(not call["training"] and call["classes"] is None for call in past_adapter.calls))

    def test_first_session_does_not_update_si_predictor_without_previous_state(self) -> None:
        """Enabling future SI cannot invent a teacher or update its unused predictor."""

        phase = self._phase("consolidation", settings=replace(self.settings, si_weight=1.))
        phase.si_optimizer = tf.keras.optimizers.AdamW(learning_rate=0.01, weight_decay=0.)
        before = _values(phase.heads["si_predictor"].weights)
        phase.train_step(None)
        self.assertEqual(phase.trace[0]["si"], 0.)
        self.assertEqual(int(phase.si_optimizer.iterations.numpy()), 0)
        for old, variable in zip(before, phase.heads["si_predictor"].weights):
            np.testing.assert_array_equal(variable.numpy(), old)

    def test_pretrain_is_vi_only_and_preserves_mi_state(self) -> None:
        """Initial VI never invokes a modulation or changes either MI head."""

        phase = self._phase("pretrain", labels=None)
        mi_before = _values(phase.heads["mi"].weights + phase.heads["predictor"].weights)
        gates_before = _values(self.adapter.bank.variables_for())
        self.adapter.calls.clear()
        phase.train_step(None)
        for before, variable in zip(mi_before, phase.heads["mi"].weights + phase.heads["predictor"].weights):
            np.testing.assert_array_equal(variable.numpy(), before)
        for before, variable in zip(gates_before, self.adapter.bank.variables_for()):
            np.testing.assert_array_equal(variable.numpy(), before)
        self.assertEqual(len(self.adapter.calls), 1)
        self.assertIsNone(self.adapter.calls[0]["classes"])
        self.assertEqual(phase.trace[0]["mi"], 0.)
        self.assertEqual(phase.trace[0]["opl"], 0.)

    def test_mi_disabled_preserves_both_mi_models(self) -> None:
        """The matched VI baseline does not update inactive MI BN or parameters."""

        phase = self._phase("consolidation", settings=replace(self.settings, mi_weight=0.))
        before = _values(phase.heads["mi"].weights + phase.heads["predictor"].weights)
        self.adapter.calls.clear()
        phase.train_step(None)
        for value, variable in zip(before, phase.heads["mi"].weights + phase.heads["predictor"].weights):
            np.testing.assert_array_equal(variable.numpy(), value)
        self.assertEqual(len(self.adapter.calls), 1)
        self.assertIsNone(self.adapter.calls[0]["classes"])

    def test_heads_are_reused_without_initialization_or_bn_mutation(self) -> None:
        """Initial pretraining heads can pass into the same session unchanged."""

        heads = build_tmcl_heads(6, self.settings, seed=31)
        before = {key: _values(value.weights) for key, value in heads.items()}
        phase = self._phase("consolidation", heads=heads)
        for key, head in heads.items():
            self.assertIs(phase.heads[key], head)
            for value, variable in zip(before[key], head.weights):
                np.testing.assert_array_equal(variable.numpy(), value)

    def test_acquisition_requires_current_positive_and_negative_rows(self) -> None:
        """Old gates cannot manufacture absent supervised positive images."""

        with self.assertRaisesRegex(ValueError, "positive and negative"):
            self._phase("acquisition", classes=[0])
        with self.assertRaisesRegex(ValueError, "integer class IDs"):
            self._phase("acquisition", labels=self.labels.astype("float32"))

    def test_bernoulli_acquisition_resamples_only_undefined_opl_batches(self) -> None:
        """Reported rejection counts distinguish structural conditioning from fixed halves."""

        phase = self._phase("acquisition")

        class FixedSampler:
            """Supply invalid counts followed by a valid unbalanced OPL draw."""

            def __init__(self) -> None:
                """Set a deterministic sequence with three structural rejections."""

                self.counts = iter([0, 1, 8, 2])

            def binomial(self, count: int, probability: float) -> int:
                """Return the next prescribed count for the Bernoulli mixture."""

                return next(self.counts)

            def choice(self, population: object, size: int | None = None, 
                       replace: bool = True) -> object:
                """Select only explicit source rows while retaining requested multiplicity."""

                return population[0] if size is None else np.resize(population, size)

        phase.rng = FixedSampler()
        images, focus, positives = phase._draw()
        self.assertEqual((focus, positives, len(images)), (2, 2, 8))
        self.assertEqual(phase.rejected_acquisition_batches, 3)
        self.assertNotEqual(positives, self.settings.batch_size // 2)
        for image in images[:positives].numpy():
            self.assertTrue(any(np.array_equal(image, row) for row in self.images[self.labels == focus]))
        for image in images[positives:].numpy():
            self.assertTrue(any(np.array_equal(image, row) for row in self.images[self.labels != focus]))

    def test_three_row_acquisition_is_the_minimal_valid_bernoulli_batch(self) -> None:
        """The structural OPL limit is two positives plus one negative, not a fixed half."""

        phase = self._phase("acquisition", settings=replace(self.settings, batch_size=3))
        images, focus, positive_count = phase._draw()
        self.assertEqual((len(images), positive_count), (3, 2))
        self.assertIn(focus, [2, 3])

    def test_native_dit_vi_mi_update_preserves_primary_classifier(self) -> None:
        """The real DiT adapter supports clean VI/MI without training its class head."""

        wrapper = make_rgb_wrapper()
        adapter = TMCLDiTAdapter(wrapper, seed=43)
        adapter.bank.add([0, 1])
        phase = self._phase("consolidation", adapter=adapter, labels=None, classes=[0, 1])
        before = _values(wrapper.network.classifier.weights)
        gates = _values(adapter.bank.variables_for())
        phase.train_step(None)
        for value, variable in zip(before, wrapper.network.classifier.weights):
            np.testing.assert_array_equal(variable.numpy(), value)
        for value, variable in zip(gates, adapter.bank.variables_for()):
            np.testing.assert_array_equal(variable.numpy(), value)
        self.assertTrue(phase.updated_names)


class TMCLAugmentationGraphTests(unittest.TestCase):
    """Keep graph execution limited to the native stateless image transforms."""

    def setUp(self) -> None:
        """Bind real wrapper scaling to a small eager-only feature fixture."""

        self.policy = tf.keras.mixed_precision.global_policy().name
        tf.keras.mixed_precision.set_global_policy("float32")
        self.wrapper = make_rgb_wrapper()
        self.adapter = _Adapter()
        self.adapter.wrapper = self.wrapper
        self.images = np.random.default_rng(23).uniform(-1., 1., (12, 4, 4, 3)).astype("float32")
        self.settings = TMCLSettings(
            batch_size=8, augmentation_views=4, projector_hidden=12, 
            projector_dim=6, predictor_hidden=8
        )
        self.phase = TMCLPhase(
            self.adapter, self.images, labels=None, classes=[0, 1, 2, 3], 
            settings=self.settings, phase="consolidation", seed=29
        )
        self.phase.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.03), run_eagerly=True)

    def tearDown(self) -> None:
        """Release isolated native models and restore the incoming dtype policy."""

        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy(self.policy)

    def test_cached_views_match_eager_seeds_and_batches_without_state_changes(self) -> None:
        """Four real views preserve streams and model state through one cached trace."""

        variables = (list(self.wrapper.network.weights) + list(self.adapter.network.weights)
                     + [self.adapter.target_only] + list(self.adapter.bank.variables_for())
                     + [variable for head in self.phase.heads.values() for variable in head.weights]
                     + list(self.phase.optimizer.variables))
        before = _values(variables)
        sampler_before = deepcopy(self.phase.rng.bit_generator.state)
        calls_before = deepcopy(self.adapter.calls)
        tf.random.set_seed(101)
        expected_random = tf.random.uniform(tuple([8])).numpy()
        tf.random.set_seed(101)
        observed: list[tuple[np.ndarray, ...]] = []
        for count, seed in ((3, 34), (7, 35), (3, 34)):
            images = tf.convert_to_tensor(self.images[:count])
            actual = self.phase._augmentation(images, tf.constant(seed, dtype=tf.int64))
            expected = consolidation_views(
                images, wrapper=self.wrapper, num_views=4, image_size=None, seed=seed
            )
            self.assertEqual(len(actual), 4)
            for reference, value in zip(expected, actual):
                np.testing.assert_allclose(value.numpy(), reference.numpy(), atol=2e-6)
                self.assertEqual(tuple(value.shape), (count, 4, 4, 3))
                self.assertEqual(value.dtype, tf.float32)
            observed.append(tuple(value.numpy().copy() for value in actual))
        for first, repeated in zip(observed[0], observed[2]):
            np.testing.assert_array_equal(first, repeated)
        self.assertTrue(any(not np.array_equal(first, changed[:3])
                            for first, changed in zip(observed[0], observed[1])))
        self.assertEqual(self.phase._augmentation.experimental_get_tracing_count(), 1)
        for expected_state, variable in zip(before, variables):
            np.testing.assert_array_equal(variable.numpy(), expected_state)
        self.assertEqual(self.phase.rng.bit_generator.state, sampler_before)
        self.assertEqual(self.adapter.calls, calls_before)
        self.assertEqual((self.phase.step_number, self.phase.example_draws, self.phase.view_draws), (0, 0, 0))
        self.assertEqual(self.phase.trace, [])
        np.testing.assert_array_equal(tf.random.uniform(tuple([8])).numpy(), expected_random)

    def test_phase_forwards_exact_step_seeds_while_feature_calls_stay_eager(self) -> None:
        """Compilation leaves the existing step-derived streams and feature hooks intact."""

        cached = self.phase._augmentation
        calls: list[dict[str, object]] = []

        def checked_augmentation(images: tf.Tensor, seed: tf.Tensor) -> tuple[tf.Tensor, ...]:
            """Check each runtime tensor seed against the unchanged eager image policy."""

            self.assertTrue(tf.executing_eagerly())
            self.assertEqual(seed.dtype, tf.int64)
            self.assertEqual(seed.shape.rank, 0)
            actual = cached(images, seed)
            expected = consolidation_views(
                images, wrapper=self.wrapper, num_views=4, image_size=None, seed=seed
            )
            for reference, value in zip(expected, actual):
                np.testing.assert_allclose(value.numpy(), reference.numpy(), atol=2e-6)
            calls.append({"seed": int(seed.numpy()), "rows": len(images)})
            return actual

        self.adapter.calls.clear()
        with patch.object(self.phase, "_augmentation", side_effect=checked_augmentation):
            for _ in range(2):
                self.phase.train_step(None)
        expected_seeds = [derive_seed(29, step, "augmentation") for step in range(2)]
        self.assertEqual([call["seed"] for call in calls], expected_seeds)
        self.assertEqual([row["augmentation_seed"] for row in self.phase.trace], expected_seeds)
        self.assertEqual([call["rows"] for call in calls], [8, 8])
        self.assertEqual([call["rows"] for call in self.adapter.calls], [32, 8, 24, 32, 8, 24])
        self.assertEqual(cached.experimental_get_tracing_count(), 1)

    def test_acquisition_graph_preserves_flip_policy_for_dynamic_batches(self) -> None:
        """The cached acquisition branch remains a single exact flip-only view."""

        phase = TMCLPhase(
            self.adapter, self.images, labels=np.repeat([2, 3], 6), classes=[2, 3], 
            settings=self.settings, phase="acquisition", seed=29
        )
        for count, seed in ((3, 71), (8, 72), (3, 71)):
            images = tf.convert_to_tensor(self.images[:count])
            actual = phase._augmentation(images, tf.constant(seed, dtype=tf.int64))
            self.assertEqual(len(actual), 1)
            expected = acquisition_augmentation(images, seed=seed)
            np.testing.assert_array_equal(actual[0].numpy(), expected.numpy())
        self.assertEqual(phase._augmentation.experimental_get_tracing_count(), 1)


# Direct invocation runs the same bounded unittest cases as discovery.
if __name__ == "__main__":
    unittest.main()
