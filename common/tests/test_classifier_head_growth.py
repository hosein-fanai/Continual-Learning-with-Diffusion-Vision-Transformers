"""Classifier expansion and opt-in optimizer prefix preservation regressions."""

import unittest
import numpy as np
import tensorflow as tf

from common.keras_compat import register_optimizer_variables
from common.model import expand_classifier_head


class ClassifierHeadGrowthTests(unittest.TestCase):
    """Check independent Keras growth and continuing optimizer trajectories."""

    def setUp(self) -> None:
        """Reset Keras naming and random state before each isolated fixture."""

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(17)

    def test_sequential_clone_preserves_nested_mask_and_old_weights(self) -> None:
        """A nested pretrained-style trunk and old columns survive growth."""

        inputs = tf.keras.Input(tuple([3]), name="backbone_input")
        hidden = tf.keras.layers.Dense(4, trainable=False, name="frozen")(inputs)
        hidden = tf.keras.layers.BatchNormalization(trainable=False, name="bn")(hidden)
        hidden = tf.keras.layers.Dense(4, name="tuned")(hidden)
        backbone = tf.keras.Model(inputs, hidden, name="backbone")
        model = tf.keras.Sequential([
            tf.keras.Input(tuple([3]), name="input"), 
            backbone, 
            tf.keras.layers.Dropout(.25, seed=13, name="dropout"), 
            tf.keras.layers.Dense(
                2, activation="softmax", kernel_initializer=tf.keras.initializers.Constant(.25), bias_initializer=tf.keras.initializers.Constant(.5), 
                dtype="float64", 
                name="head"
            )
        ], name="teacher")
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        for index, variable in enumerate(model.weights):
            variable.assign(tf.ones_like(variable) * (index + 1) / 10)
        old_weights = [value.numpy().copy() for value in model.weights]
        old_prediction = model(tf.ones((2, 3)), training=False).numpy()
        expanded = expand_classifier_head(model, 4)

        self.assertIs(type(expanded), tf.keras.Sequential)
        self.assertFalse(expanded.compiled)
        self.assertEqual(expanded.name, model.name)
        self.assertEqual(expanded.output_shape, (None, 4))
        self.assertEqual(expanded.layers[-1].dtype_policy.name, "float64")
        self.assertEqual(expanded.layers[-1].activation, model.layers[-1].activation)
        self.assertFalse(expanded.get_layer("backbone").get_layer("frozen").trainable)
        self.assertFalse(expanded.get_layer("backbone").get_layer("bn").trainable)
        self.assertTrue(expanded.get_layer("backbone").get_layer("tuned").trainable)
        for original, cloned in zip(model.layers[:-1], expanded.layers[:-1]):
            for old, new in zip(original.get_weights(), cloned.get_weights()):
                np.testing.assert_array_equal(old, new)
        for old, new in zip(model.weights, expanded.weights):
            self.assertIsNot(old, new)
        np.testing.assert_array_equal(expanded.layers[-1].kernel[:, :2], old_weights[-2])
        np.testing.assert_array_equal(expanded.layers[-1].bias[:2], old_weights[-1])
        np.testing.assert_array_equal(expanded.layers[-1].kernel[:, 2:], .25)
        np.testing.assert_array_equal(expanded.layers[-1].bias[2:], .5)
        for before, current in zip(old_weights, model.weights):
            np.testing.assert_array_equal(before, current)
        np.testing.assert_array_equal(old_prediction, model(tf.ones((2, 3)), training=False))

    def test_functional_growth_and_equal_width_noop(self) -> None:
        """Functional growth clones the graph; equal width retains compiled state."""

        inputs = tf.keras.Input(tuple([3]), name="images")
        hidden = tf.keras.layers.Dense(4, activation="relu", name="features")(inputs)
        outputs = tf.keras.layers.Dense(2, name="head")(hidden)
        model = tf.keras.Model(inputs, outputs, name="functional_teacher")
        model.compile(optimizer="adam", loss="mse")
        optimizer = model.optimizer
        self.assertIs(expand_classifier_head(model, 2), model)
        self.assertIs(model.optimizer, optimizer)
        self.assertTrue(model.compiled)
        expanded = expand_classifier_head(model, 4)
        self.assertIs(type(expanded), type(model))
        self.assertFalse(expanded.compiled)
        self.assertEqual([layer.name for layer in expanded.layers], [layer.name for layer in model.layers])
        for old, new in zip(model.layers[1].weights, expanded.layers[1].weights):
            self.assertIsNot(old, new)
            np.testing.assert_array_equal(old, new)
        for old, new in zip(model.layers[-1].weights, expanded.layers[-1].weights):
            self.assertIsNot(old, new)
            np.testing.assert_array_equal(old, new[..., :2])

    def test_rejects_unsupported_topology_before_changing_source(self) -> None:
        """Unbuilt, bias-free, LoRA, and indirect classifier heads are rejected."""

        unsupported = [
            tf.keras.Sequential([tf.keras.layers.Dense(2)]), 
            tf.keras.Sequential([
                tf.keras.Input(tuple([3])), tf.keras.layers.Dense(2, use_bias=False)
            ]), 
            tf.keras.Sequential([
                tf.keras.Input(tuple([3])), tf.keras.layers.Dense(2, lora_rank=1)
            ]), 
            tf.keras.Sequential([
                tf.keras.Input(tuple([3])), tf.keras.layers.Dense(2), 
                tf.keras.layers.Activation("softmax")
            ]), 
            tf.keras.Sequential([
                tf.keras.Input((2, 3)), tf.keras.layers.Dense(2)
            ])
        ]
        for model in unsupported:
            with self.subTest(model=model.name):
                before = [value.numpy().copy() for value in model.weights]
                with self.assertRaises(ValueError):
                    expand_classifier_head(model, 4)
                for old, current in zip(before, model.weights):
                    np.testing.assert_array_equal(old, current)

    def test_rejects_multiple_inputs_outputs_and_shrinking(self) -> None:
        """The supported graph has one input/output and an integral growing width."""

        left = tf.keras.Input(tuple([3]), name="left")
        right = tf.keras.Input(tuple([3]), name="right")
        output = tf.keras.layers.Dense(2)(tf.keras.layers.Add()([left, right]))
        with self.assertRaises(ValueError):
            expand_classifier_head(tf.keras.Model([left, right], output), 4)
        first = tf.keras.layers.Dense(2)(left)
        second = tf.keras.layers.Dense(2)(left)
        with self.assertRaises(ValueError):
            expand_classifier_head(tf.keras.Model(left, [first, second]), 4)
        model = tf.keras.Model(left, first)
        for width in (1, 2.0, True):
            with self.subTest(width=width), self.assertRaises(ValueError):
                expand_classifier_head(model, width)

    def test_adam_prefixes_keep_old_next_update_and_initialize_new_slots(self) -> None:
        """Old Adam columns continue their trajectory while new moments start fresh."""

        model = tf.keras.Sequential([
            tf.keras.Input(tuple([3])), 
            tf.keras.layers.Dense(4, name="features"), 
            tf.keras.layers.Dense(2, name="head")
        ], name="teacher")
        optimizer = tf.keras.optimizers.Adam(.01, amsgrad=True)
        gradients = [tf.ones_like(variable) * .4 for variable in model.trainable_variables]
        for _ in range(3):
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        expanded = expand_classifier_head(model, 4)
        old_state = [variable.numpy().copy() for variable in optimizer.variables]
        replacement = register_optimizer_variables(
            optimizer, expanded.trainable_variables, preserve_slot_prefixes=True
        )
        self.assertEqual(int(replacement.iterations.numpy()), 3)
        for slots in (replacement._momentums, replacement._velocities, replacement._velocity_hats):
            for slot in slots[-2:]:
                np.testing.assert_array_equal(slot[..., 2:], 0.)
        grown_gradients = [tf.ones_like(variable) * .4 for variable in expanded.trainable_variables]
        replacement.apply_gradients(zip(grown_gradients, expanded.trainable_variables))
        for old, current in zip(old_state, optimizer.variables):
            np.testing.assert_array_equal(old, current)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        for old, new in zip(model.trainable_variables, expanded.trainable_variables):
            np.testing.assert_allclose(old, new[..., :old.shape[-1]], rtol=1e-7, atol=1e-7)

    def test_unbuilt_source_optimizer_survives_independent_registration(self) -> None:
        """A growth candidate cannot consume the source optimizer's first registry."""

        for optimizer in (
            tf.keras.optimizers.Adam(.01), 
            tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(.01))
        ):
            with self.subTest(optimizer=type(optimizer).__name__):
                old = tf.Variable([1., 2.], name="head")
                expanded = tf.Variable([1., 2., 3.], name="head")
                optimizer.iterations.assign(2)
                original_state = [value.numpy().copy() for value in optimizer.variables]
                replacement = register_optimizer_variables(
                    optimizer, [expanded], preserve_slot_prefixes=True
                )
                self.assertIsNot(replacement, optimizer)
                self.assertFalse(optimizer.built)
                self.assertTrue(replacement.built)
                self.assertEqual(int(replacement.iterations.numpy()), 2)
                for before, current in zip(original_state, optimizer.variables):
                    np.testing.assert_array_equal(before, current)
                optimizer.build([old])
                owner = getattr(optimizer, "inner_optimizer", optimizer)
                self.assertIs(owner._trainable_variables[0], old)

    def test_ambiguous_relative_paths_fail_without_changing_source(self) -> None:
        """Two backbones with the same local head name cannot share guessed slots."""

        sources = [
            tf.keras.Sequential([
                tf.keras.Input(tuple([3])), tf.keras.layers.Dense(2, name="head")
            ], name=name)
            for name in ("left", "right")
        ]
        old_variables = [value for source in sources for value in source.trainable_variables]
        optimizer = tf.keras.optimizers.Adam(.01)
        optimizer.apply_gradients([(tf.ones_like(value), value) for value in old_variables])
        old_state = [value.numpy().copy() for value in optimizer.variables]
        head = tf.keras.layers.Dense(4, name="head")
        head(tf.ones((1, 3)))
        with self.assertRaisesRegex(ValueError, "Ambiguous optimizer variable path"):
            register_optimizer_variables(
                optimizer, head.trainable_variables, preserve_slot_prefixes=True
            )
        for before, current in zip(old_state, optimizer.variables):
            np.testing.assert_array_equal(before, current)

    def test_default_resize_keeps_fresh_slots(self) -> None:
        """Callers that do not opt in retain the previous reset-on-resize behavior."""

        old = tf.Variable([1., 2.], name="head")
        optimizer = tf.keras.optimizers.Adam(.01)
        optimizer.apply_gradients([(tf.ones_like(old), old)])
        expanded = tf.Variable([1., 2., 3.], name="head")
        replacement = register_optimizer_variables(optimizer, [expanded])
        self.assertEqual(int(replacement.iterations.numpy()), 1)
        np.testing.assert_array_equal(replacement._momentums[0], 0.)
        np.testing.assert_array_equal(replacement._velocities[0], 0.)

    def test_loss_scale_growth_preserves_inner_prefixes_and_counters(self) -> None:
        """Loss-scaling counters and Adam moments survive an expanded variable."""

        old = tf.Variable([1., 2.], name="head")
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            tf.keras.optimizers.Adam(.01), initial_scale=128.
        )
        optimizer.apply_gradients([(tf.ones_like(old) * 128., old)])
        expanded = tf.Variable([1., 2., 3.], name="head")
        replacement = register_optimizer_variables(
            optimizer, [expanded], preserve_slot_prefixes=True
        )
        self.assertEqual(int(replacement.iterations.numpy()), 1)
        self.assertEqual(float(replacement.dynamic_scale.numpy()), 128.)
        self.assertEqual(int(replacement.step_counter.numpy()), 1)
        for old_slots, new_slots in (
            (optimizer.inner_optimizer._momentums, replacement.inner_optimizer._momentums), 
            (optimizer.inner_optimizer._velocities, replacement.inner_optimizer._velocities)
        ):
            np.testing.assert_array_equal(old_slots[0], new_slots[0][:2])
            np.testing.assert_array_equal(new_slots[0][2:], 0.)


# Run the focused growth checks when invoked directly.
if __name__ == "__main__":
    unittest.main()
