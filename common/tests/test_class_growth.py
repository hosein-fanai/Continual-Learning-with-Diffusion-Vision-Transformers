"""Task-boundary class rebuilding preserves learned state under Keras 3."""

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.recovery import load_task_checkpoint, save_task_checkpoint
from diffusion.models.transformer.di_t_classifier import DiTClassifier
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier
from diffusion.models.wrapper.diffusion_classifier_v2 import DiffusionClassifierV2


def make_wrapper(wrapper_class=DiffusionClassifier, **kwargs):
    network = DiTClassifier(
        num_classes=None, use_cfg=True, timesteps=4, image_size=4,
        channels=1, patch_size=2, dim=4, depth=2, mha_num_heads=1,
        vit_block_mlp_ratio=1., clf_mha_num_heads=1,
        clf_vit_block_mlp_ratio=1., clf_depth=2, label_embed_trainable=True,
        cls_token_regularizer_ids=[0, 1],
        cls_token_regularizer_kwargs={"start": 0, "end": 1, "mlp_ratio": 2.},
        clf_cls_token_regularizer_ids=[0, 1],
        clf_cls_token_regularizer_kwargs={"start": 0, "end": 1, "mlp_ratio": 2.},
        clf_distil_token_type="new_weight", name="growth_network", seed=17,
    )
    model = wrapper_class(
        network=network, scheduler_name="linear", test_steps=2,
        use_ema=True, seed=17, **kwargs,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="mse")
    return model


class ClassGrowthTests(unittest.TestCase):
    def tearDown(self):
        tf.keras.backend.clear_session()

    def test_growth_preserves_weights_ema_optimizer_and_teacher(self):
        model = make_wrapper()
        model._check_new_labels(y=np.array([7, 3]), verbose=False)
        self.assertEqual(model.seen_classes, {3: 0, 7: 1})
        images = tf.reshape(tf.linspace(-1., 1., 32), (2, 4, 4, 1))
        dataset = tf.data.Dataset.from_tensor_slices((images, [3, 7])).batch(2)
        history = model.fit(dataset, epochs=1, verbose=0).history
        self.assertTrue(all(np.isfinite(value).all() for value in history.values()))
        model.network.patch_embedder.patch_projector.trainable = False

        # Distinguish existing EMA state from raw state before expanding both.
        for variable in model.ema_network.weights:
            if tf.as_dtype(variable.dtype).is_floating:
                variable.assign_add(tf.ones_like(variable) * .25)
        teacher = model.snapshot_teacher_network("raw")
        model.set_teacher_network(teacher)
        teacher_before = teacher.get_weights()
        old_raw = model.network
        old_ema = model.ema_network
        raw_before = old_raw.get_weights()
        ema_before = old_ema.get_weights()
        optimizer_before = model.optimizer
        slots_before = {
            variable.name: variable.numpy().copy()
            for variable in optimizer_before.variables
        }
        wrapper_streams = {
            name: stream.get_weights() for name, stream in model._random_streams.items()
        }

        model._check_new_labels(y=np.array([11, 7, 19]), verbose=False)
        self.assertEqual(model.seen_classes, {3: 0, 7: 1, 11: 2, 19: 3})
        self.assertIsNot(model.network, old_raw)
        self.assertIsNot(model.ema_network, old_ema)
        self.assertFalse(model.network.patch_embedder.patch_projector.trainable)
        self.assertIs(model.teacher_network, teacher)
        self.assertEqual(model.network.num_labels, 5)
        self.assertEqual(model.network.classifier.layers[-1].units, 4)
        self.assertEqual(model.network.distil_classifier.layers[-1].units, 4)
        self.assertFalse(
            {id(v) for v in old_raw.weights + old_ema.weights}
            & {id(v) for v in model.weights}
        )

        expanded_count = 0
        for old_raw_value, old_ema_value, raw, ema in zip(
            raw_before, ema_before, model.network.get_weights(), model.ema_network.get_weights()
        ):
            prefix = tuple(slice(0, size) for size in old_raw_value.shape)
            np.testing.assert_array_equal(raw[prefix], old_raw_value)
            np.testing.assert_array_equal(ema[prefix], old_ema_value)
            if raw.shape != old_raw_value.shape:
                expanded_count += 1
                new_positions = np.ones(raw.shape, dtype=bool)
                new_positions[prefix] = False
                np.testing.assert_array_equal(ema[new_positions], raw[new_positions])
        self.assertGreater(expanded_count, 8)
        # The null condition remains the first row, alongside the old classes.
        np.testing.assert_array_equal(
            model.network.label_embedder.get_weights()[0][:3],
            old_raw.label_embedder.get_weights()[0],
        )
        for actual, expected in zip(teacher.get_weights(), teacher_before):
            np.testing.assert_array_equal(actual, expected)
        for name, stream in model._random_streams.items():
            for actual, expected in zip(stream.get_weights(), wrapper_streams[name]):
                np.testing.assert_array_equal(actual, expected)

        self.assertIsNot(model.optimizer, optimizer_before)
        self.assertEqual(int(model.optimizer.iterations.numpy()), 1)
        reset_slots = 0
        for variable in model.optimizer.variables:
            previous = slots_before[variable.name]
            if tuple(variable.shape) == previous.shape:
                np.testing.assert_array_equal(variable.numpy(), previous)
            else:
                reset_slots += 1
                np.testing.assert_array_equal(variable.numpy(), np.zeros(variable.shape))
        self.assertGreater(reset_slots, 0)
        second_dataset = tf.data.Dataset.from_tensor_slices((images, [11, 19])).batch(2)
        model.fit(second_dataset, epochs=1, verbose=0)
        self.assertEqual(int(model.optimizer.iterations.numpy()), 2)
        evaluation = model.evaluate(second_dataset, verbose=0, return_dict=True)
        self.assertTrue(all(np.isfinite(value) for value in evaluation.values()))

    def test_grown_config_and_weight_file_restore(self):
        model = make_wrapper()
        model._check_new_labels(y=np.array([3, 7]), verbose=False)
        model._check_new_labels(y=np.array([11]), verbose=False)
        for variable in model.network.weights:
            if tf.as_dtype(variable.dtype).is_floating:
                variable.assign_add(tf.ones_like(variable) * .125)
        clone = type(model).from_config(model.get_config())
        clone.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="mse")
        self.assertEqual(clone.seen_classes, model.seen_classes)
        self.assertTrue(clone.network.dynamic_num_classes)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "grown.weights.h5")
            model.save_weights(path)
            clone.load_weights(path)
        self.assertEqual(len(model.weights), len(clone.weights))
        for expected, actual in zip(model.get_weights(), clone.get_weights()):
            np.testing.assert_array_equal(actual, expected)
        clone._check_new_labels(y=np.array([19]), verbose=False)
        self.assertEqual(clone.seen_classes, {3: 0, 7: 1, 11: 2, 19: 3})

    def test_failed_ema_construction_keeps_live_state(self):
        model = make_wrapper()
        model._check_new_labels(y=np.array([3, 7]), verbose=False)
        raw, ema, optimizer = model.network, model.ema_network, model.optimizer
        before = model.get_weights()
        reconstruct = DiTClassifier.from_config
        calls = 0

        def fail_second(config):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("EMA construction failed")
            return reconstruct(config)

        with patch.object(DiTClassifier, "from_config", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "EMA construction failed"):
                model._check_new_labels(y=np.array([11]), verbose=False)
        self.assertIs(model.network, raw)
        self.assertIs(model.ema_network, ema)
        self.assertIs(model.optimizer, optimizer)
        self.assertEqual(model.seen_classes, {3: 0, 7: 1})
        for expected, actual in zip(before, model.get_weights()):
            np.testing.assert_array_equal(actual, expected)

    def test_grown_task_checkpoint_restores_all_variables(self):
        model = make_wrapper()
        model._check_new_labels(y=np.array([3, 7]), verbose=False)
        teacher = model.snapshot_teacher_network("raw")
        model.set_teacher_network(teacher)
        model._check_new_labels(y=np.array([11]), verbose=False)
        model.optimizer.apply_gradients(
            (tf.ones_like(v) * .1, v) for v in model.network.trainable_variables
        )
        for role, network in enumerate((model.network, model.ema_network, teacher)):
            for index, variable in enumerate(network.variables):
                if tf.as_dtype(variable.dtype).is_floating:
                    variable.assign(tf.ones_like(variable) * (role + (index + 1) / 1000.))
        images = tf.ones((2, 4, 4, 1))
        model.noisify(images)
        trackables = {"model": model, "optimizer": model.optimizer, "teacher": teacher}
        expected_values = {
            name: [v.numpy().copy() for v in value.variables]
            for name, value in trackables.items()
        }
        clone = type(model).from_config(model.get_config())
        clone.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="mse")
        clone._register_optimizer_variables()
        clone.set_teacher_network(type(teacher).from_config(teacher.get_config()))
        restored_trackables = {
            "model": clone, "optimizer": clone.optimizer,
            "teacher": clone.teacher_network,
        }
        with tempfile.TemporaryDirectory() as directory:
            save_task_checkpoint(
                directory, 1,
                {"class_order": [3, 7, 11], "task_groups": [[3, 7], [11]],
                 "seen_classes": list(model.seen_classes.items())},
                trackables,
            )
            expected_noise = model.noisify(images)[1].numpy()
            checkpoint = load_task_checkpoint(
                directory, trackables=restored_trackables, assert_consumed=True,
            )
        self.assertEqual(checkpoint.next_task_index, 2)
        self.assertEqual(dict(checkpoint.experiment_state["seen_classes"]), clone.seen_classes)
        self.assertEqual(clone.seen_classes, model.seen_classes)
        self.assertEqual(clone.teacher_network.num_classes, 2)
        for name, value in restored_trackables.items():
            self.assertEqual(len(value.variables), len(expected_values[name]))
            for actual, expected in zip(value.variables, expected_values[name]):
                np.testing.assert_array_equal(actual.numpy(), expected)
        np.testing.assert_array_equal(clone.noisify(images)[1].numpy(), expected_noise)

    def test_v2_replaces_both_optimizer_registries(self):
        model = make_wrapper(DiffusionClassifierV2)
        model._check_new_labels(y=np.array([3, 7]), verbose=False)
        old_variable_ids = {id(variable) for variable in model.network.weights}
        for optimizer, variables in (
            (model.gen_optimizer, model.gen_trainable_variables),
            (model.clf_optimizer, model.clf_trainable_variables),
        ):
            optimizer.apply_gradients((tf.ones_like(v) * .1, v) for v in variables)
        old_generator, old_classifier = model.gen_optimizer, model.clf_optimizer
        model._check_new_labels(y=np.array([11]), verbose=False)
        self.assertIsNot(model.gen_optimizer, old_generator)
        self.assertIsNot(model.clf_optimizer, old_classifier)
        self.assertIs(model.optimizer, model.gen_optimizer)
        self.assertFalse(old_variable_ids & {id(v) for v in model.weights})
        for optimizer, variables in (
            (model.gen_optimizer, model.gen_trainable_variables),
            (model.clf_optimizer, model.clf_trainable_variables),
        ):
            self.assertEqual(int(optimizer.iterations.numpy()), 1)
            self.assertEqual(
                {id(v) for v in optimizer._trainable_variables},
                {id(v) for v in variables},
            )
            optimizer.apply_gradients((tf.ones_like(v) * .1, v) for v in variables)
            self.assertEqual(int(optimizer.iterations.numpy()), 2)


if __name__ == "__main__":
    unittest.main()
