"""Exercise factory compilation with the installed Keras default and XLA."""

import unittest
import numpy as np
import tensorflow as tf

from common.model import _get_classifier_model, get_compile_args


class JitDefaultsTests(unittest.TestCase):
    def test_factory_default_and_explicit_xla_training(self):
        """The default metrics train correctly and caller JIT overrides survive."""
        for override in ({}, {"jit_compile": True}, {"jit_compile": False}):
            with self.subTest(override=override):
                tf.keras.backend.clear_session()
                model = _get_classifier_model(
                    class_num=2, model_type="dnn", verbose=0, seed=13,
                    architecture_kwargs={"input_shape": (4,)},
                    compile_args={**get_compile_args(), **override},
                )
                reference = tf.keras.Sequential([
                    tf.keras.layers.Input((4,)), tf.keras.layers.Dense(2),
                ])
                reference.compile(**override)
                self.assertEqual(model.jit_compile, reference.jit_compile)
                x = np.arange(32, dtype=np.float32).reshape(8, 4) / 32
                y = np.arange(8) % 2
                before = [weight.numpy().copy() for weight in model.trainable_weights]
                result = model.train_on_batch(x, y, return_dict=True)
                self.assertTrue(all(np.isfinite(value) for value in result.values()))
                self.assertTrue(any(
                    not np.array_equal(old, new.numpy())
                    for old, new in zip(before, model.trainable_weights)
                ))
                result = model.test_on_batch(x[:3], y[:3], return_dict=True)
                self.assertTrue(all(np.isfinite(value) for value in result.values()))


if __name__ == "__main__":
    unittest.main()
