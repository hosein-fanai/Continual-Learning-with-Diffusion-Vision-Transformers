"""Exercise every supported real application backbone without downloading weights.

These are architecture/factory tests, not tests of ImageNet parameter quality.
Each actual Keras application is initialized with weights=None, then the project
factory attaches its real preprocessing and classifier. Four fine-tuning policies
are checked; frozen and three-layer-tail models execute real head updates.
"""

from __future__ import annotations

import gc
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.model import _PRETRAINED_CONV_BASES, get_model
from common.runtime import configure_runtime


BACKBONES = (
    'Xception', 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 
    'EfficientNetV2B3', 'EfficientNetV2S', 'EfficientNetV2M', 'EfficientNetV2L'
)


class BackboneArchitectureMatrixTests(unittest.TestCase):
    """Verify eight actual application graphs under four factory freezing policies."""

    def test_real_backbones_freezing_and_head_updates(self) -> None:
        """Check 32 policy cases and 16 updates, preserving frozen kernels and BN state."""

        self.assertEqual(set(BACKBONES), set(_PRETRAINED_CONV_BASES.values()))
        self.addCleanup(tf.keras.backend.clear_session)
        cases = 0
        updates = 0
        for name in BACKBONES:
            tf.keras.backend.clear_session()
            gc.collect()
            configure_runtime(817, 'float32')
            size = 71 if name == 'Xception' else 32
            constructor = getattr(tf.keras.applications, name)
            options = {} if name == 'Xception' else {'include_preprocessing': True}
            base = constructor(include_top=False, weights=None, 
                               input_shape=(size, size, 3), **options)
            images = tf.reshape(tf.linspace(31.0, 223.0, 2 * 32 * 32 * 3), (2, 32, 32, 3))
            labels = tf.constant([0, 0], tf.int32)
            bn_layers = [layer for layer in base.layers
                         if isinstance(layer, tf.keras.layers.BatchNormalization)]
            early_kernel = next(layer.kernel for layer in base.layers if hasattr(layer, 'kernel'))
            for tail in (0, 3, None, len(base.layers) + 5):
                with self.subTest(backbone=name, trainable_tail=tail):
                    # Reuse the actual application graph; replace only its weight-loading request.
                    with patch.object(tf.keras.applications, name, return_value=base) as factory:
                        model = get_model(
                            3, model_type='pretrained', conv_base_name=name, 
                            resize=(size, size), num_last_not_frozen=tail, 
                            dropout_rate=0.0, verbose=0, 
                            compile_args={'optimizer': tf.keras.optimizers.SGD(0.05), 
                                          'run_eagerly': True, 'jit_compile': False}
                        )
                    self.assertEqual(factory.call_args.kwargs['weights'], 'imagenet')
                    cutoff = 0 if tail is None else max(0, len(base.layers) - tail)
                    for index, layer in enumerate(base.layers):
                        expected = index >= cutoff and not isinstance(layer, tf.keras.layers.BatchNormalization)
                        self.assertEqual(bool(layer.trainable), expected, (name, tail, layer.name))
                    probabilities = model(images, training=False)
                    self.assertEqual(tuple(probabilities.shape), (2, 3))
                    np.testing.assert_allclose(tf.reduce_sum(probabilities, axis=-1), [1.0, 1.0], 
                                               rtol=1e-6, atol=1e-6)
                    self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(probabilities))))
                    # Exercise real updates for the frozen-head and bounded fine-tuning routes.
                    if tail in (0, 3):
                        old_kernel = early_kernel.numpy().copy()
                        old_head_bias = model.layers[-1].bias.numpy().copy()
                        old_bn = [(layer.moving_mean.numpy().copy(), layer.moving_variance.numpy().copy())
                                  for layer in bn_layers]
                        with tf.GradientTape() as tape:
                            trained = model(images, training=True)
                            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, trained))
                        gradients = tape.gradient(loss, model.trainable_variables)
                        self.assertTrue(all(gradient is not None for gradient in gradients))
                        self.assertTrue(all(bool(tf.reduce_all(tf.math.is_finite(gradient))) for gradient in gradients))
                        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                        self.assertEqual(int(model.optimizer.iterations), 1)
                        self.assertFalse(np.array_equal(old_head_bias, model.layers[-1].bias.numpy()))
                        np.testing.assert_array_equal(early_kernel.numpy(), old_kernel)
                        for layer, (mean, variance) in zip(bn_layers, old_bn):
                            np.testing.assert_array_equal(layer.moving_mean.numpy(), mean)
                            np.testing.assert_array_equal(layer.moving_variance.numpy(), variance)
                        updates += 1
                    cases += 1
                    del model
            del base, bn_layers, early_kernel
        self.assertEqual(cases, 32)
        self.assertEqual(updates, 16)


# Permit direct execution without constructing large networks during import.
if __name__ == '__main__':
    unittest.main()
