"""Focused tests for ordinary and mixed-precision custom gradient updates."""

from __future__ import annotations

from unittest import TestCase, main

import tensorflow as tf

from common.gradients import apply_policy_gradients


class GradientTests(TestCase):
    """Verify correct updates, one-time unscaling, and disconnected variables."""

    def setUp(self: GradientTests) -> None:
        """Remember the caller's numeric policy before each test.

        Returns:
            None.
        """

        self.previous_policy = tf.keras.mixed_precision.global_policy().name

    def tearDown(self: GradientTests) -> None:
        """Restore the caller's numeric policy after each test.

        Returns:
            None.
        """

        tf.keras.mixed_precision.set_global_policy(self.previous_policy)

    def test_float32_update_and_none_gradient_filtering(
        self: GradientTests,
    ) -> None:
        """Apply an ordinary gradient and omit a disconnected variable.

        Returns:
            None.
        """

        tf.keras.mixed_precision.set_global_policy("float32")
        trained = tf.Variable(2.0)
        disconnected = tf.Variable(7.0)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        with tf.GradientTape() as tape:
            loss = trained ** 2

        pairs = apply_policy_gradients(
            tape,
            optimizer,
            loss,
            [trained, disconnected],
        )

        self.assertEqual(len(pairs), 1)
        self.assertIs(pairs[0][1], trained)
        self.assertAlmostEqual(float(trained), 1.6, places=6)
        self.assertAlmostEqual(float(disconnected), 7.0, places=6)
        self.assertEqual(int(optimizer.iterations), 1)

    def test_mixed_float16_scales_and_unscales_once(
        self: GradientTests,
    ) -> None:
        """Match the float32 update despite a large fixed loss scale.

        Returns:
            None.
        """

        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        trained = tf.Variable(2.0, dtype=tf.float32)
        disconnected = tf.Variable(7.0, dtype=tf.float32)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            tf.keras.optimizers.SGD(learning_rate=0.1),
            dynamic=False,
            initial_scale=128.0,
        )
        with tf.GradientTape() as tape:
            loss = tf.cast(trained, tf.float16) ** 2

        pairs = apply_policy_gradients(
            tape,
            optimizer,
            loss,
            [trained, disconnected],
        )

        self.assertEqual(len(pairs), 1)
        self.assertIs(pairs[0][1], trained)
        self.assertAlmostEqual(float(pairs[0][0]), 4.0, places=6)
        self.assertAlmostEqual(float(trained), 1.6, places=6)
        self.assertAlmostEqual(float(disconnected), 7.0, places=6)
        self.assertEqual(int(optimizer.iterations), 1)

    def test_empty_and_fully_disconnected_updates_are_noops(
        self: GradientTests,
    ) -> None:
        """Return no pairs and leave optimizer iterations unchanged.

        Returns:
            None.
        """

        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        variable = tf.Variable(3.0)
        with tf.GradientTape() as empty_tape:
            empty_loss = variable ** 2
        self.assertEqual(
            apply_policy_gradients(empty_tape, optimizer, empty_loss, []),
            [],
        )

        disconnected = tf.Variable(5.0)
        with tf.GradientTape() as disconnected_tape:
            loss = tf.constant(1.0)
        self.assertEqual(
            apply_policy_gradients(
                disconnected_tape,
                optimizer,
                loss,
                [disconnected],
            ),
            [],
        )
        self.assertEqual(int(optimizer.iterations), 0)


# Run only this focused suite when the file is executed directly.
if __name__ == "__main__":
    main()
