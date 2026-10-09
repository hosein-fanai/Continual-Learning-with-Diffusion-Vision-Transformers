"""Check explicit rebuild policies for layers owned by several nested parents."""

import unittest

import tensorflow as tf

from common.tests import test_fit_teacher as teacher_fixtures
from diffusion.models.wrapper.teacher_rebuild import (
    copy_frozen_teacher_weights, restore_teacher_trainability, teacher_layer_trainability
)


class SharedTeacherRebuildTests(unittest.TestCase):
    """Keep shared-layer flags stable and reject incompatible parameter reset requests."""

    def make_graph(self) -> tuple:
        """Build two nested parents that share one independently addressable dense layer."""

        shared = tf.keras.layers.Dense(2, use_bias=False, kernel_initializer="ones", name="shared")
        left = tf.keras.Sequential([tf.keras.Input(tuple([2])), shared], name="left")
        right = tf.keras.Sequential([tf.keras.Input(tuple([2])), shared], name="right")
        inputs = tf.keras.Input(tuple([2]))
        output = tf.keras.layers.Add()([left(inputs), right(inputs)])
        model = tf.keras.Model(inputs, output)
        return model, left, right, shared

    def tearDown(self) -> None:
        """Release the test graphs after each independent case."""

        tf.keras.backend.clear_session()

    def test_restore_waits_for_every_parent_before_shared_child(self) -> None:
        """A later frozen parent must not overwrite the shared child's saved explicit flag."""

        model, left, right, shared = self.make_graph()
        right.trainable = False
        shared.trainable = True
        state = teacher_layer_trainability(model)
        model.trainable = False
        restore_teacher_trainability(model, state)
        self.assertTrue(left.trainable)
        self.assertFalse(right.trainable)
        self.assertTrue(shared.trainable)

    def test_rebuild_rejects_conflicting_shared_parent_policies(self) -> None:
        """One shared kernel cannot simultaneously restart and preserve its old values."""

        source, _, right, shared = self.make_graph()
        candidate, _, _, _ = self.make_graph()
        right.trainable = False
        shared.trainable = True
        state = teacher_layer_trainability(source)
        with self.assertRaisesRegex(ValueError, "conflicting frozen parent"):
            copy_frozen_teacher_weights(source, candidate, state)

    def test_rebuild_rejects_source_variable_aliases(self) -> None:
        """A clone must own fresh variables before any reset or copy is considered."""

        source, _, _, _ = self.make_graph()
        state = teacher_layer_trainability(source)
        with self.assertRaisesRegex(ValueError, "variable structure"):
            copy_frozen_teacher_weights(source, source, state)

    def test_initial_native_growth_materializes_complete_frozen_mask(self) -> None:
        """A grown native attachment builds lazy children before retaining its layer mask."""

        fixture = teacher_fixtures.FitTeacherTests()
        teacher = fixture.make_network(True, None)
        teacher.patch_embedder.trainable = False
        teacher.add_depths("vision_transformer_block")
        appended = teacher.layers_dicts[-1][teacher.VTB]
        appended.trainable = False
        model = fixture.make_wrapper(True, num_classes=None, teacher_network=teacher)
        model._restore_native_teacher_fit_state("previous")
        self.assertFalse(teacher.patch_embedder.trainable)
        self.assertTrue(all(not layer.trainable for layer in appended._flatten_layers()))
        self.assertGreater(len(teacher.trainable_variables), 0)
        model.set_teacher_network(teacher)
        self.assertFalse(teacher.trainable)


# Direct invocation runs only the structural shared-teacher regressions.
if __name__ == "__main__":
    unittest.main()
