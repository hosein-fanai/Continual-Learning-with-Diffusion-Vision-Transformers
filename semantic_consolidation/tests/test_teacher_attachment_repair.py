"""Preserve preattached teacher identities and training ownership during adaptation."""

import unittest

import numpy as np
import tensorflow as tf

from common.tests import test_head_teachers as teacher_fixtures
from semantic_consolidation.model import adapt_model


class TeacherAttachmentRepairTests(unittest.TestCase):
    """Exercise the adapter against real tiny networks and independent optimizers."""

    def setUp(self) -> None:
        """Reuse the supported tiny teacher fixture without inheriting its tests."""

        self.fixture = teacher_fixtures.HeadTeacherTests()
        self.fixture.setUp()

    def tearDown(self) -> None:
        """Restore the caller's policy and release the isolated Keras graph."""

        self.fixture.tearDown()

    def test_all_eight_preattachment_contracts_preserve_live_supervision(self) -> None:
        """Current or specialist roles survive both deferred-construction settings."""

        for roles in (tuple(["current"]), tuple(["classifier"]), tuple(["noise"]), 
                      ("classifier", "noise")):
            for deferred in (False, True):
                with self.subTest(roles=roles, deferred=deferred):
                    classifier_active = "current" in roles or "classifier" in roles
                    noise_active = "current" in roles or "noise" in roles
                    base = self.fixture.make_model(
                        current_teacher_network=self.fixture.make_network() if "current" in roles else None, 
                        classifier_teacher_network=self.fixture.make_classifier() if "classifier" in roles else None, 
                        noise_teacher_network=self.fixture.noise_network() if "noise" in roles else None, 
                        defer_teacher=deferred, 
                        clf_distil_loss_coef=float(classifier_active), 
                        noise_distil_loss_coef=float(noise_active)
                    )
                    owners = {}
                    for role in roles:
                        base.set_current_teacher_network(
                            base.get_teacher_network(role), class_ids=[1, 0], 
                            task_class_ids=[1], teacher_name=role
                        )
                        owners[role] = base.get_teacher_model(role)
                    base.optimizer.iterations.assign(7)
                    before = base.network.get_weights()
                    adapted = adapt_model(base, controller=object())
                    self.assertIs(adapted.network, base.network)
                    self.assertIs(adapted.optimizer, base.optimizer)
                    self.assertEqual(int(adapted.optimizer.iterations), 7)
                    self.assertEqual(bool(adapted.use_classifier_distil), classifier_active)
                    self.assertEqual(bool(adapted.use_noise_distil_loss), noise_active)
                    self.fixture.assert_weights_equal(adapted.network, before)
                    for role in roles:
                        teacher = base.get_teacher_network(role)
                        self.assertIs(adapted.get_teacher_network(role), teacher)
                        self.assertIs(adapted.get_teacher_model(role), owners[role])
                        self.assertFalse(teacher.trainable)
                        self.assertEqual(getattr(adapted, f"{role}_teacher_class_ids"), (1, 0))
                        self.assertEqual(getattr(adapted, f"{role}_teacher_task_class_ids"), tuple([1]))
                    for head, active in (("classifier", classifier_active), ("noise", noise_active)):
                        spec = adapted._current_teacher_spec(head)
                        self.assertEqual(spec is not None, active)
                        # Enabled heads retain the explicitly permuted label mapping.
                        if active:
                            self.assertEqual(spec["class_ids"], (1, 0))
                            self.assertEqual(spec["task_class_ids"], tuple([1]))

    def test_native_previous_and_current_owners_keep_compiled_optimizer_state(self) -> None:
        """Reattachment retains teacher optimizers and their existing iteration state."""

        base = self.fixture.make_model(
            teacher_network=self.fixture.make_network(), 
            current_teacher_network=self.fixture.make_network(), 
            classifier_teacher_network=None, noise_teacher_network=None
        )
        owners = {}
        for role in ("previous", "current"):
            base.compile_teacher(
                teacher_name=role, optimizer=tf.keras.optimizers.Adam(.002), 
                loss="mse", run_eagerly=True, jit_compile=False
            )
            owners[role] = base.get_teacher_model(role)
            owners[role].optimizer.iterations.assign(3)
        adapted = adapt_model(base, controller=object())
        for role, owner in owners.items():
            self.assertIs(adapted.get_teacher_model(role), owner)
            self.assertIs(adapted.get_teacher_model(role).optimizer, owner.optimizer)
            self.assertEqual(int(owner.optimizer.iterations), 3)
            self.assertFalse(adapted.get_teacher_network(role).trainable)

    def test_image_teacher_retains_fine_tuning_mask_and_optimizer_after_adaptation(self) -> None:
        """A selected teacher fit still updates its tail while its frozen backbone stays fixed."""

        base = self.fixture.make_model(noise_teacher_network=None, noise_distil_loss_coef=0.)
        teacher = base.get_teacher_network("classifier")
        optimizer = teacher.optimizer
        original_mask = base._get_keras_teacher_fit_state("classifier")
        before = teacher.get_weights()
        student_before = base.network.get_weights()
        adapted = adapt_model(base, controller=object())
        self.assertIs(adapted._get_keras_teacher_fit_state("classifier"), original_mask)
        self.assertIs(adapted.get_teacher_model("classifier").optimizer, optimizer)
        adapted.fit_teacher(self.fixture.dataset(), teacher_name="classifier", epochs=1, verbose=0)
        self.assertEqual(int(optimizer.iterations), 1)
        self.assertEqual(int(adapted.optimizer.iterations), 0)
        self.assertFalse(teacher.trainable)
        self.fixture.assert_weights_equal(adapted.network, student_before)
        for layer, was_trainable in original_mask[1]:
            # Previously frozen layers must remain bitwise unchanged after teacher fitting.
            if not was_trainable:
                for variable in layer.weights:
                    index = next(index for index, candidate in enumerate(teacher.weights) if candidate is variable)
                    np.testing.assert_array_equal(variable.numpy(), before[index])
        self.assertTrue(any(not np.array_equal(old, new) for old, new in zip(before, teacher.get_weights())))

    def test_post_adaptation_attachment_and_alias_guards_still_apply(self) -> None:
        """Adapted models keep the public attachment lifecycle and independence guard."""

        base = self.fixture.make_model(
            classifier_teacher_network=None, noise_teacher_network=None, defer_teacher=True
        )
        adapted = adapt_model(base, controller=object())
        classifier = self.fixture.make_classifier()
        adapted.set_classifier_teacher_network(classifier, class_ids=[1, 0], task_class_ids=[1])
        self.assertIs(adapted.get_teacher_network("classifier"), classifier)
        self.assertTrue(adapted.use_classifier_distil)
        with self.assertRaises(ValueError):
            adapted.set_current_teacher_network(self.fixture.make_network())
        with self.assertRaises(ValueError):
            adapted.set_teacher_network(adapted.network)


# Direct execution runs these focused regression cases.
if __name__ == "__main__":
    unittest.main()
