"""Bounded native TMCL session and checkpoint integration contracts.

Tiny RGB DiTs and fixed synthetic current-only cohorts exercise existing
train_model and task recovery; no CIFAR download or benchmark scoring occurs.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.recovery import load_task_checkpoint
from semantic_consolidation.tests.test_tmcl_phases import make_rgb_wrapper
from semantic_consolidation.tmcl import TMCLTrainer
from semantic_consolidation.tmcl_config import TMCLSettings


class TMCLTrainerTests(unittest.TestCase):
    """Prove current-only phase budgets and strict completed-session restoration."""

    def setUp(self) -> None:
        """Create two disjoint current cohorts with much smaller labelled subsets."""

        self.policy = tf.keras.mixed_precision.global_policy().name
        tf.keras.mixed_precision.set_global_policy("float32")
        rng = np.random.default_rng(67)
        self.images = [rng.uniform(-0.8, 0.8, (8, 4, 4, 3)).astype("float32") for _ in range(2)]
        self.labels = [np.repeat([0, 1], 2).astype("int32"), np.repeat([2, 3], 2).astype("int32")]
        self.settings = TMCLSettings(
            pretrain_epochs=1, acquisition_epochs=1, consolidation_epochs=1, 
            batch_size=4, warmup_epochs=0, augmentation_views=2, 
            projector_hidden=8, projector_dim=6, predictor_hidden=8, seed=71
        )
        self.folder = tempfile.TemporaryDirectory(prefix="tmcl_session_test_")

    def tearDown(self) -> None:
        """Remove only this test's temporary checkpoint and restore framework state."""

        self.folder.cleanup()
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy(self.policy)

    def _trainer(self, settings: TMCLSettings | None = None) -> TMCLTrainer:
        """Build the same fixed four-class native architecture and two-task schedule."""

        return TMCLTrainer(make_rgb_wrapper(num_classes=4), [[0, 1], [2, 3]], settings or self.settings)

    def _run(self, trainer: TMCLTrainer, session: int) -> dict[str, object]:
        """Supply only the active task's unlabelled images and four labelled rows."""

        return trainer.run_session(self.images[session], self.images[session][:4], self.labels[session], verbose=0)

    def _assert_rejected_before_fit(self, changes: dict[str, object]) -> None:
        """Allow settings or trainer preflight rejection, but forbid any training or mutation."""

        trainer = None
        before = []
        with patch.object(TMCLTrainer, "_fit", side_effect=AssertionError("Invalid recipe reached fitting.")) as fit:
            with self.assertRaises(ValueError):
                settings = replace(self.settings, **changes)
                trainer = self._trainer(settings)
                before = [variable.numpy().copy() for variable in trainer.adapter.network.variables]
                self._run(trainer, 0)
            fit.assert_not_called()
        # A trainer-level guard must be just as atomic as an earlier settings-level guard.
        if trainer is not None:
            self.assertFalse(trainer._in_progress)
            self.assertEqual(trainer.records, [])
            self.assertEqual(trainer.adapter.bank.classes, [])
            self.assertIsNone(trainer.heads)
            for value, variable in zip(before, trainer.adapter.network.variables):
                np.testing.assert_array_equal(variable.numpy(), value)

    def test_invalid_acquisition_batch_is_rejected_before_pretraining(self) -> None:
        """An undefined OPL batch cannot be discovered after a costly VI prefix."""

        self._assert_rejected_before_fit({"batch_size": 2})

    def test_si_without_any_representation_budget_is_rejected_before_acquisition(self) -> None:
        """A missing VI head must fail before gates are trained or a snapshot is attempted."""

        self._assert_rejected_before_fit({"pretrain_epochs": 0, "consolidation_epochs": 0, "si_weight": 1.})

    def test_si_without_consolidation_is_rejected_even_with_initial_vi(self) -> None:
        """An unused future SI teacher cannot silently turn into a pretraining-only recipe."""

        self._assert_rejected_before_fit({"consolidation_epochs": 0, "si_weight": 1.})

    def test_si_without_vi_is_rejected_before_delayed_projector_gradients(self) -> None:
        """Unsupported delayed VI-head clocks cannot reuse an already advanced backbone Adam step."""

        self._assert_rejected_before_fit({"pretrain_epochs": 0, "vi_weight": 0., "si_weight": 1.})

    def test_two_row_vi_only_batches_do_not_inherit_acquisition_restrictions(self) -> None:
        """Correlation-valid VI baselines remain available when no OPL phase executes."""

        settings = replace(self.settings, batch_size=2, mi_weight=0.)
        self.assertEqual(settings.budgets(8, first_session=True)["acquisition"], 0)

    def test_invalid_pnr_view_pairing_is_rejected_before_first_session(self) -> None:
        """Unsupported four-view PNR pairing cannot surface only after a previous target exists."""

        self._assert_rejected_before_fit({"si_weight": 1., "si_method": "pnr", "si_objective": "ghosh", "augmentation_views": 2})

    def test_disjoint_tasks_use_full_current_epoch_budgets(self) -> None:
        """No historical images are required to retain and use the first task's gates."""

        trainer = self._trainer()
        first = self._run(trainer, 0)
        old = [variable.numpy().copy() for variable in trainer.adapter.bank.variables_for([0, 1])]
        second = self._run(trainer, 1)
        self.assertEqual(first["current_examples"], 8)
        self.assertEqual(first["labelled_examples"], 4)
        self.assertEqual(first["steps_per_epoch"], 2)
        self.assertEqual(first["pretrain"]["updates"], 2)
        self.assertEqual(second["pretrain"]["updates"], 0)
        for record in (first, second):
            self.assertEqual(record["acquisition"]["updates"], 2)
            self.assertEqual(record["consolidation"]["updates"], 2)
            self.assertEqual(record["consolidation"]["example_draws"], 8)
            self.assertFalse(record["image_replay"])
            self.assertTrue(all(record["invariants"].values()))
        self.assertEqual(second["retained_classes"], [0, 1, 2, 3])
        self.assertEqual(int(trainer.representation_optimizer.iterations.numpy()), 6)
        self.assertEqual(int(trainer.mi_optimizer.iterations.numpy()), 4)
        for before, variable in zip(old, trainer.adapter.bank.variables_for([0, 1])):
            np.testing.assert_array_equal(variable.numpy(), before)
        with self.assertRaisesRegex(RuntimeError, "unfinished schedule"):
            self._run(trainer, 1)

    def test_checkpoint_restores_features_and_exact_next_session(self) -> None:
        """A completed boundary reproduces the next seeded session without refitting task one."""

        uninterrupted = self._trainer()
        self._run(uninterrupted, 0)
        checkpoint = uninterrupted.save(Path(self.folder.name) / "reset_heads")
        expected = uninterrupted.adapter.features(tf.convert_to_tensor(self.images[0]), training=False).numpy()
        restored = self._trainer()
        restored.restore(checkpoint)
        actual = restored.adapter.features(tf.convert_to_tensor(self.images[0]), training=False).numpy()
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(restored.records), 1)
        self.assertEqual(set(restored.heads), {"vi", "mi", "predictor"})
        for key in uninterrupted.heads:
            for before, after in zip(uninterrupted.heads[key].weights, restored.heads[key].weights):
                np.testing.assert_array_equal(after.numpy(), before.numpy())
        for attribute in ("representation_optimizer", "mi_optimizer"):
            for before, after in zip(getattr(uninterrupted, attribute).variables, getattr(restored, attribute).variables):
                np.testing.assert_array_equal(after.numpy(), before.numpy())
        self._run(uninterrupted, 1)
        self._run(restored, 1)
        for before, after in zip(uninterrupted.adapter.network.variables, restored.adapter.network.variables):
            np.testing.assert_array_equal(after.numpy(), before.numpy())
        for before, after in zip(uninterrupted.adapter.bank.variables_for(), restored.adapter.bank.variables_for()):
            np.testing.assert_array_equal(after.numpy(), before.numpy())
        for attribute in ("representation_optimizer", "mi_optimizer"):
            for before, after in zip(getattr(uninterrupted, attribute).variables, getattr(restored, attribute).variables):
                np.testing.assert_array_equal(after.numpy(), before.numpy())

    def test_no_reset_heads_round_trip_preserves_bn_and_projection_state(self) -> None:
        """The explicit no-reset ablation restores all three trained head models."""

        settings = replace(self.settings, reset_projectors=False)
        trainer = self._trainer(settings)
        self._run(trainer, 0)
        checkpoint = trainer.save(Path(self.folder.name) / "retained_heads")
        restored = self._trainer(settings)
        restored.restore(checkpoint)
        self.assertEqual(set(restored.heads), {"vi", "mi", "predictor"})
        for key in trainer.heads:
            for before, after in zip(trainer.heads[key].weights, restored.heads[key].weights):
                np.testing.assert_array_equal(after.numpy(), before.numpy())
        self.assertEqual(restored.records, trainer.records)

    def test_acquisition_only_checkpoint_restores_without_representation_heads(self) -> None:
        """Explicit acquisition-only ablations restore the bank without fabricating optimizer state."""

        settings = replace(self.settings, pretrain_epochs=0, consolidation_epochs=0)
        trainer = self._trainer(settings)
        record = self._run(trainer, 0)
        self.assertEqual(record["pretrain"]["updates"], 0)
        self.assertEqual(record["consolidation"]["updates"], 0)
        self.assertEqual(record["acquisition"]["updates"], 2)
        self.assertIsNone(trainer.heads)
        self.assertIsNone(trainer.representation_optimizer)
        checkpoint = trainer.save(Path(self.folder.name) / "acquisition_only")
        restored = self._trainer(settings)
        restored.restore(checkpoint)
        self.assertEqual(restored.records, trainer.records)
        self.assertIsNone(restored.heads)
        self.assertIsNone(restored.representation_optimizer)
        self.assertIsNone(restored.mi_optimizer)
        self.assertIsNone(restored.si_optimizer)
        for before, after in zip(trainer.adapter.network.variables, restored.adapter.network.variables):
            np.testing.assert_array_equal(after.numpy(), before.numpy())
        for before, after in zip(trainer.adapter.bank.variables_for(), restored.adapter.bank.variables_for()):
            np.testing.assert_array_equal(after.numpy(), before.numpy())
        second = self._run(restored, 1)
        self.assertEqual(second["retained_classes"], [0, 1, 2, 3])
        self.assertIsNone(restored.heads)

    def test_checkpoint_rejects_changed_settings_or_source_identity(self) -> None:
        """A matching architecture cannot legitimize a different scientific/source recipe."""

        trainer = self._trainer()
        self._run(trainer, 0)
        checkpoint = trainer.save(Path(self.folder.name) / "identity")
        changed_settings = self._trainer(replace(self.settings, mi_weight=0.5))
        with self.assertRaises(ValueError):
            changed_settings.restore(checkpoint)
        self.assertFalse(changed_settings._in_progress)
        self.assertEqual(changed_settings.adapter.bank.classes, [])
        changed_source = self._trainer()
        changed_source.identity = "deliberately_different_source_fingerprint"
        with self.assertRaises(ValueError):
            changed_source.restore(checkpoint)
        self.assertFalse(changed_source._in_progress)
        self.assertEqual(changed_source.adapter.bank.classes, [])

    def test_failed_trackable_restore_cannot_continue_with_partial_state(self) -> None:
        """A restore failure after bank mutation leaves the trainer unusable until replaced."""

        settings = replace(self.settings, pretrain_epochs=0, consolidation_epochs=0)
        trainer = self._trainer(settings)
        self._run(trainer, 0)
        checkpoint = trainer.save(Path(self.folder.name) / "restore_failure")
        restored = self._trainer(settings)

        def fail_trackable_stage(checkpoint_path: object, **kwargs: object) -> object:
            """Permit immutable manifest validation, then simulate strict runtime restore failure."""

            # The second loader call owns variable restoration after bank reconstruction.
            if "trackables" in kwargs:
                raise RuntimeError("Injected trackable restoration failure.")
            return load_task_checkpoint(checkpoint_path, **kwargs)

        with patch("semantic_consolidation.tmcl.load_task_checkpoint", side_effect=fail_trackable_stage):
            with self.assertRaisesRegex(RuntimeError, "Injected"):
                restored.restore(checkpoint)
        self.assertTrue(restored._in_progress)
        self.assertEqual(restored.records, [])
        self.assertEqual(restored.adapter.bank.classes, [0, 1])
        with patch.object(TMCLTrainer, "_fit") as fit:
            with self.assertRaises(RuntimeError):
                self._run(restored, 0)
            fit.assert_not_called()

    def test_si_snapshot_precedes_head_reset_and_restores_its_frozen_state(self) -> None:
        """The next session sees the previous trained VI state, not reset student heads."""

        settings = replace(self.settings, si_weight=1.)
        trainer = self._trainer(settings)
        self._run(trainer, 0)
        self.assertEqual(int(trainer.si_optimizer.iterations.numpy()), 0)
        self.assertFalse(trainer.previous["adapter"].network.trainable)
        self.assertFalse(trainer.previous["projector"].trainable)
        self.assertIsNot(trainer.previous["projector"], trainer.heads["vi"])
        self.assertTrue(any(not np.array_equal(old.numpy(), current.numpy())
                            for old, current in zip(trainer.previous["projector"].weights, trainer.heads["vi"].weights)))
        checkpoint = trainer.save(Path(self.folder.name) / "with_previous_state")
        restored = self._trainer(settings)
        restored.restore(checkpoint)
        for component in ("adapter", "projector"):
            first = trainer.previous[component]
            second = restored.previous[component]
            first_variables = first.network.variables if component == "adapter" else first.variables
            second_variables = second.network.variables if component == "adapter" else second.variables
            for before, after in zip(first_variables, second_variables):
                np.testing.assert_array_equal(after.numpy(), before.numpy())
        previous = restored.previous
        frozen_variables = previous["adapter"].network.variables + previous["projector"].variables
        frozen_before = [variable.numpy().copy() for variable in frozen_variables]
        record = self._run(restored, 1)
        self.assertEqual(int(restored.si_optimizer.iterations.numpy()), 2)
        self.assertTrue(all(np.isfinite(row["si"]) for row in record["consolidation"]["trace"]))
        self.assertTrue(any(row["si"] != 0. for row in record["consolidation"]["trace"]))
        self.assertIsNot(restored.previous["adapter"], previous["adapter"])
        for before, variable in zip(frozen_before, frozen_variables):
            np.testing.assert_array_equal(variable.numpy(), before)


# Direct invocation retains the repository's separate-process unittest entry point.
if __name__ == "__main__":
    unittest.main()
