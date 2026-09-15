"""Budget, provenance, adaptive timing, and existing-fit integration checks."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import tensorflow as tf

from common.dataloader import get_dataset
from common.keras_compat import optimizer_iterations
from semantic_consolidation.scheduling import (
    PhaseSchedule, ScheduleSettings, _cycle_indices, execute_schedule, split_task_pool,
)


class _TraceModel(tf.keras.Model):
    """A real Keras optimizer with recorded input identities and provenance."""

    def __init__(self, old_classes: int = 2) -> None:
        """Expose a tiny real optimizer and the wrapper's class/provenance metadata."""

        super().__init__()
        self.slope = tf.Variable(0., dtype=tf.float32)
        self.seen_classes = {10: 0, 11: 1, 12: 2, 13: 3}
        self.teacher_network = SimpleNamespace(num_classes=old_classes) if old_classes else None
        self.network = SimpleNamespace(num_classes=4)
        self.rows = []
        self.compile(optimizer=tf.keras.optimizers.SGD(0.001), run_eagerly=True)

    def train_step(self, data: tuple) -> dict[str, tf.Tensor]:
        """Record exactly the rows consumed while applying one scalar update."""

        images, labels = data[:2]
        mask = data[2] if len(data) == 3 else tf.zeros_like(labels, dtype=tf.bool)
        self.rows.append((images.numpy().reshape(-1).tolist(), labels.numpy().tolist(), mask.numpy().tolist()))
        with tf.GradientTape() as tape:
            loss = tf.square(self.slope - tf.reduce_mean(tf.cast(labels, tf.float32)))
            scaled = self.optimizer.scale_loss(loss) if hasattr(self.optimizer, "scale_loss") else loss
        self.optimizer.apply_gradients([(tape.gradient(scaled, self.slope), self.slope)])
        return {"loss": loss}

    def test_step(self, data: tuple) -> dict[str, tf.Tensor]:
        """Return a finite held-out loss without changing the trained scalar."""

        return {"loss": tf.constant(1., tf.float32)}


class SchedulingTests(unittest.TestCase):
    """Verify data separation, adaptive timing, and actual Keras update budgets."""

    @staticmethod
    def _pool(metadata: bool = True) -> tf.data.Dataset:
        """Mix recognizable old and current rows using nontrivial input label IDs."""

        values = np.arange(10, dtype=np.float32).reshape(10, 1)
        labels = np.asarray([10, 10, 11, 11, 12, 12, 12, 13, 13, 13], np.int32)
        return get_dataset(values, labels, batch_size=4, shuffle_buffer=0,
                           drop_remainder=False,
                           metadata=np.arange(10) < 4 if metadata else None)

    @staticmethod
    def _timeline(settings: ScheduleSettings, values: list, has_replay: bool = True) -> list:
        """Collect the state machine's complete policy decisions for fixed evidence."""

        scheduler = PhaseSchedule(settings, has_replay)
        result = []
        index = 0
        while True:
            wake = scheduler.next_wake()
            # An exhausted wake budget has already flushed all remaining replay.
            if not wake:
                break
            replay, reason = scheduler.after_wake(values[index])
            index += 1
            result.append((wake, replay, reason))
        return result

    def test_drift_changes_timing_without_changing_budgets(self) -> None:
        """Low drift defers replay; high drift and fixed timing spend identical totals."""

        settings = ScheduleSettings(mode="drift", wake_updates=6, replay_updates=5,
                                    wake_block_updates=2, replay_block_updates=2,
                                    drift_threshold=0.1)
        low = self._timeline(settings, [0.01, 0.02, 0.03])
        high = self._timeline(settings, [0.2, 0.3, 0.4])
        fixed = self._timeline(replace(settings, mode="fixed"), [0.01, 0.02, 0.03])
        self.assertEqual([row[1] for row in low], [0, 0, 5])
        self.assertEqual([row[1] for row in high], [2, 2, 1])
        self.assertEqual([row[1] for row in fixed], [2, 2, 1])
        for timeline in (low, high, fixed):
            self.assertEqual(sum(row[0] for row in timeline), 6)
            self.assertEqual(sum(row[1] for row in timeline), 5)
        self.assertEqual(low[-1][-1], "final_budget_flush")

    def test_first_task_and_state_validation(self) -> None:
        """First-task replay is absent and invalid observation order raises."""

        settings = ScheduleSettings(mode="drift", wake_updates=3, replay_updates=9,
                                    wake_block_updates=2)
        result = self._timeline(settings, [None, None], has_replay=False)
        self.assertEqual([row[:2] for row in result], [(2, 0), (1, 0)])
        state = PhaseSchedule(settings, True)
        with self.assertRaises(RuntimeError):
            state.after_wake(0.1)
        state.next_wake()
        with self.assertRaises(RuntimeError):
            state.next_wake()
        with self.assertRaises(ValueError):
            state.after_wake(None)
        for value in (float("nan"), float("inf"), -0.1, 0.9):
            with self.assertRaises(ValueError):
                state.after_wake(value)
        state.after_wake(0.1)

    def test_settings_reject_invalid_budgets(self) -> None:
        """Fractional/negative budgets and impossible JS values cannot configure runs."""

        for values in ({"mode": "sleep"}, {"wake_updates": 0}, {"wake_updates": True},
                       {"replay_updates": -1}, {"batch_size": 1.5},
                       {"wake_block_updates": 0}, {"replay_block_updates": 0},
                       {"drift_threshold": True}, {"drift_threshold": -0.1},
                       {"drift_threshold": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ScheduleSettings(**values)

    def test_deterministic_cycling_is_independent_of_blocks(self) -> None:
        """Splitting a stream into fit blocks preserves all row identities and order."""

        whole = _cycle_indices(7, 0, 25, seed=11)
        parts = np.concatenate([_cycle_indices(7, 0, 5, 11),
                                _cycle_indices(7, 5, 13, 11),
                                _cycle_indices(7, 18, 7, 11)])
        np.testing.assert_array_equal(whole, parts)
        np.testing.assert_array_equal(np.sort(whole[:7]), np.arange(7))
        self.assertFalse(np.array_equal(whole, _cycle_indices(7, 0, 25, 12)))

    def test_split_uses_explicit_metadata_and_shared_label_mapping(self) -> None:
        """Explicit provenance takes priority over the disjoint-class fallback."""

        wrapper = _TraceModel()
        explicit = split_task_pool(wrapper, self._pool())
        fallback = split_task_pool(wrapper, self._pool(metadata=False))
        self.assertEqual(explicit["provenance"], "explicit_replay_mask")
        self.assertEqual(fallback["provenance"], "disjoint_previous_teacher_vocabulary")
        for key in ("current_images", "current_labels", "replay_images", "replay_labels"):
            np.testing.assert_array_equal(explicit[key], fallback[key])
        np.testing.assert_array_equal(explicit["replay_labels"], [10, 10, 11, 11])
        # Explicit false provenance does not relabel old real rows as replay.
        current_only = get_dataset(np.arange(2, dtype=np.float32).reshape(2, 1),
                                   np.asarray([10, 12], np.int32), batch_size=2,
                                   shuffle_buffer=0, metadata=np.zeros(2, bool))
        self.assertEqual(len(split_task_pool(wrapper, current_only)["replay_labels"]), 0)

    def test_reject_infinite_empty_and_invalid_provenance(self) -> None:
        """Reject unbounded pools and novel labels incorrectly marked as replay."""

        wrapper = _TraceModel()
        with self.assertRaises(ValueError):
            split_task_pool(wrapper, self._pool().repeat())
        with self.assertRaises(ValueError):
            split_task_pool(wrapper, self._pool().take(0))
        bad = get_dataset(np.zeros((2, 1), np.float32), np.asarray([12, 13], np.int32),
                          metadata=np.ones(2, bool), shuffle_buffer=0, batch_size=2)
        with self.assertRaises(ValueError):
            split_task_pool(wrapper, bad)

    def test_real_fit_preserves_multiset_and_updates_across_modes(self) -> None:
        """Execute fixed, drift, and interleaved policies on one data multiset."""

        settings = ScheduleSettings(mode="fixed", wake_updates=4, replay_updates=3,
                                    wake_block_updates=2, replay_block_updates=1, batch_size=3)
        phase_rows, audits = {}, {}
        for mode in ("fixed", "drift", "interleaved"):
            with self.subTest(mode=mode):
                model = _TraceModel()
                probe_updates = []
                def probe(wrapper: _TraceModel, images: np.ndarray, labels: np.ndarray) -> dict:
                    """Report deterministic low drift and count each actual scoring call."""

                    probe_updates.append(int(wrapper.optimizer.iterations.numpy()))
                    return {"mean_js": 0.001, "scorer_example_forwards": 2 * len(labels)}
                history, audit, effective = execute_schedule(
                    model, self._pool(), {"epochs": 200, "verbose": 0},
                    replace(settings, mode=mode), fit_function=model.fit,
                    drift_probe=probe, seed=17,
                )
                self.assertEqual(int(model.optimizer.iterations.numpy()), 7)
                self.assertEqual(audit["presentations"], {"wake": 12, "replay": 9})
                self.assertEqual(audit["replaced_common_epochs"], 200)
                self.assertEqual(audit["wake_updates"], 4)
                self.assertEqual(audit["replay_updates"], 3)
                self.assertTrue(all(np.isfinite(history.history["loss"])))
                self.assertEqual(sum(len(batch[1]) for batch in effective), 10)
                phase_rows[mode] = {"wake": [], "replay": []}
                for images, labels, masks in model.rows:
                    self.assertTrue(all(masks) or not any(masks))
                    phase = "replay" if all(masks) else "wake"
                    phase_rows[mode][phase].extend(images)
                    self.assertTrue(all(label < 12 if phase == "replay" else label >= 12 for label in labels))
                audits[mode] = audit
                self.assertEqual(len(probe_updates), 4 if mode == "interleaved" else 2)
        self.assertEqual(phase_rows["fixed"], phase_rows["drift"])
        self.assertEqual(phase_rows["fixed"], phase_rows["interleaved"])
        self.assertEqual(audits["fixed"]["exposure_sequence_sha256"], audits["drift"]["exposure_sequence_sha256"])
        self.assertEqual(audits["fixed"]["exposure_sequence_sha256"], audits["interleaved"]["exposure_sequence_sha256"])
        self.assertEqual([row["phase"] for row in audits["drift"]["blocks"]], ["wake", "wake", "replay"])
        self.assertEqual([row["phase"] for row in audits["fixed"]["blocks"]], ["wake", "replay", "wake", "replay"])

    @unittest.skipUnless(hasattr(tf.keras.mixed_precision.LossScaleOptimizer, "scale_loss"),
                         "Exercises the Keras 3 loss-scale wrapper's distinct counter.")
    def test_loss_scaled_optimizer_reports_applied_joint_and_fixed_budgets(self) -> None:
        """Count inner optimizer updates even if the wrapper's own counter stays zero."""
        for mode in ("joint", "fixed"):
            with self.subTest(mode=mode):
                model = _TraceModel()
                inner = tf.keras.optimizers.SGD(0.001)
                model.compile(optimizer=tf.keras.mixed_precision.LossScaleOptimizer(inner, initial_scale=8.),
                              run_eagerly=True)
                _, audit, _ = execute_schedule(
                    model, self._pool(), {"epochs": 1, "verbose": 0},
                    ScheduleSettings(mode=mode, wake_updates=2, replay_updates=1,
                                     wake_block_updates=1, replay_block_updates=1, batch_size=2),
                    fit_function=model.fit,
                )
                self.assertEqual(int(optimizer_iterations(model.optimizer).numpy()), 3)
                self.assertEqual(int(inner.iterations.numpy()), 3)
                # Ordinary joint fitting reports only its overall optimizer delta.
                if mode == "joint":
                    self.assertEqual(audit["updates"], 3)
                # Explicit schedules also preserve the current/replay decomposition.
                else:
                    self.assertEqual((audit["wake_updates"], audit["replay_updates"]), (2, 1))

    def test_selector_runs_after_wake_and_effective_pool_is_selected(self) -> None:
        """Each post-wake replacement reaches replay fitting and downstream routes."""

        model = _TraceModel()
        calls = []
        def selector(wrapper: _TraceModel, images: np.ndarray, labels: np.ndarray, wake_updates: int) -> tuple:
            """Make selected rows identifiable and record the live optimizer count."""

            calls.append((int(wrapper.optimizer.iterations.numpy()), wake_updates))
            return images + 100., labels, {"policy": "fixture", "rows": len(labels)}
        _, audit, effective = execute_schedule(
            model, self._pool(), {"verbose": 0},
            ScheduleSettings(mode="fixed", wake_updates=2, replay_updates=2,
                             wake_block_updates=1, replay_block_updates=1, batch_size=2),
            fit_function=model.fit, replay_selector=selector,
        )
        self.assertEqual(calls, [(1, 1), (3, 2)])
        self.assertTrue(audit["selection_changed_during_schedule"])
        replay_images = np.concatenate([np.asarray(batch[0])[np.asarray(batch[2])] for batch in effective])
        np.testing.assert_array_equal(replay_images.reshape(-1), np.arange(4) + 200.)
        replay_batches = [images for images, _, mask in model.rows if all(mask)]
        self.assertTrue(all(image >= 100. for image in replay_batches[0]))
        self.assertTrue(all(image >= 200. for image in replay_batches[1]))

    def test_partial_execution_groups_preserve_scheduled_updates_and_rows(self) -> None:
        """Grouped callbacks must accept odd and single-update finite blocks."""
        for mode in ("fixed", "drift", "interleaved"):
            results = []
            for execution_steps in (1, 2):
                with self.subTest(mode=mode, execution_steps=execution_steps):
                    model = _TraceModel()
                    model.steps_per_execution = execution_steps
                    _, audit, _ = execute_schedule(
                        model, self._pool(), {"verbose": 0},
                        ScheduleSettings(mode=mode, wake_updates=3, replay_updates=3,
                                         wake_block_updates=3, replay_block_updates=3, batch_size=3),
                        fit_function=model.fit, drift_probe=lambda *args: 0.01, seed=17,
                    )
                    self.assertEqual(int(model.optimizer.iterations.numpy()), 6)
                    self.assertEqual(audit["presentations"], {"wake": 9, "replay": 9})
                    results.append((model.rows, audit["exposure_sequence_sha256"]))
            self.assertEqual(*results)

    def test_joint_is_unchanged_and_first_task_skips_replay(self) -> None:
        """Joint mode retains epoch fitting; task one cannot spend historical replay."""

        model = _TraceModel()
        history, audit, original = execute_schedule(
            model, self._pool(), {"epochs": 2, "verbose": 0},
            ScheduleSettings(), fit_function=model.fit,
        )
        self.assertEqual(audit["updates"], 6)
        self.assertFalse(audit["fixed_budget"])
        self.assertEqual(len(history.epoch), 2)
        model = _TraceModel(old_classes=0)
        current = get_dataset(np.zeros((3, 1), np.float32), np.asarray([10, 11, 11], np.int32),
                              batch_size=2, shuffle_buffer=0, drop_remainder=False)
        _, audit, _ = execute_schedule(
            model, current, {"verbose": 0},
            ScheduleSettings(mode="drift", wake_updates=2, replay_updates=4, batch_size=2),
            fit_function=model.fit,
        )
        self.assertEqual(audit["updates"], 2)
        self.assertEqual(audit["effective_replay_updates"], 0)

    def test_stopping_or_skipped_optimizer_updates_cannot_claim_a_matched_budget(self) -> None:
        """Callbacks cannot silently shorten the declared fixed-update experiment."""

        class StopEarly(tf.keras.callbacks.Callback):
            """Stop after one batch to exercise the runtime budget guard."""

            def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
                """Force an incomplete fit even though it otherwise completes normally."""

                self.model.stop_training = True
        model = _TraceModel()
        with self.assertRaisesRegex(RuntimeError, "budget incomplete"):
            execute_schedule(
                model, self._pool(), {"verbose": 0, "callbacks": [StopEarly()]},
                ScheduleSettings(mode="fixed", wake_updates=2, wake_block_updates=2, batch_size=2),
                fit_function=model.fit,
            )
        with self.assertRaisesRegex(ValueError, "Early stopping"):
            execute_schedule(
                model, self._pool(), {"callbacks": [tf.keras.callbacks.EarlyStopping()]},
                ScheduleSettings(mode="fixed"), fit_function=model.fit,
            )

    def test_existing_diffusion_fit_keeps_replay_only_kd_and_teacher_boundaries(self) -> None:
        """Run the real V1 objective on pure current/replay batches after class growth."""

        from common.model import get_model

        model = get_model(
            model_name="dit_classifier", task="joint", image_shape=(4, 4, 1),
            class_num=None, seed=29, dtype_policy="float32", show_network_summary=False,
            model_kwargs={
                "num_classes": None, "timesteps": 8, "patch_size": 2,
                "dim": 4, "depth": 1, "mha_num_heads": 1, "vit_block_mlp_ratio": 1.,
                "clf_mha_num_heads": 1, "clf_vit_block_mlp_ratio": 1.,
                "classifier_mlp_ratio": 1, "classifier_only_distil_token": True,
                "clf_distil_token_type": "new_weight", "compile_args": {"run_eagerly": True},
            },
            wrapper_kwargs={
                "use_ema": False, "p_uncond": 1., "clf_loss_coef": 1.,
                "test_noisified_max_timesteps": 0, "test_steps": 2,
                "defer_teacher": True, "noise_distil_loss_coef": 0.1,
                "clf_distil_loss_coef": 0.1, "clf_distil_type": "soft",
                "clf_distil_scope": "replay_only",
            },
        )
        model._check_new_labels(y=np.asarray([0, 1], np.int32), verbose=False)
        model.set_teacher_network(model.snapshot_teacher_network("raw"))
        model._check_new_labels(y=np.asarray([2, 3], np.int32), verbose=False)
        optimizer = model.optimizer
        teacher_values = [variable.numpy().copy() for variable in model.teacher_network.weights]
        values = np.linspace(-1., 1., 8 * 4 * 4, dtype=np.float32).reshape(8, 4, 4, 1)
        labels = np.repeat(np.arange(4, dtype=np.int32), 2)
        dataset = get_dataset(values, labels, metadata=labels < 2, batch_size=2, shuffle_buffer=0)
        _, audit, _ = execute_schedule(
            model, dataset, {"verbose": 0},
            ScheduleSettings(mode="fixed", wake_updates=1, replay_updates=1, batch_size=2),
        )
        self.assertIs(model.optimizer, optimizer)
        self.assertEqual(audit["updates"], 2)
        self.assertEqual(audit["blocks"][0]["history"]["clf_distil_loss"], [0.])
        self.assertGreater(audit["blocks"][1]["history"]["clf_distil_loss"][0], 0.)
        self.assertTrue(np.isfinite(audit["blocks"][1]["history"]["noise_distil_loss"]).all())
        for before, variable in zip(teacher_values, model.teacher_network.weights):
            np.testing.assert_array_equal(before, variable.numpy())


# Direct invocation runs the same discoverable unittest suite.
if __name__ == "__main__":
    unittest.main()
