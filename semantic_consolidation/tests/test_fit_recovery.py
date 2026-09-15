"""Actual optimizer-step and shuffled-iterator recovery on tiny TensorFlow models."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.recovery import load_task_checkpoint, save_task_progress
from semantic_consolidation.config import RouteSettings
from semantic_consolidation.controller import RouteController, _ElapsedBudget
from semantic_consolidation.fit_recovery import FitCheckpoint, _variables
from semantic_consolidation.memory import ClassBalancedPool, ModulationBank
from semantic_consolidation.model import adapt_model
from semantic_consolidation.phases import RoutePhase
from semantic_consolidation.tests.test_phases import _make_wrapper


class _TickClock(tf.keras.callbacks.Callback):
    """Advance a controlled active-work clock by one second per real update."""

    recovery_state_scope = "stateless"

    def __init__(self, clock: list[float]) -> None:
        """Attach the test clock without changing model training state.

        Args:
            clock (list[float]): One-element monotonic-seconds clock owned by the test.

        Returns:
            initialized (None): None; stores the shared controlled clock.

        Raises:
            ValueError: If the caller supplies an invalid clock to subsequent callbacks.
        """
        super().__init__()
        self.clock = clock

    def get_recovery_config(self) -> dict[str, object]:
        """Declare this deterministic test-only callback's fixed time increment.

        Returns:
            config (dict[str, object]): One second per completed training batch.

        Raises:
            None: This method has no failure conditions.
        """
        return {"seconds_per_batch": 1.}

    def on_train_batch_end(self, batch: int, logs: dict[str, object] | None = None) -> None:
        """Account for one actual completed update in the controlled clock.

        Args:
            batch (int): Zero-based batch supplied by native Keras.
            logs (dict[str, object] | None): Existing metrics, left unchanged.

        Returns:
            advanced (None): None; adds one second to the test clock.

        Raises:
            IndexError: If the test clock no longer contains its scalar value.
        """
        self.clock[0] += 1.


class FitRecoveryTests(unittest.TestCase):
    """Compare real continued updates and histories with uninterrupted fits."""

    def test_shuffled_allocation_restores_completed_and_active_replay_fits(self) -> None:
        """Restore nonuniform shuffled KD, including its next update and audit rows.

        Returns:
            checked (None): None; native, checkpointed and resumed weights,
                optimizer state, histories and allocation state agree exactly.

        Raises:
            AssertionError: If a completed fit or active fit loses allocation state.
        """
        from allocation_study.tests.test_weighting import RealWrapperWeightingTests
        from allocation_study.weighting import AllocationSettings, allocation_run

        images = tf.reshape(tf.linspace(-1., 1., 128), [8, 4, 4, 1])
        labels = tf.constant([0, 0, 0, 0, 1, 1, 1, 1], tf.int32)

        def execute(root: Path, checkpointed: bool = True, interrupt: bool = False,
                    resume: bool = False) -> tuple:
            """Use real replay objectives and the common numeric progress serializer."""
            tf.keras.backend.clear_session()
            base = RealWrapperWeightingTests._wrapper()
            base._check_new_labels(y=tf.constant([0, 1], tf.int32), verbose=False)
            head = base.network.classifier.layers[-1]
            head.kernel.assign(np.random.default_rng(89).normal(0., 1., head.kernel.shape).astype("float32"))
            head.bias.assign([1., -1.])
            base.set_teacher_network(base.snapshot_teacher_network("raw"))
            base._check_new_labels(y=tf.constant([2, 3], tf.int32), verbose=False)
            owner = adapt_model(base, RouteController(RouteSettings(seed=17, checkpoint_interval=1)))
            owner.map_num_parallel_calls = 1
            owner.compile(optimizer=base.optimizer, run_eagerly=True)

            def writer(state: dict, iterator: object) -> Path:
                """Interrupt after the first committed batch of the second replay fit."""
                path = save_task_progress(root, 0, {
                    "class_order": [0, 1, 2, 3], "task_groups": [[0, 1], [2, 3]],
                    "active_task_index": 0, "fit_progress": state,
                }, {"iterator": iterator})
                # Leave one completed fit and one active fit to reconstruct on resume.
                if interrupt and len(state["stages"]) == 2 and state["stages"][-1]["batch"] == 1:
                    raise RuntimeError("intentional allocation interruption")
                return path

            # The native run independently checks that persistence preserves training.
            if checkpointed:
                owner.configure_fit_checkpoint(writer, load_task_checkpoint(root) if resume else None)
            dataset = tf.data.Dataset.from_tensor_slices((images, labels, tf.ones(8, tf.bool))).batch(8)
            histories = []
            with allocation_run(owner, AllocationSettings("shuffled_confidence_drift", seed=3)) as runtime:
                for repeats in (1, 3):
                    with runtime.phase(owner, "replay"):
                        histories.append(owner.fit_joint(dataset.repeat(repeats), epochs=1, verbose=0).history)
                allocation = runtime.get_recovery_state()
                self.assertEqual(runtime.audit()["batch_count"], 4)
                self.assertEqual(runtime.audit()["eligible_rows"], 32)
                self.assertGreater(np.ptp(allocation["records"][:, 6]), 0.)
            return [value.numpy().copy() for value in _variables(owner)], histories, allocation

        with tempfile.TemporaryDirectory(prefix="allocation-recovery-") as directory:
            root = Path(directory)
            expected = execute(root / "native", checkpointed=False)
            checkpointed = execute(root / "complete")
            with self.assertRaisesRegex(RuntimeError, "intentional allocation interruption"):
                execute(root / "interrupted", interrupt=True)
            resumed = execute(root / "interrupted", resume=True)
            for actual in (checkpointed, resumed):
                self.assertEqual(len(actual[0]), len(expected[0]))
                for before, after in zip(expected[0], actual[0]):
                    np.testing.assert_array_equal(before, after)
                self.assertEqual(expected[1], actual[1])
                for name, before in expected[2].items():
                    np.testing.assert_array_equal(before, actual[2][name])

    def _exercise(self, phase: str, fit_options: dict[str, object] | None = None) -> None:
        """Interrupt a committed middle batch and rebuild all objects before resume.

        Args:
            phase (str): joint, acquisition or consolidation objective to exercise.
            fit_options (dict[str, object] | None): Optional native joint controls;
                None checks two complete epochs of shuffled finite batches.

        Returns:
            checked (None): None; checks exact final variables/history and bounded
                intermediate checkpoint retention on the real native TensorFlow path.

        Raises:
            AssertionError: If reconstruction, cursor recovery or final state differs.
            Exception: Propagates unexpected native training and checkpoint failures.
        """
        images = np.random.default_rng(47).normal(size=(8, 4, 4, 1)).astype("float32")
        labels = np.repeat([0, 1], 4).astype("int32")
        settings = RouteSettings(seed=41, batch_size=4, checkpoint_interval=1, noise_levels=(0, 2))

        def execute(root: Path, interrupt: bool = False, resume: bool = False,
                    checkpointed: bool = True) -> tuple:
            """Run one reproducible real fit through the shared progress writer.

            Args:
                root (Path): Isolated task-checkpoint directory for this attempt.
                interrupt (bool): Raise after the second batch commit when True.
                resume (bool): Inspect and restore the latest committed fit when True.
                checkpointed (bool): False uses the unchanged original Keras fit,
                    providing an independent no-interruption equivalence reference.

            Returns:
                result (tuple): Independent final numeric variables and epoch history.

            Raises:
                RuntimeError: Intentional interruption after a durable batch commit.
                Exception: Propagates real training or persistence failures.
            """
            tf.keras.backend.clear_session()
            owner = adapt_model(_make_wrapper(), RouteController(settings))
            checkpoint = load_task_checkpoint(root) if resume else None
            clock = [1000. if resume else 0.]

            def writer(state: dict[str, object], iterator: object) -> Path:
                """Commit an exact iterator and then optionally interrupt this attempt.

                Args:
                    state (dict[str, object]): Numeric fit/sampler state and cursors.
                    iterator (object): Active checkpointable tf.data iterator.

                Returns:
                    path (Path): Committed native task checkpoint.

                Raises:
                    RuntimeError: Deliberate interruption after batch two is durable.
                """
                path = save_task_progress(root, 0, {"class_order": [0, 1], "task_groups": [[0, 1]],
                    "active_task_index": 0, "fit_progress": state}, {"iterator": iterator})
                # Persistence takes ten controlled seconds while each real batch takes one.
                if fit_options and fit_options.get("clock_budget"):
                    clock[0] += 10.
                # Stop after persistence, so the resumed first update must follow batch two.
                stop_boundary = bool(fit_options and fit_options.get("interrupt_after_stop"))
                interrupted_batch = 2 if fit_options and fit_options.get("execution_steps") == 2 else 1
                active = state["stages"][-1]
                # Stop either inside training or after a saved callback stopping boundary.
                if interrupt and ((stop_boundary and active["stop_training"] and active["batch"] == 0)
                                  or (not stop_boundary and active["batch"] == interrupted_batch)):
                    raise RuntimeError("intentional committed interruption")
                return path

            # A separate legacy run verifies that enabling persistence does not change updates.
            if checkpointed:
                owner.configure_fit_checkpoint(writer, checkpoint)
            # Joint fitting uses actual shuffled batches across two epochs.
            if phase == "joint":
                model = owner
                options = {"epochs": 2, "verbose": 0} | (fit_options or {})
                mapped = options.pop("mapped_validation", False)
                timed = options.pop("clock_budget", False)
                options.pop("interrupt_after_stop", None)
                owner.steps_per_execution = options.pop("execution_steps", 1)
                dataset = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(8, seed=67).batch(2)
                # Exercise native mapped training and validation with fresh callback owners per attempt.
                if mapped:
                    owner.map_preprocess = True
                    owner.map_num_parallel_calls = 1
                    options.update(validation_data=tf.data.Dataset.from_tensor_slices((images, labels)).batch(2),
                                   validation_steps=2, validation_freq=2,
                                   callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=100.,
                                       patience=0, restore_best_weights=True)])
                # A restart's arbitrary monotonic origin must not consume the saved active allowance.
                if timed:
                    stopper = _ElapsedBudget(3.)
                    options["callbacks"] = [_TickClock(clock), stopper]
                    with patch("time.perf_counter", side_effect=lambda: clock[0]):
                        history = owner.fit_joint(dataset, **options)
                    self.assertEqual(stopper.elapsed, 3.)
                    self.assertTrue(stopper.reached)
                    # Only known completed writes are retained; their time is outside fit work.
                    if checkpointed:
                        self.assertEqual(owner.fit_checkpoint.state["stages"][-1]["seconds"], 3.)
                        self.assertGreater(owner.fit_checkpoint.state["checkpoint_seconds"], 0.)
                # Ordinary controls use the real monotonic clock.
                else:
                    history = owner.fit_joint(dataset, **options)
            # Local phases use independent gates and a fixed consolidation target.
            else:
                bank = ModulationBank(settings, int(owner.network.classifier.layers[-1].kernel.shape[0]), 53)
                bank.add([0, 1])
                target = owner.snapshot_teacher_network("raw") if phase == "consolidation" else None
                model = RoutePhase(owner, bank, ClassBalancedPool(images, labels), settings,
                    phase, [0, 1], 59, target=target, frozen_bank=bank.frozen() if target is not None else None)
                model.compile(optimizer=tf.keras.optimizers.Adam(0.001), run_eagerly=True)
                dataset = tf.data.Dataset.from_tensor_slices(np.arange(4, dtype="int32")).batch(1)
                history = model.fit(dataset, epochs=1, verbose=0)
            return [value.numpy().copy() for value in _variables(model)], history.history

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, expected_history = execute(root / "reference")
            legacy, legacy_history = execute(root / "legacy", checkpointed=False)
            self.assertEqual(len(reference), len(legacy))
            for expected, observed in zip(reference, legacy):
                np.testing.assert_array_equal(expected, observed)
            self.assertEqual(expected_history, legacy_history)
            with self.assertRaisesRegex(RuntimeError, "intentional committed"):
                execute(root / "interrupted", interrupt=True)
            actual, history = execute(root / "interrupted", resume=True)
            self.assertEqual(len(reference), len(actual))
            for expected, observed in zip(reference, actual):
                np.testing.assert_array_equal(expected, observed)
            self.assertEqual(expected_history, history)
            self.assertLessEqual(len(list((root / "interrupted").glob(".progress-*"))), 2)

    def test_acquisition_resumes_exact_focus_sampler_and_optimizer(self) -> None:
        """Check exact acquisition gates, Adam slots, local focus cycle and metrics.

        Returns:
            checked (None): None; real interrupted and uninterrupted states match.

        Raises:
            AssertionError: If the acquisition continuation differs.
        """
        self._exercise("acquisition")

    def test_consolidation_resumes_predictor_target_and_optimizer(self) -> None:
        """Check exact predictor/student continuation with an independent fixed target.

        Returns:
            checked (None): None; real interrupted and uninterrupted states match.

        Raises:
            AssertionError: If consolidation continuation differs.
        """
        self._exercise("consolidation")

    def test_joint_resumes_exact_shuffled_iterator_across_epochs(self) -> None:
        """Check real joint updates, tracked randomness and reshuffled epoch order.

        Returns:
            checked (None): None; actual final network/optimizer/metrics match exactly.

        Raises:
            AssertionError: If any saved batch, epoch history or state is repeated/lost.
        """
        self._exercise("joint")

    def test_partial_epochs_retain_remaining_shuffled_batches(self) -> None:
        """Check explicit two-step epochs retain the remaining finite iterator.

        Returns:
            checked (None): None; legacy, persisted and resumed updates match.

        Raises:
            AssertionError: If partial epochs reshuffle or discard unused batches.
        """
        self._exercise("joint", {"steps_per_epoch": 2, "epochs": 3, "initial_epoch": 1})

    def test_long_epochs_match_native_finite_exhaustion(self) -> None:
        """Check explicit long epochs stop at native finite exhaustion.

        Returns:
            checked (None): None; exact update and history comparison passes.

        Raises:
            AssertionError: If finite exhaustion repeats or loses a batch.
        """
        self._exercise("joint", {"steps_per_epoch": 6})

    def test_mapped_validation_and_early_stopping_preserve_native_updates(self) -> None:
        """Check mapped inputs, sparse validation and best-weight stopping together.

        Returns:
            checked (None): None; legacy and both checkpoint paths match exactly.

        Raises:
            AssertionError: If preprocessing, validation or callback state changes.
        """
        self._exercise("joint", {"mapped_validation": True, "epochs": 3})

    def test_elapsed_budget_excludes_writes_and_restart_downtime(self) -> None:
        """Check a three-second allowance survives ten-second writes and a new clock.

        Returns:
            checked (None): None; exactly three real updates and active seconds
                match ordinary fitting and interrupted continuation.

        Raises:
            AssertionError: If persistence or restart downtime consumes training time.
        """
        self._exercise("joint", {"clock_budget": True, "epochs": 5})

    def test_compiled_multi_step_execution_matches_native_batches(self) -> None:
        """Check native groups of two optimizer steps retain the configured execution.

        Returns:
            checked (None): None; legacy, persisted and resumed numeric states match.

        Raises:
            AssertionError: If compiled execution groups change update counts or data order.
        """
        self._exercise("joint", {"execution_steps": 2})

    def test_saved_stopping_epoch_does_not_repeat_callbacks_or_validation(self) -> None:
        """Resume after a stopping epoch without starting an additional epoch.

        Returns:
            checked (None): None; restored best weights, metrics and history match.

        Raises:
            AssertionError: If a committed validation/stopping boundary is repeated.
        """
        self._exercise("joint", {"mapped_validation": True, "epochs": 3, "interrupt_after_stop": True})

    def test_zero_extra_updates_do_not_reuse_previous_fit_duration(self) -> None:
        """An empty extra-training allowance reports no earlier joint-fit time.

        Returns:
            checked (None): None; the real controller applies zero extra updates
                and reports only the empty branch's controlled elapsed duration.

        Raises:
            AssertionError: If an earlier fit's duration is attributed to absent work.
        """
        settings = RouteSettings(seed=41, condition="extra_joint", acquisition_steps=0,
                                 consolidation_steps=0, noise_levels=(0,))
        controller = RouteController(settings)
        owner = adapt_model(_make_wrapper(), controller)
        owner._checkpoint_elapsed_seconds = 123.
        images = np.zeros((4, 4, 4, 1), dtype="float32")
        dataset = tf.data.Dataset.from_tensor_slices((images, np.array([0, 0, 1, 1], dtype="int32"))).batch(2)
        with patch("time.perf_counter", return_value=17.):
            record = controller.run(owner, dataset, {})
        self.assertEqual(record["extra_joint_updates"], 0)
        self.assertEqual(record["extra_joint_seconds"], 0.)
