"""Optional optimizer-step recovery using the existing compiled Keras training step.

Completed fit calls retain numeric state differences and their histories. On
restart the ordinary task controller reconstructs its targets and scheduling
decisions, restoring completed fits instead of applying their updates again.
The active fit resumes its saved tf.data iterator, optimizer, metrics and sampler.
The epoch adapter is isolated here for TensorFlow 2.20 / Keras 3.11.2. It supports
the native array-backed route pipeline; mapped random preprocessing requires one
map worker and the wrapper's explicitly tracked seed streams.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
import random
import time

import numpy as np
import tensorflow as tf
from keras.src.backend.tensorflow.trainer import TFEpochIterator

from common.recovery import (SCHEMA_VERSION, callback_recovery_state, capture_rng_state, fingerprint_state,
                             restore_callback_recovery_state, restore_rng_state)


def _variables(model: tf.keras.Model) -> list[object]:
    """Collect distinct model, optimizer, metric and untracked gate variables.

    Args:
        model (tf.keras.Model): Built route wrapper or compiled RoutePhase.

    Returns:
        variables (list[object]): Stable ordered assignable TensorFlow/Keras
            variables, including integer random counters and optimizer iterations.

    Raises:
        AttributeError: If the model is uncompiled or lacks the route contract.
    """
    optimizer = model.optimizer.variables
    values = [*model.variables, *(optimizer() if callable(optimizer) else optimizer)]
    for metric in model.metrics:
        values.extend(metric.variables)
    bank = getattr(model, "bank", None)
    # Gate banks intentionally remain outside Keras model serialization.
    if bank is not None:
        values.extend(variable for key in sorted(bank.vectors) for variable in bank.vectors[key])
    return list({id(value): value for value in values}.values())


def _callback_state(callbacks: list[object]) -> dict[str, object]:
    """Copy explicit callback state and the built-in per-fit counters used here.

    Args:
        callbacks (list[object]): Existing Keras callbacks attached to this fit.

    Returns:
        state (dict[str, object]): Detached declared state plus numeric stopping,
            scheduling and batch counters. No callback object is serialized.

    Raises:
        Exception: Propagates an explicitly declared callback getter failure.
    """
    names = ("wait", "best", "best_weights", "stopped_epoch", "cooldown_counter",
             "batches", "elapsed", "reached")
    return deepcopy({"declared": callback_recovery_state(callbacks),
                     "local": [{name: getattr(callback, name) for name in names if hasattr(callback, name)
                                and not hasattr(getattr(callback, name), "assign")}
                               for callback in callbacks]})


def _restore_callbacks(callbacks: list[object], state: dict[str, object]) -> None:
    """Restore previously validated callback counters and active elapsed budgets.

    Args:
        callbacks (list[object]): Reconstructed callbacks in their original order.
        state (dict[str, object]): Saved declared state and per-fit numeric fields.

    Returns:
        restored (None): None; elapsed timers exclude restart downtime.

    Raises:
        ValueError: If the callback count or declared state contract changed.
    """
    # The same configured fit must reconstruct the same callback owners.
    if len(callbacks) != len(state["local"]):
        raise ValueError("Fit checkpoint callback count differs from the configured fit.")
    restore_callback_recovery_state(callbacks, state["declared"])
    for callback, values in zip(callbacks, state["local"]):
        for name, value in values.items():
            setattr(callback, name, deepcopy(value))
        # Monotonic origins belong to the new process; elapsed work belongs to the run.
        if "elapsed" in values and hasattr(callback, "started"):
            callback.started = time.perf_counter() - values["elapsed"]


class FitCheckpoint:
    """Retain one task's completed fit states and its active optimizer-step cursor."""

    def __init__(self, owner: object, writer: Callable, checkpoint: object | None = None) -> None:
        """Attach an existing atomic common writer and optional inspected checkpoint.

        Args:
            owner (object): Semantic adapter owning checkpoint_interval and observers.
            writer (Callable): Common callback accepting plain state and a tf.data
                iterator and returning the committed checkpoint path.
            checkpoint (object | None): Inspected common TaskCheckpoint or None
                for a fresh task; its iterator is restored only at the active fit.

        Returns:
            initialized (None): None; copies numeric state without changing training.

        Raises:
            ValueError: If checkpoint state has an invalid schema or numeric payload.
        """
        self.owner, self.writer, self.checkpoint = owner, writer, checkpoint
        state = checkpoint.experiment_state["fit_progress"] if checkpoint is not None else {
            "schema_version": 1, "stages": [], "checkpoint_seconds": 0.}
        self.validate(state)
        self.state, self.position = deepcopy(state), 0

    @staticmethod
    def validate(state: dict[str, object]) -> None:
        """Reject malformed progress state before common changes live model state.

        Args:
            state (dict[str, object]): Versioned fit sequence with numeric variable
                changes, callback/sampler state and typed epoch/batch cursors.

        Returns:
            validated (None): None; validation is read-only.

        Raises:
            ValueError: If sequence ordering, schema, cursors or variable arrays
                are invalid. Exact reconstructed variable shapes are checked at fit entry.
        """
        # Only the declared numeric progress format may enter reconstruction.
        if (not isinstance(state, dict) or set(state) != {"schema_version", "stages", "checkpoint_seconds"}
                or state["schema_version"] != 1 or not isinstance(state["stages"], list)
                or not isinstance(state["checkpoint_seconds"], (int, float))
                or not np.isfinite(state["checkpoint_seconds"]) or state["checkpoint_seconds"] < 0):
            raise ValueError("Invalid semantic fit checkpoint schema.")
        for index, stage in enumerate(state["stages"]):
            required = {"signature", "initial", "changes", "epoch", "batch", "complete", "history",
                        "epochs", "callbacks", "phase", "rng", "seconds", "observer", "stop_training",
                        "iterator", "steps_seen"}
            # An unfinished fit can occur only at the end of the committed sequence.
            if (not isinstance(stage, dict) or set(stage) != required
                    or type(stage["epoch"]) is not int or stage["epoch"] < 0
                    or type(stage["batch"]) is not int or stage["batch"] < 0
                    or type(stage["complete"]) is not bool
                    or (not stage["complete"] and index != len(state["stages"]) - 1)
                    or not isinstance(stage["changes"], dict) or not isinstance(stage["signature"], list)
                    or not isinstance(stage["initial"], str) or not isinstance(stage["history"], dict)
                    or type(stage["steps_seen"]) is not int or stage["steps_seen"] < 0
                    or not isinstance(stage["iterator"], dict)
                    or not all(isinstance(key, str) and isinstance(value, bytes)
                               for key, value in stage["iterator"].items())
                    or not isinstance(stage["seconds"], (int, float)) or stage["seconds"] < 0
                    or not np.isfinite(stage["seconds"])):
                raise ValueError("Invalid semantic fit stage or cursor.")
            callbacks = stage["callbacks"]
            # Local callback arrays and counters have the same ordered ownership as declared state.
            if (type(stage["stop_training"]) is not bool or not isinstance(stage["epochs"], list)
                    or any(type(epoch) is not int or epoch < 0 for epoch in stage["epochs"])
                    or not isinstance(callbacks, dict) or set(callbacks) != {"declared", "local"}
                    or not isinstance(callbacks["declared"], list) or not isinstance(callbacks["local"], list)
                    or len(callbacks["declared"]) != len(callbacks["local"])
                    or any(not isinstance(values, dict) for values in callbacks["local"])):
                raise ValueError("Invalid semantic fit callback or epoch history state.")
            observer = stage["observer"]
            # Observer elapsed values are active durations, never old process clock origins.
            if observer is not None and (not isinstance(observer, dict)
                    or set(observer) != {"curves", "curve_seconds", "fit_elapsed", "elapsed", "resource_segments"}
                    or not isinstance(observer["curves"], list) or not isinstance(observer["resource_segments"], list)
                    or any(type(observer[name]) not in (int, float) or not np.isfinite(observer[name])
                           or observer[name] < 0 for name in ("curve_seconds", "fit_elapsed", "elapsed"))):
                raise ValueError("Invalid semantic fit observer state.")
            try:
                rng = stage["rng"]
                # This adapter captures exactly the two global streams, without undeclared RNG owners.
                if not isinstance(rng, dict) or set(rng) != {"schema_version", "python_global", "numpy_global"} or rng["schema_version"] != SCHEMA_VERSION:
                    raise ValueError("Invalid semantic fit random-state schema.")
                random.Random().setstate(rng["python_global"])
                numpy_state = rng["numpy_global"]
                np.random.RandomState(0).set_state((numpy_state["bit_generator"], numpy_state["keys"],
                    numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
                # Phase counters and its independent NumPy generator must form a complete state.
                if stage["phase"] is not None:
                    local = stage["phase"]
                    required_phase = {"step_number", "focus_cycle", "focus_counts", "updated_names",
                                      "example_draws", "view_draws", "trace", "rng"}
                    # Never install undeclared phase attributes or negative update counters.
                    if (not isinstance(local, dict) or set(local) != required_phase
                            or any(type(local[name]) is not int or local[name] < 0
                                   for name in ("step_number", "example_draws", "view_draws"))
                            or not isinstance(local["focus_cycle"], list)
                            or any(type(value) is not int or value < 0 for value in local["focus_cycle"])
                            or not isinstance(local["focus_counts"], dict)
                            or any(not isinstance(key, str) or not key.isdecimal() or type(value) is not int or value < 0
                                   for key, value in local["focus_counts"].items())
                            or not isinstance(local["updated_names"], set) or not isinstance(local["trace"], list)):
                        raise ValueError("Invalid semantic phase counters.")
                    np.random.default_rng(0).bit_generator.state = local["rng"]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Invalid semantic fit random or phase state.") from error
            for position, value in stage["changes"].items():
                # Model state retains exact numeric dtypes; no implicit coercion is permitted.
                if (not isinstance(position, str) or not position.isdecimal()
                        or not 0 <= int(position) < len(stage["signature"])
                        or not isinstance(value, np.ndarray) or value.dtype.kind not in "biufc"
                        or not np.isfinite(value).all()
                        or [list(value.shape), value.dtype.name] != stage["signature"][int(position)]):
                    raise ValueError("Invalid semantic fit variable state.")

    def fit(self, model: tf.keras.Model, dataset: tf.data.Dataset, options: dict[str, object]) -> tf.keras.callbacks.History:
        """Continue one existing objective with its original finite data iterator.

        Args:
            model (tf.keras.Model): Compiled semantic wrapper or RoutePhase;
                existing compiled train_function/train_step and optimizer remain authoritative.
            dataset (tf.data.Dataset): Finite raw batches with float32 images and
                integer labels/provenance, or integer phase ticks.
            options (dict[str, object]): Existing epochs, callbacks, validation,
                verbosity, initial_epoch and optional steps_per_epoch controls.
                Native compiled execution groups remain indivisible checkpoint boundaries.

        Returns:
            history (tf.keras.callbacks.History): Complete original epoch history,
                including committed epochs from previous attempts. Completed fits
                restore their state without repeating optimizer updates.

        Raises:
            ValueError: If a fit/input/variable/callback contract changed or the
                iterator lacks a finite/explicit update bound, mapped random
                preprocessing is parallel, or TensorFlow cannot checkpoint its state.
            Exception: Propagates existing training, callback, evaluation or atomic
                checkpoint errors. Work after the last commit may be repeated.
        """
        allowed = {"epochs", "initial_epoch", "verbose", "callbacks", "validation_data",
                   "steps_per_epoch", "shuffle", "validation_steps", "validation_freq",
                   "batch_size", "validation_batch_size"}
        # Unsupported options must never silently alter the requested training protocol.
        if set(options) - allowed:
            raise ValueError(f"Unsupported checkpointed fit controls: {sorted(set(options) - allowed)}.")
        epochs = int(options.get("epochs", 1))
        initial_epoch = int(options.get("initial_epoch", 0))
        steps = options.get("steps_per_epoch")
        count = int(tf.data.experimental.cardinality(dataset).numpy())
        # Explicitly bounded repeats are accepted; otherwise the pool must be finite.
        if steps is None and count < 0:
            raise ValueError("Checkpointed fitting requires a finite dataset or explicit steps_per_epoch.")
        steps = int(steps if steps is not None else count)
        # Negative cursors or empty update windows cannot describe a resumable fit.
        if initial_epoch < 0 or epochs < initial_epoch or steps <= 0:
            raise ValueError("Checkpointed fitting requires positive steps and 0 <= initial_epoch <= epochs.")
        callbacks = list(options.get("callbacks") or [])
        recovery_callbacks = list(callbacks)
        allocation = getattr(self.owner, "_scheduled_kd_allocation", None)
        # The loss hook owns growing audit tensors and the next shuffle seed.
        # Reuse explicit recovery hooks without adding it to Keras callbacks/weights.
        if allocation is not None:
            recovery_callbacks.append(allocation)
        verbosity = options.get("verbose", 0)
        callback_list = tf.keras.callbacks.CallbackList(callbacks, add_history=True,
            add_progbar=bool(verbosity), model=model, epochs=epochs, steps=steps, verbose=verbosity)
        phase = getattr(model, "phase", None)
        # Public direct calls must establish vocabulary before variable/optimizer snapshots.
        if phase is None:
            model._check_new_labels(x=dataset, verbose=False)
        old_bounds = None
        try:
            # Use the same wrapper preprocessing and timestep bounds as the public fit method.
            if phase is None:
                old_bounds = (model._active_min_timestep, model._active_max_timestep)
                model.set_timestep_bounds(model.train_noisified_min_timesteps, model.train_noisified_max_timesteps)
                # Existing mapped preprocessing prepares the same training views before batching.
                if model.map_preprocess:
                    # Concurrent RNG-to-row assignment has no atomic iterator/weight boundary.
                    if model.map_num_parallel_calls != 1:
                        raise ValueError("Checkpointed mapped preprocessing requires map_num_parallel_calls=1.")
                    model._preprocess_training = True
                    dataset = dataset.map(model.prep_inputs_map, num_parallel_calls=model.map_num_parallel_calls)
                    model._preprocess_training = None
                    # The wrapper's stateful seed streams are restored separately as model variables.
                    # TensorFlow still saves buffered mapped values and the complete input cursor.
                    data_options = tf.data.Options()
                    data_options.experimental_external_state_policy = tf.data.experimental.ExternalStatePolicy.IGNORE
                    dataset = dataset.with_options(data_options)
            # Keras 3.11.2 owns partial-epoch retention, exhaustion and reshuffle semantics.
            native = TFEpochIterator(x=dataset, steps_per_epoch=options.get("steps_per_epoch"),
                                     distribute_strategy=model.distribute_strategy,
                                     steps_per_execution=model.steps_per_execution)
            model._maybe_symbolic_build(iterator=native)
            native.reset()
            # Build the same optimizer ownership set used by the original route train_step.
            if phase == "acquisition":
                eligible = [value for key in model.classes for value in model.bank.vectors[key]]
            # Consolidation owns its configured student subgraph and temporary predictor.
            elif phase == "consolidation":
                eligible = list(model.wrapper.network.classifier.trainable_variables) if (
                    model.settings.consolidation_scope == "semantic") else list(model.wrapper.network.trainable_variables)
                eligible += model.predictor.trainable_variables
            # Joint fitting retains the wrapper's original network ownership.
            else:
                eligible = model.network.trainable_variables
            model.optimizer.build(eligible)
            # Native fitting clears previous-fit metrics before its first epoch.
            # Their stale totals are neither an initial model identity nor persistent task state.
            model.reset_metrics()
            variables = _variables(model)
            initial = [np.array(value.numpy(), copy=True) for value in variables]
            signature = [[list(value.shape), value.dtype.name] for value in initial]
            previous = self.state["stages"][self.position] if self.position < len(self.state["stages"]) else None
            # Reconstructing prior fits must reach exactly their original starting state.
            if previous is not None and (previous["signature"] != signature or previous["initial"] != fingerprint_state(initial)):
                raise ValueError("Fit checkpoint does not match the reconstructed initial variables.")
            model.stop_training = False
            # Completed fits do not reopen callback resources or repeat callback side effects.
            if previous is not None and previous["complete"]:
                history = tf.keras.callbacks.History()
                history.set_model(model)
                history.history, history.epoch = {}, []
            # Fresh and partially completed fits initialize callback resources once per attempt.
            else:
                callback_list.on_train_begin()
                history = model.history
            stage = deepcopy(previous) if previous is not None else {
                "signature": signature, "initial": fingerprint_state(initial), "changes": {},
                "epoch": initial_epoch, "batch": 0, "complete": False, "history": {}, "epochs": [],
                "callbacks": _callback_state(recovery_callbacks), "phase": None, "rng": capture_rng_state(),
                "seconds": 0., "observer": None, "stop_training": False,
                "iterator": {}, "steps_seen": 0,
            }
            iterator = None
            started, prior_seconds = time.perf_counter(), stage["seconds"]
            excluded_seconds = 0.

            def restore() -> None:
                """Restore a validated stage after callback initialization.

                Returns:
                    restored (None): None; installs numeric variables and local phase
                        sampler state, callbacks, metrics and random streams.

                Raises:
                    ValueError: If callback ownership changed during reconstruction.
                """
                for index, value in stage["changes"].items():
                    variables[int(index)].assign(value)
                _restore_callbacks(recovery_callbacks, stage["callbacks"])
                history.history, history.epoch = deepcopy(stage["history"]), list(stage["epochs"])
                model.stop_training = stage["stop_training"]
                # Phase sampler state is independent of Keras network variables.
                if phase is not None and stage["phase"] is not None:
                    local = deepcopy(stage["phase"])
                    model.rng.bit_generator.state = local.pop("rng")
                    local["focus_counts"] = {int(key): value for key, value in local["focus_counts"].items()}
                    for name, value in local.items():
                        setattr(model, name, value)
                observer = self.owner.experimental_controller
                # Learning-curve callbacks own diagnostics outside the Keras variable graph.
                if observer is not None and stage["observer"] is not None:
                    saved = deepcopy(stage["observer"])
                    observer.curves, observer.curve_seconds = saved["curves"], saved["curve_seconds"]
                    observer.fit_started = time.perf_counter() - saved["fit_elapsed"]
                    observer.started = time.perf_counter() - saved["elapsed"]
                    observer.monitor.previous_segments = saved["resource_segments"]
                restore_rng_state(stage["rng"])
                # TensorFlow's native iterator state also restores shared shuffle seed resources.
                # Completed stages need this because later controller operations reread the same pool.
                if stage["iterator"]:
                    native._current_iterator = iter(native._get_iterator())
                    native._current_iterator._restore_from_tensors({
                        key: tf.io.parse_tensor(value, out_type=tf.variant)
                        for key, value in stage["iterator"].items()})
                    native._steps_seen = stage["steps_seen"]

            def commit(complete: bool = False) -> None:
                """Commit a completed optimizer step or fit boundary through common.

                Args:
                    complete (bool): Whether all callbacks and epoch history for this
                        fit have completed successfully.

                Returns:
                    committed (None): None; retains exact state changes and current
                        iterator, excluding checkpoint I/O from training-time budgets.

                Raises:
                    Exception: Propagates state encoding or atomic writer failures.
                """
                nonlocal excluded_seconds
                # Drain in-flight native mapping before reading its wrapper-owned random counters.
                if native._current_iterator is not None:
                    stage["iterator"] = {key: tf.io.serialize_tensor(value).numpy()
                                         for key, value in native._current_iterator._serialize_to_tensors().items()}
                stage["changes"] = {str(index): np.array(value.numpy(), copy=True) for index, value in enumerate(variables)
                                    if not np.array_equal(value.numpy(), initial[index])}
                stage.update(complete=complete, history=deepcopy(history.history), epochs=list(history.epoch),
                             callbacks=_callback_state(recovery_callbacks), rng=capture_rng_state(),
                             seconds=prior_seconds + time.perf_counter() - started - excluded_seconds,
                             stop_training=bool(model.stop_training))
                stage["steps_seen"] = native._steps_seen
                # Explicit sampler counters preserve focus order and every committed trace row.
                if phase is not None:
                    stage["phase"] = {
                        "step_number": model.step_number, "focus_cycle": list(model.focus_cycle),
                        "focus_counts": {str(key): value for key, value in model.focus_counts.items()},
                        "updated_names": set(model.updated_names), "example_draws": model.example_draws,
                        "view_draws": model.view_draws, "trace": [dict(row) for row in model.trace],
                        "rng": deepcopy(model.rng.bit_generator.state),
                    }
                observer = self.owner.experimental_controller
                # Copy observer callback records so resumed fits do not duplicate measured epochs.
                if observer is not None:
                    stage["observer"] = deepcopy({"curves": observer.curves,
                        "curve_seconds": observer.curve_seconds,
                        "fit_elapsed": time.perf_counter() - observer.fit_started,
                        "elapsed": time.perf_counter() - observer.started,
                        "resource_segments": [*getattr(observer.monitor, "previous_segments", []),
                                              observer.monitor.snapshot(include_previous=False)]})
                # This stage is appended once, then replaced by later immutable commits.
                if self.position == len(self.state["stages"]):
                    self.state["stages"].append(stage)
                # Later commits replace the active entry while retaining earlier completed fits.
                else:
                    self.state["stages"][self.position] = stage
                saved_started = time.perf_counter()
                self.writer(self.state, native._current_iterator)
                elapsed = time.perf_counter() - saved_started
                excluded_seconds += elapsed
                self.state["checkpoint_seconds"] += elapsed
                # Observer clocks use the same active-work scope as phase budget callbacks.
                if observer is not None:
                    observer.fit_started += elapsed
                    observer.started += elapsed
                for callback in callbacks:
                    # Time-budget controls count active training, not persistence overhead.
                    if hasattr(callback, "elapsed") and hasattr(callback, "started"):
                        callback.started += elapsed

            # Completed calls restore their final state and return the original history.
            if previous is not None and stage["complete"]:
                restore()
                self.position += 1
                model._checkpoint_elapsed_seconds = stage["seconds"]
                return history
            # Restore the native iterator, including shuffle resources and partial-epoch counters.
            if previous is not None:
                restore()
            first = stage["batch"]
            logs = {}
            model.make_train_function()
            remaining_epochs = () if model.stop_training and first == 0 else range(stage["epoch"], epochs)
            for epoch in remaining_epochs:
                # A partially committed epoch retains metric accumulators and callback state.
                if first == 0:
                    model.reset_metrics()
                    callback_list.on_epoch_begin(epoch)
                logs = model._get_metrics_result_or_logs({}) if first else {}
                configured_steps = native.steps_per_epoch
                # A saved partial epoch retains its current iterator even when ordinary epochs reshuffle.
                if first:
                    native.steps_per_epoch = max(steps - first, 1)
                try:
                    with native.catch_stop_iteration():
                        batches = () if model.stop_training else native
                        for begin, end, iterator in batches:
                            batch = first + begin
                            # A committed stopping decision does not permit an extra update after restart.
                            if model.stop_training or batch >= steps:
                                break
                            callback_list.on_train_batch_begin(batch)
                            logs = model.train_function(iterator)
                            callback_list.on_train_batch_end(first + end, logs)
                            previous_batch = stage["batch"]
                            stage["epoch"], stage["batch"] = epoch, first + end + 1
                            # Epoch boundaries commit after validation; intermediate commits bound update loss.
                            if (stage["batch"] < steps and stage["batch"] // self.owner.checkpoint_interval
                                    > previous_batch // self.owner.checkpoint_interval):
                                commit()
                            # Break at the same yielded batch as native Keras when a callback stops.
                            if model.stop_training:
                                break
                finally:
                    native.steps_per_epoch = configured_steps
                logs = model._get_metrics_result_or_logs(logs)
                # Validation uses the existing wrapper evaluation and clean timestep rules.
                frequency = options.get("validation_freq", 1)
                validate = (epoch + 1) % frequency == 0 if isinstance(frequency, int) else epoch + 1 in frequency
                # Honor integer or explicit-epoch validation schedules through the original evaluator.
                if options.get("validation_data") is not None and validate:
                    validation = model.evaluate(options["validation_data"], verbose=0, return_dict=True,
                                                steps=options.get("validation_steps"), callbacks=callback_list)
                    logs.update({"val_" + name: value for name, value in validation.items()})
                callback_list.on_epoch_end(epoch, logs)
                stage["epoch"], stage["batch"] = epoch + 1, 0
                first = 0
                # Epoch commits preserve complete history even below the update interval.
                commit()
                # Preserve the callback's completed stopping decision at the epoch boundary.
                if model.stop_training:
                    break
            # Match native optimizer finalization before callbacks may restore their best weights.
            if isinstance(model.optimizer, tf.keras.optimizers.Optimizer) and epochs > 0:
                model.optimizer.finalize_variable_values(model.trainable_weights)
            callback_list.on_train_end(logs)
            commit(complete=True)
            model._checkpoint_elapsed_seconds = stage["seconds"]
            self.position += 1
            return history
        finally:
            # Match the public wrapper fit's timestep cleanup on success and interruption.
            if old_bounds is not None:
                model._preprocess_training = None
                model.set_timestep_bounds(*old_bounds)
