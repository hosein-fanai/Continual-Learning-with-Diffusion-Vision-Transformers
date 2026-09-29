"""Plateau controls for mutable rates and a monotonically advancing cosine clock."""

from __future__ import annotations

import tensorflow as tf

import math

import warnings

from collections.abc import Mapping


@tf.keras.utils.register_keras_serializable(package="continual_learning")
class OffsetCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule, tf.Module):
    """Cosine decay whose independent virtual clock can jump forward on a plateau.

    The optimizer's iteration counter is never changed. ``offset`` is a tracked
    variable, so an already traced training graph observes subsequent jumps.
    Serialization retains its current value and creates an independent variable.
    """

    def __init__(
        self, 
        initial_learning_rate: float, 
        decay_steps: int, 
        min_learning_rate: float = 0.0, 
        offset: float = 0.0, 
        name: str = "offset_cosine_decay"
    ) -> None:
        """Create a float32 cosine schedule with a checkpointed clock offset.

        Args:
            initial_learning_rate (float): Rate at virtual step zero, above zero.
            decay_steps (int): Positive virtual duration; later steps use the floor.
            min_learning_rate (float): Final rate in [0, initial_learning_rate].
            offset (float): Nonnegative initial virtual steps added to actual steps.
            name (str): TensorFlow module name used for the tracked clock variable.

        Returns:
            None: Stores scalar settings and a nontrainable float32 offset.

        Raises:
            ValueError: A setting makes the cosine interval or floor invalid.
        """

        tf.Module.__init__(self, name=name)

        values = (
            initial_learning_rate, 
            decay_steps, 
            min_learning_rate, 
            offset
        )

        # Nonfinite settings would contaminate every schedule evaluation.
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(
                "Cosine rate, duration, floor and offset must be finite."
            )
        # Keep the virtual clock forward-moving and its decay duration defined.
        if decay_steps <= 0 or initial_learning_rate <= 0 or offset < 0:
            raise ValueError(
                "Cosine rate/duration must be positive and offset nonnegative."
            )
        # The cosine must decay from the initial rate to a valid lower floor.
        if not 0 <= min_learning_rate <= initial_learning_rate:
            raise ValueError(
                "min_learning_rate must lie between zero and the initial rate."
                )

        self.initial_learning_rate = float(initial_learning_rate)
        self.decay_steps = float(decay_steps)
        self.min_learning_rate = float(min_learning_rate)
        self.offset = tf.Variable(
            float(offset), 
            trainable=False, 
            dtype=tf.float32, 
            name="virtual_step_offset"
        )

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        """Evaluate the cosine at actual steps plus the current virtual offset.

        Args:
            step (tf.Tensor): Nonnegative integer or floating scalar/array of
                optimizer steps, converted to float32. The offset is broadcast.

        Returns:
            tf.Tensor: Float32 rates with the input shape; virtual steps at or
            beyond decay_steps produce min_learning_rate. No state is changed.
        """

        step = tf.minimum(tf.cast(step, tf.float32) + self.offset, self.decay_steps)
        fraction = 0.5 * (1.0 + tf.cos(math.pi * step / self.decay_steps))

        return self.min_learning_rate + (
            self.initial_learning_rate -
            self.min_learning_rate
        ) * fraction

    def jump(
        self, 
        step: int | tf.Tensor, 
        factor: float, 
        min_learning_rate: float = 0.0
    ) -> float:
        """Advance the virtual clock to reduce the current rate by ``factor``.

        Args:
            step (int | tf.Tensor): Current scalar optimizer iteration, read eagerly.
            factor (float): Multiplicative reduction strictly between zero and one.
            min_learning_rate (float): Nonnegative floor for this jump. A floor
                above the current rate leaves the rate and clock unchanged.

        Returns:
            float: New rate at step. The tracked offset can only increase; the
            optimizer iteration and momentum/variance slots are never modified.
            Subsequent steps continue decaying toward the schedule's own floor.

        Raises:
            ValueError: The reduction factor or requested floor is invalid.
        """

        # A jump must reduce the rate without introducing an invalid floor.
        if not 0 < factor < 1 or not math.isfinite(min_learning_rate) \
        or min_learning_rate < 0:
            raise ValueError(
                "Require 0 < factor < 1 and a nonnegative finite rate floor."
            )

        actual_step = float(tf.keras.backend.get_value(step))
        current = float(self(actual_step).numpy())
        floor = max(self.min_learning_rate, float(min_learning_rate))
        target = max(floor, current * factor)

        # Never increase the rate or invert a constant cosine schedule.
        if target >= current or \
        self.initial_learning_rate == self.min_learning_rate:
            return current

        ratio = (target - self.min_learning_rate) / (
            self.initial_learning_rate -
            self.min_learning_rate
        )

        virtual_step = self.decay_steps * math.acos(
            max(-1.0, min(1.0, 2 * ratio - 1))
        ) / math.pi
        self.offset.assign(max(
            float(self.offset.numpy()), 
            virtual_step - actual_step
        ))

        return float(self(actual_step).numpy())

    def get_config(self) -> dict:
        """Return scalar settings and the eagerly read float32 clock offset.

        Returns:
            dict[str, float | str]: Constructor arguments that recreate an
            independent schedule at the same offset, including the module name.
        """

        return {
            "initial_learning_rate": self.initial_learning_rate, 
            "decay_steps": self.decay_steps, 
            "min_learning_rate": self.min_learning_rate, 
            "offset": float(self.offset.numpy()), 
            "name": self.name
        }


class PlateauLearningRate(tf.keras.callbacks.Callback):
    """Reduce a scalar rate or jump an ``OffsetCosineDecay`` after a plateau."""

    def __init__(
        self, 
        monitor: str = "val_loss", 
        patience: int = 5, 
        factor: float = 0.5, 
        min_learning_rate: float = 1e-6, 
        mode: str = "auto", 
        min_delta: float = 0.0, 
        verbose: int = 0
    ) -> None:
        """Configure a per-fit plateau detector and its rate reduction.

        Args:
            monitor (str): Scalar key read from epoch-end logs.
            patience (int): Consecutive non-improving epochs before reducing; zero
                reduces on the first non-improvement. Stored without coercion.
            factor (float): Rate multiplier after each plateau; callers choose a
                finite value in (0, 1) for a reduction.
            min_learning_rate (float): Nonnegative floor for a reduction; an
                already lower rate is preserved.
            mode (str): 'min' or 'max'; 'auto' maximizes names containing acc/auc.
            min_delta (float): Tolerance applied directly as best +/- min_delta;
                conventionally nonnegative, but stored without clipping.
            verbose (int): Nonzero prints actual reductions with one-based epochs.

        Returns:
            None: Best score and wait count initialize empty and reset each fit.
            Numeric domains are the caller's responsibility. Float controls are
            converted to Python float; cosine jumps enforce their own constraints.

        Raises:
            ValueError: mode is not one of min, max, or auto.
        """

        super().__init__()
        # The direction chooses which observations count as improvements.
        if mode not in ("min", "max", "auto"):
            raise ValueError("Require mode min/max/auto.")

        self.monitor, self.patience, self.factor = monitor, patience, float(factor)
        self.min_learning_rate, self.min_delta = float(min_learning_rate), float(min_delta)
        self.mode, self.verbose = mode, verbose
        self.wait = 0
        self.best = None

    def on_train_begin(self, logs: Mapping[str, object] | None = None) -> None:
        """Reset the best score/wait count and check the effective optimizer rate.

        Args:
            logs (Mapping[str, object] | None): Keras start-of-fit logs, unused.

        Returns:
            None: Records the optimization direction without changing rates,
            optimizer iterations, slots, or a cosine schedule's offset.

        Raises:
            ValueError: The effective optimizer uses a schedule other than
                OffsetCosineDecay, which cannot honor plateau jumps.
        """

        self.wait, self.best = 0, None
        self._maximize = self.mode == "max" or (
            self.mode == "auto" and
            any(name in self.monitor.lower()
            for name in ("acc", "auc"))
        )

        optimizer = self.model.optimizer
        # Mixed-float16 compilation wraps the optimizer that owns the actual rate.
        while isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            optimizer = optimizer.inner_optimizer
        rate = getattr(optimizer, "_learning_rate", None)
        # Ordinary immutable schedules cannot honor an in-place plateau reduction.
        if isinstance(rate, tf.keras.optimizers.schedules.LearningRateSchedule) and \
        not isinstance(rate, OffsetCosineDecay):
            raise ValueError(
                "Plateau reduction with a schedule requires optimizer.plateau_jump=True."
            )

    def on_epoch_end(self, epoch: int, logs: dict[str, object] | None = None) -> None:
        """Update patience and reduce the effective rate after a plateau.

        Args:
            epoch (int): Zero-based epoch, used only in verbose output.
            logs (dict[str, object] | None): Mutable Keras scalar metrics. Missing
                monitor values warn and leave patience unchanged; nonfinite values
                count as non-improvements. A reduction adds learning_rate to logs.

        Returns:
            None: May update the best score, wait count, scalar rate or cosine
            offset. Loss-scale wrappers retain their scale and optimizer slots.
        """

        # Missing observations neither improve the best score nor consume patience.
        if logs is None or self.monitor not in logs:
            warnings.warn(
                f"PlateauLearningRate cannot find metric {self.monitor!r}.", 
                stacklevel=2
            )

            return

        current = float(logs[self.monitor])
        improved = math.isfinite(current) and (self.best is None or (
            current > self.best + self.min_delta if
            self._maximize else current < self.best - self.min_delta
        ))
        # A finite improvement starts a fresh plateau window.
        if improved:
            self.best, self.wait = current, 0
            return

        self.wait += 1
        # Continue accumulating non-improving epochs until patience is exhausted.
        if self.wait < self.patience:
            return

        optimizer = self.model.optimizer
        # Change the inner rate, not LossScaleOptimizer's unused placeholder rate.
        while isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            optimizer = optimizer.inner_optimizer
        schedule = getattr(optimizer, "_learning_rate", None)
        old_rate = float(tf.keras.backend.get_value(optimizer.learning_rate))
        # Advance the tracked cosine clock without resetting optimizer iterations.
        if isinstance(schedule, OffsetCosineDecay):
            new_rate = schedule.jump(optimizer.iterations, self.factor, self.min_learning_rate)
        # Mutable scalar rates are reduced directly and never raised to the floor.
        else:
            new_rate = min(old_rate, max(self.min_learning_rate, old_rate * self.factor))
            optimizer.learning_rate.assign(new_rate)

        self.wait = 0
        logs["learning_rate"] = new_rate
        # Announce only actual rate reductions when verbose output is enabled.
        if self.verbose and new_rate < old_rate:
            print(
                f"Epoch {epoch + 1}: plateau reduced learning rate to {new_rate:.6g}.", 
                flush=True
            )


class ValidationEnsembleAccuracy(tf.keras.callbacks.Callback):
    """Add the configured held-out ensemble score before stopping/logging callbacks."""

    def __init__(self, validation_data: tf.data.Dataset, 
                 **ensemble_kwargs: object) -> None:
        """Store held-out batches and options for the wrapper's ensemble evaluator.

        Args:
            validation_data (tf.data.Dataset): Batched image/label pairs in the
                attached diffusion classifier's input dtype and label convention.
            **ensemble_kwargs (object): Keyword options forwarded unchanged to
                evaluate_ensemble_accuracy, including seed/timestep controls.
                verbose defaults to False when omitted.

        Returns:
            None: Stores the dataset by reference and a shallow options copy.
        """

        super().__init__()

        self.validation_data = validation_data
        self.ensemble_kwargs = dict(ensemble_kwargs)
        self.ensemble_kwargs.setdefault("verbose", False)

    def on_epoch_end(self, epoch: int, logs: dict[str, object] | None = None) -> None:
        """Add a held-out ensemble score before downstream epoch controls run.

        Args:
            epoch (int): Keras zero-based epoch, unused by the evaluator.
            logs (dict[str, object] | None): Mutable epoch metrics receiving a
                Python float under val_ensemble_accuracy when a mapping is present.

        Returns:
            None: Generator-only phases are skipped. Other phases invoke the
            attached model's evaluator even if logs is None; its RNG behavior
            follows ensemble_kwargs and the wrapper's sampling contract.
        """

        # Generator-only phases have no held-out classifier ensemble to score.
        if getattr(self.model, "_train_part", None) == "generator":
            return

        score = self.model.evaluate_ensemble_accuracy(
            self.validation_data, 
            **self.ensemble_kwargs
        )
        # Publish the score for subsequent epoch callbacks through Keras shared logs.
        if logs is not None:
            logs["val_ensemble_accuracy"] = float(score)
