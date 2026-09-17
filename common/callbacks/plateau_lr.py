"""Plateau controls for mutable rates and a monotonically advancing cosine clock."""

from __future__ import annotations

import tensorflow as tf

import math

import warnings


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
        tf.Module.__init__(self, name=name)

        values = (
            initial_learning_rate, 
            decay_steps, 
            min_learning_rate, 
            offset
        )

        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(
                "Cosine rate, duration, floor and offset must be finite."
            )
        if decay_steps <= 0 or initial_learning_rate <= 0 or offset < 0:
            raise ValueError(
                "Cosine rate/duration must be positive and offset nonnegative."
            )
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
        """Evaluate cosine decay at the actual step plus the current offset."""

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
        """Advance the virtual clock to reduce the current rate by ``factor``."""

        if not 0 < factor < 1 or not math.isfinite(min_learning_rate) \
        or min_learning_rate < 0:
            raise ValueError(
                "Require 0 < factor < 1 and a nonnegative finite rate floor."
            )

        actual_step = float(tf.keras.backend.get_value(step))
        current = float(self(actual_step).numpy())
        floor = max(self.min_learning_rate, float(min_learning_rate))
        target = max(floor, current * factor)

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
        """Return reconstructable settings and the current virtual-clock offset."""

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
        super().__init__()
        if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
            raise ValueError("patience must be a positive integer.")
        if not 0 < factor < 1 or not math.isfinite(min_learning_rate) or min_learning_rate < 0:
            raise ValueError(
                "Require 0 < factor < 1 and a nonnegative finite rate floor."
            )
        if mode not in ("min", "max", "auto") or min_delta < 0 or not math.isfinite(min_delta):
            raise ValueError(
                "Require mode min/max/auto and nonnegative finite min_delta."
            )

        self.monitor, self.patience, self.factor = monitor, patience, float(factor)
        self.min_learning_rate, self.min_delta = float(min_learning_rate), float(min_delta)
        self.mode, self.verbose = mode, verbose
        self.wait = 0
        self.best = None

    def on_train_begin(self, logs=None) -> None:
        """Start fresh plateau bookkeeping for this fit without rewinding its clock."""

        self.wait, self.best = 0, None
        self._maximize = self.mode == "max" or (
            self.mode == "auto" and 
            any(name in self.monitor.lower() 
            for name in ("acc", "auc"))
        )

        rate = getattr(self.model.optimizer, "_learning_rate", None)
        if isinstance(rate, tf.keras.optimizers.schedules.LearningRateSchedule) and \
        not isinstance(rate, OffsetCosineDecay):
            raise ValueError(
                "Plateau reduction with a schedule requires optimizer.plateau_jump=True."
            )

    def on_epoch_end(self, epoch, logs=None) -> None:
        """Apply a reduction after the configured number of non-improving epochs."""

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
        if improved:
            self.best, self.wait = current, 0
            return

        self.wait += 1
        if self.wait < self.patience:
            return

        optimizer = self.model.optimizer
        schedule = getattr(optimizer, "_learning_rate", None)
        old_rate = float(tf.keras.backend.get_value(optimizer.learning_rate))
        if isinstance(schedule, OffsetCosineDecay):
            new_rate = schedule.jump(optimizer.iterations, self.factor, self.min_learning_rate)
        else:
            new_rate = min(old_rate, max(self.min_learning_rate, old_rate * self.factor))
            optimizer.learning_rate.assign(new_rate)

        self.wait = 0
        logs["learning_rate"] = new_rate
        if self.verbose and new_rate < old_rate:
            print(
                f"Epoch {epoch + 1}: plateau reduced learning rate to {new_rate:.6g}.", 
                flush=True
            )


class ValidationEnsembleAccuracy(tf.keras.callbacks.Callback):
    """Add the configured held-out ensemble score before stopping/logging callbacks."""

    def __init__(self, validation_data, **ensemble_kwargs) -> None:
        super().__init__()

        self.validation_data = validation_data
        self.ensemble_kwargs = dict(ensemble_kwargs)
        self.ensemble_kwargs.setdefault("verbose", False)

    def on_epoch_end(self, epoch, logs=None) -> None:
        """Evaluate only classifier-capable phases and update the shared log mapping."""

        if getattr(self.model, "_train_part", None) == "generator":
            return

        score = self.model.evaluate_ensemble_accuracy(
            self.validation_data, 
            **self.ensemble_kwargs
        )
        if logs is not None:
            logs["val_ensemble_accuracy"] = float(score)
