"""Stop numerically divergent fits without ranking finite HPO candidates."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import tensorflow as tf


class TrainingDiverged(FloatingPointError):
    """A non-finite logged loss, with JSON-safe partial training evidence.

    ``evidence`` contains zero-based epoch/batch indices where available. The
    offending value is a string so JSON never contains NaN or Infinity numbers.
    ``evidence_path`` is None when no file was requested or writing failed;
    evidence remains available to the caller even after a file-system error.
    """

    def __init__(self, evidence: Mapping[str, object], evidence_path: Path | None = None):
        self.evidence = deepcopy(dict(evidence))
        self.evidence_path = evidence_path
        self.phase = self.evidence["phase"]
        self.epoch = self.evidence["epoch"]
        self.batch = self.evidence["batch"]
        self.metric = self.evidence["metric"]
        super().__init__(
            f"Non-finite {self.metric}={self.evidence['value']} during "
            f"{self.phase} ({self.evidence['hook']}, "
            f"epoch={self.epoch}, batch={self.batch}; indices are zero-based)."
        )


class NonFiniteLossGuard(tf.keras.callbacks.Callback):
    """Raise ``TrainingDiverged`` for scalar NaN/Inf losses at epoch end.

    Only ``loss`` and names ending in ``_loss`` are termination signals. Finite
    losses, including very large ones, are left untouched. No Optuna reporting
    or ranking is performed. Only epoch-end logs are checked, including
    aggregate training and validation losses; batches are not inspected.

    Args:
        phase: Explicit fit phase, or infer ``model._train_part`` at fit start
            and use ``joint`` when the model has no named phase.
        evidence_dir: Optional output directory. On divergence, atomically write
            ``hpo-divergence-<phase>.json``. Use a distinct directory per trial.

    Evidence records finite epoch history from the current fit, finite metrics
    from the preceding successful epoch, and finite metrics beside the offending
    value. Each ``on_train_begin`` resets state, including when V2 reuses the
    callback for its next phase. Raising bypasses Keras ``on_train_end`` cleanup;
    this callback owns no persistent file handle or TensorBoard writer.
    """

    def __init__(
        self, *, phase: str | None = None, evidence_dir: str | Path | None = None
    ) -> None:
        super().__init__()
        if phase is not None and (not isinstance(phase, str) or not phase.strip()):
            raise ValueError("phase must be a nonempty string or None.")
        self._explicit_phase = phase
        self.evidence_dir = None if evidence_dir is None else Path(evidence_dir)
        self.phase = phase or "joint"
        self.epoch: int | None = None
        self.partial_history: list[dict[str, object]] = []
        self.last_finite_metrics: dict[str, float] = {}

    def on_train_begin(self, logs=None) -> None:
        del logs
        inferred = getattr(self.model, "_train_part", None)
        self.phase = self._explicit_phase or (
            inferred if isinstance(inferred, str) and inferred else "joint"
        )
        self.epoch = None
        self.partial_history = []
        self.last_finite_metrics = {}

    def _check(self, logs) -> dict[str, float]:
        finite: dict[str, float] = {}
        invalid_losses: list[tuple[str, float]] = []
        for name, value in (logs or {}).items():
            if not isinstance(name, str) or value is None or isinstance(value, (str, bytes)):
                continue
            # TensorFlow's direct __array__ conversion can reject bfloat16;
            # eager callback tensors expose the supported NumPy form explicitly.
            try:
                array = np.asarray(value.numpy() if tf.is_tensor(value) else value)
            except (TypeError, ValueError):
                continue
            if array.ndim != 0 or np.iscomplexobj(array):
                continue
            try:
                scalar = float(array)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(scalar):
                finite[name] = scalar
            elif name == "loss" or name.endswith("_loss"):
                invalid_losses.append((name, scalar))

        if invalid_losses:
            metric, value = invalid_losses[0]
            evidence = {
                "reason": "nonfinite_loss",
                "phase": self.phase,
                "epoch": self.epoch,
                "batch": None,
                "hook": "epoch_end",
                "metric": metric,
                "value": str(value),
                "nonfinite_losses": {name: str(number) for name, number in invalid_losses},
                "finite_metrics": finite,
                "last_finite_metrics": dict(self.last_finite_metrics),
                "partial_history": deepcopy(self.partial_history),
            }
            evidence_path = None
            if self.evidence_dir is not None:
                phase_name = re.sub(r"[^A-Za-z0-9_.-]", "_", self.phase)
                destination = self.evidence_dir / f"hpo-divergence-{phase_name}.json"
                try:
                    self.evidence_dir.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(".json.tmp")
                    temporary.write_text(
                        json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8"
                    )
                    temporary.replace(destination)
                    evidence_path = destination
                except OSError as error:
                    evidence["evidence_write_error"] = f"{type(error).__name__}: {error}"
            raise TrainingDiverged(evidence, evidence_path)

        if finite:
            self.last_finite_metrics = finite
        return finite

    def on_epoch_end(self, epoch, logs=None) -> None:
        self.epoch = int(epoch)
        finite = self._check(logs)
        self.partial_history.append({"epoch": self.epoch, "metrics": finite})


__all__ = ["NonFiniteLossGuard", "TrainingDiverged"]
