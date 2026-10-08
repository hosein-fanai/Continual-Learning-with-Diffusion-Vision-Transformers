"""Forward finite validation epochs to the single HPO pruning coordinator."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import tensorflow as tf

from common.callbacks.hpo_guard import NonFiniteLossGuard
from common.hpo_pruning import TrialPerformancePruned, report_epoch, validate_exchange, write_atomic_json


class EpochPruningCallback(tf.keras.callbacks.Callback):
    """Report validation loss and stop only on the coordinator's matching decision."""

    def __init__(self, exchange: Mapping[str, object], evidence_dir: str | Path | None = None) -> None:
        """Bind one training attempt and optional partial-history artifact location."""

        super().__init__()
        self.exchange = validate_exchange(exchange)
        self.evidence_dir = None if evidence_dir is None else Path(evidence_dir)
        self._guard = NonFiniteLossGuard(phase="generation", evidence_dir=evidence_dir)
        self._last_step = -1

    def on_train_begin(self, logs: Mapping[str, object] | None = None) -> None:
        """Reset partial history at the beginning of the single supported fit."""

        self._guard.set_model(self.model)
        self._guard.on_train_begin(logs)
        self._last_step = -1

    def on_epoch_end(self, epoch: int, logs: Mapping[str, object] | None = None) -> None:
        """Report one finite scalar and retain partial history if its trial is pruned."""

        # Replayed or decreasing epochs cannot be compared as a fresh observation.
        if epoch <= self._last_step:
            raise ValueError("HPO pruning epochs must increase strictly.")
        self._guard.on_epoch_end(epoch, logs)
        metrics = self._guard.partial_history[-1]["metrics"]
        monitor = self.exchange["monitor"]
        # Missing validation cannot silently become a training-loss pruning decision.
        if monitor not in metrics:
            raise ValueError(f"HPO pruning requires finite scalar epoch metric {monitor!r}.")
        self._last_step = int(epoch)
        prune, report = report_epoch(self.exchange, self._last_step, metrics[monitor])
        # Successful reports continue training until a coordinator explicitly prunes.
        if not prune:
            return
        evidence = {
            "reason": "performance_pruning", 
            "phase": "generation", 
            "epoch": self._last_step, 
            "metric": monitor, 
            "value": metrics[monitor], 
            "report": report, 
            "partial_history": deepcopy(self._guard.partial_history)
        }
        evidence_path = None
        # Evidence remains on the exception even if artifact persistence fails.
        if self.evidence_dir is not None:
            try:
                self.evidence_dir.mkdir(parents=True, exist_ok=True)
                destination = self.evidence_dir / "hpo-pruning.json"
                write_atomic_json(destination, evidence)
                evidence_path = destination
            except OSError as error:
                evidence["evidence_write_error"] = f"{type(error).__name__}: {error}"
        raise TrialPerformancePruned(evidence, evidence_path)
