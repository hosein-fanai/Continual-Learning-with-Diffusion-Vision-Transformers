"""Explicit paper-derived TMCL budgets, separate from the legacy joint route.

Defaults follow Appendix A of https://arxiv.org/html/2505.14125v3. The native
DiT backbone, TensorFlow augmentations and 2048-wide projectors are declared
adaptations. The released launcher instead uses 50 acquisition epochs, a .024
Barlow scale and no projector reset; those are not silently mixed into defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class TMCLSettings:
    """Representation-only learning; epoch lengths use all current images.

    Zero epochs explicitly skip a phase for bounded checks or ablations.
    ``mi_weight=0`` selects the matched view-invariance baseline. Images are
    clean model-coordinate inputs, not diffusion-corrupted inputs. No joint
    classifier CE or denoising objective is implied by these settings.
    """

    pretrain_epochs: int = 250
    acquisition_epochs: int = 100
    consolidation_epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 0.001
    acquisition_learning_rate: float = 0.01
    weight_decay: float = 0.0001
    modulation_decay_start: float = 0.4
    modulation_decay_end: float = 0.04
    min_learning_rate: float = 0.000001
    acquisition_min_learning_rate: float = 0.00001
    warmup_epochs: int = 10
    augmentation_views: int = 4
    projector_hidden: int = 2048
    projector_dim: int = 2048
    predictor_hidden: int = 2048
    vi_weight: float = 1.0
    mi_weight: float = 1.0
    si_weight: float = 0.0
    si_method: str = "cassle"
    si_objective: str = "barlow"
    si_redundancy_weight: float = 0.0051
    si_barlow_scale: float = 0.1
    si_negative_weight: float = 1.0
    barlow_scale: float = 0.1
    redundancy_weight: float = 0.005
    modulation_init_std: float = 0.02
    reset_projectors: bool = True
    seed: int = 101

    def __post_init__(self) -> None:
        """Reject ambiguous counts and unsupported branch combinations."""

        for name in ("pretrain_epochs", "acquisition_epochs", "consolidation_epochs", "warmup_epochs"):
            value = getattr(self, name)
            # Update budgets must count complete epochs exactly.
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        for name in ("batch_size", "augmentation_views", "projector_hidden", "projector_dim", "predictor_hidden"):
            value = getattr(self, name)
            # Correlation and view pairing need nontrivial batch/view dimensions.
            minimum = 2 if name in ("batch_size", "augmentation_views") else 1
            # Reject dimensions that cannot define the selected objective.
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        # State-invariance choices name distinct, explicitly implemented objectives.
        if self.si_method not in ("cassle", "pnr") or self.si_objective not in ("barlow", "ghosh"):
            raise ValueError("SI requires cassle/pnr and barlow/ghosh objective names.")
        # A phase without either representation objective cannot update its encoder.
        if self.vi_weight == 0 and self.mi_weight == 0:
            raise ValueError("At least one of VI and MI must be enabled.")
        # Initial representation pretraining contains VI alone.
        if self.vi_weight == 0 and self.pretrain_epochs != 0:
            raise ValueError("VI-free ablations must explicitly disable VI pretraining.")
        # The sparse OPL update must contain two positive rows and at least one negative.
        if self.mi_weight != 0 and self.acquisition_epochs and self.batch_size < 3:
            raise ValueError("Acquisition requires batch_size >= 3.")
        # SI compares learned VI projections and is defined only during consolidation.
        if self.si_weight != 0 and (self.vi_weight == 0 or self.consolidation_epochs == 0):
            raise ValueError("SI requires active VI and a nonzero consolidation budget.")
        # The released Ghosh PNR cross-pair construction uses exactly four views.
        if (self.si_weight != 0 and self.si_method == "pnr"
                and self.si_objective == "ghosh" and self.augmentation_views != 4):
            raise ValueError("Ghosh PNR requires exactly four augmentation views.")
        # Session reset policy is a scientific choice, never a truthy string.
        if type(self.reset_projectors) is not bool:
            raise TypeError("reset_projectors must be boolean.")
        # Named reproducible streams require an exact integer master seed.
        if type(self.seed) is not int:
            raise TypeError("seed must be an integer.")

    def steps_per_epoch(self, current_examples: int) -> int:
        """Size virtual epochs by all current images, including unlabelled rows."""

        # Empty or fractional cohorts cannot define the paper's virtual epoch.
        if type(current_examples) is not int or current_examples < 2:
            raise ValueError("current_examples must be an integer >= 2.")
        return ceil(current_examples / self.batch_size)

    def budgets(self, current_examples: int, first_session: bool) -> dict[str, int]:
        """Return exact update budgets without using the sparse labelled count."""

        steps = self.steps_per_epoch(current_examples)
        return {
            "pretrain": self.pretrain_epochs * steps if first_session else 0, 
            "acquisition": self.acquisition_epochs * steps if self.mi_weight != 0 else 0, 
            "consolidation": self.consolidation_epochs * steps
        }
