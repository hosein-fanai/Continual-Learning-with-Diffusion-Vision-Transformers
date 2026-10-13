"""TMCL VI/MI sessions using the existing DiT, trainer and recovery core.

This entry point deliberately does not call the legacy joint-fit controller.
It implements representation pretraining, modulation acquisition and current-
image consolidation. Classification readouts are separate downstream objects.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from common.dataloader import get_dataset
from common.recovery import (
    capture_rng_state, fingerprint_state, load_task_checkpoint, 
    restore_rng_state, save_task_checkpoint
)
from common.runtime import derive_seed
from common.study_artifacts import source_files
from common.train import train_model
from semantic_consolidation.controller import _json_value, weight_digest
from semantic_consolidation.tmcl_adapter import TMCLDiTAdapter
from semantic_consolidation.tmcl_config import TMCLSettings
from semantic_consolidation.tmcl_optimizers import TMCLAdamW
from semantic_consolidation.tmcl_phases import TMCLPhase, build_tmcl_heads


class _PhaseRate(tf.keras.callbacks.Callback):
    """Set the paper's one-based virtual learning-rate clock before each update."""

    def __init__(self, schedule: object, offset: int) -> None:
        """Bind a session schedule and its completed representation prefix."""

        super().__init__()
        self.schedule = schedule
        self.offset = offset

    def on_train_batch_begin(self, batch: int, logs: dict | None = None) -> None:
        """Exclude intervening acquisition updates from the representation clock."""

        step = self.offset + self.model.step_number + 1
        optimizers = [self.model.optimizer, getattr(self.model, "mi_optimizer", None), 
                      getattr(self.model, "si_optimizer", None), 
                      *getattr(self.model, "gate_optimizers", {}).values()]
        for optimizer in optimizers:
            # A disabled MI branch need not own an optimizer.
            if optimizer is not None:
                optimizer.learning_rate.assign(self.schedule(step))


class TMCLTrainer:
    """Own completed-session state while reusing native project components.

    Supply an already built float32 DiT wrapper and a fixed disjoint class
    schedule. ``run_session`` receives current unlabelled images separately
    from the current labelled subset. It never retrieves prior images and
    never fits the joint classifier or denoiser. Inputs use the wrapper's
    model coordinates; call ``wrapper.preprocess`` on raw pixels first.
    """

    def __init__(self, wrapper: object, task_classes: list[list[int]], 
                 settings: TMCLSettings | None = None) -> None:
        """Bind architecture, full schedule and source identity before learning."""

        self.settings = settings if settings is not None else TMCLSettings()
        self.task_classes = [list(group) for group in task_classes]
        self.class_order = [label for group in self.task_classes for label in group]
        # A fixed unambiguous class schedule is also the recovery identity.
        if (not self.task_classes or any(not group for group in self.task_classes)
                or any(type(label) is not int or label < 0 for label in self.class_order)
                or len(self.class_order) != len(set(self.class_order))):
            raise ValueError("TMCL needs a nonempty disjoint schedule of integer classes.")
        self.adapter = TMCLDiTAdapter(
            wrapper, modulation_init_std=self.settings.modulation_init_std, seed=self.settings.seed
        )
        self.records: list[dict[str, object]] = []
        self.heads: dict[str, tf.keras.Model] | None = None
        self.representation_optimizer: tf.keras.optimizers.Optimizer | None = None
        self.mi_optimizer: tf.keras.optimizers.Optimizer | None = None
        self.si_optimizer: tf.keras.optimizers.Optimizer | None = None
        self.previous: dict[str, object] | None = None
        self.feature_dim: int | None = None
        self._in_progress = False
        network_config = _json_value(self.adapter.network.get_config())
        # Keras assigns process-local display suffixes; they do not identify model mathematics.
        network_config.pop("name", None)
        self.identity = fingerprint_state({
            "method": "tmcl_dit_vi_mi_v1", "settings": asdict(self.settings), 
            "network": network_config, "tasks": self.task_classes, 
            "network_type": type(self.adapter.network).__module__ + "." + type(self.adapter.network).__qualname__, 
            "wrapper_type": type(wrapper).__module__ + "." + type(wrapper).__qualname__, 
            "preprocess_type": wrapper.preprocess_type, 
            "source": source_files()
        })

    def _build_representation_optimizer(self) -> None:
        """Register stable backbone/projector objects once and preserve Adam moments."""

        # The same native parameter identities persist across pretraining and all sessions.
        if self.representation_optimizer is not None:
            return
        groups = {
            "representation_optimizer": self.adapter.backbone_variables + list(self.heads["vi"].trainable_variables), 
            "mi_optimizer": list(self.heads["mi"].trainable_variables) + list(self.heads["predictor"].trainable_variables)
        }
        # SI heads begin their own Adam clock only when a previous session exists.
        if self.settings.si_weight != 0:
            groups["si_optimizer"] = list(self.heads["si_predictor"].trainable_variables)
        for name, variables in groups.items():
            optimizer = TMCLAdamW(
                learning_rate=self.settings.learning_rate * self.settings.batch_size / 256., 
                weight_decay=self.settings.weight_decay
            )
            optimizer.exclude_from_weight_decay(var_list=[value for value in variables if len(value.shape) == 1])
            optimizer.build(variables)
            setattr(self, name, optimizer)

    def _fit(self, phase: TMCLPhase, steps: int, epoch_steps: int, 
             schedule_steps: int, schedule_offset: int, verbose: int | str) -> dict[str, object]:
        """Fit exact updates using stable representation moments and a phase clock."""

        # Skipped phases neither allocate an optimizer nor consume random draws.
        if steps == 0:
            return {"updates": 0, "seconds": 0., "history": {}, "example_draws": 0, "view_draws": 0}
        acquisition = phase.phase == "acquisition"
        peak = (self.settings.acquisition_learning_rate if acquisition else self.settings.learning_rate) * self.settings.batch_size / 256.
        minimum = self.settings.acquisition_min_learning_rate if acquisition else self.settings.min_learning_rate
        warmup = self.settings.warmup_epochs * epoch_steps
        # A shortened qualification must explicitly choose a shorter warmup.
        if warmup >= schedule_steps and warmup != 0:
            raise ValueError("warmup_epochs must be shorter than the session learning-rate trajectory.")
        rate = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0. if warmup else peak, 
            decay_steps=schedule_steps - warmup, alpha=minimum / peak, 
            warmup_target=peak if warmup else None, warmup_steps=warmup
        )
        # Newly introduced gates have no prior optimizer moments; old gates are never selected.
        if acquisition:
            phase.gate_optimizers = {
                class_id: TMCLAdamW(learning_rate=peak, weight_decay=0.)
                for class_id in phase.classes
            }
            optimizer = next(iter(phase.gate_optimizers.values()))
        # Backbone and all projector moments survive phase and session boundaries.
        else:
            self._build_representation_optimizer()
            optimizer = self.representation_optimizer
            phase.mi_optimizer = self.mi_optimizer
            phase.si_optimizer = self.si_optimizer
        initial_iterations = int(optimizer.iterations.numpy())
        phase.compile(optimizer=optimizer, run_eagerly=True)
        ticks = get_dataset(
            np.arange(steps, dtype="int32"), batch_size=1, 
            shuffle_buffer=0, drop_remainder=False, prefetch=False
        )
        started = time.perf_counter()
        history = train_model(
            None, phase, ticks, save_config_=False, 
            extra_callbacks=[_PhaseRate(rate, schedule_offset)], epochs=1, results_path=None, 
            show_images=True, save_gifs=False, report_every_epoch=False, 
            save_weights=False, use_tensorboard=False, verbose=verbose
        )
        updates = (sum(int(value.iterations.numpy()) for value in phase.gate_optimizers.values())
                   if acquisition else int(optimizer.iterations.numpy()) - initial_iterations)
        # A skipped or incomplete optimizer update invalidates the declared budget.
        if updates != steps or phase.step_number != steps:
            raise RuntimeError(f"Expected {steps} {phase.phase} updates, observed {updates}.")
        return {
            "updates": updates, "seconds": time.perf_counter() - started, "history": history, 
            "optimizer_total_updates": (sum(int(value.iterations.numpy()) for value in phase.gate_optimizers.values())
                                        if acquisition else int(optimizer.iterations.numpy())), 
            "optimizer_initial_updates": initial_iterations, 
            "learning_rate_schedule_steps": schedule_steps, "learning_rate_offset": schedule_offset, 
            "example_draws": phase.example_draws, "view_draws": phase.view_draws, 
            "focus_class_updates": dict(phase.focus_counts), 
            "gradient_variable_names": sorted(phase.updated_names), "trace": list(phase.trace)
        }

    def run_session(self, current_images: np.ndarray, labelled_images: np.ndarray, 
                    labelled_labels: np.ndarray, verbose: int | str = 0) -> dict[str, object]:
        """Learn the next session using current images and its labelled subset.

        No labels are passed to pretraining or consolidation. The labelled
        subset can be much smaller; all phase epoch lengths still use the
        full ``current_images`` count. Only complete sessions are recoverable.
        A failed partial session requires restoration of the previous boundary.
        """

        session = len(self.records)
        # Never silently rerun completed work or train over a failed partial session.
        if self._in_progress or session >= len(self.task_classes):
            raise RuntimeError("Restore a completed boundary or supply an unfinished schedule.")
        images = np.asarray(current_images, dtype="float32")
        labelled = np.asarray(labelled_images, dtype="float32")
        labels = np.asarray(labelled_labels)
        classes = self.task_classes[session]
        # Later SI sessions require the exact preceding representation, not a fresh target.
        if session and self.settings.si_weight != 0 and self.previous is None:
            raise RuntimeError("SI requires a restored preceding-session representation.")
        # Tensor geometry and label alignment are mathematical input contracts.
        if (images.ndim != 4 or images.shape[-1] != 3 or len(images) < 2 or labelled.ndim != 4
                or labelled.shape[1:] != images.shape[1:] or labels.ndim != 1
                or len(labelled) != len(labels) or not np.issubdtype(labels.dtype, np.integer)
                or not np.isfinite(images).all() or not np.isfinite(labelled).all()):
            raise ValueError("Supply finite NHWC current/labelled images and aligned integer labels.")
        budgets = self.settings.budgets(len(images), first_session=session == 0)
        # Current-task labels can select only newly acquired gates, never old images.
        if budgets["acquisition"] and (set(labels.tolist()) != set(classes)
                or len(classes) < 2 or any(np.sum(labels == value) < 2 for value in classes)):
            raise ValueError("Acquisition needs at least two labelled rows per new class and two classes.")
        epoch_steps = self.settings.steps_per_epoch(len(images))
        # Validate coupled schedules before pretraining or bank changes.
        representation_steps = budgets["pretrain"] + budgets["consolidation"]
        # Each nonempty learning-rate trajectory must include a decay interval.
        if any(0 < count <= self.settings.warmup_epochs * epoch_steps
               for count in (representation_steps, budgets["acquisition"])):
            raise ValueError("Executed learning-rate trajectories must be longer than warmup_epochs.")
        features = self.adapter.features(tf.convert_to_tensor(images[:2]), training=False)
        self.feature_dim = int(features.shape[-1])
        # Allocate stable head objects before registering persistent optimizer ownership.
        if self.heads is None and representation_steps:
            self.heads = build_tmcl_heads(self.feature_dim, self.settings, seed=self.settings.seed)
        self._in_progress = True
        record = {
            "session": session + 1, "new_classes": list(classes), 
            "current_examples": len(images), "labelled_examples": len(labelled), 
            "steps_per_epoch": epoch_steps, "settings": asdict(self.settings), 
            "method": "tmcl_dit_vi_mi_v1", "image_replay": False, 
            "representation_input": "clean, timestep zero, CFG null condition", 
            "classifier_readout": "not fitted by representation phases", 
            "input_sha256": {
                key: fingerprint_state({"shape": list(value.shape), "dtype": str(value.dtype), 
                                        "bytes": hashlib.sha256(value.tobytes()).hexdigest()})
                for key, value in (("current", images), ("labelled", labelled), ("labels", labels))
            }
        }
        # Pretraining occurs once and contains VI only, before acquiring any gates.
        if budgets["pretrain"]:
            pretrain = TMCLPhase(
                self.adapter, images, None, [], self.settings, "pretrain", 
                heads=self.heads, seed=derive_seed(self.settings.seed, session, "tmcl_pretrain")
            )
            record["pretrain"] = self._fit(pretrain, budgets["pretrain"], epoch_steps, representation_steps, 0, verbose)
            self.heads = pretrain.heads
        # Later sessions do not repeat the initial pretraining prefix.
        else:
            record["pretrain"] = {"updates": 0, "seconds": 0.}
        bank = self.adapter.bank
        old_digest = weight_digest(bank.variables_for())
        old_classes = bank.classes
        network_digest = weight_digest(self.adapter.network.variables)
        # The VI-only ablation retains no unused randomly initialized modulation memory.
        if self.settings.mi_weight != 0:
            bank.add(classes)
        # Sparse labels affect acquisition batches, never the declared epoch length.
        if budgets["acquisition"]:
            acquisition = TMCLPhase(
                self.adapter, labelled, labels, classes, self.settings, "acquisition", 
                heads=self.heads, seed=derive_seed(self.settings.seed, session, "tmcl_acquisition")
            )
            record["acquisition"] = self._fit(acquisition, budgets["acquisition"], epoch_steps, budgets["acquisition"], 0, verbose)
        # Skipping acquisition is an explicit budget/MI ablation.
        else:
            record["acquisition"] = {"updates": 0, "seconds": 0.}
        # Frozen encoder state and independently retained old gates must remain exact.
        if (network_digest != weight_digest(self.adapter.network.variables)
                or old_digest != weight_digest(bank.variables_for(old_classes))):
            raise RuntimeError("Acquisition changed the frozen network or an old class modulation.")
        bank_digest = weight_digest(bank.variables_for())
        previous_variables = ([] if self.previous is None else
                              list(self.previous["adapter"].network.variables) + list(self.previous["projector"].variables))
        previous_digest = weight_digest(previous_variables)
        # Consolidation draws all retained gates independently of current image labels.
        if budgets["consolidation"]:
            consolidation = TMCLPhase(
                self.adapter, images, None, bank.classes, self.settings, "consolidation", 
                heads=self.heads, previous=self.previous, 
                seed=derive_seed(self.settings.seed, session, "tmcl_consolidation")
            )
            record["consolidation"] = self._fit(
                consolidation, budgets["consolidation"], epoch_steps, representation_steps, budgets["pretrain"], verbose
            )
            self.heads = consolidation.heads
        # Zero consolidation is kept visible as an explicit incomplete-method ablation.
        else:
            record["consolidation"] = {"updates": 0, "seconds": 0.}
        # Optimizer momentum and weight decay may not change frozen modulation memory.
        if bank_digest != weight_digest(bank.variables_for()):
            raise RuntimeError("Consolidation changed a frozen class modulation.")
        # Past-model normalization state is part of the stopped SI target identity.
        if previous_digest != weight_digest(previous_variables):
            raise RuntimeError("Consolidation changed the frozen previous-session SI target.")
        record["retained_classes"] = bank.classes
        record["invariants"] = {
            "acquisition_network_frozen": True, "old_gates_frozen": True, 
            "consolidation_gates_frozen": True, "previous_si_target_frozen": True
        }
        # Preserve the learned VI projector before resetting any session-end head values.
        if self.settings.si_weight != 0:
            self.previous = self._snapshot_previous()
        # Paper Appendix A resets all heads in place; backbone optimizer moments persist.
        if self.settings.reset_projectors and self.heads is not None:
            fresh = build_tmcl_heads(
                self.feature_dim, self.settings, seed=derive_seed(self.settings.seed, session, "tmcl_reset_heads")
            )
            for key, head in self.heads.items():
                head.set_weights(fresh[key].get_weights())
        record = _json_value(record)
        self.records.append(record)
        self._in_progress = False
        return record

    def _snapshot_previous(self) -> dict[str, object]:
        """Clone the native representation and VI projector for stopped SI targets."""

        network = self.adapter.wrapper.snapshot_teacher_network("raw")
        previous_adapter = TMCLDiTAdapter(
            SimpleNamespace(network=network), modulation_init_std=self.settings.modulation_init_std, 
            seed=self.settings.seed
        )
        projector = build_tmcl_heads(self.feature_dim, self.settings, seed=self.settings.seed)["vi"]
        projector.set_weights(self.heads["vi"].get_weights())
        projector.trainable = False
        return {"adapter": previous_adapter, "projector": projector}

    def _trackables(self) -> dict[str, object]:
        """Use common recovery's complete Keras-variable and bank ownership graph."""

        previous = ({"previous_network": self.previous["adapter"].network, 
                     "previous_projector": self.previous["projector"]} if self.previous is not None else {})
        return {"network": self.adapter.network, "bank": self.adapter.bank, 
                "representation_optimizer": self.representation_optimizer, 
                "mi_optimizer": self.mi_optimizer, 
                "si_optimizer": self.si_optimizer, 
                **{f"head_{key}": value for key, value in (self.heads or {}).items()}, **previous}

    def save(self, checkpoint_root: str | Path) -> Path:
        """Atomically save a completed session using common task recovery."""

        # Partial fits are not advertised as restartable session boundaries.
        if self._in_progress or not self.records:
            raise RuntimeError("Only completed TMCL sessions can be saved.")
        return save_task_checkpoint(
            checkpoint_root, len(self.records) - 1, 
            {"records": self.records, "bank": self.adapter.bank.snapshot(), 
             "feature_dim": self.feature_dim, "has_heads": self.heads is not None, 
             "has_previous": self.previous is not None}, 
            trackables=self._trackables(), class_order=self.class_order, task_groups=self.task_classes, 
            rng_state=capture_rng_state(), fingerprint=self.identity
        )

    def restore(self, checkpoint_path: str | Path) -> None:
        """Restore a completed session with strict source/configuration matching."""

        # Restore into an unused trainer to avoid merging incompatible partial histories.
        if self.records or self._in_progress:
            raise RuntimeError("Create a fresh TMCLTrainer before restoring a boundary.")
        checkpoint = load_task_checkpoint(
            checkpoint_path, expected_class_order=self.class_order, 
            expected_task_groups=self.task_classes, expected_fingerprint=self.identity
        )
        state = checkpoint.state
        self._in_progress = True
        self.adapter.bank.restore(state["bank"])
        self.feature_dim = state["feature_dim"]
        # A no-reset recipe retains its trained projection/normalization states.
        if state["has_heads"]:
            self.heads = build_tmcl_heads(self.feature_dim, self.settings, seed=self.settings.seed)
            self._build_representation_optimizer()
        # The checkpoint restores this independent target after matching its native structure.
        if state["has_previous"]:
            self.previous = self._snapshot_previous()
        load_task_checkpoint(
            checkpoint.task_dir, trackables=self._trackables(), expected_class_order=self.class_order, 
            expected_task_groups=self.task_classes, expected_fingerprint=self.identity
        )
        restore_rng_state(checkpoint.rng_state)
        self.records = state["records"]
        self._in_progress = False
