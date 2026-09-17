"""Three-phase task-boundary orchestration and mechanism diagnostics.

The existing learner supplies the exact current/generated-replay task pool and
owns all task schedules, teachers, class growth, and continual score matrices.
This controller adds only the missing semantic phases and their audit records.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

from common.dataloader import get_dataset
from common.keras_compat import optimizer_iterations
from common.mechanistic import calibration_metrics
from common.runtime import derive_seed
from common.train import train_model
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier
from semantic_consolidation.memory import ClassBalancedPool, ModulationBank, affine_modulation
from semantic_consolidation.diagnostics import (
    balanced_probe, class_geometry, diagnostic_view, gate_coverage,
    numeric_changes, one_vs_rest, probe_batches,
)
from semantic_consolidation.objectives import contrastive_alignment_loss, normalized_features
from semantic_consolidation.experimental_diagnostics import descriptive_linear_cka
from semantic_consolidation.phases import RoutePhase, semantic_features


def weight_digest(variables: list[tf.Variable]) -> str:
    """Hash tensor values and shapes to detect changes, without retaining a copy.

    Args:
        variables (list[tf.Variable]): Ordered eager TensorFlow or Keras variables/tensors;
            each exposes its shape, dtype and numpy value.

    Returns:
        digest (str): Hexadecimal SHA-256 string of ordered shape, dtype and tensor-value
            bytes.

    Raises:
        AttributeError: If an input lacks an eager numpy value.
    """

    digest = hashlib.sha256()
    for variable in variables:
        values = np.asarray(variable.numpy())
        digest.update(str(values.shape).encode())
        digest.update(values.dtype.str.encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def tensor_bytes(variables: list[tf.Variable]) -> int:
    """Count actual tensor bytes, excluding runtime/allocator/Python overhead.

    Args:
        variables (list[tf.Variable]): Ordered eager TensorFlow or Keras variables/tensors;
            each exposes its shape, dtype and numpy value.

    Returns:
        nbytes (int): Integer raw tensor payload bytes; excludes allocator and Python
            overhead.

    Raises:
        TypeError: If a tensor has an unknown shape or unsupported dtype.
    """

    return sum(int(np.prod(variable.shape)) * tf.as_dtype(variable.dtype).size for variable in variables)


def _optimizer_variables(optimizer: object) -> list[tf.Variable]:
    """Read optimizer slots across callable and property-style TensorFlow APIs.

    Args:
        optimizer (object): Compiled optimizer exposing numerical variables and its
            iteration counter.

    Returns:
        variables (list[tf.Variable]): Ordered list of optimizer variables, including
            iterations and existing slots.

    Raises:
        AttributeError: If the optimizer does not expose variables.
    """
    variables = optimizer.variables
    return list(variables() if callable(variables) else variables)


class _ElapsedBudget(tf.keras.callbacks.Callback):
    """Stop extra joint training at the first batch boundary after a time budget."""

    def __init__(self, seconds: float) -> None:
        """Store the measured per-task allowance and initialize elapsed-time evidence.

        Args:
            seconds (float): Finite positive elapsed-time allowance or measured sampling
                duration in seconds, according to the operation.

        Returns:
            initialized (None): None; stores the allowance and clears timer state.

        Raises:
            None: No input validation or fallible numerical work is performed here.
        """
        super().__init__()
        self.seconds = seconds
        self.started = 0.
        self.elapsed = 0.
        self.reached = False

    def on_train_begin(self, logs: dict | None = None) -> None:
        """Start timing when Keras begins the extra joint fit.

        Args:
            logs (dict | None): Optional Keras batch or epoch log mapping; this callback does
                not modify it.

        Returns:
            started (None): None; records the monotonic fit start time.

        Raises:
            None: No configuration or numerical errors are raised by this callback.
        """
        self.started = time.perf_counter()

    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        """Stop only at a completed batch after the measured time allowance.

        Args:
            batch (int): Zero-based completed Keras batch index; only elapsed time selects
                stopping.
            logs (dict | None): Optional Keras batch or epoch log mapping; this callback does
                not modify it.

        Returns:
            updated (None): None; updates elapsed seconds and sets model.stop_training after the
                allowance is reached.

        Raises:
            AttributeError: If Keras has not attached a model when stopping is required.
        """
        self.elapsed = time.perf_counter() - self.started
        # Stop after completed work reaches the allowance; record batch-boundary overshoot.
        if self.elapsed >= self.seconds:
            self.reached = True
            self.model.stop_training = True


def _arrays(dataset: tf.data.Dataset, mapping: dict) -> tuple[np.ndarray, np.ndarray]:
    """Materialize one finite task pool, preserving repeated observations.

    Args:
        dataset (tf.data.Dataset): Finite batched tf.data.Dataset containing raw images,
            sparse original labels and optional binary replay provenance.
        mapping (dict): Original-label to dense introduction-order integer mapping owned by
            the wrapper.

    Returns:
        arrays (tuple[np.ndarray, np.ndarray]): (images, labels): concatenated image ndarray
            preserving its dtype and dense int32 label vector.

    Raises:
        ValueError: If the dataset is unbounded, empty or has unsupported batch structure.
        KeyError: If an original label has no dense mapping.
    """

    cardinality = int(tf.data.experimental.cardinality(dataset).numpy())
    # Route phases require a dataset with known finite cardinality.
    if cardinality < 0:
        raise ValueError("Route phases require a dataset with known finite cardinality.")
    images, labels = [], []
    for batch in dataset.as_numpy_iterator():
        # Expected raw (images, labels[, replay_mask]) batches.
        if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
            raise ValueError("Expected raw (images, labels[, replay_mask]) batches.")
        images.append(np.asarray(batch[0]))
        labels.append(np.asarray(batch[1]).reshape(-1))
    # Route phase data must not be empty.
    if not images:
        raise ValueError("Route phase data must not be empty.")
    x, y = np.concatenate(images), np.concatenate(labels)
    return x, np.asarray([mapping[int(label)] for label in y], dtype="int32")


def representation_statistics(features: np.ndarray) -> dict[str, float | None]:
    """Measure collapse on actual hidden features, not classifier probabilities.

    Args:
        features (np.ndarray): Finite nonempty hidden matrix [N, D]; rows are normalized in
            float32 before float64 summaries.

    Returns:
        statistics (dict[str, float | None]): Mapping of Python float summaries; off-diagonal
            cosine is None for a single row.

    Raises:
        ValueError: If feature rank is invalid.
        tf.errors.InvalidArgumentError: If features are empty or nonfinite.
        np.linalg.LinAlgError: If singular-value decomposition does not converge.
    """

    z = normalized_features(features).numpy().astype("float64")
    centered = z - np.mean(z, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    mass = float(np.sum(singular))
    # A nonconstant centered cloud has a normalized singular spectrum.
    if mass > 1e-12:
        probabilities = singular[singular > 0] / mass
        rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    # An exactly collapsed cloud has no positive effective rank.
    else:
        rank = 0.
    n = len(z)
    cosine = float((np.sum(z @ z.T) - np.sum(z * z)) / (n * (n - 1))) if n > 1 else None
    return {
        "mean_normalized_feature_std": float(np.mean(np.std(z, axis=0))),
        "centered_effective_rank": rank,
        "mean_off_diagonal_cosine": cosine,
    }


def _json_value(value: object) -> object:
    """Emit strict JSON: unavailable numeric statistics become null.

    Args:
        value (object): Nested mappings, sequences, NumPy arrays/scalars or ordinary JSON-
            compatible values.

    Returns:
        converted (object): Nested Python values with NumPy payloads converted and nonfinite
            floats replaced by None.

    Raises:
        RecursionError: If a cyclic or excessively deep input cannot be recursively
            traversed.
    """

    # Convert nested records recursively while retaining their field identities.
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    # Convert nested sequence entries to JSON-compatible values.
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    # Array payloads need recursive conversion to built-in numeric lists.
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    # NumPy scalars must become built-in scalars before JSON encoding.
    if isinstance(value, np.generic):
        return _json_value(value.item())
    # Unavailable numerical statistics are null rather than nonstandard JSON NaN.
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class RouteController:
    """Hold persistent modulators and run isolated phases after each joint fit."""

    def __init__(self, settings: object) -> None:
        """Initialize the persistent gate bank and task-boundary diagnostic state.

        Args:
            settings (object): Validated settings instance for this component; its fields select
                the behavior described above.

        Returns:
            initialized (None): None; starts empty per-task records and no allocated gate bank.

        Raises:
            None: Settings are validated by the route configuration before this constructor.
        """
        self.settings = settings
        self.records: list[dict[str, object]] = []
        self.bank: ModulationBank | None = None
        self.introduced: set[int] = set()
        self._boundary: dict | None = None
        self.diagnostic_seconds = 0.
        # Display controls follow each fit and stay outside scientific checkpoint state.
        self.verbose: bool | int | str = False
        self._diagnostic_stage = "validation"

    def _fit(self, phase: RoutePhase, steps: int, random_control: bool = False) -> dict:
        """Use existing data and training APIs for a fixed number of phase updates.

        Args:
            phase (RoutePhase): Constructed RoutePhase for acquisition or consolidation, sharing
                the live wrapper and finite pool.
            steps (int): Nonnegative phase optimizer-update count. Zero returns a
                zero-work audit; a positive count fits exactly that many phase updates.
            random_control (bool): If True, compute acquisition gradients with zero learning
                rate while retaining the declared update count.

        Returns:
            audit (dict): Mapping with actual updates, timing, loss history, float trace values,
                exposures and variable names; zero steps returns a zero-work audit.

        Raises:
            RuntimeError: If observed optimizer applications differ from the requested steps.
            ValueError: If the phase pool cannot provide valid positive/negative rows.
            tf.errors.InvalidArgumentError: If the phase objective or gradients are nonfinite.
        """

        # Identify semantic phases before their Keras progress and loss output.
        if self.verbose:
            print(f"Semantic {phase.phase}: {steps} optimizer updates.", flush=True)
        # An explicit zero-step phase records no optimizer work or data presentations.
        if steps == 0:
            return {
                "updates": 0, "seconds": 0., "history": {}, "optimizer_bytes": 0,
                "focus_class_updates": phase.focus_counts,
                "untrained_focus_classes": list(phase.classes),
                "example_draws": 0, "semantic_image_noise_draws": 0,
                "supervised_view_draws": 0, "image_noise_draws": 0,
                "gradient_variable_names": [], "trace": [],
                "zero_learning_rate_control": random_control,
            }
        phase.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0. if random_control else self.settings.learning_rate
            ), run_eagerly=True,
        )
        ticks = get_dataset(
            np.arange(steps, dtype="int32"), batch_size=1,
            shuffle_buffer=0, drop_remainder=False, prefetch=False,
        )
        started = time.perf_counter()
        history = train_model(
            None, phase, ticks, save_config_=False, epochs=1, verbose=self.verbose,
            results_path=None, show_images=True, save_gifs=False,
            report_every_epoch=False, save_weights=False, use_tensorboard=False,
        )
        elapsed = getattr(phase, "_checkpoint_elapsed_seconds", time.perf_counter() - started)
        updates = int(optimizer_iterations(phase.optimizer).numpy())
        # Incomplete optimizer work invalidates the declared fixed phase budget.
        if updates != steps:
            raise RuntimeError(f"Expected {steps} {phase.phase} updates, observed {updates}.")
        return {
            "updates": updates, "seconds": elapsed, "history": history,
            "optimizer_bytes": tensor_bytes(_optimizer_variables(phase.optimizer)),
            "example_draws": phase.example_draws,
            "semantic_image_noise_draws": phase.view_draws,
            "image_noise_draws": phase.view_draws + (
                phase.example_draws if phase.phase == "consolidation" else 0
            ),
            "gradient_variable_names": sorted(phase.updated_names),
            "zero_learning_rate_control": random_control, "trace": phase.trace,
            "focus_class_updates": phase.focus_counts,
            "untrained_focus_classes": [c for c, count in phase.focus_counts.items() if count == 0],
            "supervised_view_draws": phase.example_draws if phase.phase == "consolidation" else 0,
        }

    def _probe_data(self, dataset: tf.data.Dataset | None, mapping: dict) -> tuple | None:
        """Keep a reproducible balanced validation subset fixed across all phases.

        Args:
            dataset (tf.data.Dataset | None): Finite batched tf.data.Dataset containing raw
                images, sparse original labels and optional binary replay provenance.
            mapping (dict): Original-label to dense introduction-order integer mapping owned by
                the wrapper.

        Returns:
            probe (tuple | None): Selected (images, dense int32 labels) arrays, or None when no
                validation dataset was supplied.

        Raises:
            ValueError: If supplied validation is empty, unbounded or malformed.
            KeyError: If a validation label is absent from the dense mapping.
        """

        # Without supplied validation, no diagnostic dataset may be substituted.
        if dataset is None:
            return None
        return balanced_probe(*_arrays(dataset, mapping), self.settings)

    def _probe(
        self, wrapper: object, probe: tuple | None, old: set[int],
        target: object, frozen_bank: dict, predictor: object,
        views: list | None = None,
    ) -> tuple[dict, np.ndarray | None, list | None]:
        """Evaluate task-free features plus explicitly labeled, oracle-free gate diagnostics.

        Args:
            wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
                schedules and existing training or inference APIs.
            probe (tuple | None): Optional pair of float32 images and aligned dense integer
                labels selected from held-out validation.
            old (set[int]): Set of dense class IDs introduced before the current task.
            target (object): Independent frozen raw network used only to construct consolidation
                targets; None is valid only without target alignment.
            frozen_bank (dict): Mapping from dense class IDs to detached float32 gain/bias
                tensors [D].
            predictor (object): Temporary task-local projection layer; None compares deployed
                hidden features directly.
            views (list | None): Optional cached list of exact input/noise/target tensors; None
                constructs the comparison views once.

        Returns:
            observation (tuple[dict, np.ndarray | None, list | None]): (metrics,
                hidden_features, views): nested diagnostic dict, optional float32 [N, D] array,
                and optional cached tensor-view list.

        Raises:
            KeyError: If a requested gate is absent from frozen_bank.
            ValueError: If network outputs or supplied probe rows are incompatible.
            tf.errors.InvalidArgumentError: If paired feature or probability calculations are
                invalid.
        """

        started = time.perf_counter()
        # Missing validation produces an unavailable observation instead of fabricated metrics.
        if probe is None:
            if self.verbose:
                print(f"Boundary diagnostics ({self._diagnostic_stage}): validation unavailable.", flush=True)
            return {"split": "unavailable"}, None, None
        x, y = probe
        progress = None
        if self.verbose:
            print(
                f"Boundary diagnostics ({self._diagnostic_stage}): "
                f"evaluating {len(x)} validation samples...",
                flush=True,
            )
            progress = tf.keras.utils.Progbar(
                len(x), verbose=1 if self.verbose == "auto" else int(self.verbose),
                unit_name="sample",
            )
        features, probabilities = [], []
        for start in range(0, len(x), self.settings.batch_size):
            images = tf.convert_to_tensor(x[start:start + self.settings.batch_size])
            times = tf.zeros((len(images),), dtype=tf.int32)
            hidden, predicted = semantic_features(wrapper.network, images, times)
            features.append(hidden.numpy())
            probabilities.append(predicted.numpy())
            if progress is not None:
                progress.update(min(start + self.settings.batch_size, len(x)))
        features, probabilities = np.concatenate(features), np.concatenate(probabilities)
        predictions = probabilities.argmax(axis=1)
        is_old = np.isin(y, list(old))
        confusion = np.zeros((wrapper.network.num_classes,) * 2, dtype="int64")
        np.add.at(confusion, (y, predictions), 1)
        result = {
            "split": "validation", "examples": len(y),
            "input_sha256": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
            "calibration_bins": 15, "calibration": calibration_metrics(probabilities, y, bins=15),
            "old_accuracy": float(np.mean(predictions[is_old] == y[is_old])) if is_old.any() else None,
            "new_accuracy": float(np.mean(predictions[~is_old] == y[~is_old])) if (~is_old).any() else None,
            "per_class_recall": {
                int(c): float(np.mean(predictions[y == c] == c)) for c in np.unique(y)
            },
            "confusion": confusion, "representation": representation_statistics(features),
            "clean_accuracy": float(np.mean(predictions == y)),
            "class_geometry": class_geometry(features, y, old),
            "representation_scope": "predictor-free unmodulated clean classifier hidden projection",
        }
        # Every selected gate is applied to every mixed batch. No row label or
        # batch position chooses its gate; target tensors are cached unchanged.
        if target is not None and frozen_bank:
            coverage = gate_coverage(
                list(frozen_bank), old, self.settings.probe_max_gates,
                derive_seed(self.settings.seed, len(self.records), "alignment_probe_gates"),
            )
            if self.verbose:
                print(
                    f"Boundary diagnostics ({self._diagnostic_stage}): "
                    f"comparing frozen-target features across {len(coverage['gate_ids'])} gates...",
                    flush=True,
                )
            # Construct diagnostic views only once so both endpoints share exact target tensors.
            if views is None:
                views = []
                for batch_id, indices in enumerate(probe_batches(len(x), self.settings.batch_size)):
                    images = tf.convert_to_tensor(x[indices])
                    # Instance contrast requires a negative row in the diagnostic batch.
                    if len(images) < 2:
                        continue
                    for draw, level in enumerate(self.settings.noise_levels):
                        noised, times = diagnostic_view(
                            wrapper, images, level,
                            derive_seed(self.settings.seed, len(self.records), "probe_noise", batch_id, draw),
                        )
                        target_features, _ = semantic_features(target, noised, times)
                        views.append({
                            "images": noised, "times": times, "target_hidden": target_features,
                            "batch_id": batch_id, "noise_level": level,
                            "input_sha256": hashlib.sha256(noised.numpy().tobytes()).hexdigest(),
                            "class_counts": {int(c): int(np.sum(y[indices] == c)) for c in np.unique(y[indices])},
                        })
            comparisons, predicted_stats = [], []
            for view in views:
                hidden, _ = semantic_features(wrapper.network, view["images"], view["times"])
                predicted = hidden if predictor is None else predictor(hidden, training=False)
                predicted_stats.append(representation_statistics(predicted.numpy()))
                for focus in coverage["gate_ids"]:
                    target_features = view["target_hidden"]
                    # The unmodulated-distillation control omits the target gate explicitly.
                    if self.settings.condition != "unmodulated_feature_distillation":
                        target_features = affine_modulation(target_features, *frozen_bank[focus], self.settings)
                    row = {
                        "gate_id": focus, "gate_group": "old" if focus in old else "new",
                        "batch_id": view["batch_id"], "noise_level": view["noise_level"],
                        "examples": len(hidden), "negatives_per_row": len(hidden) - 1,
                        "class_counts": view["class_counts"],
                        "input_sha256": view["input_sha256"],
                    }
                    for name, value in (("hidden", hidden), ("predictor", predicted)):
                        row[name + "_infonce"] = float(contrastive_alignment_loss(
                            value, target_features, temperature=self.settings.temperature
                        ).numpy())
                        row[name + "_target_cosine"] = float(tf.reduce_mean(tf.reduce_sum(
                            normalized_features(value) * normalized_features(target_features), axis=1
                        )).numpy())
                    comparisons.append(row)
            aggregates = {}
            for group in ("selected_gates", "old", "new"):
                rows = [r for r in comparisons if group == "selected_gates" or r["gate_group"] == group]
                count = sum(r["examples"] for r in rows)
                aggregates[group] = {
                    "example_gate_noise_comparisons": count, "batch_gate_noise_comparisons": len(rows),
                    **{name: sum(r[name] * r["examples"] for r in rows) / count if count else None
                       for name in ("hidden_infonce", "hidden_target_cosine", "predictor_infonce", "predictor_target_cosine")},
                }
            result["frozen_target_alignment"] = {
                "gate_coverage": coverage, "comparisons": comparisons, "aggregates": aggregates,
                "aggregation": "example-weighted batches, equal selected gates and noise levels; no reliability weights",
                "predictor_convention": "same temporary predictor evaluated at both endpoints; training-only measurement",
                "deployed_representation_metrics": ["hidden_infonce", "hidden_target_cosine"],
                "targets_modulated": self.settings.condition != "unmodulated_feature_distillation",
                "predictor_representation_by_view": predicted_stats,
                "views": len(views),
            }
        elapsed = time.perf_counter() - started
        self.diagnostic_seconds += elapsed
        if self.verbose:
            measures = [f"accuracy={result['clean_accuracy']:.4f}"]
            for group in ("old", "new"):
                value = result[f"{group}_accuracy"]
                measures.append(f"{group}_accuracy={value:.4f}" if value is not None else f"{group}_accuracy=unavailable")
            print(
                f"Boundary diagnostics ({self._diagnostic_stage}) finished in {elapsed:.1f}s: "
                + " - ".join(measures),
                flush=True,
            )
        return result, features, views

    def _functional(self, wrapper: object, probe: tuple | None, old: set[int], gates: list[int]) -> tuple:
        """Clean old-gate behavior on one fixed cohort, without a teacher or predictor.

        Args:
            wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
                schedules and existing training or inference APIs.
            probe (tuple | None): Optional pair of float32 images and aligned dense integer
                labels selected from held-out validation.
            old (set[int]): Set of dense class IDs introduced before the current task.
            gates (list[int]): Dense gate IDs to evaluate on every supplied held-out row.

        Returns:
            observation (tuple): (metrics, features): functional dict and optional float32
                hidden array; missing validation is reported explicitly.

        Raises:
            KeyError: If a requested gate does not exist.
            ValueError: If supplied probe features or probabilities are incompatible.
        """

        unmodulated, features, _ = self._probe(wrapper, probe, old, None, {}, None)
        # Functional gate comparisons require an observed held-out hidden representation.
        if features is None:
            return {"unmodulated": unmodulated, "gates": {}, "availability": "validation_unavailable"}, None
        started = time.perf_counter()
        labels = probe[1]
        measured = {}
        for focus in gates:
            modulated = self.bank.apply(features, focus).numpy()
            measured[focus] = {
                "modulated": one_vs_rest(modulated, labels, focus),
                "unmodulated": one_vs_rest(features, labels, focus),
            }
        self.diagnostic_seconds += time.perf_counter() - started
        return {"unmodulated": unmodulated, "gates": measured, "availability": "measured"}, features

    def before_joint(self, wrapper: object, validation: tf.data.Dataset | None) -> None:
        """Capture bounded numeric features after head expansion, before any joint fit.

        Args:
            wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
                schedules and existing training or inference APIs.
            validation (tf.data.Dataset | None): Optional finite held-out validation dataset
                with original labels; None leaves probe measurements unavailable.

        Returns:
            captured (None): None; caches the fixed validation cohort, pre-joint features and
                old-gate digest.

        Raises:
            ValueError: If supplied validation cannot form a finite mapped cohort.
            KeyError: If validation labels are unmapped.
        """

        self.diagnostic_seconds = 0.
        self._diagnostic_stage = "before joint training"
        if self.verbose:
            print("Boundary diagnostics: selecting validation probes before joint training...", flush=True)
        started = time.perf_counter()
        probe = self._probe_data(validation, wrapper.seen_classes)
        old = set(self.introduced)
        available = sorted(old & set(self.bank.vectors)) if self.bank is not None else []
        coverage = gate_coverage(available, old, self.settings.probe_max_gates,
                                 derive_seed(self.settings.seed, len(self.records), "functional_probe_gates"))
        stage, features = self._functional(wrapper, probe, old, coverage["gate_ids"])
        self._boundary = {
            "probe": probe, "pre_joint": stage, "pre_features": features,
            "gate_coverage": coverage, "old_gate_parameters": self._old_gate_digest(old),
            "unavailable_old_gate_ids": sorted(old - set(available)),
            "before_joint_seconds": time.perf_counter() - started,
        }
        self.diagnostic_seconds = self._boundary["before_joint_seconds"]

    def _old_gate_digest(self, old: set[int]) -> str:
        """Hash existing old parameters independently of their measured behavior.

        Args:
            old (set[int]): Set of dense class IDs introduced before the current task.

        Returns:
            digest (str): SHA-256 of existing old-gate variables in sorted class order,
                including the empty-bank digest.

        Raises:
            AttributeError: If a stored gate does not expose an eager numpy value.
        """

        variables = [v for c in sorted(old) if self.bank is not None and c in self.bank.vectors
                     for v in self.bank.vectors[c]]
        return weight_digest(variables)

    @staticmethod
    def _functional_change(first: dict, last: dict, before: np.ndarray | None, after: np.ndarray | None) -> dict:
        """Separate unmodulated drift and class-geometry changes at fixed inputs.

        Args:
            first (dict): Earlier nested functional diagnostic mapping on the fixed validation
                cohort.
            last (dict): Later nested functional diagnostic mapping on the same validation
                cohort.
            before (np.ndarray | None): Optional earlier float32 hidden matrix [N, D].
            after (np.ndarray | None): Optional later aligned float32 hidden matrix [N, D].

        Returns:
            changes (dict): Nested numeric deltas and optional hidden-feature drift/CKA; missing
                endpoints stay unavailable.

        Raises:
            ValueError: If supplied hidden arrays do not preserve row correspondence.
        """

        # A missing endpoint makes phase drift unavailable rather than zero.
        if before is None or after is None:
            return {"availability": "boundary_or_validation_unavailable",
                    "hidden_feature_cka": None, "hidden_feature_cka_sample_count": None,
                    "hidden_feature_cka_unavailable_reason": "boundary_or_validation_unavailable"}
        cka = descriptive_linear_cka(before, after)
        return {
            "definition": "after minus before on identical clean held-out inputs",
            "hidden_feature_cka": cka["linear_cka"],
            "hidden_feature_cka_sample_count": cka["sample_count"],
            "hidden_feature_cka_unavailable_reason": cka["linear_cka_unavailable_reason"],
            "hidden_mean_squared_change": float(np.mean((after - before) ** 2)),
            "class_geometry": numeric_changes(first["unmodulated"]["class_geometry"], last["unmodulated"]["class_geometry"]),
            "gates": numeric_changes(first["gates"], last["gates"]),
        }

    def _finish_boundary(self, wrapper: object, record: dict, boundary: dict, post: dict,
                         post_features: np.ndarray | None, endpoint: str = "post_consolidation") -> None:
        """Report the third boundary and release the persistent diagnostic cache.

        Args:
            wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
                schedules and existing training or inference APIs.
            record (dict): Mutable per-task diagnostic mapping receiving measured counts,
                boundaries and resource fields.
            boundary (dict): Cached pre-joint probe, feature arrays, gate coverage and parameter
                digest for the current task.
            post (dict): Measured post-joint functional diagnostic mapping for the same held-out
                cohort.
            post_features (np.ndarray | None): Optional float32 post-joint hidden feature matrix
                [N, D] aligned with the cached probe.
            endpoint (str): JSON field name for the final measured boundary, normally
                post_consolidation or post_extra_joint.

        Returns:
            finished (None): None; adds final functional measurements and resource fields, then
                releases the boundary cache.

        Raises:
            ValueError: If the cached cohort or aligned features are incompatible.
        """

        started, previous_seconds = time.perf_counter(), self.diagnostic_seconds
        old = set(self.introduced)
        self._diagnostic_stage = "after consolidation" if endpoint == "post_consolidation" else "final task boundary"
        final, features = self._functional(wrapper, boundary["probe"], old, boundary["gate_coverage"]["gate_ids"])
        record["functional_drift"] = {
            "scope": "fixed permitted validation cohort; clean inputs; pre_joint is after head expansion",
            "gate_coverage": boundary["gate_coverage"],
            "unavailable_old_gate_ids": boundary["unavailable_old_gate_ids"],
            "pre_joint": boundary["pre_joint"], "post_joint": post, endpoint: final,
            "joint_change": self._functional_change(boundary["pre_joint"], post, boundary["pre_features"], post_features),
            "consolidation_change" if endpoint == "post_consolidation" else "extra_joint_change":
                self._functional_change(post, final, post_features, features),
            "old_gate_parameters_unchanged_since_pre_joint": (
                boundary["old_gate_parameters"] == self._old_gate_digest(old)
                if boundary["old_gate_parameters"] is not None and boundary["gate_coverage"]["available_gate_ids"] else None
            ),
        }
        record["before_joint_diagnostic_seconds"] = boundary["before_joint_seconds"]
        # Include change/CKA reductions as well as the nested final probe, once.
        self.diagnostic_seconds = previous_seconds + time.perf_counter() - started
        record["diagnostic_seconds"] = self.diagnostic_seconds
        record["cached_boundary_feature_bytes"] = sum(
            features.nbytes for features in (boundary["pre_features"], post_features) if features is not None
        )
        self._boundary = None
        if self.verbose:
            print(f"Route boundary diagnostics complete: {self.diagnostic_seconds:.1f}s total.", flush=True)

    def run(self, wrapper: object, dataset: tf.data.Dataset, fit_kwargs: dict) -> dict:
        """Execute missing phases before common evaluates or snapshots this increment.

        Args:
            wrapper (object): Live diffusion classifier exposing its raw network, class mapping,
                schedules and existing training or inference APIs.
            dataset (tf.data.Dataset): Finite batched tf.data.Dataset containing raw images,
                sparse original labels and optional binary replay provenance.
            fit_kwargs (dict): Existing fit arguments, including optional validation_data and
                callbacks; explicit schedules own update budgets.

        Returns:
            record (dict): Strict-JSON-compatible per-task dict of phases, controls, invariants,
                diagnostics, memory and committed updates.

        Raises:
            ValueError: If a semantic task introduces no classes, lacks pool coverage/pairs or
                changes projection width.
            RuntimeError: If a frozen target, old gate, prior teacher, backbone boundary or
                declared update budget is violated.
        """

        started = time.perf_counter()
        self.verbose = fit_kwargs.get("verbose", self.verbose)
        settings = self.settings
        joint_updates = int(fit_kwargs.get("route_joint_updates", 0))
        seen = set(range(wrapper.network.num_classes))
        new = sorted(seen - self.introduced)
        old = set(self.introduced)
        record = {
            "diagnostic_schema_version": 2,
            "task": len(self.records) + 1, "condition": settings.condition,
            "new_classes": new, "seen_classes": sorted(seen),
            "joint_updates": joint_updates, "acquisition": {"updates": 0},
            "consolidation": {"updates": 0}, "extra_joint_updates": 0,
            "consolidation_scope": settings.consolidation_scope,
            "gradient_boundary": (
                "eligible: primary classifier projection/head and temporary predictor; shared denoising backbone frozen"
                if settings.consolidation_scope == "semantic" else
                "eligible: connected classifier/shared-backbone variables and temporary predictor; backbone ablation"
            ),
            "noise_pairing": "same image and same noise tensor for student/target at each level; no cross-noise alignment",
        }
        boundary = self._boundary
        if boundary is None:
            # Direct controller callers cannot retrospectively measure pre-joint
            # behavior. Still report the observable later boundaries honestly.
            self.before_joint(wrapper, fit_kwargs.get("validation_data"))
            boundary = self._boundary
            boundary.update(pre_joint={"availability": "pre_joint_hook_not_called"},
                            pre_features=None, old_gate_parameters=None, before_joint_seconds=0.)
        probe = boundary["probe"]
        self._diagnostic_stage = "after joint training"
        post_joint, post_features = self._functional(wrapper, probe, old, boundary["gate_coverage"]["gate_ids"])
        # Joint training changed old modulation parameters.
        if boundary["old_gate_parameters"] is not None and boundary["old_gate_parameters"] != self._old_gate_digest(old):
            raise RuntimeError("Joint training changed old modulation parameters.")
        # Platform and extra-joint controls preserve the original joint training objective.
        if settings.condition in ("baseline", "extra_joint", "time_matched_joint"):
            record["gradient_boundary"] = "consolidation not run; original joint training boundaries apply"
            # The extra-training controls spend their declared semantic-phase allowance on joint fits.
            if settings.condition in ("extra_joint", "time_matched_joint"):
                steps = settings.acquisition_steps + settings.consolidation_steps
                before = int(optimizer_iterations(wrapper.optimizer).numpy())
                extra_started = time.perf_counter()
                # Measured-time controls stop at batch boundaries instead of prescribing update counts.
                if settings.condition == "time_matched_joint":
                    budget = settings.extra_joint_seconds[len(self.records)]
                    stopper = _ElapsedBudget(budget)
                    extra_fit = getattr(wrapper, "fit_joint", None) or (lambda *args, **options: DiffusionClassifier.fit(wrapper, *args, **options))
                    # Name the time-budgeted control before its training progress.
                    if self.verbose:
                        print(f"Extra joint training: {budget:g}s budget.", flush=True)
                    extra_history = extra_fit(
                        dataset, epochs=1_000_000, verbose=self.verbose, callbacks=[stopper],
                    )
                    # Time-control epoch safety cap reached before its budget.
                    if not stopper.reached:
                        raise RuntimeError("Time-control epoch safety cap reached before its budget.")
                    record["extra_joint_requested_seconds"] = budget
                    record["extra_joint_training_seconds"] = stopper.elapsed
                    record["extra_joint_overshoot_seconds"] = max(0., stopper.elapsed - budget)
                    record["extra_joint_history"] = extra_history.history
                elif steps:
                    # Call the existing wrapper's fit, bypassing only the route adapter.
                    finite = dataset.repeat().take(steps)
                    extra_fit = getattr(wrapper, "fit_joint", None) or (lambda *args, **options: DiffusionClassifier.fit(wrapper, *args, **options))
                    # Name the fixed-update control before its training progress.
                    if self.verbose:
                        print(f"Extra joint training: {steps} optimizer updates.", flush=True)
                    extra_history = extra_fit(
                        finite, epochs=1, verbose=self.verbose, steps_per_epoch=steps,
                    )
                    record["extra_joint_history"] = extra_history.history
                record["extra_joint_seconds"] = (getattr(wrapper, "_checkpoint_elapsed_seconds", time.perf_counter() - extra_started)
                                                  if steps or settings.condition == "time_matched_joint"
                                                  else time.perf_counter() - extra_started)
                record["extra_joint_updates"] = int(optimizer_iterations(wrapper.optimizer).numpy()) - before
                # Extra-joint control did not consume its exact update budget.
                if settings.condition == "extra_joint" and record["extra_joint_updates"] != steps:
                    raise RuntimeError("Extra-joint control did not consume its exact update budget.")
            self._finish_boundary(wrapper, record, boundary, post_joint, post_features, "post_extra_joint")
            record["total_updates"] = joint_updates + record["extra_joint_updates"]
            record["route_seconds"] = time.perf_counter() - started
            self.introduced = seen
            self.records.append(_json_value(record))
            return self.records[-1]

        x, y = _arrays(dataset, wrapper.seen_classes)
        pool = ClassBalancedPool(x, y)
        # Each route call must introduce new classes; arbitrary refits are unsupported.
        if not new:
            raise ValueError("Each route call must introduce new classes; arbitrary refits are unsupported.")
        # The task pool must cover all seen classes; increase replay exposure.
        if set(pool.indices) != seen and settings.retain_modulators:
            raise ValueError("The task pool must cover all seen classes; increase replay exposure.")
        focus_classes = seen if settings.retain_modulators else set(new)
        insufficient = {class_id: len(pool.indices.get(class_id, ()))
                        for class_id in sorted(focus_classes)
                        if len(pool.indices.get(class_id, ())) < 2}
        # Validate every future focus before any gate is initialized or updated.
        if insufficient:
            raise ValueError(f"Every semantic focus class needs two positive rows; observed {insufficient}.")
        first = tf.convert_to_tensor(x[:2], dtype=tf.float32)
        hidden, _ = semantic_features(wrapper.network, first, tf.zeros((len(first),), tf.int32))
        # Initialize the persistent gate bank once the actual projection width is available.
        if self.bank is None:
            self.bank = ModulationBank(settings, int(hidden.shape[-1]), settings.seed)
        # Semantic projection dimension must remain fixed across tasks.
        if self.bank.dimension != int(hidden.shape[-1]):
            raise ValueError("Semantic projection dimension must remain fixed across tasks.")
        self.bank.add(new)
        missing = set(pool.indices) - set(self.bank.vectors)
        # True-class CE control requires retained modulators for every pool label.
        if missing and settings.acquisition_objective == "true_class_ce":
            raise ValueError("True-class CE control requires retained modulators for every pool label.")
        old_variables = [v for c in old if c in self.bank.vectors for v in self.bank.vectors[c]]
        old_digest = weight_digest(old_variables)
        network_digest = weight_digest(wrapper.network.weights)
        prior_teacher = wrapper.teacher_network
        prior_digest = weight_digest(prior_teacher.weights) if prior_teacher is not None else None
        acquisition = RoutePhase(
            wrapper, self.bank, pool, settings, "acquisition", new,
            derive_seed(settings.seed, len(self.records), "acquisition"),
        )
        record["acquisition"] = self._fit(
            acquisition, settings.acquisition_steps, random_control=settings.condition == "random"
        )
        # Modulation acquisition changed the frozen acquired network.
        if network_digest != weight_digest(wrapper.network.weights):
            raise RuntimeError("Modulation acquisition changed the frozen acquired network.")
        # Acquisition changed old modulation variables.
        if old_digest != weight_digest(old_variables):
            raise RuntimeError("Acquisition changed old modulation variables.")
        # Snapshot AFTER acquisition and keep it separate from the prior-task teacher.
        snapshot_started = time.perf_counter()
        target = wrapper.snapshot_teacher_network("raw")
        record["target_snapshot_seconds"] = time.perf_counter() - snapshot_started
        # Consolidation requires a separate, frozen target network.
        if target is wrapper.network or target is prior_teacher or target.trainable:
            raise RuntimeError("Consolidation requires a separate, frozen target network.")
        target_digest = weight_digest(target.weights)
        frozen_bank = self.bank.frozen()
        bank_digest = weight_digest([v for pair in self.bank.vectors.values() for v in pair])
        semantic_ids = {id(variable) for variable in wrapper.network.classifier.weights}
        fixed_variables = [v for v in wrapper.network.weights if id(v) not in semantic_ids]
        fixed_digest = weight_digest(fixed_variables)
        semantic_digest = weight_digest(wrapper.network.classifier.weights)
        consolidation = RoutePhase(
            wrapper, self.bank, pool, settings, "consolidation", sorted(self.bank.vectors),
            derive_seed(settings.seed, len(self.records), "consolidation"),
            target=target, frozen_bank=frozen_bank,
        )
        self._diagnostic_stage = "before consolidation"
        record["before_consolidation"], before_features, views = self._probe(
            wrapper, probe, old, target, frozen_bank, consolidation.predictor
        )
        record["consolidation"] = self._fit(consolidation, settings.consolidation_steps)
        self._diagnostic_stage = "after consolidation"
        record["after_consolidation"], after_features, _ = self._probe(
            wrapper, probe, old, target, frozen_bank, consolidation.predictor, views
        )
        # Two-row centered CKA is degenerate; use actual aligned row counts.
        record.update(hidden_feature_cka=None, hidden_feature_cka_sample_count=None,
                      hidden_feature_cka_unavailable_reason="boundary_or_validation_unavailable")
        # Temporal alignment requires observed features at both boundaries.
        if before_features is not None and after_features is not None:
            cka_started = time.perf_counter()
            cka = descriptive_linear_cka(before_features, after_features)
            record.update(hidden_feature_cka=cka["linear_cka"],
                          hidden_feature_cka_sample_count=cka["sample_count"],
                          hidden_feature_cka_unavailable_reason=cka["linear_cka_unavailable_reason"])
            self.diagnostic_seconds += time.perf_counter() - cka_started
        frozen_unchanged = fixed_digest == weight_digest(fixed_variables)
        # Semantic-only consolidation changed the denoising/shared path.
        if settings.consolidation_scope == "semantic" and not frozen_unchanged:
            raise RuntimeError("Semantic-only consolidation changed the denoising/shared path.")
        # Consolidation changed its frozen target.
        if target_digest != weight_digest(target.weights):
            raise RuntimeError("Consolidation changed its frozen target.")
        # Consolidation changed frozen modulators.
        if bank_digest != weight_digest([v for pair in self.bank.vectors.values() for v in pair]):
            raise RuntimeError("Consolidation changed frozen modulators.")
        # Route phases changed the prior-task retention teacher.
        if prior_teacher is not None and prior_digest != weight_digest(prior_teacher.weights):
            raise RuntimeError("Route phases changed the prior-task retention teacher.")
        record["invariants"] = {
            "acquisition_network_unchanged": True, "old_modulators_unchanged": True,
            "consolidation_target_unchanged": True, "modulators_frozen_during_consolidation": True,
            "prior_teacher_unchanged": True, "nonsemantic_weights_unchanged": frozen_unchanged,
            "semantic_weights_changed": semantic_digest != weight_digest(wrapper.network.classifier.weights),
        }
        record["memory_bytes"] = {
            "student": tensor_bytes(wrapper.network.weights),
            "prior_task_teacher": tensor_bytes(prior_teacher.weights) if prior_teacher is not None else 0,
            "consolidation_target": tensor_bytes(target.weights),
            "frozen_modulator_copy": self.bank.nbytes,
            "persistent_modulators": self.bank.nbytes if settings.retain_modulators else 0,
            "acquisition_modulators": self.bank.nbytes,
            "predictor": tensor_bytes(consolidation.predictor.weights),
            "joint_optimizer": tensor_bytes(_optimizer_variables(wrapper.optimizer)),
            "phase_pool_images_labels": x.nbytes + y.nbytes,
            "validation_probe_images_labels": sum(a.nbytes for a in probe) if probe else 0,
        }
        record["pool_class_counts"] = {c: len(indices) for c, indices in pool.indices.items()}
        self._finish_boundary(wrapper, record, boundary, post_joint, post_features)
        record["memory_bytes"]["cached_boundary_features"] = record["cached_boundary_feature_bytes"]
        record["memory_bytes"]["cached_alignment_views"] = sum(
            tensor_bytes([view[name] for name in ("images", "times", "target_hidden")]) for view in (views or [])
        )
        record["total_updates"] = joint_updates + record["acquisition"]["updates"] + record["consolidation"]["updates"]
        record["route_seconds"] = time.perf_counter() - started
        self.introduced = seen
        # The discard-gates ablation releases persistent modulation state after consolidation.
        if not settings.retain_modulators:
            self.bank.vectors.clear()
        self.records.append(_json_value(record))
        return self.records[-1]

    def save(self, directory: str | Path) -> None:
        """Write strict JSON diagnostics, step/update CSVs, and numeric bank arrays.

        Args:
            directory (str | Path): Output directory for this operation, resolved using ordinary
                pathlib path semantics.

        Returns:
            saved (None): None; writes route JSON, CSV ledgers and float32 gate arrays in a
                compressed NPZ.

        Raises:
            OSError: If the directory or an artifact cannot be written.
        """

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "route_metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(_json_value(self.records), stream, indent=2, allow_nan=False)
        fields = ["task", "condition", "joint_updates", "acquisition_updates", "consolidation_updates", "extra_joint_updates", "total_updates", "route_seconds"]
        with (directory / "route_resources.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in self.records:
                row = {name: record.get(name) for name in fields}
                row["acquisition_updates"] = record["acquisition"]["updates"]
                row["consolidation_updates"] = record["consolidation"]["updates"]
                writer.writerow(row)
        trace_fields = ["task", "phase", "step", "focus_class", "examples", "loss", "ce", "semantic_loss"]
        with (directory / "route_steps.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=trace_fields)
            writer.writeheader()
            for record in self.records:
                for phase in ("acquisition", "consolidation"):
                    for row in record[phase].get("trace", []):
                        writer.writerow({"task": record["task"], "phase": phase, **row})
        vectors = self.bank.vectors if self.bank is not None else {}
        arrays = {f"class_{class_id}_{kind}": variable.numpy()
                  for class_id, pair in vectors.items()
                  for kind, variable in zip(("gain", "bias"), pair)}
        np.savez_compressed(directory / "modulators.npz", **arrays)
