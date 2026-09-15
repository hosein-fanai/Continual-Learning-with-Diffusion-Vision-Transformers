"""A fit-boundary adapter that retains the existing joint model and learner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any
import time

import numpy as np
import tensorflow as tf

from common.keras_compat import optimizer_iterations
from diffusion.models.transformer.di_t_classifier import DiTClassifier
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier
from semantic_consolidation.memory import ModulationBank


class SemanticConsolidationClassifier(DiffusionClassifier):
    """Run isolated route phases after joint acquisition and before task reports.

    The shared learner still owns replay, dense class growth, retention teachers,
    evaluation, and ordinary checkpoint artifacts. The plain Python controller
    owns its explicit sidecar state and is excluded from the Keras weight graph.
    """

    def __init__(self, route_controller: Any = None, extensions: Any = None, **kwargs: object) -> None:
        """Attach plain controllers outside the serialized Keras weight graph.

        Args:
            route_controller (Any): Optional plain Python RouteController attached outside Keras
                weight tracking.
            extensions (Any): Optional plain Python ExtensionController for scheduling, replay
                selection and held-out inference diagnostics.
            kwargs (object): Keyword arguments forwarded to the existing fit or constructor API,
                subject to the documented phase controls.

        Returns:
            initialized (None): None; ordinary wrapper initialization completes before attaching
                untracked controllers.

        Raises:
            TypeError: If wrapper constructor keywords are unsupported.
            ValueError: If inherited model, schedule or teacher configuration is invalid.
        """
        super().__init__(**kwargs)
        object.__setattr__(self, "route_controller", route_controller)
        object.__setattr__(self, "section10_controller", extensions)
        object.__setattr__(self, "experimental_controller", None)
        object.__setattr__(self, "fit_checkpoint", None)

    @property
    def checkpoint_interval(self) -> int:
        """Return the opt-in number of completed updates between progress commits.

        Returns:
            interval (int): Nonnegative configured update interval; zero retains
                only the ordinary completed-task checkpoints.

        Raises:
            None: Missing controllers expose the disabled interval zero.
        """
        return self.route_controller.settings.checkpoint_interval if self.route_controller is not None else 0

    def configure_fit_checkpoint(self, writer: object, checkpoint: object | None = None) -> None:
        """Attach the common progress writer for the next task's existing fits.

        Args:
            writer (object): Callable common atomic writer accepting plain state
                and its checkpointable current tf.data iterator.
            checkpoint (object | None): Inspected TaskCheckpoint for interrupted
                progress, or None for a new task.

        Returns:
            configured (None): None; creates untracked Python recovery state.

        Raises:
            ValueError: If saved progress metadata violates the declared schema.
        """
        from semantic_consolidation.fit_recovery import FitCheckpoint
        object.__setattr__(self, "fit_checkpoint", FitCheckpoint(self, writer, checkpoint))

    @staticmethod
    def validate_fit_checkpoint(state: dict[str, object]) -> None:
        """Validate phase progress before common configures or restores runtime state.

        Args:
            state (dict[str, object]): Decoded numeric fit sequence and cursors.

        Returns:
            validated (None): None; no caller state is changed.

        Raises:
            ValueError: If the progress payload is malformed or inconsistent.
        """
        from semantic_consolidation.fit_recovery import FitCheckpoint
        FitCheckpoint.validate(state)

    def fit_joint(self, x: tf.data.Dataset, y: object = None, **kwargs: object) -> tf.keras.callbacks.History:
        """Use the existing joint objective with optional committed fit progress.

        Args:
            x (tf.data.Dataset): Finite raw float32 image/integer-label batches,
                optionally with boolean replay provenance.
            y (object): None; labels are already present in the supplied dataset.
            kwargs (object): Existing Keras fit controls, including callbacks and
                validation data, or explicit schedule block budgets.

        Returns:
            history (tf.keras.callbacks.History): Complete joint epoch history;
                ordinary fits are unchanged when progress is not enabled.

        Raises:
            ValueError: If progress or fit data cannot satisfy its declared contract.
            Exception: Propagates existing Keras training or checkpoint failures.
        """
        # No progress state means the unchanged inherited fit implementation.
        if self.fit_checkpoint is None:
            return super().fit(x=x, y=y, **kwargs)
        return self.fit_checkpoint.fit(self, x, kwargs)

    def get_task_checkpoint_config(self) -> dict[str, object]:
        """Bind the semantic treatment to the common task-recovery fingerprint.

        Common calls this optional hook only for task checkpointing or recovery.
        The route seed has already been resolved by the runner. All settings,
        including phase budgets and gate bounds, enter the authenticated run
        descriptor; changing a treatment cannot silently reuse its old bank.

        Returns:
            config (dict[str, object]): Versioned ordinary-Python mapping containing
                a detached copy of every resolved RouteSettings field.

        Raises:
            ValueError: If the controller or resolved seed is missing.
        """
        controller = self.route_controller
        # Every saved treatment needs a controller and a resolved random seed.
        if controller is None or controller.settings.seed is None:
            raise ValueError("Task recovery requires a route controller with a resolved seed.")
        settings = controller.settings
        return {"schema_version": 1, "route": asdict(settings)}

    def get_task_checkpoint_state(self) -> dict[str, object]:
        """Export persistent semantic state at the initial or a completed task boundary.

        Common already checkpoints live network/teacher/optimizer variables and
        random generators. This hook adds the completed diagnostic records, dense
        introduced-class set, raw float32 gain/bias vectors and optional controller/
        observer records with retained float32 cohorts. The initial boundary has
        no records, introduced classes or bank. Phase predictors,
        frozen acquired targets and phase optimizers are temporary and are newly
        constructed for the next task using its deterministic phase seed.

        Returns:
            state (dict[str, object]): Versioned plain mapping with independent
                records and optional bank metadata plus float32 arrays [C, D].
                C is zero when gates were explicitly discarded; a platform-only
                control has no bank and stores None instead.

        Raises:
            ValueError: If the controller is unavailable or no task
                boundary exists, including an active pre-joint diagnostic cache.
        """
        self.get_task_checkpoint_config()
        controller = self.route_controller
        # A checkpoint must follow all three phases, never an active task boundary.
        if (controller._boundary is not None
                or (controller.records and controller.introduced != set(range(self.network.num_classes)))
                or (not controller.records and controller.introduced)):
            raise ValueError("Semantic task checkpoints require a fully completed task.")
        bank_state = None
        bank = controller.bank
        # Preserve a discarded bank's width as well as retained class vectors.
        if bank is not None:
            classes = sorted(bank.vectors)
            bank_state = {"dimension": bank.dimension, "classes": classes}
            for index, name in enumerate(("gain", "bias")):
                bank_state[name] = (np.stack([bank.vectors[c][index].numpy() for c in classes])
                                    if classes else np.empty((0, bank.dimension), dtype="float32"))
        from semantic_consolidation.recovery import completed_observer_state
        from semantic_consolidation.controller import _json_value
        extension = self.section10_controller
        return {"schema_version": 2, "records": deepcopy(controller.records),
                "introduced": sorted(controller.introduced), "bank": bank_state,
                "extensions": None if extension is None else _json_value(extension.records),
                "observer": completed_observer_state(self.experimental_controller)}

    def validate_task_checkpoint_state(self, state: dict[str, object], completed_tasks: int,
                                       class_count: int) -> None:
        """Validate the semantic payload before common changes destination state.

        The authoritative saved schedule supplies the completed class count, so
        validation does not require class expansion or restored network weights.
        The built projection width is invariant across supported class growth.

        Args:
            state (dict[str, object]): Saved versioned mapping. Bank gain and bias
                must be finite float32 arrays [C, D]; records and class IDs use
                ordinary Python dictionaries, lists and nonnegative integers.
            completed_tasks (int): Nonnegative completed-task cursor authenticated by
                the common learner; must equal the number of semantic records.
            class_count (int): Nonnegative number of dense classes in the completed
                schedule; zero at the initial boundary, independent of the current head width.

        Returns:
            validated (None): None; leaves models, optimizers, controllers and
                random streams unchanged when either accepting or rejecting state.

        Raises:
            ValueError: If protocol support, schema, task order, class coverage,
                feature width, array shape/dtype or finite-value checks fail.
        """
        self.get_task_checkpoint_config()
        controller = self.route_controller
        settings = controller.settings
        # Only a complete, explicitly versioned controller payload can be restored.
        version = state.get("schema_version") if isinstance(state, dict) else None
        expected_keys = {"schema_version", "records", "introduced", "bank"}
        # Version one remains readable for completed core tasks without optional state.
        if version == 2:
            expected_keys |= {"extensions", "observer"}
        # No undeclared state or invalid task cursor may reach live restoration.
        if (not isinstance(state, dict) or set(state) != expected_keys
                or type(version) is not int or version not in (1, 2)
                or type(completed_tasks) is not int or completed_tasks < 0
                or type(class_count) is not int or class_count < 0):
            raise ValueError("Invalid semantic task checkpoint schema or completed-task cursor.")
        from semantic_consolidation.recovery import validate_observer_state
        extension_records = state.get("extensions")
        # Extension records are mandatory exactly when their controller is configured.
        if (self.section10_controller is not None) != (extension_records is not None):
            raise ValueError("Semantic checkpoint extension ownership differs from the configured route.")
        if extension_records is not None:
            # One ordered extension record must correspond to every completed route task.
            if (not isinstance(extension_records, list) or len(extension_records) != completed_tasks
                    or any(not isinstance(row, dict) or row.get("task") != index
                           for index, row in enumerate(extension_records, 1))):
                raise ValueError("Semantic checkpoint extension task records are invalid.")
        validate_observer_state(self.experimental_controller, state.get("observer"), completed_tasks, class_count)
        records, introduced = state["records"], state["introduced"]
        expected_seen = list(range(class_count))
        # The restored raw network and controller must describe the same dense classes.
        if (not isinstance(introduced, list) or any(type(c) is not int for c in introduced)
                or introduced != expected_seen or not isinstance(records, list)
                or len(records) != completed_tasks):
            raise ValueError("Semantic checkpoint class coverage or record count differs from the restored task.")
        previous = []
        for task, record in enumerate(records, 1):
            # Every completed record must describe one strict dense-class introduction.
            if not isinstance(record, dict):
                raise ValueError("Semantic checkpoint task records must be dictionaries.")
            seen = record.get("seen_classes")
            # Ordered records must cover each introduced prefix exactly once.
            if (type(record.get("task")) is not int or record["task"] != task
                    or record.get("condition") != settings.condition or not isinstance(seen, list)
                    or any(type(c) is not int for c in seen) or seen != list(range(len(seen)))
                    or len(seen) <= len(previous) or len(seen) > len(expected_seen)
                    or record.get("new_classes") != seen[len(previous):]):
                raise ValueError("Semantic checkpoint records have an incompatible task or class order.")
            previous = seen
        # A partial diagnostic history cannot stand in for the completed raw model.
        if previous != introduced:
            raise ValueError("Semantic checkpoint final record does not cover the restored classes.")
        bank_state = state["bank"]
        has_bank = bool(completed_tasks) and settings.condition not in ("baseline", "extra_joint", "time_matched_joint")
        # Semantic treatments always allocated a bank, even if they discarded its vectors.
        if (bank_state is not None) != has_bank:
            raise ValueError("Semantic checkpoint bank presence disagrees with the treatment.")
        # Restore a real semantic bank only after validating all metadata and arrays.
        if bank_state is not None:
            # The primary Dense input is the already-built semantic projection width.
            dimension = int(self.network.classifier.layers[-1].kernel.shape[0])
            classes = introduced if settings.retain_modulators else []
            # Bank metadata must agree with the restored projection and retention policy.
            if (not isinstance(bank_state, dict) or set(bank_state) != {"dimension", "classes", "gain", "bias"}
                    or type(bank_state.get("dimension")) is not int or bank_state["dimension"] != dimension
                    or bank_state.get("classes") != classes
                    or any(type(c) is not int for c in bank_state["classes"])):
                raise ValueError("Semantic checkpoint bank width or retained class IDs are incompatible.")
            for name in ("gain", "bias"):
                values = bank_state[name]
                # Reject coercion of corrupted, nonfinite or differently typed gate arrays.
                if (not isinstance(values, np.ndarray) or values.dtype != np.dtype("float32")
                        or values.shape != (len(classes), dimension) or not np.isfinite(values).all()):
                    raise ValueError("Semantic checkpoint gates must be finite float32 arrays with matching shape.")
    def restore_task_checkpoint_state(self, state: dict[str, object], completed_tasks: int) -> None:
        """Install a detached bank after the common preflight and TensorFlow restore.

        Args:
            state (dict[str, object]): Validated versioned records, dense class IDs
                and optional finite float32 bank arrays [C, D].
            completed_tasks (int): Number of completed task records, including zero
                when restoring the initial boundary before the first fit.

        Returns:
            restored (None): None; replaces semantic state with independent
                variables without advancing caller-owned random generators.

        Raises:
            ValueError: If state fails the same preflight used before common
                restoration, including a changed projection width or class count.
        """
        self.validate_task_checkpoint_state(state, completed_tasks, self.network.num_classes if completed_tasks else 0)
        controller = self.route_controller
        settings = controller.settings
        records, introduced = state["records"], state["introduced"]
        bank_state, restored_bank = state["bank"], None
        # Construct independent variables only after the complete payload is valid.
        if bank_state is not None:
            dimension, classes = bank_state["dimension"], bank_state["classes"]
            restored_bank = ModulationBank(settings, dimension, settings.seed)
            restored_bank.add(classes)
            for row, class_id in enumerate(classes):
                for variable, name in zip(restored_bank.vectors[class_id], ("gain", "bias")):
                    variable.assign(bank_state[name][row])
        controller.records = deepcopy(records)
        controller.introduced = set(introduced)
        controller.bank = restored_bank
        controller._boundary = None
        controller.diagnostic_seconds = 0.
        # Completed extension tasks retain diagnostics but no task-local candidate pool.
        if self.section10_controller is not None:
            self.section10_controller.records = deepcopy(state["extensions"])
            self.section10_controller.candidates = []
            self.section10_controller.pending = None
            self.section10_controller.accepting_candidates = True
        from semantic_consolidation.recovery import restore_observer_state
        restore_observer_state(self.experimental_controller, state.get("observer"))

    def set_teacher_network(self, teacher_network: tf.keras.Model | None) -> None:
        """Observe the completed teacher only after the common setter succeeds.

        Args:
            teacher_network (tf.keras.Model | None): Independent frozen raw teacher, compatible
                wrapper, or None when clearing the teacher is allowed by existing KD settings.

        Returns:
            installed (None): None; installs the teacher with existing validation before
                notifying the optional observer.

        Raises:
            TypeError: If the teacher type is unsupported.
            ValueError: If the teacher is incompatible with the existing network, diffusion
                schedule or KD requirements.
        """
        super().set_teacher_network(teacher_network)
        observer = getattr(self, "experimental_controller", None)
        # An optional observer records only a successfully installed teacher.
        if observer is not None:
            observer.teacher_boundary(self)

    def fit(self, x: object = None, y: object = None, **kwargs: object) -> tf.keras.callbacks.History:
        """Complete joint fitting, then acquire modulators and consolidate them.

        Args:
            x (object): Finite tf.data.Dataset of raw images and sparse original labels, with
                optional replay provenance.
            y (object): Must be None because labels belong to the dataset, not a separate
                argument.
            kwargs (object): Keyword arguments forwarded to the existing fit or constructor API,
                subject to the documented phase controls.

        Returns:
            history (tf.keras.callbacks.History): Keras History from joint or scheduled fitting;
                semantic records remain in their controller sidecars.

        Raises:
            TypeError: If x is not a tf.data.Dataset or y is supplied separately.
            ValueError: If phase pools or optional observer settings are invalid.
            RuntimeError: If phase budgets or frozen-state invariants fail.
        """

        # Route fitting requires the shared learner's labeled tf.data.Dataset.
        if y is not None or not isinstance(x, tf.data.Dataset):
            raise TypeError("Route fitting requires the shared learner's labeled tf.data.Dataset.")
        observer = self.experimental_controller
        # Prepare held-out observation before the live task training begins.
        if observer is not None:
            kwargs = observer.before_task(self, kwargs.get("validation_data"), kwargs)
        before = int(optimizer_iterations(self.optimizer).numpy())
        # Capture the pre-joint feature boundary only when semantic phases are attached.
        if self.route_controller is not None:
            self.route_controller.before_joint(self, kwargs.get("validation_data"))
        extensions = self.section10_controller
        # Opt-in scheduling preserves the existing fit and optimizer implementation.
        if extensions is not None:
            history, effective_dataset = extensions.fit(self, x, kwargs, self.fit_joint)
        # With no scheduling extension, preserve the existing ordinary fit behavior.
        else:
            history, effective_dataset = self.fit_joint(x=x, y=y, **kwargs), x
        joint_updates = int(optimizer_iterations(self.optimizer).numpy()) - before
        # Semantic phases follow joint learning and precede completed-task reports.
        if self.route_controller is not None:
            phase_kwargs = dict(kwargs)
            phase_kwargs["route_joint_updates"] = joint_updates
            self.route_controller.run(self, effective_dataset, phase_kwargs)
            # Persistence work is measured separately from semantic optimizer allowances.
            if self.fit_checkpoint is not None:
                self.route_controller.records[-1]["checkpointing"] = {
                    "interval_updates": self.checkpoint_interval,
                    "io_seconds": self.fit_checkpoint.state["checkpoint_seconds"],
                    "scope": "Measured completed progress writes in this task; excluded from matched training-time allowances. Restart downtime is excluded. A write interrupted before its timer returns has no measured duration.",
                }
        # Evaluate the completed cognitive intervention before the teacher advances.
        if extensions is not None:
            extensions.after_task(self, kwargs.get("validation_data"))
        # Observe the completed intervention before the shared learner advances its teacher.
        if observer is not None:
            observer.after_task(self)
        return history

    def sample(self, network_name: str = "ema", labels: Any = None, x_t: Any = None,
               steps: int | None = None, scale: float | None = None, eta: float | None = None,
               return_x_ts: bool = False, return_x0s: bool = False,
               seed: int | None = None, verbose: bool = False) -> Any:
        """Retain the common learner's generated candidates until post-wake selection.

        Args:
            network_name (str): Existing raw or ema branch name; requesting EMA requires actual
                EMA weights where validated.
            labels (Any): Optional integer CFG condition IDs; None uses the existing sampler
                default.
            x_t (Any): Optional initial floating noise tensor for reverse sampling; None lets
                the existing sampler create its initial state.
            steps (int | None): Optional positive number of reverse sampling steps; None
                inherits the wrapper default.
            scale (float | None): Optional finite classifier-free guidance scale; None inherits
                the wrapper sampling default.
            eta (float | None): Optional reverse-sampling stochasticity coefficient; None
                inherits the wrapper default.
            return_x_ts (bool): Whether to include intermediate noisy reverse-trajectory states
                in the existing sampler output.
            return_x0s (bool): Whether to include intermediate clean-image estimates in the
                existing sampler output.
            seed (int | None): Explicit integer random seed; local or derived streams preserve
                reproducibility without reseeding caller-owned generators.
            verbose (bool): Whether the delegated sampler emits progress output.

        Returns:
            samples (Any): Existing sampler result: normalized floating NHWC images in [0, 1],
                optionally with requested trajectory outputs.

        Raises:
            ValueError: If inherited sampler controls or enabled candidate-capture
                shapes/conditions are invalid.
        """

        started = time.perf_counter()
        result = super().sample(
            network_name=network_name, labels=labels, x_t=x_t, steps=steps,
            scale=scale, eta=eta, return_x_ts=return_x_ts,
            return_x0s=return_x0s, seed=seed, verbose=verbose,
        )
        # Auxiliary sampling outside the between-task capture window is ignored.
        if self.section10_controller is not None:
            self.section10_controller.capture(self, result, labels)
        # Optional generation observers receive actual sample outputs and elapsed sampling cost.
        if self.experimental_controller is not None:
            self.experimental_controller.capture(self, result, labels, time.perf_counter() - started)
        return result


def adapt_model(base: DiffusionClassifier, controller: Any, extensions: Any = None) -> SemanticConsolidationClassifier:
    """Wrap the factory-created live raw network without changing its weights.

    Constructor metadata serializes its network, so that entry is deliberately
    replaced by the existing object. Calling from_config here would construct a
    fresh network. The optimizer is retained with its slots and iteration count.

    Args:
        base (DiffusionClassifier): Already factory-created V1 DiffusionClassifier wrapping
            exactly DiTClassifier with a compiled optimizer.
        controller (Any): Plain Python RouteController managing persistent gates and task-
            boundary semantic phases.
        extensions (Any): Optional plain Python ExtensionController for scheduling, replay
            selection and held-out inference diagnostics.

    Returns:
        adapted (SemanticConsolidationClassifier): SemanticConsolidationClassifier sharing
            the exact live raw network and compiled optimizer with preserved execution
            settings.

    Raises:
        TypeError: If base is not the supported V1 wrapper/raw-network pair.
        ValueError: If EMA, CFG, float32, hidden projection or deterministic-feature
            requirements are violated.
    """

    # Route one currently supports the V1 DiffusionClassifier with DiTClassifier.
    if type(base) is not DiffusionClassifier or type(base.network) is not DiTClassifier:
        raise TypeError("Route one currently supports the V1 DiffusionClassifier with DiTClassifier.")
    # Route one requires no EMA, CFG enabled, and float32 precision.
    if base.use_ema or not base.network.use_cfg or base.dtype_policy.name != "float32":
        raise ValueError("Route one requires no EMA, CFG enabled, and float32 precision.")
    # Route one requires a hidden classifier projection (classifier_mlp_ratio > 0).
    if not base.network.classifier_mlp_ratio or base.network.classifier_mlp_ratio <= 0:
        raise ValueError("Route one requires a hidden classifier projection (classifier_mlp_ratio > 0).")
    for prefix in ("", "clf_"):
        reshapers = getattr(base.network, f"{prefix}reshaper_ids_dict", {}) or {}
        options = getattr(base.network, f"{prefix}reshaper_kwargs", {}) or {}
        # Route one requires deterministic semantic features; stochastic variational flattening is
        # unsupported.
        if "flatten" in reshapers.values() and options.get("add_kl", False):
            raise ValueError("Route one requires deterministic semantic features; stochastic variational flattening is unsupported.")
    constructor = dict(base.get_config())
    constructor["network"] = base.network
    constructor["teacher_network"] = base.teacher_network
    adapted = SemanticConsolidationClassifier(route_controller=controller, extensions=extensions, **constructor)
    compile_args = {
        "optimizer": base.optimizer,
        "loss": base.loss,
        "run_eagerly": base.run_eagerly,
    }
    # Preserve the public Keras execution control; retain its older attribute fallback.
    steps = getattr(base, "steps_per_execution", getattr(base, "_steps_per_execution", None))
    # Preserve the existing compiled steps-per-execution control when TensorFlow exposes it.
    if steps is not None:
        compile_args["steps_per_execution"] = int(steps.numpy() if hasattr(steps, "numpy") else steps)
    jit_compile = getattr(base, "jit_compile", None)
    # Preserve the original compilation policy when the runtime provides a JIT setting.
    if jit_compile is not None:
        compile_args["jit_compile"] = jit_compile
    adapted.compile(**compile_args)
    return adapted
