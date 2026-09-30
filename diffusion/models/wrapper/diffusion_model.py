"""Training, evaluation, EMA, noising, and sampling for raw diffusion networks.

Networks in ``diffusion.models.transformer`` implement tensor transformations.
This module wraps one such network in the stateful Keras training protocol and
owns the diffusion process around it.

Supported raw models also include convolutional denoisers and composed
encoder/decoder networks implementing the wrapper protocol. Importing defines
the API; run_self_tests performs explicit regression work. Sampling mutates RNG
state, training mutates weights/optimizers, and runtime teachers remain outside
the serialized student configuration and tracked weight tree.
"""

import tensorflow as tf
from tensorflow.keras import metrics, losses, callbacks, optimizers, models

import numpy as np

from importlib import import_module

import inspect

import os

from collections.abc import Mapping
from typing import Literal, Sequence, get_args

from . import (
    NetworkName, 
    TrainType, 
    ClusteringType, 
    copy_network_weights_by_layer
)

from common.argument_saver import ArgumentSaverModel
from common.gradients import apply_policy_gradients
from common.keras_compat import (
    compute_compiled_loss, 
    optimizer_iterations, 
    register_optimizer_variables, 
    variable_path
)
from common.runtime import derive_seed, effective_seed
from common.random import SeedStream
from common.validation import require
from common.model import validate_progressive_depth_growth

from autoencoder.variational_autoencoder import VariationalAutoencoder

from diffusion import TeacherName
from diffusion.callbacks.batch_loss_plateau import BatchLossPlateau
from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer
from diffusion.models.transformer.di_t_decoder import DiTDecoder
from diffusion.layers.embedding.base_embedding import BaseEmbedding
from diffusion.schedulers import make_schedule, SchedulerName


class DiffusionModel(ArgumentSaverModel):
    """Orchestrate diffusion training and sampling around a raw diffusion network.

    The wrapped ``network`` predicts noise from ``(x_t, timestep, label)``.  This
    model constructs the schedule, samples forward-process noise, applies
    classifier-free guidance, computes configured losses, updates raw and EMA
    weights, manages progressive timestep/resolution/depth curricula, and runs
    generalized DDIM/DDPM sampling.

    This wrapper is deliberately separate from
    ``diffusion.models.transformer``: raw network classes own architecture and
    intermediate features; wrapper classes own optimization and diffusion
    state.  Call ``compile`` on the wrapper, not only on the raw network.

    Constructor keyword defaults and value-dependent behavior are documented in
    __init__. After construction call compile before fitting or reading metric
    trackers. Dynamic class expansion and depth growth mutate network topology;
    active timestep/resolution changes invalidate cached Keras execution functions.

    Attributes:
        network (ArgumentSaverModel): Trainable raw prediction network.
        ema_network (ArgumentSaverModel | None): Config-cloned exponential
            moving-average network, initialized with exactly the raw weights.
        schedules (dict[str, tf.Tensor]): One-dimensional schedule tensors,
            including ``alpha_bar``, ``sqrt_alpha_bar``, and
            ``sqrt_one_minus_alpha_bar``.
        use_image_loss (bool): Initially ``image_loss_coef > 0``.
        use_kl_loss (bool): Initially true only when ``kl_loss_coef > 0`` and
            the raw network has a KL-enabled reshaper.
        use_ctr_loss (bool): Initially true only when ``ctr_loss_coef > 0`` and
            class-token regularizer depths exist.
        show_separate_noise_losses (bool): Whether progress metrics split the
            full noise loss into conditional and unconditional rows.
        preprocess_type (Literal["standardize", "min-max"] | None): External-pixel to model-coordinate conversion.
            Public preprocess and postprocess own both directions.
        map_preprocess (bool): Whether datasets are mapped through
            :meth:`prep_inputs_map` before Keras consumes them.
        seen_classes (dict[object, int]): Dataset labels mapped to consecutive
            zero-based conditioning IDs in dynamic-class mode. It is the same
            dictionary stored in ``_init_config``, so newly observed labels are
            reflected immediately in wrapper configuration.
        teacher_network (tf.keras.Model | None): Independent raw teacher, frozen
            outside fit_teacher and excluded from student weights/config.
        trainable_teacher (bool): Enable separately compiled teacher training through
            fit_teacher. Ordinary student fitting never optimizes teacher weights.
        teacher_training (str): Continual learning trains the teacher before each task
            with "each_task", or only before the first task with "first_task".
        use_noise_distil_loss (bool): Positive noise-teaching coefficient with an
            attached compatible teacher; deferred teacher-free tasks leave it false.
        seed (int | None): Validated default random seed; None supplies no wrapper seed.
        _active_min_timestep (int): Inclusive current forward-noising lower bound.
        _active_max_timestep (int): Exclusive upper bound; [0, 0) means clean-only.
        _current_resolution (int): Active square image width, initially image_size.
    """

    def __init__(
        self, 
        network: ArgumentSaverModel, 
        use_ema: bool = True, 
        teacher_network: tf.keras.Model | None = None, 
        noise_teacher_network: tf.keras.Model | None = None, 
        teacher_noise_input_type: Literal[
            "images", 
            "images_timesteps", 
            "images_timesteps_labels"
        ] = "images_timesteps_labels", 
        current_teacher_network: tf.keras.Model | None = None, 
        previous_teacher_noise_loss_weight: float = 1., 
        current_teacher_noise_loss_weight: float = 1., 
        dual_teacher_scope: Literal["task", "all"] = "task", 
        defer_teacher: bool = False, 
        trainable_teacher: bool = False, 
        teacher_training: Literal["each_task", "first_task"] = "each_task", 
        test_network_name: NetworkName = "ema", 
        ema_decay: float = 0.999, 
        scheduler_name: SchedulerName = "clipped_cosine", 
        modify_first_t: bool = False, 
        p_uncond: float = 0.1, 
        train_cfg_scale: float | None = None, 
        test_cfg_scale: float = 4., 
        test_steps: int = 50, 
        test_eta: float = 0., 
        noise_loss_coef: float = 1., 
        show_separate_noise_losses: bool = False, 
        noise_distil_loss_coef: float = 0., 
        image_loss_coef: float = 0., 
        kl_loss_coef: float = 0., 
        ctr_loss_coef: float = 0., 
        kl_train_type: TrainType = "cond", 
        ctr_train_type: TrainType = "cond", 
        train_noisified_min_timesteps: int = 0, 
        train_noisified_max_timesteps: int | None = -1, 
        test_noisified_min_timesteps: int = 0, 
        test_noisified_max_timesteps: int | None = -1, 
        preprocess_type: Literal["standardize", "min-max"] | None = "standardize", 
        resize_method: str = "area", 
        resize_antialias: bool = True, 
        swap_noise_image: bool = False, 
        map_preprocess: bool = False, 
        map_num_parallel_calls: int | None = 1, 
        seen_classes: dict[object, int] = {}, 
        seed: int | None = None, 
        **kwargs: object
    ) -> None:
        """Initialize diffusion state and an optional EMA network.

        The wrapper owns preprocessing/schedules, checkpointed random streams, metric
        objects and replaceable raw/EMA containers. Floating schedule and loss/metric
        accumulators use dtype_policy.variable_dtype; native forward computation follows
        each network's compute policy. Runtime teacher models are frozen and excluded
        from student weight tracking/serialization. Constructing an EMA clone can build
        new variables and copy weights, but no training/optimizer update is performed.
        Seed None delegates to effective_seed and stream construction instead of
        restarting an existing deterministic sequence. Passthrough preprocessing has
        no declared clip bounds; only scaled sampling clips to [0,255].

        Args:
            network (ArgumentSaverModel): Built/configurable raw transformer or
                convolutional diffusion network.  Subclasses can extend the wrapper with additional prediction objectives.
            use_ema (bool): Clone ``network`` from its Keras config and maintain
                exponential moving-average weights after each train step.
                Deferred raw and cloned networks are built before the initial
                weight copy.
                Defaults to ``True``.
            teacher_network (tf.keras.Model | None): Independent raw diffusion network or
                wrapper whose raw network is frozen outside fit_teacher.
                None installs no teacher. An attached teacher forces deferred
                reconstruction because runtime teacher objects are not serialized.
                Defaults to ``None``.
            noise_teacher_network (tf.keras.Model | None): Independent current-task
                epsilon specialist. Train it with fit_teacher(..., teacher_name="noise").
                A noise-only native DiT receives its own DiffusionModel trainer.
                Specialist teachers cannot be combined with current_teacher_network.
                teacher_network remains the separate previous-task snapshot.
                None (default) leaves the specialist role empty.
                Defaults to ``None``.
            teacher_noise_input_type (str): Inputs to a plain callable epsilon teacher:
                ``"images"`` passes x_t, ``"images_timesteps"`` passes (x_t, t),
                and ``"images_timesteps_labels"`` passes (x_t, t, labels).
                The default is ``"images_timesteps_labels"``. Native teachers that
                declare ``full_return`` retain their existing diffusion interface.
            current_teacher_network (tf.keras.Model | None): Optional independent
                current-task teacher, frozen and excluded from serialized configuration.
                Set its local-to-student class mapping with set_current_teacher_network.
                None (default) leaves the shared current role empty.
                Defaults to ``None``.
            previous_teacher_noise_loss_weight (float): Nonnegative weight for the
                previous teacher's independently normalized noise KD loss. Defaults to 1.
                Zero disables this role. Setting both role noise weights to zero disables
                noise KD and requires no teacher, even with a positive global coefficient.
                Defaults to ``1.0``.
            current_teacher_noise_loss_weight (float): Nonnegative weight for the
                current teacher's independently normalized noise KD loss. Defaults to 1.
                Set to zero to disable the current noise objective.
                Defaults to ``1.0``.
            dual_teacher_scope (str): With a current teacher attached, ``"task"``
                restricts each noise teacher to its taught classes, including CFG-null
                rows. ``"all"`` allows current-teacher targets on every row while
                retaining the previous teacher's existing condition-vocabulary mask.
                Defaults to ``"task"``; a lone previous teacher preserves legacy scope.
            defer_teacher (bool): Permit a positive teacher objective to start
                without a teacher so continual learning can attach one later.
                Defaults to ``False``.
            trainable_teacher (bool): Enable fit_teacher. Native teachers receive an
                independent optimizer with the student's compile settings and reuse
                this wrapper family's supervised training and growth logic, without
                EMA, auxiliary losses, or another distillation teacher.
                Native teachers own their diffusion training state. Defaults to False.
            teacher_training (str): With trainable_teacher=True, continual learning
                fits the teacher before every task ("each_task", default), or only
                before the first task ("first_task"). Explicit fit_teacher calls
                always train the teacher regardless of this scheduling option.
                Defaults to ``'each_task'``.
            test_network_name (NetworkName): Default evaluation branch, raw or ema. When EMA
                is disabled, ema
                resolves to the raw network through get_network.
                Defaults to ``'ema'``.
            ema_decay (float): EMA retention in ``[0,1)``.  New EMA weight is
                ``decay * old_ema + (1-decay) * raw``.
                Defaults to ``0.999``.
            scheduler_name (SchedulerName): One of ``"linear"``,
                ``"scaled_linear"``, ``"squaredcos_cap_v2"``,
                ``"clipped_cosine"``, ``"sigmoid"``, ``"quadratic"``,
                ``"ve"``, ``"karras"``, ``"sub_vp"``, or ``"logistic"``.
                Defaults to ``'clipped_cosine'``.
            modify_first_t (bool): Force timestep 0 to have signal rate 1,
                noise rate 0, and cumulative alpha 1 after schedule creation.
                Defaults to ``False``.
            p_uncond (float): Per-example probability of replacing a shifted
                class label with null ID 0 during training.  It is forced to 0
                when ``network.use_cfg=False``.
                Defaults to ``0.1``.
            train_cfg_scale (float | None): CFG scale during training.  ``None``
                runs only the conditional network pass; a number additionally
                runs the null-label pass and combines predictions.
                Defaults to ``None``.
            test_cfg_scale (float): CFG scale for evaluation/sampling, forced to
                1 when CFG is disabled.
                Defaults to ``4.0``.
            test_steps (int): Default reverse-sampling evaluations in
                ``[2, network.timesteps]``.
                Defaults to ``50``.
            test_eta (float): Default stochasticity in ``[0,1]``: 0 is
                deterministic DDIM and 1 is DDPM-equivalent only for consecutive
                full-schedule steps.
                Defaults to ``0.0``.
            noise_loss_coef (float): Multiplier for prediction-vs-noise loss.
                Defaults to ``1.0``.
            show_separate_noise_losses (bool): When true, report the unchanged
                full noise loss as ``total_noise_loss`` and additionally report
                ``cond_noise_loss`` and ``uncond_noise_loss`` from non-null and
                null-label rows. These metrics do not change optimization.
                Defaults to ``False``.
            noise_distil_loss_coef (float): Multiplier for matching a frozen
                teacher's noise prediction on the same noisy inputs.
                Defaults to ``0.0``.
            image_loss_coef (float): Multiplier for reconstructed-image loss; 0
                disables it during normal training.
                Defaults to ``0.0``.
            kl_loss_coef (float): Multiplier for variational reshaper KL loss; 0
                disables it.
                Defaults to ``0.0``.
            ctr_loss_coef (float): Multiplier for auxiliary class-token
                regularizer loss; 0 disables it.
                Defaults to ``0.0``.
            kl_train_type (TrainType): ``"cond"`` uses conditional latent
                statistics; ``"uncond"`` uses the null-label forward pass.
                Defaults to ``'cond'``.
            ctr_train_type (TrainType): ``"cond"`` or ``"uncond"`` source for
                auxiliary regularizer predictions.  ``"uncond"`` requires a
                non-None ``train_cfg_scale``.
                Defaults to ``'cond'``.
            train_noisified_min_timesteps (int): Inclusive lower bound used by
                :meth:`fit`; default 0.
                Defaults to ``0``.
            train_noisified_max_timesteps (int | None): Exclusive training upper
                bound; -1 becomes ``network.timesteps`` and None becomes 0.
                Defaults to ``-1``.
            test_noisified_min_timesteps (int): Inclusive evaluation lower bound.
                Defaults to ``0``.
            test_noisified_max_timesteps (int | None): Exclusive evaluation
                upper bound; -1 becomes ``network.timesteps`` and None becomes 0.
                Defaults to ``-1``.
            preprocess_type (Literal["standardize", "min-max"] | None): Fixed conversion owned by the
                wrapper. "standardize" maps raw [0,255] to [-1,1]; "min-max" maps raw pixels to [0,1];
                None passes coordinates through. Public preprocess/postprocess accept canonical per-call
                overrides. Only these three values are accepted, with no alias/empty-string
                normalization. Sampling clips scaled outputs in raw [0,255] units; passthrough has no
                clipping bounds. Defaults to "standardize".
            resize_method (str): Interpolation method passed to tf.image.resize when the
                active resolution
                differs from image_size; TensorFlow validates supported names.
                Defaults to ``'area'``.
            resize_antialias (bool): Antialias flag passed to ``tf.image.resize``.
                Defaults to ``True``.
            swap_noise_image (bool): Train the raw output to reconstruct
                ``x_t`` and route :meth:`sample` to :meth:`sample_vae`;
                this mode requires a compatible KL bottleneck.
                Defaults to ``False``.
            map_preprocess (bool): Map ``tf.data.Dataset`` inputs through
                :meth:`prep_inputs_map` in :meth:`fit`, :meth:`evaluate`, and
                each progressive stage. Custom train/test steps then consume
                the prepared tensors directly. The default false preserves
                online preparation in the training device path.
                Defaults to ``False``.
            map_num_parallel_calls (int | None): Positive parallel-call value
                forwarded to ``Dataset.map``. ``None`` selects
                ``tf.data.AUTOTUNE``.
                Defaults to ``1``.
            seen_classes (dict[object, int]): Saved real-label to
                zero-based condition mapping for a grown continual
                model. ``{}`` starts with no observed classes. A nonempty
                mapping restores dynamic growth and expands a smaller raw/EMA
                topology before checkpoint weights are loaded. The model's
                dictionary is retained by reference in the wrapper config.
                Defaults to ``{}``.
            seed (int | None): Default TensorFlow random seed for noising,
                label dropout, latent draws, and sampling; per-call seeds override.
                Defaults to ``None``.
                None leaves the effective seed unspecified; named random streams obtain their own
                initial bases rather than using a fixed reproducible seed.
            **kwargs (object): Forwarded unchanged to tf.keras.Model through ArgumentSaverModel,
                including name (str), trainable (bool), dtype (dtype name/policy) and other
                supported Keras options. Keras rejects unsupported keyword names.

        Returns:
            None: Schedule tensors, active bounds/resolution, loss flags, and
            raw/EMA networks and metric trackers are initialized before compilation
            by :meth:`compile`.

        Raises:
            TypeError: The raw model lacks the serialization-aware diffusion interface.
            AssertionError: EMA, noising bounds, guidance, teacher, or sampler settings
                violate
                constructor invariants.
            ValueError: Schedule generation, teacher compatibility, seed normalization, or
                delegated network construction fails.
        """

        super().__init__(**kwargs)
        self._check_assertions(locals())
        self._save_init_args(
            locals(), 
            exclude=(
                "self", "kwargs", "__class__", 
                "network", "teacher_network", 
                "current_teacher_network", 
                "noise_teacher_network"
            )
        )
        # A trainable teacher must never join the student's tracked variables.
        object.__setattr__(self, "teacher_network", teacher_network)
        object.__setattr__(self, "current_teacher_network", current_teacher_network)
        object.__setattr__(self, "noise_teacher_network", noise_teacher_network)
        for head in self.get_teacher_names()[2:]:
            object.__setattr__(self, f"{head}_teacher_class_ids", None)
            object.__setattr__(self, f"{head}_teacher_task_class_ids", None)
        object.__setattr__(self, "current_teacher_class_ids", None)
        object.__setattr__(self, "current_teacher_task_class_ids", None)
        # Public Sequential replacement keeps the wrapper's tracked state stable
        # when a task boundary reconstructs its raw and EMA networks.
        self._network_holder = models.Sequential(name=f"{self.name}__raw")
        self._network_holder.add(network, rebuild=False)
        self._ema_holder = models.Sequential(name=f"{self.name}__ema")
        DiffusionModel._refresh_loss_flags(self)
        DiffusionModel._create_metrics(self)

        self.network.build()
        # Clone and initialize the EMA network when EMA tracking is enabled.
        if self.use_ema:
            ema_config = self.network.get_config()
            ema_config["name"] = self.network.name + "_ema"

            self._ema_holder.add(
                self.network.__class__.from_config(ema_config), 
                rebuild=False
            )
            self.ema_network.build()
            self.ema_network.set_weights(
                self.network.get_weights()
            )
        # Reconstruct a saved vocabulary before checkpoint weights are loaded.
        if self.seen_classes:
            self.network.dynamic_num_classes = True
            # Composite decoders must remain growable with their encoders.
            if hasattr(self.network, "decoder"):
                self.network.decoder.dynamic_num_classes = True

            # Keep the EMA topology on the same dynamic-class contract.
            if self.use_ema:
                self.ema_network.dynamic_num_classes = True
                # Keep an attached EMA decoder on the same dynamic contract.
                if hasattr(self.ema_network, "decoder"):
                    self.ema_network.decoder.dynamic_num_classes = True
            
            seen_num_classes = len(self.seen_classes)
            # Restore the saved class width before loading expanded weights.
            if seen_num_classes > self.network.num_classes:
                self._rebuild_classes(seen_num_classes)

        network_config = tf.keras.utils.serialize_keras_object(self.network)
        network_config["module"] = self.network.__class__.__module__
        self._init_config["network"] = network_config
        self._init_config.pop("teacher_network", None)
        self._init_config.pop("current_teacher_network", None)
        self._init_config.pop("noise_teacher_network", None)
        self._init_config["seen_classes"] = self.seen_classes

        self.image_size = self.network.image_size
        self.channels = self.network.channels
        self.timesteps = self.network.timesteps
        self.use_cfg = self.network.use_cfg
        self.p_uncond = 0. if not self.use_cfg else self.p_uncond
        self.test_cfg_scale = 1. if not self.use_cfg else self.test_cfg_scale
        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        self.noise_loss_coef = tf.constant(
            self.noise_loss_coef, 
            dtype=stable_dtype
        )
        self.image_loss_coef = tf.constant(
            self.image_loss_coef, 
            dtype=stable_dtype
        )
        self.kl_loss_coef = tf.constant(
            self.kl_loss_coef, 
            dtype=stable_dtype
        )
        self.ctr_loss_coef = tf.constant(
            self.ctr_loss_coef, 
            dtype=stable_dtype
        )
        self.noise_distil_loss_coef = tf.constant(
            self.noise_distil_loss_coef, 
            dtype=stable_dtype
        )
        self.use_image_loss = bool(self.image_loss_coef > 0.)
        self.train_noisified_max_timesteps = 0 if self.train_noisified_max_timesteps is None \
                                            else self.train_noisified_max_timesteps
        self.train_noisified_max_timesteps = self.timesteps if self.train_noisified_max_timesteps == -1 \
                                            else int(self.train_noisified_max_timesteps)
        self.test_noisified_max_timesteps = 0 if self.test_noisified_max_timesteps is None \
                                            else self.test_noisified_max_timesteps
        self.test_noisified_max_timesteps = self.timesteps if self.test_noisified_max_timesteps == -1 \
                                            else int(self.test_noisified_max_timesteps)
        self.map_num_parallel_calls = tf.data.AUTOTUNE if self.map_num_parallel_calls is None \
                                    else int(self.map_num_parallel_calls)
        self.seed = effective_seed(None, seed=self.seed)
        self._random_streams = {
            name: SeedStream(
                derive_seed(self.seed, "diffusion", name), 
                name=f"{self.name}__{name}_random"
            )
            for name in ("timesteps", "noise", "cfg", "sampling")
        }

        self._preprocess_training = None
        self._map_preprocess_without_teacher = bool(self.map_preprocess)

        # An installed runtime teacher makes teacher-free
        # configuration reconstruction permissible.
        if any(self.get_teacher_network(name) is not None 
            for name in self.get_teacher_names()):
            self.defer_teacher = True
            self._init_config["defer_teacher"] = True

        self.load_schedules()
        self.set_timestep_bounds()
        DiffusionModel.set_current_resolution(self)
        self.set_teacher_network(self.teacher_network)
        for teacher_name in self.get_teacher_names()[1:]:
            DiffusionModel.set_current_teacher_network(
                self, 
                self.get_teacher_network(teacher_name), 
                teacher_name=teacher_name
            )

    def _check_assertions(self, local_vars: dict[str, object]) -> None:
        """Validate schedule, EMA, sampler, and auxiliary-loss choices.

        Args:
            local_vars (dict[str, object]): Wrapper constructor namespace.

        Returns:
            None: Valid constructor settings pass without changing them.

        Raises:
            TypeError: The raw model or required diffusion/serialization attributes are
                absent.
            AssertionError: Noising ranges, EMA decay, sampling steps/eta, dropout
                probability,
                teacher requirements, train types, or restoration width are incompatible.
            ValueError: A shared-current and noise-specialist attachment are supplied together, or
                preprocess_type is not None, "standardize" or "min-max".
        """

        network = local_vars["network"]

        # A shared current teacher and per-head specialists are alternative topologies.
        if local_vars["current_teacher_network"] is not None \
        and local_vars["noise_teacher_network"] is not None:
            raise ValueError("current_teacher_network cannot be combined with per-head teachers.")

        # Require the configuration and metadata protocol used by the wrapper.
        if not isinstance(network, ArgumentSaverModel):
            raise TypeError(
                "network must inherit common.argument_saver.ArgumentSaverModel."
            )
        for attribute in (
            "timesteps", "image_size", "channels", "use_cfg", 
            "build", "set_current_resolution", "get_config"
        ):
            # Report a missing raw-network capability before building the wrapper.
            if not hasattr(network, attribute):
                raise TypeError(f"network must define {attribute!r}.")

        # Image conversion belongs to the wrapper, with fixed public pixel bounds.
        if local_vars["preprocess_type"] not in (None, "standardize", "min-max"):
            raise ValueError(
                f"Unknown diffusion preprocess_type: {local_vars['preprocess_type']!r}."
            )

        for prefix in ("train", "test"):
            t_min = local_vars[f"{prefix}_noisified_min_timesteps"]
            raw_t_max = local_vars[f"{prefix}_noisified_max_timesteps"]
            require(
                0 <= t_min < network.timesteps, 
                f"{prefix}_noisified_min_timesteps must be in "
                f"[0, {network.timesteps})."
            )

            t_max = 0 if raw_t_max is None else raw_t_max
            t_max = network.timesteps if t_max == -1 else t_max
            require(
                (t_min == 0 and t_max == 0) or
                t_min < t_max <= network.timesteps, 
                f"{prefix}_noisified_max_timesteps must be 0 for "
                f"clean-only inputs or in ({t_min}, {network.timesteps}]."
            )

        require(
            local_vars["test_network_name"] in get_args(NetworkName), 
            f"test_network_name must be one of {get_args(NetworkName)}."
        )

        require(
            0. <= local_vars["ema_decay"] < 1., 
            "ema_decay must be in the range of [0., 1.)."
        )

        require(
            2 <= local_vars["test_steps"] <= network.timesteps, 
            "steps must be in the range of [2, timesteps]."
        )

        require(
            0. <= local_vars["test_eta"] <= 1., 
            "eta must be in the range of [0., 1.]."
        )

        require(
            0. <= local_vars["p_uncond"] <= 1., 
            "p_uncond must be in the range of [0., 1.]."
        )

        require(
            local_vars["teacher_training"] in ("each_task", "first_task"), 
            "teacher_training must be 'each_task' or 'first_task'."
        )

        require(
            local_vars["teacher_noise_input_type"] in (
                "images", "images_timesteps", 
                "images_timesteps_labels"
            ), 
            "teacher_noise_input_type must be 'images', "
            "'images_timesteps', or 'images_timesteps_labels'."
        )

        for name in (
            "previous_teacher_noise_loss_weight", 
            "current_teacher_noise_loss_weight"
        ):
            require(
                np.isfinite(local_vars[name]) and local_vars[name] >= 0., 
                f"{name} must be finite and nonnegative."
            )
        require(
            local_vars["dual_teacher_scope"] in ("task", "all"), 
            "dual_teacher_scope must be 'task' or 'all'."
        )

        # Only effective noise objectives need a current or deferred teacher.
        if local_vars["noise_distil_loss_coef"] > 0. and (
            local_vars["previous_teacher_noise_loss_weight"] > 0.
            or local_vars["current_teacher_noise_loss_weight"] > 0.
        ):
            require(
                local_vars["teacher_network"] is not None
                or local_vars["current_teacher_network"] is not None
                or local_vars["noise_teacher_network"] is not None
                or local_vars["defer_teacher"], 
                "teacher_network is required when noise_distil_loss_coef "
                "is positive unless defer_teacher=True."
            )

        require(
            local_vars["kl_train_type"] in get_args(TrainType), 
            f"kl_train_type can be one of {TrainType}."
        )

        require(
            local_vars["ctr_train_type"] in get_args(TrainType), 
            f"ctr_train_type can be one of {TrainType}."
        )

        # Unconditional auxiliary objectives require a CFG prediction path.
        if local_vars["kl_train_type"] == "uncond" or \
        local_vars["ctr_train_type"] == "uncond":
            require(
                local_vars["network"].use_cfg and
                local_vars["train_cfg_scale"] is not None, 
                "Unconditional auxiliary losses require "
                "CFG and a non-None train_cfg_scale."
            )

        # A restored label mapping must cover the network's existing vocabulary.
        if local_vars["seen_classes"]:
            require(
                network.num_classes <= len(local_vars["seen_classes"]), 
                "seen_classes cannot be smaller than network.num_classes."
            )

    def _get_progressive_timestep_boundaries(
        self, 
        stages_num: int, 
        clustering_type: ClusteringType = "log_snr"
    ) -> list[int]:
        """Return N+1 monotonically increasing curriculum boundaries.

        ``uniform`` reproduces the simple equal-timestep partition.

        ``log_snr`` is a practical SNR-aware partition: boundaries are chosen
        at approximately equal intervals of log-SNR under the *existing full*
        diffusion schedule.  It keeps the original T and schedule unchanged;
        only the timesteps sampled for a curriculum stage are restricted.

        The returned boundaries are Python integers, not a tensor. SNR projection
        reads alpha_bar eagerly into a float64 NumPy array, so it requires eager
        schedule access; no model, schedule or random state is changed.

        Args:
            stages_num (int): Number of intervals, in ``1..timesteps``.
            clustering_type (ClusteringType): ``"uniform"`` spaces integer
                timesteps approximately evenly; ``"log_snr"`` projects evenly
                spaced log-SNR targets to the closest schedule indices.
                Defaults to ``'log_snr'``.

        Returns:
            list[int]: ``stages_num + 1`` strictly increasing boundaries with
            endpoints 0 and ``timesteps``.

        Raises:
            AssertionError: If ``stages_num`` is outside the valid range.
            ValueError: If the clustering name is unsupported or a strict
                partition cannot be constructed.
        """

        require(
            1 <= stages_num <= self.timesteps, 
            f"num_stages must be in [1, {self.timesteps}] range, "
            f"but got {stages_num}."
        )

        # Divide the discrete timestep axis evenly for uniform clustering.
        if clustering_type == "uniform":
            boundaries = np.rint(
                np.linspace(0, self.timesteps, stages_num + 1)
            ).astype(np.int32)
        # Partition the schedule by approximately equal log-SNR intervals.
        elif clustering_type == "log_snr": # This is just an estimation
            alpha_bar = np.asarray(
                self.schedules["alpha_bar"].numpy(), 
                dtype=np.float64
            )
            eps = np.finfo(np.float64).eps
            alpha_bar = np.clip(alpha_bar, eps, 1. - eps)
            log_snr = np.log(alpha_bar) - np.log1p(-alpha_bar)

            targets = np.linspace(
                log_snr[0], log_snr[-1], 
                stages_num + 1
            )
            boundaries = np.asarray([
                int(np.argmin(np.abs(log_snr - target)))
                for target in targets
            ], dtype=np.int32)
            boundaries[0] = 0
            boundaries[-1] = self.timesteps

            # Nearest-neighbour projection can create duplicate indices when
            # T is small or the SNR curve is very steep. Make the boundaries
            # strictly increasing while preserving both endpoints.
            for i in range(1, len(boundaries) - 1):
                boundaries[i] = max(boundaries[i], boundaries[i - 1] + 1)
            for i in range(len(boundaries) - 2, 0, -1):
                boundaries[i] = min(boundaries[i], boundaries[i + 1] - 1)
        # Reject clustering strategies outside the documented alternatives.
        else:
            raise ValueError(
                f"clustering must be one of {ClusteringType}."
            )

        # Reject collapsed timestep intervals after boundary projection.
        if np.any(np.diff(boundaries) <= 0):
            raise ValueError(
                "Could not construct strictly increasing timestep clusters. "
                "Use fewer stages or uniform clustering."
            )

        return boundaries.tolist()

    def _register_optimizer_variables(
        self, 
        optimizer: optimizers.Optimizer | None = None, 
        variables: list[tf.Variable] | None = None
    ) -> optimizers.Optimizer | None:
        """Register current network variables while retaining optimizer state.

        Progressive depth growth creates trainable variables after compilation.
        Modern optimizers may need replacement to extend their variable registry;
        the helper transfers iterations and compatible accumulated state. Omitting the
        arguments uses this wrapper's optimizer and all raw-network variables.

        Variables retain their individual model-variable dtypes and shapes; this helper
        does not cast them. If the optimizer registry is reconstructed, the active
        self.optimizer reference is replaced only when that optimizer was selected.
        Existing iterations and compatible slots are retained by the shared helper.

        Args:
            optimizer (tf.keras.optimizers.Optimizer | None): Optimizer that
                should know the current variable set, or
                ``None`` to use ``self.optimizer`` when it exists.
                Defaults to ``None``.
            variables (list[tf.Variable] | None): Variables to register, or
                ``None`` for every trainable
                variable in the raw diffusion network.
                Defaults to ``None``.

        Returns:
            Optimizer | None: The registered optimizer, or None before compilation.

        Raises:
            ValueError: A supplied optimizer exposes neither legacy slot creation nor build.
        """

        optimizer = getattr(self, "optimizer", None) if optimizer is None else optimizer
        variables = self.network.trainable_variables if variables is None else variables

        # Skip registration when the wrapper has not been compiled yet.
        if optimizer is None:
            return

        registered_optimizer = register_optimizer_variables(optimizer, variables)

        # Replace the active optimizer reference when its registry is rebuilt.
        if optimizer is getattr(self, "optimizer", None):
            self.optimizer = registered_optimizer

        return registered_optimizer

    def _refresh_loss_flags(self) -> None:
        """Refresh auxiliary-loss flags from the current network topology.

        KL loss is available when a KL reshaper is configured, and class-token
        regularization is available when at least one regularizer depth exists.
        The flags are recomputed after progressive depth additions because a
        newly appended layer may enable either loss.

        Args:
            None.

        Returns:
            None: ``use_noise_distil_loss``, ``use_kl_loss`` and
            ``use_ctr_loss`` are updated in place.

        Raises:
            No explicit exceptions are raised. The native network must expose the reshaper and
                token-regularizer metadata read by this helper.
        """

        self.use_noise_distil_loss = bool(
            self.noise_distil_loss_coef > 0. and 
            bool(self._noise_teacher_specs())
        )
        self.use_kl_loss = bool(
            self.kl_loss_coef > 0. and
            "flatten" in self.network.reshaper_ids_dict.values() and
            self.network.reshaper_kwargs.get("add_kl", False)
        )
        self.use_ctr_loss = bool(
            self.ctr_loss_coef > 0. and
            len(self.network.cls_token_regularizer_ids) > 0
        )

    def _add_depths(
        self, 
        depth_spec: object
    ) -> dict[str, dict[str, int]]:
        """Grow raw and EMA networks after a completed progressive stage.

        Validate the requested change on independent configuration clones before
        changing either live branch. Invalid stage or shape specifications leave
        the live networks unchanged.

        The network owns interpretation of ``depth_spec``. This wrapper applies
        that same specification to raw and EMA copies, builds newly created
        variables, initializes the new EMA weights from their raw counterparts,
        refreshes loss flags, registers optimizer variables and invalidates
        compiled execution functions. Existing weights and optimizer state are
        retained.

        Args:
            depth_spec (object): A depth specification accepted by the wrapped
                network.

        Returns:
            growth (dict[str, dict[str, int]]): Wrapped network's branch-wise
            before/added/after depth report.

        Raises:
            ValueError: If a depth specification is invalid, or raw and EMA
                growth creates incompatible weights.
        """

        for network in (self.network, self.ema_network):
            # Validate both branches before changing either live architecture.
            if network is not None:
                self._validate_progressive_growth(
                    network, 
                    {"stage_tasks": "depths_only", "depths": [depth_spec]}
                )

        raw_weight_ids = {
            id(weight) 
            for weight in self.network.weights
        }
        # Record existing EMA weights only when that network copy exists.
        ema_weight_ids = {
            id(weight) 
            for weight in self.ema_network.weights
        } if self.ema_network is not None else set()

        growth = self.network.add_depths(depth_spec)
        self.network.build()

        # Mirror the raw network's structural growth in the EMA clone.
        if self.ema_network is not None:
            self.ema_network.add_depths(
                depth_spec
            )
            self.ema_network.build()

            # Select newly allocated raw weights for initial EMA synchronization.
            raw_weights = [
                weight for weight in self.network.weights
                if id(weight) not in raw_weight_ids
            ]
            # Select newly allocated EMA weights while retaining their existing moving averages.
            ema_weights = [
                weight for weight in self.ema_network.weights
                if id(weight) not in ema_weight_ids
            ]

            # Guard against raw/EMA architectures diverging during growth.
            if len(raw_weights) != len(ema_weights):
                raise ValueError(
                    "Raw and EMA progressive depths have different weights."
                )

            for raw_weight, ema_weight in zip(raw_weights, ema_weights):
                ema_weight.assign(raw_weight)

        network_config = tf.keras.utils.serialize_keras_object(self.network)
        network_config["module"] = self.network.__class__.__module__
        self._init_config["network"] = network_config
        self._refresh_loss_flags()
        self._register_optimizer_variables()
        self.train_function = None
        self.test_function = None
        self.predict_function = None

        return growth

    def _rebuild_classes(self, num_classes: int) -> None:
        """Reconstruct class-expanded networks before replacing the live copies.

        Existing weights, random streams and EMA prefixes are retained. New EMA
        rows/columns start from raw weights. Optimizer registration remains with
        the caller so subclass variable selections refresh after replacement.

        Args:
            num_classes (int): New number of real classes, excluding the CFG null
                condition. The class-discovery caller supplies an expanded width.

        Returns:
            result (None): Replaces raw/EMA models and serialized raw configuration
                after successful reconstruction; existing class parameters survive.

        Raises:
            ValueError: If the architecture or stored weights cannot support the
                requested class expansion.
        """

        replacements = []
        for network in (self.network, self.ema_network):
            # Skip the EMA branch when moving averages are disabled.
            if network is None:
                continue

            config = network.get_config()
            config["num_classes"] = num_classes
            # Keep an attached decoder's class vocabulary aligned with its encoder.
            if "decoder_kwargs" in config:
                config["decoder_kwargs"]["num_classes"] = num_classes

            expanded = network.__class__.from_config(config)
            expanded.build()
            expanded.set_current_resolution(network.current_resolution)
            expanded.dynamic_num_classes = True
            # Preserve dynamic class discovery in an attached decoder too.
            if hasattr(expanded, "decoder"):
                expanded.decoder.dynamic_num_classes = True

            # Initialize new EMA parameters from the already-expanded raw copy.
            if replacements:
                copy_network_weights_by_layer(replacements[0], expanded)
            copy_network_weights_by_layer(
                network, 
                expanded, 
                allow_class_growth=True
            )
            replacements.append(expanded)

        # Construct and validate both copies before changing either live branch.
        for holder, expanded in zip(
            (self._network_holder, self._ema_holder), 
            replacements
        ):
            holder.pop(rebuild=False)
            holder.add(expanded, rebuild=False)

        network_config = tf.keras.utils.serialize_keras_object(self.network)
        network_config["module"] = self.network.__class__.__module__
        self._init_config["network"] = network_config

    def _check_new_labels(
        self, 
        x: object | None = None, 
        y: object | None = None, 
        original_labels: Mapping[object, object] | None = None, 
        verbose: int | bool = True
    ) -> None:
        """Discover labels and reconstruct a dynamic network before fitting.

        Only dynamic_num_classes networks are expanded. Each call assigns newly
        encountered labels in np.unique's sorted order after the existing mapping.
        Supply sparse labels, not one-hot matrices. Scanning a dataset consumes one
        complete iteration and requires it to terminate; a known infinite cardinality
        is rejected before iteration. Replay metadata after a dataset's labels is ignored.

        Args:
            x (tf.data.Dataset | object | None): Keras inputs.  A dataset must
                yield ``(images, labels)`` batches.
                Defaults to ``None``.
                None without separate y or a packaged supervised pair leaves discovery unchanged; Keras
                handles missing training input later.
            y (object | None): Separate Keras labels.  When supplied, these take
                precedence over labels contained in ``x``.
                Defaults to ``None``.
                Sparse integer array/tensor [N] or [N,1] is expected; None uses x labels. The native
                discovery helper uses np.unique without ordinary-teacher sparse-dtype validation.
            original_labels (Mapping[object, object] | None): Input-label to
                original dataset-label mapping when a caller has remapped its
                targets. Used only for reporting; class discovery and head
                indices still use the input labels. Must cover each new label.
                Defaults to ``None``, which reports the input labels directly.
            verbose (int | bool): Whether to print newly discovered labels.
                Defaults to ``True``.

        Returns:
            None: New real labels are mapped to consecutive zero-based targets,
            the raw/EMA networks are replaced within the same wrapper, and the
            wrapper initialization config sees the updated mapping.

        Raises:
            ValueError: Dynamic label discovery receives a dataset known to be infinite.
            KeyError: original_labels is supplied but omits a newly discovered label.
            Exception: Incompatible label tensors or delegated network growth failures
                propagate.
        """

        # Preserve every legacy behavior for an explicitly sized network.
        if not self.network.dynamic_num_classes:
            return

        # Explicit targets take precedence over labels packaged with Keras inputs.
        data = y if y is not None else x
        # Scan only labels, and refuse a dataset known to repeat forever.
        if isinstance(data, tf.data.Dataset):
            cardinality = int(
                tf.data.experimental.cardinality(data).numpy()
            )
            # Label discovery cannot exhaust a dataset that repeats forever.
            if cardinality == int(tf.data.INFINITE_CARDINALITY):
                raise ValueError(
                    "Dynamic class discovery requires a finite dataset."
                )

            labels = set()
            for batch in data:
                labels.update(
                    np.unique(batch[1].numpy()).tolist()
                )

            data = list(labels)
        # Leave Keras to report missing inputs when no labels were supplied.
        elif y is None:
            # Read labels from a packaged supervised input pair when no separate targets exist.
            if isinstance(x, (tuple, list)) and len(x) >= 2:
                data = x[1]
            # Leave missing or unlabeled inputs for the eventual Keras input validation.
            else:
                return

        new_classes = [
            label.item() 
            for label in np.unique(data)
            if label.item() not in self.seen_classes
        ]

        # Refresh symbolic outputs, optimizer variables, and cached traces once.
        if len(new_classes) > 0:
            # Resolve dataset identities before growth so incomplete reporting 
            # metadata cannot leave the vocabulary partially updated.
            discovered_labels = new_classes if original_labels is None else [
                original_labels[label] 
                for label in new_classes
            ]
            self._rebuild_classes(
                len(self.seen_classes) + 
                len(new_classes)
            )

            for real_label in new_classes:
                self.seen_classes[real_label] = len(self.seen_classes)

            # Report the labels added during this scan when requested.
            if verbose:
                print("The network has found new Classes:", discovered_labels)

            self._register_optimizer_variables()
            self.train_function = None
            self.test_function = None
            self.predict_function = None

    def _map_classes(self, classes: tf.Tensor) -> tf.Tensor:
        """Map real dataset labels to zero-based dynamic condition IDs.

        The fixed-width path returns its input object unchanged. The dynamic path
        constructs map keys/values in classes.dtype, does not change seen_classes, and
        asserts that every input label has already been observed.

        Args:
            classes (tf.Tensor): Integer dataset labels of arbitrary shape.

        Returns:
            tf.Tensor: Wrapper class IDs with the same shape and dtype.

        Raises:
            ValueError: Dynamic mode has not observed any class yet.
            tf.errors.InvalidArgumentError: At least one input label is absent from the saved
                dynamic vocabulary.
        """

        # Fixed-width networks retain the historical zero-based label contract.
        if not self.network.dynamic_num_classes:
            return classes

        # Reject class mapping before the first vocabulary has been observed.
        if not self.seen_classes:
            raise ValueError(
                "No classes have been observed by this dynamic model."
            )

        real_classes = tf.constant(
            list(self.seen_classes.keys()), 
            dtype=classes.dtype
        )
        wrapper_classes = tf.constant(
            list(self.seen_classes.values()), 
            dtype=classes.dtype
        )

        matches = tf.equal(classes[..., None], real_classes)
        tf.debugging.assert_equal(
            tf.reduce_any(matches, axis=-1), 
            tf.ones_like(classes, dtype=tf.bool), 
            message="Dataset contains a class that has not been observed during fit."
        )

        return tf.gather(
            wrapper_classes, 
            tf.argmax(matches, axis=-1, output_type=tf.int32)
        )

    def _is_prepared_dataset_spec(self, element_spec: object) -> bool:
        """Return whether one dataset element is already wrapper-prepared.

        Raw supervised datasets may contain a third replay-provenance tensor,
        so arity greater than two alone cannot distinguish raw data from the
        seven-tensor diffusion representation. Phase-specific wrappers can
        override this method for their own prepared arity.

        Args:
            element_spec (object): ``tf.data.Dataset.element_spec`` value.

        Returns:
            bool: True for the base seven-or-more-tensor prepared contract.

        Raises:
            This tuple/list arity check raises no explicit exceptions; it does not validate tensor
                contents.
        """

        return isinstance(element_spec, (tuple, list)) and len(element_spec) >= 7

    def _prepare_sampling_labels(
        self: "DiffusionModel", 
        network: ArgumentSaverModel, 
        labels: tf.Tensor | Sequence[int], 
        samples_per_label: int = 1
    ) -> tf.Tensor:
        """Normalize and validate explicit network condition IDs.

        Args:
            network (ArgumentSaverModel): Selected raw or EMA network.
            labels (tf.Tensor | Sequence[int]): Integer condition IDs in
                ``[0, network.num_labels)``; an empty vector is supported.
            samples_per_label (int): Positive number of contiguous repetitions
                per condition. Fractions and booleans are rejected.
                Defaults to ``1``.

        Returns:
            repeated_labels (tf.Tensor): Rank-one int32 IDs with
                ``len(labels) * samples_per_label`` entries, including an empty
                tensor for an empty request.

        Raises:
            ValueError: IDs are not integers, the repeat count is not a positive
                integer, or a static shape is not a vector.
            tf.errors.InvalidArgumentError: A runtime shape or vocabulary bound
                fails in ordinary TensorFlow execution.
        """

        # Require an exact positive repeat count, excluding booleans.
        if isinstance(samples_per_label, (bool, np.bool_)) or not isinstance(
            samples_per_label, (int, np.integer)
        ) or samples_per_label < 1:
            raise ValueError("samples_per_label must be a positive integer.")

        labels = tf.ensure_shape(tf.convert_to_tensor(labels), tuple([None]))
        # Only an empty label vector can safely lose an inferred floating dtype.
        if not labels.dtype.is_integer:
            # Reject nonempty floating labels instead of truncating class IDs.
            if labels.shape.num_elements() != 0:
                raise ValueError("Sampling label IDs must have an integer dtype.")

            labels = tf.cast(labels, tf.int32)

        labels = tf.cast(labels, tf.int64)
        num_labels = network.num_labels

        # Eager assertions return None; only graph assertion operations become dependencies.
        with tf.control_dependencies([
            assertion for assertion in (
                tf.debugging.assert_non_negative(
                    labels, 
                    message="Sampling label IDs must be nonnegative."
                ), 
                tf.debugging.assert_less(
                    labels, 
                    tf.cast(num_labels, labels.dtype), 
                    message="label IDs exceed the selected network vocabulary."
                )
            )
            if assertion is not None
        ]):
            return tf.repeat(tf.cast(labels, tf.int32), samples_per_label)

    def _mask_unknown_teacher_labels(
        self, 
        labels: tf.Tensor, 
        teacher_network: tf.keras.Model | None = None
    ) -> tf.Tensor:
        """Replace conditions beyond an older teacher's vocabulary with null ID zero.

        The label tensor keeps its integer dtype/shape. With no teacher num_labels
        metadata it is returned unchanged; otherwise IDs at/above that width become
        zero. Negative IDs are not separately rejected or remapped by this helper.

        Args:
            labels (tf.Tensor): Student network condition IDs, including any CFG offset,
                with arbitrary shape and integer dtype.
            teacher_network (tf.keras.Model | None): Explicit independent teacher;
                None selects the legacy previous-teacher slot.
                Defaults to ``None``.

        Returns:
            tf.Tensor: Same-shaped, same-dtype labels. IDs below teacher.num_labels
            are retained; other IDs become zero. If the teacher has no num_labels
            attribute, returns labels unchanged. This does not compute the separate
            mask used to exclude new-class rows from noise distillation.

        Raises:
            No explicit exceptions are raised. TensorFlow comparison/conversion errors propagate
                when labels are incompatible with the teacher vocabulary limit.
        """

        teacher = self.teacher_network if teacher_network is None else teacher_network
        teacher_num_labels = getattr(teacher, "num_labels", None)

        # Teachers without a declared vocabulary receive the original condition IDs.
        if teacher_num_labels is None:
            return labels
        return tf.where(
            labels < tf.cast(teacher_num_labels, labels.dtype), 
            labels, 
            tf.zeros_like(labels)
        )

    def _compute_base_loss(
        self, 
        y_true: tf.Tensor, 
        y_pred: tf.Tensor, 
        sample_weight: tf.Tensor | None = None
    ) -> tf.Tensor:
        """Evaluate the compiled prediction loss for one pair of tensors.

        Image/noise targets normally have shape [B,H,W,C]. They may use different
        real-floating prediction/target dtypes; the compatibility helper performs loss
        arithmetic in self.dtype_policy.variable_dtype. A custom compiled reduction
        may return per-example/per-pixel values rather than a scalar. No optimizer or
        metric update is performed.

        The first call can build the compiled-loss container and align its dtype to the
        variable policy, copying resolved loss objects before changing their dtype.
        These compile-container changes do not modify a caller-shared loss object.

        Args:
            y_true (tf.Tensor): Floating reference images or noise targets.
            y_pred (tf.Tensor): Predictions with the shape expected by the
                compiled loss. Loss math uses the model's stable dtype.
            sample_weight (tf.Tensor | None): Optional numeric weights
                broadcastable to the loss values; ``None`` uses equal weights.
                Defaults to ``None``.

        Returns:
            data_loss (tf.Tensor): Floating loss in the model's stable dtype
                and reduction shape, normally a scalar. Keras 3's loss
                container excludes layer regularizers handled by the wrapper.

        Raises:
            ValueError: The model has no compiled loss, receives nested rather than single
                target/prediction tensors, or Keras rejects the target/weight structure.
            tf.errors.InvalidArgumentError: Compiled-loss tensor shapes are incompatible.
        """

        return compute_compiled_loss(self, y_true, y_pred, sample_weight)

    def _create_metrics(self) -> None:
        """Allocate diffusion loss/accuracy trackers before the wrapper is built.

        Assigns fresh Keras metric objects to the wrapper; all accumulators use
        ``self.dtype_policy.variable_dtype``. Repeating this helper replaces trackers
        and therefore discards their accumulated measurements.

        Args:
            None.

        Returns:
            None: Trackers are installed for total, noise, split-noise, each teacher role, image, KL
                and token losses, plus token accuracy.

        Raises:
            This helper adds no validation or explicit exception. Keras metric construction errors
                propagate.
        """

        stable_dtype = self.dtype_policy.variable_dtype

        self.total_loss_tracker = metrics.Mean(
            dtype=stable_dtype, 
            name="loss"
        )
        self.noise_loss_tracker = metrics.Mean(
            dtype=stable_dtype, 
            name="total_noise_loss"
                if self.show_separate_noise_losses
                else "noise_loss"
        )
        self.noise_distil_loss_tracker = metrics.Mean(
            dtype=stable_dtype, 
            name="noise_distil_loss"
        )
        self.previous_teacher_noise_distil_loss_tracker = metrics.Mean(
            dtype=stable_dtype, 
            name="previous_teacher_noise_distil_loss"
        )
        self.current_teacher_noise_distil_loss_tracker = metrics.Mean(
            dtype=stable_dtype, 
            name="current_teacher_noise_distil_loss"
        )
        self.cond_noise_loss_tracker = metrics.Mean(
            dtype=stable_dtype, 
            name="cond_noise_loss"
        )
        self.uncond_noise_loss_tracker = metrics.Mean(
            dtype=stable_dtype, 
            name="uncond_noise_loss"
        )
        self.image_loss_tracker = metrics.Mean(
            dtype=stable_dtype, 
            name="image_loss"
        )
        self.kl_loss_tracker = metrics.Mean(
            dtype=stable_dtype, 
            name="kl_loss"
        )
        self.ctr_loss_tracker = metrics.Mean(
            dtype=stable_dtype, 
            name="ctr_loss"
        )
        self.ctr_accuracy_tracker = metrics.SparseCategoricalAccuracy(
            dtype=stable_dtype, 
            name="ctr_accuracy"
        )

    def _validate_progressive_growth(self, model: object, fit_kwargs: Mapping[str, object]) -> None:
        """Preflight a depth curriculum on independent native-network clones.

        Delegates to validate_progressive_depth_growth without mutating the live model
        or its optimizer; validating a requested stage can construct and build clones.

        Args:
            model (object): Wrapper exposing a native ``network`` with serialized growth
                configuration.
            fit_kwargs (Mapping[str, object]): Progressive-fit controls, including optional
                ``depth_stages``. Values are inspected without modifying the mapping.

        Returns:
            None: No value is returned.

        Raises:
            ValueError: The delegated progressive-growth validator rejects an unsupported or
                structurally incompatible requested depth stage. Network reconstruction/build
                failures propagate.
        """

        validate_progressive_depth_growth(model, fit_kwargs)

    def get_teacher_names(self) -> tuple[TeacherName, ...]:
        """List role identifiers in the order used by teacher lifecycle operations.

        Args:
            None.

        Returns:
            tuple[TeacherName, ...]: ``("previous", "current", "noise")``. These are selectors, not the
                attached model objects; no state changes.

        Raises:
            This fixed tuple accessor raises no explicit exceptions.
        """

        return ("previous", "current", "noise")

    def _training_optimizers(self) -> tuple:
        """Expose the student update state for optimizer-independence checks.

        Args:
            None.

        Returns:
            tuple[tf.keras.optimizers.Optimizer | None, ...]: One entry, self.optimizer when present
                and None before compilation. The optimizer is returned by reference and is not
                created or reset.

        Raises:
            This attribute accessor raises no explicit exceptions.
        """

        return tuple([getattr(self, "optimizer", None)])

    def _model_optimizers(self, model: tf.keras.Model | None) -> tuple:
        """Read update owners without compiling or constructing a model.

        Args:
            model (tf.keras.Model | None): Diffusion wrapper or ordinary Keras model. None
                represents an absent training owner.

        Returns:
            tuple[tf.keras.optimizers.Optimizer | None, ...]: A diffusion wrapper's
                _training_optimizers result; otherwise a one-entry tuple containing model.optimizer
                or None. No tensor conversion or optimizer mutation occurs.

        Raises:
            No explicit exceptions are raised. An overridden wrapper optimizer accessor may
                propagate its own errors.
        """

        return model._training_optimizers() if isinstance(model, DiffusionModel) \
            else tuple([getattr(model, "optimizer", None)])

    def _check_teacher_optimizer(
        self, optimizer: object, other_model: tf.keras.Model | None
    ) -> None:
        """Reject aliased update state before a teacher is compiled or fitted.

        Args:
            optimizer (object): Candidate Keras optimizer, optimizer name/config, or None. None
                leaves default optimizer creation to the eventual compile call.
            other_model (tf.keras.Model | None): Training owner whose optimizer objects and
                iteration variables must remain independent; None has no optimizer.

        Returns:
            None: Only identity is checked; neither model nor optimizer is mutated.

        Raises:
            ValueError: A selected optimizer is the same object as another training owner's
                optimizer, or shares its iteration variable.
        """

        counter = optimizer_iterations(optimizer)
        # An omitted optimizer lets Keras create its own independent default.
        if optimizer is not None and any(
            optimizer is other or (
                counter is not None and counter is optimizer_iterations(other)
            )
            for other in self._model_optimizers(other_model)
        ):
            raise ValueError("Teacher and student must use independent optimizers.")

    def get_teacher_network(
        self, teacher_name: TeacherName = "previous"
    ) -> tf.keras.Model | None:
        """Read a teacher attachment without constructing its training owner.

        Args:
            teacher_name (TeacherName): Role returned by get_teacher_names; the default ``"previous"``
                selects the historical teacher slot. A valid unattached role is distinct from an
                invalid name.

        Returns:
            tf.keras.Model | None: The stored raw teacher object, or None for a valid empty slot.
                The previous role reads teacher_network; current/noise read their corresponding
                specialist attributes.

        Raises:
            ValueError: The role name is not one of get_teacher_names().
        """

        # Unknown names must never silently train or query a different teacher.
        if teacher_name not in self.get_teacher_names():
            raise ValueError(f"teacher_name must be one of {self.get_teacher_names()}.")
        attribute = "teacher_network" if teacher_name == "previous" else f"{teacher_name}_teacher_network"
        return getattr(self, attribute, None)

    def _teacher_state_attribute(self, teacher_name: TeacherName) -> str:
        """Resolve the attribute holding a role's independent cached trainer.

        Args:
            teacher_name (TeacherName): Required supported role identifier.

        Returns:
            str: ``"_teacher_model"`` for previous, otherwise ``"_<role>_teacher_model"``. This only
                computes a name and validates the role.

        Raises:
            ValueError: The role name is not one of get_teacher_names().
        """

        self.get_teacher_network(teacher_name)
        return "_teacher_model" if teacher_name == "previous" else f"_{teacher_name}_teacher_model"

    def _remember_teacher_fit_state(
        self, network: tf.keras.Model | None, teacher_name: TeacherName
    ) -> None:
        """Discard an obsolete cached training owner after an attachment changes.

        Args:
            network (tf.keras.Model | None): Native diffusion model or wrapper to attach; None
                clears the selected attachment. This is a model object with its own variable dtypes,
                not an image tensor.
            teacher_name (TeacherName): Supported role whose cache is inspected.

        Returns:
            None: Sets only the selected cache to None when its raw network is not network;
                reattachment of the same network retains compilation and optimizer state.

        Raises:
            ValueError: The role name is not one of get_teacher_names().
        """

        attribute = self._teacher_state_attribute(teacher_name)
        cached = getattr(self, attribute, None)
        # A replacement model must not inherit another teacher's training state.
        if cached is not None and getattr(cached, "network", cached) is not network:
            object.__setattr__(self, attribute, None)

    def _attach_fitted_teacher(
        self, 
        teacher: tf.keras.Model, 
        teacher_name: TeacherName
    ) -> None:
        """Reattach a trained or grown teacher and restore its frozen inference state.

        Uses the public role setter, which refreshes loss flags and invalidates cached
        student execution functions. Current roles retain taught-class metadata; a
        dynamic specialist derives its columns from its updated vocabulary.

        Args:
            teacher (tf.keras.Model): Selected native wrapper or ordinary teacher returned by the
                fitting/growth path. A native wrapper supplies its raw network and vocabulary
                metadata.
            teacher_name (TeacherName): Supported role to replace.

        Returns:
            None: No value is returned.

        Raises:
            ValueError: The selected role, class mapping, or teacher protocol fails attachment
                validation. AssertionError from refreshed objective compatibility checks propagates.
                Fitting changes already made to the teacher are not rolled back.
        """

        # The previous role retains its established setter and snapshot semantics.
        if teacher_name == "previous":
            self.set_teacher_network(teacher)
        # Current roles retain independent taught support and refreshed dynamic columns.
        else:
            mapping = teacher.seen_classes if isinstance(teacher, DiffusionModel) else getattr(
                teacher, "_diffusion_seen_classes", {}
            )
            self.set_current_teacher_network(
                teacher, 
                class_ids=None if teacher_name in self.get_teacher_names()[2:] and mapping else getattr(
                    self, f"{teacher_name}_teacher_class_ids"
                ), 
                task_class_ids=getattr(self, f"{teacher_name}_teacher_task_class_ids"), 
                teacher_name=teacher_name
            )

    def get_teacher_model(
        self, 
        teacher_name: TeacherName = "previous"
    ) -> tf.keras.Model:
        """Return the selected teacher's training owner, creating its cache if necessary.

        Args:
            teacher_name (TeacherName): Role returned by get_teacher_names; the default ``"previous"``
                selects the historical teacher slot. A valid unattached role is distinct from an
                invalid name.

        Returns:
            tf.keras.Model: Independent native training wrapper around the attached network. Noise
                specialists use DiffusionModel; other roles use the student wrapper family. The
                current resolution is synchronized, but this call does not fit or compile the owner.

        Raises:
            ValueError: The role is unknown or has no attached network. Native wrapper construction
                and current-resolution validation also propagate their documented errors.
        """

        return self._get_teacher_model(teacher_name)

    def _teacher_vocabulary_size(self, network: tf.keras.Model) -> int:
        """Read the real-class width of a native conditioning vocabulary.

        Args:
            network (tf.keras.Model): Native diffusion network exposing integer num_classes; its CFG
                null label is not part of this count.

        Returns:
            int: network.num_classes, read without changing the model.

        Raises:
            No explicit exceptions are raised. A network without the required num_classes attribute
                raises AttributeError.
        """

        return network.num_classes

    def _current_teacher_spec(
        self, head: Literal["noise"] = "noise"
    ) -> dict[str, object] | None:
        """Resolve the current noise specialist or shared current-task teacher.

        Args:
            head (Literal["noise"]): Noise role selector; defaults to ``"noise"``. Subclasses may
                extend this selector to other heads.

        Returns:
            dict[str, object] | None: Current-role specification with network, configured weight,
                class_ids and task_class_ids; None when neither selected specialist nor shared
                current teacher is attached. Zero weight does not remove an attachment from this
                description.

        Raises:
            ValueError: The role name is not one of get_teacher_names().
        """

        return self._resolve_current_teacher_spec(head, self.current_teacher_noise_loss_weight)

    def _resolve_current_teacher_spec(
        self, 
        head: str, 
        weight: float
    ) -> dict[str, object] | None:
        """Describe a current teacher's independent column order and taught support.

        Prefers the head-specific attachment over the shared current role. Persistent
        dataset-label metadata is translated to current student class IDs; unavailable
        student classes use -1. Missing taught support defaults to nonnegative mapped
        columns. This method neither changes the models nor caches the result.

        Args:
            head (str): Supported specialist selector, such as noise or classifier in a subclass.
            weight (float): Role-specific loss multiplier copied unchanged into the result; zero is
                retained.

        Returns:
            dict[str, object] | None: ``role="current"``, the Keras network object, weight, optional
                tuple[int, ...] class_ids mapping teacher columns to student IDs, and optional
                taught task_class_ids. None means no attachment.

        Raises:
            ValueError: The role name is not one of get_teacher_names().
        """

        teacher_name = head if self.get_teacher_network(head) is not None else "current"
        network = self.get_teacher_network(teacher_name)
        # Attachment presence preserves mapped-target structure even for a disabled weight.
        if network is None:
            return None

        class_ids = getattr(self, f"{teacher_name}_teacher_class_ids")
        task_ids = getattr(self, f"{teacher_name}_teacher_task_class_ids")
        mapping = getattr(network, "_diffusion_seen_classes", {})
        # Persistent teacher columns follow dataset identities, independently of student order.
        if mapping and (class_ids is None or getattr(network, "_diffusion_dynamic_classes", False)):
            width = self._teacher_vocabulary_size(network)
            columns = [-1] * width
            for label, column in mapping.items():
                columns[column] = self.seen_classes.get(label, -1) \
                                  if self.network.dynamic_num_classes else int(label)
            class_ids = tuple(columns)
        # Fixed native teachers use their declared leading class vocabulary.
        if class_ids is None and getattr(network, "num_classes", None) is not None:
            class_ids = tuple(range(network.num_classes))
        # A map's supported columns are the default taught support until the caller narrows it.
        if task_ids is None and class_ids is not None:
            task_ids = tuple(value for value in class_ids if value >= 0)

        return dict(
            role="current", network=network, weight=weight, 
            class_ids=class_ids, task_class_ids=task_ids
        )

    def _teacher_model_options(self, network: tf.keras.Model) -> dict[str, object]:
        """Construct independent native training configuration from student settings.

        Copies the outer initialization dictionary and teacher vocabulary, retains the
        network object and dtype policy, disables EMA/recursive teachers and auxiliary
        losses, and enables a nonzero supervised noise objective. No live model is
        compiled or modified.

        Args:
            network (tf.keras.Model): Attached native network to train. Optional
                _diffusion_seen_classes metadata initializes the new owner's vocabulary.

        Returns:
            dict[str, object]: Constructor keyword arguments for the selected native wrapper family;
                values include Python settings, a Keras network and a dtype-policy object, not image
                tensors.

        Raises:
            This configuration helper raises no explicit exceptions.
        """

        options = dict(self._init_config)
        options.update(
            network=network, 
            teacher_network=None, 
            current_teacher_network=None, 
            noise_teacher_network=None, 
            trainable_teacher=False, 
            defer_teacher=False, 
            use_ema=False, 
            test_network_name="raw", 
            swap_noise_image=False, 
            noise_loss_coef=options.get("noise_loss_coef", 1.) or 1., 
            noise_distil_loss_coef=0., 
            image_loss_coef=0., 
            kl_loss_coef=0., 
            ctr_loss_coef=0., 
            seen_classes=dict(getattr(
                network, 
                "_diffusion_seen_classes", 
                {}
            )), 
            dtype=self.dtype_policy
        )

        return options

    def _get_teacher_model(
        self, 
        teacher_name: TeacherName = "previous"
    ) -> tf.keras.Model:
        """Get or create the independent native wrapper that owns teacher training.

        Reuses a cache only while its raw network is the attached object. A new owner
        disables recursive teacher objectives and EMA; noise specialists use the base
        DiffusionModel. Updates the owner's active resolution on every call and keeps
        the cached owner outside the student's tracked Keras weight tree.

        Args:
            teacher_name (TeacherName): Role returned by get_teacher_names; the default ``"previous"``
                selects the historical teacher slot. A valid unattached role is distinct from an
                invalid name.

        Returns:
            tf.keras.Model: Cached native training wrapper; its optimizer is initialized later by
                compile_teacher or fit_teacher.

        Raises:
            ValueError: The role is unknown or has no attached network. Native wrapper construction
                and current-resolution validation also propagate their documented errors.
        """

        network = self.get_teacher_network(teacher_name)
        # Querying an empty slot cannot construct a meaningful training owner.
        if network is None:
            raise ValueError(f"No {teacher_name} teacher is attached.")

        cache_attribute = self._teacher_state_attribute(teacher_name)
        teacher = getattr(self, cache_attribute, None)
        # A newly attached raw teacher needs its own training state.
        if teacher is None or getattr(teacher, "network", None) is not network:
            options = self._teacher_model_options(network)
            wrapper_type = DiffusionModel if teacher_name == "noise" else type(self)
            # Noise specialists use only the base diffusion wrapper's constructor options.
            if wrapper_type is DiffusionModel:
                parameters = inspect.signature(DiffusionModel.__init__).parameters
                options = {key: value for key, value in options.items() if key in parameters or key == "dtype"}
            teacher = wrapper_type(**options)
            object.__setattr__(self, cache_attribute, teacher)

        teacher.set_current_resolution(self._current_resolution)

        return teacher

    def _check_new_teacher_labels(
        self, 
        x: object | None = None, 
        y: object | None = None, 
        original_labels: Mapping[object, object] | None = None, 
        teacher_name: TeacherName = "previous", 
        verbose: int | bool = True
    ) -> None:
        """Prepare a native teacher's condition vocabulary before fitting.

        Temporarily enables the raw teacher's training state and delegates class discovery
        and structural growth to its independent owner. The finally block reattaches
        and freezes the teacher even when discovery or growth fails.

        Args:
            x (object | None): Raw image input or finite supervised tf.data.Dataset. Defaults to
                None; when y is present its labels take precedence.
            y (object | None): Sparse integer label array/tensor, normally [N] or [N,1]. Defaults to
                None, which discovers labels from x.
            original_labels (Mapping[object, object] | None): Reporting-only mapping of input labels
                to display labels; None (default) prints the input IDs.
                Defaults to ``None``.
            teacher_name (TeacherName): Role returned by get_teacher_names; the default ``"previous"``
                selects the historical teacher slot. A valid unattached role is distinct from an
                invalid name.
            verbose (int | bool): Print newly discovered labels when truthy; defaults to True.

        Returns:
            None: No value is returned.

        Raises:
            ValueError: The role is unknown or has no attached network. Native wrapper construction
                and current-resolution validation also propagate their documented errors.
            KeyError: original_labels omits a newly discovered ID. Errors from the selected owner's
                label discovery or compatible network growth propagate.
        """

        teacher = self._get_teacher_model(teacher_name)
        teacher.network.trainable = True
        try:
            teacher._check_new_labels(
                x=x, y=y, original_labels=original_labels, verbose=verbose
            )
        finally:
            self._attach_fitted_teacher(teacher, teacher_name)

    def _validate_teacher_optimizers(self, teacher_name: TeacherName) -> None:
        """Check the selected owner against the student and all other teacher roles.

        May lazily construct the selected training owner, but never recompiles an
        optimizer or changes its iteration/slot values.

        Args:
            teacher_name (TeacherName): Required supported role with an attached teacher.

        Returns:
            None: No value is returned.

        Raises:
            ValueError: The role is unknown or has no attached network. Native wrapper construction
                and current-resolution validation also propagate their documented errors.
            ValueError: A selected optimizer is the same object as another training owner's
                optimizer, or shares its iteration variable.
        """

        selected = self._get_teacher_model(teacher_name)
        for optimizer in self._model_optimizers(selected):
            self._check_teacher_optimizer(optimizer, self)
            for other_name in self.get_teacher_names():
                # Attached trainers must remain independent of every other teacher.
                if other_name != teacher_name:
                    other = getattr(
                        self, self._teacher_state_attribute(other_name), 
                        self.get_teacher_network(other_name)
                    )
                    self._check_teacher_optimizer(optimizer, other)

    def _compile_teacher(
        self, 
        teacher_name: TeacherName = "previous"
    ) -> None:
        """Apply independent student compile defaults unless an explicit override exists.

        Checks all optimizer identities first. An explicitly compiled teacher retains
        its settings; otherwise the student's serialized compile configuration is
        deserialized for the teacher. Native weights are trainable during compilation
        and frozen in a finally block. The owner's compile state and caches may change.

        Args:
            teacher_name (TeacherName): Role returned by get_teacher_names; the default ``"previous"``
                selects the historical teacher slot. A valid unattached role is distinct from an
                invalid name.

        Returns:
            None: No value is returned.

        Raises:
            ValueError: The role is unknown or has no attached network. Native wrapper construction
                and current-resolution validation also propagate their documented errors.
            ValueError: A selected optimizer is the same object as another training owner's
                optimizer, or shares its iteration variable.
            TypeError: Unsupported compile keywords or non-deserializable compilation objects are
                rejected by delegated Keras deserialization/compilation.
        """

        self._validate_teacher_optimizers(teacher_name)

        teacher = self._get_teacher_model(teacher_name)
        # Public teacher compilation survives subsequent student recompilation.
        if getattr(teacher, "_teacher_compile_explicit", False):
            for optimizer in self._model_optimizers(teacher):
                self._check_teacher_optimizer(optimizer, self)
            
            return

        compile_config = tf.keras.utils.deserialize_keras_object(
            self.get_compile_config()
        )
        teacher.network.trainable = True
        try:
            teacher.compile(**compile_config)
        finally:
            # The teacher stays frozen outside its explicitly selected training calls.
            teacher.network.trainable = False

    def _refresh_jit_support(self) -> None:
        """Set supports_jit from the active resizing and vocabulary configuration.

        Disables automatic JIT eligibility for unsupported online image/positional
        interpolation and dynamic class mapping, whose runtime assertions must remain
        effective. Reads layer metadata and changes only the wrapper's supports_jit flag.

        Args:
            None.

        Returns:
            None: No value is returned.

        Raises:
            This capability inspection raises no explicit exceptions.
        """

        resized = getattr(self, "_current_resolution", self.image_size) != self.image_size
        online_resize = resized and not self.map_preprocess
        supported_resize = self.resize_method == "nearest" or (
            self.resize_method == "bilinear" and 
            not self.resize_antialias
        )
        positional_resize = resized and any(
            isinstance(layer, BaseEmbedding) 
            and layer.grid_size is not None
            and layer.pos_embed_type is not None
            and layer.pos_interpolation_method not in ("nearest", "bilinear")
            for layer in self.network._flatten_layers()
        )
        # XLA drops the runtime unknown-label assertion in _map_classes.
        self.supports_jit = (
            (not online_resize or supported_resize) 
            and not positional_resize
            and not self.network.dynamic_num_classes
        )

    def _remap_teacher_conditions(
        self, 
        labels: tf.Tensor, 
        class_ids: Sequence[int] | None
    ) -> tf.Tensor:
        """Map student conditions to teacher-local IDs while preserving CFG null zero.

        ``class_ids[j]`` is the student class represented by teacher output j.
        Unknown conditions become zero; separate row eligibility controls which
        such predictions contribute to KD. An absent map preserves legacy IDs.

        The output retains the input integer dtype and shape [B]. Mapping constants
        are built in that dtype; internal argmax uses int32 before recasting. No random
        stream, vocabulary, model weight or class map is changed.

        Args:
            labels (tf.Tensor): Integer student condition IDs of arbitrary shape;
                real labels include the wrapper's CFG offset.
            class_ids (Sequence[int] | None): Unique zero-based student IDs in
                teacher-local output order; None preserves the original tensor.

        Returns:
            remapped (tf.Tensor): Same-shaped, same-dtype teacher-local condition
                IDs; unknown classes and CFG null conditions map to zero.

        Raises:
            No explicit exceptions are raised. TensorFlow reports incompatible condition/map dtypes
                or shapes during equality and gather operations.
        """

        # Legacy teachers already share the student's leading class-ID ordering.
        if class_ids is None:
            return labels

        offset = int(self.use_cfg)
        support = tf.constant(tuple(class_ids), dtype=labels.dtype)
        matches = tf.equal(labels[..., None] - offset, support)
        local_ids = tf.cast(tf.argmax(matches, axis=-1), labels.dtype) + offset
        known = tf.reduce_any(matches, axis=-1)

        # Null conditions must never become a class ID after local vocabulary remapping.
        if self.use_cfg:
            known = tf.logical_and(known, labels != 0)

        return tf.where(known, local_ids, tf.zeros_like(labels))

    def _noise_teacher_specs(self) -> tuple[dict[str, object], ...]:
        """Return active independent noise teachers in previous/current role order.

        Runtime class maps describe each teacher's output order and taught support
        in student class IDs. They are deliberately absent from dataset tensors and
        the student's tracked weights and serialized configuration.

        Returns a tuple of dict[str, object] entries containing role (str), network
        (Keras object), weight (float), class_ids (tuple[int, ...] | None) and
        task_class_ids (tuple[int, ...] | None). No model execution or state mutation
        occurs; zero-weight/absent roles are omitted and an empty tuple is valid.

        Args:
            None.

        Returns:
            tuple[dict[str, object], ...]: Active role specifications in previous/current order,
                with independent column and taught-support metadata as described above.

        Raises:
            This metadata-only resolver raises no explicit exceptions; it does not evaluate teacher
                outputs or validate their shapes.
        """

        specifications = []
        previous = self.teacher_network
        # The legacy slot remains the previous-task teacher when its weight is positive.
        if previous is not None and self.previous_teacher_noise_loss_weight > 0.:
            class_count = getattr(previous, "num_classes", None)
            # Some external conditional teachers expose only the condition-vocabulary width.
            if class_count is None and getattr(previous, "num_labels", None) is not None:
                class_count = previous.num_labels - int(self.network.use_cfg)

            task_ids = getattr(previous, "_diffusion_task_class_ids", None)
            seen = getattr(previous, "_diffusion_seen_classes", {})

            # Explicit task metadata distinguishes taught classes from a fixed wider head.
            if task_ids is not None:
                task_ids = tuple(task_ids)
            # Dynamic snapshots preserve the student IDs of classes seen before the task.
            elif seen:
                task_ids = tuple(sorted(seen.values()))
            # Legacy external teachers without coverage metadata use their declared width.
            else:
                task_ids = tuple(range(int(class_count))) if class_count is not None else None

            specifications.append({
                "role": "previous", 
                "network": previous, 
                "weight": self.previous_teacher_noise_loss_weight, 
                "class_ids": None, "task_class_ids": task_ids
            })

        current = self._current_teacher_spec("noise")
        # Each current noise target comes only from its selected head specialist.
        if current is not None and current["weight"] > 0.:
            specifications.append(current)

        return tuple(specifications)

    def _predict_teacher_noise(
        self, 
        x_t: tf.Tensor, 
        t: tf.Tensor, 
        cond_labels: tf.Tensor, 
        uncond_labels: tf.Tensor | None = None, 
        scale: float | None = None, 
        teacher_network: tf.keras.Model | None = None
    ) -> tf.Tensor:
        """Return a frozen teacher epsilon target with the student's CFG convention.

        Use the base noise protocol explicitly so extended wrappers can also
        distil from an ordinary noise-only model.
        An explicit teacher_network selects an independent role without mutating
        the wrapper's teacher references during parallel dataset preparation.

        Args:
            x_t (tf.Tensor): Floating noisy images ``[B,H,W,C]`` shared with the
                student and already corrupted under this wrapper's schedule.
            t (tf.Tensor): Integer schedule indices ``[B]``.
            cond_labels (tf.Tensor): Integer teacher-compatible condition IDs ``[B]``.
            uncond_labels (tf.Tensor | None): Null IDs ``[B]`` required for CFG;
                None is valid when guidance is disabled.
                Defaults to ``None``.
            scale (float | None): Guidance coefficient; None returns the conditional
                prediction. With CFG, zero selects null, one conditional, and larger
                values extrapolate conditional-minus-null predictions.
                Defaults to ``None``.
            teacher_network (tf.keras.Model | None): Explicit independent teacher;
                None selects the attached previous-task teacher.
                Defaults to ``None``.

        Returns:
            epsilon (tf.Tensor): Detached floating epsilon target ``[B,H,W,C]``
                in the teacher prediction dtype, computed in inference mode.

        Raises:
            TypeError: A callable noise teacher returns a structured, sparse/ragged or nonfloating
                prediction.
            ValueError: The selected teacher is missing or a callable prediction has an incompatible
                static image shape.
            tf.errors.InvalidArgumentError: Callable epsilon has a mismatched dynamic shape or
                nonfinite values. Native network execution errors propagate.
        """

        (eps_c, eps_u), *_ = DiffusionModel.call_network(
            self, x_t, t, 
            cond_labels, 
            uncond_labels, 
            scale, 
            network_name="teacher", 
            teacher_network=teacher_network, 
            training=False
        )
        # Guidance is the same epsilon combination used by denoise; image
        # reconstruction is unnecessary when preparing a distillation target.
        eps = eps_u + scale * (eps_c - eps_u) if self.use_cfg and scale is not None else eps_c

        return tf.stop_gradient(eps)

    def _compute_single_teacher_noise_loss(
        self, 
        teacher_noises_pred: tf.Tensor, 
        noises_pred: tf.Tensor, 
        teacher_noise_mask: tf.Tensor | None = None
    ) -> tf.Tensor | float:
        """Compute the optional masked noise-distillation loss.

        This helper computes whenever called; the surrounding loss aggregator gates
        disabled KD. Teacher predictions are stop-gradient targets. A supplied row mask
        is rescaled by batch_size/sum(mask) before compiled_loss, so its usual batch-mean
        reduction averages selected examples; a zero mask contributes zero for finite
        inputs. A custom compiled loss retains its own reduction semantics.

        Both predictions are real-floating tensors [B,H,W,C]. The optional mask is
        Boolean or numeric [B], cast to policy variable dtype before normalization;
        returned compiled-loss tensors use that same stable dtype. Only the student
        prediction remains differentiable; no tracker/optimizer is updated here.

        Args:
            teacher_noises_pred (tf.Tensor): Frozen teacher noise predictions.
            noises_pred (tf.Tensor): Student noise predictions.
            teacher_noise_mask (tf.Tensor | None): Samples taught by the
                previous network, or None to use the whole batch.
                Defaults to ``None``.

        Returns:
            tf.Tensor: Compiled teacher-to-student loss, normally a scalar. It is not
            multiplied by noise_distil_loss_coef here.

        Raises:
            No explicit exceptions are raised here. The compiled base-loss helper propagates its
                missing-loss and target/weight-shape errors; TensorFlow mask broadcasting/reshape
                failures also propagate.
        """

        # Compute squared errors and exposure weights in the stable policy dtype.
        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        teacher_noises_pred = tf.cast(teacher_noises_pred, stable_dtype)
        noises_pred = tf.cast(noises_pred, stable_dtype)
        noise_distil_sample_weight = None

        # Normalize a teacher mask by selected exposure so 
        # absent teacher classes do not dilute the loss.
        if teacher_noise_mask is not None:
            noise_distil_sample_weight = tf.cast(
                teacher_noise_mask, 
                noises_pred.dtype
            )
            noise_distil_sample_weight *= tf.math.divide_no_nan(
                tf.cast(tf.shape(noises_pred)[0], noises_pred.dtype), 
                tf.reduce_sum(noise_distil_sample_weight)
            )
            noise_distil_sample_weight = tf.reshape(
                noise_distil_sample_weight, 
                tf.concat([
                    tf.shape(noise_distil_sample_weight)[:1], 
                    tf.ones(
                        tuple([tf.rank(noises_pred) - 2]), 
                        dtype=tf.int32
                    )
                ], axis=0)
            )

        noise_distil_loss = self._compute_base_loss(
            tf.stop_gradient(teacher_noises_pred), 
            noises_pred, 
            sample_weight=noise_distil_sample_weight
        )

        return noise_distil_loss

    def _teacher_uses_native_noise_api(
        self, 
        teacher_network: tf.keras.Model | None = None
    ) -> bool:
        """Identify the native full-return protocol without probing model outputs.

        An explicit ``full_return`` parameter marks the repository's diffusion
        interface. Accepting arbitrary keyword arguments alone does not: ordinary
        Keras teachers must not receive wrapper-specific call arguments.

        Args:
            teacher_network (tf.keras.Model | None): Candidate teacher to inspect;
                None selects the attached previous-task teacher.
                Defaults to ``None``.

        Returns:
            native (bool): True only when call explicitly declares full_return;
                false for absent teachers and uninspectable call signatures.

        Raises:
            No explicit exceptions are raised. TypeError and ValueError from inspect.signature are
                caught and treated as an unsupported native signature.
        """

        teacher = self.teacher_network if teacher_network is None else teacher_network
        # A deferred or cleared teacher has no native inference capabilities.
        if teacher is None:
            return False

        try:
            parameters = inspect.signature(
                getattr(teacher, "call", teacher)
            ).parameters
        except (TypeError, ValueError):
            return False

        return "full_return" in parameters

    @property
    def network(self) -> models.Model:
        """Read the replaceable raw network held by the wrapper.

        Args:
            None.

        Returns:
            tf.keras.Model: First model in _network_holder; returned by reference, with its own
                layer dtypes and weights. The accessor does not build, clone or modify it.

        Raises:
            This accessor raises no explicit exceptions for the initialized raw-network holder.
        """

        return self._network_holder.layers[0]

    @property
    def ema_network(self) -> models.Model | None:
        """Read the optional exponential-moving-average model.

        Args:
            None.

        Returns:
            tf.keras.Model | None: First model in _ema_holder, or None when the holder is empty.
                Returned by reference; no build, weight copy or update occurs.

        Raises:
            This accessor raises no explicit exceptions.
        """

        return self._ema_holder.layers[0] if self._ema_holder.layers else None

    @property
    def current_timesteps_bounds(self) -> tuple[int, int]:
        """Return active forward-noising bounds as ``[minimum, maximum)``.

        Args:
            None.

        Returns:
            tuple[int, int]: Inclusive minimum and exclusive maximum timestep.

        Raises:
            This state accessor raises no explicit exceptions.
        """

        return self._active_min_timestep, self._active_max_timestep

    @property
    def current_resolution(self) -> tuple[int, int]:
        """Return the square image resolution currently processed.

        Args:
            None.

        Returns:
            tuple[int, int]: Active positive integer resolution of the wrapper
            and raw network, respectively.

        Raises:
            This state accessor raises no explicit exceptions.
        """

        return self._current_resolution, self.network.current_resolution

    @property
    def metrics(self) -> list[metrics.Metric]:
        """Return Keras metric trackers reset between fit/evaluate epochs.

        Metric objects already exist after construction. Their scalar accumulators
        use the policy variable dtype; reading this list does not reset any values.

        Args:
            None.

        Returns:
            list[tf.keras.metrics.Metric]: Total, noise, optional split-noise,
            distillation, image, KL, class-token regularizer loss trackers and
            regularizer accuracy tracker. They already exist after construction.

        Raises:
            This accessor raises no explicit exceptions; it returns existing metric objects without
                resetting them.
        """

        return [
            self.total_loss_tracker, 
            self.noise_loss_tracker, 
            self.cond_noise_loss_tracker, 
            self.uncond_noise_loss_tracker, 
            self.noise_distil_loss_tracker, 
            self.image_loss_tracker, 
            self.kl_loss_tracker, 
            self.ctr_loss_tracker, 
            self.ctr_accuracy_tracker
        ] + ([
            self.previous_teacher_noise_distil_loss_tracker, 
            self.current_teacher_noise_distil_loss_tracker
        ] if self._current_teacher_spec("noise") is not None else [])

    @classmethod
    def from_config(
        cls, 
        config: Mapping[str, object]
    ) -> "DiffusionModel":
        """Reconstruct a wrapper and independent raw network from serialized settings.

        The top-level mapping is shallow-copied. A serialized network with module
        metadata is imported by module/class name; one without it uses the Keras
        object registry. A live Keras network is cloned from its configuration.
        Legacy show_network_summary is discarded; weights and optimizer slots are not restored.

        Args:
            config (Mapping[str, object]): Constructor configuration with a required
                network entry and optional wrapper/Keras settings. Runtime teacher
                objects are normally absent from saved configurations.

        Returns:
            DiffusionModel: New instance of cls, including specialized subclasses,
            with independent raw/EMA state initialized by its constructor.

        Raises:
            KeyError: Required serialized network fields are absent.
            ImportError: An explicitly named network module cannot be imported.
            Exception: Keras deserialization and constructor incompatibilities propagate.
        """

        config = dict(config)
        config.pop("show_network_summary", None)
        network = config["network"]
        # Serialized raw-network mappings need deserialization before wrapper construction.
        if isinstance(network, Mapping):
            network_config = dict(network)
            module_name = network_config.pop("module", None)

            # Use Keras registration when serialization provides no explicit Python module.
            if module_name is None:
                network = tf.keras.utils.deserialize_keras_object(
                    network_config
                )
            # Resolve repository network classes through their stored module and class name.
            else:
                network_type = getattr(
                    import_module(module_name), 
                    network_config["class_name"].rsplit(">", 1)[-1]
                )
                network = network_type.from_config(network_config["config"])
        # Clone a supplied live network configuration so reconstruction owns an independent model.
        elif isinstance(network, tf.keras.Model):
            network = network.__class__.from_config(network.get_config())

        config["network"] = network

        return cls(**config)

    def build(self, input_shape: object | None = None) -> None:
        """Build execution networks; Sequential holders only track replacements.

        input_shape=None supplies no wrapper input-shape metadata; child models build
        from their own saved configurations. The argument is shape metadata (for example
        a tuple or TensorShape), not an image array with a numerical dtype.

        Args:
            input_shape (object | None): Optional Keras input-shape metadata passed
                to the wrapper's base build; child networks own their geometry.
                Defaults to ``None``.

        Returns:
            result (None): Builds any unbuilt raw/EMA networks and marks this
                wrapper built without rebuilding existing child weights.

        Raises:
            No explicit exceptions are raised here. Native raw/EMA build and delegated Keras
                Model.build errors propagate; a successful earlier build is not rolled back if a
                later build fails.
        """

        for network in (self.network, self.ema_network):
            # Build each existing prediction branch before marking the wrapper built.
            if network is not None and not network.built:
                network.build()

        super().build(input_shape)

    def compile(
        self, 
        loss: losses.Loss | str = "mse", 
        **kwargs: object
    ) -> None:
        """Configure the compiled prediction loss, optimizer, and trackers.

        Args:
            loss (tf.keras.losses.Loss | str): Per-example/base loss used for
                both noise and image reconstruction, default ``"mse"``.
                Defaults to ``'mse'``.
            **kwargs (object): Keras compile options, empty by default. Accepted keys
                include optimizer
                (instance/name), run_eagerly, steps_per_execution, jit_compile where
                supported, metrics, weighted_metrics, and loss_weights. Omitted values
                retain Keras defaults, including optimizer="rmsprop" in the supported
                Keras implementation. Custom steps report the wrapper-owned trackers.

        Returns:
            None: Configures the compiled prediction loss and optimizer, installs sparse
            cross-entropy, and resets the diffusion/auxiliary metric trackers created
            during construction, including currently disabled objectives. With
            trainable_teacher=True, an attached teacher also receives an independent
            optimizer and the same compile settings unless compile_teacher
            configured it explicitly. Each supported role retains independent
            training state.

        Raises:
            ValueError: A selected optimizer is the same object as another training owner's
                optimizer, or shares its iteration variable.
            Keras compile/optimizer deserialization errors propagate, as do attached native-teacher
                compilation failures. The student may already be compiled when a later teacher
                compilation fails.
        """

        # Reject aliases before replacing any existing student compile state.
        for teacher_name in self.get_teacher_names():
            network = self.get_teacher_network(teacher_name)
            # Student compilation cannot take over an attached teacher optimizer.
            if network is not None:
                teacher = getattr(
                    self, self._teacher_state_attribute(teacher_name), None
                )
                self._check_teacher_optimizer(kwargs.get("optimizer"), teacher)

        self._requested_jit_compile = kwargs.get("jit_compile", "auto")
        self._refresh_jit_support()
        super().compile(loss=loss, **kwargs)

        self.scce_loss_fn = losses.sparse_categorical_crossentropy
        self.reset_metrics()

        # All subclass metric state is created by its constructor before this lock.
        if not self.built:
            self.build(())

        # Teacher compilation uses fresh optimizer state, separate from the student.
        if self.trainable_teacher:
            teacher_names = self.get_teacher_names()[2:] if any(
                self.get_teacher_network(name) is not None for name in self.get_teacher_names()[2:]
            ) else ("previous", "current")
            for teacher_name in teacher_names:
                # Each trainable attachment owns independent compilation and update state.
                if self.get_teacher_network(teacher_name) is not None:
                    self._compile_teacher(teacher_name)

    def fit(
        self, 
        x: object | None = None, 
        y: object | None = None, 
        **kwargs: object
    ) -> callbacks.History:
        """Fit under configured training timestep bounds, then restore bounds.

        For array input, images normally have shape [N,H,W,C] or [N,H,W] and numeric
        external-pixel dtype; sparse targets are integer [N] or [N,1]. Dataset elements
        are batched versions of that contract (or the documented prepared tuples).
        Raw preparation applies the wrapper's preprocessing, not loader-fitted statistics.
        None y uses dataset-carried labels; None x is forwarded for Keras input handling.
        The returned History or evaluation scalars are Keras/Python result objects, not
        an image tensor with a batch dtype.

        Args:
            x (tf.data.Dataset | object | None): Keras input yielding
                ``(images, labels)``; images are float ``[B,H,W,C]`` (normally
                raw ``[0,255]``) and labels are integer ``[B]``. When
                ``map_preprocess=True``, this must be a ``tf.data.Dataset`` and
                is mapped through :meth:`prep_inputs_map` before fitting.
                Defaults to ``None``.
            y (tf.data.Dataset | object | None): Optional separate Keras targets;
                custom steps normally consume labels from ``x`` instead.
                Defaults to ``None``.
            **kwargs (object): Forwarded to ``tf.keras.Model.fit``.  Accepted standard
                keys include ``batch_size``, ``epochs``, ``verbose``,
                ``callbacks``, ``validation_data``, ``shuffle``,
                ``steps_per_epoch``, ``validation_steps``, and
                ``initial_epoch``.

        Returns:
            tf.keras.callbacks.History: Keras training history.  Entry timestep
            bounds are restored even when Keras raises an exception.

        Raises:
            ValueError: Dynamic class discovery encounters a known-infinite dataset or delegated
                class growth is incompatible. Keras input/compile validation, dataset mapping and
                callback exceptions propagate. The active bounds and preprocessing mode are
                restored, but completed weight/vocabulary updates are retained.
        """

        self._check_new_labels(
            x=x, y=y, 
            verbose=kwargs.get("verbose", True)
        )

        prev_t_min = self._active_min_timestep
        prev_t_max = self._active_max_timestep
        self.set_timestep_bounds(
            self.train_noisified_min_timesteps, 
            self.train_noisified_max_timesteps
        )

        try:
            # Prepare dataset batches in the input pipeline when requested.
            if self.map_preprocess:
                self._preprocess_training = True
                x = x.map(
                    self.prep_inputs_map, 
                    num_parallel_calls=self.map_num_parallel_calls
                )

                validation_data = kwargs.get("validation_data")
                # Apply the equivalent clean-input preparation to validation.
                if validation_data is not None:
                    train_t_min = self._active_min_timestep
                    train_t_max = self._active_max_timestep
                    self.set_timestep_bounds(
                        self.test_noisified_min_timesteps, 
                        self.test_noisified_max_timesteps
                    )

                    try:
                        self._preprocess_training = False
                        kwargs["validation_data"] = validation_data.map(
                            self.prep_inputs_map, 
                            num_parallel_calls=self.map_num_parallel_calls
                        )
                    finally:
                        self.set_timestep_bounds(
                            train_t_min, 
                            train_t_max
                        )

                self._preprocess_training = None

            return super().fit(x=x, y=y, **kwargs)
        finally:
            self._preprocess_training = None
            self.set_timestep_bounds(
                prev_t_min, 
                prev_t_max
            )

    def evaluate(
        self, 
        x: object | None = None, 
        y: object | None = None, 
        network_name: NetworkName | None = None, 
        **kwargs: object
    ) -> float | list[float] | dict[str, float]:
        """Evaluate the raw or EMA network under test timestep bounds.

        For array input, images normally have shape [N,H,W,C] or [N,H,W] and numeric
        external-pixel dtype; sparse targets are integer [N] or [N,1]. Dataset elements
        are batched versions of that contract (or the documented prepared tuples).
        Raw preparation applies the wrapper's preprocessing, not loader-fitted statistics.
        None y uses dataset-carried labels; None x is forwarded for Keras input handling.
        The returned History or evaluation scalars are Keras/Python result objects, not
        an image tensor with a batch dtype.

        Args:
            x (tf.data.Dataset | object | None): Keras input yielding image and
                label tensors. When ``map_preprocess=True``, this must be a
                ``tf.data.Dataset`` and is mapped through
                :meth:`prep_inputs_map` before evaluation.
                Defaults to ``None``.
            y (tf.data.Dataset | object | None): Optional separate targets.
                Defaults to ``None``.
            network_name (NetworkName | None): ``"ema"`` or ``"raw"`` for this call.
                With ``use_ema=False``, ``"ema"`` resolves to the raw network.
                None inherits ``test_network_name``, including validation in fit.
                Defaults to ``None``.
            **kwargs (object): Forwarded to ``tf.keras.Model.evaluate``.  Standard keys
                include ``batch_size``, ``verbose``, ``sample_weight``, ``steps``,
                ``callbacks``, and ``return_dict``.

        Returns:
            float | list[float] | dict[str, float]: Standard Keras evaluation
            result.  Active timestep bounds and the previously selected test
            network are restored even when Keras raises an exception.

        Raises:
            ValueError: An explicitly selected network is unsupported or delegated label
                preprocessing cannot map the dataset classes. Keras evaluation/data errors
                propagate; temporary network selection, preprocessing mode and timestep bounds are
                restored in the finally block.
        """

        # Validation inside fit temporarily changes bounds, whose setter clears
        # cached functions. Keep the active fit trace for the restored train
        # bounds: Keras does not rebuild it between validation and the next epoch.
        # Omitted selectors inherit the configured evaluation branch.
        network_name = self.test_network_name if network_name is None else network_name

        prev_train_function = self.train_function
        prev_t_min = self._active_min_timestep
        prev_t_max = self._active_max_timestep
        self.set_timestep_bounds(
            self.test_noisified_min_timesteps, 
            self.test_noisified_max_timesteps
        )

        prev_test_network_name = self.test_network_name
        # Rebuild the test function when evaluation switches network variants.
        if network_name != self.test_network_name:
            self.test_network_name = network_name
            self.test_function = None

        try:
            # Prepare evaluation batches in the CPU input pipeline on request.
            if self.map_preprocess:
                # Keras calls this override for validation inside ``fit``. Its
                # validation dataset was already prepared above, so do not map
                # the resulting seven/eight-tensor element a second time.
                already_prepared = self._is_prepared_dataset_spec(
                    x.element_spec
                )
                # Map only raw two-tensor image/label dataset elements.
                if not already_prepared:
                    self._preprocess_training = False
                    x = x.map(
                        self.prep_inputs_map, 
                        num_parallel_calls=self.map_num_parallel_calls
                    )
                    self._preprocess_training = None

            return super().evaluate(x=x, y=y, **kwargs)
        finally:
            self._preprocess_training = None
            self.set_timestep_bounds(
                prev_t_min, 
                prev_t_max
            )

            # Restore the prior test network and invalidate the temporary trace.
            if prev_test_network_name != self.test_network_name:
                self.test_network_name = prev_test_network_name
                self.test_function = None
            self.train_function = prev_train_function

    def summary(self, **kwargs: object) -> None:
        """Print/return the raw network's Keras model summary.

        Args:
            **kwargs (object): Forwarded to ``network.summary``; supported keys include
                ``line_length``, ``positions``, ``print_fn``, ``expand_nested``,
                and ``show_trainable`` (TensorFlow-version dependent).

        Returns:
            None: Keras summary output is written through ``print_fn``.

        Raises:
            No explicit exceptions are raised here; raw-network Keras summary errors, including an
                unsupported summary keyword, propagate.
        """

        return self.network.summary(**kwargs)

    def save_weights(
        self, 
        filepath: str, 
        create_dir: bool = True, 
        overwrite: bool = True
    ) -> None:
        """Save model weights, optionally creating missing parent directories.

        Args:
            filepath (str): Destination ending in ``.weights.h5`` under
                Keras 3. Includes tracked raw/EMA weights and random state.
            create_dir (bool): Recursively create missing parent directories
                when True. Defaults to True.
            overwrite (bool): Whether to overwrite an existing weights file
                without prompting. Defaults to True.

        Returns:
            result (None): Weights are written to ``filepath``. The wrapper
                is built if needed; this file alone is not a task checkpoint.

        Raises:
            ValueError: Keras rejects the filename or unbuildable model state.
            OSError: The destination cannot be created or written.
        """

        # Create missing parent directories only when the caller requests it.
        if create_dir:
            parent_dir = os.path.dirname(filepath)
            # A filename in the current directory has no parent to create.
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        # Expose tracked weights before Keras validates the save operation.
        if not self.built:
            self.build(())

        return super().save_weights(filepath, overwrite=overwrite)

    def load_weights(
        self, 
        filepath: str, 
        *args: object, 
        **kwargs: object
    ) -> None:
        """Restore tracked weights, building the wrapper before loading.

        Args:
            filepath (str): Keras-compatible weights file for the already
                configured raw/EMA architecture and numeric policy.
            *args (object): Positional options forwarded to Keras loading.
            **kwargs (object): Keras loading options such as ``skip_mismatch``;
                its default false rejects incompatible weight shapes.

        Returns:
            result (None): Keras restores matching variables in place. The
                wrapper need not be compiled; loading does not reconstruct
                class/depth growth metadata or the continual task cursor.

        Raises:
            ValueError: The format, model geometry, or weights are incompatible.
            TypeError: Keras rejects a forwarded loading option.
            OSError: The file is missing, unreadable, or invalid.
        """

        # Expose tracked weights before Keras matches checkpoint variables.
        if not self.built:
            self.build(())

        return super().load_weights(filepath, *args, **kwargs)

    def train_step(
        self, 
        inputs: tuple[tf.Tensor, ...]
    ) -> dict[str, tf.Tensor]:
        """Perform one joint diffusion optimization step on the raw network.

        Raw batches contain numeric external images [B,H,W,C] (or [B,H,W] before
        channel insertion) and sparse integer labels [B]. Prepared batch layout/dtypes
        follow the relevant prep_inputs/prep_clfv2_inputs adapter; classifier variants
        can carry bool/numeric replay provenance [B] and floating per-role teacher
        targets. Returned metric values are scalar tensors in their trackers' policy
        variable dtype. Raw preparation advances the appropriate saved random streams;
        prepared tensors reuse their already sampled corruption.

        Args:
            inputs (tuple[tf.Tensor, ...]): Clean images and integer classes, or
                seven prepared tensors plus the optional noise-teacher
                prediction and mask when ``map_preprocess=True``.

        Returns:
            dict[str, tf.Tensor]: Running enabled loss/accuracy metrics.  Noise
            loss is always present; total/image/KL/regularizer values appear
            according to active loss flags.

        Raises:
            No independent input validator runs here. Errors from input preparation, teacher/network
                execution, compiled loss, gradient application and EMA compatibility propagate. A
                failure after optimizer application does not undo that update.
        """

        # Keras' on-batch APIs include an absent sample-weight placeholder.
        if len(inputs) == 3 and inputs[2] is None:
            inputs = inputs[:2]

        # Prepare raw pairs passed directly to a wrapper configured for mapped training.
        if self.map_preprocess and len(inputs) == 2:
            self._preprocess_training = True
            prepared_inputs = self.prep_inputs_map(*inputs)
            self._preprocess_training = None
        # Online preparation handles unmapped batches; mapped batches already contain diffusion
        # tensors.
        else:
            # Noising occurs inside this step only when input-pipeline preparation is disabled.
            prepared_inputs = self.prep_inputs(
                inputs
            ) if not self.map_preprocess else inputs
        # Detach cached teacher noise and vocabulary masks from teacher-enabled batches.
        if self.use_noise_distil_loss:
            teacher_noises_pred = prepared_inputs[-2]
            teacher_noise_mask = prepared_inputs[-1]
            prepared_inputs = prepared_inputs[:-2]
        # Teacher-free training passes no noise target or teacher-vocabulary mask.
        else:
            teacher_noises_pred = None
            teacher_noise_mask = None
        (x0, noises, t, x_t, cfg_labels, 
        uncond_labels, classes) = prepared_inputs

        with tf.GradientTape() as tape:
            outputs = self.forward_and_compute_loss(
                "raw", x0, noises, t, x_t, 
                cond_labels=cfg_labels, 
                uncond_labels=uncond_labels, 
                classes=classes, 
                cfg_scale=self.train_cfg_scale, 
                teacher_noises_pred=teacher_noises_pred, 
                teacher_noise_mask=teacher_noise_mask, 
                training=True
            )
            (loss, noise_loss, cond_noise_loss, 
            uncond_noise_loss, noise_distil_loss, 
            image_loss, kl_loss, ctr_loss, 
            ctr_preds) = outputs

        self.apply_grads(tape, loss)
        self.update_ema()
        results = self.get_results_dict(
            noise_loss, 
            cond_noise_loss=cond_noise_loss, 
            uncond_noise_loss=uncond_noise_loss, 
            noise_distil_loss=noise_distil_loss, 
            teacher_noise_mask=teacher_noise_mask, 
            total_loss=loss, 
            image_loss=image_loss, 
            kl_loss=kl_loss, 
            ctr_loss=ctr_loss, 
            ctr_preds=ctr_preds, 
            classes=classes, 
            cond_labels=cfg_labels
        )

        return results

    def test_step(
        self, 
        inputs: tuple[tf.Tensor, ...]
    ) -> dict[str, tf.Tensor]:
        """Evaluate one batch using the configured raw/EMA test network.

        Raw batches contain numeric external images [B,H,W,C] (or [B,H,W] before
        channel insertion) and sparse integer labels [B]. Prepared batch layout/dtypes
        follow the relevant prep_inputs/prep_clfv2_inputs adapter; classifier variants
        can carry bool/numeric replay provenance [B] and floating per-role teacher
        targets. Returned metric values are scalar tensors in their trackers' policy
        variable dtype. Raw preparation advances the appropriate saved random streams;
        prepared tensors reuse their already sampled corruption.

        Args:
            inputs (tuple[tf.Tensor, ...]): Clean images and integer classes, or
                seven prepared tensors plus the optional noise-teacher
                prediction and mask when ``map_preprocess=True``.

        Returns:
            dict[str, tf.Tensor]: Running evaluation metrics.  Image loss is
            explicitly evaluated even when its training coefficient is zero.

        Raises:
            No independent input validator runs here. Errors from input preparation, teacher/network
                execution, compiled loss and metric input validation propagate; no optimizer update
                is attempted.
        """

        # Keras' on-batch APIs include an absent sample-weight placeholder.
        if len(inputs) == 3 and inputs[2] is None:
            inputs = inputs[:2]

        # Prepare raw evaluation pairs with label dropout disabled in mapped mode.
        if self.map_preprocess and len(inputs) == 2:
            self._preprocess_training = False
            prepared_inputs = self.prep_inputs_map(*inputs)
            self._preprocess_training = None
        # Use precomputed tensors in mapped evaluation and prepare clean pairs otherwise.
        else:
            # Online evaluation preparation is needed only without mapped preprocessing.
            prepared_inputs = self.prep_inputs(
                inputs, 
                use_label_dropout=False
            ) if not self.map_preprocess else inputs
        # Extract the precomputed teacher prediction and valid-condition mask for evaluation.
        if self.use_noise_distil_loss:
            teacher_noises_pred = prepared_inputs[-2]
            teacher_noise_mask = prepared_inputs[-1]
            prepared_inputs = prepared_inputs[:-2]
        # Teacher-free evaluation supplies no distillation target or mask.
        else:
            teacher_noises_pred = None
            teacher_noise_mask = None
        (x0, noises, t, x_t, cond_labels, 
        uncond_labels, classes) = prepared_inputs

        outputs = self.forward_and_compute_loss(
            self.test_network_name, 
            x0, noises, t, x_t, 
            cond_labels=cond_labels, 
            uncond_labels=uncond_labels, 
            classes=classes, 
            cfg_scale=self.test_cfg_scale, 
            teacher_noises_pred=teacher_noises_pred, 
            teacher_noise_mask=teacher_noise_mask, 
            use_image_loss=True, 
            training=False
        )
        (loss, noise_loss, cond_noise_loss, 
        uncond_noise_loss, noise_distil_loss, 
        image_loss, kl_loss, ctr_loss, 
        ctr_preds) = outputs

        results = self.get_results_dict(
            noise_loss, 
            cond_noise_loss=cond_noise_loss, 
            uncond_noise_loss=uncond_noise_loss, 
            noise_distil_loss=noise_distil_loss, 
            teacher_noise_mask=teacher_noise_mask, 
            total_loss=loss, 
            image_loss=image_loss, 
            kl_loss=kl_loss, 
            ctr_loss=ctr_loss, 
            ctr_preds=ctr_preds, 
            classes=classes, 
            cond_labels=cond_labels, 
            use_image_loss=True
        )

        return results

    def fit_progressively(
        self, 
        stage_tasks: Sequence[str | tuple | set | dict] | 
                    Literal[
                        "timesteps_only", 
                        "resolutions_only", 
                        "depths_only"
                    ], 
        stages_num: int | None = None, 
        stages_verbose: bool = True, 
        stage_epochs: int = 1, 
        final_epochs: int | None = None, 
        timestep_boundaries: Sequence[tuple[int, int] | None] | None = None, 
        timestep_clustering_type: ClusteringType = "log_snr", 
        resolutions: Sequence[int | None] | None = None, 
        depths: Sequence[object | None] | None = None, 
        pacing_type: Literal["fixed", "plateau"] = "fixed", 
        earlystopping_type: Literal["batch_wise", "epoch_wise"] = "epoch_wise", 
        monitor: str = "val_noise_loss", 
        patience: int = 10, 
        min_delta: float = 1e-3, 
        stopper_mode: str = "min", 
        **fit_kwargs: object
    ) -> callbacks.History:
        """Train through a user-defined sequence of progressive stages.

        Pass ``stage_tasks`` as a list for a mixed curriculum. Each element
        describes only the values that change before that training stage. A
        value not mentioned by the element keeps its value from the previous
        stage. Timestep ranges and resolutions can change separately or
        together and are applied before their stage. Depth additions are applied
        after their stage, retaining trained layers, optimizer state and old EMA
        values. The complete depth schedule is validated on a separate model
        before class discovery or training.

        ``stage_tasks="timesteps_only"`` creates one timestep task for every
        entry in ``timestep_boundaries``. If those boundaries are omitted,
        ``stages_num`` stages are generated with
        ``_get_progressive_timestep_boundaries``. Likewise,
        ``stage_tasks="resolutions_only"`` uses every entry in ``resolutions``
        or generates ``stages_num`` low-to-high power-of-two resolution stages.
        For example, three generated resolution stages are
        ``[image_size // 4, image_size // 2, image_size]``.
        ``stage_tasks="depths_only"`` creates one stage for every entry in
        ``depths``; depth specifications cannot be generated automatically.

        Args:
            stage_tasks (Sequence[str | tuple | set | dict] | Literal[ "timesteps_only", "resolutions_only", "depths_only"]):
                A list
                of ordered stage descriptions, or
                ``"timesteps_only"``, ``"resolutions_only"``, or
                ``"depths_only"``. A list's length is the number of training
                stages. Strings and two-item tuples change one value; sets and
                dictionaries may combine timestep/resolution updates and
                depth entries.
            stages_num (int | None): Optional number of generated stages. For an explicit
                mixed task list, its length determines the stage count. In
                either ``*_only`` mode, supplied values determine the count.
                ``stages_num`` is therefore needed only when values must be
                generated.
                Defaults to ``None``.
            stages_verbose (bool): Whether to print each stage's resolved state.
                Defaults to ``True``.
            stage_epochs (int): Number of epochs allocated to every listed stage.
                With plateau pacing, this is the maximum for each stage.
                Defaults to ``1``.
            final_epochs (int | None): Epochs for a final full-timestep, native-resolution
                stage. ``None`` uses ``stage_epochs`` and ``0`` disables it.
                Defaults to ``None``.
            timestep_boundaries (Sequence[tuple[int, int] | None] | None): Optional
                stage-indexed sequence of
                ``(lower_bound, upper_bound)`` pairs. An entry is read only when
                the corresponding task requests ``"timesteps"`` without an
                inline pair, so unused positions may be ``None``. When omitted,
                cumulative easy-to-hard ranges are generated from ``stages_num``.
                Defaults to ``None``.
            timestep_clustering_type (ClusteringType): It is only used when the method
                automatically
                generates timestep boundaries, and it can be one of ('uniform', 'log_snr').
                Defaults to ``'log_snr'``.
            resolutions (Sequence[int | None] | None): Optional stage-indexed resolution
                values. An entry is
                read only when the corresponding task requests ``"resolution"``
                without an inline value, so unused positions may be ``None``.
                Values may increase, decrease, repeat, or exceed ``image_size``;
                the network's normal resolution requirements still apply. When
                omitted, ``stages_num`` low-to-high resolutions are generated by
                repeatedly dividing ``image_size`` by powers of two.
                Defaults to ``None``.
            depths (Sequence[object | None] | None): Optional stage-indexed depth
                specifications. An entry is
                read only when the corresponding task requests ``"depth"``
                without an inline value. Empty specifications request no added
                layers; nonempty specifications must preserve existing output-head
                shapes and use the raw network's supported stage types.
                Defaults to ``None``.
            pacing_type (Literal["fixed", "plateau"]): ``"fixed"`` requests ``stage_epochs``
                without an added plateau callback.
                Caller callbacks may still stop early. ``"plateau"``
                may advance sooner using the selected early-stopping callback.
                Defaults to ``'fixed'``.
            earlystopping_type (Literal["batch_wise", "epoch_wise"]): Under plateau pacing,
                ``"epoch_wise"`` uses
                Keras ``EarlyStopping`` and ``"batch_wise"`` uses
                ``BatchLossPlateau``.
                Defaults to ``'epoch_wise'``.
            monitor (str): Metric name monitored by plateau pacing.
                Defaults to ``'val_noise_loss'``.
            patience (int): Number of non-improving epochs or batches tolerated by
                the selected early-stopping callback.
                Defaults to ``10``.
            min_delta (float): Minimum monitored improvement.
                Defaults to ``0.001``.
            stopper_mode (str): ``min``, ``max``, or ``auto`` direction for both
                epoch-wise and batch-wise plateau pacing.
                Defaults to ``'min'``.
            **fit_kwargs (object): Normal Keras ``fit`` arguments such as ``x``,
                ``validation_data``, ``callbacks``, ``steps_per_epoch`` and
                ``verbose``. ``epochs`` and ``initial_epoch`` are managed here.

        Returns:
            history (tf.keras.callbacks.History): Merged metrics and a
                ``progressive_stages`` record of every resolved stage, including
                its network depth and any ``depth_growth`` result. The
                model's timestep bounds and resolution are restored to their
                entry values after completion or interruption. Input data must
                be reiterable because each stage invokes a separate Keras
                ``fit`` call.

        Raises:
            AssertionError: Managed epoch arguments, pacing/monitor choices, or timestep
                bounds violate the progressive training contract.
            ValueError: A shorthand lacks required values/counts, a stage is malformed,
                or delegated growth/resolution/schedule compatibility fails.

        Examples:
            fit_progressively("timesteps_only", stages_num=4, x=dataset)
            fit_progressively("resolutions_only", stages_num=3, x=dataset)
            fit_progressively(
                "resolutions_only", resolutions=[16, 32, 64], x=dataset
            )

        Accepted stage syntax is:

            "timesteps"
            ("timesteps", (lower_bound, upper_bound))
            "resolution"
            ("resolution", resolution_value)
            "depth"
            ("depth", depth_specification)
            {"timesteps", "resolution", "depth"}
            {
                "timesteps": (lower_bound, upper_bound), 
                "resolution": resolution_value, 
                "depth": depth_specification
            }

        A string or set names changes without providing their values. Their
        values are read from ``timestep_boundaries[stage_index]`` and
        ``resolutions[stage_index]`` or ``depths[stage_index]`` respectively.
        A dictionary value of ``None`` has the same meaning. Inline tuple or
        dictionary values take precedence over the companion sequences.

        The depth grammar represents one layer by a string, several
        depths by a list, and several layer types in one depth by a set or
        dictionary, using the raw model's existing ``add_depths`` syntax.
        Appended stages remain part of the model after this call returns.
        ``None``, an empty list, or a list containing only ``None`` requests
        no added layers. A depth stage with an omitted inline value still
        requires its stage-indexed entry in ``depths``.

        For example:

            stage_tasks = [
                {"timesteps": (700, 1000), "resolution": 16},
                "timesteps",
                ("resolution", 32),
                {
                    "timesteps", "resolution"
                },
            ]
            timestep_boundaries = [None, (300, 1000), None, (0, 1000)]
            resolutions = [None, None, None, 64]

        This produces stages ``(700: 1000, 16)``, ``(300: 1000, 16)``,
        ``(300: 1000, 32)``, and ``(0: 1000, 64)``.
        No direction, native-size ceiling, or implicit priority between
        the strategies is imposed.
        """

        self._validate_progressive_growth(
            self, 
            {"stage_tasks": stage_tasks, "depths": depths}
        )

        self._check_new_labels(
            x=fit_kwargs.get("x"), 
            y=fit_kwargs.get("y"), 
            verbose=fit_kwargs.get("verbose", True)
        )

        require(
            "epochs" not in fit_kwargs and "initial_epoch" not in fit_kwargs, 
            "Do not pass epochs/initial_epoch to fit_progressively(); "
            "use stage_epochs and final_epochs instead."
        )
        require(
            timestep_clustering_type in get_args(ClusteringType), 
            "timestep_clustering_type must be one of "
            f"{get_args(ClusteringType)} but not "
            f"{timestep_clustering_type}."
        )
        require(
            pacing_type in (vals:=("fixed", "plateau")), 
            f"pacing_type must be one of {vals} but not {pacing_type}."
        )
        require(
            earlystopping_type in (vals:=("batch_wise", "epoch_wise")), 
            f"earlystopping_type must be one of {vals} but not {earlystopping_type}."
        )

        # Follow the opt-in aggregate metric rename for progressive callbacks.
        if self.show_separate_noise_losses and \
        monitor.removeprefix("val_") == "noise_loss":
            monitor = monitor.replace("noise_loss", "total_noise_loss")

        require(
            monitor.removeprefix("val_") in (vals:=self.metrics_names), 
            f"monitor must be one of {vals} (or with val_) but not {monitor}."
        )

        # Recognize shorthand single-operation curricula; mixed lists define their own stages.
        only_task = stage_tasks if stage_tasks in (
            "timesteps_only", 
            "resolutions_only", 
            "depths_only"
        ) else None
        # Infer timestep-only stage count from the supplied boundaries.
        if only_task == "timesteps_only" and timestep_boundaries is not None:
            stages_num = len(timestep_boundaries)
        # Infer resolution-only stage count from the supplied resolutions.
        elif only_task == "resolutions_only" and resolutions is not None:
            stages_num = len(resolutions)
        # Infer depth-only stage count from the supplied depth specifications.
        elif only_task == "depths_only" and depths is not None:
            stages_num = len(depths)
        # An explicit mixed curriculum has one stage per task description.
        elif only_task is None:
            stages_num = len(stage_tasks)
        # Require a count when shorthand stages cannot be inferred from values.
        elif stages_num is None:
            raise ValueError(
                f"stages_num is required when {only_task!r} values are omitted."
            )

        # Depth stages cannot be generated without explicit layer specifications.
        if only_task == "depths_only" and depths is None:
            raise ValueError(
                "depths must be provided for depths_only training."
            )

        stages_num = int(stages_num)
        # An omitted final-stage budget inherits the per-stage epoch budget.
        final_epochs = stage_epochs if final_epochs is None else int(final_epochs)

        needs_timesteps = only_task == "timesteps_only" or any(
            task == "timesteps" or
            isinstance(task, (set, frozenset)) and "timesteps" in task or
            isinstance(task, dict) and "timesteps" in task and
            task["timesteps"] is None or
            isinstance(task, (tuple, list)) and len(task) == 2 and
            task[0] == "timesteps" and task[1] is None
            for task in stage_tasks
        )
        needs_resolution = only_task == "resolutions_only" or any(
            task == "resolution" or
            isinstance(task, (set, frozenset)) and "resolution" in task or
            isinstance(task, dict) and "resolution" in task and
            task["resolution"] is None or
            isinstance(task, (tuple, list)) and len(task) == 2 and
            task[0] == "resolution" and task[1] is None
            for task in stage_tasks
        )

        # Generate timestep boundaries only when requested and not supplied.
        if needs_timesteps and timestep_boundaries is None:
            boundaries = self._get_progressive_timestep_boundaries(
                stages_num, 
                timestep_clustering_type
            )
            timestep_boundaries = [
                (lower_bound, boundaries[-1])
                for lower_bound in reversed(boundaries[:-1])
            ]

        # Generate a low-to-high resolution sequence when none was supplied.
        if needs_resolution and resolutions is None:
            resolutions = [
                self.image_size // 2**power
                for power in range(stages_num - 1, -1, -1)
            ]

        # Expand a shorthand curriculum into ordinary stage task names.
        if only_task is not None:
            task_name = {
                "timesteps_only": "timesteps", 
                "resolutions_only": "resolution", 
                "depths_only": "depth" 
            }[only_task]
            stage_tasks = [task_name] * stages_num

        user_callbacks = list(fit_kwargs.pop("callbacks", []) or [])
        merged_history = {}
        stage_records = []
        all_epochs = []
        epoch_cursor = 0
        previous_min_timestep = self._active_min_timestep
        previous_max_timestep = self._active_max_timestep
        previous_resolution = self._current_resolution


        def run_stage(
            stage_id: int, 
            updates: dict[str, object], 
            epochs: int, 
            final: bool = False
        ) -> dict[str, object]:
            """Fit one resolved curriculum stage and merge its history.

            Args:
                stage_id (int): One-based stage identifier.
                updates (dict[str, object]): Resolved ``timesteps``,
                    ``resolution``, and/or ``depth`` descriptions for recording.
                epochs (int): Maximum epochs allocated to this invocation.
                final (bool): Label the stage ``"final"`` and skip plateau
                    early stopping when true.
                    Defaults to ``False``.

            Returns:
                dict[str, object]: Stage record containing active bounds,
                resolution, pre-growth network depth, epoch count, and its raw
                Keras history dictionary.

            Raises:
                No independent exception translation is performed. Callback, data mapping and Keras
                    fitting exceptions propagate through the enclosing progressive trainer; completed
                    stage updates remain.
            """

            nonlocal epoch_cursor


            stage_callbacks = list(user_callbacks)
            # Add plateau stopping only to non-final progressive stages.
            if pacing_type == "plateau" and not final:
                # Monitor progress once per epoch with Keras early stopping.
                if earlystopping_type == "epoch_wise":
                    stage_callbacks.append(callbacks.EarlyStopping(
                        monitor=monitor, 
                        min_delta=min_delta, 
                        patience=patience, 
                        mode=stopper_mode, 
                        verbose=stages_verbose
                    ))
                # Monitor progress per batch with the project callback.
                elif earlystopping_type == "batch_wise":
                    stage_callbacks.append(BatchLossPlateau(
                        monitor=monitor.removeprefix("val_"), 
                        patience=patience, 
                        min_delta=min_delta, 
                        mode=stopper_mode
                    ))

            # Print the resolved stage state when progress output is requested.
            if stages_verbose:
                # Distinguish the final full-task stage from numbered curriculum stages in progress
                # output.
                name = "final/full-task" if final \
                    else f"{stage_id}/{len(stage_tasks)}"
                print(
                    f"Progressive stage {name}: changes={updates}, "
                    f"resolution={self._current_resolution}, sampling t in "
                    f"[{self._active_min_timestep}, "
                    f"{self._active_max_timestep}) range."
                )

            stage_fit_kwargs = dict(fit_kwargs)
            # Retrace preparation under this stage's active bounds/resolution.
            # Prepare each progressive dataset under its current stage state.
            if self.map_preprocess:
                stage_x = stage_fit_kwargs.get("x")
                self._preprocess_training = True
                stage_fit_kwargs["x"] = stage_x.map(
                    self.prep_inputs_map, 
                    num_parallel_calls=self.map_num_parallel_calls
                )

                validation_data = stage_fit_kwargs.get("validation_data")
                # Prepare an optional stage validation dataset consistently.
                if validation_data is not None:
                    stage_t_min = self._active_min_timestep
                    stage_t_max = self._active_max_timestep
                    self.set_timestep_bounds(
                        self.test_noisified_min_timesteps, 
                        self.test_noisified_max_timesteps
                    )

                    try:
                        self._preprocess_training = False
                        stage_fit_kwargs["validation_data"] = validation_data.map(
                            self.prep_inputs_map, 
                            num_parallel_calls=self.map_num_parallel_calls
                        )
                    finally:
                        self.set_timestep_bounds(
                            stage_t_min, 
                            stage_t_max
                        )

                self._preprocess_training = None

            history = super(DiffusionModel, self).fit(
                callbacks=stage_callbacks, 
                initial_epoch=epoch_cursor, 
                epochs=epoch_cursor + epochs, 
                **stage_fit_kwargs
            )

            actual_epochs = list(history.epoch)
            all_epochs.extend(actual_epochs)
            epoch_cursor += len(actual_epochs)

            for key, values in history.history.items():
                merged_history.setdefault(key, []).extend(values)

            # Store the final-stage sentinel separately from ordinary one-based stage IDs.
            stage_record = {
                "stage": "final" if final else stage_id, 
                "updates": updates, 
                "min_timestep": self._active_min_timestep, 
                "max_timestep": self._active_max_timestep, 
                "resolution": self._current_resolution, 
                "network_depth": self.network.depth, 
                "epochs_ran": len(actual_epochs), 
                "history": history.history
            }
            stage_records.append(stage_record)

            return stage_record


        try:
            for stage_index, task in enumerate(stage_tasks):
                # Interpret a string as one update whose value comes from its sequence.
                if isinstance(task, str):
                    updates = {task: None}
                # Preserve inline values from a dictionary stage description.
                elif isinstance(task, dict):
                    updates = dict(task)
                # Interpret a set as several updates resolved from companion sequences.
                elif isinstance(task, (set, frozenset)):
                    updates = dict.fromkeys(task)
                # Interpret a two-item sequence as one inline task/value pair.
                elif (
                    isinstance(task, (tuple, list)) and len(task) == 2
                    and isinstance(task[0], str)
                    and task[0] in ("timesteps", "resolution", "depth")
                ):
                    updates = {task[0]: task[1]}
                # Reject malformed or unsupported stage descriptions.
                else:
                    raise ValueError(
                        f"Invalid stage task at index {stage_index}: {task!r}."
                    )

                # Reject misspelled task names before silently training an unchanged stage.
                if set(updates) - {"timesteps", "resolution", "depth"}:
                    raise ValueError(
                        f"Unsupported progressive task at index {stage_index}: {task!r}."
                    )

                # Resolve and apply this stage's timestep bounds.
                if "timesteps" in updates:
                    bounds = updates["timesteps"]
                    # Resolve an omitted inline timestep pair from the stage-indexed boundary
                    # sequence.
                    bounds = timestep_boundaries[stage_index] if bounds is None else bounds
                    bounds = tuple(bounds)
                    self.set_timestep_bounds(*bounds)
                    updates["timesteps"] = (self._active_min_timestep, self._active_max_timestep)

                # Resolve and apply this stage's input resolution.
                if "resolution" in updates:
                    resolution = updates["resolution"]
                    # Resolve an omitted inline resolution from the stage-indexed resolution
                    # sequence.
                    resolution = resolutions[stage_index] if resolution is None else resolution
                    resolution = int(resolution)
                    self.set_current_resolution(resolution)
                    updates["resolution"] = resolution

                # Resolve the depth specification for post-stage growth.
                if "depth" in updates:
                    depth_spec = updates["depth"]
                    # Resolve an omitted inline growth specification from the companion depth
                    # sequence.
                    depth_spec = depths[stage_index] if depth_spec is None else depth_spec
                    updates["depth"] = depth_spec

                stage_record = run_stage(
                    stage_id=stage_index + 1, 
                    updates=updates, 
                    epochs=stage_epochs
                )

                # Grow requested layers after the stage and record the result.
                if "depth" in updates:
                    stage_record["depth_growth"] = self._add_depths(
                        updates["depth"]
                    )
                    stage_record["post_network_depth"] = self.network.depth

            # Run the optional final stage at full timesteps and native resolution.
            if final_epochs > 0:
                self.set_timestep_bounds()
                self.set_current_resolution()

                run_stage(
                    stage_id=len(stage_tasks) + 1, 
                    updates={
                        "timesteps": (
                            self._active_min_timestep, 
                            self._active_max_timestep
                        ), 
                        "resolution": self._current_resolution 
                    }, 
                    epochs=final_epochs, 
                    final=True
                )
        finally:
            self._preprocess_training = None
            self.set_timestep_bounds(
                previous_min_timestep, 
                previous_max_timestep
            )
            self.set_current_resolution(
                previous_resolution
            )

        history = callbacks.History()
        history.set_model(self)
        history.history = merged_history
        history.epoch = all_epochs
        history.progressive_stages = stage_records
        history.timestep_boundaries = timestep_boundaries
        history.stage_tasks = stage_tasks
        history.resolutions = resolutions
        history.depths = depths
        history.stages_num = stages_num

        return history

    def _teacher_fit_methods(self) -> tuple[str, ...]:
        """List native training entry points offered by this wrapper family.

        Args:
            None.

        Returns:
            tuple[str, ...]: ``("fit", "fit_progressively")``. Subclasses extend this list for their
                phase-specific training methods; no state changes.

        Raises:
            This fixed tuple accessor raises no explicit exceptions.
        """

        return ("fit", "fit_progressively")

    def fit_teacher(
        self, 
        x: object | None = None, 
        y: object | None = None, 
        fit_method: str = "fit", 
        teacher_name: TeacherName = "previous", 
        **kwargs: object
    ) -> callbacks.History | dict[str, list]:
        """Fit only the selected native teacher with an independent optimizer.

        Raw images use the teacher owner's preprocessing. Newly created owners inherit
        diffusion settings without EMA, recursive distillation or auxiliary losses;
        attached native wrappers retain their own configuration. Native class discovery
        can grow their condition vocabulary. Teacher weights and its optimizer/metrics
        change; the student's weights and optimizer do not. The finally block restores
        the student resolution on the owner and reattaches/freezes the teacher even
        after a failure, invalidating student traces that captured the old attachment.

        Args:
            x (object | None): Training images (numeric array/tensor [N,H,W,C]) or finite supervised
                tf.data.Dataset batches, as accepted by fit_method. Defaults to None, which omits
                the x keyword when delegating.
            y (object | None): Sparse integer targets [N] or [N,1] for array input. Defaults to
                None; datasets carry their own targets, and specialized fitting methods receive no y
                keyword.
            fit_method (str): Supported entry point on the selected training owner, usually fit or
                fit_progressively; defaults to "fit". V2 owners may expose phase-specific methods.
            teacher_name (TeacherName): Role returned by get_teacher_names; the default ``"previous"``
                selects the historical teacher slot. A valid unattached role is distinct from an
                invalid name.
            **kwargs (object): Forwarded training controls, including epochs, callbacks,
                validation_data, batch_size or progressive-stage settings. Omitted controls retain
                the selected method's defaults. No image dtype conversion is performed by this
                dispatcher itself.

        Returns:
            tf.keras.callbacks.History | dict[str, list]: Exact delegated history;
                standard/progressive fitting returns History, while combined phase fitting may
                return a metric-to-history dictionary.

        Raises:
            ValueError: Teacher training is disabled, the role is unknown/empty, the student is
                uncompiled, fit_method is unsupported, or teacher optimizer state aliases another
                owner. Input, curriculum, model-growth and callback exceptions from delegated
                fitting propagate; updates completed before failure remain.
        """

        # Explicit opt-in preserves the existing frozen-teacher behavior.
        if not self.trainable_teacher or self.get_teacher_network(teacher_name) is None:
            raise ValueError("fit_teacher requires trainable_teacher=True and the selected teacher.")
        # Compilation establishes the independent teacher's loss and optimizer settings.
        if not self.compiled:
            raise ValueError("Call compile before fit_teacher.")

        teacher = self.get_teacher_model(teacher_name)
        # A supplied trainer retains its own supported training entry points.
        if fit_method not in teacher._teacher_fit_methods() \
        or not callable(getattr(teacher, fit_method, None)):
            raise ValueError("fit_method must name a supported wrapper training method.")

        # A teacher may be attached or replaced after the student was compiled.
        if teacher is None or teacher.network is not self.get_teacher_network(teacher_name) \
        or not teacher.compiled:
            self._compile_teacher(teacher_name)
            teacher = self.get_teacher_model(teacher_name)
        # Omit absent inputs so specialized fit signatures retain their own defaults.
        if x is not None:
            kwargs["x"] = x
        # Separate targets are optional for batched datasets and specialized fits.
        if y is not None:
            kwargs["y"] = y

        teacher.set_current_resolution(self._current_resolution)
        teacher.network.trainable = True

        try:
            return getattr(teacher, fit_method)(**kwargs)
        finally:
            # Keep wrapper/raw resolution aligned after a progressive fit or failure.
            teacher.set_current_resolution(self._current_resolution)
            # Class discovery can replace the raw network, even before a fit failure.
            self._attach_fitted_teacher(teacher, teacher_name)

    def compile_teacher(
        self, 
        teacher_name: TeacherName = "previous", 
        **kwargs: object
    ) -> None:
        """Compile only the selected native teacher, forwarding all Keras options.

        Requires trainable_teacher=True and an attachment, but can precede student
        compilation. Defaults belong to the native trainer, including MSE. Marks the
        owner's compile settings explicit after success so subsequent student compile
        calls retain them. Teacher weights are temporarily enabled and always frozen
        afterwards. Its compile state/metric caches may reset; student training state
        is unchanged.

        Args:
            teacher_name (TeacherName): Role returned by get_teacher_names; the default ``"previous"``
                selects the historical teacher slot. A valid unattached role is distinct from an
                invalid name.
            **kwargs (object): Native-wrapper/Keras compile keywords, such as optimizer, loss,
                metrics and run_eagerly. With no overrides the native wrapper uses its compile
                defaults. Supply an optimizer independent of every other training owner.

        Returns:
            None: No value is returned.

        Raises:
            ValueError: Teacher training is disabled, the role is unknown or unattached, or the
                optimizer aliases the student/another teacher.
            TypeError: A forwarded compile keyword or compilation object is rejected by the selected
                trainer/Keras. Compilation failures propagate after the teacher is frozen again.
        """

        # Public compilation is available only for an explicitly trainable teacher.
        if not self.trainable_teacher or self.get_teacher_network(teacher_name) is None:
            raise ValueError("compile_teacher requires trainable_teacher=True and the selected teacher.")

        self._check_teacher_optimizer(kwargs.get("optimizer"), self)
        for other_name in self.get_teacher_names():
            # Current specialists must not share optimizer state with each other either.
            if other_name != teacher_name and self.get_teacher_network(other_name) is not None:
                self._check_teacher_optimizer(
                    kwargs.get("optimizer"), self.get_teacher_model(other_name)
                )

        teacher = self._get_teacher_model(teacher_name)
        teacher.network.trainable = True
        try:
            teacher.compile(**kwargs)
            object.__setattr__(teacher, "_teacher_compile_explicit", True)
        finally:
            # Native trainers need trainable variables during compilation.
            teacher.network.trainable = False

    def set_timestep_bounds(
        self, 
        min_timesteps: int | None = 0, 
        max_timesteps: int | None = -1
    ) -> None:
        """Set the active half-open timestep interval for forward noising.

        Args:
            min_timesteps (int | None): Inclusive lower bound; ``None`` uses 0.
                Defaults to ``0``.
            max_timesteps (int | None): Exclusive upper bound; ``None`` uses 0
                and -1 means the current full schedule length.
                Defaults to ``-1``.

        Returns:
            None: Changed bounds invalidate cached Keras train/test/predict
            functions so traced random ranges are rebuilt.

        Raises:
            AssertionError: Unless bounds are clean-only ``[0,0)`` or satisfy
                ``0 <= min < max <= timesteps``.
        """

        min_timesteps = int(0 if min_timesteps is None else min_timesteps)
        max_timesteps = int(0 if max_timesteps is None else max_timesteps)
        max_timesteps = self.timesteps if max_timesteps == -1 else max_timesteps

        require(
            (min_timesteps == 0 and max_timesteps == 0) or
            0 <= min_timesteps < max_timesteps <= self.timesteps, 
            "Expected clean-only bounds [0, 0) or "
            "0 <= min_timesteps < max_timesteps <= timesteps, "
            f"got [{min_timesteps}, {max_timesteps}) with T={self.timesteps}."
        )

        # Retrace train/test steps only when the active timestep range changes.
        if getattr(self, "_active_min_timestep", None) != min_timesteps or \
        getattr(self, "_active_max_timestep", None) != max_timesteps:
            self._active_min_timestep = min_timesteps
            self._active_max_timestep = max_timesteps

            self.train_function = None
            self.test_function = None
            self.predict_function = None

    def set_current_resolution(self, resolution: int | None = None) -> None:
        """Synchronize active resolution across the wrapper, student copies, and teacher.

        Updates the raw network, the EMA network when present, and an attached
        teacher that exposes set_current_resolution. Teachers without that optional
        method are skipped. Networks validate compatibility through their own setters;
        if a later setter fails, earlier network updates are not rolled back.

        Args:
            resolution (int | None): Square size accepted by the raw network;
                ``None`` restores constructor ``image_size``.
                Defaults to ``None``.

        Returns:
            None: Synchronizes supported network resolution state. A change to the
            wrapper's stored resolution invalidates cached train, test, and predict
            functions; reselecting the same value retains those caches.

        Raises:
            AssertionError: Propagated when the network rejects a nonpositive,
                nonintegral, or patch-incompatible resolution.
        """

        resolution = self.image_size if resolution is None else resolution

        self.network.set_current_resolution(
            resolution
        )
        self.ema_network.set_current_resolution(
            resolution
        ) if self.ema_network is not None else None
        
        for teacher_name in self.get_teacher_names():
            teacher = self.get_teacher_network(teacher_name)
            # Every native teacher follows the same active input geometry.
            if teacher is not None and hasattr(teacher, "set_current_resolution"):
                teacher.set_current_resolution(resolution)

        resolution = int(resolution)
        # Propagate a changed resolution to the raw and EMA networks.
        if getattr(self, "_current_resolution", None) != resolution:
            self._current_resolution = resolution

            self.train_function = None
            self.test_function = None
            self.predict_function = None

            self._refresh_jit_support()
            # Refresh JIT selection only after the wrapper has been compiled.
            if getattr(self, "compiled", False):
                requested = getattr(self, "_requested_jit_compile", "auto")
                requested = False if self.run_eagerly else requested
                # Keras skips setters when the requested value equals the old one.
                self.jit_compile = False
                self.jit_compile = self._resolve_auto_jit_compile(
                ) if requested == "auto" else requested

    def reset_seed(self, seed: int | None) -> None:
        """Reset independent checkpointed diffusion streams for a new task.

        Args:
            seed (int | None): Integer task seed restarts independent named
                streams. None marks configuration unseeded and preserves each
                live stream's base seed and counter. Integers follow
                common.runtime.effective_seed.

        Returns:
            result (None): Updates wrapper/stream seed metadata and resets
                counters only for an integer seed. Trained network and optimizer
                weights are retained.

        Raises:
            ValueError: Seed normalization in effective_seed rejects the supplied value. Named
                stream reset errors propagate; streams reset before a later failure remain reset.
        """

        self.seed = effective_seed(None, seed=seed)
        for name, stream in self._random_streams.items():
            stream.reset_seed(
                derive_seed(self.seed, "diffusion", name)
            )

    def _validate_trainable_teacher(self, network: tf.keras.Model | None, teacher_name: TeacherName) -> None:
        """Validate only teachers that are enabled for explicit native training.

        Args:
            network (tf.keras.Model | None): Native diffusion model or wrapper to attach; None
                clears the selected attachment. This is a model object with its own variable dtypes,
                not an image tensor.
            teacher_name (TeacherName): Role identifier included in the protocol-error message.

        Returns:
            None: No mutation. None or trainable_teacher=False bypasses the native training-protocol
                requirement; frozen inference callables are permitted.

        Raises:
            ValueError: A non-None teacher enabled for training is not an ArgumentSaverModel with
                the native full_return noise interface.
        """

        # Frozen callable teachers remain supported for inference-only distillation.
        if network is not None and self.trainable_teacher and not (
            isinstance(network, ArgumentSaverModel) and self._teacher_uses_native_noise_api(network)
        ):
            raise ValueError("A trainable teacher must be a native diffusion model.")

    def set_current_teacher_network(
        self, 
        network: tf.keras.Model | None, 
        class_ids: Sequence[int] | None = None, 
        task_class_ids: Sequence[int] | None = None, 
        teacher_name: Literal["current", "noise"] = "current"
    ) -> None:
        """Attach a frozen current-task teacher with explicit student-vocabulary maps.

        ``class_ids`` lists student IDs in the teacher's output-column order.
        ``task_class_ids`` limits task-scoped KD to its taught classes and defaults
        to class_ids. Native teachers without an explicit map use leading IDs.
        Runtime teachers and mappings remain outside student serialization.

        Args:
            network (tf.keras.Model | None): Independent raw teacher or diffusion
                wrapper (unwrapped to its raw network); None clears this role.
            class_ids (Sequence[int] | None): Unique nonnegative student class IDs
                in teacher-column order. None uses a native head's leading IDs or
                leaves an external callable's mapping unspecified.
                Defaults to ``None``.
            task_class_ids (Sequence[int] | None): Taught subset of class_ids used
                by task-scoped distillation; None inherits the complete class map.
                Defaults to ``None``.
            teacher_name (Literal["current", "noise"]): Shared current
                teacher or one head specialist; the default retains existing calls.
                Defaults to ``'current'``.

        Returns:
            result (None): Installs/freezes the teacher, resets per-role noise
                trackers, refreshes objective availability, and invalidates compiled
                batch functions. Teacher state stays outside student checkpoints.

        Raises:
            TypeError: A supplied teacher is not callable.
            ValueError: Teacher identity, class mapping, epsilon parameterization,
                or known preprocessing/schedule/geometry metadata conflicts with the student.
        """

        self.get_teacher_network(teacher_name)
        # The previous snapshot retains its own attachment API.
        if teacher_name == "previous":
            raise ValueError("Use set_teacher_network for the previous teacher.")
        # Shared and specialist current teachers must not compete for the same head.
        if network is not None and (
            (teacher_name == "current" and any(
                self.get_teacher_network(name) is not None for name in self.get_teacher_names()[2:]
            )) or (teacher_name != "current" and self.current_teacher_network is not None)
        ):
            raise ValueError("current_teacher_network cannot be combined with per-head teachers.")

        supplied_wrapper = network if isinstance(network, DiffusionModel) else None
        cached_wrapper = getattr(self, self._teacher_state_attribute(teacher_name), None)
        # Native wrapper metadata survives unwrapping onto its independent raw network.
        if supplied_wrapper is not None:
            network = supplied_wrapper.network
            network._diffusion_scheduler_name = supplied_wrapper.scheduler_name
            network._diffusion_modify_first_t = supplied_wrapper.modify_first_t
            network._diffusion_swap_noise_image = supplied_wrapper.swap_noise_image
            network._diffusion_preprocess_type = supplied_wrapper.preprocess_type
            object.__setattr__(network, "_diffusion_seen_classes", dict(supplied_wrapper.seen_classes))

        # Both teacher roles must remain independent of the student's live branches.
        if network is not None and (
            network is self.network or network is self.ema_network
            or any(network is self.get_teacher_network(name)
                   for name in self.get_teacher_names() if name != teacher_name)
        ):
            raise ValueError("current_teacher_network must be an independent frozen teacher.")
        # External teachers must implement ordinary callable inference.
        if network is not None and not callable(network):
            raise TypeError("current_teacher_network must be a callable model.")

        # Native teachers consume student model coordinates, including for class targets.
        if network is not None and hasattr(network, "_diffusion_preprocess_type"):
            # Explicit passthrough None is known metadata, not an absent declaration.
            if network._diffusion_preprocess_type != self.preprocess_type:
                raise ValueError("current_teacher_network preprocess_type must match the student.")

        # Clearing the current slot also clears stale vocabulary metadata.
        if network is None:
            class_ids = task_class_ids = None
        # Native output metadata supplies the conventional leading mapping when omitted.
        elif class_ids is None and getattr(network, "num_classes", None) is not None \
        and teacher_name == "current":
            class_ids = tuple(range(int(network.num_classes)))

        mappings = []
        for name, values in (
            ("class_ids", class_ids), 
            ("task_class_ids", task_class_ids)
        ):
            # An unspecified map is valid for a callable without class metadata.
            if values is None:
                mappings.append(None)
                continue

            values = tuple(values)
            # A class map must be nonempty, unique, nonnegative integer IDs.
            if not values or any(
                isinstance(value, (bool, np.bool_)) 
                or not isinstance(value, (int, np.integer))
                or value < 0 for value in values
            ) or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique nonnegative integer class IDs.")

            mappings.append(tuple(int(value) for value in values))

        class_ids, task_class_ids = mappings
        task_class_ids = class_ids if task_class_ids is None else task_class_ids

        # Every taught class must correspond to an actual teacher output column.
        if class_ids is not None and task_class_ids is not None \
        and not set(task_class_ids).issubset(class_ids):
            raise ValueError("task_class_ids must be a subset of class_ids.")
        # A declared native head width must agree with the supplied column mapping.
        if network is not None and class_ids is not None and \
        getattr(network, "num_classes", None) is not None:
            if len(class_ids) != int(network.num_classes):
                # Mismatched map widths otherwise silently relabel or omit teacher outputs.
                raise ValueError("class_ids must match the current teacher's output class count.")
        # Noise objectives require the same epsilon parameterization and forward process.
        if network is not None and teacher_name in ("current", "noise") \
        and self.noise_distil_loss_coef > 0. and self.current_teacher_noise_loss_weight > 0.:
            native = self._teacher_uses_native_noise_api(network)
            # An image-reconstruction teacher cannot supply an epsilon target.
            if getattr(network, "swap_noise_image", getattr(network, "_diffusion_swap_noise_image", False)):
                raise ValueError("An x0-prediction wrapper cannot teach epsilon distillation.")

            for name, fallback, expected in (
                ("scheduler_name", "_diffusion_scheduler_name", self.scheduler_name), 
                ("modify_first_t", "_diffusion_modify_first_t", self.modify_first_t)
            ):
                value = getattr(network, name, getattr(network, fallback, None))
                # Missing external process metadata is permitted, but known values must match.
                if value is not None and value != expected:
                    raise ValueError(f"current_teacher_network {name} must match the student.")

            for name in ("timesteps", "channels", "use_cfg"):
                value = getattr(network, name, None)
                # Native teachers require metadata; external teachers validate available values.
                if (native or value is not None) and value != getattr(self.network, name, None):
                    raise ValueError(f"current_teacher_network {name} must match the student.")

        # Clearing the only noise teacher requires the established deferred-teacher lifecycle.
        if network is None and self.teacher_network is None \
        and not any(self.get_teacher_network(name) is not None
                    for name in ("current", "noise") if name != teacher_name) \
        and self.noise_distil_loss_coef > 0. \
        and (self.previous_teacher_noise_loss_weight > 0. or self.current_teacher_noise_loss_weight > 0.) \
        and not self.defer_teacher:
            raise ValueError("Noise distillation requires an attached teacher or defer_teacher=True.")

        self._validate_trainable_teacher(network, teacher_name)

        self._remember_teacher_fit_state(network, teacher_name)
        object.__setattr__(self, f"{teacher_name}_teacher_network", network)
        object.__setattr__(self, f"{teacher_name}_teacher_class_ids", class_ids)
        object.__setattr__(self, f"{teacher_name}_teacher_task_class_ids", task_class_ids)
        # Supplied native wrappers retain their own optimizer, configuration and vocabulary.
        if supplied_wrapper is not None:
            object.__setattr__(self, self._teacher_state_attribute(teacher_name), supplied_wrapper)
            # Explicit or recovered native compilation must not be replaced by student defaults.
            if supplied_wrapper.compiled and supplied_wrapper is not cached_wrapper:
                object.__setattr__(supplied_wrapper, "_teacher_compile_explicit", True)
        # Runtime current-task teachers are always frozen during student training.
        if network is not None:
            network.trainable = False
            self.defer_teacher = True
            self._init_config["defer_teacher"] = True
            # Resolution-aware native teachers follow the student's current geometry.
            if hasattr(network, "set_current_resolution"):
                network.set_current_resolution(self._current_resolution)

        self.previous_teacher_noise_distil_loss_tracker.reset_state()
        self.current_teacher_noise_distil_loss_tracker.reset_state()
        DiffusionModel._refresh_loss_flags(self)
        self.map_preprocess = bool(self.use_noise_distil_loss) \
                            or self._map_preprocess_without_teacher
        self.train_function = None
        self.test_function = None
        self.predict_function = None

    def set_noise_teacher_network(
        self, 
        network: tf.keras.Model | None, 
        class_ids: Sequence[int] | None = None, 
        task_class_ids: Sequence[int] | None = None
    ) -> None:
        """Attach, replace or clear the current-task noise specialist.

        Delegates to set_current_teacher_network with teacher_name="noise"; freezes
        the attachment, remembers its training owner, refreshes loss flags, and
        invalidates student execution caches. Does not fit or copy teacher weights.

        Args:
            network (tf.keras.Model | None): Native diffusion model or wrapper to attach; None
                clears the selected attachment. This is a model object with its own variable dtypes,
                not an image tensor.
            class_ids (Sequence[int] | None): One student class ID per teacher condition column; -1
                denotes an unavailable column. Defaults to None, using persistent vocabulary
                metadata or native leading columns.
            task_class_ids (Sequence[int] | None): Student IDs taught by this specialist. Defaults
                to None, resolving from the class map or available teacher metadata.

        Returns:
            None: No value is returned.

        Raises:
            ValueError: Specialist/shared-current attachments conflict, the model is incompatible or
                unbuilt for training, or class/taught-support metadata is structurally invalid.
                Refreshed loss-configuration assertions propagate.
        """

        self.set_current_teacher_network(
            network, class_ids, task_class_ids, teacher_name="noise"
        )

    def set_teacher_network(
        self, 
        teacher_network: tf.keras.Model | None
    ) -> None:
        """Attach or clear an independent runtime teacher and retrace model steps.

        A supplied wrapper is unwrapped to its raw network while retaining schedule,
        preprocessing, timestep-zero, and epsilon/image-target metadata. An external raw teacher
        without this metadata requires the caller to ensure a matching forward
        process. It remains frozen outside explicit fit_teacher calls. The teacher
        is synchronized to the active resolution when supported and excluded from
        the student's tracked tree.

        Args:
            teacher_network (tf.keras.Model | None): Independent raw teacher or wrapper.
                None clears it; clearing a positive noise objective requires
                defer_teacher=True. Live raw/EMA student aliases are rejected.

        Returns:
            None: Updates teacher_network, noise-loss availability, and mapped-target
            preparation; clears cached train/test/predict functions. Active noise KD
            forces map_preprocess=True; otherwise the original mapping choice returns.

        Raises:
            ValueError: The teacher aliases the student, predicts images for epsilon KD,
                is missing when required, or disagrees on known schedule, timestep-zero,
                preprocessing, timestep-count, channel, or CFG metadata.
            TypeError: A non-None teacher is not callable.
        """

        # Reject a teacher whose swapped target is incompatible with noise teaching.
        if teacher_network is not None and self.noise_distil_loss_coef > 0. \
        and self.previous_teacher_noise_loss_weight > 0. and getattr(
            teacher_network, 
            "swap_noise_image", 
            getattr(teacher_network, "_diffusion_swap_noise_image", False)
        ):
            raise ValueError(
                "An x0-prediction wrapper cannot teach epsilon distillation."
            )

        teacher_schedule = None
        teacher_modify_first = None
        # Read forward-process metadata only from a supplied teacher.
        if teacher_network is not None:
            teacher_schedule = getattr(
                teacher_network, 
                "scheduler_name", 
                getattr(teacher_network, "_diffusion_scheduler_name", None)
            )
            teacher_modify_first = getattr(
                teacher_network, 
                "modify_first_t", 
                getattr(teacher_network, "_diffusion_modify_first_t", None)
            )

        # Unwrap teacher wrappers while preserving their 
        # forward-process metadata on the raw network.
        if isinstance(teacher_network, DiffusionModel):
            raw_teacher = teacher_network.network
            raw_teacher._diffusion_scheduler_name = teacher_schedule
            raw_teacher._diffusion_modify_first_t = teacher_modify_first
            raw_teacher._diffusion_preprocess_type = teacher_network.preprocess_type
            raw_teacher._diffusion_swap_noise_image = getattr(
                teacher_network, 
                "swap_noise_image", 
                getattr(raw_teacher, "_diffusion_swap_noise_image", False)
            )
            # Retain the teacher's own label ordering when it resumes training.
            object.__setattr__(
                raw_teacher, 
                "_diffusion_seen_classes", 
                dict(getattr(
                    teacher_network, 
                    "seen_classes", 
                    {}
                ))
            )
            teacher_network = raw_teacher

        # A teacher must be independent of both live student branches.
        if teacher_network is not None and (
            teacher_network is self.network
            or teacher_network is self.ema_network
            or any(teacher_network is self.get_teacher_network(name)
                   for name in self.get_teacher_names()[1:])
        ):
            raise ValueError(
                "teacher_network must be an independent frozen snapshot."
            )

        # Known native input coordinates must agree before either teacher objective runs.
        if teacher_network is not None and hasattr(teacher_network, "_diffusion_preprocess_type"):
            # Both networks must use the same pixel-to-model transformation.
            if teacher_network._diffusion_preprocess_type != self.preprocess_type:
                raise ValueError("teacher_network preprocess_type must match the student.")

        needs_noise_teacher = bool(
            self.noise_distil_loss_coef > 0. and 
            self.previous_teacher_noise_loss_weight > 0.
        )
        native_noise_api = self._teacher_uses_native_noise_api(teacher_network)
        # An attached teacher must support inference through its model call.
        if teacher_network is not None and not callable(teacher_network):
            raise TypeError("teacher_network must be a callable model.")

        self._validate_trainable_teacher(teacher_network, "previous")

        # Missing teachers are allowed only through the explicit deferred lifecycle.
        if teacher_network is None and self._current_teacher_spec("noise") is None \
        and self.noise_distil_loss_coef > 0. \
        and (self.previous_teacher_noise_loss_weight > 0. or 
            self.current_teacher_noise_loss_weight > 0.) \
        and not self.defer_teacher:
            raise ValueError(
                "Noise distillation requires teacher_network; "
                "set defer_teacher=True only when it will be attached later."
            )
        # Validate process alignment before enabling a noise-teaching objective.
        if teacher_network is not None and needs_noise_teacher:
            # Teacher and student must interpret timesteps with the same schedule.
            if teacher_schedule is not None \
            and teacher_schedule != self.scheduler_name:
                raise ValueError(
                    "teacher_network scheduler_name must match the student."
                )
            # Teacher and student must agree on the noiseless first-timestep convention.
            if teacher_modify_first is not None \
            and teacher_modify_first != self.modify_first_t:
                raise ValueError(
                    "teacher_network modify_first_t must match the student."
                )
            for name in ("timesteps", "channels", "use_cfg"):
                teacher_value = getattr(teacher_network, name, None)
                # Native metadata is mandatory; known callable metadata must also agree.
                if (native_noise_api or teacher_value is not None) and (
                    teacher_value != getattr(self.network, name, None)
                ):
                    raise ValueError(
                        f"teacher_network {name} must match the student."
                    )

        self._remember_teacher_fit_state(teacher_network, "previous")

        object.__setattr__(self, "teacher_network", teacher_network)
        self.previous_teacher_noise_distil_loss_tracker.reset_state()
        self.current_teacher_noise_distil_loss_tracker.reset_state()
        # Only fit_teacher temporarily enables teacher gradients.
        if self.teacher_network is not None:
            self.teacher_network.trainable = False
            # Keep resolution-aware teachers synchronized with the current student image size.
            if hasattr(self.teacher_network, "set_current_resolution"):
                self.teacher_network.set_current_resolution(
                    self._current_resolution
                )

        DiffusionModel._refresh_loss_flags(self)

        # Precompute active noise-teacher targets in mapped datasets;
        # otherwise restore the configured route.
        self.map_preprocess = True if self.use_noise_distil_loss \
                            else self._map_preprocess_without_teacher
        self.train_function = None
        self.test_function = None
        self.predict_function = None

    def snapshot_teacher_network(
        self, 
        network_name: NetworkName | Literal["teacher"] = "raw"
    ) -> tf.keras.Model:
        """Clone a prediction copy into an independent frozen raw teacher.

        The clone is built, assigned the active resolution, and populated by matching
        active layer names so dynamic class/depth replacement cannot scramble weights.
        Forward-process metadata is attached for subsequent compatibility checks.
        The snapshot is returned without installing it on the wrapper.

        Args:
            network_name (NetworkName | Literal["teacher"]): raw or ema student
                branch, or teacher for the currently attached independent teacher. An ema request falls
                back to raw when use_ema=False, as in get_network.
                Defaults to ``'raw'``.

        Returns:
            tf.keras.Model: Independent raw-network clone with trainable=False and the
            selected weights, current topology, resolution, preprocessing, and schedule metadata.

        Raises:
            ValueError: The branch is unknown or cloned layers cannot match every weight.
            Exception: Delegated serialization, construction, or shape failures propagate.
        """

        source_network = self.get_network(network_name)
        teacher_network = source_network.__class__.from_config(
            source_network.get_config()
        )
        teacher_network.build()

        # Apply the active resolution to snapshots that support resolution changes.
        if hasattr(teacher_network, "set_current_resolution"):
            teacher_network.set_current_resolution(
                self._current_resolution
            )

        copy_network_weights_by_layer(source_network, teacher_network)

        teacher_network._diffusion_scheduler_name = getattr(source_network, "_diffusion_scheduler_name", self.scheduler_name)
        teacher_network._diffusion_modify_first_t = getattr(source_network, "_diffusion_modify_first_t", self.modify_first_t)
        teacher_network._diffusion_swap_noise_image = getattr(source_network, "_diffusion_swap_noise_image", self.swap_noise_image)
        teacher_network._diffusion_preprocess_type = getattr(source_network, "_diffusion_preprocess_type", self.preprocess_type)
        teacher_network.dynamic_num_classes = source_network.dynamic_num_classes
        object.__setattr__(
            teacher_network, 
            "_diffusion_seen_classes", 
            dict(getattr(source_network, "_diffusion_seen_classes", self.seen_classes))
        )
        teacher_network.trainable = False

        return teacher_network

    def load_schedules(
        self, 
        scheduler_name: SchedulerName | None = None, 
        timesteps: int | None = None
    ) -> None:
        """Generate and store TensorFlow tensors for a noise schedule.

        The method updates the schedule family and length but does not resize the raw
        network's time embedding or reset active bounds. Callers changing timesteps must
        keep those separate contracts compatible. modify_first_t recomputes all dependent
        arrays after setting alpha_bar[0]=1, including a zero first beta.

        Each schedule entry is a rank-one tensor [timesteps] in policy variable dtype.
        Assigns self.schedules, scheduler_name and timesteps and records the chosen
        scheduler in initialization metadata; it does not resample noise, resize native
        timestep embeddings or reset random streams.

        Args:
            scheduler_name (SchedulerName | None): Supported name listed in the
                constructor docs; ``None`` reuses ``self.scheduler_name``.
                Defaults to ``None``.
            timesteps (int | None): Schedule length; None reuses the current
                length.  Keep it compatible with network.timesteps and active bounds; this
                method does not validate or resize the network embedding table.
                Defaults to ``None``.

        Returns:
            None: ``self.schedules`` maps schedule-statistic names to rank-1
            tensors in the policy variable dtype and updates schedule metadata.

        Raises:
            ValueError: make_schedule rejects an unsupported scheduler or invalid coupled schedule
                mathematics. TensorFlow schedule conversion errors propagate.
        """

        scheduler_name = self.scheduler_name if scheduler_name is None \
                        else scheduler_name
        timesteps = self.timesteps if timesteps is None else int(timesteps)

        generated_schedules = make_schedule(
            kind=scheduler_name, 
            num_steps=timesteps
        )
        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        schedules = {
            key: tf.constant(value, dtype=stable_dtype)
            for key, value in generated_schedules.items()
        }

        # Make timestep zero noiseless and recompute every dependent schedule array.
        if self.modify_first_t:
            alpha_bar = tf.tensor_scatter_nd_update(
                schedules["alpha_bar"], 
                indices=[[0]], 
                updates=[1.]
            )
            previous_alpha_bar = tf.concat((
                tf.ones_like(alpha_bar[:1]), 
                alpha_bar[:-1]
            ), axis=0)
            noise_rates = tf.sqrt(tf.maximum(1. - alpha_bar, 0.))

            schedules["alpha_bar"] = alpha_bar
            schedules["sqrt_alpha_bar"] = tf.sqrt(alpha_bar)
            schedules["sqrt_one_minus_alpha_bar"] = noise_rates
            schedules["sigmas"] = noise_rates
            schedules["betas"] = 1. - alpha_bar / previous_alpha_bar

        self.schedules = schedules
        self.scheduler_name = scheduler_name
        self.timesteps = timesteps
        self._init_config["scheduler_name"] = scheduler_name

    def get_noise_and_signal_rates(
        self, 
        t: int | tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Gather signal and noise amplitudes at one or more timesteps.

        Index tensors use int32 or int64 as accepted by tf.gather. This lookup reads
        existing schedule tensors only and does not advance a random stream.

        Args:
            t (int | tf.Tensor): Scalar or integer tensor of schedule indices.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: ``(sqrt_alpha_bar,
            sqrt_one_minus_alpha_bar)`` with the same index shape as ``t`` and
            the policy variable dtype.

        Raises:
            tf.errors.InvalidArgumentError: An index is outside the available schedule. TensorFlow
                rejects noninteger index dtypes.
        """

        a = tf.gather(self.schedules["sqrt_alpha_bar"], t)
        b = tf.gather(self.schedules["sqrt_one_minus_alpha_bar"], t)

        return a, b

    def q_sample(
        self, 
        x0: tf.Tensor, 
        t: tf.Tensor, 
        noises: tf.Tensor
    ) -> tf.Tensor:
        """Sample the variance-preserving forward process at supplied times.

        Inputs are already in model coordinates: no preprocessing, clipping or random
        draw occurs here. Noise values may use a different real-floating dtype; both
        image/noise arithmetic and schedule rates use policy variable dtype internally.
        One int32/int64 timestep per image is expected; a singleton time can broadcast.

        Args:
            x0 (tf.Tensor): Clean float images ``[B,H,W,C]``.
            t (tf.Tensor): Integer timestep IDs ``[B]``.
            noises (tf.Tensor): Standard-normal samples matching ``x0``.

        Returns:
            tf.Tensor: Noisy images ``x_t = sqrt(alpha_bar_t)*x0 +
            sqrt(1-alpha_bar_t)*noise`` with the same shape as ``x0``. A
            floating input preserves its dtype; other inputs use the model's
            compute dtype.

        Raises:
            tf.errors.InvalidArgumentError: Schedule indices are out of range or runtime image/noise
                shapes cannot broadcast. TensorFlow conversion errors propagate for nonnumeric
                inputs.
        """

        x0 = tf.convert_to_tensor(x0)
        output_dtype = x0.dtype if x0.dtype.is_floating else tf.as_dtype(
            self.compute_dtype
        )
        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)

        a, b = self.get_noise_and_signal_rates(t)
        a = tf.reshape(a, (-1, 1, 1, 1))
        b = tf.reshape(b, (-1, 1, 1, 1))

        stable_x0 = tf.cast(x0, stable_dtype)
        stable_noises = tf.cast(noises, stable_dtype)

        noisy = tf.cast(
            a * stable_x0 + b * stable_noises, 
            output_dtype
        )

        return noisy

    def noisify(
        self, 
        x0: tf.Tensor, 
        t: tf.Tensor | None = None, 
        min_timesteps: int | None = None, 
        max_timesteps: int | None = None, 
        seed: int | None = None
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Draw timesteps/noise and create a noisy image batch.

        Explicit t bypasses random timestep selection and bound validation, and is
        integer-cast before schedule lookup. With omitted t and active/overridden bounds
        [0, 0), returns x0, zero noise, and zero int32 times without random draws. An
        explicit timestep zero instead follows the actual schedule at zero.
        Timesteps and Gaussian noise have independent checkpointed counters.
        A seed override selects their seed without rewinding those counters;
        reset_seed starts a fresh reproducible sequence, including under XLA.

        Args:
            x0 (tf.Tensor): Clean float images ``[B,H,W,C]``.
            t (tf.Tensor | None): Explicit integer IDs ``[B]``.  ``None`` draws
                uniformly from ``[min_timesteps, max_timesteps)``.
                Defaults to ``None``.
            min_timesteps (int | None): Draw lower bound; ``None`` uses the
                active wrapper bound.
                Defaults to ``None``.
            max_timesteps (int | None): Exclusive draw upper bound; ``None``
                uses the active wrapper bound.
                Defaults to ``None``.
            seed (int | None): Random seed; ``None`` uses ``self.seed``.
                Defaults to ``None``.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Noisy images ``x_t``, sampled
            standard-normal noise, and int32 timestep IDs. Image tensors match
            ``x0`` after nonfloating inputs are converted to the compute dtype.

        Raises:
            ValueError: Random-draw bounds are neither clean-only [0, 0) nor a valid
                nonempty interval inside the schedule, or seed normalization fails.
            tf.errors.InvalidArgumentError: Explicit timestep indices cannot index the
                schedule.
        """

        x0 = tf.convert_to_tensor(x0)
        # Convert integer images before combining them with floating noise.
        if not x0.dtype.is_floating:
            x0 = tf.cast(x0, self.compute_dtype)

        min_timesteps = int(
            self._active_min_timestep
            if min_timesteps is None else min_timesteps
        )
        max_timesteps = int(
            self._active_max_timestep
            if max_timesteps is None else max_timesteps
        )
        seed = effective_seed(
            None, 
            task="noisify seed", 
            seed=self.seed if seed is None else seed
        )

        x_shape = tf.shape(x0)

        # Draw timesteps only when the caller did not supply explicit IDs.
        if t is None:
            # Clean-only bounds return unchanged images, zero corruption, and zero timesteps.
            if min_timesteps == 0 and max_timesteps == 0:
                return (
                    x0, 
                    tf.zeros_like(x0), 
                    tf.zeros(tuple([x_shape[0]]), dtype=tf.int32)
                )

            # Require a nonempty timestep interval inside the schedule horizon.
            if not 0 <= min_timesteps < max_timesteps <= self.timesteps:
                raise ValueError(
                    "Expected 0 <= min_timesteps < max_timesteps <= "
                    f"timesteps, got [{min_timesteps}, {max_timesteps}) "
                    f"with T={self.timesteps}."
                )

            t = tf.random.stateless_uniform(
                tuple([x_shape[0]]), 
                minval=min_timesteps, 
                maxval=max_timesteps, 
                seed=self._random_streams["timesteps"].next_seed(
                    derive_seed(seed, "diffusion", "timesteps")
                ), 
                dtype=tf.int32
            )
        # Normalize explicit timestep IDs instead of drawing random ones.
        else:
            t = tf.cast(tf.convert_to_tensor(t), tf.int32)

        noises = tf.random.stateless_normal(
            x_shape, 
            mean=0., 
            stddev=1., 
            seed=self._random_streams["noise"].next_seed(
                derive_seed(seed, "diffusion", "noise")
            ), 
            dtype=x0.dtype, 
            name="noises"
        )
        x_t = self.q_sample(x0, t, noises)

        return x_t, noises, t

    def preprocess(
        self, 
        x: tf.Tensor, 
        preprocess_type: Literal["standardize", "min-max"] | None = None
    ) -> tf.Tensor:
        """Convert external pixels into the configured model coordinates.

        All scaling uses public pixel bounds, never batch or dataset statistics.
        This method preserves shape and does not clip. Fit/evaluate preparation
        calls it once; q_sample, noisify, and raw networks consume model coordinates.

        Numeric arrays/tensors of any image shape are accepted by tensor conversion;
        normal input is [B,H,W,C]. Even passthrough casts to the policy variable dtype.
        Per-call overrides accept only the same canonical mode names as the constructor. No statistics or model state are fitted.

        Args:
            x (tf.Tensor): Numeric images in raw [0,255] pixel units for scaling
                modes, or already chosen model coordinates for passthrough.
                Numeric tensor/NumPy-compatible array of arbitrary image shape, normally [B,H,W,C];
                converted to the policy variable dtype.
            preprocess_type (Literal["standardize", "min-max"] | None): None selects self.preprocess_type.
                "standardize" maps raw pixels to [-1,1]; "min-max" maps them to
                [0,1]. A constructor value of None leaves values unchanged.
                The None default resolves to the configured mode; it is not a per-call request to
                override a scaling wrapper into passthrough.
                Defaults to ``None``.

        Returns:
            tf.Tensor: Same-shaped images in the policy's stable variable dtype.

        Raises:
            ValueError: The selected mode is unsupported. Fitted dataset
                normalization belongs to non-diffusion loaders.
        """

        mode = self.preprocess_type if preprocess_type is None else preprocess_type
        images = tf.cast(
            tf.convert_to_tensor(x), 
            self.dtype_policy.variable_dtype
        )

        # Signed scaling uses fixed public pixel bounds for every split and task.
        if mode == "standardize":
            return images / 127.5 - 1.

        # Unit scaling also avoids fitting or retaining dataset statistics.
        if mode == "min-max":
            return images / 255.

        # Passthrough supports explicitly prepared model-space inputs.
        if mode is None:
            return images

        raise ValueError(f"Unknown diffusion preprocess_type: {mode!r}.")

    def prepare_images(self, x: tf.Tensor) -> tf.Tensor:
        """Prepare clean external images for the active diffusion resolution.

        Share deterministic image preparation between training and ensemble
        evaluation. Pixel conversion happens once through preprocess; rank-three
        batches gain a channel axis. When the active resolution differs from
        image_size, resize using the wrapper's configured interpolation settings.
        No labels, timesteps or noise are generated and no random streams advance.

        Args:
            x (tf.Tensor): Numeric external image batch shaped [B,H,W,C], or
                [B,H,W] for grayscale images. Pixel units follow preprocess_type.

        Returns:
            tf.Tensor: Clean model-coordinate images shaped [B,H,W,C], resized
            to the active square resolution when it differs from image_size.
            Preprocessing uses the policy variable dtype; resizing follows
            tf.image.resize's output dtype.

        Raises:
            ValueError: The preprocessing mode or image shape is unsupported.
            tf.errors.InvalidArgumentError: TensorFlow rejects the image resize.
        """

        images = self.preprocess(x)
        images = images[..., None] if images.shape.rank == 3 else images

        # Follow the same progressive-resolution policy for every input path.
        if self._current_resolution != self.image_size:
            images = tf.image.resize(
                images, 
                size=(self._current_resolution, self._current_resolution), 
                method=self.resize_method, 
                antialias=self.resize_antialias
            )

        return images

    def postprocess(
        self, 
        x: tf.Tensor, 
        preprocess_type: Literal["standardize", "min-max"] | None = None, 
        clip: bool = False
    ) -> tf.Tensor:
        """Invert preprocess to external pixel units using the same process type.

        Input shape is arbitrary and preserved, normally [B,H,W,C]. Numeric input is
        converted to a tensor and cast to self.dtype_policy.variable_dtype, including
        passthrough mode. No random, fitted-statistics or model state changes occur.

        Args:
            x (tf.Tensor): Numeric images in the selected model coordinates.
                Numeric tensor/NumPy-compatible array of arbitrary image shape, normally [B,H,W,C];
                converted to the policy variable dtype.
            preprocess_type (Literal["standardize", "min-max"] | None): None selects self.preprocess_type.
                The supported modes match preprocess exactly.
                The None default resolves to the configured mode; it is not a per-call request to
                override a scaling wrapper into passthrough.
                Defaults to ``None``.
            clip (bool): Clip scaled outputs to [0,255] for generated-image export.
                False preserves exact affine inversion and out-of-range values.
                Passthrough modes have no declared pixel bounds and never clip.
                Defaults to ``False``.

        Returns:
            tf.Tensor: Same-shaped stable floating pixels, or unchanged values
            for passthrough. Sampling requests clipping; teacher inference does not.

        Raises:
            ValueError: The selected process type is unsupported.
        """

        mode = self.preprocess_type if preprocess_type is None else preprocess_type
        images = tf.cast(
            tf.convert_to_tensor(x), 
            self.dtype_policy.variable_dtype
        )
        
        # Restore byte-scale units from signed diffusion coordinates.
        if mode == "standardize":
            images = (images + 1.) * 127.5
        # Restore byte-scale units from unit model coordinates.
        elif mode == "min-max":
            images = images * 255.
        # Passthrough is an exact value-preserving operation without clipping.
        elif mode is None:
            return images
        # A misspelled mode must not silently change the image representation.
        else:
            raise ValueError(f"Unknown diffusion preprocess_type: {mode!r}.")
        
        return tf.clip_by_value(images, 0., 255.) if clip else images

    def get_network(
        self, 
        network_name: NetworkName | Literal["teacher"]
    ) -> tf.keras.Model:
        """Resolve a raw, EMA, or runtime teacher prediction copy without cloning it.

        Args:
            network_name (NetworkName | Literal["teacher"]): raw selects the student;
                ema selects the moving average, falling back to raw if use_ema=False;
                teacher selects an independently attached runtime teacher.

        Returns:
            tf.keras.Model: Existing selected network object, including supported
            transformer, convolutional, composed, or callable teachers.

        Raises:
            ValueError: The selector is unknown or teacher is requested before attachment.
        """

        # Resolve the runtime frozen teacher separately from raw and EMA prediction copies.
        if network_name == "teacher":
            # Reject teacher selection when no independent teacher is attached.
            if self.teacher_network is None:
                raise ValueError("No teacher_network is attached.")

            return self.teacher_network

        # Fall back to raw weights when no EMA network exists.
        if not self.use_ema and network_name == "ema":
            network_name = "raw"

        # Select the EMA predictor when requested.
        if network_name == "ema":
            network = self.ema_network
        # Select the trainable raw predictor when requested.
        elif network_name == "raw":
            network = self.network
        # Reject unknown network selectors.
        else:
            raise ValueError(
                f"network_name needs to be one of {NetworkName}, "
                f"but not: {network_name}"
            )

        return network

    def update_ema(
        self, 
        variables: Sequence[tf.Variable] | None = None
    ) -> bool:
        """Update all or a selected aligned subset of EMA weights.

        ``variables=None`` preserves the ordinary single-optimizer behavior.
        Split optimizers pass their active raw trainable variables so batches
        from one phase cannot decay untouched weights owned by the other.

        Weights are aligned by current raw/EMA list position. Each selected floating EMA value
        receives ema_decay * EMA + (1 - ema_decay) * raw; nonfloating state is copied
        exactly from the raw variable. Returning True indicates an
        enabled successful pass even when an explicit empty selection updates no weight.

        Selected entries are existing Keras/TensorFlow variable objects with individual
        shapes and variable dtypes. Compatible floating EMA variables are updated in
        place from corresponding raw values; this operation does not advance optimizer
        iterations or random streams. Explicit variable selection does not cast or
        replace the source objects.

        Args:
            variables (Sequence[tf.Variable] | None): Raw trainables whose EMA counterparts
                should decay. None selects every
                weight; an empty sequence selects none. Nontrainable state sharing a
                selected trainable layer scope also decays.
                Defaults to ``None``.

        Returns:
            bool: False when EMA is disabled; true after a successful update.

        Raises:
            AssertionError: If raw and EMA topologies have different weight
                counts.
        """

        # Report that no EMA update occurred when EMA is disabled.
        if not self.use_ema:
            return False

        require(
            len(self.network.weights) == len(self.ema_network.weights), 
            "Raw and EMA networks must have the same topology."
        )

        # An omitted selection updates every EMA weight; an explicit selection uses raw identities.
        selected_ids = None if variables is None else {
            id(variable) for variable in variables
        }
        # Track selected layer scopes so associated nontrainable state follows their trainables.
        selected_scopes = set() if variables is None else {
            variable_path(variable).rsplit("/", 1)[0] 
            for variable in variables
        }

        for w, ew in zip(self.network.weights, self.ema_network.weights):
            selected = selected_ids is None or id(w) in selected_ids or (
                not w.trainable and 
                variable_path(w).rsplit("/", 1)[0] in selected_scopes
            )
            # Decay only selected trainables and associated mutable layer state.
            if selected:
                ew.assign(
                    ew * self.ema_decay + w * (1 - self.ema_decay)
                    if tf.as_dtype(w.dtype).is_floating else w
                )

        return True

    def apply_grads(
        self, 
        tape: tf.GradientTape, 
        loss: tf.Tensor, 
        variables: list[tf.Variable]| None = None
    ) -> None:
        """Differentiate a scalar loss and apply gradients with the optimizer.

        The loss is unscaled; apply_policy_gradients handles a LossScaleOptimizer
        exactly once. An explicit empty variable group performs no work. Variables
        retain their own Keras variable dtypes/shapes and the scalar objective uses
        the wrapper's policy variable dtype.

        Args:
            tape (tf.GradientTape): Tape that recorded ``loss`` computation.
            loss (tf.Tensor): Scalar differentiable objective.
                Rank-zero floating tensor; wrapper losses use self.dtype_policy.variable_dtype.
            variables (list[tf.Variable] | None): Variables to update; ``None``
                selects all raw-network trainable variables.
                Defaults to ``None``.
                Each entry is a live Keras/TensorFlow variable with its own shape and variable dtype; no
                image-array shape applies.

        Returns:
            None: Optimizer slots, iterations, and variables are updated.

        Raises:
            ValueError: The selected nonempty variable group is wholly disconnected from loss. Empty
                selection is a no-op; partial disconnections are delegated to Keras.
                GradientTape/optimizer errors propagate, and dynamic loss scaling may skip a
                nonfinite update without advancing optimizer iterations.
        """

        # Update all raw trainable variables unless a subset was supplied.
        if variables is None:
            variables = self.network.trainable_variables

        apply_policy_gradients(
            tape, 
            self.optimizer, 
            loss, 
            variables
        )

    def get_cfg_labels(
        self, 
        labels: tf.Tensor, 
        seed: int | None = None
    ) -> tf.Tensor:
        """Apply classifier-free label dropout using its saved random stream.

        Repeated calls advance the stream, including with an explicit seed;
        reset_seed starts a fresh sequence.

        Labels are integer network condition IDs [B]; output has the identical dtype
        and shape. The float32 stateless uniform draw uses the named CFG stream, which
        advances even with an explicit seed. This helper applies the configured dropout
        probability directly; it does not shift dataset IDs or map dynamic classes.

        Args:
            labels (tf.Tensor): Shifted integer labels ``[B]`` where ID 0 is
                reserved for the null condition.
            seed (int | None): Random seed; ``None`` uses ``self.seed``.
                Defaults to ``None``.

        Returns:
            tf.Tensor: Same shape/dtype as ``labels``; each element becomes 0
            independently with probability ``p_uncond``.

        Raises:
            ValueError: Seed derivation rejects the configured or supplied seed. TensorFlow
                shape/type errors from label masking propagate.
        """

        seed = self.seed if seed is None else seed

        mask = tf.random.stateless_uniform(
            tuple([tf.shape(labels)[0]]), 
            seed=self._random_streams["cfg"].next_seed(
                derive_seed(seed, "diffusion", "cfg")
            )
        ) < self.p_uncond
        masked_labels = tf.where(
            mask, 
            tf.zeros_like(labels), 
            labels
        )

        return masked_labels

    def prep_inputs(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor], 
        use_label_dropout: bool = True, 
        seed: int | None = None
    ) -> tuple[
        tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, 
        tf.Tensor, tf.Tensor, tf.Tensor
    ]:
        """Prepare one dataset batch for diffusion loss computation.

        The input images are numeric raw external pixels [B,H,W,C], or [B,H,W] with an
        inserted channel, unless the wrapper is configured for passthrough. preprocess
        casts to policy variable dtype; resizing follows tf.image.resize's output dtype.
        Noise and noisy images retain the resulting floating image dtype. Timesteps are
        int32 [B]; class/condition/null IDs retain the sparse input label integer dtype.
        Random noising and optional CFG dropout advance their independent saved streams.
        No model weights or metric state are updated.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Clean images ``[B,H,W,C]`` in
                external pixel units and sparse integer dataset labels ``[B]``; dynamic models
                map labels through seen_classes before applying any CFG offset.
            use_label_dropout (bool): Apply CFG dropout to shifted labels.
                Defaults to ``True``.
            seed (int | None): Seed forwarded to noising and CFG dropout; None lets both
                operations
                use the wrapper seed.
                Defaults to ``None``.

        Returns:
            tuple: ``(x0, noises, t, x_t, cfg_labels, uncond_labels, classes)``.
            Images are preprocessed once and resized to the active resolution when necessary;
            ``cfg_labels`` are real labels shifted by one under CFG and possibly
            replaced by 0; ``uncond_labels`` are all 0.  With
            ``swap_noise_image=True``, ``noises`` is the noisy image ``x_t``.

        Raises:
            ValueError: The configured preprocessing mode, dynamic class vocabulary or noising
                interval is invalid.
            tf.errors.InvalidArgumentError: A dynamic input label is unseen, or TensorFlow
                resizing/noising detects incompatible inputs. Unpacking requires an image/label
                pair.
        """

        x0, labels = inputs

        x0 = self.prepare_images(x0)

        classes = self._map_classes(labels)
        labels = classes + int(self.use_cfg)
        x_t, noises, t = self.noisify(x0, seed=seed)
        cfg_labels = self.get_cfg_labels(
            labels, 
            seed=seed
        ) if use_label_dropout else labels
        uncond_labels = tf.zeros_like(labels)

        noises = x_t if self.swap_noise_image else noises

        return x0, noises, t, x_t, cfg_labels, uncond_labels, classes

    def prep_inputs_map(
        self, 
        x0: tf.Tensor, 
        labels: tf.Tensor
    ) -> tuple[tf.Tensor, ...]:
        """Prepare one dataset element.

        This small adapter matches the positional signature expected by
        ``tf.data.Dataset.map``. Classifier wrappers may override it to
        append additional precomputed targets.

        Returned image/noise tensors are floating [B,H,W,C], times int32 [B], and
        class/condition IDs use the original integer label dtype [B], as in prep_inputs.
        Teacher epsilon tensors [B,H,W,C] retain prediction dtype and masks are bool [B].
        Independent roles append tuples of targets/masks; the legacy single previous
        teacher appends tensors directly. Preparation advances noising/dropout streams
        and teacher calls may build then freeze previously lazy inference layers.

        Args:
            x0 (tf.Tensor): Clean image batch ``[B,H,W,C]``.
                Numeric raw external images [B,H,W,C] or [B,H,W]; preprocessing converts them to model
                coordinates and policy variable dtype before optional resizing.
            labels (tf.Tensor): Integer dataset labels ``[B]``.
                Sparse integer dataset IDs [B]; integer dtype is preserved by dynamic mapping before
                network-specific casting.

        Returns:
            tuple[tf.Tensor, ...]: The seven tensors returned by
            :meth:`prep_inputs`, followed by the noise-teacher prediction and
            mask when noise distillation is active.

        Raises:
            ValueError: Delegated raw preparation or teacher selection rejects its input contract.
                Callable/native teacher output checks and TensorFlow condition-remapping errors
                propagate during dataset iteration.
        """

        outputs = self.prep_inputs(
            (x0, labels), 
            use_label_dropout=self._preprocess_training is not False
        )

        # Teacher-free input preparation ends after the seven student tensors.
        if not self.use_noise_distil_loss:
            return outputs

        x_t, t = outputs[3], outputs[2]
        cond_labels, uncond_labels, classes = outputs[4:7]
        # Default mapped preparation uses training guidance, matching label dropout;
        # evaluation selects its guidance only through an explicit False mode.
        cfg_scale = self.train_cfg_scale if self._preprocess_training is not False \
                    else self.test_cfg_scale
        predictions, masks = [], []
        for specification in self._noise_teacher_specs():
            teacher = specification["network"]
            teacher_labels = self._remap_teacher_conditions(
                cond_labels, 
                specification["class_ids"]
            )
            teacher_labels = self._mask_unknown_teacher_labels(teacher_labels, teacher)
            teacher_mask = tf.ones_like(cond_labels, dtype=tf.bool)
            teacher_num_labels = getattr(teacher, "num_labels", None)
            # Preserve the previous teacher's legacy unsupported-condition exclusion.
            if specification["role"] == "previous" and teacher_num_labels is not None:
                teacher_mask = cond_labels < tf.cast(teacher_num_labels, cond_labels.dtype)

            # Without CFG, local zero is a real class rather than an unconditional fallback;
            # exclude unmapped current-teacher conditions even in all-row scope.
            if specification["role"] == "current" and not self.use_cfg \
            and self.dual_teacher_scope == "all" and specification["class_ids"] is not None:
                teacher_mask = tf.reduce_any(tf.equal(
                    classes[..., None], 
                    tf.constant(specification["class_ids"], dtype=classes.dtype)
                ), axis=-1)

            # Task scopes use original class IDs even when CFG dropped the condition to zero.
            if self._current_teacher_spec("noise") is not None and self.dual_teacher_scope == "task" \
            and specification["task_class_ids"] is not None:
                membership = tf.reduce_any(tf.equal(
                    classes[..., None], 
                    tf.constant(specification["task_class_ids"], dtype=classes.dtype)
                ), axis=-1)
                teacher_mask = tf.logical_and(teacher_mask, membership)
            
            predictions.append(self._predict_teacher_noise(
                x_t, t, 
                teacher_labels, 
                uncond_labels, 
                scale=cfg_scale, 
                teacher_network=teacher
            ))
            masks.append(teacher_mask)

        # Preserve the established tensor-valued tail for the single previous-teacher API.
        if self._current_teacher_spec("noise") is None:
            return *outputs, predictions[0], masks[0]
        return *outputs, tuple(predictions), tuple(masks)

    def compute_separate_noise_losses(
        self, 
        noises: tf.Tensor, 
        noises_pred: tf.Tensor, 
        cond_labels: tf.Tensor | None
    ) -> tuple[tf.Tensor | None, tf.Tensor | None]:
        """Compute reporting-only noise losses for conditional/null rows.

        Image/noise tensors are real-floating [B,H,W,C]; cond_labels is integer [B].
        Targets/predictions are cast to policy variable dtype and predictions detached
        for reporting. Returned compiled losses use that dtype (normally rank zero).
        The helper does not update the split metric trackers itself.

        Args:
            noises (tf.Tensor): Noise targets shaped like the model output.
            noises_pred (tf.Tensor): Predicted noise shaped like ``noises``.
            cond_labels (tf.Tensor | None): Post-dropout condition IDs. Null ID
                zero marks unconditional rows when CFG is enabled.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Conditional and null scalar losses computed
            every time this helper is called. Empty selections contribute finite zero;
            their trackers receive zero sample weight when condition counts are supplied
            to get_results_dict. Disabling split reporting is handled by the caller.

        Raises:
            AssertionError: cond_labels is None; splitting requires row-condition metadata.
        """

        require(
            cond_labels is not None, 
            "cond_labels are required to show separate noise losses."
        )

        # Without CFG, zero is a real class and every row is conditional.
        if self.use_cfg:
            cond_mask = cond_labels != 0
        # Keep every class-conditioned row when no null ID is reserved.
        else:
            cond_mask = tf.ones_like(cond_labels, dtype=tf.bool)

        uncond_mask = tf.logical_not(cond_mask)
        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        noises = tf.cast(noises, stable_dtype)
        noises_pred = tf.stop_gradient(tf.cast(noises_pred, stable_dtype))

        cond_has_rows = tf.reduce_any(cond_mask)
        cond_noise_loss = self._compute_base_loss(
            tf.boolean_mask(noises, cond_mask), 
            tf.boolean_mask(noises_pred, cond_mask)
        )
        cond_noise_loss = tf.where(
            cond_has_rows, 
            cond_noise_loss, 
            tf.zeros_like(cond_noise_loss)
        )

        uncond_has_rows = tf.reduce_any(uncond_mask)
        uncond_noise_loss = self._compute_base_loss(
            tf.boolean_mask(noises, uncond_mask), 
            tf.boolean_mask(noises_pred, uncond_mask)
        )
        uncond_noise_loss = tf.where(
            uncond_has_rows, 
            uncond_noise_loss, 
            tf.zeros_like(uncond_noise_loss)
        )

        return cond_noise_loss, uncond_noise_loss

    def compute_distil_noise_loss(
        self, 
        teacher_noises_pred: tf.Tensor | tuple[tf.Tensor, ...], 
        noises_pred: tf.Tensor, 
        teacher_noise_mask: tf.Tensor | tuple[tf.Tensor, ...] | None = None, 
        update_teacher_metrics: bool = False
    ) -> tf.Tensor:
        """Sum independently normalized noise KD objectives without mixing targets.

        A tensor target preserves the single previous-teacher API. Tuple targets
        and masks follow _noise_teacher_specs order. Production aggregation may
        update each teacher's population-weighted metric in the same graph branch;
        direct numerical callers leave metrics untouched by default.

        Prediction tensors are real-floating [B,H,W,C]; masks are Boolean/numeric [B].
        Per-role compiled loss arithmetic uses policy variable dtype. Optional tracker
        updates record each unweighted role loss using its eligible row population;
        the result then applies the role coefficient. Model weights are unchanged.

        Args:
            teacher_noises_pred (tf.Tensor | tuple[tf.Tensor, ...]): Floating
                epsilon targets ``[B,H,W,C]``; a tuple follows active previous/current
                role order, while a single tensor uses the previous-teacher weight.
            noises_pred (tf.Tensor): Student epsilon prediction ``[B,H,W,C]``.
            teacher_noise_mask (tf.Tensor | tuple[tf.Tensor, ...] | None): Row
                weights ``[B]``, matching per-role tuple, or None for all rows.
                Defaults to ``None``.
            update_teacher_metrics (bool): True records each unweighted role loss
                with its eligible row population. False leaves trackers unchanged.
                Defaults to ``False``.

        Returns:
            loss (tf.Tensor): Policy-variable-dtype scalar sum of independently
                normalized role losses multiplied by their role coefficients.
                Teacher targets are detached; an empty role tuple contributes zero.

        Raises:
            ValueError: Tuple targets/masks do not match the current teacher roles.
        """

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        # Legacy callers supply one tensor and retain the previous teacher's own weight.
        if not isinstance(teacher_noises_pred, (tuple, list)):
            return self._compute_single_teacher_noise_loss(
                teacher_noises_pred, noises_pred, teacher_noise_mask
            ) * tf.cast(tf.convert_to_tensor(self.previous_teacher_noise_loss_weight, dtype_hint=stable_dtype), stable_dtype)

        specifications = self._noise_teacher_specs()

        # A flat row mask must not be mistaken for a sequence of per-teacher scalar masks.
        if teacher_noise_mask is not None and not isinstance(teacher_noise_mask, (tuple, list)):
            raise ValueError("Tuple noise targets require a tuple/list of per-teacher masks.")
        
        masks = tuple(teacher_noise_mask) if teacher_noise_mask is not None \
                else tuple([None]) * len(teacher_noises_pred)
        
        # A stale mapped dataset must not silently pair targets with a different teacher role.
        if len(teacher_noises_pred) != len(specifications) or len(masks) != len(specifications):
            raise ValueError("Mapped noise targets and masks must match the active teacher roles.")
        
        terms = []
        for target, mask, specification in zip(teacher_noises_pred, masks, specifications):
            term = self._compute_single_teacher_noise_loss(target, noises_pred, mask)
            terms.append(tf.cast(tf.convert_to_tensor(specification["weight"], dtype_hint=stable_dtype), stable_dtype) * term)

            # Updating inside the loss branch remains valid for split-batch tf.cond graphs.
            if update_teacher_metrics:
                tracker = getattr(self, f'{specification["role"]}_teacher_noise_distil_loss_tracker')
                count = tf.reduce_sum(tf.cast(mask, stable_dtype)) if mask is not None \
                        else tf.cast(tf.shape(noises_pred)[0], stable_dtype)
                tracker.update_state(tf.stop_gradient(term), sample_weight=count)
        
        return tf.add_n(terms) if terms else tf.zeros((), dtype=stable_dtype)

    def compute_ctr_loss(
        self, 
        classes: tf.Tensor, 
        classes_pred_list: list[tf.Tensor]
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Average auxiliary class predictions and compute cross-entropy.

        Integer targets are [B]; each non-None prediction is real-floating [B,K],
        K=self.network.num_classes. Prediction averaging and both returned tensors use
        policy variable dtype. Loss is a scalar zero tensor when no prediction exists;
        it is never a Python float in this implementation. No metric or optimizer state
        is updated.

        Args:
            classes (tf.Tensor): Zero-based ground-truth classes ``[B]``.
            classes_pred_list (list[tf.Tensor | None]): Optional softmax tensors
                ``[B,num_classes]`` from regularizer depths.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Scalar sparse categorical loss and mean class probabilities
                [B,num_classes], both in policy variable dtype. With no tensor predictions, both
                loss and probability tensor contain zeros.

        Raises:
            No explicit exceptions are raised here. TensorFlow/Keras reports incompatible
                probability shapes or sparse class IDs outside the head width when cross-entropy is
                evaluated.
        """

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        ctr_num = 0
        ctr_loss = tf.constant(0., dtype=stable_dtype)
        ctr_preds = tf.zeros((
            tf.shape(classes)[0], 
            self.network.num_classes
        ), dtype=stable_dtype)

        for classes_pred in classes_pred_list:
            # Include each available regularizer prediction in the ensemble.
            if classes_pred is not None:
                ctr_num += 1
                ctr_preds += tf.cast(classes_pred, stable_dtype)

        # Average conditioning-token predictions before computing their auxiliary loss.
        if ctr_num > 0:
            ctr_preds /= ctr_num
            ctr_loss = tf.reduce_mean(self.scce_loss_fn(
                classes, 
                ctr_preds
            ))            

        return ctr_loss, ctr_preds

    def compute_noise_distil_image_kl_ctr_loss(
        self, 
        x0: tf.Tensor, 
        noises: tf.Tensor, 
        classes: tf.Tensor, 
        x0_pred: tf.Tensor, 
        noises_pred: tf.Tensor, 
        z_vals_list_c: list[tuple[tf.Tensor, tf.Tensor]], 
        regs_list_c: list[tf.Tensor], 
        teacher_noises_pred: tf.Tensor | None = None, 
        z_vals_list_u: list[tuple[tf.Tensor, tf.Tensor]] | None = None, 
        regs_list_u: list[tf.Tensor] | None = None, 
        kl_train_type: TrainType | None = None, 
        ctr_train_type: TrainType | None = None, 
        use_image_loss: bool | None = None, 
        cond_labels: tf.Tensor | None = None, 
        teacher_noise_mask: tf.Tensor | None = None
    ) -> tuple[
        tf.Tensor, tf.Tensor, tf.Tensor | None, tf.Tensor | None, 
        tf.Tensor | float, tf.Tensor | float, tf.Tensor | float, 
        tf.Tensor, tf.Tensor | float
    ]:
        """Compute and weight diffusion, reconstruction, KL, and token losses.

        Images, noise and predictions are real-floating [B,H,W,C] in model coordinates.
        Classes and condition IDs are integer [B]. Auxiliary probability tensors are
        [B,K]; each latent mean/log-variance pair contains matching real-floating
        [B,...] tensors, with the nonbatch shape determined by its reshaper. All enabled
        loss reductions cast to policy variable dtype. Teacher targets/masks can also
        be role-ordered tuples. Role loss trackers may update through noise KD, but this
        aggregator does not apply gradients or EMA updates.

        Args:
            x0 (tf.Tensor): Clean images ``[B,H,W,C]``.
            noises (tf.Tensor): Noise target matching ``x0``.
            classes (tf.Tensor): Zero-based classes ``[B]``.
            x0_pred (tf.Tensor): Reconstructed clean images matching ``x0``.
            noises_pred (tf.Tensor): Guided noise prediction matching ``noises``.
            z_vals_list_c (list[tuple[tf.Tensor, tf.Tensor]]): Conditional latent
                mean/log-variance pairs.
            regs_list_c (list[tf.Tensor | None]): Conditional auxiliary class
                probabilities by depth.
            teacher_noises_pred (tf.Tensor | None): Frozen teacher prediction
                on the same noisy inputs, timestep, labels, and CFG scale.
                Defaults to ``None``.
            z_vals_list_u (list[tuple[tf.Tensor, tf.Tensor]] | None): Unconditional
                latent pairs, required when KL trains unconditionally.
                Defaults to ``None``.
            regs_list_u (list[tf.Tensor] | None): Unconditional
                regularizers, required when token loss trains unconditionally.
                Defaults to ``None``.
            kl_train_type (TrainType | None): ``"cond"``/``"uncond"`` source;
                None uses the configured value.
                Defaults to ``None``.
            ctr_train_type (TrainType | None): Regularizer source; None uses the
                configured value.
                Defaults to ``None``.
            use_image_loss (bool | None): Compute reconstruction loss; None uses
                ``self.use_image_loss``.
                Defaults to ``None``.
            cond_labels (tf.Tensor | None): Conditional/possibly dropped label
                IDs ``[B]`` used only for optional split-noise reporting.
                Defaults to ``None``.
            teacher_noise_mask (tf.Tensor | None): Rows whose condition exists
                in the teacher vocabulary.
                Defaults to ``None``.

        Returns:
            tuple: Nine values: weighted total loss, raw noise loss, conditional noise
            loss, null noise loss, noise KD loss, image loss, KL loss, token loss, and mean
            token probabilities. Loss values use policy variable dtype; omitted split
            diagnostics are None, disabled auxiliary losses are scalar zero tensors, and
            disabled token predictions are the Python scalar 0. Enabled token predictions
            have shape [B, current_num_classes].

        Raises:
            ValueError: Active independent teacher targets/masks do not match the currently attached
                roles. Errors from compiled noise/image loss, KL reduction and sparse token
                cross-entropy propagate; disabled losses do not evaluate their corresponding inputs.
        """

        kl_train_type = self.kl_train_type if kl_train_type is None else kl_train_type
        ctr_train_type = self.ctr_train_type if ctr_train_type is None else ctr_train_type
        use_image_loss = self.use_image_loss if use_image_loss is None else use_image_loss

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        noises = tf.cast(noises, stable_dtype)
        noises_pred = tf.cast(noises_pred, stable_dtype)
        x0 = tf.cast(x0, stable_dtype)
        x0_pred = tf.cast(x0_pred, stable_dtype)

        noise_loss = self._compute_base_loss(
            noises, 
            noises_pred
        )
        cond_noise_loss, uncond_noise_loss = self.compute_separate_noise_losses(
            noises, 
            noises_pred, 
            cond_labels
        ) if self.show_separate_noise_losses else (None, None)
        noise_distil_loss = self.compute_distil_noise_loss(
            teacher_noises_pred, 
            noises_pred, 
            teacher_noise_mask, 
            update_teacher_metrics=True
        ) if self.use_noise_distil_loss else 0.
        image_loss = self._compute_base_loss(
            x0, 
            x0_pred
        ) if use_image_loss else 0.
        kl_loss = VariationalAutoencoder.compute_kl(
            z_vals_list_c if kl_train_type == "cond" else z_vals_list_u, 
            dtype=self.dtype_policy.variable_dtype
        ) if self.use_kl_loss else 0.
        ctr_loss, ctr_preds = self.compute_ctr_loss(
            classes, 
            regs_list_c if ctr_train_type == "cond" else regs_list_u
        ) if self.use_ctr_loss else (0., 0.)

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        noise_loss = tf.cast(noise_loss, stable_dtype)
        cond_noise_loss = tf.cast(cond_noise_loss, stable_dtype) \
                        if cond_noise_loss is not None else None
        uncond_noise_loss = tf.cast(uncond_noise_loss, stable_dtype) \
                        if uncond_noise_loss is not None else None
        noise_distil_loss = tf.cast(noise_distil_loss, stable_dtype)
        image_loss = tf.cast(image_loss, stable_dtype)
        kl_loss = tf.cast(kl_loss, stable_dtype)
        ctr_loss = tf.cast(ctr_loss, stable_dtype)
        loss = (
            noise_loss * self.noise_loss_coef + 
            noise_distil_loss * self.noise_distil_loss_coef + 
            image_loss * self.image_loss_coef + 
            kl_loss * self.kl_loss_coef + 
            ctr_loss * self.ctr_loss_coef
        )

        outputs = (
            loss, noise_loss, cond_noise_loss, 
            uncond_noise_loss, noise_distil_loss, 
            image_loss, kl_loss, ctr_loss, 
            ctr_preds
        )

        return outputs

    def call_network(
        self, 
        x_t: tf.Tensor, 
        t_batch: tf.Tensor, 
        cond_labels: tf.Tensor, 
        uncond_labels: tf.Tensor | None = None, 
        scale: float | None = None, 
        network_name: NetworkName = "raw", 
        teacher_network: tf.keras.Model | None = None, 
        training: bool = False
    ) -> tuple[
        tuple[tf.Tensor, tf.Tensor | None], 
        tuple[list[tf.Tensor], list[tf.Tensor] | None], 
        tuple[
            list[tuple[tf.Tensor, tf.Tensor]], 
            list[tuple[tf.Tensor, tf.Tensor]] | None
        ]
    ]:
        """Run conditional and, when requested, unconditional network passes.

        Floating image inputs are already in model coordinates; integer time/condition
        tensors are [B]. Native epsilon, auxiliary probabilities [B,K] and latent pairs
        [B,...] retain the selected network's output dtypes (normally its layer compute
        policy); no stable-loss cast happens here. Each latent pair has matching mean
        and log-variance shape. Native training=True may update BatchNormalization or
        stochastic-layer state; ordinary callable teachers always run in inference
        mode, are frozen after a lazy build, and return detached epsilon targets.

        Args:
            x_t (tf.Tensor): Noisy image batch ``[B,H,W,C]``.
            t_batch (tf.Tensor): Integer timesteps ``[B]``.
            cond_labels (tf.Tensor): Shifted/conditional label IDs ``[B]``.
            uncond_labels (tf.Tensor | None): Null IDs ``[B]``; required for a
                guided pass.
                Defaults to ``None``.
            scale (float | None): Non-None requests the unconditional pass when
                CFG is enabled. Combination happens later in ``denoise``.
                Defaults to ``None``.
            network_name (NetworkName | Literal["teacher"]): raw, ema, or runtime teacher
                selector passed through get_network; EMA
                falls back to raw when disabled.
                Defaults to ``'raw'``.
            teacher_network (tf.keras.Model | None): Explicit independent noise
                teacher overriding ``network_name`` selection. None uses the
                selected raw/EMA or attached teacher. Supply by keyword when
                choosing an independent teacher. The teacher-target helper
                ``_predict_teacher_noise`` detaches every returned target.
                Defaults to ``None``.
            training (bool): Final optional argument controlling native raw/EMA/teacher
                network execution. Ordinary callable teachers always run in
                inference mode. Defaults to ``False``.

        Returns:
            tuple: ``((eps_c, eps_u), (regs_c, regs_u),
            (z_vals_list_c, z_vals_list_u))``. Noise
            predictions are ``[B,H,W,C]``; without a second pass eps_u and regs_u are None
            while z_vals_list_u
            is an empty list.

        Raises:
            TypeError: A callable teacher output is not one dense real-floating tensor.
            ValueError: Network selection is invalid/missing or a callable teacher output has
                incompatible static shape.
            tf.errors.InvalidArgumentError: A callable teacher output has a mismatched dynamic shape
                or nonfinite values. Native network call/output-unpacking errors propagate.
        """

        network = self.get_network(
            network_name
        ) if teacher_network is None else teacher_network


        def run_network(
            labels: tf.Tensor
        ) -> tuple[
            tf.Tensor, 
            list[tf.Tensor], 
            list[tuple[tf.Tensor, tf.Tensor]]
        ]:
            """Run one conditional-label branch of the selected network.

            The captured floating images are [B,H,W,C] and times integer [B]. Returned
            noise has image geometry; each regularizer is floating [B,K] and each latent
            pair has two matching floating [B,...] tensors. Native output dtypes are retained.
            Ordinary callable teachers return detached epsilon, empty auxiliary lists, and
            are frozen after their call; native calls honor the captured training flag.

            Args:
                labels (tf.Tensor): Integer condition IDs of shape ``[B]``.

            Returns:
                tuple: Noise prediction, auxiliary class predictions, and an
                ordered list of latent mean/log-variance pairs.

            Raises:
                TypeError: The selected ordinary teacher returns a nondense or nonfloating epsilon
                    output.
                ValueError: Its static image geometry is incompatible.
                tf.errors.InvalidArgumentError: Dynamic epsilon geometry or finite-value checks fail.
                    Native call/unpacking errors propagate.
            """

            # Ordinary callable teachers receive only their configured input structure.
            if (network_name == "teacher" or teacher_network is not None) \
            and not self._teacher_uses_native_noise_api(network):
                # Image-only denoisers do not consume timestep or class conditions.
                if self.teacher_noise_input_type == "images":
                    teacher_inputs = x_t
                # Unconditional timestep-aware denoisers consume a two-tensor tuple.
                elif self.teacher_noise_input_type == "images_timesteps":
                    teacher_inputs = (x_t, t_batch)
                # Class-conditioned denoisers consume all three diffusion inputs.
                else:
                    teacher_inputs = (x_t, t_batch, labels)

                eps = network(teacher_inputs, training=False)
                # Lazy Keras teachers may create child layers during their first
                # call. Freeze the completed tree as well as its initial shell.
                network.trainable = False

                # Epsilon targets must be one dense tensor rather than structured outputs.
                if not tf.is_tensor(eps) or isinstance(eps, (tf.RaggedTensor, tf.SparseTensor)):
                    raise TypeError("A callable noise teacher must return one epsilon tensor.")
                # Noise distillation requires real floating-point regression targets.
                if not tf.as_dtype(eps.dtype).is_floating:
                    raise TypeError("A callable noise teacher must return floating-point epsilon.")
                # Catch known geometry mismatches before the dynamic shape assertion.
                if not x_t.shape.is_compatible_with(eps.shape):
                    raise ValueError("Callable teacher epsilon shape must match the noisy images.")
                
                shape_check = tf.debugging.assert_equal(
                    tf.shape(eps), 
                    tf.shape(x_t), 
                    message="Callable teacher epsilon shape must match the noisy images."
                )
                with tf.control_dependencies([shape_check]):
                    eps = tf.debugging.check_numerics(
                        eps, 
                        "Callable teacher epsilon must contain only finite values."
                    )

                return tf.stop_gradient(eps), [], []

            # Supply the decoder-specific encoder placeholders and unpack its output.
            if isinstance(network, DiTDecoder):
                outputs = network(
                    (x_t, t_batch, labels), 
                    encoder_cond=None, 
                    encoder_features_list=[None] * len(
                        network.encoder_feature_dims
                    ), 
                    full_return=True, 
                    training=training
                )

                return (
                    outputs["noises"], 
                    outputs["regs_list"], 
                    outputs["z_vals_list"]
                )

            outputs = network(
                (x_t, t_batch, labels), 
                full_return=True, 
                training=training
            )

            # Unpack named outputs from composite networks; positional outputs follow below.
            if isinstance(outputs, Mapping):
                return (
                    outputs["noises"], 
                    outputs.get("regs_list", []), 
                    outputs.get("z_vals_list", [])
                )

            eps, *_, regs_list, z_vals_list = outputs

            return eps, regs_list, z_vals_list


        eps_c, regs_list_c, z_vals_list_c = run_network(cond_labels)
        eps_u, regs_list_u, z_vals_list_u = run_network(uncond_labels) if self.use_cfg and scale is not None \
                                            else (None, None, [])

        return ((eps_c, eps_u), 
                (regs_list_c, regs_list_u), 
                (z_vals_list_c, z_vals_list_u))

    def denoise(
        self, 
        x_t: tf.Tensor, 
        t: tf.Tensor, 
        eps_c: tf.Tensor, 
        eps_u: tf.Tensor | None = None, 
        scale: float | None = None, 
        reshape_coefs: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Recover an ``x0`` estimate from ``x_t`` and predicted noise.

        For epsilon prediction, reconstruction is (x_t - noise_rate * eps)/signal_rate
        with arithmetic in policy variable dtype and output cast to x_t.dtype. In
        swap_noise_image mode the selected/guided raw prediction is returned as x0
        directly. Returned eps retains the selected prediction's dtype.

        Args:
            x_t (tf.Tensor): Noisy images ``[B,H,W,C]``.
            t (tf.Tensor): Scalar timestep or per-example IDs ``[B]``.
            eps_c (tf.Tensor): Conditional noise prediction matching ``x_t``.
            eps_u (tf.Tensor | None): Optional unconditional prediction.
                Defaults to ``None``.
            scale (float | None): CFG scale; None disables combination.
                Defaults to ``None``.
            reshape_coefs (bool): Reshape vector schedule rates to
                ``[B,1,1,1]`` for image broadcasting.  Scalar rates do not need
                this.
                Defaults to ``False``.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Reconstructed ``x0`` and the selected/
            guided noise, both matching ``x_t`` shape.

        Raises:
            tf.errors.InvalidArgumentError: Schedule lookup is out of range or the supplied image,
                prediction and rate shapes cannot broadcast. A requested guided pass needs a
                compatible eps_u tensor; this method performs no clipping or finiteness check.
        """

        # Combine conditional and unconditional predictions with CFG.
        if self.use_cfg and scale is not None:
            eps = eps_u + scale * (eps_c - eps_u)
        # Use the conditional prediction directly when CFG is inactive.
        else:
            eps = eps_c

        sqrt_a_t, sqrt_one_minus_a_t = self.get_noise_and_signal_rates(t)
        # Broadcast scalar schedule rates across image dimensions when requested.
        if reshape_coefs:
            sqrt_a_t = tf.reshape(
                sqrt_a_t, 
                (-1, 1, 1, 1)
            )
            sqrt_one_minus_a_t = tf.reshape(
                sqrt_one_minus_a_t, 
                (-1, 1, 1, 1)
            )

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        stable_x_t = tf.cast(x_t, stable_dtype)
        stable_eps = tf.cast(eps, stable_dtype)
        x0 = tf.cast((
            stable_x_t - sqrt_one_minus_a_t * stable_eps
            ) / sqrt_a_t, 
            x_t.dtype
        ) if not self.swap_noise_image else eps

        return x0, eps

    def forward( # call function
        self, 
        network_name: NetworkName, 
        x_t: tf.Tensor, 
        t: tf.Tensor, 
        t_batch: tf.Tensor, 
        cond_labels: tf.Tensor, 
        uncond_labels: tf.Tensor | None = None, 
        scale: float | None = None, 
        training: bool | None = None
    ) -> tuple[
        tf.Tensor, tf.Tensor, 
        tuple[list[tf.Tensor], list[tf.Tensor] | None], 
        tuple[
            list[tuple[tf.Tensor, tf.Tensor]], 
            list[tuple[tf.Tensor, tf.Tensor]] | None
        ]
    ]:
        """Run network pass(es), guidance, and algebraic x0 reconstruction.

        Floating image/prediction dtypes follow call_network and denoise: epsilon and
        auxiliary outputs retain native dtypes, while reconstructed x0 normally uses
        x_t.dtype after stable-policy arithmetic. t/t_batch and condition tensors are
        integer schedule/embedding IDs. Forward execution may advance native stochastic
        state or training-mode normalization, but applies no optimizer/EMA update.

        Args:
            network_name (NetworkName | Literal["teacher"]): Existing raw, EMA, or runtime
                teacher branch selected through get_network.
            x_t (tf.Tensor): Noisy images ``[B,H,W,C]``.
            t (tf.Tensor): Scalar or batch timestep used to gather schedule rates.
            t_batch (tf.Tensor): Batch-shaped integer timesteps ``[B]`` supplied
                to the network's embedding.
            cond_labels (tf.Tensor): Conditional label IDs ``[B]``.
            uncond_labels (tf.Tensor | None): Null labels for CFG.
                Defaults to ``None``.
            scale (float | None): Guidance scale; None skips unconditional pass.
                Defaults to ``None``.
            training (bool | None): Keras training mode. Defaults to ``None``.

        Returns:
            tuple: ``(x0, eps, (regs_c, regs_u),
            (z_vals_list_c, z_vals_list_u))``. Image tensors
            match ``x_t``; regularizers and latent pairs preserve branch outputs.

        Raises:
            ValueError: Network selection is unsupported or an attached callable teacher has
                incompatible geometry. Callable teacher type/numerics checks and native
                network/schedule errors propagate through call_network and denoise.
        """

        (eps_c, eps_u), *others = self.call_network(
            x_t, 
            t_batch, 
            cond_labels, 
            uncond_labels, 
            scale, 
            network_name=network_name, 
            training=training
        )
        x0, eps = self.denoise(
            x_t, 
            t, 
            eps_c, 
            eps_u, 
            scale, 
            reshape_coefs=(t.shape == t_batch.shape)
        )

        return x0, eps, *others

    def forward_and_compute_loss(
        self, 
        network_name: NetworkName, 
        x0: tf.Tensor, 
        noises: tf.Tensor, 
        t: tf.Tensor, 
        x_t: tf.Tensor, 
        cond_labels: tf.Tensor, 
        uncond_labels: tf.Tensor, 
        classes: tf.Tensor, 
        cfg_scale: float | None, 
        teacher_noises_pred: tf.Tensor | None = None, 
        teacher_noise_mask: tf.Tensor | None = None, 
        kl_train_type: TrainType | None = None, 
        ctr_train_type: TrainType | None = None, 
        use_image_loss: bool | None = None, 
        training: bool | None = None
    ) -> tuple[
        tf.Tensor, tf.Tensor, tf.Tensor | None, 
        tf.Tensor | None, tf.Tensor | float, 
        tf.Tensor | float, tf.Tensor | float, 
        tf.Tensor, tf.Tensor | float
    ]:
        """Run the diffusion forward prediction and compute all enabled losses.

        Args:
            network_name (NetworkName): Network used for prediction.
            x0 (tf.Tensor): Clean images ``[B,H,W,C]``.
            noises (tf.Tensor): Noise targets matching ``x0``.
            t (tf.Tensor): Per-example schedule IDs ``[B]``.
            x_t (tf.Tensor): Noisy images matching ``x0``.
            cond_labels (tf.Tensor): Conditional/possibly dropped labels ``[B]``.
            uncond_labels (tf.Tensor): Null labels ``[B]``.
            classes (tf.Tensor): Zero-based ground-truth classes ``[B]``.
            cfg_scale (float | None): Guidance scale; None avoids a second pass.
            teacher_noises_pred (tf.Tensor | None): Frozen teacher noise predictions shaped
                like x_t; required while noise
                distillation is active, ignored by the aggregator otherwise.
                Defaults to ``None``.
            teacher_noise_mask (tf.Tensor | None): Per-example teacher-vocabulary mask [B];
                None averages all rows of an
                active noise-teaching objective.
                Defaults to ``None``.
            kl_train_type (TrainType | None): Conditional/null latent source; None inherits
                the configured KL branch.
                Defaults to ``None``.
            ctr_train_type (TrainType | None): Conditional/null token source; None inherits
                the configured regularizer branch.
                Defaults to ``None``.
            use_image_loss (bool | None): Whether to compute image reconstruction loss; None
                inherits use_image_loss.
                Defaults to ``None``.
            training (bool | None): Keras training mode.
                Defaults to ``None``.

        Returns:
            tuple: Nine values from compute_noise_distil_image_kl_ctr_loss: weighted
            total, noise, conditional noise, null noise, noise KD, image, KL, token loss,
            and mean token probabilities. See that helper for absent/disabled sentinels,
            scalar dtypes, and probability shapes. This method applies no optimizer update.

        Raises:
            No additional validation is performed here. The documented call_network/denoise errors
                and enabled loss-helper failures propagate.
        """

        x0_pred, noises_pred, *others = self.forward(
            network_name, x_t, t, t, 
            cond_labels=cond_labels, 
            uncond_labels=uncond_labels, 
            scale=cfg_scale, 
            training=training
        )
        outputs = self.compute_noise_distil_image_kl_ctr_loss(
            x0, noises, classes, 
            x0_pred, noises_pred, 
            z_vals_list_c=others[1][0], 
            regs_list_c=others[0][0], 
            teacher_noises_pred=teacher_noises_pred, 
            z_vals_list_u=others[1][1], 
            regs_list_u=others[0][1], 
            kl_train_type=kl_train_type, 
            ctr_train_type=ctr_train_type, 
            use_image_loss=use_image_loss, 
            cond_labels=cond_labels, 
            teacher_noise_mask=teacher_noise_mask
        )

        return outputs

    def get_results_dict(
        self, 
        noise_loss: tf.Tensor, 
        cond_noise_loss: tf.Tensor | None = None, 
        uncond_noise_loss: tf.Tensor | None = None, 
        noise_distil_loss: tf.Tensor | None = None, 
        total_loss: tf.Tensor | None = None, 
        image_loss: tf.Tensor | None = None, 
        kl_loss: tf.Tensor | None = None, 
        ctr_loss: tf.Tensor | None = None, 
        ctr_preds: tf.Tensor | None = None, 
        classes: tf.Tensor | None = None, 
        cond_labels: tf.Tensor | None = None, 
        use_total_loss: bool | None = None, 
        use_noise_distil_loss: bool | None = None, 
        use_image_loss: bool | None = None, 
        use_kl_loss: bool | None = None, 
        use_ctr_loss: bool | None = None, 
        teacher_noise_mask: tf.Tensor | None = None
    ) -> dict[str, tf.Tensor]:
        """Update enabled diffusion metric trackers and return their results.

        Scalar loss trackers weight each batch by len(classes), or one when classes
        is absent. Noise-KD uses the sum of teacher_noise_mask when supplied;
        split-noise trackers use conditional/null row counts when cond_labels is
        available. Accuracy tracks individual examples. All values are
        running aggregates until Keras or the caller resets the metric objects.

        Loss inputs are rank-zero real-floating tensors; probability inputs are
        [B,num_classes], integer classes/conditions are [B], and optional teacher masks
        are Boolean/numeric [B] (or role-ordered mask tuples). Returned values are scalar
        tensors in their Keras trackers' policy variable dtype. Updating this method
        changes metric accumulators only, not network/optimizer weights.

        Args:
            noise_loss (tf.Tensor): Required scalar noise loss.
            cond_noise_loss (tf.Tensor | None): Conditional-row noise loss for
                optional split reporting.
                Defaults to ``None``.
            uncond_noise_loss (tf.Tensor | None): Null-row noise loss for
                optional split reporting.
                Defaults to ``None``.
            noise_distil_loss (tf.Tensor | None): Teacher-student noise loss.
                Defaults to ``None``.
            total_loss (tf.Tensor | None): Required when total tracking is on.
                Defaults to ``None``.
            image_loss (tf.Tensor | None): Required when image tracking is on.
                Defaults to ``None``.
            kl_loss (tf.Tensor | None): Required when KL tracking is on.
                Defaults to ``None``.
            ctr_loss (tf.Tensor | None): Required when token tracking is on.
                Defaults to ``None``.
            ctr_preds (tf.Tensor | None): ``[B,num_classes]`` token predictions.
                Defaults to ``None``.
            classes (tf.Tensor | None): Ground-truth classes ``[B]``.
                Defaults to ``None``.
            cond_labels (tf.Tensor | None): Post-dropout condition IDs used to
                weight split-noise means by their numbers of rows. When
                omitted, each supplied split loss receives unit weight.
                Defaults to ``None``.
            use_total_loss (bool | None): Explicit total tracker switch; None
                enables it when any auxiliary loss is enabled.
                Defaults to ``None``.
            use_noise_distil_loss (bool | None): Explicit tracker switch; None inherits
                self.use_noise_distil_loss.
                An enabled switch requires the corresponding loss/prediction inputs.
                Defaults to ``None``.
            use_image_loss (bool | None): Explicit tracker switch; None inherits
                self.use_image_loss.
                An enabled switch requires the corresponding loss/prediction inputs.
                Defaults to ``None``.
            use_kl_loss (bool | None): Explicit tracker switch; None inherits
                self.use_kl_loss.
                An enabled switch requires the corresponding loss/prediction inputs.
                Defaults to ``None``.
            use_ctr_loss (bool | None): Explicit tracker switch; None inherits
                self.use_ctr_loss.
                An enabled switch requires the corresponding loss/prediction inputs.
                Defaults to ``None``.
            teacher_noise_mask (tf.Tensor | None): Eligible teacher-row weights
                used to compute noise_distil_loss; None includes every row.
                Defaults to None; independent teacher targets can instead supply a role-ordered tuple of
                bool/numeric [B] masks.

        Returns:
            dict[str, tf.Tensor]: Current running metric values keyed by tracker
            names.

        Raises:
            AssertionError: If an enabled metric's required value is missing.
        """

        use_noise_distil_loss = self.use_noise_distil_loss if use_noise_distil_loss is None \
                                else use_noise_distil_loss
        use_image_loss = self.use_image_loss if use_image_loss is None else use_image_loss
        use_kl_loss = self.use_kl_loss if use_kl_loss is None else use_kl_loss
        use_ctr_loss = self.use_ctr_loss if use_ctr_loss is None else use_ctr_loss
        use_total_loss = use_image_loss or use_kl_loss or use_ctr_loss or use_noise_distil_loss \
                        if use_total_loss is None else use_total_loss

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        batch_weight = tf.cast(tf.shape(classes)[0], stable_dtype) \
                       if classes is not None else tf.cast(1., stable_dtype)
        results = {}

        # Update the total-loss tracker only when that loss was requested.
        if use_total_loss:
            require(
                total_loss is not None, 
                "When use_total_loss is True, total_loss cannot be None."
            )

            self.total_loss_tracker.update_state(
                total_loss, 
                sample_weight=batch_weight
            )
            results.update({
                self.total_loss_tracker.name: 
                self.total_loss_tracker.result()
            })

        self.noise_loss_tracker.update_state(
            noise_loss, 
            sample_weight=batch_weight
        )
        results.update({
            self.noise_loss_tracker.name: 
            self.noise_loss_tracker.result()
        })       

        # Update optional split means using sample counts rather than batches.
        if cond_noise_loss is not None and uncond_noise_loss is not None:
            cond_weight = tf.cast(1., stable_dtype)
            uncond_weight = tf.cast(1., stable_dtype)

            # Derive conditional/null population sizes when labels are available.
            if cond_labels is not None:
                # Under CFG split real/null conditions; without CFG every class is conditional.
                cond_mask = cond_labels != 0 if self.use_cfg else tf.ones_like(
                    cond_labels, 
                    dtype=tf.bool
                )
                cond_weight = tf.reduce_sum(tf.cast(cond_mask, stable_dtype))
                uncond_weight = tf.reduce_sum(tf.cast(
                    tf.logical_not(cond_mask), 
                    stable_dtype
                ))

            self.cond_noise_loss_tracker.update_state(
                cond_noise_loss, 
                sample_weight=cond_weight
            )
            self.uncond_noise_loss_tracker.update_state(
                uncond_noise_loss, 
                sample_weight=uncond_weight
            )
            results.update({
                self.cond_noise_loss_tracker.name:
                self.cond_noise_loss_tracker.result(), 
                self.uncond_noise_loss_tracker.name:
                self.uncond_noise_loss_tracker.result()
            })

        # Record a teacher-noise metric only when its reporting switch is active.
        if use_noise_distil_loss:
            require(
                noise_distil_loss is not None, 
                "noise_distil_loss is required when noise distillation is active."
            )

            # Dual losses are tracked separately because each teacher has its own row population.
            if self._current_teacher_spec("noise") is not None:
                weighted_means = []
                for specification in self._noise_teacher_specs():
                    tracker = getattr(self, f'{specification["role"]}_teacher_noise_distil_loss_tracker')
                    results[tracker.name] = tracker.result()
                    weighted_means.append(
                        tf.cast(tf.convert_to_tensor(specification["weight"], dtype_hint=stable_dtype), stable_dtype) * tracker.result()
                    )

                results[self.noise_distil_loss_tracker.name] = tf.add_n(weighted_means)
            # Legacy scalar targets retain the established population-weighted reporting path.
            else:
                self.noise_distil_loss_tracker.update_state(
                    noise_distil_loss, 
                    sample_weight=tf.reduce_sum(tf.cast(
                        teacher_noise_mask, stable_dtype
                    )) if teacher_noise_mask is not None else batch_weight
                )
                results[self.noise_distil_loss_tracker.name] = self.noise_distil_loss_tracker.result()

        # Update image reconstruction metrics only when image loss is active.
        if use_image_loss:
            require(
                image_loss is not None, 
                "When use_image_loss is True, image_loss cannot be None."
            )

            self.image_loss_tracker.update_state(
                image_loss, 
                sample_weight=batch_weight
            )
            results.update({
                self.image_loss_tracker.name: 
                self.image_loss_tracker.result()
            })

        # Update the KL tracker only when a KL objective is active.
        if use_kl_loss:
            require(
                kl_loss is not None, 
                "When use_kl_loss is True, kl_loss cannot be None."
            )

            self.kl_loss_tracker.update_state(
                kl_loss, 
                sample_weight=batch_weight
            )
            results.update({
                self.kl_loss_tracker.name: 
                self.kl_loss_tracker.result()
            })

        # Update class-token metrics only when their objective is active.
        if use_ctr_loss:
            require(
                ctr_loss is not None and ctr_preds is not None and 
                classes is not None, 
                "When use_ctr_loss is True, ctr_loss, "
                "ctr_preds, and classes cannot be None."
            )


            self.ctr_loss_tracker.update_state(
                ctr_loss, 
                sample_weight=batch_weight
            )
            self.ctr_accuracy_tracker.update_state(
                classes, 
                ctr_preds
            )
            results.update({
                self.ctr_loss_tracker.name: 
                self.ctr_loss_tracker.result(), 
                self.ctr_accuracy_tracker.name: 
                self.ctr_accuracy_tracker.result()
            })

        return results

    def sample_vae(
        self, 
        network_name: NetworkName = "ema", 
        labels: tf.Tensor| list | None = None, 
        add_null_label: bool = False, 
        samples_per_label: int = 1, 
        z: tf.Tensor | Sequence[tf.Tensor] | None = None, 
        seed: int | None = None
    ) -> tf.Tensor:
        """Generate images by decoding the configured variational bottleneck.

        The first ``"flatten"`` reshaper is the encoder/decoder boundary.
        Each flatten stage receives its own latent. Encoder-feature routes on
        later flatten stages build multiscale posteriors during training and
        are skipped during prior sampling. Decoder routes use the features
        available from that boundary onward.

        Labels become int32 [B] after repetition. Each latent is floating [B,D_i],
        where D_i is the configured flattened latent width for its stage, and is
        converted to policy variable dtype. Output images [B,R,R,C] use that same dtype
        after postprocess, where R is active resolution. Scaled modes clip to [0,255];
        passthrough preserves model-coordinate values without clipping. Sampling may
        advance the saved sampling stream; network/optimizer weights are not trained.

        Args:
            network_name (NetworkName): ``"ema"`` or ``"raw"`` decoder network.
                Defaults to ``'ema'``.
            labels (tf.Tensor | list[int] | None): Network condition IDs to sample.
                In dynamic mode, ``None`` shifts saved zero-based targets to
                condition IDs. Fixed-width mode selects all real conditions.
                The CFG null label is excluded unless ``add_null_label=True``.
                Explicit IDs override this default selection, including an empty
                list, and already include any CFG offset.
                Defaults to ``None``.
            add_null_label (bool): Prepend condition ID 0 to default labels for
                a CFG network. Ignored when explicit labels are supplied or CFG
                is disabled. Defaults to ``False``.
            samples_per_label (int): Repeat each selected condition contiguously
                this many times. Pass a positive integer. Defaults to ``1``.
            z (tf.Tensor | Sequence[tf.Tensor] | None): One latent batch per
                flatten stage. A tensor remains valid for a single-stage VAE.
                ``None`` draws independent standard-normal values; each batch
                size must match the repeated labels.
                Defaults to ``None``.
                A supplied tensor must be compatible with the policy variable dtype requested by
                tf.convert_to_tensor; ndarray/Python values are converted. Each per-stage rank-two shape
                is [B,D_i].
            seed (int | None): Latent random seed; None uses ``self.seed``.
                Defaults to ``None``.

        Returns:
            tf.Tensor: Decoded images ``[B,H,W,C]`` in external pixel units, normally [0,255].

        Raises:
            ValueError: If no flatten reshaper exists, it is not KL-enabled,
                or latent inputs are incompatible.
            tf.errors.InvalidArgumentError: Latent/label batches disagree or runtime latent/label
                shape and bounds checks fail. A statically incompatible latent shape/dtype can
                instead fail TensorFlow conversion/shape validation before execution.
        """

        network = self.get_network(network_name)
        flatten_ids = sorted([
            int(id_) for id_, type_ in network.reshaper_ids_dict.items()
            if type_ == "flatten"
        ])
        z_id = flatten_ids[0] if flatten_ids else None

        # Latent sampling requires an explicit flattening bottleneck.
        if z_id is None:
            raise ValueError(
                "sample_vae requires a flatten reshaper."
            )
        # Prior samples are meaningful only for a Gaussian KL bottleneck.
        if not network.reshaper_kwargs.get("add_kl", False):
            raise ValueError(
                "sample_vae requires add_kl=True in reshaper_kwargs."
            )

        reshapers = [
            network.layers_dicts[flatten_id - 1][network.R]
            for flatten_id in flatten_ids
        ]
        latent_dim_ratios = network.reshaper_kwargs.get("latent_dim_ratio") or [
            1.0 for _ in flatten_ids
        ]
        z_projectors = [
            reshaper.get_layer(
                f"{network.name_prefix}depth_{flatten_id}_{network.R[2:]}__z"
            ) if ratio != 1 else None
            for flatten_id, reshaper, ratio in zip(
                flatten_ids, reshapers, latent_dim_ratios
            )
        ]

        # Dynamic sampling uses observed classes;
        # fixed-width sampling enumerates real network labels.
        default_labels = [
            value + int(network.use_cfg)
            for value in self.seen_classes.values()
        ] if network.dynamic_num_classes else list(
            range(int(network.use_cfg), network.num_labels)
        )
        # Add a null preview only to default CFG conditions; 
        # explicit labels take precedence.
        if add_null_label and network.use_cfg:
            default_labels = [0] + default_labels
        labels = self._prepare_sampling_labels(
            network, 
            default_labels if labels is None else labels, 
            samples_per_label
        )
        n = tf.shape(labels)[0]
        seed = effective_seed(
            None, 
            task="sample seed", 
            seed=self.seed if seed is None else seed
        )
        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        ts = tf.zeros_like(labels, dtype=tf.int32)
        latent_widths = [
            int(reshaper.output_shape[1][-1]) 
            for reshaper in reshapers
        ]

        # Draw one independent latent at every variational boundary.
        if z is None:
            z_vals_list = [
                tf.random.stateless_normal(
                    shape=tf.stack((n, latent_width)), 
                    mean=0., 
                    stddev=1., 
                    seed=self._random_streams["sampling"].next_seed(
                        derive_seed(seed, "sample_vae", flatten_id)
                    ), 
                    dtype=stable_dtype
                )
                for flatten_id, latent_width in zip(
                    flatten_ids, latent_widths
                )
            ]
        # A single boundary retains the historical tensor/list input API.
        elif len(flatten_ids) == 1:
            # Unwrap an explicit one-item per-pair tensor container. A regular
            # nested Python list is instead one rank-2 latent batch; splitting
            # it here would silently keep only its first sample during ``zip``.
            z_vals_list = [z[0]] if isinstance(z, (list, tuple)) and \
            len(z) == 1 and (
                tf.is_tensor(z[0]) or isinstance(z[0], np.ndarray)
            ) else [z]
        # Multiple variational boundaries require a matching collection of latent batches.
        else:
            # Supply exactly one latent tensor for every ordered flattening stage.
            if not isinstance(z, (list, tuple)) or len(z) != len(flatten_ids):
                raise ValueError(
                    f"z must contain {len(flatten_ids)} latent tensors."
                )

            z_vals_list = list(z)

        projected_z_vals_list = []
        for latent, z_projector, latent_width in zip(
            z_vals_list, z_projectors, latent_widths
        ):
            latent = tf.ensure_shape(
                tf.convert_to_tensor(latent, dtype=stable_dtype), 
                (None, latent_width)
            )
            with tf.control_dependencies([
                assertion for assertion in tuple([
                    tf.debugging.assert_equal(
                        tf.shape(latent)[0], 
                        n, 
                        message="Latent and label batch sizes must match."
                    ) 
                ])
                if assertion is not None
            ]):
                # Project compressed latent coordinates; 
                # retain already full-width coordinates unchanged.
                projected_z_vals_list.append(
                    z_projector(latent, training=False)
                    if z_projector is not None else tf.identity(latent)
                )

        decoder_input = projected_z_vals_list[0] if len(projected_z_vals_list) == 1 \
                        else projected_z_vals_list
        # Standalone decoders require explicit absent encoder context during prior generation.
        if isinstance(network, DiTDecoder):
            images = network(
                (decoder_input, ts, labels), 
                encoder_cond=None, 
                encoder_features_list=[None] * len(
                    network.encoder_feature_dims
                ), 
                min_depth=z_id, 
                training=False
            )
        # Other variational networks decode directly from the selected flattening boundary.
        else:
            images = network(
                (decoder_input, ts, labels), 
                min_depth=z_id, 
                training=False
            )
        # Composite networks expose reconstructed images through the named noise-output field.
        if isinstance(images, Mapping):
            images = images["noises"]

        return self.postprocess(images, clip=True)

    def sample(
        self, 
        network_name: NetworkName = "ema", 
        labels: tf.Tensor| list | None = None, 
        add_null_label: bool = False, 
        samples_per_label: int = 1, 
        x_t: tf.Tensor | Sequence[tf.Tensor] | None = None, 
        steps: int | None = None, 
        scale: float | None = None, 
        eta: float | None = None, 
        return_x_ts: bool = False, 
        return_x0s: bool = False, 
        verbose: bool = False, 
        seed: int | None = None
    ) -> tf.Tensor | list[object]:
        """Generate images with generalized DDIM/DDPM reverse diffusion.

        Reverse evaluations use descending integer indices from a linspace spanning
        the full schedule, independently of active training bounds. With a supplied
        x_t and eta=0 the reverse path draws no new random noise. swap_noise_image
        delegates to sample_vae and ignores steps, scale, eta, and verbose; it rejects
        trajectory requests because no reverse chain runs.
        Sampling advances its own checkpointed stream independently of training
        noising. A seed override does not rewind it; reset_seed resets all streams.

        Reverse state/noise arithmetic and postprocessed outputs use policy variable
        dtype. Explicit image states are cast to that dtype; labels and schedule times
        are int32. Image/trajectory arrays have [B,R,R,C] at active resolution R and
        share the final output dtype. Trajectories call .numpy() and therefore require
        eager execution. Scaled modes clip every exported image to [0,255]; passthrough
        preserves values. Sampling uses inference mode and does not train weights.

        Args:
            network_name (NetworkName): ``"ema"`` or ``"raw"`` predictor.
                Defaults to ``'ema'``.
            labels (tf.Tensor | list[int] | None): Network condition IDs. In
                dynamic mode, None shifts observed zero-based targets to
                condition IDs. Fixed-width mode selects all real conditions.
                The CFG null label is excluded unless ``add_null_label=True``.
                Explicit IDs override this default selection, including an empty
                list. The repeated label count determines the batch size.
                Defaults to ``None``.
            add_null_label (bool): Prepend condition ID 0 to default labels for
                a CFG network. Ignored when explicit labels are supplied or CFG
                is disabled. Defaults to ``False``.
            samples_per_label (int): Repeat each selected condition contiguously
                this many times. Pass a positive integer. Defaults to ``1``.
            x_t (tf.Tensor | Sequence[tf.Tensor] | None): Initial Gaussian
                state ``[B,H,W,C]``. None draws it at the active resolution.
                Supplied batches must match the repeated label count.
                In ``swap_noise_image`` VAE mode this argument is instead
                passed to ``sample_vae`` as ``z`` and may contain one latent
                tensor per flatten/unflatten pair.
                Defaults to ``None``.
            steps (int | None): Number of evenly spaced reverse evaluations;
                None uses ``test_steps``. The integer-normalized value must be
                in ``[2, timesteps]``.
                Defaults to ``None``.
            scale (float | None): CFG scale; None uses ``test_cfg_scale``.  0
                follows the unconditional prediction, 1 the conditional one,
                and values above 1 extrapolate toward the condition.
                Defaults to ``None``.
            eta (float | None): Stochasticity in ``[0,1]``; None uses
                ``test_eta``.  0 gives
                deterministic DDIM, 1 is DDPM-equivalent for full consecutive
                timesteps, and values strictly between give stochastic DDIM.
                Defaults to ``None``.
            return_x_ts (bool): Include postprocessed state snapshots before
                each reverse update.
                Defaults to ``False``.
            return_x0s (bool): Include postprocessed x0 estimates at each step.
                Defaults to ``False``.
            verbose (bool): Print reverse-step progress.
                Defaults to ``False``.
            seed (int | None): Random seed for initial Gaussian noise and stochastic reverse
                steps;
                None uses self.seed. Named checkpointed stream counters still advance even
                though individual TensorFlow draws are stateless.
                Defaults to ``None``.

        Returns:
            tf.Tensor | list[object]: Final postprocessed images ``[B,H,W,C]``
            in external pixel units (normally [0,255]) when no trajectories are requested.  Otherwise returns
            ``[images, x_ts?, x0s?]`` in requested order; trajectory entries are
            lists of NumPy arrays, one per reverse step.

        Raises:
            ValueError: Reverse steps or eta are out of range, a network selector is
                unknown, or swapped variational sampling is asked to return trajectories.
            tf.errors.InvalidArgumentError: Sampling condition bounds or image/label batch
                shapes violate TensorFlow checks.
        """

        # Route sampling through the variational decoder in swapped-objective mode.
        if self.swap_noise_image:
            # The VAE shortcut returns images without a diffusion trajectory.
            if return_x_ts or return_x0s:
                raise ValueError(
                    "Sampling trajectories are unavailable "
                    "when swap_noise_image=True."
                )

            return self.sample_vae(
                network_name=network_name, 
                labels=labels, 
                add_null_label=add_null_label, 
                samples_per_label=samples_per_label, 
                z=x_t, 
                seed=seed
            )

        network = self.get_network(network_name)
        # Dynamic default labels enumerate seen classes;
        # fixed models enumerate their full real vocabulary.
        default_labels = [
            value + int(network.use_cfg)
            for value in self.seen_classes.values()
        ] if network.dynamic_num_classes else list(
            range(int(network.use_cfg), network.num_labels)
        )
        # Add a null preview only to default CFG conditions; 
        # explicit labels take precedence.
        if add_null_label and network.use_cfg:
            default_labels = [0] + default_labels
        labels = self._prepare_sampling_labels(
            network, 
            default_labels if labels is None else labels, 
            samples_per_label
        )
        n = tf.shape(labels)[0]
        seed = effective_seed(
            None, 
            task="sample seed", 
            seed=self.seed if seed is None else seed
        )
        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)

        # Draw one initial Gaussian image per requested label when absent.
        if x_t is None:
            x_t = tf.random.stateless_normal(
                tf.stack((
                    n, 
                    self._current_resolution, 
                    self._current_resolution, 
                    self.channels
                )), 
                seed=self._random_streams["sampling"].next_seed(seed), 
                dtype=stable_dtype
            )
        # Normalize and validate a caller-supplied reverse-process state.
        else:
            x_t = tf.ensure_shape(
                tf.cast(x_t, stable_dtype), 
                (
                    None, 
                    self._current_resolution, 
                    self._current_resolution, 
                    self.channels
                )
            )
            # Retain graph assertion operations while omitting eager assertions that returned None.
            with tf.control_dependencies([
                assertion for assertion in tuple([
                    tf.debugging.assert_equal(
                        tf.shape(x_t)[0], 
                        n, 
                        message="Initial-state and label batch sizes must match."
                    )
                ])
                if assertion is not None
            ]):
                x_t = tf.identity(x_t)

        steps = int(self.test_steps if steps is None else steps)
        scale = float(self.test_cfg_scale if scale is None else scale)
        eta = float(self.test_eta if eta is None else eta)

        # Keep the sampling grid within the available schedule horizon.
        if not 2 <= steps <= self.timesteps:
            raise ValueError(
                f"steps must be in [2, {self.timesteps}], got {steps!r}."
            )
        # Keep stochastic sampling strength within its finite unit interval.
        if not 0. <= float(eta) <= 1.:
            raise ValueError(
                f"eta must be a finite number in [0, 1], got {eta!r}."
            )

        ts = np.linspace(
            0, self.timesteps-1, 
            num=steps, 
            dtype="int32"
        )[::-1]
        cond_labels = labels
        uncond_labels = tf.zeros_like(
            labels, 
            dtype=tf.int32
        )
        
        steps = len(ts)
        x0s, x_ts = [], []
        for i in range(steps):
            # Report reverse-diffusion progress when requested.
            if verbose:
                print(f"\rSteps: {i+1}/{steps}", end="", flush=True)

            t = ts[i]
            t_next = ts[i + 1] if i < len(ts) - 1 else 0
            t_batch = tf.fill(tf.shape(labels), t)

            x0, eps, *_ = self.forward(
                network_name, 
                x_t, 
                t, 
                t_batch, 
                cond_labels, 
                uncond_labels, 
                scale, 
                training=False
            )

            # Capture the current noisy state for the optional trajectory.
            if return_x_ts:
                x_ts.append(self.postprocess(x_t, clip=True).numpy())
            # Capture each clean-image estimate for the optional trajectory.
            if return_x0s:
                x0s.append(self.postprocess(x0, clip=True).numpy())

            # The final t=0 prediction is already the returned clean estimate.
            # Skipping a redundant 0 -> 0 update also avoids 0/0 when timestep
            # zero was explicitly made noiseless.
            if i < steps - 1:
                alpha_bar_t = self.schedules["alpha_bar"][t]
                alpha_bar_t_next = self.schedules["alpha_bar"][t_next]
                x0_coef = tf.sqrt(alpha_bar_t_next)
                sigma_t = tf.cast(
                    eta * tf.sqrt(
                        (1. - alpha_bar_t_next) / (1. - alpha_bar_t)
                    ) * tf.sqrt(
                        1. - alpha_bar_t / alpha_bar_t_next
                    ), 
                    dtype=stable_dtype
                )
                eps_coeff = tf.cast(
                    tf.sqrt(tf.maximum(
                        1. - alpha_bar_t_next - sigma_t ** 2, 
                        0.
                    )), 
                    dtype=stable_dtype
                )

                stable_x0 = tf.cast(x0, stable_dtype)
                stable_eps = tf.cast(eps, stable_dtype)
                x_t = x0_coef * stable_x0 + eps_coeff * stable_eps
                # Add stochastic DDIM noise when eta is positive.
                if eta > 0.:
                    x_t += sigma_t * tf.random.stateless_normal(
                        tf.shape(x_t), 
                        seed=self._random_streams["sampling"].next_seed(seed), 
                        dtype=stable_dtype
                    )

        # Finish the in-place progress line after sampling.
        if verbose:
            print(flush=True)

        outputs = [self.postprocess(x0, clip=True)]
        # Append noisy-state history only when requested.
        if return_x_ts:
            outputs.append(x_ts)
        # Append clean-estimate history only when requested.
        if return_x0s:
            outputs.append(x0s)

        # Preserve the simple tensor return type when no histories were requested.
        if len(outputs) == 1:
            return outputs[0]
        return outputs


def run_self_tests() -> dict[str, str]:
    """Run deterministic end-to-end tests for DiffusionModel.

    Creates real tiny networks and runs eager training/evaluation on synthetic
    float32 image batches; no dataset download is needed. Clears the global Keras
    session before starting, sets TensorFlow's global seed to 105, and clears
    the session again on successful completion. It does not restore the previous
    random state. Assertion or TensorFlow failures stop the remaining checks.

    Args:
        None.

    Returns:
        dict[str, str]: ``{"DiffusionModel": "passed"}`` after schedule,
        noising, CFG, loss, optimizer, EMA, fit/evaluate, sampling, curriculum,
        serialization-facing state, and invalid-input checks pass.

    Raises:
        AssertionError: A numerical/state assertion fails or an invalid-input probe
                unexpectedly succeeds. TensorFlow execution errors propagate unchanged.
    """

    tf.keras.backend.clear_session()
    tf.random.set_seed(105)


    from types import SimpleNamespace
    from unittest.mock import MagicMock, Mock


    def make_network(**overrides: object) -> DiffusionTransformer:
        """Build a fresh depth-zero network for wrapper tests.

        Creates new configuration containers on every call. Overrides replace their
        corresponding defaults as a whole, so nested mappings are not deep-merged.
        The default build uses two classes, CFG, four diffusion steps, 4x4 one-channel
        images, 2x2 patches, embedding width four, one attention head, and MLP ratio one.

        The default encoder has zero transformer blocks. Additional keyword values
        pass directly to DiffusionTransformer, including build=False for tests
        that must inspect configuration before symbolic construction.

        Args:
            **overrides (object): Transformer arguments overriding test defaults.

        Returns:
            DiffusionTransformer: A built CPU-small raw network.
        """

        config = {
            "num_classes": 2, 
            "use_cfg": True, 
            "timesteps": 4, 
            "image_size": 4, 
            "channels": 1, 
            "patch_size": 2, 
            "dim": 4, 
            "depth": 0, 
            "mha_num_heads": 1, 
            "vit_block_mlp_ratio": 1.0, 
            **overrides
        }
        return DiffusionTransformer(**config)


    def make_wrapper(**overrides: object) -> DiffusionModel:
        """Build and eagerly compile a fresh test wrapper.

        Uses a supplied network override when present, otherwise the tiny factory
        network. A factory network is eagerly constructed while resolving that
        override even when a supplied network wins. The wrapper defaults to EMA
        evaluation, a linear schedule, two sampling steps, and eager Adam(1e-3)
        optimization with mean squared error. Overrides replace wrapper options
        before construction; compilation options are fixed by this fixture.

        Sets deterministic wrapper seed 17 and test_eta=0.0. Other noise, guidance,
        and auxiliary-loss settings inherit DiffusionModel defaults.

        Args:
            **overrides (object): DiffusionModel arguments overriding defaults.

        Returns:
            DiffusionModel: Compiled wrapper with a fresh raw network.
        """

        network = overrides.pop("network", make_network())
        config = {
            "network": network, 
            "preprocess_type": None, 
            "use_ema": True, 
            "test_network_name": "ema", 
            "scheduler_name": "linear", 
            "test_steps": 2, 
            "test_eta": 0.0, 
            "seed": 17, 
            **overrides
        }
        wrapper = DiffusionModel(**config)
        wrapper.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3), 
            loss="mse", 
            run_eagerly=True
        )
        return wrapper


    wrapper = make_wrapper()
    assert wrapper.image_size == wrapper.current_resolution[0] == 4
    assert wrapper.current_resolution == (4, 4)
    assert wrapper.current_timesteps_bounds == (0, 4)
    normalized_policy = make_wrapper(
        train_noisified_max_timesteps=None, 
        test_noisified_max_timesteps=None, 
        map_num_parallel_calls=np.int64(2), 
        seed=np.int64(17)
    )
    assert normalized_policy.train_noisified_max_timesteps == 0
    assert normalized_policy.test_noisified_max_timesteps == 0
    assert normalized_policy.map_num_parallel_calls == 2
    assert normalized_policy.seed == 17
    assert normalized_policy.get_config()[
        "train_noisified_max_timesteps"
    ] is None
    autotuned_mapping = make_wrapper(map_num_parallel_calls=None)
    assert autotuned_mapping.map_num_parallel_calls == tf.data.AUTOTUNE
    assert autotuned_mapping.get_config()["map_num_parallel_calls"] is None
    assert wrapper.use_ema and wrapper.ema_network is not wrapper.network
    assert len(wrapper.network.weights) == len(wrapper.ema_network.weights)
    for raw_weight, ema_weight in zip(
        wrapper.network.weights, wrapper.ema_network.weights
    ):
        tf.debugging.assert_near(raw_weight, ema_weight)
    assert [metric.name for metric in wrapper.metrics] == [
        "loss", "noise_loss", "cond_noise_loss", "uncond_noise_loss", 
        "noise_distil_loss", "image_loss", "kl_loss", "ctr_loss", 
        "ctr_accuracy"
    ]
    separate_noise_wrapper = make_wrapper(
        show_separate_noise_losses=True
    )
    assert [metric.name for metric in separate_noise_wrapper.metrics] == [
        "loss", "total_noise_loss", "cond_noise_loss", 
        "uncond_noise_loss", "noise_distil_loss", "image_loss", "kl_loss", 
        "ctr_loss", "ctr_accuracy"
    ]
    assert separate_noise_wrapper.get_config()[
        "show_separate_noise_losses"
    ] is True

    dynamic_wrapper = make_wrapper(network=make_network(
        num_classes=None, 
        cls_token_regularizer_ids=[0], 
        cls_token_regularizer_kwargs={
            "start": 0, "end": 1, "mlp_ratio": 2.0
        }
    ), seen_classes={})
    assert dynamic_wrapper.seen_classes == {}
    repeated_dynamic_data = tf.data.Dataset.from_tensor_slices(
        (tf.zeros((1, 4, 4, 1), tf.float32), tf.constant([3], tf.int32))
    ).batch(1).repeat()
    try:
        dynamic_wrapper._check_new_labels(
            x=repeated_dynamic_data, verbose=False
        )
    except ValueError as error:
        assert "finite dataset" in str(error)
    # Infinite datasets must not be scanned to discover new class IDs.
    else:
        raise AssertionError("Infinite dynamic-label scans must fail")
    dynamic_wrapper._check_new_labels(
        x=(tf.zeros((1, 4, 4, 1)), np.array([3])), 
        verbose=False
    )
    ema_regularizer = dynamic_wrapper.ema_network.labels_embed_reg
    ema_hidden_before = [
        value.copy() for value in ema_regularizer.layers[0].get_weights()
    ]
    ema_kernel, ema_bias = ema_regularizer.layers[-1].get_weights()
    ema_kernel[..., 0] = 2.0
    ema_bias[0] = 3.0
    ema_regularizer.layers[-1].set_weights([ema_kernel, ema_bias])
    dynamic_wrapper._check_new_labels(y=np.array([7]), verbose=False)
    assert dynamic_wrapper.seen_classes == {3: 0, 7: 1}
    assert (
        dynamic_wrapper._init_config["seen_classes"]
        is dynamic_wrapper.seen_classes
    )
    assert "seen_values" not in dynamic_wrapper.get_config()
    tf.debugging.assert_equal(
        dynamic_wrapper._map_classes(tf.constant([7, 3], tf.int32)), 
        tf.constant([1, 0], tf.int32)
    )
    for dynamic_network in (
        dynamic_wrapper.network, 
        dynamic_wrapper.ema_network
    ):
        assert dynamic_network.num_classes == 2
        assert dynamic_network.labels_embed_reg.layers[-1].units == 2
    for expected, actual in zip(
        ema_hidden_before, 
        dynamic_wrapper.ema_network.labels_embed_reg.layers[0].get_weights()
    ):
        np.testing.assert_array_equal(expected, actual)
    raw_kernel, raw_bias = (
        dynamic_wrapper.network.labels_embed_reg.layers[-1].get_weights()
    )
    ema_kernel, ema_bias = (
        dynamic_wrapper.ema_network.labels_embed_reg.layers[-1].get_weights()
    )
    np.testing.assert_array_equal(
        ema_kernel[..., 0], 
        np.full_like(ema_kernel[..., 0], 2.0)
    )
    assert ema_bias[0] == 3.0
    np.testing.assert_array_equal(ema_kernel[..., -1], raw_kernel[..., -1])
    assert ema_bias[-1] == raw_bias[-1]

    # Before ``compile`` there is no optimizer to register variables with;
    # both the implicit and explicit-variable forms are documented no-ops.
    uncompiled = DiffusionModel(
        network=make_network(), 
        use_ema=False, 
        test_network_name="raw", 
        scheduler_name="linear", 
        test_steps=2, 
        test_eta=0.0 
    )
    assert getattr(uncompiled, "optimizer", None) is None
    assert uncompiled._register_optimizer_variables() is None
    assert uncompiled._register_optimizer_variables(variables=[]) is None

    required_schedule_keys = {
        "betas", "alpha_bar", "sqrt_alpha_bar", 
        "sqrt_one_minus_alpha_bar", "sigmas", "timesteps"
    }
    assert required_schedule_keys <= set(wrapper.schedules)
    assert all(value.dtype == tf.float32 for value in wrapper.schedules.values())
    assert all(value.shape == tuple([4]) for value in wrapper.schedules.values())
    assert wrapper._get_progressive_timestep_boundaries(2, "uniform") == [0, 2, 4]
    log_boundaries = wrapper._get_progressive_timestep_boundaries(2, "log_snr")
    assert log_boundaries[0] == 0 and log_boundaries[-1] == 4
    assert log_boundaries[0] < log_boundaries[1] < log_boundaries[2]
    for bad_stage_count in (0, 5):
        try:
            wrapper._get_progressive_timestep_boundaries(bad_stage_count)
        except AssertionError:
            pass
        # Zero stages or more stages than timesteps must be rejected.
        else:
            raise AssertionError("Invalid progressive stage counts must fail")
    try:
        wrapper._get_progressive_timestep_boundaries(2, "unknown")
    except ValueError:
        pass
    # Unsupported timestep-clustering names must be rejected.
    else:
        raise AssertionError("Unknown timestep clustering must fail")

    for scheduler_name in (
        "linear", "scaled_linear", "squaredcos_cap_v2", "clipped_cosine", 
        "sigmoid", "quadratic", "ve", "karras", "sub_vp", "logistic"
    ):
        wrapper.load_schedules(scheduler_name=scheduler_name, timesteps=4)
        assert wrapper.scheduler_name == scheduler_name
        assert wrapper.get_config()["scheduler_name"] == scheduler_name
        assert wrapper.timesteps == 4
        assert wrapper.schedules["alpha_bar"].shape == tuple([4])
    wrapper.load_schedules("linear", 4)
    modified = make_wrapper(modify_first_t=True)
    assert float(modified.schedules["sqrt_alpha_bar"][0]) == 1.0
    assert float(modified.schedules["sqrt_one_minus_alpha_bar"][0]) == 0.0
    assert float(modified.schedules["alpha_bar"][0]) == 1.0
    tf.debugging.assert_near(
        modified.schedules["sqrt_alpha_bar"] ** 2, 
        modified.schedules["alpha_bar"]
    )
    tf.debugging.assert_near(
        modified.schedules["sqrt_one_minus_alpha_bar"] ** 2, 
        1. - modified.schedules["alpha_bar"]
    )
    tf.debugging.assert_near(
        modified.schedules["sigmas"], 
        modified.schedules["sqrt_one_minus_alpha_bar"]
    )
    tf.debugging.assert_near(
        tf.math.cumprod(1. - modified.schedules["betas"]), 
        modified.schedules["alpha_bar"]
    )
    modified_clean = tf.ones((2, 4, 4, 1), dtype=tf.float32)
    modified_x, _, modified_t = modified.noisify(
        modified_clean, 
        t=tf.constant([0, 1], dtype=tf.int32)
    )
    tf.debugging.assert_equal(modified_t, [0, 1])
    tf.debugging.assert_equal(modified_x[0], modified_clean[0])

    wrapper.set_timestep_bounds(1, 3)
    assert wrapper.current_timesteps_bounds == (1, 3)
    wrapper.set_timestep_bounds(np.int64(1), np.int64(4))
    assert wrapper.current_timesteps_bounds == (1, 4)
    wrapper.set_timestep_bounds(None, None)
    assert wrapper.current_timesteps_bounds == (0, 0)
    clean = tf.zeros((2, 4, 4, 1), dtype=tf.float32)
    clean_x, clean_noise, clean_t = wrapper.noisify(clean)
    tf.debugging.assert_equal(clean_x, clean)
    tf.debugging.assert_equal(clean_noise, tf.zeros_like(clean))
    tf.debugging.assert_equal(clean_t, tf.zeros(tuple([2]), tf.int32))
    wrapper.set_timestep_bounds()
    assert wrapper.current_timesteps_bounds == (0, 4)
    for invalid_bounds in ((-1, 2), (2, 2), (3, 2), (0, 5)):
        try:
            wrapper.set_timestep_bounds(*invalid_bounds)
        except AssertionError:
            pass
        # Negative, empty, reversed, or out-of-horizon timestep bounds must fail.
        else:
            raise AssertionError(f"Invalid timestep bounds accepted: {invalid_bounds}")
    wrapper.set_current_resolution(8)
    assert wrapper.current_resolution == (8, 8)
    wrapper.set_current_resolution(None)
    assert wrapper.current_resolution == (4, 4)

    images = tf.reshape(tf.linspace(-1.0, 1.0, 32), (2, 4, 4, 1))
    classes = tf.constant([0, 1], dtype=tf.uint8)
    fixed_t = tf.constant([0, 3], dtype=tf.int32)
    fixed_noise = tf.ones_like(images)
    split_targets = tf.concat([
        tf.ones_like(images[:1]) * 2., 
        tf.ones_like(images[1:]) * 4.
    ], axis=0)
    split_losses = separate_noise_wrapper.compute_noise_distil_image_kl_ctr_loss(
        tf.zeros_like(images), 
        split_targets, 
        classes, 
        tf.zeros_like(images), 
        tf.zeros_like(images), 
        (None, None), 
        [None], 
        cond_labels=tf.constant([1, 0], dtype=tf.uint8)
    )
    split_results = separate_noise_wrapper.get_results_dict(
        split_losses[1], 
        cond_noise_loss=split_losses[2], 
        uncond_noise_loss=split_losses[3], 
        cond_labels=tf.constant([1, 0], dtype=tf.uint8)
    )
    assert set(split_results) == {
        "total_noise_loss", "cond_noise_loss", "uncond_noise_loss"
    }
    tf.debugging.assert_near(split_results["total_noise_loss"], 10.)
    tf.debugging.assert_near(split_results["cond_noise_loss"], 4.)
    tf.debugging.assert_near(split_results["uncond_noise_loss"], 16.)
    tf.debugging.assert_near(split_losses[0], split_losses[1])
    tf.debugging.assert_equal(
        separate_noise_wrapper.cond_noise_loss_tracker.count, 1.
    )
    tf.debugging.assert_equal(
        separate_noise_wrapper.uncond_noise_loss_tracker.count, 1.
    )

    no_cfg_separate_noise_wrapper = make_wrapper(
        network=make_network(use_cfg=False), 
        use_ema=False, 
        test_network_name="raw", 
        show_separate_noise_losses=True
    )
    no_cfg_split_losses = (
        no_cfg_separate_noise_wrapper.compute_noise_distil_image_kl_ctr_loss(
            tf.zeros_like(images), 
            split_targets, 
            classes, 
            tf.zeros_like(images), 
            tf.zeros_like(images), 
            (None, None), 
            [None], 
            cond_labels=classes
        )
    )
    no_cfg_split_results = no_cfg_separate_noise_wrapper.get_results_dict(
        no_cfg_split_losses[1], 
        cond_noise_loss=no_cfg_split_losses[2], 
        uncond_noise_loss=no_cfg_split_losses[3], 
        cond_labels=classes
    )
    tf.debugging.assert_near(no_cfg_split_results["cond_noise_loss"], 10.)
    tf.debugging.assert_near(no_cfg_split_results["uncond_noise_loss"], 0.)
    tf.debugging.assert_equal(
        no_cfg_separate_noise_wrapper.cond_noise_loss_tracker.count, 2.
    )
    tf.debugging.assert_equal(
        no_cfg_separate_noise_wrapper.uncond_noise_loss_tracker.count, 0.
    )
    signal, noise_rate = wrapper.get_noise_and_signal_rates(fixed_t)
    assert signal.shape == noise_rate.shape == tuple([2])
    sampled = wrapper.q_sample(images, fixed_t, fixed_noise)
    expected = (
        signal[:, None, None, None] * images
        + noise_rate[:, None, None, None] * fixed_noise
    )
    tf.debugging.assert_near(sampled, expected)
    float64_images = tf.cast(images, tf.float64)
    float64_sample, float64_noise, _ = wrapper.noisify(
        float64_images, t=fixed_t, seed=19
    )
    assert float64_sample.dtype == float64_noise.dtype == tf.float64
    integer_images = tf.ones((1, 4, 4, 1), dtype=tf.int32)
    integer_sample, integer_noise, normalized_t = wrapper.noisify(
        integer_images, t=tf.constant([1.9]), seed=19
    )
    assert integer_sample.dtype == integer_noise.dtype == tf.as_dtype(
        wrapper.compute_dtype
    )
    tf.debugging.assert_equal(normalized_t, tf.constant([1], tf.int32))
    integer_q_sample = wrapper.q_sample(
        integer_images, 
        tf.constant([1], tf.int32), 
        tf.ones_like(integer_images)
    )
    assert integer_q_sample.dtype == tf.as_dtype(wrapper.compute_dtype)
    x_t, noises, returned_t = wrapper.noisify(images, t=fixed_t, seed=19)
    assert x_t.shape == noises.shape == images.shape
    tf.debugging.assert_equal(returned_t, fixed_t)
    random_x_t, random_noise, random_t = wrapper.noisify(
        images, min_timesteps=1, max_timesteps=3, seed=19
    )
    assert random_x_t.shape == random_noise.shape == images.shape
    assert bool(tf.reduce_all((1 <= random_t) & (random_t < 3)))
    for invalid_noisify_kwargs in tuple([
        {"min_timesteps": 2, "max_timesteps": 2}
    ]):
        try:
            wrapper.noisify(images, seed=19, **invalid_noisify_kwargs)
        except (TypeError, ValueError, tf.errors.InvalidArgumentError):
            pass
        # A non-clean empty timestep interval must be rejected by noising.
        else:
            raise AssertionError(
                f"Invalid noising inputs accepted: {invalid_noisify_kwargs}"
            )

    processed = wrapper.postprocess(
        tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0]), "standardize", clip=True
    )
    tf.debugging.assert_near(processed, [0.0, 0.0, 127.5, 255.0, 255.0])
    no_dropout = make_wrapper(p_uncond=0.0)
    shifted = tf.constant([1, 2], dtype=tf.uint8)
    tf.debugging.assert_equal(no_dropout.get_cfg_labels(shifted), shifted)
    all_dropout = make_wrapper(p_uncond=1.0)
    tf.debugging.assert_equal(
        all_dropout.get_cfg_labels(shifted), tf.zeros_like(shifted)
    )

    prepared = wrapper.prep_inputs((images, classes), use_label_dropout=False, seed=23)
    assert len(prepared) == 7
    clean, prepared_noise, prepared_t, prepared_x_t, cfg_labels, nulls, original = prepared
    assert clean.shape == prepared_noise.shape == prepared_x_t.shape == images.shape
    tf.debugging.assert_equal(cfg_labels, tf.constant([1, 2], dtype=tf.uint8))
    tf.debugging.assert_equal(nulls, tf.zeros_like(classes))
    tf.debugging.assert_equal(original, classes)
    resized_wrapper = make_wrapper()
    resized_wrapper.set_current_resolution(8)
    resized_prepared = resized_wrapper.prep_inputs((images, classes), seed=23)
    assert resized_prepared[0].shape == (2, 8, 8, 1)

    empty_ctr_loss, empty_ctr_pred = wrapper.compute_ctr_loss(classes, [None, None])
    assert empty_ctr_loss == 0.0 and empty_ctr_pred.shape == (2, 2)
    prediction1 = tf.constant([[0.8, 0.2], [0.1, 0.9]])
    prediction2 = tf.constant([[0.6, 0.4], [0.3, 0.7]])
    ctr_loss, ctr_prediction = wrapper.compute_ctr_loss(
        classes, [prediction1, None, prediction2]
    )
    tf.debugging.assert_near(ctr_prediction, (prediction1 + prediction2) / 2)
    assert float(ctr_loss) > 0.0

    conditional = tf.ones_like(images)
    unconditional = tf.zeros_like(images)
    reconstructed, selected_eps = wrapper.denoise(
        x_t, fixed_t, conditional, unconditional, scale=1.0, 
        reshape_coefs=True
    )
    assert reconstructed.shape == selected_eps.shape == images.shape
    tf.debugging.assert_equal(selected_eps, conditional)
    _, guided_eps = wrapper.denoise(
        x_t, fixed_t, conditional, unconditional, scale=2.0, 
        reshape_coefs=True
    )
    tf.debugging.assert_equal(guided_eps, 2.0 * conditional)
    network_outputs = wrapper.call_network(
        x_t, fixed_t, cfg_labels, nulls, scale=2.0, 
        network_name="raw", training=False
    )
    assert network_outputs[0][0].shape == network_outputs[0][1].shape == images.shape
    assert network_outputs[1][0] == [None]
    forward = wrapper.forward(
        "raw", x_t, fixed_t, fixed_t, cfg_labels, nulls, 
        scale=2.0, training=False
    )
    assert forward[0].shape == forward[1].shape == images.shape
    losses_tuple = wrapper.forward_and_compute_loss(
        "raw", images, noises, fixed_t, x_t, cfg_labels, nulls, classes, 
        cfg_scale=None, use_image_loss=True, training=False
    )
    assert len(losses_tuple) == 9
    assert all(
        value is None or bool(tf.reduce_all(tf.math.is_finite(value)))
        for value in losses_tuple[:7] + losses_tuple[8:]
    )

    weighted = make_wrapper(
        noise_loss_coef=0.5, 
        image_loss_coef=0.25, 
        train_noisified_min_timesteps=1, 
        train_noisified_max_timesteps=3, 
        test_noisified_min_timesteps=2, 
        test_noisified_max_timesteps=4, 
        resize_method="bilinear", 
        resize_antialias=False 
    )
    assert float(weighted.noise_loss_coef) == 0.5
    assert float(weighted.image_loss_coef) == 0.25
    assert weighted.use_image_loss is True
    assert weighted.train_noisified_min_timesteps == 1
    assert weighted.train_noisified_max_timesteps == 3
    assert weighted.test_noisified_min_timesteps == 2
    assert weighted.test_noisified_max_timesteps == 4
    assert weighted.resize_method == "bilinear"
    assert weighted.resize_antialias is False
    weighted.set_timestep_bounds(
        weighted.train_noisified_min_timesteps, 
        weighted.train_noisified_max_timesteps 
    )
    weighted_prepared = weighted.prep_inputs((images, classes), seed=47)
    assert bool(tf.reduce_all((1 <= weighted_prepared[2]) & (weighted_prepared[2] < 3)))
    weighted.set_timestep_bounds(
        weighted.test_noisified_min_timesteps, 
        weighted.test_noisified_max_timesteps
    )
    assert bool(
        tf.reduce_all(
            weighted.noisify(images, seed=47)[2]
            >= weighted.test_noisified_min_timesteps
        )
    )
    weighted.set_timestep_bounds()
    weighted_losses = weighted.forward_and_compute_loss(
        "raw", 
        weighted_prepared[0], 
        weighted_prepared[1], 
        weighted_prepared[2], 
        weighted_prepared[3], 
        weighted_prepared[4], 
        weighted_prepared[5], 
        weighted_prepared[6], 
        cfg_scale=None, 
        use_image_loss=True, 
        training=False 
    )
    tf.debugging.assert_near(
        weighted_losses[0], 
        0.5 * weighted_losses[1] + 0.25 * weighted_losses[5], 
        atol=1e-5 
    )
    weighted.set_current_resolution(8)
    assert weighted.prep_inputs((images, classes), seed=47)[0].shape == (
        2, 8, 8, 1
    )
    weighted.set_current_resolution(None)

    # Exercise every direct metric-input contract independently.  Total,
    # image, and KL tracking each require their scalar, while CTR tracking
    # requires all of its loss, prediction, and label inputs.
    results_probe = make_wrapper()
    common_result_flags = {
        "use_total_loss": False, 
        "use_image_loss": False, 
        "use_kl_loss": False, 
        "use_ctr_loss": False 
    }
    missing_result_cases = (
        ({**common_result_flags, "use_total_loss": True}, "total_loss"), 
        ({**common_result_flags, "use_image_loss": True}, "image_loss"), 
        ({**common_result_flags, "use_kl_loss": True}, "kl_loss") 
    )
    for result_flags, missing_name in missing_result_cases:
        try:
            results_probe.get_results_dict(
                noise_loss=tf.constant(0.0), 
                **result_flags
            )
        except AssertionError as error:
            assert missing_name in str(error)
        # Enabled total, image, or KL tracking must reject its missing scalar.
        else:
            raise AssertionError(
                f"Enabled {missing_name} tracking must require its input"
            )
    valid_ctr_predictions = tf.one_hot(classes, depth=2, dtype=tf.float32)
    for ctr_inputs, missing_name in (
        ({"ctr_preds": valid_ctr_predictions, "classes": classes}, "ctr_loss"), 
        ({"ctr_loss": tf.constant(0.0), "classes": classes}, "ctr_preds"), 
        (
            {
                "ctr_loss": tf.constant(0.0), 
                "ctr_preds": valid_ctr_predictions 
            }, 
            "classes" 
        )
    ):
        try:
            results_probe.get_results_dict(
                noise_loss=tf.constant(0.0), 
                use_total_loss=False, 
                use_image_loss=False, 
                use_kl_loss=False, 
                use_ctr_loss=True, 
                **ctr_inputs 
            )
        except AssertionError as error:
            assert "ctr_loss, ctr_preds, and classes" in str(error)
        # Token tracking must reject missing loss, prediction, or label inputs.
        else:
            raise AssertionError(
                f"CTR tracking must reject a missing {missing_name}"
            )
    complete_results = results_probe.get_results_dict(
        noise_loss=tf.constant(1.0), 
        total_loss=tf.constant(2.0), 
        image_loss=tf.constant(3.0), 
        kl_loss=tf.constant(4.0), 
        ctr_loss=tf.constant(5.0), 
        ctr_preds=valid_ctr_predictions, 
        classes=classes, 
        use_total_loss=True, 
        use_image_loss=True, 
        use_kl_loss=True, 
        use_ctr_loss=True 
    )
    assert set(complete_results) == {
        "loss", "noise_loss", "image_loss", "kl_loss", "ctr_loss", 
        "ctr_accuracy"
    }

    training_results = wrapper.train_step((images, classes))
    assert "noise_loss" in training_results
    testing_results = wrapper.test_step((images, classes))
    assert {"loss", "noise_loss", "image_loss"} <= set(testing_results)
    eval_dropout = make_wrapper(
        p_uncond=1.0, show_separate_noise_losses=True
    )
    eval_dropout_results = eval_dropout.test_step((images, classes))
    assert float(eval_dropout_results["cond_noise_loss"]) > 0.
    assert float(eval_dropout_results["uncond_noise_loss"]) == 0.
    eval_dropout._preprocess_training = True
    mapped_training = eval_dropout.prep_inputs_map(images, classes)
    tf.debugging.assert_equal(mapped_training[4], tf.zeros_like(shifted))
    eval_dropout._preprocess_training = False
    mapped_eval = eval_dropout.prep_inputs_map(images, classes)
    tf.debugging.assert_equal(mapped_eval[4], shifted)
    eval_dropout._preprocess_training = None
    dataset = tf.data.Dataset.from_tensor_slices((images, classes)).batch(2)
    history = wrapper.fit(dataset, epochs=1, verbose=0)
    assert len(history.history["noise_loss"]) == 1
    evaluated = wrapper.evaluate(
        dataset, network_name="raw", return_dict=True, verbose=0
    )
    assert "noise_loss" in evaluated
    separate_noise_wrapper.reset_metrics()
    separate_training_results = separate_noise_wrapper.train_step(
        (images, classes)
    )
    assert {
        "total_noise_loss", "cond_noise_loss", "uncond_noise_loss"
    } <= set(separate_training_results)
    assert "noise_loss" not in separate_training_results
    separate_history = separate_noise_wrapper.fit(
        dataset, epochs=1, verbose=0
    )
    assert {
        "total_noise_loss", "cond_noise_loss", "uncond_noise_loss"
    } <= set(separate_history.history)
    progressive_separate_history = separate_noise_wrapper.fit_progressively(
        "timesteps_only", 
        timestep_boundaries=[(0, 4)], 
        stages_verbose=False, 
        stage_epochs=1, 
        final_epochs=0, 
        x=dataset, 
        verbose=0
    )
    assert "total_noise_loss" in progressive_separate_history.history
    assert wrapper.test_network_name == "ema"
    summary_lines = []
    wrapper.summary(print_fn=summary_lines.append)
    assert any("Total params" in line for line in summary_lines)
    for invalid_labels in ([1.9], [True]):
        with np.testing.assert_raises(ValueError):
            wrapper._prepare_sampling_labels(wrapper.network, invalid_labels)
    for invalid_labels in ([-1], [2 ** 32 + 1]):
        with np.testing.assert_raises(tf.errors.InvalidArgumentError):
            wrapper._prepare_sampling_labels(wrapper.network, invalid_labels)
    for invalid_count in (0, -1, 1.5, True):
        with np.testing.assert_raises(ValueError):
            wrapper._prepare_sampling_labels(wrapper.network, [1], invalid_count)
    assert wrapper._prepare_sampling_labels(wrapper.network, [], 2).shape == tuple([0])

    raw_sample = wrapper.sample(
        network_name="raw", labels=[1, 2], steps=2, eta=0.0, seed=29
    )
    assert raw_sample.shape == (2, 4, 4, 1)
    assert bool(tf.reduce_all((0.0 <= raw_sample) & (raw_sample <= 1.0)))
    trajectories = wrapper.sample(
        network_name="raw", labels=[1], steps=3, eta=1.0, 
        return_x_ts=True, return_x0s=True, seed=29
    )
    assert len(trajectories) == 3
    assert trajectories[0].shape == (1, 4, 4, 1)
    assert len(trajectories[1]) == len(trajectories[2]) == 3
    all_label_sample = wrapper.sample(
        network_name="raw", labels=None, steps=2, eta=0.0, seed=29
    )
    assert all_label_sample.shape == (wrapper.network.num_classes, 4, 4, 1)
    supplied_state = tf.zeros((1, 4, 4, 1), dtype=tf.float32)
    supplied_sample = wrapper.sample(
        network_name="raw", labels=[1], x_t=supplied_state, 
        steps=2, eta=0.0, seed=29
    )
    assert supplied_sample.shape == supplied_state.shape
    states_only = wrapper.sample(
        network_name="raw", labels=[1], steps=2, eta=0.0, 
        return_x_ts=True, return_x0s=False, seed=29
    )
    clean_only = wrapper.sample(
        network_name="raw", labels=[1], steps=2, eta=0.0, 
        return_x_ts=False, return_x0s=True, verbose=True, seed=29
    )
    assert len(states_only) == len(clean_only) == 2
    assert len(states_only[1]) == len(clean_only[1]) == 2

    for invalid_sample_kwargs in (
        {"steps": 1, "eta": 0.0}, 
        {"steps": 5, "eta": 0.0}, 
        {"steps": 2, "eta": -0.1}, 
        {"steps": 2, "eta": 1.1}
    ):
        try:
            wrapper.sample(
                network_name="raw", 
                labels=[1], 
                seed=31, 
                **invalid_sample_kwargs
            )
        except (TypeError, ValueError):
            pass
        # Invalid reverse-step counts or eta outside [0,1] must be rejected.
        else:
            raise AssertionError(
                f"Invalid sampling overrides accepted: {invalid_sample_kwargs}"
            )
    invalid_sampling_inputs = (
        {"labels": [[1]]}, 
        {"labels": [wrapper.network.num_labels]}, 
        {"labels": [1], "x_t": tf.zeros((2, 4, 4, 1))}, 
        {"labels": [1], "x_t": tf.zeros((1, 2, 4, 1))}
    )
    for invalid_inputs in invalid_sampling_inputs:
        try:
            wrapper.sample(
                network_name="raw", 
                steps=2, 
                eta=0.0, 
                seed=31, 
                **invalid_inputs
            )
        except (TypeError, ValueError, tf.errors.InvalidArgumentError):
            pass
        # Invalid sampling labels or initial-image shapes must be rejected.
        else:
            raise AssertionError(
                f"Invalid sampling inputs accepted: {invalid_inputs}"
            )
    try:
        wrapper.sample_vae(network_name="raw", labels=[1])
    except ValueError as error:
        assert "flatten reshaper" in str(error)
    # VAE sampling must reject networks without a flatten bottleneck.
    else:
        raise AssertionError("VAE sampling without a bottleneck must fail")


    def make_variational_network(
        latent_dim_ratio: float = 1.0, 
        add_kl: bool = True, 
        build: bool = True, 
        connection_ids_dict: dict[int, list[int]] | None = None
    ) -> DiffusionTransformer:
        """Build a tiny KL bottleneck network for wrapper self-tests.

        Extends the base tiny factory to depth four, using transformer blocks at
        depths 1 and 4 and a flatten/unflatten pair at depths 2 and 3. A learned class
        token and token regularizers exercise auxiliary-loss paths. No connector routes
        are installed unless connection_ids_dict is supplied.

        Args:
            latent_dim_ratio (float): Latent-width multiplier forwarded as a one-element
                ratio list to the
                flatten reshaper. Values below one exercise learned latent projections;
                1.0 uses the full flattened width when KL sampling is enabled.
                Defaults to ``1.0``.
            add_kl (bool): Whether the flatten reshaper exposes a KL latent.
                Defaults to ``True``.
            build (bool): Whether to symbolically build the raw network.
                Defaults to ``True``.
            connection_ids_dict (dict[int, list[int]] | None): Depth-to-source-depth routes
                forwarded to the network. None explicitly
                supplies an empty mapping; a provided mapping is used without merging.
                Defaults to ``None``.

        Returns:
            DiffusionTransformer: New four-depth network with the requested bottleneck.
            build=True symbolically builds the network; False leaves construction deferred.
        """

        # Use a connector-free variational test fixture unless custom routes are supplied.
        return make_network(
            depth=4, 
            vit_block_ids=[1, 4], 
            cls_token_type="new_weight", 
            cls_token_regularizer_ids=[None], 
            reshaper_ids_dict={2: "flatten", 3: "unflatten"}, 
            reshaper_kwargs={
                "add_kl": add_kl, 
                "latent_dim_ratio": [latent_dim_ratio]
            }, 
            connection_ids_dict=(
                {} if connection_ids_dict is None else connection_ids_dict
            ), 
            build=build 
        )


    variational_cond = make_wrapper(
        network=make_variational_network(), 
        kl_loss_coef=0.01, 
        ctr_loss_coef=0.01, 
        kl_train_type="cond", 
        ctr_train_type="cond", 
        train_cfg_scale=None 
    )
    assert variational_cond.use_kl_loss and variational_cond.use_ctr_loss
    variational_cond_results = variational_cond.train_step((images, classes))
    assert {"kl_loss", "ctr_loss", "ctr_accuracy"} <= set(
        variational_cond_results
    )
    variational_uncond = make_wrapper(
        network=make_variational_network(), 
        kl_loss_coef=0.01, 
        ctr_loss_coef=0.01, 
        kl_train_type="uncond", 
        ctr_train_type="uncond", 
        train_cfg_scale=1.0 
    )
    variational_uncond_results = variational_uncond.train_step((images, classes))
    assert {"kl_loss", "ctr_loss", "ctr_accuracy"} <= set(
        variational_uncond_results
    )

    tensor_vae_labels = tf.constant([1, 2], dtype=tf.uint8)
    vae_images = variational_cond.sample_vae(
        network_name="raw", labels=tensor_vae_labels, seed=53
    )
    assert vae_images.shape == (2, 4, 4, 1)
    assert bool(tf.reduce_all((0.0 <= vae_images) & (vae_images <= 1.0)))
    full_reshaper = variational_cond.network.layers_dicts[1][
        variational_cond.network.R
    ]
    full_latent_width = int(full_reshaper.output_shape[1][-1])
    supplied_full_latent = tf.zeros((2, full_latent_width), dtype=tf.float32)
    supplied_vae_images = variational_cond.sample_vae(
        network_name="raw", labels=tensor_vae_labels, z=supplied_full_latent
    )
    assert supplied_vae_images.shape == (2, 4, 4, 1)
    supplied_sequence_images = variational_cond.sample_vae(
        network_name="raw", 
        labels=tensor_vae_labels, 
        z=[supplied_full_latent]
    )
    tf.debugging.assert_near(
        supplied_sequence_images, 
        supplied_vae_images
    )
    supplied_nested_images = variational_cond.sample_vae(
        network_name="raw", 
        labels=tensor_vae_labels, 
        z=supplied_full_latent.numpy().tolist()
    )
    tf.debugging.assert_near(
        supplied_nested_images, 
        supplied_vae_images
    )
    for invalid_z in (
        tf.zeros((1, full_latent_width), dtype=tf.float32), 
        tf.zeros((2, full_latent_width + 1), dtype=tf.float32), 
        tf.zeros((2, 1, full_latent_width), dtype=tf.float32)
    ):
        try:
            variational_cond.sample_vae(
                network_name="raw", 
                labels=tensor_vae_labels, 
                z=invalid_z
            )
        except (TypeError, ValueError, tf.errors.InvalidArgumentError):
            pass
        # VAE sampling must reject latent batch, width, or rank mismatches.
        else:
            raise AssertionError(
                f"Invalid VAE sampling latent accepted: {invalid_z.shape}"
            )
    list_vae_images = variational_cond.sample_vae(
        network_name="raw", labels=[1, 2], seed=53
    )
    assert list_vae_images.shape == (2, 4, 4, 1)
    default_label_vae_images = variational_cond.sample_vae(
        network_name="raw", labels=None, seed=53
    )
    assert default_label_vae_images.shape == (
        variational_cond.network.num_classes, 4, 4, 1
    )
    try:
        variational_cond.sample_vae(
            network_name="raw", labels=[[1, 2]], seed=53
        )
    except (ValueError, tf.errors.InvalidArgumentError):
        pass
    # VAE sampling must reject a matrix of conditioning labels.
    else:
        raise AssertionError("VAE labels must be one-dimensional")
    projected_vae = make_wrapper(
        network=make_variational_network(latent_dim_ratio=0.5), 
        use_ema=False, 
        test_network_name="raw"
    )
    random_projected_images = projected_vae.sample_vae(
        network_name="raw", labels=tensor_vae_labels, seed=53
    )
    assert random_projected_images.shape == (2, 4, 4, 1)
    projected_reshaper = projected_vae.network.layers_dicts[1][
        projected_vae.network.R
    ]
    projected_latent_width = int(projected_reshaper.output_shape[1][-1])
    projected_latent = tf.zeros((2, projected_latent_width), dtype=tf.float32)
    projected_images = projected_vae.sample_vae(
        network_name="raw", labels=tensor_vae_labels, z=projected_latent
    )
    assert projected_images.shape == (2, 4, 4, 1)

    # Mirror the central multi-level U-DiT bridge with training-only encoder
    # routes on its later flatten stages. Distinct ratios also cover a
    # full-width middle latent, for which no `/z` projector exists.
    multilevel_network = make_network(
        depth=8, 
        vit_block_ids=[1, 8], 
        connection_ids_dict={
            4: [1], 
            6: [0], 
            8: [3, 5, 7]
        }, 
        reshaper_ids_dict={
            2: "flatten", 3: "unflatten", 
            4: "flatten", 5: "unflatten", 
            6: "flatten", 7: "unflatten"
        }, 
        reshaper_kwargs={
            "add_kl": True, 
            "latent_dim_ratio": [0.5, 1.0, 0.25]
        }
    )
    multilevel_outputs = multilevel_network(
        (images, tf.zeros_like(classes), tensor_vae_labels), 
        full_return=True, 
        training=False
    )
    assert [
        int(z_mean.shape[-1])
        for z_mean, _ in multilevel_outputs[-1]
    ] == [8, 16, 4]
    assert multilevel_network.get_config()["reshaper_kwargs"][
        "latent_dim_ratio"
    ] == [0.5, 1.0, 0.25]
    # A bounded resume needs latents only for flatten stages that execute
    # before its exclusive maximum depth.
    truncated_multilevel, *_ = multilevel_network.encode(
        (
            [tf.zeros((2, 16), dtype=tf.float32)], 
            tf.zeros_like(classes), 
            tensor_vae_labels
        ), 
        min_depth=2, 
        max_depth=3, 
        training=False
    )
    assert truncated_multilevel.shape == (2, 4, 4)
    multilevel_vae = make_wrapper(
        network=multilevel_network
    )
    assert multilevel_vae.ema_network.reshaper_kwargs[
        "latent_dim_ratio"
    ] == [0.5, 1.0, 0.25]
    random_multilevel_images = multilevel_vae.sample_vae(
        network_name="ema", 
        labels=tensor_vae_labels, 
        seed=59
    )
    assert random_multilevel_images.shape == (2, 4, 4, 1)
    multilevel_latents = [
        tf.zeros((2, width), dtype=tf.float32)
        for width in (8, 16, 4)
    ]
    supplied_multilevel_images = multilevel_vae.sample_vae(
        network_name="raw", 
        labels=tensor_vae_labels, 
        z=multilevel_latents
    )
    assert supplied_multilevel_images.shape == (2, 4, 4, 1)
    try:
        multilevel_vae.sample_vae(
            network_name="raw", 
            labels=tensor_vae_labels, 
            z=multilevel_latents[:-1]
        )
    except ValueError as error:
        assert "3 latent tensors" in str(error)
    # A multilevel decoder must reject an incomplete list of supplied latents.
    else:
        raise AssertionError("Every multilevel VAE latent must be supplied")

    non_variational = make_wrapper(
        network=make_variational_network(add_kl=False), 
        use_ema=False, 
        test_network_name="raw"
    )
    try:
        non_variational.sample_vae(
            network_name="raw", labels=tensor_vae_labels, seed=53
        )
    except ValueError as error:
        assert "add_kl=True" in str(error)
    # A flatten bottleneck without KL sampling must not enable VAE generation.
    else:
        raise AssertionError("VAE sampling without add_kl must fail")

    original_bounds = wrapper.current_timesteps_bounds
    original_resolution = wrapper.current_resolution
    progressive_history = wrapper.fit_progressively(
        stage_tasks=[
            {"timesteps": (2, 4)}, 
            {"resolution": 4} 
        ], 
        x=dataset, 
        stages_verbose=False, 
        stage_epochs=1, 
        final_epochs=0, 
        verbose=0 
    )
    assert len(progressive_history.progressive_stages) == 2
    assert wrapper.current_timesteps_bounds == original_bounds
    assert wrapper.current_resolution == original_resolution
    assert wrapper._add_depths(None)["network"]["added"] == 0

    syntax_wrapper = make_wrapper()
    syntax_history = syntax_wrapper.fit_progressively(
        stage_tasks=[
            "timesteps", 
            ("resolution", 4), 
            ["resolution", 4], 
            {"timesteps", "resolution"}, 
            frozenset({"timesteps"}), 
            {"timesteps": (0, 4), "resolution": 4} 
        ], 
        timestep_boundaries=[(2, 4), None, None, (1, 4), (0, 4), None], 
        resolutions=[None, None, None, 4, None, None], 
        x=dataset, 
        stages_verbose=False, 
        stage_epochs=0, 
        final_epochs=0, 
        verbose=0
    )
    assert len(syntax_history.progressive_stages) == 6
    assert syntax_history.epoch == []
    assert syntax_history.progressive_stages[3]["updates"] == {
        "timesteps": (1, 4), 
        "resolution": 4 
    }

    autogenerated_timesteps = make_wrapper().fit_progressively(
        "timesteps_only", 
        stages_num=2, 
        timestep_clustering_type="uniform", 
        x=dataset, 
        stages_verbose=False, 
        stage_epochs=0, 
        final_epochs=0, 
        verbose=0 
    )
    assert autogenerated_timesteps.timestep_boundaries == [(2, 4), (0, 4)]
    assert autogenerated_timesteps.stage_tasks == ["timesteps", "timesteps"]
    autogenerated_resolutions = make_wrapper().fit_progressively(
        "resolutions_only", 
        stages_num=2, 
        x=dataset, 
        stages_verbose=False, 
        stage_epochs=0, 
        final_epochs=0, 
        verbose=0
    )
    assert autogenerated_resolutions.resolutions == [2, 4]
    assert [
        record["resolution"]
        for record in autogenerated_resolutions.progressive_stages
    ] == [2, 4]

    supplied_timesteps = make_wrapper().fit_progressively(
        "timesteps_only", 
        timestep_boundaries=[(3, 4), (0, 4)], 
        x=dataset, 
        stages_verbose=False, 
        stage_epochs=0, 
        final_epochs=0, 
        verbose=0
    )
    assert supplied_timesteps.stages_num == 2
    supplied_resolutions = make_wrapper().fit_progressively(
        "resolutions_only", 
        resolutions=[2, 4], 
        x=dataset, 
        stages_verbose=False, 
        stage_epochs=0, 
        final_epochs=0, 
        verbose=0
    )
    assert supplied_resolutions.stages_num == 2

    depth_wrapper = make_wrapper()
    depth_history = depth_wrapper.fit_progressively(
        "depths_only", 
        depths=[None, "vision_transformer_block"], 
        x=dataset, 
        stages_verbose=False, 
        stage_epochs=0, 
        final_epochs=1, 
        verbose=0
    )
    assert depth_wrapper.network.depth == depth_wrapper.ema_network.depth == 1
    assert (
        depth_history.progressive_stages[0]["depth_growth"]["network"]["added"]
        == 0
    )
    assert (
        depth_history.progressive_stages[1]["depth_growth"]["network"]["added"]
        == 1
    )
    assert depth_history.progressive_stages[-1]["stage"] == "final"
    assert depth_history.progressive_stages[-1]["network_depth"] == 1
    assert depth_history.progressive_stages[-1]["epochs_ran"] == 1

    epoch_plateau = make_wrapper().fit_progressively(
        [{"resolution": 4}], 
        x=dataset, 
        stages_verbose=False, 
        stage_epochs=4, 
        final_epochs=0, 
        pacing_type="plateau", 
        earlystopping_type="epoch_wise", 
        monitor="noise_loss", 
        patience=0, 
        min_delta=1e9, 
        verbose=0
    )
    assert epoch_plateau.progressive_stages[0]["epochs_ran"] == 2
    batch_plateau = make_wrapper().fit_progressively(
        [{"resolution": 4}], 
        x=dataset, 
        stages_verbose=False, 
        stage_epochs=5, 
        final_epochs=0, 
        pacing_type="plateau", 
        earlystopping_type="batch_wise", 
        monitor="noise_loss", 
        patience=1, 
        min_delta=1e9, 
        verbose=0 
    )
    assert batch_plateau.progressive_stages[0]["epochs_ran"] == 2

    failing_progressive = make_wrapper()
    failing_entry_bounds = failing_progressive.current_timesteps_bounds
    failing_entry_resolution = failing_progressive.current_resolution
    try:
        failing_progressive.fit_progressively(
            [{"timesteps": (2, 4), "resolution": 2}, object()], 
            x=dataset, 
            stages_verbose=False, 
            stage_epochs=0, 
            final_epochs=0, 
            verbose=0 
        )
    except ValueError as error:
        assert "Invalid stage task at index 1" in str(error)
    # An unsupported curriculum stage object must fail after restoring entry state.
    else:
        raise AssertionError("Invalid progressive stage objects must fail")
    assert failing_progressive.current_timesteps_bounds == failing_entry_bounds
    assert failing_progressive.current_resolution == failing_entry_resolution

    for unknown_task in (
        "resoluton", 
        {"resoluton"}, 
        {"resolution": 2, "resoluton": 4}
    ):
        with np.testing.assert_raises_regex(ValueError, "Unsupported progressive task"):
            failing_progressive.fit_progressively(
                [unknown_task], 
                x=dataset, 
                stages_verbose=False, 
                stage_epochs=0, 
                final_epochs=0, 
                verbose=0
            )
        assert failing_progressive.current_timesteps_bounds == failing_entry_bounds
        assert failing_progressive.current_resolution == failing_entry_resolution

    for forbidden_fit_argument in ({"epochs": 1}, {"initial_epoch": 0}):
        try:
            wrapper.fit_progressively(
                [{"resolution": 4}], 
                x=dataset, 
                stage_epochs=0, 
                final_epochs=0, 
                **forbidden_fit_argument
            )
        except AssertionError:
            pass
        # Direct epochs/initial_epoch arguments must not override curriculum pacing.
        else:
            raise AssertionError("Managed progressive epoch arguments must fail")
    for invalid_progressive_control in (
        {"timestep_clustering_type": "unknown"}, 
        {"pacing_type": "unknown"}, 
        {"earlystopping_type": "unknown"}, 
        {"monitor": "unknown"} 
    ):
        try:
            wrapper.fit_progressively(
                [{"resolution": 4}], 
                x=dataset, 
                stage_epochs=0, 
                final_epochs=0, 
                **invalid_progressive_control
            )
        except AssertionError:
            pass
        # Unsupported clustering, pacing, stopping, or monitor selectors must fail.
        else:
            raise AssertionError(
                f"Expected invalid progressive control: {invalid_progressive_control}"
            )
    for only_mode, missing_values in (
        ("timesteps_only", {}), 
        ("resolutions_only", {}), 
        ("depths_only", {"stages_num": 1}) 
    ):
        try:
            wrapper.fit_progressively(
                only_mode, 
                stage_epochs=0, 
                final_epochs=0, 
                **missing_values 
            )
        except ValueError:
            pass
        # Single-task curricula must reject absent stage counts or required stage values.
        else:
            raise AssertionError(f"Missing values must fail for {only_mode}")

    serialization_network = make_network(build=False)
    serialization_network.built = True
    serialization_wrapper = DiffusionModel(
        network=serialization_network, 
        use_ema=False, 
        test_network_name="raw", 
        test_steps=2
    )
    wrapper_config = serialization_wrapper.get_config()
    assert isinstance(wrapper_config["network"], dict)
    serialization_clone = DiffusionModel.from_config(wrapper_config)
    assert serialization_clone is not serialization_wrapper
    assert serialization_clone.network is not serialization_wrapper.network
    assert serialization_clone.network.get_config() \
        == serialization_wrapper.network.get_config()

    policy_wrapper = make_wrapper(
        use_ema=False, 
        test_network_name="raw", 
        trainable=False, 
        dynamic=True, 
        dtype="float64", 
        name="policy_wrapper" 
    )
    assert policy_wrapper.name == "policy_wrapper"
    assert policy_wrapper.trainable is False
    assert policy_wrapper.dtype_policy.name == "float64"
    assert policy_wrapper.dynamic is True
    assert policy_wrapper.sample(
        network_name="raw", labels=[1], steps=2, eta=0.0, seed=61
    ).dtype == tf.float64

    ema_before = [value.numpy().copy() for value in wrapper.ema_network.weights]
    wrapper.network.weights[0].assign_add(tf.ones_like(wrapper.network.weights[0]))
    assert wrapper.update_ema() is True
    assert not np.array_equal(ema_before[0], wrapper.ema_network.weights[0].numpy())
    assert wrapper.get_network("raw") is wrapper.network
    assert wrapper.get_network("ema") is wrapper.ema_network
    try:
        wrapper.get_network("unknown")
    except ValueError:
        pass
    # Selectors outside the supported network names must be rejected.
    else:
        raise AssertionError("Unknown network names must fail")

    topology_probe = SimpleNamespace(
        use_ema=True, 
        network=SimpleNamespace(weights=[tf.Variable(0.0)]), 
        ema_network=SimpleNamespace(weights=[]), 
        ema_decay=0.9 
    )
    try:
        DiffusionModel.update_ema(topology_probe)
    except AssertionError as error:
        assert "same topology" in str(error)
    # EMA must reject raw and averaged networks with different weight counts.
    else:
        raise AssertionError("Raw/EMA topology mismatch must fail")

    # Progressive growth has two distinct EMA-alignment guards: the number of
    # newly created weights must match, then every aligned shape must assign.
    raw_count_weights = MagicMock()
    raw_count_weights.__iter__.side_effect = [
        iter(()), 
        iter(tuple([tf.Variable(0.0)])) 
    ]
    ema_count_weights = MagicMock()
    ema_count_weights.__iter__.side_effect = [iter(()), iter(())]
    progressive_count_probe = SimpleNamespace(
        network=SimpleNamespace(
            weights=raw_count_weights, 
            add_depths=Mock(return_value={"network": {"added": 1}}), 
            build=Mock(return_value=None)
        ), 
        ema_network=SimpleNamespace(
            weights=ema_count_weights, 
            add_depths=Mock(return_value={"network": {"added": 0}}), 
            build=Mock(return_value=None)
        )
    )
    try:
        DiffusionModel._add_depths(progressive_count_probe, "probe")
    except ValueError as error:
        assert "progressive depths have different weights" in str(error)
    # Depth growth must reject unequal numbers of new raw and EMA weights.
    else:
        raise AssertionError(
            "Progressive raw/EMA new-weight count mismatch must fail"
        )

    raw_shape_weights = MagicMock()
    raw_shape_weights.__iter__.side_effect = [
        iter(()), 
        iter(tuple([tf.Variable(tf.zeros(tuple([1])))]))
    ]
    ema_shape_weights = MagicMock()
    ema_shape_weights.__iter__.side_effect = [
        iter(()), 
        iter(tuple([tf.Variable(tf.zeros(tuple([2])))]))
    ]
    progressive_shape_probe = SimpleNamespace(
        network=SimpleNamespace(
            weights=raw_shape_weights, 
            add_depths=Mock(return_value={"network": {"added": 1}}), 
            build=Mock(return_value=None)
        ), 
        ema_network=SimpleNamespace(
            weights=ema_shape_weights, 
            add_depths=Mock(return_value={"network": {"added": 1}}), 
            build=Mock(return_value=None)
        ) 
    )
    try:
        DiffusionModel._add_depths(progressive_shape_probe, "probe")
    except ValueError as error:
        assert "shape" in str(error).lower()
    # Depth growth must reject incompatible shapes in aligned raw and EMA weights.
    else:
        raise AssertionError(
            "Progressive raw/EMA new-weight shape mismatch must fail"
        )

    without_ema = make_wrapper(
        network=make_network(use_cfg=False), 
        use_ema=False, 
        test_network_name="raw", 
        p_uncond=0.9, 
        test_cfg_scale=9.0 
    )
    assert without_ema.ema_network is None
    assert without_ema.p_uncond == 0.0 and without_ema.test_cfg_scale == 1.0
    assert without_ema.update_ema() is False
    no_cfg_prepared = without_ema.prep_inputs(
        (images, classes), use_label_dropout=True, seed=59
    )
    tf.debugging.assert_equal(no_cfg_prepared[4], classes)
    assert "noise_loss" in without_ema.train_step((images, classes))
    no_cfg_sample = without_ema.sample(
        network_name="raw", labels=None, steps=2, eta=0.0, seed=59
    )
    assert no_cfg_sample.shape == (without_ema.network.num_labels, 4, 4, 1)
    assert without_ema.get_network("ema") is without_ema.network

    swap = make_wrapper(swap_noise_image=True)
    swap_prepared = swap.prep_inputs((images, classes), seed=31)
    tf.debugging.assert_near(swap_prepared[1], swap_prepared[3])
    try:
        swap.sample(network_name="raw", labels=[1])
    except ValueError as error:
        assert "flatten reshaper" in str(error)
    # Swapped-target sampling must use the VAE path and require its bottleneck.
    else:
        raise AssertionError("swap_noise_image must route through VAE sampling")

    invalid_cases = (
        {"test_network_name": "unknown"}, 
        {"ema_decay": -0.1}, 
        {"ema_decay": 1.0}, 
        {"ema_decay": float("nan")}, 
        {"test_steps": True}, 
        {"test_steps": 1}, 
        {"test_steps": 5}, 
        {"test_eta": -0.1}, 
        {"test_eta": 1.1}, 
        {"test_eta": float("nan")}, 
        {"train_noisified_min_timesteps": -1}, 
        {"test_noisified_min_timesteps": 3, 
         "test_noisified_max_timesteps": 2}, 
        {"test_noisified_max_timesteps": 5}, 
        {"p_uncond": -0.25}, 
        {"p_uncond": 1.25}, 
        {"p_uncond": float("nan")}, 
        {"kl_train_type": "unknown"}, 
        {"kl_train_type": "uncond", "train_cfg_scale": None}, 
        {"ctr_train_type": "unknown"}, 
        {"ctr_train_type": "uncond", "train_cfg_scale": None} 
    )
    for overrides in invalid_cases:
        try:
            DiffusionModel(
                network=make_network(), 
                **{"test_steps": 2, **overrides}
            )
        except AssertionError:
            pass
        # Invalid wrapper modes, probability bounds, or timestep controls must fail.
        else:
            raise AssertionError(f"Expected invalid wrapper config: {overrides}")

    for invalid_seed in (-1, 2 ** 32):
        try:
            DiffusionModel(network=make_network(), test_steps=2, seed=invalid_seed)
        except (TypeError, ValueError):
            pass
        # Wrapper seeds outside the supported unsigned 32-bit range must fail.
        else:
            raise AssertionError("Invalid wrapper seeds must fail")

    tf.keras.backend.clear_session()
    return {"DiffusionModel": "passed"}


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
