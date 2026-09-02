"""Training, evaluation, EMA, noising, and sampling for raw diffusion networks.

Networks in ``diffusion.models.transformer`` implement tensor transformations.
This module wraps one such network in the stateful Keras training protocol and
owns the diffusion process around it.
"""

import tensorflow as tf
from tensorflow.keras import metrics, losses, callbacks, optimizers

import numpy as np

from importlib import import_module

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Literal, Sequence, get_args

from . import (
    NetworkName, 
    TrainType, 
    ClusteringType, 
    copy_network_weights_by_layer
)

from common.argument_saver import ArgumentSaverModel
from common.gradients import apply_policy_gradients
from common.runtime import derive_seed, validate_model_dtype_policy
from common.validation import require

from autoencoder.variational_autoencoder import VariationalAutoencoder

from diffusion.callbacks.batch_loss_plateau import BatchLossPlateau
from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer
from diffusion.models.transformer.di_t_decoder import DiTDecoder
from diffusion.schedulers import make_schedule, SchedulerName


class DiffusionModel(ArgumentSaverModel):
    """Orchestrate diffusion training and sampling around a raw transformer.

    The wrapped ``network`` predicts noise from ``(x_t, timestep, label)``.  This
    model constructs the schedule, samples forward-process noise, applies
    classifier-free guidance, computes configured losses, updates raw and EMA
    weights, manages progressive timestep/resolution/depth curricula, and runs
    generalized DDIM/DDPM sampling.

    This wrapper is deliberately separate from
    ``diffusion.models.transformer``: transformer classes own architecture and
    intermediate features; wrapper classes own optimization and diffusion
    state.  Call ``compile`` on the wrapper, not only on the raw network.

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
        map_preprocess (bool): Whether datasets are mapped through
            :meth:`prep_inputs_map` before Keras consumes them.
        seen_classes (dict[object, int]): Dataset labels mapped to consecutive
            zero-based classifier targets in dynamic-class mode. It is the same
            dictionary stored in ``_init_config``, so newly observed labels are
            reflected immediately in wrapper configuration.
    """

    def __init__(
        self, 
        network: ArgumentSaverModel, 
        teacher_network: tf.keras.Model | None = None, 
        use_ema: bool = True, 
        defer_teacher: bool = False, 
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
        image_loss_coef: float = 0., 
        kl_loss_coef: float = 0., 
        ctr_loss_coef: float = 0., 
        kl_train_type: TrainType = "cond", 
        ctr_train_type: TrainType = "cond", 
        train_noisified_min_timesteps: int = 0, 
        train_noisified_max_timesteps: int | None = -1, 
        test_noisified_min_timesteps: int = 0, 
        test_noisified_max_timesteps: int | None = -1, 
        resize_method: str = "area", 
        resize_antialias: bool = True, 
        swap_noise_image: bool = False, 
        map_preprocess: bool = False, 
        map_num_parallel_calls: int | None = 1, 
        seen_classes: dict[object, int] = {}, 
        noise_distil_loss_coef: float = 0., 
        seed: int | None = None, 
        **kwargs: object
    ) -> None:
        """Initialize diffusion state and an optional EMA network.

        Args:
            network (ArgumentSaverModel): Built/configurable raw transformer or
                convolutional diffusion network.  A classifier subclass is
                accepted by classifier wrappers.
            use_ema (bool): Clone ``network`` from its Keras config and maintain
                exponential moving-average weights after each train step.
                Deferred raw and cloned networks are built before the initial
                weight copy.
            test_network_name (NetworkName): ``"ema"`` or ``"raw"`` network
                selected by the default test step.  Use ``"raw"`` when
                ``use_ema=False``.
            ema_decay (float): EMA retention in ``[0,1)``.  New EMA weight is
                ``decay * old_ema + (1-decay) * raw``.
            scheduler_name (SchedulerName): One of ``"linear"``,
                ``"scaled_linear"``, ``"squaredcos_cap_v2"``,
                ``"clipped_cosine"``, ``"sigmoid"``, ``"quadratic"``,
                ``"ve"``, ``"karras"``, ``"sub_vp"``, or ``"logistic"``.
            modify_first_t (bool): Force timestep 0 to have signal rate 1,
                noise rate 0, and cumulative alpha 1 after schedule creation.
            p_uncond (float): Per-example probability of replacing a shifted
                class label with null ID 0 during training.  It is forced to 0
                when ``network.use_cfg=False``.
            train_cfg_scale (float | None): CFG scale during training.  ``None``
                runs only the conditional network pass; a number additionally
                runs the null-label pass and combines predictions.
            test_cfg_scale (float): CFG scale for evaluation/sampling, forced to
                1 when CFG is disabled.
            test_steps (int): Default reverse-sampling evaluations in
                ``[2, network.timesteps]``.
            test_eta (float): Default stochasticity in ``[0,1]``: 0 is
                deterministic DDIM and 1 is DDPM-equivalent only for consecutive
                full-schedule steps.
            noise_loss_coef (float): Multiplier for prediction-vs-noise loss.
            show_separate_noise_losses (bool): When true, report the unchanged
                full noise loss as ``total_noise_loss`` and additionally report
                ``cond_noise_loss`` and ``uncond_noise_loss`` from non-null and
                null-label rows. These metrics do not change optimization.
            image_loss_coef (float): Multiplier for reconstructed-image loss; 0
                disables it during normal training.
            kl_loss_coef (float): Multiplier for variational reshaper KL loss; 0
                disables it.
            ctr_loss_coef (float): Multiplier for auxiliary class-token
                regularizer loss; 0 disables it.
            kl_train_type (TrainType): ``"cond"`` uses conditional latent
                statistics; ``"uncond"`` uses the null-label forward pass.
            ctr_train_type (TrainType): ``"cond"`` or ``"uncond"`` source for
                auxiliary regularizer predictions.  ``"uncond"`` requires a
                non-None ``train_cfg_scale``.
            train_noisified_min_timesteps (int): Inclusive lower bound used by
                :meth:`fit`; default 0.
            train_noisified_max_timesteps (int | None): Exclusive training upper
                bound; -1 becomes ``network.timesteps`` and None becomes 0.
            test_noisified_min_timesteps (int): Inclusive evaluation lower bound.
            test_noisified_max_timesteps (int | None): Exclusive evaluation
                upper bound; -1 becomes ``network.timesteps`` and None becomes 0.
            resize_antialias (bool): Antialias flag passed to ``tf.image.resize``.
            swap_noise_image (bool): Train the raw output as a clean-image
                prediction instead of epsilon and route :meth:`sample` to
                :meth:`sample_vae`; this mode requires a compatible KL bottleneck.
            map_preprocess (bool): Map ``tf.data.Dataset`` inputs through
                :meth:`prep_inputs_map` in :meth:`fit`, :meth:`evaluate`, and
                each progressive stage. Custom train/test steps then consume
                the prepared tensors directly. The default false preserves
                online preparation in the training device path.
            map_num_parallel_calls (int | None): Positive parallel-call value
                forwarded to ``Dataset.map``. ``None`` selects
                ``tf.data.AUTOTUNE``.
            seen_classes (Mapping[object, int] | None): Saved real-label to
                zero-based classifier-target mapping for a grown continual
                model. ``None`` starts with no observed classes. A nonempty
                mapping restores dynamic growth and expands a smaller raw/EMA
                topology before checkpoint weights are loaded. The normalized
                dictionary is retained by reference in the wrapper config.
            noise_distil_loss_coef (float): Multiplier for matching a frozen
                teacher's noise prediction on the same noisy inputs.
            teacher_network (tf.keras.Model | None): Runtime-only frozen raw
                diffusion network or wrapper used by noise distillation.
            defer_teacher (bool): Permit a positive teacher objective to start
                without a teacher so continual learning can attach one later.
            seed (int | None): Default TensorFlow random seed for noising,
                label dropout, latent draws, and sampling; per-call seeds override.
            **kwargs (object): Standard ``tf.keras.Model`` keys: ``name`` (str),
                ``trainable`` (bool), ``dtype`` (dtype name/policy), and
                ``dynamic`` (bool).

        Returns:
            None: Schedule tensors, active bounds/resolution, loss flags, and
            raw/EMA networks are initialized; metric trackers are created later
            by :meth:`compile`.
        """

        super().__init__(**kwargs)
        self._check_assertions(locals())
        self._save_init_args(locals())
        DiffusionModel._refresh_loss_flags(self)

        self.network.build()
        # Clone and initialize the EMA network when EMA tracking is enabled.
        if self.use_ema:
            ema_config = self.network.get_config()
            ema_config["name"] = self.network.name + "_ema"

            self.ema_network = self.network.__class__.from_config(
                ema_config
            )
            self.ema_network.build()
            self.ema_network.set_weights(
                self.network.get_weights()
            )
        # Keep the EMA slot empty when EMA tracking is disabled.
        else:
            self.ema_network = None

        # Replay class growth in the same raw/EMA order used during fitting.
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
            for _ in range(seen_num_classes - self.network.num_classes):
                self.network.add_class()
                # Mirror each raw addition in the EMA topology when enabled.
                if self.ema_network is not None:
                    self.ema_network.add_class(
                        source_network=self.network
                    )

            self.network.build()
            # Refresh the replacement EMA classifier container as well.
            if self.ema_network is not None:
                self.ema_network.build()

        network_config = tf.keras.utils.serialize_keras_object(self.network)
        network_config["module"] = self.network.__class__.__module__
        self._init_config["network"] = network_config
        self._init_config.pop("teacher_network", None)
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
        self.seed = self._normalize_seed(self.seed)

        self._preprocess_training = None
        self._map_preprocess_without_teacher = bool(self.map_preprocess)

        if self.teacher_network is not None:
            self.defer_teacher = True
            self._init_config["defer_teacher"] = True

        self.load_schedules()
        self.set_timestep_bounds()
        DiffusionModel.set_current_resolution(self)
        self.set_teacher_network(self.teacher_network)
        self.build(())

    def _check_assertions(self, local_vars: dict[str, object]) -> None:
        """Validate schedule, EMA, sampler, and auxiliary-loss choices.

        Args:
            local_vars (dict[str, object]): Wrapper constructor namespace.

        Returns:
            None: Invalid EMA decay, sampling-step/eta ranges, train-type
            values, or unconditional regularizer configuration raise
            ``AssertionError``.
        """

        network = local_vars["network"]
        # Require the serialization-aware model interface used by this wrapper.
        if not isinstance(network, ArgumentSaverModel):
            raise TypeError(
                "network must inherit common.argument_saver.ArgumentSaverModel."
            )
        for attribute in (
            "timesteps", "image_size", "channels", "use_cfg", 
            "build", "set_current_resolution", "get_config"
        ):
            # Fail early when the network omits a required diffusion attribute.
            if not hasattr(network, attribute):
                raise TypeError(f"network must define {attribute!r}.")

        for name in (
            "use_ema", "modify_first_t", "resize_antialias", 
            "swap_noise_image", "map_preprocess", 
            "show_separate_noise_losses", "defer_teacher"
        ):
            require(
                isinstance(local_vars[name], bool), 
                f"{name} must be boolean."
            )

        require(
            local_vars["map_num_parallel_calls"] is None or (
                isinstance(local_vars["map_num_parallel_calls"], Integral)
                and not isinstance(local_vars["map_num_parallel_calls"], bool)
                and local_vars["map_num_parallel_calls"] > 0
            ),
            "map_num_parallel_calls must be None or a positive integer.",
        )

        for prefix in ("train", "test"):
            t_min = local_vars[f"{prefix}_noisified_min_timesteps"]
            raw_t_max = local_vars[f"{prefix}_noisified_max_timesteps"]
            require(
                isinstance(t_min, Integral)
                and not isinstance(t_min, bool)
                and 0 <= t_min < network.timesteps,
                f"{prefix}_noisified_min_timesteps must be in "
                f"[0, {network.timesteps}).",
            )
            require(
                raw_t_max is None or (
                    isinstance(raw_t_max, Integral)
                    and not isinstance(raw_t_max, bool)
                ),
                f"{prefix}_noisified_max_timesteps must be an integer or None.",
            )
            t_max = 0 if raw_t_max is None else raw_t_max
            t_max = network.timesteps if t_max == -1 else t_max
            require(
                (t_min == 0 and t_max == 0) or
                t_min < t_max <= network.timesteps,
                f"{prefix}_noisified_max_timesteps must be 0 for clean-only "
                f"inputs or in ({t_min}, {network.timesteps}].",
            )

        require(local_vars["test_network_name"] in get_args(NetworkName), \
            f"test_network_name must be one of {get_args(NetworkName)}.")

        require(isinstance(local_vars["ema_decay"], Real) and \
            not isinstance(local_vars["ema_decay"], bool) and \
            np.isfinite(local_vars["ema_decay"]) and \
            0. <= local_vars["ema_decay"] < 1., \
            "ema_decay must be in the range of [0., 1.).")

        require(isinstance(local_vars["test_steps"], Integral) and \
            not isinstance(local_vars["test_steps"], bool) and \
            2 <= local_vars["test_steps"] <= network.timesteps, \
            "steps must be in the range of [2, timesteps].")

        require(isinstance(local_vars["test_eta"], Real) and \
            not isinstance(local_vars["test_eta"], bool) and \
            np.isfinite(local_vars["test_eta"]) and \
            0. <= local_vars["test_eta"] <= 1., \
            "eta must be in the range of [0., 1.].")

        require(isinstance(local_vars["p_uncond"], Real) and \
            not isinstance(local_vars["p_uncond"], bool) and \
            np.isfinite(local_vars["p_uncond"]) and \
            0. <= local_vars["p_uncond"] <= 1., \
            "p_uncond must be in the range of [0., 1.].")

        for name in ("train_cfg_scale", "test_cfg_scale"):
            value = local_vars[name]
            require(value is None or (
                isinstance(value, Real)
                and not isinstance(value, bool)
                and np.isfinite(value)
            ), f"{name} must be None or a finite number.")

        for name in (
            "noise_loss_coef", "image_loss_coef", 
            "kl_loss_coef", "ctr_loss_coef", "noise_distil_loss_coef"
        ):
            value = local_vars[name]
            require(isinstance(value, Real) and \
                not isinstance(value, bool) and np.isfinite(value) and value >= 0., \
                f"{name} must be a finite nonnegative number.")

        if local_vars["noise_distil_loss_coef"] > 0.:
            require(
                local_vars["teacher_network"] is not None
                or local_vars["defer_teacher"],
                "teacher_network is required when noise_distil_loss_coef is "
                "positive unless defer_teacher=True.",
            )
            require(
                not local_vars["swap_noise_image"],
                "noise distillation requires epsilon prediction, not "
                "swap_noise_image mode.",
            )

        if local_vars["swap_noise_image"]:
            require(
                local_vars["image_loss_coef"] == 0.,
                "swap_noise_image already optimizes clean-image prediction; "
                "image_loss_coef must be zero to avoid double weighting it.",
            )

        require(local_vars["kl_train_type"] in get_args(TrainType), \
            f"kl_train_type can be one of {TrainType}.")

        require(local_vars["ctr_train_type"] in get_args(TrainType), \
            f"ctr_train_type can be one of {TrainType}.")

        # Require CFG and a training scale for unconditional auxiliary losses.
        if local_vars["kl_train_type"] == "uncond" or \
        local_vars["ctr_train_type"] == "uncond":
            require(
                local_vars["network"].use_cfg and
                local_vars["train_cfg_scale"] is not None, 
                "Unconditional auxiliary losses require "
                "CFG and a non-None train_cfg_scale."
            )

        # Normalize persisted continual state before constructor serialization.
        require(
            isinstance(local_vars["seen_classes"], dict), 
            "seen_classes must be a mapping."
        )

        # Validate restoration width only when saved continual state is present.
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

        Args:
            stages_num (int): Number of intervals, in ``1..timesteps``.
            clustering_type (ClusteringType): ``"uniform"`` spaces integer
                timesteps approximately evenly; ``"log_snr"`` projects evenly
                spaced log-SNR targets to the closest schedule indices.

        Returns:
            list[int]: ``stages_num + 1`` strictly increasing boundaries with
            endpoints 0 and ``timesteps``.

        Raises:
            AssertionError: If ``stages_num`` is outside the valid range.
            ValueError: If the clustering name is unsupported or a strict
                partition cannot be constructed.
        """

        require(1 <= stages_num <= self.timesteps, \
            f"num_stages must be in [1, {self.timesteps}] range, "\
            f"but got {stages_num}.")


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

        # Reject collapsed clusters that contain no discrete timesteps.
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
    ) -> None:
        """Register current network variables with an existing optimizer.

        Progressive depth growth creates trainable variables after compilation.
        This method adds them to the supplied optimizer without replacing the
        optimizer or losing its iterations and accumulated state. Omitting the
        arguments uses this wrapper's optimizer and all raw-network variables.

        Args:
            optimizer (tf.keras.optimizers.Optimizer | None): Optimizer that
                should know the current variable set, or
                ``None`` to use ``self.optimizer`` when it exists.
            variables (list[tf.Variable] | None): Variables to register, or
                ``None`` for every trainable
                variable in the raw diffusion network.

        Returns:
            ``None``. If no optimizer exists yet, the method has no effect.
        """

        optimizer = getattr(self, "optimizer", None) if optimizer is None else optimizer
        variables = self.network.trainable_variables if variables is None else variables

        # Skip registration when the wrapper has not been compiled yet.
        if optimizer is None:
            return

        # Register variables through the legacy TensorFlow 2.10 optimizer API.
        if hasattr(optimizer, "_create_all_weights"):
            optimizer._create_all_weights(variables)
        # Use the newer optimizer build API when available.
        elif hasattr(optimizer, "build"):
            optimizer.build(variables)
        # Fail clearly when the optimizer exposes no variable-registration API.
        else:
            raise ValueError(
                "Failed to register new variables to the optimizer."
            )

    def _refresh_loss_flags(self) -> None:
        """Refresh auxiliary-loss flags from the current network topology.

        KL loss is available when a KL reshaper is configured, and class-token
        regularization is available when at least one regularizer depth exists.
        The flags are recomputed after progressive depth additions because a
        newly appended layer may enable either loss.

        Returns:
            ``None``. ``use_noise_distil_loss``, ``use_kl_loss`` and 
            ``use_ctr_loss`` are updated in place.
        """

        self.use_noise_distil_loss = bool(
            self.noise_distil_loss_coef > 0. and 
            self.teacher_network is not None
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
            dict[str, dict[str, int]]: Wrapped network's branch-wise
            before/added/after depth report.

        Raises:
            ValueError: If raw and EMA growth creates different numbers or
                shapes of weights.
        """

        raw_weight_ids = {id(weight) for weight in self.network.weights}
        ema_weight_ids = {id(weight) for weight in self.ema_network.weights} \
                        if self.ema_network is not None else set()

        growth = self.network.add_depths(depth_spec)
        self.network.build()

        # Mirror the raw network's structural growth in the EMA clone.
        if self.ema_network is not None:
            self.ema_network.add_depths(
                depth_spec
            )
            self.ema_network.build()

            raw_weights = [
                weight for weight in self.network.weights
                if id(weight) not in raw_weight_ids
            ]
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

    def _check_new_labels(
        self, 
        x: object | None = None, 
        y: object | None = None, 
        verbose: int | bool = True
    ) -> None:
        """Discover dataset labels and expand a dynamic network before fitting.

        Args:
            x (tf.data.Dataset | object | None): Keras inputs.  A dataset must
                yield ``(images, labels)`` batches.
            y (object | None): Separate Keras labels.  When supplied, these take
                precedence over labels contained in ``x``.
            verbose (int | bool): Whether to print newly discovered labels.

        Returns:
            None: New real labels are mapped to consecutive zero-based targets,
            the raw/EMA label vocabularies are expanded in place, and the
            wrapper initialization config sees the updated mapping.
        """

        # Preserve every legacy behavior for an explicitly sized network.
        if not self.network.dynamic_num_classes:
            return

        data = y if y is not None else x
        # Scan only labels, and refuse a dataset known to repeat forever.
        if isinstance(data, tf.data.Dataset):
            labels = set()
            for batch in data:
                labels.update(
                    np.unique(batch[1].numpy()).tolist()
                )

            data = list(labels)
        # Leave Keras to report missing inputs when no labels were supplied.
        elif y is None:
            return

        new_classes = []
        for label in np.unique(data):
            real_label = label.item()
            # Expand exactly once for every newly observed real label.
            if real_label not in self.seen_classes:
                wrapper_label = len(self.seen_classes)
                self.seen_classes[real_label] = wrapper_label
                new_classes.append(real_label)

                self.network.add_class()
                # Keep the EMA topology aligned and share only new parameters.
                if self.ema_network is not None:
                    self.ema_network.add_class(
                        source_network=self.network
                    )

        # Refresh symbolic outputs, optimizer variables, and cached traces once.
        if len(new_classes) > 0:
            # Report the labels added during this scan when requested.
            if verbose:
                print("Found new classes:", new_classes)

            self.network.build()
            # Refresh the EMA symbolic output after mirroring the growth.
            if self.ema_network is not None:
                self.ema_network.build()

            self._register_optimizer_variables()
            self.train_function = None
            self.test_function = None
            self.predict_function = None

    def _map_classes(self, classes: tf.Tensor) -> tf.Tensor:
        """Map real dataset labels to zero-based dynamic classifier targets.

        Args:
            classes (tf.Tensor): Integer dataset labels of arbitrary shape.

        Returns:
            tf.Tensor: Wrapper class IDs with the same shape and dtype.

        Raises:
            ValueError: If dynamic mode has not observed any class yet.
        """

        # Fixed-width networks retain the historical zero-based label contract.
        if not self.network.dynamic_num_classes:
            return classes

        # Dynamic evaluation cannot map labels before the first training scan.
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

        Raw classifier datasets may contain a third replay-provenance tensor,
        so arity greater than two alone cannot distinguish raw data from the
        seven-tensor diffusion representation. Phase-specific wrappers can
        override this method for their own prepared arity.

        Args:
            element_spec (object): ``tf.data.Dataset.element_spec`` value.

        Returns:
            bool: True for the base seven-or-more-tensor prepared contract.
        """

        return isinstance(element_spec, (tuple, list)) and len(element_spec) >= 7

    def _prepare_sampling_labels(
        self, 
        network: ArgumentSaverModel, 
        labels: tf.Tensor | Sequence[int]
    ) -> tf.Tensor:
        """Normalize and validate explicit network condition IDs.

        Args:
            network (ArgumentSaverModel): Selected raw or EMA network.
            labels (tf.Tensor | Sequence[int]): One condition ID per sample.

        Returns:
            tf.Tensor: Nonempty int32 vector whose IDs are valid for network.

        Raises:
            TypeError: If labels cannot form an integer tensor.
            ValueError: If labels are not a vector or contain an invalid ID.
        """

        try:
            labels = tf.convert_to_tensor(labels)
        except (TypeError, ValueError) as error:
            raise TypeError("labels must be an integer tensor or sequence.") \
                from error
        # Network condition IDs must use an integer, non-Boolean dtype.
        if not labels.dtype.is_integer or labels.dtype == tf.bool:
            raise TypeError("labels must have an integer dtype.")
        # Reject a statically known non-vector label structure early.
        if labels.shape.rank is not None and labels.shape.rank != 1:
            raise ValueError(
                "labels must be a one-dimensional tensor or sequence."
            )
        # Require at least one requested sample when static shape is available.
        if labels.shape.rank == 1 and labels.shape[0] == 0:
            raise ValueError("labels must contain at least one condition ID.")

        num_labels = getattr(network, "num_labels", None)
        # Sampling requires a finite, positive network condition vocabulary.
        if isinstance(num_labels, bool) or not isinstance(num_labels, Integral) \
        or num_labels < 1:
            raise ValueError(
                "The selected network must expose a positive num_labels."
            )

        static_labels = tf.get_static_value(labels)
        # Report statically visible out-of-vocabulary labels before graph tracing.
        if static_labels is not None and (
        np.any(static_labels < 0) or 
        np.any(static_labels >= num_labels)):
            raise ValueError(
                f"labels must contain network IDs in [0, {num_labels})."
            )

        label_assertions = (
            tf.debugging.assert_rank(
                labels, 1, 
                message="labels must be one-dimensional."
            ), 
            tf.debugging.assert_positive(
                tf.size(labels), 
                message="labels must not be empty."
            ),
            tf.debugging.assert_greater_equal(
                labels, 
                tf.cast(0, labels.dtype), 
                message="label IDs must be nonnegative."
            ),
            tf.debugging.assert_less(
                labels, 
                tf.cast(num_labels, labels.dtype), 
                message="label IDs exceed the selected network vocabulary."
            )
        )
        with tf.control_dependencies([
            assertion for assertion in label_assertions
            if assertion is not None
        ]):
            return tf.cast(tf.identity(labels), tf.int32)

    def _mask_unknown_teacher_labels(
        self, 
        labels: tf.Tensor
    ) -> tf.Tensor:
        """Replace condition IDs outside a past teacher vocabulary with null."""

        teacher_num_labels = getattr(
            self.teacher_network, 
            "num_labels", 
            None
        )

        if teacher_num_labels is None:
            return labels
        return tf.where(
            labels < tf.cast(teacher_num_labels, labels.dtype), 
            labels, 
            tf.zeros_like(labels)
        )

    @staticmethod
    def _normalize_seed(
        seed: int | None, 
        name: str = "seed"
    ) -> int | None:
        """Validate and normalize a TensorFlow/NumPy-compatible seed.

        Args:
            seed (int | None): Optional non-Boolean integral seed.
            name (str): Public argument name used in error messages.

        Returns:
            int | None: A plain Python integer in ``[0, 2**32)``, or None.

        Raises:
            TypeError: If the value is not an integer or None.
            ValueError: If the integer is outside the shared runtime range.
        """

        # Preserve None as the explicit request for advancing runtime randomness.
        if seed is None:
            return None

        # Reject Booleans and non-integral seeds before normalizing NumPy scalars.
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError(f"{name} must be a non-Boolean integer or None.")

        seed = int(seed)

        # Keep seeds inside the unsigned range shared by runtime seed helpers.
        if not 0 <= seed < 2 ** 32:
            raise ValueError(f"{name} must be in [0, 2**32).")

        return seed

    @property
    def current_timesteps_bounds(self) -> tuple[int, int]:
        """Return active forward-noising bounds as ``[minimum, maximum)``.

        Returns:
            tuple[int, int]: Inclusive minimum and exclusive maximum timestep.
        """

        return self._active_min_timestep, self._active_max_timestep

    @property
    def current_resolution(self) -> tuple[int, int]:
        """Return the square image resolution currently processed.

        Returns:
            tuple[int, int]: Active positive integer resolution of the wrapper
            and raw network, respectively.
        """

        return self._current_resolution, self.network.current_resolution

    @property
    def metrics(self) -> list[metrics.Metric]:
        """Return Keras metric trackers reset between fit/evaluate epochs.

        Returns:
            list[tf.keras.metrics.Metric]: Total, noise, split-noise, image, 
            KL, class-token regularizer loss trackers and regularizer
            accuracy tracker. They exist after :meth:`compile`.
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
        ]

    def compile(
        self, 
        loss: losses.Loss | str = "mse", 
        **kwargs: object
    ) -> None:
        """Configure the compiled prediction loss, optimizer, and trackers.

        Args:
            loss (tf.keras.losses.Loss | str): Per-example/base loss used for
                both noise and image reconstruction, default ``"mse"``.
            **kwargs (object): Arguments forwarded to ``tf.keras.Model.compile``.  Useful
                keys include ``optimizer`` (optimizer instance/name),
                ``run_eagerly`` (bool), ``steps_per_execution`` (int),
                ``jit_compile`` (bool where supported), ``metrics``,
                ``weighted_metrics``, and ``loss_weights``.  Custom train/test
                steps report the trackers defined here.

        Returns:
            None: Loss helpers and enabled metric trackers are initialized.
        """

        super().compile(loss=loss, **kwargs)

        self.scce_loss_fn = losses.sparse_categorical_crossentropy

        stable_dtype = self.dtype_policy.variable_dtype
        self.total_loss_tracker = metrics.Mean(
            name="loss", 
            dtype=stable_dtype
        )
        self.noise_loss_tracker = metrics.Mean(
            name="total_noise_loss" if self.show_separate_noise_losses \
                else "noise_loss", 
            dtype=stable_dtype
        )
        self.noise_distil_loss_tracker = metrics.Mean(
            name="noise_distil_loss", 
            dtype=stable_dtype
        )
        self.cond_noise_loss_tracker = metrics.Mean(
            name="cond_noise_loss", 
            dtype=stable_dtype
        )
        self.uncond_noise_loss_tracker = metrics.Mean(
            name="uncond_noise_loss", 
            dtype=stable_dtype
        )
        self.image_loss_tracker = metrics.Mean(
            name="image_loss", 
            dtype=stable_dtype
        )
        self.kl_loss_tracker = metrics.Mean(
            name="kl_loss", 
            dtype=stable_dtype
        )
        self.ctr_loss_tracker = metrics.Mean(
            name="ctr_loss", 
            dtype=stable_dtype
        )
        self.ctr_accuracy_tracker = metrics.SparseCategoricalAccuracy(
            name="ctr_accuracy", 
            dtype=stable_dtype
        )

    def fit(
        self, 
        x: object | None = None, 
        y: object | None = None, 
        **kwargs: object
    ) -> callbacks.History:
        """Fit under configured training timestep bounds, then restore bounds.

        Args:
            x (tf.data.Dataset | object | None): Keras input yielding
                ``(images, labels)``; images are float ``[B,H,W,C]`` (normally
                scaled to ``[-1,1]``) and labels are integer ``[B]``. When
                ``map_preprocess=True``, this must be a ``tf.data.Dataset`` and
                is mapped through :meth:`prep_inputs_map` before fitting.
            y (tf.data.Dataset | object | None): Optional separate Keras targets;
                custom steps normally consume labels from ``x`` instead.
            **kwargs (object): Forwarded to ``tf.keras.Model.fit``.  Accepted standard
                keys include ``batch_size``, ``epochs``, ``verbose``,
                ``callbacks``, ``validation_data``, ``shuffle``,
                ``steps_per_epoch``, ``validation_steps``, and
                ``initial_epoch``.

        Returns:
            tf.keras.callbacks.History: Keras training history.  Entry timestep
            bounds are restored even when Keras raises an exception.
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
                # Restrict mapped preprocessing to the dataset API it targets.
                if not isinstance(x, tf.data.Dataset):
                    raise TypeError(
                        "map_preprocess=True requires x to be a tf.data.Dataset."
                    )

                self._preprocess_training = True
                x = x.map(
                    self.prep_inputs_map, 
                    num_parallel_calls=self.map_num_parallel_calls
                )

                validation_data = kwargs.get("validation_data")
                # Apply the equivalent clean-input preparation to validation.
                if validation_data is not None:
                    # Require validation to use the same dataset input contract.
                    if not isinstance(validation_data, tf.data.Dataset):
                        raise TypeError(
                            "map_preprocess=True requires validation_data "
                            "to be a tf.data.Dataset."
                        )

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
        network_name: NetworkName = "ema", 
        **kwargs: object
    ) -> float | list[float] | dict[str, float]:
        """Evaluate the raw or EMA network under test timestep bounds.

        Args:
            x (tf.data.Dataset | object | None): Keras input yielding image and
                label tensors. When ``map_preprocess=True``, this must be a
                ``tf.data.Dataset`` and is mapped through
                :meth:`prep_inputs_map` before evaluation.
            y (tf.data.Dataset | object | None): Optional separate targets.
            network_name (NetworkName): ``"ema"`` or ``"raw"`` for this call.
                With ``use_ema=False``, ``"ema"`` resolves to the raw network.
            **kwargs (object): Forwarded to ``tf.keras.Model.evaluate``.  Standard keys
                include ``batch_size``, ``verbose``, ``sample_weight``, ``steps``,
                ``callbacks``, and ``return_dict``.

        Returns:
            float | list[float] | dict[str, float]: Standard Keras evaluation
            result.  Active timestep bounds and the previously selected test
            network are restored even when Keras raises an exception.
        """

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
                # Restrict mapped evaluation to TensorFlow datasets.
                if not isinstance(x, tf.data.Dataset):
                    raise TypeError(
                        "map_preprocess=True requires x to be a tf.data.Dataset."
                    )

                # Keras calls this override for validation inside ``fit``. Its
                # validation dataset was already prepared above, so do not map
                # the resulting seven/eight-tensor element a second time.
                element_spec = x.element_spec
                already_prepared = self._is_prepared_dataset_spec(
                    element_spec
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

    def summary(self, **kwargs: object) -> None:
        """Print/return the raw network's Keras model summary.

        Args:
            **kwargs (object): Forwarded to ``network.summary``; supported keys include
                ``line_length``, ``positions``, ``print_fn``, ``expand_nested``,
                and ``show_trainable`` (TensorFlow-version dependent).

        Returns:
            None: Keras summary output is written through ``print_fn``.
        """

        return self.network.summary(**kwargs)

    def train_step(
        self, 
        inputs: tuple[tf.Tensor, ...]
    ) -> dict[str, tf.Tensor]:
        """Perform one joint diffusion optimization step on the raw network.

        Args:
            inputs (tuple[tf.Tensor, ...]): Clean images and integer classes, or
                seven prepared tensors plus the optional noise-teacher
                prediction and mask when ``map_preprocess=True``.

        Returns:
            dict[str, tf.Tensor]: Running enabled loss/accuracy metrics.  Noise
            loss is always present; total/image/KL/regularizer values appear
            according to active loss flags.
        """

        if self.map_preprocess and len(inputs) == 2:
            self._preprocess_training = True
            prepared_inputs = self.prep_inputs_map(*inputs)
            self._preprocess_training = None
        else:
            prepared_inputs = self.prep_inputs(
                inputs
            ) if not self.map_preprocess else inputs
        if self.use_noise_distil_loss:
            teacher_noises_pred = prepared_inputs[-2]
            teacher_noise_mask = prepared_inputs[-1]
            prepared_inputs = prepared_inputs[:-2]
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

        Args:
            inputs (tuple[tf.Tensor, ...]): Clean images and integer classes, or
                seven prepared tensors plus the optional noise-teacher
                prediction and mask when ``map_preprocess=True``.

        Returns:
            dict[str, tf.Tensor]: Running evaluation metrics.  Image loss is
            explicitly evaluated even when its training coefficient is zero.
        """

        if self.map_preprocess and len(inputs) == 2:
            self._preprocess_training = False
            prepared_inputs = self.prep_inputs_map(*inputs)
            self._preprocess_training = None
        else:
            prepared_inputs = self.prep_inputs(
                inputs, 
                use_label_dropout=False
            ) if not self.map_preprocess else inputs
        if self.use_noise_distil_loss:
            teacher_noises_pred = prepared_inputs[-2]
            teacher_noise_mask = prepared_inputs[-1]
            prepared_inputs = prepared_inputs[:-2]
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
        stage. Timestep ranges, resolutions, and model depth can therefore be
        changed separately or together without one strategy taking priority.
        Timestep and resolution updates are applied before their stage. A
        depth update is appended after its stage has completed successfully,
        so the new layers start training in the next stage. This makes the
        last depth update train during ``final_epochs`` when it is nonzero.

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

        Examples:

            fit_progressively("timesteps_only", stages_num=4, x=dataset)
            fit_progressively("resolutions_only", stages_num=3, x=dataset)
            fit_progressively(
                "resolutions_only", resolutions=[16, 32, 64], x=dataset
            )
            fit_progressively(
                "depths_only", 
                depths=[
                    "vision_transformer_block",
                    {"local_mixer", "vision_transformer_block"}
                ],
                final_epochs=1, x=dataset
            )

        Supported stage elements are:

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
                "depth": depth_specification, 
            }

        A string or set names changes without providing their values. Their
        values are read from ``timestep_boundaries[stage_index]`` and
        ``resolutions[stage_index]`` or ``depths[stage_index]`` respectively.
        A dictionary value of ``None`` has the same meaning. Inline tuple or
        dictionary values take precedence over the companion sequences.

        A depth specification names layers supported by the transformer's
        normal layer factories. A string adds one depth containing that layer;
        a list adds several depths; and a set or dictionary puts several layer
        types in one depth. For example::

            depth_specification = [
                "vision_transformer_block",
                {
                    "feature_connector": {"ids": [-1]},
                    "local_mixer": True,
                },
            ]

        Exact supported names are ``feature_connector``,
        ``cross_attention_connector``, ``vision_transformer_block``,
        ``local_mixer``, ``downsampler``, ``upsampler``, ``reshaper``, and
        ``cls_token_regularizer``. The existing model-wide layer kwargs are
        reused. Connector dictionaries may provide ``ids`` (an integer or ID
        iterable); transformer block dictionaries may provide ``use_decoder``
        (bool) and ``mlp_output_dim`` (int or None); a reshaper value is
        ``"flatten"`` or ``"unflatten"``. Added sequences must leave the final feature shape
        compatible with the already-trained output head. This
        API deliberately accepts the project's supported layer types rather
        than arbitrary Keras layers, because each supported type has a defined
        call signature and location in the transformer depth. New layers are
        built in both raw and EMA networks, their initial weights are copied
        into EMA, and their variables are registered with the active optimizer
        before the following training stage.

        For example:

            stage_tasks = [
                {"timesteps": (700, 1000), "resolution": 16}, 
                "timesteps", 
                ("resolution", 32), 
                {
                    "timesteps", "resolution", "depth"
                }, 
            ]
            timestep_boundaries = [None, (300, 1000), None, (0, 1000)]
            resolutions = [None, None, None, 64]
            depths = [None, None, None, "vision_transformer_block"]

        This produces stages ``(700: 1000, 16)``, ``(300: 1000, 16)``,
        ``(300: 1000, 32)``, and ``(0: 1000, 64)``, then appends a transformer
        block. No direction, native-size ceiling, or implicit priority between
        the strategies is imposed.

        Args:
            stage_tasks (Sequence[str | tuple | set | dict] | Literal[
                "timesteps_only", "resolutions_only", "depths_only"]): A list
                of ordered stage descriptions, or
                ``"timesteps_only"``, ``"resolutions_only"``, or
                ``"depths_only"``. A list's length is the number of training
                stages. Strings and two-item tuples change one value; sets and
                dictionaries may combine all three progressive operations.
            stages_num (int | None): Optional number of generated stages. For an explicit
                mixed task list, its length determines the stage count. In
                either ``*_only`` mode, supplied values determine the count.
                ``stages_num`` is therefore needed only when values must be
                generated.
            stages_verbose (bool): Whether to print each stage's resolved state.
            stage_epochs (int): Number of epochs allocated to every listed stage.
                With plateau pacing, this is the maximum for each stage.
            final_epochs (int | None): Epochs for a final full-timestep, native-resolution
                stage. ``None`` uses ``stage_epochs`` and ``0`` disables it.
            timestep_boundaries (Sequence[tuple[int, int] | None] | None): Optional stage-indexed sequence of
                ``(lower_bound, upper_bound)`` pairs. An entry is read only when
                the corresponding task requests ``"timesteps"`` without an
                inline pair, so unused positions may be ``None``. When omitted,
                cumulative easy-to-hard ranges are generated from ``stages_num``.
            timestep_clustering_type (ClusteringType): It is only used when the method automatically
                generates timestep boundaries, and it can be one of ('uniform', 'log_snr').
            resolutions (Sequence[int | None] | None): Optional stage-indexed resolution values. An entry is
                read only when the corresponding task requests ``"resolution"``
                without an inline value, so unused positions may be ``None``.
                Values may increase, decrease, repeat, or exceed ``image_size``;
                the network's normal resolution requirements still apply. When
                omitted, ``stages_num`` low-to-high resolutions are generated by
                repeatedly dividing ``image_size`` by powers of two.
            depths (Sequence[object | None] | None): Optional stage-indexed depth specifications. An entry is
                read only when the corresponding task requests ``"depth"``
                without an inline value. A specification may add any number of
                supported layer dictionaries to ``network.layers_dicts``.
                Appended depths persist after this method returns.
            pacing_type (Literal["fixed", "plateau"]): ``"fixed"`` always runs ``stage_epochs``. ``"plateau"``
                may advance sooner using the selected early-stopping callback.
            earlystopping_type (Literal["batch_wise", "epoch_wise"]): Under plateau pacing, ``"epoch_wise"`` uses
                Keras ``EarlyStopping`` and ``"batch_wise"`` uses
                ``BatchLossPlateau``.
            monitor (str): Metric name monitored by plateau pacing.
            patience (int): Number of non-improving epochs or batches tolerated by
                the selected early-stopping callback.
            min_delta (float): Minimum monitored improvement.
            stopper_mode (str): Keras early-stopping mode used by epoch-wise pacing.
            **fit_kwargs (object): Normal Keras ``fit`` arguments such as ``x``,
                ``validation_data``, ``callbacks``, ``steps_per_epoch`` and
                ``verbose``. ``epochs`` and ``initial_epoch`` are managed here.

        Returns:
            tf.keras.callbacks.History: Merged metrics and a
            ``progressive_stages`` record of every resolved stage, including
            its pre-addition network depth and any ``depth_growth`` result. The
            model's timestep bounds and resolution are restored to their entry 
            values after completion or interruption; completed structural depth
            additions are intentionally retained. Input data must be reiterable
            because each stage invokes a separate Keras ``fit`` call.
        """

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
                "depths_only": "depth", 
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

            Returns:
                dict[str, object]: Stage record containing active bounds,
                resolution, pre-growth network depth, epoch count, and its raw
                Keras history dictionary.
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
                        # mode=stopper_mode
                    ))

            # Print the resolved stage state when progress output is requested.
            if stages_verbose:
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
                # Require a dataset before installing the mapping operation.
                if not isinstance(stage_x, tf.data.Dataset):
                    raise TypeError(
                        "map_preprocess=True requires x to be a tf.data.Dataset."
                    )

                self._preprocess_training = True
                stage_fit_kwargs["x"] = stage_x.map(
                    self.prep_inputs_map, 
                    num_parallel_calls=self.map_num_parallel_calls
                )

                validation_data = stage_fit_kwargs.get("validation_data")
                # Prepare an optional stage validation dataset consistently.
                if validation_data is not None:
                    # Require validation to satisfy the mapped dataset contract.
                    if not isinstance(validation_data, tf.data.Dataset):
                        raise TypeError(
                            "map_preprocess=True requires validation_data "
                            "to be a tf.data.Dataset."
                        )

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

                # Resolve and apply this stage's timestep bounds.
                if "timesteps" in updates:
                    bounds = updates["timesteps"]
                    bounds = timestep_boundaries[stage_index] if bounds is None else bounds
                    bounds = tuple(bounds)
                    self.set_timestep_bounds(*bounds)
                    updates["timesteps"] = (self._active_min_timestep, self._active_max_timestep)

                # Resolve and apply this stage's input resolution.
                if "resolution" in updates:
                    resolution = updates["resolution"]
                    resolution = resolutions[stage_index] if resolution is None else resolution
                    resolution = int(resolution)
                    self.set_current_resolution(resolution)
                    updates["resolution"] = resolution

                # Resolve the depth specification for post-stage growth.
                if "depth" in updates:
                    depth_spec = updates["depth"]
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
                        "resolution": self._current_resolution, 
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

    def set_timestep_bounds(
        self, 
        min_timesteps: int | None = 0, 
        max_timesteps: int | None = -1
    ) -> None:
        """Set the active half-open timestep interval for forward noising.

        Args:
            min_timesteps (int | None): Inclusive lower bound; ``None`` uses 0.
            max_timesteps (int | None): Exclusive upper bound; ``None`` uses 0
                and -1 means the current full schedule length.

        Returns:
            None: Changed bounds invalidate cached Keras train/test/predict
            functions so traced random ranges are rebuilt.

        Raises:
            AssertionError: Unless bounds are clean-only ``[0,0)`` or satisfy
                ``0 <= min < max <= timesteps``.
        """

        min_timesteps = 0 if min_timesteps is None else min_timesteps
        max_timesteps = 0 if max_timesteps is None else max_timesteps
        max_timesteps = self.timesteps if max_timesteps == -1 else max_timesteps

        require(
            isinstance(min_timesteps, Integral) and 
            not isinstance(min_timesteps, bool), 
            "min_timesteps must be an integer."
        )
        require(
            isinstance(max_timesteps, Integral) and 
            not isinstance(max_timesteps, bool), 
            "max_timesteps must be an integer."
        )
        min_timesteps = int(min_timesteps)
        max_timesteps = int(max_timesteps)
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
        """Synchronize active resolution across wrapper, raw, and EMA networks.

        Args:
            resolution (int | None): Square size accepted by the raw network;
                ``None`` restores constructor ``image_size``.

        Returns:
            None: Changed values invalidate cached Keras execution functions.

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
        if self.teacher_network is not None and hasattr(
            self.teacher_network, "set_current_resolution"
        ):
            self.teacher_network.set_current_resolution(resolution)

        resolution = int(resolution)
        # Propagate a changed resolution to the raw and EMA networks.
        if getattr(self, "_current_resolution", None) != resolution:
            self._current_resolution = resolution

            self.train_function = None
            self.test_function = None
            self.predict_function = None

    def set_teacher_network(
        self, 
        teacher_network: tf.keras.Model | None
    ) -> None:
        """Attach or clear the independent frozen runtime teacher.

        Noise teachers created by :meth:`snapshot_teacher_network` retain the
        student's schedule metadata. When a wrapper is supplied directly, its
        schedule and timestep-zero convention must also match. A bare external
        raw network has no schedule metadata, so its caller remains responsible
        for ensuring that it was trained with the same forward process.
        """

        if teacher_network is not None and self.noise_distil_loss_coef > 0. \
        and getattr(
            teacher_network, 
            "swap_noise_image", 
            getattr(teacher_network, "_diffusion_swap_noise_image", False)
        ):
            raise ValueError(
                "An x0-prediction wrapper cannot teach epsilon distillation."
            )

        teacher_schedule = None
        teacher_modify_first = None
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

        if teacher_network is not None and getattr(
            teacher_network, "network", None
        ) is not None:
            raw_teacher = teacher_network.network
            raw_teacher._diffusion_scheduler_name = teacher_schedule
            raw_teacher._diffusion_modify_first_t = teacher_modify_first
            raw_teacher._diffusion_swap_noise_image = getattr(
                teacher_network, 
                "swap_noise_image", 
                getattr(raw_teacher, "_diffusion_swap_noise_image", False)
            )
            teacher_network = raw_teacher

        if teacher_network is not None:
            validate_model_dtype_policy(
                teacher_network, 
                self.dtype_policy, 
                role="teacher_network"
            )
        if teacher_network is not None and (
            teacher_network is self.network
            or teacher_network is self.ema_network
        ):
            raise ValueError(
                "teacher_network must be an independent frozen snapshot."
            )

        needs_noise_teacher = bool(self.noise_distil_loss_coef > 0.)
        if teacher_network is None and needs_noise_teacher \
        and not self.defer_teacher:
            raise ValueError(
                "Noise distillation requires teacher_network; "
                "set defer_teacher=True only when it will be attached later."
            )
        if teacher_network is not None and needs_noise_teacher:
            if teacher_schedule is not None \
            and teacher_schedule != self.scheduler_name:
                raise ValueError(
                    "teacher_network scheduler_name must match the student."
                )
            if teacher_modify_first is not None \
            and teacher_modify_first != self.modify_first_t:
                raise ValueError(
                    "teacher_network modify_first_t must match the student."
                )
            for name in ("timesteps", "channels", "use_cfg"):
                if getattr(teacher_network, name, None) != getattr(
                    self.network, name, None
                ):
                    raise ValueError(
                        f"teacher_network {name} must match the student."
                    )

        object.__setattr__(self, "teacher_network", teacher_network)
        if self.teacher_network is not None:
            self.teacher_network.trainable = False
            if hasattr(self.teacher_network, "set_current_resolution"):
                self.teacher_network.set_current_resolution(
                    self._current_resolution
                )

        DiffusionModel._refresh_loss_flags(self)

        self.map_preprocess = True if self.use_noise_distil_loss \
                            else self._map_preprocess_without_teacher
        self.train_function = None
        self.test_function = None
        self.predict_function = None

    def snapshot_teacher_network(
        self, 
        network_name: NetworkName = "raw"
    ) -> tf.keras.Model:
        """Clone a raw or EMA branch as an independent frozen teacher."""

        source_network = self.get_network(network_name)
        teacher_network = source_network.__class__.from_config(
            source_network.get_config()
        )
        teacher_network.build()

        if hasattr(teacher_network, "set_current_resolution"):
            teacher_network.set_current_resolution(
                self._current_resolution
            )

        copy_network_weights_by_layer(source_network, teacher_network)

        teacher_network._diffusion_scheduler_name = self.scheduler_name
        teacher_network._diffusion_modify_first_t = self.modify_first_t
        teacher_network._diffusion_swap_noise_image = self.swap_noise_image
        teacher_network.trainable = False

        return teacher_network

    def load_schedules(
        self, 
        scheduler_name: SchedulerName | None = None, 
        timesteps: int | None = None
    ) -> None:
        """Generate and store TensorFlow tensors for a noise schedule.

        Args:
            scheduler_name (SchedulerName | None): Supported name listed in the
                constructor docs; ``None`` reuses ``self.scheduler_name``.
            timesteps (int | None): Schedule length; None reuses the current
                length.  It must equal ``network.timesteps`` so schedule indices
                cannot exceed or disagree with the network embedding table.

        Returns:
            None: ``self.schedules`` maps schedule-statistic names to rank-1
            tensors in the policy variable dtype and updates schedule metadata.
        """

        scheduler_name = self.scheduler_name if scheduler_name is None \
                        else scheduler_name
        timesteps = self.timesteps if timesteps is None else int(timesteps)

        # Keep schedule indexing compatible with the network's timestep embedding.
        if timesteps != self.network.timesteps:
            raise ValueError(
                "Schedule timesteps must equal network.timesteps; rebuilding "
                "the schedule does not resize network timestep embeddings."
            )

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

    def get_noise_and_signal_rates(
        self, 
        t: int | tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Gather signal and noise amplitudes at one or more timesteps.

        Args:
            t (int | tf.Tensor): Scalar or integer tensor of schedule indices.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: ``(sqrt_alpha_bar,
            sqrt_one_minus_alpha_bar)`` with the same index shape as ``t`` and
            the policy variable dtype.
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

        Args:
            x0 (tf.Tensor): Clean float images ``[B,H,W,C]``.
            t (tf.Tensor): Integer timestep IDs ``[B]``.
            noises (tf.Tensor): Standard-normal samples matching ``x0``.

        Returns:
            tf.Tensor: Noisy images ``x_t = sqrt(alpha_bar_t)*x0 +
            sqrt(1-alpha_bar_t)*noise`` with the same shape/dtype as ``x0``.

        Raises:
            TypeError: If ``x0`` does not have a floating dtype.
        """

        # Reject integer images because forward diffusion requires real arithmetic.
        if not x0.dtype.is_floating:
            raise TypeError("x0 must have a floating dtype.")

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        a, b = self.get_noise_and_signal_rates(t)
        a = tf.reshape(a, (-1, 1, 1, 1))
        b = tf.reshape(b, (-1, 1, 1, 1))
        stable_x0 = tf.cast(x0, stable_dtype)
        stable_noises = tf.cast(noises, stable_dtype)
        noisy = a * stable_x0 + b * stable_noises

        return tf.cast(noisy, x0.dtype)

    def noisify(
        self, 
        x0: tf.Tensor, 
        t: tf.Tensor | None = None, 
        min_timesteps: int | None = None, 
        max_timesteps: int | None = None, 
        seed: int | None = None
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Draw timesteps/noise and create a noisy image batch.

        Args:
            x0 (tf.Tensor): Clean float images ``[B,H,W,C]``.
            t (tf.Tensor | None): Explicit integer IDs ``[B]``.  ``None`` draws
                uniformly from ``[min_timesteps, max_timesteps)``.
            min_timesteps (int | None): Draw lower bound; ``None`` uses the
                active wrapper bound.
            max_timesteps (int | None): Exclusive draw upper bound; ``None``
                uses the active wrapper bound.
            seed (int | None): Random seed; ``None`` uses ``self.seed``.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Noisy images ``x_t``, sampled
            standard-normal noise, and int32 timesteps, with image tensors
            matching ``x0`` shape and floating dtype.

        Raises:
            TypeError: If ``x0`` does not have a floating dtype.
        """

        # Generate noise only for floating-point clean images.
        if not x0.dtype.is_floating:
            raise TypeError("x0 must have a floating dtype.")

        min_timesteps = self._active_min_timestep if min_timesteps is None else min_timesteps
        max_timesteps = self._active_max_timestep if max_timesteps is None else max_timesteps
        seed = self._normalize_seed(
            self.seed if seed is None else seed, 
            "noisify seed"
        )

        x_shape = tf.shape(x0)

        # Draw timesteps only when the caller did not supply explicit IDs.
        if t is None:
            # Reject invalid lower bounds before TensorFlow reports an RNG error.
            if isinstance(min_timesteps, bool) or not isinstance(
                min_timesteps, Integral
            ):
                raise TypeError("min_timesteps must be an integer.")
            # Reject invalid upper bounds before TensorFlow reports an RNG error.
            if isinstance(max_timesteps, bool) or not isinstance(
                max_timesteps, Integral
            ):
                raise TypeError("max_timesteps must be an integer.")

            min_timesteps = int(min_timesteps)
            max_timesteps = int(max_timesteps)

            if min_timesteps == 0 and max_timesteps == 0:
                return (
                    x0, 
                    tf.zeros_like(x0), 
                    tf.zeros((x_shape[0],), dtype=tf.int32)
                )

            # Require a nonempty half-open range inside the diffusion horizon.
            if not 0 <= min_timesteps < max_timesteps <= self.timesteps:
                raise ValueError(
                    "Expected 0 <= min_timesteps < max_timesteps <= "
                    f"timesteps, got [{min_timesteps}, {max_timesteps}) "
                    f"with T={self.timesteps}."
                )

            t = tf.random.uniform(
                (x_shape[0],), 
                minval=min_timesteps, 
                maxval=max_timesteps, 
                dtype=tf.int32, 
                seed=seed
            )
        # Validate caller-supplied timestep IDs before forward noising.
        else:
            t = tf.convert_to_tensor(t)

            # Timesteps are discrete schedule indices, never floating or Boolean.
            if not t.dtype.is_integer or t.dtype == tf.bool:
                raise TypeError("t must have an integer dtype.")
            # Reject a statically known non-vector timestep structure early.
            if t.shape.rank is not None and t.shape.rank != 1:
                raise ValueError("t must be a one-dimensional tensor.")
            timestep_assertions = (
                tf.debugging.assert_rank(
                    t, 1, 
                    message="t must be one-dimensional."
                ), 
                tf.debugging.assert_equal(
                    tf.shape(t)[0], 
                    x_shape[0], 
                    message="t batch size must match x0."
                ), 
                tf.debugging.assert_greater_equal(
                    t, 
                    tf.cast(0, t.dtype), 
                    message="timestep IDs must be nonnegative."
                ), 
                tf.debugging.assert_less(
                    t, 
                    tf.cast(self.timesteps, t.dtype), 
                    message="timestep IDs must be less than timesteps."
                )
            )
            with tf.control_dependencies([
                assertion for assertion in timestep_assertions
                if assertion is not None
            ]):
                t = tf.cast(tf.identity(t), tf.int32)

        noises = tf.random.normal(
            x_shape, 
            mean=0., 
            stddev=1., 
            dtype=x0.dtype,
            seed=seed, 
            name="noises"
        )
        x_t = self.q_sample(x0, t, noises)

        return x_t, noises, t

    def postprocess(self, x: tf.Tensor) -> tf.Tensor:
        """Convert model-space images from nominal ``[-1,1]`` to ``[0,1]``.

        Args:
            x (tf.Tensor): Numeric tensor of any shape.

        Returns:
            tf.Tensor: ``(x + 1) / 2`` clipped elementwise to ``[0,1]``.
        """

        x = (x + 1) / 2
        x = tf.clip_by_value(x, 0., 1.)

        return x

    def get_network(
        self, 
        network_name: NetworkName | Literal["teacher"]
    ) -> ArgumentSaverModel:
        """Resolve the raw, EMA, or internal teacher prediction network.

        Args:
            network_name (NetworkName | Literal["teacher"]): Selected branch.

        Returns:
            DiffusionTransformer: Selected network instance.
        """

        if network_name == "teacher":
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

        selected_ids = None if variables is None else {
            id(variable) for variable in variables
        }
        selected_scopes = set() if variables is None else {
            variable.name.rsplit("/", 1)[0] for variable in variables
        }

        for w, ew in zip(self.network.weights, self.ema_network.weights):
            selected = selected_ids is None or \
                id(w) in selected_ids or (
                    not w.trainable and 
                    w.name.rsplit("/", 1)[0] in selected_scopes
                )
            if selected:
                ew.assign(
                    ew * self.ema_decay + w * (1 - self.ema_decay)
                )

        return True

    def apply_grads(
        self, 
        tape: tf.GradientTape, 
        loss: tf.Tensor, 
        variables: list[tf.Variable]| None = None
    ) -> None:
        """Differentiate a scalar loss and apply gradients with the optimizer.

        Args:
            tape (tf.GradientTape): Tape that recorded ``loss`` computation.
            loss (tf.Tensor): Scalar differentiable objective.
            variables (list[tf.Variable] | None): Variables to update; ``None``
                selects all raw-network trainable variables.

        Returns:
            None: Optimizer slots, iterations, and variables are updated.
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
        """Apply classifier-free label dropout.

        Args:
            labels (tf.Tensor): Shifted integer labels ``[B]`` where ID 0 is
                reserved for the null condition.
            seed (int | None): Random seed; ``None`` uses ``self.seed``.

        Returns:
            tf.Tensor: Same shape/dtype as ``labels``; each element becomes 0
            independently with probability ``p_uncond``.
        """

        seed = self.seed if seed is None else seed

        mask = tf.random.uniform(
            (tf.shape(labels)[0],), 
            seed=seed
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

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Clean images ``[B,H,W,C]`` in
                model space and zero-based integer classes ``[B]``.
            use_label_dropout (bool): Apply CFG dropout to shifted labels.
            seed (int | None): Seed forwarded to noising and label dropout.

        Returns:
            tuple: ``(x0, noises, t, x_t, cfg_labels, uncond_labels, classes)``.
            Images are resized to the active resolution when necessary;
            ``cfg_labels`` are real labels shifted by one under CFG and possibly
            replaced by 0; ``uncond_labels`` are all 0.  With
            ``swap_noise_image=True``, the prediction target is clean ``x0``.
        """

        x0, labels = inputs

        x0 = tf.image.resize(x0, 
            size=(
                self._current_resolution, 
                self._current_resolution
            ), 
            method=self.resize_method, 
            antialias=self.resize_antialias
        ) if self._current_resolution != self.image_size else x0

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

        Args:
            x0 (tf.Tensor): Clean image batch ``[B,H,W,C]``.
            labels (tf.Tensor): Integer dataset labels ``[B]``.

        Returns:
            tuple[tf.Tensor, ...]: The seven tensors returned by
            :meth:`prep_inputs`, followed by the noise-teacher prediction and
            mask when noise distillation is active.
        """

        outputs = self.prep_inputs(
            (x0, labels), 
            use_label_dropout=self._preprocess_training is not False
        )

        if not self.use_noise_distil_loss:
            return outputs

        x_t = outputs[3]
        t = outputs[2]
        cond_labels = outputs[4]
        uncond_labels = outputs[5]
        cfg_scale = self.train_cfg_scale if self._preprocess_training \
                    else self.test_cfg_scale

        teacher_noise_mask = tf.ones_like(
            cond_labels, 
            dtype=tf.bool
        )
        teacher_num_labels = getattr(
            self.teacher_network, 
            "num_labels", 
            None
        )
        if teacher_num_labels is not None:
            teacher_noise_mask = cond_labels < tf.cast(
                teacher_num_labels, 
                cond_labels.dtype
            )

        teacher_cond_labels = self._mask_unknown_teacher_labels(
            cond_labels
        )

        _, teacher_noises_pred, *_ = self.forward(
            "teacher", x_t, t, t, 
            cond_labels=teacher_cond_labels, 
            uncond_labels=uncond_labels, 
            scale=cfg_scale, 
            training=False
        )
        teacher_noises_pred = tf.stop_gradient(
            teacher_noises_pred
        )

        return *outputs, teacher_noises_pred, teacher_noise_mask

    def compute_separate_noise_losses(
        self, 
        noises: tf.Tensor, 
        noises_pred: tf.Tensor, 
        cond_labels: tf.Tensor | None
    ) -> tuple[tf.Tensor | None, tf.Tensor | None]:
        """Compute reporting-only noise losses for conditional/null rows.

        Args:
            noises (tf.Tensor): Noise targets shaped like the model output.
            noises_pred (tf.Tensor): Predicted noise shaped like ``noises``.
            cond_labels (tf.Tensor | None): Post-dropout condition IDs. Null ID
                zero marks unconditional rows when CFG is enabled.

        Returns:
            tuple[tf.Tensor | None, tf.Tensor | None]: Conditional and
            unconditional scalar losses, or two ``None`` values when split
            reporting is disabled. An empty side has a finite zero loss; its
            metric receives zero sample weight in :meth:`get_results_dict`.
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
        noises_pred = tf.stop_gradient(noises_pred)

        cond_has_rows = tf.reduce_any(cond_mask)
        cond_noise_loss = self.compiled_loss(
            tf.boolean_mask(noises, cond_mask), 
            tf.boolean_mask(noises_pred, cond_mask)
        )
        cond_noise_loss = tf.where(
            cond_has_rows, 
            cond_noise_loss, 
            tf.zeros_like(cond_noise_loss)
        )

        uncond_has_rows = tf.reduce_any(uncond_mask)
        uncond_noise_loss = self.compiled_loss(
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
        teacher_noises_pred: tf.Tensor, 
        noises_pred: tf.Tensor, 
        teacher_noise_mask: tf.Tensor | None = None
    ) -> tf.Tensor | float:
        """Compute the optional masked noise-distillation loss.

        Args:
            teacher_noises_pred (tf.Tensor): Frozen teacher noise predictions.
            noises_pred (tf.Tensor): Student noise predictions.
            teacher_noise_mask (tf.Tensor | None): Samples taught by the
                previous network, or None to use the whole batch.

        Returns:
            tf.Tensor | float: Distillation loss, or 0.0 when disabled.
        """

        noise_distil_sample_weight = None

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
                        (tf.rank(noises_pred) - 2,), 
                        dtype=tf.int32
                    )
                ], axis=0)
            )

        noise_distil_loss = self.compiled_loss(
            tf.stop_gradient(teacher_noises_pred), 
            noises_pred, 
            sample_weight=noise_distil_sample_weight
        )

        return noise_distil_loss

    def compute_ctr_loss(
        self, 
        classes: tf.Tensor, 
        classes_pred_list: list[tf.Tensor]
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Average auxiliary class predictions and compute cross-entropy.

        Args:
            classes (tf.Tensor): Zero-based ground-truth classes ``[B]``.
            classes_pred_list (list[tf.Tensor | None]): Optional softmax tensors
                ``[B,num_classes]`` from regularizer depths.

        Returns:
            tuple[tf.Tensor | float, tf.Tensor]: Mean sparse categorical loss
            (0.0 when no predictions exist) and averaged class probabilities.
            The latter is zeros ``[B,num_classes]`` when the list has no tensor.
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

        # Average available predictions before computing token classification loss.
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
            z_vals_list_u (list[tuple[tf.Tensor, tf.Tensor]] | None): Unconditional
                latent pairs, required when KL trains unconditionally.
            regs_list_u (list[tf.Tensor] | None): Unconditional
                regularizers, required when token loss trains unconditionally.
            kl_train_type (TrainType | None): ``"cond"``/``"uncond"`` source;
                None uses the configured value.
            ctr_train_type (TrainType | None): Regularizer source; None uses the
                configured value.
            use_image_loss (bool | None): Compute reconstruction loss; None uses
                ``self.use_image_loss``.
            cond_labels (tf.Tensor | None): Conditional/possibly dropped label
                IDs ``[B]`` used only for optional split-noise reporting.
            teacher_noise_mask (tf.Tensor | None): Rows whose condition exists
                in the teacher vocabulary.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor | None, tf.Tensor | None,
            tf.Tensor | float, tf.Tensor | float, tf.Tensor | float, tf.Tensor]:
            Weighted total, raw noise loss, optional conditional/unconditional 
            noise losses, teacher-student noise loss, image loss, KL loss, 
            class-token loss, and averaged token predictions.
        """

        kl_train_type = self.kl_train_type if kl_train_type is None else kl_train_type
        ctr_train_type = self.ctr_train_type if ctr_train_type is None else ctr_train_type
        use_image_loss = self.use_image_loss if use_image_loss is None else use_image_loss

        noise_loss = self.compiled_loss(
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
            teacher_noise_mask
        ) if self.use_noise_distil_loss else 0.
        image_loss = self.compiled_loss(
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

        Args:
            x_t (tf.Tensor): Noisy image batch ``[B,H,W,C]``.
            t_batch (tf.Tensor): Integer timesteps ``[B]``.
            cond_labels (tf.Tensor): Shifted/conditional label IDs ``[B]``.
            uncond_labels (tf.Tensor | None): Null IDs ``[B]``; required for a
                guided pass.
            scale (float | None): Non-None requests the unconditional pass when
                CFG is enabled.  Combination happens later in ``compute_eps``.
            network_name (NetworkName): ``"raw"`` or ``"ema"``.
            training (bool): Keras training mode.

        Returns:
            tuple: ``((eps_c, eps_u), (regs_c, regs_u),
            (z_vals_list_c, z_vals_list_u))``. Noise
            predictions are ``[B,H,W,C]``; unconditional members are None when
            no second pass runs.
        """
        network = self.get_network(network_name)


        def run_network(
            labels: tf.Tensor
        ) -> tuple[
            tf.Tensor, 
            list[tf.Tensor], 
            list[tuple[tf.Tensor, tf.Tensor]]
        ]:
            """Run one conditional-label branch of the selected network.

            Args:
                labels (tf.Tensor): Integer condition IDs of shape ``[B]``.

            Returns:
                tuple: Noise prediction, auxiliary class predictions, and an
                ordered list of latent mean/log-variance pairs.
            """

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

            if isinstance(outputs, Mapping):
                return (
                    outputs["noises"], 
                    outputs.get("regs_list", []), 
                    outputs.get("z_vals_list", [])
                )

            eps, *_, regs_list, z_vals_list = outputs

            return eps, regs_list, z_vals_list


        eps_c, regs_list_c, z_vals_list_c = run_network(cond_labels)
        eps_u, regs_list_u, z_vals_list_u = run_network(uncond_labels) \
                                    if self.use_cfg and scale is not None \
                                    else (None, None, [])

        return ((eps_c, eps_u), 
                (regs_list_c, regs_list_u), 
                (z_vals_list_c, z_vals_list_u))

    def compute_eps(
        self, 
        eps_c: tf.Tensor, 
        eps_u: tf.Tensor | None = None, 
        scale: float | None = None
    ) -> tf.Tensor:
        """Combine conditional/unconditional noise with CFG.

        Args:
            eps_c (tf.Tensor): Conditional prediction of any image-like shape.
            eps_u (tf.Tensor | None): Unconditional prediction of the same shape.
            scale (float | None): Guidance scale.  With CFG and a non-None value,
                returns ``eps_u + scale*(eps_c-eps_u)``; otherwise ``eps_c``.

        Returns:
            tf.Tensor: Selected or guided noise prediction.
        """

        # Combine conditional and unconditional predictions with CFG.
        if self.use_cfg and scale is not None:
            eps = eps_u + scale * (eps_c - eps_u)
        # Use the conditional prediction directly when CFG is inactive.
        else:
            eps = eps_c

        return eps

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

        Args:
            x_t (tf.Tensor): Noisy images ``[B,H,W,C]``.
            t (tf.Tensor): Scalar timestep or per-example IDs ``[B]``.
            eps_c (tf.Tensor): Conditional noise prediction matching ``x_t``.
            eps_u (tf.Tensor | None): Optional unconditional prediction.
            scale (float | None): CFG scale; None disables combination.
            reshape_coefs (bool): Reshape vector schedule rates to
                ``[B,1,1,1]`` for image broadcasting.  Scalar rates do not need
                this.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Reconstructed ``x0`` and the selected/
            guided noise, both matching ``x_t`` shape.
        """

        eps = self.compute_eps(
            eps_c, 
            eps_u, 
            scale
        )

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
        x0 = (
            stable_x_t - sqrt_one_minus_a_t * stable_eps
        ) / sqrt_a_t

        return tf.cast(x0, x_t.dtype), eps

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

        Args:
            network_name (NetworkName): ``"raw"`` or ``"ema"``.
            x_t (tf.Tensor): Noisy images ``[B,H,W,C]``.
            t (tf.Tensor): Scalar or batch timestep used to gather schedule rates.
            t_batch (tf.Tensor): Batch-shaped integer timesteps ``[B]`` supplied
                to the network's embedding.
            cond_labels (tf.Tensor): Conditional label IDs ``[B]``.
            uncond_labels (tf.Tensor | None): Null labels for CFG.
            scale (float | None): Guidance scale; None skips unconditional pass.
            training (bool | None): Keras training mode.

        Returns:
            tuple: ``(x0, eps, (regs_c, regs_u),
            (z_vals_list_c, z_vals_list_u))``. Image tensors
            match ``x_t``; regularizers and latent pairs preserve branch outputs.
        """

        (eps_c, eps_u), *others = self.call_network(
            x_t, 
            t_batch, 
            cond_labels, 
            uncond_labels, 
            scale, 
            network_name, 
            training
        )

        # if self.swap_noise_image:
        #     x0 = self.compute_eps(eps_c, eps_u, scale)
        #     eps = x0
        # else:
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
            kl_train_type (TrainType | None): Conditional/unconditional KL source.
            ctr_train_type (TrainType | None): Token-regularizer source.
            use_image_loss (bool | None): Reconstruction-loss override.
            training (bool | None): Keras training mode.

        Returns:
            tuple[tf.Tensor, ...]: Weighted total, noise, optional split-noise,
            image, KL, and class-token losses plus averaged token class
            probabilities, in that order.
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
        use_ctr_loss: bool | None = None
    ) -> dict[str, tf.Tensor]:
        """Update enabled diffusion metric trackers and return their results.

        Args:
            noise_loss (tf.Tensor): Required scalar noise loss.
            noise_distil_loss (tf.Tensor | None): Teacher-student noise loss.
            cond_noise_loss (tf.Tensor | None): Conditional-row noise loss for
                optional split reporting.
            uncond_noise_loss (tf.Tensor | None): Null-row noise loss for
                optional split reporting.
            total_loss (tf.Tensor | None): Required when total tracking is on.
            image_loss (tf.Tensor | None): Required when image tracking is on.
            kl_loss (tf.Tensor | None): Required when KL tracking is on.
            ctr_loss (tf.Tensor | None): Required when token tracking is on.
            ctr_preds (tf.Tensor | None): ``[B,num_classes]`` token predictions.
            classes (tf.Tensor | None): Ground-truth classes ``[B]``.
            cond_labels (tf.Tensor | None): Post-dropout condition IDs used to
                weight split-noise means by their numbers of rows. When
                omitted, each supplied split loss receives unit weight.
            use_total_loss (bool | None): Explicit total tracker switch; None
                enables it when any auxiliary loss is enabled.
            use_noise_distil_loss (bool | None): Noise-distillation tracker
                override.
            use_image_loss (bool | None): Explicit image tracker switch.
            use_kl_loss (bool | None): Explicit KL tracker switch.
            use_ctr_loss (bool | None): Explicit token loss/accuracy switch.

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

        if use_noise_distil_loss:
            require(
                noise_distil_loss is not None, 
                "noise_distil_loss is required when noise distillation is active."
            )

            self.noise_distil_loss_tracker.update_state(
                noise_distil_loss, 
                sample_weight=batch_weight
            )
            results.update({
                self.noise_distil_loss_tracker.name: 
                self.noise_distil_loss_tracker.result()
            })

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
        z: tf.Tensor | Sequence[tf.Tensor] | None = None, 
        seed: int | None = None
    ) -> tf.Tensor:
        """Generate images by decoding the configured variational bottleneck.

        The first ``"flatten"`` reshaper is the encoder/decoder boundary.
        Each flatten stage receives its own latent and later connections may
        not route around the boundary to deterministic encoder features.

        Args:
            network_name (NetworkName): ``"ema"`` or ``"raw"`` decoder network.
            labels (tf.Tensor | list[int] | None): Condition IDs, one per sample.
                In dynamic mode, ``None`` shifts saved zero-based targets to
                condition IDs and excludes the CFG null label. Fixed-width
                mode likewise samples each class condition once and excludes
                the CFG null label. Explicit values are already network label
                IDs, not unshifted dataset classes.
            z (tf.Tensor | Sequence[tf.Tensor] | None): One latent batch per
                flatten stage. A tensor remains valid for a single-stage VAE.
                ``None`` draws independent standard-normal values; each batch
                size must match labels.
            seed (int | None): Latent random seed; None uses ``self.seed``.

        Returns:
            tf.Tensor: Decoded, postprocessed images ``[B,H,W,C]`` in ``[0,1]``.

        Raises:
            ValueError: If no flatten reshaper exists, it is not KL-enabled, or
                a later connection bypasses the bottleneck.
        """

        network = self.get_network(network_name)
        flatten_ids = sorted([
            int(id_) for id_, type_ in network.reshaper_ids_dict.items()
            if type_ == "flatten"
        ])
        z_id = flatten_ids[0] if flatten_ids else None

        # Require a flattening boundary before attempting latent decoding.
        if z_id is None:
            raise ValueError(
                "sample_vae requires a flatten reshaper."
            )
        # Require the flattening reshaper to expose a variational latent.
        if not network.reshaper_kwargs.get("add_kl", False):
            raise ValueError(
                "sample_vae requires add_kl=True in reshaper_kwargs."
            )

        for ids_dict in (
            network.connection_ids_dict, 
            network.cross_attention_ids_dict
        ):
            for depth, ids in ids_dict.items():
                # Reject decoder routes that bypass the variational bottleneck.
                if depth > z_id and any(id_ < z_id for id_ in ids):
                    raise ValueError(
                        "VAE decoder connections cannot use "
                        "features before the flatten reshaper."
                    )

        reshapers = [
            network.layers_dicts[flatten_id - 1][network.R]
            for flatten_id in flatten_ids
        ]
        z_projectors = [
            reshaper.get_layer(
                f"{network.name_prefix}depth_{flatten_id}_{network.R[2:]}/z"
            ) if network.reshaper_kwargs.get(
                "latent_dim_ratio", 1
            ) != 1 else None
            for flatten_id, reshaper in zip(flatten_ids, reshapers)
        ]

        default_labels = [
            value + int(network.use_cfg)
            for value in self.seen_classes.values()
        ] if network.dynamic_num_classes else list(
            range(int(network.use_cfg), network.num_labels)
        )
        labels = self._prepare_sampling_labels(
            network, 
            default_labels if labels is None else labels
        )
        n = tf.shape(labels)[0]
        seed = self._normalize_seed(
            self.seed if seed is None else seed, 
            "sample seed"
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
                tf.random.normal(
                    shape=tf.stack((n, latent_width)), 
                    mean=0., 
                    stddev=1., 
                    dtype=stable_dtype, 
                    seed=derive_seed(seed, "sample_vae", flatten_id)
                )
                for flatten_id, latent_width in zip(
                    flatten_ids, latent_widths
                )
            ]
        # A single boundary retains the historical tensor/list input API.
        elif len(flatten_ids) == 1:
            if isinstance(z, (list, tuple)):
                if len(z) != 1:
                    raise ValueError("z must contain one latent tensor.")

                z_vals_list = list(z)
            else:
                z_vals_list = [z]
        else:
            if not isinstance(z, (list, tuple)) or len(z) != len(flatten_ids):
                raise ValueError(
                    f"z must contain {len(flatten_ids)} latent tensors."
                )

            z_vals_list = list(z)

        projected_z_vals_list = []
        for latent, latent_width, z_projector in zip(
            z_vals_list, latent_widths, z_projectors
        ):
            try:
                latent = tf.convert_to_tensor(latent)
            except (TypeError, ValueError) as error:
                raise TypeError("z must be a floating tensor.") from error
            # Latent arithmetic requires a floating dtype.
            if not latent.dtype.is_floating:
                raise TypeError("z must have a floating dtype.")
            # The variational bottleneck consumes one vector per example.
            if latent.shape.rank is not None and latent.shape.rank != 2:
                raise ValueError("z must be a rank-2 latent tensor.")
            # Reject a statically incompatible bottleneck width early.
            if latent.shape.rank == 2 and latent.shape[-1] is not None \
            and int(latent.shape[-1]) != latent_width:
                raise ValueError(
                    f"z width must be {latent_width}, got {latent.shape[-1]}."
                )
            # Reject a statically incompatible batch size early.
            if latent.shape.rank == 2 and latent.shape[0] is not None \
            and labels.shape[0] is not None \
            and int(latent.shape[0]) != int(labels.shape[0]):
                raise ValueError("z batch size must match labels.")
            z_assertions = (
                tf.debugging.assert_rank(
                    latent, 
                    2,
                    message="z must be a rank-2 latent tensor."
                ), 
                tf.debugging.assert_equal(
                    tf.shape(latent)[0], 
                    n, 
                    message="z batch size must match labels.",
                ), 
                tf.debugging.assert_equal(
                    tf.shape(latent)[1], 
                    latent_width, 
                    message="z width does not match the variational bottleneck."
                )
            )
            with tf.control_dependencies([
                assertion for assertion in z_assertions
                if assertion is not None
            ]):
                latent = tf.cast(tf.identity(latent), stable_dtype)

            projected_z_vals_list.append(
                z_projector(latent, training=False)
                if z_projector is not None else latent
            )

        decoder_input = projected_z_vals_list[0] \
                        if len(projected_z_vals_list) == 1 \
                        else projected_z_vals_list
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
        else:
            images = network(
                (decoder_input, ts, labels), 
                min_depth=z_id, 
                training=False
            )
        if isinstance(images, Mapping):
            images = images["noises"]

        return self.postprocess(images)

    def sample(
        self, 
        network_name: NetworkName = "ema", 
        labels: tf.Tensor| list | None = None, 
        x_t: tf.Tensor | None = None, 
        steps: int | None = None, 
        scale: float | None = None, 
        eta: float | None = None, 
        return_x_ts: bool = False, 
        return_x0s: bool = False, 
        seed: int | None = None, 
        verbose: bool = False
    ) -> tf.Tensor | list[object]:
        """Generate images with generalized DDIM/DDPM reverse diffusion.

        Args:
            network_name (NetworkName): ``"ema"`` or ``"raw"`` predictor.
            labels (tf.Tensor | list[int] | None): Network condition IDs. In
                dynamic mode, None shifts observed zero-based targets to
                condition IDs and excludes the CFG null label. Fixed-width
                mode likewise samples each class condition once and excludes
                the CFG null label. The number of labels is the batch size.
            x_t (tf.Tensor | None): Initial Gaussian state ``[B,H,W,C]``.  None
                draws it at the active resolution.  In ``swap_noise_image`` VAE
                mode this argument is instead passed to ``sample_vae`` as ``z``.
            steps (int | None): Number of evenly spaced reverse evaluations;
                None uses ``test_steps``.  Must be an integer in
                ``[2, timesteps]``.
            scale (float | None): CFG scale; None uses ``test_cfg_scale``.  0
                follows the unconditional prediction, 1 the conditional one,
                and values above 1 extrapolate toward the condition.
            eta (float | None): Stochasticity in ``[0,1]``; None uses
                ``test_eta``.  0 gives
                deterministic DDIM, 1 is DDPM-equivalent for full consecutive
                timesteps, and values strictly between give stochastic DDIM.
            return_x_ts (bool): Include postprocessed state snapshots before
                each reverse update.
            return_x0s (bool): Include postprocessed x0 estimates at each step.
            seed (int | None): Random seed for initial/step noise.
            verbose (bool): Print reverse-step progress.

        Returns:
            tf.Tensor | list[object]: Final postprocessed images ``[B,H,W,C]``
            in ``[0,1]`` when no trajectories are requested.  Otherwise returns
            ``[images, x_ts?, x0s?]`` in requested order; trajectory entries are
            lists of NumPy arrays, one per reverse step.
        """

        for name, value in (
            ("return_x_ts", return_x_ts), 
            ("return_x0s", return_x0s), 
            ("verbose", verbose)
        ):
            # Sampling-control flags must be true Booleans.
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean.")

        # Route sampling through the variational decoder in swapped-objective mode.
        if self.swap_noise_image:
            # The direct variational path has no reverse-diffusion trajectory.
            if return_x_ts or return_x0s:
                raise ValueError(
                    "Sampling trajectories are unavailable "
                    "when swap_noise_image=True."
                )

            return self.sample_vae(
                network_name=network_name, 
                labels=labels, 
                z=x_t, 
                seed=seed
            )

        network = self.get_network(network_name)
        default_labels = [
            value + int(network.use_cfg)
            for value in self.seen_classes.values()
        ] if network.dynamic_num_classes else list(
            range(int(network.use_cfg), network.num_labels)
        )

        labels = self._prepare_sampling_labels(
            network,
            default_labels if labels is None else labels,
        )
        n = tf.shape(labels)[0]
        seed = self._normalize_seed(
            self.seed if seed is None else seed, 
            "sample seed"
        )
        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)

        # Draw one initial Gaussian image per requested label when absent.
        if x_t is None:
            x_t = tf.random.normal(
                tf.stack((
                    n, 
                    self._current_resolution, 
                    self._current_resolution, 
                    self.channels
                )),
                dtype=stable_dtype, 
                seed=seed
            )
        # Normalize and validate a caller-supplied reverse-process state.
        else:
            try:
                x_t = tf.convert_to_tensor(x_t)
            except (TypeError, ValueError) as error:
                raise TypeError("x_t must be a floating image tensor.") from error
            # Reverse diffusion requires real-valued image states.
            if not x_t.dtype.is_floating:
                raise TypeError("x_t must have a floating dtype.")
            # Sampling operates on NHWC image batches.
            if x_t.shape.rank is not None and x_t.shape.rank != 4:
                raise ValueError("x_t must be a rank-4 image tensor.")
            expected_tail = (
                self._current_resolution, 
                self._current_resolution, 
                self.channels
            )
            # Validate every statically known image dimension before tracing.
            if x_t.shape.rank == 4:
                for axis, (actual, expected) in enumerate(
                    zip(x_t.shape[1:], expected_tail), start=1
                ):
                    # Reject a statically incompatible spatial or channel axis.
                    if actual is not None and int(actual) != expected:
                        raise ValueError(
                            f"x_t axis {axis} must have size {expected}, "
                            f"got {actual}."
                        )
                # Reject a statically incompatible sample count early.
                if x_t.shape[0] is not None and labels.shape[0] is not None \
                and int(x_t.shape[0]) != int(labels.shape[0]):
                    raise ValueError("x_t batch size must match labels.")
            x_t_assertions = (
                tf.debugging.assert_rank(
                    x_t, 4, message="x_t must be a rank-4 image tensor."
                ),
                tf.debugging.assert_equal(
                    tf.shape(x_t)[0], n,
                    message="x_t batch size must match labels.",
                ),
                tf.debugging.assert_equal(
                    tf.shape(x_t)[1:],
                    tf.constant(expected_tail, dtype=tf.int32),
                    message="x_t spatial/channel shape is incompatible.",
                ),
            )
            with tf.control_dependencies([
                assertion for assertion in x_t_assertions
                if assertion is not None
            ]):
                x_t = tf.cast(tf.identity(x_t), stable_dtype)

        steps = self.test_steps if steps is None else steps
        scale = self.test_cfg_scale if scale is None else scale
        eta = self.test_eta if eta is None else eta

        # Validate the requested number of reverse steps against the schedule.
        if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) \
                or not 2 <= int(steps) <= self.timesteps:
            raise ValueError(
                f"steps must be an integer in [2, {self.timesteps}], got {steps!r}."
            )
        # Validate stochasticity as a finite scalar in the documented range.
        if isinstance(eta, bool) or not isinstance(
                eta, (int, float, np.integer, np.floating)) \
                or not np.isfinite(eta) or not 0. <= float(eta) <= 1.:
            raise ValueError(
                f"eta must be a finite number in [0, 1], got {eta!r}."
            )
        # Guidance may extrapolate in either direction but must remain finite.
        if isinstance(scale, bool) or not isinstance(scale, Real) \
        or not np.isfinite(scale):
            raise ValueError(
                f"scale must be a finite number, got {scale!r}."
            )

        steps = int(steps)
        eta = float(eta)
        scale = float(scale)
        ts = np.linspace(
            0, self.timesteps-1, 
            num=steps, 
            dtype="int32"
        )[::-1]
        cond_labels = labels
        uncond_labels = tf.zeros_like(labels, dtype=tf.int32)
        
        steps = len(ts)
        x0s, x_ts = [], []
        for i in range(steps):
            # Report reverse-diffusion progress when requested.
            if verbose:
                print(f"\rSteps: {i+1}/{steps}", end="")

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
                x_ts.append(self.postprocess(x_t).numpy())
            # Capture each clean-image estimate for the optional trajectory.
            if return_x0s:
                x0s.append(self.postprocess(x0).numpy())

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
                            1. - alpha_bar_t_next - sigma_t ** 2, 0.0
                    )),
                    dtype=stable_dtype
                )

                stable_x0 = tf.cast(x0, stable_dtype)
                stable_eps = tf.cast(eps, stable_dtype)
                x_t = x0_coef * stable_x0 + eps_coeff * stable_eps
                # Add stochastic DDIM noise when eta is positive.
                if eta > 0.:
                    x_t += sigma_t * tf.random.normal(
                        tf.shape(x_t),
                        dtype=stable_dtype,
                        seed=seed
                    )

        # Finish the in-place progress line after sampling.
        if verbose:
            print()

        outputs = [self.postprocess(x0)]
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

    @classmethod
    def from_config(
        cls, 
        config: Mapping[str, object]
    ) -> "DiffusionModel":
        """Reconstruct an independent wrapper and nested raw network."""

        config = dict(config)
        network = config["network"]

        if isinstance(network, Mapping):
            network_config = dict(network)
            module_name = network_config.pop("module", None)

            if module_name is None:
                network = tf.keras.utils.deserialize_keras_object(
                    network_config
                )
            else:
                network_type = getattr(
                    import_module(module_name), 
                    network_config["class_name"]
                )
                network = network_type.from_config(network_config["config"])
        elif isinstance(network, tf.keras.Model):
            network = network.__class__.from_config(network.get_config())

        config["network"] = network

        return cls(**config)


def run_self_tests() -> dict[str, str]:
    """Run deterministic end-to-end tests for DiffusionModel.

    Args:
        None.

    Returns:
        dict[str, str]: ``{"DiffusionModel": "passed"}`` after schedule,
        noising, CFG, loss, optimizer, EMA, fit/evaluate, sampling, curriculum,
        serialization-facing state, and invalid-input checks pass.
    """

    tf.keras.backend.clear_session()
    tf.random.set_seed(105)


    from types import SimpleNamespace
    from unittest.mock import MagicMock, Mock


    def make_network(**overrides: object) -> DiffusionTransformer:
        """Build a fresh depth-zero network for wrapper tests.

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

        Args:
            **overrides (object): DiffusionModel arguments overriding defaults.

        Returns:
            DiffusionModel: Compiled wrapper with a fresh raw network.
        """

        network = overrides.pop("network", make_network())
        config = {
            "network": network,
            "use_ema": True,
            "test_network_name": "ema",
            "scheduler_name": "linear",
            "test_steps": 2,
            "test_eta": 0.0,
            "seed": 17,
            **overrides,
        }
        wrapper = DiffusionModel(**config)
        wrapper.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="mse",
            run_eagerly=True,
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
        seed=np.int64(17),
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
        "loss", "noise_loss", "noise_distil_loss", "image_loss", "kl_loss",
        "ctr_loss", "ctr_accuracy",
    ]
    separate_noise_wrapper = make_wrapper(
        show_separate_noise_losses=True
    )
    assert [metric.name for metric in separate_noise_wrapper.metrics] == [
        "loss", "total_noise_loss", "cond_noise_loss",
        "uncond_noise_loss", "noise_distil_loss", "image_loss", "kl_loss",
        "ctr_loss", "ctr_accuracy",
    ]
    assert separate_noise_wrapper.get_config()[
        "show_separate_noise_losses"
    ] is True

    dynamic_wrapper = make_wrapper(network=make_network(
        num_classes=None,
        cls_token_regularizer_ids=[0],
        cls_token_regularizer_kwargs={
            "start": 0, "end": 1, "mlp_ratio": 2.0
        },
    ))
    dynamic_wrapper._check_new_labels(y=np.array([3]), verbose=False)
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
        tf.constant([1, 0], tf.int32),
    )
    for dynamic_network in (
        dynamic_wrapper.network,
        dynamic_wrapper.ema_network,
    ):
        assert dynamic_network.num_classes == 2
        assert dynamic_network.labels_embed_reg.layers[-1].units == 2
    for expected, actual in zip(
        ema_hidden_before,
        dynamic_wrapper.ema_network.labels_embed_reg.layers[0].get_weights(),
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
        np.full_like(ema_kernel[..., 0], 2.0),
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
        test_eta=0.0, 
    )
    assert getattr(uncompiled, "optimizer", None) is None
    assert uncompiled._register_optimizer_variables() is None
    assert uncompiled._register_optimizer_variables(variables=[]) is None

    required_schedule_keys = {
        "betas", "alpha_bar", "sqrt_alpha_bar",
        "sqrt_one_minus_alpha_bar", "sigmas", "timesteps",
    }
    assert required_schedule_keys <= set(wrapper.schedules)
    assert all(value.dtype == tf.float32 for value in wrapper.schedules.values())
    assert all(value.shape == (4,) for value in wrapper.schedules.values())
    assert wrapper._get_progressive_timestep_boundaries(2, "uniform") == [0, 2, 4]
    log_boundaries = wrapper._get_progressive_timestep_boundaries(2, "log_snr")
    assert log_boundaries[0] == 0 and log_boundaries[-1] == 4
    assert log_boundaries[0] < log_boundaries[1] < log_boundaries[2]
    for bad_stage_count in (0, 5):
        try:
            wrapper._get_progressive_timestep_boundaries(bad_stage_count)
        except AssertionError:
            pass
        else:
            raise AssertionError("Invalid progressive stage counts must fail")
    try:
        wrapper._get_progressive_timestep_boundaries(2, "unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown timestep clustering must fail")

    for scheduler_name in (
        "linear", "scaled_linear", "squaredcos_cap_v2", "clipped_cosine",
        "sigmoid", "quadratic", "ve", "karras", "sub_vp", "logistic",
    ):
        wrapper.load_schedules(scheduler_name=scheduler_name, timesteps=4)
        assert wrapper.scheduler_name == scheduler_name
        assert wrapper.timesteps == 4
        assert wrapper.schedules["alpha_bar"].shape == (4,)
    wrapper.load_schedules("linear", 4)
    schedule_before_invalid_reload = tf.identity(wrapper.schedules["alpha_bar"])
    try:
        wrapper.load_schedules("quadratic", 3)
    except ValueError:
        pass
    else:
        raise AssertionError("A schedule/network timestep mismatch must fail")
    assert wrapper.scheduler_name == "linear" and wrapper.timesteps == 4
    tf.debugging.assert_equal(
        wrapper.schedules["alpha_bar"], schedule_before_invalid_reload
    )
    modified = make_wrapper(modify_first_t=True)
    assert float(modified.schedules["sqrt_alpha_bar"][0]) == 1.0
    assert float(modified.schedules["sqrt_one_minus_alpha_bar"][0]) == 0.0
    assert float(modified.schedules["alpha_bar"][0]) == 1.0
    tf.debugging.assert_near(
        modified.schedules["sqrt_alpha_bar"] ** 2,
        modified.schedules["alpha_bar"],
    )
    tf.debugging.assert_near(
        modified.schedules["sqrt_one_minus_alpha_bar"] ** 2,
        1. - modified.schedules["alpha_bar"],
    )
    tf.debugging.assert_near(
        modified.schedules["sigmas"],
        modified.schedules["sqrt_one_minus_alpha_bar"],
    )
    tf.debugging.assert_near(
        tf.math.cumprod(1. - modified.schedules["betas"]),
        modified.schedules["alpha_bar"],
    )

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
    tf.debugging.assert_equal(clean_t, tf.zeros((2,), tf.int32))
    wrapper.set_timestep_bounds()
    assert wrapper.current_timesteps_bounds == (0, 4)
    for invalid_bounds in ((-1, 2), (2, 2), (3, 2), (0, 5)):
        try:
            wrapper.set_timestep_bounds(*invalid_bounds)
        except AssertionError:
            pass
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
        tf.ones_like(images[1:]) * 4.,
    ], axis=0)
    split_losses = separate_noise_wrapper.compute_noise_distil_image_kl_ctr_loss(
        tf.zeros_like(images),
        split_targets,
        classes,
        tf.zeros_like(images),
        tf.zeros_like(images),
        (None, None),
        [None],
        cond_labels=tf.constant([1, 0], dtype=tf.uint8),
    )
    split_results = separate_noise_wrapper.get_results_dict(
        split_losses[1],
        cond_noise_loss=split_losses[2],
        uncond_noise_loss=split_losses[3],
        cond_labels=tf.constant([1, 0], dtype=tf.uint8),
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
        show_separate_noise_losses=True,
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
            cond_labels=classes,
        )
    )
    no_cfg_split_results = no_cfg_separate_noise_wrapper.get_results_dict(
        no_cfg_split_losses[1],
        cond_noise_loss=no_cfg_split_losses[2],
        uncond_noise_loss=no_cfg_split_losses[3],
        cond_labels=classes,
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
    assert signal.shape == noise_rate.shape == (2,)
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
    try:
        wrapper.noisify(tf.ones((1, 4, 4, 1), dtype=tf.int32), t=fixed_t[:1])
    except TypeError:
        pass
    else:
        raise AssertionError("Integer clean images must fail clearly")
    x_t, noises, returned_t = wrapper.noisify(images, t=fixed_t, seed=19)
    assert x_t.shape == noises.shape == images.shape
    tf.debugging.assert_equal(returned_t, fixed_t)
    random_x_t, random_noise, random_t = wrapper.noisify(
        images, min_timesteps=1, max_timesteps=3, seed=19
    )
    assert random_x_t.shape == random_noise.shape == images.shape
    assert bool(tf.reduce_all((1 <= random_t) & (random_t < 3)))
    for invalid_noisify_kwargs in (
        {"min_timesteps": 2, "max_timesteps": 2},
        {"min_timesteps": True, "max_timesteps": 2},
        {"t": tf.constant([[0], [1]], dtype=tf.int32)},
        {"t": tf.constant([0], dtype=tf.int32)},
        {"t": tf.constant([-1, 0], dtype=tf.int32)},
        {"t": tf.constant([0, 4], dtype=tf.int32)},
        {"t": tf.constant([0.0, 1.0])},
    ):
        try:
            wrapper.noisify(images, seed=19, **invalid_noisify_kwargs)
        except (TypeError, ValueError, tf.errors.InvalidArgumentError):
            pass
        else:
            raise AssertionError(
                f"Invalid noising inputs accepted: {invalid_noisify_kwargs}"
            )

    processed = wrapper.postprocess(tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0]))
    tf.debugging.assert_near(processed, [0.0, 0.0, 0.5, 1.0, 1.0])
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
    tf.debugging.assert_equal(wrapper.compute_eps(conditional), conditional)
    tf.debugging.assert_equal(
        wrapper.compute_eps(conditional, unconditional, scale=2.0),
        2.0 * conditional,
    )
    reconstructed, selected_eps = wrapper.denoise(
        x_t, fixed_t, conditional, unconditional, scale=1.0, 
        reshape_coefs=True,
    )
    assert reconstructed.shape == selected_eps.shape == images.shape
    network_outputs = wrapper.call_network(
        x_t, fixed_t, cfg_labels, nulls, scale=2.0, 
        network_name="raw", training=False,
    )
    assert network_outputs[0][0].shape == network_outputs[0][1].shape == images.shape
    assert network_outputs[1][0] == [None]
    forward = wrapper.forward(
        "raw", x_t, fixed_t, fixed_t, cfg_labels, nulls, 
        scale=2.0, training=False,
    )
    assert forward[0].shape == forward[1].shape == images.shape
    losses_tuple = wrapper.forward_and_compute_loss(
        "raw", images, noises, fixed_t, x_t, cfg_labels, nulls, classes,
        cfg_scale=None, use_image_loss=True, training=False,
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
        resize_antialias=False, 
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
        weighted.train_noisified_max_timesteps, 
    )
    weighted_prepared = weighted.prep_inputs((images, classes), seed=47)
    assert bool(tf.reduce_all((1 <= weighted_prepared[2]) & (weighted_prepared[2] < 3)))
    weighted.set_timestep_bounds(
        weighted.test_noisified_min_timesteps,
        weighted.test_noisified_max_timesteps,
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
        training=False, 
    )
    tf.debugging.assert_near(
        weighted_losses[0], 
        0.5 * weighted_losses[1] + 0.25 * weighted_losses[5],
        atol=1e-5, 
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
        "use_ctr_loss": False, 
    }
    missing_result_cases = (
        ({**common_result_flags, "use_total_loss": True}, "total_loss"), 
        ({**common_result_flags, "use_image_loss": True}, "image_loss"), 
        ({**common_result_flags, "use_kl_loss": True}, "kl_loss"), 
    )
    for result_flags, missing_name in missing_result_cases:
        try:
            results_probe.get_results_dict(
                noise_loss=tf.constant(0.0), 
                **result_flags,
            )
        except AssertionError as error:
            assert missing_name in str(error)
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
                "ctr_preds": valid_ctr_predictions, 
            }, 
            "classes", 
        ),
    ):
        try:
            results_probe.get_results_dict(
                noise_loss=tf.constant(0.0), 
                use_total_loss=False, 
                use_image_loss=False, 
                use_kl_loss=False, 
                use_ctr_loss=True, 
                **ctr_inputs, 
            )
        except AssertionError as error:
            assert "ctr_loss, ctr_preds, and classes" in str(error)
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
        use_ctr_loss=True, 
    )
    assert set(complete_results) == {
        "loss", "noise_loss", "image_loss", "kl_loss", "ctr_loss",
        "ctr_accuracy",
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
        dataset, network_name="raw", verbose=0, return_dict=True
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
        verbose=0,
    )
    assert "total_noise_loss" in progressive_separate_history.history
    assert wrapper.test_network_name == "ema"
    summary_lines = []
    wrapper.summary(print_fn=summary_lines.append)
    assert any("Total params" in line for line in summary_lines)

    raw_sample = wrapper.sample(
        network_name="raw", labels=[1, 2], steps=2, eta=0.0, seed=29
    )
    assert raw_sample.shape == (2, 4, 4, 1)
    assert bool(tf.reduce_all((0.0 <= raw_sample) & (raw_sample <= 1.0)))
    trajectories = wrapper.sample(
        network_name="raw", labels=[1], steps=3, eta=1.0,
        return_x_ts=True, return_x0s=True, seed=29,
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
        steps=2, eta=0.0, seed=29,
    )
    assert supplied_sample.shape == supplied_state.shape
    states_only = wrapper.sample(
        network_name="raw", labels=[1], steps=2, eta=0.0,
        return_x_ts=True, return_x0s=False, seed=29,
    )
    clean_only = wrapper.sample(
        network_name="raw", labels=[1], steps=2, eta=0.0,
        return_x_ts=False, return_x0s=True, seed=29, verbose=True,
    )
    assert len(states_only) == len(clean_only) == 2
    assert len(states_only[1]) == len(clean_only[1]) == 2

    for invalid_sample_kwargs in (
        {"steps": 1, "eta": 0.0},
        {"steps": 5, "eta": 0.0},
        {"steps": 2, "eta": -0.1},
        {"steps": 2, "eta": 1.1},
        {"steps": 2, "scale": float("inf")},
        {"steps": 2, "return_x_ts": 1},
    ):
        try:
            wrapper.sample(
                network_name="raw",
                labels=[1],
                seed=31,
                **invalid_sample_kwargs,
            )
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(
                f"Invalid sampling overrides accepted: {invalid_sample_kwargs}"
            )
    invalid_sampling_inputs = (
        {"labels": []},
        {"labels": [[1]]},
        {"labels": [1.0]},
        {"labels": [-1]},
        {"labels": [wrapper.network.num_labels]},
        {"labels": [1], "x_t": tf.zeros((2, 4, 4, 1))},
        {"labels": [1], "x_t": tf.zeros((1, 2, 4, 1))},
        {"labels": [1], "x_t": tf.zeros((1, 4, 4, 1), tf.int32)},
    )
    for invalid_inputs in invalid_sampling_inputs:
        try:
            wrapper.sample(
                network_name="raw",
                steps=2,
                eta=0.0,
                seed=31,
                **invalid_inputs,
            )
        except (TypeError, ValueError, tf.errors.InvalidArgumentError):
            pass
        else:
            raise AssertionError(
                f"Invalid sampling inputs accepted: {invalid_inputs}"
            )
    try:
        wrapper.sample_vae(network_name="raw", labels=[1])
    except ValueError as error:
        assert "flatten reshaper" in str(error)
    else:
        raise AssertionError("VAE sampling without a bottleneck must fail")


    def make_variational_network(
        latent_dim_ratio: float = 1.0,
        add_kl: bool = True,
        build: bool = True,
        connection_ids_dict: dict[int, list[int]] | None = None,
    ) -> DiffusionTransformer:
        """Build a tiny KL bottleneck network for wrapper self-tests.

        Args:
            latent_dim_ratio (float): Latent-to-flattened-width ratio.
            add_kl (bool): Whether the flatten reshaper exposes a KL latent.
            build (bool): Whether to symbolically build the raw network.
            connection_ids_dict (dict[int, list[int]] | None): Optional routes
                used to exercise the VAE bypass validator.

        Returns:
            DiffusionTransformer: A two-depth flatten/unflatten network.
        """

        return make_network(
            depth=2, 
            vit_block_ids=[1], 
            cls_token_type="new_weight", 
            cls_token_regularizer_ids=[None], 
            reshaper_ids_dict={1: "flatten", 2: "unflatten"}, 
            reshaper_kwargs={
                "add_kl": add_kl, 
                "latent_dim_ratio": latent_dim_ratio, 
            }, 
            connection_ids_dict=(
                {} if connection_ids_dict is None else connection_ids_dict
            ), 
            build=build, 
        )


    variational_cond = make_wrapper(
        network=make_variational_network(), 
        kl_loss_coef=0.01, 
        ctr_loss_coef=0.01, 
        kl_train_type="cond", 
        ctr_train_type="cond", 
        train_cfg_scale=None, 
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
        train_cfg_scale=1.0, 
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
    full_reshaper = variational_cond.network.layers_dicts[0][
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
        z=[supplied_full_latent],
    )
    tf.debugging.assert_near(
        supplied_sequence_images,
        supplied_vae_images,
    )
    for invalid_z in (
        tf.zeros((1, full_latent_width), dtype=tf.float32),
        tf.zeros((2, full_latent_width + 1), dtype=tf.float32),
        tf.zeros((2, full_latent_width), dtype=tf.int32),
        tf.zeros((2, 1, full_latent_width), dtype=tf.float32),
    ):
        try:
            variational_cond.sample_vae(
                network_name="raw",
                labels=tensor_vae_labels,
                z=invalid_z,
            )
        except (TypeError, ValueError, tf.errors.InvalidArgumentError):
            pass
        else:
            raise AssertionError("Invalid VAE sampling latents must fail")
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
    except ValueError as error:
        assert "one-dimensional" in str(error)
    else:
        raise AssertionError("VAE labels must be one-dimensional")
    projected_vae = make_wrapper(
        network=make_variational_network(latent_dim_ratio=0.5),
        use_ema=False,
        test_network_name="raw",
    )
    random_projected_images = projected_vae.sample_vae(
        network_name="raw", labels=tensor_vae_labels, seed=53
    )
    assert random_projected_images.shape == (2, 4, 4, 1)
    projected_reshaper = projected_vae.network.layers_dicts[0][
        projected_vae.network.R
    ]
    projected_latent_width = int(projected_reshaper.output_shape[1][-1])
    projected_latent = tf.zeros((2, projected_latent_width), dtype=tf.float32)
    projected_images = projected_vae.sample_vae(
        network_name="raw", labels=tensor_vae_labels, z=projected_latent
    )
    assert projected_images.shape == (2, 4, 4, 1)
    non_variational = make_wrapper(
        network=make_variational_network(add_kl=False),
        use_ema=False,
        test_network_name="raw",
    )
    try:
        non_variational.sample_vae(
            network_name="raw", labels=tensor_vae_labels, seed=53
        )
    except ValueError as error:
        assert "add_kl=True" in str(error)
    else:
        raise AssertionError("VAE sampling without add_kl must fail")
    bypass_vae = make_wrapper(
        network=make_variational_network(
            build=False,
            connection_ids_dict={2: [0]},
        ),
        use_ema=False,
        test_network_name="raw",
    )
    try:
        bypass_vae.sample_vae(
            network_name="raw", labels=tf.constant([1], dtype=tf.uint8)
        )
    except ValueError as error:
        assert "cannot use features before" in str(error)
    else:
        raise AssertionError("VAE routes bypassing the bottleneck must fail")

    original_bounds = wrapper.current_timesteps_bounds
    original_resolution = wrapper.current_resolution
    progressive_history = wrapper.fit_progressively(
        stage_tasks=[
            {"timesteps": (2, 4)}, 
            {"resolution": 4}, 
        ],
        x=dataset, 
        stages_verbose=False, 
        stage_epochs=1, 
        final_epochs=0, 
        verbose=0, 
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
            {"timesteps": (0, 4), "resolution": 4}, 
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
        "resolution": 4, 
    }

    autogenerated_timesteps = make_wrapper().fit_progressively(
        "timesteps_only", 
        stages_num=2, 
        timestep_clustering_type="uniform", 
        x=dataset, 
        stages_verbose=False, 
        stage_epochs=0, 
        final_epochs=0, 
        verbose=0, 
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
        verbose=0, 
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
            verbose=0, 
        )
    except ValueError as error:
        assert "Invalid stage task at index 1" in str(error)
    else:
        raise AssertionError("Invalid progressive stage objects must fail")
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
        else:
            raise AssertionError("Managed progressive epoch arguments must fail")
    for invalid_progressive_control in (
        {"timestep_clustering_type": "unknown"}, 
        {"pacing_type": "unknown"}, 
        {"earlystopping_type": "unknown"}, 
        {"monitor": "unknown"}, 
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
        else:
            raise AssertionError(
                f"Expected invalid progressive control: {invalid_progressive_control}"
            )
    for only_mode, missing_values in (
        ("timesteps_only", {}), 
        ("resolutions_only", {}), 
        ("depths_only", {"stages_num": 1}), 
    ):
        try:
            wrapper.fit_progressively(
                only_mode, 
                stage_epochs=0, 
                final_epochs=0, 
                **missing_values, 
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"Missing values must fail for {only_mode}")

    serialization_network = make_network(build=False)
    serialization_network.built = True
    serialization_wrapper = DiffusionModel(
        network=serialization_network,
        use_ema=False,
        test_network_name="raw",
        test_steps=2,
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
        name="policy_wrapper", 
        trainable=False, 
        dtype="float64", 
        dynamic=True, 
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
    else:
        raise AssertionError("Unknown network names must fail")

    topology_probe = SimpleNamespace(
        use_ema=True, 
        network=SimpleNamespace(weights=[tf.Variable(0.0)]), 
        ema_network=SimpleNamespace(weights=[]), 
        ema_decay=0.9, 
    )
    try:
        DiffusionModel.update_ema(topology_probe)
    except AssertionError as error:
        assert "same topology" in str(error)
    else:
        raise AssertionError("Raw/EMA topology mismatch must fail")

    # Progressive growth has two distinct EMA-alignment guards: the number of
    # newly created weights must match, then every aligned shape must assign.
    raw_count_weights = MagicMock()
    raw_count_weights.__iter__.side_effect = [
        iter(()), 
        iter((tf.Variable(0.0),)), 
    ]
    ema_count_weights = MagicMock()
    ema_count_weights.__iter__.side_effect = [iter(()), iter(())]
    progressive_count_probe = SimpleNamespace(
        network=SimpleNamespace(
            weights=raw_count_weights, 
            add_depths=Mock(return_value={"network": {"added": 1}}), 
            build=Mock(return_value=None),
        ), 
        ema_network=SimpleNamespace(
            weights=ema_count_weights, 
            add_depths=Mock(return_value={"network": {"added": 0}}), 
            build=Mock(return_value=None),
        )
    )
    try:
        DiffusionModel._add_depths(progressive_count_probe, "probe")
    except ValueError as error:
        assert "progressive depths have different weights" in str(error)
    else:
        raise AssertionError(
            "Progressive raw/EMA new-weight count mismatch must fail"
        )

    raw_shape_weights = MagicMock()
    raw_shape_weights.__iter__.side_effect = [
        iter(()),
        iter((tf.Variable(tf.zeros((1,))),)),
    ]
    ema_shape_weights = MagicMock()
    ema_shape_weights.__iter__.side_effect = [
        iter(()),
        iter((tf.Variable(tf.zeros((2,))),)),
    ]
    progressive_shape_probe = SimpleNamespace(
        network=SimpleNamespace(
            weights=raw_shape_weights, 
            add_depths=Mock(return_value={"network": {"added": 1}}), 
            build=Mock(return_value=None),
        ), 
        ema_network=SimpleNamespace(
            weights=ema_shape_weights, 
            add_depths=Mock(return_value={"network": {"added": 1}}), 
            build=Mock(return_value=None),
        ), 
    )
    try:
        DiffusionModel._add_depths(progressive_shape_probe, "probe")
    except ValueError as error:
        assert "shape" in str(error).lower()
    else:
        raise AssertionError(
            "Progressive raw/EMA new-weight shape mismatch must fail"
        )

    without_ema = make_wrapper(
        network=make_network(use_cfg=False), 
        use_ema=False, 
        test_network_name="raw", 
        p_uncond=0.9, 
        test_cfg_scale=9.0, 
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
    else:
        raise AssertionError("swap_noise_image must route through VAE sampling")

    invalid_cases = (
        {"use_ema": 1},
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
        {"map_num_parallel_calls": False},
        {"map_num_parallel_calls": 0},
        {"train_noisified_min_timesteps": -1},
        {"test_noisified_min_timesteps": 3,
         "test_noisified_max_timesteps": 2},
        {"test_noisified_max_timesteps": 5},
        {"p_uncond": -0.25},
        {"p_uncond": 1.25},
        {"p_uncond": float("nan")},
        {"train_cfg_scale": float("inf")},
        {"test_cfg_scale": float("nan")},
        {"noise_loss_coef": -1.0},
        {"image_loss_coef": float("nan")},
        {"modify_first_t": 1},
        {"resize_antialias": 1},
        {"swap_noise_image": 0},
        {"show_separate_noise_losses": 1},
        {"kl_train_type": "unknown"},
        {"kl_train_type": "uncond", "train_cfg_scale": None},
        {"ctr_train_type": "unknown"}, 
        {"ctr_train_type": "uncond", "train_cfg_scale": None}, 
    )
    for overrides in invalid_cases:
        try:
            DiffusionModel(
                network=make_network(),
                **{"test_steps": 2, **overrides},
            )
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Expected invalid wrapper config: {overrides}")

    for invalid_seed in (True, -1, 2 ** 32, 1.5):
        try:
            DiffusionModel(network=make_network(), test_steps=2, seed=invalid_seed)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("Invalid wrapper seeds must fail")

    try:
        DiffusionModel(network=object(), test_steps=2)
    except TypeError:
        pass
    else:
        raise AssertionError("A non-diffusion network must fail clearly")

    tf.keras.backend.clear_session()
    return {"DiffusionModel": "passed"}


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
