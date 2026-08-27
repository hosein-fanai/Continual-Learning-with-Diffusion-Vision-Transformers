"""Training, evaluation, EMA, noising, and sampling for raw diffusion networks.

Networks in ``diffusion.models.transformer`` implement tensor transformations.
This module wraps one such network in the stateful Keras training protocol and
owns the diffusion process around it.
"""

import tensorflow as tf
from tensorflow.keras import metrics, losses, callbacks, optimizers

import numpy as np

from numbers import Integral, Real
from typing import Literal, Sequence, get_args

from . import NetworkName, TrainType, ClusteringType

from common.argument_saver import ArgumentSaverModel

from autoencoder.variational_autoencoder import VariationalAutoencoder

from diffusion.callbacks.batch_loss_plateau import BatchLossPlateau
from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer
from diffusion.models.transformer.di_t_decoder import DiTDecoder
from diffusion.schedulers import make_schedule, SchedulerName


# @tf.keras.saving.register_keras_serializable()
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
        use_ema: bool = True, 
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
            resize_method (str): ``tf.image.resize`` method used during
                progressive-resolution input preparation, for example ``"area"``
                or ``"bilinear"``.
            resize_antialias (bool): Antialias flag passed to ``tf.image.resize``.
            swap_noise_image (bool): Train the output against ``x_t`` instead of
                sampled Gaussian noise and route :meth:`sample` to
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

        self._init_config["seen_classes"] = self.seen_classes
        self.image_size = self.network.image_size
        self.channels = self.network.channels
        self.timesteps = self.network.timesteps
        self.use_cfg = self.network.use_cfg
        self.p_uncond = 0. if not self.use_cfg else self.p_uncond
        self.test_cfg_scale = 1. if not self.use_cfg else self.test_cfg_scale
        self.noise_loss_coef = tf.constant(self.noise_loss_coef, dtype=tf.float32)
        self.image_loss_coef = tf.constant(self.image_loss_coef, dtype=tf.float32)
        self.kl_loss_coef = tf.constant(self.kl_loss_coef, dtype=tf.float32)
        self.ctr_loss_coef = tf.constant(self.ctr_loss_coef, dtype=tf.float32)
        self.use_image_loss = bool(self.image_loss_coef > 0.)
        self.train_noisified_max_timesteps = self.timesteps if self.train_noisified_max_timesteps == -1 \
                                            else self.train_noisified_max_timesteps
        self.train_noisified_max_timesteps = 0 if self.train_noisified_max_timesteps is None \
                                            else self.train_noisified_max_timesteps
        self.test_noisified_max_timesteps = self.timesteps if self.test_noisified_max_timesteps == -1 \
                                            else self.test_noisified_max_timesteps
        self.test_noisified_max_timesteps = 0 if self.test_noisified_max_timesteps is None \
                                            else self.test_noisified_max_timesteps
        self.map_num_parallel_calls = tf.data.AUTOTUNE if self.map_num_parallel_calls is None \
                                    else self.map_num_parallel_calls
        self._preprocess_training = None

        self.load_schedules()
        self.set_timestep_bounds()
        # Avoid subclass resolution hooks until subclass fields are initialized.
        DiffusionModel.set_current_resolution(self)
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
            "show_separate_noise_losses"
        ):
            assert isinstance(local_vars[name], bool), f"{name} must be boolean."

        assert local_vars["test_network_name"] in get_args(NetworkName), \
            f"test_network_name must be one of {get_args(NetworkName)}."

        assert isinstance(local_vars["ema_decay"], Real) and \
            not isinstance(local_vars["ema_decay"], bool) and \
            np.isfinite(local_vars["ema_decay"]) and \
            0. <= local_vars["ema_decay"] < 1., \
            "ema_decay must be in the range of [0., 1.)."

        assert isinstance(local_vars["test_steps"], Integral) and \
            not isinstance(local_vars["test_steps"], bool) and \
            2 <= local_vars["test_steps"] <= network.timesteps, \
            "steps must be in the range of [2, timesteps]."

        assert isinstance(local_vars["test_eta"], Real) and \
            not isinstance(local_vars["test_eta"], bool) and \
            np.isfinite(local_vars["test_eta"]) and \
            0. <= local_vars["test_eta"] <= 1., \
            "eta must be in the range of [0., 1.]."

        assert isinstance(local_vars["p_uncond"], Real) and \
            not isinstance(local_vars["p_uncond"], bool) and \
            np.isfinite(local_vars["p_uncond"]) and \
            0. <= local_vars["p_uncond"] <= 1., \
            "p_uncond must be in the range of [0., 1.]."

        for name in ("train_cfg_scale", "test_cfg_scale"):
            value = local_vars[name]
            assert value is None or (
                isinstance(value, Real)
                and not isinstance(value, bool)
                and np.isfinite(value)
            ), f"{name} must be None or a finite number."

        for name in (
            "noise_loss_coef", "image_loss_coef",
            "kl_loss_coef", "ctr_loss_coef"
        ):
            value = local_vars[name]
            assert isinstance(value, Real) and \
                not isinstance(value, bool) and np.isfinite(value) and value >= 0., \
                f"{name} must be a finite nonnegative number."

        assert local_vars["kl_train_type"] in get_args(TrainType), \
            f"kl_train_type can be one of {TrainType}."

        assert local_vars["ctr_train_type"] in get_args(TrainType), \
            f"ctr_train_type can be one of {TrainType}."

        # Require CFG and a training scale for unconditional auxiliary losses.
        if local_vars["kl_train_type"] == "uncond" or \
        local_vars["ctr_train_type"] == "uncond":
            assert local_vars["network"].use_cfg and \
                local_vars["train_cfg_scale"] is not None, \
                "Unconditional auxiliary losses require "\
                "CFG and a non-None train_cfg_scale."

        # Normalize persisted continual state before constructor serialization.
        assert isinstance(local_vars["seen_classes"], dict), \
            "seen_classes must be a mapping."

        # Validate restoration width only when saved continual state is present.
        if local_vars["seen_classes"]:
            assert network.num_classes <= len(local_vars["seen_classes"]), \
                "seen_classes cannot be smaller than network.num_classes."

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

        assert 1 <= stages_num <= self.timesteps, \
            f"num_stages must be in [1, {self.timesteps}] range, "\
            f"but got {stages_num}."


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
            ``None``. ``use_kl_loss`` and ``use_ctr_loss`` are updated in place.
        """

        self.use_kl_loss = bool(
            self.kl_loss_coef > 0. and
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
        # Materialize the finite label stream from a Keras dataset.
        if isinstance(data, tf.data.Dataset):
            data = np.concatenate([
                batch[1].numpy() for batch in data
            ], axis=0)
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
            list[tf.keras.metrics.Metric]: Total, noise, optional split-noise,
            image, KL, class-token regularizer loss trackers and regularizer
            accuracy tracker. They exist after :meth:`compile`.
        """

        metric_trackers = [
            self.total_loss_tracker, 
            self.noise_loss_tracker, 
        ]
        # Expose split trackers only when split-noise reporting is enabled.
        if self.show_separate_noise_losses:
            metric_trackers.extend([
                self.cond_noise_loss_tracker,
                self.uncond_noise_loss_tracker,
            ])
        metric_trackers.extend([
            self.image_loss_tracker, 
            self.kl_loss_tracker, 
            self.ctr_loss_tracker, 
            self.ctr_accuracy_tracker
        ])

        return metric_trackers

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

        self.total_loss_tracker = metrics.Mean(name="loss")
        self.noise_loss_tracker = metrics.Mean(
            name="total_noise_loss" if self.show_separate_noise_losses \
                else "noise_loss"
        )
        self.cond_noise_loss_tracker = metrics.Mean(
            name="cond_noise_loss"
        )
        self.uncond_noise_loss_tracker = metrics.Mean(
            name="uncond_noise_loss"
        )
        self.image_loss_tracker = metrics.Mean(name="image_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        self.ctr_loss_tracker = metrics.Mean(name="ctr_loss")
        self.ctr_accuracy_tracker = metrics.SparseCategoricalAccuracy(
            name="ctr_accuracy"
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
                already_prepared = isinstance(
                    element_spec, (tuple, list)
                ) and len(element_spec) > 2
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
                the seven prepared tensors when ``map_preprocess=True``.

        Returns:
            dict[str, tf.Tensor]: Running enabled loss/accuracy metrics.  Noise
            loss is always present; total/image/KL/regularizer values appear
            according to active loss flags.
        """

        prepared_inputs = self.prep_inputs(
            inputs
        ) if not self.map_preprocess else inputs
        (x0, noises, t, x_t, cfg_labels, 
        uncond_labels, classes) = prepared_inputs

        with tf.GradientTape() as tape:
            (loss, noise_loss, cond_noise_loss, 
            uncond_noise_loss, image_loss, kl_loss, 
            ctr_loss, ctr_preds) = self.forward_and_compute_loss(
                "raw", x0, noises, t, x_t, 
                cond_labels=cfg_labels, 
                uncond_labels=uncond_labels, 
                classes=classes, 
                cfg_scale=self.train_cfg_scale, 
                training=True
            )

        self.apply_grads(tape, loss)
        self.update_ema()
        results = self.get_results_dict(
            noise_loss, 
            cond_noise_loss=cond_noise_loss, 
            uncond_noise_loss=uncond_noise_loss, 
            total_loss=loss, 
            image_loss=image_loss, 
            kl_loss=kl_loss, 
            ctr_loss=ctr_loss, 
            ctr_preds=ctr_preds, 
            classes=classes,
            cond_labels=cfg_labels,
        )

        return results

    def test_step(
        self, 
        inputs: tuple[tf.Tensor, ...]
    ) -> dict[str, tf.Tensor]:
        """Evaluate one batch using the configured raw/EMA test network.

        Args:
            inputs (tuple[tf.Tensor, ...]): Clean images and integer classes, or
                the seven prepared tensors when ``map_preprocess=True``.

        Returns:
            dict[str, tf.Tensor]: Running evaluation metrics.  Image loss is
            explicitly evaluated even when its training coefficient is zero.
        """

        prepared_inputs = self.prep_inputs(
            inputs, use_label_dropout=False
        ) if not self.map_preprocess else inputs
        (x0, noises, t, x_t, cond_labels, 
        uncond_labels, classes) = prepared_inputs

        (loss, noise_loss, cond_noise_loss, 
        uncond_noise_loss, image_loss, kl_loss, 
        ctr_loss, ctr_preds) = self.forward_and_compute_loss(
            self.test_network_name, 
            x0, noises, t, x_t, 
            cond_labels=cond_labels, 
            uncond_labels=uncond_labels, 
            classes=classes, 
            cfg_scale=self.test_cfg_scale, 
            use_image_loss=True, 
            training=False
        )

        results = self.get_results_dict(
            noise_loss, 
            cond_noise_loss=cond_noise_loss, 
            uncond_noise_loss=uncond_noise_loss, 
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

        assert "epochs" not in fit_kwargs and "initial_epoch" not in fit_kwargs, \
            "Do not pass epochs/initial_epoch to fit_progressively(); "\
            "use stage_epochs and final_epochs instead."
        assert timestep_clustering_type in get_args(ClusteringType), \
                    "timestep_clustering_type must be one of "\
                    f"{get_args(ClusteringType)} but not "\
                    f"{timestep_clustering_type}."
        assert pacing_type in (vals:=("fixed", "plateau")), \
            f"pacing_type must be one of {vals} but not {pacing_type}."
        assert earlystopping_type in (vals:=("batch_wise", "epoch_wise")), \
            f"earlystopping_type must be one of {vals} but not {earlystopping_type}."

        # Follow the opt-in aggregate metric rename for progressive callbacks.
        if self.show_separate_noise_losses and \
        monitor.removeprefix("val_") == "noise_loss":
            monitor = monitor.replace("noise_loss", "total_noise_loss")
        assert monitor.removeprefix("val_") in (vals:=self.metrics_names), \
            f"monitor must be one of {vals} (or with val_) but not {monitor}."

        only_task = stage_tasks if stage_tasks in (
            "timesteps_only", "resolutions_only", "depths_only"
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
        min_timesteps: int | None = None, 
        max_timesteps: int | None = None
    ) -> None:
        """Set the active half-open timestep interval for forward noising.

        Args:
            min_timesteps (int | None): Inclusive lower bound; ``None`` means 0.
            max_timesteps (int | None): Exclusive upper bound; ``None`` means
                the current full schedule length.

        Returns:
            None: Changed bounds invalidate cached Keras train/test/predict
            functions so traced random ranges are rebuilt.

        Raises:
            AssertionError: Unless ``0 <= min < max <= timesteps``.
        """
        min_timesteps = 0 if min_timesteps is None else min_timesteps
        max_timesteps = self.timesteps if max_timesteps is None else max_timesteps


        assert isinstance(min_timesteps, int) and \
            not isinstance(min_timesteps, bool), \
            "min_timesteps must be an integer."
        assert isinstance(max_timesteps, int) and \
            not isinstance(max_timesteps, bool), \
            "max_timesteps must be an integer."
        assert 0 <= min_timesteps < max_timesteps <= self.timesteps, \
            "Expected 0 <= min_timesteps < max_timesteps <= timesteps, "\
            f"got [{min_timesteps}, {max_timesteps}) with T={self.timesteps}."


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

        resolution = int(resolution)
        # Propagate a changed resolution to the raw and EMA networks.
        if getattr(self, "_current_resolution", None) != resolution:
            self._current_resolution = resolution

            self.train_function = None
            self.test_function = None
            self.predict_function = None

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
            None: ``self.schedules`` maps schedule-statistic names to float32
            rank-1 tensors and schedule metadata attributes are updated.
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
        schedules = {
            key: tf.constant(value, dtype=tf.float32)
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
            dtype ``tf.float32``.
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

        a, b = self.get_noise_and_signal_rates(t)
        a = tf.cast(tf.reshape(a, (-1, 1, 1, 1)), x0.dtype)
        b = tf.cast(tf.reshape(b, (-1, 1, 1, 1)), x0.dtype)
        noises = tf.cast(noises, x0.dtype)

        return a * x0 + b * noises

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
        seed = self.seed if seed is None else seed

        x_shape = tf.shape(x0)

        t = tf.random.uniform(
            (x_shape[0],), 
            minval=min_timesteps, 
            maxval=max_timesteps, 
            dtype=tf.int32, 
            seed=seed
        ) if t is None else t
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

    def get_network(self, network_name: NetworkName) -> ArgumentSaverModel:
        """Resolve the raw or EMA prediction network by name.

        Args:
            network_name (NetworkName): Exactly ``"raw"`` or ``"ema"``.

        Returns:
            DiffusionTransformer: Selected network instance.
        """

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
                "network_name needs to be one of {NetworkName}, "
                f"but not: {network_name}"
            )

        return network

    def update_ema(self) -> bool:
        """Update every EMA weight from its aligned raw-network weight.

        Returns:
            bool: False when EMA is disabled; true after a successful update.

        Raises:
            AssertionError: If raw and EMA topologies have different weight
                counts.
        """

        # Report that no EMA update occurred when EMA is disabled.
        if not self.use_ema:
            return False


        assert len(self.network.weights) == len(self.ema_network.weights), \
            "Raw and EMA networks must have the same topology."


        for w, ew in zip(self.network.weights, self.ema_network.weights):
            ew.assign(self.ema_decay * ew + (1 - self.ema_decay) * w)

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

        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

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
            ``swap_noise_image=True``, the returned ``noises`` target is ``x_t``.
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
            :meth:`prep_inputs`.
        """

        return self.prep_inputs(
            (x0, labels),
            use_label_dropout=self._preprocess_training is not False
        )

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

        # Compute reporting-only conditional losses without changing `loss`.
        if self.show_separate_noise_losses:
            assert cond_labels is not None, \
                "cond_labels are required to show separate noise losses."


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
                tf.boolean_mask(noises_pred, cond_mask),
            )
            cond_noise_loss = tf.where(
                cond_has_rows,
                cond_noise_loss,
                tf.zeros_like(cond_noise_loss),
            )
            uncond_has_rows = tf.reduce_any(uncond_mask)
            uncond_noise_loss = self.compiled_loss(
                tf.boolean_mask(noises, uncond_mask),
                tf.boolean_mask(noises_pred, uncond_mask),
            )
            uncond_noise_loss = tf.where(
                uncond_has_rows,
                uncond_noise_loss,
                tf.zeros_like(uncond_noise_loss),
            )
        # Leave both reporting losses absent when the feature is disabled.
        else:
            cond_noise_loss = None
            uncond_noise_loss = None

        return cond_noise_loss, uncond_noise_loss

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
        ctr_num = 0
        ctr_loss = 0.
        ctr_preds = tf.zeros((
            tf.shape(classes)[0], 
            self.network.num_classes
        ))

        for classes_pred in classes_pred_list:
            # Include each available regularizer prediction in the ensemble.
            if classes_pred is not None:
                ctr_num += 1
                ctr_preds += classes_pred

        # Average available predictions before computing token classification loss.
        if ctr_num > 0:
            ctr_preds /= ctr_num
            ctr_loss = tf.reduce_mean(self.scce_loss_fn(
                classes, 
                ctr_preds
            ))            

        return ctr_loss, ctr_preds

    def compute_noise_image_kl_ctr_loss(
        self, 
        x0: tf.Tensor, 
        noises: tf.Tensor, 
        classes: tf.Tensor, 
        x0_pred: tf.Tensor, 
        noises_pred: tf.Tensor, 
        z_vals_c: tuple[tf.Tensor, tf.Tensor], 
        regs_list_c: list[tf.Tensor], 
        z_vals_u: tuple[tf.Tensor, tf.Tensor] | None = None, 
        regs_list_u: list[tf.Tensor] | None = None,
        kl_train_type: TrainType | None = None, 
        ctr_train_type: TrainType | None = None, 
        use_image_loss: bool | None = None, 
        cond_labels: tf.Tensor | None = None
    ) -> tuple[
        tf.Tensor, tf.Tensor, tf.Tensor | None, tf.Tensor | None,
        tf.Tensor | float, tf.Tensor | float, tf.Tensor | float, tf.Tensor
    ]:
        """Compute and weight diffusion, reconstruction, KL, and token losses.

        Args:
            x0 (tf.Tensor): Clean images ``[B,H,W,C]``.
            noises (tf.Tensor): Noise target matching ``x0``.
            classes (tf.Tensor): Zero-based classes ``[B]``.
            x0_pred (tf.Tensor): Reconstructed clean images matching ``x0``.
            noises_pred (tf.Tensor): Guided noise prediction matching ``noises``.
            z_vals_c (tuple[tf.Tensor | None, tf.Tensor | None]): Conditional
                latent mean/log variance.
            regs_list_c (list[tf.Tensor | None]): Conditional auxiliary class
                probabilities by depth.
            z_vals_u (tuple[tf.Tensor | None, tf.Tensor | None] | None):
                Unconditional latent statistics, required when KL trains uncond.
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

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor | None, tf.Tensor | None,
            tf.Tensor | float, tf.Tensor | float, tf.Tensor | float, tf.Tensor]:
            Weighted total, raw noise loss, optional conditional/unconditional
            noise losses, image loss, KL loss, class-token loss, and averaged
            token predictions.
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
        )
        image_loss = self.compiled_loss(
            x0, 
            x0_pred
        ) if use_image_loss else 0.
        kl_loss = VariationalAutoencoder.compute_kl(
            z_mean=z_vals_c[0] if kl_train_type == "cond" else z_vals_u[0], 
            z_log_var=z_vals_c[1] if kl_train_type == "cond" else z_vals_u[1]
        ) if self.use_kl_loss else 0.
        ctr_loss, ctr_preds = self.compute_ctr_loss(
            classes, 
            regs_list_c if ctr_train_type == "cond" else regs_list_u
        ) if self.use_ctr_loss else (0., 0.)

        loss = (
            noise_loss * self.noise_loss_coef + 
            image_loss * self.image_loss_coef + 
            kl_loss * self.kl_loss_coef + 
            ctr_loss * self.ctr_loss_coef
        )

        outputs = (
            loss, noise_loss, cond_noise_loss, 
            uncond_noise_loss, image_loss, kl_loss, 
            ctr_loss, ctr_preds
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
        tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor] | None]
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
            tuple: ``((eps_c, eps_u), (regs_c, regs_u), (z_c, z_u))``.  Noise
            predictions are ``[B,H,W,C]``; unconditional members are None when
            no second pass runs.
        """
        network = self.get_network(network_name)


        def run_network(
            labels: tf.Tensor,
        ) -> tuple[tf.Tensor, list[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
            """Run one conditional-label branch of the selected network.

            Args:
                labels (tf.Tensor): Integer condition IDs of shape ``[B]``.

            Returns:
                tuple[tf.Tensor, list[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
                Noise prediction, auxiliary class predictions, and latent
                mean/log-variance tensors for the selected label branch.
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
                    outputs["z_vals"]
                )

            eps, *_, regs_list, z_vals = network(
                (x_t, t_batch, labels), 
                full_return=True, 
                training=training
            )

            return eps, regs_list, z_vals


        eps_c, regs_list_c, z_vals_c = run_network(cond_labels)
        eps_u, regs_list_u, z_vals_u = run_network(uncond_labels) \
                                    if self.use_cfg and scale is not None \
                                    else (None, None, (None, None))

        return ((eps_c, eps_u), 
                (regs_list_c, regs_list_u), 
                (z_vals_c, z_vals_u))

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
            sqrt_a_t = tf.reshape(sqrt_a_t, (-1, 1, 1, 1))
            sqrt_one_minus_a_t = tf.reshape(sqrt_one_minus_a_t, (-1, 1, 1, 1))

        x0 = (x_t - sqrt_one_minus_a_t * eps) / sqrt_a_t

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
        tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor] | None]
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
            tuple: ``(x0, eps, (regs_c, regs_u), (z_c, z_u))``.  Image tensors
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
        cfg_scale: float, 
        kl_train_type: TrainType | None = None, 
        ctr_train_type: TrainType | None = None, 
        use_image_loss: bool | None = None, 
        training: bool | None = None
    ) -> tuple[
        tf.Tensor, tf.Tensor, tf.Tensor | None, tf.Tensor | None,
        tf.Tensor | float, tf.Tensor | float, tf.Tensor | float, tf.Tensor
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
            network_name, 
            x_t, t, t, 
            cond_labels=cond_labels, 
            uncond_labels=uncond_labels, 
            scale=cfg_scale, 
            training=training
        )
        outputs = self.compute_noise_image_kl_ctr_loss(
            x0, noises, classes, 
            x0_pred, noises_pred, 
            z_vals_c=others[1][0], 
            regs_list_c=others[0][0], 
            z_vals_u=others[1][1], 
            regs_list_u=others[0][1], 
            kl_train_type=kl_train_type, 
            ctr_train_type=ctr_train_type, 
            use_image_loss=use_image_loss, 
            cond_labels=cond_labels
        )

        return outputs

    def get_results_dict(
        self, 
        noise_loss: tf.Tensor, 
        cond_noise_loss: tf.Tensor | None = None, 
        uncond_noise_loss: tf.Tensor | None = None, 
        total_loss: tf.Tensor | None = None, 
        image_loss: tf.Tensor | None = None, 
        kl_loss: tf.Tensor | None = None, 
        ctr_loss: tf.Tensor | None = None, 
        ctr_preds: tf.Tensor | None = None, 
        classes: tf.Tensor | None = None, 
        cond_labels: tf.Tensor | None = None,
        use_total_loss: bool | None = None, 
        use_image_loss: bool | None = None, 
        use_kl_loss: bool | None = None, 
        use_ctr_loss: bool | None = None
    ) -> dict[str, tf.Tensor]:
        """Update enabled diffusion metric trackers and return their results.

        Args:
            noise_loss (tf.Tensor): Required scalar noise loss.
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
            use_image_loss (bool | None): Explicit image tracker switch.
            use_kl_loss (bool | None): Explicit KL tracker switch.
            use_ctr_loss (bool | None): Explicit token loss/accuracy switch.

        Returns:
            dict[str, tf.Tensor]: Current running metric values keyed by tracker
            names.

        Raises:
            AssertionError: If an enabled metric's required value is missing.
        """

        use_image_loss = self.use_image_loss if use_image_loss is None else use_image_loss
        use_kl_loss = self.use_kl_loss if use_kl_loss is None else use_kl_loss
        use_ctr_loss = self.use_ctr_loss if use_ctr_loss is None else use_ctr_loss
        use_total_loss = use_image_loss or use_kl_loss or use_ctr_loss \
                        if use_total_loss is None else use_total_loss
        batch_weight = tf.cast(tf.shape(classes)[0], tf.float32) \
                       if classes is not None else 1.

        results = {}

        # Update the total-loss tracker only when that loss was requested.
        if use_total_loss:
            assert total_loss is not None, \
                "When use_total_loss is True, total_loss cannot be None."


            self.total_loss_tracker.update_state(
                total_loss, sample_weight=batch_weight
            )
            results.update({
                self.total_loss_tracker.name: 
                self.total_loss_tracker.result()
            })

        self.noise_loss_tracker.update_state(
            noise_loss, sample_weight=batch_weight
        )
        results.update({
            self.noise_loss_tracker.name: 
            self.noise_loss_tracker.result(), 
        })       

        # Update optional split means using sample counts rather than batches.
        if cond_noise_loss is not None and uncond_noise_loss is not None:
            cond_weight = 1.
            uncond_weight = 1.
            # Derive conditional/null population sizes when labels are available.
            if cond_labels is not None:
                cond_mask = cond_labels != 0 if self.use_cfg else tf.ones_like(
                    cond_labels, dtype=tf.bool
                )
                cond_weight = tf.reduce_sum(tf.cast(cond_mask, tf.float32))
                uncond_weight = tf.reduce_sum(tf.cast(
                    tf.logical_not(cond_mask), tf.float32
                ))

            self.cond_noise_loss_tracker.update_state(
                cond_noise_loss, sample_weight=cond_weight
            )
            self.uncond_noise_loss_tracker.update_state(
                uncond_noise_loss, sample_weight=uncond_weight
            )
            results.update({
                self.cond_noise_loss_tracker.name:
                self.cond_noise_loss_tracker.result(), 
                self.uncond_noise_loss_tracker.name:
                self.uncond_noise_loss_tracker.result()
            })

        # Update image reconstruction metrics only when image loss is active.
        if use_image_loss:
            assert image_loss is not None, \
                "When use_image_loss is True, image_loss cannot be None."


            self.image_loss_tracker.update_state(
                image_loss, sample_weight=batch_weight
            )
            results.update({
                self.image_loss_tracker.name: 
                self.image_loss_tracker.result()
            })

        # Update the KL tracker only when a KL objective is active.
        if use_kl_loss:
            assert kl_loss is not None, \
                "When use_kl_loss is True, kl_loss cannot be None."


            self.kl_loss_tracker.update_state(
                kl_loss, sample_weight=batch_weight
            )
            results.update({
                self.kl_loss_tracker.name: 
                self.kl_loss_tracker.result(), 
            })

        # Update class-token metrics only when their objective is active.
        if use_ctr_loss:
            assert ctr_loss is not None and ctr_preds is not None and \
                classes is not None, "When use_ctr_loss is True, "\
                "ctr_loss, ctr_preds, and classes cannot be None."


            self.ctr_loss_tracker.update_state(
                ctr_loss, sample_weight=batch_weight
            )
            self.ctr_accuracy_tracker.update_state(
                classes, 
                ctr_preds
            )
            results.update({
                self.ctr_loss_tracker.name: 
                self.ctr_loss_tracker.result(), 
                self.ctr_accuracy_tracker.name: 
                self.ctr_accuracy_tracker.result(), 
            })

        return results

    def sample_vae(
        self, 
        network_name: NetworkName = "ema", 
        labels: tf.Tensor| list | None = None, 
        z: tf.Tensor | None = None, 
        seed: int | None = None
    ) -> tf.Tensor:
        """Generate images by decoding the configured variational bottleneck.

        The first ``"flatten"`` reshaper is treated as the encoder/decoder
        boundary.  Sampling resumes the raw network at that depth, so later
        connections may not route around the boundary to earlier features.

        Args:
            network_name (NetworkName): ``"ema"`` or ``"raw"`` decoder network.
            labels (tf.Tensor | list[int] | None): Condition IDs, one per sample.
                In dynamic mode, ``None`` shifts saved zero-based targets to
                condition IDs and excludes the CFG null label. Fixed-width
                mode likewise samples each class condition once and excludes
                the CFG null label. Explicit values are already network label
                IDs, not unshifted dataset classes.
            z (tf.Tensor | None): Latent batch ``[B, latent_width]``.  ``None``
                draws standard normal values; its batch size must match labels.
            seed (int | None): Latent random seed; None uses ``self.seed``.

        Returns:
            tf.Tensor: Decoded, postprocessed images ``[B,H,W,C]`` in ``[0,1]``.

        Raises:
            ValueError: If no flatten reshaper exists, it is not KL-enabled, or
                a later connection bypasses the bottleneck.
        """

        network = self.get_network(network_name)
        z_id = None
        for id_, type_ in network.reshaper_ids_dict.items():
            # Locate the first flattening reshaper that defines the latent boundary.
            if type_ == "flatten":
                z_id = int(id_)
                break

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

        reshaper = network.layers_dicts[z_id-1][network.R]
        z_projector = reshaper.get_layer(
            f"{network.name_prefix}depth_{z_id}_{network.R[2:]}/z"
        ) if network.reshaper_kwargs.get("latent_dim_ratio", 1) != 1 else None

        default_labels = [
            value + int(network.use_cfg)
            for value in self.seen_classes.values()
        ] if network.dynamic_num_classes else list(
            range(int(network.use_cfg), network.num_labels)
        )
        labels = tf.cast(tf.convert_to_tensor(
            default_labels if labels is None else labels
        ), tf.int32)
        # Require one condition label per generated sample.
        if labels.shape.rank != 1:
            raise ValueError(
                "labels must be a one-dimensional tensor or list."
            )
        n = len(labels)
        seed = self.seed if seed is None else seed
        ts = tf.zeros(
            shape=(n,), 
            dtype=tf.int32
        )
        z = tf.random.normal(
            shape=(
                n, 
                reshaper.output_shape[1][-1]
            ), 
            mean=0., 
            stddev=1., 
            seed=seed
        ) if z is None else z
        z = z_projector(
            z, 
            training=False
        ) if z_projector is not None else z

        images = network(
            (z, ts, labels), 
            min_depth=z_id, 
            training=False
        )
        images = self.postprocess(images)

        return images

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

        # Route sampling through the variational decoder in swapped-objective mode.
        if self.swap_noise_image:
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

        labels = default_labels if labels is None else labels
        n = len(labels)
        seed = self.seed if seed is None else seed
        x_t = tf.random.normal((
            n, 
            self._current_resolution, # self.image_size, 
            self._current_resolution, # self.image_size, 
            self.channels
        ), seed=seed) if x_t is None else x_t
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
        steps = int(steps)
        eta = float(eta)
        ts = np.linspace(
            0, self.timesteps-1, 
            num=steps, 
            dtype="int32"
        )[::-1]
        cond_labels = tf.constant(labels)
        uncond_labels = tf.zeros((n,), dtype=tf.uint8)
        
        steps = len(ts)
        x0s, x_ts = [], []
        for i in range(steps):
            # Report reverse-diffusion progress when requested.
            if verbose:
                print(f"\rSteps: {i+1}/{steps}", end="")

            t = ts[i]
            t_next = ts[i + 1] if i < len(ts) - 1 else 0
            t_batch = tf.fill((n,), t)

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
                    dtype=tf.float32
                )
                eps_coeff = tf.cast(
                    tf.sqrt(tf.maximum(
                            1. - alpha_bar_t_next - sigma_t ** 2, 0.0
                    )),
                    dtype=tf.float32
                )

                x_t = x0_coef * x0 + eps_coeff * eps
                # Add stochastic DDIM noise when eta is positive.
                if eta > 0.:
                    x_t += sigma_t * tf.random.normal(
                        tf.shape(x_t),
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
    assert wrapper.use_ema and wrapper.ema_network is not wrapper.network
    assert len(wrapper.network.weights) == len(wrapper.ema_network.weights)
    for raw_weight, ema_weight in zip(
        wrapper.network.weights, wrapper.ema_network.weights
    ):
        tf.debugging.assert_near(raw_weight, ema_weight)
    assert [metric.name for metric in wrapper.metrics] == [
        "loss", "noise_loss", "image_loss", "kl_loss", "ctr_loss",
        "ctr_accuracy",
    ]
    separate_noise_wrapper = make_wrapper(
        show_separate_noise_losses=True
    )
    assert [metric.name for metric in separate_noise_wrapper.metrics] == [
        "loss", "total_noise_loss", "cond_noise_loss",
        "uncond_noise_loss", "image_loss", "kl_loss", "ctr_loss",
        "ctr_accuracy",
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
    wrapper.set_timestep_bounds(None, None)
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
    split_losses = separate_noise_wrapper.compute_noise_image_kl_ctr_loss(
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
        no_cfg_separate_noise_wrapper.compute_noise_image_kl_ctr_loss(
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
    assert len(losses_tuple) == 8
    assert all(
        value is None or bool(tf.reduce_all(tf.math.is_finite(value)))
        for value in losses_tuple[:7]
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
        0.5 * weighted_losses[1] + 0.25 * weighted_losses[4], 
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
    ):
        try:
            wrapper.sample(
                network_name="raw",
                labels=[1],
                seed=31,
                **invalid_sample_kwargs,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Invalid sampling overrides accepted: {invalid_sample_kwargs}"
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
    assert wrapper_config["network"] is serialization_wrapper.network
    try:
        DiffusionModel.from_config(wrapper_config)
    except ValueError as error:
        assert "cannot be saved" in str(error)
    else:
        raise AssertionError(
            "Wrapper reconstruction with an embedded network currently must fail"
        )

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
    ).dtype == tf.float32

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
            DiffusionModel(network=make_network(), **overrides)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Expected invalid wrapper config: {overrides}")

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
