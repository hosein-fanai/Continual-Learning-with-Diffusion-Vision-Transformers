import tensorflow as tf
from tensorflow.keras import metrics, losses, callbacks, optimizers

import numpy as np

from typing import Literal, Sequence, get_args

from common.argument_saver import ArgumentSaverModel

from autoencoder.variational_autoencoder import VariationalAutoencoder

from . import NetworkName, TrainType, ClusteringType

from diffusion.callbacks.batch_loss_plateau import BatchLossPlateau

from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer
from diffusion.schedulers import make_schedule, SchedulerName


# @tf.keras.saving.register_keras_serializable()
class DiffusionModel(ArgumentSaverModel):
    """
    
    """

    def __init__(
        self, 
        network: DiffusionTransformer, 
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
        image_loss_coef: float = 0., 
        kl_loss_coef: float = 0., 
        ctr_loss_coef: float = 0., 
        kl_train_type: TrainType = "cond", 
        ctr_train_type: TrainType = "cond", 
        train_noisified_min_timesteps: int = 0, 
        train_noisified_max_timesteps: int | None = None, 
        test_noisified_min_timesteps: int = 0, 
        test_noisified_max_timesteps: int | None = None, 
        resize_method: str = "area", 
        resize_antialias: bool = True, 
        swap_noise_image: bool = False, 
        seed: int | None = None, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self._check_assertions(locals())
        self._save_init_args(locals())
        self._refresh_loss_flags()

        if self.use_ema:
            self.ema_network = self.network.__class__.from_config(
                self.network.get_config()
            )
            self.ema_network.set_weights(
                self.network.get_weights()
            )
        else:
            self.ema_network = None

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
        self.train_noisified_max_timesteps = self.timesteps if self.train_noisified_max_timesteps is None \
                                            else self.train_noisified_max_timesteps
        self.test_noisified_max_timesteps = self.timesteps if self.test_noisified_max_timesteps is None \
                                            else self.test_noisified_max_timesteps
        self.use_image_loss = bool(self.image_loss_coef > 0.)

        self.load_schedules()
        self.set_timestep_bounds()
        self.set_current_resolution()
        self.build(())

    def _check_assertions(self, local_vars: dict):
        assert 0. <= local_vars["ema_decay"] < 1., \
            "ema_decay must be in the range of [0., 1.)."

        assert 2 <= local_vars["test_steps"] <= local_vars["network"].timesteps, \
            "steps must be in the range of [2, timesteps]."

        assert 0. <= local_vars["test_eta"] <= 1., \
            "eta must be in the range of [0., 1.]."

        assert local_vars["ctr_train_type"] in get_args(TrainType), \
            f"ctr_train_type can be one of {TrainType}."

        if local_vars["ctr_train_type"] == "uncond":
            assert local_vars["train_cfg_scale"] is not None, \
                "ctr_train_type can be uncond only when train_cfg_scale is not None."

    def _get_progressive_timestep_boundaries(
        self, 
        stages_num: int, 
        clustering_type: ClusteringType = "log_snr", 
    ) -> list[int]:
        """Return N+1 monotonically increasing curriculum boundaries.

        ``uniform`` reproduces the simple equal-timestep partition.

        ``log_snr`` is a practical SNR-aware partition: boundaries are chosen
        at approximately equal intervals of log-SNR under the *existing full*
        diffusion schedule.  It keeps the original T and schedule unchanged;
        only the timesteps sampled for a curriculum stage are restricted.
        """

        assert 1 <= stages_num <= self.timesteps, \
            f"num_stages must be in [1, {self.timesteps}] range, "\
            f"but got {stages_num}."


        if clustering_type == "uniform":
            boundaries = np.rint(
                np.linspace(0, self.timesteps, stages_num + 1)
            ).astype(np.int32)
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
        else:
            raise ValueError(
                f"clustering must be one of {ClusteringType}."
            )

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
            optimizer: Optimizer that should know the current variable set, or
                ``None`` to use ``self.optimizer`` when it exists.
            variables: Variables to register, or ``None`` for every trainable
                variable in the raw diffusion network.

        Returns:
            ``None``. If no optimizer exists yet, the method has no effect.
        """

        optimizer = getattr(self, "optimizer", None) if optimizer is None else optimizer
        variables = self.network.trainable_variables if variables is None else variables

        if optimizer is None:
            return

        if hasattr(optimizer, "_create_all_weights"):
            optimizer._create_all_weights(variables)
        elif hasattr(optimizer, "build"):
            optimizer.build(variables)
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
            depth_spec: A depth specification accepted by the wrapped network.

        Returns:
            The wrapped network's branch-wise before/added/after depth report.

        Raises:
            ValueError: If raw and EMA growth creates different numbers or
                shapes of weights.
        """

        raw_weight_ids = {id(weight) for weight in self.network.weights}
        ema_weight_ids = {id(weight) for weight in self.ema_network.weights} \
                        if self.ema_network is not None else set()

        growth = self.network.add_depths(depth_spec)
        self.network.build_model()

        if self.ema_network is not None:
            self.ema_network.add_depths(
                depth_spec
            )
            self.ema_network.build_model()

            raw_weights = [
                weight for weight in self.network.weights
                if id(weight) not in raw_weight_ids
            ]
            ema_weights = [
                weight for weight in self.ema_network.weights
                if id(weight) not in ema_weight_ids
            ]

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

    @property
    def current_timesteps_bounds(self) -> tuple[int, int]:
        return self._active_min_timestep, self._active_max_timestep

    @property
    def current_resolution(self) -> tuple[int, int]:
        """Return the square image resolution currently processed.

        Returns:
            The active positive integer resolution of both the 
            Wrapper and the Network.
        """

        return self._current_resolution, self.network.current_resolution

    @property
    def metrics(self):
        return [
            self.total_loss_tracker, 
            self.noise_loss_tracker, 
            self.image_loss_tracker, 
            self.kl_loss_tracker, 
            self.ctr_loss_tracker, 
            self.ctr_accuracy_tracker
        ]

    def compile(self, loss: losses.Loss | str = "mse", 
                **kwargs):
        super().compile(loss=loss, **kwargs)

        self.scce_loss_fn = losses.sparse_categorical_crossentropy

        self.total_loss_tracker = metrics.Mean(name="loss")
        self.noise_loss_tracker = metrics.Mean(name="noise_loss")
        self.image_loss_tracker = metrics.Mean(name="image_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        self.ctr_loss_tracker = metrics.Mean(name="ctr_loss")
        self.ctr_accuracy_tracker = metrics.SparseCategoricalAccuracy(name="ctr_accuracy")

    def fit(self, x: tf.data.Dataset | None = None, 
            y: tf.data.Dataset | None = None, **kwargs) -> dict | list[float]:
        prev_t_min = self._active_min_timestep
        prev_t_max = self._active_max_timestep

        self.set_timestep_bounds(
            self.train_noisified_min_timesteps, 
            self.train_noisified_max_timesteps, 
        )

        fit_results = super().fit(x=x, y=y, **kwargs)

        self.set_timestep_bounds(
            prev_t_min, 
            prev_t_max, 
        )

        return fit_results

    def evaluate(self, x: tf.data.Dataset | None = None, 
                y: tf.data.Dataset | None = None, 
                network_name: NetworkName = "ema", 
                **kwargs) -> dict | list[float]:
        prev_t_min = self._active_min_timestep
        prev_t_max = self._active_max_timestep
        prev_test_network_name = self.test_network_name

        self.set_timestep_bounds(
            self.test_noisified_min_timesteps, 
            self.test_noisified_max_timesteps, 
        )

        if network_name != self.test_network_name:
            self.test_network_name = network_name
            self.test_function = None

        eval_results = super().evaluate(x=x, y=y, **kwargs)

        self.set_timestep_bounds(
            prev_t_min, 
            prev_t_max, 
        )

        if prev_test_network_name != self.test_network_name:
            self.test_network = prev_test_network_name
            self.test_function = None

        return eval_results

    def summary(self, **kwargs):
        return self.network.summary(**kwargs)

    def train_step(self, inputs: tuple[tf.Tensor, tf.Tensor]
                ) -> dict:
        (x0, noises, 
        t, x_t, 
        cfg_labels, 
        uncond_labels, 
        classes) = self.prep_inputs(inputs)

        with tf.GradientTape() as tape:
            (loss, noise_loss, image_loss, 
            kl_loss, ctr_loss, ctr_preds) = self.forward_and_compute_loss(
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
            total_loss=loss, 
            image_loss=image_loss, 
            kl_loss=kl_loss, 
            ctr_loss=ctr_loss, 
            ctr_preds=ctr_preds, 
            classes=classes
        )

        return results

    def test_step(self, inputs: tuple[tf.Tensor, tf.Tensor]
                ) -> dict:
        (x0, noises, 
        t, x_t, 
        cond_labels, 
        uncond_labels, 
        classes) = self.prep_inputs(inputs)

        (loss, noise_loss, image_loss, 
        kl_loss, ctr_loss, ctr_preds) = self.forward_and_compute_loss(
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
            total_loss=loss, 
            image_loss=image_loss, 
            kl_loss=kl_loss, 
            ctr_loss=ctr_loss, 
            ctr_preds=ctr_preds, 
            classes=classes, 
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
                depths=["vit_block", {"local_mixer", "vit_block"}], 
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
                "vit_block",
                {"connection": {"ids": [-1]}, "local_mixer": True},
            ]

        Supported names are ``connection``, ``cross_attention``,
        ``vit_block``/``decoder_block``, ``local_mixer``, ``downsample``,
        ``upsample``, ``reshaper``, and ``cls_token_regularizer``. The existing
        model-wide layer kwargs are reused. Connector dictionaries may provide
        ``ids``; transformer block dictionaries may provide ``use_decoder``
        and ``mlp_output_dim``; a reshaper value is ``"flatten"`` or
        ``"unflatten"``. Added sequences must leave the final feature shape
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
            depths = [None, None, None, "vit_block"]

        This produces stages ``(700: 1000, 16)``, ``(300: 1000, 16)``,
        ``(300: 1000, 32)``, and ``(0: 1000, 64)``, then appends a transformer
        block. No direction, native-size ceiling, or implicit priority between
        the strategies is imposed.

        Args:
            stage_tasks: A list of ordered stage descriptions, or
                ``"timesteps_only"``, ``"resolutions_only"``, or
                ``"depths_only"``. A list's length is the number of training
                stages. Strings and two-item tuples change one value; sets and
                dictionaries may combine all three progressive operations.
            stages_num: Optional number of generated stages. For an explicit
                mixed task list, its length determines the stage count. In
                either ``*_only`` mode, supplied values determine the count.
                ``stages_num`` is therefore needed only when values must be
                generated.
            stages_verbose: Whether to print each stage's resolved state.
            stage_epochs: Number of epochs allocated to every listed stage.
                With plateau pacing, this is the maximum for each stage.
            final_epochs: Epochs for a final full-timestep, native-resolution
                stage. ``None`` uses ``stage_epochs`` and ``0`` disables it.
            timestep_boundaries: Optional stage-indexed sequence of
                ``(lower_bound, upper_bound)`` pairs. An entry is read only when
                the corresponding task requests ``"timesteps"`` without an
                inline pair, so unused positions may be ``None``. When omitted,
                cumulative easy-to-hard ranges are generated from ``stages_num``.
            timestep_clustering_type: It is only used when the method automatically 
                generates timestep boundaries, and it can be one of ('uniform', 'log_snr').
            resolutions: Optional stage-indexed resolution values. An entry is
                read only when the corresponding task requests ``"resolution"``
                without an inline value, so unused positions may be ``None``.
                Values may increase, decrease, repeat, or exceed ``image_size``;
                the network's normal resolution requirements still apply. When
                omitted, ``stages_num`` low-to-high resolutions are generated by
                repeatedly dividing ``image_size`` by powers of two.
            depths: Optional stage-indexed depth specifications. An entry is
                read only when the corresponding task requests ``"depth"``
                without an inline value. A specification may add any number of
                supported layer dictionaries to ``network.layers_dicts``.
                Appended depths persist after this method returns.
            pacing_type: ``"fixed"`` always runs ``stage_epochs``. ``"plateau"``
                may advance sooner using the selected early-stopping callback.
            earlystopping_type: Under plateau pacing, ``"epoch_wise"`` uses
                Keras ``EarlyStopping`` and ``"batch_wise"`` uses
                ``BatchLossPlateau``.
            monitor: Metric name monitored by plateau pacing.
            patience: Number of non-improving epochs or batches tolerated by
                the selected early-stopping callback.
            min_delta: Minimum monitored improvement.
            stopper_mode: Keras early-stopping mode used by epoch-wise pacing.
            **fit_kwargs: Normal Keras ``fit`` arguments such as ``x``,
                ``validation_data``, ``callbacks``, ``steps_per_epoch`` and
                ``verbose``. ``epochs`` and ``initial_epoch`` are managed here.

        Returns:
            A Keras ``History`` containing merged metrics and a
            ``progressive_stages`` record of every resolved stage, including
            its pre-addition network depth and any ``depth_growth`` result. The
            model's timestep bounds and resolution are restored to their entry 
            values after completion or interruption; completed structural depth
            additions are intentionally retained. Input data must be reiterable
            because each stage invokes a separate Keras ``fit`` call.
        """

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
        assert monitor.removeprefix("val_") in (vals:=self.metrics_names), \
            f"monitor must be one of {vals} (or with val_) but not {monitor}."


        only_task = stage_tasks if stage_tasks in (
            "timesteps_only", "resolutions_only", "depths_only"
        ) else None
        if only_task == "timesteps_only" and timestep_boundaries is not None:
            stages_num = len(timestep_boundaries)
        elif only_task == "resolutions_only" and resolutions is not None:
            stages_num = len(resolutions)
        elif only_task == "depths_only" and depths is not None:
            stages_num = len(depths)
        elif only_task is None:
            stages_num = len(stage_tasks)
        elif stages_num is None:
            raise ValueError(
                f"stages_num is required when {only_task!r} values are omitted."
            )

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

        if needs_timesteps and timestep_boundaries is None:
            boundaries = self._get_progressive_timestep_boundaries(
                stages_num, 
                timestep_clustering_type
            )
            timestep_boundaries = [
                (lower_bound, boundaries[-1])
                for lower_bound in reversed(boundaries[:-1])
            ]

        if needs_resolution and resolutions is None:
            resolutions = [
                self.image_size // 2**power
                for power in range(stages_num - 1, -1, -1)
            ]

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


        def run_stage(stage_id, updates, epochs, final=False):
            nonlocal epoch_cursor


            stage_callbacks = list(user_callbacks)
            if pacing_type == "plateau" and not final:
                if earlystopping_type == "epoch_wise":
                    stage_callbacks.append(callbacks.EarlyStopping(
                        monitor=monitor, 
                        min_delta=min_delta, 
                        patience=patience, 
                        mode=stopper_mode, 
                        verbose=stages_verbose
                    ))
                elif earlystopping_type == "batch_wise":
                    stage_callbacks.append(BatchLossPlateau(
                        monitor=monitor, 
                        patience=patience, 
                        min_delta=min_delta, 
                        # mode=stopper_mode
                    ))

            if stages_verbose:
                name = "final/full-task" if final \
                    else f"{stage_id}/{len(stage_tasks)}"
                print(
                    f"Progressive stage {name}: changes={updates}, "
                    f"resolution={self._current_resolution}, sampling t in "
                    f"[{self._active_min_timestep}, "
                    f"{self._active_max_timestep}) range."
                )

            history = super(DiffusionModel, self).fit(
                callbacks=stage_callbacks, 
                initial_epoch=epoch_cursor, 
                epochs=epoch_cursor + epochs, 
                **fit_kwargs,
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
                "history": history.history, 
            }
            stage_records.append(stage_record)

            return stage_record


        try:
            for stage_index, task in enumerate(stage_tasks):
                if isinstance(task, str):
                    updates = {task: None}
                elif isinstance(task, dict):
                    updates = dict(task)
                elif isinstance(task, (set, frozenset)):
                    updates = dict.fromkeys(task)
                elif (
                    isinstance(task, (tuple, list)) and len(task) == 2
                    and isinstance(task[0], str)
                    and task[0] in ("timesteps", "resolution", "depth")
                ):
                    updates = {task[0]: task[1]}
                else:
                    raise ValueError(
                        f"Invalid stage task at index {stage_index}: {task!r}."
                    )

                if "timesteps" in updates:
                    bounds = updates["timesteps"]
                    bounds = timestep_boundaries[stage_index] if bounds is None else bounds
                    bounds = tuple(bounds)
                    self.set_timestep_bounds(*bounds)
                    updates["timesteps"] = (self._active_min_timestep, self._active_max_timestep)

                if "resolution" in updates:
                    resolution = updates["resolution"]
                    resolution = resolutions[stage_index] if resolution is None else resolution
                    resolution = int(resolution)
                    self.set_current_resolution(resolution)
                    updates["resolution"] = resolution

                if "depth" in updates:
                    depth_spec = updates["depth"]
                    depth_spec = depths[stage_index] if depth_spec is None else depth_spec
                    updates["depth"] = depth_spec

                stage_record = run_stage(
                    stage_id=stage_index + 1, 
                    updates=updates, 
                    epochs=stage_epochs, 
                )

                if "depth" in updates:
                    stage_record["depth_growth"] = self._add_depths(
                        updates["depth"]
                    )
                    stage_record["post_network_depth"] = self.network.depth

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
                    final=True, 
                )
        finally:
            self.set_timestep_bounds(
                previous_min_timestep, previous_max_timestep
            )
            self.set_current_resolution(previous_resolution)

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
        max_timesteps: int | None = None, 
    ):
        min_timesteps = 0 if min_timesteps is None else min_timesteps
        max_timesteps = self.timesteps if max_timesteps is None else max_timesteps


        assert 0 <= min_timesteps < max_timesteps <= self.timesteps, \
            "Expected 0 <= min_timesteps < max_timesteps <= timesteps, "\
            f"got [{min_timesteps}, {max_timesteps}) with T={self.timesteps}."


        if getattr(self, "_active_min_timestep", None) != min_timesteps or \
        getattr(self, "_active_max_timestep", None) != max_timesteps:
            self._active_min_timestep = min_timesteps
            self._active_max_timestep = max_timesteps

            self.train_function = None
            self.test_function = None
            self.predict_function = None

    def set_current_resolution(self, resolution: int | None = None):
        resolution = self.image_size if resolution is None else resolution

        self.network.set_current_resolution(
            resolution
        )
        self.ema_network.set_current_resolution(
            resolution
        ) if self.ema_network is not None else None

        resolution = int(resolution)
        if getattr(self, "_current_resolution", None) != resolution:
            self._current_resolution = resolution

            self.train_function = None
            self.test_function = None
            self.predict_function = None

    def load_schedules(
        self, 
        scheduler_name: SchedulerName | None = None, 
        timesteps: int | None = None
    ):
        scheduler_name = self.scheduler_name if scheduler_name is None else scheduler_name
        timesteps = self.timesteps if timesteps is None else timesteps

        self.schedules = make_schedule(
            kind=scheduler_name, 
            num_steps=timesteps
        )
        self.scheduler_name = scheduler_name
        self.timesteps = timesteps

        for keys in self.schedules.keys():
            self.schedules[keys] = tf.constant(
                self.schedules[keys], 
                dtype=tf.float32
            )

        if self.modify_first_t:
            self.schedules["sqrt_alpha_bar"] = tf.tensor_scatter_nd_update(
                self.schedules["sqrt_alpha_bar"], 
                indices=[[0]], 
                updates=[1.]
            )
            self.schedules["sqrt_one_minus_alpha_bar"] = tf.tensor_scatter_nd_update(
                self.schedules["sqrt_one_minus_alpha_bar"], 
                indices=[[0]], 
                updates=[0.]
            )
            self.schedules["alpha_bar"] = tf.tensor_scatter_nd_update(
                self.schedules["alpha_bar"], 
                indices=[[0]], 
                updates=[1.]
            )

    def get_noise_and_signal_rates(self, t: int | tf.Tensor):
        a = tf.gather(self.schedules["sqrt_alpha_bar"], t)
        b = tf.gather(self.schedules["sqrt_one_minus_alpha_bar"], t)

        return a, b

    def q_sample(
        self, 
        x0: tf.Tensor, 
        t: tf.Tensor, 
        noises: tf.Tensor
    ):
        a, b = self.get_noise_and_signal_rates(t)
        a = tf.reshape(a, (-1, 1, 1, 1))
        b = tf.reshape(b, (-1, 1, 1, 1))

        return a * x0 + b * noises

    def noisify(
        self, 
        x0: tf.Tensor, 
        t: tf.Tensor | None = None, 
        min_timesteps: int | None = None, 
        max_timesteps: int | None = None, 
        seed: int | None = None
    ):
        min_timesteps = self._active_min_timestep if min_timesteps is None else min_timesteps
        max_timesteps = self._active_max_timestep if max_timesteps is None else max_timesteps
        seed = self.seed if seed is None else seed

        x_shape = tf.shape(x0)

        t = tf.random.uniform(
            (x_shape[0],), 
            minval=min_timesteps, 
            maxval=max_timesteps, 
            dtype=tf.int32, 
            seed=seed, 
        ) if t is None else t
        noises = tf.random.normal(
            x_shape, 
            mean=0., 
            stddev=1., 
            dtype=tf.float32, 
            seed=seed, 
            name="noises"
        )
        x_t = self.q_sample(x0, t, noises)

        return x_t, noises, t

    def postprocess(self, x: tf.Tensor) -> tf.Tensor:
        x = (x + 1) / 2
        x = tf.clip_by_value(x, 0., 1.)

        return x

    def get_network(self, network_name: NetworkName):
        if not self.use_ema:
            assert network_name != "ema", \
                "network_name cannot be ema when use_ema is False."


        if network_name == "ema":
            network = self.ema_network
        elif network_name == "raw":
            network = self.network
        else:
            raise ValueError(
                f"network_name needs to be one of {NetworkName}, but not: {network_name}"
            )

        return network

    def update_ema(self):
        if not self.use_ema:
            return False


        assert len(self.network.weights) == len(self.ema_network.weights), \
            "Raw and EMA networks must have the same topology."


        for w, ew in zip(self.network.weights, self.ema_network.weights):
            ew.assign(self.ema_decay * ew + (1 - self.ema_decay) * w)

        return True

    def apply_grads(self, tape: tf.GradientTape, 
                    loss: tf.Tensor, 
                    variables: list[tf.Variable]| None = None):
        if variables is None:
            variables = self.network.trainable_variables

        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

    def get_cfg_labels(self, labels: tf.Tensor, 
                       seed: int | None = None):
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

    def prep_inputs(self, inputs: tuple[tf.Tensor, tf.Tensor], 
                    use_label_dropout: bool = True, 
                    seed: int | None = None):
        x0, labels = inputs

        x0 = tf.image.resize(x0, 
            size=(
                self._current_resolution, 
                self._current_resolution
            ), 
            method=self.resize_method, 
            antialias=self.resize_antialias
        ) if self._current_resolution != self.image_size else x0

        classes = labels
        labels = labels + int(self.use_cfg)
        x_t, noises, t = self.noisify(x0, seed=seed)
        cfg_labels = self.get_cfg_labels(
            labels, 
            seed=seed
        ) if use_label_dropout else labels
        uncond_labels = tf.zeros_like(labels)

        noises = x_t if self.swap_noise_image else noises

        return x0, noises, t, x_t, cfg_labels, uncond_labels, classes

    def compute_ctr_loss(self, classes: tf.Tensor, 
                        classes_pred_list: list[tf.Tensor]
                        ) -> tuple[tf.Tensor, tf.Tensor]:
        ctr_num = 0
        ctr_loss = 0.
        ctr_preds = tf.zeros((
            tf.shape(classes)[0], 
            self.network.num_classes
        ))

        for classes_pred in classes_pred_list:
            if classes_pred is not None:
                ctr_num += 1
                ctr_preds += classes_pred

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
        regs_list_u: list[tf.Tensor] = None, 
        kl_train_type: TrainType | None = None, 
        ctr_train_type: TrainType | None = None, 
        use_image_loss: bool | None = None
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        kl_train_type = self.kl_train_type if kl_train_type is None else kl_train_type
        ctr_train_type = self.ctr_train_type if ctr_train_type is None else ctr_train_type
        use_image_loss = self.use_image_loss if use_image_loss is None else use_image_loss

        noise_loss = self.compiled_loss(
            noises, 
            noises_pred
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

        return loss, noise_loss, image_loss, kl_loss, ctr_loss, ctr_preds

    def call_network(
        self, 
        x_t: tf.Tensor, 
        t_batch: tf.Tensor, 
        cond_labels: tf.Tensor, 
        uncond_labels: tf.Tensor | None = None, 
        scale: float | None = None, 
        network_name: NetworkName = "raw", 
        training: bool = False
    ) -> tuple[tuple[tf.Tensor, tf.Tensor], 
        tuple[list[tf.Tensor], list[tf.Tensor]]]:
        network = self.get_network(network_name)

        eps_c, *_, regs_list_c, z_vals_c = network(
            (x_t, t_batch, cond_labels), 
            full_return=True, 
            training=training
        )
        eps_u, *_, regs_list_u, z_vals_u = network(
            (x_t, t_batch, uncond_labels), 
            full_return=True, 
            training=training
        ) if self.use_cfg and scale is not None else (None, None, (None, None))

        return ((eps_c, eps_u), 
                (regs_list_c, regs_list_u), 
                (z_vals_c, z_vals_u))

    def compute_eps(
        self, 
        eps_c: tf.Tensor, 
        eps_u: tf.Tensor | None = None, 
        scale: float | None = None, 
    ) -> tf.Tensor:
        if self.use_cfg and scale is not None:
            eps = eps_u + scale * (eps_c - eps_u)
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
        eps = self.compute_eps(
            eps_c, 
            eps_u, 
            scale
        )

        sqrt_a_t, sqrt_one_minus_a_t = self.get_noise_and_signal_rates(t)
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
    ):
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
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
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
            use_image_loss=use_image_loss
        )

        return outputs

    def get_results_dict(
        self, 
        noise_loss: tf.Tensor, 
        total_loss: tf.Tensor | None = None, 
        image_loss: tf.Tensor | None = None, 
        kl_loss: tf.Tensor | None = None, 
        ctr_loss: tf.Tensor | None = None, 
        ctr_preds: tf.Tensor | None = None, 
        classes: tf.Tensor | None = None, 
        use_total_loss: bool | None = None, 
        use_image_loss: bool | None = None, 
        use_kl_loss: bool | None = None, 
        use_ctr_loss: bool | None = None, 
    ) -> dict:
        use_image_loss = self.use_image_loss if use_image_loss is None else use_image_loss
        use_kl_loss = self.use_kl_loss if use_kl_loss is None else use_kl_loss
        use_ctr_loss = self.use_ctr_loss if use_ctr_loss is None else use_ctr_loss
        use_total_loss = use_image_loss or use_kl_loss or use_ctr_loss \
                        if use_total_loss is None else use_total_loss

        results = {}

        if use_total_loss:
            assert total_loss is not None, \
                "When use_total_loss is True, "\
                "total_loss cannot be None."


            self.total_loss_tracker.update_state(total_loss)
            results.update({
                self.total_loss_tracker.name: 
                self.total_loss_tracker.result()
            })

        self.noise_loss_tracker.update_state(noise_loss)
        results.update({
            self.noise_loss_tracker.name: 
            self.noise_loss_tracker.result(), 
        })

        if use_image_loss:
            assert image_loss is not None, \
                "When use_image_loss is True, "\
                "image_loss cannot be None."


            self.image_loss_tracker.update_state(image_loss)
            results.update({
                self.image_loss_tracker.name: 
                self.image_loss_tracker.result(), 
            })

        if use_kl_loss:
            assert kl_loss is not None, \
                "When use_kl_loss is True, kl_loss cannot be None."


            self.kl_loss_tracker.update_state(kl_loss)
            results.update({
                self.kl_loss_tracker.name: 
                self.kl_loss_tracker.result(), 
            })

        if use_ctr_loss:
            assert ctr_loss is not None and ctr_preds is not None and \
                classes is not None, "When use_ctr_loss is True, "\
                "ctr_loss, ctr_preds, and classes cannot be None."


            self.ctr_loss_tracker.update_state(ctr_loss)
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
    ):
        """Generate images by sampling the configured variational bottleneck."""

        network = self.get_network(network_name)
        z_id = None
        for id_, type_ in network.reshaper_ids_dict.items():
            if type_ == "flatten":
                z_id = int(id_)
                break

        if z_id is None:
            raise ValueError(
                "sample_vae requires a flatten reshaper."
            )

        if not network.reshaper_kwargs.get("add_kl", False):
            raise ValueError(
                "sample_vae requires add_kl=True in reshaper_kwargs."
            )

        for ids_dict in (
            network.connection_ids_dict, 
            network.cross_attention_ids_dict
        ):
            for depth, ids in ids_dict.items():
                if depth > z_id and any(id_ < z_id for id_ in ids):
                    raise ValueError(
                        "VAE decoder connections cannot use "
                        "features before the flatten reshaper."
                    )

        reshaper = network.layers_dicts[z_id-1][network.R]
        z_projector = reshaper.get_layer(
            f"{network.name_prefix}depth_{z_id}_{network.R[2:]}/z"
        ) if network.reshaper_kwargs.get("latent_dim_ratio", 1) != 1 else None

        labels = list(
            range(network.num_labels)
        ) if labels is None else labels
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
    ):
        """
        Generalized DDIM/DDPM sampler.

        eta = 0.0 -> deterministic DDIM
        eta = 1.0 -> DDPM-equivalent for full consecutive timesteps
        0 < eta < 1 -> stochastic DDIM
        """

        if self.swap_noise_image:
            return self.sample_vae(
                network_name=network_name, 
                labels=labels, 
                z=x_t, 
                seed=seed
            )

        labels = list(
            range(self.network.num_labels)
        ) if labels is None else labels
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

            if return_x_ts:
                x_ts.append(self.postprocess(x_t).numpy())
            if return_x0s:
                x0s.append(self.postprocess(x0).numpy())

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
            if eta > 0. and t_next > 0:
                x_t += sigma_t * tf.random.normal(
                    tf.shape(x_t), 
                    seed=seed
                )

        if verbose:
            print()

        outputs = [self.postprocess(x0)]
        if return_x_ts:
            outputs.append(x_ts)
        if return_x0s:
            outputs.append(x0s)
        if len(outputs) == 1:
            return outputs[0]

        return outputs
