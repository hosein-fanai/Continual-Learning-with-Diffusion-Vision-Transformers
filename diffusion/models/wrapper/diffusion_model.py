from typing import Literal, get_args

import tensorflow as tf
from tensorflow.keras import metrics, losses, callbacks

import numpy as np

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
        seed: int | None = None, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self._check_assertions(locals())
        self._save_init_args(locals())

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
        self.use_kl_loss = bool(self.kl_loss_coef > 0. and 
                                self.network.reshaper_kwargs.get("add_kl", False))
        self.use_ctr_loss = bool(self.ctr_loss_coef > 0. and 
                                len(self.network.cls_token_regularizer_ids) > 0)

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
        num_stages: int, 
        clustering_type: ClusteringType = "log_snr", 
    ) -> list[int]:
        """Return N+1 monotonically increasing curriculum boundaries.

        ``uniform`` reproduces the simple equal-timestep partition.

        ``log_snr`` is a practical SNR-aware partition: boundaries are chosen
        at approximately equal intervals of log-SNR under the *existing full*
        diffusion schedule.  It keeps the original T and schedule unchanged;
        only the timesteps sampled for a curriculum stage are restricted.
        """

        assert 1 <= num_stages <= self.timesteps, \
            f"num_stages must be in [1, {self.timesteps}] range, "\
            f"but got {num_stages}."


        if clustering_type == "uniform":
            boundaries = np.rint(
                np.linspace(0, self.timesteps, num_stages + 1)
            ).astype(np.int32)
        elif clustering_type == "log_snr": # This is just an estimation
            alpha_bar = np.asarray(
                self.schedules["alpha_bar"].numpy(),
                dtype=np.float64,
            )
            eps = np.finfo(np.float64).eps
            alpha_bar = np.clip(alpha_bar, eps, 1. - eps)
            log_snr = np.log(alpha_bar) - np.log1p(-alpha_bar)

            targets = np.linspace(
                log_snr[0], log_snr[-1], num_stages + 1
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

        self.load_schedules()

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

        if self._active_min_timestep != self.train_noisified_min_timesteps or \
        self._active_max_timestep != self.train_noisified_max_timesteps:
            self.set_timestep_bounds(
                self.train_noisified_min_timesteps, 
                self.train_noisified_max_timesteps, 
            )
            self.train_function = None

        fit_results = super().fit(x=x, y=y, **kwargs)

        if self._active_min_timestep != prev_t_min or \
        self._active_max_timestep != prev_t_max:
            self.set_timestep_bounds(
                prev_t_min, 
                prev_t_max, 
            )
            self.train_function = None

        return fit_results

    def evaluate(self, x: tf.data.Dataset | None = None, 
                y: tf.data.Dataset | None = None, 
                network_name: NetworkName = "ema", 
                **kwargs) -> dict | list[float]:
        prev_t_min = self._active_min_timestep
        prev_t_max = self._active_max_timestep

        if self._active_min_timestep != self.test_noisified_min_timesteps or \
        self._active_max_timestep != self.test_noisified_max_timesteps:
            self.set_timestep_bounds(
                self.test_noisified_min_timesteps, 
                self.test_noisified_max_timesteps, 
            )
            self.test_function = None

        prev_test_network_name = self.test_network_name

        if network_name != self.test_network_name:
            self.test_network_name = network_name
            self.test_function = None # to force tf recreate compute graph for test_step

        eval_results = super().evaluate(x=x, y=y, **kwargs)

        if self._active_min_timestep != prev_t_min or \
        self._active_max_timestep != prev_t_max:
            self.set_timestep_bounds(
                prev_t_min, 
                prev_t_max, 
            )
            self.test_function = None

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
        num_stages: int = 20, 
        stage_epochs: int = 1, 
        final_epochs: int | None = None, 
        clustering_type: ClusteringType = "log_snr", 
        pacing_type: Literal["fixed", "plateau"] = "fixed", 
        timestep_boundaries: list[int] | tuple[int, ...] | None = None, 
        earlystopping_type: Literal["batch_wise", "epoch_wise"] = "epoch_wise", 
        monitor: str = "noise_loss", 
        patience: int = 10, 
        min_delta: float = 0., 
        stopper_mode: str = "min", 
        verbose_stages: bool = True, 
        **fit_kwargs
    ):
        """Train with an easy-to-hard diffusion-timestep curriculum.

        Stage 1 samples only the highest-timestep (highest-noise) cluster.
        Each following stage *accumulates* one harder, lower-timestep cluster:

            [l_N, T) -> [l_{N-1}, T) -> ... -> [0, T).

        The network, optimizer, EMA weights, timestep embedding table and noise
        schedule are kept intact across stages. Only the random training
        timestep support changes. After the curriculum, ``final_epochs`` can
        continue ordinary full-range diffusion training, as done in the
        task-difficulty curriculum literature.

        ``pacing='fixed'`` trains every curriculum stage for ``stage_epochs``.
        ``pacing='plateau'`` treats ``stage_epochs`` as a maximum and advances
        when the monitored batch log has not improved for ``patience_batches``.

        Any normal Keras ``fit`` arguments (x, validation_data, callbacks, 
        steps_per_epoch, verbose, ...) are passed through ``fit_kwargs``.
        """

        assert "epochs" not in fit_kwargs and "initial_epoch" not in fit_kwargs, \
            "Do not pass epochs/initial_epoch to fit_progressively(); "\
            "use stage_epochs and final_epochs instead."


        final_epochs = stage_epochs if final_epochs is None else int(final_epochs)
        boundaries = self._get_progressive_timestep_boundaries(
            num_stages=num_stages, 
            clustering_type=clustering_type, 
        ) if timestep_boundaries is None else timestep_boundaries
        num_stages = len(boundaries) - 1

        boundaries = [int(v) for v in boundaries]
        if len(boundaries) < 2:
            raise ValueError("timestep_boundaries needs at least two values.")
        if boundaries[0] != 0 or boundaries[-1] != self.timesteps:
            raise ValueError(
                "timestep_boundaries must start at 0 and end at self.timesteps."
            )
        if any(b <= a for a, b in zip(boundaries[:-1], boundaries[1:])):
            raise ValueError("timestep_boundaries must be strictly increasing.")

        if verbose_stages:
            print("Initiated boundaries:", boundaries)

        user_callbacks = list(fit_kwargs.pop("callbacks", []) or [])
        merged_history = {}
        stage_records = []
        all_epochs = []
        epoch_cursor = 0


        def run_stage(stage_id, min_t, max_t, epochs, final=False):
            nonlocal epoch_cursor


            self.set_timestep_bounds(min_t, max_t)
            self.train_function = None
            self.test_function = None

            stage_callbacks = list(user_callbacks)
            if pacing_type == "plateau" and not final:
                if earlystopping_type == "epoch_wise":
                    stage_callbacks.append(callbacks.EarlyStopping(
                        monitor=monitor, 
                        min_delta=min_delta, 
                        patience=patience, 
                        mode=stopper_mode, 
                        verbose=verbose_stages
                    ))
                elif earlystopping_type == "batch_size":
                    stage_callbacks.append(BatchLossPlateau(
                        monitor=monitor, 
                        patience=patience, 
                        min_delta=min_delta, 
                        # mode=stopper_mode
                    ))
                else:
                    raise ValueError(
                        f"earlystopping_type must be one of (epoch_wise, batch_wise), but not {earlystopping_type}"
                    )
            elif pacing_type != "fixed":
                raise ValueError("pacing_type must be one of ('plateau', 'fixed').")

            if verbose_stages:
                name = "final/full-range" if final else f"{stage_id}/{num_stages}"
                print(
                    f"Progressive stage {name}: sampling t in "
                    f"[{min_t}, {max_t}) range."
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

            stage_records.append({
                "stage": "final" if final else stage_id, 
                "min_timestep": min_t, 
                "max_timestep": max_t, 
                "epochs_ran": len(actual_epochs), 
                "history": history.history, 
            })


        try:
            # C_N is easiest.  The paper's task-accumulation version then adds
            # C_{N-1}, C_{N-2}, ... until the full support is present.
            for stage_id in range(1, num_stages + 1):
                boundary_id = num_stages - stage_id
                run_stage(
                    stage_id=stage_id, 
                    min_t=boundaries[boundary_id], 
                    max_t=self.timesteps, 
                    epochs=stage_epochs, 
                )

            if final_epochs > 0:
                run_stage(
                    stage_id=num_stages + 1, 
                    min_t=0, 
                    max_t=self.timesteps, 
                    epochs=final_epochs, 
                    final=True, 
                )
        finally:
            # Normal fit/evaluate/noisify behavior must remain unchanged after
            # the progressive call, even if training was interrupted.
            self.set_timestep_bounds()
            self.train_function = None
            self.test_function = None

        history = callbacks.History()
        history.set_model(self)
        history.history = merged_history
        history.epoch = all_epochs
        history.progressive_stages = stage_records
        history.timestep_boundaries = boundaries

        return history

    def set_timestep_bounds(
        self, 
        min_timesteps: int | None = None, 
        max_timesteps: int | None = None
    ):
        min_timesteps = 0 if min_timesteps is None else min_timesteps
        max_timesteps = self.timesteps if max_timesteps is None else max_timesteps


        assert 0 <= min_timesteps < max_timesteps <= self.timesteps, \
            "Expected 0 <= min_timesteps < max_timesteps <= timesteps, "\
            f"got [{min_timesteps}, {max_timesteps}) with T={self.timesteps}."


        self._active_min_timestep = min_timesteps
        self._active_max_timestep = max_timesteps

    def set_current_resolution(self, resolution: int | None = None):
        resolution = self.image_size if resolution is None else resolution

        assert int(resolution) == resolution, \
            "resolution must be an integer."
        resolution = int(resolution)
        assert resolution > 0, \
            "resolution must be positive."
        assert resolution % self.network.patch_size == 0, \
            "resolution must be divisible by patch_size."

        self.network.set_current_resolution(
            resolution
        )
        self.ema_network.set_current_resolution(
            resolution
        ) if self.ema_network is not None else None

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

        for w, ew in zip(self.network.weights, 
                        self.ema_network.weights):
            ew.assign(self.ema_decay * ew + (1 - self.ema_decay) * w)

        return True

    def apply_grads(self, tape: tf.GradientTape, 
                    loss: tf.Tensor, 
                    variables: tf.Variable | None = None):
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
        use_total_loss = use_image_loss or use_kl_loss or use_ctr_loss if use_total_loss is None else use_total_loss

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
            assert ctr_loss is not None and \
                ctr_preds is not None and \
                classes is not None, \
                "When use_ctr_loss is True, "\
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
