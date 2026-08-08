from typing import get_args

import tensorflow as tf
from tensorflow.keras import metrics, losses

import numpy as np

from common.argument_saver import ArgumentSaverModel

from . import NetworkName, TrainType

from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer
from diffusion.schedulers import make_schedule, SchedulerName


# @tf.keras.saving.register_keras_serializable()
class DiffusionModel(ArgumentSaverModel):

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
        ctr_loss_coef: float = 0., 
        ctr_train_type: TrainType = "cond", 
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

        self.test_network = self.get_network(self.test_network_name)
        self.image_size = self.network.image_size
        self.channels = self.network.channels
        self.timesteps = self.network.timesteps
        self.use_cfg = self.network.use_cfg
        self.p_uncond = 0. if not self.use_cfg else self.p_uncond
        self.test_cfg_scale = 1. if not self.use_cfg else self.test_cfg_scale
        self.use_image_loss = bool(self.image_loss_coef > 0.)
        self.use_ctr_loss = bool(self.ctr_loss_coef > 0.)
        self.noise_loss_coef = tf.constant(self.noise_loss_coef, dtype=tf.float32)
        self.image_loss_coef = tf.constant(self.image_loss_coef, dtype=tf.float32)
        self.ctr_loss_coef = tf.constant(self.ctr_loss_coef, dtype=tf.float32)

        self.load_schedules()
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

    @property
    def metrics(self):
        return [
            self.total_loss_tracker, 
            self.noise_loss_tracker, 
            self.image_loss_tracker, 
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
        self.ctr_loss_tracker = metrics.Mean(name="ctr_loss")
        self.ctr_accuracy_tracker = metrics.SparseCategoricalAccuracy(name="ctr_accuracy")

    def evaluate(self, x: tf.data.Dataset | None = None, 
                y: tf.data.Dataset | None = None, 
                network_name: NetworkName = "ema", 
                **kwargs) -> dict | list[float]:
        self.test_network = self.get_network(network_name)
        self.test_function = None # to force tf recreate compute graph for test_step

        eval_results = super().evaluate(x=x, y=y, **kwargs)

        self.test_network = self.get_network(self.test_network_name)
        self.test_function = None

        return eval_results

    def summary(self, **kwargs):
        return self.network.summary(**kwargs)

    def fit_progressively(self, **kwargs):
        pass

    def train_step(self, inputs: tuple[tf.Tensor, tf.Tensor]
                ) -> dict:
        (x0, noises, 
        t, x_t, 
        cfg_labels, 
        uncond_labels, 
        classes) = self.prep_inputs(inputs)

        with tf.GradientTape() as tape:
            loss, noise_loss, image_loss, ctr_loss, ctr_preds = self.forward_and_compute_loss(
                self.network, x0, noises, t, x_t, 
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

        loss, noise_loss, image_loss, ctr_loss, ctr_preds = self.forward_and_compute_loss(
            self.test_network, x0, 
            noises, t, x_t, 
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
            ctr_loss=ctr_loss, 
            ctr_preds=ctr_preds, 
            classes=classes, 
            use_image_loss=True
        )

        return results

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
            self.schedules[keys] = tf.constant(self.schedules[keys], dtype=tf.float32)

    def get_noise_and_signal_rates(self, t: tf.Tensor):
        a = tf.gather(self.schedules["sqrt_alpha_cumprod"], t)
        b = tf.gather(self.schedules["sqrt_one_minus_alpha_cumprod"], t)

        return a, b

    def q_sample(self, x0: tf.Tensor, 
                t: int, noise):
        a, b = self.get_noise_and_signal_rates(t)

        a = tf.reshape(a, (-1, 1, 1, 1))
        b = tf.reshape(b, (-1, 1, 1, 1))

        return a * x0 + b * noise

    def noisify(self, x0: tf.Tensor, 
                t: tf.Tensor | None = None, 
                max_timesteps: int | None = None, 
                seed: int | None = None):
        max_timesteps = self.timesteps if max_timesteps is None else max_timesteps

        t = tf.random.uniform(
            (tf.shape(x0)[0],), 
            minval=0, 
            maxval=max_timesteps, 
            dtype=tf.int32, 
            seed=seed, 
        ) if t is None else t

        noise = tf.random.normal(tf.shape(x0), dtype=tf.float32)
        x_t = self.q_sample(x0, t, noise)

        return x_t, noise, t

    def postprocess(self, x: tf.Tensor) -> tf.Tensor:
        x = (x + 1) / 2
        x = tf.clip_by_value(x, 0., 1.)

        return x

    def get_network(self, network_name: NetworkName):
        if not self.use_ema:
            assert network_name != "ema", \
                "network_name cannot be ema when use_ema is False."


        network = self.ema_network if network_name == "ema" else self.network

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

    def get_cfg_labels(self, labels: tf.Tensor):
        mask = tf.random.uniform(
            (tf.shape(labels)[0],)
        ) < self.p_uncond
        masked_labels = tf.where(
            mask, 
            tf.zeros_like(labels), 
            labels
        )

        return masked_labels
    
    def prep_inputs(self, inputs: tuple[tf.Tensor, tf.Tensor], 
                    use_label_dropout: bool = True):
        x0, labels = inputs

        classes = labels
        labels = labels + int(self.use_cfg)
        x_t, noises, t = self.noisify(x0)
        cfg_labels = self.get_cfg_labels(
            labels
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

    def compute_noise_image_ctr_loss(
        self, 
        x0: tf.Tensor, 
        noises: tf.Tensor, 
        classes: tf.Tensor, 
        x0_pred: tf.Tensor, 
        noises_pred: tf.Tensor, 
        regs_list_c: list[tf.Tensor], 
        regs_list_u: list[tf.Tensor] = None, 
        ctr_train_type: TrainType | None = None, 
        use_image_loss: bool | None = None
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
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
        ctr_loss, ctr_preds = self.compute_ctr_loss(
            classes, 
            regs_list_c if ctr_train_type == "cond" else regs_list_u
        ) if self.use_ctr_loss else (0., 0.)

        loss = (
            noise_loss * self.noise_loss_coef + 
            image_loss * self.image_loss_coef + 
            ctr_loss * self.ctr_loss_coef
        )

        return loss, noise_loss, image_loss, ctr_loss, ctr_preds

    def call_network(
        self, 
        x_t: tf.Tensor, 
        t_batch: tf.Tensor, 
        cond_labels: tf.Tensor, 
        uncond_labels: tf.Tensor | None = None, 
        scale: float | None = None, 
        network_name: NetworkName = "raw", 
        training: bool = False
    ) -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[list[tf.Tensor], list[tf.Tensor]]]:
        network = self.get_network(network_name)

        eps_c, *_, regs_list_c = network(
            (x_t, t_batch, cond_labels), 
            full_return=True, 
            training=training
        )
        eps_u, *_, regs_list_u = network(
            (x_t, t_batch, uncond_labels), 
            full_return=True, 
            training=training
        ) if self.use_cfg and scale is not None else (None, None)

        return (eps_c, eps_u), (regs_list_c, regs_list_u)

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
        outputs = self.compute_noise_image_ctr_loss(
            x0, noises, classes, 
            x0_pred, noises_pred, 
            regs_list_c=others[0][0], 
            regs_list_u=others[0][1], 
            ctr_train_type=ctr_train_type, 
            use_image_loss=use_image_loss
        )

        return outputs

    def get_results_dict(
        self, 
        noise_loss: tf.Tensor, 
        total_loss: tf.Tensor | None = None, 
        image_loss: tf.Tensor | None = None, 
        ctr_loss: tf.Tensor | None = None, 
        ctr_preds: tf.Tensor | None = None, 
        classes: tf.Tensor | None = None, 
        use_total_loss: bool | None = None, 
        use_image_loss: bool | None = None, 
        use_ctr_loss: bool | None = None, 
    ) -> dict:
        use_image_loss = self.use_image_loss if use_image_loss is None else use_image_loss
        use_ctr_loss = self.use_ctr_loss and len(self.network.cls_token_regularizer_ids) > 0 \
                    if use_ctr_loss is None else use_ctr_loss
        use_total_loss = use_image_loss or use_ctr_loss if use_total_loss is None else use_total_loss

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
        x_t = tf.random.normal((
            n, 
            self.image_size, 
            self.image_size, 
            self.channels
        )) if x_t is None else x_t
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

            alpha_bar_t = self.schedules["alpha_cumprod"][t]
            alpha_bar_t_next = self.schedules["alpha_cumprod"][t_next]
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
                x_t = x_t + sigma_t * tf.random.normal(tf.shape(x_t))

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
