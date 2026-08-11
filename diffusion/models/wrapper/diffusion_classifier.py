from typing import get_args

import tensorflow as tf
from tensorflow.keras import metrics

from autoencoder.variational_autoencoder import VariationalAutoencoder

from . import NetworkName, TrainType

from diffusion.models.wrapper.diffusion_model import DiffusionModel
from diffusion.metrics.ensemble_accuracy import EnsembleAccuracy


class DiffusionClassifier(DiffusionModel):
    """
    
    """

    def __init__(
        self, 
        mask_by_nulls: bool = True, 
        mask_by_t_threshold: bool = False, 
        mask_t_percentage: int = 70, 
        modify_first_t: bool = False, 
        use_ensemble_loss_instead: bool = False, 
        clf_train_type: TrainType = "cond", 
        clf_loss_coef: float = 8.6e-3, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self._check_clf_assertions(locals())
        self._save_init_args(locals())

        self.clf_loss_coef = tf.constant(
            self.clf_loss_coef, 
            dtype=tf.float32
        )
        self.filter_t_threshold = tf.constant(
            int(self.mask_t_percentage / 100 * self.timesteps), 
            dtype=tf.int32
        )

        self.use_clf_kl_loss = bool(self.kl_loss_coef > 0. and 
                            self.network.clf_reshaper_kwargs.get("add_kl", False))
        self.use_clf_ctr_loss = bool(self.ctr_loss_coef > 0. and 
                            len(self.network.clf_cls_token_regularizer_ids) > 0)

    def _check_clf_assertions(self, local_vars: dict):
        if local_vars["mask_by_nulls"]:
            assert self.p_uncond > 0., "mask_by_nulls is not campatible with p_uncond = 0."

        assert local_vars["clf_train_type"] in get_args(TrainType), \
            f"clf_train_type can only be one of {TrainType}."

        if local_vars["clf_train_type"] == "uncond":
            assert local_vars["train_cfg_scale"] is not None, \
                "clf_train_type can be uncond only when train_cfg_scale is not None."

    @property
    def metrics(self):
        return [
            *super().metrics, 
            self.clf_loss_tracker, 
            self.accuracy_tracker, 
            self.clf_kl_loss_tracker, 
            self.clf_ctr_loss_tracker, 
            self.clf_ctr_accuracy_tracker
        ]

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.ensemble_loss_fn = EnsembleAccuracy(
            self, max_t=4
        ) if self.use_ensemble_loss_instead else None

        self.clf_loss_tracker = metrics.Mean(name="classifier_loss")
        self.accuracy_tracker = metrics.SparseCategoricalAccuracy(name="classifier_accuracy")
        self.clf_kl_loss_tracker = metrics.Mean(name="clf_kl_loss")
        self.clf_ctr_loss_tracker = metrics.Mean(name="clf_ctr_loss")
        self.clf_ctr_accuracy_tracker = metrics.SparseCategoricalAccuracy(name="clf_ctr_accuracy")

    def train_step(self, inputs: tuple[tf.Tensor, tf.Tensor]
                ) -> dict:
        (x0, noises, 
        t, x_t, 
        cfg_labels, 
        uncond_labels, 
        classes) = self.prep_inputs(inputs)

        clf_loss_mask = tf.ones_like(cfg_labels, dtype=tf.float32)
        if self.mask_by_nulls:
            null_ids = (cfg_labels == 0)
            clf_loss_mask = clf_loss_mask * tf.cast(null_ids, dtype=tf.float32)
        if self.mask_by_t_threshold:
            not_exceeded_t_threshold_ids = (t <= self.filter_t_threshold)
            clf_loss_mask = clf_loss_mask * tf.cast(not_exceeded_t_threshold_ids, dtype=tf.float32)
        clf_acc_mask = tf.cast(clf_loss_mask, dtype=tf.bool)

        with tf.GradientTape() as tape:
            (x0_pred, noises_pred, 
            (regs_list_c, regs_list_u), 
            (z_vals_c, z_vals_u), 
            (classes_pred_c, classes_pred_u), 
            (clf_regs_list_c, clf_regs_list_u), 
            (clf_z_vals_c, clf_z_vals_u)) = self.forward(
                "raw", x_t, t, t, 
                cond_labels=cfg_labels, 
                uncond_labels=uncond_labels, 
                scale=self.train_cfg_scale, 
                training=True
            )

            (loss1, noise_loss, image_loss, 
            kl_loss1, ctr_loss1, ctr_preds1) = self.compute_noise_image_kl_ctr_loss(
                x0, noises, classes, 
                x0_pred, noises_pred, 
                z_vals_c, regs_list_c, 
                z_vals_u, regs_list_u,  
            )
            (loss2, clf_loss, kl_loss2, 
            ctr_loss2, classes_pred, 
            ctr_preds2) = self.compute_clf_kl_ctr_loss(
                classes, classes_pred_c, 
                clf_z_vals_c, clf_regs_list_c, 
                clf_z_vals_u, classes_pred_u, 
                clf_regs_list_u, clf_loss_mask, 
                x0=x0, training=True
            )
            loss = loss1 + loss2

        self.apply_grads(tape, loss)
        self.update_ema()
        results = self.get_results_dict(
            noise_loss, 
            total_loss=loss, 
            image_loss=image_loss, 
            kl_loss=kl_loss1, 
            ctr_loss=ctr_loss1, 
            ctr_preds=ctr_preds1, 
            classes=classes, 
            use_total_loss=True
        )
        results.update(self.get_clf_results_dict(
            clf_loss, 
            classes, 
            classes_pred, 
            clf_acc_mask, 
            clf_kl_loss=kl_loss2, 
            clf_ctr_loss=ctr_loss2, 
            clf_ctr_preds=ctr_preds2, 
            use_total_loss=False, 
        ))

        return results

    def test_step(self, inputs: tuple[tf.Tensor, tf.Tensor]
                ) -> dict:
        (x0, noises, 
        t, x_t, 
        cond_labels, 
        uncond_labels, 
        classes) = self.prep_inputs(inputs)

        (loss1, noise_loss, image_loss, 
        kl_loss1, ctr_loss1, ctr_preds1) = super().forward_and_compute_loss(
            self.test_network_name, 
            x0, noises, t, x_t, 
            cond_labels=cond_labels, 
            uncond_labels=uncond_labels, 
            classes=classes, 
            cfg_scale=self.test_cfg_scale, 
            # kl_train_type="cond", 
            # ctr_train_type="cond", 
            use_image_loss=True, 
            training=False
        )

        # check classification loss with only NULL class 
        #   and zero timesteps as input labels for all 
        #   output classes, which makes the test clf_loss 
        #   different than the train clf_loss
        zero_ts = tf.zeros_like(t, dtype=tf.int32)
        classes_pred, *_, clf_regs_list, clf_z_vals = self.get_network(self.test_network_name).predict_class(
            (x0, zero_ts, uncond_labels), 
            full_return=True, 
            training=False
        )
        (loss2, clf_loss, kl_loss2, 
        ctr_loss2, classes_pred, 
        ctr_preds2) = self.compute_clf_kl_ctr_loss(
            classes, None, None, None, 
            classes_pred, clf_z_vals, 
            clf_regs_list, 
            clf_train_type="uncond", 
            kl_train_type="uncond", 
            ctr_train_type="uncond", 
            training=False
        )
        loss = loss1 + loss2

        results = self.get_results_dict(
            noise_loss, 
            total_loss=loss, 
            image_loss=image_loss, 
            kl_loss=kl_loss1, 
            ctr_loss=ctr_loss1, 
            ctr_preds=ctr_preds1, 
            classes=classes, 
            use_image_loss=True
        )
        results.update(self.get_clf_results_dict(
            clf_loss, 
            classes, 
            classes_pred, 
            clf_kl_loss=kl_loss2, 
            clf_ctr_loss=ctr_loss2, 
            clf_ctr_preds=ctr_preds2, 
            use_total_loss=False, 
        ))

        return results

    def load_schedules(self, **kwargs):
        super().load_schedules(**kwargs)

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

    def call_network(
        self, 
        x_t: tf.Tensor, 
        t_batch: tf.Tensor, 
        cond_labels: tf.Tensor, 
        uncond_labels: tf.Tensor | None = None, 
        scale: float | None = None, 
        network_name: NetworkName = "raw", 
        training: bool = False
    ):
        network = self.get_network(network_name)

        output_dict_c = network(
            (x_t, t_batch, cond_labels), 
            full_return=True, 
            training=training
        )
        output_dict_u = network(
            (x_t, t_batch, uncond_labels), 
            full_return=True, 
            training=training
        ) if network.use_cfg and scale is not None else {}

        eps_c, eps_u = output_dict_c["noises"], output_dict_u.get("noises", None)
        regs_list_c, regs_list_u = output_dict_c["regs_list"], output_dict_u.get("regs_list", None)
        z_vals_c, z_vals_u = output_dict_c["z_vals"], output_dict_u.get("z_vals", None)
        classes_pred_c, classes_pred_u = output_dict_c["classes"], output_dict_u.get("classes", None)
        clf_regs_list_c, clf_regs_list_u = output_dict_c["clf_regs_list"], output_dict_u.get("clf_regs_list", None)
        clf_z_vals_c, clf_z_vals_u = output_dict_c["clf_z_vals"], output_dict_u.get("clf_z_vals", None)

        return ((eps_c, eps_u), 
                (regs_list_c, regs_list_u), 
                (z_vals_c, z_vals_u), 
                (classes_pred_c, classes_pred_u), 
                (clf_regs_list_c, clf_regs_list_u), 
                (clf_z_vals_c, clf_z_vals_u))

    def compute_clf_kl_ctr_loss(
        self, 
        classes: tf.Tensor, 
        classes_pred_c: tf.Tensor, 
        clf_z_vals_c: tuple[tf.Tensor, tf.Tensor], 
        clf_regs_list_c: list[tf.Tensor], 
        classes_pred_u: tf.Tensor | None = None, 
        clf_z_vals_u: tuple[tf.Tensor, tf.Tensor] | None = None, 
        clf_regs_list_u: list[tf.Tensor] | None = None, 
        clf_loss_mask: tf.Tensor | None = None, 
        clf_train_type: TrainType | None = None, 
        kl_train_type: TrainType | None = None, 
        ctr_train_type: TrainType | None = None, 
        x0: tf.Tensor = None, 
        training: bool | None = None
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        clf_train_type = self.clf_train_type if clf_train_type is None else clf_train_type
        kl_train_type = self.kl_train_type if kl_train_type is None else kl_train_type
        ctr_train_type = self.ctr_train_type if ctr_train_type is None else ctr_train_type

        if self.ensemble_loss_fn is not None:
            classes_pred = self.ensemble_loss_fn.ensemble_predict_batched(
                x0, 
                training=training
            )
            clf_loss = tf.reduce_mean(self.scce_loss_fn(
                classes, 
                classes_pred
            ))
        else:
            classes_pred = classes_pred_c if clf_train_type == "cond" else classes_pred_u
            clf_loss = self.scce_loss_fn(
                classes, 
                classes_pred
            )
            clf_loss = tf.math.divide_no_nan(
                tf.reduce_sum(
                    clf_loss * clf_loss_mask
                ), 
                tf.reduce_sum(clf_loss_mask)
            ) if clf_loss_mask is not None else tf.reduce_mean(clf_loss)

        clf_kl_loss = VariationalAutoencoder.compute_kl(
            z_mean=clf_z_vals_c[0] if kl_train_type == "cond" else clf_z_vals_u[0], 
            z_log_var=clf_z_vals_c[1] if kl_train_type == "cond" else clf_z_vals_u[1]
        ) if self.use_clf_kl_loss else 0.
        clf_ctr_loss, clf_ctr_preds = self.compute_ctr_loss(
            classes, 
            clf_regs_list_c if ctr_train_type == "cond" else clf_regs_list_u
        ) if self.use_clf_ctr_loss else (0., 0.)

        loss = (
            clf_loss * self.clf_loss_coef + 
            clf_kl_loss * self.kl_loss_coef + 
            clf_ctr_loss * self.ctr_loss_coef
        )

        return loss, clf_loss, clf_kl_loss, clf_ctr_loss, classes_pred, clf_ctr_preds

    def get_clf_results_dict(
        self, 
        clf_loss: tf.Tensor, 
        classes: tf.Tensor, 
        classes_pred: tf.Tensor, 
        clf_acc_mask: tf.Tensor | None = None, 
        total_loss: tf.Tensor | None = None, 
        clf_kl_loss: tf.Tensor | None = None, 
        clf_ctr_loss: tf.Tensor | None = None, 
        clf_ctr_preds: tf.Tensor | None = None, 
        use_total_loss: tf.Tensor | None = None, 
        use_kl_loss: tf.Tensor | None = None, 
        use_ctr_loss: tf.Tensor | None = None
    ) -> dict:
        clf_acc_mask = slice(
            None
        ) if clf_acc_mask is None else clf_acc_mask
        use_kl_loss = self.use_clf_kl_loss if use_kl_loss is None else use_kl_loss
        use_ctr_loss = self.use_clf_ctr_loss if use_ctr_loss is None else use_ctr_loss
        use_total_loss = use_ctr_loss or use_kl_loss if use_total_loss is None else use_total_loss

        self.clf_loss_tracker.update_state(clf_loss)
        self.accuracy_tracker.update_state(
            classes[clf_acc_mask], 
            classes_pred[clf_acc_mask]
        )

        results = {}

        if use_total_loss:
            assert total_loss is not None, \
                "When use_total_loss is True, total_loss cannot be None."


            self.total_loss_tracker.update_state(total_loss)
            results.update({
                self.total_loss_tracker.name: 
                self.total_loss_tracker.result()
            })

        results.update({
            self.clf_loss_tracker.name: 
            self.clf_loss_tracker.result(), 
        })

        if use_kl_loss:
            assert clf_kl_loss is not None, \
                "When use_kl_loss is True, kl_loss cannot be None."


            self.clf_kl_loss_tracker.update_state(clf_kl_loss)
            results.update({
                self.clf_kl_loss_tracker.name: 
                self.clf_kl_loss_tracker.result(), 
            })

        if use_ctr_loss:
            assert clf_ctr_loss is not None \
            and clf_ctr_preds is not None, \
                "When use_ctr_loss is True, "\
                "clf_ctr_loss and clf_ctr_preds cannot be None."


            self.clf_ctr_loss_tracker.update_state(clf_ctr_loss)
            results.update({
                self.clf_ctr_loss_tracker.name: 
                self.clf_ctr_loss_tracker.result()
            })

        results.update({
            self.accuracy_tracker.name: 
            self.accuracy_tracker.result()
        })

        if use_ctr_loss:
            self.clf_ctr_accuracy_tracker.update_state(
                classes, 
                clf_ctr_preds
            )
            results.update({
                self.clf_ctr_accuracy_tracker.name: 
                self.clf_ctr_accuracy_tracker.result()
            })

        return results
