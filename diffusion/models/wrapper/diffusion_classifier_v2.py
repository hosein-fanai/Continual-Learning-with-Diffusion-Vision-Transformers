import tensorflow as tf
from tensorflow.keras import optimizers

from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier


class DiffusionClassifierV2(DiffusionClassifier):
    """
    
    """

    def __init__(
        self, 
        clf_loss_coef: float = 1., 
        clf_vars_embedding_ids: list[int] = [], 
        clf_vars_noise_part_ids: list[int] = [], 
        clf_train_noisified_max_timesteps: int | None = None, 
        clf_test_noisified_max_timesteps: int | None = None, 
        **kwargs
    ):
        super().__init__(
            clf_loss_coef=clf_loss_coef, 
            **kwargs
        )
        self._check_clfv2_assertions(locals())
        self._save_init_args(locals())

        self.clf_vars_embedding_ids = self.network._handle_ids(
            self.clf_vars_embedding_ids, 
            depth=None, 
            min_id=1, 
            max_id=5, 
        )
        self.clf_vars_noise_part_ids = self.network._handle_ids(
            self.clf_vars_noise_part_ids, 
            depth=self.network.depth, 
            min_id=1, 
            max_id=self.network.depth, 
        )
        self.network.set_max_encoder_num(max([
            self.network.max_encoder_num, 
            *self.clf_vars_noise_part_ids
        ]))

        self.clf_trainable_variables = None
        self.gen_trainable_variables = None
        self._train_part = None
        self._test_part = None
        
    def _check_clfv2_assertions(self, local_vars):
        for id_ in local_vars["clf_vars_embedding_ids"]:
            assert id_ is None or 0 <= id_ <= 4 , \
                "clf_vars_embedding_ids can only include (None, 0, 1, 2, 3, 4)."

        for id_ in local_vars["clf_vars_noise_part_ids"]:
            assert -self.network.depth <= id_ < self.network.depth and id_ != 0, \
                "clf_vars_noise_part_ids items can only be in [-depth, 0)+(0, depth] range."

    def _set_clf_variables(self):
        self.clf_trainable_variables = []

        for id in self.clf_vars_embedding_ids:
            if id == 0:
                self.clf_trainable_variables.extend(
                    self.network.patch_embedder.trainable_variables
                )
            if id == 1:
                self.clf_trainable_variables.extend(
                    self.network.time_embedder.trainable_variables
                )
            if id == 2:
                self.clf_trainable_variables.extend(
                    self.network.label_embedder.trainable_variables
                )
            if id == 3:
                self.clf_trainable_variables.extend(
                    self.network.labels_embed_reg.trainable_variables
                )
            if id == 4 and not self.network.classifier_only_cls_token:
                self.clf_trainable_variables.extend(
                    self.network.cls_token.trainable_variables
                )

        for layers_dict_id in self.clf_vars_noise_part_ids:
            self.clf_trainable_variables.extend(
                self.network.layers_dicts[layers_dict_id-1].trainable_variables 
            )

        if self.network.clf_labels_embed_reg is not None:
            self.clf_trainable_variables.extend(
                self.network.clf_labels_embed_reg.trainable_variables
            )

        if self.network.classifier_only_cls_token:
            self.clf_trainable_variables.extend(
                self.network.cls_token.trainable_variables
            )

        for clf_layers_dict in self.network.clf_layers_dicts:
            self.clf_trainable_variables.extend(
                clf_layers_dict.trainable_variables 
            )

        self.clf_trainable_variables.extend(
            self.network.classifier.trainable_variables
        )

    def _set_gen_variables(self):
        assert self.clf_trainable_variables is not None

        clf_variable_ids = {id(v) for v in self.clf_trainable_variables}

        self.gen_trainable_variables = []
        for v in self.network.trainable_variables:
            if id(v) not in clf_variable_ids:
                self.gen_trainable_variables.append(v)

    def _switch_train_part(self, part_name):
        if self._train_part != part_name:
            self._train_part = part_name
            self.train_function = None

    def _switch_test_part(self, part_name):
        if self._test_part != part_name:
            self._test_part = part_name
            self.test_function = None

    def _merge_result_dicts(self, dicts, names):
        if None in dicts:
            dicts = list(dicts)
            names = list(names)

            id_ = dicts.index(None)
            dicts.pop(id_)
            names.pop(id_)

            return self._merge_result_dicts(dicts, names)

        same_keys = []
        for dict1 in dicts:
            for dict2 in dicts:
                if dict1 is dict2:
                    continue
                for key in dict1:
                    if key in dict2:
                        same_keys.append(key)

        merged_dict = {}
        for dict_, name in zip(dicts, names):
            for key in set(same_keys):
                merged_dict[f"{name}_{key}"] = dict_[key]
                del dict_[key]
                # merged_dict.update(dict_)
                # del merged_dict[key]

            merged_dict.update(dict_)

        return merged_dict

    @property
    def clf_vars_names(self):
        if self.clf_trainable_variables is None:
            return []

        return self.network.get_variables_names(self.clf_trainable_variables)

    @property
    def gen_vars_names(self):
        if self.gen_trainable_variables is None:
            return []

        return self.network.get_variables_names(self.gen_trainable_variables)

    def compile(self, **kwargs):
        self._set_clf_variables()
        self._set_gen_variables()

        super().compile(**kwargs)

        self.gen_optimizer = self.optimizer
        self.clf_optimizer = optimizers.deserialize(
            optimizers.serialize(self.optimizer)
        )

    def prep_clfv2_inputs(self, inputs, 
                        noisified_max_timesteps):
        x0, labels = inputs

        classes = labels
        labels = labels + int(self.use_cfg)
        uncond_labels = tf.zeros_like(labels, dtype=tf.uint8)

        if noisified_max_timesteps is not None:
            x_t, _, t = self.noisify(
                x0, 
                max_timesteps=noisified_max_timesteps
            )
        else:
            x_t = x0
            t = tf.zeros_like(labels, dtype=tf.int32)

        return t, x_t, uncond_labels, classes

    def fit_generator(self, **kwargs):
        active_part_name = "generator"
        self._switch_train_part(active_part_name)
        self._switch_test_part(active_part_name)

        return super().fit(**kwargs)

    def fit_generator_progressively(self, **kwargs):
        """Progressively train only the generative diffusion objective.

        This mirrors ``fit_generator`` but dispatches to DiffusionModel's
        timestep-curriculum trainer. The classifier/discriminator phase can be
        trained normally afterwards with ``fit_discriminator``.
        """

        active_part_name = "generator"
        self._switch_train_part(active_part_name)
        self._switch_test_part(active_part_name)

        return super().fit_progressively(**kwargs)

    def fit_discriminator(self, **kwargs):
        active_part_name = "discriminator"
        self._switch_train_part(active_part_name)
        self._switch_test_part(active_part_name)

        return super().fit(**kwargs)
    
    def fit(self, gen_kwargs, clf_kwargs):
        gen_history = self.fit_generator(**gen_kwargs).history
        clf_history = self.fit_discriminator(**clf_kwargs).history
        merged_history = self._merge_result_dicts(
            (gen_history, clf_history), 
            ("generator", "discriminator")
        )

        return merged_history

    def evaluate_generator(self, **kwargs):
        active_part_name = "generator"
        self._switch_test_part(active_part_name)

        return super().evaluate(**kwargs)

    def evaluate_discriminator(self, **kwargs):
        active_part_name = "discriminator"
        self._switch_test_part(active_part_name)

        return super().evaluate(**kwargs)

    def evaluate(self, eval_both=False, 
                test_part=None, **kwargs):
        test_part = self._test_part if test_part is None else test_part
        kwargs["return_dict"] = True

        gen_eval = self.evaluate_generator(
            **kwargs
        ) if test_part == "generator" or eval_both else None
        clf_eval = self.evaluate_discriminator(
            **kwargs
        ) if test_part == "discriminator" or eval_both else None
        merged_history = self._merge_result_dicts(
            (gen_eval, clf_eval), 
            ("generator", "discriminator")
        )

        return merged_history

    def generator_train_step(self, inputs):
        (x0, noises, 
        t, x_t, 
        cfg_labels, 
        uncond_labels, 
        classes) = self.prep_inputs(inputs)

        with tf.GradientTape() as tape:
            (loss, noise_loss, image_loss, 
            kl_loss, ctr_loss, ctr_preds) = self.forward_and_compute_loss(
                "raw", 
                x0, noises, t, x_t, 
                cond_labels=cfg_labels, 
                uncond_labels=uncond_labels, 
                classes=classes, 
                cfg_scale=self.train_cfg_scale, 
                training=True
            )

        self.apply_grads(tape, loss, self.gen_trainable_variables)
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

    def generator_test_step(self, inputs):
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

    def discriminator_train_step(self, inputs):
        t, x_t, uncond_labels, classes = self.prep_clfv2_inputs(
            inputs, 
            self.clf_train_noisified_max_timesteps
        )

        with tf.GradientTape() as tape:
            classes_pred, *_, clf_regs_list, clf_z_vals = self.network.predict_class(
                (x_t, t, uncond_labels), 
                full_return=True, 
                training=True
            )
            (loss, clf_loss, kl_loss, 
            ctr_loss, classes_pred, 
            ctr_preds) = self.compute_clf_kl_ctr_loss(
                classes, None, None, None, 
                classes_pred, clf_z_vals, 
                clf_regs_list, 
                clf_train_type="uncond", 
                kl_train_type="uncond", 
                ctr_train_type="uncond", 
                training=True
            )

        self.apply_grads(tape, loss, self.clf_trainable_variables)
        self.update_ema()
        results = self.get_clf_results_dict(
            clf_loss, 
            classes, 
            classes_pred, 
            total_loss=loss, 
            clf_kl_loss=kl_loss, 
            clf_ctr_loss=ctr_loss, 
            clf_ctr_preds=ctr_preds
        )

        return results

    def discriminator_test_step(self, inputs):
        t, x_t, uncond_labels, classes = self.prep_clfv2_inputs(
            inputs, 
            self.clf_test_noisified_max_timesteps
        )

        (classes_pred, *_, 
        clf_regs_list, 
        clf_z_vals) = self.get_network(self.test_network_name).predict_class(
            (x_t, t, uncond_labels), 
            full_return=True, 
            training=False
        )
        (loss, clf_loss, kl_loss, 
        ctr_loss, classes_pred, 
        ctr_preds) = self.compute_clf_kl_ctr_loss(
            classes, None, None, None, 
            classes_pred, clf_z_vals, 
            clf_regs_list, 
            clf_train_type="uncond", 
            kl_train_type="uncond", 
            ctr_train_type="uncond", 
            training=False
        )

        results = self.get_clf_results_dict(
            clf_loss, 
            classes, 
            classes_pred, 
            total_loss=loss, 
            clf_kl_loss=kl_loss, 
            clf_ctr_loss=ctr_loss, 
            clf_ctr_preds=ctr_preds
        )

        return results

    def train_step(self, inputs):
        if self._train_part == "generator":
            self.optimizer = self.gen_optimizer
            return self.generator_train_step(inputs)

        if self._train_part == "discriminator":
            self.optimizer = self.clf_optimizer
            return self.discriminator_train_step(inputs)

        raise ValueError(f"Unknown training part: {self._train_part}")

    def test_step(self, inputs):
        if self._test_part == "generator":
            return self.generator_test_step(inputs)

        if self._test_part == "discriminator":
            return self.discriminator_test_step(inputs)

        raise ValueError(f"Unknown training part: {self._test_part}")
