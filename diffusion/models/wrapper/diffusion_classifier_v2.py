"""Alternating generator/discriminator optimization for diffusion classifiers."""

import tensorflow as tf
from tensorflow.keras import optimizers

from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier


class DiffusionClassifierV2(DiffusionClassifier):
    """Split a ``DiTClassifier`` variable set across two optimizers.

    The generator phase optimizes diffusion losses over all raw-network
    variables not assigned to the classifier group.  The discriminator phase
    optimizes classifier losses over the selected shared embeddings/main depths
    plus all classifier-branch/head variables.  :meth:`compile` clones the
    configured Keras optimizer so both groups have independent slot state.

    Initial split state is deliberately lazy: ``clf_trainable_variables`` and
    ``gen_trainable_variables`` are None, and ``_train_part``/``_test_part`` are
    None until a phase-specific fit/evaluate method is used.  ``compile`` builds
    the groups and optimizers.
    """

    def __init__(
        self, 
        clf_loss_coef: float = 1., 
        clf_vars_embedding_ids: list[int] = [], 
        clf_vars_noise_part_ids: list[int] = [], 
        clf_train_noisified_max_timesteps: int | None = None, 
        clf_test_noisified_max_timesteps: int | None = None, 
        **kwargs: object
    ) -> None:
        """Configure separate generator and classifier optimization.

        This classifier variant assigns selected embedding and transformer
        depths to the classifier optimizer while the remaining variables use
        the generator optimizer. After progressive network growth, the original
        transformer ID specification is resolved again so negative IDs still
        select depths relative to the expanded network.

        Args:
            clf_loss_coef: Multiplier applied to the classifier objective.
            clf_vars_embedding_ids: Embedding groups trained by the classifier
                optimizer.  IDs are: 0 patch embedder, 1 time embedder, 2 label
                embedder, 3 main depth-0 label regularizer, and 4 the shared main
                class token when ``classifier_only_cls_token=False``.  ``None``
                passes legacy expansion to IDs 1..5; ID 5 is reserved and has no
                effect in the current selector.  Selected optional layers must
                exist.  Default ``[]`` selects no shared embedding explicitly;
                a classifier-only token is always included automatically.
            clf_vars_noise_part_ids: Transformer depth IDs assigned to the
                classifier optimizer.  For network depth N, explicit positives
                are 1..N-1; negatives are ``-N..-1`` and normalize by
                ``id + N + 1`` (so ``-1`` selects final depth N).  Zero is
                invalid.  Negative IDs are re-resolved after progressive growth.
            clf_train_noisified_max_timesteps: Optional exclusive timestep cap
                used while fitting the classifier part.  ``None`` trains its
                classifier on clean images at timestep 0.
            clf_test_noisified_max_timesteps: Optional exclusive timestep cap
                used while evaluating the classifier part.  ``None`` evaluates
                clean images at timestep 0.
            **kwargs: Constructor arguments forwarded to
                ``DiffusionClassifier`` and ``DiffusionModel``.  These include
                ``network=DiTClassifier(...)``, classifier mask/train settings,
                EMA/schedule/CFG/loss/timestep/resize options, and Keras model
                keys ``name``, ``trainable``, ``dtype``, and ``dynamic``.

        Returns:
            ``None``. The wrapper, variable selectors, and split-training state
            are initialized in place.
        """

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
        
    def _check_clfv2_assertions(self, local_vars: dict[str, object]) -> None:
        """Validate shared-embedding and main-depth variable selectors.

        Args:
            local_vars (dict[str, object]): V2 constructor namespace.

        Returns:
            None.  Invalid embedding IDs or zero/out-of-range main depth IDs
            raise ``AssertionError`` (or ``TypeError`` for nonnumeric depth IDs).
        """
        for id_ in local_vars["clf_vars_embedding_ids"]:
            assert id_ is None or 0 <= id_ <= 4 , \
                "clf_vars_embedding_ids can only include (None, 0, 1, 2, 3, 4)."

        for id_ in local_vars["clf_vars_noise_part_ids"]:
            assert -self.network.depth <= id_ < self.network.depth and id_ != 0, \
                "clf_vars_noise_part_ids items can only be in [-depth, 0)+(0, depth] range."

    def _set_clf_variables(self) -> None:
        """Assemble variables owned by the classifier optimizer.

        Selected shared embeddings and main-transformer stages are added first,
        followed by every existing classifier depth, classifier-specific
        regularizer/token, and classifier-head variable.  Variables are kept in
        discovery order; the method does not deduplicate shared selections.

        Returns:
            None.  ``clf_trainable_variables`` becomes ``list[tf.Variable]``.

        Raises:
            AttributeError: If an explicitly selected optional embedder or
                regularizer does not exist for the network configuration.
        """
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

    def _set_gen_variables(self) -> None:
        """Assign all remaining raw-network variables to the generator group.

        Returns:
            None.  ``gen_trainable_variables`` contains every network trainable
            variable whose object identity is absent from the classifier group.

        Raises:
            AssertionError: If classifier variables have not been initialized.
        """
        assert self.clf_trainable_variables is not None

        clf_variable_ids = {id(v) for v in self.clf_trainable_variables}

        self.gen_trainable_variables = []
        for v in self.network.trainable_variables:
            if id(v) not in clf_variable_ids:
                self.gen_trainable_variables.append(v)

    def _switch_train_part(self, part_name: str) -> None:
        """Select a training phase and invalidate the cached train function.

        Args:
            part_name (str): Conventionally ``"generator"`` or
                ``"discriminator"``; validation occurs in :meth:`train_step`.

        Returns:
            None.  Re-selecting the active name leaves the cache intact.
        """
        if self._train_part != part_name:
            self._train_part = part_name
            self.train_function = None

    def _switch_test_part(self, part_name: str) -> None:
        """Select an evaluation phase and invalidate the cached test function.

        Args:
            part_name (str): Conventionally ``"generator"`` or
                ``"discriminator"``.

        Returns:
            None.
        """
        if self._test_part != part_name:
            self._test_part = part_name
            self.test_function = None

    def _merge_result_dicts(
        self,
        dicts: tuple[dict | None, ...] | list[dict | None],
        names: tuple[str, ...] | list[str]
    ) -> dict:
        """Merge phase result dictionaries, prefixing colliding metric names.

        Args:
            dicts (Sequence[dict | None]): Result mappings; None entries are
                recursively discarded.  Mappings are mutated when collisions
                are removed.
            names (Sequence[str]): Prefix aligned with each mapping, normally
                ``"generator"`` and ``"discriminator"``.

        Returns:
            dict[str, object]: Merged values.  A key appearing in more than one
            mapping becomes ``"<phase>_<key>"`` for each phase; unique keys are
            unchanged.  All-None input returns an empty dictionary.
        """
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

            merged_dict.update(dict_)

        return merged_dict

    @property
    def clf_vars_names(self):
        """Return TensorFlow names assigned to classifier optimization.

        Returns:
            list[str]: Empty before variable groups are built; otherwise names
            in ``clf_trainable_variables`` order.
        """
        if self.clf_trainable_variables is None:
            return []

        return self.network.get_variables_names(self.clf_trainable_variables)

    @property
    def gen_vars_names(self):
        """Return TensorFlow names assigned to generator optimization.

        Returns:
            list[str]: Empty before variable groups are built; otherwise names
            in ``gen_trainable_variables`` order.
        """
        if self.gen_trainable_variables is None:
            return []

        return self.network.get_variables_names(self.gen_trainable_variables)

    def compile(self, **kwargs):
        """Compile both phases and clone an independent classifier optimizer.

        Args:
            **kwargs: Forwarded through ``DiffusionClassifier.compile`` to
                ``DiffusionModel.compile``/``tf.keras.Model.compile``.  Useful
                accepted keys are ``loss`` (default MSE), ``optimizer`` (an
                optimizer instance or serializable name/config),
                ``run_eagerly``, ``steps_per_execution``, ``jit_compile`` where
                supported, ``metrics``, ``weighted_metrics``, and
                ``loss_weights``.  The optimizer must support Keras
                serialize/deserialize for cloning.

        Returns:
            None.  ``gen_optimizer`` references the compiled optimizer and
            ``clf_optimizer`` is a newly deserialized optimizer of the same
            configuration with independent iterations/slots.
        """
        self._set_clf_variables()
        self._set_gen_variables()

        super().compile(**kwargs)

        self.gen_optimizer = self.optimizer
        self.clf_optimizer = optimizers.deserialize(
            optimizers.serialize(self.optimizer)
        )

    def _register_optimizer_variables(self) -> None:
        """Refresh split variable groups after progressive depth growth.

        The original ``clf_vars_noise_part_ids`` constructor values are passed
        through the network's legacy ID handler again at the new transformer
        depth. Consequently negative IDs keep their relative meaning after
        layers are appended. Generator variables are registered with the
        generator optimizer and classifier variables with the classifier
        optimizer without replacing either optimizer.

        Returns:
            ``None``. Variable groups and optimizer variable registries are
            updated in place.
        """

        self.clf_vars_noise_part_ids = self.network._handle_ids(
            self._init_config["clf_vars_noise_part_ids"], 
            depth=self.network.depth, 
            min_id=1, 
            max_id=self.network.depth, 
        )
        self.network.set_max_encoder_num(max([
            self.network.max_encoder_num, 
            *self.clf_vars_noise_part_ids
        ]))
        self._set_clf_variables()
        self._set_gen_variables()

        super()._register_optimizer_variables(
            getattr(self, "gen_optimizer", getattr(self, "optimizer", None)), 
            self.gen_trainable_variables
        )
        super()._register_optimizer_variables(
            getattr(self, "clf_optimizer", None), 
            self.clf_trainable_variables
        )

    def prep_clfv2_inputs(self, inputs, 
                        noisified_max_timesteps):
        """Prepare clean or bounded-noise inputs for a classifier-only phase.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Clean float images
                ``[B,H,W,C]`` and zero-based classes ``[B]``.
            noisified_max_timesteps (int | None): Exclusive noising upper bound.
                A number draws timesteps from the active minimum to this bound;
                None leaves images clean and sets all times to 0.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: Integer timesteps
            ``[B]``, clean/noisy images at active resolution, uint8 null labels
            ``[B]``, and original zero-based classes ``[B]``.
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
        """Fit only the generator variable group with diffusion objectives.

        Args:
            **kwargs: Forwarded to ``DiffusionModel.fit`` and ultimately Keras
                ``fit``.  Standard keys include ``x``, ``y``, ``batch_size``,
                ``epochs``, ``verbose``, ``callbacks``, ``validation_data``,
                ``shuffle``, ``steps_per_epoch``, and ``validation_steps``.

        Returns:
            tf.keras.callbacks.History: Generator-phase Keras history.
        """
        active_part_name = "generator"
        self._switch_train_part(active_part_name)
        self._switch_test_part(active_part_name)

        return super().fit(**kwargs)

    def fit_generator_progressively(self, **kwargs):
        """Progressively train only the generative diffusion objective.

        This mirrors ``fit_generator`` but dispatches to DiffusionModel's
        progressive curriculum trainer. The classifier/discriminator phase can
        be trained normally afterwards with ``fit_discriminator``.

        Args:
            **kwargs: Exact arguments accepted by ``fit_progressively``:
                ``stage_tasks``, optional ``stages_num``, ``stages_verbose``,
                ``stage_epochs``, ``final_epochs``, ``timestep_boundaries``,
                ``timestep_clustering_type``, ``resolutions``, ``depths``, pacing
                and early-stopping options, plus standard Keras fit keys such as
                ``x``, ``validation_data``, ``callbacks``, and step counts.

        Returns:
            tf.keras.callbacks.History: Merged progressive generator history.
        """

        active_part_name = "generator"
        self._switch_train_part(active_part_name)
        self._switch_test_part(active_part_name)

        return super().fit_progressively(**kwargs)

    def fit_discriminator(self, **kwargs):
        """Fit only classifier-owned variables with classifier objectives.

        Args:
            **kwargs: Forwarded to Keras fit through ``DiffusionModel.fit``;
                accepted common keys are ``x``, ``y``, ``batch_size``, ``epochs``,
                ``verbose``, ``callbacks``, ``validation_data``, ``shuffle``, and
                train/validation step counts.

        Returns:
            tf.keras.callbacks.History: Discriminator-phase Keras history.
        """
        active_part_name = "discriminator"
        self._switch_train_part(active_part_name)
        self._switch_test_part(active_part_name)

        return super().fit(**kwargs)

    def fit_discriminator_progressively(self, **kwargs):
        """Progressively train only the discriminator diffusion objective.

        This mirrors ``fit_generator`` but dispatches to DiffusionModel's
        progressive curriculum trainer.

        Args:
            **kwargs: Exact progressive and Keras fit keys described by
                :meth:`fit_generator_progressively`.

        Returns:
            tf.keras.callbacks.History: Merged progressive discriminator history.
        """

        active_part_name = "discriminator"
        self._switch_train_part(active_part_name)
        self._switch_test_part(active_part_name)

        return super().fit_progressively(**kwargs)

    def fit(
        self,
        gen_kwargs: dict[str, object],
        clf_kwargs: dict[str, object]
    ) -> dict[str, list]:
        """Fit generator then discriminator and merge their history mappings.

        Args:
            gen_kwargs (dict[str, object]): Keys accepted by
                :meth:`fit_generator`, such as ``x``, ``epochs``, callbacks, and
                validation/step options.
            clf_kwargs (dict[str, object]): Independent keys accepted by
                :meth:`fit_discriminator`.

        Returns:
            dict[str, list]: Merged ``History.history`` values.  Colliding names
            receive ``generator_`` and ``discriminator_`` prefixes; unlike
            standard Keras ``fit``, this method returns the mapping, not History.
        """
        gen_history = self.fit_generator(**gen_kwargs).history
        clf_history = self.fit_discriminator(**clf_kwargs).history
        merged_history = self._merge_result_dicts(
            (gen_history, clf_history), 
            ("generator", "discriminator")
        )

        return merged_history

    def evaluate_generator(self, **kwargs):
        """Evaluate only generator/diffusion metrics.

        Args:
            **kwargs: Forwarded to ``DiffusionModel.evaluate``; accepted keys
                include ``x``, ``y``, ``network_name`` (``"raw"``/``"ema"``),
                ``batch_size``, ``verbose``, ``steps``, ``callbacks``, and
                ``return_dict``.

        Returns:
            float | list[float] | dict[str, float]: Standard Keras result.
        """
        active_part_name = "generator"
        self._switch_test_part(active_part_name)

        return super().evaluate(**kwargs)

    def evaluate_discriminator(self, **kwargs):
        """Evaluate only classifier/discriminator metrics.

        Args:
            **kwargs: Same evaluation keys accepted by
                :meth:`evaluate_generator`.

        Returns:
            float | list[float] | dict[str, float]: Standard Keras result.
        """
        active_part_name = "discriminator"
        self._switch_test_part(active_part_name)

        return super().evaluate(**kwargs)

    def evaluate(self, eval_both=False, 
                test_part=None, **kwargs):
        """Evaluate one selected phase or both and merge result dictionaries.

        Args:
            eval_both (bool): Evaluate generator then discriminator regardless of
                ``test_part``.
            test_part (str | None): ``"generator"`` or ``"discriminator"``;
                None reuses the most recently selected test phase.  If no phase
                has been selected and ``eval_both=False``, the result is empty.
            **kwargs: Forwarded to both phase evaluators.  ``return_dict=True``
                is forced; other accepted keys include ``x``, ``y``,
                ``network_name``, ``batch_size``, ``verbose``, ``steps``, and
                ``callbacks``.

        Returns:
            dict[str, float]: Merged metrics; collisions are phase-prefixed.
        """
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
        """Perform one diffusion update on generator-owned variables.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Clean images ``[B,H,W,C]`` and
                zero-based classes ``[B]``.

        Returns:
            dict[str, tf.Tensor]: Running enabled diffusion metrics.  EMA is
            updated after the generator gradient step.
        """
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
        """Evaluate diffusion objectives without applying gradients.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Clean images and zero-based
                classes.

        Returns:
            dict[str, tf.Tensor]: Running diffusion evaluation metrics, including
            image loss for this test path.
        """
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
        """Perform one classifier-only update on classifier-owned variables.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Clean images ``[B,H,W,C]`` and
                zero-based classes ``[B]``.

        Returns:
            dict[str, tf.Tensor]: Running classifier loss/accuracy and enabled
            classifier auxiliary metrics.  Prediction uses null labels and
            clean or bounded-noise input from ``prep_clfv2_inputs``.
        """
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
        """Evaluate classifier-only objectives with the selected test network.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Clean images and zero-based
                classes.

        Returns:
            dict[str, tf.Tensor]: Running classifier evaluation metrics.
        """
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
        """Dispatch a Keras training batch to the active optimization phase.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Clean images and classes.

        Returns:
            dict[str, tf.Tensor]: Generator or discriminator metric mapping.

        Raises:
            ValueError: If neither phase was selected by a phase-specific fit
                method.  The active phase also selects its own optimizer.
        """
        if self._train_part == "generator":
            self.optimizer = self.gen_optimizer
            return self.generator_train_step(inputs)

        if self._train_part == "discriminator":
            self.optimizer = self.clf_optimizer
            return self.discriminator_train_step(inputs)

        raise ValueError(f"Unknown training part: {self._train_part}")

    def test_step(self, inputs):
        """Dispatch a Keras evaluation batch to the active phase.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Clean images and classes.

        Returns:
            dict[str, tf.Tensor]: Generator or discriminator metric mapping.

        Raises:
            ValueError: If no known test phase is active.
        """
        if self._test_part == "generator":
            return self.generator_test_step(inputs)

        if self._test_part == "discriminator":
            return self.discriminator_test_step(inputs)

        raise ValueError(f"Unknown training part: {self._test_part}")
