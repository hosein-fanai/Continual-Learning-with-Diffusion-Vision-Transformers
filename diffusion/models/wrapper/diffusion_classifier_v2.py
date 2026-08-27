"""Alternating generator/discriminator optimization for diffusion classifiers."""

import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier
from diffusion.models.wrapper.diffusion_model import DiffusionModel


class DiffusionClassifierV2(DiffusionClassifier):
    """Split a classifier-capable diffusion network across two optimizers.

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

        This classifier variant assigns selected embedding and main-network
        depths to the classifier optimizer while the remaining variables use
        the generator optimizer. After progressive network growth, the original
        transformer ID specification is resolved again so negative IDs still
        select depths relative to the expanded network.

        Args:
            clf_loss_coef (float): Multiplier applied to the classifier objective.
            clf_vars_embedding_ids (list[int]): Embedding groups trained by the classifier
                optimizer.  IDs are: 0 patch embedder, 1 time embedder, 2 label
                embedder, 3 main depth-0 label regularizer, and 4 the shared main
                class token when ``classifier_only_cls_token=False``.  ``None``
                expands to IDs 0..4; optional absent layers are skipped.
                Default ``[]`` selects no shared embedding explicitly; a
                classifier-only token is always included automatically.
            clf_vars_noise_part_ids (list[int]): Main-network depth IDs assigned to the
                classifier optimizer.  For network depth N, explicit positives
                are 1..N; negatives are ``-N..-1`` and normalize by
                ``id + N + 1`` (so ``-1`` selects final depth N).  Zero is
                invalid.  Negative IDs are re-resolved after progressive growth.
            clf_train_noisified_max_timesteps (int | None): Optional exclusive timestep cap
                used while fitting the classifier part. ``None`` trains on
                clean images at timestep 0; ``-1`` uses ``self.timesteps``.
            clf_test_noisified_max_timesteps (int | None): Optional exclusive timestep cap
                used while evaluating the classifier part. ``None`` evaluates
                clean images at timestep 0; ``-1`` uses ``self.timesteps``.
            **kwargs (object): Constructor arguments forwarded to
                ``DiffusionClassifier`` and ``DiffusionModel``.  These include
                ``network=DiTClassifier(...)``,
                ``DiTEncoderDecoderClassifier(...)``, or
                ``UNetClassifier(...)``, classifier mask/train settings,
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
            min_id=0,
            max_id=4,
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

        self.clf_train_noisified_max_timesteps = self.timesteps if self.clf_train_noisified_max_timesteps == -1 \
                                                else self.clf_train_noisified_max_timesteps
        self.clf_train_noisified_max_timesteps = 0 if self.clf_train_noisified_max_timesteps is None \
                                                else self.clf_train_noisified_max_timesteps
        self.clf_test_noisified_max_timesteps = self.timesteps if self.clf_test_noisified_max_timesteps == -1 \
                                                else self.clf_test_noisified_max_timesteps
        self.clf_test_noisified_max_timesteps = 0 if self.clf_test_noisified_max_timesteps is None \
                                                else self.clf_test_noisified_max_timesteps
        self.clf_trainable_variables = None
        self.gen_trainable_variables = None
        self._active_trainable_variables = None
        self._train_part = None
        self._test_part = None

    def _check_clfv2_assertions(self, local_vars: dict[str, object]) -> None:
        """Validate shared-embedding and main-depth variable selectors.

        Args:
            local_vars (dict[str, object]): V2 constructor namespace.

        Returns:
            None: Invalid embedding IDs or zero/out-of-range main depth IDs
            raise ``AssertionError`` (or ``TypeError`` for nonnumeric depth IDs).
        """

        for id_ in local_vars["clf_vars_embedding_ids"]:
            assert id_ is None or 0 <= id_ <= 4 , \
                "clf_vars_embedding_ids can only include (None, 0, 1, 2, 3, 4)."

        for id_ in local_vars["clf_vars_noise_part_ids"]:
            assert -self.network.depth <= id_ <= self.network.depth and id_ != 0, \
                "clf_vars_noise_part_ids items can only be in [-depth, 0) or [1, depth]."

        for name in (
            "clf_train_noisified_max_timesteps", 
            "clf_test_noisified_max_timesteps"
        ):
            value = local_vars[name]
            assert value is None or (
                isinstance(value, int)
                and not isinstance(value, bool)
                and (value == -1 or 1 <= value <= self.timesteps)
            ), f"{name} must be None, -1, or an integer in [1, timesteps]."

        assert self.use_cfg, \
            "DiffusionClassifierV2 requires CFG "\
            "for its null-label classifier phase."

    def _set_clf_variables(self) -> None:
        """Assemble variables owned by the classifier optimizer.

        Selected shared embeddings and main-network stages are added first,
        followed by every existing classifier depth, classifier-specific
        regularizer/token, and classifier-head variable.  Variables are kept in
        discovery order and deduplicated by object identity so overlapping
        selectors cannot apply a gradient twice.

        Returns:
            None: ``clf_trainable_variables`` becomes ``list[tf.Variable]``.

        Raises:
            AttributeError: If an explicitly selected optional embedder or
                regularizer does not exist for the network configuration.
        """

        self.clf_trainable_variables = []

        for embedding_id in self.clf_vars_embedding_ids:
            # Assign patch-embedding variables to the classifier optimizer for ID 0.
            if embedding_id == 0:
                self.clf_trainable_variables.extend(
                    self.network.patch_embedder.trainable_variables
                )
            # Assign an available time embedder for selector ID 1.
            if embedding_id == 1 and self.network.time_embedder is not None:
                self.clf_trainable_variables.extend(
                    self.network.time_embedder.trainable_variables
                )
            # Assign an available label embedder for selector ID 2.
            if embedding_id == 2 and self.network.label_embedder is not None:
                self.clf_trainable_variables.extend(
                    self.network.label_embedder.trainable_variables
                )
            # Assign the main label-regularizer head for selector ID 3.
            if embedding_id == 3 and self.network.labels_embed_reg is not None:
                self.clf_trainable_variables.extend(
                    self.network.labels_embed_reg.trainable_variables
                )
            # Assign the shared main class token for selector ID 4 when usable.
            if embedding_id == 4 and not self.network.classifier_only_cls_token and \
                    self.network.cls_token is not None:
                self.clf_trainable_variables.extend(
                    self.network.cls_token.trainable_variables
                )

        for layers_dict_id in self.clf_vars_noise_part_ids:
            self.clf_trainable_variables.extend(
                self.network.layers_dicts[layers_dict_id-1].trainable_variables 
            )

        # Always train an available classifier-side label regularizer here.
        if self.network.clf_labels_embed_reg is not None:
            self.clf_trainable_variables.extend(
                self.network.clf_labels_embed_reg.trainable_variables
            )

        # Assign a classifier-only class token to the classifier optimizer.
        if self.network.classifier_only_cls_token and \
        self.network.cls_token is not None:
            self.clf_trainable_variables.extend(
                self.network.cls_token.trainable_variables
            )

        # Train either classifier-only or shared distillation token here.
        if self.network.distil_token is not None:
            self.clf_trainable_variables.extend(
                self.network.distil_token.trainable_variables
            )

        for clf_layers_dict in self.network.clf_layers_dicts:
            self.clf_trainable_variables.extend(
                clf_layers_dict.trainable_variables 
            )

        self.clf_trainable_variables.extend(
            self.network.classifier.trainable_variables
        )

        # Keep the distillation softmax head in the classifier variable group.
        if self.network.distil_classifier is not None:
            self.clf_trainable_variables.extend(
                self.network.distil_classifier.trainable_variables
            )

        seen_variable_ids = set()
        unique_variables = []
        for variable in self.clf_trainable_variables:
            # Deduplicate shared Keras variables by object identity.
            if id(variable) not in seen_variable_ids:
                seen_variable_ids.add(id(variable))
                unique_variables.append(variable)

        self.clf_trainable_variables = unique_variables

    def _set_gen_variables(self) -> None:
        """Assign all remaining raw-network variables to the generator group.

        Returns:
            None: ``gen_trainable_variables`` contains every network trainable
            variable whose object identity is absent from the classifier group.

        Raises:
            AssertionError: If classifier variables have not been initialized.
        """

        assert self.clf_trainable_variables is not None


        clf_variable_ids = {id(v) for v in self.clf_trainable_variables}

        self.gen_trainable_variables = []
        for v in self.network.trainable_variables:
            # Assign every variable not owned by the classifier to the generator.
            if id(v) not in clf_variable_ids:
                self.gen_trainable_variables.append(v)

    def _switch_train_part(self, part_name: str) -> None:
        """Select a training phase and invalidate the cached train function.

        Args:
            part_name (str): Conventionally ``"generator"`` or
                ``"discriminator"``; validation occurs in :meth:`train_step`.

        Returns:
            None: Re-selecting the active name leaves the cache intact.
        """

        # Retrace training only when switching optimizer phases.
        if self._train_part != part_name:
            self._train_part = part_name
            self.train_function = None

    def _switch_test_part(self, part_name: str) -> None:
        """Select an evaluation phase and invalidate the cached test function.

        Args:
            part_name (str): Conventionally ``"generator"`` or
                ``"discriminator"``.

        Returns:
            None: The active test phase is updated in place.
        """

        # Retrace evaluation only when switching metric phases.
        if self._test_part != part_name:
            self._test_part = part_name
            self.test_function = None

    def _merge_result_dicts(
        self,
        dicts: tuple[dict | None, ...] | list[dict | None],
        names: tuple[str, ...] | list[str]
    ) -> dict[str, object]:
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

        # Remove absent phase results before merging dictionaries.
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
                # Do not compare a result mapping with itself.
                if dict1 is dict2:
                    continue
                for key in dict1:
                    # Track keys shared by more than one phase for prefixing.
                    if key in dict2:
                        same_keys.append(key)

        merged_dict = {}
        for dict_, name in zip(dicts, names):
            for key in set(same_keys):
                merged_dict[f"{name}_{key}"] = dict_[key]
                del dict_[key]

            merged_dict.update(dict_)

        return merged_dict

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

        # Refresh the active phase reference after rebuilding variable groups.
        if self._train_part == "generator":
            self._active_trainable_variables = self.gen_trainable_variables
        # Keep discriminator growth aligned with its refreshed variable group.
        elif self._train_part == "discriminator":
            self._active_trainable_variables = self.clf_trainable_variables

    @property
    def clf_vars_names(self) -> list[str]:
        """Return TensorFlow names assigned to classifier optimization.

        Returns:
            list[str]: Empty before variable groups are built; otherwise names
            in ``clf_trainable_variables`` order.
        """

        # Report an empty classifier group before variable partitioning.
        if self.clf_trainable_variables is None:
            return []

        return self.network.get_variables_names(self.clf_trainable_variables)

    @property
    def gen_vars_names(self) -> list[str]:
        """Return TensorFlow names assigned to generator optimization.

        Returns:
            list[str]: Empty before variable groups are built; otherwise names
            in ``gen_trainable_variables`` order.
        """

        # Report an empty generator group before variable partitioning.
        if self.gen_trainable_variables is None:
            return []

        return self.network.get_variables_names(self.gen_trainable_variables)

    def compile(self, **kwargs: object) -> None:
        """Compile both phases and clone an independent classifier optimizer.

        Args:
            **kwargs (object): Forwarded through ``DiffusionClassifier.compile`` to
                ``DiffusionModel.compile``/``tf.keras.Model.compile``.  Useful
                accepted keys are ``loss`` (default MSE), ``optimizer`` (an
                optimizer instance or serializable name/config),
                ``run_eagerly``, ``steps_per_execution``, ``jit_compile`` where
                supported, ``metrics``, ``weighted_metrics``, and
                ``loss_weights``.  The optimizer must support Keras
                serialize/deserialize for cloning.

        Returns:
            None: ``gen_optimizer`` references the compiled optimizer and
            ``clf_optimizer`` is a newly deserialized optimizer of the same
            configuration with independent iterations/slots.
        """

        super().compile(**kwargs)

        self._set_clf_variables()
        self._set_gen_variables()

        self.gen_optimizer = self.optimizer
        self.clf_optimizer = optimizers.deserialize(
            optimizers.serialize(self.optimizer)
        )

    def apply_grads(
        self, 
        tape: tf.GradientTape, 
        loss: tf.Tensor, 
        variables: list[tf.Variable] | None = None
    ) -> None:
        """Apply one phase's gradients to its selected variable group.

        Args:
            tape (tf.GradientTape): Tape that recorded ``loss``.
            loss (tf.Tensor): Scalar phase objective.
            variables (list[tf.Variable] | None): Explicit variables, or the
                currently active generator/discriminator group.

        Returns:
            None: Gradients are applied through the active phase optimizer.
        """

        # Fall back to generator variables for direct generator-step calls.
        if variables is None:
            variables = self._active_trainable_variables \
                        or self.gen_trainable_variables

        optimizer = self.clf_optimizer if (
            self._train_part == "discriminator"
            or variables is self.clf_trainable_variables
        ) else self.gen_optimizer
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

    def prep_clfv2_inputs(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor], 
        noisified_max_timesteps: int | None,
        return_x0: bool = False
    ) -> tuple[tf.Tensor, ...]:
        """Prepare clean or bounded-noise inputs for a classifier-only phase.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Clean float images
                ``[B,H,W,C]`` and zero-based classes ``[B]``.
            noisified_max_timesteps (int | None): Exclusive noising upper bound.
                A number draws timesteps from the active minimum to this bound;
                None leaves images clean and sets all times to 0.
            return_x0 (bool): Append the resized clean images for an ensemble
                loss that performs its own noising.

        Returns:
            tuple[tf.Tensor, ...]: Integer timesteps ``[B]``, clean/noisy images
            at active resolution, uint8 null labels ``[B]``, and original
            zero-based classes ``[B]``. Resized clean images are appended when
            ``return_x0=True``.
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
        uncond_labels = tf.zeros_like(labels, dtype=tf.uint8)

        # Noisify classifier inputs within the configured timestep cap.
        if noisified_max_timesteps is not None and \
        noisified_max_timesteps != 0:
            x_t, _, t = self.noisify(
                x0, 
                max_timesteps=noisified_max_timesteps
            )
        # Use clean inputs at timestep zero when None selects clean-only mode.
        else:
            x_t = x0
            t = tf.zeros_like(
                labels, 
                dtype=tf.int32
            )

        outputs = (t, x_t, uncond_labels, classes)

        return (*outputs, x0) if return_x0 else outputs

    def prep_inputs_map(
        self, 
        x0: tf.Tensor, 
        labels: tf.Tensor
    ) -> tuple[tf.Tensor, ...]:
        """Prepare one generator or discriminator input-pipeline batch.

        Args:
            x0 (tf.Tensor): Clean image batch.
            labels (tf.Tensor): Dataset class labels.
        Returns:
            tuple[tf.Tensor, ...]: Seven diffusion tensors for the generator,
            or five classifier tensors (including clean images) with optional
            teacher probabilities for the discriminator.
        """

        # Generator mapping performs only the inherited diffusion prep.
        if self._test_part != "discriminator":
            return DiffusionModel.prep_inputs_map(self, x0, labels)

        max_timesteps = self.clf_test_noisified_max_timesteps \
                        if self._preprocess_training is False \
                        else self.clf_train_noisified_max_timesteps
        prepared_inputs = self.prep_clfv2_inputs(
            (x0, labels), 
            max_timesteps,
            return_x0=True
        )

        # Preserve the ordinary discriminator batch without teacher targets.
        if not self.use_teacher:
            return prepared_inputs

        teacher_labels = self._predict_teacher_labels(
            prepared_inputs[1], 
            prepared_inputs[0], 
            prepared_inputs[2]
        )

        return (*prepared_inputs, teacher_labels)

    def _fit_selected_part(
        self,
        part_name: str,
        progressive: bool,
        fit_kwargs: dict[str, object],
    ) -> callbacks.History:
        """Fit one optimizer phase and always restore neutral wrapper state.

        Args:
            part_name (str): ``"generator"`` or ``"discriminator"``.
            progressive (bool): Use the progressive curriculum trainer when
                true, otherwise use ordinary Keras fitting.
            fit_kwargs (dict[str, object]): Arguments for the selected fit API.

        Returns:
            tf.keras.callbacks.History: History returned by the selected fit.

        Raises:
            ValueError: If ``part_name`` is not a supported optimizer phase.
        """

        # Select the optimizer and live variable group for the requested phase.
        if part_name == "generator":
            optimizer = self.gen_optimizer
            variables = self.gen_trainable_variables
        # Select the independently cloned classifier optimizer.
        elif part_name == "discriminator":
            optimizer = self.clf_optimizer
            variables = self.clf_trainable_variables
        # Reject internal phase names that would silently skip optimization.
        else:
            raise ValueError("part_name must be 'generator' or 'discriminator'.")

        self._switch_train_part(part_name)
        self._switch_test_part(part_name)
        self.optimizer = optimizer
        self._active_trainable_variables = variables

        try:
            # Route curriculum arguments to the actual progressive trainer.
            if progressive:
                return super().fit_progressively(**fit_kwargs)

            return super().fit(**fit_kwargs)
        finally:
            self._switch_train_part("")
            self._switch_test_part("")
            self.optimizer = self.gen_optimizer
            self._active_trainable_variables = None

    def fit_generator(self, **kwargs: object) -> callbacks.History:
        """Fit only the generator variable group with diffusion objectives.

        Args:
            **kwargs (object): Forwarded to ``DiffusionModel.fit`` and ultimately Keras
                ``fit``.  Standard keys include ``x``, ``y``, ``batch_size``,
                ``epochs``, ``verbose``, ``callbacks``, ``validation_data``,
                ``shuffle``, ``steps_per_epoch``, and ``validation_steps``.

        Returns:
            tf.keras.callbacks.History: Generator-phase Keras history.
        """

        return self._fit_selected_part("generator", False, dict(kwargs))

    def fit_generator_progressively(self, **kwargs: object) -> callbacks.History:
        """Progressively train only the generative diffusion objective.

        This mirrors ``fit_generator`` but dispatches to DiffusionModel's
        progressive curriculum trainer. The classifier/discriminator phase can
        be trained normally afterwards with ``fit_discriminator``.

        Args:
            **kwargs (object): Exact arguments accepted by ``fit_progressively``:
                ``stage_tasks``, optional ``stages_num``, ``stages_verbose``,
                ``stage_epochs``, ``final_epochs``, ``timestep_boundaries``,
                ``timestep_clustering_type``, ``resolutions``, ``depths``, pacing
                and early-stopping options, plus standard Keras fit keys such as
                ``x``, ``validation_data``, ``callbacks``, and step counts.

        Returns:
            tf.keras.callbacks.History: Merged progressive generator history.
        """

        return self._fit_selected_part("generator", True, dict(kwargs))
        
    def fit_discriminator(self, **kwargs: object) -> callbacks.History:
        """Fit only classifier-owned variables with classifier objectives.

        Args:
            **kwargs (object): Forwarded to Keras fit through ``DiffusionModel.fit``;
                accepted common keys are ``x``, ``y``, ``batch_size``, ``epochs``,
                ``verbose``, ``callbacks``, ``validation_data``, ``shuffle``, and
                train/validation step counts.

        Returns:
            tf.keras.callbacks.History: Discriminator-phase Keras history.
        """

        return self._fit_selected_part("discriminator", False, dict(kwargs))
        
    def fit_discriminator_progressively(self, **kwargs: object) -> callbacks.History:
        """Progressively train only the discriminator diffusion objective.

        This mirrors ``fit_generator`` but dispatches to DiffusionModel's
        progressive curriculum trainer.

        Args:
            **kwargs (object): Exact progressive and Keras fit keys described by
                :meth:`fit_generator_progressively`.

        Returns:
            tf.keras.callbacks.History: Merged progressive discriminator history.
        """

        return self._fit_selected_part("discriminator", True, dict(kwargs))
        
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

    def evaluate_generator(
        self, **kwargs: object
    ) -> float | list[float] | dict[str, float]:
        """Evaluate only generator/diffusion metrics.

        Args:
            **kwargs (object): Forwarded to ``DiffusionModel.evaluate``; accepted keys
                include ``x``, ``y``, ``network_name`` (``"raw"``/``"ema"``),
                ``batch_size``, ``verbose``, ``steps``, ``callbacks``, and
                ``return_dict``.

        Returns:
            float | list[float] | dict[str, float]: Standard Keras result.
        """

        active_part_name = "generator"
        self._switch_test_part(active_part_name)

        return super().evaluate(**kwargs)

    def evaluate_discriminator(
        self, **kwargs: object
    ) -> float | list[float] | dict[str, float]:
        """Evaluate only classifier/discriminator metrics.

        Args:
            **kwargs (object): Same evaluation keys accepted by
                :meth:`evaluate_generator`.

        Returns:
            float | list[float] | dict[str, float]: Standard Keras result.
        """

        active_part_name = "discriminator"
        self._switch_test_part(active_part_name)

        return super().evaluate(**kwargs)

    def evaluate(
        self,
        eval_both: bool = False,
        test_part: str | None = None,
        **kwargs: object,
    ) -> dict[str, float]:
        """Evaluate one selected phase or both and merge result dictionaries.

        Args:
            eval_both (bool): Evaluate generator then discriminator regardless of
                ``test_part``.
            test_part (str | None): ``"generator"`` or ``"discriminator"``;
                None reuses the most recently selected test phase.  If no phase
                has been selected and ``eval_both=False``, the result is empty.
            **kwargs (object): Forwarded to both phase evaluators.  ``return_dict=True``
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

    def generator_train_step(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """Run the inherited diffusion update for the generator phase.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Clean images and class labels,
                or the prepared diffusion tuple when mapping is enabled.

        Returns:
            dict[str, tf.Tensor]: Running generator loss metrics.
        """

        return DiffusionModel.train_step(self, inputs)

    def generator_test_step(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """Run the inherited diffusion evaluation for the generator phase.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Clean images and class labels,
                or the prepared diffusion tuple when mapping is enabled.

        Returns:
            dict[str, tf.Tensor]: Running generator evaluation metrics.
        """

        return DiffusionModel.test_step(self, inputs)

    def discriminator_train_step(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """Perform one classifier-only update on classifier-owned variables.

        Args:
            inputs (tuple[tf.Tensor, ...]): Clean images and zero-based classes,
                or the prepared discriminator tensors from ``map_preprocess``.

        Returns:
            dict[str, tf.Tensor]: Running classifier loss/accuracy and enabled
            classifier auxiliary metrics.  Prediction uses null labels and
            clean or bounded-noise input from ``prep_clfv2_inputs``.
        """

        prepared_inputs =  self.prep_clfv2_inputs(
            inputs, 
            self.clf_train_noisified_max_timesteps, 
            return_x0=True
        ) if not self.map_preprocess else inputs
        # Separate the mapped teacher target from discriminator inputs.
        if self.use_teacher:
            prepared_inputs, teacher_labels = (
                prepared_inputs[:-1], prepared_inputs[-1]
            )
        # Keep ordinary discriminator training teacher-free.
        else:
            teacher_labels = None
        t, x_t, uncond_labels, classes, x0 = prepared_inputs

        with tf.GradientTape() as tape:
            class_outputs = self.network.predict_class(
                (x_t, t, uncond_labels), 
                full_return=True, 
                training=True
            )
            classes_pred = class_outputs[0]
            clf_regs_list = class_outputs[3]
            clf_z_vals = class_outputs[4]
            distil_preds = None
            # Read the independent distillation head when it is active.
            if self.use_distil_loss:
                distil_preds = class_outputs[5]

            (loss, clf_loss, kl_loss, 
            ctr_loss, distil_loss, 
            classes_pred, ctr_preds, 
            distil_preds) = self.compute_clf_kl_ctr_distil_loss(
                classes, None, None, None, None, 
                classes_pred, clf_z_vals, 
                clf_regs_list, distil_preds, 
                clf_train_type="uncond", 
                kl_train_type="uncond", 
                ctr_train_type="uncond",
                teacher_labels=teacher_labels,
                x0=x0,
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
            clf_distil_loss=distil_loss, 
            clf_ctr_preds=ctr_preds, 
            clf_distil_preds=distil_preds
        )

        return results

    def discriminator_test_step(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """Evaluate classifier-only objectives with the selected test network.

        Args:
            inputs (tuple[tf.Tensor, ...]): Clean images and zero-based classes,
                or the prepared discriminator tensors from ``map_preprocess``.

        Returns:
            dict[str, tf.Tensor]: Running classifier evaluation metrics.
        """

        prepared_inputs =  self.prep_clfv2_inputs(
            inputs, 
            self.clf_test_noisified_max_timesteps,
            return_x0=True
        ) if not self.map_preprocess else inputs
        # Separate the mapped teacher target from discriminator inputs.
        if self.use_teacher:
            prepared_inputs, teacher_labels = (
                prepared_inputs[:-1], prepared_inputs[-1]
            )
        # Keep ordinary discriminator evaluation teacher-free.
        else:
            teacher_labels = None
        t, x_t, uncond_labels, classes, x0 = prepared_inputs

        class_outputs = self.get_network(self.test_network_name).predict_class(
            (x_t, t, uncond_labels), 
            full_return=True, 
            training=False
        )
        classes_pred = class_outputs[0]
        clf_regs_list = class_outputs[3]
        clf_z_vals = class_outputs[4]
        distil_preds = None
        # Read the independent distillation head when it is active.
        if self.use_distil_loss:
            distil_preds = class_outputs[5]

        (loss, clf_loss, kl_loss, 
        ctr_loss, distil_loss, 
        classes_pred, ctr_preds, 
        distil_preds) = self.compute_clf_kl_ctr_distil_loss(
            classes, None, None, None, None, 
            classes_pred, clf_z_vals, 
            clf_regs_list, distil_preds, 
            clf_train_type="uncond", 
            kl_train_type="uncond", 
            ctr_train_type="uncond", 
            teacher_labels=teacher_labels,
            x0=x0,
            training=False
        )

        results = self.get_clf_results_dict(
            clf_loss, 
            classes, 
            classes_pred, 
            total_loss=loss, 
            clf_kl_loss=kl_loss, 
            clf_ctr_loss=ctr_loss, 
            clf_distil_loss=distil_loss, 
            clf_ctr_preds=ctr_preds, 
            clf_distil_preds=distil_preds
        )

        return results

    def train_step(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """Dispatch a Keras training batch to the active optimization phase.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Clean images and classes.

        Returns:
            dict[str, tf.Tensor]: Generator or discriminator metric mapping.

        Raises:
            ValueError: If neither phase was selected by a phase-specific fit
                method.  The active phase also selects its own optimizer.
        """

        # Dispatch generator-phase training with its optimizer.
        if self._train_part == "generator":
            return self.generator_train_step(inputs)

        # Dispatch classifier-phase training with its optimizer.
        if self._train_part == "discriminator":
            return self.discriminator_train_step(inputs)

        raise ValueError(f"Unknown training part: {self._train_part}")

    def test_step(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """Dispatch a Keras evaluation batch to the active phase.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Clean images and classes.

        Returns:
            dict[str, tf.Tensor]: Generator or discriminator metric mapping.

        Raises:
            ValueError: If no known test phase is active.
        """

        # Dispatch evaluation to generator metrics for that active phase.
        if self._test_part == "generator":
            return self.generator_test_step(inputs)

        # Dispatch evaluation to classifier metrics for that active phase.
        if self._test_part == "discriminator":
            return self.discriminator_test_step(inputs)

        raise ValueError(f"Unknown training part: {self._test_part}")


def run_self_tests() -> dict[str, str]:
    """Run deterministic tests for split classifier/generator optimization.

    Args:
        None.

    Returns:
        dict[str, str]: ``{"DiffusionClassifierV2": "passed"}`` after variable
        selection, optimizer separation, dispatch, fit/evaluate, progressive,
        merge, clean/noisy preparation, and invalid-input checks pass.
    """

    tf.keras.backend.clear_session()
    tf.random.set_seed(107)


    from diffusion.models.transformer.di_t_classifier import DiTClassifier


    def make_network(**overrides: object) -> DiTClassifier:
        """Build a fresh tiny DiTClassifier for V2 tests.

        Args:
            **overrides (object): Classifier-network option overrides.

        Returns:
            DiTClassifier: A built test network.
        """

        config = {
            "num_classes": 2, 
            "use_cfg": True, 
            "timesteps": 4, 
            "image_size": 4, 
            "channels": 1, 
            "patch_size": 2, 
            "dim": 4, 
            "depth": 1, 
            "mha_num_heads": 1, 
            "vit_block_mlp_ratio": 1.0, 
            "clf_mha_num_heads": 1, 
            "clf_vit_block_mlp_ratio": 1.0, 
            "feature_aggregation_ids_dict": {1: (-1,)}, 
            "clf_connection_ids_dict": {-1: (-1,)}, 
            **overrides
        }

        return DiTClassifier(**config)


    def make_wrapper(**overrides: object) -> DiffusionClassifierV2:
        """Build and compile a fresh two-optimizer wrapper.

        Args:
            **overrides (object): V2/base-wrapper option overrides.

        Returns:
            DiffusionClassifierV2: An eagerly compiled test wrapper.
        """

        network = overrides.pop("network", make_network())
        config = {
            "network": network, 
            "use_ema": True, 
            "test_network_name": "ema", 
            "scheduler_name": "linear", 
            "test_steps": 2, 
            "mask_by_nulls": False, 
            "p_uncond": 0.0, 
            "seed": 43, 
            **overrides
        }
        wrapper = DiffusionClassifierV2(**config)
        assert wrapper.clf_trainable_variables is None
        assert wrapper.gen_trainable_variables is None
        assert wrapper._train_part is None and wrapper._test_part is None
        assert wrapper.clf_vars_names == wrapper.gen_vars_names == []
        wrapper.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3), 
            loss="mse", 
            run_eagerly=True, 
        )

        return wrapper

    wrapper = make_wrapper(clf_vars_noise_part_ids=[-1])
    assert abs(float(wrapper.clf_loss_coef) - 1.0) < 1e-7
    assert wrapper.clf_vars_noise_part_ids == [1]
    assert wrapper.clf_trainable_variables
    assert wrapper.gen_trainable_variables
    assert wrapper.gen_optimizer is wrapper.optimizer
    assert wrapper.clf_optimizer is not wrapper.gen_optimizer
    assert type(wrapper.clf_optimizer) is type(wrapper.gen_optimizer)
    assert wrapper.clf_vars_names and wrapper.gen_vars_names
    clf_ids = {id(value) for value in wrapper.clf_trainable_variables}
    gen_ids = {id(value) for value in wrapper.gen_trainable_variables}
    assert clf_ids.isdisjoint(gen_ids)
    assert clf_ids | gen_ids == {id(value) for value in wrapper.network.trainable_variables}

    duplicate_selection = make_wrapper(clf_vars_noise_part_ids=[-1, 1])
    duplicate_ids = [
        id(value) for value in duplicate_selection.clf_trainable_variables
    ]
    assert len(duplicate_ids) == len(set(duplicate_ids))

    positive_depth = make_wrapper(
        network=make_network(depth=2), 
        clf_vars_noise_part_ids=[1], 
        clf_loss_coef=0.25, 
    )
    assert positive_depth.clf_vars_noise_part_ids == [1]
    assert abs(float(positive_depth.clf_loss_coef) - 0.25) < 1e-7
    first_depth_ids = {
        id(value)
        for value in positive_depth.network.layers_dicts[0].trainable_variables
    }
    assert first_depth_ids <= {
        id(value) for value in positive_depth.clf_trainable_variables
    }

    shared_network = make_network(
        classifier_only_cls_token=False, 
        cls_token_type="new_weight", 
        clf_cls_token_type=None, 
        cls_token_regularizer_ids=[0], 
    )
    shared = make_wrapper(
        network=shared_network, 
        clf_vars_embedding_ids=[0, 1, 2, 3, 4], 
    )
    assert shared.clf_vars_embedding_ids == [0, 1, 2, 3, 4]
    assert shared.clf_trainable_variables
    expanded = DiffusionClassifierV2(
        network=make_network(), 
        clf_vars_embedding_ids=[None], 
        mask_by_nulls=False, 
        use_ema=False, 
        test_network_name="raw", 
        test_steps=2, 
    )
    assert expanded.clf_vars_embedding_ids == [0, 1, 2, 3, 4]
    try:
        expanded._set_gen_variables()
    except AssertionError:
        pass
    else:
        raise AssertionError("Generator variables require classifier variables first")
    expanded.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        run_eagerly=True,
    )
    assert expanded.clf_trainable_variables

    regularized_network = make_network(clf_cls_token_regularizer_ids=[0])
    regularized_split = make_wrapper(network=regularized_network)
    regularizer_ids = {
        id(value) for value in 
        regularized_network.clf_labels_embed_reg.trainable_variables
    }
    assert regularizer_ids <= {
        id(value) for value in 
        regularized_split.clf_trainable_variables
    }

    unique = wrapper._merge_result_dicts(
        ({"a": 1}, {"b": 2}), 
        ("generator", "discriminator")
    )
    assert unique == {"a": 1, "b": 2}
    collided = wrapper._merge_result_dicts(
        ({"loss": 1, "a": 2}, {"loss": 3, "b": 4}),
        ("generator", "discriminator"),
    )
    assert collided == {
        "generator_loss": 1, "a": 2,
        "discriminator_loss": 3, "b": 4,
    }
    assert wrapper._merge_result_dicts((None, {"x": 1}), ("a", "b")) == {"x": 1}
    assert wrapper._merge_result_dicts((None, None), ("a", "b")) == {}

    wrapper.train_function = object()
    wrapper._switch_train_part("generator")
    assert wrapper._train_part == "generator" and wrapper.train_function is None
    retained_train_function = object()
    wrapper.train_function = retained_train_function
    wrapper._switch_train_part("generator")
    assert wrapper.train_function is retained_train_function
    wrapper.test_function = object()
    wrapper._switch_test_part("discriminator")
    assert wrapper._test_part == "discriminator" and wrapper.test_function is None
    retained_test_function = object()
    wrapper.test_function = retained_test_function
    wrapper._switch_test_part("discriminator")
    assert wrapper.test_function is retained_test_function

    images = tf.reshape(tf.linspace(-1.0, 1.0, 32), (2, 4, 4, 1))
    classes = tf.constant([0, 1], dtype=tf.uint8)
    clean_t, clean_x, clean_nulls, clean_classes = wrapper.prep_clfv2_inputs(
        (images, classes), None
    )
    tf.debugging.assert_equal(clean_t, tf.zeros((2,), dtype=tf.int32))
    tf.debugging.assert_near(clean_x, images)
    tf.debugging.assert_equal(clean_nulls, tf.zeros((2,), dtype=tf.uint8))
    tf.debugging.assert_equal(clean_classes, classes)
    noisy_t, noisy_x, _, _ = wrapper.prep_clfv2_inputs((images, classes), 2)
    assert noisy_x.shape == images.shape
    assert bool(tf.reduce_all((0 <= noisy_t) & (noisy_t < 2)))
    prepared_with_clean = wrapper.prep_clfv2_inputs(
        (images, classes), 2, return_x0=True
    )
    assert len(prepared_with_clean) == 5
    tf.debugging.assert_near(prepared_with_clean[-1], images)

    capped = make_wrapper(
        clf_train_noisified_max_timesteps=2, 
        clf_test_noisified_max_timesteps=3, 
    )
    assert capped.clf_train_noisified_max_timesteps == 2
    assert capped.clf_test_noisified_max_timesteps == 3
    capped._switch_train_part("discriminator")
    capped._switch_test_part("discriminator")
    assert "classifier_loss" in capped.train_step((images, classes))
    assert "classifier_accuracy" in capped.test_step((images, classes))

    ensemble = make_wrapper(use_ensemble_loss_instead=True)
    ensemble._switch_train_part("discriminator")
    ensemble._switch_test_part("discriminator")
    assert "classifier_loss" in ensemble.train_step((images, classes))
    assert "classifier_loss" in ensemble.test_step((images, classes))

    wrapper._switch_train_part("generator")
    wrapper._switch_test_part("generator")
    generator_train = wrapper.train_step((images, classes))
    generator_test = wrapper.test_step((images, classes))
    assert "noise_loss" in generator_train
    assert {"noise_loss", "image_loss"} <= set(generator_test)
    separate_noise_wrapper = make_wrapper(
        clf_vars_noise_part_ids=[-1],
        show_separate_noise_losses=True,
    )
    separate_noise_wrapper._switch_train_part("generator")
    separate_generator_train = separate_noise_wrapper.train_step(
        (images, classes)
    )
    assert {
        "total_noise_loss", "cond_noise_loss", "uncond_noise_loss"
    } <= set(separate_generator_train)
    assert "noise_loss" not in separate_generator_train
    wrapper._switch_train_part("discriminator")
    wrapper._switch_test_part("discriminator")
    discriminator_train = wrapper.train_step((images, classes))
    discriminator_test = wrapper.test_step((images, classes))
    assert "classifier_loss" in discriminator_train
    assert "classifier_accuracy" in discriminator_test
    wrapper._train_part = None
    try:
        wrapper.train_step((images, classes))
    except ValueError:
        pass
    else:
        raise AssertionError("An unset training phase must fail")
    wrapper._test_part = None
    try:
        wrapper.test_step((images, classes))
    except ValueError:
        pass
    else:
        raise AssertionError("An unset test phase must fail")

    dataset = tf.data.Dataset.from_tensor_slices((images, classes)).batch(2)
    continual_v2 = make_wrapper(
        network=make_network(
            num_classes=None,
            clf_distil_token_type="new_weight",
        ),
        defer_teacher=True,
        distil_loss_coef=1.0,
    )
    assert continual_v2.teacher_network is None
    assert continual_v2.use_distil_loss is False
    continual_v2._check_new_labels(y=classes, verbose=False)
    v2_teacher = continual_v2.snapshot_teacher_network("raw")
    continual_v2._check_new_labels(
        y=tf.constant([2], dtype=tf.uint8),
        verbose=False,
    )
    continual_v2.set_teacher_network(v2_teacher)
    assert continual_v2.use_distil_loss
    assert continual_v2.map_preprocess
    assert v2_teacher.num_classes == 2
    assert continual_v2.network.num_classes == 3
    new_v2_dataset = tf.data.Dataset.from_tensor_slices((
        images[:1],
        tf.constant([2], dtype=tf.uint8),
    )).batch(1)
    continual_v2_history = continual_v2.fit_discriminator(
        x=new_v2_dataset,
        epochs=1,
        verbose=0,
    )
    assert "distil_loss" in continual_v2_history.history

    gen_history = wrapper.fit_generator(x=dataset, epochs=1, verbose=0)
    clf_history = wrapper.fit_discriminator(x=dataset, epochs=1, verbose=0)
    assert "noise_loss" in gen_history.history
    assert "classifier_loss" in clf_history.history
    gen_eval = wrapper.evaluate_generator(
        x=dataset, network_name="raw", verbose=0, return_dict=True
    )
    clf_eval = wrapper.evaluate_discriminator(
        x=dataset, network_name="raw", verbose=0, return_dict=True
    )
    assert "noise_loss" in gen_eval and "classifier_loss" in clf_eval
    both_eval = wrapper.evaluate(
        eval_both=True, x=dataset, network_name="raw", verbose=0
    )
    assert "noise_loss" in both_eval and "classifier_loss" in both_eval
    selected_eval = wrapper.evaluate(
        test_part="generator", x=dataset, network_name="raw", verbose=0
    )
    assert "noise_loss" in selected_eval
    assert wrapper.evaluate(
        test_part="unknown", x=dataset, network_name="raw", verbose=0
    ) == {}

    combined = wrapper.fit(
        {"x": dataset, "epochs": 1, "verbose": 0}, 
        {"x": dataset, "epochs": 1, "verbose": 0}, 
    )
    assert "noise_loss" in combined and "classifier_loss" in combined
    progressive_gen = wrapper.fit_generator_progressively(
        stage_tasks=[{"timesteps": (1, 4)}], 
        x=dataset, 
        stages_verbose=False, 
        stage_epochs=1, 
        final_epochs=0, 
        verbose=0, 
    )
    assert len(progressive_gen.progressive_stages) == 1
    progressive_clf = wrapper.fit_discriminator_progressively(
        stage_tasks=[{"resolution": 4}], 
        x=dataset, 
        stages_verbose=False, 
        stage_epochs=1, 
        final_epochs=0, 
        verbose=0, 
    )
    assert len(progressive_clf.progressive_stages) == 1

    growing = make_wrapper(clf_vars_noise_part_ids=[-1])
    growth = growing._add_depths("vision_transformer_block")
    assert growth["network"] == {"before": 1, "added": 1, "after": 2}
    assert growing.clf_vars_noise_part_ids == [2]
    assert growing.clf_trainable_variables and growing.gen_trainable_variables
    growing._register_optimizer_variables()

    policy = DiffusionClassifierV2(
        network=make_network(), 
        mask_by_nulls=False, 
        p_uncond=0.0, 
        use_ema=False, 
        test_network_name="raw", 
        scheduler_name="linear", 
        test_steps=2, 
        name="policy_classifier_v2", 
        trainable=False, 
        dtype="float64", 
    )
    assert policy.name == "policy_classifier_v2"
    assert policy.trainable is False
    assert policy.dtype_policy.name == "float64"
    policy_config = policy.get_config()
    assert policy_config["clf_loss_coef"] == 1.0
    assert policy_config["clf_train_noisified_max_timesteps"] is None
    assert policy_config["clf_test_noisified_max_timesteps"] is None
    assert policy_config["name"] == "policy_classifier_v2"
    assert policy_config["trainable"] is False
    assert policy_config["dtype"] == "float64"
    try:
        DiffusionClassifierV2.from_config(policy_config)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError(
            "V2 config cloning must expose nested-model serialization limits."
        )

    for invalid_embeddings in ([-1], [5]):
        try:
            DiffusionClassifierV2(
                network=make_network(), 
                clf_vars_embedding_ids=invalid_embeddings, 
                mask_by_nulls=False, 
                test_steps=2, 
            )
        except AssertionError:
            pass
        else:
            raise AssertionError("Invalid embedding variable IDs must fail")
    valid_final_depth = DiffusionClassifierV2(
        network=make_network(),
        clf_vars_noise_part_ids=[1],
        mask_by_nulls=False,
        test_steps=2,
    )
    assert valid_final_depth.clf_vars_noise_part_ids == [1]
    full_timestep_caps = DiffusionClassifierV2(
        network=make_network(),
        clf_train_noisified_max_timesteps=-1,
        clf_test_noisified_max_timesteps=-1,
        mask_by_nulls=False,
        test_steps=2,
    )
    assert full_timestep_caps.clf_train_noisified_max_timesteps == 4
    assert full_timestep_caps.clf_test_noisified_max_timesteps == 4
    for invalid_depth_ids in ([0], [2], [-2]):
        try:
            DiffusionClassifierV2(
                network=make_network(), 
                clf_vars_noise_part_ids=invalid_depth_ids, 
                mask_by_nulls=False, 
                test_steps=2, 
            )
        except AssertionError:
            pass
        else:
            raise AssertionError("Invalid main-depth variable IDs must fail")
    wrapper._train_part = "unknown"
    try:
        wrapper.train_step((images, classes))
    except ValueError:
        pass
    else:
        raise AssertionError("An unknown training phase must fail")
    wrapper._test_part = "unknown"
    try:
        wrapper.test_step((images, classes))
    except ValueError:
        pass
    else:
        raise AssertionError("An unknown test phase must fail")

    tf.keras.backend.clear_session()
    return {"DiffusionClassifierV2": "passed"}


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
