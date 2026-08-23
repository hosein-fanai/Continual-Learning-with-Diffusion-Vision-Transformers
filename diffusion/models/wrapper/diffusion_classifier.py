"""Joint diffusion-and-classification training wrapper.

This wrapper expects the raw feature-routing and classifier head implemented by
``diffusion.models.transformer.di_t_classifier.DiTClassifier`` and adds losses,
metrics, EMA use, timestep masking, and Keras train/test steps.
"""

import tensorflow as tf
from tensorflow.keras import callbacks, metrics

from math import ceil

from typing import get_args

from . import NetworkName, TrainType

from autoencoder.variational_autoencoder import VariationalAutoencoder

from diffusion.models.wrapper.diffusion_model import DiffusionModel
from diffusion.metrics.ensemble_accuracy import EnsembleAccuracy


class DiffusionClassifier(DiffusionModel):
    """Train a ``DiTClassifier`` as a denoiser and image classifier jointly.

    The inherited diffusion objective updates the raw noise branch.  A second
    sparse-categorical objective trains the classifier probabilities, with
    optional example masking and optional classifier-side KL/class-token
    regularization.  ``DiTClassifier`` owns the architecture and all ``clf_*``
    layer attributes; this wrapper owns ``clf_*`` losses and metrics.

    Attributes:
        clf_loss_coef (tf.Tensor): Float32 scalar initialized from the
            constructor coefficient.
        filter_t_threshold (tf.Tensor): Int32 scalar
            ``ceil(mask_t_percentage / 100 * timesteps) - 1``.  The inclusive
            comparison therefore selects the requested count of leading
            timesteps; 0 percent selects none.
        use_clf_kl_loss (bool | None): True only when a classifier reshaper is
            present, KL-enabled, and ``kl_loss_coef > 0``; None when the network
            exposes no classifier reshaper metadata.
        use_clf_ctr_loss (bool | None): True only when classifier regularizer
            depths exist and ``ctr_loss_coef > 0``; None when unsupported.
    """

    def __init__(
        self, 
        mask_by_nulls: bool = True, 
        mask_by_t_threshold: bool = False, 
        mask_t_percentage: int = 70, 
        use_ensemble_loss_instead: bool = False, 
        clf_train_type: TrainType = "cond", 
        clf_loss_coef: float = 8.6e-3, 
        **kwargs: object
    ) -> None:
        """Initialize classifier-loss behavior around a raw classifier network.

        Args:
            mask_by_nulls (bool): Restrict classifier loss and accuracy to
                examples whose post-dropout CFG label is null ID 0.  This is a
                selection mask, not merely removal of null examples, and
                requires ``p_uncond > 0``.
            mask_by_t_threshold (bool): Additionally select only examples with
                sampled ``t <= filter_t_threshold``.
            mask_t_percentage (int): Percentage in ``[0,100]`` used to construct the inclusive
                threshold ``ceil(percentage / 100 * timesteps) - 1``.  For
                T=1000 and 70, exactly timesteps 0 through 699 are selected;
                0 percent selects no examples.
            use_ensemble_loss_instead (bool): Ignore the current forward pass's
                class probabilities for classifier loss and use
                ``EnsembleAccuracy(self, max_t=4).ensemble_predict_batched`` on
                clean images instead.  The resulting ensemble probabilities are
                also returned for accuracy.
            clf_train_type (TrainType): ``"cond"`` uses predictions from the
                conditional/possibly dropped-label pass; ``"uncond"`` uses the
                explicit null-label pass and requires ``train_cfg_scale``.
            clf_loss_coef (float): Scalar multiplier for classifier
                cross-entropy, default ``8.6e-3``.
            **kwargs (object): Arguments forwarded to ``DiffusionModel``.  Required in
                normal use is ``network=DiTClassifier(...)``; supported wrapper
                keys include EMA/scheduler/CFG settings, all four diffusion loss
                coefficients and train types, timestep bounds, resize options,
                ``swap_noise_image``, ``seed``, and standard Keras ``Model`` keys
                ``name``, ``trainable``, ``dtype``, and ``dynamic``.

        Returns:
            None: Classifier coefficients, threshold, and loss flags are
            initialized; trackers are created by :meth:`compile`.
        """

        super().__init__(**kwargs)
        self._check_clf_assertions(locals())
        self._save_init_args(locals())
        self._refresh_loss_flags()

        self.clf_loss_coef = tf.constant(
            self.clf_loss_coef, 
            dtype=tf.float32
        )
        self.filter_t_threshold = tf.constant(
            ceil(self.mask_t_percentage / 100 * self.timesteps) - 1,
            dtype=tf.int32
        )

    def _check_clf_assertions(self, local_vars: dict[str, object]) -> None:
        """Validate classifier training choices after base initialization.

        Null-label masking requires classifier-free guidance dropout. Selecting
        unconditional classifier training additionally requires a CFG scale,
        while conditional training has no such scale requirement.

        Args:
            local_vars (dict[str, object]): Classifier constructor arguments.

        Returns:
            None: Invalid combinations raise an assertion with the relevant
            configuration requirement.
        """

        # Null-only masking requires a nonzero probability of null labels.
        if local_vars["mask_by_nulls"]:
            assert self.p_uncond > 0., "mask_by_nulls is not campatible with p_uncond = 0."

        assert isinstance(local_vars["mask_by_nulls"], bool), \
            "mask_by_nulls must be boolean."
        assert isinstance(local_vars["mask_by_t_threshold"], bool), \
            "mask_by_t_threshold must be boolean."
        assert isinstance(local_vars["use_ensemble_loss_instead"], bool), \
            "use_ensemble_loss_instead must be boolean."
        assert isinstance(local_vars["mask_t_percentage"], int) and \
            not isinstance(local_vars["mask_t_percentage"], bool) and \
            0 <= local_vars["mask_t_percentage"] <= 100, \
            "mask_t_percentage must be an integer in [0, 100]."
        assert isinstance(local_vars["clf_loss_coef"], (int, float)) and \
            not isinstance(local_vars["clf_loss_coef"], bool) and \
            local_vars["clf_loss_coef"] >= 0., \
            "clf_loss_coef must be a nonnegative number."
        # The four-step ensemble requires at least four available timesteps.
        if local_vars["use_ensemble_loss_instead"]:
            assert self.timesteps >= 4, \
                "use_ensemble_loss_instead requires at least four timesteps."

        assert local_vars["clf_train_type"] in get_args(TrainType), \
            f"clf_train_type can only be one of {TrainType}."

        # Unconditional classifier training requires an explicit CFG pass.
        if local_vars["clf_train_type"] == "uncond":
            assert self.use_cfg and self.train_cfg_scale is not None, \
                "Unconditional classifier training requires CFG and train_cfg_scale."

    def _refresh_loss_flags(self) -> None:
        """Refresh diffusion and classifier auxiliary-loss availability.

        The base flags describe the diffusion branch. This override adds the
        classifier KL and class-token regularizer flags from the classifier's
        own reshaper and regularizer metadata. It is called at construction and
        again after progressive depth growth.

        Returns:
            ``None``. The four loss flags are updated on this wrapper.
        """

        super()._refresh_loss_flags()

        self.use_clf_kl_loss = bool(
            self.kl_loss_coef > 0. and
            self.network.clf_reshaper_kwargs.get("add_kl", False)
        ) if getattr(self.network, "clf_reshaper_kwargs", None) is not None else None
        self.use_clf_ctr_loss = bool(
            self.ctr_loss_coef > 0. and
            len(self.network.clf_cls_token_regularizer_ids) > 0
        ) if getattr(self.network, "clf_cls_token_regularizer_ids", None) is not None else None

    @property
    def metrics(self) -> list[metrics.Metric]:
        """Return diffusion and classifier metric trackers.

        Returns:
            list[tf.keras.metrics.Metric]: Base diffusion trackers followed by
            classifier loss/accuracy, classifier KL loss, classifier token loss,
            and classifier token accuracy.  Available after :meth:`compile`.
        """

        return [
            *super().metrics, 
            self.clf_loss_tracker, 
            self.accuracy_tracker, 
            self.clf_kl_loss_tracker, 
            self.clf_ctr_loss_tracker, 
            self.clf_ctr_accuracy_tracker
        ]

    def compile(self, **kwargs: object) -> None:
        """Compile the wrapper and create classifier metrics/loss helper.

        Args:
            **kwargs (object): Forwarded to ``DiffusionModel.compile``.  Accepted keys
                include ``loss`` (loss object/name; default MSE), ``optimizer``
                (required for training), ``run_eagerly``,
                ``steps_per_execution``, ``jit_compile`` where supported,
                ``metrics``, ``weighted_metrics``, and ``loss_weights``.

        Returns:
            None: Creates ``EnsembleAccuracy`` when enabled plus five
            classifier metric trackers.
        """

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
                ) -> dict[str, tf.Tensor]:
        """Perform one joint raw-network diffusion/classifier update.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Clean float images
                ``[B,H,W,C]`` and zero-based integer classes ``[B]``.

        Returns:
            dict[str, tf.Tensor]: Running diffusion and classifier metrics.  The
            classifier mask selects CFG-null and/or low-timestep examples when
            configured; divide-no-nan makes an empty selection contribute zero.
        """

        (x0, noises, 
        t, x_t, 
        cfg_labels, 
        uncond_labels, 
        classes) = self.prep_inputs(inputs)

        clf_loss_mask = tf.ones_like(cfg_labels, dtype=tf.float32)
        # Restrict classifier metrics and loss to CFG-dropped examples.
        if self.mask_by_nulls:
            null_ids = (cfg_labels == 0)
            clf_loss_mask = clf_loss_mask * tf.cast(null_ids, dtype=tf.float32)
        # Restrict classifier metrics and loss to the configured leading timesteps.
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
                ) -> dict[str, tf.Tensor]:
        """Evaluate diffusion plus unconditional clean-image classification.

        Diffusion metrics use noisified inputs and the configured test CFG scale.
        Classifier metrics instead call ``predict_class`` on clean ``x0`` with
        timestep 0 and null labels, so test classifier loss intentionally differs
        from masked/noisy training classifier loss.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): Clean images ``[B,H,W,C]`` and
                zero-based classes ``[B]``.

        Returns:
            dict[str, tf.Tensor]: Running enabled diffusion and classifier
            losses/accuracies.
        """

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

    def fit_progressively(self,**kwargs: object) -> callbacks.History:
        """Train a classifier diffusion model with three progressive tasks.

        This override delegates stage execution, timestep changes, resolution
        changes, EMA growth and optimizer registration to
        ``DiffusionModel.fit_progressively``. It only extends the depth syntax
        and history for a ``DiTClassifier``. An unscoped depth specification
        grows the diffusion transformer. To grow either or both branches, use
        ``{"network": network_spec, "classifier": classifier_spec}``.

        A classifier string adds one classifier depth containing that layer, a
        list adds several depths, and a set or dictionary combines layers in
        one depth. Exact classifier layer names are ``feature_aggregator``,
        ``feature_connector``, ``cross_attention_aggregator``,
        ``cross_attention_connector``, ``vision_transformer_block``,
        ``local_mixer``, ``downsampler``, ``upsampler``, ``reshaper`` and
        ``cls_token_regularizer``. For example::

            depths = [{
                "network": "vision_transformer_block",
                "classifier": [
                    {
                        "feature_connector": {"ids": [-1]},
                        "vision_transformer_block": True,
                    }
                ],
            }]

        Timestep and resolution updates happen before a stage. Depth growth
        happens after a successful stage and first trains in the next listed
        stage, or in ``final_epochs`` when it follows the last listed stage.

        Args:
            **kwargs (object): Arguments forwarded to
                ``DiffusionModel.fit_progressively``. They include ordered
                stage descriptions, optional generated-stage counts, timestep
                boundaries, resolutions, depth specifications, pacing and
                early-stopping controls, plus standard Keras fit inputs,
                validation data, callbacks, and step options. Epoch indices are
                managed by the base method.

        Returns:
            tf.keras.callbacks.History: Merged history from the base
            implementation. Every
            ``progressive_stages`` item additionally records the classifier
            depth before its stage and, when depth growth was requested, the
            classifier depth after that growth.
        """

        classifier_depth = self.network.clf_depth
        history = super().fit_progressively(**kwargs)

        for stage in history.progressive_stages:
            stage["classifier_depth"] = classifier_depth
            growth = stage.get("depth_growth", {}).get("classifier")
            # Record and carry forward classifier depth after structural growth.
            if growth is not None:
                stage["post_classifier_depth"] = growth["after"]
                classifier_depth = growth["after"]

        return history

    def evaluate_ensemble_accuracy(
        self, 
        dataset: tf.data.Dataset, 
        verbose: bool = True, 
        **kwargs: object
    ) -> float:
        """Evaluate timestep-ensembled classifier accuracy on a dataset.

        The default evaluates the first ``min(128, timesteps)`` steps using
        the configured test network.

        Args:
            dataset (tf.data.Dataset): Finite batched ``(images, labels)``
                evaluation dataset.
            verbose (bool): Print batch progress when true.
            **kwargs (object): Options forwarded to :class:`EnsembleAccuracy`,
                including ``netwrok_name``, ``compute_type``, ``weighted``,
                ``max_t``, ``t_chunk_size``, and ``seed``.

        Returns:
            float: Sparse categorical accuracy across the full dataset.
        """

        # Default to the configured test network, or raw when EMA is disabled.
        kwargs.setdefault(
            "netwrok_name", 
            self.test_network_name
        )
        kwargs.setdefault(
            "max_t", 
            min(128, self.timesteps)
        )

        ensemble_accuracy = EnsembleAccuracy(
            self, 
            **kwargs
        )
        accuracy_value = ensemble_accuracy.evaluate(
            dataset, verbose=verbose
        )

        return float(accuracy_value)

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
        tuple[list[tf.Tensor | None], list[tf.Tensor | None] | None], 
        tuple[
            tuple[tf.Tensor | None, tf.Tensor | None],
            tuple[tf.Tensor | None, tf.Tensor | None] | None,
        ], 
        tuple[tf.Tensor, tf.Tensor | None], 
        tuple[list[tf.Tensor | None], list[tf.Tensor | None] | None], 
        tuple[
            tuple[tf.Tensor | None, tf.Tensor | None],
            tuple[tf.Tensor | None, tf.Tensor | None] | None,
        ]
    ]:
        """Run conditional and optional unconditional ``DiTClassifier`` passes.

        Args:
            x_t (tf.Tensor): Noisy images ``[B,H,W,C]``.
            t_batch (tf.Tensor): Integer timesteps ``[B]``.
            cond_labels (tf.Tensor): Conditional/possibly dropped labels ``[B]``.
            uncond_labels (tf.Tensor | None): Null labels ``[B]``.
            scale (float | None): Non-None requests an unconditional pass when
                CFG is enabled; combination is performed by ``compute_eps``.
            network_name (NetworkName): ``"raw"`` or ``"ema"``.
            training (bool): Keras training mode.

        Returns:
            tuple: Six conditional/unconditional pairs in order: noise tensors,
            main regularizer lists, main latent-statistic pairs, classifier
            probabilities ``[B,num_classes]``, classifier regularizer lists, and
            classifier latent-statistic pairs.  Unconditional members are None
            when no second pass is requested.
        """

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
        classes_pred_c: tf.Tensor | None,
        clf_z_vals_c: tuple[tf.Tensor | None, tf.Tensor | None] | None,
        clf_regs_list_c: list[tf.Tensor | None] | None,
        classes_pred_u: tf.Tensor | None = None, 
        clf_z_vals_u: tuple[tf.Tensor | None, tf.Tensor | None] | None = None,
        clf_regs_list_u: list[tf.Tensor | None] | None = None,
        clf_loss_mask: tf.Tensor | None = None, 
        clf_train_type: TrainType | None = None, 
        kl_train_type: TrainType | None = None, 
        ctr_train_type: TrainType | None = None, 
        x0: tf.Tensor | None = None,
        training: bool | None = None
    ) -> tuple[
        tf.Tensor, tf.Tensor, tf.Tensor | float, tf.Tensor | float,
        tf.Tensor, tf.Tensor | float
    ]:
        """Compute weighted classifier, classifier-KL, and token objectives.

        Args:
            classes (tf.Tensor): Zero-based targets ``[B]``.
            classes_pred_c (tf.Tensor | None): Conditional class probabilities
                ``[B,num_classes]``.
            clf_z_vals_c (tuple[tf.Tensor, tf.Tensor] | None): Conditional
                classifier latent mean/log variance.
            clf_regs_list_c (list[tf.Tensor | None] | None): Conditional
                classifier token predictions.
            classes_pred_u (tf.Tensor | None): Unconditional probabilities.
            clf_z_vals_u (tuple[tf.Tensor, tf.Tensor] | None): Unconditional
                classifier latent statistics.
            clf_regs_list_u (list[tf.Tensor | None] | None): Unconditional token
                predictions.
            clf_loss_mask (tf.Tensor | None): Float per-example mask ``[B]``;
                None averages all cross-entropies.
            clf_train_type (TrainType | None): Probability source; None uses the
                configured ``clf_train_type``.
            kl_train_type (TrainType | None): Classifier KL source; None uses the
                base setting.
            ctr_train_type (TrainType | None): Classifier token-loss source.
            x0 (tf.Tensor | None): Clean images required by ensemble loss.
            training (bool | None): Mode passed to the ensemble predictor.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor | float, tf.Tensor | float,
            tf.Tensor, tf.Tensor | float]: Weighted classifier total, raw
            classifier loss, classifier KL loss, classifier token loss,
            selected/ensemble probabilities, and averaged token probabilities.
        """
        clf_train_type = self.clf_train_type if clf_train_type is None else clf_train_type
        kl_train_type = self.kl_train_type if kl_train_type is None else kl_train_type
        ctr_train_type = self.ctr_train_type if ctr_train_type is None else ctr_train_type

        # Use clean-image ensemble predictions when ensemble loss is enabled.
        if self.ensemble_loss_fn is not None:
            classes_pred = self.ensemble_loss_fn.ensemble_predict_batched(
                x0, 
                training=training
            )
            clf_loss = tf.reduce_mean(self.scce_loss_fn(
                classes, 
                classes_pred
            ))
        # Otherwise select the configured conditional or unconditional prediction.
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
        use_total_loss: bool | None = None,
        use_kl_loss: bool | None = None,
        use_ctr_loss: bool | None = None
    ) -> dict[str, tf.Tensor]:
        """Update classifier metric trackers and return current values.

        Args:
            clf_loss (tf.Tensor): Required scalar classifier loss.
            classes (tf.Tensor): Zero-based targets ``[B]``.
            classes_pred (tf.Tensor): Class probabilities ``[B,num_classes]``.
            clf_acc_mask (tf.Tensor | None): Boolean selector ``[B]`` for
                accuracy only; None selects all samples.
            total_loss (tf.Tensor | None): Required when total tracking is on.
            clf_kl_loss (tf.Tensor | None): Required when classifier KL is on.
            clf_ctr_loss (tf.Tensor | None): Required when token loss is on.
            clf_ctr_preds (tf.Tensor | None): Token probabilities for accuracy.
            use_total_loss (bool | None): Explicit total tracker switch; None
                enables it for classifier KL/token auxiliaries.
            use_kl_loss (bool | None): Classifier KL tracker override.
            use_ctr_loss (bool | None): Classifier token tracker override.

        Returns:
            dict[str, tf.Tensor]: Current classifier metrics keyed by name.

        Raises:
            AssertionError: If a requested optional metric lacks its input.
        """
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

        # Update total loss only when the caller enabled that tracker.
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

        # Update classifier KL loss only when its objective is active.
        if use_kl_loss:
            assert clf_kl_loss is not None, \
                "When use_kl_loss is True, kl_loss cannot be None."


            self.clf_kl_loss_tracker.update_state(clf_kl_loss)
            results.update({
                self.clf_kl_loss_tracker.name: 
                self.clf_kl_loss_tracker.result(), 
            })

        # Update classifier token loss only when predictions are available.
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

        # Track classifier token accuracy alongside its active loss.
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


def run_self_tests() -> dict[str, str]:
    """Run CPU-small joint diffusion/classification wrapper tests.

    Args:
        None.

    Returns:
        dict[str, str]: ``{"DiffusionClassifier": "passed"}`` after masking,
        conditional/unconditional, ensemble, auxiliary-loss, metric,
        optimization, evaluation, progressive-growth, and rejection checks.
    """

    tf.keras.backend.clear_session()
    tf.random.set_seed(106)


    from diffusion.models.transformer.di_t_classifier import DiTClassifier


    def make_network(**overrides: object) -> DiTClassifier:
        """Construct a fresh tiny classifier network with safe mutable IDs.

        Args:
            **overrides (object): Values replacing classifier-network defaults.

        Returns:
            DiTClassifier: A built raw classifier network.
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


    def make_wrapper(**overrides: object) -> DiffusionClassifier:
        """Construct and compile a fresh tiny classifier wrapper.

        Args:
            **overrides (object): Wrapper arguments replacing test defaults.

        Returns:
            DiffusionClassifier: An eagerly compiled wrapper.
        """

        network = overrides.pop("network", make_network())
        config = {
            "network": network, 
            "use_ema": True, 
            "test_network_name": "ema", 
            "scheduler_name": "linear", 
            "test_steps": 2, 
            "p_uncond": 1.0, 
            "seed": 37, 
            **overrides
        }
        wrapper = DiffusionClassifier(**config)
        wrapper.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3), 
            loss="mse", 
            run_eagerly=True, 
        )

        return wrapper


    wrapper = make_wrapper(mask_by_nulls=True, mask_by_t_threshold=True)
    assert wrapper.mask_by_nulls and wrapper.mask_by_t_threshold
    assert int(wrapper.filter_t_threshold) == 2
    assert abs(float(wrapper.clf_loss_coef) - 8.6e-3) < 1e-7
    assert wrapper.use_clf_kl_loss is False
    assert wrapper.use_clf_ctr_loss is False
    assert wrapper.ensemble_loss_fn is None
    assert [metric.name for metric in wrapper.metrics][-5:] == [
        "classifier_loss", "classifier_accuracy", "clf_kl_loss",
        "clf_ctr_loss", "clf_ctr_accuracy",
    ]

    threshold_only = make_wrapper(
        mask_by_nulls=False, 
        mask_by_t_threshold=True, 
        mask_t_percentage=25, 
        p_uncond=0.0, 
    )
    assert threshold_only.mask_by_nulls is False
    assert threshold_only.mask_by_t_threshold is True
    assert int(threshold_only.filter_t_threshold) == 0
    empty_threshold = make_wrapper(
        mask_by_nulls=False, 
        mask_t_percentage=0,
        p_uncond=0.0, 
    )
    full_threshold = make_wrapper(
        mask_by_nulls=False, 
        mask_t_percentage=100,
        p_uncond=0.0, 
    )
    assert int(empty_threshold.filter_t_threshold) == -1
    assert int(full_threshold.filter_t_threshold) == 3

    images = tf.reshape(tf.linspace(-1.0, 1.0, 32), (2, 4, 4, 1))
    classes = tf.constant([0, 1], dtype=tf.uint8)
    t = tf.constant([0, 3], dtype=tf.int32)
    x_t, _, _ = wrapper.noisify(images, t=t, seed=41)
    cond_labels = tf.constant([1, 2], dtype=tf.uint8)
    null_labels = tf.zeros_like(cond_labels)
    conditional_only = wrapper.call_network(
        x_t, t, cond_labels, null_labels, scale=None,
        network_name="raw", training=False,
    )
    assert len(conditional_only) == 6
    assert conditional_only[0][0].shape == images.shape
    assert conditional_only[0][1] is None
    assert conditional_only[3][0].shape == (2, 2)
    both = wrapper.call_network(
        x_t, t, cond_labels, null_labels, scale=2.0,
        network_name="raw", training=False,
    )
    assert all(pair[1] is not None for pair in both)

    mask = tf.constant([1.0, 0.0])
    clf_values = wrapper.compute_clf_kl_ctr_loss(
        classes, 
        both[3][0], both[5][0], both[4][0], 
        classes_pred_u=both[3][1], 
        clf_z_vals_u=both[5][1], 
        clf_regs_list_u=both[4][1], 
        clf_loss_mask=mask, 
        clf_train_type="cond", 
    )
    assert len(clf_values) == 6 and float(clf_values[1]) >= 0.0
    empty_values = wrapper.compute_clf_kl_ctr_loss(
        classes, 
        both[3][0], both[5][0], both[4][0], 
        classes_pred_u=both[3][1], 
        clf_z_vals_u=both[5][1], 
        clf_regs_list_u=both[4][1], 
        clf_loss_mask=tf.zeros((2,)), 
        clf_train_type="uncond",
    )
    assert float(empty_values[1]) == 0.0

    train_results = wrapper.train_step((images, classes))
    assert {
        "loss", "noise_loss", "classifier_loss", "classifier_accuracy"
    } <= set(train_results)
    test_results = wrapper.test_step((images, classes))
    assert {
        "loss", "noise_loss", "image_loss", "classifier_loss",
        "classifier_accuracy",
    } <= set(test_results)
    dataset = tf.data.Dataset.from_tensor_slices((images, classes)).batch(2)
    history = wrapper.fit(dataset, epochs=1, verbose=0)
    assert len(history.history["classifier_loss"]) == 1
    evaluation = wrapper.evaluate(
        dataset, network_name="raw", verbose=0, return_dict=True
    )
    assert "classifier_accuracy" in evaluation

    unmasked = make_wrapper(
        mask_by_nulls=False, 
        mask_by_t_threshold=False, 
        p_uncond=0.0, 
    )
    assert "classifier_loss" in unmasked.train_step((images, classes))
    unconditional = make_wrapper(
        clf_train_type="uncond", 
        train_cfg_scale=1.0, 
        mask_by_nulls=False, 
    )
    try:
        unconditional.train_step((images, classes))
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError(
            "The current positional unconditioned classifier-loss mismatch changed"
        )

    ensemble = make_wrapper(
        use_ensemble_loss_instead=True, 
        mask_by_nulls=False, 
    )
    assert ensemble.ensemble_loss_fn is not None
    ensemble_predictions = ensemble.compute_clf_kl_ctr_loss(
        classes, 
        both[3][0], both[5][0], both[4][0], 
        x0=images, 
        training=False, 
    )[4]
    assert ensemble_predictions.shape == (2, 2)
    tf.debugging.assert_near(
        tf.reduce_sum(ensemble_predictions, axis=-1), 
        tf.ones((2,)), atol=1e-5
    )

    auxiliary_network = make_network(
        clf_depth=2, 
        clf_vit_block_ids=[], 
        clf_reshaper_ids_dict={1: "flatten", 2: "unflatten"}, 
        clf_reshaper_kwargs={"add_kl": True, "latent_dim_ratio": 1.0}, 
        clf_cls_token_regularizer_ids=[None], 
        force_global_avg_pooling=True, 
    )
    auxiliary = make_wrapper(
        network=auxiliary_network, 
        kl_loss_coef=0.01, 
        ctr_loss_coef=0.01, 
        mask_by_nulls=False, 
    )
    assert auxiliary.use_clf_kl_loss and auxiliary.use_clf_ctr_loss
    auxiliary_outputs = auxiliary.network(
        (x_t, t, cond_labels), 
        full_return=True, 
        training=False
    )
    auxiliary_losses = auxiliary.compute_clf_kl_ctr_loss(
        classes, 
        auxiliary_outputs["classes"], 
        auxiliary_outputs["clf_z_vals"], 
        auxiliary_outputs["clf_regs_list"], 
    )
    assert float(auxiliary_losses[2]) >= 0.0
    assert float(auxiliary_losses[3]) >= 0.0
    auxiliary_metrics = auxiliary.get_clf_results_dict(
        auxiliary_losses[1], classes, 
        auxiliary_losses[4], 
        total_loss=auxiliary_losses[0], 
        clf_kl_loss=auxiliary_losses[2], 
        clf_ctr_loss=auxiliary_losses[3], 
        clf_ctr_preds=auxiliary_losses[5], 
    )
    assert {"clf_kl_loss", "clf_ctr_loss", "clf_ctr_accuracy"} <= set(
        auxiliary_metrics
    )
    auxiliary_test = auxiliary.test_step((images, classes))
    assert {"clf_kl_loss", "clf_ctr_loss", "clf_ctr_accuracy"} <= set(
        auxiliary_test
    )

    from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer
    raw_network = DiffusionTransformer(
        num_classes=2, 
        use_cfg=True, 
        timesteps=4, 
        image_size=4, 
        channels=1, 
        patch_size=2, 
        dim=4, 
        depth=0, 
        mha_num_heads=1, 
        vit_block_mlp_ratio=1.0, 
    )
    metadata_free = DiffusionClassifier(
        network=raw_network, 
        mask_by_nulls=False, 
        use_ema=False, 
        test_network_name="raw", 
        scheduler_name="linear", 
        test_steps=2, 
    )
    assert metadata_free.use_clf_kl_loss is None
    assert metadata_free.use_clf_ctr_loss is None

    progressive = make_wrapper(mask_by_nulls=False)
    progressive_history = progressive.fit_progressively(
        stage_tasks=[
            {"depth": {"classifier": "vision_transformer_block"}},
            {"depth": {"classifier": "vision_transformer_block"}},
        ],
        x=dataset, 
        stages_verbose=False, 
        stage_epochs=1, 
        final_epochs=0, 
        verbose=0, 
    )
    first_record, second_record = progressive_history.progressive_stages
    assert first_record["classifier_depth"] == 1
    assert first_record["post_classifier_depth"] == 2
    assert first_record["depth_growth"]["classifier"]["added"] == 1
    assert second_record["classifier_depth"] == 2
    assert second_record["post_classifier_depth"] == 3
    assert second_record["depth_growth"]["classifier"]["added"] == 1

    branch_growth = make_wrapper(mask_by_nulls=False)
    network_growth = branch_growth._add_depths({
        "network": "vision_transformer_block", 
        "classifier": [], 
    })
    assert network_growth["network"]["added"] == 1
    assert network_growth["classifier"]["added"] == 0
    both_growth = branch_growth._add_depths({
        "network": "vision_transformer_block", 
        "classifier": "vision_transformer_block", 
    })
    assert both_growth["network"]["added"] == 1
    assert both_growth["classifier"]["added"] == 1

    policy = DiffusionClassifier(
        network=make_network(), 
        mask_by_nulls=False, 
        use_ema=False, 
        test_network_name="raw", 
        scheduler_name="linear", 
        test_steps=2, 
        name="policy_classifier_wrapper", 
        trainable=False, 
        dtype="float64", 
    )
    assert policy.name == "policy_classifier_wrapper"
    assert policy.trainable is False
    assert policy.dtype_policy.name == "float64"
    policy_config = policy.get_config()
    assert policy_config["mask_t_percentage"] == 70
    assert "name" not in policy_config and "dtype" not in policy_config
    try:
        DiffusionClassifier.from_config(policy_config)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError(
            "Wrapper config cloning must expose nested-model serialization limits."
        )

    for kwargs in (
        {"mask_by_nulls": True, "p_uncond": 0.0},
        {"mask_t_percentage": -1, "mask_by_nulls": False},
        {"mask_t_percentage": 101, "mask_by_nulls": False},
        {"clf_train_type": "unknown", "mask_by_nulls": False},
        {
            "clf_train_type": "uncond", "train_cfg_scale": None,
            "mask_by_nulls": False,
        },
    ):
        try:
            DiffusionClassifier(
                network=make_network(), test_steps=2, **kwargs
            )
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Expected invalid classifier wrapper: {kwargs}")
    try:
        wrapper.get_clf_results_dict(
            tf.constant(1.0), classes, both[3][0], 
            use_total_loss=True,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("Missing requested total loss must fail")
    try:
        wrapper.get_clf_results_dict(
            tf.constant(1.0), classes, both[3][0], 
            use_kl_loss=True,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("Missing requested classifier KL loss must fail")
    try:
        wrapper.get_clf_results_dict(
            tf.constant(1.0), classes, both[3][0], 
            use_ctr_loss=True,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("Missing requested classifier token loss must fail")

    tf.keras.backend.clear_session()

    return {"DiffusionClassifier": "passed"}


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
