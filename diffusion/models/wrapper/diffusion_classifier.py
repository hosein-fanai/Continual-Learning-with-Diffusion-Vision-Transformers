"""Joint diffusion-and-classification training wrapper.

This wrapper expects the raw feature-routing and classifier head implemented by
``diffusion.models.transformer.di_t_classifier.DiTClassifier`` and adds losses,
metrics, EMA use, timestep masking, and Keras train/test steps.
"""

import tensorflow as tf
from tensorflow.keras import callbacks, metrics, losses

import numpy as np

from math import ceil

from numbers import Integral, Real
from typing import get_args, Literal

from . import NetworkName, TrainType, copy_network_weights_by_layer

from common.runtime import validate_model_dtype_policy
from common.validation import require

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
        use_distil_loss (bool): True only when a teacher, a student
            distillation token, and a positive ``distil_loss_coef`` are all
            present.
        use_distil_ctr_loss (bool): Whether classifier regularizers use their
            frozen teacher target in ``"distil"`` or ``"both"`` mode.
    """

    def __init__(
        self, 
        teacher_network: tf.keras.Model | None = None, 
        distil_type: Literal["hard", "soft"] = "hard", 
        mask_by_nulls: bool = True, 
        mask_by_t_threshold: bool = False, 
        mask_t_percentage: int = 70, 
        use_ensemble_loss_instead: bool = False, 
        clf_train_type: TrainType = "cond", 
        clf_loss_coef: float = 8.6e-3, 
        distil_loss_coef: float = 0., 
        clf_acc_coef: float = .5, 
        ctr_acc_coef: float = 0., 
        distil_acc_coef: float = .5, 
        defer_teacher: bool = False, 
        distil_temperature: float = 1.,
        distil_scope: Literal[
            "old_classes", "replay_only", "current_and_replay"
        ] = "current_and_replay",
        **kwargs: object
    ) -> None:
        """Initialize classifier-loss behavior around a raw classifier network.

        Args:
            teacher_network (tf.keras.Model | None): Frozen classifier used to
                create teacher probabilities. A wrapper with a raw ``network``
                attribute is also accepted.
            distil_type (Literal["hard", "soft"]): ``"hard"`` applies
                sparse cross-entropy to the teacher argmax; ``"soft"`` applies
                teacher-to-student KL divergence.
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
                a four-timestep raw-network ensemble on clean images instead.
                The resulting probabilities are also returned for accuracy.
            clf_train_type (TrainType): ``"cond"`` uses predictions from the
                conditional/possibly dropped-label pass; ``"uncond"`` uses the
                explicit null-label pass and requires ``train_cfg_scale``.
            clf_loss_coef (float): Scalar multiplier for classifier
                cross-entropy, default ``8.6e-3``.
            distil_loss_coef (float): Multiplier for the distillation-token
                objective. A positive value enables dataset mapping when the
                teacher and token are present.
            clf_acc_coef (float): Primary-head coefficient used only for the
                wrapper's ``total_accuracy`` prediction.
            distil_acc_coef (float): Distillation-head coefficient used only
                for the wrapper's ``total_accuracy`` prediction.
            ctr_acc_coef (float): Classifier-regularizer coefficient used only
                for the wrapper's ``total_accuracy`` prediction.
            defer_teacher (bool): Allow a configured teacher objective to start
                without a teacher. This is intended for continual learning,
                where task 1 has no past model and later tasks call
                :meth:`set_teacher_network` with a frozen snapshot. The default
                keeps the ordinary constructor's strict teacher requirement.
            distil_temperature (float): Positive soft-distillation
                temperature. ``1`` preserves the historical direct
                probability KL exactly; other values soften both teacher and
                student probabilities and apply the standard ``T**2`` scale.
                Hard distillation remains teacher-argmax cross-entropy.
            distil_scope (Literal["old_classes", "replay_only",
                "current_and_replay"]): Examples used by teacher-targeted
                losses. ``"old_classes"`` selects labels represented by the
                frozen teacher, ``"replay_only"`` selects an explicit replay
                mask supplied as the third dataset tensor, and the default
                preserves the historical all-example behavior.
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

        # Runtime teachers must never become part of serialized model config.
        self._init_config.pop("teacher_network", None)
        self._map_preprocess_without_teacher = bool(self.map_preprocess)
        self.set_teacher_network(self.teacher_network)

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        self.clf_loss_coef = tf.constant(
            self.clf_loss_coef,
            dtype=stable_dtype
        )
        self.distil_loss_coef = tf.constant(
            self.distil_loss_coef,
            dtype=stable_dtype
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
            require(self.p_uncond > 0., \
                "mask_by_nulls is not campatible with p_uncond = 0.")

        require(isinstance(local_vars["mask_t_percentage"], Integral) and \
            not isinstance(local_vars["mask_t_percentage"], bool) and \
            0 <= local_vars["mask_t_percentage"] <= 100, \
            "mask_t_percentage must be an integer in [0, 100].")

        for name in (
            "clf_loss_coef", "distil_loss_coef", "clf_acc_coef",
            "ctr_acc_coef", "distil_acc_coef"
        ):
            value = local_vars[name]
            require(isinstance(value, Real) and not isinstance(value, bool) and \
                np.isfinite(value) and value >= 0., \
                f"{name} must be a finite nonnegative number.")

        require(local_vars["distil_type"] in ("hard", "soft"), \
            "distil_type must be either 'hard' or 'soft'.")
        require(isinstance(local_vars["distil_temperature"], Real) and \
            not isinstance(local_vars["distil_temperature"], bool) and \
            np.isfinite(local_vars["distil_temperature"]) and \
            local_vars["distil_temperature"] > 0., \
            "distil_temperature must be a finite positive number.")
        require(local_vars["distil_scope"] in (
            "old_classes", "replay_only", "current_and_replay"
        ), "distil_scope must be 'old_classes', 'replay_only', or " \
            "'current_and_replay'.")

        # A positive token objective requires targets from a teacher network.
        if (local_vars["distil_loss_coef"] > 0. and
        getattr(self.network, "distil_token", None) is not None
        ) or (local_vars["kwargs"].get("ctr_loss_coef", 0.) > 0. and(
        getattr(self.network, "clf_cls_token_regularizer_kwargs", None
        ) or getattr(self.network, "cls_token_regularizer_kwargs", {})
        ).get("train_type", "normal") in ("distil", "both")
        ):
            require(local_vars["teacher_network"] is not None \
                or local_vars["defer_teacher"], \
                "teacher_network is required for distillation " \
                "training unless defer_teacher=True.")

        # The four-step ensemble requires at least four available timesteps.
        if local_vars["use_ensemble_loss_instead"]:
            require(self.timesteps >= 4, \
                "use_ensemble_loss_instead requires at least four timesteps.")

        require(local_vars["clf_train_type"] in get_args(TrainType), \
            f"clf_train_type can only be one of {TrainType}.")

        # Unconditional classifier training requires an explicit CFG pass.
        if local_vars["clf_train_type"] == "uncond":
            require(self.use_cfg and self.train_cfg_scale is not None, \
                "Unconditional classifier training requires CFG and train_cfg_scale.")

    def _refresh_loss_flags(self) -> None:
        """Refresh diffusion and classifier auxiliary-loss availability.

        The base flags describe the diffusion branch. This override adds the
        classifier KL and class-token regularizer flags from the classifier's
        own reshaper and regularizer metadata. It is called at construction and
        again after progressive depth growth.

        Returns:
            ``None``. Classifier and distillation loss flags are updated.
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
        self.use_distil_loss = bool(
            self.teacher_network is not None and
            self.distil_loss_coef > 0. and
            self.network.distil_token is not None
        )

        regularizer_kwargs = getattr(
            self.network, "clf_cls_token_regularizer_kwargs", None
        )
        regularizer_kwargs = getattr(
            self.network, "cls_token_regularizer_kwargs", {}
        ) if regularizer_kwargs is None else regularizer_kwargs
        self.use_distil_ctr_loss = bool(
            self.teacher_network is not None and
            self.ctr_loss_coef > 0. and
            regularizer_kwargs.get("train_type", "normal") in (
                "distil", "both"
            )
        )

        self.use_teacher = self.use_distil_loss or self.use_distil_ctr_loss
        self.use_total_accuracy = bool(
            self.use_distil_loss and self.distil_acc_coef > 0.
        ) or bool(
            self.use_clf_ctr_loss and self.ctr_acc_coef > 0.
        )

    def _predict_teacher_labels(
        self, 
        x_t: tf.Tensor, 
        t: tf.Tensor, 
        labels: tf.Tensor
    ) -> tf.Tensor:
        """Return frozen teacher probabilities for one prepared batch.

        Args:
            x_t (tf.Tensor): Clean or noisified teacher images.
            t (tf.Tensor): Per-example timestep IDs.
            labels (tf.Tensor): Condition IDs supplied to the teacher.

        Returns:
            tf.Tensor: Teacher class probabilities ``[B,num_classes]``.
        """

        # Prefer the classifier-only path when the teacher exposes it.
        if hasattr(self.teacher_network, "predict_class"):
            teacher_labels = self.teacher_network.predict_class(
                (x_t, t, labels), 
                training=False
            )
        # Fall back to the teacher's ordinary forward API.
        else:
            teacher_labels = self.teacher_network(
                (x_t, t, labels), 
                training=False
            )

            # Extract classifier probabilities from project network dictionaries.
            if isinstance(teacher_labels, dict):
                teacher_labels = teacher_labels["classes"]

        return tf.stop_gradient(teacher_labels)

    def _mask_unknown_teacher_labels(
        self, 
        labels: tf.Tensor
    ) -> tf.Tensor:
        """Replace condition IDs outside the past teacher vocabulary with null.

        Args:
            labels (tf.Tensor): Prepared student condition IDs. Existing past
                class IDs remain unchanged; newly introduced IDs may exceed the
                teacher's embedding table.

        Returns:
            tf.Tensor: Condition IDs safe for the teacher. Teachers without
            ``num_labels`` retain the supplied labels for compatibility.
        """

        teacher_num_labels = getattr(
            self.teacher_network, 
            "num_labels",
            None
        )

        # Preserve external teacher semantics when vocabulary metadata is absent.
        if teacher_num_labels is None:
            return labels

        return tf.where(
            labels < tf.cast(teacher_num_labels, labels.dtype), 
            labels, 
            tf.zeros_like(labels)
        )

    @property
    def metrics(self) -> list[metrics.Metric]:
        """Return diffusion and classifier metric trackers.

        Returns:
            list[tf.keras.metrics.Metric]: Base diffusion trackers followed by
            classifier losses and classifier/combined/token accuracies.
            Available after :meth:`compile`.
        """

        return [
            *super().metrics, 
            self.clf_loss_tracker, 
            self.clf_kl_loss_tracker, 
            self.clf_ctr_loss_tracker, 
            self.distil_loss_tracker, 
            self.total_accuracy_tracker, 
            self.accuracy_tracker, 
            self.clf_ctr_accuracy_tracker, 
            self.distil_accuracy_tracker
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
            None: Creates ``EnsembleAccuracy`` when enabled, classifier
            trackers, and the optional distillation trackers.
        """

        super().compile(**kwargs)

        self.ensemble_loss_fn = EnsembleAccuracy(
            self,
            network_name="raw",
            max_t=4,
            seed=self.seed,
            dtype=self.dtype_policy.variable_dtype,
        ) if self.use_ensemble_loss_instead else None
        self.kld_loss_fn = losses.kullback_leibler_divergence

        primary_accuracy_name = "cls_token_accuracy" if self.network.clf_has_cls_token\
                                else "avg_pooling_accuracy"
        configured_distillation = bool(
            self.distil_loss_coef > 0. and 
            self.network.distil_token is not None
        )

        stable_dtype = self.dtype_policy.variable_dtype
        self.clf_loss_tracker = metrics.Mean(
            name="classifier_loss",
            dtype=stable_dtype,
        )
        self.clf_kl_loss_tracker = metrics.Mean(
            name="clf_kl_loss",
            dtype=stable_dtype,
        )
        self.clf_ctr_loss_tracker = metrics.Mean(
            name="clf_ctr_loss",
            dtype=stable_dtype,
        )
        self.distil_loss_tracker = metrics.Mean(
            name="distil_loss",
            dtype=stable_dtype,
        )
        self.total_accuracy_tracker = metrics.SparseCategoricalAccuracy(
            name="total_accuracy",
            dtype=stable_dtype,
        )
        self.accuracy_tracker = metrics.SparseCategoricalAccuracy(
            name=primary_accuracy_name if configured_distillation \
                else "classifier_accuracy",
            dtype=stable_dtype,
        )
        self.clf_ctr_accuracy_tracker = metrics.SparseCategoricalAccuracy(
            name="clf_ctr_accuracy",
            dtype=stable_dtype,
        )
        self.distil_accuracy_tracker = metrics.SparseCategoricalAccuracy(
            name="distil_token_accuracy",
            dtype=stable_dtype,
        )

    def _prepare_classifier_batch(
        self,
        inputs: tuple[tf.Tensor, ...],
        use_label_dropout: bool = True,
    ) -> tuple[tuple[tf.Tensor, ...], tf.Tensor | None, tf.Tensor | None]:
        """Separate student inputs, teacher targets, and replay provenance.

        Args:
            inputs (tuple[tf.Tensor, ...]): Raw ``(images, classes)`` or
                ``(images, classes, replay_mask)`` data, or their mapped
                seven-tensor equivalents with an optional teacher target and
                final replay mask.
            use_label_dropout (bool): Whether raw input preparation applies
                classifier-free label dropout.

        Returns:
            tuple[tuple[tf.Tensor, ...], tf.Tensor | None, tf.Tensor | None]:
            Seven student tensors, optional teacher probabilities, and the
            optional boolean/float replay mask.

        Raises:
            ValueError: Mapped input has an unexpected number of tensors.
        """

        replay_mask = None
        # Mapped data already contains the seven diffusion input tensors.
        if self.map_preprocess:
            expected_length = 7 + int(self.use_teacher)
            # Remove a supplied replay mask before teacher-target extraction.
            if len(inputs) == expected_length + 1:
                inputs, replay_mask = inputs[:-1], inputs[-1]
            # Reject mapped structures that cannot be unambiguously decoded.
            elif len(inputs) != expected_length:
                raise ValueError(
                    "Mapped classifier batches must contain seven student "
                    "tensors, an optional teacher target, and an optional "
                    "final replay mask."
                )
            prepared_inputs = inputs
        # Treat a raw third tensor as replay provenance, not sample weighting.
        else:
            raw_inputs = inputs
            # Separate optional replay provenance from the raw supervised pair.
            if len(inputs) == 3:
                raw_inputs, replay_mask = inputs[:2], inputs[-1]
            prepared_inputs = self.prep_inputs(
                raw_inputs,
                use_label_dropout=use_label_dropout,
            )

        # Separate the mapped teacher target from the student input tensors.
        if self.use_teacher:
            prepared_inputs, teacher_labels = (
                prepared_inputs[:-1], prepared_inputs[-1]
            )
        # Keep the ordinary training and evaluation paths teacher-free.
        else:
            teacher_labels = None

        return prepared_inputs, teacher_labels, replay_mask

    def train_step(
        self, 
        inputs: tuple[tf.Tensor, ...]
    ) -> dict[str, tf.Tensor]:
        """Perform one joint raw-network diffusion/classifier update.

        Args:
            inputs (tuple[tf.Tensor, ...]): Clean images and zero-based classes,
                optionally followed by a replay mask, or the prepared tensors
                supplied by ``map_preprocess``.

        Returns:
            dict[str, tf.Tensor]: Running diffusion and classifier metrics.  The
            classifier mask selects CFG-null and/or low-timestep examples when
            configured; divide-no-nan makes an empty selection contribute zero.
        """

        prepared_inputs, teacher_labels, replay_mask = (
            self._prepare_classifier_batch(inputs)
        )
        (x0, noises, 
        t, x_t, 
        cfg_labels, 
        uncond_labels, 
        classes) = prepared_inputs

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        clf_loss_mask = tf.ones_like(cfg_labels, dtype=stable_dtype)
        # Restrict classifier metrics and loss to CFG-dropped examples.
        if self.mask_by_nulls:
            null_ids = (cfg_labels == 0)
            clf_loss_mask = clf_loss_mask * tf.cast(
                null_ids,
                dtype=stable_dtype
            )
        # Restrict classifier metrics and loss to the configured leading timesteps.
        if self.mask_by_t_threshold:
            not_exceeded_t_threshold_ids = (t <= self.filter_t_threshold)
            clf_loss_mask = clf_loss_mask * tf.cast(
                not_exceeded_t_threshold_ids,
                dtype=stable_dtype
            )
        clf_acc_mask = tf.cast(
            clf_loss_mask, 
            dtype=tf.bool
        )

        with tf.GradientTape() as tape:
            forward_outputs = self.forward(
                "raw", x_t, t, t, 
                cond_labels=cfg_labels, 
                uncond_labels=uncond_labels, 
                scale=self.train_cfg_scale, 
                training=True
            )
            (x0_pred, noises_pred,
            (regs_list_c, regs_list_u),
            (z_vals_c, z_vals_u),
            (classes_pred_c, classes_pred_u),
            (clf_regs_list_c, clf_regs_list_u),
            (clf_z_vals_c, clf_z_vals_u)) = forward_outputs[:7]

            distil_pred_c = None
            distil_pred_u = None
            # The transformer appends one independent distillation-head pair.
            if self.use_distil_loss:
                distil_pred_c, distil_pred_u = forward_outputs[7]

            (loss1, noise_loss, cond_noise_loss, 
            uncond_noise_loss, image_loss, kl_loss1, 
            ctr_loss1, ctr_preds1) = self.compute_noise_image_kl_ctr_loss(
                x0, noises, classes, 
                x0_pred, noises_pred, 
                z_vals_c, regs_list_c, 
                z_vals_u, regs_list_u,
                cond_labels=cfg_labels,
            )
            (loss2, clf_loss, kl_loss2, 
            ctr_loss2, distil_loss, 
            classes_pred, ctr_preds2, 
            distil_preds) = self.compute_clf_kl_ctr_distil_loss(
                classes, classes_pred_c, 
                clf_z_vals_c, clf_regs_list_c, 
                distil_pred_c=distil_pred_c, 
                classes_pred_u=classes_pred_u, 
                clf_z_vals_u=clf_z_vals_u, 
                clf_regs_list_u=clf_regs_list_u, 
                distil_pred_u=distil_pred_u, 
                clf_loss_mask=clf_loss_mask, 
                teacher_labels=teacher_labels, 
                replay_mask=replay_mask,
                x0=x0, training=True
            )

            loss = loss1 + loss2

        self.apply_grads(tape, loss)
        self.update_ema()
        results = self.get_results_dict(
            noise_loss, 
            cond_noise_loss=cond_noise_loss, 
            uncond_noise_loss=uncond_noise_loss, 
            total_loss=loss, 
            image_loss=image_loss, 
            kl_loss=kl_loss1, 
            ctr_loss=ctr_loss1, 
            ctr_preds=ctr_preds1, 
            classes=classes, 
            cond_labels=cfg_labels,
            use_total_loss=True
        )
        results.update(self.get_clf_results_dict(
            clf_loss, 
            classes, 
            classes_pred, 
            clf_acc_mask, 
            clf_kl_loss=kl_loss2, 
            clf_ctr_loss=ctr_loss2, 
            clf_distil_loss=distil_loss, 
            clf_ctr_preds=ctr_preds2, 
            clf_distil_preds=distil_preds, 
            use_total_loss=False,
            distil_acc_mask=self._distillation_metric_mask(
                classes,
                teacher_labels,
                replay_mask,
                clf_acc_mask,
            ) if self.use_distil_loss else None,
        ))

        return results

    def test_step(
        self, 
        inputs: tuple[tf.Tensor, ...]
    ) -> dict[str, tf.Tensor]:
        """Evaluate diffusion plus unconditional clean-image classification.

        Diffusion metrics use noisified inputs and the configured test CFG scale.
        Classifier metrics instead call ``predict_class`` on clean ``x0`` with
        timestep 0 and null labels, so test classifier loss intentionally differs
        from masked/noisy training classifier loss.

        Args:
            inputs (tuple[tf.Tensor, ...]): Clean images and zero-based classes,
                optionally followed by a replay mask, or the prepared tensors
                supplied by ``map_preprocess``.

        Returns:
            dict[str, tf.Tensor]: Running enabled diffusion and classifier
            losses/accuracies.
        """

        prepared_inputs, teacher_labels, replay_mask = (
            self._prepare_classifier_batch(
                inputs,
                use_label_dropout=False,
            )
        )
        (x0, noises, 
        t, x_t, 
        cond_labels, 
        uncond_labels, 
        classes) = prepared_inputs

        (loss1, noise_loss, cond_noise_loss, 
        uncond_noise_loss, image_loss, kl_loss1, 
        ctr_loss1, ctr_preds1) = super().forward_and_compute_loss(
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
        class_outputs = self.get_network(self.test_network_name).predict_class(
            (x0, zero_ts, uncond_labels), 
            full_return=True, 
            training=False
        )

        classes_pred = class_outputs[0]
        clf_regs_list = class_outputs[3]
        clf_z_vals = class_outputs[4]
        # Read the independent distillation head when configured.
        if self.use_distil_loss:
            distil_preds = class_outputs[5]
        # Avoid reporting token metrics when the network has no active token loss.
        else:
            distil_preds = None

        (loss2, clf_loss, kl_loss2, 
        ctr_loss2, distil_loss, 
        classes_pred, ctr_preds2, 
        distil_preds) = self.compute_clf_kl_ctr_distil_loss(
            classes, None, None, None, None, 
            classes_pred, clf_z_vals, 
            clf_regs_list, distil_preds, 
            clf_train_type="uncond", 
            kl_train_type="uncond", 
            ctr_train_type="uncond", 
            teacher_labels=teacher_labels, 
            replay_mask=replay_mask,
            x0=x0,
            training=False
        )

        loss = loss1 + loss2

        results = self.get_results_dict(
            noise_loss, 
            cond_noise_loss=cond_noise_loss, 
            uncond_noise_loss=uncond_noise_loss, 
            total_loss=loss, 
            image_loss=image_loss, 
            kl_loss=kl_loss1, 
            ctr_loss=ctr_loss1, 
            ctr_preds=ctr_preds1, 
            classes=classes, 
            cond_labels=cond_labels,
            use_image_loss=True
        )
        results.update(self.get_clf_results_dict(
            clf_loss, 
            classes, 
            classes_pred, 
            clf_kl_loss=kl_loss2, 
            clf_ctr_loss=ctr_loss2, 
            clf_distil_loss=distil_loss, 
            clf_ctr_preds=ctr_preds2, 
            clf_distil_preds=distil_preds, 
            use_total_loss=False,
            distil_acc_mask=self._distillation_metric_mask(
                classes,
                teacher_labels,
                replay_mask,
            ) if self.use_distil_loss else None,
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
                including ``network_name`` (or legacy ``netwrok_name``),
                ``compute_type``, ``weighted``,
                ``max_t``, ``t_chunk_size``, and ``seed``.

        Returns:
            float: Sparse categorical accuracy across the full dataset.
        """

        # Replay-only continual datasets use their third tensor as KD
        # provenance, not as an accuracy sample weight. Remove that metadata
        # before the generic ensemble metric interprets Keras triples.
        element_spec = getattr(dataset, "element_spec", None)

        # Strip replay metadata only from the documented replay-only triple.
        if self.distil_scope == "replay_only" \
        and isinstance(element_spec, (tuple, list)) \
        and len(element_spec) == 3:
            def _drop_replay_provenance(
                images: tf.Tensor, 
                labels: tf.Tensor, 
                replay_mask: tf.Tensor
            ) -> tuple[tf.Tensor, tf.Tensor]:
                """Remove KD-only provenance before generic accuracy evaluation.

                Args:
                    images (tf.Tensor): Clean image batch.
                    labels (tf.Tensor): Sparse class-label batch.
                    replay_mask (tf.Tensor): Per-row replay indicator, unused by accuracy.

                Returns:
                    tuple[tf.Tensor, tf.Tensor]: The unchanged image/label pair.
                """

                del replay_mask

                return images, labels


            dataset = dataset.map(
                _drop_replay_provenance, 
                num_parallel_calls=self.map_num_parallel_calls
            )

        # Default to the configured test network, or raw when EMA is disabled.
        # Prefer the corrected selector unless either spelling was supplied.
        if "network_name" not in kwargs:
            kwargs["network_name"] = self.test_network_name

        kwargs.setdefault(
            "max_t", 
            min(128, self.timesteps)
        )
        kwargs.setdefault(
            "clf_acc_coef", 
            self.clf_acc_coef
        )
        kwargs.setdefault(
            "ctr_acc_coef",
            self.ctr_acc_coef if self.use_clf_ctr_loss else 0.
        )
        kwargs.setdefault(
            "distil_acc_coef", 
            self.distil_acc_coef if self.use_distil_loss else 0.
        )
        kwargs.setdefault("seed", self.seed)
        kwargs.setdefault("dtype", self.dtype_policy.variable_dtype)

        ensemble_accuracy = EnsembleAccuracy(
            self, 
            **kwargs
        )
        accuracy_value = ensemble_accuracy.evaluate(
            dataset, 
            verbose=verbose
        )

        return float(accuracy_value)

    def prep_inputs_map(
        self, 
        x0: tf.Tensor, 
        labels: tf.Tensor,
        replay_mask: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, ...]:
        """Prepare one input-pipeline batch and optional teacher targets.

        Args:
            x0 (tf.Tensor): Clean image batch.
            labels (tf.Tensor): Dataset class labels.
            replay_mask (tf.Tensor | None): Optional per-example replay
                provenance. The continual learner supplies this as the third
                dataset tensor for ``distil_scope="replay_only"``.

        Returns:
            tuple[tf.Tensor, ...]: The seven values from :meth:`prep_inputs`
            and, during distillation, frozen teacher probabilities. A supplied
            replay mask is retained as the final tensor.
        """

        prepared_inputs = super().prep_inputs_map(x0, labels)

        # Preserve the ordinary mapped batch when no teacher target is used.
        if not self.use_teacher:
            return prepared_inputs if replay_mask is None else (
                *prepared_inputs, replay_mask
            )

        # Evaluate validation teachers on the established clean-input path.
        if self._preprocess_training is False:
            teacher_x = prepared_inputs[0] # x0
            teacher_t = tf.zeros_like(prepared_inputs[2])
            teacher_labels_in = prepared_inputs[5] # uncond_labels
        # Match training teachers to the student's noisified classifier path.
        else:
            teacher_x = prepared_inputs[3] # x_t
            teacher_t = prepared_inputs[2] # t
            teacher_labels_in = prepared_inputs[
                4 if self.clf_train_type == "cond" else 5
            ] # cfg_labels or uncond_labels

        # Map only unseen condition IDs to null for narrower past teachers.
        teacher_labels_in = self._mask_unknown_teacher_labels(
            teacher_labels_in
        )

        teacher_labels = self._predict_teacher_labels(
            teacher_x, 
            teacher_t, 
            teacher_labels_in
        )

        teacher_inputs = (*prepared_inputs, teacher_labels)
        return teacher_inputs if replay_mask is None else (
            *teacher_inputs, replay_mask
        )

    def set_current_resolution(self, resolution: int | None = None) -> None:
        """Set the active student and compatible teacher resolution.

        Args:
            resolution (int | None): Square input size, or ``None`` for the
                student's constructor size.

        Returns:
            None: Student, EMA, and a compatible frozen teacher are updated.
        """

        resolution = self.image_size if resolution is None else resolution
        super().set_current_resolution(resolution)

        # Keep compatible teacher positional embeddings aligned with the student.
        if self.teacher_network is not None and hasattr(
            self.teacher_network, "set_current_resolution"
        ):
            self.teacher_network.set_current_resolution(
                resolution
            )

    def set_teacher_network(
        self, 
        teacher_network: tf.keras.Model | None
    ) -> None:
        """Attach or clear the frozen runtime teacher used for distillation.

        Args:
            teacher_network (tf.keras.Model | None): A compatible raw
                classifier, a diffusion-classifier wrapper, or ``None``. A
                wrapper is reduced to its raw network. Clearing a required
                teacher is allowed only when ``defer_teacher=True``.

        Returns:
            None: Teacher loss flags, input mapping, active resolution, and
            cached Keras execution functions are updated in place.

        Raises:
            ValueError: If a required teacher is cleared without deferred mode,
            or the student/its live EMA network is supplied as its own teacher.
        """

        # Unwrap a diffusion wrapper to its effective raw classifier network.
        if teacher_network is not None and getattr(
            teacher_network, "network", None
        ) is not None:
            teacher_network = teacher_network.network

        # Reject stale serialized layer policies before teacher inference.
        if teacher_network is not None:
            validate_model_dtype_policy(
                teacher_network,
                self.dtype_policy,
                role="teacher_network",
            )

        # A live student branch is not a frozen snapshot and would be disabled.
        if teacher_network is not None and (
            teacher_network is self.network
            or teacher_network is self.ema_network
        ):
            raise ValueError(
                "teacher_network must be an independent frozen snapshot."
            )

        # Preserve strict ordinary construction while allowing task-1 deferral.
        if teacher_network is None and \
        (getattr(self, "use_distil_loss", False) or
         getattr(self, "use_distil_ctr_loss", False)) \
        and not self.defer_teacher:
            raise ValueError(
                "A configured distillation objective requires teacher_network; "
                "set defer_teacher=True only when it will be attached later."
            )

        # Runtime teachers are external targets, not checkpoint-owned submodels.
        object.__setattr__(self, "teacher_network", teacher_network)

        # Keep teacher variables outside optimization and align image geometry.
        if self.teacher_network is not None:
            self.teacher_network.trainable = False
            self.map_preprocess = True

            # Mirror progressive resolution on compatible classifier networks.
            if hasattr(self.teacher_network, "set_current_resolution"):
                self.teacher_network.set_current_resolution(
                    self._current_resolution
                )
        # Restore the caller's original mapping choice after teacher removal.
        else:
            self.map_preprocess = self._map_preprocess_without_teacher

        self._refresh_loss_flags()
        self.train_function = None
        self.test_function = None
        self.predict_function = None

    def snapshot_teacher_network(
        self, 
        network_name: NetworkName = "raw"
    ) -> tf.keras.Model:
        """Clone one current classifier branch as an independent frozen teacher.

        The clone uses the selected raw/EMA network's current serialized
        topology. This retains dynamic class width and progressive depth, then
        copies its weights and active resolution without optimizer state.

        Args:
            network_name (NetworkName): ``"raw"`` or ``"ema"``. ``"ema"``
                falls back to raw when EMA tracking is disabled, matching
                :meth:`get_network`.

        Returns:
            tf.keras.Model: A built, non-trainable classifier network whose
            variables are independent from the student.
        """

        source_network = self.get_network(network_name)
        teacher_network = source_network.__class__.from_config(
            source_network.get_config()
        )
        teacher_network.build()

        # Reproduce the active progressive resolution after base construction.
        if hasattr(teacher_network, "set_current_resolution"):
            teacher_network.set_current_resolution(
                self._current_resolution
            )

        copy_network_weights_by_layer(
            source_network, 
            teacher_network
        )
        teacher_network.trainable = False

        return teacher_network

    def call_network(
        self, 
        x_t: tf.Tensor, 
        t_batch: tf.Tensor, 
        cond_labels: tf.Tensor, 
        uncond_labels: tf.Tensor | None = None, 
        scale: float | None = None, 
        network_name: NetworkName = "raw", 
        training: bool = False
    ) -> tuple[tuple[object, object | None], ...]:
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
            tuple[tuple[object, object | None], ...]: Six conditional/
            unconditional pairs in order: noise tensors, main regularizer
            lists, main latent-statistic pairs, classifier probabilities,
            classifier regularizer lists, and classifier latent-statistic
            pairs. Distillation mode appends one distillation-token probability
            pair. Unconditional members are None when no second pass is
            requested.
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

        eps_c, eps_u = output_dict_c["noises"], output_dict_u.get("noises")
        regs_list_c, regs_list_u = output_dict_c["regs_list"], output_dict_u.get("regs_list")
        z_vals_c, z_vals_u = output_dict_c["z_vals"], output_dict_u.get("z_vals")
        classes_pred_c, classes_pred_u = output_dict_c["classes"], output_dict_u.get("classes")
        clf_regs_list_c, clf_regs_list_u = output_dict_c["clf_regs_list"], output_dict_u.get("clf_regs_list")
        clf_z_vals_c, clf_z_vals_u = output_dict_c["clf_z_vals"], output_dict_u.get("clf_z_vals")

        outputs = (
            (eps_c, eps_u), 
            (regs_list_c, regs_list_u), 
            (z_vals_c, z_vals_u), 
            (classes_pred_c, classes_pred_u), 
            (clf_regs_list_c, clf_regs_list_u), 
            (clf_z_vals_c, clf_z_vals_u)
        )

        # Append the independent distillation pair only when it is optimized.
        if self.use_distil_loss:
            outputs += ((
                output_dict_c["distil_classes"], 
                output_dict_u.get("distil_classes")
            ),)

        return outputs

    def compute_clf_loss(
        self, 
        classes: tf.Tensor, 
        classes_pred: tf.Tensor, 
        clf_loss_mask: tf.Tensor | None = None, 
        x0: tf.Tensor | None = None, 
        training: bool | None = None
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute masked primary-head classification loss.

        Args:
            classes (tf.Tensor): Zero-based targets shaped ``[B]``.
            classes_pred (tf.Tensor): Primary class probabilities shaped
                ``[B,num_classes]``.
            clf_loss_mask (tf.Tensor | None): Optional float sample weights
                shaped ``[B]``.
            x0 (tf.Tensor | None): Clean images required by ensemble loss.
            training (bool | None): Mode forwarded to ensemble prediction.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Scalar loss and the primary or
            ensemble probabilities used to compute it.
        """

        # Use clean-image ensemble predictions when ensemble loss is enabled.
        if self.ensemble_loss_fn is not None:
            classes_pred = self.ensemble_loss_fn.ensemble_predict_batched(
                x0, 
                training=training
            )

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        stable_predictions = tf.cast(classes_pred, stable_dtype)
        clf_loss = tf.cast(
            self.scce_loss_fn(classes, stable_predictions), stable_dtype
        )
        clf_loss_mask = tf.cast(clf_loss_mask, stable_dtype) \
            if clf_loss_mask is not None else None
        clf_loss = tf.math.divide_no_nan(
            tf.reduce_sum(clf_loss * clf_loss_mask),
            tf.reduce_sum(clf_loss_mask)
        ) if clf_loss_mask is not None else tf.reduce_mean(clf_loss)

        return clf_loss, classes_pred

    def _distillation_metric_mask(
        self,
        classes: tf.Tensor,
        teacher_labels: tf.Tensor | None,
        replay_mask: tf.Tensor | None,
        classifier_mask: tf.Tensor | None = None,
    ) -> tf.Tensor | None:
        """Return the exact rows represented by scoped KD metrics.

        Args:
            classes (tf.Tensor): Zero-based student targets shaped ``[B]``.
            teacher_labels (tf.Tensor | None): Frozen teacher probabilities;
                their width defines the old-class vocabulary.
            replay_mask (tf.Tensor | None): Per-row replay provenance.
            classifier_mask (tf.Tensor | None): Optional classifier selection
                mask intersected with the KD scope.

        Returns:
            tf.Tensor | None: Boolean row selector, or ``None`` when both the
            KD scope and classifier metrics include every row.

        Raises:
            ValueError: If the configured scope lacks required metadata.
        """

        scope_mask = None
        # Select targets represented by the frozen teacher vocabulary.
        if self.distil_scope == "old_classes":
            # Vocabulary membership cannot be inferred without teacher width.
            if teacher_labels is None:
                raise ValueError(
                    "teacher_labels are required for old_classes metrics."
                )
            scope_mask = tf.reshape(classes, (-1,)) < tf.cast(
                tf.shape(teacher_labels)[-1], classes.dtype
            )
        # Select only rows explicitly originating from replay.
        elif self.distil_scope == "replay_only":
            # Replay provenance must be supplied explicitly by the learner.
            if replay_mask is None:
                raise ValueError(
                    "replay_mask is required for replay_only metrics."
                )
            scope_mask = tf.cast(tf.reshape(replay_mask, (-1,)), tf.bool)

        # Classifier timestep/CFG filtering also applies to KD measurements.
        if classifier_mask is not None:
            classifier_mask = tf.cast(
                tf.reshape(classifier_mask, (-1,)), tf.bool
            )
            scope_mask = classifier_mask if scope_mask is None else tf.logical_and(
                scope_mask, classifier_mask
            )

        return scope_mask

    def compute_distil_loss(
        self, 
        teacher_labels: tf.Tensor, 
        distil_preds: tf.Tensor,
        distil_type: Literal["hard", "soft"] | None = None, 
        distil_loss_mask: tf.Tensor | None = None,
        classes: tf.Tensor | None = None,
        replay_mask: tf.Tensor | None = None,
        distil_temperature: float | None = None,
        distil_scope: Literal[
            "old_classes", "replay_only", "current_and_replay"
        ] | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute hard-label CE or soft-label KL distillation loss.

        Args:
            teacher_labels (tf.Tensor): Teacher probabilities
                ``[B,num_classes]``.
            distil_preds (tf.Tensor): Distillation-head probabilities with the
                same batch and class dimensions.
            distil_type (Literal["hard", "soft"] | None): Loss mode;
                ``None`` uses the wrapper's token-distillation setting.
            distil_loss_mask (tf.Tensor | None): Optional float sample weights
                shaped ``[B]``.
            classes (tf.Tensor | None): Sparse zero-based targets used to
                identify examples from teacher-known classes.
            replay_mask (tf.Tensor | None): Boolean/float replay provenance
                used only by the ``"replay_only"`` scope.
            distil_temperature (float | None): Positive soft-KD temperature;
                ``None`` uses the configured value. Hard KD ignores it.
            distil_scope (Literal["old_classes", "replay_only",
                "current_and_replay"] | None): Example scope; ``None`` uses
                the configured value.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Scalar unweighted distillation loss
            and the unchanged student probabilities.

        Raises:
            ValueError: Temperature or scope is invalid, or the selected
                scope lacks its required class labels or replay mask.
        """

        distil_type = self.distil_type if distil_type is None else distil_type
        distil_temperature = self.distil_temperature \
            if distil_temperature is None else distil_temperature
        distil_scope = self.distil_scope \
            if distil_scope is None else distil_scope
        # Keep probability softening within its finite positive domain.
        if not isinstance(distil_temperature, Real) \
        or isinstance(distil_temperature, bool) \
        or not np.isfinite(distil_temperature) \
        or distil_temperature <= 0.:
            raise ValueError(
                "distil_temperature must be a finite positive number."
            )
        # Reject unknown sample-selection policies before tensor computation.
        if distil_scope not in (
            "old_classes", "replay_only", "current_and_replay"
        ):
            raise ValueError(
                "distil_scope must be 'old_classes', 'replay_only', or "
                "'current_and_replay'."
            )

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        stable_distil_preds = tf.cast(distil_preds, stable_dtype)
        teacher_labels = tf.cast(teacher_labels, stable_dtype)

        teacher_width = tf.shape(teacher_labels)[-1]
        student_width = tf.shape(distil_preds)[-1]
        known_teacher_labels = teacher_labels[:, :student_width]
        known_teacher_labels = tf.math.divide_no_nan(
            known_teacher_labels,
            tf.reduce_sum(
                known_teacher_labels,
                axis=-1,
                keepdims=True,
            ),
        )
        missing_width = tf.maximum(
            student_width - tf.shape(known_teacher_labels)[-1], 0
        )
        teacher_labels = tf.pad(
            known_teacher_labels,
            [[0, 0], [0, missing_width]]
        )

        # Convert teacher probabilities to labels for hard distillation.
        if distil_type == "hard":
            hard_labels = tf.argmax(
                teacher_labels, 
                axis=-1, 
                output_type=tf.int32
            )
            distil_loss = self.scce_loss_fn(
                hard_labels, 
                stable_distil_preds
            )
        # Preserve the complete teacher distribution for soft distillation.
        else:
            # Keep T=1 on the exact historical direct-probability KL path.
            if distil_temperature == 1.:
                distil_loss = self.kld_loss_fn(
                    teacher_labels,
                    stable_distil_preds,
                )
            # Recover temperature-scaled distributions from probabilities.
            else:
                epsilon = tf.cast(tf.keras.backend.epsilon(), stable_dtype)
                teacher_soft = tf.nn.softmax(
                    tf.math.log(tf.maximum(known_teacher_labels, epsilon))
                    / tf.cast(distil_temperature, stable_dtype),
                    axis=-1,
                )
                # New student classes are outside the frozen teacher support;
                # append exact zeros only after temperature softening.
                teacher_soft = tf.pad(
                    teacher_soft,
                    [[0, 0], [0, missing_width]],
                )
                student_soft = tf.nn.softmax(
                    tf.math.log(tf.maximum(stable_distil_preds, epsilon))
                    / tf.cast(distil_temperature, stable_dtype),
                    axis=-1,
                )
                distil_loss = self.kld_loss_fn(
                    teacher_soft,
                    student_soft,
                ) * tf.cast(distil_temperature ** 2, stable_dtype)

        distil_loss = tf.cast(distil_loss, stable_dtype)
        scope_mask = None
        # Select all examples whose ground-truth class exists in the teacher.
        if distil_scope == "old_classes":
            # Require labels before deriving membership in the old vocabulary.
            if classes is None:
                raise ValueError(
                    "classes are required for distil_scope='old_classes'."
                )
            scope_mask = tf.reshape(classes, (-1,)) < tf.cast(
                teacher_width,
                classes.dtype,
            )
        # Replay provenance cannot be inferred from class IDs in cumulative CL.
        elif distil_scope == "replay_only":
            # Require the learner's explicit row-level provenance indicator.
            if replay_mask is None:
                raise ValueError(
                    "replay_mask is required for "
                    "distil_scope='replay_only'."
                )
            scope_mask = tf.reshape(replay_mask, (-1,))

        distil_loss_mask = tf.cast(distil_loss_mask, stable_dtype) \
            if distil_loss_mask is not None else None
        # Intersect the scope selector with any classifier-training mask.
        if scope_mask is not None:
            scope_mask = tf.cast(scope_mask, stable_dtype)
            distil_loss_mask = scope_mask if distil_loss_mask is None \
                else distil_loss_mask * scope_mask
        distil_loss = tf.math.divide_no_nan(
            tf.reduce_sum(
                distil_loss * distil_loss_mask
            ), 
            tf.reduce_sum(distil_loss_mask)
        ) if distil_loss_mask is not None else tf.reduce_mean(distil_loss)

        return distil_loss, distil_preds

    def compute_distil_ctr_loss(
        self, 
        classes: tf.Tensor, 
        classes_pred_list: list[tf.Tensor], 
        teacher_labels: tf.Tensor | None = None,
        replay_mask: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor | float, tf.Tensor | float]:
        """Compute ordinary or teacher-targeted token regularizer loss.

        Args:
            classes (tf.Tensor): Zero-based dataset targets shaped ``[B]``.
            classes_pred_list (list[tf.Tensor]): Available regularizer class
                probabilities.
            teacher_labels (tf.Tensor | None): Frozen teacher probabilities
                used by ``"distil"`` and ``"both"`` training modes.
            replay_mask (tf.Tensor | None): Optional replay provenance passed
                to scoped teacher-targeted regularization.

        Returns:
            tuple[tf.Tensor | float, tf.Tensor | float]: Regularizer loss and
            averaged regularizer probabilities, or zeros when disabled.
        """

        clf_ctr_loss, clf_ctr_preds = self.compute_ctr_loss(
            classes, 
            classes_pred_list
        ) if self.use_clf_ctr_loss else (0., 0.)

        # Retain ordinary regularizer targets when distillation is inactive.
        if not self.use_distil_ctr_loss:
            return clf_ctr_loss, clf_ctr_preds

        regularizer_kwargs = getattr(
            self.network, "clf_cls_token_regularizer_kwargs", None
        )
        regularizer_kwargs = getattr(
            self.network, "cls_token_regularizer_kwargs", {}
        ) if regularizer_kwargs is None else regularizer_kwargs
        regularizer_train_type = regularizer_kwargs.get(
            "train_type", "normal"
        )
        # Replace or blend the regularizer target according to its train mode.
        if self.use_clf_ctr_loss and regularizer_train_type in (
            "distil", "both"
        ):
            distil_ctr_loss, distil_ctr_preds = self.compute_distil_loss(
                teacher_labels, 
                clf_ctr_preds, 
                regularizer_kwargs.get("distil_type", "hard"),
                classes=classes,
                replay_mask=replay_mask,
            )
            clf_ctr_loss = distil_ctr_loss if regularizer_train_type == "distil" \
                        else (clf_ctr_loss + distil_ctr_loss) / 2.

        return clf_ctr_loss, clf_ctr_preds

    def compute_clf_kl_ctr_distil_loss(
        self, 
        classes: tf.Tensor, 
        classes_pred_c: tf.Tensor | None, 
        clf_z_vals_c: tuple[tf.Tensor | None, tf.Tensor | None] | None, 
        clf_regs_list_c: list[tf.Tensor | None] | None, 
        distil_pred_c: tf.Tensor | None = None, 
        classes_pred_u: tf.Tensor | None = None, 
        clf_z_vals_u: tuple[tf.Tensor | None, tf.Tensor | None] | None = None, 
        clf_regs_list_u: list[tf.Tensor | None] | None = None, 
        distil_pred_u: tf.Tensor | None = None, 
        clf_loss_mask: tf.Tensor | None = None, 
        clf_train_type: TrainType | None = None, 
        kl_train_type: TrainType | None = None, 
        ctr_train_type: TrainType | None = None, 
        teacher_labels: tf.Tensor | None = None, 
        x0: tf.Tensor | None = None, 
        training: bool | None = None,
        replay_mask: tf.Tensor | None = None,
    ) -> tuple[
        tf.Tensor, tf.Tensor, tf.Tensor | float, tf.Tensor | float,
        tf.Tensor | float, tf.Tensor, tf.Tensor | float, tf.Tensor | None
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
            distil_pred_c (tf.Tensor | None): Conditional distillation-head
                probabilities.
            classes_pred_u (tf.Tensor | None): Unconditional probabilities.
            clf_z_vals_u (tuple[tf.Tensor, tf.Tensor] | None): Unconditional
                classifier latent statistics.
            clf_regs_list_u (list[tf.Tensor | None] | None): Unconditional token
                predictions.
            distil_pred_u (tf.Tensor | None): Unconditional distillation-head
                probabilities.
            clf_loss_mask (tf.Tensor | None): Float per-example mask ``[B]``;
                None averages all cross-entropies.
            clf_train_type (TrainType | None): Probability source; None uses the
                configured ``clf_train_type``.
            kl_train_type (TrainType | None): Classifier KL source; None uses the
                base setting.
            ctr_train_type (TrainType | None): Classifier token-loss source.
            teacher_labels (tf.Tensor | None): Frozen teacher probabilities
                used when the regularizer ``train_type`` is ``"distil"`` or
                ``"both"``.
            x0 (tf.Tensor | None): Clean images required by ensemble loss.
            training (bool | None): Mode passed to the ensemble predictor.
            replay_mask (tf.Tensor | None): Optional replay provenance for
                scoped distillation losses.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor | float, tf.Tensor | float,
            tf.Tensor | float, tf.Tensor, tf.Tensor | float, tf.Tensor | None]:
            Weighted classifier total, raw classifier loss, classifier KL loss,
            classifier token loss, distillation loss, selected probabilities,
            averaged token probabilities, and distillation probabilities.
        """

        clf_train_type = self.clf_train_type if clf_train_type is None else clf_train_type
        kl_train_type = self.kl_train_type if kl_train_type is None else kl_train_type
        ctr_train_type = self.ctr_train_type if ctr_train_type is None else ctr_train_type

        clf_loss, classes_pred = self.compute_clf_loss(
            classes, 
            classes_pred=classes_pred_c if clf_train_type == "cond" else classes_pred_u, 
            clf_loss_mask=clf_loss_mask, 
            x0=x0, 
            training=training
        )
        clf_kl_loss = VariationalAutoencoder.compute_kl(
            z_mean=clf_z_vals_c[0] if kl_train_type == "cond" else clf_z_vals_u[0], 
            z_log_var=clf_z_vals_c[1] if kl_train_type == "cond" else clf_z_vals_u[1],
            dtype=self.dtype_policy.variable_dtype,
        ) if self.use_clf_kl_loss else 0.
        clf_ctr_loss, clf_ctr_preds = self.compute_distil_ctr_loss(
            classes, 
            classes_pred_list=clf_regs_list_c if ctr_train_type == "cond" else clf_regs_list_u, 
            teacher_labels=teacher_labels,
            replay_mask=replay_mask,
        )
        # Compute the independent distillation-token objective when enabled.
        if self.use_distil_loss:
            clf_distil_loss, distil_preds = self.compute_distil_loss(
                teacher_labels,
                distil_preds=(
                    distil_pred_c if clf_train_type == "cond" else distil_pred_u
                ),
                distil_loss_mask=clf_loss_mask,
                classes=classes,
                replay_mask=replay_mask,
            )
        # Keep disabled distillation outputs explicit for metric dispatch.
        else:
            clf_distil_loss, distil_preds = 0., None

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        clf_loss = tf.cast(clf_loss, stable_dtype)
        clf_kl_loss = tf.cast(clf_kl_loss, stable_dtype)
        clf_ctr_loss = tf.cast(clf_ctr_loss, stable_dtype)
        clf_distil_loss = tf.cast(clf_distil_loss, stable_dtype)
        loss = (
            clf_loss * self.clf_loss_coef + 
            clf_kl_loss * self.kl_loss_coef + 
            clf_ctr_loss * self.ctr_loss_coef + 
            clf_distil_loss * self.distil_loss_coef
        )

        outputs = (
            loss, clf_loss, clf_kl_loss, 
            clf_ctr_loss, clf_distil_loss, 
            classes_pred, clf_ctr_preds, 
            distil_preds
        )


        return outputs

    def get_clf_results_dict(
        self, 
        clf_loss: tf.Tensor, 
        classes: tf.Tensor, 
        classes_pred: tf.Tensor, 
        clf_acc_mask: tf.Tensor | None = None, 
        total_loss: tf.Tensor | None = None, 
        clf_kl_loss: tf.Tensor | None = None, 
        clf_ctr_loss: tf.Tensor | None = None, 
        clf_distil_loss: tf.Tensor | None = None, 
        clf_ctr_preds: tf.Tensor | None = None, 
        clf_distil_preds: tf.Tensor | None = None, 
        use_total_loss: bool | None = None,
        use_kl_loss: bool | None = None,
        use_ctr_loss: bool | None = None, 
        use_distil_loss: bool | None = None,
        distil_acc_mask: tf.Tensor | None = None,
    ) -> dict[str, tf.Tensor]:
        """Update classifier metric trackers and return current values.

        Args:
            clf_loss (tf.Tensor): Required scalar classifier loss.
            classes (tf.Tensor): Zero-based targets ``[B]``.
            classes_pred (tf.Tensor): Class probabilities ``[B,num_classes]``.
            clf_acc_mask (tf.Tensor | None): Boolean selector ``[B]`` for
                masked loss and accuracy tracking; None selects all samples.
            total_loss (tf.Tensor | None): Required when total tracking is on.
            clf_kl_loss (tf.Tensor | None): Required when classifier KL is on.
            clf_ctr_loss (tf.Tensor | None): Required when token loss is on.
            clf_distil_loss (tf.Tensor | None): Required when distillation is
                active.
            clf_ctr_preds (tf.Tensor | None): Token probabilities for accuracy.
            clf_distil_preds (tf.Tensor | None): Distillation-head class
                probabilities.
            use_total_loss (bool | None): Explicit total tracker switch; None
                enables it for classifier KL/token auxiliaries.
            use_kl_loss (bool | None): Classifier KL tracker override.
            use_ctr_loss (bool | None): Classifier token tracker override.
            use_distil_loss (bool | None): Distillation tracker override.
            distil_acc_mask (tf.Tensor | None): Optional boolean selector for
                distillation loss/accuracy accounting. ``None`` preserves the
                historical classifier-mask scope.

        Returns:
            dict[str, tf.Tensor]: Current classifier metrics keyed by name.

        Raises:
            AssertionError: If a requested optional metric lacks its input.
        """

        clf_acc_mask = slice(None) if clf_acc_mask is None else clf_acc_mask
        distil_acc_mask = slice(None) if distil_acc_mask is None else distil_acc_mask
        use_kl_loss = self.use_clf_kl_loss if use_kl_loss is None else use_kl_loss
        use_ctr_loss = self.use_clf_ctr_loss if use_ctr_loss is None else use_ctr_loss
        use_distil_loss = self.use_distil_loss if use_distil_loss is None else use_distil_loss
        use_total_loss = use_kl_loss or use_ctr_loss or use_distil_loss \
                        if use_total_loss is None else use_total_loss
        distil_acc_mask = clf_acc_mask if distil_acc_mask is None else distil_acc_mask

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        batch_weight = tf.cast(tf.shape(classes)[0], stable_dtype)
        selected_weight = tf.reduce_sum(
            tf.cast(clf_acc_mask, stable_dtype)
        ) if clf_acc_mask is not None else batch_weight 
        distil_selected_weight = tf.reduce_sum(
                tf.cast(distil_acc_mask, stable_dtype)
        ) if distil_acc_mask is not None else batch_weight 

        # TensorFlow 2.10 misreads one-column predictions as binary outputs.
        if self.network.dynamic_num_classes:
            # Pad the primary prediction while the dynamic head has one class.
            if classes_pred.shape[-1] == 1:
                classes_pred = tf.concat([
                    classes_pred, 
                    tf.zeros_like(classes_pred)
                ], axis=-1)

            # Apply the same compatibility padding to token predictions.
            if tf.is_tensor(clf_ctr_preds) and clf_ctr_preds.shape[-1] == 1:
                clf_ctr_preds = tf.concat([
                    clf_ctr_preds, 
                    tf.zeros_like(clf_ctr_preds)
                ], axis=-1)

            # Apply the same compatibility padding to distillation predictions.
            if tf.is_tensor(clf_distil_preds) and clf_distil_preds.shape[-1] == 1:
                clf_distil_preds = tf.concat([
                    clf_distil_preds, 
                    tf.zeros_like(clf_distil_preds)
                ], axis=-1)

        self.clf_loss_tracker.update_state(
            clf_loss, 
            sample_weight=selected_weight
        )
        self.accuracy_tracker.update_state(
            classes[clf_acc_mask], 
            classes_pred[clf_acc_mask]
        )

        results = {}

        # Update total loss only when the caller enabled that tracker.
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

        results.update({
            self.clf_loss_tracker.name: 
            self.clf_loss_tracker.result()
        })

        # Update classifier KL loss only when its objective is active.
        if use_kl_loss:
            require(
                clf_kl_loss is not None, 
                "When use_kl_loss is True, kl_loss cannot be None."
            )

            self.clf_kl_loss_tracker.update_state(
                clf_kl_loss, 
                sample_weight=batch_weight
            )
            results.update({
                self.clf_kl_loss_tracker.name: 
                self.clf_kl_loss_tracker.result()
            })

        # Update classifier token loss only when predictions are available.
        if use_ctr_loss:
            require(
                clf_ctr_loss is not None and clf_ctr_preds is not None, 
                "When use_ctr_loss is True, clf_ctr_loss "
                "and clf_ctr_preds cannot be None."
            )

            self.clf_ctr_loss_tracker.update_state(
                clf_ctr_loss, 
                sample_weight=batch_weight
            )
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

        # Combine every enabled auxiliary prediction into total accuracy.
        if (use_ctr_loss and self.ctr_acc_coef > 0.) or (
            use_distil_loss and self.distil_acc_coef > 0.
        ):
            total_preds = classes_pred[clf_acc_mask] * self.clf_acc_coef

            # Add classifier-regularizer predictions when weighted in.
            if use_ctr_loss and self.ctr_acc_coef > 0.:
                require(
                    clf_ctr_preds is not None, 
                    "ctr_acc_coef > 0 requires clf_ctr_preds."
                )

                total_preds += (
                    clf_ctr_preds[clf_acc_mask] * self.ctr_acc_coef
                )

            # Add independent distillation-head predictions when weighted in.
            if use_distil_loss and self.distil_acc_coef > 0.:
                require(
                    clf_distil_preds is not None, 
                    "distil_acc_coef > 0 requires clf_distil_preds."
                )

                distil_component = clf_distil_preds[clf_acc_mask]

                # A scoped KD head contributes only on rows in its valid scope.
                if not isinstance(distil_acc_mask, slice):
                    selected_scope = tf.cast(
                        distil_acc_mask[clf_acc_mask], 
                        distil_component.dtype
                    )[:, tf.newaxis]
                    distil_component = distil_component * selected_scope

                total_preds += distil_component * self.distil_acc_coef

            self.total_accuracy_tracker.update_state(
                classes[clf_acc_mask], 
                total_preds,
            )
            results.update({
                self.total_accuracy_tracker.name: 
                self.total_accuracy_tracker.result()
            })

        # Track the independent distillation objective and prediction accuracy.
        if use_distil_loss:
            require(
                clf_distil_loss is not None and clf_distil_preds is not None, 
                "When use_distil_loss is True, clf_distil_loss "
                "and clf_distil_preds cannot be None."
            )

            self.distil_loss_tracker.update_state(
                clf_distil_loss, 
                sample_weight=distil_selected_weight
            )
            self.distil_accuracy_tracker.update_state(
                classes[distil_acc_mask],
                clf_distil_preds[distil_acc_mask]
            )

            results.update({
                self.distil_loss_tracker.name: 
                self.distil_loss_tracker.result(), 
                self.distil_accuracy_tracker.name: 
                self.distil_accuracy_tracker.result()
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
    assert [metric.name for metric in wrapper.metrics][-8:] == [
        "classifier_loss", "clf_kl_loss", "clf_ctr_loss", "distil_loss",
        "total_accuracy", "classifier_accuracy", "clf_ctr_accuracy",
        "distil_token_accuracy",
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
    clf_values = wrapper.compute_clf_kl_ctr_distil_loss(
        classes, 
        both[3][0], both[5][0], both[4][0], 
        classes_pred_u=both[3][1], 
        clf_z_vals_u=both[5][1], 
        clf_regs_list_u=both[4][1], 
        clf_loss_mask=mask, 
        clf_train_type="cond", 
    )
    assert len(clf_values) == 8 and float(clf_values[1]) >= 0.0
    empty_values = wrapper.compute_clf_kl_ctr_distil_loss(
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
    separate_noise_wrapper = make_wrapper(
        mask_by_nulls=True,
        show_separate_noise_losses=True,
    )
    separate_noise_results = separate_noise_wrapper.train_step(
        (images, classes)
    )
    assert {
        "total_noise_loss", "cond_noise_loss", "uncond_noise_loss",
        "classifier_loss",
    } <= set(separate_noise_results)
    assert "noise_loss" not in separate_noise_results
    dataset = tf.data.Dataset.from_tensor_slices((images, classes)).batch(2)
    history = wrapper.fit(dataset, epochs=1, verbose=0)
    assert len(history.history["classifier_loss"]) == 1
    evaluation = wrapper.evaluate(
        dataset, network_name="raw", verbose=0, return_dict=True
    )
    assert "classifier_accuracy" in evaluation

    try:
        DiffusionClassifier(
            network=make_network(clf_distil_token_type="new_weight"),
            distil_loss_coef=1.0,
            mask_by_nulls=False,
            p_uncond=0.0,
            use_ema=False,
            test_network_name="raw",
            test_steps=2,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError(
            "A configured token objective requires a teacher by default."
        )

    positional_teacher = make_network()
    positional_compatibility = DiffusionClassifier(
        positional_teacher,
        "soft",
        False,
        network=make_network(),
        use_ema=False,
        test_network_name="raw",
        test_steps=2,
    )
    assert positional_compatibility.teacher_network is positional_teacher
    assert positional_compatibility.distil_type == "soft"
    assert positional_compatibility.mask_by_nulls is False
    assert positional_compatibility.defer_teacher is False

    continual_student = make_wrapper(
        network=make_network(
            num_classes=None,
            clf_distil_token_type="new_weight",
        ),
        defer_teacher=True,
        distil_loss_coef=1.0,
        mask_by_nulls=False,
        p_uncond=0.0,
    )
    assert continual_student.teacher_network is None
    assert continual_student.use_distil_loss is False
    assert continual_student.map_preprocess is False
    assert continual_student.accuracy_tracker.name == "cls_token_accuracy"
    assert continual_student.get_config()["defer_teacher"] is True
    assert "teacher_network" not in continual_student.get_config()

    external_teacher = tf.keras.Sequential()
    continual_student.set_teacher_network(external_teacher)
    tf.debugging.assert_equal(
        continual_student._mask_unknown_teacher_labels(classes),
        classes,
    )
    continual_student.set_teacher_network(None)

    continual_student._check_new_labels(y=classes, verbose=False)
    continual_student._add_depths({
        "classifier": "vision_transformer_block"
    })
    continual_student.set_current_resolution(8)
    teacher = continual_student.snapshot_teacher_network("ema")
    assert teacher is not continual_student.ema_network
    assert teacher.num_classes == 2
    assert teacher.clf_depth == continual_student.network.clf_depth == 2
    assert teacher._current_resolution == 8
    assert teacher.trainable is False and not teacher.trainable_weights
    assert {
        id(weight) for weight in teacher.weights
    }.isdisjoint({
        id(weight) for weight in continual_student.ema_network.weights
    })

    snapshot_images = tf.image.resize(images, (8, 8))
    snapshot_times = tf.zeros((2,), dtype=tf.int32)
    snapshot_labels = tf.constant([1, 2], dtype=tf.uint8)
    source_outputs = continual_student.ema_network(
        (snapshot_images, snapshot_times, snapshot_labels),
        training=False,
    )
    teacher_outputs = teacher(
        (snapshot_images, snapshot_times, snapshot_labels),
        training=False,
    )
    for output_name in ("noises", "classes", "distil_classes"):
        tf.debugging.assert_near(
            source_outputs[output_name],
            teacher_outputs[output_name],
        )

    continual_student._check_new_labels(
        y=tf.constant([2], dtype=tf.uint8),
        verbose=False,
    )
    student_weight_ids = {
        id(weight) for weight in continual_student.weights
    }
    continual_student.train_function = object()
    continual_student.test_function = object()
    continual_student.set_teacher_network(teacher)
    assert {
        id(weight) for weight in continual_student.weights
    } == student_weight_ids
    assert student_weight_ids.isdisjoint({
        id(weight) for weight in teacher.weights
    })
    tf.debugging.assert_equal(
        continual_student._mask_unknown_teacher_labels(
            tf.constant([1, 3], dtype=tf.uint8)
        ),
        tf.constant([1, 0], dtype=tf.uint8),
    )
    assert continual_student.use_distil_loss
    assert continual_student.use_teacher
    assert continual_student.accuracy_tracker.name == "cls_token_accuracy"
    assert continual_student.map_preprocess
    assert continual_student.train_function is None
    assert continual_student.test_function is None

    new_task_images = images[:1]
    new_task_classes = tf.constant([2], dtype=tf.uint8)
    prepared_distillation = continual_student.prep_inputs_map(
        new_task_images,
        new_task_classes,
    )
    assert len(prepared_distillation) == 8
    assert prepared_distillation[-1].shape == (1, 2)
    distil_step = continual_student.train_step(prepared_distillation)
    assert {"distil_loss", "distil_token_accuracy"} <= set(distil_step)

    new_task_dataset = tf.data.Dataset.from_tensor_slices((
        new_task_images,
        new_task_classes,
    )).batch(1)
    distil_progressive = continual_student.fit_progressively(
        stage_tasks=[{"resolution": 8}],
        x=new_task_dataset,
        validation_data=new_task_dataset,
        stages_verbose=False,
        stage_epochs=1,
        final_epochs=0,
        verbose=0,
    )
    assert len(distil_progressive.progressive_stages) == 1
    continual_student.set_teacher_network(None)
    assert continual_student.teacher_network is None
    assert continual_student.use_distil_loss is False
    assert continual_student.map_preprocess is False

    from diffusion.models.transformer.di_t_encoder_decoder_classifier import (
        DiTEncoderDecoderClassifier,
    )
    composite_network = DiTEncoderDecoderClassifier(
        encoder_kwargs={
            "num_classes": None,
            "use_cfg": True,
            "timesteps": 4,
            "image_size": 4,
            "channels": 1,
            "patch_size": 2,
            "dim": 4,
            "depth": 0,
            "mha_num_heads": 1,
            "vit_block_mlp_ratio": 1.0,
            "feature_aggregation_ids_dict": {1: (-1,)},
            "clf_connection_ids_dict": {-1: (-1,)},
            "clf_distil_token_type": "new_weight",
        },
        decoder_kwargs={
            "depth": 0,
            "shift_inputs": False,
            "use_unpatchify": True,
        },
    )
    composite_student = make_wrapper(
        network=composite_network,
        defer_teacher=True,
        distil_loss_coef=1.0,
        mask_by_nulls=False,
        p_uncond=0.0,
        use_ema=False,
        test_network_name="raw",
    )
    composite_student._check_new_labels(y=classes, verbose=False)
    # Reproduce a valid clone whose nested decoder has a different runtime name.
    composite_student.network.decoder._name = "past_runtime_decoder"
    composite_teacher = composite_student.snapshot_teacher_network("raw")
    assert len(composite_teacher.weights) == len(
        composite_student.network.weights
    )
    composite_inputs = (
        images,
        tf.zeros((2,), dtype=tf.int32),
        tf.constant([1, 2], dtype=tf.uint8),
    )
    composite_source_outputs = composite_student.network(
        composite_inputs,
        training=False,
    )
    composite_teacher_outputs = composite_teacher(
        composite_inputs,
        training=False,
    )
    for output_name in ("noises", "classes", "distil_classes"):
        tf.debugging.assert_near(
            composite_source_outputs[output_name],
            composite_teacher_outputs[output_name],
        )

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
    unconditional_results = unconditional.train_step((images, classes))
    assert "classifier_loss" in unconditional_results

    ensemble = make_wrapper(
        use_ensemble_loss_instead=True, 
        mask_by_nulls=False, 
    )
    assert ensemble.ensemble_loss_fn is not None
    assert ensemble.ensemble_loss_fn.network is ensemble.network
    ensemble_predictions = ensemble.compute_clf_kl_ctr_distil_loss(
        classes, 
        both[3][0], both[5][0], both[4][0], 
        x0=images, 
        training=False, 
    )[5]
    assert ensemble_predictions.shape == (2, 2)
    tf.debugging.assert_near(
        tf.reduce_sum(ensemble_predictions, axis=-1), 
        tf.ones((2,)), atol=1e-5
    )
    assert "classifier_loss" in ensemble.test_step((images, classes))

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
    auxiliary_losses = auxiliary.compute_clf_kl_ctr_distil_loss(
        classes, 
        auxiliary_outputs["classes"], 
        auxiliary_outputs["clf_z_vals"], 
        auxiliary_outputs["clf_regs_list"], 
    )
    assert float(auxiliary_losses[2]) >= 0.0
    assert float(auxiliary_losses[3]) >= 0.0
    auxiliary_metrics = auxiliary.get_clf_results_dict(
        auxiliary_losses[1], classes, 
        auxiliary_losses[5],
        total_loss=auxiliary_losses[0], 
        clf_kl_loss=auxiliary_losses[2], 
        clf_ctr_loss=auxiliary_losses[3], 
        clf_ctr_preds=auxiliary_losses[6],
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
    assert policy_config["name"] == "policy_classifier_wrapper"
    assert policy_config["trainable"] is False
    assert policy_config["dtype"] == "float64"
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
        {"mask_t_percentage": True, "mask_by_nulls": False},
        {"clf_loss_coef": -1., "mask_by_nulls": False},
        {"distil_loss_coef": float("nan"), "mask_by_nulls": False},
        {"clf_acc_coef": -1., "mask_by_nulls": False},
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
